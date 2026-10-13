#!/usr/bin/env python3
"""
Phase 1 — Baseline speech sourcing / synthesis.

Produces three diverse ~45–60 s pristine oratorical references:
  A) Persuasive / Oratory   — high dynamic contrast, intentional pausing
  B) Competitive Debate     — fast assertive cadence, structured emphasis
  C) Interpretive Reading   — emotionally rich, wide pitch variance

Primary path: Edge neural English voices (`edge-tts`), then 16 kHz mono PCM16.
Algorithmic formant synthesis remains available only with `--formant`.
Outputs:
  dataset/raw_baseline/speech_{A,B,C}.{wav,txt,meta.json}
  dataset/processed_baseline/speech_{A,B,C}_baseline.wav  (16 kHz mono PCM16, -1 dBFS)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from scipy import signal

# Allow `python scripts/setup_baselines.py` from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.audio_ops import (  # noqa: E402
    TARGET_SR,
    peak_normalize,
    write_wav_pcm16,
)

console = Console()

RAW_DIR = ROOT / "dataset" / "raw_baseline"
PROC_DIR = ROOT / "dataset" / "processed_baseline"
SR = TARGET_SR
RNG_SEED = 42


# ---------------------------------------------------------------------------
# Transcripts & phrase windows (used later by flaw injection)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BaselineSpec:
    sample_key: str  # A | B | C
    style: str
    transcript: str
    # Approximate phrase windows on the synthetic timeline (sec)
    # Used by augment_flaws.py as injection anchors
    phrase_windows: dict[str, tuple[float, float, str]]


SPEECH_A = BaselineSpec(
    sample_key="A",
    style="persuasive_oratory",
    transcript=(
        "Today I want to tell you three stories from my life. "
        "The first story is about connecting the dots. "
        "You cannot connect the dots looking forward. "
        "You can only connect them looking backward. "
        "So you have to trust that the dots will somehow connect in your future. "
        "You have to trust in something: your gut, destiny, life, karma, whatever. "
        "Because believing that the dots will connect down the road "
        "will give you the confidence to follow your heart and intuition. "
        "Somehow they already know what you truly want to become. "
        "Everything else is secondary. Stay hungry. Stay foolish."
    ),
    phrase_windows={
        "clause_1": (8.0, 14.0, "You cannot connect the dots looking forward"),
        "clause_2": (22.0, 28.0, "to follow your heart and intuition"),
        "clause_3": (30.0, 36.0, "they somehow already know what you truly want to become"),
        "clause_4": (40.0, 46.0, "Stay hungry. Stay foolish"),
    },
)

SPEECH_B = BaselineSpec(
    sample_key="B",
    style="parliamentary_debate",
    transcript=(
        "Mister Speaker, this house believes that open debate strengthens democracy. "
        "First, transparency forces leaders to justify every policy choice in public. "
        "Second, structured opposition exposes weak arguments before they become law. "
        "Third, citizens who hear both sides make sharper, more informed decisions. "
        "Our opponents claim speed matters more than scrutiny. "
        "We reject that false tradeoff. Rushed legislation is fragile legislation. "
        "Vote for deliberation. Vote for accountability. Vote for this motion."
    ),
    phrase_windows={
        "clause_1": (6.0, 12.0, "transparency forces leaders to justify every policy choice"),
        "clause_2": (16.0, 22.0, "structured opposition exposes weak arguments"),
        "clause_3": (28.0, 34.0, "Rushed legislation is fragile legislation"),
        "clause_4": (38.0, 44.0, "Vote for deliberation. Vote for accountability"),
    },
)

SPEECH_C = BaselineSpec(
    sample_key="C",
    style="interpretive_expressive",
    transcript=(
        "The night was wide and quiet, and the river carried old secrets downstream. "
        "She stood at the edge and listened to the wind braid through the reeds. "
        "Memory rose like mist: a childhood kitchen, warm bread, a lullaby half-forgotten. "
        "Grief and gratitude arrived together, neither asking permission. "
        "She spoke softly to the dark, naming every fear she had outgrown. "
        "And when the first light touched the water, her voice found its full range again — "
        "low as thunder, bright as glass, unbroken."
    ),
    phrase_windows={
        "clause_1": (7.0, 13.0, "the river carried old secrets downstream"),
        "clause_2": (18.0, 24.0, "Memory rose like mist"),
        "clause_3": (30.0, 36.0, "Grief and gratitude arrived together"),
        "clause_4": (42.0, 50.0, "low as thunder, bright as glass, unbroken"),
    },
)

BASELINES = [SPEECH_A, SPEECH_B, SPEECH_C]

# Distinct neural English speakers. The first voice is preferred; the second is a fallback.
NEURAL_VOICES = {
    "A": ("en-US-GuyNeural", "en-US-ChristopherNeural"),
    "B": ("en-GB-RyanNeural", "en-US-EricNeural"),
    "C": ("en-US-JennyNeural", "en-US-AriaNeural"),
}


# ---------------------------------------------------------------------------
# Algorithmic formant speech synthesis
# ---------------------------------------------------------------------------

def _glottal_pulse_train(f0: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a simple Liljencrants-Fant-ish impulse train from an F0 contour."""
    n = len(f0)
    y = np.zeros(n, dtype=np.float32)
    phase = 0.0
    for i in range(n):
        f = max(40.0, float(f0[i]))
        phase += 2.0 * np.pi * f / sr
        if phase >= 2.0 * np.pi:
            phase -= 2.0 * np.pi
            # Soft glottal pulse + tiny noise jitter
            y[i] = 1.0 + 0.05 * float(rng.standard_normal())
        else:
            y[i] = 0.02 * float(rng.standard_normal())
    # Shape pulses with a short decaying envelope via IIR
    b, a = signal.butter(1, 120 / (sr / 2), btype="low")
    shaped = signal.lfilter(b, a, y).astype(np.float32)
    return shaped


def _formant_filter(y: np.ndarray, sr: int, formants: list[tuple[float, float]]) -> np.ndarray:
    """Cascade resonant bandpass filters approximating vowel formants (freq, Q)."""
    out = np.zeros_like(y, dtype=np.float32)
    for freq, q in formants:
        freq = float(np.clip(freq, 80.0, sr / 2 - 200))
        bw = freq / max(q, 0.5)
        low = max(40.0, freq - bw / 2)
        high = min(sr / 2 - 100, freq + bw / 2)
        if high <= low:
            continue
        sos = signal.butter(2, [low, high], btype="band", fs=sr, output="sos")
        out += signal.sosfilt(sos, y).astype(np.float32)
    return out


def _prosody_contour(
    n: int,
    sr: int,
    style: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (f0_hz, amplitude_env, pause_mask) for the given oratorical style.
    pause_mask: 1 = speech, 0 = intentional pause.
    """
    t = np.arange(n) / sr

    if style == "persuasive_oratory":
        base_f0 = 145.0
        # Wide dynamics, long intentional pauses every ~7–9 s
        vibrato = 4.0 * np.sin(2 * np.pi * 4.5 * t)
        phrase = 18.0 * np.sin(2 * np.pi * 0.12 * t)
        declination = -12.0 * (t / t[-1])
        f0 = base_f0 + vibrato + phrase + declination
        amp = 0.55 + 0.35 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.15 * t))
        amp *= 0.7 + 0.3 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.04 * t + 0.5))
        pause_mask = np.ones(n, dtype=np.float32)
        for p_start in [6.5, 14.5, 21.0, 29.5, 37.0, 45.0]:
            s = int(p_start * sr)
            e = int((p_start + 0.55) * sr)
            if s < n:
                pause_mask[s:min(e, n)] = 0.0

    elif style == "parliamentary_debate":
        base_f0 = 165.0
        # Fast cadence, short micro-pauses, assertive F0 resets
        vibrato = 2.0 * np.sin(2 * np.pi * 5.5 * t)
        emphasis = 22.0 * signal.square(2 * np.pi * 0.35 * t, duty=0.25)
        emphasis = signal.savgol_filter(emphasis, 51, 2) if n > 51 else emphasis
        f0 = base_f0 + vibrato + 0.45 * emphasis
        amp = 0.65 + 0.25 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.55 * t))
        pause_mask = np.ones(n, dtype=np.float32)
        for p_start in np.arange(3.5, t[-1], 3.8):
            s = int(p_start * sr)
            e = int((p_start + 0.18) * sr)
            if s < n:
                pause_mask[s:min(e, n)] = 0.0

    else:  # interpretive_expressive
        base_f0 = 175.0
        # Wide pitch variance, emotional arcs
        vibrato = 6.5 * np.sin(2 * np.pi * 5.0 * t)
        arc = 35.0 * np.sin(2 * np.pi * 0.045 * t) + 20.0 * np.sin(2 * np.pi * 0.09 * t + 1.2)
        f0 = base_f0 + vibrato + arc
        amp = 0.45 + 0.45 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.08 * t))
        # Soft breath pauses
        pause_mask = np.ones(n, dtype=np.float32)
        for p_start in [9.0, 17.5, 26.0, 35.5, 47.0]:
            s = int(p_start * sr)
            e = int((p_start + 0.4) * sr)
            if s < n:
                pause_mask[s:min(e, n)] = 0.0

    # Gentle random walk on F0 for naturalness
    jitter = np.cumsum(rng.normal(0, 0.15, size=n)).astype(np.float32)
    jitter = jitter - np.linspace(jitter[0], jitter[-1], n)
    f0 = (f0 + jitter).astype(np.float32)
    f0 = np.clip(f0, 80.0, 320.0)
    amp = np.clip(amp, 0.05, 1.0).astype(np.float32)
    return f0, amp, pause_mask


def synthesize_speech(
    style: str,
    duration_sec: float,
    sr: int = SR,
    seed: int = RNG_SEED,
) -> np.ndarray:
    """
    Offline formant synthesizer producing speech-like oratorical audio.
    Not intelligible ASR content, but prosodically matches the target style.
    """
    rng = np.random.default_rng(seed + hash(style) % 10_000)
    n = int(duration_sec * sr)
    f0, amp, pause_mask = _prosody_contour(n, sr, style, rng)
    source = _glottal_pulse_train(f0, sr, rng)

    # Time-varying vowel formant targets (F1, F2, F3)
    t = np.arange(n) / sr
    if style == "persuasive_oratory":
        f1 = 550 + 120 * np.sin(2 * np.pi * 0.7 * t)
        f2 = 1400 + 250 * np.sin(2 * np.pi * 0.55 * t + 0.4)
        f3 = 2400 + 180 * np.sin(2 * np.pi * 0.4 * t)
    elif style == "parliamentary_debate":
        f1 = 500 + 90 * np.sin(2 * np.pi * 1.4 * t)
        f2 = 1550 + 300 * np.sin(2 * np.pi * 1.1 * t + 0.8)
        f3 = 2500 + 150 * np.sin(2 * np.pi * 0.9 * t)
    else:
        f1 = 600 + 180 * np.sin(2 * np.pi * 0.5 * t)
        f2 = 1300 + 400 * np.sin(2 * np.pi * 0.35 * t + 1.0)
        f3 = 2600 + 220 * np.sin(2 * np.pi * 0.28 * t + 0.3)

    # Piecewise formant filtering over short hops for time-varying vowels
    hop = sr // 20  # 50 ms
    out = np.zeros(n, dtype=np.float32)
    for i in range(0, n, hop):
        j = min(n, i + hop)
        chunk = source[i:j]
        formants = [
            (float(np.mean(f1[i:j])), 8.0),
            (float(np.mean(f2[i:j])), 10.0),
            (float(np.mean(f3[i:j])), 12.0),
        ]
        filtered = _formant_filter(chunk, sr, formants)
        # Overlap-add with Hann
        w = np.hanning(len(filtered)).astype(np.float32)
        if len(filtered) < 4:
            w = np.ones(len(filtered), dtype=np.float32)
        out[i:j] += filtered * w

    # Normalize partial OLA gain
    out = out / (np.max(np.abs(out)) + 1e-9)

    # Amplitude envelope + pauses
    env = amp * pause_mask
    # Smooth pause edges (~15 ms) to avoid clicks in the baseline itself
    ker = max(3, int(0.015 * sr))
    if ker % 2 == 0:
        ker += 1
    env = signal.savgol_filter(env, ker, 2).astype(np.float32)
    out = (out * env).astype(np.float32)

    # Mild high-shelf presence + breath noise during pauses
    breath = (0.01 * rng.standard_normal(n)).astype(np.float32)
    b_hp, a_hp = signal.butter(2, 2000 / (sr / 2), btype="high")
    breath = signal.lfilter(b_hp, a_hp, breath).astype(np.float32)
    out = out + breath * (0.35 + 0.65 * (1.0 - pause_mask))

    # Soft limiter then peak normalize to -1 dBFS
    out = np.tanh(1.4 * out).astype(np.float32)
    return peak_normalize(out)


# ---------------------------------------------------------------------------
# Optional Edge-TTS path
# ---------------------------------------------------------------------------

def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _align_phrase_windows(
    spec: BaselineSpec,
    words: list[dict],
    duration: float,
) -> dict[str, tuple[float, float, str]]:
    """Place each clause on the spoken words, then scale into the real duration."""
    spoken: list[tuple[str, float, float]] = []
    for word in words:
        start = float(word["start"])
        end = float(word["end"])
        for token in _tokens(str(word.get("text", ""))):
            spoken.append((token, start, end))

    aligned: dict[str, tuple[float, float, str]] = {}
    design = _target_duration(spec.style)
    scale = duration / design if design else 1.0
    for name, (start, end, phrase) in spec.phrase_windows.items():
        target = _tokens(phrase)
        found: tuple[float, float] | None = None
        if target and spoken:
            for i in range(len(spoken)):
                if spoken[i][0] != target[0]:
                    continue
                cursor = i
                ok = True
                for token in target:
                    while cursor < len(spoken) and spoken[cursor][0] != token:
                        cursor += 1
                        if cursor - i > len(target) + 4:
                            break
                    if cursor >= len(spoken) or spoken[cursor][0] != token:
                        ok = False
                        break
                    cursor += 1
                if ok:
                    found = (spoken[i][1], spoken[cursor - 1][2])
                    break
        if found is None:
            ns = max(0.0, start * scale)
            ne = min(duration, end * scale)
        else:
            ns, ne = found
        if ne - ns < 0.35:
            ne = min(duration, ns + 0.35)
        ns = min(max(0.0, ns), max(0.0, duration - 0.4))
        ne = min(duration, max(ne, ns + 0.35))
        aligned[name] = (round(ns, 3), round(ne, 3), phrase)
    return aligned


async def _edge_tts_synthesize(text: str, voice: str, mp3_path: Path) -> list[dict]:
    import edge_tts

    communicate = edge_tts.Communicate(text, voice)
    audio = bytearray()
    words: list[dict] = []
    async for chunk in communicate.stream():
        kind = chunk.get("type")
        if kind == "audio":
            audio.extend(chunk["data"])
        elif kind == "WordBoundary":
            # Edge reports offset/duration in 100-nanosecond ticks.
            start = float(chunk["offset"]) / 10_000_000.0
            dur = float(chunk["duration"]) / 10_000_000.0
            words.append({"text": chunk.get("text", ""), "start": start, "end": start + dur})
    if not audio:
        raise RuntimeError(f"{voice} returned no audio")
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    mp3_path.write_bytes(audio)
    return words


def _mp3_to_pcm16(mp3_path: Path) -> np.ndarray:
    """Decode Edge MP3 to 16 kHz mono float audio. Peak normalization happens on write."""
    import librosa

    y, _sr = librosa.load(str(mp3_path), sr=SR, mono=True)
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        raise RuntimeError(f"Decoded empty audio from {mp3_path.name}")
    return y


def synthesize_neural(spec: BaselineSpec, mp3_path: Path) -> tuple[np.ndarray, str, list[dict]]:
    voices = NEURAL_VOICES[spec.sample_key]
    errors: list[str] = []
    for voice in voices:
        try:
            words = asyncio.run(_edge_tts_synthesize(spec.transcript, voice, mp3_path))
            audio = _mp3_to_pcm16(mp3_path)
            return audio, voice, words
        except Exception as exc:  # noqa: BLE001 — try the alternate neural voice
            errors.append(f"{voice}: {exc}")
            console.print(f"[yellow]{voice} failed ({exc}).[/yellow]")
    raise RuntimeError("Neural synthesis failed: " + "; ".join(errors))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _target_duration(style: str) -> float:
    # Keep each sample in the 45–60 s band with style-appropriate pacing
    return {"persuasive_oratory": 52.0, "parliamentary_debate": 48.0, "interpretive_expressive": 55.0}[
        style
    ]


def setup_one(spec: BaselineSpec, use_neural: bool = True) -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    key = spec.sample_key
    raw_wav = RAW_DIR / f"speech_{key}.wav"
    raw_mp3 = RAW_DIR / f"speech_{key}.mp3"
    raw_txt = RAW_DIR / f"speech_{key}.txt"
    raw_meta = RAW_DIR / f"speech_{key}.meta.json"
    proc_wav = PROC_DIR / f"speech_{key}_baseline.wav"

    raw_txt.write_text(spec.transcript + "\n", encoding="utf-8")

    voice = ""
    words: list[dict] = []
    if use_neural:
        y, voice, words = synthesize_neural(spec, raw_mp3)
        source = f"edge_tts:{voice}"
    else:
        duration = _target_duration(spec.style)
        y = synthesize_speech(spec.style, duration, sr=SR, seed=RNG_SEED)
        source = "algorithmic_formant"

    y = peak_normalize(y)
    write_wav_pcm16(raw_wav, y, SR)
    write_wav_pcm16(proc_wav, y, SR)
    dur = float(len(y) / SR)
    windows = _align_phrase_windows(spec, words, dur)

    meta = {
        "baseline_id": f"speech_{key}_baseline",
        "sample_key": key,
        "style": spec.style,
        "source": source,
        "voice": voice,
        "duration_sec": round(dur, 3),
        "sample_rate": SR,
        "channels": 1,
        "bit_depth": 16,
        "peak_dbfs": -1.0,
        "transcript_path": str(raw_txt.relative_to(ROOT)).replace("\\", "/"),
        "raw_audio_path": str(raw_wav.relative_to(ROOT)).replace("\\", "/"),
        "processed_audio_path": str(proc_wav.relative_to(ROOT)).replace("\\", "/"),
        "phrase_windows": {
            name: {"start_sec": w[0], "end_sec": w[1], "text": w[2]}
            for name, w in windows.items()
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    raw_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize pristine baseline speeches.")
    parser.add_argument(
        "--formant",
        action="store_true",
        help="Use offline formant synthesis instead of neural English voices.",
    )
    args = parser.parse_args()

    console.print(
        Panel.fit(
            "[bold]Track C | Phase 1[/bold]\nBaseline speech synthesis (16 kHz mono PCM16, -1 dBFS)",
            border_style="cyan",
        )
    )

    results = []
    for spec in BASELINES:
        console.print(f"\n[bold green]> speech_{spec.sample_key}[/bold green] ({spec.style})")
        meta = setup_one(spec, use_neural=not args.formant)
        results.append(meta)
        console.print(
            f"  source={meta['source']}  duration={meta['duration_sec']:.2f}s  "
            f"-> {meta['processed_audio_path']}"
        )

    index_path = RAW_DIR / "baselines_index.json"
    index_path.write_text(
        json.dumps(
            {
                "dataset_version": "1.0.0",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "baselines": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    console.print(f"\n[bold]Wrote[/bold] {index_path.relative_to(ROOT)}")
    console.print("[bold green]Baseline setup complete.[/bold green]")


if __name__ == "__main__":
    main()
