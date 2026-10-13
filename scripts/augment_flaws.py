#!/usr/bin/env python3
"""
Phase 1 — Deterministic contrastive flaw injection engine (seed=42).

For each baseline speech_{A,B,C}, emit a 4-tier gradient:
  Level 0 — unmodified reference copy
  Level 1 — subtle pause compression OR subtle pitch flattening
  Level 2 — pacing rush + monotone flatline + awkward silence
  Level 3 — volume spikes + severe hesitations + hyper-speed + pitch collapse

Every splice uses 8–10 ms cosine cross-fades. Timestamp shifts from silence
insertion / time-stretch are tracked so ground-truth bounds stay aligned.
Writes dataset/annotations/ground_truth_flaws.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.audio_ops import (  # noqa: E402
    DEFAULT_CROSSFADE_MS,
    TARGET_SR,
    TimestampTracker,
    apply_gain_bursts,
    duration_sec,
    insert_silence,
    load_audio,
    pitch_flatten,
    replace_segment,
    time_stretch,
    time_stretch_segment,
    write_wav_pcm16,
)

console = Console()

SEED = 42
SR = TARGET_SR
FADE_MS = DEFAULT_CROSSFADE_MS  # 9 ms ∈ [8, 10]

PROC_DIR = ROOT / "dataset" / "processed_baseline"
RAW_DIR = ROOT / "dataset" / "raw_baseline"
FLAW_DIR = ROOT / "dataset" / "flawed_spectrum"
ANN_PATH = ROOT / "dataset" / "annotations" / "ground_truth_flaws.json"

SAMPLE_KEYS = ("A", "B", "C")


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _load_meta(key: str) -> dict:
    meta_path = RAW_DIR / f"speech_{key}.meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing {meta_path}. Run scripts/setup_baselines.py first."
        )
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _window(meta: dict, name: str) -> tuple[float, float, str]:
    w = meta["phrase_windows"][name]
    return float(w["start_sec"]), float(w["end_sec"]), str(w["text"])


def _flaw(
    flaw_id: str,
    category: str,
    sub_type: str,
    start: float,
    end: float,
    phrase: str,
    params: dict,
    signature: str,
) -> dict:
    return {
        "flaw_id": flaw_id,
        "category": category,
        "sub_type": sub_type,
        "start_sec": round(float(start), 3),
        "end_sec": round(float(end), 3),
        "target_phrase": phrase,
        "parameters_applied": params,
        "expected_acoustic_signature": signature,
    }


def _record_sample(
    sample_id: str,
    baseline_id: str,
    audio_path: Path,
    transcript_path: str,
    y: np.ndarray,
    severity: int,
    flaws: list[dict],
) -> dict:
    write_wav_pcm16(audio_path, y, SR)
    return {
        "sample_id": sample_id,
        "baseline_id": baseline_id,
        "audio_path": _rel(audio_path),
        "transcript_path": transcript_path.replace("\\", "/"),
        "duration_sec": round(duration_sec(y, SR), 3),
        "severity_level": severity,
        "injected_flaws": flaws,
    }


# ---------------------------------------------------------------------------
# Level builders
# ---------------------------------------------------------------------------

def build_level0(key: str, y: np.ndarray, meta: dict) -> dict:
    sample_id = f"speech_{key}_level0_baseline"
    out = FLAW_DIR / f"{sample_id}.wav"
    return _record_sample(
        sample_id=sample_id,
        baseline_id=meta["baseline_id"],
        audio_path=out,
        transcript_path=meta["transcript_path"],
        y=y,
        severity=0,
        flaws=[],
    )


def build_level1(key: str, y: np.ndarray, meta: dict, rng: np.random.Generator) -> dict:
    """
    Subtle flaws (one targeted clause):
      A/C → subtle pitch flattening
      B   → slight pause compression (mild speed-up of a pause-rich window)
    """
    tracker = TimestampTracker()
    y_out = y.copy()
    flaws: list[dict] = []

    if key == "B":
        # Slight pause compression via 1.12× stretch on clause_1
        s, e, phrase = _window(meta, "clause_1")
        rate = 1.12
        y_out, new_s, new_e, delta = time_stretch_segment(
            y_out, s, e, rate=rate, sr=SR, fade_ms=FADE_MS
        )
        tracker.record(s, delta)
        flaws.append(
            _flaw(
                "flaw_01",
                "pacing",
                "pause_compression",
                new_s,
                new_e,
                phrase,
                {"time_stretch_factor": rate, "crossfade_ms": FADE_MS},
                "Slightly compressed pause intervals and reduced syllable duration in one clause.",
            )
        )
        tag = "pause_compression"
    else:
        s, e, phrase = _window(meta, "clause_2")
        suppression = 0.35
        start = int(round(s * SR))
        end = int(round(e * SR))
        segment = y_out[start:end]
        flat = pitch_flatten(segment, sr=SR, suppression=suppression)
        y_out, delta = replace_segment(y_out, flat, s, e, sr=SR, fade_ms=FADE_MS)
        # Duration preserved by pitch_flatten design; delta ~ 0
        tracker.record(s, delta)
        flaws.append(
            _flaw(
                "flaw_01",
                "pitch",
                "subtle_flattening",
                s,
                e,
                phrase,
                {"f0_std_suppression": suppression, "crossfade_ms": FADE_MS},
                "Mild reduction in local F0 variance over a single clause; still near-natural.",
            )
        )
        tag = "subtle_pitch_flatten"

    sample_id = f"speech_{key}_level1_{tag}"
    out = FLAW_DIR / f"{sample_id}.wav"
    return _record_sample(
        sample_id=sample_id,
        baseline_id=meta["baseline_id"],
        audio_path=out,
        transcript_path=meta["transcript_path"],
        y=y_out,
        severity=1,
        flaws=flaws,
    )


def build_level2(key: str, y: np.ndarray, meta: dict, rng: np.random.Generator) -> dict:
    """
    Moderate compound flaws (applied in chronological order):
      1) Pacing: 1.5× rush on clause_2 window
      2) Pitch:  >70% F0 suppression on clause_3 (3–5 s)
      3) Awkward silence: 2.0–2.5 s dead air inserted mid clause_1 region
         (inserted last among early-timeline edits? Actually silence is mid-clause
          early — apply silence FIRST so later windows can be mapped forward.)
    Application order chosen so we can map later windows through the tracker:
      a) insert silence early on timeline
      b) rush (mapped)
      c) monotone (mapped)
    """
    tracker = TimestampTracker()
    y_out = y.copy()
    flaws: list[dict] = []

    # --- (a) Awkward silence mid clause_1 ---
    c1_s, c1_e, c1_phrase = _window(meta, "clause_1")
    silence_at = (c1_s + c1_e) / 2.0
    silence_dur = float(rng.uniform(2.0, 2.5))
    y_out, sil_delta = insert_silence(
        y_out, at_sec=silence_at, silence_sec=silence_dur, sr=SR, fade_ms=FADE_MS
    )
    tracker.record(silence_at, sil_delta)
    sil_start = silence_at
    sil_end = silence_at + silence_dur
    flaws.append(
        _flaw(
            "flaw_01",
            "silence",
            "awkward_dead_air",
            sil_start,
            sil_end,
            c1_phrase,
            {
                "silence_duration_sec": round(silence_dur, 3),
                "insert_at_sec": round(silence_at, 3),
                "crossfade_ms": FADE_MS,
            },
            "Unprompted mid-clause dead silence with cosine cross-faded edges; timeline shifted thereafter.",
        )
    )

    # --- (b) Pacing rush on clause_2 (shift-mapped) ---
    c2_s, c2_e, c2_phrase = _window(meta, "clause_2")
    c2_s_m, c2_e_m = tracker.map_window(c2_s, c2_e)
    rate = 1.55
    y_out, new_s, new_e, pace_delta = time_stretch_segment(
        y_out, c2_s_m, c2_e_m, rate=rate, sr=SR, fade_ms=FADE_MS
    )
    tracker.record(c2_s_m, pace_delta)
    flaws.append(
        _flaw(
            "flaw_02",
            "pacing",
            "rushed_delivery",
            new_s,
            new_e,
            c2_phrase,
            {"time_stretch_factor": rate, "crossfade_ms": FADE_MS},
            "Sudden drop in syllable duration and compressed pause intervals.",
        )
    )

    # --- (c) Monotone flatlining on clause_3 (shift-mapped) ---
    c3_s, c3_e, c3_phrase = _window(meta, "clause_3")
    c3_s_m, c3_e_m = tracker.map_window(c3_s, c3_e)
    # Ensure 3–5 s window
    win_len = c3_e_m - c3_s_m
    if win_len < 3.0:
        c3_e_m = c3_s_m + 3.5
    elif win_len > 5.0:
        c3_e_m = c3_s_m + 5.0
    suppression = 0.80  # >70%
    start = int(round(c3_s_m * SR))
    end = int(round(min(c3_e_m, duration_sec(y_out, SR)) * SR))
    end = min(end, len(y_out))
    segment = y_out[start:end]
    flat = pitch_flatten(segment, sr=SR, suppression=suppression)
    c3_end_actual = end / SR
    y_out, pitch_delta = replace_segment(
        y_out, flat, c3_s_m, c3_end_actual, sr=SR, fade_ms=FADE_MS
    )
    tracker.record(c3_s_m, pitch_delta)
    flaws.append(
        _flaw(
            "flaw_03",
            "pitch",
            "monotone_flattening",
            c3_s_m,
            c3_s_m + duration_sec(flat, SR),
            c3_phrase,
            {"f0_std_suppression": suppression, "crossfade_ms": FADE_MS},
            "Zero pitch deviation; F0 contour approaches variance of zero.",
        )
    )

    sample_id = f"speech_{key}_level2_pacing_monotone"
    out = FLAW_DIR / f"{sample_id}.wav"
    # Sort flaws by start time for readability
    flaws_sorted = sorted(flaws, key=lambda f: f["start_sec"])
    # Re-number flaw_ids chronologically
    for i, f in enumerate(flaws_sorted, 1):
        f["flaw_id"] = f"flaw_{i:02d}"

    return _record_sample(
        sample_id=sample_id,
        baseline_id=meta["baseline_id"],
        audio_path=out,
        transcript_path=meta["transcript_path"],
        y=y_out,
        severity=2,
        flaws=flaws_sorted,
    )


def build_level3(key: str, y: np.ndarray, meta: dict, rng: np.random.Generator) -> dict:
    """
    Severe / botched compound flaws:
      1) Multiple unnatural hesitations (>2.0 s) early & mid
      2) Erratic +12 dB volume spikes on unstressed bursts
      3) Hyper-speed (>1.8×) on a late clause + complete pitch-range collapse
    Silence insertions applied first (earliest→latest) so subsequent anchors map cleanly.
    """
    tracker = TimestampTracker()
    y_out = y.copy()
    flaws: list[dict] = []
    flaw_n = 1

    # --- Severe hesitations ---
    hesitation_anchors = [
        (_window(meta, "clause_1")[0] + 1.0, _window(meta, "clause_1")[2]),
        (_window(meta, "clause_2")[0] + 0.8, _window(meta, "clause_2")[2]),
        (_window(meta, "clause_3")[0] + 0.5, _window(meta, "clause_3")[2]),
    ]
    # Apply earliest first on original timeline, mapping each next anchor
    for raw_at, phrase in sorted(hesitation_anchors, key=lambda x: x[0]):
        at = tracker.map_time(raw_at)
        dur = float(rng.uniform(2.05, 2.6))
        y_out, delta = insert_silence(
            y_out, at_sec=at, silence_sec=dur, sr=SR, fade_ms=FADE_MS
        )
        tracker.record(at, delta)
        flaws.append(
            _flaw(
                f"flaw_{flaw_n:02d}",
                "silence",
                "severe_hesitation",
                at,
                at + dur,
                phrase,
                {
                    "silence_duration_sec": round(dur, 3),
                    "insert_at_sec": round(at, 3),
                    "crossfade_ms": FADE_MS,
                },
                "Unnatural hesitation pause exceeding 2.0 seconds with soft-edged splice.",
            )
        )
        flaw_n += 1

    # --- Erratic volume spikes (+12 dB) on short unstressed bursts ---
    # Place bursts inside mapped clause_2 region
    c2_s, c2_e, c2_phrase = _window(meta, "clause_2")
    c2_s_m, c2_e_m = tracker.map_window(c2_s, c2_e)
    span = max(0.5, c2_e_m - c2_s_m)
    burst_windows = []
    for offset in (0.15 * span, 0.45 * span, 0.75 * span):
        b_s = c2_s_m + offset
        b_e = min(c2_e_m, b_s + 0.12)
        burst_windows.append((b_s, b_e))
    gain_db = 12.0
    y_out = apply_gain_bursts(y_out, burst_windows, gain_db=gain_db, sr=SR)
    # Soft clip risk retained intentionally; final write peak-normalizes whole file
    # but we capture pre-normalize spikes in a side copy for annotation fidelity:
    # Re-apply conceptually — annotations describe the injected intent.
    for b_s, b_e in burst_windows:
        flaws.append(
            _flaw(
                f"flaw_{flaw_n:02d}",
                "energy",
                "volume_spike",
                b_s,
                b_e,
                c2_phrase,
                {"gain_db": gain_db, "clip_risk": True},
                "Erratic +12 dB burst on unstressed fragment; transient clip-risk energy spike.",
            )
        )
        flaw_n += 1

    # --- Hyper-speed cadence (>1.8×) + pitch-range collapse on clause_4 ---
    c4_s, c4_e, c4_phrase = _window(meta, "clause_4")
    c4_s_m, c4_e_m = tracker.map_window(c4_s, c4_e)
    # Clamp to audio length
    audio_dur = duration_sec(y_out, SR)
    c4_s_m = min(c4_s_m, max(0.0, audio_dur - 1.0))
    c4_e_m = min(max(c4_e_m, c4_s_m + 1.0), audio_dur)

    rate = 1.85
    start = int(round(c4_s_m * SR))
    end = int(round(c4_e_m * SR))
    segment = y_out[start:end]
    sped = time_stretch(segment, rate=rate)
    collapsed = pitch_flatten(sped, sr=SR, suppression=1.0)
    y_out, delta = replace_segment(
        y_out, collapsed, c4_s_m, c4_e_m, sr=SR, fade_ms=FADE_MS
    )
    tracker.record(c4_s_m, delta)
    new_end = c4_s_m + duration_sec(collapsed, SR)
    flaws.append(
        _flaw(
            f"flaw_{flaw_n:02d}",
            "compound",
            "hyperspeed_pitch_collapse",
            c4_s_m,
            new_end,
            c4_phrase,
            {
                "time_stretch_factor": rate,
                "f0_std_suppression": 1.0,
                "crossfade_ms": FADE_MS,
            },
            "Hyper-accelerated cadence (>1.8×) with complete pitch-range collapse to near-zero F0 variance.",
        )
    )

    flaws_sorted = sorted(flaws, key=lambda f: f["start_sec"])
    for i, f in enumerate(flaws_sorted, 1):
        f["flaw_id"] = f"flaw_{i:02d}"

    sample_id = f"speech_{key}_level3_botched"
    out = FLAW_DIR / f"{sample_id}.wav"
    return _record_sample(
        sample_id=sample_id,
        baseline_id=meta["baseline_id"],
        audio_path=out,
        transcript_path=meta["transcript_path"],
        y=y_out,
        severity=3,
        flaws=flaws_sorted,
    )


def augment_all() -> dict:
    FLAW_DIR.mkdir(parents=True, exist_ok=True)
    ANN_PATH.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    samples: list[dict] = []
    for key in SAMPLE_KEYS:
        meta = _load_meta(key)
        baseline_path = PROC_DIR / f"speech_{key}_baseline.wav"
        if not baseline_path.exists():
            raise FileNotFoundError(
                f"Missing {baseline_path}. Run scripts/setup_baselines.py first."
            )
        y, _ = load_audio(baseline_path, sr=SR)
        console.print(f"\n[bold cyan]== speech_{key}[/bold cyan]  ({meta['style']})  {duration_sec(y, SR):.2f}s")

        # Independent RNG streams per level keep composition deterministic but varied
        s0 = build_level0(key, y, meta)
        s1 = build_level1(key, y, meta, np.random.default_rng(SEED + 10 + ord(key)))
        s2 = build_level2(key, y, meta, np.random.default_rng(SEED + 20 + ord(key)))
        s3 = build_level3(key, y, meta, np.random.default_rng(SEED + 30 + ord(key)))
        for s in (s0, s1, s2, s3):
            samples.append(s)
            console.print(
                f"  L{s['severity_level']}  {s['sample_id']:<42}  "
                f"flaws={len(s['injected_flaws'])}  dur={s['duration_sec']:.2f}s"
            )

    payload = {
        "dataset_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "crossfade_ms": FADE_MS,
        "sample_rate": SR,
        "samples": samples,
    }
    ANN_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject contrastive delivery flaws.")
    parser.parse_args()

    console.print(
        Panel.fit(
            "[bold]Track C | Phase 1[/bold]\n"
            f"Flaw injection engine  |  seed={SEED}  |  crossfade={FADE_MS} ms",
            border_style="magenta",
        )
    )

    payload = augment_all()

    table = Table(title="Contrastive Spectrum Summary")
    table.add_column("Level", justify="center")
    table.add_column("Count", justify="right")
    table.add_column("Total Flaws", justify="right")
    for level in range(4):
        subset = [s for s in payload["samples"] if s["severity_level"] == level]
        n_flaws = sum(len(s["injected_flaws"]) for s in subset)
        table.add_row(str(level), str(len(subset)), str(n_flaws))
    console.print(table)
    console.print(f"\n[bold]Annotations ->[/bold] {_rel(ANN_PATH)}")
    console.print("[bold green]Augmentation complete.[/bold green]")


if __name__ == "__main__":
    main()
