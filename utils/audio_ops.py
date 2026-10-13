"""
Core DSP operations for contrastive speech dataset engineering.

Provides resampling, normalization, cross-faded splicing, time-stretch,
pitch flattening, energy scaling, and deterministic timestamp shift tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import librosa
import numpy as np
import soundfile as sf
from scipy import signal

TARGET_SR = 16_000
TARGET_PEAK_DBFS = -1.0
DEFAULT_CROSSFADE_MS = 9.0  # midpoint of required 8–10 ms range
TIMESTAMP_SHIFT_EPS = 1e-6


# ---------------------------------------------------------------------------
# I/O & formatting
# ---------------------------------------------------------------------------

def load_audio(path: str | Path, sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Load audio as float32 mono at the requested sample rate."""
    y, file_sr = sf.read(str(path), always_2d=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = to_mono(y)
    if file_sr != sr:
        y = resample(y, file_sr, sr)
    return y, sr


def to_mono(y: np.ndarray) -> np.ndarray:
    """Collapse multi-channel audio to mono by averaging channels."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        return y
    return np.mean(y, axis=-1).astype(np.float32)


def resample(y: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """Resample waveform to target_sr using librosa (high-quality Kaiser window)."""
    if orig_sr == target_sr:
        return np.asarray(y, dtype=np.float32)
    return librosa.resample(
        np.asarray(y, dtype=np.float32),
        orig_sr=orig_sr,
        target_sr=target_sr,
        res_type="kaiser_best",
    ).astype(np.float32)


def peak_normalize(y: np.ndarray, peak_dbfs: float = TARGET_PEAK_DBFS) -> np.ndarray:
    """Peak-normalize so max(|y|) corresponds to peak_dbfs (default -1.0 dBFS)."""
    y = np.asarray(y, dtype=np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak < 1e-12:
        return y.copy()
    target_peak = 10.0 ** (peak_dbfs / 20.0)
    return (y * (target_peak / peak)).astype(np.float32)


def write_wav_pcm16(path: str | Path, y: np.ndarray, sr: int = TARGET_SR) -> None:
    """Write mono 16-bit PCM WAV after peak normalization to -1.0 dBFS."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y_out = peak_normalize(to_mono(np.asarray(y, dtype=np.float32)))
    sf.write(str(path), y_out, sr, subtype="PCM_16")


def duration_sec(y: np.ndarray, sr: int = TARGET_SR) -> float:
    return float(len(y) / sr) if sr > 0 else 0.0


def sec_to_samples(t: float, sr: int = TARGET_SR) -> int:
    return int(round(t * sr))


def samples_to_sec(n: int, sr: int = TARGET_SR) -> float:
    return float(n / sr)


# ---------------------------------------------------------------------------
# Cross-fades
# ---------------------------------------------------------------------------

def _fade_length_samples(sr: int, fade_ms: float = DEFAULT_CROSSFADE_MS) -> int:
    n = max(1, int(round(sr * fade_ms / 1000.0)))
    return n


def linear_crossfade(
    left: np.ndarray,
    right: np.ndarray,
    sr: int = TARGET_SR,
    fade_ms: float = DEFAULT_CROSSFADE_MS,
) -> np.ndarray:
    """Join two segments with a linear equal-gain cross-fade."""
    return _crossfade(left, right, sr, fade_ms, curve="linear")


def cosine_crossfade(
    left: np.ndarray,
    right: np.ndarray,
    sr: int = TARGET_SR,
    fade_ms: float = DEFAULT_CROSSFADE_MS,
) -> np.ndarray:
    """Join two segments with a raised-cosine (equal-power) cross-fade."""
    return _crossfade(left, right, sr, fade_ms, curve="cosine")


def _crossfade(
    left: np.ndarray,
    right: np.ndarray,
    sr: int,
    fade_ms: float,
    curve: Literal["linear", "cosine"] = "cosine",
) -> np.ndarray:
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)

    n_fade = _fade_length_samples(sr, fade_ms)
    n_fade = min(n_fade, len(left), len(right))
    if n_fade <= 0:
        return np.concatenate([left, right]).astype(np.float32)

    if curve == "linear":
        fade_out = np.linspace(1.0, 0.0, n_fade, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, n_fade, dtype=np.float32)
    else:
        # Raised-cosine / Hann half-windows (equal-power)
        t = np.linspace(0.0, np.pi / 2.0, n_fade, dtype=np.float32)
        fade_out = np.cos(t).astype(np.float32)
        fade_in = np.sin(t).astype(np.float32)

    mixed = left[-n_fade:] * fade_out + right[:n_fade] * fade_in
    return np.concatenate([left[:-n_fade], mixed, right[n_fade:]]).astype(np.float32)


def splice_with_crossfade(
    y: np.ndarray,
    insert: np.ndarray,
    start_sec: float,
    end_sec: float | None = None,
    sr: int = TARGET_SR,
    fade_ms: float = DEFAULT_CROSSFADE_MS,
    curve: Literal["linear", "cosine"] = "cosine",
) -> tuple[np.ndarray, float]:
    """
    Replace [start_sec, end_sec) with `insert`, cross-fading at both edges.

    If end_sec is None, insert is spliced in (insertion, not replacement).
    Returns (new_audio, net_duration_delta_sec).
    """
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    insert = np.asarray(insert, dtype=np.float32).reshape(-1)

    start = max(0, sec_to_samples(start_sec, sr))
    if end_sec is None:
        end = start
    else:
        end = min(len(y), max(start, sec_to_samples(end_sec, sr)))

    left = y[:start]
    right = y[end:]
    removed = end - start
    added = len(insert)
    delta_sec = samples_to_sec(added - removed, sr)

    if len(left) == 0 and len(right) == 0:
        return insert.copy(), delta_sec
    if len(left) == 0:
        out = _crossfade(insert, right, sr, fade_ms, curve) if len(right) else insert
        return out, delta_sec
    if len(right) == 0:
        out = _crossfade(left, insert, sr, fade_ms, curve) if len(insert) else left
        return out, delta_sec
    if len(insert) == 0:
        out = _crossfade(left, right, sr, fade_ms, curve)
        return out, delta_sec

    mid = _crossfade(left, insert, sr, fade_ms, curve)
    out = _crossfade(mid, right, sr, fade_ms, curve)
    return out, delta_sec


def replace_segment(
    y: np.ndarray,
    replacement: np.ndarray,
    start_sec: float,
    end_sec: float,
    sr: int = TARGET_SR,
    fade_ms: float = DEFAULT_CROSSFADE_MS,
) -> tuple[np.ndarray, float]:
    """Replace a temporal window with a (possibly different-length) segment."""
    return splice_with_crossfade(
        y, replacement, start_sec, end_sec, sr=sr, fade_ms=fade_ms, curve="cosine"
    )


def insert_silence(
    y: np.ndarray,
    at_sec: float,
    silence_sec: float,
    sr: int = TARGET_SR,
    fade_ms: float = DEFAULT_CROSSFADE_MS,
) -> tuple[np.ndarray, float]:
    """Insert dead silence at `at_sec` with cosine cross-fades; returns (audio, delta)."""
    n_sil = max(1, sec_to_samples(silence_sec, sr))
    silence = np.zeros(n_sil, dtype=np.float32)
    return splice_with_crossfade(
        y, silence, at_sec, end_sec=None, sr=sr, fade_ms=fade_ms, curve="cosine"
    )


# ---------------------------------------------------------------------------
# Time / pitch / energy transforms
# ---------------------------------------------------------------------------

def time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    """
    Time-stretch audio by `rate` (>1 = faster/shorter, <1 = slower/longer).
    Uses librosa phase vocoder; preserves pitch.
    """
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if abs(rate - 1.0) < 1e-6 or len(y) < 16:
        return y.copy()
    # librosa.effects.time_stretch: rate > 1 speeds up
    return librosa.effects.time_stretch(y, rate=float(rate)).astype(np.float32)


def time_stretch_segment(
    y: np.ndarray,
    start_sec: float,
    end_sec: float,
    rate: float,
    sr: int = TARGET_SR,
    fade_ms: float = DEFAULT_CROSSFADE_MS,
) -> tuple[np.ndarray, float, float, float]:
    """
    Time-stretch [start_sec, end_sec) and splice back with cross-fades.

    Returns (new_audio, new_start_sec, new_end_sec, duration_delta_sec).
    """
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    start = max(0, sec_to_samples(start_sec, sr))
    end = min(len(y), max(start + 1, sec_to_samples(end_sec, sr)))
    segment = y[start:end]
    stretched = time_stretch(segment, rate)
    new_y, delta = replace_segment(
        y, stretched, start_sec, end_sec, sr=sr, fade_ms=fade_ms
    )
    new_end = start_sec + samples_to_sec(len(stretched), sr)
    return new_y, start_sec, new_end, delta


def pitch_shift(y: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """Shift pitch by n_steps semitones without changing duration."""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if abs(n_steps) < 1e-6 or len(y) < 16:
        return y.copy()
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=float(n_steps)).astype(np.float32)


def pitch_flatten(
    y: np.ndarray,
    sr: int = TARGET_SR,
    suppression: float = 0.8,
    hop_length: int = 256,
) -> np.ndarray:
    """
    Suppress F0 variance toward the segment mean pitch.

    `suppression` in [0, 1]: 0 = unchanged, 1 = fully flatten to mean F0.
    Implemented via short-time resampling of pitch periods (PSOLA-lite using
    librosa pyin + resampling of frames), falling back to mild low-pass
    envelope when F0 tracking fails.
    """
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    suppression = float(np.clip(suppression, 0.0, 1.0))
    if suppression < 1e-3 or len(y) < sr // 4:
        return y.copy()

    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C5"),
        sr=sr,
        hop_length=hop_length,
    )
    if f0 is None or np.all(np.isnan(f0)):
        # Fallback: attenuate spectral brightness / micro-modulation via mild LP
        b, a = signal.butter(2, 800 / (sr / 2), btype="low")
        smoothed = signal.filtfilt(b, a, y).astype(np.float32)
        return ((1.0 - suppression) * y + suppression * smoothed).astype(np.float32)

    voiced = ~np.isnan(f0)
    if not np.any(voiced):
        b, a = signal.butter(2, 800 / (sr / 2), btype="low")
        smoothed = signal.filtfilt(b, a, y).astype(np.float32)
        return ((1.0 - suppression) * y + suppression * smoothed).astype(np.float32)

    mean_f0 = float(np.nanmean(f0))
    # Ratio map: pull each frame's F0 toward mean_f0
    ratio = np.ones_like(f0, dtype=np.float32)
    ratio[voiced] = (mean_f0 / np.maximum(f0[voiced], 1e-3)) ** suppression

    # Apply via overlapping chunk pitch-shifts toward mean F0.
    n_fft = 512 if len(y) < 4096 else 1024
    out = np.zeros_like(y)
    weight = np.zeros_like(y)
    win = max(hop_length * 4, n_fft)
    for i, r in enumerate(ratio):
        center = i * hop_length
        if center >= len(y):
            break
        s = max(0, center - win // 2)
        e = min(len(y), center + win // 2)
        chunk = y[s:e]
        if len(chunk) < n_fft:
            out[s:e] += chunk
            weight[s:e] += 1.0
            continue
        # Convert frequency ratio to semitones
        n_steps = 12.0 * np.log2(float(r) + 1e-12) if r > 0 else 0.0
        # Clamp extreme shifts for stability
        n_steps = float(np.clip(n_steps, -4.0, 4.0))
        if abs(n_steps) > 0.05:
            shifted = librosa.effects.pitch_shift(
                chunk, sr=sr, n_steps=n_steps, n_fft=n_fft
            ).astype(np.float32)
        else:
            shifted = chunk
        # Match length
        if len(shifted) < (e - s):
            shifted = np.pad(shifted, (0, (e - s) - len(shifted)))
        else:
            shifted = shifted[: e - s]
        w = np.hanning(len(shifted)).astype(np.float32)
        out[s:e] += shifted * w
        weight[s:e] += w

    weight = np.maximum(weight, 1e-6)
    flattened = (out / weight).astype(np.float32)
    # Blend with original for partial suppression already encoded in ratios;
    # residual blend keeps unvoiced regions natural.
    return flattened


def apply_gain_db(y: np.ndarray, gain_db: float) -> np.ndarray:
    """Scale amplitude by gain_db (positive = louder). May clip if not re-normalized."""
    y = np.asarray(y, dtype=np.float32)
    return (y * (10.0 ** (gain_db / 20.0))).astype(np.float32)


def apply_gain_bursts(
    y: np.ndarray,
    burst_windows: list[tuple[float, float]],
    gain_db: float,
    sr: int = TARGET_SR,
) -> np.ndarray:
    """
    Apply short gain bursts over [(start, end), ...] windows with soft edges.
    """
    out = np.asarray(y, dtype=np.float32).copy()
    fade = _fade_length_samples(sr, 8.0)
    scale = 10.0 ** (gain_db / 20.0)
    for start_sec, end_sec in burst_windows:
        s = max(0, sec_to_samples(start_sec, sr))
        e = min(len(out), sec_to_samples(end_sec, sr))
        if e <= s:
            continue
        env = np.ones(e - s, dtype=np.float32) * scale
        n = min(fade, (e - s) // 2)
        if n > 0:
            ramp = np.linspace(1.0, scale, n, dtype=np.float32)
            env[:n] = ramp
            env[-n:] = ramp[::-1]
        out[s:e] *= env
    return out


# ---------------------------------------------------------------------------
# Timestamp shift tracking
# ---------------------------------------------------------------------------

@dataclass
class TimestampTracker:
    """
    Tracks how injected edits shift subsequent timeline coordinates.

    When silence is inserted or a segment is time-stretched, every flaw /
    word boundary at or after the edit point must be shifted by the net
    duration delta so ground-truth labels stay sample-accurate.
    """

    edits: list[tuple[float, float]] = field(default_factory=list)
    # each edit: (anchor_time_on_current_timeline, delta_sec)

    def record(self, anchor_sec: float, delta_sec: float) -> None:
        if abs(delta_sec) < TIMESTAMP_SHIFT_EPS:
            return
        self.edits.append((float(anchor_sec), float(delta_sec)))

    def map_time(self, t: float) -> float:
        """Map an original-timeline time through all recorded edits in order."""
        mapped = float(t)
        # Edits are recorded in application order on the evolving timeline.
        # Replaying them: for each edit, if mapped >= anchor, shift by delta.
        # Because anchors were on the then-current timeline, we need to apply
        # chronologically as stored (application order).
        for anchor, delta in self.edits:
            if mapped + TIMESTAMP_SHIFT_EPS >= anchor:
                mapped += delta
        return mapped

    def map_window(self, start: float, end: float) -> tuple[float, float]:
        return self.map_time(start), self.map_time(end)

    def cumulative_shift_after(self, t: float) -> float:
        return self.map_time(t) - t
