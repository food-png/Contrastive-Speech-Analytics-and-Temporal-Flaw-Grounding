"""
Phase 3 — Speaker-agnostic acoustic feature extraction.

Frame grid: 16 kHz, n_fft=1024 (~64 ms), hop=512 (~32 ms, ~31.25 Hz).
Pitch is z-scored on voiced frames; energy is dB relative to the 95th-percentile RMS.
Pacing (WPM / pause ratio) is fused from Phase 2 word/pause alignments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

EPS = 1e-8
SR = 16_000
N_FFT = 1024
HOP_LENGTH = 512
FMIN = 65.0
FMAX = 400.0
F0_GAP_MAX_SEC = 0.20
WPM_WINDOW_SEC = 3.0
JSON_HZ = 20.0
MFCC_N = 13
ENERGY_FLOOR_DB = -60.0


def _as_mono(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)
    return y


def _finite(x: np.ndarray, fill: float = 0.0) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64)
    bad = ~np.isfinite(out)
    if np.any(bad):
        out[bad] = fill
    return out


def _match_length(x: np.ndarray, t: int, axis: int = -1) -> np.ndarray:
    x = np.asarray(x)
    n = x.shape[axis]
    if n == t:
        return x
    if n > t:
        sl = [slice(None)] * x.ndim
        sl[axis] = slice(0, t)
        return x[tuple(sl)]
    pad_width = [(0, 0)] * x.ndim
    pads = list(pad_width)
    pads[axis] = (0, t - n)
    return np.pad(x, pads, mode="edge" if n > 0 else "constant")


def _interp_short_f0_gaps(
    f0: np.ndarray,
    times: np.ndarray,
    voiced: np.ndarray,
    max_gap_sec: float = F0_GAP_MAX_SEC,
) -> np.ndarray:
    """Linearly fill unvoiced holes shorter than max_gap_sec; leave longer gaps as NaN."""
    filled = np.where(voiced & np.isfinite(f0) & (f0 > 0), f0.astype(np.float64), np.nan)
    n = len(filled)
    i = 0
    while i < n:
        if not np.isnan(filled[i]):
            i += 1
            continue
        j = i
        while j < n and np.isnan(filled[j]):
            j += 1
        left = i - 1
        right = j
        gap_dur = times[min(j, n - 1)] - times[i]
        if left >= 0 and right < n and gap_dur <= max_gap_sec:
            filled[i:j] = np.interp(
                times[i:j],
                [times[left], times[right]],
                [filled[left], filled[right]],
            )
        i = j
    return filled


def _load_alignment(path: str | Path | None) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    if path is None:
        return [], []
    p = Path(path)
    if not p.exists():
        return [], []
    data = json.loads(p.read_text(encoding="utf-8"))
    words = [
        (float(w["start_sec"]), float(w["end_sec"]))
        for w in data.get("words", [])
        if float(w.get("end_sec", 0)) > float(w.get("start_sec", 0))
    ]
    pauses = [
        (float(q["start_sec"]), float(q["end_sec"]))
        for q in data.get("pauses", [])
        if float(q.get("end_sec", 0)) > float(q.get("start_sec", 0))
    ]
    return words, pauses


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _rolling_pacing(
    times: np.ndarray,
    words: list[tuple[float, float]],
    pauses: list[tuple[float, float]],
    duration: float,
    window_sec: float = WPM_WINDOW_SEC,
) -> tuple[np.ndarray, np.ndarray]:
    half = window_sec / 2.0
    wpm = np.zeros(len(times), dtype=np.float64)
    pause_ratio = np.zeros(len(times), dtype=np.float64)
    if duration <= 0:
        return wpm, pause_ratio

    # If pauses missing, treat gaps between words as pause (plus leading/trailing).
    if not pauses and words:
        cursor = 0.0
        inferred: list[tuple[float, float]] = []
        for s, e in words:
            if s - cursor >= 0.15:
                inferred.append((cursor, s))
            cursor = max(cursor, e)
        if duration - cursor >= 0.15:
            inferred.append((cursor, duration))
        pauses = inferred

    for i, t in enumerate(times):
        w0 = max(0.0, t - half)
        w1 = min(duration, t + half)
        width = max(w1 - w0, EPS)
        n_words = 0
        for s, e in words:
            if e > w0 and s < w1:
                n_words += 1
        wpm[i] = n_words * (60.0 / window_sec)
        cov = 0.0
        for s, e in pauses:
            cov += _overlap(w0, w1, s, e)
        pause_ratio[i] = float(np.clip(cov / width, 0.0, 1.0))
    return wpm, pause_ratio


def _spectral_flux(y: np.ndarray, sr: int, n_fft: int, hop: int) -> np.ndarray:
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=n_fft, center=True))
    # Half-wave rectified Euclidean flux over frequency bins
    diff = np.diff(stft, axis=1, prepend=stft[:, :1])
    flux = np.sqrt(np.sum(np.maximum(diff, 0.0) ** 2, axis=0) + EPS)
    peak = float(np.percentile(flux, 95)) if flux.size else 1.0
    return (flux / (peak + EPS)).astype(np.float64)


@dataclass
class AcousticTimeSeries:
    time_seconds: np.ndarray
    f0_raw: np.ndarray
    f0_norm: np.ndarray
    energy_norm: np.ndarray
    rolling_wpm: np.ndarray
    spectral_flux: np.ndarray
    mfcc_matrix: np.ndarray
    voiced_mask: np.ndarray
    metadata: dict = field(default_factory=dict)
    pause_ratio: np.ndarray | None = None
    spectral_centroid: np.ndarray | None = None

    def to_dict(self, downsample_hz: float | None = None) -> dict:
        t = self.time_seconds
        if downsample_hz and downsample_hz > 0 and t.size > 2:
            t_new = np.arange(t[0], t[-1] + 1e-9, 1.0 / downsample_hz, dtype=np.float64)
            if t_new.size == 0:
                t_new = t[:1]

            def _i(x: np.ndarray) -> list[float]:
                return np.interp(t_new, t, x).astype(np.float64).round(5).tolist()

            voiced_ds = (np.interp(t_new, t, self.voiced_mask.astype(np.float64)) >= 0.5).astype(int)
            mfcc_ds = [
                np.interp(t_new, t, self.mfcc_matrix[k]).astype(np.float64).round(5).tolist()
                for k in range(self.mfcc_matrix.shape[0])
            ]
            payload = {
                "time_seconds": t_new.round(5).tolist(),
                "f0_raw": _i(self.f0_raw),
                "f0_norm": _i(self.f0_norm),
                "energy_norm": _i(self.energy_norm),
                "rolling_wpm": _i(self.rolling_wpm),
                "spectral_flux": _i(self.spectral_flux),
                "voiced_mask": voiced_ds.tolist(),
                "mfcc_matrix": mfcc_ds,
                "metadata": self.metadata,
            }
            if self.pause_ratio is not None:
                payload["pause_ratio"] = _i(self.pause_ratio)
            if self.spectral_centroid is not None:
                payload["spectral_centroid"] = _i(self.spectral_centroid)
            return payload

        payload = {
            "time_seconds": self.time_seconds.astype(np.float64).round(5).tolist(),
            "f0_raw": self.f0_raw.astype(np.float64).round(5).tolist(),
            "f0_norm": self.f0_norm.astype(np.float64).round(5).tolist(),
            "energy_norm": self.energy_norm.astype(np.float64).round(5).tolist(),
            "rolling_wpm": self.rolling_wpm.astype(np.float64).round(5).tolist(),
            "spectral_flux": self.spectral_flux.astype(np.float64).round(5).tolist(),
            "voiced_mask": self.voiced_mask.astype(int).tolist(),
            "mfcc_matrix": self.mfcc_matrix.astype(np.float64).round(5).tolist(),
            "metadata": self.metadata,
        }
        if self.pause_ratio is not None:
            payload["pause_ratio"] = self.pause_ratio.astype(np.float64).round(5).tolist()
        if self.spectral_centroid is not None:
            payload["spectral_centroid"] = self.spectral_centroid.astype(np.float64).round(5).tolist()
        return payload

    def save_npz(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(path),
            time_seconds=self.time_seconds.astype(np.float32),
            f0_raw=self.f0_raw.astype(np.float32),
            f0_norm=self.f0_norm.astype(np.float32),
            energy_norm=self.energy_norm.astype(np.float32),
            rolling_wpm=self.rolling_wpm.astype(np.float32),
            spectral_flux=self.spectral_flux.astype(np.float32),
            mfcc_matrix=self.mfcc_matrix.astype(np.float32),
            voiced_mask=self.voiced_mask.astype(np.uint8),
            pause_ratio=(
                self.pause_ratio.astype(np.float32)
                if self.pause_ratio is not None
                else np.zeros_like(self.time_seconds, dtype=np.float32)
            ),
            spectral_centroid=(
                self.spectral_centroid.astype(np.float32)
                if self.spectral_centroid is not None
                else np.zeros_like(self.time_seconds, dtype=np.float32)
            ),
            metadata_json=np.array(json.dumps(self.metadata)),
        )

    def save_json(self, output_path: str | Path, downsample_hz: float = JSON_HZ) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(downsample_hz=downsample_hz)) + "\n", encoding="utf-8")


class AcousticFeatureExtractor:
    def __init__(
        self,
        sr: int = SR,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
    ) -> None:
        self.sr = int(sr)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)

    def extract_features(
        self,
        audio_path: str | Path,
        alignment_path: str | Path | None = None,
    ) -> AcousticTimeSeries:
        audio_path = Path(audio_path)
        y, file_sr = sf.read(str(audio_path), always_2d=False)
        y = _as_mono(y)
        if file_sr != self.sr:
            y = librosa.resample(y, orig_sr=file_sr, target_sr=self.sr)
        duration = float(len(y) / self.sr) if self.sr else 0.0

        # Reference frame grid from STFT (center=True)
        stft = librosa.stft(
            y, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft, center=True
        )
        t_frames = int(stft.shape[1])
        times = librosa.frames_to_time(
            np.arange(t_frames), sr=self.sr, hop_length=self.hop_length, n_fft=self.n_fft
        ).astype(np.float64)

        # --- Pitch / voicing ---
        f0_pyin, voiced_flag, _ = librosa.pyin(
            y,
            fmin=FMIN,
            fmax=FMAX,
            sr=self.sr,
            frame_length=self.n_fft,
            hop_length=self.hop_length,
            fill_na=np.nan,
        )
        f0_pyin = _match_length(np.asarray(f0_pyin, dtype=np.float64), t_frames)
        if voiced_flag is None:
            voiced = np.isfinite(f0_pyin) & (f0_pyin > 0)
        else:
            voiced = _match_length(np.asarray(voiced_flag, dtype=bool), t_frames)
            voiced = voiced & np.isfinite(f0_pyin) & (f0_pyin > 0)

        f0_filled = _interp_short_f0_gaps(f0_pyin, times, voiced, F0_GAP_MAX_SEC)
        voiced_vals = f0_filled[voiced & np.isfinite(f0_filled)]
        if voiced_vals.size >= 2:
            mu = float(np.mean(voiced_vals))
            sigma = float(np.std(voiced_vals))
        elif voiced_vals.size == 1:
            mu = float(voiced_vals[0])
            sigma = 0.0
        else:
            mu, sigma = 0.0, 0.0
        f0_raw = np.where(np.isfinite(f0_filled), f0_filled, 0.0)
        f0_norm = (f0_raw - mu) / (sigma + EPS)
        f0_norm = np.where(np.isfinite(f0_filled), f0_norm, 0.0)
        # Unvoiced frames that were not interpolated stay at 0 deviation
        f0_norm = _finite(f0_norm, 0.0)
        f0_raw = _finite(f0_raw, 0.0)

        # --- RMS energy → relative dB ---
        rms = librosa.feature.rms(
            y=y, frame_length=self.n_fft, hop_length=self.hop_length, center=True
        )[0]
        rms = _match_length(np.asarray(rms, dtype=np.float64), t_frames)
        rms95 = float(np.percentile(rms, 95)) if rms.size else 0.0
        energy_db = 20.0 * np.log10((rms + EPS) / (rms95 + EPS))
        energy_norm = np.clip(energy_db, ENERGY_FLOOR_DB, 0.0)

        # --- Spectral centroid, flux, MFCCs ---
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        centroid = _match_length(np.asarray(centroid, dtype=np.float64), t_frames)
        flux = _match_length(_spectral_flux(y, self.sr, self.n_fft, self.hop_length), t_frames)

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=self.sr,
            n_mfcc=MFCC_N,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        mfcc = _match_length(np.asarray(mfcc, dtype=np.float64), t_frames, axis=1)
        # Per-coefficient z-score over time
        mfcc_mu = mfcc.mean(axis=1, keepdims=True)
        mfcc_sd = mfcc.std(axis=1, keepdims=True)
        mfcc_norm = (mfcc - mfcc_mu) / (mfcc_sd + EPS)
        mfcc_norm = _finite(mfcc_norm, 0.0)

        # --- Pacing from Phase 2 alignments ---
        words, pauses = _load_alignment(alignment_path)
        rolling_wpm, pause_ratio = _rolling_pacing(times, words, pauses, duration)

        pause_dur = sum(max(0.0, e - s) for s, e in pauses)
        pause_pct = (pause_dur / duration * 100.0) if duration > 0 else 0.0
        median_wpm = float(np.median(rolling_wpm)) if rolling_wpm.size else 0.0

        sample_id = audio_path.stem
        metadata = {
            "sample_id": sample_id,
            "audio_path": str(audio_path).replace("\\", "/"),
            "alignment_path": str(alignment_path).replace("\\", "/") if alignment_path else None,
            "duration_sec": round(duration, 4),
            "sr": self.sr,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "n_frames": t_frames,
            "frame_rate_hz": round(self.sr / self.hop_length, 4),
            "mean_f0": round(mu, 4),
            "std_f0": round(sigma, 4),
            "median_wpm": round(median_wpm, 4),
            "pause_percentage": round(pause_pct, 4),
            "voiced_percentage": round(float(np.mean(voiced) * 100.0), 4) if voiced.size else 0.0,
            "rms95": round(float(rms95), 8),
        }

        return AcousticTimeSeries(
            time_seconds=times,
            f0_raw=f0_raw.astype(np.float64),
            f0_norm=f0_norm.astype(np.float64),
            energy_norm=_finite(energy_norm).astype(np.float64),
            rolling_wpm=_finite(rolling_wpm).astype(np.float64),
            spectral_flux=_finite(flux).astype(np.float64),
            mfcc_matrix=mfcc_norm.astype(np.float64),
            voiced_mask=voiced.astype(bool),
            metadata=metadata,
            pause_ratio=_finite(pause_ratio).astype(np.float64),
            spectral_centroid=_finite(centroid).astype(np.float64),
        )
