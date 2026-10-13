#!/usr/bin/env python3
"""
Phase 3 — Feature QA + contrast plot for speech A baseline vs Level-2.

Checks:
  1. Every WAV has matching .npz and .json
  2. Arrays are non-empty, finite, and share frame length T
  3. Time axis covers the audio duration (within one hop)
Writes dataset/features/qc_plots/speech_A_contrast.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.feature_extractor import HOP_LENGTH, SR  # noqa: E402

console = Console()

AUDIO_DIRS = [
    ROOT / "dataset" / "processed_baseline",
    ROOT / "dataset" / "flawed_spectrum",
]
FEAT_NPZ = ROOT / "dataset" / "features" / "npz"
FEAT_JSON = ROOT / "dataset" / "features" / "json"
PLOT_DIR = ROOT / "dataset" / "features" / "qc_plots"

REQUIRED = (
    "time_seconds",
    "f0_raw",
    "f0_norm",
    "energy_norm",
    "rolling_wpm",
    "spectral_flux",
    "mfcc_matrix",
    "voiced_mask",
)
HOP_SEC = HOP_LENGTH / SR


def _load_npz(path: Path) -> dict:
    with np.load(str(path), allow_pickle=True) as z:
        data = {k: z[k] for k in z.files}
    meta = {}
    if "metadata_json" in data:
        raw = data["metadata_json"]
        meta = json.loads(str(raw.item() if hasattr(raw, "item") else raw))
    data["metadata"] = meta
    return data


def _audio_duration(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.frames / info.samplerate) if info.samplerate else 0.0


def _check_pack(sid: str, wav: Path, npz_path: Path, json_path: Path) -> list[str]:
    errors: list[str] = []
    if not npz_path.exists():
        errors.append(f"{sid}: missing npz")
        return errors
    if not json_path.exists():
        errors.append(f"{sid}: missing json")

    data = _load_npz(npz_path)
    for key in REQUIRED:
        if key not in data:
            errors.append(f"{sid}: npz missing '{key}'")
            continue
        arr = np.asarray(data[key])
        if arr.size == 0:
            errors.append(f"{sid}: '{key}' is empty")
            continue
        if key != "voiced_mask" and not np.all(np.isfinite(arr)):
            errors.append(f"{sid}: '{key}' contains NaN/Inf")

    t = np.asarray(data.get("time_seconds", []))
    if t.size == 0:
        return errors
    n = int(t.shape[0])
    for key in ("f0_raw", "f0_norm", "energy_norm", "rolling_wpm", "spectral_flux", "voiced_mask"):
        if key in data and int(np.asarray(data[key]).shape[-1]) != n:
            errors.append(
                f"{sid}: {key} length {np.asarray(data[key]).shape[-1]} != T={n}"
            )
    if "mfcc_matrix" in data:
        mfcc = np.asarray(data["mfcc_matrix"])
        if mfcc.ndim != 2 or mfcc.shape[0] != 13 or mfcc.shape[1] != n:
            errors.append(f"{sid}: mfcc_matrix shape {mfcc.shape} != (13, {n})")

    dur = _audio_duration(wav)
    if t[0] < -HOP_SEC or t[0] > HOP_SEC * 2:
        errors.append(f"{sid}: time[0]={t[0]:.4f} not near 0")
    if t[-1] + HOP_SEC < dur - 2 * HOP_SEC or t[-1] > dur + 2 * HOP_SEC:
        errors.append(
            f"{sid}: time[-1]={t[-1]:.3f}s does not cover duration {dur:.3f}s"
        )

    if json_path.exists():
        js = json.loads(json_path.read_text(encoding="utf-8"))
        for key in ("time_seconds", "f0_norm", "energy_norm", "rolling_wpm"):
            seq = js.get(key, [])
            if not seq:
                errors.append(f"{sid}: json '{key}' empty")
            else:
                arr = np.asarray(seq, dtype=np.float64)
                if not np.all(np.isfinite(arr)):
                    errors.append(f"{sid}: json '{key}' has NaN/Inf")
    return errors


def _plot_contrast() -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        console.print("[yellow]matplotlib not installed; skipping PNG plot.[/yellow]")
        return None

    baseline_wav = ROOT / "dataset" / "processed_baseline" / "speech_A_baseline.wav"
    flawed_wav = ROOT / "dataset" / "flawed_spectrum" / "speech_A_level2_pacing_monotone.wav"
    b_npz = FEAT_NPZ / "speech_A_baseline_features.npz"
    f_npz = FEAT_NPZ / "speech_A_level2_pacing_monotone_features.npz"
    if not (baseline_wav.exists() and flawed_wav.exists() and b_npz.exists() and f_npz.exists()):
        console.print("[yellow]Missing speech_A pair; skipping contrast plot.[/yellow]")
        return None

    y_b, sr_b = sf.read(str(baseline_wav), always_2d=False)
    y_f, sr_f = sf.read(str(flawed_wav), always_2d=False)
    b = _load_npz(b_npz)
    f = _load_npz(f_npz)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOT_DIR / "speech_A_contrast.png"

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False)
    fig.suptitle("Track C  |  speech_A baseline vs Level-2 (pacing + monotone)", fontsize=12)

    t_wb = np.arange(len(y_b)) / float(sr_b)
    t_wf = np.arange(len(y_f)) / float(sr_f)
    axes[0].plot(t_wb, y_b, color="#1f77b4", lw=0.4, label="baseline", alpha=0.85)
    axes[0].plot(t_wf, y_f, color="#d62728", lw=0.4, label="level2", alpha=0.7)
    axes[0].set_ylabel("amplitude")
    axes[0].set_title("Panel 1 — Waveforms")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_xlim(0, max(t_wb[-1], t_wf[-1]))

    axes[1].plot(b["time_seconds"], b["f0_norm"], color="#1f77b4", lw=1.0, label="baseline F0_norm")
    axes[1].plot(f["time_seconds"], f["f0_norm"], color="#d62728", lw=1.0, label="level2 F0_norm")
    axes[1].axhline(0.0, color="gray", lw=0.6, ls="--")
    axes[1].set_ylabel("z-score")
    axes[1].set_title("Panel 2 — Normalized pitch (flatlining = low variance)")
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].plot(b["time_seconds"], b["energy_norm"], color="#1f77b4", lw=1.0, label="baseline RMS")
    axes[2].plot(f["time_seconds"], f["energy_norm"], color="#d62728", lw=1.0, label="level2 RMS")
    axes[2].set_ylabel("dB rel")
    axes[2].set_title("Panel 3 — Relative RMS energy")
    axes[2].legend(loc="upper right", fontsize=8)

    axes[3].plot(b["time_seconds"], b["rolling_wpm"], color="#1f77b4", lw=1.2, label="baseline WPM")
    axes[3].plot(f["time_seconds"], f["rolling_wpm"], color="#d62728", lw=1.2, label="level2 WPM")
    axes[3].set_ylabel("WPM")
    axes[3].set_xlabel("time (s)")
    axes[3].set_title("Panel 4 — Rolling articulation rate (3 s window)")
    axes[3].legend(loc="upper right", fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def verify() -> int:
    console.print(
        Panel.fit(
            "[bold]Track C | Phase 3[/bold]\nFeature verification & contrast plot",
            border_style="green",
        )
    )

    wavs: list[Path] = []
    for d in AUDIO_DIRS:
        if d.is_dir():
            wavs.extend(sorted(d.glob("*.wav")))
    if not wavs:
        console.print("[red]No WAV files found.[/red]")
        return 1

    errors: list[str] = []
    rows: list[tuple[str, float, float, float, str]] = []

    for wav in wavs:
        sid = wav.stem
        npz_path = FEAT_NPZ / f"{sid}_features.npz"
        json_path = FEAT_JSON / f"{sid}_features.json"
        errors.extend(_check_pack(sid, wav, npz_path, json_path))
        if not npz_path.exists():
            continue
        data = _load_npz(npz_path)
        md = data.get("metadata") or {}
        std_f0 = float(md.get("std_f0", np.std(data["f0_raw"])))
        pause = float(md.get("pause_percentage", 0.0))
        wpm = float(md.get("median_wpm", 0.0))
        kind = "baseline" if "level0" in sid or sid.endswith("_baseline") else "flawed"
        if "level" in sid and "level0" not in sid:
            kind = "flawed"
        rows.append((sid, std_f0, pause, wpm, kind))

    table = Table(title="Acoustic footprint overview")
    table.add_column("sample_id")
    table.add_column("role")
    table.add_column("std F0 (Hz)", justify="right")
    table.add_column("pause %", justify="right")
    table.add_column("median WPM", justify="right")
    for sid, std_f0, pause, wpm, kind in rows:
        table.add_row(sid, kind, f"{std_f0:.2f}", f"{pause:.1f}", f"{wpm:.1f}")
    console.print(table)

    plot_path = _plot_contrast()
    if plot_path:
        console.print(f"Contrast plot -> {plot_path.relative_to(ROOT)}")

    if errors:
        console.print("\n[red bold]Errors[/red bold]")
        for e in errors:
            console.print(f"  • {e}")
        console.print(f"\n[red]FAILED[/red] - {len(errors)} error(s)")
        return 1

    console.print("\n[bold green]PASSED[/bold green] - feature packs are finite and time-aligned.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify acoustic feature packs.")
    parser.parse_args()
    raise SystemExit(verify())


if __name__ == "__main__":
    main()
