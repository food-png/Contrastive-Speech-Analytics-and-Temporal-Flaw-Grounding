#!/usr/bin/env python3
"""
Phase 4 — Timeline inspection plots.

Writes dataset/grounding/plots/<sample_id>_grounding.png for each detection JSON
(or a single --sample_id).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from rich.console import Console
from rich.panel import Panel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

console = Console()

FLAW_DIR = ROOT / "dataset" / "flawed_spectrum"
GROUND_DIR = ROOT / "dataset" / "grounding"
PLOT_DIR = GROUND_DIR / "plots"
ANN_PATH = ROOT / "dataset" / "annotations" / "ground_truth_flaws.json"

CAT_COLORS = {
    "pacing": "#e67e22",
    "pitch": "#8e44ad",
    "energy": "#c0392b",
    "pause": "#2980b9",
    "silence": "#2980b9",
    "compound": "#16a085",
}

LEVEL_RE = re.compile(r"speech_([ABC])_level(\d+)", re.I)


def _load_gt(sample_id: str) -> list[dict]:
    if not ANN_PATH.exists():
        return []
    data = json.loads(ANN_PATH.read_text(encoding="utf-8"))
    for s in data.get("samples", []):
        if s.get("sample_id") == sample_id:
            return list(s.get("injected_flaws", []))
    return []


def _shade(ax, intervals: list[tuple[float, float, str]], alpha: float = 0.18) -> None:
    for s, e, cat in intervals:
        ax.axvspan(s, e, color=CAT_COLORS.get(cat, "#7f8c8d"), alpha=alpha, lw=0)


def plot_one(sample_id: str) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        console.print("[yellow]matplotlib missing; cannot plot.[/yellow]")
        return None

    det_path = GROUND_DIR / f"{sample_id}_detected.json"
    wav_path = FLAW_DIR / f"{sample_id}.wav"
    if not det_path.exists() or not wav_path.exists():
        console.print(f"[yellow]Skip {sample_id}: missing detection or wav.[/yellow]")
        return None

    det = json.loads(det_path.read_text(encoding="utf-8"))
    tracks = det.get("residual_summary", {}).get("tracks", {})
    resid_path = GROUND_DIR / "residuals" / f"{sample_id}_residuals.npz"
    if not tracks and resid_path.exists():
        with np.load(str(resid_path)) as z:
            tracks = {k: z[k] for k in z.files}
    gt_flaws = _load_gt(sample_id)
    det_flaws = det.get("flaws", [])

    y, sr = sf.read(str(wav_path), always_2d=False)
    if getattr(y, "ndim", 1) > 1:
        y = np.mean(y, axis=-1)
    t_wav = np.arange(len(y)) / float(sr)

    t = np.asarray(tracks.get("time", []), dtype=np.float64)
    f0_c = np.asarray(tracks.get("f0_cand", []), dtype=np.float64)
    f0_b = np.asarray(tracks.get("f0_base", []), dtype=np.float64)
    wpm_c = np.asarray(tracks.get("wpm_cand", []), dtype=np.float64)
    wpm_b = np.asarray(tracks.get("wpm_base", []), dtype=np.float64)
    e_c = np.asarray(tracks.get("energy_cand", []), dtype=np.float64)
    e_b = np.asarray(tracks.get("energy_base", []), dtype=np.float64)
    mono = np.asarray(tracks.get("monotone_mask", []), dtype=bool)
    rush = np.asarray(tracks.get("rush_mask", []), dtype=bool)
    drag = np.asarray(tracks.get("drag_mask", []), dtype=bool)
    spike = np.asarray(tracks.get("spike_mask", []), dtype=bool)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOT_DIR / f"{sample_id}_grounding.png"

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=False)
    fig.suptitle(
        f"Track C | Temporal grounding — {sample_id}\n"
        f"GT flaws={len(gt_flaws)}  Detected={len(det_flaws)}",
        fontsize=11,
    )

    # Track 1: waveform + GT (green) + detections (red boxes)
    ax = axes[0]
    ax.plot(t_wav, y, color="#34495e", lw=0.35, alpha=0.85)
    ymin, ymax = float(np.min(y)), float(np.max(y))
    pad = 0.05 * max(ymax - ymin, 1e-3)
    for g in gt_flaws:
        ax.axvspan(g["start_sec"], g["end_sec"], color="#27ae60", alpha=0.28, label="_gt")
    for d in det_flaws:
        rect = Rectangle(
            (d["start_sec"], ymin - pad),
            max(0.05, d["end_sec"] - d["start_sec"]),
            (ymax - ymin) + 2 * pad,
            fill=False,
            edgecolor="#e74c3c",
            linewidth=1.4,
        )
        ax.add_patch(rect)
    ax.set_ylabel("amp")
    ax.set_title("Track 1 — Waveform  |  green=GT  red boxes=detected")
    ax.set_xlim(0, t_wav[-1] if len(t_wav) else 1)
    ax.grid(True, alpha=0.25)

    # Track 2: pitch
    ax = axes[1]
    if t.size:
        ax.plot(t, f0_b, color="#3498db", lw=1.0, label="F0 base (mapped)", alpha=0.85)
        ax.plot(t, f0_c, color="#e74c3c", lw=1.0, label="F0 cand", alpha=0.85)
        if mono.size == t.size:
            _shade(ax, _mask_spans(t, mono, "pitch"), alpha=0.22)
        for g in gt_flaws:
            if g.get("category") == "pitch":
                ax.axvspan(g["start_sec"], g["end_sec"], color="#27ae60", alpha=0.15)
    ax.axhline(0.0, color="gray", lw=0.5, ls="--")
    ax.set_ylabel("F0 z")
    ax.set_title("Track 2 — Pitch contours with monotone zones")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)

    # Track 3: WPM
    ax = axes[2]
    if t.size:
        ax.plot(t, wpm_b, color="#3498db", lw=1.1, label="WPM base")
        ax.plot(t, wpm_c, color="#e74c3c", lw=1.1, label="WPM cand")
        if rush.size == t.size:
            _shade(ax, _mask_spans(t, rush, "pacing"), alpha=0.20)
        if drag.size == t.size:
            _shade(ax, _mask_spans(t, drag, "pacing"), alpha=0.12)
        for g in gt_flaws:
            if g.get("category") == "pacing":
                ax.axvspan(g["start_sec"], g["end_sec"], color="#27ae60", alpha=0.15)
    ax.set_ylabel("WPM")
    ax.set_title("Track 3 — Rolling cadence (shaded = rushed/dragged)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)

    # Track 4: energy
    ax = axes[3]
    if t.size:
        ax.plot(t, e_b, color="#3498db", lw=1.0, label="E base")
        ax.plot(t, e_c, color="#e74c3c", lw=1.0, label="E cand")
        if spike.size == t.size:
            idx = np.where(spike)[0]
            if idx.size:
                ax.scatter(t[idx], e_c[idx], c="#c0392b", s=12, zorder=5, label="spike")
        for g in gt_flaws:
            if g.get("category") in ("energy", "silence", "pause"):
                ax.axvspan(
                    g["start_sec"],
                    g["end_sec"],
                    color="#27ae60" if g.get("category") != "energy" else "#f1c40f",
                    alpha=0.18,
                )
        for d in det_flaws:
            if d.get("category") == "pause":
                ax.axvspan(d["start_sec"], d["end_sec"], color="#2980b9", alpha=0.20)
    ax.set_ylabel("dB rel")
    ax.set_xlabel("time (s)")
    ax.set_title("Track 4 — Relative energy with spike / pause markers")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)

    for ax in axes:
        if t_wav.size:
            ax.set_xlim(0, t_wav[-1])

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def _mask_spans(t: np.ndarray, mask: np.ndarray, cat: str) -> list[tuple[float, float, str]]:
    spans: list[tuple[float, float, str]] = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        spans.append((float(t[i]), float(t[min(j - 1, n - 1)]), cat))
        i = j
    return spans


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot temporal grounding timelines.")
    parser.add_argument("--sample-id", type=str, default=None, help="Plot a single sample.")
    args = parser.parse_args()

    console.print(
        Panel.fit(
            "[bold]Track C | Phase 4[/bold]\nTemporal grounding visualization",
            border_style="cyan",
        )
    )

    if args.sample_id:
        ids = [args.sample_id]
    else:
        ids = sorted(p.name.replace("_detected.json", "") for p in GROUND_DIR.glob("*_detected.json"))

    if not ids:
        console.print("[red]No detection JSONs found. Run scripts/run_temporal_grounding.py first.[/red]")
        raise SystemExit(1)

    written = 0
    for sid in ids:
        out = plot_one(sid)
        if out:
            console.print(f"  [green]OK[/green] {out.relative_to(ROOT)}")
            written += 1
    console.print(f"\nWrote {written} plot(s) -> {PLOT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
