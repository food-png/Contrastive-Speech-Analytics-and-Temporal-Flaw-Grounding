#!/usr/bin/env python3
"""
Phase 3 — Batch acoustic feature extraction.

Writes:
  dataset/features/npz/<sample_id>_features.npz
  dataset/features/json/<sample_id>_features.json   (20 Hz curves)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.feature_extractor import AcousticFeatureExtractor  # noqa: E402

console = Console()

AUDIO_DIRS = [
    ROOT / "dataset" / "processed_baseline",
    ROOT / "dataset" / "flawed_spectrum",
]
ALIGN_DIR = ROOT / "dataset" / "alignments"
FEAT_NPZ = ROOT / "dataset" / "features" / "npz"
FEAT_JSON = ROOT / "dataset" / "features" / "json"


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def collect_wavs() -> list[Path]:
    wavs: list[Path] = []
    for d in AUDIO_DIRS:
        if d.is_dir():
            wavs.extend(sorted(d.glob("*.wav")))
    return wavs


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract speaker-agnostic acoustic features.")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    console.print(
        Panel.fit(
            "[bold]Track C | Phase 3[/bold]\n"
            "Acoustic feature extraction  |  16 kHz  |  n_fft=1024  |  hop=512",
            border_style="cyan",
        )
    )

    wavs = collect_wavs()
    if not wavs:
        console.print("[red]No WAV files found. Run Phase 1 first.[/red]")
        raise SystemExit(1)

    FEAT_NPZ.mkdir(parents=True, exist_ok=True)
    FEAT_JSON.mkdir(parents=True, exist_ok=True)
    extractor = AcousticFeatureExtractor()

    failures: list[str] = []
    written = 0

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task("extracting", total=len(wavs))
        for wav in wavs:
            sid = wav.stem
            progress.update(task, description=sid)
            npz_path = FEAT_NPZ / f"{sid}_features.npz"
            json_path = FEAT_JSON / f"{sid}_features.json"
            if args.skip_existing and npz_path.exists() and json_path.exists():
                progress.advance(task)
                continue
            align_path = ALIGN_DIR / f"{sid}_aligned.json"
            try:
                series = extractor.extract_features(
                    wav, alignment_path=align_path if align_path.exists() else None
                )
                series.save_npz(npz_path)
                series.save_json(json_path)
                md = series.metadata
                console.print(
                    f"  [green]OK[/green] {sid:<42} T={md['n_frames']:<5}  "
                    f"std_f0={md['std_f0']:.2f}  "
                    f"WPM={md['median_wpm']:.1f}  "
                    f"pause={md['pause_percentage']:.1f}%"
                )
                written += 1
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{sid}: {exc}")
                console.print(f"  [red]FAIL[/red] {sid}: {exc}")
            progress.advance(task)

    console.print(f"\nWrote {written} feature pack(s) -> {_rel(FEAT_NPZ.parent)}")
    if failures:
        console.print(f"[red]{len(failures)} failure(s)[/red]")
        for f in failures:
            console.print(f"  • {f}")
        raise SystemExit(1)
    console.print("[bold green]Feature extraction complete.[/bold green]")


if __name__ == "__main__":
    main()
