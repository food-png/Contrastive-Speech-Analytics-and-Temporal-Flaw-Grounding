#!/usr/bin/env python3
"""
Phase 2 — Batch forced alignment over baseline + flawed-spectrum WAVs.

Reads transcripts from dataset/raw_baseline/speech_{A,B,C}.txt and writes
dataset/alignments/<sample_id>_aligned.json for every clip.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.aligner import ForcedAligner  # noqa: E402

console = Console()

AUDIO_DIRS = [
    ROOT / "dataset" / "processed_baseline",
    ROOT / "dataset" / "flawed_spectrum",
]
RAW_DIR = ROOT / "dataset" / "raw_baseline"
ALIGN_DIR = ROOT / "dataset" / "alignments"
ANN_PATH = ROOT / "dataset" / "annotations" / "ground_truth_flaws.json"
STEM_KEY = re.compile(r"speech_([A-Ca-c])")


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _sample_id(wav: Path) -> str:
    return wav.stem


def _transcript_for(wav: Path, ann_index: dict[str, str]) -> Path:
    sid = wav.stem
    if sid in ann_index:
        p = ROOT / ann_index[sid]
        if p.exists():
            return p
    m = STEM_KEY.search(sid)
    if not m:
        raise FileNotFoundError(f"Cannot infer transcript key from {wav.name}")
    key = m.group(1).upper()
    txt = RAW_DIR / f"speech_{key}.txt"
    if not txt.exists():
        raise FileNotFoundError(f"Missing transcript {txt} for {wav.name}")
    return txt


def _load_ann_transcripts() -> dict[str, str]:
    if not ANN_PATH.exists():
        return {}
    data = json.loads(ANN_PATH.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for sample in data.get("samples", []):
        sid = sample.get("sample_id")
        tpath = sample.get("transcript_path")
        if sid and tpath:
            out[sid] = tpath.replace("\\", "/")
    return out


def collect_wavs() -> list[Path]:
    wavs: list[Path] = []
    for d in AUDIO_DIRS:
        if not d.is_dir():
            continue
        wavs.extend(sorted(d.glob("*.wav")))
    return wavs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run forced alignment over the contrastive dataset.")
    parser.add_argument(
        "--backend",
        choices=("mms", "wav2vec2", "energy"),
        default="mms",
        help="Preferred alignment backend (default: mms).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip sample_id values that already have an alignment JSON.",
    )
    args = parser.parse_args()

    console.print(
        Panel.fit(
            "[bold]Track C | Phase 2[/bold]\n"
            f"Forced alignment  |  backend={args.backend}",
            border_style="cyan",
        )
    )

    wavs = collect_wavs()
    if not wavs:
        console.print("[red]No WAV files found. Run Phase 1 scripts first.[/red]")
        raise SystemExit(1)

    ALIGN_DIR.mkdir(parents=True, exist_ok=True)
    ann_index = _load_ann_transcripts()
    aligner = ForcedAligner(prefer=args.backend)

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
        task = progress.add_task("aligning", total=len(wavs))
        for wav in wavs:
            sid = _sample_id(wav)
            out_path = ALIGN_DIR / f"{sid}_aligned.json"
            progress.update(task, description=sid)
            if args.skip_existing and out_path.exists():
                progress.advance(task)
                continue
            try:
                transcript_path = _transcript_for(wav, ann_index)
                text = transcript_path.read_text(encoding="utf-8")
                result = aligner.align_file(
                    audio_path=wav,
                    transcript=text,
                    sample_id=sid,
                    rel_audio=_rel(wav),
                    transcript_path=_rel(transcript_path),
                )
                payload = result.model_dump()
                out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                n_words = len(result.words)
                n_pauses = len(result.pauses)
                console.print(
                    f"  [green]OK[/green] {sid:<42} words={n_words:3d}  "
                    f"pauses={n_pauses:3d}  backend={result.backend}"
                )
                written += 1
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{sid}: {exc}")
                console.print(f"  [red]FAIL[/red] {sid}: {exc}")
            progress.advance(task)

    console.print(f"\nWrote {written} alignment file(s) -> {_rel(ALIGN_DIR)}")
    if failures:
        console.print(f"[red]{len(failures)} failure(s)[/red]")
        for f in failures:
            console.print(f"  • {f}")
        raise SystemExit(1)
    console.print("[bold green]Alignment batch complete.[/bold green]")


if __name__ == "__main__":
    main()
