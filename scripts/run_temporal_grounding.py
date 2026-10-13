#!/usr/bin/env python3
"""
Phase 4 — Batch contrastive temporal grounding + ground-truth evaluation.

Writes dataset/grounding/<candidate_id>_detected.json and
dataset/grounding/evaluation_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.metrics import (  # noqa: E402
    aggregate_metrics,
    det_from_grounding,
    evaluate_detection,
    gt_from_annotation,
)
from utils.temporal_grounding import TemporalFlawGrounder  # noqa: E402

console = Console()

FLAW_DIR = ROOT / "dataset" / "flawed_spectrum"
FEAT_DIR = ROOT / "dataset" / "features" / "npz"
ALIGN_DIR = ROOT / "dataset" / "alignments"
ANN_PATH = ROOT / "dataset" / "annotations" / "ground_truth_flaws.json"
OUT_DIR = ROOT / "dataset" / "grounding"

LEVEL_RE = re.compile(r"speech_([ABC])_level(\d+)", re.I)


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _baseline_id(sample_id: str) -> str:
    m = LEVEL_RE.match(sample_id)
    if not m:
        raise ValueError(f"Cannot parse baseline from {sample_id}")
    return f"speech_{m.group(1).upper()}_baseline"


def _level(sample_id: str) -> int:
    m = LEVEL_RE.match(sample_id)
    return int(m.group(2)) if m else -1


def collect_candidates() -> list[Path]:
    if not FLAW_DIR.is_dir():
        return []
    return sorted(FLAW_DIR.glob("*.wav"))


def load_gt_index() -> dict[str, dict]:
    if not ANN_PATH.exists():
        return {}
    data = json.loads(ANN_PATH.read_text(encoding="utf-8"))
    return {s["sample_id"]: s for s in data.get("samples", [])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Contrastive temporal flaw grounding.")
    parser.add_argument("--skip-level0", action="store_true", help="Skip Level-0 copies.")
    args = parser.parse_args()

    console.print(
        Panel.fit(
            "[bold]Track C | Phase 4[/bold]\n"
            "Contrastive temporal grounding & anomaly detection",
            border_style="magenta",
        )
    )

    candidates = collect_candidates()
    if not candidates:
        console.print("[red]No flawed_spectrum WAV files found.[/red]")
        raise SystemExit(1)

    gt_index = load_gt_index()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grounder = TemporalFlawGrounder()

    evals = []
    failures: list[str] = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task("grounding", total=len(candidates))
        for wav in candidates:
            sid = wav.stem
            progress.update(task, description=sid)
            level = _level(sid)
            if args.skip_level0 and level == 0:
                progress.advance(task)
                continue

            try:
                base_id = _baseline_id(sid)
                cand_feat = FEAT_DIR / f"{sid}_features.npz"
                base_feat = FEAT_DIR / f"{base_id}_features.npz"
                cand_align = ALIGN_DIR / f"{sid}_aligned.json"
                base_align = ALIGN_DIR / f"{base_id}_aligned.json"
                for p in (cand_feat, base_feat, cand_align, base_align):
                    if not p.exists():
                        raise FileNotFoundError(p)

                result = grounder.ground_files(
                    candidate_id=sid,
                    baseline_id=base_id,
                    cand_feat_path=cand_feat,
                    base_feat_path=base_feat,
                    cand_align_path=cand_align,
                    base_align_path=base_align,
                )
                out_path = OUT_DIR / f"{sid}_detected.json"
                payload = result.model_dump()
                tracks = payload.get("residual_summary", {}).pop("tracks", None)
                out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                if tracks:
                    resid_dir = OUT_DIR / "residuals"
                    resid_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        str(resid_dir / f"{sid}_residuals.npz"),
                        **{
                            k: np.asarray(v)
                            for k, v in tracks.items()
                        },
                    )

                gt_sample = gt_index.get(sid, {})
                gt_iv = gt_from_annotation(gt_sample)
                det_iv = det_from_grounding(payload)
                ev = evaluate_detection(
                    candidate_id=sid,
                    gt_intervals=gt_iv,
                    det_intervals=det_iv,
                    severity_level=int(gt_sample.get("severity_level", level)),
                )
                evals.append(ev)
                console.print(
                    f"  [green]OK[/green] {sid:<42} "
                    f"L{ev.severity_level}  gt={ev.n_gt} det={ev.n_det}  "
                    f"P={ev.precision:.2f} R={ev.recall:.2f} F1={ev.f1:.2f}  "
                    f"IoU={ev.mean_iou:.2f}"
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{sid}: {exc}")
                console.print(f"  [red]FAIL[/red] {sid}: {exc}")
            progress.advance(task)

    table = Table(title="Temporal grounding vs ground truth")
    table.add_column("candidate_id")
    table.add_column("L", justify="center")
    table.add_column("GT", justify="right")
    table.add_column("Det", justify="right")
    table.add_column("TP", justify="right")
    table.add_column("Prec", justify="right")
    table.add_column("Rec", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("IoU", justify="right")
    table.add_column("BndErr", justify="right")
    for ev in evals:
        table.add_row(
            ev.candidate_id,
            str(ev.severity_level),
            str(ev.n_gt),
            str(ev.n_det),
            str(ev.true_positives),
            f"{ev.precision:.2f}",
            f"{ev.recall:.2f}",
            f"{ev.f1:.2f}",
            f"{ev.mean_iou:.2f}",
            f"{ev.mean_boundary_error:.2f}",
        )
    console.print(table)

    overall = aggregate_metrics(evals)
    console.print(
        Panel.fit(
            f"[bold]Overall[/bold]\n"
            f"Precision = {overall['precision']:.3f}   "
            f"Recall = {overall['recall']:.3f}   "
            f"F1 = {overall['f1']:.3f}\n"
            f"Mean IoU = {overall['mean_iou']:.3f}   "
            f"Mean boundary error = {overall['mean_boundary_error']:.3f}s\n"
            f"TP={overall['true_positives']}  "
            f"FP={overall['false_positives']}  "
            f"FN={overall['false_negatives']}",
            border_style="green",
        )
    )

    summary = {
        "overall": overall,
        "per_sample": [e.model_dump() for e in evals],
    }
    summary_path = OUT_DIR / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    console.print(f"Summary -> {_rel(summary_path)}")

    if failures:
        console.print(f"[red]{len(failures)} failure(s)[/red]")
        for f in failures:
            console.print(f"  • {f}")
        raise SystemExit(1)
    console.print("[bold green]Temporal grounding complete.[/bold green]")


if __name__ == "__main__":
    main()
