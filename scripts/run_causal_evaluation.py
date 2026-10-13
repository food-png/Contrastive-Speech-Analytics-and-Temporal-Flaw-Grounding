#!/usr/bin/env python3
"""
Phase 5 — Batch causal explanation + rubric scoring.

Reads dataset/grounding/*_detected.json and writes
dataset/evaluations/<sample_id>_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.env_loader import load_project_env  # noqa: E402
from utils.causal_explainer import CausalExplainer  # noqa: E402
from utils.rubric_scorer import RubricScorer  # noqa: E402

load_project_env()

console = Console()

GROUND_DIR = ROOT / "dataset" / "grounding"
ALIGN_DIR = ROOT / "dataset" / "alignments"
EVAL_DIR = ROOT / "dataset" / "evaluations"
ANN_PATH = ROOT / "dataset" / "annotations" / "ground_truth_flaws.json"
LEVEL_RE = re.compile(r"level(\d+)", re.I)


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _level(sample_id: str, ann_index: dict) -> int:
    if sample_id in ann_index:
        return int(ann_index[sample_id].get("severity_level", 0))
    m = LEVEL_RE.search(sample_id)
    return int(m.group(1)) if m else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal explanation & rubric scoring.")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Attempt Gemini/LLM enrichment when GEMINI_API_KEY is set.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Force deterministic templates only (default unless --llm).",
    )
    args = parser.parse_args()
    use_llm = bool(args.llm) and not bool(args.no_llm)

    console.print(
        Panel.fit(
            "[bold]Track C | Phase 5[/bold]\n"
            f"Causal explainability & rubric scoring  |  llm={'on' if use_llm else 'off'}",
            border_style="cyan",
        )
    )

    det_files = sorted(GROUND_DIR.glob("*_detected.json"))
    if not det_files:
        console.print("[red]No grounding detections found. Run Phase 4 first.[/red]")
        raise SystemExit(1)

    ann_index: dict = {}
    if ANN_PATH.exists():
        data = json.loads(ANN_PATH.read_text(encoding="utf-8"))
        ann_index = {s["sample_id"]: s for s in data.get("samples", [])}

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    explainer = CausalExplainer(prefer_llm=use_llm)
    scorer = RubricScorer(explainer=explainer)

    if use_llm and not (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("LLM_ENDPOINT")
    ):
        console.print(
            "[yellow]--llm set but no GEMINI_API_KEY/LLM_ENDPOINT found; "
            "using deterministic templates.[/yellow]"
        )

    rows: list[dict] = []
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
        task = progress.add_task("evaluating", total=len(det_files))
        for path in det_files:
            sid = path.name.replace("_detected.json", "")
            progress.update(task, description=sid)
            try:
                detection = json.loads(path.read_text(encoding="utf-8"))
                align_path = ALIGN_DIR / f"{sid}_aligned.json"
                level = _level(sid, ann_index)
                diagnoses = explainer.explain_detection(
                    detection,
                    alignment_path=align_path if align_path.exists() else None,
                    use_llm=use_llm,
                )
                report = scorer.score(
                    detection,
                    diagnoses=diagnoses,
                    alignment_path=str(align_path) if align_path.exists() else None,
                    severity_level=level,
                    use_llm=False,  # already explained
                )
                out = EVAL_DIR / f"{sid}_report.json"
                out.write_text(
                    json.dumps(report.model_dump(), indent=2) + "\n",
                    encoding="utf-8",
                )
                dims = report.dimension_scores
                rows.append(
                    {
                        "sample_id": sid,
                        "level": level,
                        "pacing": dims["pacing_score"],
                        "dynamics": dims["dynamic_energy_score"],
                        "variety": dims["tonal_variety_score"],
                        "pauses": dims["pause_discipline_score"],
                        "overall": report.overall_score,
                        "grade": report.letter_grade,
                        "flag": report.primary_causal_flag,
                    }
                )
                console.print(
                    f"  [green]OK[/green] {sid:<42} "
                    f"L{level}  {report.overall_score:5.1f} ({report.letter_grade})  "
                    f"flag={report.primary_causal_flag}"
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{sid}: {exc}")
                console.print(f"  [red]FAIL[/red] {sid}: {exc}")
            progress.advance(task)

    table = Table(title="Delivery evaluation scorecards")
    table.add_column("sample_id")
    table.add_column("L", justify="center")
    table.add_column("Pacing", justify="right")
    table.add_column("Dynamics", justify="right")
    table.add_column("Variety", justify="right")
    table.add_column("Pauses", justify="right")
    table.add_column("Overall", justify="right")
    table.add_column("Grade", justify="center")
    table.add_column("Primary Causal Flag")
    for r in rows:
        table.add_row(
            r["sample_id"],
            str(r["level"]),
            f"{r['pacing']:.1f}",
            f"{r['dynamics']:.1f}",
            f"{r['variety']:.1f}",
            f"{r['pauses']:.1f}",
            f"{r['overall']:.1f}",
            r["grade"],
            r["flag"],
        )
    console.print(table)
    console.print(f"Reports -> {_rel(EVAL_DIR)}")

    if failures:
        console.print(f"[red]{len(failures)} failure(s)[/red]")
        for f in failures:
            console.print(f"  • {f}")
        raise SystemExit(1)
    console.print("[bold green]Causal evaluation complete.[/bold green]")


if __name__ == "__main__":
    main()
