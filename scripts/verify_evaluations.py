#!/usr/bin/env python3
"""
Phase 5 — Evaluation QA.

1. Every grounding detection has a matching evaluation report.
2. Within each speech key (A/B/C): Score(L0) > Score(L1) > Score(L2) > Score(L3).
3. Every diagnosed flaw has non-empty math_evidence, causal_rationale, impact,
   actionable_feedback.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

console = Console()

GROUND_DIR = ROOT / "dataset" / "grounding"
EVAL_DIR = ROOT / "dataset" / "evaluations"

REQUIRED_FIELDS = (
    "math_evidence",
    "causal_rationale",
    "impact",
    "actionable_feedback",
)
LEVEL_RE = re.compile(r"speech_([ABC])_level(\d+)", re.I)


def verify() -> int:
    console.print(
        Panel.fit(
            "[bold]Track C | Phase 5[/bold]\nEvaluation verification",
            border_style="green",
        )
    )

    det_files = sorted(GROUND_DIR.glob("*_detected.json"))
    if not det_files:
        console.print("[red]No grounding detections found.[/red]")
        return 1

    errors: list[str] = []
    reports: dict[str, dict] = {}

    for det in det_files:
        sid = det.name.replace("_detected.json", "")
        report_path = EVAL_DIR / f"{sid}_report.json"
        if not report_path.exists():
            errors.append(f"MISSING report for {sid}")
            continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"{sid}: invalid JSON ({exc})")
            continue
        reports[sid] = report

        score = float(report.get("overall_score", -1))
        if not (1.0 <= score <= 100.0):
            errors.append(f"{sid}: overall_score {score} out of [1, 100]")

        for i, d in enumerate(report.get("detailed_flaw_diagnoses", [])):
            for field in REQUIRED_FIELDS:
                val = d.get(field)
                if val is None or (isinstance(val, str) and not val.strip()):
                    errors.append(f"{sid} diagnosis[{i}] empty '{field}'")

        # Level-0 should have empty diagnoses
        m = LEVEL_RE.match(sid)
        if m and int(m.group(2)) == 0 and report.get("detailed_flaw_diagnoses"):
            # Allow empty only; if detections were empty diagnoses must be empty
            det_data = json.loads(det.read_text(encoding="utf-8"))
            if not det_data.get("flaws") and report.get("detailed_flaw_diagnoses"):
                errors.append(f"{sid}: Level-0 report has unexpected diagnoses")

    # Monotonicity per speech letter
    by_key: dict[str, dict[int, float]] = {"A": {}, "B": {}, "C": {}}
    for sid, report in reports.items():
        m = LEVEL_RE.match(sid)
        if not m:
            continue
        key = m.group(1).upper()
        level = int(m.group(2))
        by_key[key][level] = float(report["overall_score"])

    for key, levels in by_key.items():
        if not levels:
            continue
        ordered = [levels[i] for i in sorted(levels)]
        idxs = sorted(levels)
        for a, b in zip(idxs, idxs[1:]):
            if levels[a] <= levels[b]:
                errors.append(
                    f"speech_{key}: Score(L{a})={levels[a]:.2f} "
                    f"is not > Score(L{b})={levels[b]:.2f}"
                )
        # Soft band checks
        if 0 in levels and not (92 <= levels[0] <= 98):
            errors.append(f"speech_{key} L0 score {levels[0]:.2f} outside 92–98")
        if 1 in levels and not (80 <= levels[1] <= 89):
            errors.append(f"speech_{key} L1 score {levels[1]:.2f} outside 80–89")
        if 2 in levels and not (65 <= levels[2] <= 79):
            errors.append(f"speech_{key} L2 score {levels[2]:.2f} outside 65–79")
        if 3 in levels and not (levels[3] < 55):
            errors.append(f"speech_{key} L3 score {levels[3]:.2f} not < 55")
        _ = ordered

    table = Table(title="Evaluation scores by gradient")
    table.add_column("speech")
    table.add_column("L0", justify="right")
    table.add_column("L1", justify="right")
    table.add_column("L2", justify="right")
    table.add_column("L3", justify="right")
    for key in ("A", "B", "C"):
        levels = by_key[key]
        table.add_row(
            f"speech_{key}",
            f"{levels.get(0, float('nan')):.1f}" if 0 in levels else "—",
            f"{levels.get(1, float('nan')):.1f}" if 1 in levels else "—",
            f"{levels.get(2, float('nan')):.1f}" if 2 in levels else "—",
            f"{levels.get(3, float('nan')):.1f}" if 3 in levels else "—",
        )
    console.print(table)

    if errors:
        console.print("\n[red bold]Errors[/red bold]")
        for e in errors:
            console.print(f"  • {e}")
        console.print(f"\n[red]FAILED[/red] - {len(errors)} error(s)")
        return 1

    console.print(
        "\n[bold green]PASSED[/bold green] - reports complete, fields filled, "
        "scores strictly decrease across levels."
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify causal evaluation reports.")
    parser.parse_args()
    raise SystemExit(verify())


if __name__ == "__main__":
    main()
