#!/usr/bin/env python3
"""
Phase 2 — Alignment QA.

1. Every WAV in processed_baseline/ and flawed_spectrum/ has a matching JSON.
2. Word timestamps are monotonically non-decreasing.
3. Timestamps stay within audio duration (+100 ms).
4. Summary table: words, pauses, mean word duration, max pause.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.aligner import AlignmentResult, PAUSE_MIN_SEC  # noqa: E402

console = Console()

AUDIO_DIRS = [
    ROOT / "dataset" / "processed_baseline",
    ROOT / "dataset" / "flawed_spectrum",
]
ALIGN_DIR = ROOT / "dataset" / "alignments"
TIME_TOL_SEC = 0.100


def _audio_duration(path: Path) -> float:
    info = sf.info(str(path))
    if info.samplerate <= 0:
        return 0.0
    return float(info.frames / info.samplerate)


def _check_monotonic(result: AlignmentResult) -> list[str]:
    errors: list[str] = []
    words = result.words
    for i, w in enumerate(words):
        if w.start_sec > w.end_sec + 1e-9:
            errors.append(
                f"{result.sample_id} word[{i}] {w.word!r}: "
                f"start {w.start_sec} > end {w.end_sec}"
            )
        if i + 1 < len(words):
            nxt = words[i + 1]
            if w.end_sec - nxt.start_sec > 1e-4:
                errors.append(
                    f"{result.sample_id} word[{i}] end {w.end_sec} > "
                    f"word[{i+1}] start {nxt.start_sec}"
                )
    return errors


def _check_duration(result: AlignmentResult, wav: Path) -> list[str]:
    errors: list[str] = []
    dur = _audio_duration(wav)
    limit = dur + TIME_TOL_SEC
    for i, w in enumerate(result.words):
        if w.end_sec > limit:
            errors.append(
                f"{result.sample_id} word[{i}] end {w.end_sec:.3f} > duration {dur:.3f}"
            )
        if w.start_sec < -1e-6:
            errors.append(f"{result.sample_id} word[{i}] negative start")
    for i, p in enumerate(result.pauses):
        if p.end_sec > limit:
            errors.append(
                f"{result.sample_id} pause[{i}] end {p.end_sec:.3f} > duration {dur:.3f}"
            )
        if p.duration_sec + 1e-6 < PAUSE_MIN_SEC:
            errors.append(
                f"{result.sample_id} pause[{i}] duration {p.duration_sec:.3f} < {PAUSE_MIN_SEC}"
            )
    return errors


def verify() -> int:
    console.print(
        Panel.fit(
            "[bold]Track C | Phase 2[/bold]\nAlignment verification",
            border_style="green",
        )
    )

    wavs = []
    for d in AUDIO_DIRS:
        if d.is_dir():
            wavs.extend(sorted(d.glob("*.wav")))

    if not wavs:
        console.print("[red]No WAV files found in dataset audio folders.[/red]")
        return 1

    if not ALIGN_DIR.is_dir():
        console.print(f"[red]Missing alignments directory:[/red] {ALIGN_DIR}")
        return 1

    errors: list[str] = []
    rows: list[tuple[str, int, int, float, float]] = []

    json_stems = {p.name.replace("_aligned.json", "") for p in ALIGN_DIR.glob("*_aligned.json")}

    for wav in wavs:
        sid = wav.stem
        json_path = ALIGN_DIR / f"{sid}_aligned.json"
        if not json_path.exists():
            errors.append(f"MISSING alignment JSON for {sid}")
            continue
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
            result = AlignmentResult.model_validate(raw)
        except (json.JSONDecodeError, ValidationError) as exc:
            errors.append(f"{sid}: invalid JSON/schema ({exc})")
            continue

        errors.extend(_check_monotonic(result))
        errors.extend(_check_duration(result, wav))

        n_words = len(result.words)
        n_pauses = len(result.pauses)
        if n_words:
            avg_dur = sum(w.end_sec - w.start_sec for w in result.words) / n_words
        else:
            avg_dur = 0.0
            errors.append(f"{sid}: zero words aligned")
        max_pause = max((p.duration_sec for p in result.pauses), default=0.0)
        rows.append((sid, n_words, n_pauses, avg_dur, max_pause))

    orphan_json = json_stems - {w.stem for w in wavs}
    if orphan_json:
        for stem in sorted(orphan_json):
            errors.append(f"ORPHAN alignment JSON (no matching wav): {stem}")

    table = Table(title="Forced alignment summary")
    table.add_column("sample_id")
    table.add_column("words", justify="right")
    table.add_column("pauses", justify="right")
    table.add_column("avg word dur (s)", justify="right")
    table.add_column("max pause (s)", justify="right")
    for sid, n_w, n_p, avg_d, max_p in rows:
        table.add_row(sid, str(n_w), str(n_p), f"{avg_d:.3f}", f"{max_p:.3f}")
    console.print(table)

    if errors:
        console.print("\n[red bold]Errors[/red bold]")
        for e in errors:
            console.print(f"  • {e}")
        console.print(f"\n[red]FAILED[/red] - {len(errors)} error(s)")
        return 1

    console.print("\n[bold green]PASSED[/bold green] - alignments are monotonic and in-range.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify forced-alignment JSON files.")
    parser.parse_args()
    raise SystemExit(verify())


if __name__ == "__main__":
    main()
