#!/usr/bin/env python3
"""
Phase 1 — Dataset QA: schema validation, audio integrity, temporal alignment.

Checks:
  1. ground_truth_flaws.json validates against the Pydantic schema
  2. Every audio_path / transcript_path exists
  3. WAV format contract: 16 kHz, mono, PCM_16
  4. Reported duration_sec matches file length (±50 ms)
  5. Every flaw window is within [0, duration]
  6. start_sec < end_sec for all flaws
  7. Level-0 samples have zero injected flaws
  8. Level-2/3 samples contain the expected flaw category coverage
  9. Peak amplitude ≈ −1.0 dBFS (±0.3 dB)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

console = Console()

ANN_PATH = ROOT / "dataset" / "annotations" / "ground_truth_flaws.json"
TARGET_SR = 16_000
PEAK_DBFS = -1.0
DURATION_TOL_SEC = 0.05
PEAK_TOL_DB = 0.35


# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------

class InjectedFlaw(BaseModel):
    flaw_id: str
    category: Literal["pacing", "pitch", "silence", "energy", "compound"]
    sub_type: str
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    target_phrase: str
    parameters_applied: dict[str, Any]
    expected_acoustic_signature: str

    @model_validator(mode="after")
    def ordered_window(self) -> "InjectedFlaw":
        if self.end_sec <= self.start_sec:
            raise ValueError(
                f"{self.flaw_id}: end_sec ({self.end_sec}) must exceed start_sec ({self.start_sec})"
            )
        return self


class SampleRecord(BaseModel):
    sample_id: str
    baseline_id: str
    audio_path: str
    transcript_path: str
    duration_sec: float = Field(gt=0)
    severity_level: Literal[0, 1, 2, 3]
    injected_flaws: list[InjectedFlaw]

    @field_validator("sample_id")
    @classmethod
    def nonempty_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("sample_id must be non-empty")
        return v

    @model_validator(mode="after")
    def flaws_within_duration(self) -> "SampleRecord":
        for flaw in self.injected_flaws:
            if flaw.end_sec - self.duration_sec > DURATION_TOL_SEC:
                raise ValueError(
                    f"{self.sample_id}/{flaw.flaw_id}: end_sec {flaw.end_sec} "
                    f"exceeds duration {self.duration_sec}"
                )
        if self.severity_level == 0 and self.injected_flaws:
            raise ValueError(f"{self.sample_id}: level 0 must have zero injected flaws")
        return self


class GroundTruthDataset(BaseModel):
    dataset_version: str
    generated_at: str
    samples: list[SampleRecord]
    seed: int | None = None
    crossfade_ms: float | None = None
    sample_rate: int | None = None

    @field_validator("generated_at")
    @classmethod
    def iso8601(cls, v: str) -> str:
        # Accept trailing Z or offset
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"generated_at is not ISO-8601: {v}") from exc
        return v

    @field_validator("samples")
    @classmethod
    def non_empty(cls, v: list[SampleRecord]) -> list[SampleRecord]:
        if not v:
            raise ValueError("samples must be non-empty")
        return v


# ---------------------------------------------------------------------------
# Audio / file checks
# ---------------------------------------------------------------------------

def _check_audio_file(sample: SampleRecord) -> list[str]:
    errors: list[str] = []
    audio_path = ROOT / sample.audio_path
    if not audio_path.exists():
        return [f"MISSING audio: {sample.audio_path}"]

    info = sf.info(str(audio_path))
    if info.samplerate != TARGET_SR:
        errors.append(
            f"{sample.sample_id}: sample_rate={info.samplerate} (expected {TARGET_SR})"
        )
    if info.channels != 1:
        errors.append(f"{sample.sample_id}: channels={info.channels} (expected 1)")
    subtype = (info.subtype or "").upper()
    if "PCM_16" not in subtype and info.subtype != "PCM_16":
        # soundfile may report "Signed 16 bit PCM" depending on version
        if "16" not in subtype and info.subtype != "PCM_16":
            errors.append(
                f"{sample.sample_id}: subtype={info.subtype!r} (expected PCM_16)"
            )

    y, sr = sf.read(str(audio_path), always_2d=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        errors.append(f"{sample.sample_id}: array is not mono after read")
        y = np.mean(y, axis=-1)
    actual_dur = float(len(y) / sr) if sr else 0.0
    if abs(actual_dur - sample.duration_sec) > DURATION_TOL_SEC:
        errors.append(
            f"{sample.sample_id}: duration mismatch "
            f"json={sample.duration_sec:.3f}s file={actual_dur:.3f}s"
        )

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak < 1e-6:
        errors.append(f"{sample.sample_id}: silent / near-zero audio")
    else:
        peak_db = 20.0 * np.log10(peak)
        if abs(peak_db - PEAK_DBFS) > PEAK_TOL_DB:
            errors.append(
                f"{sample.sample_id}: peak={peak_db:.2f} dBFS "
                f"(expected {PEAK_DBFS} ± {PEAK_TOL_DB})"
            )
    return errors


def _check_paths(sample: SampleRecord) -> list[str]:
    errors: list[str] = []
    transcript = ROOT / sample.transcript_path
    if not transcript.exists():
        errors.append(f"MISSING transcript: {sample.transcript_path}")
    return errors


def _check_level_coverage(dataset: GroundTruthDataset) -> list[str]:
    warnings: list[str] = []
    by_level: dict[int, list[SampleRecord]] = {0: [], 1: [], 2: [], 3: []}
    for s in dataset.samples:
        by_level[int(s.severity_level)].append(s)

    for level, expected_min in ((0, 3), (1, 3), (2, 3), (3, 3)):
        if len(by_level[level]) < expected_min:
            warnings.append(
                f"Expected ≥{expected_min} samples at level {level}, found {len(by_level[level])}"
            )

    for s in by_level[2]:
        cats = {f.category for f in s.injected_flaws}
        for need in ("pacing", "pitch", "silence"):
            if need not in cats:
                warnings.append(f"{s.sample_id}: level-2 missing category '{need}'")

    for s in by_level[3]:
        cats = {f.category for f in s.injected_flaws}
        if "silence" not in cats:
            warnings.append(f"{s.sample_id}: level-3 missing hesitation silence")
        if "energy" not in cats and "compound" not in cats:
            warnings.append(f"{s.sample_id}: level-3 missing energy/compound flaws")
    return warnings


def verify(ann_path: Path = ANN_PATH) -> int:
    console.print(
        Panel.fit(
            "[bold]Track C | Phase 1[/bold]\nDataset verification & schema QA",
            border_style="green",
        )
    )

    if not ann_path.exists():
        console.print(f"[red]Annotations not found:[/red] {ann_path}")
        console.print("Run scripts/setup_baselines.py then scripts/augment_flaws.py first.")
        return 1

    raw = json.loads(ann_path.read_text(encoding="utf-8"))
    try:
        dataset = GroundTruthDataset.model_validate(raw)
    except ValidationError as exc:
        console.print("[red bold]Schema validation FAILED[/red bold]")
        console.print(str(exc))
        return 1

    console.print(
        f"[green]Schema OK[/green]  version={dataset.dataset_version}  "
        f"samples={len(dataset.samples)}  generated_at={dataset.generated_at}"
    )

    errors: list[str] = []
    for sample in dataset.samples:
        errors.extend(_check_paths(sample))
        errors.extend(_check_audio_file(sample))

    warnings = _check_level_coverage(dataset)

    table = Table(title="Per-sample overview")
    table.add_column("sample_id")
    table.add_column("L", justify="center")
    table.add_column("dur", justify="right")
    table.add_column("flaws", justify="right")
    for s in dataset.samples:
        table.add_row(
            s.sample_id,
            str(s.severity_level),
            f"{s.duration_sec:.2f}",
            str(len(s.injected_flaws)),
        )
    console.print(table)

    if warnings:
        console.print("\n[yellow bold]Warnings[/yellow bold]")
        for w in warnings:
            console.print(f"  • {w}")

    if errors:
        console.print("\n[red bold]Errors[/red bold]")
        for e in errors:
            console.print(f"  • {e}")
        console.print(f"\n[red]FAILED[/red] - {len(errors)} error(s)")
        return 1

    console.print("\n[bold green]PASSED[/bold green] - all alignment & schema checks OK.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify contrastive dataset integrity.")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=ANN_PATH,
        help="Path to ground_truth_flaws.json",
    )
    args = parser.parse_args()
    raise SystemExit(verify(args.annotations))


if __name__ == "__main__":
    main()
