"""Load precomputed samples and build dashboard payloads."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
EVAL_DIR = DATASET / "evaluations"
FEAT_JSON = DATASET / "features" / "json"
FLAW_DIR = DATASET / "flawed_spectrum"
BASE_DIR = DATASET / "processed_baseline"
UPLOAD_DIR = DATASET / "uploads"

SAFE_ID = re.compile(r"^[A-Za-z0-9_\-]+$")
MAX_CURVE_POINTS = 480


def assert_sample_id(sample_id: str) -> str:
    if not SAFE_ID.match(sample_id):
        raise ValueError(f"Invalid sample id: {sample_id}")
    return sample_id


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _decimate(values: list[Any], step: int) -> list[Any]:
    if step <= 1:
        return list(values)
    return list(values[::step])


def curve_payload(path: Path) -> dict[str, list[Any]]:
    """Return chart-sized F0 / energy / WPM curves."""
    data = _load_json(path)
    times = list(data.get("time_seconds") or [])
    n = len(times)
    step = max(1, n // MAX_CURVE_POINTS) if n else 1
    keys = ("time_seconds", "f0_norm", "energy_norm", "rolling_wpm")
    out: dict[str, list[Any]] = {}
    for key in keys:
        series = data.get(key) or []
        if len(series) != n and key != "time_seconds":
            series = list(series)[:n] + [None] * max(0, n - len(series))
        out[key] = _decimate(series, step)
    return out


def resolve_audio(sample_id: str) -> Path | None:
    assert_sample_id(sample_id)
    for folder in (FLAW_DIR, BASE_DIR, UPLOAD_DIR):
        path = folder / f"{sample_id}.wav"
        if path.is_file():
            return path
    return None


def audio_url(path: Path) -> str:
    rel = path.relative_to(DATASET).as_posix()
    return f"/dataset/{rel}"


def list_samples() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(EVAL_DIR.glob("*_report.json")):
        report = _load_json(path)
        sid = str(report.get("sample_id") or path.name.replace("_report.json", ""))
        diagnoses = report.get("detailed_flaw_diagnoses") or []
        counts = report.get("flaw_count_by_severity") or {}
        flaw_count = len(diagnoses) if diagnoses else sum(int(v) for v in counts.values())
        rows.append(
            {
                "sample_id": sid,
                "level": int(report.get("severity_level") or 0),
                "overall_score": float(report.get("overall_score") or 0),
                "letter_grade": str(report.get("letter_grade") or ""),
                "flaw_count": int(flaw_count),
                "baseline_id": str(report.get("baseline_id") or ""),
            }
        )
    rows.sort(key=lambda r: (r["sample_id"][:8], r["level"], r["sample_id"]))
    return rows


def sample_payload(sample_id: str) -> dict[str, Any]:
    sid = assert_sample_id(sample_id)
    report_path = EVAL_DIR / f"{sid}_report.json"
    feat_path = FEAT_JSON / f"{sid}_features.json"
    audio = resolve_audio(sid)
    if not report_path.is_file():
        raise FileNotFoundError(f"No evaluation report for {sid}")
    if not feat_path.is_file():
        raise FileNotFoundError(f"No feature curves for {sid}")
    if audio is None:
        raise FileNotFoundError(f"No audio for {sid}")

    report = _load_json(report_path)
    baseline_id = str(report.get("baseline_id") or "")
    if not baseline_id:
        m = re.match(r"(speech_[A-C])_", sid)
        baseline_id = f"{m.group(1)}_baseline" if m else ""

    baseline_audio = resolve_audio(baseline_id) if baseline_id else None
    base_feat = FEAT_JSON / f"{baseline_id}_features.json" if baseline_id else None

    return {
        "sample_id": sid,
        "baseline_id": baseline_id,
        "severity_level": int(report.get("severity_level") or 0),
        "overall_score": float(report.get("overall_score") or 0),
        "letter_grade": str(report.get("letter_grade") or ""),
        "dimension_scores": report.get("dimension_scores") or {},
        "flaw_count_by_severity": report.get("flaw_count_by_severity") or {},
        "executive_summary": str(report.get("executive_summary") or ""),
        "primary_causal_flag": str(report.get("primary_causal_flag") or "none"),
        "audio_url": audio_url(audio),
        "baseline_audio_url": audio_url(baseline_audio) if baseline_audio else None,
        "features": curve_payload(feat_path),
        "baseline_features": curve_payload(base_feat) if base_feat and base_feat.is_file() else None,
        "flaws": report.get("detailed_flaw_diagnoses") or [],
    }
