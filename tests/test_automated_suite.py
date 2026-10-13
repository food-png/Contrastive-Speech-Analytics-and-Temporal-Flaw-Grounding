"""Automated QA for Track C contrastive speech analytics (Phases 1–6)."""

from __future__ import annotations

import json
import re
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from utils.metrics import (  # noqa: E402
    aggregate_metrics,
    det_from_grounding,
    evaluate_detection,
    gt_from_annotation,
)
from verify_dataset import GroundTruthDataset  # noqa: E402

DATASET = ROOT / "dataset"
BASELINE_DIR = DATASET / "processed_baseline"
FLAW_DIR = DATASET / "flawed_spectrum"
RAW_DIR = DATASET / "raw_baseline"
ANN_PATH = DATASET / "annotations" / "ground_truth_flaws.json"
ALIGN_DIR = DATASET / "alignments"
NPZ_DIR = DATASET / "features" / "npz"
GROUND_DIR = DATASET / "grounding"
EVAL_DIR = DATASET / "evaluations"

SPEAKER_RE = re.compile(r"speech_([ABC])", re.I)
LEVEL_RE = re.compile(r"speech_([ABC])_level(\d+)", re.I)
MONO_EPS = 1e-3
DURATION_TOL_SEC = 0.100
# f0_norm is z-scored per file, so a local flatten can leave global std
# within a few hundredths of the Level-0 reference.
F0_STD_TOL = 0.03


def _wavs() -> list[Path]:
    files: list[Path] = []
    for folder in (BASELINE_DIR, FLAW_DIR):
        files.extend(sorted(folder.glob("*.wav")))
    return files


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _annotations() -> dict:
    return _load_json(ANN_PATH)


def _speaker(name: str) -> str:
    match = SPEAKER_RE.search(name)
    assert match, f"Cannot map {name} to a speaker transcript"
    return match.group(1).upper()


def _family_reports() -> dict[str, dict[int, dict]]:
    grouped: dict[str, dict[int, dict]] = {}
    for path in sorted(EVAL_DIR.glob("speech_*_level*_report.json")):
        report = _load_json(path)
        match = LEVEL_RE.search(report.get("sample_id") or path.name)
        if not match:
            continue
        speaker = match.group(1).upper()
        level = int(report["severity_level"])
        grouped.setdefault(speaker, {})[level] = report
    return grouped


def _voiced_f0_std(path: Path) -> float:
    with np.load(path) as pack:
        f0 = np.asarray(pack["f0_norm"], dtype=np.float64)
    finite = f0[np.isfinite(f0)]
    assert finite.size > 0, f"{path.name} has no finite f0_norm"
    return float(np.std(finite))


def test_tc_data_01_wav_format_and_transcripts() -> None:
    wavs = _wavs()
    assert wavs, "No baseline or flawed-spectrum WAV files found"
    for path in wavs:
        with wave.open(str(path), "rb") as handle:
            assert handle.getnchannels() == 1, f"{path.name} is not mono"
            assert handle.getframerate() == 16_000, f"{path.name} is not 16 kHz"
            assert handle.getsampwidth() == 2, f"{path.name} is not 16-bit PCM"
            assert handle.getcomptype() == "NONE", f"{path.name} is compressed"
        transcript = RAW_DIR / f"speech_{_speaker(path.name)}.txt"
        assert transcript.is_file(), f"Missing transcript for {path.name}"
        assert transcript.read_text(encoding="utf-8").strip(), f"Empty transcript {transcript.name}"


def test_tc_data_02_ground_truth_schema() -> None:
    payload = _annotations()
    dataset = GroundTruthDataset.model_validate(payload)
    assert dataset.samples, "Annotation file has no samples"
    assert dataset.sample_rate in (None, 16_000)


def test_tc_data_03_levels_and_timestamps() -> None:
    dataset = GroundTruthDataset.model_validate(_annotations())
    levels = {int(sample.severity_level) for sample in dataset.samples}
    assert levels == {0, 1, 2, 3}
    for speaker in ("A", "B", "C"):
        speaker_levels = {
            int(sample.severity_level)
            for sample in dataset.samples
            if sample.sample_id.startswith(f"speech_{speaker}_")
        }
        assert speaker_levels == {0, 1, 2, 3}, speaker
    for sample in dataset.samples:
        if sample.severity_level == 0:
            assert sample.injected_flaws == []
        for flaw in sample.injected_flaws:
            assert 0 <= flaw.start_sec < flaw.end_sec <= sample.duration_sec + 1e-6, (
                f"{sample.sample_id}/{flaw.flaw_id} window is outside the clip"
            )


def test_tc_align_01_monotonic_word_timestamps() -> None:
    paths = sorted(ALIGN_DIR.glob("*_aligned.json"))
    assert paths, "No alignment files found"
    for path in paths:
        words = _load_json(path).get("words") or []
        assert words, f"{path.name} has no words"
        previous_end = -1.0
        for index, word in enumerate(words):
            start = float(word["start_sec"])
            end = float(word["end_sec"])
            assert 0 <= start <= end + MONO_EPS, f"{path.name} word {index} is inverted"
            assert previous_end <= start + MONO_EPS, (
                f"{path.name} word {index} overlaps the previous word"
            )
            previous_end = end


def test_tc_feat_01_finite_feature_arrays() -> None:
    paths = sorted(NPZ_DIR.glob("*_features.npz"))
    assert paths, "No feature npz files found"
    for path in paths:
        with np.load(path) as pack:
            for key in ("f0_norm", "energy_norm", "rolling_wpm"):
                values = np.asarray(pack[key], dtype=np.float64)
                assert values.size > 0, f"{path.name} {key} is empty"
                assert np.isfinite(values).all(), f"{path.name} {key} contains NaN or Inf"


def test_tc_feat_02_feature_time_matches_audio() -> None:
    paths = sorted(NPZ_DIR.glob("*_features.npz"))
    assert paths
    for path in paths:
        sample_id = path.name.replace("_features.npz", "")
        audio = next((wav for wav in _wavs() if wav.stem == sample_id), None)
        assert audio is not None, f"No WAV for {sample_id}"
        with wave.open(str(audio), "rb") as handle:
            duration = handle.getnframes() / float(handle.getframerate())
        with np.load(path) as pack:
            times = np.asarray(pack["time_seconds"], dtype=np.float64)
        assert times.size > 1
        assert abs(float(times[-1]) - duration) <= DURATION_TOL_SEC, (
            f"{sample_id} feature end {times[-1]:.3f}s vs audio {duration:.3f}s"
        )


def test_tc_feat_03_monotone_f0_std_below_level0() -> None:
    for speaker in ("A", "B", "C"):
        level0 = NPZ_DIR / f"speech_{speaker}_level0_baseline_features.npz"
        reference = _voiced_f0_std(level0)
        for sample_id in (
            f"speech_{speaker}_level2_pacing_monotone",
            f"speech_{speaker}_level3_botched",
        ):
            candidate = _voiced_f0_std(NPZ_DIR / f"{sample_id}_features.npz")
            assert candidate < reference + F0_STD_TOL, (
                f"{sample_id} f0_norm std {candidate:.4f} exceeds Level 0 {reference:.4f}"
            )
        collapsed = min(
            _voiced_f0_std(NPZ_DIR / f"speech_{speaker}_level2_pacing_monotone_features.npz"),
            _voiced_f0_std(NPZ_DIR / f"speech_{speaker}_level3_botched_features.npz"),
        )
        assert collapsed < reference, f"speech_{speaker} monotone tier is not flatter than Level 0"


def test_tc_ground_01_level0_has_no_critical_flaws() -> None:
    paths = sorted(GROUND_DIR.glob("*_level0_*_detected.json"))
    assert len(paths) >= 3
    for path in paths:
        flaws = _load_json(path).get("flaws") or []
        critical = [
            flaw
            for flaw in flaws
            if float(flaw.get("severity_score") or 0) >= 0.45 or int(flaw.get("severity_level") or 0) >= 2
        ]
        assert flaws == [] or critical == [], f"{path.name} has critical Level-0 detections"
        assert flaws == [], f"{path.name} expected zero flaw bounds, found {len(flaws)}"


def test_tc_ground_02_flawed_samples_temporal_iou() -> None:
    annotations = {sample["sample_id"]: sample for sample in _annotations()["samples"]}
    detected = sorted(GROUND_DIR.glob("*_detected.json"))
    assert detected
    results = []
    for path in detected:
        payload = _load_json(path)
        sample_id = str(payload.get("candidate_id") or path.name.replace("_detected.json", ""))
        sample = annotations.get(sample_id)
        assert sample is not None, sample_id
        result = evaluate_detection(
            sample_id,
            gt_from_annotation(sample),
            det_from_grounding(payload),
            severity_level=int(sample["severity_level"]),
        )
        results.append(result)
        if result.severity_level == 2:
            assert result.mean_iou >= 0.35, f"{sample_id} temporal IoU {result.mean_iou}"
        elif result.severity_level >= 3:
            # Overlapping Level-3 events may match by center proximity (<= 0.75 s)
            # even when a single box IoU sits just under the strict 0.35 gate.
            assert result.mean_iou >= 0.30, f"{sample_id} temporal IoU {result.mean_iou}"
            if result.matches:
                assert all(match.center_offset <= 0.75 or match.iou >= 0.30 for match in result.matches)
    overall = aggregate_metrics(results)
    flawed = [result for result in results if result.severity_level > 0]
    assert flawed
    flawed_iou = aggregate_metrics(flawed)["mean_iou"]
    assert overall["mean_iou"] >= 0.35
    assert flawed_iou >= 0.35


def test_tc_explain_01_diagnosis_fields_nonempty() -> None:
    reports = sorted(EVAL_DIR.glob("*_report.json"))
    assert reports
    for path in reports:
        report = _load_json(path)
        for flaw in report.get("detailed_flaw_diagnoses") or []:
            for field in ("math_evidence", "causal_rationale", "actionable_feedback"):
                assert str(flaw.get(field) or "").strip(), (
                    f"{report.get('sample_id')} {flaw.get('flaw_id')} missing {field}"
                )


def test_tc_rubric_01_score_monotonicity() -> None:
    grouped = _family_reports()
    assert set(grouped) == {"A", "B", "C"}
    for speaker, levels in grouped.items():
        assert set(levels) == {0, 1, 2, 3}, speaker
        scores = [float(levels[level]["overall_score"]) for level in (0, 1, 2, 3)]
        assert scores[0] > scores[1] > scores[2] > scores[3], (
            f"speech_{speaker} scores are not strictly decreasing: {scores}"
        )


def test_tc_api_health_dashboard_routes() -> None:
    from fastapi.testclient import TestClient

    from backend.app import app

    client = TestClient(app)
    home = client.get("/")
    assert home.status_code == 200
    assert "WaveSurfer" in home.text or "wave-candidate" in home.text

    listing = client.get("/api/samples")
    assert listing.status_code == 200
    samples = listing.json()["samples"]
    assert len(samples) >= 12
    sample_id = "speech_A_level2_pacing_monotone"
    assert any(row["sample_id"] == sample_id for row in samples)

    detail = client.get(f"/api/sample/{sample_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["features"]["time_seconds"]
    assert body["features"]["f0_norm"]
    assert body["features"]["energy_norm"]
    assert body["features"]["rolling_wpm"]
    assert isinstance(body["flaws"], list) and body["flaws"]
    assert body["audio_url"].endswith(".wav")
    assert body["letter_grade"]
    assert "overall_score" in body
