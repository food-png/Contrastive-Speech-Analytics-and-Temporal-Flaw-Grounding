"""On-the-fly align → features → grounding → causal score for an uploaded WAV."""

from __future__ import annotations

import json
from pathlib import Path

from utils.aligner import ForcedAligner
from utils.causal_explainer import CausalExplainer
from utils.feature_extractor import AcousticFeatureExtractor
from utils.rubric_scorer import RubricScorer
from utils.temporal_grounding import TemporalFlawGrounder

from backend.catalog import (
    BASE_DIR,
    DATASET,
    EVAL_DIR,
    FEAT_JSON,
    ROOT,
    UPLOAD_DIR,
    assert_sample_id,
    sample_payload,
)

RAW_DIR = DATASET / "raw_baseline"
ALIGN_DIR = DATASET / "alignments"
FEAT_NPZ = DATASET / "features" / "npz"


def baseline_transcript(baseline_id: str) -> str:
    assert_sample_id(baseline_id)
    letter = baseline_id.split("_")[1] if "_" in baseline_id else ""
    path = RAW_DIR / f"speech_{letter}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"No transcript for baseline {baseline_id}")
    return path.read_text(encoding="utf-8")


def evaluate_upload(
    wav_bytes: bytes,
    baseline_id: str,
    transcript: str | None = None,
    sample_id: str | None = None,
) -> dict:
    baseline_id = assert_sample_id(baseline_id)
    base_wav = BASE_DIR / f"{baseline_id}.wav"
    base_align = ALIGN_DIR / f"{baseline_id}_aligned.json"
    base_npz = FEAT_NPZ / f"{baseline_id}_features.npz"
    if not base_wav.is_file() or not base_align.is_file() or not base_npz.is_file():
        raise FileNotFoundError(
            f"Baseline {baseline_id} is missing audio, alignment, or features."
        )

    text = (transcript or "").strip() or baseline_transcript(baseline_id)
    sid = assert_sample_id(sample_id or "upload_live")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ALIGN_DIR.mkdir(parents=True, exist_ok=True)
    FEAT_NPZ.mkdir(parents=True, exist_ok=True)
    FEAT_JSON.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    wav_path = UPLOAD_DIR / f"{sid}.wav"
    wav_path.write_bytes(wav_bytes)

    aligner = ForcedAligner()
    alignment = aligner.align_file(
        wav_path,
        text,
        sample_id=sid,
        rel_audio=str(wav_path.relative_to(ROOT)).replace("\\", "/"),
    )
    align_path = ALIGN_DIR / f"{sid}_aligned.json"
    align_path.write_text(
        json.dumps(alignment.model_dump(), indent=2) + "\n",
        encoding="utf-8",
    )

    extractor = AcousticFeatureExtractor()
    series = extractor.extract_features(wav_path, alignment_path=align_path)
    series.metadata["sample_id"] = sid
    npz_path = FEAT_NPZ / f"{sid}_features.npz"
    json_path = FEAT_JSON / f"{sid}_features.json"
    series.save_npz(npz_path)
    series.save_json(json_path)

    grounder = TemporalFlawGrounder()
    grounding = grounder.ground_files(
        candidate_id=sid,
        baseline_id=baseline_id,
        cand_feat_path=npz_path,
        base_feat_path=base_npz,
        cand_align_path=align_path,
        base_align_path=base_align,
    )
    detection = grounding.model_dump()

    explainer = CausalExplainer(prefer_llm=False)
    scorer = RubricScorer(explainer=explainer)
    diagnoses = explainer.explain_detection(
        detection,
        alignment_path=align_path,
        use_llm=False,
    )
    report = scorer.score(
        detection,
        diagnoses=diagnoses,
        alignment_path=str(align_path),
        severity_level=-1,
        use_llm=False,
    )
    report_path = EVAL_DIR / f"{sid}_report.json"
    report_path.write_text(
        json.dumps(report.model_dump(), indent=2) + "\n",
        encoding="utf-8",
    )
    return sample_payload(sid)
