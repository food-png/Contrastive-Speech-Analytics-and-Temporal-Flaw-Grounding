"""Shared utilities for Track C contrastive speech dataset engineering."""

from .feature_extractor import AcousticFeatureExtractor, AcousticTimeSeries
from .metrics import EvalResult, aggregate_metrics, evaluate_detection, temporal_iou
from .temporal_grounding import DetectedFlaw, GroundingResult, TemporalFlawGrounder
from .causal_explainer import CausalDiagnosis, CausalExplainer
from .rubric_scorer import DeliveryEvaluationReport, RubricScorer
from .aligner import (
    AlignmentResult,
    ForcedAligner,
    PauseSegment,
    WordSegment,
    align_audio,
    normalize_transcript,
)
from .audio_ops import (
    TIMESTAMP_SHIFT_EPS,
    TimestampTracker,
    apply_gain_db,
    cosine_crossfade,
    insert_silence,
    linear_crossfade,
    load_audio,
    peak_normalize,
    pitch_flatten,
    replace_segment,
    resample,
    splice_with_crossfade,
    time_stretch_segment,
    to_mono,
    write_wav_pcm16,
)

__all__ = [
    "AcousticFeatureExtractor",
    "AcousticTimeSeries",
    "CausalDiagnosis",
    "CausalExplainer",
    "DeliveryEvaluationReport",
    "DetectedFlaw",
    "EvalResult",
    "GroundingResult",
    "RubricScorer",
    "TemporalFlawGrounder",
    "aggregate_metrics",
    "evaluate_detection",
    "temporal_iou",
    "AlignmentResult",
    "ForcedAligner",
    "PauseSegment",
    "WordSegment",
    "align_audio",
    "normalize_transcript",
    "TIMESTAMP_SHIFT_EPS",
    "TimestampTracker",
    "apply_gain_db",
    "cosine_crossfade",
    "insert_silence",
    "linear_crossfade",
    "load_audio",
    "peak_normalize",
    "pitch_flatten",
    "replace_segment",
    "resample",
    "splice_with_crossfade",
    "time_stretch_segment",
    "to_mono",
    "write_wav_pcm16",
]
