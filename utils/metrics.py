"""
Phase 4 — Temporal detection metrics (IoU, Precision, Recall, F1, boundary error).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

# Ground-truth "silence" ↔ detector "pause"
CATEGORY_ALIASES = {
    "silence": "pause",
    "pause": "pause",
    "pacing": "pacing",
    "pitch": "pitch",
    "energy": "energy",
    "compound": "compound",
}

IOU_THRESH = 0.35
CENTER_OFFSET_THRESH = 0.75


class Interval(BaseModel):
    category: str
    start_sec: float
    end_sec: float
    sub_type: str = ""
    flaw_id: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    @property
    def center(self) -> float:
        return 0.5 * (self.start_sec + self.end_sec)

    @property
    def canon_category(self) -> str:
        return CATEGORY_ALIASES.get(self.category, self.category)


class MatchRecord(BaseModel):
    gt_id: str
    det_id: str
    category: str
    iou: float
    start_err: float
    end_err: float
    center_offset: float


class EvalResult(BaseModel):
    candidate_id: str
    severity_level: int = 0
    n_gt: int = 0
    n_det: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    mean_iou: float = 0.0
    mean_start_err: float = 0.0
    mean_end_err: float = 0.0
    mean_boundary_error: float = 0.0
    matches: list[MatchRecord] = Field(default_factory=list)
    unmatched_gt: list[str] = Field(default_factory=list)
    unmatched_det: list[str] = Field(default_factory=list)


def temporal_iou(d_start: float, d_end: float, g_start: float, g_end: float) -> float:
    """Intersection-over-Union for two 1-D time intervals."""
    inter = max(0.0, min(d_end, g_end) - max(d_start, g_start))
    union = max(d_end, g_end) - min(d_start, g_start)
    if union <= 0:
        return 0.0
    return float(inter / union)


def intervals_match(
    det: Interval,
    gt: Interval,
    iou_thresh: float = IOU_THRESH,
    center_thresh: float = CENTER_OFFSET_THRESH,
) -> tuple[bool, float]:
    if det.canon_category != gt.canon_category:
        return False, 0.0
    iou = temporal_iou(det.start_sec, det.end_sec, gt.start_sec, gt.end_sec)
    center_off = abs(det.center - gt.center)
    ok = iou >= iou_thresh or center_off <= center_thresh
    return ok, iou


def gt_from_annotation(sample: dict[str, Any]) -> list[Interval]:
    out: list[Interval] = []
    for f in sample.get("injected_flaws", []):
        out.append(
            Interval(
                category=str(f.get("category", "")),
                start_sec=float(f["start_sec"]),
                end_sec=float(f["end_sec"]),
                sub_type=str(f.get("sub_type", "")),
                flaw_id=str(f.get("flaw_id", "")),
            )
        )
    return out


def det_from_grounding(result: dict[str, Any] | Any) -> list[Interval]:
    if hasattr(result, "model_dump"):
        data = result.model_dump()
    else:
        data = result
    out: list[Interval] = []
    for f in data.get("flaws", []):
        out.append(
            Interval(
                category=str(f.get("category", "")),
                start_sec=float(f["start_sec"]),
                end_sec=float(f["end_sec"]),
                sub_type=str(f.get("sub_type", "")),
                flaw_id=str(f.get("flaw_id", "")),
            )
        )
    return out


def evaluate_detection(
    candidate_id: str,
    gt_intervals: list[Interval],
    det_intervals: list[Interval],
    severity_level: int = 0,
    iou_thresh: float = IOU_THRESH,
    center_thresh: float = CENTER_OFFSET_THRESH,
) -> EvalResult:
    """
    Greedy one-to-one matching by highest IoU among category-compatible pairs.
    """
    n_gt = len(gt_intervals)
    n_det = len(det_intervals)

    # Level-0 / empty GT: any detection is FP; zero detections → perfect
    if n_gt == 0:
        fp = n_det
        return EvalResult(
            candidate_id=candidate_id,
            severity_level=severity_level,
            n_gt=0,
            n_det=n_det,
            true_positives=0,
            false_positives=fp,
            false_negatives=0,
            precision=1.0 if fp == 0 else 0.0,
            recall=1.0,
            f1=1.0 if fp == 0 else 0.0,
            unmatched_det=[d.flaw_id or f"det_{i}" for i, d in enumerate(det_intervals)],
        )

    # Score matrix
    pairs: list[tuple[float, int, int]] = []
    for gi, g in enumerate(gt_intervals):
        for di, d in enumerate(det_intervals):
            ok, iou = intervals_match(d, g, iou_thresh, center_thresh)
            if ok:
                pairs.append((iou, gi, di))
    pairs.sort(reverse=True, key=lambda x: x[0])

    used_gt: set[int] = set()
    used_det: set[int] = set()
    matches: list[MatchRecord] = []

    for iou, gi, di in pairs:
        if gi in used_gt or di in used_det:
            continue
        g = gt_intervals[gi]
        d = det_intervals[di]
        used_gt.add(gi)
        used_det.add(di)
        matches.append(
            MatchRecord(
                gt_id=g.flaw_id or f"gt_{gi}",
                det_id=d.flaw_id or f"det_{di}",
                category=g.canon_category,
                iou=round(iou, 4),
                start_err=round(abs(d.start_sec - g.start_sec), 4),
                end_err=round(abs(d.end_sec - g.end_sec), 4),
                center_offset=round(abs(d.center - g.center), 4),
            )
        )

    tp = len(matches)
    fp = n_det - tp
    fn = n_gt - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    mean_iou = float(np.mean([m.iou for m in matches])) if matches else 0.0
    mean_s = float(np.mean([m.start_err for m in matches])) if matches else 0.0
    mean_e = float(np.mean([m.end_err for m in matches])) if matches else 0.0

    return EvalResult(
        candidate_id=candidate_id,
        severity_level=severity_level,
        n_gt=n_gt,
        n_det=n_det,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        mean_iou=round(mean_iou, 4),
        mean_start_err=round(mean_s, 4),
        mean_end_err=round(mean_e, 4),
        mean_boundary_error=round(0.5 * (mean_s + mean_e), 4),
        matches=matches,
        unmatched_gt=[
            gt_intervals[i].flaw_id or f"gt_{i}"
            for i in range(n_gt)
            if i not in used_gt
        ],
        unmatched_det=[
            det_intervals[i].flaw_id or f"det_{i}"
            for i in range(n_det)
            if i not in used_det
        ],
    )


def aggregate_metrics(results: list[EvalResult]) -> dict[str, float]:
    """Micro-averaged Precision / Recall / F1 across the dataset."""
    tp = sum(r.true_positives for r in results)
    fp = sum(r.false_positives for r in results)
    fn = sum(r.false_negatives for r in results)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    matched = [r for r in results if r.matches]
    mean_iou = float(np.mean([r.mean_iou for r in matched])) if matched else 0.0
    mean_be = float(np.mean([r.mean_boundary_error for r in matched])) if matched else 0.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_iou": round(mean_iou, 4),
        "mean_boundary_error": round(mean_be, 4),
        "n_samples": len(results),
    }
