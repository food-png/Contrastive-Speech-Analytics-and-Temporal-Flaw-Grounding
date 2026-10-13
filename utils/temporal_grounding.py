"""
Phase 4 — Contrastive temporal flaw grounding.

Word-anchored baseline↔candidate alignment, residual tracks
(pitch / energy / pacing / pause), morphological interval extraction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

EPS = 1e-8
ROLL_WIN_SEC = 1.5
MONOTONE_RATIO = 0.35
RUSH_RATIO = 1.40
DRAG_RATIO = 0.65
ENERGY_DROP_DB = -14.0
ENERGY_SPIKE_DB = 12.0
AWKWARD_PAUSE_SEC = 1.5
PAUSE_MULT = 3.0
MERGE_GAP_SEC = 0.55
BOUNDARY_PAD_SEC = 0.10
MIN_DUR_SEC = 0.25
SPIKE_MIN_DUR = 0.08

Category = Literal["pacing", "pitch", "energy", "pause"]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class DetectedFlaw(BaseModel):
    flaw_id: str
    category: Category
    sub_type: str
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    duration_sec: float = Field(ge=0)
    severity_score: float = Field(ge=0, le=1)
    mathematical_evidence: dict[str, Any] = Field(default_factory=dict)
    target_phrase: str = ""

    @model_validator(mode="after")
    def ordered(self) -> "DetectedFlaw":
        if self.end_sec < self.start_sec:
            raise ValueError("end_sec precedes start_sec")
        expected = round(self.end_sec - self.start_sec, 6)
        if abs(self.duration_sec - expected) > 0.05:
            self.duration_sec = expected
        self.severity_score = float(np.clip(self.severity_score, 0.0, 1.0))
        return self


class GroundingResult(BaseModel):
    candidate_id: str
    baseline_id: str
    flaws: list[DetectedFlaw]
    residual_summary: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_feature_npz(path: str | Path) -> dict[str, np.ndarray | dict]:
    path = Path(path)
    with np.load(str(path), allow_pickle=True) as z:
        data = {k: z[k] for k in z.files}
    meta: dict = {}
    if "metadata_json" in data:
        raw = data["metadata_json"]
        meta = json.loads(str(raw.item() if hasattr(raw, "item") else raw))
    out: dict[str, np.ndarray | dict] = {
        "time": np.asarray(data["time_seconds"], dtype=np.float64),
        "f0_norm": np.asarray(data["f0_norm"], dtype=np.float64),
        "f0_raw": np.asarray(data["f0_raw"], dtype=np.float64),
        "energy": np.asarray(data["energy_norm"], dtype=np.float64),
        "wpm": np.asarray(data["rolling_wpm"], dtype=np.float64),
        "voiced": np.asarray(data["voiced_mask"]).astype(bool),
        "metadata": meta,
    }
    if "pause_ratio" in data:
        out["pause_ratio"] = np.asarray(data["pause_ratio"], dtype=np.float64)
    return out


def load_alignment(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    words = [
        {
            "word": str(w.get("word", "")),
            "start": float(w["start_sec"]),
            "end": float(w["end_sec"]),
        }
        for w in data.get("words", [])
        if float(w.get("end_sec", 0)) > float(w.get("start_sec", 0))
    ]
    pauses = [
        {"start": float(p["start_sec"]), "end": float(p["end_sec"])}
        for p in data.get("pauses", [])
        if float(p.get("end_sec", 0)) > float(p.get("start_sec", 0))
    ]
    return {"words": words, "pauses": pauses}


# ---------------------------------------------------------------------------
# Word-anchored warp
# ---------------------------------------------------------------------------

def _piecewise_map(
    t_cand: np.ndarray,
    base_words: list[dict],
    cand_words: list[dict],
) -> np.ndarray:
    """
    Map candidate times onto the baseline timeline via shared word indices.
    Outside word spans, linearly extrapolate from nearest anchors.
    """
    n = min(len(base_words), len(cand_words))
    if n == 0:
        # identity-ish: stretch by duration ratio
        if t_cand.size == 0:
            return t_cand.copy()
        return t_cand.copy()

    # Anchor points: start and end of each paired word
    src = []  # candidate
    dst = []  # baseline
    for i in range(n):
        src.extend([cand_words[i]["start"], cand_words[i]["end"]])
        dst.extend([base_words[i]["start"], base_words[i]["end"]])
    src_a = np.asarray(src, dtype=np.float64)
    dst_a = np.asarray(dst, dtype=np.float64)
    # Enforce strictly increasing source for np.interp
    order = np.argsort(src_a)
    src_a, dst_a = src_a[order], dst_a[order]
    # Deduplicate nearly-identical source times
    keep = np.concatenate([[True], np.diff(src_a) > 1e-6])
    src_a, dst_a = src_a[keep], dst_a[keep]
    if src_a.size < 2:
        return t_cand.copy()
    return np.interp(t_cand, src_a, dst_a, left=dst_a[0], right=dst_a[-1])


def _resample_on_cand(
    t_cand: np.ndarray,
    t_base_mapped: np.ndarray,
    t_base: np.ndarray,
    series: np.ndarray,
) -> np.ndarray:
    """Sample a baseline series at baseline-times corresponding to each candidate frame."""
    if t_base.size < 2 or series.size == 0:
        return np.zeros_like(t_cand)
    return np.interp(t_base_mapped, t_base, series, left=series[0], right=series[-1])


def _rolling_std(x: np.ndarray, times: np.ndarray, win_sec: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float64)
    half = win_sec / 2.0
    for i, t in enumerate(times):
        m = (times >= t - half) & (times <= t + half)
        vals = x[m]
        vals = vals[np.isfinite(vals)]
        out[i] = float(np.std(vals)) if vals.size >= 2 else 0.0
    return out


def _mask_to_intervals(
    mask: np.ndarray,
    times: np.ndarray,
    merge_gap: float = MERGE_GAP_SEC,
    min_dur: float = MIN_DUR_SEC,
) -> list[tuple[float, float]]:
    """Convert boolean frame mask → merged [start, end] intervals."""
    if mask.size == 0 or times.size == 0:
        return []
    mask = np.asarray(mask, dtype=bool)
    intervals: list[tuple[float, float]] = []
    i = 0
    n = len(mask)
    dt = float(np.median(np.diff(times))) if n > 1 else 0.032
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        start = float(times[i])
        end = float(times[min(j, n - 1)]) + (dt if j >= n else 0.0)
        if j < n:
            end = float(times[j - 1]) + dt
        intervals.append((start, max(start + 1e-3, end)))
        i = j

    if not intervals:
        return []

    # Merge gaps < merge_gap
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s - pe <= merge_gap:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))

    return [(s, e) for s, e in merged if (e - s) >= min_dur]


def _phrase_for_window(
    words: list[dict],
    start: float,
    end: float,
    max_words: int = 10,
) -> str:
    hits = [w["word"] for w in words if w["end"] > start and w["start"] < end]
    if not hits:
        return ""
    return " ".join(hits[:max_words])


def _severity_from_ratio(ratio: float, soft: float, hard: float) -> float:
    """Map a magnitude ratio into [0, 1] between soft and hard thresholds."""
    if hard <= soft:
        return float(np.clip(ratio, 0.0, 1.0))
    return float(np.clip((ratio - soft) / (hard - soft + EPS), 0.0, 1.0))


def _overlap_frac(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    dur = max(1e-6, a1 - a0)
    return inter / dur


def _postprocess_flaws(
    flaws: list[DetectedFlaw],
    cand_pauses: list[dict],
) -> list[DetectedFlaw]:
    """
    Suppress low-severity noise, drop drag intervals that are mostly pauses,
    and NMS-merge overlapping same-category detections.
    """
    if not flaws:
        return []

    kept: list[DetectedFlaw] = []
    for f in flaws:
        # Always keep awkward pauses and volume spikes above mild severity
        if f.sub_type in ("awkward_pause", "volume_spike"):
            if f.severity_score >= 0.15:
                kept.append(f)
            continue
        # Drop micro / weak residuals
        if f.severity_score < 0.28 and f.duration_sec < 0.8:
            continue
        if f.sub_type == "dragging_cadence":
            # If mostly overlapping a long pause, the pause detector owns it
            pause_cov = 0.0
            for p in cand_pauses:
                pause_cov = max(
                    pause_cov,
                    _overlap_frac(f.start_sec, f.end_sec, p["start"], p["end"]),
                )
            if pause_cov >= 0.55:
                continue
            if f.duration_sec < 0.6:
                continue
        if f.sub_type == "under_projection" and f.duration_sec < 0.6:
            continue
        if f.sub_type == "monotone_flattening" and f.duration_sec < 0.7:
            continue
        if f.severity_score < 0.22:
            continue
        kept.append(f)

    # Sort by severity then merge overlaps within category
    kept.sort(key=lambda x: (-x.severity_score, x.start_sec))
    final: list[DetectedFlaw] = []
    for f in kept:
        absorbed = False
        for i, g in enumerate(final):
            if g.category != f.category:
                continue
            iou = (
                max(0.0, min(f.end_sec, g.end_sec) - max(f.start_sec, g.start_sec))
                / max(1e-6, max(f.end_sec, g.end_sec) - min(f.start_sec, g.start_sec))
            )
            if iou >= 0.35 or _overlap_frac(f.start_sec, f.end_sec, g.start_sec, g.end_sec) >= 0.5:
                # Keep the higher-severity span; expand bounds
                new_start = min(f.start_sec, g.start_sec)
                new_end = max(f.end_sec, g.end_sec)
                winner = f if f.severity_score >= g.severity_score else g
                final[i] = DetectedFlaw(
                    flaw_id=winner.flaw_id,
                    category=winner.category,
                    sub_type=winner.sub_type,
                    start_sec=round(new_start, 3),
                    end_sec=round(new_end, 3),
                    duration_sec=round(new_end - new_start, 3),
                    severity_score=max(f.severity_score, g.severity_score),
                    mathematical_evidence=winner.mathematical_evidence,
                    target_phrase=winner.target_phrase or f.target_phrase,
                )
                absorbed = True
                break
        if not absorbed:
            final.append(f)

    final = _merge_adjacent_same_category(final, MERGE_GAP_SEC)
    final = _relax_bounds(final, BOUNDARY_PAD_SEC)
    final.sort(key=lambda x: (x.start_sec, x.category))
    return final


def _merge_adjacent_same_category(
    flaws: list[DetectedFlaw],
    gap_sec: float,
) -> list[DetectedFlaw]:
    """Join same-category boxes separated by a short micro-gap."""
    grouped: dict[str, list[DetectedFlaw]] = {}
    for flaw in flaws:
        grouped.setdefault(flaw.category, []).append(flaw)
    merged: list[DetectedFlaw] = []
    for group in grouped.values():
        group.sort(key=lambda item: item.start_sec)
        current = group[0]
        for nxt in group[1:]:
            if nxt.start_sec - current.end_sec <= gap_sec:
                start = min(current.start_sec, nxt.start_sec)
                end = max(current.end_sec, nxt.end_sec)
                winner = current if current.severity_score >= nxt.severity_score else nxt
                current = DetectedFlaw(
                    flaw_id=winner.flaw_id,
                    category=winner.category,
                    sub_type=winner.sub_type,
                    start_sec=round(start, 3),
                    end_sec=round(end, 3),
                    duration_sec=round(end - start, 3),
                    severity_score=max(current.severity_score, nxt.severity_score),
                    mathematical_evidence=winner.mathematical_evidence,
                    target_phrase=winner.target_phrase or nxt.target_phrase,
                )
            else:
                merged.append(current)
                current = nxt
        merged.append(current)
    return merged


def _relax_bounds(flaws: list[DetectedFlaw], pad_sec: float) -> list[DetectedFlaw]:
    """Cover onset/offset transients without moving the interval backward of 0."""
    relaxed: list[DetectedFlaw] = []
    for flaw in flaws:
        start = max(0.0, flaw.start_sec - pad_sec)
        end = flaw.end_sec + pad_sec
        relaxed.append(
            DetectedFlaw(
                flaw_id=flaw.flaw_id,
                category=flaw.category,
                sub_type=flaw.sub_type,
                start_sec=round(start, 3),
                end_sec=round(end, 3),
                duration_sec=round(end - start, 3),
                severity_score=flaw.severity_score,
                mathematical_evidence=flaw.mathematical_evidence,
                target_phrase=flaw.target_phrase,
            )
        )
    return relaxed


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class TemporalFlawGrounder:
    """Contrastive residual analysis for temporal flaw grounding."""

    def __init__(
        self,
        roll_win_sec: float = ROLL_WIN_SEC,
        monotone_ratio: float = MONOTONE_RATIO,
        rush_ratio: float = RUSH_RATIO,
        drag_ratio: float = DRAG_RATIO,
        merge_gap_sec: float = MERGE_GAP_SEC,
        min_dur_sec: float = MIN_DUR_SEC,
    ) -> None:
        self.roll_win_sec = roll_win_sec
        self.monotone_ratio = monotone_ratio
        self.rush_ratio = rush_ratio
        self.drag_ratio = drag_ratio
        self.merge_gap_sec = merge_gap_sec
        self.min_dur_sec = min_dur_sec

    def ground(
        self,
        candidate_id: str,
        baseline_id: str,
        cand_feat: dict,
        base_feat: dict,
        cand_align: dict,
        base_align: dict,
    ) -> GroundingResult:
        t_c = np.asarray(cand_feat["time"], dtype=np.float64)
        t_b = np.asarray(base_feat["time"], dtype=np.float64)
        if t_c.size == 0:
            return GroundingResult(
                candidate_id=candidate_id,
                baseline_id=baseline_id,
                flaws=[],
                residual_summary={"error": "empty candidate timeline"},
            )

        base_words = base_align.get("words", [])
        cand_words = cand_align.get("words", [])
        cand_pauses = cand_align.get("pauses", [])
        base_pauses = base_align.get("pauses", [])

        # Word-anchored map: candidate time → baseline time
        t_base_of_cand = _piecewise_map(t_c, base_words, cand_words)

        # Optional DTW refinement on coarse energy envelopes for robustness
        try:
            t_base_of_cand = self._optional_dtw_refine(
                t_c, t_b, t_base_of_cand,
                np.asarray(cand_feat["energy"]),
                np.asarray(base_feat["energy"]),
            )
        except Exception:  # noqa: BLE001
            pass

        f0_c = np.asarray(cand_feat["f0_norm"], dtype=np.float64)
        f0_b_on_c = _resample_on_cand(t_c, t_base_of_cand, t_b, np.asarray(base_feat["f0_norm"]))
        e_c = np.asarray(cand_feat["energy"], dtype=np.float64)
        e_b_on_c = _resample_on_cand(t_c, t_base_of_cand, t_b, np.asarray(base_feat["energy"]))
        wpm_c = np.asarray(cand_feat["wpm"], dtype=np.float64)
        wpm_b_on_c = _resample_on_cand(t_c, t_base_of_cand, t_b, np.asarray(base_feat["wpm"]))
        voiced_c = np.asarray(cand_feat["voiced"]).astype(bool)

        # --- Pitch variance residual ---
        sigma_c = _rolling_std(f0_c, t_c, self.roll_win_sec)
        sigma_b = _rolling_std(f0_b_on_c, t_c, self.roll_win_sec)
        # Strong monotone OR mild local collapse with low absolute sigma
        monotone_mask = voiced_c & (sigma_b > 0.08) & (
            (sigma_c < self.monotone_ratio * sigma_b)
            | ((sigma_c < 0.55 * sigma_b) & (sigma_c < 0.18))
        )

        # --- Energy residual ---
        delta_e = np.abs(e_c - e_b_on_c)
        # Under-projection: candidate much quieter where baseline is loud & voiced
        under_mask = (
            voiced_c
            & (e_b_on_c >= -6.0)
            & ((e_c - e_b_on_c) <= ENERGY_DROP_DB)
        )
        # Spikes: candidate much louder (impulse-friendly)
        spike_mask = (e_c - e_b_on_c) >= ENERGY_SPIKE_DB

        # --- Pacing ratio ---
        pacing_ratio = (wpm_c + EPS) / (wpm_b_on_c + EPS)
        # Require baseline to have measurable articulation to avoid /0 artifacts
        rush_mask = (wpm_b_on_c >= 40.0) & (pacing_ratio >= self.rush_ratio)
        drag_mask = (wpm_b_on_c >= 40.0) & (pacing_ratio <= self.drag_ratio) & (wpm_c <= 70.0)

        flaws: list[DetectedFlaw] = []
        fid = 1

        # Pitch intervals
        for s, e in _mask_to_intervals(monotone_mask, t_c, self.merge_gap_sec, self.min_dur_sec):
            m = (t_c >= s) & (t_c <= e)
            ratio = float(np.median(sigma_c[m] / (sigma_b[m] + EPS))) if np.any(m) else 1.0
            sev = _severity_from_ratio(1.0 - ratio, 0.3, 0.9)
            flaws.append(
                DetectedFlaw(
                    flaw_id=f"det_{fid:02d}",
                    category="pitch",
                    sub_type="monotone_flattening",
                    start_sec=round(s, 3),
                    end_sec=round(e, 3),
                    duration_sec=round(e - s, 3),
                    severity_score=sev,
                    mathematical_evidence={
                        "sigma_ratio_median": round(ratio, 4),
                        "monotone_ratio_threshold": self.monotone_ratio,
                        "mean_sigma_cand": round(float(np.mean(sigma_c[m])), 4) if np.any(m) else 0.0,
                        "mean_sigma_base": round(float(np.mean(sigma_b[m])), 4) if np.any(m) else 0.0,
                    },
                    target_phrase=_phrase_for_window(cand_words, s, e),
                )
            )
            fid += 1

        # Pacing rush / drag
        for s, e in _mask_to_intervals(rush_mask, t_c, self.merge_gap_sec, self.min_dur_sec):
            m = (t_c >= s) & (t_c <= e)
            pr = float(np.median(pacing_ratio[m])) if np.any(m) else self.rush_ratio
            sev = _severity_from_ratio(pr, self.rush_ratio, 2.0)
            flaws.append(
                DetectedFlaw(
                    flaw_id=f"det_{fid:02d}",
                    category="pacing",
                    sub_type="rushed_delivery",
                    start_sec=round(s, 3),
                    end_sec=round(e, 3),
                    duration_sec=round(e - s, 3),
                    severity_score=sev,
                    mathematical_evidence={
                        "pacing_ratio_median": round(pr, 4),
                        "threshold": self.rush_ratio,
                        "mean_wpm_cand": round(float(np.mean(wpm_c[m])), 2) if np.any(m) else 0.0,
                        "mean_wpm_base": round(float(np.mean(wpm_b_on_c[m])), 2) if np.any(m) else 0.0,
                    },
                    target_phrase=_phrase_for_window(cand_words, s, e),
                )
            )
            fid += 1

        for s, e in _mask_to_intervals(drag_mask, t_c, self.merge_gap_sec, self.min_dur_sec):
            m = (t_c >= s) & (t_c <= e)
            pr = float(np.median(pacing_ratio[m])) if np.any(m) else self.drag_ratio
            sev = _severity_from_ratio(self.drag_ratio - pr, 0.0, 0.5)
            flaws.append(
                DetectedFlaw(
                    flaw_id=f"det_{fid:02d}",
                    category="pacing",
                    sub_type="dragging_cadence",
                    start_sec=round(s, 3),
                    end_sec=round(e, 3),
                    duration_sec=round(e - s, 3),
                    severity_score=sev,
                    mathematical_evidence={
                        "pacing_ratio_median": round(pr, 4),
                        "threshold": self.drag_ratio,
                    },
                    target_phrase=_phrase_for_window(cand_words, s, e),
                )
            )
            fid += 1

        # Energy under-projection (drops)
        for s, e in _mask_to_intervals(under_mask, t_c, self.merge_gap_sec, self.min_dur_sec):
            m = (t_c >= s) & (t_c <= e)
            drop = float(np.median(e_c[m] - e_b_on_c[m])) if np.any(m) else ENERGY_DROP_DB
            sev = _severity_from_ratio(-drop, 14.0, 40.0)
            flaws.append(
                DetectedFlaw(
                    flaw_id=f"det_{fid:02d}",
                    category="energy",
                    sub_type="under_projection",
                    start_sec=round(s, 3),
                    end_sec=round(e, 3),
                    duration_sec=round(e - s, 3),
                    severity_score=sev,
                    mathematical_evidence={
                        "median_delta_db": round(drop, 3),
                        "threshold_db": ENERGY_DROP_DB,
                    },
                    target_phrase=_phrase_for_window(cand_words, s, e),
                )
            )
            fid += 1

        # Volume spikes — allow short impulses
        for s, e in _mask_to_intervals(spike_mask, t_c, self.merge_gap_sec, SPIKE_MIN_DUR):
            m = (t_c >= s) & (t_c <= e)
            peak = float(np.max(e_c[m] - e_b_on_c[m])) if np.any(m) else ENERGY_SPIKE_DB
            sev = _severity_from_ratio(peak, ENERGY_SPIKE_DB, 24.0)
            flaws.append(
                DetectedFlaw(
                    flaw_id=f"det_{fid:02d}",
                    category="energy",
                    sub_type="volume_spike",
                    start_sec=round(s, 3),
                    end_sec=round(e, 3),
                    duration_sec=round(e - s, 3),
                    severity_score=sev,
                    mathematical_evidence={
                        "peak_delta_db": round(peak, 3),
                        "threshold_db": ENERGY_SPIKE_DB,
                    },
                    target_phrase=_phrase_for_window(cand_words, s, e),
                )
            )
            fid += 1

        # Awkward pauses (alignment-level)
        for cp in cand_pauses:
            c_dur = cp["end"] - cp["start"]
            if c_dur < AWKWARD_PAUSE_SEC:
                continue
            # Baseline pause overlapping the mapped window, or continuous speech
            b_start = float(np.interp(cp["start"], t_c, t_base_of_cand))
            b_end = float(np.interp(cp["end"], t_c, t_base_of_cand))
            overlapping = [
                p for p in base_pauses
                if p["end"] > b_start and p["start"] < b_end
            ]
            if overlapping:
                b_dur = max(p["end"] - p["start"] for p in overlapping)
                is_awkward = c_dur > PAUSE_MULT * max(b_dur, 0.05)
            else:
                # Continuous baseline speech in that mapped region
                is_awkward = True
                b_dur = 0.0
            if not is_awkward:
                continue
            sev = _severity_from_ratio(c_dur, AWKWARD_PAUSE_SEC, 3.5)
            flaws.append(
                DetectedFlaw(
                    flaw_id=f"det_{fid:02d}",
                    category="pause",
                    sub_type="awkward_pause",
                    start_sec=round(cp["start"], 3),
                    end_sec=round(cp["end"], 3),
                    duration_sec=round(c_dur, 3),
                    severity_score=sev,
                    mathematical_evidence={
                        "cand_pause_sec": round(c_dur, 3),
                        "base_pause_sec": round(float(b_dur), 3),
                        "delta_pause_sec": round(c_dur - float(b_dur), 3),
                        "mult_threshold": PAUSE_MULT,
                    },
                    target_phrase=_phrase_for_window(cand_words, cp["start"], cp["end"]),
                )
            )
            fid += 1

        flaws = _postprocess_flaws(flaws, cand_pauses)
        for i, f in enumerate(flaws, 1):
            f.flaw_id = f"det_{i:02d}"

        residual_summary = {
            "n_frames": int(t_c.size),
            "mean_abs_energy_delta": round(float(np.mean(delta_e)), 4),
            "mean_pacing_ratio": round(float(np.mean(pacing_ratio)), 4),
            "monotone_frame_fraction": round(float(np.mean(monotone_mask)), 4),
            "rush_frame_fraction": round(float(np.mean(rush_mask)), 4),
            "drag_frame_fraction": round(float(np.mean(drag_mask)), 4),
            "n_detected_flaws": len(flaws),
            "categories": sorted({f.category for f in flaws}),
            # Residual tracks for plotting (downsampled lightly)
            "tracks": {
                "time": t_c[::2].round(4).tolist(),
                "f0_cand": f0_c[::2].round(4).tolist(),
                "f0_base": f0_b_on_c[::2].round(4).tolist(),
                "energy_cand": e_c[::2].round(4).tolist(),
                "energy_base": e_b_on_c[::2].round(4).tolist(),
                "wpm_cand": wpm_c[::2].round(4).tolist(),
                "wpm_base": wpm_b_on_c[::2].round(4).tolist(),
                "pacing_ratio": pacing_ratio[::2].round(4).tolist(),
                "sigma_f0_cand": sigma_c[::2].round(4).tolist(),
                "sigma_f0_base": sigma_b[::2].round(4).tolist(),
                "monotone_mask": monotone_mask[::2].astype(int).tolist(),
                "rush_mask": rush_mask[::2].astype(int).tolist(),
                "drag_mask": drag_mask[::2].astype(int).tolist(),
                "spike_mask": spike_mask[::2].astype(int).tolist(),
                "under_mask": under_mask[::2].astype(int).tolist(),
            },
        }

        return GroundingResult(
            candidate_id=candidate_id,
            baseline_id=baseline_id,
            flaws=flaws,
            residual_summary=residual_summary,
        )

    def _optional_dtw_refine(
        self,
        t_c: np.ndarray,
        t_b: np.ndarray,
        t_map: np.ndarray,
        e_c: np.ndarray,
        e_b: np.ndarray,
    ) -> np.ndarray:
        """Blend piecewise word map with a coarse fastdtw path on energy (downsampled)."""
        try:
            from fastdtw import fastdtw
            from scipy.spatial.distance import euclidean
        except ImportError:
            return t_map

        step = max(1, len(t_c) // 200)
        c_idx = np.arange(0, len(t_c), step)
        b_idx = np.arange(0, len(t_b), step)
        if c_idx.size < 4 or b_idx.size < 4:
            return t_map
        xc = e_c[c_idx].reshape(-1, 1)
        xb = e_b[b_idx].reshape(-1, 1)
        _, path = fastdtw(xc, xb, dist=euclidean, radius=10)
        if not path:
            return t_map
        src = np.array([t_c[c_idx[i]] for i, _ in path], dtype=np.float64)
        dst = np.array([t_b[b_idx[j]] for _, j in path], dtype=np.float64)
        keep = np.concatenate([[True], np.diff(src) > 1e-6])
        src, dst = src[keep], dst[keep]
        if src.size < 2:
            return t_map
        dtw_map = np.interp(t_c, src, dst, left=dst[0], right=dst[-1])
        # Blend: trust word anchors more
        return 0.7 * t_map + 0.3 * dtw_map

    def ground_files(
        self,
        candidate_id: str,
        baseline_id: str,
        cand_feat_path: str | Path,
        base_feat_path: str | Path,
        cand_align_path: str | Path,
        base_align_path: str | Path,
    ) -> GroundingResult:
        return self.ground(
            candidate_id=candidate_id,
            baseline_id=baseline_id,
            cand_feat=load_feature_npz(cand_feat_path),
            base_feat=load_feature_npz(base_feat_path),
            cand_align=load_alignment(cand_align_path),
            base_align=load_alignment(base_align_path),
        )
