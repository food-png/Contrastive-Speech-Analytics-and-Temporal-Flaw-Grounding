"""
Phase 5 — Deterministic causal explanation templates + optional LLM enrichment.

Never raises on missing API keys or network failure; always returns a usable diagnosis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from utils.api_guard import cap_text, get_api_guard
from utils.env_loader import load_project_env

load_project_env()


class CausalDiagnosis(BaseModel):
    flaw_id: str
    category: str
    sub_type: str
    start_sec: float
    end_sec: float
    duration_sec: float
    severity_score: float = Field(ge=0, le=1)
    target_phrase: str = ""
    math_evidence: str
    causal_rationale: str
    impact: str
    actionable_feedback: str
    llm_enrichment: str | None = None
    mathematical_evidence_raw: dict[str, Any] = Field(default_factory=dict)


def _fmt(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)


def _phrase(flaw: dict[str, Any], fallback: str = "the marked clause") -> str:
    p = str(flaw.get("target_phrase") or "").strip()
    return p if p else fallback


def _neighbors_from_alignment(
    align_path: Path | None,
    start: float,
    end: float,
) -> tuple[str, str]:
    if align_path is None or not align_path.exists():
        return ("…", "…")
    try:
        data = json.loads(align_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ("…", "…")
    words = data.get("words", [])
    prev_w, next_w = "…", "…"
    for w in words:
        if float(w.get("end_sec", 0)) <= start + 1e-3:
            prev_w = str(w.get("word", prev_w))
        if float(w.get("start_sec", 0)) >= end - 1e-3:
            next_w = str(w.get("word", next_w))
            break
    return prev_w, next_w


class CausalExplainer:
    """Map Phase-4 detections into structured causal diagnoses."""

    def __init__(self, prefer_llm: bool = True) -> None:
        self.prefer_llm = prefer_llm

    def explain_flaw(
        self,
        flaw: dict[str, Any],
        alignment_path: str | Path | None = None,
        use_llm: bool | None = None,
        api_key: str | None = None,
    ) -> CausalDiagnosis:
        category = str(flaw.get("category", "")).lower()
        sub_type = str(flaw.get("sub_type", "")).lower()
        evidence = dict(flaw.get("mathematical_evidence") or {})
        start = _fmt(flaw.get("start_sec"))
        end = _fmt(flaw.get("end_sec"))
        duration = _fmt(flaw.get("duration_sec"), max(0.0, end - start))
        severity = float(min(1.0, max(0.0, _fmt(flaw.get("severity_score")))))
        phrase = _phrase(flaw)
        align = Path(alignment_path) if alignment_path else None

        math_s, cause, impact, fix = self._template(
            category, sub_type, evidence, duration, phrase, start, end, align
        )

        diagnosis = CausalDiagnosis(
            flaw_id=str(flaw.get("flaw_id", "det_xx")),
            category=category,
            sub_type=sub_type,
            start_sec=round(start, 3),
            end_sec=round(end, 3),
            duration_sec=round(duration, 3),
            severity_score=round(severity, 4),
            target_phrase=phrase if phrase != "the marked clause" else str(flaw.get("target_phrase") or ""),
            math_evidence=math_s,
            causal_rationale=cause,
            impact=impact,
            actionable_feedback=fix,
            mathematical_evidence_raw=evidence,
        )

        do_llm = self.prefer_llm if use_llm is None else use_llm
        if do_llm:
            enrichment = self.enrich_with_llm(diagnosis.model_dump(), api_key=api_key)
            # enrich_with_llm returns deterministic text when LLM unavailable;
            # only store when it differs from the packed deterministic summary.
            packed = f"{cause} {impact}"
            if enrichment and enrichment.strip() and enrichment.strip() != packed.strip():
                diagnosis.llm_enrichment = enrichment.strip()
        return diagnosis

    def explain_detection(
        self,
        detection: dict[str, Any],
        alignment_path: str | Path | None = None,
        use_llm: bool | None = None,
        api_key: str | None = None,
    ) -> list[CausalDiagnosis]:
        out: list[CausalDiagnosis] = []
        for flaw in detection.get("flaws", []):
            out.append(
                self.explain_flaw(
                    flaw,
                    alignment_path=alignment_path,
                    use_llm=use_llm,
                    api_key=api_key,
                )
            )
        return out

    # ------------------------------------------------------------------
    # Templates
    # ------------------------------------------------------------------

    def _template(
        self,
        category: str,
        sub_type: str,
        evidence: dict[str, Any],
        duration: float,
        phrase: str,
        start: float,
        end: float,
        align: Path | None,
    ) -> tuple[str, str, str, str]:
        key = f"{category}/{sub_type}"

        if category == "pacing" and "rush" in sub_type:
            speed = _fmt(evidence.get("pacing_ratio_median"), 1.5)
            measured = _fmt(evidence.get("mean_wpm_cand"), 180.0)
            base = _fmt(evidence.get("mean_wpm_base"), measured / max(speed, 1e-6))
            math_s = (
                f"Speaking rate reached {measured:.0f} WPM "
                f"({speed:.1f}x reference rate of {base:.0f} WPM)."
            )
            cause = f"Rapid syllable compression across phrase '{phrase}'."
            impact = (
                "Audience retention drops as cognitive processing time is truncated; "
                "removes natural rhetorical weight."
            )
            fix = (
                "Decelerate through key clauses, allowing consonants to fully "
                "articulate before the next clause."
            )
            return math_s, cause, impact, fix

        if category == "pacing" and "drag" in sub_type:
            speed = _fmt(evidence.get("pacing_ratio_median"), 0.6)
            measured = _fmt(evidence.get("mean_wpm_cand"), 70.0)
            base = _fmt(evidence.get("mean_wpm_base"), 120.0)
            math_s = (
                f"Speaking rate fell to {measured:.0f} WPM "
                f"({speed:.1f}x reference rate of {base:.0f} WPM)."
            )
            cause = f"Excessive syllable elongation across phrase '{phrase}'."
            impact = (
                "Momentum collapses; adjudicators perceive hesitation and loss of "
                "argumentative drive."
            )
            fix = (
                "Tighten clause transitions and reclaim a steady mid-tempo cadence "
                "near the reference rate."
            )
            return math_s, cause, impact, fix

        if category == "pitch" or "monotone" in sub_type or "flatten" in sub_type:
            ratio = _fmt(evidence.get("sigma_ratio_median"), 0.3)
            cand_std = _fmt(evidence.get("mean_sigma_cand"), ratio)
            base_std = _fmt(evidence.get("mean_sigma_base"), max(cand_std, 1e-3))
            if base_std > 1e-6:
                drop_pct = max(0.0, (1.0 - (cand_std / base_std)) * 100.0)
            else:
                drop_pct = max(0.0, (1.0 - ratio) * 100.0)
            math_s = (
                f"F0 variance collapsed by {drop_pct:.1f}% relative to reference "
                f"standard deviation ({cand_std:.2f} vs {base_std:.2f})."
            )
            cause = f"Absence of fundamental pitch inflection during '{phrase}'."
            impact = (
                "Flat acoustic profile induces listener fatigue and fails to signal "
                "thematic hierarchy."
            )
            fix = (
                "Elevate pitch on primary nouns or verbs to indicate emphasis; "
                "utilize downward inflection for declarative finality."
            )
            return math_s, cause, impact, fix

        if category in ("pause", "silence") or "pause" in sub_type or "hesitation" in sub_type:
            cand_p = _fmt(evidence.get("cand_pause_sec"), duration)
            base_p = _fmt(evidence.get("base_pause_sec"), 0.0)
            prev_w, next_w = _neighbors_from_alignment(align, start, end)
            math_s = (
                f"Unscheduled silence of {cand_p:.2f}s "
                f"(reference pause was {base_p:.2f}s)."
            )
            cause = f"Prolonged hesitation between '{prev_w}' and '{next_w}'."
            impact = (
                "Breaks syntactic momentum and signals hesitation or memory lapse "
                "to adjudicators."
            )
            fix = (
                "Maintain steady breathing cadence; if pausing for dramatic effect, "
                "cap duration at 0.8–1.2s."
            )
            return math_s, cause, impact, fix

        if category == "energy" and "spike" in sub_type:
            surge = _fmt(evidence.get("peak_delta_db"), 12.0)
            math_s = f"Loudness surged by +{surge:.1f} dB relative to speaker baseline."
            cause = f"Uncontrolled acoustic burst on unstressed syllable '{phrase}'."
            impact = "Distorts dynamic equilibrium and produces jarring acoustic artifacts."
            fix = (
                "Project from the diaphragm with steady breath support rather than "
                "vocal tract pushing."
            )
            return math_s, cause, impact, fix

        if category == "energy" and ("under" in sub_type or "drop" in sub_type):
            drop = abs(_fmt(evidence.get("median_delta_db"), 14.0))
            math_s = f"Loudness fell by −{drop:.1f} dB on an emphatic reference region."
            cause = f"Under-projected delivery through '{phrase}'."
            impact = (
                "Emphasis landmarks disappear; key claims lose audible hierarchy "
                "in the hall."
            )
            fix = (
                "Support emphatic nouns with increased subglottal pressure while "
                "keeping pitch variety intact."
            )
            return math_s, cause, impact, fix

        # Generic fallback
        math_s = (
            f"Residual anomaly ({category}/{sub_type}) spanning {duration:.2f}s "
            f"with severity { _fmt(evidence.get('severity_score'), 0.5):.2f}."
            if evidence
            else f"Residual anomaly ({category}/{sub_type}) spanning {duration:.2f}s."
        )
        cause = f"Acoustic deviation detected around '{phrase}'."
        impact = "Delivery clarity and rhetorical control are reduced in this window."
        fix = "Review the marked interval and re-record with deliberate prosodic control."
        _ = key
        return math_s, cause, impact, fix

    # ------------------------------------------------------------------
    # Optional LLM layer
    # ------------------------------------------------------------------

    def enrich_with_llm(
        self,
        structured_flaw_dict: dict[str, Any],
        api_key: str | None = None,
    ) -> str:
        """
        Return a 2-sentence conversational summary when an LLM is available;
        otherwise return the deterministic cause+impact string.
        """
        deterministic = (
            f"{structured_flaw_dict.get('causal_rationale', '')} "
            f"{structured_flaw_dict.get('impact', '')}"
        ).strip()
        system = (
            "You are a competitive oratory coach. In exactly two concise sentences, "
            "explain this delivery flaw using only the numbers and phrase in the prompt. "
            "Do not invent measurements."
        )
        try:
            return get_api_guard().call_llm(
                compact_flaw_prompt(structured_flaw_dict),
                system_instruction=system,
                fallback_text=deterministic,
                extra_gemini_key=api_key,
            )
        except Exception:  # noqa: BLE001 — guaranteed local template
            return deterministic


def compact_flaw_prompt(flaw: dict[str, Any]) -> str:
    """Send only the phrase span and acoustic deltas, not the full report object."""
    raw = dict(flaw.get("mathematical_evidence_raw") or {})
    deltas = {str(key): raw[key] for key in list(raw)[:8]}
    payload = {
        "category": flaw.get("category"),
        "sub_type": flaw.get("sub_type"),
        "start_sec": flaw.get("start_sec"),
        "end_sec": flaw.get("end_sec"),
        "target_phrase": str(flaw.get("target_phrase") or "")[:160],
        "math_evidence": str(flaw.get("math_evidence") or "")[:280],
        "deltas": deltas,
    }
    return cap_text(json.dumps(payload, ensure_ascii=True), limit=900)
