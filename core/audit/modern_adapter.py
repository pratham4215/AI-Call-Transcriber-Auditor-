"""Adapter joining validated model output, registry, scoring and review routing.

This is intentionally not called by the legacy pipeline until rollout approval.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping

from .modern_contract import require_valid_modern_evaluation
from .modern_registry import load_registry
from .modern_reporting import score_to_dict
from .modern_review import review_required
from .modern_scoring import score_competencies


def evaluate_payload(
    payload: Mapping[str, Any],
    *,
    wrong_call_tagging: bool = False,
    fake_booking: bool = False,
    critical_crm_inconsistency: bool = False,
    contradictory_evidence: bool = False,
    uncertain_na: bool = False,
    low_confidence_critical_decision: bool = False,
) -> Dict[str, Any]:
    """Validate model payload and return deterministic score + review decision."""
    require_valid_modern_evaluation(payload)
    raw_registry, specs = load_registry()
    call_type = str(payload["call_type"])
    expected = {s.competency_id for s in specs}
    entries = payload["competencies"]
    actual = {str(item["id"]) for item in entries}
    if actual != expected:
        raise ValueError(
            f"competency ID mismatch; missing={sorted(expected-actual)}, "
            f"unknown={sorted(actual-expected)}"
        )
    ratings = {
        str(item["id"]): {
            "rating": item["rating"],
            "na_reason": item.get("na_reason"),
        }
        for item in entries
    }
    score = score_competencies(
        ratings, specs, call_type=call_type,
        expected_total=float(raw_registry["total_points"]),
        wrong_call_tagging=wrong_call_tagging,
        fake_booking=fake_booking,
    )
    output = score_to_dict(
        score, call_type=call_type,
        competency_metadata={row["id"]: row for row in raw_registry["competencies"]},
    )
    review = review_required(
        payload,
        wrong_call_tagging=wrong_call_tagging,
        fake_booking=fake_booking,
        critical_crm_inconsistency=critical_crm_inconsistency,
        contradictory_evidence=contradictory_evidence,
        uncertain_na=uncertain_na,
        low_confidence_critical_decision=low_confidence_critical_decision,
    )
    # Include scoring-derived critical-zero flags.
    if score.critical_zero_ids:
        review["required"] = True
        review["reasons"] = list(dict.fromkeys(
            review["reasons"] + [
                "Critical competency rated Zero: " + ", ".join(score.critical_zero_ids)
            ]
        ))
    output["human_review"] = review
    output["registry_version"] = raw_registry["registry_version"]
    return output
