"""Adapter joining validated model output, registry, scoring and review routing.

This is intentionally not called by the legacy pipeline until rollout approval.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping

from .crm_integrity import score_crm_integrity
from .modern_contract import require_valid_modern_evaluation
from .modern_registry import load_registry
from .modern_reporting import score_to_dict
from .modern_review import review_required
from .modern_scoring import score_competencies


def evaluate_payload(
    payload: Mapping[str, Any],
    *,
    crm_integrity_checks: Mapping[str, str] | None = None,
    wrong_call_tagging: bool = False,
    fake_booking: bool = False,
    critical_crm_inconsistency: bool = False,
    contradictory_evidence: bool = False,
    uncertain_na: bool = False,
    low_confidence_critical_decision: bool = False,
) -> Dict[str, Any]:
    """Validate model payload and return deterministic score + review decision.

    If CRM check states are supplied from trusted CRM/rule logic, they replace
    the model's CRM-integrity rating with the deterministic non-additive result.
    """
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

    crm_integrity = None
    if crm_integrity_checks is not None:
        crm_integrity = score_crm_integrity(crm_integrity_checks)
        ratings["crm_documentation_integrity"] = {
            "rating": crm_integrity.rating,
            "na_reason": None,
        }
        critical_crm_inconsistency = critical_crm_inconsistency or bool(
            crm_integrity.material_failures
        )

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
    if score.critical_zero_ids:
        review["required"] = True
        review["reasons"] = list(dict.fromkeys(
            review["reasons"] + [
                "Critical competency rated Zero: " + ", ".join(score.critical_zero_ids)
            ]
        ))
    output["human_review"] = review
    output["registry_version"] = raw_registry["registry_version"]
    recording_result = next(
        (item for item in score.results if item.competency_id == "recording_disclosure"),
        None,
    )
    output["compliance_flags"] = {
        "recording_disclosure_failure": bool(recording_result and recording_result.rating == "Zero"),
        "recording_disclosure_rating": recording_result.rating if recording_result else None,
    }
    if crm_integrity is not None:
        output["crm_integrity_diagnostics"] = {
            "rating": crm_integrity.rating,
            "points": crm_integrity.points,
            "weight": crm_integrity.weight,
            "check_states": crm_integrity.check_states,
            "material_failures": list(crm_integrity.material_failures),
            "minor_defects": list(crm_integrity.minor_defects),
            "source": "deterministic_crm_rules",
        }
    else:
        output["crm_integrity_diagnostics"] = {
            "source": "model_rating_pending_crm_rule_inputs",
            "warning": "Provide crm_integrity_checks to enforce deterministic CRM scoring.",
        }
    return output
