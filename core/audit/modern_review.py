"""Human-review routing policy for modern audit results."""
from __future__ import annotations

from typing import Any, Dict, Mapping, List


def collect_review_reasons(
    evaluation: Mapping[str, Any],
    *,
    wrong_call_tagging: bool = False,
    fake_booking: bool = False,
    critical_crm_inconsistency: bool = False,
    contradictory_evidence: bool = False,
    uncertain_na: bool = False,
    low_confidence_critical_decision: bool = False,
) -> List[str]:
    """Return stable, deduplicated reasons that require reviewer attention."""
    reasons: List[str] = []
    if wrong_call_tagging:
        reasons.append("Wrong call tagging / CRM disposition mismatch")
    if fake_booking:
        reasons.append("Booking genuineness failed or requires adjudication")
    if critical_crm_inconsistency:
        reasons.append("Critical CRM inconsistency")
    if contradictory_evidence:
        reasons.append("Contradictory transcript/CRM evidence")
    if uncertain_na:
        reasons.append("N/A applicability is uncertain")
    if low_confidence_critical_decision:
        reasons.append("Low confidence on a critical decision")

    competencies = evaluation.get("competencies") or []
    if isinstance(competencies, list):
        for item in competencies:
            if not isinstance(item, Mapping):
                reasons.append("Malformed competency output")
                continue
            cid = str(item.get("id") or "unknown")
            if item.get("human_review") is True:
                reasons.append(f"Competency flagged for review: {cid}")
            if item.get("confidence") == "Low":
                reasons.append(f"Low confidence competency: {cid}")
            if item.get("rating") == "N/A" and not str(item.get("na_reason") or "").strip():
                reasons.append(f"N/A lacks a reason: {cid}")

    top_review = evaluation.get("human_review")
    if isinstance(top_review, Mapping):
        extra = top_review.get("reasons") or []
        if isinstance(extra, list):
            reasons.extend(str(r).strip() for r in extra if str(r).strip())
        if top_review.get("required") is True and not extra:
            reasons.append("Model requested human review without a reason")

    # Stable de-duplication preserves the order in which rules fired.
    return list(dict.fromkeys(reasons))


def review_required(evaluation: Mapping[str, Any], **flags: Any) -> Dict[str, Any]:
    reasons = collect_review_reasons(evaluation, **flags)
    return {"required": bool(reasons), "reasons": reasons}
