"""Stable JSON serialization helpers for modern audit scores."""
from __future__ import annotations

from typing import Any, Dict, Mapping

from .modern_scoring import AuditScore


def score_to_dict(
    score: AuditScore,
    *,
    call_type: str,
    competency_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    human_review_reasons: list[str] | None = None,
) -> Dict[str, Any]:
    """Serialize a deterministic score without losing individual diagnostics."""
    metadata = competency_metadata or {}
    critical_zeros = set(score.critical_zero_ids)
    items = []
    for item in score.results:
        extra = metadata.get(item.competency_id, {})
        items.append({
            "id": item.competency_id,
            "name": extra.get("name", item.competency_id),
            "rating": item.rating,
            "weight": item.weight,
            "earned": item.earned,
            "na_reason": item.na_reason,
            "critical": bool(extra.get("critical", False)),
            "critical_zero": item.competency_id in critical_zeros,
        })
    reasons = list(human_review_reasons or [])
    if critical_zeros:
        reasons.append("Critical competency rated Zero: " + ", ".join(sorted(critical_zeros)))
    if score.overridden:
        reasons.append("Final score override: " + str(score.override_reason))
    return {
        "schema_version": "modern-audit-score-v1",
        "call_type": call_type,
        "score": {
            "earned": score.earned,
            "possible": score.possible,
            "percentage": score.percentage,
            "overridden": score.overridden,
            "override_reason": score.override_reason,
        },
        "competencies": items,
        "human_review": {
            "required": bool(reasons),
            "reasons": reasons,
        },
    }
