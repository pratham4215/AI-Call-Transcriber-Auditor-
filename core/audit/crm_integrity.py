"""Deterministic, non-additive CRM & Documentation Integrity evaluator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

CHECKS = {
    "booking", "follow_up", "snr", "escalation", "cre_remarks", "customer_remarks"
}
STATES = {"pass", "minor_defect", "material_failure", "not_applicable"}


@dataclass(frozen=True)
class CRMIntegrityResult:
    rating: str
    points: int
    weight: int
    check_states: dict[str, str]
    material_failures: tuple[str, ...]
    minor_defects: tuple[str, ...]


def score_crm_integrity(
    checks: Mapping[str, str],
    *,
    weight: int = 10,
    partial_points: int = 5,
) -> CRMIntegrityResult:
    """Aggregate independent CRM checks into one 10-point competency.

    Checks are diagnostic, not additive. A material failure cannot be offset by
    other correct fields. CRE remarks are mandatory; Customer remarks may be N/A
    under the agreed applicability rule.
    """
    missing = CHECKS - set(checks)
    unknown = set(checks) - CHECKS
    if missing:
        raise ValueError(f"missing CRM integrity checks: {sorted(missing)}")
    if unknown:
        raise ValueError(f"unknown CRM integrity checks: {sorted(unknown)}")
    normalized = {key: str(value).strip().lower() for key, value in checks.items()}
    invalid = {key: value for key, value in normalized.items() if value not in STATES}
    if invalid:
        raise ValueError(f"invalid CRM check states: {invalid}")
    if normalized["cre_remarks"] == "not_applicable":
        raise ValueError("CRE remarks are mandatory; N/A is prohibited")

    material = tuple(sorted(k for k, v in normalized.items() if v == "material_failure"))
    minor = tuple(sorted(k for k, v in normalized.items() if v == "minor_defect"))
    if material:
        rating, points = "Zero", 0
    elif minor:
        rating, points = "Partial", partial_points
    else:
        rating, points = "Full", weight
    return CRMIntegrityResult(rating, points, weight, normalized, material, minor)
