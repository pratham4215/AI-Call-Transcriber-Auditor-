"""Deterministic scoring for modern competency audits.

No model-generated points are accepted. The caller supplies the approved competency
weight registry and validated ratings. N/A receives full credit by the currently
recorded project decision; revisit that policy before production rollout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence

RATINGS = {"Full", "Partial", "Zero", "N/A"}


@dataclass(frozen=True)
class CompetencySpec:
    competency_id: str
    weight: float
    partial_fraction: float = 0.5
    allow_na: bool = False
    critical: bool = False

    def __post_init__(self) -> None:
        if not self.competency_id.strip():
            raise ValueError("competency_id must be non-empty")
        if self.weight < 0:
            raise ValueError(f"{self.competency_id}: weight cannot be negative")
        if not 0 <= self.partial_fraction <= 1:
            raise ValueError(f"{self.competency_id}: partial_fraction must be between 0 and 1")


@dataclass(frozen=True)
class CompetencyResult:
    competency_id: str
    rating: str
    weight: float
    earned: float
    na_reason: Optional[str] = None


@dataclass(frozen=True)
class AuditScore:
    results: tuple[CompetencyResult, ...]
    earned: float
    possible: float
    percentage: float
    overridden: bool = False
    override_reason: Optional[str] = None


def validate_registry(registry: Sequence[CompetencySpec], expected_total: float = 100.0) -> None:
    if not registry:
        raise ValueError("registry must contain at least one competency")
    ids = [item.competency_id for item in registry]
    if len(ids) != len(set(ids)):
        raise ValueError("competency IDs must be unique")
    total = sum(item.weight for item in registry)
    if abs(total - expected_total) > 1e-8:
        raise ValueError(f"registry weights total {total:g}; expected {expected_total:g}")


def score_competencies(
    ratings: Mapping[str, Mapping[str, object]],
    registry: Sequence[CompetencySpec],
    *,
    expected_total: float = 100.0,
    wrong_call_tagging: bool = False,
    fake_booking: bool = False,
) -> AuditScore:
    """Score exactly the registry entries and apply integrity overrides last.

    N/A is full credit and remains in the denominator (project decision).
    A competency may be N/A only if its registry spec allows it and a reason
    is supplied. Missing/invalid ratings fail closed to an exception.
    """
    validate_registry(registry, expected_total)
    expected_ids = {s.competency_id for s in registry}
    unknown = set(ratings) - expected_ids
    missing = expected_ids - set(ratings)
    if unknown:
        raise ValueError(f"unknown competency IDs: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing competency ratings: {sorted(missing)}")

    results = []
    earned_total = 0.0
    possible_total = 0.0
    critical_zero = False
    for spec in registry:
        entry = ratings[spec.competency_id]
        if not isinstance(entry, Mapping):
            raise ValueError(f"{spec.competency_id}: rating entry must be an object")
        rating = entry.get("rating")
        if rating not in RATINGS:
            raise ValueError(f"{spec.competency_id}: invalid rating {rating!r}")
        na_reason = entry.get("na_reason")
        if rating == "N/A":
            if not spec.allow_na:
                raise ValueError(f"{spec.competency_id}: N/A is not allowed")
            if not isinstance(na_reason, str) or not na_reason.strip():
                raise ValueError(f"{spec.competency_id}: N/A requires a reason")
            points = spec.weight
        else:
            if na_reason not in (None, ""):
                raise ValueError(f"{spec.competency_id}: na_reason only applies to N/A")
            points = {
                "Full": spec.weight,
                "Partial": spec.weight * spec.partial_fraction,
                "Zero": 0.0,
            }[rating]
        if spec.critical and rating == "Zero":
            critical_zero = True
        results.append(CompetencyResult(
            spec.competency_id, str(rating), spec.weight, round(points, 4),
            str(na_reason).strip() if rating == "N/A" else None,
        ))
        earned_total += points
        possible_total += spec.weight

    reason = "FAKE_BOOKING" if fake_booking else "WRONG_CALL_TAGGING" if wrong_call_tagging else None
    overridden = bool(reason)
    if overridden:
        earned_total = 0.0
    elif critical_zero:
        # Critical-zero policy forces the final score to zero while preserving
        # each competency's earned points for diagnostics.
        reason = "CRITICAL_COMPETENCY_ZERO"
        overridden = True
        earned_total = 0.0

    percentage = round(100 * earned_total / possible_total, 2) if possible_total else 0.0
    return AuditScore(tuple(results), round(earned_total, 4), round(possible_total, 4),
                      percentage, overridden, reason)
