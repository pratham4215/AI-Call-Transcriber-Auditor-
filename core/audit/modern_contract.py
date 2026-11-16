"""Validation helpers for the proposed modern audit contract.

This module is intentionally isolated from the legacy pipeline until the
12-competency registry and rollout decision are approved.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

CONTRACT_VERSION = "modern-audit-evaluation-v1"
CALL_TYPES = {"BOOKING", "FOLLOW_UPS", "SERVICE_NT_REQUIRED", "ESCALATION"}
RATINGS = {"Full", "Partial", "Zero", "N/A"}
CONFIDENCE = {"High", "Medium", "Low"}
EVIDENCE_SOURCES = {"transcript", "crm"}


def validate_modern_evaluation(payload: Any) -> List[str]:
    """Return validation errors; an empty list means the payload is structurally valid.

    This performs dependency-free structural checks. JSON Schema validation can
    additionally be run against schemas/modern-audit-evaluation-v1.schema.json.
    """
    errors: List[str] = []
    if not isinstance(payload, Mapping):
        return ["payload must be a JSON object"]

    required_top = {"schema_version", "call_type", "competencies", "human_review"}
    missing = required_top - set(payload)
    if missing:
        errors.append(f"missing top-level fields: {sorted(missing)}")
    extras = set(payload) - required_top
    if extras:
        errors.append(f"unexpected top-level fields: {sorted(extras)}")

    if payload.get("schema_version") != CONTRACT_VERSION:
        errors.append(f"schema_version must be {CONTRACT_VERSION!r}")
    if payload.get("call_type") not in CALL_TYPES:
        errors.append("call_type is unsupported")

    competencies = payload.get("competencies")
    if not isinstance(competencies, list) or not competencies:
        errors.append("competencies must be a non-empty array")
        competencies = []

    seen_ids = set()
    for i, item in enumerate(competencies):
        prefix = f"competencies[{i}]"
        if not isinstance(item, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        required = {"id", "rating", "evidence", "reasoning", "confidence", "human_review"}
        missing_item = required - set(item)
        extra_item = set(item) - (required | {"na_reason"})
        if missing_item:
            errors.append(f"{prefix} missing fields: {sorted(missing_item)}")
        if extra_item:
            errors.append(f"{prefix} unexpected fields: {sorted(extra_item)}")

        pid = item.get("id")
        if not isinstance(pid, str) or not pid.strip():
            errors.append(f"{prefix}.id must be a non-empty string")
        elif pid in seen_ids:
            errors.append(f"duplicate competency id: {pid}")
        else:
            seen_ids.add(pid)

        rating = item.get("rating")
        if rating not in RATINGS:
            errors.append(f"{prefix}.rating is invalid")
        if rating == "N/A":
            if not isinstance(item.get("na_reason"), str) or not item["na_reason"].strip():
                errors.append(f"{prefix}.na_reason is required for N/A")
        elif "na_reason" in item and item["na_reason"] is not None:
            errors.append(f"{prefix}.na_reason must be null/absent unless rating is N/A")

        if not isinstance(item.get("reasoning"), str) or not item["reasoning"].strip():
            errors.append(f"{prefix}.reasoning must be non-empty")
        if item.get("confidence") not in CONFIDENCE:
            errors.append(f"{prefix}.confidence is invalid")
        if not isinstance(item.get("human_review"), bool):
            errors.append(f"{prefix}.human_review must be boolean")

        evidence = item.get("evidence")
        if not isinstance(evidence, list):
            errors.append(f"{prefix}.evidence must be an array")
            continue
        for j, ev in enumerate(evidence):
            ep = f"{prefix}.evidence[{j}]"
            if not isinstance(ev, Mapping):
                errors.append(f"{ep} must be an object")
                continue
            if set(ev) != {"source", "reference", "quote_or_fact"}:
                errors.append(f"{ep} must contain only source, reference, quote_or_fact")
            if ev.get("source") not in EVIDENCE_SOURCES:
                errors.append(f"{ep}.source must be transcript or crm")
            for field in ("reference", "quote_or_fact"):
                if not isinstance(ev.get(field), str) or not ev[field].strip():
                    errors.append(f"{ep}.{field} must be non-empty")

    review = payload.get("human_review")
    if not isinstance(review, Mapping):
        errors.append("human_review must be an object")
    else:
        if set(review) != {"required", "reasons"}:
            errors.append("human_review must contain only required and reasons")
        if not isinstance(review.get("required"), bool):
            errors.append("human_review.required must be boolean")
        reasons = review.get("reasons")
        if not isinstance(reasons, list) or any(not isinstance(x, str) or not x.strip() for x in reasons):
            errors.append("human_review.reasons must be an array of non-empty strings")
        elif review.get("required") is True and not reasons:
            errors.append("human_review.reasons cannot be empty when review is required")

    return errors


def require_valid_modern_evaluation(payload: Any) -> None:
    """Raise ValueError with all validation errors if the payload is invalid."""
    errors = validate_modern_evaluation(payload)
    if errors:
        raise ValueError("Invalid modern audit evaluation: " + "; ".join(errors))
