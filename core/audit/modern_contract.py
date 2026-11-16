"""Strict runtime validation for modern CRE audit model payloads."""
from __future__ import annotations

import re\nfrom typing import Any, List, Mapping

CONTRACT_VERSION = "modern-audit-evaluation-v1"
CALL_TYPES = {"BOOKING", "FOLLOW_UPS", "SERVICE_NT_REQUIRED", "ESCALATION"}
RATINGS = {"Full", "Partial", "Zero", "N/A"}
CONFIDENCE = {"High", "Medium", "Low"}
EVIDENCE_SOURCES = {"transcript", "crm"}
BOOKING_FIELDS = {
    "genuine", "commitment_type", "customer_agreement_confirmed", "agreement_turn_id",
    "final_customer_stance", "final_stance_turn_id", "date_confirmed", "date_turn_id",
    "time_confirmed", "time_turn_id", "address_or_location_confirmed",
    "address_or_location_turn_id", "fake_booking_reason",
}


def validate_modern_evaluation(payload: Any) -> List[str]:
    """Return structural validation errors; no errors means valid structure."""
    errors: List[str] = []
    if not isinstance(payload, Mapping):
        return ["payload must be a JSON object"]

    required_top = {"schema_version", "call_type", "competencies", "human_review"}
    if payload.get("call_type") == "BOOKING":
        required_top.add("booking_genuine")
    missing = required_top - set(payload)
    if missing:
        errors.append(f"missing top-level fields: {sorted(missing)}")
    extras = set(payload) - required_top
    if extras:
        errors.append(f"unexpected top-level fields: {sorted(extras)}")
    if payload.get("schema_version") != CONTRACT_VERSION:
        errors.append(f"schema_version must be {CONTRACT_VERSION!r}")
    call_type = payload.get("call_type")
    if call_type not in CALL_TYPES:
        errors.append("call_type is unsupported")
    elif call_type == "BOOKING":
        bg = payload.get("booking_genuine")
        if not isinstance(bg, Mapping):
            errors.append("booking_genuine must be an object for BOOKING")
        else:
            if set(bg) != BOOKING_FIELDS:
                errors.append("booking_genuine must contain exactly the required booking fields")
            for field in ("genuine", "customer_agreement_confirmed", "date_confirmed", "time_confirmed", "address_or_location_confirmed"):
                if not isinstance(bg.get(field), bool):
                    errors.append(f"booking_genuine.{field} must be boolean")
            if bg.get("commitment_type") not in {
                "date_time_confirmed", "date_only", "existing_booking_confirmed",
                "vague_promise", "declined", "not_discussed", "no_customer_agreement",
            }:
                errors.append("booking_genuine.commitment_type is invalid")
            if bg.get("final_customer_stance") not in {
                "confirmed", "conditional", "declined", "cancelled",
                "disputed", "not_discussed", "unclear",
            }:
                errors.append("booking_genuine.final_customer_stance is invalid")
            if bg.get("fake_booking_reason") not in {
                "none", "explicit_decline", "explicit_cancellation",
                "customer_disputed_booking", "no_customer_agreement",
                "no_booking_discussion", "conditional_or_vague_booking",
            }:
                errors.append("booking_genuine.fake_booking_reason is invalid")
            for field in ("agreement_turn_id", "final_stance_turn_id", "date_turn_id", "time_turn_id", "address_or_location_turn_id"):
                if not isinstance(bg.get(field), str):
                    errors.append(f"booking_genuine.{field} must be a string")

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
        if not evidence:
            errors.append(f"{prefix}.evidence must contain at least one evidence item")
        for j, ev in enumerate(evidence):
            ep = f"{prefix}.evidence[{j}]"
            if not isinstance(ev, Mapping):
                errors.append(f"{ep} must be an object")
                continue
            if set(ev) != {"source", "reference", "quote_or_fact"}:
                errors.append(f"{ep} must contain only source, reference, quote_or_fact")
            if ev.get("source") not in EVIDENCE_SOURCES:
                errors.append(f"{ep}.source must be transcript or crm")
            if ev.get("source") == "transcript" and not re.fullmatch(r"T\d{3}", str(ev.get("reference") or "")):
                errors.append(f"{ep}.reference must be a transcript T### ID")
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
    errors = validate_modern_evaluation(payload)
    if errors:
        raise ValueError("Invalid modern audit evaluation: " + "; ".join(errors))
