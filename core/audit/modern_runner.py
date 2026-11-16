"""One-call modern audit runner (model evaluation once; Python scoring after)."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .llm import extract_json
from .modern_adapter import evaluate_payload
from .modern_prompt import build_modern_evaluation_prompt
from .pipeline import booked_on_earlier_call, enforce_booking_integrity


def run_modern_audit(
    client: Any,
    turns: Sequence[Any],
    call_type: str,
    crm_details: Mapping[str, Any] | None = None,
    **integrity_flags: Any,
) -> dict[str, Any]:
    """Run one structured LLM evaluation and deterministic post-processing.

    `client` must expose LLMClient.generate(system, user, schema, single_pass=True).
    Booking genuineness is reconciled by the existing deterministic booking rule
    after this same model response; there is no second transcript/model pass here.
    """
    system, user, schema = build_modern_evaluation_prompt(turns, call_type, crm_details)
    raw = client.generate(system, user, schema, single_pass=True)
    payload = extract_json(raw)

    if call_type == "BOOKING":
        booking = payload.get("booking_genuine")
        if not isinstance(booking, dict):
            raise ValueError("Booking evaluation must include booking_genuine")
        enforce_booking_integrity(
            payload, list(turns), booked_on_earlier_call(dict(crm_details or {}))
        )
        integrity_flags["fake_booking"] = bool(integrity_flags.get("fake_booking") or booking.get("genuine") is False)
        if booking.get("too_short") is True:
            integrity_flags["low_confidence_critical_decision"] = True
        # The narrow legacy contradiction adjudication is deliberately not
        # another LLM call in modern mode; route it to human review instead.
        from .pipeline import booking_needs_adjudication
        if booking_needs_adjudication(payload, list(turns), booked_on_earlier_call(dict(crm_details or {}))):
            integrity_flags["low_confidence_critical_decision"] = True
            integrity_flags["contradictory_evidence"] = True

    if call_type == "BOOKING":\n        payload.pop("evidence_ledger", None)\n        payload.pop("trigger_ledger", None)\n    result = evaluate_payload(payload, **integrity_flags)
    if call_type == "BOOKING":
        result["booking_genuine"] = {
            key: value for key, value in payload["booking_genuine"].items()
            if key in {
                "genuine", "commitment_type", "customer_agreement_confirmed",
                "agreement_turn_id", "final_customer_stance", "final_stance_turn_id",
                "date_confirmed", "date_turn_id", "time_confirmed", "time_turn_id",
                "address_or_location_confirmed", "address_or_location_turn_id",
                "fake_booking_reason", "too_short",
            }
        }
        result["booking_genuine_evidence"] = payload["booking_genuine"].get("evidence", "")
    return result
