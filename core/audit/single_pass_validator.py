"""
single_pass_validator.py — Deterministic post-LLM validation for single-pass audits.

All validation is code — zero LLM retries.  A response that fails validation is
immediately flagged for manual review.  No transcript is ever re-sent.
"""
from __future__ import annotations

import dataclasses
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set

from .single_pass_prompts import CALL_TYPE_PARAM_ORDER
from .params import allowed_score_levels

VALID_RATINGS    = {"Full", "Good", "Partial", "Zero"}
VALID_EVIDENCE   = {"supported", "good", "partial", "absent", "contradicted", "no_trigger"}
VALID_COMMITMENT_TYPES = {
    "date_time_confirmed", "date_only", "existing_booking_confirmed", "vague_promise", "declined",
    "not_discussed", "no_customer_agreement",
}
VALID_FAKE_BOOKING_REASONS = {
    "none", "explicit_decline", "explicit_cancellation",
    "customer_disputed_booking", "no_customer_agreement",
    "no_booking_discussion", "conditional_or_vague_booking",
}
SCHEMA_VERSION   = "audit-single-pass-v3"

# evidence_state → expected rating per derivation rule
EVIDENCE_TO_RATING: Dict[str, str] = {
    "supported":    "Full",
    "good":         "Good",
    "partial":      "Partial",
    "absent":       "Zero",
    "contradicted": "Zero",
    "no_trigger":   "Full",
}


# ─────────────────────────────────────────────────────────────────────────────
# RESULT
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class ValidationResult:
    valid: bool
    errors: List[str] = dataclasses.field(default_factory=list)
    warnings: List[str] = dataclasses.field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_tid_map(turns: List[Any]) -> Dict[str, Any]:
    """Map T001 → Turn for O(1) lookup."""
    return {f"T{t.index:03d}": t for t in turns}


def _is_agent_turn(turn: Any) -> bool:
    spk = (turn.speaker or "").lower()
    return "agent" in spk or spk.startswith("cre")


def _is_customer_turn(turn: Any) -> bool:
    spk = (turn.speaker or "").lower()
    return "customer" in spk


def _normalise_quote_text(text: str) -> str:
    """Normalise harmless Unicode and whitespace differences for comparison."""
    value = unicodedata.normalize("NFKC", text or "").casefold()
    value = value.replace("\u200b", "").replace("\ufeff", "")
    return re.sub(r"\s+", " ", value).strip()


def _normalise_quote_words(text: str) -> str:
    """Return the same text with punctuation reduced to word boundaries."""
    value = _normalise_quote_text(text)
    characters = [
        char if unicodedata.category(char)[:1] in {"L", "M", "N"} else " "
        for char in value
    ]
    return re.sub(r"\s+", " ", "".join(characters)).strip()


def _quote_in_turn(quote: str, turn: Any) -> bool:
    """Check that one quote segment is a substring of the turn text."""
    if not quote:
        return True  # empty quote is only flagged separately
    exact_quote = _normalise_quote_text(quote)
    exact_turn = _normalise_quote_text(turn.text or "")
    if exact_quote in exact_turn:
        return True
    word_quote = _normalise_quote_words(quote)
    word_turn = _normalise_quote_words(turn.text or "")
    return bool(word_quote) and word_quote in word_turn


def _quote_supported_by_turns(quote: str, turns: List[Any]) -> bool:
    """Verify every ellipsis-separated verbatim segment against cited turns.

    Models sometimes join two exact snippets with ``...``.  The join itself is
    not transcript text, so each non-empty segment is checked independently;
    every segment must still occur in at least one cited Agent turn.
    """
    segments = [
        segment.strip()
        for segment in re.split(r"(?:\.{3,}|…+)", quote or "")
        if segment.strip()
    ]
    if not segments:
        return not (quote or "").strip()
    return all(any(_quote_in_turn(segment, turn) for turn in turns) for segment in segments)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

def validate_single_pass_response(
    payload: Dict[str, Any],
    call_type: str,
    turns: List[Any],
    auto_full: Optional[Dict[str, Any]] = None,
    crm_details: Optional[Dict[str, Any]] = None,
) -> ValidationResult:
    """Validate a parsed LLM JSON response deterministically.

    Parameters
    ----------
    payload     : parsed JSON dict (already extracted from LLM text)
    call_type   : deterministic call type (BOOKING / SERVICE_NT_REQUIRED / etc.)
    turns       : list of Turn objects from parse_transcript()
    auto_full   : pre-rated parameters excluded from LLM scoring
    crm_details : normalised CRM dict (used for CRM citation checks)

    Returns a ValidationResult; result.valid == False triggers manual review.
    """
    result = ValidationResult(valid=True)
    auto_pids: Set[str] = set((auto_full or {}).keys())
    tid_map: Dict[str, Any] = _build_tid_map(turns)
    valid_tids: Set[str] = set(tid_map.keys())
    agent_tids: Set[str] = {tid for tid, t in tid_map.items() if _is_agent_turn(t)}
    customer_tids: Set[str] = {tid for tid, t in tid_map.items() if _is_customer_turn(t)}

    # ── 1. Top-level structure ────────────────────────────────────────────────
    if not isinstance(payload, dict):
        result.add_error("Response is not a JSON object.")
        return result

    # ── 2. Required top-level fields ──────────────────────────────────────────
    required_top = ["evidence_ledger", "booking_genuine"]
    for field in required_top:
        if field not in payload:
            result.add_error(f"Missing required top-level field: {field!r}.")

    evidence_ledger: Dict = payload.get("evidence_ledger") or {}
    audit_table: Dict     = payload.get("audit") or {}

    # ── 3. Parameter coverage ─────────────────────────────────────────────────
    all_pids     = CALL_TYPE_PARAM_ORDER.get(call_type, [])
    llm_pids     = [p for p in all_pids if p not in auto_pids]
    llm_pid_set  = set(llm_pids)

    ledger_keys = set(evidence_ledger.keys())
    audit_keys  = set(audit_table.keys())

    # Missing params in evidence_ledger
    missing_ev = llm_pid_set - ledger_keys
    if missing_ev:
        result.add_error(
            f"evidence_ledger missing parameters: {sorted(missing_ev)}."
        )

    # Auto_full params must NOT appear in the model evidence ledger. Legacy v1
    # payloads may still contain an audit table, which is checked if present.
    for pid in auto_pids:
        if pid in ledger_keys:
            result.add_error(
                f"Auto-scored param {pid!r} must NOT appear in evidence_ledger."
            )
        if pid in audit_keys:
            result.add_error(
                f"Auto-scored param {pid!r} must NOT appear in audit."
            )

    # Extra params not in llm_pids (and not in auto_pids)
    extra_ev = ledger_keys - llm_pid_set - auto_pids
    if extra_ev:
        result.add_warning(f"Unexpected parameters in evidence_ledger: {sorted(extra_ev)}.")

    # ── 4. Per-parameter evidence validation ──────────────────────────────────
    for pid in llm_pids:
        ev = evidence_ledger.get(pid)
        au = audit_table.get(pid)
        if not ev:
            continue  # already flagged as missing above

        if not isinstance(ev, dict):
            result.add_error(f"[{pid}] evidence_ledger entry is not a dict.")
            continue
        if au is not None and not isinstance(au, dict):
            result.add_error(f"[{pid}] audit entry is not a dict.")
            continue

        # 4a. evidence_state valid
        estate = ev.get("evidence_state")
        allowed_levels = set(allowed_score_levels(call_type, pid))
        allowed_states = {"supported", "absent", "contradicted", "no_trigger"}
        if "Good" in allowed_levels:
            allowed_states.add("good")
        if "Partial" in allowed_levels:
            allowed_states.add("partial")
        if estate not in allowed_states:
            result.add_error(
                f"[{pid}] invalid evidence_state {estate!r} for this parameter "
                f"(must be one of {sorted(allowed_states)})."
            )

        # 4b. rating valid
        rating = au.get("rating") if isinstance(au, dict) else None
        if rating is not None and rating not in allowed_levels:
            result.add_error(
                f"[{pid}] invalid audit rating {rating!r} "
                f"(must be one of {sorted(allowed_levels)})."
            )

        # 4c. rating-state agreement
        if estate in EVIDENCE_TO_RATING and rating in allowed_levels:
            expected_rating = EVIDENCE_TO_RATING[estate]
            if rating != expected_rating:
                result.add_error(
                    f"[{pid}] rating-state disagreement: evidence_state={estate!r} "
                    f"requires rating={expected_rating!r} but audit has {rating!r}."
                )

        # 4d. agent_turn_ids must exist and belong to Agent turns
        for tid in (ev.get("agent_turn_ids") or []):
            if tid not in valid_tids:
                result.add_error(
                    f"[{pid}] agent_turn_ids contains unknown T-ID {tid!r}."
                )
            elif tid not in agent_tids:
                result.add_warning(
                    f"[{pid}] agent_turn_ids contains {tid!r} which is not an Agent turn."
                )

        # 4e. customer_trigger_turn_ids must exist and belong to Customer turns
        for tid in (ev.get("customer_trigger_turn_ids") or []):
            if tid not in valid_tids:
                result.add_error(
                    f"[{pid}] customer_trigger_turn_ids contains unknown T-ID {tid!r}."
                )
            elif tid not in customer_tids:
                result.add_warning(
                    f"[{pid}] customer_trigger_turn_ids contains {tid!r} which is not a Customer turn."
                )

        # 4f. decision_basis must not be empty
        if not (ev.get("decision_basis") or "").strip():
            result.add_warning(f"[{pid}] decision_basis is empty.")

    # ── 5. booking_genuine (BOOKING only) ─────────────────────────────────────
    needs_genuine = call_type == "BOOKING"
    bg = payload.get("booking_genuine")
    if needs_genuine:
        if bg is None:
            result.add_error("booking_genuine is null but required for BOOKING calls.")
        elif isinstance(bg, dict):
            required_integrity = {
                "genuine", "commitment_type",
                "customer_agreement_confirmed", "agreement_turn_id",
                "final_customer_stance", "final_stance_turn_id",
                "date_confirmed", "date_turn_id", "time_confirmed", "time_turn_id",
                "address_or_location_confirmed", "address_or_location_turn_id",
                "fake_booking_reason",
            }
            missing_integrity = required_integrity - set(bg)
            if missing_integrity:
                result.add_error(
                    f"booking_genuine missing integrity fields: {sorted(missing_integrity)}."
                )
            commitment_type = bg.get("commitment_type", "")
            agreement = bg.get("customer_agreement_confirmed") is True
            agreement_tid = bg.get("agreement_turn_id", "")
            final_stance = bg.get("final_customer_stance", "")
            final_stance_tid = bg.get("final_stance_turn_id", "")
            fake_reason = bg.get("fake_booking_reason")
            if commitment_type not in VALID_COMMITMENT_TYPES:
                result.add_error(
                    f"booking_genuine.commitment_type {commitment_type!r} is invalid."
                )
            for field in (
                "agreement_turn_id", "date_turn_id", "time_turn_id",
                "address_or_location_turn_id", "final_stance_turn_id",
            ):
                tid = bg.get(field, "")
                if tid and tid not in valid_tids:
                    result.add_error(f"booking_genuine.{field} {tid!r} is not a valid T-ID.")
            for fact_field, turn_field in (
                ("date_confirmed", "date_turn_id"),
                ("time_confirmed", "time_turn_id"),
                ("address_or_location_confirmed", "address_or_location_turn_id"),
            ):
                confirmed = bg.get(fact_field) is True
                fact_tid = bg.get(turn_field, "")
                if confirmed and not fact_tid:
                    result.add_error(
                        f"booking_genuine.{turn_field} is required when {fact_field} is true."
                    )
                elif not confirmed and fact_tid:
                    result.add_error(
                        f"booking_genuine.{turn_field} must be empty when {fact_field} is false."
                    )
            if agreement:
                if not agreement_tid:
                    result.add_error("A genuine Customer agreement must cite agreement_turn_id.")
                elif agreement_tid in valid_tids and agreement_tid not in customer_tids:
                    result.add_error("booking_genuine.agreement_turn_id must be a Customer turn.")
            elif agreement_tid:
                result.add_error("agreement_turn_id must be empty when Customer agreement is false.")
            if final_stance not in {
                "confirmed", "conditional", "declined", "cancelled",
                "disputed", "not_discussed", "unclear",
            }:
                result.add_error(
                    f"booking_genuine.final_customer_stance {final_stance!r} is invalid."
                )
            if final_stance_tid and final_stance_tid in valid_tids and final_stance_tid not in customer_tids:
                result.add_error("booking_genuine.final_stance_turn_id must be a Customer turn.")
            if final_stance in {"confirmed", "conditional", "declined", "cancelled", "disputed"} and not final_stance_tid:
                result.add_error(
                    "booking_genuine.final_stance_turn_id is required for a decisive Customer stance."
                )
            if fake_reason not in VALID_FAKE_BOOKING_REASONS:
                result.add_error(f"booking_genuine.fake_booking_reason {fake_reason!r} is invalid.")

            if bg.get("genuine") is False:
                if commitment_type not in {
                    "declined", "not_discussed", "no_customer_agreement",
                    "date_only", "vague_promise",
                }:
                    result.add_error(
                        "booking_genuine may be false only for an explicit decline/cancellation/"
                        "dispute, no booking discussion, or no Customer agreement."
                    )
                if agreement:
                    result.add_error("A booking cannot be fake while Customer agreement is confirmed.")
                if fake_reason in {None, "none"}:
                    result.add_error("A fake booking must include a specific fake_booking_reason.")
            elif not agreement:
                result.add_error("A genuine Booking requires unambiguous Customer agreement.")
            elif final_stance != "confirmed":
                result.add_error("A genuine Booking requires final_customer_stance='confirmed'.")
            elif commitment_type not in {"date_time_confirmed", "date_only", "existing_booking_confirmed"}:
                result.add_error("A genuine Booking requires a direct dated commitment.")
            # A follow-up on a booking made in an earlier call needs no date in this one.
            elif commitment_type != "existing_booking_confirmed" and bg.get("date_confirmed") is not True:
                result.add_error("A genuine Booking requires a confirmed appointment date/day.")
            elif fake_reason != "none":
                result.add_error("A genuine Booking must use fake_booking_reason='none'.")

            detail_count = sum(bool(bg.get(field)) for field in (
                "date_confirmed", "time_confirmed", "address_or_location_confirmed",
            ))
            expected_state = "supported" if detail_count == 3 else "partial" if detail_count == 2 else "absent"
            booking_entry = evidence_ledger.get("booking_details") or {}
            if booking_entry.get("evidence_state") != expected_state:
                result.add_error(
                    "booking_details evidence_state is inconsistent with independently confirmed "
                    f"date/time/location elements: expected {expected_state!r}."
                )
    else:
        if bg is not None:
            result.add_warning(
                "booking_genuine is populated for a non-BOOKING call type (should be null)."
            )

    # ── 6. Disallowed fields (coaching, score, grade) ─────────────────────────
    _forbidden = ["coaching_dialogue", "coaching", "score", "grade", "risk"]
    for f in _forbidden:
        if f in payload:
            result.add_error(
                f"Disallowed field {f!r} present in response — remove it."
            )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CRM FIELD PATH RESOLVER
# ─────────────────────────────────────────────────────────────────────────────

def _crm_field_exists(crm: Any, field_path: str) -> bool:
    """Check that a dot-separated path exists anywhere in the CRM dict."""
    parts = field_path.split(".")
    node: Any = crm
    for part in parts:
        if isinstance(node, dict):
            # Try exact key, then case-insensitive search
            if part in node:
                node = node[part]
            else:
                lower_part = part.lower()
                found = False
                for k, v in node.items():
                    if k.lower() == lower_part:
                        node = v
                        found = True
                        break
                if not found:
                    return False
        else:
            # Reached a leaf before consuming the path
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# RATING DERIVATION (called by pipeline after validation passes)
# ─────────────────────────────────────────────────────────────────────────────

def derive_ratings_from_evidence(
    payload: Dict[str, Any],
    call_type: str,
    auto_full: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Return {pid: rating} for all params, merging auto_full and LLM output.

    If the LLM rating disagrees with the evidence_state derivation, the
    evidence_state takes precedence (pipeline already logged the disagreement
    as a validation warning/error before calling this).
    """
    result: Dict[str, str] = {}

    # Auto-scored params
    for pid, info in (auto_full or {}).items():
        result[pid] = info.get("rating", "Full")

    # LLM-scored params — enforce evidence_state derivation
    evidence_ledger = payload.get("evidence_ledger") or {}
    audit_table     = payload.get("audit") or {}

    all_pids = CALL_TYPE_PARAM_ORDER.get(call_type, [])
    auto_pids = set((auto_full or {}).keys())

    for pid in all_pids:
        if pid in auto_pids:
            continue
        ev = evidence_ledger.get(pid)
        au = audit_table.get(pid)
        if ev and isinstance(ev, dict):
            estate = ev.get("evidence_state")
            if estate in EVIDENCE_TO_RATING:
                derived = EVIDENCE_TO_RATING[estate]
                # Escalation current-service bypass is deliberately stricter
                # than generic no-trigger handling. It requires a Customer turn
                # and an explicit inapplicability decision marker from the
                # parameter-specific prompt; ordinary omission remains Zero.
                if call_type == "ESCALATION" and pid == "current_service_info" and estate == "no_trigger":
                    basis = str(ev.get("decision_basis") or "").strip().upper()
                    customer_turns = ev.get("customer_trigger_turn_ids") or []
                    derived = (
                        "Full"
                        if basis.startswith("CURRENT_SERVICE_INAPPLICABLE:") and customer_turns
                        else "Zero"
                    )
                result[pid] = (
                    derived
                    if derived in allowed_score_levels(call_type, pid)
                    else "Zero"
                )
                continue
        if au and isinstance(au, dict):
            rating = au.get("rating")
            if rating in allowed_score_levels(call_type, pid):
                result[pid] = rating
                continue
        # Fallback: no info → Zero (conservative)
        result[pid] = "Zero"

    return result
