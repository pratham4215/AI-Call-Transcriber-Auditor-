"""
rules.py — Deterministic scoring rules applied BEFORE and AFTER the LLM call.

Functions
─────────────────────────────────────────────────────────────────────────────
  Pre-LLM   compute_param_applicability()  — returns params pre-rated from CRM data
  Post-LLM  apply_business_rule_overrides() — corrects transcript-gated Zeros
  Helpers   _applicability_state(), _is_omission_zero(), _no_active_warranty()
            service_stage_applicability(), build_applicability_note()
  Routing   resolve_audit_route(), normalize_result_payload()
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .params import (
    WEIGHTS,
    _extract_cre_disposition,
    normalize_call_type,
    normalize_transcript_type,
)

# ─────────────────────────────────────────────────────────────────────────────
# APPLICABILITY STATE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _applicability_state(value: Any) -> str:
    """Classify a CRM status value into Active/Applicable, Inactive/Inapplicable, or Unknown."""
    if value in (None, ""):
        return "Unknown"
    text = re.sub(r"[^A-Z0-9]+", " ", str(value).upper()).strip()
    inactive_markers = (
        "INACTIVE", "EXPIRED", "NOT ACTIVE", "NOT AVAILABLE",
        "UNAVAILABLE", "NOT ENROLLED", "NOT APPLICABLE",
    )
    if text in {"N A", "NA", "NONE", "NO"} or any(
        marker in text for marker in inactive_markers
    ):
        return "Inactive/Inapplicable"
    if text in {"YES", "TRUE", "AVAILABLE"} or any(
        marker in text for marker in ("ACTIVE", "VALID", "ENROLLED")
    ):
        return "Active/Applicable"
    return "Unknown"


EARLY_LIFE_MAX_KM = 10000


def service_stage_applicability(details: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return service-stage facts used by both pre- and post-LLM rule checks."""
    details = details or {}
    vehicle = details.get("vehicle", {})
    if not isinstance(vehicle, dict):
        vehicle = {}
    service_type = str(vehicle.get("next_service_type") or "").strip().upper()
    is_free_service = bool(
        re.fullmatch(r"FR\s*[123]", service_type)
        or "FREE SERVICE" in service_type
    )
    raw_odometer = vehicle.get("odometer_km")
    try:
        odometer_km = float(str(raw_odometer).replace(",", ""))
    except (TypeError, ValueError):
        odometer_km = None
    is_low_km = odometer_km is not None and odometer_km <= EARLY_LIFE_MAX_KM
    return {
        "next_service_type": service_type or None,
        "odometer_km": odometer_km,
        "free_service_due": is_free_service,
        "early_life_low_km": is_free_service and is_low_km,
    }


def _no_active_warranty(details: Dict[str, Any]) -> bool:
    """True when ALL warranty products (OEM, EW, MCP) are explicitly inactive."""
    profile = (details or {}).get("service_profile", {})
    if not isinstance(profile, dict):
        return False
    states = [
        _applicability_state(profile.get("oem_status")),
        _applicability_state(profile.get("ew_status_or_expiry")),
        _applicability_state(profile.get("mcp_status")),
    ]
    return all(s == "Inactive/Inapplicable" for s in states)


def build_applicability_note(details: Optional[Dict[str, Any]]) -> str:
    """Build the deterministic applicability block sent to the LLM in the user prompt."""
    details = details or {}
    profile = details.get("service_profile", {})
    if not isinstance(profile, dict):
        profile = {}
    products = {
        "OEM": profile.get("oem_status"),
        "EW": profile.get("ew_status_or_expiry"),
        "MCP": profile.get("mcp_status"),
    }
    states = {name: _applicability_state(value) for name, value in products.items()}
    active = [name for name, state in states.items() if state == "Active/Applicable"]
    unknown = [name for name, state in states.items() if state == "Unknown"]
    if active:
        warranty_instruction = (
            "Confirmed active products: " + ", ".join(active) + ". Only these "
            "confirmed products create a warranty-discussion expectation."
        )
    elif unknown:
        warranty_instruction = (
            "No product is confirmed active and some statuses are unknown. Do not "
            "penalize omission solely because CRM data is missing."
        )
    else:
        warranty_instruction = (
            "All OEM/EW/MCP products are explicitly inactive or inapplicable; a "
            "warranty pitch is not required."
        )
    product_lines = [
        f"- {name}: {states[name]} (CRM value: {products[name]!r})"
        for name in ("OEM", "EW", "MCP")
    ]
    service_stage = service_stage_applicability(details)
    if service_stage["early_life_low_km"]:
        service_instruction = (
            "Early-life free-service vehicle confirmed. Previous-service narration "
            "and asking current KMS are not required; correct omission earns Full."
        )
    else:
        service_instruction = (
            "Early-life/low-KM bypass is not proven; apply normal previous-service "
            "and current-KMS requirements."
        )
    return "\n".join([
        *product_lines,
        f"- Warranty decision: {warranty_instruction}",
        f"- Vehicle stage: next_service_type={service_stage['next_service_type']!r}, "
        f"odometer_km={service_stage['odometer_km']!r}",
        f"- Service-history/KMS decision: {service_instruction}",
    ])


# ─────────────────────────────────────────────────────────────────────────────
# PRE-LLM: compute_param_applicability
# ─────────────────────────────────────────────────────────────────────────────

def compute_param_applicability(
    table: str,
    crm_details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Return params pre-rated by deterministic CRM rules BEFORE LLM scoring.

    Only params whose rating can be proven from call type or CRM data alone —
    no transcript needed — are included. Transcript-gated corrections are
    handled post-LLM in apply_business_rule_overrides.
    """
    auto_full: Dict[str, Dict[str, Any]] = {}
    details = crm_details or {}
    active = set(WEIGHTS.get(table, {}).keys())
    schema_ver = details.get("schema_version", "")

    def _rate(pid: str, rating: str, reason: str, evidence: str) -> None:
        if pid in active:
            auto_full[pid] = {
                "rating": rating,
                "reason": reason,
                "evidence": f"AUTO-SCORE: {evidence}",
                "source": "crm_deterministic",
            }

    def _af(pid: str, reason: str, evidence: str) -> None:
        _rate(pid, "Full", reason, evidence)

    # ── Warranty ─────────────────────────────────────────────────────────────
    if _no_active_warranty(details):
        _af(
            "warranty_info",
            "All warranty products (OEM, EW, MCP) are explicitly inactive",
            "CRM confirms OEM, EW and MCP are all inactive/inapplicable. "
            "No warranty pitch was required; correct omission earns Full.",
        )

    # ── Pickup / drop ─────────────────────────────────────────────────────────
    # BOOKING: always LLM-scored — did CRE mention pickup/drop to the customer?
    # Non-BOOKING: irrelevant parameter, auto-Full.
    profile = details.get("service_profile", {})
    if isinstance(profile, dict):
        if table != "BOOKING":
            _af(
                "pickup_drop",
                f"Pickup/drop not scored for {table} calls",
                f"Table is {table} — free pickup/drop is only audited for BOOKING calls. "
                "Correct non-mention earns Full.",
            )

    # ── Early-life vehicle ────────────────────────────────────────────────────
    stage = service_stage_applicability(details)
    odo_text = (
        f"{stage['odometer_km']:g}" if stage["odometer_km"] is not None else "unknown"
    )
    if stage["early_life_low_km"]:
        _af(
            "prev_service_info",
            "Early-life free-service vehicle — previous-service narration not required",
            f"CRM confirms {stage['next_service_type']} with odometer {odo_text} km "
            "(≤ 10,000 km). Previous-service narration was not required.",
        )
        _af(
            "ask_kms",
            "Early-life free-service vehicle — asking current KMS not required",
            f"CRM confirms {stage['next_service_type']} with odometer {odo_text} km "
            "(≤ 10,000 km). Asking current KMS was not required.",
        )

    # ── Cost estimate: unconditional Full bypass ─────────────────────────────
    _af(
        "cost_estimate",
        "Cost estimate bypassed unconditionally",
        "Cost estimate scoring is bypassed. Full marks awarded for all call types.",
    )

    # ── Conflict management: auto-Full for non-escalation calls ───────────────
    if table != "ESCALATION":
        _af(
            "conflict_mgmt",
            "No conflict expected in non-escalation call",
            f"Table is {table} — conflict management is only critical for ESCALATION calls. "
            "No conflict arose; correct non-use earns Full.",
        )

    # ── CRM-deterministic scoring (Excel and JSON transaction schemas) ──────────
    if schema_ver in ("crm-excel-v1", "crm-txn-v1", "crm-excel-v2", "crm-merged-v1"):

        # Remarks tagging is transcript-dependent and binary. Empty fields are
        # deterministically Zero; otherwise the LLM compares whichever remark(s)
        # are present with the call. Identical CRE/customer text is acceptable.
        remarks = details.get("remarks", {})
        cre_r  = str(remarks.get("cre_remark")      or "").strip()
        cust_r = str(remarks.get("customer_remark") or "").strip()

        def _filled(r: str) -> bool:
            return bool(r) and r not in ("—", "N/A", "None")

        cre_ok  = _filled(cre_r)
        cust_ok = _filled(cust_r)
        if not cre_ok and not cust_ok:
            _rate("remarks_tagging", "Zero",
                  "Both CRE and customer remarks empty",
                  "CRM shows both CRE remark and customer remark fields are empty. Zero.")

        # Proper tagging: CRM Primary Disposition → call type must match audit table
        crm_call_type = details.get("audit_call_type")
        if crm_call_type:
            if crm_call_type == table:
                _rate("proper_tagging", "Full",
                      "CRM disposition matches audit call type",
                      f"CRM Primary Disposition maps to '{crm_call_type}', matches "
                      f"audit table '{table}'. Correct tagging — Full.")
            else:
                _rate("proper_tagging", "Zero",
                      "CRM disposition does not match audit call type",
                      f"CRM Primary Disposition maps to '{crm_call_type}' but audit "
                      f"table is '{table}'. Mis-tagged — Zero.")

        # Task / CRM entry: did CRE fill the correct form for this disposition?
        _score_task_crm(details, table, _rate)

    return auto_full


def _score_task_crm(
    details: Dict[str, Any],
    table: str,
    _rate: Any,
) -> None:
    """Score task_crm deterministically from CRM Excel fields."""
    disp = details.get("disposition", {})
    bk   = details.get("booking", {})
    fu   = details.get("follow_up", {})

    def _val(d: Any, k: str) -> Optional[str]:
        if not isinstance(d, dict):
            return None
        v = d.get(k)
        return str(v).strip() if v is not None and str(v).strip() not in ("", "None") else None

    if table == "BOOKING":
        # Book My Service: workshop + service_type + booking_date_time required
        workshop  = _val(bk, "workshop")
        svc_type  = _val(bk, "service_type")
        bk_dt     = _val(bk, "booking_date_time")
        t_start   = _val(bk, "time_start")
        filled    = sum(1 for f in (workshop, svc_type, bk_dt, t_start) if f)
        if filled >= 3:
            _rate("task_crm", "Full",
                  "Booking task fully entered in CRM",
                  f"CRM booking: workshop='{workshop}', service_type='{svc_type}', "
                  f"date_time='{bk_dt}', time_start='{t_start}'. Task fully filled — Full.")
        elif filled >= 1:
            _rate("task_crm", "Partial",
                  "Booking task partially entered in CRM",
                  f"CRM booking fields partially filled (workshop={workshop}, "
                  f"service_type={svc_type}, date={bk_dt}). Incomplete task — Partial.")
        else:
            _rate("task_crm", "Zero",
                  "No booking task entered in CRM",
                  "CRM shows no booking entry (no workshop, service type, or date). "
                  "Task not created — Zero.")

    elif table == "SERVICE_NT_REQUIRED":
        # SNR: select reason required
        snr_reason = _val(disp, "snr_main_reason")
        if snr_reason:
            _rate("task_crm", "Full",
                  "SNR reason entered in CRM",
                  f"CRM SNR main reason = '{snr_reason}'. Task correctly created — Full.")
        else:
            _rate("task_crm", "Zero",
                  "No SNR reason entered in CRM",
                  "CRM shows no SNR main reason selected. Task not created — Zero.")

    elif table == "FOLLOW_UPS":
        # Call Me Later: follow-up date + reason both required
        fu_date   = _val(fu, "date")
        fu_reason = _val(fu, "reason")
        if fu_date and fu_reason:
            _rate("task_crm", "Full",
                  "Follow-up task fully entered in CRM",
                  f"CRM follow_up date='{fu_date}', reason='{fu_reason}'. Task fully filled — Full.")
        elif fu_date or fu_reason:
            _rate("task_crm", "Partial",
                  "Follow-up task partially entered in CRM",
                  f"CRM follow_up date='{fu_date}', reason='{fu_reason}'. "
                  "One field missing — Partial.")
        else:
            _rate("task_crm", "Zero",
                  "No follow-up task entered in CRM",
                  "CRM shows no follow-up date or reason. Task not created — Zero.")

    elif table == "ESCALATION":
        # Others → ESCALATION: disposition status + sub-reason
        status     = _val(disp, "status")
        sub_reason = _val(disp, "snr_distance_sub_reason") or _val(disp, "snr_main_reason")
        is_esc     = status is not None and "ESCALATION" in status.upper()
        if is_esc and sub_reason:
            _rate("task_crm", "Full",
                  "Escalation task with reason entered in CRM",
                  f"CRM disposition='{status}', reason='{sub_reason}'. Task fully entered — Full.")
        elif is_esc:
            _rate("task_crm", "Partial",
                  "Escalation tagged but no reason entered",
                  f"CRM disposition='{status}' but no escalation reason selected. "
                  "Incomplete task — Partial.")
        else:
            _rate("task_crm", "Zero",
                  "Escalation not tagged in CRM",
                  f"CRM disposition='{status}' — ESCALATION not selected. Task not created — Zero.")


# ─────────────────────────────────────────────────────────────────────────────
# POST-LLM: _is_omission_zero, apply_business_rule_overrides
# ─────────────────────────────────────────────────────────────────────────────

def _is_omission_zero(entry: Dict[str, Any]) -> bool:
    """True when a Zero rating is because the parameter was omitted, not contradicted."""
    if str(entry.get("rating", "")).strip().lower() != "zero":
        return False
    evidence = str(entry.get("evidence", "")).strip().lower()
    contradiction_markers = (
        "incorrect", "inaccurate", "wrong", "false", "contradict",
        "misleading", "overpromise",
    )
    if any(marker in evidence for marker in contradiction_markers):
        return False
    omission_markers = (
        "no evidence", "did not", "didn't", "not ask", "not provide",
        "not provided", "not mention", "no mention", "not discuss",
        "omitted", "missing", "absent", "was not", "were not",
        "not stated", "not given", "not offered", "not present",
        "no indication", "no reference", "not applicable",
        "not raised", "not observed", "cannot find",
        "no explicit", "not explicitly",
        "no cost", "no estimate", "not estimate",
        "no conflict", "conflict did not", "no escalation",
        "no objection", "no argument", "no complaint raised",
        "no anger", "no frustration", "no dispute",
        "no dissatisfaction", "no dissatisfied", "no grievance",
        "no complaint", "not express dissatisfaction",
        "no issue raised", "no issue expressed",
        "no billing", "no higher billing", "no billing concern",
        "not question billing", "not question the bill",
        "not raised billing", "no pricing concern",
        "no cost concern", "no invoice", "not challenge",
    )
    return not evidence or any(marker in evidence for marker in omission_markers)


def apply_business_rule_overrides(
    result: Dict[str, Any],
    table: str,
    details: Dict[str, Any],
    auto_full: Optional[Dict[str, Any]] = None,
) -> None:
    """Apply post-LLM business-rule corrections in place on result['audit']."""
    details = details or {}
    audit = result.get("audit", {})
    if not isinstance(audit, dict):
        return
    applied: List[Dict[str, str]] = []

    service_stage = service_stage_applicability(details)
    odometer_text = (
        f"{service_stage['odometer_km']:g}"
        if service_stage["odometer_km"] is not None else "unknown"
    )

    conditional_rules: List[Tuple[str, bool, str, str]] = [
        (
            "prev_service_info", service_stage["early_life_low_km"],
            "EARLY_LIFE_FREE_SERVICE_PREVIOUS_HISTORY_BYPASS",
            "Correct omission: CRM confirms an early-life free-service vehicle "
            f"({service_stage['next_service_type']}, {odometer_text} km); "
            "previous-service narration was not required.",
        ),
        (
            "ask_kms", service_stage["early_life_low_km"],
            "EARLY_LIFE_FREE_SERVICE_KMS_BYPASS",
            "Correct omission: CRM confirms an early-life free-service vehicle "
            f"({service_stage['next_service_type']}, {odometer_text} km); "
            "asking current KMS was not required.",
        ),
        # dissatisfaction_inquiry and higher_billing only appear in ESCALATION table;
        # if LLM gives Zero due to omission (no dissatisfaction/billing raised), upgrade to Full.
        (
            "dissatisfaction_inquiry", True,
            "NO_DISSATISFACTION_EXPRESSED_BYPASS",
            "Correct omission: the customer did not express dissatisfaction, "
            "a complaint or a grievance during this call. No dissatisfaction "
            "inquiry was required; Full marks awarded.",
        ),
        (
            "higher_billing", True,
            "NO_BILLING_CONCERN_RAISED_BYPASS",
            "Correct omission: the customer did not question billing, pricing, "
            "invoice amount or extra charges. No higher-billing explanation was "
            "required; Full marks awarded.",
        ),
        (
            "warranty_info", _no_active_warranty(details),
            "ALL_WARRANTY_INACTIVE_BYPASS",
            "Correct omission: CRM confirms all warranty products (OEM, EW, MCP) "
            "are explicitly inactive or inapplicable. No warranty pitch was expected.",
        ),
    ]
    for parameter, applies, rule, evidence in conditional_rules:
        entry = audit.get(parameter)
        if applies and isinstance(entry, dict) and _is_omission_zero(entry):
            audit[parameter] = {"rating": "Full", "evidence": evidence}
            applied.append({"parameter": parameter, "rule": rule})

    # Enforce pre-determined CRM ratings — LLM cannot override these.
    if auto_full:
        for pid, info in auto_full.items():
            forced = info.get("rating", "Full")
            entry = audit.get(pid)
            if isinstance(entry, dict) and str(entry.get("rating", "")).strip().lower() != forced.lower():
                audit[pid] = {"rating": forced, "evidence": info["evidence"]}
                applied.append({
                    "parameter": pid,
                    "rule": info.get("source", "crm_deterministic").upper(),
                    "reason": info.get("reason", ""),
                })

    if applied:
        result["business_rule_overrides"] = applied


# ─────────────────────────────────────────────────────────────────────────────
# ROUTING AND NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def resolve_audit_route(
    classification: Dict[str, str],
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Decide which workbook form to use for scoring (audit_type)."""
    details = details or {}
    transcript_type = normalize_transcript_type(classification.get("call_type"))
    cre_disposition = _extract_cre_disposition(details)
    disposition_to_type = {
        "BOOK_MY_SERVICE": "BOOKING",
        "SNR": "SERVICE_NT_REQUIRED",
        "CALL_ME_LATER": "FOLLOW_UPS",
        "ESCALATION": "ESCALATION",
        "DND": "DND",
    }
    crm_type = disposition_to_type.get(cre_disposition)
    if cre_disposition in {"VOICE_MAIL", "NO_CONVERSATION"}:
        return {
            "cre_disposition": cre_disposition,
            "transcript_type": transcript_type,
            "audit_type": None,
            "audit_scope": "No recording - operational disposition only",
            "audit_type_source": "CRE disposition",
            "disposition_match": False,
            "auditor_comment": "No recording is expected for this CRM disposition.",
        }

    supplied_audit_type = details.get("audit_call_type") or details.get("report_type")
    supplied_type = normalize_call_type(supplied_audit_type) if supplied_audit_type else None

    # DND is a safety disposition and must never enter a weighted audit form.
    if crm_type == "DND" or transcript_type == "DND":
        return {
            "cre_disposition": cre_disposition or "Not supplied",
            "transcript_type": transcript_type,
            "audit_type": None,
            "audit_scope": "DND disposition verification only",
            "audit_type_source": "CRE disposition" if crm_type == "DND" else "AI transcript",
            "disposition_match": crm_type in {None, "DND"} and transcript_type == "DND",
            "auditor_comment": (
                "Customer requested no further calls; weighted workbook audit not applicable."
            ),
        }

    # An explicit audit type may resolve an incorrect CRE disposition, but it
    # must agree with the transcript classification.
    if supplied_audit_type:
        if supplied_type != transcript_type:
            return {
                "cre_disposition": cre_disposition or "Not supplied",
                "transcript_type": transcript_type,
                "audit_type": None,
                "audit_scope": "Manual review - routing conflict",
                "audit_type_source": "CRM audit type conflict",
                "disposition_match": False,
                "auditor_comment": (
                    f"CRM audit type {supplied_type} conflicts with transcript type {transcript_type}."
                ),
            }
        base_type = supplied_type
        disposition_match = crm_type is None or crm_type == supplied_type
        audit_type_source = "CRM audit type"
    else:
        # Never silently run the Booking form when transcript evidence says the
        # call is Escalation, Follow-up, or Service Not Required (or vice versa).
        if crm_type and crm_type != transcript_type:
            return {
                "cre_disposition": cre_disposition or "Not supplied",
                "transcript_type": transcript_type,
                "audit_type": None,
                "audit_scope": "Manual review - CRM/transcript routing conflict",
                "audit_type_source": "CRM/transcript conflict",
                "disposition_match": False,
                "auditor_comment": (
                    f"CRE disposition routes to {crm_type}, but transcript evidence routes to "
                    f"{transcript_type}. No weighted form was selected automatically."
                ),
            }
        base_type = crm_type or transcript_type
        disposition_match = True
        audit_type_source = "CRE disposition" if crm_type else "AI transcript fallback"

    route: Dict[str, Any] = {
        "cre_disposition": cre_disposition or "Not supplied",
        "transcript_type": transcript_type,
        "audit_type": base_type if base_type != "DND" else None,
        "audit_scope": "CRE performance score only",
        "audit_type_source": audit_type_source,
        "disposition_match": disposition_match,
        "auditor_comment": "",
    }
    if not disposition_match:
        route["auditor_comment"] = "CRE disposition does not match transcript evidence."
    return route


def normalize_result_payload(result: Dict[str, Any], table: str) -> Dict[str, Any]:
    """Ensure result['audit'] contains exactly the expected parameters."""
    meta = result.setdefault("metadata", {})
    meta["call_type"] = table
    audit = result.setdefault("audit", {})
    if not isinstance(audit, dict):
        audit = {}
    normalized_audit: Dict[str, Dict[str, str]] = {}
    for pid in WEIGHTS[table]:
        entry = audit.get(pid)
        if not isinstance(entry, dict):
            entry = {}
        normalized_audit[pid] = {
            "rating": str(entry.get("rating", "")).strip(),
            "evidence": str(entry.get("evidence", "")).strip(),
        }
    result["audit"] = normalized_audit
    return result
