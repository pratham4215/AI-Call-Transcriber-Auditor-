"""
crm_merger.py — Merge Excel (CALL DATA.xlsx) and JSON sidecar CRM dicts.

Precedence rules
----------------
Excel wins for:
  audit_call_type, disposition, booking, follow_up, remarks, DMS status,
  CRE identity, call metadata, next_service_type, service_due_date,
  campaign, workshop (service profile).

JSON wins for:
  warranty products (oem_status, ew_status_or_expiry, mcp_status),
  pickup_drop_status, odometer_km, variant, fuel_type, sale_date,
  average_km_per_day, last_authorized_service, workshop_code,
  customer address / phone, full service_history from DMS.

Rationale: Excel is the authoritative record of what the CRE did on this
specific call. JSON contains vehicle master data fetched from OEM/DMS that
the CRE never modifies — warranty status, odometer, full service history.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Vehicle fields where JSON (vehicle master) is authoritative; Excel is ignored
_JSON_WINS_VEHICLE = frozenset({
    "variant", "fuel_type", "sale_date",
    "odometer_km", "average_km_per_day", "customer_type",
    "next_service_drop_date",
})

# Service-profile fields from OEM/DMS — JSON wins
_JSON_WINS_SERVICE_PROFILE = frozenset({
    "oem_status", "ew_status_or_expiry", "mcp_status",
    "pickup_drop_status", "workshop_code", "last_authorized_service",
})


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow merge: override wins for any non-None value."""
    result = dict(base)
    for k, v in override.items():
        if v is not None:
            result[k] = v
    return result


def merge_crm(
    excel_crm: Optional[Dict[str, Any]],
    json_crm:  Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Merge Excel and JSON CRM dicts into a single crm_details dict.

    - If both are None   → returns None.
    - If json_crm only   → returns json_crm unchanged.
    - If excel_crm only  → returns excel_crm unchanged (schema_version stays crm-excel-v2).
    - If both present    → returns merged dict with schema_version 'crm-merged-v1'.
    """
    if excel_crm is None and json_crm is None:
        return None
    if excel_crm is None:
        return json_crm
    if json_crm is None:
        return excel_crm

    merged: Dict[str, Any] = {}

    # schema_version — merged enables all deterministic scoring paths
    merged["schema_version"] = "crm-merged-v1"

    # audit_call_type — Excel wins (sheet name is most reliable source)
    merged["audit_call_type"] = (
        excel_crm.get("audit_call_type") or json_crm.get("audit_call_type")
    )

    # crm_source — Excel wins (actual call metadata); JSON has API response meta
    json_cs  = json_crm.get("crm_source") or {}
    excel_cm = excel_crm.get("call_meta") or {}
    merged["crm_source"] = _merge_dicts(json_cs, excel_cm)

    # call_meta — pass through from Excel
    if excel_crm.get("call_meta"):
        merged["call_meta"] = excel_crm["call_meta"]

    # cre — Excel wins (CRE who made this specific call)
    merged["cre"] = _merge_dicts(
        json_crm.get("cre") or {},
        {k: v for k, v in (excel_crm.get("cre") or {}).items() if v is not None},
    )

    # customer — JSON wins for contact details; Excel fills name if JSON has none
    merged_customer = dict(json_crm.get("customer") or {})
    excel_customer  = excel_crm.get("customer") or {}
    if not merged_customer.get("name") and excel_customer.get("name"):
        merged_customer["name"] = excel_customer["name"]
    merged["customer"] = merged_customer

    # vehicle — selective precedence
    merged_vehicle = dict(json_crm.get("vehicle") or {})
    for field, value in (excel_crm.get("vehicle") or {}).items():
        if field in _JSON_WINS_VEHICLE:
            continue
        if value is not None:
            merged_vehicle[field] = value
    merged["vehicle"] = merged_vehicle

    # service_profile — selective precedence
    # JSON wins for warranty/master fields — BUT only when JSON has a real value.
    # If JSON has None for a JSON-wins field, the Excel-derived value is used as fallback.
    merged_sp = dict(json_crm.get("service_profile") or {})
    for field, value in (excel_crm.get("service_profile") or {}).items():
        if field in _JSON_WINS_SERVICE_PROFILE:
            if merged_sp.get(field) is not None:
                continue  # JSON has a real value — keep it
            # JSON is None → fall back to Excel
        if value is not None:
            merged_sp[field] = value
    merged["service_profile"] = merged_sp

    # disposition — Excel wins (what CRE actually entered in CRM)
    json_disp  = json_crm.get("disposition") or {}
    excel_disp = excel_crm.get("disposition") or {}
    merged["disposition"] = (
        _merge_dicts(json_disp, excel_disp)
        if isinstance(json_disp, dict) and isinstance(excel_disp, dict)
        else (excel_disp or json_disp)
    )

    # remarks — Excel wins (live values from this call's form)
    merged["remarks"] = _merge_dicts(
        json_crm.get("remarks") or {},
        excel_crm.get("remarks") or {},
    )

    # booking — Excel wins (authoritative booking record for this call)
    excel_booking = excel_crm.get("booking")
    json_booking  = json_crm.get("booking")
    if excel_booking:
        merged["booking"] = _merge_dicts(json_booking or {}, excel_booking)
    elif json_booking:
        merged["booking"] = json_booking

    # follow_up — Excel wins
    excel_fu = excel_crm.get("follow_up")
    json_fu  = json_crm.get("follow_up")
    if excel_fu:
        merged["follow_up"] = _merge_dicts(json_fu or {}, excel_fu)
    elif json_fu:
        merged["follow_up"] = json_fu

    # service_history — JSON wins (full DMS history); Excel only has SNR snapshot
    json_hist  = json_crm.get("service_history") or []
    excel_hist = excel_crm.get("service_history") or []
    merged["service_history"] = json_hist or excel_hist

    # Pass-through extras
    for extra in ("model_category", "campaign_tagging"):
        val = excel_crm.get(extra) or json_crm.get(extra)
        if val is not None:
            merged[extra] = val

    return merged
