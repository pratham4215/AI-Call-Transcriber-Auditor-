"""
crm_call_data.py — CRM loader for data/CALL DATA.xlsx, indexed by registration number.

Covers all four call types across sheets:
  BOOKING             → BOOKING
  FOLLOW UPS          → FOLLOW_UPS
  SERVICE NT REQUIRED → SERVICE_NT_REQUIRED
  ESCALATION          → ESCALATION

Column-label anomaly in the source Excel (documented once, handled internally):
  'cre_name' column = employee code  (e.g. 'EMP_A_01')
  'cre_id'   column = display name   (e.g. 'Firstname Lastname')
  Labels are inverted; we map correctly to cre.username and cre.name.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import openpyxl  # type: ignore
except ImportError:
    openpyxl = None  # type: ignore

from .utils import mask_phone

logger = logging.getLogger(__name__)

_DEFAULT_EXCEL = Path(__file__).resolve().parents[2] / "data" / "CALL DATA.xlsx"
CALL_DATA_EXCEL_PATH = Path(os.environ.get("CALL_DATA_EXCEL_PATH", str(_DEFAULT_EXCEL)))

# Sheet name → internal call type used by audit tables
_SHEET_TO_CALL_TYPE: Dict[str, str] = {
    "BOOKING":             "BOOKING",
    "FOLLOW UPS":          "FOLLOW_UPS",
    "SERVICE NT REQUIRED": "SERVICE_NT_REQUIRED",
    "ESCALATION":          "ESCALATION",
}

_NSD_TRIGGER_LABELS: Dict[str, str] = {
    "TENURE":  "calendar-based (tenure)",
    "MILEAGE": "odometer-based (mileage)",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_reg(reg: Any) -> str:
    """Uppercase, strip whitespace and hyphens from a registration number."""
    return re.sub(r"[\s\-]", "", str(reg or "")).upper().strip()


def _s(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _dt(v: Any) -> Optional[str]:
    if isinstance(v, (datetime, date)):
        return v.strftime("%Y-%m-%d")
    return _s(v)


# ── Row → crm_details ─────────────────────────────────────────────────────────

def _row_to_crm(row: Dict[str, Any], call_type: str) -> Dict[str, Any]:
    """
    Map a CALL DATA.xlsx row (snake_case headers) to the audit's stable CRM schema.

    Handles the inverted cre_name/cre_id column labels: cre_name holds the
    employee code; cre_id holds the display name.
    """
    nsd_type    = _s(row.get("nsd_due_type"))
    nsd_trigger = _s(row.get("nsd_genrated_by"))
    pickup      = _s(row.get("pick_up_option"))
    nsd_due     = _dt(row.get("nsd_due_date"))

    nsd_parts: List[str] = []
    if nsd_trigger:
        nsd_parts.append(_NSD_TRIGGER_LABELS.get(nsd_trigger.upper(), nsd_trigger))
    if nsd_type:
        nsd_parts.append(f"service type {nsd_type}")
    if nsd_due:
        nsd_parts.append(f"due {nsd_due}")
    nsd_context = "; ".join(nsd_parts) or None

    # SNR last-service (used as service_history entry for SNR/ESCALATION calls)
    service_history: List[Dict[str, Any]] = []
    last_svc_date = _dt(row.get("snr_last_service_date"))
    if last_svc_date:
        service_history.append({
            "service_date": last_svc_date,
            "workshop":     (_s(row.get("snr_last_service_workshop"))
                             or _s(row.get("authorized_dealer_name"))
                             or _s(row.get("unauthorized_workshop_name"))),
            "service_type": _s(row.get("snr_workshop_type")),
            "odometer_km":  None,
        })

    # Disposition
    disposition: Dict[str, Any] = {
        "status":                  _s(row.get("primary_disposition")),
        "type":                    _s(row.get("disposition_form_type")),
        "snr_main_reason":         _s(row.get("snr_main_reason")),
        "snr_distance_sub_reason": _s(row.get("snr_distance_sub_reason")),
        "snr_workshop_type":       _s(row.get("snr_workshop_type")),
        "authorized_dealer":       _s(row.get("authorized_dealer_name")),
        "unauthorized_workshop":   _s(row.get("unauthorized_workshop_name")),
    }

    # Booking
    booking: Optional[Dict[str, Any]] = None
    bk_dt  = _dt(row.get("booking_date_time"))
    bk_svc = _s(row.get("booked_service_type"))
    bk_loc = _s(row.get("booking_workshop_location"))
    if bk_dt or bk_svc or bk_loc:
        booking = {
            "booking_date_time": bk_dt,
            "service_date":      bk_dt,
            "service_type":      bk_svc,
            "workshop":          bk_loc,
            "time_start":        _s(row.get("booking_time_start")),
            "time_end":          _s(row.get("booking_time_end")),
            "status":            _s(row.get("booked_status")),
            "pick_up_option":    pickup,
            # CRM database only: False = booked on an earlier call that this one follows up
            "made_in_this_call": row.get("booked_this_call"),
        }

    # Follow-up
    follow_up: Optional[Dict[str, Any]] = None
    fu_date   = _dt(row.get("follow_up_date"))
    fu_reason = _s(row.get("follow_up_reason"))
    if fu_date or fu_reason:
        follow_up = {"date": fu_date, "reason": fu_reason}

    crm: Dict[str, Any] = {
        "schema_version": "crm-excel-v2",
        "cre": {
            # Column labels are inverted in the source Excel:
            #   cre_id   = display name  (e.g. "Firstname Lastname")
            #   cre_name = employee code (e.g. "EMP_B_02")
            "name":        _s(row.get("cre_id")),
            "username":    _s(row.get("cre_name")),
            "employee_id": _s(row.get("cre_name")),
        },
        "customer": {
            "name": _s(row.get("customer_name")),
        },
        "vehicle": {
            "model":               _s(row.get("model")),
            "model_category":      _s(row.get("model_category")),
            "registration_number": _s(row.get("registration_number")),
            "vin_number":          _s(row.get("vin_number")),
            "engine_number":       _s(row.get("engine_number")),
            "chassis_number":      _s(row.get("chassis_number")),
            "service_due_date":    nsd_due,
            "next_service_type":   nsd_type,
            "nsd_trigger":         nsd_trigger,
            "nsd_context":         nsd_context,
        },
        "service_profile": {
            "campaign":                  _s(row.get("call_campaign")),
            "campaign_tagging":          _s(row.get("campaign_tagging")),
            "workshop":                  _s(row.get("work_shop_name")),
            # PICKUP_DROP_REQUIRED → pickup was offered and booked; CUSTOMER_DRIVE_IN → unknown
            "pickup_drop_status":        ("YES" if pickup == "PICKUP_DROP_REQUIRED" else None),
            "dms_verified":              _s(row.get("is_verified_with_dms")),
            "snr_last_service_workshop": _s(row.get("snr_last_service_workshop")),
            "snr_last_service_date":     last_svc_date,
        },
        "call_meta": {
            "call_date":      _dt(row.get("call_date")),
            "call_duration":  _s(row.get("call_duration")),
            "submitted_date": _dt(row.get("submitted_date")),
        },
        "remarks": {
            "cre_remark":      mask_phone(_s(row.get("cre_remark")) or ""),
            "customer_remark": mask_phone(_s(row.get("customer_remark")) or ""),
        },
        "disposition":   disposition,
        "service_history": service_history,
        "audit_call_type": call_type,
    }

    if booking:
        crm["booking"] = booking
    if follow_up:
        crm["follow_up"] = follow_up

    return crm


# ── Cache & public API ────────────────────────────────────────────────────────

# normalized_reg → (crm_dict, sheet_name)
_CACHE: Optional[Dict[str, Tuple[Dict[str, Any], str]]] = None
_CACHE_LOCK = threading.RLock()


def _load() -> Dict[str, Tuple[Dict[str, Any], str]]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    if openpyxl is None:
        logger.warning("[CRM_CALL_DATA] openpyxl not installed — CALL DATA.xlsx not loaded")
        _CACHE = {}
        return _CACHE

    if not CALL_DATA_EXCEL_PATH.exists():
        logger.debug("[CRM_CALL_DATA] Excel not found at %s", CALL_DATA_EXCEL_PATH)
        _CACHE = {}
        return _CACHE

    wb = openpyxl.load_workbook(str(CALL_DATA_EXCEL_PATH), read_only=True, data_only=True)
    index: Dict[str, Tuple[Dict[str, Any], str]] = {}

    for sheet_name, call_type in _SHEET_TO_CALL_TYPE.items():
        if sheet_name not in wb.sheetnames:
            logger.debug("[CRM_CALL_DATA] Sheet '%s' not in workbook", sheet_name)
            continue

        ws = wb[sheet_name]
        headers: Optional[List[str]] = None
        rows_loaded = 0

        for row_values in ws.iter_rows(values_only=True):
            if headers is None:
                headers = [
                    str(h).strip().lower().replace(" ", "_") if h is not None else f"_col{i}"
                    for i, h in enumerate(row_values)
                ]
                continue

            # Stop on first completely empty row (avoids scanning Excel's 1M row limit)
            if all(v is None for v in row_values):
                break

            row = {headers[i]: row_values[i]
                   for i in range(min(len(headers), len(row_values)))}

            raw_reg = row.get("registration_number")
            if not raw_reg:
                continue
            norm = normalize_reg(raw_reg)
            if not norm:
                continue

            if norm in index:
                existing_sheet = index[norm][1]
                logger.warning(
                    "[CRM_CALL_DATA] Duplicate registration '%s' in sheets '%s' and '%s'. "
                    "Keeping first match (sheet '%s').",
                    raw_reg, existing_sheet, sheet_name, existing_sheet,
                )
                continue

            index[norm] = (_row_to_crm(row, call_type), sheet_name)
            rows_loaded += 1

        logger.debug("[CRM_CALL_DATA] Sheet '%s': %d records loaded", sheet_name, rows_loaded)

    wb.close()
    _CACHE = index
    logger.info("[CRM_CALL_DATA] Total %d records loaded from %s", len(index), CALL_DATA_EXCEL_PATH)
    return _CACHE


def reload() -> int:
    """Force re-read from disk. Returns number of rows loaded."""
    global _CACHE
    with _CACHE_LOCK:
        _CACHE = None
        return len(_load())


def peek(stem: str) -> Optional[Dict[str, Any]]:
    """crm_details for a registration without logging a miss (crm_db picks calls with it)."""
    with _CACHE_LOCK:
        hit = _load().get(normalize_reg(stem))
    return hit[0] if hit else None


def get_crm_by_stem(stem: str) -> Optional[Tuple[Dict[str, Any], str]]:
    """
    Look up a CRM row by registration number stem (e.g. 'GJ01AB1234').

    Registration numbers are normalized before matching (uppercase, no spaces
    or hyphens) so 'gj-01-ab-1234', 'GJ01AB1234', and 'GJ 01 AB 1234' all match.

    Returns (crm_dict, sheet_name) or None.
    Logs a warning when no match is found.
    """
    norm = normalize_reg(stem)
    if not norm:
        return None
    with _CACHE_LOCK:
        result = _load().get(norm)
    if result is None:
        logger.warning(
            "[CRM_CALL_DATA] No row found for stem '%s' (normalized: '%s')", stem, norm
        )
    return result
