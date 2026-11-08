"""
crm_db.py — everything the audit needs about a call, from the CRM database (crmdb).

A call is one lead_transaction row (what the CRE did on one call). For it:

  call data       lead_transaction + its disposition form (booked_service /
                  call_later / service_not_required_disposition /
                  other_disposition) + users, service_station, campaigns.
                  Columns are named like CALL DATA.xlsx (an export of these
                  tables), so crm_call_data._row_to_crm maps them unchanged.
  vehicle master  lead (lead_drop once the lead is dropped) + service history
                  from kst_billing_data, shaped like the CRM API response in
                  the data/ JSON files, so normalize_crm_details reads it.

Recording stems: 'GJ01AB1234_12345678' (registration + bizRTC CDR id, as the
importer names them) is that exact call. A bare registration 'GJ01AB1234' (the
hand-made data/ set) resolves to one of the vehicle's calls: the call CALL
DATA.xlsx describes for it (same call time) when the Excel has the vehicle,
else the latest call with a recording.

CRM_SOURCE: auto → DB when configured, CALL DATA.xlsx on a miss or DB error;
db → DB only; excel → CALL DATA.xlsx only.
"""
from __future__ import annotations

import logging
import re
import threading
import time as _time
import uuid
from datetime import date, datetime, time, timedelta, tzinfo
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

from config import CRM_SOURCE, DB_TIMEZONE
from core import db

from . import crm_call_data
from .crm_call_data import _row_to_crm, normalize_reg

logger = logging.getLogger(__name__)

# Reported as the CRM context source in audit results (no host or credentials).
SOURCE_PATH = Path("crmdb.lead_transaction")

CALL_TYPES = ("BOOKING", "FOLLOW_UPS", "SERVICE_NT_REQUIRED", "ESCALATION")
# lead_transaction.lead_type: which programme the vehicle's lead belongs to.
LEAD_TYPES = ("UNNATI", "PRAGATI")
HISTORY_LIMIT = 10      # service-history entries per vehicle
_CACHE_TTL = 300.0      # seconds a stem → call lookup is reused


def _zone(name: str) -> Optional[tzinfo]:
    if not name:
        return None
    try:
        return datetime.strptime(name, "%z").tzinfo  # "+05:30"
    except ValueError:
        pass
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name)
    except Exception:
        logger.warning("[CRM_DB] Unknown DB_TIMEZONE %r; timestamps keep the server zone", name)
        return None


_TZ = _zone(DB_TIMEZONE)


class CallData(NamedTuple):
    crm: Dict[str, Any]                       # call data, CALL DATA.xlsx shape
    source: Path                              # reported as the CRM context source
    vehicle: Optional[Dict[str, Any]] = None  # vehicle master, CRM API response shape
    transaction_id: Optional[str] = None
    cdr_id: Optional[str] = None              # for the recording decrypt API
    picked: str = ""                          # how the call was chosen for a recording


def enabled() -> bool:
    return CRM_SOURCE != "excel" and db.is_configured()


# ── SQL ───────────────────────────────────────────────────────────────────────

# Dispositioned calls only; is_deleted is the CRM's 'TRUE'/'FALSE' text flag.
_CALL_FILTER = "lt.is_deleted = 'FALSE' AND lt.call_date IS NOT NULL AND lt.disposition_form_type IS NOT NULL"

_CALL_TYPE_SQL = """(CASE upper(lt.disposition_form_type)
    WHEN 'BOOKED_SERVICE' THEN 'BOOKING'
    WHEN 'BOOK_MY_SERVICE' THEN 'BOOKING'
    WHEN 'CALL_LATER' THEN 'FOLLOW_UPS'
    WHEN 'SERVICE_NOT_REQUIRED' THEN 'SERVICE_NT_REQUIRED'
    -- Only the call's own status: nightly re-allocation rows (status FRESH) keep
    -- pointing at the lead's earlier escalation form.
    WHEN 'OTHER_DISPOSITION' THEN CASE WHEN upper(lt.status) = 'ESCALATION' THEN 'ESCALATION' END
END)"""

# The booking form was created when this call was saved (same CRE action).
_BOOKED_THIS_CALL = ("bs.created_at BETWEEN lt.created_at - interval '15 minutes' "
                     "AND lt.created_at + interval '15 minutes'")

# Aliases are the CALL DATA.xlsx headers, including its swapped CRE labels
# (cre_name = employee code, cre_id = display name) that _row_to_crm expects.
_CALL_SELECT = f"""SELECT
    lt.id AS transaction_id, lt.lead_id,
    cr.cdr_id AS cdr_id,          -- bizRTC's call id, what the recording decrypt API takes
    cr.disposition AS cdr_disposition,
    cr.call_duration AS cdr_call_duration,  -- bizRTC's "HH:MM:SS"
    lt.crm_cdr_id,                -- the CRM's own id for the call record
    (lt.recordingfile IS NOT NULL AND lt.recordingfile <> '') AS has_recording,
    lt.audit_status AS crm_audit_status,
    {_CALL_TYPE_SQL} AS _call_type,
    upper(lt.lead_type) AS lead_type,  -- UNNATI / PRAGATI
    c.name AS call_campaign, lt.campaign_id::text AS campaign_id,
    lt.registration_number, lt.vin_number, lt.engine_number, lt.chassis_number,
    lt.call_date,
    cre.user_name AS cre_name, cre.full_name AS cre_id,
    lt.created_at AS submitted_date,
    sub.user_name AS submitted_cre_id, sub.full_name AS submitted_cre_name,
    lt.model, lt.customer_name,
    ss.name AS work_shop_name,
    lt.nsd_source AS nsd_genrated_by,
    lt.next_service_date AS nsd_due_date,
    lt.next_service_type AS nsd_due_type,
    lt.call_duration,
    lt.status AS primary_disposition,
    lt.disposition_form_type,
    coalesce(lt.follow_up_date, cl.follow_up_date_time) AS follow_up_date,
    coalesce(lt.follow_up_reason, cl.follow_up_reason) AS follow_up_reason,
    snr.main_reason AS snr_main_reason,
    snr.distance_sub_reason AS snr_distance_sub_reason,
    snr.last_service_workshop AS snr_last_service_workshop,
    snr.workshop_type AS snr_workshop_type,
    -- CALL DATA.xlsx fills this for every call type, not just SNR forms
    coalesce(snr.last_service_date, lt.last_service_date) AS snr_last_service_date,
    snr.authorized_dealer_name, snr.unauthorized_workshop_name,
    snr.verified_with_dms AS is_verified_with_dms,
    lt.booking_date_time, lt.booked_service_type, lt.booking_workshop_location,
    bs.booking_time_start, bs.booking_time_end, bs.booked_status,
    bs.service_delivery_option AS pick_up_option,
    -- false: the booking form was made on an earlier call, which this call follows up
    ({_BOOKED_THIS_CALL}) AS booked_this_call,
    -- A booking made on this call can hold the CRE's remarks only in its form;
    -- remarks added to the booking on a later call never count for this one.
    coalesce(nullif(btrim(lt.cre_remark), ''),
             CASE WHEN {_BOOKED_THIS_CALL} THEN nullif(btrim(bs.cre_remarks), '') END) AS cre_remark,
    coalesce(nullif(btrim(lt.customer_remark), ''),
             CASE WHEN {_BOOKED_THIS_CALL} THEN nullif(btrim(bs.customer_remarks), '') END) AS customer_remark,
    lt.campaign_tagging,
    od.escalation_reason
FROM lead_transaction lt
LEFT JOIN users cre ON cre.user_id = lt.assigned_cre
LEFT JOIN users sub ON sub.user_id = lt.created_by
LEFT JOIN campaigns c ON c.id = lt.campaign_id
LEFT JOIN service_station ss ON ss.id = lt.service_station_id
LEFT JOIN booked_service bs
       ON lt.disposition_form_type = 'BOOKED_SERVICE' AND bs.id = lt.disposition_id
LEFT JOIN call_later cl
       ON lt.disposition_form_type = 'CALL_LATER' AND cl.id = lt.disposition_id
LEFT JOIN service_not_required_disposition snr
       ON lt.disposition_form_type = 'SERVICE_NOT_REQUIRED' AND snr.id = lt.disposition_id
LEFT JOIN other_disposition od
       ON lt.disposition_form_type = 'OTHER_DISPOSITION' AND od.id = lt.disposition_id
LEFT JOIN cdr_records cr ON cr.id = lt.cdr_id"""

_LEAD_COLUMNS = (
    "registration_number", "vin_number", "model", "customer_name", "mobile_number",
    "customer_address", "next_service_date", "next_service_drop_date", "next_service_type",
    "odo_meter_reading", "avr_running_per_day", "last_service_date", "last_service_type",
    "sale_date", "sale_dealer", "oem_status", "oem_expiry_date", "mcp_status", "mcp_expiry_date",
    "ew_status", "ew_expiry_date", "insurance_status", "insurance_expiry_date", "customer_type",
    "kcp_plan_name", "kcp_expiry_date", "kcp_expiry_km", "lead_type", "dnd_status",
)


def _lead_select(table: str, id_column: str, dropped_date: str) -> str:
    columns = ", ".join(f"x.{name}" for name in _LEAD_COLUMNS)
    return f"""SELECT '{table}' AS origin, x.{id_column} AS lead_id, {dropped_date} AS dropped_date, {columns},
        ss.name AS service_station_name, ss.ws_code, c.name AS campaign_name,
        u.full_name AS assigned_cre_name, u.user_name AS assigned_cre_code
    FROM {table} x
    LEFT JOIN service_station ss ON ss.id = x.service_station_id
    LEFT JOIN campaigns c ON c.id = x.campaign_id
    LEFT JOIN users u ON u.user_id = x.assigned_cre
    WHERE x.{id_column} = ANY(CAST(:ids AS uuid[]))"""


# A lead moves to lead_drop (keeping its id in lead_drop.lead) when it is dropped.
_VEHICLE_SQL = (
    _lead_select("lead", "id", "NULL::timestamptz")
    + "\nUNION ALL\n"
    + _lead_select("lead_drop", "lead", "x.dropped_date")
)

_HISTORY_SQL = """SELECT upper(registration_number) AS reg, bill_date_time, job_card_date_time,
       service_type, bill_amount, station, city, mileage, service_adviser,
       job_card_number, variant
FROM kst_billing_data
WHERE is_deleted = 'FALSE' AND upper(registration_number) = ANY(:regs)
ORDER BY job_card_date_time DESC NULLS LAST"""


# ── Value formatting ──────────────────────────────────────────────────────────

def _local(value: datetime) -> datetime:
    if value.tzinfo is not None and _TZ is not None:
        value = value.astimezone(_TZ)
    return value.replace(tzinfo=None)


def _pg_array_text(items: List[Any]) -> str:
    """Array as CALL DATA.xlsx shows it: '{REZO}', '{FLEET DATA,PMS 2 3 4 5}'."""
    return "{" + ",".join("" if item is None else str(item) for item in items) + "}"


def _as_export_text(value: Any) -> Any:
    """DB value → the text CALL DATA.xlsx holds for it ('2026-07-06 12:30:30', '{REZO}')."""
    if isinstance(value, datetime):
        # _row_to_crm cuts datetime objects to the date, which drops booking times.
        return _local(value).strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, (date, time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        seconds = int(value.total_seconds())
        return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, list):
        return _pg_array_text(value)
    return value


def _dmy(value: Any) -> Optional[str]:
    """DD-MM-YYYY, the CRM API's date format (the pipeline parses warranty dates in it)."""
    if isinstance(value, datetime):
        value = _local(value)
    if isinstance(value, (datetime, date)):
        return value.strftime("%d-%m-%Y")
    return None


def _export_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {str(key).lower(): _as_export_text(value) for key, value in row.items()}


# ── Row builders ──────────────────────────────────────────────────────────────

def _call_crm(row: Dict[str, Any]) -> Dict[str, Any]:
    # An unmapped disposition leaves audit_call_type empty, so the pipeline
    # falls back to the filename / folder like it does for JSON-only calls.
    crm = _row_to_crm(row, row.get("_call_type"))
    crm["call_meta"].update(transaction_id=row.get("transaction_id"), cdr_id=row.get("cdr_id"),
                            lead_type=row.get("lead_type"))
    if row.get("escalation_reason"):
        crm["disposition"]["escalation_reason"] = row["escalation_reason"]
    return crm


def _history_entry(bill: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "lastServiceDate":     _dmy(bill.get("bill_date_time") or bill.get("job_card_date_time")),
        "lastServiceType":     bill.get("service_type"),
        "lastBillAmount":      bill.get("bill_amount"),
        "lastServiceStation":  bill.get("station"),
        "lastServiceCity":     bill.get("city"),
        "lastOdometerReading": bill.get("mileage"),
        "jobCardNumber":       bill.get("job_card_number"),
        "serviceAdvisorName":  bill.get("service_adviser"),
    }


def _status_on(status: Any, expiry: Any, on: Optional[date]) -> Any:
    """ACTIVE/INACTIVE as of the call date when the expiry is known; the lead's flag lags expiry."""
    if expiry is None or on is None or str(status or "").upper() not in ("", "ACTIVE", "INACTIVE"):
        return status
    expires = _local(expiry).date() if isinstance(expiry, datetime) else expiry
    return "ACTIVE" if expires >= on else "INACTIVE"


def _vehicle_payload(lead: Dict[str, Any], bills: List[Dict[str, Any]], on: Optional[date] = None) -> Dict[str, Any]:
    """Vehicle master in the CRM API response shape of the data/ JSON files (statuses as of `on`)."""
    latest_bill = bills[0] if bills else {}
    nsd = {
        "nextServiceDate":     _dmy(lead.get("next_service_date")),
        "nextServiceDropDate": _dmy(lead.get("next_service_drop_date")),
        "nextServiceType":     lead.get("next_service_type"),
    }
    vehicle_ids = {
        "registrationNumber": lead.get("registration_number"),
        "vinNumber":          lead.get("vin_number"),
        "model":              lead.get("model"),
    }
    return {
        "code": 200,
        "message": f"crmdb.{lead['origin']}",
        "data": {
            "lead": {
                **vehicle_ids, **nsd,
                "customerName":       lead.get("customer_name"),
                "odoMeterReading":    lead.get("odo_meter_reading"),
                "avrRunningPerDay":   lead.get("avr_running_per_day"),
                "campaignName":       lead.get("campaign_name"),
                "serviceStationName": lead.get("service_station_name"),
                "assignedCREName":    lead.get("assigned_cre_name"),
                "assignedCREId":      lead.get("assigned_cre_code"),
            },
            "vehicleDetails": {
                **vehicle_ids, **nsd,
                "variant":         latest_bill.get("variant"),
                "fuelType":        None,  # not stored in crmdb
                "saleDateTime":    _dmy(lead.get("sale_date")),
                "city":            latest_bill.get("city"),
                "odometerReading": lead.get("odo_meter_reading"),
                "nsdWorkshop":     lead.get("ws_code"),
                "oemStatus":       _status_on(lead.get("oem_status"), lead.get("oem_expiry_date"), on),
                "oemWarrantyDate": _dmy(lead.get("oem_expiry_date")),
                "mcpStatus":       _status_on(lead.get("mcp_status"), lead.get("mcp_expiry_date"), on),
                "mcpStatusDate":   _dmy(lead.get("mcp_expiry_date")),
                "ewStatus":        lead.get("ew_status"),
                "ewDate":          _dmy(lead.get("ew_expiry_date")),
                "customerType":    lead.get("customer_type"),
            },
            "customerDetails": {
                "name":    lead.get("customer_name"),
                "mobile":  list(lead.get("mobile_number") or []),
                "address": list(lead.get("customer_address") or []),
            },
            # No Kataria bill on record: the lead's own last-service fields stand in.
            "serviceHistory": [_history_entry(bill) for bill in bills[:HISTORY_LIMIT]] or ([{
                "lastServiceDate": _dmy(lead.get("last_service_date")),
                "lastServiceType": lead.get("last_service_type"),
            }] if lead.get("last_service_date") else []),
        },
    }


def _seconds(value: Any) -> Optional[int]:
    """bizRTC's "HH:MM:SS" call length → seconds (None when missing or malformed)."""
    match = re.fullmatch(r"(\d+):(\d{2}):(\d{2})", str(value or "").strip())
    return int(match[1]) * 3600 + int(match[2]) * 60 + int(match[3]) if match else None


def _summary(row: Dict[str, Any]) -> Dict[str, Any]:
    """One call for listings: ids, call type, CRE and outcome (no customer details)."""
    return {
        "transaction_id":        row.get("transaction_id"),
        "cdr_id":                row.get("cdr_id"),
        "crm_cdr_id":            row.get("crm_cdr_id"),
        "registration_number":   row.get("registration_number"),
        "call_date":             row.get("call_date"),
        "call_type":             row.get("_call_type") or "UNKNOWN",
        "lead_type":             row.get("lead_type"),
        "cre":                  {"code": row.get("cre_name"), "name": row.get("cre_id")},
        "disposition":           row.get("primary_disposition"),
        "disposition_form_type": row.get("disposition_form_type"),
        "workshop":              row.get("work_shop_name"),
        "campaign":              row.get("call_campaign"),
        "campaign_id":           row.get("campaign_id"),
        "call_duration":         row.get("call_duration"),
        "call_seconds":          _seconds(row.get("cdr_call_duration")),
        "has_recording":         bool(row.get("has_recording")),
        "connected":             str(row.get("cdr_disposition") or "").upper() == "ANSWERED",
        "crm_audit_status":      row.get("crm_audit_status"),
    }


# ── Queries ───────────────────────────────────────────────────────────────────

def _call_rows(transaction_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not transaction_ids:
        return {}
    rows = db.fetch_all(f"{_CALL_SELECT}\nWHERE lt.id = ANY(CAST(:ids AS uuid[]))", {"ids": transaction_ids})
    return {str(row["transaction_id"]): _export_row(row) for row in rows}


def lead_types(transaction_ids: Iterable[str]) -> Dict[str, Optional[str]]:
    """transaction id → UNNATI / PRAGATI. Primary-key lookup only, so it stays fast
    for thousands of calls (the full call query does not)."""
    ids = sorted({str(i) for i in transaction_ids if i})
    found: Dict[str, Optional[str]] = {}
    for start in range(0, len(ids), 1000):
        for row in db.fetch_all(
            "SELECT lt.id::text AS id, upper(lt.lead_type) AS lead_type FROM lead_transaction lt "
            "WHERE lt.id = ANY(CAST(:ids AS uuid[]))",
            {"ids": ids[start:start + 1000]},
        ):
            found[row["id"]] = row["lead_type"]
    return found


def _vehicles(calls: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """lead_id → vehicle payload for the given call rows."""
    calls = list(calls)
    lead_ids = sorted({row["lead_id"] for row in calls if row.get("lead_id")})
    if not lead_ids:
        return {}
    def rank(lead: Dict[str, Any]) -> Tuple[bool, float]:
        # The live lead wins; otherwise the most recent drop.
        dropped = lead.get("dropped_date")
        return lead["origin"] == "lead", dropped.timestamp() if dropped else float("-inf")

    leads: Dict[str, Dict[str, Any]] = {}
    for lead in db.fetch_all(_VEHICLE_SQL, {"ids": lead_ids}):
        key = str(lead["lead_id"])
        if key not in leads or rank(lead) > rank(leads[key]):
            leads[key] = lead

    regs = sorted({normalize_reg(lead.get("registration_number")) for lead in leads.values()} - {""})
    bills: Dict[str, List[Dict[str, Any]]] = {}
    for bill in db.fetch_all(_HISTORY_SQL, {"regs": regs}) if regs else []:
        bills.setdefault(normalize_reg(bill["reg"]), []).append(bill)

    call_days: Dict[str, date] = {}
    for row in calls:
        try:
            call_days[str(row["lead_id"])] = date.fromisoformat(str(row.get("call_date"))[:10])
        except (KeyError, ValueError):
            pass
    return {
        key: _vehicle_payload(lead, bills.get(normalize_reg(lead.get("registration_number")), []), call_days.get(key))
        for key, lead in leads.items()
    }


def _call_data(row: Dict[str, Any], vehicles: Dict[str, Dict[str, Any]], picked: str) -> CallData:
    return CallData(
        crm=_call_crm(row),
        source=SOURCE_PATH,
        vehicle=vehicles.get(str(row.get("lead_id"))),
        transaction_id=row.get("transaction_id"),
        cdr_id=row.get("cdr_id"),
        picked=picked,
    )


_AUDITED_FORMS = {"BOOKED_SERVICE", "BOOK_MY_SERVICE", "CALL_LATER", "SERVICE_NOT_REQUIRED"}


def _auditable(candidate: Dict[str, Any]) -> bool:
    """One of the four audited call types (not a no-contact attempt or a system status change)."""
    form = str(candidate.get("disposition_form_type") or "").upper()
    return form in _AUDITED_FORMS or (form == "OTHER_DISPOSITION"
                                      and str(candidate.get("status") or "").upper() == "ESCALATION")


def _anchor(reg: str) -> Tuple[Optional[Any], Optional[str]]:
    """(call time, disposition form) CALL DATA.xlsx gives for the vehicle; time is a datetime or a bare date."""
    row = crm_call_data.peek(reg) or {}
    value = (row.get("call_meta") or {}).get("call_date")
    form = (row.get("disposition") or {}).get("type")
    for fmt, is_date in (("%Y-%m-%d %H:%M:%S", False), ("%Y-%m-%d", True)):
        try:
            parsed = datetime.strptime(str(value), fmt)
            return (parsed.date() if is_date else parsed), form
        except (TypeError, ValueError):
            continue
    return None, None


def _pick(candidates: List[Dict[str, Any]], anchor: Tuple[Optional[Any], Optional[str]]
          ) -> Tuple[Optional[Dict[str, Any]], str]:
    """Choose the recording's call among the vehicle's calls (newest first)."""
    when, form = anchor
    if when is not None:
        if isinstance(when, datetime):
            near = [c for c in candidates if abs((_local(c["call_date"]) - when).total_seconds()) <= 120]
        else:
            near = [c for c in candidates if _local(c["call_date"]).date() == when]
        # A no-contact attempt can share the minute with the real call: match the Excel's form.
        near.sort(key=lambda c: (c.get("disposition_form_type") != form, not _auditable(c)))
        if near:
            return near[0], "excel_call_time"
    for label, keep in (("latest_recorded_call", lambda c: c["has_recording"] and _auditable(c)),
                        ("latest_recorded_call", lambda c: c["has_recording"]),
                        ("latest_call", lambda c: True)):
        found = next((c for c in candidates if keep(c)), None)
        if found:
            return found, label
    return None, ""


_REG_BATCH = 25  # registrations per lookup: each uses the trigram index on 20M+ rows


def _from_db(regs: List[str], with_vehicle: bool) -> Dict[str, Optional[CallData]]:
    candidates: Dict[str, List[Dict[str, Any]]] = {reg: [] for reg in regs}
    for start in range(0, len(regs), _REG_BATCH):
        for row in db.fetch_all(
            f"""SELECT lt.id, lt.registration_number, lt.call_date, lt.disposition_form_type, lt.status,
                       (lt.recordingfile IS NOT NULL AND lt.recordingfile <> '') AS has_recording
                FROM lead_transaction lt
                WHERE lower(lt.registration_number) = ANY(:regs) AND {_CALL_FILTER}
                ORDER BY lt.call_date DESC""",
            {"regs": [reg.lower() for reg in regs[start:start + _REG_BATCH]]},
        ):
            reg = normalize_reg(row["registration_number"])
            if reg in candidates:
                candidates[reg].append(row)

    picks = {reg: _pick(rows, _anchor(reg)) for reg, rows in candidates.items()}
    rows = _call_rows([str(row["id"]) for row, _ in picks.values() if row])
    vehicles = _vehicles(rows.values()) if with_vehicle else {}
    result: Dict[str, Optional[CallData]] = {}
    for reg, (row, how) in picks.items():
        full = rows.get(str(row["id"])) if row else None
        result[reg] = _call_data(full, vehicles, how) if full else None
    return result


def _from_cdr(cdr_ids: List[str], with_vehicle: bool) -> Dict[str, Optional[CallData]]:
    """bizRTC CDR id → that exact call (recordings brought in by the importer)."""
    rows: Dict[str, Dict[str, Any]] = {}
    for row in db.fetch_all(f"{_CALL_SELECT}\nWHERE cr.cdr_id = ANY(:cdrs)", {"cdrs": cdr_ids}):
        row = _export_row(row)
        rows.setdefault(str(row["cdr_id"]), row)
    vehicles = _vehicles(rows.values()) if with_vehicle else {}
    return {cdr: (_call_data(rows[cdr], vehicles, "cdr_id") if cdr in rows else None) for cdr in cdr_ids}


_cache: Dict[str, Tuple[float, bool, Optional[CallData]]] = {}
_cache_lock = threading.Lock()


def _cached(kind: str, keys: List[str], with_vehicle: bool, fetch) -> Dict[str, Optional[CallData]]:
    """fetch(missing keys, with_vehicle) through a short-lived cache keyed by kind:key."""
    now = _time.monotonic()
    found: Dict[str, Optional[CallData]] = {}
    with _cache_lock:
        for key in keys:
            hit = _cache.get(f"{kind}:{key}")
            # An entry with vehicle data also answers a lookup without it.
            if hit and now - hit[0] < _CACHE_TTL and (hit[1] or not with_vehicle):
                found[key] = hit[2]
    missing = [key for key in keys if key not in found]
    if missing:
        fresh = fetch(missing, with_vehicle)
        with _cache_lock:
            for key, call in fresh.items():
                _cache[f"{kind}:{key}"] = (now, with_vehicle, call)
        found.update(fresh)
    return found


# ── Lookups used by the audit pipeline and /api/calls, /api/crm ───────────────

_PINNED_STEM_RE = re.compile(r"^(?P<reg>.+?)_(?P<cdr>\d{5,})$")


def parse_stem(stem: str) -> Tuple[str, Optional[str]]:
    """(registration, bizRTC CDR id or None) for a recording stem.

    'GJ01AB1234' is a vehicle (its call is picked); 'GJ01AB1234_12345678' is one
    exact call, as the importer names recordings.
    """
    match = _PINNED_STEM_RE.match(str(stem).strip())
    if match:
        return normalize_reg(match["reg"]), match["cdr"]
    return normalize_reg(stem), None


def get_call_crm_many(stems: Iterable[str], with_vehicle: bool = True) -> Dict[str, CallData]:
    """stem → CallData for every recording stem with call data."""
    by_reg: Dict[str, List[str]] = {}
    by_cdr: Dict[str, List[str]] = {}
    for stem in stems:
        reg, cdr = parse_stem(stem)
        if cdr:
            by_cdr.setdefault(cdr, []).append(stem)
        elif reg:
            by_reg.setdefault(reg, []).append(stem)

    found_reg: Dict[str, CallData] = {}
    found_cdr: Dict[str, CallData] = {}
    if (by_reg or by_cdr) and enabled():
        try:
            if by_reg:
                found_reg = {k: v for k, v in _cached("reg", list(by_reg), with_vehicle, _from_db).items() if v}
            if by_cdr:
                found_cdr = {k: v for k, v in _cached("cdr", list(by_cdr), with_vehicle, _from_cdr).items() if v}
        except Exception as exc:
            if CRM_SOURCE == "db":
                raise
            logger.warning("[CRM_DB] Lookup failed, using CALL DATA.xlsx: %s", db.describe_error(exc))

    # Only vehicle stems fall back to the Excel: a CDR-pinned stem is one exact call.
    if CRM_SOURCE != "db":
        for reg in by_reg.keys() - found_reg.keys():
            hit = crm_call_data.get_crm_by_stem(reg)
            if hit:
                found_reg[reg] = CallData(hit[0], crm_call_data.CALL_DATA_EXCEL_PATH, picked="excel")

    result = {stem: found_reg[reg] for reg, group in by_reg.items() if reg in found_reg for stem in group}
    result.update({stem: found_cdr[cdr] for cdr, group in by_cdr.items() if cdr in found_cdr for stem in group})
    return result


def get_call_crm(stem: str, with_vehicle: bool = True) -> Optional[CallData]:
    """CallData for one recording stem, or None."""
    return get_call_crm_many([stem], with_vehicle).get(stem)


def crm_details(call: CallData) -> Dict[str, Any]:
    """The merged crm_details the audit works from: call data + vehicle master."""
    from .crm_merger import merge_crm
    from .pipeline import normalize_crm_details

    vehicle = normalize_crm_details(call.vehicle) if call.vehicle else None
    return merge_crm(call.crm, vehicle) or call.crm


# ── Queries behind /api/db/* ──────────────────────────────────────────────────

def get_call(transaction_id: str) -> Optional[CallData]:
    """Everything for one call by lead_transaction id. Raises ValueError for a malformed id."""
    tid = str(uuid.UUID(str(transaction_id)))
    row = _call_rows([tid]).get(tid)
    return _call_data(row, _vehicles([row]), "transaction_id") if row else None


def _day_start(day: date) -> datetime:
    return datetime.combine(day, time.min, tzinfo=_TZ) if _TZ else datetime.combine(day, time.min)


# cre_id for calls that have no assigned CRE.
NO_CRE = "none"
NO_CAMPAIGN = "none"

# bizRTC's own record of the call: the customer actually picked up.
_CONNECTED_SQL = "upper(cr.disposition) = 'ANSWERED'"
# Call length in seconds from bizRTC's "HH:MM:SS" text (lead_transaction.call_duration is empty).
_SECONDS_SQL = ("(CASE WHEN cr.call_duration ~ '^[0-9]+:[0-9]{2}:[0-9]{2}$' "
                "THEN extract(epoch from cr.call_duration::interval) END)")


def _where(
    *, call_type: Optional[str] = None, cre: Optional[str] = None, cre_id: Optional[str] = None,
    station: Optional[str] = None, lead_type: Optional[str] = None,
    registration: Optional[str] = None, date_from: Optional[date] = None, date_to: Optional[date] = None,
    has_recording: Optional[bool] = None, connected: Optional[bool] = None,
    cdr_in: Optional[List[str]] = None, cdr_not_in: Optional[List[str]] = None,
    min_seconds: Optional[int] = None, max_seconds: Optional[int] = None,
    campaign: Optional[str] = None, disposition: Optional[str] = None,
) -> Optional[Tuple[List[str], Dict[str, Any]]]:
    """(WHERE clauses, params) shared by list_calls and count_calls; None when nothing can match."""
    where = [_CALL_FILTER]
    params: Dict[str, Any] = {}
    if call_type in CALL_TYPES:
        where.append(f"{_CALL_TYPE_SQL} = :call_type")
        params["call_type"] = call_type
    elif call_type != "ALL":
        where.append(f"{_CALL_TYPE_SQL} IS NOT NULL")
    if cre:
        # users is small; matching ids first lets lead_transaction use its assigned_cre index.
        ids = db.fetch_all("SELECT user_id FROM users WHERE user_name ILIKE :p OR full_name ILIKE :p",
                           {"p": f"%{cre.strip()}%"})
        if not ids:
            return None
        where.append("lt.assigned_cre = ANY(CAST(:cre_ids AS uuid[]))")
        params["cre_ids"] = [str(row["user_id"]) for row in ids]
    if cre_id == NO_CRE:
        where.append("lt.assigned_cre IS NULL")
    elif cre_id:
        where.append("lt.assigned_cre = CAST(:cre_id AS uuid)")
        params["cre_id"] = cre_id
    if station:
        ids = db.fetch_all("SELECT id FROM service_station WHERE name ILIKE :p OR ws_code ILIKE :p OR code ILIKE :p",
                           {"p": f"%{station.strip()}%"})
        if not ids:
            return None
        where.append("lt.service_station_id = ANY(CAST(:station_ids AS uuid[]))")
        params["station_ids"] = [str(row["id"]) for row in ids]
    if lead_type:
        where.append("upper(lt.lead_type) = :lead_type")
        params["lead_type"] = lead_type.upper()
    if registration:
        where.append("lower(lt.registration_number) LIKE :reg")
        params["reg"] = f"%{normalize_reg(registration).lower()}%"
    if date_from:
        where.append("lt.call_date >= :date_from")
        params["date_from"] = _day_start(date_from)
    if date_to:
        where.append("lt.call_date < :date_to")
        params["date_to"] = _day_start(date_to + timedelta(days=1))
    if has_recording is not None:
        where.append("lt.recordingfile IS NOT NULL AND lt.recordingfile <> ''" if has_recording
                     else "(lt.recordingfile IS NULL OR lt.recordingfile = '')")
    if connected is not None:
        where.append(_CONNECTED_SQL if connected else f"NOT coalesce({_CONNECTED_SQL}, false)")
    if cdr_in is not None:
        if not cdr_in:
            return None
        where.append("cr.cdr_id = ANY(:cdr_in)")
        params["cdr_in"] = list(cdr_in)
    if cdr_not_in:
        where.append("NOT (cr.cdr_id = ANY(:cdr_not_in))")
        params["cdr_not_in"] = list(cdr_not_in)
    # Call length bounds (seconds): min inclusive, max exclusive; calls without a length don't match.
    if min_seconds is not None:
        where.append(f"{_SECONDS_SQL} >= :min_seconds")
        params["min_seconds"] = int(min_seconds)
    if max_seconds is not None:
        where.append(f"{_SECONDS_SQL} < :max_seconds")
        params["max_seconds"] = int(max_seconds)
    # Campaign by id ("none" = calls without one).
    if campaign == NO_CAMPAIGN:
        where.append("lt.campaign_id IS NULL")
    elif campaign:
        where.append("lt.campaign_id::text = :campaign_id")
        params["campaign_id"] = str(campaign)
    # The CRE's outcome on the call (lead_transaction.status), e.g. BOOKED_CONFIRMED.
    if disposition:
        where.append("upper(lt.status) = :disposition")
        params["disposition"] = str(disposition).upper()
    return where, params


def list_calls(*, limit: int = 50, offset: int = 0, **filters: Any) -> Tuple[List[Dict[str, Any]], bool]:
    """(page of call summaries, more pages?), newest call first.

    Filters (see _where): call_type (one of CALL_TYPES, "ALL" for every
    transaction, None for the four audited types), cre, cre_id, station, lead_type,
    registration, date_from, date_to, has_recording, connected, cdr_in, cdr_not_in.
    No total here: count_calls_by_cre gives the totals for the same filters.
    """
    key = repr(("list", limit, offset, sorted(filters.items())))
    hit = _list_cache.get(key)
    if hit and _time.monotonic() - hit[0] < _list_ttl(filters):
        return hit[1]
    built = _where(**filters)
    if built is None:
        return [], False
    where, params = built
    rows = db.fetch_all(
        f"{_CALL_SELECT}\nWHERE {' AND '.join(where)}\nORDER BY lt.call_date DESC LIMIT :limit OFFSET :offset",
        {**params, "limit": limit + 1, "offset": offset},
    )
    rows = [_export_row(row) for row in rows]
    result = ([_summary(row) for row in rows[:limit]], len(rows) > limit)
    _remember(_list_cache, key, result)
    return result


def list_call_ids(**filters: Any) -> List[str]:
    """Transaction ids of every call matching the filters (see _where), newest first:
    the dashboard's "select all" without loading the calls themselves."""
    built = _where(**filters)
    if built is None:
        return []
    where, params = built
    rows = db.fetch_all(
        "SELECT lt.id FROM lead_transaction lt LEFT JOIN cdr_records cr ON cr.id = lt.cdr_id "
        f"WHERE {' AND '.join(where)} ORDER BY lt.call_date DESC",
        params,
    )
    return [str(row["id"]) for row in rows]


# Day lists and counts are reused: a past day hardly changes, today keeps growing.
_list_cache: Dict[str, Tuple[float, Any]] = {}
_count_cache: Dict[str, Tuple[float, Dict[str, int]]] = {}


def _list_ttl(filters: Dict[str, Any]) -> float:
    day = filters.get("date_to") or filters.get("date_from")
    return 600.0 if isinstance(day, date) and day < date.today() else 60.0


def _remember(cache: Dict[str, Tuple[float, Any]], key: str, value: Any) -> None:
    if len(cache) > 500:  # plenty for a few days of browsing and filters
        cache.clear()
    cache[key] = (_time.monotonic(), value)


def count_calls_by_cre(**filters: Any) -> List[Dict[str, Any]]:
    """Calls per CRE and audited call type for the same filters as list_calls, by CRE name
    (cached like the lists): [{"id", "code", "name", "counts": {call_type: n}, "total"}].
    Calls without an assigned CRE come last, under the id NO_CRE."""
    key = repr(sorted(filters.items()))
    hit = _count_cache.get(key)
    if hit and _time.monotonic() - hit[0] < _list_ttl(filters):
        return hit[1]
    built = _where(**{**filters, "call_type": filters.get("call_type")})
    cres: Dict[str, Dict[str, Any]] = {}
    if built is not None:
        where, params = built
        # Only the joins the filters need: counting a day touches tens of thousands of rows.
        for row in db.fetch_all(
            f"SELECT lt.assigned_cre AS cre_uuid, u.user_name AS code, u.full_name AS name, "
            f"{_CALL_TYPE_SQL} AS call_type, count(*) AS n FROM lead_transaction lt "
            f"LEFT JOIN cdr_records cr ON cr.id = lt.cdr_id LEFT JOIN users u ON u.user_id = lt.assigned_cre "
            f"WHERE {' AND '.join(where)} GROUP BY 1, 2, 3, 4",
            params,
        ):
            if not row["call_type"]:
                continue
            cre_id = str(row["cre_uuid"]) if row["cre_uuid"] else NO_CRE
            cre = cres.setdefault(cre_id, {"id": cre_id, "code": row["code"], "name": row["name"],
                                           "counts": {}, "total": 0})
            cre["counts"][row["call_type"]] = int(row["n"])
            cre["total"] += int(row["n"])
    result = sorted(cres.values(), key=lambda c: (c["id"] == NO_CRE, str(c["name"] or c["code"] or "").lower()))
    _remember(_count_cache, key, result)
    return result


def count_calls_by_campaign(**filters: Any) -> List[Dict[str, Any]]:
    """Calls per campaign for the same filters (cached like the lists), most calls first:
    [{"id", "name", "n"}]. Calls without a campaign are listed under the id NO_CAMPAIGN."""
    key = "campaign:" + repr(sorted(filters.items()))
    hit = _count_cache.get(key)
    if hit and _time.monotonic() - hit[0] < _list_ttl(filters):
        return hit[1]
    built = _where(**filters)
    result: List[Dict[str, Any]] = []
    if built is not None:
        where, params = built
        for row in db.fetch_all(
            "SELECT lt.campaign_id::text AS id, c.name AS name, count(*) AS n FROM lead_transaction lt "
            "LEFT JOIN cdr_records cr ON cr.id = lt.cdr_id LEFT JOIN campaigns c ON c.id = lt.campaign_id "
            f"WHERE {' AND '.join(where)} GROUP BY 1, 2 ORDER BY n DESC",
            params,
        ):
            result.append({"id": row["id"] or NO_CAMPAIGN, "name": row["name"] or ("No campaign" if not row["id"] else "Unnamed"),
                           "n": int(row["n"])})
    _remember(_count_cache, key, result)
    return result


def count_calls_by_disposition(**filters: Any) -> List[Dict[str, Any]]:
    """Calls per CRE outcome (lead_transaction.status) for the same filters, most calls first:
    [{"id": "BOOKED_CONFIRMED", "n"}] (cached like the lists)."""
    key = "disposition:" + repr(sorted(filters.items()))
    hit = _count_cache.get(key)
    if hit and _time.monotonic() - hit[0] < _list_ttl(filters):
        return hit[1]
    built = _where(**filters)
    result: List[Dict[str, Any]] = []
    if built is not None:
        where, params = built
        for row in db.fetch_all(
            "SELECT upper(lt.status) AS id, count(*) AS n FROM lead_transaction lt "
            "LEFT JOIN cdr_records cr ON cr.id = lt.cdr_id "
            f"WHERE {' AND '.join(where)} GROUP BY 1 ORDER BY n DESC",
            params,
        ):
            if row["id"]:
                result.append({"id": row["id"], "n": int(row["n"])})
    _remember(_count_cache, key, result)
    return result


def vehicle_calls(registration: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Every dispositioned call on record for one vehicle, newest first."""
    reg = normalize_reg(registration)
    if not reg:
        return []
    rows = db.fetch_all(
        f"{_CALL_SELECT}\nWHERE lower(lt.registration_number) = :reg AND {_CALL_FILTER}\n"
        "ORDER BY lt.call_date DESC LIMIT :limit",
        {"reg": reg.lower(), "limit": limit},
    )
    return [_summary(_export_row(row)) for row in rows]
