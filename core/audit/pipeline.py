"""
pipeline.py — End-to-end audit pipeline: per-call processing + batch main().

Entry points
─────────────────────────────────────────────────────────────────────────────
  process_one()    — audit a single transcript file (used by workers/engine.py)
  main()           — CLI entry point for batch folder runs
  classify_call()  — LLM call-type classification
  extract_names()  — LLM name extraction
  verify_booking_genuine() — LLM booking genuineness check
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .brio import CHECK_BELOW, brio_enabled, confidence, score_json
from .llm import (
    LLMClient,
    OllamaClient,
    extract_json,
    make_client,
    single_pass_options,
)
from .params import (
    CALL_TYPE_LABELS,
    FILENAME_CALL_TYPE_MAP,
    WEIGHTS,
    WEIGHT_SOURCE,
    _extract_cre_disposition,
    normalize_transcript_type,
)
from .prompts import (
    BOOKING_GENUINE_SYSTEM_PROMPT,
    CLASSIFICATION_SYSTEM_PROMPT,
    NAME_SYSTEM_PROMPT,
    build_classification_prompt,
    build_name_prompt,
)
from .single_pass_prompts import (
    build_booking_adjudication_prompt,
    build_single_pass_prompt,
)
from .single_pass_validator import (
    SCHEMA_VERSION,
    ValidationResult,
    derive_ratings_from_evidence,
)
from .excel_report import append_audit_report, build_non_scored_summary, save_call_evidence
from .lora_audit import lora_enabled, run_lora_audit
from .rules import (
    compute_param_applicability,
    resolve_audit_route,
)
from .scoring import build_management_summary
from .utils import _redact_payload, redact_call_id

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# TRANSCRIPT LOADING
# ─────────────────────────────────────────────────────────────────────────────

TRANSCRIPT_GLOBS = ["*.transcript.json", "ElevenLabsOutput.txt", "*.txt", "*.srt", "*.vtt"]

_SRT_TIME = re.compile(
    r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})"
    r"(?:\s*\[?\s*(Speaker\s*\d+)\s*\]?)?",
    re.IGNORECASE,
)
_ROLE_RE = re.compile(
    r"^(Agent|Customer)\s*\[([\d.]+)s-([\d.]+)s\]:\s*(.+)$", re.IGNORECASE
)


@dataclass
class Turn:
    index: int
    start: str
    end: str
    speaker: str
    text: str


def parse_transcript(raw: str) -> List[Turn]:
    # Diarized format produced by the standalone transcriber takes priority
    diarized: List[Turn] = []
    for line in raw.splitlines():
        m = _ROLE_RE.match(line.strip())
        if m:
            diarized.append(
                Turn(
                    len(diarized) + 1,
                    f"{m.group(2)}s",
                    f"{m.group(3)}s",
                    m.group(1).capitalize(),
                    m.group(4).strip(),
                )
            )
    if diarized:
        return diarized

    lines = raw.splitlines()
    turns: List[Turn] = []
    i, idx = 0, 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if line.isdigit():
            i += 1
            if i >= n:
                break
            line = lines[i].strip()
        m = _SRT_TIME.search(line)
        if m:
            start, end, spk = m.group(1), m.group(2), (m.group(3) or "Speaker ?").strip()
            i += 1
            buff: List[str] = []
            while (
                i < n and lines[i].strip()
                and not _SRT_TIME.search(lines[i])
                and not lines[i].strip().isdigit()
            ):
                buff.append(lines[i].strip())
                i += 1
            text = " ".join(buff).strip()
            if text:
                idx += 1
                turns.append(Turn(idx, start, end, spk, text))
        else:
            i += 1
    if not turns and raw.strip():
        turns.append(Turn(1, "", "", "Speaker ?", raw.strip()))
    return turns


def _duration_seconds(value: Any) -> Optional[float]:
    text = str(value or "").strip().replace(",", ".")
    if not text:
        return None
    if ":" in text:
        try:
            parts = [float(part) for part in text.split(":")]
            if len(parts) == 3:
                return parts[0] * 3600 + parts[1] * 60 + parts[2]
            if len(parts) == 2:
                return parts[0] * 60 + parts[1]
        except ValueError:
            return None
    match = re.search(r"([\d.]+)", text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def transcript_quality_issue(
    turns: List[Turn], expected_duration_seconds: Optional[float] = None
) -> Optional[str]:
    """Detect truncation by comparing transcript coverage with known audio/call duration.

    Short calls are valid, so turn count or transcript duration alone must never
    reject them.  A failure is reported only when trusted duration metadata says
    substantial audio is missing.
    """

    timed_ends: List[float] = []
    for turn in turns:
        end_seconds = _duration_seconds(turn.end)
        if end_seconds is not None:
            timed_ends.append(end_seconds)
    if not timed_ends or expected_duration_seconds is None:
        return None
    transcript_end = max(timed_ends)
    expected = float(expected_duration_seconds)
    missing = expected - transcript_end
    if expected >= 45.0 and missing >= 30.0 and transcript_end < expected * 0.60:
        return (
            f"transcript covers {transcript_end:.1f}s of {expected:.1f}s "
            f"({len(turns)} turns); audio decode/transcription is probably incomplete"
        )
    return None


def _transcript_audio_duration(path: Path, crm_details: Dict[str, Any]) -> Optional[float]:
    """Read trusted source duration from transcript metadata, falling back to CRM."""
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(payload, dict):
                duration = _duration_seconds(payload.get("audio_duration_seconds"))
                if duration is not None:
                    return duration
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    call_meta = crm_details.get("call_meta") or crm_details.get("crm_source") or {}
    if isinstance(call_meta, dict):
        return _duration_seconds(call_meta.get("call_duration"))
    return None


def read_transcript(path: Path) -> str:
    """Read the structured transcript (v2 ``turns`` list or v1 ``text``) or legacy text."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.name.lower().endswith(".transcript.json"):
        payload = json.loads(raw)
        payload = payload if isinstance(payload, dict) else {}
        turns, text = payload.get("turns"), payload.get("text")
        if isinstance(turns, list):
            return "\n".join(str(turn) for turn in turns)
        if not isinstance(text, str):
            raise ValueError(f"Structured transcript has no turns or text: {path.name}")
        return text
    return raw


def dialogue_text(turns: List[Turn]) -> str:
    return "\n".join(f"[{t.speaker}] {t.text}" for t in turns)


# ─────────────────────────────────────────────────────────────────────────────
# LLM WRAPPERS
# ─────────────────────────────────────────────────────────────────────────────

def classify_call(
    dialogue: str,
    client: LLMClient,
    crm_details: Optional[Dict[str, Any]] = None,
    retries: int = 1,
) -> Dict[str, str]:
    prompt = build_classification_prompt(dialogue, crm_details)
    last_error = ""
    for _ in range(retries + 1):
        try:
            data = extract_json(client.generate(CLASSIFICATION_SYSTEM_PROMPT, prompt))
            call_type = normalize_transcript_type(data.get("call_type"))
            evidence = str(data.get("decisive_evidence", "")).strip()
            rationale = str(data.get("rationale", "")).strip()
            confidence = str(data.get("confidence", "")).strip().title()
            if not evidence or not rationale:
                raise ValueError("Classification evidence and rationale are required")
            if confidence not in {"High", "Medium", "Low"}:
                confidence = "Low"
            return {
                "call_type": call_type,
                "decisive_evidence": evidence,
                "rationale": rationale,
                "confidence": confidence,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            prompt += (
                "\n\nCORRECTION: Return one supported call type with non-empty decisive "
                "evidence and rationale. Output strict JSON only."
            )
    raise ValueError(f"Call-type classification failed: {last_error}")


def verify_booking_genuine(
    dialogue: str, client: LLMClient, retries: int = 1
) -> Dict[str, Any]:
    """Return {"genuine": bool, "evidence": str} for a BOOKING call transcript."""
    user = (
        "Analyse the following call transcript and determine whether a genuine "
        "service booking was confirmed.\n\n"
        f"TRANSCRIPT:\n{dialogue}\n\n"
        'Respond with ONLY: {"genuine": true | false, "evidence": "..."}'
    )
    for _ in range(retries + 1):
        try:
            data = extract_json(client.generate(BOOKING_GENUINE_SYSTEM_PROMPT, user))
            genuine = bool(data.get("genuine", True))
            evidence = str(data.get("evidence", "")).strip() or "No supporting evidence provided."
            return {"genuine": genuine, "evidence": evidence}
        except Exception:  # noqa: BLE001
            user += "\n\nREMINDER: Output STRICT valid JSON only, nothing else."
    return {"genuine": True, "evidence": "Validation check failed — defaulting to genuine."}


def extract_names(
    dialogue: str, client: LLMClient, retries: int = 1
) -> Optional[Dict[str, str]]:
    user = build_name_prompt(dialogue)
    for _ in range(retries + 1):
        try:
            data = extract_json(client.generate(NAME_SYSTEM_PROMPT, user))
            cre = str(data.get("cre_name", "")).strip() or "Not identified"
            cust = str(data.get("customer_name", "")).strip() or "Not identified"
            return {"cre_name": cre, "customer_name": cust}
        except Exception:  # noqa: BLE001
            user += "\n\nREMINDER: Output STRICT valid JSON only, nothing else."
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CRM HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_FILENAME_CODE_RE = re.compile(r"[-_\s]+([A-Za-z]+)\s*$")


def transcript_stem(path: Path) -> str:
    """Return the recording stem for current JSON and legacy transcript files."""
    name = path.name
    suffix = ".transcript.json"
    return name[:-len(suffix)] if name.lower().endswith(suffix) else path.stem


def _call_type_from_filename(path: Path) -> Optional[str]:
    m = _FILENAME_CODE_RE.search(transcript_stem(path))
    if m:
        return FILENAME_CALL_TYPE_MAP.get(m.group(1).upper())
    return None


def _first_crm_value(*values: Any) -> Any:
    for v in values:
        if v not in (None, "", []):
            return v
    return None


def _crm_status_text(status: Any, detail: Any = None) -> Any:
    values = [str(v).strip() for v in (status, detail) if v not in (None, "")]
    return " - ".join(values) if values else None


def _resolve_ew_status(ew_type: Any, ew_date: Any) -> Optional[str]:
    """Resolve extended warranty status to ACTIVE/EXPIRED based on expiry date.

    extWarrantyType values are Roman-numeral plan tiers (III, VI, etc.).
    _applicability_state() can't classify them, so we derive Active/Expired
    from the expiry date and prepend it as a readable marker.
    """
    if ew_type is None and ew_date is None:
        return None
    if ew_date is not None:
        from datetime import date as _date
        try:
            parts = str(ew_date).strip().split("-")
            if len(parts) == 3:
                day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
                expiry = _date(year, month, day)
                status = "ACTIVE" if expiry >= _date.today() else "EXPIRED"
                label = f"{status} - {ew_date}"
                if ew_type:
                    label += f" (Plan {ew_type})"
                return label
        except (ValueError, TypeError):
            pass
    return _crm_status_text(ew_type, ew_date)


# dispositionFormType → stable CRM disposition type (routing key)
_TXNFORM_TO_DISP_TYPE: Dict[str, str] = {
    "BOOK_MY_SERVICE":    "BOOK_MY_SERVICE",
    "SERVICE_NOT_REQUIRED": "SNR",
    "CALL_LATER":         "CALL_ME_LATER",
}
# leadStatus fallback when dispositionFormType is OTHER_DISPOSITION
_LEAD_STATUS_TO_DISP_TYPE: Dict[str, str] = {
    "ESCALATION":     "ESCALATION",
    "BOOK_MY_SERVICE": "BOOK_MY_SERVICE",
}


def _normalize_transaction(txn: Dict[str, Any]) -> Dict[str, Any]:
    """Map a JSON2 flat call-transaction record to the audit's stable CRM schema.

    Produces the same field layout as normalize_crm_details() so all downstream
    routing, rules, and scoring work without modification.
    """
    mobile_raw = txn.get("mobileNumber")
    if isinstance(mobile_raw, list):
        mobile = next((m for m in mobile_raw if m not in (None, "")), None)
    else:
        mobile = mobile_raw or None

    addr_raw = txn.get("customerAddress")
    address = (addr_raw[0] if addr_raw else None) if isinstance(addr_raw, list) else addr_raw

    lead_status = txn.get("leadStatus") or ""
    disp_form   = txn.get("dispositionFormType") or ""
    disp_type = (
        _TXNFORM_TO_DISP_TYPE.get(disp_form)
        or _LEAD_STATUS_TO_DISP_TYPE.get(lead_status)
    )
    # _CRM_DISP_TO_CALL_TYPE is defined later in this module; referenced at call-time
    audit_call_type = _CRM_DISP_TO_CALL_TYPE.get(disp_type or "")  # type: ignore[name-defined]

    normalized: Dict[str, Any] = {
        "schema_version": "crm-txn-v1",
        "crm_source": {
            "cdr_id":       txn.get("cdrId"),
            "call_date":    txn.get("callDate") or txn.get("cdrCallDate"),
            "call_duration": txn.get("callDuration"),
            "call_type":    txn.get("callType"),
            "recording_url": txn.get("recordingfile"),
        },
        "cre": {
            "name":     txn.get("creName"),
            "username": txn.get("creUsername"),
            "role":     txn.get("userRole"),
        },
        "customer": {
            "name":    txn.get("customerName"),
            "phone":   mobile,
            "address": address,
        },
        "vehicle": {
            "model":               txn.get("model"),
            "registration_number": txn.get("registrationNumber"),
            "vin_number":          txn.get("vinNumber"),
            "next_service_type":   txn.get("nextServiceType"),
            "service_due_date":    txn.get("nextServiceDate"),
            "odometer_km":         txn.get("odoMeterReading"),
        },
        "service_profile": {
            "campaign": txn.get("campaignName"),
            "last_authorized_service": (
                f"{txn.get('lastServiceDate')} | {txn.get('lastServiceStation')}"
                if txn.get("lastServiceDate") else None
            ),
        },
        "remarks": {
            "cre_remark":      txn.get("creRemark") or None,
            "customer_remark": txn.get("customerRemark") or None,
        },
        "disposition": {
            "type":   disp_type,
            "name":   lead_status,
            "status": lead_status,   # used by _score_task_crm for ESCALATION check
        },
    }

    if audit_call_type:
        normalized["audit_call_type"] = audit_call_type

    bk_dt  = txn.get("bookingDateTime")
    bk_svc = txn.get("bookedServiceType")
    bk_loc = txn.get("bookingWorkshopLocation")
    if bk_dt or bk_svc or bk_loc:
        normalized["booking"] = {
            "booking_date_time": bk_dt,  # _score_task_crm key
            "service_date":      bk_dt,  # assess_crm_completeness key
            "service_type":      bk_svc,
            "workshop":          bk_loc,
        }

    fu_date   = txn.get("followUpDate")
    fu_reason = txn.get("followUpReason")
    if fu_date or fu_reason:
        normalized["follow_up"] = {
            "date":   fu_date,
            "reason": fu_reason,
        }

    return normalized


def normalize_crm_details(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the CRM API response into the audit's stable schema."""
    api_data = payload.get("data")
    if not isinstance(api_data, dict):
        return payload

    # JSON2 format: flat call-transaction history list
    transactions = api_data.get("transactions")
    if isinstance(transactions, list) and transactions:
        txn = next(
            (t for t in transactions if isinstance(t, dict) and t.get("recordingfile")),
            transactions[0] if isinstance(transactions[0], dict) else None,
        )
        if txn:
            return _normalize_transaction(txn)

    if not any(key in api_data for key in ("lead", "vehicleDetails", "customerDetails")):
        return payload

    lead = api_data.get("lead") if isinstance(api_data.get("lead"), dict) else {}
    vehicle = api_data.get("vehicleDetails") if isinstance(api_data.get("vehicleDetails"), dict) else {}
    customer = api_data.get("customerDetails") if isinstance(api_data.get("customerDetails"), dict) else {}
    history = api_data.get("serviceHistory") if isinstance(api_data.get("serviceHistory"), list) else []

    mobile = customer.get("mobile")
    if isinstance(mobile, list):
        mobile = next((item for item in mobile if item not in (None, "")), None)
    address_parts = [
        vehicle.get(key) for key in ("address1", "address2", "address3")
        if vehicle.get(key) not in (None, "")
    ]
    address = _first_crm_value(
        customer.get("address"), customer.get("pickupAddress"),
        lead.get("customerAddress"), ", ".join(map(str, address_parts)) or None,
    )

    normalized_history: List[Dict[str, Any]] = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        normalized_history.append({
            "service_date": _first_crm_value(entry.get("lastServiceDate"), entry.get("dateTime")),
            "service_type": entry.get("lastServiceType"),
            "odometer_km": entry.get("lastOdometerReading"),
            "bill_amount": entry.get("lastBillAmount"),
            "workshop": _first_crm_value(entry.get("lastServiceStation"), entry.get("dealerName")),
            "service_city": entry.get("lastServiceCity"),
            "service_advisor": _first_crm_value(entry.get("serviceAdvisorName"), entry.get("sa")),
            "job_card_number": entry.get("jobCardNumber"),
        })
    latest = normalized_history[0] if normalized_history else {}

    normalized: Dict[str, Any] = {
        "schema_version": "crm-api-v1",
        "crm_source": {
            "response_code": payload.get("code"),
            "message": payload.get("message"),
            "assigned_to_current_user": api_data.get("assignedToCurrentUser"),
        },
        "cre": {"name": lead.get("assignedCREName"), "employee_id": lead.get("assignedCREId")},
        "customer": {
            "name": _first_crm_value(customer.get("name"), lead.get("customerName")),
            "phone": _first_crm_value(mobile, lead.get("mobileNumber")),
            "address": address,
            "city": _first_crm_value(customer.get("customerCity"), vehicle.get("city")),
            "pin_code": _first_crm_value(customer.get("pinCode"), vehicle.get("pinCode")),
        },
        "vehicle": {
            "model": _first_crm_value(vehicle.get("model"), lead.get("model")),
            "variant": vehicle.get("variant"),
            "registration_number": _first_crm_value(
                vehicle.get("registrationNumber"), lead.get("registrationNumber")
            ),
            "vin_number": _first_crm_value(vehicle.get("vinNumber"), lead.get("vinNumber")),
            "fuel_type": vehicle.get("fuelType"),
            "sale_date": vehicle.get("saleDateTime"),
            "service_due_date": _first_crm_value(
                vehicle.get("nextServiceDate"), lead.get("nextServiceDate")
            ),
            "next_service_drop_date": _first_crm_value(
                vehicle.get("nextServiceDropDate"), lead.get("nextServiceDropDate")
            ),
            "next_service_type": _first_crm_value(
                vehicle.get("nextServiceType"), lead.get("nextServiceType")
            ),
            "odometer_km": _first_crm_value(
                vehicle.get("updatedOdometerReading"), vehicle.get("odometerReading"),
                lead.get("odoMeterReading")
            ),
            "average_km_per_day": lead.get("avrRunningPerDay"),
            "customer_type": vehicle.get("customerType"),
        },
        "service_profile": {
            "campaign": _first_crm_value(lead.get("campaignName"), lead.get("campaignTagging")),
            "workshop": _first_crm_value(
                lead.get("serviceStationName"), vehicle.get("outletName")
            ),
            "workshop_code": _first_crm_value(
                vehicle.get("nsdWorkshop"), vehicle.get("workShopCode")
            ),
            "mcp_status": _crm_status_text(vehicle.get("mcpStatus"), vehicle.get("mcpStatusDate")),
            "ew_status_or_expiry": _resolve_ew_status(
                _first_crm_value(vehicle.get("ewStatus"), vehicle.get("extWarrantyType")),
                _first_crm_value(vehicle.get("extendedWarrantyDue"), vehicle.get("ewDate")),
            ),
            "oem_status": _crm_status_text(vehicle.get("oemStatus"), vehicle.get("oemWarrantyDate")),
            "pickup_drop_status": lead.get("pickupDrop"),
            "last_authorized_service": (
                f"{latest.get('service_date')} | {latest.get('service_type')} | "
                f"{latest.get('workshop')}" if latest else None
            ),
        },
        "service_history": normalized_history,
    }

    workflow_sources = {
        "disposition": ("disposition", "callDisposition"),
        "booking": ("booking", "bookMyService"),
        "follow_up": ("followUp", "follow_up", "callMeLater"),
    }
    for target, source_keys in workflow_sources.items():
        source = next(
            (api_data.get(k) for k in source_keys if isinstance(api_data.get(k), dict)),
            None,
        )
        if source is not None:
            normalized[target] = source

    # Populate remarks from disposition.creRemarks / disposition.customerRemarks
    # so rules.py _score_remarks_tagging can auto-score from the combined endpoint payload
    disp_obj = normalized.get("disposition")
    if isinstance(disp_obj, dict):
        cre_r  = disp_obj.get("creRemarks") or disp_obj.get("cre_remarks")
        cust_r = disp_obj.get("customerRemarks") or disp_obj.get("customer_remarks")
        if cre_r is not None or cust_r is not None:
            normalized.setdefault("remarks", {}).update({
                "cre_remark":      cre_r or None,
                "customer_remark": cust_r or None,
            })

    supplied_type = _first_crm_value(
        payload.get("audit_call_type"), api_data.get("audit_call_type")
    )
    if supplied_type:
        normalized["audit_call_type"] = supplied_type
    return normalized


def load_crm_details(
    path: Path, explicit_path: Optional[Path] = None
) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    # 1. JSON sidecar (vehicle master / DMS data)
    stem = transcript_stem(path)
    candidates = ([explicit_path] if explicit_path else []) + [
        path.with_name(f"{stem}.details.json"),
        path.with_name(f"{stem}.context.json"),
        path.parent / "details.json",
        path.parent / "context.json",
    ]
    json_crm: Optional[Dict[str, Any]] = None
    json_path: Optional[Path] = None
    call_snapshot: Optional[Dict[str, Any]] = None
    for candidate in candidates:
        if candidate is None or not candidate.exists():
            continue
        data = json.loads(candidate.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"CRM details must be a JSON object: {candidate}")
        json_crm = normalize_crm_details(data)
        json_path = candidate
        # Files written by the CRM importer also hold this call's own record.
        if isinstance(data.get("call_data"), dict):
            call_snapshot = data["call_data"]
        break

    # 1b. Imported call: everything was captured at import, no lookup needed
    if call_snapshot is not None:
        from .crm_merger import merge_crm as _merge
        return _merge(call_snapshot, json_crm), json_path

    # 2. Call data by registration number (CRM database, else CALL DATA.xlsx;
    #    call-specific disposition data). Without a JSON sidecar, the database's
    #    vehicle master (lead + service history) stands in for it.
    try:
        from .crm_db import get_call_crm as _cd_get
        from .crm_merger import merge_crm as _merge
        call = _cd_get(stem)
        if call is not None:
            if json_crm is None and call.vehicle is not None:
                json_crm = normalize_crm_details(call.vehicle)
            return _merge(call.crm, json_crm), json_path or call.source
    except Exception as exc:
        logger.warning("[PIPELINE] Call data lookup failed for '%s': %s", stem, exc)

    # 3. JSON-only (no Excel match)
    if json_crm is not None:
        return json_crm, json_path

    # 4. Legacy Excel lookup by phone number in stem
    try:
        from .crm_excel import get_crm_by_stem, CRM_EXCEL_PATH
        details = get_crm_by_stem(stem)
        if details:
            return details, CRM_EXCEL_PATH
    except Exception:
        pass

    return None, None


def assess_crm_completeness(details: Dict[str, Any]) -> Dict[str, Any]:
    disposition = _extract_cre_disposition(details)
    required: Dict[str, List[Tuple[str, str]]] = {
        "BOOK_MY_SERVICE": [
            ("booking", "workshop"), ("booking", "service_type"),
            ("booking", "sub_service_type"), ("booking", "service_date"),
            ("booking", "service_time"), ("booking", "service_option"),
        ],
        "SNR": [("disposition", "reason")],
        "CALL_ME_LATER": [
            ("follow_up", "date"), ("follow_up", "time"), ("follow_up", "reason"),
        ],
        "ESCALATION": [("disposition", "reason")],
        "DND": [("disposition", "reason")],
    }
    if not disposition:
        return {"status": "Disposition not supplied", "missing_fields": []}
    missing = []
    for section, field_name in required.get(disposition, []):
        section_data = details.get(section, {})
        value = section_data.get(field_name) if isinstance(section_data, dict) else None
        if value in (None, ""):
            missing.append(f"{section}.{field_name}")
    return {
        "status": "Complete" if disposition and not missing else "Incomplete",
        "missing_fields": missing,
    }


def apply_crm_metadata(result: Dict[str, Any], details: Dict[str, Any]) -> None:
    meta = result.setdefault("metadata", {})
    customer = details.get("customer", {}) or {}
    vehicle = details.get("vehicle", {}) or {}
    cre = details.get("cre", {}) or {}
    if not isinstance(customer, dict):
        customer = {}
    if not isinstance(vehicle, dict):
        vehicle = {}
    if not isinstance(cre, dict):
        cre = {}
    if customer.get("name"):
        meta["customer_name"] = customer["name"]
    if cre.get("name"):
        meta["cre_name"] = cre["name"]
    if vehicle.get("registration_number"):
        meta["vehicle_number"] = vehicle["registration_number"]
    vehicle_parts = [
        str(vehicle.get(key, "")).strip()
        for key in ("model", "fuel_type", "sale_date")
        if str(vehicle.get(key, "")).strip()
    ]
    if vehicle_parts:
        meta["vehicle_details"] = " | ".join(vehicle_parts)


# ─────────────────────────────────────────────────────────────────────────────
# DETERMINISTIC CALL-TYPE ROUTING (no LLM)
# ─────────────────────────────────────────────────────────────────────────────

_CRM_DISP_TO_CALL_TYPE: Dict[str, str] = {
    "BOOK_MY_SERVICE": "BOOKING",
    "SNR":             "SERVICE_NT_REQUIRED",
    "CALL_ME_LATER":   "FOLLOW_UPS",
    "ESCALATION":      "ESCALATION",
}

def determine_audit_type_deterministic(
    path: Path,
    crm_details: Dict[str, Any],
) -> Tuple[Optional[str], str, str]:
    """Return (call_type | None, evidence, confidence) without any LLM call.

    Priority order:
    1. crm_details["audit_call_type"]  (explicitly set by CRM caller)
    2. Filename code → FILENAME_CALL_TYPE_MAP  (BOD/BMS/ESC/CML/SNR)
    3. CRM primary disposition → call type mapping
    4. None → mark for manual review
    """
    # 1. Explicit CRM override
    explicit = crm_details.get("audit_call_type")
    if explicit:
        ct = normalize_transcript_type(str(explicit))
        return ct, f"CRM audit_call_type={explicit!r}", "High"

    # 2. Filename code
    fn_type = _call_type_from_filename(path)
    if fn_type:
        code = re.search(r"[-_\s]+([A-Za-z]+)\s*$", transcript_stem(path))
        code_str = code.group(1).upper() if code else ""
        return fn_type, f"Filename code {code_str!r}", "High"

    # 3. CRM disposition
    disp = _extract_cre_disposition(crm_details)
    if disp and disp in _CRM_DISP_TO_CALL_TYPE:
        return _CRM_DISP_TO_CALL_TYPE[disp], f"CRM disposition={disp!r}", "Medium"

    # 4. Cannot determine
    return None, "No filename code, CRM audit_call_type, or known disposition found.", "Low"


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-PASS AUDIT DICT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_standard_audit_dict(
    payload: Dict[str, Any],
    call_type: str,
    auto_full: Dict[str, Any],
    model_audit: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, Dict[str, str]]:
    """Convert single-pass payload + auto_full into the {pid: {rating, evidence}} format
    that score_audit(), validation, JSON storage, and Excel export expect.

    model_audit holds fine-tuned model ratings; deterministic auto_full ratings
    stay authoritative over them.
    """
    ratings = derive_ratings_from_evidence(payload, call_type, auto_full)
    evidence_ledger = payload.get("evidence_ledger") or {}
    old_audit       = payload.get("audit") or {}
    result: Dict[str, Dict[str, str]] = {}

    for pid, rating in ratings.items():
        # Remarks quality is binary: an accurate entered remark earns Full;
        # every other level is forced to Zero.
        if pid == "remarks_tagging" and str(rating).strip().lower() != "full":
            rating = "Zero"
        # Prefer evidence from single-pass evidence_ledger.decision_basis
        ev_entry = evidence_ledger.get(pid)
        au_entry = old_audit.get(pid)
        if isinstance(ev_entry, dict) and ev_entry.get("decision_basis"):
            evidence = ev_entry["decision_basis"]
        elif isinstance(ev_entry, dict) and ev_entry.get("evidence_state"):
            evidence = "Rated by the stock model (rating only, no written reason)."
        elif isinstance(au_entry, dict) and au_entry.get("evidence"):
            evidence = au_entry["evidence"]
        elif pid in auto_full:
            evidence = auto_full[pid].get("evidence", "Auto-scored")
        else:
            evidence = "No evidence supplied."
        result[pid] = {"rating": rating.capitalize(), "evidence": str(evidence)}

    for pid, entry in (model_audit or {}).items():
        if pid in result and pid not in auto_full:
            result[pid] = entry
    return result


def _ratings_to_check(
    audit: Dict[str, Dict[str, str]],
    rule_pids: set,
    model_audit: Optional[Dict[str, Dict[str, Any]]],
    picked: Dict[str, float],
    booking_decided_later: bool,
) -> Tuple[Dict[str, float], List[str]]:
    """Model confidence per rating (Brio scoring only) and the ratings to check.

    A rating set by a rule is not the model's, so it has no confidence. The
    Booking verdict counts too, unless the adjudication call decided it.
    """
    scores: Dict[str, float] = {
        pid: entry["confidence"] for pid, entry in (model_audit or {}).items() if "confidence" in entry
    }
    for path, value in picked.items():
        parts = path.split(".")
        if len(parts) == 3 and parts[0] == "evidence_ledger" and parts[2] == "evidence_state":
            scores[parts[1]] = value
    scores = {pid: value for pid, value in scores.items() if pid in audit and pid not in rule_pids}
    if "booking_genuine.genuine" in picked and not booking_decided_later:
        scores["booking_genuine"] = picked["booking_genuine.genuine"]
    check = sorted((pid for pid, value in scores.items() if value < CHECK_BELOW), key=scores.get)
    return scores, check


def _evidence_record(
    audit: Dict[str, Dict[str, str]],
    payload: Dict[str, Any],
    rule_pids: set,
    model_audit: Optional[Dict[str, Dict[str, str]]],
    call_type: str,
    confidences: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """What already backs each rating: its source, reason and transcript turn IDs.

    Nothing extra is generated; turn IDs exist only where a model already cited them.
    """
    ledger = payload.get("evidence_ledger") or {}
    booking = payload.get("booking_genuine") if isinstance(payload.get("booking_genuine"), dict) else None
    params: Dict[str, Any] = {}
    for pid, entry in audit.items():
        cited = ledger.get(pid) if isinstance(ledger.get(pid), dict) else {}
        turns = [*(cited.get("agent_turn_ids") or []), *(cited.get("customer_trigger_turn_ids") or [])]
        if pid in rule_pids:
            source = "Rule"
        elif pid in (model_audit or {}):
            source = "Fine-tuned model"
        elif pid == "booking_details":
            source = "Booking facts"
            turns = [booking.get(k) for k in ("date_turn_id", "time_turn_id", "address_or_location_turn_id")
                     if booking and booking.get(k)] if booking else []
        else:
            source = "Stock model"
        params[pid] = {
            "rating": entry.get("rating"),
            "source": source,
            "evidence": entry.get("evidence", ""),
            "turns": list(dict.fromkeys(str(t) for t in turns if t)),
        }
        if pid in (confidences or {}):
            params[pid]["confidence"] = confidences[pid]
            params[pid]["check"] = confidences[pid] < CHECK_BELOW
    record: Dict[str, Any] = {"call_type": call_type, "params": params}
    if call_type == "BOOKING" and booking:
        record["booking"] = {
            "genuine": booking.get("genuine"),
            "reason": booking.get("fake_booking_reason"),
            "evidence": booking.get("evidence", ""),
            "turns": list(dict.fromkeys(
                str(booking[k]) for k in ("agreement_turn_id", "final_stance_turn_id", "date_turn_id")
                if booking.get(k)
            )),
        }
        if "booking_genuine" in (confidences or {}):
            record["booking"]["confidence"] = confidences["booking_genuine"]
            record["booking"]["check"] = confidences["booking_genuine"] < CHECK_BELOW
        if booking.get("too_short"):
            record["booking"]["check"] = True
    return record


_BOOKING_DAYPART_RE = re.compile(
    r"\b(?:morning|afternoon|evening|savar(?:e)?|bapor(?:e)?|saanj(?:e)?|shaam)\b|"
    r"સવાર(?:ે)?|બપોર(?:ે)?|સાંજ(?:ે)?|सुबह|दोपहर|शाम",
    re.IGNORECASE,
)
_BOOKING_DATE_RE = re.compile(
    r"\b(?:today|tomorrow|day after tomorrow|aaj|kal|parso|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"somvar|mangalvar|budhvar|guruvaar|shukravar|shanivar|ravivar|"
    r"\d{1,2}(?:st|nd|rd|th)|\d{1,2}\s+(?:date|tarikh)|"
    r"\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b|"
    r"આજે|કાલે|પરમદિવસે|સોમવાર|મંગળવાર|બુધવાર|ગુરુવાર|શુક્રવાર|શનિવાર|રવિવાર|"
    r"\d{1,2}\s*તારીખ|"
    r"आज|कल|परसों|सोमवार|मंगलवार|बुधवार|गुरुवार|शुक्रवार|शनिवार|रविवार|"
    r"\d{1,2}\s*तारीख",
    re.IGNORECASE,
)
_BOOKING_TIME_RE = re.compile(
    _BOOKING_DAYPART_RE.pattern
    + r"|\b\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?|baje|vagye|vage)\b"
      r"|વાગ્યે|વાગ્યા|બજે|बजे",
    re.IGNORECASE,
)
_BOOKING_CONTEXT_RE = re.compile(
    r"\b(?:service|pickup|pick\s*up|appointment|booking|vehicle|car|driver)\b|"
    r"સર્વિસ|પિક\s*અપ|ગાડી|ડ્રાઇવર|बुकिंग|सर्विस|गाड़ी|ड्राइवर",
    re.IGNORECASE,
)
_BOOKING_ASSENT_RE = re.compile(
    r"\b(?:yes|yeah|okay|ok|fine|confirmed|available|free|done|haan|ha|theek)\b|"
    r"હા|ઠીક|બરાબર|વાંધો\s*નહીં|हाँ|हां|ठीक",
    re.IGNORECASE,
)
_BOOKING_REJECTION_RE = re.compile(
    r"\b(?:no|not|cannot|can't|won't|maybe|later|call\s*back)\b|"
    r"ના|નથી|શાયદ|नहीं|नही|मत|शायद",
    re.IGNORECASE,
)
_BOOKING_UNRESOLVED_RE = re.compile(
    r"\b(?:maybe|not\s+(?:fixed|decided|sure)|depends?|if\s+(?:possible|convenient)|"
    r"when\s+(?:free|convenient)|"
    r"ask\s+(?:him|her|them|family|husband|wife)|check\s+(?:and|then)|"
    r"will\s+(?:check|ask|confirm|decide)|let\s+you\s+know)\b|"
    r"puchh?(?:\s*(?:kar|ke|ine|i))?|dekh\s*(?:kar|ke)|baat\s+karke|"
    r"joine|janavi\s+daish|nakki\s+nathi|possible\s+hoy|"
    r"free\s+hoy\s+to|anukulta\s+hoy\s+to|"
    r"aavta\s+mahina.*koi.*tarikh|next\s+month.*(?:any|some|not\s+fixed)|"
    r"પૂછી|જોઈને|નક્કી\s+નથી|શક્ય\s+હોય|અનુકૂળતા.*હોય.*તો|"
    r"આવતા\s+મહિના.*કોઈ.*તારીખ|"
    r"पूछकर|देखकर|तय\s+नहीं|पक्का\s+नहीं|अगर\s+संभव|"
    r"अगले\s+महीने.*(?:कोई|किसी).*तारीख",
    re.IGNORECASE,
)
_BOOKING_CALLBACK_RE = re.compile(
    r"\b(?:i\s+will\s+call|will\s+call|call\s+(?:you|back|later)|phone\s+(?:you|back))\b|"
    r"hu\s+phone\s+karish|phone\s+karine|call\s+kar(?:i|ish|ine)|"
    r"હું\s+ફોન\s+કરીશ|ફોન\s+કરીને|(?:કોલ|call)\s+કરી(?:શ|ને)?|"
    r"मैं\s+फोन\s+करूँगा|फोन\s+करके|कॉल\s+करूँगा",
    re.IGNORECASE,
)
_BOOKING_LOGISTICS_RE = re.compile(
    r"\b(?:pickup|pick\s*up|driver|coordinate|coordination|arrival|location|address)\b|"
    r"પિક\s*અપ|ડ્રાઇવર|સરનામ|લોકેશન|ड्राइवर|पिक\s*अप|पता|लोकेशन",
    re.IGNORECASE,
)
_BOOKING_DECLINE_RE = re.compile(
    r"\b(?:cancel(?:led)?|do\s+not\s+book|don't\s+book|cannot\s+come|"
    r"won't\s+come|not\s+possible)\b|"
    r"નથી\s+કરાવ|નહીં\s+કરાવ|રદ\s+કર|नहीं\s+करवा|मत\s+बुक|रद्द\s+कर",
    re.IGNORECASE,
)
_BOOKING_DIRECT_ACTION_RE = re.compile(
    r"\b(?:book\s+(?:it|the\s+slot)|confirm\s+(?:it|the\s+booking|the\s+slot)|"
    r"appointment\s+(?:is\s+)?confirmed|pickup\s+(?:is\s+)?(?:fine|confirmed)|"
    r"pick\s*up\s+(?:is\s+)?(?:fine|confirmed)|self\s+(?:come|bring)|will\s+(?:come|bring)|"
    r"service\s+(?:on|at|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b|"
    r"book\s+kar\s+do|confirm\s+kar|mukva(?:ni|nu|sho)|karavi\s+lo|aavishu|"
    r"બુક\s+કરી|કન્ફર્મ\s+કર|મૂકવાની|મૂકી\s+દો|પિક\s*અપ|જાતે\s+આવ|આવીશું|"
    r"बुक\s+कर|कन्फर्म\s+कर|कर\s+दीजिए|आएंगे|खुद\s+आ",
    re.IGNORECASE,
)
_BOOKING_PROPOSAL_RE = re.compile(
    r"\b(?:appointment|booking|slot|book|confirm|pickup|pick\s*up|"
    r"come|bring|service)\b|"
    r"અપોઇન્ટમેન્ટ|બુક|કન્ફર્મ|પિક\s*અપ|મૂક|આવ|"
    r"अपॉइंटमेंट|बुक|कन्फर्म|पिक\s*अप|आओ|आएंगे",
    re.IGNORECASE,
)
_BOOKING_CONCRETE_PROPOSAL_RE = re.compile(
    r"\b(?:book\s+(?:it|for)|confirm\s+(?:it|for)|slot\s+(?:for|on|at)|"
    r"pickup\s+(?:on|at)|pick\s*up\s+(?:on|at)|service\s+(?:on|at|today|tomorrow)|"
    r"come\s+(?:on|at|today|tomorrow)|bring\s+(?:on|at|today|tomorrow))\b|"
    r"બુક\s+કરી|કન્ફર્મ\s+કર|પિક\s*અપ.*(?:આજે|કાલે|તારીખ)|"
    r"बुक\s+कर|कन्फर्म\s+कर|पिक\s*अप.*(?:आज|कल|तारीख)",
    re.IGNORECASE,
)


def _customer_assents(text: str) -> bool:
    value = text or ""
    if re.search(r"\bno\s+(?:problem|issue)\b|વાંધો\s*નહીં", value, re.IGNORECASE):
        return True
    return bool(_BOOKING_ASSENT_RE.search(value)) and not _BOOKING_REJECTION_RE.search(value)


def _confirmed_booking_fact_turn(
    turns: List[Turn], fact_pattern: re.Pattern[str]
) -> Optional[Turn]:
    """Find Customer confirmation of a booking fact or an Agent proposal accepted soon after."""
    for index, turn in enumerate(turns):
        speaker = turn.speaker.strip().lower()
        text = turn.text or ""
        if (
            speaker == "customer"
            and fact_pattern.search(text)
            and (_customer_assents(text) or _BOOKING_DIRECT_ACTION_RE.search(text))
        ):
            return turn
        if (
            speaker == "agent"
            and fact_pattern.search(text)
            and _BOOKING_CONTEXT_RE.search(text)
        ):
            # Only the first Customer response to the proposal can confirm it.
            # Looking eight turns ahead allowed an unrelated later "okay" to
            # confirm an earlier date/time and created false bookings.
            for reply in turns[index + 1:index + 4]:
                if reply.speaker.strip().lower() != "customer":
                    continue
                return reply if _customer_assents(reply.text or "") else None
    return None


def _final_booking_stance(turns: List[Turn]) -> tuple[Optional[str], Optional[Turn]]:
    """Return the last high-confidence Customer booking decision in chronology
    (None when a later acceptance reverses an earlier no or maybe)."""
    stance: Optional[str] = None
    evidence_turn: Optional[Turn] = None
    for index, turn in enumerate(turns):
        if turn.speaker.strip().lower() != "customer":
            continue
        text = turn.text or ""
        if _BOOKING_UNRESOLVED_RE.search(text):
            stance, evidence_turn = "conditional", turn
            continue
        if _BOOKING_DECLINE_RE.search(text):
            stance, evidence_turn = "declined", turn
            continue
        if _BOOKING_CALLBACK_RE.search(text):
            # A callback preserves an earlier confirmation only when the
            # Customer explicitly says it is for appointment logistics.
            if stance == "confirmed" and _BOOKING_LOGISTICS_RE.search(text):
                continue
            stance, evidence_turn = "conditional", turn
            continue
        if _BOOKING_DIRECT_ACTION_RE.search(text):
            stance, evidence_turn = "confirmed", turn
            continue
        if _customer_assents(text):
            # Treat assent as booking acceptance only when it directly answers
            # the immediately preceding Agent booking proposal.  A generic
            # "okay" elsewhere in the call is not commitment evidence.
            prior = turns[index - 1] if index > 0 else None
            if (
                prior is not None
                and prior.speaker.strip().lower() == "agent"
                and _BOOKING_PROPOSAL_RE.search(prior.text or "")
                and (
                    _BOOKING_CONCRETE_PROPOSAL_RE.search(prior.text or "")
                    or _BOOKING_DATE_RE.search(prior.text or "")
                    or _BOOKING_TIME_RE.search(prior.text or "")
                )
            ):
                stance, evidence_turn = "confirmed", turn
    if stance in ("conditional", "declined") and evidence_turn is not None:
        # A later yes to the Agent's booking talk can reverse an earlier no/maybe that
        # the patterns above can't read as a confirmation; the LLM, told that the
        # last acceptance wins, decides. On 2,507 human-audited Bookings this halved
        # genuine calls forced fake (88 -> 42) and missed 1 of 77 human fakes.
        for index, turn in enumerate(turns):
            if turn.index <= evidence_turn.index or turn.speaker.strip().lower() != "customer":
                continue
            prior = turns[index - 1] if index > 0 else None
            text = turn.text or ""
            if (
                prior is not None
                and prior.speaker.strip().lower() == "agent"
                and (_BOOKING_PROPOSAL_RE.search(prior.text or "") or _BOOKING_CONTEXT_RE.search(prior.text or ""))
                and (_customer_assents(text) or _BOOKING_DIRECT_ACTION_RE.search(text))
                and not _BOOKING_UNRESOLVED_RE.search(text)
                and not _BOOKING_DECLINE_RE.search(text)
            ):
                return None, None
    return stance, evidence_turn


# A Booking call this short (only greetings, or cut off) can't show whether the booking
# is real: it is scored normally and flagged for a person, never marked fake.
_TOO_SHORT_TURNS = 4


# A follow-up on a booking made in an earlier call is fake only when the Customer
# explicitly turns it down (the user's rule, 2026-10-14).
_BOOKING_REJECTED_STANCES = {"declined", "cancelled", "disputed"}
_BOOKING_REJECTED_REASONS = {"explicit_decline", "explicit_cancellation", "customer_disputed_booking"}


def booked_on_earlier_call(crm_details: Optional[Dict[str, Any]]) -> bool:
    """True when the CRM booking form predates this call (the call follows it up)."""
    return ((crm_details or {}).get("booking") or {}).get("made_in_this_call") is False


def booking_needs_adjudication(
    payload: Dict[str, Any], turns: List[Turn], booked_earlier: bool = False
) -> bool:
    """Return True only for the narrow high-risk Booking contradiction.

    Normal Booking calls remain single-pass.  A second compact check is used
    only when the main model claims a genuine booking despite 0/6 combined
    booking details and Python cannot ground a decisive Customer stance.
    """
    bg = payload.get("booking_genuine")
    if not isinstance(bg, dict) or bg.get("genuine") is not True:
        return False
    # The adjudicator demands a dated commitment: a follow-up on an existing booking
    # has none to give, and a call too short to judge is never marked fake.
    if (booked_earlier or bg.get("commitment_type") == "existing_booking_confirmed"
            or len(turns) <= _TOO_SHORT_TURNS):
        return False
    detail_count = sum(
        bg.get(field) is True
        for field in (
            "date_confirmed", "time_confirmed", "address_or_location_confirmed",
        )
    )
    if detail_count >= 2:
        return False
    stance, _ = _final_booking_stance(turns)
    return stance is None


def apply_booking_adjudication(
    payload: Dict[str, Any], adjudication: Dict[str, Any], turns: List[Turn]
) -> None:
    """Validate and apply the compact Booking adjudicator's transcript facts."""
    if not isinstance(adjudication, dict):
        raise ValueError("Booking adjudication is not an object")
    decision = str(adjudication.get("decision") or "").strip().upper()
    final_stance = str(adjudication.get("final_stance") or "").strip().lower()
    commitment_tid = str(adjudication.get("customer_commitment_turn_id") or "").strip()
    date_confirmed = adjudication.get("date_or_day_confirmed") is True
    date_tid = str(adjudication.get("date_or_day_turn_id") or "").strip()
    reason_code = str(adjudication.get("reason_code") or "").strip().lower()
    rationale = str(adjudication.get("rationale") or "").strip()

    allowed_stances = {
        "confirmed", "conditional", "declined", "cancelled", "disputed",
        "no_customer_agreement", "not_discussed",
    }
    allowed_reasons = {
        "confirmed_dated_appointment", "conditional_callback",
        "no_customer_agreement", "explicit_decline", "cancelled",
        "customer_disputed_booking", "agent_only_claim", "no_concrete_date",
        "no_booking_discussion",
    }
    if decision not in {"GENUINE", "NOT_GENUINE"}:
        raise ValueError("Booking adjudication has an invalid decision")
    if final_stance not in allowed_stances or reason_code not in allowed_reasons:
        raise ValueError("Booking adjudication has an invalid stance or reason")
    if not rationale or len(rationale) > 180:
        raise ValueError("Booking adjudication rationale is missing or too long")
    turn_map = {f"T{turn.index:03d}": turn for turn in turns}
    if commitment_tid:
        commitment_turn = turn_map.get(commitment_tid)
        if commitment_turn is None or commitment_turn.speaker.strip().lower() != "customer":
            raise ValueError("Booking adjudication commitment turn is not a Customer turn")
    if date_tid:
        date_turn = turn_map.get(date_tid)
        if date_turn is None or not _BOOKING_DATE_RE.search(date_turn.text or ""):
            raise ValueError("Booking adjudication date turn has no grounded day/date")
    if date_confirmed != bool(date_tid):
        raise ValueError("Booking adjudication date flag and turn ID disagree")

    bg = payload.get("booking_genuine")
    if not isinstance(bg, dict):
        raise ValueError("Booking payload has no booking_genuine object")
    if decision == "GENUINE":
        if (
            final_stance != "confirmed"
            or reason_code != "confirmed_dated_appointment"
            or not commitment_tid
            or not date_confirmed
            or not date_tid
        ):
            raise ValueError("GENUINE adjudication lacks a grounded dated commitment")
        bg.update({
            "genuine": True,
            "customer_agreement_confirmed": True,
            "agreement_turn_id": commitment_tid,
            "final_customer_stance": "confirmed",
            "final_stance_turn_id": commitment_tid,
            "date_confirmed": True,
            "date_turn_id": date_tid,
            "commitment_type": (
                "date_time_confirmed" if bg.get("time_confirmed") is True else "date_only"
            ),
            "fake_booking_reason": "none",
            "evidence": rationale,
        })
        return

    if final_stance == "confirmed" or reason_code == "confirmed_dated_appointment":
        raise ValueError("NOT_GENUINE adjudication contradicts its stance or reason")
    conditional_reasons = {"conditional_callback", "no_concrete_date"}
    fake_reason = (
        "conditional_or_vague_booking"
        if reason_code in conditional_reasons or final_stance == "conditional"
        else reason_code
    )
    bg.update({
        "genuine": False,
        "customer_agreement_confirmed": False,
        "agreement_turn_id": "",
        "final_customer_stance": final_stance,
        "final_stance_turn_id": commitment_tid,
        "date_confirmed": date_confirmed,
        "date_turn_id": date_tid if date_confirmed else "",
        "commitment_type": (
            "vague_promise" if fake_reason == "conditional_or_vague_booking"
            else "no_customer_agreement"
        ),
        "fake_booking_reason": fake_reason,
        "evidence": rationale,
    })


def _reconcile_customer_daypart(payload: Dict[str, Any], turns: List[Turn]) -> None:
    """Recover explicit Customer-accepted date/daypart facts missed by the LLM."""
    bg = payload.get("booking_genuine")
    if not isinstance(bg, dict):
        return
    date_turn = _confirmed_booking_fact_turn(turns, _BOOKING_DATE_RE)
    time_turn = _confirmed_booking_fact_turn(turns, _BOOKING_TIME_RE)
    if date_turn is not None:
        # The date and Customer agreement are separate LLM fields.  Recover
        # agreement even when the model already found the date; previously the
        # recovery was nested under date_confirmed=False, which could leave a
        # genuinely accepted booking marked fake.
        if bg.get("date_confirmed") is not True:
            bg["date_confirmed"] = True
            bg["date_turn_id"] = f"T{date_turn.index:03d}"
    if bg.get("time_confirmed") is not True and time_turn is not None:
        bg["time_confirmed"] = True
        bg["time_turn_id"] = f"T{time_turn.index:03d}"

    detected_stance, stance_turn = _final_booking_stance(turns)
    if detected_stance is not None and stance_turn is not None:
        bg["final_customer_stance"] = detected_stance
        bg["final_stance_turn_id"] = f"T{stance_turn.index:03d}"
        if detected_stance == "confirmed":
            bg["customer_agreement_confirmed"] = True
            bg["agreement_turn_id"] = f"T{stance_turn.index:03d}"
        else:
            bg["customer_agreement_confirmed"] = False
            bg["agreement_turn_id"] = ""

    if (
        bg.get("final_customer_stance") == "confirmed"
        and bg.get("customer_agreement_confirmed") is True
        and bg.get("date_confirmed") is True
    ):
        bg["commitment_type"] = (
            "date_time_confirmed" if bg.get("time_confirmed") is True else "date_only"
        )
        bg["fake_booking_reason"] = "none"


def enforce_booking_integrity(
    payload: Dict[str, Any], turns: Optional[List[Turn]] = None, booked_earlier: bool = False
) -> None:
    """Deterministically reconcile Booking detail scoring and genuineness.

    The LLM extracts semantic facts; Python converts those facts into the exact
    combined 6/4/0 level and prevents a Booking with no Customer agreement from
    being reported as genuine. ``booked_earlier``: the CRM booking predates this
    call, which is then fake only on an explicit decline, cancellation or dispute.
    """
    bg = payload.get("booking_genuine")
    if not isinstance(bg, dict):
        return
    if turns:
        _reconcile_customer_daypart(payload, turns)

    detail_fields = (
        "date_confirmed", "time_confirmed", "address_or_location_confirmed",
    )
    detail_count = sum(bg.get(field) is True for field in detail_fields)
    detail_state = "supported" if detail_count == 3 else "partial" if detail_count == 2 else "absent"
    detail_rating = "Full" if detail_count == 3 else "Partial" if detail_count == 2 else "Zero"

    ledger = payload.setdefault("evidence_ledger", {})
    detail_entry = ledger.get("booking_details")
    if not isinstance(detail_entry, dict):
        detail_entry = {}
        ledger["booking_details"] = detail_entry
    detail_entry["evidence_state"] = detail_state
    detail_entry["decision_basis"] = (
        "BOOKING_DETAIL_CHECK: "
        f"date={'yes' if bg.get('date_confirmed') is True else 'no'}, "
        f"time={'yes' if bg.get('time_confirmed') is True else 'no'}, "
        f"address/location={'yes' if bg.get('address_or_location_confirmed') is True else 'no'}; "
        f"deterministic level={detail_rating}."
    )

    triggers = payload.setdefault("trigger_ledger", {})
    triggers["date_confirmed"] = bg.get("date_confirmed") is True
    triggers["time_confirmed"] = bg.get("time_confirmed") is True
    triggers["address_or_location_confirmed"] = (
        bg.get("address_or_location_confirmed") is True
    )

    if turns is not None and len(turns) <= _TOO_SHORT_TURNS:
        bg.update({
            "genuine": True,
            "fake_booking_reason": "none",
            "too_short": True,
            "evidence": f"Too short to judge ({len(turns)} turns): not marked fake; please check the call.",
        })
        triggers["booking_commitment_expressed"] = False
        return

    agreement = bg.get("customer_agreement_confirmed") is True
    final_stance = str(bg.get("final_customer_stance") or "").strip()
    commitment_type = str(bg.get("commitment_type") or "").strip()
    reason = str(bg.get("fake_booking_reason") or "").strip()
    if (
        booked_earlier
        and final_stance not in _BOOKING_REJECTED_STANCES
        and reason not in _BOOKING_REJECTED_REASONS
    ):
        if not (bg.get("genuine") is True and commitment_type in {
            "date_time_confirmed", "date_only", "existing_booking_confirmed",
        }):
            bg["commitment_type"] = "existing_booking_confirmed"
            existing_note = str(bg.get("evidence") or "").strip()
            bg["evidence"] = ("Follows up a booking made on an earlier call; the Customer did not "
                              f"decline or cancel it. {existing_note}").strip()
        bg["genuine"] = True
        bg["fake_booking_reason"] = "none"
        triggers["booking_commitment_expressed"] = True
        return
    if (
        agreement
        and final_stance == "confirmed"
        and bg.get("date_confirmed") is True
        and commitment_type not in {"vague_promise", "declined", "not_discussed"}
    ):
        commitment_type = (
            "date_time_confirmed" if bg.get("time_confirmed") is True else "date_only"
        )
        bg["commitment_type"] = commitment_type
        reason = "none"
        bg["fake_booking_reason"] = reason
    dated = commitment_type in {"date_time_confirmed", "date_only"} and bg.get("date_confirmed") is True
    # A follow-up on a booking made in an earlier call needs no date in this one.
    existing = commitment_type == "existing_booking_confirmed"
    direct_booking = (
        agreement
        and final_stance == "confirmed"
        and (dated or existing)
        and reason == "none"
    )
    if direct_booking and turns:
        customer_tids = {
            f"T{turn.index:03d}"
            for turn in turns
            if turn.speaker.strip().lower() == "customer"
        }
        direct_booking = (
            str(bg.get("agreement_turn_id") or "") in customer_tids
            and str(bg.get("final_stance_turn_id") or "") in customer_tids
        )
    if direct_booking:
        bg["genuine"] = True
        triggers["booking_commitment_expressed"] = True
        return

    # Only a direct, unconditional dated appointment is a genuine Booking.
    # Agent-only proposals and unresolved/conditional answers are fake.
    bg["genuine"] = False
    triggers["booking_commitment_expressed"] = False
    conditional = final_stance in {"conditional", "unclear"} or commitment_type == "vague_promise"
    if conditional:
        bg["customer_agreement_confirmed"] = False
        bg["agreement_turn_id"] = ""
        reason = "conditional_or_vague_booking"
        bg["fake_booking_reason"] = reason
        if commitment_type != "vague_promise":
            bg["commitment_type"] = "vague_promise"
    if reason in {"", "none"}:
        reason = (
            "no_booking_discussion"
            if bg.get("commitment_type") == "not_discussed"
            else "no_customer_agreement"
        )
        bg["fake_booking_reason"] = reason
    if bg.get("commitment_type") not in {
        "declined", "not_discussed", "no_customer_agreement",
        "date_only", "vague_promise",
    }:
        bg["commitment_type"] = "no_customer_agreement"
    existing = str(bg.get("evidence") or "").strip()
    prefix = f"Booking not genuine: {reason.replace('_', ' ')}."
    bg["evidence"] = f"{prefix} {existing}".strip()


_THANKS_RE = re.compile(
    r"\b(?:thank(?:s|\s+you)?|dhanyava+a+d|shukriya)\b|"
    r"આભાર|ધન્યવાદ|થેન્ક\s*યુ|થેન્ક\s*યૂ|"
    r"धन्यवाद|शुक्रिया|थैंक\s*यू",
    re.IGNORECASE,
)


def _agent_thanked_at_close(turns: List[Turn]) -> bool:
    """Return True only when an explicit thanks occurs in the last two Agent turns."""
    closing_agent_turns = [t for t in turns if t.speaker.strip().lower() == "agent"][-2:]
    return any(_THANKS_RE.search(t.text or "") for t in closing_agent_turns)


# ─────────────────────────────────────────────────────────────────────────────
# CORE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

NAME_EXTRACTION_PASS = True


def find_transcripts(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    found: List[Path] = []
    seen: set = set()
    for pat in TRANSCRIPT_GLOBS:
        for p in input_path.rglob(pat):
            if p.is_file() and p.resolve() not in seen:
                seen.add(p.resolve())
                found.append(p)
    return sorted(found)


def call_id_for(path: Path, root: Path) -> str:
    if path.parent != root and path.parent.name:
        return path.parent.name
    return transcript_stem(path)


def process_one(
    path: Path,
    client: LLMClient,
    root: Path,
    out_dir: Path,
    retries: int = 1,
    details_path: Optional[Path] = None,
    skip_classify: bool = False,   # ignored — routing is always deterministic
    skip_names: bool = False,       # ignored — names come from single-pass LLM response
    skip_scoring: bool = False,
    skip_coaching: bool = False,    # coaching is always built deterministically
    progress: Optional[Any] = None,
    *,
    _lora: Optional[Tuple[Any, Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Single-pass audit pipeline.

    One LLM call is made for normal transcripts (unless skip_scoring is True).
    The LLM call simultaneously:
      - extracts CRE/customer names and vehicle details
      - detects triggers
      - scores all active parameters with evidence
      - determines booking genuineness (BOOKING only)

    With AUDIT_LORA_MODE=active, the fine-tuned model first rates its trained
    transcript parameters from call type + transcript only; the stock call then
    covers only the remaining (CRM-dependent/untrained) parameters, or all of
    them if the fine-tuned output is rejected.

    A narrow Booking contradiction (main model says genuine while combined
    booking details are 0/6 and Python finds no grounded Customer stance) gets
    one compact transcript-only adjudication call. If JSON parsing fails,
    `retries` controls failure-only retries; the production worker uses one
    retry. `skip_classify`, `skip_names`, and `skip_coaching` are retained for
    call-site compatibility.
    """
    from .scoring import grade_band

    def _progress(percent: int, message: str) -> None:
        if callable(progress):
            progress(f"[{max(0, min(100, percent))}%] {message}")

    _progress(8, "Reading transcript")
    if retries > 0:
        print(
            f"    [INFO] Up to {retries} failure-only LLM retry attempt(s) enabled.",
            flush=True,
        )

    def _retry_after_failure(reason: str) -> Optional[Dict[str, Any]]:
        print(
            f"    [RETRY] {reason}; retrying once with deterministic settings "
            f"({retries} remaining before this retry).",
            file=sys.stderr,
            flush=True,
        )
        return process_one(
            path=path,
            client=client,
            root=root,
            out_dir=out_dir,
            retries=retries - 1,
            details_path=details_path,
            skip_classify=skip_classify,
            skip_names=skip_names,
            skip_scoring=skip_scoring,
            skip_coaching=skip_coaching,
            progress=progress,
            _lora=(lora_audit, lora_info) if lora_info["status"] != "off" else None,
        )

    raw = read_transcript(path)
    turns = parse_transcript(raw)
    # Turn IDs Brio may pick, as the prompt numbers them. Agreement and final-stance
    # fields must name a Customer turn (enforce_booking_integrity requires it).
    customer_ids = [f"T{turn.index:03d}" for turn in turns if turn.speaker.strip().lower() == "customer"]
    turn_ids = {
        "": [f"T{turn.index:03d}" for turn in turns],
        "agreement_turn_id": customer_ids,
        "final_stance_turn_id": customer_ids,
        "customer_commitment_turn_id": customer_ids,
    }
    cid = call_id_for(path, root)
    print(f"  auditing '{redact_call_id(cid)}' ({len(turns)} turns)...", flush=True)

    crm_details, context_path = load_crm_details(path, details_path)
    crm_details = crm_details or {}
    _progress(18, "Loaded CRM context and transcript turns")
    llm_runtimes: Dict[str, float] = {}
    llm_metrics: Dict[str, Any] = {}
    lora_audit: Optional[Dict[str, Dict[str, str]]] = None
    lora_info: Dict[str, Any] = {"status": "off"}

    # ── 1. Deterministic call-type routing (zero LLM calls) ────────────────────
    call_type, ct_evidence, ct_confidence = determine_audit_type_deterministic(
        path, crm_details
    )
    classification = {
        "call_type": call_type or "UNKNOWN",
        "decisive_evidence": ct_evidence,
        "rationale": "Deterministic routing — no LLM call.",
        "confidence": ct_confidence,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    # ── 2. Non-scored: unknown call type — bail before resolve_audit_route ────
    if call_type is None:
        print(
            f"    [MANUAL REVIEW] Could not determine call type. Evidence: {ct_evidence}",
            flush=True,
        )
        classification["manual_review"] = True
        classification["manual_review_reason"] = "Deterministic routing could not determine call type."
        null_route: Dict[str, Any] = {"audit_type": None, "audit_scope": "manual_review"}
        summary = build_non_scored_summary(cid, classification, null_route, crm_details)
        excel_result = {
            "metadata": {"call_type": call_type, "cre_name": summary.get("cre_name")},
            "audit": {},
            "crm_details": _redact_payload(crm_details),
            "other_comments": summary.get("auditor_comment", ""),
        }
        excel_path = append_audit_report(out_dir, cid, excel_result, None, summary)
        print(f"    [OK] Non-scored / manual review -> {excel_path.name}")
        return {"summary": summary, "excel": str(excel_path)}

    # resolve_audit_route only called when call_type is known
    route = resolve_audit_route(classification, crm_details)
    route["crm_completeness"] = assess_crm_completeness(crm_details)
    table = route["audit_type"]
    print(
        f"    call_type={call_type} | audit={CALL_TYPE_LABELS.get(table, 'Non-scored')}",
        flush=True,
    )

    wrong_tag_table = table if table in WEIGHTS else route.get("transcript_type")
    wrong_call_tagging = (
        route.get("disposition_match") is False
        and wrong_tag_table in WEIGHTS
    )
    if wrong_call_tagging:
        total_possible = sum(WEIGHTS[wrong_tag_table].values())
        possible_label = (
            str(int(total_possible))
            if float(total_possible).is_integer()
            else str(round(total_possible, 2))
        )
        summary = build_non_scored_summary(cid, classification, route, crm_details)
        summary.update({
            "call_type": CALL_TYPE_LABELS[wrong_tag_table],
            "score": 0.0,
            "score_display": f"0/{possible_label}",
            "score_pct": 0.0,
            "grade": "Not Genuine",
            "risk": "High",
            "wrong_call_tagging": True,
            "disposition": "WRONG CALL TAGGING — score overridden to 0",
        })
        excel_result = {
            "metadata": {"call_type": wrong_tag_table, "cre_name": summary.get("cre_name")},
            "audit": {
                pid: {"rating": "Zero", "evidence": "Wrong call tagging."}
                for pid in WEIGHTS[wrong_tag_table]
            },
            "crm_details": _redact_payload(crm_details),
        }
        excel_path = append_audit_report(
            out_dir, cid, excel_result, wrong_tag_table, summary
        )
        _progress(100, "Wrong call tagging saved as 0 in Excel")
        return {"summary": summary, "excel": str(excel_path)}

    if table is None:
        summary = build_non_scored_summary(cid, classification, route, crm_details)
        excel_result = {
            "metadata": {"call_type": call_type, "cre_name": summary.get("cre_name")},
            "audit": {},
            "crm_details": _redact_payload(crm_details),
        }
        excel_path = append_audit_report(out_dir, cid, excel_result, None, summary)
        return {"summary": summary, "excel": str(excel_path)}

    expected_duration = _transcript_audio_duration(path, crm_details)
    quality_issue = transcript_quality_issue(turns, expected_duration)
    if quality_issue:
        message = f"Transcript quality gate failed: {quality_issue}. Re-transcribe before auditing."
        print(f"    [MANUAL REVIEW] {message}", file=sys.stderr, flush=True)
        _progress(100, message)
        return None

    print(f"    workbook audit type={CALL_TYPE_LABELS[table]}", flush=True)

    # ── 3. Pre-LLM auto-scoring (deterministic, no LLM) ───────────────────────
    auto_full = compute_param_applicability(table, crm_details)
    _progress(28, f"Prepared {CALL_TYPE_LABELS[table]} scoring controls")
    if auto_full:
        print(f"    [AUTO-FULL] {len(auto_full)} param(s): {list(auto_full)}", flush=True)

    # ── 4. Skip path ───────────────────────────────────────────────────────────
    if skip_scoring:
        print("    [SKIP] main scoring LLM call disabled.", flush=True)
        sp_payload: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "fixed_call_type": table,
            "metadata": {},
            "evidence_ledger": {},
            "booking_genuine": None,
            "trigger_ledger": {},
            "transcript_assessment": {
                "speaker_roles_clear": True,
                "transcript_complete": True,
                "call_type_consistent": True,
                "call_type_mismatch_evidence": [],
            },
        }
        validation = ValidationResult(valid=True)
    else:
        # ── 5a. Fine-tuned model rates its trained transcript parameters ───────
        # Transcript + call type only (never CRM). On any failure the stock
        # model below keeps every parameter, exactly as in stock-only mode.
        if _lora is not None:  # failure-only retry: reuse, never re-run
            lora_audit, lora_info = _lora
        elif lora_enabled():
            _progress(30, "Fine-tuned model is rating transcript parameters")
            _t0 = time.time()
            lora_audit, lora_info = run_lora_audit(turns, table)
            llm_runtimes["lora_ratings"] = round(time.time() - _t0, 1)
            if lora_audit is None:
                print(
                    f"    [LORA FALLBACK] {lora_info.get('model')}: {lora_info.get('reason')}; "
                    "stock model scores all parameters.",
                    file=sys.stderr, flush=True,
                )
            else:
                print(f"    [LORA] {len(lora_audit)} param(s) rated by {lora_info['model']} "
                      f"({lora_info.get('scoring')})", flush=True)

        # ── 5b. Build single-pass prompt (parameters not rated above) ──────────
        system_prompt, user_prompt, schema = build_single_pass_prompt(
            turns, table, crm_details if crm_details else None, auto_full,
            model_owned=set(lora_audit or ()),
        )
        _progress(35, "LLM is auditing the complete call")

        # ── 6. LLM call (one normally; failure-only retry when configured) ────
        # Brio picks each evidence_state by probability when the fine-tuned
        # model was accepted (lean schema); any Brio failure generates instead.
        _t0 = time.time()
        raw_output = None
        if lora_audit and brio_enabled() and isinstance(client, OllamaClient):
            try:
                raw_output, brio_probs = score_json(
                    client, system_prompt, user_prompt, schema,
                    single_pass_options(), timeout=600, turn_ids=turn_ids,
                )
                llm_metrics = {
                    "scoring": "brio",
                    "brio_confidence": {path: confidence(p) for path, p in brio_probs.items()},
                }
                print(f"    [BRIO] stock model picked {len(brio_probs)} value(s) "
                      f"in {time.time() - _t0:.1f}s", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"    [BRIO FALLBACK] {type(exc).__name__}: {exc}; generating instead.",
                      file=sys.stderr, flush=True)
        if raw_output is None:
            try:
                raw_output = client.generate(
                    system_prompt, user_prompt,
                    format_schema=schema,
                    single_pass=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"    [ERROR] LLM call failed: {exc}", file=sys.stderr)
                if retries > 0:
                    return _retry_after_failure("LLM request failed")
                return None
            metrics_getter = getattr(client, "get_last_metrics", None)
            if callable(metrics_getter):
                llm_metrics = metrics_getter() or {}
        llm_runtimes["single_pass"] = round(time.time() - _t0, 1)
        _progress(88, "LLM response received")

        # ── 7. Parse JSON (no retry on failure) ────────────────────────────────
        try:
            sp_payload = extract_json(raw_output)
            _progress(92, "Parsed structured audit evidence")
        except Exception as exc:  # noqa: BLE001
            output_chars = len(raw_output) if isinstance(raw_output, str) else 0
            done_reason = llm_metrics.get("done_reason", "not reported")
            print(
                f"    [MANUAL REVIEW] JSON parse failed: {exc} "
                f"(output_chars={output_chars}, done_reason={done_reason})",
                file=sys.stderr,
            )
            if retries > 0:
                return _retry_after_failure("JSON parsing failed")
            return None

        # ── 8. Deterministic validation (DISABLED) ───────────────────────────
        validation = ValidationResult(valid=True)

    # ── 9. Deterministic Booking integrity reconciliation ─────────────────────
    needs_booking_adjudication = False
    if table == "BOOKING":
        booked_earlier = booked_on_earlier_call(crm_details)
        needs_booking_adjudication = (
            not skip_scoring and booking_needs_adjudication(sp_payload, turns, booked_earlier)
        )
        enforce_booking_integrity(sp_payload, turns, booked_earlier)
        if needs_booking_adjudication:
            _progress(94, "Resolving ambiguous Booking commitment")
            adj_system, adj_user, adj_schema = build_booking_adjudication_prompt(turns)
            _adj_t0 = time.time()
            adj_raw = None
            if brio_enabled() and isinstance(client, OllamaClient):
                try:
                    adj_raw, _ = score_json(
                        client, adj_system, adj_user, adj_schema,
                        single_pass_options(), timeout=600, turn_ids=turn_ids,
                    )
                    # Dry run: a Brio answer the checks reject is regenerated,
                    # not sent to manual review.
                    apply_booking_adjudication(copy.deepcopy(sp_payload), json.loads(adj_raw), turns)
                except Exception as exc:  # noqa: BLE001
                    adj_raw = None
                    print(f"    [BRIO FALLBACK] Booking adjudication {type(exc).__name__}: {exc}; "
                          "generating instead.", file=sys.stderr, flush=True)
            try:
                if adj_raw is None:
                    adj_raw = client.generate(
                        adj_system,
                        adj_user,
                        format_schema=adj_schema,
                        single_pass=True,
                    )
                adjudication = extract_json(adj_raw)
                apply_booking_adjudication(sp_payload, adjudication, turns)
                enforce_booking_integrity(sp_payload, turns)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"    [MANUAL REVIEW] Booking adjudication failed: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                _progress(100, "Booking adjudication failed; manual review required")
                return None
            llm_runtimes["booking_adjudication"] = round(time.time() - _adj_t0, 1)
            _progress(96, "Ambiguous Booking commitment resolved")
        booking_check = sp_payload.get("booking_genuine") or {}
        print(
            "    [BOOKING CHECK] "
            f"genuine={booking_check.get('genuine')} "
            f"agreement={booking_check.get('customer_agreement_confirmed')} "
            f"date={booking_check.get('date_confirmed')} "
            f"time={booking_check.get('time_confirmed')} "
            f"reason={booking_check.get('fake_booking_reason')}",
            flush=True,
        )

    # ── 10. Build standard audit dict from single-pass payload ────────────────
    standard_audit = _build_standard_audit_dict(sp_payload, table, auto_full, lora_audit)
    rule_set = set(auto_full)
    if "thank_you" in standard_audit and not _agent_thanked_at_close(turns):
        rule_set.add("thank_you")
        standard_audit["thank_you"] = {
            "rating": "Zero",
            "evidence": "No explicit thank-you was found in the CRE's closing turns.",
        }

    # ── 11. Assemble result in the shape downstream code expects ──────────────
    result: Dict[str, Any] = {
        "metadata": sp_payload.get("metadata") or {},
        "audit": standard_audit,
        "evidence_ledger": sp_payload.get("evidence_ledger") or {},
        "trigger_ledger": sp_payload.get("trigger_ledger") or {},
        "transcript_assessment": sp_payload.get("transcript_assessment") or {},
        "single_pass": True,
        "booking_adjudicated": needs_booking_adjudication,
        "validation_warnings": validation.warnings,
    }
    result["confidence"], result["check"] = _ratings_to_check(
        standard_audit, rule_set, lora_audit, llm_metrics.get("brio_confidence") or {},
        booking_decided_later=needs_booking_adjudication,
    )
    too_short = table == "BOOKING" and (sp_payload.get("booking_genuine") or {}).get("too_short") is True
    if too_short and "booking_genuine" not in result["check"]:
        result["check"].append("booking_genuine")

    # booking_genuine comes from the same LLM response
    if table == "BOOKING":
        result["booking_genuine"] = sp_payload.get("booking_genuine")
        bg = result["booking_genuine"]
        if isinstance(bg, dict) and not bg.get("genuine", True):
            print(
                f"    [BOOKING] NOT GENUINE — Evidence: "
                f"{str(bg.get('evidence', ''))[:120]}",
                flush=True,
            )

    result["classification"] = classification
    result["audit_route"] = route
    result["crm_details"] = _redact_payload(crm_details)
    result["external_context"] = {
        "provided": bool(crm_details),
        "source": context_path.name if context_path else None,
    }
    result["llm_runtimes"] = llm_runtimes
    result["llm_metrics"] = llm_metrics
    result["lora_audit"] = lora_info

    # Call type is a deterministic routing decision and is deliberately absent
    # from the model-editable metadata schema. Inject it for report/scoring code.
    result.setdefault("metadata", {})["call_type"] = table

    # CRM-sourced names override LLM-extracted names where available
    apply_crm_metadata(result, crm_details)

    # ── 11. Score and write the consolidated Excel workbook only ──────────────
    summary = build_management_summary(cid, result, table)
    _progress(97, "Calculated deterministic scores")

    excel_path = append_audit_report(out_dir, cid, result, table, summary)
    save_call_evidence(out_dir, cid, _redact_payload(
        _evidence_record(standard_audit, sp_payload, rule_set, lora_audit, table, result["confidence"])
    ))
    _progress(100, "Audit scores saved to Excel")

    final_score = summary.get("score")
    final_score_display = summary.get("score_display") or final_score
    final_grade = summary.get("grade") or grade_band(final_score)
    print(
        f"    [OK] {final_score_display if final_score_display is not None else 'N/A'} "
        f"({final_grade}) [{table}] -> {excel_path.name}"
    )
    return {"summary": summary, "excel": str(excel_path)}


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

LLM_BACKEND = "ollama"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Offline weighted /100 call-audit report generator.")
    ap.add_argument("--input", "-i", required=True,
                    help="Transcript file OR a directory to scan recursively.")
    ap.add_argument("--out", "-o", default="reports", help="Output directory.")
    ap.add_argument("--backend", "-b", default=LLM_BACKEND,
                    choices=["transformers", "ollama", "openai"], help="LLM backend.")
    ap.add_argument("--model", "-m", default=None,
                    help="Model id/path (defaults per backend).")
    ap.add_argument("--crm-snapshot", "--details", dest="details", type=Path,
                    default=None, help="CRM snapshot JSON (single-transcript mode).")
    ap.add_argument("--print", dest="do_print", action="store_true",
                    help="Also print the report(s) to stdout.")
    ap.add_argument("--skip-classify", dest="skip_classify", action="store_true",
                    help="Skip classification LLM call; fall back to CRM disposition or BOOKING.")
    ap.add_argument("--skip-names", dest="skip_names", action="store_true",
                    help="Skip name-extraction LLM call.")
    ap.add_argument("--skip-scoring", dest="skip_scoring", action="store_true",
                    help="Skip main scoring LLM call; output classification only.")
    ap.add_argument("--skip-coaching", dest="skip_coaching", action="store_true",
                    help="Omit coaching_dialogue from the audit.")
    args = ap.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 2

    transcripts = find_transcripts(input_path)
    if not transcripts:
        print(f"No transcripts found under {input_path}", file=sys.stderr)
        return 2

    if args.details and not args.details.exists():
        print(f"CRM details file not found: {args.details}", file=sys.stderr)
        return 2

    root = input_path if input_path.is_dir() else input_path.parent
    out_dir = Path(args.out)

    print(f"Backend: {args.backend} | Model: {args.model or '(default)'}")
    print(f"Scoring weights: {WEIGHT_SOURCE}")
    print("Loading model...")
    client = make_client(args.backend, args.model)

    total = len(transcripts)
    print(f"Found {total} transcript(s). Auditing one at a time...\n")

    done = failed = 0
    for path in transcripts:
        try:
            outcome = process_one(
                path, client, root, out_dir, 1,
                args.details, args.skip_classify, args.skip_names,
                args.skip_scoring, args.skip_coaching,
            )
        except Exception as exc:
            outcome = None
            print(f"    [ERROR] {exc}", file=sys.stderr)
        if outcome is not None:
            done += 1
            if args.do_print:
                print(json.dumps(outcome["summary"], ensure_ascii=False, indent=2))
        else:
            failed += 1
        print(f"[{done + failed}/{total}] done ({done} ok, {failed} failed)")

    print(f"\nDone. {done}/{total} audited. Reports in: {out_dir.resolve()}")
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
