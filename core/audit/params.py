"""
params.py — Scoring parameters, weights, and call-type definitions.

This is the primary file you will edit:
  - Change a parameter question           → PARAM_LABELS
  - Change grading instructions           → PARAM_SCORING_NOTES
  - Change point weights (fallback only)  → WEIGHTS
  - Add / remove a call type              → CALL_TYPES + WEIGHTS + WORKBOOK_TO_TABLE

The live weights come from resources/call_type_parameters.xlsx when it is present.
The WEIGHTS dict below is a portable fallback only.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import openpyxl  # type: ignore
except Exception:
    openpyxl = None  # type: ignore

# ── Workbook config ────────────────────────────────────────────────────────────
WORKBOOK_FILENAME = "call_type_parameters.xlsx"
WORKBOOK_CANDIDATE_PATHS = [
    Path(__file__).resolve().parents[2] / "resources" / WORKBOOK_FILENAME,
    Path(__file__).resolve().parents[3] / "resources" / WORKBOOK_FILENAME,
    Path("/home/pratham/NEW/required files") / "call type parameters.xlsx",
    Path(r"C:\Users\Dell\Desktop\CRE") / "call type parameters.xlsx",
]

# ── Grading scale ──────────────────────────────────────────────────────────────
# Parameters that can use CRM/DMS sidecar context for scoring.
EXTERNAL_CONTEXT_PARAMS = {
    "remarks_tagging", "proper_tagging", "task_crm",
}

# ── Filename → call-type mapping ───────────────────────────────────────────────
# Audio stem suffix codes determine call type when LLM classification is skipped.
FILENAME_CALL_TYPE_MAP: Dict[str, str] = {
    "BOD": "BOOKING",
    "BMS": "BOOKING",
    "ESC": "ESCALATION",
    "CML": "FOLLOW_UPS",
    "SNR": "SERVICE_NT_REQUIRED",
}

_FILENAME_CODE_RE = re.compile(r"[-_\s]+([A-Za-z]+)\s*$")


def _call_type_from_filename(path: Path) -> Optional[str]:
    """Return the call type encoded in the audio filename, or None."""
    m = _FILENAME_CODE_RE.search(path.stem)
    if m:
        return FILENAME_CALL_TYPE_MAP.get(m.group(1).upper())
    return None


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER DEFINITIONS
# Each key matches an entry in WEIGHTS.
# ─────────────────────────────────────────────────────────────────────────────

# PARAM_LABELS — the question sent to the LLM for each parameter.
# Edit here to change what the LLM is asked.
PARAM_LABELS: Dict[str, str] = {
    # ── Opening ───────────────────────────────────────────────────────────────
    "greet":               "Did CRE give a clear opening greeting?",
    "introduce":           "Did CRE introduce herself & mention she is calling from Kataria?",
    "confirm_vehicle_name":"Did CRE confirm the vehicle number AND the customer name?",
    "inform_recording":    "Did CRE inform the customer the call may be recorded?",
    "reason_for_call":     "Did CRE state the reason for the call?",
    "permission_to_talk":  "Did CRE take permission to talk?",

    # ── Product & Service Knowledge ───────────────────────────────────────────
    "prev_service_info":   "Did CRE provide the previous service information?",
    "ask_kms":             "Did CRE ask the current KMS of the vehicle?",
    "current_service_info":"Did CRE provide current service info based on tenure/mileage?",
    "cost_estimate":       "Did CRE provide a cost estimate for the service?",
    "warranty_info":       "Did CRE pitch OEM warranty / Extended Warranty / MCP?",
    "pickup_drop":         "Did CRE give free pick & drop information?",
    "booking_details":     "Did CRE confirm the booking date, time and address?",

    # ── Customer Handling ────────────────────────────────────────────────────
    "listen_understand":   "Did CRE listen to and understand the customer's need?",
    "dissatisfaction_inquiry": "Did CRE inquire into the customer's dissatisfaction?",
    "info_gathering":      "Did CRE gather information and offer solutions?",
    "reason_snr":          "Did CRE ask the reason for SERVICE NOT REQUIRED?",
    "professionalism":     "Overall demeanour and professionalism.",
    "conflict_mgmt":       "Did CRE manage conflict constructively?",
    "higher_billing":      "Did CRE explain the reason for higher billing?",

    # ── Software / CRM ───────────────────────────────────────────────────────
    "remarks_tagging":     "Did CRE enter proper customer & CRE remarks?",
    "proper_tagging":      "Did CRE apply proper tagging?",
    "task_crm":            "Did CRE manage the task properly in CRM?",

    # ── Closing ──────────────────────────────────────────────────────────────
    "future_service":      "Did CRE try to convince the customer for future service?",
    "standard_comm":       "Was CRE communication as per standards?",
    "thank_you":           "Did CRE close the call with a proper thank-you note?",
}

# PARAM_SCORING_NOTES — grading guidance sent alongside each question.
# Edit here to tighten or loosen how a parameter is scored.
PARAM_SCORING_NOTES: Dict[str, str] = {
    # ── Opening ───────────────────────────────────────────────────────────────
    "greet":               "Binary: Full for any clear Indian, regional, English, religious or time-based opening greeting, including reasonable ASR/transliterated forms; otherwise Zero.",
    "introduce":           "Full requires CRE name and dealership; Partial for only one of them; Zero for neither.",
    "confirm_vehicle_name":"Full requires both customer and vehicle confirmation; Partial for only one; Zero for neither.",
    "inform_recording":    "Full requires a clear recording disclosure; otherwise Zero.",
    "reason_for_call":     "Full requires a clear service-related purpose early in the call.",
    "permission_to_talk":  "Full requires explicitly checking whether the customer can talk; otherwise Zero.",

    # ── Product & Service Knowledge ───────────────────────────────────────────
    "prev_service_info":   "Full requires accurate, relevant prior-service details, not a vague claim.",
    "ask_kms":             "Full requires an explicit current odometer question and usable response/attempt.",
    "current_service_info":"Full requires a service recommendation tied to tenure or mileage.",
    "cost_estimate":       "Unconditional Full bypass; configured options are retained for workbook alignment.",
    "warranty_info": (
        "Full requires an accurate explanation of active OEM/EW/MCP benefits, or correct omission "
        "when every product is explicitly inactive/inapplicable; CRM evidence must identify the "
        "applicable status."
    ),
    "pickup_drop": (
        "Full requires clearly offering available pickup/drop support, or correct omission when "
        "CRM explicitly shows it is unavailable."
    ),
    "booking_details":     "Combined 6/4/0 control: Full for date+time+address, Partial for at least two, otherwise Zero.",

    # ── Customer Handling ────────────────────────────────────────────────────
    "listen_understand":   "Full requires acknowledgment and a response tailored to the stated need.",
    "dissatisfaction_inquiry": "Full requires probing the cause, impact and expected resolution.",
    "info_gathering":      "Full requires relevant questions followed by a tailored solution.",
    "reason_snr":          "Full requires capturing the specific reason service is not required.",
    "professionalism":     "Assess courtesy, clarity, ownership, interruptions, pressure and tone overall.",
    "conflict_mgmt":       "Full requires acknowledgment, de-escalation, ownership and a workable next step.",
    "higher_billing":      "Full requires a transparent, relevant explanation of the bill difference.",

    # ── Software / CRM ───────────────────────────────────────────────────────
    "remarks_tagging":     "Binary: Full when any entered CRE/customer remark accurately reflects the call; otherwise Zero. Identical remarks are allowed.",
    "proper_tagging":      "Use supplied CRM context showing the outcome tag matches the conversation.",
    "task_crm":            "Use supplied CRM context showing correct task status, owner and next action.",

    # ── Closing ──────────────────────────────────────────────────────────────
    "future_service":      "Full requires a credible future-service commitment or specific follow-up effort.",
    "standard_comm":       "Full requires clear, compliant, respectful and structured communication.",
    "thank_you":           "Full requires an explicit professional thank-you near the close; otherwise Zero.",
}

# ── Scorecard section grouping (display order only) ───────────────────────────
SECTIONS: List[Tuple[str, List[str]]] = [
    ("Opening", [
        "greet", "introduce", "confirm_vehicle_name",
        "inform_recording", "reason_for_call", "permission_to_talk",
    ]),
    ("Product & Service Knowledge", [
        "prev_service_info", "ask_kms", "current_service_info",
        "cost_estimate", "warranty_info", "pickup_drop",
        "booking_details",
    ]),
    ("Customer Handling", [
        "listen_understand", "dissatisfaction_inquiry", "info_gathering",
        "reason_snr", "professionalism", "conflict_mgmt", "higher_billing",
    ]),
    ("Software / CRM", [
        "remarks_tagging", "proper_tagging", "task_crm",
    ]),
    ("Closing", [
        "future_service", "standard_comm", "thank_you",
    ]),
]

# ─────────────────────────────────────────────────────────────────────────────
# POINT WEIGHTS PER CALL TYPE
# Each call type's weights must sum to 100 (or workbook total).
# This table is a fallback only — live weights load from the workbook.
# ─────────────────────────────────────────────────────────────────────────────
WEIGHTS: Dict[str, Dict[str, float]] = {
    "BOOKING": {
        "greet": 2, "introduce": 2, "confirm_vehicle_name": 4,
        "inform_recording": 4, "reason_for_call": 4, "permission_to_talk": 2,
        "prev_service_info": 5, "ask_kms": 5, "current_service_info": 5,
        "cost_estimate": 6, "warranty_info": 6, "pickup_drop": 6,
        "booking_details": 6, "listen_understand": 6, "info_gathering": 6,
        "professionalism": 18, "conflict_mgmt": 5,
        "remarks_tagging": 4, "standard_comm": 2, "thank_you": 2,
    },
    "FOLLOW_UPS": {
        "greet": 2, "introduce": 2, "confirm_vehicle_name": 4,
        "inform_recording": 4, "reason_for_call": 4, "permission_to_talk": 2,
        "prev_service_info": 8, "ask_kms": 8, "current_service_info": 10,
        "cost_estimate": 6, "warranty_info": 6, "pickup_drop": 6,
        "professionalism": 14, "remarks_tagging": 6, "standard_comm": 2,
        "proper_tagging": 6, "task_crm": 8, "thank_you": 2,
    },
    "SERVICE_NT_REQUIRED": {
        "greet": 2, "introduce": 2, "confirm_vehicle_name": 2,
        "inform_recording": 4, "reason_for_call": 4, "permission_to_talk": 2,
        "prev_service_info": 5, "ask_kms": 5, "current_service_info": 5,
        "cost_estimate": 5, "warranty_info": 6, "listen_understand": 6,
        "reason_snr": 8, "professionalism": 18, "conflict_mgmt": 5,
        "remarks_tagging": 5, "standard_comm": 2,
        "future_service": 8, "proper_tagging": 4, "thank_you": 2,
    },
    "ESCALATION": {
        "greet": 2, "introduce": 2, "confirm_vehicle_name": 2,
        "inform_recording": 4, "reason_for_call": 4, "permission_to_talk": 2,
        "prev_service_info": 5, "ask_kms": 5, "current_service_info": 5,
        "cost_estimate": 6, "warranty_info": 6, "listen_understand": 6,
        "dissatisfaction_inquiry": 4, "professionalism": 14, "conflict_mgmt": 10,
        "remarks_tagging": 5, "standard_comm": 2,
        "higher_billing": 4, "future_service": 5, "proper_tagging": 5,
        "thank_you": 2,
    },
}

# Workbook sheet names → internal call type keys.
WORKBOOK_TO_TABLE = {
    "Booking":             "BOOKING",
    "Service Nt Required": "SERVICE_NT_REQUIRED",
    "Follow ups":          "FOLLOW_UPS",
    "Escalation":          "ESCALATION",
}

# Column order in the workbook's "main" sheet (excluding leading call-type and trailing Marks columns).
WORKBOOK_QUESTION_ORDER = {
    "BOOKING": [
        "greet", "introduce", "confirm_vehicle_name", "inform_recording",
        "reason_for_call", "permission_to_talk", "prev_service_info",
        "ask_kms", "current_service_info", "cost_estimate", "warranty_info",
        "pickup_drop", "booking_details", "listen_understand",
        "dissatisfaction_inquiry", "info_gathering", "reason_snr",
        "professionalism", "conflict_mgmt", "dms_validation",
        "remarks_tagging", "standard_comm", "higher_billing",
        "future_service", "proper_tagging", "task_crm", "thank_you",
    ],
    "FOLLOW_UPS": [
        "greet", "introduce", "confirm_vehicle_name", "inform_recording",
        "reason_for_call", "permission_to_talk", "prev_service_info",
        "ask_kms", "current_service_info", "cost_estimate", "warranty_info",
        "pickup_drop", "booking_details", "listen_understand",
        "dissatisfaction_inquiry", "info_gathering", "reason_snr",
        "professionalism", "conflict_mgmt", "dms_validation",
        "remarks_tagging", "standard_comm", "higher_billing",
        "future_service", "proper_tagging", "task_crm", "thank_you",
    ],
    "SERVICE_NT_REQUIRED": [
        "greet", "introduce", "confirm_vehicle_name", "inform_recording",
        "reason_for_call", "permission_to_talk", "prev_service_info",
        "ask_kms", "current_service_info", "cost_estimate", "warranty_info",
        "pickup_drop", "booking_details", "listen_understand",
        "dissatisfaction_inquiry", "info_gathering", "reason_snr",
        "professionalism", "conflict_mgmt", "dms_validation",
        "remarks_tagging", "standard_comm", "higher_billing",
        "future_service", "proper_tagging", "task_crm", "thank_you",
    ],
    "ESCALATION": [
        "greet", "introduce", "confirm_vehicle_name", "inform_recording",
        "reason_for_call", "permission_to_talk", "prev_service_info",
        "ask_kms", "current_service_info", "cost_estimate", "warranty_info",
        "pickup_drop", "booking_details", "listen_understand",
        "dissatisfaction_inquiry", "info_gathering", "reason_snr",
        "professionalism", "conflict_mgmt", "dms_validation",
        "remarks_tagging", "standard_comm", "higher_billing",
        "future_service", "proper_tagging", "task_crm", "thank_you",
    ],
}


def _load_weights_from_workbook() -> Optional[Dict[str, Dict[str, float]]]:
    """Load live point weights from the workbook. Returns None if unavailable."""
    if openpyxl is None:
        return None
    path = next((p for p in WORKBOOK_CANDIDATE_PATHS if p.exists()), None)
    if path is None:
        return None
    try:
        wb = openpyxl.load_workbook(path, data_only=True)
    except Exception:
        return None
    if "main" not in wb.sheetnames:
        return None
    loaded: Dict[str, Dict[str, float]] = {}
    for sheet_name, table in WORKBOOK_TO_TABLE.items():
        order = WORKBOOK_QUESTION_ORDER.get(table, [])
        for row in wb["main"].iter_rows(min_row=2, values_only=True):
            if not row or row[0] is None:
                continue
            if str(row[0]).strip() != sheet_name:
                continue
            weights: Dict[str, float] = {
                pid: float(v)
                for pid, v in zip(order, row[1:1 + len(order)])
                if isinstance(v, (int, float)) and float(v) > 0
            }
            loaded[table] = weights
            break
    return loaded or None


# Load live weights; fall back to embedded table.
_workbook_weights = _load_weights_from_workbook()
if _workbook_weights:
    WEIGHTS = _workbook_weights
    WEIGHT_SOURCE = "workbook"
else:
    WEIGHT_SOURCE = "embedded"

# Exact score ladders from Kataria_Scoring_Parameters.xlsx.  The LLM selects a
# qualitative evidence level; Python is the sole authority that maps that level
# to points.  Intentional business exceptions (cost bypass and binary remarks)
# are represented by deterministic rules, not by changing these maxima.
# DMS validation is not scored; its points moved to professionalism
# (Booking 14->18, Escalation 10->14, SNR 10->18) so every table totals 100.
SCORE_OPTIONS: Dict[str, Dict[str, Dict[str, float]]] = {
    "BOOKING": {
        "greet": {"Full": 2, "Zero": 0},
        "introduce": {"Full": 2, "Partial": 1, "Zero": 0},
        "confirm_vehicle_name": {"Full": 4, "Partial": 2, "Zero": 0},
        "inform_recording": {"Full": 4, "Zero": 0},
        "reason_for_call": {"Full": 4, "Zero": 0},
        "permission_to_talk": {"Full": 2, "Zero": 0},
        "prev_service_info": {"Full": 5, "Zero": 0},
        "ask_kms": {"Full": 5, "Zero": 0},
        "current_service_info": {"Full": 5, "Zero": 0},
        "cost_estimate": {"Full": 6, "Partial": 4, "Zero": 0},
        "warranty_info": {"Full": 6, "Partial": 4, "Zero": 0},
        "pickup_drop": {"Full": 6, "Partial": 4, "Zero": 0},
        "booking_details": {"Full": 6, "Partial": 4, "Zero": 0},
        "listen_understand": {"Full": 6, "Partial": 3, "Zero": 0},
        "info_gathering": {"Full": 6, "Partial": 3, "Zero": 0},
        "professionalism": {"Full": 18, "Good": 10, "Partial": 5, "Zero": 0},
        "conflict_mgmt": {"Full": 5, "Partial": 3, "Zero": 0},
        "remarks_tagging": {"Full": 4, "Zero": 0},
        "standard_comm": {"Full": 2, "Zero": 0},
        "thank_you": {"Full": 2, "Zero": 0},
    },
    "ESCALATION": {
        "greet": {"Full": 2, "Zero": 0},
        "introduce": {"Full": 2, "Partial": 1, "Zero": 0},
        "confirm_vehicle_name": {"Full": 2, "Partial": 1, "Zero": 0},
        "inform_recording": {"Full": 4, "Zero": 0},
        "reason_for_call": {"Full": 4, "Zero": 0},
        "permission_to_talk": {"Full": 2, "Zero": 0},
        "prev_service_info": {"Full": 5, "Zero": 0},
        "ask_kms": {"Full": 5, "Zero": 0},
        "current_service_info": {"Full": 5, "Partial": 3, "Zero": 0},
        "cost_estimate": {"Full": 6, "Zero": 0},
        "warranty_info": {"Full": 6, "Partial": 3, "Zero": 0},
        "listen_understand": {"Full": 6, "Partial": 3, "Zero": 0},
        "dissatisfaction_inquiry": {"Full": 4, "Zero": 0},
        "professionalism": {"Full": 14, "Good": 8, "Partial": 3, "Zero": 0},
        "higher_billing": {"Full": 4, "Zero": 0},
        "conflict_mgmt": {"Full": 10, "Partial": 5, "Zero": 0},
        "future_service": {"Full": 5, "Zero": 0},
        "remarks_tagging": {"Full": 5, "Zero": 0},
        "proper_tagging": {"Full": 5, "Zero": 0},
        "standard_comm": {"Full": 2, "Zero": 0},
        "thank_you": {"Full": 2, "Zero": 0},
    },
    "FOLLOW_UPS": {
        "greet": {"Full": 2, "Zero": 0},
        "introduce": {"Full": 2, "Partial": 1, "Zero": 0},
        "confirm_vehicle_name": {"Full": 4, "Partial": 2, "Zero": 0},
        "inform_recording": {"Full": 4, "Zero": 0},
        "reason_for_call": {"Full": 4, "Zero": 0},
        "permission_to_talk": {"Full": 2, "Zero": 0},
        "prev_service_info": {"Full": 8, "Partial": 4, "Zero": 0},
        "ask_kms": {"Full": 8, "Zero": 0},
        "current_service_info": {"Full": 10, "Partial": 6, "Zero": 0},
        "cost_estimate": {"Full": 6, "Zero": 0},
        "warranty_info": {"Full": 6, "Partial": 3, "Zero": 0},
        "pickup_drop": {"Full": 6, "Partial": 3, "Zero": 0},
        "professionalism": {"Full": 14, "Good": 8, "Partial": 4, "Zero": 0},
        "task_crm": {"Full": 8, "Partial": 4, "Zero": 0},
        "proper_tagging": {"Full": 6, "Zero": 0},
        "remarks_tagging": {"Full": 6, "Zero": 0},
        "standard_comm": {"Full": 2, "Zero": 0},
        "thank_you": {"Full": 2, "Zero": 0},
    },
    "SERVICE_NT_REQUIRED": {
        "greet": {"Full": 2, "Zero": 0},
        "introduce": {"Full": 2, "Partial": 1, "Zero": 0},
        "confirm_vehicle_name": {"Full": 2, "Partial": 1, "Zero": 0},
        "inform_recording": {"Full": 4, "Zero": 0},
        "reason_for_call": {"Full": 4, "Zero": 0},
        "permission_to_talk": {"Full": 2, "Zero": 0},
        "prev_service_info": {"Full": 5, "Zero": 0},
        "ask_kms": {"Full": 5, "Zero": 0},
        "current_service_info": {"Full": 5, "Zero": 0},
        "cost_estimate": {"Full": 5, "Zero": 0},
        "warranty_info": {"Full": 6, "Partial": 3, "Zero": 0},
        "listen_understand": {"Full": 6, "Partial": 3, "Zero": 0},
        "reason_snr": {"Full": 8, "Partial": 4, "Zero": 0},
        "professionalism": {"Full": 18, "Partial": 9, "Zero": 0},
        "conflict_mgmt": {"Full": 5, "Partial": 3, "Zero": 0},
        "future_service": {"Full": 8, "Partial": 4, "Zero": 0},
        "proper_tagging": {"Full": 4, "Zero": 0},
        "remarks_tagging": {"Full": 5, "Zero": 0},
        "standard_comm": {"Full": 2, "Zero": 0},
        "thank_you": {"Full": 2, "Zero": 0},
    },
}

# Maxima and totals are derived from the exact score ladders above.  This also
# corrects ESCALATION professionalism to 10 and makes every table total 100.
WEIGHTS = {
    table: {pid: levels["Full"] for pid, levels in params.items()}
    for table, params in SCORE_OPTIONS.items()
}
WEIGHT_SOURCE = "workbook-options"


def allowed_score_levels(table: str, pid: str) -> tuple[str, ...]:
    return tuple(SCORE_OPTIONS[table][pid].keys())


def points_for_level(table: str, pid: str, level: str) -> float:
    return float(SCORE_OPTIONS[table][pid][level])


# ── Call-type constants ────────────────────────────────────────────────────────
CALL_TYPES = ["BOOKING", "SERVICE_NT_REQUIRED", "FOLLOW_UPS", "ESCALATION"]

CALL_TYPE_LABELS = {
    "BOOKING":              "Booking",
    "SERVICE_NT_REQUIRED":  "Service Nt Required",
    "FOLLOW_UPS":           "Follow ups",
    "ESCALATION":           "Escalation",
}

TRANSCRIPT_TYPES = ["BOOKING", "SERVICE_NT_REQUIRED", "FOLLOW_UPS", "ESCALATION", "DND"]

TRANSCRIPT_TYPE_LABELS = {
    **CALL_TYPE_LABELS,
    "DND": "DND",
}

CRM_DISPOSITION_ALIASES: Dict[str, str] = {
    "BOOK_MY_SERVICE":    "BOOK_MY_SERVICE",
    "BOOK_SERVICE":       "BOOK_MY_SERVICE",
    "BOOKING":            "BOOK_MY_SERVICE",
    "SNR":                "SNR",
    "SERVICE_NT_REQUIRED":"SNR",
    "SERVICE_NOT_REQUIRED":"SNR",
    "CALL_ME_LATER":      "CALL_ME_LATER",
    "FOLLOW_UPS":         "CALL_ME_LATER",
    "OTHER_ESCALATION":   "ESCALATION",
    "ESCALATION":         "ESCALATION",
    "DND":                "DND",
    "VOICE_MAIL":         "VOICE_MAIL",
    "VOICEMAIL":          "VOICE_MAIL",
    "NO_CONVERSATION":    "NO_CONVERSATION",
}

CALL_TYPE_ALIASES: Dict[str, str] = {
    "BOOKING":             "BOOKING",
    "SERVICE_NT_REQUIRED": "SERVICE_NT_REQUIRED",
    "FOLLOW_UPS":          "FOLLOW_UPS",
    "ESCALATION":          "ESCALATION",
}


# ── Normalisation helpers ──────────────────────────────────────────────────────
def _normalization_key(raw: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(raw or "").strip().upper()).strip("_")


def normalize_call_type(raw: Any) -> str:
    text = _normalization_key(raw)
    normalized = CALL_TYPE_ALIASES.get(text)
    if normalized not in CALL_TYPES:
        raise ValueError(f"Unsupported call type: {raw!r}")
    return normalized


def normalize_transcript_type(raw: Any) -> str:
    text = _normalization_key(raw)
    aliases = {
        "BOOKED":               "BOOKING",
        "SNR":                  "SERVICE_NT_REQUIRED",
        "SERVICE_NOT_REQUIRED": "SERVICE_NT_REQUIRED",
        "FOLLOW_UP":            "FOLLOW_UPS",
        "FOLLOWUP":             "FOLLOW_UPS",
        "DO_NOT_DISTURB":       "DND",
    }
    normalized = aliases.get(text, text)
    if normalized not in TRANSCRIPT_TYPES:
        raise ValueError(f"Unsupported transcript disposition: {raw!r}")
    return normalized


def _extract_cre_disposition(details: Dict[str, Any]) -> str:
    disposition = details.get("disposition")
    if isinstance(disposition, dict):
        raw = disposition.get("type") or disposition.get("name")
    else:
        raw = disposition
    raw = raw or details.get("cre_disposition") or details.get("call_disposition")
    if not raw:
        return ""
    key = _normalization_key(raw)
    return CRM_DISPOSITION_ALIASES.get(key, key)
