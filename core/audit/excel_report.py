"""Excel-only audit register writer.

The visible workbook follows the audit-register layout supplied in Book4.xlsx,
with human-auditor workflow columns removed. It is the only persistent audit
output; the UI reconstructs its score-only view from the hidden workbook index.
"""
from __future__ import annotations

import json
import shutil
import threading
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from .params import (
    CALL_TYPE_LABELS,
    SCORE_OPTIONS,
    WEIGHTS,
    allowed_score_levels,
    points_for_level,
)
from .utils import _display, redact_call_id

try:
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment
except ImportError:  # pragma: no cover - deployment dependency guard
    load_workbook = None  # type: ignore[assignment]
    Alignment = None  # type: ignore[assignment]


REPORT_WORKBOOK_NAME = "audit_reports.xlsx"
# Per-call evidence (source, reason, transcript turn IDs) for the dashboard.
# Kept beside the workbook, one small file per call, so Excel saves stay fast.
EVIDENCE_DIR = "evidence"
REPORT_TEMPLATE = Path(__file__).resolve().parents[2] / "resources" / "audit_report_template.xlsx"
REPORT_SHEET = "AUDIT REPORTS"
INDEX_SHEET = "__INDEX"
RUNTIME_SHEET = "__RUNTIMES"
CALL_TYPE_SHEETS = {
    "BOOKING": "BOOKING",
    "ESCALATION": "ESCALATION",
    "SERVICE_NT_REQUIRED": "SERVICE NOT REQUIRED",
    "FOLLOW_UPS": "FOLLOW UP",
}

_WRITE_LOCK = threading.Lock()

PARAMETER_COLUMNS = [
    "greet",
    "introduce",
    "confirm_vehicle_name",
    "inform_recording",
    "reason_for_call",
    "permission_to_talk",
    "prev_service_info",
    "ask_kms",
    "current_service_info",
    "cost_estimate",
    "warranty_info",
    "pickup_drop",
    "booking_details",
    "listen_understand",
    "dissatisfaction_inquiry",
    "info_gathering",
    "reason_snr",
    "professionalism",
    "conflict_mgmt",
    "dms_validation",          # template column kept; not scored any more -> N/A
    "remarks_tagging",
    "standard_comm",
    "higher_billing",
    "future_service",
    "proper_tagging",
    "task_crm",
    "thank_you",
]

LEGACY_HUMAN_COLUMNS = {"Name of Audit Executive", "Name of SMM"}


def _nested(data: Dict[str, Any], *keys: str) -> Any:
    node: Any = data
    for key in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _text(value: Any) -> str:
    return "" if value in (None, "") else str(value).strip()


def _earned(result: Dict[str, Any], table: Optional[str], pid: str) -> float:
    if not table or table not in WEIGHTS:
        return 0.0
    if pid not in WEIGHTS[table]:
        return 0.0
    entry = (result.get("audit") or {}).get(pid) or {}
    rating = _text(entry.get("rating")).title()
    if rating == "Half":  # legacy report compatibility
        rating = "Partial"
    if rating not in allowed_score_levels(table, pid):
        rating = "Zero"
    return round(points_for_level(table, pid, rating), 2)


def _parameter_values(result: Dict[str, Any], table: Optional[str]) -> list[Any]:
    weights = WEIGHTS.get(table, {}) if table else {}
    values: list[Any] = []
    for pid in PARAMETER_COLUMNS:
        if float(weights.get(pid, 0.0)) > 0:
            values.append(_earned(result, table, pid))
        else:
            values.append("N/A")
    return values


def _major_comments(summary: Dict[str, Any]) -> str:
    statuses = []
    if summary.get("booking_genuine") is False:
        statuses.append("FAKE_BOOKING")
    if summary.get("wrong_call_tagging") is True:
        statuses.append("WRONG_CALL_TAGGING")
    if not statuses and summary.get("booking_integrity_warning"):
        statuses.append("BOOKING_INTEGRITY_WARNING")
    return ";".join(statuses)


@lru_cache(maxsize=1)
def _column_names() -> Dict[str, str]:
    """Parameter id -> its column header in the report template."""
    sheet = load_workbook(REPORT_TEMPLATE, read_only=True)[REPORT_SHEET]
    header = next(sheet.iter_rows(max_row=1, values_only=True))
    first = header.index("Greeting")
    return dict(zip(PARAMETER_COLUMNS, header[first:first + len(PARAMETER_COLUMNS)]))


def _check_comment(result: Dict[str, Any]) -> str:
    """Ratings the model was unsure of (Brio confidence below the check level), and a
    Booking call too short to judge (no confidence)."""
    check = result.get("check") or []
    if not check:
        return ""
    scores = result.get("confidence") or {}
    names = _column_names()
    return "Check: " + "; ".join(
        f"{'Booking genuine' if pid == 'booking_genuine' else names.get(pid, pid)}"
        + (f" ({scores[pid]:.0%})" if pid in scores else "")
        for pid in check
    )


def build_non_scored_summary(
    call_id: str,
    classification: Dict[str, Any],
    route: Dict[str, Any],
    details: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the JSON/Excel summary for a non-scored operational disposition."""
    customer = details.get("customer", {}) if isinstance(details.get("customer"), dict) else {}
    disposition = str(route.get("cre_disposition") or "Not scored").replace("_", " ").title()
    is_dnd = route.get("transcript_type") == "DND" or route.get("cre_disposition") == "DND"
    return {
        "call_id": redact_call_id(call_id),
        "cre_name": _display(
            details.get("cre", {}).get("name") if isinstance(details.get("cre"), dict) else None
        ),
        "customer_name": _display(customer.get("name")),
        "call_type": disposition,
        "score": None,
        "grade": "Not scored",
        "risk": "Compliance action" if is_dnd else "Not assessed",
        "confidence": classification.get("confidence", "Low"),
        "coverage_pct": None,
        "disposition": (
            "Suppress future calling and verify CRM update"
            if is_dnd else "Operational disposition only"
        ),
        "final_outcome": disposition,
        "core_objection": (
            "Customer requested no further calls" if is_dnd else "No recording available"
        ),
        "cre_disposition": route.get("cre_disposition", "Not supplied"),
        "transcript_type": route.get("transcript_type", "Not assessed"),
        "audit_scope": route.get("audit_scope", "Disposition verification only"),
        "audit_type_source": route.get("audit_type_source", "CRE disposition"),
        "disposition_match": route.get("disposition_match"),
        "auditor_comment": route.get("auditor_comment", ""),
        "crm_details_status": route.get("crm_completeness", {}).get("status", "Not assessed"),
        "crm_missing_fields": ", ".join(
            route.get("crm_completeness", {}).get("missing_fields", [])
        ),
    }


def _build_row(
    call_id: str,
    result: Dict[str, Any],
    table: Optional[str],
    summary: Dict[str, Any],
) -> list[Any]:
    crm = result.get("crm_details") or {}
    meta = result.get("metadata") or {}
    vehicle_number = _text(meta.get("vehicle_number")) or _text(
        _nested(crm, "vehicle", "registration_number")
    )
    phone = _text(_nested(crm, "customer", "phone"))
    contact_vehicle = " / ".join(value for value in (phone, vehicle_number) if value)
    booking_type = (
        _text(_nested(crm, "booking", "service_type"))
        or _text(_nested(crm, "vehicle", "next_service_type"))
        or _text(_nested(crm, "service_profile", "campaign"))
    )
    cre_name = _text(meta.get("cre_name")) or _text(summary.get("cre_name"))
    cre_id = _text(_nested(crm, "cre", "employee_id"))
    call_date = _text(_nested(crm, "call_meta", "call_date"))
    other_comments = _check_comment(result)
    report_type = CALL_TYPE_LABELS.get(table, _text(summary.get("call_type")) or "Not scored")

    parameter_values = _parameter_values(result, table)
    if summary.get("booking_genuine") is False or summary.get("wrong_call_tagging") is True:
        parameter_values = [0 if value != "N/A" else value for value in parameter_values]

    return [
        report_type,
        datetime.now().astimezone().replace(tzinfo=None),
        booking_type,
        cre_name,
        cre_id,
        call_date,
        contact_vehicle or call_id,
        *parameter_values,
        _major_comments(summary),
        other_comments,
        summary.get("score_display") or summary.get("score"),
    ]


def _remove_legacy_human_columns(sheet: Any) -> None:
    """Upgrade an existing manual-audit workbook to the AI-only layout."""
    legacy_indexes = [
        cell.column
        for cell in sheet[1]
        if _text(cell.value) in LEGACY_HUMAN_COLUMNS
    ]
    for column_index in sorted(legacy_indexes, reverse=True):
        sheet.delete_cols(column_index, 1)


def _table_from_report_label(value: Any) -> Optional[str]:
    key = " ".join(_text(value).upper().replace("-", " ").replace("_", " ").split())
    return {
        "BOOKING": "BOOKING",
        "ESCALATION": "ESCALATION",
        "SERVICE NOT REQUIRED": "SERVICE_NT_REQUIRED",
        "SERVICE NT REQUIRED": "SERVICE_NT_REQUIRED",
        "FOLLOW UP": "FOLLOW_UPS",
        "FOLLOW UPS": "FOLLOW_UPS",
    }.get(key)


def _ensure_call_type_sheets(workbook: Any) -> None:
    """Create four output sheets and migrate the old combined sheet safely."""
    if REPORT_SHEET not in workbook.sheetnames:
        source = next((workbook[n] for n in CALL_TYPE_SHEETS.values() if n in workbook.sheetnames), None)
        if source is None:
            raise KeyError("Audit workbook has no report sheet to use as a template")
        for sheet_name in CALL_TYPE_SHEETS.values():
            if sheet_name not in workbook.sheetnames:
                created = workbook.copy_worksheet(source)
                created.title = sheet_name
                if created.max_row > 1:
                    created.delete_rows(2, created.max_row - 1)
        return

    legacy = workbook[REPORT_SHEET]
    _remove_legacy_human_columns(legacy)
    legacy_rows = {
        row_number: [legacy.cell(row_number, col).value for col in range(1, legacy.max_column + 1)]
        for row_number in range(2, legacy.max_row + 1)
        if any(legacy.cell(row_number, col).value is not None for col in range(1, legacy.max_column + 1))
    }
    legacy_index = []
    if INDEX_SHEET in workbook.sheetnames:
        old_index = workbook[INDEX_SHEET]
        for row in old_index.iter_rows(min_row=2, values_only=True):
            if row and _text(row[0]) and len(row) > 1 and row[1] is not None:
                legacy_index.append((_text(row[0]), int(row[1])))
        workbook.remove(old_index)

    for sheet_name in CALL_TYPE_SHEETS.values():
        if sheet_name not in workbook.sheetnames:
            created = workbook.copy_worksheet(legacy)
            created.title = sheet_name
            if created.max_row > 1:
                created.delete_rows(2, created.max_row - 1)
            created.freeze_panes = "A2"

    migrated_locations: Dict[int, tuple[str, int]] = {}
    for old_row, values in legacy_rows.items():
        table = _table_from_report_label(values[0] if values else None)
        if table not in CALL_TYPE_SHEETS:
            continue
        target = workbook[CALL_TYPE_SHEETS[table]]
        new_row = max(2, target.max_row + 1)
        for column, value in enumerate(values, 1):
            target.cell(new_row, column, value)
        migrated_locations[old_row] = (target.title, new_row)

    workbook.remove(legacy)
    index_sheet = workbook.create_sheet(INDEX_SHEET)
    index_sheet.append(["call_id", "sheet", "row"])
    for call_id, old_row in legacy_index:
        location = migrated_locations.get(old_row)
        if location:
            index_sheet.append([call_id, location[0], location[1]])
    index_sheet.sheet_state = "hidden"


RUNTIME_COLUMNS = ["call_id", "transcription_s", "audit_s", "updated_at", "stock_llm_s", "finetuned_llm_s"]


def _ensure_runtime_sheet(workbook: Any) -> Any:
    if RUNTIME_SHEET not in workbook.sheetnames:
        sheet = workbook.create_sheet(RUNTIME_SHEET)
        sheet.append(RUNTIME_COLUMNS)
    else:
        sheet = workbook[RUNTIME_SHEET]
        for column, name in enumerate(RUNTIME_COLUMNS, 1):  # older workbooks lack the LLM columns
            if sheet.cell(1, column).value is None:
                sheet.cell(1, column, name)
    sheet.sheet_state = "hidden"
    return sheet


def _runtime_row(sheet: Any, call_id: str) -> int:
    for row in sheet.iter_rows(min_row=2, values_only=False):
        if _text(row[0].value) == call_id:
            return row[0].row
    row_number = max(2, sheet.max_row + 1)
    sheet.cell(row_number, 1, call_id)
    return row_number


def record_call_runtime(out_dir: Path, call_id: str, phase: str, runtime_s: Any) -> None:
    """Persist one completed job runtime in the consolidated Excel workbook."""
    if load_workbook is None or phase not in {"transcription", "audit"}:
        return
    try:
        seconds = round(float(runtime_s), 1)
    except (TypeError, ValueError):
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / REPORT_WORKBOOK_NAME
    with _WRITE_LOCK:
        if not report_path.exists():
            shutil.copy2(REPORT_TEMPLATE, report_path)
        workbook = load_workbook(report_path)
        _ensure_call_type_sheets(workbook)
        sheet = _ensure_runtime_sheet(workbook)
        row_number = _runtime_row(sheet, call_id)
        sheet.cell(row_number, 2 if phase == "transcription" else 3, seconds)
        sheet.cell(row_number, 4, datetime.now().astimezone().replace(tzinfo=None))
        temporary_path = report_path.with_name(f".{report_path.name}.tmp.xlsx")
        workbook.save(temporary_path)
        workbook.close()
        temporary_path.replace(report_path)


def append_audit_report(
    out_dir: Path,
    call_id: str,
    result: Dict[str, Any],
    table: Optional[str],
    summary: Dict[str, Any],
) -> Path:
    """Insert or replace one call row in its call-type Excel sheet."""
    if load_workbook is None:
        raise RuntimeError("openpyxl is required for Excel audit reports")
    if not REPORT_TEMPLATE.exists():
        raise FileNotFoundError(f"Excel report template not found: {REPORT_TEMPLATE}")

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / REPORT_WORKBOOK_NAME
    with _WRITE_LOCK:
        if not report_path.exists():
            shutil.copy2(REPORT_TEMPLATE, report_path)

        workbook = load_workbook(report_path)
        _ensure_call_type_sheets(workbook)
        _ensure_runtime_sheet(workbook)
        for visible_name in CALL_TYPE_SHEETS.values():
            _remove_legacy_human_columns(workbook[visible_name])
        if INDEX_SHEET not in workbook.sheetnames:
            index_sheet = workbook.create_sheet(INDEX_SHEET)
            index_sheet.append(["call_id", "sheet", "row"])
            index_sheet.sheet_state = "hidden"
        else:
            index_sheet = workbook[INDEX_SHEET]

        if table not in CALL_TYPE_SHEETS:
            temporary_path = report_path.with_name(f".{report_path.name}.tmp.xlsx")
            workbook.save(temporary_path)
            workbook.close()
            temporary_path.replace(report_path)
            return report_path

        sheet = workbook[CALL_TYPE_SHEETS[table]]

        row_number = None
        index_row_number = None
        for row in index_sheet.iter_rows(min_row=2, values_only=False):
            if _text(row[0].value) == call_id:
                index_row_number = row[0].row
                if _text(row[1].value) == sheet.title:
                    row_number = int(row[2].value)
                break
        if row_number is None:
            row_number = max(2, sheet.max_row + 1)
            if index_row_number is None:
                index_sheet.append([call_id, sheet.title, row_number])
            else:
                index_sheet.cell(index_row_number, 2, sheet.title)
                index_sheet.cell(index_row_number, 3, row_number)

        values = _build_row(call_id, result, table, summary)
        for column, value in enumerate(values, 1):
            sheet.cell(row_number, column, value)

        # LLM time inside this audit: stock model (single pass + booking
        # adjudication) and the fine-tuned model (empty when it did not run).
        llm = result.get("llm_runtimes") or {}
        stock = [llm[key] for key in ("single_pass", "booking_adjudication") if llm.get(key) is not None]
        runtime_sheet = workbook[RUNTIME_SHEET]
        runtime_row = _runtime_row(runtime_sheet, call_id)
        # .value (not cell(..., value)) so None clears a time left by an earlier audit
        runtime_sheet.cell(runtime_row, 5).value = round(sum(stock), 1) if stock else None
        runtime_sheet.cell(runtime_row, 6).value = llm.get("lora_ratings")

        sheet.row_dimensions[row_number].height = 42
        for cell in sheet[row_number]:
            cell.alignment = Alignment(vertical="center", wrap_text=True)
        sheet.cell(row_number, 2).number_format = "yyyy-mm-dd hh:mm:ss"
        sheet.cell(row_number, 37).number_format = "General"
        sheet.freeze_panes = "A2"

        if sheet.tables:
            table_object = next(iter(sheet.tables.values()))
            table_object.ref = f"A1:AK{max(2, sheet.max_row)}"

        temporary_path = report_path.with_name(f".{report_path.name}.tmp.xlsx")
        workbook.save(temporary_path)
        workbook.close()
        temporary_path.replace(report_path)
    return report_path


def list_audited_call_ids(out_dir: Path) -> set[str]:
    """Return call IDs present in the Excel audit index."""
    report_path = out_dir / REPORT_WORKBOOK_NAME
    if load_workbook is None or not report_path.exists():
        return set()
    with _WRITE_LOCK:
        workbook = load_workbook(report_path, read_only=True, data_only=True)
        try:
            if INDEX_SHEET not in workbook.sheetnames:
                return set()
            return {
                _text(row[0])
                for row in workbook[INDEX_SHEET].iter_rows(min_row=2, values_only=True)
                if row and _text(row[0])
            }
        finally:
            workbook.close()


def read_audit_scores(out_dir: Path, call_id: str) -> Optional[Dict[str, Any]]:
    """Reconstruct the website's score-only report from the Excel workbook."""
    report_path = out_dir / REPORT_WORKBOOK_NAME
    if load_workbook is None or not report_path.exists():
        return None
    with _WRITE_LOCK:
        workbook = load_workbook(report_path, read_only=True, data_only=True)
        try:
            if INDEX_SHEET not in workbook.sheetnames:
                return None
            location = None
            for row in workbook[INDEX_SHEET].iter_rows(min_row=2, values_only=True):
                if row and _text(row[0]) == call_id and len(row) >= 3:
                    location = (_text(row[1]), int(row[2]))
                    break
            if not location or location[0] not in workbook.sheetnames:
                return None

            sheet_name, row_number = location
            table = next(
                (key for key, value in CALL_TYPE_SHEETS.items() if value == sheet_name),
                None,
            )
            if table is None:
                return None
            sheet = workbook[sheet_name]
            transcription_s = audit_s = stock_llm_s = finetuned_llm_s = None
            if RUNTIME_SHEET in workbook.sheetnames:
                for runtime_row in workbook[RUNTIME_SHEET].iter_rows(min_row=2, values_only=True):
                    if runtime_row and _text(runtime_row[0]) == call_id:
                        cells = [*runtime_row[1:6], None, None, None, None, None]
                        transcription_s, audit_s, _, stock_llm_s, finetuned_llm_s = cells[:5]
                        break
            status_text = _text(sheet.cell(row_number, 35).value).upper()
            fake_booking = "FAKE_BOOKING" in status_text
            wrong_call_tagging = "WRONG_CALL_TAGGING" in status_text
            audit: Dict[str, Dict[str, Any]] = {}
            earned = possible = 0.0
            first_score_column = 8
            for offset, pid in enumerate(PARAMETER_COLUMNS):
                value = sheet.cell(row_number, first_score_column + offset).value
                if value in (None, "", "N/A") or pid not in WEIGHTS.get(table, {}):
                    continue
                points = float(value)
                options = SCORE_OPTIONS[table][pid]
                rating = next(
                    (level for level, configured in options.items() if float(configured) == points),
                    "Zero",
                )
                audit[pid] = {"rating": rating, "evidence": ""}
                earned += points
                possible += float(WEIGHTS[table][pid])

            forced_zero = fake_booking or wrong_call_tagging
            if forced_zero:
                earned = 0.0
            overall = round(100 * earned / possible, 1) if possible else None
            from .scoring import grade_band
            score_display = (
                f"{int(earned) if earned.is_integer() else round(earned, 2)}/"
                f"{int(possible) if possible.is_integer() else round(possible, 2)}"
            ) if possible else "N/A"
            return {
                "weight_table": table,
                "parameter_weights": WEIGHTS[table],
                "parameter_score_options": SCORE_OPTIONS[table],
                "management_summary": {
                    "call_type": CALL_TYPE_LABELS[table],
                    "score": earned,
                    "score_display": score_display,
                    "score_pct": overall,
                    "grade": "Not Genuine" if forced_zero else grade_band(overall),
                    "risk": "High" if overall is not None and overall < 60 else (
                        "Medium" if overall is not None and overall < 75 else "Low"
                    ),
                    "confidence": "Stored in Excel",
                    "wrong_call_tagging": wrong_call_tagging,
                },
                "audit_result": {
                    "metadata": {
                        "call_type": table,
                        "cre_name": _text(sheet.cell(row_number, 4).value),
                    },
                    "audit": audit,
                    "llm_runtimes": {
                        "single_pass": audit_s,
                        "transcription": transcription_s,
                        "stock_llm": stock_llm_s,
                        "finetuned_llm": finetuned_llm_s,
                    },
                    "booking_genuine": (
                        {"genuine": False, "fake_booking_reason": "stored_excel_status"}
                        if fake_booking else None
                    ),
                },
            }
        finally:
            workbook.close()


def _evidence_path(out_dir: Path, call_id: str) -> Optional[Path]:
    name = Path(str(call_id)).name
    if not name or name != str(call_id) or name.startswith("."):
        return None
    return Path(out_dir) / EVIDENCE_DIR / f"{name}.json"


def save_call_evidence(out_dir: Path, call_id: str, evidence: Dict[str, Any]) -> None:
    """Replace one call's evidence file (written atomically)."""
    path = _evidence_path(out_dir, call_id)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(evidence, ensure_ascii=False, indent=1), encoding="utf-8")
    temporary.replace(path)


def read_call_evidence(out_dir: Path, call_id: str) -> Optional[Dict[str, Any]]:
    path = _evidence_path(out_dir, call_id)
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# ── Bulk reads for the dashboard's day list, CRE table and export ─────────────

_SCORES_CACHE: Dict[str, Any] = {"key": None, "scores": {}}


def _index_locations(workbook: Any) -> Dict[tuple, str]:
    """(sheet title, row) → call_id from the hidden index sheet."""
    locations: Dict[tuple, str] = {}
    if INDEX_SHEET in workbook.sheetnames:
        for row in workbook[INDEX_SHEET].iter_rows(min_row=2, values_only=True):
            if row and len(row) >= 3 and _text(row[0]) and row[2] is not None:
                locations[(_text(row[1]), int(row[2]))] = _text(row[0])
    return locations


def read_all_scores(out_dir: Path) -> Dict[str, Dict[str, Any]]:
    """call_id → score summary for every audited call, read in one pass.

    Same arithmetic as read_audit_scores. Cached until the workbook changes, so
    the dashboard can list a whole day without reopening Excel per call.
    """
    path = out_dir / REPORT_WORKBOOK_NAME
    if load_workbook is None or not path.exists():
        return {}
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if _SCORES_CACHE["key"] == key:
        return _SCORES_CACHE["scores"]

    from .scoring import grade_band

    scores: Dict[str, Dict[str, Any]] = {}
    with _WRITE_LOCK:
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            locations = _index_locations(workbook)
            for table, sheet_name in CALL_TYPE_SHEETS.items():
                if sheet_name not in workbook.sheetnames:
                    continue
                weights = WEIGHTS.get(table, {})
                rows = workbook[sheet_name].iter_rows(min_row=2, values_only=True)
                for row_number, values in enumerate(rows, start=2):
                    call_id = locations.get((sheet_name, row_number))
                    if not call_id:
                        continue
                    values = [*values, *([None] * 40)]
                    params: Dict[str, list] = {}
                    earned = possible = 0.0
                    for offset, pid in enumerate(PARAMETER_COLUMNS):
                        value = values[7 + offset]
                        if value in (None, "", "N/A") or pid not in weights:
                            continue
                        try:
                            points = float(value)
                        except (TypeError, ValueError):
                            continue
                        params[pid] = [points, float(weights[pid])]
                        earned += points
                        possible += float(weights[pid])
                    status = _text(values[34]).upper()
                    forced_zero = "FAKE_BOOKING" in status or "WRONG_CALL_TAGGING" in status
                    if forced_zero:
                        earned = 0.0
                    pct = round(100 * earned / possible, 1) if possible else None
                    audited_at = values[1]
                    scores[call_id] = {
                        "table": table,
                        "score_pct": pct,
                        "grade": "Not Genuine" if forced_zero else grade_band(pct),
                        "cre_name": _text(values[3]),
                        "cre_id": _text(values[4]),
                        "call_date": _text(values[5]),
                        "audited_at": audited_at.isoformat(sep=" ") if hasattr(audited_at, "isoformat")
                                      else _text(audited_at),
                        "params": params,
                    }
        finally:
            workbook.close()
    _SCORES_CACHE.update(key=key, scores=scores)
    return scores


def export_calls(out_dir: Path, call_ids: set) -> Optional[bytes]:
    """The audit workbook reduced to the given calls (one sheet per call type), as .xlsx bytes."""
    import io
    from copy import copy

    path = out_dir / REPORT_WORKBOOK_NAME
    if load_workbook is None or not path.exists():
        return None
    with _WRITE_LOCK:
        workbook = load_workbook(path)
    locations = _index_locations(workbook)
    keep: Dict[str, list] = {}
    for (sheet_name, row_number), call_id in locations.items():
        if call_id in call_ids:
            keep.setdefault(sheet_name, []).append(row_number)
    for sheet_name in CALL_TYPE_SHEETS.values():
        if sheet_name not in workbook.sheetnames:
            continue
        sheet = workbook[sheet_name]
        rows = [[(cell.value, copy(cell._style)) for cell in sheet[number]]
                for number in sorted(keep.get(sheet_name, []))]
        if sheet.max_row > 1:
            sheet.delete_rows(2, sheet.max_row - 1)
        for row_number, cells in enumerate(rows, start=2):
            for column, (value, style) in enumerate(cells, start=1):
                target = sheet.cell(row_number, column, value)
                target._style = style
            sheet.row_dimensions[row_number].height = 42
        if sheet.tables:
            next(iter(sheet.tables.values())).ref = f"A1:AK{max(2, len(rows) + 1)}"
    for hidden in (INDEX_SHEET, RUNTIME_SHEET):
        if hidden in workbook.sheetnames:
            workbook.remove(workbook[hidden])
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()
