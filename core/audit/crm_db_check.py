"""
crm_db_check.py — check crm_db against CALL DATA.xlsx and the data/ JSON files.

    python -m core.audit.crm_db_check
    python -m core.audit.crm_db_check remarks [yesterday|YYYY-MM-DD]

The second form counts, for one day's audited-type calls, how often the call's
own remark (what the audit reads) is empty while its disposition form has one.

Read-only. Prints match counts per field, query timings, and example values
only for non-personal fields (dates, statuses, codes, workshops); customer
names, phones, addresses, remarks, VIN/engine/chassis and CRE names are only
counted, never printed.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Tuple

from config import DATA_DIR
from core import db
from core.audit import crm_call_data, crm_db

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac"}

# Example values are printed only for these call-data fields.
_SHOWN_PREFIXES = (
    "audit_call_type", "call_meta.", "disposition.", "booking.", "follow_up.",
    "cre.username", "cre.employee_id", "vehicle.registration_number", "vehicle.model",
    "vehicle.service_due_date", "vehicle.next_service_type", "vehicle.nsd_",
    "service_profile.campaign", "service_profile.workshop", "service_profile.pickup_drop_status",
    "service_profile.dms_verified", "service_profile.snr_", "service_history",
)
# Present only in the DB version, by design.
_DB_ONLY = {"call_meta.transaction_id", "call_meta.cdr_id", "disposition.escalation_reason"}


def _flatten(value: Any, prefix: str = "") -> Iterator[Tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _flatten(item, f"{prefix}{key}.")
    elif isinstance(value, list):
        yield prefix.rstrip("."), json.dumps(value, default=str, sort_keys=True)
    else:
        yield prefix.rstrip("."), value


def _timed(label: str, fn: Callable[[], Any]) -> Any:
    started = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:
        print(f"  FAILED {label}: {db.describe_error(exc)}")
        return None
    print(f"  {label}: {time.perf_counter() - started:.1f}s")
    return result


def _section(title: str) -> None:
    print(f"\n== {title} ==")


def check_connection() -> bool:
    _section("connection")
    info = db.health()
    print(f"  connected={info.get('connected')} read_only={info.get('read_only')} "
          f"database={info.get('database')} {info.get('error', '')}")
    return bool(info.get("connected"))


def check_recordings() -> Dict[str, crm_db.CallData]:
    stems = sorted({p.stem for p in DATA_DIR.rglob("*") if p.suffix.lower() in AUDIO_EXTS})
    _section(f"local recordings ({len(stems)}) -> calls in the DB")
    found = _timed("lookup with vehicle master", lambda: crm_db.get_call_crm_many(stems)) or {}
    db_calls = {stem: call for stem, call in found.items() if call.source == crm_db.SOURCE_PATH}
    print(f"  found in DB: {len(db_calls)}/{len(stems)}  "
          f"how picked: {dict(Counter(call.picked for call in db_calls.values()))}")
    missing = [stem for stem in stems if stem not in db_calls]
    if missing:
        print(f"  not found in DB: {missing[:20]}{' ...' if len(missing) > 20 else ''}")
    _timed("same lookup again (cache)", lambda: crm_db.get_call_crm_many(stems))
    return db_calls


def check_call_data(db_calls: Dict[str, crm_db.CallData]) -> None:
    _section("call data: DB vs CALL DATA.xlsx (same recording)")
    matches: Counter = Counter()
    totals: Counter = Counter()
    examples: Dict[str, list] = defaultdict(list)
    compared = 0
    for stem, call in sorted(db_calls.items()):
        excel = crm_call_data.peek(stem)
        if excel is None:
            continue
        compared += 1
        ours = dict(_flatten(call.crm))
        theirs = dict(_flatten(excel))
        for field in sorted((ours.keys() | theirs.keys()) - _DB_ONLY):
            totals[field] += 1
            a, b = ours.get(field), theirs.get(field)
            if a == b or (a in (None, "") and b in (None, "")):
                matches[field] += 1
            elif len(examples[field]) < 3 and field.startswith(_SHOWN_PREFIXES):
                examples[field].append(f"{stem}: db={a!r} excel={b!r}")
    print(f"  compared {compared} recordings that are in both")
    for field in sorted(totals, key=lambda f: (matches[f] / totals[f], f)):
        line = f"  {matches[field]:>3}/{totals[field]:<3} {field}"
        if matches[field] < totals[field] and not field.startswith(_SHOWN_PREFIXES):
            line += "   (values not shown)"
        print(line)
        for example in examples[field]:
            print(f"          {example}")


def check_vehicle_master(db_calls: Dict[str, crm_db.CallData]) -> None:
    from core.audit.pipeline import normalize_crm_details

    _section("vehicle master: DB lead/lead_drop + kst_billing_data vs data/ JSON files")
    with_vehicle = [call for call in db_calls.values() if call.vehicle]
    print(f"  calls with vehicle master: {len(with_vehicle)}/{len(db_calls)}  "
          f"from: {dict(Counter(call.vehicle['message'] for call in with_vehicle))}")
    history = [len(call.vehicle["data"]["serviceHistory"]) for call in with_vehicle]
    if history:
        print(f"  service history entries per vehicle: min {min(history)} max {max(history)} "
              f"avg {sum(history) / len(history):.1f}; with none: {history.count(0)}")
    fields = ("vehicle.variant", "vehicle.sale_date", "vehicle.next_service_type", "vehicle.service_due_date",
              "vehicle.odometer_km", "service_profile.workshop_code", "service_profile.oem_status",
              "service_profile.mcp_status", "service_profile.ew_status_or_expiry", "vehicle.fuel_type")
    filled: Counter = Counter()
    same: Counter = Counter()
    both: Counter = Counter()
    examples: Dict[str, list] = defaultdict(list)
    for stem, call in db_calls.items():
        if not call.vehicle:
            continue
        ours = dict(_flatten(normalize_crm_details(call.vehicle)))
        sidecar = next(DATA_DIR.rglob(f"{stem}.json"), None)
        theirs = dict(_flatten(normalize_crm_details(json.loads(sidecar.read_text(encoding="utf-8"))))) if sidecar else {}
        for field in fields:
            if ours.get(field) not in (None, ""):
                filled[field] += 1
            if theirs.get(field) not in (None, ""):
                both[field] += 1
                if ours.get(field) == theirs.get(field):
                    same[field] += 1
                elif len(examples[field]) < 2 and field != "vehicle.odometer_km":
                    examples[field].append(f"{stem}: db={ours.get(field)!r} json={theirs.get(field)!r}")
    for field in fields:
        print(f"  {field:<38} filled in DB {filled[field]:>3}/{len(with_vehicle):<3} "
              f"same as JSON {same[field]:>3}/{both[field]}")
        for example in examples[field]:
            print(f"          {example}")


def check_api_queries(db_calls: Dict[str, crm_db.CallData]) -> None:
    _section("API queries")
    page = _timed("list_calls newest 20", lambda: crm_db.list_calls(limit=20))
    if page:
        calls, has_more = page
        print(f"  {len(calls)} calls, has_more={has_more}; call types: {dict(Counter(c['call_type'] for c in calls))}")
        for call in calls[:3]:
            print(f"    {call['call_date']}  {call['call_type']:<19} cre={call['cre']['code']} "
                  f"disp={call['disposition']} recording={call['has_recording']} cdr_id={call['cdr_id']}")
    for call_type in crm_db.CALL_TYPES:
        page = _timed(f"list_calls call_type={call_type} with recording, 10",
                      lambda ct=call_type: crm_db.list_calls(call_type=ct, has_recording=True, limit=10))
        if page:
            print(f"    {len(page[0])} found; types {sorted({c['call_type'] for c in page[0]})}")
    sample = next(iter(db_calls.values()), None)
    if sample:
        call = _timed("get_call by transaction id", lambda: crm_db.get_call(sample.transaction_id))
        if call:
            print(f"    same call: {call.transaction_id == sample.transaction_id}, "
                  f"vehicle master: {call.vehicle is not None}, cdr_id: {call.cdr_id}")
        reg = sample.crm["vehicle"]["registration_number"]
        history = _timed(f"vehicle_calls {reg}", lambda: crm_db.vehicle_calls(reg))
        if history is not None:
            print(f"    {len(history)} calls; types {dict(Counter(c['call_type'] for c in history))}")
        merged = _timed("crm_details (merged, as the audit sees it)", lambda: crm_db.crm_details(sample))
        if merged:
            print(f"    sections: {sorted(merged)}")


# Remarks typed into the disposition form itself (booking / SNR / follow-up / other).
_FORM_CRE = ("coalesce(nullif(btrim(bs.cre_remarks), ''), nullif(btrim(snr.cre_remarks), ''), "
             "nullif(btrim(cl.cre_feedback), ''), nullif(btrim(od.cre_feedback), ''))")
_FORM_CUSTOMER = ("coalesce(nullif(btrim(bs.customer_remarks), ''), nullif(btrim(snr.customer_remarks), ''), "
                  "nullif(btrim(cl.customer_remarks), ''), nullif(btrim(od.customer_remarks), ''))")


def check_remarks(day) -> None:
    """Call remark vs disposition-form remark, counted per call type for one day."""
    from datetime import timedelta

    _section(f"remarks on {day}: the call's own remark (what the audit reads) vs its disposition form")
    sql_from = crm_db._CALL_SELECT[crm_db._CALL_SELECT.index("FROM lead_transaction"):]
    parts = []
    for name, call_col, form_col in (("cre", "nullif(btrim(lt.cre_remark), '')", _FORM_CRE),
                                     ("customer", "nullif(btrim(lt.customer_remark), '')", _FORM_CUSTOMER)):
        parts += [
            f"count(*) FILTER (WHERE {call_col} IS NULL) AS {name}_empty",
            f"count(*) FILTER (WHERE {call_col} IS NULL AND {form_col} IS NOT NULL) AS {name}_only_in_form",
            f"count(*) FILTER (WHERE {call_col} IS NULL AND {form_col} IS NOT NULL "
            f"AND {crm_db._BOOKED_THIS_CALL}) AS {name}_same_call",
            f"count(*) FILTER (WHERE {call_col} IS NOT NULL AND {form_col} IS NOT NULL "
            f"AND {call_col} <> {form_col}) AS {name}_differs",
        ]
    rows = _timed("count", lambda: db.fetch_all(
        f"SELECT {crm_db._CALL_TYPE_SQL} AS call_type, count(*) AS calls, {', '.join(parts)}\n{sql_from}\n"
        f"WHERE {crm_db._CALL_FILTER} AND {crm_db._CALL_TYPE_SQL} IS NOT NULL "
        "AND lt.call_date >= :start AND lt.call_date < :end GROUP BY 1 ORDER BY 1",
        {"start": crm_db._day_start(day), "end": crm_db._day_start(day + timedelta(days=1))},
    )) or []
    print(f"  {'call type':<20}{'calls':>7} | {'CRE remark: empty':>18}{'only in form':>14}{'same call':>11}"
          f"{'differs':>9} | {'customer: empty':>16}{'only in form':>14}{'same call':>11}{'differs':>9}")
    for row in rows:
        print(f"  {row['call_type']:<20}{row['calls']:>7} | {row['cre_empty']:>18}{row['cre_only_in_form']:>14}"
              f"{row['cre_same_call']:>11}{row['cre_differs']:>9} | {row['customer_empty']:>16}"
              f"{row['customer_only_in_form']:>14}{row['customer_same_call']:>11}{row['customer_differs']:>9}")
    print("  'only in form': the call record's remark is empty but its disposition form has one.")
    print("  'same call'   : of those, booking forms created on this call; the import now uses their remark.")


def main() -> None:
    # Windows consoles (cp1252) cannot print every workshop/model name; never crash on one.
    sys.stdout.reconfigure(errors="replace")
    if len(sys.argv) > 1 and sys.argv[1] == "remarks":
        from core.audit.importer import parse_day

        if check_connection():
            check_remarks(parse_day(sys.argv[2] if len(sys.argv) > 2 else "yesterday"))
        return
    if not crm_db.enabled():
        print("CRM database not configured (DB_* in .env) or CRM_SOURCE=excel")
        return
    if not check_connection():
        return
    db_calls = check_recordings()
    if db_calls:
        check_call_data(db_calls)
        check_vehicle_master(db_calls)
    check_api_queries(db_calls)


if __name__ == "__main__":
    main()
