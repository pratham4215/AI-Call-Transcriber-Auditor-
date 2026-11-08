"""
importer.py — bring CRM calls into data/ for transcription and audit.

For each call (lead_transaction id):
  1. call data from the CRM database (crm_db.get_call); vehicle master from the
     CRM API's leads-ai endpoint (the source of the hand-made data/ JSON files),
     else from the database
  2. recording: CRM decrypt API by bizRTC CDR id -> playable link -> audio
  3. data/audios/<CALL_TYPE>/<REG>_<CDR_ID>.mp3, and beside it <same>.json:
     the vehicle master in the CRM API shape the data/ JSON files use, plus
     "call_data" (this call's CALL DATA.xlsx-shaped record) and "source" ids,
     so a later audit needs neither the database nor the API.

    python -m core.audit.importer --audit                    yesterday's calls, then transcribe + audit
    python -m core.audit.importer --date 2026-10-12 --call-type BOOKING --limit 5 --audit
    python -m core.audit.importer --date today --dry-run     list what would be picked
    python -m core.audit.importer                            import only (no audit)
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from config import DATA_DIR, IMPORT_WORKERS
from core import crm_api
from core.recordings import RecordingError, fetch_recording

from . import crm_db
from .crm_call_data import normalize_reg


def parse_day(value: Any) -> date:
    """'today', 'yesterday' or YYYY-MM-DD."""
    text = str(value).strip().lower()
    if text in ("today", "yesterday"):
        return date.today() - timedelta(days=1 if text == "yesterday" else 0)
    return date.fromisoformat(text)


def _stem(registration: Any, cdr_id: Any) -> str:
    reg = re.sub(r"[^A-Z0-9]", "", normalize_reg(registration))
    return f"{reg or 'NOREG'}_{cdr_id}"


def stem_for(call: crm_db.CallData) -> str:
    """<REG>_<CDR_ID>: one file per call, parsed back by crm_db.parse_stem."""
    return _stem(call.crm["vehicle"].get("registration_number"), call.cdr_id)


def already_imported(summary: Dict[str, Any]) -> bool:
    """True when a crm_db.list_calls summary's recording is already in data/."""
    folder = DATA_DIR / str(summary.get("call_type") or "")
    stem = _stem(summary.get("registration_number"), summary.get("cdr_id"))
    return folder.exists() and any(p.suffix.lower() != ".json" and not p.name.endswith(".part")
                                   for p in folder.glob(f"{stem}.*"))


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _vehicle_master(call: crm_db.CallData) -> Tuple[Dict[str, Any], str]:
    """(vehicle master in the CRM API shape, where it came from).

    The CRM's own AI endpoint (leads-ai) is what the hand-made data/ JSON files
    were fetched from: it carries extended warranty, fuel type, variant and the
    DMS service history that the database lacks for many vehicles. The
    database's lead + billing data is the fallback (e.g. the API token expired).
    """
    reg = normalize_reg(call.crm["vehicle"].get("registration_number"))
    try:
        payload = crm_api.get_lead(reg)
        if isinstance(payload, dict) and isinstance(payload.get("data"), dict) and payload["data"]:
            return payload, "crm_api.leads-ai"
    except Exception:
        pass
    return call.vehicle or {"code": 200, "message": "crmdb", "data": {"lead": {}}}, "crmdb"


def _sidecar(call: crm_db.CallData) -> Dict[str, Any]:
    vehicle, vehicle_source = _vehicle_master(call)
    payload = json.loads(json.dumps(vehicle, default=str))
    payload["call_data"] = call.crm
    payload["source"] = {
        "transaction_id": call.transaction_id,
        "cdr_id": call.cdr_id,
        "vehicle_master": vehicle_source,
        "imported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return payload


def _timed_sidecar(call: crm_db.CallData) -> Tuple[Dict[str, Any], float]:
    started = time.perf_counter()
    return _sidecar(call), time.perf_counter() - started


# Vehicle lookups run beside the recording fetch; long-lived threads keep their CRM API connections.
_VEHICLE_POOL = ThreadPoolExecutor(max_workers=IMPORT_WORKERS, thread_name_prefix="vehicle-api")


def import_call(transaction_id: str, overwrite: bool = False) -> Dict[str, Any]:
    """Import one call. Returns a result row (no customer details) with status
    imported / skipped (already there) / error."""
    started = time.perf_counter()
    result: Dict[str, Any] = {"transaction_id": transaction_id, "stem": None, "call_type": None}
    timing: Dict[str, float] = {}  # seconds per step: db, decrypt, download, vehicle_api
    try:
        step = time.perf_counter()
        call = crm_db.get_call(transaction_id)
        timing["db"] = time.perf_counter() - step
        if call is None:
            raise LookupError("call not found")
        call_type = call.crm.get("audit_call_type")
        result.update(call_type=call_type, cdr_id=call.cdr_id)
        if call_type not in crm_db.CALL_TYPES:
            raise LookupError("not one of the audited call types")
        if not call.cdr_id:
            raise LookupError("no bizRTC CDR id, so no recording")

        stem = stem_for(call)
        folder = DATA_DIR / call_type
        result["stem"] = stem
        existing = next((p for p in folder.glob(f"{stem}.*") if p.suffix.lower() != ".json"
                         and not p.name.endswith(".part")), None) if folder.exists() else None
        if existing and not overwrite:
            result["status"] = "skipped"
            return result

        vehicle = _VEHICLE_POOL.submit(_timed_sidecar, call)  # leads-ai while the recording downloads
        audio, ext = fetch_recording(call.cdr_id, timing)
        sidecar, timing["vehicle_api"] = vehicle.result()
        folder.mkdir(parents=True, exist_ok=True)
        _write_atomic(folder / f"{stem}.{ext}", audio)
        _write_atomic(folder / f"{stem}.json",
                      json.dumps(sidecar, ensure_ascii=False, indent=1, default=str).encode("utf-8"))
        result.update(status="imported", bytes=len(audio), vehicle_master=sidecar["source"]["vehicle_master"])
    except (LookupError, ValueError, RecordingError) as exc:
        result.update(status="error", error=str(exc))
    except Exception as exc:  # DB / network: keep going with the next call
        from core import db
        result.update(status="error", error=db.describe_error(exc))
    finally:
        result["seconds"] = round(time.perf_counter() - started, 1)
        result["timing"] = {name: round(seconds, 1) for name, seconds in timing.items()}
    return result


def import_calls(transaction_ids: Iterable[str], overwrite: bool = False, on_result=None,
                 workers: int = IMPORT_WORKERS) -> List[Dict[str, Any]]:
    """Import calls, `workers` at a time (each call is network-bound: DB, CRM API, bizRTC).

    on_result(results so far) is called after every call; with workers > 1 the
    results arrive in completion order.
    """
    ids = list(transaction_ids)
    results: List[Dict[str, Any]] = []
    lock = threading.Lock()

    def finished(row: Dict[str, Any]) -> None:
        with lock:
            results.append(row)
            if on_result:
                on_result(list(results))

    if workers <= 1 or len(ids) <= 1:
        for transaction_id in ids:
            finished(import_call(transaction_id, overwrite))
        return results
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(import_call, tid, overwrite) for tid in ids]):
            finished(future.result())
    return results


def timing_summary(results: List[Dict[str, Any]]) -> str:
    """Average seconds per step over the imported calls, e.g. 'db 1.1s | decrypt 6.0s | ...'."""
    rows = [r for r in results if r.get("status") == "imported" and r.get("timing")]
    if not rows:
        return ""
    steps = ("db", "decrypt", "download", "vehicle_api")
    avg = {s: sum(r["timing"].get(s, 0) for r in rows) / len(rows) for s in steps}
    mb = sum(r.get("bytes", 0) for r in rows) / len(rows) / 1e6
    text = " | ".join(f"{s} {avg[s]:.1f}s" for s in steps) + f" | {mb:.1f} MB per recording"
    if avg["download"]:
        text += f" ({mb * 8 / avg['download']:.1f} Mbit/s)"
    return text


def find_calls(
    *, day_from: date, day_to: date, call_type: Optional[str] = None, cre: Optional[str] = None,
    station: Optional[str] = None, limit: int = 10, skip_existing: bool = False, max_pages: int = 30,
    connected: Optional[bool] = True, lead_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Audited-type calls with a recording, newest first (crm_db.list_calls summaries).

    skip_existing leaves out calls already in data/, so fetching again for the
    same day brings in new calls (looks through at most max_pages x 100 calls).
    """
    calls, offset = [], 0
    for _ in range(max_pages):
        page, has_more = crm_db.list_calls(
            call_type=call_type, cre=cre, station=station, date_from=day_from, date_to=day_to,
            has_recording=True, connected=connected, lead_type=lead_type, limit=100, offset=offset,
        )
        calls += [call for call in page
                  if call.get("cdr_id") and not (skip_existing and already_imported(call))]
        if len(calls) >= limit or not has_more:
            break
        offset += len(page)
    return calls[:limit]


def _run_in_app(selection: Dict[str, Any]) -> None:
    """Ask the running app to pick, import, transcribe and audit; follow its progress."""
    import requests

    from config import FLASK_PORT

    base = f"http://127.0.0.1:{FLASK_PORT}"
    try:
        resp = requests.post(f"{base}/api/db/run", json=selection, timeout=30)
    except requests.ConnectionError:
        raise SystemExit(f"The app is not running on port {FLASK_PORT}: start it with  python wsgi.py")
    if resp.status_code != 202:
        raise SystemExit(f"error: {resp.json().get('error', resp.status_code)}")
    job_id, last = resp.json()["job_id"], None
    while True:
        job = requests.get(f"{base}/api/job/{job_id}", timeout=30).json()
        if job.get("progress") != last:
            last = job.get("progress")
            print(last, flush=True)
        if job.get("status") == "error":
            raise SystemExit(f"error: {job.get('error')}")
        if job.get("status") == "done":
            for row in (job.get("result") or {}).get("errors", []):
                print(f"  failed: {row.get('stem') or row['transaction_id']}: {row.get('error')}")
            if job.get("batch_id"):
                print(f"Transcription and audit running: {base}/batch")
            return
        time.sleep(3)


if __name__ == "__main__":
    import argparse
    import sys

    sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(
        description="Pick calls from the CRM database and import them (recording + call data) into data/; "
                    "with --audit the running app also transcribes and audits them.")
    parser.add_argument("--date", type=parse_day, help="call date: today, yesterday (default) or YYYY-MM-DD")
    parser.add_argument("--from", dest="day_from", type=parse_day, help="first call date (with --to)")
    parser.add_argument("--to", dest="day_to", type=parse_day, help="last call date")
    parser.add_argument("--call-type", choices=crm_db.CALL_TYPES)
    parser.add_argument("--cre", help="CRE employee code or name")
    parser.add_argument("--station", help="workshop name or code")
    parser.add_argument("--lead-type", choices=crm_db.LEAD_TYPES, type=str.upper, help="UNNATI or PRAGATI")
    parser.add_argument("--limit", type=int, default=10, help="how many calls (newest first, default 10)")
    parser.add_argument("--audit", action="store_true", help="also transcribe and audit (needs the app running)")
    parser.add_argument("--dry-run", action="store_true", help="only list the calls that would be imported")
    parser.add_argument("--overwrite", action="store_true", help="re-download calls already in data/")
    parser.add_argument("--reaudit", action="store_true", help="with --audit: audit again calls that have a report")
    parser.add_argument("--workers", type=int, default=IMPORT_WORKERS,
                        help=f"calls fetched at the same time (default {IMPORT_WORKERS}, IMPORT_WORKERS in .env)")
    parser.add_argument("--transaction", action="append", default=[], metavar="ID",
                        help="import this lead_transaction id instead of picking by date (repeatable)")
    args = parser.parse_args()

    day_from = args.day_from or args.date or parse_day("yesterday")
    day_to = args.day_to or args.date or day_from
    if args.audit and not args.dry_run:
        _run_in_app({"date_from": day_from.isoformat(), "date_to": day_to.isoformat(),
                     "call_type": args.call_type, "cre": args.cre, "station": args.station,
                     "lead_type": args.lead_type, "limit": args.limit,
                     "overwrite": args.overwrite, "reaudit": args.reaudit})
        raise SystemExit

    ids = list(args.transaction)
    if not ids:
        picking = time.perf_counter()
        calls = find_calls(day_from=day_from, day_to=day_to, call_type=args.call_type, cre=args.cre,
                           station=args.station, lead_type=args.lead_type, limit=args.limit,
                           skip_existing=not args.overwrite)
        print(f"{len(calls)} calls with a recording, {day_from} to {day_to} "
              f"(picked in {time.perf_counter() - picking:.1f}s; calls already in data/ left out)")
        for call in calls:
            print(f"  {call['call_date']}  {call['call_type']:<19} {call['lead_type'] or '-':<7} "
              f"{call['registration_number'] or '-':<11} "
                  f"cre={call['cre']['code']} cdr={call['cdr_id']}")
        if args.dry_run:
            raise SystemExit
        ids = [call["transaction_id"] for call in calls]

    def report(results: List[Dict[str, Any]]) -> None:
        row = results[-1]
        steps = " | ".join(f"{k} {v}s" for k, v in (row.get("timing") or {}).items())
        print(f"[{len(results)}/{len(ids)}] {row.get('status'):<8} {row.get('stem') or row['transaction_id']} "
              f"{row.get('call_type') or ''} {row.get('error', '')} ({row['seconds']}s: {steps})", flush=True)

    importing = time.perf_counter()
    results = import_calls(ids, args.overwrite, on_result=report, workers=args.workers)
    elapsed = time.perf_counter() - importing
    counts = {status: sum(r["status"] == status for r in results) for status in ("imported", "skipped", "error")}
    print(f"done: {counts} in {elapsed:.1f}s with {args.workers} worker(s)"
          + (f" = {elapsed / counts['imported']:.1f}s per imported call" if counts["imported"] else ""))
    summary = timing_summary(results)
    if summary:
        print(f"average per call: {summary}")
