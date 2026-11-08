"""
crm_api.py — the Kataria CRM's API for the AI audit (CRM_API_URL, smr-api.katariacrm.com/crm).

Authenticated with CRM_API_TOKEN (the ktr-ai-... key) as "Authorization: Bearer"
(the API's "bearerAuth" scheme).

  GET  /api/v1/leads-ai?registrationNumber=...        vehicle, customer, service history
                                                       and call history for one vehicle
  GET  /api/v1/leads-ai/audit-report?startDate&endDate calls in a date range
  POST /api/v1/cdr/recording/decrypt?cdrId=...         playable link for a call recording

Check the endpoints against the local data/ JSON files and CALL DATA.xlsx
(prints status and field names only, never values):
    python -m core.crm_api check GJ01AB1234
"""
from __future__ import annotations

import json
import threading
from typing import Any, Dict, Optional, Set

import requests
from requests.adapters import HTTPAdapter

from config import CRM_API_TIMEOUT, CRM_API_TOKEN, CRM_API_URL

LEADS_AI = "/api/v1/leads-ai"
AUDIT_REPORT = "/api/v1/leads-ai/audit-report"
DECRYPT = "/api/v1/cdr/recording/decrypt"

_local = threading.local()


def session() -> requests.Session:
    """This thread's keep-alive session: no new TLS handshake per request to the CRM / bizRTC."""
    current = getattr(_local, "session", None)
    if current is None:
        current = requests.Session()
        adapter = HTTPAdapter(pool_connections=4, pool_maxsize=8)
        current.mount("https://", adapter)
        current.mount("http://", adapter)
        _local.session = current
    return current


class CrmApiError(RuntimeError):
    pass


def request(method: str, path: str, params: Optional[Dict[str, Any]] = None) -> requests.Response:
    if not CRM_API_TOKEN:
        raise CrmApiError("CRM_API_TOKEN is not set in .env")
    token = CRM_API_TOKEN
    if token.lower().startswith("bearer "):  # pasted with its scheme
        token = token[7:].strip()
    # requests drops the Authorization header if the API redirects to another host.
    return session().request(
        method, CRM_API_URL.rstrip("/") + path, params=params,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        timeout=CRM_API_TIMEOUT,
    )


def message(resp: requests.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        return (resp.reason or "")[:120]
    if isinstance(body, dict):
        for key in ("message", "error", "detail"):
            if isinstance(body.get(key), (str, int)):
                return str(body[key])[:120]
    return (resp.reason or "")[:120]


def _data(resp: requests.Response) -> Any:
    if resp.status_code != 200:
        raise CrmApiError(f"CRM API {resp.request.method} {resp.url.split('?')[0]} returned "
                          f"HTTP {resp.status_code}: {message(resp)}")
    body = resp.json()
    return body.get("data") if isinstance(body, dict) else body


def get_lead(registration: str) -> Any:
    """The CRM's AI payload for one vehicle (the response body, as in the data/ JSON files)."""
    resp = request("GET", LEADS_AI, {"registrationNumber": registration.strip().upper()})
    _data(resp)  # raises on errors
    return resp.json()


def audit_report(start_date: str, end_date: str) -> Any:
    """Calls between two dates (YYYY-MM-DD, inclusive)."""
    return _data(request("GET", AUDIT_REPORT, {"startDate": start_date, "endDate": end_date}))


# ── Diagnostics (field names and types only) ──────────────────────────────────

def shape(value: Any, depth: int = 0) -> Any:
    """JSON structure with values replaced by their type (links flagged, not shown)."""
    if depth > 5:
        return "..."
    if isinstance(value, dict):
        return {key: shape(item, depth + 1) for key, item in list(value.items())[:60]}
    if isinstance(value, list):
        return [shape(value[0], depth + 1), f"({len(value)} items)"] if value else []
    if isinstance(value, str):
        if value.startswith(("http://", "https://")):
            last = value.split("?")[0].rsplit("/", 1)[-1]
            return f"<link .{last.rsplit('.', 1)[-1][:8] if '.' in last else '?'}>"
        return f"<str {len(value)}>"
    return type(value).__name__


def _key_paths(value: Any, prefix: str = "") -> Set[str]:
    paths: Set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            paths.add(prefix + key)
            paths |= _key_paths(item, prefix + key + ".")
    elif isinstance(value, list):
        for item in value[:50]:
            paths |= _key_paths(item, prefix.rstrip(".") + "[].")
    return paths


def _first_records(value: Any) -> Optional[list]:
    """The first list of objects inside a response (the report's rows)."""
    if isinstance(value, list):
        return value if value and isinstance(value[0], dict) else None
    if isinstance(value, dict):
        for item in value.values():
            found = _first_records(item)
            if found:
                return found
    return None


def _check(registration: str) -> None:
    from pathlib import Path

    from config import DATA_DIR
    from core.audit import crm_call_data

    print(f"== GET {LEADS_AI}?registrationNumber={registration}")
    resp = request("GET", LEADS_AI, {"registrationNumber": registration})
    print(f"   HTTP {resp.status_code} {message(resp) if resp.status_code != 200 else ''}")
    if resp.status_code == 200:
        body = resp.json()
        sidecar = next(Path(DATA_DIR).rglob(f"{registration}.json"), None)
        if sidecar:
            ours = _key_paths(body)
            local = _key_paths(json.loads(sidecar.read_text(encoding="utf-8")))
            print(f"   field names vs local {sidecar.name}: {len(ours & local)} shared, "
                  f"{len(ours - local)} only in API, {len(local - ours)} only in local file")
            for label, paths in (("only in API", ours - local), ("only in local file", local - ours)):
                if paths:
                    print(f"   {label}: {sorted(paths)[:25]}")
        print("   shape:", json.dumps(shape(body))[:2500])

    excel = crm_call_data.peek(registration)
    day = str((excel or {}).get("call_meta", {}).get("call_date") or "")[:10] or "2026-07-06"
    print(f"\n== GET {AUDIT_REPORT}?startDate={day}&endDate={day}")
    resp = request("GET", AUDIT_REPORT, {"startDate": day, "endDate": day})
    print(f"   HTTP {resp.status_code} {message(resp) if resp.status_code != 200 else ''}")
    if resp.status_code == 200:
        body = resp.json()
        rows = _first_records(body.get("data") if isinstance(body, dict) else body)
        if rows:
            fields = sorted({key for row in rows for key in row})
            headers = _excel_headers()
            norm = lambda s: str(s).replace("_", "").lower()
            same = {f for f in fields if norm(f) in {norm(h) for h in headers}}
            print(f"   {len(rows)} rows; {len(fields)} fields, {len(same)} match CALL DATA.xlsx columns")
            print(f"   fields not in the Excel: {sorted(set(fields) - same)}")
            print(f"   Excel columns not in the API: {sorted(h for h in headers if norm(h) not in {norm(f) for f in fields})}")
        print("   shape:", json.dumps(shape(body))[:2500])

    cdr_id = "12345678"
    print(f"\n== POST {DECRYPT}?cdrId={cdr_id}")
    resp = request("POST", DECRYPT, {"cdrId": cdr_id})
    print(f"   HTTP {resp.status_code} {message(resp) if resp.status_code != 200 else ''}")
    if resp.status_code == 200:
        try:
            print("   shape:", json.dumps(shape(resp.json()))[:1000])
        except ValueError:
            print(f"   content-type: {resp.headers.get('Content-Type')}, {len(resp.content):,} bytes")


def _excel_headers() -> Set[str]:
    import openpyxl

    from core.audit.crm_call_data import CALL_DATA_EXCEL_PATH
    if not CALL_DATA_EXCEL_PATH.exists():
        return set()
    wb = openpyxl.load_workbook(str(CALL_DATA_EXCEL_PATH), read_only=True, data_only=True)
    try:
        return {str(h).strip() for h in next(wb.worksheets[0].iter_rows(values_only=True)) if h is not None}
    finally:
        wb.close()


if __name__ == "__main__":
    import argparse
    import sys

    sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Check the CRM AI API (field names only, no values).")
    parser.add_argument("command", choices=["check"])
    parser.add_argument("registration", help="a registration number with a local recording, e.g. GJ01AB1234")
    args = parser.parse_args()
    _check(args.registration.strip().upper())
