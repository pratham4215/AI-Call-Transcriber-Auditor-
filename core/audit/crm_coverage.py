"""
crm_coverage.py — which CRM facts the audit actually gets, per recording.

    python -m core.audit.crm_coverage            imported calls (REG_CDR stems) only
    python -m core.audit.crm_coverage --all      also the hand-made data/ set (slower:
                                                 those look calls up in the database)

Loads crm_details exactly as the audit does (pipeline.load_crm_details with the
recording's JSON file) and counts, for every field the LLM prompt
(build_crm_anchors) and the scoring rules read, how many calls have it filled.
Prints counts only, never values.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import DATA_DIR, TRANSCRIPTS_DIR

from .crm_db import parse_stem
from .pipeline import load_crm_details

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac"}

# (label, dotted path, only for this call type or None) — what build_crm_anchors and rules.py read
FIELDS: List[Tuple[str, str, Optional[str]]] = [
    ("CRE remark", "remarks.cre_remark", None),
    ("Customer remark", "remarks.customer_remark", None),
    ("Disposition status", "disposition.status", None),
    ("Disposition form", "disposition.type", None),
    ("Audit call type", "audit_call_type", None),
    ("CRE name", "cre.name", None),
    ("CRE code", "cre.username", None),
    ("Call date", "call_meta.call_date", None),
    ("Call duration", "call_meta.call_duration", None),
    ("Registration", "vehicle.registration_number", None),
    ("Model", "vehicle.model", None),
    ("Next service type", "vehicle.next_service_type", None),
    ("Next service due", "vehicle.service_due_date", None),
    ("Odometer", "vehicle.odometer_km", None),
    ("OEM warranty", "service_profile.oem_status", None),
    ("Extended warranty", "service_profile.ew_status_or_expiry", None),
    ("MCP", "service_profile.mcp_status", None),
    ("Last service", "service_profile.last_authorized_service", None),
    ("Booking date/time", "booking.booking_date_time", "BOOKING"),
    ("Booking workshop", "booking.workshop", "BOOKING"),
    ("Follow-up date", "follow_up.date", "FOLLOW_UPS"),
    ("Follow-up reason", "follow_up.reason", "FOLLOW_UPS"),
    ("SNR reason", "disposition.snr_main_reason", "SERVICE_NT_REQUIRED"),
]


def _value(details: Dict[str, Any], path: str) -> Any:
    value: Any = details
    for key in path.split("."):
        value = value.get(key) if isinstance(value, dict) else None
    return value


def _filled(value: Any) -> bool:
    return value not in (None, "", [], {}) and str(value).strip().lower() not in ("none", "unknown", "n/a", "—")


def main() -> None:
    sys.stdout.reconfigure(errors="replace")
    include_all = "--all" in sys.argv
    audios = sorted(p for p in DATA_DIR.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
    groups: Dict[str, Dict[str, Counter]] = {}
    totals: Counter = Counter()
    sources: Counter = Counter()
    for audio in audios:
        imported = parse_stem(audio.stem)[1] is not None
        if not imported and not include_all:
            continue
        group = "imported" if imported else "hand-made"
        sidecar = audio.with_suffix(".json")
        details, _ = load_crm_details(TRANSCRIPTS_DIR / f"{audio.stem}.transcript.json",
                                      sidecar if sidecar.exists() else None)
        details = details or {}
        if imported and sidecar.exists():
            sources[json.loads(sidecar.read_text(encoding="utf-8")).get("source", {}).get("vehicle_master", "?")] += 1
        call_type = details.get("audit_call_type")
        counts = groups.setdefault(group, {"filled": Counter(), "applies": Counter()})
        totals[group] += 1
        for label, path, only in FIELDS:
            if only and call_type != only:
                continue
            counts["applies"][label] += 1
            if _filled(_value(details, path)):
                counts["filled"][label] += 1

    if not totals:
        print("No imported recordings in data/ (REG_CDR stems). Add --all to include the hand-made set.")
        return
    names = list(groups)
    print(f"recordings: {dict(totals)}" + (f"   vehicle master from: {dict(sources)}" if sources else ""))
    print(f"\n{'field the audit reads':<22}" + "".join(f"{name:>16}" for name in names))
    for label, _path, only in FIELDS:
        cells = []
        for name in names:
            applies = groups[name]["applies"][label]
            cells.append(f"{groups[name]['filled'][label]}/{applies}" if applies else "-")
        note = f"  ({only} calls)" if only else ""
        print(f"{label:<22}" + "".join(f"{cell:>16}" for cell in cells) + note)


if __name__ == "__main__":
    main()
