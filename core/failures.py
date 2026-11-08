"""
Calls whose transcription or audit failed, kept across restarts.

The dashboard lists them only on its Failed page: the Not audited list and its
Audit buttons leave them out, so a call that keeps failing isn't picked again
and again. A later successful audit of the call clears its entry.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from config import REPORTS_DIR

PATH = REPORTS_DIR / "failed_audits.json"
_LOCK = threading.Lock()
_CACHE: Dict[str, Any] = {"mtime": None, "items": {}}
_PHONE_RE = re.compile(r"(?<!\d)(?:\+91[\-\s]?)?[6-9]\d{9}(?!\d)")
# The line that says why, in the pipeline's own words, newest first.
_REASON_RE = re.compile(r"\[(?:MANUAL REVIEW|ERROR|FAILED)\]|Traceback|Error:|error:|failed", re.I)


def _mask(text: str) -> str:
    return _PHONE_RE.sub(lambda m: "XXXXXX" + re.sub(r"\D", "", m.group(0))[-4:], text)


def _load() -> Dict[str, Dict[str, Any]]:
    try:
        mtime = PATH.stat().st_mtime_ns
    except OSError:
        return {}
    if _CACHE["mtime"] != mtime:
        try:
            data = json.loads(PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        _CACHE.update(mtime=mtime, items=data if isinstance(data, dict) else {})
    return _CACHE["items"]


def _save(items: Dict[str, Dict[str, Any]]) -> None:
    PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = PATH.with_name(f".{PATH.name}.tmp")
    temporary.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    temporary.replace(PATH)
    # Two writes can share a timestamp tick: keep what was written, not a stale read.
    _CACHE.update(mtime=PATH.stat().st_mtime_ns, items=items)


def reason(log: str) -> str:
    """One line saying why it failed, from the job's output."""
    lines = [line.strip() for line in str(log or "").splitlines() if line.strip()]
    hit = next((line for line in reversed(lines) if _REASON_RE.search(line)), None)
    return _mask((hit or (lines[-1] if lines else "Failed"))[:300])


def record(stem: str, phase: str, log: str) -> None:
    """Remember that this call's transcription or audit failed."""
    with _LOCK:
        items = dict(_load())
        previous = items.get(stem) or {}
        lines = [line for line in str(log or "").splitlines() if line.strip()]
        items[stem] = {
            "stem": stem, "phase": phase, "reason": reason(log),
            "log": _mask("\n".join(lines[-25:])),
            "failed_at": time.time(), "attempts": int(previous.get("attempts", 0)) + 1,
        }
        _save(items)


def clear(stem: str) -> None:
    """The call has now been audited: it no longer counts as failed."""
    with _LOCK:
        items = _load()
        if stem in items:
            items = {key: value for key, value in items.items() if key != stem}
            _save(items)


def all_failed() -> Dict[str, Dict[str, Any]]:
    with _LOCK:
        return dict(_load())


def get(stem: str) -> Optional[Dict[str, Any]]:
    return all_failed().get(stem)
