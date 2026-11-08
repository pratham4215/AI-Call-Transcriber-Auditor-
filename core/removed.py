"""
Audited calls the admin removed from the dashboards (reviewer dashboard → Admin).

Removing hides a call everywhere (audited lists, reviews, exports, Not audited)
but deletes nothing: the audit row, evidence, transcript and recording stay on
disk, and each removal records who and when. Restore puts the call back.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, Optional

from config import REPORTS_DIR

PATH = REPORTS_DIR / "removed_calls.json"
_LOCK = threading.Lock()
_CACHE: Dict[str, Any] = {"mtime": None, "items": {}}


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
    _CACHE.update(mtime=PATH.stat().st_mtime_ns, items=items)


def remove(stem: str, by: str, reason: str = "") -> None:
    with _LOCK:
        items = dict(_load())
        items[stem] = {"stem": stem, "removed_by": by, "removed_at": time.time(), "reason": reason}
        _save(items)


def restore(stem: str) -> bool:
    with _LOCK:
        items = dict(_load())
        if stem not in items:
            return False
        items.pop(stem)
        _save(items)
        return True


def all_removed() -> Dict[str, Dict[str, Any]]:
    with _LOCK:
        return dict(_load())


def stems() -> set:
    """Removed call stems (an unreadable list hides nothing)."""
    try:
        return set(all_removed())
    except Exception:
        return set()


def get(stem: str) -> Optional[Dict[str, Any]]:
    return all_removed().get(stem)
