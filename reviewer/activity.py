"""
What reviewers do on the reviewer dashboard, for its admin page.

- Live presence (in memory): each open page sends a ping every ~20 s with the
  reviewer's name and page; a page not heard from for 60 s counts as gone.
- Activity log (SQLite, reviews/activity.db): opened calls, saved / deleted
  reviews, exports, admin actions, and one row per active minute per reviewer
  (for time spent). Rows older than ACTIVITY_KEEP_DAYS are removed.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from config import ACTIVITY_KEEP_DAYS, REVIEWS_DIR

DB_PATH = REVIEWS_DIR / "activity.db"
ONLINE_FOR_S = 60
_LOCK = threading.Lock()
_PRESENCE: Dict[str, Dict[str, Any]] = {}  # tab id -> what that tab last reported
_PRUNED = {"at": 0.0}
_SCHEMA = """
CREATE TABLE IF NOT EXISTS activity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at REAL NOT NULL, day TEXT NOT NULL, reviewer TEXT NOT NULL, ip TEXT,
    action TEXT NOT NULL, stem TEXT, detail TEXT
);
CREATE INDEX IF NOT EXISTS activity_at ON activity (at);
CREATE TABLE IF NOT EXISTS active_minutes (
    reviewer TEXT NOT NULL, day TEXT NOT NULL, minute INTEGER NOT NULL,
    PRIMARY KEY (reviewer, day, minute)
);
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    conn.row_factory = sqlite3.Row
    return conn


def _prune(conn: sqlite3.Connection) -> None:
    """Drop rows older than the kept period (checked at most once an hour)."""
    if time.time() - _PRUNED["at"] < 3600:
        return
    _PRUNED["at"] = time.time()
    cutoff = time.time() - ACTIVITY_KEEP_DAYS * 86400
    conn.execute("DELETE FROM activity WHERE at < ?", (cutoff,))
    conn.execute("DELETE FROM active_minutes WHERE day < ?", (datetime.fromtimestamp(cutoff).strftime("%Y-%m-%d"),))


def log(reviewer: str, ip: str, action: str, stem: Optional[str] = None, detail: str = "") -> None:
    now = time.time()
    with _LOCK:
        conn = _connect()
        try:
            conn.execute("INSERT INTO activity (at, day, reviewer, ip, action, stem, detail) VALUES (?, ?, ?, ?, ?, ?, ?)",
                         (now, datetime.fromtimestamp(now).strftime("%Y-%m-%d"), reviewer or "unknown", ip, action,
                          stem, detail[:500]))
            _prune(conn)
            conn.commit()
        finally:
            conn.close()


def ping(tab: str, reviewer: str, ip: str, agent: str, page: str, stem: Optional[str]) -> None:
    """A page is open: remember where, and count this minute as active for the reviewer."""
    now = time.time()
    with _LOCK:
        seen = _PRESENCE.get(tab) or {"since": now}
        if seen.get("page") != page or seen.get("stem") != stem:
            seen["since"] = now
        seen.update(tab=tab, reviewer=reviewer or "unknown", ip=ip, agent=agent[:200], page=page, stem=stem, last_seen=now)
        _PRESENCE[tab] = seen
        for key in [k for k, v in _PRESENCE.items() if now - v["last_seen"] > 3600]:
            _PRESENCE.pop(key)
        if reviewer:
            conn = _connect()
            try:
                stamp = datetime.fromtimestamp(now)
                conn.execute("INSERT OR IGNORE INTO active_minutes VALUES (?, ?, ?)",
                             (reviewer, stamp.strftime("%Y-%m-%d"), stamp.hour * 60 + stamp.minute))
                conn.commit()
            finally:
                conn.close()


def leave(tab: str) -> None:
    with _LOCK:
        _PRESENCE.pop(tab, None)


def online() -> List[Dict[str, Any]]:
    now = time.time()
    with _LOCK:
        rows = [dict(v) for v in _PRESENCE.values() if now - v["last_seen"] <= ONLINE_FOR_S]
    return sorted(rows, key=lambda r: (str(r["reviewer"]).lower(), r["since"]))


def history(reviewer: str = "", day_from: str = "", day_to: str = "", action: str = "", limit: int = 500) -> List[Dict[str, Any]]:
    where, args = ["1"], []
    if reviewer:
        where.append("reviewer = ?"); args.append(reviewer)
    if day_from:
        where.append("day >= ?"); args.append(day_from)
    if day_to:
        where.append("day <= ?"); args.append(day_to)
    if action:
        where.append("action = ?"); args.append(action)
    with _LOCK:
        conn = _connect()
        try:
            rows = conn.execute(f"SELECT * FROM activity WHERE {' AND '.join(where)} ORDER BY id DESC LIMIT ?",
                                (*args, int(limit))).fetchall()
        finally:
            conn.close()
    return [dict(r) for r in rows]


def summary(day_from: str = "", day_to: str = "") -> List[Dict[str, Any]]:
    """Per reviewer: calls opened, reviews saved, calls reviewed, active minutes, first and last seen."""
    where, args = ["1"], []
    if day_from:
        where.append("day >= ?"); args.append(day_from)
    if day_to:
        where.append("day <= ?"); args.append(day_to)
    w = " AND ".join(where)
    with _LOCK:
        conn = _connect()
        try:
            acts = conn.execute(
                f"SELECT reviewer, sum(action = 'open_call') AS opened, sum(action = 'save_review') AS saves, "
                f"count(DISTINCT CASE WHEN action = 'save_review' THEN stem END) AS calls_reviewed, "
                f"sum(action = 'delete_review') AS deletes, min(at) AS first_at, max(at) AS last_at "
                f"FROM activity WHERE {w} GROUP BY reviewer", args).fetchall()
            mins = conn.execute(f"SELECT reviewer, count(*) AS n FROM active_minutes WHERE {w} GROUP BY reviewer",
                                args).fetchall()
        finally:
            conn.close()
    minutes = {r["reviewer"]: r["n"] for r in mins}
    out = {r["reviewer"]: {**dict(r), "active_minutes": minutes.get(r["reviewer"], 0)} for r in acts}
    for name, n in minutes.items():
        out.setdefault(name, {"reviewer": name, "opened": 0, "saves": 0, "calls_reviewed": 0, "deletes": 0,
                              "first_at": None, "last_at": None, "active_minutes": n})
    return sorted(out.values(), key=lambda r: -(r["last_at"] or 0))
