"""
Human reviews of audited calls (the reviewer dashboard writes them, the main
dashboard shows them next to the AI score).

One SQLite file: saving a review again adds a new version and never changes an
earlier one. Deleting (PIN-checked in the reviewer dashboard) only marks
versions deleted, with who and when: they drop out of both dashboards but stay
in the file. The AI's own results (audit_reports.xlsx, evidence JSON) are
never touched.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from config import REVIEWS_DIR
from core.audit.params import SCORE_OPTIONS
from core.audit.scoring import grade_band

DB_PATH = REVIEWS_DIR / "reviews.db"
STATUSES = ("in_review", "reviewed")

_LOCK = threading.Lock()
_SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    stem        TEXT NOT NULL,
    reviewer    TEXT NOT NULL,
    status      TEXT NOT NULL,
    created_at  REAL NOT NULL,
    body        TEXT NOT NULL      -- JSON: params, booking, call type, comments, ASR flags
);
CREATE INDEX IF NOT EXISTS reviews_stem ON reviews (stem, id);
"""


def _connect(read_only: bool = False) -> Optional[sqlite3.Connection]:
    if read_only:
        if not DB_PATH.exists():
            return None
        conn = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True, timeout=10)
    else:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")  # the main dashboard reads while reviewers write
        conn.executescript(_SCHEMA)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(reviews)")}
        for name, kind in (("deleted_at", "REAL"), ("deleted_by", "TEXT")):  # files made before delete existed
            if name not in columns:
                conn.execute(f"ALTER TABLE reviews ADD COLUMN {name} {kind}")
        conn.commit()
    conn.row_factory = sqlite3.Row
    return conn


def _live(conn: sqlite3.Connection) -> str:
    """SQL condition for versions not deleted (a file from before delete existed has none deleted)."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(reviews)")}
    return "deleted_at IS NULL" if "deleted_at" in columns else "1"


def _row(row: sqlite3.Row) -> Dict[str, Any]:
    return {"id": row["id"], "stem": row["stem"], "reviewer": row["reviewer"], "status": row["status"],
            "created_at": row["created_at"], **json.loads(row["body"])}


def _level(levels: Dict[str, float], rating: Any) -> str:
    return next((key for key in levels if key.lower() == str(rating or "").strip().lower()), "")


def rework_score(table: str, ai_points: Dict[str, float], ai_fake: bool, ai_wrong_type: bool,
                 review: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The call's score with the reviewer's ratings: their level where they disagree, the AI's
    points elsewhere, on the AI's points table. Not genuine or wrong call type → 0, as for the AI.
    ai_points: {parameter: the AI's points} for the parameters scored on this call."""
    options = SCORE_OPTIONS.get(table or "", {})
    if not options or not ai_points:
        return None
    earned = possible = 0.0
    for pid, ai in ai_points.items():
        levels = options.get(pid)
        if not levels:
            continue
        mine = (review.get("params") or {}).get(pid) or {}
        level = _level(levels, mine.get("rating")) if mine.get("verdict") == "disagree" else ""
        earned += float(levels[level]) if level else float(ai or 0)
        possible += float(max(levels.values()))
    booking, call_type = review.get("booking") or {}, review.get("call_type") or {}
    fake = (booking.get("genuine") is False) if booking.get("verdict") == "disagree" else bool(ai_fake)
    wrong_type = call_type.get("verdict") == "disagree" or (call_type.get("verdict") != "agree" and bool(ai_wrong_type))
    if fake or wrong_type:
        earned = 0.0
    pct = round(100 * earned / possible, 1) if possible else None
    return {"earned": round(earned, 2), "possible": round(possible, 2), "pct": pct,
            "grade": "Not Genuine" if fake or wrong_type else grade_band(pct),
            "zero_reason": "booking not genuine" if fake else "wrong call type" if wrong_type else None}


def report_inputs(report: Dict[str, Any]) -> Dict[str, Any]:
    """rework_score's AI side from a read_audit_scores report."""
    table = report.get("weight_table")
    result = report.get("audit_result") or {}
    options = SCORE_OPTIONS.get(table or "", {})
    points = {}
    for pid, item in (result.get("audit") or {}).items():
        levels = options.get(pid)
        if levels:
            points[pid] = float(levels.get(_level(levels, item.get("rating")), 0))
    return {"table": table, "ai_points": points,
            "ai_fake": (result.get("booking_genuine") or {}).get("genuine") is False,
            "ai_wrong_type": (report.get("management_summary") or {}).get("wrong_call_tagging") is True}


def with_score(review: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    """The review with its reviewer_score, worked out now when it was saved without one."""
    if review.get("reviewer_score") is None and inputs:
        return {**review, "reviewer_score": rework_score(review=review, **inputs)}
    return review


def add_review(stem: str, reviewer: str, status: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """Save a new version of the call's review and return it."""
    if status not in STATUSES:
        raise ValueError(f"status must be one of {', '.join(STATUSES)}")
    with _LOCK:
        conn = _connect()
        try:
            cur = conn.execute(
                "INSERT INTO reviews (stem, reviewer, status, created_at, body) VALUES (?, ?, ?, ?, ?)",
                (stem, reviewer, status, time.time(), json.dumps(body, ensure_ascii=False)),
            )
            conn.commit()
            return _row(conn.execute("SELECT * FROM reviews WHERE id = ?", (cur.lastrowid,)).fetchone())
        finally:
            conn.close()


def delete(stem: str, deleted_by: str, version_id: Optional[int] = None) -> int:
    """Mark one version (or every version) of a call's review deleted; returns how many."""
    with _LOCK:
        conn = _connect()
        try:
            where, args = "stem = ? AND deleted_at IS NULL", [stem]
            if version_id is not None:
                where, args = where + " AND id = ?", args + [int(version_id)]
            cur = conn.execute(f"UPDATE reviews SET deleted_at = ?, deleted_by = ? WHERE {where}",
                               [time.time(), deleted_by, *args])
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()


def history(stem: str) -> List[Dict[str, Any]]:
    """Every version of one call's review that isn't deleted, newest first."""
    conn = _connect(read_only=True)
    if conn is None:
        return []
    try:
        return [_row(r) for r in conn.execute(
            f"SELECT * FROM reviews WHERE stem = ? AND {_live(conn)} ORDER BY id DESC", (stem,))]
    finally:
        conn.close()


def latest(stems: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, Any]]:
    """stem → the newest version of its review that isn't deleted (all reviewed calls when stems is None)."""
    conn = _connect(read_only=True)
    if conn is None:
        return {}
    try:
        live = _live(conn)
        rows = conn.execute(
            f"SELECT r.* FROM reviews r JOIN (SELECT stem, max(id) AS id FROM reviews WHERE {live} GROUP BY stem) m "
            "ON m.id = r.id"
        ).fetchall()
    finally:
        conn.close()
    wanted = set(stems) if stems is not None else None
    return {r["stem"]: _row(r) for r in rows if wanted is None or r["stem"] in wanted}
