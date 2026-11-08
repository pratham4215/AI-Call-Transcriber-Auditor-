"""
Kataria reviewer dashboard: human reviewers read audited calls and record what
should have happened, per parameter and per call.

Started on its own (reviewer_wsgi.py), never by wsgi.py. It only reads the
audit results, transcripts and recordings, and writes reviews to their own
file (core/reviews.py). No CRM database, no fetching, transcription, audits,
batches or engine controls: those routes don't exist here.

Customer phone numbers and personal details (name, address, email, PIN code)
are removed before anything reaches the browser (organisation policy).
"""
import hmac
import io
import json
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    DASHBOARD_LEAD_TYPE, DATA_DIR, REPORTS_DIR as REPORTS, REVIEW_ADMIN_PIN, REVIEW_DELETE_PIN, REVIEWS_DIR,
    TRANSCRIPTS_DIR as TRANSCRIPTS,
)
from datetime import timedelta  # noqa: E402
from flask import Flask, Response, jsonify, render_template, request, send_file, session  # noqa: E402
from core import removed, reviews  # noqa: E402
from core.audit.excel_report import EVIDENCE_DIR, read_all_scores, read_audit_scores  # noqa: E402
from core.audit.params import SCORE_OPTIONS  # noqa: E402
from reviewer import activity  # noqa: E402

app = Flask(__name__, template_folder=str(Path(__file__).resolve().parent / "templates"), static_folder=None)


def _session_key() -> bytes:
    """A random key kept beside the reviews, so admin sign-ins survive restarts."""
    path = REVIEWS_DIR / ".session_key"
    try:
        return path.read_bytes()
    except OSError:
        import secrets
        path.parent.mkdir(parents=True, exist_ok=True)
        key = secrets.token_bytes(32)
        path.write_bytes(key)
        return key


app.secret_key = _session_key()
app.config.update(PERMANENT_SESSION_LIFETIME=timedelta(days=30), SESSION_COOKIE_SAMESITE="Lax",
                  SESSION_COOKIE_NAME="kataria_review")


def is_admin() -> bool:
    return session.get("admin") is True


@app.context_processor
def _template_flags():
    return {"is_admin": is_admin()}

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac"}
CALL_TYPES = ("BOOKING", "FOLLOW_UPS", "SERVICE_NT_REQUIRED", "ESCALATION")
LEAD_TYPE = DASHBOARD_LEAD_TYPE or None
MAX_TEXT = 2000


@app.after_request
def _no_cache(response):
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


# ── Personal data (removed before anything reaches the browser) ───────────────
# A phone number: 10+ digits, also spaced ("98765 43210", "9 8 7 6 ..."), dashed
# in the usual groups, or in Devanagari / Gujarati digits. Not inside a word, so
# VIN and chassis numbers stay; dates and times (2026-07-23 12:30) don't match.
_D = r"[0-9०-९૦-૯]"
_EDGE_L, _EDGE_R = r"(?<![0-9A-Za-z०-९૦-૯])", r"(?![0-9A-Za-z०-९૦-૯])"
_NUMBER_RE = re.compile(
    rf"{_EDGE_L}(?:\+\s?91[\s-]?)?(?:{_D}(?: ?{_D}){{9,}}|{_D}{{5}}-{_D}{{5}}|{_D}{{3}}-{_D}{{3}}-{_D}{{4}}){_EDGE_R}")
_PERSONAL_KEYS = ("phone", "mobile", "whatsapp", "contact", "landline", "address", "email", "pincode",
                  "pin_code", "customername", "customer_name", "ownername", "owner_name")


def mask_text(value):
    if not isinstance(value, str):
        return value
    return _NUMBER_RE.sub("[number hidden]", value)


def strip_personal(value, parent: str = ""):
    """Drop customer contact and identity fields, mask numbers in every string."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            flat = re.sub(r"[^a-z_]", "", str(key).lower())
            if any(word in flat for word in _PERSONAL_KEYS):
                continue
            if parent == "customer" and flat in ("name", "fullname", "first_name", "last_name"):
                continue
            out[key] = strip_personal(item, flat)
        return out
    if isinstance(value, list):
        return [strip_personal(item, parent) for item in value]
    return mask_text(value)


# ── Local files ───────────────────────────────────────────────────────────────
_FILES = {"at": 0.0, "audio": {}, "sidecars": {}}
_FILES_LOCK = threading.Lock()


def _files():
    """stem → audio path and stem → JSON sidecar path, rescanned every 30 s."""
    with _FILES_LOCK:
        if time.monotonic() - _FILES["at"] > 30:
            audio, sidecars = {}, {}
            for path in DATA_DIR.rglob("*") if DATA_DIR.exists() else []:
                if not path.is_file():
                    continue
                if path.suffix.lower() in AUDIO_EXTS:
                    audio.setdefault(path.stem, path)
                elif path.suffix.lower() == ".json":
                    sidecars.setdefault(path.stem, path)
            _FILES.update(at=time.monotonic(), audio=audio, sidecars=sidecars)
        return _FILES["audio"], _FILES["sidecars"]


_JSON_CACHE: dict = {}  # path -> (mtime_ns, parsed JSON or None)


def _read_json(path):
    if path is None:
        return None
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return None
    hit = _JSON_CACHE.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = None
    _JSON_CACHE[path] = (mtime, data)
    return data


def _call_record(stem: str) -> dict:
    data = _read_json(_files()[1].get(stem)) or {}
    return data.get("call_data") if isinstance(data.get("call_data"), dict) else {}


def _evidence(stem: str) -> dict:
    name = Path(stem).name
    if name != stem or name.startswith("."):
        return {}
    data = _read_json(REPORTS / EVIDENCE_DIR / f"{name}.json")  # cached until the file changes
    return data if isinstance(data, dict) else {}


_LENGTHS: dict = {}  # transcript path -> (mtime_ns, call length in seconds or None); only the number is kept
_TURN_END_RE = re.compile(r"\[[\d.]+s-([\d.]+)s\]")


def _call_seconds(stem: str):
    """The recording's length in seconds, as saved in its transcript (the last turn's end when not saved)."""
    path = TRANSCRIPTS / f"{Path(stem).name}.transcript.json"
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return None
    hit = _LENGTHS.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    seconds = None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        value = data.get("audio_duration_seconds") if isinstance(data, dict) else None
        if isinstance(value, (int, float)) and value >= 0:
            seconds = round(value)
        elif isinstance(data, dict) and isinstance(data.get("turns"), list):
            ends = [float(m[1]) for turn in data["turns"] for m in [_TURN_END_RE.search(str(turn))] if m]
            seconds = round(max(ends)) if ends else None
    except (OSError, ValueError):
        pass
    _LENGTHS[path] = (mtime, seconds)
    return seconds


def _audited_calls() -> list:
    """Every audited call of the dashboard's lead type, with what the list shows."""
    scores = read_all_scores(REPORTS)
    latest = reviews.latest()
    audio = _files()[0]
    rows = []
    hidden = removed.stems()
    for stem, score in scores.items():
        if stem in hidden:  # removed by the admin
            continue
        call = _call_record(stem)
        meta = call.get("call_meta") or {}
        if LEAD_TYPE and meta.get("lead_type") and str(meta["lead_type"]).upper() != LEAD_TYPE:
            continue
        evidence = _evidence(stem)
        to_check = sum(1 for e in (evidence.get("params") or {}).values() if isinstance(e, dict) and e.get("check"))
        booking = evidence.get("booking") or {}
        to_check += 1 if booking.get("check") else 0
        when = str(meta.get("call_date") or score.get("call_date") or "")
        cre = call.get("cre") or {}
        review = latest.get(stem)
        if review:  # reviews saved before the reviewer score existed get one worked out now
            review = reviews.with_score(review, _score_inputs(stem, score))
        rows.append({
            "stem": stem,
            "registration": (call.get("vehicle") or {}).get("registration_number") or stem.split("_")[0],
            "call_date": when[:10] or None,
            "call_time": when[11:16] or None,
            "call_type": score.get("table"),
            "cre_name": cre.get("name") or score.get("cre_name") or None,
            "cre_code": cre.get("username") or cre.get("employee_id") or score.get("cre_id") or None,
            "workshop": (call.get("service_profile") or {}).get("workshop") or None,
            "campaign": (call.get("service_profile") or {}).get("campaign") or None,
            "disposition": str((call.get("disposition") or {}).get("status") or "").upper() or None,
            "score_pct": score.get("score_pct"),
            "grade": score.get("grade"),
            "fake_booking": booking.get("genuine") is False or score.get("grade") == "Not Genuine",
            "to_check": to_check,
            "call_seconds": _call_seconds(stem),
            "has_audio": stem in audio,
            "review": {"status": review["status"], "reviewer": review["reviewer"],
                       "created_at": review["created_at"],
                       "score_pct": (review.get("reviewer_score") or {}).get("pct"),
                       "grade": (review.get("reviewer_score") or {}).get("grade")} if review else None,
        })
    rows.sort(key=lambda r: (r["call_date"] or "", r["call_time"] or ""), reverse=True)
    return mask_rows(rows)


def mask_rows(rows: list) -> list:
    return [{key: mask_text(value) for key, value in row.items()} for row in rows]


def _turns(stem: str) -> list:
    """The transcript as turns T001.. (numbered like the audit's citations), numbers masked."""
    path = TRANSCRIPTS / f"{stem}.transcript.json"
    legacy = TRANSCRIPTS / f"{stem}.txt"
    lines = []
    if path.exists():
        payload = _read_json(path) or {}
        if isinstance(payload.get("turns"), list):
            lines = [str(turn) for turn in payload["turns"]]
        elif isinstance(payload.get("text"), str):
            lines = payload["text"].splitlines()
    elif legacy.exists():
        lines = legacy.read_text(encoding="utf-8", errors="replace").splitlines()
    turn_re = re.compile(r"^(Agent|Customer)\s*\[([\d.]+)s-([\d.]+)s\]:\s*(.*)$", re.I)
    turns = []
    for line in (line.strip() for line in lines):
        match = turn_re.match(line)
        if match:
            turns.append({"tid": f"T{len(turns) + 1:03d}", "role": match[1].title(),
                          "start": float(match[2]), "end": float(match[3]), "text": mask_text(match[4])})
    if not turns and lines:  # not diarized: one block of plain text
        return [{"tid": None, "role": None, "start": None, "end": None,
                 "text": mask_text("\n".join(lines))}]
    return turns


def _crm(stem: str) -> dict:
    """The call's CRM details as the audit saw them, without personal data (local files only)."""
    sidecar = _read_json(_files()[1].get(stem))
    try:
        if isinstance(sidecar, dict):
            from core.audit.crm_merger import merge_crm
            from core.audit.pipeline import normalize_crm_details
            call = sidecar.get("call_data") if isinstance(sidecar.get("call_data"), dict) else None
            return strip_personal(merge_crm(call, normalize_crm_details(sidecar)) or {})
        from core.audit.crm_excel import get_crm_by_stem
        return strip_personal(get_crm_by_stem(stem) or {})
    except Exception:
        return {}


# ── Pages ─────────────────────────────────────────────────────────────────────
@app.route("/")
def list_page():
    return render_template("review_list.html")


@app.route("/call/<path:stem>")
def call_page(stem):
    return render_template("review_call.html", stem=stem)


@app.route("/stats")
def stats_page():
    return render_template("review_stats.html")


# ── Read ──────────────────────────────────────────────────────────────────────
@app.route("/api/calls")
def api_calls():
    try:
        return jsonify({"items": _audited_calls()})
    except Exception as exc:
        app.logger.warning("listing audited calls failed: %s", exc)
        return jsonify({"error": "Could not read the audit results"}), 500


def _who() -> str:
    """The reviewer name a page sends (?who= or JSON reviewer), masked and trimmed."""
    body = request.get_json(silent=True) if request.is_json else None
    name = request.args.get("who") or (body or {}).get("reviewer") or ""
    return mask_text(" ".join(str(name).split())[:60]) or "unknown"


@app.route("/api/call/<path:stem>")
def api_call(stem):
    if stem in removed.stems():
        return jsonify({"error": "This call was removed by the admin"}), 404
    report = read_audit_scores(REPORTS, stem)
    if report is None:
        return jsonify({"error": "This call has no audit"}), 404
    activity.log(_who(), request.remote_addr, "open_call", stem)
    call = _call_record(stem)
    meta = call.get("call_meta") or {}
    if LEAD_TYPE and meta.get("lead_type") and str(meta["lead_type"]).upper() != LEAD_TYPE:
        return jsonify({"error": "This call has no audit"}), 404
    table = report.get("weight_table")
    evidence = strip_personal(_evidence(stem))
    return jsonify({
        "stem": stem,
        "report": strip_personal(report),
        "evidence": evidence,
        "score_options": SCORE_OPTIONS.get(table, {}),
        "turns": _turns(stem),
        "crm": _crm(stem),
        "has_audio": stem in _files()[0],
        "call_seconds": _call_seconds(stem),
        "reviews": [reviews.with_score(r, reviews.report_inputs(report)) for r in reviews.history(stem)],
    })


@app.route("/audio/<path:stem>")
def audio(stem):
    if stem not in read_all_scores(REPORTS) or stem in removed.stems():  # only listed audited calls
        return jsonify({"error": "not found"}), 404
    path = _files()[0].get(stem)
    if path is None:
        return jsonify({"error": "not found"}), 404
    return send_file(path, conditional=True)


# ── Save a review (a new version; earlier ones are kept) ──────────────────────
def _text(value, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    if len(value) > MAX_TEXT:
        raise ValueError(f"{field} is longer than {MAX_TEXT} characters")
    return mask_text(value.strip())


def _verdict(value, field: str):
    if value in (None, ""):
        return None
    if value not in ("agree", "disagree"):
        raise ValueError(f"{field} must be agree or disagree")
    return value


def _level(options: dict, rating) -> str:
    """The configured level name for a rating word (case-insensitive), or ''."""
    return next((key for key in options if key.lower() == str(rating or "").strip().lower()), "")


def clean_review(body: dict, table: str, ai_ratings: dict = None) -> dict:
    """Validate a review against the call's parameters; free text is masked like the rest.
    Picking the AI's own level counts as agreeing."""
    options = SCORE_OPTIONS.get(table, {})
    ai_ratings = ai_ratings or {}
    params = {}
    for pid, item in (body.get("params") or {}).items():
        if pid not in options or not isinstance(item, dict):
            raise ValueError(f"unknown parameter: {pid}")
        verdict = _verdict(item.get("verdict"), pid)
        rating = _level(options[pid], item.get("rating")) or None
        if verdict == "disagree":
            if not rating:
                raise ValueError(f"{pid}: pick the correct rating ({', '.join(options[pid])})")
            if rating == _level(options[pid], ai_ratings.get(pid)):
                verdict, rating = "agree", None
        else:
            rating = None
        note = _text(item.get("note"), f"{pid} note")
        if verdict or note:
            params[pid] = {"verdict": verdict, "rating": rating, "note": note}

    booking = body.get("booking") or {}
    genuine = booking.get("genuine")
    if genuine not in (None, True, False):
        raise ValueError("booking genuine must be yes, no or empty")
    call_type = body.get("call_type") or {}
    correct = call_type.get("correct") or None
    if correct is not None and correct not in CALL_TYPES:
        raise ValueError("correct call type must be one of: " + ", ".join(CALL_TYPES))

    flags = []
    for flag in body.get("asr_flags") or []:
        if not isinstance(flag, dict) or not re.fullmatch(r"T\d{3,4}", str(flag.get("turn") or "")):
            raise ValueError("an ASR flag needs a turn like T012")
        flags.append({"turn": flag["turn"], "note": _text(flag.get("note"), "ASR note")})
    if len(flags) > 200:
        raise ValueError("too many ASR flags")

    return {
        "params": params,
        "booking": {"verdict": _verdict(booking.get("verdict"), "booking"), "genuine": genuine,
                    "note": _text(booking.get("note"), "booking note")},
        "call_type": {"verdict": _verdict(call_type.get("verdict"), "call type"), "correct": correct,
                      "note": _text(call_type.get("note"), "call type note")},
        "overall": _text(body.get("overall"), "overall comment"),
        "coaching": _text(body.get("coaching"), "coaching"),
        "asr_flags": flags,
    }


@app.route("/api/call/<path:stem>/review", methods=["POST"])
def save_review(stem):
    report = read_audit_scores(REPORTS, stem)
    if report is None:
        return jsonify({"error": "This call has no audit"}), 404
    body = request.get_json(silent=True) or {}
    reviewer = " ".join(str(body.get("reviewer") or "").split())
    if not 2 <= len(reviewer) <= 60:
        return jsonify({"error": "Enter your name (2-60 characters)"}), 400
    status = body.get("status") or "reviewed"
    if status not in reviews.STATUSES:
        return jsonify({"error": "status must be in_review or reviewed"}), 400
    ai_ratings = {pid: item.get("rating") for pid, item in ((report.get("audit_result") or {}).get("audit") or {}).items()}
    try:
        clean = clean_review(body, report.get("weight_table"), ai_ratings)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    summary = report.get("management_summary") or {}
    # What the AI said when this review was written, for the agreement figures.
    clean["ai"] = {
        "score_pct": summary.get("score_pct"), "grade": summary.get("grade"),
        "ratings": ai_ratings, "call_type": report.get("weight_table"),
    }
    clean["reviewer_score"] = reviewer_score(report, clean)
    saved = reviews.add_review(stem, mask_text(reviewer), status, clean)
    score = clean["reviewer_score"] or {}
    activity.log(mask_text(reviewer), request.remote_addr, "save_review", stem,
                 f"{status}; reviewer score {score.get('pct')}")
    return jsonify(saved)


def reviewer_score(report: dict, review: dict) -> dict:
    """The call's score with the reviewer's ratings (core.reviews.rework_score)."""
    return reviews.rework_score(review=review, **reviews.report_inputs(report))


def _score_inputs(stem: str, score: dict) -> dict:
    """rework_score's AI side from the register's bulk read (no per-call workbook open)."""
    forced = score.get("grade") == "Not Genuine"
    fake = (_evidence(stem).get("booking") or {}).get("genuine") is False
    return {"table": score.get("table"), "ai_points": {pid: v[0] for pid, v in (score.get("params") or {}).items()},
            "ai_fake": fake, "ai_wrong_type": forced and not fake}


# ── Delete a review (PIN-checked; marks it deleted, the file keeps it) ────────
_PIN_FAILS: dict = {}  # client address -> times of recent wrong PINs
_PIN_LOCK = threading.Lock()
PIN_TRIES, PIN_LOCKOUT_S = 5, 600


def check_pin(kind: str, given, expected: str, locked_what: str):
    """None when the PIN is right; otherwise the error response. 5 wrong PINs from one address
    lock that kind of PIN for 10 minutes."""
    client = (request.remote_addr or "?", kind)
    now = time.time()
    with _PIN_LOCK:
        recent = [t for t in _PIN_FAILS.get(client, []) if now - t < PIN_LOCKOUT_S]
        _PIN_FAILS[client] = recent
        if len(recent) >= PIN_TRIES:
            wait = int(PIN_LOCKOUT_S - (now - recent[0])) // 60 + 1
            return jsonify({"error": f"Too many wrong PINs. Try again in {wait} minute{'s' if wait != 1 else ''}."}), 429
        if not expected or not hmac.compare_digest(str(given or "").strip(), expected):
            recent.append(now)
            left = PIN_TRIES - len(recent)
            return jsonify({"error": "Wrong PIN." + (f" {left} tr{'ies' if left != 1 else 'y'} left." if left else
                                                      f" {locked_what} is locked for 10 minutes.")}), 403
        _PIN_FAILS.pop(client, None)
    return None


@app.route("/api/call/<path:stem>/review/delete", methods=["POST"])
def delete_review(stem):
    """Body: pin (not needed when signed in as admin), reviewer, and version_id to delete one
    version (omit it to delete them all)."""
    body = request.get_json(silent=True) or {}
    if not is_admin():
        refused = check_pin("delete", body.get("pin"), REVIEW_DELETE_PIN, "Deleting")
        if refused:
            return refused
    version = body.get("version_id")
    if version is not None and not str(version).isdigit():
        return jsonify({"error": "version_id must be a number"}), 400
    who = " ".join(str(body.get("reviewer") or "").split())[:60] or "unknown"
    if is_admin():
        who = f"{who} (admin)"
    deleted = reviews.delete(stem, mask_text(who), int(version) if version is not None else None)
    if not deleted:
        return jsonify({"error": "Nothing to delete: this review was already deleted."}), 404
    activity.log(who, request.remote_addr, "delete_review", stem, f"version {version}" if version is not None else "all versions")
    report = read_audit_scores(REPORTS, stem)
    inputs = reviews.report_inputs(report) if report else {}
    return jsonify({"deleted": deleted, "reviews": [reviews.with_score(r, inputs) for r in reviews.history(stem)]})


# ── Agreement figures and export ──────────────────────────────────────────────
def _reviewed(day_from: str = "", day_to: str = "") -> list:
    """(call row, latest review) for reviewed calls, optionally within call dates."""
    latest = reviews.latest()
    scores = read_all_scores(REPORTS)
    rows = []
    for row in _audited_calls():
        review = latest.get(row["stem"])
        if review and row["stem"] in scores:
            review = reviews.with_score(review, _score_inputs(row["stem"], scores[row["stem"]]))
        if not review:
            continue
        day = row["call_date"] or ""
        if (day_from and day < day_from) or (day_to and day > day_to):
            continue
        rows.append((row, review))
    return rows


@app.route("/api/stats")
def api_stats():
    per_param: dict = {}
    booking = {"agree": 0, "disagree": 0}
    call_type = {"agree": 0, "disagree": 0}
    by_reviewer: dict = {}
    pairs = _reviewed(request.args.get("from", ""), request.args.get("to", ""))
    for row, review in pairs:
        by_reviewer[review["reviewer"]] = by_reviewer.get(review["reviewer"], 0) + 1
        for pid, item in (review.get("params") or {}).items():
            if item.get("verdict"):
                counts = per_param.setdefault(pid, {"agree": 0, "disagree": 0})
                counts[item["verdict"]] += 1
        for name, target in (("booking", booking), ("call_type", call_type)):
            verdict = (review.get(name) or {}).get("verdict")
            if verdict:
                target[verdict] += 1
    return jsonify({"reviewed_calls": len(pairs), "params": per_param, "booking": booking,
                    "call_type": call_type, "reviewers": by_reviewer})


@app.route("/api/export")
def api_export():
    from openpyxl import Workbook

    pairs = _reviewed(request.args.get("from", ""), request.args.get("to", ""))
    activity.log(_who(), request.remote_addr, "export",
                 detail=f"{len(pairs)} reviewed calls, {request.args.get('from') or 'all'} to {request.args.get('to') or 'all'}")
    book = Workbook()
    calls = book.active
    calls.title = "Reviews"
    calls.append(["Call", "Registration", "Call date", "Call type", "CRE", "AI score %", "AI grade",
                  "Reviewer score %", "Reviewer grade", "Reviewer", "Status", "Reviewed at", "Booking verdict", "Booking genuine", "Booking note",
                  "Call type verdict", "Correct call type", "Call type note", "Overall comment", "Coaching"])
    params = book.create_sheet("Parameters")
    params.append(["Call", "Parameter", "AI rating", "Verdict", "Reviewer rating", "Note"])
    flags = book.create_sheet("Transcript flags")
    flags.append(["Call", "Turn", "Note"])

    def safe(value):  # no spreadsheet formulas from typed text
        return "'" + value if isinstance(value, str) and value.startswith(("=", "+", "-", "@")) else value

    for row, review in pairs:
        b, ct = review.get("booking") or {}, review.get("call_type") or {}
        calls.append([safe(v) for v in (
            row["stem"], row["registration"], row["call_date"], row["call_type"],
            " ".join(filter(None, [row["cre_name"], f"({row['cre_code']})" if row["cre_code"] else ""])),
            row["score_pct"], row["grade"], (review.get("reviewer_score") or {}).get("pct"),
            (review.get("reviewer_score") or {}).get("grade"), review["reviewer"], review["status"],
            datetime.fromtimestamp(review["created_at"]).strftime("%Y-%m-%d %H:%M"),
            b.get("verdict"), {True: "Yes", False: "No"}.get(b.get("genuine")), b.get("note"),
            ct.get("verdict"), ct.get("correct"), ct.get("note"), review.get("overall"), review.get("coaching"),
        )])
        ai = (review.get("ai") or {}).get("ratings") or {}
        for pid, item in (review.get("params") or {}).items():
            params.append([safe(v) for v in (row["stem"], pid, ai.get(pid), item.get("verdict"),
                                              item.get("rating"), item.get("note"))])
        for flag in review.get("asr_flags") or []:
            flags.append([safe(v) for v in (row["stem"], flag.get("turn"), flag.get("note"))])
    buffer = io.BytesIO()
    book.save(buffer)
    name = f"reviews-{request.args.get('from') or 'all'}{'-' + request.args['to'] if request.args.get('to') else ''}.xlsx"
    return Response(buffer.getvalue(), headers={"Content-Disposition": f"attachment; filename={name}"},
                    mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ── Presence: every open page says where it is every ~20 s ────────────────────
@app.route("/api/ping", methods=["POST"])
def api_ping():
    body = request.get_json(silent=True) or {}
    tab = str(body.get("tab") or "")[:40]
    page = str(body.get("page") or "")[:20]
    if not re.fullmatch(r"[A-Za-z0-9-]{8,40}", tab) or page not in ("list", "call", "stats", "admin"):
        return jsonify({"error": "bad ping"}), 400
    stem = str(body.get("stem") or "")[:80] or None
    name = _who()  # a page that has no name yet shows as "unknown" and counts no active time
    activity.ping(tab, "" if name == "unknown" else name, request.remote_addr or "?",
                  request.headers.get("User-Agent", ""), page, stem)
    return jsonify({"ok": True})


@app.route("/api/leave", methods=["POST"])
def api_leave():
    body = request.get_json(silent=True) or {}
    activity.leave(str(body.get("tab") or "")[:40])
    return jsonify({"ok": True})


# ── Admin (PIN sign-in; stays signed in on that browser for 30 days) ──────────
def _admin_only():
    return None if is_admin() else (jsonify({"error": "Admin only: sign in with the admin PIN"}), 403)


@app.route("/api/admin/login", methods=["POST"])
def admin_login():
    body = request.get_json(silent=True) or {}
    refused = check_pin("admin", body.get("pin"), REVIEW_ADMIN_PIN, "Admin sign-in")
    if refused:
        activity.log(_who(), request.remote_addr, "admin_login_failed")
        return refused
    session.permanent = True
    session["admin"] = True
    activity.log(_who(), request.remote_addr, "admin_login")
    return jsonify({"admin": True})


@app.route("/api/admin/logout", methods=["POST"])
def admin_logout():
    if is_admin():
        activity.log(_who(), request.remote_addr, "admin_logout")
    session.pop("admin", None)
    return jsonify({"admin": False})


@app.route("/admin")
def admin_page():
    return render_template("review_admin.html")


@app.route("/api/admin/online")
def admin_online():
    return _admin_only() or jsonify({"items": activity.online()})


@app.route("/api/admin/activity")
def admin_activity():
    a = request.args
    return _admin_only() or jsonify({"items": activity.history(
        a.get("reviewer", ""), a.get("from", ""), a.get("to", ""), a.get("action", ""),
        min(max(int(a.get("limit", 500) or 500), 1), 5000))})


@app.route("/api/admin/summary")
def admin_summary():
    return _admin_only() or jsonify({"items": activity.summary(request.args.get("from", ""), request.args.get("to", ""))})


@app.route("/api/admin/removed")
def admin_removed():
    if not is_admin():
        return _admin_only()
    scores = read_all_scores(REPORTS)
    items = []
    for stem, info in removed.all_removed().items():
        call = _call_record(stem)
        when = str((call.get("call_meta") or {}).get("call_date") or (scores.get(stem) or {}).get("call_date") or "")
        items.append({**info, "registration": (call.get("vehicle") or {}).get("registration_number") or stem.split("_")[0],
                      "call_date": when[:10] or None, "call_type": (scores.get(stem) or {}).get("table"),
                      "score_pct": (scores.get(stem) or {}).get("score_pct")})
    items.sort(key=lambda i: -i["removed_at"])
    return jsonify({"items": mask_rows(items)})


@app.route("/api/admin/call/<path:stem>/remove", methods=["POST"])
def admin_remove(stem):
    if not is_admin():
        return _admin_only()
    if stem not in read_all_scores(REPORTS):
        return jsonify({"error": "This call has no audit"}), 404
    reason = mask_text(str((request.get_json(silent=True) or {}).get("reason") or "").strip()[:500])
    removed.remove(stem, f"{_who()} (admin)", reason)
    activity.log(_who(), request.remote_addr, "remove_call", stem, reason)
    return jsonify({"removed": True})


@app.route("/api/admin/call/<path:stem>/restore", methods=["POST"])
def admin_restore(stem):
    if not is_admin():
        return _admin_only()
    if not removed.restore(stem):
        return jsonify({"error": "This call isn't removed"}), 404
    activity.log(_who(), request.remote_addr, "restore_call", stem)
    return jsonify({"restored": True})
