"""
Kataria AI Audit — Flask backend
"""
import re
import os
import sys
import copy
import json
import uuid
import time
import ipaddress
import signal
import shutil
import threading
import subprocess
from datetime import date
from pathlib import Path

# Ensure project root is on sys.path so config and workers resolve correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    BASE, DATA_DIR, TRANSCRIPTS_DIR as TRANSCRIPTS, REPORTS_DIR as REPORTS,
    OLLAMA_MODEL, CRM_SOURCE, DASHBOARD_LEAD_TYPE,
)
from flask import Flask, g, jsonify, request, render_template, Response, send_file
from core import db as _db
from core import failures as _failures
from core import removed as _removed
from core import reviews as _reviews
from core.audit.crm_call_data import normalize_reg
from core.audit.crm_excel import get_crm_by_stem
from core.audit.excel_report import (
    export_calls,
    list_audited_call_ids,
    read_all_scores,
    read_audit_scores,
    read_call_evidence,
    record_call_runtime,
)
from core.audit.utils import _redact_payload
try:
    # Call data by registration number: PostgreSQL, else CALL DATA.xlsx
    from core.audit import crm_db as _crm_db
    _cd_get_crm = _crm_db.get_call_crm  # → CallData (crm, source, vehicle, ...)
except Exception:
    _crm_db = None
    _cd_get_crm = None

app = Flask(__name__, template_folder=str(BASE / "web" / "templates"))


@app.after_request
def _disable_api_caching(response):
    """Live job, batch, and artifact state must never be served from cache."""
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        started = getattr(g, "request_started", None)
        if started is not None:
            elapsed = time.perf_counter() - started
            response.headers["Server-Timing"] = f"app;dur={elapsed * 1000:.0f}"
            if elapsed > 1.0 and request.path not in ("/api/audio",) and not request.path.startswith("/api/audio/"):
                print(f"[slow] {request.method} {request.full_path.rstrip('?')} {elapsed:.1f}s", flush=True)
    return response


@app.before_request
def _start_request_timer():
    g.request_started = time.perf_counter()

# ── Core module paths ─────────────────────────────────────────────────────────
TRANSCRIBER    = str(BASE / "core" / "transcriber.py")
AUDITOR_MODULE = "core.audit.pipeline"

PYTHON        = sys.executable

AUDIO_EXTS    = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac"}


def _iter_audio() -> list:
    """Recursively yield all audio files under DATA_DIR (sorted by name)."""
    return sorted(
        (path for path in DATA_DIR.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTS),
        key=lambda path: path.name,
    )


def _scan_data() -> tuple[list[Path], dict[str, Path]]:
    """Build the audio/sidecar inventory with one recursive directory walk."""
    audio: list[Path] = []
    sidecars: dict[str, Path] = {}
    for path in DATA_DIR.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in AUDIO_EXTS:
            audio.append(path)
        elif suffix == ".json":
            sidecars.setdefault(path.stem, path)
    audio.sort(key=lambda path: path.name)
    return audio, sidecars


def _find_audio(stem: str) -> Path | None:
    for f in DATA_DIR.rglob("*"):
        if f.stem == stem and f.suffix.lower() in AUDIO_EXTS:
            return f
    return None


def _find_json(stem: str) -> Path | None:
    for f in DATA_DIR.rglob("*.json"):
        if f.stem == stem:
            return f
    return None


def _find_transcript(stem: str) -> Path | None:
    """Prefer the current JSON artifact but continue reading legacy transcripts."""
    current = TRANSCRIPTS / f"{stem}.transcript.json"
    if current.exists():
        return current
    legacy = TRANSCRIPTS / f"{stem}.txt"
    return legacy if legacy.exists() else None


def _normalise_call_type(value: object) -> str:
    key = re.sub(r"[^A-Z]+", "_", str(value or "").upper()).strip("_")
    aliases = {
        "BOOKING": "BOOKING",
        "ESCALATION": "ESCALATION",
        "FOLLOW_UP": "FOLLOW_UPS",
        "FOLLOW_UPS": "FOLLOW_UPS",
        "SERVICE_NT_REQUIRED": "SERVICE_NT_REQUIRED",
        "SERVICE_NOT_REQUIRED": "SERVICE_NT_REQUIRED",
    }
    return aliases.get(key, "UNKNOWN")


# Only calls on this lead type (UNNATI / PRAGATI) are listed, fetched and exported; None = both.
LEAD_TYPE = DASHBOARD_LEAD_TYPE or None


def _call_type_for_audio(audio: Path, excel_crm: dict | None) -> str:
    if excel_crm:
        call_type = _normalise_call_type(excel_crm.get("audit_call_type"))
        if call_type != "UNKNOWN":
            return call_type
    try:
        relative_parts = audio.relative_to(DATA_DIR).parts[:-1]
    except ValueError:
        relative_parts = audio.parts[:-1]
    for part in reversed(relative_parts):
        call_type = _normalise_call_type(part)
        if call_type != "UNKNOWN":
            return call_type
    return "UNKNOWN"


# ── Job registry ─────────────────────────────────────────────────────────────
JOBS: dict = {}
_JOBS_LOCK = threading.Lock()
_ACTIVE: dict = {}
_ACTIVE_LOCK = threading.Lock()
_RUNNING_PROCESSES: dict[str, subprocess.Popen] = {}
_PROCESS_LOCK = threading.Lock()

# ── Batch registry ────────────────────────────────────────────────────────────
BATCHES: dict = {}
_BATCHES_LOCK = threading.Lock()

# ── Phone-mask (PII protection — required by org policy) ─────────────────────
_PHONE_RE = re.compile(r"(?<!\d)(?:\+91[\-\s]?)?[6-9]\d{9}(?!\d)")


def _mask(s: str) -> str:
    def _m(match):
        digits = re.sub(r"\D", "", match.group(0))
        return "XXXXXX" + digits[-4:] if len(digits) >= 4 else "XXXXXX"
    return _PHONE_RE.sub(_m, s)


def _safe(name: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", _mask(name))


def _crm_for_browser(details: dict) -> dict:
    """Return only browser-safe CRM data; audit internals keep the full record."""
    blocked = {
        "phone", "phonenumber", "mobile", "mobilenumber", "mobile_number",
        "address", "customeraddress", "customer_address", "pincode", "pin_code",
    }

    def _strip(value):
        if isinstance(value, dict):
            return {
                key: _strip(item)
                for key, item in value.items()
                if re.sub(r"[^a-z_]", "", str(key).lower()) not in blocked
            }
        if isinstance(value, list):
            return [_strip(item) for item in value]
        return value

    return _strip(_redact_payload(details))


# ── Engine ────────────────────────────────────────────────────────────────────
try:
    from workers.engine import get_engine
    _ENGINE_AVAILABLE = True
except ImportError:
    _ENGINE_AVAILABLE = False


def _use_engine() -> bool:
    if not _ENGINE_AVAILABLE:
        return False
    eng = get_engine()
    if eng.status == "idle":
        eng.start()
    return eng.status in {"loading", "ready"}


# ── Subprocess job runner ─────────────────────────────────────────────────────
MAX_LOG_LINES = 80
MAX_RETAINED_JOBS = 1000
MAX_RETAINED_BATCHES = 50
MAX_RUN_CALLS = 20000  # ticked CRM calls in one run: a whole day (~12k connected calls)


def _persist_runtime(stem_key: str, runtime_s: float) -> None:
    if ":" not in stem_key:
        return
    prefix, stem = stem_key.split(":", 1)
    phase = {"tx": "transcription", "au": "audit"}.get(prefix)
    if not phase or not stem:
        return
    try:
        record_call_runtime(REPORTS, stem, phase, runtime_s)
    except Exception as exc:  # runtime persistence must not fail the completed job
        print(f"[runtime] Could not persist {phase} runtime: {exc}", file=sys.stderr)


def _run(job_id: str, cmd: list, stem_key: str = ""):
    try:
        started_at = time.time()
        with _JOBS_LOCK:
            JOBS[job_id]["status"] = "running"
            JOBS[job_id]["log"] = []
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=str(BASE),
            start_new_session=(os.name == "posix"),
        )
        with _PROCESS_LOCK:
            _RUNNING_PROCESSES[job_id] = proc
        for line in proc.stdout:
            line = line.rstrip("\n")
            with _JOBS_LOCK:
                log = JOBS[job_id].setdefault("log", [])
                log.append(line)
                if len(log) > MAX_LOG_LINES:
                    log.pop(0)
                JOBS[job_id]["progress"] = line
        proc.wait()
        with _JOBS_LOCK:
            if proc.returncode != 0:
                tail = "\n".join(JOBS[job_id].get("log", [])[-30:])
                JOBS[job_id]["status"] = "error"
                JOBS[job_id]["error"] = tail or f"Process exited with code {proc.returncode}"
            else:
                runtime_s = round(time.time() - started_at, 1)
                JOBS[job_id]["runtime_s"] = runtime_s
                JOBS[job_id]["status"] = "done"
        if proc.returncode == 0 and stem_key:
            _persist_runtime(stem_key, runtime_s)
        with _JOBS_LOCK:
            outcome = (JOBS[job_id]["status"] == "done", JOBS[job_id].get("error") or "")
        _note_outcome(stem_key, *outcome)
    except Exception as exc:
        with _JOBS_LOCK:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = str(exc)
        _note_outcome(stem_key, False, str(exc))
    finally:
        with _PROCESS_LOCK:
            _RUNNING_PROCESSES.pop(job_id, None)
        if stem_key:
            with _ACTIVE_LOCK:
                if _ACTIVE.get(stem_key) == job_id:
                    _ACTIVE.pop(stem_key, None)
        _invalidate_call_rows()  # new transcript / audit report


def _note_outcome(stem_key: str, success: bool, log: str) -> None:
    """Failed transcriptions and audits go to the Failed page; a finished audit clears it."""
    prefix, _, stem = (stem_key or "").partition(":")
    phase = {"tx": "transcription", "au": "audit"}.get(prefix)
    if not phase or not stem:
        return
    try:
        if not success:
            _failures.record(stem, phase, log)
        elif phase == "audit":
            _failures.clear(stem)
    except Exception as exc:  # the failure list must not break the job itself
        print(f"[dashboard] could not update the failed list: {exc}", flush=True)


def _new_job(status="queued") -> tuple:
    job_id = str(uuid.uuid4())
    with _JOBS_LOCK:
        if len(JOBS) >= MAX_RETAINED_JOBS:
            removable = [
                key for key, value in JOBS.items()
                if value.get("status") not in {"queued", "running"}
            ]
            for key in removable[: max(1, len(JOBS) - MAX_RETAINED_JOBS + 1)]:
                JOBS.pop(key, None)
        JOBS[job_id] = {"status": status}
    return job_id


def _done_cb(job_id: str, stem_key: str):
    """Callback for engine-dispatched jobs."""
    def _cb(jid: str, success: bool, log: str):
        runtime_s = None
        with _JOBS_LOCK:
            started_at = JOBS[job_id].get("started_at")
            if success:
                JOBS[job_id]["status"] = "done"
            else:
                JOBS[job_id]["status"] = "error"
                JOBS[job_id]["error"] = log
            JOBS[job_id]["progress"] = log.splitlines()[-1] if log else ""
            if started_at is not None:
                runtime_s = round(time.time() - started_at, 1)
                JOBS[job_id]["runtime_s"] = runtime_s
        _invalidate_call_rows()  # new transcript / audit report
        if success and runtime_s is not None:
            _persist_runtime(stem_key, runtime_s)
        _note_outcome(stem_key, success, log)
        with _ACTIVE_LOCK:
            if _ACTIVE.get(stem_key) == job_id:
                _ACTIVE.pop(stem_key, None)
    return _cb


def _progress_cb(job_id: str):
    """Store live worker progress without exposing unmasked phone numbers."""
    def _cb(message: str):
        safe_message = _mask(str(message))
        match = re.match(r"^\[(\d{1,3})%\]\s*(.*)$", safe_message)
        with _JOBS_LOCK:
            if job_id in JOBS:
                if match:
                    JOBS[job_id]["phase_percent"] = max(0, min(100, int(match.group(1))))
                    JOBS[job_id]["progress"] = match.group(2)
                else:
                    JOBS[job_id]["progress"] = safe_message
    return _cb


def _terminate_child_processes() -> int:
    """Terminate only subprocesses launched by this dashboard process."""
    with _PROCESS_LOCK:
        processes = list(_RUNNING_PROCESSES.values())
    stopped = 0
    for proc in processes:
        if proc.poll() is not None:
            continue
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
            stopped += 1
        except Exception:
            continue
    deadline = time.time() + 2.0
    for proc in processes:
        if proc.poll() is not None:
            continue
        remaining = max(0.0, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
        except Exception:
            try:
                if os.name == "posix":
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    proc.kill()
            except Exception:
                pass
    return stopped


def _terminate_owned_ollama() -> tuple[bool, str]:
    """Kill Ollama only when owned by the OS user running this dashboard."""
    if os.name != "posix" or not hasattr(os, "getuid"):
        return False, "Ollama termination is supported on the Linux DGX host only."
    pkill = shutil.which("pkill")
    if not pkill:
        return False, "pkill is not installed on this host."
    completed = subprocess.run(
        [pkill, "-TERM", "-u", str(os.getuid()), "-x", "ollama"],
        capture_output=True, text=True, timeout=5, check=False,
    )
    # pkill: 0 matched; 1 means no matching process. Both are safe outcomes.
    if completed.returncode in {0, 1}:
        return completed.returncode == 0, ""
    return False, (completed.stderr or "Could not terminate Ollama.").strip()


def _schedule_server_stop(delay: float = 0.75) -> None:
    """Stop this dashboard process after the HTTP response has been flushed."""
    def _stop():
        os.kill(os.getpid(), signal.SIGTERM)

    timer = threading.Timer(delay, _stop)
    timer.daemon = True
    timer.start()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/batch")
def batch_page():
    return render_template("batch.html")


@app.route("/diagrams")
def diagrams_page():
    return render_template("diagrams.html")


@app.route("/not-audited")
def not_audited_page():
    return render_template("not_audited.html")


@app.route("/fake-bookings")
def fake_bookings_page():
    return render_template("fake_bookings.html")


_SIDECAR_CALLS: dict = {}  # JSON path -> (mtime_ns, call record or None)


def _sidecar_call_data(path: Path | None) -> dict | None:
    """The call record an importer JSON file carries (cached until the file changes)."""
    if path is None:
        return None
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return None
    hit = _SIDECAR_CALLS.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        call = data.get("call_data") if isinstance(data, dict) and isinstance(data.get("call_data"), dict) else None
    except (OSError, ValueError):
        call = None
    _SIDECAR_CALLS[path] = (mtime, call)
    return call


_LEGACY_CALLS: dict = {}   # stem -> call record of a hand-made recording (historical: never changes)
_LEGACY_MISSES: dict = {}  # stem -> when its lookup last failed (retried after 10 minutes)
_LEGACY_BATCH = 100       # stems per lookup, so one slow query can't time out the whole set


_LEGACY_LOOKUP = {"running": False}


def _legacy_lookup(todo: list) -> None:
    started = time.monotonic()
    found = {}
    try:
        for start in range(0, len(todo), _LEGACY_BATCH):
            try:
                # One lookup per batch instead of one per recording
                found.update(_crm_db.get_call_crm_many(todo[start:start + _LEGACY_BATCH], with_vehicle=False))
            except Exception:
                pass
        for stem in todo:
            if stem in found:
                _LEGACY_CALLS[stem] = found[stem].crm
            else:
                _LEGACY_MISSES[stem] = started
        print(f"[dashboard] looked up {len(todo)} recordings without a call record: "
              f"{len(found)} found in {time.monotonic() - started:.0f}s")
    finally:
        _LEGACY_LOOKUP["running"] = False
        _invalidate_call_rows()


def _legacy_call_data(stems: list) -> dict:
    """Call records for recordings without one in their JSON file, looked up once per
    process in the background: the page lists them once they are found."""
    now = time.monotonic()
    todo = [s for s in stems if s not in _LEGACY_CALLS and now - _LEGACY_MISSES.get(s, -1e9) > 600]
    if todo and _crm_db is not None and not _LEGACY_LOOKUP["running"]:
        _LEGACY_LOOKUP["running"] = True
        threading.Thread(target=_legacy_lookup, args=(todo,), daemon=True, name="legacy-lookup").start()
    return {s: _LEGACY_CALLS[s] for s in stems if s in _LEGACY_CALLS}


_TX_LEAD_TYPES: dict = {}  # transaction id -> lead type of a call imported before lead_type was saved
_TX_MISSES: dict = {}      # transaction id -> when its lookup last failed (retried after 10 minutes)


def _tx_lead_types(transaction_ids: list) -> dict:
    """Lead types of imported calls whose JSON file predates lead_type, looked up once per process."""
    now = time.monotonic()
    todo = [t for t in transaction_ids if t not in _TX_LEAD_TYPES and now - _TX_MISSES.get(t, -1e9) > 600]
    if todo and _crm_db is not None:
        try:
            found = _crm_db.lead_types(todo)
        except Exception as exc:
            print(f"[dashboard] lead type lookup failed: {exc}")
            found = {}
        for tx in todo:
            if found.get(tx):
                _TX_LEAD_TYPES[tx] = found[tx]
            else:
                _TX_MISSES[tx] = now
    return {t: _TX_LEAD_TYPES[t] for t in transaction_ids if t in _TX_LEAD_TYPES}


_ROWS_CACHE: dict = {"at": 0.0, "rows": None}
_ROWS_LOCK = threading.Lock()


def _invalidate_call_rows() -> None:
    with _ROWS_LOCK:
        _ROWS_CACHE["rows"] = None


def _call_rows() -> list:
    """Every recording in data/ with what the dashboard lists (see _build_call_rows).

    A day view fires several requests at once; they share one scan, reused for a
    few seconds. Downloads invalidate it so a new recording shows straight away.
    """
    with _ROWS_LOCK:
        if _ROWS_CACHE["rows"] is not None and time.monotonic() - _ROWS_CACHE["at"] < 3.0:
            return _ROWS_CACHE["rows"]
        rows = _build_call_rows()
        _ROWS_CACHE.update(at=time.monotonic(), rows=rows)
        return rows


def _build_call_rows() -> list:
    """Every recording in data/ of the dashboard's lead type with what the dashboard lists:
    call date and time, registration, CRE, workshop, call type, pipeline status and audit score."""
    if not DATA_DIR.exists():
        return []
    audio_files, sidecars = _scan_data()
    try:
        scores = read_all_scores(REPORTS)
    except Exception:
        scores = {}
    # Imported calls carry their own record; the hand-made set, and calls imported
    # before lead_type was saved, are looked up.
    own = {f.stem: _sidecar_call_data(sidecars.get(f.stem)) for f in audio_files}
    meta = {stem: (call or {}).get("call_meta") or {} for stem, call in own.items()}
    untyped = [stem for stem in own if not meta[stem].get("lead_type")]
    # By transaction id (fast) when the file has one, else the full stem lookup.
    tx_types = _tx_lead_types([str(meta[s]["transaction_id"]) for s in untyped if meta[s].get("transaction_id")])
    legacy = _legacy_call_data([s for s in untyped if not meta[s].get("transaction_id")])
    rows = []
    for f in audio_files:
        stem = f.stem
        sidecar = sidecars.get(stem)
        crm = own[stem] or legacy.get(stem) or {}
        lead_type = (meta[stem].get("lead_type")
                     or tx_types.get(str(meta[stem].get("transaction_id")))
                     or ((legacy.get(stem) or {}).get("call_meta") or {}).get("lead_type"))
        score = scores.get(stem) or {}
        cre = crm.get("cre") or {}
        when = str((crm.get("call_meta") or {}).get("call_date") or score.get("call_date") or "")
        rows.append({
            "name": f.name,
            "stem": stem,
            "registration": (crm.get("vehicle") or {}).get("registration_number") or stem.split("_")[0],
            "call_date": when[:10] or None,
            "call_time": when[11:16] or None,
            "cre_code": cre.get("username") or cre.get("employee_id") or score.get("cre_id") or None,
            "cre_name": cre.get("name") or score.get("cre_name") or None,
            "workshop": (crm.get("service_profile") or {}).get("workshop") or None,
            "has_crm": sidecar is not None or bool(crm) or get_crm_by_stem(stem) is not None,
            "has_excel": bool(crm),
            "has_transcript": _find_transcript(stem) is not None,
            "has_report": stem in scores,
            "score_pct": score.get("score_pct"),
            "grade": score.get("grade"),
            "call_type": _call_type_for_audio(f, crm or None),
            "lead_type": str(lead_type).upper() if lead_type else None,
            "transaction_id": str(meta[stem]["transaction_id"]) if meta[stem].get("transaction_id") else None,
        })
    # Recordings of the other lead type, or whose lead type isn't known, are not listed;
    # neither are calls the reviewer-dashboard admin removed (their files stay on disk).
    hidden = _removed.stems()
    return [row for row in rows if (not LEAD_TYPE or row["lead_type"] == LEAD_TYPE) and row["stem"] not in hidden]


@app.route("/api/calls")
def list_calls():
    return jsonify(_call_rows())


@app.route("/audited")
def audited_page():
    return render_template("audited_calls.html")


@app.route("/failed")
def failed_page():
    return render_template("failed_calls.html")


@app.route("/api/failed")
def failed_calls():
    """Calls whose transcription or audit failed (one day, or all dates), newest failure first."""
    day = None
    if request.args.get("date"):
        day = _day_arg()
        if day is None:
            return jsonify({"error": "date must be YYYY-MM-DD"}), 400
    try:
        failed = _failures.all_failed()
    except Exception:
        return jsonify({"error": "The failed list could not be read"}), 500
    keys = ("stem", "registration", "call_date", "call_time", "call_type", "cre_code", "cre_name", "workshop",
            "has_transcript", "has_report")
    items = [{**{key: row[key] for key in keys},
              **{key: failed[row["stem"]].get(key) for key in ("phase", "reason", "log", "failed_at", "attempts")}}
             for row in _call_rows() if row["stem"] in failed and (not day or row["call_date"] == day)]
    items.sort(key=lambda item: item["failed_at"] or 0, reverse=True)
    return jsonify({"items": items})


@app.route("/api/audited-calls")
def audited_calls():
    """Audited local calls (one day, or all dates), newest first."""
    day = None
    if request.args.get("date"):
        day = _day_arg()
        if day is None:
            return jsonify({"error": "date must be YYYY-MM-DD"}), 400
    keys = ("stem", "registration", "call_date", "call_time", "call_type", "lead_type", "cre_code", "cre_name",
            "workshop", "score_pct", "grade")
    try:
        reviewed = _reviews.latest()
    except Exception:  # an unreadable review file must not hide the audits
        reviewed = {}
    items = [{**{key: row[key] for key in keys},
              "review": {k: reviewed[row["stem"]][k] for k in ("status", "reviewer", "created_at")}
              if row["stem"] in reviewed else None}
             for row in _call_rows() if row["has_report"] and (not day or row["call_date"] == day)]
    items.sort(key=lambda item: (item["call_date"] or "", item["call_time"] or ""), reverse=True)
    return jsonify({"items": items})


def _day_arg():
    day = request.args.get("date", "")
    try:
        date.fromisoformat(day)
    except ValueError:
        return None
    return day


@app.route("/api/cre-performance")
def cre_performance():
    """Per-CRE results for one day's audited calls: calls, average score, call
    types, and the parameters where they lost the largest share of points."""
    day = _day_arg()
    if day is None:
        return jsonify({"error": "date must be YYYY-MM-DD"}), 400
    try:
        scores = read_all_scores(REPORTS)
    except Exception:
        scores = {}
    groups: dict = {}
    for row in _call_rows():
        if row["call_date"] != day or not row["has_report"]:
            continue
        key = row["cre_code"] or row["cre_name"] or "Unknown"
        group = groups.setdefault(key, {
            "cre_code": row["cre_code"], "cre_name": row["cre_name"], "calls": 0,
            "scores": [], "call_types": {}, "params": {}, "not_genuine": 0,
        })
        group["calls"] += 1
        if row["grade"] == "Not Genuine":
            group["not_genuine"] += 1
        group["call_types"][row["call_type"]] = group["call_types"].get(row["call_type"], 0) + 1
        if row["score_pct"] is not None:
            group["scores"].append(row["score_pct"])
        for pid, (points, weight) in ((scores.get(row["stem"]) or {}).get("params") or {}).items():
            earned = group["params"].setdefault(pid, [0.0, 0.0])
            earned[0] += points
            earned[1] += weight
    cres = []
    for group in groups.values():
        weakest = sorted(
            ((pid, round(100 * got / weight, 1)) for pid, (got, weight) in group["params"].items() if weight),
            key=lambda item: item[1],
        )
        cres.append({
            "cre_code": group["cre_code"],
            "cre_name": group["cre_name"],
            "calls": group["calls"],
            "avg_score": round(sum(group["scores"]) / len(group["scores"]), 1) if group["scores"] else None,
            "call_types": group["call_types"],
            "not_genuine": group["not_genuine"],
            "weakest": [{"param": pid, "pct": pct} for pid, pct in weakest if pct < 100][:3],
        })
    cres.sort(key=lambda c: (c["avg_score"] is None, c["avg_score"] or 0))
    return jsonify({"date": day, "cres": cres})


@app.route("/api/export")
def export_day():
    """The selected day's audits as an Excel file (same layout as the audit workbook)."""
    from io import BytesIO

    day = _day_arg()
    if day is None:
        return jsonify({"error": "date must be YYYY-MM-DD"}), 400
    stems = {row["stem"] for row in _call_rows() if row["call_date"] == day and row["has_report"]}
    if not stems:
        return jsonify({"error": f"No audited calls on {day}"}), 404
    data = export_calls(REPORTS, stems)
    if data is None:
        return jsonify({"error": "Audit workbook not found"}), 404
    suffix = f"-{LEAD_TYPE.lower()}" if LEAD_TYPE else ""
    return send_file(
        BytesIO(data), as_attachment=True, download_name=f"kataria-audits-{day}{suffix}.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.route("/api/fake-bookings")
def fake_bookings():
    """Booking calls judged not genuine (one day, or all dates), newest first, with
    the transcript turns the verdict cites and the customer's last three turns."""
    day = None
    if request.args.get("date"):
        day = _day_arg()
        if day is None:
            return jsonify({"error": "date must be YYYY-MM-DD"}), 400
    from core.audit.pipeline import parse_transcript, read_transcript

    items = []
    for row in _call_rows():
        if row["call_type"] != "BOOKING" or (day and row["call_date"] != day):
            continue
        booking = (read_call_evidence(REPORTS, row["stem"]) or {}).get("booking")
        if not isinstance(booking, dict):
            # Audited before evidence files existed: only the grade says it is fake.
            if row["grade"] != "Not Genuine":
                continue
            booking = {}
        elif booking.get("genuine") is not False:
            continue
        turns = []
        path = _find_transcript(row["stem"])
        if path is not None:
            try:
                turns = parse_transcript(read_transcript(path))
            except (OSError, ValueError):
                pass
        by_id = {f"T{turn.index:03d}": turn for turn in turns}
        customer = [turn for turn in turns if turn.speaker == "Customer"][-3:]
        items.append({
            **{key: row[key] for key in (
                "stem", "registration", "call_date", "call_time", "lead_type", "cre_code", "cre_name", "workshop",
            )},
            "reason": booking.get("reason"),
            "evidence": booking.get("evidence") or "",
            "confidence": booking.get("confidence"),
            "cited": [
                {"id": tid, "speaker": by_id[tid].speaker, "text": by_id[tid].text}
                for tid in booking.get("turns") or [] if tid in by_id
            ],
            "last_customer": [{"id": f"T{turn.index:03d}", "text": turn.text} for turn in customer],
        })
    items.sort(key=lambda item: (item["call_date"] or "", item["call_time"] or ""), reverse=True)
    return jsonify({"items": items})


@app.route("/api/crm/<path:stem>")
def get_crm(stem):
    excel_crm = None
    json_crm  = None
    call      = None

    # 1. Call data (CRM database / CALL DATA.xlsx) by registration number
    if _cd_get_crm is not None:
        try:
            call = _cd_get_crm(stem)
            if call:
                excel_crm = call.crm
        except Exception:
            pass

    # 2. JSON sidecar — normalize into stable schema; without one, the
    #    database's vehicle master stands in for it
    sidecar = _find_json(stem)
    if sidecar or (call is not None and call.vehicle is not None):
        try:
            from core.audit.pipeline import normalize_crm_details as _norm
            payload = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar else call.vehicle
            json_crm = _norm(payload)
            # Imported calls carry their own record: show what the audit will use.
            if isinstance(payload.get("call_data"), dict):
                excel_crm = payload["call_data"]
        except Exception:
            pass

    # 3. Merge and return
    if excel_crm is not None or json_crm is not None:
        try:
            from core.audit.crm_merger import merge_crm as _merge
            merged = _merge(excel_crm, json_crm)
            if merged:
                return jsonify(_crm_for_browser(merged))
        except Exception:
            pass

    # 4. Fallback: raw sidecar JSON
    if sidecar:
        try:
            raw_sidecar = json.loads(sidecar.read_text(encoding="utf-8"))
            return jsonify(_crm_for_browser(raw_sidecar))
        except Exception:
            return jsonify({"error": "CRM data could not be safely rendered"}), 500

    # 5. Legacy phone-based Excel
    details = get_crm_by_stem(stem)
    if details:
        return jsonify(_crm_for_browser(details))
    return jsonify({"error": "not found"}), 404


@app.route("/api/crm-debug")
def crm_debug():
    """Diagnostic endpoint — check Excel loader status."""
    info: dict = {}
    try:
        from core.audit.crm_excel import (
            CRM_EXCEL_PATH, list_loaded, get_crm_by_stem as _gs
        )
        info["excel_path"] = str(CRM_EXCEL_PATH)
        info["excel_exists"] = CRM_EXCEL_PATH.exists()
        try:
            import openpyxl
            info["openpyxl"] = openpyxl.__version__
        except ImportError:
            info["openpyxl"] = None
        loaded = list_loaded()
        info["rows_loaded"] = len(loaded)
        info["phones_masked"] = loaded
        # test each audio file stem
        stems_tested = {}
        for f in _iter_audio():
            if f.suffix.lower() in AUDIO_EXTS:
                r = _gs(f.stem)
                stems_tested[_mask(f.stem)] = "MATCH" if r else "no match"
        info["stems"] = stems_tested
    except Exception as exc:
        info["error"] = str(exc)
    return jsonify(info)


# ── CRM database (crmdb): calls, call data, vehicle master ────────────────────

def _call_db_unavailable():
    if _crm_db is None or not _crm_db.enabled():
        return jsonify({"error": "CRM database not configured: set DB_* in .env"}), 503
    return None


def _call_db_failed(exc: Exception):
    # Driver detail stays in the server log; /api/db/health reports connection problems.
    app.logger.warning("CRM database query failed: %s", _db.describe_error(exc))
    return jsonify({"error": "CRM database query failed"}), 502


def _local_status() -> dict:
    """stem → dashboard row for every recording in data/ (with transcript / audit status)."""
    return {row["stem"]: row for row in _call_rows()}


def _failed_stems() -> set:
    try:
        return set(_failures.all_failed())
    except Exception:  # an unreadable failed list must not break the day list
        return set()


def _local_by_transaction() -> dict:
    """CRM transaction id → dashboard row of a recording already in data/ (imported calls)."""
    return {row["transaction_id"]: row for row in _call_rows() if row.get("transaction_id")}


def _with_local_status(call: dict, local: dict) -> dict:
    """A CRM call plus whether this dashboard has it downloaded, transcribed and audited.

    Imported recordings are named <REG>_<CDR> (one file per call); a call without
    a CDR id can only match the hand-made set's registration-named files.
    """
    from core.audit.importer import _stem

    reg = normalize_reg(call.get("registration_number"))
    stem = _stem(reg, call["cdr_id"]) if call.get("cdr_id") else reg
    row = local.get(stem) or {}
    return {
        **call,
        "stem": stem,
        "has_audio": bool(row),
        "has_transcript": bool(row.get("has_transcript")),
        "has_report": bool(row.get("has_report")),
        "score_pct": row.get("score_pct"),
        "grade": row.get("grade"),
    }


@app.route("/api/db/health")
def db_health():
    """Connection check for the CRM database (no credentials in the response)."""
    info = _db.health()
    info.update(crm_source=CRM_SOURCE, crm_from_db=bool(_crm_db is not None and _crm_db.enabled()))
    return jsonify(info), (503 if info["configured"] and not info["connected"] else 200)


@app.route("/api/db/calls")
def db_calls():
    """CRE calls from the CRM database, newest first, with local audio/transcript/report status.

    Query params: call_type (one of the four audit call types, default all four;
    "all" also lists no-contact attempts and status changes), cre (employee code
    or name), station (workshop name or code), registration (partial),
    date_from / date_to (YYYY-MM-DD, call date, inclusive), has_recording
    (true/false), limit (default 50, max 500), offset.
    Page with offset while has_more is true.
    """
    unavailable = _call_db_unavailable()
    if unavailable:
        return unavailable
    args = request.args
    try:
        call_type = None
        if args.get("call_type", "").strip().lower() == "all":
            call_type = "ALL"
        elif args.get("call_type"):
            call_type = _normalise_call_type(args["call_type"])
            if call_type == "UNKNOWN":
                raise ValueError(f"call_type must be one of: {', '.join(_crm_db.CALL_TYPES)}, all")
        date_from = date.fromisoformat(args["date_from"]) if args.get("date_from") else None
        date_to = date.fromisoformat(args["date_to"]) if args.get("date_to") else None
        has_recording = None
        if args.get("has_recording"):
            flag = args["has_recording"].strip().lower()
            if flag not in {"true", "false", "1", "0", "yes", "no"}:
                raise ValueError("has_recording must be true or false")
            has_recording = flag in {"true", "1", "yes"}
        limit = min(max(int(args.get("limit", 50)), 1), 500)
        offset = max(int(args.get("offset", 0)), 0)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        calls, has_more = _crm_db.list_calls(
            call_type=call_type, cre=args.get("cre"), station=args.get("station"), lead_type=LEAD_TYPE,
            registration=args.get("registration"), date_from=date_from, date_to=date_to,
            has_recording=has_recording, limit=limit, offset=offset,
        )
    except Exception as exc:
        return _call_db_failed(exc)
    local = _local_status()
    return jsonify({
        "limit": limit, "offset": offset, "has_more": has_more,
        "items": [_with_local_status(call, local) for call in calls],
    })


@app.route("/api/db/calls/<transaction_id>")
def db_call(transaction_id):
    """Everything the audit needs for one call (by lead_transaction id): call data,
    vehicle master and service history, merged as the audit sees it (browser-safe)."""
    unavailable = _call_db_unavailable()
    if unavailable:
        return unavailable
    try:
        call = _crm_db.get_call(transaction_id)
    except ValueError:
        return jsonify({"error": "transaction_id must be a UUID"}), 400
    except Exception as exc:
        return _call_db_failed(exc)
    if call is None:
        return jsonify({"error": "not found"}), 404
    summary = _with_local_status(
        {"registration_number": call.crm["vehicle"]["registration_number"], "cdr_id": call.cdr_id},
        _local_status(),
    )
    return jsonify({
        **summary,
        "transaction_id": call.transaction_id,
        "cdr_id": call.cdr_id,
        "call_type": call.crm.get("audit_call_type") or "UNKNOWN",
        "has_vehicle_master": call.vehicle is not None,
        "crm_details": _crm_for_browser(_crm_db.crm_details(call)),
    })


@app.route("/api/db/run", methods=["POST"])
def db_run():
    """Pick calls from the CRM database, import them (recording + call data) and
    transcribe + audit them. No ids to supply.

    JSON body, all optional: date ("yesterday" by default, "today" or YYYY-MM-DD)
    or date_from / date_to; call_type; cre; station; limit (default 20, max 200);
    overwrite (re-download calls already in data/); reaudit (audit again calls
    that already have a report).
    Returns a job id: /api/job/<id> shows picking and importing, then the
    batch_id whose transcription and audit appear on /batch. Calls that already
    have an audit report are not audited again.
    """
    unavailable = _call_db_unavailable()
    if unavailable:
        return unavailable
    from core.audit import importer as _importer

    body = request.json or {}
    try:
        if body.get("date_from"):
            day_from = _importer.parse_day(body["date_from"])
            day_to = _importer.parse_day(body.get("date_to") or body["date_from"])
        else:
            day_from = day_to = _importer.parse_day(body.get("date") or "yesterday")
        if day_to < day_from:
            raise ValueError("date_to is before date_from")
        call_type = None
        if body.get("call_type"):
            call_type = _normalise_call_type(body["call_type"])
            if call_type == "UNKNOWN":
                raise ValueError(f"call_type must be one of: {', '.join(_crm_db.CALL_TYPES)}")
        limit = min(max(int(body.get("limit", 20)), 1), 200)
        # Calls ticked in the dashboard: import exactly these instead of picking by date.
        # A whole day can be ticked at once ("select all"); the dashboard confirms big runs.
        chosen = body.get("transaction_ids")
        if chosen is not None and (not isinstance(chosen, list) or not 1 <= len(chosen) <= MAX_RUN_CALLS
                                   or not all(isinstance(item, str) for item in chosen)):
            raise ValueError(f"transaction_ids must be a list of 1-{MAX_RUN_CALLS} ids")
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    overwrite = bool(body.get("overwrite"))
    reaudit = bool(body.get("reaudit"))

    job_id = _new_job("running")

    def update(**fields):
        with _JOBS_LOCK:
            JOBS[job_id].update(fields)

    update(started_at=time.time(), progress="picking calls from the CRM database",
           selection={"date_from": day_from.isoformat(), "date_to": day_to.isoformat(),
                      "call_type": call_type, "lead_type": LEAD_TYPE, "limit": limit,
                      "overwrite": overwrite, "reaudit": reaudit})

    def work():
        try:
            if chosen is not None:
                ids = list(dict.fromkeys(chosen))
            else:
                calls = _importer.find_calls(day_from=day_from, day_to=day_to, call_type=call_type,
                                             cre=body.get("cre"), station=body.get("station"), limit=limit,
                                             skip_existing=not overwrite, lead_type=LEAD_TYPE)
                ids = [call["transaction_id"] for call in calls]
            # Calls already downloaded aren't looked up or downloaded again: they go
            # straight to the queue (unless overwrite asks for a fresh copy).
            have = {} if overwrite else _local_by_transaction()
            results = [{"transaction_id": tid, "stem": have[tid]["stem"], "status": "skipped"}
                       for tid in ids if tid in have]
            todo = [tid for tid in ids if tid not in have]
            update(found=len(ids), progress=f"importing 0/{len(todo)} ({len(results)} already downloaded)")
            results += _importer.import_calls(
                todo, overwrite,
                on_result=lambda done: update(
                    progress=f"importing {len(done)}/{len(todo)} ({len(ids) - len(todo)} already downloaded)"),
            ) if todo else []
            _invalidate_call_rows()
            try:
                audited = list_audited_call_ids(REPORTS)
            except Exception:
                audited = set()
            stems = [r["stem"] for r in results
                     if r["status"] in ("imported", "skipped") and (reaudit or r["stem"] not in audited)]
            batch_id = _start_batch(stems, "both") if stems else None
            counts = {s: sum(r["status"] == s for r in results) for s in ("imported", "skipped", "error")}
            update(
                status="done", batch_id=batch_id, runtime_s=round(time.time() - JOBS[job_id]["started_at"], 1),
                progress=f"{counts['imported']} imported, {counts['skipped']} already there, "
                         f"{counts['error']} failed; {len(stems)} queued for transcription and audit",
                result={**counts, "queued_for_audit": len(stems), "timing": _importer.timing_summary(results),
                        "errors": [r for r in results if r["status"] == "error"][:50]},
            )
        except Exception as exc:
            update(status="error", error=_db.describe_error(exc))

    threading.Thread(target=work, daemon=True).start()
    return jsonify({"job_id": job_id}), 202


@app.route("/api/day")
def day_calls():
    """One day's connected calls from the CRM database, for the dashboard list.

    Query: date (YYYY-MM-DD, default yesterday); cre_id (a CRE's user id, or
    "none") and/or call_type to get one group's page (offset, limit <= 200),
    otherwise counts only; cre, station, registration; status = audited |
    not_audited | hi | mid | lo (this dashboard's audits). Connected = bizRTC
    recorded the call as ANSWERED. Returns per-type counts and per-CRE counts
    (by CRE name), the requested page with each call's local status, and this
    day's local recordings that aren't CRM calls (the hand-made set).
    """
    from core.audit.importer import parse_day

    args = request.args
    try:
        day = parse_day(args.get("date") or "yesterday")
        call_type = None
        if args.get("call_type"):
            call_type = _normalise_call_type(args["call_type"])
            if call_type == "UNKNOWN":
                raise ValueError("call_type must be one of: " + ", ".join(_crm_db.CALL_TYPES if _crm_db else ()))
        cre_id = (args.get("cre_id") or "").strip() or None
        if cre_id and cre_id != "none":
            try:
                cre_id = str(uuid.UUID(cre_id))
            except ValueError:
                raise ValueError("cre_id must be a CRE user id or none") from None
        limit = min(max(int(args.get("limit", 100)), 1), 200)
        offset = max(int(args.get("offset", 0)), 0)
        status = args.get("status") or ""
        if status not in ("", "audited", "not_audited", "hi", "mid", "lo"):
            raise ValueError("status must be audited, not_audited, hi, mid or lo")
        # Call length bounds in seconds (bizRTC's call duration): min inclusive, max exclusive.
        seconds = {}
        for key in ("min_seconds", "max_seconds"):
            if args.get(key, "").strip():
                value = int(args[key])
                if not 0 <= value <= 86400:
                    raise ValueError(f"{key} must be between 0 and 86400")
                seconds[key] = value
        disposition = (args.get("disposition") or "").strip().upper() or None
        if disposition and not re.fullmatch(r"[A-Z_]{1,40}", disposition):
            raise ValueError("disposition must be a CRM status such as BOOKED_CONFIRMED")
        campaign = (args.get("campaign") or "").strip() or None
        if campaign and campaign != "none":
            try:
                campaign = str(uuid.UUID(campaign))
            except ValueError:
                raise ValueError("campaign must be a campaign id or none") from None
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    rows = _call_rows()
    local = {row["stem"]: row for row in rows}
    day_iso = day.isoformat()
    pinned = {row["stem"]: _crm_db.parse_stem(row["stem"])[1] if _crm_db else None for row in rows}
    local_only = [row for row in rows if row["call_date"] == day_iso and not pinned[row["stem"]]]
    if _crm_db is None or not _crm_db.enabled():
        return jsonify({"date": day_iso, "db": False, "lead_type": LEAD_TYPE, "counts": {}, "cres": [],
                        "items": [], "has_more": False,
                        "local_only": local_only})

    audited = {pinned[row["stem"]]: row for row in rows if row["has_report"] and pinned[row["stem"]]}
    # Calls whose audit failed are listed only on the Failed page, never as "not audited".
    failed = {cdr for cdr in (_crm_db.parse_stem(stem)[1] for stem in _failed_stems()) if cdr} - set(audited)
    # Calls the admin removed are hidden here too, so they aren't picked up for audit again.
    removed_cdrs = {cdr for cdr in (_crm_db.parse_stem(stem)[1] for stem in _removed.stems()) if cdr}
    filters = {
        "date_from": day, "date_to": day, "connected": True,
        "cre": args.get("cre") or None, "station": args.get("station") or None,
        "registration": args.get("registration") or None, "lead_type": LEAD_TYPE, **seconds,
        "campaign": campaign, "disposition": disposition,
    }
    if status == "audited":
        filters["cdr_in"] = sorted(audited)
    elif status == "not_audited":
        filters["cdr_not_in"] = sorted(set(audited) | failed | removed_cdrs)
    elif status:
        bucket = {"hi": lambda p: p >= 75, "mid": lambda p: 60 <= p < 75, "lo": lambda p: p < 60}[status]
        filters["cdr_in"] = sorted(cdr for cdr, row in audited.items()
                                   if row["score_pct"] is not None and bucket(row["score_pct"]))
    try:
        # "Select all": every matching call's id, optionally one call type or CRE.
        if args.get("ids_only"):
            ids = _crm_db.list_call_ids(call_type=call_type, cre_id=cre_id, **filters)
            return jsonify({"date": day_iso, "ids": ids})
        # A group's page doesn't re-count the day: the dashboard asks for counts once, alongside.
        counts, cres, campaigns, dispositions = {}, [], [], []
        if call_type or cre_id:
            items, has_more = _crm_db.list_calls(call_type=call_type, cre_id=cre_id,
                                                 limit=limit, offset=offset, **filters)
        else:
            cres = _crm_db.count_calls_by_cre(**filters)
            for cre in cres:
                for name, n in cre["counts"].items():
                    counts[name] = counts.get(name, 0) + n
            # The campaign list ignores the chosen campaign, so the other options stay listed.
            campaigns = _crm_db.count_calls_by_campaign(**{**filters, "campaign": None})
            dispositions = _crm_db.count_calls_by_disposition(**{**filters, "disposition": None})
            items, has_more = [], False
    except Exception as exc:
        return _call_db_failed(exc)
    failed_today = sum(1 for row in rows if row["call_date"] == day_iso and pinned[row["stem"]] in failed)
    return jsonify({
        "date": day_iso, "db": True, "lead_type": LEAD_TYPE, "counts": counts, "cres": cres, "call_type": call_type, "cre_id": cre_id,
        "offset": offset, "failed": failed_today, "campaigns": campaigns, "campaign": campaign,
        "dispositions": dispositions, "disposition": disposition,
        "has_more": has_more, "items": [_with_local_status(item, local) for item in items],
        "local_only": local_only,
    })


@app.route("/api/db/recording", methods=["POST"])
def db_recording():
    """Download one call's recording on demand (when Play is pressed): CRM decrypt
    API, then bizRTC. Body: {"transaction_id": "..."}. Returns the local stem."""
    unavailable = _call_db_unavailable()
    if unavailable:
        return unavailable
    from core.audit import importer as _importer

    transaction_id = str((request.json or {}).get("transaction_id") or "").strip()
    if not transaction_id:
        return jsonify({"error": "transaction_id is required"}), 400
    have = _local_by_transaction().get(transaction_id)
    if have:  # already downloaded: no CRM lookup, no download
        return jsonify({"transaction_id": transaction_id, "stem": have["stem"], "status": "skipped"})
    result = _importer.import_call(transaction_id)
    _invalidate_call_rows()
    if result.get("status") in ("imported", "skipped"):
        return jsonify(result)
    return jsonify({**result, "error": result.get("error") or "Recording could not be downloaded"}), 502


@app.route("/api/db/vehicles/<path:registration>/calls")
def db_vehicle_calls(registration):
    """Every dispositioned call on record for one vehicle, newest first."""
    unavailable = _call_db_unavailable()
    if unavailable:
        return unavailable
    try:
        calls = _crm_db.vehicle_calls(registration)
    except Exception as exc:
        return _call_db_failed(exc)
    if not calls:
        return jsonify({"error": "not found"}), 404
    local = _local_status()
    return jsonify({
        "registration_number": normalize_reg(registration),
        "calls": [_with_local_status(call, local) for call in calls],
    })


@app.route("/api/transcript/<path:stem>")
def get_transcript(stem):
    p = _find_transcript(stem)
    if p is None:
        return jsonify({"error": "not found"}), 404
    from core.audit.pipeline import read_transcript
    try:
        return jsonify({"text": read_transcript(p)})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422


@app.route("/api/audio/<path:stem>")
def get_audio(stem):
    """Call recording for the player; conditional=True answers Range requests so it can seek."""
    path = _find_audio(stem)
    if path is None:
        return jsonify({"error": "audio file not found"}), 404
    return send_file(path, conditional=True)


@app.route("/api/report/<path:stem>")
def get_report(stem):
    report = read_audit_scores(REPORTS, stem)
    if report is None:
        return jsonify({"error": "not found"}), 404
    evidence = read_call_evidence(REPORTS, stem)
    if evidence:
        report["evidence"] = evidence
    # Human reviews from the reviewer dashboard: shown beside the AI score, never replacing it.
    try:
        versions = _reviews.history(stem)
    except Exception:  # an unreadable review file must not hide the audit
        versions = []
    if versions:
        latest = _reviews.with_score(versions[0], _reviews.report_inputs(report))  # older reviews: score worked out now
        report["human_review"] = {**latest, "versions": len(versions)}
    return jsonify(report)


@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    stem = (request.json or {}).get("stem", "")
    audio_path = _find_audio(stem)
    if not audio_path:
        return jsonify({"error": "audio file not found"}), 404
    audio = str(audio_path)

    stem_key = f"tx:{stem}"
    with _ACTIVE_LOCK:
        existing = _ACTIVE.get(stem_key)
        if existing:
            with _JOBS_LOCK:
                st = JOBS.get(existing, {}).get("status")
            if st in ("queued", "running"):
                return jsonify({"job_id": existing, "reused": True})

    TRANSCRIPTS.mkdir(parents=True, exist_ok=True)
    transcript_output = str(TRANSCRIPTS / f"{stem}.transcript.json")
    job_id = _new_job()
    with _ACTIVE_LOCK:
        _ACTIVE[stem_key] = job_id

    if _use_engine():
        with _JOBS_LOCK:
            JOBS[job_id]["status"] = "running"
            JOBS[job_id]["engine"] = True
            JOBS[job_id]["started_at"] = time.time()
        get_engine().tx.submit(
            job_id, audio, transcript_output, _done_cb(job_id, stem_key), _progress_cb(job_id)
        )
    else:
        cmd = [PYTHON, TRANSCRIBER, audio, transcript_output]
        threading.Thread(target=_run, args=(job_id, cmd, stem_key), daemon=True).start()

    return jsonify({"job_id": job_id})


@app.route("/api/audit", methods=["POST"])
def audit():
    stem = (request.json or {}).get("stem", "")
    transcript = _find_transcript(stem)
    if transcript is None:
        return jsonify({"error": "transcript not found — run transcription first"}), 404

    stem_key = f"au:{stem}"
    with _ACTIVE_LOCK:
        existing = _ACTIVE.get(stem_key)
        if existing:
            with _JOBS_LOCK:
                st = JOBS.get(existing, {}).get("status")
            if st in ("queued", "running"):
                return jsonify({"job_id": existing, "reused": True})

    crm_json = _find_json(stem)
    crm_path = str(crm_json) if crm_json else None

    REPORTS.mkdir(parents=True, exist_ok=True)
    body = request.json or {}
    selected_model = OLLAMA_MODEL
    skip_classify = body.get("skip_classify", False)
    skip_names    = body.get("skip_names",    False)
    skip_scoring  = body.get("skip_scoring",  False)
    skip_coaching = body.get("skip_coaching", False)

    job_id = _new_job()
    with _JOBS_LOCK:
        JOBS[job_id]["model"] = selected_model
    with _ACTIVE_LOCK:
        _ACTIVE[stem_key] = job_id

    if _use_engine():
        with _JOBS_LOCK:
            JOBS[job_id]["status"] = "running"
            JOBS[job_id]["engine"] = True
            JOBS[job_id]["started_at"] = time.time()
        get_engine().au.submit(
            job_id, str(transcript), str(REPORTS), crm_path,
            skip_classify, skip_names, skip_scoring, skip_coaching,
            _done_cb(job_id, stem_key), _progress_cb(job_id), selected_model,
        )
    else:
        # The fallback uses the same validated single-pass implementation as
        # the persistent worker. The legacy multi-pass auditor is never used.
        cmd = [PYTHON, "-m", AUDITOR_MODULE, "--input", str(transcript),
               "--out", str(REPORTS), "--model", selected_model]
        if crm_path:
            cmd += ["--crm-snapshot", crm_path]
        if skip_classify:
            cmd.append("--skip-classify")
        if skip_names:
            cmd.append("--skip-names")
        if skip_scoring:
            cmd.append("--skip-scoring")
        if skip_coaching:
            cmd.append("--skip-coaching")
        threading.Thread(target=_run, args=(job_id, cmd, stem_key), daemon=True).start()

    return jsonify({"job_id": job_id})


@app.route("/api/job/<job_id>")
def job_status(job_id):
    with _JOBS_LOCK:
        job = dict(JOBS.get(job_id, {"status": "not_found"}))
    out = {k: v for k, v in job.items() if k != "log"}
    return jsonify(out)


@app.route("/api/cuda")
def cuda_status():
    try:
        import torch
        available = torch.cuda.is_available()
        device = torch.cuda.get_device_name(0) if available else None
        return jsonify({"cuda": available, "device": device})
    except ImportError:
        return jsonify({"cuda": False, "device": None})


# ── Engine control ────────────────────────────────────────────────────────────

# The engine starts with `python wsgi.py` and stops with it (Ctrl+C); jobs start it
# too when the app runs another way (_use_engine).
@app.route("/api/engine/status")
def engine_status():
    if not _ENGINE_AVAILABLE:
        return jsonify({"status": "unavailable"})
    return jsonify(get_engine().state_dict())


@app.route("/api/emergency-stop", methods=["POST"])
def emergency_stop():
    """Hard-stop Kataria work, user-owned Ollama, and this dashboard process."""
    # The DGX itself or a machine on the office network (the dashboard is opened over the LAN).
    try:
        client = ipaddress.ip_address(request.remote_addr or "")
    except ValueError:
        client = None
    if client is None or not (client.is_loopback or client.is_private):
        return jsonify({"error": "Emergency stop is allowed only from the DGX or the office network."}), 403
    if (request.json or {}).get("confirm") != "STOP_KATARIA_AND_OLLAMA":
        return jsonify({"error": "explicit emergency-stop confirmation is required"}), 400

    child_count = _terminate_child_processes()
    ollama_stopped, ollama_error = _terminate_owned_ollama()

    with _JOBS_LOCK:
        for job in JOBS.values():
            if job.get("status") in {"queued", "running"}:
                job.update(status="cancelled", error="Emergency stop requested.")
    with _BATCHES_LOCK:
        for batch in BATCHES.values():
            if batch.get("status") == "running":
                batch["status"] = "cancelled"
                batch["updated_at"] = time.time()
                batch["revision"] = batch.get("revision", 0) + 1
                for item in batch.get("items", []):
                    if item.get("status") in {"queued", "running"}:
                        item.update(
                            status="cancelled", phase="cancelled",
                            job_status="cancelled", progress="Emergency stop requested",
                        )

    _schedule_server_stop()
    return jsonify({
        "ok": True,
        "dashboard_stopping": True,
        "child_processes_stopped": child_count,
        "ollama_stopped": ollama_stopped,
        "warning": ollama_error or None,
    })


# ── Batch processing ──────────────────────────────────────────────────────────

def _batch_update_item(batch_id: str, stem: str, **updates) -> None:
    with _BATCHES_LOCK:
        batch = BATCHES.get(batch_id)
        if not batch:
            return
        item_index = batch.get("_item_index", {}).get(stem)
        if item_index is not None:
            batch["items"][item_index].update(updates)
            batch["revision"] = batch.get("revision", 0) + 1
            batch["updated_at"] = time.time()
            total_steps = batch.get("total_steps", 0)
            active_fraction = sum(
                float(item.get("phase_percent", 0)) / 100
                for item in batch.get("items", [])
                if item.get("status") == "running"
            )
            if total_steps:
                progress_steps = min(total_steps, batch.get("completed_steps", 0) + active_fraction)
                batch["percent"] = round(100 * progress_steps / total_steps)


def _wait_batch_job(batch_id: str, stem: str, job_id: str, phase: str) -> tuple[str, str]:
    """Mirror one job's live status into its batch item."""
    started = time.time()
    while time.time() - started < 7200:
        with _JOBS_LOCK:
            job = dict(JOBS.get(job_id, {"status": "not_found"}))
        status = job.get("status", "not_found")
        progress = _mask(str(job.get("progress") or phase.title()))
        phase_percent = max(0, min(100, int(job.get("phase_percent") or 0)))
        _batch_update_item(
            batch_id, stem, phase=phase, job_status=status,
            progress=progress, phase_percent=phase_percent,
            elapsed_s=round(time.time() - started, 1),
        )
        if status in {"done", "error", "not_found"}:
            return status, _mask(str(job.get("error") or ""))
        time.sleep(0.75)
    return "error", "Timed out while processing this recording."


def _complete_batch_step(batch_id: str, stem: str) -> None:
    with _BATCHES_LOCK:
        batch = BATCHES.get(batch_id)
        if not batch:
            return
        batch["completed_steps"] = min(
            batch.get("total_steps", 0), batch.get("completed_steps", 0) + 1
        )
        item_index = batch.get("_item_index", {}).get(stem)
        if item_index is not None:
            item = batch["items"][item_index]
            item["phase_percent"] = 0
            item["completed_steps"] = min(
                batch.get("phase_count", 1), item.get("completed_steps", 0) + 1
            )
        total_steps = batch.get("total_steps", 0)
        batch["percent"] = round(100 * batch["completed_steps"] / total_steps) if total_steps else 0
        batch["revision"] = batch.get("revision", 0) + 1
        batch["updated_at"] = time.time()


def _finish_batch_item(batch_id: str, stem: str, successful: bool, error: str, started: float) -> None:
    """Finalize one item once and account for skipped phases after a failure."""
    _invalidate_call_rows()
    with _BATCHES_LOCK:
        batch = BATCHES.get(batch_id)
        if not batch:
            return
        item_index = batch.get("_item_index", {}).get(stem)
        if item_index is None:
            return
        item = batch["items"][item_index]
        if item.get("status") in {"completed", "failed"}:
            return

        phase_count = batch.get("phase_count", 1)
        missing_steps = max(0, phase_count - item.get("completed_steps", 0))
        batch["completed_steps"] = min(
            batch.get("total_steps", 0), batch.get("completed_steps", 0) + missing_steps
        )
        item.update({
            "status": "completed" if successful else "failed",
            "phase": "complete" if successful else "failed",
            "job_status": "done" if successful else "error",
            "progress": "Completed" if successful else (error or "Failed"),
            "phase_percent": 0,
            "completed_steps": phase_count,
            "elapsed_s": round(time.time() - started, 1),
            "error": error if not successful else "",
            "has_transcript": _find_transcript(stem) is not None,
            "has_report": bool(item.get("has_report")),
        })
        batch["done"] += 1
        batch["successful" if successful else "failed"] += 1
        batch["revision"] = batch.get("revision", 0) + 1
        batch["updated_at"] = time.time()
        total_steps = batch.get("total_steps", 0)
        batch["percent"] = (
            round(100 * batch["completed_steps"] / total_steps) if total_steps else 0
        )
        if batch["done"] >= batch.get("total", 0):
            batch["status"] = "done"
            batch["percent"] = 100
            batch["finished_at"] = time.time()


def _run_batch_phase(
    batch_id: str, stem: str, phase: str, model: str,
) -> tuple[bool, str]:
    _batch_update_item(
        batch_id, stem, status="running", phase=phase,
        job_status="starting", progress=f"Starting {phase}…",
        phase_percent=0, elapsed_s=0,
    )
    endpoint = "/api/transcribe" if phase == "transcribing" else "/api/audit"
    payload = {"stem": stem}
    if phase == "auditing":
        payload.update({
            "skip_classify": True, "skip_names": False,
            "skip_scoring": False, "skip_coaching": True,
            "model": model,
        })
    response = app.test_client().post(endpoint, json=payload, content_type="application/json")
    data = response.get_json() or {}
    job_id = data.get("job_id")
    if not job_id:
        error = _mask(str(data.get("error") or f"Could not start {phase}."))
        _batch_update_item(batch_id, stem, job_status="error", progress=error)
        _complete_batch_step(batch_id, stem)
        return False, error

    status, error = _wait_batch_job(batch_id, stem, job_id, phase)
    _complete_batch_step(batch_id, stem)
    if status != "done":
        return False, error or f"{phase.title()} failed."
    with _JOBS_LOCK:
        runtime_s = JOBS.get(job_id, {}).get("runtime_s")
    artifact_updates = {"has_transcript": _find_transcript(stem) is not None}
    if phase == "auditing":
        artifact_updates.update(has_report=True, runtime_s=runtime_s, **_batch_result(stem))
    else:
        artifact_updates["tx_runtime_s"] = runtime_s
    _batch_update_item(batch_id, stem, **artifact_updates)
    return True, ""


def _batch_result(stem: str) -> dict:
    """Score summary for the batch page, read back from the saved Excel row."""
    try:
        report = read_audit_scores(REPORTS, stem)
    except Exception:  # a report read failure must not fail the batch item
        report = None
    if not report:
        return {}
    summary = report.get("management_summary") or {}
    booking = (report.get("audit_result") or {}).get("booking_genuine") or {}
    flag = (
        "Wrong call tagging" if summary.get("wrong_call_tagging")
        else "Fake booking" if booking.get("genuine") is False
        else ""
    )
    return {
        "call_type": summary.get("call_type"),
        "score_display": summary.get("score_display"),
        "score_pct": summary.get("score_pct"),
        "grade": summary.get("grade"),
        "flag": flag,
    }


def _batch_stem(batch_id: str, stem: str, mode: str, model: str):
    """Run requested phases for one recording and maintain live batch state."""
    started = time.time()
    successful = True
    error = ""
    if mode == "both" and _find_transcript(stem) is not None:
        # Already transcribed: straight to the audit.
        _complete_batch_step(batch_id, stem)
        _batch_update_item(batch_id, stem, has_transcript=True, tx_runtime_s=0)
    elif mode in ("transcribe", "both"):
        successful, error = _run_batch_phase(batch_id, stem, "transcribing", model)
    if successful and mode in ("audit", "both"):
        successful, error = _run_batch_phase(batch_id, stem, "auditing", model)
    _finish_batch_item(batch_id, stem, successful, error, started)


@app.route("/api/batch", methods=["POST"])
def batch_run():
    body = request.json or {}
    stems = list(dict.fromkeys(body.get("stems", [])))
    mode = body.get("mode", "both")
    if mode not in {"transcribe", "audit", "both"}:
        return jsonify({"error": "invalid batch mode"}), 400

    if not stems:
        return jsonify({"error": "no stems provided"}), 400
    return jsonify({"batch_id": _start_batch(stems, mode)})


def _start_batch(stems: list, mode: str) -> str:
    """Queue recordings for transcription and/or audit, one at a time; returns the batch id."""
    selected_model = OLLAMA_MODEL

    batch_id = str(uuid.uuid4())
    phase_count = 2 if mode == "both" else 1
    items = [
        {
            "id": index, "name": _mask(stem), "status": "queued",
            "phase": "queued", "job_status": "queued", "progress": "Waiting",
            "phase_percent": 0, "elapsed_s": 0, "error": "",
            "completed_steps": 0, "has_transcript": False, "has_report": False,
        }
        for index, stem in enumerate(stems)
    ]
    with _BATCHES_LOCK:
        if len(BATCHES) >= MAX_RETAINED_BATCHES:
            removable = [
                key for key, value in BATCHES.items()
                if value.get("status") != "running"
            ]
            for key in removable[: max(1, len(BATCHES) - MAX_RETAINED_BATCHES + 1)]:
                BATCHES.pop(key, None)
        BATCHES[batch_id] = {
            "status": "running",
            "revision": 0,
            "started_at": time.time(),
            "updated_at": time.time(),
            "total": len(stems),
            "done": 0,
            "successful": 0,
            "failed": 0,
            "percent": 0,
            "completed_steps": 0,
            "total_steps": len(stems) * phase_count,
            "phase_count": phase_count,
            "items": items,
            "_item_index": {stem: index for index, stem in enumerate(stems)},
            "mode": mode,
            "model": selected_model,
        }

    def _run_batch():
        # One recording at a time, in the order selected, until Stop is pressed.
        for stem in stems:
            with _BATCHES_LOCK:
                if BATCHES.get(batch_id, {}).get("stopping"):
                    break
            try:
                _batch_stem(batch_id, stem, mode, selected_model)
            except Exception as exc:
                _finish_batch_item(
                    batch_id, stem, False,
                    _mask(str(exc)) or "Unexpected batch error",
                    time.time(),
                )
        _close_stopped_batch(batch_id)

    threading.Thread(target=_run_batch, daemon=True).start()
    return batch_id


def _close_stopped_batch(batch_id: str) -> None:
    """After Stop: the calls still waiting are cancelled and the batch ends."""
    with _BATCHES_LOCK:
        batch = BATCHES.get(batch_id)
        if not batch or not batch.get("stopping") or batch.get("status") != "running":
            return
        for item in batch.get("items", []):
            if item.get("status") == "queued":
                item.update(status="cancelled", phase="cancelled", job_status="cancelled",
                            progress="Stopped before it started")
        batch["cancelled"] = sum(item.get("status") == "cancelled" for item in batch.get("items", []))
        batch.update(status="cancelled", finished_at=time.time(), updated_at=time.time(),
                     revision=batch.get("revision", 0) + 1)
    _invalidate_call_rows()


@app.route("/api/batch/<batch_id>/stop", methods=["POST"])
def batch_stop(batch_id):
    """Stop a batch: the call in progress finishes (its transcript and audit stay
    whole), the calls still waiting are not run."""
    with _BATCHES_LOCK:
        batch = BATCHES.get(batch_id)
        if batch is None:
            return jsonify({"error": "batch not found"}), 404
        if batch.get("status") != "running":
            return jsonify({"ok": True, "status": batch.get("status")})
        # The batch's own loop sees this before its next call and ends the batch.
        batch["stopping"] = True
        batch["revision"] = batch.get("revision", 0) + 1
    return jsonify({"ok": True, "status": "stopping"})


@app.route("/api/batch/latest")
def batch_latest():
    """Most recent batch, so the batch page and dashboard can find it after a reload."""
    with _BATCHES_LOCK:
        if not BATCHES:
            return jsonify({"status": "none"})
        batch_id, batch = max(BATCHES.items(), key=lambda kv: kv[1].get("started_at", 0))
        return jsonify({
            "batch_id": batch_id, "status": batch.get("status"), "stopping": bool(batch.get("stopping")),
            "done": batch.get("done", 0), "total": batch.get("total", 0),
        })


@app.route("/api/batch/<batch_id>")
def batch_status(batch_id):
    with _BATCHES_LOCK:
        source = BATCHES.get(batch_id)
        if source is None:
            return jsonify({"status": "not_found"})
        batch = copy.deepcopy(source)
        batch.pop("_item_index", None)
    batch["elapsed_s"] = round((batch.get("finished_at") or time.time()) - batch["started_at"], 1)
    return jsonify(batch)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
