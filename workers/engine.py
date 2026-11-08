"""
engine.py — persistent model worker for Kataria AI Audit
Loads pyannote, NeMo ASR, and keeps qwen3:32b resident in Ollama.
Import with get_engine(); call engine.start() / engine.stop().
"""
from __future__ import annotations

import importlib.util
import io
import queue
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, Optional


from config import BASE  # noqa: F401

_TX_MODULE_PATH = str(Path(__file__).resolve().parent.parent / "core" / "transcriber.py")

# ── thread-safe stdout capture ────────────────────────────────────────────────
_CAPTURE_LOCK = threading.Lock()


def _capture(fn: Callable, log_cb: Callable[[str], None]) -> Any:
    """Run fn() while routing stdout/stderr to log_cb (one call per line)."""
    buf = io.StringIO()
    with _CAPTURE_LOCK:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = buf
        try:
            result = fn()
        finally:
            sys.stdout, sys.stderr = old_out, old_err
    for line in buf.getvalue().splitlines():
        log_cb(line)
    return result


# ── importlib helpers ─────────────────────────────────────────────────────────

def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── TranscriptionWorker ───────────────────────────────────────────────────────

class TranscriptionWorker:
    def __init__(self):
        self._engine = None
        self._loaded = False
        self._load_error: Optional[str] = None
        self._load_time = 0.0
        self._tx_mod = None
        self._state_lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self, status_cb: Callable[[str], None] = print):
        if self._running:
            return
        with self._state_lock:
            if self._load_error:
                self._queue = queue.Queue()
                self._load_error = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, args=(status_cb,), daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._queue.put(None)  # sentinel
        if self._thread:
            self._thread.join(timeout=5)

    def submit(self, job_id: str, audio_path: str, save_path: str,
               done_cb: Callable[[str, bool, str], None],
               progress_cb: Optional[Callable[[str], None]] = None):
        with self._state_lock:
            load_error = self._load_error
            if load_error is None:
                self._queue.put((job_id, audio_path, save_path, done_cb, progress_cb))
                return
        done_cb(job_id, False, load_error)

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def load_time(self) -> float:
        return self._load_time

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def _loop(self, status_cb):
        status_cb("[TX] Loading transcription models...")
        t0 = time.time()
        try:
            self._tx_mod = _load_module(_TX_MODULE_PATH, "transcriber_v3")
            self._engine = self._tx_mod.TranscriberEngine()
            self._engine.load(progress=status_cb)
            self._loaded = True
            self._load_time = round(time.time() - t0, 1)
            status_cb(f"[TX] Models ready ({self._load_time}s)")
        except Exception:
            load_error = traceback.format_exc()
            pending_jobs = []
            with self._state_lock:
                self._load_error = load_error
                self._running = False
                while True:
                    try:
                        pending = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if pending is not None:
                        pending_jobs.append(pending)
            status_cb(f"[TX] Load failed:\n{load_error}")
            for job_id, _audio, _save, done_cb, _progress_cb in pending_jobs:
                done_cb(job_id, False, load_error)
            return

        while self._running:
            item = self._queue.get()
            if item is None:
                break
            job_id, audio_path, save_path, done_cb, progress_cb = item
            status_cb(f"[TX] Starting {Path(audio_path).name}")
            try:
                log_lines = []

                def _log(line):
                    status_cb(f"[TX] {line}")
                    log_lines.append(line)
                    if progress_cb:
                        progress_cb(line)

                if progress_cb:
                    progress_cb("[2%] Preparing transcription")
                _capture(
                    lambda: self._engine.process_file(audio_path, save_path, progress=_log),
                    _log,
                )
                done_cb(job_id, True, "\n".join(log_lines[-10:]))
            except Exception:
                err = traceback.format_exc()
                status_cb(f"[TX] Error: {err}")
                done_cb(job_id, False, err)


# ── AuditWorker ───────────────────────────────────────────────────────────────

class AuditWorker:
    def __init__(self):
        self._au_mod = None
        self._client = None
        self._client_class = None
        self._clients: Dict[str, Any] = {}
        self._loaded = False
        self._load_error: Optional[str] = None
        self._load_time = 0.0
        self._queue: queue.Queue = queue.Queue()
        self._state_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self, status_cb: Callable[[str], None] = print):
        if self._running:
            return
        with self._state_lock:
            if self._load_error:
                self._queue = queue.Queue()
                self._load_error = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, args=(status_cb,), daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=5)

    def submit(self, job_id: str, transcript_path: str, out_dir: str,
               crm_path: Optional[str], skip_classify: bool, skip_names: bool,
               skip_scoring: bool, skip_coaching: bool,
               done_cb: Callable[[str, bool, str], None],
               progress_cb: Optional[Callable[[str], None]] = None,
               model: Optional[str] = None):
        item = (job_id, transcript_path, out_dir, crm_path,
                skip_classify, skip_names, skip_scoring, skip_coaching,
                done_cb, progress_cb, model)
        with self._state_lock:
            load_error = self._load_error
            if load_error is None:
                self._queue.put(item)
                return
        done_cb(job_id, False, load_error)

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def load_time(self) -> float:
        return self._load_time

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def _preload_models(self, status_cb):
        """Load the stock model, and the fine-tuned model when active, into
        Ollama now so the first audit does not pay the model-load time."""
        from core.audit.llm import SINGLE_PASS_CONTEXT_TOKENS
        from core.audit.lora_audit import LORA_OPTIONS, get_lora_client, lora_enabled
        targets = [(self._client, SINGLE_PASS_CONTEXT_TOKENS)]
        if lora_enabled():
            targets.append((get_lora_client(), LORA_OPTIONS["num_ctx"]))
        for client, num_ctx in targets:
            status_cb(f"[AU] Loading {client.model} into Ollama...")
            try:
                client.preload({"num_ctx": num_ctx})
                status_cb(f"[AU] {client.model} loaded")
            except Exception as exc:  # the model still loads on the first audit
                status_cb(f"[AU] Could not preload {client.model}: {exc}")

    def _loop(self, status_cb):
        status_cb("[AU] Loading audit module...")
        t0 = time.time()
        try:
            # Import from the split package (core/audit/) instead of monolith file.
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from core.audit.llm import OllamaClient
            from core.audit.pipeline import process_one as _process_one
            self._client = OllamaClient(keep_alive=-1)
            self._client_class = OllamaClient
            self._clients = {self._client.model: self._client}
            self._process_one = _process_one
            self._preload_models(status_cb)
            self._loaded = True
            self._load_time = round(time.time() - t0, 1)
            status_cb(f"[AU] Audit module ready ({self._load_time}s)")
        except Exception:
            load_error = traceback.format_exc()
            pending_jobs = []
            with self._state_lock:
                self._load_error = load_error
                self._running = False
                while True:
                    try:
                        pending = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if pending is not None:
                        pending_jobs.append(pending)
            status_cb(f"[AU] Load failed:\n{load_error}")
            for pending in pending_jobs:
                job_id, done_cb = pending[0], pending[-3]
                done_cb(job_id, False, load_error)
            return

        while self._running:
            item = self._queue.get()
            if item is None:
                break
            (job_id, transcript_path, out_dir, crm_path,
             skip_classify, skip_names, skip_scoring, skip_coaching,
             done_cb, progress_cb, model) = item
            status_cb(f"[AU] Starting {Path(transcript_path).name}")
            try:
                log_lines = []

                def _log(line):
                    status_cb(f"[AU] {line}")
                    log_lines.append(line)
                    if progress_cb:
                        progress_cb(line)

                _po = self._process_one
                tp = Path(transcript_path)
                root = tp.parent
                dp = Path(crm_path) if crm_path else None
                selected_model = model or self._client.model
                _cli = self._clients.get(selected_model)
                if _cli is None:
                    _cli = self._client_class(model=selected_model, keep_alive=-1)
                    self._clients[selected_model] = _cli

                def _run():
                    return _po(
                        path=tp,
                        client=_cli,
                        root=root,
                        out_dir=Path(out_dir),
                        # Structured output already constrains the response;
                        # never repeat an entire multi-minute production audit.
                        retries=0,
                        details_path=dp,
                        skip_classify=skip_classify,
                        skip_names=skip_names,
                        skip_scoring=skip_scoring,
                        skip_coaching=skip_coaching,
                        progress=progress_cb,
                    )

                if progress_cb:
                    progress_cb("[5%] Preparing audit evidence and CRM context")
                result = _capture(_run, _log)
                if progress_cb:
                    progress_cb("[100%] Audit complete")
                if result is not None:
                    done_cb(job_id, True, "\n".join(log_lines[-10:]))
                else:
                    failure_detail = "\n".join(log_lines[-12:]).strip()
                    done_cb(
                        job_id,
                        False,
                        failure_detail or "Audit failed without a diagnostic message.",
                    )
            except Exception:
                err = traceback.format_exc()
                status_cb(f"[AU] Error: {err}")
                done_cb(job_id, False, err)


def unload_ollama_models(log: Callable[[str], None] = print) -> None:
    """Unload every model Ollama holds in memory (like `ollama stop <model>` for each).
    The Ollama service keeps running, so the next start only reloads the models."""
    import json
    import urllib.request
    from config import OLLAMA_HOST

    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/ps", timeout=5) as r:
            names = [m["name"] for m in json.loads(r.read()).get("models", [])]
    except Exception as exc:  # Ollama not running: nothing to unload
        log(f"[ENGINE] Ollama not reachable ({exc}); no models to unload")
        return
    for name in names:
        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/generate", data=json.dumps({"model": name, "keep_alive": 0}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                r.read()
            log(f"[ENGINE] Unloaded {name} from Ollama")
        except Exception as exc:
            log(f"[ENGINE] Could not unload {name}: {exc}")


# ── Engine singleton ──────────────────────────────────────────────────────────

class Engine:
    def __init__(self):
        self.tx = TranscriptionWorker()
        self.au = AuditWorker()
        self._status: str = "idle"
        self._started = False
        self._start_time: Optional[float] = None
        self._log = deque(maxlen=200)
        self._lock = threading.Lock()

    def _log_line(self, line: str):
        with self._lock:
            self._log.append(line)

    def start(self):
        if self._started:
            return
        self._started = True
        self._status = "loading"
        self._start_time = time.time()
        self.tx.start(status_cb=self._log_line)
        self.au.start(status_cb=self._log_line)

    def stop(self):
        self.tx.stop()
        self.au.stop()
        self._started = False
        self._status = "idle"

    def shutdown(self, log: Callable[[str], None] = print):
        """Stop the workers and free Ollama's memory: what Ctrl+C on wsgi.py does."""
        self.stop()
        unload_ollama_models(log)

    @property
    def status(self) -> str:
        if not self._started:
            return "idle"
        if self.tx.load_error or self.au.load_error:
            return "error"
        if self.tx.loaded and self.au.loaded:
            return "ready"
        return "loading"

    def state_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "tx_loaded": self.tx.loaded,
            "au_loaded": self.au.loaded,
            "tx_load_time": self.tx.load_time,
            "au_load_time": self.au.load_time,
            "tx_error": self.tx.load_error,
            "au_error": self.au.load_error,
            "recent_log": list(self._log)[-20:],
        }


_ENGINE_INSTANCE: Optional[Engine] = None
_ENGINE_LOCK = threading.Lock()


def get_engine() -> Engine:
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        with _ENGINE_LOCK:
            if _ENGINE_INSTANCE is None:
                _ENGINE_INSTANCE = Engine()
    return _ENGINE_INSTANCE
