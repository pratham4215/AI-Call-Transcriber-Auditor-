"""
Central configuration — all paths and runtime settings.
Override any value via environment variable.
"""
import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent

# Load a local .env when python-dotenv is installed. Environment variables that
# are already exported by the service keep precedence over values in the file.
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional during lightweight tooling
    load_dotenv = None
if load_dotenv is not None:
    load_dotenv(BASE / ".env", override=False)

# ── Data directories ──────────────────────────────────────────────────────────
DATA_DIR      = Path(os.environ.get("DATA_DIR",      str(BASE / "data")))
TRANSCRIPTS_DIR = Path(os.environ.get("TRANSCRIPTS_DIR", str(BASE / "transcripts")))
REPORTS_DIR   = Path(os.environ.get("REPORTS_DIR",   str(BASE / "reports")))
MODELS_DIR    = Path(os.environ.get("MODELS_DIR",    str(BASE / "models")))
RESOURCES_DIR = BASE / "resources"

# ── ASR model ─────────────────────────────────────────────────────────────────
NEMO_MODEL    = str(MODELS_DIR / "asr.nemo")
ASR_BATCH_SIZE = int(os.environ.get("ASR_BATCH_SIZE", "16"))

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_HOST  = os.environ.get("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:32b")

# Audit generation is evidence extraction, not open-ended reasoning. Keeping
# thinking disabled and decoding deterministic reduces latency and run-to-run
# variance while the post-LLM validator protects correctness.
AUDIT_ENABLE_THINKING = os.environ.get("AUDIT_ENABLE_THINKING", "false").lower() == "true"
AUDIT_TEMPERATURE = float(os.environ.get("AUDIT_TEMPERATURE", "0"))
AUDIT_MAX_TOKENS = int(os.environ.get("AUDIT_MAX_TOKENS", "6144"))
AUDIT_CONTEXT_TOKENS = int(os.environ.get("AUDIT_CONTEXT_TOKENS", "12288"))

# Fine-tuned transcript-rating model (Qwen3-32B + shared LoRA, served by Ollama).
# "active" rates its trained transcript parameters; anything else, including an
# unset or empty value, keeps the stock-only audit.
AUDIT_LORA_MODE  = os.environ.get("AUDIT_LORA_MODE", "off").strip().lower()
AUDIT_LORA_MODEL = os.environ.get("AUDIT_LORA_MODEL", "audit-lora-v1:latest").strip()

# How ratings are read from the models: "generate" (the model writes the JSON)
# or "brio" (each rating is picked by its probability; free text is still
# written). Brio is faster; any Brio failure falls back to generate for that call.
AUDIT_SCORING = os.environ.get("AUDIT_SCORING", "generate").strip().lower()

# ── PostgreSQL (CRM / call data) ──────────────────────────────────────────────
# DATABASE_URL takes precedence; otherwise the connection is built from DB_*.
# Sessions are read-only: the audit app never writes to the CRM database.
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
DB_HOST      = os.environ.get("DB_HOST", "").strip()
DB_PORT      = int(os.environ.get("DB_PORT", "").strip() or "5432")
DB_NAME      = os.environ.get("DB_NAME", "").strip()
DB_USER      = os.environ.get("DB_USER", "").strip()
DB_PASSWORD  = os.environ.get("DB_PASSWORD", "")
DB_SSLMODE   = os.environ.get("DB_SSLMODE", "").strip()
# timestamptz values are shown in this zone, matching the CALL DATA.xlsx
# exports: a UTC offset (+05:30, IST, needs no tz database) or a zone name.
DB_TIMEZONE  = os.environ.get("DB_TIMEZONE", "+05:30").strip()
DB_CONNECT_TIMEOUT      = int(os.environ.get("DB_CONNECT_TIMEOUT", "5"))
DB_STATEMENT_TIMEOUT_MS = int(os.environ.get("DB_STATEMENT_TIMEOUT_MS", "15000"))

# Call-level CRM source: "auto" (the CRM database when configured, CALL DATA.xlsx
# on a miss or DB error), "db" (database only) or "excel" (CALL DATA.xlsx only).
CRM_SOURCE = os.environ.get("CRM_SOURCE", "auto").strip().lower()

# ── Kataria CRM API (AI endpoints + recording decrypt) ────────────────────────
# Sent as "Authorization: Bearer <CRM_API_TOKEN>" (the ktr-ai-... key).
CRM_API_URL     = os.environ.get("CRM_API_URL", "https://smr-api.katariacrm.com/crm").strip()
CRM_API_TOKEN   = os.environ.get("CRM_API_TOKEN", "").strip()
CRM_API_TIMEOUT = int(os.environ.get("CRM_API_TIMEOUT", "60"))
# Calls fetched from the CRM at the same time (each waits on the DB, the CRM API and bizRTC).
IMPORT_WORKERS = max(1, int(os.environ.get("IMPORT_WORKERS", "30")))
# When a day's list loads, the first N calls of each call type are downloaded in the
# background so Play is instant (0 = off).
PREFETCH_ON_LIST = max(0, int(os.environ.get("PREFETCH_ON_LIST", "20")))

# The dashboard lists only calls on leads of this type (lead_transaction.lead_type):
# UNNATI or PRAGATI; empty shows both.
DASHBOARD_LEAD_TYPE = os.environ.get("DASHBOARD_LEAD_TYPE", "UNNATI").strip().upper()

# ── Reviewer dashboard (reviewer_wsgi.py: human reviews of audited calls) ─────
REVIEWS_DIR   = Path(os.environ.get("REVIEWS_DIR", str(BASE / "reviews")))
REVIEWER_HOST = os.environ.get("REVIEWER_HOST", "0.0.0.0")
REVIEWER_PORT = int(os.environ.get("REVIEWER_PORT", "5001"))
# HTTPS for the reviewer dashboard: certificate and key file paths (both empty = plain HTTP).
REVIEWER_SSL_CERT = os.environ.get("REVIEWER_SSL_CERT", "").strip()
REVIEWER_SSL_KEY  = os.environ.get("REVIEWER_SSL_KEY", "").strip()
# PIN the reviewer dashboard asks for before deleting a review.
REVIEW_DELETE_PIN = os.environ.get("REVIEW_DELETE_PIN", "").strip()
# Admin of the reviewer dashboard: who is online, activity, delete reviews, remove calls.
REVIEW_ADMIN_PIN = os.environ.get("REVIEW_ADMIN_PIN", "").strip()
ACTIVITY_KEEP_DAYS = int(os.environ.get("ACTIVITY_KEEP_DAYS", "30"))

# ── Flask ─────────────────────────────────────────────────────────────────────
FLASK_HOST   = os.environ.get("FLASK_HOST", "0.0.0.0")
FLASK_PORT   = int(os.environ.get("FLASK_PORT", "5000"))
FLASK_DEBUG  = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
