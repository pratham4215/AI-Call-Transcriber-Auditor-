"""
utils.py — Cross-cutting helpers: PII masking, display, CSV safety.

All helpers here are pure functions with no circular dependencies.
They may be imported safely from any other submodule.
"""
from __future__ import annotations

import re
from typing import Any

# ── PII redaction ─────────────────────────────────────────────────────────────
# Set REDACT_PII = False only for internal debugging; never in production.
REDACT_PII = True

_PHONE_RE = re.compile(r"(?<!\d)(?:\+91[\-\s]?)?[6-9]\d{9}(?!\d)")


def mask_phone(s: str) -> str:
    """Replace Indian mobile numbers with XXXXXX<last4>."""
    if not REDACT_PII or not s:
        return s

    def _m(match: re.Match) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        return "XXXXXX" + digits[-4:] if len(digits) >= 4 else "XXXXXX"

    return _PHONE_RE.sub(_m, s)


def redact_call_id(call_id: str) -> str:
    return mask_phone(call_id)


# ── Display helpers ───────────────────────────────────────────────────────────

def _clean(v: Any) -> str:
    """Stringify, strip, and mask phone numbers."""
    return mask_phone(str(v).strip()) if v is not None else ""


def _display(v: Any, default: str = "Not identified") -> str:
    """Clean value, collapse whitespace, return default when empty."""
    value = _clean(v).replace("\r", " ").replace("\n", " ").strip()
    return value or default


def _md(v: Any, default: str = "Not identified") -> str:
    """Like _display but also escapes Markdown table pipe characters."""
    return _display(v, default).replace("|", "\\|")


# ── JSON payload redaction ────────────────────────────────────────────────────

def _redact_payload(value: Any) -> Any:
    """Recursively mask phone numbers in a nested dict/list/str payload."""
    if isinstance(value, dict):
        return {k: _redact_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_payload(item) for item in value]
    if isinstance(value, str):
        return mask_phone(value)
    return value


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _csv_safe(value: Any) -> str:
    """Prevent formula injection in CSV exports."""
    text = _display(value, "")
    return "'" + text if text.startswith(("=", "+", "-", "@")) else text
