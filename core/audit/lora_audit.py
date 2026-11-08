"""
lora_audit.py — Fine-tuned transcript-rating model (Qwen3-32B + shared LoRA).

The deployed model (``AUDIT_LORA_MODEL``, default ``audit-lora-v1:latest``) was
trained on call type + speaker-labelled transcript only, and returns one flat
JSON object mapping each trained parameter of that call type to a rating word.
Source of truth: Fine-Tuned_Qwen3-32B_Model_Information_Sheet_V1 (sections 5-7).

This module owns only that contract:
  - the trained parameters and rating ladders per call type
  - the training prompt template (system + user), with no CRM or rubric text
  - strict parsing: exact keys, ladder-valid values, nothing else
  - mapping the model's keys/ratings onto production parameter ids/levels

It never scores, never applies business rules and never sees CRM data. Any
failure returns None so the caller keeps the stock audit for those parameters.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from .brio import brio_enabled, confidence, score_ratings
from .llm import OLLAMA_HOST, OllamaClient
from .params import allowed_score_levels

try:
    from config import AUDIT_LORA_MODE, AUDIT_LORA_MODEL
except ImportError:  # pragma: no cover - package imported without project root
    AUDIT_LORA_MODE, AUDIT_LORA_MODEL = "off", ""

logger = logging.getLogger(__name__)

# ── Trained contract (model sheet section 5, in trained key order) ────────────
Z, P, G, F = "Zero", "Partial", "Good", "Full"
_ZF, _ZPF, _ZPGF = (Z, F), (Z, P, F), (Z, P, G, F)

TRAINED_PARAMS: Dict[str, List[Tuple[str, Tuple[str, ...]]]] = {
    "BOOKING": [
        ("greeting", _ZF), ("introduction", _ZPF),
        ("vehicle_number_and_customer_name", _ZPF), ("recording_info", _ZF),
        ("reason_of_calling", _ZF), ("permission_to_talk", _ZF),
        ("listening_and_understanding", _ZPF),
        ("information_gathering_and_solutions", _ZPF),
        ("professionalism", _ZPGF), ("conflict_management", _ZPF),
        ("standard_communication", _ZF), ("thank_you", _ZF),
    ],
    "FOLLOW_UPS": [
        ("greeting", _ZF), ("introduction", _ZPF),
        ("vehicle_number_and_customer_name", _ZPF), ("recording_info", _ZF),
        ("reason_of_calling", _ZF), ("permission_to_talk", _ZF),
        ("professionalism", _ZPGF), ("standard_communication", _ZF),
        ("thank_you", _ZF),
    ],
    "ESCALATION": [
        ("greeting", _ZF), ("introduction", _ZPF),
        ("vehicle_number_and_customer_name", _ZPF), ("recording_info", _ZF),
        ("reason_of_calling", _ZF), ("permission_to_talk", _ZF),
        ("listening_and_understanding", _ZPF), ("dissatisfaction_inquiry", _ZF),
        ("professionalism", _ZPGF), ("conflict_management", _ZPF),
        ("standard_communication", _ZF), ("higher_billing_explanation", _ZF),
        ("future_service_convincing", _ZF), ("thank_you", _ZF),
    ],
    "SERVICE_NT_REQUIRED": [
        ("greeting", _ZF), ("introduction", _ZPF),
        ("vehicle_number_and_customer_name", _ZPF), ("recording_info", _ZF),
        ("reason_of_calling", _ZF), ("permission_to_talk", _ZF),
        ("listening_and_understanding", _ZPF),
        ("reason_for_services_not_required", _ZPF),
        ("professionalism", _ZPF), ("conflict_management", _ZPF),
        ("standard_communication", _ZF), ("future_service_convincing", _ZPF),
        ("thank_you", _ZF),
    ],
}

# Call-type label exactly as written in the training system message.
TRAINED_CALL_TYPE_LABELS = {
    "BOOKING": "BOOKING",
    "FOLLOW_UPS": "FOLLOW UP",
    "ESCALATION": "ESCALATION",
    "SERVICE_NT_REQUIRED": "SERVICE NOT REQUIRED",
}

# Model key → production parameter id.
MODEL_KEY_TO_PID = {
    "greeting": "greet",
    "introduction": "introduce",
    "vehicle_number_and_customer_name": "confirm_vehicle_name",
    "recording_info": "inform_recording",
    "reason_of_calling": "reason_for_call",
    "permission_to_talk": "permission_to_talk",
    "listening_and_understanding": "listen_understand",
    "information_gathering_and_solutions": "info_gathering",
    "dissatisfaction_inquiry": "dissatisfaction_inquiry",
    "reason_for_services_not_required": "reason_snr",
    "professionalism": "professionalism",
    "conflict_management": "conflict_mgmt",
    "higher_billing_explanation": "higher_billing",
    "future_service_convincing": "future_service",
    "standard_communication": "standard_comm",
    "thank_you": "thank_you",
}


def owned_pids(table: str) -> List[str]:
    """Production parameter ids the fine-tuned model rates for this call type."""
    return [MODEL_KEY_TO_PID[key] for key, _ in TRAINED_PARAMS.get(table, [])]


# ── Runtime settings (Modelfile: temperature 0, num_ctx 12288) ────────────────
LORA_OPTIONS = {"temperature": 0, "num_ctx": 12288, "num_predict": 256}
LORA_TIMEOUT_S = 180
TRAINED_MAX_SEQ_TOKENS = 8192
_OUTPUT_TOKEN_RESERVE = 160
_OMITTED_LINE = "... [middle of call omitted for length] ..."


def lora_enabled() -> bool:
    if AUDIT_LORA_MODE not in {"active", "off"}:
        logger.warning("[LORA] Unknown AUDIT_LORA_MODE=%r; using stock-only audit.", AUDIT_LORA_MODE)
    return AUDIT_LORA_MODE == "active" and bool(AUDIT_LORA_MODEL)


_CLIENT: Optional[OllamaClient] = None
_CLIENT_LOCK = threading.Lock()


def get_lora_client() -> OllamaClient:
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = OllamaClient(model=AUDIT_LORA_MODEL, host=OLLAMA_HOST, keep_alive=-1)
        return _CLIENT


# ── Prompt (training template, model sheet section 6) ─────────────────────────
_SYSTEM_TEMPLATE = """You are a call-quality auditor for a car dealership. You review one outbound call made by a CRE (Customer Relationship Executive) to a customer.

Call type: {call_type}

The transcript is multilingual (Gujarati, Hindi and English, often code-switched within a sentence). Judge what was said and meant, not grammar or spelling, and never penalise transcription noise.

Rate ONLY these parameters, using ONLY the ratings allowed for each:
{parameters}

Rating meanings:
- Full: the expectation was clearly met in the call.
- Good: largely met, with a minor shortfall.
- Partial: partially met or done incompletely.
- Zero: not done, or done wrongly.
(Only the ratings listed for a parameter are valid for that parameter.)

Rules:
- Base every rating only on evidence present in the transcript.
- Do not infer facts that were never spoken.
- Do not compute totals, scores or grades.
- Reply with a single JSON object and nothing else: {{"<parameter>": "<rating>", ...}} containing exactly the parameters listed above."""

_SECONDS_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*s?\s*$")
_SRT_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})[,.](\d{3})$")


def _clock(ts: str) -> Optional[str]:
    """'12.3s' or SRT '00:00:12,300' → training form '00:12.3'."""
    text = str(ts or "").strip()
    srt = _SRT_RE.match(text)
    if srt:
        h, m, s, ms = (int(part) for part in srt.groups())
        seconds = h * 3600 + m * 60 + s + ms / 1000
    else:
        plain = _SECONDS_RE.match(text)
        if not plain:
            return None
        seconds = float(plain.group(1))
    minutes, rest = divmod(round(seconds, 1), 60)
    return f"{int(minutes):02d}:{rest:04.1f}"


def _estimate_tokens(text: str) -> int:
    """Conservative Qwen token estimate: ~4 ASCII chars per token, 1 per other char."""
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    return ascii_chars // 4 + (len(text) - ascii_chars) + 1


def _fit_lines(lines: List[str], budget: int) -> List[str]:
    """Training truncation (drop_middle_turns): keep opening and closing turns."""
    if sum(_estimate_tokens(line) for line in lines) <= budget:
        return lines
    budget -= _estimate_tokens(_OMITTED_LINE)
    head: List[str] = []
    tail: List[str] = []
    lo, hi = 0, len(lines) - 1
    while lo <= hi:
        line = lines[lo] if len(head) <= len(tail) else lines[hi]
        cost = _estimate_tokens(line)
        if cost > budget:
            break
        budget -= cost
        if len(head) <= len(tail):
            head.append(line)
            lo += 1
        else:
            tail.insert(0, line)
            hi -= 1
    return head + [_OMITTED_LINE] + tail


def build_lora_prompt(turns: List[Any], table: str) -> Optional[Tuple[str, str]]:
    """Return (system, user) in the trained format, or None if turns do not fit it."""
    params = TRAINED_PARAMS.get(table)
    if not params or not turns:
        return None
    lines: List[str] = []
    for turn in turns:
        speaker = str(turn.speaker or "").strip().capitalize()
        start, end = _clock(turn.start), _clock(turn.end)
        if speaker not in {"Agent", "Customer"} or start is None or end is None:
            return None  # the model only saw Agent/Customer turns with timestamps
        lines.append(f"T{turn.index:03d} | {speaker} | {start}-{end} | {turn.text}")

    system = _SYSTEM_TEMPLATE.format(
        call_type=TRAINED_CALL_TYPE_LABELS[table],
        parameters="\n".join(f"- {key}: {' / '.join(ladder)}" for key, ladder in params),
    )
    budget = TRAINED_MAX_SEQ_TOKENS - _estimate_tokens(system) - _OUTPUT_TOKEN_RESERVE
    lines = _fit_lines(lines, budget)
    user = f"Transcript ({len(turns)} turns):\n" + "\n".join(lines) + "\n\nReturn the JSON object now."
    return system, user


# ── Strict output parsing ─────────────────────────────────────────────────────
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _no_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate keys")
    return dict(pairs)


def parse_lora_output(raw: str, table: str) -> Dict[str, str]:
    """Parse the model output; raise ValueError unless it matches the contract exactly."""
    text = _THINK_RE.sub("", raw or "").strip()
    fenced = _FENCE_RE.match(text)
    if fenced:
        text = fenced.group(1)
    try:
        data = json.loads(text, object_pairs_hook=_no_duplicate_keys)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"not a single JSON object ({exc})") from None
    if not isinstance(data, dict):
        raise ValueError("output is not a JSON object")
    ladders = dict(TRAINED_PARAMS[table])
    missing = sorted(set(ladders) - set(data))
    extra = sorted(set(data) - set(ladders))
    if missing or extra:
        raise ValueError(f"key mismatch (missing={missing}, extra={extra})")
    invalid = {
        key: value for key, value in data.items()
        if not isinstance(value, str) or value not in ladders[key]
    }
    if invalid:
        raise ValueError(f"ratings outside the {table} ladder: {invalid}")
    return data


def to_production_audit(
    ratings: Dict[str, str], table: str, model: str,
    probs: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Dict[str, str]]:
    """Map model ratings onto production pids and levels.

    Production scoring is authoritative: where production configures a binary
    control (e.g. ``introduce``) and the model returns ``Partial``, the
    production rubric ("either element missing → Zero") applies.
    ``probs`` (Brio scoring) adds the model's confidence to the evidence text.
    """
    audit: Dict[str, Dict[str, str]] = {}
    for key, rating in ratings.items():
        pid = MODEL_KEY_TO_PID[key]
        level = rating if rating in allowed_score_levels(table, pid) else "Zero"
        sure = f"{probs[key][rating]:.0%} confident; " if probs else ""
        evidence = f"Rated {rating} by the fine-tuned model ({sure}rating only, no written proof)."
        if level != rating:
            evidence += f" {rating} is not a production level for this check, so it scores Zero."
        audit[pid] = {"rating": level, "evidence": evidence}
        if probs:
            audit[pid]["confidence"] = confidence(probs[key])
    return audit


def run_lora_audit(
    turns: List[Any], table: str, client: Optional[OllamaClient] = None
) -> Tuple[Optional[Dict[str, Dict[str, str]]], Dict[str, Any]]:
    """Rate the trained parameters for one call.

    Returns (audit_entries | None, info). None means the output could not be
    safely accepted; ``info["reason"]`` says why and the caller falls back to
    the stock model for every parameter.
    """
    client = client or get_lora_client()
    info: Dict[str, Any] = {"model": client.model, "status": "fallback"}
    prompt = build_lora_prompt(turns, table)
    if prompt is None:
        info["reason"] = "transcript is not in the trained Agent/Customer timestamped turn format"
        return None, info
    if brio_enabled():
        try:
            ratings, probs = score_ratings(
                client, *prompt, TRAINED_PARAMS[table], LORA_OPTIONS, LORA_TIMEOUT_S
            )
            info.update(status="accepted", scoring="brio")
            return to_production_audit(ratings, table, client.model, probs), info
        except Exception as exc:  # noqa: BLE001 - Brio failure falls back to generation
            logger.warning("[BRIO] fine-tuned ratings failed (%s: %s); generating instead.",
                           type(exc).__name__, exc)
    try:
        raw = client.generate_ratings(*prompt, options=LORA_OPTIONS, timeout=LORA_TIMEOUT_S)
        info["metrics"] = client.get_last_metrics()
        ratings = parse_lora_output(raw, table)
    except Exception as exc:  # noqa: BLE001 - every failure falls back to stock
        info["reason"] = f"{type(exc).__name__}: {exc}"
        return None, info
    info.update(status="accepted", scoring="generate")
    return to_production_audit(ratings, table, client.model), info
