"""
brio.py — Brio scoring: pick each rating by its probability, not by writing JSON.

Instead of letting the model write the whole JSON answer token by token, the
reply is built here and the model is asked only one question per choice: at
the point where the value goes ('{"greeting": "'), how likely is each allowed
option? The most likely option is written in and the next question asked, so
every choice is conditioned on the earlier ones, as in normal generation.

Tested on 2026-10-06 against normal generation:
  - fine-tuned model, 545 test calls: same accuracy, 97.7% same ratings, 2.6x faster
  - stock model (lean call), 40 real audits: 96.5% same evidence_state, ~49% faster

Every option list (enum) and yes/no field is picked this way, and so is a
turn-ID field ("..._turn_id"): its options are the call's own turn IDs or "".
Other free text (decision_basis, rationale, lists of turn IDs) is still
generated, constrained to its JSON Schema. Every function raises on any
problem so the caller falls back to generation.
"""
from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Dict, List, Tuple

from .llm import OllamaClient

try:
    from config import AUDIT_SCORING
except ImportError:  # pragma: no cover - package imported without project root
    AUDIT_SCORING = "generate"

logger = logging.getLogger(__name__)

# Qwen3 chat template with thinking off: what Ollama's chat endpoint sends.
QWEN3_PROMPT = (
    "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
TOP_LOGPROBS = 20
# Below this confidence a rating is flagged for a human check: in the 545-call
# test, ratings under 0.7 matched the reference only about half the time.
CHECK_BELOW = 0.70
_FREE_TEXT_TOKENS = 512
_MAX_STEPS = 6          # tokens read for one option ("date" + "_only" = 2, "T" + "0" + "1" + "2" = 4)
_STRING_END = '"'       # text after an option inside a JSON string
_LITERAL_END = ",}] \n"  # text after a true/false literal


def brio_enabled() -> bool:
    if AUDIT_SCORING not in {"brio", "generate"}:
        logger.warning("[BRIO] Unknown AUDIT_SCORING=%r; using generate.", AUDIT_SCORING)
    return AUDIT_SCORING == "brio"


def _fits(option: str, typed: str, token: str, ends: str) -> bool:
    """Can the option still be written after ``typed`` + ``token``?"""
    cut = min((token.index(c) for c in ends if c in token), default=None)
    if cut is not None:                      # the token closes the value
        return typed + token[:cut] == option
    return option.startswith(typed + token)


def _pick(client: OllamaClient, prompt: str, options: List[str], ends: str,
          run_options: Dict[str, Any], timeout: float) -> Tuple[str, Dict[str, float]]:
    """Most likely option and the probability of each.

    One model step reads the next-token candidates. A candidate that fits one
    option counts for it ("sup" -> supported). When the most likely candidate
    fits several ("date" -> date_time_confirmed / date_only), it is written
    and the next token decides among those options.
    """
    probs = {option: 0.0 for option in options}
    alive, typed, share = list(options), "", 1.0
    for _ in range(_MAX_STEPS):
        out = client.complete_raw(prompt + typed, {**run_options, "num_predict": 1},
                                  top_logprobs=TOP_LOGPROBS, timeout=timeout)
        single: Dict[str, float] = {}
        shared: Dict[str, Tuple[float, List[str]]] = {}
        for cand in out["logprobs"][0]["top_logprobs"]:
            token = cand.get("token") or ""
            if not typed and ends == _LITERAL_END:
                token = token.lstrip(" ")    # ': true' may arrive as ' true'
            if not token:
                continue
            hits = [option for option in alive if _fits(option, typed, token, ends)]
            p = math.exp(cand["logprob"])
            if len(hits) == 1:
                single[hits[0]] = single.get(hits[0], 0.0) + p
            elif hits:
                shared[token] = (shared.get(token, (0.0, hits))[0] + p, hits)
        total = sum(single.values()) + sum(p for p, _ in shared.values())
        if total <= 0:
            raise ValueError(f"no allowed option among the model's top tokens {alive}")
        for option, p in single.items():
            probs[option] += share * p / total
        best = max(shared.items(), key=lambda kv: kv[1][0], default=None)
        if best is None or share * best[1][0] / total <= max(probs.values()):
            break
        token, (p, hits) = best
        typed, share, alive = typed + token, share * p / total, hits
    mass = sum(probs.values())
    if mass <= 0:
        raise ValueError(f"no option decided within {_MAX_STEPS} tokens: {alive}")
    probs = {option: p / mass for option, p in probs.items()}
    return max(probs, key=probs.get), probs


def _token_cap(schema: Dict[str, Any]) -> int:
    """Tokens one free-text value may take: a token is at least one character, so a
    maxLength bounds it (plus quotes and separators)."""
    rule = schema.get("items", schema)
    if "maxLength" not in rule:
        return _FREE_TEXT_TOKENS
    count = schema.get("maxItems", 1) if "items" in schema else 1
    return min(_FREE_TEXT_TOKENS, count * (rule["maxLength"] + 3) + 2)


def _write(client: OllamaClient, prompt: str, schema: Dict[str, Any],
           run_options: Dict[str, Any], timeout: float) -> Any:
    """Generate one free-text JSON value, constrained to its schema."""
    out = client.complete_raw(prompt, {**run_options, "num_predict": _token_cap(schema)},
                              fmt=schema, timeout=timeout)
    value = json.loads(out["response"])
    texts = value if isinstance(value, list) else [value]
    rule = schema.get("items", schema)
    for text in texts:
        if isinstance(text, str) and (
            len(text) > rule.get("maxLength", len(text))
            or ("pattern" in rule and not re.fullmatch(rule["pattern"], text))
        ):
            raise ValueError(f"generated value {text!r} breaks its schema")
    return value


def score_ratings(client: OllamaClient, system: str, user: str,
                  ladders: List[Tuple[str, Tuple[str, ...]]], run_options: Dict[str, Any],
                  timeout: float) -> Tuple[Dict[str, str], Dict[str, Dict[str, float]]]:
    """Fine-tuned model: {"param": "Rating", ...} in trained key order."""
    prompt = QWEN3_PROMPT.format(system=system, user=user)
    reply = "{"
    ratings: Dict[str, str] = {}
    probs: Dict[str, Dict[str, float]] = {}
    for i, (key, ladder) in enumerate(ladders):
        reply += ("" if i == 0 else ", ") + f'"{key}": "'
        ratings[key], probs[key] = _pick(client, prompt + reply, list(ladder), _STRING_END,
                                         run_options, timeout)
        reply += ratings[key] + '"'
    return ratings, probs


def score_json(client: OllamaClient, system: str, user: str, schema: Dict[str, Any],
               run_options: Dict[str, Any], timeout: float,
               turn_ids: Dict[str, List[str]]) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """Stock model: fill a JSON Schema object in property order. Option lists,
    yes/no fields and single turn IDs are picked by Brio; other values are
    generated. ``turn_ids`` maps a turn-ID field name to the transcript's IDs
    it may name ("T001"...), "" for every other turn-ID field. Written on its
    own, a turn ID came out as junk ("]=>, ", "T008|"), which the Booking
    checks then read as no Customer agreement.

    Returns the JSON text the generation call would have returned, plus the
    probabilities of every picked value keyed by its path
    ("evidence_ledger.ask_kms.evidence_state", "booking_genuine.genuine").
    """
    prompt = QWEN3_PROMPT.format(system=system, user=user)
    probs: Dict[str, Dict[str, float]] = {}

    def fill(node: Dict[str, Any], reply: str, path: str) -> Tuple[Any, str]:
        kind = node.get("type")
        if kind == "object" and "properties" in node:
            value: Dict[str, Any] = {}
            reply += ("" if not path else " ") + "{"
            for i, (key, child) in enumerate(node["properties"].items()):
                reply += ("" if i == 0 else ", ") + f'"{key}":'
                value[key], reply = fill(child, reply, f"{path}.{key}" if path else key)
            return value, reply + "}"
        if kind == "string" and ("enum" in node or path.endswith("turn_id")):
            reply += ' "'
            field = path.rsplit(".", 1)[-1]
            options = list(node["enum"]) if "enum" in node else ["", *turn_ids.get(field, turn_ids[""])]
            choice, probs[path] = _pick(client, prompt + reply, options, _STRING_END,
                                        run_options, timeout)
            return choice, reply + choice + '"'
        if kind == "boolean":
            choice, probs[path] = _pick(client, prompt + reply, ["true", "false"], _LITERAL_END,
                                        run_options, timeout)
            return choice == "true", reply + " " + choice
        value = _write(client, prompt + reply + " ", node, run_options, timeout)
        return value, reply + " " + json.dumps(value, ensure_ascii=False)

    payload, _ = fill(schema, "", "")
    return json.dumps(payload, ensure_ascii=False), probs


def confidence(probs: Dict[str, float]) -> float:
    """Probability of the chosen option."""
    return round(max(probs.values()), 3)
