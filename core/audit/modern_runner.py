"""One-call modern audit runner (model evaluation once; Python scoring after)."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .llm import extract_json
from .modern_adapter import evaluate_payload
from .modern_prompt import build_modern_evaluation_prompt


def run_modern_audit(
    client: Any,
    turns: Sequence[Any],
    call_type: str,
    crm_details: Mapping[str, Any] | None = None,
    **integrity_flags: Any,
) -> dict[str, Any]:
    """Run one structured LLM evaluation and deterministic post-processing.

    `client` must expose the existing LLMClient.generate(system, user, schema,
    single_pass=True) interface. No LoRA call or second transcript read occurs.
    """
    system, user, schema = build_modern_evaluation_prompt(turns, call_type, crm_details)
    raw = client.generate(system, user, schema, single_pass=True)
    payload = extract_json(raw)
    return evaluate_payload(payload, **integrity_flags)
