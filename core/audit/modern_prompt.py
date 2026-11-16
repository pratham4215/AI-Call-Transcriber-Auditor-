"""Single-context prompt builder for modern CRE competency evaluation.

The transcript is inserted once. The model returns competency ratings/evidence only;
the Python scoring engine owns all points and overrides.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Sequence, Tuple

from .modern_registry import load_registry


SYSTEM_PROMPT = """You are an evidence-first CRE call auditor for Kataria.
Evaluate the entire conversation once, then judge each competency independently.
Calls may contain Gujarati, Hindi, English, transliteration, code-switching and ASR
errors. Interpret meaning and context rather than matching exact keywords.

HARD RULES:
- The call type supplied by the system is fixed. Never reclassify it.
- Read the complete transcript before assigning any competency rating.
- Use only the supplied transcript and CRM facts. CRM facts are reference context,
  not proof that a customer said or agreed to something.
- Every rating must have evidence references (T-ID for transcript or a CRM field path)
  and a concise reason. Quote short evidence or state a precise CRM fact.
- Full means the competency was fulfilled accurately and sufficiently.
- Partial means it was attempted/fulfilled incompletely but without material failure.
- Zero means missing where required, materially wrong, misleading or failed.
- N/A is allowed only when that competency's registry rule allows it. Supply a
  specific na_reason. Never use N/A merely because evidence is missing or uncertain.
- Use confidence High/Medium/Low. Flag uncertainty, contradiction, critical risk or
  a competency needing human adjudication.
- Do not calculate points, weights, percentages, grades, risk scores or incentives.
- Do not invent customer statements, CRM values, dates, appointment commitments or
  products. Do not penalize a customer refusal by itself.
- CRE remarks are mandatory. CRM & Documentation Integrity is never N/A.
- Fake Booking and Wrong Call Tagging are separate deterministic integrity controls;
  do not set the final score.
Return only the JSON object matching the supplied schema. No markdown or prose.
"""


def build_modern_evaluation_prompt(
    turns: Sequence[Any],
    call_type: str,
    crm_details: Mapping[str, Any] | None = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """Return (system prompt, user prompt, JSON schema) with transcript exactly once."""
    raw, specs = load_registry()
    if call_type not in raw["call_types"]:
        raise ValueError(f"unsupported call type: {call_type!r}")

    transcript_lines = []
    for i, turn in enumerate(turns, start=1):
        if isinstance(turn, Mapping):
            tid = str(turn.get("index") or i)
            speaker = str(turn.get("speaker") or "Unknown")
            text = str(turn.get("text") or "")
        else:
            tid = str(getattr(turn, "index", i))
            speaker = str(getattr(turn, "speaker", "Unknown"))
            text = str(getattr(turn, "text", ""))
        try:
            tid_label = f"T{int(tid):03d}"
        except (TypeError, ValueError):
            tid_label = f"T{tid}"\n        transcript_lines.append(f"{tid_label} [{speaker}]: {text}")
    transcript_block = "\n".join(transcript_lines)

    crm_json = json.dumps(crm_details or {}, ensure_ascii=False, indent=2, default=str)
    competency_lines = []
    for row in raw["competencies"]:
        competency_lines.append(json.dumps({
            "id": row["id"], "name": row["name"],
            "definition": row["definition"], "full": row["full"], "partial": row["partial"],
            "zero": row["zero"], "na_rule": row.get("na_rule"),
            "allow_na": row["allow_na"],
        }, ensure_ascii=False))
    user_prompt = (
        f"FIXED CALL TYPE: {call_type}\n\n"
        "=== COMPLETE TRANSCRIPT (read once; use T-IDs as evidence references) ===\n"
        f"{transcript_block}\n"
        "=== END TRANSCRIPT ===\n\n"
        "=== CRM FACTS (reference only; cite exact field paths) ===\n"
        f"{crm_json}\n"
        "=== END CRM FACTS ===\n\n"
        "=== COMPETENCY RUBRICS ===\n"
        + "\n".join(competency_lines)
        + "\n=== END RUBRICS ===\n\nEvaluate all 12 competencies independently. "
        "Return exactly one result per registry ID."
    )

    schema: Dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "call_type", "competencies", "human_review"],
        "properties": {
            "schema_version": {"const": "modern-audit-evaluation-v1"},
            "call_type": {"const": call_type},
            "competencies": {
                "type": "array", "minItems": 12, "maxItems": 12,
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["id", "rating", "evidence", "reasoning", "confidence", "human_review"],
                    "properties": {
                        "id": {"enum": [s.competency_id for s in specs]},
                        "rating": {"enum": ["Full", "Partial", "Zero", "N/A"]},
                        "evidence": {
                            "type": "array", "minItems": 1,
                            "items": {
                                "type": "object", "additionalProperties": False,
                                "required": ["source", "reference", "quote_or_fact"],
                                "properties": {
                                    "source": {"enum": ["transcript", "crm"]},
                                    "reference": {"type": "string", "minLength": 1},
                                    "quote_or_fact": {"type": "string", "minLength": 1},
                                },
                            },
                        },
                        "reasoning": {"type": "string", "minLength": 1},
                        "na_reason": {"type": ["string", "null"]},
                        "confidence": {"enum": ["High", "Medium", "Low"]},
                        "human_review": {"type": "boolean"},
                    },
                },
            },
            "human_review": {
                "type": "object", "additionalProperties": False,
                "required": ["required", "reasons"],
                "properties": {
                    "required": {"type": "boolean"},
                    "reasons": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    }

    if call_type == "BOOKING":
        user_prompt += (
            "\n\n=== BOOKING GENUINENESS CONTROL ===\n"
            "A CRM booking tag is not proof of a genuine appointment. Confirm explicit Customer "
            "agreement and a dated appointment commitment from Customer turns. CRE-only claims, "
            "a vague promise, a date without Customer agreement, or explicit Customer decline/"
            "cancellation/dispute are not genuine. A booking made on an earlier call may be "
            "confirmed on this follow-up without repeating date/time. If too short or ambiguous, "
            "do not invent; flag human review. Cite exact Customer T-IDs for agreement/date.\n"
            "=== END BOOKING CONTROL ==="
        )
        schema["required"].append("booking_genuine")
        schema["properties"]["booking_genuine"] = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "genuine", "commitment_type", "customer_agreement_confirmed", "agreement_turn_id",
                "final_customer_stance", "final_stance_turn_id", "date_confirmed", "date_turn_id",
                "time_confirmed", "time_turn_id", "address_or_location_confirmed",
                "address_or_location_turn_id", "fake_booking_reason",
            ],
            "properties": {
                "genuine": {"type": "boolean"},
                "commitment_type": {"enum": [
                    "date_time_confirmed", "date_only", "existing_booking_confirmed",
                    "vague_promise", "declined", "not_discussed", "no_customer_agreement",
                ]},
                "customer_agreement_confirmed": {"type": "boolean"},
                "agreement_turn_id": {"type": "string"},
                "final_customer_stance": {"enum": [
                    "confirmed", "conditional", "declined", "cancelled",
                    "disputed", "not_discussed", "unclear",
                ]},
                "final_stance_turn_id": {"type": "string"},
                "date_confirmed": {"type": "boolean"},
                "date_turn_id": {"type": "string"},
                "time_confirmed": {"type": "boolean"},
                "time_turn_id": {"type": "string"},
                "address_or_location_confirmed": {"type": "boolean"},
                "address_or_location_turn_id": {"type": "string"},
                "fake_booking_reason": {"enum": [
                    "none", "explicit_decline", "explicit_cancellation",
                    "customer_disputed_booking", "no_customer_agreement",
                    "no_booking_discussion", "conditional_or_vague_booking",
                ]},
            },
        }

    return SYSTEM_PROMPT, user_prompt, schema
