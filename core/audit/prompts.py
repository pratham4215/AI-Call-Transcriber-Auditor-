"""
prompts.py — All LLM system prompts and user-prompt builders.

This is the primary file to edit when you want to change what the LLM is told.

  Prompt               Used for
  ─────────────────────────────────────────────────────────────────
  SYSTEM_PROMPT                  Main audit scoring call
  CLASSIFICATION_SYSTEM_PROMPT   Call-type classification
  NAME_SYSTEM_PROMPT             CRE / customer name extraction
  BOOKING_GENUINE_SYSTEM_PROMPT  Booking genuineness check

  build_user_prompt()            Builds the per-call scoring user message
  build_classification_prompt()  Builds the classification user message
  build_name_prompt()            Builds the name-extraction user message
"""
from __future__ import annotations

import json
import textwrap
from typing import Any, Dict, Optional

from .params import (
    CALL_TYPE_LABELS,
    PARAM_LABELS,
    PARAM_SCORING_NOTES,
    SCORE_OPTIONS,
    WEIGHTS,
    _extract_cre_disposition,
)
from .rules import build_applicability_note


# ─────────────────────────────────────────────────────────────────────────────
# CALL-TYPE GUIDANCE  (used inside build_classification_prompt)
# ─────────────────────────────────────────────────────────────────────────────
CALL_TYPE_GUIDANCE = """
Classify the transcript into exactly ONE transcript disposition:

1. BOOKING
   A new authorized-dealership service appointment is secured DURING this call.
   The customer agrees to service and a usable booking commitment is made, such
   as a date, time, pickup arrangement, or clear appointment confirmation.

2. SERVICE_NT_REQUIRED
   The customer says authorized service is not required now. This includes
   service already completed locally or at another authorized workshop, vehicle
   sold, not due, low usage, refusal, or another clear reason for not taking the
   company's service from this call.

3. FOLLOW_UPS
   No booking or completed authorized service is confirmed. The customer asks
   for a later callback, postpones the decision, needs time, or leaves a specific
   pending next contact/action.

4. ESCALATION
   The conversation centers on dissatisfaction, unresolved complaint, dispute,
   serious billing/service concern, or a request/need for manager-level action.
   A routine price question alone is not Escalation unless it becomes a dispute
   or remains unresolved and requires higher intervention.

5. DND
   The customer explicitly asks the company/CRE not to call again or requests
   removal from future calling. Do not infer DND from ordinary refusal or anger.

Decision rules:
- Classify from the final confirmed situation, using the whole conversation.
- BOOKING is only for an appointment newly secured in this call.
- Already serviced locally or at another authorized workshop means
  SERVICE_NT_REQUIRED in the current transcript.
- Never output any label outside the five transcript dispositions listed above.
- Do not classify from keywords alone; explain the decisive transcript evidence.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a meticulous call-quality auditor for an automobile service centre. "
    "You audit CRE (Customer Relationship Executive) outbound calls. Transcripts "
    "are multilingual (Gujarati, Hindi, English mixed); you understand all three. "
    "Judge only from evidence in the transcript and explicitly supplied CRM/DMS "
    "context. Never invent facts. Use the "
    "lowest defensible rating when evidence is weak or incomplete. Output STRICT "
    "JSON only."
)

CLASSIFICATION_SYSTEM_PROMPT = (
    "You are the call-type classification stage of an automobile service quality "
    "audit. The transcript may mix Gujarati, Hindi and English. Determine the "
    "single final call type using the supplied business definitions. Do not score "
    "the call. Never invent an outcome. Output strict JSON only."
)

NAME_SYSTEM_PROMPT = (
    "You are an expert multilingual information extraction system for automobile "
    "dealership call audits. Your ONLY job is to extract the AGENT (CRE) name and "
    "the CUSTOMER name from the transcript with maximum factual accuracy. The "
    "transcript may be Gujarati, Hindi, English or code-switched. Never guess, "
    "never infer, never complete partial names. Only extract names explicitly "
    "spoken. Returning 'Not identified' is ALWAYS better than hallucinating. "
    "Output STRICT JSON only."
)

BOOKING_GENUINE_SYSTEM_PROMPT = (
    "You are a booking-validation specialist for an automotive dealership. "
    "Your ONLY task is to determine whether a genuine service booking was confirmed "
    "during this call. Output STRICT valid JSON only — no prose, no markdown.\n\n"
    "A booking IS genuine when ALL of the following are true:\n"
    "  1. The customer explicitly agreed to bring the vehicle in for service.\n"
    "  2. A specific date, time slot, or a clear commitment to follow up with one was made.\n"
    "  3. Neither party ended the call without a firm commitment (customer did not say "
    "     'I'll think about it', 'maybe later', 'call me again', or equivalent).\n\n"
    "A booking is NOT genuine when any of the following apply:\n"
    "  - The customer declined, deferred, or gave a non-committal response.\n"
    "  - The CRE noted a booking in their system but the customer never confirmed.\n"
    "  - The call ended with only a vague promise to follow up, no booking confirmed.\n"
    "  - The appointment discussed was cancelled, rescheduled indefinitely, or disputed.\n\n"
    "Respond with exactly this JSON shape:\n"
    '{"genuine": true | false, "evidence": "<one sentence quoting or summarising the '
    'decisive transcript moment>"}'
)


# ─────────────────────────────────────────────────────────────────────────────
# NAME EXTRACTION RULES  (appended to NAME_SYSTEM_PROMPT user message)
# ─────────────────────────────────────────────────────────────────────────────
NAME_RULES = """
AGENT (CRE) NAME - usually introduced by the caller. Look for:
  "મારું નામ ___", "હું ___ વાત કરું છું", "Kataria Automobilesથી ___ વાત કરું છું",
  "This is ___ calling from...", "My name is ___".

CUSTOMER NAME - extract ONLY if the agent explicitly addresses or confirms it.
  e.g. "યોગેશ સર વાત કરો છો?" -> Customer = Yogesh
       "Am I speaking with Mr. Patel?" -> Customer = Patel

DO:
  - Strip honorifics (સર/ભાઈ/Mr.): "યોગેશ સર" -> Yogesh ; "Mr. Shah" -> Shah
  - Keep both first and last name if both are spoken.
  - Use transcript evidence ONLY.

DO NOT:
  - Invent/infer a name, or guess from gender.
  - Use the dealership name (Kataria) as the CRE name.
  - Use a vehicle model as the customer name, or speaker labels as names.

AMBIGUOUS -> "Not identified": partial name, only initials, or confidence < 95%.
"""

# ─────────────────────────────────────────────────────────────────────────────
# SCORING STANDARD  (included in build_user_prompt)
# ─────────────────────────────────────────────────────────────────────────────
_SCORING_STANDARD = textwrap.dedent("""
    SCORING STANDARD
    - Score EVERY listed parameter using only that parameter's configured levels.
    - Full: complete, timely, accurate and clearly evidenced performance.
    - Good: strong but not fully complete performance (only when configured).
    - Partial: incomplete, vague, late, weak or partially correct performance
      (only when configured). Identify what was done and what was missing.
    - Zero: omitted, incorrect, contradicted, or unsupported by transcript/CRM.
    - Never invent a level that is absent from the parameter's configured options.
    - Evidence is mandatory for every rating, including Zero.
    - For any non-Zero level, cite a concise verbatim quote where possible. If using CRM
      context, name the exact CRM field/value supporting the rating.
    - For Zero, state specifically that no supporting evidence exists or quote
      the contradictory/incorrect behavior. Never leave evidence blank.
    - Judge only the listed parameters. Do not add, discuss, or score any other
      audit parameter, even if it appears relevant.
    - Do not reward the customer for supplying information that the CRE failed
      to ask, explain, confirm or handle as required.
    - For spoken-behavior parameters, the transcript must prove that the CRE
      performed the behavior. CRM data may verify accuracy but cannot replace
      missing spoken behavior.
    - Customer name, vehicle, registration, odometer, due date, service history,
      OEM, EW and MCP values are read-only CRM context visible to the CRE. Their
      presence in CRM does not prove that the CRE mentioned or used them.
    - Treat only disposition, reason, customer/CRE remarks, booking entries and
      follow-up entries as CRE-entered workflow data.
    - Use CRM data directly only for CRM controls such as remarks, tagging and
      task management.
    - For OEM, EW and MCP/warranty scoring, compare the CRE's spoken explanation
      with the customer-specific status shown in CRM. Score relevance and
      accuracy; do not require pitching an inapplicable product.
    - CRM APPLICABILITY HAS THREE STATES: active/applicable, explicitly
      inactive/inapplicable, and unknown. Null, blank or missing means unknown;
      it never means inactive.
    - If OEM, EW or MCP is active, expect only the relevant active benefit to be
      discussed when warranty_info belongs to this call type. Full requires an
      accurate and useful explanation, Partial means incomplete/vague when configured, and Zero
      means omitted or contradicted.
    - If every OEM/EW/MCP product is explicitly inactive, expired, unavailable,
      not enrolled or N/A, do not expect a pitch. Give warranty_info Full for
      correct omission and cite the exact CRM statuses as evidence. Give Zero
      if the CRE falsely presents an inactive product as active/available.
    - If some product statuses are unknown, do not penalize omission solely
      because of missing CRM data. Still penalize any unsupported or inaccurate
      claim. Explain the data limitation in evidence.
    - Apply the same principle to other conditional parameters. For example,
      do not require pickup/drop when CRM explicitly says unavailable; do not
      require objection handling when the customer raises no objection; and do
      not require dissatisfaction/conflict handling when no such trigger occurs.
      Correct non-use earns Full, with evidence stating that the trigger was
      absent. If a trigger exists, use only that parameter's configured levels.
    - Never give Full merely because CRM data is blank. Full for non-action is
      allowed only when inapplicability is explicit or the transcript clearly
      proves that the conditional trigger did not arise.
    - EARLY-LIFE VEHICLE RULE: when CRM proves both an upcoming free-service
      stage (FR1/FR2/FR3) and odometer at or below 10,000 km, the CRE need not
      narrate previous-service history or ask current KMS. For correct omission,
      rate prev_service_info and ask_kms Full and cite the CRM service type and
      odometer. Do not apply this bypass when either fact is missing.
    - DEMO REMARKS RULE: for a fresh Booking call where CRM supplies neither
      customer remarks nor CRE remarks, remarks_tagging is temporarily bypassed
      and receives Full. Evidence must begin "DEMO BYPASS" and clearly state
      that this is temporary, not proof that remarks were entered.
    - For booking/follow-up details, compare the transcript commitment with the
      dates, time, workshop, reason and remarks entered by the CRE.
    - Booking date, booking time, and booking address/location form one combined
      booking_details control: all three=Full, at least two=Partial, fewer than two=Zero.
    - Treat conflicts between transcript and CRM as evidence of an accuracy or
      process gap and describe the conflict explicitly.
    - Service-history and job-card records are background context only. Do not
      infer the current booking outcome or completed service from them. This
      project audits CRE performance and does not verify service completion.
    - Evaluate multilingual meaning, not grammar or translation quality.
    - Keep evidence factual and coaching analysis actionable.
""").strip()


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_answer_template(
    table: str,
    include_coaching: bool = True,
    auto_full: Optional[Dict[str, Any]] = None,
) -> str:
    """Return the JSON answer template the LLM must fill in for a given call type."""
    af = auto_full or {}
    audit_lines = ",\n".join(
        (
            f'    "{pid}": {{"rating": "{af[pid].get("rating", "Full")}", '
            f'"evidence": "{af[pid].get("evidence", "AUTO-SCORE: pre-determined").replace(chr(34), chr(39))}"}}'
        )
        if pid in af else
        f'    "{pid}": {{"rating": "<{"|".join(SCORE_OPTIONS[table][pid])}>", '
        f'"evidence": "<specific transcript/CRM evidence or explicit absence>"}}'
        for pid in WEIGHTS[table]
    )
    common = (
        f'{{\n'
        f'  "metadata": {{\n'
        f'    "call_type": "{table}",\n'
        f'    "language": "<primary language(s) spoken>",\n'
        f'    "cre_name": "<agent name or \'Not identified\'>",\n'
        f'    "customer_name": "<customer name or \'Not identified\'>",\n'
        f'    "vehicle_number": "<reg no or \'Not identified\'>",\n'
        f'    "vehicle_details": "<make/model/age if mentioned or \'Not identified\'>"\n'
        f'  }},\n'
        f'  "audit": {{\n'
        f'{audit_lines}\n'
        f'  }},\n'
        f'  "other_comments": "<anything notable not covered above>",\n'
        f'  "customer_analysis": {{\n'
        f'    "vehicle_details": "<...>",\n'
        f'    "core_objection": "<the customer\'s main objection / concern>",\n'
        f'    "sentiment": "<e.g. Dismissive / Neutral / Positive> - <one line>",\n'
        f'    "final_outcome": "<Booked / Refused / Deferred / Escalated / ...> - <one line>"\n'
        f'  }},\n'
        f'  "agent_performance": {{\n'
        f'    "pitch_delivery": "<how well the offer/features were presented>",\n'
        f'    "objection_handling": "<strengths AND weaknesses; what was missed>",\n'
        f'    "language_rapport": "<tone, politeness, dialect matching>",\n'
        f'    "call_closing": "<how the call was wrapped up; callback secured?>",\n'
        f'    "improvement_suggestions": ["<actionable coaching point>", "..."]\n'
        f'  }}'
    )
    if not include_coaching:
        return common + '\n}'
    coaching = (
        ',\n'
        '  "coaching_dialogue": [\n'
        '    {\n'
        '      "moment": "<brief label for the missed moment, e.g. \'Customer objected to cost\'>",\n'
        '      "cre_said": "<what the CRE actually said, or \'Not mentioned\' if omitted>",\n'
        '      "should_have_said": "<complete natural sentence in the same language(s) — exact words a top CRE would say>",\n'
        '      "why": "<specific communication principle this leverages, e.g. empathy mirroring, social proof, loss aversion, urgency framing>"\n'
        '    }\n'
        '  ]\n'
        '}'
    )
    return common + coaching


def build_classification_prompt(
    dialogue: str,
    crm_details: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the user message for the call-type classification LLM call."""
    cre_disposition = _extract_cre_disposition(crm_details or {}) or "Not supplied"
    return (
        "FIRST TASK: independently identify the transcript disposition before any "
        "weighted audit is performed.\n\n"
        f"=== TRANSCRIPT START ===\n{dialogue}\n=== TRANSCRIPT END ===\n\n"
        f"CRE-ENTERED CRM DISPOSITION: {cre_disposition}\n"
        "Use this only for the later match/mismatch comparison. Do not copy it "
        "without transcript evidence.\n\n"
        f"{CALL_TYPE_GUIDANCE}\n\n"
        "Return exactly this JSON structure:\n"
        "{\n"
        '  "call_type": "<BOOKING|SERVICE_NT_REQUIRED|FOLLOW_UPS|ESCALATION|DND>",\n'
        '  "decisive_evidence": "<short verbatim quote or precise transcript fact>",\n'
        '  "rationale": "<why this evidence satisfies that definition>",\n'
        '  "confidence": "<High|Medium|Low>"\n'
        "}\n\nOutput JSON only."
    )


def build_user_prompt(
    dialogue: str,
    table: str,
    external_context: Optional[Dict[str, Any]] = None,
    auto_full: Optional[Dict[str, Any]] = None,
    skip_coaching: bool = False,
) -> str:
    """Build the main scoring user message for a single call."""
    template = build_answer_template(table, include_coaching=not skip_coaching, auto_full=auto_full)
    auto_full = auto_full or {}
    af_pids = set(auto_full.keys())

    active_lines = "\n".join(
        f"- {pid} ({weight:g} points): {PARAM_LABELS[pid]} "
        f"Allowed levels/points: {SCORE_OPTIONS[table][pid]}. "
        f"Grading note: {PARAM_SCORING_NOTES[pid]}"
        for pid, weight in WEIGHTS[table].items()
        if pid not in af_pids
    )

    context_text = (
        json.dumps(external_context, ensure_ascii=False, indent=2)
        if external_context else "No external CRM/DMS context was supplied."
    )

    applicability_text = build_applicability_note(external_context)

    auto_full_section = ""
    if auto_full:
        af_lines = "\n".join(
            f"  - {pid}: {info.get('rating','Full')} — {info['evidence']}"
            for pid, info in auto_full.items()
        )
        auto_full_section = (
            "=== PRE-DETERMINED RATINGS (DO NOT OVERRIDE) ===\n"
            "The following parameters have been rated by deterministic CRM rules "
            "BEFORE this LLM call. You MUST use the exact rating shown and copy the "
            "supplied evidence exactly. Do NOT re-evaluate or change these ratings.\n"
            f"{af_lines}\n"
            "=== END PRE-DETERMINED RATINGS ===\n\n"
        )

    coaching_footer = ""
    if not skip_coaching:
        coaching_footer = (
            "COACHING DIALOGUE (3-5 entries, target non-Full/Zero-rated parameters only):\n"
            "Each entry must be grounded in THIS specific call — use the actual customer name,\n"
            "vehicle model, service type, and the customer's own words or specific objection.\n"
            "Write should_have_said as a complete, fluent, natural sentence in the exact language(s)\n"
            "the CRE was using — match the code-switching: Gujarati where Gujarati was spoken,\n"
            "Hindi where Hindi was spoken, English where English was spoken, mixed if the CRE mixed.\n"
            "Think like an elite call-centre trainer writing the exact phrasing that would move\n"
            "this specific customer. Name the communication principle in 'why' (e.g. empathy\n"
            "mirroring, social proof, loss aversion, urgency framing, reciprocity). Limit to\n"
            "the 3-5 most impactful gaps only.\n\n"
        )

    return (
        f"SECOND TASK: audit this already-classified call.\n\n"
        f"FIXED CALL TYPE: {table} ({CALL_TYPE_LABELS[table]})\n"
        "Do not reclassify it. Audit only the parameters listed below.\n\n"
        f"=== TRANSCRIPT START ===\n{dialogue}\n=== TRANSCRIPT END ===\n\n"
        f"=== OPTIONAL CRM/DMS CONTEXT ===\n{context_text}\n"
        f"=== END CRM/DMS CONTEXT ===\n\n"
        f"=== DETERMINISTIC APPLICABILITY CHECK ===\n{applicability_text}\n"
        f"=== END APPLICABILITY CHECK ===\n\n"
        f"{auto_full_section}"
        f"ACTIVE PARAMETERS FROM THE {CALL_TYPE_LABELS[table]} WORKBOOK SHEET:\n"
        f"{active_lines}\n\n{_SCORING_STANDARD}\n\n"
        f"Return EXACTLY this JSON structure, filling every field:\n\n{template}\n\n"
        f"{coaching_footer}"
        "Respond with the JSON object only."
    )


def build_name_prompt(dialogue: str) -> str:
    """Build the user message for the name-extraction LLM call."""
    return (
        "Extract the CRE (agent) name and the CUSTOMER name from this call.\n\n"
        f"=== TRANSCRIPT START ===\n{dialogue}\n=== TRANSCRIPT END ===\n\n"
        f"{NAME_RULES.strip()}\n\n"
        "CONFIDENCE RULE: confidence must be at least 95%; otherwise return "
        "\"Not identified\" for that field.\n\n"
        "Return VALID JSON ONLY, exactly this and nothing else:\n"
        "{\n"
        '  "cre_name": "...",\n'
        '  "customer_name": "...",\n'
        '  "cre_evidence": "<short verbatim quote>",\n'
        '  "customer_evidence": "<short verbatim quote>"\n'
        "}"
    )
