# Modern Audit Migration — Working Baseline

**Working branch:** `modern-audit-work`  
**Baseline reviewed:** `main` (repository state inspected through GitHub)

This document records repository facts and the implementation sequence. It is not a replacement scorecard and does not silently change production scoring.

## Current architecture (verified)

- `core/audit/params.py`: parameter labels, grading notes, fallback weights, call-type mappings and applicability helpers.
- `resources/call_type_parameters.xlsx`: live score weights when found; fallback weights in Python are not necessarily active.
- `core/audit/prompts.py`, `single_pass_prompts.py`: current prompt paths.
- `core/audit/single_pass_validator.py`: validator exists; project analysis reports validation is currently disabled in pipeline.
- `core/audit/rules.py`: deterministic rules and CRM-backed evaluation.
- `core/audit/lora_audit.py`: optional LoRA ratings.
- `core/audit/pipeline.py`: orchestration, call routing, model calls, booking integrity and result assembly.
- `core/audit/scoring.py`: weighted score calculation and management summary.
- Current code has parameter IDs across call types; this is not yet equivalent to a unified 12-competency scorecard.

## Migration principles

1. Keep transcription, diarization, CRM import and report/dashboard interfaces stable unless a verified defect requires a change.
2. Evaluate each competency against the complete available conversation, not a rigid phrase/position checklist.
3. Every model rating must carry transcript/CRM evidence and a concise rationale.
4. Keep applicability explicit: N/A is allowed only where the competency's rule permits it, with a reason.
5. Keep point arithmetic, weights, critical overrides and score aggregation deterministic in Python.
6. Preserve call-type-specific requirements and CRM checks; do not infer missing CRM facts from transcript.
7. A critical integrity override changes the final call score but must not erase individual competency ratings.
8. Low confidence, contradictory evidence, uncertain applicability and critical integrity cases must be reviewable by a human.
9. Do not claim accuracy or readiness without a representative human-labeled validation set.

## Implementation sequence

- [x] Freeze the 12-competency / 100-point modern weights in `resources/modern_competency_registry.json` (legacy workbook remains unchanged).
- [x] Define a versioned evaluation JSON contract (`schemas/modern-audit-evaluation-v1.schema.json`).
- [x] Add validation for required IDs, rating enums, evidence/T-IDs, N/A reason and confidence.
- [x] Implement deterministic scoring from registry, N/A, call-type applicability and integrity overrides.
- [~] Experimental one-call runner + transcript CLI added; legacy `process_one()` is intentionally not switched yet.
- [x] Unit tests for score math, applicability, invalid output, CRM integrity, review and runner (CI passing).
- [~] Synthetic all-Full/CRM fixtures and adapter tests added; real fixtures for all four call types remain pending.
- [ ] Run a human-audit comparison pilot; report agreement and error categories (not started).
- [ ] Only then enable the new scorecard for production traffic (blocked by pilot/review).

## Safety gates

- Do not replace live workbook weights until the approved scorecard is reconciled.
- Do not treat a model's self-reported confidence as calibrated probability.
- Do not automatically convert malformed or missing ratings into a favorable score.
- Do not deploy a new scorecard before tests and human comparison pass.
