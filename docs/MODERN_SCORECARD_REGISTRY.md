# Modern Scorecard Registry — Approval Required

Status: **draft / blocked for production scoring**

This file is the single source of truth format for the proposed 12-competency scorecard. The competency IDs and weights must be populated from the final approved scorecard before the modern scoring engine is wired into `pipeline.py`.

## Registry requirements

- Exactly 12 unique competency IDs.
- Total weight = 100.
- Call-type applicability explicit for Booking, Follow-up, SNR and Escalation.
- Partial fraction explicitly defined per competency (not assumed to be 50% universally).
- N/A allowed only for named competencies and explicit conditions.
- Critical status and score override policy explicit.
- Evidence requirements defined per competency.
- CRM source requirements separated from transcript evidence.
- CRE remarks mandatory; CRM & Documentation Integrity is 10 points and N/A prohibited (per previously recorded decisions).
- Customer remarks applicability follows the agreed rule; do not silently turn missing remarks into a Zero.
- DMS and cost-estimation parameter are excluded from the new scorecard.
- Fake Booking and Wrong Tag Used force final score to zero while retaining competency diagnostics.

## Required row format

| id | competency | weight | partial_fraction | call_types | allow_na | critical | evidence_policy |
|---|---|---:|---:|---|---|---|---|
| TODO | Approved name | TODO | TODO | TODO | TODO | TODO | TODO |

Do not infer the final 12 names or weights from the legacy 27-parameter scorecard. They are a separate design.
