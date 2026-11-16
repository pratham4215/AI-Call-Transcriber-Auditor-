# Modern CRE Audit — Scoring Specification

**Scorecard:** 12 competencies, 100 points  
**Status:** Weights locked; detailed applicability/rubric pilot validation remains open.

## Locked competency weights

| # | Competency | Points |
|---:|---|---:|
| 1 | Greeting & Introduction | 8 |
| 2 | Recording Disclosure & Compliance | 6 |
| 3 | Permission to Talk | 10 |
| 4 | Customer & Vehicle Identification | 6 |
| 5 | Purpose & Context Establishment | 6 |
| 6 | Customer & Vehicle Understanding | 8 |
| 7 | Service Knowledge & Accuracy | 10 |
| 8 | Value Communication | 10 |
| 9 | Customer Interaction & Resolution | 8 |
| 10 | Customer Action & Disposition | 10 |
| 11 | CRM & Documentation Integrity | 10 |
| 12 | Professional Communication & Closure | 8 |
| | **Total** | **100** |

## Rating semantics

- **Full:** requirement fulfilled accurately and sufficiently.
- **Partial:** attempted/fulfilled but incomplete, without material failure.
- **Zero:** missing where required, materially incorrect, misleading or failed.
- **N/A:** only for a competency with an explicit allowed condition; a concrete reason is mandatory. Missing evidence or model uncertainty is not itself N/A.

Points are assigned in Python, never by the model. Partial fractions are registry-controlled. Current N/A policy credits the full weight and retains the competency in the denominator; verify this policy before production rollout.

## Special controls

### Recording Disclosure & Compliance (6)
- Full 6: clear recording disclosure near the beginning.
- Partial 3: incomplete/unclear attempted disclosure.
- Zero 0: missing, materially unclear or too late.
- N/A 6 only if the Customer interrupts before disclosure can be completed; otherwise N/A is prohibited.
- Zero raises a recording-disclosure compliance-failure flag and review signal.

### Value Communication (10)
- Full 10: accurately communicates an eligible customer benefit/product.
- Partial 5: relevant offer attempted but incomplete/unclear.
- Zero 0: eligible benefit ignored or communicated misleadingly.
- N/A 10 only if CRM/call context shows no eligible product/opportunity.
- Refusal by Customer is not itself a penalty.

### Customer Action & Disposition (10)
- Full 10 / Partial 5 / Zero 0; N/A 10 only when no action/disposition is applicable under the final rule.
- A Customer's refusal alone is not a penalty; evaluate accurate outcome capture and next-step clarity.

### CRM & Documentation Integrity (10; N/A prohibited)
- Full 10: all applicable checks accurate and complete.
- Partial 5: minor documentation defect.
- Zero 0: material CRM failure/inconsistency.
- Diagnostic checks (not additive): Booking, Follow-up, SNR, Escalation, CRE remarks, Customer remarks.
- Correct checks cannot compensate for a material failure.
- CRE remarks are mandatory on every call. Customer remarks follow the agreed applicability rule.
- DMS is excluded. CRM remarks are not edited by the auditor.

## Final-score overrides

- **Fake Booking:** final call score = 0.
- **Wrong Call Tag Used:** final call score = 0.
- Preserve each competency's original points/rating for diagnosis.
- A critical competency Zero is a review/risk flag; it is not a blanket score override unless explicitly defined above.

## Human review triggers

- Wrong call tagging or fake booking.
- Critical CRM inconsistency.
- Contradictory evidence.
- Low confidence on a critical decision.
- Uncertain N/A.
- Low-confidence or explicitly flagged competency.
- Missing deterministic CRM-integrity inputs (model-only CRM mark is provisional).

## Implementation boundary

- `resources/modern_competency_registry.json` stores the current 12-item registry and definitions.
- `core/audit/modern_prompt.py` evaluates the transcript once and emits evidence/rating JSON only.
- `core/audit/modern_scoring.py` and `crm_integrity.py` own deterministic points.
- `core/audit/modern_adapter.py` joins evaluation, scoring, integrity and review.
- `tools/run_modern_audit.py` is experimental and writes JSON; legacy production pipeline/Excel remains unchanged.
