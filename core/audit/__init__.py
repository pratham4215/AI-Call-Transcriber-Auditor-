"""
core/audit — Split audit package.

Import the pieces you need directly from their submodules, or use this
__init__ for the most-used public API.

  from core.audit.params   import PARAM_LABELS, PARAM_SCORING_NOTES, WEIGHTS
  from core.audit.prompts  import build_user_prompt, build_classification_prompt
  from core.audit.rules    import compute_param_applicability, apply_business_rule_overrides
  from core.audit.scoring  import score_audit, grade_band, build_management_summary
  from core.audit.excel_report import append_audit_report
  from core.audit.llm      import OllamaClient, make_client, extract_json
  from core.audit.pipeline import process_one, classify_call
"""
from .llm import OllamaClient, TransformersClient, extract_json, make_client
from .params import PARAM_LABELS, PARAM_SCORING_NOTES, WEIGHTS
from .prompts import build_classification_prompt, build_name_prompt, build_user_prompt
from .excel_report import append_audit_report, build_non_scored_summary
from .rules import (
    apply_business_rule_overrides,
    build_applicability_note,
    compute_param_applicability,
    normalize_result_payload,
    resolve_audit_route,
)
from .scoring import build_management_summary, grade_band, score_audit
from .utils import mask_phone, redact_call_id
from .crm_excel import get_crm_by_phone, get_crm_by_stem, reload as reload_crm_excel

_PIPELINE_EXPORTS = {
    "process_one", "classify_call", "extract_names", "verify_booking_genuine",
}


def __getattr__(name):
    """Load pipeline entry points lazily so ``python -m core.audit.pipeline`` is safe."""
    if name in _PIPELINE_EXPORTS:
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "PARAM_LABELS", "PARAM_SCORING_NOTES", "WEIGHTS",
    "build_user_prompt", "build_classification_prompt", "build_name_prompt",
    "compute_param_applicability", "apply_business_rule_overrides",
    "resolve_audit_route", "normalize_result_payload", "build_applicability_note",
    "score_audit", "grade_band", "build_management_summary",
    "append_audit_report", "build_non_scored_summary",
    "OllamaClient", "TransformersClient", "make_client", "extract_json",
    "process_one", "classify_call", "extract_names", "verify_booking_genuine",
    "mask_phone", "redact_call_id",
    "get_crm_by_phone", "get_crm_by_stem", "reload_crm_excel",
]
