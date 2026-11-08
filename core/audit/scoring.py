"""
scoring.py — Weighted /100 scoring, grade bands, and management summary.

Main entry points
─────────────────────────────────────────────────────────────────────────────
  score_audit()           — returns (sections, earned, possible, overall_pct)
  grade_band()            — converts a pct to Excellent/Good/Average/etc.
  build_management_summary() — full KPI dict used by the report and API
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .params import (
    CALL_TYPE_LABELS,
    PARAM_LABELS,
    SECTIONS,
    WEIGHTS,
    allowed_score_levels,
    points_for_level,
    normalize_call_type,
)
from .utils import _clean, _display, redact_call_id

# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL CONTROLS — a Zero here upgrades risk to High regardless of score
# ─────────────────────────────────────────────────────────────────────────────
CRITICAL_CONTROL_PARAMS = {
    "inform_recording", "warranty_info", "professionalism", "conflict_mgmt",
}

# ─────────────────────────────────────────────────────────────────────────────
# SCORE AUDIT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SectionScore:
    earned: float = 0.0
    possible: float = 0.0
    items: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def pct(self) -> Optional[float]:
        return round(100 * self.earned / self.possible, 1) if self.possible else None


def _norm_rating(raw: str, pid: str, table: str) -> str:
    r = (raw or "").strip().lower()
    aliases = {
        "full": "Full", "yes": "Full", "y": "Full", "excellent": "Full",
        "good": "Good", "level_2": "Good",
        "half": "Partial", "partial": "Partial", "part": "Partial",
        "average": "Partial", "avg": "Partial", "level_3": "Partial",
        "zero": "Zero", "no": "Zero", "n": "Zero",
    }
    level = aliases.get(r, "Zero")
    return level if level in allowed_score_levels(table, pid) else "Zero"


def score_audit(
    audit: Dict[str, Any], table: str
) -> Tuple[Dict[str, SectionScore], float, float, Optional[float]]:
    """Return (sections, total_earned, total_possible, overall_/100).

    Only parameters present in the call type's weight table are scored.
    """
    weights = WEIGHTS[table]
    sections: Dict[str, SectionScore] = {}
    total_earned = total_possible = 0.0

    for sec_name, pids in SECTIONS:
        for pid in pids:
            if pid not in weights:
                continue
            ss = sections.setdefault(sec_name, SectionScore())
            weight = weights[pid]
            question = PARAM_LABELS[pid]
            entry = audit.get(pid) or {}
            rating = _norm_rating(str(entry.get("rating", "")), pid, table)
            evidence = str(entry.get("evidence", "")).strip() or (
                "No supporting transcript or CRM/DMS evidence was supplied."
            )
            earned = points_for_level(table, pid, rating)
            ss.earned += earned
            ss.possible += weight
            total_earned += earned
            total_possible += weight
            ss.items.append({
                "id": pid, "question": question, "rating": rating,
                "evidence": evidence, "earned": round(earned, 2),
                "possible": weight, "scored": True,
            })

    overall = round(100 * total_earned / total_possible, 1) if total_possible else None
    return sections, round(total_earned, 2), round(total_possible, 2), overall


def grade_band(pct: Optional[float]) -> str:
    if pct is None:
        return "N/A"
    if pct >= 90:
        return "Excellent"
    if pct >= 75:
        return "Good"
    if pct >= 60:
        return "Average"
    if pct >= 40:
        return "Needs Improvement"
    return "Poor"


# ─────────────────────────────────────────────────────────────────────────────
# MANAGEMENT SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def _management_disposition(overall: Optional[float], confidence: str) -> str:
    if overall is None or confidence == "Low":
        return "Manual QA review required"
    if overall >= 90:
        return "Recognize performance; monitor routinely"
    if overall >= 75:
        return "Pass with targeted coaching"
    if overall >= 60:
        return "Structured coaching and re-audit"
    return "Immediate corrective coaching and re-audit"


def build_management_summary(
    call_id: str, result: Dict[str, Any], table: Optional[str]
) -> Dict[str, Any]:
    """Build the management KPI dict used by the Flask API and the CSV summary."""
    meta = result.get("metadata", {})
    route = result.get("audit_route", {})
    summary: Dict[str, Any] = {
        "call_id": redact_call_id(call_id),
        "call_type": CALL_TYPE_LABELS[normalize_call_type(meta.get("call_type"))],
        "cre_name": _display(meta.get("cre_name")),
        "customer_name": _display(meta.get("customer_name")),
        "score": None, "score_display": None, "score_pct": None,
        "grade": "Not scored", "risk": "Not assessed",
        "confidence": "Not assessed", "coverage_pct": None,
        "evidence_pct": None, "disposition": "Review call outcome only",
        "cre_disposition": route.get("cre_disposition", "Not supplied"),
        "transcript_type": route.get("transcript_type", "Not supplied"),
        "audit_scope": route.get("audit_scope", "CRE performance score only"),
        "audit_type_source": route.get("audit_type_source", "Not supplied"),
        "disposition_match": route.get("disposition_match"),
        "auditor_comment": route.get("auditor_comment", ""),
        "crm_details_status": route.get("crm_completeness", {}).get("status", "Not assessed"),
        "crm_missing_fields": ", ".join(
            route.get("crm_completeness", {}).get("missing_fields", [])
        ),
    }
    if table is None:
        return summary

    sections, total_earned, total_possible, overall = score_audit(
        result.get("audit", {}), table
    )

    # BOOKING: force overall to 0 if booking was not genuine
    booking_genuine_info = result.get("booking_genuine")
    booking_route_confirmed = not route or route.get("transcript_type") == "BOOKING"
    not_genuine = (
        table == "BOOKING"
        and booking_route_confirmed
        and isinstance(booking_genuine_info, dict)
        and not booking_genuine_info.get("genuine", True)
    )
    wrong_call_tagging = route.get("disposition_match") is False
    if not_genuine or wrong_call_tagging:
        overall = 0.0

    items = [item for section in sections.values() for item in section.items]
    scored = [item for item in items if item["scored"]]
    capacity = sum(WEIGHTS[table].values())
    coverage = round(100 * total_possible / capacity, 1) if capacity else 0.0
    with_evidence = sum(1 for item in scored if _clean(item.get("evidence")))
    evidence_pct = round(100 * with_evidence / len(scored), 1) if scored else 0.0
    if coverage >= 85 and evidence_pct >= 85:
        confidence = "High"
    elif coverage >= 65 and evidence_pct >= 65:
        confidence = "Medium"
    else:
        confidence = "Low"

    booking_integrity_warning = table == "BOOKING" and any(
        item["id"] == "booking_details" and item["rating"] == "Zero"
        for item in scored
    )
    critical_zeros = {
        item["id"] for item in scored
        if item["id"] in CRITICAL_CONTROL_PARAMS and item["rating"] == "Zero"
    }
    if overall is not None and (
        overall < 60 or {"professionalism", "conflict_mgmt"} & critical_zeros
    ):
        risk = "High"
    elif overall is not None and (overall < 75 or critical_zeros):
        risk = "Medium"
    else:
        risk = "Low"

    def _marks(value: float) -> str:
        return str(int(value)) if float(value).is_integer() else str(round(value, 2))

    summary.update({
        "score": total_earned,
        "score_display": f"{_marks(total_earned)}/{_marks(total_possible)}",
        "score_pct": overall,
        "grade": grade_band(overall), "risk": risk,
        "confidence": confidence, "coverage_pct": coverage,
        "evidence_pct": evidence_pct,
        "disposition": _management_disposition(overall, confidence),
        "raw_earned": total_earned, "raw_possible": total_possible,
        "judgeable_capacity": round(capacity, 2),
    })
    if not_genuine or wrong_call_tagging:
        summary["score"] = 0.0
        summary["score_display"] = f"0/{_marks(total_possible)}"
        summary["score_pct"] = 0.0
        summary["grade"] = "Not Genuine"
        summary["risk"] = "High"
        summary["disposition"] = (
            "BOOKING NOT GENUINE — score overridden to 0"
            if not_genuine
            else "WRONG CALL TAGGING — score overridden to 0"
        )
        summary["wrong_call_tagging"] = wrong_call_tagging
    if not_genuine:
        summary["booking_genuine"] = False
        summary["booking_genuine_evidence"] = booking_genuine_info.get("evidence", "")
    elif table == "BOOKING" and booking_route_confirmed and isinstance(booking_genuine_info, dict):
        summary["booking_genuine"] = True
        summary["booking_genuine_evidence"] = booking_genuine_info.get("evidence", "")
    if booking_integrity_warning:
        summary["booking_integrity_warning"] = (
            "POSSIBLE FAKE BOOKING — booking date, time and address/location scored 0/6. "
            "Hard-check the Customer's explicit booking agreement against the complete call."
        )
        if summary["risk"] == "Low":
            summary["risk"] = "Medium"
    return summary
