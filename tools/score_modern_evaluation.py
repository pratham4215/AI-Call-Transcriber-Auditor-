#!/usr/bin/env python3
"""Score a saved modern competency evaluation JSON without loading ASR/LLM."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.audit.modern_adapter import evaluate_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_json", type=Path, help="JSON matching modern-audit-evaluation-v1")
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path (default: stdout)")
    parser.add_argument("--wrong-call-tagging", action="store_true")
    parser.add_argument("--fake-booking", action="store_true")
    parser.add_argument("--critical-crm-inconsistency", action="store_true")
    parser.add_argument("--contradictory-evidence", action="store_true")
    parser.add_argument("--uncertain-na", action="store_true")
    parser.add_argument("--low-confidence-critical-decision", action="store_true")
    args = parser.parse_args()

    try:
        payload = json.loads(args.evaluation_json.read_text(encoding="utf-8"))
        result = evaluate_payload(
            payload,
            wrong_call_tagging=args.wrong_call_tagging,
            fake_booking=args.fake_booking,
            critical_crm_inconsistency=args.critical_crm_inconsistency,
            contradictory_evidence=args.contradictory_evidence,
            uncertain_na=args.uncertain_na,
            low_confidence_critical_decision=args.low_confidence_critical_decision,
        )
    except (OSError, json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
        print(f"Modern evaluation rejected: {exc}", file=sys.stderr)
        return 2

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
        print(f"Modern score: {result['score']['percentage']:.2f}/100")
        print(f"Human review: {result['human_review']['required']}")
        print(f"Output: {args.out}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
