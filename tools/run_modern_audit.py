#!/usr/bin/env python3
"""Run the isolated modern audit path on one transcript (experimental)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.audit.llm import make_client
from core.audit.pipeline import (
    determine_audit_type_deterministic,
    load_crm_details,
    parse_transcript,
    read_transcript,
)
from core.audit.modern_runner import run_modern_audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", type=Path, help="One transcript file")
    parser.add_argument("--crm-snapshot", type=Path, default=None, help="CRM JSON sidecar")
    parser.add_argument("--crm-integrity-checks", type=Path, default=None, help="JSON with six deterministic CRM integrity check states")
    parser.add_argument("--backend", choices=["ollama", "transformers", "openai"], default="ollama")
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", type=Path, default=Path("reports/modern"))
    parser.add_argument("--wrong-call-tagging", action="store_true")
    parser.add_argument("--fake-booking", action="store_true")
    parser.add_argument("--critical-crm-inconsistency", action="store_true")
    parser.add_argument("--contradictory-evidence", action="store_true")
    parser.add_argument("--uncertain-na", action="store_true")
    parser.add_argument("--low-confidence-critical-decision", action="store_true")
    args = parser.parse_args()

    if not args.transcript.is_file():
        parser.error(f"transcript file not found: {args.transcript}")
    if args.crm_snapshot and not args.crm_snapshot.is_file():
        parser.error(f"CRM snapshot not found: {args.crm_snapshot}")
    if args.crm_integrity_checks and not args.crm_integrity_checks.is_file():
        parser.error(f"CRM integrity checks file not found: {args.crm_integrity_checks}")

    raw = read_transcript(args.transcript)
    turns = parse_transcript(raw)
    crm, _context_path = load_crm_details(args.transcript, args.crm_snapshot)
    crm = crm or {}
    call_type, route_evidence, route_confidence = determine_audit_type_deterministic(
        args.transcript, crm
    )
    if call_type is None:
        print(f"Call type unresolved; manual review. Evidence: {route_evidence}", file=sys.stderr)
        return 2

    crm_integrity_checks = None
    if args.crm_integrity_checks:
        crm_integrity_checks = json.loads(args.crm_integrity_checks.read_text(encoding="utf-8"))
        if not isinstance(crm_integrity_checks, dict):
            parser.error("CRM integrity checks JSON must be an object")

    client = make_client(args.backend, args.model)
    result = run_modern_audit(
        client, turns, call_type, crm,
        crm_integrity_checks=crm_integrity_checks,
        wrong_call_tagging=args.wrong_call_tagging,
        fake_booking=args.fake_booking,
        critical_crm_inconsistency=args.critical_crm_inconsistency,
        contradictory_evidence=args.contradictory_evidence,
        uncertain_na=args.uncertain_na,
        low_confidence_critical_decision=args.low_confidence_critical_decision,
    )
    result["metadata"] = {
        "transcript": args.transcript.name,
        "call_type": call_type,
        "routing_evidence": route_evidence,
        "routing_confidence": route_confidence,
        "runner": "modern_one_pass_experimental",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / f"{args.transcript.stem}.modern-audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Modern score: {result['score']['percentage']:.2f}/100")
    print(f"Human review: {result['human_review']['required']}")
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
