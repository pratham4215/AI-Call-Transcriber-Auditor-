#!/usr/bin/env python3
"""Quick smoke check for the modern registry and deterministic score engine."""
from __future__ import annotations

import sys
from pathlib import Path

# Allow execution as `python tools/validate_modern_registry.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.audit.modern_registry import load_registry
from core.audit.modern_scoring import score_competencies


def main() -> int:
    raw, specs = load_registry()
    all_full = {s.competency_id: {"rating": "Full"} for s in specs}
    all_zero = {s.competency_id: {"rating": "Zero"} for s in specs}
    full = score_competencies(all_full, specs, call_type="BOOKING")
    zero = score_competencies(all_zero, specs, call_type="BOOKING")
    assert len(specs) == 12, f"expected 12 competencies, got {len(specs)}"
    assert sum(s.weight for s in specs) == 100, "weights must total 100"
    assert full.percentage == 100, f"all Full should score 100, got {full.percentage}"
    assert zero.percentage == 0, f"all Zero should score 0, got {zero.percentage}"
    print(f"Registry: {raw['registry_version']} ({len(specs)} competencies / {full.possible:g} points)")
    print(f"Smoke scoring: all Full={full.percentage:.2f}; all Zero={zero.percentage:.2f}")
    print("Registry/scoring smoke check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
