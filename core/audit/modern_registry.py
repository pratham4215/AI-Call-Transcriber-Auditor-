"""Load and validate the approved/draft modern competency registry."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .modern_scoring import CompetencySpec, validate_12_competency_registry

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "resources" / "modern_competency_registry.json"


def load_registry(path: Path = REGISTRY_PATH) -> Tuple[Dict[str, Any], List[CompetencySpec]]:
    """Read registry JSON and produce scoring specs after strict validation."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("total_points") != 100:
        raise ValueError("registry total_points must be 100")
    rows = raw.get("competencies")
    if not isinstance(rows, list):
        raise ValueError("registry competencies must be an array")
    specs = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each competency registry row must be an object")
        specs.append(CompetencySpec(
            competency_id=row["id"],
            weight=float(row["weight"]),
            partial_fraction=float(row.get("partial_fraction", 0.5)),
            allow_na=bool(row.get("allow_na", False)),
            critical=bool(row.get("critical", False)),
            applicable_call_types=tuple(row.get("applicable_call_types", raw["call_types"])),
        ))
    validate_12_competency_registry(specs)
    ids = [row.get("id") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("registry competency IDs must be unique")
    return raw, specs
