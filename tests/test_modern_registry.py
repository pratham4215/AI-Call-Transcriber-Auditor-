"""Tests for the 12-competency registry file."""
import unittest

from core.audit.modern_registry import load_registry
from core.audit.modern_scoring import validate_12_competency_registry


class ModernRegistryTests(unittest.TestCase):
    def test_registry_has_12_competencies_and_100_points(self):
        raw, specs = load_registry()
        self.assertEqual(len(specs), 12)
        self.assertEqual(sum(s.weight for s in specs), 100)
        self.assertEqual(raw["registry_version"], "modern-cre-audit-12-v1")

    def test_production_gate_rejects_wrong_count(self):
        _, specs = load_registry()
        with self.assertRaisesRegex(ValueError, "exactly 12"):
            validate_12_competency_registry(specs[:11])


if __name__ == "__main__":
    unittest.main()
