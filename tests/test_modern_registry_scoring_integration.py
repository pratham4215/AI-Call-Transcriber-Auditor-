"""End-to-end test from checked-in registry through deterministic scoring."""
import unittest

from core.audit.modern_registry import load_registry
from core.audit.modern_scoring import score_competencies


class ModernRegistryScoringIntegrationTests(unittest.TestCase):
    def test_registry_full_marks_produce_100(self):
        raw, specs = load_registry()
        ratings = {spec.competency_id: {"rating": "Full"} for spec in specs}
        result = score_competencies(ratings, specs, call_type="BOOKING")
        self.assertEqual(result.possible, raw["total_points"])
        self.assertEqual(result.earned, 100)
        self.assertEqual(result.percentage, 100)
        self.assertFalse(result.overridden)

    def test_registry_zero_marks_produce_zero_without_override(self):
        _, specs = load_registry()
        ratings = {spec.competency_id: {"rating": "Zero"} for spec in specs}
        result = score_competencies(ratings, specs, call_type="FOLLOW_UPS")
        self.assertEqual(result.earned, 0)
        self.assertEqual(result.percentage, 0)
        self.assertFalse(result.overridden)


if __name__ == "__main__":
    unittest.main()
