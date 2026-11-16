"""Regression tests for the modern scoring edge cases."""
import unittest
from core.audit.modern_scoring import CompetencySpec, score_competencies


class ModernScoringEdgeTests(unittest.TestCase):
    def setUp(self):
        self.specs = [
            CompetencySpec("remarks", 10, allow_na=False),
            CompetencySpec("crm", 10, allow_na=False),
            CompetencySpec("service", 80, allow_na=True),
        ]

    def test_remarks_cannot_be_na(self):
        with self.assertRaisesRegex(ValueError, "N/A is not allowed"):
            score_competencies({
                "remarks": {"rating": "N/A", "na_reason": "x"},
                "crm": {"rating": "Full"},
                "service": {"rating": "Full"},
            }, self.specs)

    def test_wrong_tag_override_preserves_individual_points(self):
        result = score_competencies({
            "remarks": {"rating": "Full"},
            "crm": {"rating": "Partial"},
            "service": {"rating": "Full"},
        }, self.specs, wrong_call_tagging=True)
        self.assertEqual(result.percentage, 0)
        self.assertEqual(result.results[0].earned, 10)
        self.assertEqual(result.results[1].earned, 5)
        self.assertEqual(result.override_reason, "WRONG_CALL_TAGGING")

    def test_partial_fraction_is_per_competency(self):
        result = score_competencies({
            "remarks": {"rating": "Partial"},
            "crm": {"rating": "Partial"},
            "service": {"rating": "Partial"},
        }, self.specs)
        self.assertEqual(result.earned, 50)  # 5 + 5 + 40

    def test_empty_na_reason_fails(self):
        with self.assertRaisesRegex(ValueError, "N/A requires a reason"):
            score_competencies({
                "remarks": {"rating": "Full"},
                "crm": {"rating": "Full"},
                "service": {"rating": "N/A", "na_reason": " "},
            }, self.specs)


if __name__ == "__main__":
    unittest.main()
