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

    def run_score(self, ratings, **kwargs):
        return score_competencies(ratings, self.specs, call_type="BOOKING", **kwargs)

    def test_remarks_cannot_be_na(self):
        with self.assertRaisesRegex(ValueError, "N/A is not allowed"):
            self.run_score({"remarks": {"rating": "N/A", "na_reason": "x"}, "crm": {"rating": "Full"}, "service": {"rating": "Full"}})

    def test_wrong_tag_override_preserves_individual_points(self):
        result = self.run_score({"remarks": {"rating": "Full"}, "crm": {"rating": "Partial"}, "service": {"rating": "Full"}}, wrong_call_tagging=True)
        self.assertEqual(result.percentage, 0)
        self.assertEqual(result.results[0].earned, 10)
        self.assertEqual(result.results[1].earned, 5)
        self.assertEqual(result.override_reason, "WRONG_CALL_TAGGING")

    def test_partial_fraction_is_per_competency(self):
        result = self.run_score({"remarks": {"rating": "Partial"}, "crm": {"rating": "Partial"}, "service": {"rating": "Partial"}})
        self.assertEqual(result.earned, 50)  # 5 + 5 + 40

    def test_empty_na_reason_fails(self):
        with self.assertRaisesRegex(ValueError, "N/A requires a reason"):
            self.run_score({"remarks": {"rating": "Full"}, "crm": {"rating": "Full"}, "service": {"rating": "N/A", "na_reason": " "}})

    def test_non_applicable_call_type_requires_na(self):
        specs = [CompetencySpec("booking_only", 100, allow_na=True, applicable_call_types=("BOOKING",))]
        with self.assertRaisesRegex(ValueError, "must be N/A for ESCALATION"):
            score_competencies({"booking_only": {"rating": "Full"}}, specs, call_type="ESCALATION")


if __name__ == "__main__":
    unittest.main()
