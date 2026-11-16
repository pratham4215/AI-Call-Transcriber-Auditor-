"""Unit tests for modern deterministic scoring."""
import unittest

from core.audit.modern_scoring import CompetencySpec, score_competencies, validate_registry


def registry():
    return [
        CompetencySpec("a", 40, partial_fraction=0.5, allow_na=True),
        CompetencySpec("b", 40, partial_fraction=0.5, critical=True),
        CompetencySpec("c", 20, partial_fraction=0.25),
    ]


class ModernScoringTests(unittest.TestCase):
    def test_weights_must_total_100(self):
        with self.assertRaises(ValueError):
            validate_registry([CompetencySpec("x", 99)])

    def test_full_partial_zero_math(self):
        result = score_competencies({
            "a": {"rating": "Full"}, "b": {"rating": "Partial"},
            "c": {"rating": "Zero"},
        }, registry())
        self.assertEqual(result.earned, 60)
        self.assertEqual(result.possible, 100)
        self.assertEqual(result.percentage, 60)

    def test_na_requires_permission_and_reason(self):
        with self.assertRaises(ValueError):
            score_competencies({
                "a": {"rating": "N/A"}, "b": {"rating": "Full"},
                "c": {"rating": "Full"},
            }, registry())
        with self.assertRaises(ValueError):
            score_competencies({
                "a": {"rating": "N/A", "na_reason": "not applicable"},
                "b": {"rating": "Full"}, "c": {"rating": "Full"},
            }, registry())

    def test_na_full_credit_policy(self):
        result = score_competencies({
            "a": {"rating": "N/A", "na_reason": "approved condition"},
            "b": {"rating": "Full"}, "c": {"rating": "Full"},
        }, registry())
        self.assertEqual(result.percentage, 100)

    def test_critical_zero_is_flagged_not_automatic_override(self):
        result = score_competencies({
            "a": {"rating": "Full"}, "b": {"rating": "Zero"},
            "c": {"rating": "Full"},
        }, registry())
        self.assertFalse(result.overridden)
        self.assertEqual(result.critical_zero_ids, ("b",))
        self.assertEqual(result.percentage, 60)

    def test_fake_booking_override(self):
        result = score_competencies({
            "a": {"rating": "Full"}, "b": {"rating": "Full"},
            "c": {"rating": "Full"},
        }, registry(), fake_booking=True)
        self.assertEqual(result.override_reason, "FAKE_BOOKING")
        self.assertEqual(result.percentage, 0)

    def test_missing_and_unknown_ids_fail(self):
        with self.assertRaises(ValueError):
            score_competencies({"a": {"rating": "Full"}}, registry())
        with self.assertRaises(ValueError):
            score_competencies({
                "a": {"rating": "Full"}, "b": {"rating": "Full"},
                "c": {"rating": "Full"}, "extra": {"rating": "Full"},
            }, registry())


if __name__ == "__main__":
    unittest.main()
