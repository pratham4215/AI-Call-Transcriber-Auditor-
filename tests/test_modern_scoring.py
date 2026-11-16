"""Unit tests for modern deterministic scoring."""
import unittest

from core.audit.modern_scoring import CompetencySpec, score_competencies, validate_registry


def registry():
    return [
        CompetencySpec("a", 40, partial_fraction=0.5, allow_na=True),
        CompetencySpec("b", 40, partial_fraction=0.5, critical=True),
        CompetencySpec("c", 20, partial_fraction=0.25),
    ]


def score(ratings, **kwargs):
    return score_competencies(ratings, registry(), call_type="BOOKING", **kwargs)


class ModernScoringTests(unittest.TestCase):
    def test_weights_must_total_100(self):
        with self.assertRaises(ValueError):
            validate_registry([CompetencySpec("x", 99)])

    def test_full_partial_zero_math(self):
        result = score({"a": {"rating": "Full"}, "b": {"rating": "Partial"}, "c": {"rating": "Zero"}})
        self.assertEqual(result.earned, 60)
        self.assertEqual(result.possible, 100)
        self.assertEqual(result.percentage, 60)

    def test_na_requires_permission_and_reason(self):
        with self.assertRaises(ValueError):
            score({"a": {"rating": "N/A"}, "b": {"rating": "Full"}, "c": {"rating": "Full"}})
        with self.assertRaises(ValueError):
            score({"a": {"rating": "N/A", "na_reason": " "}, "b": {"rating": "Full"}, "c": {"rating": "Full"}})

    def test_na_full_credit_policy(self):
        result = score({"a": {"rating": "N/A", "na_reason": "approved condition"}, "b": {"rating": "Full"}, "c": {"rating": "Full"}})
        self.assertEqual(result.percentage, 100)

    def test_critical_zero_is_flagged_not_automatic_override(self):
        result = score({"a": {"rating": "Full"}, "b": {"rating": "Zero"}, "c": {"rating": "Full"}})
        self.assertFalse(result.overridden)
        self.assertEqual(result.critical_zero_ids, ("b",))
        self.assertEqual(result.percentage, 60)

    def test_fake_booking_override(self):
        result = score({"a": {"rating": "Full"}, "b": {"rating": "Full"}, "c": {"rating": "Full"}}, fake_booking=True)
        self.assertEqual(result.override_reason, "FAKE_BOOKING")
        self.assertEqual(result.percentage, 0)

    def test_missing_and_unknown_ids_fail(self):
        with self.assertRaises(ValueError):
            score({"a": {"rating": "Full"}})
        with self.assertRaises(ValueError):
            score({"a": {"rating": "Full"}, "b": {"rating": "Full"}, "c": {"rating": "Full"}, "extra": {"rating": "Full"}})


if __name__ == "__main__":
    unittest.main()
