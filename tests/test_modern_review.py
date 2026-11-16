"""Tests for deterministic human-review routing."""
import unittest

from core.audit.modern_review import review_required


class ModernReviewTests(unittest.TestCase):
    def test_low_confidence_and_integrity_flags_route_review(self):
        result = review_required({
            "competencies": [{"id": "recording", "rating": "Zero", "confidence": "Low", "human_review": False}],
            "human_review": {"required": False, "reasons": []},
        }, wrong_call_tagging=True, critical_crm_inconsistency=True)
        self.assertTrue(result["required"])
        self.assertIn("Wrong call tagging / CRM disposition mismatch", result["reasons"])
        self.assertIn("Low confidence competency: recording", result["reasons"])

    def test_review_reasons_deduplicated(self):
        result = review_required({
            "competencies": [{"id": "x", "rating": "N/A", "na_reason": "", "confidence": "High", "human_review": True}],
            "human_review": {"required": True, "reasons": ["Manual check", "Manual check"]},
        }, uncertain_na=True)
        self.assertEqual(len(result["reasons"]), len(set(result["reasons"])))
        self.assertTrue(result["required"])

    def test_clean_evaluation_does_not_route_review(self):
        result = review_required({
            "competencies": [{"id": "x", "rating": "Full", "confidence": "High", "human_review": False}],
            "human_review": {"required": False, "reasons": []},
        })
        self.assertFalse(result["required"])
        self.assertEqual(result["reasons"], [])


if __name__ == "__main__":
    unittest.main()
