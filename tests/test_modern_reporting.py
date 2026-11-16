"""Tests for modern score serialization."""
import unittest

from core.audit.modern_scoring import CompetencySpec, score_competencies
from core.audit.modern_reporting import score_to_dict


class ModernReportingTests(unittest.TestCase):
    def test_keeps_item_marks_when_score_overridden(self):
        registry = [CompetencySpec("a", 50), CompetencySpec("b", 50, critical=True)]
        score = score_competencies({
            "a": {"rating": "Full"}, "b": {"rating": "Zero"},
        }, registry, wrong_call_tagging=True)
        output = score_to_dict(score, call_type="BOOKING")
        self.assertEqual(output["score"]["percentage"], 0)
        self.assertEqual(output["competencies"][0]["earned"], 50)
        self.assertEqual(output["competencies"][1]["rating"], "Zero")
        self.assertTrue(output["human_review"]["required"])

    def test_na_reason_and_metadata_preserved(self):
        registry = [CompetencySpec("a", 100, allow_na=True)]
        score = score_competencies({
            "a": {"rating": "N/A", "na_reason": "not applicable"},
        }, registry)
        output = score_to_dict(
            score, call_type="ESCALATION",
            competency_metadata={"a": {"name": "Approved competency", "critical": True}},
        )
        item = output["competencies"][0]
        self.assertEqual(item["name"], "Approved competency")
        self.assertEqual(item["na_reason"], "not applicable")
        self.assertTrue(item["critical"])
        self.assertEqual(output["score"]["percentage"], 100)


if __name__ == "__main__":
    unittest.main()
