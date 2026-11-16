"""Unit tests for the isolated modern evaluation contract."""
import unittest

from core.audit.modern_contract import validate_modern_evaluation


def valid_payload():
    return {
        "schema_version": "modern-audit-evaluation-v1",
        "call_type": "FOLLOW_UPS",
        "competencies": [{
            "id": "greeting",
            "rating": "Full",
            "evidence": [{
                "source": "transcript",
                "reference": "T-03",
                "quote_or_fact": "CRE greeted the customer.",
            }],
            "reasoning": "A clear opening greeting is present.",
            "confidence": "High",
            "human_review": False,
        }],
        "human_review": {"required": False, "reasons": []},
    }


class ModernContractTests(unittest.TestCase):
    def test_valid_payload(self):
        self.assertEqual(validate_modern_evaluation(valid_payload()), [])

    def test_na_requires_reason(self):
        payload = valid_payload()
        payload["competencies"][0]["rating"] = "N/A"
        self.assertTrue(any("na_reason is required" in e for e in validate_modern_evaluation(payload)))

    def test_duplicate_competency_ids_rejected(self):
        payload = valid_payload()
        payload["competencies"].append(dict(payload["competencies"][0]))
        self.assertTrue(any("duplicate competency id" in e for e in validate_modern_evaluation(payload)))

    def test_empty_evidence_reference_rejected(self):
        payload = valid_payload()
        payload["competencies"][0]["evidence"][0]["reference"] = ""
        self.assertTrue(any("reference must be non-empty" in e for e in validate_modern_evaluation(payload)))

    def test_review_requires_reason(self):
        payload = valid_payload()
        payload["human_review"] = {"required": True, "reasons": []}
        self.assertTrue(any("reasons cannot be empty" in e for e in validate_modern_evaluation(payload)))


if __name__ == "__main__":
    unittest.main()
