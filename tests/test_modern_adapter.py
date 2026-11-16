"""End-to-end modern adapter tests."""
import unittest

from core.audit.modern_registry import load_registry
from core.audit.modern_adapter import evaluate_payload


def payload(rating="Full"):
    _, specs = load_registry()
    return {
        "schema_version": "modern-audit-evaluation-v1",
        "call_type": "BOOKING",
        "competencies": [{
            "id": s.competency_id,
            "rating": rating,
            "evidence": [{"source": "transcript", "reference": "T-1", "quote_or_fact": "Relevant evidence."}],
            "reasoning": "Test rationale.",
            "confidence": "High",
            "human_review": False,
        } for s in specs],
        "human_review": {"required": False, "reasons": []},
    }


class ModernAdapterTests(unittest.TestCase):
    def test_full_payload_scores_100(self):
        result = evaluate_payload(payload())
        self.assertEqual(result["score"]["percentage"], 100)
        self.assertEqual(len(result["competencies"]), 12)
        self.assertEqual(result["registry_version"], "modern-cre-audit-12-v1")
        self.assertTrue(result["human_review"]["required"])
        self.assertTrue(any("deterministic CRM check inputs" in reason for reason in result["human_review"]["reasons"]))

    def test_wrong_tag_overrides_final_but_preserves_items(self):
        result = evaluate_payload(payload(), wrong_call_tagging=True)
        self.assertEqual(result["score"]["percentage"], 0)
        self.assertEqual(result["competencies"][0]["earned"], result["competencies"][0]["weight"])
        self.assertTrue(result["human_review"]["required"])

    def test_deterministic_crm_failure_overrides_model_rating(self):
        checks = {
            "booking": "material_failure", "follow_up": "not_applicable",
            "snr": "not_applicable", "escalation": "not_applicable",
            "cre_remarks": "pass", "customer_remarks": "not_applicable",
        }
        result = evaluate_payload(payload(), crm_integrity_checks=checks)
        crm = next(item for item in result["competencies"] if item["id"] == "crm_documentation_integrity")
        self.assertEqual(crm["rating"], "Zero")
        self.assertEqual(crm["earned"], 0)
        self.assertEqual(result["score"]["percentage"], 90)
        self.assertEqual(result["crm_integrity_diagnostics"]["source"], "deterministic_crm_rules")
        self.assertTrue(result["human_review"]["required"])

    def test_recording_zero_sets_compliance_flag(self):
        p = payload()
        next(item for item in p["competencies"] if item["id"] == "recording_disclosure")["rating"] = "Zero"
        result = evaluate_payload(p)
        self.assertTrue(result["compliance_flags"]["recording_disclosure_failure"])
        self.assertEqual(result["compliance_flags"]["recording_disclosure_rating"], "Zero")

    def test_incomplete_competency_set_rejected(self):
        p = payload()
        p["competencies"].pop()
        with self.assertRaisesRegex(ValueError, "competency ID mismatch"):
            evaluate_payload(p)

    def test_low_confidence_routes_review(self):
        p = payload()
        p["competencies"][0]["confidence"] = "Low"
        result = evaluate_payload(p)
        self.assertTrue(result["human_review"]["required"])
        self.assertTrue(any("Low confidence competency" in r for r in result["human_review"]["reasons"]))


if __name__ == "__main__":
    unittest.main()
