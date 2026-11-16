"""Tests that modern runner calls the model once and scores afterward."""
import json
import unittest
from pathlib import Path

from core.audit.modern_runner import run_modern_audit
from core.audit.pipeline import Turn


class FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0
        self.last_user = None
        self.last_schema = None

    def generate(self, system, user, format_schema=None, *, single_pass=False):
        self.calls += 1
        self.last_user = user
        self.last_schema = format_schema
        assert single_pass is True
        return json.dumps(self.payload)


class ModernRunnerTests(unittest.TestCase):
    def test_one_model_call_then_python_score(self):
        fixture = Path(__file__).parent / "fixtures" / "modern_evaluation_all_full.json"
        payload = json.loads(fixture.read_text(encoding="utf-8"))
        payload["call_type"] = "FOLLOW_UPS"
        payload.pop("booking_genuine", None)
        client = FakeClient(payload)
        result = run_modern_audit(
            client,
            [{"index": 1, "speaker": "Agent", "text": "Hello"}],
            "FOLLOW_UPS",
            {"vehicle": {"kms": 10000}},
        )
        self.assertEqual(client.calls, 1)
        self.assertEqual(client.last_user.count("T001 [Agent]:"), 1)
        self.assertEqual(result["score"]["percentage"], 100)
        self.assertEqual(len(result["competencies"]), 12)
        self.assertIn("competencies", client.last_schema["properties"])

    def test_booking_genuineness_reconciled_without_second_model_call(self):
        fixture = Path(__file__).parent / "fixtures" / "modern_evaluation_all_full.json"
        payload = json.loads(fixture.read_text(encoding="utf-8"))
        client = FakeClient(payload)
        turns = [{"index": i, "speaker": "Agent", "text": "Service details."} for i in range(1, 8)]
        turns.append({"index": 8, "speaker": "Customer", "text": "I confirm the appointment on 15 December."})
        result = run_modern_audit(client, turns, "BOOKING", {"booking": {"made_in_this_call": True}})
        self.assertEqual(client.calls, 1)
        self.assertTrue(result["booking_genuine"]["genuine"])
        self.assertEqual(result["score"]["percentage"], 100)

    def test_fake_booking_override_derived_from_booking_control(self):
        fixture = Path(__file__).parent / "fixtures" / "modern_evaluation_all_full.json"
        payload = json.loads(fixture.read_text(encoding="utf-8"))
        payload["booking_genuine"].update({
            "genuine": False, "commitment_type": "no_customer_agreement",
            "customer_agreement_confirmed": False, "agreement_turn_id": "",
            "final_customer_stance": "not_discussed", "final_stance_turn_id": "",
            "date_confirmed": False, "date_turn_id": "", "time_confirmed": False,
            "time_turn_id": "", "address_or_location_confirmed": False,
            "address_or_location_turn_id": "", "fake_booking_reason": "no_customer_agreement",
        })
        client = FakeClient(payload)
        turns = [{"index": i, "speaker": "Agent", "text": "Service details."} for i in range(1, 4)]
        turns.append({"index": 4, "speaker": "Customer", "text": "No appointment was agreed."})
        result = run_modern_audit(client, turns, "BOOKING")
        self.assertEqual(client.calls, 1)
        self.assertFalse(result["booking_genuine"]["genuine"])
        self.assertEqual(result["score"]["percentage"], 0)
        self.assertEqual(result["score"]["override_reason"], "FAKE_BOOKING")

    def test_invalid_model_payload_rejected(self):
        client = FakeClient({"not": "an evaluation"})
        with self.assertRaises(ValueError):
            run_modern_audit(client, [], "BOOKING")
        self.assertEqual(client.calls, 1)


if __name__ == "__main__":
    unittest.main()
