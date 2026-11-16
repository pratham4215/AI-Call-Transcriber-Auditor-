"""Tests that modern runner calls the model once and scores afterward."""
import json
import unittest
from pathlib import Path

from core.audit.modern_runner import run_modern_audit


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
        client = FakeClient(payload)
        result = run_modern_audit(
            client,
            [{"index": 1, "speaker": "Agent", "text": "Hello"}],
            "BOOKING",
            {"vehicle": {"kms": 10000}},
        )
        self.assertEqual(client.calls, 1)
        self.assertEqual(client.last_user.count("T001 [Agent]:"), 1)
        self.assertEqual(result["score"]["percentage"], 100)
        self.assertEqual(len(result["competencies"]), 12)
        self.assertIn("competencies", client.last_schema["properties"])

    def test_invalid_model_payload_rejected(self):
        client = FakeClient({"not": "an evaluation"})
        with self.assertRaises(ValueError):
            run_modern_audit(client, [], "BOOKING")
        self.assertEqual(client.calls, 1)


if __name__ == "__main__":
    unittest.main()
