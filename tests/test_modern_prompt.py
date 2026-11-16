"""Tests for the one-pass modern prompt builder."""
import json
import unittest

from core.audit.modern_prompt import build_modern_evaluation_prompt


class ModernPromptTests(unittest.TestCase):
    def test_transcript_inserted_once_and_weights_hidden(self):
        turns = [
            {"index": 1, "speaker": "Agent", "text": "Namaste, Kataria thi bolu chu."},
            {"index": 2, "speaker": "Customer", "text": "Haan bolo."},
        ]
        system, user, schema = build_modern_evaluation_prompt(turns, "BOOKING", {"warranty": "active"})
        self.assertEqual(user.count("=== COMPLETE TRANSCRIPT"), 1)
        self.assertEqual(user.count("T001 [Agent]:"), 1)
        self.assertEqual(user.count("T002 [Customer]:"), 1)
        self.assertNotIn("weight_not_for_model", user)
        self.assertNotIn('"weight":', user)
        self.assertIn("evaluate the entire conversation once", system.lower())
        self.assertEqual(schema["properties"]["call_type"]["const"], "BOOKING")
        self.assertEqual(schema["properties"]["competencies"]["minItems"], 12)
        json.dumps(schema)  # schema is JSON serializable

    def test_unsupported_call_type_rejected(self):
        with self.assertRaises(ValueError):
            build_modern_evaluation_prompt([], "DMS")


if __name__ == "__main__":
    unittest.main()
