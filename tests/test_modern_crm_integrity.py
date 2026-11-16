"""Tests for CRM integrity non-additive aggregation."""
import unittest

from core.audit.crm_integrity import score_crm_integrity


BASE = {
    "booking": "pass", "follow_up": "pass", "snr": "not_applicable",
    "escalation": "not_applicable", "cre_remarks": "pass",
    "customer_remarks": "not_applicable",
}


class CRMIntegrityTests(unittest.TestCase):
    def test_all_applicable_checks_pass_full_10(self):
        result = score_crm_integrity(BASE)
        self.assertEqual((result.rating, result.points, result.weight), ("Full", 10, 10))

    def test_minor_defect_partial_5(self):
        checks = dict(BASE, follow_up="minor_defect")
        result = score_crm_integrity(checks)
        self.assertEqual((result.rating, result.points), ("Partial", 5))
        self.assertEqual(result.minor_defects, ("follow_up",))

    def test_material_failure_zero_not_compensated(self):
        checks = dict(BASE, booking="material_failure", follow_up="pass", snr="pass")
        result = score_crm_integrity(checks)
        self.assertEqual((result.rating, result.points), ("Zero", 0))
        self.assertEqual(result.material_failures, ("booking",))

    def test_cre_remarks_cannot_be_na(self):
        with self.assertRaisesRegex(ValueError, "CRE remarks are mandatory"):
            score_crm_integrity(dict(BASE, cre_remarks="not_applicable"))

    def test_missing_check_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "missing CRM integrity checks"):
            score_crm_integrity({"cre_remarks": "pass"})


if __name__ == "__main__":
    unittest.main()
