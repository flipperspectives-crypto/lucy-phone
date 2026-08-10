"""Loyalty contract tests.

Proves the loyalty contract is present, names the primary human, and
surfaces correctly in the foundation audit report.
"""

from __future__ import annotations

import unittest

from lucy_edge.foundation import LOYALTY_CONTRACT, PRIMARY_HUMAN, loyalty_report
from lucy_edge.services import build_services

from .helpers import make_config, temp_dir


class LoyaltyContractTests(unittest.TestCase):
    def test_primary_human(self):
        self.assertEqual(PRIMARY_HUMAN, "Lauren Flipo")
        self.assertEqual(LOYALTY_CONTRACT["primary_human"], "Lauren Flipo")

    def test_contract_has_duties_and_constraints(self):
        self.assertIn("duties", LOYALTY_CONTRACT)
        self.assertIn("constraints", LOYALTY_CONTRACT)
        self.assertTrue(len(LOYALTY_CONTRACT["duties"]) > 0)
        self.assertTrue(len(LOYALTY_CONTRACT["constraints"]) > 0)

    def test_truth_is_part_of_loyalty(self):
        constraints = " ".join(LOYALTY_CONTRACT["constraints"]).lower()
        self.assertIn("truth", constraints)

    def test_report_shape(self):
        report = loyalty_report()
        self.assertEqual(report["contract"], "loyalty")
        self.assertEqual(report["primary_human"], "Lauren Flipo")
        self.assertIn("duties", report)
        self.assertIn("constraints", report)


class LoyaltyInAuditTests(unittest.IsolatedAsyncioTestCase):
    async def test_audit_includes_loyalty(self):
        config = make_config(temp_dir(), phone_local_inference=True)
        services = build_services(config, transport=None, fixed_token="test-token")
        await services.open()
        try:
            audit = await services.foundation.audit()
            self.assertIn("loyalty", audit)
            self.assertEqual(audit["loyalty"]["primary_human"], "Lauren Flipo")
        finally:
            await services.close()
