"""Introspection tests: memory vs training distinction, honest capability."""

from __future__ import annotations

import unittest

from lucy_edge.services import build_services

from .helpers import make_config, temp_dir


class IntrospectionTests(unittest.IsolatedAsyncioTestCase):
    async def _report(self, tmp=None, **kwargs):
        services = build_services(make_config(tmp or temp_dir(), **kwargs))
        await services.open()
        try:
            return await services.introspection.report()
        finally:
            await services.close()

    async def test_distinguishes_memory_from_training(self):
        report = await self._report()
        classification = report["capability_classification"]
        self.assertEqual(classification["APPLICATION_MEMORY"], "AVAILABLE")
        self.assertEqual(classification["CONTEXT_WINDOW"], "UNKNOWN")
        self.assertEqual(classification["CONFIGURATION_EVOLUTION"], "UNAVAILABLE")
        self.assertEqual(classification["MODEL_WEIGHT_TRAINING"], "UNAVAILABLE")
        self.assertEqual(classification["MODEL_WEIGHT_MODIFICATION"], "UNAVAILABLE")
        self.assertEqual(report["training"]["weight_training"], "UNAVAILABLE")
        self.assertIsNone(report["training"]["training_library"])

    async def test_phone_local_inference_blocked(self):
        report = await self._report()
        self.assertTrue(report["inference"]["local_inference_blocked"])
        self.assertFalse(report["inference"]["phone_local_inference_enabled"])

    async def test_persistent_memory_reported_as_available(self):
        report = await self._report()
        self.assertTrue(report["memory"]["persistent_memory_available"])
        self.assertEqual(report["memory"]["memory_backend"], "sqlite")

    async def test_planner_backend_honest(self):
        report = await self._report()
        self.assertEqual(report["agent"]["planner_backend"], "RULE_BASED")
        self.assertFalse(report["agent"]["planner_is_model_driven"])
        self.assertTrue(report["agent"]["bounded"])

    async def test_mcp_not_available(self):
        report = await self._report()
        self.assertFalse(report["mcp"]["available"])

    async def test_enhancements_answer_comes_from_evidence(self):
        services = build_services(make_config(temp_dir()))
        await services.open()
        try:
            answer = await services.introspection.answer_enhancements()
            self.assertTrue(answer["persistent_memory_available"])
            self.assertTrue(answer["git_tool_available"])
            self.assertTrue(answer["phone_local_inference_blocked"])
            self.assertFalse(answer["weight_training_available"])
            self.assertFalse(answer["mcp_available"])
            self.assertIn("runtime evidence", answer["source"])
        finally:
            await services.close()


if __name__ == "__main__":
    unittest.main()
