"""Provider abstraction tests (mock provider logic, no live network)."""

from __future__ import annotations

import unittest

from lucy_edge.providers.base import (
    BaseProvider,
    Capability,
    CapabilityUnavailable,
    ProviderError,
)
from lucy_edge.providers.mock import MockProvider
from lucy_edge.providers.registry import ProviderRegistry


class MockProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_mock_chat_works_and_is_simulated(self):
        provider = MockProvider()
        response = await provider.chat(
            [{"role": "user", "content": "hello"}], model="lucy:mock"
        )
        self.assertTrue(response.simulated)
        self.assertEqual(response.message, "NEXUS SAFE")
        self.assertTrue(provider.calls)

    async def test_mock_generate_and_list(self):
        provider = MockProvider()
        result = await provider.generate("hi", model="qwen3:1.7b-mock")
        self.assertEqual(result.text, "NEXUS SAFE")
        self.assertTrue(result.simulated)
        models = await provider.list_models()
        self.assertGreaterEqual(len(models), 1)
        self.assertEqual(models[0].family, "mock")

    async def test_mock_runtime_metrics_marked_synthetic(self):
        provider = MockProvider()
        metrics = await provider.runtime_metrics()
        self.assertTrue(metrics.telemetry_available)
        self.assertTrue(metrics.simulated)
        self.assertIn("synthetic", metrics.note)

    async def test_unsupported_capability_is_explicit(self):
        base = BaseProvider()
        self.assertEqual(base.capabilities(), set())
        with self.assertRaises(CapabilityUnavailable):
            await base.health()
        self.assertFalse(base.supports(Capability.CHAT))


class RegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_registry_register_get_names(self):
        registry = ProviderRegistry()
        registry.register(MockProvider())
        self.assertIsNotNone(registry.get("mock"))
        self.assertIn("mock", registry.names())

    async def test_healthy_summary(self):
        registry = ProviderRegistry()
        registry.register(MockProvider())
        health = await registry.healthy()
        self.assertTrue(health["mock"])

    async def test_require_unknown_provider(self):
        registry = ProviderRegistry()
        with self.assertRaises(ProviderError):
            registry.require("missing")


if __name__ == "__main__":
    unittest.main()
