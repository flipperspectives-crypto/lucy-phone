"""Provider abstraction tests (mock + ollama-client logic, no live network)."""

from __future__ import annotations

import unittest

from lucy_edge.providers.base import (
    BaseProvider,
    Capability,
    CapabilityUnavailable,
    ProviderError,
)
from lucy_edge.providers.mock import MockProvider
from lucy_edge.providers.ollama import OllamaProvider
from lucy_edge.providers.registry import ProviderRegistry

from .helpers import FakeTransport, make_ollama_online_transport


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


class OllamaProviderClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_detect_uses_version_endpoint(self):
        transport = make_ollama_online_transport()
        provider = OllamaProvider(base_url="http://10.0.0.5:11434", transport=transport)
        result = await provider.detect()
        self.assertIn(("GET", "/api/version", None), transport.calls)
        self.assertEqual(result["base_url"], "http://10.0.0.5:11434")

    async def test_endpoint_is_configurable(self):
        transport = make_ollama_online_transport()
        provider = OllamaProvider(base_url="http://192.168.1.50:11434", transport=transport)
        await provider.detect()
        self.assertEqual(provider.base_url, "http://192.168.1.50:11434")

    async def test_health_online(self):
        transport = make_ollama_online_transport()
        provider = OllamaProvider(base_url="http://host:11434", transport=transport)
        health = await provider.health()
        self.assertTrue(health.ok)
        self.assertEqual(health.version, "0.4.7")

    async def test_health_offline_on_error(self):
        transport = FakeTransport(fail_with=ProviderError("connection refused"))
        provider = OllamaProvider(base_url="http://host:11434", transport=transport)
        health = await provider.health()
        self.assertFalse(health.ok)
        self.assertEqual(health.state.value, "OFFLINE")

    async def test_list_models_parses_tags(self):
        transport = FakeTransport()
        transport.on(
            "GET",
            "/api/tags",
            {
                "models": [
                    {
                        "name": "qwen3:1.7b",
                        "digest": "abc123",
                        "details": {"family": "qwen3", "parameter_size": "1.7b"},
                    }
                ]
            },
        )
        provider = OllamaProvider(base_url="http://host:11434", transport=transport)
        models = await provider.list_models()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].name, "qwen3:1.7b")
        self.assertEqual(models[0].parameter_size, "1.7b")

    async def test_chat_posts_with_stream_false(self):
        transport = FakeTransport()
        transport.on(
            "POST",
            "/api/chat",
            {
                "model": "qwen3:1.7b",
                "message": {"content": "hello"},
                "done": True,
                "eval_count": 2,
                "model": "qwen3:1.7b",
            },
        )
        provider = OllamaProvider(base_url="http://host:11434", transport=transport)
        response = await provider.chat(
            [{"role": "user", "content": "hi"}], model="qwen3:1.7b"
        )
        method, path, payload = transport.calls[0]
        self.assertEqual((method, path), ("POST", "/api/chat"))
        self.assertFalse(payload["stream"])
        self.assertEqual(response.model, "qwen3:1.7b")

    async def test_generate_posts_and_parses(self):
        transport = FakeTransport()
        transport.on(
            "POST",
            "/api/generate",
            {
                "model": "qwen3:1.7b",
                "response": "hello world",
                "eval_count": 2,
                "eval_duration": 5_000_000,
                "digest": "sha-abc",
            },
        )
        provider = OllamaProvider(base_url="http://host:11434", transport=transport)
        result = await provider.generate("say hello", model="qwen3:1.7b")
        self.assertEqual(result.text, "hello world")
        self.assertEqual(result.eval_duration_ms, 5.0)
        self.assertEqual(result.digest, "sha-abc")

    async def test_stream_chat_with_fake_transport(self):
        transport = FakeTransport()
        transport.on(
            "POST",
            "/api/chat",
            [
                {"message": {"content": "hel"}, "done": False},
                {"message": {"content": "lo"}, "done": True},
            ],
        )
        provider = OllamaProvider(base_url="http://host:11434", transport=transport)
        chunks = [c async for c in provider.stream_chat([], model="qwen3:1.7b")]
        self.assertEqual(chunks[0].text, "hel")
        self.assertTrue(chunks[-1].done)

    async def test_unload(self):
        transport = FakeTransport()
        transport.on("POST", "/api/generate", {"done": True})
        provider = OllamaProvider(base_url="http://host:11434", transport=transport)
        self.assertTrue(await provider.unload("qwen3:1.7b"))
        self.assertEqual(transport.calls[0][2]["keep_alive"], 0)

    async def test_runtime_metrics_not_fabricated(self):
        provider = OllamaProvider(base_url="http://host:11434", transport=FakeTransport())
        metrics = await provider.runtime_metrics()
        self.assertFalse(metrics.telemetry_available)
        self.assertFalse(metrics.simulated)


class RegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_registry_register_get_names(self):
        registry = ProviderRegistry()
        registry.register(MockProvider())
        self.assertIsNotNone(registry.get("mock"))
        self.assertIn("mock", registry.names())

    async def test_healthy_summary(self):
        registry = ProviderRegistry()
        registry.register(MockProvider())
        registry.register(
            OllamaProvider(
                base_url="http://host:11434",
                transport=FakeTransport(fail_with=ProviderError("refused")),
            )
        )
        health = await registry.healthy()
        self.assertTrue(health["mock"])
        self.assertFalse(health["ollama"])

    async def test_require_unknown_provider(self):
        registry = ProviderRegistry()
        with self.assertRaises(ProviderError):
            registry.require("missing")


if __name__ == "__main__":
    unittest.main()
