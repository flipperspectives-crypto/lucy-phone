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


class OllamaTimeoutTests(unittest.TestCase):
    """The Ollama provider timeout is configurable and fail-fast on connect."""

    def test_default_request_timeout_is_120s(self):
        from lucy_edge.providers.ollama import AiohttpTransport

        transport = AiohttpTransport(base_url="http://host:11434")
        self.assertEqual(transport.timeout, 120.0)

    def test_default_connect_timeout_is_5s(self):
        from lucy_edge.providers.ollama import AiohttpTransport

        transport = AiohttpTransport(base_url="http://host:11434")
        self.assertEqual(transport.connect_timeout, 5.0)

    def test_timeout_passed_through_to_transport(self):
        from lucy_edge.providers.ollama import OllamaProvider

        provider = OllamaProvider(
            base_url="http://host:11434",
            request_timeout=90.0,
            connect_timeout=3.0,
        )
        self.assertEqual(provider._transport.timeout, 90.0)
        self.assertEqual(provider._transport.connect_timeout, 3.0)

    def test_provider_config_defaults(self):
        from lucy_edge.config import ProviderConfig

        cfg = ProviderConfig()
        self.assertEqual(cfg.request_timeout, 120.0)
        self.assertEqual(cfg.connect_timeout, 5.0)

    def test_env_var_overrides_timeout(self):
        import importlib
        import os

        from lucy_edge import config as config_mod

        os.environ["NEXUS_OLLAMA_REQUEST_TIMEOUT"] = "150"
        os.environ["NEXUS_OLLAMA_CONNECT_TIMEOUT"] = "7"
        try:
            importlib.reload(config_mod)
            cfg = config_mod.ProviderConfig()
            # Env overrides apply at load_config time, not on the model itself,
            # so verify the override mapping exists and casts correctly.
            self.assertIn("NEXUS_OLLAMA_REQUEST_TIMEOUT", config_mod._ENV_OVERRIDES)
            self.assertIn("NEXUS_OLLAMA_CONNECT_TIMEOUT", config_mod._ENV_OVERRIDES)
            path, ty = config_mod._ENV_OVERRIDES["NEXUS_OLLAMA_REQUEST_TIMEOUT"]
            self.assertEqual(path, "providers.request_timeout")
            self.assertEqual(ty, float)
        finally:
            del os.environ["NEXUS_OLLAMA_REQUEST_TIMEOUT"]
            del os.environ["NEXUS_OLLAMA_CONNECT_TIMEOUT"]
            importlib.reload(config_mod)

    def test_load_config_applies_timeout_env(self):
        import os
        from lucy_edge.config import load_config

        os.environ["NEXUS_OLLAMA_REQUEST_TIMEOUT"] = "200"
        os.environ["NEXUS_OLLAMA_CONNECT_TIMEOUT"] = "10"
        try:
            cfg = load_config()
            self.assertEqual(cfg.providers.request_timeout, 200.0)
            self.assertEqual(cfg.providers.connect_timeout, 10.0)
        finally:
            del os.environ["NEXUS_OLLAMA_REQUEST_TIMEOUT"]
            del os.environ["NEXUS_OLLAMA_CONNECT_TIMEOUT"]


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

    async def test_registry_passes_config_timeouts_to_provider(self):
        from lucy_edge.config import LucyEdgeConfig
        from lucy_edge.providers.registry import build_default_registry

        config = LucyEdgeConfig()
        config.providers.ollama_base_url = "http://10.202.5.66:11434"
        config.providers.request_timeout = 90.0
        config.providers.connect_timeout = 3.0
        registry = build_default_registry(config)
        ollama = registry.get("ollama")
        self.assertIsNotNone(ollama)
        self.assertEqual(ollama._transport.timeout, 90.0)
        self.assertEqual(ollama._transport.connect_timeout, 3.0)


if __name__ == "__main__":
    unittest.main()
