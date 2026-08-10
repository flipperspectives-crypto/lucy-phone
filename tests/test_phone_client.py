"""Phone client tests.

Proves the client is control-plane only: its source contains no inference
path, and its method surface matches the documented control-plane set.
"""

from __future__ import annotations

import ast
import inspect
import unittest
from pathlib import Path

from lucy_edge.phone.client import LucyEdgeClient, TokenProvider

from .helpers import temp_dir


class PhoneClientSourceTests(unittest.TestCase):
    def test_client_source_has_no_inference_path(self):
        source = inspect.getsource(LucyEdgeClient)
        self.assertNotIn("import ollama", source)
        self.assertNotIn(".generate(", source)
        self.assertNotIn("provider.chat", source)
        self.assertNotIn("load_model", source)
        self.assertNotIn("train", source)
        self.assertNotIn("ollama", source.lower())

    def test_client_does_not_import_providers(self):
        module_path = Path(inspect.getfile(LucyEdgeClient))
        tree = ast.parse(module_path.read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        self.assertFalse(any("provider" in imp.lower() for imp in imports))
        self.assertFalse(any("infer" in imp.lower() for imp in imports))

    def test_control_plane_method_surface(self):
        client_methods = {
            name for name in dir(LucyEdgeClient) if not name.startswith("_")
        }
        # All documented control-plane methods exist, and none of them are
        # inference / model-lifecycle / training operations.
        self.assertTrue(LucyEdgeClient.CONTROL_PLANE_METHODS <= client_methods)
        banned = {
            "inference",
            "generate",
            "generate_local",
            "load_model",
            "unload",
            "train",
            "benchmark",
            "start_ollama",
            "serve",
        }
        self.assertTrue(client_methods.isdisjoint(banned))

    def test_token_provider_never_exposes_value_in_repr(self):
        tmp = temp_dir()
        token_file = f"{tmp}/operator.token"
        Path(token_file).write_text("super-secret-token-value")
        provider = TokenProvider(token_file=token_file)
        # The value is read on demand and never cached on the provider.
        self.assertEqual(provider.token(), "super-secret-token-value")
        self.assertNotIn("super-secret-token-value", str(provider.__dict__))
        self.assertNotIn("super-secret-token-value", repr(provider))


class TokenProviderTests(unittest.TestCase):
    def test_reads_file_without_printing(self):
        tmp = temp_dir()
        token_file = f"{tmp}/operator.token"
        Path(token_file).write_text("super-secret-token-value\n")
        provider = TokenProvider(token_file=token_file)
        self.assertEqual(provider.token(), "super-secret-token-value")


class ClientEndToEndTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from aiohttp.test_utils import TestClient, TestServer

        from lucy_edge.gateway.server import create_app
        from .helpers import make_config
        from .services_open import open_services

        self.tmp = temp_dir()
        token_file = f"{self.tmp}/operator.token"
        Path(token_file).write_text("test-token")
        services = await open_services(make_config(self.tmp))
        self.services = services
        app = create_app(services)
        server = TestServer(app)
        self.client = TestClient(server)
        await self.client.start_server()
        base = f"http://{self.client.host}:{self.client.port}"
        self.edge = LucyEdgeClient(base_url=base, token_provider=TokenProvider(token_file))

    async def asyncTearDown(self):
        await self.edge.close()
        await self.client.close()
        await self.services.close()

    async def test_health_via_client(self):
        health = await self.edge.health()
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["service"], "lucy_edge")

    async def test_chat_via_client_mock(self):
        result = await self.edge.chat(
            model="qwen3:1.7b", provider="mock", messages=[{"role": "user", "content": "hi"}]
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["message"], "NEXUS SAFE")

    async def test_stream_via_client(self):
        chunks = []
        async for chunk in self.edge.stream_chat(
            model="qwen3:1.7b", provider="mock", messages=[{"role": "user", "content": "hi"}]
        ):
            chunks.append(chunk)
        self.assertGreaterEqual(len(chunks), 1)
        self.assertTrue(chunks[-1]["done"])

    async def test_introspect_via_client(self):
        report = await self.edge.introspect()
        self.assertEqual(report["training"]["weight_training"], "UNAVAILABLE")

    async def test_task_submit_and_status_via_client(self):
        submitted = await self.edge.submit_task("check runtime health")
        run_id = submitted["run_id"]
        status = await self.edge.task_status(run_id)
        self.assertIn("state", status)

    async def test_evidence_and_hosts_via_client(self):
        evidence = await self.edge.evidence(limit=5)
        self.assertIn("records", evidence)
        hosts = await self.edge.remote_hosts()
        self.assertIn("hosts", hosts)

    async def test_client_does_not_start_ollama(self):
        # The client's surface is control-plane only; it has no method that
        # loads weights, starts a server, or runs generation.
        banned = {"load_model", "generate", "serve", "start_ollama", "train", "benchmark"}
        surface = set(LucyEdgeClient.CONTROL_PLANE_METHODS)
        self.assertTrue(surface.isdisjoint(banned))


if __name__ == "__main__":
    unittest.main()
