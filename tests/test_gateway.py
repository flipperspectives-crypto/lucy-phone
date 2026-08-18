"""Gateway HTTP tests using aiohttp TestClient (mock provider, no live model)."""

from __future__ import annotations

import unittest

from aiohttp.test_utils import TestClient, TestServer

from lucy_edge.gateway.server import TASKS_KEY, create_app

from lucy_edge.hardware.snapshot import HardwareSnapshot

from .helpers import local_checkpoint, make_config, temp_dir, wait_until
from .services_open import open_services


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = temp_dir()
        self.services = await open_services(make_config(self.tmp))
        self.app = create_app(self.services)
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()
        self.auth = {"Authorization": "Bearer test-token"}

    async def asyncTearDown(self):
        await self.client.close()
        await self.services.close()

    async def test_health_requires_no_auth(self):
        resp = await self.client.get("/health")
        self.assertEqual(resp.status, 200)
        body = await resp.json()
        self.assertEqual(body["service"], "lucy_edge")

    async def test_phone_local_chat_denied(self):
        resp = await self.client.post(
            "/v1/chat",
            json={"model": "qwen3:1.7b", "provider": "ollama", "messages": []},
            headers=self.auth,
        )
        self.assertEqual(resp.status, 200)
        body = await resp.json()
        self.assertFalse(body["ok"])
        self.assertEqual(body["routing"]["decision"], "DENY")
        self.assertEqual(body["routing"]["reason_code"], "LOCAL_INFERENCE_DISABLED")

    async def test_mock_chat_allowed(self):
        resp = await self.client.post(
            "/v1/chat",
            json={"model": "qwen3:1.7b", "provider": "mock", "messages": [{"role": "user", "content": "hi"}]},
            headers=self.auth,
        )
        self.assertEqual(resp.status, 200)
        body = await resp.json()
        self.assertTrue(body["ok"])
        self.assertTrue(body["simulated"])
        self.assertEqual(body["message"], "NEXUS SAFE")

    async def test_local_lucy_chat_on_device(self):
        """A chat request routed to the local_lucy provider runs on-device
        (simulated=False) with no network/cloud, once telemetry verifies the
        phone is cool.  Proves the gateway's inference path is sovereign."""
        tmp = temp_dir()
        config = make_config(tmp, phone_local_inference=True)
        config.training.checkpoint_path = local_checkpoint(tmp)
        services = await open_services(config)
        # Headless env has no real sensors; inject a verified-cool snapshot so
        # the phone-safety gate permits local inference (proves the gate, not a
        # bypass).
        services.telemetry.snapshot = lambda: HardwareSnapshot(
            host_id=config.host_id,
            host_role=config.host_role,
            cpu_temperature_c=50.0,
            ram_available_bytes=8 * 1024**3,
            thermal_source="test",
            telemetry_available=True,
        )
        app = create_app(services)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            resp = await client.post(
                "/v1/chat",
                json={
                    "model": "lucy:1.7b",
                    "provider": "local_lucy",
                    "messages": [{"role": "user", "content": "hello"}],
                },
                headers=self.auth,
            )
            self.assertEqual(resp.status, 200)
            body = await resp.json()
            self.assertTrue(body["ok"])
            self.assertFalse(body["simulated"])
            self.assertEqual(body["provider"], "local_lucy")
        finally:
            await client.close()
            await services.close()

    async def test_chat_requires_auth(self):
        resp = await self.client.post(
            "/v1/chat",
            json={"model": "qwen3:1.7b", "provider": "mock", "messages": []},
        )
        self.assertEqual(resp.status, 401)

    async def test_introspect_requires_auth_and_reports(self):
        resp = await self.client.get("/v1/lucy/introspect")
        self.assertEqual(resp.status, 401)
        resp = await self.client.get("/v1/lucy/introspect", headers=self.auth)
        self.assertEqual(resp.status, 200)
        body = await resp.json()
        self.assertEqual(body["training"]["weight_training"], "UNAVAILABLE")
        self.assertEqual(
            body["capability_classification"]["APPLICATION_MEMORY"], "AVAILABLE"
        )

    async def test_agent_submit_and_status(self):
        resp = await self.client.post(
            "/v1/agent/tasks", json={"goal": "check runtime health"}, headers=self.auth
        )
        self.assertEqual(resp.status, 202)
        body = await resp.json()
        run_id = body["run_id"]
        self.assertTrue(run_id)

        await wait_until(
            lambda: self.app[TASKS_KEY].get(run_id)
            and self.app[TASKS_KEY][run_id]["result"] is not None,
            "agent run to complete",
        )
        status = await self.client.get(f"/v1/agent/tasks/{run_id}", headers=self.auth)
        status_body = await status.json()
        self.assertEqual(status_body["final_status"], "COMPLETED")
        self.assertTrue(status_body["evidence_run_id"])

    async def test_evidence_endpoint(self):
        resp = await self.client.get("/v1/evidence", headers=self.auth)
        self.assertEqual(resp.status, 200)
        body = await resp.json()
        self.assertIn("records", body)

    async def test_hardware_snapshot(self):
        resp = await self.client.get("/v1/hardware/snapshot", headers=self.auth)
        self.assertEqual(resp.status, 200)
        body = await resp.json()
        self.assertIn("telemetry_available", body)

    async def test_rate_limit_applied(self):
        limits = self.services.config.gateway
        self.services.rate_limiter.max_requests = 2
        self.services.rate_limiter._hits.clear()
        for _ in range(2):
            resp = await self.client.get("/v1/lucy/introspect", headers=self.auth)
            self.assertEqual(resp.status, 200)
        resp = await self.client.get("/v1/lucy/introspect", headers=self.auth)
        self.assertEqual(resp.status, 429)


if __name__ == "__main__":
    unittest.main()
