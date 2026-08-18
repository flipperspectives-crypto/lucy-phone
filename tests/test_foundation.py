"""Foundation layer tests: the auditable new-foundation contract.

Proves:
  - endpoint classification is conservative (loopback / private LAN / cloud)
  - the no-cloud audit catches a cloud endpoint and fails the verdict
  - the phone-safety-policy check fails when the phone gate is off
  - the honest gap: no model registry -> GAP_IDENTIFIED, never fabricated
  - local grounding returns local citations with provenance, or says "none"
"""

from __future__ import annotations

import unittest

from lucy_edge.foundation.audit import (
    STATUS_FAIL,
    STATUS_GAP,
    STATUS_PASS,
    VERDICT_GAP,
    VERDICT_UNSAFE,
    classify_endpoint,
)
from lucy_edge.memory.schema import MemoryRecord, MemoryType, ProvenanceCategory
from lucy_edge.routing.hosts import HostRole, HostState, HostStatus
from lucy_edge.services import build_services

from .helpers import FakeTransport, local_checkpoint, make_config, trained_checkpoint, temp_dir


class EndpointClassificationTests(unittest.TestCase):
    def test_loopback(self):
        self.assertEqual(classify_endpoint("http://127.0.0.1:11434"), "LOCAL_LOOPBACK")
        self.assertEqual(classify_endpoint("http://localhost:8970"), "LOCAL_LOOPBACK")

    def test_private_lan(self):
        self.assertEqual(classify_endpoint("http://192.168.1.50:11434"), "LOCAL_PRIVATE")
        self.assertEqual(classify_endpoint("http://10.0.0.5:11434"), "LOCAL_PRIVATE")
        self.assertEqual(classify_endpoint("http://laptop.local:11434"), "LOCAL_PRIVATE")

    def test_public_cloud(self):
        self.assertEqual(classify_endpoint("http://8.8.8.8:11434"), "PUBLIC_CLOUD")
        self.assertEqual(classify_endpoint("https://api.openai.com/v1"), "PUBLIC_CLOUD")
        self.assertEqual(classify_endpoint("https://ollama.com"), "PUBLIC_CLOUD")

    def test_invalid(self):
        self.assertEqual(classify_endpoint(""), "INVALID")
        self.assertEqual(classify_endpoint("not-a-url"), "INVALID")


class FoundationAuditTests(unittest.IsolatedAsyncioTestCase):
    async def _services(self, phone_local_inference=True, host_role="PHONE"):
        config = make_config(temp_dir(), phone_local_inference=phone_local_inference, host_role=host_role)
        transport = FakeTransport()
        transport.on("GET", "/api/version", {"version": "0.4.7"})
        transport.on("GET", "/api/tags", {"models": []})
        services = build_services(config, transport=transport)
        await services.open()
        return services

    async def test_clean_deployment_reports_honest_model_gap(self):
        services = await self._services()
        try:
            audit = await services.foundation.audit()
            self.assertEqual(audit["verdict"], VERDICT_GAP)
            self.assertEqual(audit["failed_checks"], [])
            self.assertIn("model_weights_present", audit["gap_checks"])
            statuses = {c["id"]: c["status"] for c in audit["checks"]}
            self.assertEqual(statuses["no_cloud_endpoints"], STATUS_PASS)
            self.assertEqual(statuses["phone_safety_policy"], STATUS_PASS)
            self.assertEqual(statuses["local_memory"], STATUS_PASS)
            self.assertEqual(statuses["local_evidence"], STATUS_PASS)
            self.assertEqual(statuses["model_weights_present"], STATUS_GAP)
        finally:
            await services.close()

    async def test_cloud_endpoint_fails_audit(self):
        services = await self._services()
        try:
            services.hosts.register(
                HostState(
                    host_id="cloud-host",
                    role=HostRole.SERVER,
                    status=HostStatus.REGISTERED,
                    provider="ollama",
                    base_url="http://api.cloudmodel.io:11434",
                )
            )
            audit = await services.foundation.audit()
            self.assertEqual(audit["verdict"], VERDICT_UNSAFE)
            self.assertIn("no_cloud_endpoints", audit["failed_checks"])
        finally:
            await services.close()

    async def test_phone_policy_off_fails_audit(self):
        services = await self._services(phone_local_inference=False)
        try:
            audit = await services.foundation.audit()
            self.assertEqual(audit["verdict"], VERDICT_UNSAFE)
            self.assertIn("phone_safety_policy", audit["failed_checks"])
        finally:
            await services.close()

    async def test_private_lan_endpoint_allowed(self):
        services = await self._services()
        try:
            services.hosts.register(
                HostState(
                    host_id="laptop-01",
                    role=HostRole.LAPTOP,
                    status=HostStatus.REGISTERED,
                    provider="ollama",
                    base_url="http://192.168.1.50:11434",
                )
            )
            audit = await services.foundation.audit()
            statuses = {c["id"]: c["status"] for c in audit["checks"]}
            self.assertEqual(statuses["no_cloud_endpoints"], STATUS_PASS)
        finally:
            await services.close()

    async def _audit_with_checkpoint(self, checkpoint_path: str):
        config = make_config(temp_dir(), phone_local_inference=True)
        config.training.checkpoint_path = checkpoint_path
        transport = FakeTransport()
        transport.on("GET", "/api/version", {"version": "0.4.7"})
        services = build_services(config, transport=transport)
        await services.open()
        try:
            audit = await services.foundation.audit()
            return {c["id"]: c["status"] for c in audit["checks"]}
        finally:
            await services.close()

    async def test_trained_model_passes_audit(self):
        tmp = temp_dir()
        statuses = await self._audit_with_checkpoint(trained_checkpoint(tmp))
        self.assertEqual(statuses["model_weights_present"], STATUS_PASS)

    async def test_untrained_model_gaps_audit(self):
        tmp = temp_dir()
        statuses = await self._audit_with_checkpoint(local_checkpoint(tmp))
        self.assertEqual(statuses["model_weights_present"], STATUS_GAP)


class LocalGroundingTests(unittest.IsolatedAsyncioTestCase):
    async def _services(self):
        config = make_config(temp_dir(), phone_local_inference=True)
        transport = FakeTransport()
        transport.on("GET", "/api/version", {"version": "0.4.7"})
        services = build_services(config, transport=transport)
        await services.open()
        return services

    async def test_grounding_returns_local_provenance(self):
        services = await self._services()
        try:
            created = await services.memory.create(
                MemoryRecord(
                    content="the nexus gateway port is 8970",
                    source="operator",
                    memory_type=MemoryType.SEMANTIC,
                    provenance=ProvenanceCategory.USER_STATED,
                )
            )
            result = await services.grounding.ground("gateway port")
            self.assertTrue(result["grounded"])
            self.assertTrue(result["local_only"])
            sources = {c["source"] for c in result["memory_citations"]}
            self.assertEqual(sources, {"LOCAL_MEMORY"})
            self.assertTrue(
                any(c["memory_id"] == created.memory_id for c in result["memory_citations"])
            )
            self.assertIn("LOCAL_MEMORY", str(result["method"]))
        finally:
            await services.close()

    async def test_grounding_does_not_fabricate(self):
        services = await self._services()
        try:
            result = await services.grounding.ground("definitely-no-such-topic-xyz")
            self.assertFalse(result["grounded"])
            self.assertIn("no local records match", result["note"])
        finally:
            await services.close()


if __name__ == "__main__":
    unittest.main()
