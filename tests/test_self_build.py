"""Self-build tests: Lucy builds herself from her own foundation, not the clouds.

These tests pin down that Lucy Edge constructs every layer of her own runtime
from local building blocks -- her own synthetic/mock provider, her own loopback
Ollama endpoint, her own on-disk memory and evidence stores, her own registered
hosts -- and that none of her self-build path ever reaches for a cloud service.
"""

from __future__ import annotations

import ipaddress
import unittest
from pathlib import Path
from urllib.parse import urlsplit

from lucy_edge.config import RemoteHostConfig
from lucy_edge.evidence.schema import EvidenceRecord, EvidenceType
from lucy_edge.memory.schema import MemoryRecord, MemoryType, ProvenanceCategory
from lucy_edge.routing.hosts import HostRole, HostState, HostStatus
from lucy_edge.routing.policy import ReasonCode, RoutingDecision, RoutingRequest
from lucy_edge.services import build_services

from .helpers import FakeTransport, make_config, temp_dir

_LOCAL_HOSTNAMES = ("localhost", "host")


def assert_local_only(urls: list[tuple[str, str]]) -> None:
    """Assert every URL host is loopback or private (never a public cloud)."""
    for label, url in urls:
        host = urlsplit(url).hostname
        assert host, f"{label}: missing host in {url}"
        try:
            ip = ipaddress.ip_address(host)
            is_local = ip.is_loopback or ip.is_private
        except ValueError:
            is_local = host in _LOCAL_HOSTNAMES
        assert is_local, f"{label}: {url} points at a non-local (cloud?) endpoint"


def collect_urls(config) -> list[tuple[str, str]]:
    urls = [
        ("providers.ollama_base_url", config.providers.ollama_base_url),
        ("gateway.host", f"http://{config.gateway.host}:{config.gateway.port}"),
        (
            "phone_client.gateway",
            f"{config.phone_client.gateway_scheme}://"
            f"{config.phone_client.gateway_host}:{config.phone_client.gateway_port}",
        ),
    ]
    for index, host in enumerate(config.remote_hosts):
        if host.base_url:
            urls.append((f"remote_hosts[{index}].base_url", host.base_url))
    return urls


class SelfBuildFoundationTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_foundation_is_local_first(self):
        config = make_config(temp_dir())
        self.assertEqual(config.providers.default_provider, "mock")
        self.assertEqual(config.providers.ollama_base_url, "http://127.0.0.1:11434")
        self.assertEqual(config.gateway.host, "127.0.0.1")
        self.assertEqual(config.phone_client.gateway_host, "127.0.0.1")
        self.assertEqual(config.remote_hosts, [])
        assert_local_only(collect_urls(config))

    async def test_build_registers_only_self_providers(self):
        services = build_services(make_config(temp_dir()), transport=FakeTransport())
        try:
            self.assertEqual(services.providers.names(), ["mock", "ollama"])
            for provider in services.providers.all():
                base_url = getattr(provider, "base_url", None)
                if base_url is not None:
                    assert_local_only([(provider.name, base_url)])
        finally:
            await services.close()

    async def test_self_build_never_touches_the_network(self):
        cloud_guard = FakeTransport(
            fail_with=AssertionError("self-build reached the network / cloud")
        )
        tmp = temp_dir()
        services = build_services(make_config(tmp), transport=cloud_guard)
        await services.open()
        try:
            report = await services.introspection.report()
            self.assertEqual(report["inference"]["default_provider"], "mock")

            result = await services.router.route(
                RoutingRequest(model="qwen3:1.7b", provider="mock", host_role=HostRole.PHONE)
            )
            self.assertEqual(result.decision, RoutingDecision.ALLOW)

            mock = services.providers.get("mock")
            chat = await mock.chat([{"role": "user", "content": "build me"}], model="lucy:mock")
            self.assertTrue(chat.simulated)

            record = MemoryRecord(
                content="built from her own foundation, not the clouds",
                source="self-build",
                memory_type=MemoryType.EVIDENCE,
                provenance=ProvenanceCategory.KNOWN_FROM_RUNTIME,
            )
            await services.memory.create(record)
            hits = await services.memory.search("own foundation")
            self.assertGreaterEqual(len(hits), 1)

            evidence = EvidenceRecord(
                record_type=EvidenceType.BUILD,
                goal="self-build from own foundation",
                provider="mock",
                host_role="PHONE",
                provider_endpoint_class="LOCAL",
            )
            await services.evidence.append(evidence)
            self.assertEqual(await services.evidence.count(), 1)

            self.assertEqual(cloud_guard.calls, [])
        finally:
            await services.close()

    async def test_mock_build_needs_no_remote_host(self):
        services = build_services(make_config(temp_dir()))
        try:
            result = await services.router.route(
                RoutingRequest(model="qwen3:1.7b", provider="mock", host_role=HostRole.PHONE)
            )
            self.assertEqual(result.decision, RoutingDecision.ALLOW)
            self.assertEqual(result.reason_code, ReasonCode.MOCK_PROVIDER_SELECTED)
            self.assertEqual(result.provider, "mock")
            self.assertIsNone(result.target_host)
        finally:
            await services.close()

    async def test_phone_never_routes_real_inference_to_an_untrusted_host(self):
        services = build_services(make_config(temp_dir()))
        try:
            hard_denial = await services.router.route(
                RoutingRequest(model="qwen3:1.7b", provider="ollama", host_role=HostRole.PHONE)
            )
            self.assertEqual(hard_denial.decision, RoutingDecision.DENY)
            self.assertEqual(hard_denial.reason_code, ReasonCode.LOCAL_INFERENCE_DISABLED)
        finally:
            await services.close()

    async def test_phone_only_mode_stays_local_and_never_touches_cloud(self):
        config = make_config(temp_dir(), phone_local_inference=True)
        cloud_guard = FakeTransport(
            fail_with=AssertionError("phone-only mode reached the network / cloud")
        )
        config.providers.ollama_base_url = "http://127.0.0.1:11434"
        services = build_services(config, transport=cloud_guard)
        await services.open()
        try:
            untargeted = await services.router.route(
                RoutingRequest(model="qwen3:1.7b", provider="ollama", host_role=HostRole.PHONE)
            )
            self.assertEqual(untargeted.decision, RoutingDecision.DENY)
            self.assertEqual(untargeted.reason_code, ReasonCode.THERMAL_UNKNOWN)

            cloud_host = await services.router.route(
                RoutingRequest(
                    model="qwen3:1.7b",
                    provider="ollama",
                    host_role=HostRole.PHONE,
                    target_host="cloud-inference.example.com",
                )
            )
            self.assertEqual(cloud_host.decision, RoutingDecision.DENY)
            self.assertEqual(cloud_host.reason_code, ReasonCode.UNKNOWN_REMOTE_HOST)
            self.assertEqual(cloud_guard.calls, [])
        finally:
            await services.close()

    async def test_configured_host_is_unknown_until_it_registers_itself(self):
        config = make_config(temp_dir())
        config.remote_hosts = [
            RemoteHostConfig(
                host_id="laptop",
                hostname="LAPTOP",
                role="LAPTOP",
                provider="ollama",
                base_url="http://192.168.1.50:11434",
            )
        ]
        assert_local_only(collect_urls(config))
        services = build_services(config)
        try:
            host = services.hosts.get("laptop")
            self.assertIsNotNone(host)
            self.assertEqual(host.status, HostStatus.UNKNOWN)
            self.assertFalse(host.is_usable)
            self.assertIsNone(host.registered_at)

            services.hosts.register(
                HostState(
                    host_id="laptop",
                    role=HostRole.LAPTOP,
                    provider="ollama",
                    base_url="http://192.168.1.50:11434",
                )
            )
            registered = services.hosts.get("laptop")
            self.assertEqual(registered.status, HostStatus.REGISTERED)
            self.assertTrue(registered.is_usable)
            self.assertIsNotNone(registered.registered_at)
        finally:
            await services.close()

    async def test_memory_and_evidence_live_on_her_own_disk(self):
        tmp = temp_dir()
        config = make_config(tmp)
        services = build_services(config)
        await services.open()
        try:
            memory_path = Path(config.resolve(config.memory.db_path))
            evidence_dir = Path(config.resolve(config.evidence.dir_path))
            ledger_path = Path(config.resolve(config.evidence.ledger_db))
            for path in (memory_path, evidence_dir, ledger_path):
                self.assertTrue(str(path).startswith(str(Path(tmp))))

            self.assertTrue(memory_path.exists())
            self.assertTrue(ledger_path.exists())

            record = MemoryRecord(
                content="foundation note",
                source="self-build",
                memory_type=MemoryType.WORKING,
                provenance=ProvenanceCategory.USER_STATED,
            )
            await services.memory.create(record)
            self.assertGreaterEqual(await services.memory.count(), 1)

            evidence = EvidenceRecord(
                record_type=EvidenceType.BUILD,
                goal="own disk foundation",
                provider="mock",
                host_role="PHONE",
                provider_endpoint_class="LOCAL",
            )
            await services.evidence.append(evidence)
            stored = await services.evidence.get(evidence.run_id)
            self.assertEqual(stored["provider_endpoint_class"], "LOCAL")
            self.assertEqual(stored["provider"], "mock")
        finally:
            await services.close()

    async def test_introspection_reports_her_own_foundation_honestly(self):
        services = build_services(make_config(temp_dir()))
        await services.open()
        try:
            report = await services.introspection.report()
            self.assertEqual(report["inference"]["default_provider"], "mock")
            self.assertEqual(report["inference"]["real_providers_registered"], ["ollama"])
            self.assertTrue(report["inference"]["local_inference_blocked"])
            self.assertEqual(report["memory"]["memory_backend"], "sqlite")
            self.assertFalse(report["agent"]["autonomous_replication"])
            self.assertEqual(report["training"]["weight_training"], "UNAVAILABLE")
            self.assertEqual(report["capability_classification"]["MODEL_WEIGHT_TRAINING"], "UNAVAILABLE")
        finally:
            await services.close()


if __name__ == "__main__":
    unittest.main()
