"""Routing policy and phone-safety tests.

Proves:
  - phone local inference disabled by default
  - Lucy 7B phone request denied
  - remote-host routing represented
  - unknown remote host fails safely
  - provider unavailable handled
"""

from __future__ import annotations

import unittest

from lucy_edge.hardware.snapshot import HardwareSnapshot
from lucy_edge.providers.base import ProviderError
from lucy_edge.routing.hosts import HostState, HostStatus, HostRole
from lucy_edge.routing.policy import (
    ModelClass,
    ReasonCode,
    RoutingDecision,
    RoutingRequest,
    classify_model,
    class_rank,
    required_ram_bytes,
)
from lucy_edge.services import build_services

from .helpers import FakeTransport, make_config, temp_dir


def _request(model="qwen3:1.7b", provider="ollama", role=HostRole.PHONE, target=None, resources=None):
    return RoutingRequest(
        model=model,
        provider=provider,
        host_role=role,
        host_id="phone-1",
        target_host=target,
        resources=resources,
    )


def _cool_snapshot(temp: float = 50.0, ram_bytes: int = 8 * 1024**3) -> HardwareSnapshot:
    return HardwareSnapshot(
        host_id="phone-1",
        host_role=HostRole.PHONE,
        cpu_temperature_c=temp,
        ram_available_bytes=ram_bytes,
        thermal_source="test",
        telemetry_available=True,
    )


class PhonePolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_phone_local_inference_disabled_by_default(self):
        services = build_services(make_config(temp_dir()))
        result = await services.router.route(_request())
        self.assertEqual(result.decision, RoutingDecision.DENY)
        self.assertEqual(result.reason_code, ReasonCode.LOCAL_INFERENCE_DISABLED)

    async def test_config_defaults_to_disabled(self):
        config = make_config(temp_dir())
        self.assertFalse(config.phone.phone_local_inference_enabled)

    async def test_lucy_7b_phone_request_denied(self):
        config = make_config(temp_dir(), phone_local_inference=True)
        services = build_services(config)
        result = await services.router.route(
            _request(model="lucy:7b-q4_K_M", provider="ollama")
        )
        self.assertEqual(result.decision, RoutingDecision.DENY)
        self.assertEqual(result.reason_code, ReasonCode.PHONE_MODEL_DENIED)
        self.assertEqual(result.model_class, ModelClass.LUCY_7B_CLASS)

    async def test_classify_model(self):
        self.assertEqual(classify_model("lucy:7b")[2], ModelClass.LUCY_7B_CLASS)
        self.assertEqual(classify_model("qwen3:1.7b")[2], ModelClass.SMALL)
        self.assertEqual(classify_model("phi:0.5b")[2], ModelClass.MICRO)
        self.assertEqual(classify_model("mystery-model")[2], ModelClass.UNKNOWN)

    async def test_bare_lucy_tag_classifies_as_7b_by_default(self):
        self.assertEqual(classify_model("lucy:latest")[2], ModelClass.LUCY_7B_CLASS)
        self.assertEqual(classify_model("lucy:latest")[1], 7.0)

    async def test_known_sizes_override_lucy_default(self):
        self.assertEqual(classify_model("lucy:latest", known_sizes={"lucy": 1.7})[2], ModelClass.SMALL)

    async def test_lucy_latest_on_phone_denied_without_declared_size(self):
        config = make_config(temp_dir(), phone_local_inference=True)
        services = build_services(config)
        result = await services.router.route(
            _request(model="lucy:latest", resources=_cool_snapshot())
        )
        self.assertEqual(result.decision, RoutingDecision.DENY)
        self.assertEqual(result.reason_code, ReasonCode.PHONE_MODEL_DENIED)
        self.assertEqual(result.model_class, ModelClass.LUCY_7B_CLASS)

    async def test_declared_small_lucy_runs_phone_local(self):
        config = make_config(temp_dir(), phone_local_inference=True)
        config.routing.known_sizes = {"lucy": 1.7}
        transport = FakeTransport()
        transport.on("GET", "/api/version", {"version": "0.4.7"})
        services = build_services(config, transport=transport)
        result = await services.router.route(
            _request(model="lucy:latest", resources=_cool_snapshot())
        )
        self.assertEqual(result.decision, RoutingDecision.ALLOW)
        self.assertEqual(result.model_class, ModelClass.SMALL)
        self.assertIsNone(result.target_host)
        self.assertIn("phone-local", result.message)

    async def test_unknown_model_denied(self):
        services = build_services(make_config(temp_dir(), phone_local_inference=True))
        result = await services.router.route(_request(model="mystery-model"))
        self.assertEqual(result.decision, RoutingDecision.DENY)
        self.assertEqual(result.reason_code, ReasonCode.UNKNOWN_MODEL)

    async def test_mock_provider_allowed_even_when_local_disabled(self):
        services = build_services(make_config(temp_dir()))
        result = await services.router.route(_request(provider="mock"))
        self.assertEqual(result.decision, RoutingDecision.ALLOW)
        self.assertEqual(result.reason_code, ReasonCode.MOCK_PROVIDER_SELECTED)

    async def test_remote_host_routing_represented(self):
        config = make_config(temp_dir(), phone_local_inference=True)
        transport = FakeTransport()
        transport.on("GET", "/api/version", {"version": "0.4.7"})
        services = build_services(config, transport=transport)
        services.hosts.register(
            HostState(
                host_id="laptop-01",
                role=HostRole.LAPTOP,
                status=HostStatus.REGISTERED,
                provider="ollama",
            )
        )
        result = await services.router.route(_request(target="laptop-01"))
        self.assertEqual(result.decision, RoutingDecision.ROUTE)
        self.assertEqual(result.reason_code, ReasonCode.REMOTE_HOST_SELECTED)
        self.assertEqual(result.target_host, "laptop-01")

    async def test_unknown_remote_host_fails_safely(self):
        services = build_services(make_config(temp_dir(), phone_local_inference=True))
        result = await services.router.route(_request(target="not-registered"))
        self.assertEqual(result.decision, RoutingDecision.DENY)
        self.assertEqual(result.reason_code, ReasonCode.UNKNOWN_REMOTE_HOST)

    async def test_no_remote_host_with_local_disabled(self):
        services = build_services(make_config(temp_dir()))
        result = await services.router.route(_request())
        self.assertEqual(result.decision, RoutingDecision.DENY)
        self.assertEqual(result.reason_code, ReasonCode.LOCAL_INFERENCE_DISABLED)

    async def test_phone_local_inference_denied_without_telemetry(self):
        services = build_services(make_config(temp_dir(), phone_local_inference=True))
        result = await services.router.route(_request())
        self.assertEqual(result.decision, RoutingDecision.DENY)
        self.assertEqual(result.reason_code, ReasonCode.THERMAL_UNKNOWN)

    async def test_phone_local_inference_allowed_when_safe(self):
        config = make_config(temp_dir(), phone_local_inference=True)
        transport = FakeTransport()
        transport.on("GET", "/api/version", {"version": "0.4.7"})
        services = build_services(config, transport=transport)
        result = await services.router.route(_request(resources=_cool_snapshot()))
        self.assertEqual(result.decision, RoutingDecision.ALLOW)
        self.assertEqual(result.reason_code, ReasonCode.OK)
        self.assertEqual(result.provider, "ollama")
        self.assertIsNone(result.target_host)
        self.assertIn("phone-local", result.message)

    async def test_phone_local_inference_denied_when_hot(self):
        services = build_services(make_config(temp_dir(), phone_local_inference=True))
        result = await services.router.route(_request(resources=_cool_snapshot(temp=80.0)))
        self.assertEqual(result.decision, RoutingDecision.DENY)
        self.assertEqual(result.reason_code, ReasonCode.THERMAL_LIMIT)

    async def test_phone_local_inference_throttled_at_warn(self):
        services = build_services(make_config(temp_dir(), phone_local_inference=True))
        result = await services.router.route(_request(resources=_cool_snapshot(temp=70.0)))
        self.assertEqual(result.decision, RoutingDecision.THROTTLE)
        self.assertEqual(result.reason_code, ReasonCode.THERMAL_LIMIT)
        self.assertIsNotNone(result.throttle_seconds)

    async def test_phone_local_inference_denied_insufficient_ram(self):
        services = build_services(make_config(temp_dir(), phone_local_inference=True))
        result = await services.router.route(
            _request(resources=_cool_snapshot(ram_bytes=512 * 1024**2))
        )
        self.assertEqual(result.decision, RoutingDecision.DENY)
        self.assertEqual(result.reason_code, ReasonCode.INSUFFICIENT_RAM)

    async def test_phone_local_inference_provider_offline(self):
        config = make_config(temp_dir(), phone_local_inference=True)
        transport = FakeTransport(fail_with=ProviderError("refused"))
        services = build_services(config, transport=transport)
        result = await services.router.route(_request(resources=_cool_snapshot()))
        self.assertEqual(result.decision, RoutingDecision.DENY)
        self.assertEqual(result.reason_code, ReasonCode.PROVIDER_OFFLINE)

    async def test_phone_local_max_class_micro_rejects_small(self):
        config = make_config(temp_dir(), phone_local_inference=True)
        config.phone.local_inference_max_class = "MICRO"
        services = build_services(config)
        result = await services.router.route(
            _request(model="qwen3:1.7b", resources=_cool_snapshot())
        )
        self.assertEqual(result.decision, RoutingDecision.DENY)
        self.assertEqual(result.reason_code, ReasonCode.PHONE_MODEL_DENIED)

    async def test_phone_local_class_rank_and_ram_helpers(self):
        self.assertLess(class_rank(ModelClass.MICRO), class_rank(ModelClass.SMALL))
        self.assertLess(class_rank(ModelClass.SMALL), class_rank(ModelClass.LUCY_7B_CLASS))
        self.assertGreater(required_ram_bytes(ModelClass.SMALL), required_ram_bytes(ModelClass.MICRO))
        self.assertIsNone(required_ram_bytes(ModelClass.UNKNOWN))

    async def test_provider_offline_denied(self):
        config = make_config(temp_dir(), phone_local_inference=True)
        transport = FakeTransport(fail_with=ProviderError("refused"))
        services = build_services(config, transport=transport)
        services.hosts.register(
            HostState(
                host_id="laptop-01",
                role=HostRole.LAPTOP,
                status=HostStatus.REGISTERED,
                provider="ollama",
            )
        )
        result = await services.router.route(_request(target="laptop-01"))
        self.assertEqual(result.decision, RoutingDecision.DENY)
        self.assertEqual(result.reason_code, ReasonCode.PROVIDER_OFFLINE)

    async def test_provider_unavailable_handled_on_laptop(self):
        config = make_config(temp_dir(), host_role="LAPTOP")
        transport = FakeTransport(fail_with=ProviderError("refused"))
        services = build_services(config, transport=transport)
        result = await services.router.route(_request(role=HostRole.LAPTOP))
        self.assertEqual(result.decision, RoutingDecision.DENY)
        self.assertEqual(result.reason_code, ReasonCode.PROVIDER_OFFLINE)

    async def test_phone_policy_evaluate_directly(self):
        config = make_config(temp_dir())
        services = build_services(config)
        result = await services.policy.evaluate(_request(provider="ollama"))
        self.assertEqual(result.decision, RoutingDecision.DENY)
        self.assertEqual(result.reason_code, ReasonCode.LOCAL_INFERENCE_DISABLED)


if __name__ == "__main__":
    unittest.main()
