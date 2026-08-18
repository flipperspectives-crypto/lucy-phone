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

from .helpers import FakeTransport, local_checkpoint, make_config, temp_dir


def _request(model="qwen3:1.7b", provider="local_lucy", role=HostRole.PHONE, target=None, resources=None):
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
        tmp = temp_dir()
        config = make_config(tmp, phone_local_inference=True)
        config.training.checkpoint_path = local_checkpoint(tmp)
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
        tmp = temp_dir()
        config = make_config(tmp, phone_local_inference=True)
        config.training.checkpoint_path = local_checkpoint(tmp)
        transport = FakeTransport()
        transport.on("GET", "/api/version", {"version": "0.4.7"})
        services = build_services(config, transport=transport)
        services.hosts.register(
            HostState(
                host_id="laptop-01",
                role=HostRole.LAPTOP,
                status=HostStatus.REGISTERED,
                provider="local_lucy",
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
        tmp = temp_dir()
        config = make_config(tmp, phone_local_inference=True)
        config.training.checkpoint_path = local_checkpoint(tmp)
        transport = FakeTransport()
        transport.on("GET", "/api/version", {"version": "0.4.7"})
        services = build_services(config, transport=transport)
        result = await services.router.route(_request(resources=_cool_snapshot()))
        self.assertEqual(result.decision, RoutingDecision.ALLOW)
        self.assertEqual(result.reason_code, ReasonCode.OK)
        self.assertEqual(result.provider, "local_lucy")
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


class ArmFailClosedGuardTests(unittest.IsolatedAsyncioTestCase):
    """Tests for the ARM fail-closed local-inference guard."""

    async def test_arm_guard_blocks_local_inference_by_default(self):
        """On an ARM host, even with phone_local_inference_enabled=true,
        local inference is denied unless local_inference_unlocked=true."""
        from lucy_edge.routing.policy import _is_arm_host

        if not _is_arm_host():
            self.skipTest("Not on ARM host; guard not active")
        config = make_config(
            temp_dir(),
            phone_local_inference=True,
            phone_local_inference_unlocked=False,
        )
        services = build_services(config)
        result = await services.router.route(
            _request(model="qwen3:1.7b", resources=_cool_snapshot())
        )
        self.assertEqual(result.decision, RoutingDecision.DENY)
        self.assertEqual(result.reason_code, ReasonCode.ARM_LOCAL_INFERENCE_LOCKED)

    async def test_arm_guard_both_flags_required(self):
        """Both phone_local_inference_enabled AND local_inference_unlocked
        must be true for local inference on ARM."""
        from lucy_edge.routing.policy import _is_arm_host

        if not _is_arm_host():
            self.skipTest("Not on ARM host; guard not active")
        tmp = temp_dir()
        config = make_config(tmp, phone_local_inference=True)
        config.training.checkpoint_path = local_checkpoint(tmp)
        config.phone.local_inference_unlocked = True
        transport = FakeTransport()
        transport.on("GET", "/api/version", {"version": "0.4.7"})
        services = build_services(config, transport=transport)
        result = await services.router.route(
            _request(model="qwen3:1.7b", resources=_cool_snapshot())
        )
        self.assertEqual(result.decision, RoutingDecision.ALLOW)

    async def test_arm_guard_does_not_block_remote_routing(self):
        """Routing to a remote host is NOT blocked by the ARM guard,
        even when local_inference_unlocked is false. This is the intended
        way to use a phone with a laptop's Lucy host."""
        tmp = temp_dir()
        config = make_config(
            tmp,
            phone_local_inference=True,
            phone_local_inference_unlocked=False,
        )
        config.training.checkpoint_path = local_checkpoint(tmp)
        transport = FakeTransport()
        transport.on("GET", "/api/version", {"version": "0.4.7"})
        services = build_services(config, transport=transport)
        services.hosts.register(
            HostState(
                host_id="laptop-01",
                role=HostRole.LAPTOP,
                status=HostStatus.REGISTERED,
                provider="local_lucy",
            )
        )
        result = await services.router.route(
            _request(target="laptop-01")
        )
        self.assertEqual(result.decision, RoutingDecision.ROUTE)
        self.assertEqual(result.reason_code, ReasonCode.REMOTE_HOST_SELECTED)
        self.assertEqual(result.target_host, "laptop-01")

    async def test_non_arm_host_not_affected_by_guard(self):
        """On non-ARM hosts, the guard is a no-op (no ARM denial)."""
        from lucy_edge.routing.policy import _is_arm_host

        if _is_arm_host():
            self.skipTest("On ARM host; cannot test non-ARM path here")
        tmp = temp_dir()
        config = make_config(tmp, phone_local_inference=True)
        config.training.checkpoint_path = local_checkpoint(tmp)
        transport = FakeTransport()
        transport.on("GET", "/api/version", {"version": "0.4.7"})
        services = build_services(config, transport=transport)
        result = await services.router.route(
            _request(model="qwen3:1.7b", resources=_cool_snapshot())
        )
        self.assertEqual(result.decision, RoutingDecision.ALLOW)

    async def test_is_arm_host_detects_aarch64(self):
        """The _is_arm_host helper recognizes ARM architecture strings."""
        from lucy_edge.routing.policy import _is_arm_host
        import platform

        machine = platform.machine().lower()
        if machine.startswith(("arm", "aarch")):
            self.assertTrue(_is_arm_host())
        else:
            self.assertFalse(_is_arm_host())

    async def test_arm_localhost_guard_bypassed_for_remote_url(self):
        """On ARM, routing to a registered remote host is not blocked by the
        local-inference guard when local_inference_unlocked is false; the
        remote-host branch handles it."""
        from lucy_edge.routing.policy import _is_arm_host

        tmp = temp_dir()
        config = make_config(
            tmp,
            phone_local_inference=True,
            phone_local_inference_unlocked=False,
        )
        config.training.checkpoint_path = local_checkpoint(tmp)
        services = build_services(config)
        # Route to a registered remote host — should succeed (not blocked by
        # localhost guard since URL is remote).
        services.hosts.register(
            HostState(
                host_id="laptop-01",
                role=HostRole.LAPTOP,
                status=HostStatus.REGISTERED,
                provider="local_lucy",
            )
        )
        transport = FakeTransport()
        transport.on("GET", "/api/version", {"version": "0.4.7"})
        # Rebuild with transport for provider health check
        services = build_services(config, transport=transport)
        services.hosts.register(
            HostState(
                host_id="laptop-01",
                role=HostRole.LAPTOP,
                status=HostStatus.REGISTERED,
                provider="local_lucy",
            )
        )
        result = await services.router.route(_request(target="laptop-01"))
        self.assertEqual(result.decision, RoutingDecision.ROUTE)
        self.assertEqual(result.reason_code, ReasonCode.REMOTE_HOST_SELECTED)

    async def test_arm_localhost_guard_unlocked_with_override(self):
        """On ARM with localhost URL but unlocked + enabled, the guard
        is bypassed (dev override)."""
        from lucy_edge.routing.policy import _is_arm_host

        if not _is_arm_host():
            self.skipTest("Not on ARM host; guard not active")
        tmp = temp_dir()
        config = make_config(
            tmp,
            phone_local_inference=True,
            phone_local_inference_unlocked=True,
        )
        config.training.checkpoint_path = local_checkpoint(tmp)
        # Default URL is localhost
        transport = FakeTransport()
        transport.on("GET", "/api/version", {"version": "0.4.7"})
        services = build_services(config, transport=transport)
        result = await services.router.route(
            _request(model="qwen3:1.7b", resources=_cool_snapshot())
        )
        # Unlocked + enabled + cool + healthy → ALLOW (localhost guard bypassed)
        self.assertEqual(result.decision, RoutingDecision.ALLOW)


if __name__ == "__main__":
    unittest.main()
