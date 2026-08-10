"""Model router: pick provider and host for a requested model.

Inputs (where available): requested model, requested provider, available
providers, host identity/role, resources, provider health, configured remote
hosts, and the routing policy.

Returns a structured ALLOW / DENY / ROUTE / THROTTLE result with a
machine-readable reason code.
"""

from __future__ import annotations

from typing import Any, Optional

from ..config import LucyEdgeConfig
from ..providers.base import CapabilityUnavailable, ProviderError
from ..providers.registry import ProviderRegistry
from .hosts import HostRegistry, HostRole, HostState, HostStatus
from .policy import (
    ModelClass,
    ReasonCode,
    RoutingDecision,
    RoutingPolicy,
    RoutingRequest,
    RoutingResult,
)


class ModelRouter:
    def __init__(
        self,
        config: LucyEdgeConfig,
        providers: ProviderRegistry,
        hosts: HostRegistry,
        policy: Optional[RoutingPolicy] = None,
    ) -> None:
        self.config = config
        self.providers = providers
        self.hosts = hosts
        self.policy = policy or RoutingPolicy(config)

    async def health_of(self, provider_name: str) -> bool:
        provider = self.providers.get(provider_name)
        if provider is None:
            return False
        try:
            health = await provider.health()
            return health.ok
        except (CapabilityUnavailable, ProviderError, Exception):
            return False

    def _pick_remote_host(self, request: RoutingRequest) -> Optional[HostState]:
        if request.target_host:
            host = self.hosts.get(request.target_host)
            if host is None:
                return None
            return host if host.is_usable else host
        usable = self.hosts.usable()
        if not usable:
            return None
        return usable[0]

    async def route(self, request: RoutingRequest) -> RoutingResult:
        _, _, model_class = self.policy.classify(request.model)

        # 2. Mock provider: allowed for phone-safe testing, explicitly labeled.
        if request.provider == "mock":
            if not self.config.routing.allow_mock_generation:
                return RoutingResult(
                    decision=RoutingDecision.DENY,
                    reason_code=ReasonCode.MOCK_PROVIDER_SELECTED,
                    message="mock generation disabled by routing.allow_mock_generation",
                    model=request.model,
                    model_class=model_class,
                    provider="mock",
                )
            return RoutingResult(
                decision=RoutingDecision.ALLOW,
                reason_code=ReasonCode.MOCK_PROVIDER_SELECTED,
                message="mock provider selected (synthetic; no real inference)",
                model=request.model,
                model_class=model_class,
                provider="mock",
            )

        # 1. Hard phone policy + global switch (unconditional, non-mock).
        local_denial = self.policy._local_denial(request, model_class)
        if local_denial is not None:
            local_denial.evidence["router"] = "hard-phone-policy"
            return local_denial

        if model_class == ModelClass.UNKNOWN:
            return RoutingResult(
                decision=RoutingDecision.DENY,
                reason_code=ReasonCode.UNKNOWN_MODEL,
                message=f"unable to classify model '{request.model}'",
                model=request.model,
                model_class=model_class,
            )

        # 3. Phone-local inference (phone-only operation): when the phone's
        #    local inference is enabled and no remote target was requested,
        #    try running on the phone itself behind the thermal/RAM gate.
        if (
            request.host_role == HostRole.PHONE
            and request.target_host is None
            and self.policy.phone_local_inference_enabled
        ):
            gate = self.policy.phone_local_gate(request)
            if gate is not None:
                return gate
            if not await self.health_of(request.provider):
                return RoutingResult(
                    decision=RoutingDecision.DENY,
                    reason_code=ReasonCode.PROVIDER_OFFLINE,
                    message=f"provider '{request.provider}' is offline",
                    model=request.model,
                    model_class=model_class,
                    provider=request.provider,
                )
            return RoutingResult(
                decision=RoutingDecision.ALLOW,
                reason_code=ReasonCode.OK,
                message=(
                    f"phone-local inference via provider '{request.provider}' "
                    "(thermal + RAM verified)"
                ),
                model=request.model,
                model_class=model_class,
                provider=request.provider,
            )

        # 4. Any real (non-mock) generation on a PHONE host without local
        #    inference must route to a known, registered remote host.  Unknown
        #    remote host fails safely.
        if request.host_role == HostRole.PHONE:
            host = self._pick_remote_host(request)
            if request.target_host and host is None:
                return RoutingResult(
                    decision=RoutingDecision.DENY,
                    reason_code=ReasonCode.UNKNOWN_REMOTE_HOST,
                    message=f"requested remote host '{request.target_host}' is not registered",
                    model=request.model,
                    model_class=model_class,
                    provider=request.provider,
                    target_host=request.target_host,
                )
            if host is None:
                return RoutingResult(
                    decision=RoutingDecision.DENY,
                    reason_code=ReasonCode.NO_REMOTE_HOST,
                    message="no usable remote inference host registered",
                    model=request.model,
                    model_class=model_class,
                    provider=request.provider,
                )
            if not host.is_usable:
                return RoutingResult(
                    decision=RoutingDecision.DENY,
                    reason_code=ReasonCode.NO_REMOTE_HOST,
                    message=f"remote host '{host.host_id}' is {host.status.value}",
                    model=request.model,
                    model_class=model_class,
                    provider=request.provider,
                    target_host=host.host_id,
                )
            if host.provider and not await self.health_of(host.provider):
                return RoutingResult(
                    decision=RoutingDecision.DENY,
                    reason_code=ReasonCode.PROVIDER_OFFLINE,
                    message=f"provider '{host.provider}' on host '{host.host_id}' is offline",
                    model=request.model,
                    model_class=model_class,
                    provider=host.provider,
                    target_host=host.host_id,
                )
            return RoutingResult(
                decision=RoutingDecision.ROUTE,
                reason_code=ReasonCode.REMOTE_HOST_SELECTED,
                message=f"routing '{request.model}' to remote host '{host.host_id}'",
                model=request.model,
                model_class=model_class,
                provider=host.provider or request.provider,
                target_host=host.host_id,
            )

        # 4. Non-phone host (laptop/gateway): provider health gates the decision.
        if not await self.health_of(request.provider):
            return RoutingResult(
                decision=RoutingDecision.DENY,
                reason_code=ReasonCode.PROVIDER_OFFLINE,
                message=f"provider '{request.provider}' is offline",
                model=request.model,
                model_class=model_class,
                provider=request.provider,
            )

        return RoutingResult(
            decision=RoutingDecision.ALLOW,
            reason_code=ReasonCode.OK,
            message=f"provider '{request.provider}' is healthy; allowing local generation",
            model=request.model,
            model_class=model_class,
            provider=request.provider,
        )
