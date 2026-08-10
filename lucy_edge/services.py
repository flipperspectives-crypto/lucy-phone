"""Service assembly for NEXUS LUCY EDGE.

Builds one connected set of services (providers, routing, memory, evidence,
tools, agent, introspection, gateway auth) from a config.  The default runtime
uses the MockProvider, so it performs no real inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .agent.limits import AgentLimits, limits_from_config
from .agent.planner import ModelDrivenPlanner, RulePlanner
from .agent.planner_provider import MockPlannerProvider, ModelPlannerProvider
from .agent.runtime import AgentRuntime
from .config import LucyEdgeConfig
from .evidence.ledger import EvidenceLedger
from .evidence.schema import EvidenceRecord, EvidenceType
from .foundation.audit import FoundationGuard
from .foundation.grounding import LocalGrounding
from .gateway.auth import AuthService
from .gateway.rate_limit import RateLimiter
from .hardware.governor import ThermalGovernor
from .hardware.sensors import build_telemetry
from .introspection.capabilities import CapabilityIntrospection
from .introspection.runtime_report import LucyIntrospection
from .memory.admission import MemoryAdmission
from .memory.retrieval import RetrievalEngine
from .memory.store import MemoryStore
from .providers.registry import ProviderRegistry, build_default_registry
from .routing.hosts import HostRegistry, HostRole
from .routing.model_router import ModelRouter
from .routing.policy import RoutingPolicy, RoutingRequest, RoutingResult
from .tools.builtin.core import register_builtin_tools
from .tools.context import ToolContext
from .tools.permissions import PermissionPolicy, build_phone_policy
from .tools.registry import ToolRegistry


@dataclass
class LucyEdgeServices:
    config: LucyEdgeConfig
    providers: ProviderRegistry
    hosts: HostRegistry
    router: ModelRouter
    policy: RoutingPolicy
    memory: MemoryStore
    retrieval: RetrievalEngine
    admission: MemoryAdmission
    evidence: EvidenceLedger
    telemetry: Any
    governor: ThermalGovernor
    permissions: PermissionPolicy
    tools: ToolRegistry
    capabilities: CapabilityIntrospection
    introspection: LucyIntrospection
    context: ToolContext
    agent_limits: AgentLimits
    planner: Any
    auth: AuthService
    rate_limiter: RateLimiter
    foundation: Optional["FoundationGuard"] = None
    grounding: Optional["LocalGrounding"] = None
    workspace: str = "."
    _open: bool = False

    async def open(self) -> None:
        if self._open:
            return
        if self.memory is not None:
            await self.memory.open()
        if self.evidence is not None:
            await self.evidence.open()
        self._open = True

    async def close(self) -> None:
        if not self._open:
            return
        if self.memory is not None:
            await self.memory.close()
        if self.evidence is not None:
            await self.evidence.close()
        self._open = False

    def new_agent_run(self, goal: str, limits: Optional[AgentLimits] = None) -> AgentRuntime:
        return AgentRuntime(
            run_id=__import__("uuid").uuid4().hex,
            goal=goal,
            limits=limits or self.agent_limits,
            registry=self.tools,
            planner=self.planner,
            evidence=self.evidence,
            memory_retrieval=self.retrieval,
            context=self.context,
        )

    async def record_routing(
        self, request: RoutingRequest, result: RoutingResult
    ) -> str:
        if self.evidence is None:
            return ""
        record = EvidenceRecord(
            record_type=EvidenceType.ROUTING_DECISION,
            goal=f"routing decision for {request.model}",
            model=result.model,
            provider=result.provider,
            host=request.host_id,
            host_role=request.host_role.value,
            routing_decision=result.decision.value,
            routing_reason_code=result.reason_code.value,
            completion_reason=result.message,
        )
        await self.evidence.append(record)
        return record.run_id


def build_services(
    config: Optional[LucyEdgeConfig] = None,
    transport: Any = None,
    fixed_token: Optional[str] = None,
    workspace: Optional[str] = None,
) -> LucyEdgeServices:
    config = config or LucyEdgeConfig()
    base_dir = config.base_dir or "."
    workspace = workspace or base_dir

    providers = build_default_registry(config, transport=transport)
    hosts = HostRegistry(
        known=[h.model_dump() for h in config.remote_hosts]
        if config.remote_hosts
        else None
    )

    memory = MemoryStore(config.resolve(config.memory.db_path))
    retrieval = RetrievalEngine(memory)
    admission = MemoryAdmission(memory)
    evidence = EvidenceLedger(
        config.resolve(config.evidence.dir_path),
        config.resolve(config.evidence.ledger_db),
        atomic=config.evidence.atomic_writes,
    )

    telemetry = build_telemetry(
        config.host_id,
        HostRole(config.host_role) if config.host_role in HostRole.__members__ else HostRole.UNKNOWN,
    )

    policy = RoutingPolicy(config)
    router = ModelRouter(config, providers, hosts, policy=policy)
    governor = ThermalGovernor()

    permissions = build_phone_policy(workspace)
    tools = ToolRegistry(permissions)
    context = ToolContext(
        config=config,
        providers=providers,
        hosts=hosts,
        router=router,
        memory_store=memory,
        retrieval=retrieval,
        admission=admission,
        evidence=evidence,
        telemetry=telemetry,
        workspace=workspace,
    )
    register_builtin_tools(tools, context)
    context.introspection = None  # filled below

    agent_limits = limits_from_config(config)

    # Build the agent planner.  Phone-safe default is the rule-based planner.
    # When planner_backend == "model", use ModelDrivenPlanner: on a phone the
    # ModelProvider routes through ModelRouter (which denies local inference
    # and falls back to the deterministic MockProvider), so no real model runs.
    if config.agent.planner_backend == "model":
        if config.host_role == "PHONE":
            _provider = MockPlannerProvider()
        else:
            _provider = ModelPlannerProvider(config, router, providers)
        planner = ModelDrivenPlanner(agent_limits, _provider)
    else:
        planner = RulePlanner(agent_limits)

    capabilities = CapabilityIntrospection(
        config=config,
        providers=providers,
        hosts=hosts,
        memory=memory,
        evidence=evidence,
        telemetry=telemetry,
        tools=tools,
        agent_limits=agent_limits,
        policy=policy,
        planner=planner,
    )
    introspection = LucyIntrospection(capabilities, config)
    context.introspection = introspection

    auth = AuthService(
        token_file=config.resolve(config.phone_client.token_file),
        enabled=config.gateway.auth_enabled,
        fixed_token=fixed_token,
    )
    rate_limiter = RateLimiter(
        max_requests=config.gateway.max_requests_per_window,
        window_seconds=config.gateway.rate_window_seconds,
    )

    services = LucyEdgeServices(
        config=config,
        providers=providers,
        hosts=hosts,
        router=router,
        policy=policy,
        memory=memory,
        retrieval=retrieval,
        admission=admission,
        evidence=evidence,
        telemetry=telemetry,
        governor=governor,
        permissions=permissions,
        tools=tools,
        capabilities=capabilities,
        introspection=introspection,
        context=context,
        agent_limits=agent_limits,
        planner=planner,
        auth=auth,
        rate_limiter=rate_limiter,
        workspace=workspace,
    )
    services.foundation = FoundationGuard(services)
    services.grounding = LocalGrounding(retrieval, evidence)
    return services
