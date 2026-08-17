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
from .config import LucyEdgeConfig, MCPConfig, MCPServerConfig

# Loyal agent imports - lazy loaded at runtime
LOYAL_AVAILABLE = None
DevotionalCore = None
LoyalAgentRuntime = None
create_loyal_runtime = None


def _check_loyal_available() -> bool:
    """Check if lucy_core is available at runtime."""
    global LOYAL_AVAILABLE, DevotionalCore, LoyalAgentRuntime, create_loyal_runtime
    if LOYAL_AVAILABLE is not None:
        return LOYAL_AVAILABLE
    try:
        from lucy_core.devotional.core import DevotionalCore as _DevotionalCore
        from lucy_core.runtime.loyal_runtime import LoyalAgentRuntime as _LoyalAgentRuntime, create_loyal_runtime as _create_loyal_runtime
        DevotionalCore = _DevotionalCore
        LoyalAgentRuntime = _LoyalAgentRuntime
        create_loyal_runtime = _create_loyal_runtime
        LOYAL_AVAILABLE = True
    except ImportError:
        LOYAL_AVAILABLE = False
    return LOYAL_AVAILABLE
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
from .mcp import MCPAudit, MCPAllowlist, MCPRegistry
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
from .tools.registry import ToolRegistry, ToolSpec


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
    mcp_registry: Optional[MCPRegistry] = None
    foundation: Optional["FoundationGuard"] = None
    grounding: Optional["LocalGrounding"] = None
    devotional_core: Optional["DevotionalCore"] = None
    workspace: str = "."
    _transport: Any = None
    _open: bool = False

    async def open(self) -> None:
        if self._open:
            return
        if self.memory is not None:
            await self.memory.open()
        if self.evidence is not None:
            await self.evidence.open()
        await self._probe_remote_hosts()
        if self.mcp_registry is not None and self.mcp_registry.enabled:
            await self.mcp_registry.open()
            self._register_mcp_tools()
        self._open = True

    async def _probe_remote_hosts(self) -> None:
        """Probe configured remote hosts and register only the healthy ones.

        A host listed in ``config.remote_hosts`` starts as UNKNOWN.  We perform
        a real, lightweight Ollama health check (GET /api/version) against the
        host's ``base_url``; only on success do we promote it to REGISTERED via
        the existing HostRegistry.register() API.  Unreachable hosts are left
        UNKNOWN so the router refuses to trust them (never trust config alone).
        """
        if self.hosts is None:
            return
        from .providers.ollama import OllamaProvider

        for host_cfg in self.config.remote_hosts:
            base_url = host_cfg.base_url
            if not base_url:
                continue
            # Skip localhost: the ARM guard handles that via routing policy.
            try:
                provider = OllamaProvider(
                    base_url=base_url, transport=self._transport
                )
                health = await provider.health()
            except Exception:
                health = None
            if health is not None and health.ok:
                from .routing.hosts import HostState, HostStatus

                state = HostState(
                    host_id=host_cfg.host_id,
                    hostname=host_cfg.hostname,
                    role=HostRole(host_cfg.role)
                    if host_cfg.role in HostRole.__members__
                    else HostRole.UNKNOWN,
                    status=HostStatus.REGISTERED,
                    provider=host_cfg.provider,
                    base_url=base_url,
                    models=list(health.models),
                    registered_at=__import__("time").time(),
                    last_heartbeat=__import__("time").time(),
                )
                self.hosts.register(state)

    async def close(self) -> None:
        if not self._open:
            return
        if self.mcp_registry is not None:
            await self.mcp_registry.close()
        if self.memory is not None:
            await self.memory.close()
        if self.evidence is not None:
            await self.evidence.close()
        self._open = False

    def _register_mcp_tools(self) -> None:
        """Register discovered MCP tools into the host tool registry.

        MCP tools are registered as ``mcp.<server_id>.<tool>`` so they flow
        through the normal permission and evidence path.  Registration is
        idempotent per open() and never raises.
        """
        if self.mcp_registry is None or self.tools is None:
            return
        from .mcp import make_mcp_tool_func

        for tool in self.mcp_registry.all_tools():
            func = make_mcp_tool_func(self.mcp_registry, tool.server_id, tool)
            spec = ToolSpec(
                name=tool.qualified_name,
                description=f"[MCP {tool.server_id}] {tool.description}",
                func=func,
                permission_class=tool.permission_class,
            )
            try:
                self.tools.register(spec)
            except ValueError:
                pass

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

    def new_loyal_agent_run(self, goal: str, limits: Optional[AgentLimits] = None) -> "LoyalAgentRuntime":
        """Create a devotional agent runtime with loyalty/honesty gates and devotional core."""
        if not _check_loyal_available():
            raise RuntimeError("Loyal agent not available. Install lucy_core package.")
        if self.devotional_core is None:
            self.devotional_core = DevotionalCore(
                source_name="Lauren Flipo",
                retrieval=self.retrieval,
                evidence=self.evidence,
            )
        return create_loyal_runtime(
            goal=goal,
            limits=limits or self.agent_limits,
            registry=self.tools,
            devotional_core=self.devotional_core,
            evidence=self.evidence,
            memory_retrieval=self.retrieval,
            context=self.context,
            planner=self.planner,
            provider=self.providers.get(self.config.providers.default_provider),
        )

    def morning_review(self) -> "Any":
        """Create a MorningReview wired to this service's devotional core + sleep cycle."""
        if not _check_loyal_available() or self.devotional_core is None:
            raise RuntimeError("Devotional core not available.")
        from lucy_core.devotional.morning_review import MorningReview

        # Build a sleep runner that creates a loyal runtime and runs sleep
        async def _sleep_runner() -> "Any":
            runtime = self.new_loyal_agent_run("morning review sleep")
            return await runtime.sleep()

        return MorningReview(self.devotional_core, sleep_runner=_sleep_runner)

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

    permissions = build_phone_policy(workspace, allow_external=config.gemini.enabled)
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
    # When planner_backend == "model", use ModelDrivenPlanner with the
    # ModelPlannerProvider.  The provider routes through ModelRouter: when the
    # configured Ollama URL is remote it supplies the remote target host so the
    # ARM guard allows routing to e.g. a Windows laptop; when routing is denied
    # (localhost on ARM, no remote host, etc.) it safely falls back to the
    # deterministic MockPlannerProvider.  So no real model runs unless the
    # operator explicitly configures and enables remote inference.
    if config.agent.planner_backend == "model":
        _provider = ModelPlannerProvider(config, router, providers)
        planner = ModelDrivenPlanner(agent_limits, _provider)
    else:
        planner = RulePlanner(agent_limits)

    # Build the MCP registry from the explicit allowlist.  On a phone this is
    # inert unless the operator explicitly enables MCP AND configures servers.
    mcp_audit = MCPAudit(evidence)
    mcp_allowlist = MCPAllowlist(_mcp_config_from_config(config))
    mcp_registry = MCPRegistry(
        _mcp_config_from_config(config), allowlist=mcp_allowlist, audit=mcp_audit
    )

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
        mcp_registry=mcp_registry,
        training_checkpoint_path=config.training.checkpoint_path,
        training_lineage_db=config.training.lineage_db,
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
        _transport=transport,
    )
    services.mcp_registry = mcp_registry
    services.foundation = FoundationGuard(services)
    services.grounding = LocalGrounding(retrieval, evidence)
    if _check_loyal_available():
        services.devotional_core = DevotionalCore(
            source_name="Lauren Flipo",
            retrieval=retrieval,
            evidence=evidence,
        )
    return services


def _mcp_config_from_config(config: LucyEdgeConfig) -> MCPConfig:
    """Translate the YAML-level MCPConfig into the runtime MCPConfig."""
    from .mcp import MCPConfig as _RuntimeMCPConfig
    from .mcp import MCPServerConfig as _RuntimeServerConfig

    servers = []
    for s in config.mcp.servers:
        rc = _RuntimeServerConfig(
            server_id=s.server_id,
            transport=s.transport,
            command=s.command,
            args=list(s.args),
            env=dict(s.env),
            url=s.url,
            headers=dict(s.headers),
            allowed_tools=list(s.allowed_tools),
            denied_tools=list(s.denied_tools),
            connect_timeout=s.connect_timeout,
            call_timeout=s.call_timeout,
            enabled=s.enabled,
            permission_class=s.permission_class,
        )
        # Carry through an injected transport (tests wire FakeMCPTransport).
        injected = getattr(s, "_transport", None)
        if injected is not None:
            rc._transport = injected
        servers.append(rc)
    return _RuntimeMCPConfig(
        enabled=config.mcp.enabled,
        servers=servers,
        max_servers=config.mcp.max_servers,
        max_tools_per_server=config.mcp.max_tools_per_server,
        global_denied_tools=list(config.mcp.global_denied_tools),
        fail_fast=config.mcp.fail_fast,
    )
