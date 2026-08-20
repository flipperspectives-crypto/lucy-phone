"""NEXUS LUCY EDGE configuration.

Environment-overridable configuration, deliberately NOT embedding secrets.
Environment variables follow the NEXUS_* conventions:

    NEXUS_LUCY_GATEWAY_HOST
    NEXUS_LUCY_GATEWAY_PORT
    NEXUS_HOST_ROLE        (PHONE | LAPTOP | RTX4060 | UNKNOWN)
    NEXUS_HOST_ID
    NEXUS_PHONE_LOCAL_INFERENCE_ENABLED
    NEXUS_PHONE_LOCAL_INFERENCE_UNLOCKED
    NEXUS_SESSION_TOKEN    (auth token; overrides data/operator.token file)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from .version import __version__

DEFAULT_HOST_ROLE = "PHONE"
DEFAULT_DATA_DIR = "data"


def _default_host_id() -> str:
    try:
        import platform

        return platform.node() or "unknown-host"
    except Exception:
        return "unknown-host"


class PhonePolicy(BaseModel):
    """Hard phone-local-inference policy.

    phone_local_inference_enabled defaults to FALSE.  When FALSE, no local
    model generation may occur from Lucy Edge on the phone host.  This is the
    architectural response to the thermal incident (SoC 104.2 C peak).

    When TRUE, the phone may run small models locally (phone-only operation),
    but ONLY behind the routing gates: model class <= local_inference_max_class
    (7B-class is never local on a phone), verified thermal telemetry passing
    the governor, and enough free RAM for the model class.

    ARM / Android fail-closed guard
    -------------------------------
    On ARM hosts (aarch64/arm64 — phones, tablets, SBCs), local inference is
    BLOCKED regardless of ``phone_local_inference_enabled`` unless the operator
    explicitly sets ``local_inference_unlocked = True``.  This is fail-closed:
    the safe default is "do not run a local LLM on this phone".  Accidentally
    enabling local inference on a phone can hit thermal limits and drain the
    battery.  To intentionally run local inference on an ARM host, set BOTH
    ``phone_local_inference_enabled = True`` AND
    ``local_inference_unlocked = True`` (or the
    ``NEXUS_PHONE_LOCAL_INFERENCE_UNLOCKED`` env var).
    """

    phone_local_inference_enabled: bool = False
    local_inference_max_class: str = "SMALL"  # MICRO | SMALL; never LUCY_7B_CLASS
    local_inference_unlocked: bool = False  # ARM fail-closed override


class GatewayConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8970
    auth_enabled: bool = True
    max_requests_per_window: int = 120
    rate_window_seconds: float = 60.0


class ProviderConfig(BaseModel):
    # The sovereign runtime has exactly one inference provider: the locally
    # trained TinyTransformer (``local_lucy``).  No cloud, no remote inference,
    # no fallback provider.  ``mock`` is never the default and is only ever
    # registered when ``routing.allow_mock_generation`` is explicitly enabled.
    default_provider: str = "local_lucy"
    # Total time budget for a single on-device inference response.
    request_timeout: float = 120.0
    # Fail-fast ceiling for establishing a connection to the on-device
    # inference runtime.
    connect_timeout: float = 5.0
    stream_chunk_timeout: float = 5.0


class TrainingConfig(BaseModel):
    """Pointer to a locally trained Lucy checkpoint for honest introspection.

    Both fields default to None, which means no local checkpoint is wired and
    introspection reports weight-training as UNAVAILABLE.  They are only set
    once a real checkpoint produced by ``training.train`` exists.
    """

    checkpoint_path: Optional[str] = None
    lineage_db: Optional[str] = None


class RoutingConfig(BaseModel):
    default_model: str = "lucy:1.7b"
    allow_mock_generation: bool = True
    # Operator-declared parameter sizes (billions) for model families whose tag
    # does not encode a size.  Declared sizes feed classification; never guessed.
    # e.g. known_sizes: {lucy: 1.7}  -> a phone-quantized small Lucy build.
    known_sizes: dict[str, float] = Field(default_factory=dict)


class MemoryConfig(BaseModel):
    db_path: str = "data/memory.db"
    max_search_results: int = 20


class EvidenceConfig(BaseModel):
    dir_path: str = "data/evidence"
    ledger_db: str = "data/evidence.db"
    atomic_writes: bool = True


class AgentConfig(BaseModel):
    max_steps: int = 8
    max_tool_calls: int = 12
    max_failures: int = 3
    task_timeout: float = 30.0
    tool_timeout: float = 10.0
    max_output_chars: int = 10_000
    # "rule"  -> RulePlanner (deterministic, phone-safe default)
    # "model" -> ModelDrivenPlanner with a pluggable PlannerProvider.
    #            On a phone (local inference disabled) the ModelProvider routes
    #            through ModelRouter, which denies local inference and falls
    #            back to the mock provider — so NO real model runs on phone.
    planner_backend: str = "rule"


class IntrospectionConfig(BaseModel):
    runtime_name: str = "lucy_edge"
    runtime_version: str = __version__


class PhoneClientConfig(BaseModel):
    gateway_host: str = "127.0.0.1"
    gateway_port: int = 8970
    gateway_scheme: str = "http"
    token_file: str = "data/operator.token"


class MCPServerConfig(BaseModel):
    """Allowlisted MCP server (mirrors lucy_edge.mcp.MCPServerConfig).

    Declared here so it can be loaded from YAML/env.  The lucy_edge.mcp module
    re-validates on construction.
    """

    server_id: str
    transport: str = "stdio"
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    allowed_tools: list[str] = Field(default_factory=list)
    denied_tools: list[str] = Field(default_factory=list)
    connect_timeout: float = 5.0
    call_timeout: float = 10.0
    enabled: bool = True
    permission_class: str = "mcp"


class MCPConfig(BaseModel):
    enabled: bool = False
    servers: list[MCPServerConfig] = Field(default_factory=list)
    max_servers: int = 8
    max_tools_per_server: int = 32
    global_denied_tools: list[str] = Field(default_factory=list)
    fail_fast: bool = False


class FoundationConfig(BaseModel):
    """The new-foundation contract.

    A NEXUS LUCY EDGE deployment is only on its OWN foundation when it can be
    machine-verified, not merely claimed: every inference endpoint is
    loopback/private-network (never a public cloud), the phone safety policy is
    enforced, and memory and evidence live on local disk with provenance.

    This is the layer conventional cloud models cannot offer: an auditable
    guarantee about where the AI's data, decisions, and weights live.
    """

    enforce_no_cloud_endpoints: bool = True
    require_phone_policy: bool = True  # when host_role == PHONE
    allow_public_internet_tools: bool = False


class LucyEdgeConfig(BaseModel):
    host_id: str = Field(default_factory=_default_host_id)
    host_role: str = DEFAULT_HOST_ROLE
    base_dir: str = "."
    phone: PhonePolicy = Field(default_factory=PhonePolicy)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    providers: ProviderConfig = Field(default_factory=ProviderConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    introspection: IntrospectionConfig = Field(default_factory=IntrospectionConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    phone_client: PhoneClientConfig = Field(default_factory=PhoneClientConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    foundation: FoundationConfig = Field(default_factory=FoundationConfig)

    def resolve(self, path: str) -> str:
        """Resolve a config path relative to base_dir."""
        if Path(path).is_absolute():
            return path
        return str(Path(self.base_dir) / path)


_ENV_OVERRIDES: dict[str, tuple[str, Any]] = {
    "NEXUS_HOST_ID": ("host_id", str),
    "NEXUS_HOST_ROLE": ("host_role", str),
    "NEXUS_LUCY_GATEWAY_HOST": ("gateway.host", str),
    "NEXUS_LUCY_GATEWAY_PORT": ("gateway.port", int),
    "NEXUS_PHONE_LOCAL_INFERENCE_ENABLED": ("phone.phone_local_inference_enabled", bool),
    # NEXUS_PHONE_LOCAL_INFERENCE_UNLOCKED unlocks the ARM fail-closed guard.
    # Required (with PHONE_LOCAL_INFERENCE_ENABLED) to run local inference on
    # ARM hosts.  Intentionally separate so the two flags must both be set.
    "NEXUS_PHONE_LOCAL_INFERENCE_UNLOCKED": ("phone.local_inference_unlocked", bool),
}


def _cast(value: str, ty: Any) -> Any:
    if ty is bool:
        return value.strip().lower() in ("1", "true", "yes", "on")
    return ty(value)


def _apply_env_overrides(config: LucyEdgeConfig) -> LucyEdgeConfig:
    data = config.model_dump()
    for env_name, (path, ty) in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        parts = path.split(".")
        node = data
        for part in parts[:-1]:
            node = node[part]
        try:
            node[parts[-1]] = _cast(raw, ty)
        except (ValueError, TypeError):
            pass
    return LucyEdgeConfig.model_validate(data)


def load_config(path: Optional[str] = None, base_dir: str = ".") -> LucyEdgeConfig:
    """Load config from an optional YAML file, then apply env overrides."""
    if path is not None and not Path(path).exists():
        raise FileNotFoundError(f"config file not found: {path}")
    if path is not None and Path(path).exists():
        try:
            import yaml
        except ImportError:  # pragma: no cover
            raise RuntimeError("PyYAML is required to load YAML config")
        data = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(data, dict):
            raise ValueError(f"config root must be a mapping: {path}")
        config = LucyEdgeConfig.model_validate(data)
    else:
        config = LucyEdgeConfig()
    config.base_dir = base_dir
    return _apply_env_overrides(config)
