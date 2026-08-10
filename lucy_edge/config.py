"""NEXUS LUCY EDGE configuration.

Environment-overridable configuration, deliberately NOT embedding secrets.
Environment variables follow the NEXUS_* conventions:

    NEXUS_LUCY_GATEWAY_HOST
    NEXUS_LUCY_GATEWAY_PORT
    NEXUS_OLLAMA_HOST
    NEXUS_HOST_ROLE        (PHONE | LAPTOP | RTX4060 | UNKNOWN)
    NEXUS_HOST_ID
    NEXUS_SESSION_TOKEN    (auth token; overrides data/operator.token file)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from .version import __version__

DEFAULT_HOST_ROLE = "PHONE"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_DATA_DIR = "data"


def _default_host_id() -> str:
    try:
        import socket

        return socket.gethostname() or "unknown-host"
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
    """

    phone_local_inference_enabled: bool = False
    local_inference_max_class: str = "SMALL"  # MICRO | SMALL; never LUCY_7B_CLASS


class GatewayConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8970
    auth_enabled: bool = True
    max_requests_per_window: int = 120
    rate_window_seconds: float = 60.0


class ProviderConfig(BaseModel):
    default_provider: str = "mock"
    ollama_base_url: str = DEFAULT_OLLAMA_HOST
    request_timeout: float = 30.0
    stream_chunk_timeout: float = 5.0


class RoutingConfig(BaseModel):
    default_model: str = "qwen3:1.7b"
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


class RemoteHostConfig(BaseModel):
    """A remote inference host that has been configured.

    No host is ever marked REGISTERED or ONLINE merely because it is listed
    here; presence in config is UNKNOWN until a real heartbeat registers it.
    """

    host_id: str
    hostname: str = ""
    role: str = "LAPTOP"
    provider: str = "ollama"
    base_url: Optional[str] = None
    base_url_env: Optional[str] = None
    enabled: bool = True


class IntrospectionConfig(BaseModel):
    runtime_name: str = "lucy_edge"
    runtime_version: str = __version__


class PhoneClientConfig(BaseModel):
    gateway_host: str = "127.0.0.1"
    gateway_port: int = 8970
    gateway_scheme: str = "http"
    token_file: str = "data/operator.token"


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
    remote_hosts: list[RemoteHostConfig] = Field(default_factory=list)
    introspection: IntrospectionConfig = Field(default_factory=IntrospectionConfig)
    phone_client: PhoneClientConfig = Field(default_factory=PhoneClientConfig)
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
    "NEXUS_OLLAMA_HOST": ("providers.ollama_base_url", str),
    "NEXUS_PHONE_LOCAL_INFERENCE_ENABLED": ("phone.phone_local_inference_enabled", bool),
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
