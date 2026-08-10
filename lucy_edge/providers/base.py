"""Model-provider abstraction.

The provider interface is deliberately model-independent: Lucy is NOT
hardwired to Ollama.  A capability is never fabricated: anything a concrete
provider does not implement raises ``CapabilityUnavailable`` or returns an
explicit unsupported state, and ``capabilities()`` reports exactly which
capabilities exist.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Optional

from pydantic import BaseModel, Field


class ProviderState(str, Enum):
    UNKNOWN = "UNKNOWN"
    DETECTED = "DETECTED"
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    ERROR = "ERROR"


class Capability(str, Enum):
    DETECT = "detect"
    HEALTH = "health"
    LIST_MODELS = "list_models"
    MODEL_METADATA = "model_metadata"
    CHAT = "chat"
    GENERATE = "generate"
    STREAM_CHAT = "stream_chat"
    UNLOAD = "unload"
    RUNTIME_METRICS = "runtime_metrics"


class CapabilityUnavailable(Exception):
    """Raised when a provider does not implement a capability.

    This is an explicit unsupported state, never a fabricated success.
    """

    def __init__(self, capability: Capability, message: str) -> None:
        super().__init__(message)
        self.capability = capability
        self.message = message


class ProviderError(Exception):
    """A provider-level error (transport, protocol, auth, etc.)."""


class ModelInfo(BaseModel):
    name: str
    digest: Optional[str] = None
    parameter_size: Optional[str] = None
    family: Optional[str] = None
    quantization: Optional[str] = None
    details: dict[str, Any] = Field(default_factory=dict)


class ProviderHealth(BaseModel):
    state: ProviderState = ProviderState.UNKNOWN
    ok: bool = False
    latency_ms: Optional[float] = None
    message: Optional[str] = None
    version: Optional[str] = None
    models: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    provider: str
    model: str
    message: str
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    simulated: bool = False
    digest: Optional[str] = None


class GenerationResult(BaseModel):
    provider: str
    model: str
    text: str
    eval_count: Optional[int] = None
    eval_duration_ms: Optional[float] = None
    simulated: bool = False
    digest: Optional[str] = None


class StreamChunk(BaseModel):
    provider: str
    model: str
    text: str = ""
    done: bool = False


class ProviderRuntimeMetrics(BaseModel):
    provider: str
    telemetry_available: bool = False
    simulated: bool = False
    models_loaded: list[str] = Field(default_factory=list)
    process_rss_bytes: Optional[int] = None
    gpu_utilization: Optional[float] = None
    vram_used_bytes: Optional[int] = None
    note: Optional[str] = None


@dataclass
class ProviderCallRecord:
    """Deterministic record of a provider call (used by tests/evidence)."""

    method: str
    model: str
    at: float = field(default_factory=time.time)
    payload: dict[str, Any] = field(default_factory=dict)


class BaseProvider(abc.ABC):
    """Abstract model provider.

    Concrete providers override ``capabilities()`` and the methods they
    actually support.  Every other method raises ``CapabilityUnavailable``.
    """

    name: str = "base"
    kind: str = "abstract"
    simulated: bool = False

    def capabilities(self) -> set[Capability]:
        return set()

    def supports(self, capability: Capability) -> bool:
        return capability in self.capabilities()

    def capability_status(self) -> dict[str, str]:
        return {
            cap.value: "supported" if cap in self.capabilities() else "unavailable"
            for cap in Capability
        }

    # --- optional capabilities; default to explicit unsupported -------------
    async def detect(self) -> dict[str, Any]:
        raise CapabilityUnavailable(Capability.DETECT, "not supported by this provider")

    async def health(self) -> ProviderHealth:
        raise CapabilityUnavailable(Capability.HEALTH, "not supported by this provider")

    async def list_models(self) -> list[ModelInfo]:
        raise CapabilityUnavailable(
            Capability.LIST_MODELS, "not supported by this provider"
        )

    async def model_metadata(self, model: str) -> ModelInfo:
        raise CapabilityUnavailable(
            Capability.MODEL_METADATA, "not supported by this provider"
        )

    async def chat(self, messages: list[dict[str, Any]], model: str, **options: Any) -> ChatResponse:
        raise CapabilityUnavailable(Capability.CHAT, "not supported by this provider")

    async def generate(self, prompt: str, model: str, **options: Any) -> GenerationResult:
        raise CapabilityUnavailable(Capability.GENERATE, "not supported by this provider")

    async def stream_chat(
        self, messages: list[dict[str, Any]], model: str, **options: Any
    ) -> AsyncIterator[StreamChunk]:
        raise CapabilityUnavailable(
            Capability.STREAM_CHAT, "not supported by this provider"
        )
        yield  # pragma: no cover - makes this an async generator

    async def unload(self, model: str) -> bool:
        raise CapabilityUnavailable(Capability.UNLOAD, "not supported by this provider")

    async def runtime_metrics(self) -> ProviderRuntimeMetrics:
        raise CapabilityUnavailable(
            Capability.RUNTIME_METRICS, "not supported by this provider"
        )
