"""Deterministic MockProvider for phone-side testing.

The MockProvider generates NO real inference and NO model weights.  It is used
for all phone-side tests during this phase and clearly marks its output as
``simulated=True`` so no one mistakes it for real model behavior.
"""

from __future__ import annotations

import hashlib
from typing import Any, AsyncIterator

from .base import (
    BaseProvider,
    Capability,
    ChatResponse,
    GenerationResult,
    ModelInfo,
    ProviderCallRecord,
    ProviderHealth,
    ProviderRuntimeMetrics,
    ProviderState,
    StreamChunk,
)

DEFAULT_MOCK_MODELS = ["lucy:mock", "qwen3:1.7b-mock"]


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class MockProvider(BaseProvider):
    """Synthetic provider; output is always labeled simulated."""

    name = "mock"
    kind = "mock"
    simulated = True

    def __init__(self, models: list[str] | None = None, reply: str = "NEXUS SAFE") -> None:
        self._models = list(models or DEFAULT_MOCK_MODELS)
        self._reply = reply
        self.calls: list[ProviderCallRecord] = []

    def capabilities(self) -> set[Capability]:
        return {
            Capability.DETECT,
            Capability.HEALTH,
            Capability.LIST_MODELS,
            Capability.MODEL_METADATA,
            Capability.CHAT,
            Capability.GENERATE,
            Capability.STREAM_CHAT,
            Capability.UNLOAD,
            Capability.RUNTIME_METRICS,
        }

    async def detect(self) -> dict[str, Any]:
        self.calls.append(ProviderCallRecord("detect", self.name))
        return {"name": self.name, "kind": self.kind, "simulated": True}

    async def health(self) -> ProviderHealth:
        self.calls.append(ProviderCallRecord("health", self.name))
        return ProviderHealth(
            state=ProviderState.ONLINE,
            ok=True,
            latency_ms=0.0,
            message="mock provider is synthetically healthy",
            models=list(self._models),
        )

    async def list_models(self) -> list[ModelInfo]:
        self.calls.append(ProviderCallRecord("list_models", self.name))
        return [
            ModelInfo(
                name=name,
                digest=_short_hash(name),
                parameter_size="1.7b" if "1.7b" in name else None,
                family="mock",
            )
            for name in self._models
        ]

    async def model_metadata(self, model: str) -> ModelInfo:
        self.calls.append(ProviderCallRecord("model_metadata", model))
        if model not in self._models:
            raise ValueError(f"unknown model for mock provider: {model}")
        return ModelInfo(name=model, digest=_short_hash(model), family="mock")

    async def chat(self, messages: list[dict[str, Any]], model: str, **options: Any) -> ChatResponse:
        self.calls.append(ProviderCallRecord("chat", model, payload={"messages": messages}))
        return ChatResponse(
            provider=self.name,
            model=model,
            message=self._reply,
            finish_reason="stop",
            completion_tokens=len(self._reply.split()),
            simulated=True,
            digest=_short_hash(model),
        )

    async def generate(self, prompt: str, model: str, **options: Any) -> GenerationResult:
        self.calls.append(ProviderCallRecord("generate", model, payload={"prompt": prompt}))
        return GenerationResult(
            provider=self.name,
            model=model,
            text=self._reply,
            eval_count=len(self._reply.split()),
            eval_duration_ms=0.0,
            simulated=True,
            digest=_short_hash(model),
        )

    async def stream_chat(
        self, messages: list[dict[str, Any]], model: str, **options: Any
    ) -> AsyncIterator[StreamChunk]:
        self.calls.append(ProviderCallRecord("stream_chat", model, payload={"messages": messages}))
        for token in self._reply.split(" "):
            yield StreamChunk(provider=self.name, model=model, text=token + " ", done=False)
        yield StreamChunk(provider=self.name, model=model, done=True)

    async def unload(self, model: str) -> bool:
        self.calls.append(ProviderCallRecord("unload", model))
        return True

    async def runtime_metrics(self) -> ProviderRuntimeMetrics:
        self.calls.append(ProviderCallRecord("runtime_metrics", self.name))
        return ProviderRuntimeMetrics(
            provider=self.name,
            telemetry_available=True,
            simulated=True,
            models_loaded=list(self._models),
            note="mock metrics are synthetic; not real process measurements",
        )
