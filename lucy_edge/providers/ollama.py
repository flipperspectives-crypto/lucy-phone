"""Ollama provider with a configurable endpoint.

The base endpoint is NOT hardcoded.  Examples:

    127.0.0.1  -> same-host Ollama (future laptop gateway<->Ollama)
    LAN address -> remote-host Ollama

The preferred final topology is:

    phone client -> NEXUS authenticated gateway -> laptop-local Ollama

The provider supports a pluggable ``transport`` so the HTTP client logic can
be unit-tested with deterministic fake responses on the phone WITHOUT any live
network activity or live Ollama generation.  When ``transport`` is None a real
aiohttp transport is used (laptop phase only).
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Optional

from .base import (
    BaseProvider,
    Capability,
    CapabilityUnavailable,
    ChatResponse,
    GenerationResult,
    ModelInfo,
    ProviderError,
    ProviderHealth,
    ProviderRuntimeMetrics,
    ProviderState,
    StreamChunk,
)


class AiohttpTransport:
    """Real HTTP transport using aiohttp (used on the laptop later)."""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def request(self, method: str, path: str, payload: Optional[dict] = None) -> Any:
        import aiohttp

        url = f"{self.base_url}{path}"
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        ) as session:
            if method == "GET":
                async with session.get(url) as resp:
                    body = await resp.text()
            else:
                async with session.post(url, json=payload) as resp:
                    body = await resp.text()
            if resp.status >= 400:  # type: ignore[possibly-undefined]
                raise ProviderError(f"ollama http {resp.status}: {body[:500]}")
            if not body:
                return {}
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return {"raw": body}


class FakeTransportResponse:
    """Container for a canned transport response used in tests."""

    def __init__(self, data: Any) -> None:
        self.data = data


class OllamaProvider(BaseProvider):
    """Client for an Ollama-compatible inference server."""

    name = "ollama"
    kind = "ollama"

    def __init__(self, base_url: str = "http://127.0.0.1:11434", transport: Any = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._transport = transport or AiohttpTransport(self.base_url)
        self._version: Optional[str] = None

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
        }

    async def _request(self, method: str, path: str, payload: Optional[dict] = None) -> Any:
        result = await self._transport.request(method, path, payload)
        if isinstance(result, FakeTransportResponse):
            return result.data
        return result

    async def detect(self) -> dict[str, Any]:
        data = await self._request("GET", "/api/version")
        self._version = data.get("version") if isinstance(data, dict) else None
        return {"name": self.name, "base_url": self.base_url, "version": self._version}

    async def health(self) -> ProviderHealth:
        try:
            t0 = __import__("time").monotonic()
            data = await self._request("GET", "/api/version")
            latency = (__import__("time").monotonic() - t0) * 1000.0
            self._version = data.get("version") if isinstance(data, dict) else None
            return ProviderHealth(
                state=ProviderState.ONLINE,
                ok=True,
                latency_ms=round(latency, 3),
                version=self._version,
                message=f"ollama reachable at {self.base_url}",
            )
        except ProviderError as exc:
            return ProviderHealth(
                state=ProviderState.OFFLINE, ok=False, message=str(exc)
            )
        except Exception as exc:  # network layer errors
            return ProviderHealth(
                state=ProviderState.OFFLINE,
                ok=False,
                message=f"{type(exc).__name__}: {exc}",
            )

    async def list_models(self) -> list[ModelInfo]:
        data = await self._request("GET", "/api/tags")
        models: list[ModelInfo] = []
        for item in data.get("models", []):
            details = item.get("details") or {}
            models.append(
                ModelInfo(
                    name=item.get("name", ""),
                    digest=item.get("digest"),
                    parameter_size=details.get("parameter_size"),
                    family=details.get("family"),
                    quantization=details.get("quantization_level"),
                    details=details,
                )
            )
        return models

    async def model_metadata(self, model: str) -> ModelInfo:
        data = await self._request("POST", "/api/show", {"model": model, "verbose": True})
        details = data.get("details") or {}
        return ModelInfo(
            name=model,
            digest=data.get("model_info", {}).get("general.architecture")
            or data.get("digest"),
            parameter_size=details.get("parameter_size"),
            family=details.get("family"),
            quantization=details.get("quantization_level"),
            details=data,
        )

    async def chat(self, messages: list[dict[str, Any]], model: str, **options: Any) -> ChatResponse:
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        payload.update(options)
        data = await self._request("POST", "/api/chat", payload)
        message = data.get("message", {})
        return ChatResponse(
            provider=self.name,
            model=data.get("model", model),
            message=message.get("content", ""),
            finish_reason=data.get("done_reason"),
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            digest=data.get("digest") or data.get("model"),
        )

    async def generate(self, prompt: str, model: str, **options: Any) -> GenerationResult:
        payload: dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
        payload.update(options)
        data = await self._request("POST", "/api/generate", payload)
        return GenerationResult(
            provider=self.name,
            model=data.get("model", model),
            text=data.get("response", ""),
            eval_count=data.get("eval_count"),
            eval_duration_ms=(data.get("eval_duration") or 0) / 1_000_000.0,
            digest=data.get("digest") or data.get("model"),
        )

    async def stream_chat(
        self, messages: list[dict[str, Any]], model: str, **options: Any
    ) -> AsyncIterator[StreamChunk]:
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": True}
        payload.update(options)
        raw = await self._transport.request("POST", "/api/chat", payload)
        if isinstance(raw, list):
            # Deterministic chunk list (test transport / pre-parsed NDJSON).
            for item in raw:
                yield self._chunk_from(item, model)
            return
        # real streaming is only reached with a live server (laptop phase)
        raise CapabilityUnavailable(
            Capability.STREAM_CHAT,
            "streaming requires an NDJSON-capable transport",
        )

    @staticmethod
    def _chunk_from(item: Any, model: str) -> StreamChunk:
        if isinstance(item, str):
            item = json.loads(item)
        message = item.get("message", {})
        return StreamChunk(
            provider="ollama",
            model=item.get("model", model),
            text=message.get("content", ""),
            done=bool(item.get("done", False)),
        )

    async def unload(self, model: str) -> bool:
        await self._request("POST", "/api/generate", {"model": model, "keep_alive": 0})
        return True

    async def runtime_metrics(self) -> ProviderRuntimeMetrics:
        return ProviderRuntimeMetrics(
            provider=self.name,
            telemetry_available=False,
            simulated=False,
            note="ollama runtime metrics not exposed by the ollama API",
        )
