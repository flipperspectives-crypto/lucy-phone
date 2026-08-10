"""Provider registry."""

from __future__ import annotations

from typing import Any, Optional

from ..config import LucyEdgeConfig
from .base import BaseProvider, CapabilityUnavailable, ProviderError
from .mock import MockProvider
from .ollama import OllamaProvider


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, BaseProvider] = {}

    def register(self, provider: BaseProvider) -> None:
        self._providers[provider.name] = provider

    def get(self, name: str) -> Optional[BaseProvider]:
        return self._providers.get(name)

    def require(self, name: str) -> BaseProvider:
        provider = self._providers.get(name)
        if provider is None:
            raise ProviderError(f"provider not registered: {name}")
        return provider

    def names(self) -> list[str]:
        return sorted(self._providers)

    def all(self) -> list[BaseProvider]:
        return list(self._providers.values())

    async def healthy(self) -> dict[str, bool]:
        out: dict[str, bool] = {}
        for name, provider in self._providers.items():
            try:
                health = await provider.health()
                out[name] = health.ok
            except CapabilityUnavailable:
                out[name] = False
            except ProviderError:
                out[name] = False
            except Exception:
                out[name] = False
        return out

    def summary(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for name, provider in self._providers.items():
            out[name] = {
                "name": provider.name,
                "kind": provider.kind,
                "simulated": provider.simulated,
                "capabilities": provider.capability_status(),
            }
        return out


def build_default_registry(
    config: LucyEdgeConfig, transport: Any = None
) -> ProviderRegistry:
    """Build the provider registry from configuration.

    Phone-safe default: ``default_provider`` is "mock", so the default runtime
    performs no real inference.  The Ollama provider is registered but is only
    selected when routing explicitly authorizes it (future laptop phase).
    """
    registry = ProviderRegistry()
    registry.register(MockProvider())
    if transport is not None or config.providers.ollama_base_url:
        registry.register(
            OllamaProvider(
                base_url=config.providers.ollama_base_url,
                transport=transport,
                request_timeout=config.providers.request_timeout,
                connect_timeout=config.providers.connect_timeout,
            )
        )
    return registry
