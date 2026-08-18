"""Provider registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..config import LucyEdgeConfig
from .base import BaseProvider, CapabilityUnavailable, ProviderError
from .mock import MockProvider


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

    Sovereign, phone-only, on-device design: the ONLY inference provider is the
    locally trained TinyTransformer (``local_lucy``).  It is registered as a
    real (non-simulated) provider whenever a local checkpoint is wired in via
    ``config.training.checkpoint_path``.

    ``mock`` is NOT a fallback.  It is only registered when the operator
    explicitly enables generation via the mock provider
    (``routing.allow_mock_generation``) for diagnostics/tests.  If no local
    checkpoint is available and mock generation is disabled, the registry is
    empty and any inference request fails closed.

    Remote / Ollama inference is out of scope: there is no public cloud and no
    remote LLM.  ``OllamaProvider`` is not registered here.
    """
    registry = ProviderRegistry()

    # On-device inference: the locally trained Lucy model.  Lazy import keeps
    # lucy_edge independent of the standalone training package when no
    # checkpoint is configured.
    cp = config.training.checkpoint_path
    if cp and Path(cp).exists():
        try:
            from training.provider import LocalLucyProvider

            registry.register(
                LocalLucyProvider(checkpoint_path=cp, model_name="lucy-local")
            )
        except Exception:
            pass

    # Mock is opt-in only (never a silent fallback for real inference).
    if config.routing.allow_mock_generation:
        registry.register(MockProvider())

    return registry
