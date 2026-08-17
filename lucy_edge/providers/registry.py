"""Provider registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..config import DEFAULT_OLLAMA_HOST, LucyEdgeConfig
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

    Phone-only, on-device design: inference runs locally on the device.  The
    trained-from-scratch TinyTransformer (``local_lucy``) is registered as a real
    (non-simulated) provider whenever a local checkpoint is wired in via
    ``config.training.checkpoint_path``.  ``mock`` is always available as the
    safe fallback.  Ollama is NOT registered by default — external LLM inference
    is out of scope for the phone-only design (no public cloud, no remote
    inference).  It is only registered if explicitly configured, and is never
    the default.
    """
    registry = ProviderRegistry()
    registry.register(MockProvider())

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

    # Ollama is opt-in only: registered when an explicit (non-default) base URL
    # is configured or a transport is supplied.  The default localhost host is
    # NOT registered, keeping inference strictly phone-local / on-device per the
    # original design (no remote/external LLM inference).
    if transport is not None or (
        config.providers.ollama_base_url
        and config.providers.ollama_base_url != DEFAULT_OLLAMA_HOST
    ):
        registry.register(
            OllamaProvider(
                base_url=config.providers.ollama_base_url,
                transport=transport,
                request_timeout=config.providers.request_timeout,
                connect_timeout=config.providers.connect_timeout,
            )
        )
    return registry
