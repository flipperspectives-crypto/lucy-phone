"""Tool execution context: services available to builtin tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolContext:
    config: Any = None
    providers: Any = None
    hosts: Any = None
    router: Any = None
    memory_store: Any = None
    retrieval: Any = None
    admission: Any = None
    evidence: Any = None
    telemetry: Any = None
    introspection: Any = None
    agent_factory: Any = None
    workspace: str = "."
    extra: dict[str, Any] = field(default_factory=dict)

    def as_public(self) -> dict[str, Any]:
        """Context summary safe to expose via tools (no secrets)."""
        return {
            "host_role": getattr(self.config, "host_role", None),
            "host_id": getattr(self.config, "host_id", None),
            "workspace": self.workspace,
            "providers": list(self.providers.names()) if self.providers else [],
            "hosts": [h.host_id for h in (self.hosts.list() if self.hosts else [])],
        }
