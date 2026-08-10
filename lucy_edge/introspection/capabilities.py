"""Capability classification and report builder.

No capability is ever fabricated.  Missing hardware telemetry is UNKNOWN; an
absent weight-training stack reports UNAVAILABLE; persistent memory is reported
only because the SQLite store is actually present and opened.
"""

from __future__ import annotations

from typing import Any, Optional

from ..version import BUILD_PHASE, READINESS, __version__


class CapabilityIntrospection:
    def __init__(
        self,
        config: Any,
        providers: Any,
        hosts: Any,
        memory: Any,
        evidence: Any,
        telemetry: Any,
        tools: Any,
        agent_limits: Any,
        policy: Any,
        planner: Any = None,
        mcp_registry: Any = None,
    ) -> None:
        self.config = config
        self.providers = providers
        self.hosts = hosts
        self.memory = memory
        self.evidence = evidence
        self.telemetry = telemetry
        self.tools = tools
        self.agent_limits = agent_limits
        self.policy = policy
        self.planner = planner
        self.mcp_registry = mcp_registry

    async def capabilities_report(self) -> dict[str, Any]:
        memory_count = (
            await self.memory.count() if self.memory is not None else None
        )
        evidence_count = (
            await self.evidence.count() if self.evidence is not None else None
        )
        snapshot = self.telemetry.snapshot() if self.telemetry is not None else None
        hosts = self.hosts.list() if self.hosts is not None else []

        model_names: list[str] = []
        for provider in (self.providers.all() if self.providers else []):
            if getattr(provider, "simulated", False):
                continue
            model_names.append(provider.name)

        return {
            "identity": {
                "runtime_name": "lucy_edge",
                "runtime_version": __version__,
                "build_phase": BUILD_PHASE,
                "readiness": READINESS,
                "host_id": self.config.host_id if self.config else None,
                "host_role": self.config.host_role if self.config else None,
            },
            "inference": {
                "local_inference_blocked": (
                    not self.config.phone.phone_local_inference_enabled
                    if self.config and self.config.host_role == "PHONE"
                    else False
                ),
                "phone_local_inference_enabled": (
                    self.config.phone.phone_local_inference_enabled if self.config else None
                ),
                "default_provider": (
                    self.config.providers.default_provider if self.config else None
                ),
                "remote_hosts": [
                    {
                        "host_id": h.host_id,
                        "status": h.status.value,
                        "role": h.role.value,
                        "provider": h.provider,
                        "models": h.models,
                        "registered": h.registered_at is not None,
                    }
                    for h in hosts
                ],
                "real_providers_registered": model_names,
            },
            "memory": {
                "persistent_memory_available": self.memory is not None,
                "memory_backend": "sqlite" if self.memory is not None else None,
                "memory_count": memory_count,
                "fts_available": getattr(self.memory, "fts_available", None),
                "retrieval_available": (
                    self.config is not None and self.retrieval_configured()
                ),
                "admission_layer_available": True,
            },
            "capability_classification": {
                "APPLICATION_MEMORY": (
                    "AVAILABLE" if self.memory is not None else "UNAVAILABLE"
                ),
                "CONTEXT_WINDOW": "UNKNOWN",
                "CONFIGURATION_EVOLUTION": "UNAVAILABLE",
                "MODEL_WEIGHT_TRAINING": "UNAVAILABLE",
                "MODEL_WEIGHT_MODIFICATION": "UNAVAILABLE",
                "note": (
                    "APPLICATION_MEMORY is real SQLite memory, not model weights; "
                    "weight training/modification stacks are absent in this build"
                ),
            },
            "training": {
                "weight_training": "UNAVAILABLE",
                "weight_modification": "UNAVAILABLE",
                "training_library": None,
                "gradient_descent": "UNAVAILABLE",
                "loRA_adapters": "UNAVAILABLE",
                "note": "no training library is installed; no training is claimed",
            },
            "evolution": {
                "configuration_evolution": "UNAVAILABLE",
                "continuous_evolution_approved": False,
                "note": "SELF_EVOLUTION_LOOP integration is not wired into lucy_edge",
            },
            "agent": {
                "enabled": True,
                "planner_backend": self._planner_backend(),
                "planner_is_model_driven": self._planner_is_model_driven(),
                "limits": self.agent_limits.model_dump() if self.agent_limits else None,
                "bounded": True,
                "autonomous_replication": False,
                "continuous_background_evolution": False,
            },
            "tools": {
                "count": len(self.tools.names()) if self.tools else 0,
                "names": self.tools.names() if self.tools else [],
                "permission_defaults": {
                    "arbitrary_shell": "DENY",
                    "out_of_scope_filesystem": "DENY",
                    "secrets": "DENY",
                    "writes": "ASK",
                    "deletion": "ASK",
                    "destructive_git": "DENY",
                },
            },
            "telemetry": {
                "hardware_telemetry_available": bool(
                    snapshot and snapshot.telemetry_available
                ),
                "thermal_policy": {
                    "phone_local_inference_enabled": (
                        self.config.phone.phone_local_inference_enabled if self.config else None
                    )
                },
                "battery_temperature_c": snapshot.battery_temperature_c if snapshot else None,
            },
            "evidence": {
                "enabled": self.evidence is not None,
                "record_count": evidence_count,
                "storage": "sqlite+json-atomic" if self.evidence is not None else None,
            },
            "mcp": self._mcp_report(),
        }

    def retrieval_configured(self) -> bool:
        return getattr(self.config, "memory", None) is not None

    def _planner_backend(self) -> str:
        if self.planner is None:
            return "NONE"
        return getattr(self.planner, "backend", "RULE_BASED")

    def _planner_is_model_driven(self) -> bool:
        return self._planner_backend() == "MODEL_DRIVEN"

    def _mcp_report(self) -> dict[str, Any]:
        if self.mcp_registry is None or not self.mcp_registry.enabled:
            return {
                "available": False,
                "enabled": False,
                "servers_configured": 0,
                "servers_online": 0,
                "tools_discovered": 0,
                "note": "MCP is not enabled",
            }
        tools = self.mcp_registry.all_tools()
        reports = self.mcp_registry._reports
        online = sum(1 for r in reports.values() if r.ok)
        return {
            "available": self.mcp_registry.available,
            "enabled": True,
            "servers_configured": len(self.mcp_registry.server_ids()),
            "servers_online": online,
            "tools_discovered": len(tools),
            "tool_names": [t.qualified_name for t in tools],
        }

    async def runtime_report(self) -> dict[str, Any]:
        return await self.capabilities_report()
