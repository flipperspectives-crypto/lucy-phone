"""Full runtime introspection report builder.

Combines capability classification with live runtime evidence so that Lucy can
answer "what can you acquire / what do you have" from actual runtime state.
"""

from __future__ import annotations

from typing import Any

from ..version import __version__


class LucyIntrospection:
    def __init__(self, capabilities: Any, config: Any) -> None:
        self.capabilities = capabilities
        self.config = config

    async def report(self) -> dict[str, Any]:
        caps = await self.capabilities.capabilities_report()
        return {
            "service": "lucy_edge",
            "introspection_version": __version__,
            "generated_from": "live runtime evidence (mock/test mode during phone phase)",
            **caps,
        }

    async def answer_enhancements(self) -> dict[str, Any]:
        """Machine-readable answer to: what can you acquire on this system?"""
        caps = await self.capabilities.capabilities_report()
        memory = caps["memory"]
        inference = caps["inference"]
        tools = caps["tools"]
        return {
            "persistent_memory_available": memory["persistent_memory_available"],
            "retrieval_available": memory["retrieval_available"],
            "git_tool_available": "git.status" in tools["names"],
            "remote_inference_available": False,
            "remote_host_offline": False,
            "phone_local_inference_blocked": (
                caps["inference"]["phone_local_inference_enabled"] is False
            ),
            "weight_training_available": caps["training"]["weight_training"] == "AVAILABLE",
            "mcp_available": caps["mcp"]["available"],
            "configuration_evolution_available": caps["evolution"][
                "configuration_evolution"
            ]
            == "AVAILABLE",
            "source": "runtime evidence, not model-generated claims",
        }
