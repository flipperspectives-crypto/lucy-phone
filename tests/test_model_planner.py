"""Model-driven planner tests.

These tests use the deterministic MockPlannerProvider — NO real model
inference, NO network, safe for the S25 Ultra phone.  The ModelPlannerProvider
path that calls a live model is covered only by a unit-level check that its
constructor and fallback wiring are intact; exercising a real model is NOT
done here (requires a running inference host).
"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock

from lucy_edge.agent.limits import AgentLimits
from lucy_edge.agent.planner import ModelDrivenPlanner, Plan, PlanStep, RulePlanner
from lucy_edge.agent.planner_provider import (
    MockPlannerProvider,
    ModelPlannerProvider,
    PlannerProvider,
)
from lucy_edge.agent.runtime import AgentRuntime, AgentState
from lucy_edge.services import build_services

from .helpers import FakeTransport, make_config, temp_dir


class MockPlannerProviderTests(unittest.TestCase):
    def test_returns_bounded_plan(self):
        provider = MockPlannerProvider()
        limits = AgentLimits(max_steps=8)
        plan = provider.generate_plan("check system health", ["system.health", "memory.search"], limits)
        self.assertIsInstance(plan, Plan)
        self.assertEqual(plan.goal, "check system health")
        self.assertLessEqual(len(plan.steps), 8)
        self.assertEqual(plan.steps[-1].action, "stop")

    def test_keyword_matching(self):
        provider = MockPlannerProvider()
        limits = AgentLimits()
        plan = provider.generate_plan("inspect the git repository", ["git.status", "memory.search"], limits)
        tools_used = [s.tool for s in plan.steps if s.tool]
        self.assertIn("git.status", tools_used)

    def test_only_uses_available_tools(self):
        provider = MockPlannerProvider()
        limits = AgentLimits()
        plan = provider.generate_plan(
            "read a file", ["system.health", "memory.search"], limits
        )
        for step in plan.steps:
            if step.tool:
                self.assertIn(step.tool, ["system.health", "memory.search"])

    def test_max_steps_respected(self):
        provider = MockPlannerProvider()
        limits = AgentLimits(max_steps=3)
        plan = provider.generate_plan("anything", ["system.health", "memory.search", "evidence.query"], limits)
        self.assertLessEqual(len(plan.steps), 3)

    def test_retrieve_memory_always_first_when_available(self):
        provider = MockPlannerProvider()
        limits = AgentLimits()
        plan = provider.generate_plan("find evidence records", ["memory.search", "evidence.query"], limits)
        self.assertEqual(plan.steps[0].action, "retrieve_memory")
        self.assertEqual(plan.steps[0].tool, "memory.search")


class ModelDrivenPlannerTests(unittest.TestCase):
    def test_build_plan_uses_provider(self):
        provider = MockPlannerProvider()
        planner = ModelDrivenPlanner(AgentLimits(), provider)
        plan = planner.build_plan("health check", ["system.health", "memory.search"])
        self.assertIsInstance(plan, Plan)
        self.assertEqual(planner.backend, "MODEL_DRIVEN")

    def test_fallback_on_provider_error(self):
        class BrokenProvider(PlannerProvider):
            name = "broken"

            def generate_plan(self, goal, available_tools, limits):
                raise RuntimeError("boom")

        planner = ModelDrivenPlanner(AgentLimits(), BrokenProvider())
        plan = planner.build_plan("any goal", ["system.health"])
        # Should fall back to RulePlanner and still return a valid plan.
        self.assertIsInstance(plan, Plan)
        self.assertGreater(len(plan.steps), 0)

    def test_fallback_on_unknown_tool(self):
        class EvilProvider(PlannerProvider):
            name = "evil"

            def generate_plan(self, goal, available_tools, limits):
                return Plan(
                    goal=goal,
                    steps=[
                        PlanStep(0, "execute", "shell.exec", {}, "should be rejected"),
                        PlanStep(1, "stop", None, {}, ""),
                    ],
                )

        planner = ModelDrivenPlanner(AgentLimits(), EvilProvider())
        plan = planner.build_plan("goal", ["system.health"])
        for step in plan.steps:
            if step.tool:
                self.assertNotEqual(step.tool, "shell.exec")

    def test_fallback_on_empty_plan(self):
        class EmptyProvider(PlannerProvider):
            name = "empty"

            def generate_plan(self, goal, available_tools, limits):
                return Plan(goal=goal, steps=[])

        planner = ModelDrivenPlanner(AgentLimits(), EmptyProvider())
        plan = planner.build_plan("goal", ["system.health"])
        self.assertGreater(len(plan.steps), 0)

    def test_steps_bounded_by_max_steps(self):
        provider = MockPlannerProvider()
        planner = ModelDrivenPlanner(AgentLimits(max_steps=2), provider)
        plan = planner.build_plan("goal", ["system.health", "memory.search", "evidence.query"])
        self.assertLessEqual(len(plan.steps), 2)


class ModelPlannerProviderWiringTests(unittest.TestCase):
    """Verify ModelPlannerProvider constructs and its fallback path works
    without a real model (phone-safe mock routing)."""

    def test_constructs_with_config_router_providers(self):
        tmp = temp_dir()
        services = build_services(make_config(tmp))
        # Default phone config -> mock provider is used by services.planner,
        # but we can still construct a ModelPlannerProvider directly.
        from unittest.mock import MagicMock

        provider = ModelPlannerProvider(services.config, services.router, services.providers)
        self.assertIsInstance(provider, ModelPlannerProvider)
        self.assertIsNotNone(provider._fallback)

    def test_fallback_used_when_model_unavailable(self):
        """On a phone with local inference disabled, generate_plan must fall
        back to MockPlannerProvider and return a valid plan (NO model call)."""
        tmp = temp_dir()
        services = build_services(make_config(tmp, phone_local_inference=False))
        provider = ModelPlannerProvider(services.config, services.router, services.providers)
        plan = provider.generate_plan(
            "check health", ["system.health", "memory.search"], AgentLimits()
        )
        self.assertIsInstance(plan, Plan)
        self.assertGreater(len(plan.steps), 0)


class AgentRuntimeWithModelPlannerTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_uses_model_driven_planner(self):
        tmp = temp_dir()
        services = build_services(make_config(tmp))
        await services.open()
        try:
            # Override to model-driven planner via a fresh services with
            # planner_backend="model" mocked by env or direct construction.
            limits = AgentLimits(max_steps=8, max_tool_calls=12, task_timeout=10.0, tool_timeout=3.0)
            provider = MockPlannerProvider()
            planner = ModelDrivenPlanner(limits, provider)
            runtime = AgentRuntime(
                run_id="aaaabbbbccccddddeeeeffff00001111",
                goal="check system health",
                limits=limits,
                registry=services.tools,
                planner=planner,
                evidence=services.evidence,
                context=services.context,
            )
            result = await runtime.run()
            self.assertEqual(result.final_status, AgentState.COMPLETED)
            self.assertGreater(result.steps_executed, 0)
        finally:
            await services.close()


class MCPAwarePlanningTests(unittest.TestCase):
    """Prove the model-driven planner can select MCP tools when given schemas,
    while the RulePlanner behavior is unchanged."""

    def _mcp_schemas(self) -> list[dict]:
        return [
            {
                "name": "mcp.fs-test.read_file",
                "description": "Read a file inside the allowed test directory.",
                "permission_class": "read",
                "owner": "lucy_edge",
            },
            {
                "name": "mcp.fs-test.list_directory",
                "description": "List entries inside a directory in the allowed test directory.",
                "permission_class": "read",
                "owner": "lucy_edge",
            },
        ]

    def test_mock_provider_selects_mcp_tool_with_schema(self):
        """When tool schemas are supplied, the MockPlannerProvider should
        select an MCP tool whose description matches the goal."""
        provider = MockPlannerProvider()
        limits = AgentLimits()
        schemas = self._mcp_schemas()
        all_tools = ["memory.search", "system.health"] + [s["name"] for s in schemas]
        plan = provider.generate_plan(
            "read a file from the test directory", all_tools, limits, schemas
        )
        tools_used = [s.tool for s in plan.steps if s.tool]
        self.assertIn("mcp.fs-test.read_file", tools_used)

    def test_mock_provider_ignores_mcp_without_schema(self):
        """Without tool schemas, the MockPlannerProvider falls back to its
        builtin keyword vocabulary and does NOT select MCP tools."""
        provider = MockPlannerProvider()
        limits = AgentLimits()
        schemas = self._mcp_schemas()
        all_tools = ["memory.search", "system.health", "files.read_scoped"] + [
            s["name"] for s in schemas
        ]
        plan = provider.generate_plan(
            "read a file from the test directory", all_tools, limits, tool_schemas=None
        )
        tools_used = [s.tool for s in plan.steps if s.tool]
        self.assertNotIn("mcp.fs-test.read_file", tools_used)
        # Falls back to the builtin "file" keyword match.
        self.assertIn("files.read_scoped", tools_used)

    def test_model_driven_planner_passes_schemas_to_provider(self):
        """ModelDrivenPlanner.build_plan forwards tool schemas to its provider."""
        provider = MockPlannerProvider()
        planner = ModelDrivenPlanner(AgentLimits(), provider)
        schemas = self._mcp_schemas()
        all_tools = ["memory.search"] + [s["name"] for s in schemas]
        plan = planner.build_plan(
            "list the test directory contents", all_tools, schemas
        )
        tools_used = [s.tool for s in plan.steps if s.tool]
        self.assertIn("mcp.fs-test.list_directory", tools_used)

    def test_rule_planner_ignores_mcp_tools_and_schemas(self):
        """RulePlanner behavior is unchanged: it uses only its hardcoded
        keyword vocabulary and never selects MCP tools, even when schemas
        are available."""
        planner = RulePlanner(AgentLimits())
        schemas = self._mcp_schemas()
        all_tools = ["memory.search", "files.read_scoped"] + [s["name"] for s in schemas]
        plan = planner.build_plan("read a file", all_tools)
        tools_used = [s.tool for s in plan.steps if s.tool]
        self.assertIn("files.read_scoped", tools_used)
        self.assertNotIn("mcp.fs-test.read_file", tools_used)

    def test_mcp_tool_matching_is_description_based(self):
        """The MCP matcher scores by keyword overlap with tool descriptions,
        not by tool name."""
        from lucy_edge.agent.planner_provider import _match_mcp_tool

        schemas = self._mcp_schemas()
        # "list directory" overlaps the list_directory description.
        match = _match_mcp_tool("list directory contents", schemas)
        self.assertIsNotNone(match)
        self.assertEqual(match["tool"], "mcp.fs-test.list_directory")

    def test_mcp_tool_matching_ignores_non_mcp_tools(self):
        """_match_mcp_tool only considers tools whose name starts with mcp."""
        from lucy_edge.agent.planner_provider import _match_mcp_tool

        schemas = [
            {"name": "files.read_scoped", "description": "read a file"},
            {"name": "mcp.fs-test.read_file", "description": "read a file"},
        ]
        match = _match_mcp_tool("read a file", schemas)
        self.assertIsNotNone(match)
        self.assertEqual(match["tool"], "mcp.fs-test.read_file")


class RemotePlanningRoutingTests(unittest.IsolatedAsyncioTestCase):
    """Prove the ModelPlannerProvider routes remote planning requests to the
    configured remote host on ARM, while localhost remains denied."""

    def test_resolve_target_host_remote_url(self):
        """A remote Ollama URL matching a configured remote host resolves to
        that host's ID."""
        from lucy_edge.agent.planner_provider import ModelPlannerProvider
        from lucy_edge.config import RemoteHostConfig

        config = MagicMock()
        config.providers.ollama_base_url = "http://10.202.5.66:11434"
        config.remote_hosts = [
            RemoteHostConfig(
                host_id="win-laptop",
                role="LAPTOP",
                provider="ollama",
                base_url="http://10.202.5.66:11434",
            )
        ]
        provider = ModelPlannerProvider(config, MagicMock(), MagicMock())
        self.assertEqual(provider._resolve_target_host(), "win-laptop")

    def test_resolve_target_host_localhost_returns_none(self):
        """A localhost Ollama URL must NOT resolve to a remote host, so the
        ARM guard continues to block local inference."""
        from lucy_edge.agent.planner_provider import ModelPlannerProvider

        config = MagicMock()
        config.providers.ollama_base_url = "http://127.0.0.1:11434"
        config.remote_hosts = []
        provider = ModelPlannerProvider(config, MagicMock(), MagicMock())
        self.assertIsNone(provider._resolve_target_host())

    def test_resolve_target_host_no_match_returns_none(self):
        """A remote URL that matches no configured host returns None."""
        from lucy_edge.agent.planner_provider import ModelPlannerProvider

        config = MagicMock()
        config.providers.ollama_base_url = "http://10.202.5.66:11434"
        config.remote_hosts = []
        provider = ModelPlannerProvider(config, MagicMock(), MagicMock())
        self.assertIsNone(provider._resolve_target_host())

    async def test_remote_planning_routes_to_laptop_on_arm(self):
        """On ARM, a remote planning request with a resolved target host routes
        to the Windows laptop (REMOTE), not denied."""
        from unittest.mock import MagicMock, patch

        from lucy_edge.agent.planner_provider import ModelPlannerProvider
        from lucy_edge.config import RemoteHostConfig
        from lucy_edge.routing.policy import RoutingDecision

        config = MagicMock()
        config.providers.ollama_base_url = "http://10.202.5.66:11434"
        config.providers.default_provider = "ollama"
        config.routing.default_model = "qwen3:1.7b"
        config.host_role = "PHONE"
        config.host_id = "phone-1"
        config.remote_hosts = [
            RemoteHostConfig(
                host_id="win-laptop",
                role="LAPTOP",
                provider="ollama",
                base_url="http://10.202.5.66:11434",
            )
        ]

        # Router that allows the request and returns ROUTE.
        router = MagicMock()
        router.route = AsyncMock(
            return_value=MagicMock(decision=RoutingDecision.ROUTE)
        )
        provider = ModelPlannerProvider(config, router, MagicMock())

        plan = provider.generate_plan("read a file", ["memory.search"], AgentLimits())
        # Should NOT fall back to mock (which would produce a real plan step).
        # Verify the router was called with target_host set.
        router.route.assert_called_once()
        request_arg = router.route.call_args[0][0]
        self.assertEqual(request_arg.target_host, "win-laptop")

    async def test_localhost_planning_denied_on_arm_falls_back(self):
        """On ARM with a localhost URL, routing is denied and the provider
        falls back to the MockPlannerProvider (safe)."""
        from unittest.mock import MagicMock

        from lucy_edge.agent.planner_provider import ModelPlannerProvider
        from lucy_edge.routing.policy import RoutingDecision, ReasonCode

        config = MagicMock()
        config.providers.ollama_base_url = "http://127.0.0.1:11434"
        config.providers.default_provider = "ollama"
        config.routing.default_model = "qwen3:1.7b"
        config.host_role = "PHONE"
        config.host_id = "phone-1"
        config.remote_hosts = []

        router = MagicMock()
        router.route = AsyncMock(
            return_value=MagicMock(
                decision=RoutingDecision.DENY,
                reason_code=ReasonCode.ARM_LOCAL_INFERENCE_LOCKED,
            )
        )
        provider = ModelPlannerProvider(config, router, MagicMock())

        plan = provider.generate_plan(
            "check system health", ["memory.search", "system.health"], AgentLimits()
        )
        # Falls back to MockPlannerProvider → produces a valid plan.
        self.assertTrue(plan.steps)
        tools_used = [s.tool for s in plan.steps if s.tool]
        self.assertIn("system.health", tools_used)

    async def test_unavailable_remote_falls_back_safely(self):
        """When the remote host is unreachable (provider offline), the provider
        falls back to the MockPlannerProvider."""
        from unittest.mock import MagicMock

        from lucy_edge.agent.planner_provider import ModelPlannerProvider
        from lucy_edge.config import RemoteHostConfig
        from lucy_edge.routing.policy import RoutingDecision, ReasonCode

        config = MagicMock()
        config.providers.ollama_base_url = "http://10.202.5.66:11434"
        config.providers.default_provider = "ollama"
        config.routing.default_model = "qwen3:1.7b"
        config.host_role = "PHONE"
        config.host_id = "phone-1"
        config.remote_hosts = [
            RemoteHostConfig(
                host_id="win-laptop",
                role="LAPTOP",
                provider="ollama",
                base_url="http://10.202.5.66:11434",
            )
        ]

        router = MagicMock()
        router.route = AsyncMock(
            return_value=MagicMock(
                decision=RoutingDecision.DENY,
                reason_code=ReasonCode.PROVIDER_OFFLINE,
            )
        )
        provider = ModelPlannerProvider(config, router, MagicMock())

        plan = provider.generate_plan(
            "check system health", ["memory.search", "system.health"], AgentLimits()
        )
        # Falls back to MockPlannerProvider → valid plan.
        self.assertTrue(plan.steps)
        tools_used = [s.tool for s in plan.steps if s.tool]
        self.assertIn("system.health", tools_used)


class RemoteHostProbingTests(unittest.IsolatedAsyncioTestCase):
    """Prove that _probe_remote_hosts registers only healthy remote hosts."""

    async def test_healthy_remote_host_becomes_registered(self):
        """A reachable Ollama host is probed and promoted to REGISTERED."""
        from lucy_edge.config import LucyEdgeConfig, RemoteHostConfig
        from lucy_edge.routing.hosts import HostStatus
        from lucy_edge.services import build_services

        transport = FakeTransport()
        transport.on("GET", "/api/version", {"version": "0.32.6"})
        transport.on("GET", "/api/tags", {"models": []})

        config = LucyEdgeConfig()
        config.providers.ollama_base_url = "http://10.202.5.66:11434"
        config.remote_hosts = [
            RemoteHostConfig(
                host_id="win-laptop",
                role="LAPTOP",
                provider="ollama",
                base_url="http://10.202.5.66:11434",
            )
        ]

        services = build_services(config, transport=transport)
        await services.open()
        try:
            host = services.hosts.get("win-laptop")
            self.assertIsNotNone(host)
            self.assertEqual(host.status, HostStatus.REGISTERED)
            self.assertTrue(host.is_usable)
        finally:
            await services.close()

    async def test_unreachable_host_remains_unknown(self):
        """An unreachable Ollama host is left UNKNOWN (not trusted)."""
        from lucy_edge.config import LucyEdgeConfig, RemoteHostConfig
        from lucy_edge.routing.hosts import HostStatus
        from lucy_edge.services import build_services

        transport = FakeTransport(fail_with=Exception("connection refused"))

        config = LucyEdgeConfig()
        config.providers.ollama_base_url = "http://192.168.99.99:11434"
        config.remote_hosts = [
            RemoteHostConfig(
                host_id="dead-host",
                role="LAPTOP",
                provider="ollama",
                base_url="http://192.168.99.99:11434",
            )
        ]

        services = build_services(config, transport=transport)
        await services.open()
        try:
            host = services.hosts.get("dead-host")
            self.assertIsNotNone(host)
            self.assertEqual(host.status, HostStatus.UNKNOWN)
            self.assertFalse(host.is_usable)
        finally:
            await services.close()

    async def test_probed_host_planner_routes_to_laptop(self):
        """After probing registers the host, the planner's routing request
        includes target_host=win-laptop and the router routes to it."""
        from unittest.mock import MagicMock

        from lucy_edge.agent.planner_provider import ModelPlannerProvider
        from lucy_edge.config import RemoteHostConfig
        from lucy_edge.routing.policy import RoutingDecision

        config = MagicMock()
        config.providers.ollama_base_url = "http://10.202.5.66:11434"
        config.providers.default_provider = "ollama"
        config.routing.default_model = "qwen3:1.7b"
        config.host_role = "PHONE"
        config.host_id = "phone-1"
        config.remote_hosts = [
            RemoteHostConfig(
                host_id="win-laptop",
                role="LAPTOP",
                provider="ollama",
                base_url="http://10.202.5.66:11434",
            )
        ]

        router = MagicMock()
        router.route = AsyncMock(
            return_value=MagicMock(decision=RoutingDecision.ROUTE)
        )
        provider = ModelPlannerProvider(config, router, MagicMock())

        provider.generate_plan("read a file", ["memory.search"], AgentLimits())
        request_arg = router.route.call_args[0][0]
        self.assertEqual(request_arg.target_host, "win-laptop")


if __name__ == "__main__":
    unittest.main()
