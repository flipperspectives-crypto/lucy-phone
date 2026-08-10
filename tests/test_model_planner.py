"""Model-driven planner tests.

These tests use the deterministic MockPlannerProvider — NO real model
inference, NO network, safe for the S25 Ultra phone.  The ModelPlannerProvider
path that calls a live model is covered only by a unit-level check that its
constructor and fallback wiring are intact; exercising a real model is NOT
done here (requires a running inference host).
"""

from __future__ import annotations

import unittest

from lucy_edge.agent.limits import AgentLimits
from lucy_edge.agent.planner import ModelDrivenPlanner, Plan, PlanStep, RulePlanner
from lucy_edge.agent.planner_provider import (
    MockPlannerProvider,
    ModelPlannerProvider,
    PlannerProvider,
)
from lucy_edge.agent.runtime import AgentRuntime, AgentState
from lucy_edge.services import build_services

from .helpers import make_config, temp_dir


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


if __name__ == "__main__":
    unittest.main()
