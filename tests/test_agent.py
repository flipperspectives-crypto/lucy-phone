"""Bounded agent runtime tests: limits, timeouts, approvals."""

from __future__ import annotations

import asyncio
import unittest

from lucy_edge.agent.limits import AgentLimits
from lucy_edge.agent.planner import Plan, PlanStep, RulePlanner
from lucy_edge.agent.runtime import AgentRuntime, AgentState
from lucy_edge.services import build_services
from lucy_edge.tools.permissions import PermissionDecision, PermissionOutcome, PermissionPolicy
from lucy_edge.tools.registry import ToolRegistry, ToolSpec

from .helpers import make_config, temp_dir


async def _slow_tool(sleep=1.0, **kw):
    await asyncio.sleep(sleep)
    return "slow-done"


def _test_tool_policy(tmp: str) -> PermissionPolicy:
    """Like the phone policy but lets a test-only tool run without approval."""

    class TestPolicy(PermissionPolicy):
        def evaluate(self, tool, args):
            if tool in ("tests.slow", "tests.ok"):
                return PermissionDecision(PermissionOutcome.ALLOW, "test tool")
            return super().evaluate(tool, args)

    policy = TestPolicy(approved_roots=[tmp])
    policy.write_auto_allow = False
    return policy


def _plan_with(steps: list[PlanStep]) -> RulePlanner:
    class CustomPlanner(RulePlanner):
        def __init__(self):
            super().__init__(AgentLimits())

        def build_plan(self, goal, available_tools, tool_schemas=None):
            return Plan(goal=goal, steps=steps)

    return CustomPlanner()


class PlannerDegradationTests(unittest.IsolatedAsyncioTestCase):
    def test_fallback_marks_plan_degraded_and_records(self):
        from lucy_edge.agent.planner import ModelDrivenPlanner

        class FailingProvider:
            def generate_plan(self, *a, **k):
                raise RuntimeError("model planner boom")

        sink_calls = []
        planner = ModelDrivenPlanner(
            AgentLimits(), FailingProvider(), fallback_sink=lambda r: sink_calls.append(r)
        )
        plan = planner.build_plan("do something", ["memory.search"])
        self.assertTrue(plan.degraded)
        self.assertIsNotNone(plan.degradation_note)
        self.assertEqual(len(sink_calls), 1)

    def test_empty_model_plan_degrades(self):
        from lucy_edge.agent.planner import ModelDrivenPlanner, Plan, PlanStep

        class EmptyProvider:
            def generate_plan(self, *a, **k):
                return Plan(goal="x", steps=[])

        planner = ModelDrivenPlanner(AgentLimits(), EmptyProvider())
        plan = planner.build_plan("goal", ["memory.search"])
        self.assertTrue(plan.degraded)

    def test_unknown_tool_in_model_plan_degrades(self):
        from lucy_edge.agent.planner import ModelDrivenPlanner, Plan, PlanStep

        class BadToolProvider:
            def generate_plan(self, *a, **k):
                return Plan(goal="x", steps=[PlanStep(index=0, action="execute", tool="nope")])

        planner = ModelDrivenPlanner(AgentLimits(), BadToolProvider())
        plan = planner.build_plan("goal", ["memory.search"])
        self.assertTrue(plan.degraded)


class AgentLimitsTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_timeout_must_be_shorter_than_task_timeout(self):
        with self.assertRaises(ValueError):
            AgentLimits(task_timeout=5.0, tool_timeout=6.0)

    async def test_default_limits_valid(self):
        limits = AgentLimits()
        self.assertGreater(limits.max_steps, 0)
        self.assertGreater(limits.max_tool_calls, 0)
        self.assertGreater(limits.task_timeout, 0.0)


class AgentRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def _services(self, tmp: str):
        services = build_services(make_config(tmp))
        await services.open()
        return services

    async def _runtime(self, tmp, plan_steps, limits, extra_tools=()):
        services = await self._services(tmp)
        policy = _test_tool_policy(tmp)
        registry = ToolRegistry(policy)
        from lucy_edge.tools.builtin.core import register_builtin_tools

        register_builtin_tools(registry, services.context)
        for name, func in extra_tools:
            registry.register(ToolSpec(name, "test tool", func, "read"))
        runtime = AgentRuntime(
            run_id="11111111111111111111111111111111",
            goal="test goal",
            limits=limits,
            registry=registry,
            planner=_plan_with(plan_steps),
            evidence=services.evidence,
            context=services.context,
        )
        return runtime, services

    async def test_max_steps_enforced(self):
        tmp = temp_dir()
        steps = [
            PlanStep(i, "execute", "system.health", {}, "health")
            for i in range(6)
        ]
        limits = AgentLimits(max_steps=3, max_tool_calls=50, task_timeout=30.0)
        runtime, services = await self._runtime(tmp, steps, limits)
        try:
            result = await runtime.run()
            self.assertEqual(result.final_status, AgentState.COMPLETED)
            self.assertEqual(result.completion_reason, "max_steps reached")
            self.assertEqual(result.steps_executed, 3)
        finally:
            await services.close()

    async def test_max_tool_calls_enforced(self):
        tmp = temp_dir()
        steps = [
            PlanStep(i, "execute", "system.health", {}, "health")
            for i in range(6)
        ]
        limits = AgentLimits(max_steps=50, max_tool_calls=2, task_timeout=30.0)
        runtime, services = await self._runtime(tmp, steps, limits)
        try:
            result = await runtime.run()
            self.assertEqual(result.final_status, AgentState.FAILED)
            self.assertEqual(result.completion_reason, "max_tool_calls reached")
            self.assertEqual(result.tool_calls, 2)
        finally:
            await services.close()

    async def test_task_timeout_enforced(self):
        tmp = temp_dir()
        steps = [
            PlanStep(i, "execute", "tests.slow", {}, "slow")
            for i in range(4)
        ]
        limits = AgentLimits(
            max_steps=10,
            max_tool_calls=10,
            max_failures=10,
            task_timeout=0.4,
            tool_timeout=0.15,
        )
        runtime, services = await self._runtime(
            tmp, steps, limits, extra_tools=[("tests.slow", _slow_tool)]
        )
        try:
            result = await runtime.run()
            self.assertEqual(result.final_status, AgentState.TIMED_OUT)
            self.assertIn("timeout", result.completion_reason)
        finally:
            await services.close()

    async def test_tool_timeout_marks_step_timed_out(self):
        tmp = temp_dir()
        steps = [PlanStep(0, "execute", "tests.slow", {}, "slow")]
        limits = AgentLimits(
            max_steps=2,
            max_tool_calls=2,
            max_failures=5,
            task_timeout=5.0,
            tool_timeout=0.1,
        )
        runtime, services = await self._runtime(
            tmp, steps, limits, extra_tools=[("tests.slow", _slow_tool)]
        )
        try:
            result = await runtime.run()
            self.assertEqual(result.final_status, AgentState.COMPLETED)
            self.assertEqual(result.tool_calls, 1)
        finally:
            await services.close()

    async def test_approval_ask_flow(self):
        tmp = temp_dir()
        workspace = tmp
        services = build_services(make_config(tmp))
        await services.open()
        try:
            policy = PermissionPolicy(approved_roots=[workspace])
            registry = ToolRegistry(policy)
            from lucy_edge.tools.builtin.core import register_builtin_tools

            register_builtin_tools(registry, services.context)
            target = f"{workspace}/notes.txt"
            steps = [
                PlanStep(0, "execute", "files.write_scoped", {"path": target, "content": "x"}, "write")
            ]
            runtime = AgentRuntime(
                run_id="22222222222222222222222222222222",
                goal="write a note",
                limits=AgentLimits(max_steps=4, max_tool_calls=4, task_timeout=10.0, tool_timeout=3.0),
                registry=registry,
                planner=_plan_with(steps),
                evidence=services.evidence,
                context=services.context,
            )
            run_task = asyncio.create_task(runtime.run())
            await asyncio.sleep(0.05)
            self.assertEqual(runtime.state, AgentState.WAITING_APPROVAL)
            runtime.approve()
            result = await run_task
            self.assertEqual(result.final_status, AgentState.COMPLETED)
            import pathlib

            self.assertTrue(pathlib.Path(target).exists())
        finally:
            await services.close()

    async def test_approval_deny_flow(self):
        tmp = temp_dir()
        workspace = tmp
        services = build_services(make_config(tmp))
        await services.open()
        try:
            policy = PermissionPolicy(approved_roots=[workspace])
            registry = ToolRegistry(policy)
            from lucy_edge.tools.builtin.core import register_builtin_tools

            register_builtin_tools(registry, services.context)
            target = f"{workspace}/notes2.txt"
            steps = [
                PlanStep(0, "execute", "files.write_scoped", {"path": target, "content": "x"}, "write")
            ]
            runtime = AgentRuntime(
                run_id="33333333333333333333333333333333",
                goal="write a note",
                limits=AgentLimits(max_steps=4, max_tool_calls=4, task_timeout=10.0, tool_timeout=3.0),
                registry=registry,
                planner=_plan_with(steps),
                evidence=services.evidence,
                context=services.context,
            )
            run_task = asyncio.create_task(runtime.run())
            await asyncio.sleep(0.05)
            self.assertEqual(runtime.state, AgentState.WAITING_APPROVAL)
            runtime.deny()
            result = await run_task
            self.assertEqual(result.final_status, AgentState.DENIED)
        finally:
            await services.close()

    async def test_invalid_transition_fails_clearly(self):
        tmp = temp_dir()
        services = build_services(make_config(tmp))
        await services.open()
        try:
            runtime = AgentRuntime(
                run_id="44444444444444444444444444444444",
                goal="x",
                limits=AgentLimits(),
                registry=ToolRegistry(PermissionPolicy()),
            )
            with self.assertRaises(Exception):
                runtime.transition(AgentState.COMPLETED)
            runtime.transition(AgentState.PLANNING)
            runtime.transition(AgentState.RUNNING)
            self.assertEqual(runtime.state, AgentState.RUNNING)
        finally:
            await services.close()


if __name__ == "__main__":
    unittest.main()
