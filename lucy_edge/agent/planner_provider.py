"""Pluggable planner-provider interface for the model-driven planner.

A ``PlannerProvider`` produces a ``Plan`` from a goal and the list of available
tools.  The interface is deliberately synchronous to match the planner's
``build_plan()`` contract.

Phone safety
------------
``MockPlannerProvider`` is the phone-safe default: it returns a bounded,
deterministic plan with NO model inference, NO thermal impact, and NO network.
It is used whenever the host role is PHONE (``phone_local_inference_enabled``
defaults to FALSE).

``ModelPlannerProvider`` calls a real language model, but ONLY after the
``ModelRouter`` authorises the planning-model request.  On a phone with local
inference disabled the router denies the request and the provider transparently
falls back to ``MockPlannerProvider``.  This class must therefore NEVER bypass
the routing policy or the thermal governor.
"""

from __future__ import annotations

import abc
import asyncio
import json
import re
from typing import Any, Optional

from .limits import AgentLimits
from .planner import Plan, PlanStep

_VALID_ACTIONS = ("execute", "retrieve_memory", "verify", "stop")


class PlannerProvider(abc.ABC):
    """Synchronous provider that produces a ``Plan`` from a goal + tools."""

    name: str = "abstract"

    @abc.abstractmethod
    def generate_plan(self, goal: str, available_tools: list[str], limits: AgentLimits) -> Plan:
        """Return a ``Plan``.  Must be synchronous and never raise."""
        ...


class MockPlannerProvider(PlannerProvider):
    """Deterministic, phone-safe provider.  No model inference.

    Produces a bounded plan using goal-aware keyword matching (the same
    vocabulary as the rule-based planner).  Used on phones and as a guaranteed
    fallback when a real model is unavailable or denied by routing.
    """

    name = "mock"

    def generate_plan(self, goal: str, available_tools: list[str], limits: AgentLimits) -> Plan:
        steps: list[PlanStep] = []
        index = 0

        def add(action: str, tool: Optional[str], args: dict[str, Any], desc: str) -> None:
            nonlocal index
            steps.append(
                PlanStep(index=index, action=action, tool=tool, args=args, description=desc)
            )
            index += 1

        if "memory.search" in available_tools:
            add("retrieve_memory", "memory.search", {"query": goal}, "retrieve relevant memory")

        goal_lower = goal.lower()
        executed = False
        for keyword, tool, arg_key, desc in _PROVIDER_KEYWORDS:
            if keyword in goal_lower and tool in available_tools:
                args = {arg_key: goal} if arg_key else {}
                add("execute", tool, args, desc)
                executed = True
                break

        if not executed and "system.capabilities" in available_tools:
            add("execute", "system.capabilities", {}, "inspect verified runtime capabilities")

        add("verify", None, {}, "verify step outputs")
        add("stop", None, {}, "stop after bounded plan")

        steps = steps[: limits.max_steps]
        for i, s in enumerate(steps):
            s.index = i
        return Plan(goal=goal, steps=steps)


_PROVIDER_KEYWORDS = [
    ("memory", "memory.search", "query", "search persistent memory"),
    ("evidence", "evidence.query", "record_type", "query the evidence ledger"),
    ("route", "model.route", "model", "request a routing decision"),
    ("model", "model.route", "model", "request a routing decision"),
    ("health", "system.health", "", "check runtime health"),
    ("capabilit", "system.capabilities", "", "report verified runtime capabilities"),
    ("file", "files.read_scoped", "path", "read a scoped file"),
    ("git", "git.status", "", "inspect repository state"),
]


class ModelPlannerProvider(PlannerProvider):
    """Real model-driven provider.

    Calls a language model to generate a plan, but ONLY after the
    ``ModelRouter`` authorises the planning-model request.  On a phone with
    local inference disabled the router denies the request and the provider
    transparently falls back to ``MockPlannerProvider``.  JSON output is parsed
    and every step is validated against ``available_tools``.
    """

    name = "model"

    def __init__(self, config: Any, router: Any, providers: Any) -> None:
        self.config = config
        self.router = router
        self.providers = providers
        self._fallback = MockPlannerProvider()
        self._planning_model = config.routing.default_model

    def generate_plan(self, goal: str, available_tools: list[str], limits: AgentLimits) -> Plan:
        try:
            return self._try_model_plan(goal, available_tools, limits)
        except Exception:
            return self._fallback.generate_plan(goal, available_tools, limits)

    def _try_model_plan(
        self, goal: str, available_tools: list[str], limits: AgentLimits
    ) -> Plan:
        from ..routing.hosts import HostRole
        from ..routing.policy import RoutingDecision, RoutingRequest

        request = RoutingRequest(
            model=self._planning_model,
            provider=self.config.providers.default_provider,
            host_role=(
                HostRole(self.config.host_role)
                if self.config.host_role in HostRole.__members__
                else HostRole.UNKNOWN
            ),
            host_id=self.config.host_id,
        )

        result = self._run(self.router.route(request))
        if result.decision != RoutingDecision.ALLOW:
            return self._fallback.generate_plan(goal, available_tools, limits)

        provider = self.providers.get(request.provider)
        if provider is None:
            return self._fallback.generate_plan(goal, available_tools, limits)

        prompt = self._build_prompt(goal, available_tools, limits)
        response = self._run(
            provider.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self._planning_model,
            )
        )
        return self._parse_response(response.message, goal, available_tools, limits)

    def _build_prompt(self, goal: str, available_tools: list[str], limits: AgentLimits) -> str:
        return (
            "You are a bounded planning agent.  Produce a JSON plan only.\n"
            f"Goal: {goal}\n"
            f"Available tools (use ONLY these): {', '.join(available_tools)}\n"
            f"Max steps: {limits.max_steps}\n"
            "Schema:\n"
            '  {"steps": [{"action": "execute|retrieve_memory|verify|stop", '
            '"tool": "<tool_name_or_null>", "args": {}, "description": "..."}]}\n'
            "- Every tool must be from the available-tools list.\n"
            "- End with {\"action\",\"stop\"}.\n"
            "- Output ONLY the JSON object, no commentary."
        )

    def _parse_response(
        self, raw: str, goal: str, available_tools: list[str], limits: AgentLimits
    ) -> Plan:
        data = self._extract_json(raw)
        if data is None:
            return self._fallback.generate_plan(goal, available_tools, limits)

        raw_steps = data.get("steps", [])
        steps: list[PlanStep] = []
        for entry in raw_steps:
            if len(steps) >= limits.max_steps:
                break
            if not isinstance(entry, dict):
                continue
            tool = entry.get("tool")
            if tool is not None and not isinstance(tool, str):
                continue
            if tool is not None and tool not in available_tools:
                continue
            action = entry.get("action", "execute")
            if action not in _VALID_ACTIONS:
                action = "execute"
            args = entry.get("args", {})
            if not isinstance(args, dict):
                args = {}
            steps.append(
                PlanStep(
                    index=len(steps),
                    action=action,
                    tool=tool,
                    args=args,
                    description=str(entry.get("description", "")),
                )
            )

        if not steps:
            return self._fallback.generate_plan(goal, available_tools, limits)

        if steps[-1].action != "stop":
            if len(steps) < limits.max_steps:
                steps.append(
                    PlanStep(
                        index=len(steps),
                        action="stop",
                        tool=None,
                        args={},
                        description="stop after bounded plan",
                    )
                )
            else:
                steps[-1] = PlanStep(
                    index=limits.max_steps - 1,
                    action="stop",
                    tool=None,
                    args={},
                    description="stop after bounded plan",
                )

        steps = steps[: limits.max_steps]
        for i, s in enumerate(steps):
            s.index = i
        return Plan(goal=goal, steps=steps)

    @staticmethod
    def _extract_json(text: str) -> Optional[dict[str, Any]]:
        if not text:
            return None
        try:
            return json.loads(text.strip())
        except (json.JSONDecodeError, ValueError):
            pass
        block = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
        if block:
            try:
                return json.loads(block.group(1))
            except (json.JSONDecodeError, ValueError):
                pass
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except (json.JSONDecodeError, ValueError):
                pass
        return None

    @staticmethod
    def _run(coro: Any) -> Any:
        """Run an async coroutine from a synchronous call without disturbing
        any already-running event loop (the planner is invoked from inside
        ``AgentRuntime.run``)."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        # Already inside an event loop: use a thread + its own loop.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
