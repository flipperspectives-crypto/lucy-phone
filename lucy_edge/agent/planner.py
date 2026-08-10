"""Planner module: rule-based and model-driven planners.

The plan is always bounded by max_steps.  Two planners are available:

* ``RulePlanner`` — deterministic, keyword-matched, labeled RULE_BASED.
* ``ModelDrivenPlanner`` — calls a pluggable ``PlannerProvider`` (mock or real
  model) and falls back to ``RulePlanner`` on any failure.  Reported as
  MODEL_DRIVEN in introspection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .limits import AgentLimits


@dataclass
class PlanStep:
    index: int
    action: str
    tool: Optional[str] = None
    args: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "action": self.action,
            "tool": self.tool,
            "args": self.args,
            "description": self.description,
        }


@dataclass
class Plan:
    goal: str
    steps: list[PlanStep] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"goal": self.goal, "steps": [s.as_dict() for s in self.steps]}


_GOAL_KEYWORDS: dict[str, tuple[str, str, dict[str, Any], str]] = {
    "memory": ("memory.search", "query", {}, "search relevant persistent memory"),
    "evidence": ("evidence.query", "record_type", {}, "query the evidence ledger"),
    "model": ("model.route", "model", {}, "request a routing decision"),
    "route": ("model.route", "model", {}, "request a routing decision"),
    "health": ("system.health", "", {}, "check runtime health"),
    "capabilit": ("system.capabilities", "", {}, "report verified runtime capabilities"),
    "file": ("files.read_scoped", "path", {}, "read a scoped file"),
    "git": ("git.status", "", {}, "inspect repository state"),
}


class RulePlanner:
    """Deterministic planner with an explicit bounded default plan."""

    backend = "RULE_BASED"

    def __init__(self, limits: AgentLimits) -> None:
        self.limits = limits

    def build_plan(self, goal: str, available_tools: list[str]) -> Plan:
        goal_lower = goal.lower()
        steps: list[PlanStep] = []
        index = 0

        def add(action: str, tool: Optional[str], args: dict[str, Any], desc: str) -> None:
            nonlocal index
            steps.append(
                PlanStep(index=index, action=action, tool=tool, args=args, description=desc)
            )
            index += 1

        add("retrieve_memory", "memory.search", {"query": goal}, "retrieve relevant memory")

        matched = False
        for keyword, (tool, arg, args, desc) in _GOAL_KEYWORDS.items():
            if keyword in goal_lower and tool in available_tools:
                resolved_args = dict(args)
                if arg:
                    resolved_args[arg] = goal
                add("execute", tool, resolved_args, desc)
                matched = True
                break

        if not matched:
            if "system.capabilities" in available_tools:
                add(
                    "execute",
                    "system.capabilities",
                    {},
                    "inspect verified runtime capabilities",
                )

        add("verify", None, {}, "verify step outputs against expectations")
        add("record_evidence", "evidence.query", {"limit": 1}, "record run evidence")
        add("stop", None, {}, "stop after bounded plan")

        # Enforce max_steps.
        steps = steps[: self.limits.max_steps]
        for i, step in enumerate(steps):
            step.index = i
        return Plan(goal=goal, steps=steps)


class ModelDrivenPlanner:
    """Model-driven planner with a rule-based fallback.

    Delegates to a ``PlannerProvider`` (mock or real model).  The returned plan
    is validated: every tool must be in ``available_tools`` and the step count
    is bounded by ``max_steps``.  On any failure — provider error, empty plan,
    unknown tool — it transparently falls back to ``RulePlanner``.
    """

    backend = "MODEL_DRIVEN"

    def __init__(
        self,
        limits: AgentLimits,
        provider: Any,
        fallback: Optional[RulePlanner] = None,
    ) -> None:
        self.limits = limits
        self.provider = provider
        self.fallback = fallback or RulePlanner(limits)

    def build_plan(self, goal: str, available_tools: list[str]) -> Plan:
        try:
            plan = self.provider.generate_plan(goal, available_tools, self.limits)
        except Exception:
            return self.fallback.build_plan(goal, available_tools)

        if not plan.steps:
            return self.fallback.build_plan(goal, available_tools)

        for step in plan.steps:
            if step.tool and step.tool not in available_tools:
                return self.fallback.build_plan(goal, available_tools)

        steps = plan.steps[: self.limits.max_steps]
        for i, s in enumerate(steps):
            s.index = i
        return Plan(goal=goal, steps=steps)
