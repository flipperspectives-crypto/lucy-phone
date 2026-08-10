"""Rule-based bounded planner (phase 1).

The planner is deterministic and labeled RULE_BASED in introspection.  It is
NOT a language-model planner; a model-driven planner would be reported as a
separate capability later.  The plan is always bounded by max_steps.
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
