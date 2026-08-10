"""Step executor with tool-timeout enforcement."""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from typing import Any, Optional

from ..tools.registry import ToolRegistry
from .planner import PlanStep


@dataclass
class StepResult:
    step_index: int
    action: str
    status: str  # OK | FAILED | DENIED | TIMED_OUT | SKIPPED
    tool: Optional[str] = None
    output: Any = None
    output_sha256: Optional[str] = None
    error: Optional[str] = None
    duration_ms: Optional[float] = None
    permission_outcome: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "action": self.action,
            "status": self.status,
            "tool": self.tool,
            "output": self.output,
            "output_sha256": self.output_sha256,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "permission_outcome": self.permission_outcome,
        }


class Executor:
    def __init__(self, registry: ToolRegistry, tool_timeout: float) -> None:
        self.registry = registry
        self.tool_timeout = tool_timeout

    async def run_step(self, step: PlanStep, context: Any) -> StepResult:
        if step.action == "stop":
            return StepResult(step.index, step.action, "OK", step.tool)
        if step.action == "verify":
            return StepResult(step.index, step.action, "OK", step.tool)
        if step.action == "retrieve_memory":
            tool_result = await self.registry.execute(step.tool, step.args, context)
            return self._from_tool(step, tool_result)
        if step.action == "execute":
            t0 = time.monotonic()
            try:
                tool_result = await asyncio.wait_for(
                    self.registry.execute(step.tool, step.args, context),
                    timeout=self.tool_timeout,
                )
            except asyncio.TimeoutError:
                return StepResult(
                    step.index,
                    step.action,
                    "TIMED_OUT",
                    step.tool,
                    error=f"tool exceeded {self.tool_timeout}s timeout",
                    duration_ms=round((time.monotonic() - t0) * 1000.0, 3),
                )
            return self._from_tool(step, tool_result)
        return StepResult(
            step.index, step.action, "FAILED", step.tool, error=f"unknown action: {step.action}"
        )

    @staticmethod
    def _from_tool(step: PlanStep, result: Any) -> StepResult:
        if result.denied:
            return StepResult(
                step.index,
                step.action,
                "DENIED",
                step.tool,
                error=result.reason,
                permission_outcome="DENY",
                duration_ms=result.duration_ms,
            )
        if not result.ok:
            return StepResult(
                step.index,
                step.action,
                "FAILED",
                step.tool,
                error=result.error or result.reason,
                duration_ms=result.duration_ms,
            )
        return StepResult(
            step.index,
            step.action,
            "OK",
            step.tool,
            output=result.output,
            output_sha256=result.output_sha256,
            duration_ms=result.duration_ms,
        )
