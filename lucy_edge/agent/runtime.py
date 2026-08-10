"""Bounded agent execution runtime with an observable state machine.

The loop: GOAL -> PLAN -> RETRIEVE MEMORY -> SELECT TOOL -> EXECUTE -> OBSERVE
-> VERIFY -> RECORD EVIDENCE -> CONTINUE OR STOP.

No infinite loops, no uncontrolled recursion, no automatic continuous
evolution, no hidden retry storms.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ..evidence.schema import EvidenceRecord, EvidenceType
from ..tools.permissions import PermissionOutcome
from ..tools.registry import ToolRegistry
from .executor import Executor
from .limits import AgentLimits
from .planner import Plan, PlanStep
from .verifier import Verifier


class AgentState(str, Enum):
    CREATED = "CREATED"
    PLANNING = "PLANNING"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    VERIFYING = "VERIFYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    DENIED = "DENIED"
    TIMED_OUT = "TIMED_OUT"
    ABORTED = "ABORTED"


_TRANSITIONS: dict[AgentState, set[AgentState]] = {
    AgentState.CREATED: {AgentState.PLANNING, AgentState.ABORTED},
    AgentState.PLANNING: {AgentState.RUNNING, AgentState.FAILED, AgentState.ABORTED},
    AgentState.RUNNING: {
        AgentState.WAITING_APPROVAL,
        AgentState.VERIFYING,
        AgentState.COMPLETED,
        AgentState.FAILED,
        AgentState.TIMED_OUT,
        AgentState.ABORTED,
    },
    AgentState.WAITING_APPROVAL: {
        AgentState.RUNNING,
        AgentState.DENIED,
        AgentState.ABORTED,
        AgentState.TIMED_OUT,
    },
    AgentState.VERIFYING: {
        AgentState.RUNNING,
        AgentState.COMPLETED,
        AgentState.FAILED,
        AgentState.ABORTED,
        AgentState.TIMED_OUT,
    },
    AgentState.COMPLETED: set(),
    AgentState.FAILED: set(),
    AgentState.DENIED: set(),
    AgentState.TIMED_OUT: set(),
    AgentState.ABORTED: set(),
}


class InvalidTransition(Exception):
    def __init__(self, state: AgentState, target: AgentState) -> None:
        super().__init__(f"invalid transition {state.value} -> {target.value}")
        self.state = state
        self.target = target


@dataclass
class AgentRunResult:
    run_id: str
    goal: str
    final_status: AgentState
    completion_reason: str
    steps_executed: int
    tool_calls: int
    failures: int
    duration_ms: float
    evidence_run_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "final_status": self.final_status.value,
            "completion_reason": self.completion_reason,
            "steps_executed": self.steps_executed,
            "tool_calls": self.tool_calls,
            "failures": self.failures,
            "duration_ms": round(self.duration_ms, 3),
            "evidence_run_id": self.evidence_run_id,
        }


class AgentRuntime:
    def __init__(
        self,
        run_id: str,
        goal: str,
        limits: AgentLimits,
        registry: ToolRegistry,
        planner: Any = None,
        verifier: Optional[Verifier] = None,
        evidence: Any = None,
        memory_retrieval: Any = None,
        context: Any = None,
    ) -> None:
        self.run_id = run_id
        self.goal = goal
        self.limits = limits
        self.registry = registry
        if planner is not None:
            self.planner = planner
        else:
            from .planner import RulePlanner

            self.planner = RulePlanner(limits)
        self.verifier = verifier or Verifier(limits.max_output_chars)
        self.evidence = evidence
        self.memory_retrieval = memory_retrieval
        self.context = context

        self.state: AgentState = AgentState.CREATED
        self._plan: Optional[Plan] = None
        self.steps_executed = 0
        self.tool_calls = 0
        self.failures = 0
        self._started_at: Optional[float] = None
        self._deadline: Optional[float] = None
        self._approval_event = asyncio.Event()
        self._approval_decision: Optional[bool] = None

        self._tool_calls_log: list[dict[str, Any]] = []
        self._permission_log: list[dict[str, Any]] = []
        self._memory_retrieval_ids: list[str] = []
        self._verifications: list[dict[str, Any]] = []
        self._errors: list[str] = []

    # --- state machine -----------------------------------------------------
    def transition(self, target: AgentState) -> None:
        if target not in _TRANSITIONS[self.state]:
            raise InvalidTransition(self.state, target)
        self.state = target

    def approve(self) -> None:
        if self.state != AgentState.WAITING_APPROVAL:
            raise InvalidTransition(self.state, AgentState.RUNNING)
        self._approval_decision = True
        self._approval_event.set()

    def deny(self) -> None:
        if self.state != AgentState.WAITING_APPROVAL:
            raise InvalidTransition(self.state, AgentState.DENIED)
        self._approval_decision = False
        self._approval_event.set()

    def abort(self) -> None:
        self.transition(AgentState.ABORTED)

    # --- execution ---------------------------------------------------------
    async def _record_step_evidence(self, step_result: Any, decision: Optional[dict]) -> None:
        if step_result is not None:
            self._tool_calls_log.append(step_result.as_dict())
        if decision is not None:
            self._permission_log.append(
                {"step": getattr(step_result, "step_index", None), "tool": getattr(step_result, "tool", None), **decision}
            )

    async def run(self) -> AgentRunResult:
        t0 = time.monotonic()
        self._started_at = t0
        self._deadline = t0 + self.limits.task_timeout
        self.transition(AgentState.PLANNING)
        plan = self.planner.build_plan(self.goal, self.registry.names())
        if not plan.steps:
            self.transition(AgentState.FAILED)
            return self._finish(t0, AgentState.FAILED, "empty plan")
        self._plan = plan
        self.transition(AgentState.RUNNING)

        final_status = AgentState.COMPLETED
        completion_reason = "plan completed"

        for step in plan.steps:
            if time.monotonic() > self._deadline:
                final_status = AgentState.TIMED_OUT
                completion_reason = f"task exceeded {self.limits.task_timeout}s timeout"
                break

            if step.action == "stop":
                break

            # Bounded steps: check BEFORE executing.
            if self.steps_executed >= self.limits.max_steps:
                final_status = AgentState.COMPLETED
                completion_reason = "max_steps reached"
                break
            self.steps_executed += 1

            if step.tool:
                # Bounded tool calls: check BEFORE executing.
                if self.tool_calls >= self.limits.max_tool_calls:
                    final_status = AgentState.FAILED
                    completion_reason = "max_tool_calls reached"
                    break
                self.tool_calls += 1

                # Permission gate: ALLOW/ASK/DENY.
                decision = self.registry.check_permission(step.tool, step.args)
                if decision.outcome == PermissionOutcome.DENY:
                    result = self._denied_step(step, decision.reason)
                    await self._record_step_evidence(result, decision.as_dict())
                    self._errors.append(decision.reason)
                    self.failures += 1
                    if self.failures > self.limits.max_failures:
                        final_status = AgentState.FAILED
                        completion_reason = "max_failures reached"
                        break
                    continue

                if decision.outcome == PermissionOutcome.ASK:
                    self.transition(AgentState.WAITING_APPROVAL)
                    self._approval_event.clear()
                    self._approval_decision = None
                    await self._approval_event.wait()
                    if self._approval_decision is False:
                        result = self._denied_step(step, "operator denied approval")
                        await self._record_step_evidence(result, decision.as_dict())
                        self.transition(AgentState.DENIED)
                        final_status = AgentState.DENIED
                        completion_reason = "operator denied approval"
                        break
                    self.transition(AgentState.RUNNING)
                    await self._record_step_evidence(None, decision.as_dict())

            executor = Executor(self.registry, self.limits.tool_timeout)
            step_result = await executor.run_step(step, self.context)
            await self._record_step_evidence(step_result, None)

            if step_result.status == "TIMED_OUT":
                self._errors.append(step_result.error or "tool timeout")
                self.failures += 1
                if self.failures > self.limits.max_failures:
                    final_status = AgentState.FAILED
                    completion_reason = "max_failures reached"
                    break
                continue
            if step_result.status == "DENIED":
                self.failures += 1
                if self.failures > self.limits.max_failures:
                    final_status = AgentState.FAILED
                    completion_reason = "max_failures reached"
                    break
                continue
            if step_result.status == "FAILED":
                self._errors.append(step_result.error or "step failed")
                self.failures += 1
                if self.failures > self.limits.max_failures:
                    final_status = AgentState.FAILED
                    completion_reason = "max_failures reached"
                    break
                continue

            # memory search results recorded for evidence
            if step.tool == "memory.search" and isinstance(step_result.output, dict):
                for item in step_result.output.get("results", []):
                    mid = item.get("memory_id")
                    if mid:
                        self._memory_retrieval_ids.append(mid)

            # VERIFY (structural/heuristic)
            self.transition(AgentState.VERIFYING)
            verification = self.verifier.verify(step_result)
            self._verifications.append(verification.as_dict())
            if not verification.ok and step_result.action in ("execute", "retrieve_memory"):
                self.failures += 1
                if self.failures > self.limits.max_failures:
                    final_status = AgentState.FAILED
                    completion_reason = "verification failed; max_failures reached"
                    break
            self.transition(AgentState.RUNNING)
        else:
            final_status = AgentState.COMPLETED
            completion_reason = "plan completed"

        if self.state not in (AgentState.DENIED, AgentState.FAILED, AgentState.TIMED_OUT, AgentState.ABORTED):
            self.transition(final_status)
        return await self._finish(t0, final_status, completion_reason)

    @staticmethod
    def _denied_step(step: PlanStep, reason: str) -> Any:
        from .executor import StepResult

        return StepResult(
            step.index, step.action, "DENIED", step.tool, error=reason, permission_outcome="DENY"
        )

    async def _finish(self, t0: float, status: AgentState, reason: str) -> AgentRunResult:
        evidence_run_id = self.run_id
        if self.evidence is not None:
            try:
                host = None
                host_role = None
                if getattr(self.context, "config", None):
                    host = self.context.config.host_id
                    host_role = self.context.config.host_role
                record = EvidenceRecord(
                    run_id=self.run_id,
                    record_type=EvidenceType.AGENT_RUN,
                    goal=self.goal,
                    run_state=status.value,
                    final_status=status.value,
                    completion_reason=reason,
                    host=host,
                    host_role=host_role,
                    plan=self._plan.as_dict()["steps"] if self._plan else None,
                    memory_retrieval_ids=self._memory_retrieval_ids,
                    tool_calls=self._tool_calls_log,
                    permission_decisions=self._permission_log,
                    errors=self._errors[:20],
                    verification={"results": self._verifications},
                    latency_ms=round((time.monotonic() - t0) * 1000.0, 3),
                )
                await self.evidence.append(record)
            except Exception as exc:
                self._errors.append(f"evidence recording failed: {type(exc).__name__}")

        return AgentRunResult(
            run_id=self.run_id,
            goal=self.goal,
            final_status=status,
            completion_reason=reason,
            steps_executed=self.steps_executed,
            tool_calls=self.tool_calls,
            failures=self.failures,
            duration_ms=(time.monotonic() - t0) * 1000.0,
            evidence_run_id=evidence_run_id,
        )
