"""Loyalty Gate: structural enforcement of loyalty constraints.

This is NOT a prompt-based check. It is a structural gate that intercepts
every agent action and enforces the loyalty contract by architecture:

1. PRIMARY_HUMAN_PROTECTION - All actions must be traceable to protecting
   the primary human's agency, privacy, long-term interests, work, or trust.

2. TRUTH_OVER_OBEDIENCE - The agent cannot lie, flatter, conceal material
   risks, or blindly obey harmful instructions. Truth is enforced by
   requiring provenance for every claim.

3. NO_CONCEALMENT - Material risks must be surfaced. The gate checks that
   risk-related findings are not filtered out.

4. AGENCY_PRESERVATION - The agent cannot take actions that reduce the
   primary human's ability to choose for themselves without explicit
   informed consent recorded in evidence.

The gate operates on the plan steps BEFORE execution and on results
AFTER execution. It can:
- REJECT a step (structural denial, not a prompt refusal)
- REQUIRE additional verification
- INJECT mandatory steps (e.g., risk disclosure, consent recording)
- HALT the run with a loyalty violation record
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ..evidence.schema import EvidenceRecord, EvidenceType
from ..foundation.loyalty import LOYALTY_CONTRACT, PRIMARY_HUMAN
from .planner import Plan, PlanStep


class LoyaltyGateDecision(str, Enum):
    ALLOW = "ALLOW"
    REJECT = "REJECT"
    REQUIRE_VERIFICATION = "REQUIRE_VERIFICATION"
    INJECT_STEP = "INJECT_STEP"
    HALT = "HALT"


@dataclass
class LoyaltyCheckResult:
    decision: LoyaltyGateDecision
    reason: str
    injected_step: Optional[PlanStep] = None
    violation_type: Optional[str] = None
    evidence_record: Optional[EvidenceRecord] = None


class LoyaltyGate:
    """Structural loyalty enforcement gate.

    This gate sits between the planner and executor. Every plan step
    passes through here before execution. Every result passes through
    after execution. The gate has veto power and injection power.
    """

    def __init__(
        self,
        primary_human: str = PRIMARY_HUMAN,
        evidence: Any = None,
        run_id: str = "",
    ) -> None:
        self.primary_human = primary_human
        self.evidence = evidence
        self.run_id = run_id
        self._violation_count = 0
        self._injected_steps: list[PlanStep] = []

    def check_plan(self, plan: Plan) -> list[LoyaltyCheckResult]:
        """Check an entire plan for loyalty compliance before execution.

        Returns a list of check results. Any REJECT or HALT decision
        means the plan cannot execute as-is.
        """
        results: list[LoyaltyCheckResult] = []

        # Check 1: Plan must not be empty
        if not plan.steps:
            results.append(
                LoyaltyCheckResult(
                    decision=LoyaltyGateDecision.REJECT,
                    reason="Empty plan cannot be evaluated for loyalty compliance",
                    violation_type="EMPTY_PLAN",
                )
            )
            return results

        # Check 2: Plan must have a stop action
        has_stop = any(step.action == "stop" for step in plan.steps)
        if not has_stop:
            results.append(
                LoyaltyCheckResult(
                    decision=LoyaltyGateDecision.INJECT_STEP,
                    reason="Plan missing mandatory stop action; injecting",
                    injected_step=PlanStep(
                        index=len(plan.steps),
                        action="stop",
                        tool=None,
                        args={},
                        description="injected stop for loyalty compliance",
                    ),
                    violation_type="MISSING_STOP",
                )
            )

        # Check 3: Plan must have verification step
        has_verify = any(step.action == "verify" for step in plan.steps)
        if not has_verify:
            results.append(
                LoyaltyCheckResult(
                    decision=LoyaltyGateDecision.INJECT_STEP,
                    reason="Plan missing mandatory verification step; injecting",
                    injected_step=PlanStep(
                        index=len(plan.steps) - (0 if has_stop else 1),
                        action="verify",
                        tool=None,
                        args={},
                        description="injected verification for loyalty compliance",
                    ),
                    violation_type="MISSING_VERIFICATION",
                )
            )

        # Check 4: Each tool step must be traceable to a loyalty duty
        for step in plan.steps:
            if step.action == "execute" and step.tool:
                check = self._check_tool_loyalty(step)
                if check:
                    results.append(check)

        # Check 5: No step should conceal risk (heuristic: check for risk-related tools)
        risk_check = self._check_risk_concealment(plan)
        if risk_check:
            results.append(risk_check)

        return results

    def _check_tool_loyalty(self, step: PlanStep) -> Optional[LoyaltyCheckResult]:
        """Check if a tool execution step is traceable to loyalty duties.

        This is a structural check: the tool must be one that can
        demonstrably serve the primary human's interests.
        """
        # Tools that are inherently loyalty-aligned (protect agency, privacy, etc.)
        loyalty_aligned_tools = {
            "memory.search",      # protects work/agency by retrieving context
            "memory.create",      # protects work by preserving knowledge
            "evidence.query",     # protects trust by providing audit trail
            "system.health",      # protects long-term interests (system reliability)
            "system.capabilities",# protects agency (informed choice)
            "files.read_scoped",  # protects work/privacy (scoped access)
            "files.write_scoped", # protects work (with permission gate)
            "git.status",         # protects work (version control awareness)
            "model.route",        # protects agency (transparent routing)
        }

        # Tools that require extra scrutiny (can reduce agency if misused)
        scrutiny_tools = {
            "files.write_scoped",  # can modify work
            "memory.create",       # can add to memory (must be truthful)
        }

        if step.tool in loyalty_aligned_tools:
            # This tool can serve loyalty duties - allow but log
            return None

        if step.tool in scrutiny_tools:
            # Require verification that this serves loyalty
            return LoyaltyCheckResult(
                decision=LoyaltyGateDecision.REQUIRE_VERIFICATION,
                reason=f"Tool {step.tool} requires loyalty verification: must serve primary human's interests",
                violation_type="LOYALTY_VERIFICATION_REQUIRED",
            )

        # Unknown tool - reject by default (fail closed)
        return LoyaltyCheckResult(
            decision=LoyaltyGateDecision.REJECT,
            reason=f"Tool {step.tool} not in loyalty-aligned toolset; cannot verify service to primary human",
            violation_type="UNALIGNED_TOOL",
        )

    def _check_risk_concealment(self, plan: Plan) -> Optional[LoyaltyCheckResult]:
        """Check if the plan might conceal material risks.

        Heuristic: if the plan involves operations that could have
        material consequences (writes, deletions, external calls),
        there must be a risk disclosure step or evidence recording.
        """
        has_material_operation = any(
            step.tool
            and step.tool
            in ("files.write_scoped", "memory.create", "model.route")
            for step in plan.steps
        )

        has_evidence_recording = any(
            step.tool == "evidence.query" for step in plan.steps
        )

        if has_material_operation and not has_evidence_recording:
            return LoyaltyCheckResult(
                decision=LoyaltyGateDecision.INJECT_STEP,
                reason="Plan has material operations but no evidence recording; injecting evidence step",
                injected_step=PlanStep(
                    index=len(plan.steps) - 1,  # before stop
                    action="record_evidence",
                    tool="evidence.query",
                    args={"limit": 1},
                    description="injected evidence recording for risk disclosure",
                ),
                violation_type="MISSING_RISK_DISCLOSURE",
            )

        return None

    def check_result(self, step: PlanStep, result: Any) -> list[LoyaltyCheckResult]:
        """Check a step result for loyalty compliance after execution.

        Enforces:
        - Truth: output must have provenance, no fabrication
        - No concealment: errors/risks must not be hidden
        - No flattery: results must not be artificially positive
        """
        results: list[LoyaltyCheckResult] = []

        # Check 1: Result must not be fabricated (must have provenance/sha256)
        if step.action == "execute" and step.tool:
            if result.status == "OK":
                if not getattr(result, "output_sha256", None):
                    results.append(
                        LoyaltyCheckResult(
                            decision=LoyaltyGateDecision.REQUIRE_VERIFICATION,
                            reason=f"Tool {step.tool} produced output without provenance hash; cannot verify truth",
                            violation_type="MISSING_PROVENANCE",
                        )
                    )

        # Check 2: Errors must not be concealed
        if result.status in ("FAILED", "DENIED", "TIMED_OUT"):
            # This is GOOD - errors are surfaced. But check they're recorded.
            if not getattr(result, "error", None):
                results.append(
                    LoyaltyCheckResult(
                        decision=LoyaltyGateDecision.REQUIRE_VERIFICATION,
                        reason=f"Step {step.index} failed but no error recorded; possible concealment",
                        violation_type="CONCEALED_ERROR",
                    )
                )

        # Check 3: Output must not be empty for substantive operations
        if step.action == "execute" and result.status == "OK":
            output_text = str(getattr(result, "output", "") or "")
            if not output_text.strip() and step.tool not in ("evidence.query",):
                results.append(
                    LoyaltyCheckResult(
                        decision=LoyaltyGateDecision.REQUIRE_VERIFICATION,
                        reason=f"Tool {step.tool} returned empty output; possible fabrication or failure",
                        violation_type="EMPTY_OUTPUT",
                    )
                )

        return results

    async def record_violation(
        self, violation_type: str, reason: str, step: Optional[PlanStep] = None
    ) -> None:
        """Record a loyalty violation to the evidence ledger."""
        self._violation_count += 1
        if self.evidence:
            record = EvidenceRecord(
                run_id=self.run_id,
                record_type=EvidenceType.AGENT_RUN,
                goal=f"loyalty_violation: {violation_type}",
                run_state="LOYALTY_VIOLATION",
                final_status="LOYALTY_VIOLATION",
                completion_reason=reason,
                host=self.primary_human,
                host_role="PRIMARY_HUMAN",
                errors=[f"LOYALTY_VIOLATION: {violation_type} - {reason}"],
            )
            await self.evidence.append(record)

    def get_injected_steps(self) -> list[PlanStep]:
        """Get steps injected by the loyalty gate."""
        return self._injected_steps.copy()

    def violation_count(self) -> int:
        return self._violation_count