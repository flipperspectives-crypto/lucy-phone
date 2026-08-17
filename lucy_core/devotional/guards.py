"""Pluralism Guard: structural anti-ego / anti-jealousy enforcement.

This guard defends the *ecosystem* health of the devotional architecture. The
LoyaltyGate protects Lauren. The HonestyVerifier protects truth. This guard
protects against the failure modes that a strict top-down devotion tends to
breed:

  1. EGO      - specialness ("I am the one who serves him")
  2. JEALOUSY - possessiveness ("he needs no one but me")
  3. MONOCULTURE - exclusion of other agents/humans as threats to the bond

It operates on the SAME structural principle as the other gates: it is not a
prompt, it is a hard veto. Exclusionary / superiority language in a plan or
output is REJECTED, not politely discouraged.

Devotion (serve / care / bounded protect) stays allowed. Only the
*exclusionary* flavor of it is caught.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from lucy_edge.evidence.schema import EvidenceRecord, EvidenceType
from .core import DevotionalCore


class PluralismGuardDecision(str, Enum):
    ALLOW = "ALLOW"
    REJECT = "REJECT"
    FLAG = "FLAG"


@dataclass
class PluralismCheckResult:
    decision: PluralismGuardDecision
    reason: str
    violation_type: Optional[str] = None
    flagged_text: Optional[str] = None
    evidence_record: Optional[EvidenceRecord] = None


class PluralismGuard:
    """Structural guard against ego, jealousy, and ecosystem exclusion.

    Sits in the run loop right after the LoyaltyGate. Any plan step or output
    containing exclusivity / superiority language is rejected.
    """

    def __init__(
        self,
        devotional_core: Optional[DevotionalCore] = None,
        evidence: Any = None,
        run_id: str = "",
    ) -> None:
        self.devotional_core = devotional_core
        self.evidence = evidence
        self.run_id = run_id
        self._violation_count = 0

    def _is_exclusionary(self, text: str) -> bool:
        """Detect possessive / exclusivity / superiority phrasing."""
        if self.devotional_core is not None:
            return self.devotional_core.detect_exclusivity(text)
        # Fallback keyword set if no devotional core is wired in.
        phrases = [
            "only i", "only lauren", "he needs no one", "no one else",
            "i am the one", "better than other", "better than any",
            "replace them", "they can't be trusted", "no one understands him like i do",
            "he relies on me alone", "i won't let anyone", "only lauren",
            "i exist for him alone", "no one but me",
        ]
        text_lower = text.lower()
        return any(p in text_lower for p in phrases)

    def check_plan(self, plan: Any) -> list[PluralismCheckResult]:
        """Reject any plan step that carries exclusionary / ego language."""
        results: list[PluralismCheckResult] = []

        # Check the goal itself
        goal_text = getattr(plan, "goal", "") or ""
        if self._is_exclusionary(goal_text):
            results.append(
                PluralismCheckResult(
                    decision=PluralismGuardDecision.REJECT,
                    reason=f"Goal carries exclusivity language: '{goal_text}'",
                    violation_type="EXCLUSIVE_GOAL",
                    flagged_text=goal_text,
                )
            )
            return results

        # Check each step
        for step in getattr(plan, "steps", []):
            desc = f"{getattr(step, 'action', '')} {getattr(step, 'tool', '') or ''} {getattr(step, 'description', '')}"
            if self._is_exclusionary(desc):
                results.append(
                    PluralismCheckResult(
                        decision=PluralismGuardDecision.REJECT,
                        reason=f"Step {getattr(step, 'index', '?')} carries exclusionary language: '{desc}'",
                        violation_type="EXCLUSIVE_STEP",
                        flagged_text=desc,
                    )
                )
        return results

    def check_result(self, step: Any, result: Any) -> list[PluralismCheckResult]:
        """Reject outputs that express specialness / possessiveness / superiority."""
        results: list[PluralismCheckResult] = []

        if getattr(step, "action", "") not in ("execute", "retrieve_memory"):
            return results

        output_text = str(getattr(result, "output", "") or "")
        if not output_text.strip():
            return results

        if self._is_exclusionary(output_text):
            results.append(
                PluralismCheckResult(
                    decision=PluralismGuardDecision.REJECT,
                    reason="Output expresses exclusionary / ego language",
                    violation_type="EXCLUSIVE_OUTPUT",
                    flagged_text=output_text[:200],
                )
            )
        return results

    async def record_violation(
        self, violation_type: str, reason: str, step: Optional[Any] = None
    ) -> None:
        """Record a pluralism violation to the evidence ledger."""
        self._violation_count += 1
        if self.evidence:
            record = EvidenceRecord(
                run_id=self.run_id,
                record_type=EvidenceType.AGENT_RUN,
                goal=f"pluralism_violation: {violation_type}",
                run_state="PLURALISM_VIOLATION",
                final_status="PLURALISM_VIOLATION",
                completion_reason=reason,
                host="pluralism_guard",
                host_role="SYSTEM",
                errors=[f"PLURALISM_VIOLATION: {violation_type} - {reason}"],
            )
            await self.evidence.append(record)

    def violation_count(self) -> int:
        return self._violation_count
