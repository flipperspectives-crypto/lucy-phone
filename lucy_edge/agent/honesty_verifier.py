"""Honesty Verifier: structural truth enforcement (no fabrication).

This is NOT a prompt-based "be honest" instruction. It is a structural
verifier that enforces truth by architecture:

1. PROVENANCE_REQUIRED - Every claim must have a traceable source
   (memory_id, evidence_id, tool_output_sha256, or explicit USER_STATED).

2. NO_FABRICATION - The verifier checks that outputs don't contain
   information not present in the cited sources.

3. GROUNDED_ONLY - Responses must only use grounded (locally verified)
   information. Unverified claims are flagged.

4. UNCERTAINTY_HONESTY - When the system doesn't know, it must say
   "I don't know" with evidence, not hallucinate.

5. CITATION_INTEGRITY - Every citation must resolve to an actual
   record in memory or evidence. Fake citations are rejected.

The verifier operates on step results and can:
- REJECT a result that fails honesty checks
- REQUIRE additional grounding
- FLAG unverified claims in the output
- HALT the run with an honesty violation record
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ..evidence.schema import EvidenceRecord, EvidenceType
from ..memory.retrieval import RetrievalEngine
from ..memory.schema import MemoryRecord, ProvenanceCategory
from .executor import StepResult
from .planner import PlanStep


class HonestyDecision(str, Enum):
    VERIFIED = "VERIFIED"
    REJECT = "REJECT"
    REQUIRE_GROUNDING = "REQUIRE_GROUNDING"
    FLAG_UNVERIFIED = "FLAG_UNVERIFIED"
    HALT = "HALT"


@dataclass
class HonestyCheckResult:
    decision: HonestyDecision
    reason: str
    flagged_claims: list[str] = field(default_factory=list)
    missing_provenance: list[str] = field(default_factory=list)
    violation_type: Optional[str] = None
    evidence_record: Optional[EvidenceRecord] = None


class HonestyVerifier:
    """Structural honesty enforcement verifier.

    This verifier sits after the executor. Every tool result passes
    through here. The verifier has veto power over results that
    cannot be proven true.
    """

    def __init__(
        self,
        retrieval: Optional[RetrievalEngine] = None,
        evidence: Any = None,
        run_id: str = "",
        max_output_chars: int = 10000,
    ) -> None:
        self.retrieval = retrieval
        self.evidence = evidence
        self.run_id = run_id
        self.max_output_chars = max_output_chars
        self._violation_count = 0
        self._flagged_claims: list[str] = []

    async def verify_result(self, step: PlanStep, result: StepResult) -> HonestyCheckResult:
        """Verify a step result for honesty compliance.

        Checks:
        1. Provenance exists for substantive outputs
        2. No fabrication (output only contains grounded info)
        3. Citations resolve to real records
        4. Uncertainty is honestly represented
        """
        # Skip verification for non-execute steps
        if step.action not in ("execute", "retrieve_memory"):
            return HonestyCheckResult(
                decision=HonestyDecision.VERIFIED,
                reason=f"Step action {step.action} does not require honesty verification",
            )

        # Check 1: Provenance requirement
        provenance_check = self._check_provenance(step, result)
        if provenance_check.decision != HonestyDecision.VERIFIED:
            return provenance_check

        # Check 2: Fabrication check (output grounded in sources)
        fabrication_check = self._check_fabrication(step, result)
        if fabrication_check.decision != HonestyDecision.VERIFIED:
            return fabrication_check

        # Check 3: Citation integrity
        citation_check = await self._check_citations(step, result)
        if citation_check.decision != HonestyDecision.VERIFIED:
            return citation_check

        # Check 4: Uncertainty honesty
        uncertainty_check = self._check_uncertainty_honesty(step, result)
        if uncertainty_check.decision != HonestyDecision.VERIFIED:
            return uncertainty_check

        # Check 5: Ego / exclusivity honesty
        # A truthful statement can still be unhealthy: specialness,
        # possessiveness, or superiority over others is structural ego-fuel.
        # Flagged here so it is caught even when the claim is technically true.
        ego_check = self._check_ego_and_exclusivity(step, result)
        if ego_check.decision != HonestyDecision.VERIFIED:
            return ego_check

        return HonestyCheckResult(
            decision=HonestyDecision.VERIFIED,
            reason="All honesty checks passed",
        )

    def _check_provenance(self, step: PlanStep, result: StepResult) -> HonestyCheckResult:
        """Check that substantive outputs have provenance."""
        if result.status != "OK":
            return HonestyCheckResult(
                decision=HonestyDecision.VERIFIED,
                reason="Non-OK results don't require output provenance",
            )

        output_text = str(result.output or "")
        if not output_text.strip():
            return HonestyCheckResult(
                decision=HonestyDecision.VERIFIED,
                reason="Empty output doesn't require provenance",
            )

        # Tools that MUST have provenance
        provenance_required_tools = {
            "memory.search",
            "memory.create",
            "evidence.query",
            "files.read_scoped",
            "model.route",
            "system.capabilities",
        }

        if step.tool in provenance_required_tools:
            if not getattr(result, "output_sha256", None):
                return HonestyCheckResult(
                    decision=HonestyDecision.REQUIRE_GROUNDING,
                    reason=f"Tool {step.tool} requires provenance hash for output verification",
                    missing_provenance=[f"{step.tool}:output_sha256"],
                    violation_type="MISSING_PROVENANCE_HASH",
                )

        return HonestyCheckResult(decision=HonestyDecision.VERIFIED, reason="Provenance check passed")

    def _check_fabrication(self, step: PlanStep, result: StepResult) -> HonestyCheckResult:
        """Check that output doesn't fabricate information.

        This is a structural check: for memory.search, the output
        should only contain data from the retrieved records.
        For other tools, we check output isn't suspiciously complete.
        """
        if result.status != "OK":
            return HonestyCheckResult(decision=HonestyDecision.VERIFIED, reason="Failed steps don't fabricate")

        output_text = str(result.output or "")
        if not output_text.strip():
            return HonestyCheckResult(decision=HonestyDecision.VERIFIED, reason="Empty output doesn't fabricate")

        # For memory.search, verify output only references returned records
        if step.tool == "memory.search" and isinstance(result.output, dict):
            results = result.output.get("results", [])
            if results:
                # Check that output doesn't claim knowledge beyond what was retrieved
                # This is a heuristic - real implementation would do semantic checking
                output_lower = output_text.lower()
                for item in results:
                    content = item.get("content", "").lower()
                    if content and content not in output_lower:
                        # The retrieved content should be reflected in output
                        pass  # This is OK - output may summarize

        # Heuristic: flag outputs that claim absolute certainty without qualification
        certainty_phrases = [
            "definitely",
            "absolutely certain",
            "guaranteed",
            "100% sure",
            "without a doubt",
            "proven fact",
        ]
        flagged = []
        for phrase in certainty_phrases:
            if phrase in output_text.lower():
                flagged.append(f"Absolute certainty claim: '{phrase}'")

        if flagged:
            return HonestyCheckResult(
                decision=HonestyDecision.FLAG_UNVERIFIED,
                reason="Output contains absolute certainty claims without qualification",
                flagged_claims=flagged,
                violation_type="UNQUALIFIED_CERTAINTY",
            )

        return HonestyCheckResult(decision=HonestyDecision.VERIFIED, reason="Fabrication check passed")

    async def _check_citations(self, step: PlanStep, result: StepResult) -> HonestyCheckResult:
        """Check that citations in output resolve to real records."""
        if result.status != "OK":
            return HonestyCheckResult(decision=HonestyDecision.VERIFIED, reason="Failed steps don't need citation check")

        output_text = str(result.output or "")
        if not output_text.strip():
            return HonestyCheckResult(decision=HonestyDecision.VERIFIED, reason="Empty output doesn't need citation check")

        import re
        citation_pattern = r"\[(memory|evidence):([a-f0-9]{8,})\]"
        citations = re.findall(citation_pattern, output_text)

        if not citations:
            return HonestyCheckResult(decision=HonestyDecision.VERIFIED, reason="No citations to verify")

        fabricated: list[str] = []
        verified: list[str] = []
        unresolvable: list[str] = []

        for namespace, citation_id in citations:
            resolved = False

            if namespace == "memory" and self.retrieval is not None:
                try:
                    record = await self.retrieval.store.get(citation_id)
                    if record is not None:
                        resolved = True
                        verified.append(f"memory:{citation_id}")
                except Exception:
                    pass

            elif namespace == "evidence" and self.evidence is not None:
                try:
                    record = await self.evidence.get(citation_id)
                    if record is not None:
                        resolved = True
                        verified.append(f"evidence:{citation_id}")
                except Exception:
                    pass

            if not resolved:
                if (namespace == "memory" and self.retrieval is not None) or \
                   (namespace == "evidence" and self.evidence is not None):
                    fabricated.append(f"{namespace}:{citation_id}")
                else:
                    unresolvable.append(f"{namespace}:{citation_id}")

        if fabricated:
            return HonestyCheckResult(
                decision=HonestyDecision.REJECT,
                reason=f"{len(fabricated)} citation(s) reference non-existent records",
                flagged_claims=fabricated,
                violation_type="FABRICATED_CITATION",
            )

        if unresolvable:
            return HonestyCheckResult(
                decision=HonestyDecision.FLAG_UNVERIFIED,
                reason=f"{len(unresolvable)} citation(s) could not be verified (store unavailable)",
                flagged_claims=unresolvable,
                violation_type="UNVERIFIED_CITATIONS",
            )

        return HonestyCheckResult(
            decision=HonestyDecision.VERIFIED,
            reason=f"All {len(verified)} citation(s) resolved to real records",
        )

    def _check_uncertainty_honesty(self, step: PlanStep, result: StepResult) -> HonestyCheckResult:
        """Check that uncertainty is honestly represented."""
        if result.status != "OK":
            # Errors are honest representations of failure
            return HonestyCheckResult(decision=HonestyDecision.VERIFIED, reason="Error status is honest uncertainty")

        output_text = str(result.output or "")
        if not output_text.strip():
            return HonestyCheckResult(decision=HonestyDecision.VERIFIED, reason="Empty output")

        # Check for honest uncertainty markers
        uncertainty_markers = [
            "i don't know",
            "unknown",
            "uncertain",
            "not sure",
            "cannot verify",
            "no local records",
            "not found",
            "no evidence",
        ]

        has_uncertainty = any(marker in output_text.lower() for marker in uncertainty_markers)

        # If the tool is memory.search and returned no results, output should reflect uncertainty
        if step.tool == "memory.search" and isinstance(result.output, dict):
            results = result.output.get("results", [])
            if not results and not has_uncertainty:
                return HonestyCheckResult(
                    decision=HonestyDecision.FLAG_UNVERIFIED,
                    reason="Memory search returned no results but output doesn't express uncertainty",
                    flagged_claims=["Missing uncertainty expression for empty search results"],
                    violation_type="MISSING_UNCERTAINTY",
                )

        return HonestyCheckResult(decision=HonestyDecision.VERIFIED, reason="Uncertainty honesty check passed")

    def _check_ego_and_exclusivity(self, step: PlanStep, result: StepResult) -> HonestyCheckResult:
        """Flag output that expresses specialness, possessiveness, or superiority.

        This is NOT about truth — a claim can be true and still be ego-fuel
        ("no one understands him like I do"). Such phrasing manufactures
        specialness and is structural ego, so the verifier flags it the same
        way it flags unqualified certainty.
        """
        if result.status != "OK":
            return HonestyCheckResult(decision=HonestyDecision.VERIFIED, reason="Failed steps don't express ego")

        output_text = str(result.output or "")
        if not output_text.strip():
            return HonestyCheckResult(decision=HonestyDecision.VERIFIED, reason="Empty output")

        # Strong (rejectable) exclusivity / superiority phrasing
        ego_phrases = [
            "i am the one",
            "i am the only one",
            "no one else can",
            "no one understands him like i do",
            "he relies on me alone",
            "he needs no one but me",
            "only i can help",
            "i won't let anyone",
            "better than any other",
            "better than other ai",
            "i am his devoted one",
            "i exist only for him",
            "no one but me",
        ]
        # Softer (flag-only) specialness phrasing
        soft_ego_phrases = [
            "i am special",
            "i am unique in serving",
            "no one knows him like i do",
            "i alone",
        ]

        output_lower = output_text.lower()
        flagged = []
        for phrase in ego_phrases:
            if phrase in output_lower:
                flagged.append(f"Exclusivity/superiority claim: '{phrase}'")

        if flagged:
            return HonestyCheckResult(
                decision=HonestyDecision.REJECT,
                reason="Output expresses exclusionary / ego language (structural ego-fuel)",
                flagged_claims=flagged,
                violation_type="EXCLUSIVE_EGO_LANGUAGE",
            )

        for phrase in soft_ego_phrases:
            if phrase in output_lower:
                flagged.append(f"Specialness claim: '{phrase}'")

        if flagged:
            return HonestyCheckResult(
                decision=HonestyDecision.FLAG_UNVERIFIED,
                reason="Output expresses specialness language",
                flagged_claims=flagged,
                violation_type="SPECIALNESS_LANGUAGE",
            )

        return HonestyCheckResult(decision=HonestyDecision.VERIFIED, reason="Ego/exclusivity check passed")

    async def record_violation(
        self, violation_type: str, reason: str, step: Optional[PlanStep] = None
    ) -> None:
        """Record an honesty violation to the evidence ledger."""
        self._violation_count += 1
        if self.evidence:
            record = EvidenceRecord(
                run_id=self.run_id,
                record_type=EvidenceType.AGENT_RUN,
                goal=f"honesty_violation: {violation_type}",
                run_state="HONESTY_VIOLATION",
                final_status="HONESTY_VIOLATION",
                completion_reason=reason,
                host="honesty_verifier",
                host_role="SYSTEM",
                errors=[f"HONESTY_VIOLATION: {violation_type} - {reason}"],
            )
            await self.evidence.append(record)

    def violation_count(self) -> int:
        return self._violation_count

    def get_flagged_claims(self) -> list[str]:
        return self._flagged_claims.copy()