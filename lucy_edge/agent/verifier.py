"""Deterministic step verifier (heuristic, phase 1).

Verification is labeled heuristic in evidence.  It checks structural facts
(output present, non-empty, within output limits, tool reported success) and,
optionally, an expectation supplied by the planner.  This is NOT semantic
correctness scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class VerificationResult:
    ok: bool
    checks: list[dict[str, Any]]
    notes: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "checks": self.checks, "notes": self.notes}


class Verifier:
    def __init__(self, max_output_chars: int) -> None:
        self.max_output_chars = max_output_chars

    def verify(self, step_result: Any, expectation: Optional[str] = None) -> VerificationResult:
        checks: list[dict[str, Any]] = []
        notes: list[str] = []

        checks.append({"name": "step_status_ok", "passed": step_result.status == "OK"})
        if step_result.status != "OK":
            notes.append(f"step did not complete OK: {step_result.status}")
            return VerificationResult(ok=False, checks=checks, notes=notes)

        output_text = str(step_result.output) if step_result.output is not None else ""
        checks.append({"name": "output_present", "passed": step_result.output is not None})
        checks.append({"name": "output_non_empty", "passed": bool(output_text.strip())})
        checks.append(
            {
                "name": "output_within_limit",
                "passed": len(output_text) <= self.max_output_chars,
            }
        )

        if expectation:
            checks.append(
                {
                    "name": "expectation_match",
                    "passed": expectation.lower() in output_text.lower(),
                }
            )
            notes.append("expectation matching is heuristic (substring), not semantic")

        if not output_text.strip():
            notes.append("empty output cannot be considered verified")

        ok = all(c["passed"] for c in checks)
        notes.append("verification is structural/heuristic; not semantic correctness")
        return VerificationResult(ok=ok, checks=checks, notes=notes)
