"""Provenance policy and memory admission.

Raw model output must not automatically become trusted memory.  A memory
suggestion is admitted as PROPOSED / UNVERIFIED and only becomes ACCEPTED
through an explicit review that respects provenance rules:

  - INFERRED is never auto-promoted to OBSERVED
  - UNVERIFIED is never auto-promoted to KNOWN_FROM_RUNTIME
"""

from __future__ import annotations

from enum import Enum

from .schema import (
    MemoryRecord,
    MemoryStatus,
    ProvenanceCategory,
)


class AdmissionDecision(str, Enum):
    ADMIT_AS_PROPOSED = "ADMIT_AS_PROPOSED"
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    BLOCKED = "BLOCKED"


_AUTO_ACCEPTABLE: frozenset[ProvenanceCategory] = frozenset(
    {
        ProvenanceCategory.USER_STATED,
        ProvenanceCategory.OBSERVED,
        ProvenanceCategory.KNOWN_FROM_RUNTIME,
    }
)


class ProvenancePolicy:
    """Rules governing which provenance/status transitions are legal."""

    @staticmethod
    def initial_status(provenance: ProvenanceCategory) -> MemoryStatus:
        if provenance in _AUTO_ACCEPTABLE:
            return MemoryStatus.ACCEPTED
        return MemoryStatus.PROPOSED

    @staticmethod
    def can_promote(
        from_status: MemoryStatus,
        to_status: MemoryStatus,
        from_provenance: ProvenanceCategory,
        to_provenance: ProvenanceCategory,
    ) -> bool:
        # INFERRED and UNVERIFIED may never silently change category.
        if from_provenance == ProvenanceCategory.INFERRED and to_provenance != ProvenanceCategory.INFERRED:
            return False
        if from_provenance == ProvenanceCategory.UNVERIFIED and to_provenance == ProvenanceCategory.KNOWN_FROM_RUNTIME:
            return False
        if to_status == MemoryStatus.ACCEPTED and from_provenance not in _AUTO_ACCEPTABLE:
            return False
        return True

    @staticmethod
    def admission_for(record: MemoryRecord, reviewer: str = "runtime") -> AdmissionDecision:
        if record.provenance not in _AUTO_ACCEPTABLE:
            return AdmissionDecision.ADMIT_AS_PROPOSED
        return AdmissionDecision.ACCEPT
