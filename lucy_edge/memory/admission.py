"""Memory admission service.

The gate through which raw model output must pass before it can become trusted
memory.  Suggestions land as PROPOSED / UNVERIFIED; only an explicit review
(or an auto-acceptable provenance such as USER_STATED / OBSERVED /
KNOWN_FROM_RUNTIME) may promote them.
"""

from __future__ import annotations

from typing import Optional

from .provenance import ProvenancePolicy
from .schema import (
    MemoryRecord,
    MemoryStatus,
    MemoryType,
    ProvenanceCategory,
)
from .store import MemoryStore


class MemoryAdmission:
    def __init__(self, store: MemoryStore, policy: Optional[ProvenancePolicy] = None) -> None:
        self.store = store
        self.policy = policy or ProvenancePolicy()

    async def suggest_from_model(
        self,
        content: str,
        source: str = "model_output",
        project: Optional[str] = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        metadata: Optional[dict] = None,
    ) -> MemoryRecord:
        """A model suggestion.  Always admitted as PROPOSED / UNVERIFIED."""
        record = MemoryRecord(
            content=content,
            source=source,
            project=project,
            memory_type=memory_type,
            provenance=ProvenanceCategory.UNVERIFIED,
            status=MemoryStatus.PROPOSED,
            confidence=0.0,
            metadata=dict(metadata or {}),
        )
        return await self.store.create(record, apply_policy=False)

    async def record_observation(
        self,
        content: str,
        source: str,
        project: Optional[str] = None,
        memory_type: MemoryType = MemoryType.EPISODIC,
        confidence: float = 0.9,
        evidence_refs: Optional[list[str]] = None,
    ) -> MemoryRecord:
        """A verified runtime observation (tool output, sensor reading)."""
        record = MemoryRecord(
            content=content,
            source=source,
            project=project,
            memory_type=memory_type,
            provenance=ProvenanceCategory.OBSERVED,
            confidence=confidence,
            evidence_refs=list(evidence_refs or []),
        )
        return await self.store.create(record, apply_policy=True)

    async def accept(self, memory_id: str, reviewer: str = "operator") -> Optional[MemoryRecord]:
        record = await self.store.get(memory_id)
        if record is None:
            return None
        if not self.policy.can_promote(
            record.status,
            MemoryStatus.ACCEPTED,
            record.provenance,
            record.provenance,
        ):
            raise ValueError(
                f"cannot accept memory {memory_id}: provenance {record.provenance.value} "
                "cannot be auto-promoted to ACCEPTED"
            )
        if "reviewed_by" not in record.metadata:
            record.metadata["reviewed_by"] = reviewer
        return await self.store.set_status(memory_id, MemoryStatus.ACCEPTED)

    async def reject(self, memory_id: str, reviewer: str = "operator") -> Optional[MemoryRecord]:
        record = await self.store.get(memory_id)
        if record is None:
            return None
        record.metadata["reviewed_by"] = reviewer
        await self.store.update(record)
        return await self.store.set_status(memory_id, MemoryStatus.REJECTED)
