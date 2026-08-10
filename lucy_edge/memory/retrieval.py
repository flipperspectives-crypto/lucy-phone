"""Memory retrieval: ranking search hits for agent use.

Deterministic, heuristic ranking: FTS/term relevance then recency then
confidence.  Never invents results that do not exist.
"""

from __future__ import annotations

from typing import Optional

from .schema import MemoryRecord, MemoryStatus, MemoryType
from .store import MemoryStore


class RetrievalEngine:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    async def retrieve(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 8,
        min_confidence: float = 0.0,
        include_proposed: bool = False,
    ) -> list[tuple[MemoryRecord, float]]:
        statuses = None
        if not include_proposed:
            statuses = [MemoryStatus.ACCEPTED, MemoryStatus.SUPERSEDED]
        records = await self.store.search(
            query, memory_type=memory_type, limit=limit * 3, statuses=statuses
        )
        scored: list[tuple[MemoryRecord, float]] = []
        for record in records:
            if record.confidence < min_confidence:
                continue
            recency = _recency_score(record.updated_at)
            relevance = 1.0 if query.lower() in record.content.lower() else 0.5
            confidence = max(0.0, min(record.confidence, 1.0))
            score = round(0.5 * relevance + 0.3 * recency + 0.2 * confidence, 4)
            scored.append((record, score))
        scored.sort(key=lambda item: (-item[1], -item[0].updated_at))
        return scored[:limit]


def _recency_score(updated_at: float, now: Optional[float] = None) -> float:
    import time

    now = now or time.time()
    age = max(0.0, now - updated_at)
    # exponential decay: ~1.0 fresh, approaches 0 after ~30 days
    return round(__import__("math").exp(-age / (30 * 86400.0)), 4)
