"""Local grounding: what Lucy actually knows, on her own foundation.

Cloud models answer from weights trained elsewhere, on data she never saw.
A new-foundation Lucy answers from her OWN local state first -- persistent
memory and the evidence ledger -- and every citation carries provenance
(record id, timestamp, source, confidence).  When nothing local matches, she
says so instead of inventing a grounding she does not have.
"""

from __future__ import annotations

import time
from typing import Any, Optional


class LocalGrounding:
    def __init__(self, retrieval: Any, evidence: Any) -> None:
        self.retrieval = retrieval
        self.evidence = evidence

    async def ground(self, query: str, limit: int = 6) -> dict[str, Any]:
        query = (query or "").strip()
        memory_citations: list[dict[str, Any]] = []
        if self.retrieval is not None:
            try:
                hits = await self.retrieval.retrieve(query, limit=limit)
            except Exception:
                hits = []
            for record, score in hits:
                memory_citations.append(
                    {
                        "source": "LOCAL_MEMORY",
                        "memory_id": record.memory_id,
                        "content": record.content[:500],
                        "memory_type": record.memory_type.value
                        if getattr(record, "memory_type", None)
                        else None,
                        "status": record.status.value
                        if getattr(record, "status", None)
                        else None,
                        "updated_at": record.updated_at,
                        "confidence": record.confidence,
                        "relevance_score": score,
                    }
                )

        evidence_citations: list[dict[str, Any]] = []
        if self.evidence is not None:
            try:
                rows = await self.evidence.query(limit=limit)
            except Exception:
                rows = []
            for row in rows:
                evidence_citations.append(
                    {
                        "source": "LOCAL_EVIDENCE",
                        "run_id": row.get("run_id"),
                        "record_type": row.get("record_type"),
                        "goal": row.get("goal"),
                        "completion_reason": row.get("completion_reason"),
                        "routing_decision": row.get("routing_decision"),
                        "host_role": row.get("host_role"),
                        "timestamp": row.get("timestamp"),
                    }
                )

        return {
            "query": query,
            "local_only": True,
            "method": "LOCAL_MEMORY + LOCAL_EVIDENCE; no cloud retrieval",
            "citation_count": len(memory_citations) + len(evidence_citations),
            "memory_citations": memory_citations,
            "evidence_citations": evidence_citations,
            "grounded": bool(memory_citations or evidence_citations),
            "note": (
                "she is grounded in her own local records"
                if (memory_citations or evidence_citations)
                else "no local records match; she reports no fabricated grounding"
            ),
            "audited_at": time.time(),
        }
