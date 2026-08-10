"""Memory record schema."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROJECT = "project"
    PREFERENCE = "preference"
    EVIDENCE = "evidence"


class MemoryStatus(str, Enum):
    PROPOSED = "PROPOSED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    SUPERSEDED = "SUPERSEDED"


class ProvenanceCategory(str, Enum):
    USER_STATED = "USER_STATED"
    OBSERVED = "OBSERVED"
    RETRIEVED_MEMORY = "RETRIEVED_MEMORY"
    KNOWN_FROM_RUNTIME = "KNOWN_FROM_RUNTIME"
    INFERRED = "INFERRED"
    UNVERIFIED = "UNVERIFIED"


def compute_sha256(content: str, provenance: str, source: str, memory_type: str) -> str:
    canonical = json.dumps(
        {
            "content": content,
            "provenance": provenance,
            "source": source,
            "memory_type": memory_type,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def new_memory_id() -> str:
    return uuid.uuid4().hex


class MemoryRecord(BaseModel):
    memory_id: str = Field(default_factory=new_memory_id)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    content: str
    source: str = "unknown"
    project: Optional[str] = None
    memory_type: MemoryType = MemoryType.WORKING
    confidence: float = 0.0
    evidence_refs: list[str] = Field(default_factory=list)
    sha256: Optional[str] = None
    supersedes: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: MemoryStatus = MemoryStatus.ACCEPTED
    provenance: ProvenanceCategory = ProvenanceCategory.UNVERIFIED

    def ensure_sha256(self) -> str:
        if self.sha256 is None:
            self.sha256 = compute_sha256(
                self.content, self.provenance.value, self.source, self.memory_type.value
            )
        return self.sha256
