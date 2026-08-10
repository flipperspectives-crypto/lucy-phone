"""Evidence record schema.

Full UUID run IDs are required for new Lucy Edge work (no 8-character IDs).
Unknowns are recorded as None / UNKNOWN — never fabricated.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def new_run_id() -> str:
    return str(uuid.uuid4())


class EvidenceType(str, Enum):
    AGENT_RUN = "AGENT_RUN"
    ROUTING_DECISION = "ROUTING_DECISION"
    TOOL_CALL = "TOOL_CALL"
    MEMORY_EVENT = "MEMORY_EVENT"
    INTROSPECTION = "INTROSPECTION"
    BUILD = "BUILD"
    MCP_EVENT = "MCP_EVENT"
    MCP_CALL = "MCP_CALL"


def canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


class EvidenceRecord(BaseModel):
    run_id: str = Field(default_factory=new_run_id)
    record_type: EvidenceType = EvidenceType.AGENT_RUN
    timestamp: float = Field(default_factory=time.time)
    goal: Optional[str] = None
    run_state: Optional[str] = None
    model: Optional[str] = None
    model_digest: Optional[str] = None
    provider: Optional[str] = None
    provider_endpoint_class: Optional[str] = None
    host: Optional[str] = None
    host_role: Optional[str] = None
    routing_decision: Optional[str] = None
    routing_reason_code: Optional[str] = None
    plan: Optional[list[dict[str, Any]]] = None
    memory_retrieval_ids: list[str] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    permission_decisions: list[dict[str, Any]] = Field(default_factory=list)
    tool_output_hashes: dict[str, str] = Field(default_factory=dict)
    resource_snapshot: Optional[dict[str, Any]] = None
    thermal_snapshot: Optional[dict[str, Any]] = None
    latency_ms: Optional[float] = None
    errors: list[str] = Field(default_factory=list)
    verification: Optional[dict[str, Any]] = None
    completion_reason: Optional[str] = None
    final_status: Optional[str] = None
    sha256: Optional[str] = None

    def ensure_sha256(self) -> str:
        payload = self.model_dump(exclude={"sha256"})
        payload.pop("sha256", None)
        self.sha256 = hashlib.sha256(
            canonical_json(payload).encode("utf-8")
        ).hexdigest()
        return self.sha256

    def verify_sha256(self) -> bool:
        if self.sha256 is None:
            return False
        payload = self.model_dump(exclude={"sha256"})
        payload.pop("sha256", None)
        return hashlib.sha256(
            canonical_json(payload).encode("utf-8")
        ).hexdigest() == self.sha256
