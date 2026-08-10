"""Gateway request/response schemas."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from ..agent.limits import AgentLimits
from ..routing.hosts import HostState
from ..routing.policy import RoutingResult


class ChatRequest(BaseModel):
    model: str = "qwen3:1.7b"
    provider: str = "mock"
    messages: list[dict[str, Any]] = Field(default_factory=list)
    max_tokens: Optional[int] = None
    target_host: Optional[str] = None


class ChatResponse(BaseModel):
    ok: bool
    routing: RoutingResult
    provider: Optional[str] = None
    model: Optional[str] = None
    message: Optional[str] = None
    simulated: Optional[bool] = None
    error: Optional[str] = None


class AgentTaskSubmit(BaseModel):
    goal: str
    limits: Optional[AgentLimits] = None
    approval_timeout: float = 60.0


class AgentTaskStatus(BaseModel):
    run_id: str
    goal: str
    final_status: str
    completion_reason: str
    steps_executed: int
    tool_calls: int
    failures: int
    evidence_run_id: str


class EvidenceQuery(BaseModel):
    record_type: Optional[str] = None
    host_role: Optional[str] = None
    limit: int = 50


class HostRegister(BaseModel):
    host: HostState
    provider_version: Optional[str] = None


class IntrospectResponse(BaseModel):
    service: str
    report: dict[str, Any]
