"""Bounded agent limits."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class AgentLimits(BaseModel):
    max_steps: int = Field(default=8, ge=1, le=50)
    max_tool_calls: int = Field(default=12, ge=1, le=200)
    max_failures: int = Field(default=3, ge=0, le=50)
    task_timeout: float = Field(default=30.0, gt=0.0)
    tool_timeout: float = Field(default=10.0, gt=0.0)
    max_output_chars: int = Field(default=10_000, ge=256)

    @model_validator(mode="after")
    def _check(self) -> "AgentLimits":
        if self.tool_timeout >= self.task_timeout:
            raise ValueError("tool_timeout must be shorter than task_timeout")
        return self


def limits_from_config(config: "Any") -> AgentLimits:
    return AgentLimits(
        max_steps=config.agent.max_steps,
        max_tool_calls=config.agent.max_tool_calls,
        max_failures=config.agent.max_failures,
        task_timeout=config.agent.task_timeout,
        tool_timeout=config.agent.tool_timeout,
        max_output_chars=config.agent.max_output_chars,
    )
