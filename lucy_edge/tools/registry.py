"""Explicit tool registry with permission enforcement on every execution."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from .permissions import (
    PermissionDecision,
    PermissionDenied,
    PermissionOutcome,
    PermissionPolicy,
)

ToolFunc = Callable[..., Awaitable[Any]]


@dataclass
class ToolSpec:
    name: str
    description: str
    func: ToolFunc
    permission_class: str = "read"
    owner: str = "lucy_edge"
    arg_names: list[str] = field(default_factory=list)

    @property
    def tool_class(self) -> str:
        return self.name.split(".")[0]


@dataclass
class ToolResult:
    ok: bool = False
    output: Any = None
    denied: bool = False
    reason: Optional[str] = None
    output_sha256: Optional[str] = None
    duration_ms: Optional[float] = None
    error: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "denied": self.denied,
            "reason": self.reason,
            "output": self.output,
            "output_sha256": self.output_sha256,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


class ToolRegistry:
    def __init__(self, policy: PermissionPolicy) -> None:
        self.policy = policy
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"tool already registered: {spec.name}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._specs.get(name)

    def has(self, name: str) -> bool:
        return name in self._specs

    def check_permission(self, name: str, args: dict[str, Any]) -> PermissionDecision:
        return self.policy.evaluate(name, args)

    def names(self) -> list[str]:
        return sorted(self._specs)

    def list(self) -> list[dict[str, str]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "permission_class": spec.permission_class,
                "owner": spec.owner,
            }
            for spec in sorted(self._specs.values(), key=lambda s: s.name)
        ]

    async def execute(
        self,
        name: str,
        args: dict[str, Any],
        context: Any,
    ) -> ToolResult:
        spec = self._specs.get(name)
        if spec is None:
            return ToolResult(
                ok=False,
                denied=True,
                reason=f"unknown tool: {name}",
            )
        t0 = time.monotonic()
        decision = self.policy.evaluate(name, args)
        if decision.outcome == PermissionOutcome.DENY:
            return ToolResult(
                ok=False,
                denied=True,
                reason=decision.reason,
                duration_ms=round((time.monotonic() - t0) * 1000.0, 3),
            )
        try:
            output = await spec.func(**args, context=context)
        except PermissionDenied as exc:
            return ToolResult(
                ok=False,
                denied=True,
                reason=exc.reason,
                duration_ms=round((time.monotonic() - t0) * 1000.0, 3),
            )
        except Exception as exc:  # sanitized, no secrets
            return ToolResult(
                ok=False,
                denied=False,
                error=f"{type(exc).__name__}: {str(exc)[:300]}",
                duration_ms=round((time.monotonic() - t0) * 1000.0, 3),
            )
        text = str(output) if output is not None else ""
        return ToolResult(
            ok=True,
            output=output,
            denied=False,
            output_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            duration_ms=round((time.monotonic() - t0) * 1000.0, 3),
        )
