"""Tool permission policy: ALLOW / ASK / DENY.

Default principles:
  - read-only safe scoped operations  -> potentially ALLOW
  - writes inside approved workspace   -> ASK or policy-controlled
  - deletion                           -> ASK
  - arbitrary shell                    -> DENY by default
  - filesystem outside approved roots  -> DENY
  - credential access                  -> DENY
  - secret files                       -> DENY
  - destructive Git actions            -> DENY by default
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class PermissionOutcome(str, Enum):
    ALLOW = "ALLOW"
    ASK = "ASK"
    DENY = "DENY"


@dataclass(frozen=True)
class PermissionDecision:
    outcome: PermissionOutcome
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {"outcome": self.outcome.value, "reason": self.reason}


class PermissionDenied(Exception):
    def __init__(self, tool: str, reason: str) -> None:
        super().__init__(reason)
        self.tool = tool
        self.reason = reason


# Filename/substring patterns treated as secrets for ALL tools.
SECRET_FILENAMES = (
    "operator.token",
    ".env",
    "*.pem",
    "*.key",
    "id_rsa",
    "id_ed25519",
    ".netrc",
    "credentials",
    "secret",
    "token",
)


def _match_secret(path: Path) -> bool:
    name = path.name.lower()
    parts = [p.lower() for p in path.parts]
    for pattern in SECRET_FILENAMES:
        if pattern.startswith("*"):
            if name.endswith(pattern[1:]):
                return True
        elif pattern in parts or name == pattern:
            return True
    return False


@dataclass
class PermissionPolicy:
    approved_roots: list[str] = field(default_factory=list)
    write_auto_allow: bool = False
    allow_git_write: bool = False
    default_delete: PermissionOutcome = PermissionOutcome.ASK
    default_shell: PermissionOutcome = PermissionOutcome.DENY
    default_write: PermissionOutcome = PermissionOutcome.ASK

    def _canonical_roots(self) -> list[Path]:
        return [Path(r).resolve() for r in self.approved_roots]

    def _path_in_scope(self, raw_path: str) -> bool:
        try:
            target = Path(raw_path).resolve()
        except OSError:
            return False
        roots = self._canonical_roots()
        if not roots:
            return False
        return any(target == root or root in target.parents for root in roots)

    def evaluate(self, tool: str, args: dict[str, Any]) -> PermissionDecision:
        """Evaluate a tool invocation.  Class is inferred from tool name; a
        small per-tool rule set keeps the logic explicit."""
        name = tool.lower()

        # Arbitrary shell is always denied by default.
        if "shell" in name or name in ("exec", "bash", "sh", "cmd"):
            return PermissionDecision(
                self.default_shell, "arbitrary shell execution is denied by default"
            )

        if name in ("git.status", "git.diff"):
            return PermissionDecision(PermissionOutcome.ALLOW, "non-destructive git read tool")

        if name.startswith("git."):
            if self.allow_git_write:
                return PermissionDecision(
                    PermissionOutcome.ASK, "destructive git action requires approval"
                )
            return PermissionDecision(
                PermissionOutcome.DENY, "destructive git action is denied by default"
            )

        if name == "files.read_scoped":
            path = str(args.get("path", ""))
            if _match_secret(Path(path)):
                return PermissionDecision(
                    PermissionOutcome.DENY, "secret path access is denied"
                )
            if self._path_in_scope(path):
                return PermissionDecision(
                    PermissionOutcome.ALLOW, "scoped read within approved roots"
                )
            return PermissionDecision(
                PermissionOutcome.DENY, "filesystem access outside approved roots"
            )

        if name == "files.write_scoped":
            path = str(args.get("path", ""))
            if _match_secret(Path(path)):
                return PermissionDecision(
                    PermissionOutcome.DENY, "secret path access is denied"
                )
            if not self._path_in_scope(path):
                return PermissionDecision(
                    PermissionOutcome.DENY, "filesystem access outside approved roots"
                )
            if self.write_auto_allow:
                return PermissionDecision(
                    PermissionOutcome.ALLOW, "scoped write auto-allowed by policy"
                )
            return PermissionDecision(
                self.default_write, "scoped write requires approval"
            )

        if name == "files.delete_scoped":
            path = str(args.get("path", ""))
            if not self._path_in_scope(path):
                return PermissionDecision(
                    PermissionOutcome.DENY, "filesystem access outside approved roots"
                )
            return PermissionDecision(
                self.default_delete, "deletion requires approval"
            )

        if name in (
            "system.health",
            "system.capabilities",
            "model.list",
            "model.health",
            "model.route",
            "memory.search",
            "memory.inspect",
            "evidence.query",
        ):
            return PermissionDecision(PermissionOutcome.ALLOW, "read-only system tool")

        if name in ("memory.write_proposal",):
            return PermissionDecision(
                PermissionOutcome.ALLOW,
                "memory proposals are non-durable by construction",
            )

        # Anything unclassified: require approval.
        return PermissionDecision(PermissionOutcome.ASK, "unclassified tool requires approval")


def build_phone_policy(workspace: str) -> PermissionPolicy:
    """Default phone-safe policy: workspace-scoped reads, ASK writes."""
    return PermissionPolicy(
        approved_roots=[str(Path(workspace).resolve())],
        write_auto_allow=False,
        allow_git_write=False,
    )
