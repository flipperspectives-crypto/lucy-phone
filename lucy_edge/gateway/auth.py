"""Gateway authentication.

Token resolution order (never logged or printed):
    1. NEXUS_SESSION_TOKEN environment variable
    2. data/operator.token file

A stale environment token previously caused 401s, so resolution happens once
at startup and the source is reported (not the value).
"""

from __future__ import annotations

import hmac
import os
from pathlib import Path
from typing import Optional


class AuthService:
    def __init__(
        self,
        token_file: str = "data/operator.token",
        enabled: bool = True,
        fixed_token: Optional[str] = None,
    ) -> None:
        self.enabled = enabled
        self._fixed_token = fixed_token
        self.token_file = token_file
        self.source: Optional[str] = None
        self.token = self._resolve_token()

    def _resolve_token(self) -> Optional[str]:
        if self._fixed_token is not None:
            self.source = "fixed_test"
            return self._fixed_token
        env_token = os.environ.get("NEXUS_SESSION_TOKEN")
        if env_token:
            self.source = "environment:NEXUS_SESSION_TOKEN"
            return env_token
        path = Path(self.token_file)
        if path.exists():
            try:
                value = path.read_text().strip()
                if value:
                    self.source = f"file:{path}"
                    return value
            except OSError:
                pass
        self.source = "none"
        return None

    @property
    def configured(self) -> bool:
        return self.token is not None

    def check(self, authorization: Optional[str]) -> bool:
        if not self.enabled:
            return True
        if not self.token:
            return False
        if not authorization:
            return False
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() != "bearer":
            return False
        return hmac.compare_digest(value.strip(), self.token)

    def describe(self) -> dict[str, str]:
        return {"enabled": str(self.enabled), "configured": str(self.configured), "source": self.source or "none"}
