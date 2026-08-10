"""Simple in-memory sliding-window rate limiter."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class RateDecision:
    allowed: bool
    retry_after: Optional[float]
    remaining: int


class RateLimiter:
    def __init__(self, max_requests: int = 120, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, list[float]] = {}

    def _prune(self, key: str, now: float) -> None:
        hits = self._hits.get(key, [])
        cutoff = now - self.window_seconds
        self._hits[key] = [t for t in hits if t > cutoff]

    def allow(self, key: str) -> RateDecision:
        now = time.monotonic()
        self._prune(key, now)
        hits = self._hits.setdefault(key, [])
        if len(hits) >= self.max_requests:
            oldest = hits[0]
            retry_after = max(0.0, self.window_seconds - (now - oldest))
            return RateDecision(allowed=False, retry_after=round(retry_after, 3), remaining=0)
        hits.append(now)
        return RateDecision(
            allowed=True,
            retry_after=None,
            remaining=max(0, self.max_requests - len(hits)),
        )
