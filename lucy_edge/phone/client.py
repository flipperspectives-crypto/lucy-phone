"""Lightweight control-plane client.

The client only talks HTTP to the gateway.  It never imports providers, never
calls a generate function and never touches model weights.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import aiohttp


class TokenProvider:
    """Reads the auth token without ever printing or logging it."""

    def __init__(self, token_file: str = "data/operator.token") -> None:
        self.token_file = token_file

    def token(self) -> Optional[str]:
        env_token = os.environ.get("NEXUS_SESSION_TOKEN")
        if env_token:
            return env_token
        path = Path(self.token_file)
        if path.exists():
            try:
                value = path.read_text().strip()
                return value or None
            except OSError:
                return None
        return None


class LucyEdgeClient:
    """Control-plane client (no inference, no model weights, no local runtime)."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8970",
        token_provider: Optional[TokenProvider] = None,
        timeout: float = 15.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token_provider = token_provider or TokenProvider()
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    async def _session_get(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    def _headers(self) -> dict[str, str]:
        token = self.token_provider.token()
        headers = {"Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    # --- control-plane operations ------------------------------------------
    async def health(self) -> dict[str, Any]:
        session = await self._session_get()
        async with session.get(f"{self.base_url}/health") as resp:
            return await resp.json()

    async def authenticate(self) -> dict[str, Any]:
        session = await self._session_get()
        async with session.get(f"{self.base_url}/v1/auth/status", headers=self._headers()) as resp:
            return {"authenticated": resp.status == 200, "http_status": resp.status, **await resp.json()}

    async def chat(self, model: str, messages: list[dict[str, Any]], provider: str = "mock") -> dict[str, Any]:
        session = await self._session_get()
        async with session.post(
            f"{self.base_url}/v1/chat",
            json={"model": model, "messages": messages, "provider": provider},
            headers=self._headers(),
        ) as resp:
            return await resp.json()

    async def stream_chat(
        self, model: str, messages: list[dict[str, Any]], provider: str = "mock"
    ) -> AsyncIterator[dict[str, Any]]:
        session = await self._session_get()
        async with session.post(
            f"{self.base_url}/v1/chat/stream",
            json={"model": model, "messages": messages, "provider": provider},
            headers=self._headers(),
        ) as resp:
            async for line in resp.content:
                text = line.decode("utf-8", "replace").strip()
                if not text.startswith("data:"):
                    continue
                payload = text[len("data:"):].strip()
                if payload:
                    import json

                    yield json.loads(payload)

    async def submit_task(self, goal: str, limits: Optional[dict] = None) -> dict[str, Any]:
        session = await self._session_get()
        payload: dict[str, Any] = {"goal": goal}
        if limits:
            payload["limits"] = limits
        async with session.post(
            f"{self.base_url}/v1/agent/tasks", json=payload, headers=self._headers()
        ) as resp:
            return await resp.json()

    async def task_status(self, run_id: str) -> dict[str, Any]:
        session = await self._session_get()
        async with session.get(
            f"{self.base_url}/v1/agent/tasks/{run_id}", headers=self._headers()
        ) as resp:
            return await resp.json()

    async def evidence(self, limit: int = 50, record_type: Optional[str] = None) -> dict[str, Any]:
        session = await self._session_get()
        params: dict[str, str] = {"limit": str(limit)}
        if record_type:
            params["record_type"] = record_type
        async with session.get(
            f"{self.base_url}/v1/evidence", params=params, headers=self._headers()
        ) as resp:
            return await resp.json()

    async def introspect(self) -> dict[str, Any]:
        session = await self._session_get()
        async with session.get(
            f"{self.base_url}/v1/lucy/introspect", headers=self._headers()
        ) as resp:
            return await resp.json()

    async def hardware_snapshot(self) -> dict[str, Any]:
        session = await self._session_get()
        async with session.get(
            f"{self.base_url}/v1/hardware/snapshot", headers=self._headers()
        ) as resp:
            return await resp.json()

    # The control-plane method surface (used by tests to prove no inference path).
    CONTROL_PLANE_METHODS = frozenset(
        {
            "health",
            "authenticate",
            "chat",
            "stream_chat",
            "submit_task",
            "task_status",
            "evidence",
            "introspect",
            "hardware_snapshot",
        }
    )
