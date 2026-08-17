"""Gemini external calling tool — ISOLATED from Lucy's core eco-system.

This module is intentionally self-contained.  It is NOT a model provider and
does NOT participate in Lucy's inference path, the ModelRouter, the routing
policy, or the foundation audit.  It is a single, opt-in, operator-approved
tool that sends ONLY the explicit prompt it is given to Google's Gemini REST
API.

Isolation guarantees (do not weaken these):
  * No memory, context, local build, weights, or evidence are ever transmitted.
  * Only the caller-supplied ``prompt`` (plus a fixed neutral system note) leaves
    the device.
  * The API key is read from the NEXUS_GEMINI_API_KEY environment variable only;
    it is never written to config, logs, or the evidence ledger.
  * The tool lives in this file alone; nothing here imports or mutates Lucy's
    provider/routing/foundation layers.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from ..context import ToolContext
from ..registry import ToolSpec

_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
_DEFAULT_MODEL = "gemini-2.0-flash"
# Fixed, neutral instruction.  Deliberately reveals nothing about Lucy, her
# build, her host, or her local environment.
_NEUTRAL_SYSTEM = (
    "You are a helpful assistant. Do not ask about the user's system, software, "
    "local environment, or any device-specific details."
)


class GeminiTransportError(Exception):
    """Raised on transport/protocol failures.  Never carries the API key."""


class GeminiClient:
    """Minimal Gemini REST client.

    Uses aiohttp by default.  A fake transport can be injected for deterministic,
    network-free tests (mirrors the Ollama provider's pluggable transport).
    """

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        transport: Any = None,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._transport = transport

    def _url(self) -> str:
        return _GEMINI_ENDPOINT.format(model=self.model) + f"?key={self.api_key}"

    def _payload(self, prompt: str, system: Optional[str]) -> dict[str, Any]:
        parts: list[dict[str, str]] = []
        if system:
            parts.append({"text": system})
        parts.append({"text": prompt})
        return {"contents": [{"role": "user", "parts": parts}]}

    async def generate(self, prompt: str, system: Optional[str] = None) -> str:
        if not prompt or not prompt.strip():
            raise GeminiTransportError("empty prompt refused")
        payload = self._payload(prompt, system)
        if self._transport is not None:
            raw = await self._transport.request("POST", self._url(), payload)
            return self._parse(raw)
        import aiohttp

        url = self._url()
        client_timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.post(url, json=payload) as resp:
                body = await resp.text()
                if resp.status >= 400:
                    raise GeminiTransportError(f"gemini http {resp.status}: {body[:300]}")
                try:
                    data = json.loads(body)
                except json.JSONDecodeError:
                    raise GeminiTransportError("non-JSON response from gemini")
                return self._parse(data)

    @staticmethod
    def _parse(data: Any) -> str:
        if isinstance(data, dict) and "candidates" in data:
            candidates = data["candidates"]
            if candidates and "content" in candidates[0]:
                parts = candidates[0]["content"].get("parts", [])
                return "".join(p.get("text", "") for p in parts)
        # Some fake transports return a pre-parsed {"text": ...} or a raw string.
        if isinstance(data, dict) and "text" in data:
            return str(data["text"])
        if isinstance(data, str):
            return data
        raise GeminiTransportError("unexpected gemini response shape")


async def gemini_ask(prompt: str, **kw: Any) -> dict[str, Any]:
    """Isolated external call to Google Gemini.

    Sends ONLY the literal ``prompt`` plus a fixed neutral system note.  No Lucy
    memory, context, local build, weights, or activity are attached.  The actual
    cloud call is gated by the permission policy (DENY by default, ASK when
    explicitly enabled), so this body only runs after operator approval.
    """
    context: ToolContext = kw["context"]
    cfg = context.config.gemini
    api_key = cfg.api_key or os.environ.get("NEXUS_GEMINI_API_KEY")
    if not api_key:
        return {"ok": False, "error": "gemini api key not configured (NEXUS_GEMINI_API_KEY)"}

    client = GeminiClient(
        api_key=api_key,
        model=cfg.model or _DEFAULT_MODEL,
        transport=getattr(context, "gemini_transport", None),
        timeout=cfg.timeout,
    )
    try:
        text = await client.generate(prompt, system=_NEUTRAL_SYSTEM)
    except GeminiTransportError as exc:
        await _record(context, ok=False, model=client.model, reason=str(exc))
        return {"ok": False, "error": str(exc)}

    await _record(context, ok=True, model=client.model, reason=f"{len(text)} chars")
    return {"ok": True, "model": client.model, "response": text}


async def _record(context: ToolContext, ok: bool, model: str, reason: str) -> None:
    """Append a TOOL_CALL evidence record (existence + outcome only).

    We deliberately do NOT write the prompt or the response text to the local
    ledger — what left the device stays with the cloud; the ledger only proves a
    call occurred and whether it succeeded.
    """
    evidence = getattr(context, "evidence", None)
    if evidence is None or not hasattr(evidence, "append"):
        return
    try:
        from ...evidence.schema import EvidenceRecord, EvidenceType

        record = EvidenceRecord(
            record_type=EvidenceType.TOOL_CALL,
            goal="gemini.ask",
            model=model,
            provider="gemini",
            host=getattr(context.config, "host_id", None),
            host_role=getattr(context.config, "host_role", None),
            completion_reason=("ok" if ok else "error") + ": " + reason,
        )
        await evidence.append(record)
    except Exception:
        # Evidence is best-effort; never let logging break the tool call.
        pass


def register_gemini_tool(registry: Any, context: ToolContext) -> None:
    """Register ``gemini.ask`` only when explicitly enabled AND a key is present.

    When disabled (the fail-closed default), the tool simply does not exist in
    Lucy's registry, so it can never be planned or invoked.
    """
    cfg = context.config.gemini
    if not cfg.enabled:
        return
    if not (cfg.api_key or os.environ.get("NEXUS_GEMINI_API_KEY")):
        return
    registry.register(
        ToolSpec(
            "gemini.ask",
            "isolated external call to Google Gemini (sends only the prompt you provide; no local context)",
            gemini_ask,
            "external",
        )
    )
