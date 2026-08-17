"""Tests for the isolated Gemini external tool (no live network).

These verify the tool's core promises:
  * it stays in its own module and never touches Lucy's inference/router/foundation,
  * it is fail-closed (unregistered / DENIED by default),
  * it sends ONLY the prompt + a neutral system note (no memory/context leakage).
"""

from __future__ import annotations

import unittest
from typing import Any, Optional

from lucy_edge.config import LucyEdgeConfig
from lucy_edge.tools.builtin.gemini import (
    GeminiClient,
    gemini_ask,
    register_gemini_tool,
)
from lucy_edge.tools.context import ToolContext
from lucy_edge.tools.permissions import PermissionOutcome, build_phone_policy
from lucy_edge.tools.registry import ToolRegistry


class FakeGeminiTransport:
    """Deterministic transport for GeminiClient: records the request and returns
    a canned Gemini-shaped response."""

    def __init__(self, text: str = "hi from gemini") -> None:
        self.text = text
        self.last: Optional[tuple[str, str, dict[str, Any]]] = None

    async def request(self, method: str, url: str, payload: Optional[dict] = None) -> Any:
        self.last = (method, url, payload)
        return {"candidates": [{"content": {"parts": [{"text": self.text}]}}]}


class FakeEvidence:
    def __init__(self) -> None:
        self.records: list[Any] = []

    async def append(self, record: Any) -> None:
        self.records.append(record)


def make_context(enabled: bool, api_key: Optional[str] = "TESTKEY", transport=None) -> ToolContext:
    config = LucyEdgeConfig()
    config.gemini.enabled = enabled
    config.gemini.api_key = api_key
    config.gemini.model = "gemini-2.0-flash"
    ctx = ToolContext(config=config)
    if transport is not None:
        ctx.gemini_transport = transport  # type: ignore[attr-defined]
    return ctx


class RegistrationTests(unittest.IsolatedAsyncioTestCase):
    def test_disabled_by_default_is_not_registered(self):
        registry = ToolRegistry(build_phone_policy("."))
        ctx = make_context(enabled=False, api_key="KEY")
        register_gemini_tool(registry, ctx)
        self.assertNotIn("gemini.ask", registry.names())

    def test_registered_when_enabled_and_key_present(self):
        registry = ToolRegistry(build_phone_policy("."))
        ctx = make_context(enabled=True, api_key="KEY")
        register_gemini_tool(registry, ctx)
        self.assertIn("gemini.ask", registry.names())

    def test_not_registered_without_key(self):
        registry = ToolRegistry(build_phone_policy("."))
        ctx = make_context(enabled=True, api_key=None)
        register_gemini_tool(registry, ctx)
        self.assertNotIn("gemini.ask", registry.names())


class PermissionTests(unittest.IsolatedAsyncioTestCase):
    def test_gemini_denied_when_external_disabled(self):
        policy = build_phone_policy(".", allow_external=False)
        decision = policy.evaluate("gemini.ask", {"prompt": "hi"})
        self.assertEqual(decision.outcome, PermissionOutcome.DENY)

    def test_gemini_asks_when_external_enabled(self):
        policy = build_phone_policy(".", allow_external=True)
        decision = policy.evaluate("gemini.ask", {"prompt": "hi"})
        self.assertEqual(decision.outcome, PermissionOutcome.ASK)


class IsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_sends_only_prompt_and_neutral_system(self):
        transport = FakeGeminiTransport()
        ctx = make_context(enabled=True, transport=transport)
        result = await gemini_ask("what is 2+2?", context=ctx)
        self.assertTrue(result["ok"])
        self.assertEqual(result["response"], "hi from gemini")

        method, url, payload = transport.last
        self.assertEqual(method, "POST")
        self.assertIn("key=TESTKEY", url)
        # Exactly one user turn with exactly two parts: neutral system + prompt.
        self.assertEqual(len(payload["contents"]), 1)
        parts = payload["contents"][0]["parts"]
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[1]["text"], "what is 2+2?")
        # The prompt must not be augmented with Lucy internals.
        blob = str(payload)
        for forbidden in ("memory", "local build", "evidence", "weight"):
            self.assertNotIn(forbidden, blob)

    async def test_empty_prompt_refused(self):
        transport = FakeGeminiTransport()
        ctx = make_context(enabled=True, transport=transport)
        result = await gemini_ask("   ", context=ctx)
        self.assertFalse(result["ok"])
        self.assertIn("empty", result["error"])

    async def test_missing_key_reports_error(self):
        ctx = make_context(enabled=True, api_key=None)
        ctx.config.gemini.api_key = None
        import os

        os.environ.pop("NEXUS_GEMINI_API_KEY", None)
        result = await gemini_ask("hi", context=ctx)
        self.assertFalse(result["ok"])
        self.assertIn("not configured", result["error"])

    async def test_evidence_records_call_outcome_only(self):
        transport = FakeGeminiTransport()
        ctx = make_context(enabled=True, transport=transport)
        evidence = FakeEvidence()
        ctx.evidence = evidence
        await gemini_ask("hello", context=ctx)
        self.assertEqual(len(evidence.records), 1)
        rec = evidence.records[0]
        self.assertEqual(rec.record_type.value, "TOOL_CALL")
        # The ledger must NOT contain the prompt or the response text.
        self.assertNotIn("hello", str(rec.model_dump()))
        self.assertNotIn("hi from gemini", str(rec.model_dump()))


class ClientParseTests(unittest.IsolatedAsyncioTestCase):
    async def test_parse_handles_pre_parsed_text(self):
        client = GeminiClient(api_key="K", transport=FakeGeminiTransport())
        self.assertEqual(await client.generate("x"), "hi from gemini")

    async def test_parse_handles_bare_string(self):
        class StrTransport:
            async def request(self, *a, **k):
                return "bare"

        client = GeminiClient(api_key="K", transport=StrTransport())
        self.assertEqual(await client.generate("x"), "bare")

    async def test_parse_rejects_unexpected_shape(self):
        class BadTransport:
            async def request(self, *a, **k):
                return {"unexpected": True}

        client = GeminiClient(api_key="K", transport=BadTransport())
        with self.assertRaises(Exception):
            await client.generate("x")
