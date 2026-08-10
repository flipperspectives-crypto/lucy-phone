"""MCP integration tests.

All tests use ``FakeMCPTransport`` — NO real MCP server, NO subprocess, NO
network, safe for the S25 Ultra phone.  The stdio/http transports that connect
to real servers are NOT exercised here (requires external MCP servers).
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from lucy_edge.mcp import (
    FakeMCPTransport,
    MCPAllowlist,
    MCPAudit,
    MCPCallResult,
    MCPConfig,
    MCPRegistry,
    MCPServerClient,
    MCPServerConfig,
    MCPToolDescriptor,
    MCPTransportError,
    _looks_like_secret_key,
)

from .helpers import make_config, temp_dir


# -- helpers ------------------------------------------------------------------

def _server_cfg(server_id: str = "test-srv", **kw) -> MCPServerConfig:
    cfg = MCPServerConfig(server_id=server_id, **kw)
    return cfg


def _tool(name: str = "echo", description: str = "echo back") -> dict:
    return {
        "name": name,
        "description": description,
        "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}},
    }


TOOLS_OK = [_tool("echo"), _tool("add")]


# -- allowlist ----------------------------------------------------------------

class MCPAllowlistTests(unittest.TestCase):
    def test_valid_server_id(self):
        al = MCPAllowlist(MCPConfig())
        self.assertTrue(al.is_server_id_valid("my-server"))
        self.assertTrue(al.is_server_id_valid("server_1"))
        self.assertFalse(al.is_server_id_valid(""))  # empty
        self.assertFalse(al.is_server_id_valid("has space"))
        self.assertFalse(al.is_server_id_valid("a" * 65))  # too long
        self.assertFalse(al.is_server_id_valid("../etc"))  # path traversal

    def test_valid_tool_name(self):
        al = MCPAllowlist(MCPConfig())
        self.assertTrue(al.is_tool_name_valid("echo"))
        self.assertTrue(al.is_tool_name_valid("my_tool.v1"))
        self.assertFalse(al.is_tool_name_valid(""))
        self.assertFalse(al.is_tool_name_valid("has space"))
        self.assertFalse(al.is_tool_name_valid("a" * 129))

    def test_server_allowlist_enforced(self):
        cfg = MCPConfig(servers=[_server_cfg("alpha"), _server_cfg("beta")])
        al = MCPAllowlist(cfg)
        self.assertTrue(al.is_server_allowlisted("alpha"))
        self.assertFalse(al.is_server_allowlisted("gamma"))

    def test_tool_allowlist_per_server(self):
        cfg = MCPConfig(
            servers=[_server_cfg("s1", allowed_tools=["echo"])]
        )
        al = MCPAllowlist(cfg)
        self.assertTrue(al.is_tool_allowed("s1", "echo"))
        self.assertFalse(al.is_tool_allowed("s1", "add"))

    def test_tool_denylist_per_server(self):
        cfg = MCPConfig(
            servers=[_server_cfg("s1", denied_tools=["dangerous"])]
        )
        al = MCPAllowlist(cfg)
        self.assertTrue(al.is_tool_allowed("s1", "echo"))
        self.assertFalse(al.is_tool_allowed("s1", "dangerous"))

    def test_global_denylist(self):
        cfg = MCPConfig(
            global_denied_tools=["rm_rf"],
            servers=[_server_cfg("s1")],
        )
        al = MCPAllowlist(cfg)
        self.assertFalse(al.is_tool_allowed("s1", "rm_rf"))
        self.assertTrue(al.is_tool_allowed("s1", "echo"))

    def test_sanitize_args_strips_secrets(self):
        al = MCPAllowlist(MCPConfig())
        clean = al.sanitize_args({"text": "hello", "api_key": "sekret", "password": "x"})
        self.assertIn("text", clean)
        self.assertNotIn("api_key", clean)
        self.assertNotIn("password", clean)

    def test_sanitize_args_drops_non_serializable(self):
        al = MCPAllowlist(MCPConfig())
        clean = al.sanitize_args({"text": "hello", "bad": object()})
        self.assertIn("text", clean)
        self.assertNotIn("bad", clean)

    def test_looks_like_secret_key(self):
        self.assertTrue(_looks_like_secret_key("api_key"))
        self.assertTrue(_looks_like_secret_key("auth_token"))
        self.assertTrue(_looks_like_secret_key("DB_PASSWORD"))
        self.assertFalse(_looks_like_secret_key("query"))
        self.assertFalse(_looks_like_secret_key("text"))


# -- transport ----------------------------------------------------------------

class FakeMCPTransportTests(unittest.TestCase):
    def test_returns_canned_tools(self):
        transport = FakeMCPTransport(tools=TOOLS_OK)
        # Can't run async here directly; covered in client tests below.
        self.assertEqual(len(transport._tools), 2)

    def test_close_flag(self):
        transport = FakeMCPTransport()
        self.assertFalse(transport.closed)


# -- server client ------------------------------------------------------------

class MCPServerClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_discover_returns_tools(self):
        transport = FakeMCPTransport(tools=TOOLS_OK)
        client = MCPServerClient(_server_cfg("s1"), transport, MCPAllowlist(MCPConfig(servers=[_server_cfg("s1")])))
        report = await client.discover()
        self.assertTrue(report.ok)
        self.assertEqual(len(report.tools), 2)
        self.assertEqual(report.tools[0].name, "echo")
        self.assertEqual(report.tools[0].qualified_name, "mcp.s1.echo")

    async def test_discover_respects_tool_allowlist(self):
        transport = FakeMCPTransport(tools=TOOLS_OK)
        cfg = MCPConfig(servers=[_server_cfg("s1", allowed_tools=["echo"])])
        client = MCPServerClient(_server_cfg("s1"), transport, MCPAllowlist(cfg))
        report = await client.discover()
        self.assertTrue(report.ok)
        self.assertEqual(len(report.tools), 1)
        self.assertEqual(report.tools[0].name, "echo")

    async def test_discover_handles_init_failure(self):
        transport = FakeMCPTransport(fail_initialize="boom")
        cfg = MCPConfig(servers=[_server_cfg("s1")])
        client = MCPServerClient(_server_cfg("s1"), transport, MCPAllowlist(cfg))
        report = await client.discover()
        self.assertFalse(report.ok)
        self.assertIn("initialize failed", report.error)

    async def test_call_returns_output(self):
        transport = FakeMCPTransport(tools=TOOLS_OK)
        cfg = MCPConfig(servers=[_server_cfg("s1")])
        client = MCPServerClient(_server_cfg("s1"), transport, MCPAllowlist(cfg))
        await client.discover()
        result = await client.call("echo", {"text": "hi"})
        self.assertTrue(result.ok)
        self.assertIsNotNone(result.output)

    async def test_call_strips_secret_args(self):
        captured = {}

        def handler(name, args):
            captured.update(args)
            return {"content": [{"type": "text", "text": "ok"}]}

        transport = FakeMCPTransport(tools=TOOLS_OK, call_handler=handler)
        cfg = MCPConfig(servers=[_server_cfg("s1")])
        client = MCPServerClient(_server_cfg("s1"), transport, MCPAllowlist(cfg))
        await client.discover()
        await client.call("echo", {"text": "hi", "api_key": "secret"})
        self.assertIn("text", captured)
        self.assertNotIn("api_key", captured)

    async def test_call_before_init_fails_cleanly(self):
        transport = FakeMCPTransport(tools=TOOLS_OK)
        cfg = MCPConfig(servers=[_server_cfg("s1")])
        client = MCPServerClient(_server_cfg("s1"), transport, MCPAllowlist(cfg))
        result = await client.call("echo", {})
        self.assertFalse(result.ok)
        self.assertIn("not initialized", result.error)

    async def test_discover_caps_tool_count(self):
        many = [_tool(f"t{i}") for i in range(50)]
        transport = FakeMCPTransport(tools=many)
        cfg = MCPConfig(servers=[_server_cfg("s1")], max_tools_per_server=10)
        client = MCPServerClient(_server_cfg("s1"), transport, MCPAllowlist(cfg))
        report = await client.discover()
        self.assertTrue(report.ok)
        self.assertLessEqual(len(report.tools), 10)


# -- registry -----------------------------------------------------------------

class MCPRegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_inert_when_disabled(self):
        cfg = MCPConfig(enabled=False, servers=[_server_cfg("s1")])
        reg = MCPRegistry(cfg)
        reports = await reg.open()
        self.assertEqual(reports, {})
        self.assertFalse(reg.available)

    async def test_discovers_and_lists_tools(self):
        cfg = MCPConfig(
            enabled=True,
            servers=[_server_cfg("s1")],
        )
        reg = MCPRegistry(cfg)
        # Inject fake transport into the server config
        cfg.servers[0]._transport = FakeMCPTransport(tools=TOOLS_OK)
        await reg.open()
        self.assertTrue(reg.available)
        tools = reg.all_tools()
        self.assertEqual(len(tools), 2)

    async def test_failure_isolation(self):
        """One failing server must not stop the other from being discovered."""
        good_cfg = _server_cfg("good")
        bad_cfg = _server_cfg("bad")
        cfg = MCPConfig(enabled=True, servers=[good_cfg, bad_cfg])
        reg = MCPRegistry(cfg)
        good_cfg._transport = FakeMCPTransport(tools=TOOLS_OK)
        bad_cfg._transport = FakeMCPTransport(fail_initialize="kaboom")
        reports = await reg.open()
        self.assertTrue(reports["good"].ok)
        self.assertFalse(reports["bad"].ok)
        # The registry as a whole still reports available because 'good' works.
        self.assertTrue(reg.available)

    async def test_unknown_server_id_skipped(self):
        cfg = MCPConfig(enabled=True, servers=[_server_cfg("bad id!")])
        reg = MCPRegistry(cfg)
        reports = await reg.open()
        self.assertFalse(reports["bad id!"].ok)
        self.assertIn("invalid server_id", reports["bad id!"].error)

    async def test_call_routes_to_server(self):
        cfg = MCPConfig(enabled=True, servers=[_server_cfg("s1")])
        reg = MCPRegistry(cfg)
        cfg.servers[0]._transport = FakeMCPTransport(tools=TOOLS_OK)
        await reg.open()
        result = await reg.call("s1", "echo", {"text": "hi"})
        self.assertTrue(result.ok)

    async def test_call_unknown_server_fails_cleanly(self):
        cfg = MCPConfig(enabled=True)
        reg = MCPRegistry(cfg)
        await reg.open()
        result = await reg.call("nope", "echo", {})
        self.assertFalse(result.ok)
        self.assertIn("unknown MCP server", result.error)

    async def test_close_is_safe(self):
        cfg = MCPConfig(enabled=True, servers=[_server_cfg("s1")])
        reg = MCPRegistry(cfg)
        cfg.servers[0]._transport = FakeMCPTransport(tools=TOOLS_OK)
        await reg.open()
        await reg.close()
        self.assertEqual(reg.server_ids(), [])


# -- audit --------------------------------------------------------------------

class MCPAuditTests(unittest.IsolatedAsyncioTestCase):
    def test_record_discovery_writes_evidence(self):
        ledger = _fake_ledger()
        audit = MCPAudit(ledger)

        async def run():
            from lucy_edge.mcp import MCPCapabilityReport
            await audit.record_discovery(
                MCPCapabilityReport(server_id="s1", ok=True, tools=[MCPToolDescriptor("e", "d", {}, "s1")])
            )
        import asyncio
        asyncio.run(run())
        self.assertEqual(len(ledger.records), 1)
        self.assertEqual(ledger.records[0].goal, "MCP discovery: s1")

    def test_record_call_hashes_output(self):
        ledger = _fake_ledger()
        audit = MCPAudit(ledger)

        async def run():
            await audit.record_call("s1", "echo", MCPCallResult(ok=True, output="hello"))
        import asyncio
        asyncio.run(run())
        self.assertEqual(len(ledger.records), 1)
        self.assertEqual(ledger.records[0].goal, "MCP call: s1.echo")

    def test_no_ledger_is_safe(self):
        audit = MCPAudit(None)
        async def run():
            await audit.record_discovery(MagicMock(ok=True))
        import asyncio
        asyncio.run(run())  # must not raise


def _fake_ledger():
    class FakeLedger:
        def __init__(self):
            self.records = []
        async def append(self, record):
            self.records.append(record)
            return record
    return FakeLedger()


# -- permission policy --------------------------------------------------------

class MCPPermissionTests(unittest.TestCase):
    def test_mcp_tool_defaults_to_ask(self):
        from lucy_edge.tools.permissions import PermissionPolicy, PermissionOutcome
        pol = PermissionPolicy()
        d = pol.evaluate("mcp.s1.echo", {})
        self.assertEqual(d.outcome, PermissionOutcome.ASK)


if __name__ == "__main__":
    unittest.main()
