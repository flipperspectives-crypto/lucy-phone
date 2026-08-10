"""Real MCP transport tests.

Exercises ``StdioTransport`` and ``HttpTransport`` against LOCAL controlled
fixtures only:

* Stdio: a small Python script that speaks newline-delimited JSON-RPC over
  stdin/stdout (no third-party MCP server involved).
* HTTP: a local ``aiohttp`` web server that speaks JSON-RPC (direct JSON and
  SSE) on a loopback port.

No external servers, no network beyond loopback, no heavy inference.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import textwrap
import unittest
from typing import Any

import aiohttp
from aiohttp import web

from lucy_edge.mcp import (
    FakeMCPTransport,
    MCPAllowlist,
    MCPConfig,
    MCPServerClient,
    MCPServerConfig,
    MCPTransportError,
)
from lucy_edge.mcp.transport import HttpTransport, StdioTransport


# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #

TOOLS = [
    {
        "name": "echo",
        "description": "echo back the input",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
        },
    },
    {
        "name": "add",
        "description": "add two numbers",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
        },
    },
]

SERVER_INFO = {"name": "local-fixture", "version": "1.0.0", "capabilities": {"tools": {}}}


def _make_stdio_script() -> str:
    """Write a minimal MCP stdio server to a temp file and return its path."""
    script = textwrap.dedent(
        """
        import json
        import sys

        TOOLS = %(tools)s
        SERVER_INFO = %(info)s

        def handle(req):
            method = req.get("method", "")
            params = req.get("params", {}) or {}
            if method == "initialize":
                return {"protocolVersion": "2024-11-05", "capabilities": SERVER_INFO.get("capabilities", {}), "serverInfo": {"name": SERVER_INFO["name"], "version": SERVER_INFO["version"]}}
            if method == "tools/list":
                return {"tools": TOOLS}
            if method == "tools/call":
                name = params.get("name", "")
                args = params.get("arguments", {}) or {}
                if name == "echo":
                    return {"content": [{"type": "text", "text": str(args.get("text", ""))}]}
                if name == "add":
                    return {"content": [{"type": "text", "text": str(float(args.get("a", 0)) + float(args.get("b", 0)))}]}
                return {"content": [{"type": "text", "text": "unknown"}], "isError": True}
            return {}

        def main():
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                try:
                    req = json.loads(line)
                except json.JSONDecodeError:
                    continue
                result = handle(req)
                resp = {"jsonrpc": "2.0", "id": req.get("id"), "result": result}
                sys.stdout.write(json.dumps(resp) + "\\n")
                sys.stdout.flush()

        if __name__ == "__main__":
            main()
        """
    ) % {"tools": repr(TOOLS), "info": repr(SERVER_INFO)}
    fd, path = tempfile.mkstemp(prefix="lucy_mcp_stdio_", suffix=".py")
    os.write(fd, script.encode("utf-8"))
    os.close(fd)
    return path


# --------------------------------------------------------------------------- #
# Stdio transport
# --------------------------------------------------------------------------- #

class StdioTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_initialize_and_list_tools(self):
        script = _make_stdio_script()
        try:
            transport = StdioTransport(command=sys.executable, args=[script], env={})
            info = await transport.initialize(timeout=5.0)
            self.assertEqual(info["serverInfo"]["name"], "local-fixture")
            tools = await transport.list_tools(timeout=5.0)
            self.assertEqual(len(tools), 2)
            self.assertEqual(tools[0]["name"], "echo")
            await transport.close()
        finally:
            os.unlink(script)

    async def test_call_tool_echo(self):
        script = _make_stdio_script()
        try:
            transport = StdioTransport(command=sys.executable, args=[script], env={})
            await transport.initialize(timeout=5.0)
            result = await transport.call_tool("echo", {"text": "hello"}, timeout=5.0)
            self.assertEqual(result["content"][0]["text"], "hello")
            await transport.close()
        finally:
            os.unlink(script)

    async def test_call_tool_add(self):
        script = _make_stdio_script()
        try:
            transport = StdioTransport(command=sys.executable, args=[script], env={})
            await transport.initialize(timeout=5.0)
            result = await transport.call_tool("add", {"a": 2, "b": 3}, timeout=5.0)
            self.assertEqual(result["content"][0]["text"], "5.0")
            await transport.close()
        finally:
            os.unlink(script)

    async def test_rejects_empty_command(self):
        with self.assertRaises(MCPTransportError):
            StdioTransport(command="", args=[], env={})

    async def test_handles_dead_process(self):
        """A server that exits immediately must surface a clean error, not crash."""
        # Use a command that exits immediately
        if sys.platform.startswith("win"):
            transport = StdioTransport(command="cmd", args=["/c", "exit"], env={})
        else:
            transport = StdioTransport(command="true", args=[], env={})
        # initialize should fail cleanly (process exits before responding)
        with self.assertRaises(MCPTransportError):
            await transport.initialize(timeout=2.0)
        await transport.close()

    async def test_close_is_idempotent(self):
        script = _make_stdio_script()
        try:
            transport = StdioTransport(command=sys.executable, args=[script], env={})
            await transport.initialize(timeout=5.0)
            await transport.close()
            await transport.close()  # second close must not raise
        finally:
            os.unlink(script)

    async def test_concurrent_calls_are_serialized(self):
        """Multiple calls share one stdin/stdout; they must not interleave."""
        script = _make_stdio_script()
        try:
            transport = StdioTransport(command=sys.executable, args=[script], env={})
            await transport.initialize(timeout=5.0)
            results = await asyncio.gather(
                *[transport.call_tool("echo", {"text": f"msg{i}"}, timeout=5.0) for i in range(5)]
            )
            texts = [r["content"][0]["text"] for r in results]
            self.assertEqual(texts, [f"msg{i}" for i in range(5)])
            await transport.close()
        finally:
            os.unlink(script)


# --------------------------------------------------------------------------- #
# HTTP transport (local aiohttp server)
# --------------------------------------------------------------------------- #


class _DirectHandler:
    """JSON-RPC handler that returns direct JSON responses."""

    def __init__(self) -> None:
        self.tools = TOOLS
        self.server_info = SERVER_INFO
        self.calls: list[tuple[str, dict]] = []

    async def handle(self, request: web.Request) -> web.Response:
        body = await request.json()
        method = body.get("method", "")
        params = body.get("params", {}) or {}
        req_id = body.get("id")
        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": self.server_info.get("capabilities", {}),
                "serverInfo": {
                    "name": self.server_info["name"],
                    "version": self.server_info["version"],
                },
            }
        elif method == "tools/list":
            result = {"tools": self.tools}
        elif method == "tools/call":
            self.calls.append((params.get("name", ""), params.get("arguments", {})))
            name = params.get("name", "")
            args = params.get("arguments", {}) or {}
            if name == "echo":
                result = {"content": [{"type": "text", "text": str(args.get("text", ""))}]}
            elif name == "add":
                result = {
                    "content": [
                        {
                            "type": "text",
                            "text": str(float(args.get("a", 0)) + float(args.get("b", 0))),
                        }
                    ]
                }
            else:
                result = {"content": [{"type": "text", "text": "unknown"}], "isError": True}
        else:
            result = {}
        return web.json_response({"jsonrpc": "2.0", "id": req_id, "result": result})


class _SSEHandler:
    """JSON-RPC handler that returns SSE-streamed responses."""

    def __init__(self) -> None:
        self.tools = TOOLS
        self.server_info = SERVER_INFO
        self.calls: list[tuple[str, dict]] = []

    async def handle(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        method = body.get("method", "")
        params = body.get("params", {}) or {}
        req_id = body.get("id")
        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": self.server_info.get("capabilities", {}),
                "serverInfo": {
                    "name": self.server_info["name"],
                    "version": self.server_info["version"],
                },
            }
        elif method == "tools/list":
            result = {"tools": self.tools}
        elif method == "tools/call":
            self.calls.append((params.get("name", ""), params.get("arguments", {})))
            name = params.get("name", "")
            args = params.get("arguments", {}) or {}
            if name == "echo":
                result = {"content": [{"type": "text", "text": str(args.get("text", ""))}]}
            else:
                result = {"content": [{"type": "text", "text": "unknown"}], "isError": True}
        else:
            result = {}
        resp = web.StreamResponse(
            status=200,
            reason="OK",
            headers={"Content-Type": "text/event-stream"},
        )
        await resp.prepare(request)
        data = json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result})
        await resp.write(b"event: message\n")
        await resp.write(b"data: " + data.encode("utf-8") + b"\n\n")
        await resp.write_eof()
        return resp


async def _start_server(handler: Any) -> tuple[web.AppRunner, str]:
    app = web.Application()
    app.router.add_post("/mcp", handler.handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}/mcp"


class HttpTransportTests(unittest.IsolatedAsyncioTestCase):
    async def _with_server(self, handler: Any):
        runner, url = await _start_server(handler)
        try:
            return url
        finally:
            await runner.cleanup()

    async def test_direct_json_initialize_and_list(self):
        handler = _DirectHandler()
        runner, url = await _start_server(handler)
        try:
            transport = HttpTransport(url=url, headers={})
            info = await transport.initialize(timeout=5.0)
            self.assertEqual(info["serverInfo"]["name"], "local-fixture")
            tools = await transport.list_tools(timeout=5.0)
            self.assertEqual(len(tools), 2)
            await transport.close()
        finally:
            await runner.cleanup()

    async def test_direct_json_call_tool(self):
        handler = _DirectHandler()
        runner, url = await _start_server(handler)
        try:
            transport = HttpTransport(url=url, headers={})
            await transport.initialize(timeout=5.0)
            result = await transport.call_tool("add", {"a": 10, "b": 20}, timeout=5.0)
            self.assertEqual(result["content"][0]["text"], "30.0")
            await transport.close()
        finally:
            await runner.cleanup()

    async def test_sse_initialize_and_list(self):
        handler = _SSEHandler()
        runner, url = await _start_server(handler)
        try:
            transport = HttpTransport(url=url, headers={})
            info = await transport.initialize(timeout=5.0)
            self.assertEqual(info["serverInfo"]["name"], "local-fixture")
            tools = await transport.list_tools(timeout=5.0)
            self.assertEqual(len(tools), 2)
            await transport.close()
        finally:
            await runner.cleanup()

    async def test_sse_call_tool(self):
        handler = _SSEHandler()
        runner, url = await _start_server(handler)
        try:
            transport = HttpTransport(url=url, headers={})
            await transport.initialize(timeout=5.0)
            result = await transport.call_tool("echo", {"text": "sse-hello"}, timeout=5.0)
            self.assertEqual(result["content"][0]["text"], "sse-hello")
            await transport.close()
        finally:
            await runner.cleanup()

    async def test_rejects_empty_url(self):
        with self.assertRaises(MCPTransportError):
            HttpTransport(url="", headers={})

    async def test_connection_refused(self):
        """Connecting to a dead port must surface a clean error, not crash."""
        transport = HttpTransport(url="http://127.0.0.1:1/mcp", headers={})
        with self.assertRaises(MCPTransportError):
            await transport.initialize(timeout=2.0)
        await transport.close()

    async def test_close_is_idempotent(self):
        handler = _DirectHandler()
        runner, url = await _start_server(handler)
        try:
            transport = HttpTransport(url=url, headers={})
            await transport.initialize(timeout=5.0)
            await transport.close()
            await transport.close()
        finally:
            await runner.cleanup()


# --------------------------------------------------------------------------- #
# Integration: MCPServerClient + real transports
# --------------------------------------------------------------------------- #

class RealTransportClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_client_with_stdio_transport(self):
        script = _make_stdio_script()
        try:
            sc = MCPServerConfig(
                server_id="stdio-srv",
                transport="stdio",
                command=sys.executable,
                args=[script],
            )
            cfg = MCPConfig(enabled=True, servers=[sc])
            client = MCPServerClient(sc, StdioTransport(sys.executable, [script], {}), MCPAllowlist(cfg))
            report = await client.discover()
            self.assertTrue(report.ok)
            self.assertEqual(len(report.tools), 2)
            result = await client.call("echo", {"text": "via-client"})
            self.assertTrue(result.ok)
            self.assertEqual(result.output, "via-client")
            await client.shutdown()
        finally:
            os.unlink(script)

    async def test_client_with_http_transport(self):
        handler = _DirectHandler()
        runner, url = await _start_server(handler)
        try:
            sc = MCPServerConfig(
                server_id="http-srv",
                transport="http",
                url=url,
            )
            cfg = MCPConfig(enabled=True, servers=[sc])
            client = MCPServerClient(sc, HttpTransport(url, {}), MCPAllowlist(cfg))
            report = await client.discover()
            self.assertTrue(report.ok)
            self.assertEqual(len(report.tools), 2)
            result = await client.call("add", {"a": 7, "b": 8})
            self.assertTrue(result.ok)
            self.assertEqual(result.output, "15.0")
            await client.shutdown()
        finally:
            await runner.cleanup()


if __name__ == "__main__":
    unittest.main()
