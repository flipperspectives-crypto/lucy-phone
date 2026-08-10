"""Real MCP transports: stdio (subprocess) and Streamable HTTP (with SSE).

These are the production transports selected by ``MCPServerConfig.transport``.
They are exercised in tests ONLY against LOCAL controlled fixtures — a stdio
echo-server script and a local ``aiohttp`` HTTP server — never against random
third-party servers.

Protocol
--------
Both transports speak JSON-RPC 2.0:

* ``initialize``    -> server info + capabilities
* ``tools/list``    -> ``{"tools": [...]}``
* ``tools/call``    -> ``{"content": [...]}``

The stdio transport uses newline-delimited JSON over subprocess stdin/stdout
(per the MCP specification).  The HTTP transport POSTs JSON-RPC requests and
handles both direct JSON responses and ``text/event-stream`` (SSE) responses,
correlating by JSON-RPC id.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Optional

import aiohttp

from .base import MCPTransport, MCPTransportError


# --------------------------------------------------------------------------- #
# Stdio (subprocess)
# --------------------------------------------------------------------------- #


class StdioTransport(MCPTransport):
    """JSON-RPC 2.0 over stdio (subprocess).

    Spawns the configured command, sends newline-delimited JSON-RPC requests
    over stdin, reads responses from stdout.  Correlates responses to requests
    by JSON-RPC ``id``.  stderr is drained into /dev/null so a noisy server
    cannot block on a full pipe buffer.
    """

    def __init__(self, command: str, args: list[str], env: dict[str, str]) -> None:
        if not command:
            raise MCPTransportError("stdio transport requires a command")
        self._command = command
        self._args = list(args)
        self._env = dict(env)
        self._process: Optional[asyncio.subprocess.Process] = None
        self._id_counter = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._read_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._closed = False
        self._lock = asyncio.Lock()

    async def _ensure_started(self) -> None:
        if self._process is not None:
            return
        env = None
        if self._env:
            env = {**os.environ, **self._env}
        self._process = await asyncio.create_subprocess_exec(
            self._command,
            *self._args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        self._read_task = asyncio.create_task(self._read_loop())
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def _read_loop(self) -> None:
        """Read newline-delimited JSON responses from stdout."""
        try:
            while self._process and self._process.stdout and not self._closed:
                try:
                    line = await self._process.stdout.readline()
                except (asyncio.CancelledError, Exception):
                    break
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                req_id = msg.get("id")
                if req_id is not None and req_id in self._pending:
                    fut = self._pending.pop(req_id)
                    if not fut.done():
                        fut.set_result(msg)
        except asyncio.CancelledError:
            pass
        finally:
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(MCPTransportError("stdio read loop ended"))
            self._pending.clear()

    async def _drain_stderr(self) -> None:
        """Prevent the stderr pipe buffer from filling and blocking the server."""
        try:
            while self._process and self._process.stderr and not self._closed:
                try:
                    chunk = await self._process.stderr.read(4096)
                except (asyncio.CancelledError, Exception):
                    break
                if not chunk:
                    break
        except asyncio.CancelledError:
            pass

    async def _call(self, method: str, params: dict, timeout: float) -> dict:
        await self._ensure_started()
        if not self._process or not self._process.stdin:
            raise MCPTransportError("subprocess stdin not available")
        async with self._lock:
            self._id_counter += 1
            req_id = self._id_counter
            request = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
            data = json.dumps(request).encode("utf-8") + b"\n"
            fut: asyncio.Future = asyncio.get_event_loop().create_future()
            self._pending[req_id] = fut
            try:
                self._process.stdin.write(data)
                await self._process.stdin.drain()
                response = await asyncio.wait_for(fut, timeout=timeout)
                if "error" in response:
                    raise MCPTransportError(f"JSON-RPC error: {response['error']}")
                return response.get("result", {})
            except asyncio.TimeoutError:
                self._pending.pop(req_id, None)
                raise MCPTransportError(
                    f"stdio call to '{method}' timed out after {timeout}s"
                )
            except MCPTransportError:
                raise
            except Exception as exc:
                self._pending.pop(req_id, None)
                raise MCPTransportError(f"stdio call failed: {exc}")

    async def initialize(self, timeout: float) -> dict:
        return await self._call(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "lucy_edge", "version": "0.1.0"},
            },
            timeout,
        )

    async def list_tools(self, timeout: float) -> list[dict]:
        result = await self._call("tools/list", {}, timeout)
        return result.get("tools", [])

    async def call_tool(self, name: str, args: dict, timeout: float) -> dict:
        return await self._call("tools/call", {"name": name, "arguments": args}, timeout)

    async def close(self) -> None:
        self._closed = True
        if self._read_task:
            self._read_task.cancel()
        if self._stderr_task:
            self._stderr_task.cancel()
        if self._process:
            try:
                if self._process.stdin:
                    self._process.stdin.close()
            except Exception:
                pass
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                try:
                    self._process.kill()
                    await asyncio.wait_for(self._process.wait(), timeout=2.0)
                except Exception:
                    pass
            except Exception:
                pass
            self._process = None
        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()


# --------------------------------------------------------------------------- #
# Streamable HTTP (direct JSON + SSE)
# --------------------------------------------------------------------------- #


class HttpTransport(MCPTransport):
    """Streamable HTTP transport for MCP.

    POSTs JSON-RPC requests.  Handles both direct JSON responses
    (``Content-Type: application/json``) and SSE streams
    (``Content-Type: text/event-stream``), correlating by JSON-RPC ``id``.
    """

    def __init__(self, url: str, headers: dict[str, str]) -> None:
        if not url:
            raise MCPTransportError("http transport requires a url")
        self._url = url
        self._headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **headers,
        }
        self._session: Optional[aiohttp.ClientSession] = None
        self._id_counter = 0
        self._lock = asyncio.Lock()

    async def _ensure_session(self) -> None:
        if self._session is None:
            self._session = aiohttp.ClientSession(headers=self._headers)

    async def _call(self, method: str, params: dict, timeout: float) -> dict:
        await self._ensure_session()
        async with self._lock:
            self._id_counter += 1
            req_id = self._id_counter
            request = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
            data = json.dumps(request)
            try:
                async with self._session.post(
                    self._url,
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    if resp.status >= 400:
                        raise MCPTransportError(f"HTTP {resp.status}")
                    content_type = resp.headers.get("Content-Type", "")
                    if "text/event-stream" in content_type:
                        return await self._read_sse_response(req_id, resp, timeout)
                    body = await resp.json()
                    if "error" in body:
                        raise MCPTransportError(f"JSON-RPC error: {body['error']}")
                    return body.get("result", {})
            except aiohttp.ClientError as exc:
                raise MCPTransportError(f"HTTP error: {exc}")
            except asyncio.TimeoutError:
                raise MCPTransportError(
                    f"HTTP call to '{method}' timed out after {timeout}s"
                )

    async def _read_sse_response(
        self, req_id: int, resp: aiohttp.ClientResponse, timeout: float
    ) -> dict:
        """Parse an SSE stream until the JSON-RPC response with matching id arrives."""
        deadline = asyncio.get_event_loop().time() + timeout
        event_name: Optional[str] = None
        event_data_parts: list[str] = []
        try:
            async for raw in resp.content.iter_any():
                text = raw.decode("utf-8", errors="replace")
                for line in text.split("\n"):
                    line = line.rstrip("\r")
                    if line == "":
                        if event_data_parts:
                            data_str = "\n".join(event_data_parts)
                            if event_name in (None, "message"):
                                try:
                                    msg = json.loads(data_str)
                                    if msg.get("id") == req_id:
                                        if "error" in msg:
                                            raise MCPTransportError(
                                                f"JSON-RPC error: {msg['error']}"
                                            )
                                        return msg.get("result", {})
                                except json.JSONDecodeError:
                                    pass
                        event_name = None
                        event_data_parts = []
                        continue
                    if line.startswith("event:"):
                        event_name = line[6:].strip()
                    elif line.startswith("data:"):
                        event_data_parts.append(line[5:].strip())
                    elif line.startswith(":"):
                        continue
                    else:
                        event_data_parts.append(line)
                if asyncio.get_event_loop().time() > deadline:
                    raise MCPTransportError("SSE read timed out")
        except aiohttp.ClientError as exc:
            raise MCPTransportError(f"SSE read error: {exc}")
        raise MCPTransportError(
            f"no JSON-RPC response for id={req_id} in SSE stream"
        )

    async def initialize(self, timeout: float) -> dict:
        return await self._call(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "lucy_edge", "version": "0.1.0"},
            },
            timeout,
        )

    async def list_tools(self, timeout: float) -> list[dict]:
        result = await self._call("tools/list", {}, timeout)
        return result.get("tools", [])

    async def call_tool(self, name: str, args: dict, timeout: float) -> dict:
        return await self._call("tools/call", {"name": name, "arguments": args}, timeout)

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None
