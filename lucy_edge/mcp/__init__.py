"""MCP (Model Context Protocol) integration for Lucy Edge.

This module provides a modular, permissioned MCP client.  It is deliberately
NOT a full MCP SDK — it implements the minimum contract Lucy needs to discover
and invoke tools from allowlisted MCP servers, with every safety property made
explicit and testable.

Design
------
* ``MCPConfig`` declares the explicit server allowlist, per-server tool
  allowlists, and timebounds.
* ``MCPServerClient`` connects to ONE server, performs capability discovery,
  and exposes tool invocation with hard timeouts.
* ``MCPRegistry`` manages the set of configured clients, enforces the
  allowlist, and registers discovered tools into Lucy's ``ToolRegistry``.
* ``MCPAudit`` records MCP lifecycle and call events to the evidence ledger
  with secrets stripped.
* Every MCP operation is wrapped so that MCP failure NEVER propagates to the
  Lucy host: a dead, slow, or malicious server degrades to "unavailable" and
  logs an audit event, rather than raising.

Phone / safety rules
--------------------
* No server is ever connected unless it appears in the explicit allowlist.
* No tool is registered unless it passes the per-server tool allowlist.
* No MCP call bypasses the host ``PermissionPolicy`` — MCP tools are classified
  as ``mcp`` and default to ASK (operator approval) unless explicitly allowed.
* No MCP traffic triggers local model inference.
* Connection and call timeouts are enforced with ``asyncio.wait_for``.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, Field

from .base import MCPTransport, MCPTransportError
from .transport import HttpTransport, StdioTransport


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

class MCPServerConfig(BaseModel):
    """A single allowlisted MCP server.

    ``server_id`` is the stable Lucy-side identifier (used in evidence and in
    the registered tool name ``mcp.<server_id>.<tool>``).  It is NOT the server
    hostname — it is a logical name the operator chooses.
    """

    server_id: str
    # Transport: "stdio" (subprocess) or "http" (HTTP+SSE).  "stdio" is the
    # canonical MCP transport; "http" is supported for hosted servers.
    transport: str = "stdio"
    # For stdio transport:
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    # For http transport:
    url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    # Tool allowlist: an empty list means "discover, but apply global policy".
    # A non-empty list means "ONLY these tools are eligible for registration".
    allowed_tools: list[str] = Field(default_factory=list)
    denied_tools: list[str] = Field(default_factory=list)
    # Timeouts (seconds).  These are hard ceilings; MCP must be fast or refused.
    connect_timeout: float = 5.0
    call_timeout: float = 10.0
    # If False, the server is configured but not connected.
    enabled: bool = True
    # Permission class assigned to discovered tools.  Defaults to "mcp" which
    # the host PermissionPolicy treats as ASK-by-default.
    permission_class: str = "mcp"


class MCPConfig(BaseModel):
    """Top-level MCP configuration.

    ``enabled`` is the global switch.  When False, no MCP server is contacted
    and no MCP tools are registered — the registry is inert.
    """

    enabled: bool = False
    servers: list[MCPServerConfig] = Field(default_factory=list)
    # Global hard limits.
    max_servers: int = 8
    max_tools_per_server: int = 32
    # Global tool denylist (applied on top of per-server rules).
    global_denied_tools: list[str] = Field(default_factory=list)
    # If True, a server that fails at discovery time raises into the registry
    # instead of being marked unavailable.  Default False: MCP must not crash.
    fail_fast: bool = False


# --------------------------------------------------------------------------- #
# Domain objects
# --------------------------------------------------------------------------- #

@dataclass
class MCPToolDescriptor:
    """A tool discovered from an MCP server, before registration."""

    name: str
    description: str
    input_schema: dict[str, Any]
    server_id: str
    permission_class: str = "mcp"

    @property
    def qualified_name(self) -> str:
        return f"mcp.{self.server_id}.{self.name}"


@dataclass
class MCPCapabilityReport:
    """Result of capability discovery against one server."""

    server_id: str
    ok: bool
    error: Optional[str] = None
    tools: list[MCPToolDescriptor] = field(default_factory=list)
    raw_server_info: Optional[dict[str, Any]] = None
    duration_ms: float = 0.0


@dataclass
class MCPCallResult:
    """Result of invoking an MCP tool."""

    ok: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0


# --------------------------------------------------------------------------- #
# Allowlist / validation
# --------------------------------------------------------------------------- #

# Conservative identifier rules: server IDs and tool names must be short,
# single-line, shell-safe tokens.  This blocks path-traversal, injection, and
# accidental secret leakage through names.
_SERVER_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]{0,63}$")
_TOOL_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_\-\.]{0,127}$")


class MCPAllowlist:
    """Pure validation logic for MCP servers and tools.

    Separated from the client so the rules are unit-testable without any
    network or subprocess.
    """

    def __init__(self, config: MCPConfig) -> None:
        self.config = config

    def is_server_id_valid(self, server_id: str) -> bool:
        return bool(_SERVER_ID_RE.match(server_id))

    def is_tool_name_valid(self, name: str) -> bool:
        return bool(_TOOL_NAME_RE.match(name))

    def is_server_allowlisted(self, server_id: str) -> bool:
        return any(s.server_id == server_id for s in self.config.servers)

    def is_tool_allowed(self, server_id: str, tool_name: str) -> bool:
        if tool_name in self.config.global_denied_tools:
            return False
        server = self._server(server_id)
        if server is None:
            return False
        if tool_name in server.denied_tools:
            return False
        if server.allowed_tools:
            return tool_name in server.allowed_tools
        return True

    def sanitize_args(self, args: dict[str, Any]) -> dict[str, Any]:
        """Strip non-JSON-serializable or suspicious values from call args.

        MCP args must be JSON-serializable.  We coerce defensively and drop
        keys that look like secret carriers.
        """
        clean: dict[str, Any] = {}
        for k, v in (args or {}).items():
            if _looks_like_secret_key(k):
                continue
            try:
                json.dumps(v)
            except (TypeError, ValueError):
                continue
            clean[k] = v
        return clean

    def _server(self, server_id: str) -> Optional[MCPServerConfig]:
        for s in self.config.servers:
            if s.server_id == server_id:
                return s
        return None


_SECRET_KEY_SUBSTRINGS = ("token", "secret", "password", "key", "credential", "auth")


def _looks_like_secret_key(key: str) -> bool:
    k = key.lower()
    return any(sub in k for sub in _SECRET_KEY_SUBSTRINGS)


# --------------------------------------------------------------------------- #
# Transport abstraction
# --------------------------------------------------------------------------- #

class FakeMCPTransport(MCPTransport):
    """Deterministic in-memory transport for tests.

    Configure canned responses; the transport returns them in sequence without
    any network or subprocess.  This is what the test suite and phone-side
    development use — no real MCP server is required.
    """

    def __init__(
        self,
        tools: Optional[list[dict[str, Any]]] = None,
        call_handler: Optional[Any] = None,
        server_info: Optional[dict[str, Any]] = None,
        fail_initialize: Optional[str] = None,
    ) -> None:
        self._tools = tools or []
        self._call_handler = call_handler
        self._server_info = server_info or {"name": "fake-mcp", "version": "0.0.0"}
        self._fail_initialize = fail_initialize
        self.closed = False
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def initialize(self, timeout: float) -> dict[str, Any]:
        if self._fail_initialize:
            raise MCPTransportError(self._fail_initialize)
        return dict(self._server_info)

    async def list_tools(self, timeout: float) -> list[dict[str, Any]]:
        return list(self._tools)

    async def call_tool(self, name: str, args: dict[str, Any], timeout: float) -> dict[str, Any]:
        self.calls.append((name, args))
        if self._call_handler is not None:
            return self._call_handler(name, args)
        return {"content": [{"type": "text", "text": f"ok:{name}"}]}

    async def close(self) -> None:
        self.closed = True


# --------------------------------------------------------------------------- #
# Server client
# --------------------------------------------------------------------------- #

class MCPServerClient:
    """Connects to ONE MCP server, performs discovery, and invokes tools.

    The client is safe-by-construction: every public method enforces the
    timebound and converts exceptions into structured failures rather than
    raising into the caller.
    """

    def __init__(
        self,
        config: MCPServerConfig,
        transport: MCPTransport,
        allowlist: MCPAllowlist,
    ) -> None:
        self.config = config
        self.transport = transport
        self.allowlist = allowlist
        self.server_id = config.server_id
        self._initialized = False
        self._server_info: Optional[dict[str, Any]] = None

    async def discover(self) -> MCPCapabilityReport:
        t0 = time.monotonic()
        try:
            info = await asyncio.wait_for(
                self.transport.initialize(self.config.connect_timeout),
                timeout=self.config.connect_timeout,
            )
        except (asyncio.TimeoutError, MCPTransportError, Exception) as exc:
            return MCPCapabilityReport(
                server_id=self.server_id,
                ok=False,
                error=f"initialize failed: {_safe_error(exc)}",
                duration_ms=_ms(t0),
            )

        self._initialized = True
        self._server_info = info

        try:
            raw_tools = await asyncio.wait_for(
                self.transport.list_tools(self.config.connect_timeout),
                timeout=self.config.connect_timeout,
            )
        except (asyncio.TimeoutError, MCPTransportError, Exception) as exc:
            return MCPCapabilityReport(
                server_id=self.server_id,
                ok=False,
                error=f"list_tools failed: {_safe_error(exc)}",
                raw_server_info=info,
                duration_ms=_ms(t0),
            )

        tools: list[MCPToolDescriptor] = []
        for entry in raw_tools[: self.allowlist.config.max_tools_per_server]:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name", "")
            if not self.allowlist.is_tool_name_valid(name):
                continue
            if not self.allowlist.is_tool_allowed(self.server_id, name):
                continue
            tools.append(
                MCPToolDescriptor(
                    name=name,
                    description=str(entry.get("description", ""))[:1024],
                    input_schema=_coerce_schema(entry.get("inputSchema")),
                    server_id=self.server_id,
                    permission_class=self.config.permission_class,
                )
            )

        return MCPCapabilityReport(
            server_id=self.server_id,
            ok=True,
            tools=tools,
            raw_server_info=info,
            duration_ms=_ms(t0),
        )

    async def call(self, tool_name: str, args: dict[str, Any]) -> MCPCallResult:
        if not self._initialized:
            return MCPCallResult(ok=False, error="server not initialized")
        clean_args = self.allowlist.sanitize_args(args)
        t0 = time.monotonic()
        try:
            envelope = await asyncio.wait_for(
                self.transport.call_tool(tool_name, clean_args, self.config.call_timeout),
                timeout=self.config.call_timeout,
            )
        except (asyncio.TimeoutError, MCPTransportError, Exception) as exc:
            return MCPCallResult(
                ok=False,
                error=f"call failed: {_safe_error(exc)}",
                duration_ms=_ms(t0),
            )
        return MCPCallResult(
            ok=True,
            output=_extract_output(envelope),
            duration_ms=_ms(t0),
        )

    async def shutdown(self) -> None:
        try:
            await self.transport.close()
        except Exception:
            pass
        self._initialized = False


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

class MCPRegistry:
    """Manages the allowlisted set of MCP servers and their tool registrations.

    On ``open()`` it connects to each allowlisted server, runs discovery, and
    returns a capability summary.  Discovered tools are registered into Lucy's
    ``ToolRegistry`` by the caller (``build_services``) so they flow through the
    normal permission and evidence path.

    Failure isolation: a raised exception from one server never stops the others
    from being processed, and never propagates to Lucy's host run.
    """

    def __init__(
        self,
        config: MCPConfig,
        allowlist: Optional[MCPAllowlist] = None,
        audit: Optional["MCPAudit"] = None,
    ) -> None:
        self.config = config
        self.allowlist = allowlist or MCPAllowlist(config)
        self.audit = audit
        self._clients: dict[str, MCPServerClient] = {}
        self._reports: dict[str, MCPCapabilityReport] = {}
        self._open = False

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def available(self) -> bool:
        """True when MCP is enabled AND at least one server discovered tools."""
        return self.enabled and any(r.ok and r.tools for r in self._reports.values())

    def server_ids(self) -> list[str]:
        return sorted(self._clients)

    def tools_for(self, server_id: str) -> list[MCPToolDescriptor]:
        report = self._reports.get(server_id)
        return list(report.tools) if report else []

    def all_tools(self) -> list[MCPToolDescriptor]:
        out: list[MCPToolDescriptor] = []
        for sid in sorted(self._reports):
            out.extend(self._reports[sid].tools or [])
        return out

    async def open(self) -> dict[str, MCPCapabilityReport]:
        """Discover capabilities from all allowlisted servers.

        Returns a server_id -> report mapping.  Never raises.
        """
        if self._open:
            return dict(self._reports)
        if not self.config.enabled:
            self._open = True
            return {}

        reports: dict[str, MCPCapabilityReport] = {}
        for sc in self.config.servers[: self.config.max_servers]:
            if not sc.enabled:
                continue
            if not self.allowlist.is_server_id_valid(sc.server_id):
                report = MCPCapabilityReport(
                    server_id=sc.server_id, ok=False, error="invalid server_id"
                )
                client = self._client_for(sc)
            else:
                client = self._client_for(sc)
                report = await self._safe_discover_with(client, sc)
            reports[sc.server_id] = report
            self._clients[sc.server_id] = client
            if self.audit is not None:
                await self.audit.record_discovery(report)

        self._reports = reports
        self._open = True
        return dict(self._reports)

    async def call(self, server_id: str, tool_name: str, args: dict[str, Any]) -> MCPCallResult:
        if not self.config.enabled:
            return MCPCallResult(ok=False, error="MCP disabled")
        client = self._clients.get(server_id)
        if client is None:
            return MCPCallResult(ok=False, error=f"unknown MCP server: {server_id}")
        result = await client.call(tool_name, args)
        if self.audit is not None:
            await self.audit.record_call(server_id, tool_name, result)
        return result

    async def close(self) -> None:
        for client in self._clients.values():
            await client.shutdown()
        self._clients.clear()
        self._reports.clear()
        self._open = False

    async def _safe_discover_with(
        self, client: MCPServerClient, sc: MCPServerConfig
    ) -> MCPCapabilityReport:
        try:
            return await client.discover()
        except Exception as exc:
            if self.config.fail_fast:
                raise
            return MCPCapabilityReport(
                server_id=sc.server_id,
                ok=False,
                error=f"discovery crashed: {_safe_error(exc)}",
            )

    def _client_for(self, sc: MCPServerConfig) -> MCPServerClient:
        # Tests may inject a transport via sc._transport.  Otherwise build the
        # real transport from the server config (stdio or http).  If the config
        # is incomplete (empty command/url), fall back to a transport that fails
        # initialize cleanly — MCP must never crash the host over bad config.
        transport = getattr(sc, "_transport", None)
        if transport is None:
            try:
                transport = self._build_transport(sc)
            except MCPTransportError as exc:
                transport = FakeMCPTransport(fail_initialize=str(exc))
        return MCPServerClient(sc, transport, self.allowlist)

    @staticmethod
    def _build_transport(sc: MCPServerConfig) -> MCPTransport:
        if sc.transport == "http":
            return HttpTransport(sc.url, sc.headers)
        # Default: stdio subprocess transport.
        return StdioTransport(sc.command, sc.args, sc.env)


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #

class MCPAudit:
    """Records MCP lifecycle and call events to the evidence ledger.

    Secrets are never written: tool arguments are hashed, not stored; only the
    schema and outcome are persisted.
    """

    def __init__(self, evidence: Any) -> None:
        self.evidence = evidence

    async def record_discovery(self, report: MCPCapabilityReport) -> None:
        if self.evidence is None:
            return
        try:
            from ..evidence.schema import EvidenceRecord, EvidenceType

            record = EvidenceRecord(
                run_id=str(uuid.uuid4()),
                record_type=EvidenceType.MCP_EVENT,
                goal=f"MCP discovery: {report.server_id}",
                run_state="OK" if report.ok else "FAILED",
                completion_reason=report.error,
                tool_output_hashes={
                    "tool_count": str(len(report.tools)),
                },
                errors=[report.error] if report.error else [],
            )
            await self.evidence.append(record)
        except Exception:
            pass

    async def record_call(
        self, server_id: str, tool_name: str, result: MCPCallResult
    ) -> None:
        if self.evidence is None:
            return
        try:
            from ..evidence.schema import EvidenceRecord, EvidenceType

            record = EvidenceRecord(
                run_id=str(uuid.uuid4()),
                record_type=EvidenceType.MCP_CALL,
                goal=f"MCP call: {server_id}.{tool_name}",
                run_state="OK" if result.ok else "FAILED",
                completion_reason=result.error,
                latency_ms=result.duration_ms,
                tool_output_hashes={"output_hash": _hash_output(result.output)},
                errors=[result.error] if result.error else [],
            )
            await self.evidence.append(record)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Tool adapter: wraps an MCP tool as a Lucy ToolSpec-compatible function.
# --------------------------------------------------------------------------- #


def make_mcp_tool_func(
    registry: MCPRegistry, server_id: str, tool: MCPToolDescriptor
) -> Any:
    """Build an async function that routes a call to the MCP registry.

    This is the bridge between Lucy's ``ToolSpec``/``ToolRegistry`` world and
    the MCP client.  The returned function has the same signature as any other
    builtin tool: ``async def (**kw)`` where ``kw`` includes ``context`` plus
    the tool's declared arguments.
    """

    async def _mcp_tool_func(**kw: Any) -> dict[str, Any]:
        context = kw.pop("context", None)
        args = {k: v for k, v in kw.items() if not k.startswith("_")}
        result = await registry.call(server_id, tool.name, args)
        if not result.ok:
            return {"error": result.error}
        if isinstance(result.output, dict):
            return result.output
        return {"output": result.output}

    return _mcp_tool_func


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _safe_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc)[:300]}"


def _ms(t0: float) -> float:
    return round((time.monotonic() - t0) * 1000.0, 3)


def _coerce_schema(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    return {}


def _extract_output(envelope: dict[str, Any]) -> Any:
    """Pull a readable value out of a JSON-RPC ``tools/call`` result."""
    if not isinstance(envelope, dict):
        return envelope
    content = envelope.get("content")
    if isinstance(content, list) and content:
        parts: list[str] = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                parts.append(str(c.get("text", "")))
        return "\n".join(parts) if parts else envelope
    return envelope


def _hash_output(output: Any) -> str:
    import hashlib

    try:
        data = json.dumps(output, sort_keys=True, default=str)
    except (TypeError, ValueError):
        data = str(output)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]
