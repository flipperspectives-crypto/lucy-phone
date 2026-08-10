"""MCP transport abstractions and shared types.

This module exists to break the circular import between ``mcp/__init__.py``
(which defines the registry) and ``mcp/transport.py`` (which defines the
concrete transports).  Both import the abstract base class and error type
from here.
"""

from __future__ import annotations

import abc
from typing import Any


class MCPTransport(abc.ABC):
    """Minimal MCP transport.

    Implementations handle one JSON-RPC request/response exchange.  They are
    deliberately tiny: Lucy needs ``initialize``, ``tools/list``, and
    ``tools/call`` — not the full specification.
    """

    @abc.abstractmethod
    async def initialize(self, timeout: float) -> dict[str, Any]:
        """Perform the MCP ``initialize`` handshake and return server info."""

    @abc.abstractmethod
    async def list_tools(self, timeout: float) -> list[dict[str, Any]]:
        """Return the server's ``tools/list`` response."""

    @abc.abstractmethod
    async def call_tool(self, name: str, args: dict[str, Any], timeout: float) -> dict[str, Any]:
        """Invoke ``tools.call`` and return the raw result envelope."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Release any resources (process handles, connections)."""


class MCPTransportError(Exception):
    """Transport-level failure (connection, timeout, protocol)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
