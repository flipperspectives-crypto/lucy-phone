"""Remote host registration / handshake schema.

A host is never marked REGISTERED or ONLINE merely because it appears in
configuration.  Presence in config yields UNKNOWN until a real heartbeat or
registration call is received.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class HostStatus(str, Enum):
    UNKNOWN = "UNKNOWN"
    OFFLINE = "OFFLINE"
    UNHEALTHY = "UNHEALTHY"
    REGISTERED = "REGISTERED"


class HostRole(str, Enum):
    PHONE = "PHONE"
    LAPTOP = "LAPTOP"
    RTX4060 = "RTX4060"
    SERVER = "SERVER"
    UNKNOWN = "UNKNOWN"


class HostState(BaseModel):
    host_id: str
    hostname: str = ""
    platform: Optional[str] = None
    role: HostRole = HostRole.UNKNOWN
    status: HostStatus = HostStatus.UNKNOWN
    cpu_name: Optional[str] = None
    cpu_cores: Optional[int] = None
    ram_total_bytes: Optional[int] = None
    ram_free_bytes: Optional[int] = None
    gpu_name: Optional[str] = None
    vram_total_bytes: Optional[int] = None
    provider: Optional[str] = None
    provider_version: Optional[str] = None
    models: list[str] = Field(default_factory=list)
    thermal_telemetry: bool = False
    resource_telemetry: bool = False
    last_heartbeat: Optional[float] = None
    base_url: Optional[str] = None
    registered_at: Optional[float] = None
    errors: list[str] = Field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        return self.status == HostStatus.REGISTERED and not self.errors


class HostRegistry:
    """In-memory registry of remote inference hosts.

    Persistence across process restarts is intentionally deferred (phase 1 is
    phone-side; the laptop will register itself against the gateway later).
    """

    def __init__(self, known: Optional[list[dict[str, Any]]] = None) -> None:
        self._hosts: dict[str, HostState] = {}
        for item in known or []:
            self._ingest(HostState.model_validate(item))

    def _ingest(self, host: HostState) -> None:
        existing = self._hosts.get(host.host_id)
        if existing is None or host.status != HostStatus.UNKNOWN:
            self._hosts[host.host_id] = host
        else:
            self._hosts[host.host_id] = existing

    def register(self, host: HostState) -> HostState:
        now = time.time()
        host.status = HostStatus.REGISTERED
        host.registered_at = host.registered_at or now
        host.last_heartbeat = now
        host.errors = []
        self._hosts[host.host_id] = host
        return host

    def heartbeat(self, host_id: str, telemetry: Optional[dict[str, Any]] = None) -> Optional[HostState]:
        host = self._hosts.get(host_id)
        if host is None:
            return None
        host.last_heartbeat = time.time()
        if telemetry:
            if "ram_free_bytes" in telemetry:
                host.ram_free_bytes = telemetry["ram_free_bytes"]
            if "gpu_name" in telemetry:
                host.gpu_name = telemetry["gpu_name"]
        return host

    def mark_offline(self, host_id: str) -> Optional[HostState]:
        host = self._hosts.get(host_id)
        if host is not None:
            host.status = HostStatus.OFFLINE
        return host

    def mark_unhealthy(self, host_id: str, error: str) -> Optional[HostState]:
        host = self._hosts.get(host_id)
        if host is not None:
            host.status = HostStatus.UNHEALTHY
            if error not in host.errors:
                host.errors.append(error)
        return host

    def get(self, host_id: str) -> Optional[HostState]:
        return self._hosts.get(host_id)

    def list(self) -> list[HostState]:
        return sorted(self._hosts.values(), key=lambda h: h.host_id)

    def usable(self) -> list[HostState]:
        return [h for h in self.list() if h.is_usable]

    def summary(self) -> list[dict[str, Any]]:
        return [self.to_dict(h) for h in self.list()]

    @staticmethod
    def to_dict(host: HostState) -> dict[str, Any]:
        data = host.model_dump()
        data["last_heartbeat_iso"] = (
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(host.last_heartbeat))
            if host.last_heartbeat
            else None
        )
        return data
