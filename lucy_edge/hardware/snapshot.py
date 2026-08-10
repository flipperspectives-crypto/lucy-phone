"""Hardware snapshot schema.

Missing telemetry MUST be None (or UNKNOWN), never 0.  A snapshot with no
telemetry source reports ``telemetry_available=False``.
"""

from __future__ import annotations

import time
from typing import Optional

from pydantic import BaseModel, Field

from ..routing.hosts import HostRole


class HardwareSnapshot(BaseModel):
    timestamp: Optional[float] = None
    host_id: str = "unknown"
    host_role: HostRole = HostRole.UNKNOWN
    cpu_percent: Optional[float] = None
    cpu_temperature_c: Optional[float] = None
    ram_total_bytes: Optional[int] = None
    ram_available_bytes: Optional[int] = None
    gpu_name: Optional[str] = None
    gpu_utilization: Optional[float] = None
    vram_total_bytes: Optional[int] = None
    vram_used_bytes: Optional[int] = None
    vram_free_bytes: Optional[int] = None
    gpu_temperature_c: Optional[float] = None
    battery_temperature_c: Optional[float] = None
    thermal_source: Optional[str] = None
    telemetry_available: bool = False
    errors: list[str] = Field(default_factory=list)
    simulated: bool = False

    @classmethod
    def unknown(cls, host_id: str = "unknown", host_role: HostRole = HostRole.UNKNOWN) -> "HardwareSnapshot":
        return cls(
            timestamp=time.time(),
            host_id=host_id,
            host_role=host_role,
            telemetry_available=False,
            errors=["no telemetry source available"],
        )

    def summary(self) -> dict[str, object]:
        return {
            "host_id": self.host_id,
            "host_role": self.host_role.value,
            "telemetry_available": self.telemetry_available,
            "cpu_percent": self.cpu_percent,
            "cpu_temperature_c": self.cpu_temperature_c,
            "ram_available_bytes": self.ram_available_bytes,
            "gpu_name": self.gpu_name,
            "gpu_temperature_c": self.gpu_temperature_c,
            "battery_temperature_c": self.battery_temperature_c,
            "errors": self.errors,
        }
