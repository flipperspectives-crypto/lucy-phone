"""Platform telemetry adapters.

The current phone phase implements safe Linux parsing with explicit
fixture-testable pure functions and a NullTelemetry fallback.  Windows support
is a stub that honestly returns UNKNOWN telemetry (it must not pretend to read
Windows hardware it cannot see).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from .snapshot import HardwareSnapshot
from ..routing.hosts import HostRole


def parse_meminfo(text: str) -> dict[str, Optional[int]]:
    """Parse /proc/meminfo text into {total_bytes, available_bytes, ...}."""
    out: dict[str, Optional[int]] = {"total_bytes": None, "available_bytes": None}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        rest = rest.strip()
        try:
            kb = int(rest.split()[0])
        except (ValueError, IndexError):
            continue
        if key == "MemTotal":
            out["total_bytes"] = kb * 1024
        elif key == "MemAvailable":
            out["available_bytes"] = kb * 1024
    return out


def parse_cpu_deltas(prev: dict[str, int], curr: dict[str, int]) -> Optional[float]:
    """Compute a CPU busy-percent between two /proc/stat snapshots."""
    def total(d: dict[str, int]) -> int:
        return sum(v for v in d.values() if v is not None)

    d_total = total(curr) - total(prev)
    d_idle = curr.get("idle", 0) - prev.get("idle", 0)
    if d_total <= 0:
        return 0.0
    return round(100.0 * (1.0 - d_idle / d_total), 2)


def parse_proc_stat(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in text.splitlines():
        if not line.startswith("cpu "):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            out["user"] = int(parts[1])
            out["nice"] = int(parts[2])
            out["system"] = int(parts[3])
            out["idle"] = int(parts[4])
            out["iowait"] = int(parts[5]) if len(parts) > 5 else 0
            out["irq"] = int(parts[6]) if len(parts) > 6 else 0
            out["softirq"] = int(parts[7]) if len(parts) > 7 else 0
            out["steal"] = int(parts[8]) if len(parts) > 8 else 0
        except ValueError:
            continue
        break
    return out


def parse_thermal_zone(path: str) -> Optional[float]:
    """Read a single /sys/class/thermal/thermal_zone*/temp value (milliC)."""
    try:
        raw = Path(path).read_text().strip()
        milli_c = int(raw)
        return round(milli_c / 1000.0, 1)
    except (OSError, ValueError):
        return None


class LinuxTelemetry:
    """Safe Linux adapter (Termux/proot/container).

    Every read is wrapped; anything unavailable stays None and the snapshot is
    marked telemetry_available=False.
    """

    def __init__(self, host_id: str, host_role: HostRole) -> None:
        self.host_id = host_id
        self.host_role = host_role
        self._last_stat: Optional[dict[str, int]] = None
        self._last_stat_at: Optional[float] = None

    def _cpu_percent(self) -> Optional[float]:
        try:
            curr = parse_proc_stat(Path("/proc/stat").read_text())
        except OSError:
            return None
        now = time.monotonic()
        if self._last_stat is not None and self._last_stat_at is not None:
            pct = parse_cpu_deltas(self._last_stat, curr)
        else:
            pct = None
        self._last_stat = curr
        self._last_stat_at = now
        return pct

    def _ram(self) -> dict[str, Optional[int]]:
        try:
            return parse_meminfo(Path("/proc/meminfo").read_text())
        except OSError:
            return {"total_bytes": None, "available_bytes": None}

    def _thermal(self) -> tuple[Optional[float], Optional[str]]:
        base = Path("/sys/class/thermal")
        if not base.is_dir():
            return None, None
        temps: list[float] = []
        for zone in sorted(base.glob("thermal_zone*")):
            temp = parse_thermal_zone(str(zone / "temp"))
            if temp is not None:
                temps.append(temp)
        if not temps:
            return None, None
        return round(max(temps), 1), "sysfs_thermal_zone"

    def snapshot(self) -> HardwareSnapshot:
        errors: list[str] = []
        cpu_percent = self._cpu_percent()
        ram = self._ram()
        cpu_temp, thermal_source = self._thermal()
        has_any = (
            cpu_percent is not None
            or ram["total_bytes"] is not None
            or cpu_temp is not None
        )
        if not has_any:
            errors.append("no usable telemetry sources found")
        return HardwareSnapshot(
            timestamp=time.time(),
            host_id=self.host_id,
            host_role=self.host_role,
            cpu_percent=cpu_percent,
            cpu_temperature_c=cpu_temp,
            ram_total_bytes=ram["total_bytes"],
            ram_available_bytes=ram["available_bytes"],
            thermal_source=thermal_source,
            telemetry_available=has_any,
            errors=errors,
        )


class WindowsTelemetry:
    """Windows adapter stub.

    Not implemented in the phone phase.  Honestly reports UNKNOWN telemetry so
    the laptop phase must provide real WMI/psutil-backed telemetry later.
    """

    def __init__(self, host_id: str, host_role: HostRole) -> None:
        self.host_id = host_id
        self.host_role = host_role

    def snapshot(self) -> HardwareSnapshot:
        return HardwareSnapshot(
            timestamp=time.time(),
            host_id=self.host_id,
            host_role=self.host_role,
            telemetry_available=False,
            errors=["WindowsTelemetry not implemented in phone-safe phase"],
        )


class NullTelemetry:
    def __init__(self, host_id: str, host_role: HostRole) -> None:
        self.host_id = host_id
        self.host_role = host_role

    def snapshot(self) -> HardwareSnapshot:
        return HardwareSnapshot.unknown(self.host_id, self.host_role)


def build_telemetry(
    host_id: str, host_role: HostRole, platform: Optional[str] = None
) -> LinuxTelemetry | WindowsTelemetry | NullTelemetry:
    import sys as _sys

    platform = platform or _sys.platform
    if platform.startswith("linux"):
        return LinuxTelemetry(host_id, host_role)
    if platform.startswith("win"):
        return WindowsTelemetry(host_id, host_role)
    return NullTelemetry(host_id, host_role)
