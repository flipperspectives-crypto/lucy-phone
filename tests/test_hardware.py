"""Hardware sensor parsing (fixture-driven, no live hardware dependency)."""

from __future__ import annotations

import unittest
from pathlib import Path

from lucy_edge.hardware.governor import GovernorAction, ThermalGovernor
from lucy_edge.hardware.sensors import (
    build_telemetry,
    parse_cpu_deltas,
    parse_meminfo,
    parse_proc_stat,
    parse_thermal_zone,
)
from lucy_edge.hardware.snapshot import HardwareSnapshot
from lucy_edge.routing.hosts import HostRole

from .helpers import temp_dir


class MeminfoTests(unittest.TestCase):
    def test_parse_meminfo(self):
        text = "MemTotal:       16000000 kB\nMemFree:         500000 kB\nMemAvailable:   4000000 kB\n"
        result = parse_meminfo(text)
        self.assertEqual(result["total_bytes"], 16_000_000 * 1024)
        self.assertEqual(result["available_bytes"], 4_000_000 * 1024)


class ProcStatTests(unittest.TestCase):
    def test_parse_and_delta(self):
        prev_text = "cpu  1000 0 1000 9000 0 0 0 0 0 0\n"
        curr_text = "cpu  2000 0 2000 12000 0 0 0 0 0 0\n"
        prev = parse_proc_stat(prev_text)
        curr = parse_proc_stat(curr_text)
        self.assertEqual(prev["idle"], 9000)
        pct = parse_cpu_deltas(prev, curr)
        # busy grew 2000, idle grew 3000, total grew 5000 -> 40%
        self.assertEqual(pct, 40.0)

    def test_parse_cpu_deltas_unknown_when_no_change(self):
        prev = parse_proc_stat("cpu  100 0 100 1000\n")
        curr = parse_proc_stat("cpu  100 0 100 1000\n")
        self.assertEqual(parse_cpu_deltas(prev, curr), 0.0)


class ThermalZoneTests(unittest.TestCase):
    def test_parse_thermal_zone_file(self):
        tmp = temp_dir()
        zone = Path(tmp, "temp")
        zone.write_text("45200")
        self.assertEqual(parse_thermal_zone(str(zone)), 45.2)

    def test_parse_missing_returns_none(self):
        self.assertIsNone(parse_thermal_zone("/nonexistent/zone/temp"))


class TelemetryBuildTests(unittest.TestCase):
    def test_unknown_platform_uses_null_telemetry(self):
        telemetry = build_telemetry("h", HostRole.UNKNOWN, platform="plan9")
        snapshot = telemetry.snapshot()
        self.assertFalse(snapshot.telemetry_available)
        self.assertEqual(snapshot.errors, ["no telemetry source available"])

    def test_windows_stub_reports_unknown_honestly(self):
        telemetry = build_telemetry("h", HostRole.LAPTOP, platform="win32")
        snapshot = telemetry.snapshot()
        self.assertFalse(snapshot.telemetry_available)
        self.assertIn("not implemented", snapshot.errors[0])


class GovernorTests(unittest.TestCase):
    def _snap(self, temp: float):
        return HardwareSnapshot(
            host_id="phone-1",
            host_role=HostRole.PHONE,
            cpu_temperature_c=temp,
            telemetry_available=True,
            thermal_source="test",
        )

    def test_normal_below_warn(self):
        self.assertEqual(
            ThermalGovernor().evaluate(self._snap(50.0)).action, GovernorAction.NORMAL
        )

    def test_throttle_at_warn(self):
        self.assertEqual(
            ThermalGovernor().evaluate(self._snap(70.0)).action, GovernorAction.THROTTLE
        )

    def test_stop_at_critical(self):
        self.assertEqual(
            ThermalGovernor().evaluate(self._snap(80.0)).action, GovernorAction.STOP
        )


if __name__ == "__main__":
    unittest.main()
