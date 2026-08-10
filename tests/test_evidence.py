"""Evidence ledger, hashing, and telemetry-unknown tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from lucy_edge.evidence.ledger import EvidenceLedger
from lucy_edge.evidence.schema import EvidenceRecord, EvidenceType, new_run_id
from lucy_edge.evidence.writer import atomic_write_text, sanitize_for_evidence
from lucy_edge.hardware.snapshot import HardwareSnapshot
from lucy_edge.routing.hosts import HostRole
from lucy_edge.routing.policy import ReasonCode, RoutingDecision, RoutingRequest

from .helpers import make_config, temp_dir


class EvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_evidence_run_created(self):
        tmp = temp_dir()
        ledger = EvidenceLedger(f"{tmp}/evidence", f"{tmp}/evidence.db")
        await ledger.open()
        try:
            record = EvidenceRecord(
                record_type=EvidenceType.AGENT_RUN,
                goal="test evidence",
                final_status="COMPLETED",
                host_role="PHONE",
            )
            saved = await ledger.append(record)
            self.assertEqual(await ledger.count(), 1)
            file_path = Path(tmp) / "evidence" / f"{record.run_id}.json"
            self.assertTrue(file_path.exists())
            loaded = await ledger.get(record.run_id)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["run_id"], record.run_id)
        finally:
            await ledger.close()

    async def test_evidence_hash_produced_and_verified(self):
        tmp = temp_dir()
        ledger = EvidenceLedger(f"{tmp}/evidence", f"{tmp}/evidence.db")
        await ledger.open()
        try:
            record = EvidenceRecord(
                record_type=EvidenceType.AGENT_RUN, goal="hash me"
            )
            self.assertIsNone(record.sha256)
            saved = await ledger.append(record)
            self.assertIsNotNone(saved.sha256)
            self.assertEqual(len(saved.sha256), 64)
            self.assertTrue(saved.verify_sha256())
            tampered = saved.model_copy(deep=True)
            tampered.goal = "tampered"
            self.assertFalse(tampered.verify_sha256())
        finally:
            await ledger.close()

    async def test_full_uuid_run_ids(self):
        run_id = new_run_id()
        self.assertEqual(len(run_id), 36)
        self.assertNotEqual(new_run_id(), run_id)

    async def test_routing_evidence_recorded(self):
        tmp = temp_dir()
        config = make_config(tmp)
        from lucy_edge.services import build_services

        services = build_services(config)
        await services.open()
        try:
            request = RoutingRequest(
                model="qwen3:1.7b",
                provider="ollama",
                host_role=HostRole.PHONE,
                host_id="phone-1",
            )
            result = await services.router.route(request)
            run_id = await services.record_routing(request, result)
            self.assertTrue(run_id)
            records = await services.evidence.query(
                record_type=EvidenceType.ROUTING_DECISION
            )
            self.assertGreaterEqual(len(records), 1)
            self.assertEqual(records[0]["routing_decision"], RoutingDecision.DENY.value)
        finally:
            await services.close()

    async def test_secrets_sanitized_in_evidence(self):
        sanitized = sanitize_for_evidence(
            {"authorization": "Bearer abc", "api_key": "k", "goal": "ok", "meta": {"token": "x"}}
        )
        self.assertEqual(sanitized["authorization"], "<redacted>")
        self.assertEqual(sanitized["api_key"], "<redacted>")
        self.assertEqual(sanitized["meta"]["token"], "<redacted>")
        self.assertEqual(sanitized["goal"], "ok")

    async def test_atomic_write(self):
        tmp = temp_dir()
        path = f"{tmp}/evidence_file.json"
        atomic_write_text(path, json.dumps({"a": 1}))
        self.assertEqual(json.loads(Path(path).read_text()), {"a": 1})


class HardwareUnknownTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_telemetry_remains_unknown_not_zero(self):
        snapshot = HardwareSnapshot.unknown("phone-1", HostRole.PHONE)
        self.assertFalse(snapshot.telemetry_available)
        self.assertIsNone(snapshot.cpu_percent)
        self.assertIsNone(snapshot.cpu_temperature_c)
        self.assertIsNone(snapshot.gpu_name)
        self.assertIsNone(snapshot.battery_temperature_c)
        self.assertIsNone(snapshot.vram_free_bytes)

    async def test_governor_unknown_does_not_claim_cool(self):
        from lucy_edge.hardware.governor import GovernorAction, ThermalGovernor

        snapshot = HardwareSnapshot.unknown("phone-1", HostRole.PHONE)
        result = ThermalGovernor().evaluate(snapshot)
        self.assertEqual(result.action, GovernorAction.UNKNOWN)
        self.assertFalse(result.abort_seen)
        self.assertFalse(result.cancellation_proven)

    async def test_governor_stops_on_critical(self):
        from lucy_edge.hardware.governor import GovernorAction, ThermalGovernor

        snapshot = HardwareSnapshot(
            host_id="phone-1",
            host_role=HostRole.PHONE,
            cpu_temperature_c=104.2,
            telemetry_available=True,
            thermal_source="test",
        )
        result = ThermalGovernor().evaluate(snapshot)
        self.assertEqual(result.action, GovernorAction.STOP)
        self.assertTrue(result.abort_seen)
        self.assertFalse(result.cancellation_proven)

    async def test_governor_blocks_retry_when_hot(self):
        from lucy_edge.hardware.governor import GovernorAction, ThermalGovernor

        snapshot = HardwareSnapshot(
            host_id="phone-1",
            host_role=HostRole.PHONE,
            cpu_temperature_c=84.1,
            telemetry_available=True,
            thermal_source="test",
        )
        result = ThermalGovernor().before_retry(snapshot)
        self.assertEqual(result.action, GovernorAction.STOP)
        self.assertTrue(any("retry blocked" in d for d in result.details))


if __name__ == "__main__":
    unittest.main()
