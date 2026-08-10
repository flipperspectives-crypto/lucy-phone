"""Persistent memory, provenance and admission tests."""

from __future__ import annotations

import unittest

from lucy_edge.memory.provenance import ProvenancePolicy
from lucy_edge.memory.schema import (
    MemoryRecord,
    MemoryStatus,
    MemoryType,
    ProvenanceCategory,
)
from lucy_edge.memory.store import MemoryStore

from .helpers import temp_dir


class MemoryStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_read_search(self):
        store = MemoryStore(f"{temp_dir()}/memory.db")
        await store.open()
        try:
            record = MemoryRecord(
                content="the nexus gateway port is 8970",
                source="operator",
                memory_type=MemoryType.SEMANTIC,
                provenance=ProvenanceCategory.USER_STATED,
            )
            created = await store.create(record)
            fetched = await store.get(created.memory_id)
            self.assertIsNotNone(fetched)
            self.assertEqual(fetched.content, record.content)
            hits = await store.search("gateway port")
            self.assertTrue(any(h.memory_id == created.memory_id for h in hits))
        finally:
            await store.close()

    async def test_memory_survives_process_restart(self):
        path = f"{temp_dir()}/memory.db"
        store = MemoryStore(path)
        await store.open()
        record = MemoryRecord(
            content="persistent across restarts",
            source="runtime",
            provenance=ProvenanceCategory.KNOWN_FROM_RUNTIME,
        )
        created = await store.create(record)
        await store.close()

        reopened = MemoryStore(path)
        await reopened.open()
        try:
            fetched = await reopened.get(created.memory_id)
            self.assertIsNotNone(fetched)
            self.assertEqual(fetched.content, "persistent across restarts")
            self.assertEqual(await reopened.count(), 1)
        finally:
            await reopened.close()

    async def test_provenance_survives_restart(self):
        path = f"{temp_dir()}/memory.db"
        store = MemoryStore(path)
        await store.open()
        record = MemoryRecord(
            content="maybe true?",
            provenance=ProvenanceCategory.UNVERIFIED,
        )
        created = await store.create(record, apply_policy=False)
        await store.close()

        reopened = MemoryStore(path)
        await reopened.open()
        try:
            fetched = await reopened.get(created.memory_id)
            self.assertEqual(fetched.provenance, ProvenanceCategory.UNVERIFIED)
            self.assertEqual(fetched.status, MemoryStatus.PROPOSED)
        finally:
            await reopened.close()

    async def test_supersession_works(self):
        store = MemoryStore(f"{temp_dir()}/memory.db")
        await store.open()
        try:
            old = await store.create(
                MemoryRecord(
                    content="old fact", provenance=ProvenanceCategory.OBSERVED
                )
            )
            new = MemoryRecord(
                content="corrected fact", provenance=ProvenanceCategory.OBSERVED
            )
            superseded, replacement = await store.supersede(old.memory_id, new)
            self.assertEqual(superseded.status, MemoryStatus.SUPERSEDED)
            self.assertEqual(replacement.supersedes, old.memory_id)
            fetched_new = await store.get(replacement.memory_id)
            self.assertEqual(fetched_new.content, "corrected fact")
            fetched_old = await store.get(old.memory_id)
            self.assertEqual(fetched_old.status, MemoryStatus.SUPERSEDED)
        finally:
            await store.close()

    async def test_sha256_is_recorded(self):
        store = MemoryStore(f"{temp_dir()}/memory.db")
        await store.open()
        try:
            record = MemoryRecord(
                content="hashed memory",
                provenance=ProvenanceCategory.OBSERVED,
            )
            created = await store.create(record)
            self.assertIsNotNone(created.sha256)
            self.assertEqual(len(created.sha256), 64)
        finally:
            await store.close()

    async def test_search_never_crashes_on_odd_input(self):
        store = MemoryStore(f"{temp_dir()}/memory.db")
        await store.open()
        try:
            await store.create(
                MemoryRecord(
                    content="plain content",
                    provenance=ProvenanceCategory.OBSERVED,
                )
            )
            hits = await store.search("-not-a-valid-fts-query- (")
            self.assertIsInstance(hits, list)
        finally:
            await store.close()


class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_unverified_remains_unverified(self):
        store = MemoryStore(f"{temp_dir()}/memory.db")
        await store.open()
        try:
            from lucy_edge.memory.admission import MemoryAdmission

            admission = MemoryAdmission(store)
            record = await admission.suggest_from_model(
                "the model says X", source="model_output"
            )
            self.assertEqual(record.provenance, ProvenanceCategory.UNVERIFIED)
            self.assertEqual(record.status, MemoryStatus.PROPOSED)
            with self.assertRaises(ValueError):
                await admission.accept(record.memory_id)
            fetched = await store.get(record.memory_id)
            self.assertEqual(fetched.status, MemoryStatus.PROPOSED)
        finally:
            await store.close()

    async def test_observation_is_accepted(self):
        store = MemoryStore(f"{temp_dir()}/memory.db")
        await store.open()
        try:
            from lucy_edge.memory.admission import MemoryAdmission

            admission = MemoryAdmission(store)
            record = await admission.record_observation(
                "sensor reading verified", source="tool:system.health"
            )
            self.assertEqual(record.provenance, ProvenanceCategory.OBSERVED)
            self.assertEqual(record.status, MemoryStatus.ACCEPTED)
        finally:
            await store.close()

    async def test_inferred_never_promoted_to_observed(self):
        store = MemoryStore(f"{temp_dir()}/memory.db")
        await store.open()
        try:
            record = await store.create(
                MemoryRecord(
                    content="guessed",
                    provenance=ProvenanceCategory.INFERRED,
                    status=MemoryStatus.PROPOSED,
                ),
                apply_policy=False,
            )
            promoted = ProvenancePolicy.can_promote(
                MemoryStatus.PROPOSED,
                MemoryStatus.ACCEPTED,
                ProvenanceCategory.INFERRED,
                ProvenanceCategory.OBSERVED,
            )
            self.assertFalse(promoted)
            fetched = await store.get(record.memory_id)
            self.assertEqual(fetched.provenance, ProvenanceCategory.INFERRED)
        finally:
            await store.close()


if __name__ == "__main__":
    unittest.main()
