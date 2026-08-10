"""SQLite-backed durable memory store (aiosqlite).

Persistent across process restarts by design.  FTS5 is used when available;
otherwise searches fall back to LIKE and ``fts_available`` reports the truth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import aiosqlite

from .schema import MemoryRecord, MemoryStatus, MemoryType, compute_sha256
from .provenance import ProvenancePolicy

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT UNIQUE NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    content TEXT NOT NULL,
    source TEXT NOT NULL,
    project TEXT,
    memory_type TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.0,
    evidence_refs TEXT,
    sha256 TEXT NOT NULL,
    supersedes TEXT,
    status TEXT NOT NULL,
    provenance TEXT NOT NULL,
    metadata TEXT
)
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(content, source, project)
"""


def _record_to_row(record: MemoryRecord) -> tuple[Any, ...]:
    return (
        record.memory_id,
        record.created_at,
        record.updated_at,
        record.content,
        record.source,
        record.project,
        record.memory_type.value,
        record.confidence,
        json.dumps(record.evidence_refs),
        record.ensure_sha256(),
        record.supersedes,
        record.status.value,
        record.provenance.value,
        json.dumps(record.metadata),
    )


def _row_to_record(row: Any) -> MemoryRecord:
    return MemoryRecord(
        memory_id=row[1],
        created_at=row[2],
        updated_at=row[3],
        content=row[4],
        source=row[5],
        project=row[6],
        memory_type=MemoryType(row[7]),
        confidence=row[8],
        evidence_refs=json.loads(row[9]) if row[9] else [],
        sha256=row[10],
        supersedes=row[11],
        status=MemoryStatus(row[12]),
        provenance=row[13],
        metadata=json.loads(row[14]) if row[14] else {},
    )


class MemoryStore:
    def __init__(self, path: str) -> None:
        self.path = path
        self.fts_available: Optional[bool] = None
        self._db: Optional[aiosqlite.Connection] = None

    @property
    def db_path(self) -> str:
        return self.path

    async def open(self) -> None:
        parent = Path(self.path).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute(_SCHEMA)
        try:
            await self._db.execute(_FTS_SCHEMA)
            self.fts_available = True
        except Exception:
            self.fts_available = False
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    def _require(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("memory store is not open")
        return self._db

    async def _insert_row(self, record: MemoryRecord) -> None:
        db = self._require()
        cur = await db.execute(
            "INSERT INTO memories (memory_id, created_at, updated_at, content, source, "
            "project, memory_type, confidence, evidence_refs, sha256, supersedes, status, "
            "provenance, metadata) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            _record_to_row(record),
        )
        if self.fts_available:
            await db.execute(
                "INSERT INTO memories_fts (rowid, content, source, project) VALUES (?,?,?,?)",
                (cur.lastrowid, record.content, record.source, record.project),
            )
        await db.commit()

    async def _fts_rowid(self, memory_id: str) -> Optional[int]:
        db = self._require()
        cur = await db.execute(
            "SELECT rowid FROM memories WHERE memory_id = ?", (memory_id,)
        )
        row = await cur.fetchone()
        return row["rowid"] if row else None

    async def create(self, record: MemoryRecord, apply_policy: bool = True) -> MemoryRecord:
        record.ensure_sha256()
        # Store-level invariant: a record's durability always follows its
        # provenance, so UNVERIFIED/INFERRED is never persisted as ACCEPTED.
        record.status = ProvenancePolicy.initial_status(record.provenance)
        if apply_policy:
            decision = ProvenancePolicy.admission_for(record)
            if decision.value == "ADMIT_AS_PROPOSED":
                record.status = MemoryStatus.PROPOSED
        await self._insert_row(record)
        return record

    async def get(self, memory_id: str) -> Optional[MemoryRecord]:
        db = self._require()
        cur = await db.execute(
            "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
        )
        row = await cur.fetchone()
        return _row_to_record(row) if row else None

    async def update(self, record: MemoryRecord) -> Optional[MemoryRecord]:
        db = self._require()
        record.ensure_sha256()
        record.updated_at = record.updated_at or __import__("time").time()
        await db.execute(
            "UPDATE memories SET content=?, updated_at=?, source=?, project=?, "
            "memory_type=?, confidence=?, evidence_refs=?, sha256=?, supersedes=?, "
            "status=?, provenance=?, metadata=? WHERE memory_id=?",
            (
                record.content,
                record.updated_at,
                record.source,
                record.project,
                record.memory_type.value,
                record.confidence,
                json.dumps(record.evidence_refs),
                record.sha256,
                record.supersedes,
                record.status.value,
                record.provenance.value,
                json.dumps(record.metadata),
                record.memory_id,
            ),
        )
        if self.fts_available:
            rowid = await self._fts_rowid(record.memory_id)
            if rowid is not None:
                await db.execute("DELETE FROM memories_fts WHERE rowid = ?", (rowid,))
                await db.execute(
                    "INSERT INTO memories_fts (rowid, content, source, project) VALUES (?,?,?,?)",
                    (rowid, record.content, record.source, record.project),
                )
        await db.commit()
        return record

    async def set_status(self, memory_id: str, status: MemoryStatus) -> Optional[MemoryRecord]:
        record = await self.get(memory_id)
        if record is None:
            return None
        record.status = status
        record.updated_at = __import__("time").time()
        return await self.update(record)

    async def supersede(self, memory_id: str, replacement: MemoryRecord) -> tuple[MemoryRecord, MemoryRecord]:
        """Mark ``memory_id`` SUPERSEDED and insert ``replacement`` (which
        references it via ``supersedes``).  Historical evidence is preserved."""
        old = await self.get(memory_id)
        if old is None:
            raise KeyError(f"memory not found: {memory_id}")
        replacement.supersedes = memory_id
        old.status = MemoryStatus.SUPERSEDED
        old.updated_at = __import__("time").time()
        await self.update(old)
        await self._insert_row(replacement)
        return old, replacement

    async def search(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 20,
        statuses: Optional[list[MemoryStatus]] = None,
    ) -> list[MemoryRecord]:
        """Search memory.  FTS5 when available; falls back to LIKE on any FTS
        query error so search never crashes on odd input."""
        db = self._require()
        like = f"%{query}%"
        params: list[Any] = []
        where: list[str] = []

        def build_sql(base: str) -> str:
            sql = base
            if where:
                sql += " AND " + " AND ".join(where)
            return sql + " ORDER BY updated_at DESC LIMIT ?"

        if memory_type is not None:
            where.append("memory_type = ?")
            params.append(memory_type.value)
        if statuses is not None:
            placeholders = ", ".join("?" * len(statuses))
            where.append(f"status IN ({placeholders})")
            params.extend(s.value for s in statuses)

        rows: Optional[list[Any]] = None
        if self.fts_available:
            try:
                fts_params = [query, *params, int(limit)]
                sql = build_sql(
                    "SELECT * FROM memories WHERE rowid IN "
                    "(SELECT rowid FROM memories_fts WHERE memories_fts MATCH ?)"
                )
                cur = await db.execute(sql, tuple(fts_params))
                rows = await cur.fetchall()
            except Exception:
                rows = None
        if rows is None:
            like_params = [like, like, *params, int(limit)]
            sql = build_sql(
                "SELECT * FROM memories WHERE (content LIKE ? OR source LIKE ?)"
            )
            cur = await db.execute(sql, tuple(like_params))
            rows = await cur.fetchall()
        return [_row_to_record(row) for row in rows]

    async def list(
        self,
        memory_type: Optional[MemoryType] = None,
        status: Optional[MemoryStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        db = self._require()
        params: list[Any] = []
        where: list[str] = []
        if memory_type is not None:
            where.append("memory_type = ?")
            params.append(memory_type.value)
        if status is not None:
            where.append("status = ?")
            params.append(status.value)
        sql = "SELECT * FROM memories"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([int(limit), int(offset)])
        cur = await db.execute(sql, tuple(params))
        rows = await cur.fetchall()
        return [_row_to_record(row) for row in rows]

    async def count(self, status: Optional[MemoryStatus] = None) -> int:
        db = self._require()
        if status is None:
            cur = await db.execute("SELECT COUNT(*) AS c FROM memories")
        else:
            cur = await db.execute(
                "SELECT COUNT(*) AS c FROM memories WHERE status = ?", (status.value,)
            )
        row = await cur.fetchone()
        return int(row["c"]) if row else 0

    async def delete(self, memory_id: str) -> bool:
        db = self._require()
        cur = await db.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
        if self.fts_available:
            rowid = await self._fts_rowid(memory_id)
            if rowid is not None:
                await db.execute(
                    "DELETE FROM memories_fts WHERE rowid = ?", (rowid,)
                )
        await db.commit()
        return cur.rowcount > 0
