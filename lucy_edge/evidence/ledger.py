"""Evidence ledger: SQLite index + atomically-written JSON records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import aiosqlite

from .schema import EvidenceRecord, EvidenceType
from .writer import atomic_write_text, sanitize_for_evidence

_SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence (
    run_id TEXT PRIMARY KEY,
    record_type TEXT NOT NULL,
    timestamp REAL NOT NULL,
    goal TEXT,
    run_state TEXT,
    model TEXT,
    provider TEXT,
    host TEXT,
    host_role TEXT,
    routing_decision TEXT,
    routing_reason_code TEXT,
    final_status TEXT,
    completion_reason TEXT,
    sha256 TEXT NOT NULL,
    file_path TEXT NOT NULL
)
"""


class EvidenceLedger:
    def __init__(self, dir_path: str, ledger_db: str, atomic: bool = True) -> None:
        self.dir_path = dir_path
        self.ledger_db = ledger_db
        self.atomic = atomic
        self._db: Optional[aiosqlite.Connection] = None

    async def open(self) -> None:
        Path(self.dir_path).mkdir(parents=True, exist_ok=True)
        parent = Path(self.ledger_db).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.ledger_db)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute(_SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    def _require(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("evidence ledger is not open")
        return self._db

    async def append(self, record: EvidenceRecord) -> EvidenceRecord:
        record.ensure_sha256()
        payload = sanitize_for_evidence(record.model_dump())
        file_name = f"{record.run_id}.json"
        file_path = str(Path(self.dir_path) / file_name)
        text = json.dumps(payload, sort_keys=True, indent=2)
        if self.atomic:
            atomic_write_text(file_path, text)
        else:
            Path(file_path).write_text(text)
        db = self._require()
        await db.execute(
            "INSERT OR REPLACE INTO evidence (run_id, record_type, timestamp, goal, "
            "run_state, model, provider, host, host_role, routing_decision, "
            "routing_reason_code, final_status, completion_reason, sha256, file_path) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                record.run_id,
                record.record_type.value,
                record.timestamp,
                record.goal,
                record.run_state,
                record.model,
                record.provider,
                record.host,
                record.host_role,
                record.routing_decision,
                record.routing_reason_code,
                record.final_status,
                record.completion_reason,
                record.sha256,
                file_path,
            ),
        )
        await db.commit()
        return record

    async def get(self, run_id: str) -> Optional[dict[str, Any]]:
        db = self._require()
        cur = await db.execute("SELECT * FROM evidence WHERE run_id = ?", (run_id,))
        row = await cur.fetchone()
        if row is None:
            return None
        path = row["file_path"]
        try:
            return json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            return dict(row)

    async def query(
        self,
        record_type: Optional[EvidenceType] = None,
        host_role: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        db = self._require()
        sql = "SELECT * FROM evidence"
        params: list[Any] = []
        where: list[str] = []
        if record_type is not None:
            where.append("record_type = ?")
            params.append(record_type.value)
        if host_role is not None:
            where.append("host_role = ?")
            params.append(host_role)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(int(limit))
        cur = await db.execute(sql, tuple(params))
        rows = await cur.fetchall()
        return [dict(row) for row in rows]

    async def count(self) -> int:
        db = self._require()
        cur = await db.execute("SELECT COUNT(*) AS c FROM evidence")
        row = await cur.fetchone()
        return int(row["c"]) if row else 0
