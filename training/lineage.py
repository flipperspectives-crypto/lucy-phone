"""Training-run lineage ledger (stdlib sqlite3 only, no external deps).

Every from-scratch training run is recorded here so a checkpoint can always be
traced back to: the git commit it was built from, the exact data manifest
(provenance records), the hyperparameters, and its final status/loss. This is
the audit trail required by training/CHARTER.md (principle 4: interpretability
& monitoring).
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


STATUS_RUNNING = "RUNNING"
STATUS_DONE = "DONE"
STATUS_FAILED = "FAILED"


@dataclass
class TrainingRun:
    run_id: str
    started_at: float
    git_hash: str
    data_manifest_sha256: str
    hyperparams: dict
    seed: int
    checkpoint_path: str
    status: str = STATUS_RUNNING
    final_loss: Optional[float] = None
    steps: int = 0
    finished_at: Optional[float] = None
    note: str = ""

    def to_row(self):
        return (
            self.run_id,
            self.started_at,
            self.git_hash,
            self.data_manifest_sha256,
            json.dumps(self.hyperparams),
            self.seed,
            self.checkpoint_path,
            self.status,
            self.final_loss,
            self.steps,
            self.finished_at,
            self.note,
        )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS training_runs (
    run_id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    git_hash TEXT NOT NULL,
    data_manifest_sha256 TEXT NOT NULL,
    hyperparams TEXT NOT NULL,
    seed INTEGER NOT NULL,
    checkpoint_path TEXT NOT NULL,
    status TEXT NOT NULL,
    final_loss REAL,
    steps INTEGER NOT NULL,
    finished_at REAL,
    note TEXT NOT NULL
);
"""


def _current_git_hash(repo_root: str | Path = ".") -> str:
    """Best-effort git hash; falls back to 'UNKNOWN' if not a git repo."""
    try:
        import subprocess

        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "UNKNOWN"


class LineageLedger:
    def __init__(self, db_path: str | Path = "training/lineage.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute(_SCHEMA)
        self.conn.commit()

    def start_run(
        self,
        git_hash: str,
        data_manifest_sha256: str,
        hyperparams: dict,
        seed: int,
        checkpoint_path: str,
        run_id: Optional[str] = None,
        repo_root: str | Path = ".",
    ) -> TrainingRun:
        run = TrainingRun(
            run_id=run_id or uuid.uuid4().hex[:16],
            started_at=time.time(),
            git_hash=git_hash or _current_git_hash(repo_root),
            data_manifest_sha256=data_manifest_sha256,
            hyperparams=dict(hyperparams),
            seed=seed,
            checkpoint_path=str(checkpoint_path),
        )
        self.conn.execute(
            "INSERT INTO training_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            run.to_row(),
        )
        self.conn.commit()
        return run

    def update(self, run_id: str, steps: int, final_loss: Optional[float] = None, note: str = ""):
        cur = self.conn.execute(
            "SELECT status FROM training_runs WHERE run_id=?", (run_id,)
        )
        row = cur.fetchone()
        if row is None:
            raise KeyError(f"no run {run_id}")
        sets = "steps=?, final_loss=COALESCE(?, final_loss), note=?"
        params = [steps, final_loss, note, run_id]
        self.conn.execute(f"UPDATE training_runs SET {sets} WHERE run_id=?", params)
        self.conn.commit()

    def finish_run(self, run_id: str, status: str, final_loss: Optional[float] = None, note: str = ""):
        self.conn.execute(
            "UPDATE training_runs SET status=?, final_loss=COALESCE(?, final_loss), "
            "finished_at=?, note=? WHERE run_id=?",
            (status, final_loss, time.time(), note, run_id),
        )
        self.conn.commit()

    def get(self, run_id: str) -> Optional[TrainingRun]:
        cur = self.conn.execute(
            "SELECT * FROM training_runs WHERE run_id=?", (run_id,)
        )
        row = cur.fetchone()
        return _row_to_run(row) if row else None

    def all_runs(self) -> list[TrainingRun]:
        cur = self.conn.execute("SELECT * FROM training_runs ORDER BY started_at DESC")
        return [_row_to_run(r) for r in cur.fetchall()]

    def close(self):
        self.conn.close()


def _row_to_run(row) -> TrainingRun:
    return TrainingRun(
        run_id=row[0],
        started_at=row[1],
        git_hash=row[2],
        data_manifest_sha256=row[3],
        hyperparams=json.loads(row[4]),
        seed=row[5],
        checkpoint_path=row[6],
        status=row[7],
        final_loss=row[8],
        steps=row[9],
        finished_at=row[10],
        note=row[11],
    )
