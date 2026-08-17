"""Honest training-status detection for introspection.

This module deliberately uses only the standard library (json, sqlite3, os) so
that lucy_edge does not depend on the standalone ``training`` package.  It
reports ``AVAILABLE`` ONLY when a real, loadable checkpoint file exists and
(optionally) a lineage ledger records a DONE run for it.  It never fabricates
availability.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional


def check_training(
    checkpoint_path: Optional[str],
    lineage_db: Optional[str] = None,
) -> tuple[str, dict]:
    """Return (status, provenance).

    status is "AVAILABLE" only if a valid checkpoint exists; otherwise
    "UNAVAILABLE". provenance carries the git hash / data manifest / loss when a
    matching DONE lineage run is found.
    """
    if not checkpoint_path:
        return "UNAVAILABLE", {}
    p = Path(checkpoint_path)
    if not p.exists():
        return "UNAVAILABLE", {}
    try:
        sd = json.loads(p.read_text())
    except Exception:
        return "UNAVAILABLE", {}
    if not isinstance(sd, dict) or "tok_emb" not in sd or "layers" not in sd:
        return "UNAVAILABLE", {}

    provenance: dict = {
        "checkpoint": str(p),
        "params": {
            k: sd.get(k) for k in ("vocab", "d", "ctx", "L", "ff")
        },
        "source": "training.train (from-scratch, local, provenance-tagged corpus)",
    }

    if lineage_db:
        db = Path(lineage_db)
        if db.exists():
            try:
                conn = sqlite3.connect(str(db))
                cur = conn.execute(
                    "SELECT run_id, git_hash, data_manifest_sha256, status, final_loss "
                    "FROM training_runs WHERE checkpoint_path=? ORDER BY started_at DESC LIMIT 1",
                    (str(p),),
                )
                row = cur.fetchone()
                conn.close()
                if row and row[3] == "DONE":
                    provenance["lineage_run_id"] = row[0]
                    provenance["git_hash"] = row[1]
                    provenance["data_manifest_sha256"] = row[2]
                    provenance["final_loss"] = row[4]
            except Exception:
                pass
    return "AVAILABLE", provenance
