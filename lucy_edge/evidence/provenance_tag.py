"""In-house provenance + SHA-256 tagging for logs and transcripts.

Stdlib only (hashlib, json, time). Mirrors the provenance record shape emitted by
``tiny_infer.py --emit-provenance`` so every artifact in the ecosystem carries the
same tagging contract: a ``source`` category plus SHA-256 digests of the inputs
and outputs.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Input/query field names that may appear on a record, in priority order.
_PROMPT_FIELDS = ("experiential_query", "puzzle", "prompt", "induction", "query")
_RESPONSE_FIELDS = ("response", "output", "answer")


def _pick(rec: Dict[str, Any], fields: tuple) -> str:
    for f in fields:
        if f in rec and rec[f] is not None:
            return str(rec[f])
    return ""


def tag_record(rec: Dict[str, Any], source: str = "OBSERVED") -> Dict[str, Any]:
    """Return a copy of ``rec`` with provenance + SHA-256 fields attached."""
    out = dict(rec)
    out["source"] = source
    out["prompt_sha256"] = sha256_text(_pick(rec, _PROMPT_FIELDS))
    out["response_sha256"] = sha256_text(_pick(rec, _RESPONSE_FIELDS))
    out["record_sha256"] = sha256_json({k: rec[k] for k in rec})
    return out


def build_envelope(
    *,
    experiment: str,
    model: str,
    provider: Optional[str] = None,
    condition: Optional[str] = None,
    tagged_records: Optional[List[Dict[str, Any]]] = None,
    source: str = "OBSERVED",
) -> Dict[str, Any]:
    """File-level provenance envelope mirroring tiny_infer's emit-provenance record."""
    records = tagged_records or []
    return {
        "source": source,
        "inference": "srp_experiment",
        "experiment": experiment,
        "model": model,
        "provider": provider,
        "condition": condition,
        "model_config": {"model": model, "provider": provider},
        "record_count": len(records),
        "transcript_sha256": sha256_json(records),
        "schema_version": "prov-1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
