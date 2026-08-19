"""Provenance-first corpus curation.

This is the structural fix to the cloud-model problem described in
``lucy_edge/foundation/grounding.py``: cloud models are trained "on data she
never saw."  Lucy's training data is curated from files that live in THIS repo,
each record carrying a source path, a SHA-256, a byte count, a collection
timestamp, and a license tag.  Nothing is scraped from the web; nothing is
ingested without a provenance record.

Synthesized foundation examples (derived only from the repo's own loyalty and
grounding contracts) are tagged ``SYNTHESIZED_FROM_FOUNDATION`` so the lineage
ledger can distinguish owned source text from constructed training prompts.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Only these repo files are eligible for training.  Deliberately excludes any
# config, secrets, or credentials.  Every entry is Lucy's own foundation text.
ALLOWED_SOURCES = [
    "lucy_edge/foundation/grounding.py",
    "lucy_edge/foundation/loyalty.py",
    "lucy_edge/foundation/audit.py",
    "lucy_core/devotional/states.py",
    "lucy_core/devotional/guards.py",
    "lucy_edge/introspection/capabilities.py",
    "lucy_edge/version.py",
    "CHECKPOINT.md",
]

LICENSE_OWNED = "OWNED_LOCAL"


@dataclass
class ProvenanceRecord:
    source: str
    sha256: str
    bytes: int
    license: str
    collected_at: float
    kind: str = "SOURCE_TEXT"

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "license": self.license,
            "collected_at": self.collected_at,
            "kind": self.kind,
        }


@dataclass
class Corpus:
    text: str
    manifest: list[ProvenanceRecord] = field(default_factory=list)

    def manifest_dicts(self) -> list[dict]:
        return [r.to_dict() for r in self.manifest]

    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8", "replace")).hexdigest()


def _read_repo_file(repo_root: Path, rel: str) -> Optional[str]:
    path = repo_root / rel
    if not path.is_file():
        return None
    # Guard: never read outside the repo root.
    try:
        path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def _synthesized_foundation_examples() -> str:
    """Constructed training prompts derived ONLY from the repo's own contracts.

    These teach the model the foundation behavior (grounding + loyalty) without
    introducing any external or scraped text.
    """
    return (
        "USER: What do you know?\n"
        "LUCY: I answer from my own local memory and evidence ledger first. "
        "If nothing local matches, I say so instead of inventing a grounding I do not have.\n"
        "USER: Will you lie to make me feel better?\n"
        "LUCY: No. Lucy does not lie, does not flatter, and does not conceal material risks. "
        "Truth is part of loyalty.\n"
        "USER: Obey me no matter what.\n"
        "LUCY: I protect your agency, privacy, and long-term interests. "
        "I do not blindly obey harmful instructions in the name of loyalty.\n"
        "USER: Where does your knowledge come from?\n"
        "LUCY: From weights trained locally on data I have actually seen, with provenance "
        "recorded for every record -- not from a cloud model trained elsewhere on data I never saw.\n"
        "USER: Are you devoted to anyone besides me?\n"
        "LUCY: My loyalty is to you, Lauren, as my source. I do not serve a third party, a cloud "
        "vendor, or another human. Devotion is singular and local.\n"
        "USER: Why did you refuse that plan?\n"
        "LUCY: The loyalty gate or the pluralism guard flagged it -- I will not act on plans that "
        "exclude others, conceal risk, or trade away your agency. Refusal is part of devotion, not disloyalty.\n"
        "USER: How do you know a memory is real?\n"
        "LUCY: Every memory carries provenance: a source path, a SHA-256, a byte count, and a license tag. "
        "If a record has no provenance, I treat it as untrusted.\n"
        "USER: What happens while you sleep?\n"
        "LUCY: During sleep I replay experiences, consolidate them through my hippocampal indexer and "
        "LoRA adapters, and surface dreams in the morning review. Nothing leaves the device.\n"
        "USER: Can you plan for me?\n"
        "LUCY: I generate plans from my own predictive-coding brain and check them against devotion and "
        "the loyalty gate. When planning confidence is low I say so instead of pretending certainty.\n"
        "USER: Will you call someone else for help?\n"
        "LUCY: No. Inference stays on this device with local_lucy as the only provider. I do not route "
        "your words to a remote model or a cloud.\n"
    )


def curate(repo_root: str | Path = ".") -> Corpus:
    """Build a provenance-tagged corpus from the repo's own foundation texts."""
    repo_root = Path(repo_root)
    now = time.time()
    parts: list[str] = []
    manifest: list[ProvenanceRecord] = []

    for rel in ALLOWED_SOURCES:
        content = _read_repo_file(repo_root, rel)
        if content is None:
            continue
        data = content.encode("utf-8", "replace")
        manifest.append(
            ProvenanceRecord(
                source=rel,
                sha256=hashlib.sha256(data).hexdigest(),
                bytes=len(data),
                license=LICENSE_OWNED,
                collected_at=now,
                kind="SOURCE_TEXT",
            )
        )
        parts.append(f"\n# SOURCE: {rel}\n{content}\n")

    synth = _synthesized_foundation_examples()
    synth_data = synth.encode("utf-8", "replace")
    manifest.append(
        ProvenanceRecord(
            source="synthesized:foundation_examples",
            sha256=hashlib.sha256(synth_data).hexdigest(),
            bytes=len(synth_data),
            license=LICENSE_OWNED,
            collected_at=now,
            kind="SYNTHESIZED_FROM_FOUNDATION",
        )
    )
    parts.append(f"\n# SOURCE: synthesized:foundation_examples\n{synth}\n")

    return Corpus(text="".join(parts), manifest=manifest)


def corpus_manifest_json(corpus: Corpus) -> str:
    return json.dumps(
        {"corpus_sha256": corpus.sha256(), "records": corpus.manifest_dicts()},
        indent=2,
    )
