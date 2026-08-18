"""From-scratch training driver for Lucy's local model.

Stdlib-only. Curates a provenance-tagged corpus from the repo's own foundation
texts, trains a TinyTransformer with plain SGD, saves a checkpoint, and records
the run in the lineage ledger. No network, no torch/numpy.

Run as a script:  python -m training.train
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Optional

from .corpus import curate
from .lineage import LineageLedger, STATUS_DONE, STATUS_FAILED
from .tiny_transformer import TinyTransformer
from .tokenizer import ByteTokenizer


def _window_corpus(token_ids: list[int], ctx: int, stride: int) -> list[list[int]]:
    """Slide a context window over the token stream to build training sequences."""
    seqs = []
    for start in range(0, max(1, len(token_ids) - ctx + 1), stride):
        seq = token_ids[start : start + ctx]
        if len(seq) == ctx:
            seqs.append(seq)
    return seqs


def train(
    repo_root: str | Path = ".",
    checkpoint_dir: str | Path = "training/checkpoints",
    steps: int = 200,
    lr: float = 0.05,
    ctx: int = 32,
    d_model: int = 32,
    n_layers: int = 1,
    ff_mult: int = 4,
    seed: int = 1,
    batch_size: int = 4,
    stride: int = 8,
    lineage_db: str | Path = "training/lineage.db",
    git_hash: Optional[str] = None,
) -> dict:
    """Train a tiny model from scratch and persist a checkpoint + lineage entry.

    Returns a summary dict (also recorded in the lineage ledger).
    """
    tok = ByteTokenizer()
    corpus = curate(repo_root)
    token_ids = tok.encode(corpus.text.encode("utf-8", "replace"))
    if not token_ids:
        raise RuntimeError("corpus is empty; nothing to train on")

    seqs = _window_corpus(token_ids, ctx, stride)
    if not seqs:
        # corpus shorter than ctx: pad with a single short sequence
        seqs = [token_ids[:ctx] if len(token_ids) >= ctx else token_ids + [0] * (ctx - len(token_ids))]

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ledger = LineageLedger(lineage_db)
    run = ledger.start_run(
        git_hash=git_hash or "",
        data_manifest_sha256=corpus.sha256(),
        hyperparams={
            "lr": lr,
            "ctx": ctx,
            "d_model": d_model,
            "n_layers": n_layers,
            "ff_mult": ff_mult,
            "batch_size": batch_size,
            "stride": stride,
            "steps": steps,
        },
        seed=seed,
        checkpoint_path=str(ckpt_dir / "latest.json"),
        repo_root=repo_root,
    )

    m = TinyTransformer(
        vocab=tok.vocab_size, d_model=d_model, ctx=ctx, n_layers=n_layers, ff_mult=ff_mult, seed=seed
    )
    rng = random.Random(seed)
    losses = []
    try:
        for step in range(steps):
            batch = [rng.choice(seqs) for _ in range(batch_size)]
            targets = [b[1:] + [0] for b in batch]
            logits, cache = m.forward(batch)
            loss, dlogits = m.cross_entropy(logits, targets)
            losses.append(loss)
            m.backward(batch, logits, cache, dlogits)
            for name, g in m.grad.items():
                p = m.params[name]
                if isinstance(g[0], list):  # 2D
                    for i in range(len(g)):
                        row = p[i]
                        grow = g[i]
                        for j in range(len(row)):
                            row[j] -= lr * grow[j]
                else:  # 1D (layernorm gain/bias)
                    for i in range(len(g)):
                        p[i] -= lr * g[i]
            if (step + 1) % max(1, steps // 10) == 0:
                ledger.update(run.run_id, steps=step + 1, final_loss=loss)
        final_loss = losses[-1]
        # persist checkpoint (state_dict + training metadata for standalone inference)
        sd = m.state_dict()
        sd["trained_steps"] = steps
        sd["final_loss"] = final_loss
        ckpt_path = ckpt_dir / f"lucy_{run.run_id}.json"
        ckpt_path.write_text(json.dumps(sd))
        latest = ckpt_dir / "latest.json"
        latest.write_text(json.dumps(sd))
        ledger.finish_run(run.run_id, STATUS_DONE, final_loss=final_loss, note="ok")
        ledger.close()
        return {
            "run_id": run.run_id,
            "checkpoint": str(ckpt_path),
            "latest": str(latest),
            "final_loss": final_loss,
            "steps": steps,
            "git_hash": run.git_hash,
            "data_manifest_sha256": corpus.sha256(),
        }
    except Exception as e:  # pragma: no cover - defensive
        ledger.finish_run(run.run_id, STATUS_FAILED, note=str(e))
        ledger.close()
        raise


if __name__ == "__main__":
    import subprocess

    try:
        ghash = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        ghash = ""
    summary = train(git_hash=ghash)
    print(json.dumps(summary, indent=2))
