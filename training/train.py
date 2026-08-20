"""From-scratch training driver for Lucy's local model.

Stdlib-only. Curates a provenance-tagged corpus from the repo's own foundation
texts, trains a TinyTransformer with plain SGD, saves a checkpoint, and records
the run in the lineage ledger. No network, no torch/numpy.

Run as a script:  python -m training.train
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import uuid
from pathlib import Path
from typing import Optional

from .corpus import curate
from .lineage import LineageLedger, STATUS_DONE, STATUS_FAILED
from .tiny_transformer import TinyTransformer
from .tokenizer import ByteTokenizer, BPETokenizer


def _window_corpus(token_ids: list[int], ctx: int, stride: int) -> list[list[int]]:
    """Slide a context window over the token stream to build training sequences."""
    seqs = []
    for start in range(0, max(1, len(token_ids) - ctx + 1), stride):
        seq = token_ids[start : start + ctx]
        if len(seq) == ctx:
            seqs.append(seq)
    return seqs


def _sequence_loss(m: "TinyTransformer", seq: list[int]) -> float:
    """Cross-entropy of the model on a single next-token sequence."""
    if len(seq) < 2:
        return 0.0
    logits, _ = m.forward([seq[:-1]])
    loss, _ = m.cross_entropy(logits, [seq[1:]])
    return loss


def _val_loss(m: "TinyTransformer", val_seqs: list[list[int]], n: int = 8) -> Optional[float]:
    if not val_seqs:
        return None
    sample = val_seqs[:n]
    return sum(_sequence_loss(m, s) for s in sample) / len(sample)


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
    corpus_text: Optional[str] = None,
    resume: bool = False,
) -> dict:
    """Train a tiny model and persist a checkpoint + lineage entry.

    When *resume* is True, loads the existing ``latest.json`` checkpoint and
    continues training from those weights and tokenizer state.  Otherwise a
    fresh model is initialised from random weights.

    Returns a summary dict (also recorded in the lineage ledger).
    """
    if corpus_text is not None:
        # synthetic / test corpus: train on caller-supplied text instead of the
        # repo's own foundation texts (used by the synthetic pattern-learning test)
        text = corpus_text
        data_sha = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
    else:
        corpus = curate(repo_root)
        text = corpus.text
        data_sha = corpus.sha256()
    # Learned, in-house BPE tokenizer (no external vocab, nothing fetched). Trained
    # solely on the corpus we are about to train on, so the vocabulary is fully
    # owned-local and reproducible from the checkpoint.
    latest_path = Path(checkpoint_dir) / "latest.json"
    prev_steps = 0
    if resume and latest_path.exists():
        sd = json.loads(latest_path.read_text())
        tok = BPETokenizer.from_state_dict(sd["tokenizer"])
        prev_steps = sd.get("trained_steps", 0)
        print(f"  Resuming from step {prev_steps}, loss {sd.get('final_loss', '?')}")
    else:
        tok = BPETokenizer(target_vocab=512)
        tok.train([text])
    token_ids = tok.encode(text)
    if not token_ids:
        raise RuntimeError("corpus is empty; nothing to train on")

    # Deterministic held-out split (last 10%) for an honest generalization
    # signal -- training and evaluation never share these tokens.
    split = max(1, int(len(token_ids) * 0.9))
    train_ids = token_ids[:split]
    val_ids = token_ids[split:]
    seqs = _window_corpus(train_ids, ctx, stride)
    val_seqs = _window_corpus(val_ids, ctx, stride)
    if not seqs:
        seqs = [train_ids[:ctx] if len(train_ids) >= ctx else train_ids + [0] * (ctx - len(train_ids))]
    if not val_seqs:
        val_seqs = [val_ids[:ctx] if len(val_ids) >= ctx else val_ids + [0] * (ctx - len(val_ids))]

    # Fixed integrity probe (deterministic) stored in the checkpoint so the audit
    # can later verify the saved weights actually produce this loss -- a
    # tamper-evident check that the model really was trained, not just labelled.
    probe_seq = (
        token_ids[:ctx] if len(token_ids) >= ctx else token_ids + [0] * (ctx - len(token_ids))
    )

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:16]
    per_run_path = ckpt_dir / f"lucy_{run_id}.json"
    latest_path = ckpt_dir / "latest.json"

    ledger = None
    run = None
    try:
        ledger = LineageLedger(lineage_db)
        run = ledger.start_run(
            run_id=run_id,
            git_hash=git_hash or "",
            data_manifest_sha256=data_sha,
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
            checkpoint_path=str(per_run_path),
            repo_root=repo_root,
        )

        if resume and latest_path.exists():
            sd = json.loads(latest_path.read_text())
            m = TinyTransformer(
                vocab=sd["vocab"], d_model=sd["d"], ctx=sd["ctx"],
                n_layers=sd["L"], ff_mult=sd["ff"], seed=seed,
            )
            m.load_state_dict(sd)
        else:
            m = TinyTransformer(
                vocab=tok.vocab_size, d_model=d_model, ctx=ctx,
                n_layers=n_layers, ff_mult=ff_mult, seed=seed,
            )
        rng = random.Random(seed)
        losses = []
        best_loss = float("inf")
        best_sd = None
        best_probe_loss = None
        for step in range(steps):
            batch = [rng.choice(seqs) for _ in range(batch_size)]
            # next-token targets: drop the first token as a target and the last
            # token as input, so we never pad the target with a null byte (which
            # would teach the model to emit \\x00 at sequence ends).
            batch_x = [b[:-1] for b in batch]
            targets = [b[1:] for b in batch]
            logits, cache = m.forward(batch_x)
            loss, dlogits = m.cross_entropy(logits, targets)
            if not math.isfinite(loss):
                # A non-finite loss (e.g. from an extreme LR or degenerate batch)
                # would poison the weights via backprop. Skip the update so the
                # saved checkpoint always carries finite weights; the previous
                # finite state is retained and training continues from there.
                continue
            losses.append(loss)
            if loss < best_loss:
                best_loss = loss
                best_sd = json.loads(json.dumps(m.state_dict()))
                best_probe_loss = _sequence_loss(m, probe_seq)
            m.backward(batch_x, logits, cache, dlogits)
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
        final_loss = losses[-1] if losses else float("inf")
        if best_sd is None:
            best_sd = m.state_dict()
            best_loss = final_loss
            best_probe_loss = _sequence_loss(m, probe_seq)
        val_loss = _val_loss(m, val_seqs)
        # persist the BEST checkpoint (state_dict + training metadata), and a
        # per-run file whose path is recorded in the lineage ledger.
        best_sd["trained_steps"] = prev_steps + steps
        best_sd["final_loss"] = best_loss
        best_sd["probe_seq"] = probe_seq
        best_sd["probe_loss"] = best_probe_loss
        # Tamper-evident lineage link: the checkpoint self-names the run that
        # produced it, so introspection can resolve provenance by run_id instead
        # of guessing by filename (which misattributes the live latest.json).
        best_sd["lineage_run_id"] = run.run_id
        # Serialize the learned tokenizer into the checkpoint so inference can
        # reconstruct the exact vocabulary with nothing fetched or imported.
        best_sd["tokenizer"] = tok.state_dict()
        per_run_path.write_text(json.dumps(best_sd))
        latest_path.write_text(json.dumps(best_sd))
        note = f"ok; val_loss={val_loss:.4f}" if val_loss is not None else "ok"
        ledger.finish_run(run.run_id, STATUS_DONE, final_loss=best_loss, note=note)
        return {
            "run_id": run.run_id,
            "checkpoint": str(per_run_path),
            "latest": str(latest_path),
            "final_loss": best_loss,
            "final_loss_last": final_loss,
            "val_loss": val_loss,
            "steps": steps,
            "git_hash": run.git_hash,
            "data_manifest_sha256": data_sha,
        }
    except Exception as e:  # pragma: no cover - defensive
        if ledger is not None and run is not None:
            try:
                ledger.finish_run(run.run_id, STATUS_FAILED, note=str(e))
            except Exception:
                pass
        raise
    finally:
        if ledger is not None:
            ledger.close()


if __name__ == "__main__":
    import argparse
    import subprocess

    ap = argparse.ArgumentParser(description="Train Lucy's local model from scratch")
    ap.add_argument("--steps", type=int, default=200, help="SGD training steps (default 200)")
    ap.add_argument("--lr", type=float, default=0.05, help="learning rate")
    ap.add_argument("--ctx", type=int, default=32, help="context length")
    ap.add_argument("--seed", type=int, default=1, help="RNG seed")
    ap.add_argument("--resume", action="store_true", help="resume from latest checkpoint")
    args = ap.parse_args()

    try:
        ghash = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        ghash = ""
    summary = train(steps=args.steps, lr=args.lr, ctx=args.ctx, seed=args.seed, git_hash=ghash, resume=args.resume)
    print(json.dumps(summary, indent=2))
