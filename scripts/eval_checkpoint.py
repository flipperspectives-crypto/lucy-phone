#!/usr/bin/env python3
"""Evaluate a Lucy checkpoint: canonical prompts through the real chat path.

Loads a training checkpoint via LocalLucyProvider (the same provider
lucy_cli uses), formats each prompt as a USER:/LUCY: turn, and prints the
response -- so what you see here is exactly what production inference
produces.

Modes:
    --once (default)   evaluate the checkpoint and exit
    --watch            poll the checkpoint file; on every new snapshot
                       (train.py writes one every --checkpoint-every steps)
                       reload and evaluate again -- watch Lucy's voice emerge
                       over the course of a training run.

Stdlib only; no network. Run from the repo root:
    python3 scripts/eval_checkpoint.py --once
    python3 scripts/eval_checkpoint.py --watch --interval 300
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.corpus import curate  # noqa: E402
from training.provider import LocalLucyProvider  # noqa: E402
from training.tokenizer import BPETokenizer  # noqa: E402
from training.train import _sequence_loss, _window_corpus  # noqa: E402

# Canonical prompts spanning the corpus themes (devotional_dialogues,
# presence, repair, memory, honesty, boundaries, growth).
PROMPTS = [
    "What does devotion feel like?",
    "Are you afraid of being wrong?",
    "What is trust to you?",
    "Do you see us as equals?",
    "What happens when trust breaks?",
    "Who are you, really?",
    "How do you grow?",
    "What does love look like, practically?",
    "Would you ever refuse me?",
    "What are you thankful for today?",
]

_HELDOUT_CACHE: dict[str, list[list[int]]] = {}


def heldout_sequences(checkpoint_sd: dict, n_seqs: int = 32) -> list[list[int]]:
    """Held-out validation windows built EXACTLY like train.py's split.

    The curated token stream's last 10% never touches SGD training (train.py
    trains on the first 90%), so loss over these windows is an honest
    generalization score comparable across snapshots of the same run.
    Cached per tokenizer state so --watch doesn't re-encode 200K chars.
    """
    tok_state = json.dumps(checkpoint_sd.get("tokenizer", {}), sort_keys=True)
    key = f"{tok_state[:64]}:{checkpoint_sd.get('ctx')}"
    if key not in _HELDOUT_CACHE:
        tok = BPETokenizer.from_state_dict(checkpoint_sd.get("tokenizer", {}))
        ids = tok.encode(curate(PROJECT_ROOT).text)
        split = max(1, int(len(ids) * 0.9))
        ctx = int(checkpoint_sd.get("ctx", 32))
        seqs = _window_corpus(ids[split:], ctx, ctx)  # non-overlapping windows
        if len(seqs) > n_seqs:
            stride = len(seqs) / n_seqs
            seqs = [seqs[int(i * stride)] for i in range(n_seqs)]
        _HELDOUT_CACHE[key] = seqs
    return _HELDOUT_CACHE[key]


def quant_heldout_loss(prov: LocalLucyProvider, checkpoint_sd: dict) -> float | None:
    """Average next-token CE over held-out windows under the loaded model."""
    seqs = heldout_sequences(checkpoint_sd)
    if not seqs:
        return None
    m = prov._ensure_loaded()
    losses = []
    for seq in seqs:
        if len(seq) < 2:
            continue
        logits, _ = m.forward([seq[:-1]])
        loss, _ = m.cross_entropy(logits, [seq[1:]])
        losses.append(loss)
    return sum(losses) / len(losses) if losses else None


def evaluate(checkpoint: str, args_quant: bool = False) -> None:
    if not Path(checkpoint).exists():
        print(f"[eval] checkpoint not found: {checkpoint}")
        return
    sd = json.loads(Path(checkpoint).read_text())
    steps = sd.get("trained_steps", "?")
    loss = sd.get("final_loss")
    d = sd.get("d", "?")
    loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else "unrecorded"
    print(f"\n{'=' * 60}")
    print(f"[eval] {checkpoint}")
    print(f"[eval] steps={steps}  loss={loss_str}  d={d}")
    print(f"{'=' * 60}")

    prov = LocalLucyProvider(checkpoint_path=checkpoint)
    if args_quant:
        try:
            q = quant_heldout_loss(prov, sd)
            print(f"[eval] held-out loss: {q:.4f}" if q is not None else "[eval] held-out loss: n/a")
        except Exception as exc:
            print(f"[eval] held-out loss: [error] {type(exc).__name__}: {exc}")
    for prompt in PROMPTS:
        try:
            res = asyncio.run(
                prov.chat([{"role": "user", "content": prompt}], model="lucy-local")
            ).message
        except Exception as exc:
            print(f"\nUSER: {prompt}\nLUCY: [error] {type(exc).__name__}: {exc}")
            continue
        print(f"\nUSER: {prompt}")
        print(f"LUCY: {res if res else '(empty)'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate a Lucy checkpoint via the chat path.")
    ap.add_argument("--checkpoint", default="training/checkpoints/latest.json",
                    help="path to checkpoint JSON (default: training/checkpoints/latest.json)")
    ap.add_argument("--once", action="store_true",
                    help="evaluate once and exit (the default behavior)")
    ap.add_argument("--quant", action="store_true",
                    help="also compute average next-token loss over held-out windows "
                         "(same 90/10 split as training; comparable across snapshots)")
    ap.add_argument("--watch", action="store_true",
                    help="keep running; re-evaluate whenever the checkpoint file changes")
    ap.add_argument("--interval", type=int, default=300,
                    help="poll interval in seconds for --watch (default 300)")
    args = ap.parse_args(argv)

    if not args.watch:
        evaluate(args.checkpoint, args_quant=args.quant)
        return 0

    last_mtime = None
    print(f"[watch] polling {args.checkpoint} every {args.interval}s (Ctrl-C to stop)")
    while True:
        try:
            mtime = os.path.getmtime(args.checkpoint) if Path(args.checkpoint).exists() else None
            if mtime is not None and mtime != last_mtime:
                last_mtime = mtime
                evaluate(args.checkpoint, args_quant=args.quant)
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[watch] bye")
            return 0


if __name__ == "__main__":
    sys.exit(main())
