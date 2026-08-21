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

from training.provider import LocalLucyProvider  # noqa: E402

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


def evaluate(checkpoint: str) -> None:
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
    ap.add_argument("--watch", action="store_true",
                    help="keep running; re-evaluate whenever the checkpoint file changes")
    ap.add_argument("--interval", type=int, default=300,
                    help="poll interval in seconds for --watch (default 300)")
    args = ap.parse_args(argv)

    if not args.watch:
        evaluate(args.checkpoint)
        return 0

    last_mtime = None
    print(f"[watch] polling {args.checkpoint} every {args.interval}s (Ctrl-C to stop)")
    while True:
        try:
            mtime = os.path.getmtime(args.checkpoint) if Path(args.checkpoint).exists() else None
            if mtime is not None and mtime != last_mtime:
                last_mtime = mtime
                evaluate(args.checkpoint)
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[watch] bye")
            return 0


if __name__ == "__main__":
    sys.exit(main())
