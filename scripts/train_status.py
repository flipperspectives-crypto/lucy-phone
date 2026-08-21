#!/usr/bin/env python3
"""One-shot status check for an in-progress Lucy training run.

Reads the crash-recovery snapshot train.py writes every --checkpoint-every
steps and reports: progress toward target, last loss, snapshot age, whether
a trainer process is alive, and (if it died) the exact resume command.

Stdlib only; no network. Run from the repo root:
    python3 scripts/train_status.py
    python3 scripts/train_status.py --target 15000

Exit codes: 0 = running or already complete, 1 = NOT RUNNING (needs
attention), 2 = no snapshot yet (startup or never started).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

STALE_MINUTES = 10


def load_snapshot(path: str) -> dict | None:
    """Parsed checkpoint state dict, or None if missing/unreadable."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        sd = json.loads(p.read_text())
    except Exception:
        return None
    return sd if isinstance(sd, dict) else None


def find_training_pid() -> int | None:
    """PID of a live `training.train` process, or None.

    Scans /proc (Linux/Termux). Returns None on systems without /proc --
    status then reports aliveness as unknown rather than guessing.
    """
    if not os.path.isdir("/proc"):
        return None
    me = os.getpid()
    try:
        entries = [e for e in os.listdir("/proc") if e.isdigit()]
    except OSError:
        return None
    for entry in entries:
        pid = int(entry)
        if pid == me:
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\x00", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if "training.train" in cmd:
            return pid
    return None


def resume_command(saved: int, target: int) -> str:
    """The additive resume one-liner; empty string if there is nothing left."""
    remaining = max(0, target - saved)
    if remaining == 0:
        return ""
    return f"python3 -m training.train --resume --steps {remaining} --seed 1"


def render_status(
    sd: dict, target: int, pid: int | None, now: float, mtime: float | None
) -> str:
    lines = []
    steps = sd.get("trained_steps")
    loss = sd.get("final_loss")
    loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else "unrecorded"
    d = sd.get("d", "?")
    lines.append(f"[status] step {steps}/{target}  loss {loss_str}  d={d}")
    if mtime is not None:
        age_min = (now - mtime) / 60
        lines.append(f"[status] snapshot: {age_min:.0f} min old")
        if age_min > STALE_MINUTES and pid is not None:
            lines.append(
                f"[status] warning: snapshot stale but process alive "
                f"-- throttled host, or hung"
            )
    if pid is not None:
        lines.append(f"[status] trainer: RUNNING (pid {pid})")
    elif pid is None and os.path.isdir("/proc"):
        lines.append("[status] trainer: NOT RUNNING")
        cmd = resume_command(int(steps or 0), target)
        if cmd:
            lines.append("[status] resume with:")
            lines.append(f"  {cmd}")
        else:
            lines.append("[status] training complete -- evaluate it:")
            lines.append("  python3 scripts/eval_checkpoint.py --once --quant")
    else:
        lines.append("[status] trainer: unknown (no /proc on this platform)")
    return "\n".join(lines)


def main(argv=None, _pid: int | None = None) -> int:
    ap = argparse.ArgumentParser(description="Status of a Lucy training run.")
    ap.add_argument("--checkpoint", default="training/checkpoints/latest.json",
                    help="path to snapshot JSON (default: training/checkpoints/latest.json)")
    ap.add_argument("--target", type=int, default=15000,
                    help="total steps the run is aiming for (default: 15000)")
    args = ap.parse_args(argv)

    sd = load_snapshot(args.checkpoint)
    if sd is None:
        print("[status] no checkpoint yet")
        print("[status] either startup is still running (BPE + corpus encode "
              "takes ~1 min) or training never started")
        return 2
    if "trained_steps" not in sd:
        print("[status] checkpoint exists but is untrained (placeholder)")
        return 2

    pid = find_training_pid() if _pid is None else _pid
    mtime = None
    try:
        mtime = os.path.getmtime(args.checkpoint)
    except OSError:
        pass
    print(render_status(sd, args.target, pid, time.time(), mtime))
    if pid is not None:
        return 0
    # Dead (or unknowable): exit 0 only if the run reached its target.
    return 0 if int(sd.get("trained_steps") or 0) >= args.target else 1


if __name__ == "__main__":
    sys.exit(main())
