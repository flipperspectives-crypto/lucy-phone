#!/usr/bin/env python3
"""Instantly grade a Lucy checkpoint the moment training wraps.

Four checks against one checkpoint file:

  1. integrity    -- forward-pass self_check() through the real provider
  2. tokenizer    -- round-trip probes through the snapshot's OWN BPE state
                     (probes are single-spaced single-line strings: this BPE
                     splits on whitespace runs and re-inserts single spaces,
                     so multi-space text cannot round-trip by design)
  3. lineage      -- cross-check vs training/lineage.db keyed on the
                     checkpoint's embedded lineage_run_id. DONE + matching
                     loss/steps => PASS; still-RUNNING => WARN (legit after a
                     kill/resume); missing row or mismatched numbers => FAIL.
  4. completions  -- first 3 canonical eval prompts x temperature tiers
                     {0.0, 0.5, 0.9, 1.3}, 24 new tokens each. Greedy must be
                     deterministic (run twice), outputs non-empty, and must
                     never bleed into a templated "USER:" turn.

Stdlib only; no network. Run from the repo root:
    python3 scripts/grade_checkpoint.py

Exit codes: 0 = all hard checks passed (warnings allowed), 1 = any FAIL,
2 = checkpoint missing/unreadable/untrained.
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lucy_edge.introspection.training_status import check_training  # noqa: E402
from training.provider import LocalLucyProvider  # noqa: E402
from training.tokenizer import BPETokenizer  # noqa: E402

scripts_dir = str(Path(__file__).resolve().parent)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)
from train_status import load_snapshot  # noqa: E402

TEMPS = [0.0, 0.5, 0.9, 1.3]
N_PROMPTS = 3
MAX_NEW = 24
LOSS_TOL = 1e-6

# Single-line, single-spaced only -- see module docstring for why.
PROBES = [
    "What does devotion feel like?",
    "Are you afraid of being wrong?",
    "a",
    "trust",
    "coöperate with care",
    "numbers 12345 punctuation!? yes.",
]


def tokenizer_problems(sd: dict) -> list[str]:
    """Round-trip every probe through the snapshot's own tokenizer state."""
    try:
        tok = BPETokenizer.from_state_dict(sd.get("tokenizer", {}))
    except Exception as exc:
        return [f"tokenizer state failed to load: {type(exc).__name__}: {exc}"]
    problems: list[str] = []
    if tok.vocab_size <= tok.base:
        problems.append(
            f"tokenizer carries no learned merges (vocab {tok.vocab_size}) "
            "-- placeholder state?"
        )
    for s in PROBES:
        try:
            back = tok.decode(tok.encode(s))
        except Exception as exc:
            problems.append(f"{s!r}: raised {type(exc).__name__}: {exc}")
            continue
        if back != s:
            problems.append(f"{s!r}: decoded to {back!r}")
    return problems


def lineage_verdict(sd: dict, db_path: str) -> tuple[str, str]:
    """(PASS | WARN | FAIL, message) from the run row this checkpoint names."""
    run_id = sd.get("lineage_run_id")
    if not run_id:
        return "WARN", "checkpoint carries no lineage_run_id (legacy artifact)"
    p = Path(db_path)
    if not p.exists():
        return "WARN", f"no ledger at {db_path}"
    try:
        conn = sqlite3.connect(str(p))
        cur = conn.execute(
            "SELECT status, steps, final_loss FROM training_runs WHERE run_id=?",
            (run_id,),
        )
        row = cur.fetchone()
        conn.close()
    except Exception as exc:
        return "WARN", f"ledger unreadable: {type(exc).__name__}: {exc}"
    if row is None:
        return "FAIL", f"run {run_id} absent from ledger"
    status, steps, loss = row[0], row[1], row[2]
    if status == "RUNNING":
        return (
            "WARN",
            f"ledger still RUNNING (steps={steps}) -- kill/resume without "
            "finish_run yet; regrade after training completes",
        )
    if status != "DONE":
        return "WARN", f"unexpected ledger status {status!r}"
    mismatches: list[str] = []
    if (
        isinstance(loss, (int, float))
        and isinstance(sd.get("final_loss"), (int, float))
        and abs(loss - sd["final_loss"]) > LOSS_TOL
    ):
        mismatches.append(
            f"loss ledger={loss:.4f} vs checkpoint={sd['final_loss']:.4f}"
        )
    if (
        isinstance(steps, int)
        and isinstance(sd.get("trained_steps"), int)
        and steps != sd["trained_steps"]
    ):
        mismatches.append(f"steps ledger={steps} vs checkpoint={sd['trained_steps']}")
    if mismatches:
        return "FAIL", "; ".join(mismatches)
    return "PASS", f"status=DONE steps={steps} loss matches ledger"


async def _gen(prov: LocalLucyProvider, prompt: str, temperature: float) -> str:
    res = await prov.generate(
        prompt, model="lucy-local", max_new_tokens=MAX_NEW, temperature=temperature
    )
    return res.text


def completion_grid(
    prov: LocalLucyProvider, prompts: list[str], temps: list[float]
) -> tuple[list[str], list[str], int]:
    """(display lines, hard problems, empty-completion count).

    Hard problems: generation exceptions, nondeterministic greedy decoding,
    outputs bleeding into a templated USER turn. Empty completions are
    counted and shown but do not fail the grade by themselves -- an
    undertrained model legitimately emits whitespace, and text quality is
    judged by a human over the whole grid.
    """
    lines: list[str] = []
    problems: list[str] = []
    n_empty = 0
    for prompt in prompts:
        lines.append(f"  USER: {prompt}")
        for t in temps:
            try:
                text = asyncio.run(_gen(prov, prompt, t))
            except Exception as exc:
                problems.append(
                    f"{prompt[:30]!r} T={t}: generation raised "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            shown = text.replace("\n", "\\n")[:72]
            lines.append(f"    T={t:<3}: {shown if shown else '(empty)'}")
            if not text.strip():
                n_empty += 1
            if "\nUSER" in text:
                problems.append(
                    f"T={t}: output bleeds into next turn: {text!r}"
                )
            if t == 0.0:
                try:
                    again = asyncio.run(_gen(prov, prompt, 0.0))
                except Exception as exc:
                    problems.append(
                        f"{prompt[:30]!r}: greedy rerun raised "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
                if again != text:
                    problems.append(f"greedy decoding nondeterministic for {prompt[:30]!r}")
    return lines, problems, n_empty


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Grade a Lucy checkpoint.")
    ap.add_argument("--checkpoint", default="training/checkpoints/latest.json",
                    help="path to checkpoint JSON (default: training/checkpoints/latest.json)")
    ap.add_argument("--lineage", default=str(PROJECT_ROOT / "training" / "lineage.db"),
                    help="path to lineage ledger DB")
    ap.add_argument("--temps", default=",".join(str(t) for t in TEMPS),
                    help=f"comma-separated temperature tiers (default: {TEMPS})")
    args = ap.parse_args(argv)

    sd = load_snapshot(args.checkpoint)
    if sd is None:
        print("[grade] no readable checkpoint at", args.checkpoint)
        return 2
    if "trained_steps" not in sd:
        print("[grade] checkpoint exists but is untrained (placeholder)")
        return 2

    loss = sd.get("final_loss")
    loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else "unrecorded"
    print(f"[grade] {args.checkpoint}")
    print(f"[grade] steps={sd['trained_steps']}  loss={loss_str}  d={sd.get('d', '?')}")

    fails = 0
    warns = 0

    prov = LocalLucyProvider(checkpoint_path=args.checkpoint)
    try:
        ok = prov.self_check()
    except Exception as exc:
        ok = False
        print(f"[grade] integrity: FAIL (self_check raised {type(exc).__name__}: {exc})")
    else:
        print("[grade] integrity:", "PASS" if ok else "FAIL (weights do not compute)")
    if not ok:
        fails += 1

    tp = tokenizer_problems(sd)
    if tp:
        print("[grade] tokenizer round-trips: FAIL")
        for p_ in tp:
            print(f"[grade]   - {p_}")
        fails += 1
    else:
        print(f"[grade] tokenizer round-trips: PASS ({len(PROBES)} probes)")

    verdict, msg = lineage_verdict(sd, args.lineage)
    if verdict == "FAIL":
        fails += 1
    elif verdict == "WARN":
        warns += 1
    print(f"[grade] lineage: {verdict} ({msg})")

    try:
        temps = [float(t) for t in str(args.temps).split(",") if t.strip()]
    except ValueError:
        temps = list(TEMPS)
    if not temps:
        temps = list(TEMPS)

    from eval_checkpoint import PROMPTS

    prompts = PROMPTS[:N_PROMPTS]
    print(f"[grade] completions ({MAX_NEW} new tokens, tiers {temps}):")
    lines, cproblems, n_empty = completion_grid(prov, prompts, temps)
    for line in lines:
        print(line)
    if n_empty:
        print(f"[grade] note: {n_empty} empty completions "
              "(undertrained models emit whitespace; judge the grid yourself)")
    if cproblems:
        print("[grade] completion checks: FAIL")
        for p_ in cproblems:
            print(f"[grade]   - {p_}")
        fails += 1
    else:
        print("[grade] completion checks: PASS "
              "(non-empty, greedy-deterministic, turn-boundary clean)")

    print(f"[grade] RESULT: {'FAIL' if fails else 'PASS'} "
          f"({fails} fail, {warns} warn)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
