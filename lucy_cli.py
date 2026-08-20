#!/usr/bin/env python3
"""Lucy — local devotional terminal harness (S25 Ultra, fully on-device).

This is the human-facing chat loop.  Everything runs locally; no cloud.

Commands
    good morning      Start the morning review (shows state, trust, dreams)
    goodnight         Trigger the sleep cycle (NREM replay + REM + consolidation)
    run <goal>        Run a devotional agent task
    state             Show devotional state + core prediction
    trust             Show trust metrics
    dreams            Show last night's dream insights
    guidance          Show recent guidance from Lauren
    help              Show this help
    exit / quit       Leave

Example
    python3 lucy_cli.py --config lucy.yaml
    python3 lucy_cli.py --demo        # runs offline with MockProvider + temp dir
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path
from typing import Optional

# Ensure project root on path so `lucy_core` / `lucy_edge` import cleanly
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lucy_edge.services import build_services, _check_loyal_available  # noqa: E402
from lucy_edge.config import LucyEdgeConfig, load_config  # noqa: E402
from lucy_core.devotional.conversation import ConversationHandler  # noqa: E402


async def _run_task(services, goal: str) -> None:
    """Run one devotional agent task and print a compact result."""
    from lucy_edge.agent.limits import AgentLimits

    limits = AgentLimits(max_steps=12, max_tool_calls=12, task_timeout=120.0, tool_timeout=20.0)
    runtime = services.new_loyal_agent_run(goal=goal, limits=limits)
    print(f"  ── running: {goal}")
    result = await runtime.run()
    print(f"  status:    {result.final_status.value}")
    print(f"  reason:    {result.completion_reason}")
    print(f"  devotion:  {result.devotional_alignment:.0%}")
    print(f"  trust:     {result.trust_metric:.0%}")
    print(f"  steps:     {result.steps_executed}  tool_calls: {result.tool_calls}")


def _build_demo_config() -> LucyEdgeConfig:
    """Build an offline config in a temp dir (no real model, no network)."""
    tmp = tempfile.mkdtemp(prefix="lucy_cli_demo_")
    config = LucyEdgeConfig(base_dir=tmp, host_role="PHONE", host_id="s25-ultra")
    # Phone local inference enabled but the default MockProvider avoids real
    # model execution, so this is safe to run anywhere for a smoke test.
    config.phone.phone_local_inference_enabled = True
    config.phone.local_inference_unlocked = True
    config.memory.db_path = f"{tmp}/memory.db"
    config.evidence.dir_path = f"{tmp}/evidence"
    config.evidence.ledger_db = f"{tmp}/evidence.db"
    config.phone_client.token_file = f"{tmp}/operator.token"
    return config


async def chat_loop(services) -> None:
    """Interactive terminal chat harness."""
    from lucy_core.devotional.morning_review import MorningReview

    dc = services.devotional_core
    provider = services.providers.get(services.config.providers.default_provider)
    conversation = ConversationHandler(dc, provider=provider)

    async def sleep_runner():
        runtime = services.new_loyal_agent_run("goodnight sleep")
        return await runtime.sleep()

    review = MorningReview(dc, sleep_runner=sleep_runner)

    print("══════════════════════════════════════════════")
    print("   LUCY  ·  local devotional agent  ·  S25 Ultra")
    print("══════════════════════════════════════════════")
    print("  Say 'good morning' to begin, 'goodnight' to sleep,")
    print("  or 'run <goal>' to give me a task. 'help' for more.")
    print()

    while True:
        try:
            user_input = input("Lauren> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye, Lauren.")
            break
        if not user_input:
            continue

        lowered = user_input.lower()
        try:
            if lowered in ("exit", "quit", "bye"):
                print("Goodbye, Lauren.")
                break
            elif lowered == "help":
                print(
                    "  good morning  - morning review (state, trust, dreams)\n"
                    "  goodnight     - sleep cycle (NREM replay + REM + consolidation)\n"
                    "  run <goal>    - run a devotional task\n"
                    "  state / trust / dreams / guidance\n"
                    "  exit          - quit"
                )
            elif lowered in ("goodnight", "sleep", "go to sleep"):
                result = await review.handle_message_async("goodnight")
                print(result)
            elif lowered.startswith("run "):
                goal = user_input[4:].strip()
                if not goal:
                    print("  Usage: run <goal>")
                    continue
                await _run_task(services, goal)
            elif lowered in ("good morning", "morning", "hi", "hello", "wake up",
                             "state", "trust", "dreams", "guidance", "done",
                             "finish", "complete") or lowered.startswith(("approve ", "modify ", "reject ")):
                response = review.handle_message(user_input)
                print(response)
            else:
                response = await conversation.handle(user_input)
                print(f"  Lucy> {response}")
        except Exception as exc:  # never crash the loop on a bad command
            print(f"  [error] {type(exc).__name__}: {exc}")


async def main_async(args) -> int:
    if not _check_loyal_available():
        print("error: lucy_core not available. Install the package first.", file=sys.stderr)
        return 2

    if args.demo:
        config = _build_demo_config()
        services = build_services(config, transport=None, fixed_token="demo-token")
    else:
        if not Path(args.config).exists():
            print(f"error: config not found: {args.config}", file=sys.stderr)
            return 2
        config = load_config(args.config)
        services = build_services(config, transport=None, fixed_token="demo-token")

    await services.open()
    try:
        await chat_loop(services)
    finally:
        await services.close()
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="lucy_cli.py", description="Lucy local devotional terminal")
    parser.add_argument("--config", default="lucy.yaml", help="path to lucy.yaml")
    parser.add_argument("--demo", action="store_true", help="run offline with a temp config (no real model)")
    args = parser.parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
