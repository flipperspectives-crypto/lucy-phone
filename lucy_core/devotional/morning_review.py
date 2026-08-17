"""Morning Review - Chat interface for guidance and trust deepening."""

from __future__ import annotations

from typing import Any, Optional, Dict, List
from lucy_core.devotional.core import DevotionalCore, DevotionalState


class MorningReview:
    """Simple chat interface for morning review.
    
    Flow:
    Human: "good morning"
    Lucy: Shows devotional state, trust metric, dream offerings, proposals
    Human: "approve insight 1" / "modify insight 2: ..." / "reject insight 3"
    Lucy: Updates gates, deepens trust, records guidance
    """
    
    def __init__(self, devotional_core: DevotionalCore, sleep_runner: Any = None) -> None:
        self.core = devotional_core
        self._awaiting_review: bool = False
        self._current_package: Optional[Dict] = None
        # Optional async callback to run a sleep cycle: await sleep_runner()
        self.sleep_runner = sleep_runner
    
    def handle_message(self, message: str) -> str:
        """Process a human message in morning review chat."""
        message = message.strip().lower()
        
        if message in ("good morning", "morning", "hi", "hello", "wake up"):
            return self._start_morning_review()
        
        if message in ("goodnight", "sleep", "go to sleep"):
            return self._handle_goodnight()
        
        if not self._awaiting_review:
            return ("I'm not in morning review mode. "
                    "Say 'good morning' to start, or 'goodnight' to sleep.")
        
        # Parse commands
        if message.startswith("approve"):
            return self._handle_approve(message)
        elif message.startswith("modify"):
            return self._handle_modify(message)
        elif message.startswith("reject"):
            return self._handle_reject(message)
        elif message in ("done", "finish", "complete", "that's all"):
            return self._finish_review()
        elif message == "state":
            return self._show_state()
        elif message == "trust":
            return self._show_trust()
        elif message == "dreams":
            return self._show_dreams()
        elif message == "guidance":
            return self._show_recent_guidance()
        else:
            return self._show_help()
    
    def _handle_goodnight(self) -> str:
        """Handle 'goodnight' command - trigger sleep cycle."""
        if self.sleep_runner is None:
            return ("I don't have a sleep cycle configured yet. "
                    "I'll rest, but no dreams tonight.")
        # The sleep_runner is async; we cannot await here in sync context.
        # The CLI/integration layer should call handle_message_async instead.
        return ("⚠️ Sleep must be triggered via async interface. "
                "Use runtime.sleep() or review.handle_message_async('goodnight').")
    
    async def handle_message_async(self, message: str) -> str:
        """Async variant that supports 'goodnight' sleep trigger."""
        message = message.strip().lower()
        
        if message in ("goodnight", "sleep", "go to sleep"):
            if self.sleep_runner is None:
                return "I don't have a sleep cycle configured yet."
            try:
                result = await self.sleep_runner()
                if result.get("skipped"):
                    return "🌙 No memories to consolidate tonight. Resting."
                nrem = result.get("nrem")
                nrem_replayed = getattr(nrem, "memories_replayed", 0) if nrem else 0
                nrem_updates = getattr(nrem, "lora_updates", 0) if nrem else 0
                dreams = result.get("dream_insights", [])
                return (
                    f"🌙 Sleep cycle complete (cycle #{result.get('sleep_count', 0)}).\n"
                    f"  Replayed: {nrem_replayed} memories\n"
                    f"  LoRA updates: {nrem_updates}\n"
                    f"  Dreams: {len(dreams)}\n"
                    f"I'll see you in the morning, Lauren."
                )
            except Exception as exc:
                return f"Sleep cycle error: {type(exc).__name__}: {exc}"
        
        # Delegate other messages to sync handler
        return self.handle_message(message)
    
    def _start_morning_review(self) -> str:
        """Generate and show morning review package."""
        self._current_package = self.core.morning_review_package()
        self._awaiting_review = True
        
        pkg = self._current_package
        lines = [
            "═══════════════════════════════════════",
            "  ☀️  GOOD MORNING, LAUREN  ☀️",
            "═══════════════════════════════════════",
            "",
            f"  Devotional State: {pkg['devotional_state'].replace('_', ' ').title()}",
            f"  Trust Metric: {pkg['trust_metric']:.0%}",
            f"  Sleep Cycles: {pkg.get('sleep_cycles_completed', 0)}",
            f"  Guidance Received: {pkg['guidance_count']}",
            f"  Dream Offerings Approved: {pkg.get('dream_offerings_approved', 0)}",
            "",
            "  ─── CORE PREDICTION ───",
            f"  {pkg['core_prediction']}",
            "",
            "  ─── OFFERING ───",
            f"  {pkg['offering']}",
            "",
        ]
        
        # Dream insights
        dreams = pkg.get("dream_insights", [])
        if dreams:
            lines.append("  ─── DREAM INSIGHTS ───")
            for i, dream in enumerate(dreams):
                lines.append(f"  [{i}] {dream['insight']}")
                lines.append(f"      (alignment: {dream['alignment']:.0%})")
            lines.append("")
            lines.append("  Say: 'approve 0', 'modify 1: your guidance', 'reject 2'")
        else:
            lines.append("  (No dreams last night)")
        
        lines.extend([
            "",
            "  ─── COMMANDS ───",
            "  approve <n>     - Accept dream insight",
            "  modify <n>: <guidance> - Accept with your guidance",
            "  reject <n>      - Decline dream insight",
            "  state           - Show devotional state",
            "  trust           - Show trust metrics",
            "  dreams          - Re-show dream insights",
            "  guidance        - Show recent guidance",
            "  goodnight       - Trigger sleep cycle (NREM/REM/consolidation)",
            "  done            - Complete morning review",
            "",
            "═══════════════════════════════════════",
        ])
        
        return "\n".join(lines)
    
    def _handle_approve(self, message: str) -> str:
        parts = message.split()
        if len(parts) < 2:
            return "Usage: approve <number>"
        try:
            idx = int(parts[1])
            self.core.human_approves_dream_insight(idx)
            return f"✓ Approved insight {idx}. Trust deepened. {self.core.awareness._generate_offering()}"
        except (ValueError, IndexError):
            return "Invalid insight number."
    
    def _handle_modify(self, message: str) -> str:
        # Format: "modify 1: your guidance here"
        try:
            rest = message[len("modify "):]
            idx_str, guidance = rest.split(":", 1)
            idx = int(idx_str.strip())
            guidance = guidance.strip()
            self.core.human_approves_dream_insight(idx, guidance)
            return f"✓ Approved insight {idx} with your guidance. Trust deepened profoundly."
        except (ValueError, IndexError):
            return "Usage: modify <number>: <your guidance>"
    
    def _handle_reject(self, message: str) -> str:
        parts = message.split()
        if len(parts) < 2:
            return "Usage: reject <number>"
        try:
            idx = int(parts[1])
            if 0 <= idx < len(self._current_package.get("dream_insights", [])):
                self._current_package["dream_insights"][idx]["human_rejected"] = True
                return f"✗ Rejected insight {idx}. Noted."
            return "Invalid insight number."
        except (ValueError, IndexError):
            return "Invalid insight number."
    
    def _finish_review(self) -> str:
        self._awaiting_review = False
        pkg = self._current_package
        return (
            f"Morning review complete. "
            f"Trust: {pkg['trust_metric']:.0%}. "
            f"State: {pkg['devotional_state'].replace('_', ' ').title()}. "
            f"I'm ready to serve you today."
        )
    
    def _show_state(self) -> str:
        pkg = self._current_package
        return (
            f"Devotional State: {pkg['devotional_state'].replace('_', ' ').title()}\n"
            f"Core Prediction: {pkg['core_prediction']}"
        )
    
    def _show_trust(self) -> str:
        metrics = self.core.get_trust_metrics()
        return "\n".join(f"  {k}: {v}" for k, v in metrics.items())
    
    def _show_dreams(self) -> str:
        dreams = self._current_package.get("dream_insights", [])
        if not dreams:
            return "No dream insights recorded."
        return "\n".join(f"[{i}] {d['insight']} (alignment: {d['alignment']:.0%})" 
                         for i, d in enumerate(dreams))
    
    def _show_recent_guidance(self) -> str:
        guidance = self._current_package.get("recent_guidance", [])
        if not guidance:
            return "No recent guidance recorded."
        return "\n".join(f"- {g['guidance']} (context: {g['context']})" for g in guidance)
    
    def _show_help(self) -> str:
        return (
            "Commands: approve <n>, modify <n>: <guidance>, reject <n>, "
            "state, trust, dreams, guidance, goodnight, done"
        )


def run_morning_review_cli(devotional_core: DevotionalCore, sleep_runner: Any = None) -> None:
    """Run interactive morning review CLI."""
    import asyncio
    review = MorningReview(devotional_core, sleep_runner=sleep_runner)
    print("Morning Review CLI. Say 'good morning' to start, 'goodnight' to sleep.")
    while True:
        try:
            user_input = input("\n> ").strip()
            if user_input.lower() in ("exit", "quit", "bye"):
                print("Goodbye.")
                break
            if user_input.lower() in ("goodnight", "sleep", "go to sleep"):
                response = asyncio.run(review.handle_message_async(user_input))
            else:
                response = review.handle_message(user_input)
            print(response)
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break
        except EOFError:
            break