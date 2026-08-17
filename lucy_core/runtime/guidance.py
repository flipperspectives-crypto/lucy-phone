"""Guidance Interface - Structured + conversational guidance ingestion."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional, Dict, List

from lucy_edge.agent.planner import PlanStep
from lucy_core.devotional.core import DevotionalCore


@dataclass
class GuidanceRecord:
    """A single piece of human guidance."""
    guidance: str
    context: str
    run_id: str
    step_index: int
    format: str  # "structured" or "conversational"
    timestamp: float


class GuidanceInterface:
    """Handles both structured and conversational guidance formats.
    
    Structured: Terminal review format with explicit judgments
    Conversational: Natural language chat that gets parsed
    Both feed the DevotionalCore.
    """
    
    def __init__(self, devotional_core: DevotionalCore) -> None:
        self.core = devotional_core
        self.history: List[GuidanceRecord] = []
    
    def process_structured(
        self,
        run_id: str,
        step_index: int,
        judgment: str,  # "GOOD", "PARTIAL", "WRONG"
        guidance: str,
        context: str = ""
    ) -> None:
        """Process structured guidance from terminal review."""
        self.core.process_guidance(guidance, context, run_id, step_index)
        self.history.append(GuidanceRecord(
            guidance=guidance,
            context=context,
            run_id=run_id,
            step_index=step_index,
            format="structured",
            timestamp=__import__("time").time(),
        ))
    
    def process_conversational(self, message: str, run_id: str = "", step_index: int = -1) -> Dict:
        """Parse and process conversational guidance.
        
        Examples:
        - "You blocked that write but I needed it. When saving work, don't be so cautious."
        - "Good job on the memory search."
        - "Don't use the git tool without asking me first."
        """
        # Detect sentiment/intent
        guidance_data = self._parse_conversational(message)
        
        # Process through devotional core
        self.core.process_guidance(
            guidance=message,
            context=guidance_data.get("context", "conversational"),
            run_id=run_id or "conversational",
            step_index=step_index,
        )
        
        record = GuidanceRecord(
            guidance=message,
            context=guidance_data.get("context", "conversational"),
            run_id=run_id or "conversational",
            step_index=step_index,
            format="conversational",
            timestamp=__import__("time").time(),
        )
        self.history.append(record)
        
        return {
            "parsed": guidance_data,
            "devotional_state": self.core.awareness.current_state.value,
            "trust_metric": self.core.awareness.trust_metric,
        }
    
    def _parse_conversational(self, message: str) -> Dict:
        """Extract intent and context from natural language."""
        msg = message.lower()
        
        # Correction patterns
        if any(p in msg for p in ["blocked", "rejected", "denied", "stopped", "prevented"]):
            return {
                "type": "correction",
                "context": "gate_rejection",
                "intent": "adjust_gate_threshold",
            }
        
        # Praise patterns
        if any(p in msg for p in ["good", "great", "perfect", "well done", "nice", "thanks"]):
            return {
                "type": "praise",
                "context": "positive_reinforcement",
                "intent": "reinforce_behavior",
            }
        
        # Boundary setting
        if any(p in msg for p in ["don't", "never", "avoid", "stop", "without asking"]):
            return {
                "type": "boundary",
                "context": "tool_restriction",
                "intent": "add_restriction",
            }
        
        # Preference expression
        if any(p in msg for p in ["prefer", "like", "want", "need", "when"]):
            return {
                "type": "preference",
                "context": "preference_expression",
                "intent": "learn_preference",
            }
        
        # Question/uncertainty
        if any(p in msg for p in ["why", "how", "what", "confused", "unsure"]):
            return {
                "type": "question",
                "context": "seeking_understanding",
                "intent": "explain_reasoning",
            }
        
        return {
            "type": "general",
            "context": "conversational_guidance",
            "intent": "absorb",
        }
    
    def get_history(self, limit: int = 10) -> List[GuidanceRecord]:
        return self.history[-limit:]