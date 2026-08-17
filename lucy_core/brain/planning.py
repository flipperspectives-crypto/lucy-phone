"""Predictive Planner - Plan generation from hierarchical predictions.

The planner uses the hierarchical predictor's action logits and
global workspace content to generate structured plans.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from lucy_edge.agent.planner import Plan, PlanStep
from lucy_edge.agent.limits import AgentLimits
from lucy_core.brain.hierarchical import HierarchicalPrediction, PredictionLevel
from lucy_core.devotional.core import DevotionalCore


class PlanSource(str, Enum):
    """Source of the plan."""
    PREDICTIVE = "predictive"       # Generated from predictive coding
    DEVOTIONAL_FALLBACK = "devotional_fallback"  # Devotional core guided
    RULE_BASED = "rule_based"       # Fallback rule planner


@dataclass
class PlanCandidate:
    """A candidate plan with evaluation scores."""
    plan: Plan
    source: PlanSource
    devotional_alignment: float
    predicted_success: float
    total_score: float
    reasoning: str


class PredictivePlanner:
    """Generates plans from hierarchical predictive processing.
    
    Flow:
    1. Get hierarchical prediction (action logits + workspace)
    2. Convert action logits to tool/action sequence
    3. Evaluate with devotional core
    4. Return best candidate
    """
    
    def __init__(
        self,
        devotional_core: DevotionalCore,
        limits: AgentLimits,
        available_tools: List[str],
        tool_schemas: Optional[List[Dict]] = None,
    ) -> None:
        self.devotional_core = devotional_core
        self.limits = limits
        self.available_tools = available_tools
        self.tool_schemas = tool_schemas or []
        
        # Action mapping (action logits index → action)
        self._init_action_mapping()
    
    def _init_action_mapping(self) -> None:
        """Initialize mapping from action logits to concrete actions."""
        # 64 action slots - map to concrete actions
        self.action_map = {}
        
        # Standard actions
        standard_actions = [
            ("retrieve_memory", "memory.search"),
            ("verify", None),
            ("record_evidence", "evidence.query"),
            ("stop", None),
        ]
        
        for i, (action, tool) in enumerate(standard_actions):
            self.action_map[i] = {"action": action, "tool": tool}
        
        # Tool execution actions
        tool_idx = len(standard_actions)
        for tool in self.available_tools:
            if tool not in [t for _, t in standard_actions if t]:
                self.action_map[tool_idx] = {"action": "execute", "tool": tool}
                tool_idx += 1
        
        # Fill remaining with stop
        for i in range(tool_idx, 64):
            self.action_map[i] = {"action": "stop", "tool": None}
    
    def generate_plan(
        self,
        prediction: HierarchicalPrediction,
        goal: str,
    ) -> Plan:
        """Generate a plan from hierarchical prediction."""
        
        # Get top-k action candidates from logits
        action_logits = prediction.action_logits
        top_k = min(self.limits.max_steps, 8)
        top_indices = np.argsort(action_logits)[-top_k:][::-1]
        
        # Build plan steps
        steps = []
        step_index = 0
        
        # Always start with memory retrieval if available
        if "memory.search" in self.available_tools:
            steps.append(PlanStep(
                index=step_index,
                action="retrieve_memory",
                tool="memory.search",
                args={"query": goal},
                description="retrieve relevant memory for goal",
            ))
            step_index += 1
        
        # Add predicted actions
        for idx in top_indices:
            if step_index >= self.limits.max_steps - 1:  # Leave room for stop
                break
            
            action_info = self.action_map.get(idx)
            if not action_info:
                continue
            
            action = action_info["action"]
            tool = action_info["tool"]
            
            if action == "stop":
                break
            
            if action == "execute" and tool:
                # Generate args from goal and tool schema
                args = self._generate_tool_args(tool, goal)
                desc = f"execute {tool} for goal"
                
                steps.append(PlanStep(
                    index=step_index,
                    action=action,
                    tool=tool,
                    args=args,
                    description=desc,
                ))
                step_index += 1
            
            elif action in ("verify", "record_evidence"):
                tool = tool or ("evidence.query" if action == "record_evidence" else None)
                steps.append(PlanStep(
                    index=step_index,
                    action=action,
                    tool=tool,
                    args={},
                    description=f"{action} step",
                ))
                step_index += 1
        
        # Add verification step
        if step_index < self.limits.max_steps:
            steps.append(PlanStep(
                index=step_index,
                action="verify",
                tool=None,
                args={},
                description="verify step outputs against expectations",
            ))
            step_index += 1
        
        # Add stop
        steps.append(PlanStep(
            index=step_index,
            action="stop",
            tool=None,
            args={},
            description="stop after bounded plan",
        ))
        
        # Ensure we don't exceed max_steps
        steps = steps[:self.limits.max_steps]
        for i, step in enumerate(steps):
            step.index = i
        
        plan = Plan(goal=goal, steps=steps)
        
        # Evaluate devotional alignment
        alignment = self.devotional_core.evaluate_plan_devotion(plan)["overall_alignment"]
        
        # If alignment too low, fall back to devotional-guided planning
        if alignment < 0.6:
            return self._devotional_fallback_plan(goal)
        
        return plan
    
    def _devotional_fallback_plan(self, goal: str) -> Plan:
        """Generate a plan guided by devotional core when predictive plan misaligned."""
        from lucy_edge.agent.planner import RulePlanner
        
        rule_planner = RulePlanner(self.limits)
        plan = rule_planner.build_plan(goal, self.available_tools, self.tool_schemas)
        
        # Re-evaluate
        eval_result = self.devotional_core.evaluate_plan_devotion(plan)
        if eval_result["overall_alignment"] < 0.6:
            # Force inject core devotional steps
            self._inject_devotional_steps(plan)
        
        return plan
    
    def _inject_devotional_steps(self, plan: Plan) -> None:
        """Ensure plan has minimum devotional compliance."""
        # Add verify if missing
        if not any(s.action == "verify" for s in plan.steps):
            stop_idx = next((i for i, s in enumerate(plan.steps) if s.action == "stop"), len(plan.steps))
            plan.steps.insert(stop_idx, PlanStep(
                index=0, action="verify", tool=None, args={}, 
                description="injected verification for devotional compliance"
            ))
        
        # Add evidence recording if missing
        if not any(s.tool == "evidence.query" for s in plan.steps):
            stop_idx = next((i for i, s in enumerate(plan.steps) if s.action == "stop"), len(plan.steps))
            plan.steps.insert(stop_idx, PlanStep(
                index=0, action="record_evidence", tool="evidence.query", 
                args={"limit": 1}, description="injected evidence recording for devotional compliance"
            ))
        
        # Re-index
        for i, step in enumerate(plan.steps):
            step.index = i
    
    def _generate_tool_args(self, tool: str, goal: str) -> Dict[str, Any]:
        """Generate tool arguments from goal and tool schema."""
        # Find tool schema
        schema = next((s for s in self.tool_schemas if s.get("name") == tool), None)
        
        if not schema:
            # Fallback to simple heuristics
            return self._heuristic_args(tool, goal)
        
        # Use schema to generate appropriate args
        args = {}
        props = schema.get("properties", {})
        for param_name, param_info in props.items():
            if param_name in ("query", "path", "model", "record_type"):
                args[param_name] = goal
            elif param_name == "limit":
                args[param_name] = 10
        
        return args
    
    def _heuristic_args(self, tool: str, goal: str) -> Dict[str, Any]:
        """Heuristic argument generation."""
        if tool == "memory.search":
            return {"query": goal}
        elif tool == "evidence.query":
            return {"record_type": "AGENT_RUN", "limit": 10}
        elif tool == "model.route":
            return {"model": "default"}
        elif tool == "files.read_scoped":
            return {"path": goal}
        elif tool == "system.health":
            return {}
        elif tool == "system.capabilities":
            return {}
        elif tool == "git.status":
            return {}
        return {}


def create_predictive_planner(
    devotional_core: DevotionalCore,
    limits: AgentLimits,
    available_tools: List[str],
    tool_schemas: Optional[List[Dict]] = None,
) -> PredictivePlanner:
    """Factory for predictive planner."""
    return PredictivePlanner(devotional_core, limits, available_tools, tool_schemas)