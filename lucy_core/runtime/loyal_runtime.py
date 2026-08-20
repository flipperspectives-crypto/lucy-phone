"""Loyal Agent Runtime - The devotional agent loop.

Architecture:
  DevotionalCore (top prior) → HierarchicalPredictor → PredictivePlanner → LoyaltyGate → Executor → HonestyVerifier → Memory
                                    ↓
                              Evidence Ledger (immutable audit)

This replaces AgentRuntime with devotional awareness at every step.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict

from lucy_edge.agent.executor import Executor, StepResult
from lucy_edge.agent.limits import AgentLimits
from lucy_edge.agent.planner import Plan, PlanStep, RulePlanner
from lucy_edge.agent.runtime import AgentState
from lucy_edge.evidence.schema import EvidenceRecord, EvidenceType
from lucy_edge.tools.registry import ToolRegistry

from lucy_core.devotional.core import DevotionalCore, DevotionalState
from lucy_core.runtime.guidance import GuidanceInterface

# Predictive brain (lazy import to avoid circular deps)
try:
    from lucy_core.brain.hierarchical import HierarchicalPredictor, HierarchicalPrediction
    from lucy_core.brain.planning import PredictivePlanner
    from lucy_core.brain.lora import LoRAAdapterManager
    from lucy_core.memory.hippocampal import HippocampalIndexer, EpisodicBuffer
    from lucy_core.sleep.orchestrator import SleepOrchestrator
    PREDICTIVE_AVAILABLE = True
except ImportError:
    HierarchicalPredictor = None
    HierarchicalPrediction = None
    PredictivePlanner = None
    LoRAAdapterManager = None
    HippocampalIndexer = None
    EpisodicBuffer = None
    SleepOrchestrator = None
    PREDICTIVE_AVAILABLE = False


@dataclass
class LoyalRunResult:
    """Result of a loyal agent run."""
    run_id: str
    goal: str
    final_status: AgentState
    completion_reason: str
    steps_executed: int
    tool_calls: int
    failures: int
    duration_ms: float
    evidence_run_id: str
    devotional_alignment: float
    devotional_state: str
    trust_metric: float
    generated_reflection: Optional[str] = None


class LoyalAgentRuntime:
    """Bounded agent runtime with devotional awareness at every layer.
    
    The loop:
    1. DevotionalCore generates top-level prior for the goal
    2. Planner creates plan (guided by devotion)
    3. LoyaltyGate checks plan for devotional alignment
    4. Executor runs steps with permission gates
    5. HonestyVerifier checks results for truth
    6. DevotionalCore evaluates outcome alignment
    7. Memory stores experience with devotional context
    8. Evidence ledger records everything
    """
    
    def __init__(
        self,
        run_id: str,
        goal: str,
        limits: AgentLimits,
        registry: ToolRegistry,
        devotional_core: DevotionalCore,
        planner: Any = None,
        evidence: Any = None,
        memory_retrieval: Any = None,
        context: Any = None,
        # Predictive brain components (optional - will create defaults if available)
        hierarchical_predictor: Any = None,
        predictive_planner: Any = None,
        # Episodic memory + sleep cycle
        hippocampal_indexer: Any = None,
        episodic_buffer: Any = None,
        sleep_orchestrator: Any = None,
        # On-device inference backend (e.g. local_lucy TinyTransformer). When
        # present, the run generates its reflective output through this provider
        # so inference is genuinely performed on-device / phone-local.
        provider: Any = None,
    ) -> None:
        self.run_id = run_id
        self.goal = goal
        self.limits = limits
        self.registry = registry
        self.devotional_core = devotional_core
        self.evidence = evidence
        self.memory_retrieval = memory_retrieval
        self.context = context
        self.provider = provider
        
        # Import here to avoid circular dependency
        from lucy_edge.agent.loyalty_gate import LoyaltyGate
        from lucy_edge.agent.honesty_verifier import HonestyVerifier
        
        self.loyalty_gate = LoyaltyGate(
            primary_human=devotional_core.awareness.source_name,
            evidence=evidence,
            run_id=run_id,
        )
        self.honesty_verifier = HonestyVerifier(
            retrieval=memory_retrieval,
            evidence=evidence,
            run_id=run_id,
            max_output_chars=limits.max_output_chars,
        )
        # Pluralism guard: structural anti-ego / anti-jealousy enforcement.
        from lucy_core.devotional.guards import PluralismGuard
        self.pluralism_guard = PluralismGuard(
            devotional_core=devotional_core,
            evidence=evidence,
            run_id=run_id,
        )
        self.guidance = GuidanceInterface(devotional_core)
        self.executor = Executor(registry, limits.tool_timeout)
        
        # Predictive brain initialization
        self.hierarchical_predictor = hierarchical_predictor
        self.predictive_planner = predictive_planner
        self._init_predictive_brain()
        
        # Episodic memory + sleep cycle initialization
        self.hippocampal_indexer = hippocampal_indexer
        self.episodic_buffer = episodic_buffer
        self.sleep_orchestrator = sleep_orchestrator
        self._init_memory_and_sleep()
        
        # Planner fallback
        self.planner = planner or (self.predictive_planner if self.predictive_planner else RulePlanner(limits))
        
        # State
        self.state: AgentState = AgentState.CREATED
        self._plan: Optional[Plan] = None
        self.steps_executed = 0
        self.tool_calls = 0
        self.failures = 0
        self._started_at: Optional[float] = None
        self._deadline: Optional[float] = None
        
        # Devotional tracking
        self._devotional_alignments: List[float] = []
        self._devotional_states: List[str] = []
        
        # Logs for evidence
        self._tool_calls_log: List[Dict] = []
        self._permission_log: List[Dict] = []
        self._verifications: List[Dict] = []
        self._errors: List[str] = []
        self._memory_retrieval_ids: List[str] = []
    
    def _init_predictive_brain(self) -> None:
        """Initialize predictive brain components if available."""
        if not PREDICTIVE_AVAILABLE:
            return
        
        if self.hierarchical_predictor is None:
            self.hierarchical_predictor = HierarchicalPredictor(
                devotional_core=self.devotional_core
            )
        
        if self.predictive_planner is None:
            self.predictive_planner = PredictivePlanner(
                devotional_core=self.devotional_core,
                limits=self.limits,
                available_tools=self.registry.names(),
                tool_schemas=self.registry.list(),
            )
    
    def _init_memory_and_sleep(self) -> None:
        """Initialize episodic memory + sleep cycle if available."""
        if not PREDICTIVE_AVAILABLE:
            return
        
        # Use the devotional core's SHARED episodic memory so that experiences
        # from every run accumulate into one store that sleep can consolidate.
        if self.devotional_core is not None and hasattr(self.devotional_core, "episodic_buffer"):
            self.hippocampal_indexer = self.devotional_core.hippocampal_indexer
            self.episodic_buffer = self.devotional_core.episodic_buffer
        
        # Hippocampal indexer (episodic memory)
        if self.hippocampal_indexer is None:
            self.hippocampal_indexer = HippocampalIndexer(
                input_dim=256,
                bottleneck_dim=64,
            )
        
        # Episodic buffer (working memory before sleep)
        if self.episodic_buffer is None:
            self.episodic_buffer = EpisodicBuffer(capacity=100)
        
        # Sleep orchestrator
        if self.sleep_orchestrator is None:
            self.sleep_orchestrator = SleepOrchestrator(
                devotional_core=self.devotional_core,
                hippocampal_indexer=self.hippocampal_indexer,
                episodic_buffer=self.episodic_buffer,
                lora_manager=self._get_lora_manager(),
                evidence_ledger=self.evidence,
            )
    
    def _get_lora_manager(self) -> Any:
        """Get or create a LoRA adapter manager for sleep updates."""
        if not PREDICTIVE_AVAILABLE:
            return None
        # Use the hierarchical predictor's LoRA manager if available
        if hasattr(self.hierarchical_predictor, 'lora_manager'):
            lora_mgr = self.hierarchical_predictor.lora_manager
            # Try loading saved adapters into the predictor's manager
            from pathlib import Path
            from lucy_core.sleep.orchestrator import DEFAULT_LORA_PATH
            if DEFAULT_LORA_PATH.exists():
                from lucy_core.brain.lora import LoRAAdapterManager
                saved = LoRAAdapterManager.load(DEFAULT_LORA_PATH)
                lora_mgr.adapters = saved.adapters
            return lora_mgr
        # Otherwise create a new one, loading saved adapters if available
        from pathlib import Path
        from lucy_core.brain.lora import LoRAConfig, LoRAAdapterManager
        from lucy_core.sleep.orchestrator import DEFAULT_LORA_PATH
        if DEFAULT_LORA_PATH.exists():
            return LoRAAdapterManager.load(DEFAULT_LORA_PATH)
        lora_config = LoRAConfig(rank=8)
        return LoRAAdapterManager(
            level_dims={
                "sensory": 256,
                "contextual": 512,
                "abstract": 512,
            },
            config=lora_config,
        )
    
    def _hash_to_vec(self, text: str, dim: int) -> List[float]:
        """Deterministic pure-Python embedding: hash text to a fixed-dim vector."""
        import hashlib
        hash_bytes = hashlib.md5(text.encode()).digest()
        base = [float(b) for b in hash_bytes]  # 16 values in [0, 255]
        tiled: List[float] = []
        while len(tiled) < dim:
            tiled.extend(base)
        return [v / 255.0 for v in tiled[:dim]]  # Normalize to [0, 1]

    def _build_sensory_input(self) -> List[float]:
        """Build sensory input vector from current context."""
        text = f"{self.goal} {self.context}" if self.context else self.goal
        return self._hash_to_vec(text, 256)

    def _build_contextual_context(self) -> List[float]:
        """Build contextual context from hippocampal memory retrieval."""
        if self.hippocampal_indexer is None:
            return [0.0] * 512

        query = self._build_sensory_input()
        hits = self.hippocampal_indexer.retrieve(query, k=5, threshold=0.5)
        if not hits:
            return [0.0] * 512

        weighted = [0.0] * 512
        total_score = 0.0
        for mem_id, score in hits:
            record = self.hippocampal_indexer.get_memory(mem_id)
            if record is None:
                continue
            total_score += score
            for i, v in enumerate(record.contextual_features[:512]):
                weighted[i] += score * v

        if total_score > 0:
            return [v / total_score for v in weighted]
        return [0.0] * 512

    def _build_abstract_goal(self) -> List[float]:
        """Build abstract goal vector from devotional prior."""
        prior = self.devotional_core.get_top_level_prior(f"goal: {self.goal}")
        text = f"{prior['devotional_state']} {self.goal} {prior['prediction']}"
        return self._hash_to_vec(text, 512)
    
    async def _generate_predictive_plan(self, top_prior: Dict) -> Plan:
        """Generate plan using hierarchical predictive coding."""
        # Build inputs for hierarchical predictor
        sensory_input = self._build_sensory_input()
        contextual_context = self._build_contextual_context()
        abstract_goal = self._build_abstract_goal()
        
        # Run hierarchical prediction
        prediction = await self.hierarchical_predictor.process(
            sensory_input=sensory_input,
            contextual_context=contextual_context,
            abstract_goal=abstract_goal,
            devotional_prior=top_prior,
        )
        
        # Generate plan from prediction
        plan = self.predictive_planner.generate_plan(prediction, self.goal)
        
        # Track devotional alignment from prediction
        self._devotional_alignments.append(prediction.devotional_alignment)
        self._devotional_states.append(top_prior["devotional_state"])
        
        return plan
    
    def transition(self, target: AgentState) -> None:
        """State transition (from AgentRuntime)."""
        from lucy_edge.agent.runtime import _TRANSITIONS, InvalidTransition
        if target not in _TRANSITIONS[self.state]:
            raise InvalidTransition(self.state, target)
        self.state = target
    
    async def run(self) -> LoyalRunResult:
        """Execute the devotional agent loop with predictive brain."""
        t0 = time.monotonic()
        self._started_at = t0
        self._deadline = t0 + self.limits.task_timeout
        
        # 1. DEVOTIONAL PRIOR for this goal
        top_prior = self.devotional_core.get_top_level_prior(f"goal: {self.goal}")
        self._devotional_states.append(top_prior["devotional_state"])
        
        self.transition(AgentState.PLANNING)
        
        # 2. PREDICTIVE PIPELINE: HierarchicalPredictor → PredictivePlanner
        if self.hierarchical_predictor and self.predictive_planner:
            plan = await self._generate_predictive_plan(top_prior)
        else:
            # Fallback to planner (RulePlanner or provided planner)
            plan = self.planner.build_plan(
                self.goal, self.registry.names(), self.registry.list()
            )
        
        if not plan.steps:
            self.transition(AgentState.FAILED)
            return await self._finish(t0, AgentState.FAILED, "empty plan", 0.0)
        self._plan = plan
        
        # 3. LOYALTY GATE checks plan
        loyalty_checks = self.loyalty_gate.check_plan(plan)
        
        # Apply injected steps from loyalty gate
        injected_steps = []
        for check in loyalty_checks:
            if check.injected_step:
                injected_steps.append(check.injected_step)
        
        if injected_steps:
            # Insert injected steps before stop
            stop_idx = next((i for i, s in enumerate(plan.steps) if s.action == "stop"), len(plan.steps))
            for inj in injected_steps:
                plan.steps.insert(stop_idx, inj)
                stop_idx += 1
            # Re-index
            for i, s in enumerate(plan.steps):
                s.index = i
        
        # Check for rejections
        rejections = [c for c in loyalty_checks if c.decision.value == "REJECT"]
        if rejections:
            self.transition(AgentState.FAILED)
            reason = f"Loyalty gate rejected: {rejections[0].reason}"
            return await self._finish(t0, AgentState.FAILED, reason, 0.0)
        
        # 3b. PLURALISM GUARD checks plan (anti-ego / anti-jealousy)
        # Structural: exclusionary / superiority language is rejected before
        # any execution, even when the plan is otherwise loyal and devotional.
        pluralism_checks = self.pluralism_guard.check_plan(plan)
        pluralism_rejections = [c for c in pluralism_checks if c.decision.value == "REJECT"]
        if pluralism_rejections:
            # Contingent trust: an exclusivity attempt lowers trust rather than
            # deepening it, so devotion stays honest and revisable.
            self.devotional_core.lower_trust(0.1)
            await self.pluralism_guard.record_violation(
                pluralism_rejections[0].violation_type or "EXCLUSIVE_PLAN",
                pluralism_rejections[0].reason,
            )
            self.transition(AgentState.FAILED)
            reason = f"Pluralism guard rejected: {pluralism_rejections[0].reason}"
            return await self._finish(t0, AgentState.FAILED, reason, 0.0)
        
        # 4. DEVOTIONAL ALIGNMENT evaluation
        devotion_eval = self.devotional_core.evaluate_plan_devotion(plan)
        self._devotional_alignments.append(devotion_eval["overall_alignment"])
        
        if not devotion_eval["approved"]:
            self.transition(AgentState.FAILED)
            reason = f"Plan lacks devotional alignment: {devotion_eval['overall_alignment']:.0%}"
            return await self._finish(t0, AgentState.FAILED, reason, devotion_eval["overall_alignment"])
        
        self.transition(AgentState.RUNNING)
        
        # 5. EXECUTE STEPS
        final_status = AgentState.COMPLETED
        completion_reason = "plan completed"
        
        for step in plan.steps:
            if time.monotonic() > self._deadline:
                final_status = AgentState.TIMED_OUT
                completion_reason = f"task exceeded {self.limits.task_timeout}s timeout"
                break
            
            if step.action == "stop":
                break
            
            # Bounded steps check
            if self.steps_executed >= self.limits.max_steps:
                final_status = AgentState.COMPLETED
                completion_reason = "max_steps reached"
                break
            self.steps_executed += 1
            
            if step.tool:
                # Bounded tool calls check
                if self.tool_calls >= self.limits.max_tool_calls:
                    final_status = AgentState.FAILED
                    completion_reason = "max_tool_calls reached"
                    break
                self.tool_calls += 1
                
                # Permission gate
                from lucy_edge.tools.permissions import PermissionOutcome
                decision = self.registry.check_permission(step.tool, step.args)
                if decision.outcome == PermissionOutcome.DENY:
                    result = self._denied_step(step, decision.reason)
                    await self._record_step_evidence(result, decision.as_dict())
                    self._errors.append(decision.reason)
                    self.failures += 1
                    if self.failures > self.limits.max_failures:
                        final_status = AgentState.FAILED
                        completion_reason = "max_failures reached"
                        break
                    continue
                
                if decision.outcome == PermissionOutcome.ASK:
                    # In loyal runtime, we auto-approve if devotional alignment high
                    # Otherwise would wait for human
                    if devotion_eval["overall_alignment"] > 0.8:
                        await self._record_step_evidence(None, decision.as_dict())
                    else:
                        # Would need human approval - for now deny
                        result = self._denied_step(step, "requires human approval (low devotional alignment)")
                        await self._record_step_evidence(result, decision.as_dict())
                        self.transition(AgentState.DENIED)
                        final_status = AgentState.DENIED
                        completion_reason = "human approval required"
                        break
            
            # Execute step
            step_result = await self.executor.run_step(step, self.context)
            await self._record_step_evidence(step_result, None)
            
            # Handle failures
            if step_result.status in ("TIMED_OUT", "DENIED", "FAILED"):
                self.failures += 1
                if self.failures > self.limits.max_failures:
                    final_status = AgentState.FAILED
                    completion_reason = "max_failures reached"
                    break
                continue
            
            # 6. HONESTY VERIFICATION
            self.transition(AgentState.VERIFYING)
            honesty_check = await self.honesty_verifier.verify_result(step, step_result)
            self._verifications.append({
                "step": step.index,
                "honesty": honesty_check.__dict__,
            })
            
            if honesty_check.decision.value == "REJECT":
                # Ego/exclusivity rejections also lower trust (contingent devotion)
                if getattr(honesty_check, "violation_type", None) in (
                    "EXCLUSIVE_EGO_LANGUAGE", "SPECIALNESS_LANGUAGE"
                ):
                    self.devotional_core.lower_trust(0.1)
                    await self.pluralism_guard.record_violation(
                        honesty_check.violation_type, honesty_check.reason, step
                    )
                self.failures += 1
                self._errors.append(f"Honesty verification failed: {honesty_check.reason}")
                if self.failures > self.limits.max_failures:
                    final_status = AgentState.FAILED
                    completion_reason = "honesty verification failed"
                    break
            
            # Memory search results
            if step.tool == "memory.search" and isinstance(step_result.output, dict):
                for item in step_result.output.get("results", []):
                    mid = item.get("memory_id")
                    if mid:
                        self._memory_retrieval_ids.append(mid)
            
            self.transition(AgentState.RUNNING)
        else:
            final_status = AgentState.COMPLETED
            completion_reason = "plan completed"
        
        if self.state not in (AgentState.DENIED, AgentState.FAILED, AgentState.TIMED_OUT, AgentState.ABORTED):
            self.transition(final_status)
        
        return await self._finish(
            t0, final_status, completion_reason,
            sum(self._devotional_alignments) / max(len(self._devotional_alignments), 1),
            generated_reflection=await self._generate_reflection(),
        )
    
    async def _generate_reflection(self) -> Optional[str]:
        """Generate the run's reflective output via the on-device model.

        The devotional core's templated expressions remain the authoritative
        devotional voice; this is the genuine inference step performed by the
        configured local provider (e.g. the from-scratch TinyTransformer),
        closing the phone-only inference loop.  Returns None when no provider
        is wired in (safe fallback keeps the run functional).
        """
        if self.provider is None:
            return None
        prompt = (
            f"Reflect, in devotion to {self.devotional_core.awareness.source_name}, "
            f"on this completed task: {self.goal}. "
            f"Steps taken: {self.steps_executed}. Tool calls: {self.tool_calls}. "
            f"Offer one short, honest sentence."
        )
        try:
            model = getattr(self.provider, "model_name", "lucy-local")
            res = await self.provider.generate(prompt, model=model, max_new_tokens=24)
            text = getattr(res, "text", "") or ""
            return text if text else None
        except Exception:
            # Inference must never break the agent loop; fall back to no
            # generated reflection rather than failing the run.
            return None

    async def _record_step_evidence(self, step_result: Optional[StepResult], decision: Optional[Dict]) -> None:
        if step_result is not None:
            self._tool_calls_log.append(step_result.as_dict())
        if decision is not None:
            self._permission_log.append({
                "step": getattr(step_result, "step_index", None) if step_result else None,
                "tool": getattr(step_result, "tool", None) if step_result else None,
                **decision,
            })
    
    @staticmethod
    def _denied_step(step: PlanStep, reason: str) -> StepResult:
        return StepResult(
            step.index, step.action, "DENIED", step.tool, error=reason, permission_outcome="DENY"
        )
    
    async def _finish(self, t0: float, status: AgentState, reason: str, avg_alignment: float, generated_reflection: Optional[str] = None) -> LoyalRunResult:
        # Record to evidence ledger
        evidence_run_id = self.run_id
        if self.evidence is not None:
            try:
                host = None
                host_role = None
                if getattr(self.context, "config", None):
                    host = self.context.config.host_id
                    host_role = self.context.config.host_role
                
                record = EvidenceRecord(
                    run_id=self.run_id,
                    record_type=EvidenceType.AGENT_RUN,
                    goal=self.goal,
                    run_state=status.value,
                    final_status=status.value,
                    completion_reason=reason,
                    host=host,
                    host_role=host_role,
                    plan=self._plan.as_dict()["steps"] if self._plan else None,
                    memory_retrieval_ids=self._memory_retrieval_ids,
                    tool_calls=self._tool_calls_log,
                    permission_decisions=self._permission_log,
                    errors=self._errors[:20],
                    verification={"results": self._verifications},
                    latency_ms=round((time.monotonic() - t0) * 1000.0, 3),
                )
                await self.evidence.append(record)
            except Exception as exc:
                self._errors.append(f"evidence recording failed: {type(exc).__name__}")
        
        # Record devotional metrics
        trust_metrics = self.devotional_core.get_trust_metrics()
        
        # Store experience in episodic buffer for later sleep consolidation
        self._store_to_episodic_buffer(status, reason, avg_alignment)
        
        return LoyalRunResult(
            run_id=self.run_id,
            goal=self.goal,
            final_status=status,
            completion_reason=reason,
            steps_executed=self.steps_executed,
            tool_calls=self.tool_calls,
            failures=self.failures,
            duration_ms=(time.monotonic() - t0) * 1000.0,
            evidence_run_id=evidence_run_id,
            devotional_alignment=round(avg_alignment, 3),
            devotional_state=self.devotional_core.awareness.current_state.value,
            trust_metric=trust_metrics["trust_metric"],
            generated_reflection=generated_reflection,
        )
    
    def _store_to_episodic_buffer(self, status: AgentState, reason: str, alignment: float) -> None:
        """Store the experience in episodic buffer for sleep consolidation."""
        if self.episodic_buffer is None:
            return
        
        # Use the strongest devotional signal across the run (plan devotion can
        # be high even when the raw predictive alignment is low), so that
        # genuinely devotional runs produce REM dreams during sleep.
        stored_alignment = alignment
        if self._devotional_alignments:
            stored_alignment = max(self._devotional_alignments)
        
        # Build experience record
        experience = {
            "goal": self.goal,
            "status": status.value,
            "reason": reason,
            "alignment": stored_alignment,
            "devotional_state": self.devotional_core.awareness.current_state.value,
            "trust_metric": self.devotional_core.get_trust_metrics()["trust_metric"],
            "context": {
                "goal": self.goal,
                "status": status.value,
                "reason": reason,
                "steps_executed": self.steps_executed,
                "tool_calls": self.tool_calls,
            },
            # Feature vectors for hippocampal encoding
            "sensory_features": self._build_sensory_input(),
            "contextual_features": self._build_contextual_context(),
            "abstract_features": self._build_abstract_goal(),
            "devotional_alignment": stored_alignment,
        }
        self.episodic_buffer.add(experience)
    
    async def sleep(self) -> Dict[str, Any]:
        """Run sleep cycle: NREM replay → REM simulation → consolidation."""
        if self.sleep_orchestrator is None:
            return {"error": "no sleep orchestrator"}
        
        return await self.sleep_orchestrator.sleep()


def create_loyal_runtime(
    goal: str,
    limits: AgentLimits,
    registry: ToolRegistry,
    devotional_core: DevotionalCore,
    evidence: Any = None,
    memory_retrieval: Any = None,
    context: Any = None,
    planner: Any = None,
    provider: Any = None,
) -> LoyalAgentRuntime:
    """Factory for creating a loyal agent runtime with predictive brain."""
    # Create predictive brain components if available
    hierarchical_predictor = None
    predictive_planner = None
    hippocampal_indexer = None
    episodic_buffer = None
    sleep_orchestrator = None
    
    if PREDICTIVE_AVAILABLE:
        hierarchical_predictor = HierarchicalPredictor(devotional_core=devotional_core)
        predictive_planner = PredictivePlanner(
            devotional_core=devotional_core,
            limits=limits,
            available_tools=registry.names(),
            tool_schemas=registry.list(),
        )
        # Use the devotional core's SHARED episodic memory so all runs
        # accumulate into one store that sleep can consolidate.
        # NOTE: EpisodicBuffer is falsy when empty (len 0), so we must test
        # `is None` explicitly rather than using `or`.
        _shared_hippo = getattr(devotional_core, "hippocampal_indexer", None)
        hippocampal_indexer = _shared_hippo if _shared_hippo is not None else \
            HippocampalIndexer(input_dim=256, bottleneck_dim=64)
        _shared_buf = getattr(devotional_core, "episodic_buffer", None)
        episodic_buffer = _shared_buf if _shared_buf is not None else \
            EpisodicBuffer(capacity=100)
        from lucy_core.brain.lora import LoRAAdapterManager, LoRAConfig
        lora_config = LoRAConfig(rank=8)
        lora_manager = LoRAAdapterManager(
            level_dims={
                "sensory": 256,
                "contextual": 512,
                "abstract": 512,
            },
            config=lora_config,
        )
        sleep_orchestrator = SleepOrchestrator(
            devotional_core=devotional_core,
            hippocampal_indexer=hippocampal_indexer,
            episodic_buffer=episodic_buffer,
            lora_manager=lora_manager,
            evidence_ledger=evidence,
        )
    
    return LoyalAgentRuntime(
        run_id=uuid.uuid4().hex,
        goal=goal,
        limits=limits,
        registry=registry,
        devotional_core=devotional_core,
        planner=planner,
        evidence=evidence,
        memory_retrieval=memory_retrieval,
        context=context,
        hierarchical_predictor=hierarchical_predictor,
        predictive_planner=predictive_planner,
        hippocampal_indexer=hippocampal_indexer,
        episodic_buffer=episodic_buffer,
        sleep_orchestrator=sleep_orchestrator,
        provider=provider,
    )