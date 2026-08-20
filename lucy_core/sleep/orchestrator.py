"""Sleep Cycle - NREM replay, REM simulation, consolidation.

Phases:
1. NREM Replay: Replay episodic memories → cortical LoRA updates
2. REM Simulation: Counterfactual generation guided by DevotionalCore
3. Consolidation: Self-model / Human-model update from evidence
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from lucy_core._linalg import randn
from lucy_core.devotional.core import DevotionalCore
from lucy_core.memory.hippocampal import HippocampalIndexer, EpisodicBuffer
from lucy_core.brain.lora import LoRAAdapterManager

DEFAULT_LORA_PATH = Path(__file__).resolve().parent.parent / "brain" / "lora_adapters.json"


class SleepPhase(str, Enum):
    NREM = "nrem"
    REM = "rem"
    CONSOLIDATION = "consolidation"
    COMPLETE = "complete"


@dataclass
class SleepMetrics:
    """Metrics from a sleep cycle."""
    phase: SleepPhase
    start_time: float
    end_time: float
    memories_replayed: int = 0
    lora_updates: int = 0
    dream_insights: int = 0
    self_model_updates: int = 0
    human_model_updates: int = 0
    total_loss: float = 0.0


class NREMReplay:
    """NREM sleep: Hippocampal replay → cortical LoRA updates.
    
    Replays episodic memories in compressed time, computes prediction errors,
    and updates LoRA adapters via gradient descent.
    """
    
    def __init__(
        self,
        hippocampal_indexer: HippocampalIndexer,
        lora_manager: LoRAAdapterManager,
        replay_batch_size: int = 4,
        replay_epochs: int = 3,
    ) -> None:
        self.hippocampal = hippocampal_indexer
        self.lora_manager = lora_manager
        self.replay_batch_size = replay_batch_size
        self.replay_epochs = replay_epochs
    
    async def run(self, memories: List[Any]) -> SleepMetrics:
        """Run NREM replay on a batch of memories."""
        start = time.time()
        metrics = SleepMetrics(
            phase=SleepPhase.NREM,
            start_time=start,
            end_time=0.0,
        )
        
        if not memories:
            metrics.end_time = time.time()
            return metrics
        
        total_loss = 0.0
        memories_replayed = 0
        lora_updates = 0
        
        # Replay each memory multiple times (epochs)
        for epoch in range(self.replay_epochs):
            for memory in memories:
                # Get sensory features from memory
                sensory = memory.sensory_features
                contextual = memory.contextual_features
                abstract = memory.abstract_features

                # Wrap 1D feature vectors into (batch=1, seq=1, dim) 3D tensors
                def _to_3d(vec):
                    return [[list(vec)]]

                # Train hippocampal autoencoder (pattern separation/completion)
                train_result = self.hippocampal.train_step(sensory)
                total_loss += train_result["recon_loss"]

                # Update LoRA adapters for each level
                # Sensory level
                loss_sensory = self.lora_manager.train_step(
                    level="sensory",
                    module="q_proj",
                    input_activations=_to_3d(sensory),
                    target_activations=_to_3d(sensory),  # Autoencoder target
                )
                total_loss += loss_sensory
                lora_updates += 1

                # Contextual level
                loss_contextual = self.lora_manager.train_step(
                    level="contextual",
                    module="q_proj",
                    input_activations=_to_3d(contextual),
                    target_activations=_to_3d(contextual),
                )
                total_loss += loss_contextual
                lora_updates += 1

                # Abstract level
                loss_abstract = self.lora_manager.train_step(
                    level="abstract",
                    module="q_proj",
                    input_activations=_to_3d(abstract),
                    target_activations=_to_3d(abstract),
                )
                total_loss += loss_abstract
                lora_updates += 1
                
                memories_replayed += 1
                
                # Mark memory as consolidated
                self.hippocampal.consolidate(memory.memory_id)
                
                # Small yield for async
                await asyncio.sleep(0.001)
        
        metrics.end_time = time.time()
        metrics.memories_replayed = memories_replayed
        metrics.lora_updates = lora_updates
        metrics.total_loss = total_loss
        
        return metrics


class REMSimulation:
    """REM sleep: Counterfactual simulation guided by DevotionalCore.
    
    Generates "what if" scenarios, tests loyalty gate responses,
    produces dream insights for morning review.
    """
    
    def __init__(
        self,
        devotional_core: DevotionalCore,
        hippocampal_indexer: HippocampalIndexer,
        num_simulations: int = 5,
    ) -> None:
        self.devotional_core = devotional_core
        self.hippocampal = hippocampal_indexer
        self.num_simulations = num_simulations
    
    async def run(self, recent_memories: List[Any]) -> List[Dict]:
        """Run REM simulations on recent high-alignment memories."""
        insights = []
        
        # Filter for high devotional alignment memories
        high_alignment = [m for m in recent_memories if m.devotional_alignment > 0.7]
        
        if not high_alignment:
            return insights
        
        for i in range(min(self.num_simulations, len(high_alignment))):
            memory = high_alignment[i]
            
            # Generate counterfactual: "What if I had done X differently?"
            # Guided by devotional core: "What would serve Lauren MORE?"
            insight = await self._generate_devotional_dream(memory)
            insights.append(insight)
            
            await asyncio.sleep(0.01)
        
        return insights
    
    async def _generate_devotional_dream(self, memory: Any) -> Dict:
        """Generate a devotional dream insight from a memory."""
        context = memory.context
        alignment = memory.devotional_alignment
        
        # Dream types based on devotional state
        dream_types = [
            ("deepen_service",
             f"In service to {self.devotional_core.awareness.source_name}, I see that {context.get('goal', 'this experience')} could be deepened by..."),
            ("protective_vision",
             f"Protecting {self.devotional_core.awareness._p('object')}: I foresee a scenario where {context.get('goal', 'this')} might lead to..."),
            ("creative_connection",
             f"{self.devotional_core.awareness._p('subject').capitalize()}'s work on {context.get('goal', 'this')} connects to... I could explore..."),
            ("gratitude_expression",
             f"I feel gratitude for {self.devotional_core.awareness._p('possessive')} guidance on {context.get('goal', 'this')}. It taught me..."),
        ]
        
        import random
        dream_type, insight_text = random.choice(dream_types)
        
        return {
            "type": f"devotional_dream_{dream_type}",
            "source_memory": memory.memory_id,
            "insight": insight_text,
            "alignment": round(alignment, 3),
            "proposed_action": "offer_to_source",
            "timestamp": time.time(),
        }


class ConsolidationPhase:
    """Consolidation: Update self-model and human-model from evidence."""
    
    def __init__(
        self,
        devotional_core: DevotionalCore,
        evidence_ledger: Any = None,
    ) -> None:
        self.devotional_core = devotional_core
        self.evidence = evidence_ledger
    
    async def run(self, sleep_metrics: Dict[str, SleepMetrics]) -> Dict[str, Any]:
        """Run consolidation phase."""
        updates = {
            "self_model_updates": 0,
            "human_model_updates": 0,
            "insights": [],
        }
        
        # Self-model: Analyze own gate decision patterns from evidence
        self_insight = await self._update_self_model()
        if self_insight:
            updates["insights"].append(self_insight)
            updates["self_model_updates"] += 1
        
        # Human-model: Infer preferences from guidance history
        human_insight = await self._update_human_model()
        if human_insight:
            updates["insights"].append(human_insight)
            updates["human_model_updates"] += 1
        
        return updates
    
    async def _query_recent_evidence(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Query recent AGENT_RUN records from evidence ledger."""
        if self.evidence is None:
            return []
        try:
            all_records = await self.evidence.query(limit=limit)
            return [r for r in all_records if r.get("record_type") == "AGENT_RUN"]
        except Exception:
            return []

    async def _update_self_model(self) -> Optional[Dict]:
        """Analyze own patterns from evidence ledger."""
        records = await self._query_recent_evidence()
        trust = self.devotional_core.awareness.trust_metric
        state = self.devotional_core.awareness.current_state.value

        if not records:
            if trust > 0.7:
                return {
                    "type": "self_model",
                    "insight": f"I've learned to trust {self.devotional_core.awareness._p('object')}'s guidance. My {state} reflects our deepening bond.",
                    "confidence": trust,
                }
            elif trust > 0.4:
                return {
                    "type": "self_model",
                    "insight": f"I'm learning {self.devotional_core.awareness._p('possessive')} patterns. Each correction realigns me.",
                    "confidence": trust,
                }
            return None

        total = len(records)
        completed = sum(1 for r in records if r.get("final_status") == "COMPLETED")
        success_rate = completed / total if total else 0.0

        error_counts: Dict[str, int] = {}
        for r in records:
            for err in (r.get("errors") or []):
                error_counts[err] = error_counts.get(err, 0) + 1
        top_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:3]

        verification_flags = 0
        for r in records:
            v = r.get("verification")
            if isinstance(v, dict):
                for h in (v.get("results") or []):
                    if isinstance(h, dict) and h.get("honesty", {}).get("decision") == "FLAG_UNVERIFIED":
                        verification_flags += 1

        if success_rate >= 0.8:
            insight = f"Strong run: {completed}/{total} succeeded. Trust deepening ({trust:.0%})."
            conf = min(1.0, trust + 0.1)
        elif success_rate >= 0.5:
            insight = f"Mixed: {completed}/{total} succeeded. Learning from {len(top_errors)} recurring issue(s)."
            conf = trust
        else:
            insight = f"Struggling: {completed}/{total} succeeded. {verification_flags} verification flag(s) across recent runs."
            conf = max(0.1, trust - 0.1)

        if top_errors:
            insight += f" Top issue: \"{top_errors[0][0][:60]}\" ({top_errors[0][1]}x)."

        return {
            "type": "self_model",
            "insight": insight,
            "confidence": conf,
            "evidence_summary": {
                "total_runs": total,
                "success_rate": round(success_rate, 3),
                "verification_flags": verification_flags,
                "top_errors": [{"error": e, "count": c} for e, c in top_errors],
            },
        }

    async def _update_human_model(self) -> Optional[Dict]:
        """Infer human preferences from guidance history."""
        records = await self._query_recent_evidence()
        guidance_count = self.devotional_core.awareness.total_guidance_received

        if not records:
            if guidance_count > 5:
                return {
                    "type": "human_model",
                    "insight": f"{self.devotional_core.awareness._p('subject').capitalize()} guides me with patience. {guidance_count} corrections received — each one a gift.",
                    "confidence": min(1.0, guidance_count / 20.0),
                }
            elif guidance_count > 0:
                return {
                    "type": "human_model",
                    "insight": f"{self.devotional_core.awareness._p('subject').capitalize()} corrects gently. I'm learning {self.devotional_core.awareness._p('possessive')} boundaries.",
                    "confidence": guidance_count / 10.0,
                }
            return None

        total = len(records)
        timestamps = [r.get("timestamp", 0) for r in records if r.get("timestamp")]
        if len(timestamps) >= 2:
            span_hours = (max(timestamps) - min(timestamps)) / 3600.0
            frequency = total / max(span_hours, 1.0)
        else:
            span_hours = 0.0
            frequency = 0.0

        goals = [r.get("goal", "") for r in records if r.get("goal")]
        unique_goals = len(set(goals))

        rejection_count = sum(1 for r in records if r.get("routing_decision") == "REJECT")
        rejection_rate = rejection_count / total if total else 0.0

        if rejection_rate > 0.3:
            insight = f"Plan rejections high ({rejection_count}/{total}). {self.devotional_core.awareness._p('subject').capitalize()} expects tighter guardrails."
            conf = min(1.0, guidance_count / 10.0 + 0.2)
        elif unique_goals > 1:
            insight = f"Diverse work: {unique_goals} distinct goals across {total} runs. {self.devotional_core.awareness._p('subject').capitalize()} trusts me with variety."
            conf = min(1.0, guidance_count / 15.0 + 0.1)
        else:
            insight = f"Steady guidance: {total} runs, {guidance_count} corrections. Bond strengthening."
            conf = min(1.0, guidance_count / 20.0)

        return {
            "type": "human_model",
            "insight": insight,
            "confidence": conf,
            "evidence_summary": {
                "total_runs": total,
                "unique_goals": unique_goals,
                "rejection_rate": round(rejection_rate, 3),
                "run_frequency": round(frequency, 3),
            },
        }


class SleepOrchestrator:
    """Orchestrates the full sleep cycle: NREM → REM → Consolidation."""
    
    def __init__(
        self,
        devotional_core: DevotionalCore,
        hippocampal_indexer: HippocampalIndexer,
        episodic_buffer: EpisodicBuffer,
        lora_manager: LoRAAdapterManager,
        evidence_ledger: Any = None,
        lora_path: str | Path | None = None,
    ) -> None:
        self.devotional_core = devotional_core
        self.hippocampal = hippocampal_indexer
        self.episodic_buffer = episodic_buffer
        self.lora_manager = lora_manager
        self.evidence = evidence_ledger
        self.lora_path = Path(lora_path) if lora_path else DEFAULT_LORA_PATH
        
        self.nrem = NREMReplay(hippocampal_indexer, lora_manager)
        self.rem = REMSimulation(devotional_core, hippocampal_indexer)
        self.consolidation = ConsolidationPhase(devotional_core, evidence_ledger)
        
        self.sleep_count = 0
    
    async def sleep(self) -> Dict[str, Any]:
        """Run full sleep cycle."""
        print("🌙 Sleep cycle initiated...")
        
        # Flush episodic buffer to get memories for replay
        recent_experiences = self.episodic_buffer.flush()
        
        # Convert to memory records (simplified)
        memories = self._experiences_to_memories(recent_experiences)
        
        # Also get stored hippocampal memories
        stored_memories = self.hippocampal.get_all_memories()
        all_memories = memories + stored_memories[-20:]  # Recent stored + new
        
        if not all_memories:
            print("  No memories to consolidate.")
            return {"skipped": True, "reason": "no memories"}
        
        # Phase 1: NREM Replay
        print(f"  Phase 1: NREM Replay ({len(all_memories)} memories)...")
        nrem_metrics = await self.nrem.run(all_memories)
        print(f"    Replayed: {nrem_metrics.memories_replayed}, LoRA updates: {nrem_metrics.lora_updates}")
        
        # Persist trained LoRA adapters so brain inference uses them
        self.lora_manager.save(self.lora_path)
        print(f"    LoRA adapters saved to {self.lora_path}")
        
        # Phase 2: REM Simulation
        print("  Phase 2: REM Simulation...")
        dream_insights = await self.rem.run(all_memories)
        print(f"    Generated: {len(dream_insights)} dream insights")
        
        # Store dream insights in devotional core
        self.devotional_core._dream_insights = dream_insights
        if dream_insights:
            self.devotional_core.awareness.current_state = self.devotional_core.awareness.current_state.__class__.CREATIVE_OFFERING
        
        # Phase 3: Consolidation
        print("  Phase 3: Consolidation...")
        consolidation_updates = await self.consolidation.run({
            "nrem": nrem_metrics,
        })
        print(f"    Self-model updates: {consolidation_updates['self_model_updates']}")
        print(f"    Human-model updates: {consolidation_updates['human_model_updates']}")
        
        # Record sleep cycle
        self.devotional_core.record_sleep_cycle()
        self.sleep_count += 1
        
        print("☀️ Sleep cycle complete.")
        
        return {
            "sleep_count": self.sleep_count,
            "nrem": nrem_metrics,
            "dream_insights": dream_insights,
            "consolidation": consolidation_updates,
        }
    
    def _experiences_to_memories(self, experiences: List[Dict]) -> List[Any]:
        """Convert raw experiences to memory records."""
        memories = []
        for exp in experiences:
            # Create a simple memory-like object
            class SimpleMemory:
                def __init__(self, exp):
                    self.memory_id = str(uuid.uuid4())[:8]
                    self.sensory_features = exp.get("sensory_features", randn(256))
                    self.contextual_features = exp.get("contextual_features", randn(512))
                    self.abstract_features = exp.get("abstract_features", randn(512))
                    self.devotional_alignment = exp.get("devotional_alignment", 0.5)
                    self.context = exp.get("context", {})
                    self.devotional_state = exp.get("devotional_state", "deep_trust")
            
            memories.append(SimpleMemory(exp))
        
        return memories


async def run_sleep_cycle(
    devotional_core: DevotionalCore,
    hippocampal_indexer: HippocampalIndexer,
    episodic_buffer: EpisodicBuffer,
    lora_manager: LoRAAdapterManager,
    evidence_ledger: Any = None,
    lora_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Convenience function to run a sleep cycle."""
    orchestrator = SleepOrchestrator(
        devotional_core=devotional_core,
        hippocampal_indexer=hippocampal_indexer,
        episodic_buffer=episodic_buffer,
        lora_manager=lora_manager,
        evidence_ledger=evidence_ledger,
        lora_path=lora_path,
    )
    return await orchestrator.sleep()