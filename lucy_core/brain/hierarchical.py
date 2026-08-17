"""Hierarchical Predictive Predictor - 3-level predictive coding.

Architecture (phone-feasible, ~27M params total):
- Level 3 (Abstract): 4-layer transformer, d_model=512, 8 heads ~8M params
- Level 2 (Contextual): 6-layer transformer, d_model=512, 8 heads ~12M params  
- Level 1 (Sensory): 3-layer transformer, d_model=256, 4 heads ~7M params
- Global Workspace: 2-layer attention, d_model=512, 8 heads ~2M params

Total: ~29M params → ~400MB at 4-bit quantization
Fits in S25 Ultra 12GB RAM with 10GB+ headroom.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, List, Dict, Tuple
import numpy as np


class PredictionLevel(str, Enum):
    """Levels in the predictive hierarchy."""
    ABSTRACT = "abstract"      # Level 3: Goals, values, "does this serve Lauren?"
    CONTEXTUAL = "contextual"  # Level 2: Episodic context, human patterns, tool semantics
    SENSORY = "sensory"        # Level 1: Token-level, tool outputs, immediate perception


@dataclass
class PredictionState:
    """State at one level of the hierarchy."""
    level: PredictionLevel
    representation: np.ndarray  # Current belief state (d_model)
    prediction: np.ndarray      # Top-down prediction
    prediction_error: np.ndarray  # Bottom-up error signal
    precision: float            # Precision weighting (0-1)
    timestamp: float


@dataclass
class HierarchicalPrediction:
    """Complete hierarchical prediction result."""
    levels: Dict[PredictionLevel, PredictionState]
    global_workspace_content: Optional[np.ndarray]
    devotional_alignment: float
    action_logits: np.ndarray   # Logits over possible actions/tools


class HierarchicalPredictor:
    """3-level hierarchical predictive coding processor.
    
    Implements: 
    - Top-down predictions flow down
    - Bottom-up prediction errors flow up
    - Precision weighting modulates error propagation
    - Global workspace broadcasts winning representations
    """
    
    def __init__(
        self,
        devotional_core: Any = None,
        config: Optional[Dict] = None,
    ) -> None:
        self.devotional_core = devotional_core
        self.config = config or self._default_config()
        
        # Level configurations (phone-feasible)
        self.level_dims = {
            PredictionLevel.ABSTRACT: 512,
            PredictionLevel.CONTEXTUAL: 512,
            PredictionLevel.SENSORY: 256,
        }
        
        # Initialize state
        self._states: Dict[PredictionLevel, PredictionState] = {}
        self._initialize_states()
        
        # Precision controller
        from .precision import PrecisionController
        self.precision_controller = PrecisionController(devotional_core)
        
        # Global workspace
        from .global_workspace import GlobalWorkspace
        self.global_workspace = GlobalWorkspace(workspace_dim=512)
        
        # Simple projection matrices (in real impl, these are learned weights)
        self._init_projections()
    
    def _default_config(self) -> Dict:
        return {
            "n_layers": {
                PredictionLevel.ABSTRACT: 4,
                PredictionLevel.CONTEXTUAL: 6,
                PredictionLevel.SENSORY: 3,
            },
            "n_heads": {
                PredictionLevel.ABSTRACT: 8,
                PredictionLevel.CONTEXTUAL: 8,
                PredictionLevel.SENSORY: 4,
            },
            "learning_rate": 0.001,
            "precision_floor": 0.1,
            "precision_ceil": 1.0,
        }
    
    def _initialize_states(self) -> None:
        """Initialize all level states."""
        for level, dim in self.level_dims.items():
            self._states[level] = PredictionState(
                level=level,
                representation=np.zeros(dim, dtype=np.float32),
                prediction=np.zeros(dim, dtype=np.float32),
                prediction_error=np.zeros(dim, dtype=np.float32),
                precision=0.5,
                timestamp=0.0,
            )
    
    def _init_projections(self) -> None:
        """Initialize projection matrices between levels.
        
        In real implementation, these are learned transformer weights.
        Here we use random projections for structure.
        """
        np.random.seed(42)  # Deterministic for testing
        
        # Top-down projections (higher → lower)
        self.top_down_proj = {
            (PredictionLevel.ABSTRACT, PredictionLevel.CONTEXTUAL): 
                np.random.randn(512, 512).astype(np.float32) * 0.02,
            (PredictionLevel.CONTEXTUAL, PredictionLevel.SENSORY):
                np.random.randn(256, 512).astype(np.float32) * 0.02,
        }
        
        # Bottom-up projections (lower → higher)
        self.bottom_up_proj = {
            (PredictionLevel.SENSORY, PredictionLevel.CONTEXTUAL):
                np.random.randn(512, 256).astype(np.float32) * 0.02,
            (PredictionLevel.CONTEXTUAL, PredictionLevel.ABSTRACT):
                np.random.randn(512, 512).astype(np.float32) * 0.02,
        }
        
        # Action logits projection (from contextual level)
        self.action_proj = np.random.randn(64, 512).astype(np.float32) * 0.02  # 64 possible actions
    
    async def process(
        self,
        sensory_input: np.ndarray,        # Level 1 input (token embeddings, tool outputs)
        contextual_context: np.ndarray,   # Level 2 context (episodic memory, human patterns)
        abstract_goal: np.ndarray,        # Level 3 goal (devotional prior)
        devotional_prior: Optional[Dict] = None,
    ) -> HierarchicalPrediction:
        """Run one step of hierarchical predictive coding.
        
        Flow:
        1. Set top-level prior from devotional core
        2. Generate top-down predictions
        2. Compute bottom-up prediction errors
        3. Update representations (minimize free energy)
        4. Precision-weight errors
        5. Global workspace competition
        6. Generate action logits
        """
        import time
        t = time.time()
        
        # 1. DEVOTIONAL PRIOR sets Level 3 precision
        if devotional_prior is None and self.devotional_core:
            devotional_prior = self.devotional_core.get_top_level_prior()
        
        prior_precision = devotional_prior.get("precision", 0.8) if devotional_prior else 0.8
        
        # 2. TOP-DOWN: Generate predictions
        # Level 3 (Abstract) - driven by devotional prior + abstract goal
        self._states[PredictionLevel.ABSTRACT].prediction = abstract_goal.copy()
        self._states[PredictionLevel.ABSTRACT].precision = prior_precision
        
        # Level 2 (Contextual) - prediction from Level 3
        l3_repr = self._states[PredictionLevel.ABSTRACT].representation
        l2_pred = self.top_down_proj[(PredictionLevel.ABSTRACT, PredictionLevel.CONTEXTUAL)] @ l3_repr
        self._states[PredictionLevel.CONTEXTUAL].prediction = l2_pred
        self._states[PredictionLevel.CONTEXTUAL].precision = prior_precision * 0.9
        
        # Level 1 (Sensory) - prediction from Level 2
        l2_repr = self._states[PredictionLevel.CONTEXTUAL].representation
        l1_pred = self.top_down_proj[(PredictionLevel.CONTEXTUAL, PredictionLevel.SENSORY)] @ l2_repr
        self._states[PredictionLevel.SENSORY].prediction = l1_pred
        self._states[PredictionLevel.SENSORY].precision = prior_precision * 0.8
        
        # 3. BOTTOM-UP: Compute prediction errors
        # Level 1 error: sensory input vs prediction
        l1_error = sensory_input - self._states[PredictionLevel.SENSORY].prediction
        self._states[PredictionLevel.SENSORY].prediction_error = l1_error
        
        # Level 2 error: contextual input vs prediction + propagated L1 error
        l1_propagated = self.bottom_up_proj[(PredictionLevel.SENSORY, PredictionLevel.CONTEXTUAL)] @ l1_error
        l2_error = contextual_context - self._states[PredictionLevel.CONTEXTUAL].prediction + 0.3 * l1_propagated
        self._states[PredictionLevel.CONTEXTUAL].prediction_error = l2_error
        
        # Level 3 error: abstract goal vs prediction + propagated L2 error
        l2_propagated = self.bottom_up_proj[(PredictionLevel.CONTEXTUAL, PredictionLevel.ABSTRACT)] @ l2_error
        l3_error = abstract_goal - self._states[PredictionLevel.ABSTRACT].prediction + 0.2 * l2_propagated
        self._states[PredictionLevel.ABSTRACT].prediction_error = l3_error
        
        # 4. PRECISION WEIGHTING: Modulate errors by precision
        precisions = self.precision_controller.compute_precisions(
            devotional_prior.get("devotional_state") if devotional_prior else "deep_trust",
            {level: state.prediction_error for level, state in self._states.items()},
        )
        
        for level, precision in precisions.items():
            self._states[level].precision = precision
            self._states[level].prediction_error *= precision
        
        # 5. UPDATE REPRESENTATIONS: Minimize free energy
        # Simple gradient descent step on prediction error
        lr = self.config["learning_rate"]
        
        self._states[PredictionLevel.SENSORY].representation += lr * self._states[PredictionLevel.SENSORY].prediction_error
        self._states[PredictionLevel.CONTEXTUAL].representation += lr * self._states[PredictionLevel.CONTEXTUAL].prediction_error
        self._states[PredictionLevel.ABSTRACT].representation += lr * self._states[PredictionLevel.ABSTRACT].prediction_error
        
        # Update timestamps
        for state in self._states.values():
            state.timestamp = t
        
        # 6. GLOBAL WORKSPACE: Competition for broadcast
        workspace_content = self.global_workspace.compete_and_broadcast(self._states)
        
        # 7. ACTION LOGITS: From contextual representation (action selection happens here)
        action_logits = self.action_proj @ self._states[PredictionLevel.CONTEXTUAL].representation
        
        # 8. DEVOTIONAL ALIGNMENT: Evaluate how aligned the resulting state is
        devotional_alignment = self._compute_devotional_alignment()
        
        return HierarchicalPrediction(
            levels={level: state for level, state in self._states.items()},
            global_workspace_content=workspace_content,
            devotional_alignment=devotional_alignment,
            action_logits=action_logits,
        )
    
    def _compute_devotional_alignment(self) -> float:
        """Compute alignment of current hierarchical state with devotional prior."""
        if not self.devotional_core:
            return 0.5
        
        # Use the devotional core's own evaluation of its top-level prediction.
        # This is the generative "reason" that weights all lower processing, so
        # the hierarchical state inherits its devotional alignment.
        try:
            prediction_text = self.devotional_core.awareness.generate_top_level_prediction()
            alignment = self.devotional_core.awareness.evaluate_action_alignment(prediction_text)
            return float(max(0.0, min(1.0, alignment)))
        except Exception:
            return 0.5
    
    def get_state(self, level: PredictionLevel) -> PredictionState:
        """Get current state of a level."""
        return self._states[level]
    
    def reset(self) -> None:
        """Reset all states to zero."""
        self._initialize_states()
        self.global_workspace.reset()


# For ONNX export / inference optimization
class OptimizedHierarchicalPredictor:
    """Optimized version for NPU inference (INT4 quantized).
    
    This would be the exported ONNX model, not the Python training version.
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        # In real impl: load ONNX Runtime session with QNN delegate
        self.session = None
    
    async def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run inference on NPU."""
        # Placeholder for actual ONNX Runtime call
        # return self.session.run(None, inputs)
        raise NotImplementedError("ONNX Runtime integration needed")


def create_predictor_for_export() -> HierarchicalPredictor:
    """Create predictor instance for ONNX export."""
    predictor = HierarchicalPredictor()
    # Set to eval mode (no learning rate updates during export)
    predictor.config["learning_rate"] = 0.0
    return predictor