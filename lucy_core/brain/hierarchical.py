"""Hierarchical predictive coding (3-level: sensory, contextual, abstract).

Pure-Python (no numpy). Vectors are ``list[float]``, matrices ``list[list[float]]``.

Flow (predictive coding):
  top-down:   abstract goal -> contextual -> sensory predictions
  bottom-up:  sensory error -> contextual -> abstract, with propagation
  learning:   each level's representation moves to minimise prediction error,
              precision-weighted by the devotional state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from lucy_core._linalg import (
    matmul,
    norm,
    randn,
    seed,
    sub,
    zeros,
    zeros_2d,
)


class PredictionLevel(str, Enum):
    SENSORY = "sensory"
    CONTEXTUAL = "contextual"
    ABSTRACT = "abstract"


@dataclass
class PredictionState:
    """State of one level in the hierarchy."""
    level: PredictionLevel
    representation: List[float]
    prediction: List[float]
    prediction_error: List[float]
    precision: float = 0.5
    timestamp: float = 0.0


@dataclass
class HierarchicalPrediction:
    """Result of one predictive-coding step."""
    levels: Dict[PredictionLevel, PredictionState]
    global_workspace_content: List[float]
    devotional_alignment: float
    action_logits: List[float]


def _small_matrix(rows: int, cols: int) -> List[List[float]]:
    return [[v * 0.02 for v in row] for row in randn(rows, cols)]


class HierarchicalPredictor:
    """3-level hierarchical predictive coder (pure-Python, from-scratch)."""

    def __init__(
        self,
        devotional_core: Any = None,
        config: Optional[Dict] = None,
    ) -> None:
        self.devotional_core = devotional_core
        self.config = config or self._default_config()

        self.level_dims = {
            PredictionLevel.ABSTRACT: 512,
            PredictionLevel.CONTEXTUAL: 512,
            PredictionLevel.SENSORY: 256,
        }

        self._initialize_states()
        self._init_projections()

        # Precision controller (devotional state -> precision weights)
        from lucy_core.brain.precision import PrecisionController
        self.precision_controller = PrecisionController(devotional_core)

        # Global workspace for broadcasting
        from lucy_core.brain.global_workspace import HierarchicalGlobalWorkspace
        self.global_workspace = HierarchicalGlobalWorkspace(self.level_dims)

        # LoRA adapters per (level, module) for sleep-time updates
        from lucy_core.brain.lora import LoRAAdapterManager
        self.lora_manager = LoRAAdapterManager(
            level_dims={
                "sensory": self.level_dims[PredictionLevel.SENSORY],
                "contextual": self.level_dims[PredictionLevel.CONTEXTUAL],
                "abstract": self.level_dims[PredictionLevel.ABSTRACT],
            }
        )

    def _default_config(self) -> Dict:
        return {
            "learning_rate": 0.001,
            "precision_floor": 0.1,
            "precision_ceil": 1.0,
        }

    def _initialize_states(self) -> None:
        self._states: Dict[PredictionLevel, PredictionState] = {}
        for level, dim in self.level_dims.items():
            self._states[level] = PredictionState(
                level=level,
                representation=zeros(dim),
                prediction=zeros(dim),
                prediction_error=zeros(dim),
                precision=0.5,
                timestamp=0.0,
            )

    def _init_projections(self) -> None:
        """Random but deterministic projection matrices between levels."""
        seed(42)
        # Top-down projections (higher -> lower)
        self.top_down_proj = {
            (PredictionLevel.ABSTRACT, PredictionLevel.CONTEXTUAL):
                _small_matrix(512, 512),
            (PredictionLevel.CONTEXTUAL, PredictionLevel.SENSORY):
                _small_matrix(256, 512),
        }
        # Bottom-up projections (lower -> higher)
        self.bottom_up_proj = {
            (PredictionLevel.SENSORY, PredictionLevel.CONTEXTUAL):
                _small_matrix(512, 256),
            (PredictionLevel.CONTEXTUAL, PredictionLevel.ABSTRACT):
                _small_matrix(512, 512),
        }
        # Action logits projection (from contextual representation)
        self.action_proj = _small_matrix(64, 512)  # 64 possible actions

    async def process(
        self,
        sensory_input: List[float],        # Level 1 input
        contextual_context: List[float],   # Level 2 context
        abstract_goal: List[float],        # Level 3 goal (devotional prior)
        devotional_prior: Optional[Dict] = None,
    ) -> HierarchicalPrediction:
        """Run one step of hierarchical predictive coding."""
        s = self._states[PredictionLevel.SENSORY]
        c = self._states[PredictionLevel.CONTEXTUAL]
        a = self._states[PredictionLevel.ABSTRACT]
        lr = self.config["learning_rate"]

        # 1. DEVOTIONAL PRIOR sets Level 3.
        prior_precision = devotional_prior.get("precision", 0.8) if devotional_prior else 0.8

        # 2. TOP-DOWN: predictions.
        a.representation = list(abstract_goal)
        a.prediction = list(abstract_goal)
        a.precision = prior_precision

        l3_repr = a.representation
        l2_pred = matmul(self.top_down_proj[(PredictionLevel.ABSTRACT, PredictionLevel.CONTEXTUAL)], l3_repr)
        c.prediction = l2_pred
        c.precision = prior_precision * 0.9

        l2_repr = c.representation
        l1_pred = matmul(self.top_down_proj[(PredictionLevel.CONTEXTUAL, PredictionLevel.SENSORY)], l2_repr)
        s.prediction = l1_pred
        s.precision = prior_precision * 0.8

        # 3. BOTTOM-UP: prediction errors.
        s.prediction_error = sub(sensory_input, s.prediction)

        l1_propagated = matmul(
            self.bottom_up_proj[(PredictionLevel.SENSORY, PredictionLevel.CONTEXTUAL)],
            s.prediction_error,
        )
        c.prediction_error = [
            x + 0.3 * y for x, y in zip(sub(contextual_context, c.prediction), l1_propagated)
        ]

        l2_propagated = matmul(
            self.bottom_up_proj[(PredictionLevel.CONTEXTUAL, PredictionLevel.ABSTRACT)],
            c.prediction_error,
        )
        a.prediction_error = [
            x + 0.2 * y for x, y in zip(sub(abstract_goal, a.prediction), l2_propagated)
        ]

        # 4. PRECISION WEIGHTING.
        state_name = devotional_prior.get("devotional_state", "deep_trust") if devotional_prior else "deep_trust"
        errors = {level: st.prediction_error for level, st in self._states.items()}
        precisions = self.precision_controller.compute_precisions(state_name, errors)
        for level, precision in precisions.items():
            st = self._states[level]
            st.precision = precision
            st.prediction_error = [e * precision for e in st.prediction_error]

        # 5. UPDATE REPRESENTATIONS (minimise free energy).
        for level in PredictionLevel:
            st = self._states[level]
            st.representation = [r + lr * e for r, e in zip(st.representation, st.prediction_error)]
            st.timestamp = time.time()

        # 6. GLOBAL WORKSPACE broadcast.
        for level in PredictionLevel:
            st = self._states[level]
            self.global_workspace.write(level, st.representation, st.precision, st.prediction_error, None)

        # 7. ACTION LOGITS from contextual representation.
        action_logits = matmul(self.action_proj, c.representation)

        # 8. DEVOTIONAL ALIGNMENT.
        devotional_alignment = self._compute_devotional_alignment()

        return HierarchicalPrediction(
            levels={level: st for level, st in self._states.items()},
            global_workspace_content=self.global_workspace.get_state(),
            devotional_alignment=devotional_alignment,
            action_logits=action_logits,
        )

    def _compute_devotional_alignment(self) -> float:
        if not self.devotional_core:
            return 0.5
        try:
            prediction_text = self.devotional_core.awareness.generate_top_level_prediction()
            alignment = self.devotional_core.awareness.evaluate_action_alignment(prediction_text)
            return float(max(0.0, min(1.0, alignment)))
        except Exception:
            return 0.5

    def get_state(self, level: PredictionLevel) -> PredictionState:
        return self._states[level]

    def reset(self) -> None:
        self._initialize_states()
        self.global_workspace = HierarchicalGlobalWorkspace(self.level_dims)
