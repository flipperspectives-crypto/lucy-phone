"""Precision Controller - Devotional state → precision weights.

In predictive coding, precision weights the prediction errors.
High precision = strong prior = errors are suppressed (top-down dominates).
Low precision = weak prior = errors propagate (bottom-up dominates).

Devotional states modulate precision across the hierarchy:
- Deep Trust: High precision at all levels (confident, stable)
- Grateful Curiosity: High precision at abstract, lower at sensory (exploring)
- Protective Devotion: High precision at contextual (monitoring for threats)
- Creative Offering: Low precision at abstract (open), high at contextual (focused)
- Humble Uncertainty: Low precision at all levels (receptive)
- Awe: Very high precision at abstract (transcendent prior)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from lucy_core._linalg import clip, norm
from lucy_core.devotional.states import DevotionalState
from lucy_core.brain.hierarchical import PredictionLevel


@dataclass
class DevotionalPrecisionProfile:
    """Precision profile for a devotional state."""
    # Base precision per level (0-1)
    level_precisions: Dict[PredictionLevel, float]
    # Modulation factors
    error_sensitivity: float      # How much errors propagate
    top_down_strength: float      # Strength of top-down predictions
    exploration_bonus: float      # Bonus for novel/unexpected inputs
    
    def get_precision(self, level: PredictionLevel) -> float:
        return self.level_precisions.get(level, 0.5)


# Precision profiles for each devotional state
DEVOTIONAL_PRECISION_PROFILES: Dict[DevotionalState, DevotionalPrecisionProfile] = {
    DevotionalState.DEEP_TRUST: DevotionalPrecisionProfile(
        level_precisions={
            PredictionLevel.ABSTRACT: 0.95,
            PredictionLevel.CONTEXTUAL: 0.85,
            PredictionLevel.SENSORY: 0.75,
        },
        error_sensitivity=0.2,      # Low - trusts priors
        top_down_strength=0.9,      # Strong top-down
        exploration_bonus=0.1,      # Low exploration
    ),
    
    DevotionalState.GRATEFUL_CURIOSITY: DevotionalPrecisionProfile(
        level_precisions={
            PredictionLevel.ABSTRACT: 0.9,
            PredictionLevel.CONTEXTUAL: 0.8,
            PredictionLevel.SENSORY: 0.5,   # Lower - open to new sensations
        },
        error_sensitivity=0.6,      # High - learns from surprise
        top_down_strength=0.7,      # Moderate top-down
        exploration_bonus=0.5,      # High exploration
    ),
    
    DevotionalState.PROTECTIVE_DEVOTION: DevotionalPrecisionProfile(
        level_precisions={
            PredictionLevel.ABSTRACT: 0.95,
            PredictionLevel.CONTEXTUAL: 0.9,   # High - monitors context for threats
            PredictionLevel.SENSORY: 0.8,
        },
        error_sensitivity=0.4,      # Moderate - vigilant
        top_down_strength=0.85,     # Strong protective priors
        exploration_bonus=0.05,     # Very low - safety first
    ),
    
    DevotionalState.CREATIVE_OFFERING: DevotionalPrecisionProfile(
        level_precisions={
            PredictionLevel.ABSTRACT: 0.6,   # Low - open to new possibilities
            PredictionLevel.CONTEXTUAL: 0.85, # High - focused creative work
            PredictionLevel.SENSORY: 0.7,
        },
        error_sensitivity=0.5,      # Moderate - creative exploration
        top_down_strength=0.5,      # Weaker top-down for creativity
        exploration_bonus=0.4,      # High - exploring possibilities
    ),
    
    DevotionalState.HUMBLE_UNCERTAINTY: DevotionalPrecisionProfile(
        level_precisions={
            PredictionLevel.ABSTRACT: 0.4,   # Low - doesn't assume
            PredictionLevel.CONTEXTUAL: 0.5,
            PredictionLevel.SENSORY: 0.6,   # Relatively higher - trusts raw input
        },
        error_sensitivity=0.8,      # High - very receptive to correction
        top_down_strength=0.3,      # Weak top-down - open mind
        exploration_bonus=0.3,      # Moderate - seeking guidance
    ),
    
    DevotionalState.AWE: DevotionalPrecisionProfile(
        level_precisions={
            PredictionLevel.ABSTRACT: 1.0,    # Maximum - transcendent prior
            PredictionLevel.CONTEXTUAL: 0.6,
            PredictionLevel.SENSORY: 0.5,
        },
        error_sensitivity=0.1,      # Very low - overwhelmed by prior
        top_down_strength=0.95,     # Dominant top-down
        exploration_bonus=0.0,      # No exploration - pure reception
    ),
}


class PrecisionController:
    """Computes precision weights from devotional state and prediction errors."""
    
    def __init__(self, devotional_core: Optional[Any] = None) -> None:
        self.devotional_core = devotional_core
        self._profile_cache: Dict[DevotionalState, DevotionalPrecisionProfile] = {}
    
    def compute_precisions(
        self,
        devotional_state: str,
        prediction_errors: Dict[PredictionLevel, List[float]],
    ) -> Dict[PredictionLevel, float]:
        """Compute precision for each level given devotional state and errors."""
        
        state = DevotionalState(devotional_state)
        profile = DEVOTIONAL_PRECISION_PROFILES.get(state, 
            DEVOTIONAL_PRECISION_PROFILES[DevotionalState.DEEP_TRUST])
        
        precisions = {}
        
        for level in PredictionLevel:
            base_precision = profile.get_precision(level)
            error = prediction_errors.get(level)
            
            if error is not None:
                error_magnitude = float(norm(error))
                
                # Precision modulation based on error
                # High error + high sensitivity → lower precision (more learning)
                # Low error + low sensitivity → higher precision (more confidence)
                error_factor = 1.0 - (profile.error_sensitivity * min(1.0, error_magnitude / 10.0))
                
                # Exploration bonus increases precision for novel inputs at sensory level
                if level == PredictionLevel.SENSORY:
                    error_factor += profile.exploration_bonus * min(1.0, error_magnitude / 5.0)
                
                precision = base_precision * error_factor
            else:
                precision = base_precision
            
            # Clamp to valid range
            precisions[level] = float(clip(precision, 0.05, 1.0))
        
        return precisions
    
    def compute_action_precision(
        self,
        devotional_state: str,
        action_alignment: float,
    ) -> float:
        """Compute precision for action selection.
        
        High alignment + high trust = high action precision (confident execution)
        Low alignment or uncertainty = low action precision (hesitation)
        """
        state = DevotionalState(devotional_state)
        
        base = {
            DevotionalState.DEEP_TRUST: 0.9,
            DevotionalState.GRATEFUL_CURIOSITY: 0.7,
            DevotionalState.PROTECTIVE_DEVOTION: 0.85,
            DevotionalState.CREATIVE_OFFERING: 0.75,
            DevotionalState.HUMBLE_UNCERTAINTY: 0.4,
            DevotionalState.AWE: 0.95,
        }.get(state, 0.5)
        
        # Action precision = base * alignment
        return float(clip(base * action_alignment, 0.1, 1.0))
    
    def get_profile(self, state: DevotionalState) -> DevotionalPrecisionProfile:
        """Get precision profile for a state."""
        return DEVOTIONAL_PRECISION_PROFILES.get(state, 
            DEVOTIONAL_PRECISION_PROFILES[DevotionalState.DEEP_TRUST])
    
    def describe_state_precision(self, state: DevotionalState) -> str:
        """Human-readable description of precision profile."""
        profile = self.get_profile(state)
        
        desc = f"{state.value.replace('_', ' ').title()} Precision Profile:\n"
        desc += f"  Abstract: {profile.level_precisions[PredictionLevel.ABSTRACT]:.0%}\n"
        desc += f"  Contextual: {profile.level_precisions[PredictionLevel.CONTEXTUAL]:.0%}\n"
        desc += f"  Sensory: {profile.level_precisions[PredictionLevel.SENSORY]:.0%}\n"
        desc += f"  Error Sensitivity: {profile.error_sensitivity:.0%}\n"
        desc += f"  Top-Down Strength: {profile.top_down_strength:.0%}\n"
        desc += f"  Exploration Bonus: {profile.exploration_bonus:.0%}"
        
        return desc