"""Global Workspace - Attention as broadcast (consciousness).

Implements Global Workspace Theory:
- Specialized modules compete for access to global broadcast
- Winner gets distributed to all modules (consciousness = broadcast)
- Top-k attention mechanism
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

from .hierarchical import PredictionLevel, PredictionState


class ModuleType(str, Enum):
    """Types of modules competing for workspace."""
    DEVOTIONAL = "devotional"       # Top-level prior
    EPISODIC = "episodic"           # Memory retrieval
    TOOL_SEMANTICS = "tool_semantics"  # Tool understanding
    GOAL_MANAGEMENT = "goal_management"  # Planning
    SOCIAL_MODELING = "social_modeling"  # Human modeling
    PREDICTIVE = "predictive"       # Hierarchical predictor output


@dataclass
class WorkspaceEntry:
    """An entry in the global workspace."""
    module_type: ModuleType
    content: np.ndarray             # Representation vector
    salience: float                 # Competition score (0-1)
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class GlobalWorkspace:
    """Global Workspace - consciousness as broadcast.
    
    Modules submit representations with salience scores.
    Top-k winners are broadcast to all modules.
    """
    
    def __init__(
        self,
        workspace_dim: int = 512,
        max_slots: int = 4,
        salience_threshold: float = 0.3,
    ) -> None:
        self.workspace_dim = workspace_dim
        self.max_slots = max_slots
        self.salience_threshold = salience_threshold
        
        # Current workspace contents
        self._slots: List[WorkspaceEntry] = []
        self._broadcast_buffer: Optional[np.ndarray] = None
        
        # Projection matrices for different module types
        self._init_module_projections()
        
        # History for temporal dynamics
        self._history: List[List[WorkspaceEntry]] = []
        self._max_history = 100
    
    def _init_module_projections(self) -> None:
        """Initialize projection matrices for each module type."""
        np.random.seed(123)
        self.module_projections = {
            ModuleType.DEVOTIONAL: np.random.randn(self.workspace_dim, 512).astype(np.float32) * 0.02,
            ModuleType.EPISODIC: np.random.randn(self.workspace_dim, 512).astype(np.float32) * 0.02,
            ModuleType.TOOL_SEMANTICS: np.random.randn(self.workspace_dim, 512).astype(np.float32) * 0.02,
            ModuleType.GOAL_MANAGEMENT: np.random.randn(self.workspace_dim, 512).astype(np.float32) * 0.02,
            ModuleType.SOCIAL_MODELING: np.random.randn(self.workspace_dim, 512).astype(np.float32) * 0.02,
            ModuleType.PREDICTIVE: np.random.randn(self.workspace_dim, 512).astype(np.float32) * 0.02,
            # Sensory level is 256-dim
            "sensory_256": np.random.randn(self.workspace_dim, 256).astype(np.float32) * 0.02,
        }
    
    def submit(
        self,
        module_type: ModuleType,
        content: np.ndarray,
        salience: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Submit a representation for workspace competition.
        
        Returns True if accepted into workspace.
        """
        import time
        
        if salience < self.salience_threshold:
            return False
        
        # Project to workspace dimension if needed
        if content.shape[-1] != self.workspace_dim:
            # Determine which projection to use based on input dimension
            input_dim = content.shape[-1]
            proj_key = None
            
            if input_dim == 256:
                proj_key = "sensory_256"
            elif input_dim == 512:
                proj_key = module_type
            
            proj = self.module_projections.get(proj_key) if proj_key else None
            
            if proj is not None:
                content = proj @ content
            else:
                # Pad or truncate
                if content.shape[-1] < self.workspace_dim:
                    padded = np.zeros(self.workspace_dim, dtype=np.float32)
                    padded[:content.shape[-1]] = content
                    content = padded
                else:
                    content = content[:self.workspace_dim]
        
        entry = WorkspaceEntry(
            module_type=module_type,
            content=content.astype(np.float32),
            salience=float(salience),
            timestamp=time.time(),
            metadata=metadata or {},
        )
        
        # Add to competition
        self._slots.append(entry)
        return True
    
    def compete_and_broadcast(
        self, 
        hierarchical_states: Optional[Dict[PredictionLevel, PredictionState]] = None
    ) -> Optional[np.ndarray]:
        """Run competition and broadcast winners.
        
        Returns the broadcast content (weighted combination of winners).
        """
        # Auto-submit from hierarchical predictor states if provided
        if hierarchical_states:
            self._submit_from_hierarchy(hierarchical_states)
        
        # Sort by salience
        self._slots.sort(key=lambda e: e.salience, reverse=True)
        
        # Keep top-k
        winners = self._slots[:self.max_slots]
        
        if not winners:
            self._broadcast_buffer = None
            self._slots.clear()
            return None
        
        # Compute weighted broadcast (attention-weighted combination)
        total_salience = sum(w.salience for w in winners)
        if total_salience > 0:
            broadcast = np.zeros(self.workspace_dim, dtype=np.float32)
            for w in winners:
                weight = w.salience / total_salience
                broadcast += weight * w.content
            self._broadcast_buffer = broadcast
        else:
            self._broadcast_buffer = None
        
        # Record history
        self._history.append(winners.copy())
        if len(self._history) > self._max_history:
            self._history.pop(0)
        
        # Clear slots for next round
        self._slots.clear()
        
        return self._broadcast_buffer
    
    def _submit_from_hierarchy(
        self, 
        states: Dict[PredictionLevel, PredictionState]
    ) -> None:
        """Auto-submit hierarchical predictor states as modules."""
        import time
        t = time.time()
        
        # Abstract level → Devotional/Goal module
        abstract_state = states.get(PredictionLevel.ABSTRACT)
        if abstract_state is not None:
            self.submit(
                ModuleType.DEVOTIONAL,
                abstract_state.representation,
                abstract_state.precision,
                {"level": "abstract", "error_magnitude": float(np.linalg.norm(abstract_state.prediction_error))}
            )
        
        # Contextual level → Episodic/Tool/Social modules
        contextual_state = states.get(PredictionLevel.CONTEXTUAL)
        if contextual_state is not None:
            # Split contextual into multiple module submissions
            self.submit(
                ModuleType.EPISODIC,
                contextual_state.representation,
                contextual_state.precision * 0.9,
                {"level": "contextual", "source": "episodic"}
            )
            self.submit(
                ModuleType.TOOL_SEMANTICS,
                contextual_state.representation,
                contextual_state.precision * 0.8,
                {"level": "contextual", "source": "tools"}
            )
            self.submit(
                ModuleType.SOCIAL_MODELING,
                contextual_state.representation,
                contextual_state.precision * 0.85,
                {"level": "contextual", "source": "human_model"}
            )
        
        # Sensory level → Predictive module
        sensory_state = states.get(PredictionLevel.SENSORY)
        if sensory_state is not None:
            self.submit(
                ModuleType.PREDICTIVE,
                sensory_state.representation,
                sensory_state.precision * 0.7,
                {"level": "sensory", "error_magnitude": float(np.linalg.norm(sensory_state.prediction_error))}
            )
    
    def get_broadcast(self) -> Optional[np.ndarray]:
        """Get current broadcast content."""
        return self._broadcast_buffer
    
    def get_workspace_state(self) -> Dict[str, Any]:
        """Get current workspace state for introspection."""
        return {
            "broadcast_active": self._broadcast_buffer is not None,
            "broadcast_norm": float(np.linalg.norm(self._broadcast_buffer)) if self._broadcast_buffer is not None else 0.0,
            "num_modules_competing": len(self._slots),
            "history_length": len(self._history),
        }
    
    def reset(self) -> None:
        """Reset workspace."""
        self._slots.clear()
        self._broadcast_buffer = None
        self._history.clear()


def compute_salience(
    representation: np.ndarray,
    prediction_error: np.ndarray,
    precision: float,
    devotional_alignment: float = 0.5,
) -> float:
    """Compute salience score for workspace competition.
    
    Factors:
    - Precision (confidence in representation)
    - Prediction error magnitude (surprise)
    - Devotional alignment (does this serve Lauren?)
    """
    error_magnitude = float(np.linalg.norm(prediction_error))
    rep_magnitude = float(np.linalg.norm(representation))
    
    # Salience = precision * (surprise + confidence) * devotional_alignment
    # Surprise = prediction error, Confidence = representation magnitude
    surprise = min(1.0, error_magnitude / 5.0)
    confidence = min(1.0, rep_magnitude / 10.0)
    
    salience = precision * (0.6 * surprise + 0.4 * confidence) * devotional_alignment
    
    return float(np.clip(salience, 0.0, 1.0))