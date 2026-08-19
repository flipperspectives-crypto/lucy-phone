"""Hierarchical Global Workspace (HGWS) - the broadcasting blackboard.

Pure-Python (no numpy). Content vectors are ``list[float]``.

The global workspace binds the most salient/precise content into a shared
representation that all levels can attend to (the "winner-take-all" broadcast).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import math

from lucy_core._linalg import (
    matmul,
    randn,
    zeros_2d,
)
from lucy_core.brain.hierarchical import PredictionLevel, HierarchicalPredictor

RUN_DIM = 512


@dataclass
class WorkspaceEntry:
    """Content broadcast to the global workspace."""
    timestamp: float
    level: str
    content: List[float]
    confidence: float = 0.0
    prediction: object = None
    salience: float = 0.0
    precision_weight: float = 0.0


class HierarchicalGlobalWorkspace:
    """Manages the global workspace state and content broadcasting."""

    def __init__(self, dims: Optional[Dict[str, int]] = None) -> None:
        self.run_dim = RUN_DIM
        self.dims = dims or {"sensory": 256, "contextual": 512, "abstract": 512}
        self._workspace = [0.0] * self.run_dim
        self._entries: List[WorkspaceEntry] = []
        self._init_projections()

    def _init_projections(self) -> None:
        self._proj = {
            "attn": [[v * 0.02 for v in row] for row in randn(self.run_dim, self.run_dim)],
            "gate": [[v * 0.02 for v in row] for row in randn(self.run_dim, self.run_dim)],
        }

    def project_content(self, content: List[float]) -> List[float]:
        """Project arbitrary-dimensional content into the run-dimensional space."""
        if len(content) < self.run_dim:
            proj = list(content) + [0.0] * (self.run_dim - len(content))
        else:
            proj = list(content[:self.run_dim])
        return proj

    def write(
        self,
        level,
        response: List[float],
        confidence: Optional[float] = None,
        prediction=None,
        salience: Optional[float] = None,
    ) -> None:
        # Project and store
        entry = WorkspaceEntry(
            timestamp=0.0,
            level=str(level),
            content=self.project_content(response),
            confidence=confidence if confidence is not None else 0.0,
            prediction=prediction,
            salience=salience if salience is not None else 0.0,
        )
        # Broadcast
        self.broadcast(entry)

    def broadcast(self, entry: WorkspaceEntry) -> None:
        content = entry.content
        attn = [
            math.tanh(sum(self._proj["attn"][i][k] * content[k] for k in range(len(content))))
            for i in range(self.run_dim)
        ]
        # gate is (dim, dim); apply a simple per-dim diagonal gate
        gated = [attn[i] * self._proj["gate"][i][i] for i in range(self.run_dim)]
        update = [gated[i] + 0.1 * content[i] for i in range(self.run_dim)]
        self._workspace = [
            0.9 * self._workspace[i] + 0.1 * update[i] for i in range(self.run_dim)
        ]
        self._entries.append(entry)

    def query(self, content: List[float]) -> float:
        """Compute similarity of content with current workspace state."""
        proj = self.project_content(content)
        return sum(proj[i] * self._workspace[i] for i in range(self.run_dim))

    def get_state(self) -> List[float]:
        return list(self._workspace)

    def get_entries(self) -> List[WorkspaceEntry]:
        return self._entries

    def to_dict(self) -> dict:
        return {
            "workspace": list(self._workspace),
            "proj": self._proj,
            "entries": [
                {
                    "timestamp": e.timestamp,
                    "level": e.level,
                    "content": list(e.content),
                    "confidence": e.confidence,
                    "prediction": e.prediction,
                    "salience": e.salience,
                    "precision_weight": e.precision_weight,
                }
                for e in self._entries
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HierarchicalGlobalWorkspace":
        obj = cls()
        obj._workspace = list(d["workspace"])
        obj._proj = d["proj"]
        obj._entries = [
            WorkspaceEntry(
                timestamp=e["timestamp"],
                level=e["level"],
                content=list(e["content"]),
                confidence=e["confidence"],
                prediction=e["prediction"],
                salience=e["salience"],
                precision_weight=e.get("precision_weight", 0.0),
            )
            for e in d["entries"]
        ]
        return obj

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "HierarchicalGlobalWorkspace":
        with open(path, "rb") as f:
            return cls.from_dict(pickle.load(f))
