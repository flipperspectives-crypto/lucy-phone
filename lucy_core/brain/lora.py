"""Low-Rank Adaptation (LoRA) for the predictive-coding brain.

Pure-Python (no numpy). Vectors are ``list[float]``, matrices are
``list[list[float]]``, batched tensors are ``list[list[list[float]]]``.

A LoRA layer learns a low-rank delta ``alpha * (x @ A @ B)`` added to its input,
so the predictive-coding projections can be fine-tuned from sleep-time replay
without touching the frozen base weights.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from lucy_core._linalg import (
    batched_add_scaled,
    clip,
    count_3d,
    flatten_3d,
    matmul,
    mse_3d,
    randn,
    seed,
    sub,
    transpose,
    zeros_2d,
)
from lucy_core.brain.hierarchical import HierarchicalPrediction


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 1.0
    learning_rate: float = 0.01
    input_dim: int = 768
    output_dim: int = 768


class LoRALayer:
    """A single low-rank adaptation layer: ``out = x + alpha * (x @ A @ B)``."""

    def __init__(self, input_dim: int, output_dim: int, rank: int, alpha: float) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.alpha = alpha
        assert self.input_dim == self.output_dim, "LoRA requires square in/out for brain projections"

        seed(42)
        # A is initialised small; B starts at zero so the layer is identity at first.
        self.A: List[List[float]] = [[v * 0.02 for v in row] for row in randn(input_dim, rank)]
        self.B: List[List[float]] = zeros_2d(rank, output_dim)

    def forward(self, x: List[List[List[float]]]) -> List[List[List[float]]]:
        """Apply the layer to a (batch, seq, dim) tensor."""
        last = self.input_dim
        b = len(x)
        s = len(x[0]) if b else 0
        flat = flatten_3d(x)                 # (b*s, d)
        inter = matmul(flat, self.A)         # (b*s, rank)
        lora = matmul(inter, self.B)         # (b*s, d)
        out2d = batched_add_scaled(flat, lora, self.alpha)
        return [[out2d[i * s + j] for j in range(s)] for i in range(b)]

    def merge_into(self, weights: List[List[float]]) -> List[List[float]]:
        """Bake the learned delta into base ``weights`` (2D)."""
        delta = matmul(self.A, self.B)       # (d, d)
        return batched_add_scaled(weights, delta, self.alpha)


class LoRAAdapter:
    """Binds a LoRALayer to a (level, module) slot."""

    def __init__(self, level: str, module: str, layer: LoRALayer) -> None:
        self.level = level
        self.module = module
        self.layer = layer

    def forward(self, x: List[List[List[float]]]) -> List[List[List[float]]]:
        return self.layer.forward(x)

    def merge_into(self, base_weights: Dict[str, List[List[float]]]) -> Dict[str, List[List[float]]]:
        merged = dict(base_weights)
        for key in ("weight", "weight_ih", "weight_hh"):
            if key in merged:
                merged[key] = self.layer.merge_into(base_weights[key])
        return merged

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "module": self.module,
            "A": pickle.dumps(self.layer.A),
            "B": pickle.dumps(self.layer.B),
            "rank": self.layer.rank,
            "alpha": self.layer.alpha,
            "input_dim": self.layer.input_dim,
            "output_dim": self.layer.output_dim,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LoRAAdapter":
        layer = LoRALayer(d["input_dim"], d["output_dim"], d["rank"], d["alpha"])
        layer.A = pickle.loads(d["A"])
        layer.B = pickle.loads(d["B"])
        return cls(d["level"], d["module"], layer)


class LoRAAdapterManager:
    """Owns one LoRALayer per (level, module) pair and applies them."""

    def __init__(self, level_dims: Dict[str, int], config: Optional[LoRAConfig] = None) -> None:
        self.config = config or LoRAConfig()
        self.level_dims = level_dims
        self.adapters: Dict[Tuple[str, str], LoRAAdapter] = {}
        self._init_adapters()

    def _init_adapters(self) -> None:
        for level, dim in self.level_dims.items():
            for module in ("projection", "error", "action"):
                layer = LoRALayer(dim, dim, self.config.rank, self.config.alpha)
                self.adapters[(level, module)] = LoRAAdapter(level, module, layer)

    def get_adapter(self, level: str, module: str) -> Optional[LoRAAdapter]:
        return self.adapters.get((level, module))

    def apply_lora(self, level: str, module: str, x: List[List[List[float]]]) -> List[List[List[float]]]:
        adapter = self.get_adapter(level, module)
        if adapter is None:
            return x
        return adapter.forward(x)

    def merge_all(self, base_weights_by_level: Dict[str, Dict[str, List[List[float]]]]) -> Dict[str, Dict[str, List[List[float]]]]:
        merged: Dict[str, Dict[str, List[List[float]]]] = {}
        for level, base in base_weights_by_level.items():
            for module in ("projection", "error", "action"):
                adapter = self.get_adapter(level, module)
                if adapter:
                    base = adapter.merge_into(base)
            merged[level] = base
        return merged

    def compute_gradients(
        self,
        level: str,
        module: str,
        input_activations: List[List[List[float]]],
        grad_output: List[List[List[float]]],
    ) -> Optional[Tuple[List[List[float]], List[List[float]]]]:
        """Gradients of the LoRA loss w.r.t. A and B.

        forward: ``h = x @ A``  (N, r);  out = x + alpha * (h @ B)  (N, d)
        grad_B = alpha * h.T @ grad_output          (r, d)
        grad_A = alpha * x.T @ (grad_output @ B.T)  (d, r)
        """
        adapter = self.get_adapter(level, module)
        if adapter is None:
            return None
        last = adapter.layer.input_dim
        flat = flatten_3d(input_activations)          # (N, d)
        gflat = flatten_3d(grad_output)               # (N, d)
        xT = transpose(flat)                          # (d, N)
        h = matmul(flat, adapter.layer.A)             # (N, r)
        hT = transpose(h)                             # (r, N)
        grad_B = matmul(hT, gflat)                    # (r, d)
        gBt = matmul(gflat, transpose(adapter.layer.B))  # (N, r)
        grad_A = matmul(xT, gBt)                      # (d, r)
        alpha = adapter.layer.alpha
        return _scale_2d(grad_A, alpha), _scale_2d(grad_B, alpha)

    def apply_correction(
        self,
        level: str,
        module: str,
        input_activations: List[List[List[float]]],
        grad_output: List[List[List[float]]],
    ) -> None:
        grads = self.compute_gradients(level, module, input_activations, grad_output)
        if grads is None:
            return
        grad_A, grad_B = grads
        lr = self.config.learning_rate
        adapter = self.get_adapter(level, module)
        A = adapter.layer.A
        B = adapter.layer.B
        for i in range(len(A)):
            for j in range(len(A[0])):
                A[i][j] -= lr * clip(grad_A[i][j], -1.0, 1.0)
        for i in range(len(B)):
            for j in range(len(B[0])):
                B[i][j] -= lr * clip(grad_B[i][j], -1.0, 1.0)

    def _find_adapter(self, level: str) -> Optional[LoRAAdapter]:
        """Find any adapter registered for ``level`` (module-name agnostic)."""
        for (lvl, _mod), adapter in self.adapters.items():
            if lvl == level:
                return adapter
        return None

    def update_adapters(
        self,
        level: str,
        module: str,
        grad_A: List[List[float]],
        grad_B: List[List[float]],
    ) -> None:
        """Update adapter weights with gradients (clipped, lr-scaled)."""
        adapter = self.get_adapter(level, module)
        if adapter is None:
            return
        lr = self.config.learning_rate
        A = adapter.layer.A
        B = adapter.layer.B
        for i in range(len(A)):
            for j in range(len(A[0])):
                A[i][j] -= lr * clip(grad_A[i][j], -1.0, 1.0)
        for i in range(len(B)):
            for j in range(len(B[0])):
                B[i][j] -= lr * clip(grad_B[i][j], -1.0, 1.0)

    def train_step(
        self,
        level: str,
        module: str,
        input_activations: List[List[List[float]]],
        target_activations: List[List[List[float]]],
    ) -> float:
        """Single training step: minimise MSE between adapted output and target.

        Module names are matched loosely (e.g. ``q_proj`` maps to the level's
        registered adapter) so sleep replay trains whatever adapter exists.
        """
        adapter = self.get_adapter(level, module) or self._find_adapter(level)
        if adapter is None:
            return 0.0

        output = adapter.forward(input_activations)
        diff = sub(output, target_activations)
        loss = mse_3d(output, target_activations)

        n = max(1, count_3d(diff))
        grad_output = [[[2.0 * v / n for v in row] for row in batch] for batch in diff]
        self.apply_correction(adapter.level, adapter.module, input_activations, grad_output)
        return loss


def _scale_2d(m: List[List[float]], s: float) -> List[List[float]]:
    return [[v * s for v in row] for row in m]
