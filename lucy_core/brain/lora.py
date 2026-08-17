"""LoRA Adapter Manager - Sleep-time weight updates.

During sleep, cortical weights are updated via LoRA adapters
trained on replayed episodic memories. This is how the brain
"grows" without full retraining.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""
    rank: int = 16              # Low-rank dimension
    alpha: float = 32.0         # Scaling factor
    dropout: float = 0.1        # Dropout rate
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])


@dataclass
class LoRAAdapter:
    """A single LoRA adapter for one module."""
    name: str
    rank: int
    alpha: float
    # LoRA weights: A (d_model x rank), B (rank x d_model)
    A: np.ndarray  # (d_model, rank)
    B: np.ndarray  # (rank, d_model)
    dropout: float = 0.1
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """LoRA forward pass: x + alpha * (x @ A @ B)"""
        # x: (batch, seq, d_model)
        # A: (d_model, rank), B: (rank, d_model)
        # x @ A: (batch, seq, rank)
        # (x @ A) @ B: (batch, seq, d_model)
        lora_out = x @ self.A @ self.B
        return x + self.alpha * lora_out
    
    def merge_into(self, base_weight: np.ndarray) -> np.ndarray:
        """Merge LoRA into base weight matrix.
        
        For W + alpha * A @ B where W is (d_model, d_model)
        """
        lora_delta = self.alpha * (self.A @ self.B)  # (d_model, d_model)
        return base_weight + lora_delta


class LoRAAdapterManager:
    """Manages LoRA adapters for all hierarchical levels.
    
    During sleep (NREM replay):
    1. Sample episodic memories
    2. Compute prediction errors
    3. Update LoRA adapters via gradient descent
    4. Optionally merge into base weights
    """
    
    def __init__(
        self,
        level_dims: Dict[str, int],
        config: Optional[LoRAConfig] = None,
        checkpoint_dir: str = "./lucy_core/checkpoints/lora_adapters",
    ) -> None:
        self.level_dims = level_dims
        self.config = config or LoRAConfig()
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Adapters per level per target module
        self.adapters: Dict[str, Dict[str, LoRAAdapter]] = {}
        self._initialize_adapters()
        
        # Training state
        self.training_step = 0
        self.learning_rate = 1e-4
    
    def _initialize_adapters(self) -> None:
        """Initialize LoRA adapters for each level and module."""
        for level_name, dim in self.level_dims.items():
            self.adapters[level_name] = {}
            for module in self.config.target_modules:
                # Initialize A with kaiming, B with zeros (so initial output = 0)
                A = np.random.randn(dim, self.config.rank).astype(np.float32) * np.sqrt(2.0 / dim)
                B = np.zeros((self.config.rank, dim), dtype=np.float32)
                
                adapter = LoRAAdapter(
                    name=f"{level_name}_{module}",
                    rank=self.config.rank,
                    alpha=self.config.alpha,
                    A=A,
                    B=B,
                    dropout=self.config.dropout,
                )
                self.adapters[level_name][module] = adapter
    
    def get_adapter(self, level: str, module: str) -> Optional[LoRAAdapter]:
        """Get adapter for a specific level and module."""
        return self.adapters.get(level, {}).get(module)
    
    def apply_adapters(
        self, 
        level: str, 
        module_outputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Apply all adapters for a level to module outputs."""
        adapted = {}
        for module, output in module_outputs.items():
            adapter = self.get_adapter(level, module)
            if adapter is not None:
                adapted[module] = adapter.forward(output)
            else:
                adapted[module] = output
        return adapted
    
    def compute_gradients(
        self,
        level: str,
        module: str,
        input_activations: np.ndarray,  # (batch, seq, d_model)
        output_gradients: np.ndarray,    # (batch, seq, d_model) - gradient of loss w.r.t output
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradients for LoRA adapter weights.
        
        LoRA: output = x + alpha * (x @ A @ B)
        dL/dA = alpha * (x.T @ dL/doutput @ B.T)
        dL/dB = alpha * (x @ A).T @ dL/doutput
        """
        adapter = self.get_adapter(level, module)
        if adapter is None:
            return None, None
        
        # x: (batch*seq, d_model) - flatten batch and seq
        x = input_activations.reshape(-1, input_activations.shape[-1])
        # grad_output: (batch*seq, d_model)
        grad_out = output_gradients.reshape(-1, output_gradients.shape[-1])
        
        # x @ A: (batch*seq, rank)
        xA = x @ adapter.A
        
        # dL/dB = alpha * (xA).T @ grad_out
        # (rank, batch*seq) @ (batch*seq, d_model) = (rank, d_model)
        grad_B = adapter.alpha * (xA.T @ grad_out)
        
        # dL/dA = alpha * x.T @ (grad_out @ B.T)
        # (d_model, batch*seq) @ (batch*seq, rank) = (d_model, rank)
        grad_A = adapter.alpha * (x.T @ (grad_out @ adapter.B.T))
        
        return grad_A, grad_B
    
    def update_adapters(
        self,
        level: str,
        module: str,
        grad_A: np.ndarray,
        grad_B: np.ndarray,
    ) -> None:
        """Update adapter weights with gradients."""
        adapter = self.get_adapter(level, module)
        if adapter is None:
            return
        
        adapter.A -= self.learning_rate * grad_A
        adapter.B -= self.learning_rate * grad_B
    
    def train_step(
        self,
        level: str,
        module: str,
        input_activations: np.ndarray,
        target_activations: np.ndarray,
    ) -> float:
        """Single training step: minimize MSE between adapted output and target."""
        adapter = self.get_adapter(level, module)
        if adapter is None:
            return 0.0
        
        # Forward
        output = adapter.forward(input_activations)
        
        # Loss: MSE
        diff = output - target_activations
        loss = float(np.mean(diff ** 2))
        
        # Gradients
        grad_out = 2.0 * diff / diff.size  # dL/doutput
        grad_A, grad_B = self.compute_gradients(level, module, input_activations, grad_out)
        
        if grad_A is not None:
            self.update_adapters(level, module, grad_A, grad_B)
        
        return loss
    
    def save_checkpoint(self, step: int, suffix: str = "") -> str:
        """Save all adapters to disk."""
        import pickle
        
        filename = f"lora_step_{step}{suffix}.pkl"
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        # Convert to serializable format
        serializable = {}
        for level, modules in self.adapters.items():
            serializable[level] = {}
            for module, adapter in modules.items():
                serializable[level][module] = {
                    "name": adapter.name,
                    "rank": adapter.rank,
                    "alpha": adapter.alpha,
                    "A": adapter.A,
                    "B": adapter.B,
                    "dropout": adapter.dropout,
                }
        
        with open(filepath, "wb") as f:
            pickle.dump({
                "step": step,
                "config": {
                    "rank": self.config.rank,
                    "alpha": self.config.alpha,
                    "dropout": self.config.dropout,
                    "target_modules": self.config.target_modules,
                },
                "adapters": serializable,
            }, f)
        
        return filepath
    
    def load_checkpoint(self, filepath: str) -> int:
        """Load adapters from disk."""
        import pickle
        
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        
        # Restore config
        self.config.rank = data["config"]["rank"]
        self.config.alpha = data["config"]["alpha"]
        self.config.dropout = data["config"]["dropout"]
        self.config.target_modules = data["config"]["target_modules"]
        
        # Restore adapters
        self.adapters = {}
        for level, modules in data["adapters"].items():
            self.adapters[level] = {}
            for module, adapter_data in modules.items():
                self.adapters[level][module] = LoRAAdapter(
                    name=adapter_data["name"],
                    rank=adapter_data["rank"],
                    alpha=adapter_data["alpha"],
                    A=adapter_data["A"],
                    B=adapter_data["B"],
                    dropout=adapter_data["dropout"],
                )
        
        return data["step"]
    
    def list_checkpoints(self) -> List[str]:
        """List available checkpoints."""
        files = [f for f in os.listdir(self.checkpoint_dir) if f.endswith(".pkl")]
        files.sort(key=lambda x: int(x.split("_")[2].split(".")[0]) if "_" in x else 0)
        return [os.path.join(self.checkpoint_dir, f) for f in files]
    
    def merge_into_base_weights(self, base_weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Merge all adapters into base weight matrices.
        
        base_weights: {level_module: weight_matrix}
        Returns merged weights.
        """
        merged = {}
        for level, modules in self.adapters.items():
            for module, adapter in modules.items():
                key = f"{level}_{module}"
                if key in base_weights:
                    merged[key] = adapter.merge_into(base_weights[key])
                else:
                    merged[key] = adapter.A @ adapter.B * adapter.alpha
        return merged


def create_lora_manager(
    level_dims: Dict[str, int],
    checkpoint_dir: str = "./lucy_core/checkpoints/lora_adapters",
) -> LoRAAdapterManager:
    """Factory for LoRA manager."""
    return LoRAAdapterManager(level_dims, checkpoint_dir=checkpoint_dir)