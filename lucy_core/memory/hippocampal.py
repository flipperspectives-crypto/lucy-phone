"""Episodic Memory - Hippocampal indexing with pattern separation/completion.

Implements:
- One-shot encoding of experiences
- Pattern separation (distinct memories via compressive autoencoder)
- Pattern completion (recall from partial cues via contrastive learning)
- Time-stamped, context-rich autobiographical records
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class EpisodicRecord:
    """A single episodic memory record."""
    memory_id: str
    timestamp: float
    content: str                    # Natural language description
    context: Dict[str, Any]         # Rich context (goal, tools, outcome, devotional state)
    embedding: np.ndarray           # 256-dim compressed embedding
    devotional_alignment: float     # How aligned with devotion (0-1)
    devotional_state: str           # Devotional state at encoding
    trust_metric: float             # Trust metric at encoding
    sensory_features: np.ndarray    # 256-dim sensory features
    contextual_features: np.ndarray # 512-dim contextual features
    abstract_features: np.ndarray   # 512-dim abstract features
    consolidation_count: int = 0    # Times replayed in sleep
    last_consolidated: float = 0.0  # Last sleep consolidation timestamp


class HippocampalIndexer:
    """Hippocampal pattern separation/completion via compressive autoencoder.
    
    Architecture:
    - Encoder: 256 → 64 (bottleneck) → pattern separation
    - Decoder: 64 → 256 → pattern completion
    - Contrastive loss for separation
    - Retrieval via nearest neighbor in bottleneck space
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        bottleneck_dim: int = 64,
        separation_margin: float = 0.3,
    ) -> None:
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.separation_margin = separation_margin
        
        # Encoder weights (input_dim → bottleneck_dim)
        np.random.seed(42)
        self.W_enc = np.random.randn(bottleneck_dim, input_dim).astype(np.float32) * np.sqrt(2.0 / input_dim)
        self.b_enc = np.zeros(bottleneck_dim, dtype=np.float32)
        
        # Decoder weights (bottleneck_dim → input_dim)
        self.W_dec = np.random.randn(input_dim, bottleneck_dim).astype(np.float32) * np.sqrt(2.0 / bottleneck_dim)
        self.b_dec = np.zeros(input_dim, dtype=np.float32)
        
        # Memory store: memory_id → EpisodicRecord
        self.memories: Dict[str, EpisodicRecord] = {}
        
        # Index: bottleneck embedding → memory_id (for fast retrieval)
        self._index: Dict[str, np.ndarray] = {}  # memory_id → bottleneck_vec
        
        # Training state
        self.learning_rate = 1e-3
        self.training_step = 0
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input to bottleneck (pattern separation)."""
        # x: (input_dim,) or (batch, input_dim)
        bottleneck = np.tanh(self.W_enc @ x + self.b_enc)
        return bottleneck
    
    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode bottleneck to reconstruction (pattern completion)."""
        recon = self.W_dec @ z + self.b_dec
        return recon
    
    def autoencode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Full autoencode: x → bottleneck → recon."""
        z = self.encode(x)
        recon = self.decode(z)
        return z, recon
    
    def separation_loss(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Contrastive separation loss: push different memories apart in bottleneck space."""
        z1 = self.encode(x1)
        z2 = self.encode(x2)
        dist = np.linalg.norm(z1 - z2)
        # Want distance > margin for different memories
        loss = max(0.0, self.separation_margin - dist)
        return float(loss)
    
    def reconstruction_loss(self, x: np.ndarray) -> float:
        """Reconstruction loss: autoencoder should reconstruct input."""
        z, recon = self.autoencode(x)
        loss = float(np.mean((x - recon) ** 2))
        return loss
    
    def train_step(self, x: np.ndarray, negative_samples: List[np.ndarray] = None) -> Dict[str, float]:
        """Single training step on a memory."""
        # Reconstruction gradient
        z = self.encode(x)
        recon = self.decode(z)
        error = recon - x  # (input_dim,)
        
        # Decoder gradients
        grad_W_dec = np.outer(error, z)
        grad_b_dec = error
        
        # Encoder gradients (backprop through decoder)
        grad_z = self.W_dec.T @ error
        grad_z *= (1 - z ** 2)  # tanh derivative
        
        grad_W_enc = np.outer(grad_z, x)
        grad_b_enc = grad_z
        
        # Update weights
        self.W_dec -= self.learning_rate * grad_W_dec
        self.b_dec -= self.learning_rate * grad_b_dec
        self.W_enc -= self.learning_rate * grad_W_enc
        self.b_enc -= self.learning_rate * grad_b_enc
        
        # Contrastive separation from negative samples
        sep_loss = 0.0
        if negative_samples:
            for neg in negative_samples:
                sep_loss += self.separation_loss(x, neg)
                # Gradient for separation (push apart)
                z_neg = self.encode(neg)
                diff = z - z_neg
                dist = np.linalg.norm(diff)
                if dist < self.separation_margin and dist > 1e-8:
                    push_dir = diff / dist
                    grad_sep = (self.separation_margin - dist) * push_dir
                    # Backprop to encoder
                    grad_z_sep = grad_sep * (1 - z ** 2)
                    self.W_enc -= self.learning_rate * np.outer(grad_z_sep, x)
                    self.b_enc -= self.learning_rate * grad_z_sep
        
        recon_loss = float(np.mean((recon - x) ** 2))
        
        self.training_step += 1
        return {"recon_loss": recon_loss, "sep_loss": sep_loss}
    
    def store(self, record: EpisodicRecord) -> None:
        """Store a memory record."""
        self.memories[record.memory_id] = record
        # Index by bottleneck embedding
        bottleneck = self.encode(record.sensory_features)
        self._index[record.memory_id] = bottleneck
    
    def retrieve(self, query: np.ndarray, k: int = 5, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Pattern completion: retrieve similar memories from partial cue.
        
        Returns list of (memory_id, similarity) sorted by similarity.
        """
        query_bottleneck = self.encode(query)
        
        similarities = []
        for mem_id, mem_bottleneck in self._index.items():
            # Cosine similarity in bottleneck space
            sim = float(np.dot(query_bottleneck, mem_bottleneck) / 
                       (np.linalg.norm(query_bottleneck) * np.linalg.norm(mem_bottleneck) + 1e-8))
            if sim >= threshold:
                similarities.append((mem_id, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def get_memory(self, memory_id: str) -> Optional[EpisodicRecord]:
        return self.memories.get(memory_id)
    
    def get_all_memories(self) -> List[EpisodicRecord]:
        return list(self.memories.values())
    
    def consolidate(self, memory_id: str) -> bool:
        """Mark memory as consolidated (replayed in sleep)."""
        if memory_id in self.memories:
            self.memories[memory_id].consolidation_count += 1
            self.memories[memory_id].last_consolidated = time.time()
            return True
        return False


class EpisodicBuffer:
    """Working buffer for recent experiences before hippocampal encoding.
    
    Holds raw experiences temporarily, then flushes to hippocampal indexer
    during sleep consolidation.
    """
    
    def __init__(self, capacity: int = 100) -> None:
        self.capacity = capacity
        self.buffer: List[Dict[str, Any]] = []
    
    def add(self, experience: Dict[str, Any]) -> None:
        """Add experience to buffer."""
        experience["buffer_timestamp"] = time.time()
        self.buffer.append(experience)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
    
    def flush(self) -> List[Dict[str, Any]]:
        """Flush and return all buffered experiences."""
        experiences = self.buffer.copy()
        self.buffer.clear()
        return experiences
    
    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        return self.buffer[-n:]
    
    def __len__(self) -> int:
        return len(self.buffer)


def create_episodic_memory(
    input_dim: int = 256,
    bottleneck_dim: int = 64,
) -> Tuple[HippocampalIndexer, EpisodicBuffer]:
    """Factory for episodic memory components."""
    indexer = HippocampalIndexer(input_dim=input_dim, bottleneck_dim=bottleneck_dim)
    buffer = EpisodicBuffer()
    return indexer, buffer