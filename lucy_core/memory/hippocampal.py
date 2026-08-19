"""Episodic Memory - Hippocampal indexing with pattern separation/completion.

Pure-Python (no numpy). Vectors are ``list[float]``, matrices ``list[list[float]]``.

Implements a compressive autoencoder for pattern separation (distinct memories)
and completion (recall from partial cues via contrastive learning), plus
time-stamped autobiographical records.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from lucy_core._linalg import (
    matmul,
    norm,
    randn,
    seed,
    transpose,
    zeros,
    zeros_2d,
)


@dataclass
class EpisodicRecord:
    """A single episodic memory record."""
    memory_id: str
    timestamp: float
    content: str                    # Natural language description
    context: Dict[str, Any]         # Rich context (goal, tools, outcome, devotional state)
    embedding: List[float]          # 256-dim compressed embedding
    devotional_alignment: float     # How aligned with devotion (0-1)
    devotional_state: str           # Devotional state at encoding
    trust_metric: float             # Trust metric at encoding
    sensory_features: List[float]   # 256-dim sensory features
    contextual_features: List[float]  # 512-dim contextual features
    abstract_features: List[float]  # 512-dim abstract features
    consolidation_count: int = 0    # Times replayed in sleep
    last_consolidated: float = 0.0  # Last sleep consolidation timestamp


def _tanh_vec(v: List[float]) -> List[float]:
    return [math.tanh(x) for x in v]


def _outer(a: List[float], b: List[float]) -> List[List[float]]:
    return [[a[i] * b[j] for j in range(len(b))] for i in range(len(a))]


def _sub_1d(a: List[float], b: List[float]) -> List[float]:
    return [a[i] - b[i] for i in range(len(a))]


def _mul_1d(a: List[float], b: List[float]) -> List[float]:
    return [a[i] * b[i] for i in range(len(a))]


def _scale_1d(v: List[float], s: float) -> List[float]:
    return [x * s for x in v]


def _add_1d(a: List[float], b: List[float]) -> List[float]:
    return [a[i] + b[i] for i in range(len(a))]


def _dot(a: List[float], b: List[float]) -> float:
    return sum(a[i] * b[i] for i in range(len(a)))


class HippocampalIndexer:
    """Hippocampal pattern separation/completion via compressive autoencoder.

    Architecture:
    - Encoder: input_dim -> bottleneck_dim (pattern separation)
    - Decoder: bottleneck_dim -> input_dim (pattern completion)
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

        # Encoder weights (input_dim -> bottleneck_dim)
        seed(42)
        enc_scale = math.sqrt(2.0 / input_dim)
        self.W_enc: List[List[float]] = [[v * enc_scale for v in row] for row in randn(bottleneck_dim, input_dim)]
        self.b_enc: List[float] = zeros(bottleneck_dim)

        # Decoder weights (bottleneck_dim -> input_dim)
        dec_scale = math.sqrt(2.0 / bottleneck_dim)
        self.W_dec: List[List[float]] = [[v * dec_scale for v in row] for row in randn(input_dim, bottleneck_dim)]
        self.b_dec: List[float] = zeros(input_dim)

        # Memory store: memory_id -> EpisodicRecord
        self.memories: Dict[str, EpisodicRecord] = {}

        # Index: memory_id -> bottleneck embedding (for fast retrieval)
        self._index: Dict[str, List[float]] = {}

        # Training state
        self.learning_rate = 1e-3
        self.training_step = 0

    def encode(self, x: List[float]) -> List[float]:
        """Encode input to bottleneck (pattern separation)."""
        pre = _add_1d(matmul(self.W_enc, x), self.b_enc)
        return _tanh_vec(pre)

    def decode(self, z: List[float]) -> List[float]:
        """Decode bottleneck to reconstruction (pattern completion)."""
        return _add_1d(matmul(self.W_dec, z), self.b_dec)

    def autoencode(self, x: List[float]) -> Tuple[List[float], List[float]]:
        """Full autoencode: x -> bottleneck -> recon."""
        z = self.encode(x)
        recon = self.decode(z)
        return z, recon

    def separation_loss(self, x1: List[float], x2: List[float]) -> float:
        """Contrastive separation loss: push different memories apart."""
        z1 = self.encode(x1)
        z2 = self.encode(x2)
        dist = norm(_sub_1d(z1, z2))
        loss = max(0.0, self.separation_margin - dist)
        return float(loss)

    def reconstruction_loss(self, x: List[float]) -> float:
        """Reconstruction loss: autoencoder should reconstruct input."""
        z, recon = self.autoencode(x)
        mse = sum((x[i] - recon[i]) ** 2 for i in range(len(x))) / max(1, len(x))
        return float(mse)

    def train_step(self, x: List[float], negative_samples: Optional[List[List[float]]] = None) -> Dict[str, float]:
        """Single training step on a memory."""
        z = self.encode(x)
        recon = self.decode(z)
        error = _sub_1d(recon, x)  # (input_dim,)

        # Decoder gradients
        grad_W_dec = _outer(error, z)
        grad_b_dec = error

        # Encoder gradients (backprop through decoder)
        grad_z = matmul(transpose(self.W_dec), error)
        grad_z = _mul_1d(grad_z, [1.0 - zz * zz for zz in z])  # tanh derivative

        grad_W_enc = _outer(grad_z, x)
        grad_b_enc = grad_z

        # Update weights
        self.W_dec = [[self.W_dec[i][j] - self.learning_rate * grad_W_dec[i][j]
                       for j in range(len(self.W_dec[0]))] for i in range(len(self.W_dec))]
        self.b_dec = _sub_1d(self.b_dec, _scale_1d(grad_b_dec, self.learning_rate))
        self.W_enc = [[self.W_enc[i][j] - self.learning_rate * grad_W_enc[i][j]
                       for j in range(len(self.W_enc[0]))] for i in range(len(self.W_enc))]
        self.b_enc = _sub_1d(self.b_enc, _scale_1d(grad_b_enc, self.learning_rate))

        # Contrastive separation from negative samples
        sep_loss = 0.0
        if negative_samples:
            for neg in negative_samples:
                sep_loss += self.separation_loss(x, neg)
                z_neg = self.encode(neg)
                diff = _sub_1d(z, z_neg)
                dist = norm(diff)
                if dist < self.separation_margin and dist > 1e-8:
                    push_dir = _scale_1d(diff, 1.0 / dist)
                    grad_sep = _scale_1d(push_dir, self.separation_margin - dist)
                    grad_z_sep = _mul_1d(grad_sep, [1.0 - zz * zz for zz in z])
                    self.W_enc = [[self.W_enc[i][j] - self.learning_rate * _outer(grad_z_sep, x)[i][j]
                                   for j in range(len(self.W_enc[0]))] for i in range(len(self.W_enc))]
                    self.b_enc = _sub_1d(self.b_enc, _scale_1d(grad_z_sep, self.learning_rate))

        recon_loss = sum((recon[i] - x[i]) ** 2 for i in range(len(recon))) / max(1, len(recon))

        self.training_step += 1
        return {"recon_loss": recon_loss, "sep_loss": sep_loss}

    def store(self, record: EpisodicRecord) -> None:
        """Store a memory record."""
        self.memories[record.memory_id] = record
        bottleneck = self.encode(record.sensory_features)
        self._index[record.memory_id] = bottleneck

    def retrieve(self, query: List[float], k: int = 5, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Pattern completion: retrieve similar memories from partial cue."""
        query_bottleneck = self.encode(query)

        similarities = []
        for mem_id, mem_bottleneck in self._index.items():
            qn = norm(query_bottleneck)
            mn = norm(mem_bottleneck)
            sim = _dot(query_bottleneck, mem_bottleneck) / (qn * mn + 1e-8)
            if sim >= threshold:
                similarities.append((mem_id, float(sim)))

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
    """Working buffer for recent experiences before hippocampal encoding."""

    def __init__(self, capacity: int = 100) -> None:
        self.capacity = capacity
        self.buffer: List[Dict[str, Any]] = []

    def add(self, experience: Dict[str, Any]) -> None:
        experience["buffer_timestamp"] = time.time()
        self.buffer.append(experience)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def flush(self) -> List[Dict[str, Any]]:
        experiences = list(self.buffer)
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
