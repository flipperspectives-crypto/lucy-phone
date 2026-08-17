"""Dependency-free byte-level tokenizer.

Lucy trains on her OWN local bytes -- no external vocab, no cloud tokenizers.
A byte-level scheme keeps the vocabulary fixed at 256 and means any local text
she is allowed to see can be tokenized without fetching a model from elsewhere.
"""

from __future__ import annotations


class ByteTokenizer:
    """Maps text <-> list[int] over raw bytes (vocab size 256)."""

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        if isinstance(text, bytes):
            return list(text)
        return list(text.encode("utf-8", errors="replace"))

    def decode(self, ids: list[int]) -> str:
        return bytes(b & 0xFF for b in ids).decode("utf-8", errors="replace")

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "ByteTokenizer(vocab=256)"
