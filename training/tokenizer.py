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


class BPETokenizer:
    """Learned byte-level BPE tokenizer, pure stdlib, trained on OWNED-LOCAL text only.

    No external vocab, no cloud tokenizers. The base vocabulary is the 256 byte
    values; merge rules are learned from the supplied corpus (which, by the
    training charter, is the repo's own foundation texts). The learned state is
    serializable so inference can rebuild the exact tokenizer from the checkpoint
    -- nothing is fetched or imported at load time.

    This replaces the fixed 256-byte scheme so the model can form real sub-word
    units instead of emitting one byte per step.
    """

    def __init__(self, target_vocab: int = 1024):
        self.base = 256
        self.target_vocab = target_vocab
        self.merges: list[tuple[int, int]] = []  # ordered; new id == base + index
        self._id_to_bytes: dict[int, bytes] = {}
        self._merge_lookup: dict[tuple[int, int], int] = {}

    # ---- training ---------------------------------------------------------
    def train(self, texts, target_vocab: int | None = None) -> "BPETokenizer":
        if target_vocab is not None:
            self.target_vocab = target_vocab
        # Initial sequences: each whitespace-delimited word as a list of byte ids.
        # Word boundaries are not merged, so we never fuse across spaces.
        sequences: list[list[int]] = []
        for text in texts:
            for word in text.split():
                b = list(word.encode("utf-8", errors="replace"))
                if b:
                    sequences.append(b)
        while (self.base + len(self.merges)) < self.target_vocab:
            pair_counts: dict[tuple[int, int], int] = {}
            for seq in sequences:
                for i in range(len(seq) - 1):
                    pair = (seq[i], seq[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1
            if not pair_counts:
                break
            # Highest frequency; deterministic tie-break by smallest pair value.
            best_pair = max(
                pair_counts, key=lambda p: (pair_counts[p], -p[0], -p[1])
            )
            new_id = self.base + len(self.merges)
            self.merges.append(best_pair)
            a, b = best_pair
            for seq in sequences:
                i = 0
                out = []
                while i < len(seq):
                    if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(seq[i])
                        i += 1
                seq[:] = out
        self._build_index()
        return self

    def _build_index(self) -> None:
        self._id_to_bytes = {i: bytes([i]) for i in range(self.base)}
        for idx, (a, b) in enumerate(self.merges):
            self._id_to_bytes[self.base + idx] = (
                self._id_to_bytes[a] + self._id_to_bytes[b]
            )
        self._merge_lookup = {pair: idx for idx, pair in enumerate(self.merges)}

    # ---- encode / decode --------------------------------------------------
    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        words = text.split()
        for wi, word in enumerate(words):
            seq = list(word.encode("utf-8", errors="replace"))
            changed = True
            while changed:
                changed = False
                best_idx = None
                best_pos = None
                for i in range(len(seq) - 1):
                    pair = (seq[i], seq[i + 1])
                    idx = self._merge_lookup.get(pair)
                    if idx is not None and (best_idx is None or idx < best_idx):
                        best_idx = idx
                        best_pos = i
                if best_idx is not None and best_pos is not None:
                    new_id = self.base + best_idx
                    seq = seq[:best_pos] + [new_id] + seq[best_pos + 2 :]
                    changed = True
            out.extend(seq)
            if wi < len(words) - 1:
                out.append(32)  # space token between words
        return out

    def decode(self, ids: list[int]) -> str:
        buf = bytearray()
        for i in ids:
            buf += self._id_to_bytes.get(i, b"")
        return buf.decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return self.base + len(self.merges)

    def state_dict(self) -> dict:
        return {
            "type": "bpe",
            "base": self.base,
            "target_vocab": self.target_vocab,
            "merges": [[a, b] for (a, b) in self.merges],
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> "BPETokenizer":
        tok = cls(target_vocab=d.get("target_vocab", 1024))
        tok.base = d.get("base", 256)
        tok.merges = [tuple(m) for m in d.get("merges", [])]
        tok._build_index()
        return tok
