"""Local Lucy inference provider (real, from-scratch model; not simulated).

Loads a checkpoint produced by ``training.train`` and runs inference with the
pure-Python TinyTransformer. Implements the lucy_edge provider interface so it
can be registered alongside mock without faking any capability.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from lucy_edge.providers.base import (
    BaseProvider,
    Capability,
    CapabilityUnavailable,
    ChatResponse,
    GenerationResult,
    ModelInfo,
    ProviderHealth,
    StreamChunk,
)

from .tiny_transformer import TinyTransformer
from .tokenizer import ByteTokenizer, BPETokenizer

# Turn-boundary markers: the corpus is authored as "USER: ...\nLUCY: ...\n\n"
# blocks, so a well-formed answer ends when the model starts the next turn.
_TURN_MARKERS = ("\nUSER", "\n\n")


def strip_at_turn_boundary(text: str) -> str:
    """Cut generated text at the first next-turn marker and trim trailing space."""
    cut = len(text)
    for marker in _TURN_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut].rstrip()


class LocalLucyProvider(BaseProvider):
    """Runs a locally trained TinyTransformer checkpoint.

    This is a genuine inference backend: ``simulated`` is False and it reports
    exactly the capabilities it implements. It never claims remote/cloud access.
    """

    name = "local_lucy"
    kind = "local"
    simulated = False

    def __init__(self, checkpoint_path: str | Path, model_name: str = "lucy-local"):
        self.checkpoint_path = Path(checkpoint_path)
        self.model_name = model_name
        self._tok = ByteTokenizer()
        self._model: Optional[TinyTransformer] = None
        self._loaded = False

    # --- lifecycle ---------------------------------------------------------
    def _ensure_loaded(self) -> TinyTransformer:
        if not self._loaded:
            if not self.checkpoint_path.exists():
                raise CapabilityUnavailable(
                    Capability.GENERATE, f"checkpoint missing: {self.checkpoint_path}"
                )
            sd = json.loads(self.checkpoint_path.read_text())
            self._model = TinyTransformer(
                vocab=sd["vocab"],
                d_model=sd["d"],
                ctx=sd["ctx"],
                n_layers=sd["L"],
                ff_mult=sd["ff"],
            )
            self._model.load_state_dict(sd)
            # Reconstruct the exact tokenizer used in training from the checkpoint,
            # so generated text uses the same vocabulary. Falls back to the byte
            # tokenizer for legacy checkpoints that carry no tokenizer state.
            if "tokenizer" in sd:
                self._tok = BPETokenizer.from_state_dict(sd["tokenizer"])
            else:
                self._tok = ByteTokenizer()
            self._loaded = True
        return self._model  # type: ignore[return-value]

    def self_check(self) -> bool:
        """Tamper-evident integrity check.

        Verifies the saved weights actually reproduce the ``probe_loss`` recorded
        in the checkpoint by ``train()``.  A checkpoint with random weights but
        faked training metadata fails this, so the audit cannot be satisfied by a
        lie.  Checkpoints without probe metadata (legacy) are trusted on metadata.
        """
        if not self.checkpoint_path.exists():
            return False
        sd = json.loads(self.checkpoint_path.read_text())
        if "probe_seq" not in sd or "probe_loss" not in sd:
            return True  # legacy checkpoint: trust training metadata
        seq = sd["probe_seq"]
        if len(seq) < 2:
            return True
        m = self._ensure_loaded()
        logits, _ = m.forward([seq[:-1]])
        loss, _ = m.cross_entropy(logits, [seq[1:]])
        return abs(loss - sd["probe_loss"]) < 0.5

    # --- capabilities ------------------------------------------------------
    def capabilities(self) -> set[Capability]:
        return {
            Capability.DETECT,
            Capability.HEALTH,
            Capability.LIST_MODELS,
            Capability.MODEL_METADATA,
            Capability.CHAT,
            Capability.GENERATE,
        }

    async def detect(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "kind": self.kind,
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_exists": self.checkpoint_path.exists(),
            "simulated": False,
        }

    async def health(self) -> ProviderHealth:
        ok = self.checkpoint_path.exists()
        return ProviderHealth(
            state="ONLINE" if ok else "OFFLINE",
            ok=ok,
            models=[self.model_name] if ok else [],
            message=None if ok else "checkpoint not found",
        )

    async def list_models(self) -> list[ModelInfo]:
        if not self.checkpoint_path.exists():
            return []
        sd = json.loads(self.checkpoint_path.read_text())
        return [
            ModelInfo(
                name=self.model_name,
                family="tiny_transformer",
                details={
                    "vocab": sd.get("vocab"),
                    "d_model": sd.get("d"),
                    "ctx": sd.get("ctx"),
                    "n_layers": sd.get("L"),
                    "ff_mult": sd.get("ff"),
                    "source": "training.train (from-scratch, local)",
                },
            )
        ]

    async def model_metadata(self, model: str) -> ModelInfo:
        models = await self.list_models()
        if not models:
            raise CapabilityUnavailable(
                Capability.MODEL_METADATA, "no local lucy checkpoint loaded"
            )
        return models[0]

    # --- inference ---------------------------------------------------------
    def _build_chat_prompt(self, messages: list[dict[str, Any]]) -> str:
        """Format the last user turn in the corpus's USER:/LUCY: structure.

        The verbose system prompt is deliberately dropped: with a 32-token
        context window it would crowd out the actual question, and the model
        was never trained on prose system prompts -- only on USER:/LUCY:
        dialogue blocks.
        """
        user_text = ""
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                user_text = str(m.get("content", ""))
                break
        user_text = " ".join(user_text.split())  # collapse newlines/runs of space
        return f"USER: {user_text}\nLUCY:"

    def _generate_from_ids(
        self,
        ids: list[int],
        max_new: int = 24,
        temperature: float = 0.0,
        stop_boundary: bool = False,
    ) -> list[int]:
        """Greedy/temperature decode from token ids.

        With *stop_boundary*, generation halts as soon as the decoded text
        crosses into the next USER turn (or a blank-line block break), so chat
        responses end where the trained format says an answer ends.
        """
        m = self._ensure_loaded()
        if not ids:
            return []
        import math as _math
        import random as _random

        ctx_ids = ids[-m.ctx :] if len(ids) >= m.ctx else ids
        generated: list[int] = []
        for _ in range(max_new):
            window = ctx_ids[-m.ctx :]
            logits, _ = m.forward([window])
            last = logits[0][-1]
            if temperature and temperature > 0:
                mx = max(last)
                exps = [_math.exp((x - mx) / temperature) for x in last]
                s = sum(exps)
                r = _random.random()
                acc = 0.0
                nxt = len(last) - 1
                for i, e in enumerate(exps):
                    acc += e / s
                    if acc >= r:
                        nxt = i
                        break
            else:
                nxt = max(range(len(last)), key=lambda v: last[v])
            generated.append(nxt)
            ctx_ids = ctx_ids + [nxt]
            if stop_boundary:
                text = self._tok.decode(generated)
                if any(marker in text for marker in _TURN_MARKERS):
                    break
        return generated

    def _generate_tokens(self, prompt: str, max_new: int = 24, temperature: float = 0.0) -> list[int]:
        ids = self._tok.encode(prompt)
        return self._generate_from_ids(ids, max_new=max_new, temperature=temperature)

    async def generate(self, prompt: str, model: str, **options: Any) -> GenerationResult:
        max_new = int(options.get("max_new_tokens", options.get("max_new", 24)))
        temperature = float(options.get("temperature", 0.0))
        ids = self._generate_tokens(prompt, max_new=max_new, temperature=temperature)
        text = self._tok.decode(ids)
        return GenerationResult(
            provider=self.name,
            model=self.model_name,
            text=text,
            eval_count=len(ids),
            simulated=False,
        )

    async def chat(self, messages: list[dict[str, Any]], model: str, **options: Any) -> ChatResponse:
        """Conversational generation in the trained USER:/LUCY: format.

        The prompt is left-truncated at the token level so the trailing
        "LUCY:" turn marker always survives the context window, and the
        response stops at the first next-turn boundary.
        """
        m = self._ensure_loaded()
        max_new = int(options.get("max_new_tokens", options.get("max_new", 32)))
        temperature = float(options.get("temperature", 0.0))
        prompt = self._build_chat_prompt(messages)
        ids = self._tok.encode(prompt)
        cap = max(4, m.ctx - 8)  # keep room to generate inside the window
        if len(ids) > cap:
            ids = ids[-cap:]
        gen_ids = self._generate_from_ids(
            ids, max_new=max_new, temperature=temperature, stop_boundary=True
        )
        text = strip_at_turn_boundary(self._tok.decode(gen_ids))
        return ChatResponse(
            provider=self.name,
            model=self.model_name,
            message=text,
            simulated=False,
        )

    async def stream_chat(
        self, messages: list[dict[str, Any]], model: str, **options: Any
    ) -> AsyncIterator[StreamChunk]:
        res = await self.generate(
            "\n".join(m.get("content", "") for m in messages if isinstance(m, dict)),
            model,
            **options,
        )
        yield StreamChunk(provider=self.name, model=self.model_name, text=res.text, done=True)
