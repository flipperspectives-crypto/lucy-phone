#!/usr/bin/env python3
"""Standalone on-device inference for Lucy's TinyTransformer.

SELF-CONTAINED: no third-party packages (numpy/torch) and no project imports
(lucy_core/lucy_edge). Standard library only. By construction this file makes
NO network calls -- it cannot phone home -- which satisfies Lucy's phone-only,
fail-closed, no-public-cloud stance.

It loads a checkpoint JSON (the format written by ``training.train`` / the
model's ``state_dict()`` plus optional ``trained_steps`` / ``final_loss``
metadata), asserts the model dimensions match the checkpoint before running,
prints the training metadata, and runs local generation.

Usage:
    python3 tiny_infer.py --checkpoint training/checkpoints/latest.json \
        --prompt "lauren" --max-new 24
    python3 tiny_infer.py --checkpoint ckpt.json --prompt "hi" --temperature 0.8
    python3 tiny_infer.py --checkpoint ckpt.json --prompt "x" \
        --emit-provenance out.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys

# ---------------------------------------------------------------------------
# Pure-stdlib model (mirrors training/tiny_transformer.py, inference-only path)
# ---------------------------------------------------------------------------

EPS = 1e-5


def _bmm(X, W):
    """Batched matmul: X (B,T,in) @ W (in,out) -> (B,T,out)."""
    B, T = len(X), len(X[0])
    out = [[[0.0] * len(W[0]) for _ in range(T)] for _ in range(B)]
    for b in range(B):
        for t in range(T):
            row = X[b][t]
            for j in range(len(W[0])):
                s = 0.0
                for k in range(len(W)):
                    s += row[k] * W[k][j]
                out[b][t][j] = s
    return out


def _add(a, b):
    if a and isinstance(a[0], list) and isinstance(a[0][0], list):
        B, T, C = len(a), len(a[0]), len(a[0][0])
        return [
            [[a[bi][ti][ci] + b[bi][ti][ci] for ci in range(C)] for ti in range(T)]
            for bi in range(B)
        ]
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def _layernorm_forward(x, gain, bias):
    d = len(x)
    mean = sum(x) / d
    var = sum((v - mean) ** 2 for v in x) / d
    invstd = 1.0 / math.sqrt(var + EPS)
    xn = [(v - mean) * invstd for v in x]
    y = [xn[i] * gain[i] + bias[i] for i in range(d)]
    return y, (x, mean, invstd, xn, gain)


def _softmax_row(row):
    m = max(row)
    e = [math.exp(v - m) for v in row]
    s = sum(e)
    return [v / s for v in e]


def _relu(u):
    return [v if v > 0 else 0.0 for v in u]


class TinyTransformer:
    """Decoder-only Transformer, pre-LayerNorm, single head, ReLU FF.

    Inference-only subset: forward, generate, state_dict/load_state_dict.
    """

    def __init__(self, vocab=256, d_model=32, ctx=32, n_layers=1, ff_mult=4, seed=0):
        self.vocab = vocab
        self.d = d_model
        self.ctx = ctx
        self.L = n_layers
        self.ff = d_model * ff_mult
        rng = random.Random(seed)
        lim = 0.1

        def mat(r, c):
            return [[rng.uniform(-lim, lim) for _ in range(c)] for _ in range(r)]

        self.tok_emb = mat(vocab, d_model)
        self.pos_emb = mat(ctx, d_model)
        self.layers = []
        for _ in range(n_layers):
            self.layers.append(
                {
                    "ln1_gain": [1.0] * d_model,
                    "ln1_bias": [0.0] * d_model,
                    "Wq": mat(d_model, d_model),
                    "Wk": mat(d_model, d_model),
                    "Wv": mat(d_model, d_model),
                    "Wo": mat(d_model, d_model),
                    "ln2_gain": [1.0] * d_model,
                    "ln2_bias": [0.0] * d_model,
                    "W1": mat(d_model, self.ff),
                    "W2": mat(self.ff, d_model),
                }
            )
        self.lnf_gain = [1.0] * d_model
        self.lnf_bias = [0.0] * d_model

    def _ln_forward(self, x, gain, bias):
        B, T = len(x), len(x[0])
        out = [[[0.0] * self.d for _ in range(T)] for _ in range(B)]
        caches = []
        for b in range(B):
            for t in range(T):
                y, c = _layernorm_forward(x[b][t], gain, bias)
                out[b][t] = y
                caches.append(c)
        return out, caches

    def forward(self, batch_ids):
        B = len(batch_ids)
        T = len(batch_ids[0])
        d = self.d

        e = [[[0.0] * d for _ in range(T)] for _ in range(B)]
        for b in range(B):
            for t in range(T):
                tok = batch_ids[b][t]
                for i in range(d):
                    e[b][t][i] = self.tok_emb[tok][i] + self.pos_emb[t][i]

        cache_layers = []
        x = e
        for layer in self.layers:
            h, ln1_cache = self._ln_forward(x, layer["ln1_gain"], layer["ln1_bias"])
            q = _bmm(h, layer["Wq"])
            k = _bmm(h, layer["Wk"])
            v = _bmm(h, layer["Wv"])
            scale = 1.0 / math.sqrt(d)
            w = [[[0.0] * T for _ in range(T)] for _ in range(B)]
            for b in range(B):
                for t in range(T):
                    logits_t = [
                        sum(q[b][t][i] * k[b][s][i] for i in range(d)) * scale
                        for s in range(t + 1)
                    ]
                    w_bt = _softmax_row(logits_t)
                    for s in range(t + 1):
                        w[b][t][s] = w_bt[s]
            a = [[[0.0] * d for _ in range(T)] for _ in range(B)]
            for b in range(B):
                for t in range(T):
                    for s in range(t + 1):
                        ws = w[b][t][s]
                        for i in range(d):
                            a[b][t][i] += ws * v[b][s][i]
            o = _bmm(a, layer["Wo"])
            x_attn = _add(x, o)

            h2, ln2_cache = self._ln_forward(x_attn, layer["ln2_gain"], layer["ln2_bias"])
            u = _bmm(h2, layer["W1"])
            z = [[_relu(row) for row in ub] for ub in u]
            y = _bmm(z, layer["W2"])
            x = _add(x_attn, y)
            cache_layers.append(
                {
                    "h": h,
                    "w": w,
                    "v": v,
                    "ln2_cache": ln2_cache,
                    "h2": h2,
                    "u": u,
                    "z": z,
                }
            )

        hf, lnf_cache = self._ln_forward(x, self.lnf_gain, self.lnf_bias)
        logits = [[[0.0] * self.vocab for _ in range(T)] for _ in range(B)]
        for b in range(B):
            for t in range(T):
                for v in range(self.vocab):
                    logits[b][t][v] = sum(hf[b][t][i] * self.tok_emb[v][i] for i in range(d))
        return logits

    def generate(self, ids, max_new=24, temperature=0.0):
        if not ids:
            # Empty/whitespace-only prompt encodes to zero tokens; returning nothing
            # avoids building a zero-length window that would crash the forward pass.
            return []
        ctx_ids = ids[-self.ctx :] if len(ids) >= self.ctx else ids
        generated = []
        for _ in range(max_new):
            window = ctx_ids[-self.ctx :]
            logits = self.forward([window])
            last = logits[0][-1]
            if temperature and temperature > 0:
                mx = max(last)
                exps = [math.exp((x - mx) / temperature) for x in last]
                s = sum(exps)
                r = random.random()
                acc = 0.0
                nxt = len(last) - 1
                for i, ex in enumerate(exps):
                    acc += ex / s
                    if acc >= r:
                        nxt = i
                        break
            else:
                nxt = max(range(len(last)), key=lambda v: last[v])
            generated.append(nxt)
            ctx_ids = ctx_ids + [nxt]
        return generated

    def state_dict(self):
        return {
            "vocab": self.vocab,
            "d": self.d,
            "ctx": self.ctx,
            "L": self.L,
            "ff": self.ff,
            "tok_emb": self.tok_emb,
            "pos_emb": self.pos_emb,
            "layers": self.layers,
            "lnf_gain": self.lnf_gain,
            "lnf_bias": self.lnf_bias,
        }

    def load_state_dict(self, sd):
        self.vocab = sd["vocab"]
        self.d = sd["d"]
        self.ctx = sd["ctx"]
        self.L = sd["L"]
        self.ff = sd["ff"]
        self.tok_emb = sd["tok_emb"]
        self.pos_emb = sd["pos_emb"]
        self.layers = sd["layers"]
        self.lnf_gain = sd["lnf_gain"]
        self.lnf_bias = sd["lnf_bias"]


class ByteTokenizer:
    """Byte-level tokenizer, vocab 256 (mirrors training/tokenizer.py)."""

    vocab_size = 256

    def encode(self, text):
        if isinstance(text, bytes):
            return list(text)
        return list(text.encode("utf-8", errors="replace"))

    def decode(self, ids):
        return bytes(b & 0xFF for b in ids).decode("utf-8", errors="replace")


class BPETokenizer:
    """Learned byte-level BPE tokenizer (standalone mirror of training/tokenizer.py).

    Reconstructed from the checkpoint's ``tokenizer`` state so on-device inference
    uses the exact vocabulary the model was trained with -- no external vocab, no
    network. Pure stdlib only.
    """

    def __init__(self, target_vocab=1024):
        self.base = 256
        self.target_vocab = target_vocab
        self.merges = []
        self._id_to_bytes = {}
        self._merge_lookup = {}

    def _build_index(self):
        self._id_to_bytes = {i: bytes([i]) for i in range(self.base)}
        for idx, (a, b) in enumerate(self.merges):
            self._id_to_bytes[self.base + idx] = (
                self._id_to_bytes[a] + self._id_to_bytes[b]
            )
        self._merge_lookup = {pair: idx for idx, pair in enumerate(self.merges)}

    def encode(self, text):
        out = []
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
                    seq = seq[:best_pos] + [new_id] + seq[best_pos + 2:]
                    changed = True
            out.extend(seq)
            if wi < len(words) - 1:
                out.append(32)
        return out

    def decode(self, ids):
        buf = bytearray()
        for i in ids:
            buf += self._id_to_bytes.get(i, b"")
        return buf.decode("utf-8", errors="replace")

    @property
    def vocab_size(self):
        return self.base + len(self.merges)

    @classmethod
    def from_state_dict(cls, d):
        tok = cls(target_vocab=d.get("target_vocab", 1024))
        tok.base = d.get("base", 256)
        tok.merges = [tuple(m) for m in d.get("merges", [])]
        tok._build_index()
        return tok


def _load_tokenizer(sd):
    """Reconstruct the tokenizer recorded in a checkpoint (byte fallback)."""
    tok_state = sd.get("tokenizer")
    if tok_state and tok_state.get("type") == "bpe":
        return BPETokenizer.from_state_dict(tok_state)
    return ByteTokenizer()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Standalone on-device TinyTransformer inference.")
    ap.add_argument("--checkpoint", required=True, help="Path to checkpoint JSON.")
    ap.add_argument("--prompt", default="", help="Text prompt.")
    ap.add_argument("--max-new", type=int, default=24, help="Tokens to generate.")
    ap.add_argument("--temperature", type=float, default=0.0, help="0.0 = greedy.")
    ap.add_argument("--emit-provenance", default=None, help="Write a JSON provenance ledger entry here.")
    # Optional dimension overrides for CI assertions (must match checkpoint).
    ap.add_argument("--vocab", type=int, default=None)
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--ctx", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--ff", type=int, default=None)
    args = ap.parse_args(argv)

    with open(args.checkpoint, "r", encoding="utf-8") as f:
        sd = json.load(f)

    # --- training metadata (new requirement #1) ---------------------------
    if "trained_steps" in sd:
        print(f"[checkpoint] trained_steps = {sd['trained_steps']}")
    else:
        print("[checkpoint] trained_steps = unrecorded")
    if "final_loss" in sd:
        print(f"[checkpoint] final_loss    = {sd['final_loss']:.6f}")
    else:
        print("[checkpoint] final_loss    = unrecorded")

    # --- assert dims match checkpoint (new requirement #2) -----------------
    model = TinyTransformer(
        vocab=sd["vocab"], d_model=sd["d"], ctx=sd["ctx"], n_layers=sd["L"], ff_mult=sd["ff"]
    )
    model.load_state_dict(sd)

    expected = {
        "vocab": model.vocab,
        "d_model": model.d,
        "ctx": model.ctx,
        "layers": model.L,
        "ff": model.ff,
    }
    got = {
        "vocab": sd["vocab"],
        "d_model": sd["d"],
        "ctx": sd["ctx"],
        "layers": sd["L"],
        "ff": sd["ff"],
    }
    if expected != got:
        raise AssertionError(f"checkpoint dimension mismatch: model={expected} checkpoint={got}")

    overrides = {
        "vocab": args.vocab,
        "d_model": args.d_model,
        "ctx": args.ctx,
        "layers": args.layers,
        "ff": args.ff,
    }
    for name, val in overrides.items():
        if val is not None and val != expected[name]:
            raise AssertionError(
                f"--{name} override ({val}) does not match checkpoint ({expected[name]})"
            )
    print(f"[config] dims OK: vocab={expected['vocab']} d={expected['d_model']} "
          f"ctx={expected['ctx']} layers={expected['layers']} ff={expected['ff']}")

    # --- generation -------------------------------------------------------
    tok = _load_tokenizer(sd)
    prompt_ids = tok.encode(args.prompt)
    out_ids = model.generate(prompt_ids, max_new=args.max_new, temperature=args.temperature)
    text = tok.decode(out_ids)
    print("--- generated ---")
    print(text)
    print("-----------------")
    print(f"[info] prompt_tokens={len(prompt_ids)} generated_tokens={len(out_ids)}")

    # --- optional provenance ledger entry --------------------------------
    if args.emit_provenance:
        ckpt_sha = _sha256_file(args.checkpoint)
        prompt_sha = hashlib.sha256(args.prompt.encode("utf-8")).hexdigest()
        out_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        record = {
            "source": "USER_STATED",
            "inference": "local_lucy_tiny_standalone",
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": ckpt_sha,
            "prompt_sha256": prompt_sha,
            "output_sha256": out_sha,
            "model_config": expected,
            "max_new": args.max_new,
            "temperature": args.temperature,
        }
        with open(args.emit_provenance, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        print(f"[provenance] wrote {args.emit_provenance}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
