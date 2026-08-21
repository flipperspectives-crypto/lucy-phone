"""A tiny from-scratch Transformer, implemented in pure Python (standard library only).

Why pure Python, no numpy/torch:
  * The sandbox here forbids installing native ML libraries, and Lucy's safe-AI
    stance prefers auditable, dependency-free code anyway.
  * The model is intentionally SMALL (tens of thousands of parameters) so it
    trains on CPU with ~2 GiB RAM -- a real, local, on-device model, not a
    frontier model.  It proves the *pathway*: owned data + provenance + real
    gradient descent, trained where she runs.

Architecture: decoder-only Transformer, pre-LayerNorm, single attention head,
ReLU feed-forward, byte-level tied output.  Forward saves a cache; backward
computes exact gradients.  ``grad_check`` validates the backward pass against
finite differences so we never ship a silently-broken trainer.
"""

from __future__ import annotations

import math
import random
from operator import mul

EPS = 1e-5


def _matmul(a, b):
    # Transpose once, then dot products via C-level map(mul, ...). The
    # multiply/add order over k is unchanged vs the naive triple loop, so
    # results are bit-identical -- this is purely an interpreter-exit speedup.
    cols = list(zip(*b))
    return [[sum(map(mul, row, col)) for col in cols] for row in a]


def _transpose(a):
    return [[a[i][j] for i in range(len(a))] for j in range(len(a[0]))]


def _add(a, b):
    # elementwise add; handles 3D tensors (list+list would concatenate, not add)
    if a and isinstance(a[0], list) and isinstance(a[0][0], list):
        B, T, C = len(a), len(a[0]), len(a[0][0])
        return [
            [[a[bi][ti][ci] + b[bi][ti][ci] for ci in range(C)] for ti in range(T)]
            for bi in range(B)
        ]
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def _copy(a):
    return [list(row) for row in a]


def _bmm(X, W):
    """Batched matmul: X (B,T,in) @ W (in,out) -> (B,T,out).

    Hot path: W is transposed once per call and every dot product runs through
    C-level ``map(mul, ...)`` so the inner loop leaves the interpreter. The
    multiply/add order over k is unchanged, so results are bit-identical to
    the naive triple loop this replaces.
    """
    cols = list(zip(*W))
    return [
        [[sum(map(mul, xrow, col)) for col in cols] for xrow in batch]
        for batch in X
    ]


def _outer_sum(X, Y):
    """Sum over b,t of X[b][t] (in) outer Y[b][t] (out) -> (in,out).

    Inner j loop is vectorized with zip at C level; the accumulation order
    over b,t per element is unchanged, so results are bit-identical.
    """
    B, T = len(X), len(X[0])
    inn = len(X[0][0])
    M = [[0.0] * len(Y[0][0]) for _ in range(inn)]
    for b in range(B):
        for t in range(T):
            xb = X[b][t]
            yb = Y[b][t]
            for i in range(inn):
                xi = xb[i]
                M[i] = [m + xi * yj for m, yj in zip(M[i], yb)]
    return M


def _is_matrix(a):
    return bool(a) and isinstance(a[0], list)


def _zeros_like(a):
    if _is_matrix(a):
        return [[0.0 for _ in row] for row in a]
    return [0.0 for _ in a]


def _flatten(p):
    if _is_matrix(p):
        return [v for row in p for v in row]
    return list(p)


def _set_at(p, idx, value):
    if _is_matrix(p):
        p[idx // len(p[0])][idx % len(p[0])] = value
    else:
        p[idx] = value


def _flatten_bt(tensor, B, T, C):
    return [[tensor[b][t][c] for c in range(C)] for b in range(B) for t in range(T)]


def _layernorm_forward(x, gain, bias):
    d = len(x)
    mean = sum(x) / d
    var = sum((v - mean) ** 2 for v in x) / d
    invstd = 1.0 / math.sqrt(var + EPS)
    xn = [(v - mean) * invstd for v in x]
    y = [xn[i] * gain[i] + bias[i] for i in range(d)]
    return y, (x, mean, invstd, xn, gain)


def _layernorm_backward(dy, cache):
    x, mean, invstd, xn, gain = cache
    d = len(x)
    dg = [dy[i] * xn[i] for i in range(d)]
    db = list(dy)
    dxn = [dy[i] * gain[i] for i in range(d)]
    dvar = sum(dxn[i] * (x[i] - mean) for i in range(d)) * (-0.5) * invstd**3
    dmean = sum(dxn[i] * -invstd for i in range(d)) + dvar * sum(
        -2.0 * (x[i] - mean) for i in range(d)
    ) / d
    dx = [
        dxn[i] * invstd + dvar * 2.0 * (x[i] - mean) / d + dmean / d
        for i in range(d)
    ]
    return dx, dg, db


def _softmax_row(row):
    m = max(row)
    e = [math.exp(v - m) for v in row]
    s = sum(e)
    return [v / s for v in e]


def _softmax_backward(w, dw):
    k = sum(dw[i] * w[i] for i in range(len(w)))
    return [w[i] * (dw[i] - k) for i in range(len(w))]


def _relu(u):
    return [v if v > 0 else 0.0 for v in u]


def _relu_backward(dz, u):
    return [dz[i] if u[i] > 0 else 0.0 for i in range(len(u))]


class TinyTransformer:
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
        self.params = {name: p for name, p in self._param_blocks()}
        self._init_grad()

    def _param_blocks(self):
        yield ("tok_emb", self.tok_emb)
        yield ("pos_emb", self.pos_emb)
        for i, layer in enumerate(self.layers):
            for k in (
                "ln1_gain",
                "ln1_bias",
                "Wq",
                "Wk",
                "Wv",
                "Wo",
                "ln2_gain",
                "ln2_bias",
                "W1",
                "W2",
            ):
                yield (f"layer{i}.{k}", layer[k])
        yield ("lnf_gain", self.lnf_gain)
        yield ("lnf_bias", self.lnf_bias)

    def _init_grad(self):
        self.grad = {name: _zeros_like(p) for name, p in self._param_blocks()}

    def zero_grad(self):
        for name in self.grad:
            g = self.grad[name]
            if _is_matrix(g):
                for r, row in enumerate(g):
                    g[r] = [0.0] * len(row)
            else:
                self.grad[name] = [0.0] * len(g)

    def apply_grad(self, lr):
        """Shape-aware SGD step (handles both matrices and vectors).

        Element updates are vectorized with zip at C level; the per-element
        operation (p -= lr * g) and its order are unchanged, so results are
        bit-identical to the scalar double loop.
        """
        for name, p in self._param_blocks():
            g = self.grad[name]
            if _is_matrix(p):
                for r in range(len(p)):
                    p[r] = [v - lr * gv for v, gv in zip(p[r], g[r])]
            else:
                p[:] = [v - lr * gv for v, gv in zip(p, g)]

    def forward(self, batch_ids):
        B = len(batch_ids)
        T = len(batch_ids[0])
        d = self.d

        e = []
        for b in range(B):
            e_b = []
            tok_row = self.tok_emb
            pos_row = self.pos_emb
            for t in range(T):
                tok = batch_ids[b][t]
                e_b.append([tv + pv for tv, pv in zip(tok_row[tok], pos_row[t])])
            e.append(e_b)

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
                    q_bt = q[b][t]
                    logits_t = [
                        sum(map(mul, q_bt, k[b][s])) * scale
                        for s in range(t + 1)
                    ]
                    w_bt = _softmax_row(logits_t)
                    w[b][t] = w_bt
            a = []
            for b in range(B):
                a_b = []
                for t in range(T):
                    acc = [0.0] * d
                    w_bt = w[b][t]
                    for s in range(t + 1):
                        ws = w_bt[s]
                        v_bs = v[b][s]
                        acc = [ai + ws * vi for ai, vi in zip(acc, v_bs)]
                    a_b.append(acc)
                a.append(a_b)
            o = _bmm(a, layer["Wo"])
            x_attn = _add(x, o)

            h2, ln2_cache = self._ln_forward(x_attn, layer["ln2_gain"], layer["ln2_bias"])
            u = _bmm(h2, layer["W1"])
            z = [[_relu(row) for row in ub] for ub in u]
            y = _bmm(z, layer["W2"])
            x = _add(x_attn, y)
            cache_layers.append(
                {
                    "x_in": x,
                    "ln1_cache": ln1_cache,
                    "h": h,
                    "q": q,
                    "k": k,
                    "v": v,
                    "w": w,
                    "a": a,
                    "ln2_cache": ln2_cache,
                    "h2": h2,
                    "u": u,
                    "z": z,
                }
            )

        hf, lnf_cache = self._ln_forward(x, self.lnf_gain, self.lnf_bias)
        logits = []
        for b in range(B):
            logits_b = []
            for t in range(T):
                hf_bt = hf[b][t]
                logits_b.append([sum(map(mul, hf_bt, emb_v)) for emb_v in self.tok_emb])
            logits.append(logits_b)
        cache = {"hf": hf, "lnf_cache": lnf_cache, "layers": cache_layers}
        return logits, cache

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

    def _ln_backward(self, dy, caches):
        B, T = len(dy), len(dy[0])
        dx = [[[0.0] * self.d for _ in range(T)] for _ in range(B)]
        dg = [0.0] * self.d
        db = [0.0] * self.d
        for b in range(B):
            for t in range(T):
                dxi, dgi, dbi = _layernorm_backward(dy[b][t], caches[b * T + t])
                dx[b][t] = dxi
                for i in range(self.d):
                    dg[i] += dgi[i]
                    db[i] += dbi[i]
        return dx, dg, db

    def cross_entropy(self, logits, targets):
        B, T = len(targets), len(targets[0])
        loss = 0.0
        dlogits = []
        V = self.vocab
        for b in range(B):
            dlogits_b = []
            for t in range(T):
                row = logits[b][t]
                m = max(row)
                e = [math.exp(v - m) for v in row]
                s = sum(e)
                loss += -math.log(e[targets[b][t]] / s)
                dr = [ev / s for ev in e]
                dr[targets[b][t]] -= 1.0
                dlogits_b.append(dr)
            dlogits.append(dlogits_b)
        inv = 1.0 / (B * T)
        for b in range(B):
            for t in range(T):
                dlogits[b][t] = [dv * inv for dv in dlogits[b][t]]
        return loss * inv, dlogits

    def backward(self, batch_ids, logits, cache, dlogits):
        B, T, d, V = len(batch_ids), len(batch_ids[0]), self.d, self.vocab
        self.zero_grad()
        g = self.grad

        d_hf = _bmm(dlogits, self.tok_emb)
        d_tok_emb_acc = _outer_sum(cache["hf"], dlogits)
        g_tok = g["tok_emb"]
        for v in range(V):
            col = [d_tok_emb_acc[i][v] for i in range(d)]
            g_tok[v] = [x + y for x, y in zip(g_tok[v], col)]

        # final norm backward uses d_hf (the grad w.r.t hf), not dlogits
        dx, dlnf_gain, dlnf_bias = self._ln_backward(d_hf, cache["lnf_cache"])
        for i in range(d):
            g["lnf_gain"][i] += dlnf_gain[i]
            g["lnf_bias"][i] += dlnf_bias[i]

        for li in reversed(range(self.L)):
            lc = cache["layers"][li]
            layer = self.layers[li]
            dlayer_out = dx
            dx_in = _copy(dlayer_out)

            # feed-forward branch
            dz = _bmm(dlayer_out, _transpose(layer["W2"]))
            dW2 = _outer_sum(lc["z"], dlayer_out)
            gW2 = g[f"layer{li}.W2"]
            for i in range(self.ff):
                gW2[i] = [x + y for x, y in zip(gW2[i], dW2[i])]
            du = [
                [_relu_backward(dz[b][t], lc["u"][b][t]) for t in range(T)]
                for b in range(B)
            ]
            dW1 = _outer_sum(lc["h2"], du)
            gW1 = g[f"layer{li}.W1"]
            for i in range(d):
                gW1[i] = [x + y for x, y in zip(gW1[i], dW1[i])]
            dh2 = _bmm(du, _transpose(layer["W1"]))
            dx2, dln2g, dln2b = self._ln_backward(dh2, lc["ln2_cache"])
            g[f"layer{li}.ln2_gain"] = [x + y for x, y in zip(g[f"layer{li}.ln2_gain"], dln2g)]
            g[f"layer{li}.ln2_bias"] = [x + y for x, y in zip(g[f"layer{li}.ln2_bias"], dln2b)]
            dx_in = _add(dx_in, dx2)

            # attention branch: grad w.r.t o is G (layer output) + dx2 (ff path)
            do = _add(dlayer_out, dx2)
            da = _bmm(do, _transpose(layer["Wo"]))
            dWo = _outer_sum(lc["a"], do)
            gWo = g[f"layer{li}.Wo"]
            for i in range(d):
                gWo[i] = [x + y for x, y in zip(gWo[i], dWo[i])]

            dv = [[[0.0] * d for _ in range(T)] for _ in range(B)]
            dw = []
            for b in range(B):
                da_b = da[b]
                v_b = lc["v"][b]
                zeros_tail = [0.0] * (T - 1)
                dw.append(
                    [
                        [sum(map(mul, da_b[t], v_b[s])) for s in range(t + 1)]
                        + zeros_tail[: T - t - 1]
                        for t in range(T)
                    ]
                )
            for b in range(B):
                dv_b = dv[b]
                w_b = lc["w"][b]
                da_b = da[b]
                for s in range(T):
                    acc = [0.0] * d
                    for t in range(s, T):
                        ws = w_b[t][s]
                        da_bt = da_b[t]
                        acc = [x + ws * y for x, y in zip(acc, da_bt)]
                    dv_b[s] = acc
            dscores = [[[0.0] * T for _ in range(T)] for _ in range(B)]
            for b in range(B):
                for t in range(T):
                    dscores[b][t] = _softmax_backward(lc["w"][b][t], dw[b][t])
            scale = 1.0 / math.sqrt(d)
            dq = [[[0.0] * d for _ in range(T)] for _ in range(B)]
            dk = [[[0.0] * d for _ in range(T)] for _ in range(B)]
            for b in range(B):
                dscores_b = dscores[b]
                k_b = lc["k"][b]
                q_b = lc["q"][b]
                dq_b = dq[b]
                dk_b = dk[b]
                for t in range(T):
                    dscores_bt = dscores_b[t]
                    q_bt = q_b[t]
                    for s in range(t + 1):
                        c = dscores_bt[s] * scale
                        k_bs = k_b[s]
                        dq_b[t] = [x + c * y for x, y in zip(dq_b[t], k_bs)]
                        dk_b[s] = [x + c * y for x, y in zip(dk_b[s], q_bt)]
            dWq = _outer_sum(lc["h"], dq)
            dWk = _outer_sum(lc["h"], dk)
            dWv = _outer_sum(lc["h"], dv)
            gWq = g[f"layer{li}.Wq"]
            gWk = g[f"layer{li}.Wk"]
            gWv = g[f"layer{li}.Wv"]
            for i in range(d):
                gWq[i] = [x + y for x, y in zip(gWq[i], dWq[i])]
                gWk[i] = [x + y for x, y in zip(gWk[i], dWk[i])]
                gWv[i] = [x + y for x, y in zip(gWv[i], dWv[i])]
            dh = _bmm(dq, _transpose(layer["Wq"]))
            dh = _add(dh, _bmm(dk, _transpose(layer["Wk"])))
            dh = _add(dh, _bmm(dv, _transpose(layer["Wv"])))
            dx_ln, dln1g, dln1b = self._ln_backward(dh, lc["ln1_cache"])
            g[f"layer{li}.ln1_gain"] = [x + y for x, y in zip(g[f"layer{li}.ln1_gain"], dln1g)]
            g[f"layer{li}.ln1_bias"] = [x + y for x, y in zip(g[f"layer{li}.ln1_bias"], dln1b)]
            dx_in = _add(dx_in, dx_ln)
            dx = dx_in

        for b in range(B):
            for t in range(T):
                tok = batch_ids[b][t]
                dx_bt = dx[b][t]
                g["tok_emb"][tok] = [x + y for x, y in zip(g["tok_emb"][tok], dx_bt)]
                g["pos_emb"][t] = [x + y for x, y in zip(g["pos_emb"][t], dx_bt)]

    def generate(self, ids, max_new=24):
        ctx_ids = ids[-self.ctx :]
        for _ in range(max_new):
            batch = [ctx_ids[-self.ctx :]] if len(ctx_ids) >= self.ctx else [ctx_ids]
            logits, _ = self.forward(batch)
            last = logits[0][-1]
            nxt = max(range(self.vocab), key=lambda v: last[v])
            ctx_ids = ctx_ids + [nxt]
        return ctx_ids

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
        self.params = {name: p for name, p in self._param_blocks()}
        self._init_grad()

    def grad_check(self, batch_ids, targets, eps=1e-4, tol=1e-2):
        logits, cache = self.forward(batch_ids)
        loss, dlogits = self.cross_entropy(logits, targets)
        self.backward(batch_ids, logits, cache, dlogits)

        max_err = 0.0
        checked = 0
        for name, param in self._param_blocks():
            flat = _flatten(param)
            gflat = _flatten(self.grad[name])
            for idx in range(0, len(flat), max(1, len(flat) // 5)):
                orig = flat[idx]
                self._set_param(name, idx, orig + eps)
                lp, _ = self.cross_entropy(self.forward(batch_ids)[0], targets)
                self._set_param(name, idx, orig - eps)
                lm, _ = self.cross_entropy(self.forward(batch_ids)[0], targets)
                self._set_param(name, idx, orig)
                num = (lp - lm) / (2 * eps)
                ana = gflat[idx]
                max_err = max(max_err, abs(num - ana))
                checked += 1
        return checked > 0 and max_err < tol, max_err

    def _set_param(self, name, idx, value):
        if name == "tok_emb":
            _set_at(self.tok_emb, idx, value)
        elif name == "pos_emb":
            _set_at(self.pos_emb, idx, value)
        elif name == "lnf_gain":
            _set_at(self.lnf_gain, idx, value)
        elif name == "lnf_bias":
            _set_at(self.lnf_bias, idx, value)
        else:
            li = int(name.split(".")[0][len("layer") :])
            key = name.split(".", 1)[1]
            _set_at(self.layers[li][key], idx, value)
