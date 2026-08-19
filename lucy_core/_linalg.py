"""Pure-Python linear algebra for the predictive-coding brain.

Deliberately dependency-free (no numpy) so the learning core is from-scratch,
matching the project's "nothing imported from outside" sovereignty rule.  Vectors
are ``list[float]``; matrices are ``list[list[float]]``; batched tensors are
``list[list[list[float]]]`` (batch, seq, dim).

Only the small, fixed set of operations the brain actually uses are implemented.
"""

from __future__ import annotations

import math
import random

# Module-level RNG so behaviour is deterministic once seeded (mirrors the old
# np.random.seed calls at import sites).
_RNG = random.Random()


def seed(value: int) -> None:
    _RNG.seed(value)


def _std_normal() -> float:
    # Box-Muller transform (one value).
    u1 = _RNG.random()
    u2 = _RNG.random()
    return math.sqrt(-2.0 * math.log(u1 + 1e-12)) * math.cos(2.0 * math.pi * u2)


def randn(*shape: int) -> "list | list[list[float]]":
    """Standard-normal tensor of the given shape (1D or 2D)."""
    if len(shape) == 1:
        return [_std_normal() for _ in range(shape[0])]
    if len(shape) == 2:
        r, c = shape
        return [[_std_normal() for _ in range(c)] for _ in range(r)]
    raise ValueError(f"randn supports 1D/2D shapes, got {shape}")


def zeros(n: int) -> "list[float]":
    return [0.0] * n


def zeros_2d(r: int, c: int) -> "list[list[float]]":
    return [[0.0] * c for _ in range(r)]


def norm(v: "list[float]") -> float:
    return math.sqrt(sum(x * x for x in v))


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def transpose(a: "list[list[float]]") -> "list[list[float]]":
    return [[a[i][j] for i in range(len(a))] for j in range(len(a[0]))]


def _matmul_2d(a: "list[list[float]]", b: "list[list[float]]") -> "list[list[float]]":
    # (m, k) @ (k, n) -> (m, n)
    n = len(b[0])
    return [
        [sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(n)]
        for i in range(len(a))
    ]


def _matvec(a: "list[list[float]]", v: "list[float]") -> "list[float]":
    # (m, k) @ (k,) -> (m,)
    return [sum(a[i][k] * v[k] for k in range(len(v))) for i in range(len(a))]


def matmul(a, b):
    """Minimal ``@`` supporting 2D@2D, 2D@1D, and batched 3D@2D.

    3D inputs are treated as a batch of 2D matrices: (b, s, d) @ (d, r) -> (b, s, r).
    """
    if isinstance(a[0][0], list):
        # a is 3D -> batch of 2D rows
        if isinstance(b[0], list):
            return [_matmul_2d(seq, b) for seq in a]
        raise ValueError("3D @ 1D matmul not supported")
    # a is 2D
    if isinstance(b[0], list):
        return _matmul_2d(a, b)
    return _matvec(a, b)


def batched_add_scaled(x, y, scale: float):
    """Elementwise ``x + scale * y`` for matching 1D, 2D, or 3D tensors."""
    if isinstance(x[0], list):
        if isinstance(x[0][0], list):  # 3D
            return [
                [[x[bi][si][di] + scale * y[bi][si][di] for di in range(len(x[0][0]))]
                 for si in range(len(x[0]))]
                for bi in range(len(x))
            ]
        return [[x[i][j] + scale * y[i][j] for j in range(len(x[0]))] for i in range(len(x))]
    return [x[i] + scale * y[i] for i in range(len(x))]


def sub(a, b):
    """Elementwise ``a - b`` for matching tensors."""
    if isinstance(a[0], list):
        if isinstance(a[0][0], list):  # 3D
            return [
                [[a[bi][si][di] - b[bi][si][di] for di in range(len(a[0][0]))]
                 for si in range(len(a[0]))]
                for bi in range(len(a))
            ]
        return [[a[i][j] - b[i][j] for j in range(len(a[0]))] for i in range(len(a))]
    return [a[i] - b[i] for i in range(len(a))]


def scale_3d(x, s: float):
    return [
        [[v * s for v in seq] for seq in batch]
        for batch in x
    ]


def count_3d(x) -> int:
    return sum(len(seq) * len(seq[0]) for seq in x)


def mse_3d(a, b) -> float:
    n = count_3d(a)
    if n == 0:
        return 0.0
    total = sum(
        (a[bi][si][di] - b[bi][si][di]) ** 2
        for bi in range(len(a))
        for si in range(len(a[0]))
        for di in range(len(a[0][0]))
    )
    return total / n


def flatten_3d(x):
    """(b, s, d) -> (b*s, d) as a 2D list."""
    return [row for batch in x for row in batch]


def argsort_desc(v: "list[float]", top_k: int = None) -> "list[int]":
    idx = sorted(range(len(v)), key=lambda i: v[i], reverse=True)
    return idx[:top_k] if top_k else idx


def project_dim(content: "list[float]", target_dim: int) -> "list[float]":
    """Pad (with zeros) or truncate a 1D vector to ``target_dim``."""
    if len(content) == target_dim:
        return list(content)
    if len(content) < target_dim:
        out = list(content) + [0.0] * (target_dim - len(content))
        return out
    return list(content[:target_dim])
