"""Compatibility helpers for v4.x theorem code.

Re-exports all linear algebra and result functions under their
original underscore-prefixed names so that migrated theorem code
requires minimal modification.

APF v5.0
"""

import math as _math
from fractions import Fraction as _Fraction

# Re-export from clean modules under old names
from apf._result import result as _result
from apf._linalg import (
    zeros as _zeros,
    eye as _eye,
    diag as _diag,
    mat as _mat,
    mm as _mm,
    madd as _madd,
    msub as _msub,
    mscale as _mscale,
    dag as _dag,
    tr as _tr,
    det as _det,
    kron as _kron,
    outer as _outer,
    partial_trace_B as _partial_trace_B,
    vn_entropy as _vn_entropy,
    aclose as _aclose,
)


# ── -O-safe assertion replacement ─────────────────────────────────────

class CheckFailure(Exception):
    """Raised when a theorem check fails.

    Unlike AssertionError, this is NEVER stripped by python -O.
    bank.py catches this and reports it as a theorem failure.
    """
    pass


def check(condition, msg=""):
    """Assert replacement immune to python -O.

    Usage: check(x > 0, "x must be positive")
    Equivalent to: assert x > 0, "x must be positive"
    But works under python -O / PYTHONOPTIMIZE=1.
    """
    if not condition:
        raise CheckFailure(msg)


# Override _eigvalsh with monolith's original implementation.
# The original works with real-projected Hermitian matrices, which
# all v4.x tests were written against. The _linalg version handles
# complex entries differently and causes numerical mismatches.
def _eigvalsh(A):
    """Eigenvalues of Hermitian matrix (Jacobi iteration, real projection).

    Backward-compatible with v4.x monolith behavior.
    """
    n = len(A)
    M = [[A[i][j].real if isinstance(A[i][j], complex) else float(A[i][j])
          for j in range(n)] for i in range(n)]
    for _ in range(300):
        p, q, mx = 0, 1, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(M[i][j]) > mx:
                    mx = abs(M[i][j])
                    p, q = i, j
        if mx < 1e-14:
            break
        if abs(M[p][p] - M[q][q]) < 1e-15:
            theta = _math.pi / 4
        else:
            theta = 0.5 * _math.atan2(2 * M[p][q], M[p][p] - M[q][q])
        c, s = _math.cos(theta), _math.sin(theta)
        Mc = [row[:] for row in M]
        for i in range(n):
            Mc[i][p] = c * M[i][p] + s * M[i][q]
            Mc[i][q] = -s * M[i][p] + c * M[i][q]
        Mr = [row[:] for row in Mc]
        for j in range(n):
            Mr[p][j] = c * Mc[p][j] + s * Mc[q][j]
            Mr[q][j] = -s * Mc[p][j] + c * Mc[q][j]
        M = Mr
    return sorted(M[i][i] for i in range(n))


# ── Additional helpers used by v4.x code but not in _linalg ──────────

def _zvec(n):
    """Zero vector of length n."""
    return [complex(0)] * n


def _mv(A, v):
    """Matrix-vector multiply."""
    n = len(A)
    return [sum(A[i][k] * v[k] for k in range(len(v))) for i in range(n)]


def _fnorm(A):
    """Frobenius norm."""
    return _math.sqrt(sum(abs(A[i][j]) ** 2
                          for i in range(len(A))
                          for j in range(len(A[0]))))


def _vdot(u, v):
    """Complex inner product <u|v>."""
    return sum(a.conjugate() * b for a, b in zip(u, v))


def _vkron(u, v):
    """Kronecker product of vectors."""
    return [a * b for a in u for b in v]


def _vscale(c, v):
    """Scalar * vector."""
    c = complex(c)
    return [c * x for x in v]


def _vadd(u, v):
    """Vector addition."""
    return [a + b for a, b in zip(u, v)]


def _eigh_3x3(H):
    """Eigenvalues and eigenvectors for 3x3 Hermitian. Delegates to _eigh."""
    return _eigh(H)


def _eigh(H):
    """Eigenvalues and eigenvectors of Hermitian matrix via two-step Jacobi.

    Step 1: phase-remove off-diagonal element to make it real.
    Step 2: real Givens rotation to zero it.
    Returns (sorted_eigenvalues, eigenvector_column_matrix).
    """
    n = len(H)
    A = [[complex(H[i][j]) for j in range(n)] for i in range(n)]
    V = [[complex(1 if i == j else 0) for j in range(n)] for i in range(n)]

    for _ in range(800):
        mx, p, q = 0.0, 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i][j]) > mx:
                    mx = abs(A[i][j]); p, q = i, j
        if mx < 1e-13:
            break
        apq = A[p][q]
        r = abs(apq)
        if r < 1e-15:
            continue
        # Step 1: phase removal -- make A[p][q] real
        eia = apq / r
        eia_c = eia.conjugate()
        for i in range(n):
            A[i][q] *= eia_c
        for j in range(n):
            A[q][j] *= eia
        A[p][q] = complex(r)
        A[q][p] = complex(r)
        for i in range(n):
            V[i][q] *= eia_c
        # Step 2: real Jacobi rotation
        app, aqq = A[p][p].real, A[q][q].real
        if abs(app - aqq) < 1e-15:
            theta = _math.pi / 4
        else:
            theta = 0.5 * _math.atan2(2 * r, app - aqq)
        c, s = _math.cos(theta), _math.sin(theta)
        for i in range(n):
            aip, aiq = A[i][p], A[i][q]
            A[i][p] = c * aip + s * aiq
            A[i][q] = -s * aip + c * aiq
        for j in range(n):
            apj, aqj = A[p][j], A[q][j]
            A[p][j] = c * apj + s * aqj
            A[q][j] = -s * apj + c * aqj
        A[p][p] = complex(A[p][p].real)
        A[q][q] = complex(A[q][q].real)
        A[p][q] = complex(0)
        A[q][p] = complex(0)
        for i in range(n):
            vip, viq = V[i][p], V[i][q]
            V[i][p] = c * vip + s * viq
            V[i][q] = -s * vip + c * viq
    evals = [A[i][i].real for i in range(n)]
    idx = sorted(range(n), key=lambda i: evals[i])
    return ([evals[i] for i in idx],
            [[V[r][idx[c]] for c in range(n)] for r in range(n)])
