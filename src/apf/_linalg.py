"""Zero-dependency linear algebra helpers.

All functions operate on plain Python lists-of-lists (complex entries).
No numpy, no scipy. This keeps the entire APF bank stdlib-only.

APF v5.0
"""

import math as _math

__all__ = [
    'zeros', 'eye', 'diag', 'mat',
    'mm', 'madd', 'msub', 'mscale', 'dag',
    'tr', 'det', 'eigvalsh', 'kron',
    'outer', 'partial_trace_B', 'vn_entropy',
    'aclose',
]


# ── Construction ──────────────────────────────────────────────────────

def zeros(n, m=None):
    """n×m zero matrix (default: n×n)."""
    if m is None:
        m = n
    return [[complex(0)] * m for _ in range(n)]


def eye(n):
    """n×n identity matrix."""
    M = zeros(n)
    for i in range(n):
        M[i][i] = complex(1)
    return M


def diag(vals):
    """Diagonal matrix from a list of values."""
    n = len(vals)
    M = zeros(n)
    for i in range(n):
        M[i][i] = complex(vals[i])
    return M


def mat(rows):
    """Convert nested list to complex matrix."""
    return [[complex(v) for v in row] for row in rows]


# ── Arithmetic ────────────────────────────────────────────────────────

def mm(A, B):
    """Matrix multiply A @ B."""
    n, m, p = len(A), len(B), len(B[0])
    C = zeros(n, p)
    for i in range(n):
        for j in range(p):
            C[i][j] = sum(A[i][k] * B[k][j] for k in range(m))
    return C


def madd(A, B):
    """Element-wise A + B."""
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))]
            for i in range(len(A))]


def msub(A, B):
    """Element-wise A - B."""
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))]
            for i in range(len(A))]


def mscale(c, M):
    """Scalar * matrix."""
    c = complex(c)
    return [[c * M[i][j] for j in range(len(M[0]))]
            for i in range(len(M))]


def dag(M):
    """Conjugate transpose (dagger)."""
    n, m = len(M), len(M[0])
    return [[M[j][i].conjugate() for j in range(n)] for i in range(m)]


# ── Scalar quantities ─────────────────────────────────────────────────

def tr(M):
    """Trace."""
    return sum(M[i][i] for i in range(len(M)))


def det(M):
    """Determinant (2×2 and 3×3 fast paths, general via LU)."""
    n = len(M)
    if n == 1:
        return M[0][0]
    if n == 2:
        return M[0][0] * M[1][1] - M[0][1] * M[1][0]
    if n == 3:
        return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
                - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
    # General: cofactor expansion along first row
    sign = 1
    result = complex(0)
    for j in range(n):
        sub = [[M[r][c] for c in range(n) if c != j]
               for r in range(1, n)]
        result += sign * M[0][j] * det(sub)
        sign *= -1
    return result


# ── Eigenvalues (Hermitian) ───────────────────────────────────────────

def eigvalsh(M):
    """Real eigenvalues of a Hermitian matrix (Jacobi iteration).

    Returns sorted list of real eigenvalues.
    For small matrices (n <= 3) uses closed-form.
    """
    n = len(M)
    if n == 1:
        return [M[0][0].real]
    if n == 2:
        a, b = M[0][0].real, M[1][1].real
        c = abs(M[0][1])
        disc = _math.sqrt(max(0, ((a - b) / 2) ** 2 + c ** 2))
        mid = (a + b) / 2
        return sorted([mid - disc, mid + disc])

    # Jacobi eigenvalue algorithm for general Hermitian
    A = [[complex(M[i][j]) for j in range(n)] for i in range(n)]
    for _ in range(300):
        off = sum(abs(A[i][j]) ** 2
                  for i in range(n) for j in range(n) if i != j)
        if off < 1e-24:
            break
        for p in range(n):
            for q in range(p + 1, n):
                if abs(A[p][q]) < 1e-15:
                    continue
                d_pq = A[p][p].real - A[q][q].real
                if abs(d_pq) < 1e-15:
                    theta = _math.pi / 4
                else:
                    theta = 0.5 * _math.atan2(2 * abs(A[p][q]), d_pq)
                c, s = _math.cos(theta), _math.sin(theta)
                phase = (A[p][q] / abs(A[p][q])
                         if abs(A[p][q]) > 1e-15 else 1)
                s_ph = s * phase.conjugate()
                # Apply rotation
                for j in range(n):
                    apj, aqj = A[p][j], A[q][j]
                    A[p][j] = c * apj + s_ph * aqj
                    A[q][j] = -s_ph.conjugate() * apj + c * aqj
                for i in range(n):
                    aip, aiq = A[i][p], A[i][q]
                    A[i][p] = c * aip + s_ph.conjugate() * aiq
                    A[i][q] = -s_ph * aip + c * aiq
                A[p][q] = complex(0)
                A[q][p] = complex(0)
    return sorted(A[i][i].real for i in range(n))


# ── Tensor products ───────────────────────────────────────────────────

def kron(A, B):
    """Kronecker product A ⊗ B."""
    na, ma = len(A), len(A[0])
    nb, mb = len(B), len(B[0])
    C = zeros(na * nb, ma * mb)
    for i in range(na):
        for j in range(ma):
            for k in range(nb):
                for l in range(mb):
                    C[i * nb + k][j * mb + l] = A[i][j] * B[k][l]
    return C


def outer(psi, phi):
    """Outer product |psi><phi|."""
    n = len(psi)
    return [[psi[i] * phi[j].conjugate() for j in range(n)]
            for i in range(n)]


def partial_trace_B(rho_AB, dA, dB):
    """Partial trace over subsystem B: Tr_B(rho_AB)."""
    rho_A = zeros(dA)
    for i in range(dA):
        for j in range(dA):
            for k in range(dB):
                rho_A[i][j] += rho_AB[i * dB + k][j * dB + k]
    return rho_A


# ── Entropy ───────────────────────────────────────────────────────────

def vn_entropy(rho):
    """Von Neumann entropy S = -Tr(rho ln rho)."""
    eigs = eigvalsh(rho)
    return -sum(ev * _math.log(ev) for ev in eigs if ev > 1e-15)


# ── Comparison ────────────────────────────────────────────────────────

def aclose(A, B, tol=1e-10):
    """Check if two matrices are element-wise close."""
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        return False
    return all(abs(A[i][j] - B[i][j]) < tol
               for i in range(len(A)) for j in range(len(A[0])))
