from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# --- NUMBA-accelerated Steihaug–Toint CG (dense/CSR) ---------------------
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:
    from numba import njit

    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

# Status codes to map back to TRStatus
_ST_SUCCESS = 0
_ST_BOUNDARY = 1
_ST_NEG_CURV = 2
_ST_MAX_ITER = 3

MatLike = Union[np.ndarray, sp.spmatrix, spla.LinearOperator]
Vec = np.ndarray


# ---------------------------- Configuration ---------------------------- #
@dataclass
class TRConfig:
    # Trust region parameters
    delta0: float = 1.0
    delta_min: float = 1e-12
    delta_max: float = 1e6
    eta1: float = 0.1  # accept threshold
    eta2: float = 0.9  # expand threshold
    gamma1: float = 0.5  # shrink factor
    gamma2: float = 2.0  # expand factor
    # Byrd-Omojokun split
    zeta: float = 0.8  # fraction of radius for normal step
    # Solver tolerances
    cg_tol: float = 1e-8
    cg_tol_rel: float = 0.1  # relative tolerance factor
    cg_maxiter: int = 200
    neg_curv_tol: float = 1e-14
    constraint_tol: float = 1e-8
    max_active_set_iter: int = 10  # max iterations for active-set loop
    # Adaptive features
    adaptive_zeta: bool = True
    curvature_aware: bool = True
    feasibility_emphasis: bool = True
    # Numerical stability
    rcond: float = 1e-12
    reg_floor: float = 1e-10
    reg_max: float = 1e6
    # Caching / preconditioning
    cache_nullspace: bool = True
    use_prec: bool = True
    prec_kind: str = "auto_jacobi"  # options: "auto_jacobi", "none"
    # Criticality step & safeguard
    criticality_enabled: bool = True
    kappa_g: float = 1e-2  # projected-grad threshold factor
    theta_crit: float = 0.5  # radius shrink factor on criticality
    max_crit_shrinks: int = 1  # at most this many back-to-back shrinks


class TRStatus(Enum):
    SUCCESS = "success"
    BOUNDARY = "boundary"
    NEG_CURV = "negative_curvature"
    MAX_ITER = "max_iterations"
    INFEASIBLE = "infeasible"


# ---------------------------- Utilities ---------------------------- #
def safe_norm(x: Vec) -> float:
    """Numerically safe norm computation."""
    return float(np.linalg.norm(x)) if x.size > 0 else 0.0


def make_operator(A: MatLike, n: int) -> spla.LinearOperator:
    """Convert matrix-like object to LinearOperator."""
    if isinstance(A, spla.LinearOperator):
        return A
    if callable(A):
        return spla.LinearOperator((n, n), matvec=A, dtype=float)
    return spla.aslinearoperator(A)


def solve_kkt_system(
    H: MatLike,
    A: Optional[np.ndarray],
    g: Vec,
    b: Optional[Vec] = None,
    reg: float = 1e-10,
    use_iterative: bool = False,
) -> Tuple[Vec, Vec]:
    """
    Solve KKT system using stable block elimination or iterative solver:
    [H A^T] [p] = [-g]
    [A 0 ] [λ] = [-b]
    """
    n = g.size
    if A is None or A.size == 0:
        try:
            if use_iterative:
                p = spla.cgs(make_operator(H + reg * np.eye(n), n), -g)[0]
            else:
                p = la.solve(H + reg * np.eye(n), -g, assume_a="pos")
            return p, np.array([])
        except la.LinAlgError:
            p = la.lstsq(H + reg * np.eye(n), -g)[0]
            return p, np.array([])
    m = A.shape[0]
    b = np.zeros(m) if b is None else b
    if use_iterative:
        K = np.block([[H + reg * np.eye(n), A.T], [A, -reg * np.eye(m)]])
        rhs = np.concatenate([-g, -b])
        sol = spla.lsqr(make_operator(K, n + m), rhs)[0]
        return sol[:n], sol[n:]
    try:
        Hinv_g = la.solve(H + reg * np.eye(n), -g, assume_a="pos")
        Hinv_AT = la.solve(H + reg * np.eye(n), A.T, assume_a="pos")
        S = A @ Hinv_AT  # Schur complement
        rhs = A @ Hinv_g - b
        lam = la.solve(S + reg * np.eye(m), rhs)
        p = Hinv_g - Hinv_AT @ lam
        return p, lam
    except la.LinAlgError:
        K = np.block([[H + reg * np.eye(n), A.T], [A, -reg * np.eye(m)]])
        rhs = np.concatenate([-g, -b])
        sol = la.lstsq(K, rhs)[0]
        return sol[:n], sol[n:]


def nullspace_basis(A: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """Compute orthonormal nullspace basis via SVD/QR."""
    if A.size == 0:
        return np.eye(A.shape[1] if A.ndim > 1 else 0)
    try:
        U, s, VT = la.svd(A, full_matrices=True)
        tol = rcond * (s[0] if s.size > 0 else 1.0)
        rank = np.sum(s > tol)
        n = A.shape[1]
        if rank >= n:
            return np.zeros((n, 0))
        return VT.T[:, rank:]
    except la.LinAlgError:
        Q, R, P = la.qr(A.T, mode="economic", pivoting=True)
        rank = np.sum(np.abs(np.diag(R)) > rcond * np.abs(R[0, 0].max() if R.size else 1.0))
        # For QR with pivoting, reconstruct nullspace using permutation
        null_dim = A.shape[1] - rank
        if null_dim <= 0:
            return np.zeros((A.shape[1], 0))
        # Basic fallback: use SVD instead on error
        return nullspace_basis(A, rcond * 10)  # Increase rcond to avoid recursion depth

def dogleg_step(
    g: Vec, Hg: Vec, gnorm: float, Delta: float, reg: float = 1e-10
) -> Tuple[Vec, TRStatus]:
    """
    Improved dogleg with regularized Newton step.
    Solves: min g^T p + 0.5 p^T H p s.t. ||p|| <= Delta
    """
    n = g.size
    if gnorm <= 1e-14:
        return np.zeros(n), TRStatus.SUCCESS
    # Cauchy point
    gHg = np.dot(g, Hg)
    if gHg <= 1e-14:  # Non-positive curvature
        p_cauchy = -(Delta / gnorm) * g
        return p_cauchy, TRStatus.NEG_CURV
    alpha_cauchy = gnorm**2 / gHg
    p_cauchy = -alpha_cauchy * g
    if safe_norm(p_cauchy) >= Delta:
        return -(Delta / gnorm) * g, TRStatus.BOUNDARY
    # Regularized Newton step
    try:
        H_reg = 0.5 * (
            Hg.reshape(-1, 1) @ g.reshape(1, -1) + g.reshape(-1, 1) @ Hg.reshape(1, -1)
        ) + reg * np.eye(n)
        p_newton = la.solve(H_reg, -g, assume_a="pos")
        if safe_norm(p_newton) <= Delta:
            return p_newton, TRStatus.SUCCESS
    except la.LinAlgError:
        p_newton = la.lstsq(H_reg, -g)[0]
    # Dogleg between Cauchy and boundary
    s = p_newton - p_cauchy
    a = np.dot(s, s)
    b = 2 * np.dot(p_cauchy, s)
    c = np.dot(p_cauchy, p_cauchy) - Delta**2
    disc = max(0.0, b**2 - 4 * a * c)
    if a > 1e-14:
        tau = (-b + np.sqrt(disc)) / (2 * a)
        tau = np.clip(tau, 0.0, 1.0)
    else:
        tau = 1.0
    return p_cauchy + tau * s, TRStatus.BOUNDARY


# ======== Preconditioner helpers ========
class Preconditioner:
    """
    Simple protocol: apply(z) returns M^{-1} r.
    You can pass:
      - None                         -> identity
      - 1D numpy array (diag)        -> Jacobi
      - sp.spmatrix (only diag used) -> Jacobi
      - scipy.sparse.linalg.LinearOperator
      - callable(r)->z
    """

    def __init__(
        self,
        M: Union[
            None,
            np.ndarray,
            sp.spmatrix,
            spla.LinearOperator,
            Callable[[np.ndarray], np.ndarray],
        ],
    ):
        self.kind = "identity"
        self.diag = None
        self.Mop = None
        self.func = None
        if M is None:
            self.kind = "identity"
        elif isinstance(M, np.ndarray) and M.ndim == 1:
            self.kind = "diag"
            d = M.astype(np.float64, copy=False)
            self.diag = np.where(np.abs(d) > 0, 1.0 / d, 0.0)
        elif sp.isspmatrix(M):
            self.kind = "diag"
            d = M.diagonal().astype(np.float64, copy=False)
            self.diag = np.where(np.abs(d) > 0, 1.0 / d, 0.0)
        elif isinstance(M, spla.LinearOperator):
            self.kind = "linop"
            self.Mop = M
        elif callable(M):
            self.kind = "callable"
            self.func = M
        else:
            raise TypeError("Unsupported preconditioner type")

    def apply(self, r: np.ndarray) -> np.ndarray:
        if self.kind == "identity":
            return r
        if self.kind == "diag":
            return self.diag * r
        if self.kind == "linop":
            z = self.Mop @ r
            return np.asarray(z, dtype=np.float64)
        z = self.func(r)
        return np.asarray(z, dtype=np.float64)


def make_jacobi_from_H(H: MatLike) -> Optional[np.ndarray]:
    """Convenience: extract a diagonal preconditioner if cheap/available."""
    try:
        if isinstance(H, np.ndarray):
            d = H.diagonal().astype(np.float64, copy=False)
            d = np.where(np.abs(d) > 0, d, 1.0)  # guard zeros
            return d
        if sp.issparse(H):
            d = H.diagonal().astype(np.float64, copy=False)
            d = np.where(np.abs(d) > 0, d, 1.0)
            return d
    except Exception:
        pass
    return None


# ======================== Numba kernels (optional) ========================
if _HAS_NUMBA:

    @njit(cache=True, fastmath=True)
    def _pointwise_div_safe(
        out: np.ndarray, r: np.ndarray, invdiag: np.ndarray
    ) -> None:
        # out = invdiag * r  (invdiag should already be 1/diag(H) or similar)
        n = r.size
        for i in range(n):
            out[i] = invdiag[i] * r[i]

    @njit(cache=True, fastmath=True)
    def _dot(a: np.ndarray, b: np.ndarray) -> float:
        s = 0.0
        for i in range(a.size):
            s += a[i] * b[i]
        return s

    @njit(cache=True, fastmath=True)
    def _axpy(alpha: float, x: np.ndarray, y: np.ndarray) -> None:
        # y += alpha * x
        for i in range(x.size):
            y[i] += alpha * x[i]

    @njit(cache=True, fastmath=True)
    def _boundary_intersection_numba(
        p: np.ndarray, d: np.ndarray, Delta: float
    ) -> float:
        # Find tau s.t. ||p + tau d|| = Delta
        pTp = _dot(p, p)
        pTd = _dot(p, d)
        dTd = _dot(d, d)
        if dTd <= 1e-14:
            return 0.0
        disc = pTd * pTd - dTd * (pTp - Delta * Delta)
        if disc < 0.0:
            disc = 0.0
        return (-pTd + np.sqrt(disc)) / dTd

    # -------- dense matvec ----------
    @njit(cache=True, fastmath=True)
    def _dense_matvec(H: np.ndarray, x: np.ndarray, out: np.ndarray) -> None:
        n = H.shape[0]
        for i in range(n):
            s = 0.0
            row = H[i]
            for j in range(n):
                s += row[j] * x[j]
            out[i] = s

    # -------- CSR matvec (data, indices, indptr) ----------
    @njit(cache=True, fastmath=True)
    def _csr_matvec(
        data: np.ndarray,
        indices: np.ndarray,
        indptr: np.ndarray,
        x: np.ndarray,
        out: np.ndarray,
    ) -> None:
        n = indptr.size - 1
        for i in range(n):
            s = 0.0
            start = int(indptr[i])
            end = int(indptr[i + 1])
            for kk in range(start, end):
                j = int(indices[kk])
                s += data[kk] * x[j]
            out[i] = s

    @njit(cache=True, fastmath=True)
    def _steihaug_core_pcg_diag(
        matvec_kind: int,
        H_dense: np.ndarray,
        csr_data: np.ndarray,
        csr_indices: np.ndarray,
        csr_indptr: np.ndarray,
        g: np.ndarray,
        Delta: float,
        tol: float,
        maxiter: int,
        neg_curv_tol: float,
        invdiag: np.ndarray,  # 1/diag(M)  (Jacobi M ≈ diag(H))
    ) -> Tuple[np.ndarray, int, int]:
        """
        Left-preconditioned Steihaug–Toint with diagonal preconditioner.
        Returns: (p, status_code, iters)
        """
        n = g.size
        p = np.zeros(n, dtype=np.float64)
        r = -g.copy()
        z = np.empty(n, dtype=np.float64)
        _pointwise_div_safe(z, r, invdiag)
        d = z.copy()
        Hd = np.zeros(n, dtype=np.float64)

        rz = _dot(r, z)
        if np.sqrt(_dot(r, r)) <= tol:
            return p, _ST_SUCCESS, 0

        for k in range(maxiter):
            # Hd = H @ d
            if matvec_kind == 0:
                _dense_matvec(H_dense, d, Hd)
            else:
                _csr_matvec(csr_data, csr_indices, csr_indptr, d, Hd)

            dTHd = _dot(d, Hd)

            # Negative curvature?
            if dTHd <= neg_curv_tol * max(1.0, _dot(d, d)):
                tau = _boundary_intersection_numba(p, d, Delta)
                _axpy(tau, d, p)
                return p, _ST_NEG_CURV, k

            alpha = rz / max(dTHd, 1e-32)

            # boundary check using ||p + α d||
            pTp = _dot(p, p)
            pTd = _dot(p, d)
            dTd = _dot(d, d)
            pn2 = pTp + 2.0 * alpha * pTd + alpha * alpha * dTd
            if pn2 >= Delta * Delta:
                tau = _boundary_intersection_numba(p, d, Delta)
                _axpy(tau, d, p)
                return p, _ST_BOUNDARY, k

            # p += α d ; r -= α Hd
            _axpy(alpha, d, p)
            for i in range(n):
                r[i] -= alpha * Hd[i]

            if np.sqrt(_dot(r, r)) <= tol:
                return p, _ST_SUCCESS, k + 1

            # z = M^{-1} r (diag)
            _pointwise_div_safe(z, r, invdiag)
            rz_next = _dot(r, z)
            beta = rz_next / max(rz, 1e-32)

            # d = z + β d
            for i in range(n):
                d[i] = z[i] + beta * d[i]
            rz = rz_next

        return p, _ST_MAX_ITER, maxiter


# ======================== Python PCG fallback ============================
def steihaug_cg_pcg_python(
    H_op: spla.LinearOperator,
    g: np.ndarray,
    Delta: float,
    tol: float,
    maxiter: int,
    neg_curv_tol: float,
    M: Optional[Preconditioner] = None,
) -> Tuple[np.ndarray, TRStatus, int]:
    """Left-preconditioned Steihaug–Toint in pure Python with general M."""
    n = g.size
    p = np.zeros(n, dtype=np.float64)
    r = -g.astype(np.float64, copy=False)
    applyM = M.apply if M is not None else (lambda x: x)
    z = applyM(r)
    d = z.copy()
    r_norm = np.linalg.norm(r)
    if r_norm <= tol:
        return p, TRStatus.SUCCESS, 0
    rz = float(r @ z)

    for k in range(maxiter):
        Hd = H_op @ d
        dTHd = float(d @ Hd)

        if dTHd <= neg_curv_tol * max(1.0, float(d @ d)):
            tau = _boundary_intersection(p, d, Delta)
            return p + tau * d, TRStatus.NEG_CURV, k

        alpha = rz / max(dTHd, 1e-32)

        # boundary check
        pTp = float(p @ p)
        pTd = float(p @ d)
        dTd = float(d @ d)
        pn2 = pTp + 2.0 * alpha * pTd + alpha * alpha * dTd
        if pn2 >= Delta * Delta:
            tau = _boundary_intersection(p, d, Delta)
            return p + tau * d, TRStatus.BOUNDARY, k

        p = p + alpha * d
        r = r - alpha * Hd
        r_norm = np.linalg.norm(r)
        if r_norm <= tol:
            return p, TRStatus.SUCCESS, k + 1

        z = applyM(r)
        rz_next = float(r @ z)
        beta = rz_next / max(rz, 1e-32)
        d = z + beta * d
        rz = rz_next

    return p, TRStatus.MAX_ITER, maxiter


# ======================== Public solver wrapper ==========================
def steihaug_cg_fast(
    H,
    g,
    Delta,
    tol=1e-8,
    maxiter=200,
    neg_curv_tol=1e-14,
    prec: Union[
        None,
        np.ndarray,
        sp.spmatrix,
        spla.LinearOperator,
        Callable[[np.ndarray], np.ndarray],
    ] = None,
):
    """
    Steihaug–Toint trust-region CG with optional left preconditioning (PCG).
    prec can be:
      - None (identity)
      - 1D array diag(H) or custom diag (Jacobi)
      - sparse matrix (uses its diagonal)
      - LinearOperator (apply = M^{-1})
      - callable(r) -> M^{-1} r
    """
    # Fast path: Numba + diagonal preconditioner (or identity via auto-Jacobi)
    if _HAS_NUMBA and (isinstance(H, np.ndarray) or sp.isspmatrix_csr(H)):
        invdiag = None
        diag_ok = False
        if prec is None:
            # try auto Jacobi from H
            d = make_jacobi_from_H(H)
            if d is not None:
                invdiag = (1.0 / d).astype(np.float64, copy=False)
                diag_ok = True
        elif isinstance(prec, np.ndarray) and prec.ndim == 1:
            invdiag = np.where(np.abs(prec) > 0, 1.0 / prec, 0.0).astype(
                np.float64, copy=False
            )
            diag_ok = True
        elif sp.isspmatrix(prec):
            d = prec.diagonal().astype(np.float64, copy=False)
            invdiag = np.where(np.abs(d) > 0, 1.0 / d, 0.0)
            diag_ok = True

        if diag_ok:
            if isinstance(H, np.ndarray):
                p, sc, it = _steihaug_core_pcg_diag(
                    0,
                    np.ascontiguousarray(H, dtype=np.float64),
                    np.empty(0),
                    np.empty(0),
                    np.empty(0),
                    g.astype(np.float64, copy=False),
                    float(Delta),
                    float(tol),
                    int(maxiter),
                    float(neg_curv_tol),
                    invdiag.astype(np.float64, copy=False),
                )
            else:
                H = H.tocsr()
                p, sc, it = _steihaug_core_pcg_diag(
                    1,
                    np.empty((0, 0)),
                    np.ascontiguousarray(H.data, dtype=np.float64),
                    np.ascontiguousarray(H.indices, dtype=np.int64),
                    np.ascontiguousarray(H.indptr, dtype=np.int64),
                    g.astype(np.float64, copy=False),
                    float(Delta),
                    float(tol),
                    int(maxiter),
                    float(neg_curv_tol),
                    invdiag.astype(np.float64, copy=False),
                )
            status = (
                TRStatus.SUCCESS
                if sc == _ST_SUCCESS
                else (
                    TRStatus.BOUNDARY
                    if sc == _ST_BOUNDARY
                    else TRStatus.NEG_CURV if sc == _ST_NEG_CURV else TRStatus.MAX_ITER
                )
            )
            return p, status, it

    # Fallback: Python PCG with general preconditioner
    H_op = make_operator(H, g.size)
    M = Preconditioner(prec) if prec is not None else Preconditioner(None)
    return steihaug_cg_pcg_python(H_op, g, Delta, tol, maxiter, neg_curv_tol, M)


def _boundary_intersection(p: Vec, d: Vec, Delta: float) -> float:
    """Find tau such that ||p + tau*d|| = Delta."""
    pTp = np.dot(p, p)
    pTd = np.dot(p, d)
    dTd = np.dot(d, d)
    if dTd <= 1e-14:
        return 0.0
    disc = max(0.0, pTd**2 - dTd * (pTp - Delta**2))
    return (-pTd + np.sqrt(disc)) / dTd


# ---------------------------- Main TR Manager ---------------------------- #
class TrustRegionManager:
    """
    Modern Trust Region manager with Byrd-Omojokun decomposition for SQP.
    Key features:
    - Normal/tangential step decomposition for equality constraints
    - Active-set strategy for inequality constraints
    - Adaptive zeta and curvature-aware radius updates
    - Robust numerical handling with iterative solver support
    - Multiplier recovery for SQP iterations
    - Criticality step & safeguard
    """

    def __init__(self, config: Optional[TRConfig] = None):
        self.cfg = TRConfig()
        self.delta = self.cfg.delta0
        self.rejection_count = 0
        # Adaptive state
        self._rho_history = []
        self._feasibility_history = []
        self._curvature_estimate = None
        # Nullspace cache
        self._nullspace_cache: Dict[int, np.ndarray] = {}
        self._last_A_eq_hash: Optional[int] = None
        self._last_was_criticality = False

    def solve(
        self,
        H: MatLike,
        g: Vec,
        A_ineq: Optional[np.ndarray] = None,
        b_ineq: Optional[Vec] = None,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[Vec] = None,
    ) -> Tuple[Vec, Dict[str, Any], Vec, Vec]:
        """
        Solve trust region subproblem with mixed constraints.
        min g^T p + 0.5 p^T H p
        s.t. A_eq p + b_eq = 0
             A_ineq p + b_ineq <= 0
             ||p|| <= delta
        Returns: step, info dict, inequality multipliers, equality multipliers
        """
        # Input validation
        n = g.size
        if g.ndim != 1:
            raise ValueError("g must be a 1D array")
        if H.shape != (n, n):
            raise ValueError(f"H must have shape ({n}, {n})")
        if A_eq is not None:
            A_eq = np.asarray(A_eq)
            if A_eq.shape[1] != n:
                raise ValueError(f"A_eq must have {n} columns")
            if b_eq is None or b_eq.size != A_eq.shape[0]:
                raise ValueError("b_eq must match A_eq rows")
            b_eq = np.asarray(b_eq).ravel()
        if A_ineq is not None:
            A_ineq = np.asarray(A_ineq)
            if A_ineq.shape[1] != n:
                raise ValueError(f"A_ineq must have {n} columns")
            if b_ineq is None or b_ineq.size != A_ineq.shape[0]:
                raise ValueError("b_ineq must match A_ineq rows")
            b_ineq = np.asarray(b_ineq).ravel()

        if A_eq is not None and A_eq.size > 0:
            p, info, lam, nu = self._solve_with_equality_constraints(
                H, g, A_eq, b_eq, A_ineq, b_ineq
            )
        elif A_ineq is not None and A_ineq.size > 0:
            p, info, lam, nu = self._solve_with_inequality_constraints(
                H, g, A_ineq, b_ineq
            )
        else:
            p, info = self._solve_unconstrained(H, g)
            lam, nu = np.array([]), np.array([])
        return p, info, lam, nu

    # ---------- Criticality utilities ----------
    def _projected_grad_norm(self, g: Vec, A_eq: Optional[np.ndarray]) -> float:
        """
        ||P_T g|| where T is the tangent space of {A_eq p + b_eq = 0}.
        Uses the projector P = I - A^T (A A^T)^† A, implemented via a
        robust normal-equations solve with light Tikhonov.
        """
        if A_eq is None or A_eq.size == 0:
            return safe_norm(g)
        A = np.asarray(A_eq, dtype=float)
        reg = max(self.cfg.reg_floor, 1e-12)
        Ag = A @ g
        try:
            yy = la.solve(A @ A.T + reg * np.eye(A.shape[0]), Ag, assume_a="pos")
        except la.LinAlgError:
            yy = la.lstsq(A @ A.T + reg * np.eye(A.shape[0]), Ag, cond=self.cfg.rcond)[
                0
            ]
        g_tan = g - A.T @ yy
        return safe_norm(g_tan)

    # ---------- Unconstrained subproblem ----------
    def _solve_unconstrained(self, H: MatLike, g: Vec) -> Tuple[Vec, Dict[str, Any]]:
        n = g.size
        H_op = make_operator(H, n)
        g_norm = safe_norm(g)
        tol = min(self.cfg.cg_tol, self.cfg.cg_tol_rel * g_norm)

        # Criticality safeguard (unconstrained: ||P_T g|| == ||g||)
        crit_used = False
        crit_shrinks = 0
        if self.cfg.criticality_enabled:
            for _ in range(self.cfg.max_crit_shrinks):
                if safe_norm(g) <= self.cfg.kappa_g * self.delta:
                    self.delta = max(self.cfg.delta_min, self.cfg.theta_crit * self.delta)
                    crit_used = True
                    crit_shrinks += 1
                else:
                    break

        # Preconditioner (optional)
        prec = None
        if self.cfg.use_prec and self.cfg.prec_kind == "auto_jacobi":
            prec = make_jacobi_from_H(H)

        p, status, iters = steihaug_cg_fast(
            H, g, self.delta, tol, self.cfg.cg_maxiter, self.cfg.neg_curv_tol, prec=prec
        )
        info = {
            "status": status.value,
            "iterations": iters,
            "step_norm": safe_norm(p),
            "model_reduction": self._model_reduction(H_op, g, p),
            "criticality": crit_used,
            "criticality_shrinks": crit_shrinks,
            "preconditioned": prec is not None,
        }
        self._last_was_criticality = bool(crit_used)
        return p, info

    # ---------- Equality-constrained subproblem ----------
    def _solve_with_equality_constraints(
        self,
        H: MatLike,
        g: Vec,
        A_eq: np.ndarray,
        b_eq: Vec,
        A_ineq: Optional[np.ndarray] = None,
        b_ineq: Optional[Vec] = None,
    ) -> Tuple[Vec, Dict[str, Any], Vec, Vec]:
        """
        Equality-constrained trust-region subproblem via Byrd–Omojokun split.

        Solves
            min_p   g^T p + 0.5 p^T H p
            s.t.    A_eq p + b_eq = 0
                    A_ineq p + b_ineq <= 0   (handled by active-set augmentation)
                    ||p|| <= delta

        Returns
        -------
        p : np.ndarray
            Step (normal + tangential).
        info : dict
            Diagnostics for this subproblem solve.
        lam : np.ndarray
            Inequality multipliers (aligned with A_ineq rows; zero on inactive).
        nu : np.ndarray
            Equality multipliers.
        """
        n = g.size
        H_op = make_operator(H, n)

        # --- Criticality safeguard in the tangent space ---
        crit_used = False
        crit_shrinks = 0
        if self.cfg.criticality_enabled:
            for _ in range(self.cfg.max_crit_shrinks):
                pg = self._projected_grad_norm(g, A_eq)
                if pg <= self.cfg.kappa_g * self.delta:
                    self.delta = max(self.cfg.delta_min, self.cfg.theta_crit * self.delta)
                    crit_used = True
                    crit_shrinks += 1
                else:
                    break

        # --- Adaptive zeta based on current equality residual ---
        viol_norm = safe_norm(b_eq)
        zeta = (
            self._adaptive_zeta(viol_norm, len(self._feasibility_history))
            if self.cfg.adaptive_zeta
            else self.cfg.zeta
        )

        # =======================
        # Step 1: normal (feasibility) step
        # =======================
        delta_n = zeta * self.delta
        try:
            # Use true residual here: b_eq = c_E(x) at the OUTER level.
            p_n = la.lstsq(A_eq, -b_eq, cond=self.cfg.rcond)[0]
            if safe_norm(p_n) > delta_n:
                p_n = (delta_n / safe_norm(p_n)) * p_n
        except la.LinAlgError:
            p_n = np.zeros(n)

        # =======================
        # Step 2: tangential step in nullspace of A_eq
        # =======================
        A_eq_hash = hash(A_eq.tobytes()) if self.cfg.cache_nullspace else None
        if self.cfg.cache_nullspace and self._last_A_eq_hash == A_eq_hash:
            Z = self._nullspace_cache[A_eq_hash]
        else:
            Z = nullspace_basis(A_eq, self.cfg.rcond)
            if self.cfg.cache_nullspace:
                self._nullspace_cache[A_eq_hash] = Z
                self._last_A_eq_hash = A_eq_hash

        if Z.shape[1] == 0:
            # No tangent space available (constraints full rank covering R^n)
            p = p_n
            sigma = self._estimate_sigma(H_op, g, p)
            lam, nu = self._recover_multipliers(H_op, g, p, sigma, A_eq, A_ineq, None)
            info = {
                "status": "constrained_minimum",
                "normal_step_norm": safe_norm(p_n),
                "tangential_step_norm": 0.0,
                "step_norm": safe_norm(p),
                "constraint_violation": safe_norm(A_eq @ p + b_eq),
                "nullspace_dim": 0,
                "model_reduction": self._model_reduction(H_op, g, p),
                "zeta_used": zeta,
                "criticality": crit_used,
                "criticality_shrinks": crit_shrinks,
                "preconditioned_reduced": False,
                "active_constraints": [],
            }
            return p, info, lam, nu

        # Reduced TR on tangent space
        HZ = H_op @ Z
        H_reduced = Z.T @ HZ
        g_reduced = Z.T @ (H_op @ p_n + g)
        remaining_radius = np.sqrt(max(0.0, self.delta**2 - safe_norm(p_n) ** 2))

        tol_red = min(self.cfg.cg_tol, self.cfg.cg_tol_rel * safe_norm(g_reduced))
        prec_red = (
            make_jacobi_from_H(H_reduced)
            if self.cfg.use_prec and self.cfg.prec_kind == "auto_jacobi"
            else None
        )

        p_t_red, status, iters = steihaug_cg_fast(
            H_reduced,
            g_reduced,
            remaining_radius,
            tol_red,
            self.cfg.cg_maxiter,
            self.cfg.neg_curv_tol,
            prec=prec_red,
        )
        p_t = Z @ p_t_red

        # -------- Tangent Cauchy fallback (guarantees progress if projected grad ≠ 0) --------
        if safe_norm(p_t) <= 1e-14:
            g_tan = self._project_to_tangent(g + H_op @ p_n, A_eq)
            if safe_norm(g_tan) > 1e-10:
                p_t = self._constrained_cauchy_in_tangent(
                    H_op, g + H_op @ p_n, A_eq, remaining_radius
                )

        p = p_n + p_t

        # Final trust-region clipping
        sn = safe_norm(p)
        if sn > self.delta and sn > 0:
            p *= self.delta / sn

        # --- Last resort: direct KKT step if still zero and no inequalities to honor ---
        if safe_norm(p) <= 1e-14 and (A_ineq is None or A_ineq.size == 0):
            p_kkt, _ = self._kkt_equality_step(H, g, A_eq, b_eq)
            sk = safe_norm(p_kkt)
            if sk > 0:
                p = p_kkt if sk <= self.delta else (self.delta / sk) * p_kkt

        # =======================
        # Inequality handling (active-set augmentation), if provided
        # =======================
        active_idx: list[int] = []
        info: Dict[str, Any]
        if A_ineq is not None and A_ineq.size > 0:
            p, active_idx, active_info = self._active_set_loop(
                H, g, A_eq, b_eq, A_ineq, b_ineq if b_ineq is not None else np.zeros(A_ineq.shape[0]), p
            )
            info = active_info
            # Keep some core diagnostics consistent
            info.setdefault("normal_step_norm", safe_norm(p_n))
            info.setdefault("tangential_step_norm", safe_norm(p - p_n))
            info.setdefault("nullspace_dim", Z.shape[1])
            info.setdefault("zeta_used", zeta)
            info.setdefault("criticality", crit_used)
            info.setdefault("criticality_shrinks", crit_shrinks)
            info.setdefault("preconditioned_reduced", prec_red is not None)
            info["constraint_violation"] = safe_norm(A_eq @ p + b_eq)
            info["model_reduction"] = self._model_reduction(H_op, g, p)
        else:
            info = {
                "status": status.value,
                "iterations": iters,
                "normal_step_norm": safe_norm(p_n),
                "tangential_step_norm": safe_norm(p - p_n),
                "step_norm": safe_norm(p),
                "constraint_violation": safe_norm(A_eq @ p + b_eq),
                "nullspace_dim": Z.shape[1],
                "model_reduction": self._model_reduction(H_op, g, p),
                "zeta_used": zeta,
                "criticality": crit_used,
                "criticality_shrinks": crit_shrinks,
                "preconditioned_reduced": prec_red is not None,
                "active_constraints": [],
            }

        # --- Multiplier recovery ---
        sigma = self._estimate_sigma(H_op, g, p)
        lam, nu = self._recover_multipliers(
            H_op, g, p, sigma, A_eq, A_ineq, active_idx if active_idx else None
        )
        info["active_constraints"] = active_idx
        self._last_was_criticality = bool(crit_used)
        return p, info, lam, nu


    # ---------- Inequality-constrained wrapper ----------
    def _solve_with_inequality_constraints(
        self, H: MatLike, g: Vec, A_ineq: np.ndarray, b_ineq: Vec
    ) -> Tuple[Vec, Dict[str, Any], Vec, Vec]:
        """Handle inequality constraints via active-set strategy."""
        p, info = self._solve_unconstrained(H, g)
        violations = A_ineq @ p + b_ineq
        violated_idx = violations > self.cfg.constraint_tol
        if not np.any(violated_idx):
            info["active_constraints"] = []
            sigma = self._estimate_sigma(make_operator(H, g.size), g, p)
            lam, nu = self._recover_multipliers(
                make_operator(H, g.size), g, p, sigma, None, A_ineq, None
            )
            return p, info, lam, nu
        p, active_idx, info = self._active_set_loop(
            H, g, np.zeros((0, g.size)), np.array([]), A_ineq, b_ineq, p
        )
        sigma = self._estimate_sigma(make_operator(H, g.size), g, p)
        lam, nu = self._recover_multipliers(
            make_operator(H, g.size), g, p, sigma, None, A_ineq, active_idx
        )
        info["active_constraints"] = active_idx
        return p, info, lam, nu

    # ---------- Active-set loop ----------
    def _active_set_loop(
        self,
        H: MatLike,
        g: Vec,
        A_eq: np.ndarray,
        b_eq: Vec,
        A_ineq: np.ndarray,
        b_ineq: Vec,
        p_init: Vec,
    ) -> Tuple[Vec, list, Dict[str, Any]]:
        """Active-set loop for inequality constraints, augmenting equalities."""
        p = p_init
        active: set[int] = set()
        info: Dict[str, Any] = {}
        it = 0
        norms = np.linalg.norm(A_ineq, axis=1)
        while it < self.cfg.max_active_set_iter:
            violations = A_ineq @ p + b_ineq
            violated = violations > self.cfg.constraint_tol
            if not np.any(violated):
                break
            max_v = np.max(violations[violated])
            candidates = np.where(violations >= max_v - 1e-12)[0]
            worst = candidates[np.argmax(norms[candidates])]
            if worst in active:
                break
            active.add(worst)
            idx = sorted(active)
            A_aug = np.vstack([A_eq, A_ineq[idx]]) if A_eq.size > 0 else A_ineq[idx]
            b_aug = (
                np.concatenate([b_eq, b_ineq[idx]]) if b_eq.size > 0 else b_ineq[idx]
            )
            p, info_eq, _, _ = self._solve_with_equality_constraints(H, g, A_aug, b_aug)
            info.update(info_eq)
            it += 1
        info.update(
            {
                "active_set_size": len(active),
                "active_set_iterations": it,
                "active_set_indices": sorted(active),
                "status": (
                    info.get("status", "feasible")
                    if not np.any(A_ineq @ p + b_ineq > self.cfg.constraint_tol)
                    else "infeasible"
                ),
            }
        )
        return p, sorted(active), info

    # ---------- Misc helpers ----------
    def _adaptive_zeta(self, constraint_violation: float, history_len: int) -> float:
        """Adapt normal/tangential step ratio based on feasibility and history."""
        base_zeta = self.cfg.zeta
        if constraint_violation < self.cfg.constraint_tol:
            return max(0.1, base_zeta - 0.1 * (1 + history_len / 10))
        avg_viol = (
            np.mean(self._feasibility_history[-5:])
            if self._feasibility_history
            else constraint_violation
        )
        return min(0.95, base_zeta + 0.05 * (1 + np.log1p(avg_viol)))

    def _model_reduction(self, H: MatLike, g: Vec, p: Vec) -> float:
        """Predicted reduction for the quadratic model at step p."""
        H_op = make_operator(H, p.size)
        return -(float(np.dot(g, p)) + 0.5 * float(np.dot(p, H_op @ p)))

    def model_reduction_alpha(self, H: MatLike, g: Vec, p: Vec, alpha: float) -> float:
        """
        Predicted reduction for α p:
        m(0) - m(αp) = α (-gᵀp) - 0.5 α² pᵀHp
        Uses a single Hp multiplication if not cached by caller.
        """
        if p.size == 0 or alpha == 0.0:
            return 0.0
        H_op = make_operator(H, p.size)
        Hp = H_op @ p
        gTp = float(np.dot(g, p))
        pTHp = float(np.dot(p, Hp))
        return alpha * (-gTp) - 0.5 * (alpha * alpha) * pTHp

    def _estimate_sigma(self, H_op: spla.LinearOperator, g: Vec, p: Vec) -> float:
        """Estimate trust region multiplier."""
        if p.size == 0:
            return 0.0
        Hp = H_op @ p
        num = np.dot(p, Hp + g)
        den = np.dot(p, p)
        if den <= 1e-14:
            return 0.0
        sigma = -num / den
        return max(0.0, sigma)

    def _recover_multipliers(
        self,
        H_op: spla.LinearOperator,
        g: Vec,
        p: Vec,
        sigma: float,
        A_eq: Optional[np.ndarray],
        A_ineq: Optional[np.ndarray],
        active_idx: Optional[list],
    ) -> Tuple[Vec, Vec]:
        """Recover Lagrange multipliers for equality and inequality constraints."""
        n = g.size
        r = H_op @ p + g + sigma * p
        m_eq = A_eq.shape[0] if A_eq is not None else 0
        m_ineq = A_ineq.shape[0] if A_ineq is not None else 0
        nu = np.zeros(m_eq)
        lam = np.zeros(m_ineq)
        if m_eq == 0 and (not active_idx or m_ineq == 0):
            return lam, nu
        blocks = []
        if m_eq > 0:
            blocks.append(A_eq.T)
        if active_idx and m_ineq > 0:
            blocks.append(A_ineq[active_idx].T)
        if not blocks:
            return lam, nu
        AT = np.hstack(blocks)
        reg = self.cfg.reg_floor
        try:
            if AT.shape[1] > 0:
                aug_AT = np.vstack([AT, np.sqrt(reg) * np.eye(AT.shape[1])])
                aug_rhs = np.concatenate([-r, np.zeros(AT.shape[1])])
                y = la.lstsq(aug_AT, aug_rhs, cond=self.cfg.rcond)[0]
            else:
                y = np.array([])
            if m_eq > 0:
                nu = y[:m_eq]
                y = y[m_eq:]
            if active_idx and m_ineq > 0:
                lam_active = np.maximum(0.0, y)
                lam[active_idx] = lam_active
                if m_eq > 0:
                    rhs = -r - A_ineq[active_idx].T @ lam_active
                    nu = la.lstsq(A_eq.T, rhs, cond=self.cfg.rcond)[0]
        except la.LinAlgError:
            lam, nu = np.zeros(m_ineq), np.zeros(m_eq)
        return lam, nu

    def update(
        self,
        predicted_reduction: float,
        actual_reduction: float,
        step_norm: float,
        constraint_violation: float = 0.0,
        H: Optional[MatLike] = None,
        p: Optional[Vec] = None,
    ) -> bool:
        rho = self._compute_ratio(predicted_reduction, actual_reduction)
        self._rho_history.append(rho)
        self._feasibility_history.append(constraint_violation)
        if len(self._rho_history) > 10:
            self._rho_history.pop(0)
        if len(self._feasibility_history) > 10:
            self._feasibility_history.pop(0)

        # Feasibility-weighted rho
        if self.cfg.feasibility_emphasis:
            feas_weight = (
                1.0
                if constraint_violation < self.cfg.constraint_tol
                else max(0.1, 1.0 - constraint_violation / max(1.0, constraint_violation))
            )
            rho = feas_weight * rho + (1 - feas_weight) * (
                1.0 if constraint_violation < self.cfg.constraint_tol else -1.0
            )

        # Curvature-aware tweak
        if self.cfg.curvature_aware and H is not None and p is not None:
            curv = self._curvature_along(H, p)
            self._curvature_estimate = curv
            if curv is not None and curv > 1e-6:
                self.cfg.gamma2 = min(2.5, 1.25 * self.cfg.gamma2)
            elif curv is not None and abs(curv) < 1e-12:
                self.cfg.gamma2 = min(1.2, self.cfg.gamma2)

        # Criticality safeguard: never expand immediately after criticality
        if rho < self.cfg.eta1:
            self.delta *= self.cfg.gamma1
            self.rejection_count += 1
            self._last_was_criticality = False
            return True

        self.rejection_count = 0
        if (not self._last_was_criticality) and (rho >= self.cfg.eta2) and (
            step_norm >= 0.8 * self.delta
        ):
            self.delta = min(self.cfg.delta_max, self.cfg.gamma2 * self.delta)

        # reset the criticality flag after one update cycle
        self._last_was_criticality = False
        self.delta = np.clip(self.delta, self.cfg.delta_min, self.cfg.delta_max)
        return False

    def _compute_ratio(self, pred_red: float, act_red: float) -> float:
        """Compute trust region ratio with safeguards."""
        if abs(pred_red) < 1e-14:
            return 1.0 if abs(act_red) < 1e-14 else (10.0 if act_red > 0 else -10.0)
        rho = act_red / pred_red
        return np.clip(rho, -100.0, 100.0)

    def _curvature_along(self, H: MatLike, p: Vec) -> Optional[float]:
        """Compute curvature along step direction."""
        try:
            H_op = make_operator(H, p.size)
            Hp = H_op @ p
            return float(np.dot(p, Hp) / max(np.dot(p, p), 1e-16))
        except Exception:
            return None

    # Public interface
    def get_radius(self) -> float:
        return self.delta

    def set_radius(self, radius: float) -> None:
        self.delta = np.clip(radius, self.cfg.delta_min, self.cfg.delta_max)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "radius": self.delta,
            "rejections": self.rejection_count,
            "avg_ratio": np.mean(self._rho_history) if self._rho_history else None,
            "avg_feasibility": (
                np.mean(self._feasibility_history)
                if self._feasibility_history
                else None
            ),
            "curvature_estimate": self._curvature_estimate,
            "config": self.cfg,
        }
