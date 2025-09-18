from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# --- NUMBA-accelerated Steihaug–Toint CG (dense/CSR) ---------------------
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from nlp.blocks.filter import Filter

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
class TRStatus(Enum):
    SUCCESS = "success"
    BOUNDARY = "boundary"
    NEG_CURV = "negative_curvature"
    MAX_ITER = "max_iterations"
    INFEASIBLE = "infeasible"


# ---------------------------- Utilities ---------------------------- #


# ---------------------------- Utilities ---------------------------- #
def safe_norm(x: Vec) -> float:
    return float(np.linalg.norm(x)) if x.size > 0 else 0.0


def make_operator(A: MatLike, n: int) -> spla.LinearOperator:
    if isinstance(A, spla.LinearOperator):
        return A
    if callable(A):
        return spla.LinearOperator((n, n), matvec=A, dtype=float)
    return spla.aslinearoperator(A)


def _asarray1d(a: Optional[np.ndarray], n: int) -> Optional[np.ndarray]:
    if a is None:
        return None
    x = np.asarray(a, float).reshape(-1)
    if x.size != n:
        raise ValueError(f"Expected shape ({n},) got {x.shape}")
    return x


def _boundary_tau_euclid(p: Vec, d: Vec, Delta: float) -> float:
    pTp, pTd, dTd = float(p @ p), float(p @ d), float(d @ d)
    if dTd <= 1e-16:
        return 0.0
    disc = max(0.0, pTd * pTd - dTd * (pTp - Delta * Delta))
    return (-pTd + np.sqrt(disc)) / dTd


# ---------------------------- Numba kernels ---------------------------- #
if _HAS_NUMBA:
    @njit(cache=True, fastmath=True)
    def _dot(a: np.ndarray, b: np.ndarray) -> float:
        s = 0.0
        for i in range(a.size):
            s += a[i] * b[i]
        return s

    @njit(cache=True, fastmath=True)
    def _axpy(alpha: float, x: np.ndarray, y: np.ndarray) -> None:
        for i in range(x.size):
            y[i] += alpha * x[i]

    @njit(cache=True, fastmath=True)
    def _dense_matvec(H: np.ndarray, x: np.ndarray, out: np.ndarray) -> None:
        n = H.shape[0]
        for i in range(n):
            s = 0.0
            row = H[i]
            for j in range(n):
                s += row[j] * x[j]
            out[i] = s

    @njit(cache=True, fastmath=True)
    def _csr_matvec(data, indices, indptr, x, out) -> None:
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
    def _boundary_tau_numba(p: np.ndarray, d: np.ndarray, Delta: float) -> float:
        pTp = _dot(p, p)
        pTd = _dot(p, d)
        dTd = _dot(d, d)
        if dTd <= 1e-16:
            return 0.0
        disc = pTd * pTd - dTd * (pTp - Delta * Delta)
        if disc < 0.0:
            disc = 0.0
        return (-pTd + np.sqrt(disc)) / dTd

    # SSOR apply for CSR (symmetric scaling)
    @njit(cache=True, fastmath=True)
    def _ssor_apply_numba(data, indices, indptr, invD, omega, c, r):
        n = r.size
        y = np.empty(n, dtype=np.float64)
        z = np.empty(n, dtype=np.float64)
        # forward: (D + ωL) y = r
        for i in range(n):
            s = r[i]
            row_start = indptr[i]
            row_end = indptr[i+1]
            for k in range(row_start, row_end):
                j = indices[k]
                if j < i:
                    s -= omega * data[k] * y[j]
            y[i] = s * invD[i]
        # backward: (D + ωU) z = y
        for i in range(n-1, -1, -1):
            s = y[i]
            row_start = indptr[i]
            row_end = indptr[i+1]
            for k in range(row_start, row_end):
                j = indices[k]
                if j > i:
                    s -= omega * data[k] * z[j]
            z[i] = s * invD[i]
        for i in range(n):
            z[i] *= c
        return z


# ---------------------------- Preconditioners ---------------------------- #
def make_jacobi_from_H(H: MatLike) -> Optional[np.ndarray]:
    try:
        if isinstance(H, np.ndarray):
            d = H.diagonal().astype(np.float64, copy=False)
        elif sp.issparse(H):
            d = H.diagonal().astype(np.float64, copy=False)
        else:
            return None
        d = np.where(np.abs(d) > 0, d, 1.0)
        return d
    except Exception:
        return None


class MetricPreconditioner:
    """z = (L L^T)^{-1} r via two triangular solves."""
    def __init__(self, L: np.ndarray):
        self.L = np.asarray(L, dtype=np.float64)

    def apply(self, r: np.ndarray) -> np.ndarray:
        y = la.solve_triangular(self.L, r, lower=True, check_finite=False)
        z = la.solve_triangular(self.L.T, y, lower=False, check_finite=False)
        return z


class SSORPreconditionerCSR:
    """Symmetric SOR preconditioner: M^{-1} ≈ ((D+ωU) D^{-1} (D+ωL)) * c  (applied via two sweeps)."""
    def __init__(self, A_csr: sp.csr_matrix, omega: float = 1.0):
        if not sp.isspmatrix_csr(A_csr):
            A_csr = A_csr.tocsr()
        self.A = A_csr
        self.omega = float(omega)
        d = A_csr.diagonal().astype(np.float64, copy=False)
        d = np.where(d == 0, 1.0, d)
        self.invD = (self.omega / d).astype(np.float64, copy=False)
        self.c = (2.0 - self.omega) / self.omega

    def apply(self, r: np.ndarray) -> np.ndarray:
        if not _HAS_NUMBA:
            # lightweight Python fallback
            A = self.A; invD = self.invD; omega = self.omega; c = self.c
            n = r.size
            y = np.empty_like(r)
            z = np.empty_like(r)
            # forward
            for i in range(n):
                s = r[i]
                row_start, row_end = A.indptr[i], A.indptr[i+1]
                for k in range(row_start, row_end):
                    j = A.indices[k]
                    if j < i:
                        s -= omega * A.data[k] * y[j]
                y[i] = s * invD[i]
            # backward
            for i in range(n - 1, -1, -1):
                s = y[i]
                row_start, row_end = A.indptr[i], A.indptr[i+1]
                for k in range(row_start, row_end):
                    j = A.indices[k]
                    if j > i:
                        s -= omega * A.data[k] * z[j]
                z[i] = s * invD[i]
            return c * z
        return _ssor_apply_numba(self.A.data, self.A.indices, self.A.indptr,
                                 self.invD, self.omega, self.c, r)


class ILUPreconditioner:
    """Incomplete LU preconditioner using spilu on a (shifted) symmetric part."""
    def __init__(self, A: sp.spmatrix, drop_tol=1e-4, fill_factor=10.0, shift=1e-10):
        if not sp.isspmatrix(A):
            A = sp.csr_matrix(A)
        S = 0.5 * (A + A.T)
        n = S.shape[0]
        S = S + shift * sp.eye(n, format='csc')
        self.lu = spla.spilu(S.tocsc(), drop_tol=drop_tol, fill_factor=fill_factor)

    def apply(self, r: np.ndarray) -> np.ndarray:
        return self.lu.solve(r).astype(np.float64, copy=False)


# ---------------------------- Unified Steihaug–Toint CG ---------------------------- #
def _python_cg_core(
    matvec: Callable[[np.ndarray], np.ndarray],
    g: np.ndarray,
    Delta: float,
    tol: float,
    maxiter: int,
    neg_curv_tol: float,
    apply_prec: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[np.ndarray, TRStatus, int]:
    """Generic PCG core in Python (fallback)."""
    n = g.size
    p = np.zeros(n, dtype=np.float64)
    r = -g.astype(np.float64, copy=False)
    z = apply_prec(r) if apply_prec is not None else r
    d = z.copy()

    if np.linalg.norm(r) <= tol:
        return p, TRStatus.SUCCESS, 0

    rz = float(r @ z)
    for k in range(maxiter):
        Hd = matvec(d)
        dTHd = float(d @ Hd)
        if dTHd <= neg_curv_tol * max(1.0, float(d @ d)):
            tau = _boundary_tau_euclid(p, d, Delta)
            return p + tau * d, TRStatus.NEG_CURV, k

        alpha = rz / max(dTHd, 1e-32)
        pTp = float(p @ p)
        pTd = float(p @ d)
        dTd = float(d @ d)
        pn2 = pTp + 2.0 * alpha * pTd + alpha * alpha * dTd
        if pn2 >= Delta * Delta:
            tau = _boundary_tau_euclid(p, d, Delta)
            return p + tau * d, TRStatus.BOUNDARY, k

        p = p + alpha * d
        r = r - alpha * Hd
        if np.linalg.norm(r) <= tol:
            return p, TRStatus.SUCCESS, k + 1

        z = apply_prec(r) if apply_prec is not None else r
        rz_next = float(r @ z)
        beta = rz_next / max(rz, 1e-32)
        d = z + beta * d
        rz = rz_next

    return p, TRStatus.MAX_ITER, maxiter


def steihaug_cg_fast(
    H: MatLike,
    g: np.ndarray,
    Delta: float,
    tol: float = 1e-8,
    maxiter: int = 200,
    neg_curv_tol: float = 1e-14,
    prec: Union[None, np.ndarray, sp.spmatrix, spla.LinearOperator, Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[np.ndarray, TRStatus, int]:
    """
    Unified Steihaug–Toint TR-CG with optional left preconditioning M^{-1}.
    - If H is ndarray/CSR and prec is diagonal-like or SSOR, we use Numba kernels.
    - Otherwise we fall back to a generic Python core.
    """
    n = g.size

    # Build matvec
    if isinstance(H, np.ndarray):
        H_dense = np.ascontiguousarray(H, dtype=np.float64)
        if _HAS_NUMBA:
            def matvec(x: np.ndarray) -> np.ndarray:
                out = np.empty_like(x)
                _dense_matvec(H_dense, x, out)
                return out
        else:
            H_op = make_operator(H_dense, n)
            matvec = lambda x: np.asarray(H_op @ x, float)

    elif sp.isspmatrix_csr(H):
        H_csr = H.tocsr().astype(np.float64, copy=False)
        if _HAS_NUMBA:
            def matvec(x: np.ndarray) -> np.ndarray:
                out = np.empty_like(x)
                _csr_matvec(H_csr.data, H_csr.indices, H_csr.indptr, x, out)
                return out
        else:
            H_op = make_operator(H_csr, n)
            matvec = lambda x: np.asarray(H_op @ x, float)

    else:
        H_op = make_operator(H, n)
        matvec = lambda x: np.asarray(H_op @ x, float)

    # Build preconditioner application
    apply_prec: Optional[Callable[[np.ndarray], np.ndarray]] = None

    if prec is None:
        # try cheap Jacobi
        d = make_jacobi_from_H(H)
        if d is not None:
            invd = np.where(np.abs(d) > 0, 1.0 / d, 0.0).astype(np.float64, copy=False)
            apply_prec = lambda r: invd * r
    elif isinstance(prec, np.ndarray) and prec.ndim == 1:
        invd = np.where(np.abs(prec) > 0, 1.0 / prec, 0.0).astype(np.float64, copy=False)
        apply_prec = lambda r: invd * r
    elif sp.isspmatrix(prec):
        d = prec.diagonal().astype(np.float64, copy=False)
        invd = np.where(np.abs(d) > 0, 1.0 / d, 0.0)
        apply_prec = lambda r: invd * r
    elif isinstance(prec, spla.LinearOperator):
        apply_prec = lambda r: np.asarray(prec @ r, float)
    elif callable(prec):
        apply_prec = lambda r: np.asarray(prec(r), float)

    # Numba-accelerated path already sits in matvec/SSOR kernels;
    # the CG loop itself remains Pythonic (branchless enough and clearer).
    return _python_cg_core(matvec, g, Delta, tol, maxiter, neg_curv_tol, apply_prec)


# ---------------------------- Metric helpers ---------------------------- #
def _psd_chol_with_shift(S: np.ndarray, shift_min: float) -> np.ndarray:
    try:
        return la.cholesky(S, lower=True, check_finite=True)
    except la.LinAlgError:
        w, V = la.eigh(0.5 * (S + S.T))
        add = max(shift_min, 1e-12 - float(np.min(w))) if w.size else shift_min
        Spos = (V * np.maximum(w + add, shift_min)) @ V.T
        return la.cholesky(Spos, lower=True, check_finite=True)


def low_rank_cholesky_update(L: np.ndarray, U: np.ndarray) -> np.ndarray:
    k = U.shape[1]
    L_new = L.copy()
    for i in range(k):
        u = U[:, i].copy()
        for j in range(L_new.shape[0]):
            r = np.sqrt(L_new[j, j] ** 2 + u[j] ** 2)
            c = r / L_new[j, j]
            s = u[j] / L_new[j, j]
            L_new[j, j] = r
            if j + 1 < L_new.shape[0]:
                L_new[j + 1 :, j] = (L_new[j + 1 :, j] + s * u[j + 1 :]) / c
                u[j + 1 :] = c * u[j + 1 :] - s * L_new[j + 1 :, j]
    return L_new


class TrustRegionManager:
    """
    Trust Region manager with Byrd–Omojokun decomposition, SOC correction,
    ellipsoidal/Euclidean norms, optional box bounds, filter-based globalization.
    Uses implicit tangent projection (no explicit nullspace basis).
    """

    # ------------------------- Init ------------------------- #
    def __init__(self, config: Optional[TRConfig] = None):
        self.cfg = TRConfig() if config is None else config
        self.delta = self.cfg.delta0
        self.rejection_count = 0

        # Adaptive state
        self._rho_history: list[float] = []
        self._feasibility_history: list[float] = []
        self._curvature_estimate: Optional[float] = None
        self._f_history: list[float] = []
        self._last_was_criticality = False

        # Cache (metric only now)
        self.norm_type: str = self.cfg.norm_type
        self.M: Optional[np.ndarray] = None
        self._L: Optional[np.ndarray] = None  # M = L L^T

        # Box context
        self._box_ctx: Optional[
            tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]
        ] = None
        self._box_mode: str = "alpha"

    # ========================= Small utilities (DRY) ========================= #
    # ---- shapes/box ---- #
    @staticmethod
    def _as1d(v, n):
        if v is None:
            return None
        a = np.asarray(v, float).reshape(-1)
        if a.size != n:
            raise ValueError(f"expected shape ({n},) got {a.shape}")
        return a

    def _sanitize_box(self, x, lb, ub, n):
        lb = self._as1d(lb, n)
        ub = self._as1d(ub, n)
        if lb is not None and ub is not None and np.any(lb > ub):
            raise ValueError("lb > ub")
        if (lb is not None or ub is not None) and x is None:
            raise ValueError("box bounds provided but x is None")
        return x, lb, ub

    def _metric_factor(self):
        return self._L if (self.norm_type == "ellip" and self._L is not None) else None

    # ---- norms/clipping ---- #
    def _tr_norm(self, p):
        L = self._metric_factor()
        if L is None:
            return float(np.linalg.norm(p))
        y = L.T @ p
        return float(np.linalg.norm(y))

    def _clip_to_radius(self, p, Delta):
        nrm = self._tr_norm(p)
        return p if (nrm <= Delta or nrm <= 1e-16) else (Delta / nrm) * p

    def _clip_and_box(self, p):
        return self._maybe_enforce_box(self._clip_to_radius(p, self.delta))

    # ---- box handling ---- #
    def _alpha_max_box(self, x, d, lb, ub, tau=None):
        if (lb is None and ub is None) or d.size == 0:
            return 1.0
        t = self.cfg.tau_ftb if tau is None else float(tau)
        t = float(np.clip(t, 0.0, 0.999999))
        amax = 1.0
        if lb is not None:
            mask = d < 0
            if np.any(mask):
                amax = min(amax, np.min((lb[mask] - x[mask]) / d[mask]))
        if ub is not None:
            mask = d > 0
            if np.any(mask):
                amax = min(amax, np.min((ub[mask] - x[mask]) / d[mask]))
        return float(np.clip(max(0.0, amax) * t, 0.0, 1.0))

    def _enforce_box_on_step(self, x, p, lb, ub, tau=None):
        if p.size == 0 or (lb is None and ub is None):
            return p
        if self._box_mode == "projection":
            lo = lb if lb is not None else -np.inf
            hi = ub if ub is not None else +np.inf
            return np.clip(x + p, lo, hi) - x
        return self._alpha_max_box(x, p, lb, ub, tau) * p

    def _maybe_enforce_box(self, p):
        if self._box_ctx is None:
            return p
        x0, lb0, ub0 = self._box_ctx
        return self._enforce_box_on_step(x0, p, lb0, ub0)

    # ---- linear algebra helpers ---- #
    @staticmethod
    def _psd_chol_with_shift(S, shift_min):
        """Cholesky with automatic diagonal shift to make SPD."""
        try:
            return la.cholesky(S, lower=True, check_finite=True)
        except la.LinAlgError:
            w, V = la.eigh(0.5 * (S + S.T))
            add = max(shift_min, 1e-12 - float(np.min(w))) if w.size else shift_min
            Spos = (V * np.maximum(w + add, shift_min)) @ V.T
            return la.cholesky(Spos, lower=True, check_finite=True)

    @staticmethod
    def _lstsq_reg(A, b, rcond, reg=0.0):
        """min ||A x - b||^2 + reg||x||^2 via normal eq augmented."""
        m, n = A.shape
        if reg <= 0:
            return la.lstsq(A, b, cond=rcond)[0]
        augA = np.vstack([A, np.sqrt(reg) * np.eye(n)])
        augb = np.concatenate([b, np.zeros(n)])
        return la.lstsq(augA, augb, cond=rcond)[0]

    # ---- metric ---- #
    def set_metric(self, M):
        if self.norm_type != "ellip":
            self.M = self._L = None
            return
        S = 0.5 * (M + M.T)
        L = self._psd_chol_with_shift(S, self.cfg.metric_shift)
        self.M, self._L = L @ L.T, L  # store SPD & factor

    def set_metric_from_H(self, H):
        if self.norm_type != "ellip":
            self.M = self._L = None
            return
        Hd = (
            H
            if isinstance(H, np.ndarray)
            else (H.toarray() if sp.issparse(H) else np.asarray(H))
        )
        S = 0.5 * (Hd + Hd.T)
        self.set_metric(S + self.cfg.metric_shift * np.eye(S.shape[0]))

    def update_metric(self, M_new):
        if self.M is None or self._L is None:
            return self.set_metric(M_new)
        try:
            self._L = low_rank_cholesky_update(self._L, M_new - self.M)
            self.M = 0.5 * (M_new + M_new.T)
        except la.LinAlgError:
            self.set_metric(M_new)

    # ---- implicit tangent projection helpers (no explicit nullspace) ---- #
    def _AAT_solve(self, A, rhs):
        """Solve (A A^T + reg I) y = rhs (small m) robustly."""
        m = A.shape[0]
        M = A @ A.T + max(self.cfg.reg_floor, 1e-12) * np.eye(m)
        try:
            return la.solve(M, rhs, assume_a="pos")
        except la.LinAlgError:
            return la.lstsq(M, rhs, cond=self.cfg.rcond)[0]

    def _proj(self, v: np.ndarray, A: Optional[np.ndarray]) -> np.ndarray:
        """Tangent projection: P v = v - A^T (A A^T)^-1 A v."""
        if A is None or A.size == 0:
            return v
        y = self._AAT_solve(A, A @ v)
        return v - A.T @ y

    def _tangent_linop(self, H, A_eq, n: int) -> spla.LinearOperator:
        """
        LinearOperator y = P H (P x) without forming P explicitly.
        Works with ndarray/sparse/LinearOperator H.
        """
        H_op = make_operator(H, n)

        def mv(x: np.ndarray) -> np.ndarray:
            Px = self._proj(x, A_eq)
            Hx = H_op @ Px
            return self._proj(Hx, A_eq)

        return spla.LinearOperator((n, n), matvec=mv, dtype=float)

    def _projected_preconditioner(
        self, A_eq, base_prec
    ) -> Optional[Callable[[np.ndarray], np.ndarray]]:
        """
        Wrap a base preconditioner M^{-1} as ~P M^{-1} P.
        base_prec: None | 1D diag array | spmatrix(diag) | LinearOperator | callable
        """
        if base_prec is None:
            return None

        if isinstance(base_prec, np.ndarray) and base_prec.ndim == 1:
            invdiag = np.where(
                np.abs(base_prec) > 0,
                1.0 / base_prec,
                1.0 / max(1e-12, 0.01 * np.mean(np.abs(base_prec))),
            )

            def apply(r):
                return self._proj(invdiag * self._proj(r, A_eq), A_eq)

            return apply

        if sp.isspmatrix(base_prec):
            d = base_prec.diagonal().astype(np.float64, copy=False)
            invdiag = np.where(
                np.abs(d) > 0, 1.0 / d, 1.0 / max(1e-12, 0.01 * np.mean(np.abs(d)))
            )

            def apply(r):
                return self._proj(invdiag * self._proj(r, A_eq), A_eq)

            return apply

        if isinstance(base_prec, spla.LinearOperator):

            def apply(r):
                return self._proj(
                    np.asarray(base_prec @ self._proj(r, A_eq), float), A_eq
                )

            return apply

        if callable(base_prec):

            def apply(r):
                return self._proj(
                    np.asarray(base_prec(self._proj(r, A_eq)), float), A_eq
                )

            return apply

        return None

    # ---- projected grad norm (with box tweaks) ---- #
    def _projected_grad_norm(self, g, A_eq):
        if self._box_ctx is None:
            gt = self._proj(g, A_eq)
        else:
            x, lb, ub = self._box_ctx
            tol = 1e-12
            gp = g.copy()
            if lb is not None:
                gp[(x <= lb + tol) & (gp > 0)] = 0.0
            if ub is not None:
                gp[(x >= ub - tol) & (gp < 0)] = 0.0
            gt = self._proj(gp, A_eq)
        return self._tr_norm(gt)

    # ---- CG runner ---- #
    def _run_cg(self, H, g, Delta, tol, prec=None):
        L = self._metric_factor()
        if L is None:
            return steihaug_cg_fast(
                H, g, Delta, tol, self.cfg.cg_maxiter, self.cfg.neg_curv_tol, prec=prec
            )
        Mprec = MetricPreconditioner(L)
        return steihaug_cg_fast(
            H,
            g,
            Delta,
            tol,
            self.cfg.cg_maxiter,
            self.cfg.neg_curv_tol,
            prec=Mprec.apply,
        )

    # ---- model reductions & sigma ---- #
    def _model_reduction(self, H, g, p):
        H_op = make_operator(H, p.size)
        return -(float(g @ p) + 0.5 * float(p @ (H_op @ p)))

    def _pred_red_cubic(self, H, g, p, sigma):
        H_op = make_operator(H, p.size)
        quad = float(g @ p) + 0.5 * float(p @ (H_op @ p))
        return -(quad + (sigma / 3.0) * (self._tr_norm(p) ** 3))

    def _estimate_sigma(self, H_op, g, p):
        if p.size == 0:
            return 0.0
        den = self._tr_norm(p) ** 2
        if den <= 1e-14:
            return 0.0
        num = float(p @ (H_op @ p) + p @ g)
        return max(0.0, -num / den)

    # ========================= Core solvers ========================= #
    def _cg_tol(self, gnorm):
        return min(self.cfg.cg_tol, self.cfg.cg_tol_rel * max(gnorm, 1e-16))

    def _solve_unconstrained(self, H, g):
        gnorm = self._tr_norm(g)
        tol = self._cg_tol(gnorm)

        # criticality safeguard
        crit_used = False
        shrinks = 0
        if self.cfg.criticality_enabled:
            for _ in range(self.cfg.max_crit_shrinks):
                if self._projected_grad_norm(g, None) <= self.cfg.kappa_g * self.delta:
                    self.delta = max(
                        self.cfg.delta_min, self.cfg.theta_crit * self.delta
                    )
                    crit_used = True
                    shrinks += 1
                else:
                    break

        # preconditioner
        prec = None
        if (
            self.cfg.use_prec
            and self.cfg.prec_kind == "auto_jacobi"
            and self._metric_factor() is None
        ):
            prec = make_jacobi_from_H(H)

        p, status, iters = self._run_cg(H, g, self.delta, tol, prec=prec)
        p = self._clip_and_box(p)

        info = dict(
            status=status.value if isinstance(status, TRStatus) else str(status),
            iterations=int(iters),
            step_norm=self._tr_norm(p),
            model_reduction=self._model_reduction(H, g, p),
            criticality=crit_used,
            criticality_shrinks=shrinks,
            preconditioned=prec is not None,
        )
        self._last_was_criticality = crit_used
        return p, info

    def _min_norm_normal(self, Aeq, beq):
        """Compute minimum-norm normal step for current metric."""
        n = Aeq.shape[1]
        L = self._metric_factor()
        if L is None:
            return self._lstsq_reg(Aeq, -beq, self.cfg.rcond, reg=self.cfg.reg_floor)

        # p_n = M^{-1} A^T (A M^{-1} A^T)^{-1} (-b)
        def Minv(v):
            y = la.solve_triangular(L, v, lower=True, check_finite=False)
            return la.solve_triangular(L.T, y, lower=False, check_finite=False)

        AMiAT = Aeq @ Minv(Aeq.T)
        lam = self._lstsq_reg(AMiAT, -beq, self.cfg.rcond, reg=self.cfg.reg_floor)
        return Minv(Aeq.T @ lam)

    def _solve_with_equality_constraints(
        self, H, g, A_eq, b_eq, A_ineq=None, b_ineq=None
    ):
        n = g.size
        H_op = make_operator(H, n)

        # criticality in tangent
        crit_used = False
        shrinks = 0
        if self.cfg.criticality_enabled:
            for _ in range(self.cfg.max_crit_shrinks):
                if self._projected_grad_norm(g, A_eq) <= self.cfg.kappa_g * self.delta:
                    self.delta = max(
                        self.cfg.delta_min, self.cfg.theta_crit * self.delta
                    )
                    crit_used = True
                    shrinks += 1
                else:
                    break

        # normal step
        zeta = (
            self._adaptive_zeta(safe_norm(b_eq), len(self._feasibility_history))
            if self.cfg.adaptive_zeta
            else self.cfg.zeta
        )
        p_n = self._min_norm_normal(A_eq, b_eq)
        p_n = self._clip_to_radius(p_n, zeta * self.delta)

        # ensure box feasibility of p_n w/out breaking equalities (recenter)
        if self._box_ctx is not None:
            x0, lb0, ub0 = self._box_ctx
            if self._alpha_max_box(x0, p_n, lb0, ub0) < 1.0:
                xp = self._maybe_enforce_box(p_n) + x0
                p_n = self._min_norm_normal(A_eq, b_eq + A_eq @ (xp - (x0 + p_n)))
                p_n = self._clip_to_radius(p_n, zeta * self.delta)

        # ---------------- tangential step via implicit projection ----------------
        g_tilde = g + (H_op @ p_n)
        remaining_radius = np.sqrt(max(0.0, self.delta**2 - self._tr_norm(p_n) ** 2))

        Htan = self._tangent_linop(H, A_eq, n)
        gtan = self._proj(g_tilde, A_eq)

        # Choose a base preconditioner (Jacobi or metric), then project it
        base_prec = None
        if self.cfg.use_prec:
            if self.norm_type == "ellip" and (self._metric_factor() is not None):
                Mprec = MetricPreconditioner(self._metric_factor())
                base_prec = Mprec.apply  # callable
            elif self.cfg.prec_kind == "auto_jacobi":
                base_prec = make_jacobi_from_H(H)  # 1D diag if available

        prec_apply = self._projected_preconditioner(A_eq, base_prec)

        # Steihaug CG on the projected operator within the remaining TR radius
        p_t, status, iters = steihaug_cg_fast(
            Htan,
            gtan,
            remaining_radius,
            self._cg_tol(safe_norm(gtan)),
            self.cfg.cg_maxiter,
            self.cfg.neg_curv_tol,
            prec=prec_apply,
        )

        # If nearly null tangential step, fall back to constrained Cauchy in tangent
        if self._tr_norm(p_t) <= 1e-14:
            v = self._proj(g_tilde, A_eq)
            if self._tr_norm(v) > 1e-10:
                p_t = self._constrained_cauchy_in_tangent(
                    H, g_tilde, A_eq, remaining_radius
                )

        # guard TR + bounds (preserve A_eq feasibility by scaling tangential)
        p_t = self._clip_to_radius(p_t, remaining_radius)
        p = p_n + p_t
        if self._box_ctx is not None:
            x0, lb0, ub0 = self._box_ctx
            beta = self._alpha_max_box(x0 + p_n, p_t, lb0, ub0)
            if beta < 1.0:
                p = p_n + beta * p_t

        # active-set inequalities on top
        active_idx = []
        if A_ineq is not None and A_ineq.size > 0:
            p, active_idx, info = self._active_set_loop(
                H,
                g,
                A_eq,
                b_eq,
                A_ineq,
                (b_ineq if b_ineq is not None else np.zeros(A_ineq.shape[0])),
                p,
            )
        else:
            null_dim = n - (
                np.linalg.matrix_rank(A_eq, tol=self.cfg.rcond)
                if A_eq is not None and A_eq.size
                else 0
            )
            info = dict(
                status=status.value if isinstance(status, TRStatus) else str(status),
                iterations=int(iters),
                normal_step_norm=self._tr_norm(p_n),
                tangential_step_norm=self._tr_norm(p - p_n),
                step_norm=self._tr_norm(p),
                constraint_violation=safe_norm(A_eq @ p + b_eq),
                nullspace_dim=null_dim,
                model_reduction=self._model_reduction(H, g, p),
                zeta_used=zeta,
                criticality=crit_used,
                criticality_shrinks=shrinks,
                preconditioned_reduced=(prec_apply is not None),
                active_constraints=[],
            )

        sigma = self._estimate_sigma(H_op, g, p)
        lam, nu, muL, muU = self._recover_multipliers(
            H_op, g, p, sigma, A_eq, A_ineq, active_idx if active_idx else None
        )
        info["active_constraints"] = active_idx
        if hasattr(self, "_last_bounds_info"):
            info.update(
                {
                    "active_lower_bounds": self._last_bounds_info[
                        "active_lower_bounds"
                    ],
                    "active_upper_bounds": self._last_bounds_info[
                        "active_upper_bounds"
                    ],
                    "mu_lower": self._last_bounds_info["mu_lower"],
                    "mu_upper": self._last_bounds_info["mu_upper"],
                }
            )
        self._last_was_criticality = crit_used
        return p, info, lam, nu

    def _solve_with_inequality_constraints(self, H, g, A_ineq, b_ineq):
        p, info = self._solve_unconstrained(H, g)
        viol = A_ineq @ p + b_ineq
        H_op = make_operator(H, g.size)
        if not np.any(viol > self.cfg.constraint_tol):
            info["active_constraints"] = []
            lam, nu, _, _ = self._recover_multipliers(
                H_op, g, p, self._estimate_sigma(H_op, g, p), None, A_ineq, None
            )
            return p, info, lam, nu
        p, active_idx, info2 = self._active_set_loop(
            H, g, np.empty((0, g.size)), np.empty((0,)), A_ineq, b_ineq, p
        )
        info.update(info2)
        lam, nu, _, _ = self._recover_multipliers(
            H_op, g, p, self._estimate_sigma(H_op, g, p), None, A_ineq, active_idx
        )
        info["active_constraints"] = active_idx
        return p, info, lam, nu

    # ========================= Active set & multipliers ========================= #
    def _active_set_loop(self, H, g, A_eq, b_eq, A_ineq, b_ineq, p_init):
        if A_eq is None:
            A_eq = np.empty((0, g.size))
            b_eq = np.empty((0,))
        p = p_init
        active: set[int] = set()
        info = {}
        it = 0
        n = g.size
        rank_eq = (
            np.linalg.matrix_rank(A_eq, tol=self.cfg.rcond) if A_eq.size > 0 else 0
        )
        max_active = max(0, min(A_ineq.shape[0], n - rank_eq))

        while it < self.cfg.max_active_set_iter:
            viol = A_ineq @ p + b_ineq
            mask = viol > self.cfg.constraint_tol
            if not np.any(mask):
                break

            # current rank
            A_cur = (
                A_eq
                if len(active) == 0
                else (
                    np.vstack([A_eq, A_ineq[sorted(active)]])
                    if A_eq.size
                    else A_ineq[sorted(active)]
                )
            )
            rcur = np.linalg.matrix_rank(A_cur, tol=self.cfg.rcond)
            if rcur >= n or len(active) >= max_active:
                break

            # pick most violated that increases rank
            cand_order = np.argsort(-(viol[np.where(mask)[0]]))
            added = False
            last = None
            for idx in np.where(mask)[0][cand_order]:
                if idx in active:
                    continue
                A_test = (
                    A_ineq[idx : idx + 1]
                    if A_cur.size == 0
                    else np.vstack([A_cur, A_ineq[idx : idx + 1]])
                )
                if np.linalg.matrix_rank(A_test, tol=self.cfg.rcond) > rcur:
                    active.add(idx)
                    last = idx
                    added = True
                    break
            if not added:
                break

            # resolve as equalities
            ids = sorted(active)
            A_aug = np.vstack([A_eq, A_ineq[ids]]) if A_eq.size else A_ineq[ids]
            b_aug = np.concatenate([b_eq, b_ineq[ids]]) if b_eq.size else b_ineq[ids]
            try:
                p, info_eq, _, _ = self._solve_with_equality_constraints(
                    H, g, A_aug, b_aug
                )
            except Exception:
                if last in active:
                    active.remove(last)
                break

            # box pullback only (don’t scale full step)
            if self._box_ctx is not None:
                x0, lb0, ub0 = self._box_ctx
                p = (
                    np.clip(
                        x0 + p,
                        lb0 if lb0 is not None else -np.inf,
                        ub0 if ub0 is not None else +np.inf,
                    )
                    - x0
                )
            it += 1
            info.update(info_eq)

        viol = A_ineq @ p + b_ineq
        tol = max(self.cfg.constraint_tol, 1e-10)
        info.update(
            dict(
                active_set_size=len(active),
                active_set_iterations=it,
                active_set_indices=sorted(active),
                status=(
                    "feasible"
                    if (
                        viol.size == 0 or np.max(viol) <= tol + 10 * np.finfo(float).eps
                    )
                    else "infeasible"
                ),
                max_linearized_violation=float(np.max(viol)) if viol.size else 0.0,
            )
        )
        return p, sorted(active), info

    def _detect_active_bounds(self, p, tol=1e-10):
        if self._box_ctx is None:
            n = p.size
            return np.zeros(n, bool), np.zeros(n, bool)
        x, lb, ub = self._box_ctx
        L = (
            (x + p) <= ((lb if lb is not None else -np.inf) + tol)
            if lb is not None
            else np.zeros(p.size, bool)
        )
        U = (
            (x + p) >= ((ub if ub is not None else +np.inf) - tol)
            if ub is not None
            else np.zeros(p.size, bool)
        )
        return L, U

    def _recover_multipliers(self, H_op, g, p, sigma, A_eq, A_ineq, active_idx):
        """Solve [Aeq^T | Aact^T | I_L | -I_U] y ≈ -(H p + g + σp), project to λ,μ≥0."""
        n = g.size
        r = H_op @ p + g + sigma * p

        AeqT = A_eq.T if (A_eq is not None and A_eq.size) else np.empty((n, 0))
        AactT = (
            A_ineq[np.array(active_idx, int)].T
            if (active_idx and A_ineq is not None and A_ineq.size)
            else np.empty((n, 0))
        )

        actL, actU = self._detect_active_bounds(p)
        idxL = np.where(actL)[0]
        idxU = np.where(actU)[0]
        IL = np.eye(n)[:, idxL] if idxL.size else np.empty((n, 0))
        IU = np.eye(n)[:, idxU] if idxU.size else np.empty((n, 0))

        blocks = [B for B in (AeqT, AactT, IL, -IU) if B.size]
        if not blocks:
            lam = np.zeros(A_ineq.shape[0] if A_ineq is not None else 0)
            nu = np.zeros(A_eq.shape[0] if A_eq is not None else 0)
            muL = np.zeros(n)
            muU = np.zeros(n)
            self._last_bounds_info = dict(
                active_lower_bounds=[],
                active_upper_bounds=[],
                mu_lower=muL,
                mu_upper=muU,
            )
            return lam, nu, muL, muU

        AT = np.hstack(blocks)
        m = AT.shape[1]
        y = self._lstsq_reg(
            AT, -np.asarray(r, float), self.cfg.rcond, reg=self.cfg.reg_floor
        )

        ofs = 0
        nu = np.array([])
        lam_act = np.array([])
        muL_s = np.array([])
        muU_s = np.array([])
        if AeqT.size:
            nu = y[ofs : ofs + AeqT.shape[1]]
            ofs += AeqT.shape[1]
        if AactT.size:
            lam_act = np.maximum(0.0, y[ofs : ofs + AactT.shape[1]])
            ofs += AactT.shape[1]
        if IL.size:
            muL_s = np.maximum(0.0, -y[ofs : ofs + IL.shape[1]])
            ofs += IL.shape[1]
        if IU.size:
            muU_s = np.maximum(0.0, -y[ofs : ofs + IU.shape[1]])
            ofs += IU.shape[1]

        muL = np.zeros(n)
        muU = np.zeros(n)
        if idxL.size:
            muL[idxL] = muL_s
        if idxU.size:
            muU[idxU] = muU_s

        if A_ineq is not None and A_ineq.size:
            lam = np.zeros(A_ineq.shape[0])
            if active_idx and lam_act.size:
                lam[np.array(active_idx, int)] = lam_act
        else:
            lam = np.array([])

        if A_eq is None or not A_eq.size:
            nu = np.array([])

        self._last_bounds_info = dict(
            active_lower_bounds=idxL.tolist(),
            active_upper_bounds=idxU.tolist(),
            mu_lower=muL,
            mu_upper=muU,
        )
        return lam, nu, muL, muU

    # ========================= Update logic ========================= #
    def _adaptive_zeta(self, viol, hist_len):
        base = self.cfg.zeta
        if viol < self.cfg.constraint_tol:
            return max(0.1, base - 0.1 * (1 + hist_len / 10))
        avg = (
            np.mean(self._feasibility_history[-5:])
            if self._feasibility_history
            else viol
        )
        return min(0.95, base + 0.05 * (1 + np.log1p(avg)))

    def _dynamic_eta(self, viol, it):
        eta1 = self.cfg.eta1 * min(1.0, 1.0 + viol)
        eta2 = self.cfg.eta2 * max(0.5, 1.0 - it / self.cfg.max_iter)
        return float(eta1), float(eta2)

    def _compute_ratio(self, pred_red, act_red):
        if abs(pred_red) < 1e-14:
            return 1.0 if abs(act_red) < 1e-14 else (10.0 if act_red > 0 else -10.0)
        return float(np.clip(act_red / pred_red, -100.0, 100.0))

    def _curvature_along(self, H, p):
        try:
            H_op = make_operator(H, p.size)
            pTp = self._tr_norm(p) ** 2
            if pTp <= 1e-16:
                return None
            curv = float((p @ (H_op @ p)) / pTp)
            if isinstance(H, np.ndarray):
                e = np.linalg.eigvalsh(H)
                if e.size:
                    cond = max(abs(e)) / max(1e-16, min(abs(e)))
                    curv *= min(1.0, 1e3 / cond)
            return curv
        except Exception:
            return None

    def update(
        self,
        predicted_reduction,
        actual_reduction,
        step_norm,
        constraint_violation=0.0,
        H=None,
        p=None,
    ):
        self._feasibility_history.append(constraint_violation)
        if len(self._feasibility_history) > self.cfg.history_length:
            self._feasibility_history.pop(0)

        if self.cfg.non_monotone and self._f_history:
            f_ref = np.max(self._f_history[-self.cfg.non_monotone_window :])
            rho = (f_ref - actual_reduction) / max(predicted_reduction, 1e-16)
        else:
            rho = self._compute_ratio(predicted_reduction, actual_reduction)

        self._rho_history.append(rho)
        if len(self._rho_history) > self.cfg.history_length:
            self._rho_history.pop(0)

        if hasattr(self, "_last_f_new"):
            self._f_history.append(self._last_f_new)
            if len(self._f_history) > self.cfg.non_monotone_window:
                self._f_history.pop(0)

        if self.cfg.feasibility_emphasis:
            feas_ok = constraint_violation < self.cfg.constraint_tol
            w = (
                1.0
                if feas_ok
                else max(
                    0.1, 1.0 - constraint_violation / max(1.0, constraint_violation)
                )
            )
            rho = w * rho + (1 - w) * (1.0 if feas_ok else -1.0)

        eta1, eta2 = self._dynamic_eta(constraint_violation, len(self._rho_history))

        if self.cfg.curvature_aware and H is not None and p is not None:
            curv = self._curvature_along(H, p)
            self._curvature_estimate = curv
            if curv is not None and curv > 1e-6:
                self.cfg.gamma2 = min(2.5, 1.25 * self.cfg.gamma2)
            elif curv is not None and abs(curv) < 1e-12:
                self.cfg.gamma2 = min(1.2, self.cfg.gamma2)

        if rho < eta1:
            self.delta *= self.cfg.gamma1
            self.rejection_count += 1
            self._last_was_criticality = False
            rej = True
        else:
            self.rejection_count = 0
            if (
                (not self._last_was_criticality)
                and (rho >= eta2)
                and (step_norm >= 0.8 * self.delta)
            ):
                self.delta = min(self.cfg.delta_max, self.cfg.gamma2 * self.delta)
            self._last_was_criticality = False
            rej = False

        self.delta = float(np.clip(self.delta, self.cfg.delta_min, self.cfg.delta_max))
        return rej

    def set_f_current(self, f_current: float):
        self._last_f_new = f_current
        self._f_history.append(f_current)
        if len(self._f_history) > self.cfg.non_monotone_window:
            self._f_history.pop(0)

    # ========================= SOC (uses shared helpers) ========================= #
    def clip_correction_to_radius(self, s, q):
        if self._tr_norm(s + q) <= self.delta + 1e-14 or self._tr_norm(q) <= 1e-16:
            return q
        # compute boundary intersection in TR norm
        L = self._metric_factor()
        if L is None:
            pTp, pTd, dTd = float(s @ s), float(s @ q), float(q @ q)
            disc = max(0.0, pTd * pTd - dTd * (pTp - self.delta**2))
            t = 0.0 if dTd <= 1e-16 else (-pTd + np.sqrt(disc)) / dTd
        else:
            y0, yd = L.T @ s, L.T @ q
            a, b, c = (
                float(yd @ yd),
                2.0 * float(y0 @ yd),
                float(y0 @ y0) - self.delta**2,
            )
            disc = max(0.0, b * b - 4 * a * c)
            t = 0.0 if a <= 1e-16 else (-b + np.sqrt(disc)) / (2 * a)
        return float(np.clip(t, 0.0, 1.0)) * q

    def _soc_correction(
        self,
        model,
        x,
        p,
        H,
        lam_ineq,
        mu=0.0,
        wE=10.0,
        wI=1.0,
        tolE=1e-8,
        violI=0.0,
        reg=1e-8,
        sigma0=1e-10,
    ):
        n = p.size
        if model is None:
            return np.zeros(n), False, {"reason": "no_model"}
        x_trial = x + p
        d = model.eval_all(x_trial)  # expects cE,cI,JE,JI (& optionally f)
        cE, cI, JE, JI = (
            d.get("cE", None),
            d.get("cI", None),
            d.get("JE", None),
            d.get("JI", None),
        )

        rows, rhs, wts = [], [], []
        if cE is not None and JE is not None and getattr(JE, "size", 0):
            mask = np.abs(np.asarray(cE).ravel()) > tolE
            if np.any(mask):
                JE_ = JE[mask] if sp.issparse(JE) else np.asarray(JE)[mask]
                rows.append(JE_)
                rhs.append(-np.asarray(cE).ravel()[mask])
                wts.append(np.full(JE_.shape[0], wE))
        if cI is not None and JI is not None and getattr(JI, "size", 0):
            mask = np.asarray(cI).ravel() > violI
            if np.any(mask):
                JI_ = JI[mask] if sp.issparse(JI) else np.asarray(JI)[mask]
                ci = np.asarray(cI).ravel()[mask]
                if (mu > 0) and (lam_ineq is not None) and (lam_ineq.size == cI.size):
                    lam_sel = np.maximum(np.asarray(lam_ineq).ravel()[mask], 1e-12)
                    rhs_I = -(ci + (mu / lam_sel))
                else:
                    rhs_I = -ci
                rows.append(JI_)
                rhs.append(rhs_I)
                wts.append(np.full(JI_.shape[0], wI))
        if not rows:
            return np.zeros(n), False, {"reason": "no_rows"}

        J = np.vstack([R.toarray() if sp.issparse(R) else R for R in rows])
        r = np.concatenate(rhs)
        w = np.sqrt(np.maximum(np.concatenate(wts), 1e-16))
        Jw = J * w[:, None]
        rw = r * w

        q = None
        if (self.norm_type == "ellip") and (H is not None):
            Hd = (
                H
                if isinstance(H, np.ndarray)
                else (H.toarray() if sp.issparse(H) else np.asarray(H))
            )
            A = 0.5 * (Hd + Hd.T) + sigma0 * np.eye(Hd.shape[0])
            L = self._psd_chol_with_shift(A, self.cfg.metric_shift)

            def Minv(v):
                y = la.solve_triangular(L, v, lower=True, check_finite=False)
                return la.solve_triangular(L.T, y, lower=False, check_finite=False)

            # (J Minv J^T + λI) y = rw
            lam_reg = reg * max(1.0, float(np.trace(Jw.T @ Jw)) / max(1, Jw.shape[0]))
            S = Jw @ (Minv(Jw.T)) + lam_reg * np.eye(Jw.shape[0])
            y = la.solve(S, rw, assume_a="pos") if S.size else np.zeros_like(rw)
            q = -Minv(Jw.T @ y)
        else:
            m, nJ = Jw.shape
            if m >= nJ:
                JtJ = Jw.T @ Jw
                lam_reg = reg * (np.trace(JtJ) / max(1.0, nJ))
                q = -la.solve(JtJ + lam_reg * np.eye(nJ), Jw.T @ rw, assume_a="pos")
            else:
                JJt = Jw @ Jw.T
                lam_reg = reg * (np.trace(JJt) / max(1.0, m))
                y = la.solve(JJt + lam_reg * np.eye(m), rw, assume_a="pos")
                q = -(Jw.T @ y)

        q = self.clip_correction_to_radius(p, q)
        if self._box_ctx is not None:
            x0, lb0, ub0 = self._box_ctx
            # inequality guard: don't worsen cI
            dI = model.eval_all(x_trial, components=["cI", "JI"])
            cI0, JI0 = dI.get("cI", None), dI.get("JI", None)
            if (cI0 is not None) and (JI0 is not None) and getattr(JI0, "size", 0):
                JIa = JI0.toarray() if hasattr(JI0, "toarray") else np.asarray(JI0)
                inc = JIa @ q
                mask = inc > 0.0
                if np.any(mask):
                    safe = (-np.asarray(cI0).ravel()[mask]) / inc[mask]
                    if safe.size:
                        alpha = float(0.99 * max(0.0, np.min(safe)))
                        if np.isfinite(alpha) and alpha < 1.0:
                            q = alpha * q
                        elif not np.isfinite(alpha):
                            return np.zeros(n), False, {"reason": "ineq_guard"}
            q = self._enforce_box_on_step(x_trial, q, lb0, ub0)

        th0 = model.constraint_violation(x_trial)
        th1 = model.constraint_violation(x_trial + q)
        applied = th1 < 0.9 * th0
        return (q if applied else np.zeros(n)), applied, {"theta0": th0, "theta1": th1}

    def _constrained_cauchy_in_tangent(self, H, gtilde, A_eq, Delta):
        v = self._proj(gtilde, A_eq)
        nv = self._tr_norm(v)
        if nv <= 1e-14:
            return np.zeros_like(gtilde)
        H_op = make_operator(H, gtilde.size)
        vHv = float(v @ (H_op @ v))
        if vHv <= 1e-16:
            return -(Delta / nv) * v
        return self._clip_to_radius(-(nv * nv / vHv) * v, Delta)

    # ========================= Public entry ========================= #
    def solve(
        self,
        H,
        g,
        A_ineq: Optional[np.ndarray] = None,
        b_ineq: Optional[np.ndarray] = None,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[np.ndarray] = None,
        *,
        x: Optional[np.ndarray] = None,
        lb: Optional[np.ndarray] = None,
        ub: Optional[np.ndarray] = None,
        model=None,
        mu: float = 0.0,
        filter: Optional["Filter"] = None,
        f_old: Optional[float] = None,
    ):
        n = g.size
        x, lb, ub = self._sanitize_box(x, lb, ub, n)
        self._box_ctx = (x, lb, ub) if x is not None else None

        # normalize empties / validate
        def _nz(A, b):
            if A is not None and (A.size == 0 or A.shape[0] == 0):
                return None, None
            return A, b

        A_eq, b_eq = _nz(A_eq, b_eq)
        A_ineq, b_ineq = _nz(A_ineq, b_ineq)
        if g.ndim != 1:
            raise ValueError("g must be 1D")
        if H is None:
            raise ValueError("H must be provided")
        if isinstance(H, np.ndarray) and H.shape != (n, n):
            raise ValueError(f"H must be ({n},{n})")
        if A_eq is not None:
            if A_eq.shape[1] != n or b_eq is None or b_eq.size != A_eq.shape[0]:
                raise ValueError("A_eq/b_eq mismatch")
            b_eq = np.asarray(b_eq, float).ravel()
        if A_ineq is not None:
            if A_ineq.shape[1] != n or b_ineq is None or b_ineq.size != A_ineq.shape[0]:
                raise ValueError("A_ineq/b_ineq mismatch")
            b_ineq = np.asarray(b_ineq, float).ravel()

        # subproblem
        if A_eq is not None:
            p, info, lam, nu = self._solve_with_equality_constraints(
                H, g, A_eq, b_eq, A_ineq, b_ineq
            )
        elif A_ineq is not None:
            p, info, lam, nu = self._solve_with_inequality_constraints(
                H, g, A_ineq, b_ineq
            )
        else:
            p, info = self._solve_unconstrained(H, g)
            lam, nu = np.array([]), np.array([])

        # pull trial into box if base is out-of-box
        if self._box_ctx is not None:
            x0, lb0, ub0 = self._box_ctx
            if (lb0 is not None and np.any(x0 < lb0)) or (
                ub0 is not None and np.any(x0 > ub0)
            ):
                p = (
                    np.clip(
                        x0 + p,
                        lb0 if lb0 is not None else -np.inf,
                        ub0 if ub0 is not None else +np.inf,
                    )
                    - x0
                )

        # global box for cases without equalities
        if x is not None and (A_eq is None or A_eq.size == 0):
            p = self._maybe_enforce_box(p)

        # predicted reductions
        H_op = make_operator(H, g.size)
        info["step_norm"] = self._tr_norm(p)
        info["model_reduction_quad"] = self._model_reduction(H, g, p)
        sigma_est = self._estimate_sigma(H_op, g, p)
        info["sigma_est"] = sigma_est
        info["predicted_reduction_cubic"] = self._pred_red_cubic(H, g, p, sigma_est)

        # SOC (one pass)
        soc_applied = False
        soc_info = {}
        if model is not None:
            lam_for_soc = lam if lam.size else None
            q, soc_applied, soc_meta = self._soc_correction(
                model,
                x,
                p,
                H,
                lam_for_soc,
                mu=mu,
                wE=10.0,
                wI=1.0,
                tolE=1e-8,
                violI=0.0,
                reg=self.cfg.reg_floor,
                sigma0=self.cfg.metric_shift,
            )
            if soc_applied and (q is not None).all():
                p = p + q
                info["step_norm"] = self._tr_norm(p)
                info["model_reduction_quad"] = self._model_reduction(H, g, p)
            soc_info = {"soc_applied": bool(soc_applied), **soc_meta}
        info.update(soc_info)

        # trial metrics for filter / update
        theta_trial = None
        f_trial = None
        if model is not None and hasattr(model, "eval_all"):
            x_trial = x + p if x is not None else p.copy()
            dt = model.eval_all(x_trial)
            f_trial = float(dt["f"]) if "f" in dt else None
            if hasattr(model, "constraint_violation"):
                theta_trial = float(model.constraint_violation(x_trial))
        info["f_trial"] = np.nan if f_trial is None else f_trial
        info["theta_trial"] = np.nan if theta_trial is None else theta_trial

        # actual reduction
        if f_old is None and model is not None:
            if hasattr(model, "f_current"):
                f_old = float(model.f_current)
            elif x is not None:
                d0 = model.eval_all(x)
                f_old = float(d0["f"]) if "f" in d0 else None
        act_red = (
            (f_old - f_trial) if (f_old is not None and f_trial is not None) else None
        )
        info["actual_reduction"] = np.nan if act_red is None else act_red

        # filter gate
        accepted, accepted_by = True, "no_filter"
        if filter is not None and (f_trial is not None) and (theta_trial is not None):
            if filter.is_acceptable(theta_trial, f_trial, trust_radius=self.delta):
                filter.add_if_acceptable(theta_trial, f_trial, trust_radius=self.delta)
                accepted_by = "filter"
            else:
                self.delta = max(self.cfg.delta_min, self.cfg.gamma1 * self.delta)
                accepted, accepted_by = False, "rejected_by_filter"

        # trust-region update (use cubic pred if sensible)
        if accepted and (act_red is not None):
            pred = info["predicted_reduction_cubic"]
            if not np.isfinite(pred) or abs(pred) < 1e-16:
                pred = info["model_reduction_quad"]
            _ = self.update(
                pred,
                act_red,
                info["step_norm"],
                constraint_violation=(theta_trial or 0.0),
                H=H,
                p=p,
            )
            if f_trial is not None:
                self.set_f_current(f_trial)

        if not accepted:
            s_try, ls_info = self._backtrack_on_reject(
                model, x, p, lb, ub, filter, f_old
            )
            if ls_info.get("accepted", False):
                p = s_try
                info["step_norm"] = self._tr_norm(p)
                info["accepted"] = True
                info["accepted_by"] = ls_info["accepted_by"]
                info["f_trial"] = ls_info.get("f_trial", info.get("f_trial", np.nan))
                info["theta_trial"] = ls_info.get(
                    "theta_trial", info.get("theta_trial", np.nan)
                )
                # now compute rho with this accepted alpha and update Δ normally
                if np.isfinite(info["f_trial"]) and (f_old is not None):
                    act_red = f_old - info["f_trial"]
                    pred = info["predicted_reduction_cubic"]
                    if not np.isfinite(pred) or abs(pred) < 1e-16:
                        pred = info["model_reduction_quad"]
                    _ = self.update(
                        pred,
                        act_red,
                        info["step_norm"],
                        constraint_violation=(
                            info["theta_trial"]
                            if np.isfinite(info["theta_trial"])
                            else 0.0
                        ),
                        H=H,
                        p=p,
                    )

        info["accepted"] = bool(accepted)
        info["accepted_by"] = accepted_by
        return p, info, lam, nu

    def _backtrack_on_reject(
        self, model, x, s, lb, ub, filter_obj, base_f, max_tries=3
    ):
        alphas = [0.5, 0.25, 0.125][:max_tries]
        for a in alphas:
            sa = self._enforce_box_on_step(x, a * s, lb, ub)
            xt = x + sa
            dt = model.eval_all(xt)
            f_t = float(dt.get("f", np.inf))
            theta_t = (
                float(model.constraint_violation(xt))
                if hasattr(model, "constraint_violation")
                else np.inf
            )
            # filter-based accept:
            if (filter_obj is not None) and np.isfinite(f_t) and np.isfinite(theta_t):
                if filter_obj.is_acceptable(theta_t, f_t, trust_radius=self.delta):
                    filter_obj.add_if_acceptable(theta_t, f_t, trust_radius=self.delta)
                    return sa, {
                        "accepted": True,
                        "accepted_by": "ls-filter",
                        "alpha": a,
                        "f_trial": f_t,
                        "theta_trial": theta_t,
                    }
            else:
                # simple Armijo on L1 merit if no filter
                # estimate directional derivative with current multipliers if available, else use g·s
                # Here we just use sufficient decrease wrt f as a cheap guard:
                if f_t <= base_f - 1e-4 * a * abs(
                    base_f
                ):  # very light test; tune if using true merit
                    return sa, {
                        "accepted": True,
                        "accepted_by": "ls-armijo",
                        "alpha": a,
                        "f_trial": f_t,
                        "theta_trial": theta_t,
                    }
        return s, {"accepted": False}
