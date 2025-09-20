# --- kkt_core.py (put this near your linear algebra helpers) ---
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, Tuple

# ===== Module-level: Numba helpers (safe fallbacks) ======================
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numba import njit

from .blocks.reg import make_psd_advanced
from .ip_cg import *

# Expect these base classes in your codebase:
# from .base import KKTStrategy, KKTReusable

try:
    from numba import njit

    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False


try:
    import qdldl_cpp as qd

    _HAS_QDLDL = True
except Exception:
    qd = None
    _HAS_QDLDL = False


def _csr(A, shape=None):
    if sp.isspmatrix_csr(A):
        return A
    return A.tocsr() if shape is None else sp.csr_matrix(A, shape=shape)


def _upper_csc(K: sp.spmatrix) -> sp.csc_matrix:
    # 1) keep only upper triangle (incl. diagonal)
    coo = K.tocoo()
    mask = coo.row <= coo.col
    Kup = sp.csc_matrix(
        (coo.data[mask], (coo.row[mask], coo.col[mask])), shape=coo.shape
    )

    # 2) coalesce and sort
    Kup.sum_duplicates()
    Kup.sort_indices()

    # 3) dtypes qdldl likes: float64 + int32
    if Kup.dtype != np.float64:
        Kup = Kup.astype(np.float64)
    if Kup.indptr.dtype != np.int32:
        Kup.indptr = Kup.indptr.astype(np.int32, copy=False)
    if Kup.indices.dtype != np.int32:
        Kup.indices = Kup.indices.astype(np.int32, copy=False)

    # 4) ensure every diagonal exists and is finite (tiny ridge if needed)
    d = Kup.diagonal()  # returns 1D ndarray
    bad = (~np.isfinite(d)) | (np.abs(d) < 1e-300)
    if np.any(bad):
        eps = 1e-12
        Kup = Kup + sp.diags(np.where(bad, eps, 0.0), format="csc")

    # (Optional) sanity check: no lower entries remain
    # assert sp.tril(Kup, -1).nnz == 0

    return Kup


# ---------------------- Reusable handle interface ----------------------
class KKTReusable(Protocol):
    def solve(
        self,
        r1: np.ndarray,
        r2: Optional[np.ndarray],
        cg_tol: float = 1e-8,
        cg_maxit: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray]: ...


# ---------------------- Config + Cache ----------------------
@dataclass
class KKTCache:
    # method-agnostic things we may reuse between calls
    prev_dx: Optional[np.ndarray] = None
    prev_dy: Optional[np.ndarray] = None
    delta_w_last: float = 0.0
    # strategy-specific caches if needed
    strategy_state: Dict[str, Any] = None
    prev_kkt_matrix: Optional[sp.spmatrix] = None
    prev_factorization: Optional[Any] = None
    matrix_change_tol: Optional[float] = 1e-4


# ---------------------- Abstract strategy ----------------------
class KKTStrategy:
    name: str

    def factor_and_solve(
        self,
        W: sp.spmatrix,  # n×n
        G: Optional[sp.spmatrix],  # m×n or None
        r1: np.ndarray,
        r2: Optional[np.ndarray],
        cfg: SQPConfig,
        cache: KKTCache,
        *,
        refine_iters: int = 0,
        use_ordering: bool = False,
        cg_tol: float = 1e-10,
        cg_maxit: int = 200,
        gamma: Optional[float] = None,
        delta_c_lift: Optional[float] = None,
        reuse_symbolic: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[KKTReusable]]:
        raise NotImplementedError


try:
    from numba import njit

    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        def _wrap(f):
            return f

        return _wrap


# ======================= HYKKT Strategy (Numba-enabled, fixed preconds) ==================

# ------------------------------------------------------------------------------
# Utilities & Helpers
# ------------------------------------------------------------------------------


try:
    from numba import njit

    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False


# ------------------------------------------------------------------------------
# Safe diagonal LinearOperator preconditioner
# ------------------------------------------------------------------------------
def _make_diag_prec(
    diag: Optional[np.ndarray], n: int
) -> Optional[spla.LinearOperator]:
    """
    Safe diagonal LinearOperator preconditioner.
    If diag is None -> return None.
    """
    if diag is None:
        return None
    d = np.asarray(diag, dtype=np.float64).reshape(-1)
    if d.size != n:
        raise ValueError(f"Preconditioner diagonal has size {d.size}, expected {n}")

    def mv(z):
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        return d * z

    return spla.LinearOperator((n, n), matvec=mv, rmatvec=mv, dtype=np.float64)


# ------------------------------------------------------------------------------
# Pure-Python CG (works with Python callables / LinearOperator)
# ------------------------------------------------------------------------------
def _cg_matfree_py(matvec, b, x0=None, tol=1e-10, maxit=200, M=None):
    """
    Conjugate Gradient for matrix-free matvec (Python callables allowed).
    M can be a LinearOperator or None.
    """
    b = np.asarray(b, dtype=np.float64)
    n = b.size
    x = (
        np.zeros(n, dtype=np.float64)
        if x0 is None
        else np.array(x0, dtype=np.float64, copy=True)
    )

    r = b - matvec(x)
    z = r if M is None else (M @ r)
    p = z.copy()
    rz_old = float(r @ z)
    nrm0 = float(np.linalg.norm(r))
    stop = max(tol * nrm0, 0.0)
    if nrm0 <= stop:
        return x, 0

    for it in range(1, maxit + 1):
        Ap = matvec(p)
        pAp = float(p @ Ap)
        if pAp <= 0.0:
            break
        alpha = rz_old / pAp
        x += alpha * p
        r -= alpha * Ap
        if float(np.linalg.norm(r)) <= stop:
            return x, it
        z = r if M is None else (M @ r)
        rz_new = float(r @ z)
        beta = rz_new / rz_old
        rz_old = rz_new
        p = z + beta * p
    return x, it


# ------------------------------------------------------------------------------
# Numba CSR kernels & CG for Kγ = W + δI + γ GᵀG (no Python callables in-loop)
# ------------------------------------------------------------------------------
if _HAS_NUMBA:

    @njit(cache=True, fastmath=True)
    def _csr_mv(indptr, indices, data, x, out):
        out[:] = 0.0
        n = indptr.size - 1
        for i in range(n):
            s = 0.0
            rs = indptr[i]
            re = indptr[i + 1]
            for k in range(rs, re):
                s += data[k] * x[indices[k]]
            out[i] = s

    @njit(cache=True, fastmath=True)
    def _kgamma_mv_into(
        W_indptr, W_indices, W_data, G_indptr, G_indices, G_data, x, delta, gamma, out
    ):
        # out = (W + δI + γ GᵀG) x
        n = out.size
        # out := W x
        _csr_mv(W_indptr, W_indices, W_data, x, out)
        # + δ x
        if delta != 0.0:
            for i in range(n):
                out[i] += delta * x[i]
        # tmp := G x
        m = G_indptr.size - 1
        tmp = np.empty(m, dtype=np.float64)
        _csr_mv(G_indptr, G_indices, G_data, x, tmp)
        # out += γ Gᵀ tmp
        if gamma != 0.0:
            for i in range(m):
                ti = gamma * tmp[i]
                rs = G_indptr[i]
                re = G_indptr[i + 1]
                for k in range(rs, re):
                    out[G_indices[k]] += G_data[k] * ti

    # ---------- SSOR preconditioner (on W + diag) ----------
    @njit(cache=True, fastmath=True)
    def _ssor_apply_W_diag(W_indptr, W_indices, W_data, Kdiag, omega, sweeps, rhs, out):
        """
        out := M^{-1} rhs where M is SSOR(tilde K), tilde K ≈ W + diag(Kdiag).
        Two GS sweeps per 'sweeps' (forward, then backward). In-place write to 'out'.
        """
        n = rhs.size
        for i in range(n):
            out[i] = 0.0

        for _ in range(sweeps):
            # forward
            for i in range(n):
                rs = W_indptr[i]
                re = W_indptr[i + 1]
                sumLU = 0.0
                for k in range(rs, re):
                    j = W_indices[k]
                    aij = W_data[k]
                    if j != i:
                        sumLU += aij * out[j]
                out[i] = (1.0 - omega) * out[i] + omega * (rhs[i] - sumLU) / Kdiag[i]
            # backward
            for ii in range(n):
                i = n - 1 - ii
                rs = W_indptr[i]
                re = W_indptr[i + 1]
                sumLU = 0.0
                for k in range(rs, re):
                    j = W_indices[k]
                    aij = W_data[k]
                    if j != i:
                        sumLU += aij * out[j]
                out[i] = (1.0 - omega) * out[i] + omega * (rhs[i] - sumLU) / Kdiag[i]

    @njit(cache=True, fastmath=True)
    def _cg_Kgamma_numba(
        W_indptr,
        W_indices,
        W_data,
        G_indptr,
        G_indices,
        G_data,
        b,
        x,
        tol,
        maxit,
        Minv_diag,
        use_prec,
        delta,
        gamma,
        use_ssor,
        ssor_omega,
        ssor_sweeps,
        Kdiag_full,
    ):
        """
        CG specialized for Kγ x = b with Kγ = W + δI + γ GᵀG (all CSR buffers).
        - b, x are float64 arrays
        - Minv_diag is diag(M^{-1}) for Jacobi; use_prec toggles it
        - If use_ssor, apply SSOR on tilde K ≈ W + diag(Kdiag_full)
        """
        n = b.size
        r = np.empty(n, dtype=np.float64)
        z = np.empty(n, dtype=np.float64)
        p = np.empty(n, dtype=np.float64)
        Ap = np.empty(n, dtype=np.float64)

        # r := b - Kγ x
        _kgamma_mv_into(
            W_indptr,
            W_indices,
            W_data,
            G_indptr,
            G_indices,
            G_data,
            x,
            delta,
            gamma,
            Ap,
        )
        for i in range(n):
            r[i] = b[i] - Ap[i]

        # z := M^{-1} r
        if use_prec:
            if use_ssor:
                _ssor_apply_W_diag(
                    W_indptr,
                    W_indices,
                    W_data,
                    Kdiag_full,
                    ssor_omega,
                    ssor_sweeps,
                    r,
                    z,
                )
            else:
                for i in range(n):
                    z[i] = r[i] * Minv_diag[i]
        else:
            for i in range(n):
                z[i] = r[i]

        for i in range(n):
            p[i] = z[i]

        rz_old = 0.0
        nrm0 = 0.0
        for i in range(n):
            rz_old += r[i] * z[i]
            nrm0 += r[i] * r[i]
        nrm0 = np.sqrt(nrm0)
        stop = tol * nrm0
        if nrm0 <= stop:
            return 0

        for it in range(1, maxit + 1):
            _kgamma_mv_into(
                W_indptr,
                W_indices,
                W_data,
                G_indptr,
                G_indices,
                G_data,
                p,
                delta,
                gamma,
                Ap,
            )
            pAp = 0.0
            for i in range(n):
                pAp += p[i] * Ap[i]
            if pAp <= 0.0:
                return it - 1

            alpha = rz_old / pAp
            nr = 0.0
            for i in range(n):
                x[i] = x[i] + alpha * p[i]
                r[i] = r[i] - alpha * Ap[i]
                nr += r[i] * r[i]
            nr = np.sqrt(nr)
            if nr <= stop:
                return it

            if use_prec:
                if use_ssor:
                    _ssor_apply_W_diag(
                        W_indptr,
                        W_indices,
                        W_data,
                        Kdiag_full,
                        ssor_omega,
                        ssor_sweeps,
                        r,
                        z,
                    )
                else:
                    for i in range(n):
                        z[i] = r[i] * Minv_diag[i]
            else:
                for i in range(n):
                    z[i] = r[i]

            rz_new = 0.0
            for i in range(n):
                rz_new += r[i] * z[i]
            beta = rz_new / rz_old
            rz_old = rz_new

            for i in range(n):
                p[i] = z[i] + beta * p[i]
        return maxit


# ------------------------------------------------------------------------------
# Hutchinson diag(K^{-1}) via a matrix-free inner solve
# ------------------------------------------------------------------------------
def hutch_diag_Kinv_via_inner(
    inner_solve, n: int, probes: int = 8, clip: float = 1e-12, seed: int = 12345
):
    """
    Estimate diag(K^{-1}) ≈ E[(K^{-1} z) ⊙ z], z in {±1}^n
    Uses provided `inner_solve(rhs) -> x` (e.g., numba K-CG).
    """
    if probes <= 0:
        return None
    rng = np.random.default_rng(seed)
    acc = np.zeros(n, dtype=float)
    # Running variance (kept for potential future adaptive stopping)
    mean = np.zeros(n, dtype=float)
    m2 = np.zeros(n, dtype=float)
    for t in range(1, probes + 1):
        z = rng.integers(0, 2, size=n, dtype=np.int8).astype(float)
        z[z == 0] = -1.0
        v = inner_solve(z)
        s = v * z
        acc += s
        # Welford update
        delta = s - mean
        mean += delta / t
        m2 += delta * (s - mean)
    d = acc / float(probes)
    d[~np.isfinite(d)] = clip
    return np.maximum(d, clip)


# ------------------------------------------------------------------------------
# HYKKT Strategy (expects KKTStrategy, KKTReusable to be defined elsewhere)
# ------------------------------------------------------------------------------
class HYKKTStrategy(KKTStrategy):
    """
    HYKKT Schur strategy (fast version):
      • CHOLMOD path with sticky exact Schur for small m
      • Matrix-free CG path for K_gamma using Numba CSR kernels + SSOR option
      • Improved Schur preconditioning via Hutchinson-refined Jacobi
      • Adaptive gamma control and inexact-Newton forcing for CG
      • Aggressive reuse of factorizations & preconditioners
    """

    name = "hykkt"

    def __init__(self, cholmod_loader: Optional[Callable[[], Any]] = None):
        self._cholmod_loader = cholmod_loader
        # Cache
        self._cache_key = None
        self._cholK = None
        self._cholS = None
        self._S_explicit = None
        self._last_K_pattern = None
        self._last_delta = None
        self._last_gamma = None

        # Soft state for adaptivity across calls
        self._last_cg_iters = None
        self._last_gamma_user = None

        # Optional ILU cache (matrix pattern keyed)
        self._ilu_key = None
        self._ilu = None

    # ---------- utilities ----------
    @staticmethod
    def _normest_rowsum_inf(A: sp.spmatrix) -> float:
        A = A.tocsr() if not sp.isspmatrix_csr(A) else A
        if A.shape[0] == 0:
            return 0.0
        absA = A.copy()
        absA.data = np.abs(absA.data)
        rs = np.add.reduceat(absA.data, absA.indptr[:-1])
        return float(rs.max()) if rs.size else 0.0

    @staticmethod
    def _diag_of_GtG(G: sp.spmatrix) -> np.ndarray:
        return np.array((G.multiply(G)).sum(axis=0)).ravel().astype(float)

    @staticmethod
    def _forcing_tol(
        base_tol: float, mu: Optional[float], phase: Optional[str]
    ) -> float:
        # Inexact-Newton forcing (looser early, tighter late)
        if mu is not None:
            return max(base_tol, min(0.5, math.sqrt(max(mu, 0.0))) * base_tol)
        if phase == "early":
            return max(base_tol, 1e-2)
        if phase == "mid":
            return max(base_tol, 3e-3)
        if phase == "late":
            return base_tol
        return base_tol

    @staticmethod
    def _clip_pos(x: np.ndarray, eps: float) -> np.ndarray:
        y = np.asarray(x, dtype=float).copy()
        y[~np.isfinite(y)] = eps
        return np.maximum(y, eps)

    def _hutch_diag_Kinv(
        self, cholK, n: int, probes: int, clip: float
    ) -> Optional[np.ndarray]:
        """Hutchinson diagonal estimate for K^{-1}: mean((K^{-1} z) ⊙ z), z ∈ {±1}^n."""
        if probes <= 0 or cholK is None:
            return None
        acc = np.zeros(n, dtype=float)
        rng = np.random.default_rng(12345)
        for _ in range(probes):
            z = rng.integers(0, 2, size=n, dtype=np.int8).astype(float)
            z[z == 0] = -1.0
            v = cholK.solve_A(z)
            acc += v * z
        acc /= float(probes)
        return self._clip_pos(acc, clip)

    # ---------- gamma adaptation ----------
    def _adapt_gamma(
        self,
        gamma: float,
        cg_iters: Optional[int],
        target: int,
        bounds: Tuple[float, float],
        inc: float,
        dec: float,
    ) -> float:
        if cg_iters is None:
            return gamma
        gmin, gmax = bounds
        if cg_iters > max(10, target):
            gamma = min(gamma * inc, gmax)
        elif cg_iters < max(5, target // 2):
            gamma = max(gamma * dec, gmin)
        return gamma

    # ---------- main entry ----------
    def factor_and_solve(self, W, G, r1, r2, cfg, reg, cache, **kw):
        assert G is not None and r2 is not None, "HYKKT requires equality constraints"

        # sizes & formats
        W = W.tocsr() if not sp.isspmatrix_csr(W) else W
        G = G.tocsr() if not sp.isspmatrix_csr(G) else G
        n = int(W.shape[0])
        m, p = G.shape
        assert p == n

        # knobs (existing)
        cg_tol = float(kw.get("cg_tol", 1e-10))
        cg_maxit = int(kw.get("cg_maxit", 200))
        cholmod_factor = bool(kw.get("cholmod_factor", True))
        assemble_schur = bool(kw.get("assemble_schur_if_m_small", True))
        schur_cut = float(kw.get("schur_dense_cutoff", 0.25))
        jacobi_schur_prec = bool(kw.get("jacobi_schur_prec", True))
        use_ilu_prec = bool(kw.get("use_ilu_prec", False))
        ilu_drop_tol = float(kw.get("ilu_drop_tol", 0.0))
        ilu_fill = float(kw.get("ilu_fill", 1.0))

        # new knobs
        adaptive_gamma = bool(kw.get("adaptive_gamma", True))
        target_cg = int(kw.get("target_cg", max(10, min(50, m // 5 if m > 0 else 10))))
        gamma_bounds = tuple(kw.get("gamma_bounds", (1e-3, 1e6)))
        gamma_increase = float(kw.get("gamma_increase", 3.0))
        gamma_decrease = float(kw.get("gamma_decrease", 0.5))
        mu = kw.get("mu", None)
        phase = kw.get("phase", None)
        hutch_probes = int(kw.get("hutch_probes", 0))
        hutch_clip = float(kw.get("hutch_clip", 1e-12))
        sticky_schur = bool(kw.get("sticky_schur", True))
        schur_reuse_tol = float(kw.get("schur_reuse_tol", 0.0))
        save_cg_stats = bool(kw.get("save_cg_stats", False))

        # shifts
        delta = float(cache.delta_w_last or 0.0)

        # gamma heuristic if not provided
        user_gamma = kw.get("gamma", None)
        if user_gamma is None:
            num = self._normest_rowsum_inf(W) + delta
            den = max(self._normest_rowsum_inf(G), 1.0)
            gamma = max(1.0, num / (den * den))
        else:
            gamma = float(user_gamma)
            self._last_gamma_user = gamma
        gamma = float(np.clip(gamma, gamma_bounds[0], gamma_bounds[1]))

        # cache keys
        pattern_key = (W.nnz, G.nnz, W.shape, G.shape)
        K_key = (pattern_key, delta, gamma)

        # Try CHOLMOD
        cholmod = None
        # if cholmod_factor:
        #     try:
        #         cholmod = self._cholmod_loader() if self._cholmod_loader else __import__("sksparse.cholmod", fromlist=["cholmod"])
        #     except Exception:
        #         cholmod = None

        # --- PATH A: CHOLMOD factorization of K_gamma (+ optional exact Schur) ---
        if cholmod is not None:
            if (self._cholK is None) or (K_key != self._cache_key):
                # K = W + delta*I + gamma * (G^T @ G)
                K = W.copy().tocsr()
                if delta != 0.0:
                    K = (K + delta * sp.eye(n, format="csr")).tocsr()
                if gamma != 0.0:
                    GtG = (G.T @ G).tocsr()
                    K = (K + gamma * GtG).tocsr()

                self._cholK = cholmod.cholesky(K.tocsc())
                self._cache_key = K_key
                self._last_K_pattern = pattern_key
                self._last_delta = delta
                self._last_gamma = gamma
                # Reset Schur cache if not sticky
                if not sticky_schur:
                    self._cholS = None
                    self._S_explicit = None
                K_diag = K.diagonal().astype(float)
            else:
                # approximate diag(K) cheaply without rebuilding
                K_diag = (
                    W.diagonal().astype(float) + delta + gamma * self._diag_of_GtG(G)
                )

            cholK = self._cholK

            # Right-hand sides for Schur
            svec = r1 + gamma * (G.T @ r2)
            Kinvsvec = cholK.solve_A(svec)
            rhs_s = (G @ Kinvsvec) - r2

            # Exact Schur for small m (sticky)
            use_exact_S = assemble_schur and (m <= max(1, int(schur_cut * n)))
            need_S_build = False
            if use_exact_S:
                if self._cholS is None:
                    need_S_build = True
                elif not sticky_schur:
                    need_S_build = True
                elif sticky_schur and schur_reuse_tol > 0.0:
                    g_rel = abs(
                        (gamma - (self._last_gamma or gamma)) / max(1.0, abs(gamma))
                    )
                    need_S_build = g_rel > schur_reuse_tol
            if use_exact_S and need_S_build:
                GT_dense = np.asfortranarray(G.T.toarray())  # (n, m)
                Z = cholK.solve_A(GT_dense)  # (n, m)
                S_dense = G @ Z  # (m, m)
                S = sp.csc_matrix(np.asarray(S_dense))
                self._S_explicit = S
                self._cholS = cholmod.cholesky(S)

            if self._cholS is not None:
                # Direct solve with exact Schur
                dy = self._cholS.solve_A(rhs_s)
                dx = cholK.solve_A(svec - (G.T @ dy))
                cg_iters = 0
            else:
                # Schur CG with improved Jacobi precond
                diagK = np.maximum(K_diag, 1e-12)
                inv_diagK = 1.0 / diagK

                # Optional Hutchinson refinement for diag(K^{-1})
                diag_Kinv = None
                if hutch_probes > 0:
                    diag_Kinv = self._hutch_diag_Kinv(
                        cholK, n, hutch_probes, hutch_clip
                    )
                if diag_Kinv is None:
                    diag_Kinv = inv_diagK  # fallback

                # diag( S ) ≈ (G.^2) @ diag(K^{-1})
                Sdiag_hat = np.asarray(
                    (G.multiply(G)).dot(diag_Kinv), dtype=float
                ).ravel()
                Sinv_diag = (
                    1.0 / np.maximum(Sdiag_hat, 1e-12) if jacobi_schur_prec else None
                )
                M = _make_diag_prec(Sinv_diag, m) if jacobi_schur_prec else None

                def S_mv(y: np.ndarray) -> np.ndarray:
                    return G @ cholK.solve_A(G.T @ y)

                # Forcing strategy on Schur tolerance
                schur_tol = self._forcing_tol(cg_tol, mu, phase)
                x0 = cache.prev_dy if cache.prev_dy is not None else None

                # Python CG here (cholK.solve_A is a Python method)
                dy, cg_iters = _cg_matfree_py(
                    S_mv, rhs_s, x0=x0, tol=schur_tol, maxit=cg_maxit, M=M
                )
                dx = cholK.solve_A(svec - (G.T @ dy))

            # Adaptive gamma update (for next call)
            if adaptive_gamma:
                new_gamma = self._adapt_gamma(
                    gamma,
                    cg_iters,
                    target_cg,
                    gamma_bounds,
                    gamma_increase,
                    gamma_decrease,
                )
                self._last_gamma = new_gamma  # hint next time

            # Save stats and warm-starts
            if save_cg_stats:
                cache.hykkt_stats = {
                    "schur_cg_iters": cg_iters,
                    "m": m,
                    "n": n,
                    "gamma": gamma,
                }
            cache.prev_dx, cache.prev_dy = dx, dy
            self._last_cg_iters = cg_iters

            # Reusable solver
            if self._cholS is not None:

                class _Reusable(KKTReusable):
                    def __init__(self, cholK, cholS, G, gamma):
                        self.cholK = cholK
                        self.cholS = cholS
                        self.G = G
                        self.gamma = float(gamma)

                    def solve(self, r1n, r2n, cg_tol=1e-8, cg_maxit=200):
                        svec_n = r1n + self.gamma * (self.G.T @ r2n)
                        rhs_s_n = (self.G @ self.cholK.solve_A(svec_n)) - r2n
                        dyn = self.cholS.solve_A(rhs_s_n)
                        dxn = self.cholK.solve_A(svec_n - (self.G.T @ dyn))
                        return dxn, dyn

                return dx, dy, _Reusable(cholK, self._cholS, G, gamma)
            else:

                class _Reusable(KKTReusable):
                    def __init__(self, cholK, G, gamma, Sinv_diag):
                        self.cholK = cholK
                        self.G = G
                        self.gamma = float(gamma)
                        self.Sinv_diag = Sinv_diag

                    def solve(self, r1n, r2n, cg_tol=1e-8, cg_maxit=200):
                        svec_n = r1n + self.gamma * (self.G.T @ r2n)
                        rhs_s_n = (self.G @ self.cholK.solve_A(svec_n)) - r2n

                        def S_mv_n(y):
                            return self.G @ self.cholK.solve_A(self.G.T @ y)

                        M_n = (
                            _make_diag_prec(self.Sinv_diag, self.G.shape[0])
                            if self.Sinv_diag is not None
                            else None
                        )
                        dyn, _ = _cg_matfree_py(
                            S_mv_n, rhs_s_n, x0=None, tol=cg_tol, maxit=cg_maxit, M=M_n
                        )
                        dxn = self.cholK.solve_A(svec_n - (self.G.T @ dyn))
                        return dxn, dyn

                return (
                    dx,
                    dy,
                    _Reusable(
                        cholK, G, gamma, Sinv_diag if jacobi_schur_prec else None
                    ),
                )

        # --- PATH B: Matrix-free CG for K_gamma (Numba-accelerated) -----
        # Build Kγ diag and its inverse
        diagW = np.asarray(W.diagonal(), dtype=np.float64)
        diagGtG = self._diag_of_GtG(G)
        Kdiag = np.maximum(diagW + delta + gamma * diagGtG, 1e-12)
        Minv_diag = 1.0 / Kdiag
        use_prec = True

        # SSOR knobs
        use_ssor = bool(kw.get("kgamma_use_ssor", True))
        ssor_omega = float(kw.get("kgamma_ssor_omega", 1.4))  # 1.2–1.6 often good
        ssor_sweeps = int(kw.get("kgamma_ssor_sweeps", 1))  # 1 sweep default

        # Prepare CSR arrays for Numba
        W_csr = W.tocsr()
        G_csr = G.tocsr()
        W_indptr = W_csr.indptr.astype(np.int64)
        W_indices = W_csr.indices.astype(np.int64)
        W_data = W_csr.data.astype(np.float64)
        G_indptr = G_csr.indptr.astype(np.int64)
        G_indices = G_csr.indices.astype(np.int64)
        G_data = G_csr.data.astype(np.float64)

        # Forcing on inner K-solve tolerance
        inner_tol = self._forcing_tol(max(1e-2 * cg_tol, 1e-12), mu, phase)
        inner_maxit = max(10 * cg_maxit, 200)

        def inner_solve(q: np.ndarray) -> np.ndarray:
            q = np.asarray(q, dtype=np.float64)
            x = np.zeros_like(q, dtype=np.float64)
            if _HAS_NUMBA:
                _ = _cg_Kgamma_numba(
                    W_indptr,
                    W_indices,
                    W_data,
                    G_indptr,
                    G_indices,
                    G_data,
                    q,
                    x,
                    float(inner_tol),
                    int(inner_maxit),
                    Minv_diag,
                    use_prec,
                    float(delta),
                    float(gamma),
                    use_ssor,
                    ssor_omega,
                    ssor_sweeps,
                    Kdiag,
                )
                return x
            else:
                # Fallback to Python CG if Numba unavailable
                def K_mv_py(v):
                    return (W @ v) + delta * v + gamma * (G.T @ (G @ v))

                M_K_py = _make_diag_prec(Minv_diag, n)
                x_py, _ = _cg_matfree_py(
                    K_mv_py, q, x0=None, tol=inner_tol, maxit=inner_maxit, M=M_K_py
                )
                return x_py

        # Schur operator & rhs
        svec = r1 + gamma * (G.T @ r2)
        rhs_s = (G @ inner_solve(svec)) - r2

        # Jacobi preconditioner for Schur: diag(Ŝ) using Hutchinson-refined diag(K^{-1})
        if jacobi_schur_prec:
            diag_Kinv = hutch_diag_Kinv_via_inner(
                inner_solve,
                n,
                probes=max(0, hutch_probes),
                clip=hutch_clip,
                seed=kw.get("hutch_seed", 12345),
            )
            if diag_Kinv is None:
                diag_Kinv = 1.0 / Kdiag  # fallback
            Sdiag_hat = np.asarray((G.multiply(G)).dot(diag_Kinv), dtype=float).ravel()
            Sinv_diag = 1.0 / np.maximum(Sdiag_hat, 1e-12)
            M_S = _make_diag_prec(Sinv_diag, m)
        else:
            Sinv_diag = None
            M_S = None

        # Forcing on Schur tolerance
        schur_tol = self._forcing_tol(cg_tol, mu, phase)
        x0 = cache.prev_dy if cache.prev_dy is not None else None

        def S_mv(y: np.ndarray) -> np.ndarray:
            return G @ inner_solve(G.T @ y)

        # Outer Schur CG in Python (kept as-is)
        dy, cg_iters = _cg_matfree_py(
            S_mv, rhs_s, x0=x0, tol=schur_tol, maxit=cg_maxit, M=M_S
        )
        dx = inner_solve(svec - (G.T @ dy))

        # Adaptive gamma update (for next calls)
        if adaptive_gamma:
            new_gamma = self._adapt_gamma(
                gamma, cg_iters, target_cg, gamma_bounds, gamma_increase, gamma_decrease
            )
            self._last_gamma = new_gamma

        if save_cg_stats:
            cache.hykkt_stats = {
                "schur_cg_iters": cg_iters,
                "m": m,
                "n": n,
                "gamma": gamma,
                "path": "numba",
            }

        cache.prev_dx, cache.prev_dy = dx, dy
        self._last_cg_iters = cg_iters

        class _Reusable(KKTReusable):
            def __init__(self, G, inner_solve, gamma, Sinv_diag):
                self.G = G
                self.inner_solve = inner_solve
                self.gamma = float(gamma)
                self.Sinv_diag = Sinv_diag

            def solve(self, r1n, r2n, cg_tol=1e-8, cg_maxit=200):
                svec_n = r1n + self.gamma * (self.G.T @ r2n)
                rhs_s_n = (self.G @ self.inner_solve(svec_n)) - r2n

                def S_mv_n(y):
                    return self.G @ self.inner_solve(self.G.T @ y)

                M_n = (
                    _make_diag_prec(self.Sinv_diag, self.G.shape[0])
                    if self.Sinv_diag is not None
                    else None
                )
                dyn, _ = _cg_matfree_py(
                    S_mv_n, rhs_s_n, x0=None, tol=cg_tol, maxit=cg_maxit, M=M_n
                )
                dxn = self.inner_solve(svec_n - (self.G.T @ dyn))
                return dxn, dyn

        return dx, dy, _Reusable(G, inner_solve, gamma, Sinv_diag)


# ---------------------- Indefinite LDLᵀ ----------------------
class LDLStrategy(KKTStrategy):
    name = "ldl"  # covers 'qdldl' and 'lldl' choices

    def __init__(self, prefer_lldl: bool = False):
        self.prefer_lldl = prefer_lldl

    def factor_and_solve(self, W, G, r1, r2, cfg, regularizer, cache, **kw):
        refine_iters = kw.get("refine_iters", 0)
        method_req = kw.get("method", "qdldl")
        n = W.shape[0]
        mE = 0 if G is None else G.shape[0]

        delta_w_last = float(cache.delta_w_last or 0.0)
        delta_w = 0.0
        delta_c = 0.0
        max_attempts = 10

        def _build_K():
            if mE > 0:
                Wcsr = _csr(W + delta_w * sp.eye(n))
                B22 = -delta_c * sp.eye(mE) if delta_c > 0 else sp.csr_matrix((mE, mE))
                K = sp.vstack(
                    [sp.hstack([Wcsr, G.T]), sp.hstack([G, B22])], format="csc"
                )
                rhs = np.concatenate([r1, r2])
            else:
                K = _csr(W + delta_w * sp.eye(n))
                rhs = r1
            return K, rhs

        for attempt in range(max_attempts):
            K, rhs = _build_K()
            K, _ = make_psd_advanced(K, regularizer, attempt)
            K_up = _upper_csc(K)
            nsys = K_up.shape[0]
            perm = None  # (plug AMD if you want)

            try:
                used = None
                if (self.prefer_lldl or method_req == "lldl") and globals().get(
                    "_HAS_LLDL", False
                ):
                    fac = lldl_factorize(K_up, perm=perm)
                    D = fac.D
                    used = "lldl"

                    def do_solve(b):
                        return fac.solve(b)

                else:
                    if not globals().get("_HAS_QDLDL", False):
                        raise RuntimeError("qdldl unavailable")
                    fac = qd.factorize(
                        K_up.indptr, K_up.indices, K_up.data, nsys, perm=perm
                    )
                    D = fac.D
                    used = "qdldl"

                    def do_solve(b):
                        return (
                            qd.solve_refine(fac, b, refine_iters)
                            if refine_iters > 0
                            else qd.solve(fac, b)
                        )

                num_pos = int(np.sum(D > 0))
                num_neg = int(np.sum(D < 0))
                num_zero = int(np.sum(np.abs(D) < 1e-12))

                if mE > 0:
                    ok = num_pos == n and num_neg == mE and num_zero == 0
                else:
                    ok = num_pos == n and num_neg == 0 and num_zero == 0

                if ok:
                    sol = do_solve(rhs)
                    if mE > 0:
                        dx, dy = sol[:n], sol[n:]
                    else:
                        dx, dy = sol, np.zeros(0)

                    class _Reusable(KKTReusable):
                        def __init__(self, fac, used, n, mE):
                            self.fac, self.used, self.n, self.mE = fac, used, n, mE

                        def solve(self, r1n, r2n=None, **_):
                            if self.used == "qdldl":
                                rhsn = (
                                    r1n if self.mE == 0 else np.concatenate([r1n, r2n])
                                )
                                soln = qd.solve(self.fac, rhsn)
                            else:
                                rhsn = (
                                    r1n if self.mE == 0 else np.concatenate([r1n, r2n])
                                )
                                soln = self.fac.solve(rhsn)
                            return (
                                (soln[: self.n], soln[self.n :])
                                if self.mE > 0
                                else (soln, np.zeros(0))
                            )

                    cache.delta_w_last = delta_w
                    cache.prev_dx, cache.prev_dy = dx, dy
                    return dx, dy, _Reusable(fac, used, n, mE)

                # adjust regs
                if num_zero > 0 or (mE > 0 and num_neg != mE):
                    delta_c = max(delta_c, 1e-8)
                    if delta_w_last == 0:
                        delta_w = (
                            cfg.ip_hess_reg0
                            if attempt == 0
                            else 8.0 * max(cfg.ip_hess_reg0, delta_w)
                        )
                    else:
                        delta_w = (
                            max(1e-20, 1 / 3 * delta_w_last)
                            if attempt == 0
                            else 8.0 * max(1e-20, delta_w)
                        )
                    if delta_w > 1e40:
                        raise ValueError("Inertia correction failed")
            except Exception:
                # Sparse fallback
                sol = spla.spsolve(K.tocsc(), rhs)
                if mE > 0:
                    dx, dy = sol[:n], sol[n:]
                else:
                    dx, dy = sol, np.zeros(0)
                return dx, dy, None

        # give up
        return np.zeros(n), np.zeros(mE), None


# ---------------------- Registry and dispatcher ----------------------
class KKTSolverRegistry:
    def __init__(self):
        self._map: Dict[str, KKTStrategy] = {}

    def register(self, strategy: KKTStrategy):
        self._map[strategy.name] = strategy

    def get(self, name: str) -> KKTStrategy:
        if name not in self._map:
            raise KeyError(f"Unknown KKT method '{name}'")
        return self._map[name]


# Build a default registry you can import and reuse
def _load_cholmod():
    import sksparse.cholmod as cholmod

    return cholmod


DEFAULT_KKT_REGISTRY = KKTSolverRegistry()
DEFAULT_KKT_REGISTRY.register(HYKKTStrategy(cholmod_loader=_load_cholmod))
DEFAULT_KKT_REGISTRY.register(LDLStrategy(prefer_lldl=False))  # name "ldl"

# Optional aliases if you want:
DEFAULT_KKT_REGISTRY._map["qdldl"] = DEFAULT_KKT_REGISTRY._map["ldl"]
DEFAULT_KKT_REGISTRY._map["lldl"] = LDLStrategy(prefer_lldl=True)
