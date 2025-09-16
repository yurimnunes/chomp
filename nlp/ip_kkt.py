# --- kkt_core.py (put this near your linear algebra helpers) ---
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, Tuple

# ===== Module-level: Numba helpers (safe fallbacks) ======================
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .blocks.reg import Regularizer, make_preconditioner_only, make_psd_advanced
from .ip_cg import *

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


def _normest_rowsum_inf(A: sp.spmatrix) -> float:
    A = _csr(A)
    return float(np.max(np.abs(A).sum(axis=1))) if A.shape[0] else 1.0


def _cg_matfree(matvec, b, x0=None, tol=1e-10, maxit=200, M=None):
    n = b.size
    x = np.zeros(n) if x0 is None else x0.copy()
    r = b - matvec(x)
    z = r if M is None else M(r)
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
        nr = float(np.linalg.norm(r))
        if nr <= stop:
            return x, it
        z = r if M is None else M(r)
        rz_new = float(r @ z)
        beta = rz_new / rz_old
        rz_old = rz_new
        p = z + beta * p
    return x, it


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
        def _wrap(f): return f
        return _wrap

@njit(cache=True, fastmath=True)
def _csr_matvec_numba(indptr, indices, data, x):
    n = indptr.size - 1
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        acc = 0.0
        row_start = indptr[i]
        row_end = indptr[i + 1]
        for k in range(row_start, row_end):
            acc += data[k] * x[indices[k]]
        y[i] = acc
    return y

@njit(cache=True, fastmath=True)
def _csr_t_matvec_numba(indptr, indices, data, x, ncols):
    # y = (A^T) @ x, A is CSR with shape (nrows, ncols)
    y = np.zeros(ncols, dtype=np.float64)
    nrows = indptr.size - 1
    for i in range(nrows):
        xi = x[i]
        row_start = indptr[i]
        row_end = indptr[i + 1]
        for k in range(row_start, row_end):
            j = indices[k]
            y[j] += data[k] * xi
    return y

@njit(cache=True, fastmath=True)
def _kgamma_mv_numba(W_indptr, W_indices, W_data,
                     G_indptr, G_indices, G_data,
                     x, delta, gamma, n):
    # yW = W @ x
    yW = _csr_matvec_numba(W_indptr, W_indices, W_data, x)
    # yG = G @ x
    yG = _csr_matvec_numba(G_indptr, G_indices, G_data, x)
    # gty = G^T @ (G @ x)
    gty = _csr_t_matvec_numba(G_indptr, G_indices, G_data, yG, n)
    # combine: W x + delta x + gamma gty
    return yW + delta * x + gamma * gty


# ======================= HYKKT Strategy (Numba-enabled) ==================
class HYKKTStrategy(KKTStrategy):
    """
    HYKKT Schur strategy (fast version):
      • CHOLMOD path with sticky exact Schur for small m
      • Matrix-free CG path for K_gamma using Numba CSR kernels
      • Improved Schur preconditioning via Hutchinson-refined Jacobi
      • Adaptive gamma control and inexact-Newton forcing for CG
      • Aggressive reuse of factorizations & preconditioners

    Extra Tunables (kw in addition to previous):
      - adaptive_gamma: bool (default True)
      - target_cg: int (default: min(50, max(10, m//5)))   # desired Schur-CG iters
      - gamma_bounds: (gmin, gmax) (default (1e-3, 1e6))
      - gamma_increase: float (default 3.0)                # factor when CG is slow
      - gamma_decrease: float (default 0.5)                # factor when CG is very fast
      - mu: float or None (default None)                   # barrier/merit; enables forcing
      - phase: str in {"early","mid","late"} or None       # coarse forcing hint
      - hutch_probes: int (default 0 => disabled; try 8–16)
      - hutch_clip: float (default 1e-12)                  # avoid tiny/neg diagonals
      - sticky_schur: bool (default True)
      - schur_reuse_tol: float (default 0.0)               # if >0, allow γ drift w/out rebuilding S
      - save_cg_stats: bool (default False)                # export stats to cache
    """
    name = "hykkt"

    def __init__(self, cholmod_loader: Optional[Callable[[], Any]] = None):
        self._cholmod_loader = cholmod_loader
        # Cache
        self._cache_key = None          # (pattern_key, delta, gamma)
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
    def _matfree_Kgam_mv_python(W: sp.csr_matrix, G: sp.csr_matrix, x: np.ndarray, delta: float, gamma: float) -> np.ndarray:
        return (W @ x) + delta * x + gamma * (G.T @ (G @ x))

    @staticmethod
    def _make_numba_K_mv(W: sp.csr_matrix, G: sp.csr_matrix, delta: float, gamma: float):
        if _HAS_NUMBA:
            W_csr = W.tocsr() if not sp.isspmatrix_csr(W) else W
            G_csr = G.tocsr() if not sp.isspmatrix_csr(G) else G

            W_indptr = np.asarray(W_csr.indptr, dtype=np.int64)
            W_indices = np.asarray(W_csr.indices, dtype=np.int64)
            W_data = np.asarray(W_csr.data, dtype=np.float64)

            G_indptr = np.asarray(G_csr.indptr, dtype=np.int64)
            G_indices = np.asarray(G_csr.indices, dtype=np.int64)
            G_data = np.asarray(G_csr.data, dtype=np.float64)

            n = W_csr.shape[0]

            def mv(x: np.ndarray) -> np.ndarray:
                x64 = np.asarray(x, dtype=np.float64)
                return _kgamma_mv_numba(W_indptr, W_indices, W_data,
                                        G_indptr, G_indices, G_data,
                                        x64, float(delta), float(gamma), int(n))
            return mv
        else:
            def mv(x: np.ndarray) -> np.ndarray:
                return HYKKTStrategy._matfree_Kgam_mv_python(W, G, x, delta, gamma)
            return mv

    @staticmethod
    def _forcing_tol(base_tol: float, mu: Optional[float], phase: Optional[str]) -> float:
        # Inexact-Newton forcing (looser early, tighter late)
        if mu is not None:
            # classic: eta ~ min(0.5, sqrt(mu)) scaled onto base_tol
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

    def _hutch_diag_Kinv(self, cholK, n: int, probes: int, clip: float) -> Optional[np.ndarray]:
        """Hutchinson diagonal estimate for K^{-1}: mean((K^{-1} z) ⊙ z), z ∈ {±1}^n."""
        if probes <= 0 or cholK is None:
            return None
        acc = np.zeros(n, dtype=float)
        rng = np.random.default_rng(12345)  # deterministic; you may move to cache RNG
        for _ in range(probes):
            z = rng.integers(0, 2, size=n, dtype=np.int8).astype(float)
            z[z == 0] = -1.0
            v = cholK.solve_A(z)
            acc += v * z
        acc /= float(probes)
        return self._clip_pos(acc, clip)

    # ---------- gamma adaptation ----------
    def _adapt_gamma(self, gamma: float, cg_iters: Optional[int], target: int, bounds: Tuple[float, float],
                     inc: float, dec: float) -> float:
        if cg_iters is None:
            return gamma
        gmin, gmax = bounds
        # If Schur-CG too slow, increase gamma; if extremely fast, decrease
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
        if cholmod_factor:
            try:
                cholmod = self._cholmod_loader() if self._cholmod_loader else __import__("sksparse.cholmod", fromlist=["cholmod"])
            except Exception:
                cholmod = None

        # --- PATH A: CHOLMOD factorization of K_gamma (+ optional exact Schur) ---
        if cholmod is not None:
            need_refactor = (self._cholK is None) or (pattern_key != self._last_K_pattern) or (abs((self._last_delta or 0.0) - delta) > 0.0) or (not sticky_schur and (self._last_gamma is None or self._last_gamma != gamma))
            # If sticky_schur is True, allow small γ drift without rebuilding Schur; rebuild K if γ changed
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
                # Reset Schur cache (we’ll decide below whether to rebuild)
                if not sticky_schur:
                    self._cholS = None
                    self._S_explicit = None
                K_diag = K.diagonal().astype(float)
            else:
                # approximate diag(K) cheaply without rebuilding
                K_diag = (W.diagonal().astype(float) + delta + gamma * self._diag_of_GtG(G))

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
                    # reuse S if gamma hasn't moved too much (heuristic)
                    g_rel = abs((gamma - (self._last_gamma or gamma)) / max(1.0, abs(gamma)))
                    need_S_build = (g_rel > schur_reuse_tol)
            if use_exact_S and need_S_build:
                GT_dense = np.asfortranarray(G.T.toarray())  # (n, m)
                Z = cholK.solve_A(GT_dense)                  # (n, m)
                S_dense = G @ Z                              # (m, m)
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
                # Base diag approx from K-diag:
                diagK = np.maximum(K_diag, 1e-12)
                inv_diagK = 1.0 / diagK

                # Optional Hutchinson refinement for diag(K^{-1})
                diag_Kinv = None
                if hutch_probes > 0:
                    diag_Kinv = self._hutch_diag_Kinv(cholK, n, hutch_probes, hutch_clip)
                if diag_Kinv is None:
                    diag_Kinv = inv_diagK  # fall back

                # diag( S ) ≈ (G.^2) @ diag(K^{-1})
                Sdiag_hat = np.asarray((G.multiply(G)).dot(diag_Kinv), dtype=float).ravel()
                Sinv_diag = 1.0 / np.maximum(Sdiag_hat, 1e-12) if jacobi_schur_prec else None
                M = spla.LinearOperator((m, m), matvec=lambda z: Sinv_diag * z) if jacobi_schur_prec else None

                def S_mv(y: np.ndarray) -> np.ndarray:
                    return G @ cholK.solve_A(G.T @ y)

                # Forcing strategy on Schur tolerance
                schur_tol = self._forcing_tol(cg_tol, mu, phase)

                x0 = cache.prev_dy if cache.prev_dy is not None else None
                dy, cg_info = _cg_matfree(S_mv, rhs_s, x0=x0, tol=schur_tol, maxit=cg_maxit, M=M)
                cg_iters = int(getattr(cg_info, "iters", getattr(cg_info, "niter", 0)) or 0)
                dx = cholK.solve_A(svec - (G.T @ dy))

            # Adaptive gamma update (for next call)
            if adaptive_gamma:
                new_gamma = self._adapt_gamma(gamma, cg_iters, target_cg, gamma_bounds, gamma_increase, gamma_decrease)
                self._last_gamma = new_gamma  # used as hint next time

            # Save stats and warm-starts
            if save_cg_stats:
                cache.hykkt_stats = {"schur_cg_iters": cg_iters, "m": m, "n": n, "gamma": gamma}
            cache.prev_dx, cache.prev_dy = dx, dy
            self._last_cg_iters = cg_iters

            # Reusable solver
            if self._cholS is not None:
                class _Reusable(KKTReusable):
                    def __init__(self, cholK, cholS, G, gamma):
                        self.cholK = cholK; self.cholS = cholS
                        self.G = G; self.gamma = float(gamma)
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
                        self.cholK = cholK; self.G = G; self.gamma = float(gamma); self.Sinv_diag = Sinv_diag
                    def solve(self, r1n, r2n, cg_tol=1e-8, cg_maxit=200):
                        svec_n = r1n + self.gamma * (self.G.T @ r2n)
                        rhs_s_n = (self.G @ self.cholK.solve_A(svec_n)) - r2n
                        def S_mv_n(y): return self.G @ self.cholK.solve_A(self.G.T @ y)
                        M_n = None if self.Sinv_diag is None else spla.LinearOperator((self.G.shape[0], self.G.shape[0]), matvec=lambda z: self.Sinv_diag * z)
                        dyn, _ = _cg_matfree(S_mv_n, rhs_s_n, x0=None, tol=cg_tol, maxit=cg_maxit, M=M_n)
                        dxn = self.cholK.solve_A(svec_n - (self.G.T @ dyn))
                        return dxn, dyn
                return dx, dy, _Reusable(cholK, G, gamma, Sinv_diag if jacobi_schur_prec else None)

        # --- PATH B: Matrix-free CG for K_gamma (Numba-accelerated) -----
        # Jacobi preconditioner for K_gamma
        diagW = np.array(W.diagonal(), dtype=float)
        diagGtG = self._diag_of_GtG(G)
        Kdiag = np.maximum(diagW + delta + gamma * diagGtG, 1e-12)

        # Optional ILU preconditioner (explicit K), cached on pattern
        M_K = spla.LinearOperator((n, n), matvec=lambda z: z / Kdiag)
        if use_ilu_prec:
            ilu_key = (pattern_key, )
            if self._ilu is None or self._ilu_key != ilu_key:
                try:
                    K_exp = (W + delta * sp.eye(n, format="csr") + gamma * (G.T @ G).tocsr()).tocsc()
                    self._ilu = spla.spilu(K_exp, drop_tol=ilu_drop_tol, fill_factor=ilu_fill)
                    self._ilu_key = ilu_key
                except Exception:
                    self._ilu = None
                    self._ilu_key = None
            if self._ilu is not None:
                ilu = self._ilu
                M_K = spla.LinearOperator((n, n), matvec=lambda z: ilu.solve(z))

        # Numba-powered K_gamma matvec
        K_mv = self._make_numba_K_mv(W, G, delta, gamma)

        # Forcing on inner K-solve tolerance (easier than 10x cg strategy)
        inner_tol = self._forcing_tol(max(1e-2 * cg_tol, 1e-12), mu, phase)
        inner_maxit = max(10 * cg_maxit, 200)

        def inner_solve(q: np.ndarray) -> np.ndarray:
            x, _ = _cg_matfree(K_mv, q, x0=None, tol=inner_tol, maxit=inner_maxit, M=M_K)
            return x

        # Schur operator & rhs
        svec = r1 + gamma * (G.T @ r2)
        rhs_s = (G @ inner_solve(svec)) - r2

        # Cheap Jacobi preconditioner for Schur: diag(Ŝ) = (G.^2) @ (1/Kdiag)
        M_S = None
        if jacobi_schur_prec:
            Sinv_diag = 1.0 / np.maximum(
                np.asarray((G.multiply(G)).dot(1.0 / Kdiag), dtype=float).ravel(), 1e-12
            )
            M_S = spla.LinearOperator((m, m), matvec=lambda z: Sinv_diag * z)
        else:
            Sinv_diag = None

        # Forcing on Schur tolerance
        schur_tol = self._forcing_tol(cg_tol, mu, phase)
        x0 = cache.prev_dy if cache.prev_dy is not None else None

        def S_mv(y: np.ndarray) -> np.ndarray:
            return G @ inner_solve(G.T @ y)

        dy, cg_info = _cg_matfree(S_mv, rhs_s, x0=x0, tol=schur_tol, maxit=cg_maxit, M=M_S)
        cg_iters = int(getattr(cg_info, "iters", getattr(cg_info, "niter", 0)) or 0)
        dx = inner_solve(svec - (G.T @ dy))

        # Adaptive gamma update (for next calls through CHOL path too)
        if adaptive_gamma:
            new_gamma = self._adapt_gamma(gamma, cg_iters, target_cg, gamma_bounds, gamma_increase, gamma_decrease)
            self._last_gamma = new_gamma

        if save_cg_stats:
            cache.hykkt_stats = {"schur_cg_iters": cg_iters, "m": m, "n": n, "gamma": gamma, "path": "numba"}

        cache.prev_dx, cache.prev_dy = dx, dy
        self._last_cg_iters = cg_iters

        class _Reusable(KKTReusable):
            def __init__(self, G, inner_solve, gamma, Sinv_diag):
                self.G = G; self.inner_solve = inner_solve; self.gamma = float(gamma); self.Sinv_diag = Sinv_diag
            def solve(self, r1n, r2n, cg_tol=1e-8, cg_maxit=200):
                svec_n = r1n + self.gamma * (self.G.T @ r2n)
                rhs_s_n = (self.G @ self.inner_solve(svec_n)) - r2n
                def S_mv_n(y): return self.G @ self.inner_solve(self.G.T @ y)
                M_n = None if self.Sinv_diag is None else spla.LinearOperator((self.G.shape[0], self.G.shape[0]), matvec=lambda z: self.Sinv_diag * z)
                dyn, _ = _cg_matfree(S_mv_n, rhs_s_n, x0=None, tol=cg_tol, maxit=cg_maxit, M=M_n)
                dxn = self.inner_solve(svec_n - (self.G.T @ dyn))
                return dxn, dyn

        return dx, dy, _Reusable(G, inner_solve, gamma, Sinv_diag)

# ---------------------- Lifted SPD ----------------------
# ---------------------- Lifted SPD (with CHOLMOD) ----------------------
class LiftedStrategy(KKTStrategy):
    """
    LiftedKKT (τ-relaxation) — consistent with InteriorPointStepper.solve_KKT:
      r1 = rhs_x, r2 = -cE.

    System:
      K_tau = W + δ_w I + (1/τ) Gᵀ G
      K_tau dx = r1 - (1/τ) Gᵀ r2        # IMPORTANT: minus sign (since r2 = -cE)

    Returns:
      (dx, dy) with dy = 0 by default (true LiftedKKT has no equality dual step).
      Set return_pseudo_dy=True to get dy_hat = (G dx - r2)/τ if you really need it.
    """
    name = "lifted"

    def __init__(self, cholmod_loader: Optional[Callable[[], Any]] = None):
        self._cholmod_loader = cholmod_loader
        # tiny cache keyed on sparsity pattern only (values change every iter)
        self._pat = None
        self._chol = None
        self._solveA = None
        self._K = None
        self._used = "none"

    @staticmethod
    def _eye(n: int):
        return sp.eye(n, format="csr")

    def factor_and_solve(self, W, G, r1, r2, cfg, reg, cache, **kw):
        # --- inputs ---
        assert G is not None and r2 is not None, "LiftedKKT needs equality Jacobian/residual"
        W = W.tocsr() if not sp.isspmatrix_csr(W) else W
        G = G.tocsr() if not sp.isspmatrix_csr(G) else G
        n = int(W.shape[0])

        # params
        delta_w = float(cache.delta_w_last or 0.0)
        tau = float(kw.get("tau", getattr(cfg, "lift_tau", 1e-8)))
        tau = max(tau, 1e-16)
        prefer_factorized = bool(kw.get("prefer_factorized", True))
        max_refine = int(kw.get("max_refine", 2))
        refine_tol = float(kw.get("refine_tol", 1e-10))
        return_pseudo_dy = bool(kw.get("return_pseudo_dy", False))

        # --- assemble K_tau ---
        K = W.copy().tocsr()
        if delta_w != 0.0:
            K = (K + delta_w * self._eye(n)).tocsr()
        if tau > 0.0:
            K = (K + (1.0 / tau) * (G.T @ G).tocsr()).tocsr()
        K_csc = K.tocsc()

        # --- build RHS consistent with your (r1, r2) ---
        # r2 = -cE  ⇒  r1 - (1/τ) Gᵀ r2 = r1 + (1/τ) Gᵀ cE
        rhs = np.asarray(r1, float).ravel() - (1.0 / tau) * (G.T @ np.asarray(r2, float).ravel())

        # --- (re)factorization cache on pattern only ---
        pat = (W.shape, W.nnz, G.nnz)
        if pat != self._pat:
            self._chol = None
            self._solveA = None
            self._K = None
            self._used = "none"
            self._pat = pat

        # --- factor: CHOLMOD → factorized → spsolve ---
        solveA = None
        used = "none"
        try:
            cholmod = self._cholmod_loader() if self._cholmod_loader else __import__("sksparse.cholmod", fromlist=["cholmod"])
            chol = cholmod.cholesky(K_csc)
            solveA = chol.solve_A
            used = "cholmod"
            self._chol = chol
        except Exception:
            if prefer_factorized:
                try:
                    solveA = spla.factorized(K_csc)
                    used = "factorized"
                except Exception:
                    solveA = None
                    used = "spsolve"

        # --- solve & refinement (important for small τ) ---
        dx = solveA(rhs) if solveA is not None else spla.spsolve(K_csc, rhs)
        bnorm = np.linalg.norm(rhs)
        for _ in range(max_refine):
            r = rhs - (K_csc @ dx)
            if np.linalg.norm(r) <= refine_tol * (bnorm + 1.0):
                break
            e = solveA(r) if solveA is not None else spla.spsolve(K_csc, r)
            dx = dx + e

        # LiftedKKT has no equality dual; keep dnu neutral unless explicitly requested.
        if return_pseudo_dy:
            dy = (G @ dx - np.asarray(r2, float).ravel()) * (1.0 / tau)  # δc-style pseudo
        else:
            dy = np.zeros_like(np.asarray(r2, float).ravel())

        # save small state
        self._solveA, self._K, self._used = solveA, K_csc, used
        cache.prev_dx, cache.prev_dy = dx, dy

        class _Reusable(KKTReusable):
            def __init__(self, K, solveA):
                self.K = K; self.solveA = solveA
            def solve(self, r1n, r2n, **_):
                rhsn = np.asarray(r1n, float).ravel() - (1.0 / tau) * (G.T @ np.asarray(r2n, float).ravel())
                dxn = self.solveA(rhsn) if self.solveA is not None else spla.spsolve(self.K, rhsn)
                dyn = np.zeros_like(np.asarray(r2n, float).ravel())
                return dxn, dyn

        return dx, dy, _Reusable(K_csc, solveA)

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
DEFAULT_KKT_REGISTRY.register(LiftedStrategy())
DEFAULT_KKT_REGISTRY.register(LDLStrategy(prefer_lldl=False))  # name "ldl"

# Optional aliases if you want:
DEFAULT_KKT_REGISTRY._map["qdldl"] = DEFAULT_KKT_REGISTRY._map["ldl"]
DEFAULT_KKT_REGISTRY._map["lldl"] = LDLStrategy(prefer_lldl=True)
