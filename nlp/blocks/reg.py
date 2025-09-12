"""
regularizer_advanced.py

Enhanced regularization utilities with advanced preconditioning strategies.
No external AMG dependency (pyamg is NOT used).

Features:
- Symmetry enforcement and (optional) RCM permutation for sparsity/conditioning
- Symmetric Ruiz equilibration (scales rows/cols)
- TR-aware adaptive Tikhonov damping
- Low-rank PSD bump via small-eigen pair correction
- Inertia-aware path (optional if qdldl_cpp available)
- Multiple preconditioners without external libs:
    * 'none'          : Identity
    * 'jacobi'        : Diagonal scaling
    * 'bjacobi'       : Block-Jacobi with cached small LU blocks
    * 'ssor'          : Symmetric SOR linear operator
    * 'ilu'           : SciPy ILU (spilu) with drop/fill tuning
    * 'shift_invert'  : Cached factorization of (H - σI) for eigends
- Factorization & preconditioner caching keyed by sparsity pattern
- Dense & sparse paths with robust fallbacks
"""

from __future__ import annotations

import hashlib
import logging
import time
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import (
    LinearOperator,
    eigsh,
    spilu,
    splu,
    spsolve,
    svds,
)

# ---------- optional inertia via qdldl (NOT required) ----------
try:
    import qdldl_cpp as qd  # symmetric LDLᵀ with inertia
    HAS_QDLDL = True
except Exception:
    HAS_QDLDL = False


# ---------- telemetry ----------
@dataclass
class RegInfo:
    mode: str
    sigma: float
    cond_before: float
    cond_after: float
    min_eig_before: float
    min_eig_after: float
    rank_def: int
    inertia_before: Tuple[int, int, int]
    inertia_after: Tuple[int, int, int]
    nnz_before: int = 0
    nnz_after: int = 0
    precond_type: str = "none"
    precond_setup_time: float = 0.0
    eigensolve_iterations: int = 0
    eigensolve_converged: bool = True


# ---------- helpers ----------
ArrayLike = Union[np.ndarray, sp.spmatrix]


def _sym(A: ArrayLike) -> ArrayLike:
    return 0.5 * (A + A.T)


def _is_sparse(A: ArrayLike) -> bool:
    return sp.issparse(A)


def _nnz(A: ArrayLike) -> int:
    return A.nnz if sp.issparse(A) else A.size


def _perm_signature(A: sp.spmatrix) -> str:
    if not sp.isspmatrix_csr(A):
        A = A.tocsr()
    h = hashlib.blake2b(digest_size=16)
    h.update(np.int64(A.shape[0]).tobytes())
    h.update(A.indptr.tobytes())
    h.update(A.indices.tobytes())
    return h.hexdigest()


def _safe_dense(A: ArrayLike) -> np.ndarray:
    return A if isinstance(A, np.ndarray) else A.toarray()


def _eig_extents_dense(Hd: np.ndarray) -> Tuple[float, float]:
    w = la.eigvalsh(Hd)
    return float(w[0]), float(w[-1])


def _cond_from_extents(lmin: float, lmax: float) -> float:
    if lmax <= 0:
        return np.inf
    denom = max(abs(lmin), 1e-16)
    return float(lmax / denom)


def _ruiz_equilibrate(H: sp.spmatrix, iters: int = 3, norm: str = "l2"):
    """Symmetric Ruiz equilibration: returns (H_scaled, diag_vec d) with Hs = D H D."""
    if not sp.issparse(H):
        H = sp.csr_matrix(H)
    n = H.shape[0]
    d = np.ones(n, dtype=float)
    for _ in range(max(1, iters)):
        if norm == "linf":
            r = np.maximum(1e-16, np.array(np.abs(H).sum(axis=1)).ravel())
        else:
            r = np.maximum(
                1e-16, np.sqrt(np.array(H.multiply(H).sum(axis=1)).ravel())
            )
        s = 1.0 / np.sqrt(r)
        D = sp.diags(s)
        H = D @ H @ D
        d *= s
    return H, d


def _apply_diag_scaling(H: sp.spmatrix, d: np.ndarray) -> sp.spmatrix:
    D = sp.diags(d)
    return D @ H @ D


def _inertia_qdldl(H: sp.spmatrix) -> Tuple[int, int, int]:
    """Return (n_pos, n_neg, n_zero) via LDLᵀ; requires symmetric CSC."""
    if not HAS_QDLDL:
        return (0, 0, 0)
    Hc = H if sp.isspmatrix_csc(H) else H.tocsc()
    try:
        fact = qd.factorize(Hc)
        return fact.inertia()
    except Exception:
        return (0, 0, 0)


def _tikhonov(H: ArrayLike, sigma: float) -> ArrayLike:
    if sp.issparse(H):
        return H + sigma * sp.eye(H.shape[0], format="csr")
    return H + sigma * np.eye(H.shape[0])


def _low_rank_psd_bump_from_small_eigs(
    H: ArrayLike, eigvals: np.ndarray, eigvecs: np.ndarray, floor: float
) -> ArrayLike:
    """Raise eigenvalues below 'floor' via low-rank update V diag(delta) Vᵀ."""
    mask = eigvals < floor
    if not np.any(mask):
        return H
    V = eigvecs[:, mask]
    delta = (floor - eigvals[mask]).astype(float)
    if sp.issparse(H):
        upd = (V * delta) @ V.T  # dense low rank
        Hreg = H + sp.csr_matrix(upd)
    else:
        Hreg = H + V @ (np.diag(delta) @ V.T)
    return _sym(Hreg)


# ---------- block utilities for Block-Jacobi ----------
def _make_blocks(n: int, block_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return start/end indices (inclusive-exclusive) for contiguous blocks."""
    starts = np.arange(0, n, block_size)
    ends = np.minimum(starts + block_size, n)
    return starts, ends


# ---------- minimal Smoothed-Aggregation AMG (no pyamg) ----------
class MiniAMG:
    """
    Tiny Smoothed-Aggregation AMG builder and one V-cycle applier.
    Assumptions: A is real symmetric positive definite (SPD), CSR.
    Design goals: robustness over ultimate speed; no external deps.
    """

    def __init__(
        self,
        A: csr_matrix,
        theta: float = 0.08,        # strength threshold in [0,1)
        jacobi_omega: float = 0.7,  # smoother weight for Jacobi and P-smoothing
        presmooth: int = 1,
        postsmooth: int = 1,
        max_levels: int = 10,
        min_coarse: int = 40,       # stop when n <= min_coarse
    ):
        if not sp.isspmatrix_csr(A):
            A = A.tocsr()
        self.A0 = A
        self.theta = float(theta)
        self.jw = float(jacobi_omega)
        self.presmooth = int(presmooth)
        self.postsmooth = int(postsmooth)
        self.max_levels = int(max_levels)
        self.min_coarse = int(min_coarse)

        self.levels = []  # each: dict(A, D, P, R)
        self._build_hierarchy()

    # ----- public API: apply M^{-1} approximately via one V-cycle -----
    def solve(self, rhs: np.ndarray) -> np.ndarray:
        if rhs.ndim != 1:
            rhs = np.asarray(rhs).ravel()
        x = np.zeros_like(rhs)
        return self._vcycle(0, x, rhs)

    # ----- hierarchy construction -----
    def _build_hierarchy(self):
        A = self.A0
        for lev in range(self.max_levels):
            n = A.shape[0]
            D = A.diagonal().astype(float)
            D[D == 0] = 1.0
            level = {"A": A.tocsr(), "D": D}
            self.levels.append(level)

            if n <= self.min_coarse or lev == self.max_levels - 1:
                # stop; no prolongation at coarsest level
                break

            # strength of connection: |a_ij| >= theta * sqrt(a_ii a_jj)
            S = self._strength_graph(A, self.theta)

            # greedy aggregation
            agg = self._greedy_aggregate(S)  # size n, values in [0, ncoarse-1]
            Nc = int(agg.max()) + 1
            if Nc <= 0 or Nc >= n:
                # failed coarsening -> stop
                break

            # tentative prolongator T: P_ij = 1 if i in aggregate j
            P = self._tentative_prolongator(agg, Nc, n)

            # smooth prolongator: P = (I - ω D^{-1} A) T
            Dinv = 1.0 / D
            PT = P  # rename
            # y = A @ T  (sparse @ sparse -> sparse; but we need dense columns)
            # We'll compute A*(each column of T). Since T has one 1 per row, we can do:
            AT_cols = []
            PT_csr = PT.tocsr()
            for j in range(Nc):
                # column j of T is indicator of aggregate j: t_j[i] = 1 if agg[i]==j
                idx = PT_csr[:, j].nonzero()[0]  # rows with 1
                v = np.zeros(n)
                v[idx] = 1.0
                AT_cols.append(A @ v)
            AT = np.vstack(AT_cols).T  # shape (n, Nc)
            Psm = PT - sp.diags(self.jw * Dinv) @ sp.csr_matrix(AT)

            # coarse matrix: Ac = P^T A P
            Ac = (Psm.T @ (A @ Psm)).tocsr()

            # save and continue
            level["P"] = Psm.tocsr()
            level["R"] = level["P"].T  # symmetric case
            A = Ac

    # ----- components -----
    @staticmethod
    def _strength_graph(A: csr_matrix, theta: float) -> csr_matrix:
        A = A.tocsr()
        n = A.shape[0]
        D = A.diagonal()
        # s_ij = |a_ij| / sqrt(a_ii a_jj) >= theta
        # We form a mask on off-diagonals.
        rows, cols = A.nonzero()
        mask = rows != cols
        rows = rows[mask]; cols = cols[mask]
        data = np.abs(A.data[A.has_sorted_indices and mask or np.isin(np.arange(A.nnz), np.where(mask)[0])])
        # robust denom
        denom = np.sqrt(np.maximum(D[rows], 1e-32) * np.maximum(D[cols], 1e-32))
        val = np.zeros_like(data)
        nz = denom > 0
        val[nz] = (data[nz] / denom[nz]) >= theta
        # build 0/1 strength matrix on off-diagonals; keep symmetric
        S = sp.csr_matrix((val.astype(float), (rows, cols)), shape=A.shape)
        S = S.maximum(S.T)
        return S

    @staticmethod
    def _greedy_aggregate(S: csr_matrix) -> np.ndarray:
        n = S.shape[0]
        agg = -np.ones(n, dtype=int)
        visited = np.zeros(n, dtype=bool)
        cur = 0
        indptr, indices = S.indptr, S.indices

        for i in range(n):
            if visited[i]:
                continue
            # make i a seed
            agg[i] = cur
            visited[i] = True
            # attach strong neighbors to this aggregate
            start, end = indptr[i], indptr[i + 1]
            nbrs = indices[start:end]
            for j in nbrs:
                if not visited[j]:
                    agg[j] = cur
                    visited[j] = True
            cur += 1

        # any leftover (isolated) nodes: assign singleton aggregate
        for i in range(n):
            if agg[i] < 0:
                agg[i] = cur
                cur += 1

        return agg

    @staticmethod
    def _tentative_prolongator(agg: np.ndarray, Nc: int, n: int) -> csr_matrix:
        # one 1 per row, mapping fine node -> its aggregate
        rows = np.arange(n, dtype=int)
        cols = agg
        data = np.ones(n, dtype=float)
        return sp.csr_matrix((data, (rows, cols)), shape=(n, Nc))

    # ----- smoothing helpers -----
    @staticmethod
    def _jacobi(A: csr_matrix, D: np.ndarray, x: np.ndarray, b: np.ndarray, w: float, iters: int):
        # x <- x + w D^{-1}(b - A x)
        Dinv = 1.0 / D
        for _ in range(max(0, iters)):
            r = b - A @ x
            x += w * (Dinv * r)
        return x

    # ----- V-cycle -----
    def _vcycle(self, k: int, x: np.ndarray, b: np.ndarray) -> np.ndarray:
        level = self.levels[k]
        A, D = level["A"], level["D"]

        # pre-smooth
        x = self._jacobi(A, D, x, b, self.jw, self.presmooth)

        # coarse correction or solve
        if "P" not in level:
            # coarsest: dense solve fallback
            try:
                Ad = A.toarray()
                x = la.solve(Ad, b, assume_a="sym")
            except Exception:
                # a few extra Jacobi if solve fails
                x = self._jacobi(A, D, x, b, self.jw, 12)
            return x

        # compute residual
        r = b - A @ x
        # restrict
        R = level["R"]
        bc = R @ r
        # coarse solve via recursion (zero initial guess)
        xc = np.zeros_like(bc)
        xc = self._vcycle(k + 1, xc, bc)
        # prolongate and correct
        P = level["P"]
        x += P @ xc

        # post-smooth
        x = self._jacobi(A, D, x, b, self.jw, self.postsmooth)
        return x


# ---------- main class ----------
class Regularizer:
    """
    SOTA regularizer for Hessians with:
      - Symmetry enforcement
      - Optional RCM permutation (improves bandwidth/conditioning)
      - Symmetric Ruiz equilibration
      - Inertia via LDLᵀ (qdldl) when available (optional)
      - Cached shift-invert, ILU, Block-Jacobi, SSOR, and MINI-AMG preconditioners
      - Trust-region & gradient-aware σ adaptation
      - Spectral floor via low-rank PSD bump
      - Fallback Tikhonov/SVD regularization

    Public API:
      regularize(H, iteration=0, model_quality=None, constraint_count=0, grad_norm=None, tr_radius=None)
        -> (H_reg, RegInfo)
    """

    # ---------------- minimal Smoothed-Aggregation AMG (no pyamg) ----------------
    class _MiniAMG:
        """
        Tiny Smoothed-Aggregation AMG and one V-cycle apply.
        Assumes A is real SPD, CSR. Focus: robustness & no external deps.
        """
        def __init__(
            self,
            A: csr_matrix,
            theta: float = 0.08,
            jacobi_omega: float = 0.7,
            presmooth: int = 1,
            postsmooth: int = 1,
            max_levels: int = 10,
            min_coarse: int = 40,
        ):
            if not sp.isspmatrix_csr(A):
                A = A.tocsr()
            self.A0 = A
            self.theta = float(theta)
            self.jw = float(jacobi_omega)
            self.presmooth = int(presmooth)
            self.postsmooth = int(postsmooth)
            self.max_levels = int(max_levels)
            self.min_coarse = int(min_coarse)
            self.levels = []  # dicts with A, D, P, R
            self._build_hierarchy()

        def solve(self, rhs: np.ndarray) -> np.ndarray:
            rhs = np.asarray(rhs).ravel()
            x = np.zeros_like(rhs)
            return self._vcycle(0, x, rhs)

        def _build_hierarchy(self):
            A = self.A0
            for lev in range(self.max_levels):
                n = A.shape[0]
                D = A.diagonal().astype(float)
                D[D == 0] = 1.0
                L = {"A": A.tocsr(), "D": D}
                self.levels.append(L)

                if n <= self.min_coarse or lev == self.max_levels - 1:
                    break

                S = self._strength_graph(A, self.theta)
                agg = self._greedy_aggregate(S)
                Nc = int(agg.max()) + 1
                if Nc <= 0 or Nc >= n:
                    break

                P = self._tentative_prolongator(agg, Nc, n)  # (n, Nc), 0/1
                # Smooth: P = (I - ω D^{-1} A) T
                Dinv = 1.0 / D
                # Compute A @ each column of T (indicator vectors)
                # (sparse-friendly but dense in columns; fine for small Nc per level)
                AT_cols = []
                PT_csr = P.tocsr()
                for j in range(Nc):
                    idx = PT_csr[:, j].nonzero()[0]
                    v = np.zeros(n)
                    v[idx] = 1.0
                    AT_cols.append(A @ v)
                AT = np.vstack(AT_cols).T  # (n, Nc)
                Psm = P - sp.diags(self.jw * Dinv) @ sp.csr_matrix(AT)

                Ac = (Psm.T @ (A @ Psm)).tocsr()
                L["P"] = Psm.tocsr()
                L["R"] = L["P"].T
                A = Ac

        @staticmethod
        def _strength_graph(A: csr_matrix, theta: float) -> csr_matrix:
            A = A.tocsr()
            n = A.shape[0]
            D = A.diagonal()
            rows, cols = A.nonzero()
            mask = rows != cols
            rows = rows[mask]
            cols = cols[mask]
            # pull the corresponding abs(data) safely
            # (handle unsorted indices conservatively)
            data = np.abs(A[rows, cols]).A1
            denom = np.sqrt(np.maximum(D[rows], 1e-32) * np.maximum(D[cols], 1e-32))
            val = (denom > 0) & ((data / denom) >= theta)
            S = sp.csr_matrix((val.astype(float), (rows, cols)), shape=A.shape)
            S = S.maximum(S.T)
            return S

        @staticmethod
        def _greedy_aggregate(S: csr_matrix) -> np.ndarray:
            n = S.shape[0]
            agg = -np.ones(n, dtype=int)
            visited = np.zeros(n, dtype=bool)
            indptr, indices = S.indptr, S.indices
            cur = 0
            for i in range(n):
                if visited[i]:
                    continue
                agg[i] = cur
                visited[i] = True
                nbrs = indices[indptr[i]:indptr[i+1]]
                for j in nbrs:
                    if not visited[j]:
                        agg[j] = cur
                        visited[j] = True
                cur += 1
            for i in range(n):
                if agg[i] < 0:
                    agg[i] = cur
                    cur += 1
            return agg

        @staticmethod
        def _tentative_prolongator(agg: np.ndarray, Nc: int, n: int) -> csr_matrix:
            rows = np.arange(n, dtype=int)
            cols = agg
            data = np.ones(n, dtype=float)
            return sp.csr_matrix((data, (rows, cols)), shape=(n, Nc))

        @staticmethod
        def _jacobi(A: csr_matrix, D: np.ndarray, x: np.ndarray, b: np.ndarray, w: float, iters: int):
            Dinv = 1.0 / D
            for _ in range(max(0, iters)):
                r = b - A @ x
                x += w * (Dinv * r)
            return x

        def _vcycle(self, k: int, x: np.ndarray, b: np.ndarray) -> np.ndarray:
            L = self.levels[k]
            A, D = L["A"], L["D"]
            x = self._jacobi(A, D, x, b, self.jw, self.presmooth)
            if "P" not in L:
                try:
                    x = la.solve(A.toarray(), b, assume_a="sym")
                except Exception:
                    x = self._jacobi(A, D, x, b, self.jw, 12)
                return x
            r = b - A @ x
            bc = L["R"] @ r
            xc = np.zeros_like(bc)
            xc = self._vcycle(k + 1, xc, bc)
            x += L["P"] @ xc
            x = self._jacobi(A, D, x, b, self.jw, self.postsmooth)
            return x
    # ---------------- end mini AMG ----------------

    def __init__(self, cfg):
        # config getter supporting dict or object
        def _cfg(name, default):
            return getattr(cfg, name, cfg.get(name, default) if isinstance(cfg, dict) else default)

        # mode & floors
        self.cfg = cfg
        self.mode = str(_cfg("reg_mode", "AUTO"))  # AUTO|EIGEN_MOD|INERTIA_FIX|SPECTRAL|TIKHONOV
        self.sigma = float(_cfg("reg_sigma", 1e-8))
        self.sigma_min = float(_cfg("reg_sigma_min", 1e-12))
        self.sigma_max = float(_cfg("reg_sigma_max", 1e6))
        self.target_cond = float(_cfg("reg_target_cond", 1e12))
        self.min_eig_floor = float(_cfg("reg_min_eig_thresh", 1e-8))

        # adaptation
        self.adapt_factor = float(_cfg("reg_adapt_factor", 2.0))
        self.iter_tol = float(_cfg("iter_tol", 1e-6))
        self.max_iter = int(_cfg("max_iter", 500))
        self.k_eigs = int(_cfg("k_eigs", 16))  # small spectrum sample

        # permutation + scaling
        self.use_rcm = bool(_cfg("reg_use_rcm", True))         # RCM instead of AMD (no AMD in SciPy)
        self.use_ruiz = bool(_cfg("reg_use_ruiz", True))
        self.ruiz_iters = int(_cfg("reg_ruiz_iters", 3))

        # preconditioning strategy
        self.use_preconditioning = bool(_cfg("use_preconditioning", True))
        # auto|none|jacobi|bjacobi|ssor|ilu|shift_invert|amg
        self.precond_type = str(_cfg("precond_type", "auto"))
        self.ilu_drop_tol = float(_cfg("ilu_drop_tol", 1e-3))
        self.ilu_fill_factor = float(_cfg("ilu_fill_factor", 10))
        self.shift_invert_mode = str(_cfg("shift_invert_mode", "buckling"))

        # block-jacobi
        self.bjacobi_block_size = int(_cfg("bjacobi_block_size", 64))

        # ssor
        self.ssor_omega = float(_cfg("ssor_omega", 1.0))

        # trust-region aware damping
        self.use_tr_aware = bool(_cfg("reg_tr_aware", True))
        self.tr_c = float(_cfg("reg_tr_c", 1e-2))  # σ >= c * ||g|| / Δ

        # caches
        self._ilu_cache: Dict[str, object] = {}
        self._bjacobi_cache: Dict[Tuple[str, int], object] = {}
        self._factor_cache: Dict[Tuple[str, float], object] = {}  # (pattern_hash, sigma) -> factor
        self._perm_cache: Dict[str, np.ndarray] = {}

        # history
        self.sigma_history: list[float] = []
        self.cond_history: list[float] = []

    # ---------- public API ----------
    def regularize(
        self,
        H: ArrayLike,
        iteration: int = 0,
        model_quality: Optional[float] = None,
        constraint_count: int = 0,
        grad_norm: Optional[float] = None,
        tr_radius: Optional[float] = None,
    ) -> Tuple[ArrayLike, RegInfo]:
        n = H.shape[0]
        if H.shape[0] != H.shape[1]:
            raise ValueError(f"Hessian must be square, got {H.shape}")

        was_sparse = _is_sparse(H)
        H = _sym(H) if not was_sparse else _sym(H).asformat("csr")

        # Optional RCM permutation (symmetric bandwidth reduction)
        perm = None
        if was_sparse and self.use_rcm and n >= 200:
            try:
                perm = sp.csgraph.reverse_cuthill_mckee(H, symmetric_mode=True)
                H = H[perm][:, perm]
            except Exception:
                perm = None

        # Optional Ruiz equilibration
        dscale = None
        if was_sparse and self.use_ruiz:
            H, dscale = _ruiz_equilibrate(H, iters=self.ruiz_iters, norm="l2")

        nnz_before = _nnz(H)
        analysis = self._analyze(H, n)

        # Adapt σ (cond & TR-aware)
        self._adapt_sigma(analysis, iteration, grad_norm, tr_radius)

        # Decide mode
        mode = self.mode
        if mode == "AUTO":
            if analysis["min_eig"] < -1e-10 or analysis["cond_num"] > self.target_cond:
                mode = "EIGEN_MOD"
            else:
                mode = "TIKHONOV"

        if mode == "EIGEN_MOD":
            H_reg, info = self._eigen_bump(H, analysis)
        elif mode == "INERTIA_FIX":
            H_reg, info = self._inertia_fix(H, analysis, constraint_count)
        elif mode == "SPECTRAL":
            H_reg, info = self._spectral_floor(H, analysis)
        else:  # TIKHONOV
            H_reg, info = self._tikhonov_floor(H, analysis)

        # Undo scaling/permutation
        if was_sparse and self.use_ruiz and dscale is not None:
            invd = 1.0 / dscale
            H_reg = _apply_diag_scaling(H_reg, invd)

        if was_sparse and perm is not None:
            iperm = np.empty_like(perm)
            iperm[perm] = np.arange(len(perm))
            H_reg = H_reg[iperm][:, iperm]

        info.nnz_before = nnz_before
        info.nnz_after = _nnz(H_reg)
        self.sigma_history.append(info.sigma)
        self.cond_history.append(info.cond_after)
        return H_reg, info

    # ---------- analysis ----------
    def _analyze(self, H: ArrayLike, n: int) -> Dict:
        is_sparse = _is_sparse(H)
        if not is_sparse and n <= 800:
            Hd = _safe_dense(H)
            lmin, lmax = _eig_extents_dense(Hd)
            cond = _cond_from_extents(lmin, lmax)
            inertia = (np.nan, np.nan, np.nan)
            return {
                "min_eig": lmin,
                "max_eig": lmax,
                "cond_num": cond,
                "inertia": inertia,
                "precond_type": "none",
                "precond_setup_time": 0.0,
            }

        # Sparse route
        inertia = (0, 0, 0)
        if HAS_QDLDL:
            try:
                inertia = _inertia_qdldl(H)
            except Exception:
                pass

        precond_type = self._select_precond(H, inertia_hint=inertia)
        lmin, lmax, psetup = self._extents_sparse(H, precond_type)

        cond = _cond_from_extents(lmin, lmax)
        return {
            "min_eig": lmin,
            "max_eig": lmax,
            "cond_num": cond,
            "inertia": inertia,
            "precond_type": precond_type,
            "precond_setup_time": psetup,
        }

    # ---------- improved AUTO preconditioner selection ----------
    def _select_precond(self, H: csr_matrix, inertia_hint: Tuple[int, int, int] = (0, 0, 0)) -> str:
        if not self.use_preconditioning or self.precond_type == "none":
            return "none"
        if self.precond_type != "auto":
            return self.precond_type

        n = H.shape[0]
        nnz = H.nnz
        density = nnz / max(1, n * n)
        avg_deg = nnz / max(1, n)
        diag = H.diagonal()
        diag_nz_ratio = float(np.mean(np.abs(diag) > 1e-14))
        weak_diag = float(np.mean(np.abs(diag) < 1e-10))
        # quick band proxy: mean |i-j| of nonzeros (sampled)
        sample = min(nnz, 20000)
        if sample > 0:
            rows, cols = H.nonzero()
            idx = np.random.RandomState(0).choice(len(rows), size=sample, replace=False)
            band_mean = float(np.mean(np.abs(rows[idx] - cols[idx])))
        else:
            band_mean = 0.0

        # SPD hint from inertia (if available)
        npos, nneg, nzero = inertia_hint if all(isinstance(x, (int, np.integer)) for x in inertia_hint) else (0, 0, 0)
        spd_hint = (nneg == 0 and (nzero == 0 or nzero < max(1, 0.01 * n)))

        # Heuristics:
        # 1) Very sparse & large & (SPD-ish): AMG
        if (n >= 1500 and density < 0.01 and avg_deg < 20 and spd_hint):
            return "amg"

        # 2) Very sparse but SPD unknown / indefinite: shift-invert helps eigen edges; ILU often strong
        if density < 0.01:
            # if diagonal is weak, ILU can be better than simple point smoothers
            return "ilu" if weak_diag > 0.05 else "shift_invert"

        # 3) Sparse to moderate density: ILU tends to be a robust default
        if density < 0.10:
            return "ilu"

        # 4) Moderately banded (low band_mean) and diagonally present: SSOR can help
        if band_mean < 0.1 * n and diag_nz_ratio > 0.95:
            return "ssor"

        # 5) Blocky structures (heuristic): if many short dense windows along diagonal -> Block-Jacobi
        #    We detect via average local density in windows of size b
        b = min(self.bjacobi_block_size, n)
        # crude local density proxy: nnz / (#blocks * b^2)
        num_blocks = int(np.ceil(n / b))
        approx_block_density = nnz / max(1, num_blocks * (b * b))
        if approx_block_density > 0.2 and density < 0.35:
            return "bjacobi"

        # 6) Fallbacks
        if density < 0.35:
            return "bjacobi"
        return "jacobi"

    # ---------- preconditioners ----------
    def _jacobi_operator(self, H: csr_matrix) -> LinearOperator:
        d = H.diagonal().astype(float)
        d[np.abs(d) < 1e-16] = 1.0
        invd = 1.0 / d
        return LinearOperator(H.shape, matvec=lambda x: invd * x)

    def _bjacobi_operator(self, H: csr_matrix) -> LinearOperator:
        key = (_perm_signature(H), int(self.bjacobi_block_size))
        cache = self._bjacobi_cache.get(key)
        n = H.shape[0]

        if cache is None:
            starts = np.arange(0, n, self.bjacobi_block_size)
            ends = np.minimum(starts + self.bjacobi_block_size, n)
            blocks = []
            for s, e in zip(starts, ends):
                Hi = (H[s:e, s:e]).tocsc()
                try:
                    lu = splu(Hi)  # exact LU of small block
                    blocks.append(("lu", s, e, lu))
                except Exception:
                    diag = Hi.diagonal()
                    diag[np.abs(diag) < 1e-16] = 1.0
                    blocks.append(("diag", s, e, diag))
            self._bjacobi_cache[key] = blocks
            cache = blocks

        def matvec(x: np.ndarray) -> np.ndarray:
            y = np.zeros_like(x)
            for kind, s, e, obj in cache:
                xi = x[s:e]
                if kind == "lu":
                    y[s:e] = obj.solve(xi)
                else:  # diag
                    y[s:e] = xi / obj
            return y

        return LinearOperator(H.shape, matvec=matvec)

    def _ssor_operator(self, H: csr_matrix) -> LinearOperator:
        omega = self.ssor_omega
        if not (0.0 < omega < 2.0):
            omega = 1.0

        Hc = H.tocsc()
        D = Hc.diagonal().astype(float)
        D[np.abs(D) < 1e-16] = 1.0
        invD = 1.0 / D
        L = sp.tril(Hc, k=-1).tocsc()
        U = sp.triu(Hc, k=1).tocsc()

        def forward(x):
            y = np.zeros_like(x)
            for i in range(Hc.shape[0]):
                s = x[i] - omega * L[i, :i].toarray().ravel().dot(y[:i])
                y[i] = s / (D[i])
            return y

        def backward(y):
            z = np.zeros_like(y)
            yprime = invD * y
            for i in range(Hc.shape[0] - 1, -1, -1):
                s = yprime[i] - omega * U[i, i + 1 :].toarray().ravel().dot(z[i + 1 :])
                z[i] = s / (D[i])
            return z

        return LinearOperator(H.shape, matvec=lambda x: backward(forward(x)))

    def _ilu_operator(self, H: csr_matrix) -> LinearOperator:
        key = _perm_signature(H)
        ilu = self._ilu_cache.get(key)
        if ilu is None:
            try:
                ilu = spilu(H.tocsc(), drop_tol=self.ilu_drop_tol, fill_factor=self.ilu_fill_factor)
                self._ilu_cache[key] = ilu
            except Exception as e:
                warnings.warn(f"ILU failed: {e}; using identity preconditioner.")
                return LinearOperator(H.shape, matvec=lambda x: x)
        return LinearOperator(H.shape, matvec=lambda x: ilu.solve(x))

    def _amg_operator(self, H: csr_matrix) -> LinearOperator:
        try:
            amg = self._MiniAMG(
                H,
                theta=0.08,
                jacobi_omega=0.7,
                presmooth=1,
                postsmooth=1,
                max_levels=10,
                min_coarse=40,
            )
        except Exception as e:
            warnings.warn(f"AMG setup failed: {e}; using identity preconditioner.")
            return LinearOperator(H.shape, matvec=lambda x: x)
        return LinearOperator(H.shape, matvec=lambda x: amg.solve(x))

    def _factorize_shift(self, H: csr_matrix, sigma: float):
        key = (_perm_signature(H), float(sigma))
        fac = self._factor_cache.get(key)
        if fac is not None:
            return fac
        A = (H - sigma * sp.eye(H.shape[0], format="csr")).tocsc()
        try:
            fac = sp.linalg.factorized(A)  # cached closure
        except Exception:
            fac = None
        self._factor_cache[key] = fac
        return fac

    # ---------- eigen edge helpers ----------
    def _eig_edge(
        self,
        H: csr_matrix,
        which: str,
        sigma: Optional[float] = None,
        mode: str = "buckling",
        M: Optional[LinearOperator] = None,
    ) -> float:
        if sigma is None:
            vals = eigsh(
                H, k=1, which=which, M=M,
                return_eigenvectors=False, tol=self.iter_tol, maxiter=self.max_iter
            )
            return float(vals[0])

        fac = self._factorize_shift(H, sigma)
        if fac is None:
            vals = eigsh(
                H, k=1, which=which, sigma=sigma, mode=mode, M=M,
                return_eigenvectors=False, tol=self.iter_tol, maxiter=self.max_iter
            )
            return float(vals[0])

        def op(x):
            return fac(x) if callable(fac) else spsolve(H - sigma * sp.eye(H.shape[0], format="csr"), x)

        OPinv = LinearOperator(shape=H.shape, matvec=op)
        vals = eigsh(
            H, k=1, which=which, OPinv=OPinv, sigma=sigma, mode=mode, M=M,
            return_eigenvectors=False, tol=self.iter_tol, maxiter=self.max_iter
        )
        return float(vals[0])

    def _extents_sparse(self, H: csr_matrix, precond_type: str) -> Tuple[float, float, float]:
        setup_t0 = time.time()
        M = None

        try:
            if precond_type == "shift_invert":
                lmin = self._eig_edge(H, which="SM", sigma=0.0, mode=self.shift_invert_mode)
                lmax = self._eig_edge(H, which="LA")
            else:
                if precond_type == "jacobi":
                    M = self._jacobi_operator(H)
                elif precond_type == "bjacobi":
                    M = self._bjacobi_operator(H)
                elif precond_type == "ssor":
                    M = self._ssor_operator(H)
                elif precond_type == "ilu":
                    M = self._ilu_operator(H)
                elif precond_type == "amg":
                    M = self._amg_operator(H)
                # else: M=None

                lmin = self._eig_edge(H, which="SA", M=M)
                lmax = self._eig_edge(H, which="LA", M=M)

            setup_time = time.time() - setup_t0
            return float(lmin), float(lmax), float(setup_time)
        except Exception as e:
            logging.debug(f"_extents_sparse fallback due to {e}")
            try:
                lmin = float(eigsh(H, k=1, which="SA", return_eigenvectors=False))
                lmax = float(eigsh(H, k=1, which="LA", return_eigenvectors=False))
                return lmin, lmax, time.time() - setup_t0
            except Exception:
                diag = H.diagonal() if sp.issparse(H) else np.diag(H)
                return float(np.min(diag)), float(np.max(diag)), time.time() - setup_t0

    # ---------- strategies ----------
    def _eigen_bump(self, H: ArrayLike, analysis: Dict) -> Tuple[ArrayLike, RegInfo]:
        floor = max(self.min_eig_floor, self.sigma)
        n = H.shape[0]
        k = min(self.k_eigs, max(1, n // 20))
        try:
            vals, vecs = eigsh(H, k=k, which="SA", tol=self.iter_tol, maxiter=self.max_iter)
            Hreg = _low_rank_psd_bump_from_small_eigs(H, vals, vecs, floor)
        except Exception:
            Hreg = _tikhonov(H, floor)

        post = self._post_analyze(Hreg, n)
        return Hreg, RegInfo(
            mode="EIGEN_MOD",
            sigma=self.sigma,
            cond_before=analysis["cond_num"],
            cond_after=post["cond_num"],
            min_eig_before=analysis["min_eig"],
            min_eig_after=post["min_eig"],
            rank_def=post["rank_def"],
            inertia_before=analysis["inertia"],
            inertia_after=post["inertia"],
            precond_type=analysis.get("precond_type", "none"),
            precond_setup_time=analysis.get("precond_setup_time", 0.0),
        )

    def _inertia_fix(self, H: ArrayLike, analysis: Dict, m_eq: int) -> Tuple[ArrayLike, RegInfo]:
        n = H.shape[0]
        target_pos = max(1, n - max(0, m_eq))
        k = min(self.k_eigs, max(2, n // 15))
        try:
            vals, vecs = eigsh(H, k=k, which="SA", tol=self.iter_tol, maxiter=self.max_iter)
            pos_now = int(np.sum(vals > 1e-12))
            if pos_now >= target_pos:
                Hreg = H
            else:
                Hreg = _low_rank_psd_bump_from_small_eigs(H, vals, vecs, max(self.sigma, self.min_eig_floor))
        except Exception:
            Hreg = _tikhonov(H, max(self.sigma, self.min_eig_floor))

        post = self._post_analyze(Hreg, n)
        return Hreg, RegInfo(
            mode="INERTIA_FIX",
            sigma=self.sigma,
            cond_before=analysis["cond_num"],
            cond_after=post["cond_num"],
            min_eig_before=analysis["min_eig"],
            min_eig_after=post["min_eig"],
            rank_def=post["rank_def"],
            inertia_before=analysis["inertia"],
            inertia_after=post["inertia"],
            precond_type=analysis.get("precond_type", "none"),
            precond_setup_time=analysis.get("precond_setup_time", 0.0),
        )

    def _spectral_floor(self, H: ArrayLike, analysis: Dict) -> Tuple[ArrayLike, RegInfo]:
        is_sparse = _is_sparse(H)
        n = H.shape[0]
        floor = max(self.sigma, self.min_eig_floor)
        try:
            if is_sparse or n > 800:
                try:
                    vals, vecs = eigsh(H, k=min(self.k_eigs, max(1, n - 2)), which="SA",
                                       tol=self.iter_tol, maxiter=self.max_iter)
                    Hreg = _low_rank_psd_bump_from_small_eigs(H, vals, vecs, floor)
                except Exception:
                    U, s, Vt = svds(H, k=min(self.k_eigs, max(1, n - 2)),
                                    tol=self.iter_tol, maxiter=self.max_iter)
                    s = np.maximum(s, floor)
                    Hreg = U @ np.diag(s) @ Vt
                    Hreg = sp.csr_matrix(_sym(Hreg)) if is_sparse else _sym(Hreg)
            else:
                Hd = _safe_dense(H)
                U, s, Vt = la.svd(Hd, full_matrices=False)
                s = np.maximum(s, floor)
                Hreg = _sym(U @ (s[:, None] * Vt))
        except Exception:
            Hreg = _tikhonov(H, floor)

        post = self._post_analyze(Hreg, n)
        return Hreg, RegInfo(
            mode="SPECTRAL",
            sigma=self.sigma,
            cond_before=analysis["cond_num"],
            cond_after=post["cond_num"],
            min_eig_before=analysis["min_eig"],
            min_eig_after=post["min_eig"],
            rank_def=post["rank_def"],
            inertia_before=analysis["inertia"],
            inertia_after=post["inertia"],
            precond_type=analysis.get("precond_type", "none"),
            precond_setup_time=analysis.get("precond_setup_time", 0.0),
        )

    def _tikhonov_floor(self, H: ArrayLike, analysis: Dict) -> Tuple[ArrayLike, RegInfo]:
        floor = max(self.sigma, self.min_eig_floor)
        if analysis["cond_num"] > self.target_cond and analysis["max_eig"] > 0:
            target_min = analysis["max_eig"] / self.target_cond
            floor = max(floor, target_min)
        Hreg = _tikhonov(H, floor)
        post = self._post_analyze(Hreg, H.shape[0])
        return Hreg, RegInfo(
            mode="TIKHONOV",
            sigma=floor,
            cond_before=analysis["cond_num"],
            cond_after=post["cond_num"],
            min_eig_before=analysis["min_eig"],
            min_eig_after=post["min_eig"],
            rank_def=post["rank_def"],
            inertia_before=analysis["inertia"],
            inertia_after=post["inertia"],
            precond_type=analysis.get("precond_type", "none"),
            precond_setup_time=analysis.get("precond_setup_time", 0.0),
        )

    # ---------- adaptation & post ----------
    def _adapt_sigma(
        self,
        analysis: Dict,
        iteration: int,
        grad_norm: Optional[float],
        tr_radius: Optional[float],
    ) -> None:
        if analysis["cond_num"] > self.target_cond:
            self.sigma = min(self.sigma_max, self.sigma * self.adapt_factor)
        elif analysis["cond_num"] < max(1.0, self.target_cond * 1e-2) and analysis["min_eig"] >= -1e-12:
            self.sigma = max(self.sigma_min, self.sigma / self.adapt_factor)

        if self.use_tr_aware and grad_norm is not None and tr_radius is not None and tr_radius > 0:
            self.sigma = max(self.sigma, self.tr_c * float(grad_norm) / float(tr_radius))

        if iteration < 3:
            self.sigma = max(self.sigma, 1e-8)

    def _post_analyze(self, H: ArrayLike, n: int) -> Dict:
        is_sparse = _is_sparse(H)
        if not is_sparse and n <= 800:
            Hd = _safe_dense(H)
            lmin, lmax = _eig_extents_dense(Hd)
            cond = _cond_from_extents(lmin, lmax)
            return {
                "min_eig": lmin,
                "max_eig": lmax,
                "cond_num": cond,
                "rank_def": int(np.sum(np.abs(la.eigvalsh(Hd)) < 1e-10)),
                "inertia": (np.nan, np.nan, np.nan),
            }
        try:
            lmin = float(eigsh(H, k=1, which="SA", return_eigenvectors=False))
            lmax = float(eigsh(H, k=1, which="LA", return_eigenvectors=False))
        except Exception:
            diag = H.diagonal() if is_sparse else np.diag(H)
            lmin, lmax = float(np.min(diag)), float(np.max(diag))
        cond = _cond_from_extents(lmin, lmax)
        inertia = _inertia_qdldl(H) if HAS_QDLDL and is_sparse else (np.nan, np.nan, np.nan)
        rank_def = 0 if np.isnan(inertia[2]) else int(inertia[2])
        return {
            "min_eig": lmin,
            "max_eig": lmax,
            "cond_num": cond,
            "rank_def": rank_def,
            "inertia": inertia,
        }

    # ---------- utilities ----------
    def get_statistics(self) -> Dict:
        return {
            "current_sigma": self.sigma,
            "mode": self.mode,
            "avg_condition": (float(np.mean(self.cond_history)) if self.cond_history else np.nan),
            "sigma_range": ((min(self.sigma_history), max(self.sigma_history)) if self.sigma_history else (np.nan, np.nan)),
            "ilu_cache_size": len(self._ilu_cache),
            "bjacobi_cache_size": len(self._bjacobi_cache),
            "fact_cache_size": len(self._factor_cache),
        }

    def reset(self) -> None:
        self.sigma_history.clear()
        self.cond_history.clear()
        self._ilu_cache.clear()
        self._bjacobi_cache.clear()
        self._factor_cache.clear()
        self.sigma = float(getattr(self.cfg, "reg_sigma", 1e-8))

# ---------- convenience API ----------
def make_psd_advanced(
    H: ArrayLike,
    regularizer: Regularizer,
    iteration: int = 0,
    model_quality: Optional[float] = None,
    constraint_count: int = 0,
) -> Tuple[ArrayLike, RegInfo]:
    """Convert H to a numerically PSD-usable matrix via the enhanced regularizer."""
    return regularizer.regularize(
        H,
        iteration=iteration,
        model_quality=model_quality,
        constraint_count=constraint_count,
    )
