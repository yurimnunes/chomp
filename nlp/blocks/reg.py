"""
Enhanced regularization utilities with advanced preconditioning strategies.

Adds multiple preconditioning approaches to improve convergence and robustness
of iterative eigenvalue computations for large/sparse Hessians.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import (
    LinearOperator,
    eigsh,
    spsolve,
    svds,
)

# Optional advanced preconditioners
try:
    import pyamg

    HAS_PYAMG = True
except ImportError:
    HAS_PYAMG = False
    warnings.warn("pyamg not available - multilevel preconditioning disabled")

# ======================================
# Data structures (unchanged)
# ======================================


# ======================================
# Preconditioning Infrastructure
# ======================================

# regularizer_advanced.py
# SOTA, optimization-friendly Hessian regularizer with adaptive spectral fixes,
# inertia-aware corrections, trust-region-aware damping, Ruiz equilibration,
# optional AMD permutation, and cached preconditioners/factorizations.


import hashlib
import warnings

# ---------- optional deps ----------
try:
    import pyamg  # type: ignore

    HAS_PYAMG = True
except Exception:
    HAS_PYAMG = False

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


# ---------- helpers (self-contained) ----------
def _sym(A: Union[np.ndarray, sp.spmatrix]) -> Union[np.ndarray, sp.spmatrix]:
    return 0.5 * (A + A.T)


def _is_sparse(A) -> bool:
    return sp.issparse(A)


def _nnz(A) -> int:
    return A.nnz if sp.issparse(A) else A.size


def _perm_signature(A: sp.spmatrix) -> str:
    # hash sparsity pattern to cache factorizations/preconditioners
    if not sp.isspmatrix_csr(A):
        A = A.tocsr()
    h = hashlib.blake2b(digest_size=16)
    h.update(np.int64(A.shape[0]).tobytes())
    h.update(A.indptr.tobytes())
    h.update(A.indices.tobytes())
    return h.hexdigest()


def _ruiz_equilibrate(H: sp.spmatrix, iters: int = 3, norm: str = "l2"):
    """Symmetric Ruiz equilibration: D H D, returns (H_scaled, diag_vec d)."""
    if not sp.issparse(H):
        H = sp.csr_matrix(H)
    n = H.shape[0]
    d = np.ones(n, dtype=float)
    for _ in range(max(1, iters)):
        if norm == "linf":
            r = np.maximum(1e-16, np.array(np.abs(H).sum(axis=1)).ravel())
        else:  # l2
            r = np.maximum(1e-16, np.sqrt(np.array(H.multiply(H).sum(axis=1)).ravel()))
        s = 1.0 / np.sqrt(r)
        D = sp.diags(s)
        H = D @ H @ D
        d *= s
    return H, d


def _apply_diag_scaling(H: sp.spmatrix, d: np.ndarray) -> sp.spmatrix:
    D = sp.diags(d)
    return D @ H @ D


def _undo_scaling_vector(x: np.ndarray, d: np.ndarray) -> np.ndarray:
    # For eigvectors in scaled space v_scaled = D v_orig, but we only need H-reg result; no-op.
    return x


def _safe_dense(A: Union[np.ndarray, sp.spmatrix]) -> np.ndarray:
    return A if isinstance(A, np.ndarray) else A.toarray()


def _eig_extents_dense(Hd: np.ndarray) -> Tuple[float, float]:
    w = la.eigvalsh(Hd)
    return float(w[0]), float(w[-1])


def _cond_from_extents(lmin: float, lmax: float) -> float:
    if lmax <= 0:  # pathological
        return np.inf
    denom = max(abs(lmin), 1e-16)
    return float(lmax / denom)


def _inertia_qdldl(H: sp.spmatrix) -> Tuple[int, int, int]:
    # pos, neg, zero counts via LDLᵀ; requires symmetric input
    if not HAS_QDLDL:
        return (0, 0, 0)
    Hc = H if sp.isspmatrix_csc(H) else H.tocsc()
    try:
        fact = qd.factorize(Hc)
        return fact.inertia()
    except Exception:
        return (0, 0, 0)


def _low_rank_psd_bump_from_small_eigs(
    H: Union[np.ndarray, sp.spmatrix],
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    floor: float,
) -> Union[np.ndarray, sp.spmatrix]:
    """Raise eigenvalues below 'floor' via low-rank update V diag(delta) Vᵀ."""
    mask = eigvals < floor
    if not np.any(mask):
        return H
    V = eigvecs[:, mask]
    delta = (floor - eigvals[mask]).astype(float)
    # H + V diag(delta) Vᵀ
    if sp.issparse(H):
        upd = (V * delta) @ V.T  # dense small-rank
        Hreg = H + sp.csr_matrix(upd)
    else:
        Hreg = H + V @ (np.diag(delta) @ V.T)
    # re-symmetrize numerically
    return _sym(Hreg)


def _tikhonov(H, sigma):
    if sp.issparse(H):
        return H + sigma * sp.eye(H.shape[0], format="csr")
    return H + sigma * np.eye(H.shape[0])


# ---------- main class ----------
class Regularizer:
    """
    SOTA regularizer for Hessians with:
      - Symmetry enforcement
      - Optional AMD permutation
      - Ruiz equilibration (symmetric)
      - Inertia via LDLᵀ (qdldl) when available
      - Cached shift-invert / ILU / AMG preconditioners
      - Trust-region & gradient-aware σ adaptation
      - Spectral floor via low-rank PSD bump (pivoted, via small eigs)
      - Fallback Tikhonov and SVD regularization
    """

    # ---------- ctor / config ----------
    def __init__(self, cfg):
        # baselines
        self.cfg = cfg
        self.mode = getattr(
            cfg, "reg_mode", "AUTO"
        )  # AUTO|EIGEN_MOD|INERTIA_FIX|SPECTRAL|TIKHONOV
        self.sigma = float(getattr(cfg, "reg_sigma", 1e-8))
        self.sigma_min = float(getattr(cfg, "reg_sigma_min", 1e-12))
        self.sigma_max = float(getattr(cfg, "reg_sigma_max", 1e6))
        self.target_cond = float(getattr(cfg, "reg_target_cond", 1e12))
        self.min_eig_floor = float(getattr(cfg, "reg_min_eig_thresh", 1e-8))

        # adaptation
        self.adapt_factor = float(getattr(cfg, "reg_adapt_factor", 2.0))
        self.iter_tol = float(getattr(cfg, "iter_tol", 1e-6))
        self.max_iter = int(getattr(cfg, "max_iter", 500))
        self.k_eigs = int(getattr(cfg, "k_eigs", 16))  # small spectrum sample

        # permutation + scaling
        self.use_amd = bool(getattr(cfg, "reg_use_amd", True))
        self.use_ruiz = bool(getattr(cfg, "reg_use_ruiz", True))
        self.ruiz_iters = int(getattr(cfg, "reg_ruiz_iters", 3))

        # preconditioning strategy
        self.use_preconditioning = bool(getattr(cfg, "use_preconditioning", True))
        self.precond_type = str(
            getattr(cfg, "precond_type", "auto")
        )  # auto|none|ilu|amg|shift_invert
        self.ilu_drop_tol = float(getattr(cfg, "ilu_drop_tol", 1e-3))
        self.ilu_fill_factor = float(getattr(cfg, "ilu_fill_factor", 10))
        self.amg_threshold = int(getattr(cfg, "amg_threshold", 7000))
        self.shift_invert_mode = str(getattr(cfg, "shift_invert_mode", "buckling"))
        self.adaptive_precond = bool(getattr(cfg, "adaptive_precond", True))

        # trust-region aware damping
        self.use_tr_aware = bool(getattr(cfg, "reg_tr_aware", True))
        self.tr_c = float(getattr(cfg, "reg_tr_c", 1e-2))  # σ >= c*||g||/Δ

        # caches
        self._ilu_cache: Dict[str, object] = {}
        self._factor_cache: Dict[Tuple[str, float], object] = (
            {}
        )  # (pattern_hash, sigma) -> factor
        self._perm_cache: Dict[str, np.ndarray] = {}

        # history
        self.sigma_history: list[float] = []
        self.cond_history: list[float] = []

    # ---------- public API ----------
    def regularize(
        self,
        H: Union[np.ndarray, sp.spmatrix],
        iteration: int = 0,
        model_quality: Optional[float] = None,
        constraint_count: int = 0,
        grad_norm: Optional[float] = None,
        tr_radius: Optional[float] = None,
    ) -> Tuple[Union[np.ndarray, sp.spmatrix], RegInfo]:
        """Main entry. Returns (H_reg, RegInfo)."""
        n = H.shape[0]
        if H.shape[0] != H.shape[1]:
            raise ValueError(f"Hessian must be square, got {H.shape}")

        was_sparse = _is_sparse(H)
        H = _sym(H) if not was_sparse else _sym(H).asformat("csr")

        # Optional AMD permutation (reorders for ILU/LDL conditioning)
        perm = None
        if was_sparse and self.use_amd and n >= 200:
            try:
                perm = sp.csgraph.reverse_cuthill_mckee(H, symmetric_mode=True)
                H = H[perm][:, perm]
            except Exception:
                perm = None

        # Optional Ruiz equilibration (stabilizes all downstream ops)
        dscale = None
        if was_sparse and self.use_ruiz:
            H, dscale = _ruiz_equilibrate(H, iters=self.ruiz_iters, norm="l2")

        nnz_before = _nnz(H)
        analysis = self._analyze(H, n)

        # Adapt σ: condition number & TR-aware
        self._adapt_sigma(analysis, iteration, grad_norm, tr_radius)

        # Mode selection
        mode = self.mode
        if mode == "AUTO":
            # fast decision: if clear indefiniteness near zero or large neg eigs -> EIGEN_MOD (low-rank bump)
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

        # Undo scaling/permutation for downstream users (structure-only)
        if was_sparse and self.use_ruiz and dscale is not None:
            # Undo D H D -> H = D^{-1} H_reg D^{-1}. But for a *regularized* Hessian
            # we can simply rescale with the inverse to keep equivalence:
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
    def _analyze(self, H: Union[np.ndarray, sp.spmatrix], n: int) -> Dict:
        is_sparse = _is_sparse(H)
        if not is_sparse and n <= 800:
            Hd = _safe_dense(H)
            lmin, lmax = _eig_extents_dense(Hd)
            cond = _cond_from_extents(lmin, lmax)
            inertia = (np.nan, np.nan, np.nan)  # not computed
            return {
                "min_eig": lmin,
                "max_eig": lmax,
                "cond_num": cond,
                "inertia": inertia,
                "precond_type": "none",
                "precond_setup_time": 0.0,
            }

        # Sparse path: attempt inertia quickly via LDLᵀ if available
        inertia = (0, 0, 0)
        if HAS_QDLDL:
            try:
                inertia = _inertia_qdldl(H)
            except Exception:
                pass

        # Get small and large eigenvalues (lightweight)
        precond_type = self._select_precond(H)
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

    def _select_precond(self, H: sp.spmatrix) -> str:
        if not self.use_preconditioning or self.precond_type == "none":
            return "none"
        if self.precond_type != "auto":
            if self.precond_type == "amg" and not HAS_PYAMG:
                return "none"
            return self.precond_type
        n = H.shape[0]
        density = H.nnz / max(1, n * n)
        if n > self.amg_threshold and HAS_PYAMG and density < 0.05:
            return "amg"  # large, very sparse, likely elliptic-ish
        if density > 0.3:
            return "ilu"
        return "shift_invert"

    def _extents_sparse(
        self, H: sp.spmatrix, precond_type: str
    ) -> Tuple[float, float, float]:
        setup_time = 0.0
        try:
            if precond_type == "shift_invert":
                # Use factorization cache for (H - σI) with σ≈0
                lmin = self._eig_edge(
                    H, which="SM", sigma=0.0, mode=self.shift_invert_mode
                )
                lmax = self._eig_edge(H, which="LA")
            elif precond_type == "amg" and HAS_PYAMG:
                ml = pyamg.ruge_stuben_solver(H, max_levels=8, max_coarse=200)
                M = LinearOperator(
                    H.shape, matvec=lambda x: ml.solve(x, tol=1e-8, maxiter=1)
                )
                lmin = float(
                    eigsh(
                        H,
                        k=1,
                        which="SA",
                        M=M,
                        tol=self.iter_tol,
                        maxiter=self.max_iter,
                        return_eigenvectors=False,
                    )
                )
                lmax = float(
                    eigsh(
                        H,
                        k=1,
                        which="LA",
                        M=M,
                        tol=self.iter_tol,
                        maxiter=self.max_iter,
                        return_eigenvectors=False,
                    )
                )
            elif precond_type == "ilu":
                M = self._ilu_operator(H)
                lmin = float(
                    eigsh(
                        H,
                        k=1,
                        which="SA",
                        M=M,
                        tol=self.iter_tol,
                        maxiter=self.max_iter,
                        return_eigenvectors=False,
                    )
                )
                lmax = float(
                    eigsh(
                        H,
                        k=1,
                        which="LA",
                        M=M,
                        tol=self.iter_tol,
                        maxiter=self.max_iter,
                        return_eigenvectors=False,
                    )
                )
            else:
                lmin = float(
                    eigsh(
                        H,
                        k=1,
                        which="SA",
                        tol=self.iter_tol,
                        maxiter=self.max_iter,
                        return_eigenvectors=False,
                    )
                )
                lmax = float(
                    eigsh(
                        H,
                        k=1,
                        which="LA",
                        tol=self.iter_tol,
                        maxiter=self.max_iter,
                        return_eigenvectors=False,
                    )
                )
            return float(lmin), float(lmax), setup_time
        except Exception as e:
            logging.debug(f"_extents_sparse fallback due to {e}")
            try:
                lmin = float(eigsh(H, k=1, which="SA", return_eigenvectors=False))
                lmax = float(eigsh(H, k=1, which="LA", return_eigenvectors=False))
                return lmin, lmax, setup_time
            except Exception:
                # crude fallback
                diag = H.diagonal() if sp.issparse(H) else np.diag(H)
                return float(np.min(diag)), float(np.max(diag)), setup_time

    def _ilu_operator(self, H: sp.spmatrix) -> LinearOperator:
        key = _perm_signature(H)
        ilu = self._ilu_cache.get(key)
        if ilu is None:
            try:
                ilu = sp.linalg.spilu(
                    H.tocsc(),
                    drop_tol=self.ilu_drop_tol,
                    fill_factor=self.ilu_fill_factor,
                )
                self._ilu_cache[key] = ilu
            except Exception as e:
                warnings.warn(f"ILU failed: {e}; using identity preconditioner.")
                return LinearOperator(H.shape, matvec=lambda x: x)
        return LinearOperator(H.shape, matvec=lambda x: ilu.solve(x))

    def _factorize_shift(self, H: sp.spmatrix, sigma: float):
        key = (_perm_signature(H), float(sigma))
        fac = self._factor_cache.get(key)
        if fac is not None:
            return fac
        A = (H - sigma * sp.eye(H.shape[0], format="csr")).tocsc()
        try:
            fac = sp.linalg.factorized(A)  # cached sparse LU solver closure
        except Exception:
            # fallback to direct spsolve each time
            fac = None
        self._factor_cache[key] = fac
        return fac

    def _eig_edge(
        self,
        H: sp.spmatrix,
        which: str,
        sigma: Optional[float] = None,
        mode: str = "buckling",
    ) -> float:
        if sigma is None:
            vals = eigsh(
                H,
                k=1,
                which=which,
                return_eigenvectors=False,
                tol=self.iter_tol,
                maxiter=self.max_iter,
            )
            return float(vals[0])
        fac = self._factorize_shift(H, sigma)
        if fac is None:
            vals = eigsh(
                H,
                k=1,
                sigma=sigma,
                mode=mode,
                return_eigenvectors=False,
                tol=self.iter_tol,
                maxiter=self.max_iter,
            )
            return float(vals[0])

        # Build OPinv via cached factorization
        def op(x):
            return (
                fac(x)
                if callable(fac)
                else spsolve(H - sigma * sp.eye(H.shape[0], format="csr"), x)
            )

        OPinv = LinearOperator(shape=H.shape, matvec=op)
        vals = eigsh(
            H,
            k=1,
            which=which,
            OPinv=OPinv,
            sigma=sigma,
            mode=mode,
            return_eigenvectors=False,
            tol=self.iter_tol,
            maxiter=self.max_iter,
        )
        return float(vals[0])

    # ---------- strategies ----------
    def _eigen_bump(
        self, H, analysis: Dict
    ) -> Tuple[Union[np.ndarray, sp.spmatrix], RegInfo]:
        """Low-rank PSD bump to raise small/negative eigenvalues to floor."""
        floor = max(self.min_eig_floor, self.sigma)
        n = H.shape[0]
        k = min(self.k_eigs, max(1, n // 20))
        try:
            # focus on the small end
            vals, vecs = eigsh(
                H, k=k, which="SA", tol=self.iter_tol, maxiter=self.max_iter
            )
            Hreg = _low_rank_psd_bump_from_small_eigs(H, vals, vecs, floor)
        except Exception:
            Hreg = _tikhonov(H, floor)

        post = self._post_analyze(Hreg, H.shape[0])
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

    def _inertia_fix(
        self, H, analysis: Dict, m_eq: int
    ) -> Tuple[Union[np.ndarray, sp.spmatrix], RegInfo]:
        """Make Hessian have ≥ n-m_eq positive directions via targeted bump."""
        n = H.shape[0]
        target_pos = max(1, n - max(0, m_eq))
        k = min(self.k_eigs, max(2, n // 15))
        try:
            vals, vecs = eigsh(
                H, k=k, which="SA", tol=self.iter_tol, maxiter=self.max_iter
            )
            pos_now = int(np.sum(vals > 1e-12))
            if pos_now >= target_pos:
                Hreg = H
            else:
                # push the most negative up to floor
                Hreg = _low_rank_psd_bump_from_small_eigs(
                    H, vals, vecs, max(self.sigma, self.min_eig_floor)
                )
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

    def _spectral_floor(
        self, H, analysis: Dict
    ) -> Tuple[Union[np.ndarray, sp.spmatrix], RegInfo]:
        """SVD-based floor on singular values (dense fallback) or svds (sparse)."""
        is_sparse = _is_sparse(H)
        n = H.shape[0]
        floor = max(self.sigma, self.min_eig_floor)
        try:
            if is_sparse or n > 800:
                U, s, Vt = svds(
                    H,
                    k=min(self.k_eigs, n - 1),
                    tol=self.iter_tol,
                    maxiter=self.max_iter,
                )
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

    def _tikhonov_floor(
        self, H, analysis: Dict
    ) -> Tuple[Union[np.ndarray, sp.spmatrix], RegInfo]:
        floor = max(self.sigma, self.min_eig_floor)
        # If cond too large, increase floor to meet target condition
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
        # condition-based
        if analysis["cond_num"] > self.target_cond:
            self.sigma = min(self.sigma_max, self.sigma * self.adapt_factor)
        elif (
            analysis["cond_num"] < max(1.0, self.target_cond * 1e-2)
            and analysis["min_eig"] >= -1e-12
        ):
            self.sigma = max(self.sigma_min, self.sigma / self.adapt_factor)

        # trust-region aware (Levenberg-style)
        if (
            self.use_tr_aware
            and grad_norm is not None
            and tr_radius is not None
            and tr_radius > 0
        ):
            self.sigma = max(
                self.sigma, self.tr_c * float(grad_norm) / float(tr_radius)
            )

        # early iters: avoid being too small
        if iteration < 3:
            self.sigma = max(self.sigma, 1e-8)

    def _post_analyze(self, H: Union[np.ndarray, sp.spmatrix], n: int) -> Dict:
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
        # sparse: quick endpoints
        try:
            lmin = float(eigsh(H, k=1, which="SA", return_eigenvectors=False))
            lmax = float(eigsh(H, k=1, which="LA", return_eigenvectors=False))
        except Exception:
            diag = H.diagonal() if is_sparse else np.diag(H)
            lmin, lmax = float(np.min(diag)), float(np.max(diag))
        cond = _cond_from_extents(lmin, lmax)
        inertia = (
            _inertia_qdldl(H) if HAS_QDLDL and is_sparse else (np.nan, np.nan, np.nan)
        )
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
            "avg_condition": (
                float(np.mean(self.cond_history)) if self.cond_history else np.nan
            ),
            "sigma_range": (
                (min(self.sigma_history), max(self.sigma_history))
                if self.sigma_history
                else (np.nan, np.nan)
            ),
            "ilu_cache_size": len(self._ilu_cache),
            "fact_cache_size": len(self._factor_cache),
        }

    def reset(self) -> None:
        self.sigma_history.clear()
        self.cond_history.clear()
        self._ilu_cache.clear()
        self._factor_cache.clear()
        self.sigma = float(getattr(self.cfg, "reg_sigma", 1e-8))


def make_psd_advanced(
    H: Union[np.ndarray, sp.spmatrix],
    regularizer: Regularizer,
    iteration: int = 0,
    model_quality: Optional[float] = None,
    constraint_count: int = 0,
) -> Tuple[Union[np.ndarray, sp.spmatrix], RegInfo]:
    """
    Convert H to a numerically PSD-usable matrix via the enhanced regularizer.

    Now includes preconditioning telemetry in the returned RegInfo.
    """
    return regularizer.regularize(
        H,
        iteration=iteration,
        model_quality=model_quality,
        constraint_count=constraint_count,
    )
