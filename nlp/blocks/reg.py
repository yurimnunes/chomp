"""
Enhanced regularization utilities with advanced preconditioning strategies.

Adds multiple preconditioning approaches to improve convergence and robustness
of iterative eigenvalue computations for large/sparse Hessians.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import (
    ArpackNoConvergence,
    LinearOperator,
    cg,
    eigsh,
    gmres,
    spilu,
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

@dataclass
class RegInfo:
    """Summary of a regularization pass."""
    mode: str
    sigma: float
    cond_before: float
    cond_after: float
    min_eig_before: float
    min_eig_after: float
    rank_def: int
    inertia_before: Tuple[int, int, int]  # (pos, neg, zero)
    inertia_after: Tuple[int, int, int]
    nnz_before: int = 0
    nnz_after: int = 0
    # New preconditioning telemetry
    precond_type: str = "none"
    precond_setup_time: float = 0.0
    eigensolve_iterations: int = 0
    eigensolve_converged: bool = True


# ======================================
# Preconditioning Infrastructure
# ======================================

class PreconditionerFactory:
    """Factory for creating various types of preconditioners."""
    
    @staticmethod
    def create_ilu_preconditioner(
        H: sp.spmatrix, 
        drop_tol: float = 1e-4, 
        fill_factor: int = 10,
        shift: float = 1e-8
    ) -> LinearOperator:
        """Create ILU(0) preconditioner with optional shift for stability."""
        import time
        start_time = time.time()
        
        try:
            # Add small shift to improve diagonal dominance
            H_shifted = H + shift * sp.eye(H.shape[0], format=H.format)
            
            # Convert to CSC for better factorization performance
            H_csc = H_shifted.tocsc()
            
            # Compute incomplete LU factorization
            ilu = spilu(H_csc, drop_tol=drop_tol, fill_factor=fill_factor)
            
            def matvec(x):
                return ilu.solve(x)
            
            def rmatvec(x):
                return ilu.solve(x)  # Symmetric case
            
            setup_time = time.time() - start_time
            logging.debug(f"ILU preconditioner setup in {setup_time:.3f}s")
            
            precond = LinearOperator(
                H.shape, matvec=matvec, rmatvec=rmatvec, dtype=H.dtype
            )
            precond.setup_time = setup_time
            return precond
            
        except Exception as e:
            logging.warning(f"ILU preconditioner failed: {e}. Using identity.")
            return LinearOperator(H.shape, matvec=lambda x: x)
    
    @staticmethod
    def create_amg_preconditioner(H: sp.spmatrix) -> Optional[LinearOperator]:
        """Create algebraic multigrid preconditioner (requires pyamg)."""
        if not HAS_PYAMG:
            return None
            
        import time
        start_time = time.time()
        
        try:
            # Ensure matrix is in CSR format for AMG
            H_csr = H.tocsr()
            
            # Create AMG hierarchy (Ruge-Stuben classical AMG)
            ml = pyamg.ruge_stuben_solver(H_csr, max_levels=10, max_coarse=100)
            
            def matvec(x):
                return ml.solve(x, tol=1e-8, maxiter=1, cycle='V')
            
            setup_time = time.time() - start_time
            logging.debug(f"AMG preconditioner setup in {setup_time:.3f}s")
            
            precond = LinearOperator(H.shape, matvec=matvec, dtype=H.dtype)
            precond.setup_time = setup_time
            return precond
            
        except Exception as e:
            logging.warning(f"AMG preconditioner failed: {e}")
            return None
    
    @staticmethod
    def create_shift_invert_operator(
        H: Union[np.ndarray, sp.spmatrix], 
        sigma: float = 0.0,
        solver_type: str = "spsolve"
    ) -> LinearOperator:
        """Create shift-and-invert operator (H - σI)^(-1) for eigenvalue problems."""
        import time
        start_time = time.time()
        
        n = H.shape[0]
        is_sparse = sp.issparse(H)
        
        if is_sparse:
            # Sparse shift-and-invert
            H_shifted = H - sigma * sp.eye(n, format=H.format)
            
            if solver_type == "spsolve":
                # Direct sparse solve
                def matvec(x):
                    return spsolve(H_shifted, x)
            else:
                # Iterative solve with preconditioning
                precond = PreconditionerFactory.create_ilu_preconditioner(
                    H_shifted, drop_tol=1e-3
                )
                
                def matvec(x):
                    sol, info = gmres(H_shifted, x, M=precond, tol=1e-8, maxiter=100)
                    if info != 0:
                        logging.warning(f"GMRES failed with info={info}")
                    return sol
        else:
            # Dense shift-and-invert
            try:
                H_shifted = H - sigma * np.eye(n)
                L, U = la.lu_factor(H_shifted)
                
                def matvec(x):
                    return la.lu_solve((L, U), x)
            except la.LinAlgError:
                # Fallback to least squares
                def matvec(x):
                    return la.lstsq(H_shifted, x)[0]
        
        setup_time = time.time() - start_time
        logging.debug(f"Shift-invert operator setup in {setup_time:.3f}s")
        
        op = LinearOperator(H.shape, matvec=matvec, dtype=H.dtype)
        op.setup_time = setup_time
        return op


# ======================================
# Enhanced Regularizer
# ======================================

class Regularizer:
    """
    Advanced regularization for pathological Hessians with preconditioning support.
    """

    def __init__(self, cfg):
        self.cfg = cfg

        # Base parameters (unchanged)
        self.sigma = getattr(cfg, "reg_sigma", 1e-8)
        self.sigma_min = getattr(cfg, "reg_sigma_min", 1e-12)
        self.sigma_max = getattr(cfg, "reg_sigma_max", 1e6)
        self.target_cond = getattr(cfg, "reg_target_cond", 1e12)
        self.min_eig_thresh = getattr(cfg, "reg_min_eig_thresh", 1e-8)
        self.mode = getattr(cfg, "reg_mode", "EIGEN_MOD")

        # Adaptation controls (unchanged)
        self.adapt_factor = getattr(cfg, "reg_adapt_factor", 2.0)
        self.model_quality_threshold_low = getattr(cfg, "reg_model_quality_low", 0.5)
        self.model_quality_threshold_high = getattr(cfg, "reg_model_quality_high", 2.0)
        self.model_quality_factor_inc = getattr(cfg, "reg_model_quality_inc", 1.5)
        self.model_quality_factor_dec = getattr(cfg, "reg_model_quality_dec", 0.7)

        # History (unchanged)
        self.sigma_history: list[float] = []
        self.cond_history: list[float] = []
        self.model_quality_history: list[float] = []
        self.success_history: list[bool] = []

        # Iterative settings (unchanged)
        self.iter_tol = getattr(cfg, "iter_tol", 1e-6)
        self.max_iter = getattr(cfg, "max_iter", 1000)
        self.k_eigs = getattr(cfg, "k_eigs", 10)

        # NEW: Preconditioning settings
        self.use_preconditioning = getattr(cfg, "use_preconditioning", True)
        self.precond_type = getattr(cfg, "precond_type", "auto")  # 'auto', 'ilu', 'amg', 'shift_invert', 'none'
        self.ilu_drop_tol = getattr(cfg, "ilu_drop_tol", 1e-4)
        self.ilu_fill_factor = getattr(cfg, "ilu_fill_factor", 10)
        self.amg_threshold = getattr(cfg, "amg_threshold", 5000)  # Use AMG for matrices larger than this
        self.shift_invert_mode = getattr(cfg, "shift_invert_mode", "buckling")  # 'normal', 'buckling', 'cayley'
        self.adaptive_precond = getattr(cfg, "adaptive_precond", True)  # Switch preconditioners based on problem

        # Preconditioning telemetry
        self.precond_stats = {
            "setup_times": [],
            "solve_iterations": [],
            "convergence_failures": 0,
            "type_switches": 0
        }

    # --------------------------- Enhanced matrix analysis ---------------------------

    def _analyze_matrix(self, H: Union[np.ndarray, sp.spmatrix]) -> Dict:
        """Enhanced matrix analysis with preconditioned eigenvalue computations."""
        n = H.shape[0]
        is_sparse = sp.issparse(H)
        density = (H.nnz / (n * n)) if is_sparse else 1.0
        use_iterative = is_sparse or (n > 500 and density < 0.1)
        
        # Determine preconditioner strategy
        precond_type = self._select_preconditioner_type(H, n, density)
        
        try:
            if use_iterative:
                return self._analyze_matrix_iterative(H, precond_type)
            else:
                return self._analyze_matrix_direct(H)
        except Exception as e:
            logging.warning(f"Enhanced matrix analysis failed: {e}. Using fallback.")
            return self._fallback_analysis(H)

    def _select_preconditioner_type(self, H: Union[np.ndarray, sp.spmatrix], n: int, density: float) -> str:
        """Automatically select the best preconditioning strategy."""
        if not self.use_preconditioning or self.precond_type == "none":
            return "none"
        
        if self.precond_type != "auto":
            return self.precond_type
        
        # Adaptive selection based on problem characteristics
        is_sparse = sp.issparse(H)
        
        if not is_sparse or n < 100:
            return "none"  # Direct methods work fine
        elif n > self.amg_threshold and HAS_PYAMG and density < 0.1:
            return "amg"  # Large sparse problems benefit from multilevel
        elif density > 0.5:
            return "ilu"  # Fairly dense sparse matrices
        else:
            return "shift_invert"  # Good for eigenvalue problems near zero

    def _analyze_matrix_iterative(self, H: Union[np.ndarray, sp.spmatrix], precond_type: str) -> Dict:
        """Analyze matrix using preconditioned iterative methods."""
        n = H.shape[0]
        precond = None
        setup_time = 0.0
        
        # Create preconditioner
        if precond_type == "ilu":
            precond = PreconditionerFactory.create_ilu_preconditioner(
                H, self.ilu_drop_tol, self.ilu_fill_factor, self.sigma
            )
            setup_time = getattr(precond, 'setup_time', 0.0)
        elif precond_type == "amg":
            precond = PreconditionerFactory.create_amg_preconditioner(H)
            setup_time = getattr(precond, 'setup_time', 0.0) if precond else 0.0
        elif precond_type == "shift_invert":
            # For eigenvalue problems, we'll use shift-invert in the eigsh call
            pass
        
        try:
            if precond_type == "shift_invert":
                # Use shift-and-invert for better convergence near zero
                min_eig, min_vec = eigsh(H, k=1, sigma=0.0, mode=self.shift_invert_mode,
                                       tol=self.iter_tol, maxiter=self.max_iter)
                max_eig, max_vec = eigsh(H, k=1, which="LA", 
                                       tol=self.iter_tol, maxiter=self.max_iter)
            else:
                # Use regular preconditioned eigenvalue solve
                min_eig, min_vec = eigsh(H, k=1, which="SA", M=precond,
                                       tol=self.iter_tol, maxiter=self.max_iter)
                max_eig, max_vec = eigsh(H, k=1, which="LA", M=precond,
                                       tol=self.iter_tol, maxiter=self.max_iter)
            
            min_eig = float(min_eig[0])
            max_eig = float(max_eig[0])
            cond_num = max_eig / max(abs(min_eig), 1e-16) if max_eig > 1e-16 else np.inf

            # Compute more eigenvalues for inertia/rank with preconditioning
            k_sample = min(self.k_eigs, n - 1)
            if precond_type == "shift_invert":
                small_eigs, small_vecs = eigsh(H, k=k_sample, sigma=0.0, mode=self.shift_invert_mode,
                                             tol=self.iter_tol, maxiter=self.max_iter)
            else:
                small_eigs, small_vecs = eigsh(H, k=k_sample, which="SA", M=precond,
                                             tol=self.iter_tol, maxiter=self.max_iter)
            
            rank = n - int(np.sum(np.abs(small_eigs) < 1e-12))
            rank_def = n - rank
            pos_eigs = int(np.sum(small_eigs > 1e-12))
            neg_eigs = int(np.sum(small_eigs < -1e-12))
            zero_eigs = k_sample - pos_eigs - neg_eigs
            
            # Extrapolate to full matrix (approximate)
            if k_sample < n:
                remaining = n - k_sample
                zero_eigs += remaining  # Conservative estimate
            
            self.precond_stats["setup_times"].append(setup_time)
            
            return {
                "eigvals": small_eigs,
                "eigvecs": small_vecs,
                "min_eig": min_eig,
                "max_eig": max_eig,
                "cond_num": cond_num,
                "rank_def": rank_def,
                "inertia": (pos_eigs, neg_eigs, zero_eigs),
                "is_psd": min_eig >= -1e-12,
                "is_singular": rank_def > 0,
                "precond_type": precond_type,
                "precond_setup_time": setup_time,
            }
            
        except ArpackNoConvergence as e:
            self.precond_stats["convergence_failures"] += 1
            logging.warning(f"Preconditioned eigensolve failed: {e}. Trying fallback.")
            
            # Try with relaxed tolerance and different preconditioner
            if precond_type != "none" and self.adaptive_precond:
                self.precond_stats["type_switches"] += 1
                return self._analyze_matrix_iterative(H, "none")  # Fallback to no preconditioning
            else:
                # Last resort: very relaxed solve
                min_eig, _ = eigsh(H, k=1, which="SA", tol=self.iter_tol * 100, maxiter=self.max_iter * 3)
                max_eig, _ = eigsh(H, k=1, which="LA", tol=self.iter_tol * 100, maxiter=self.max_iter * 3)
                return self._create_basic_analysis(float(min_eig[0]), float(max_eig[0]), n, precond_type)

    def _analyze_matrix_direct(self, H: Union[np.ndarray, sp.spmatrix]) -> Dict:
        """Direct eigenvalue analysis for small/dense matrices."""
        H_dense = H if isinstance(H, np.ndarray) else H.toarray()
        eigvals, eigvecs = la.eigh(H_dense)
        
        min_eig = float(np.min(eigvals))
        max_eig = float(np.max(eigvals))
        cond_num = max_eig / max(abs(min_eig), 1e-16) if max_eig > 1e-16 else np.inf
        
        n = len(eigvals)
        rank = int(np.sum(np.abs(eigvals) > 1e-12))
        rank_def = n - rank
        pos_eigs = int(np.sum(eigvals > 1e-12))
        neg_eigs = int(np.sum(eigvals < -1e-12))
        zero_eigs = int(np.sum(np.abs(eigvals) <= 1e-12))
        
        return {
            "eigvals": eigvals,
            "eigvecs": eigvecs,
            "min_eig": min_eig,
            "max_eig": max_eig,
            "cond_num": cond_num,
            "rank_def": rank_def,
            "inertia": (pos_eigs, neg_eigs, zero_eigs),
            "is_psd": min_eig >= -1e-12,
            "is_singular": rank_def > 0,
            "precond_type": "none",
            "precond_setup_time": 0.0,
        }

    def _create_basic_analysis(self, min_eig: float, max_eig: float, n: int, precond_type: str) -> Dict:
        """Create minimal analysis when full eigendecomposition fails."""
        cond_num = max_eig / max(abs(min_eig), 1e-16) if max_eig > 1e-16 else np.inf
        
        return {
            "eigvals": np.array([min_eig, max_eig]),
            "eigvecs": np.eye(n)[:, :2] if n >= 2 else np.eye(n),
            "min_eig": min_eig,
            "max_eig": max_eig,
            "cond_num": cond_num,
            "rank_def": 1 if abs(min_eig) < 1e-12 else 0,
            "inertia": (1, 0, 1) if min_eig >= 0 else (0, 1, 1),  # Rough estimate
            "is_psd": min_eig >= -1e-12,
            "is_singular": abs(min_eig) < 1e-12,
            "precond_type": precond_type,
            "precond_setup_time": 0.0,
        }

    def _fallback_analysis(self, H: Union[np.ndarray, sp.spmatrix]) -> Dict:
        """Emergency fallback when all analysis methods fail."""
        n = H.shape[0]
        return {
            "eigvals": np.zeros(n),
            "eigvecs": np.eye(n) if not sp.issparse(H) else None,
            "min_eig": 0.0,
            "max_eig": 0.0,
            "cond_num": np.inf,
            "rank_def": n,
            "inertia": (0, 0, n),
            "is_psd": False,
            "is_singular": True,
            "precond_type": "none",
            "precond_setup_time": 0.0,
        }

    # --------------------------- Enhanced regularization strategies ---------------------------

    def _eigenvalue_modification(
        self, H: Union[np.ndarray, sp.spmatrix], analysis: Dict
    ) -> Tuple[Union[np.ndarray, sp.spmatrix], RegInfo]:
        """Enhanced eigenvalue modification with preconditioning."""
        is_sparse = sp.issparse(H)
        threshold = max(self.min_eig_thresh, self.sigma)
        nnz_before = H.nnz if is_sparse else H.size
        precond_type = analysis.get("precond_type", "none")

        if is_sparse or H.shape[0] > 500:
            try:
                # Use the same preconditioning strategy as in analysis
                if precond_type == "shift_invert":
                    small_eigs, small_vecs = eigsh(
                        H, k=self.k_eigs, sigma=0.0, mode=self.shift_invert_mode,
                        tol=self.iter_tol, maxiter=self.max_iter
                    )
                elif precond_type in ["ilu", "amg"]:
                    if precond_type == "ilu":
                        precond = PreconditionerFactory.create_ilu_preconditioner(
                            H, self.ilu_drop_tol, self.ilu_fill_factor, self.sigma
                        )
                    else:
                        precond = PreconditionerFactory.create_amg_preconditioner(H)
                    
                    small_eigs, small_vecs = eigsh(
                        H, k=self.k_eigs, which="SA", M=precond,
                        tol=self.iter_tol, maxiter=self.max_iter
                    )
                else:
                    small_eigs, small_vecs = eigsh(
                        H, k=self.k_eigs, which="SA", 
                        tol=self.iter_tol, maxiter=self.max_iter
                    )
                
                small_mask = small_eigs < threshold
                if np.any(small_mask):
                    mod_eigs = np.where(small_mask, self.sigma, small_eigs)
                    V = small_vecs[:, small_mask]
                    D = mod_eigs[small_mask] - small_eigs[small_mask]
                    if is_sparse:
                        # Low-rank sparse update
                        update = sp.csr_matrix((V.T * D).T @ V)
                        H_reg = H + update
                    else:
                        H_reg = H + V @ np.diag(D) @ V.T
                else:
                    H_reg = H
                    
            except ArpackNoConvergence as e:
                logging.warning(f"Preconditioned eigenvalue modification failed: {e}. Falling back to Tikhonov.")
                return self._adaptive_tikhonov(H, analysis)
        else:
            # Use direct method for small matrices
            eigvals, eigvecs = analysis["eigvals"], analysis["eigvecs"]
            eigvals_mod = eigvals.copy()
            mask = eigvals_mod < threshold
            eigvals_mod[mask] = self.sigma
            H_reg = eigvecs @ np.diag(eigvals_mod) @ eigvecs.T
            if not np.allclose(H_reg, H_reg.T, rtol=1e-10, atol=1e-10):
                H_reg = 0.5 * (H_reg + H_reg.T)

        post = self._analyze_matrix(H_reg)
        nnz_after = H_reg.nnz if sp.issparse(H_reg) else H_reg.size
        
        return H_reg, RegInfo(
            mode="EIGEN_MOD",
            sigma=self.sigma,
            cond_before=analysis["cond_num"],
            cond_after=post["cond_num"],
            min_eig_before=analysis["min_eig"],
            min_eig_after=post["min_eig"],
            rank_def=analysis["rank_def"],
            inertia_before=analysis["inertia"],
            inertia_after=post["inertia"],
            nnz_before=nnz_before,
            nnz_after=nnz_after,
            precond_type=precond_type,
            precond_setup_time=analysis.get("precond_setup_time", 0.0),
        )

    # --------------------------- Rest of the methods remain the same ---------------------------
    # (regularize, _adapt_sigma, _inertia_correction, _spectral_regularization, 
    #  _adaptive_tikhonov, get_statistics, reset, etc.)

    def regularize(
        self,
        H: Union[np.ndarray, sp.spmatrix],
        iteration: int = 0,
        model_quality: Optional[float] = None,
        constraint_count: int = 0,
        grad_norm: Optional[float] = None,
    ) -> Tuple[Union[np.ndarray, sp.spmatrix], RegInfo]:
        """Main regularization entry point with enhanced preconditioning."""
        # Validate
        if H.shape[0] != H.shape[1]:
            raise ValueError(f"Hessian must be square, got shape {H.shape}")
        was_sparse = sp.issparse(H)
        n = H.shape[0]
        if n == 0:
            return H, RegInfo(
                mode="TIKHONOV", sigma=0.0, cond_before=1.0, cond_after=1.0,
                min_eig_before=1.0, min_eig_after=1.0, rank_def=0,
                inertia_before=(0, 0, 0), inertia_after=(0, 0, 0),
                nnz_before=0, nnz_after=0,
            )

        # Ensure symmetry
        if not was_sparse:
            if not np.allclose(H, H.T, rtol=1e-10, atol=1e-10):
                logging.warning("Hessian is not symmetric; symmetrizing")
                H = 0.5 * (H + H.T)
        else:
            diff = H - H.T
            if diff.nnz > 0 and not np.allclose(diff.data, 0, rtol=1e-10, atol=1e-10):
                logging.warning("Sparse Hessian is not symmetric; symmetrizing")
                H = 0.5 * (H + H.T)

        nnz_before = H.nnz if was_sparse else H.size

        # Enhanced spectral analysis with preconditioning
        self.k_eigs = min(self.k_eigs, max(10, n // 10))
        analysis = self._analyze_matrix(H)

        # Adapt σ (same as before)
        self._adapt_sigma(analysis, model_quality, iteration, grad_norm)

        # Choose and apply strategy (same logic, but eigenvalue_modification is enhanced)
        if self.mode == "EIGEN_MOD":
            H_reg, info = self._eigenvalue_modification(H, analysis)
        elif self.mode == "INERTIA_FIX":
            if not (0 <= constraint_count < n):
                raise ValueError(f"Invalid constraint_count {constraint_count}; must be in [0, {n-1}]")
            H_reg, info = self._inertia_correction(H, analysis, constraint_count)
        elif self.mode == "SPECTRAL":
            H_reg, info = self._spectral_regularization(H, analysis)
        else:  # 'TIKHONOV'
            H_reg, info = self._adaptive_tikhonov(H, analysis)

        # Update history and return (same as before)
        self.sigma_history.append(info.sigma)
        self.cond_history.append(info.cond_after)
        if model_quality is not None:
            self.model_quality_history.append(model_quality)

        if was_sparse and not sp.issparse(H_reg):
            orig_fmt = H.getformat()
            H_reg = self._cast_to_format(H_reg, orig_fmt)

        info.nnz_before = nnz_before
        info.nnz_after = H_reg.nnz if sp.issparse(H_reg) else H_reg.size
        return H_reg, info

    def _adapt_sigma(
        self,
        analysis: Dict,
        model_quality: Optional[float],
        iteration: int,
        grad_norm: Optional[float] = None,
    ) -> None:
        """Adapt σ (unchanged from original)."""
        if analysis["cond_num"] > self.target_cond:
            self.sigma = min(self.sigma_max, self.sigma * self.adapt_factor)
        elif analysis["cond_num"] < self.target_cond * 0.1 and analysis["is_psd"]:
            self.sigma = max(self.sigma_min, self.sigma / self.adapt_factor)

        if model_quality is not None and len(self.model_quality_history) > 0:
            k = min(3, len(self.model_quality_history))
            weights = np.exp(-np.arange(k) / 2.0)
            recent_quality = float(np.average(self.model_quality_history[-k:], weights=weights[::-1]))
            if model_quality < recent_quality * self.model_quality_threshold_low:
                self.sigma = min(self.sigma_max, self.sigma * self.model_quality_factor_inc)
            elif model_quality > recent_quality * self.model_quality_threshold_high:
                self.sigma = max(self.sigma_min, self.sigma * self.model_quality_factor_dec)

        if grad_norm is not None:
            self.sigma = max(self.sigma, 1e-6 * float(grad_norm))

        if iteration < 5:
            self.sigma = max(self.sigma, 1e-6)

    def _inertia_correction(
        self, H: Union[np.ndarray, sp.spmatrix], analysis: Dict, m_eq: int
    ) -> Tuple[Union[np.ndarray, sp.spmatrix], RegInfo]:
        """Enhanced inertia correction with preconditioning support."""
        is_sparse = sp.issparse(H)
        n = H.shape[0]
        nnz_before = H.nnz if is_sparse else H.size
        target_pos = max(1, n - m_eq)
        target_neg = min(m_eq, n - 1)
        precond_type = analysis.get("precond_type", "none")

        if is_sparse or n > 500:
            try:
                # Use enhanced eigenvalue computation for negative spectrum
                if precond_type == "shift_invert":
                    neg_eigs, neg_vecs = eigsh(
                        H, k=target_neg + 10, sigma=-self.sigma, mode=self.shift_invert_mode,
                        tol=self.iter_tol, maxiter=self.max_iter
                    )
                elif precond_type in ["ilu", "amg"]:
                    if precond_type == "ilu":
                        precond = PreconditionerFactory.create_ilu_preconditioner(
                            H, self.ilu_drop_tol, self.ilu_fill_factor, self.sigma
                        )
                    else:
                        precond = PreconditionerFactory.create_amg_preconditioner(H)
                    
                    neg_eigs, neg_vecs = eigsh(
                        H, k=target_neg + 10, which="SA", M=precond,
                        tol=self.iter_tol, maxiter=self.max_iter
                    )
                else:
                    neg_eigs, neg_vecs = eigsh(
                        H, k=target_neg + 10, which="SA",
                        tol=self.iter_tol, maxiter=self.max_iter
                    )
                
                current_neg = int(np.sum(neg_eigs < -1e-12))
                if current_neg > target_neg:
                    excess = current_neg - target_neg
                    V = neg_vecs[:, :excess]
                    D = 2.0 * np.abs(neg_eigs[:excess]) + self.sigma
                    if is_sparse:
                        update = sp.csr_matrix((V.T * D).T @ V)
                        H_reg = H + update
                    else:
                        H_reg = H + V @ np.diag(D) @ V.T
                else:
                    H_reg = H
            except ArpackNoConvergence as e:
                logging.warning(f"Preconditioned inertia correction failed: {e}. Falling back to Tikhonov.")
                return self._adaptive_tikhonov(H, analysis)
        else:
            # Direct method for small matrices (unchanged)
            eigvals, eigvecs = analysis["eigvals"], analysis["eigvecs"]
            idx = np.argsort(eigvals)
            eigvals_sorted = eigvals[idx]
            eigvecs_sorted = eigvecs[:, idx]

            eigvals_mod = eigvals_sorted.copy()
            neg_idx = np.where(eigvals_mod < 0)[0]
            flip_count = min(len(neg_idx), target_pos - int(np.sum(eigvals_mod > self.sigma)))
            if flip_count > 0:
                eigvals_mod[neg_idx[-flip_count:]] = self.sigma

            if target_neg > 0:
                pos_idx = np.where(eigvals_mod > 0)[0]
                current_neg = int(np.sum(eigvals_mod < -self.sigma))
                if current_neg < target_neg and len(pos_idx) > target_pos:
                    extra = min(len(pos_idx) - target_pos, target_neg - current_neg)
                    if extra > 0:
                        eigvals_mod[pos_idx[:extra]] = -self.sigma

            tiny = np.abs(eigvals_mod) < self.sigma
            eigvals_mod[tiny] = np.sign(eigvals_mod[tiny]) * self.sigma
            eigvals_mod[eigvals_mod == 0] = self.sigma

            H_reg = eigvecs_sorted @ np.diag(eigvals_mod) @ eigvecs_sorted.T
            if not np.allclose(H_reg, H_reg.T, rtol=1e-10, atol=1e-10):
                H_reg = 0.5 * (H_reg + H_reg.T)

        post = self._analyze_matrix(H_reg)
        nnz_after = H_reg.nnz if sp.issparse(H_reg) else H_reg.size
        return H_reg, RegInfo(
            mode="INERTIA_FIX",
            sigma=self.sigma,
            cond_before=analysis["cond_num"],
            cond_after=post["cond_num"],
            min_eig_before=analysis["min_eig"],
            min_eig_after=post["min_eig"],
            rank_def=analysis["rank_def"],
            inertia_before=analysis["inertia"],
            inertia_after=post["inertia"],
            nnz_before=nnz_before,
            nnz_after=nnz_after,
            precond_type=precond_type,
            precond_setup_time=analysis.get("precond_setup_time", 0.0),
        )

    def _spectral_regularization(
        self, H: Union[np.ndarray, sp.spmatrix], analysis: Dict
    ) -> Tuple[Union[np.ndarray, sp.spmatrix], RegInfo]:
        """SVD-based regularization (unchanged, but uses existing analysis)."""
        is_sparse = sp.issparse(H)
        nnz_before = H.nnz if is_sparse else H.size
        precond_type = analysis.get("precond_type", "none")
        
        try:
            if is_sparse or H.shape[0] > 500:
                U, s, Vt = svds(H, k=self.k_eigs, tol=self.iter_tol, maxiter=self.max_iter)
            else:
                U, s, Vt = la.svd(H if isinstance(H, np.ndarray) else H.toarray(), full_matrices=False)

            s_reg = np.maximum(s, self.sigma)
            if self.target_cond < np.inf:
                s_max = float(np.max(s_reg))
                min_allowed = s_max / self.target_cond
                s_reg = np.maximum(s_reg, min_allowed)

            H_reg = U @ np.diag(s_reg) @ Vt
            if is_sparse:
                H_reg = sp.csr_matrix(H_reg)
            if not np.allclose(H_reg, H_reg.T, rtol=1e-10, atol=1e-10):
                H_reg = 0.5 * (H_reg + H_reg.T)

            post = self._analyze_matrix(H_reg)
            nnz_after = H_reg.nnz if sp.issparse(H_reg) else H_reg.size
            return H_reg, RegInfo(
                mode="SPECTRAL",
                sigma=self.sigma,
                cond_before=analysis["cond_num"],
                cond_after=post["cond_num"],
                min_eig_before=analysis["min_eig"],
                min_eig_after=post["min_eig"],
                rank_def=analysis["rank_def"],
                inertia_before=analysis["inertia"],
                inertia_after=post["inertia"],
                nnz_before=nnz_before,
                nnz_after=nnz_after,
                precond_type=precond_type,
                precond_setup_time=analysis.get("precond_setup_time", 0.0),
            )
        except la.LinAlgError as e:
            logging.warning(f"SVD failed: {e}. Falling back to Tikhonov.")
            return self._adaptive_tikhonov(H, analysis)

    def _adaptive_tikhonov(
        self, H: Union[np.ndarray, sp.spmatrix], analysis: Dict
    ) -> Tuple[Union[np.ndarray, sp.spmatrix], RegInfo]:
        """Adaptive Tikhonov regularization (unchanged)."""
        is_sparse = sp.issparse(H)
        nnz_before = H.nnz if is_sparse else H.size
        precond_type = analysis.get("precond_type", "none")

        if analysis["cond_num"] > self.target_cond:
            max_eig = analysis["max_eig"]
            target_min = max_eig / self.target_cond
            sigma_needed = max(target_min - analysis["min_eig"], self.sigma)
            sigma_use = min(sigma_needed, self.sigma_max)
        else:
            sigma_use = self.sigma

        if is_sparse:
            H_reg = H + sp.diags([sigma_use] * H.shape[0], format="csr")
        else:
            H_reg = H + sigma_use * np.eye(H.shape[0])

        post = self._analyze_matrix(H_reg)
        nnz_after = H_reg.nnz if sp.issparse(H_reg) else H_reg.size
        return H_reg, RegInfo(
            mode="TIKHONOV",
            sigma=sigma_use,
            cond_before=analysis["cond_num"],
            cond_after=post["cond_num"],
            min_eig_before=analysis["min_eig"],
            min_eig_after=post["min_eig"],
            rank_def=analysis["rank_def"],
            inertia_before=analysis["inertia"],
            inertia_after=post["inertia"],
            nnz_before=nnz_before,
            nnz_after=nnz_after,
            precond_type=precond_type,
            precond_setup_time=analysis.get("precond_setup_time", 0.0),
        )

    # --------------------------- Enhanced telemetry ---------------------------

    def get_statistics(self) -> Dict:
        """Enhanced telemetry including preconditioning statistics."""
        base_stats = {
            "current_sigma": self.sigma,
            "avg_condition": float(np.mean(self.cond_history)) if self.cond_history else 0.0,
            "sigma_adaptations": len(self.sigma_history),
            "mode": self.mode,
            "sigma_range": (
                (min(self.sigma_history), max(self.sigma_history)) if self.sigma_history else (0.0, 0.0)
            ),
        }
        
        # Add preconditioning statistics
        precond_stats = {
            "preconditioning_enabled": self.use_preconditioning,
            "current_precond_type": self.precond_type,
            "avg_precond_setup_time": float(np.mean(self.precond_stats["setup_times"])) if self.precond_stats["setup_times"] else 0.0,
            "convergence_failures": self.precond_stats["convergence_failures"],
            "precond_type_switches": self.precond_stats["type_switches"],
            "total_precond_setups": len(self.precond_stats["setup_times"]),
        }
        
        return {**base_stats, **precond_stats}

    def reset(self) -> None:
        """Reset adaptation and preconditioning history."""
        self.sigma_history.clear()
        self.cond_history.clear()
        self.model_quality_history.clear()
        self.success_history.clear()
        self.sigma = getattr(self.cfg, "reg_sigma", 1e-8)
        
        # Reset preconditioning stats
        self.precond_stats = {
            "setup_times": [],
            "solve_iterations": [],
            "convergence_failures": 0,
            "type_switches": 0
        }

    @staticmethod
    def _cast_to_format(A: np.ndarray, fmt: str) -> sp.spmatrix:
        """Cast dense array to sparse matrix format (unchanged)."""
        if fmt == "csc":
            return sp.csc_matrix(A)
        if fmt == "csr":
            return sp.csr_matrix(A)
        return sp.csr_matrix(A)


# ======================================
# Enhanced integration helper
# ======================================

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
        H, iteration=iteration, model_quality=model_quality, constraint_count=constraint_count
    )