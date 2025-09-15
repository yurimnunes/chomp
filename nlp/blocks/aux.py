# sqp_clean.py
# Clean, SOTA TR-SQP components with L2 trust region via σ-search.
# Sparse-safe, compact structure, and clear documentation.

from __future__ import annotations

# =========================
# Standard library
# =========================
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

# =========================
# Third-party
# =========================
import numpy as np
import piqp_cpp as piqp
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import ArpackNoConvergence, eigsh, svds

import ad as AD
from nlp.blocks.reg import Regularizer


# ======================================
# Enums
# ======================================
class RegMode(Enum):
    """Supported regularization strategies."""

    TIKHONOV = "tikhonov"
    EIGEN_MOD = "eigen_mod"
    INERTIA_FIX = "inertia_fix"
    SPECTRAL = "spectral"


def _cfg_to_piqp(cfg: "SQPConfig") -> piqp.PIQPSettings:
    """
    Map SQPConfig essentials into PIQP settings.
    Keep these minimal; tune the rest via PIQP itself if needed.
    """
    S = piqp.PIQPSettings()
    S.eps_abs = cfg.piqp_eps_abs
    S.eps_rel = cfg.piqp_eps_rel
    S.max_iter = cfg.piqp_max_iter
    S.verbose = cfg.piqp_verbose
    # Sensible scaling from TR radius (harmless defaults otherwise)
    S.rho_init = cfg.tr_delta0 * 1e-2
    S.delta_init = cfg.tr_delta0 * 1e-1
    S.rho_floor = 1e-12
    S.delta_floor = 1e-12
    return S


# ======================================
# Global configuration
# ======================================
@dataclass
class SQPConfig:
    """
    Global configuration for TR–SQP components.

    Notes
    -----
    • Keep fields stable; downstream managers read these dynamically.
    • Avoid duplicates: fields here are unique (earlier duplicates were removed).
    """

    # ---------------- Core toggles ----------------
    mode: str = "auto"  # {"auto","ip","sqp"}
    verbose: bool = True
    use_filter: bool = True
    use_line_search: bool = False
    use_trust_region: bool = True
    use_soc: bool = True
    use_funnel: bool = False
    use_watchdog: bool = True
    use_nonmonotone_ls: bool = False
    use_active_set_prediction: bool = False
    hessian_mode: str = "exact"  # {"exact","bfgs","lbfgs","hybrid","gn"}

    # ---------------- Tolerances ----------------
    tol_feas: float = 1e-5
    tol_stat: float = 1e-5
    tol_comp: float = 1e-5
    adaptive_tol: bool = True

    # ---------------- Filter (basic) ----------------
    filter_gamma_theta: float = 1e-5
    filter_gamma_f: float = 1e-5
    filter_theta_min: float = 1e-8
    filter_margin_min: float = 1e-5

    # Extended Filter (capacity / margins)
    filter_max_size: int = 100
    iter_scale_factor: float = 50.0
    switch_theta: float = 0.9
    switch_f: float = 1.0

    # ---------------- Line search ----------------
    ls_armijo_f: float = 1e-4
    ls_armijo_theta: float = 1e-4
    ls_backtrack: float = 0.5
    ls_max_iter: int = 25
    ls_nonmonotone_M: int = 5
    ls_min_alpha: float = 1e-12
    ls_use_wolfe: bool = False
    ls_wolfe_c: float = 0.9
    ls_theta_restoration: float = 1e2

    # ---------------- Second-Order Correction (SOC) ----------------
    soc_kappa: float = 0.8
    soc_tol: float = 1e-8
    soc_max_corrections: int = 2

    # ---------------- PIQP (QP backend) ----------------
    piqp_eps_abs: float = 1e-8
    piqp_eps_rel: float = 1e-8
    piqp_max_iter: int = 500
    piqp_verbose: bool = False

    # ---------------- Hessian options ----------------
    lbfgs_memory: int = 10
    bfgs_min_curv: float = 1e-8
    bfgs_powell_damp: bool = True

    # ---------------- Regularization (linear solves) ----------------
    reg_mode: str = "AUTO"  # {"EIGEN_MOD","TIKHONOV","INERTIA_FIX","SPECTRAL"}
    reg_sigma: float = 1e-8
    reg_target_cond: float = 1e12
    reg_adapt_factor: float = 2.0

    # ---------------- Funnel (optional) ----------------
    funnel_initial_tau: float = 1.0
    funnel_delta: float = 0.1
    funnel_sigma: float = 1e-4
    funnel_beta: float = 0.99
    funnel_kappa: float = 0.1
    funnel_min_tau: float = 1e-8
    funnel_max_history: int = 100


# ======================================
# Small helpers
# ======================================
def _clean_vec(v, m: int) -> np.ndarray:
    """
    Return a 1-D float array of length m with None/NaN/inf handled.
    - None -> zeros
    - wrong-length / scalar -> resized to length m
    - NaN -> 0, +inf/-inf -> large finite
    """
    if m <= 0:
        return np.zeros(0, dtype=float)
    if v is None:
        return np.zeros(m, dtype=float)
    a = np.asarray(v, dtype=float).ravel()
    if a.size != m:
        a = np.resize(a, m).astype(float, copy=False)
    return np.nan_to_num(a, nan=0.0, posinf=1e20, neginf=-1e20)


def _zero_mat(m: int, n: int):
    """CSR zero matrix of shape (m, n)."""
    return (
        sp.csr_matrix((m, n))
        if m > 0 and n > 0
        else sp.csr_matrix((max(m, 0), max(n, 0)))
    )


# ======================================
# AutoDiff Model
# ======================================
class Model:
    """
    Encapsulates objective, constraints, and derivatives for the SQP stack.
    Supports sparse Jacobians/Hessians (if `use_sparse=True`).
    """

    def __init__(
        self,
        f: Callable,
        c_ineq: List[Callable] | None = None,
        c_eq: List[Callable] | None = None,
        n: int | None = None,
        lb: np.ndarray | None = None,
        ub: np.ndarray | None = None,
        use_sparse: bool = False,
    ):
        """Initialize model with objective `f`, constraints, and dimension `n`."""
        if n is None or n <= 0:
            raise ValueError(f"Number of variables n must be positive, got {n}")
        if not callable(f):
            raise ValueError("Objective function f must be callable")
        if c_ineq is not None and not all(callable(ci) for ci in c_ineq):
            raise ValueError("All inequality constraints must be callable")
        if c_eq is not None and not all(callable(ce) for ce in c_eq):
            raise ValueError("All equality constraints must be callable")

        self.n = n
        self.f = f
        self.cI_funcs = c_ineq or []
        self.cE_funcs = c_eq or []
        self.m_ineq = len(self.cI_funcs)
        self.m_eq = len(self.cE_funcs)
        self.lb = lb
        self.ub = ub
        self.use_sparse = use_sparse
        self._cache: Dict[str, object] = {}
        self._cache_x: Optional[Tuple[float, ...]] = None
        self._compile_derivatives()

    def _compile_derivatives(self):
        """Compile symbolic derivatives via `ad`."""
        try:
            self.f_grad = AD.sym_grad(self.f, self.n)
            self.f_hess = AD.sym_hess(self.f, self.n)
            self.cI_grads = [AD.sym_grad(ci, self.n) for ci in self.cI_funcs]
            self.cI_hess = [AD.sym_hess(ci, self.n) for ci in self.cI_funcs]
            self.cE_grads = [AD.sym_grad(ce, self.n) for ce in self.cE_funcs]
            self.cE_hess = [AD.sym_hess(ce, self.n) for ce in self.cE_funcs]
        except Exception as e:
            logging.error(f"AD compilation failed: {e}")
            raise RuntimeError("Failed to compile derivatives")

    def eval_all(
        self, x: np.ndarray, components: Optional[List[str]] = None
    ) -> Dict[str, Union[float, np.ndarray, sp.spmatrix]]:
        """
        Evaluate requested components at `x`.

        Parameters
        ----------
        components : subset of {"f","g","H","cI","JI","cE","JE"}
        """
        if x.shape[0] != self.n:
            raise ValueError(f"Input x shape {x.shape} does not match n={self.n}")
        if not np.all(np.isfinite(x)):
            raise ValueError("Non-finite values in x")

        components = components or ["f", "g", "H", "cI", "JI", "cE", "JE"]
        valid = {"f", "g", "H", "cI", "JI", "cE", "JE"}
        if not set(components) <= valid:
            raise ValueError(
                f"Invalid components: {components}, must be subset of {sorted(valid)}"
            )

        x_key = tuple(x)
        if x_key == self._cache_x:
            cached = {k: v for k, v in self._cache.items() if k in components}
            if len(cached) == len(components):
                return cached  # full cache hit

        res: Dict[str, object] = {}

        # Objective
        if "f" in components:
            try:
                res["f"] = float(AD.val(self.f, x))
                if not np.isfinite(res["f"]):
                    logging.warning(f"Non-finite objective value: f={res['f']}")
                    res["f"] = float("inf")
            except Exception as e:
                logging.error(f"Objective evaluation failed: {e}")
                res["f"] = float("inf")

        # Gradient
        if "g" in components:
            try:
                g = np.asarray(self.f_grad(x), dtype=float)
                res["g"] = g if np.all(np.isfinite(g)) else np.zeros(self.n)
                if not np.all(np.isfinite(res["g"])):
                    logging.warning("Non-finite gradient values")
            except Exception as e:
                logging.error(f"Gradient evaluation failed: {e}")
                res["g"] = np.zeros(self.n)

        # Hessian
        if "H" in components:
            try:
                H = np.asarray(self.f_hess(x), dtype=float)
                if self.use_sparse:
                    H = sp.csr_matrix(H)
                # Symmetrize if needed
                if (sp.issparse(H) and (H - H.T).nnz != 0) or (
                    not sp.issparse(H) and not np.allclose(H, H.T, rtol=1e-10)
                ):
                    logging.debug("Symmetrizing objective Hessian")
                    H = 0.5 * (H + H.T)
                # Finite check
                if not np.all(np.isfinite(H.data if sp.issparse(H) else H)):
                    logging.warning("Non-finite Hessian values")
                    H = (
                        sp.eye(self.n, format="csr")
                        if self.use_sparse
                        else np.eye(self.n)
                    )
                res["H"] = H
            except Exception as e:
                logging.error(f"Hessian evaluation failed: {e}")
                res["H"] = (
                    sp.eye(self.n, format="csr") if self.use_sparse else np.eye(self.n)
                )

        # Inequalities
        if "cI" in components and self.cI_funcs:
            try:
                cI = np.array([AD.val(ci, x) for ci in self.cI_funcs], dtype=float)
                res["cI"] = cI if np.all(np.isfinite(cI)) else np.zeros(self.m_ineq)
                if not np.all(np.isfinite(res["cI"])):
                    logging.warning("Non-finite inequality constraint values")
            except Exception as e:
                logging.error(f"Inequality constraint evaluation failed: {e}")
                res["cI"] = np.zeros(self.m_ineq)
        else:
            res["cI"] = None

        if "JI" in components and self.cI_funcs:
            try:
                JI_dense = [np.asarray(gi(x), dtype=float) for gi in self.cI_grads]
                JI = (
                    sp.vstack([sp.csr_matrix(ji) for ji in JI_dense], format="csr")
                    if self.use_sparse
                    else np.vstack(JI_dense)
                )
                res["JI"] = (
                    JI
                    if np.all(np.isfinite(JI.data if sp.issparse(JI) else JI))
                    else (
                        sp.csr_matrix((self.m_ineq, self.n))
                        if self.use_sparse
                        else np.zeros((self.m_ineq, self.n))
                    )
                )
                if res["JI"] is not None and not np.all(
                    np.isfinite(res["JI"].data if sp.issparse(res["JI"]) else res["JI"])
                ):
                    logging.warning("Non-finite inequality Jacobian values")
            except Exception as e:
                logging.error(f"Inequality Jacobian evaluation failed: {e}")
                res["JI"] = (
                    sp.csr_matrix((self.m_ineq, self.n))
                    if self.use_sparse
                    else np.zeros((self.m_ineq, self.n))
                )
        else:
            res["JI"] = None

        # Equalities
        if "cE" in components and self.cE_funcs:
            try:
                cE = np.array([AD.val(ce, x) for ce in self.cE_funcs], dtype=float)
                res["cE"] = cE if np.all(np.isfinite(cE)) else np.zeros(self.m_eq)
                if not np.all(np.isfinite(res["cE"])):
                    logging.warning("Non-finite equality constraint values")
            except Exception as e:
                logging.error(f"Equality constraint evaluation failed: {e}")
                res["cE"] = np.zeros(self.m_eq)
        else:
            res["cE"] = None

        if "JE" in components and self.cE_funcs:
            try:
                JE_dense = [np.asarray(ge(x), dtype=float) for ge in self.cE_grads]
                JE = (
                    sp.vstack([sp.csr_matrix(je) for je in JE_dense], format="csr")
                    if self.use_sparse
                    else np.vstack(JE_dense)
                )
                res["JE"] = (
                    JE
                    if np.all(np.isfinite(JE.data if sp.issparse(JE) else JE))
                    else (
                        sp.csr_matrix((self.m_eq, self.n))
                        if self.use_sparse
                        else np.zeros((self.m_eq, self.n))
                    )
                )
                if res["JE"] is not None and not np.all(
                    np.isfinite(res["JE"].data if sp.issparse(res["JE"]) else res["JE"])
                ):
                    logging.warning("Non-finite equality Jacobian values")
            except Exception as e:
                logging.error(f"Equality Jacobian evaluation failed: {e}")
                res["JE"] = (
                    sp.csr_matrix((self.m_eq, self.n))
                    if self.use_sparse
                    else np.zeros((self.m_eq, self.n))
                )
        else:
            res["JE"] = None

        self._cache = res
        self._cache_x = x_key
        return res  # type: ignore[return-value]

    def lagrangian_hessian(
        self, x: np.ndarray, lam: np.ndarray, nu: np.ndarray
    ) -> Union[np.ndarray, sp.spmatrix]:
        """
        Robust ∇²_x L(x,λ,ν) = H_f(x) + Σ_i λ_i H_{cI_i}(x) + Σ_j ν_j H_{cE_j}(x).

        Hardening features:
        • Per-term np.errstate to avoid runtime floating warnings.
        • NaN/Inf → 0.0 sanitization + magnitude clipping.
        • Strict symmetrization for every piece.
        • Graceful handling of m_ineq==0 / m_eq==0 / missing callables.
        • Final "finite" check with identity fallback.
        • Optional tiny diagonal floor to prevent near-singular diagonals.

        Tunables (via self.cfg, optional):
        - multiplier_threshold (default 1e-8)
        - hess_clip_max (default 1e12)
        - hess_diag_floor (default 0.0; set >0 to enforce tiny diagonal)
        """
        import logging

        # --- config & shapes ---
        n = int(getattr(self, "n", x.shape[0]))
        mI = int(getattr(self, "m_ineq", 0))
        mE = int(getattr(self, "m_eq", 0))
        use_sparse = bool(getattr(self, "use_sparse", False))

        cfg = getattr(self, "cfg", None)
        multiplier_threshold = float(getattr(cfg, "multiplier_threshold", 1e-8))
        clip_max = float(getattr(cfg, "hess_clip_max", 1e12))
        diag_floor = float(getattr(cfg, "hess_diag_floor", 0.0))

        # --- validate inputs softly ---
        if x.shape[0] != n:
            raise ValueError(f"Incompatible x shape: expected ({n},), got {x.shape}")
        lam = np.asarray(lam, dtype=float).ravel()
        nu = np.asarray(nu, dtype=float).ravel()

        if lam.size != mI:
            logging.warning(
                f"[lagrangian_hessian] λ size {lam.size} != m_ineq {mI}; clipping to min."
            )
            lam = lam[:mI]
        if nu.size != mE:
            logging.warning(
                f"[lagrangian_hessian] ν size {nu.size} != m_eq {mE}; clipping to min."
            )
            nu = nu[:mE]

        if not (
            np.all(np.isfinite(x))
            and np.all(np.isfinite(lam))
            and np.all(np.isfinite(nu))
        ):
            raise ValueError("Non-finite values in x, lam, or nu")

        # --- helpers ---
        def _to_type(A):
            """Cast to configured storage type (sparse CSR or dense ndarray)."""
            if use_sparse:
                return (
                    A if sp.issparse(A) else sp.csr_matrix(np.asarray(A, dtype=float))
                )
            else:
                return A.toarray() if sp.issparse(A) else np.asarray(A, dtype=float)

        def _symmetrize(A):
            return (A + A.T) * 0.5 if sp.issparse(A) else 0.5 * (A + A.T)

        def _sanitize(A):
            """Replace NaN/Inf → 0, clip |A|, keep type."""
            if sp.issparse(A):
                data = A.data.copy()
                # NaN/Inf -> 0
                bad = ~np.isfinite(data)
                if np.any(bad):
                    data[bad] = 0.0
                # clip absurd magnitudes
                np.clip(data, -clip_max, clip_max, out=data)
                A = sp.csr_matrix((data, A.indices, A.indptr), shape=A.shape)
                return A
            else:
                B = np.asarray(A, dtype=float)
                # NaN/Inf -> 0
                if not np.isfinite(B).all():
                    B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
                # clip
                np.clip(B, -clip_max, clip_max, out=B)
                return B

        def _ensure_shape(A):
            """Ensure n×n; if mismatched, warn and skip by returning None."""
            if A.shape != (n, n):
                logging.warning(
                    f"[lagrangian_hessian] Hessian piece has shape {A.shape}, expected {(n,n)}; skipping."
                )
                return None
            return A

        def _add_piece(H_acc, weight, Hi_callable):
            if Hi_callable is None or abs(weight) <= multiplier_threshold:
                return H_acc
            try:
                with np.errstate(all="ignore"):
                    H_i = Hi_callable(x)
            except Exception as e:
                logging.warning(
                    f"[lagrangian_hessian] constraint Hessian raised {e}; skipping."
                )
                return H_acc

            H_i = _to_type(H_i)
            H_i = _ensure_shape(H_i)
            if H_i is None:
                return H_acc

            H_i = _sanitize(H_i)
            H_i = _symmetrize(H_i)

            return H_acc + weight * H_i

        # --- base Hessian (objective) ---
        try:
            with np.errstate(all="ignore"):
                H_base = self.eval_all(x, components=["H"]).get("H", None)
        except Exception as e:
            logging.warning(
                f"[lagrangian_hessian] eval_all failed to get H: {e}; using zeros."
            )
            H_base = None

        if H_base is None:
            H = sp.csr_matrix((n, n)) if use_sparse else np.zeros((n, n), dtype=float)
        else:
            H = _to_type(H_base)
            H = _ensure_shape(H)
            if H is None:
                H = (
                    sp.csr_matrix((n, n))
                    if use_sparse
                    else np.zeros((n, n), dtype=float)
                )
            H = _sanitize(H)
            H = _symmetrize(H)
            if use_sparse:
                H = H.tocsr()

        # --- add inequality pieces ---
        cI_hess_list = getattr(self, "cI_hess", None) or []
        if len(cI_hess_list) < lam.size:
            logging.warning(
                f"[lagrangian_hessian] cI_hess length {len(cI_hess_list)} < λ size {lam.size}; extras ignored."
            )
        for li, Hi in zip(lam, cI_hess_list):
            H = _add_piece(H, li, Hi)

        # --- add equality pieces ---
        cE_hess_list = getattr(self, "cE_hess", None) or []
        if len(cE_hess_list) < nu.size:
            logging.warning(
                f"[lagrangian_hessian] cE_hess length {len(cE_hess_list)} < ν size {nu.size}; extras ignored."
            )
        for ni, Hi in zip(nu, cE_hess_list):
            H = _add_piece(H, ni, Hi)

        # --- optional tiny diagonal floor to avoid near-singular diagonals ---
        if diag_floor > 0.0:
            if use_sparse:
                d = H.diagonal()
                fix = (np.abs(d) < diag_floor) | ~np.isfinite(d)
                if np.any(fix):
                    add = sp.diags(diag_floor * fix.astype(float))
                    H = (H + add).tocsr()
            else:
                d = np.diag(H)
                fix = (np.abs(d) < diag_floor) | ~np.isfinite(d)
                if np.any(fix):
                    H = H.copy()
                    idx = np.where(fix)[0]
                    H[idx, idx] = diag_floor

        # --- final sanity: finite or fallback ---
        if use_sparse:
            if not np.all(np.isfinite(H.data)):
                logging.warning(
                    "[lagrangian_hessian] non-finite entries after assembly; falling back to identity."
                )
                H = sp.eye(n, format="csr")
        else:
            if not np.all(np.isfinite(H)):
                logging.warning(
                    "[lagrangian_hessian] non-finite entries after assembly; falling back to identity."
                )
                H = np.eye(n)

        return H

    def constraint_violation(self, x: np.ndarray) -> float:
        """L1 norm of violations (cI⁺ + |cE|) scaled by problem size; robust to None."""
        if x.shape[0] != self.n:
            raise ValueError(f"Input x shape {x.shape} does not match n={self.n}")
        if not np.all(np.isfinite(x)):
            raise ValueError("Non-finite values in x")

        d = self.eval_all(x, components=["cI", "cE"])
        mI, mE = self.m_ineq, self.m_eq

        cI = _clean_vec(d.get("cI", None), mI) if mI > 0 else np.zeros(0, dtype=float)  # type: ignore[arg-type]
        cE = _clean_vec(d.get("cE", None), mE) if mE > 0 else np.zeros(0, dtype=float)  # type: ignore[arg-type]

        scale = max(1.0, self.n, mI + mE)
        theta = 0.0
        if mI > 0:
            theta += float(np.sum(np.maximum(0.0, cI))) / scale
        if mE > 0:
            theta += float(np.sum(np.abs(cE))) / scale

        if not np.isfinite(theta):
            logging.warning(f"Non-finite constraint violation: {theta}")
            theta = float("inf")
        return theta

    def kkt_residuals(
        self, x: np.ndarray, lam: np.ndarray, nu: np.ndarray
    ) -> Dict[str, float]:
        """Compute KKT residuals: stationarity, feasibility, and complementarity."""
        if (
            x.shape[0] != self.n
            or lam.shape[0] != self.m_ineq
            or nu.shape[0] != self.m_eq
        ):
            raise ValueError(
                f"Incompatible shapes: x={x.shape}, lam={lam.shape}, nu={nu.shape}"
            )
        if not (
            np.all(np.isfinite(x))
            and np.all(np.isfinite(lam))
            and np.all(np.isfinite(nu))
        ):
            raise ValueError("Non-finite values in x, lam, or nu")

        mI, mE, n = self.m_ineq, self.m_eq, self.n
        d = self.eval_all(x, components=["g", "JI", "cI", "JE", "cE"])
        g = np.asarray(d.get("g", np.zeros(n)), dtype=float).ravel()
        cI = _clean_vec(d.get("cI", None), mI) if mI > 0 else np.zeros(0, dtype=float)
        cE = _clean_vec(d.get("cE", None), mE) if mE > 0 else np.zeros(0, dtype=float)
        JI = d.get("JI", None)
        JE = d.get("JE", None)
        if JI is None and mI > 0:
            JI = _zero_mat(mI, n)
        if JE is None and mE > 0:
            JE = _zero_mat(mE, n)
        lam = _clean_vec(lam, mI) if mI > 0 else np.zeros(0, dtype=float)
        nu = _clean_vec(nu, mE) if mE > 0 else np.zeros(0, dtype=float)

        # Ensure non-negative multipliers for inequalities
        lam = np.maximum(lam, 0.0)

        # Stationarity: g + JIᵀ λ + JEᵀ ν
        rL = g.copy()
        if mI > 0:
            rL += (JI.T @ lam) if sp.issparse(JI) else (np.asarray(JI, float).T @ lam)
        if mE > 0:
            rL += (JE.T @ nu) if sp.issparse(JE) else (np.asarray(JE, float).T @ nu)

        # Scaling factor based on gradient norm
        scale_g = max(1.0, float(np.linalg.norm(g, ord=np.inf)))
        stat_inf = float(np.linalg.norm(rL, ord=np.inf)) / scale_g
        ineq_inf = (
            float(np.linalg.norm(np.maximum(0.0, cI), ord=np.inf)) / scale_g
            if mI > 0
            else 0.0
        )
        eq_inf = float(np.linalg.norm(cE, ord=np.inf)) / scale_g if mE > 0 else 0.0

        # Complementarity: max(|λ_i * cI_i|) for inequality constraints
        comp_inf = 0.0
        if mI > 0:
            comp_terms = np.abs(lam * cI)  # λ_i * g_i should be ≈ 0
            comp_inf = (
                float(np.max(comp_terms)) / scale_g if comp_terms.size > 0 else 0.0
            )

        residuals = {
            "stat": stat_inf,
            "ineq": ineq_inf,
            "eq": eq_inf,
            "comp": comp_inf,
        }

        # Log raw residuals for debugging
        if not all(np.isfinite(v) for v in residuals.values()):
            logging.warning(
                f"Non-finite KKT residuals: {residuals}, raw comp terms: {comp_terms if mI > 0 else []}"
            )
            residuals = {k: float("inf") for k in residuals}
        elif comp_inf > 1e-4:  # Log large complementarity terms
            logging.debug(f"KKT comp terms: {comp_terms if mI > 0 else []}")

        return residuals

    def reset_cache(self):
        """Clear evaluation cache."""
        self._cache.clear()
        self._cache_x = None


# ======================================
# Restoration (weighted L1 feasibility)
# ======================================
class RestorationManager:
    """
    Phase-1, TR-bounded elastic restoration:
        minimize  (eps/2)||p||_2^2 + wE * 1^T sE + wI * 1^T sI
        s.t.      JE p - sE <= -cE,   -JE p - sE <=  cE,   sE >= 0
                  JI p - sI <= -cI,                          sI >= 0
                  lb_p <= p <= ub_p    (box from TR and model bounds)

    Implemented as a QP in variables z = [p; sE; sI], solved with PIQP.
    """

    def __init__(self, cfg: SQPConfig):
        self.cfg = cfg
        self.wE = 1.0
        self.wI = 1.0
        self._piqp_solver: Optional[piqp.PIQPSolver] = None
        self._piqp_pattern_key: Optional[Tuple] = None
        self._last_z: Optional[np.ndarray] = None  # warm start cache

    # small sparse helpers
    @staticmethod
    def _I(n: int):
        return sp.identity(n, format="csc")

    @staticmethod
    def _Z(m: int, n: int):
        return sp.csc_matrix((m, n))

    def try_restore(self, model: "Model", x: np.ndarray, tr_radius: float):
        """Attempt a restoration step; return (p, info) or (None, info) if not improving θ."""
        data = model.eval_all(x)
        n = x.size

        # Residuals/Jacobians; normalize None -> empty
        cE = data.get("cE")
        JE = data.get("JE")
        cI = data.get("cI")
        JI = data.get("JI")
        if cE is None:
            cE = np.zeros(0)
        if cI is None:
            cI = np.zeros(0)
        if JE is None:
            JE = self._Z(0, n)
        if JI is None:
            JI = self._Z(0, n)
        if sp.issparse(JE) and JE.format != "csc":
            JE = JE.tocsc()
        if sp.issparse(JI) and JI.format != "csc":
            JI = JI.tocsc()

        mE, mI = cE.size, cI.size
        N = n + mE + mI  # z = [p; sE; sI]

        # TR radius fallback
        if (tr_radius is None) or (tr_radius <= 0.0):
            tr_radius = getattr(self.cfg, "tr_delta0", 1.0)

        # Optional bounds on x → bounds on p
        lb = getattr(model, "lb", None)
        ub = getattr(model, "ub", None)
        if lb is None:
            lb = -np.inf * np.ones(n)
        if ub is None:
            ub = +np.inf * np.ones(n)
        lb_p = np.maximum(lb - x, -tr_radius * np.ones(n))
        ub_p = np.minimum(ub - x, +tr_radius * np.ones(n))

        # ---------- QP matrices (sparse) ----------
        eps = 1e-8
        P_blocks = [eps * self._I(n)]
        if mE > 0:
            P_blocks.append(self._Z(mE, mE))
        if mI > 0:
            P_blocks.append(self._Z(mI, mI))
        P = sp.block_diag(P_blocks, format="csc")

        if (mE + mI) > 0:
            q = np.concatenate(
                [
                    np.zeros(n),
                    (self.wE * np.ones(mE)) if mE > 0 else np.zeros(0),
                    (self.wI * np.ones(mI)) if mI > 0 else np.zeros(0),
                ]
            )
        else:
            q = np.zeros(n)

        G_rows, h_parts = [], []

        # Equalities as elastic inequalities + nonnegativity of sE
        if mE > 0:
            G1 = sp.hstack([JE, -self._I(mE), self._Z(mE, mI)], format="csc")
            G2 = sp.hstack([-JE, -self._I(mE), self._Z(mE, mI)], format="csc")
            Ge0 = sp.hstack(
                [self._Z(mE, n), -self._I(mE), self._Z(mE, mI)], format="csc"
            )
            G_rows += [G1, G2, Ge0]
            h_parts += [-cE, cE, np.zeros(mE)]

        # Inequalities elastic + nonnegativity of sI
        if mI > 0:
            Gi = sp.hstack([JI, self._Z(mI, mE), -self._I(mI)], format="csc")
            Gi0 = sp.hstack([self._Z(mI, n + mE), -self._I(mI)], format="csc")
            G_rows += [Gi, Gi0]
            h_parts += [-cI, np.zeros(mI)]

        # Box on p
        Gp_up = sp.hstack([self._I(n), self._Z(n, mE), self._Z(n, mI)], format="csc")
        Gp_lo = sp.hstack([-self._I(n), self._Z(n, mE), self._Z(n, mI)], format="csc")
        G_rows += [Gp_up, Gp_lo]
        h_parts += [ub_p, -lb_p]

        G = sp.vstack(G_rows, format="csc") if G_rows else self._Z(0, N)
        h = np.concatenate(h_parts) if h_parts else np.zeros(0)

        # ---------- PIQP setup/update + warm start ----------
        pattern_key = (P.shape, int(P.nnz), G.shape, int(G.nnz))
        reuse = (self._piqp_solver is not None) and (
            pattern_key == self._piqp_pattern_key
        )

        if not reuse:
            settings = _cfg_to_piqp(self.cfg)
            solver = piqp.PIQPSolver(settings)
            solver.setup(
                P, q, None, None, G, h
            )  # all as inequalities; no equalities block
            self._piqp_solver = solver
            self._piqp_pattern_key = pattern_key
            self._last_z = None
        else:
            self._piqp_solver.update_values(P, q, None, None, G, h, same_pattern=True)

        # Warm start via PIQP.warm_start(x, y, z, s, same_pattern)
        m_eq = 0
        m_G = G.shape[0]

        def _fit(v, size):
            if size == 0:
                return None
            if v is None:
                return np.zeros(size, dtype=float)
            v = np.asarray(v, dtype=float).ravel()
            if v.size == size:
                return v
            out = np.zeros(size, dtype=float)
            out[: min(size, v.size)] = v[: min(size, v.size)]
            return out

        self._piqp_solver.use_last_as_warm_start(False)
        if (self._last_z is not None) and (self._last_z.size == N):
            self._piqp_solver.warm_start(
                _fit(self._last_z, N),
                _fit(None, m_eq),
                _fit(None, m_G),
                _fit(None, m_G),
                True,
            )

        # Solve
        self._piqp_solver.solve()
        z = np.asarray(self._piqp_solver.get_x(), dtype=float)
        if z is None or z.size != N or not np.all(np.isfinite(z)):
            return None, {"ok": False, "reason": "restore_solve_failed"}

        self._last_z = z.copy()
        p = z[:n].copy()

        # ---------- accept only if θ decreases ----------
        theta_old = model.constraint_violation(x)
        theta_new = model.constraint_violation(x + p)
        ok = np.isfinite(theta_new) and (theta_new < theta_old - 1e-12)

        # Adaptive weights if no improvement
        if not ok:
            nE = float(np.linalg.norm(cE, 1)) if mE > 0 else 0.0
            nI = float(np.linalg.norm(cI, 1)) if mI > 0 else 0.0
            if nE >= nI:
                self.wE = min(self.wE * 2.0, 1e6)
            else:
                self.wI = min(self.wI * 2.0, 1e6)
            return None, {"ok": False, "reason": "restore_no_improvement"}

        return p, {
            "ok": True,
            "theta_old": float(theta_old),
            "theta_new": float(theta_new),
            "wE": float(self.wE),
            "wI": float(self.wI),
        }


# ======================================
# Hessian Manager (exact / BFGS / L-BFGS)
# ======================================
class HessianManager:
    """
    Manage Hessian approximations for SQP: exact, BFGS, L-BFGS, and hybrid.
    Supports sparse matrices and efficient L-BFGS matvecs.
    """

    def __init__(
        self, n: int, cfg: "SQPConfig", regularizer: Optional["Regularizer"] = None
    ):
        self.cfg = cfg
        self.n = n
        self.regularizer = regularizer
        self.use_sparse = getattr(cfg, "use_sparse_hessian", False)
        self.H = sp.eye(n, format="csr") if self.use_sparse else np.eye(n)
        self.memory: List[Tuple[np.ndarray, np.ndarray, float]] = []  # (s, y, rho)
        self.curvature_threshold = getattr(cfg, "bfgs_curvature_threshold", 1e-8)
        self.lbfgs_memory = getattr(cfg, "lbfgs_memory", 10)

    def get_hessian(
        self, model: "Model", x: np.ndarray, lam: np.ndarray, nu: np.ndarray
    ) -> Union[np.ndarray, sp.spmatrix]:
        """
        Return the Hessian according to cfg.hessian_mode, applying regularization if set.
        """
        if not (
            x.shape[0] == self.n
            and lam.shape[0] == model.m_ineq
            and nu.shape[0] == model.m_eq
        ):
            raise ValueError(
                f"Incompatible shapes: x={x.shape}, lam={lam.shape}, nu={nu.shape}, expected ({self.n}, {model.m_ineq}, {model.m_eq})"
            )
        if not (
            np.all(np.isfinite(x))
            and np.all(np.isfinite(lam))
            and np.all(np.isfinite(nu))
        ):
            raise ValueError("Non-finite values in x, lam, or nu")

        mode = self.cfg.hessian_mode
        if mode == "exact":
            H = model.lagrangian_hessian(x, lam, nu)
            H = sp.csr_matrix(H) if self.use_sparse and not sp.issparse(H) else H
        elif mode in ("bfgs", "hybrid"):
            H = self.H
        elif mode == "lbfgs":
            # L-BFGS Hessian is implicit; return identity (used for matvec only)
            H = sp.eye(self.n, format="csr") if self.use_sparse else np.eye(self.n)
        else:
            raise ValueError(f"Unknown hessian_mode {mode}")

        # Ensure symmetry
        if sp.issparse(H):
            diff = H - H.T
            if diff.nnz > 0 and not np.allclose(diff.data, 0, rtol=1e-10, atol=1e-10):
                logging.warning("Hessian is not symmetric; symmetrizing")
                H = 0.5 * (H + H.T)
        else:
            if not np.allclose(H, H.T, rtol=1e-10, atol=1e-10):
                logging.warning("Hessian is not symmetric; symmetrizing")
                H = 0.5 * (H + H.T)

        # Apply regularizer if provided
        if self.regularizer is not None:
            H, _ = self.regularizer.regularize(
                H,
                iteration=getattr(self.cfg, "iteration", 0),
                model_quality=getattr(self.cfg, "model_quality", None),
                constraint_count=model.m_eq,
                grad_norm=np.linalg.norm(model.eval_all(x)["g"]),
            )
        return H

    def update(self, s: np.ndarray, y: np.ndarray):
        """Update approximation (BFGS/L-BFGS/hybrid)."""
        if s.shape[0] != self.n or y.shape[0] != self.n:
            raise ValueError(
                f"Incompatible shapes: s={s.shape}, y={y.shape}, expected ({self.n},)"
            )
        if not (np.all(np.isfinite(s)) and np.all(np.isfinite(y))):
            logging.warning("Non-finite values in s or y; skipping update")
            return

        mode = self.cfg.hessian_mode
        sy = np.dot(s, y)
        if sy <= self.curvature_threshold:
            logging.warning(
                f"Curvature s^T y = {sy} <= {self.curvature_threshold}; skipping update"
            )
            return

        if mode == "bfgs":
            self._bfgs_update(s, y)
        elif mode == "lbfgs":
            self._lbfgs_update(s, y)
        elif mode == "hybrid":
            self._bfgs_update(s, y)

    def _bfgs_update(self, s: np.ndarray, y: np.ndarray):
        """BFGS update: H_{k+1} = Vᵀ H_k V + ρ s sᵀ,  with V = I - ρ s yᵀ, ρ = 1/(sᵀy)."""
        s, y = np.asarray(s), np.asarray(y)
        rho = 1.0 / max(np.dot(s, y), self.curvature_threshold)
        if self.use_sparse:
            I = sp.eye(self.n, format="csr")
            s = s.reshape(-1, 1)
            y = y.reshape(-1, 1)
            V = I - rho * (s @ y.T)
            ssT = rho * (s @ s.T)
            H_dense = V.T @ self.H @ V + ssT
            self.H = sp.csr_matrix(H_dense) if sp.issparse(self.H) else H_dense
        else:
            V = np.eye(self.n) - rho * np.outer(s, y)
            self.H = V.T @ self.H @ V + rho * np.outer(s, s)

    def _lbfgs_update(self, s: np.ndarray, y: np.ndarray):
        """Store (s, y, ρ) pair for L-BFGS two-loop recursion."""
        s, y = np.asarray(s), np.asarray(y)
        rho = 1.0 / max(np.dot(s, y), self.curvature_threshold)
        self.memory.append((s.copy(), y.copy(), rho))
        if len(self.memory) > self.lbfgs_memory:
            self.memory.pop(0)

    def lbfgs_matvec(self, v: np.ndarray) -> np.ndarray:
        """Return (approx) Hv via standard two-loop recursion."""
        if v.shape[0] != self.n:
            raise ValueError(f"Incompatible shape: v={v.shape}, expected ({self.n},)")
        if not np.all(np.isfinite(v)):
            raise ValueError("Non-finite values in v")

        q = v.copy()
        alpha = np.zeros(len(self.memory))
        # First loop: backward
        for i in range(len(self.memory) - 1, -1, -1):
            s, y, rho = self.memory[i]
            alpha[i] = rho * np.dot(s, q)
            q -= alpha[i] * y
        # Initial Hessian scaling
        gamma = getattr(self.cfg, "lbfgs_gamma", 1.0)
        if len(self.memory) > 0:
            s_last, y_last, _ = self.memory[-1]
            gamma = np.dot(s_last, y_last) / max(np.dot(y_last, y_last), 1e-16)
        r = gamma * q
        # Second loop: forward
        for i in range(len(self.memory)):
            s, y, rho = self.memory[i]
            beta = rho * np.dot(y, r)
            r += s * (alpha[i] - beta)
        return r

    def reset(self):
        """Reset Hessian and memory."""
        self.H = sp.eye(self.n, format="csr") if self.use_sparse else np.eye(self.n)
        self.memory.clear()
