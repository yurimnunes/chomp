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
    use_line_search: bool = True
    use_trust_region: bool = True
    use_soc: bool = True
    use_funnel: bool = False
    use_watchdog: bool = False
    use_nonmonotone_ls: bool = False
    use_active_set_prediction: bool = False
    hessian_mode: str = "exact"  # {"exact","bfgs","lbfgs","hybrid","gn"}
    
    tr_delta0: float = 0.1

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
            self.f_val = AD.sym_val(self.f, self.n)
            self.f_grad = AD.sym_grad(self.f, self.n)
            self.f_hess = AD.sym_hess(self.f, self.n)
            self.cI_vals = [AD.sym_val(ci, self.n) for ci in self.cI_funcs]
            self.cE_vals = [AD.sym_val(ce, self.n) for ce in self.cE_funcs]
            self.cI_grads = [AD.sym_grad(ci, self.n) for ci in self.cI_funcs]
            self.cI_hess = [AD.sym_hess(ci, self.n) for ci in self.cI_funcs]
            self.cE_grads = [AD.sym_grad(ce, self.n) for ce in self.cE_funcs]
            self.cE_hess = [AD.sym_hess(ce, self.n) for ce in self.cE_funcs]
        except Exception as e:
            logging.error(f"AD compilation failed: {e}")
            raise RuntimeError("Failed to compile derivatives")
        
    def eval_all(
        self, x: np.ndarray, components: Optional[list[str]] = None
    ) -> dict[str, float | np.ndarray | sp.spmatrix | None]:
        """
        Faster evaluator for {"f","g","H","cI","JI","cE","JE"} with light allocations
        and cheap sparse assembly. Preserves previous behavior & cache keys.
        """
        # --- very cheap guards ---
        n = self.n
        if x.shape[0] != n:
            raise ValueError(f"Input x shape {x.shape} does not match n={n}")
        if not np.isfinite(x).all():
            raise ValueError("Non-finite values in x")

        want = components or ["f", "g", "H", "cI", "JI", "cE", "JE"]
        want_set = set(want)

        # --- cache fast-path ---
        x_key = tuple(x)  # stable & cheap
        if x_key == self._cache_x:
            cached = self._cache
            # Return only if we have everything requested
            if all(k in cached for k in want):
                # NOTE: return references; caller should treat as read-only
                return {k: cached[k] for k in want}

        # --- bind locals to reduce attribute lookups ---
        use_sparse = self.use_sparse
        mI, mE = self.m_ineq, self.m_eq
        cI_funcs, cE_funcs = self.cI_vals, self.cE_vals
        f_grad, f_hess = self.f_grad, self.f_hess
        cI_grads, cE_grads = self.cI_grads, self.cE_grads
        AD_val = AD.val  # local alias

        res: dict[str, object] = {}

        # ---------- Objective value ----------
        if "f" in want_set:
            try:
                fv = float(self.f_val(x))
                if not np.isfinite(fv):
                    fv = float("inf")
                res["f"] = fv
            except Exception:
                res["f"] = float("inf")

        # ---------- Gradient ----------
        if "g" in want_set:
            try:
                g = f_grad(x)
                g = g if isinstance(g, np.ndarray) and g.dtype == float else np.asarray(g, dtype=float)
                if not np.isfinite(g).all():
                    g = np.zeros(n, dtype=float)
                res["g"] = g
            except Exception:
                res["g"] = np.zeros(n, dtype=float)

        # ---------- Hessian (always symmetrize; skip costly checks) ----------
        if "H" in want_set:
            try:
                H = f_hess(x)
                if sp.issparse(H):
                    H = H.tocsr()
                    H = (H + H.T) * 0.5
                    data = H.data
                    if not np.isfinite(data).all():
                        H = sp.eye(n, format="csr")
                else:
                    H = np.asarray(H, dtype=float, order="C")
                    H = 0.5 * (H + H.T)
                    if not np.isfinite(H).all():
                        H = np.eye(n, dtype=float)
                if use_sparse and not sp.issparse(H):
                    H = sp.csr_matrix(H)
                elif not use_sparse and sp.issparse(H):
                    H = H.toarray()
                res["H"] = H
            except Exception:
                res["H"] = sp.eye(n, format="csr") if use_sparse else np.eye(n, dtype=float)

        # ---------- Inequalities ----------
        haveI = mI > 0
        if "cI" in want_set:
            if haveI:
                try:
                    # Preallocate once; cheap loop (callable AD is Python-bound anyway)
                    cI = np.empty(mI, dtype=float)
                    for i, ci in enumerate(cI_funcs):
                        cI[i] = float(ci(x))
                    if not np.isfinite(cI).all():
                        cI = np.zeros(mI, dtype=float)
                    res["cI"] = cI
                except Exception:
                    res["cI"] = np.zeros(mI, dtype=float)
            else:
                res["cI"] = None

        if "JI" in want_set:
            if haveI:
                try:
                    # Build dense row-block once, then convert if needed
                    JI_rows = np.empty((mI, n), dtype=float)
                    for i, gi in enumerate(cI_grads):
                        row = gi(x)
                        row = row if (isinstance(row, np.ndarray) and row.dtype == float) else np.asarray(row, dtype=float)
                        JI_rows[i, :] = row
                    if use_sparse:
                        JI = sp.csr_matrix(JI_rows)
                        if not np.isfinite(JI.data).all():
                            JI = sp.csr_matrix((mI, n))
                    else:
                        if not np.isfinite(JI_rows).all():
                            JI_rows.fill(0.0)
                        JI = JI_rows
                    res["JI"] = JI
                except Exception:
                    res["JI"] = sp.csr_matrix((mI, n)) if use_sparse else np.zeros((mI, n), dtype=float)
            else:
                res["JI"] = None

        # ---------- Equalities ----------
        haveE = mE > 0
        if "cE" in want_set:
            if haveE:
                try:
                    cE = np.empty(mE, dtype=float)
                    for j, ce in enumerate(cE_funcs):
                        cE[j] = float(ce(x))
                    if not np.isfinite(cE).all():
                        cE = np.zeros(mE, dtype=float)
                    res["cE"] = cE
                except Exception:
                    res["cE"] = np.zeros(mE, dtype=float)
            else:
                res["cE"] = None

        if "JE" in want_set:
            if haveE:
                try:
                    JE_rows = np.empty((mE, n), dtype=float)
                    for j, ge in enumerate(cE_grads):
                        row = ge(x)
                        row = row if (isinstance(row, np.ndarray) and row.dtype == float) else np.asarray(row, dtype=float)
                        JE_rows[j, :] = row
                    if use_sparse:
                        JE = sp.csr_matrix(JE_rows)
                        if not np.isfinite(JE.data).all():
                            JE = sp.csr_matrix((mE, n))
                    else:
                        if not np.isfinite(JE_rows).all():
                            JE_rows.fill(0.0)
                        JE = JE_rows
                    res["JE"] = JE
                except Exception:
                    res["JE"] = sp.csr_matrix((mE, n)) if use_sparse else np.zeros((mE, n), dtype=float)
            else:
                res["JE"] = None

        # --- update cache with everything we just computed and mark x ---
        # Important: keep only the keys we know (avoid stale)
        self._cache = res
        self._cache_x = x_key

        # Return only requested subset
        return {k: res.get(k, None) for k in want}
    
    def lagrangian_hessian(
        self, x: np.ndarray, lam: np.ndarray, nu: np.ndarray
    ) -> np.ndarray | sp.spmatrix:
        """
        Fast ∇²_x L assembly with unconditional symmetrization and minimal conversions.
        Keeps the same hardening and fallbacks you had.
        """
        n = self.n
        mI, mE = self.m_ineq, self.m_eq
        use_sparse = self.use_sparse

        cfg = getattr(self, "cfg", None)
        multiplier_threshold = float(getattr(cfg, "multiplier_threshold", 1e-8))
        clip_max = float(getattr(cfg, "hess_clip_max", 1e12))
        diag_floor = float(getattr(cfg, "hess_diag_floor", 0.0))

        if x.shape[0] != n:
            raise ValueError(f"Incompatible x shape: expected ({n},), got {x.shape}")
        lam = np.asarray(lam, dtype=float).ravel()
        nu = np.asarray(nu, dtype=float).ravel()
        if lam.size != mI:
            lam = lam[:mI]
        if nu.size != mE:
            nu = nu[:mE]
        if not (np.isfinite(x).all() and np.isfinite(lam).all() and np.isfinite(nu).all()):
            raise ValueError("Non-finite values in x, lam, or nu")

        def _sanitize_dense(A: np.ndarray) -> np.ndarray:
            if not np.isfinite(A).all():
                A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
            np.clip(A, -clip_max, clip_max, out=A)
            return A

        def _sanitize_sparse(A: sp.spmatrix) -> sp.spmatrix:
            A = A.tocsr()
            d = A.data
            if not np.isfinite(d).all():
                d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
                A = sp.csr_matrix((d, A.indices, A.indptr), shape=A.shape)
            np.clip(A.data, -clip_max, clip_max, out=A.data)
            return A

        # Base Hessian
        try:
            H = self.eval_all(x, components=["H"])["H"]
        except Exception:
            H = None

        if H is None:
            H = sp.csr_matrix((n, n)) if use_sparse else np.zeros((n, n), dtype=float)
        else:
            if sp.issparse(H):
                H = (H + H.T) * 0.5
                H = _sanitize_sparse(H)
                if not use_sparse:
                    H = H.toarray()
            else:
                H = np.asarray(H, dtype=float, order="C")
                H = 0.5 * (H + H.T)
                H = _sanitize_dense(H)
                if use_sparse:
                    H = sp.csr_matrix(H)

        # Add constraints’ Hessians
        def _add_piece(H_acc, w: float, Hi_fun):
            if (Hi_fun is None) or (abs(w) <= multiplier_threshold):
                return H_acc
            try:
                Hi = Hi_fun(x)
            except Exception:
                return H_acc
            if sp.issparse(Hi):
                Hi = (Hi + Hi.T) * 0.5
                Hi = _sanitize_sparse(Hi)
                if use_sparse:
                    return H_acc + (w * Hi)
                else:
                    return H_acc + (w * Hi.toarray())
            else:
                Hi = np.asarray(Hi, dtype=float, order="C")
                Hi = 0.5 * (Hi + Hi.T)
                Hi = _sanitize_dense(Hi)
                if use_sparse:
                    return H_acc + (w * sp.csr_matrix(Hi))
                else:
                    return H_acc + (w * Hi)

        for w, Hi in zip(lam, getattr(self, "cI_hess", ())):
            H = _add_piece(H, w, Hi)
        for w, Hi in zip(nu, getattr(self, "cE_hess", ())):
            H = _add_piece(H, w, Hi)

        # Tiny diagonal floor if requested
        if diag_floor > 0.0:
            if use_sparse:
                d = H.diagonal()
                mask = (np.abs(d) < diag_floor) | ~np.isfinite(d)
                if np.any(mask):
                    H = (H + sp.diags(diag_floor * mask.astype(float))).tocsr()
            else:
                d = np.diag(H)
                mask = (np.abs(d) < diag_floor) | ~np.isfinite(d)
                if np.any(mask):
                    idx = np.where(mask)[0]
                    H[idx, idx] = diag_floor

        # Final finite check
        if use_sparse:
            if not np.isfinite(H.data).all():
                H = sp.eye(n, format="csr")
        else:
            if not np.isfinite(H).all():
                H = np.eye(n, dtype=float)
        return H

    def constraint_violation(self, x: np.ndarray) -> float:
        """L1 norm of violations (cI⁺ + |cE|) scaled by problem size; robust to None."""
        # if x.shape[0] != self.n:
        #     raise ValueError(f"Input x shape {x.shape} does not match n={self.n}")
        # if not np.all(np.isfinite(x)):
        #     raise ValueError("Non-finite values in x")

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
    
    def kkt_residuals(self, x: np.ndarray, lam: np.ndarray, nu: np.ndarray) -> dict[str, float]:
        """Cheaper assembly with shared eval and fewer conversions."""
        n, mI, mE = self.n, self.m_ineq, self.m_eq
        lam = np.asarray(lam, dtype=float).ravel()
        nu  = np.asarray(nu,  dtype=float).ravel()
        if x.shape[0] != n or lam.shape[0] != mI or nu.shape[0] != mE:
            raise ValueError(f"Incompatible shapes: x={x.shape}, lam={lam.shape}, nu={nu.shape}")
        if not (np.isfinite(x).all() and np.isfinite(lam).all() and np.isfinite(nu).all()):
            raise ValueError("Non-finite values in x, lam, or nu")

        need = ["g"]
        if mI: need += ["JI","cI"]
        if mE: need += ["JE","cE"]
        d = self.eval_all(x, components=need)

        g  = np.asarray(d.get("g", np.zeros(n, dtype=float)), dtype=float).ravel()
        scale_g = max(1.0, float(np.linalg.norm(g, ord=np.inf)))

        # Stationarity rL = g + JI^T lam + JE^T nu
        rL = g.copy()
        if mI:
            JI = d.get("JI", None)
            if JI is None:
                pass
            elif sp.issparse(JI):
                rL += JI.T @ np.maximum(lam, 0.0)
            else:
                rL += np.asarray(JI, dtype=float, order="C").T @ np.maximum(lam, 0.0)
        if mE:
            JE = d.get("JE", None)
            if JE is None:
                pass
            elif sp.issparse(JE):
                rL += JE.T @ nu
            else:
                rL += np.asarray(JE, dtype=float, order="C").T @ nu

        stat_inf = float(np.linalg.norm(rL, ord=np.inf)) / scale_g

        # Feasibility
        ineq_inf = 0.0
        comp_inf = 0.0
        if mI:
            cI = np.asarray(d.get("cI", np.zeros(mI, dtype=float)), dtype=float).ravel()
            cI_plus = np.maximum(0.0, cI)
            ineq_inf = float(np.linalg.norm(cI_plus, ord=np.inf)) / scale_g
            comp = np.abs(np.maximum(lam, 0.0) * cI)
            comp_inf = float(comp.max() if comp.size else 0.0) / scale_g

        eq_inf = 0.0
        if mE:
            cE = np.asarray(d.get("cE", np.zeros(mE, dtype=float)), dtype=float).ravel()
            eq_inf = float(np.linalg.norm(cE, ord=np.inf)) / scale_g

        res = {"stat": stat_inf, "ineq": ineq_inf, "eq": eq_inf, "comp": comp_inf}
        if not all(np.isfinite(v) for v in res.values()):
            res = {k: float("inf") for k in res}
        return res

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
