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

    delta0: float = 1.0  # initial TR radius
    delta_min: float = 1e-12
    delta_max: float = 1e6
    eta1: float = 0.1  # accept threshold
    eta2: float = 0.9  # expand threshold
    gamma1: float = 0.5  # shrink factor
    gamma2: float = 2.0  # expand factor
    # Byrd-Omojokun split
    zeta: float = 0.8  # fraction of radius for normal step
    # Solver tolerances
    cg_tol: float = 1e-12
    cg_tol_rel: float = 0.1  # relative tolerance factor
    cg_maxiter: int = 200
    neg_curv_tol: float = 1e-14
    constraint_tol: float = 1e-10
    max_active_set_iter: int = 100  # max iterations for active-set loop
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

    # -------- NEW: TR geometry --------
    norm_type: str = "2"       # {"2", "ellip"}
    metric_shift: float = 1e-8 # small ridge to make M ≻ 0 when built from H
    tau_ftb: float = 0.995  # fraction-to-boundary safety factor for box feasibility
    history_length: int = 10  # for adaptive strategies
    non_monotone: bool = False  # use non-monotone TR radius update
    non_monotone_window: int = 5  # window size for non-monotone updates
    max_iter: int = 100  # max TR iterations (for safety)

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


def _as_float_array(a, shape=None) -> np.ndarray:
    out = np.asarray(a, dtype=float)
    if shape is not None and out.shape != shape:
        out = out.reshape(shape)
    return out


def _finite_or_zero(a: np.ndarray) -> np.ndarray:
    if np.isfinite(a).all():
        return a
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def _clean_vec(v, m: int) -> np.ndarray:
    if v is None:
        return np.zeros(m, dtype=float)
    a = np.asarray(v, dtype=float).ravel()
    if a.size != m:
        a = (a[:m] if a.size > m else np.pad(a, (0, m - a.size)))
    return _finite_or_zero(a)


class Model:
    """
    Encapsulates objective, constraints, and derivatives for the SQP/IP stack.
    Keeps identical behavior; faster by fusing evals and cutting Python overhead.
    """

    __slots__ = (
        "n", "f", "cI_funcs", "cE_funcs", "m_ineq", "m_eq",
        "lb", "ub", "use_sparse",
        "f_grad", "f_hess", "cI_grads", "cE_grads", "cI_hess", "cE_hess",
        "_cache", "_cache_x",
        "_dense_JI_cache", "_dense_JE_cache",
        "cfg",
    )

    def __init__(
        self,
        f: Callable,
        c_ineq: List[Callable] | None = None,
        c_eq: List[Callable] | None = None,
        n: int | None = None,
        lb: np.ndarray = np.array([]),
        ub: np.ndarray = np.array([]),
        use_sparse: bool = False,
        *,
        cfg: object | None = None,
    ):
        if n is None or n <= 0:
            raise ValueError(f"Number of variables n must be positive, got {n}")
        if not callable(f):
            raise ValueError("Objective function f must be callable")
        if c_ineq is not None and not all(callable(ci) for ci in c_ineq):
            raise ValueError("All inequality constraints must be callable")
        if c_eq is not None and not all(callable(ce) for ce in c_eq):
            raise ValueError("All equality constraints must be callable")

        self.n = int(n)
        self.f = f
        self.cI_funcs = c_ineq or []
        self.cE_funcs = c_eq or []
        self.m_ineq = len(self.cI_funcs)
        self.m_eq = len(self.cE_funcs)
        self.lb = _as_float_array(lb)
        self.ub = _as_float_array(ub)
        self.use_sparse = bool(use_sparse)

        self._cache: Dict[str, object] = {}
        self._cache_x: Optional[Tuple[float, ...]] = None
        self._dense_JI_cache: Optional[np.ndarray] = None  # cached dense view of JI at _cache_x
        self._dense_JE_cache: Optional[np.ndarray] = None  # cached dense view of JE at _cache_x

        self.cfg = cfg

        self._compile_derivatives()

    # ---------- derivative compilation ----------
    def _compile_derivatives(self) -> None:
        self.f_grad = AD.sym_grad(self.f, self.n)   # returns grad; .value(x) gives scalar
        self.f_hess = AD.sym_hess(self.f, self.n)   # returns Hessian

        # Constraints: keep GradFn for values/gradients; Hessians if solver needs them.
        self.cI_grads = [AD.sym_grad(ci, self.n) for ci in self.cI_funcs]
        self.cE_grads = [AD.sym_grad(ce, self.n) for ce in self.cE_funcs]

        self.cI_hess = [AD.sym_hess(ci, self.n) for ci in self.cI_funcs]
        self.cE_hess = [AD.sym_hess(ce, self.n) for ce in self.cE_funcs]
        
        print(f"Compiled derivatives: n={self.n}, m_ineq={self.m_ineq}, m_eq={self.m_eq}")

    # ---------- fused evaluation ----------
    def eval_all(
        self, x: np.ndarray, components: Optional[list[str]] = None
    ) -> dict[str, float | np.ndarray | sp.spmatrix | None]:
        """
        Returns any subset of {"f","g","H","cI","JI","cE","JE"} quickly.
        Fuses constraint value/grad loops, minimizes conversions, caches results at x.
        """
        x = _as_float_array(x).ravel()
        n = self.n
        if x.shape[0] != n:
            raise ValueError(f"Input x shape {x.shape} does not match n={n}")
        if not np.isfinite(x).all():
            raise ValueError("Non-finite values in x")

        want = components or ["f", "g", "H", "cI", "JI", "cE", "JE"]
        want_set = set(want)

        # cache key (immutable snapshot)
        x_key = tuple(x.tolist())

        # cache fast path
        if x_key == self._cache_x:
            c = self._cache
            if all(k in c for k in want):
                # return view of cached keys
                return {k: c.get(k, None) for k in want}

        use_sparse = self.use_sparse
        mI, mE = self.m_ineq, self.m_eq

        res: dict[str, object] = {}

        # ---------- objective ----------
        if "f" in want_set:
            try:
                fv = float(self.f_grad.value(x))  # same tape as f_grad
            except Exception:
                fv = float("inf")
            if not np.isfinite(fv):
                fv = float("inf")
            res["f"] = fv

        if "g" in want_set:
            try:
                g = self.f_grad(x)
                g = g if isinstance(g, np.ndarray) and g.dtype == float else _as_float_array(g)
                if not np.isfinite(g).all():
                    g = np.zeros(n, dtype=float)
            except Exception:
                g = np.zeros(n, dtype=float)
            res["g"] = g

        if "H" in want_set:
            try:
                H = self.f_hess(x)
                if sp.issparse(H):
                    # symmetrize lightly
                    H = (H + H.T) * 0.5
                    if use_sparse:
                        res["H"] = H.tocsr()
                    else:
                        res["H"] = H.toarray()
                else:
                    H = _as_float_array(H, (n, n))
                    # Optional: enforce symmetry if your AD doesn’t
                    # H = 0.5*(H + H.T)
                    if use_sparse:
                        res["H"] = sp.csr_matrix(H)
                    else:
                        res["H"] = H
            except Exception:
                res["H"] = sp.eye(n, format="csr") if use_sparse else np.eye(n, dtype=float)

        # ---------- constraints: fused loops ----------
        # Inequalities
        haveI = mI > 0 and (("cI" in want_set) or ("JI" in want_set))
        if haveI:
            cI_vals = None
            JI_rows = None
            try:
                if "cI" in want_set:
                    cI_vals = np.empty(mI, dtype=float)
                if "JI" in want_set:
                    JI_rows = np.empty((mI, n), dtype=float)
                # single pass
                for i, gi in enumerate(self.cI_grads):
                    if cI_vals is not None:
                        cI_vals[i] = float(gi.value(x))
                    if JI_rows is not None:
                        row = gi(x)
                        JI_rows[i, :] = row if (isinstance(row, np.ndarray) and row.dtype == float) \
                                              else _as_float_array(row, (n,))
                if cI_vals is not None:
                    res["cI"] = _finite_or_zero(cI_vals)
                if JI_rows is not None:
                    if self.use_sparse:
                        JI = sp.csr_matrix(JI_rows)  # one conversion
                        res["JI"] = JI
                        self._dense_JI_cache = JI_rows  # keep dense view for downstream use
                    else:
                        res["JI"] = _finite_or_zero(JI_rows)
                        self._dense_JI_cache = res["JI"]  # type: ignore[assignment]
            except Exception:
                if "cI" in want_set:
                    res["cI"] = np.zeros(mI, dtype=float)
                if "JI" in want_set:
                    res["JI"] = sp.csr_matrix((mI, n)) if self.use_sparse else np.zeros((mI, n), dtype=float)
                self._dense_JI_cache = None

        else:
            if "cI" in want_set: res["cI"] = None
            if "JI" in want_set: res["JI"] = None
            self._dense_JI_cache = None

        # Equalities
        haveE = self.m_eq > 0 and (("cE" in want_set) or ("JE" in want_set))
        if haveE:
            cE_vals = None
            JE_rows = None
            try:
                if "cE" in want_set:
                    cE_vals = np.empty(self.m_eq, dtype=float)
                if "JE" in want_set:
                    JE_rows = np.empty((self.m_eq, n), dtype=float)
                for j, ge in enumerate(self.cE_grads):
                    if cE_vals is not None:
                        cE_vals[j] = float(ge.value(x))
                    if JE_rows is not None:
                        row = ge(x)
                        JE_rows[j, :] = row if (isinstance(row, np.ndarray) and row.dtype == float) \
                                              else _as_float_array(row, (n,))
                if cE_vals is not None:
                    res["cE"] = _finite_or_zero(cE_vals)
                if JE_rows is not None:
                    if self.use_sparse:
                        JE = sp.csr_matrix(JE_rows)
                        res["JE"] = JE
                        self._dense_JE_cache = JE_rows
                    else:
                        res["JE"] = _finite_or_zero(JE_rows)
                        self._dense_JE_cache = res["JE"]  # type: ignore[assignment]
            except Exception:
                if "cE" in want_set:
                    res["cE"] = np.zeros(self.m_eq, dtype=float)
                if "JE" in want_set:
                    res["JE"] = sp.csr_matrix((self.m_eq, n)) if self.use_sparse else np.zeros((self.m_eq, n), dtype=float)
                self._dense_JE_cache = None
        else:
            if "cE" in want_set: res["cE"] = None
            if "JE" in want_set: res["JE"] = None
            self._dense_JE_cache = None

        # update cache & return
        self._cache = res
        self._cache_x = x_key
        return {k: res.get(k, None) for k in want}

    # ---------- Lagrangian Hessian ----------
    def lagrangian_hessian(self, x: np.ndarray, lam: np.ndarray, nu: np.ndarray) -> np.ndarray | sp.spmatrix:
        n, mI, mE = self.n, self.m_ineq, self.m_eq
        x = _as_float_array(x, (n,))
        lam = _as_float_array(lam).ravel()[:mI]
        nu  = _as_float_array(nu ).ravel()[:mE]
        if not (np.isfinite(x).all() and np.isfinite(lam).all() and np.isfinite(nu).all()):
            raise ValueError("Non-finite values in x, lam, or nu")

        cfg = getattr(self, "cfg", None)
        multiplier_threshold = float(getattr(cfg, "multiplier_threshold", 1e-8)) if cfg else 1e-8
        clip_max = float(getattr(cfg, "hess_clip_max", 1e12)) if cfg else 1e12
        use_sparse = self.use_sparse

        def _sanitize_dense(A: np.ndarray) -> np.ndarray:
            np.clip(A, -clip_max, clip_max, out=A)
            A[~np.isfinite(A)] = 0.0
            return A

        def _sanitize_sparse(A: sp.spmatrix) -> sp.spmatrix:
            A = A.tocsr(copy=True)
            d = A.data
            bad = ~np.isfinite(d)
            if bad.any():
                d[bad] = 0.0
            np.clip(A.data, -clip_max, clip_max, out=A.data)
            return A

        # Base Hessian directly from AD (avoid dict path)
        try:
            H = self.f_hess(x)
        except Exception:
            H = None

        if H is None:
            H = sp.csr_matrix((n, n)) if use_sparse else np.zeros((n, n), dtype=float)
        elif sp.issparse(H):
            H = (H + H.T) * 0.5
            H = _sanitize_sparse(H)
            if not use_sparse:
                H = H.toarray()
        else:
            H = _as_float_array(H, (n, n))
            # Optional symmetry enforcement
            # H = 0.5*(H + H.T)
            H = _sanitize_dense(H)
            if use_sparse:
                H = sp.csr_matrix(H)

        def _add_piece(H_acc, w: float, Hi_fun):
            if Hi_fun is None or abs(w) <= multiplier_threshold:
                return H_acc
            try:
                Hi = Hi_fun(x)
            except Exception:
                return H_acc

            if sp.issparse(Hi):
                Hi = (Hi + Hi.T) * 0.5
                Hi = _sanitize_sparse(Hi)
                return (H_acc + (w * Hi)) if use_sparse else (H_acc + (w * Hi.toarray()))
            else:
                Hi = _as_float_array(Hi, (n, n))
                Hi = 0.5 * (Hi + Hi.T)
                Hi = _sanitize_dense(Hi)
                return (H_acc + (w * sp.csr_matrix(Hi))) if use_sparse else (H_acc + (w * Hi))

        # Add constraint Hessians
        for w, Hi in zip(lam, self.cI_hess):
            H = _add_piece(H, float(w), Hi)
        for w, Hi in zip(nu, self.cE_hess):
            H = _add_piece(H, float(w), Hi)

        # Final finite check
        if use_sparse:
            if not np.isfinite(H.data).all():
                H = sp.eye(n, format="csr")
        else:
            if not np.isfinite(H).all():
                H = np.eye(n, dtype=float)
        return H

    # ---------- norms/diagnostics ----------
    def constraint_violation(self, x: np.ndarray) -> float:
        x = _as_float_array(x, (self.n,))
        d = self.eval_all(x, components=["cI", "cE"])
        mI, mE = self.m_ineq, self.m_eq

        cI = _clean_vec(d.get("cI", None), mI) if mI > 0 else np.zeros(0, dtype=float)
        cE = _clean_vec(d.get("cE", None), mE) if mE > 0 else np.zeros(0, dtype=float)

        scale = float(max(1, self.n, mI + mE))
        theta = 0.0
        if mI: theta += float(np.maximum(0.0, cI).sum()) / scale
        if mE: theta += float(np.abs(cE).sum()) / scale
        if not np.isfinite(theta):
            logging.warning("Non-finite constraint violation")
            theta = float("inf")
        return theta

    def kkt_residuals(self, x: np.ndarray, lam: np.ndarray, nu: np.ndarray) -> dict[str, float]:
        n, mI, mE = self.n, self.m_ineq, self.m_eq
        x  = _as_float_array(x, (n,))
        lam = _as_float_array(lam).ravel()[:mI]
        nu  = _as_float_array(nu ).ravel()[:mE]
        if not (np.isfinite(x).all() and np.isfinite(lam).all() and np.isfinite(nu).all()):
            raise ValueError("Non-finite values in x, lam, or nu")

        need = ["g"]
        if mI: need += ["JI", "cI"]
        if mE: need += ["JE", "cE"]
        d = self.eval_all(x, components=need)

        g = _as_float_array(d.get("g", np.zeros(n, dtype=float)), (n,))
        scale_g = max(1.0, float(np.linalg.norm(g, ord=np.inf)))

        rL = g.copy()
        if mI:
            JI = d.get("JI", None)
            lam_p = np.maximum(lam, 0.0)
            if JI is not None:
                if sp.issparse(JI): rL += JI.T @ lam_p
                else:               rL += _as_float_array(JI) .T @ lam_p
        if mE:
            JE = d.get("JE", None)
            if JE is not None:
                if sp.issparse(JE): rL += JE.T @ nu
                else:               rL += _as_float_array(JE).T @ nu

        stat_inf = float(np.linalg.norm(rL, ord=np.inf)) / scale_g

        ineq_inf = 0.0
        comp_inf = 0.0
        if mI:
            cI = _as_float_array(d.get("cI", np.zeros(mI, dtype=float))).ravel()
            cI_plus = np.maximum(0.0, cI)
            ineq_inf = float(np.linalg.norm(cI_plus, ord=np.inf)) / scale_g
            comp = np.abs(np.maximum(lam, 0.0) * cI)
            comp_inf = float(comp.max() if comp.size else 0.0) / scale_g

        eq_inf = 0.0
        if mE:
            cE = _as_float_array(d.get("cE", np.zeros(mE, dtype=float))).ravel()
            eq_inf = float(np.linalg.norm(cE, ord=np.inf)) / scale_g

        res = {"stat": stat_inf, "ineq": ineq_inf, "eq": eq_inf, "comp": comp_inf}
        if not all(np.isfinite(v) for v in res.values()):
            res = {k: float("inf") for k in res}
        return res

    # ---------- cache ----------
    def reset_cache(self) -> None:
        self._cache.clear()
        self._cache_x = None
        self._dense_JI_cache = None
        self._dense_JE_cache = None

    # ---------- SOC step (unchanged interface; faster assembly) ----------
    def compute_soc_step(
        self,
        rE: np.ndarray | None,
        rI: np.ndarray | None,
        mu: float,
        *,
        active_tol: float = 1e-6,
        w_eq: float = 1.0,
        w_ineq: float = 1.0,
        gamma: float = 1e-8,
    ) -> tuple[np.ndarray, np.ndarray]:
        n, mE, mI = self.n, self.m_eq, self.m_ineq

        # normalize residuals
        rE = None if (rE is None or (hasattr(rE, "__len__") and len(rE) == 0)) else _as_float_array(rE).ravel()
        rI = None if (rI is None or (hasattr(rI, "__len__") and len(rI) == 0)) else _as_float_array(rI).ravel()

        # Grab JE/JI from cache (eval_all should have been called at base point)
        JE = self._cache.get("JE", None)
        JI = self._cache.get("JI", None)

        # If missing, recompute at cached x (cheap)
        if (JE is None and mE > 0) or (JI is None and mI > 0):
            if self._cache_x is not None:
                x_arr = np.asarray(self._cache_x, dtype=float)
                need = (["JE"] if (JE is None and mE > 0) else []) + (["JI"] if (JI is None and mI > 0) else [])
                d = self.eval_all(x_arr, components=need)
                JE = d.get("JE", JE)
                JI = d.get("JI", JI)

        def _dense(M, cached_dense):
            if M is None:
                return None
            if sp.issparse(M):
                # prefer the already-constructed dense view if it matches this x
                return cached_dense if cached_dense is not None else M.toarray()
            return _as_float_array(M)

        JE_d = _dense(JE, self._dense_JE_cache) if mE > 0 else None
        JI_d = _dense(JI, self._dense_JI_cache) if mI > 0 else None

        blocks = []

        if (mE > 0) and (rE is not None) and (JE_d is not None) and JE_d.size:
            blocks.append((w_eq, JE_d, rE, None))  # no row weights

        if (mI > 0) and (rI is not None) and (JI_d is not None) and JI_d.size:
            sel = (rI > 0.0) | (np.abs(rI) >= active_tol)
            W = np.where(sel, 1.0, 0.0)
            blocks.append((w_ineq, JI_d, rI, W))

        if not blocks:
            return np.zeros(n, dtype=float), (np.zeros(mI, dtype=float) if mI > 0 else np.zeros(0, dtype=float))

        # Normal equations: (A^T A + gamma I) dx = -A^T b
        AtA = np.eye(n, dtype=float) * float(gamma)
        ATb = np.zeros(n, dtype=float)

        for w, J, r, W in blocks:
            if w == 0.0 or J.size == 0:
                continue
            if W is None:
                JW = J
                rW = r
            else:
                # Row scaling via (J.T * W).T
                JW = (J.T * W).T
                rW = W * r
            JWw = w * JW
            rWw = w * rW
            # Accumulate
            AtA += JWw.T @ JWw
            ATb += JWw.T @ rWw

        # Solve
        try:
            dx_cor = -np.linalg.solve(AtA, ATb)
        except np.linalg.LinAlgError:
            dx_cor = -np.linalg.lstsq(AtA, ATb, rcond=None)[0]

        if (mI > 0) and (rI is not None) and (JI_d is not None) and JI_d.size:
            ds_cor = -(rI + JI_d @ dx_cor)
        else:
            ds_cor = np.zeros(0, dtype=float)

        # Make finite
        if not np.isfinite(dx_cor).all():
            dx_cor = _finite_or_zero(dx_cor)
        if ds_cor.size and not np.isfinite(ds_cor).all():
            ds_cor = _finite_or_zero(ds_cor)

        return dx_cor, ds_cor

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
