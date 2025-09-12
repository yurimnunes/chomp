# sqp_ip.py
# InteriorPointStepper: stabilized slacks-only barrier IPM w/ bounds
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import qdldl_cpp as qd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import LinearOperator, cg

# External project modules (unchanged API)
from nlp.blocks.linesearch import LineSearcher

from .blocks.aux import HessianManager, Model, SQPConfig
from .blocks.filter import Filter, Funnel
from .blocks.reg import Regularizer, make_psd_advanced


# ------------------ sparse helpers ------------------
def _isspm(A) -> bool:
    return sp.isspmatrix(A)


def _csr(A, shape=None):
    if A is None:
        if shape is None:
            raise ValueError("shape required when A is None")
        return sp.csr_matrix(shape)
    return A.tocsr() if _isspm(A) else sp.csr_matrix(A)


def _sym(A):
    if _isspm(A):
        return (A + A.T) * 0.5
    return 0.5 * (A + A.T)


def _diag(v: np.ndarray):
    return sp.diags(v) if v.ndim == 1 else sp.diags(np.asarray(v).ravel())


def _as_array_or_zeros(v, m):
    if m == 0:
        return np.zeros(0, dtype=float)
    if v is None:
        return np.zeros(m, dtype=float)
    a = np.asarray(v, dtype=float).ravel()
    if a.size != m:
        a = a.reshape(m)
    return a


# ------------------ IP state ------------------
@dataclass
class IPState:
    mI: int
    mE: int
    s: np.ndarray
    lam: np.ndarray
    nu: np.ndarray
    zL: np.ndarray  # duals for lower bounds
    zU: np.ndarray  # duals for upper bounds
    mu: float
    initialized: bool = False

    @staticmethod
    def from_model(model: Model, x: np.ndarray, cfg: SQPConfig) -> "IPState":
        mI, mE = model.m_ineq, model.m_eq
        d = model.eval_all(x, components=["cI", "cE"])
        cI = np.zeros(mI) if (mI == 0 or d["cI"] is None) else np.asarray(d["cI"], float)
        # Start safely interior: s ~ max(1, -cI + 1e-3), λ ≈ μ/s
        mu0 = max(getattr(cfg, "ip_mu_init", 1e-2), 1e-12)
        if mI > 0:
            s0 = np.maximum(1.0, -cI + 1e-3)
            lam0 = np.maximum(1e-8, mu0 / np.maximum(s0, 1e-12))
        else:
            s0 = np.zeros(0)
            lam0 = np.zeros(0)
        nu0 = np.zeros(mE)

        # bounds (ℓ ≤ x ≤ u); treat ±∞ as inactive
        lb = getattr(model, "lb", None)
        ub = getattr(model, "ub", None)
        lb = np.full_like(x, -np.inf) if lb is None else np.asarray(lb, float)
        ub = np.full_like(x, +np.inf) if ub is None else np.asarray(ub, float)
        hasL = np.isfinite(lb)
        hasU = np.isfinite(ub)
        sL = np.where(hasL, np.maximum(1e-12, x - lb), 1.0)
        sU = np.where(hasU, np.maximum(1e-12, ub - x), 1.0)
        zL0 = np.where(hasL, np.maximum(1e-8, mu0 / sL), 0.0)
        zU0 = np.where(hasU, np.maximum(1e-8, mu0 / sU), 0.0)

        return IPState(
            mI=mI, mE=mE, s=s0, lam=lam0, nu=nu0, zL=zL0, zU=zU0, mu=mu0, initialized=True
        )


def _get_bounds(model: Model, x: np.ndarray):
    """Return (lb, ub, hasL, hasU) as float arrays and boolean masks."""
    n = x.size
    lb = getattr(model, "lb", None)
    ub = getattr(model, "ub", None)
    lb = np.full(n, -np.inf, dtype=float) if lb is None else np.asarray(lb, float).reshape(-1)
    ub = np.full(n, +np.inf, dtype=float) if ub is None else np.asarray(ub, float).reshape(-1)
    hasL = np.isfinite(lb)
    hasU = np.isfinite(ub)
    return lb, ub, hasL, hasU


def _safe_pos_div(num: np.ndarray, den: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return num / np.maximum(den, eps)


def safe_inf_norm(v: np.ndarray) -> float:
    if v is None or v.size == 0:
        return 0.0
    return float(np.linalg.norm(v, np.inf))


# --- the restoration model (unchanged API) ---
class RestorationModel:
    """
    Variables:  bar_x = [ x (n) ; p (m) ; n (m) ], where m = base.m_eq + base.m_ineq
    Equalities: c_rest(bar_x) = [cE_base(x); cI_base(x)] - p + n = 0
    Inequalities: none (p,n >= 0 are handled as simple bounds)
    Objective:   f(bar_x) = (ζ/2) * (x - x_R)^T D_R (x - x_R) + 1^T p + 1^T n
    """

    def __init__(self, base_model, x_R, D_R, zeta: float, rho: float = 1.0):
        self.base = base_model
        self.x_R = np.asarray(x_R, dtype=float)
        self.D_R = _csr(D_R, (base_model.n, base_model.n))
        self.zeta = float(zeta)
        self.rho = float(rho)

        # sizes
        self.n_orig = int(base_model.n)
        self.m_base_E = int(base_model.m_eq)
        self.m_base_I = int(base_model.m_ineq)
        self.m_eq = self.m_base_E + self.m_base_I
        self.m_ineq = 0
        self.n = self.n_orig + 2 * self.m_eq  # x + p + n

        # bounds so IP sees p>=0, n>=0 and original x-bounds
        lb_x = getattr(self.base, "lb", None)
        ub_x = getattr(self.base, "ub", None)
        lb_x = (np.full(self.n_orig, -np.inf, float) if lb_x is None
                else np.asarray(lb_x, float).reshape(-1))
        ub_x = (np.full(self.n_orig, +np.inf, float) if ub_x is None
                else np.asarray(ub_x, float).reshape(-1))

        if self.m_eq > 0:
            zeros_m = np.zeros(self.m_eq, dtype=float)
            inf_m   = np.full(self.m_eq, +np.inf, dtype=float)
            self.lb = np.concatenate([lb_x, zeros_m, zeros_m])
            self.ub = np.concatenate([ub_x,  inf_m,   inf_m])
        else:
            self.lb = lb_x.copy()
            self.ub = ub_x.copy()

    # ------------ helpers ------------
    def split(self, bar_x):
        n = self.n_orig
        m = self.m_eq
        if m > 0:
            x  = bar_x[:n]
            p  = bar_x[n : n + m]
            nn = bar_x[n + m : n + 2 * m]
        else:
            x  = bar_x[:n]
            p  = np.zeros(0, dtype=float)
            nn = np.zeros(0, dtype=float)
        return x, p, nn

    # ------------ objective ------------
    def f_scalar(self, bar_x):
        x, p, nn = self.split(bar_x)
        dx = x - self.x_R
        quad = 0.5 * self.zeta * float(dx @ (self.D_R @ dx))
        lin  = self.rho * (p.sum() + nn.sum())
        return quad + lin

    # ------------ API: eval_all ------------
    def eval_all(self, bar_x, components=("f", "g", "cE", "JE", "H")):
        need_f = "f" in components
        need_g = "g" in components
        need_c = any(k in components for k in ("cE", "JE", "Aeq", "beq"))
        need_J = "JE" in components
        need_H = ("H" in components) or ("lagrangian_hessian" in components)

        out = {}

        # cE & JE for restoration equalities
        if need_c or need_J:
            x, p, nn = self.split(bar_x)
            d = self.base.eval_all(x, components=["cE", "cI", "JE", "JI"])
            cE = _as_array_or_zeros(d.get("cE", None), self.m_base_E)
            cI = _as_array_or_zeros(d.get("cI", None), self.m_base_I)
            JE_x = _csr(d.get("JE", None), (self.m_base_E, self.n_orig))
            JI_x = _csr(d.get("JI", None), (self.m_base_I, self.n_orig))

            if self.m_eq > 0:
                c_stack = (np.concatenate([cE, cI]) if (self.m_base_E and self.m_base_I)
                           else (cE if self.m_base_E else cI))
                Jx = (sp.vstack([JE_x, JI_x], format="csr") if (self.m_base_E and self.m_base_I)
                      else (JE_x if self.m_base_E else JI_x))
                c_rest = c_stack - p + nn
                I = sp.eye(self.m_eq, format="csr")
                JE = sp.hstack([Jx, -I, I], format="csr")
            else:
                c_rest = np.zeros(0, dtype=float)
                JE = sp.csr_matrix((0, self.n_orig))
                Jx = sp.csr_matrix((0, self.n_orig))

            out["cE"] = c_rest
            out["cI"] = np.zeros(0, dtype=float)
            if need_J:
                out["JE"] = JE
                out["JI"] = sp.csr_matrix((0, self.n))  # 0 × (n+2m)

        # f & g
        if need_f or need_g:
            x, p, nn = self.split(bar_x)
            dx = x - self.x_R
            if need_f:
                out["f"] = 0.5 * self.zeta * float(dx @ (self.D_R @ dx)) + self.rho * (p.sum() + nn.sum())
            if need_g:
                gx = self.zeta * (self.D_R @ dx)
                if self.m_eq > 0:
                    gp = self.rho * np.ones(self.m_eq, dtype=float)
                    gn = self.rho * np.ones(self.m_eq, dtype=float)
                    out["g"] = np.concatenate([np.asarray(gx).ravel(), gp, gn])
                else:
                    out["g"] = np.asarray(gx).ravel()

        # Hessian (objective-only)
        if need_H:
            if self.m_eq > 0:
                Zm = sp.csr_matrix((self.m_eq, self.m_eq))
                H = sp.block_diag((self.zeta * self.D_R, Zm, Zm), format="csr")
            else:
                H = self.zeta * self.D_R
            out["H"] = H
            if "lagrangian_hessian" in components:
                out["lagrangian_hessian"] = H
        return out

    # ------------ REQUIRED by stepper ------------
    def lagrangian_hessian(self, bar_x, lam, nu):
        """
        Hessian of the Lagrangian wrt bar_x. Since constraints are linearized
        and have no curvature here, it equals the objective Hessian:
        block-diag( ζ·D_R , 0_m , 0_m ).
        """
        if self.m_eq > 0:
            Zm = sp.csr_matrix((self.m_eq, self.m_eq))
            return sp.block_diag((self.zeta * self.D_R, Zm, Zm), format="csr")
        return self.zeta * self.D_R

    def constraint_violation(self, bar_x) -> float:
        """
        Infinity-norm of restoration equalities only:
        θ = || [cE_base(x); cI_base(x)] − p + n ||_∞
        """
        x, p, nn = self.split(bar_x)
        d = self.base.eval_all(x, components=["cE", "cI"])
        cE = _as_array_or_zeros(d.get("cE", None), self.m_base_E)
        cI = _as_array_or_zeros(d.get("cI", None), self.m_base_I)
        if self.m_eq == 0:
            return 0.0
        c_stack = (np.concatenate([cE, cI]) if (self.m_base_E and self.m_base_I)
                   else (cE if self.m_base_E else cI))
        cres = c_stack - p + nn
        return float(np.linalg.norm(cres, np.inf))

# ------------------ stepper ------------------
class InteriorPointStepper:
    """
    Slacks-only barrier IP stepper with bounds:

        min f(x) - μ Σ log s
        s.t. c_I(x) + s = 0,  c_E(x) = 0,  s>0,  ℓ ≤ x ≤ u

    Uses Mehrotra predictor-corrector (σ from affine step) + centrality
    correction, unified fraction-to-boundary, SPD KKT default, inertia
    correction fallback, and a tiny x trust-region cap.
    """

    def __init__(
        self,
        cfg: SQPConfig,
        hess: HessianManager,
        regularizer: Regularizer,
        tr=None,
        flt: Optional[Filter] = None,
        funnel: Optional[Funnel] = None,
        ls: Optional[LineSearcher] = None,
        soc=None,
    ):
        self.cfg = cfg
        self.hess = hess
        self.reg = regularizer
        self.tr = tr
        self.filter = flt
        self.funnel = funnel
        self.ls = ls
        self.soc = soc
        self._ensure_cfg_defaults()

    # ---------- default knobs ----------
    def _ensure_cfg_defaults(self):
        def add(name, val):
            if not hasattr(self.cfg, name):
                setattr(self.cfg, name, val)

        add("ip_match_ref_form", False)   # we run Mehrotra PC by default
        add("ip_exact_hessian", True)
        add("ip_hess_reg0", 1e-4)
        add("ip_eq_reg", 1e-4)

        # barrier & predictor-corrector
        add("ip_mu_init", 1e-2)
        add("ip_mu_min", 1e-12)
        add("ip_sigma_power", 3.0)
        add("ip_tau", 0.995)       # fraction-to-boundary factor
        add("ip_tau_min", 0.99)
        add("ip_alpha_max", 1.0)

        # line-search bridge defaults
        add("ip_ls_max", 25)
        add("ip_alpha_min", 1e-10)
        add("ip_alpha_backtrack", 0.5)
        add("ip_armijo_coeff", 1e-4)
        if not hasattr(self.cfg, "ls_backtrack"): self.cfg.ls_backtrack = float(self.cfg.ip_alpha_backtrack)
        if not hasattr(self.cfg, "ls_armijo_f"):  self.cfg.ls_armijo_f  = float(self.cfg.ip_armijo_coeff)
        if not hasattr(self.cfg, "ls_max_iter"):  self.cfg.ls_max_iter  = int(self.cfg.ip_ls_max)
        if not hasattr(self.cfg, "ls_min_alpha"): self.cfg.ls_min_alpha = float(self.cfg.ip_alpha_min)

        # μ schedule (legacy)
        add("ip_mu_reduce_every", 10)
        add("ip_mu_reduce", 0.2)

        # error/control
        add("ip_rho_init", 1.0)
        add("ip_rho_inc", 10.0)
        add("ip_rho_max", 1e6)

        # Σ construction safety
        add("sigma_eps_abs", 1e-8)
        add("sigma_cap", 1e8)

        # restoration
        add("ip_rest_max_it", 8)
        add("ip_rest_max_retry", 5)
        add("ip_rest_mu", 1.0)
        add("ip_rest_zeta_init", 1.0)
        add("ip_rest_zeta_update", 5.0)
        add("ip_rest_zeta_max", 1e8)
        add("ip_rest_theta_drop", 0.5)
        add("ip_rest_theta_abs", 1e-6)

        # SPD KKT path default
        add("ip_kkt_method", "qdldl")  # "lifted" | "hykkt" | "qdldl"

        # tiny x trust-region
        add("ip_dx_max", 1e3)  # scale step when ||dx|| too large

        # conservative μ when infeasible
        add("ip_theta_clip", 1e-2)
        add("ip_enable_higher_order", True)  # keep off by default
        add("ip_ho_blend", 0.3)               # τ₂: how much of dx₂ to add
        add("ip_ho_rel_cap", 0.5)            # cap ‖dx₂‖ ≤ 0.5·‖dx‖

    # ---------- utilities ----------
    @staticmethod
    def _max_step_ftb(z: np.ndarray, dz: np.ndarray, tau: float) -> float:
        if z.size == 0 or dz.size == 0:
            return 1.0
        neg = dz < 0
        if not np.any(neg):
            return 1.0
        return float(min(1.0, tau * np.min(-z[neg] / dz[neg])))

    def _alpha_fraction_to_boundary(
        self,
        x, dx, s, ds, lmb, dlmb, lb, ub, hasL, hasU, tau: float,
    ) -> float:
        alphas = [1.0]

        # ineq: s > 0, λ > 0
        if s.size:
            neg = ds < 0
            if np.any(neg): alphas.append(float(np.min(-s[neg] / ds[neg])))
            neg = dlmb < 0
            if np.any(neg): alphas.append(float(np.min(-lmb[neg] / dlmb[neg])))

        # lower bounds: sL = x-ℓ > 0
        if np.any(hasL):
            sL = x - lb
            dxL = dx
            mask = hasL & (dxL < 0)
            if np.any(mask):
                alphas.append(float(np.min(-sL[mask] / dxL[mask])))

        # upper bounds: sU = u-x > 0 ⇒ sU step is -dx
        if np.any(hasU):
            sU = ub - x
            dxU = -dx
            mask = hasU & (dxU < 0)
            if np.any(mask):
                alphas.append(float(np.min(-sU[mask] / dxU[mask])))

        a = tau * min(alphas) if alphas else 1.0
        return float(max(0.0, min(1.0, a)))

    def _safe_reinit_slacks_duals(
        self, cI: np.ndarray, JI, x: np.ndarray, lb, ub, hasL, hasU, mu: float
    ):
        """Ensure (s, λ, zL, zU) are strictly interior & complementary."""
        n = x.size
        mI = int(cI.size)
        if mI > 0:
            s_floor = np.maximum(1e-8, np.minimum(1.0, np.abs(cI)))
            s0 = np.maximum(s_floor, -cI + 1e-3)
            lam0 = np.maximum(1e-8, mu / np.maximum(s0, 1e-12))
        else:
            s0 = np.zeros(0)
            lam0 = np.zeros(0)
        zL0 = np.zeros(n); zU0 = np.zeros(n)
        sL = np.where(hasL, np.maximum(1e-12, x - lb), 1.0)
        sU = np.where(hasU, np.maximum(1e-12, ub - x), 1.0)
        zL0[hasL] = np.maximum(1e-8, mu / sL[hasL])
        zU0[hasU] = np.maximum(1e-8, mu / sU[hasU])
        return s0, lam0, zL0, zU0

    # ---------- KKT solver ----------
    def solve_KKT(
        self,
        Wmat,
        rhs_x,
        JE_mat,
        rpE,
        use_ordering: bool = False,
        reuse_symbolic: bool = True,
        refine_iters: int = 0,
        *,
        method: str = "qdldl",  # default SPD
        gamma: float = None,
        delta_c_lift: float = None,
        cg_tol: float = 1e-10,
        cg_maxit: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve KKT:
            [ W   G^T ][dx] = [ rhs_x ]
            [ G    0  ][dy]   [ -rpE   ]

        method:
        - "hykkt"  : SPD K_gamma = (W + δ_w I) + γ G^T G, CG on Schur
        - "lifted" : SPD Schur with finite δ_c (dy = 1/δ_c (G dx - r2))
        - "qdldl"  : inertia-corrected LDL^T with qdldl
        """
        def _upper_csc(K):
            K = K.tocoo()
            mask = K.row <= K.col
            Kup = sp.coo_matrix((K.data[mask], (K.row[mask], K.col[mask])), shape=K.shape)
            return Kup.tocsc()

        def _normest_rowsum_inf(A: sp.spmatrix) -> float:
            A = _csr(A)
            return float(np.max(np.abs(A).sum(axis=1))) if A.shape[0] else 1.0

        n = Wmat.shape[0]
        mE = 0 if JE_mat is None else JE_mat.shape[0]
        W = _csr(Wmat, (n, n))
        G = None if JE_mat is None else _csr(JE_mat, (mE, n))
        r1 = rhs_x
        r2 = -rpE if mE > 0 else None

        # SPD paths if equality present
        if method in ("hykkt", "lifted") and mE > 0:
            delta_w_last = float(getattr(self, "_delta_w_last", 0.0) or 0.0)
            if gamma is None or delta_c_lift is None:
                num = _normest_rowsum_inf(W) + delta_w_last
                den = max(_normest_rowsum_inf(G), 1.0)
                gamma_hat = max(1.0, num / (den * den))

            if method == "hykkt":
                if gamma is None: gamma = gamma_hat
                Kgam = W + delta_w_last * sp.eye(n, format="csr") + gamma * (G.T @ G)
                Kgam_csc = Kgam.tocsc()

                def solve_Kgam(b):
                    return spla.spsolve(Kgam_csc, b)

                def S_mv(y):
                    return G @ solve_Kgam(G.T @ y)

                S = LinearOperator((mE, mE), matvec=S_mv, rmatvec=S_mv, dtype=float)
                svec = r1 + gamma * (G.T @ r2)
                rhs_schur = (G @ solve_Kgam(svec)) - r2
                dy, info = cg(S, rhs_schur, rtol=cg_tol, maxiter=cg_maxit)
                if info != 0:
                    # Retry with larger gamma; else fallback
                    gamma *= 5.0
                    Kgam = W + delta_w_last * sp.eye(n, format="csr") + gamma * (G.T @ G)
                    Kgam_csc = Kgam.tocsc()

                    def S_mv2(y):
                        return G @ spla.spsolve(Kgam_csc, G.T @ y)

                    S2 = LinearOperator((mE, mE), matvec=S_mv2, rmatvec=S_mv2, dtype=float)
                    svec = r1 + gamma * (G.T @ r2)
                    rhs_schur = (G @ spla.spsolve(Kgam_csc, svec)) - r2
                    dy, info = cg(S2, rhs_schur, rtol=max(cg_tol, 1e-8), maxiter=cg_maxit)
                    if info != 0:
                        method = "qdldl"

                if method == "hykkt" and info == 0:
                    rhs_dx = r1 + gamma * (G.T @ r2) - (G.T @ dy)
                    dx = solve_Kgam(rhs_dx)
                    self._delta_w_last = delta_w_last
                    return dx, dy

            if method == "lifted":
                if delta_c_lift is None: delta_c_lift = 1.0 / gamma_hat
                Kspd = W + delta_w_last * sp.eye(n, format="csr") + (1.0 / delta_c_lift) * (G.T @ G)
                rhs_dx = r1 - (1.0 / delta_c_lift) * r2
                dx = spla.spsolve(Kspd.tocsc(), rhs_dx)
                dy = (G @ dx - r2) * (1.0 / delta_c_lift)
                return dx, dy

        # LDLᵀ fallback with inertia correction
        delta_w_last = getattr(self, "_delta_w_last", 0.0)
        delta_w = 0.0
        delta_c = 0.0
        attempts = 0
        max_attempts = 10
        while attempts < max_attempts:
            if mE > 0:
                Wcsr = _csr(W + delta_w * sp.eye(n))
                JEcsr = G
                B22 = -delta_c * sp.eye(mE) if delta_c > 0 else sp.csr_matrix((mE, mE))
                K = sp.vstack([sp.hstack([Wcsr, JEcsr.T]), sp.hstack([JEcsr, B22])], format="csc")
                rhs = np.concatenate([r1, r2])
            else:
                K = _csr(W + delta_w * sp.eye(n))
                rhs = r1

            K, _ = make_psd_advanced(K, self.reg, attempts)
            K_upper = _upper_csc(K)
            nsys = K_upper.shape[0]

            # AMD ordering from your codebase
            from .aux.amd import AMDReorderingArray
            amd_ = AMDReorderingArray(aggressive_absorption=True)
            perm, _ = amd_.compute_fill_reducing_permutation(K_upper)

            try:
                fac = qd.factorize(K_upper.indptr, K_upper.indices, K_upper.data, nsys, perm=perm)
                D = fac.D
                num_pos = int(np.sum(D > 0))
                num_neg = int(np.sum(D < 0))
                num_zero = int(np.sum(np.abs(D) < 1e-12))

                nvar = n + mE
                if num_pos == n and num_neg == mE and num_zero == 0:
                    self._delta_w_last = delta_w
                    sol = qd.solve_refine(fac, rhs, refine_iters) if refine_iters > 0 else qd.solve(fac, rhs)
                    return (sol[:n], sol[n:]) if mE > 0 else (sol, np.zeros(0))

                if num_zero > 0:
                    delta_c = max(delta_c, 1e-8)

                if delta_w_last == 0:
                    delta_w = self.cfg.ip_hess_reg0 if attempts == 0 else 8.0 * max(self.cfg.ip_hess_reg0, delta_w)
                else:
                    delta_w = (max(1e-20, 1/3 * delta_w_last) if attempts == 0 else 8.0 * max(1e-20, delta_w))

                if delta_w > 1e40:
                    raise ValueError("Inertia correction failed: too large delta_w")

                attempts += 1

            except Exception:
                sol = spla.spsolve(K.tocsc(), rhs)
                return (sol[:n], sol[n:]) if mE > 0 else (sol, np.zeros(0))

        return None, None  # failed

    # ---------- error ----------
    def _compute_error(self, model: Model, x, lam, nu, zL, zU, mu: float = 0.0) -> float:
        data = model.eval_all(x, components=["f", "g", "cI", "JI", "cE", "JE"])
        g = np.asarray(data["g"], float)
        cI = np.asarray(data["cI"], float) if data["cI"] is not None else np.zeros(0)
        cE = np.asarray(data["cE"], float) if data["cE"] is not None else np.zeros(0)
        JI = data["JI"]
        JE = data["JE"]

        lb, ub, hasL, hasU = _get_bounds(model, x)
        sL = np.where(hasL, x - lb, 1.0)
        sU = np.where(hasU, ub - x, 1.0)

        r_d = g + (JI.T @ lam if JI is not None else 0) + (JE.T @ nu if JE is not None else 0) - zL + zU
        r_comp_L = np.where(hasL, sL * zL - mu, 0.0)
        r_comp_U = np.where(hasU, sU * zU - mu, 0.0)
        r_comp_slacks = (self.s * lam - mu * np.ones_like(self.s) if self.mI > 0 else np.zeros(0))

        s_max = float(getattr(self.cfg, "ip_s_max", 100.0))
        denom = self.mI + self.mE + x.size
        sum_mults = np.sum(np.abs(lam)) + np.sum(np.abs(nu)) + np.sum(np.abs(zL)) + np.sum(np.abs(zU))
        s_d = max(s_max, (sum_mults / max(1, denom))) / s_max
        s_c = (max(s_max, (np.sum(np.abs(zL)) + np.sum(np.abs(zU))) / max(1, x.size)) / s_max)

        err = max(
            np.linalg.norm(r_d, np.inf) / s_d,
            np.linalg.norm(cE, np.inf) if self.mE > 0 else 0.0,
            np.linalg.norm(cI, np.inf) if self.mI > 0 else 0.0,
            max(np.linalg.norm(r_comp_L, np.inf), np.linalg.norm(r_comp_U, np.inf)) / s_c,
            (np.linalg.norm(r_comp_slacks, np.inf) / s_c) if r_comp_slacks.size > 0 else 0.0,
        )
        return err

    # ---------- feasibility restoration ----------
    def _feasibility_restoration(self, model: Model, x: np.ndarray, mu: float, flt: Filter):
        print("Entering feasibility restoration phase")
        tol = float(getattr(self.cfg, "tol", 1e-8))
        theta_R = float(model.constraint_violation(x))
        if theta_R <= tol:
            print("Restoration not needed (already feasible enough).")
            return x.copy()

        dR = model.eval_all(x, components=["cE", "cI"])
        cE_R = _as_array_or_zeros(dR.get("cE", None), model.m_eq)
        cI_R = _as_array_or_zeros(dR.get("cI", None), model.m_ineq)
        c_R = np.concatenate([cE_R, cI_R]) if (cE_R.size or cI_R.size) else np.zeros(0)

        p0 = np.maximum(0.0, c_R)
        n0 = np.maximum(0.0, -c_R)

        x_R = x.copy()
        n = int(model.n)
        m = int(c_R.size)
        D_R_vec = 1.0 / np.maximum(1.0, np.abs(x_R))
        D_R = _diag(D_R_vec)

        bar_x0 = np.concatenate([x_R, p0, n0]) if m > 0 else x_R.copy()

        theta_target_rel = float(self.cfg.ip_rest_theta_drop) * theta_R
        theta_target_abs = max(float(self.cfg.ip_rest_theta_abs), tol**0.75)
        theta_goal = max(theta_target_abs, theta_target_rel)

        zeta = float(self.cfg.ip_rest_zeta_init)
        zeta_M = float(self.cfg.ip_rest_zeta_max)
        zeta_u = float(self.cfg.ip_rest_zeta_update)

        best_x = x_R.copy()
        best_theta = theta_R
        improved = False

        for retry in range(int(self.cfg.ip_rest_max_retry) + 1):
            print(f"[Restoration] attempt {retry+1}/{self.cfg.ip_rest_max_retry+1} with ζ={zeta:.2e}")
            resto_model = RestorationModel(model, x_R, D_R, zeta, rho=1.0)

            resto_cfg = copy.deepcopy(self.cfg)
            resto_cfg.ip_mu_init = float(self.cfg.ip_rest_mu)
            resto_cfg.ip_mu_min = max(1e-10, resto_cfg.ip_mu_min)

            resto_stepper = InteriorPointStepper(
                resto_cfg, self.hess, self.reg, self.tr, None, self.funnel, self.ls, soc=None
            )
            ip_state_resto = IPState.from_model(resto_model, bar_x0, resto_cfg)

            bar_x = bar_x0.copy()
            lamR = np.zeros(resto_model.m_eq, dtype=float)
            nuR = np.zeros(0, dtype=float)

            max_it = int(self.cfg.ip_rest_max_it)
            for k in range(max_it):
                bar_x, lamR, nuR, info = resto_stepper.step(resto_model, bar_x, lamR, nuR, k, ip_state_resto)
                x_try = bar_x[:n]
                theta_try = float(model.constraint_violation(x_try))
                print(f"   iter {k:02d}: θ={theta_try:.3e}")

                if theta_try + 1e-16 < best_theta:
                    best_theta = theta_try
                    best_x = x_try.copy()
                    improved = True
                if theta_try <= theta_goal:
                    break

            if best_theta <= theta_goal:
                break

            if zeta * zeta_u > zeta_M:
                print("[Restoration] ζ hit maximum; stopping escalation.")
                break
            zeta *= zeta_u

            d_best = model.eval_all(best_x, components=["cE", "cI"])
            cE_best = _as_array_or_zeros(d_best.get("cE", None), model.m_eq)
            cI_best = _as_array_or_zeros(d_best.get("cI", None), model.m_ineq)
            c_best = (np.concatenate([cE_best, cI_best]) if (cE_best.size or cI_best.size) else np.zeros(0))
            p_best = np.maximum(0.0, c_best)
            n_best = np.maximum(0.0, -c_best)
            bar_x0 = np.concatenate([best_x, p_best, n_best]) if c_best.size > 0 else best_x.copy()

        if not improved:
            raise ValueError("Restoration failed to decrease constraint violation")

        d_new = model.eval_all(best_x, components=["cI"])
        cI_new = _as_array_or_zeros(d_new.get("cI", None), model.m_ineq)
        self.s = np.maximum(1e-8, -cI_new) if cI_new.size else np.zeros(0)

        print(f"[Restoration] success: θ {theta_R:.3e} → {best_theta:.3e}")
        return best_x

    # ---------- one IP iteration ----------
    def step(
        self,
        model: Model,
        x: np.ndarray,
        lam: np.ndarray,
        nu: np.ndarray,
        it: int,
        ip_state: Optional[IPState] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        # init state
        st = ip_state if (ip_state and ip_state.initialized) else IPState.from_model(model, x, self.cfg)
        n = model.n
        mI, mE = st.mI, st.mE
        s = st.s.copy()
        lmb = st.lam.copy()
        nuv = st.nu.copy()
        zL = st.zL.copy()
        zU = st.zU.copy()
        mu = float(st.mu)
        tau = max(self.cfg.ip_tau_min, self.cfg.ip_tau)

        # evaluate model
        data = model.eval_all(x, components=["f", "g", "cI", "JI", "cE", "JE"])
        f = float(data["f"])
        g = np.asarray(data["g"], float)
        cI = np.zeros(mI) if mI == 0 else np.asarray(data["cI"], float)
        cE = np.zeros(mE) if mE == 0 else np.asarray(data["cE"], float)
        JI = data["JI"] if mI > 0 else None
        JE = data["JE"] if mE > 0 else None
        theta = model.constraint_violation(x)

        self.mI = mI
        self.mE = mE
        self.s = s

        # bounds & slacks
        lb, ub, hasL, hasU = _get_bounds(model, x)
        sL = np.where(hasL, np.maximum(1e-12, x - lb), 1.0)
        sU = np.where(hasU, np.maximum(1e-12, ub - x), 1.0)

        # if interior invalid -> reinit pairs
        bad = False
        if mI > 0:
            bad |= (not np.all(np.isfinite(s))) or np.any(s <= 0) or (not np.all(np.isfinite(lmb))) or np.any(lmb <= 0)
        bad |= (not np.all(np.isfinite(zL))) or np.any(zL < 0) or (not np.all(np.isfinite(zU))) or np.any(zU < 0)
        if bad:
            s, lmb, zL, zU = self._safe_reinit_slacks_duals(cI, JI, x, lb, ub, hasL, hasU, mu)
            sL = np.where(hasL, np.maximum(1e-12, x - lb), 1.0)
            sU = np.where(hasU, np.maximum(1e-12, ub - x), 1.0)

        # Hessian
        H = (model.lagrangian_hessian(x, lmb, nuv) if self.cfg.ip_exact_hessian
             else self.hess.get_hessian(model, x, lmb, nuv))
        H, _ = make_psd_advanced(H, self.reg, it)

        # residuals
        r_d = g + (JI.T @ lmb if JI is not None else 0) + (JE.T @ nuv if JE is not None else 0) - zL + zU
        r_pE = cE
        r_pI = cI + s if mI > 0 else np.zeros(0)

        err_0 = self._compute_error(model, x, lmb, nuv, zL, zU, 0.0)
        tol = float(getattr(self.cfg, "tol", 1e-8))
        if err_0 <= tol:
            info = {
                "mode": "ip", "step_norm": 0.0, "accepted": True, "converged": True,
                "f": f, "theta": theta, "stat": safe_inf_norm(r_d),
                "ineq": safe_inf_norm(np.maximum(0, cI)) if mI > 0 else 0.0,
                "eq": safe_inf_norm(cE) if mE > 0 else 0.0,
                "comp": 0.0, "ls_iters": 0, "alpha": 0.0, "rho": 0.0,
                "tr_radius": (self.tr.radius if self.tr else 0.0), "mu": mu,
            }
            return x, lmb, nuv, info

        # robust Σ
        eps_abs = float(getattr(self.cfg, "sigma_eps_abs", 1e-8))
        cap_val = float(getattr(self.cfg, "sigma_cap", 1e8))
        Sigma_L = np.where(hasL, zL / np.maximum(sL, eps_abs), 0.0)
        Sigma_U = np.where(hasU, zU / np.maximum(sU, eps_abs), 0.0)
        Sigma_x_vec = np.clip(Sigma_L + Sigma_U, 0.0, cap_val)
        if mI > 0:
            s_floor = np.maximum(1e-8, np.minimum(1.0, np.abs(cI)))
            s_safe = np.maximum(s, s_floor)
            Sigma_s_vec = np.clip(lmb / np.maximum(s_safe, eps_abs), 0.0, cap_val)
        else:
            Sigma_s_vec = np.zeros(0)

        # W
        W = H + _diag(Sigma_x_vec) + (JI.T @ _diag(Sigma_s_vec) @ JI if mI > 0 else 0)

        # Newton RHS for x
        rhs_x = -r_d

        # predictor: Mehrotra affine
        dx_aff, dnu_aff = self.solve_KKT(
            W, -r_d, JE, r_pE,
            method=getattr(self.cfg, "ip_kkt_method", "lifted"), cg_tol=1e-8, cg_maxit=200
        )
        if dx_aff is None:
            dx_aff, dnu_aff = self.solve_KKT(W, -r_d, JE, r_pE, method="qdldl")

        ds_aff = (-r_pI - (JI @ dx_aff if (mI > 0 and JI is not None) else 0)) if mI > 0 else np.zeros(0)
        dlam_aff = (-(s * lmb) - lmb * ds_aff) / np.maximum(s, 1e-16) if mI > 0 else np.zeros(0)
        dzL_aff = np.where(hasL, (-sL * zL - zL * dx_aff) / np.maximum(sL, 1e-16), 0.0)
        dzU_aff = np.where(hasU, (-sU * zU + zU * dx_aff) / np.maximum(sU, 1e-16), 0.0)

        # fraction-to-boundary for affine
        def _ftb_pos(z, dz):
            if z.size == 0: return 1.0
            neg = dz < 0
            return 1.0 if not np.any(neg) else float(min(1.0, self.cfg.ip_tau * np.min(-z[neg] / dz[neg])))

        alpha_aff = 1.0
        if mI > 0:
            alpha_aff = min(alpha_aff, _ftb_pos(s, ds_aff))
            alpha_aff = min(alpha_aff, _ftb_pos(lmb, dlam_aff))
        if np.any(hasL):
            alpha_aff = min(alpha_aff, _ftb_pos(sL[hasL], dx_aff[hasL]))
            alpha_aff = min(alpha_aff, _ftb_pos(zL[hasL], dzL_aff[hasL]))
        if np.any(hasU):
            alpha_aff = min(alpha_aff, _ftb_pos(sU[hasU], -dx_aff[hasU]))
            alpha_aff = min(alpha_aff, _ftb_pos(zU[hasU], dzU_aff[hasU]))

        mu_min = float(self.cfg.ip_mu_min)
        parts = []
        if mI > 0:
            s_aff = s + alpha_aff * ds_aff
            lam_aff = lmb + alpha_aff * dlam_aff
            parts.append(np.dot(s_aff, lam_aff))
        if np.any(hasL):
            sL_aff = sL + alpha_aff * dx_aff
            zL_aff = zL + alpha_aff * dzL_aff
            parts.append(np.dot(sL_aff[hasL], zL_aff[hasL]))
        if np.any(hasU):
            sU_aff = sU - alpha_aff * dx_aff
            zU_aff = zU + alpha_aff * dzU_aff
            parts.append(np.dot(sU_aff[hasU], zU_aff[hasU]))
        denom = (mI + int(np.sum(hasL)) + int(np.sum(hasU)))
        mu_aff = max(mu_min, (sum(parts) / max(1, denom)) if parts else mu)

        pwr = float(getattr(self.cfg, "ip_sigma_power", 3.0))
        sigma = float(np.clip((mu_aff / max(mu, mu_min)) ** pwr, 0.0, 1.0))

        # conservative μ if infeasible
        if float(theta) > float(self.cfg.ip_theta_clip):
            sigma = max(sigma, 0.5)

        comp_scale = 0.0
        if mI > 0:
            comp_scale += float(np.dot(s, lmb)) / max(1, mI)
        if np.any(hasL):
            comp_scale += float(np.sum((x[hasL] - lb[hasL]) * zL[hasL])) / max(1, int(np.sum(hasL)))
        if np.any(hasU):
            comp_scale += float(np.sum((ub[hasU] - x[hasU]) * zU[hasU])) / max(1, int(np.sum(hasU)))
        if comp_scale > 10.0 * mu:
            mu = min(comp_scale, 10.0)

        mu = max(mu_min, sigma * mu_aff)

        # corrector: centrality RHS (lightweight)
        if mI > 0:
            rc_s = (mu - s * lmb)
            rhs_x += (JI.T @ (_diag(Sigma_s_vec) @ (rc_s / np.maximum(lmb, 1e-12))))
        if np.any(hasL):
            rc_L = (mu - sL * zL)
            rhs_x += np.where(hasL, rc_L / np.maximum(sL, 1e-12), 0.0)
        if np.any(hasU):
            rc_U = (mu - sU * zU)
            rhs_x -= np.where(hasU, rc_U / np.maximum(sU, 1e-12), 0.0)

        # corrector solve
        dx, dnu = self.solve_KKT(
            W, rhs_x, JE, r_pE,
            method=getattr(self.cfg, "ip_kkt_method", "lifted"),
            cg_tol=1e-8, cg_maxit=200
        )
        if dx is None:
            dx, dnu = self.solve_KKT(W, rhs_x, JE, r_pE, method="qdldl")
            
        # recover ds, dλ
        ds = (-r_pI - (JI @ dx if (mI > 0 and JI is not None) else 0)) if mI > 0 else np.zeros(0)
        dlam = ((-(lmb - mu / np.maximum(s, 1e-16)) - _safe_pos_div(mu, s**2, eps=eps_abs) * ds)
                if mI > 0 else np.zeros(0))

        # bound duals steps via complementarity linearization
        dzL = np.where(hasL, _safe_pos_div(mu - sL * zL - zL * dx, sL, eps=eps_abs), 0.0)
        dzU = np.where(hasU, _safe_pos_div(mu - sU * zU + zU * dx, sU, eps=eps_abs), 0.0)

        # ----- Optional higher-order (2nd-order) correction -----
        if getattr(self.cfg, "ip_enable_higher_order", False) and dx.size:
            # Build a bounded quadratic RHS using current Σ_s and JI
            try:
                if mI > 0 and JI is not None and Sigma_s_vec.size:
                    # RHS ≈ - JIᵀ (diag(Σ_s) JI dx)   (truncated 2nd order term)
                    rhs_x2 = -(JI.T @ (_diag(Sigma_s_vec) @ (JI @ dx)))
                else:
                    rhs_x2 = np.zeros_like(dx)

                # Solve with the same SPD system (reuse W, JE); keep equality RHS = 0
                dx2, dnu2 = self.solve_KKT(
                    W, rhs_x2, JE, np.zeros_like(r_pE) if mE > 0 else None,
                    method=getattr(self.cfg, "ip_kkt_method", "lifted"),
                    cg_tol=1e-8, cg_maxit=200
                )
                if dx2 is None:
                    dx2, dnu2 = self.solve_KKT(W, rhs_x2, JE, np.zeros_like(r_pE) if mE > 0 else None, method="qdldl")

                # Safety: cap size of dx2 relative to dx and blend
                tau2   = float(getattr(self.cfg, "ip_ho_blend", 0.3))   # 0.1–0.5 is typical
                caprel = float(getattr(self.cfg, "ip_ho_rel_cap", 0.5)) # ‖dx2‖ ≤ caprel·‖dx‖
                n_dx   = float(np.linalg.norm(dx))
                n_dx2  = float(np.linalg.norm(dx2))
                if n_dx2 > caprel * max(1e-16, n_dx):
                    dx2 *= (caprel * max(1e-16, n_dx) / n_dx2)

                # Update primals with the correction
                dx += tau2 * dx2

                # Update inequality pieces consistently (only if present)
                if mI > 0:
                    ds2   = -(JI @ dx2) if JI is not None else np.zeros(mI)
                    ds   += tau2 * ds2
                    dlam += tau2 * ( - _safe_pos_div(mu, np.maximum(s,1e-16)**2, eps=eps_abs) * ds2 )

                # Bound dual corrections (linearized)
                if np.any(hasL):
                    dzL += tau2 * _safe_pos_div(- zL * dx2, sL, eps=eps_abs)
                if np.any(hasU):
                    dzU += tau2 * _safe_pos_div(  zU * dx2, sU, eps=eps_abs)

            except Exception:
                # If anything goes weird, just skip the HO step
                pass


        # tiny x trust-region cap
        dx_norm = float(np.linalg.norm(dx))
        dx_cap = float(self.cfg.ip_dx_max)
        if dx_norm > dx_cap and dx_norm > 0:
            scale = dx_cap / dx_norm
            dx *= scale
            if mI > 0: ds *= scale; dlam *= scale
            dzL *= scale; dzU *= scale

        # unified fraction-to-boundary
        alpha_ftb = self._alpha_fraction_to_boundary(
            x, dx, s, ds, lmb, dlam, lb, ub, hasL, hasU, tau
        )

        # line-search (if provided)
        alpha_max = min(alpha_ftb, float(self.cfg.ip_alpha_max))
        try:
            alpha, ls_iters, needs_restoration = self.ls.search_ip(
                model=model, x=x, dx=dx, ds=ds, s=s, mu=mu,
                d_phi=float(g @ dx),
                theta0=theta, alpha_max=alpha_max,
            )
        except Exception:
            alpha, ls_iters, needs_restoration = min(1.0, alpha_max), 0, False

        # early restoration / recenter
        if (alpha <= float(self.cfg.ls_min_alpha)) and needs_restoration:
            try:
                x_new = self._feasibility_restoration(model, x, mu, self.filter)
                data_new = model.eval_all(x_new, ["f", "g", "cI", "cE"])
                lmb_new = np.maximum(1e-8, lmb) if mI > 0 else lmb
                nu_new  = np.zeros(mE) if mE > 0 else nuv
                zL_new  = np.where(hasL, np.maximum(1e-8, zL), 0.0)
                zU_new  = np.where(hasU, np.maximum(1e-8, zU), 0.0)
                info = {
                    "mode": "ip", "step_norm": float(np.linalg.norm(x_new - x)),
                    "accepted": True, "converged": False,
                    "f": float(data_new["f"]), "theta": float(model.constraint_violation(x_new)),
                    "stat": 0.0, "ineq": 0.0, "eq": 0.0, "comp": 0.0,
                    "ls_iters": ls_iters, "alpha": 0.0, "rho": 0.0,
                    "tr_radius": (self.tr.radius if self.tr else 0.0), "mu": mu,
                }
                st.s, st.lam, st.nu, st.zL, st.zU = s, lmb_new, nu_new, zL_new, zU_new
                return x_new, lmb_new, nu_new, info
            except ValueError:
                # recenter at current x: reset pairs and take tiny safe step along -g
                s, lmb, zL, zU = self._safe_reinit_slacks_duals(cI, JI, x, lb, ub, hasL, hasU, mu=max(mu, 1e-2))
                dxf = -g / max(1.0, np.linalg.norm(g))
                alpha = min(alpha_max, 1e-2)
                x_new = x + alpha * dxf
                info = {
                    "mode": "ip", "step_norm": float(np.linalg.norm(x_new - x)),
                    "accepted": True, "converged": False,
                    "f": model.eval_all(x_new, ["f"])["f"], "theta": float(model.constraint_violation(x_new)),
                    "stat": 0.0, "ineq": 0.0, "eq": 0.0, "comp": 0.0,
                    "ls_iters": ls_iters, "alpha": alpha, "rho": 0.0,
                    "tr_radius": (self.tr.radius if self.tr else 0.0), "mu": mu,
                }
                st.s, st.lam, st.nu, st.zL, st.zU = s, lmb, nuv, zL, zU
                return x_new, lmb, nuv, info

        # accept step (cap bound duals with σ-neighborhood)
        alpha_bz = alpha  # we already limited by fraction-to-boundary for z via dz formulas
        x_new  = x + alpha * dx
        s_new  = s + alpha * ds if mI > 0 else s
        lmb_new = lmb + alpha * dlam if mI > 0 else lmb
        nu_new  = nuv + alpha * dnu if mE > 0 else nuv
        zL_new  = zL + alpha_bz * dzL
        zU_new  = zU + alpha_bz * dzU

        ksig = 1e10  # big box to avoid collapse but keep positivity
        sL_new = np.where(hasL, x_new - lb, 1.0)
        sU_new = np.where(hasU, ub - x_new, 1.0)
        if np.any(hasL):
            sL_clip = np.maximum(sL_new, 1e-16)
            zL_new = np.maximum(mu / (ksig * sL_clip), np.minimum(zL_new, ksig * mu / sL_clip))
        if np.any(hasU):
            sU_clip = np.maximum(sU_new, 1e-16)
            zU_new = np.maximum(mu / (ksig * sU_clip), np.minimum(zU_new, ksig * mu / sU_clip))

        # recompute at x_new (for report)
        data_new = model.eval_all(x_new, ["f", "g", "cI", "JI", "cE", "JE"])
        f_new  = float(data_new["f"])
        g_new  = np.asarray(data_new["g"], float)
        cI_new = np.asarray(data_new["cI"], float) if mI > 0 else np.zeros(0)
        cE_new = np.asarray(data_new["cE"], float) if mE > 0 else np.zeros(0)
        JI_new = data_new["JI"] if mI > 0 else None
        JE_new = data_new["JE"] if mE > 0 else None
        theta_new = model.constraint_violation(x_new)

        r_d_new = g_new \
            + (JI_new.T @ lmb_new if (mI > 0 and JI_new is not None) else 0) \
            + (JE_new.T @ nu_new  if (mE > 0 and JE_new is not None) else 0) \
            - zL_new + zU_new

        r_comp_L_new = np.where(hasL, (x_new - lb) * zL_new - mu, 0.0) if np.any(hasL) else 0.0
        r_comp_U_new = np.where(hasU, (ub - x_new) * zU_new - mu, 0.0) if np.any(hasU) else 0.0
        r_comp_s_new = (s_new * lmb_new - mu) if mI > 0 else np.zeros(0)

        kkt_new = {
            "stat": np.linalg.norm(r_d_new, np.inf),
            "ineq": np.linalg.norm(np.maximum(0, cI_new), np.inf) if mI > 0 else 0.0,
            "eq":   np.linalg.norm(cE_new, np.inf) if mE > 0 else 0.0,
            "comp": max(
                (np.linalg.norm(r_comp_L_new, np.inf) if np.any(hasL) else 0.0),
                (np.linalg.norm(r_comp_U_new, np.inf) if np.any(hasU) else 0.0),
            ) + (np.linalg.norm(r_comp_s_new, np.inf) if r_comp_s_new.size > 0 else 0.0),
        }

        converged = (
            kkt_new["stat"] <= tol and kkt_new["ineq"] <= tol and kkt_new["eq"] <= tol
            and kkt_new["comp"] <= tol and mu <= tol / 10
        )

        info = {
            "mode": "ip",
            "step_norm": float(np.linalg.norm(x_new - x)),
            "accepted": True,
            "converged": converged,
            "f": f_new,
            "theta": theta_new,
            "stat": kkt_new["stat"],
            "ineq": kkt_new["ineq"],
            "eq": kkt_new["eq"],
            "comp": kkt_new["comp"],
            "ls_iters": 0,
            "alpha": float(alpha),
            "rho": 0.0,
            "tr_radius": (self.tr.radius if self.tr else 0.0),
            "mu": mu,
        }

        # persist state
        st.s, st.lam, st.nu, st.zL, st.zU, st.mu = (s_new, lmb_new, nu_new, zL_new, zU_new, mu)
        return x_new, lmb_new, nu_new, info
