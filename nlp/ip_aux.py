
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix as _csr

# External project modules (unchanged API)
from .blocks.aux import Model, SQPConfig
from .ip_cg import *


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
    tau_shift: float = 0.0  # shift parameter for shifted barrier
    initialized: bool = False

    @staticmethod
    def from_model(model: Model, x: np.ndarray, cfg: SQPConfig) -> "IPState":
        mI, mE = model.m_ineq, model.m_eq
        d = model.eval_all(x, components=["cI", "cE"])
        cI = np.zeros(mI) if (mI == 0 or d["cI"] is None) else np.asarray(d["cI"], float)

        mu0 = max(getattr(cfg, "ip_mu_init", 1e-2), 1e-12)
        use_shifted = bool(getattr(cfg, "ip_use_shifted_barrier", True))
        tau_shift = float(getattr(cfg, "ip_shift_tau", 0.0)) if use_shifted else 0.0

        if mI > 0:
            s0 = np.maximum(1.0, -cI + 1e-3)
            lam0 = np.maximum(1e-8, mu0 / np.maximum(s0 + tau_shift, 1e-12) if tau_shift > 0 else mu0 / np.maximum(s0, 1e-12))
        else:
            s0 = np.zeros(0)
            lam0 = np.zeros(0)
        nu0 = np.zeros(mE)

        lb = getattr(model, "lb", None)
        ub = getattr(model, "ub", None)
        lb = np.full_like(x, -np.inf) if lb is None else np.asarray(lb, float)
        ub = np.full_like(x, +np.inf) if ub is None else np.asarray(ub, float)
        hasL = np.isfinite(lb)
        hasU = np.isfinite(ub)
        sL = np.where(hasL, np.maximum(1e-12, x - lb), 1.0)
        sU = np.where(hasU, np.maximum(1e-12, ub - x), 1.0)

        bound_shift = float(getattr(cfg, "ip_shift_bounds", 0.0)) if use_shifted else 0.0
        zL0 = np.where(hasL, np.maximum(1e-8, mu0 / np.maximum(sL + bound_shift, 1e-12)), 0.0)
        zU0 = np.where(hasU, np.maximum(1e-8, mu0 / np.maximum(sU + bound_shift, 1e-12)), 0.0)

        return IPState(
            mI=mI, mE=mE, s=s0, lam=lam0, nu=nu0, zL=zL0, zU=zU0, mu=mu0,
            tau_shift=tau_shift, initialized=True
        )


def _csr(A, shape=None):
    if A is None:
        if shape is None:
            raise ValueError("shape required when A is None")
        return sp.csr_matrix(shape)
    return A.tocsr() if _isspm(A) else sp.csr_matrix(A)


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
