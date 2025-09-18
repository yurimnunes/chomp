from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp

#from .blocks.tr import *
import tr_cpp as tr_cpp

# Reuse shared infrastructure
from .blocks.aux import (
    HessianManager,
    Model,
    RestorationManager,
    SQPConfig,
)
from .blocks.filter import *
from .blocks.linesearch import *
from .blocks.qp import *
from .blocks.reg import DOptStabilizer, Regularizer, make_psd_advanced
from .blocks.soc import *


def _to_dense_if_needed(matrix, requires_dense: bool):
    """Convert sparse matrix to dense if required by solver."""
    return matrix.toarray() if sp.issparse(matrix) and requires_dense else matrix

# =============================================================================
# SQP Stepper
# =============================================================================
class SQPStepper:
    """
    Trust-Region SQP stepper with optional Byrd–Omojokun composite step:
      p = n + t, with TR management, filter acceptance, SOC correction, and bounds.

    Changes vs your previous version:
      • Bounds are stacked into the QP subproblem (box-aware step).
      • Line search step α is always capped by the box.
      • SOC is triggered by violation not reducing enough (configurable).
      • Inequality multipliers are split: user-ineq vs bound-ineq.
    """

    # -------------- helpers --------------
    @staticmethod
    def _validate_config(cfg: SQPConfig) -> SQPConfig:
        defaults = {
            "tol_feas": 1e-6,
            "tol_stat": 1e-6,
            "tol_comp": 1e-6,
            "tol_obj_change": 1e-12,
            "tr_delta0": 1.0,
            "tr_delta_min": 1e-6,
            "tr_gamma_dec": 0.5,
            "tr_eta_lo": 0.1,
            "act_tol": 1e-12,
            "cs_ineq_weight": 1.0,
            "cs_damping": 1e-8,
            "cond_threshold": 1e6,
            "feas_opt_ratio": 1e2,
            "use_soc": True,
            "filter_theta_min": 1e-8,
            "hessian_mode": "exact",
            # Stabilization (disabled by default)
            "stabilize_dopt": False,
            "dopt_sigma_E": 1e-2,
            "dopt_sigma_I": 1e-2,
            "dopt_mu_target": 1e-6,
            "dopt_active_tol": 5e-3,
            "dopt_max_shift": 1e-1,
            "dopt_scaling": "colnorm",
            # New: SOC trigger based on violation improvement
            "soc_violation_ratio": 0.9,   # do SOC if θ_lin > 0.9 * θ0
        }
        for k, v in defaults.items():
            if not hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    @staticmethod
    def _alpha_max_box(x: np.ndarray,
                       p: np.ndarray,
                       lb: Optional[np.ndarray],
                       ub: Optional[np.ndarray]) -> float:
        if lb is None and ub is None:
            return 1.0
        lo = lb if lb is not None else -np.inf
        hi = ub if ub is not None else +np.inf
        amax = 1.0
        pos = p > 0
        neg = p < 0
        if np.any(pos):
            amax = min(amax, float(np.min((hi[pos] - x[pos]) / p[pos])))
        if np.any(neg):
            amax = min(amax, float(np.min((lo[neg] - x[neg]) / p[neg])))
        return max(0.0, amax)

    @staticmethod
    def _stack_rows(A, B):
        if A is None or (hasattr(A, "size") and A.size == 0):
            return B
        if B is None or (hasattr(B, "size") and B.size == 0):
            return A
        return np.vstack([A, B])

    @staticmethod
    def _stack_vec(a, b):
        if a is None or (hasattr(a, "size") and a.size == 0):
            return b
        if b is None or (hasattr(b, "size") and b.size == 0):
            return a
        return np.concatenate([a, b])

    # -------------- init --------------
    def __init__(
        self,
        cfg: SQPConfig,
        hess: HessianManager,
        tr: Optional[TrustRegionManager],
        ls: Optional[LineSearcher],
        qp: QPSolver,
        soc: SOCCorrector,
        regularizer: Regularizer,
        restoration: Optional[RestorationManager] = None,
    ):
        self.cfg = self._validate_config(cfg)
        self.hess = hess
        tr_conf = tr_cpp.TRConfig()
        self.tr = tr_cpp.TrustRegionManager(tr_conf)
        self.ls = ls
        self.flt = ls.filter if ls is not None else Filter(self.cfg)
        self.qp = qp
        self.soc = soc
        self.regularizer = regularizer
        self.dopt = DOptStabilizer(self.cfg)
        self.restoration = restoration
        self.requires_dense = getattr(qp, "requires_dense", False)

    # -------------- main step --------------
    def step(
        self,
        model: Model,
        x: np.ndarray,
        lam: np.ndarray,   # inequality multipliers at x (user)
        nu: np.ndarray,    # equality multipliers at x
        it: int,
    ):
        # ---- 0) Box: project x once for robustness -----------------------
        lb = getattr(model, "lb", None)
        ub = getattr(model, "ub", None)
        lb = None if lb is None else np.asarray(lb, float).ravel()
        ub = None if ub is None else np.asarray(ub, float).ravel()
        if (lb is not None and lb.size != x.size) or (ub is not None and ub.size != x.size):
            raise ValueError("model.lb/ub must have same length as x")
        if lb is not None or ub is not None:
            lo = lb if lb is not None else -np.inf
            hi = ub if ub is not None else +np.inf
            x = np.clip(x, lo, hi)

        # ---- 1) Evaluate once at x (skip objective Hessian here) ----------
        need = ["f", "g", "cI", "JI", "cE", "JE"]
        d0 = model.eval_all(x, components=need)
        f0  = float(d0["f"])
        g0  = np.asarray(d0["g"], float).ravel()
        JI  = d0.get("JI", None)
        JE  = d0.get("JE", None)
        cI  = d0.get("cI", None)
        cE  = d0.get("cE", None)
        theta0 = float(model.constraint_violation(x))

        # ---- 2) Lagrangian Hessian + PSD fix ------------------------------
        H_raw = model.lagrangian_hessian(x, lam, nu)
        H_psd, _ = self.regularizer.regularize(H_raw, it, np.linalg.norm(g0), self.tr.delta)

        # keep TR metric consistent if using ellipsoidal norm
        if self.tr and getattr(self.tr, "norm_type", "2") == "ellip":
            self.tr.update_metric(H_psd)

        # (optional) densify only if required by your TR/QP backend
        Hd  = _to_dense_if_needed(H_psd, self.requires_dense)
        JId = _to_dense_if_needed(JI, self.requires_dense)
        JEd = _to_dense_if_needed(JE, self.requires_dense)
        cId = None if cI is None else np.asarray(cI, float).ravel()
        cEd = None if cE is None else np.asarray(cE, float).ravel()

        # linearize JI @ p + cI <= 0, JE @ p + cE = 0
        # ---- 3) Trust-region solve (TR handles box/SOC/filter/Δ) ----------
        p, tr_info, lam_user, nu_new = self.tr.solve_dense(
            Hd, g0,
            A_ineq=JId, b_ineq=cId,
            A_eq=JEd,  b_eq=cEd,
            x=x,
            model=model,
            mu=getattr(model, "mu", 0.0) or 0.0,   # barrier scalar if using IP; NOT nu
            filter=self.flt,                       # your Fletcher–Leyffer filter
            f_old=f0,                              # let TR compute actual/pred ratios cleanly
        )

        # TR hard-fail guard
        if (p is None) or (not np.all(np.isfinite(p))):
            info = self._pack_info(
                step_norm=0.0, accepted=False, converged=False,
                f_val=f0, theta=theta0,
                kkt={"stat": np.inf, "ineq": np.inf, "eq": np.inf, "comp": np.inf},
                ls_iters=0, alpha=0.0, rho=0.0
            )
            return x, lam, nu, info
        

        # ---- 4) Trial evaluation -----------------------------------------
        s = p  # TR already clipped to Δ and box, and applied SOC if any
        x_trial = x + s
        d1 = model.eval_all(x_trial, components=["f","g","JI","JE","cI","cE"])
        f1  = float(d1["f"])
        g1  = np.asarray(d1["g"], float).ravel()
        JE1 = d1.get("JE", None)
        JI1 = d1.get("JI", None)
        cE1 = d1.get("cE", None)
        cI1 = d1.get("cI", None)
        theta1 = float(model.constraint_violation(x_trial))

        # Bound multipliers from TR (already expanded to size n)
        z_L = tr_info.get("mu_lower", np.zeros_like(x))
        z_U = tr_info.get("mu_upper", np.zeros_like(x))

        # ---- 5) KKT residuals at trial -----------------------------------
        def kkt_full(g, JE, JI, cE, cI, xval):
            r = g.copy()
            if JE is not None and getattr(JE, "size", 0):
                r += JE.T @ nu_new
            if JI is not None and getattr(JI, "size", 0) and lam_user is not None and lam_user.size:
                r += JI.T @ lam_user
            r += (z_U - z_L)
            stat = float(np.linalg.norm(r, ord=np.inf))

            feas_eq = float(np.linalg.norm(cE, ord=np.inf)) if cE is not None else 0.0
            feas_in = float(np.linalg.norm(np.maximum(cI, 0.0), ord=np.inf)) if cI is not None else 0.0
            feas_box = 0.0
            if lb is not None:
                feas_box = max(feas_box, float(np.linalg.norm(np.maximum(lb - xval, 0.0), ord=np.inf)))
            if ub is not None:
                feas_box = max(feas_box, float(np.linalg.norm(np.maximum(xval - ub, 0.0), ord=np.inf)))
            ineq = max(feas_in, feas_box)

            comp = 0.0
            if cI is not None and lam_user is not None and lam_user.size:
                comp = max(comp, float(np.linalg.norm(lam_user * np.maximum(cI, 0.0), ord=np.inf)))
            if lb is not None:
                comp = max(comp, float(np.linalg.norm(z_L * np.maximum(xval - lb, 0.0), ord=np.inf)))
            if ub is not None:
                comp = max(comp, float(np.linalg.norm(z_U * np.maximum(ub - xval, 0.0), ord=np.inf)))
            return {"stat": stat, "eq": feas_eq, "ineq": ineq, "comp": comp}

        kkt = kkt_full(g1, JE1, JI1, cE1, cI1, x_trial)

        # ---- 6) Convergence test -----------------------------------------
        g_scale = max(
            1.0,
            np.linalg.norm(g0) + max(
                np.linalg.norm(cE) if cE is not None else 0.0,
                np.linalg.norm(cI) if cI is not None else 0.0,
            ),
        )
        converged = (
            kkt["stat"]/g_scale <= self.cfg.tol_stat and
            kkt["ineq"]/g_scale <= self.cfg.tol_feas and
            kkt["eq"]  /g_scale <= self.cfg.tol_feas and
            (kkt["comp"]/g_scale <= self.cfg.tol_comp or kkt["ineq"] <= 1e-10)
        ) or (abs(f1 - f0) < self.cfg.tol_obj_change)

        # ---- 7) Optional quasi-Newton update of external H approx ---------
        if self.cfg.hessian_mode in ["bfgs","lbfgs","hybrid"]:
            self.hess.update(s, g1 - g0)

        # ---- 8) Pack & return --------------------------------------------
        step_norm = float(self.tr._tr_norm(s)) if (self.tr and hasattr(self.tr, "_tr_norm")) \
                    else float(np.linalg.norm(s))
        rho = tr_info.get("rho", None)  # the TR can compute/store this
        info = self._pack_info(
            step_norm=step_norm,
            accepted=bool(tr_info.get("accepted", True)),
            converged=converged,
            f_val=f1,
            theta=theta1,
            kkt=kkt,
            ls_iters=0,
            alpha=1.0,
            rho=(float(rho) if rho is not None else 0.0),
        )
        info["delta"] = getattr(self.tr, "delta", None)
        info["tr"] = tr_info
        return x_trial, (lam_user if lam_user is not None else np.zeros(0)), nu_new, info

    # -------------- utilities --------------
    def _is_kkt(self, kkt, g_norm, c_norm):
        scale = max(1.0, g_norm + c_norm)
        return (
            kkt["stat"] / scale <= self.cfg.tol_stat
            and kkt["ineq"] / scale <= self.cfg.tol_feas
            and kkt["eq"] / scale <= self.cfg.tol_feas
            and (kkt["comp"] / scale <= self.cfg.tol_comp or kkt["ineq"] <= 1e-10)
        )

    def _pack_info(
        self,
        step_norm: float,
        accepted: bool,
        converged: bool,
        f_val: float,
        theta: float,
        kkt: Dict,
        ls_iters: int,
        alpha: float,
        rho: float,
    ) -> Dict:
        return {
            "step_norm": step_norm,
            "accepted": accepted,
            "converged": converged,
            "f": f_val,
            "theta": theta,
            "stat": kkt["stat"],
            "ineq": kkt["ineq"],
            "eq": kkt["eq"],
            "comp": kkt["comp"],
            "ls_iters": ls_iters,
            "alpha": alpha,
            "rho": rho,
            "tr_radius": (self.tr.delta if self.tr else 0.0),
        }