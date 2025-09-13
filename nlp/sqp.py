# nlp_hybrid.py
# Hybrid NLP Solver: Adaptive Interior-Point (Mehrotra, slacks-only barrier) + TR-SQP
# - Inequalities: cI(x) + s = 0, s > 0 (log barrier on slacks)
# - Uses Mehrotra predictor-corrector with fraction-to-boundary rule for IP
# - TR-SQP with persistent PIQP QP, optional L2-CG trust-region, SOC corrector
# - Auto-switches between IP and SQP based on θ/μ thresholds and stall detection
# - Refactored SQP logic into SQPStepper class for modularity
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import scipy.sparse as sp

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
from .blocks.reg import Regularizer, make_psd_advanced
from .blocks.soc import *
from .blocks.tr import *


def _to_dense_if_needed(matrix, requires_dense: bool):
    """Convert sparse matrix to dense if required by solver."""
    return matrix.toarray() if sp.issparse(matrix) and requires_dense else matrix

# =============================================================================
# SQP Stepper
# =============================================================================
class SQPStepper:
    """
    Trust-Region SQP stepper with optional Byrd-Omojokun composite step:
    - p = n + t, where:
      - n: normal step (reduces constraint violation via damped least-squares)
      - t: tangential step (optimizes Lagrangian in null space of JE, within TR)
    - Supports sparse matrices for efficiency.
    - Returns (x_out, lam_out, nu_out, info_dict).
    """
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
        self.tr = tr
        self.ls = ls
        self.flt = ls.filter if ls is not None else None
        self.qp = qp
        self.soc = soc
        self.regularizer = regularizer
        self.restoration = restoration
        self.requires_dense = getattr(qp, "requires_dense", False)

    def _validate_config(self, cfg: SQPConfig) -> SQPConfig:
        """Ensure SQPConfig has required fields with sensible defaults."""
        defaults = {
            "tol_feas": 1e-6,  # Relaxed from 1e-8
            "tol_stat": 1e-6,  # Relaxed from 1e-8
            "tol_comp": 1e-6,  # Relaxed from 1e-8
            "tol_obj_change": 1e-8,  # New: stop if objective change is small
            "tr_delta0": 1.0,
            "tr_delta_min": 1e-6,
            "tr_gamma_dec": 0.5,
            "tr_eta_lo": 0.1,
            "act_tol": 1e-6,
            "cs_ineq_weight": 1.0,
            "cs_damping": 1e-8,
            "cond_threshold": 1e6,
            "feas_opt_ratio": 1e2,
            "use_composite_step": False,
            "use_byrd_omojokun": False,
            "byrd_omojokun_kappa": 0.8,
            "byrd_omojokun_penalty": 1e2,
            "use_soc": False,
            "filter_theta_min": 1e-8,
            "hessian_mode": "exact",
        }
        for key, value in defaults.items():
            if not hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg

    def step(
        self,
        model: Model,
        x: np.ndarray,
        lam: np.ndarray,
        nu: np.ndarray,
        it: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Patched SQP step:
        - Always get p from TR (which already does normal/tangential & active-set).
        - Line search on p -> α.
        - (Optional) SOC after line search; clip with TR radius via TR helper.
        - Predicted reduction uses model at α p (NO SOC).
        - Actual reduction uses f(x + α p + q) (WITH SOC).
        - TR update uses (pred_red, act_red, ||s||, θ_new).
        """
        # --- Evaluate model at current point ---
        data0 = model.eval_all(x)
        f0: float = float(data0["f"])
        g0: np.ndarray = data0["g"]
        theta0: float = model.constraint_violation(x)
        kkt0 = model.kkt_residuals(x, lam, nu)

        # --- Build (regularized) Hessian of the Lagrangian ---
        H = self.hess.get_hessian(model, x, lam, nu)
        H, _ = make_psd_advanced(
            H, self.regularizer, it, model_quality=0.0, constraint_count=len(nu)
        )

        # --- Optionally refresh TR metric (elliptic norm) from H ---
        if self.tr and hasattr(self.tr, "set_metric_from_H") and getattr(self.tr, "norm_type", "2") == "ellip":
            self.tr.set_metric_from_H(H)

        # --- Solve TR subproblem (composite step lives inside TR) ---
        p, tr_info, lam_new, nu_new = self.tr.solve(
            _to_dense_if_needed(H, self.requires_dense),
            g0,
            _to_dense_if_needed(data0["JI"], self.requires_dense),
            data0["cI"],
            _to_dense_if_needed(data0["JE"], self.requires_dense),
            data0["cE"],
        )

        # Guard: TR failed or returned non-finite step
        if p is None or not np.all(np.isfinite(p)):
            # shrink TR and try restoration if available
            if self.tr:
                self.tr.delta *= self.cfg.tr_gamma_dec
            info = self._pack_info(0.0, False, False, f0, theta0, kkt0, 0, 0.0, 0.0)
            if self.restoration:
                trR = self.tr.delta if self.tr else self.cfg.tr_delta0
                pr, meta = self.restoration.try_restore(model, x, trR)
                if meta.get("ok", False) and pr is not None and np.all(np.isfinite(pr)):
                    x2 = x + pr
                    d2 = model.eval_all(x2)
                    k2 = model.kkt_residuals(x2, lam, nu)
                    info = self._pack_info(
                        float(np.linalg.norm(pr)), True, False,
                        float(d2["f"]), model.constraint_violation(x2), k2, 0, 0.0, 0.0
                    )
                    if self.tr:
                        self.tr.delta = max(self.cfg.tr_delta_min, 0.5 * trR)
                    return x2, lam, nu, info
            return x, lam, nu, info

        # --- Line search on p ---
        alpha, ls_iters, _ = (1.0, 0, None)
        if self.ls:
            alpha, ls_iters, _ = self.ls.search_sqp(model, x, p)
            if alpha <= 1e-12:
                # Try restoration if step is blocked
                info = self._pack_info(0.0, False, False, f0, theta0, kkt0, ls_iters, 0.0, 0.0)
                if self.restoration:
                    trR = self.tr.delta if self.tr else self.cfg.tr_delta0
                    pr, meta = self.restoration.try_restore(model, x, trR)
                    if meta.get("ok", False) and pr is not None and np.all(np.isfinite(pr)):
                        x2 = x + pr
                        d2 = model.eval_all(x2)
                        k2 = model.kkt_residuals(x2, lam, nu)
                        info = self._pack_info(
                            float(np.linalg.norm(pr)), True, False,
                            float(d2["f"]), model.constraint_violation(x2), k2, ls_iters, 0.0, 0.0
                        )
                        if self.tr:
                            self.tr.delta = max(self.cfg.tr_delta_min, 0.5 * trR)
                        return x2, lam, nu, info
                return x, lam, nu, info

        # --- Predicted reduction for α p (NO SOC here) ---
        pred_red = self.tr.model_reduction_alpha(
            _to_dense_if_needed(H, self.requires_dense), g0, p, alpha
        )

        # --- Apply SOC after line search (and clip to TR radius) ---
        s = alpha * p
        x_trial = x + s
        if self.cfg.use_soc and alpha < 1.0:
            dx_corr, _needs_rest = self.soc.compute_correction(model, x_trial, self.tr.delta)
            # Ensure ||αp + q|| <= Δ
            q = self.tr.clip_correction_to_radius(s, dx_corr)
            s = s + q
            x_trial = x + s

        # --- Actual reduction & feasibility at trial point ---
        step_norm = (
            float(self.tr._tr_norm(s)) if self.tr and hasattr(self.tr, "_tr_norm")
            else float(np.linalg.norm(s))
        )
        d_trial = model.eval_all(x_trial)
        f_new = float(d_trial["f"])
        act_red = f0 - f_new
        theta_new = model.constraint_violation(x_trial)

        # --- Convergence check at trial point ---
        new_kkt = model.kkt_residuals(x_trial, lam_new, nu_new)
        if self._is_kkt(new_kkt) or abs(f0 - f_new) < self.cfg.tol_obj_change:
            info = self._pack_info(step_norm, True, True, f_new, theta_new, new_kkt, ls_iters, alpha, 1.0)
            return x_trial, lam_new, nu_new, info

        # --- Trust-region update (uses predicted w/o SOC, actual with SOC) ---
        rho = act_red / max(pred_red, 1e-16)
        need_rest = self.tr.update(
            predicted_reduction=pred_red,
            actual_reduction=act_red,
            step_norm=step_norm,
            constraint_violation=theta_new,
            H=_to_dense_if_needed(H, self.requires_dense),
            p=s,
        )

        # --- Filter / acceptance decision ---
        accept = (rho >= self.cfg.tr_eta_lo and act_red > -1e-16)
        if self.flt:
            accept = accept and self.flt.is_acceptable(theta_new, f_new, trust_radius=self.tr.delta)
        else:
            if theta0 >= self.cfg.filter_theta_min:
                accept = accept and (theta_new <= theta0 * 1.05)

        if not accept or need_rest:
            # Try restoration if available
            if self.restoration:
                trR = self.tr.delta if self.tr else self.cfg.tr_delta0
                pr, meta = self.restoration.try_restore(model, x, trR)
                if meta.get("ok", False) and pr is not None and np.all(np.isfinite(pr)):
                    x2 = x + pr
                    d2 = model.eval_all(x2)
                    k2 = model.kkt_residuals(x2, lam, nu)
                    info = self._pack_info(
                        float(np.linalg.norm(pr)), False, False,
                        float(d2["f"]), model.constraint_violation(x2), k2, ls_iters, alpha, rho
                    )
                    if self.tr:
                        self.tr.delta = max(self.cfg.tr_delta_min, 0.5 * trR)
                    return x2, lam, nu, info
            info = self._pack_info(0.0, False, False, f0, theta0, kkt0, ls_iters, alpha, rho)
            return x, lam, nu, info

        # --- Accept step; optionally update Hessian approximation ---
        if self.cfg.hessian_mode in ["bfgs", "lbfgs", "hybrid"]:
            g_new = d_trial["g"]
            self.hess.update(s, g_new - g0)

        x_out = x_trial
        lam_out, nu_out = lam_new, nu_new
        info = self._pack_info(step_norm, True, False, f_new, theta_new, new_kkt, ls_iters, alpha, rho)
        return x_out, lam_out, nu_out, info


    def _is_kkt(self, kkt: Dict[str, float]) -> bool:
        """Check KKT conditions with relaxed complementarity for inactive constraints."""
        stat_tol = getattr(self.cfg, "tol_stat", 1e-4)
        feas_tol = getattr(self.cfg, "tol_feas", 1e-4)
        comp_tol = getattr(self.cfg, "tol_comp", 1e-4)
        # Relax complementarity if constraints are inactive (θ ≈ 0)
        comp_check = kkt["comp"] <= comp_tol or kkt["ineq"] <= 1e-10
        return (
            kkt["stat"] <= stat_tol
            and kkt["ineq"] <= feas_tol
            and kkt["eq"] <= feas_tol
            and comp_check
        )

    def _pack_info(
        self,
        step_norm: float,
        accepted: bool,
        converged: bool,
        f: float,
        theta: float,
        kkt: Dict,
        ls_iters: int,
        alpha: float,
        rho: float,
    ) -> Dict:
        """Pack step information with KKT diagnostics."""
        return {
            "step_norm": step_norm,
            "accepted": accepted,
            "converged": converged,
            "f": f,
            "theta": theta,
            "stat": kkt["stat"],
            "ineq": kkt["ineq"],
            "eq": kkt["eq"],
            "comp": kkt["comp"],
            "ls_iters": ls_iters,
            "alpha": alpha,
            "rho": rho,
            "tr_radius": self.tr.delta if self.tr else 0.0,
        }