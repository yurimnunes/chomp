from __future__ import annotations

from typing import Dict, Optional, Tuple

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
from .blocks.reg import DOptStabilizer, Regularizer, make_psd_advanced
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
        self.dopt = DOptStabilizer(self.cfg)

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
            "use_soc": False,
            "filter_theta_min": 1e-8,
            "hessian_mode": "exact",
            # --- dOPT (Gill–Saunders style) stabilization ---
            "stabilize_dopt": True,  # toggle
            "dopt_sigma_E": 1e-2,  # base equalities shift gain
            "dopt_sigma_I": 1e-2,  # base inequalities shift gain
            "dopt_mu_target": 1e-6,  # target complementarity for near-active I rows
            "dopt_active_tol": 5e-3,  # consider I-rows with cI in [-tol, +inf) as near-active
            "dopt_max_shift": 1e-1,  # cap ||r||_∞ to avoid over-shifting
            "dopt_scaling": "colnorm",  # {"colnorm", "ruiz"} (simple column-norm by default)
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
        - Predicted reduction reflects the ACTUAL step s = α p (+ q if SOC).
        - Actual reduction uses f(x + s). TR update uses (pred_red, act_red, ||s||, θ_new).
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

        # --- Optionally refresh TR metric (ellipsoidal) ---
        if self.tr and getattr(self.tr, "norm_type", "2") == "ellip":
            self.tr.update_metric(H)

        Hd = _to_dense_if_needed(H, self.requires_dense)  # reuse

        JI = _to_dense_if_needed(data0["JI"], self.requires_dense)
        JE = _to_dense_if_needed(data0["JE"], self.requires_dense)
        cI = data0["cI"]
        cE = data0["cE"]

        # --- (optional) Gill–Saunders style stabilization (shifted linearization) ---
        if getattr(self.cfg, "stabilize_dopt", True):
            rE, rI, _meta = self.dopt.compute_shifts(JE, JI, cE, cI, lam, nu)
            cE_shift = (cE + rE) if (cE is not None and rE is not None) else cE
            cI_shift = (cI + rI) if (cI is not None and rI is not None) else cI
        else:
            cE_shift, cI_shift = cE, cI

        # --- Solve TR subproblem ---
        p, tr_info, lam_new, nu_new = self.tr.solve(
            Hd,
            g0,
            JI,
            cI_shift,
            JE,
            cE_shift,
        )

        # Guard: TR failed or returned non-finite step
        if p is None or not np.all(np.isfinite(p)):
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
                        float(np.linalg.norm(pr)),
                        True,
                        False,
                        float(d2["f"]),
                        model.constraint_violation(x2),
                        k2,
                        0,
                        0.0,
                        0.0,
                    )
                    if self.tr:
                        self.tr.delta = max(self.cfg.tr_delta_min, 0.5 * trR)
                    return x2, lam, nu, info
            return x, lam, nu, info

        # --- Line search on p ---
        alpha, ls_iters, _ = (1.0, 0, None)
        if self.ls:
            alpha, ls_iters, _ = self.ls.search_sqp(model, x, p)
            amax = self.tr._alpha_max_box(x, p, model.lb, model.ub)  # safe cap
            alpha = min(alpha, amax)
            if alpha <= 1e-12:
                info = self._pack_info(
                    0.0, False, False, f0, theta0, kkt0, ls_iters, 0.0, 0.0
                )
                if self.restoration:
                    trR = self.tr.delta if self.tr else self.cfg.tr_delta0
                    pr, meta = self.restoration.try_restore(model, x, trR)
                    if (
                        meta.get("ok", False)
                        and pr is not None
                        and np.all(np.isfinite(pr))
                    ):
                        x2 = x + pr
                        d2 = model.eval_all(x2)
                        k2 = model.kkt_residuals(x2, lam, nu)
                        info = self._pack_info(
                            float(np.linalg.norm(pr)),
                            True,
                            False,
                            float(d2["f"]),
                            model.constraint_violation(x2),
                            k2,
                            ls_iters,
                            0.0,
                            0.0,
                        )
                        if self.tr:
                            self.tr.delta = max(self.cfg.tr_delta_min, 0.5 * trR)
                        return x2, lam, nu, info
                return x, lam, nu, info

        # --- Predicted reduction for s_lin = α p (NO SOC yet) ---
        try:
            pred_red_lin = self.tr.model_reduction_alpha(Hd, g0, p, alpha)
        except Exception:
            s_lin = alpha * p
            pred_red_lin = -(g0 @ s_lin + 0.5 * (s_lin @ (Hd @ s_lin)))

        # --- Apply SOC after line search (and clip to TR radius) ---
        s = alpha * p
        x_trial = x + s

        # FIX: compute violation at the line-searched trial point (before SOC)
        theta_lin = model.constraint_violation(
            x_trial
        )  # predicted feasibility after line search

        pred_red = pred_red_lin  # will update if SOC is used
        if self.cfg.use_soc and alpha < 1.0:
            self.soc.cfg.tr_norm_type = getattr(
                self.tr, "norm_type", "2"
            )  # "2" or "ellip"

            dx_corr, need_rest = self.soc.compute_correction(
                model,
                x_trial,
                Delta=self.tr.delta,
                s=alpha * p,
                H=H,
                mu=0.0,
                theta0=theta0,
                f0=f0,
                pred_df=pred_red,
                pred_dtheta=(
                    theta0 - theta_lin
                ),  # FIX: was theta_new (undefined); use pre-SOC theta
            )
            q = self.tr.clip_correction_to_radius(s, dx_corr)  # ensure ‖s+q‖ ≤ Δ
            amax_q = self.tr._alpha_max_box(x + s, q, model.lb, model.ub)
            q *= amax_q

            # exact quadratic increment for predicted reduction with s+q
            Hs = Hd @ s
            Hq = Hd @ q
            pred_red = pred_red_lin - ((g0 + Hs) @ q + 0.5 * (q @ Hq))

            s = s + q
            x_trial = x + s
            # Optional: you could recompute theta_lin here if you want to log "pre-acceptance" feasibility

        # --- Actual reduction & feasibility at trial point ---
        step_norm = (
            float(self.tr._tr_norm(s))
            if self.tr and hasattr(self.tr, "_tr_norm")
            else float(np.linalg.norm(s))
        )
        d_trial = model.eval_all(x_trial)
        f_new = float(d_trial["f"])
        act_red = f0 - f_new
        theta_new = model.constraint_violation(x_trial)  # now defined for later use

        if self.tr:
            self.tr.set_f_current(f_new)  # New: provide f_new for history

        # --- Convergence check at trial point ---
        new_kkt = model.kkt_residuals(x_trial, lam_new, nu_new)
        c_norm = max(
            np.linalg.norm(cE) if cE is not None else 0.0,
            np.linalg.norm(cI) if cI is not None else 0.0,
        )
        if (
            self._is_kkt(new_kkt, np.linalg.norm(g0), c_norm)
            or abs(f0 - f_new) < self.cfg.tol_obj_change
        ):
            info = self._pack_info(
                step_norm, True, True, f_new, theta_new, new_kkt, ls_iters, alpha, 1.0
            )
            return x_trial, lam_new, nu_new, info

        # --- Trust-region update (uses predicted for ACTUAL step, actual reduction) ---
        rho = act_red / max(pred_red, 1e-16)
        need_rest = self.tr.update(
            predicted_reduction=pred_red,
            actual_reduction=act_red,
            step_norm=step_norm,
            constraint_violation=theta_new,
            H=Hd,
            p=s,
        )

        # --- Filter / acceptance decision ---
        accept = rho >= self.cfg.tr_eta_lo and act_red > -1e-16
        if self.flt:
            accept = accept and self.flt.is_acceptable(
                theta_new, f_new, trust_radius=self.tr.delta
            )
        else:
            if theta0 >= self.cfg.filter_theta_min:
                accept = accept and (theta_new <= theta0 * 1.05)

        if not accept or need_rest:
            if self.restoration:
                trR = self.tr.delta if self.tr else self.cfg.tr_delta0
                pr, meta = self.restoration.try_restore(model, x, trR)
                if meta.get("ok", False) and pr is not None and np.all(np.isfinite(pr)):
                    x2 = x + pr
                    d2 = model.eval_all(x2)
                    k2 = model.kkt_residuals(x2, lam, nu)
                    info = self._pack_info(
                        float(np.linalg.norm(pr)),
                        False,
                        False,
                        float(d2["f"]),
                        model.constraint_violation(x2),
                        k2,
                        ls_iters,
                        alpha,
                        rho,
                    )
                    if self.tr:
                        self.tr.delta = max(self.cfg.tr_delta_min, 0.5 * trR)
                    return x2, lam, nu, info
            info = self._pack_info(
                0.0, False, False, f0, theta0, kkt0, ls_iters, alpha, rho
            )
            return x, lam, nu, info

        # --- Accept step; optionally update Hessian approximation ---
        if self.cfg.hessian_mode in ["bfgs", "lbfgs", "hybrid"]:
            g_new = d_trial["g"]
            self.hess.update(s, g_new - g0)

        x_out = x_trial
        lam_out, nu_out = lam_new, nu_new
        info = self._pack_info(
            step_norm, True, False, f_new, theta_new, new_kkt, ls_iters, alpha, rho
        )
        return x_out, lam_out, nu_out, info

    def _is_kkt(self, kkt, g_norm, c_norm):
        scale = max(1.0, g_norm + c_norm)
        return (
            kkt["stat"] / scale <= self.cfg.tol_stat
            and kkt["ineq"] / scale <= self.cfg.tol_feas
            and kkt["eq"] / scale <= self.cfg.tol_feas
            and (kkt["comp"] / scale <= self.cfg.tol_comp or kkt["ineq"] <= 1e-10)
        )

    # def _is_kkt(self, kkt: Dict[str, float]) -> bool:
    #     """Check KKT conditions with relaxed complementarity for inactive constraints."""
    #     stat_tol = getattr(self.cfg, "tol_stat", 1e-4)
    #     feas_tol = getattr(self.cfg, "tol_feas", 1e-4)
    #     comp_tol = getattr(self.cfg, "tol_comp", 1e-4)
    #     # Relax complementarity if constraints are inactive (θ ≈ 0)
    #     comp_check = kkt["comp"] <= comp_tol or kkt["ineq"] <= 1e-10
    #     return (
    #         kkt["stat"] <= stat_tol
    #         and kkt["ineq"] <= feas_tol
    #         and kkt["eq"] <= feas_tol
    #         and comp_check
    #     )

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
