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


# Utility functions
def _to_sparse(matrix, format="csr"):
    """Convert matrix to sparse CSR format if not already sparse."""
    return sp.csr_matrix(matrix, copy=False) if not sp.issparse(matrix) else matrix.tocsr()

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

    def _estimate_condition_number(self, H: sp.spmatrix | np.ndarray, max_iter: int = 10) -> float:
        """Estimate condition number of H using power iteration."""
        if sp.issparse(H):
            H = H.tocsr()
            n = H.shape[0]
            v = np.random.randn(n)
            v /= np.linalg.norm(v)
            for _ in range(max_iter):
                v = H @ v
                v /= np.linalg.norm(v)
            lambda_max = float(np.abs(v.T @ (H @ v)))
            try:
                from scipy.sparse.linalg import spsolve
                v = np.random.randn(n)
                v /= np.linalg.norm(v)
                for _ in range(max_iter):
                    v = spsolve(H, v)
                    v /= np.linalg.norm(v)
                lambda_min = 1.0 / float(np.abs(v.T @ spsolve(H, v)))
            except:
                lambda_min = 1e-8
        else:
            eigvals = np.linalg.eigvalsh(H)
            lambda_max = np.max(np.abs(eigvals))
            lambda_min = np.min(np.abs(eigvals))
        return lambda_max / max(lambda_min, 1e-8)

    def _tr_solve_with_radius(
        self,
        H: sp.spmatrix | np.ndarray,
        g: np.ndarray,
        JE: Optional[sp.spmatrix | np.ndarray],
        cE: Optional[np.ndarray],
        JI: Optional[sp.spmatrix | np.ndarray],
        cI: Optional[np.ndarray],
        Delta: float,
        tol: Optional[float] = None,
        act_tol: Optional[float] = None,
    ) -> np.ndarray:
        """Solve trust-region subproblem with radius Delta."""
        H = _to_sparse(H)
        JE = _to_sparse(JE) if JE is not None else None
        JI = _to_sparse(JI) if JI is not None else None
        cE = np.asarray(cE) if cE is not None else None
        cI = np.asarray(cI) if cI is not None else None
        g = np.asarray(g)

        if self.tr is None:
            lb = -Delta * np.ones_like(g)
            ub = Delta * np.ones_like(g)
            p, _, _ = self.qp.solve(
                _to_dense_if_needed(H, self.requires_dense),
                g,
                _to_dense_if_needed(JI, self.requires_dense),
                cI,
                _to_dense_if_needed(JE, self.requires_dense),
                cE,
                lb,
                ub,
            )
            return p

        old_radius = float(self.tr.delta)
        self.tr.delta = float(Delta)
        try:
            p, _, _ = self.tr.solve(
                _to_dense_if_needed(H, getattr(self.tr, "requires_dense", False)),
                g,
                JE=_to_dense_if_needed(JE, getattr(self.tr, "requires_dense", False)),
                cE=cE,
                JI=_to_dense_if_needed(JI, getattr(self.tr, "requires_dense", False)),
                cI=cI,
                tol=tol or getattr(self.cfg, "tr_l2_sigma_tol", 1e-6),
                act_tol=act_tol or getattr(self.cfg, "act_tol", 1e-6),
            )
        finally:
            self.tr.delta = old_radius
        return p

    def _normal_step(
        self,
        model: Model,
        x: np.ndarray,
        JE: Optional[sp.spmatrix | np.ndarray],
        cE: Optional[np.ndarray],
        JI: Optional[sp.spmatrix | np.ndarray],
        cI: Optional[np.ndarray],
        Delta: float,
    ) -> np.ndarray:
        """Compute normal step to reduce constraint violation."""
        act_tol = getattr(self.cfg, "act_tol", 1e-6)
        w_ineq = getattr(self.cfg, "cs_ineq_weight", 1.0)
        mu_damp = getattr(self.cfg, "cs_damping", 1e-8)

        A_blocks, r_blocks = [], []
        if JE is not None and cE is not None and JE.size > 0 and cE.size > 0:
            A_blocks.append(_to_sparse(JE))
            r_blocks.append(np.asarray(-cE))
        if JI is not None and cI is not None and JI.size > 0 and cI.size > 0:
            cI = np.asarray(cI)
            active = cI > -act_tol
            if np.any(active):
                JI_active = _to_sparse(JI[active] if sp.issparse(JI) else np.asarray(JI)[active])
                r_ia = -cI[active]
                w = float(np.sqrt(w_ineq))
                A_blocks.append(w * JI_active)
                r_blocks.append(w * r_ia)

        if not A_blocks:
            return np.zeros(model.n)

        A = sp.vstack(A_blocks, format="csr")
        r = np.concatenate(r_blocks)
        Hn = A.T @ A + mu_damp * sp.eye(A.shape[1], format="csr")
        gn = -(A.T @ r)
        n, _, _ = self.qp.solve(Hn, gn, tr_radius=Delta)
        return n

    def _tangential_step(
        self,
        H: sp.spmatrix | np.ndarray,
        gL: np.ndarray,
        JE: Optional[sp.spmatrix | np.ndarray],
        JI_active: Optional[sp.spmatrix | np.ndarray],
        rhs_active: Optional[np.ndarray],
        Delta: float,
        n: np.ndarray,
    ) -> np.ndarray:
        """Compute tangential step to optimize Lagrangian."""
        H = _to_sparse(H)
        JE = _to_sparse(JE) if JE is not None else None
        JI_active = _to_sparse(JI_active) if JI_active is not None else None
        n_tr = float(self.tr._tr_norm(n)) if self.tr and hasattr(self.tr, "_tr_norm") else float(np.linalg.norm(n))
        Delta_rem = float(np.sqrt(max(0.0, Delta * Delta - n_tr * n_tr)))
        if Delta_rem <= 1e-16:
            return np.zeros_like(n)

        cI_t = -np.asarray(rhs_active) if rhs_active is not None and rhs_active.size > 0 else None
        JE_eq = JE if JE is not None and JE.size > 0 else None
        cE_eq = np.zeros(JE_eq.shape[0]) if JE_eq is not None else None

        t, _, _ = self.qp.solve(
            H,
            gL,
            A_eq=JE_eq,
            b_eq=cE_eq,
            A_ineq=JI_active,
            b_ineq=cI_t,
            tr_radius=Delta_rem,
        )
        return t

    def _compute_xi(self, theta0: float, Delta: float) -> float:
        """Compute relaxation parameter xi for Byrd-Omojokun."""
        kappa = getattr(self.cfg, "byrd_omojokun_kappa", 0.8)
        return min(theta0, kappa * Delta)

    def _byrd_omojokun_step(
        self,
        model: Model,
        x: np.ndarray,
        H: sp.spmatrix | np.ndarray,
        g0: np.ndarray,
        JE: Optional[sp.spmatrix | np.ndarray],
        cE: Optional[np.ndarray],
        JI: Optional[sp.spmatrix | np.ndarray],
        cI: Optional[np.ndarray],
        Delta: float,
        lam_hint: np.ndarray,
        nu_hint: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Byrd-Omojokun composite step with relaxed equality constraints."""
        theta0 = model.constraint_violation(x)
        xi = self._compute_xi(theta0, Delta)
        H = _to_sparse(H)
        JE = _to_sparse(JE) if JE is not None else None
        JI = _to_sparse(JI) if JI is not None else None
        cE = np.asarray(cE) if cE is not None else None
        cI = np.asarray(cI) if cI is not None else None
        g0 = np.asarray(g0)

        H_input = _to_dense_if_needed(H, self.requires_dense)
        JE_input = _to_dense_if_needed(JE, self.requires_dense)
        JI_input = _to_dense_if_needed(JI, self.requires_dense)

        if getattr(self.qp, "supports_residual_bound", False):
            p, lam_new, nu_new = self.qp.solve(
                H_input, g0, JI=JI_input, cI=cI, JE=JE_input, cE=cE, residual_bound=xi, tr_radius=Delta
            )
        else:
            mu_penalty = getattr(self.cfg, "byrd_omojokun_penalty", 1e2)
            H_penalty = H + mu_penalty * (JE.T @ JE if JE is not None else sp.csr_matrix((model.n, model.n)))
            g_penalty = g0 + mu_penalty * (JE.T @ cE if JE is not None and cE is not None else np.zeros_like(g0))
            p, lam_new, nu_new = self.qp.solve(
                _to_dense_if_needed(H_penalty, self.requires_dense),
                g_penalty,
                A_ineq=JI_input,
                b_ineq=cI,
                tr_radius=Delta,
            )

        if p is None or not np.all(np.isfinite(p)):
            return self._composite_step(model, x, H, g0, JE, cE, JI, cI, Delta, lam_hint, nu_hint)
        return p, lam_new, nu_new

    def _composite_step(
        self,
        model: Model,
        x: np.ndarray,
        H: sp.spmatrix | np.ndarray,
        g0: np.ndarray,
        JE: Optional[sp.spmatrix | np.ndarray],
        cE: Optional[np.ndarray],
        JI: Optional[sp.spmatrix | np.ndarray],
        cI: Optional[np.ndarray],
        Delta: float,
        lam_hint: np.ndarray,
        nu_hint: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Wrapper for composite step: Uses Byrd-Omojokun or normal + tangential step."""
        if getattr(self.cfg, "use_byrd_omojokun", False):
            return self._byrd_omojokun_step(model, x, H, g0, JE, cE, JI, cI, Delta, lam_hint, nu_hint)

        act_tol = getattr(self.cfg, "act_tol", 1e-6)
        n = self._normal_step(model, x, JE, cE, JI, cI, Delta)
        JI_act, rhs_active = None, None
        if JI is not None and cI is not None and JI.size > 0 and cI.size > 0:
            cI = np.asarray(cI)
            ci_n = cI + (JI @ n if sp.issparse(JI) else np.asarray(JI) @ n)
            active = ci_n >= -act_tol
            if np.any(active):
                JI_act = _to_sparse(JI[active] if sp.issparse(JI) else np.asarray(JI)[active])
                rhs_active = -(cI[active] + (JI_act @ n if sp.issparse(JI_act) else JI_act @ n))

        gL = g0.copy()
        if JI is not None and lam_hint is not None and lam_hint.size > 0:
            gL += JI.T @ lam_hint if sp.issparse(JI) else np.asarray(JI).T @ lam_hint
        if JE is not None and nu_hint is not None and nu_hint.size > 0:
            gL += JE.T @ nu_hint if sp.issparse(JE) else np.asarray(JE).T @ nu_hint

        t = self._tangential_step(H, gL, JE, JI_act, rhs_active, Delta, n)
        p = n + t
        H_input = _to_dense_if_needed(H, self.requires_dense)
        JE_input = _to_dense_if_needed(JE, self.requires_dense)
        JI_input = _to_dense_if_needed(JI, self.requires_dense)
        _, lam_new, nu_new = self.qp.solve(H_input, g0, JI_input, cI, JE_input, cE)
        return p, lam_new, nu_new

    def step(
        self,
        model: Model,
        x: np.ndarray,
        lam: np.ndarray,
        nu: np.ndarray,
        it: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Main SQP step: Compute step, update multipliers, and apply TR/line-search."""
        ls_iters, alpha, rho = 0, 1.0, 0.0
        data = model.eval_all(x)
        f0, g0 = float(data["f"]), data["g"]
        theta0 = model.constraint_violation(x)
        kkt = model.kkt_residuals(x, lam, nu)

        # Compute regularized Hessian
        H = self.hess.get_hessian(model, x, lam, nu)
        H, _ = make_psd_advanced(H, self.regularizer, it, model_quality=rho, constraint_count=len(nu))

        # Adaptive composite step decision
        cond_H = self._estimate_condition_number(H)
        feas_opt_ratio = theta0 / max(kkt["stat"], 1e-8)
        use_cs = (
            theta0 > self.cfg.tol_feas
            or cond_H > self.cfg.cond_threshold
            or feas_opt_ratio > self.cfg.feas_opt_ratio
        ) and self.cfg.use_composite_step

        # Update trust-region metric
        Delta = self.tr.delta if self.tr else self.cfg.tr_delta0
        if self.tr and hasattr(self.tr, "set_metric_from_H") and getattr(self.tr, "norm_type", "2") == "ellip":
            self.tr.set_metric_from_H(H)

        # Compute step
        if use_cs:
            p, _ = self._composite_step(
                model, x, H, g0, data["JE"], data["cE"], data["JI"], data["cI"], Delta, lam, nu
            )
        else:
            p, lam_new, nu_new = self.qp.solve(
                _to_dense_if_needed(H, self.requires_dense),
                g0,
                _to_dense_if_needed(data["JI"], self.requires_dense),
                data["cI"],
                _to_dense_if_needed(data["JE"], self.requires_dense),
                data["cE"],
                tr_radius=Delta,
            )
            if p is None or not np.all(np.isfinite(p)):
                if self.tr:
                    self.tr.delta *= self.cfg.tr_gamma_dec
                info = self._pack_info(0.0, False, False, f0, theta0, kkt, 0, 0.0, 0.0)
                # try restoration if available
                if self.restoration:
                    print("QP failed, trying restoration...")
                    trR = self.tr.delta if self.tr else self.cfg.tr_delta0
                    pr, meta = self.restoration.try_restore(model, x, trR)
                    if meta.get("ok", False) and pr is not None and np.all(np.isfinite(pr)):
                        x2 = x + pr
                        d2 = model.eval_all(x2)
                        k2 = model.kkt_residuals(x2, lam, nu)
                        info = self._pack_info(
                            float(np.linalg.norm(pr)), True, False, float(d2["f"]), model.constraint_violation(x2), k2, 0, 0.0, 0.0
                        )
                        if self.tr:
                            self.tr.delta = max(self.cfg.tr_delta_min, 0.5 * trR)
                        return x2, lam, nu, info
                return x, lam, nu, info

        # Check for degenerate composite step
        if self.cfg.use_composite_step:
            gL_tmp = g0.copy()
            if data["JI"] is not None and lam_new.size > 0:
                gL_tmp += data["JI"].T @ lam_new
            if data["JE"] is not None and nu_new.size > 0:
                gL_tmp += data["JE"].T @ nu_new
            p_norm = float(np.linalg.norm(p))
            Hp = H.dot(p) if sp.issparse(H) else H @ p
            pred_red_cs = -(float(np.dot(gL_tmp, p)) + 0.5 * float(np.dot(p, Hp)))
            if p_norm <= 1e-10 or pred_red_cs <= 1e-16 or not np.isfinite(pred_red_cs):
                p, _, _, lam_new, nu_new = self.tr.solve(
                    _to_dense_if_needed(H, self.requires_dense),
                    g0,
                    _to_dense_if_needed(data["JI"], self.requires_dense),
                    data["cI"],
                    _to_dense_if_needed(data["JE"], self.requires_dense),
                    data["cE"],
                    #tr_radius=self.tr.delta if self.tr else None,
                )
                if p is None or not np.all(np.isfinite(p)):
                    if self.tr:
                        self.tr.delta *= self.cfg.tr_gamma_dec
                    info = self._pack_info(0.0, False, False, f0, theta0, kkt, 0, 0.0, 0.0)
                    return x, lam, nu, info

        # Line search
        if self.ls:
            alpha, ls_iters, _ = self.ls.search_sqp(model, x, p)
            if alpha <= 1e-12:
                if self.restoration:
                    trR = self.tr.delta if self.tr else self.cfg.tr_delta0
                    pr, meta = self.restoration.try_restore(model, x, trR)
                    if meta.get("ok", False) and pr is not None and np.all(np.isfinite(pr)):
                        x2 = x + pr
                        d2 = model.eval_all(x2)
                        k2 = model.kkt_residuals(x2, lam, nu)
                        info = self._pack_info(
                            float(np.linalg.norm(pr)), True, False, float(d2["f"]), model.constraint_violation(x2), k2, ls_iters, 0.0, 0.0
                        )
                        if self.tr:
                            self.tr.delta = max(self.cfg.tr_delta_min, 0.5 * trR)
                        return x2, lam, nu, info
                info = self._pack_info(0.0, False, False, f0, theta0, kkt, ls_iters, 0.0, 0.0)
                return x, lam, nu, info

        # Trial point and SOC
        s = alpha * p
        x_trial = x + s
        if self.cfg.use_soc and alpha < 1.0:
            Delta_for_soc = self.tr.delta if self.tr else float(np.linalg.norm(s))
            dx_corr, _needs_rest = self.soc.compute_correction(model, x_trial, Delta_for_soc)
            s += dx_corr
            x_trial = x + s

        # Compute reductions
        step_norm = float(self.tr._tr_norm(s)) if self.tr and hasattr(self.tr, "_tr_norm") else float(np.linalg.norm(s))
        f_new = float(model.eval_all(x_trial)["f"])
        gL = g0.copy()
        if data["JI"] is not None and lam_new.size > 0:
            gL += data["JI"].T @ lam_new
        if data["JE"] is not None and nu_new.size > 0:
            gL += data["JE"].T @ nu_new
        Hs = H.dot(s) if sp.issparse(H) else H @ s
        pred_red = -(float(np.dot(gL, s)) + 0.5 * float(np.dot(s, Hs)))
        act_red = f0 - f_new
        rho = act_red / max(pred_red, 1e-16)
        theta_new = model.constraint_violation(x_trial)

        # Check convergence at trial point
        new_d = model.eval_all(x_trial)
        new_kkt = model.kkt_residuals(x_trial, lam_new, nu_new)
        if self._is_kkt(new_kkt) or abs(f0 - f_new) < self.cfg.tol_obj_change:
            info = self._pack_info(
                step_norm, True, True, float(new_d["f"]), theta_new, new_kkt, ls_iters, alpha, rho
            )
            return x_trial, lam_new, nu_new, info

        # Trust-region update
        need_rest = self.tr.update(pred_red, act_red, step_norm, theta0=theta0) if self.tr else False
        accept = (rho >= self.cfg.tr_eta_lo and act_red > -1e-16) if self.tr else True
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
                        float(np.linalg.norm(pr)), False, False, float(d2["f"]), model.constraint_violation(x2), k2, ls_iters, alpha, rho
                    )
                    if self.tr:
                        self.tr.delta = max(self.cfg.tr_delta_min, 0.5 * trR)
                    return x2, lam, nu, info
            info = self._pack_info(0.0, False, False, f0, theta0, kkt, ls_iters, alpha, rho)
            return x, lam, nu, info

        # Update Hessian and accept step
        if self.cfg.hessian_mode in ["bfgs", "lbfgs", "hybrid"]:
            g_new = model.eval_all(x_trial)["g"]
            self.hess.update(s, g_new - g0)

        x_out = x_trial
        lam_out, nu_out = lam_new, nu_new
        info = self._pack_info(
            step_norm, True, False, float(new_d["f"]), theta_new, new_kkt, ls_iters, alpha, rho
        )
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