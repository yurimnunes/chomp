# nlp_hybrid.py
# Hybrid NLP Solver: Adaptive Interior-Point (Mehrotra, slacks-only barrier) + TR-SQP
# - Inequalities: cI(x) + s = 0, s > 0 (log barrier ONLY on s)
# - Fraction-to-boundary (Mehrotra predictor-corrector) for IP
# - TR-SQP with persistent PIQP QP, optional L2-CG trust-region path, SOC corrector
# - Auto-switch: IP <-> SQP via θ/μ thresholds and stall detection
# - SQP logic refactored into SQPStepper class
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

# ---- reuse shared infra (no duplication; we assume sqp_aux exists) ----
from .blocks.aux import (
    HessianManager,
    Model,
    RestorationManager,
    SQPConfig,  # we extend via _ensure_cfg_fields
)
from .blocks.filter import *
from .blocks.linesearch import *
from .blocks.qp import *
from .blocks.reg import Regularizer, make_psd_advanced
from .blocks.soc import *
from .blocks.tr import *
from .dfo import *
from .dfo_model import *
from .ip import *


# =============================================================================
# SQP Stepper (new)
# =============================================================================
class SQPStepper:
    """
    TR-SQP stepper with optional Composite Step (Fletcher-style):
      p = n + t, where
        - n: normal step (reduces violation via damped LS under TR)
        - t: tangential step (improves Lagrangian in Null(J_eq), TR-box proxy)
    Returns (x_out, lam_out, nu_out, info_dict).
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
        self.cfg = cfg
        self.hess = hess
        self.tr = tr
        self.ls = ls
        self.qp = qp
        self.soc = soc
        self.regularizer = regularizer
        self.restoration = restoration

    def _estimate_condition_number(self, H, max_iter=10):
        """
        Estimate the condition number of a sparse or dense matrix H using power iteration.
        Returns a rough estimate of cond(H) = ||H||_2 / ||H^-1||_2.
        """
        if sp.issparse(H):
            H = H.tocsr()
            # Approximate largest eigenvalue via power iteration
            n = H.shape[0]
            v = np.random.randn(n)
            v = v / np.linalg.norm(v)
            for _ in range(max_iter):
                v = H @ v
                v = v / np.linalg.norm(v)
            lambda_max = float(np.abs(v.T @ H @ v))
            # Approximate smallest eigenvalue (inverse iteration)
            try:
                from scipy.sparse.linalg import spsolve

                v = np.random.randn(n)
                v = v / np.linalg.norm(v)
                for _ in range(max_iter):
                    v = spsolve(H, v)
                    v = v / np.linalg.norm(v)
                lambda_min = 1.0 / float(np.abs(v.T @ spsolve(H, v)))
            except:
                lambda_min = 1e-8  # Fallback for singular matrices
            return lambda_max / max(lambda_min, 1e-8)
        else:
            # Dense case
            eigvals = np.linalg.eigvalsh(H)
            lambda_max = np.max(np.abs(eigvals))
            lambda_min = np.min(np.abs(eigvals))
            return lambda_max / max(lambda_min, 1e-8)

    def _tr_solve_with_radius(
        self, H, g, JE, cE, JI, cI, Delta, tol=None, act_tol=None
    ):
        """
        Call TR.solve with a temporary radius Delta, then restore the original radius.
        If no TR manager exists, fall back to a conservative box-QP with ||p||_∞ ≤ Delta.
        Supports sparse H, JE, JI.
        """
        # Convert to sparse if dense, prefer CSR for efficiency
        H = sp.csr_matrix(H) if not sp.issparse(H) else H.tocsr()
        JE = sp.csr_matrix(JE) if JE is not None and not sp.issparse(JE) else JE
        JI = sp.csr_matrix(JI) if JI is not None and not sp.issparse(JI) else JI
        cE = np.asarray(cE) if cE is not None else None
        cI = np.asarray(cI) if cI is not None else None
        g = np.asarray(g)

        if self.tr is None:
            lb = -Delta * np.ones_like(g)
            ub = Delta * np.ones_like(g)
            # Convert to dense if solver doesn't support sparse
            H_dense = (
                H.toarray()
                if hasattr(self.qp, "requires_dense") and self.qp.requires_dense
                else H
            )
            JE_dense = (
                JE.toarray()
                if JE is not None
                and hasattr(self.qp, "requires_dense")
                and self.qp.requires_dense
                else JE
            )
            JI_dense = (
                JI.toarray()
                if JI is not None
                and hasattr(self.qp, "requires_dense")
                and self.qp.requires_dense
                else JI
            )
            p, _, _ = self.qp.solve(H_dense, g, JI_dense, cI, JE_dense, cE, lb, ub)
            return p

        old_radius = float(self.tr.radius)
        self.tr.radius = float(Delta)
        try:
            # Assume self.tr.solve supports sparse inputs; convert if needed
            H_dense = (
                H.toarray()
                if hasattr(self.tr, "requires_dense") and self.tr.requires_dense
                else H
            )
            JE_dense = (
                JE.toarray()
                if JE is not None
                and hasattr(self.tr, "requires_dense")
                and self.tr.requires_dense
                else JE
            )
            JI_dense = (
                JI.toarray()
                if JI is not None
                and hasattr(self.tr, "requires_dense")
                and self.tr.requires_dense
                else JI
            )
            p, _, _ = self.tr.solve(
                H_dense,
                g,
                JE=JE_dense,
                cE=cE,
                JI=JI_dense,
                cI=cI,
                tol=(
                    getattr(self.cfg, "tr_l2_sigma_tol", 1e-6) if tol is None else tol
                ),
                act_tol=(
                    getattr(self.cfg, "act_tol", 1e-6) if act_tol is None else act_tol
                ),
            )
        finally:
            self.tr.radius = old_radius
        return p

    def _normal_step(self, model, x, JE, cE, JI, cI, Delta):
        """
        TR-aware normal step (phase-1 LS model):
            Hn = A^T A + μI, gn = -(A^T r),
            n = argmin ½ n^T Hn n + gn^T n s.t. ||·||_TR ≤ Delta.
        Equalities and *near-active* inequalities enter via LS residuals.
        Supports sparse JE, JI, and A.
        """
        act_tol = getattr(self.cfg, "act_tol", 1e-6)
        w_ineq = getattr(self.cfg, "cs_ineq_weight", 1.0)
        mu_damp = getattr(self.cfg, "cs_damping", 1e-8)

        A_blocks, r_blocks = [], []
        # Equalities as LS targets
        if (JE is not None) and (cE is not None) and (JE.size > 0) and (cE.size > 0):
            A_blocks.append(sp.csr_matrix(JE) if not sp.issparse(JE) else JE.tocsr())
            r_blocks.append(np.asarray(-cE))
        # Near-active inequalities
        if (JI is not None) and (cI is not None) and (JI.size > 0) and (cI.size > 0):
            cI = np.asarray(cI)
            active = cI > -act_tol
            if np.any(active):
                JI_active = (
                    sp.csr_matrix(JI[active])
                    if not sp.issparse(JI)
                    else JI.tocsr()[active]
                )
                r_ia = -cI[active]
                w = float(np.sqrt(w_ineq))
                A_blocks.append(w * JI_active)
                r_blocks.append(w * r_ia)

        if not A_blocks:
            return np.zeros(model.n)

        # Stack sparse matrices and dense vectors
        A = sp.vstack(A_blocks, format="csr")
        r = np.concatenate(r_blocks)

        # LS-derived quadratic model
        Hn = A.T @ A  # Sparse matrix product
        Hn = Hn + mu_damp * sp.eye(A.shape[1], format="csr")  # Add damping
        gn = -(A.T @ r)  # Sparse-dense matrix-vector product

        # Solve true TR subproblem (no hard equalities/ineqs here)
        n, _, _ = self.qp.solve(
            Hn, gn, JE=None, cE=None, JI=None, cI=None, tr_radius=Delta
        )
        return n

    def _tangential_step(self, H, gL, JE, JI_active, rhs_active, Delta, n):
        """
        TR-aware tangential step t (full space):
            min ½ t^T H t + gL^T t
            s.t. JE t = 0 (tangent to equalities)
                JI_active t <= -(cI_active + JI_active n) (wrt current normal n)
                ||t||_TR ≤ Δ_rem, Δ_rem^2 = Δ^2 - ||n||_TR^2
        Notes:
        - We **use the passed Δ** (caller’s trust radius), not self.tr.radius.
        - Inequalities are passed to TR.solve as JI p + cI ≤ 0 with cI_t = (cI_act + JI_act n) = -rhs_active.
        - Supports sparse H, JE, JI_active.
        """
        # Convert to sparse if needed
        H = sp.csr_matrix(H) if not sp.issparse(H) else H.tocsr()
        JE = sp.csr_matrix(JE) if JE is not None and not sp.issparse(JE) else JE
        JI_active = (
            sp.csr_matrix(JI_active)
            if JI_active is not None and not sp.issparse(JI_active)
            else JI_active
        )

        # Remaining radius in TR norm if available; else Euclidean
        if (self.tr is not None) and hasattr(self.tr, "_tr_norm"):
            n_tr = float(self.tr._tr_norm(n))
            Delta_rem_sq = max(0.0, Delta * Delta - n_tr * n_tr)
        else:
            n_eu = float(np.linalg.norm(n))
            Delta_rem_sq = max(0.0, Delta * Delta - n_eu * n_eu)

        if Delta_rem_sq <= 1e-16:  # Numerical tolerance
            return np.zeros_like(n)

        Delta_rem = float(np.sqrt(Delta_rem_sq))

        # Active ineqs translated for TR.solve signature
        A_ineq_t = None
        cI_t = None
        if (
            (JI_active is not None)
            and (rhs_active is not None)
            and (JI_active.size > 0)
            and (rhs_active.size > 0)
        ):
            A_ineq_t = JI_active
            cI_t = -np.asarray(rhs_active)  # Dense vector

        # Equalities for tangency
        JE_eq = JE if (JE is not None and JE.size > 0) else None
        cE_eq = np.zeros(JE_eq.shape[0]) if JE_eq is not None else None

        # Solve TR subproblem for t with equalities + active ineqs
        t, _, _ = self.qp.solve(
            H,
            gL,
            A_eq=JE_eq,
            b_eq=cE_eq,
            A_ineq=A_ineq_t,
            b_ineq=cI_t,
            tr_radius=Delta_rem,
        )
        return t

    def _compute_xi(self, theta0: float, Delta: float) -> float:
        """
        Compute the relaxation parameter xi for Byrd-Omojokun.
        xi = min(theta0, kappa * Delta), where kappa is a tuning parameter.
        """
        kappa = getattr(self.cfg, "byrd_omojokun_kappa", 0.8)
        return min(theta0, kappa * Delta)

    def _byrd_omojokun_step(
        self, model, x, H, g0, JE, cE, JI, cI, Delta, lam_hint, nu_hint
    ):
        """
        Byrd-Omojokun composite step: Solve a single QP with relaxed normal-step constraint.
        min ½ p^T H p + g0^T p
        s.t. JE p + cE <= xi (relaxed equality constraint)
            JI p + cI <= 0 (inequality constraints)
            ||p||_TR <= Delta
        Supports sparse H, JE, JI.
        """
        act_tol = getattr(self.cfg, "act_tol", 1e-6)
        theta0 = model.constraint_violation(x)

        # Convert to sparse if needed
        H = sp.csr_matrix(H) if not sp.issparse(H) else H.tocsr()
        JE = sp.csr_matrix(JE) if JE is not None and not sp.issparse(JE) else JE
        JI = sp.csr_matrix(JI) if JI is not None and not sp.issparse(JI) else JI
        cE = np.asarray(cE) if cE is not None else None
        cI = np.asarray(cI) if cI is not None else None
        g0 = np.asarray(g0)

        # Compute relaxation parameter xi
        xi = self._compute_xi(theta0, Delta)

        # Prepare QP inputs
        H_input = (
            H.toarray()
            if hasattr(self.qp, "requires_dense") and self.qp.requires_dense
            else H
        )
        JE_input = (
            JE.toarray()
            if JE is not None
            and hasattr(self.qp, "requires_dense")
            and self.qp.requires_dense
            else JE
        )
        JI_input = (
            JI.toarray()
            if JI is not None
            and hasattr(self.qp, "requires_dense")
            and self.qp.requires_dense
            else JI
        )

        # If QP solver supports bound constraints on residuals, use them directly
        if (
            hasattr(self.qp, "supports_residual_bound")
            and self.qp.supports_residual_bound
        ):
            p, lam_new, nu_new = self.qp.solve(
                H_input,
                g0,
                JI=JI_input,
                cI=cI,
                JE=JE_input,
                cE=cE,
                lb=None,
                ub=None,
                residual_bound=xi,
                tr_radius=Delta,
            )
        else:
            # Approximate residual constraint with penalty term
            mu_penalty = getattr(self.cfg, "byrd_omojokun_penalty", 1e2)
            H_penalty = H + mu_penalty * (
                JE.T @ JE if JE is not None else sp.csr_matrix((model.n, model.n))
            )
            g_penalty = g0 + mu_penalty * (
                JE.T @ cE if JE is not None and cE is not None else np.zeros_like(g0)
            )
            H_penalty_input = (
                H_penalty.toarray()
                if hasattr(self.qp, "requires_dense") and self.qp.requires_dense
                else H_penalty
            )
            p, lam_new, nu_new = self.qp.solve(
                H_penalty_input,
                g_penalty,
                A_ineq=JI_input,
                b_ineq=cI,
                A_eq=None,
                B_eq=None,
                lb=None,
                ub=None,
                tr_radius=Delta,
            )

        # Check if QP solve was successful
        if p is None or not np.all(np.isfinite(p)):
            # Fallback to original composite step
            return self._composite_step(
                model, x, H, g0, JE, cE, JI, cI, Delta, lam_hint, nu_hint
            )

        return p, lam_new, nu_new

    def _composite_step(
        self, model, x, H, g0, JE, cE, JI, cI, Delta, lam_hint, nu_hint
    ):
        """
        Wrapper for composite step: Use Byrd-Omojokun if enabled, else original normal-tangential.
        """
        if getattr(self.cfg, "use_byrd_omojokun", False):
            return self._byrd_omojokun_step(
                model, x, H, g0, JE, cE, JI, cI, Delta, lam_hint, nu_hint
            )
        else:
            # Original normal + tangential step implementation
            act_tol = getattr(self.cfg, "act_tol", 1e-6)
            # [Rest of the original _composite_step implementation]
            n = self._normal_step(model, x, JE, cE, JI, cI, Delta)
            JI_act, rhs_active = None, None
            if (
                (JI is not None)
                and (cI is not None)
                and (JI.size > 0)
                and (cI.size > 0)
            ):
                cI = np.asarray(cI)
                ci_n = cI + (JI @ n) if sp.issparse(JI) else np.asarray(JI) @ n
                active = ci_n >= -act_tol
                if np.any(active):
                    JI_act = JI[active] if sp.issparse(JI) else np.asarray(JI)[active]
                    JI_act = (
                        sp.csr_matrix(JI_act) if not sp.issparse(JI_act) else JI_act
                    )
                    rhs_active = -(
                        cI[active] + (JI_act @ n if sp.issparse(JI_act) else JI_act @ n)
                    )
            gL = g0.copy()
            if (JI is not None) and (lam_hint is not None) and (lam_hint.size > 0):
                gL += (
                    JI.T @ lam_hint if sp.issparse(JI) else np.asarray(JI).T @ lam_hint
                )
            if (JE is not None) and (nu_hint is not None) and (nu_hint.size > 0):
                gL += JE.T @ nu_hint if sp.issparse(JE) else np.asarray(JE).T @ nu_hint
            t = self._tangential_step(H, gL, JE, JI_act, rhs_active, Delta, n)
            p = n + t
            H_dense = (
                H.toarray()
                if hasattr(self.qp, "requires_dense") and self.qp.requires_dense
                else H
            )
            JE_dense = (
                JE.toarray()
                if JE is not None
                and hasattr(self.qp, "requires_dense")
                and self.qp.requires_dense
                else JE
            )
            JI_dense = (
                JI.toarray()
                if JI is not None
                and hasattr(self.qp, "requires_dense")
                and self.qp.requires_dense
                else JI
            )
            _, lam_new, nu_new = self.qp.solve(
                H_dense, g0, JI_dense, cI, JE_dense, cE, None, None
            )
            return p, lam_new, nu_new

    # ----------------------------
    # Main step
    # ----------------------------
    def step(
        self, model: Model, x: np.ndarray, lam: np.ndarray, nu: np.ndarray, it: int
    ):
        ls_iters, alpha, rho = 0, 1.0, 0.0

        data = model.eval_all(x)
        f0, g0 = float(data["f"]), data["g"]
        theta0 = model.constraint_violation(x)
        kkt = model.kkt_residuals(x, lam, nu)
        # Hessian (regularized PSD)
        H = self.hess.get_hessian(model, x, lam, nu)
        H, _ = make_psd_advanced(
            H, self.regularizer, it, model_quality=rho, constraint_count=len(nu)
        )

        # >>> Adaptive strategy for composite vs. legacy mode
        tol_feas = getattr(self.cfg, "tol_feas", 1e-8)
        cond_threshold = getattr(self.cfg, "cond_threshold", 1e6)
        feas_opt_ratio_threshold = getattr(self.cfg, "feas_opt_ratio", 1e2)

        # Estimate Hessian condition number
        cond_H = self._estimate_condition_number(H)
        # Compute feasibility-to-optimality ratio
        feas_opt_ratio = theta0 / max(kkt["stat"], 1e-8)

        # Decide to use composite step if:
        # 1. Constraint violation is significant, OR
        # 2. Hessian is ill-conditioned, OR
        # 3. Feasibility dominates stationarity
        use_cs = (
            theta0 > tol_feas
            or cond_H > cond_threshold
            or feas_opt_ratio > feas_opt_ratio_threshold
        ) and (getattr(self.cfg, "use_composite_step", False))

        # >>> refresh ellipsoidal TR metric from current (regularized) H
        if (
            (self.tr is not None)
            and hasattr(self.tr, "set_metric_from_H")
            and getattr(self.tr, "norm_type", "2") == "ellip"
        ):
            self.tr.set_metric_from_H(H)

        # Trust-radius or surrogate Delta
        Delta = self.tr.radius if self.tr else getattr(self.cfg, "tr_delta0", 1.0)

        if use_cs:
            p, lam_new, nu_new = self._composite_step(
                model,
                x,
                H,
                g0,
                data["JE"],
                data["cE"],
                data["JI"],
                data["cI"],
                Delta,
                lam,
                nu,
            )
        else:
            # Legacy: TR-L2 path (if available) or box-QP
            if self.tr:
                p, lam_new, nu_new = self.qp.solve(
                    H,
                    g0,
                    data["JI"],
                    data["cI"],
                    data["JE"],
                    data["cE"],
                    tr_radius=self.tr.radius,
                )
                if (p is None) or (not np.all(np.isfinite(p))):
                    if self.tr:
                        self.tr.radius *= getattr(self.cfg, "tr_gamma_dec", 0.5)
                    kkt = model.kkt_residuals(x, lam, nu)
                    info = self._pack_info(
                        0.0, False, False, f0, theta0, kkt, 0, 0.0, 0.0
                    )
                    return x, lam, nu, info
            else:
                p, lam_new, nu_new = self.qp.solve(
                    H, g0, data["JI"], data["cI"], data["JE"], data["cE"], None, None
                )

        # SAFEGUARD: if composite step is degenerate, fall back to legacy compute step
        if getattr(self.cfg, "use_composite_step", False):
            # predicted reduction with Lagrangian at current x
            gL_tmp = g0.copy()
            if data["JI"] is not None and lam_new.size > 0:
                gL_tmp += data["JI"].T @ lam_new
            if data["JE"] is not None and nu_new.size > 0:
                gL_tmp += data["JE"].T @ nu_new

            p_norm = float(np.linalg.norm(p))
            if sp.issparse(H):
                Hp = H.dot(p)
            else:
                Hp = H @ p
            pred_red_cs = -(float(np.dot(gL_tmp, p)) + 0.5 * float(np.dot(p, Hp)))

            if (
                (p_norm <= 1e-10)
                or (pred_red_cs <= 1e-16)
                or (not np.isfinite(pred_red_cs))
            ):
                # --- fallback to TR/QP path -------------
                if self.tr:
                    p, lam_new, nu_new = self.qp.solve(
                        H,
                        g0,
                        data["JI"],
                        data["cI"],
                        data["JE"],
                        data["cE"],
                        tr_radius=self.tr.radius,
                    )
                    if (p is None) or (not np.all(np.isfinite(p))):
                        if self.tr:
                            self.tr.radius *= getattr(self.cfg, "tr_gamma_dec", 0.5)
                        kkt = model.kkt_residuals(x, lam, nu)
                        info = self._pack_info(
                            0.0, False, False, f0, theta0, kkt, 0, 0.0, 0.0
                        )
                        return x, lam, nu, info
                else:
                    p, lam_new, nu_new = self.qp.solve(
                        H,
                        g0,
                        data["JI"],
                        data["cI"],
                        data["JE"],
                        data["cE"],
                        None,
                        None,
                    )

        # Early KKT with new multipliers at current x
        kkt = model.kkt_residuals(x, lam_new, nu_new)
        if self._is_kkt(kkt):
            info = self._pack_info(0.0, True, True, f0, theta0, kkt, ls_iters, 1.0, 0.0)
            return x, lam_new, nu_new, info

        # Line search
        if self.ls is not None:
            alpha, ls_iters, _ = self.ls.search_sqp(model, x, p)
            if alpha <= 1e-12:
                # Feasibility restoration attempt (TR-bounded)
                if getattr(self, "restoration", None) is not None:
                    trR = (
                        self.tr.radius
                        if self.tr
                        else getattr(self.cfg, "tr_delta0", 1.0)
                    )
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
                            self.tr.radius = max(
                                getattr(self.cfg, "tr_delta_min", 1e-6), 0.5 * trR
                            )
                        return x2, lam, nu, info
                # No improvement / no restoration
                info = self._pack_info(
                    0.0, False, False, f0, theta0, kkt, ls_iters, 0.0, 0.0
                )
                return x, lam, nu, info

        # Trial point (+ optional SOC)
        s = alpha * p
        x_trial = x + s
        if getattr(self.cfg, "use_soc", False) and alpha < 1.0:
            Delta_for_soc = self.tr.radius if self.tr else float(np.linalg.norm(s))
            dx_corr, _needs_rest = self.soc.compute_correction(
                model, x_trial, Delta_for_soc
            )
            s += dx_corr
            x_trial = x + s

        # Predicted / actual reductions
        if self.tr is not None and hasattr(self.tr, "_tr_norm"):
            step_norm = float(self.tr._tr_norm(s))  # ellipsoidal or chosen TR norm
        else:
            step_norm = float(np.linalg.norm(s))

        f_new = float(model.eval_all(x_trial)["f"])
        gL = g0.copy()
        if data["JI"] is not None and lam_new.size > 0:
            gL += data["JI"].T @ lam_new
        if data["JE"] is not None and nu_new.size > 0:
            gL += data["JE"].T @ nu_new

        Hs = (_as_csc_symmetric(H) @ s) if sp.issparse(H) else H.dot(s)
        pred_red = -(float(np.dot(gL, s)) + 0.5 * float(np.dot(s, Hs)))
        act_red = f0 - f_new
        rho = act_red / max(pred_red, 1e-16)
        act_size = int((lam_new > 1e-8).sum()) if lam_new.size > 0 else 0
        theta_new = model.constraint_violation(x_trial)

        # TR update (and potential restoration request)
        need_rest = self.tr.update(pred_red, act_red, step_norm) if self.tr else False

        accept = (
            ((rho >= getattr(self.cfg, "tr_eta_lo", 0.1)) and (act_red > -1e-16))
            if self.tr
            else True
        )
        if theta0 >= getattr(self.cfg, "filter_theta_min", 1e-8):
            accept = accept and (theta_new <= theta0 * 1.05)

        if not accept or need_rest:
            # Feasibility restoration as a phase-1 step
            if getattr(self, "restoration", None) is not None:
                trR = self.tr.radius if self.tr else getattr(self.cfg, "tr_delta0", 1.0)
                pr, meta = self.restoration.try_restore(model, x, trR)
                if meta.get("ok", False) and pr is not None and np.all(np.isfinite(pr)):
                    x2 = x + pr
                    d2 = model.eval_all(x2)
                    k2 = model.kkt_residuals(x2, lam, nu)
                    info = self._pack_info(
                        float(np.linalg.norm(pr)),
                        False,  # restoration is not a “normal” accepted step
                        False,
                        float(d2["f"]),
                        model.constraint_violation(x2),
                        k2,
                        ls_iters,
                        (alpha if self.ls else 1.0),
                        rho,
                    )
                    if self.tr:
                        self.tr.radius = max(
                            getattr(self.cfg, "tr_delta_min", 1e-6), 0.5 * trR
                        )
                    return x2, lam, nu, info
            # fallback: keep current point
            info = self._pack_info(
                0.0,
                False,
                False,
                f0,
                theta0,
                kkt,
                ls_iters,
                (alpha if self.ls else 1.0),
                rho,
            )
            return x, lam, nu, info

        # Accepted
        if getattr(self.cfg, "hessian_mode", "exact") in ["bfgs", "lbfgs", "hybrid"]:
            g_new = model.eval_all(x_trial)["g"]
            self.hess.update(s, g_new - g0)

        x_out = x_trial
        lam_out, nu_out = lam_new, nu_new

        new_d = model.eval_all(x_out)
        new_kkt = model.kkt_residuals(x_out, lam_out, nu_out)
        info = self._pack_info(
            step_norm,
            True,
            False,
            float(new_d["f"]),
            model.constraint_violation(x_out),
            new_kkt,
            ls_iters,
            (alpha if self.ls else 1.0),
            rho,
        )

        return x_out, lam_out, nu_out, info

    # ----------------------------
    # Utilities
    # ----------------------------
    def _is_kkt(self, kkt: Dict[str, float]) -> bool:
        return (
            kkt["stat"] <= getattr(self.cfg, "tol_stat", 1e-8)
            and kkt["ineq"] <= getattr(self.cfg, "tol_feas", 1e-8)
            and kkt["eq"] <= getattr(self.cfg, "tol_feas", 1e-8)
            and kkt["comp"] <= getattr(self.cfg, "tol_comp", 1e-8)
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
            "tr_radius": (self.tr.radius if self.tr else 0.0),
        }


# ---- small helper: extend config with new fields if missing ----
def _ensure_cfg_fields(cfg: SQPConfig):
    def add(name, val):
        if not hasattr(cfg, name):
            setattr(cfg, name, val)


# =============================================================================
# Main Solver (Hybrid)
# =============================================================================
class NLPSolver:
    def __init__(
        self,
        f: Callable,
        c_ineq: List[Callable] = None,
        c_eq: List[Callable] = None,
        x0: np.ndarray = None,
        config: SQPConfig = None,
    ):
        self.cfg = config if config is not None else SQPConfig()
        self.x = np.asarray(x0, float)
        self.n = len(self.x)

        # ---- ensure defaults for auto/hybrid ----
        self._ensure_auto_defaults(self.cfg)

        # Model & shared managers
        self.model = Model(f, c_ineq, c_eq, self.n)
        self.hess = HessianManager(self.n, self.cfg)
        self.rest = RestorationManager(self.cfg)
        self.regularizer = Regularizer(self.cfg)

        # Acceptance managers
        self.filter = Filter(self.cfg) if self.cfg.use_filter else None
        self.funnel = Funnel(self.cfg) if self.cfg.use_funnel else None

        # Step infrastructure
        self.tr = TrustRegionManager(self.cfg) if self.cfg.use_trust_region else None
        self.ls = (
            LineSearcher(self.cfg, self.filter) if self.cfg.use_line_search else None
        )
        self.qp = QPSolver(self.cfg, self.regularizer)
        self.soc = SOCCorrector(self.cfg)

        self.rest = RestorationManager(self.cfg)

        # SQP stepper
        self.sqp_stepper = SQPStepper(
            self.cfg,
            self.hess,
            self.tr,
            self.ls,
            self.qp,
            self.soc,
            self.regularizer,
            self.rest,
        )

        mI = len(c_ineq) if c_ineq else 0
        mE = len(c_eq) if c_eq else 0

        # IP stepper + state
        self.ip_state = IPState.from_model(self.model, self.x, self.cfg)
        self.ip_stepper = InteriorPointStepper(
            self.cfg,
            self.hess,
            self.regularizer,
            tr=self.tr,
            flt=self.filter,
            funnel=self.funnel,
            ls=self.ls,
            soc=self.soc,
        )
        mI = len(c_ineq) if c_ineq else 0
        mE = len(c_eq) if c_eq else 0
        self.dfo_state = DFOExactState(self.n, mI, mE, self.cfg)
        self.dfo_stepper = DFOExactPenaltyStepper(
            self.cfg,
            regularizer = self.regularizer,
            n=self.n,
            mI=mI,
            mE=mE,
            tr=self.tr,
        )

        _ensure_cfg_fields(self.cfg)  # add missing fields if needed

        # Multipliers (outer view)
        self.lam = np.zeros(mI)
        self.nu = np.zeros(mE)

        # Initial mode
        self.mode = self.cfg.mode
        if self.mode not in ("ip", "sqp", "auto", "dfo"):
            self.mode = "auto"

        if self.mode == "auto":
            theta0 = self.model.constraint_violation(self.x)
            self.mode = (
                "ip"
                if theta0 > max(self.cfg.ip_switch_theta, 10 * self.cfg.tol_feas)
                else "sqp"
            )

        if self.filter:
            self.filter.add_if_acceptable(
                self.model.constraint_violation(self.x),
                float(self.model.eval_all(self.x)["f"]),
            )

        # Watchdog
        self.watchdog_patience = 5
        self.watchdog_counter = 0
        self.best_point = (self.x.copy(), float("inf"), float("inf"))

        # Auto-mode trackers
        self._prev_kkt = None
        self._prev_theta = None
        self._reject_streak = 0
        self._small_alpha_streak = 0
        self._no_progress_streak = 0
        self._last_switch_iter = -(10**9)  # ensure we can switch early if needed

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def solve(
        self, max_iter: int = 200, tol: float = 1e-8, verbose: bool = True
    ) -> Tuple[np.ndarray, List[Dict]]:
        hist: List[Dict] = []
        for k in range(max_iter):
            if self.mode == "ip":
                info = self._ip_step(k)
            elif self.mode == "sqp":
                info = self._sqp_step(k)
            elif self.mode == "dfo":
                x_out, lam_out, nu_out, info = self.dfo_stepper.step(
                    self.model, self.x, self.dfo_state, k
                )
                # Update streaks / stats
                self._update_streaks(info)
                
                if info["converged"]:
                    if verbose:
                        print(f"✓ Converged at iteration {k}")
                    self.x = x_out
                    self.lam, self.nu = lam_out, nu_out
                    hist.append(info)
                    break

                if info["accepted"]:
                    if self.cfg.use_watchdog:
                        self._watchdog_update(x_out)
                    self.x = x_out
                    self.lam, self.nu = lam_out, nu_out
            else:
                # should not happen; fallback to SQP
                info = self._sqp_step(k)

            # track iteration info
            hist.append(info)
            if verbose:
                self._print_iteration(k, info)

            # Global KKT terminate (works for any mode)
            kkt = self.model.kkt_residuals(self.x, self.lam, self.nu)
            if (
                kkt["stat"] <= self.cfg.tol_stat
                and kkt["ineq"] <= self.cfg.tol_feas
                and kkt["eq"] <= self.cfg.tol_feas
                and kkt["comp"] <= self.cfg.tol_comp
            ):
                if verbose:
                    print(f"✓ Converged at iteration {k}")
                break

            # Auto-mode switch decision
            if self.cfg.mode == "auto":
                switched = self._maybe_switch_mode(k, info, kkt)
                if switched and verbose:
                    # annotate the last info dict with the switch
                    hist[-1] = {**hist[-1], "switched_to": self.mode}

        return self.x, hist

    # -------------------------------------------------------------------------
    # IP iteration (delegates to InteriorPointStepper)
    # -------------------------------------------------------------------------
    def _ip_step(self, it: int) -> Dict:
        x_out, lam_out, nu_out, info = self.ip_stepper.step(
            self.model, self.x, self.lam, self.nu, it, ip_state=self.ip_state
        )
        # Update streaks / stats
        self._update_streaks(info)

        if info["accepted"]:
            self.x = x_out
            # Mirror IP multipliers to outer state for unified reporting / warm-start
            self.lam, self.nu = lam_out, nu_out

        return info

    # -------------------------------------------------------------------------
    # SQP iteration (delegates to SQPStepper)
    # -------------------------------------------------------------------------
    def _sqp_step(self, it: int) -> Dict:
        x_out, lam_out, nu_out, info = self.sqp_stepper.step(
            self.model, self.x, self.lam, self.nu, it
        )

        # Update streaks / stats
        self._update_streaks(info)

        if info["accepted"]:
            if self.cfg.use_watchdog:
                self._watchdog_update(x_out)
            self.x = x_out
            self.lam, self.nu = lam_out, nu_out

        return info

    # -------------------------------------------------------------------------
    # Auto-mode helpers
    # -------------------------------------------------------------------------
    def _ensure_auto_defaults(self, cfg: SQPConfig) -> None:
        """Fill in sensible defaults for auto-mode thresholds/hysteresis."""

        def add(name, val):
            if not hasattr(cfg, name):
                setattr(cfg, name, val)

        # initial mode decision
        add("ip_switch_theta", 1e-3)  # if θ0 > this → start in IP

        # IP → SQP switch: when feasibility is good and barrier is small
        add("auto_ip2sqp_theta_cut", 5e-5)  # θ below this means go SQP
        add("auto_ip2sqp_mu_cut", 1e-6)  # μ small → go SQP
        add("auto_ip_min_iters", 3)  # stay at least N iters before switching

        # SQP → IP switch: when infeasibility dominates / stalls
        add("auto_sqp2ip_theta_blowup", 1e-2)  # θ above → go IP
        add("auto_sqp2ip_stall_iters", 3)  # no progress for N iters → go IP
        add("auto_sqp2ip_reject_streak", 2)  # too many rejects → go IP
        add("auto_sqp2ip_small_alpha_streak", 3)  # step sizes too small → go IP
        add("auto_sqp_min_iters", 3)

        # Hysteresis and min gap between switches
        add("auto_hysteresis_factor", 2.0)  # make switch bands non-overlapping
        add("auto_min_iter_between_switches", 2)

        # Small step detection
        add("auto_small_alpha", 1e-6)

    def _maybe_switch_mode(self, it: int, info: Dict, kkt: Dict[str, float]) -> bool:
        """Decide whether to switch modes (only in cfg.mode == 'auto')."""
        if self.cfg.mode != "auto":
            return False
        if it - self._last_switch_iter < int(self.cfg.auto_min_iter_between_switches):
            return False  # too soon to switch again

        theta = float(info.get("theta", self.model.constraint_violation(self.x)))
        alpha = float(info.get("alpha", 1.0))
        mu = float(info.get("mu", 0.0))

        # compute "progress" on feasibility / stationarity if we have previous
        progressed = True
        if self._prev_kkt is not None:
            prev_theta = self._prev_theta
            prev_stat = self._prev_kkt["stat"]
            # consider progress if either stat or theta dropped by a factor
            progressed = (theta <= prev_theta / 1.1) or (kkt["stat"] <= prev_stat / 1.1)

        # decision branches
        switched = False
        if self.mode == "ip":
            # Only consider switching if we've stayed a few iters in IP
            if it - self._last_switch_iter >= int(self.cfg.auto_ip_min_iters):
                ip2sqp = (theta <= float(self.cfg.auto_ip2sqp_theta_cut)) and (
                    mu <= float(self.cfg.auto_ip2sqp_mu_cut)
                )
                # Also switch if KKT stationarity is already small and feasibility ok
                ip2sqp |= (kkt["stat"] <= 5.0 * self.cfg.tol_stat) and (
                    theta
                    <= self.cfg.auto_ip2sqp_theta_cut * self.cfg.auto_hysteresis_factor
                )
                if ip2sqp:
                    self._switch_to("sqp", it)
                    switched = True

        elif self.mode == "sqp":
            if it - self._last_switch_iter >= int(self.cfg.auto_sqp_min_iters):
                # blow-up in infeasibility OR repeated stalls/rejects → go IP
                sqp2ip = (
                    theta >= float(self.cfg.auto_sqp2ip_theta_blowup)
                    or self._no_progress_streak >= int(self.cfg.auto_sqp2ip_stall_iters)
                    or self._reject_streak >= int(self.cfg.auto_sqp2ip_reject_streak)
                    or self._small_alpha_streak
                    >= int(self.cfg.auto_sqp2ip_small_alpha_streak)
                )
                if sqp2ip:
                    self._switch_to("ip", it)
                    switched = True

        # update previous for next call
        self._prev_kkt = dict(kkt)
        self._prev_theta = theta
        return switched

    def _switch_to(self, target: str, it: int) -> None:
        """Perform warm-started switch between modes."""
        if target not in ("ip", "sqp") or target == self.mode:
            return
        self._last_switch_iter = it

        if target == "ip":
            # Re-initialize IP state at current x (warm-start multipliers from outer)
            self.ip_state = IPState.from_model(self.model, self.x, self.cfg)
            # If sizes match, reuse current multipliers to warm start the barrier
            if self.lam.size == self.ip_state.lam.size:
                self.ip_state.lam = np.maximum(1e-6, np.copy(self.lam))
            if self.nu.size == self.ip_state.nu.size:
                self.ip_state.nu = np.copy(self.nu)
            # heuristic μ warm-start from complementarity
            if self.ip_state.mI > 0:
                s = self.ip_state.s
                lam = self.ip_state.lam
                mu0 = max(self.cfg.ip_mu_init, min(1e-1, (s @ lam) / max(1, s.size)))
                self.ip_state.mu = mu0
        else:  # target == "sqp"
            # nothing special; keep current (x, lam, nu) as warm start
            pass

        # Reset streaks on switch
        self._reject_streak = 0
        self._small_alpha_streak = 0
        self._no_progress_streak = 0

        self.mode = target

    def _update_streaks(self, info: Dict) -> None:
        """Update per-iteration streak counters used in auto-switch logic."""
        accepted = bool(info.get("accepted", False))
        alpha = float(info.get("alpha", 1.0))
        theta = float(info.get("theta", self.model.constraint_violation(self.x)))
        # update reject / alpha streaks
        self._reject_streak = 0 if accepted else self._reject_streak + 1
        self._small_alpha_streak = (
            (self._small_alpha_streak + 1)
            if alpha <= float(self.cfg.auto_small_alpha)
            else 0
        )
        # update "no progress" streak based on θ compared to last θ
        if self._prev_theta is None:
            self._no_progress_streak = 0
        else:
            improved = (
                theta <= self._prev_theta * 0.99 or theta <= self._prev_theta - 1e-14
            )
            self._no_progress_streak = 0 if improved else (self._no_progress_streak + 1)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _is_kkt(self, kkt: Dict[str, float]) -> bool:
        return (
            kkt["stat"] <= self.cfg.tol_stat
            and kkt["ineq"] <= self.cfg.tol_feas
            and kkt["eq"] <= self.cfg.tol_feas
            and kkt["comp"] <= self.cfg.tol_comp
        )

    def _print_iteration(self, k: int, info: Dict):
        status = "A" if info["accepted"] else "R"
        if info["converged"]:
            status += "*"
        mode = info.get("mode", self.mode)
        mu_s = f" μ={info['mu']:.2e}" if "mu" in info else ""
        sw = f" ->{info['switched_to']}" if "switched_to" in info else ""
        print(
            f"[{k:3d}] {status} [{mode}]{sw} step={info['step_norm']:.2e} "
            f"f={info['f']:.6e} θ={info['theta']:.2e} α={info['alpha']:.2e} "
            f"Δ={info['tr_radius']:.2e}{mu_s}"
        )

    def _watchdog_update(self, x_cand: Optional[np.ndarray] = None) -> None:
        x_eval = self.x if x_cand is None else np.asarray(x_cand, float)
        d = self.model.eval_all(x_eval)
        f_new = float(d["f"])
        th_new = self.model.constraint_violation(x_eval)
        x_best, f_best, th_best = self.best_point

        rel = 1e-3
        th_small = self.cfg.filter_theta_min
        improved_theta = th_new < max(th_best * (1.0 - rel), th_best - 1e-14)
        improved_f_small_th = (
            th_new <= max(th_small, 1e-12)
            and th_best <= max(th_small, 1e-12)
            and f_new < max(f_best * (1.0 - rel), f_best - 1e-14)
        )
        if (th_best == float("inf")) or improved_theta or improved_f_small_th:
            self.best_point = (x_eval.copy(), f_new, th_new)
            self.watchdog_counter = 0
            if self.filter:
                self.filter.add_if_acceptable(th_new, f_new)
            return

        self.watchdog_counter += 1
        if self.watchdog_counter < self.watchdog_patience:
            return

        self.watchdog_counter = 0
        x_b, f_b, th_b = self.best_point
        if np.all(np.isfinite(x_b)):
            self.x = x_b.copy()
            # cheap multiplier refresh
            try:
                data_rb = self.model.eval_all(self.x)
                H_rb, _ = make_psd_advanced(
                    self.hess.get_hessian(self.model, self.x, self.lam, self.nu),
                    self.regularizer,
                )
                _, lam_rb, nu_rb = self.qp.solve(
                    H_rb,
                    data_rb["g"],
                    data_rb["JI"],
                    data_rb["cI"],
                    data_rb["JE"],
                    data_rb["cE"],
                    lb=None,
                    ub=None,
                    warm_from_last=False,
                )
                self.lam, self.nu = lam_rb, nu_rb
            except Exception:
                pass
            if self.tr:
                self.tr.radius = max(1e-12, self.cfg.tr_gamma_dec * self.tr.radius)
            if self.filter:
                self.filter.reset()
                self.filter.add_if_acceptable(th_b, f_b)
