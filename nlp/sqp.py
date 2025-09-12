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


def _norm_mat(mat, n_expected: int) -> Tuple[np.ndarray | sp.spmatrix, int, int]:
    """Return (M, m, n) with M 2-D, ensuring n == n_expected when possible.
    Accepts dense/sparse and 1-D vectors (row). Transposes (n,1) to (1,n)."""
    if mat is None:
        return None, 0, n_expected
    if sp.issparse(mat):
        M = mat.tocsr()
        m, n = M.shape
        # Handle accidental column vector (n,1) representing a single row
        if n != n_expected and m == n_expected and n == 1:
            M = M.T.tocsr()
            m, n = M.shape
        return M, m, n
    # dense
    M = np.asarray(mat, dtype=float)
    if M.ndim == 1:
        # treat as a single row
        M = M.reshape(1, -1)
    m, n = M.shape
    if n != n_expected and m == n_expected and n == 1:
        # mistaken column vector; make it a row
        M = M.T
        m, n = M.shape
    return M, m, n

def _norm_vec(vec, m_expected: int) -> np.ndarray:
    """Return 1-D vector of length m_expected."""
    if vec is None:
        return None
    v = np.asarray(vec, dtype=float).reshape(-1)
    if v.size != m_expected:
        raise ValueError(f"RHS length mismatch: expected {m_expected}, got {v.size}")
    return v


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

    def _composite_step(
        self,
        model: "Model",
        x: np.ndarray,
        H: sp.spmatrix | np.ndarray,
        g0: np.ndarray,
        JE: Optional[sp.spmatrix | np.ndarray],
        cE: Optional[np.ndarray],
        JI: Optional[sp.spmatrix | np.ndarray],
        cI: Optional[np.ndarray],
        Delta: float,
        lam_hint: Optional[np.ndarray] = None,
        nu_hint: Optional[np.ndarray] = None,
        tol: Optional[float] = None,
        act_tol: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        One-pass composite step using self.tr.solve.
        Returns (p, lam_new, nu_new).
        """
        if self.tr is None:
            raise RuntimeError("Trust-region solver (self.tr) is required for composite step.")
        print("Using composite step (TR-SQP)...")
        # --- config ---
        cfg = getattr(self, "cfg", None)
        act_tol = act_tol if act_tol is not None else getattr(cfg, "act_tol", 1e-6)
        byrd_omojokun = bool(getattr(cfg, "use_byrd_omojokun", False))
        tol = tol if tol is not None else getattr(cfg, "tr_l2_sigma_tol", 1e-6)
        w_ineq = float(getattr(cfg, "cs_ineq_weight", 1.0))
        mu_damp = float(getattr(cfg, "cs_damping", 1e-8))
        mu_penalty = float(getattr(cfg, "byrd_omojokun_penalty", 1e2))
        kappa = float(getattr(cfg, "byrd_omojokun_kappa", 0.8))

        # --- normalize inputs ---
        H = _to_sparse(H)
        JE = _to_sparse(JE) if JE is not None else None
        JI = _to_sparse(JI) if JI is not None else None
        g0 = np.asarray(g0)
        cE = np.asarray(cE) if cE is not None else None
        cI = np.asarray(cI) if cI is not None else None

        def _as_backend(mat):
            return _to_dense_if_needed(mat, getattr(self.tr, "requires_dense", False))

        # --- temporarily set trust-region radius ---
        old_radius = float(self.tr.delta)
        self.tr.delta = float(Delta)
        try:
            # 1) Byrd–Omojokun fast path
            if byrd_omojokun:
                theta0 = float(model.constraint_violation(x))
                xi = min(theta0, kappa * float(Delta))

                # residual-bound path
                if getattr(self.tr, "supports_residual_bound", False):
                    p, _, lam_new, nu_new = self.tr.solve(
                        _as_backend(H), g0,
                        A_ineq=_as_backend(JI) if JI is not None else None, b_ineq=cI,
                        A_eq=_as_backend(JE) if JE is not None else None, b_eq=cE,
                        #tol=tol, act_tol=act_tol,
                        #residual_bound=xi,
                    )
                    if p is not None and np.all(np.isfinite(p)):
                        return p, lam_new, nu_new

                # penalty fallback (relax equalities)
                H_pen = H + (mu_penalty * (JE.T @ JE) if JE is not None else sp.csr_matrix(H.shape))
                g_pen = g0 + (mu_penalty * (JE.T @ cE) if (JE is not None and cE is not None) else 0.0)
                p, _, lam_new, nu_new = self.tr.solve(
                    _as_backend(H_pen), g_pen,
                    A_ineq=_as_backend(JI) if JI is not None else None, b_ineq=cI,
                    #tol=tol, act_tol=act_tol,
                )
                if p is not None and np.all(np.isfinite(p)):
                    return p, lam_new, nu_new
                # else: fall through to N+T

            # 2) Normal step (reduce violation on active subset)
            A_blocks, r_blocks = [], []
            if JE is not None and cE is not None and JE.size and cE.size:
                A_blocks.append(JE)
                r_blocks.append(-cE)

            JI_act_mask = None
            if JI is not None and cI is not None and JI.size and cI.size:
                cI_arr = np.asarray(cI)
                JI_arr = JI if sp.issparse(JI) else np.asarray(JI)
                JI_act_mask = cI_arr > -act_tol
                if np.any(JI_act_mask):
                    JI_active = _to_sparse(JI_arr[JI_act_mask])
                    w = float(np.sqrt(max(w_ineq, 1e-16)))
                    A_blocks.append(w * JI_active)
                    r_blocks.append(w * (-cI_arr[JI_act_mask]))

            if A_blocks:
                A = sp.vstack(A_blocks, format="csr")
                r = np.concatenate(r_blocks)
                Hn = A.T @ A + mu_damp * sp.eye(A.shape[1], format="csr")
                gn = -(A.T @ r)
                n, _, _, _ = self.tr.solve(_as_backend(Hn), _to_dense_if_needed(gn, getattr(self.tr, "requires_dense", False)))
                                        #tol=tol, act_tol=act_tol)
            else:
                n = np.zeros_like(g0)

            # 3) Tangential step (optimize Lagrangian in remaining radius with active constraints)
            # Determine active inequalities AFTER taking n
            JI_act = None
            rhs_active = None
            if JI is not None and cI is not None and JI.size and cI.size:
                JI_arr = JI if sp.issparse(JI) else np.asarray(JI)
                cI_arr = np.asarray(cI)
                ci_n = cI_arr + (JI_arr @ n)
                act2 = ci_n >= -act_tol
                if np.any(act2):
                    JI_act = _to_sparse(JI_arr[act2])
                    rhs_active = -(cI_arr[act2] + ((JI_act @ n) if sp.issparse(JI_act) else (JI_act @ n)))
                    rhs_active = np.asarray(rhs_active).reshape(-1)  # ensure 1D

            # Lagrangian gradient with hints
            gL = g0.copy()
            if JI is not None and lam_hint is not None and lam_hint.size:
                gL += (JI.T @ lam_hint) if sp.issparse(JI) else (np.asarray(JI).T @ lam_hint)
            if JE is not None and nu_hint is not None and nu_hint.size:
                gL += (JE.T @ nu_hint) if sp.issparse(JE) else (np.asarray(JE).T @ nu_hint)

            # remaining TR for tangential move
            n_norm = float(self.tr._tr_norm(n)) if hasattr(self.tr, "_tr_norm") else float(np.linalg.norm(n))
            Delta_rem = float(np.sqrt(max(0.0, Delta * Delta - n_norm * n_norm)))

            if Delta_rem <= 1e-16:
                t = np.zeros_like(n)
            else:
                # dimension n from gradient (or use H.shape[0] if you prefer)
                n_dim = int(np.asarray(gL).reshape(-1).size)

                # Equalities for tangential step: JE * t = 0
                if JE is not None and getattr(JE, "size", 0):
                    A_eq_raw = _as_backend(JE)
                    A_eq, me, ne = _norm_mat(A_eq_raw, n_dim)
                    if ne != n_dim:
                        raise ValueError(f"A_eq must have {n_dim} columns; got {ne}")
                    b_eq = np.zeros(me, dtype=float)
                else:
                    A_eq, b_eq = None, None

                # Inequalities: only active set after n (JI_act * t <= rhs_active)
                if JI_act is not None and getattr(JI_act, "size", 0):
                    A_ineq_raw = _as_backend(JI_act)
                    A_ineq, mi, ni = _norm_mat(A_ineq_raw, n_dim)
                    if ni != n_dim:
                        raise ValueError(f"A_ineq must have {n_dim} columns; got {ni}")
                    b_ineq = _norm_vec(rhs_active, mi)
                else:
                    A_ineq, b_ineq = None, None

                # Solve tangential QP
                t, _, _, _ = self.tr.solve(
                    _as_backend(H),
                    _to_dense_if_needed(gL, getattr(self.tr, "requires_dense", False)),
                    A_ineq=A_ineq,
                    b_ineq=b_ineq,
                    A_eq=A_eq,
                    b_eq=b_eq,
                    # tol=tol, act_tol=act_tol,
                )

            p = n + t

            # 4) Refresh multipliers for reporting (full KKT system, no extra constraints)
            _, lam_new, nu_new = self.qp.solve(
                _as_backend(H), g0,
                 _to_dense_if_needed(JI, self.requires_dense),
                cI,
                 _to_dense_if_needed(JE, self.requires_dense),
                cE,
                #tol=tol, act_tol=act_tol,
            )

            if not np.all(np.isfinite(p)):
                p = np.zeros_like(g0)

            return p, lam_new, nu_new

        finally:
            self.tr.delta = old_radius


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
            p, lam_new, nu_new = self._composite_step(
                model, x, H, g0, data["JE"], data["cE"], data["JI"], data["cI"], Delta, lam, nu
            )
        else:
            p, _, lam_new, nu_new = self.tr.solve(
                _to_dense_if_needed(H, self.requires_dense),
                g0,
                _to_dense_if_needed(data["JI"], self.requires_dense),
                data["cI"],
                _to_dense_if_needed(data["JE"], self.requires_dense),
                data["cE"],
                #tr_radius=Delta,
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
                p, _, lam_new, nu_new = self.tr.solve(
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
            print("Applying SOC correction...")
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
        need_rest = self.tr.update(pred_red, act_red, step_norm) if self.tr else False
        accept = (rho >= self.cfg.tr_eta_lo and act_red > -1e-16) if self.tr else True
        if self.flt:
            accept = accept and self.flt.is_acceptable(theta_new, f_new, trust_radius=Delta)
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