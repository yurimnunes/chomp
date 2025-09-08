from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

try:
    import numba as nb
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False


class TrustRegionManager:
    """
    Trust-Region manager for both DFO (derivative-free) and SQP workflows.

    This class maintains a radius Δ and a metric defining ||·||_TR, decides how
    to update Δ from model-vs-actual reduction, and solves the trust-region
    subproblem (TRSP) with optional (in)equality constraints.

    Capabilities
    ------------
    • Multiple TR norms:
        - "ellip":   ||p||_TR = ||L p||₂ with L built from H (curvature-aware)
        - "2":       Euclidean norm
        - "inf":     Infinity norm
        - "scaled":  Diagonal scaling from diag(L) if available; else Euclidean
    • Metric building:
        - Robust Cholesky with adaptive regularization for dense H
        - Diagonal fallback for sparse/ill-conditioned H
    • Updates:
        - Acceptance/rejection via ρ = (actual)/(predicted) reduction
        - DFO vs SQP parameterizations via `configure_for_algorithm`
        - Curvature- and boundary-aware growth/shrink heuristics
    • Solvers:
        - Conjugate-Gradient (CG) TR method (dense/sparse/operator)
        - Optional numba-accelerated dense kernel
        - Equality constraints via null-space reduction
        - Inequalities via simple active-set + projection fallback
    • Diagnostics:
        - Rolling statistics on ρ, boundary usage, curvature
        - CG counters and metric condition estimate

    Configuration
    -------------
    All knobs live in `TrustRegionManager._Defaults`. Pass a lightweight object
    with matching attributes to override (e.g., a SimpleNamespace). Unknown
    fields are ignored; unspecified ones use defaults.

    Key fields (see `_Defaults` for the full list):
        tr_delta0, tr_delta_min, tr_delta_max
        tr_eta_lo, tr_eta_hi
        tr_gamma_dec, tr_gamma_inc
        tr_boundary_tol, tr_boundary_trigger, tr_max_rejections
        tr_norm_type ∈ {"ellip", "2", "inf", "scaled"}

        cg_tol, cg_max_iter, cg_restart_every, cg_neg_curv_tol

        act_tol, max_ws_add, ws_removal_tol

        chol_jitter, diag_floor, metric_regularization, metric_condition_threshold

        rho_smooth, pr_threshold, tr_grow_requires_boundary
        dfo_mode, curvature_aware_growth, feasibility_weighted_updates
    """

    class _Defaults:
        # ---------------------------- Core trust region ----------------------------
        tr_delta0 = 1.0
        tr_delta_min = 1e-12
        tr_delta_max = 1e6
        tr_eta_lo = 0.1
        tr_eta_hi = 0.9
        tr_gamma_dec = 0.5
        tr_gamma_inc = 2.0
        tr_boundary_tol = 1e-8
        tr_max_rejections = 3
        tr_norm_type = "2"  # {"ellip", "2", "inf", "scaled"}

        # ----------------------------- CG parameters -------------------------------
        cg_tol = 1e-8
        cg_max_iter = 200
        cg_restart_every = 50
        cg_neg_curv_tol = 1e-14

        # --------------------------- Constraint handling ---------------------------
        act_tol = 1e-8
        max_ws_add = 10
        ws_removal_tol = 1e-10

        # ---------------------- Ellipsoidal metric computation ---------------------
        chol_jitter = 1e-10
        diag_floor = 1e-12
        metric_regularization = 1e-8
        metric_condition_threshold = 1e12

        # ----------------------------- Adaptive behavior ---------------------------
        rho_smooth = 0.3
        pr_threshold = 1e-16
        tr_boundary_trigger = 0.8
        tr_grow_requires_boundary = True

        # ---------------------- Algorithm-specific adaptations ---------------------
        dfo_mode = False                  # enable more conservative DFO settings
        curvature_aware_growth = True
        feasibility_weighted_updates = True

    def __init__(self, cfg=None):
        """
        Initialize the manager.

        Parameters
        ----------
        cfg : object, optional
            Any object whose attributes mirror `_Defaults`. Only matching keys are
            read; others are ignored. If None, defaults are used.
        """
        self.cfg = self._merge_cfg(cfg)
        self.radius = float(self.cfg.tr_delta0)
        self.rej = 0
        self.norm_type = str(self.cfg.tr_norm_type)
        self._n_last = None

        # Ellipsoidal metric: L so that ||p||_TR = ||L p||₂
        self.L = None
        self.Linv = None
        self._metric_condition = None
        self._last_metric_H_hash = None

        # Adaptation state
        self._rho_smoothed = None
        self._recent_ratios = []
        self._recent_boundaries = []
        self._recent_curvatures = []
        self._max_history = 10

        # Performance counters
        self._cg_stats = {
            "total_iterations": 0,
            "neg_curv_exits": 0,
            "boundary_hits": 0,
        }

    @staticmethod
    def _merge_cfg(cfg):
        """Return a defaults instance with user overrides applied (by attribute name)."""
        d = TrustRegionManager._Defaults()
        if cfg is None:
            return d
        for k in d.__dict__.keys():
            if hasattr(cfg, k):
                setattr(d, k, getattr(cfg, k))
        return d

    # ========================= Ellipsoidal Metric Management =========================

    def set_metric_from_H(self, H: Union[np.ndarray, sp.spmatrix], force_rebuild: bool = False):
        """
        Build an ellipsoidal metric L such that ||L p||₂ ≤ Δ defines the TR.

        Dense H → attempt robust Cholesky on H_sym + λI with growing λ.
        Sparse/ill-conditioned/failed Cholesky → fall back to diagonal scaling.

        Parameters
        ----------
        H : ndarray or spmatrix
            Symmetric (or nearly) Hessian approximation. If None or empty, reset metric.
        force_rebuild : bool
            If True, rebuild even if H hash matches the previous one.
        """
        if H is None:
            self._reset_metric()
            return

        n = H.shape[0]
        if n == 0:
            self._reset_metric()
            return

        # Cheap change detection to avoid rebuilds
        H_hash = hash(H.tobytes()) if hasattr(H, "tobytes") else id(H)
        if (not force_rebuild) and H_hash == self._last_metric_H_hash and self.L is not None:
            return
        self._last_metric_H_hash = H_hash

        if sp.issparse(H):
            self._build_diagonal_metric(H)
            return

        if isinstance(H, np.ndarray):
            ok = self._build_cholesky_metric(H)
            if not ok:
                self._build_diagonal_metric(H)
        else:
            self._build_diagonal_metric(H)

    def _build_cholesky_metric(self, H: np.ndarray) -> bool:
        """
        Attempt Cholesky-based metric with adaptive regularization.

        Returns
        -------
        bool
            True if a numerically healthy L and Linv were built, else False.
        """
        n = H.shape[0]
        H_sym = 0.5 * (H + H.T)

        # Gershgorin-based condition estimate
        try:
            diag = np.diag(H_sym)
            off = np.sum(np.abs(H_sym), axis=1) - np.abs(diag)
            lam_min = np.min(diag - off)
            lam_max = np.max(diag + off)
            cond_est = lam_max / lam_min if (lam_max > 0 and lam_min > 0) else 1e16
        except Exception:
            cond_est = 1e16

        base = self.cfg.chol_jitter
        reg = min(1e-4, cond_est * base) if cond_est > self.cfg.metric_condition_threshold else base

        I = np.eye(n)
        for _ in range(8):
            try:
                M = H_sym + reg * I
                L = la.cholesky(M, lower=True, overwrite_a=False, check_finite=False)
                if np.all(np.isfinite(L)) and np.min(np.diag(L)) > 1e-14:
                    self.L = L
                    try:
                        self.Linv = la.solve_triangular(L, I, lower=True, check_finite=False)
                        self._metric_condition = cond_est
                        return True
                    except Exception:
                        pass
            except la.LinAlgError:
                pass
            reg *= 10.0
        return False

    def _build_diagonal_metric(self, H: Union[np.ndarray, sp.spmatrix]):
        """
        Diagonal fallback metric for sparse/ill-conditioned H.

        Uses a robust floor on diag(H) with mean-abs regularization for positivity.
        """
        diag = H.diagonal() if sp.issparse(H) else np.diag(H)
        floor = self.cfg.diag_floor
        diag_reg = np.where(diag > floor, diag, np.maximum(floor, np.mean(np.abs(diag)) + floor))
        diag_reg = np.maximum(diag_reg, floor)

        sqrt_d = np.sqrt(diag_reg)
        self.L = np.diag(sqrt_d)
        self.Linv = np.diag(1.0 / sqrt_d)
        self._metric_condition = float(np.max(sqrt_d) / max(np.min(sqrt_d), 1e-32))

    def _reset_metric(self):
        """Reset metric to 'none' (||p||_TR falls back to Euclidean)."""
        self.L = None
        self.Linv = None
        self._metric_condition = None
        self._last_metric_H_hash = None

    def ellip_norm(self, p: np.ndarray) -> float:
        """Return ||L p||₂ if L exists; otherwise ||p||₂."""
        if self.L is None:
            return np.linalg.norm(p)
        try:
            return np.linalg.norm(self.L @ p)
        except Exception:
            return np.linalg.norm(p)

    def _tr_norm(self, p: np.ndarray) -> float:
        """Compute ||p||_TR under the selected norm type."""
        if self.norm_type == "ellip":
            return self.ellip_norm(p)
        elif self.norm_type == "2":
            return np.linalg.norm(p)
        elif self.norm_type == "inf":
            return np.linalg.norm(p, ord=np.inf)
        elif self.norm_type == "scaled":
            if self.L is not None:
                try:
                    return np.linalg.norm(np.diag(self.L) * p)
                except Exception:
                    pass
            return np.linalg.norm(p)
        return np.linalg.norm(p)

    # ========================= Adaptive Trust Region Updates =========================

    def update(
        self,
        pred_red: float,
        act_red: float,
        step_norm: float,
        theta0: Optional[float] = None,
        kkt: Optional[Dict] = None,
        act_sz: Optional[int] = None,
        H: Optional[np.ndarray] = None,
        step: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Update Δ based on ratio ρ = act_red / pred_red and heuristics.

        This implements the standard accept/expand vs reject/shrink decision, with
        smoothing, boundary usage, curvature cues, and DFO/SQP mode tweaks.

        Parameters
        ----------
        pred_red : float
            Predicted reduction from the model (≥ 0 for expected improvement).
        act_red : float
            Actual reduction measured on the true objective (≥ 0 if improved).
        step_norm : float
            ||p||_TR of the attempted step.
        theta0 : dict/float, optional
            Unused placeholder for future feasibility-weighted updates.
        kkt : dict, optional
            Optional KKT diagnostics (unused, reserved for future).
        act_sz : int, optional
            Size of active set (helps temper growth when many constraints bind).
        H : ndarray, optional
            Hessian used to analyze curvature along `step`.
        step : ndarray, optional
            Step vector used for curvature analysis.

        Returns
        -------
        bool
            True if repeated rejections suggest a structural issue (caller may
            wish to rebuild models/metric); False otherwise.
        """
        cfg = self.cfg

        rho = self._compute_robust_ratio(pred_red, act_red)
        self._update_performance_history(rho, step_norm, H, step)
        eta_lo, eta_hi, gamma_dec, gamma_inc = self._get_adaptive_parameters()
        used_boundary = self._analyze_boundary_usage(step_norm)
        curvature_info = (self._analyze_curvature(H, step) if (H is not None and step is not None) else {})

        if rho < eta_lo:
            # Reject → shrink
            self.rej += 1
            shrink = self._compute_shrink_factor(rho, curvature_info)
            self.radius = max(cfg.tr_delta_min, shrink * self.radius)
            if self.rej >= cfg.tr_max_rejections:
                return True
        else:
            # Accept
            self.rej = 0
            if rho >= eta_hi and (used_boundary or not cfg.tr_grow_requires_boundary):
                grow = self._compute_growth_factor(rho, curvature_info, act_sz)
                self.radius = min(cfg.tr_delta_max, grow * self.radius)

        self.radius = float(np.clip(self.radius, cfg.tr_delta_min, cfg.tr_delta_max))
        return False

    def _compute_robust_ratio(self, pred_red: float, act_red: float) -> float:
        """Return a clamped, robust ρ handling tiny/NaN/inf cases."""
        if not (np.isfinite(pred_red) and np.isfinite(act_red)):
            return -10.0
        if abs(pred_red) < self.cfg.pr_threshold:
            if act_red > 0:
                return 10.0
            elif abs(act_red) < self.cfg.pr_threshold:
                return 1.0
            else:
                return -10.0
        rho = act_red / pred_red
        if not np.isfinite(rho):
            return -10.0 if act_red < 0 else 10.0
        return float(np.clip(rho, -1e6, 1e6))

    def _update_performance_history(self, rho: float, step_norm: float, H: Optional[np.ndarray], step: Optional[np.ndarray]):
        """Update rolling histories (ρ, boundary usage, curvature) and smoothed ρ."""
        self._recent_ratios.append(rho)
        self._recent_boundaries.append(step_norm / max(self.radius, 1e-32))

        if H is not None and step is not None:
            try:
                curvature = step @ (H @ step) / max(np.dot(step, step), 1e-16)
                self._recent_curvatures.append(curvature)
            except Exception:
                pass

        for hist in (self._recent_ratios, self._recent_boundaries, self._recent_curvatures):
            while len(hist) > self._max_history:
                hist.pop(0)

        a = self.cfg.rho_smooth
        self._rho_smoothed = rho if self._rho_smoothed is None else (a * rho + (1 - a) * self._rho_smoothed)

    def _get_adaptive_parameters(self) -> Tuple[float, float, float, float]:
        """Return (η_lo, η_hi, γ_dec, γ_inc) tuned for DFO vs SQP."""
        if self.cfg.dfo_mode:
            eta_lo = max(0.01, self.cfg.tr_eta_lo * 0.5)
            eta_hi = min(0.95, self.cfg.tr_eta_hi * 1.1)
            gamma_dec = self.cfg.tr_gamma_dec * 0.8
            gamma_inc = self.cfg.tr_gamma_inc * 0.8
        else:
            eta_lo = self.cfg.tr_eta_lo
            eta_hi = self.cfg.tr_eta_hi
            gamma_dec = self.cfg.tr_gamma_dec
            gamma_inc = self.cfg.tr_gamma_inc
        return eta_lo, eta_hi, gamma_dec, gamma_inc

    def _analyze_boundary_usage(self, step_norm: float) -> bool:
        """Return True if the step meaningfully used the TR boundary (recent-context aware)."""
        trig = self.cfg.tr_boundary_trigger
        if step_norm < trig * self.radius:
            return False
        if len(self._recent_boundaries) >= 3:
            recent = float(np.mean(self._recent_boundaries[-3:]))
            if recent < 0.5:
                return step_norm >= 0.95 * self.radius
        return True

    def _analyze_curvature(self, H: np.ndarray, step: np.ndarray) -> Dict[str, float]:
        """Compute curvature stats along `step` (and the metric-weighted variant if L exists)."""
        try:
            sn2 = float(np.dot(step, step))
            if sn2 < 1e-16:
                return {}
            Hs = H @ step if not sp.issparse(H) else H @ step
            curv = float(np.dot(step, Hs) / sn2)
            if self.L is not None:
                try:
                    Ls = self.L @ step
                    ecurv = float(np.dot(Ls, Ls) / sn2)
                except Exception:
                    ecurv = curv
            else:
                ecurv = curv
            return {"curvature": curv, "ellip_curvature": ecurv, "is_negative": curv < -1e-12, "is_nearly_singular": abs(curv) < 1e-12}
        except Exception:
            return {}

    def _compute_shrink_factor(self, rho: float, curvature_info: Dict) -> float:
        """Return γ_dec adjusted by failure severity and curvature cues."""
        base = self.cfg.tr_gamma_dec
        # Very poor agreement → stronger shrink; (order chosen for clarity)
        if rho < -2.0:
            base *= 0.25
        elif rho < -0.5:
            base *= 0.5
        # Negative curvature: prefer keeping some room (slightly less shrink)
        if curvature_info.get("is_negative", False):
            base *= 0.8
        return base

    def _compute_growth_factor(self, rho: float, curvature_info: Dict, act_sz: Optional[int]) -> float:
        """Return γ_inc tempered by recent ρ, curvature, and constraint density."""
        base = self.cfg.tr_gamma_inc

        if len(self._recent_ratios) >= 3:
            r = float(np.mean(self._recent_ratios[-3:]))
            if r < 0.5:
                base = min(base, 1.2)
            elif r > 2.0:
                base = min(base * 1.2, 3.0)

        if self.cfg.curvature_aware_growth and curvature_info:
            c = curvature_info.get("curvature", 0.0)
            if c > 1e-6:
                base = min(base * 1.3, 2.5)
            elif curvature_info.get("is_nearly_singular", False):
                base = min(base, 1.1)

        if act_sz is not None and self._n_last is not None:
            dens = act_sz / max(1, self._n_last)
            if dens > 0.5:
                base = min(base, 1.5)

        return base

    # ========================= Trust Region Subproblem Solver =========================

    def solve(
        self,
        H,
        g,
        JE=None,
        cE=None,
        JI=None,
        cI=None,
        tol=None,
        max_iter=None,
        act_tol=None,
        max_ws_add=None,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Solve the TR subproblem

            minimize    ½ pᵀ H p + gᵀ p
            subject to  JE p + cE = 0
                        JI p + cI ≤ 0
                        ||p||_TR ≤ Δ

        where ||·||_TR follows `tr_norm_type`.

        Returns
        -------
        p : ndarray
            The computed step.
        p_norm : float
            Its trust-region norm.
        info : dict
            Status and solver diagnostics (iterations, boundary/neg-curv exits, etc.).
        """
        cfg = self.cfg
        n = g.size
        Delta = float(self.radius)

        if not (np.isfinite(Delta) and Delta > 0):
            return np.zeros(n), 0.0, {"status": "invalid_radius"}

        g_norm = np.linalg.norm(g)
        if not np.isfinite(g_norm):
            return np.zeros(n), 0.0, {"status": "invalid_gradient"}
        if g_norm == 0:
            return np.zeros(n), 0.0, {"status": "zero_gradient"}

        tol = cfg.cg_tol if tol is None else float(tol)
        max_iter = cfg.cg_max_iter if max_iter is None else int(max_iter)
        act_tol = cfg.act_tol if act_tol is None else float(act_tol)
        max_ws_add = cfg.max_ws_add if max_ws_add is None else int(max_ws_add)

        JE = self._normalize_matrix(JE); cE = self._normalize_vector(cE)
        JI = self._normalize_matrix(JI); cI = self._normalize_vector(cI)
        self._n_last = n

        if self.norm_type == "ellip":
            self.set_metric_from_H(H)

        if JE is not None and JE.size > 0:
            return self._solve_with_equalities(H, g, JE, cE, JI, cI, Delta, tol, max_iter, act_tol, max_ws_add)
        elif JI is not None and JI.size > 0:
            return self._solve_with_inequalities(H, g, JI, cI, Delta, tol, max_iter, act_tol, max_ws_add)
        else:
            return self._solve_unconstrained(H, g, Delta, tol, max_iter)

    def _solve_unconstrained(self, H, g, Delta, tol, max_iter) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Solve unconstrained TRSP, choosing ellipsoidal or Euclidean path automatically."""
        try:
            if self.norm_type == "ellip" and self.L is not None:
                return self._solve_ellipsoidal_unconstrained(H, g, Delta, tol, max_iter)
            else:
                return self._solve_euclidean_unconstrained(H, g, Delta, tol, max_iter)
        except Exception as e:
            p_c = self._cauchy_point(H, g, Delta)
            return p_c, self._tr_norm(p_c), {"status": "fallback_cauchy", "error": str(e)}

    def _solve_ellipsoidal_unconstrained(self, H, g, Delta, tol, max_iter) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Solve TRSP with ||p||_TR = ||L p||₂ by transforming to y = L p, ||y||₂ ≤ Δ.

        Minimizes ½ yᵀ (L^{-T} H L^{-1}) y + (L^{-T} g)ᵀ y via CG in y-space,
        then maps back p = L^{-1} y. Falls back to Euclidean if anything fails.
        """
        try:
            W = self.Linv; WT = W.T  # L^{-1}, L^{-T}

            if sp.issparse(H):
                def Htilde_mv(y): return WT @ (H @ (W @ y))
                gtilde = WT @ g
                y, info = self._cg_tr_callable(Htilde_mv, gtilde, Delta, tol, max_iter)
                p = W @ y
            else:
                Htilde = WT @ H @ W
                gtilde = WT @ g
                y, info = self._cg_tr_dense(Htilde, gtilde, Delta, tol, max_iter)
                p = W @ y

            p_norm = self._tr_norm(p)
            if p_norm > Delta * (1 + 1e-10):
                p = self._project_to_boundary(p, Delta); p_norm = Delta
            return p, p_norm, info
        except Exception:
            return self._solve_euclidean_unconstrained(H, g, Delta, tol, max_iter)

    def _solve_euclidean_unconstrained(self, H, g, Delta, tol, max_iter) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Solve TRSP with Euclidean (or 'scaled') norm via CG (dense/sparse/operator)."""
        if sp.issparse(H):
            p, info = self._cg_tr_sparse(H, g, Delta, tol, max_iter)
        elif isinstance(H, np.ndarray):
            p, info = self._cg_tr_dense(H, g, Delta, tol, max_iter)
        else:
            p, info = self._cg_tr_callable(H, g, Delta, tol, max_iter)

        p_norm = np.linalg.norm(p)
        if p_norm > Delta * (1 + 1e-10):
            p = (Delta / p_norm) * p; p_norm = Delta
        return p, p_norm, info

    def _solve_with_equalities(self, H, g, JE, cE, JI, cI, Delta, tol, max_iter, act_tol, max_ws_add) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Equality handling via particular solution + nullspace reduction:
            Find p = p_part + Z z, with JE p + cE = 0 and Z a basis for null(JE).
            Solve reduced TRSP in z; then handle inequalities if present.
        """
        try:
            n = g.size
            m_eq = JE.shape[0]

            p_part = self._compute_particular_solution(JE, cE)
            Z = self._compute_nullspace_basis(JE)

            if Z.shape[1] == 0:
                p_norm = self._tr_norm(p_part)
                if JI is not None and JI.size > 0:
                    viol = JI @ p_part + cI
                    if np.any(viol > act_tol):
                        return p_part, p_norm, {"status": "equality_infeasible"}
                return p_part, p_norm, {"status": "equality_constrained"}

            try:
                if sp.issparse(H):
                    Hz = Z.T @ (H @ Z)
                    gz = Z.T @ (g + H @ p_part)
                else:
                    Hz = Z.T @ H @ Z
                    gz = Z.T @ (g + H @ p_part)
            except Exception:
                Hz = np.eye(Z.shape[1]) * self.cfg.metric_regularization
                gz = Z.T @ g

            if self.norm_type == "ellip" and self.L is not None:
                pz = self._solve_nullspace_ellipsoidal(Hz, gz, Z, p_part, Delta, tol, max_iter)
            else:
                if Z.shape[1] <= 100 and isinstance(Hz, np.ndarray):
                    pz, _ = self._cg_tr_dense(Hz, gz, Delta, tol, max_iter)
                else:
                    pz, _ = self._cg_tr_sparse(Hz, gz, Delta, tol, max_iter)

            p = p_part + Z @ pz

            if JI is not None and JI.size > 0:
                p, p_norm, info = self._handle_inequality_violations(p, H, g, JI, cI, Delta, act_tol, max_ws_add)
            else:
                p_norm = self._tr_norm(p); info = {"status": "nullspace_solved"}

            return p, p_norm, info

        except Exception as e:
            p_c = self._cauchy_point(H, g, Delta)
            return p_c, self._tr_norm(p_c), {"status": "equality_fallback", "error": str(e)}

    def _solve_with_inequalities(self, H, g, JI, cI, Delta, tol, max_iter, act_tol, max_ws_add) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Inequalities via a simple active-set loop:
            - promote violated inequalities to equalities,
            - solve the equality-constrained TRSP,
            - iterate until feasible or reaching `max_ws_add`.
        """
        active_set = set()
        iteration = 0

        while iteration <= max_ws_add:
            if len(active_set) > 0:
                idx = list(active_set)
                JE_active = JI[idx]; cE_active = cI[idx]
            else:
                JE_active = None; cE_active = None

            p, p_norm, info = self._solve_with_equalities(
                H, g, JE_active, cE_active, None, None, Delta, tol, max_iter, act_tol, max_ws_add
            )

            if JI is None or JI.size == 0:
                break

            violations = JI @ p + cI
            violated = violations > act_tol
            if not np.any(violated):
                break

            if iteration >= max_ws_add:
                info["note"] = "max_active_set_iterations"
                break

            violated_indices = np.where(violated)[0]
            worst_idx = violated_indices[np.argmax(violations[violated_indices])]
            if worst_idx in active_set:
                break  # avoid loops
            active_set.add(worst_idx)
            iteration += 1

        info["active_set_size"] = len(active_set)
        info["active_set_iterations"] = iteration
        return p, p_norm, info

    # ========================= CG Solvers =========================

    def _cg_tr_dense(self, H, g, Delta, tol, max_iter) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Dense CG TR solver; uses Numba kernel when available (fast path)."""
        if NUMBA_OK and isinstance(H, np.ndarray) and isinstance(g, np.ndarray):
            return cg_tr_numba_or_numpy(H, g, Delta, tol, max_iter)
        return self._cg_tr_python(H, g, Delta, tol, max_iter)

    def _cg_tr_sparse(self, H, g, Delta, tol, max_iter) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Sparse/effective-operator CG TR solver (pure Python kernel)."""
        return self._cg_tr_python(H, g, Delta, tol, max_iter)

    def _cg_tr_callable(self, H_op, g, Delta, tol, max_iter) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Callable-Hessian CG TR solver (expects H_op(d) → Hd)."""
        return self._cg_tr_python(H_op, g, Delta, tol, max_iter)

    def _cg_tr_python(self, H, g, Delta, tol, max_iter) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Robust CG TR method (Python reference kernel).

        Exits on negative curvature or boundary intersection by stepping to the
        TR boundary along the current direction.
        """
        n = g.size
        p = np.zeros(n, dtype=float)
        r = -g.copy()
        d = r.copy()
        rr = float(np.dot(r, r))

        info = {"iterations": 0, "status": "unknown"}

        if np.sqrt(rr) < tol:
            info["status"] = "initial_convergence"
            return p, info

        restart_threshold = self.cfg.cg_restart_every
        neg_curv_tol = self.cfg.cg_neg_curv_tol

        for k in range(max_iter):
            info["iterations"] = k + 1

            try:
                Hd = H(d) if callable(H) else (H @ d if sp.issparse(H) else np.dot(H, d))
            except Exception as e:
                info["status"] = "hessian_error"; info["error"] = str(e)
                break

            dHd = float(np.dot(d, Hd))
            dd = float(np.dot(d, d))

            if dHd <= neg_curv_tol * max(1.0, dd):
                tau = self._find_boundary_intersection(p, d, Delta)
                p_final = p + tau * d
                info["status"] = "negative_curvature"
                self._cg_stats["neg_curv_exits"] += 1
                return p_final, info

            alpha = rr / max(dHd, 1e-16)
            p_trial = p + alpha * d

            if self._tr_norm(p_trial) > Delta * (1 + 1e-12):
                tau = self._find_boundary_intersection(p, d, Delta)
                p_final = p + tau * d
                info["status"] = "boundary_hit"
                self._cg_stats["boundary_hits"] += 1
                return p_final, info

            r_new = r - alpha * Hd
            rr_new = float(np.dot(r_new, r_new))

            if np.sqrt(rr_new) < tol:
                info["status"] = "converged"
                return p_trial, info

            beta = rr_new / max(rr, 1e-16)
            d = r_new.copy() if ((k + 1) % restart_threshold == 0) else (r_new + beta * d)

            p, r, rr = p_trial, r_new, rr_new

        info["status"] = "max_iterations"
        self._cg_stats["total_iterations"] += info["iterations"]
        return p, info

    def _find_boundary_intersection(self, p: np.ndarray, d: np.ndarray, Delta: float) -> float:
        """Return τ ≥ 0 such that ||p + τ d||_TR = Δ (ellipsoidal-aware when L exists)."""
        if self.norm_type == "ellip" and self.L is not None:
            return self._find_ellipsoidal_intersection(p, d, Delta)
        return self._find_euclidean_intersection(p, d, Delta)

    def _find_euclidean_intersection(self, p: np.ndarray, d: np.ndarray, Delta: float) -> float:
        """Closed-form intersection with Euclidean TR boundary."""
        pp = float(np.dot(p, p)); pd = float(np.dot(p, d)); dd = float(np.dot(d, d))
        if dd < 1e-16:
            return 0.0
        c = pp - Delta * Delta
        disc = pd * pd - dd * c
        if disc < 0:
            return 0.0
        root = np.sqrt(disc)
        t1 = (-pd + root) / dd
        t2 = (-pd - root) / dd
        return max(t1, t2, 0.0)

    def _find_ellipsoidal_intersection(self, p: np.ndarray, d: np.ndarray, Delta: float) -> float:
        """Quadratic intersection with ||L(p + τ d)||₂ = Δ."""
        try:
            Lp = self.L @ p; Ld = self.L @ d
            pp = float(np.dot(Lp, Lp)); pd = float(np.dot(Lp, Ld)); dd = float(np.dot(Ld, Ld))
            if dd < 1e-16:
                return 0.0
            c = pp - Delta * Delta
            disc = pd * pd - dd * c
            if disc < 0:
                return 0.0
            root = np.sqrt(disc)
            t1 = (-pd + root) / dd
            t2 = (-pd - root) / dd
            return max(t1, t2, 0.0)
        except Exception:
            return self._find_euclidean_intersection(p, d, Delta)

    # ========================= Helper Methods =========================

    def _compute_particular_solution(self, JE: np.ndarray, cE: np.ndarray) -> np.ndarray:
        """Return the minimum-norm particular solution to JE p + cE = 0."""
        try:
            return la.lstsq(JE, -cE, rcond=1e-12)[0]
        except Exception:
            return np.zeros(JE.shape[1])

    def _compute_nullspace_basis(self, JE: np.ndarray) -> np.ndarray:
        """Return an orthonormal basis Z for null(JE)."""
        try:
            Q, _ = la.qr(JE.T, mode="complete")
            return Q[:, JE.shape[0]:]
        except Exception:
            n = JE.shape[1]
            try:
                proj = JE.T @ la.solve(JE @ JE.T, JE)
                return np.eye(n) - proj
            except Exception:
                return np.eye(n)

    def _solve_nullspace_ellipsoidal(self, Hz, gz, Z, p_part, Delta, tol, max_iter):
        """
        Reduced-space solve under ellipsoidal metric.

        Note
        ----
        This path currently treats the reduced problem with standard CG in z.
        For high accuracy, one can incorporate Zᵀ Lᵀ L Z into the TR term.
        """
        try:
            if Z.shape[1] <= 50:
                pz, _ = self._cg_tr_dense(Hz, gz, Delta, tol, max_iter)
            else:
                pz, _ = self._cg_tr_sparse(Hz, gz, Delta, tol, max_iter)
            return pz
        except Exception:
            return np.zeros(gz.size)

    def _handle_inequality_violations(self, p, H, g, JI, cI, Delta, act_tol, max_ws_add):
        """Projection for small violations; otherwise delegate to active-set loop."""
        violations = JI @ p + cI
        violated = violations > act_tol
        if not np.any(violated):
            return p, self._tr_norm(p), {"status": "feasible"}

        max_violation = float(np.max(violations))
        if max_violation < 10 * act_tol:
            p_proj = self._project_to_inequalities(p, JI, cI, act_tol)
            p_norm = self._tr_norm(p_proj)
            if p_norm <= Delta * (1 + 1e-10):
                return p_proj, p_norm, {"status": "projected"}

        return self._solve_with_inequalities(H, g, JI, cI, Delta, 1e-8, 100, act_tol, max_ws_add)

    def _project_to_inequalities(self, p, JI, cI, tol):
        """Project onto the most violated linear inequality iteratively (up to 10 passes)."""
        p_proj = p.copy()
        for _ in range(10):
            violations = JI @ p_proj + cI
            violated = violations > tol
            if not np.any(violated):
                break
            worst_idx = int(np.argmax(violations))
            ji = JI[worst_idx]; ci = cI[worst_idx]
            viol = float(violations[worst_idx])
            ji_norm_sq = float(np.dot(ji, ji))
            if ji_norm_sq > 1e-12:
                p_proj = p_proj - (viol / ji_norm_sq) * ji
        return p_proj

    def _project_to_boundary(self, p: np.ndarray, Delta: float) -> np.ndarray:
        """Scale p to the TR boundary if ||p||_TR > Δ."""
        p_norm = self._tr_norm(p)
        if p_norm <= Delta or p_norm == 0:
            return p
        return (Delta / p_norm) * p

    def _cauchy_point(self, H, g, Delta) -> np.ndarray:
        """
        Return a Cauchy (steepest descent) point as a robust fallback.

        Chooses α = min(‖g‖/⟨d,Hd⟩, α_TR) for positive curvature; otherwise α_TR.
        """
        g_norm = np.linalg.norm(g)
        if g_norm == 0:
            return np.zeros_like(g)

        d = -g / g_norm
        try:
            Hd = H(d) if callable(H) else (H @ d if sp.issparse(H) else H @ d)
            curv = float(np.dot(d, Hd))
            if curv > 1e-12:
                alpha_opt = g_norm / curv
                if self.norm_type == "ellip" and self.L is not None:
                    Ld = self.L @ d; m = np.linalg.norm(Ld)
                    alpha_tr = Delta / m if m > 1e-12 else Delta
                else:
                    alpha_tr = Delta
                alpha = min(alpha_opt, alpha_tr)
            else:
                if self.norm_type == "ellip" and self.L is not None:
                    Ld = self.L @ d; m = np.linalg.norm(Ld)
                    alpha = Delta / m if m > 1e-12 else Delta
                else:
                    alpha = Delta
        except Exception:
            alpha = Delta / g_norm

        return alpha * (-g)

    @staticmethod
    def _normalize_matrix(M):
        """Return None for empty matrices; otherwise the input as-is."""
        if M is None:
            return None
        if hasattr(M, "size") and M.size == 0:
            return None
        return M

    @staticmethod
    def _normalize_vector(v):
        """Return None for empty vectors; otherwise a contiguous 1-D float array."""
        if v is None:
            return None
        if hasattr(v, "size") and v.size == 0:
            return None
        return np.asarray(v, dtype=float).ravel()

    # ========================= Public Interface =========================

    def get_radius(self) -> float:
        """Return the current trust-region radius Δ."""
        return self.radius

    def set_radius(self, radius: float):
        """Set Δ with min/max clipping."""
        self.radius = float(np.clip(radius, self.cfg.tr_delta_min, self.cfg.tr_delta_max))

    def reset(self):
        """Reset Δ, rejection counter, histories, and metric."""
        self.radius = self.cfg.tr_delta0
        self.rej = 0
        self._rho_smoothed = None
        self._recent_ratios.clear()
        self._recent_boundaries.clear()
        self._recent_curvatures.clear()
        self._reset_metric()

    def get_stats(self) -> Dict[str, Any]:
        """Return diagnostic snapshot of recent performance and metric quality."""
        return {
            "radius": self.radius,
            "rejections": self.rej,
            "smoothed_ratio": self._rho_smoothed,
            "cg_stats": self._cg_stats.copy(),
            "metric_condition": self._metric_condition,
            "recent_performance": {
                "avg_ratio": (np.mean(self._recent_ratios) if self._recent_ratios else None),
                "avg_boundary_usage": (np.mean(self._recent_boundaries) if self._recent_boundaries else None),
                "avg_curvature": (np.mean(self._recent_curvatures) if self._recent_curvatures else None),
            },
        }

    def configure_for_algorithm(self, algorithm: str):
        """
        Apply preset TR parameters for a given algorithm family.

        Parameters
        ----------
        algorithm : {"dfo", "sqp"}
            'dfo' → more conservative settings and stronger smoothing.
            'sqp' → standard wide-acceptance settings and faster growth.
        """
        if algorithm.lower() == "dfo":
            self.cfg.dfo_mode = True
            self.cfg.tr_eta_lo = 0.01
            self.cfg.tr_eta_hi = 0.85
            self.cfg.tr_gamma_dec = 0.25
            self.cfg.tr_gamma_inc = 1.3
            self.cfg.rho_smooth = 0.1
        elif algorithm.lower() == "sqp":
            self.cfg.dfo_mode = False
            self.cfg.tr_eta_lo = 0.1
            self.cfg.tr_eta_hi = 0.9
            self.cfg.tr_gamma_dec = 0.5
            self.cfg.tr_gamma_inc = 2.0
            self.cfg.rho_smooth = 0.3


# ========================= Numba-Accelerated Dense Solver =========================

def cg_tr_numba_or_numpy(H, g, Delta, tol, max_iter):
    """Dispatch to numba kernel when available; otherwise use NumPy reference."""
    if NUMBA_OK and isinstance(H, np.ndarray) and isinstance(g, np.ndarray):
        return _cg_tr_numba(H, g.astype(np.float64), float(Delta), float(tol), int(max_iter))
    else:
        return _cg_tr_numpy_reference(H, g, Delta, tol, max_iter)


def _cg_tr_numpy_reference(H, g, Delta, tol, max_iter):
    """Dense CG TR reference kernel (NumPy-only)."""
    n = g.size
    p = np.zeros(n, dtype=float)
    r = -g.copy()
    d = r.copy()
    rr = float(np.dot(r, r))

    def boundary_intersection(p, d, Delta):
        pp = float(np.dot(p, p)); pd = float(np.dot(p, d)); dd = float(np.dot(d, d))
        c = pp - Delta * Delta
        disc = max(0.0, pd * pd - dd * c)
        root = np.sqrt(disc)
        t1 = (-pd + root) / max(dd, 1e-32)
        t2 = (-pd - root) / max(dd, 1e-32)
        return max(t1, t2, 0.0)

    for iteration in range(max_iter):
        Hd = H @ d
        dHd = float(np.dot(d, Hd)); dd = float(np.dot(d, d))

        if dHd <= 1e-14 * max(1.0, dd):
            tau = boundary_intersection(p, d, Delta)
            return p + tau * d, {"status": "negative_curvature", "iterations": iteration + 1}

        alpha = rr / max(dHd, 1e-32)
        p_trial = p + alpha * d

        if np.linalg.norm(p_trial) > Delta * (1 + 1e-12):
            tau = boundary_intersection(p, d, Delta)
            return p + tau * d, {"status": "boundary_hit", "iterations": iteration + 1}

        r = r - alpha * Hd
        rr_new = float(np.dot(r, r))

        if np.sqrt(rr_new) < tol:
            return p_trial, {"status": "converged", "iterations": iteration + 1}

        beta = rr_new / max(rr, 1e-32)
        d = r + beta * d
        p, rr = p_trial, rr_new

    return p, {"status": "max_iterations", "iterations": max_iter}


# numba-optimized kernel (optional)
if NUMBA_OK:

    @nb.njit(cache=True, fastmath=True)
    def _boundary_tau_numba(p, d, D):
        pp = 0.0; dd = 0.0; pd = 0.0
        for i in range(p.size):
            pp += p[i] * p[i]
            dd += d[i] * d[i]
            pd += p[i] * d[i]
        c = pp - D * D
        disc = pd * pd - dd * c
        if disc < 0.0:
            disc = 0.0
        root = np.sqrt(disc)
        denom = dd if dd > 1e-32 else 1e-32
        t1 = (-pd + root) / denom
        t2 = (-pd - root) / denom
        return t1 if t1 > t2 else t2

    @nb.njit(cache=True, fastmath=True)
    def _cg_tr_numba(H, g, Delta, tol, max_iter):
        n = g.size
        p = np.zeros(n, dtype=np.float64)
        r = -g.copy()
        d = r.copy()
        rr = 0.0
        for i in range(n):
            rr += r[i] * r[i]

        for iteration in range(max_iter):
            Hd = H @ d
            dHd = 0.0; dd = 0.0
            for i in range(n):
                dHd += d[i] * Hd[i]
                dd  += d[i] * d[i]

            if dHd <= 1e-14 * (1.0 if dd <= 1.0 else dd):
                tau = _boundary_tau_numba(p, d, Delta)
                return p + tau * d, {"status": "negative_curvature"}

            alpha = rr / (dHd if dHd > 1e-32 else 1e-32)
            p_trial = p + alpha * d

            nrm = 0.0
            for i in range(n):
                nrm += p_trial[i] * p_trial[i]
            nrm = np.sqrt(nrm)

            if nrm > Delta * 1.000000001:
                tau = _boundary_tau_numba(p, d, Delta)
                return p + tau * d, {"status": "boundary_hit"}

            for i in range(n):
                r[i] = r[i] - alpha * Hd[i]

            rr_new = 0.0
            for i in range(n):
                rr_new += r[i] * r[i]

            if np.sqrt(rr_new) < tol:
                return p_trial, {"status": "converged"}

            beta = rr_new / (rr if rr > 1e-32 else 1e-32)
            for i in range(n):
                d[i] = r[i] + beta * d[i]

            p = p_trial
            rr = rr_new

        return p, {"status": "max_iterations"}

else:

    def _cg_tr_numba(H, g, Delta, tol, max_iter):
        return _cg_tr_numpy_reference(H, g, Delta, tol, max_iter)
