import logging
from typing import Optional, Tuple

import numpy as np

from .aux import *  # assumes Model exposes n, (optional) lb/ub; keep your imports


class SOCCorrector:
    """
    Polished SOC:
      • Row selection: |cE|>tol, cI>violation_tol (near-active only).
      • Weighting: equalities > inequalities.
      • Optional complementarity target for I-rows: target ≈ mu / nu (if (mu,nu) given).
      • Scaling: mild row/col equilibration improves conditioning.
      • Solve:
          - If H given: projected-Newton via Schur (J M^{-1} J^T + lam I) y = -c, corr = -M^{-1} J^T y
            with M = H + sigma I  (sigma small ridge).
          - Else: Tikhonov-regularized least-squares for J corr = -c.
      • TR-consistent cap:
          - If tr_norm_type == "ellip": cap in ||·||_M (M = H + sigma I).
          - Else (Euclidean TR): cap in ||·||_2 even if H is provided.
      • Bounds: clip step so x+s within [lb,ub] if model exposes lb/ub.
      • Slack guard: ensure new slacks remain > 0 (if s & mu are provided).
    Returns (corr, needs_restoration).
    """

    def __init__(self, cfg: "SQPConfig"):
        self.cfg = cfg
        self._cached_eval: Optional[dict] = None
        self._cached_x_trial: Optional[np.ndarray] = None

    # ---------- small helpers ----------
    @staticmethod
    def _is_bad(arr) -> bool:
        return (arr is not None) and (np.any(~np.isfinite(np.asarray(arr))))

    @staticmethod
    def _norm_M(s: np.ndarray, M: Optional[np.ndarray]) -> float:
        if M is None:
            return float(np.linalg.norm(s))
        v = M @ s
        val = float(s @ v)
        return float(np.sqrt(max(val, 0.0)))

    @staticmethod
    def _clip_to_ball_M(s: np.ndarray, M: Optional[np.ndarray], radius: float) -> np.ndarray:
        if not np.isfinite(radius) or radius <= 0.0:
            return s
        nrm = SOCCorrector._norm_M(s, M)
        if nrm <= radius or nrm <= 1e-16:
            return s
        return (radius / nrm) * s

    @staticmethod
    def _diag_equilibrate(J: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Very light row/column equilibration (one pass):
        returns (J_eq, D_row, D_col) such that J_eq = D_row @ J @ D_col
        """
        rs = np.sqrt(np.maximum(np.sum(J * J, axis=1), 1e-16))
        Dr = 1.0 / rs
        J1 = (Dr[:, None] * J)
        cs = np.sqrt(np.maximum(np.sum(J1 * J1, axis=0), 1e-16))
        Dc = 1.0 / cs
        J_eq = J1 * Dc[None, :]
        return J_eq, Dr, Dc

    def _apply_bounds_clip(self, x_trial: np.ndarray, s: np.ndarray, model: "Model") -> np.ndarray:
        lb = getattr(model, "lb", None)
        ub = getattr(model, "ub", None)
        if lb is None and ub is None:
            return s
        x_new = x_trial + s
        if lb is not None:
            x_new = np.maximum(x_new, np.asarray(lb))
        if ub is not None:
            x_new = np.minimum(x_new, np.asarray(ub))
        return x_new - x_trial

    # ---------- main entry ----------
    def compute_correction(
        self,
        model: "Model",
        x_trial: np.ndarray,
        Delta: Optional[float] = None,
        s: Optional[np.ndarray] = None,         # step already taken before SOC (for remaining radius)
        mu: float = 0.0,                        # IP mu, optional (for complementarity target)
        funnel: Optional["Funnel"] = None,
        filter: Optional["Filter"] = None,
        theta0: Optional[float] = None,
        f0: Optional[float] = None,
        pred_df: float = 0.0,
        pred_dtheta: float = 0.0,
        H: Optional[np.ndarray] = None,         # curvature for projected-Newton (optional)
        nu: Optional[np.ndarray] = None,        # inequality multipliers (optional)
    ) -> Tuple[np.ndarray, bool]:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        n = model.n
        if not getattr(self.cfg, "use_soc", False):
            return np.zeros(n), False

        # Cache model eval at x_trial
        if self._cached_x_trial is None or not np.allclose(self._cached_x_trial, x_trial):
            d = model.eval_all(x_trial)
            self._cached_eval = d
            self._cached_x_trial = x_trial.copy()
        else:
            d = self._cached_eval

        cE, cI, JE, JI = d["cE"], d["cI"], d["JE"], d["JI"]
        if self._is_bad(cE) or self._is_bad(cI):
            logging.debug("SOC: invalid constraint values at x_trial")
            return np.zeros(n), True

        # Selection masks
        soc_tol = float(getattr(self.cfg, "soc_tol", 1e-8))
        soc_violation = float(getattr(self.cfg, "soc_violation", 0.0))

        rows = []
        rhs = []
        wts = []  # per-row weights

        # Equalities: |cE| > tol
        if cE is not None and JE is not None and getattr(JE, "size", 0):
            mE_mask = np.abs(np.asarray(cE).ravel()) > soc_tol
            if np.any(mE_mask):
                JE_ = JE[mE_mask] if sp.issparse(JE) else np.asarray(JE)[mE_mask]
                rows.append(JE_)
                rhs.append(-np.asarray(cE).ravel()[mE_mask])
                wE = float(getattr(self.cfg, "soc_wE", 10.0))
                wts.append(np.full(JE_.shape[0], wE))

        # Inequalities: cI > violation threshold (near-active only)
        if cI is not None and JI is not None and getattr(JI, "size", 0):
            mI_mask = np.asarray(cI).ravel() > soc_violation
            if np.any(mI_mask):
                JI_ = JI[mI_mask] if sp.issparse(JI) else np.asarray(JI)[mI_mask]
                ci_sel = np.asarray(cI).ravel()[mI_mask].copy()

                # Complementarity-aware target if (mu, nu) available
                if (mu is not None and mu > 0.0) and (nu is not None and nu.size == cI.size):
                    tiny = 1e-12
                    target = mu / np.maximum(np.abs(np.asarray(nu).ravel()[mI_mask]), tiny)
                    rhs_I = -(ci_sel - np.maximum(target, 0.0))
                else:
                    rhs_I = -ci_sel  # pure feasibility

                rows.append(JI_)
                rhs.append(rhs_I)
                wI = float(getattr(self.cfg, "soc_wI", 1.0))
                wts.append(np.full(JI_.shape[0], wI))

        if not rows:
            logging.debug("SOC: no rows selected (nothing to correct).")
            return np.zeros(n), False

        # Build dense design; convert once if needed
        if any(sp.issparse(R) for R in rows):
            rows = [R.toarray() if sp.issparse(R) else R for R in rows]
        J = np.vstack(rows)            # (m x n)
        r = np.concatenate(rhs)        # (m,)
        w = np.concatenate(wts)        # (m,)

        # Diagonal scaling (one-pass equilibrate)
        try:
            J_eq, Dr, Dc = self._diag_equilibrate(J)
            r_eq = Dr * r
            w_eq = Dr * w
        except Exception:
            J_eq, r_eq, w_eq = J, r, w
            Dc = np.ones(J.shape[1])

        # Apply per-row weights (W^{1/2})
        Wsqrt = np.sqrt(np.maximum(w_eq, 1e-16))
        Jw = J_eq * Wsqrt[:, None]
        rw = r_eq * Wsqrt

        # Solver knobs
        reg = float(getattr(self.cfg, "soc_reg", 1e-8))
        sigma = float(getattr(self.cfg, "soc_sigma", 1e-10))  # ridge for M = H + sigma I

        # Solve for correction in original coords
        corr_eq = np.zeros(n)

        try:
            if H is not None:
                # Projected-Newton (curvature-aware)
                def Minv_mv(v):
                    A = H + sigma * np.eye(n)
                    if n <= 400:
                        return np.linalg.solve(A, v)
                    else:
                        from scipy.sparse.linalg import cg
                        out, _ = cg(A, v, maxiter=200, tol=1e-10)
                        return out

                lam = reg * max(1.0, float(np.trace(Jw.T @ Jw)) / max(1, Jw.shape[0]))

                def S_mv(y):
                    Jy = Jw.T @ y
                    MinvJy = Minv_mv(Jy)
                    return Jw @ MinvJy + lam * y

                from scipy.sparse.linalg import cg
                y, ok = cg(spla.aslinearoperator(S_mv), rw, maxiter=500, tol=1e-10)
                if ok != 0:
                    S_dense = Jw @ np.linalg.solve(H + sigma * np.eye(n), Jw.T) + lam * np.eye(Jw.shape[0])
                    y = np.linalg.solve(S_dense, rw)

                corr_eq = -Minv_mv(Jw.T @ y)

            else:
                # Tikhonov-regularized LS
                m, nJ = Jw.shape
                if m >= nJ:
                    JtJ = Jw.T @ Jw
                    lam = reg * (np.trace(JtJ) / max(1.0, nJ))
                    corr_eq = np.linalg.solve(JtJ + lam * np.eye(nJ), Jw.T @ rw)
                else:
                    JJt = Jw @ Jw.T
                    lam = reg * (np.trace(JJt) / max(1.0, m))
                    y = np.linalg.solve(JJt + lam * np.eye(m), rw)
                    corr_eq = Jw.T @ y

        except Exception as e:
            logging.debug(f"SOC: solve failed ({e}); returning zero correction")
            corr_eq = np.zeros(n)

        # Undo column scaling
        Dc_safe = np.where(np.abs(Dc) > 0, Dc, 1.0)
        corr = corr_eq / Dc_safe

        # ---- TR-consistent cap ----
        theta_scale = max(float(getattr(self.cfg, "soc_theta_scale", 1.0)), 1e-16)
        theta0_val  = float(theta0 or 0.0)
        min_norm    = float(getattr(self.cfg, "soc_min_norm", 1e-16))
        kappa       = float(getattr(self.cfg, "soc_kappa", 0.9))
        cap_abs     = float(getattr(self.cfg, "soc_cap_abs", 0.0))

        # Which norm does the TR use? ("2" or "ellip"); set this from SQP each iter:
        tr_norm_type = getattr(self.cfg, "tr_norm_type", "2")
        use_M_cap = (tr_norm_type == "ellip")  # only cap in M-norm if TR is ellipsoidal

        M_for_norm = (H + sigma * np.eye(n)) if (use_M_cap and H is not None) else None

        # Remaining radius in the SAME norm as the cap
        if Delta is not None and Delta > 0.0:
            if s is not None and s.size:
                rem = max(
                    0.0,
                    float(Delta) - (self._norm_M(s, M_for_norm) if use_M_cap else float(np.linalg.norm(s))),
                )
            else:
                rem = float(Delta)
            cap = kappa * rem / (1.0 + theta0_val / theta_scale)
        else:
            cap = (cap_abs / (1.0 + theta0_val / theta_scale)) if cap_abs > 0.0 else np.inf

        if np.isfinite(cap):
            if use_M_cap:
                corr = self._clip_to_ball_M(corr, M_for_norm, max(cap, min_norm))
            else:
                nrm = float(np.linalg.norm(corr))
                if nrm > max(cap, min_norm):
                    corr = (cap / max(nrm, 1e-16)) * corr

        # Enforce variable bounds if model carries lb/ub
        corr = self._apply_bounds_clip(x_trial, corr, model)

        # Slack positivity guard (linearized)
        if s is not None and mu > 0.0 and (cI is not None) and (JI is not None) and getattr(JI, "size", 0):
            JI_arr = JI.toarray() if hasattr(JI, "toarray") else np.asarray(JI)
            inc_I = JI_arr @ corr
            s_tilde = s - (np.asarray(cI).ravel() + inc_I)
            if np.any(s_tilde <= 0.0):
                alpha_max = 0.99 * np.min(np.where(
                    inc_I > -1e-16, np.inf, (s - np.asarray(cI).ravel()) / (-inc_I)
                ))
                if np.isfinite(alpha_max) and alpha_max < 1.0:
                    corr = alpha_max * corr
                else:
                    logging.debug("SOC: slacks would be nonpositive; request restoration.")
                    return corr, True

        # Optional acceptability via funnel/filter
        needs_restoration = False
        if (funnel or filter) and (theta0 is not None) and (f0 is not None):
            x_corr = x_trial + corr
            d_corr = model.eval_all(x_corr)
            f_corr = float(d_corr["f"])
            cE_c, cI_c = d_corr["cE"], d_corr["cI"]
            thE = float(np.sum(np.abs(cE_c))) if (cE_c is not None) else 0.0
            thI = float(np.sum(np.maximum(0.0, np.asarray(cI_c).ravel()))) if (cI_c is not None) else 0.0
            theta_corr = thE + thI
            ok = funnel.is_acceptable(theta0, f0, theta_corr, f_corr, pred_df, pred_dtheta) if funnel else \
                 filter.is_acceptable(theta_corr, f_corr)
            if not ok:
                logging.debug("SOC: corrected point rejected by Funnel/Filter.")
                needs_restoration = True

        # Ensure vector length n
        if corr.shape[0] != n:
            z = np.zeros(n); z[:corr.shape[0]] = corr
            corr = z

        return corr, needs_restoration

    def reset(self):
        self._cached_eval = None
        self._cached_x_trial = None
