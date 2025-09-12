import logging
from typing import Optional, Tuple

import numpy as np

from .aux import *


def _safe_solve(J: np.ndarray, r: np.ndarray, reg: float = 1e-8) -> np.ndarray:
    try:
        m, n = J.shape
        if m >= n:
            JtJ = J.T @ J
            lam = reg * (np.trace(JtJ) / max(1.0, n))
            return np.linalg.solve(JtJ + lam * np.eye(n), J.T @ r)
        else:
            JJt = J @ J.T
            lam = reg * (np.trace(JJt) / max(1.0, m))
            y = np.linalg.solve(JJt + lam * np.eye(m), r)
            return J.T @ y
    except np.linalg.LinAlgError:
        logging.debug(
            "SOC: linear solve failed (singular/ill-conditioned), returning zero correction"
        )
        return np.zeros(J.shape[1])


class SOCCorrector:
    def __init__(self, cfg: "SQPConfig"):
        self.cfg = cfg
        self._cached_eval: Optional[dict] = None
        self._cached_x_trial: Optional[np.ndarray] = None

    def compute_correction(
        self,
        model: "Model",
        x_trial: np.ndarray,
        Delta: Optional[float] = None,
        s: Optional[np.ndarray] = None,
        mu: float = 0.0,
        funnel: Optional["Funnel"] = None,
        filter: Optional["Filter"] = None,
        theta0: Optional[float] = None,
        f0: Optional[float] = None,
        pred_df: float = 0.0,
        pred_dtheta: float = 0.0,
    ) -> Tuple[np.ndarray, bool]:
        """
        Feasibility-oriented SOC at x_trial:
        solve   J(x_trial) * corr = -c(x_trial)   (violated rows only)
        with Tikhonov regularization and optional TR/absolute capping.
        Returns (corr, needs_restoration).
        """
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        n = model.n
        if not getattr(self.cfg, "use_soc", False):
            return np.zeros(n), False

        # Cache evals
        if self._cached_x_trial is None or not np.allclose(self._cached_x_trial, x_trial):
            d = model.eval_all(x_trial)
            self._cached_eval = d
            self._cached_x_trial = x_trial.copy()
        else:
            d = self._cached_eval

        cE, cI, JE, JI = d["cE"], d["cI"], d["JE"], d["JI"]

        def _bad(arr):
            return (arr is not None) and (np.any(~np.isfinite(np.asarray(arr))))

        if _bad(cE) or _bad(cI):
            logging.debug("SOC: invalid constraint values at x_trial")
            return np.zeros(n), True

        # Row selection
        rows = []
        rhs = []

        soc_tol = float(getattr(self.cfg, "soc_tol", 1e-8))
        soc_violation = float(getattr(self.cfg, "soc_violation", 0.0))

        # Equalities: |cE| > tol
        if cE is not None and JE is not None and JE.size:
            maskE = np.abs(np.asarray(cE).ravel()) > soc_tol
            if np.any(maskE):
                if sp.issparse(JE):
                    rows.append(JE[maskE])
                else:
                    rows.append(np.asarray(JE)[maskE])
                rhs.append(-np.asarray(cE).ravel()[maskE])

        # Inequalities: cI > 0 (or configured threshold)
        if cI is not None and JI is not None and JI.size:
            maskI = np.asarray(cI).ravel() > soc_violation
            if np.any(maskI):
                if sp.issparse(JI):
                    rows.append(JI[maskI])
                else:
                    rows.append(np.asarray(JI)[maskI])
                rhs.append(-np.asarray(cI).ravel()[maskI])

        if not rows:
            logging.debug("SOC: no rows selected (nothing to correct).")
            return np.zeros(n), False

        # Build J (prefer sparse) and r
        if any(sp.issparse(R) for R in rows):
            rows = [R if sp.issparse(R) else sp.csr_matrix(R) for R in rows]
            J = sp.vstack(rows, format="csr")
        else:
            J = np.vstack(rows)
        r = np.concatenate(rhs)

        # Regularized least squares: solve J corr ≈ r
        reg = float(getattr(self.cfg, "soc_reg", 1e-8))
        try:
            if sp.issparse(J):
                # normal equations with sparse ops
                JT = J.transpose().tocsr()
                JtJ = (JT @ J).tocsc()
                lam = reg * (JtJ.diagonal().sum() / max(1, n))
                corr = spla.spsolve(JtJ + lam * sp.eye(n, format="csc"), JT @ r)
            else:
                m, nJ = J.shape
                if m >= nJ:
                    JtJ = J.T @ J
                    lam = reg * (np.trace(JtJ) / max(1.0, nJ))
                    corr = np.linalg.solve(JtJ + lam * np.eye(nJ), J.T @ r)
                else:
                    JJt = J @ J.T
                    lam = reg * (np.trace(JJt) / max(1.0, m))
                    y = np.linalg.solve(JJt + lam * np.eye(m), r)
                    corr = J.T @ y
        except Exception as e:
            logging.debug(f"SOC: solve failed ({e}); returning zero correction")
            corr = np.zeros(J.shape[1] if sp.issparse(J) else J.shape[1])

        # Cap by TR or absolute bound
        nrm = float(np.linalg.norm(corr))
        theta_scale = max(float(getattr(self.cfg, "soc_theta_scale", 1.0)), 1e-16)
        theta0_val = float(theta0 or 0.0)
        min_norm = float(getattr(self.cfg, "soc_min_norm", 1e-16))
        if Delta is not None and Delta > 0.0:
            kappa = float(getattr(self.cfg, "soc_kappa", 0.9))
            cap = kappa * float(Delta) / (1.0 + theta0_val / theta_scale)
        else:
            cap_abs = float(getattr(self.cfg, "soc_cap_abs", 0.0))
            cap = cap_abs / (1.0 + theta0_val / theta_scale) if cap_abs > 0.0 else np.inf

        if np.isfinite(cap) and nrm > max(cap, min_norm):
            corr = (cap / max(nrm, 1e-16)) * corr
            logging.debug(
                f"SOC: scaled correction ‖Δx‖ {nrm:.2e} → {cap:.2e} "
                f"(Delta={'None' if Delta is None else f'{Delta:.3e}'})"
            )

        # Optional acceptability check (filter/funnel). Also ensure slacks stay > 0.
        needs_restoration = False
        if (funnel or filter) and (theta0 is not None) and (f0 is not None):
            x_corr = x_trial + corr
            d_corr = model.eval_all(x_corr)
            f_corr = float(d_corr["f"])
            cE_c, cI_c, JE_c, JI_c = d_corr["cE"], d_corr["cI"], d_corr["JE"], d_corr["JI"]

            # Maintain s > 0 if slacks are in play
            if s is not None and mu > 0.0 and (cI_c is not None):
                inc_I = (JI_c @ corr) if (JI_c is not None and getattr(JI_c, "size", 0)) else 0.0
                s_tilde = s - (np.asarray(cI) + inc_I)  # keep cI + s ≈ 0 near x_trial
                if np.any(s_tilde <= 0.0):
                    logging.debug("SOC: slacks would be nonpositive; request restoration.")
                    return corr, True
                rI_corr = np.asarray(cI_c) + s_tilde
            else:
                rI_corr = cI_c

            thE = float(np.sum(np.abs(cE_c))) if (cE_c is not None) else 0.0
            thI = float(np.sum(np.abs(rI_corr))) if (rI_corr is not None) else 0.0
            theta_corr = thE + thI

            ok = funnel.is_acceptable(theta0, f0, theta_corr, f_corr, pred_df, pred_dtheta) if funnel else \
                filter.is_acceptable(theta_corr, f_corr)

            if not ok:
                logging.debug("SOC: corrected point rejected by Funnel/Filter.")
                needs_restoration = True

        # Always return a vector of length n
        if corr.shape[0] != n:
            # If we solved in a reduced space, pad/project to full n (simple pad with zeros)
            z = np.zeros(n)
            z[:corr.shape[0]] = corr  # assumes variables ordered compatibly
            corr = z

        return corr, needs_restoration

    def reset(self):
        self._cached_eval = None
        self._cached_x_trial = None
