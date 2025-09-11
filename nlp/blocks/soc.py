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
        Delta: Optional[float] = None,  # <-- TR radius now optional
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
        Compute a feasibility-improving SOC correction at `x_trial`.

        If `Delta` is None or <= 0, no TR-based cap is applied. You may set
        `cfg.soc_cap_abs > 0` to enforce an absolute cap (scaled by 1/(1+θ0/soc_theta_scale)).
        """
        if not getattr(self.cfg, "use_soc", False):
            return np.zeros(model.n), False

        if self._cached_x_trial is None or not np.allclose(
            self._cached_x_trial, x_trial
        ):
            d = model.eval_all(x_trial)
            self._cached_eval = d
            self._cached_x_trial = x_trial.copy()
        else:
            d = self._cached_eval

        cE, cI, JE, JI = d["cE"], d["cI"], d["JE"], d["JI"]

        def _bad(arr):
            return (arr is not None) and (np.any(~np.isfinite(arr)))

        if _bad(cE) or _bad(cI):
            logging.debug("SOC: invalid (NaN/Inf) constraint values at x_trial")
            return np.zeros(model.n), True

        rows, rhs = [], []

        if cE is not None and JE is not None and JE.size:
            maskE = np.abs(cE) > getattr(self.cfg, "soc_tol", 1e-8)
            if np.any(maskE):
                rows.append(JE[maskE])
                rhs.append(-cE[maskE])

        if cI is not None and JI is not None and JI.size:
            maskI = cI > getattr(self.cfg, "soc_violation", 0.0)
            if np.any(maskI):
                rows.append(JI[maskI])
                rhs.append(-cI[maskI])

        if not rows:
            logging.debug(
                "SOC: nothing to correct (no violated constraints beyond thresholds)"
            )
            return np.zeros(model.n), False

        J = np.vstack(rows)
        r = np.concatenate(rhs)

        corr = _safe_solve(J, r, reg=getattr(self.cfg, "soc_reg", 1e-8))

        # ----- CAP LOGIC (TR optional) -----
        nrm = float(np.linalg.norm(corr))
        theta_scale = max(getattr(self.cfg, "soc_theta_scale", 1.0), 1e-16)
        theta0_val = float(theta0 or 0.0)
        min_norm = getattr(self.cfg, "soc_min_norm", 1e-16)

        if Delta is not None and Delta > 0.0:
            # Trust-region-based cap
            kappa = getattr(self.cfg, "soc_kappa", 0.9)
            cap = float(kappa) * float(Delta) / (1.0 + theta0_val / theta_scale)
        else:
            # No TR: use absolute cap if configured; otherwise no cap
            cap_abs = float(getattr(self.cfg, "soc_cap_abs", 0.0))
            if cap_abs > 0.0:
                cap = cap_abs / (1.0 + theta0_val / theta_scale)
            else:
                cap = np.inf  # no cap

        if np.isfinite(cap) and nrm > max(cap, min_norm):
            corr *= cap / max(nrm, 1e-16)
            logging.debug(
                f"SOC: scaled correction ‖Δx‖ {nrm:.2e} → {cap:.2e} "
                f"(Delta={'None' if Delta is None else f'{Delta:.3e}'})"
            )

        # ----- Acceptability checks (optional) -----
        needs_restoration = False
        if (funnel or filter) and (theta0 is not None) and (f0 is not None):
            x_corr = x_trial + corr
            d_corr = model.eval_all(x_corr)
            f_corr = float(d_corr["f"])

            cE_c, cI_c, JE_c, JI_c = (
                d_corr["cE"],
                d_corr["cI"],
                d_corr["JE"],
                d_corr["JI"],
            )

            if s is not None and mu > 0 and (cI_c is not None):
                inc_I = JI_c @ corr if (JI_c is not None and JI_c.size) else 0.0
                # keep cI + s ≈ 0 around x_trial
                s_tilde = s - (cI + inc_I)
                if np.any(s_tilde <= 0.0):
                    logging.debug(
                        "SOC: corrected slacks would be nonpositive; requesting restoration"
                    )
                    return corr, True
                rI_corr = cI_c + s_tilde
            else:
                rI_corr = cI_c

            thE = float(np.sum(np.abs(cE_c))) if (cE_c is not None) else 0.0
            thI = float(np.sum(np.abs(rI_corr))) if (rI_corr is not None) else 0.0
            theta_corr = thE + thI

            if funnel is not None:
                ok = funnel.is_acceptable(
                    theta0, f0, theta_corr, f_corr, pred_df, pred_dtheta
                )
            else:
                ok = filter.is_acceptable(theta_corr, f_corr)

            if not ok:
                logging.debug(
                    "SOC: corrected point rejected by acceptor (Funnel/Filter)"
                )
                needs_restoration = True

        return corr, needs_restoration

    def reset(self):
        self._cached_eval = None
        self._cached_x_trial = None
