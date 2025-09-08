import logging
from collections import deque
from typing import Optional, Tuple

import numpy as np

from .aux import *


class LineSearcher:
    """Line search for SQP/IP with Armijo conditions and Funnel/Filter support.

    - `search(...)`  : SQP-only (x-space). Uses f for Armijo (optionally nonmonotone).
    - `search_ip(...)`: IP-aware (x- and s-steps). Uses φ=f-μ∑log s for Armijo,
                        and (θ, f) for Funnel/Filter acceptability.
    """

    def __init__(
        self,
        cfg: "SQPConfig",
        flt: Optional["Filter"] = None,
        funnel: Optional["Funnel"] = None,
    ):
        self.cfg = cfg
        self.filter = flt
        self.funnel = funnel
        self.f_hist = (
            deque(maxlen=getattr(cfg, "ls_nonmonotone_M", 0))
            if getattr(cfg, "use_nonmonotone_ls", False)
            else None
        )
        if flt is not None and funnel is not None:
            raise ValueError("Cannot use both Filter and Funnel simultaneously")

        # basic param guards
        self.cfg.ls_backtrack = float(
            max(1e-4, min(0.99, getattr(self.cfg, "ls_backtrack", 0.5)))
        )
        self.cfg.ls_armijo_f = float(max(1e-12, getattr(self.cfg, "ls_armijo_f", 1e-4)))
        self.cfg.ls_max_iter = int(max(1, getattr(self.cfg, "ls_max_iter", 20)))
        self.cfg.ls_min_alpha = float(
            max(0.0, getattr(self.cfg, "ls_min_alpha", 1e-12))
        )
        self.cfg.ls_use_wolfe = bool(getattr(self.cfg, "ls_use_wolfe", False))
        self.cfg.ls_wolfe_c = float(
            max(self.cfg.ls_armijo_f, min(0.999, getattr(self.cfg, "ls_wolfe_c", 0.9)))
        )

    # --------------------------- SQP (x-only) ---------------------------
    def search_sqp(
        self,
        model: "Model",
        x: np.ndarray,
        p: np.ndarray,
        s: Optional[
            np.ndarray
        ] = None,  # ignored in SQP; kept for signature compatibility
        mu: float = 0.0,  # ignored in SQP
        theta0: float = None,  # optional, used for Funnel/Filter checks
    ) -> Tuple[float, int, bool]:
        """
        SQP line search (x-space only).
        Armijo on f (nonmonotone if enabled), optional Wolfe curvature,
        acceptability via Funnel/Filter on (θ, f).
        """
        # Evaluate at base
        d0 = model.eval_all(x)
        f0 = float(d0["f"])
        g = d0["g"]
        # dimensionality guard
        if p.shape != x.shape:
            raise ValueError(
                "SQP LineSearcher.search expects p to have same shape as x (no slack segment)."
            )
        gTp = float(np.dot(g, p))

        # descent test (allow tiny violations)
        if gTp >= -1e-12:
            logging.debug(f"Non-descent direction: gTp={gTp:.2e}")
            return 0.0, 0, True

        # nonmonotone reference
        if self.f_hist is not None:
            self.f_hist.append(f0)
            f_ref = max(self.f_hist)
        else:
            f_ref = f0

        # predicted reductions for Funnel (unit step)
        pred_df = max(0.0, -gTp)
        theta0 = theta0 if theta0 is not None else model.constraint_violation(x)
        theta_trial = None

        alpha = 1.0
        min_alpha = self.cfg.ls_min_alpha * (1.0 + np.linalg.norm(x))

        for it in range(self.cfg.ls_max_iter):
            x_trial = x + alpha * p
            d_trial = model.eval_all(x_trial)
            f_trial = float(d_trial["f"])

            if not np.isfinite(f_trial):
                alpha *= self.cfg.ls_backtrack
                continue

            # Armijo on f
            armijo_ok = f_trial <= f_ref - self.cfg.ls_armijo_f * alpha * gTp

            # Optional Wolfe curvature (on f)
            if self.cfg.ls_use_wolfe:
                g_trial = d_trial["g"]
                gTp_trial = float(np.dot(g_trial, p))
                wolfe_ok = gTp_trial >= self.cfg.ls_wolfe_c * gTp
            else:
                wolfe_ok = True

            # Acceptability via Funnel/Filter on (θ, f)
            theta_trial = model.constraint_violation(x_trial)
            acceptable_ok = True
            if self.funnel is not None:
                # For SQP, the funnel should reason about f, not φ
                pred_dtheta = max(0.0, theta0 - model.constraint_violation(x + p))
                acceptable_ok = self.funnel.is_acceptable(
                    theta0, f0, theta_trial, f_trial, pred_df, pred_dtheta
                )
            elif self.filter is not None:
                acceptable_ok = self.filter.is_acceptable(theta_trial, f_trial)

            if armijo_ok and wolfe_ok and acceptable_ok:
                if self.f_hist is not None:
                    self.f_hist[-1] = f_trial
                if self.funnel is not None:
                    self.funnel.add_if_acceptable(
                        theta0, f0, theta_trial, f_trial, pred_df, pred_dtheta
                    )
                elif self.filter is not None:
                    self.filter.add_if_acceptable(theta_trial, f_trial)
                return alpha, it, False

            alpha *= self.cfg.ls_backtrack
            if alpha < min_alpha:
                break

        needs_restoration = (
            theta_trial if theta_trial is not None else theta0
        ) > getattr(self.cfg, "ls_theta_restoration", 1e3)
        return 0.0, self.cfg.ls_max_iter, needs_restoration

    # ----------------------------- IP (x,s) ------------------------------
    def search_ip(
        self,
        model: "Model",
        x: np.ndarray,
        dx: np.ndarray,
        ds: np.ndarray,
        s: np.ndarray,
        mu: float,
        theta0: float,
        alpha_max: float = 1.0,
    ) -> Tuple[float, int, bool]:
        """
        Interior-point line search:
        - Armijo on φ(x,s;μ) = f(x) - μ Σ log s
        - Acceptability via Funnel/Filter on (θ, f) (not φ)
        - Respects fraction-to-boundary by capping α ≤ alpha_max
        """
        # Base evaluations
        d0 = model.eval_all(x)
        f0 = float(d0["f"])
        g0 = d0["g"]
        cE0, cI0 = d0["cE"], d0["cI"]

        if s is None or s.size == 0 or np.any(s <= 0.0):
            return 0.0, 0, True

        # φ0 and θ0
        phi0 = float(f0 - mu * np.sum(np.log(np.maximum(s, 1e-16))))
        if theta0 is None:
            rI0 = (cI0 + s) if (cI0 is not None) else None
            thE = float(np.abs(cE0).sum()) if cE0 is not None else 0.0
            thI = float(np.abs(rI0).sum()) if rI0 is not None else 0.0
            theta0 = thE + thI

        # dφ = g·dx - μ Σ ds_i/s_i
        dphi = float(g0 @ dx) - float(mu * np.sum(ds / np.maximum(s, 1e-16)))

        # Predictions for Funnel (unit step)
        pred_df = max(0.0, float(-(g0 @ dx)))

        def theta_lin():
            cE_lin = (
                cE0 + (d0["JE"] @ dx if d0["JE"] is not None else 0.0)
                if cE0 is not None
                else None
            )
            inc_I = d0["JI"] @ dx if d0["JI"] is not None else 0.0
            rI_lin = (cI0 + s + inc_I + ds) if cI0 is not None else None
            thE = float(np.abs(cE_lin).sum()) if cE_lin is not None else 0.0
            thI = float(np.abs(rI_lin).sum()) if rI_lin is not None else 0.0
            return thE + thI

        theta_pred = theta_lin()
        pred_dtheta = max(0.0, theta0 - theta_pred)

        alpha = float(min(1.0, max(0.0, alpha_max)))
        min_alpha = self.cfg.ls_min_alpha * (1.0 + np.linalg.norm(x))

        it = 0
        while it < self.cfg.ls_max_iter:
            alpha = min(alpha, alpha_max)

            x_t = x + alpha * dx
            s_t = s + alpha * ds
            if np.any(s_t <= 0.0):
                alpha *= self.cfg.ls_backtrack
                it += 1
                if alpha < min_alpha:
                    break
                continue

            d_t = model.eval_all(x_t)
            f_t = float(d_t["f"])
            phi_t = float(f_t - mu * np.sum(np.log(np.maximum(s_t, 1e-16))))

            # Armijo on φ
            armijo_ok = phi_t <= phi0 + self.cfg.ls_armijo_f * alpha * dphi

            # Acceptability via Funnel/Filter on (θ,f)
            cE_t, cI_t = d_t["cE"], d_t["cI"]
            rI_t = (cI_t + s_t) if (cI_t is not None) else None
            thE = float(np.abs(cE_t).sum()) if cE_t is not None else 0.0
            thI = float(np.abs(rI_t).sum()) if rI_t is not None else 0.0
            theta_t = thE + thI

            acceptable_ok = True
            if self.funnel is not None:
                acceptable_ok = self.funnel.is_acceptable(
                    theta0, f0, theta_t, f_t, pred_df, pred_dtheta
                )
            elif self.filter is not None:
                acceptable_ok = self.filter.is_acceptable(theta_t, f_t)

            if armijo_ok and acceptable_ok:
                # update nonmonotone history only when φ isn’t used (SQP mode)
                if self.f_hist is not None and mu <= 0.0:
                    self.f_hist.append(f_t)
                if self.funnel is not None:
                    self.funnel.add_if_acceptable(
                        theta0, f0, theta_t, f_t, pred_df, pred_dtheta
                    )
                elif self.filter is not None:
                    self.filter.add_if_acceptable(theta_t, f_t)
                return alpha, it, False

            alpha *= self.cfg.ls_backtrack
            it += 1
            if alpha < min_alpha:
                break

        needs_restoration = theta0 > getattr(self.cfg, "ls_theta_restoration", 1e3)
        return 0.0, it, needs_restoration

    def reset(self):
        """Reset non-monotone history."""
        if self.f_hist is not None:
            self.f_hist.clear()