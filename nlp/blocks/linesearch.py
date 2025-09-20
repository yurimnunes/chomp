import logging
import time
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

    # ----------------------------- IP (x,s) ------------------------------
    def search_ip(
        self, model: "Model", x: np.ndarray, dx: np.ndarray, ds: np.ndarray, s: np.ndarray,
        mu: float, d_phi: float, theta0: float, alpha_max: float = 1.0,
    ) -> Tuple[float, int, bool]:
        """
        Interior-point line search with fraction-to-boundary:
        - Armijo on φ(x,s;μ) = f(x) - μ Σ log s
        - Acceptability via Funnel/Filter on (θ, f)
        - Ensures s + α ds > 0 (fraction-to-boundary)
        """
        # --- local bindings (avoid repeated attribute lookups)
        cfg = self.cfg
        ls_backtrack = cfg.ls_backtrack
        ls_armijo_f = cfg.ls_armijo_f
        ls_max_iter = cfg.ls_max_iter
        ls_min_alpha = cfg.ls_min_alpha
        funnel = self.funnel
        flt = self.filter
        f_hist = self.f_hist

        # --- base eval (single call; request only needed pieces)
        d0 = model.eval_all(x, components=("f", "g", "cE", "cI", "JE", "JI"))
        f0 = float(d0["f"])
        g0 = d0["g"]
        cE0, cI0 = d0["cE"], d0["cI"]
        JE, JI = d0["JE"], d0["JI"]

        # quick slack checks
        if s is None or s.size == 0 or np.any(s <= 0.0):
            # invalid slack state at base point
            return 1.0, 0, True

        # barrier epsilon (reuse)
        barrier_eps = max(1e-8 * mu, 1e-16)

        # φ0 and θ0
        # (sum of logs: use maximum to avoid log(≤0) when very small)
        phi0 = float(f0 - mu * np.sum(np.log(np.maximum(s, barrier_eps))))

        if theta0 is None:
            # θ = ||cE||_1 + ||cI + s||_1
            thE = float(np.add.reduce(np.abs(cE0))) if cE0 is not None else 0.0
            if cI0 is not None:
                rI0 = cI0 + s
                thI = float(np.add.reduce(np.abs(rI0)))
            else:
                thI = 0.0
            theta0 = thE + thI

        # descent check on φ
        if d_phi >= -1e-12:
            # non-descent direction for barrier
            return 1.0, 0, True

        # --- fraction-to-boundary α_max
        tau_ftb = getattr(cfg, "ip_fraction_to_boundary_tau", 0.995)
        if ds is not None:
            neg = ds < 0.0
            if np.any(neg):
                # α ≤ (1 - τ) s_i / -ds_i
                alpha_bounds = (1.0 - tau_ftb) * s[neg] / (-ds[neg])
                # guard in case alpha_max was passed < computed bound
                am = float(alpha_bounds.min())
                if am < alpha_max:
                    alpha_max = am
        alpha_max = 1.0e-16 if alpha_max < 1.0e-16 else alpha_max

        # --- Funnel predictions at unit step (precompute linearizations once)
        # pred_df = max(0, -(g^T dx))
        pred_df = float(-(g0 @ dx))
        if pred_df < 0.0:
            pred_df = 0.0

        # θ linear prediction: θ_lin = ||cE0 + JE dx||_1 + ||(cI0 + s) + (JI dx + ds)||_1
        je_dx = JE @ dx if JE is not None else None
        ji_dx = JI @ dx if JI is not None else None

        thE_lin = 0.0
        if cE0 is not None:
            if je_dx is None:
                # rare: no JE but cE0 present -> just ||cE0||_1
                thE_lin = float(np.add.reduce(np.abs(cE0)))
            else:
                thE_lin = float(np.add.reduce(np.abs(cE0 + je_dx)))

        if cI0 is not None:
            # rI_lin = (cI0 + s) + (ji_dx + ds or ds alone)
            if ji_dx is None:
                rI_lin = cI0 + s + ds
            else:
                rI_lin = cI0 + s + ji_dx + ds
            thI_lin = float(np.add.reduce(np.abs(rI_lin)))
        else:
            thI_lin = 0.0

        theta_pred = thE_lin + thI_lin
        pred_dtheta = theta0 - theta_pred
        if pred_dtheta < 0.0:
            pred_dtheta = 0.0

        # --- line search loop
        alpha = 1.0 if alpha_max > 1.0 else float(alpha_max)
        it = 0

        while it < ls_max_iter:
            x_t = x + alpha * dx
            s_t = s + alpha * ds

            # quick positivity gate before log/allocs
            if np.any(s_t <= 0.0):
                alpha *= ls_backtrack
                it += 1
                continue

            try:
                d_t = model.eval_all(x_t, components=("f", "cE", "cI"))
                f_t = float(d_t["f"])
                if not np.isfinite(f_t):
                    alpha *= ls_backtrack; it += 1; continue

                # φ(x_t, s_t; μ)
                phi_t = float(f_t - mu * np.sum(np.log(np.maximum(s_t, barrier_eps))))
                if not np.isfinite(phi_t):
                    alpha *= ls_backtrack; it += 1; continue
            except Exception as e:
                # robust backtrack on any evaluation failure
                alpha *= ls_backtrack
                it += 1
                continue

            # Armijo on φ
            if phi_t <= phi0 + ls_armijo_f * alpha * d_phi:
                # Acceptability via Funnel/Filter on (θ, f)
                cE_t, cI_t = d_t["cE"], d_t["cI"]

                thE_t = float(np.add.reduce(np.abs(cE_t))) if cE_t is not None else 0.0
                if cI_t is not None:
                    rI_t = cI_t + s_t
                    thI_t = float(np.add.reduce(np.abs(rI_t)))
                else:
                    thI_t = 0.0
                theta_t = thE_t + thI_t

                acceptable_ok = True
                if funnel is not None:
                    acceptable_ok = funnel.is_acceptable(theta0, f0, theta_t, f_t, pred_df, pred_dtheta)
                elif flt is not None:
                    acceptable_ok = flt.is_acceptable(theta_t, f_t)

                if acceptable_ok:
                    if f_hist is not None and mu <= 0.0:
                        f_hist.append(f_t)
                    if funnel is not None:
                        funnel.add_if_acceptable(theta0, f0, theta_t, f_t, pred_df, pred_dtheta)
                    elif flt is not None:
                        flt.add_if_acceptable(theta_t, f_t)
                    return alpha, it, False

            alpha *= ls_backtrack
            it += 1

        needs_restoration = theta0 > getattr(cfg, "ls_theta_restoration", 1e3)
        # Optional: keep debug logging only (avoid prints in hot code)
        if alpha < ls_min_alpha * (1.0 + np.linalg.norm(x)):
            logging.debug("Line search failed: step size below minimum "
                        f"(alpha={alpha:.2e}) after {it} iters; needs_restoration={needs_restoration}")
        else:
            logging.debug("Line search failed: max iterations reached "
                        f"(iters={it}); needs_restoration={needs_restoration}")
        return 1.0, it, needs_restoration
