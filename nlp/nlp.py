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
from .sqp import *


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
        lb: np.ndarray = None,
        ub: np.ndarray = None,
        x0: np.ndarray = None,
        config: SQPConfig = None,
    ):
        self.cfg = config if config is not None else SQPConfig()
        self.x = np.asarray(x0, float)
        self.n = len(self.x)

        # ---- ensure defaults for auto/hybrid ----
        self._ensure_auto_defaults(self.cfg)

        # Model & shared managers
        self.model = Model(f, c_ineq, c_eq, self.n, lb, ub)
        self.hess = HessianManager(self.n, self.cfg)
        self.rest = RestorationManager(self.cfg)
        self.regularizer = Regularizer(self.cfg)

        # Acceptance managers
        self.filter = Filter(self.cfg) if self.cfg.use_filter else None
        self.funnel = Funnel(self.cfg) if self.cfg.use_funnel else None

        # Step infrastructure
        self.tr = TrustRegionManager(self.cfg) if self.cfg.use_trust_region else None
        self.ls = (
            LineSearcher(self.cfg, self.filter, self.funnel) if self.cfg.use_line_search else None
        )
        self.qp = QPSolver(self.cfg, Regularizer(self.cfg))
        self.soc = SOCCorrector(self.cfg)

        # SQP stepper
        self.sqp_stepper = SQPStepper(
            self.cfg,
            self.hess,
            TrustRegionManager(self.cfg),
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
            soc=self.soc,
        )
        mI = len(c_ineq) if c_ineq else 0
        mE = len(c_eq) if c_eq else 0
        self.dfo_state = DFOExactState(self.n, mI, mE, self.cfg)
        self.dfo_stepper = DFOExactPenaltyStepper(
            self.cfg,
            regularizer=self.regularizer,
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
