"""
Filter & Funnel acceptance mechanisms for nonlinear optimization (SQP/IP).

This module implements two complementary acceptance strategies:

1) Fletcher–Leyffer Filter (class `Filter`)
   - Maintains a set of 'non-dominated' pairs (θ, f) (constraint violation, objective).
   - A trial point (θ, f) is acceptable if it is not dominated by any stored pair,
     under adaptive margins in both θ and f.
   - Uses a heap keyed by -θ to quickly reference the worst infeasibility level seen.

2) Scalar Funnel (class `Funnel`)
   - Maintains a single infeasibility bound τ (the 'funnel width').
   - A trial point must satisfy θ_new ≤ τ and pass either:
       (a) an 'f-type' Armijo test on objective decrease, or
       (b) an 'h-type' sufficient infeasibility decrease, with τ updated (contracted)
           only on h-type iterations.

Both mechanisms assume minimization of f and nonnegative infeasibility θ ≥ 0.

Required SQPConfig fields (for `Filter`):
    filter_theta_min : float   # minimum θ used for scaling/decay
    filter_gamma_theta : float # base θ-margin factor (0 < γ_θ ≤ 1)
    filter_gamma_f : float     # base f-margin factor (0 < γ_f ≤ 1)
    filter_margin_min : float  # absolute minimum margin safeguard (> 0)
    iter_scale_factor : float  # controls margin decay with iterations (> 0)
    filter_max_size : int      # max stored pairs in the filter (≥ 1)
    switch_theta : float       # switching condition scale on θ (≥ 0)
    switch_f : float           # switching condition scale on f   (≥ 0)
    tr_delta0 : float          # reference TR radius for margin scaling (> 0)

Required SQPConfig fields (for `Funnel`):
    funnel_initial_tau : float   # initial funnel width τ_0 (> 0)
    funnel_sigma : float         # Armijo fraction in (0, 1)
    funnel_delta : float         # f-type vs h-type switch threshold coefficient (≥ 0)
    funnel_beta : float          # h-type relaxation factor in (0, 1)
    funnel_kappa : float         # τ update convex-combination weight in (0, 1)
    funnel_min_tau : float       # lower bound on τ (> 0)
    funnel_max_history : int     # cap on stored (θ, f) history (≥ 0)

Notes
-----
- θ ('theta') denotes a nonnegative measure of constraint violation.
- `Filter` stores entries as heap items `(-theta, f)` so the minimum heap key
  corresponds to the largest θ (i.e., 'worst' infeasibility) for fast reference.
- Margins g_θ, g_f adapt to the current θ scale, iteration count, and (optionally)
  the trust-region radius to avoid stalling and over-stringency.
"""

from __future__ import annotations

import heapq
import logging
from heapq import heappop, heappush
from typing import List, Optional, Set, Tuple

import numpy as np


class Filter:
    """
    Fletcher–Leyffer-style filter for SQP/IP globalization.
    The filter stores pairs (θ, f) and rejects trial points that lie in the
    forbidden region induced by any stored pair, with adaptive margins on both
    infeasibility and objective. This yields a robust globalization strategy that
    allows progress on either feasibility or optimality without line-search
    parameters that tightly couple them.

    Parameters
    ----------
    cfg : SQPConfig
        Configuration object providing the attributes documented in the module
        docstring under "Required SQPConfig fields (for Filter)".

    Attributes
    ----------
    entries : List[Tuple[float, float]]
        Min-heap storing (-θ_i, f_i) pairs. Negative θ is used so the smallest
        heap key corresponds to the largest θ (worst infeasibility).
    iter : int
        Number of accepted insertions. Used to adapt margins over iterations.
    initial_theta : Optional[float]
        θ scale captured from the first acceptable point (≥ 1e-8).
    initial_f : Optional[float]
        |f| scale captured from the first acceptable point (≥ 1e-8).
    """

    def __init__(self, cfg: 'SQPConfig'):
        self.cfg = cfg
        self.entries = [(-cfg.filter_theta_min * 10, -np.inf)]  # Initial pair (θ_max, -∞) per IPOPT (21)
        self.iter = 0
        self.initial_theta = None
        self.initial_f = None
        # -- basic config validation
        required_attrs = [
            'filter_theta_min', 'filter_gamma_theta', 'filter_gamma_f',
            'filter_margin_min', 'iter_scale_factor', 'filter_max_size',
            'switch_theta', 'switch_f', 'tr_delta0'
        ]
        for attr in required_attrs:
            if not hasattr(cfg, attr):
                raise ValueError(f"Missing required config attribute: {attr}")
        if cfg.filter_max_size < 1:
            raise ValueError(f"filter_max_size must be positive, got {cfg.filter_max_size}")
        if cfg.filter_theta_min <= 0:
            raise ValueError(f"filter_theta_min must be positive, got {cfg.filter_theta_min}")
        if cfg.tr_delta0 <= 0:
            raise ValueError(f"tr_delta0 must be positive, got {cfg.tr_delta0}")

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _margins(self, trust_radius: Optional[float] = None) -> Tuple[float, float]:
        """
        Compute adaptive margins (g_θ, g_f) for infeasibility and objective tests.
        The margins decay with the current θ scale (via θ_max and filter_theta_min)
        and with iterations (via iter_scale_factor). Optionally, they scale up with
        the trust-region radius to avoid over-pruning when larger steps are allowed.

        Parameters
        ----------
        trust_radius : float, optional
            Current trust-region radius Δ. If provided, margins are scaled by
            max(1, Δ / tr_delta0).

        Returns
        -------
        g_theta, g_f : float
            Positive margins applied in dominance tests (lower-bounded by 1e-8).
        """
        theta_max = -self.entries[0][0] if self.entries else 1.0  # Largest θ from heap top
        # Decay in [0.1, 1.0] based on θ scale
        decay = min(1.0, max(0.1, theta_max / self.cfg.filter_theta_min))
        # Iteration-based decay factor
        iter_scale = 1.0 / (1.0 + self.iter / self.cfg.iter_scale_factor)
        g_theta = max(self.cfg.filter_margin_min, self.cfg.filter_gamma_theta * decay * iter_scale)
        g_f = max(self.cfg.filter_margin_min, self.cfg.filter_gamma_f * decay * iter_scale)
        if trust_radius is not None:
            scale = max(1.0, float(trust_radius) / self.cfg.tr_delta0)
            g_theta *= scale
            g_f *= scale
        # Numerical floor
        g_theta = max(g_theta, 1e-8)
        g_f = max(g_f, 1e-8)
        logging.debug(f"[Filter] margins: gθ={g_theta:.3e}, gƒ={g_f:.3e}, θ_max={theta_max:.3e}, iter={self.iter}")
        return g_theta, g_f

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def is_acceptable(self, theta: float, f: float, trust_radius: Optional[float] = None) -> bool:
        """
        Check if (θ, f) is acceptable w.r.t. the current filter.
        A point is rejected if it falls inside the forbidden region of any stored
        pair (θ_i, f_i), after applying adaptive margins and a switching rule that
        can still allow acceptance when there is clear progress predominantly in
        one metric.

        Parameters
        ----------
        theta : float
            Nonnegative constraint violation (θ ≥ 0).
        f : float
            Objective value (to be minimized).
        trust_radius : float, optional
            Trust-region radius for context-aware margins.

        Returns
        -------
        bool
            True if acceptable, False otherwise.
        """
        if not (np.isfinite(theta) and np.isfinite(f)):
            logging.warning(f"[Filter] invalid point: θ={theta}, f={f}")
            return False
        if theta < 0:
            logging.warning(f"[Filter] negative θ: {theta}")
            return False
        gθ, gƒ = self._margins(trust_radius)
        # Scale tolerances based on the first point
        theta_scale = self.initial_theta if self.initial_theta is not None else max(theta, 1.0)
        f_scale = self.initial_f if self.initial_f is not None else max(abs(f), 1.0)
        epsilon = 1e-8 * max(theta_scale, f_scale)
        # Forbidden region test with switching condition
        for t_i_heap, f_i in self.entries:
            t_i = -t_i_heap  # Convert back from heap key
            in_forbidden = (theta >= (1.0 - gθ) * t_i - epsilon) and (f >= f_i - gƒ * theta - epsilon)
            if in_forbidden:
                # Switching condition: accept if clear progress in one metric
                switch_theta = self.cfg.switch_theta * theta_scale
                switch_f = self.cfg.switch_f * f_scale
                if theta < switch_theta or f < f_i - switch_f:  # Simplified switching
                    logging.debug(f"[Filter] accept via switching: (θ={theta:.3e}, f={f:.3e}) vs (θ={t_i:.3e}, f={f_i:.3e})")
                    continue
                logging.debug(f"[Filter] reject (θ={theta:.3e}, f={f:.3e}) dominated by (θ={t_i:.3e}, f={f_i:.3e})")
                return False
        return True

    def add_if_acceptable(self, theta: float, f: float, trust_radius: Optional[float] = None) -> bool:
        """
        Try to insert (θ, f) into the filter, removing dominated entries.
        If acceptable, the new point is inserted and any existing entries that are
        dominated by (θ, f) under the current margins are discarded. If the filter
        is full, the worst entry is removed first (largest θ, then larger f tie-break).

        Parameters
        ----------
        theta : float
            Nonnegative constraint violation.
        f : float
            Objective value.
        trust_radius : float, optional
            Trust-region radius for context-aware margins.

        Returns
        -------
        bool
            True if the point was accepted and inserted, False otherwise.
        """
        if not (np.isfinite(theta) and np.isfinite(f)):
            logging.warning(f"[Filter] invalid point: θ={theta}, f={f}")
            return False
        if theta < 0:
            logging.warning(f"[Filter] negative θ: {theta}")
            return False
        # Initialize scales from first accepted point
        if self.initial_theta is None:
            self.initial_theta = max(theta, 1e-8)
        if self.initial_f is None:
            self.initial_f = max(abs(f), 1e-8)
        gθ, gƒ = self._margins(trust_radius)
        epsilon = 1e-8 * max(self.initial_theta, self.initial_f)
        # Check acceptability and build new heap
        temp_entries = []
        acceptable = True
        for t_i_heap, f_i in self.entries:
            t_i = -t_i_heap
            if (theta < (1.0 - gθ) * t_i + epsilon) or (f < f_i - gƒ * theta + epsilon):
                heapq.heappush(temp_entries, (t_i_heap, f_i))
            else:
                acceptable = False
                break
        if acceptable:
            heapq.heappush(temp_entries, (-theta, f))
            if len(temp_entries) > self.cfg.filter_max_size:
                heapq.heappop(temp_entries)  # Remove worst (largest θ)
            self.entries = temp_entries
            self.iter += 1
            logging.debug(f"[Filter] accept (θ={theta:.3e}, f={f:.3e}); size={len(self.entries)}")
            return True
        return False

    def reset(self) -> None:
        """Clear all entries, reset counters and initial scales."""
        self.entries = [(-self.cfg.filter_theta_min * 10, -np.inf)]  # Reinitialize with θ_max
        self.iter = 0
        self.initial_theta = None
        self.initial_f = None
        logging.info("[Filter] reset")
    
class Funnel:
    """
    Scalar funnel mechanism for globalization with separate f- and h-type steps.

    A trial point (θ_new, f_new) is acceptable if:
      (i)  θ_new ≤ τ  (must stay within the funnel), and
      (ii) it passes either:
           - f-type: Armijo decrease in objective:   Δf_actual ≥ σ Δf_pred
           - h-type: sufficient infeasibility decrease relative to current τ:
                     Δθ_actual ≥ σ Δθ_pred  and  θ_new ≤ β τ
    On h-type acceptance, the funnel width τ is updated by a convex combination
    between θ_new and the previous τ to contract the funnel.

    Parameters
    ----------
    cfg : SQPConfig
        Configuration object providing the attributes documented in the module
        docstring under "Required SQPConfig fields (for Funnel)".

    Attributes
    ----------
    tau : float
        Current funnel width (infeasibility bound).
    iter : int
        Count of accepted points (for diagnostics only).
    history : List[Tuple[float, float]]
        Optional bounded history of accepted (θ, f) for debugging/plots.
    """

    def __init__(self, cfg: 'SQPConfig'):
        self.cfg = cfg
        self.tau: float = float(self.cfg.funnel_initial_tau)
        self.iter: int = 0
        self.history: List[Tuple[float, float]] = []  # (theta, f)

    # --------------- internals --------------- #
    def _armijo_f(self, actual_df: float, predicted_df: float) -> bool:
        """Armijo condition for objective decrease: actual ≥ σ * predicted."""
        return actual_df >= self.cfg.funnel_sigma * predicted_df

    def _armijo_theta(self, actual_dtheta: float, predicted_dtheta: float) -> bool:
        """Armijo condition for infeasibility decrease: actual ≥ σ * predicted."""
        return actual_dtheta >= self.cfg.funnel_sigma * predicted_dtheta

    # --------------- public API --------------- #
    def is_acceptable(
        self,
        current_theta: float, current_f: float,
        new_theta: float, new_f: float,
        predicted_df: float, predicted_dtheta: float
    ) -> bool:
        """
        Check funnel acceptability of a trial point.

        Parameters
        ----------
        current_theta, current_f : float
            Metrics at the incumbent point (θ_k, f_k).
        new_theta, new_f : float
            Metrics at the trial point (θ_{k+1}, f_{k+1}).
        predicted_df : float
            Model-predicted decrease in objective (≥ 0 meaning 'expected improvement').
        predicted_dtheta : float
            Model-predicted decrease in infeasibility (≥ 0).

        Returns
        -------
        bool
            True if acceptable, False otherwise.
        """
        if new_theta < 0 or not np.isfinite(new_f) or current_theta < 0:
            return False

        eps = 1e-10
        actual_df = current_f - new_f          # positive if objective decreased
        actual_dtheta = current_theta - new_theta  # positive if infeasibility decreased

        # 1) Must be inside the funnel
        if new_theta > self.tau + eps:
            return False

        # 2) Switching: decide f-type vs h-type
        #    If predicted objective decrease is 'large enough' relative to θ^2,
        #    prefer f-type; otherwise do h-type checks.
        if predicted_df >= self.cfg.funnel_delta * (current_theta ** 2) - eps:
            # f-type: Armijo on objective
            return self._armijo_f(actual_df, predicted_df)
        else:
            # h-type: sufficient feasibility progress and close enough to funnel center
            if new_theta <= self.cfg.funnel_beta * self.tau + eps:
                return self._armijo_theta(actual_dtheta, predicted_dtheta)
            return False

    def add_if_acceptable(
        self,
        current_theta: float, current_f: float,
        new_theta: float, new_f: float,
        predicted_df: float, predicted_dtheta: float
    ) -> bool:
        """
        Accept a trial point if it meets funnel criteria; update τ on h-type.

        Parameters
        ----------
        (see `is_acceptable`)

        Returns
        -------
        bool
            True if accepted (and τ possibly updated), False otherwise.
        """
        if not self.is_acceptable(current_theta, current_f, new_theta, new_f, predicted_df, predicted_dtheta):
            logging.debug(f"[Funnel] reject (θ={new_theta:.3e}, f={new_f:.3e})")
            return False

        eps = 1e-10
        is_f_type = predicted_df >= self.cfg.funnel_delta * (current_theta ** 2) - eps

        # Optional bounded history for diagnostics
        self.history.append((new_theta, new_f))
        if len(self.history) > self.cfg.funnel_max_history:
            self.history.pop(0)

        # τ update on h-type only (contract toward θ_new)
        if not is_f_type:
            self.tau = max(
                self.cfg.funnel_min_tau,
                (1 - self.cfg.funnel_kappa) * new_theta + self.cfg.funnel_kappa * self.tau
            )
            logging.debug(f"[Funnel] τ updated to {self.tau:.3e}")

        self.iter += 1
        logging.debug(f"[Funnel] accept (θ={new_theta:.3e}, f={new_f:.3e}); "
                      f"type={'f' if is_f_type else 'h'}, τ={self.tau:.3e}")
        return True

    def reset(self) -> None:
        """Reset τ to its initial value and clear history/counters."""
        self.tau = float(self.cfg.funnel_initial_tau)
        self.history.clear()
        self.iter = 0
        logging.info("[Funnel] reset")
