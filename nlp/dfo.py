from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.linalg import norm

# your stack
from .blocks.aux import Model, SQPConfig
from .blocks.reg import Regularizer, make_psd_advanced
from .blocks.tr import TrustRegionManager

# compact DFO core (with RBF dy-derivatives + equality L1 support)
from .dfo_model import DFOConfig, DFOExactPenalty, TRModel


class DFOExactState:
    """Exact-penalty state (kept small)."""
    def __init__(self, n: int, m_ineq: int, m_eq: int, cfg: SQPConfig):
        self.n = n
        self.mI = m_ineq
        self.mE = m_eq
        self.cfg = cfg
        self.mu = float(getattr(cfg, 'dfo_penalty_mu0', 10.0))
        self.eps_active = float(getattr(cfg, 'dfo_penalty_eps_active', 1e-6))
        self.radius = float(getattr(cfg, 'dfo_al_tr0', 0.1))
        self.lam = np.ones(self.mI) * getattr(cfg, 'dfo_al_lam0', 0.1) if self.mI > 0 else np.zeros(0)
        self.nu = np.zeros(self.mE)
        self.penalty_updates = 0
        self.violation_history: List[float] = []

    def phi(self, f: float, cI: Optional[np.ndarray], cE: Optional[np.ndarray]) -> float:
        cI_arr = np.asarray(cI) if cI is not None else np.zeros(self.mI)
        cE_arr = np.asarray(cE) if cE is not None else np.zeros(self.mE)
        return float(f + self.mu * (np.maximum(0.0, cI_arr).sum() + np.abs(cE_arr).sum()))

    def update_multipliers(self, cI: Optional[np.ndarray], cE: Optional[np.ndarray]):
        max_mult = getattr(self.cfg, 'dfo_al_max_multiplier', 1e6)
        if self.mI > 0 and cI is not None:
            ci = np.asarray(cI).ravel()
            self.lam = np.clip(np.maximum(0.0, self.lam + 0.1 * np.maximum(0.0, ci)), 0.0, max_mult)
        if self.mE > 0 and cE is not None:
            ce = np.asarray(cE).ravel()
            self.nu = np.clip(self.nu + 0.1 * ce, -max_mult, max_mult)

    def adaptive_mu_update(self, viol_current: float, viol_prev: float) -> bool:
        self.violation_history.append(viol_current)
        if len(self.violation_history) > 10:
            self.violation_history.pop(0)
        target = getattr(self.cfg, 'dfo_penalty_target_viol', 1e-6)
        mu_inc = getattr(self.cfg, 'dfo_penalty_mu_inc', 5.0)
        mu_max = getattr(self.cfg, 'dfo_penalty_mu_max', 1e6)
        # If violation stalls above target, bump mu
        if viol_current > target and viol_current > 0.9 * viol_prev:
            old_mu = self.mu
            self.mu = min(mu_max, mu_inc * self.mu)
            if self.mu > old_mu:
                self.penalty_updates += 1
                return True
        return False


def _make_dfo_cfg(cfg: SQPConfig) -> DFOConfig:
    # Keep defaults robust; only override what you need from SQPConfig
    return DFOConfig(
        huber_delta=getattr(cfg, 'dfo_huber_delta', None),
        ridge=float(getattr(cfg, 'dfo_ridge', 1e-6)),
        dist_w_power=float(getattr(cfg, 'dfo_dist_w_power', 0.3)),
        eig_floor=float(getattr(cfg, 'dfo_eig_floor', 1e-8)),
        # max_pts=int(getattr(cfg, 'dfo_max_pts', 60)),
        model_radius_mult=float(getattr(cfg, 'dfo_model_radius_mult', 2.0)),
        use_quadratic_if=int(getattr(cfg, 'dfo_use_quadratic_if', 25)),
        mu=float(getattr(cfg, 'dfo_penalty_mu0', 10.0)),               # will be synced from state.mu each step
        eps_active=float(getattr(cfg, 'dfo_penalty_eps_active', 1e-6)),
        tr_inc=float(getattr(cfg, 'dfo_tr_inc', 1.6)),
        tr_dec=float(getattr(cfg, 'dfo_tr_dec', 0.5)),
        eta0=float(getattr(cfg, 'dfo_eta0', 0.05)),
        eta1=float(getattr(cfg, 'dfo_eta1', 0.25)),
        gp_max_pts=int(getattr(cfg, 'dfo_rbf_max_pts', 100)),
        crit_beta1=float(getattr(cfg, 'dfo_crit_beta1', 1.0)),
        crit_beta2=float(getattr(cfg, 'dfo_crit_beta2', 2.0)),
        min_pts_gp=int(getattr(cfg, 'dfo_rbf_min_pts', 5)),
    )


class DFOExactPenaltyStepper:
    """Plugs SQP loop into the compact DFO core; TRModel geometry managed by the core."""
    def __init__(self, cfg: SQPConfig, tr: Optional[TrustRegionManager], regularizer: Regularizer, n: int, mI: int, mE: int):
        self.cfg = cfg
        self.tr = tr
        self.regularizer = regularizer
        self.n = n
        self.mI = mI
        self.mE = mE
        self.core = DFOExactPenalty(n, mI, mE, _make_dfo_cfg(cfg))
        # Create a TRModel; attach to core so geometry is auto-maintained
        q = 1 + mI + mE
        trmodel = TRModel(self.n, q, _make_dfo_cfg(self.cfg))
        self.core.attach_trmodel(trmodel)  # single attach; the core will maintain geometry
        self._eval_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_hits = 0
        if self.tr is not None:
            self.tr.configure_for_algorithm('dfo')

    # ------------- utilities -------------
    def _ckey(self, x: np.ndarray) -> str:
        return str(hash(np.asarray(x, float).ravel().tobytes()))

    def _eval_cached(self, nlp_model: Model, x: np.ndarray) -> Dict[str, Any]:
        k = self._ckey(x)
        if k not in self._eval_cache:
            self._eval_cache[k] = nlp_model.eval_all(x)
        else:
            self._cache_hits += 1
        return self._eval_cache[k]

    # ------------- public API -------------
    def step(self, nlp_model: Model, x: np.ndarray, state: DFOExactState, it: int):
        import logging
        logging.basicConfig(level=logging.DEBUG)

        # Register oracle
        def make_model_oracle(model: Model, mI: int, mE: int):
            def oracle(x_in: np.ndarray):
                out = model.eval_all(x_in, components=['f', 'cI', 'cE'])
                f = float(out['f'])
                cI = out['cI'] if out['cI'] is not None else np.zeros(mI)
                cE = out['cE'] if out['cE'] is not None else np.zeros(mE)
                return f, cI, cE
            return oracle
        self.core.set_oracle(make_model_oracle(nlp_model, self.mI, self.mE))

        # Sync penalty with core
        self.core.cfg.mu = float(state.mu)
        self.core.cfg.eps_active = float(state.eps_active)

        # Radius (ℓ∞ TR)
        Delta = self.tr.get_radius() if self.tr is not None else state.radius
        tr_min = float(getattr(self.cfg, 'dfo_al_tr_min', 1e-10))
        if self.tr is not None:
            self.tr.set_radius(max(tr_min, Delta))
        else:
            state.radius = max(tr_min, Delta)
        Delta = max(tr_min, Delta)

        # Ensure the core sees the center sample
        d0 = self._eval_cached(nlp_model, x)
        cI0 = np.asarray(d0.get('cI', np.zeros(self.mI))) if self.mI else np.zeros(0)
        cE0 = np.asarray(d0.get('cE', np.zeros(self.mE))) if self.mE else np.zeros(0)
        self.core.add_sample(x, float(d0['f']), cI0, cE0)

        # Criticality loop
        Delta, eps_new, sigma_crit = self.core.criticality_loop(xk=x, eps=state.eps_active, Delta_in=Delta)
        state.eps_active = float(eps_new)

        # Local model & step proposal
        fit = self.core.fit_local(center=x, Delta=Delta)
        h, step_info = self.core.propose_step(x, Delta, fit, cI0=cI0, cE0=cE0)
        step_norm_inf = float(norm(h, ord=np.inf))
        sigma = float(step_info.get('sigma', sigma_crit))
        step_info['sigma'] = sigma
        H_reg, _ = make_psd_advanced(fit.H, self.regularizer, it)

        # Trial evaluation
        x_trial = x + h
        d1 = self._eval_cached(nlp_model, x_trial)
        f0 = float(d0['f'])
        f1 = float(d1['f'])
        cI1 = np.asarray(d1.get('cI', np.zeros(self.mI))) if self.mI else np.zeros(0)
        cE1 = np.asarray(d1.get('cE', np.zeros(self.mE))) if self.mE else np.zeros(0)

        # Exact penalty values
        phi0 = state.phi(f0, cI0, cE0)
        phi1 = state.phi(f1, cI1, cE1)
        act_red = phi0 - phi1

        # Predicted reductions
        pred_red_quad = float(step_info.get('pred_red_quad', max(1e-16, -(fit.g @ h) - 0.5 * (h @ (fit.H @ h)))))
        pred_red_cons = self.core._conservative_pred_red(sigma=float(sigma), Delta=Delta, n=self.n)
        pred_red_tr = max(1e-16, pred_red_cons)

        # Ratio for accept/reject
        rho = float(act_red / max(pred_red_quad, 1e-16))
        eta0 = float(getattr(self.cfg, 'dfo_eta0', self.core.cfg.eta0))
        eta1 = float(getattr(self.cfg, 'dfo_eta1', self.core.cfg.eta1))
        is_FL = bool(step_info.get('is_FL', fit.diag.get('is_FL', True)))

        # Standard DFO accept/reject
        if rho >= eta1:
            accepted = True
            Delta_new = max(Delta, self.core.cfg.tr_inc * step_norm_inf)
        elif rho >= eta0:
            accepted = True
            Delta_new = max(Delta, step_norm_inf) if is_FL else Delta
        else:
            accepted = False
            Delta_new = self.core.cfg.tr_dec * Delta

        # Criticality-aware cap on Delta
        sig = max(float(sigma), 0.0)
        beta1 = float(getattr(self.core.cfg, 'crit_beta1', 1.0))
        beta2 = float(getattr(self.core.cfg, 'crit_beta2', 2.0))
        cap1 = max(1e-12, beta1 * max(sig, 1e-16))
        cap2 = beta2 * max(sig, 1e-16)
        Delta_new = min(Delta_new, cap1, cap2)

        # Feasibility filter
        theta0 = nlp_model.constraint_violation(x)
        theta1 = nlp_model.constraint_violation(x_trial)
        filter_theta_min = float(getattr(self.cfg, 'filter_theta_min', 1e-6))
        if theta0 > filter_theta_min:
            if not (theta1 <= 1.05 * theta0 + 1e-16):
                accepted = False

        # Apply TR update
        if self.tr is not None:
            self.tr.update(pred_red=pred_red_tr, act_red=act_red, step_norm=step_norm_inf,
                        theta0=theta0, kkt=None, act_sz=0, H=H_reg, step=h)
            curR = self.tr.get_radius()
        else:
            state.radius = max(tr_min, float(Delta_new))
            curR = state.radius
            
        self.core.trmodel.radius  = curR

        # Update state
        x_out = x_trial if accepted else x
        self.core.add_sample(x_trial, f1, cI1, cE1)
        if accepted:
            self.core.on_accept(x_out, f1, cI1, cE1)
        vcur = max(np.maximum(0.0, cI1).max() if cI1.size else 0.0,
                np.abs(cE1).max() if cE1.size else 0.0)
        vprev = max(np.maximum(0.0, cI0).max() if cI0.size else 0.0,
                    np.abs(cE0).max() if cE0.size else 0.0)
        if state.adaptive_mu_update(vcur, vprev):
            self.core.cfg.mu = float(state.mu)

        # KKT stats
        kkt = nlp_model.kkt_residuals(x_out, state.lam, state.nu)
        stat = kkt.get('stat', 1e6)
        ineq = kkt.get('ineq', 0.0)
        eq = kkt.get('eq', 0.0)
        comp = kkt.get('comp', 0.0)

        # SOTA Convergence Check
        tol = getattr(self.cfg, 'tol', 1e-8)  # From run_solve
        max_stagnation = getattr(self.cfg, 'max_stagnation', 10)  # Max iterations without progress
        state.violation_history.append(vcur)
        if len(state.violation_history) > max_stagnation:
            state.violation_history.pop(0)

        # Convergence criteria
        converged = False
        if it > 0:
            # 1. Step size criterion
            step_criterion = step_norm_inf <= tol * max(1.0, norm(x_out))
            # 2. Objective stagnation
            f_history = getattr(state, 'f_history', [])
            f_history.append(f1)
            if len(f_history) > max_stagnation:
                f_history.pop(0)
            f_change = max(abs(f1 - f) for f in f_history) if f_history else float('inf')
            obj_criterion = f_change <= tol * max(1.0, abs(f1))
            # 3. Constraint violation
            viol_criterion = vcur <= tol
            # 4. Criticality measure
            crit_criterion = sigma <= tol
            # 5. Stagnation (repeated rejections)
            rejection_streak = getattr(state, 'rejection_streak', 0)
            if not accepted:
                state.rejection_streak = rejection_streak + 1
            else:
                state.rejection_streak = 0
            stagnation_criterion = state.rejection_streak >= max_stagnation
            # 6. KKT residuals
            kkt_criterion = stat <= tol and ineq <= tol and eq <= tol and comp <= tol

            # Converged if step size is small, objective/constraints are stable, or KKT conditions are met
            converged = (
                (step_criterion and viol_criterion and crit_criterion) or
                (obj_criterion and viol_criterion and crit_criterion) or
                kkt_criterion or
                stagnation_criterion
            )
            # if converged:
            #     logging.info(f"Converged at iteration {it}: "
            #                 f"step={step_norm_inf:.2e}, f_change={f_change:.2e}, "
            #                 f"viol={vcur:.2e}, sigma={sigma:.2e}, "
            #                 f"rejection_streak={state.rejection_streak}, "
            #                 f"KKT(stat={stat:.2e}, ineq={ineq:.2e}, eq={eq:.2e}, comp={comp:.2e})")

        # Store history for next iteration
        state.f_history = getattr(state, 'f_history', [])
        state.f_history.append(f1)
        if len(state.f_history) > max_stagnation:
            state.f_history.pop(0)

        # Info dictionary
        info = {
            'step_norm': step_norm_inf,
            'accepted': bool(accepted),
            'converged': bool(converged),
            'f': float(self._eval_cached(nlp_model, x_out)['f']),
            'theta': nlp_model.constraint_violation(x_out),
            'stat': stat,
            'ineq': ineq,
            'eq': eq,
            'comp': comp,
            'ls_iters': 0,
            'alpha': float(step_info.get('t', 0.0)),
            'rho': float(rho),
            'sigma': float(sigma),
            'pred_red_cons': float(pred_red_cons),
            'pred_red_quad': float(pred_red_quad),
            'tr_radius': float(curR),
            'mode': 'dfo_exact_penalty_core',
            'solve_success': True,
            'solve_info': {'core': step_info},
            'model_samples': int(self.core.arrays()[0].shape[0]),
            'cache_hits': int(self._cache_hits),
            'penalty_mu': float(state.mu),
            'used_multiplier_step': bool(step_info.get('used_multiplier_step', False)),
        }
        return x_out, state.lam, state.nu, info