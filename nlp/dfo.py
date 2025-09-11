# python/l1_dfo_stepper.py
from __future__ import annotations

from typing import Dict, Optional, Tuple

import l1core as L1  # your pybind11 module
import numpy as np


def _constraint_violation_from_fvalues(fvals: np.ndarray, con_lb: np.ndarray, con_ub: np.ndarray) -> float:
    """
    fvals layout assumed: [f, c1, c2, ..., cm]
    Violation θ = ||[max(lb - c, 0); max(c - ub, 0)]||_1
    (Same spirit as many NLP filters; feel free to swap to L∞ if you prefer.)
    """
    if fvals.ndim != 1:
        fvals = fvals.ravel()
    if con_lb.size == 0 and con_ub.size == 0:
        return 0.0
    cvals = fvals[1:]  # constraints slice
    lb = con_lb if con_lb.size else np.full_like(cvals, -np.inf, dtype=float)
    ub = con_ub if con_ub.size else np.full_like(cvals,  np.inf, dtype=float)
    viol_lb = np.clip(lb - cvals, 0.0, np.inf)
    viol_ub = np.clip(cvals - ub, 0.0, np.inf)
    return float(np.sum(viol_lb) + np.sum(viol_ub))

class L1DFOStepper:
    """
    Single-step DFO analogue of SQPStepper.step:
      step(model, x, lam, nu, it) -> (x_out, lam_out, nu_out, info)

    Notes:
    - `model` here is not used (DFO doesn’t need exact grads/Jacobians).
      Pass any object with at least `n` (dimension) if you need symmetry with SQP.
    - `lam` and `nu` are placeholders; returned unchanged (or updated if you later
      add multiplier heuristics).
    - Uses your l1core TR model, criticality test, TR step, and radius update.
    """

    def __init__(
        self,
        func: L1.Funcao,
        trm: Optional[L1.TRModel] = None,
        options: Optional[L1.TROptions] = L1.Options(),
        mu: Optional[float] = 1.0,
        epsilon: Optional[float] = 1.0,
        delta: Optional[float] = 0.1,
        lam_penalty: Optional[float] = 1.0,
        var_lb: Optional[np.ndarray] = np.array([]),
        var_ub: Optional[np.ndarray] = np.array([]),
        con_lb: Optional[np.ndarray] = np.array([]),
        con_ub: Optional[np.ndarray] = np.array([]) ,
        x0: Optional[np.ndarray] = np.array([])
    ):
        self.func = func
        self.trm = trm
        self.opt = options
        self.mu = float(mu)
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.lam_penalty = float(lam_penalty)
        self.var_lb = np.asarray(var_lb, dtype=float)
        self.var_ub = np.asarray(var_ub, dtype=float)
        self.con_lb = np.asarray(con_lb, dtype=float)
        self.con_ub = np.asarray(con_ub, dtype=float)

        # cached thresholds
        self.eta_1 = self.opt.eta_1
        self.eta_2 = self.opt.eta_2
        self.gamma_dec = self.opt.gamma_dec
        self.gamma_inc = self.opt.gamma_inc
        self.radius_max = self.opt.radius_max
        self.tol_radius = self.opt.tol_radius
        self.tol_measure = self.opt.tol_measure
        self.eps_c = self.opt.eps_c

        n_functions = 1 + len(func.con)
        initial_points = np.empty((func.n, 0), dtype=float)
        bl = self.var_lb
        bu = self.var_ub
        if len(x0) == 0:
            x0 = initial_points[:, 0].copy()
            x1 = x0.copy()
            rng = np.random.default_rng()
            for i in range(n):
                lbi = bl[i] if i < bl.size else -np.inf
                ubi = bu[i] if i < bu.size else  np.inf
                has_lb = np.isfinite(lbi)
                has_ub = np.isfinite(ubi)
                if has_lb and has_ub and lbi < ubi:
                    x1[i] = rng.uniform(lbi, ubi)
                else:
                    x1[i] = x0[i] + rng.uniform(-0.25, 0.25)
            x1 = L1.projectToBounds(x1, bl, bu)
            initial_points = np.c_[initial_points, x1]
            k = 2

        # Evaluate f,con at initial points (project first)
        initial_f_values = np.empty((n_functions, k))
        for j in range(k):
            initial_points[:, j] = L1.projectToBounds(initial_points[:, j], bl, bu)
            initial_f_values[:, j] = func.calcAll(initial_points[:, j])
        

    def _project(self, x: np.ndarray) -> np.ndarray:
        return L1.projectToBounds(x, self.var_lb, self.var_ub)

    def _pack_info(
        self,
        step_norm: float,
        accepted: bool,
        converged: bool,
        f_val: float,
        theta: float,
        measure: float,
        alpha: float,
        rho: float,
    ) -> Dict:
        # Keep SQP-like keys; map DFO quantities accordingly.
        return {
            "step_norm": float(step_norm),
            "accepted": bool(accepted),
            "converged": bool(converged),
            "f": float(f_val),
            "theta": float(theta),     # aggregate constraint violation
            "stat": float(measure),    # use l1 criticality measure as "stationarity"
            "ineq": float(theta),      # same θ (no equality split in this DFO scaffold)
            "eq": 0.0,
            "comp": 0.0,               # no complementarity in L1DFO
            "ls_iters": 0,             # no line search here
            "alpha": float(alpha),     # we treat the TR step as full step (alpha = 1)
            "rho": float(rho),         # ared/pred ratio
            "tr_radius": float(self.trm.radius),
        }

    def step(
        self,
        model: Optional[object],
        x: np.ndarray,
        lam: Optional[np.ndarray],
        nu: Optional[np.ndarray],
        it: int,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Dict]:
        """
        Perform one L1-DFO trust-region step around current TR center.
        Returns: (x_out, lam_out, nu_out, info)
        """
        x = np.asarray(x, dtype=float).copy()
        x = self._project(x)

        # Ensure TR polynomials are fresh at this iteration
        self.trm.computePolynomialModels()
        fx_model, fmodel_g, fmodel_H = self.trm.getModelMatrices(0)  # model of f at center
        cmodel = self.trm.extractConstraintsFromTRModel(self.con_lb, self.con_ub)

        # Criticality measure and direction (uses your C++ core)
        measure, d, is_eactive = L1.l1CriticalityMeasureAndDescentDirection(
            self.trm, cmodel, x, self.mu, self.epsilon, self.var_lb, self.var_ub
        )

        # Optional criticality step (epsilon reduction) if measure is tiny
        if measure <= self.eps_c:
            thr_m = 1e3 * self.tol_measure
            thr_r = self.opt.initial_radius
            self.epsilon, _, _ = self.trm.trCriticalityStep(
                self.func, self.mu, self.epsilon,
                self.var_lb, self.var_ub, self.con_lb, self.con_ub,
                thr_m, thr_r, self.opt
            )
            # Rebuild models after epsilon change
            self.trm.computePolynomialModels()
            fx_model, fmodel_g, fmodel_H = self.trm.getModelMatrices(0)
            cmodel = self.trm.extractConstraintsFromTRModel(self.con_lb, self.con_ub)
            measure, d, is_eactive = L1.l1CriticalityMeasureAndDescentDirection(
                self.trm, cmodel, x, self.mu, self.epsilon, self.var_lb, self.var_ub
            )

        # If near-stationary (for the L1 penalty subproblem), stop
        # (caller can treat info["converged"] to drive outer loop)
        # We still compute info at current x:
        p_now, fvals_now = L1.l1_function(self.func, self.con_lb, self.con_ub, self.mu, x.copy())
        theta_now = _constraint_violation_from_fvalues(fvals_now, self.con_lb, self.con_ub)
        if measure < self.tol_measure:
            info = self._pack_info(
                step_norm=0.0,
                accepted=False,
                converged=True,
                f_val=float(fvals_now[0]),
                theta=theta_now,
                measure=measure,
                alpha=0.0,
                rho=0.0,
            )
            return x, lam, nu, info

        # Propose TR step (model QP inside l1core); returns x_step (already “x+s”) or a raw step?
        # Your binding returns the next point; we’ll project it and treat as trial:
        x_step, pred, lam_out = L1.l1TrustRegionStep(
            self.trm, cmodel, x, self.epsilon, self.lam_penalty, self.mu, self.trm.radius, self.var_lb, self.var_ub
        )
        # Update the internal penalty lambda if changed by core:
        self.lam_penalty = float(lam_out)

        x_trial = self._project(x_step)
        s = x_trial - x
        pred = float(pred) if np.isfinite(pred) else -np.inf

        # Evaluate trial (true black-box composite)
        rho = -np.inf
        accepted = False
        if pred > 0.0 and np.all(np.isfinite(x_trial)):
            p_trial, fvals_trial = L1.l1_function(self.func, self.con_lb, self.con_ub, self.mu, x_trial.copy())
            if np.all(np.isfinite(fvals_trial)):
                ared = L1.evaluatePDescent(self.trm.fValues[:, self.trm.trCenter], fvals_trial, self.con_lb, self.con_ub, self.mu)
                rho = float(ared / max(pred, 1e-16))
                # Accept rules (match your driver)
                geom_ok = self.trm.isLambdaPoised(self.opt)
                if (rho >= self.eta_2) or (rho > self.eta_1 and geom_ok):
                    # accept and recenter
                    x = x_trial
                    accepted = True
                    _ = L1.changeTRCenter(self.trm, x_trial, fvals_trial, self.opt)
                elif np.isinf(rho):
                    _ = L1.ensureImprovement(self.trm, self.func, self.var_lb, self.var_ub, self.opt)
                else:
                    _ = L1.try2addPoint(self.trm, x_trial, fvals_trial, self.func, self.var_lb, self.var_ub, self.opt)
            else:
                # non-finite trial eval: force model repair
                _ = L1.ensureImprovement(self.trm, self.func, self.var_lb, self.var_ub, self.opt)
        else:
            # No predictive decrease: force a model improvement
            _ = L1.ensureImprovement(self.trm, self.func, self.var_lb, self.var_ub, self.opt)

        # TR radius update (match your rules)
        if pred > 0.0 and np.isfinite(rho):
            if rho < self.eta_2:
                # shrink
                self.trm.radius *= self.gamma_dec
            else:
                # grow (capped)
                s_inf = min(self.trm.radius, float(np.max(np.abs(s))) if s.size else 0.0)
                growth = max(1.0, self.gamma_inc * (s_inf / max(1e-16, self.trm.radius)))
                self.trm.radius = min(growth * self.trm.radius, self.radius_max)

        # Final info (true f, θ at current x)
        p_now, fvals_now = L1.l1_function(self.func, self.con_lb, self.con_ub, self.mu, x.copy())
        f_true = float(fvals_now[0])
        theta = _constraint_violation_from_fvalues(fvals_now, self.con_lb, self.con_ub)

        step_norm = float(np.linalg.norm(s, ord=2))
        alpha = 1.0 if accepted else 0.0
        converged = (step_norm <= self.tol_radius) or (measure < self.tol_measure)

        info = self._pack_info(
            step_norm=step_norm,
            accepted=accepted,
            converged=converged,
            f_val=f_true,
            theta=theta,
            measure=measure,
            alpha=alpha,
            rho=rho if np.isfinite(rho) else -np.inf,
        )
        return x, lam, nu, info
