# python/l1_dfo_stepper.py
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import l1core as L1  # pybind11 module
import numpy as np


def _constraint_violation_from_fvalues(fvals: np.ndarray,
                                       con_lb: np.ndarray,
                                       con_ub: np.ndarray) -> float:
    if fvals.ndim != 1:
        fvals = fvals.ravel()
    if con_lb.size == 0 and con_ub.size == 0:
        return 0.0
    cvals = fvals[1:]
    lb = con_lb if con_lb.size else np.full_like(cvals, -np.inf, dtype=float)
    ub = con_ub if con_ub.size else np.full_like(cvals,  np.inf, dtype=float)
    viol_lb = np.clip(lb - cvals, 0.0, np.inf)
    viol_ub = np.clip(cvals - ub, 0.0, np.inf)
    return float(np.sum(viol_lb) + np.sum(viol_ub))


class L1DFOStepper:
    """
    Single-step DFO analogue of SQPStepper.step.
    Returns (x_out, lam_out=None, nu_out=None, info).
    """

    def __init__(
        self,
        model,  # needs at least .n, .f, .cI_funcs, .cE_funcs
        trm: Optional[L1.TRModel] = None,
        options: Optional[L1.Options] = None,
        *,
        mu: float = 1.0,
        epsilon: float = 0.5,
        delta: float = 1e-4,
        lam_penalty: float = 0.01,
        var_lb: Optional[np.ndarray] = None,
        var_ub: Optional[np.ndarray] = None,
        con_lb: Optional[np.ndarray] = None,
        con_ub: Optional[np.ndarray] = None,
        x0: Optional[np.ndarray | Sequence[np.ndarray]] = None,
        min_seed_points: Optional[int] = None,   # default n+1 if None
    ):
        # ---- Basic problem assembly (no bound mutation) ---------------------
        func = L1.Funcao()
        func.addObjective(model.f)
        for c in getattr(model, "cI_funcs", []):
            func.addConstraint(c)
        for c in getattr(model, "cE_funcs", []):
            func.addConstraint(c)

        self.n = int(model.n)
        self.func = func
        self.trm = trm
        self.mu = float(mu)
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.lam_penalty = float(lam_penalty)

        # Bounds (default to ±inf)
        self.var_lb = np.full(self.n, -np.inf) if var_lb is None else np.asarray(var_lb, float).reshape(-1)
        self.var_ub = np.full(self.n,  np.inf) if var_ub is None else np.asarray(var_ub, float).reshape(-1)
        assert self.var_lb.shape == (self.n,) and self.var_ub.shape == (self.n,)
        # Constraint bounds (match func.con count or be empty)
        m = len(getattr(func, "con", []))
        self.con_lb = (np.full(m, -np.inf) if con_lb is None else np.asarray(con_lb, float).reshape(-1))
        self.con_ub = (np.full(m,  np.inf) if con_ub is None else np.asarray(con_ub, float).reshape(-1))
        if m > 0:
            assert self.con_lb.shape == (m,) and self.con_ub.shape == (m,)

        # Options (create if None)
        self.opt = options if options is not None else L1.Options()
        # Reasonable defaults (tweak if your C++ has its own defaults)
        self.opt.tol_radius = getattr(self.opt, "tol_radius", 1e-6)
        self.opt.tol_f = getattr(self.opt, "tol_f", 1e-6)
        self.opt.tol_measure = getattr(self.opt, "tol_measure", 1e-4)
        self.opt.tol_con = getattr(self.opt, "tol_con", 1e-4)
        self.opt.pivot_threshold = getattr(self.opt, "pivot_threshold", 1.0/8)
        self.opt.initial_radius = getattr(self.opt, "initial_radius", 1.0)
        self.opt.radius_max = getattr(self.opt, "radius_max", 5.0)
        self.opt.max_it = getattr(self.opt, "max_it", 100)
        self.opt.verbose = getattr(self.opt, "verbose", False)

        # Cache thresholds for quick access
        self.eta_1 = self.opt.eta_1
        self.eta_2 = self.opt.eta_2
        self.gamma_dec = self.opt.gamma_dec
        self.gamma_inc = self.opt.gamma_inc
        self.radius_max = self.opt.radius_max
        self.tol_radius = self.opt.tol_radius
        self.tol_measure = self.opt.tol_measure
        self.eps_c = self.opt.eps_c

        # ---- Build initial interpolation set from x0 ------------------------
        X0 = self._normalize_x0(x0)  # shape (n, k0) or None
        need = (min_seed_points if min_seed_points is not None else max(2, self.n + 1))
        initial_points = self._seed_points(X0, need)

        # Evaluate f & constraints at seeds
        n_functions = 1 + m
        k = initial_points.shape[1]
        fvals = np.empty((n_functions, k))
        for j in range(k):
            pj = L1.projectToBounds(initial_points[:, j], self.var_lb, self.var_ub)
            initial_points[:, j] = pj
            fvals[:, j] = func.calcAll(pj)

        # ---- Create / refresh TR model -------------------------------------
        self.trm = L1.TRModel(initial_points, fvals, self.opt)
        self.trm.rebuildModel(self.opt)
        self.trm.computePolynomialModels()

    # --------------------------- helpers -------------------------------------
    def _normalize_x0(self, x0):
        """Return x0 as (n,k) float array or None."""
        if x0 is None:
            return None
        x0 = np.asarray(x0, dtype=float)
        if x0.ndim == 1:
            assert x0.size == self.n, "x0 length must match model.n"
            return x0.reshape(self.n, 1)
        if x0.ndim == 2:
            # accept (n,k) or (k,n)
            if x0.shape[0] == self.n:
                return x0.copy()
            if x0.shape[1] == self.n:
                return x0.T.copy()
            raise ValueError(f"x0 shape {x0.shape} not compatible with n={self.n}")
        # list/sequence of vectors
        if isinstance(x0, (list, tuple)) and len(x0) > 0 and np.asarray(x0[0]).ndim == 1:
            M = np.stack([np.asarray(v, float).reshape(-1) for v in x0], axis=1)
            return self._normalize_x0(M)
        raise ValueError("Unsupported x0 format")

    def _seed_points(self, X0: Optional[np.ndarray], need: int) -> np.ndarray:
        """
        Ensure at least `need` seeds. If X0 is missing/short, add projected
        randomized and coordinate-perturbed points within bounds.
        """
        rng = np.random.default_rng()
        lb, ub = self.var_lb, self.var_ub

        seeds = []
        if X0 is not None:
            for j in range(X0.shape[1]):
                seeds.append(L1.projectToBounds(X0[:, j], lb, ub))

        if not seeds:
            # No x0 provided: start from mid-box (or zeros if unbounded)
            mid = np.where(np.isfinite(lb) & np.isfinite(ub), 0.5 * (lb + ub), 0.0)
            seeds.append(L1.projectToBounds(mid, lb, ub))

        # Radius for local sampling
        rad = float(self.opt.initial_radius)

        # Fill up to `need`
        while len(seeds) < need:
            base = seeds[0]
            # Try coordinate-perturbed points first
            i = (len(seeds) - 1) % self.n
            step = np.zeros(self.n)
            # make a step that respects bounds roughly
            span = (ub[i] - lb[i]) if (np.isfinite(ub[i]) and np.isfinite(lb[i])) else 2.0*rad
            mag = min(rad, 0.25 * abs(span)) if np.isfinite(span) else rad
            direction = 1.0 if (len(seeds) // self.n) % 2 == 0 else -1.0
            step[i] = direction * (0.5 + 0.5 * rng.random()) * max(1e-8, mag)
            cand = L1.projectToBounds(base + step, lb, ub)

            # If projection collapses the move (degenerate box), random box sample
            if np.allclose(cand, base):
                rand = np.array([
                    rng.uniform(lb[k], ub[k]) if (np.isfinite(lb[k]) and np.isfinite(ub[k]))
                    else base[k] + rng.uniform(-rad, rad)
                    for k in range(self.n)
                ])
                cand = L1.projectToBounds(rand, lb, ub)

            # Avoid duplicates
            if all(not np.allclose(cand, s) for s in seeds):
                seeds.append(cand)

            # Safety to avoid infinite loops
            if len(seeds) >= need + 5*self.n:
                break

        return np.column_stack(seeds[:need])

    # ----------------------------- API ---------------------------------------
    def _project(self, x: np.ndarray) -> np.ndarray:
        return L1.projectToBounds(x, self.var_lb, self.var_ub)

    def _pack_info(self, step_norm, accepted, converged,
                   f_val, theta, measure, alpha, rho) -> Dict:
        return {
            "step_norm": float(step_norm),
            "accepted": bool(accepted),
            "converged": bool(converged),
            "f": float(f_val),
            "theta": float(theta),
            "stat": float(measure),
            "ineq": float(theta),
            "eq": 0.0,
            "comp": 0.0,
            "ls_iters": 0,
            "alpha": float(alpha),
            "rho": float(rho),
            "tr_radius": float(self.trm.radius),
        }

    def step(self, model: Optional[object], x: np.ndarray, it: int
             ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Dict]:
        x = np.asarray(x, float).copy()
        self.trm.computePolynomialModels()
        fx_model, fmodel_g, fmodel_H = self.trm.getModelMatrices(0)
        cmodel = self.trm.extractConstraintsFromTRModel(self.con_lb, self.con_ub)

        measure, d, is_eactive = L1.l1CriticalityMeasureAndDescentDirection(
            self.trm, cmodel, x, self.mu, self.epsilon, self.var_lb, self.var_ub
        )

        # Optional criticality step
        if measure <= self.eps_c:
            thr_m = 1e3 * self.tol_measure
            thr_r = self.opt.initial_radius
            self.epsilon, _, _ = self.trm.trCriticalityStep(
                self.func, self.mu, self.epsilon,
                self.var_lb, self.var_ub, self.con_lb, self.con_ub,
                thr_m, thr_r, self.opt
            )
            self.trm.computePolynomialModels()
            fx_model, fmodel_g, fmodel_H = self.trm.getModelMatrices(0)
            cmodel = self.trm.extractConstraintsFromTRModel(self.con_lb, self.con_ub)
            measure, d, is_eactive = L1.l1CriticalityMeasureAndDescentDirection(
                self.trm, cmodel, x, self.mu, self.epsilon, self.var_lb, self.var_ub
            )

        # Near-stationary?
        p_now, fvals_now = L1.l1_function(self.func, self.con_lb, self.con_ub, self.mu, x.copy())
        theta_now = _constraint_violation_from_fvalues(fvals_now, self.con_lb, self.con_ub)
        if measure < self.tol_measure:
            info = self._pack_info(0.0, False, True, float(fvals_now[0]), theta_now, measure, 0.0, 0.0)
            return x, None, None, info

        # TR step
        x_step, pred, lam_out = L1.l1TrustRegionStep(
            self.trm, cmodel, x, self.epsilon, self.lam_penalty, self.mu,
            self.trm.radius, self.var_lb, self.var_ub
        )
        self.lam_penalty = float(lam_out)

        x_trial = self._project(x_step)
        s = x_trial - x
        pred = float(pred) if np.isfinite(pred) else -np.inf

        rho = -np.inf
        accepted = False
        if pred > 0.0 and np.all(np.isfinite(x_trial)):
            p_trial, fvals_trial = L1.l1_function(self.func, self.con_lb, self.con_ub, self.mu, x_trial.copy())
            if np.all(np.isfinite(fvals_trial)):
                ared = L1.evaluatePDescent(self.trm.fValues[:, self.trm.trCenter],
                                           fvals_trial, self.con_lb, self.con_ub, self.mu)
                rho = float(ared / max(pred, 1e-16))
                geom_ok = self.trm.isLambdaPoised(self.opt)
                if (rho >= self.eta_2) or (rho > self.eta_1 and geom_ok):
                    x = x_trial
                    accepted = True
                    _ = L1.changeTRCenter(self.trm, x_trial, fvals_trial, self.opt)
                elif np.isinf(rho):
                    _ = L1.ensureImprovement(self.trm, self.func, self.var_lb, self.var_ub, self.opt)
                else:
                    _ = L1.try2addPoint(self.trm, x_trial, fvals_trial,
                                        self.func, self.var_lb, self.var_ub, self.opt)
            else:
                _ = L1.ensureImprovement(self.trm, self.func, self.var_lb, self.var_ub, self.opt)
        else:
            _ = L1.ensureImprovement(self.trm, self.func, self.var_lb, self.var_ub, self.opt)

        # Radius update
        if pred > 0.0 and np.isfinite(rho):
            if rho < self.eta_2:
                self.trm.radius *= self.gamma_dec
            else:
                s_inf = min(self.trm.radius, float(np.max(np.abs(s))) if s.size else 0.0)
                growth = max(1.0, self.gamma_inc * (s_inf / max(1e-16, self.trm.radius)))
                self.trm.radius = min(growth * self.trm.radius, self.radius_max)

        # Final info
        p_now, fvals_now = L1.l1_function(self.func, self.con_lb, self.con_ub, self.mu, x.copy())
        f_true = float(fvals_now[0])
        theta = _constraint_violation_from_fvalues(fvals_now, self.con_lb, self.con_ub)
        step_norm = float(np.linalg.norm(s, ord=2))
        alpha = 1.0 if accepted else 0.0
        converged = (step_norm <= self.tol_radius) or (measure < self.tol_measure)

        info = self._pack_info(step_norm, accepted, converged, f_true, theta, measure, alpha,
                               rho if np.isfinite(rho) else -np.inf)
        return x, None, None, info
