from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gpytorch
import numpy as np
import torch
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from numpy.linalg import lstsq, norm
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import linprog
from scipy.spatial.distance import cdist

from .dfo_aux import *
from .dfo_aux import (
    _csv_basis,
    _distance_weights,
    _huber_weights,
    _improved_fps,
    _select_poised_rows,
    _solve_weighted_ridge,
)
from .dfo_tr import *


# GPyTorch GP Model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(MaternKernel(nu=2.5))
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# Modified Objective Model with GP
class _ObjectiveModelBase:
    def fit_objective(self, Y: np.ndarray, rhs: np.ndarray, n: int, cfg) -> tuple:
        raise NotImplementedError
    

class DFOGPModel(_ObjectiveModelBase):
    """
    CUDA-aware GP objective model with:
      - ARD Matérn 2.5 + mild priors
      - y-standardization (train on z=(y-mean)/std, scale grads/H back)
      - Warm start + early stopping (Adam) + short LBFGS polish
      - Accurate autograd Hessian by default (cfg.gp_hess_mode = "autograd")
      - Optional fast diag Hessian from lengthscales ("diag_from_lengthscale")
      - Acquisition computed on-device
    """

    class _State:
        def __init__(self):
            self.best_sd = None   # {"model": sd, "lik": sd}
            self.fits_done = 0
            self.frozen = False

    def __init__(self):
        self._dev = None
        self._dtype = None
        self.state = DFOGPModel._State()
        # cached for acquisition
        self.gp_model = None
        self.likelihood = None
        self.X_train = None
        self.y_train = None
        # standardization cache
        self._y_mean = None
        self._y_std = None

    # ---------- helpers ----------
    def _pick_device(self, cfg):
        want = getattr(cfg, "gp_device", "auto")
        if want == "cuda" or (want == "auto" and torch.cuda.is_available()):
            return torch.device("cuda")
        return torch.device("cpu")

    def _pick_dtype(self, cfg):
        dt = getattr(cfg, "gp_dtype", "float32").lower()
        return torch.float64 if dt == "float64" else torch.float32

    def _new_model(self, X, y, likelihood):
        from gpytorch.priors import GammaPrior, LogNormalPrior

        class ExactGPARD(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, lik):
                super().__init__(train_x, train_y, lik)
                self.mean_module = gpytorch.means.ConstantMean()
                ard_dims = train_x.shape[-1]
                base = MaternKernel(
                    nu=2.5,
                    ard_num_dims=ard_dims,
                    lengthscale_prior=LogNormalPrior(0.0, 0.5),
                )
                self.covar_module = ScaleKernel(
                    base_kernel=base,
                    outputscale_prior=GammaPrior(2.0, 0.5),
                )

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        return ExactGPARD(X, y, likelihood)

    def _maybe_warm_load(self, model, likelihood):
        sd = self.state.best_sd
        if sd is None:
            return
        try:
            model.load_state_dict(sd["model"])
            likelihood.load_state_dict(sd["lik"])
        except Exception:
            pass

    def _save_best(self, model, likelihood):
        self.state.best_sd = {
            "model": {k: v.detach().clone() for k, v in model.state_dict().items()},
            "lik":   {k: v.detach().clone() for k, v in likelihood.state_dict().items()},
        }

    # ---------- main ----------
    def fit_objective(self, Y: np.ndarray, rhs: np.ndarray, n: int, cfg) -> tuple:
        m = Y.shape[0]
        if m < cfg.min_pts_gp:
            return np.zeros(n), cfg.eig_floor * np.eye(n), {
                "model": "gp", "m_used": 0, "fallback": "insufficient_points"
            }

        # Thin if too many points
        if m > cfg.gp_max_pts:
            Phi_full, _, _ = _csv_basis(Y)
            sel = _select_poised_rows(Phi_full, cfg.gp_max_pts)
            Y = Y[sel]; rhs = rhs[sel]; m = Y.shape[0]

        # device/dtype
        if self._dev is None:   self._dev = self._pick_device(cfg)
        if self._dtype is None: self._dtype = self._pick_dtype(cfg)

        # --- y standardization for conditioning ---
        y_np = rhs.astype(np.float64, copy=False)
        y_mean = float(np.mean(y_np))
        y_std  = float(np.std(y_np) + 1e-8)
        self._y_mean, self._y_std = y_mean, y_std

        y_std_t = torch.as_tensor((y_np - y_mean) / y_std, device=self._dev, dtype=self._dtype)
        X_train = torch.as_tensor(Y, device=self._dev, dtype=self._dtype)

        try:
            # Likelihood with prior + noise floor
            from gpytorch.priors import GammaPrior
            likelihood = GaussianLikelihood(
                noise_prior=GammaPrior(1.1, 0.05),
                noise_constraint=gpytorch.constraints.GreaterThan(max(cfg.gp_noise, 1e-6)),
            ).to(self._dev, self._dtype)

            model = self._new_model(X_train, y_std_t, likelihood).to(self._dev, self._dtype)

            # Warm-start
            if getattr(cfg, "gp_warm_start", True):
                self._maybe_warm_load(model, likelihood)

            # Train (Adam) with early stopping + short LBFGS polish (unless frozen)
            steps    = int(getattr(cfg, "gp_train_steps", 25))
            patience = int(getattr(cfg, "gp_patience", 5))
            lr       = float(getattr(cfg, "gp_lr", 0.08))

            model.train(); likelihood.train()
            best_loss = float("inf"); stall = 0

            if not self.state.frozen and steps > 0:
                opt = torch.optim.Adam(model.parameters(), lr=lr)
                mll = ExactMarginalLogLikelihood(likelihood, model)

                with gpytorch.settings.cholesky_jitter(1e-4):
                    for _ in range(steps):
                        opt.zero_grad(set_to_none=True)
                        out = model(X_train)
                        loss = -mll(out, y_std_t)
                        loss.backward()
                        opt.step()

                        L = float(loss.detach().cpu())
                        if L + 1e-7 < best_loss:
                            best_loss = L; stall = 0
                            self._save_best(model, likelihood)
                        else:
                            stall += 1
                            if stall >= patience:
                                break

                    # restore best
                    if self.state.best_sd is not None:
                        model.load_state_dict(self.state.best_sd["model"])
                        likelihood.load_state_dict(self.state.best_sd["lik"])

                    # quick LBFGS polish (few steps)
                    try:
                        lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=5, history_size=10)
                        def closure():
                            lbfgs.zero_grad(set_to_none=True)
                            out = model(X_train)
                            loss = -mll(out, y_std_t)
                            loss.backward()
                            return loss
                        with gpytorch.settings.cholesky_jitter(1e-4):
                            lbfgs.step(closure)
                        self._save_best(model, likelihood)
                        model.load_state_dict(self.state.best_sd["model"])
                        likelihood.load_state_dict(self.state.best_sd["lik"])
                    except Exception:
                        pass

                self.state.fits_done += 1
                freeze_after = int(getattr(cfg, "gp_freeze_after", 0) or 0)
                if freeze_after > 0 and self.state.fits_done >= freeze_after:
                    self.state.frozen = True

            # ---- Evaluate ∇ and ∇² of original-scale mean at x0=0 ----
            model.eval(); likelihood.eval()
            x0 = torch.zeros(1, n, device=self._dev, dtype=self._dtype, requires_grad=True)

            # Predict standardized mean; map back to original scale with y_std,y_mean.
            # Only derivatives of the mean matter; +y_mean is constant.
            with gpytorch.settings.fast_pred_var():
                m_std = likelihood(model(x0)).mean  # standardized

            # Gradient (original scale): grad = y_std * d m_std / dx
            grad_std = torch.autograd.grad(m_std, x0, create_graph=True)[0]  # shape (1,n)
            g = (y_std * grad_std).detach().flatten().cpu().numpy()

            # Hessian
            mode = getattr(cfg, "gp_hess_mode", "autograd")
            if mode == "diag_from_lengthscale":
                with torch.no_grad():
                    os = model.covar_module.outputscale.clamp_min(cfg.eig_floor)
                    ls = model.covar_module.base_kernel.lengthscale[0].clamp_min(1e-6)
                    diag = (os / (ls * ls)).detach().cpu().numpy()
                H = np.diag(np.maximum((y_std * diag).astype(rhs.dtype, copy=False), cfg.eig_floor))
            else:
                H = np.zeros((n, n), dtype=rhs.dtype)
                for i in range(n):
                    gi = grad_std[0, i]
                    # d/dx_j of standardized grad -> multiply by y_std to map to original
                    gii = torch.autograd.grad(gi, x0, retain_graph=True)[0][0]  # shape (n,)
                    row = (y_std * gii).detach().cpu().numpy()
                    H[i, :] = row
                # symmetrize + PSD clamp
                H = 0.5 * (H + H.T)
                try:
                    w, V = np.linalg.eigh(H)
                    H = (V * np.maximum(w, cfg.eig_floor)) @ V.T
                except Exception:
                    H = H + cfg.eig_floor * np.eye(n, dtype=H.dtype)

            info = {
                "model": "gp",
                "m_used": int(m),
                "fallback": None,
                "kernel": "matern_2.5_ard",
                "device": str(self._dev),
                "dtype": str(self._dtype).split(".")[-1],
                "noise": float(likelihood.noise.detach().cpu().item()),
                "fits_done": int(self.state.fits_done),
                "frozen": bool(self.state.frozen),
                "ystd": float(y_std),
                "ymean": float(y_mean),
                "hess_mode": mode,
            }

            # cache for acquisition
            self.gp_model = model
            self.likelihood = likelihood
            self.X_train = X_train
            self.y_train = y_std_t  # NOTE: stored standardized targets

            return g, H, info

        except Exception as e:
            # Linear fallback
            Phi = np.hstack([np.ones((m, 1)), Y])
            wts = _huber_weights(rhs, cfg.huber_delta) * _distance_weights(Y, cfg.dist_w_power)
            coef = _solve_weighted_ridge(Phi, rhs, wts, cfg.ridge)
            g = np.zeros(n, dtype=rhs.dtype)
            if coef.size >= 1 + n:
                g[:] = coef[1:1+n]
            H = cfg.eig_floor * np.eye(n, dtype=rhs.dtype)
            info = {"model": "gp", "m_used": int(m), "fallback": f"linear (exception: {str(e)})"}
            return g, H, info

    # ---------- acquisition ----------
    def get_acquisition(self, X: np.ndarray, acq_type: str = "ei", xi: float = 0.01) -> np.ndarray:
        if not hasattr(self, "gp_model") or self.gp_model is None:
            return np.zeros(len(X), dtype=float)
        Xt = torch.as_tensor(X, device=self._dev, dtype=self._dtype)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if acq_type == "ei":
                acq = expected_improvement(self.gp_model, self.likelihood, Xt, self.y_train, xi)
            elif acq_type == "logei":
                acq = log_expected_improvement(self.gp_model, self.likelihood, Xt, self.y_train, xi)
            else:
                raise ValueError(f"Unknown acquisition function: {acq_type}")
        return acq.detach().cpu().numpy()

class _TMOptions:
    pivot_threshold = 1e-3
    exchange_threshold = 1e-3
    tol_radius = 1e-6
    add_threshold = 1e-3
    
# ===============================
# DFOExactPenalty
# ===============================
class DFOExactPenalty:
    def __init__(self, n, m_ineq, m_eq, cfg: DFOConfig = None, objective_model: _ObjectiveModelBase = None):
        self.n = n
        self.m_ineq = m_ineq
        self.m_eq = m_eq
        self.cfg = cfg or DFOConfig()
        self.X: List[np.ndarray] = []
        self.F: List[float] = []
        self.C: List[np.ndarray] = []
        self.last_diag: Dict = {}
        self.obj_model = DFOGPModel() if objective_model is None else objective_model
        self.trmodel: Optional[TRModel] = None
        self._tm_opts = _TMOptions()

    # -------- TR hooks --------
    def attach_trmodel(self, trmodel: "TRModel" = None, tm_options=None):
        if trmodel is None:
            q = 1 + self.m_ineq + self.m_eq
            trmodel = TRModel(self.n, q, self.cfg)
        self.trmodel = trmodel
        if tm_options is not None:
            self._tm_opts = tm_options

    def _tm_funcs(self):
        mI, mE = self.m_ineq, self.m_eq
        def _call(x_abs: np.ndarray):
            f, cI, cE = self.oracle_eval(x_abs)
            fvec = np.concatenate((
                np.array([float(f)], dtype=float),
                np.asarray(cI, float).ravel() if mI else np.zeros(0),
                np.asarray(cE, float).ravel() if mE else np.zeros(0),
            ))
            return fvec, True
        return _call

    def _tm_seed_if_needed(self, center: np.ndarray):
        if self.trmodel is None:
            return
        if self.trmodel.pointsAbs.shape[0] == 0:
            f, cI, cE = self.oracle_eval(center)
            fvec = np.concatenate((
                np.array([float(f)], dtype=float),
                np.asarray(cI, float).ravel() if self.m_ineq else np.zeros(0),
                np.asarray(cE, float).ravel() if self.m_eq else np.zeros(0),
            ))
            self.trmodel.append_raw_sample(center, fvec)
            self.trmodel.ensure_basis_initialized()

    def _tm_append_if_new(self, x: np.ndarray, f: float, ci: np.ndarray, ce: np.ndarray):
        if self.trmodel is None:
            return
        fvec = np.concatenate((
            np.array([float(f)], dtype=float),
            np.asarray(ci, float).ravel() if self.m_ineq else np.zeros(0),
            np.asarray(ce, float).ravel() if self.m_eq else np.zeros(0),
        ))
        self.trmodel.append_raw_sample(x, fvec)

    def _tm_maintain_geometry(self, center: np.ndarray, Delta: float):
        if self.trmodel is None:
            return
        self.trmodel.radius = float(max(Delta, 1e-12))
        try:
            if self.trmodel.hasDistantPoints(self._tm_opts) or self.trmodel.isOld(self._tm_opts):
                self.trmodel.rebuildModel(self._tm_opts)
        except Exception:
            pass
        tries = 0
        while not self.trmodel.isLambdaPoised(self._tm_opts) and tries <= max(1, self.n):
            self.trmodel.ensure_improvement(self._tm_funcs(), None, None, self._tm_opts)
            tries += 1

    def on_accept(self, x_new: np.ndarray, f: float, cI: Optional[np.ndarray], cE: Optional[np.ndarray]):
        if self.trmodel is None:
            return
        fvec = np.concatenate((
            np.array([float(f)], dtype=float),
            np.asarray(cI, float).ravel() if (cI is not None and cI.size) else (np.zeros(self.m_ineq)),
            np.asarray(cE, float).ravel() if (cE is not None and cE.size) else (np.zeros(self.m_eq)),
        ))
        self.trmodel.change_tr_center(x_new, fvec, self._tm_opts)

    def set_objective_model(self, model: _ObjectiveModelBase):
        self.obj_model = model

    # -------- Samples store --------
    def _is_duplicate_x(self, x: np.ndarray, atol: float = 1e-12, rtol: float = 1e-9) -> bool:
        if not self.X:
            return False
        X = np.vstack(self.X)
        dx = np.linalg.norm(X - x[None, :], axis=1)
        return bool(np.any(dx <= atol + rtol * max(1.0, np.linalg.norm(x))))


    def arrays(self):
        if not self.X:
            return np.zeros((0, self.n)), np.zeros((0,)), np.zeros((0, self.m_ineq + self.m_eq))
        return np.vstack(self.X), np.asarray(self.F, float), np.vstack(self.C)

    def add_sample(self, x: np.ndarray, f: float, cI: np.ndarray = None, cE: np.ndarray = None):
        """
        Add a new sample point to the model, using polynomial pivoting and orthogonalization.

        Args:
            x (np.ndarray): Input point of shape (n,).
            f (float): Objective function value.
            cI (np.ndarray, optional): Inequality constraint values of shape (m_ineq,).
            cE (np.ndarray, optional): Equality constraint values of shape (m_eq,).
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        if self._is_duplicate_x(x):
            # logging.debug(f"Point {x} rejected: duplicate within tolerance")
            return

        ci = np.asarray(cI, dtype=np.float64).ravel() if cI is not None else np.zeros(self.m_ineq, dtype=np.float64)
        ce = np.asarray(cE, dtype=np.float64).ravel() if cE is not None else np.zeros(self.m_eq, dtype=np.float64)
        c = np.concatenate([ci, ce]) if (self.m_ineq + self.m_eq) > 0 else np.zeros(0, dtype=np.float64)

        # Compute constraint violation for weighting
        vI = np.maximum(0.0, ci).max() if ci.size else 0.0
        vE = np.abs(ce).max() if ce.size else 0.0
        viol = max(vI, vE)
        viol_weight = 1.0 / (1.0 + viol)

        # Check if the new point improves the polynomial basis
        if self.trmodel is not None and self.X:
            Delta = max(self.trmodel.radius, 1e-12)
            X_all = np.vstack(self.X) if self.X else np.zeros((0, self.n), dtype=np.float64)
            Y_current = (X_all - self.trmodel.center[None, :]) / Delta if X_all.shape[0] > 0 else np.zeros((0, self.n))
            y_new = (x - self.trmodel.center) / Delta
            Phi_current, _, _ = _csv_basis(Y_current)
            Phi_new, _, _ = _csv_basis(np.vstack([Y_current, y_new]) if Y_current.shape[0] > 0 else y_new[None, :])

            try:
                sel = _select_poised_rows(Phi_current, min(Phi_current.shape[1], Y_current.shape[0])) if Phi_current.shape[0] > 0 else np.array([])
                leverage = np.sum((Phi_new[-1] @ np.linalg.pinv(Phi_current[sel]))**2) if sel.size > 0 else 1.0
                leverage_threshold = getattr(self.cfg, 'leverage_threshold', 1e-3)
                
                if leverage < leverage_threshold * viol_weight:
                    # logging.debug(f"Point {x} rejected: leverage score {leverage:.2e} below threshold {leverage_threshold:.2e}")
                    return
            except np.linalg.LinAlgError as e:
                logging.warning(f"Leverage score computation failed: {e}. Accepting point without pivoting check.")
                leverage = 1.0

        # Append the new point
        self.X.append(x)
        self.F.append(float(f))
        self.C.append(c)
        if self.trmodel is not None:
            fvec = np.concatenate([np.array([float(f)], dtype=np.float64), ci, ce])
            self._tm_append_if_new(x, f, ci, ce)

        # Prune if exceeding max_pts
        if len(self.X) > self.cfg.max_pts:
            logging.debug(f"Sample set size {len(self.X)} exceeds max_pts {self.cfg.max_pts}. Pruning.")
            X_all = np.vstack(self.X) if self.X else np.zeros((0, self.n), dtype=np.float64)
            C_all = np.vstack(self.C) if self.C else np.zeros((0, self.m_ineq + self.m_eq), dtype=np.float64)
            F_all = np.array(self.F, dtype=np.float64) if self.F else np.zeros((0,), dtype=np.float64)
            Delta = max(self.trmodel.radius, 1e-12) if self.trmodel else 1.0
            Y_all = (X_all - self.trmodel.center[None, :]) / Delta if self.trmodel and X_all.shape[0] > 0 else X_all
            Phi, _, _ = _csv_basis(Y_all)

            try:
                idx_keep = _select_poised_rows(Phi, min(self.cfg.max_pts, X_all.shape[0]))
                idx_keep = np.asarray(idx_keep, dtype=np.int64)  # Ensure integer indices
                if np.max(idx_keep) >= X_all.shape[0]:
                    logging.warning(f"Invalid indices in idx_keep: {idx_keep}. Adjusting to valid range.")
                    idx_keep = idx_keep[idx_keep < X_all.shape[0]]
                if len(idx_keep) == 0:
                    logging.warning("No valid indices after pruning. Using distance-based fallback.")
                    raise np.linalg.LinAlgError("Empty index set")
                idx_keep = np.sort(idx_keep)
                # Construct fvals with objective and constraints
                fvals_new = np.hstack([
                    F_all[idx_keep].reshape(-1, 1),
                    C_all[idx_keep]
                ])
                self.X = [X_all[i] for i in idx_keep]
                self.F = [F_all[i] for i in idx_keep]
                self.C = [C_all[i] for i in idx_keep]
                if self.trmodel is not None:
                    self.trmodel.pointsAbs = X_all[idx_keep]
                    self.trmodel.fvals = fvals_new
                    self.trmodel.rebuild_interp()
            except np.linalg.LinAlgError as e:
                logging.warning(f"Pruning failed: {e}. Using distance-based pruning.")
                d = norm(X_all - self.trmodel.center[None, :], axis=1) if self.trmodel and X_all.shape[0] > 0 else norm(X_all, axis=1)
                idx_keep = np.argsort(d)[:min(self.cfg.max_pts, X_all.shape[0])]
                idx_keep = np.sort(idx_keep)
                self.X = [X_all[i] for i in idx_keep]
                self.F = [F_all[i] for i in idx_keep]
                self.C = [C_all[i] for i in idx_keep]
                if self.trmodel is not None:
                    self.trmodel.pointsAbs = X_all[idx_keep]
                    self.trmodel.fvals = C_all[idx_keep]
                    self.trmodel.rebuild_interp()

    def _improve_models_FL(self, center: np.ndarray, Delta: float, budget: int = 2) -> None:
        """
        Improve the local model by adding new sample points within the trust region, ensuring synchronized pruning.

        Args:
            center (np.ndarray): Trust region center of shape (n,).
            Delta (float): Trust region radius.
            budget (int): Maximum number of new points to add.
        """
        if budget <= 0 or not hasattr(self, "oracle") or self.oracle is None:
            return
        Delta = float(max(Delta, 1e-12))

        # Get current working subset
        Xs, Fs, Cs = self._pick_working_subset(center, Delta)
        Y = (Xs - center[None, :]) / Delta if Xs.shape[0] > 0 else np.zeros((0, self.n), dtype=np.float64)

        # Remove low-leverage points, synchronizing with trmodel
        if Xs.shape[0] >= self.cfg.min_pts_gp:
            Phi, _, _ = _csv_basis(Y)
            try:
                sel = _select_poised_rows(Phi, min(Phi.shape[1], Xs.shape[0]))
                lev = np.sum((Phi @ np.linalg.pinv(Phi[sel]))**2, axis=1)
                low_idx_local = np.argsort(lev)[:min(budget, len(lev))]
                X_all = np.vstack(self.X) if self.X else np.zeros((0, self.n), dtype=np.float64)
                C_all = np.vstack(self.C) if self.C else np.zeros((0, self.m_ineq + self.m_eq), dtype=np.float64)
                F_all = np.array(self.F, dtype=np.float64) if self.F else np.zeros((0,), dtype=np.float64)
                global_map = []
                for xi in Xs:
                    cand = np.where((X_all == xi).all(axis=1))[0]
                    j = int(cand[0]) if cand.size else int(np.argmin(norm(X_all - xi[None, :], axis=1)))
                    global_map.append(j)
                global_map = np.asarray(global_map, dtype=int)
                to_drop_global = set(global_map[low_idx_local].tolist())
                if to_drop_global:
                    # logging.debug(f"Pruning {len(to_drop_global)} low-leverage points from global set")
                    idx_keep = np.array([i for i in range(X_all.shape[0]) if i not in to_drop_global])
                    idx_keep = np.sort(idx_keep)
                    idx_keep = np.asarray(idx_keep, dtype=np.int64)  # Ensure integer indices

                    # check if idx_keep is empty
                    if len(idx_keep) == 0:
                        logging.warning("No valid indices after pruning. Using distance-based fallback.")
                        pass

                    # Construct fvals with objective and constraints
                    fvals_new = np.hstack([
                        F_all[idx_keep].reshape(-1, 1),
                        C_all[idx_keep]
                    ])
                    self.X = [X_all[i] for i in idx_keep]
                    self.F = [F_all[i] for i in idx_keep]
                    self.C = [C_all[i] for i in idx_keep]
                    if self.trmodel is not None:
                        self.trmodel.pointsAbs = X_all[idx_keep]
                        self.trmodel.fvals = fvals_new
                        self.trmodel.rebuild_interp()
                    Xs, Fs, Cs = self._pick_working_subset(center, Delta)
                    Y = (Xs - center[None, :]) / Delta if Xs.shape[0] > 0 else np.zeros((0, self.n), dtype=np.float64)
            except np.linalg.LinAlgError as e:
                logging.warning(f"Leverage score computation failed: {e}. Falling back to distance-based pruning.")
                d = norm(Xs - center[None, :], axis=1) if Xs.shape[0] > 0 else np.zeros(0)
                low_idx_local = np.argsort(d)[:min(budget, len(d))]
                X_all = np.vstack(self.X) if self.X else np.zeros((0, self.n), dtype=np.float64)
                C_all = np.vstack(self.C) if self.C else np.zeros((0, self.m_ineq + self.m_eq), dtype=np.float64)
                F_all = np.array(self.F, dtype=np.float64) if self.F else np.zeros((0,), dtype=np.float64)
                global_map = []
                for xi in Xs:
                    cand = np.where((X_all == xi).all(axis=1))[0]
                    j = int(cand[0]) if cand.size else int(np.argmin(norm(X_all - xi[None, :], axis=1)))
                    global_map.append(j)
                global_map = np.asarray(global_map, dtype=int)
                to_drop_global = set(global_map[low_idx_local].tolist())
                if to_drop_global:
                    logging.debug(f"Pruning {len(to_drop_global)} distance-based points from global set")
                    idx_keep = np.array([i for i in range(X_all.shape[0]) if i not in to_drop_global])
                    idx_keep = np.sort(idx_keep)
                    idx_keep = np.asarray(idx_keep, dtype=np.int64)  # Ensure integer indices

                    if len(idx_keep) == 0:
                        logging.warning("No valid indices after pruning. Using distance-based fallback.")
                        pass
                    # Construct fvals with objective and constraints
                    fvals_new = np.hstack([
                        F_all[idx_keep].reshape(-1, 1),
                        C_all[idx_keep]
                    ])
                    self.X = [X_all[i] for i in idx_keep]
                    self.F = [F_all[i] for i in idx_keep]
                    self.C = [C_all[i] for i in idx_keep]
                    if self.trmodel is not None:
                        self.trmodel.pointsAbs = X_all[idx_keep]
                        self.trmodel.fvals = fvals_new
                        self.trmodel.rebuild_interp()
                    Xs, Fs, Cs = self._pick_working_subset(center, Delta)
                    Y = (Xs - center[None, :]) / Delta if Xs.shape[0] > 0 else np.zeros((0, self.n), dtype=np.float64)

        # Adjust budget based on model quality
        budget = min(budget, self.cfg.min_pts_gp - Xs.shape[0] if Xs.shape[0] < self.cfg.min_pts_gp else budget)
        if budget <= 0:
            return

        # Generate candidate points using quasi-Monte Carlo (Sobol)
        box = Delta * self.cfg.model_radius_mult
        num_cand = max(10, 5 * self.n)
        sampler = qmc.Sobol(d=self.n, scramble=True, seed=np.random.randint(0, 10000))
        cand = sampler.random(num_cand)
        cand = qmc.scale(cand, -box, box) + center
        cand = np.unique(cand, axis=0)

        # Select new points using acquisition function or distance-based selection
        if self.trmodel and self.trmodel.interp and hasattr(self.trmodel.interp, "get_acquisition"):
            try:
                acq_vals = self.trmodel.interp.get_acquisition(cand, self.cfg.acquisition, self.cfg.xi)
                X_cur, _, C_cur = self.arrays()
                if C_cur.shape[0] > 0:
                    cI_cur = C_cur[:, :self.m_ineq] if self.m_ineq else np.zeros((C_cur.shape[0], 0))
                    vI = np.maximum(0.0, cI_cur).max(axis=1)
                    vE = np.abs(C_cur[:, self.m_ineq:]) if self.m_eq else np.zeros((C_cur.shape[0], 0))
                    vE = vE.max(axis=1) if vE.size else np.zeros(C_cur.shape[0])
                    weights = 1.0 / (1.0 + vI + vE)
                    acq_vals *= weights[np.argmin(cdist(cand, X_cur), axis=1)]
                pick = np.argsort(-acq_vals)[:budget]
            except Exception as e:
                logging.warning(f"Acquisition function failed: {e}. Falling back to distance-based selection.")
                X_cur, _, _ = self.arrays()
                if X_cur.shape[0] > 0:
                    D = cdist(cand, X_cur, metric="euclidean")
                    minD = np.min(D, axis=1)
                else:
                    minD = np.full(len(cand), np.inf)
                pick = np.argsort(-minD)[:budget]
        else:
            X_cur, _, _ = self.arrays()
            if X_cur.shape[0] > 0:
                D = cdist(cand, X_cur, metric="euclidean")
                minD = np.min(D, axis=1)
            else:
                minD = np.full(len(cand), np.inf)
            pick = np.argsort(-minD)[:budget]

        # Add selected points with pivoting check
        for x_new in cand[pick]:
            f_new, cI_new, cE_new = self.oracle_eval(x_new)
            self.add_sample(x_new, f_new, cI_new, cE_new)
    # -------- Local fitting --------
    def fit_local(self, center: np.ndarray, Delta: float) -> "FitResult":
        self._tm_seed_if_needed(center)
        self._tm_maintain_geometry(center, Delta)

        X, F, C = self.arrays()
        if X.shape[0] < self.cfg.min_pts_gp:
            # Add initial samples if insufficient
            self._improve_models_FL(center, Delta, budget=max(6, self.n + 1))
            X, F, C = self.arrays()
        if X.shape[0] < self.cfg.min_pts_gp:
            return self._fallback()

        max_pts = min(self.cfg.max_pts, X.shape[0])
        d = norm(X - center[None, :], axis=1)
        idx_close = np.argsort(d)[:max_pts]
        Xs, Fs = X[idx_close], F[idx_close]
        Cs = C[idx_close] if C.size else np.zeros((len(idx_close), self.m_ineq + self.m_eq))

        Delta = float(max(Delta, 1e-12))
        Y = (Xs - center[None, :]) / Delta

        k_geo = min(len(Y), max(self.cfg.min_pts_gp, min(self.cfg.use_quadratic_if + 5, len(Y))))
        fps = _improved_fps(Y, k_geo)
        Y, Fs, Cs = Y[fps], Fs[fps], Cs[fps]

        mask_rad = (norm(Y, axis=1) <= self.cfg.model_radius_mult)
        if np.any(mask_rad) and mask_rad.sum() >= self.cfg.min_pts_gp:
            Y, Fs, Cs = Y[mask_rad], Fs[mask_rad], Cs[mask_rad]

        if Y.shape[0] < self.cfg.min_pts_gp:
            self._improve_models_FL(center, Delta, budget=self.cfg.min_pts_gp - Y.shape[0])
            X, F, C = self.arrays()
            d = norm(X - center[None, :], axis=1)
            idx_close = np.argsort(d)[:max_pts]
            Xs, Fs = X[idx_close], F[idx_close]
            Cs = C[idx_close] if C.size else np.zeros((len(idx_close), self.m_ineq + self.m_eq))
            Y = (Xs - center[None, :]) / Delta
            fps = _improved_fps(Y, k_geo)
            Y, Fs, Cs = Y[fps], Fs[fps], Cs[fps]
            mask_rad = (norm(Y, axis=1) <= self.cfg.model_radius_mult)
            if np.any(mask_rad) and mask_rad.sum() >= self.cfg.min_pts_gp:
                Y, Fs, Cs = Y[mask_rad], Fs[mask_rad], Cs[mask_rad]

        # Objective interpolation works on shifted values (rhs = F - F0)
        rhs = Fs - Fs[0]

        # Fit objective g,H at scaled center (0)
        g, H, info = self.obj_model.fit_objective(Y, rhs, self.n, self.cfg)

        # Fit Jacobians (linear models) for constraints
        A_ineq = None
        A_eq = None
        Hc_list = None
        if self.m_ineq > 0:
            A_ineq = self._fit_jac(Y, Cs[:, :self.m_ineq]) / Delta
        if self.m_eq > 0:
            A_eq = self._fit_jac(Y, Cs[:, self.m_ineq:self.m_ineq + self.m_eq]) / Delta

        # Optionally diagonal quadratic for inequalities (conservative curvature)
        if self.m_ineq > 0 and Y.shape[0] >= self.cfg.use_quadratic_if:
            Hc_list = self._fit_constraints_diag_quadratic(Y, Cs[:, :self.m_ineq]) / (Delta**2)

        # Rescale g,H back to original (since Y = (X-center)/Delta)
        g = g / Delta
        H = H / (Delta**2)

        self.last_diag = {
            "deg": 2 if Y.shape[0] >= self.cfg.use_quadratic_if else 1,
            "n_pts": int(Y.shape[0]),
            "used_rbf": True,
            "rbf": info,
            "obj_info": info,
        }
        return FitResult(g=g, H=H, A_ineq=A_ineq, A_eq=A_eq, Hc_list=Hc_list, diag=self.last_diag)

    # -------- Criticality --------
    def criticality_measure(self, g, A_ineq, A_eq, cI0, cE0, eps) -> Tuple[float, np.ndarray]:
        cI0 = np.asarray(cI0).ravel() if (cI0 is not None and self.m_ineq > 0) else np.zeros(self.m_ineq)
        cE0 = np.asarray(cE0).ravel() if (cE0 is not None and self.m_eq > 0) else np.zeros(self.m_eq)
        A_mask, V_mask, _ = self._split_sets(cI0, eps)
        E_mask = (np.abs(cE0) <= eps) if cE0.size else np.zeros(0, dtype=bool)
        grad_mp1 = self._mp1_grad(g, A_ineq, V_mask)
        d = self._desc_dir_lp(grad_mp1, A_ineq, A_mask)
        lin = float(g @ d)
        pen = 0.0
        if A_ineq is not None and A_mask.size and np.any(A_mask):
            pen += float(self.cfg.mu * np.maximum(0.0, (A_ineq[A_mask] @ d)).sum())
        if A_eq is not None and E_mask.size and np.any(E_mask):
            pen += float(self.cfg.mu * np.abs(A_eq[E_mask] @ d).sum())
        sigma = float(-(lin + pen))
        return sigma, d

    # -------- Fitting utilities --------
    def _fit_jac(self, Y: np.ndarray, Csub: np.ndarray) -> np.ndarray:
        rhs = Csub - Csub[0:1, :]
        w = _distance_weights(Y, self.cfg.dist_w_power)
        sw = np.sqrt(np.clip(w, 1e-8, np.inf))
        Yw = Y * sw[:, None]
        A = Yw.T @ Yw + 1e-6 * np.eye(Y.shape[1])
        Jt = []
        for j in range(rhs.shape[1]):
            bj = rhs[:, j]
            mad = np.median(np.abs(bj - np.median(bj))) if bj.size else 0.0
            s_j = max(np.linalg.norm(bj), 1.4826 * mad, 1.0)
            bw = Yw.T @ ((bj / s_j) * sw)
            try:
                sol_scaled = np.linalg.solve(A, bw)
            except Exception:
                sol_scaled = lstsq(Yw, (bj / s_j) * sw, rcond=1e-10)[0]
            Jt.append(sol_scaled / s_j)
        return np.array(Jt)

    def _fit_constraints_diag_quadratic(self, Y: np.ndarray, Cineq: np.ndarray) -> np.ndarray:
        m, n = Y.shape
        nb = Cineq.shape[1] if Cineq.ndim == 2 else 0
        if nb == 0:
            return np.array([], dtype=object)
        Phi = np.hstack([Y, 0.5 * (Y ** 2)])
        Hc_list: List[np.ndarray] = []
        for i in range(nb):
            rhs = Cineq[:, i] - Cineq[0, i]
            w = _distance_weights(Y, self.cfg.dist_w_power)
            coef = _solve_weighted_ridge(Phi, rhs, w, self.cfg.ridge)
            d = coef[n:]
            Hc_list.append(np.diag(np.maximum(d, 0.0)))
        return np.array(Hc_list, dtype=object)

    def _fallback(self) -> FitResult:
        return FitResult(
            g=np.zeros(self.n),
            H=self.cfg.eig_floor * np.eye(self.n),
            A_ineq=np.zeros((self.m_ineq, self.n)) if self.m_ineq else None,
            A_eq=np.zeros((self.m_eq, self.n)) if self.m_eq else None,
            Hc_list=None,
            diag={"deg": 1, "n_pts": 0, "used_rbf": True, "fallback": "insufficient_points"},
        )

    # -------- Set splitting & grad mod --------
    def _split_sets(self, cI: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        A = np.abs(cI) <= eps   # almost active
        V = cI > eps            # clearly violated
        S = cI < -eps           # satisfied
        return A, V, S

    def _mp1_grad(self, g: np.ndarray, A_ineq: Optional[np.ndarray], Vmask: np.ndarray) -> np.ndarray:
        if A_ineq is None or not np.any(Vmask):
            return g.copy()
        return g + self.cfg.mu * A_ineq[Vmask].sum(axis=0)

    def _estimate_multipliers(self, grad_mp1: np.ndarray, A_ineq: Optional[np.ndarray], A_mask: np.ndarray) -> Optional[np.ndarray]:
        if A_ineq is None or not np.any(A_mask):
            return None
        A_act = A_ineq[A_mask]
        if A_act.size == 0:
            return None
        try:
            lam = lstsq(A_act.T, -grad_mp1, rcond=1e-10)[0]
        except Exception:
            return None
        if lam is None:
            return None
        if self.cfg.mult_project_lambda:
            mu = float(self.cfg.mu)
            lam = np.clip(lam, 0.0, mu)
        return lam

    def _effective_H_and_g(self, g: np.ndarray, H: np.ndarray,
                           A_ineq: Optional[np.ndarray], Hc_list: Optional[List[np.ndarray]],
                           lam: Optional[np.ndarray], A_mask: np.ndarray, V_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        g_eff = self._mp1_grad(g, A_ineq, V_mask)
        H_eff = H.copy()
        if A_ineq is not None and lam is not None and np.any(A_mask):
            act_idx = np.flatnonzero(A_mask)
            g_eff = g_eff + (lam @ A_ineq[act_idx])
            if Hc_list is not None and len(Hc_list) >= act_idx.size:
                for k, j in enumerate(act_idx):
                    Hj = Hc_list[j] if j < len(Hc_list) else None
                    if Hj is not None:
                        H_eff = H_eff + lam[k] * Hj
        try:
            w_eig, V = np.linalg.eigh(0.5 * (H_eff + H_eff.T))
            H_eff = (V * np.maximum(w_eig, self.cfg.eig_floor)) @ V.T
        except Exception:
            H_eff = 0.5 * (H_eff + H_eff.T) + self.cfg.eig_floor * np.eye(self.n)
        return H_eff, g_eff

    def _solve_box_newton(self, H: np.ndarray, g: np.ndarray, Delta: float) -> np.ndarray:
        n = g.size
        try:
            stepN = -np.linalg.solve(H, g)
        except Exception:
            d = -g
            Hd = H @ d
            denom = float(d @ Hd) if np.isfinite(d @ Hd) else 0.0
            alpha = (float(d @ d) / max(denom, 1e-16)) if denom > 0 else 1.0
            stepN = alpha * d
        if self.cfg.mult_newton_clip_inf:
            b = float(Delta)
            stepN = np.minimum(b, np.maximum(-b, stepN))
        infn = norm(stepN, ord=np.inf)
        if infn <= 1e-16:
            return -g / (norm(g) + 1e-16)
        return stepN

    # -------- Model value with exact penalty (incl. equalities) --------
    def _model_mp_value(self, t: float, d: np.ndarray,
                        g: np.ndarray, H: np.ndarray,
                        cI0: np.ndarray,
                        A_ineq: Optional[np.ndarray],
                        Hc_list: Optional[List[np.ndarray]],
                        cE0: Optional[np.ndarray],
                        A_eq: Optional[np.ndarray]) -> float:
        # Quadratic objective piece
        f_quad = 0.5 * t * t * (d @ (H @ d)) + t * (g @ d)

        pen = 0.0
        # Inequality exact penalty (piecewise linear, + optional conservative curvature)
        if A_ineq is not None and A_ineq.size:
            aTd = A_ineq @ d
            c_mid = cI0 + t * aTd
            pos_idx = np.flatnonzero(c_mid > 0.0)
            if pos_idx.size:
                pen += self.cfg.mu * float(np.sum(c_mid[pos_idx]))
            if Hc_list is not None:
                for idx in pos_idx:
                    if idx < len(Hc_list) and Hc_list[idx] is not None:
                        pen += 0.5 * self.cfg.mu * float(d @ (Hc_list[idx] @ d)) * (t * t)

        # Equality exact penalty (L1 on linearized equalities)
        if A_eq is not None and cE0 is not None and A_eq.size and cE0.size:
            e_lin = cE0 + t * (A_eq @ d)
            pen += self.cfg.mu * float(np.sum(np.abs(e_lin)))

        return f_quad + pen

    # -------- Breakpoints (ineq + eq) --------
    def _breakpoints_eq(self, cE0: np.ndarray, d: np.ndarray,
                        A_eq: Optional[np.ndarray], Delta: float) -> List[float]:
        t_list: List[float] = []
        if A_eq is None or A_eq.shape[0] == 0:
            return t_list
        infn = norm(d, ord=np.inf)
        if infn <= 1e-16:
            return t_list
        t_tr = Delta / infn
        den_floor = 1e-12
        aTd = A_eq @ d
        for i in range(A_eq.shape[0]):
            a = float(cE0[i])
            b = float(aTd[i])
            if abs(b) > den_floor:
                t = -a / b
                if 1e-12 < t <= t_tr and np.isfinite(t):
                    t_list.append(t)
        # de-duplicate and sort/prune
        t_list = sorted(set(t_list))
        pruned: List[float] = []
        for t in t_list:
            if not pruned or (t - pruned[-1]) > 1e-10:
                pruned.append(t)
        if len(pruned) > 64:
            pruned = pruned[:64]
        return pruned

    def _breakpoints(self, c0: np.ndarray, d: np.ndarray,
                     A_ineq: Optional[np.ndarray], Hc_list: Optional[List[np.ndarray]],
                     Delta: float,
                     cE0: Optional[np.ndarray] = None,
                     A_eq: Optional[np.ndarray] = None) -> List[float]:
        t_list: List[float] = []

        # Inequality breakpoints (where c_i + t a_i^T d crosses 0 or quadratic variant)
        if A_ineq is not None and A_ineq.shape[0] > 0:
            infn = norm(d, ord=np.inf)
            if infn > 1e-16:
                t_tr = Delta / infn
                den_floor = 1e-10
                min_sep = 1e-10
                aTd = A_ineq @ d
                for i in range(A_ineq.shape[0]):
                    a = float(c0[i])
                    b = float(aTd[i])
                    q = 0.0
                    if Hc_list is not None and i < len(Hc_list) and Hc_list[i] is not None:
                        try:
                            q = float(d @ (Hc_list[i] @ d))
                        except Exception:
                            q = 0.0
                    if abs(q) <= 1e-14:
                        if abs(b) > den_floor:
                            t = -a / b
                            if 1e-12 < t <= t_tr and np.isfinite(t):
                                t_list.append(t)
                    else:
                        A2, B2, C2 = 0.5 * q, b, a
                        disc = B2 * B2 - 4.0 * A2 * C2
                        if disc >= 0.0:
                            sdisc = np.sqrt(disc)
                            for t in ((-B2 - sdisc) / (2.0 * A2), (-B2 + sdisc) / (2.0 * A2)):
                                if 1e-12 < t <= t_tr and np.isfinite(t):
                                    t_list.append(float(t))

                # prune near-duplicates
                t_list = sorted(set(t_list))
                pruned = []
                for t in t_list:
                    if not pruned or (t - pruned[-1]) > min_sep:
                        pruned.append(t)
                t_list = pruned

        # Equality breakpoints (kinks for |c_E + t a_E d|)
        t_eq = self._breakpoints_eq(cE0 if cE0 is not None else np.zeros(0), d, A_eq, Delta)
        if t_eq:
            t_list = sorted(set(t_list).union(t_eq))

        if len(t_list) > 64:
            t_list = t_list[:64]
        return t_list

    # -------- VK correction --------
    def _vk_correction(self, A_ineq: Optional[np.ndarray], cI_h: Optional[np.ndarray],
                       A_eq: Optional[np.ndarray], cE_h: Optional[np.ndarray],
                       h: np.ndarray, Delta: float) -> np.ndarray:
        n = self.n
        if (A_ineq is None and A_eq is None):
            return np.zeros(n)
        v = np.zeros(n)
        box = Delta
        def proj_inf(z):
            lo = -box - h
            hi = box - h
            return np.minimum(hi, np.maximum(lo, z))
        prev_rn = np.inf
        for k in range(self.cfg.vk_gn_maxit):
            r_list = []
            J_list = []
            if A_ineq is not None and cI_h is not None and cI_h.size:
                mask = (cI_h + (A_ineq @ v)) > 0.0
                if np.any(mask):
                    r_list.append(cI_h[mask] + (A_ineq[mask] @ v))
                    J_list.append(A_ineq[mask])
            if A_eq is not None and cE_h is not None and cE_h.size:
                r_list.append(cE_h + (A_eq @ v))
                J_list.append(A_eq)
            if not J_list:
                break
            r = np.concatenate(r_list)
            J = np.vstack(J_list)
            rn = norm(r)
            if rn >= 0.99 * prev_rn and k >= 2:
                break
            prev_rn = rn
            try:
                step = -np.linalg.solve(J.T @ J + 1e-8 * np.eye(n), J.T @ r)
            except Exception:
                step = -lstsq(J, r, rcond=1e-10)[0]
            v_new = proj_inf(v + step)
            if norm(v_new - v) <= self.cfg.vk_gn_tol * max(1.0, norm(v)):
                v = v_new
                break
            v = v_new
        return v

    # -------- Violation measure --------
    def _violation_measure(self, cI: Optional[np.ndarray], cE: Optional[np.ndarray]) -> float:
        vI = float(np.maximum(0.0, cI).max()) if (cI is not None and cI.size) else 0.0
        vE = float(np.abs(cE).max()) if (cE is not None and cE.size) else 0.0
        return max(vI, vE)

    # -------- Oracle glue --------
    def set_oracle(self, oracle_fn):
        self.oracle = oracle_fn

    def _coerce_vec(self, v, m_expected: int, name: str) -> np.ndarray:
        if m_expected == 0:
            return np.zeros(0, dtype=float)
        if v is None:
            return np.zeros(m_expected, dtype=float)
        v = np.asarray(v, dtype=float).ravel()
        if v.size == 1 and m_expected == 1:
            return v.astype(float)
        if v.size != m_expected:
            raise ValueError(f"Oracle returned {name} of length {v.size}, expected {m_expected}.")
        return v

    def oracle_eval(self, x: np.ndarray):
        if not hasattr(self, "oracle") or self.oracle is None:
            raise RuntimeError("No oracle set. Call core.set_oracle(...) first.")
        out = self.oracle(np.asarray(x, float))
        if not isinstance(out, tuple):
            raise ValueError("Oracle must return a tuple (f, cI) or (f, cI, cE).")
        if len(out) == 2:
            f, cI = out
            cE = None
        elif len(out) == 3:
            f, cI, cE = out
        else:
            raise ValueError("Oracle must return (f, cI) or (f, cI, cE).")
        f = float(np.asarray(f, dtype=float))
        cI = self._coerce_vec(cI, self.m_ineq, "cI")
        cE = self._coerce_vec(cE, self.m_eq, "cE")
        return f, cI, cE

    # -------- Geometry helpers --------
    def _pick_working_subset(self, center: np.ndarray, Delta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X, F, C = self.arrays()
        if X.shape[0] == 0:
            return X, F, C
        max_pts = min(self.cfg.max_pts, X.shape[0])
        d = norm(X - center[None, :], axis=1)
        idx_close = np.argsort(d)[:max_pts]
        Xs, Fs = X[idx_close], F[idx_close]
        Cs = C[idx_close] if C.size else np.zeros((len(idx_close), self.m_ineq + self.m_eq))
        Delta = float(max(Delta, 1e-12))
        Y = (Xs - center[None, :]) / Delta
        k_geo = min(len(Y), max(self.cfg.min_pts_gp, min(self.cfg.use_quadratic_if + 5, len(Y))))
        fps = _improved_fps(Y, k_geo)
        Xs, Fs, Cs, Y = Xs[fps], Fs[fps], Cs[fps], Y[fps]
        mask_rad = (norm(Y, axis=1) <= self.cfg.model_radius_mult)
        if np.any(mask_rad) and mask_rad.sum() >= self.cfg.min_pts_gp:
            Xs, Fs, Cs = Xs[mask_rad], Fs[mask_rad], Cs[mask_rad]
        return Xs, Fs, Cs

    def _is_FL_enough(self, center: np.ndarray, Delta: float) -> bool:
        Xs, Fs, Cs = self._pick_working_subset(center, Delta)
        m = Xs.shape[0]
        if m < self.cfg.min_pts_gp:
            return False
        Y = (Xs - center[None, :]) / max(Delta, 1e-12)
        Phi, _, _ = _csv_basis(Y)
        p = Phi.shape[1]
        if m < p:
            Phi_lin = np.hstack([np.ones((m, 1)), Y])
            try:
                _, s_lin, _ = np.linalg.svd(Phi_lin, full_matrices=False)
                cond_lin = s_lin[0] / max(s_lin[-1], 1e-16)
                return cond_lin <= 1e8 and s_lin[-1] >= 1e-8
            except Exception:
                return False
        sel = _select_poised_rows(Phi, p)
        Phi_sq = Phi[sel, :]
        try:
            _, s, _ = np.linalg.svd(Phi_sq, full_matrices=False)
            cond = s[0] / max(s[-1], 1e-16)
            return (cond <= 1e8) and (s[-1] >= 1e-8)
        except Exception:
            return False
        
    # -------- Criticality loop --------
    def criticality_loop(self, xk: np.ndarray, eps: float, Delta_in: float) -> Tuple[float, float, float]:
        beta1 = float(self.cfg.crit_beta1)
        beta2 = float(self.cfg.crit_beta2)
        beta3 = float(self.cfg.crit_beta3)
        Delta = float(Delta_in)
        eps_c = float(eps)
        sigma_last = np.inf

        self._tm_seed_if_needed(xk)
        self._tm_maintain_geometry(xk, Delta)

        X, _, C = self.arrays()
        if X.shape[0]:
            j0 = int(np.argmin(np.linalg.norm(X - xk[None, :], axis=1)))
            cI0 = C[j0, :self.m_ineq] if self.m_ineq else np.zeros(0)
            cE0 = C[j0, self.m_ineq:self.m_ineq + self.m_eq] if self.m_eq else np.zeros(0)
        else:
            cI0 = np.zeros(self.m_ineq)
            cE0 = np.zeros(self.m_eq)

        for _ in range(20):
            if not self._is_FL_enough(xk, Delta):
                self._improve_models_FL(xk, Delta, budget=max(6, self.n + 1))
            fit = self.fit_local(center=xk, Delta=Delta)
            sigma, _ = self.criticality_measure(fit.g, fit.A_ineq, fit.A_eq, cI0, cE0, eps_c)
            sigma_last = float(max(sigma, 0.0))
            if Delta <= beta1 * max(sigma_last, 1e-16):
                break
            if cI0.size and (np.linalg.norm(cI0, ord=np.inf) > beta3 * max(sigma_last, 1e-16)):
                eps_c *= 0.5
            Delta *= 0.5
            if np.isfinite(sigma_last) and sigma_last > 0.0:
                Delta = min(Delta_in, max(Delta, beta2 * sigma_last))
        return float(Delta), float(eps_c), float(sigma_last)

    # -------- Direction LP --------
    def _desc_dir_lp(self, grad_mp1: np.ndarray, A_ineq: Optional[np.ndarray], Amask: np.ndarray) -> np.ndarray:
        n = grad_mp1.size
        if A_ineq is None or not np.any(Amask):
            g = grad_mp1
            return -g / (np.linalg.norm(g) + 1e-16)

        idx_all = np.flatnonzero(Amask)
        row_norms = np.linalg.norm(A_ineq[idx_all], axis=1)
        lp_max_active = int(getattr(self.cfg, "lp_max_active", 32))
        take = min(lp_max_active, idx_all.size)
        sel = idx_all[np.argsort(-row_norms)[:take]]
        mA = sel.size
        Ai = A_ineq[sel]
        mu = float(getattr(self.cfg, "mu", self.cfg.mu))

        # Try Gurobi if available
        try:
            import gurobipy as gp
            from gurobipy import GRB
            tl = float(getattr(self.cfg, "lp_time_limit", 0.10))
            verbose = bool(getattr(self.cfg, "lp_verbose", False))
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 1 if verbose else 0)
            env.setParam("TimeLimit", tl)
            env.setParam("Presolve", 2)
            env.setParam("Method", 1)
            env.start()
            model = gp.Model("dir_lp", env=env)
            d = model.addMVar(shape=n, lb=-1.0, ub=1.0, name="d")
            theta = model.addMVar(shape=mA, lb=0.0, name="theta")
            model.addConstr(Ai @ d - theta <= 0, name="active_rows")
            model.setObjective(grad_mp1 @ d + mu * theta.sum(), GRB.MINIMIZE)
            model.optimize()
            if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and d.X is not None:
                x_d = d.X
                if np.linalg.norm(x_d, ord=np.inf) > 1e-12:
                    return x_d
        except Exception:
            pass

        # HiGHS fallback
        try:
            c = np.concatenate([grad_mp1, mu * np.ones(mA)])
            bounds = [(-1, 1)] * n + [(0, None)] * mA
            A_ub = np.hstack([-Ai, np.eye(mA)])
            b_ub = np.zeros(mA)
            options = {
                "presolve": True,
                "time_limit": float(getattr(self.cfg, "lp_time_limit", 0.10)),
                "dual_feasibility_tolerance": 1e-9,
                "primal_feasibility_tolerance": 1e-9
            }
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs", options=options)
            if res.success and res.x is not None:
                x = res.x[:n]
                if np.linalg.norm(x, ord=np.inf) > 1e-12:
                    return x
        except Exception:
            pass

        # Steepest descent fallback
        g = grad_mp1
        return -g / (np.linalg.norm(g) + 1e-16)

    # -------- Step proposal --------
    def propose_step(self, x: np.ndarray, Delta: float, fit: FitResult, 
                        cI0: Optional[np.ndarray], cE0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
            g, H, A_ineq, A_eq, Hc_list = fit.g, fit.H, fit.A_ineq, fit.A_eq, fit.Hc_list
            cI0 = np.asarray(cI0, dtype=np.float64).ravel() if (cI0 is not None and self.m_ineq > 0) else np.zeros(self.m_ineq, dtype=np.float64)
            cE0 = np.asarray(cE0, dtype=np.float64).ravel() if (cE0 is not None and self.m_eq > 0) else np.zeros(self.m_eq, dtype=np.float64)
            eps = self.cfg.eps_active
            A_mask, V_mask, _ = self._split_sets(cI0, eps)
            E_mask = (np.abs(cE0) <= eps) if cE0.size else np.zeros(0, dtype=bool)
            grad_mp1 = self._mp1_grad(g, A_ineq, V_mask)
            d_default = self._desc_dir_lp(grad_mp1, A_ineq, A_mask)
            
            if norm(d_default, ord=np.inf) <= 1e-14:
                return np.zeros_like(d_default), {
                    "t": 0.0, "d": d_default, "breaks": [], "sigma": 0.0,
                    "pred_red_cons": 0.0, "pred_red_quad": 0.0, "Delta_out": float(Delta),
                    "used_multiplier_step": False, "is_FL": True
                }

            lin_term = float(g @ d_default)
            pen_term = 0.0
            if A_ineq is not None and np.any(A_mask):
                pen_term += float(self.cfg.mu * np.maximum(0.0, (A_ineq[A_mask] @ d_default)).sum())
            if A_eq is not None and np.any(E_mask):
                pen_term += float(self.cfg.mu * np.abs(A_eq[E_mask] @ d_default).sum())
            sigma = float(-(lin_term + pen_term))

            # Adaptive trust region radius
            if sigma < self.cfg.eps_c:
                sig = max(sigma, 1e-16)
                target1 = max(self.cfg.crit_beta1 * sig, 1e-12)
                target2 = self.cfg.crit_beta2 * sig
                Delta = min(Delta, target1, target2)

            breaks = [0.0] + self._breakpoints(cI0, d_default, A_ineq, Hc_list, Delta, cE0, A_eq)
            t_tr = Delta / max(norm(d_default, ord=np.inf), 1e-16)
            if not breaks or breaks[-1] < t_tr:
                breaks.append(t_tr)

            def mp_interval_value(t: float, d: np.ndarray) -> float:
                return self._model_mp_value(t, d, g, H, cI0, A_ineq, Hc_list, cE0, A_eq)

            t_best = 0.0
            val_best = 0.0
            for a, b in zip(breaks[:-1], breaks[1:]):
                t_mid = 0.5 * (a + b)
                val = mp_interval_value(t_mid, d_default)
                if t_best == 0.0 or val < val_best:
                    coef2 = float(d_default @ (H @ d_default))
                    slope_pen = 0.0
                    if A_ineq is not None and A_ineq.size:
                        aTd_all = A_ineq @ d_default
                        c_mid = cI0 + t_mid * aTd_all
                        pos_mask = np.flatnonzero(c_mid > 0.0)
                        if pos_mask.size:
                            slope_pen += self.cfg.mu * float(np.sum(aTd_all[pos_mask]))
                        if Hc_list is not None:
                            for idx in pos_mask:
                                if idx < len(Hc_list) and Hc_list[idx] is not None:
                                    slope_pen += self.cfg.mu * float(d_default @ (Hc_list[idx] @ d_default)) * t_mid
                    if A_eq is not None and cE0 is not None and A_eq.size and cE0.size:
                        aTdE = A_eq @ d_default
                        e_mid = cE0 + t_mid * aTdE
                        slope_pen += self.cfg.mu * float(np.sum(np.sign(e_mid) * aTdE))
                    coef1 = float(g @ d_default) + slope_pen
                    t_star = t_mid
                    if abs(coef2) > 1e-16:
                        t_star = float(np.clip(-coef1 / coef2, a + 1e-12, b - 1e-12))
                    for t in (t_mid, t_star):
                        vv = mp_interval_value(t, d_default)
                        if t_best == 0.0 or vv < val_best:
                            t_best, val_best = t, vv

            h_default = t_best * d_default
            cI_h = cI0 + (A_ineq @ h_default) if (A_ineq is not None and h_default.size) else None
            cE_h = cE0 + (A_eq @ h_default) if (A_eq is not None and h_default.size) else None
            v_default = self._vk_correction(A_ineq, cI_h, A_eq, cE_h, h_default, Delta)
            s_default = h_default + v_default
            pred_red_cons = self._conservative_pred_red(sigma=sigma, Delta=Delta, n=self.n)
            pred_red_quad_default = max(1e-16, -(g @ s_default) - 0.5 * (s_default @ (H @ s_default)))

            # Augmented Lagrangian step
            used_mult = False
            s_choice = s_default
            pred_red_quad_choice = pred_red_quad_default
            s_mult = None  # Initialize s_mult to None
            if self.cfg.use_multiplier_step and sigma <= self.cfg.mult_sigma_thresh:
                lam = self._estimate_multipliers(grad_mp1, A_ineq, A_mask)
                if lam is not None and np.all(lam >= -1e-12) and np.all(lam <= self.cfg.mu + 1e-12):
                    H_eff, g_eff = self._effective_H_and_g(g, H, A_ineq, Hc_list, lam, A_mask, V_mask)
                    h_mult = self._solve_box_newton(H_eff, g_eff, Delta)
                    cI_hm = cI0 + (A_ineq @ h_mult) if (A_ineq is not None and h_mult.size) else None
                    cE_hm = cE0 + (A_eq @ h_mult) if (A_eq is not None and h_mult.size) else None
                    v_mult = self._vk_correction(A_ineq, cI_hm, A_eq, cE_hm, h_mult, Delta)
                    s_mult = h_mult + v_mult
                    mp_def = mp_interval_value(1.0, s_default)
                    mp_mul = mp_interval_value(1.0, s_mult)
                    if mp_mul < mp_def:
                        s_choice = s_mult
                        used_mult = True
                        pred_red_quad_choice = max(1e-16, -(g @ s_mult) - 0.5 * (s_mult @ (H @ s_mult)))

            # Feasibility decrease check
            theta0 = self._violation_measure(cI0, cE0)
            cI_def = cI0 + (A_ineq @ s_default) if (A_ineq is not None and s_default is not None and s_default.size) else None
            cE_def = cE0 + (A_eq @ s_default) if (A_eq is not None and s_default is not None and s_default.size) else None
            theta_def = self._violation_measure(cI_def, cE_def)
            theta_mult = None
            if s_mult is not None:  # Safe check since s_mult is initialized
                cI_mul = cI0 + (A_ineq @ s_mult) if (A_ineq is not None and s_mult.size) else None
                cE_mul = cE0 + (A_eq @ s_mult) if (A_eq is not None and s_mult.size) else None
                theta_mult = self._violation_measure(cI_mul, cE_mul)
            theta_choice = theta_mult if (used_mult and theta_mult is not None) else theta_def
            tol = float(self.cfg.fl_tol)
            is_FL = (theta_choice <= theta0 * (1.0 - tol)) or (theta_choice <= theta0 + 1e-16)

            return s_choice, {
                "t": float(t_best),
                "d": d_default,
                "breaks": breaks,
                "sigma": float(sigma),
                "pred_red_cons": float(pred_red_cons),
                "pred_red_quad": float(pred_red_quad_choice),
                "Delta_out": float(Delta),
                "used_multiplier_step": bool(used_mult),
                "is_FL": bool(is_FL),
            }
    # -------- Conservative predicted reduction --------
    def _conservative_pred_red(self, sigma: float, Delta: float, n: int) -> float:
        kfd, kH, kc = self.cfg.k_fd, self.cfg.k_H, self.cfg.k_c
        piece = min(sigma / max(n * n * kH, 1e-16),
                    self.cfg.eps_c / max(n * kc, 1e-16),
                    Delta)
        return max(1e-16, 0.5 * kfd * sigma * piece)
