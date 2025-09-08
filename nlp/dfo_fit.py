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
