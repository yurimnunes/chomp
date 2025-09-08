from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import gpytorch
import numpy as np
import torch
from numpy.linalg import lstsq, norm
from scipy.linalg import cholesky, solve_triangular


# Modified DFOConfig
@dataclass
class DFOConfig:
    huber_delta: Optional[float] = None
    ridge: float = 1e-6
    dist_w_power: float = 0.3
    eig_floor: float = 1e-8
    max_pts: int = 200
    model_radius_mult: float = 2.0
    use_quadratic_if: int = 25
    use_full_quadratic_if: int = 40
    mu: float = 10.0
    eps_active: float = 1e-6
    tr_inc: float = 1.6
    tr_dec: float = 0.5
    eta0: float = 0.05
    eta1: float = 0.25
    crit_beta1: float = 0.5
    crit_beta2: float = 2.0
    crit_beta3: float = 0.5
    eps_c: float = 1e-4
    k_fd: float = 1.0
    k_H: float = 1.0
    k_c: float = 1.0
    vk_gn_maxit: int = 10
    vk_gn_tol: float = 1e-8
    lp_max_active: int = 32
    lp_time_limit: float = 0.10
    lp_verbose: bool = False
    use_multiplier_step: bool = True
    mult_sigma_thresh: float = 1e-3
    mult_project_lambda: bool = True
    capu_enabled: bool = True
    mult_newton_clip_inf: bool = True
    fl_tol: float = 1e-3
    gp_noise: float = 1e-2  # Noise level for GP likelihood
    gp_max_pts: int = 120  # Max points for GP
    min_pts_gp: int = 6  # Min points for GP
    acquisition: str = "ei"  # "ei" or "logei"
    xi: float = 0.01  # Exploration parameter for EI/LogEI
    gp_device: str = "cuda"  # "cuda" | "cpu" | "auto"
    gp_dtype: str = "float32"  # "float32" | "float64"
    gp_lr: float = 0.06
    gp_train_steps: int = 25
    gp_patience: int = 6
    gp_warm_start: bool = True
    gp_freeze_after: int = (
        2  # after this many successful fit rounds, stop re-optimizing
    )
    gp_hess_mode: str = "autograd"  # "diag_from_lengthscale" | "autograd"
    min_cand: int = 100  # min candidate points for acquisition optimization
    capu_history_len: int = 6  # length of history for CAPU
    capu_viol_eps: float = 1e-3  # tolerance for CAPU violation
    capu_persist_thresh: int = 3  # iters of violation before CAPU triggers


# Existing FitResult and CriticalityState (unchanged)
@dataclass
class FitResult:
    g: np.ndarray
    H: np.ndarray
    A_ineq: Optional[np.ndarray]
    A_eq: Optional[np.ndarray]
    Hc_list: Optional[List[np.ndarray]]
    diag: Dict


@dataclass
class CriticalityState:
    eps: float
    Delta_bct: float
    sigma_eps: float = 1.0
    Delta_eps: float = 1.0


# Existing helper functions (unchanged)
def _huber_weights(r: np.ndarray, delta: Optional[float]) -> np.ndarray:
    if r.size == 0:
        return np.ones_like(r)
    a = np.abs(r)
    if delta is None:
        mad = np.median(a)
        delta = max(1.5 * mad, np.percentile(a, 75)) if np.any(a > 0) else 1.0
    w = np.ones_like(a)
    mask = a > delta
    if np.any(mask):
        w[mask] = delta / np.maximum(a[mask], 1e-16)
    return w


def _distance_weights(Y: np.ndarray, p: float) -> np.ndarray:
    if Y.size == 0:
        return np.ones(0)
    rn = np.linalg.norm(Y, axis=1)
    return 1.0 / (1.0 + rn) ** p


def _solve_weighted_ridge(
    Phi: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float
) -> np.ndarray:
    if Phi.size == 0:
        return np.zeros(0)
    sw = np.sqrt(np.clip(w, 1e-12, np.inf))
    Pw = Phi * sw[:, None]
    yw = y * sw
    A = Pw.T @ Pw + lam * np.eye(Phi.shape[1])
    b = Pw.T @ yw
    try:
        L = cholesky(A, lower=True)
        z = solve_triangular(L, b, lower=True)
        x = solve_triangular(L.T, z, lower=False)
        return x
    except Exception:
        try:
            return np.linalg.solve(A, b)
        except Exception:
            return lstsq(Pw, yw, rcond=1e-10)[0]


from typing import Union

import numpy as np


def _improved_fps(Y: np.ndarray, k: int) -> np.ndarray:
    """
    Perform Farthest Point Sampling (FPS) to select k points from Y maximizing their spread.

    Args:
        Y (np.ndarray): Input matrix of shape (m, n), where m is the number of points
                       and n is the number of features.
        k (int): Number of points to select (must be positive and <= m).

    Returns:
        np.ndarray: Array of k (or fewer) indices of selected points, sorted in order of selection.

    Raises:
        ValueError: If Y is not a 2D NumPy array, k is non-positive, or k exceeds m.
    """
    # Input validation
    if not isinstance(Y, np.ndarray) or Y.ndim != 2:
        raise ValueError("Y must be a 2D NumPy array")
    m, _ = Y.shape
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if m == 0:
        return np.array([], dtype=np.int64)
    if k >= m:
        return np.arange(m, dtype=np.int64)

    # Initialize
    # Start with the point closest to the origin (min ||y||^2)
    d2 = np.einsum("ij,ij->i", Y, Y)  # Squared norms of all points
    start = np.argmin(d2)
    sel = [start]
    mask = np.ones(m, dtype=bool)
    mask[start] = False
    d2 = np.full(
        m, np.inf, dtype=Y.dtype
    )  # Track min squared distance to selected points
    d2[start] = 0.0
    last = Y[start : start + 1, :]  # Shape (1, n)

    # Iteratively select k points
    for _ in range(1, k):
        # Update squared distances to the last selected point
        diff = Y - last  # Broadcasting: (m, n) - (1, n)
        d2 = np.minimum(d2, np.einsum("ij,ij->i", diff, diff))

        # Select the farthest remaining point
        idx = np.argmax(np.where(mask, d2, -np.inf))
        if not np.isfinite(d2[idx]):  # Degenerate case (e.g., duplicate points)
            break
        sel.append(idx)
        mask[idx] = False
        last = Y[idx : idx + 1, :]

    return np.array(sel, dtype=np.int64)


from typing import Tuple

import numpy as np


def _csv_basis(Y: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Compute a polynomial basis (constant, linear, and quadratic terms) for input matrix Y.

    Args:
        Y (np.ndarray): Input matrix of shape (m, n), where m is the number of samples
                        and n is the number of features.

    Returns:
        Tuple[np.ndarray, int, int]: A tuple containing:
            - Phi (np.ndarray): Basis matrix of shape (m, 1 + n + n*(n+1)//2) with
                               constant, linear, and quadratic terms.
            - n (int): Number of features in Y.
            - nquad (int): Number of quadratic terms.

    Raises:
        ValueError: If Y is not a 2D NumPy array or has invalid dimensions.
    """
    # Input validation
    if not isinstance(Y, np.ndarray) or Y.ndim != 2:
        raise ValueError("Y must be a 2D NumPy array")

    m, n = Y.shape
    if m == 0 or n == 0:
        raise ValueError("Y must have non-zero dimensions")

    # Constant term: ones vector of shape (m, 1)
    ones = np.ones((m, 1), dtype=Y.dtype)

    # Linear terms: Y itself
    lin = Y

    # Quadratic terms: compute Y[:, i] * Y[:, j] for i <= j efficiently
    # Create indices for upper triangular matrix (including diagonal)
    i, j = np.triu_indices(n)
    # Compute quadratic terms using broadcasting
    quads = Y[:, i] * Y[:, j]

    # Combine all terms into the basis matrix Phi
    Phi = np.hstack([ones, lin, quads])

    # Number of quadratic terms: n*(n+1)//2
    nquad = n * (n + 1) // 2

    return Phi, n, nquad


from typing import Union

import numpy as np
from scipy.linalg import svd


def _select_poised_rows(Phi: np.ndarray, k: int) -> np.ndarray:
    """
    Select the top k rows of matrix Phi based on leverage scores from SVD or row norms.

    Args:
        Phi (np.ndarray): Input matrix of shape (m, n), where m is the number of rows
                         and n is the number of columns.
        k (int): Number of rows to select (must be positive and <= m).

    Returns:
        np.ndarray: Sorted indices of the top k rows, based on leverage scores (from SVD)
                    or row norms (if SVD fails).

    Raises:
        ValueError: If Phi is not a 2D NumPy array, k is non-positive, or k exceeds m.
    """
    # Input validation
    if not isinstance(Phi, np.ndarray) or Phi.ndim != 2:
        raise ValueError("Phi must be a 2D NumPy array")
    m, _ = Phi.shape
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if k > m:
        return np.arange(m, dtype=np.int64)

    try:
        # Compute SVD with SciPy for efficiency
        U, _, _ = svd(Phi, full_matrices=False, lapack_driver="gesvd")
        # Leverage scores: sum of squared entries of U along rows
        lev = np.sum(U**2, axis=1)
    except np.linalg.LinAlgError:
        # Fallback to row norms if SVD fails
        lev = np.sum(Phi**2, axis=1)  # Equivalent to np.linalg.norm(Phi, axis=1)**2

    # Select top k indices based on leverage scores (descending order)
    idx = np.argsort(-lev)[:k]
    return np.sort(idx)


# Acquisition functions
def expected_improvement(
    gp_model, likelihood, X: torch.Tensor, y: torch.Tensor, xi: float = 0.01
) -> torch.Tensor:
    """
    Compute Expected Improvement (EI) for a GP model.
    Args:
        gp_model: Trained GPyTorch model.
        likelihood: GPyTorch likelihood.
        X: Candidate points (n_samples, n_dims).
        y: Observed outputs (n_samples,).
        xi: Exploration parameter.
    Returns:
        EI values for each candidate point.
    """
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(gp_model(X))
        mu = observed_pred.mean
        sigma = observed_pred.variance.sqrt()
        y_best = y.min()
        z = (y_best - mu - xi) / sigma
        ei = (y_best - mu - xi) * torch.distributions.Normal(0, 1).cdf(
            z
        ) + sigma * torch.distributions.Normal(0, 1).log_prob(z).exp()
        ei = torch.clamp(ei, min=0.0)
    return ei


def log_expected_improvement(
    gp_model, likelihood, X: torch.Tensor, y: torch.Tensor, xi: float = 0.01
) -> torch.Tensor:
    """
    Compute Log Expected Improvement (LogEI).
    Args:
        gp_model: Trained GPyTorch model.
        likelihood: GPyTorch likelihood.
        X: Candidate points (n_samples, n_dims).
        y: Observed outputs (n_samples,).
        xi: Exploration parameter.
    Returns:
        LogEI values for each candidate point.
    """
    ei = expected_improvement(gp_model, likelihood, X, y, xi)
    return torch.log(ei + 1e-10)
