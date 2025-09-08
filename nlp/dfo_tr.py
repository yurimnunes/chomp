import logging
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
from scipy.linalg import qr
from scipy.spatial.distance import cdist
from scipy.stats import qmc  # For Sobol sequences

from .dfo_aux import *
from .dfo_aux import _csv_basis


class TRModel:
    def __init__(self, n: int, q: int, cfg: Optional[DFOConfig] = None):
        """
        Initialize a Trust Region Model for derivative-free optimization.

        Args:
            n (int): Number of input dimensions.
            q (int): Number of output dimensions.
            cfg (DFOConfig, optional): Configuration for the trust region model.
        """
        self.n = n
        self.q = q
        self.cfg = cfg or DFOConfig()
        self.center = np.zeros(n, dtype=np.float64)
        self.radius = 1.0
        self.pointsAbs = np.zeros((0, n), dtype=np.float64)
        self.fvals = np.zeros((0, q), dtype=np.float64)
        self.interp: Optional[DFOGPModel] = None
        self.min_pts = max(n + 1, self.cfg.min_pts_gp)
        self.max_pts = self.cfg.gp_max_pts

    def append_raw_sample(self, x: np.ndarray, fvec: np.ndarray) -> None:
        """
        Append a new sample point and its function values to the model.

        Args:
            x (np.ndarray): Input point of shape (n,).
            fvec (np.ndarray): Function values of shape (q,).
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        fvec = np.asarray(fvec, dtype=np.float64).ravel().reshape(1, -1)
        
        # Validate fvec dimensions
        if self.fvals.size and fvec.shape[1] != self.fvals.shape[1]:
            logging.error(f"Dimension mismatch: fvec has {fvec.shape[1]} columns, expected {self.fvals.shape[1]}")
            raise ValueError(f"fvec dimension {fvec.shape[1]} does not match self.fvals dimension {self.fvals.shape[1]}")
        
        if self.pointsAbs.shape[0] > 0:
            if np.any(norm(self.pointsAbs - x[None, :], axis=1) < 1e-12):
                return
        self.pointsAbs = np.vstack([self.pointsAbs, x]) if self.pointsAbs.size else x[None, :]
        self.fvals = np.vstack([self.fvals, fvec]) if self.fvals.size else fvec
        if self.pointsAbs.shape[0] > self.max_pts * 1.5:
            self.rebuildModel(None)
            
    def ensure_basis_initialized(self) -> None:
        """Initialize or rebuild the interpolation model."""
        self.rebuild_interp()

    def rebuild_interp(self) -> None:
        """Rebuild the Gaussian Process interpolation model."""
        if self.pointsAbs.shape[0] < self.min_pts:
            self.interp = None
            return
        Y = (self.pointsAbs - self.center[None, :]) / max(self.radius, 1e-12)
        try:
            self.interp = DFOGPModel()
            self.interp.fit_objective(Y, self.fvals[:, 0], self.n, self.cfg)
        except Exception:
            self.interp = None

    def hasDistantPoints(self, opts) -> bool:
        """Check if any points are outside 1.5 * radius from the center."""
        if self.pointsAbs.shape[0] == 0:
            return False
        dist = norm(self.pointsAbs - self.center[None, :], axis=1)
        return np.any(dist > 1.5 * self.radius)

    def isOld(self, opts) -> bool:
        """Check if the model has too many points."""
        return self.pointsAbs.shape[0] > self.max_pts


    def rebuildModel(self, opts: Any) -> None:
        """
        Rebuild the model by selecting a well-conditioned subset of max_pts points using pivoted QR decomposition
        on the polynomial basis matrix for improved poisedness.

        Args:
            opts: Options for the trust region algorithm (not used in this implementation).
        """
        if self.pointsAbs.shape[0] == 0:
            return

        # Compute normalized points and polynomial basis
        Y = (self.pointsAbs - self.center[None, :]) / max(self.radius, 1e-12)
        Phi, _, _ = _csv_basis(Y)

        try:
            # Use pivoted QR on Phi.T to select rows (points) that maximize the conditioning of the basis
            _, _, P = qr(Phi.T, pivoting=True, mode='economic')
            idx = P[:self.max_pts]
        except np.linalg.LinAlgError:
            # Fallback to distance-based selection if QR fails
            dist = norm(self.pointsAbs - self.center[None, :], axis=1)
            idx = np.argsort(dist)[:self.max_pts]

        # Sort indices for consistency (optional, but matches original behavior)
        idx = np.sort(idx)

        self.pointsAbs = self.pointsAbs[idx]
        self.fvals = self.fvals[idx]
        self.rebuild_interp()

    def isLambdaPoised(self, opts: Any) -> bool:
        """Check if the interpolation set is lambda-poised."""
        if self.interp is None or self.pointsAbs.shape[0] < self.min_pts:
            return False
        Phi, _, _ = _csv_basis((self.pointsAbs - self.center[None, :]) / max(self.radius, 1e-12))
        try:
            _, s, _ = np.linalg.svd(Phi, full_matrices=False)
            return s[-1] >= 1e-8 and (s[0] / s[-1]) <= 1e8
        except Exception:
            return False
        
    def ensure_improvement(self, call_fn: Callable[[np.ndarray], Tuple[np.ndarray, bool]], 
                          arg1, arg2, opts) -> None:
        """
        Add a new point to improve the trust region model using an acquisition function or distance-based selection.

        Args:
            call_fn (Callable): Function to evaluate candidate points, returning (fvec, success).
            arg1, arg2: Additional arguments for call_fn (not used in this implementation).
            opts: Options for the trust region algorithm (not used in this implementation).

        Raises:
            ValueError: If the model has too many points or inputs are invalid.
        """
        if self.pointsAbs.shape[0] >= self.max_pts:
            return

        # Generate candidate points
        num_cand = max(50, 10 * self.n)
        cand = []

        # Axis-aligned points
        for i in range(self.n):
            e = np.zeros(self.n, dtype=np.float64)
            e[i] = 1.0
            cand.append(self.center + self.radius * e)
            cand.append(self.center - self.radius * e)

        # Quasi-Monte Carlo points (Sobol sequence)
        sampler = qmc.Sobol(d=self.n, scramble=True, seed=np.random.randint(0, 10000))
        sobol_points = sampler.random(num_cand - 2 * self.n)
        # Scale to trust region ball
        sobol_points = qmc.scale(sobol_points, -self.radius, self.radius) + self.center
        cand.extend(sobol_points)
        cand = np.unique(np.array(cand, dtype=np.float64), axis=0)

        # Select new point
        if self.interp and hasattr(self.interp, "get_acquisition"):
            # Use Expected Improvement (EI) with exploration-exploitation trade-off
            try:
                acq_vals = self.interp.get_acquisition(
                    cand, acquisition="ei", xi=self.cfg.xi
                )
                j = np.argmax(acq_vals)
            except Exception:
                # Fallback to distance-based selection if acquisition fails
                j = self._distance_based_selection(cand)
        else:
            # Fallback to distance-based selection
            j = self._distance_based_selection(cand)

        # Evaluate and append the new point
        x_new = cand[j]
        fvec, ok = call_fn(x_new)
        if ok:
            self.append_raw_sample(x_new, fvec)
            self.rebuild_interp()

    def _distance_based_selection(self, cand: np.ndarray) -> int:
        """
        Select a candidate point by maximizing the minimum distance to existing points.

        Args:
            cand (np.ndarray): Candidate points of shape (num_cand, n).

        Returns:
            int: Index of the selected candidate point.
        """
        if self.pointsAbs.size:
            D = cdist(cand, self.pointsAbs, metric='euclidean')
            minD = np.min(D, axis=1)
        else:
            minD = np.full(len(cand), self.radius, dtype=np.float64)
        return np.argmax(minD)

    def change_tr_center(self, x_new: np.ndarray, fvec: np.ndarray, opts) -> None:
        """
        Update the trust region center and append the new point.

        Args:
            x_new (np.ndarray): New center point of shape (n,).
            fvec (np.ndarray): Function value at x_new of shape (q,).
            opts: Options for the trust region algorithm.
        """
        self.center = np.asarray(x_new, dtype=np.float64).ravel()
        self.append_raw_sample(x_new, fvec)
        self.rebuild_interp()