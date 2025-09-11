# improved_byrd_omojokun_tr.py
# State-of-the-art Trust Region manager with Byrd–Omojokun split
# Improvements: better numerics, cleaner API, modern algorithmic enhancements

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

MatLike = Union[np.ndarray, sp.spmatrix, spla.LinearOperator]
Vec = np.ndarray


# ---------------------------- Configuration ---------------------------- #
@dataclass
class TRConfig:
    # Trust region parameters
    delta0: float = 1.0
    delta_min: float = 1e-12
    delta_max: float = 1e6
    eta1: float = 0.1  # accept threshold
    eta2: float = 0.9  # expand threshold
    gamma1: float = 0.5  # shrink factor
    gamma2: float = 2.0  # expand factor

    # Byrd-Omojokun split
    zeta: float = 0.8  # fraction of radius for normal step

    # Solver tolerances
    cg_tol: float = 1e-8
    cg_maxiter: int = 200
    neg_curv_tol: float = 1e-14
    constraint_tol: float = 1e-8

    # Adaptive features
    adaptive_zeta: bool = True
    curvature_aware: bool = True
    feasibility_emphasis: bool = True

    # Numerical stability
    rcond: float = 1e-12
    reg_floor: float = 1e-10


class TRStatus(Enum):
    SUCCESS = "success"
    BOUNDARY = "boundary"
    NEG_CURV = "negative_curvature"
    MAX_ITER = "max_iterations"
    INFEASIBLE = "infeasible"


# ---------------------------- Utilities ---------------------------- #
def safe_norm(x: Vec) -> float:
    """Numerically safe norm computation."""
    return float(np.linalg.norm(x)) if x.size > 0 else 0.0


def make_operator(A: MatLike, n: int) -> spla.LinearOperator:
    """Convert matrix-like object to LinearOperator."""
    if isinstance(A, spla.LinearOperator):
        return A
    if callable(A):
        return spla.LinearOperator((n, n), matvec=A, dtype=float)
    return spla.aslinearoperator(A)


def solve_kkt_system(
    H: np.ndarray,
    A: Optional[np.ndarray],
    g: Vec,
    b: Optional[Vec] = None,
    reg: float = 1e-10,
) -> Tuple[Vec, Vec]:
    """
    Solve KKT system using stable block elimination:
    [H  A^T] [p]   [-g]
    [A   0 ] [λ] = [-b]
    """
    n = H.shape[0]
    if A is None or A.size == 0:
        try:
            p = la.solve(H + reg * np.eye(n), -g, assume_a="pos")
            return p, np.array([])
        except la.LinAlgError:
            p = la.lstsq(H + reg * np.eye(n), -g)[0]
            return p, np.array([])

    m = A.shape[0]
    b = np.zeros(m) if b is None else b

    # Block elimination: solve (A H^{-1} A^T) λ = A H^{-1} g - b
    try:
        Hinv_g = la.solve(H + reg * np.eye(n), -g, assume_a="pos")
        Hinv_AT = la.solve(H + reg * np.eye(n), A.T, assume_a="pos")

        S = A @ Hinv_AT  # Schur complement
        rhs = A @ Hinv_g - b

        lam = la.solve(S + reg * np.eye(m), rhs)
        p = Hinv_g - Hinv_AT @ lam

        return p, lam
    except la.LinAlgError:
        # Fallback to least squares
        K = np.block([[H + reg * np.eye(n), A.T], [A, np.zeros((m, m))]])
        rhs = np.concatenate([-g, -b])
        sol = la.lstsq(K, rhs)[0]
        return sol[:n], sol[n:]


def nullspace_basis(A: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """Compute orthonormal nullspace basis via SVD."""
    if A.size == 0:
        return np.eye(A.shape[1] if A.ndim > 1 else 0)

    U, s, VT = la.svd(A, full_matrices=False)
    tol = rcond * (s[0] if s.size > 0 else 1.0)
    rank = np.sum(s > tol)

    if rank >= A.shape[1]:
        return np.zeros((A.shape[1], 0))

    return VT[rank:].T


def dogleg_step(g: Vec, Hg: Vec, gnorm: float, Delta: float) -> Tuple[Vec, TRStatus]:
    """
    Improved dogleg with better conditioning.
    Solves: min g^T p + 0.5 p^T H p  s.t. ||p|| <= Delta
    """
    n = g.size
    if gnorm <= 1e-14:
        return np.zeros(n), TRStatus.SUCCESS

    # Cauchy point
    gHg = np.dot(g, Hg)
    if gHg <= 1e-14:  # Non-positive curvature
        p_cauchy = -(Delta / gnorm) * g
        return p_cauchy, TRStatus.NEG_CURV

    alpha_cauchy = gnorm**2 / gHg
    p_cauchy = -alpha_cauchy * g

    if safe_norm(p_cauchy) >= Delta:
        return -(Delta / gnorm) * g, TRStatus.BOUNDARY

    # Newton step
    try:
        p_newton = la.solve(
            0.5
            * (
                Hg.reshape(-1, 1) @ g.reshape(1, -1)
                + g.reshape(-1, 1) @ Hg.reshape(1, -1)
            ),
            -g,
        )
        if safe_norm(p_newton) <= Delta:
            return p_newton, TRStatus.SUCCESS
    except (la.LinAlgError, np.linalg.LinAlgError):
        pass

    # Dogleg between Cauchy and boundary
    s = p_newton - p_cauchy if "p_newton" in locals() else -p_cauchy
    a = np.dot(s, s)
    b = 2 * np.dot(p_cauchy, s)
    c = np.dot(p_cauchy, p_cauchy) - Delta**2

    disc = max(0.0, b**2 - 4 * a * c)
    if a > 1e-14:
        tau = (-b + np.sqrt(disc)) / (2 * a)
        tau = np.clip(tau, 0.0, 1.0)
    else:
        tau = 1.0

    return p_cauchy + tau * s, TRStatus.BOUNDARY


def steihaug_cg(
    H_op: spla.LinearOperator,
    g: Vec,
    Delta: float,
    tol: float = 1e-8,
    maxiter: int = 200,
) -> Tuple[Vec, TRStatus, int]:
    """
    Steihaug-Toint CG with improved boundary handling.
    """
    n = g.size
    p = np.zeros(n)
    r = -g.copy()
    d = r.copy()

    rTr = np.dot(r, r)
    if np.sqrt(rTr) <= tol:
        return p, TRStatus.SUCCESS, 0

    for k in range(maxiter):
        Hd = H_op.matvec(d)
        dTHd = np.dot(d, Hd)

        # Check for negative curvature
        if dTHd <= 1e-14 * max(1.0, np.dot(d, d)):
            # Find boundary intersection
            tau = _boundary_intersection(p, d, Delta)
            return p + tau * d, TRStatus.NEG_CURV, k

        alpha = rTr / dTHd
        p_next = p + alpha * d

        # Check trust region boundary
        if safe_norm(p_next) >= Delta:
            tau = _boundary_intersection(p, d, Delta)
            return p + tau * d, TRStatus.BOUNDARY, k

        r_next = r - alpha * Hd
        rTr_next = np.dot(r_next, r_next)

        if np.sqrt(rTr_next) <= tol:
            return p_next, TRStatus.SUCCESS, k + 1

        beta = rTr_next / rTr
        d = r_next + beta * d

        p, r, rTr = p_next, r_next, rTr_next

    return p, TRStatus.MAX_ITER, maxiter


def _boundary_intersection(p: Vec, d: Vec, Delta: float) -> float:
    """Find tau such that ||p + tau*d|| = Delta."""
    pTp = np.dot(p, p)
    pTd = np.dot(p, d)
    dTd = np.dot(d, d)

    if dTd <= 1e-14:
        return 0.0

    disc = max(0.0, pTd**2 - dTd * (pTp - Delta**2))
    return (-pTd + np.sqrt(disc)) / dTd


# ---------------------------- Main TR Manager ---------------------------- #
class TrustRegionManager:
    """
    Modern Trust Region manager with Byrd-Omojokun decomposition.

    Key improvements:
    - Cleaner, more focused API
    - Better numerical stability
    - Adaptive parameter selection
    - Efficient constraint handling
    """

    def __init__(self, config: Optional[TRConfig] = None):
        self.cfg = TRConfig()
        self.delta = self.cfg.delta0
        self.rejection_count = 0

        # Adaptive state
        self._rho_history = []
        self._feasibility_history = []
        self._curvature_estimate = None

    def solve(
        self,
        H: MatLike,
        g: Vec,
        A_ineq: Optional[np.ndarray] = None,
        b_ineq: Optional[Vec] = None,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[Vec] = None,
    ) -> Tuple[Vec, Dict[str, Any]]:
        """
        Solve trust region subproblem with mixed constraints.

        min   g^T p + 0.5 p^T H p
        s.t.  A_eq p + b_eq = 0
              A_ineq p + b_ineq <= 0
              ||p|| <= delta
        """
        if A_eq is not None and A_eq.size > 0:
            return self._solve_with_equality_constraints(
                H, g, A_eq, b_eq, A_ineq, b_ineq
            )
        elif A_ineq is not None and A_ineq.size > 0:
            return self._solve_with_inequality_constraints(H, g, A_ineq, b_ineq)
        else:
            return self._solve_unconstrained(H, g)

    def _solve_unconstrained(self, H: MatLike, g: Vec) -> Tuple[Vec, Dict[str, Any]]:
        """Solve unconstrained TR subproblem."""
        n = g.size
        H_op = make_operator(H, n)

        # Try Steihaug CG first
        p, status, iters = steihaug_cg(
            H_op, g, self.delta, self.cfg.cg_tol, self.cfg.cg_maxiter
        )

        info = {
            "status": status.value,
            "iterations": iters,
            "step_norm": safe_norm(p),
            "model_reduction": self._model_reduction(H_op, g, p),
        }

        return p, info

    def _solve_with_equality_constraints(
        self,
        H: MatLike,
        g: Vec,
        A_eq: np.ndarray,
        b_eq: Vec,
        A_ineq: Optional[np.ndarray] = None,
        b_ineq: Optional[Vec] = None,
    ) -> Tuple[Vec, Dict[str, Any]]:
        """
        Byrd-Omojokun approach for equality constrained problems.
        """
        n = g.size
        m_eq = A_eq.shape[0]
        H_op = make_operator(H, n)

        # Adaptive zeta based on constraint violation
        b_eq = np.zeros(m_eq) if b_eq is None else b_eq
        viol_norm = safe_norm(b_eq)
        zeta = (
            self._adaptive_zeta(viol_norm) if self.cfg.adaptive_zeta else self.cfg.zeta
        )

        # Step 1: Normal step (feasibility restoration)
        delta_n = zeta * self.delta
        try:
            p_n = la.lstsq(A_eq, -b_eq)[0]  # Minimum norm solution
            if safe_norm(p_n) > delta_n:
                p_n = (delta_n / safe_norm(p_n)) * p_n
        except la.LinAlgError:
            p_n = np.zeros(n)

        # Step 2: Tangential step in nullspace
        Z = nullspace_basis(A_eq, self.cfg.rcond)

        if Z.shape[1] == 0:  # Full rank constraints
            info = {
                "status": "constrained_minimum",
                "normal_step_norm": safe_norm(p_n),
                "tangential_step_norm": 0.0,
                "constraint_violation": safe_norm(A_eq @ p_n + b_eq),
            }
            return p_n, info

        # Reduced problem in nullspace
        g_reduced = Z.T @ (H_op @ p_n + g)
        H_reduced = Z.T @ (H_op @ Z)

        # Remaining trust region radius
        remaining_radius = np.sqrt(max(0.0, self.delta**2 - safe_norm(p_n) ** 2))

        # Solve reduced TR problem
        p_t_reduced, status, iters = steihaug_cg(
            spla.aslinearoperator(H_reduced),
            g_reduced,
            remaining_radius,
            self.cfg.cg_tol,
            self.cfg.cg_maxiter,
        )

        p_t = Z @ p_t_reduced
        p = p_n + p_t

        # Ensure we stay within trust region
        if safe_norm(p) > self.delta:
            p = (self.delta / safe_norm(p)) * p

        info = {
            "status": status.value,
            "iterations": iters,
            "normal_step_norm": safe_norm(p_n),
            "tangential_step_norm": safe_norm(p_t),
            "step_norm": safe_norm(p),
            "constraint_violation": safe_norm(A_eq @ p + b_eq),
            "nullspace_dim": Z.shape[1],
            "model_reduction": self._model_reduction(H_op, g, p),
            "zeta_used": zeta,
        }

        return p, info

    def _solve_with_inequality_constraints(
        self, H: MatLike, g: Vec, A_ineq: np.ndarray, b_ineq: Vec
    ) -> Tuple[Vec, Dict[str, Any]]:
        """
        Handle inequality constraints via active set strategy.
        """
        # Start with unconstrained solution
        p, info = self._solve_unconstrained(H, g)

        # Check constraint violations
        violations = A_ineq @ p + b_ineq
        violated_idx = violations > self.cfg.constraint_tol

        if not np.any(violated_idx):
            info["active_constraints"] = []
            return p, info

        # Simple active set: add most violated constraint and resolve as equality
        most_violated = np.argmax(violations)
        A_active = A_ineq[most_violated : most_violated + 1]
        b_active = b_ineq[most_violated : most_violated + 1]

        p, eq_info = self._solve_with_equality_constraints(H, g, A_active, b_active)
        info.update(eq_info)
        info["active_constraints"] = [most_violated]

        return p, info

    def _adaptive_zeta(self, constraint_violation: float) -> float:
        """Adapt normal/tangential step ratio based on feasibility."""
        base_zeta = self.cfg.zeta
        if constraint_violation < self.cfg.constraint_tol:
            return max(0.1, base_zeta - 0.2)  # Focus more on optimality
        else:
            return min(0.95, base_zeta + 0.1)  # Focus more on feasibility

    def _model_reduction(self, H_op: spla.LinearOperator, g: Vec, p: Vec) -> float:
        """Compute predicted model reduction."""
        if p.size == 0:
            return 0.0
        return -(np.dot(g, p) + 0.5 * np.dot(p, H_op @ p))

    def update(
        self,
        predicted_reduction: float,
        actual_reduction: float,
        step_norm: float,
        **kwargs,
    ) -> bool:
        """
        Update trust region radius based on step quality.
        Returns True if step should be rejected.
        """
        rho = self._compute_ratio(predicted_reduction, actual_reduction)
        self._rho_history.append(rho)

        # Keep limited history
        if len(self._rho_history) > 10:
            self._rho_history.pop(0)

        # Decision logic
        if rho < self.cfg.eta1:
            # Poor step - shrink radius
            self.delta *= self.cfg.gamma1
            self.rejection_count += 1
            reject = True
        else:
            # Good step - accept
            self.rejection_count = 0
            reject = False

            if rho >= self.cfg.eta2 and step_norm >= 0.8 * self.delta:
                # Excellent step at boundary - expand
                self.delta = min(self.cfg.delta_max, self.cfg.gamma2 * self.delta)

        # Enforce bounds
        self.delta = np.clip(self.delta, self.cfg.delta_min, self.cfg.delta_max)

        return reject

    def _compute_ratio(self, pred_red: float, act_red: float) -> float:
        """Compute trust region ratio with safeguards."""
        if abs(pred_red) < 1e-14:
            return 1.0 if abs(act_red) < 1e-14 else (10.0 if act_red > 0 else -10.0)

        rho = act_red / pred_red
        return np.clip(rho, -100.0, 100.0)

    # Public interface
    def get_radius(self) -> float:
        return self.delta

    def set_radius(self, radius: float) -> None:
        self.delta = np.clip(radius, self.cfg.delta_min, self.cfg.delta_max)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "radius": self.delta,
            "rejections": self.rejection_count,
            "avg_ratio": np.mean(self._rho_history) if self._rho_history else None,
            "config": self.cfg,
        }
