"""
Trust-Region–aware QP wrapper around PIQP (piqp_cpp).

This module exposes a `QPSolver` that solves trust-region SQP subproblems of the form

    minimize_p      ½ pᵀ H p + gᵀ p
    subject to      A_eq p = -b_eq
                    A_ineq p ≤ -b_ineq
                    lb ≤ x_k + p ≤ ub         (variable bounds at current iterate x_k)
              (+)   optional trust-region on a preconditioned step y = L p

Conventions (SQP linearization)
-------------------------------
You should pass Jacobians and residuals at the current iterate x_k:
  - Equalities:   A_eq = J c_eq(x_k),  b_eq = c_eq(x_k)     → solver enforces A_eq p = -b_eq
  - Inequalities: A_in = J c_in(x_k),  b_in = c_in(x_k)     → solver enforces A_in p ≤ -b_in
  - Variable bounds: lb ≤ x_k ≤ ub   → converted internally to step-bounds (lb - x_k) ≤ p ≤ (ub - x_k)

Trust-region options (set via `solve(..., tr_mode=..., tr_radius=...)`):
1) tr_mode="sigma"  (soft cap; extra vars y and t):
   - Enforce  |y_i| ≤ t and optionally t ≤ Δ (recommended).
   - Adds penalty  σ t + (ε_t/2) t² to the objective and adaptively tunes σ
     (up to a small number of extra PIQP solves) to hit t ≈ Δ within tolerance.
   - Decision vector becomes [p; y; t].

2) tr_mode="l2_poly"  (no extra vars; polyhedral approx):
   - Adds directional cuts that approximate { ||L p||₂ ≤ Δ }.
   - Variants:
       * "outer": outer approximation via u_k^T L p ≤ Δ
       * "inner": inner approximation via u_k^T L p ≤ βΔ with β < 1

The preconditioner L defaults to a curvature-aware diagonal derived from the (symmetrized, PSD-ified)
Hessian block P. This makes the trust region roughly curvature-aware while keeping things cheap.

Notes
-----
- H is symmetrized and made numerically PSD on the p-block (sparse path uses a safe diagonal shift).
- All inputs accept dense or sparse; they are internally converted to CSC/CSR as appropriate.
- The solver reuses sparsity patterns and warm-starts across calls whenever possible.
"""

from __future__ import annotations

import numpy as np
import piqp_cpp as piqp
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from .aux import SQPConfig, _cfg_to_piqp
from .reg import Regularizer, make_psd_advanced

# -----------------------------------------------------------------------------#
# Defaults
# -----------------------------------------------------------------------------#
SIGMA_INIT_DEFAULT = 1e-6
SIGMA_MIN = 1e-16
SIGMA_MAX = 1e16
SIGMA_GROW = 10.0       # multiplicative step when expanding/contracting σ
SIGMA_TOL = 0.1         # accept if |t - Δ| ≤ SIGMA_TOL * max(Δ, 1)
SIGMA_MAX_SOLVES = 3    # max extra re-solves after initial σ

TR_POLY_FACETS_DEFAULT = 256   # number of random directions for outer/inner
TR_POLY_SHRINK_INNER = 0.90    # inner-approx shrink factor β∈(0,1]
TR_POLY_SEED = 0               # RNG seed for reproducible directions
TR_PRECOND_DEFAULT = True      # use L from P (curvature-aware)


# =============================================================================
# Helpers
# =============================================================================
def _as_csc_symmetric(H) -> sp.csc_matrix:
    """Return a CSC copy of H symmetrized as ½(H + Hᵀ)."""
    P = H.tocsc(copy=True) if sp.issparse(H) else sp.csc_matrix(H)
    return 0.5 * (P + P.T)


def _force_psd_sparse(P_csc: sp.csc_matrix, eps: float = 1e-12) -> sp.csc_matrix:
    """
    Ensure P is numerically PSD by shifting the diagonal if the smallest eigenvalue is negative.
    Uses a cheap Lanczos estimate for the smallest algebraic eigenvalue.
    """
    try:
        lam_min = eigsh(P_csc, k=1, which="SA", return_eigenvectors=False)[0]
    except Exception:
        lam_min = None
    if lam_min is not None and lam_min < 0:
        tau = float(-lam_min + eps)
        P_csc = P_csc + sp.eye(P_csc.shape[0], format="csc") * tau
    return P_csc


def _choose_L_from_P(P_csc: sp.csc_matrix, n: int, mode: str = "auto") -> sp.csc_matrix:
    """
    Improved trust region preconditioner that handles ill-conditioning better.

    mode:
      - "auto": Automatic selection based on diagonal conditioning (default)
      - "identity": L = I
      - "jacobi": L = diag( sqrt(max(|diag(P)|, 1e-12)) )
      - "jacobi_reg": L = diag( max(|diag(P)|, rel_floor)^1/4 )
      - "norm": L = α I with α = sqrt(||P||_F / n)
    """
    if mode == "identity":
        return sp.eye(n, format="csc")

    d = np.array(P_csc.diagonal()).ravel()
    d_abs = np.abs(d)
    d_min, d_max = (np.min(d_abs), np.max(d_abs)) if d_abs.size else (0.0, 0.0)

    if mode == "norm":
        P_norm = sp.linalg.norm(P_csc, "fro")
        if P_norm < 1e-14:
            return sp.eye(n, format="csc")
        scale = np.sqrt(P_norm / n)
        return sp.diags(np.full(n, scale), format="csc")

    if mode == "auto":
        if d_max < 1e-12:
            return sp.eye(n, format="csc")
        elif d_min / d_max < 1e-8:
            P_norm = sp.linalg.norm(P_csc, "fro")
            scale = np.sqrt(max(P_norm / n, 1e-8))
            return sp.diags(np.full(n, scale), format="csc")
        else:
            mode = "jacobi_reg"

    if mode == "jacobi":
        d_scaled = np.sqrt(np.maximum(d_abs, 1e-12))
        return sp.diags(d_scaled, format="csc")

    if mode == "jacobi_reg":
        d_floor = max(d_max * 1e-6, 1e-10)  # relative floor
        d_reg = np.maximum(d_abs, d_floor)
        d_scaled = np.power(d_reg, 0.25)
        return sp.diags(d_scaled, format="csc")

    raise ValueError(f"Unknown preconditioner mode: {mode}")


def _pad_cols(M: sp.csc_matrix | None, add_cols: int) -> sp.csc_matrix | None:
    """Pad a matrix with `add_cols` zero columns on the right (if M is not None)."""
    if M is None or add_cols <= 0:
        return M
    zeros = sp.csc_matrix((M.shape[0], add_cols))
    return sp.hstack([M, zeros], format="csc")


# =============================================================================
# QP Solver with optional Trust-Region augmentation
# =============================================================================
class QPSolver:
    """
    Thin wrapper over PIQP with optional trust-region augmentation.

    Parameters
    ----------
    cfg : SQPConfig
        SQP/PIQP configuration. Converted via `_cfg_to_piqp(cfg)`.
    regularizer : Regularizer, optional
        Regularization helper passed to `make_psd_advanced` for the H (p-block).

    Attributes
    ----------
    solver : piqp.PIQPSolver
        The underlying PIQP instance.
    last_x, last_y, last_z, last_s : np.ndarray | None
        Cached primal/dual/s data from the last solve.
    """

    def __init__(self, cfg: SQPConfig, regularizer: Regularizer | None = None):
        self.cfg = cfg
        self.regularizer = regularizer
        self.solver = piqp.PIQPSolver(_cfg_to_piqp(cfg))
        self._is_setup = False

        # Identity blocks for bounds (row slicing efficient with CSR)
        self.I = None       # CSR
        self.negI = None    # CSR

        # Pattern + last solution cache
        self._last_dims: tuple[int, int, int] | None = None  # (n_vars, m_eq, m_ineq)
        self.last_x = self.last_y = self.last_z = self.last_s = None

    # ------------------------------ internals ------------------------------
    def _ensure_eye(self, n: int) -> None:
        """Allocate (or resize) cached identity matrices for bounds (CSR for fast row slicing)."""
        if self.I is None or self.I.shape[0] != n:
            self.I = sp.eye(n, format="csr")
            self.negI = (-1.0) * self.I

    def _build_G_h_for_p(self, n, A_ineq, b_ineq, lb, ub, *, xk: np.ndarray | None):
        """
        Build G,h for PIQP with conventions:
          - A_ineq p ≤ -b_ineq   (user inequalities)
          - (lb - xk) ≤ p ≤ (ub - xk)   (variable bounds turned into step bounds; xk REQUIRED)

        Returns:
          G, h, m_ineq_user   (where m_ineq_user excludes bound/TR rows)
        """
        self._ensure_eye(n)
        G_blocks, h_parts = [], []
        m_ineq = 0

        if A_ineq is not None and b_ineq is not None:
            G_blocks.append(sp.csc_matrix(A_ineq))
            h_parts.append(-np.asarray(b_ineq, float))
            m_ineq = A_ineq.shape[0]

        # Bounds must be converted to step-bounds with current iterate xk
        if (lb is not None or ub is not None) and xk is None:
            raise ValueError(
                "SQP step requires current iterate xk to convert variable bounds "
                "lb ≤ x ≤ ub into (lb - xk) ≤ p ≤ (ub - xk)."
            )
        if xk is not None:
            if lb is not None:
                lb_step = np.asarray(lb, float) - np.asarray(xk, float)
            else:
                lb_step = None
            if ub is not None:
                ub_step = np.asarray(ub, float) - np.asarray(xk, float)
            else:
                ub_step = None
        else:
            lb_step = ub_step = None  # no bounds

        # Add finite upper bounds:  p ≤ ub_step
        if ub_step is not None:
            mask = np.isfinite(ub_step)
            if np.any(mask):
                # CSR row slicing then to CSC for PIQP
                G_blocks.append(self.I[mask, :].tocsc())
                h_parts.append(ub_step[mask])

        # Add finite lower bounds: -p ≤ -lb_step  (i.e., p ≥ lb_step)
        if lb_step is not None:
            mask = np.isfinite(lb_step)
            if np.any(mask):
                G_blocks.append(self.negI[mask, :].tocsc())
                h_parts.append((-lb_step[mask]).astype(float))

        G = sp.vstack(G_blocks, format="csc") if G_blocks else None
        h = np.concatenate(h_parts) if h_parts else None
        return G, h, m_ineq

    def _build_A_b_for_p(self, A_eq, b_eq):
        """
        Build equality matrix A and vector b for base variables p only.

        Convention
        ----------
        - User passes A_eq p = -b_eq (i.e., b_eq stores the original c_E(x_k) values).

        Returns
        -------
        A : sp.csc_matrix | None
        b : np.ndarray | None
        """
        A = sp.csc_matrix(A_eq) if A_eq is not None else None
        b = -b_eq if b_eq is not None else None
        return A, b

    # ------------------------ TR: sigma (soft) augmentation ------------------------
    def _augment_with_sigma(self, *, P, q, A, b, G, h, Delta, sigma: float, eps_t: float):
        """
        Soft TR via an auxiliary scalar t:

            y = L p,  |y_i| ≤ t,  and  (optionally) t ≤ Δ
            objective += σ t + (ε_t/2) t²

        Adds variables y (n) and t (1). Decision vector: [p; y; t].
        """
        n = q.size
        L = _choose_L_from_P(P, n)  # (n x n)

        # Equalities: [-L  I  0] [p;y;t] = 0
        A_blk = sp.hstack([-L, sp.eye(n, format="csc"), sp.csc_matrix((n, 1))], format="csc")
        if A is None:
            A_aug = A_blk
            b_aug = np.zeros(n)
        else:
            A_aug = sp.vstack([_pad_cols(A, n + 1), A_blk], format="csc")
            b_aug = np.concatenate([b if b is not None else np.zeros(A.shape[0]), np.zeros(n)])

        # Inequalities: |y_i| ≤ t  (two-sided)
        Zpn = sp.csc_matrix((n, n))
        ones_t = sp.csc_matrix(np.ones((n, 1)))
        Gy_up = sp.hstack([Zpn, sp.eye(n, format="csc"), -ones_t], format="csc")   # y - t ≤ 0
        Gy_dn = sp.hstack([Zpn, -sp.eye(n, format="csc"), -ones_t], format="csc")  # -y - t ≤ 0

        G_list = [Gy_up, Gy_dn]
        h_list = [np.zeros(n), np.zeros(n)]

        # Optional outer hard cap on t (recommended)
        if Delta is not None and np.isfinite(Delta):
            Gt = sp.hstack(
                [sp.csc_matrix((1, n)), sp.csc_matrix((1, n)),
                 sp.csc_matrix(([1.0], ([0], [0])), shape=(1, 1))],
                format="csc",
            )  # t ≤ Δ
            G_list.append(Gt)
            h_list.append(np.array([Delta], float))

        G_new = sp.vstack(G_list, format="csc")
        h_new = np.concatenate(h_list)

        if G is None:
            G_aug, h_aug = G_new, h_new
        else:
            G_aug = sp.vstack([_pad_cols(G, n + 1), G_new], format="csc")
            h_aug = np.concatenate([h, h_new])

        # Objective extension: zero block for y, ε_t on t
        P_aug = sp.block_diag(
            (P, sp.csc_matrix((n, n)), sp.csc_matrix(([eps_t], ([0], [0])), shape=(1, 1)) ),
            format="csc",
        )
        q_aug = np.concatenate([q, np.zeros(n), np.array([sigma], float)])

        return P_aug, q_aug, A_aug, b_aug, G_aug, h_aug, (n + 1)  # y(n) + t(1)

    # ------------------------ TR: L2 polyhedral approximation ------------------------
    def _augment_with_l2_poly_outer(self, *, P, q, A, b, G, h, Delta: float,
                                    m_facets: int = TR_POLY_FACETS_DEFAULT,
                                    precond: bool = TR_PRECOND_DEFAULT,
                                    seed: int = TR_POLY_SEED):
        """
        Outer polyhedral approximation to { p : ||L p||₂ ≤ Δ }.

        Adds constraints: u_k^T L p ≤ Δ for unit vectors u_k.
        No extra variables; adds ~(m_facets + 2n) inequality constraints.
        """
        n = q.size
        L = _choose_L_from_P(P, n) if precond else sp.eye(n, format="csc")

        rng = np.random.default_rng(seed)
        U_rand = rng.standard_normal((m_facets, n))
        U_rand /= np.linalg.norm(U_rand, axis=1, keepdims=True) + 1e-18

        U_coord = np.vstack([np.eye(n), -np.eye(n)])
        U = np.vstack([U_rand, U_coord])

        M = sp.csr_matrix(U) @ L  # (num_directions x n)
        G_new = M.tocsc()
        h_new = np.full(M.shape[0], float(Delta))

        if G is None:
            G_aug, h_aug = G_new, h_new
        else:
            G_aug = sp.vstack([G, G_new], format="csc")
            h_aug = np.concatenate([h, h_new])

        return P, q, A, b, G_aug, h_aug, 0  # no new vars

    def _augment_with_l2_poly_inner(self, *, P, q, A, b, G, h, Delta: float,
                                    m_facets: int = TR_POLY_FACETS_DEFAULT,
                                    shrink: float = TR_POLY_SHRINK_INNER,
                                    precond: bool = TR_PRECOND_DEFAULT,
                                    seed: int = TR_POLY_SEED):
        """
        Inner polyhedral approximation to { p : ||L p||₂ ≤ Δ }.

        Adds constraints: u_k^T L p ≤ βΔ for unit vectors u_k, where β < 1.
        Inner approximation is more conservative but guarantees feasibility.
        """
        n = q.size
        L = _choose_L_from_P(P, n) if precond else sp.eye(n, format="csc")

        effective_facets = max(m_facets, 128)  # safer minimum
        rng = np.random.default_rng(seed)
        U_rand = rng.standard_normal((effective_facets, n))
        U_rand /= np.linalg.norm(U_rand, axis=1, keepdims=True) + 1e-18

        U_coord = np.vstack([np.eye(n), -np.eye(n)])
        U = np.vstack([U_rand, U_coord])

        M = sp.csr_matrix(U) @ L
        G_new = M.tocsc()
        h_new = np.full(M.shape[0], shrink * float(Delta))

        if G is None:
            G_aug, h_aug = G_new, h_new
        else:
            G_aug = sp.vstack([G, G_new], format="csc")
            h_aug = np.concatenate([h, h_new])

        return P, q, A, b, G_aug, h_aug, 0  # no new vars

    def solve(
        self,
        H,
        g,
        A_ineq=None,
        b_ineq=None,
        A_eq=None,
        b_eq=None,
        *,
        xk: np.ndarray | None,                 # <-- REQUIRED when lb/ub are provided (SQP iterate)
        lb=None,
        ub=None,
        tr_radius: float | None = None,
        tr_mode: str = "l2_poly",              # "sigma" | "l2_poly"
        tr_variant: str = "outer",             # for l2_poly: "outer" | "inner"
        tr_sigma: float = 0.0,                 # initial σ for sigma-mode (if ≤ 0 uses default)
        tr_eps_t: float = 1e-6,                # tiny quadratic on t (sigma-mode)
        tr_facets: int = TR_POLY_FACETS_DEFAULT,  # number of random directions
        tr_shrink: float = TR_POLY_SHRINK_INNER,  # shrink factor for inner approx
        tr_precond: bool = TR_PRECOND_DEFAULT,    # use curvature-aware preconditioner
        tr_seed: int = TR_POLY_SEED,              # RNG seed
    ):
        """
        Solve the QP with optional trust-region augmentation.

        Problem (base)
        --------------
            minimize_p    ½ pᵀ H p + gᵀ p
            subject to    A_eq p = -b_eq
                          A_ineq p ≤ -b_ineq
                          lb ≤ x_k + p ≤ ub   (x_k provided via xk=...)

        Trust-Region (optional)
        -----------------------
        If `tr_radius` is finite and positive, add a constraint on y = L p depending on `tr_mode`.

        Returns
        -------
        p : np.ndarray (n,)
            Primal step for original variables.
        lam_ineq : np.ndarray (m_ineq_user,)
            Duals for *user* inequalities (nonnegative). Bound/TR duals are omitted.
        nu_eq : np.ndarray (m_eq_user,)
            Duals for user equalities.
        """
        # --- Quadratic objective (ensure PSD & symmetry on p-block) ---
        if sp.issparse(H):
            P = _as_csc_symmetric(H)
            P = _force_psd_sparse(P)  # ensure PSD (sparse path)
        else:
            H_np = np.asarray(H, float)
            H_psd, _ = make_psd_advanced(H_np, self.regularizer)
            P = sp.csc_matrix(H_psd)

        q = np.asarray(g, float).ravel()
        n = q.size

        # --- Base constraints on p only ---
        A, b = self._build_A_b_for_p(A_eq, b_eq)
        m_eq_user = A.shape[0] if A is not None else 0

        G, h, m_ineq_user = self._build_G_h_for_p(
            n, A_ineq, b_ineq, lb, ub, xk=xk
        )

        # --- Optional TR augmentation ---
        n_added = 0
        t_index = None  # index of t in decision vector for sigma-mode

        if tr_radius is not None and np.isfinite(tr_radius) and tr_radius > 0:
            if tr_mode == "sigma":
                sigma_cur = float(tr_sigma) if tr_sigma > 0 else SIGMA_INIT_DEFAULT
                P, q, A, b, G, h, n_added = self._augment_with_sigma(
                    P=P, q=q, A=A, b=b, G=G, h=h,
                    Delta=float(tr_radius),
                    sigma=sigma_cur,
                    eps_t=float(tr_eps_t),
                )
                # In sigma-mode, t is the last scalar
                t_index = q.size - 1

            elif tr_mode == "l2_poly":
                if tr_variant == "outer":
                    P, q, A, b, G, h, n_added = self._augment_with_l2_poly_outer(
                        P=P, q=q, A=A, b=b, G=G, h=h,
                        Delta=float(tr_radius),
                        m_facets=tr_facets,
                        precond=tr_precond,
                        seed=tr_seed,
                    )
                elif tr_variant == "inner":
                    P, q, A, b, G, h, n_added = self._augment_with_l2_poly_inner(
                        P=P, q=q, A=A, b=b, G=G, h=h,
                        Delta=float(tr_radius),
                        m_facets=tr_facets,
                        shrink=tr_shrink,
                        precond=tr_precond,
                        seed=tr_seed,
                    )
                else:
                    raise ValueError(f"Unknown l2_poly variant '{tr_variant}'. Use 'outer' or 'inner'.")
            else:
                raise ValueError(f"Unknown tr_mode '{tr_mode}'. Use 'sigma' or 'l2_poly'.")

        # --- PIQP setup/update (reuse sparsity pattern if possible) ---
        m_eq = A.shape[0] if A is not None else 0
        m_G = G.shape[0] if G is not None else 0
        n_vars = n + n_added

        same_pattern = self._is_setup and (self._last_dims == (n_vars, m_eq, m_G))
        if not self._is_setup or not same_pattern:
            self.solver.setup(P, q, A, b, G, h)
            self._last_dims = (n_vars, m_eq, m_G)
            self._is_setup = True
        else:
            self.solver.update_values(P, q, A, b, G, h, same_pattern=True)

        # --- Prox centers / warm start (hooks) ---
        # if set_prox_centers:
        #     self.solver.set_prox_centers(prox_x, prox_ineq, prox_eq)
        # if warm_from_last and self.last_x is not None:
        #     self.solver.warm_start(self.last_x, self.last_y, self.last_z, self.last_s)

        # === Solve (inner function) ===
        def _do_solve():
            self.solver.solve()
            x = self.solver.get_x()
            y = self.solver.get_y()
            z = self.solver.get_z()
            s = self.solver.get_s()
            return (
                np.asarray(x, float) if x is not None else np.zeros(n_vars),
                np.asarray(y, float) if y is not None else np.zeros(m_eq),
                np.asarray(z, float) if z is not None else np.zeros(m_G),
                np.asarray(s, float) if s is not None else None,
            )

        # First solve
        x, y_full, z_full, s_full = _do_solve()

        # --- Sigma-mode: adaptive σ tuning to make t ≈ Δ ---
        if (tr_radius is not None and np.isfinite(tr_radius) and tr_radius > 0
            and tr_mode == "sigma"):
            Delta = float(tr_radius)
            t_val = x[t_index]
            tol_abs = SIGMA_TOL * max(Delta, 1.0)

            if not np.isfinite(t_val) or abs(t_val - Delta) > tol_abs:
                sigma_low, sigma_high = None, None
                sigma_cur = float(q[-1])  # current σ is last entry of q

                def _resolve_with_sigma(sigma_new: float):
                    q_new = q.copy()
                    q_new[-1] = float(np.clip(sigma_new, SIGMA_MIN, SIGMA_MAX))
                    self.solver.update_values(P, q_new, A, b, G, h, same_pattern=True)
                    x2, y2, z2, s2 = _do_solve()
                    return q_new, x2, y2, z2, s2

                # Coarse adjust
                if not np.isfinite(t_val) or (t_val > Delta + tol_abs):
                    for _ in range(3):
                        sigma_try = min(SIGMA_MAX, sigma_cur * SIGMA_GROW)
                        q, x, y_full, z_full, s_full = _resolve_with_sigma(sigma_try)
                        t_val = x[t_index]
                        sigma_cur = float(q[-1])
                        if np.isfinite(t_val) and t_val <= Delta + tol_abs:
                            sigma_high = sigma_cur
                            break
                        if sigma_cur >= SIGMA_MAX:
                            break
                elif t_val < Delta - tol_abs:
                    for _ in range(3):
                        sigma_try = max(SIGMA_MIN, sigma_cur / SIGMA_GROW)
                        q, x, y_full, z_full, s_full = _resolve_with_sigma(sigma_try)
                        t_val = x[t_index]
                        sigma_cur = float(q[-1])
                        if np.isfinite(t_val) and t_val >= Delta - tol_abs:
                            sigma_low = sigma_cur
                            break
                        if sigma_cur <= SIGMA_MIN:
                            break

                # Fine-tuning (log-scale bisection-ish)
                solves_left = SIGMA_MAX_SOLVES
                while solves_left > 0 and np.isfinite(t_val) and abs(t_val - Delta) > tol_abs:
                    solves_left -= 1
                    if (sigma_low is not None) and (sigma_high is not None):
                        sig_next = np.exp(0.5 * (np.log(sigma_low) + np.log(sigma_high)))
                    elif sigma_high is not None:
                        sig_next = min(SIGMA_MAX, sigma_high * np.sqrt(SIGMA_GROW))
                    elif sigma_low is not None:
                        sig_next = max(SIGMA_MIN, sigma_low / np.sqrt(SIGMA_GROW))
                    else:
                        sig_next = (min(SIGMA_MAX, sigma_cur * np.sqrt(SIGMA_GROW))
                                    if t_val > Delta else
                                    max(SIGMA_MIN, sigma_cur / np.sqrt(SIGMA_GROW)))

                    q, x, y_full, z_full, s_full = _resolve_with_sigma(sig_next)
                    t_val = x[t_index]
                    sigma_cur = float(q[-1])

                    if not np.isfinite(t_val):
                        sigma_low = None
                        sigma_high = sigma_cur
                        continue
                    if t_val > Delta + tol_abs:
                        sigma_low = sigma_cur
                    elif t_val < Delta - tol_abs:
                        sigma_high = sigma_cur
                    else:
                        break  # close enough

        # --- Cache final solution and extract original variables ---
        self.last_x, self.last_y, self.last_z, self.last_s = x, y_full, z_full, s_full

        # Extract solution for original variables p
        p = x[:n]

        print(z_full)
        # Extract dual variables for user constraints only (exclude bounds and TR constraints)
        lam_ineq = np.maximum(0.0, z_full[:m_ineq_user]) if m_ineq_user > 0 else np.zeros(0)
        nu_eq = y_full[:m_eq_user] if m_eq_user > 0 else np.zeros(0)

        return p, lam_ineq, nu_eq
