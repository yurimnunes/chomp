"""
Trust-Region–aware QP wrapper around PIQP (piqp_cpp).

This module exposes a `QPSolver` that solves trust-region subproblems of the form

    minimize_p      ½ pᵀ H p + gᵀ p
    subject to      A_eq p = -b_eq
                    A_ineq p ≤ -b_ineq
                    lb ≤ p ≤ ub
              (+)   optional trust-region on a preconditioned step y = L p

Trust-region options (set via `solve(..., tr_mode=..., tr_radius=...)`):

1) tr_mode="ellip_inf"  (hard cap; extra vars y):
   - Enforce  |y_i| ≤ Δ with y = L p.
   - Adds n variables (y) and 2n inequalities; no cost terms on y.

2) tr_mode="sigma"  (soft cap; extra vars y and t):
   - Enforce  |y_i| ≤ t and t ≤ Δ (outer hard cap optional but recommended).
   - Adds penalty  σ t + (ε_t/2) t² to the objective and adaptively tunes σ
     (up to a small number of extra PIQP solves) to hit t ≈ Δ within tolerance.
   - Decision vector becomes [p; y; t].

3) tr_mode="l2_poly"  (no extra vars; polyhedral approx):
   - Adds directional cuts that approximate { ||L p||₂ ≤ Δ }.
   - Two variants:
       * "dir": random unit directions (plus ±e_i) yielding symmetric cuts.
       * "l1_inner": inner approximation via ||y||₁ ≤ Δ with y = L p.

The preconditioner L defaults to diag( sqrt( max(diag(P), 1e-12) ) ), where P is
the (symmetrized) PSD-ified Hessian block. This makes the trust region roughly
curvature-aware while keeping things cheap and sparse.

Notes
-----
- H is made numerically PSD via `make_psd_advanced`; this only touches the p-block.
- All inputs accept dense or sparse; they are internally converted to CSC.
- The solver reuses sparsity patterns and warm-starts across calls whenever possible.
"""

from __future__ import annotations

import numpy as np
import piqp_cpp as piqp
import scipy.sparse as sp

from .aux import SQPConfig, _cfg_to_piqp
from .reg import Regularizer, make_psd_advanced

# -----------------------------------------------------------------------------#
# Defaults (can be moved into SQPConfig if you prefer centralization)
# -----------------------------------------------------------------------------#
SIGMA_INIT_DEFAULT = 1e-6
SIGMA_MIN = 1e-16
SIGMA_MAX = 1e16
SIGMA_GROW = 10.0       # multiplicative step when expanding/contracting σ
SIGMA_TOL = 0.1         # accept if |t - Δ| ≤ SIGMA_TOL * max(Δ, 1)
SIGMA_MAX_SOLVES = 2    # max extra re-solves after initial σ

TR_POLY_FACETS_DEFAULT = 64   # number of random directions for dir-variant
TR_POLY_SHRINK_INNER = 0.95   # inner-approx shrink factor β∈(0,1]
TR_POLY_SEED = 0              # RNG seed for reproducible directions
TR_PRECOND_DEFAULT = True     # use L from P (curvature-aware)


# =============================================================================
# Helpers
# =============================================================================
def _as_csc_symmetric(H) -> sp.csc_matrix:
    """Return a CSC copy of H symmetrized as ½(H + Hᵀ)."""
    P = H.tocsc(copy=True) if sp.issparse(H) else sp.csc_matrix(H)
    return 0.5 * (P + P.T)


def _choose_L_from_P(P_csc: sp.csc_matrix, n: int) -> sp.csc_matrix:
    """
    Cheap curvature-aware scaling L = diag( sqrt( max(diag(P), 1e-12) ) ).
    This behaves like a Jacobi preconditioner for the TR geometry.
    """
    d = np.array(P_csc.diagonal()).ravel()
    d = np.sqrt(np.maximum(d, 1e-12))
    return sp.diags(d, format="csc")


def _pad_cols(M: sp.csc_matrix | None, add_cols: int) -> sp.csc_matrix | None:
    """Pad a matrix with `add_cols` zero columns on the right (if M is not None)."""
    if M is None:
        return None
    if add_cols <= 0:
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

        # Identity blocks for bounds
        self.I = None
        self.negI = None

        # Pattern + last solution cache
        self._last_dims: tuple[int, int, int] | None = None  # (n_vars, m_eq, m_ineq)
        self.last_x = self.last_y = self.last_z = self.last_s = None

    # ------------------------------ internals ------------------------------
    def _ensure_eye(self, n: int) -> None:
        """Allocate (or resize) cached identity matrices for bounds."""
        if self.I is None or self.I.shape[0] != n:
            self.I = sp.eye(n, format="csc")
            self.negI = (-1.0) * self.I

    def _build_G_h_for_p(self, n, A_ineq, b_ineq, lb, ub):
        """
        Build inequality matrix G and vector h for base variables p only.

        Conventions
        -----------
        - User passes A_ineq p ≤ -b_ineq (i.e., b_ineq stores the original c_I(x) values).
        - Variable bounds are appended as p ≤ ub and -p ≤ -lb.

        Returns
        -------
        G : sp.csc_matrix | None
        h : np.ndarray | None
        m_ineq : int
            Number of user-supplied inequality rows (excludes bound rows).
        """
        self._ensure_eye(n)

        G_blocks, h_parts = [], []
        m_ineq = 0

        if A_ineq is not None and b_ineq is not None:
            G_blocks.append(sp.csc_matrix(A_ineq))
            # convention: A_ineq p ≤ -b_ineq  -> h = -b_ineq
            h_parts.append(-np.asarray(b_ineq, dtype=float))
            m_ineq = A_ineq.shape[0]

        if ub is not None:
            G_blocks.append(self.I)  # p ≤ ub
            h_parts.append(np.asarray(ub, dtype=float))

        if lb is not None:
            G_blocks.append(self.negI)  # -p ≤ -lb  (i.e., p ≥ lb)
            h_parts.append(-np.asarray(lb, dtype=float))

        G = sp.vstack(G_blocks, format="csc") if G_blocks else None
        h = np.concatenate(h_parts) if h_parts else None
        return G, h, m_ineq

    def _build_A_b_for_p(self, A_eq, b_eq):
        """
        Build equality matrix A and vector b for base variables p only.

        Convention
        ----------
        - User passes A_eq p = -b_eq (i.e., b_eq stores the original c_E(x) values).

        Returns
        -------
        A : sp.csc_matrix | None
        b : np.ndarray | None
        """
        A = sp.csc_matrix(A_eq) if A_eq is not None else None
        b = -b_eq if b_eq is not None else None
        return A, b

    # ------------------------ TR: ellip_inf augmentation ------------------------
    def _augment_with_ellip_inf(self, *, P, q, A, b, G, h, Delta):
        """
        Hard box in preconditioned step-space: y = L p,  |y|_∞ ≤ Δ.

        Adds n variables (y) and constraints:
            Equalities:     y - L p = 0
            Inequalities:   -Δ ≤ y ≤ Δ
        Objective is unchanged (y has zero cost).
        """
        n = q.size
        L = _choose_L_from_P(P, n)  # (n x n)

        # Equalities: [-L  I] [p;y] = 0
        A_blk = sp.hstack([-L, sp.eye(n, format="csc")], format="csc")
        if A is None:
            A_aug = A_blk
            b_aug = np.zeros(n)
        else:
            A_aug = sp.vstack([_pad_cols(A, n), A_blk], format="csc")
            b_aug = np.concatenate([b if b is not None else np.zeros(A.shape[0]), np.zeros(n)])

        # Inequalities on y
        Gy_up = sp.hstack([sp.csc_matrix((n, n)), sp.eye(n, format="csc")], format="csc")   # y ≤ Δ
        Gy_dn = sp.hstack([sp.csc_matrix((n, n)), -sp.eye(n, format="csc")], format="csc")  # -y ≤ Δ
        G_new = sp.vstack([Gy_up, Gy_dn], format="csc")
        h_new = np.concatenate([Delta * np.ones(n), Delta * np.ones(n)])

        if G is None:
            G_aug, h_aug = G_new, h_new
        else:
            G_aug = sp.vstack([_pad_cols(G, n), G_new], format="csc")
            h_aug = np.concatenate([h, h_new])

        # Objective extension: zero block for y
        P_aug = sp.block_diag((P, sp.csc_matrix((n, n))), format="csc")
        q_aug = np.concatenate([q, np.zeros(n)])

        return P_aug, q_aug, A_aug, b_aug, G_aug, h_aug, n  # n added vars

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

    def _augment_with_l2_poly_dir(
        self,
        *,
        P, q, A, b, G, h,
        Delta: float,
        m_facets: int = TR_POLY_FACETS_DEFAULT,
        precond: bool = TR_PRECOND_DEFAULT,
        inner: bool = True,
        shrink: float | None = None,
        seed: int = TR_POLY_SEED,
    ):
        """
        Polyhedral approximation to { p : ||L p||₂ ≤ Δ } via directional cuts.

        No extra variables; we add rows:
            -βΔ ≤ u_kᵀ L p ≤ βΔ,   for k = 1..(m_facets + 2n)
        where {u_k} are random unit vectors plus the axis-aligned ±e_i.

        Parameters
        ----------
        inner : bool
            If True, use an inner approximation with shrink factor β∈(0,1].
            If False, β=1.0 (outer approximation). Defaults to inner.
        shrink : float | None
            Override for β when inner=True. Defaults to 0.95.
        """
        n = q.size

        # Choose L (preconditioned or identity)
        L = _choose_L_from_P(P, n) if precond else sp.eye(n, format="csc")

        # Random unit directions + ±e_i to guarantee a box at minimum
        rng = np.random.default_rng(seed)
        U = rng.standard_normal((m_facets, n))
        U /= np.linalg.norm(U, axis=1, keepdims=True) + 1e-18
        U = np.vstack([U, np.eye(n), -np.eye(n)])

        # Effective operator on p
        M = sp.csr_matrix(U) @ L  # (m x n)

        # Inner vs outer scaling
        beta = TR_POLY_SHRINK_INNER if (inner and shrink is None) else (shrink if inner else 1.0)
        beta = float(np.clip(beta, 1e-6, 1.0))
        rhs = beta * float(Delta)

        # Build ±M p ≤ rhs
        G_new = sp.vstack([M, -M], format="csc")
        h_new = np.concatenate([np.full(M.shape[0], rhs), np.full(M.shape[0], rhs)])

        if G is None:
            G_aug, h_aug = G_new, h_new
        else:
            G_aug = sp.vstack([G, G_new], format="csc")
            h_aug = np.concatenate([h, h_new])

        return P, q, A, b, G_aug, h_aug, 0  # no new vars

    def _augment_with_l2_l1_inner(self, *, P, q, A, b, G, h, Delta: float, precond: bool = TR_PRECOND_DEFAULT):
        """
        Inner approximation: ||L p||₂ ≤ Δ  ⇒  introduce y = L p and enforce ||y||₁ ≤ Δ.

        Adds variables y (n) and t (n) with constraints:
            y - L p = 0
            -t ≤ y ≤ t
            -t ≤ 0  (i.e., t ≥ 0)
            1ᵀ t ≤ Δ
        Objective unchanged for [y; t].
        """
        n = q.size
        L = _choose_L_from_P(P, n) if precond else sp.eye(n, format="csc")

        # Equalities: [-L  I  0] [p;y;t] = 0
        A_blk = sp.hstack([-L, sp.eye(n, format="csc"), sp.csc_matrix((n, n))], format="csc")
        if A is None:
            A_aug, b_aug = A_blk, np.zeros(n)
        else:
            A_aug = sp.vstack([_pad_cols(A, 2 * n), A_blk], format="csc")
            b_aug = np.concatenate([b if b is not None else np.zeros(A.shape[0]), np.zeros(n)])

        # Inequalities for -t ≤ y ≤ t and t ≥ 0 plus 1ᵀ t ≤ Δ
        Zpn = sp.csc_matrix((n, n))
        I = sp.eye(n, format="csc")
        negI = -I

        Gy_up = sp.hstack([Zpn, I,    negI], format="csc")   # y - t ≤ 0
        Gy_dn = sp.hstack([Zpn, -I,   negI], format="csc")   # -y - t ≤ 0
        Gt_nn = sp.hstack([sp.csc_matrix((n, n)), sp.csc_matrix((n, n)), negI], format="csc")  # -t ≤ 0
        Gsum  = sp.hstack([sp.csc_matrix((1, n)), sp.csc_matrix((1, n)), sp.csc_matrix(np.ones((1, n)))], format="csc")  # 1ᵀ t ≤ Δ

        G_new = sp.vstack([Gy_up, Gy_dn, Gt_nn, Gsum], format="csc")
        h_new = np.concatenate([np.zeros(n), np.zeros(n), np.zeros(n), np.array([float(Delta)])])

        if G is None:
            G_aug, h_aug = G_new, h_new
        else:
            G_aug = sp.vstack([_pad_cols(G, 2 * n), G_new], format="csc")
            h_aug = np.concatenate([h, h_new])

        # Extend objective with zeros for [y; t]
        P_aug = sp.block_diag((P, sp.csc_matrix((2 * n, 2 * n))), format="csc")
        q_aug = np.concatenate([q, np.zeros(2 * n)])

        return P_aug, q_aug, A_aug, b_aug, G_aug, h_aug, 2 * n

    # ------------------------------ public API ------------------------------
    def solve(
        self,
        H,
        g,
        A_ineq=None,
        b_ineq=None,
        A_eq=None,
        b_eq=None,
        lb=None,
        ub=None,
        *,
        warm_from_last: bool = True,
        set_prox_centers: bool = False,
        prox_x=None,
        prox_ineq=None,
        prox_eq=None,
        tr_radius: float | None = None,
        tr_mode: str = "l2_poly",   # "ellip_inf" | "sigma" | "l2_poly"
        tr_sigma: float = 0.0,      # initial σ for sigma-mode (if ≤ 0 uses SIGMA_INIT_DEFAULT)
        tr_eps_t: float = 1e-8,     # tiny quadratic on t (sigma-mode)
    ):
        """
        Solve the QP with optional trust-region augmentation.

        Problem (base)
        --------------
            minimize_p    ½ pᵀ H p + gᵀ p
            subject to    A_eq p = -b_eq
                          A_ineq p ≤ -b_ineq
                          lb ≤ p ≤ ub

        Trust-Region (optional)
        -----------------------
        If `tr_radius` is finite, add a constraint on y = L p depending on `tr_mode`.

        Parameters
        ----------
        H : array_like or spmatrix
            Quadratic term (n×n). Will be symmetrized and made PSD numerically.
        g : array_like (n,)
            Linear term.
        A_ineq, b_ineq : array_like or spmatrix, array_like
            User-supplied inequalities as A_ineq p ≤ -b_ineq.
        A_eq, b_eq : array_like or spmatrix, array_like
            User-supplied equalities as A_eq p = -b_eq.
        lb, ub : array_like, optional
            Component-wise bounds on p.
        warm_from_last : bool
            Keep PIQP warm-start from previous call (recommended).
        set_prox_centers, prox_x, prox_ineq, prox_eq : any
            Hooks for your external proximal/warm-start logic (left as-is).
        tr_radius : float | None
            Trust-region radius Δ. If None/inf, TR augmentation is disabled.
        tr_mode : {"ellip_inf", "sigma", "l2_poly"}
            Select TR strategy (see module docstring).
        tr_sigma : float
            Initial σ for sigma-mode. If ≤ 0, uses SIGMA_INIT_DEFAULT.
        tr_eps_t : float
            Small quadratic weight on t (sigma-mode) to stabilize the scalar.

        Returns
        -------
        p : np.ndarray (n,)
            Primal step for original variables.
        lam_ineq : np.ndarray (m_ineq_user,)
            Duals for user inequalities (nonnegative). Bound duals are omitted.
        nu_eq : np.ndarray (m_eq_user,)
            Duals for user equalities.
        """
        # --- Quadratic objective (ensure PSD & symmetry on p-block) ---
        if sp.issparse(H):
            P = _as_csc_symmetric(H)
        else:
            H_np = np.asarray(H, float)
            H_psd, _ = make_psd_advanced(H_np, self.regularizer)
            P = sp.csc_matrix(H_psd)

        q = np.asarray(g, float).ravel()
        n = q.size

        # --- Base constraints on p only ---
        A, b = self._build_A_b_for_p(A_eq, b_eq)
        m_eq_user = A.shape[0] if A is not None else 0

        G, h, m_ineq_user = self._build_G_h_for_p(n, A_ineq, b_ineq, lb, ub)

        # --- Optional TR augmentation ---
        n_added = 0
        t_index = None  # index of t in decision vector for sigma-mode

        if tr_radius is not None and np.isfinite(tr_radius):
            if tr_mode == "ellip_inf":
                P, q, A, b, G, h, n_added = self._augment_with_ellip_inf(
                    P=P, q=q, A=A, b=b, G=G, h=h, Delta=float(tr_radius)
                )
            elif tr_mode == "sigma":
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
                variant = getattr(self.cfg, "tr_poly_variant", "dir")
                if variant == "dir":
                    P, q, A, b, G, h, n_added = self._augment_with_l2_poly_dir(
                        P=P, q=q, A=A, b=b, G=G, h=h,
                        Delta=float(tr_radius),
                        m_facets=getattr(self.cfg, "tr_poly_facets", TR_POLY_FACETS_DEFAULT),
                        precond=getattr(self.cfg, "tr_poly_precond", TR_PRECOND_DEFAULT),
                        inner=getattr(self.cfg, "tr_poly_inner", False),
                        shrink=getattr(self.cfg, "tr_poly_shrink", None),
                        seed=getattr(self.cfg, "tr_poly_seed", TR_POLY_SEED),
                    )
                elif variant == "l1_inner":
                    P, q, A, b, G, h, n_added = self._augment_with_l2_l1_inner(
                        P=P, q=q, A=A, b=b, G=G, h=h,
                        Delta=float(tr_radius),
                        precond=getattr(self.cfg, "tr_poly_precond", TR_PRECOND_DEFAULT),
                    )
                else:
                    raise ValueError(f"Unknown l2_poly variant '{variant}'")
            else:
                raise ValueError(f"Unknown tr_mode '{tr_mode}'")

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

        # --- Prox centers / warm start (hook points; left as-is) ---
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

        # --- Sigma-mode: adapt σ to make t ≈ Δ (within relative tolerance) ---
        if tr_radius is not None and np.isfinite(tr_radius) and tr_mode == "sigma":
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

                # If t too large → increase σ; if too small → decrease σ.
                if not np.isfinite(t_val) or (t_val > Delta + tol_abs):
                    for _ in range(3):  # coarse expansion
                        sigma_try = min(SIGMA_MAX, sigma_cur * SIGMA_GROW)
                        q, x, y_full, z_full, s_full = _resolve_with_sigma(sigma_try)
                        t_val = x[t_index]
                        sigma_cur = float(q[-1])
                        if np.isfinite(t_val) and t_val <= Delta + tol_abs:
                            sigma_high, sigma_low = sigma_cur, None
                            break
                else:
                    for _ in range(3):  # coarse contraction
                        sigma_try = max(SIGMA_MIN, sigma_cur / SIGMA_GROW)
                        q, x, y_full, z_full, s_full = _resolve_with_sigma(sigma_try)
                        t_val = x[t_index]
                        sigma_cur = float(q[-1])
                        if np.isfinite(t_val) and (t_val > Delta - tol_abs):
                            sigma_low, sigma_high = sigma_cur, None
                            break

                # A few guided steps (log-bisection if both sides known)
                solves_left = SIGMA_MAX_SOLVES
                while solves_left > 0:
                    solves_left -= 1
                    if (sigma_low is not None) and (sigma_high is not None):
                        sig_next = np.exp(0.5 * (np.log(sigma_low) + np.log(sigma_high)))
                    elif sigma_high is not None:
                        sig_next = min(SIGMA_MAX, sigma_high * np.sqrt(SIGMA_GROW))
                    elif sigma_low is not None:
                        sig_next = max(SIGMA_MIN, sigma_low / np.sqrt(SIGMA_GROW))
                    else:
                        break

                    q, x, y_full, z_full, s_full = _resolve_with_sigma(sig_next)
                    t_val = x[t_index]

                    if not np.isfinite(t_val):
                        sigma_low = None
                        sigma_high = float(q[-1])
                        continue

                    if t_val > Delta + tol_abs:
                        sigma_low = float(q[-1])    # need stronger penalty
                        if sigma_high is None:
                            continue
                    elif t_val < Delta - tol_abs:
                        sigma_high = float(q[-1])   # can relax penalty
                        if sigma_low is None:
                            continue
                    else:
                        break  # good enough

        # --- Cache and map back to original variables ---
        self.last_x, self.last_y, self.last_z, self.last_s = x, y_full, z_full, s_full

        p = x[:n]
        lam_ineq = np.maximum(0.0, z_full[:m_ineq_user]) if m_ineq_user > 0 else np.zeros(0)
        nu_eq = y_full[:m_eq_user]
        return p, lam_ineq, nu_eq
