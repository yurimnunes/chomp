from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.linalg import lstsq, norm
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import linprog
from scipy.spatial.distance import cdist

# Optional geometry integration (safe to import even if unused)
try:
    from tr_model import TRModel
    from tr_model_ops import change_tr_center, ensure_improvement
except Exception:
    TRModel = None  # type: ignore

# ===============================
# Small numerics helpers
# ===============================

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

def _solve_weighted_ridge(Phi: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
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

def _improved_fps(Y: np.ndarray, k: int) -> np.ndarray:
    m = Y.shape[0]
    if k >= m:
        return np.arange(m, dtype=int)
    if m == 0:
        return np.array([], dtype=int)
    norms = np.linalg.norm(Y, axis=1)
    start = int(np.argmin(norms))
    sel = [start]
    rem = set(range(m)) - {start}
    if rem:
        mind = np.full(m, np.inf)
        mind[start] = 0.0
        for _ in range(1, k):
            if not rem:
                break
            jlast = sel[-1]
            for idx in list(rem):
                d = norm(Y[idx] - Y[jlast])
                if d < mind[idx]:
                    mind[idx] = d
            j = max(rem, key=lambda i: mind[i])
            sel.append(j)
            rem.remove(j)
    return np.array(sel, dtype=int)

# ===============================
# CSV basis & poised row selection
# ===============================

def _csv_basis(Y: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    CSV polynomial basis at 0 (center), on normalized coords Y=(X-center)/Delta.
    Columns: [1, y1..yn, {y_i y_j}_{i<=j}].
    Returns: Phi, n, nquad
    """
    m, n = Y.shape
    ones = np.ones((m, 1), dtype=Y.dtype)
    lin = Y
    quads = []
    for i in range(n):
        for j in range(i, n):
            quads.append(Y[:, i:i + 1] * Y[:, j:j + 1])
    Phi = np.hstack([ones, lin] + (quads if quads else []))
    nquad = n * (n + 1) // 2
    return Phi, n, nquad

def _select_poised_rows(Phi: np.ndarray, k: int) -> np.ndarray:
    """
    Pick k rows that form a poised subset via leverage scores (SVD-based).
    """
    m, _ = Phi.shape
    if k >= m:
        return np.arange(m, dtype=int)
    try:
        U, s, Vt = np.linalg.svd(Phi, full_matrices=False)
        lev = (U**2).sum(axis=1)
    except Exception:
        lev = np.linalg.norm(Phi, axis=1) ** 2
    idx = np.argsort(-lev)[:k]
    return np.sort(idx)

# ===============================
# Config / FitResult / Criticality
# ===============================

@dataclass
class DFOConfig:
    # sampling & regression
    huber_delta: Optional[float] = None
    ridge: float = 1e-6
    dist_w_power: float = 0.3
    eig_floor: float = 1e-8
    max_pts: int = 60
    model_radius_mult: float = 2.0
    use_quadratic_if: int = 25  # min samples to allow quadratic; else linear

    # penalty / TR thresholds
    mu: float = 10.0
    eps_active: float = 1e-6
    tr_inc: float = 1.6
    tr_dec: float = 0.5
    eta0: float = 0.05
    eta1: float = 0.25

    # criticality step (Algorithm 3)
    crit_beta1: float = 0.5
    crit_beta2: float = 2.0
    crit_beta3: float = 0.5
    eps_c: float = 1e-4

    # conservative prediction constants
    k_fd: float = 1.0     # κ_fd
    k_H: float  = 1.0     # κ_H
    k_c: float  = 1.0     # κ_c

    # VK correction solver
    vk_gn_maxit: int = 10
    vk_gn_tol: float  = 1e-8

    # LP direction caps
    lp_max_active: int = 32
    lp_time_limit: float = 0.10
    lp_verbose: bool = False

    # multiplier-step controls
    use_multiplier_step: bool = True
    mult_sigma_thresh: float = 1e-3
    mult_project_lambda: bool = True
    mult_newton_clip_inf: bool = True

    # feasible-landing tolerance
    fl_tol: float = 1e-3

    # ===== Conn-style MFN quadratic objective =====
    use_conn_mfn_objective: bool = True
    conn_exact_if_possible: bool = True
    mfn_lambda: float = 1e-10
    mfn_bias_lin: float = 0.0

    # ===== RBF objective (optional) =====
    use_rbf_objective: bool = False
    rbf_kind: str = "wendland_c2"     # "cubic" | "wendland_c2" | "gaussian" | "mq"
    rbf_epsilon: float = 1.0
    rbf_support: float = 1.5
    rbf_lambda: float = 1e-10
    rbf_max_pts: int = 120

    # ===== NFP polynomial (default in this stack) =====
    use_nfp_objective: bool = True

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

# ======================================================================
# Objective-model strategy interfaces
# ======================================================================

class _ObjectiveModelBase:
    """
    Interface: implement fit_objective(Y, rhs, n, cfg) -> (g, H, info_dict)
      - Y   : (m, n) normalized points ( (X - center) / Delta )
      - rhs : (m,)   objective values (F - F[0]) aligned with Y
      - n   : dimension
      - cfg : DFOConfig
    """
    def fit_objective(self, Y: np.ndarray, rhs: np.ndarray, n: int, cfg) -> tuple:
        raise NotImplementedError

# --------- RBF ---------

class DFORBFModel(_ObjectiveModelBase):
    """RBF with linear tail (a + b^T y), Tikhonov on K, ∇ and ∇² via radial derivatives."""

    @staticmethod
    def _rbf_funcs(cfg):
        kind = (cfg.rbf_kind or "cubic").lower()
        eps = float(cfg.rbf_epsilon)

        def cubic_phi(r): return (eps*r)**3
        def cubic_d1(r): return 3 * (eps**3) * (r**2)
        def cubic_d2(r): return 6 * (eps**3) * r

        def wendland_c2_phi(r):
            rho = float(cfg.rbf_support); s = np.maximum(0.0, 1.0 - (eps*r)/max(rho,1e-16))
            return (s**4) * (4*(eps*r)/max(rho,1e-16) + 1.0)
        def wendland_c2_d1(r):
            rho = float(cfg.rbf_support); e_rho = eps/max(rho,1e-16)
            rr = e_rho * r
            s = np.maximum(0.0, 1.0 - rr)
            return s**4*(4*e_rho) + (4*rr+1.0)*4*(s**3)*(-e_rho)
        def wendland_c2_d2(r):
            rho = float(cfg.rbf_support); e_rho = eps/max(rho,1e-16)
            rr = e_rho * r
            s = np.maximum(0.0, 1.0 - rr)
            term1 = 4*e_rho * (4*(s**3)*(-e_rho))
            term2 = (4*rr + 1.0) * 12*(s**2) * (e_rho**2)
            term3 = 4*(s**3) * (4*e_rho*e_rho)
            return term1 + term2 + term3

        def gaussian_phi(r): return np.exp(-(eps*r)**2)
        def gaussian_d1(r): return -2*(eps**2)*r*np.exp(-(eps*r)**2)
        def gaussian_d2(r): return (4*(eps**4)*(r**2) - 2*(eps**2))*np.exp(-(eps*r)**2)

        def mq_phi(r): return np.sqrt(1.0 + (eps*r)**2)
        def mq_d1(r):
            t = (eps*r)**2
            return (eps**2)*r/np.sqrt(1.0 + t)
        def mq_d2(r):
            t = (eps*r)**2; denom = (1.0 + t)**1.5
            return (eps**2)/np.sqrt(1.0 + t) - (eps**4)*(r**2)/denom

        if kind == "cubic": return cubic_phi, cubic_d1, cubic_d2
        if kind in ("wendland","wendland_c2","wendland-c2"): return wendland_c2_phi, wendland_c2_d1, wendland_c2_d2
        if kind in ("gauss","gaussian"): return gaussian_phi, gaussian_d1, gaussian_d2
        if kind in ("mq","multiquadric"): return mq_phi, mq_d1, mq_d2
        return cubic_phi, cubic_d1, cubic_d2

    def fit_objective(self, Y: np.ndarray, rhs: np.ndarray, n: int, cfg) -> tuple:
        m = Y.shape[0]
        if m > cfg.rbf_max_pts:
            Phi_full, _, _ = _csv_basis(Y)
            sel = _select_poised_rows(Phi_full, cfg.rbf_max_pts)
            Y = Y[sel]; rhs = rhs[sel]; m = Y.shape[0]

        phi, d1, d2 = self._rbf_funcs(cfg)
        D = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=2)
        K = phi(D)
        if cfg.rbf_lambda > 0:
            K = K + cfg.rbf_lambda * np.eye(m)

        P = np.hstack([np.ones((m, 1)), Y])
        Z = np.zeros((n+1, n+1))
        A = np.block([[K, P],
                      [P.T, Z]])
        b = np.concatenate([rhs, np.zeros(n+1)])
        try:
            sol = np.linalg.solve(A, b)
        except Exception:
            sol = lstsq(A, b, rcond=1e-12)[0]

        w = sol[:m]
        b_lin = sol[m+1:]

        g = np.zeros(n); H = np.zeros((n, n)); I = np.eye(n)
        ri = np.linalg.norm(Y, axis=1)
        for i in range(m):
            r = float(ri[i]); yi = Y[i]
            if r <= 1e-16:
                d2r = float(d2(0.0))
                if np.isfinite(d2r) and abs(d2r) > 0:
                    H += w[i] * d2r * (I / float(n))
                continue
            ui = yi / r
            d1r = float(d1(r)); d2r = float(d2(r))
            g += w[i] * d1r * ui
            outer = np.outer(ui, ui)
            H += w[i] * (d2r * outer + (d1r / r) * (I - outer))

        g += b_lin
        try:
            eig, V = np.linalg.eigh(0.5*(H+H.T))
            H = (V * np.maximum(eig, cfg.eig_floor)) @ V.T
        except Exception:
            H = 0.5*(H+H.T) + cfg.eig_floor*np.eye(n)

        info = {"rbf_kind": cfg.rbf_kind, "m_used": int(m), "model": "rbf"}
        return g, H, info

# --------- Conn/MFN on CSV ---------

class DFOConnMFNModel(_ObjectiveModelBase):
    """CSV polynomial with MFN on quadratic block."""

    @staticmethod
    def _mfn_Rq_sqrt(n: int) -> np.ndarray:
        w = []
        for i in range(n):
            for j in range(i, n):
                w.append(2.0 if i == j else np.sqrt(2.0))
        return np.asarray(w, float)

    def fit_objective(self, Y: np.ndarray, rhs: np.ndarray, n: int, cfg) -> tuple:
        Phi, n_chk, nquad = _csv_basis(Y)      # [1 | lin(n) | quad(nq)]
        assert n_chk == n
        m, p = Phi.shape
        idx_lin = 1 + n

        Phi_use, y_use = Phi, rhs
        if m > p and cfg.use_conn_mfn_objective:
            sel = _select_poised_rows(Phi, p)
            Phi_use = Phi[sel]; y_use = rhs[sel]

        w_samp = _huber_weights(y_use, cfg.huber_delta) * _distance_weights(Y[:Phi_use.shape[0]], cfg.dist_w_power)
        sw = np.sqrt(np.clip(w_samp, 1e-12, np.inf))
        Pw = Phi_use * sw[:, None]
        yw = y_use * sw

        R = np.zeros((p, p))
        if cfg.mfn_bias_lin > 0.0:
            R[:idx_lin, :idx_lin] = cfg.mfn_bias_lin * np.eye(idx_lin)
        if nquad > 0:
            Rq = self._mfn_Rq_sqrt(n)
            R[idx_lin:, idx_lin:] = cfg.mfn_lambda * np.diag(Rq*Rq)

        A = Pw.T @ Pw + R
        b = Pw.T @ yw
        try:
            if cfg.use_conn_mfn_objective and cfg.conn_exact_if_possible and Pw.shape[0] == p:
                coef = np.linalg.solve(A, b)
            else:
                L = cholesky(A, lower=True)
                z = solve_triangular(L, b, lower=True)
                coef = solve_triangular(L.T, z, lower=False)
        except Exception:
            try:
                coef = np.linalg.solve(A, b)
            except Exception:
                coef = lstsq(Pw, yw, rcond=1e-10)[0]

        g = np.zeros(n); H = np.zeros((n, n))
        if len(coef) >= 1 + n:
            g[:] = coef[1:1+n]
            q = coef[1+n:]; k = 0
            for i in range(n):
                for j in range(i, n):
                    bb = q[k]
                    if i == j: H[i, i] += 2.0*bb
                    else: H[i, j] += bb; H[j, i] += bb
                    k += 1

        try:
            eig, V = np.linalg.eigh(0.5*(H+H.T))
            H = (V * np.maximum(eig, cfg.eig_floor)) @ V.T
        except Exception:
            H = 0.5*(H+H.T) + cfg.eig_floor*np.eye(n)

        info = {"model": "conn_mfn", "m_used": int(Phi_use.shape[0]), "p": int(p)}
        return g, H, info

# --------- NFP design (matrix) + NFP model ---------

def nfp_design(Y: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Build NFP-like design matrix (upper triangular monomials):
      [1 | y_1..y_n | {y_i y_j}_{i<=j}]
    Returns (Phi, n, nquad).
    """
    m, n = Y.shape
    cols = [np.ones((m, 1)), Y]
    quads = []
    for i in range(n):
        yi = Y[:, i:i+1]
        for j in range(i, n):
            yj = Y[:, j:j+1]
            quads.append(yi * yj)
    if quads:
        cols.append(np.hstack(quads))
    Phi = np.hstack(cols)
    nquad = n * (n + 1) // 2
    return Phi, n, nquad

class DFONFPModel(_ObjectiveModelBase):
    """NFP polynomial model (c + g^T y + 1/2 y^T H y) on nfp_design()."""

    @staticmethod
    def _mfn_Rq_sqrt(n: int) -> np.ndarray:
        w = []
        for i in range(n):
            for j in range(i, n):
                w.append(2.0 if i == j else np.sqrt(2.0))
        return np.asarray(w, float)

    def fit_objective(self, Y: np.ndarray, rhs: np.ndarray, n: int, cfg) -> tuple:
        Phi, n_chk, nquad = nfp_design(Y)
        assert n_chk == n
        m, p = Phi.shape
        idx_lin = 1 + n

        Phi_use, y_use = Phi, rhs
        if m > p and getattr(cfg, "use_conn_mfn_objective", False):
            sel = _select_poised_rows(Phi, p)
            Phi_use = Phi[sel]; y_use = rhs[sel]

        w_samp = _huber_weights(y_use, cfg.huber_delta) * _distance_weights(Y[:Phi_use.shape[0]], cfg.dist_w_power)
        sw = np.sqrt(np.clip(w_samp, 1e-12, np.inf))
        Pw = Phi_use * sw[:, None]
        yw = y_use * sw

        R = np.zeros((p, p))
        if getattr(cfg, "mfn_bias_lin", 0.0) > 0.0:
            R[:idx_lin, :idx_lin] = cfg.mfn_bias_lin * np.eye(idx_lin)
        if nquad > 0 and getattr(cfg, "mfn_lambda", 0.0) > 0.0:
            Rq = self._mfn_Rq_sqrt(n)
            R[idx_lin:, idx_lin:] = cfg.mfn_lambda * np.diag(Rq * Rq)

        A = Pw.T @ Pw + R
        b = Pw.T @ yw
        try:
            if getattr(cfg, "conn_exact_if_possible", False) and Pw.shape[0] == p:
                coef = np.linalg.solve(A, b)
            else:
                L = cholesky(A, lower=True)
                z = solve_triangular(L, b, lower=True)
                coef = solve_triangular(L.T, z, lower=False)
        except Exception:
            try:
                coef = np.linalg.solve(A, b)
            except Exception:
                coef = lstsq(Pw, yw, rcond=1e-10)[0]

        g = np.zeros(n); H = np.zeros((n, n))
        if len(coef) >= 1 + n:
            g[:] = coef[1:1+n]
            q = coef[1+n:]
            k = 0
            for i in range(n):
                for j in range(i, n):
                    bb = q[k]
                    if i == j:
                        H[i, i] += 2.0 * bb
                    else:
                        H[i, j] += bb
                        H[j, i] += bb
                    k += 1

        try:
            eig, V = np.linalg.eigh(0.5*(H+H.T))
            H = (V * np.maximum(eig, cfg.eig_floor)) @ V.T
        except Exception:
            H = 0.5*(H+H.T) + cfg.eig_floor*np.eye(n)

        info = {"model": "nfp", "m_used": int(Phi_use.shape[0]), "p": int(p)}
        return g, H, info

# --------- Ridge (CSV) ---------

class DFORidgeModel(_ObjectiveModelBase):
    """
    Weighted ridge regression on a CSV polynomial basis.
    - Degree: 'auto' (default) uses quadratic if enough points, otherwise linear.
              You can force 1 or 2 via cfg.ridge_deg = 1|2.
    """

    def _basis(self, Y: np.ndarray, n: int, cfg) -> tuple[np.ndarray, int]:
        deg = getattr(cfg, "ridge_deg", "auto")
        if deg == 1:
            Phi = np.hstack([np.ones((Y.shape[0], 1)), Y])
            return Phi, 1
        if deg == 2:
            Phi, n_chk, _ = _csv_basis(Y)
            assert n_chk == n
            return Phi, 2
        # auto
        Phi2, n_chk, _ = _csv_basis(Y)
        assert n_chk == n
        p2 = Phi2.shape[1]
        if Y.shape[0] >= max(p2, getattr(cfg, "use_quadratic_if", n + 2)):
            return Phi2, 2
        Phi1 = np.hstack([np.ones((Y.shape[0], 1)), Y])
        return Phi1, 1

    def fit_objective(self, Y: np.ndarray, rhs: np.ndarray, n: int, cfg) -> tuple:
        Phi, deg = self._basis(Y, n, cfg)
        w = _huber_weights(rhs, cfg.huber_delta) * _distance_weights(Y, cfg.dist_w_power)
        coef = _solve_weighted_ridge(Phi, rhs, w, getattr(cfg, "ridge", 1e-6))

        g = np.zeros(n); H = np.zeros((n, n))
        if deg == 1:
            if len(coef) >= 1 + n:
                g[:] = coef[1:1+n]
        else:
            if len(coef) >= 1 + n:
                g[:] = coef[1:1+n]
                q = coef[1+n:]
                k = 0
                for i in range(n):
                    for j in range(i, n):
                        b = q[k]
                        if i == j: H[i, i] += 2.0*b
                        else: H[i, j] += b; H[j, i] += b
                        k += 1

        try:
            w_eig, V = np.linalg.eigh(0.5*(H+H.T))
            H = (V * np.maximum(w_eig, cfg.eig_floor)) @ V.T
        except Exception:
            H = 0.5*(H+H.T) + cfg.eig_floor*np.eye(n)

        info = {"model": "ridge", "deg": int(deg), "p": int(Phi.shape[1])}
        return g, H, info

# ======================================================================
# DFOExactPenalty: core + (optional) TRModel attachment
# ======================================================================

class DFOExactPenalty:
    def __init__(self, n, m_ineq, m_eq, cfg: "DFOConfig" = None, objective_model: _ObjectiveModelBase = None):
        self.n = n; self.m_ineq = m_ineq; self.m_eq = m_eq
        self.cfg = cfg or DFOConfig()
        self.X: List[np.ndarray] = []; self.F: List[float] = []; self.C: List[np.ndarray] = []
        self.last_diag: Dict = {}

        if objective_model is None:
            if getattr(self.cfg, "use_nfp_objective", False):
                self.obj_model = DFONFPModel()
            elif getattr(self.cfg, "use_rbf_objective", False):
                self.obj_model = DFORBFModel()
            elif getattr(self.cfg, "use_conn_mfn_objective", False):
                self.obj_model = DFOConnMFNModel()
            else:
                self.obj_model = DFORidgeModel()
        else:
            self.obj_model = objective_model

        # ---- Optional TRModel integration ----
        self.trmodel: Optional[TRModel] = None
        # Minimal options consumed by tr_model_ops.* (kept as attributes)
        class _TMOptions:
            pivot_threshold     = 1e-3
            exchange_threshold  = 1e-3
            tol_radius          = 1e-6
            add_threshold       = 1e-3
        self._tm_opts = _TMOptions()

    # -------------- TRModel integration --------------
    def attach_trmodel(self, trmodel: "TRModel", tm_options=None):
        """
        Attach an interpolation TRModel. When attached, samples fed to the DFO
        core are mirrored to the TR set; fit/criticality keep the set poised; and
        on accepted steps you can call core.on_accept(x_new, f, cI, cE) so the
        TR center tracks the iterate.
        """
        self.trmodel = trmodel
        if tm_options is not None:
            self._tm_opts = tm_options

    def _tm_funcs(self):
        """Adapter for tr_model_ops.ensure_improvement: (x) -> (fvec, ok)."""
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
        if self.trmodel.pointsAbs.size == 0:
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
        if self.trmodel.pointsAbs.size:
            X = self.trmodel.pointsAbs.T
            if np.any(np.linalg.norm(X - x[None, :], axis=1) <= 1e-12):
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
        # Rebuild if “old” or distant points appear
        try:
            if self.trmodel.hasDistantPoints(self._tm_opts) or self.trmodel.isOld(self._tm_opts):
                self.trmodel.rebuildModel(self._tm_opts)
        except Exception:
            pass
        # Ensure λ-poisedness by adding a few points if necessary
        tries = 0
        while not self.trmodel.isLambdaPoised(self._tm_opts) and tries <= max(1, self.n):
            ensure_improvement(self.trmodel, self._tm_funcs(), None, None, self._tm_opts)  # type: ignore
            tries += 1

    def on_accept(self, x_new: np.ndarray, f: float, cI: Optional[np.ndarray], cE: Optional[np.ndarray]):
        """Move TR center to the accepted iterate (call from the stepper)."""
        if self.trmodel is None:
            return
        fvec = np.concatenate((
            np.array([float(f)], dtype=float),
            np.asarray(cI, float).ravel() if (cI is not None and cI.size) else (np.zeros(self.m_ineq)),
            np.asarray(cE, float).ravel() if (cE is not None and cE.size) else (np.zeros(self.m_eq)),
        ))
        change_tr_center(self.trmodel, x_new, fvec, self._tm_opts)  # type: ignore

    # -------------- objective-model selection --------------
    def set_objective_model(self, model: _ObjectiveModelBase):
        self.obj_model = model

    # ---------------- sampling API ----------------
    def _is_duplicate_x(self, x: np.ndarray, atol: float = 1e-12, rtol: float = 1e-9) -> bool:
        if not self.X:
            return False
        X = np.vstack(self.X)
        dx = np.linalg.norm(X - x[None, :], axis=1)
        return bool(np.any(dx <= atol + rtol * max(1.0, np.linalg.norm(x))))

    def add_sample(self, x: np.ndarray, f: float, cI: np.ndarray = None, cE: np.ndarray = None):
        x = np.asarray(x, float).ravel()
        if self._is_duplicate_x(x):
            return
        ci = np.asarray(cI, float).ravel() if cI is not None else np.zeros(self.m_ineq)
        ce = np.asarray(cE, float).ravel() if cE is not None else np.zeros(self.m_eq)
        c = np.concatenate([ci, ce]) if (self.m_ineq + self.m_eq) > 0 else np.zeros(0)
        self.X.append(x); self.F.append(float(f)); self.C.append(c)

        # Mirror into TRModel if attached (cheap append; rebuild/poise handled elsewhere)
        if self.trmodel is not None:
            self._tm_append_if_new(x, f, ci, ce)

    def arrays(self):
        if not self.X:
            return np.zeros((0, self.n)), np.zeros((0,)), np.zeros((0, self.m_ineq + self.m_eq))
        return np.vstack(self.X), np.asarray(self.F, float), np.vstack(self.C)

    # ---------------- local fit ----------------
    def fit_local(self, center: np.ndarray, Delta: float) -> "FitResult":
        # Keep TR geometry healthy around this center (if attached)
        self._tm_seed_if_needed(center)
        self._tm_maintain_geometry(center, Delta)

        X, F, C = self.arrays()
        if X.shape[0] < max(3, self.n + 1):
            return self._fallback()

        max_pts = min(self.cfg.max_pts, X.shape[0])
        d = norm(X - center[None, :], axis=1)
        idx_close = np.argsort(d)[:max_pts]
        Xs, Fs = X[idx_close], F[idx_close]
        Cs = C[idx_close] if C.size else np.zeros((len(idx_close), self.m_ineq + self.m_eq))

        Delta = float(max(Delta, 1e-12))
        Y = (Xs - center[None, :]) / Delta

        # geometry subset + model radius
        k_geo = min(len(Y), max(self.n + 1, min(self.cfg.use_quadratic_if + 5, len(Y))))
        fps = _improved_fps(Y, k_geo)
        Y, Fs, Cs = Y[fps], Fs[fps], Cs[fps]
        mask_rad = (norm(Y, axis=1) <= self.cfg.model_radius_mult)
        if np.any(mask_rad) and mask_rad.sum() >= max(self.n + 1, 3):
            Y, Fs, Cs = Y[mask_rad], Fs[mask_rad], Cs[mask_rad]

        rhs = Fs - Fs[0]
        g, H, info = self.obj_model.fit_objective(Y, rhs, self.n, self.cfg)

        A_ineq = None; A_eq = None; Hc_list = None
        if self.m_ineq > 0:
            A_ineq = self._fit_jac(Y, Cs[:, :self.m_ineq]) / Delta
        if self.m_eq > 0:
            A_eq = self._fit_jac(Y, Cs[:, self.m_ineq:self.m_ineq + self.m_eq]) / Delta
        if self.m_ineq > 0 and Y.shape[0] >= self.cfg.use_quadratic_if:
            Hc_list = self._fit_constraints_diag_quadratic(Y, Cs[:, :self.m_ineq]) / (Delta**2)

        g = g / Delta; H = H / (Delta**2)

        self.last_diag = {
            "deg": 2 if Y.shape[0] >= self.cfg.use_quadratic_if else 1,
            "n_pts": int(Y.shape[0]),
            "used_conn_mfn": (info.get("model") == "conn_mfn"),
            "used_rbf": (info.get("model") == "rbf"),
            "rbf": info if info.get("model") == "rbf" else {},
            "obj_info": info,
        }
        return FitResult(g=g, H=H, A_ineq=A_ineq, A_eq=A_eq, Hc_list=Hc_list, diag=self.last_diag)

    # ---------- criticality / jacobian / diag-quadratic ----------
    def criticality_measure(self, g, A_ineq, A_eq, cI0, cE0, eps) -> Tuple[float, np.ndarray]:
        cI0 = np.asarray(cI0).ravel() if (cI0 is not None and self.m_ineq > 0) else np.zeros(self.m_ineq)
        cE0 = np.asarray(cE0).ravel() if (cE0 is not None and self.m_eq > 0) else np.zeros(self.m_eq)
        A_mask, V_mask, _ = self._split_sets(cI0, eps)
        E_mask = (np.abs(cE0) <= eps) if cE0.size else np.zeros(0, dtype=bool)
        grad_mp1 = self._mp1_grad(g, A_ineq, V_mask)
        d = self._desc_dir_lp(grad_mp1, A_ineq, A_mask)
        lin = float(g @ d); pen = 0.0
        if A_ineq is not None and A_mask.size and np.any(A_mask):
            pen += float(self.cfg.mu * np.maximum(0.0, (A_ineq[A_mask] @ d)).sum())
        if A_eq is not None and E_mask.size and np.any(E_mask):
            pen += float(self.cfg.mu * np.abs(A_eq[E_mask] @ d).sum())
        sigma = float(-(lin + pen))
        return sigma, d

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
            d = coef[n:]  # quad diag terms
            Hc_list.append(np.diag(np.maximum(d, 0.0)))
        return np.array(Hc_list, dtype=object)

    def _fallback(self) -> FitResult:
        return FitResult(
            g=np.zeros(self.n),
            H=self.cfg.eig_floor * np.eye(self.n),
            A_ineq=np.zeros((self.m_ineq, self.n)) if self.m_ineq else None,
            A_eq=np.zeros((self.m_eq, self.n)) if self.m_eq else None,
            Hc_list=None,
            diag={"deg": 1, "n_pts": 0, "used_conn_mfn": False, "used_rbf": False},
        )

    # ---------- EP pieces ----------
    def _split_sets(self, cI: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        A = np.abs(cI) <= eps   # nearly-active
        V = cI > eps            # violated
        S = cI < -eps           # satisfied
        return A, V, S

    def _mp1_grad(self, g: np.ndarray, A_ineq: Optional[np.ndarray], Vmask: np.ndarray) -> np.ndarray:
        if A_ineq is None or not np.any(Vmask):
            return g.copy()
        return g + self.cfg.mu * A_ineq[Vmask].sum(axis=0)

    # ===== multiplier-step helpers =====
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
            w_eig, V = np.linalg.eigh(0.5*(H_eff+H_eff.T))
            H_eff = (V * np.maximum(w_eig, self.cfg.eig_floor)) @ V.T
        except Exception:
            H_eff = 0.5*(H_eff+H_eff.T) + self.cfg.eig_floor*np.eye(self.n)
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

    def _model_mp_value(self, t: float, d: np.ndarray,
                        g: np.ndarray, H: np.ndarray,
                        cI0: np.ndarray,
                        A_ineq: Optional[np.ndarray],
                        Hc_list: Optional[List[np.ndarray]]) -> float:
        f_quad = 0.5 * t * t * (d @ (H @ d)) + t * (g @ d)
        pen = 0.0
        if A_ineq is not None and A_ineq.size:
            aTd = A_ineq @ d
            c_mid = cI0 + t * aTd
            pos_idx = np.flatnonzero(c_mid > 0.0)
            if pos_idx.size:
                pen += self.cfg.mu * float(np.sum(cI0[pos_idx] + t * aTd[pos_idx]))
                if Hc_list is not None:
                    for idx in pos_idx:
                        if idx < len(Hc_list) and Hc_list[idx] is not None:
                            q = float(d @ (Hc_list[idx] @ d))
                            pen += 0.5 * self.cfg.mu * q * t * t
        return f_quad + pen

    # ---------- LP direction ----------
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

        # Gurobi (optional)
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

        # SciPy HiGHS
        try:
            c = np.concatenate([grad_mp1, mu * np.ones(mA)])
            bounds = [(-1, 1)] * n + [(0, None)] * mA
            A_ub = np.hstack([-Ai, np.eye(mA)])
            b_ub = np.zeros(mA)
            options = {"presolve": True,
                       "time_limit": float(getattr(self.cfg, "lp_time_limit", 0.10)),
                       "dual_feasibility_tolerance": 1e-9,
                       "primal_feasibility_tolerance": 1e-9}
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs", options=options)
            if res.success and res.x is not None:
                x = res.x[:n]
                if np.linalg.norm(x, ord=np.inf) > 1e-12:
                    return x
        except Exception:
            pass

        g = grad_mp1
        return -g / (np.linalg.norm(g) + 1e-16)

    # ---------- conservative prediction ----------
    def _conservative_pred_red(self, sigma: float, Delta: float, n: int) -> float:
        kfd, kH, kc = self.cfg.k_fd, self.cfg.k_H, self.cfg.k_c
        piece = min(sigma / max(n*n*kH, 1e-16),
                    self.cfg.eps_c / max(n*kc, 1e-16),
                    Delta)
        return max(1e-16, 0.5 * kfd * sigma * piece)

    # ---------- breakpoints ----------
    def _breakpoints(self, c0: np.ndarray, d: np.ndarray,
                     A_ineq: Optional[np.ndarray], Hc_list: Optional[List[np.ndarray]], Delta: float) -> List[float]:
        t_list: List[float] = []
        if A_ineq is None or A_ineq.shape[0] == 0:
            return t_list

        infn = norm(d, ord=np.inf)
        if infn <= 1e-16:
            return t_list
        t_tr = Delta / infn

        den_floor = 1e-10
        min_sep = 1e-10

        aTd = A_ineq @ d
        for i in range(A_ineq.shape[0]):
            a = float(c0[i]); b = float(aTd[i])
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

        if not t_list:
            return t_list

        t_list = sorted(set(t_list))
        pruned = []
        for t in t_list:
            if not pruned or (t - pruned[-1]) > min_sep:
                pruned.append(t)
        if len(pruned) > 64:
            pruned = pruned[:64]
        return pruned

    # ---------- VK correction ----------
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
            hi =  box - h
            return np.minimum(hi, np.maximum(lo, z))

        prev_rn = np.inf
        for k in range(self.cfg.vk_gn_maxit):
            r_list = []; J_list = []

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

    # ---------- violation metric ----------
    def _violation_measure(self, cI: Optional[np.ndarray], cE: Optional[np.ndarray]) -> float:
        vI = float(np.maximum(0.0, cI).max()) if (cI is not None and cI.size) else 0.0
        vE = float(np.abs(cE).max()) if (cE is not None and cE.size) else 0.0
        return max(vI, vE)

    # -------- ORACLE PLUMBING --------
    def set_oracle(self, oracle_fn):
        """
        Register an oracle function:
            oracle_fn(x: np.ndarray) -> (f: float, cI: np.ndarray|None, cE: np.ndarray|None)
        """
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
            f, cI = out; cE = None
        elif len(out) == 3:
            f, cI, cE = out
        else:
            raise ValueError("Oracle must return (f, cI) or (f, cI, cE).")
        f = float(np.asarray(f, dtype=float))
        cI = self._coerce_vec(cI, self.m_ineq, "cI")
        cE = self._coerce_vec(cE, self.m_eq,   "cE")
        return f, cI, cE

    # -------- INTERNAL: subset mirroring fit_local --------
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
        k_geo = min(len(Y), max(self.n + 1, min(self.cfg.use_quadratic_if + 5, len(Y))))
        fps = _improved_fps(Y, k_geo)
        Xs, Fs, Cs, Y = Xs[fps], Fs[fps], Cs[fps], Y[fps]

        mask_rad = (norm(Y, axis=1) <= self.cfg.model_radius_mult)
        if np.any(mask_rad) and mask_rad.sum() >= max(self.n + 1, 3):
            Xs, Fs, Cs = Xs[mask_rad], Fs[mask_rad], Cs[mask_rad]
        return Xs, Fs, Cs

    # ---------- FL hooks ----------
    def _is_FL_enough(self, center: np.ndarray, Delta: float) -> bool:
        Xs, Fs, Cs = self._pick_working_subset(center, Delta)
        m = Xs.shape[0]
        if m < self.n + 1:
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

    def _improve_models_FL(self, center: np.ndarray, Delta: float, budget: int = 2) -> None:
        if budget <= 0:
            return
        Xs, Fs, Cs = self._pick_working_subset(center, Delta)
        m = Xs.shape[0]
        if m == 0:
            return

        Delta = float(max(Delta, 1e-12))
        Y = (Xs - center[None, :]) / Delta
        Phi, _, _ = _csv_basis(Y)
        p = Phi.shape[1]

        # --- remove (a few) lowest-leverage rows
        X_all = np.vstack(self.X)
        global_map = []
        for xi in Xs:
            # exact matching preferred; otherwise nearest match
            cand = np.where((X_all == xi).all(axis=1))[0]
            j = int(cand[0]) if cand.size else int(np.argmin(np.linalg.norm(X_all - xi[None, :], axis=1)))
            global_map.append(j)
        global_map = np.asarray(global_map, dtype=int)

        if m >= p:
            try:
                sel_sq = _select_poised_rows(Phi, p)
                # rough leverage surrogate
                lev = np.sum((Phi @ np.linalg.pinv(Phi[sel_sq]))**2, axis=1)
            except Exception:
                lev = np.linalg.norm(Phi, axis=1)**2

            low_idx_local = np.argsort(lev)[:min(budget, len(lev))]
            to_drop_global = set(global_map[low_idx_local].tolist())
            if to_drop_global:
                for gidx in sorted(to_drop_global, reverse=True):
                    del self.X[gidx]; del self.F[gidx]; del self.C[gidx]

        # --- add (a few) maximin points on the l∞ box
        if not hasattr(self, "oracle") or self.oracle is None:
            return

        n = self.n
        box = Delta * self.cfg.model_radius_mult
        cand = []

        # ±axes
        for i in range(n):
            e = np.zeros(n); e[i] = 1.0
            cand.append(center + box * e)
            cand.append(center - box * e)

        # corners (random subset)
        if n <= 10:
            for _ in range(min(2*n, 64)):
                sgn = np.sign(np.random.randn(n))
                sgn[sgn == 0] = 1.0
                cand.append(center + box * sgn)

        # random unit l∞ directions
        for _ in range(max(10, 5*n)):
            d = np.random.randn(n)
            d /= max(norm(d, ord=np.inf), 1e-12)
            cand.append(center + box * d)

        cand = np.unique(np.asarray(cand), axis=0)

        X_cur, _, _ = self.arrays()
        if X_cur.shape[0] > 0:
            D = cdist(cand, X_cur, metric="euclidean")
            minD = np.min(D, axis=1)
        else:
            minD = np.full(len(cand), np.inf)

        pick = np.argsort(-minD)[:budget]
        to_add = cand[pick]
        for x_new in to_add:
            f_new, cI_new, cE_new = self.oracle_eval(x_new)
            self.add_sample(x_new, f_new, cI_new, cE_new)

    def criticality_loop(self, xk: np.ndarray, eps: float, Delta_in: float) -> Tuple[float, float, float]:
        beta1 = float(self.cfg.crit_beta1)
        beta2 = float(self.cfg.crit_beta2)
        beta3 = float(self.cfg.crit_beta3)

        Delta = float(Delta_in)
        eps_c = float(eps)
        sigma_last = np.inf

        # keep TR geometry alive
        self._tm_seed_if_needed(xk)
        self._tm_maintain_geometry(xk, Delta)

        X, _, C = self.arrays()
        if X.shape[0]:
            j0 = int(np.argmin(np.linalg.norm(X - xk[None, :], axis=1)))
            cI0 = C[j0, :self.m_ineq] if self.m_ineq else np.zeros(0)
            cE0 = C[j0, self.m_ineq:self.m_ineq + self.m_eq] if self.m_eq else np.zeros(0)
        else:
            cI0 = np.zeros(self.m_ineq); cE0 = np.zeros(self.m_eq)

        for _ in range(20):
            if not self._is_FL_enough(xk, Delta):
                self._improve_models_FL(xk, Delta)

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

    # ---------------- single step proposal ----------------
    def propose_step(self, x: np.ndarray, Delta: float, fit: FitResult,
                     cI0: Optional[np.ndarray], cE0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        g, H, A_ineq, A_eq, Hc_list = fit.g, fit.H, fit.A_ineq, fit.A_eq, fit.Hc_list
        cI0 = np.asarray(cI0).ravel() if (cI0 is not None and self.m_ineq > 0) else np.zeros(self.m_ineq)
        cE0 = np.asarray(cE0).ravel() if (cE0 is not None and self.m_eq > 0) else np.zeros(self.m_eq)

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

        lin_term = float(g @ d_default); pen_term = 0.0
        if A_ineq is not None and np.any(A_mask):
            pen_term += float(self.cfg.mu * np.maximum(0.0, (A_ineq[A_mask] @ d_default)).sum())
        if A_eq is not None and np.any(E_mask):
            pen_term += float(self.cfg.mu * np.abs(A_eq[E_mask] @ d_default).sum())
        sigma = float(-(lin_term + pen_term))

        if sigma < self.cfg.eps_c:
            sig = max(sigma, 1e-16)
            target1 = max(self.cfg.crit_beta1 * sig, 1e-12)
            target2 = self.cfg.crit_beta2 * sig
            Delta = min(Delta, target1, target2)

        breaks = [0.0] + self._breakpoints(cI0, d_default, A_ineq, Hc_list, Delta)
        t_tr = Delta / max(norm(d_default, ord=np.inf), 1e-16)
        if not breaks or breaks[-1] < t_tr:
            breaks.append(t_tr)

        def mp_interval_value(t: float, d: np.ndarray) -> float:
            return self._model_mp_value(t, d, g, H, cI0, A_ineq, Hc_list)

        t_best = 0.0; val_best = 0.0
        for a, b in zip(breaks[:-1], breaks[1:]):
            t_mid = 0.5 * (a + b)
            val = mp_interval_value(t_mid, d_default)
            if t_best == 0.0 or val < val_best:
                coef2 = float(d_default @ (H @ d_default))
                slope_pen = 0.0
                if A_ineq is not None and A_ineq.size:
                    aTd_all = (A_ineq @ d_default)
                    c_mid = cI0 + t_mid * aTd_all
                    pos_mask = np.flatnonzero(c_mid > 0.0)
                    if pos_mask.size:
                        slope_pen += self.cfg.mu * float(np.sum(aTd_all[pos_mask]))
                        if Hc_list is not None:
                            for idx in pos_mask:
                                if idx < len(Hc_list) and Hc_list[idx] is not None:
                                    slope_pen += self.cfg.mu * float(d_default @ (Hc_list[idx] @ d_default)) * t_mid
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

        used_mult = False
        s_choice = s_default
        pred_red_quad_choice = pred_red_quad_default
        s_mult = None

        if self.cfg.use_multiplier_step and sigma <= self.cfg.mult_sigma_thresh:
            lam = self._estimate_multipliers(grad_mp1=grad_mp1, A_ineq=A_ineq, A_mask=A_mask)
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

        theta0 = self._violation_measure(cI0, cE0)
        cI_def = cI0 + (A_ineq @ s_default) if (A_ineq is not None and s_default is not None and s_default.size) else None
        cE_def = cE0 + (A_eq   @ s_default) if (A_eq   is not None and s_default is not None and s_default.size) else None
        theta_def = self._violation_measure(cI_def, cE_def)
        theta_mult = None
        if s_mult is not None:
            cI_mul = cI0 + (A_ineq @ s_mult) if (A_ineq is not None and s_mult.size) else None
            cE_mul = cE0 + (A_eq   @ s_mult) if (A_eq   is not None and s_mult.size) else None
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
