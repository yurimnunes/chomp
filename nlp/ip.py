# sqp_ip.py
# InteriorPointStepper: slacks-only barrier IPM with "reference" mode (μ/s²)
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import qdldl_cpp as qd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import LinearOperator, cg

from nlp.blocks.linesearch import LineSearcher

from .blocks.aux import HessianManager, Model, SQPConfig
from .blocks.filter import Filter, Funnel
from .blocks.reg import Regularizer, make_psd_advanced


# ------------------ sparse helpers ------------------
def _isspm(A) -> bool:
    return sp.isspmatrix(A)


def _csr(A, shape=None):
    if A is None:
        if shape is None:
            raise ValueError("shape required when A is None")
        return sp.csr_matrix(shape)
    return A.tocsr() if _isspm(A) else sp.csr_matrix(A)


def _sym(A):
    if _isspm(A):
        return (A + A.T) * 0.5
    return 0.5 * (A + A.T)


def _diag(v: np.ndarray):
    return sp.diags(v) if v.ndim == 1 else sp.diags(np.asarray(v).ravel())


# ------------------ IP state ------------------
@dataclass
class IPState:
    mI: int
    mE: int
    s: np.ndarray
    lam: np.ndarray
    nu: np.ndarray
    # z: np.ndarray  # duals for bounds x >= 0 (new: primal-dual for bounds)
    zL: np.ndarray  # duals for lower bounds
    zU: np.ndarray  # duals for upper bounds
    mu: float
    initialized: bool = False

    @staticmethod
    def from_model(model: Model, x: np.ndarray, cfg: SQPConfig) -> "IPState":
        mI, mE = model.m_ineq, model.m_eq
        d = model.eval_all(x, components=["cI", "cE"])
        cI = (
            np.zeros(mI) if (mI == 0 or d["cI"] is None) else np.asarray(d["cI"], float)
        )
        s0 = np.maximum(1.0, -cI + 1.0) if mI > 0 else np.zeros(0)
        lam = np.ones_like(s0)
        nu = np.zeros(mE)
        mu0 = max(getattr(cfg, "ip_mu_init", 1e-1), 1e-12)
        # z0 = np.ones_like(x)  # new: initialize z for bounds
        # bounds (ℓ ≤ x ≤ u); treat ±∞ as inactive
        lb = getattr(model, "lb", None)
        ub = getattr(model, "ub", None)
        lb = np.full_like(x, -np.inf) if lb is None else np.asarray(lb, float)
        ub = np.full_like(x, +np.inf) if ub is None else np.asarray(ub, float)
        # Active masks
        hasL = np.isfinite(lb)
        hasU = np.isfinite(ub)
        # Initialize zL, zU strictly positive where active, else 0
        zL0 = np.where(hasL, 1.0, 0.0) * np.ones_like(x)
        zU0 = np.where(hasU, 1.0, 0.0) * np.ones_like(x)
        return IPState(
            mI=mI, mE=mE, s=s0, zL=zL0, zU=zU0, lam=lam, nu=nu, mu=mu0, initialized=True
        )


def _get_bounds(model: Model, x: np.ndarray):
    """Return (lb, ub, hasL, hasU) as float arrays and boolean masks."""
    n = x.size
    lb = getattr(model, "lb", None)
    ub = getattr(model, "ub", None)
    lb = (
        np.full(n, -np.inf, dtype=float)
        if lb is None
        else np.asarray(lb, float).reshape(-1)
    )
    ub = (
        np.full(n, +np.inf, dtype=float)
        if ub is None
        else np.asarray(ub, float).reshape(-1)
    )
    hasL = np.isfinite(lb)
    hasU = np.isfinite(ub)
    return lb, ub, hasL, hasU


def _safe_pos_div(num: np.ndarray, den: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return num / np.maximum(den, eps)


def safe_inf_norm(v: np.ndarray) -> float:
    """Compute infinity norm safely, handling empty arrays."""
    if v is None or v.size == 0:
        return 0.0
    return np.linalg.norm(v, np.inf)


def _as_array_or_zeros(a, n: int) -> np.ndarray:
    if a is None:
        return np.zeros(n, dtype=float)
    a = np.asarray(a, dtype=float)
    return a if a.size else np.zeros(n, dtype=float)


# --- small helpers (shape/CSR safe) ---
def _as_array_or_zeros(v, m):
    if m == 0:
        return np.zeros(0, dtype=float)
    if v is None:
        return np.zeros(m, dtype=float)
    a = np.asarray(v, dtype=float).ravel()
    if a.size != m:
        a = a.reshape(m)
    return a


def _csr(A, shape=None):
    if A is None:
        if shape is None:
            raise ValueError("shape required when A is None")
        return sp.csr_matrix(shape)
    return A if sp.isspmatrix(A) else sp.csr_matrix(A)


# --- the restoration model ---
class RestorationModel:
    """
    bar_x = [ x (n) ; p (m) ; n (m) ]  where m = base.m_eq + base.m_ineq
    Equalities: cE_rest(bar_x) = [cE_base(x); cI_base(x)] - p + n = 0
    Inequalities: none (p,n >= 0 are simple bounds handled by the IP/barrier)
    Objective: f = (zeta/2)*(x-x_R)^T D_R (x-x_R) + 1^T p + 1^T n
    """

    def __init__(self, base_model, x_R, D_R, zeta: float, rho: float = 1.0):
        self.base = base_model
        self.x_R = np.asarray(x_R, dtype=float)
        self.D_R = _csr(D_R, (base_model.n, base_model.n))
        self.zeta = float(zeta)
        self.rho = float(rho)

        # sizes
        self.n_orig = int(base_model.n)
        self.m_base_E = int(base_model.m_eq)
        self.m_base_I = int(base_model.m_ineq)
        self.m_eq = self.m_base_E + self.m_base_I
        self.m_ineq = 0
        self.n = self.n_orig + 2 * self.m_eq  # x + p + n

        # -------- bounds so IP sees p>=0, n>=0 and original x-bounds --------
        # Original x bounds (default to ±inf if base does not define):
        lb_x = getattr(self.base, "lb", None)
        ub_x = getattr(self.base, "ub", None)
        lb_x = (np.full(self.n_orig, -np.inf, float) if lb_x is None
                else np.asarray(lb_x, float).reshape(-1))
        ub_x = (np.full(self.n_orig, +np.inf, float) if ub_x is None
                else np.asarray(ub_x, float).reshape(-1))

        if self.m_eq > 0:
            zeros_m = np.zeros(self.m_eq, dtype=float)
            inf_m   = np.full(self.m_eq, +np.inf, dtype=float)
            # p >= 0, n >= 0
            self.lb = np.concatenate([lb_x, zeros_m, zeros_m])
            self.ub = np.concatenate([ub_x,  inf_m,   inf_m])
        else:
            self.lb = lb_x.copy()
            self.ub = ub_x.copy()

    # ------------ splitting / joining ------------
    def split(self, bar_x):
        n = self.n_orig
        m = self.m_eq
        if m > 0:
            x = bar_x[:n]
            p = bar_x[n : n + m]
            nn = bar_x[n + m : n + 2 * m]
        else:
            x = bar_x[:n]
            p = np.zeros(0, dtype=float)
            nn = np.zeros(0, dtype=float)
        return x, p, nn

    # ------------ scalar objective ------------
    def f_scalar(self, bar_x):
        x, p, nn = self.split(bar_x)
        dx = x - self.x_R
        quad = 0.5 * self.zeta * float(dx @ (self.D_R @ dx))
        lin = self.rho * (p.sum() + nn.sum())
        return quad + lin

    # ------------ API: eval_all ------------
    def eval_all(self, bar_x, components=("f", "g", "cE", "JE", "H")):
        """
        Returns a dict with any subset of:
          - "f": scalar objective at bar_x
          - "g": gradient wrt bar_x (shape [n + 2m])
          - "cE": equality residuals cE_rest(bar_x)  (shape [m])
          - "cI": empty array (shape [0])
          - "JE": Jacobian of cE_rest wrt bar_x (shape [m × (n+2m)])
          - "JI": shape [0 × (n+2m)]
          - "H" : Hessian of f wrt bar_x (sparse CSR)
          - "lagrangian_hessian": same as H here (no constraint curvature)
        """
        need_f = "f" in components
        need_g = "g" in components
        need_c = any(k in components for k in ("cE", "JE", "Aeq", "beq"))
        need_J = "JE" in components
        need_H = ("H" in components) or ("lagrangian_hessian" in components)

        out = {}

        # ----- cE & JE (restoration equalities) -----
        if need_c or need_J:
            x, p, nn = self.split(bar_x)

            # Query base model once
            # We need base cE, cI, JE, JI (all wrt x)
            d = self.base.eval_all(x, components=["cE", "cI", "JE", "JI"])
            cE = _as_array_or_zeros(d.get("cE", None), self.m_base_E)
            cI = _as_array_or_zeros(d.get("cI", None), self.m_base_I)

            JE_x = _csr(d.get("JE", None), (self.m_base_E, self.n_orig))
            JI_x = _csr(d.get("JI", None), (self.m_base_I, self.n_orig))

            # Stack base constraints
            if self.m_eq > 0:
                c_stack = (
                    np.concatenate([cE, cI])
                    if (self.m_base_E and self.m_base_I)
                    else (cE if self.m_base_E else cI)
                )
                Jx = (
                    sp.vstack([JE_x, JI_x], format="csr")
                    if (self.m_base_E and self.m_base_I)
                    else (JE_x if self.m_base_E else JI_x)
                )
            else:
                c_stack = np.zeros(0, dtype=float)
                Jx = sp.csr_matrix((0, self.n_orig))

            # c_rest = c_stack - p + nn
            if self.m_eq > 0:
                c_rest = c_stack - p + nn
            else:
                c_rest = c_stack  # empty

            out["cE"] = c_rest
            out["cI"] = np.zeros(0, dtype=float)  # no inequalities here

            if need_J:
                # JE_rest = [ Jx | -I | +I ]
                if self.m_eq > 0:
                    I = sp.eye(self.m_eq, format="csr")
                    JE = sp.hstack([Jx, -I, I], format="csr")
                else:
                    JE = sp.csr_matrix((0, self.n_orig))  # 0 × n (no p,n blocks)
                out["JE"] = JE

                # Always provide an empty JI with consistent width
                out["JI"] = sp.csr_matrix((0, self.n))  # 0 × (n+2m)

        # ----- f & g -----
        if need_f or need_g:
            x, p, nn = self.split(bar_x)
            dx = x - self.x_R
            if need_f:
                quad = 0.5 * self.zeta * float(dx @ (self.D_R @ dx))
                lin = self.rho * (p.sum() + nn.sum())
                out["f"] = quad + lin

            if need_g:
                # g_x = zeta * D_R * (x - x_R)
                gx = self.zeta * (self.D_R @ dx)
                if self.m_eq > 0:
                    gp = self.rho * np.ones(self.m_eq, dtype=float)
                    gn = self.rho * np.ones(self.m_eq, dtype=float)
                    out["g"] = np.concatenate([np.asarray(gx).ravel(), gp, gn])
                else:
                    out["g"] = np.asarray(gx).ravel()

        # ----- Hessian (objective-only) -----
        if need_H:
            if self.m_eq > 0:
                Zm = sp.csr_matrix((self.m_eq, self.m_eq))
                H = sp.block_diag((self.zeta * self.D_R, Zm, Zm), format="csr")
            else:
                H = self.zeta * self.D_R
            out["H"] = H
            if "lagrangian_hessian" in components:
                out["lagrangian_hessian"] = H  # no constraint curvature added

        return out


def _safe_ratio_pos(
    num: np.ndarray,
    den: np.ndarray,
    eps_abs: float = 1e-12,
    eps_rel: float = 1e-12,
    clip_min: float = 1e-12,
    clip_max: float = 1e12,
    both_small_value: float = 1.0,
) -> np.ndarray:
    """
    Compute num/den for nonnegative vectors robustly.

    - Denominator is floored by eps_abs + eps_rel*max(1, den).
    - When both num and den are ~0, returns both_small_value (keeps diagonal sane).
    - Result is clipped to [clip_min, clip_max] to avoid ill-conditioning.
    """
    den_floor = eps_abs + eps_rel * np.maximum(1.0, den)
    # where both ~0 -> set to benign constant (e.g., 1.0)
    both_small = (num <= den_floor) & (den <= den_floor)
    out = num / np.maximum(den, den_floor)
    if both_small_value is not None:
        out = np.where(both_small, both_small_value, out)
    # final safety clamp
    return np.clip(out, clip_min, clip_max)


# ------------------ stepper ------------------
class InteriorPointStepper:
    """
    Slacks-only barrier IP stepper:
        min f(x) - μ Σ log s
        s.t. c_I(x) + s = 0,  c_E(x) = 0,  s>0

    Modes:
      - Reference-form (cfg.ip_match_ref_form=True): W = H + JI^T diag(μ/s^2) JI,
        rhs_x = -r1 - JI^T(-r2 + μ/s^2 * r3), ds = -r3 - JI dx, dλ = -r2 - μ/s^2 ds
      - Mehrotra PC (cfg.ip_match_ref_form=False): classic λ/s scaling with predictor-corrector
    """

    def __init__(
        self,
        cfg: SQPConfig,
        hess: HessianManager,
        regularizer: Regularizer,
        tr=None,
        flt=None,
        funnel=None,
        ls=None,  # <-- new: pass a LineSearcher instance here
        soc=None,  # <-- new: pass a SOCCorrector instance here
    ):
        self.cfg = cfg
        self.hess = hess
        self.reg = regularizer
        self.tr = tr
        self.filter = None
        self.funnel = Funnel(self.cfg)
        self.ls = LineSearcher(self.cfg, None, self.funnel) if ls is None else ls
        self.soc = soc  # may be None; if None we skip SOC rescue
        self._ensure_cfg_defaults()

    def _ensure_cfg_defaults(self):
        def add(name, val):
            if not hasattr(self.cfg, name):
                setattr(self.cfg, name, val)

        # algebra / hessian mode
        add("ip_match_ref_form", False)
        add("ip_exact_hessian", True)
        add("ip_hess_reg0", 1e-4)
        add("ip_eq_reg", 1e-4)

        # barrier & predictor-corrector (used only if ip_match_ref_form=False)
        add("ip_mu_init", 1e-2)
        add("ip_mu_min", 1e-12)
        add("ip_sigma_power", 3.0)
        add("ip_tau", 0.995)
        add("ip_alpha_max", 1.0)

        # legacy IP line-search knobs (kept for fallback)
        add("ip_ls_max", 25)
        add("ip_alpha_min", 1e-10)
        add("ip_alpha_backtrack", 0.5)
        add("ip_armijo_coeff", 1e-4)

        # μ schedule (reference-style)
        add("ip_mu_reduce_every", 10)
        add("ip_mu_reduce", 0.2)

        # feasibility penalty for merit (used by barrier merit; LineSearcher does not need it)
        add("ip_rho_init", 1.0)
        add("ip_rho_inc", 10.0)
        add("ip_rho_max", 1e6)

        add("ip_kappa_eps", 10.0)  # kappa_epsilon
        add("ip_kappa_mu", 0.2)
        add("ip_theta_mu", 1.5)
        add("ip_tau_min", 0.99)
        add("ip_gamma_theta", 1e-5)
        add("ip_gamma_phi", 1e-5)
        add("ip_delta", 1.0)
        add("ip_s_theta", 1.1)
        add("ip_s_phi", 2.3)
        add("ip_eta_phi", 1e-4)
        add("ip_kappa_soc", 0.99)
        add("ip_pmax_soc", 4)
        add("ip_kappa_sigma", 1e10)  # kappa_Sigma for z correction
        add("ip_delta_w_min", 1e-20)  # for inertia correction
        add("ip_delta_w_0", 1e-4)
        add("ip_delta_w_max", 1e40)
        add("ip_kappa_w_plus_bar", 100.0)
        add("ip_kappa_w_plus", 8.0)
        add("ip_kappa_w_minus", 1 / 3)
        add("ip_kappa_c", 0.25)
        add("ip_delta_c_bar", 1e-8)
        # --- restoration controls (feasibility-only subproblem) ---
        add("ip_rest_max_it", 8)  # IP iterations per restoration attempt
        add("ip_rest_max_retry", 5)  # number of ζ escalations
        add("ip_rest_mu", 1.0)  # starting μ for the restoration IP
        add("ip_rest_zeta_init", 1.0)  # proximal weight ζ initial
        add("ip_rest_zeta_update", 5.0)  # ζ multiplier each retry (≥ 2)
        add("ip_rest_zeta_max", 1e8)  # cap for ζ
        add("ip_rest_theta_drop", 0.5)  # accept if θ_new ≤ θ_R * this
        add("ip_rest_theta_abs", 1e-6)  # absolute θ target floor (use with tol)

        add("funnel_kappa_initial", 1.5)  # initial funnel kappa

        # --------- bridge legacy ip_* to generic ls_* if missing ---------
        # (we don't overwrite ls_* if you already set them elsewhere)
        if not hasattr(self.cfg, "ls_backtrack") and hasattr(
            self.cfg, "ip_alpha_backtrack"
        ):
            self.cfg.ls_backtrack = float(self.cfg.ip_alpha_backtrack)
        if not hasattr(self.cfg, "ls_armijo_f") and hasattr(
            self.cfg, "ip_armijo_coeff"
        ):
            self.cfg.ls_armijo_f = float(self.cfg.ip_armijo_coeff)
        if not hasattr(self.cfg, "ls_max_iter") and hasattr(self.cfg, "ip_ls_max"):
            self.cfg.ls_max_iter = int(self.cfg.ip_ls_max)
        if not hasattr(self.cfg, "ls_min_alpha") and hasattr(self.cfg, "ip_alpha_min"):
            self.cfg.ls_min_alpha = float(self.cfg.ip_alpha_min)

    def solve_KKT(
        self,
        Wmat,
        rhs_x,
        JE_mat,
        rpE,
        use_ordering: bool = False,
        reuse_symbolic: bool = True,
        refine_iters: int = 0,
        *,
        method: str = "hykkt",  # "qdldl" (default), "hykkt", "lifted"
        gamma: float = None,  # HyKKT parameter (if None, auto-heuristic)
        delta_c_lift: float = None,  # Lifted-KKT parameter (if None, auto-heuristic)
        cg_tol: float = 1e-10,
        cg_maxit: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve KKT:
            [ W   G^T ][dx] = [ rhs_x ]
            [ G    0  ][dy]   [ -rpE   ]

        method:
        - "qdldl"  : inertia-corrected LDL^T with qdldl (your default)
        - "hykkt"  : SPD K_gamma = (W + δ_w I) + γ G^T G, CG on Schur
        - "lifted" : SPD Schur with finite δ_c (dy = 1/δ_c (G dx - r2))
        """

        def _csr(A, shape=None):
            if A is None:
                if shape is None:
                    raise ValueError("shape required when A is None")
                return sp.csr_matrix(shape)
            return A.tocsr() if sp.isspmatrix(A) else sp.csr_matrix(A)

        def _upper_csc(K):
            # assumes K is symmetric; extract upper triangle in CSC
            K = K.tocoo()
            mask = K.row <= K.col
            Kup = sp.coo_matrix(
                (K.data[mask], (K.row[mask], K.col[mask])), shape=K.shape
            )
            return Kup.tocsc()

        def _normest_rowsum_inf(A: sp.spmatrix) -> float:
            A = _csr(A)
            return float(np.max(np.abs(A).sum(axis=1))) if A.shape[0] else 1.0

        n = Wmat.shape[0]
        mE = 0 if JE_mat is None else JE_mat.shape[0]
        W = _csr(Wmat, (n, n))
        G = None if JE_mat is None else _csr(JE_mat, (mE, n))
        r1 = rhs_x
        r2 = -rpE if mE > 0 else None

        # ---------------------------
        # SPD paths: HyKKT / Lifted
        # ---------------------------
        if method in ("hykkt", "lifted") and mE > 0:
            delta_w_last = float(getattr(self, "_delta_w_last", 0.0) or 0.0)
            # Heuristic scale for gamma (or 1/delta_c): ||W|| / ||G||^2 (row-sum ∞-norm)
            if gamma is None or delta_c_lift is None:
                num = _normest_rowsum_inf(W) + delta_w_last
                den = max(_normest_rowsum_inf(G), 1.0)
                gamma_hat = max(1.0, num / (den * den))  # ≈ ||W|| / ||G||^2

            if method == "hykkt":
                # K_gamma = (W + δ_w I) + γ G^T G  (SPD)
                if gamma is None:
                    gamma = gamma_hat
                Kgam = W + delta_w_last * sp.eye(n, format="csr") + gamma * (G.T @ G)
                Kgam_csc = Kgam.tocsc()

                # factor/solve helper (reuse_symbolic hook left minimal here)
                def solve_Kgam(b):
                    return spla.spsolve(Kgam_csc, b)

                # Schur operator S y = G K_gamma^{-1} G^T y  (SPD)
                def S_mv(y):
                    return G @ solve_Kgam(G.T @ y)

                S = LinearOperator((mE, mE), matvec=S_mv, rmatvec=S_mv, dtype=float)

                # RHS for Schur: G K^{-1}(r1 + γ G^T r2) - r2
                svec = r1 + gamma * (G.T @ r2)
                rhs_schur = (G @ solve_Kgam(svec)) - r2

                dy, info = cg(S, rhs_schur, rtol=cg_tol, maxiter=cg_maxit)
                if info != 0:
                    # One retry with larger gamma; otherwise fall back to qdldl
                    gamma *= 5.0
                    Kgam = (
                        W + delta_w_last * sp.eye(n, format="csr") + gamma * (G.T @ G)
                    )
                    Kgam_csc = Kgam.tocsc()

                    def S_mv2(y):
                        return G @ spla.spsolve(Kgam_csc, G.T @ y)

                    S2 = LinearOperator(
                        (mE, mE), matvec=S_mv2, rmatvec=S_mv2, dtype=float
                    )
                    svec = r1 + gamma * (G.T @ r2)
                    rhs_schur = (G @ spla.spsolve(Kgam_csc, svec)) - r2
                    dy, info = cg(
                        S2, rhs_schur, tol=max(cg_tol, 1e-8), maxiter=cg_maxit
                    )
                    if info != 0:
                        method = "qdldl"  # fallback below

                if method == "hykkt" and info == 0:
                    rhs_dx = r1 + gamma * (G.T @ r2) - (G.T @ dy)
                    dx = solve_Kgam(rhs_dx)
                    # keep previous δ_w memory
                    self._delta_w_last = delta_w_last
                    return dx, dy

            if method == "lifted":
                # (K + (1/δ_c) G^T G) dx = r1 - (1/δ_c) r2 ; dy = (G dx - r2)/δ_c
                if delta_c_lift is None:
                    delta_c_lift = 1.0 / gamma_hat
                Kspd = (
                    W
                    + delta_w_last * sp.eye(n, format="csr")
                    + (1.0 / delta_c_lift) * (G.T @ G)
                )
                rhs_dx = r1 - (1.0 / delta_c_lift) * r2
                dx = spla.spsolve(Kspd.tocsc(), rhs_dx)
                dy = (G @ dx - r2) * (1.0 / delta_c_lift)
                return dx, dy

        # --------------------------------------
        # Default: QDLDL inertia-corrected LDLᵀ
        # --------------------------------------
        # Modified for inertia correction (Algorithm IC)
        delta_w_last = getattr(self, "_delta_w_last", 0.0)  # persist across calls
        delta_w = 0.0
        delta_c = 0.0
        attempts = 0
        max_attempts = 10
        while attempts < max_attempts:
            # Assemble KKT with δ_w, δ_c
            if mE > 0:
                Wcsr = _csr(W + delta_w * sp.eye(n))
                JEcsr = G
                B22 = -delta_c * sp.eye(mE) if delta_c > 0 else sp.csr_matrix((mE, mE))
                K = sp.vstack(
                    [sp.hstack([Wcsr, JEcsr.T]), sp.hstack([JEcsr, B22])], format="csc"
                )
                rhs = np.concatenate([r1, r2])
            else:
                K = _csr(W + delta_w * sp.eye(n))
                rhs = r1

            K, _ = make_psd_advanced(K, self.reg, attempts)
            K_upper = _upper_csc(K)
            nsys = K_upper.shape[0]
            from .aux.amd import AMDReorderingArray

            amd_ = AMDReorderingArray(aggressive_absorption=True)
            perm, stats = amd_.compute_fill_reducing_permutation(K_upper)
            # perm = _ordering_perm(K_upper)

            try:
                fac = qd.factorize(
                    K_upper.indptr, K_upper.indices, K_upper.data, nsys, perm=perm
                )
                # inertia from D
                D = fac.D
                num_pos = int(np.sum(D > 0))
                num_neg = int(np.sum(D < 0))
                num_zero = int(np.sum(np.abs(D) < 1e-12))

                if num_pos == n and num_neg == mE and num_zero == 0:
                    self._delta_w_last = delta_w
                    x = qd.solve(fac, rhs)
                    if refine_iters > 0:
                        x = qd.solve_refine(fac, rhs, refine_iters)
                    return (x[:n], x[n:]) if mE > 0 else (x, np.zeros(0))

                # Adjust deltas (IC policy)
                if num_zero > 0:
                    delta_c = self.cfg.ip_delta_c_bar * (self.mu**self.cfg.ip_kappa_c)

                if delta_w_last == 0:
                    delta_w = (
                        self.cfg.ip_delta_w_0
                        if attempts == 0
                        else self.cfg.ip_kappa_w_plus_bar * delta_w
                    )
                else:
                    delta_w = (
                        max(
                            self.cfg.ip_delta_w_min,
                            self.cfg.ip_kappa_w_minus * delta_w_last,
                        )
                        if attempts == 0
                        else self.cfg.ip_kappa_w_plus * delta_w
                    )

                if delta_w > self.cfg.ip_delta_w_max:
                    raise ValueError("Inertia correction failed: too large delta_w")

                attempts += 1

            except Exception:
                # Fallback to SciPy if QDLDL fails
                sol = spla.spsolve(K.tocsc(), rhs)
                return (sol[:n], sol[n:]) if mE > 0 else (sol, np.zeros(0))

        return None, None  # failed

    def _compute_error(
        self, model: Model, x, lam, nu, zL, zU, mu: float = 0.0
    ) -> float:
        """
        IPOPT-style optimality error E_mu / E_0 with general simple bounds.
        Uses:
        stationarity:  g + JI^T λ + JE^T ν - zL + zU = 0
        equalities:    cE = 0
        inequalities:  cI ≤ 0  (reported via model.constraint_violation elsewhere)
        complement.:   (x-ℓ)∘zL = μ,  (u-x)∘zU = μ  on active bounds
        """
        data = model.eval_all(x, components=["f", "g", "cI", "JI", "cE", "JE"])
        g = np.asarray(data["g"], float)
        cI = np.asarray(data["cI"], float) if data["cI"] is not None else np.zeros(0)
        cE = np.asarray(data["cE"], float) if data["cE"] is not None else np.zeros(0)
        JI = data["JI"]
        JE = data["JE"]

        # bounds & masks
        lb, ub, hasL, hasU = _get_bounds(model, x)
        sL = np.where(hasL, x - lb, 1.0)  # slack for lower bounds
        sU = np.where(hasU, ub - x, 1.0)  # slack for upper bounds

        # stationarity with bound duals
        r_d = (
            g
            + (JI.T @ lam if JI is not None else 0)
            + (JE.T @ nu if JE is not None else 0)
            - zL
            + zU
        )

        # complementarity on active bounds
        r_comp_L = np.where(hasL, sL * zL - mu, 0.0)
        r_comp_U = np.where(hasU, sU * zU - mu, 0.0)

        # complementarity for inequality slacks
        r_comp_slacks = (
            self.s * lam - mu * np.ones_like(self.s) if self.mI > 0 else np.zeros(0)
        )

        # scaling (same spirit as your original)
        s_max = float(getattr(self.cfg, "ip_s_max", 100.0))
        denom = self.mI + self.mE + x.size
        sum_mults = (
            np.sum(np.abs(lam))
            + np.sum(np.abs(nu))
            + np.sum(np.abs(zL))
            + np.sum(np.abs(zU))
        )
        s_d = max(s_max, (sum_mults / max(1, denom))) / s_max
        s_c = (
            max(s_max, (np.sum(np.abs(zL)) + np.sum(np.abs(zU))) / max(1, x.size))
            / s_max
        )

        err = max(
            np.linalg.norm(r_d, np.inf) / s_d,
            np.linalg.norm(cE, np.inf) if self.mE > 0 else 0.0,
            np.linalg.norm(cI, np.inf) if self.mI > 0 else 0.0,
            max(np.linalg.norm(r_comp_L, np.inf), np.linalg.norm(r_comp_U, np.inf))
            / s_c,
            (
                (np.linalg.norm(r_comp_slacks, np.inf) / s_c)
                if r_comp_slacks.size > 0
                else 0.0
            ),
        )
        return err

    def _feasibility_restoration(
        self, model: Model, x: np.ndarray, mu: float, flt: Filter
    ):
        """
        Feasibility-only restoration:
        min  (ζ/2)||x - x_R||_D^2 + 1ᵀp + 1ᵀn
        s.t. [c_E(x); c_I(x)] - p + n = 0,   p≥0, n≥0

        Runs a short IP on the restoration model, escalating ζ until
        we either (i) drive θ down by a factor, or (ii) meet an absolute floor.
        """

        print("Entering feasibility restoration phase")

        tol = float(getattr(self.cfg, "tol", 1e-8))

        # Current feasibility on the ORIGINAL model
        theta_R = float(model.constraint_violation(x))
        if theta_R <= tol:
            print("Restoration not needed (already feasible enough).")
            return x.copy()

        # --------- build initial p/n from current constraints ----------
        dR = model.eval_all(x, components=["cE", "cI"])
        cE_R = _as_array_or_zeros(dR.get("cE", None), model.m_eq)
        cI_R = _as_array_or_zeros(dR.get("cI", None), model.m_ineq)
        c_R = np.concatenate([cE_R, cI_R]) if (cE_R.size or cI_R.size) else np.zeros(0)

        # initial p,n so that c - p + n = 0
        p0 = np.maximum(0.0, c_R)
        n0 = np.maximum(0.0, -c_R)

        # prox scaling around x_R (avoid overpenalizing tiny coords)
        x_R = x.copy()
        n = int(model.n)
        m = int(c_R.size)

        # D_R = diag(1 / max(1, |x|))
        D_R_vec = 1.0 / np.maximum(1.0, np.abs(x_R))
        D_R = _diag(
            D_R_vec
        )  # keep your helper; sparse or dense is fine as long as RestorationModel expects it

        # initial stacked var for restoration IP
        bar_x0 = np.concatenate([x_R, p0, n0]) if m > 0 else x_R.copy()

        # --------- targets & ζ schedule ----------
        theta_target_rel = float(self.cfg.ip_rest_theta_drop) * theta_R
        theta_target_abs = max(float(self.cfg.ip_rest_theta_abs), tol**0.75)
        theta_goal = max(theta_target_abs, theta_target_rel)

        zeta = float(self.cfg.ip_rest_zeta_init)
        zeta_M = float(self.cfg.ip_rest_zeta_max)
        zeta_u = float(self.cfg.ip_rest_zeta_update)

        best_x = x_R.copy()
        best_theta = theta_R
        improved = False

        for retry in range(int(self.cfg.ip_rest_max_retry) + 1):
            print(
                f"[Restoration] attempt {retry+1}/{self.cfg.ip_rest_max_retry+1} with ζ={zeta:.2e}"
            )

            # equality-only restoration model (reports f, g, cE, JE, H, etc. w.r.t. [x;p;n])
            resto_model = RestorationModel(model, x_R, D_R, zeta, rho=1.0)

            # dedicated filter bounded by current θ (optional)
            # resto_flt = Filter(self.cfg)
            # try:
            #     resto_flt.theta_max = 10.0 * theta_R
            # except Exception:
            #     pass

            # firm IP settings for restoration (disable SOC etc.)
            resto_cfg = copy.deepcopy(self.cfg)
            resto_cfg.ip_mu_init = float(self.cfg.ip_rest_mu)
            resto_cfg.ip_mu_min = max(1e-10, resto_cfg.ip_mu_min)
            resto_cfg.ip_pmax_soc = 0  # SOC off in restoration

            # short IP loop on the restoration model
            resto_stepper = InteriorPointStepper(
                resto_cfg,
                self.hess,
                self.reg,
                self.tr,
                None,
                self.funnel,
                self.ls,
                soc=None,
            )
            ip_state_resto = IPState.from_model(resto_model, bar_x0, resto_cfg)

            bar_x = bar_x0.copy()
            lamR = np.zeros(
                resto_model.m_eq, dtype=float
            )  # only equalities in restoration
            nuR = np.zeros(0, dtype=float)

            max_it = int(self.cfg.ip_rest_max_it)
            for k in range(max_it):
                bar_x, lamR, nuR, info = resto_stepper.step(
                    resto_model, bar_x, lamR, nuR, k, ip_state_resto
                )

                # map back to original x and measure feasibility on ORIGINAL model
                x_try = bar_x[:n]
                theta_try = float(model.constraint_violation(x_try))
                print(f"   iter {k:02d}: θ={theta_try:.3e}")

                # track best feasibility
                if theta_try + 1e-16 < best_theta:
                    best_theta = theta_try
                    best_x = x_try.copy()
                    improved = True

                # early exit if goal met within this attempt
                if theta_try <= theta_goal:
                    break

            # if goal already met, stop escalating ζ
            if best_theta <= theta_goal:
                break

            # otherwise, escalate ζ (up to max) and warm-start next attempt from best_x
            if zeta * zeta_u > zeta_M:
                print("[Restoration] ζ hit maximum; stopping escalation.")
                break
            zeta *= zeta_u

            # warm-start bar_x0 from best_x so far (recompute p,n for c(best_x))
            d_best = model.eval_all(best_x, components=["cE", "cI"])
            cE_best = _as_array_or_zeros(d_best.get("cE", None), model.m_eq)
            cI_best = _as_array_or_zeros(d_best.get("cI", None), model.m_ineq)
            c_best = (
                np.concatenate([cE_best, cI_best])
                if (cE_best.size or cI_best.size)
                else np.zeros(0)
            )
            p_best = np.maximum(0.0, c_best)
            n_best = np.maximum(0.0, -c_best)
            bar_x0 = (
                np.concatenate([best_x, p_best, n_best])
                if c_best.size > 0
                else best_x.copy()
            )

        # --------- outcome ----------
        if not improved:
            raise ValueError("Restoration failed to decrease constraint violation")

        # Update inequality slacks for the ORIGINAL problem so the next IP step is consistent:
        # s := max(1e-8, -c_I(best_x))
        d_new = model.eval_all(best_x, components=["cI"])
        cI_new = _as_array_or_zeros(d_new.get("cI", None), model.m_ineq)
        self.s = np.maximum(1e-8, -cI_new) if cI_new.size else np.zeros(0)

        print(f"[Restoration] success: θ {theta_R:.3e} → {best_theta:.3e}")
        return best_x

    # -------------- one iteration --------------
    def step(
        self,
        model: Model,
        x: np.ndarray,
        lam: np.ndarray,
        nu: np.ndarray,
        it: int,
        ip_state: Optional[IPState] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        One IP iteration with general simple bounds (ℓ ≤ x ≤ u).
        Uses duals zL, zU, diagonal Σ_x = zL/sL + zU/sU in the Newton system.
        """
        # --- initialize state ---
        st = (
            ip_state
            if (ip_state and ip_state.initialized)
            else IPState.from_model(model, x, self.cfg)
        )
        n = model.n
        mI, mE = st.mI, st.mE
        s = st.s.copy()
        lmb = st.lam.copy()
        nuv = st.nu.copy()
        zL = st.zL.copy()
        zU = st.zU.copy()
        mu = float(st.mu)
        tau = max(self.cfg.ip_tau_min, 1 - mu)

        # --- evaluate model ---
        data = model.eval_all(x, components=["f", "g", "cI", "JI", "cE", "JE"])
        f = float(data["f"])
        g = np.asarray(data["g"], float)
        cI = np.zeros(mI) if mI == 0 else np.asarray(data["cI"], float)
        cE = np.zeros(mE) if mE == 0 else np.asarray(data["cE"], float)
        JI = data["JI"] if mI > 0 else None
        JE = data["JE"] if mE > 0 else None
        theta = model.constraint_violation(x)

        self.mI = mI
        self.mE = mE
        self.s = s

        # --- bounds & slacks ---
        lb, ub, hasL, hasU = _get_bounds(model, x)
        sL = np.where(hasL, x - lb, 1.0)
        sU = np.where(hasU, ub - x, 1.0)

        # --- Hessian ---
        H = (
            model.lagrangian_hessian(x, lmb, nuv)
            if self.cfg.ip_exact_hessian
            else self.hess.get_hessian(model, x, lmb, nuv)
        )
        H, _ = make_psd_advanced(H, self.reg, it)

        # --- residuals (KKT) ---
        r_d = (
            g
            + (JI.T @ lmb if JI is not None else 0)
            + (JE.T @ nuv if JE is not None else 0)
            - zL
            + zU
        )
        r_pE = cE
        r_pI = cI + s if mI > 0 else np.zeros(0)
        r_comp_s = s * lmb - mu * np.ones(mI) if mI > 0 else np.zeros(0)

        err_0 = self._compute_error(model, x, lmb, nuv, zL, zU, 0.0)
        tol = float(getattr(self.cfg, "tol", 1e-8))
        if err_0 <= tol:
            info = self._pack_info(
                step_norm=0.0,
                accepted=True,
                converged=True,
                f=f,
                theta=theta,
                kkt={
                    "stat": np.linalg.norm(r_d, np.inf),
                    "ineq": (
                        np.linalg.norm(np.maximum(0, cI), np.inf) if mI > 0 else 0.0
                    ),
                    "eq": np.linalg.norm(cE, np.inf) if mE > 0 else 0.0,
                    "comp": 0.0,  # at E_0 we don't include μ-comps
                },
                alpha=0.0,
                rho=0.0,
                mu=mu,
            )
            return x, lmb, nuv, info

        # Solve barrier problem (11)
        # make safe if x is 0

        # --- diagonal Σ_x = zL/sL + zU/sU ---
        eps_abs = float(getattr(self.cfg, "sigma_eps_abs", 1e-12))
        Sigma_L = np.where(hasL, _safe_pos_div(zL, sL, eps=eps_abs), 0.0)
        Sigma_U = np.where(hasU, _safe_pos_div(zU, sU, eps=eps_abs), 0.0)
        # Keep your robust clamp via _safe_ratio_pos; here we just sum (already safe & nonnegative)
        Sigma_x_vec = Sigma_L + Sigma_U

        # slacks for inequalities
        Sigma_s_vec = _safe_pos_div(lmb, s, eps=eps_abs) if mI > 0 else np.zeros(0)

        # W = H + diag(Σ_x) + JI^T diag(Σ_s) JI
        W = H + _diag(Sigma_x_vec) + (JI.T @ _diag(Sigma_s_vec) @ JI if mI > 0 else 0)

        # Newton RHS for x-block
        rhs_x = -r_d

        # Solve KKT for (dx, dnu) with your solver path selection
        dx, dnu = self.solve_KKT(W, rhs_x, JE, r_pE)

        if dx is None:
            # Try feasibility restoration on original model
            try:
                x_new = self._feasibility_restoration(model, x, mu, self.filter)
                dx = x_new - x
            except ValueError as e:
                info = self._pack_info(
                    step_norm=0.0,
                    accepted=False,
                    converged=False,
                    f=f,
                    theta=theta,
                    kkt={
                        "stat": np.linalg.norm(r_d, np.inf),
                        "ineq": (
                            np.linalg.norm(np.maximum(0, cI), np.inf)
                            if mI > 0
                            else 0.0
                        ),
                        "eq": np.linalg.norm(cE, np.inf) if mE > 0 else 0.0,
                        "comp": 0.0,
                    },
                    alpha=0.0,
                    rho=0.0,
                    mu=mu,
                )
                info["error"] = str(e)
                return x, lmb, nuv, info

        # ----- Mehrotra predictor for adaptive μ -----
        # (run with μ=0 to get affine directions)
        if mI > 0 or np.any(hasL) or np.any(hasU):
            # reuse same W, rhs_x (they do not include μ explicitly)
            dx_aff, dnu_aff = self.solve_KKT(W, -r_d, JE, r_pE)

            # inequality slacks: ds_aff = -r_pI - JI dx_aff
            ds_aff = (
                -r_pI - (JI @ dx_aff if (mI > 0 and JI is not None) else 0)
                if mI > 0 else np.zeros(0)
            )

            # affine dλ from: diag(λ) ds + diag(s) dλ = - s∘λ   (μ=0)
            if mI > 0:
                dlam_aff = (-s * lmb - lmb * ds_aff) / np.maximum(s, 1e-16)
            else:
                dlam_aff = np.zeros(0)

            # bounds (lower): zL*dx + sL*dzL = - sL*zL   (μ=0, sL = x-lb)
            dzL_aff = np.where(
                hasL, (-sL * zL - zL * dx_aff) / np.maximum(sL, 1e-16), 0.0
            )

            # bounds (upper): -zU*dx + sU*dzU = - sU*zU   (μ=0, sU = ub-x)
            dzU_aff = np.where(
                hasU, (-sU * zU + zU * dx_aff) / np.maximum(sU, 1e-16), 0.0
            )

            # fraction-to-boundary for the affine step to keep all > 0
            def _ftb_pos(z, dz):
                if z.size == 0: return 1.0
                neg = dz < 0
                if not np.any(neg): return 1.0
                return float(min(1.0, self.cfg.ip_tau * np.min(-z[neg] / dz[neg])))

            alpha_aff = 1.0
            if mI > 0:
                alpha_aff = min(alpha_aff, _ftb_pos(s,    ds_aff))
                alpha_aff = min(alpha_aff, _ftb_pos(lmb,  dlam_aff))
            if np.any(hasL):
                alpha_aff = min(alpha_aff, _ftb_pos(zL[hasL], dzL_aff[hasL]))
                alpha_aff = min(alpha_aff, _ftb_pos(sL[hasL], dx_aff[hasL]))   # sL = x-lb
            if np.any(hasU):
                alpha_aff = min(alpha_aff, _ftb_pos(zU[hasU], dzU_aff[hasU]))
                alpha_aff = min(alpha_aff, _ftb_pos(sU[hasU], -dx_aff[hasU]))  # sU = ub-x

            # μ_aff from all complementarity pairs (ineq + active bounds)
            mu_min = float(self.cfg.ip_mu_min)
            parts = []
            if mI > 0:
                s_aff   = s   + alpha_aff * ds_aff
                lam_aff = lmb + alpha_aff * dlam_aff
                parts.append(np.dot(s_aff, lam_aff))
            if np.any(hasL):
                sL_aff = sL + alpha_aff * dx_aff
                zL_aff = zL + alpha_aff * dzL_aff
                parts.append(np.dot(sL_aff[hasL], zL_aff[hasL]))
            if np.any(hasU):
                sU_aff = sU - alpha_aff * dx_aff
                zU_aff = zU + alpha_aff * dzU_aff
                parts.append(np.dot(sU_aff[hasU], zU_aff[hasU]))

            denom = (mI + int(np.sum(hasL)) + int(np.sum(hasU)))
            mu_aff = max(mu_min, (sum(parts) / max(1, denom)) if parts else mu)

            # σ = (μ_aff / μ)^p  (clipped to [0,1])
            pwr = float(getattr(self.cfg, "ip_sigma_power", 3.0))
            sigma = float(np.clip((mu_aff / max(mu, mu_min)) ** pwr, 0.0, 1.0))

            # final μ for the main step
            mu = max(mu_min, sigma * mu_aff)


        # recover (ds, dλ) for inequalities (reference form)
        ds = (
            -r_pI - (JI @ dx if (mI > 0 and JI is not None) else 0)
            if mI > 0
            else np.zeros(0)
        )
        dlam = (
            (-(lmb - mu / s) - _safe_pos_div(mu, s**2, eps=eps_abs) * ds)
            if mI > 0
            else np.zeros(0)
        )

        # Newton steps for bound duals via complementarity:
        #   sL*zL = μ with sL = x - lb        => zL*dx + sL*dzL = μ - sL*zL
        #   sU*zU = μ with sU = ub - x (dsU=-dx) => -zU*dx + sU*dzU = μ - sU*zU
        dzL = np.where(
            hasL, _safe_pos_div(mu - sL * zL - zL * dx, sL, eps=eps_abs), 0.0
        )
        dzU = np.where(
            hasU, _safe_pos_div(mu - sU * zU + zU * dx, sU, eps=eps_abs), 0.0
        )

        # Higher-order correction (2nd-order Taylor expansion)
        # Quadratic correction for x: accounts for 2nd-order terms in complementarity
        rhs_x2 = - (JI.T @ (Sigma_s_vec * (JI @ dx))) if mI > 0 else np.zeros(n)
        dx2, dnu2 = self.solve_KKT(W, rhs_x2, JE, np.zeros(mE))  # Reuse W, JE
        dx += 0.5 * dx2  # Blend second-order correction
        # Update ds, dlam for inequalities
        if mI > 0:
            ds2 = - (JI @ dx2) if JI is not None else np.zeros(mI)
            ds += 0.5 * ds2
            dlam2 = - _safe_pos_div(mu, s**2, eps=eps_abs) * ds2
            dlam += 0.5 * dlam2
        # Update dzL, dzU for bounds
        dzL2 = np.where(
            hasL, _safe_pos_div(- zL * dx2, sL, eps=eps_abs), 0.0
        )
        dzU2 = np.where(
            hasU, _safe_pos_div(zU * dx2, sU, eps=eps_abs), 0.0
        )
        dzL += 0.5 * dzL2
        dzU += 0.5 * dzU2
            
        # fraction-to-boundary limits
        def _ftb(v, dv):
            return self._max_step_ftb(v, dv, tau) if v.size else 1.0

        alpha_L = _ftb(sL[hasL], dx[hasL]) if np.any(hasL) else 1.0  # keep sL+α dx > 0
        alpha_U = _ftb(sU[hasU], -dx[hasU]) if np.any(hasU) else 1.0  # keep sU-α dx > 0
        alpha_zL = _ftb(zL[hasL], dzL[hasL]) if np.any(hasL) else 1.0
        alpha_zU = _ftb(zU[hasU], dzU[hasU]) if np.any(hasU) else 1.0
        alpha_max_x = min(alpha_L, alpha_U)
        alpha_max_s = _ftb(s, ds) if mI > 0 else 1.0
        alpha_max_z = min(alpha_zL, alpha_zU)
        alpha_max = min(alpha_max_x, alpha_max_s)

        # barrier merit pieces (if you need them for your LS)
        # phi_0 etc. kept as in your previous code; we defer to your external LineSearcher
        try:
            alpha, ls_iters, needs_restoration = self.ls.search_ip(
                model=model,
                x=x,
                dx=dx,
                ds=ds,
                s=s,
                mu=mu,
                d_phi=float(g @ dx)
                - float(np.sum(mu / np.maximum(x, 1e-16) * dx)),  # lightweight proxy
                theta0=theta,
                alpha_max=alpha_max,
            )
        except Exception as e:
            # fallback
            alpha = min(1.0, alpha_max)
            ls_iters = 0
            needs_restoration = False

        # optional SOC block
        if ls_iters == 0 and alpha == 0.0 and needs_restoration:
            for soc_iter in range(self.cfg.ip_pmax_soc):
                cE_soc = (
                    cE + (JE @ dx if JE is not None else 0) if mE > 0 else np.zeros(0)
                )
                cI_soc = (
                    cI + (JI @ dx if JI is not None else 0) if mI > 0 else np.zeros(0)
                )
                rhs_soc = (
                    -np.concatenate([cE_soc, cI_soc]) if mE + mI > 0 else np.zeros(0)
                )
                print("  SOC attempt", soc_iter, "violation", np.linalg.norm(rhs_soc))
                dx_soc, _ = self.solve_KKT(
                    W, rhs_soc[:n], JE, rhs_soc[n:] if mE > 0 else None
                )
                alpha_soc = self._max_step_ftb(x, dx + dx_soc, tau)
                x_soc = x + alpha_soc * (dx + dx_soc)
                theta_soc = model.constraint_violation(x_soc)
                if theta_soc < theta:
                    dx = dx + dx_soc  # Update direction
                    alpha, ls_iters, needs_restoration = self.ls.search_ip(
                        model=model,
                        x=x,
                        dx=dx,
                        ds=ds,
                        s=s,
                        mu=mu,
                        theta0=theta,
                        alpha_max=alpha_soc,
                    )
                    break
                if soc_iter == self.cfg.ip_pmax_soc - 1:
                    break

        if alpha == 0.0 and needs_restoration:
            try:
                x_new = self._feasibility_restoration(model, x, mu, self.filter)
                data_new = model.eval_all(x_new, ["g", "cI", "JI", "cE", "JE"])
                # rough multiplier resets (same style as yours)
                lmb_new = np.maximum(1e-8, lmb) if mI > 0 else lmb
                nu_new = np.zeros(mE) if mE > 0 else nuv
                # keep duals positive and bounded
                zL_new = np.where(hasL, np.maximum(1e-8, zL), 0.0)
                zU_new = np.where(hasU, np.maximum(1e-8, zU), 0.0)
                info = self._pack_info(
                    step_norm=np.linalg.norm(x_new - x),
                    accepted=True,
                    converged=False,
                    f=model.eval_all(x_new, ["f"])["f"],
                    theta=model.constraint_violation(x_new),
                    kkt={},
                    alpha=0.0,
                    rho=0.0,
                    mu=mu,
                    ls_iters=ls_iters,
                )
                st.s, st.lam, st.nu, st.zL, st.zU = s, lmb_new, nu_new, zL_new, zU_new
                return x_new, lmb_new, nu_new, info
            except ValueError as e:
                info = self._pack_info(
                    step_norm=0.0,
                    accepted=False,
                    converged=False,
                    f=f,
                    theta=theta,
                    kkt={
                        "stat": np.linalg.norm(r_d, np.inf),
                        "ineq": (
                            np.linalg.norm(np.maximum(0, cI), np.inf)
                            if mI > 0
                            else 0.0
                        ),
                        "eq": np.linalg.norm(cE, np.inf) if mE > 0 else 0.0,
                        "comp": 0.0,
                    },
                    alpha=0.0,
                    rho=0.0,
                    mu=mu,
                    ls_iters=ls_iters,
                )
                info["error"] = str(e)
                return x, lmb, nuv, info

        # --- accept step ---
        alpha_bz = min(alpha, alpha_max_z)
        x_new = x + alpha * dx
        s_new = s + alpha * ds if mI > 0 else s
        lmb_new = lmb + alpha * dlam if mI > 0 else lmb
        nu_new = nuv + alpha * dnu if mE > 0 else nuv
        zL_new = zL + alpha_bz * dzL
        zU_new = zU + alpha_bz * dzU

        # σ-neighborhood correction for bound duals
        ksig = float(self.cfg.ip_kappa_sigma)
        sL_new = np.where(hasL, x_new - lb, 1.0)
        sU_new = np.where(hasU, ub - x_new, 1.0)
        if np.any(hasL):
            sL_clip = np.maximum(sL_new, 1e-16)
            zL_new = np.maximum(
                mu / (ksig * sL_clip), np.minimum(zL_new, ksig * mu / sL_clip)
            )
        if np.any(hasU):
            sU_clip = np.maximum(sU_new, 1e-16)
            zU_new = np.maximum(
                mu / (ksig * sU_clip), np.minimum(zU_new, ksig * mu / sU_clip)
            )

        # --- recompute basic measures for reporting ---
        # --- recompute basic measures for reporting (AT x_new) ---
        data_new = model.eval_all(x_new, ["f", "g", "cI", "JI", "cE", "JE"])
        f_new  = float(data_new["f"])
        g_new  = np.asarray(data_new["g"], float)
        cI_new = np.asarray(data_new["cI"], float) if mI > 0 else np.zeros(0)
        cE_new = np.asarray(data_new["cE"], float) if mE > 0 else np.zeros(0)
        JI_new = data_new["JI"] if mI > 0 else None
        JE_new = data_new["JE"] if mE > 0 else None
        theta_new = model.constraint_violation(x_new)

        # stationarity at x_new
        r_d_new = (
            g_new
            + (JI_new.T @ lmb_new if (mI > 0 and JI_new is not None) else 0)
            + (JE_new.T @ nu_new  if (mE > 0 and JE_new is not None) else 0)
            - zL_new + zU_new
        )

        # complementarity at x_new
        r_comp_L_new = np.where(hasL, (x_new - lb) * zL_new - mu, 0.0) if np.any(hasL) else 0.0
        r_comp_U_new = np.where(hasU, (ub - x_new) * zU_new - mu, 0.0) if np.any(hasU) else 0.0
        r_comp_s_new = (s_new * lmb_new - mu) if mI > 0 else np.zeros(0)

        kkt_new = {
            "stat": np.linalg.norm(r_d_new, np.inf),
            "ineq": np.linalg.norm(np.maximum(0, cI_new), np.inf) if mI > 0 else 0.0,
            "eq":   np.linalg.norm(cE_new, np.inf) if mE > 0 else 0.0,
            "comp": max(
                (np.linalg.norm(r_comp_L_new, np.inf) if np.any(hasL) else 0.0),
                (np.linalg.norm(r_comp_U_new, np.inf) if np.any(hasU) else 0.0),
            ) + (np.linalg.norm(r_comp_s_new, np.inf) if r_comp_s_new.size > 0 else 0.0),
        }

        converged = (
            kkt_new["stat"] <= tol
            and kkt_new["ineq"] <= tol
            and kkt_new["eq"] <= tol
            and kkt_new["comp"] <= tol
            and mu <= tol / 10
        )

        info = self._pack_info(
            step_norm=np.linalg.norm(x_new - x),
            accepted=True,
            converged=converged,
            f=f_new,
            theta=theta_new,
            kkt=kkt_new,
            alpha=alpha,
            rho=0.0,
            mu=mu,
        )

        # persist state
        st.s, st.lam, st.nu, st.zL, st.zU, st.mu = (
            s_new,
            lmb_new,
            nu_new,
            zL_new,
            zU_new,
            mu,
        )
        return x_new, lmb_new, nu_new, info

    # -------------- helpers --------------
    @staticmethod
    def _max_step_ftb(z: np.ndarray, dz: np.ndarray, tau: float) -> float:
        if z.size == 0 or dz.size == 0:
            return 1.0
        neg = dz < 0
        if not np.any(neg):
            return 1.0
        return float(min(1.0, tau * np.min(-z[neg] / dz[neg])))

    @staticmethod
    def _barrier_merit(
        f: float, mu: float, s: np.ndarray, rho: float, cE: np.ndarray, cI: np.ndarray
    ) -> float:
        term_bar = -mu * (np.sum(np.log(s)) if s.size else 0.0)
        term_pen = rho * (
            (np.sum(np.abs(cE)) if cE.size else 0.0)
            + (np.sum(np.abs(cI + s)) if (cI.size or s.size) else 0.0)
        )
        return f + term_bar + term_pen

    def _barrier_merit_and_dir(self, f, mu, s, dx, ds, g, rho, cE, cI):
        phi0 = self._barrier_merit(f, mu, s, rho, cE, cI)
        # Reference: dphi = g·dx + rho*|cE| (ineq part not linearized)
        dphi = float(g @ dx) + (rho * float(np.sum(np.abs(cE))) if cE.size else 0.0)
        return phi0, dphi

    def _pack_info(
        self,
        step_norm: float,
        accepted: bool,
        converged: bool,
        f: float,
        theta: float,
        kkt: Dict[str, float],
        alpha: float,
        rho: float,
        mu: float,
    ) -> Dict:
        return {
            "mode": "ip",
            "step_norm": step_norm,
            "accepted": accepted,
            "converged": converged,
            "f": f,
            "theta": theta,
            "stat": kkt["stat"],
            "ineq": kkt["ineq"],
            "eq": kkt["eq"],
            "comp": kkt["comp"],
            "ls_iters": 0,
            "alpha": alpha,
            "rho": rho,
            "tr_radius": (self.tr.radius if self.tr else 0.0),
            "mu": mu,
        }
