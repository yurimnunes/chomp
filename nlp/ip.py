# sqp_ip.py (refactor/cleanup — drop-in)
from __future__ import annotations

import copy
import time
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from kkt_core_cpp import KKTReusable, solve_kkt_csr

# External project modules (unchanged API)
from nlp.blocks.linesearch import LineSearcher

from .blocks.aux import HessianManager, Model, SQPConfig
from .blocks.filter import Filter, Funnel
from .blocks.reg import Regularizer, make_psd_advanced
from .ip_aux import *
from .ip_cg import *

# imports at top of sqp_ip.py
from .ip_kernels import (
    _div,
    _safe_pos_div,
    k_alpha_fraction_to_boundary,
    k_build_sigmas,
    k_cap_bound_duals_sigma_box,
    k_compute_complementarity,
    k_dz_bounds_from_dx,
    k_max_step_ftb,
)
from .ip_kkt import *
from .ip_kkt import _csr

EPS_DIV = 1e-16


def _diag(v: np.ndarray):
    v = np.asarray(v).ravel()
    return sp.diags(v)


def _as_array_or_zeros(v, m: int) -> np.ndarray:
    if m <= 0:
        return np.zeros(0, float)
    if v is None:
        return np.zeros(m, float)
    a = np.asarray(v, float).ravel()
    return a if a.size == m else a.reshape(m)


# ------------------ defaults ------------------
_CFG_DEFAULTS: Dict[str, object] = dict(
    ip_exact_hessian=True,
    ip_hess_reg0=1e-4,
    ip_eq_reg=1e-4,
    # shifted barrier
    ip_use_shifted_barrier=False,
    ip_shift_tau=0.1,
    ip_shift_bounds=0.1,
    ip_shift_adaptive=True,
    # barrier & PC
    ip_mu_init=1e-2,
    ip_mu_min=1e-12,
    ip_sigma_power=3.0,
    # fraction-to-boundary
    ip_tau_pri=0.995,
    ip_tau_dual=0.99,
    ip_tau=0.995,  # legacy single tau kept
    ip_alpha_max=1.0,
    # LS bridge
    ip_ls_max=30,
    ip_alpha_min=1e-10,
    ip_alpha_backtrack=0.5,
    ip_armijo_coeff=1e-4,
    # μ schedule (legacy knobs kept)
    ip_mu_reduce_every=10,
    ip_mu_reduce=0.2,
    # control
    ip_rho_init=1.0,
    ip_rho_inc=10.0,
    ip_rho_max=1e6,
    # Σ safety
    sigma_eps_abs=1e-8,
    sigma_cap=1e8,
    # restoration
    ip_rest_max_it=8,
    ip_rest_max_retry=5,
    ip_rest_mu=1.0,
    ip_rest_zeta_init=1.0,
    ip_rest_zeta_update=5.0,
    ip_rest_zeta_max=1e8,
    ip_rest_theta_drop=0.5,
    ip_rest_theta_abs=1e-6,
    # KKT path
    ip_kkt_method="hykkt",  # "lifted"|"hykkt"|"qdldl"
    # small trust region on dx
    ip_dx_max=1e3,
    # conservative μ when infeasible
    ip_theta_clip=1e-2,
    ip_enable_higher_order=False,
    ip_ho_blend=0.3,
    ip_ho_rel_cap=0.5,
    # Gondzio MC
    ip_gondzio_mc=False,
    ip_gondzio_iters=3,
    ip_gondzio_blend=0.4,
    ip_mc_alpha_thresh=0.8,
    ip_mc_dispersion=8.0,
)


# ------------------ small data helpers ------------------
def _get_bounds(model: Model, x: np.ndarray):
    n = x.size
    lb = getattr(model, "lb", None)
    ub = getattr(model, "ub", None)
    lb = np.full(n, -np.inf) if lb is None else np.asarray(lb, float).ravel()
    ub = np.full(n, +np.inf) if ub is None else np.asarray(ub, float).ravel()
    hasL = np.isfinite(lb)
    hasU = np.isfinite(ub)
    return lb, ub, hasL, hasU


def _compute_kkt_matrix(W, JI, Sigma_s_vec, mI):
    """Helper to compute W + JI^T * Sigma_s * JI for comparison and storage."""
    if sp.isspmatrix_csr(W) or sp.isspmatrix_csc(W):
        W = W.tocsr(copy=True)
    else:
        W = sp.csr_matrix(W)
    if mI > 0 and JI is not None and Sigma_s_vec.size:
        JIw = JI * Sigma_s_vec[:, None]
        W = W + (JI.T @ JIw)
    return W


def _matrix_difference_norm(A, B):
    """Compute Frobenius norm of difference between two sparse matrices."""
    if A is None or B is None:
        return float("inf")
    diff = A - B
    return sp.linalg.norm(diff, "fro")


def _csr_buffers(A: sp.spmatrix):
    A = A.tocsr()
    return (
        A.indptr.astype(np.int32),
        A.indices.astype(np.int32),
        A.data.astype(np.float64),
    )


# ------------------ stepper ------------------
class InteriorPointStepper:
    """
    Slacks-only barrier IP stepper with bounds and optional shifted barrier.
    Uses Mehrotra predictor-corrector + centrality; HYKKT (default) with fallback.
    """

    # ---------- init & cfg ----------
    def __init__(
        self,
        cfg: SQPConfig,
        hess: HessianManager,
        regularizer: Regularizer,
        tr=None,
        flt: Optional[Filter] = None,
        funnel: Optional[Funnel] = None,
        ls: Optional[LineSearcher] = None,
        soc=None,
    ):
        self.cfg = cfg
        self.hess = hess
        self.reg = regularizer
        self.tr = tr
        self.filter = flt
        self.funnel = funnel
        self.ls = ls
        self.soc = soc
        self._load_defaults()
        self._kkt_cache = KKTCache()

    def _load_defaults(self):
        for k, v in _CFG_DEFAULTS.items():
            if not hasattr(self.cfg, k):
                setattr(self.cfg, k, v)
        # mirror to LS bridge if unset
        self.cfg.ls_backtrack = getattr(
            self.cfg, "ls_backtrack", float(self.cfg.ip_alpha_backtrack)
        )
        self.cfg.ls_armijo_f = getattr(
            self.cfg, "ls_armijo_f", float(self.cfg.ip_armijo_coeff)
        )
        self.cfg.ls_max_iter = getattr(self.cfg, "ls_max_iter", int(self.cfg.ip_ls_max))
        self.cfg.ls_min_alpha = getattr(
            self.cfg, "ls_min_alpha", float(self.cfg.ip_alpha_min)
        )

    def _adaptive_shift_slack(self, s: np.ndarray, cI: np.ndarray, it: int) -> float:
        if s.size == 0:
            return 0.0
        base = float(getattr(self.cfg, "ip_shift_tau", 1e-3))
        min_s = np.min(s)
        max_v = np.max(np.abs(cI)) if cI.size else 0.0
        if min_s < 1e-6 or max_v > 1e2:
            return min(1.0, base * (1.0 + 0.1 * it))
        if min_s > 1e-2 and max_v < 1e-2:
            return max(0.0, base * (1.0 - 0.05 * it))
        return base

    def _adaptive_shift_bounds(self, x, lb, ub, hasL, hasU, it: int) -> float:
        if not (np.any(hasL) or np.any(hasU)):
            return 0.0
        sL = np.where(hasL, x - lb, np.inf)
        sU = np.where(hasU, ub - x, np.inf)
        m = min(
            np.min(sL[hasL]) if np.any(hasL) else np.inf,
            np.min(sU[hasU]) if np.any(hasU) else np.inf,
        )
        b0 = float(getattr(self.cfg, "ip_shift_bounds", 0.0))
        if m < 1e-8:
            return min(1.0, max(b0, 1e-3) * (1 + 0.05 * it))
        if m > 1e-2:
            return max(0.0, b0 * 0.9)
        return b0

    def _safe_reinit_slacks_duals(
        self, cI, JI, x, lb, ub, hasL, hasU, mu: float, tau_shift: float
    ):
        n = x.size
        mI = int(cI.size)
        if mI > 0:
            s_floor = np.maximum(1e-8, np.minimum(1.0, np.abs(cI)))
            s0 = np.maximum(s_floor, -cI + 1e-3)
            lam0 = np.maximum(1e-8, mu / np.maximum(s0 + tau_shift, 1e-12))
        else:
            s0 = np.zeros(0)
            lam0 = np.zeros(0)
        zL0 = np.zeros(n)
        zU0 = np.zeros(n)
        sL = np.where(hasL, np.maximum(1e-12, x - lb), 1.0)
        sU = np.where(hasU, np.maximum(1e-12, ub - x), 1.0)
        bshift = float(getattr(self.cfg, "ip_shift_bounds", 0.0))
        if np.any(hasL):
            zL0[hasL] = np.maximum(1e-8, mu / np.maximum(sL[hasL] + bshift, 1e-12))
        if np.any(hasU):
            zU0[hasU] = np.maximum(1e-8, mu / np.maximum(sU[hasU] + bshift, 1e-12))
        return s0, lam0, zL0, zU0

    def _build_sigmas(
        self,
        *,
        zL,
        zU,
        sL,
        sU,
        hasL,
        hasU,
        lmb,
        s,
        cI,
        tau_shift,
        bound_shift,
        use_shifted,
    ):
        eps_abs = float(getattr(self.cfg, "sigma_eps_abs", 1e-8))
        cap_val = float(getattr(self.cfg, "sigma_cap", 1e8))
        return k_build_sigmas(
            zL,
            zU,
            sL,
            sU,
            hasL,
            hasU,
            lmb,
            s,
            cI,
            tau_shift,
            bound_shift,
            use_shifted,
            eps_abs,
            cap_val,
        )

    def _dz_bounds_from_dx(
        self,
        *,
        dx,
        zL,
        zU,
        sL,
        sU,
        hasL,
        hasU,
        bound_shift,
        use_shifted,
        mu: float | None,
    ):
        use_mu = 0 if mu is None else 1
        dzL, dzU = k_dz_bounds_from_dx(
            dx,
            zL,
            zU,
            sL,
            sU,
            hasL,
            hasU,
            bound_shift,
            use_shifted,
            0.0 if mu is None else mu,
            use_mu,
        )
        return dzL, dzU

    def _cap_bound_duals_sigma_box(
        self,
        *,
        zL_new,
        zU_new,
        sL_new,
        sU_new,
        hasL,
        hasU,
        use_shifted,
        bound_shift,
        mu,
        ksig: float = 1e10,
    ):
        k_cap_bound_duals_sigma_box(
            zL_new,
            zU_new,
            sL_new,
            sU_new,
            hasL,
            hasU,
            use_shifted,
            bound_shift,
            mu,
            float(ksig),
        )
        return zL_new, zU_new

    # ---------- public KKT (unchanged signature/semantics) ----------
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
        method: str = "hykkt",  # "hykkt"|"lifted"|"qdldl"|"lldl" -> we map to "hykkt" or "ldl"
        gamma: float | None = None,
        delta_c_lift: float | None = None,
        cg_tol: float = 1e-6,
        cg_maxit: int = 200,
        return_reusable: bool = False,
        Sigma_s_vec=None,
        JI=None,
        mI: int = 0,
        delta: float = 0.0,  # added: feed regularization on W if you want
        assemble_schur_if_m_small: bool = True,
        jacobi_schur_prec: bool = True,
    ) -> (
        Tuple[np.ndarray, np.ndarray]
        | Tuple[np.ndarray, np.ndarray, Optional[KKTReusable]]
    ):
        n = Wmat.shape[0]
        mE = 0 if JE_mat is None else JE_mat.shape[0]
        W = Wmat.tocsr() if sp.isspmatrix(Wmat) else sp.csr_matrix(Wmat)
        G = (
            None
            if JE_mat is None
            else (JE_mat.tocsr() if sp.isspmatrix(JE_mat) else sp.csr_matrix(JE_mat))
        )
        r1 = np.asarray(rhs_x, dtype=float).reshape(n)
        r2 = None if mE == 0 else -np.asarray(rpE, dtype=float).reshape(mE)

        cache = self._kkt_cache

        # Build current KKT (if you have additional inequality blocks, compute them here)
        current_kkt_matrix = _compute_kkt_matrix(W, JI, Sigma_s_vec, mI)

        # Potential reuse logic (only meaningful for direct factorizations, but we keep it)
        reuse_factorization = False
        if cache.prev_kkt_matrix is not None and method in ["qdldl", "lldl"]:
            matrix_diff = _matrix_difference_norm(
                current_kkt_matrix, cache.prev_kkt_matrix
            )
            if matrix_diff < cache.matrix_change_tol:
                reuse_factorization = True

        # Map method to our two backends
        method_cpp = "hykkt" if method == "hykkt" else "ldl"
        # check if there are equality constraints
        if mE == 0 and method_cpp == "hykkt":
            method_cpp = "ldl"  # no equality constraints, HYKKT not needed

        # Prepare CSR buffers
        print("Preparing CSR buffers for KKT solve...")
        print(W)
        print("r1", r1)
        print("r2", r2)
        print(G)
        W_ip, W_ix, W_data = _csr_buffers(W)
        if mE > 0 and G is not None:
            G_ip, G_ix, G_data = _csr_buffers(G)
        else:
            G_ip = G_ix = G_data = None

        # Call C++ binding
        dx, dy, reusable = solve_kkt_csr(
            W_ip,
            W_ix,
            W_data,
            n,
            G_ip if mE > 0 else None,
            G_ix if mE > 0 else None,
            G_data if mE > 0 else None,
            mE,
            r1,
            None if r2 is None else r2,
            method_cpp,
            float(delta),
            None if gamma is None else float(gamma),
            bool(assemble_schur_if_m_small),
            bool(jacobi_schur_prec),
            float(cg_tol),
            int(cg_maxit),
        )

        # Cache for next call (only keep factorization for direct method)
        cache.prev_kkt_matrix = current_kkt_matrix
        cache.prev_factorization = reusable if method_cpp == "ldl" else None
        self._delta_w_last = cache.delta_w_last
        self._prev_step = (dx, dy)

        return (dx, dy, reusable) if return_reusable else (dx, dy)

    # ---------- error & μ update ----------
    def _compute_error(
        self, model: Model, x, lam, nu, zL, zU, mu: float = 0.0, *, s=None, mI=None
    ) -> float:
        data = model.eval_all(x, components=["f", "g", "cI", "JI", "cE", "JE"])
        g = np.asarray(data["g"], float)

        # Use model.m_ineq / model.m_eq for sizes to avoid None.size errors
        mI_decl = int(getattr(model, "m_ineq", 0))
        mE_decl = int(getattr(model, "m_eq", 0))

        cI = _as_array_or_zeros(data.get("cI", None), mI_decl)
        cE = _as_array_or_zeros(data.get("cE", None), mE_decl)

        JI, JE = data["JI"], data["JE"]

        lb, ub, hasL, hasU = _get_bounds(model, x)
        sL = np.where(hasL, x - lb, 1.0)
        sU = np.where(hasU, ub - x, 1.0)

        r_d = (
            g
            + (JI.T @ lam if JI is not None else 0)
            + (JE.T @ nu if JE is not None else 0)
            - zL
            + zU
        )

        use_shifted = bool(getattr(self.cfg, "ip_use_shifted_barrier", False))
        tau_shift = float(getattr(self, "tau_shift", 0.0)) if use_shifted else 0.0
        bshift = (
            float(getattr(self.cfg, "ip_shift_bounds", 0.0)) if use_shifted else 0.0
        )

        if mI is None:
            mI = int(getattr(self, "mI", 0))
        if s is None:
            s = getattr(self, "s", np.zeros(0))

        if use_shifted:
            r_comp_L = np.where(hasL, (sL + bshift) * zL - mu, 0.0)
            r_comp_U = np.where(hasU, (sU + bshift) * zU - mu, 0.0)
            r_comp_s = ((s + tau_shift) * lam - mu) if mI > 0 else np.zeros(0)
        else:
            r_comp_L = np.where(hasL, sL * zL - mu, 0.0)
            r_comp_U = np.where(hasU, sU * zU - mu, 0.0)
            r_comp_s = (s * lam - mu) if mI > 0 else np.zeros(0)

        s_max = float(getattr(self.cfg, "ip_s_max", 100.0))
        denom = mI + mE_decl + x.size
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

        return float(
            max(
                np.linalg.norm(r_d, np.inf) / s_d,
                np.linalg.norm(cE, np.inf) if mE_decl > 0 else 0.0,
                np.linalg.norm(cI, np.inf) if mI > 0 else 0.0,
                max(np.linalg.norm(r_comp_L, np.inf), np.linalg.norm(r_comp_U, np.inf))
                / s_c,
                (np.linalg.norm(r_comp_s, np.inf) / s_c) if r_comp_s.size else 0.0,
            )
        )

    def compute_complementarity(self, s, lam, mu, tau_shift, use_shifted):
        return float(
            k_compute_complementarity(s, lam, mu, tau_shift, 1 if use_shifted else 0)
        )

    def update_mu(
        self,
        mu: float,
        s: np.ndarray,
        lam: np.ndarray,
        theta: float,
        kkt: Dict,
        accepted: bool,
        cond_H: float,
        sigma: float,
        mu_aff: float,
        use_shifted: bool,
        tau_shift: float,
    ) -> float:
        mu_min = float(getattr(self.cfg, "ip_mu_min", 1e-12))
        kappa = float(getattr(self.cfg, "kappa_mu", 1.5))
        theta_tol = float(getattr(self.cfg, "tol_feas", 1e-6))
        comp_tol = float(getattr(self.cfg, "tol_comp", 1e-6))
        cond_max = float(getattr(self.cfg, "cond_threshold", 1e6))
        comp = self.compute_complementarity(s, lam, mu, tau_shift, use_shifted)
        good = (
            accepted
            and theta <= theta_tol
            and comp <= comp_tol
            and kkt["stat"] <= float(getattr(self.cfg, "tol_stat", 1e-6))
            and (np.isnan(cond_H) or cond_H <= cond_max)
        )

        # Adaptive mu reduction based on complementarity progress
        comp_ratio = comp / max(mu, 1e-12)
        mu_base = max(mu_min, sigma * mu_aff)
        if good and comp_ratio < 0.1:  # Superlinear reduction near convergence
            mu_new = mu_base * min(0.1, (mu_aff / mu) ** 2)
        elif (
            comp_ratio > 10.0 or theta > 10 * theta_tol
        ):  # Conservative when far from feasibility
            mu_new = min(10.0 * mu, mu_base * 1.2)
        else:
            mu_new = min(mu_base, mu**kappa)

        return max(mu_new, mu_min)

    # ---------- Gondzio MC (unchanged behavior; compact form) ----------
    def _gondzio_multicorrector(
        self,
        *,
        x,
        s,
        lmb,
        zL,
        zU,
        lb,
        ub,
        hasL,
        hasU,
        JI,
        JE,
        W,
        Sigma_s_vec,
        r_pE,
        reusable,
        dx,
        dnu,
        ds,
        dlam,
        dzL,
        dzU,
        mu,
        mu_aff,
        sigma,
        alpha_aff,
        tau_shift,
        bound_shift,
        use_shifted,
    ):
        mI = int(s.size)
        alpha_thresh = float(getattr(self.cfg, "ip_mc_alpha_thresh", 0.8))

        comp_vec = []
        if mI > 0:
            comp_vec.append(
                (s + (tau_shift if use_shifted else 0.0)) * np.maximum(lmb, 1e-16)
            )
        if np.any(hasL):
            comp_vec.append(
                (
                    (x - lb)
                    + (bound_shift if (use_shifted and bound_shift > 0) else 0.0)
                )[hasL]
                * np.maximum(zL[hasL], 1e-16)
            )
        if np.any(hasU):
            comp_vec.append(
                (
                    (ub - x)
                    + (bound_shift if (use_shifted and bound_shift > 0) else 0.0)
                )[hasU]
                * np.maximum(zU[hasU], 1e-16)
            )
        comp_vec = np.concatenate(comp_vec) if comp_vec else np.array([mu])

        disp = float(np.max(comp_vec) / max(1e-16, np.min(comp_vec)))
        if not (
            alpha_aff < alpha_thresh
            or disp > float(getattr(self.cfg, "ip_mc_dispersion", 8.0))
        ):
            return dx, dnu, ds, dlam, dzL, dzU

        iters_mc = int(getattr(self.cfg, "ip_gondzio_iters", 2))
        blend_mc = float(getattr(self.cfg, "ip_gondzio_blend", 0.3))
        mu_target_base = max(
            float(getattr(self.cfg, "ip_mu_min", 1e-12)), sigma * mu_aff
        )

        def _rhs_delta(mu_tgt: float):
            rhs = np.zeros_like(dx)
            if mI > 0 and Sigma_s_vec.size:
                rhs += JI.T @ (
                    _diag(Sigma_s_vec) @ ((mu_tgt - mu) / np.maximum(lmb, 1e-12))
                )
            if np.any(hasL):
                denomL = (x - lb) + (
                    bound_shift if (use_shifted and bound_shift > 0) else 0.0
                )
                rhs += np.where(hasL, (mu_tgt - mu) / np.maximum(denomL, 1e-12), 0.0)
            if np.any(hasU):
                denomU = (ub - x) + (
                    bound_shift if (use_shifted and bound_shift > 0) else 0.0
                )
                rhs -= np.where(hasU, (mu_tgt - mu) / np.maximum(denomU, 1e-12), 0.0)
            return rhs

        for k_mc in range(iters_mc):
            mu_tgt = mu_target_base if k_mc == 0 else 0.5 * mu_target_base
            rhs_x_mc = _rhs_delta(mu_tgt)
            if (reusable is not None) and hasattr(reusable, "solve"):
                dx_mc, dnu_mc = reusable.solve(
                    rhs_x_mc, r_pE, cg_tol=1e-8, cg_maxit=120
                )
            else:
                dx_mc, dnu_mc = self.solve_KKT(
                    W,
                    rhs_x_mc,
                    JE,
                    r_pE,
                    method=getattr(self.cfg, "ip_kkt_method", "hykkt"),
                    cg_tol=1e-8,
                    cg_maxit=120,
                )[:2]
                if dx_mc is None:
                    dx_mc, dnu_mc = self.solve_KKT(
                        W, rhs_x_mc, JE, r_pE, method="qdldl"
                    )[:2]

            dx_try = dx + blend_mc * dx_mc
            dnu_try = dnu + blend_mc * dnu_mc
            ds_mc = -(JI @ dx_mc) if (mI > 0 and JI is not None) else np.zeros_like(ds)
            ds_try = ds + blend_mc * ds_mc

            if mI > 0:
                denom_s = (
                    (s + (tau_shift if use_shifted else 0.0)) if use_shifted else s
                )
                dlam_mc = ((mu_tgt - mu) - lmb * ds_mc) / np.maximum(denom_s, 1e-12)
                dlam_try = dlam + blend_mc * dlam_mc
            else:
                dlam_try = dlam

            if np.any(hasL):
                denomL = (x - lb) + (
                    bound_shift if (use_shifted and bound_shift > 0) else 0.0
                )
                dzL_mc = np.where(hasL, _safe_pos_div(-(zL * dx_mc), denomL), 0.0)
                dzL_try = dzL + blend_mc * dzL_mc
            else:
                dzL_try = dzL

            if np.any(hasU):
                denomU = (ub - x) + (
                    bound_shift if (use_shifted and bound_shift > 0) else 0.0
                )
                dzU_mc = np.where(hasU, _safe_pos_div((zU * dx_mc), denomU), 0.0)
                dzU_try = dzU + blend_mc * dzU_mc
            else:
                dzU_try = dzU

            alpha_chk = k_alpha_fraction_to_boundary(
                x,
                dx_try,
                s,
                ds_try,
                lmb,
                dlam_try,
                lb,
                ub,
                hasL,
                hasU,
                float(
                    getattr(self.cfg, "ip_tau_pri", getattr(self.cfg, "ip_tau", 0.995))
                ),
                float(
                    getattr(self.cfg, "ip_tau_dual", getattr(self.cfg, "ip_tau", 0.995))
                ),
            )
            if alpha_chk >= 0.5 * max(alpha_aff, 1e-12):
                dx, dnu, ds, dlam, dzL, dzU = (
                    dx_try,
                    dnu_try,
                    ds_try,
                    dlam_try,
                    dzL_try,
                    dzU_try,
                )
            else:
                scale = (0.5 * max(alpha_aff, 1e-12)) / max(alpha_chk, 1e-12)
                sc = min(1.0, max(0.2, scale))
                dx += sc * (dx_try - dx)
                dnu += sc * (dnu_try - dnu)
                ds += sc * (ds_try - ds)
                dlam += sc * (dlam_try - dlam)
                dzL += sc * (dzL_try - dzL)
                dzU += sc * (dzU_try - dzU)

        return dx, dnu, ds, dlam, dzL, dzU

    # ---------- higher-order correction (unchanged logic; compact) ----------
    def _higher_order_correction(
        self,
        *,
        dx,
        dnu,
        ds,
        dlam,
        dzL,
        dzU,
        JI,
        JE,
        W,
        r_pE,
        s,
        lmb,
        zL,
        zU,
        sL,
        sU,
        hasL,
        hasU,
        mu,
        tau_shift,
        bound_shift,
        use_shifted,
    ):
        mI = int(s.size)
        rhs_x2 = (
            -(
                JI.T
                @ (
                    _diag(
                        np.maximum(
                            0.0,
                            lmb
                            / np.maximum(
                                s + (tau_shift if use_shifted else 0.0), 1e-12
                            ),
                        )
                    )
                    @ (JI @ dx)
                )
            )
            if (mI > 0 and JI is not None)
            else np.zeros_like(dx)
        )

        dx2, dnu2 = self.solve_KKT(
            W,
            rhs_x2,
            JE,
            (np.zeros_like(r_pE) if (r_pE is not None and np.size(r_pE)) else None),
            method=getattr(self.cfg, "ip_kkt_method", "lifted"),
            cg_tol=1e-8,
            cg_maxit=200,
        )
        if dx2 is None:
            dx2, dnu2 = self.solve_KKT(
                W,
                rhs_x2,
                JE,
                (np.zeros_like(r_pE) if (r_pE is not None and np.size(r_pE)) else None),
                method="qdldl",
            )

        tau2 = float(getattr(self.cfg, "ip_ho_blend", 0.3))
        caprel = float(getattr(self.cfg, "ip_ho_rel_cap", 0.5))
        n_dx = float(np.linalg.norm(dx)) or 1e-16
        n_dx2 = float(np.linalg.norm(dx2))
        if n_dx2 > caprel * n_dx:
            dx2 *= (caprel * n_dx) / max(n_dx2, 1e-16)

        dx = dx + tau2 * dx2
        dnu = dnu + tau2 * dnu2

        if mI > 0:
            ds2 = (-(JI @ dx2)) if JI is not None else np.zeros_like(ds)
            ds = ds + tau2 * ds2
            denom = (s + tau_shift) if use_shifted else s
            dlam += tau2 * _div(mu - denom * lmb - lmb * ds2, denom)

        dL_inc, dU_inc = self._dz_bounds_from_dx(
            dx=dx2,
            zL=zL,
            zU=zU,
            sL=sL,
            sU=sU,
            hasL=hasL,
            hasU=hasU,
            bound_shift=bound_shift,
            use_shifted=use_shifted,
            mu=None,
        )
        dzL += tau2 * dL_inc
        dzU += tau2 * dU_inc
        return dx, dnu, ds, dlam, dzL, dzU

    # ---------- predictor (Mehrotra affine) ----------
    def _mehrotra_affine_predictor(
        self,
        *,
        W,
        r_d,
        JE,
        r_pE,
        JI,
        r_pI,
        s,
        lmb,
        zL,
        zU,
        sL,
        sU,
        hasL,
        hasU,
        use_shifted: bool,
        tau_shift: float,
        bound_shift: float,
        mu: float,
        theta: float,
    ):
        mI = int(s.size)

        dx_aff, dnu_aff = self.solve_KKT(
            W,
            -r_d,
            JE,
            r_pE,
            method=getattr(self.cfg, "ip_kkt_method", "hykkt"),
            cg_tol=1e-8,
            cg_maxit=200,
        )
        if dx_aff is None:
            dx_aff, dnu_aff = self.solve_KKT(W, -r_d, JE, r_pE, method="qdldl")
        print("dx_aff", dx_aff)
        ds_aff = (
            (-(r_pI + (JI @ dx_aff if (mI > 0 and JI is not None) else 0)))
            if mI > 0
            else np.zeros(0)
        )
        if mI > 0:
            denom = (s + tau_shift) if use_shifted else s
            dlam_aff = _div(-(denom * lmb) - lmb * ds_aff, denom)
        else:
            dlam_aff = np.zeros(0)

        if use_shifted and bound_shift > 0:
            sL_shift = sL + bound_shift
            sU_shift = sU + bound_shift
            dzL_aff = np.where(
                hasL, _div(-(sL_shift * zL) - zL * dx_aff, sL_shift), 0.0
            )
            dzU_aff = np.where(
                hasU, _div(-(sU_shift * zU) + zU * dx_aff, sU_shift), 0.0
            )
        else:
            dzL_aff = np.where(hasL, _div(-sL * zL - zL * dx_aff, sL), 0.0)
            dzU_aff = np.where(hasU, _div(-sU * zU + zU * dx_aff, sU), 0.0)

        tau_pri = float(
            getattr(self.cfg, "ip_tau_pri", getattr(self.cfg, "ip_tau", 0.995))
        )
        tau_dual = float(
            getattr(self.cfg, "ip_tau_dual", getattr(self.cfg, "ip_tau", 0.995))
        )

        alpha_aff = 1.0
        if mI > 0:
            alpha_aff = min(alpha_aff, k_max_step_ftb(s, ds_aff, tau_pri))
            alpha_aff = min(alpha_aff, k_max_step_ftb(lmb, dlam_aff, tau_dual))
        if np.any(hasL):
            alpha_aff = min(alpha_aff, k_max_step_ftb(sL[hasL], dx_aff[hasL], tau_pri))
            alpha_aff = min(
                alpha_aff, k_max_step_ftb(zL[hasL], dzL_aff[hasL], tau_dual)
            )
        if np.any(hasU):
            alpha_aff = min(
                alpha_aff, k_max_step_ftb(sU[hasU], (-dx_aff)[hasU], tau_pri)
            )
            alpha_aff = min(
                alpha_aff, k_max_step_ftb(zU[hasU], dzU_aff[hasU], tau_dual)
            )

        mu_min = float(self.cfg.ip_mu_min)
        
        print('alpha_aff before cap', alpha_aff)
        # print ds_aff
        print(ds_aff)
        
        print(dlam_aff)
        print(dzL_aff)
        print(dzU_aff)
        print(lmb)
        print(s)
        print(use_shifted)
        print(sL)
        print(sU)

        parts = []
        if mI > 0:
            s_aff = s + alpha_aff * ds_aff
            lam_af = lmb + alpha_aff * dlam_aff
            parts.append(np.dot((s_aff + tau_shift) if use_shifted else s_aff, lam_af))
        if np.any(hasL):
            sL_aff = sL + alpha_aff * dx_aff
            zL_af = zL + alpha_aff * dzL_aff
            parts.append(
                np.dot(
                    (
                        (sL_aff + bound_shift)[hasL]
                        if (use_shifted and bound_shift > 0)
                        else sL_aff[hasL]
                    ),
                    zL_af[hasL],
                )
            )
        if np.any(hasU):
            sU_aff = sU - alpha_aff * dx_aff
            zU_af = zU + alpha_aff * dzU_aff
            parts.append(
                np.dot(
                    (
                        (sU_aff + bound_shift)[hasU]
                        if (use_shifted and bound_shift > 0)
                        else sU_aff[hasU]
                    ),
                    zU_af[hasU],
                )
            )

        denom = mI + int(np.sum(hasL)) + int(np.sum(hasU))
        print('denom', denom)
        print('parts', sum(parts) if parts else 0.0)
        mu_aff = max(mu_min, (sum(parts) / max(1, denom)) if parts else mu)

        pwr = float(getattr(self.cfg, "ip_sigma_power", 3.0))
        print('alpha_aff', alpha_aff)
        print('mu_aff', mu_aff)
        print('mu', mu)
        print('pwr', pwr)
        sigma = (
            0.0
            if (alpha_aff > 0.9)
            else float(
                np.clip(
                    ((1.0 - alpha_aff) ** 2) * (mu_aff / max(mu, mu_min)) ** pwr,
                    0.0,
                    1.0,
                )
            )
        )
        print('sigma', sigma)
        if float(theta) > float(self.cfg.ip_theta_clip):
            sigma = max(sigma, 0.5)

        return dict(
            dx_aff=dx_aff,
            dnu_aff=dnu_aff,
            ds_aff=ds_aff,
            dlam_aff=dlam_aff,
            dzL_aff=dzL_aff,
            dzU_aff=dzU_aff,
            alpha_aff=alpha_aff,
            mu_aff=mu_aff,
            sigma=sigma,
        )

    # ---------- one IP iteration ----------
    def step(
        self,
        model: Model,
        x: np.ndarray,
        lam: np.ndarray,
        nu: np.ndarray,
        it: int,
        ip_state: Optional[IPState] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        cfg = self.cfg
        # --- state
        st = (
            ip_state
            if (ip_state and ip_state.initialized)
            else IPState.from_model(model, x, cfg)
        )
        n, mI, mE = model.n, st.mI, st.mE
        s = st.s.copy()
        lmb = st.lam.copy()
        nuv = st.nu.copy()
        zL = st.zL.copy()
        zU = st.zU.copy()
        mu = float(st.mu)

        use_shifted = bool(getattr(cfg, "ip_use_shifted_barrier", False))
        tau_shift = float(st.tau_shift) if use_shifted else 0.0
        shift_adapt = bool(getattr(cfg, "ip_shift_adaptive", False))

        # --- eval (request only what we need)
        d0 = model.eval_all(x, components=("f", "g", "cI", "JI", "cE", "JE"))
        f = float(d0["f"])
        g = np.asarray(d0["g"], float)
        cI = np.asarray(d0["cI"], float) if mI > 0 else None
        cE = np.asarray(d0["cE"], float) if mE > 0 else None
        JI = d0["JI"] if mI > 0 else None
        JE = d0["JE"] if mE > 0 else None
        theta = model.constraint_violation(x)

        self.mI, self.mE, self.s = mI, mE, s

        # --- bounds & shifts
        lb, ub, hasL, hasU = _get_bounds(model, x)
        if use_shifted and shift_adapt:
            tau_shift = self._adaptive_shift_slack(
                s, (cI if cI is not None else np.zeros(0)), it
            )
            st.tau_shift = tau_shift
        bound_shift = (
            float(getattr(cfg, "ip_shift_bounds", 0.0)) if use_shifted else 0.0
        )
        if use_shifted and shift_adapt:
            bound_shift = self._adaptive_shift_bounds(x, lb, ub, hasL, hasU, it)

        # --- interior check
        bad = False
        if mI > 0:
            bad |= (
                (not np.all(np.isfinite(s)))
                or np.any(s <= 0)
                or (not np.all(np.isfinite(lmb)))
                or np.any(lmb <= 0)
            )
        bad |= (
            (not np.all(np.isfinite(zL)))
            or np.any(zL < 0)
            or (not np.all(np.isfinite(zU)))
            or np.any(zU < 0)
        )
        if bad:
            s, lmb, zL, zU = self._safe_reinit_slacks_duals(
                (cI if cI is not None else np.zeros(0)),
                JI,
                x,
                lb,
                ub,
                hasL,
                hasU,
                mu,
                tau_shift,
            )

        # --- Hessian + PSD regularization
        H = (
            model.lagrangian_hessian(x, lmb, nuv)
            if cfg.ip_exact_hessian
            else self.hess.get_hessian(model, x, lmb, nuv)
        )
        #H, _ = make_psd_advanced(H, self.reg, it)

        # --- residuals (branch on presence to avoid zeros allocs)
        print(JI)
        print(JE)
        print('lmb', lmb)
        print('nuv', nuv)
        r_d = (
            g
            + ((JI.T @ lmb) if JI is not None else 0)
            + ((JE.T @ nuv) if JE is not None else 0)
            - zL
            + zU
        )
        print('rd', r_d)
        r_pE = cE if mE > 0 else None
        r_pI = (cI + s) if mI > 0 else None
        print('rpE', r_pE)
        print('rpI', r_pI)

        # --- quick exit
        err_0 = self._compute_error(
            model,
            x,
            lmb,
            nuv,
            zL,
            zU,
            0.0,
            s=(s if mI > 0 else None),
            mI=(mI if mI > 0 else None),
        )
        tol = float(getattr(cfg, "tol", 1e-8))
        if err_0 <= tol:
            info = dict(
                mode="ip",
                step_norm=0.0,
                accepted=True,
                converged=True,
                f=f,
                theta=theta,
                stat=safe_inf_norm(r_d),
                ineq=(safe_inf_norm(np.maximum(0, cI)) if mI > 0 else 0.0),
                eq=(safe_inf_norm(cE) if mE > 0 else 0.0),
                comp=0.0,
                ls_iters=0,
                alpha=0.0,
                rho=0.0,
                tr_radius=(self.tr.radius if self.tr else 0.0),
                mu=mu,
            )
            return x, lmb, nuv, info

        # --- Σ terms (precompute sL/sU once)
        sL = np.where(hasL, np.maximum(1e-12, x - lb), 1.0)
        sU = np.where(hasU, np.maximum(1e-12, ub - x), 1.0)
        Sigma_x_vec, Sigma_s_vec = self._build_sigmas(
            zL=zL,
            zU=zU,
            sL=sL,
            sU=sU,
            hasL=hasL,
            hasU=hasU,
            lmb=lmb,
            s=(s if mI > 0 else np.zeros(0)),
            cI=(cI if mI > 0 else np.zeros(0)),
            tau_shift=tau_shift,
            bound_shift=bound_shift,
            use_shifted=use_shifted,
        )

        # --- form W efficiently
        # Add diagonal in-place when possible
        if sp.isspmatrix_csr(H) or sp.isspmatrix_csc(H):
            H = H.tocsr(copy=True)
            dH = H.diagonal()
            H.setdiag(dH + Sigma_x_vec)
            W = H
        else:
            W = H + _diag(Sigma_x_vec)

        if mI > 0 and JI is not None and Sigma_s_vec.size:
            # JIw = JI * Sigma_s along rows, so JIw = diag(Sigma_s) @ JI
            JIw = JI * Sigma_s_vec[:, None]
            W = W + (JI.T @ JIw)

        rhs_x = -r_d
        
        print('rhs_x', rhs_x)
        print('W', W)
        print("zl", zL)
        print("zu", zU)
        print("sl", sL)
        print("su", sU)

        # --- Mehrotra affine predictor
        aff = self._mehrotra_affine_predictor(
            W=W,
            r_d=r_d,
            JE=JE,
            r_pE=(r_pE if r_pE is not None else np.zeros(0)),
            JI=JI,
            r_pI=(r_pI if r_pI is not None else np.zeros(0)),
            s=(s if mI > 0 else np.zeros(0)),
            lmb=lmb,
            zL=zL,
            zU=zU,
            sL=sL,
            sU=sU,
            hasL=hasL,
            hasU=hasU,
            use_shifted=use_shifted,
            tau_shift=tau_shift,
            bound_shift=bound_shift,
            mu=mu,
            theta=theta,
        )
        alpha_aff, mu_aff, sigma = aff["alpha_aff"], aff["mu_aff"], aff["sigma"]
        print(aff)

        # --- μ pre-corrector (conservative)
        comp = self.compute_complementarity(
            (s if mI > 0 else np.zeros(0)),
            (lmb if mI > 0 else np.zeros(0)),
            mu,
            tau_shift,
            use_shifted,
        )
        if comp * max(1, mI) > 10.0 * mu:
            mu = min(comp * max(1, mI), 10.0)
        print('mu pre-corrector', mu)
        print('sigma', sigma)
        mu = max(float(cfg.ip_mu_min), sigma * mu_aff)
        print(mu)

        # --- corrector RHS (centrality + bounds) with precomputed denoms
        if mI > 0 and JI is not None and Sigma_s_vec.size:
            rc_s = (mu - (s + tau_shift) * lmb) if use_shifted else (mu - s * lmb)
            rhs_x = rhs_x + JI.T @ (
                _diag(Sigma_s_vec) @ (rc_s / np.maximum(lmb, 1e-12))
            )

        if hasL.any():
            denomL = sL + (bound_shift if (use_shifted and bound_shift > 0) else 0.0)
            rc_L = mu - denomL * zL
            rhs_x = rhs_x + np.where(hasL, rc_L / np.maximum(denomL, 1e-12), 0.0)

        if hasU.any():
            denomU = sU + (bound_shift if (use_shifted and bound_shift > 0) else 0.0)
            rc_U = mu - denomU * zU
            rhs_x = rhs_x - np.where(hasU, rc_U / np.maximum(denomU, 1e-12), 0.0)

        print('rhs_x corrector', rhs_x)
        print('mu', mu)
        print('mu_aff', mu_aff)

        # --- corrector solve (reuse enabled)
        dx, dnu, reusable = self.solve_KKT(
            W,
            rhs_x,
            JE,
            (cE if cE is not None else None),
            method=getattr(cfg, "ip_kkt_method", "qdldl"),
            cg_tol=1e-8,
            cg_maxit=200,
            return_reusable=True,
            Sigma_s_vec=Sigma_s_vec,
            JI=JI,
            mI=mI,
        )
        print('dx', dx)
        if dx is None:
            dx, dnu = self.solve_KKT(
                W,
                rhs_x,
                JE,
                (cE if cE is not None else None),
                method="qdldl",
                Sigma_s_vec=Sigma_s_vec,
                JI=JI,
                mI=mI,
            )
            reusable = None

        # --- recover ds, dλ, dz
        ds = (
            (-(r_pI + (JI @ dx if (mI > 0 and JI is not None) else 0)))
            if mI > 0
            else None
        )
        if mI > 0:
            denom_s = (s + tau_shift) if use_shifted else s
            dlam = _div(mu - denom_s * lmb - (lmb * ds), denom_s)
        else:
            dlam = None

        dzL, dzU = self._dz_bounds_from_dx(
            dx=dx,
            zL=zL,
            zU=zU,
            sL=sL,
            sU=sU,
            hasL=hasL,
            hasU=hasU,
            bound_shift=bound_shift,
            use_shifted=use_shifted,
            mu=mu,
        )

        # --- Gondzio MC (optional)
        if bool(getattr(cfg, "ip_gondzio_mc", False)) and (
            mI + int(hasL.sum()) + int(hasU.sum()) > 0
        ):
            dx, dnu, ds, dlam, dzL, dzU = self._gondzio_multicorrector(
                x=x,
                s=(s if mI > 0 else np.zeros(0)),
                lmb=lmb,
                zL=zL,
                zU=zU,
                lb=lb,
                ub=ub,
                hasL=hasL,
                hasU=hasU,
                JI=JI,
                JE=JE,
                W=W,
                Sigma_s_vec=Sigma_s_vec,
                r_pE=(cE if cE is not None else np.zeros(0)),
                reusable=reusable,
                dx=dx,
                dnu=dnu,
                ds=(ds if ds is not None else np.zeros(0)),
                dlam=(dlam if dlam is not None else np.zeros(0)),
                dzL=dzL,
                dzU=dzU,
                mu=mu,
                mu_aff=mu_aff,
                sigma=sigma,
                alpha_aff=alpha_aff,
                tau_shift=tau_shift,
                bound_shift=bound_shift,
                use_shifted=use_shifted,
            )

        # --- tiny trust region on dx (+ scale companions)
        dx_norm = float(np.linalg.norm(dx))
        dx_cap = float(cfg.ip_dx_max)
        if dx_norm > dx_cap and dx_norm > 0:
            sc = dx_cap / dx_norm
            dx *= sc
            dzL *= sc
            dzU *= sc
            if mI > 0:
                ds *= sc
                dlam *= sc

        # --- fraction-to-boundary + LS
        alpha_ftb = k_alpha_fraction_to_boundary(
            x,
            dx,
            (s if mI > 0 else np.zeros(0)),
            (ds if mI > 0 else np.zeros(0)),
            lmb,
            (dlam if mI > 0 else np.zeros(0)),
            lb,
            ub,
            hasL,
            hasU,
            float(getattr(cfg, "ip_tau_pri", getattr(cfg, "ip_tau", 0.995))),
            float(getattr(cfg, "ip_tau_dual", getattr(cfg, "ip_tau", 0.995))),
        )
        print('alpha_ftb', alpha_ftb)
        alpha_max = min(alpha_ftb, float(cfg.ip_alpha_max))

        # try:
        # alpha, ls_iters, needs_restoration = self.ls.search_ip(
        #     model=model,
        #     x=x,
        #     dx=dx,
        #     ds=(ds if mI > 0 else np.zeros(0)),
        #     s=(s if mI > 0 else np.zeros(0)),
        #     mu=mu,
        #     d_phi=float(g @ dx),
        #     theta0=theta,
        #     alpha_max=alpha_max,
        # )
        # except Exception:
        alpha, ls_iters, needs_restoration = min(1.0, alpha_max), 0, False
        print('alpha after ls', alpha)

        # --- early restoration
        if (alpha <= float(cfg.ls_min_alpha)) and needs_restoration:
            try:
                print("Entering feasibility restoration")
                x_new = self._feasibility_restoration(model, x, mu, self.filter)
                dN = model.eval_all(x_new, ("f", "g", "cI", "cE"))
                lmb_n = np.maximum(1e-8, lmb) if mI > 0 else lmb
                nu_n = np.zeros(mE) if mE > 0 else nuv
                zL_n = np.where(hasL, np.maximum(1e-8, zL), 0.0)
                zU_n = np.where(hasU, np.maximum(1e-8, zU), 0.0)
                info = dict(
                    mode="ip",
                    step_norm=float(np.linalg.norm(x_new - x)),
                    accepted=True,
                    converged=False,
                    f=float(dN["f"]),
                    theta=float(model.constraint_violation(x_new)),
                    stat=0.0,
                    ineq=0.0,
                    eq=0.0,
                    comp=0.0,
                    ls_iters=ls_iters,
                    alpha=0.0,
                    rho=0.0,
                    tr_radius=(self.tr.radius if self.tr else 0.0),
                    mu=mu,
                )
                st.s, st.lam, st.nu, st.zL, st.zU = s, lmb_n, nu_n, zL_n, zU_n
                return x_new, lmb_n, nu_n, info
            except ValueError:
                s, lmb, zL, zU = self._safe_reinit_slacks_duals(
                    (cI if cI is not None else np.zeros(0)),
                    JI,
                    x,
                    lb,
                    ub,
                    hasL,
                    hasU,
                    mu=max(mu, 1e-2),
                    tau_shift=tau_shift,
                )
                dxf = -g / max(1.0, np.linalg.norm(g))
                alpha = min(alpha_max, 1e-2)
                x_new = x + alpha * dxf
                info = dict(
                    mode="ip",
                    step_norm=float(np.linalg.norm(x_new - x)),
                    accepted=True,
                    converged=False,
                    f=model.eval_all(x_new, ["f"])["f"],
                    theta=float(model.constraint_violation(x_new)),
                    stat=0.0,
                    ineq=0.0,
                    eq=0.0,
                    comp=0.0,
                    ls_iters=ls_iters,
                    alpha=alpha,
                    rho=0.0,
                    tr_radius=(self.tr.radius if self.tr else 0.0),
                    mu=mu,
                )
                st.s, st.lam, st.nu, st.zL, st.zU = s, lmb, nuv, zL, zU
                return x_new, lmb, nuv, info

        # --- accept step (+ σ-box cap)
        print('alpha', alpha)
        print('dx', dx)
        x_new = x + alpha * dx
        s_new = (s + alpha * ds) if mI > 0 else s
        lmb_new = (lmb + alpha * dlam) if mI > 0 else lmb
        nu_new = (nuv + alpha * dnu) if mE > 0 else nuv
        zL_new = zL + alpha * dzL
        zU_new = zU + alpha * dzU

        sL_new = np.where(hasL, x_new - lb, 1.0)
        sU_new = np.where(hasU, ub - x_new, 1.0)
        zL_new, zU_new = self._cap_bound_duals_sigma_box(
            zL_new=zL_new,
            zU_new=zU_new,
            sL_new=sL_new,
            sU_new=sU_new,
            hasL=hasL,
            hasU=hasU,
            use_shifted=use_shifted,
            bound_shift=bound_shift,
            mu=mu,
            ksig=1e10,
        )

        # --- report (re-evaluate; keep correctness over micro-opts)
        dN = model.eval_all(x_new, ("f", "g", "cI", "JI", "cE", "JE"))
        f_new, g_new = float(dN["f"]), np.asarray(dN["g"], float)
        cI_new = np.asarray(dN["cI"], float) if mI > 0 else None
        cE_new = np.asarray(dN["cE"], float) if mE > 0 else None
        JI_new = dN["JI"] if mI > 0 else None
        JE_new = dN["JE"] if mE > 0 else None
        theta_new = model.constraint_violation(x_new)

        r_d_new = (
            g_new
            + ((JI_new.T @ lmb_new) if (mI > 0 and JI_new is not None) else 0)
            + ((JE_new.T @ nu_new) if (mE > 0 and JE_new is not None) else 0)
            - zL_new
            + zU_new
        )

        if use_shifted:
            r_comp_L_new = (sL_new + bound_shift) * zL_new - mu if hasL.any() else 0.0
            r_comp_U_new = (sU_new + bound_shift) * zU_new - mu if hasU.any() else 0.0
            r_comp_s_new = ((s_new + tau_shift) * lmb_new - mu) if mI > 0 else None
        else:
            r_comp_L_new = (sL_new * zL_new - mu) if hasL.any() else 0.0
            r_comp_U_new = (sU_new * zU_new - mu) if hasU.any() else 0.0
            r_comp_s_new = (s_new * lmb_new - mu) if mI > 0 else None

        kkt_new = dict(
            stat=np.linalg.norm(r_d_new, np.inf),
            ineq=(np.linalg.norm(np.maximum(0, cI_new), np.inf) if mI > 0 else 0.0),
            eq=(np.linalg.norm(cE_new, np.inf) if mE > 0 else 0.0),
            comp=max(
                (np.linalg.norm(r_comp_L_new, np.inf) if hasL.any() else 0.0),
                (np.linalg.norm(r_comp_U_new, np.inf) if hasU.any() else 0.0),
            )
            + (
                np.linalg.norm(r_comp_s_new, np.inf)
                if (mI > 0 and r_comp_s_new is not None)
                else 0.0
            ),
        )

        converged = (
            kkt_new["stat"] <= tol
            and kkt_new["ineq"] <= tol
            and kkt_new["eq"] <= tol
            and kkt_new["comp"] <= tol
            and mu <= tol / 10
        )

        # Avoid costly dense cond unless explicitly requested
        if bool(getattr(cfg, "ip_debug_condH", False)):
            try:
                cond_H = np.linalg.cond(H.toarray() if hasattr(H, "toarray") else H)
            except Exception:
                cond_H = np.nan
        else:
            cond_H = np.nan

        mu = self.update_mu(
            mu=mu,
            s=(s_new if mI > 0 else np.zeros(0)),
            lam=(lmb_new if mI > 0 else np.zeros(0)),
            theta=theta_new,
            kkt=kkt_new,
            accepted=True,
            cond_H=cond_H,
            sigma=sigma,
            mu_aff=mu_aff,
            use_shifted=use_shifted,
            tau_shift=tau_shift,
        )

        info = dict(
            mode="ip",
            step_norm=float(np.linalg.norm(x_new - x)),
            accepted=True,
            converged=converged,
            f=f_new,
            theta=theta_new,
            stat=kkt_new["stat"],
            ineq=kkt_new["ineq"],
            eq=kkt_new["eq"],
            comp=kkt_new["comp"],
            ls_iters=0,
            alpha=float(alpha),
            rho=0.0,
            tr_radius=(self.tr.radius if self.tr else 0.0),
            mu=mu,
            shifted_barrier=use_shifted,
            tau_shift=tau_shift,
            bound_shift=bound_shift,
        )

        st.s, st.lam, st.nu, st.zL, st.zU, st.mu = (
            s_new,
            lmb_new,
            nu_new,
            zL_new,
            zU_new,
            mu,
        )
        st.tau_shift = tau_shift
        return x_new, lmb_new, nu_new, info
