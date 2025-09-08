# sqp_ip.py
# InteriorPointStepper: slacks-only barrier IPM with "reference" mode (μ/s²)
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import qdldl_cpp as qd
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .blocks.aux import HessianManager, Model, SQPConfig
from .blocks.filter import Filter
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
    z: np.ndarray  # duals for bounds x >= 0 (new: primal-dual for bounds)
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
        z0 = np.ones_like(x)  # new: initialize z for bounds
        return IPState(
            mI=mI, mE=mE, s=s0, z=z0, lam=lam, nu=nu, mu=mu0, initialized=True
        )


def _upper_csc(A):
    """Ensure upper-triangular CSC with int64/float64 dtypes."""
    A = sp.csc_matrix(A, copy=False)
    U = sp.triu(A, format="csc")
    U.sort_indices()
    # exact dtypes required by bindings
    U.indptr = U.indptr.astype(np.int64, copy=False)
    U.indices = U.indices.astype(np.int64, copy=False)
    U.data = U.data.astype(np.float64, copy=False)
    return U


def _ordering_perm(A_csc):
    """Return permutation (int64) or None."""
    try:
        # Prefer AMD via scikit-sparse / CHOLMOD if available
        from sksparse.cholmod import cholesky_AAt

        G = (A_csc + A_csc.T).astype(np.float64)
        F = cholesky_AAt(G)
        return np.asarray(F.P(), dtype=np.int64)
    except Exception:
        try:
            from scipy.sparse.csgraph import reverse_cuthill_mckee as rcm

            G = ((A_csc + A_csc.T) > 0).astype(np.int8)
            return np.asarray(rcm(G, symmetric_mode=True), dtype=np.int64)
        except Exception:
            return None

def safe_inf_norm(v: np.ndarray) -> float:
    """Compute infinity norm safely, handling empty arrays."""
    return np.linalg.norm(v, np.inf) if v.size > 0 else 0.0

# ------------------ stepper ------------------
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
        self.filter = Filter(self.cfg)
        self.funnel = funnel
        self.ls = ls  # may be None; if None we fall back to internal LS
        self.soc = soc  # may be None; if None we skip SOC rescue
        self._ensure_cfg_defaults()

    def _ensure_cfg_defaults(self):
        def add(name, val):
            if not hasattr(self.cfg, name):
                setattr(self.cfg, name, val)

        # algebra / hessian mode
        add("ip_match_ref_form", False)
        add("ip_exact_hessian", False)
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
        use_ordering=False,
        reuse_symbolic=True,
        refine_iters=0,
    ):
        # Modified for inertia correction (Algorithm IC)
        n = Wmat.shape[0]
        mE = 0 if JE_mat is None else JE_mat.shape[0]
        delta_w_last = getattr(self, "_delta_w_last", 0.0)  # persist across calls
        delta_w = 0.0
        delta_c = 0.0
        attempts = 0
        max_attempts = 10  # limit
        while attempts < max_attempts:
            # Assemble KKT as before, but with delta_w I in W, -delta_c I in bottom-right
            if mE > 0:
                Wcsr = _csr(Wmat + delta_w * sp.eye(n))
                JEcsr = _csr(JE_mat)
                B22 = -delta_c * sp.eye(mE) if delta_c > 0 else sp.csr_matrix((mE, mE))
                K = sp.vstack(
                    [sp.hstack([Wcsr, JEcsr.T]), sp.hstack([JEcsr, B22])], format="csc"
                )
                rhs = np.concatenate([rhs_x, -rpE])
            else:
                K = _csr(Wmat + delta_w * sp.eye(n))
                rhs = rhs_x
            K_upper = _upper_csc(K)
            nsys = K_upper.shape[0]
            perm = _ordering_perm(K_upper) if use_ordering else None
            # Factorize and check inertia (QDLDL provides D values for inertia count)
            try:
                fac = qd.factorize(
                    K_upper.indptr, K_upper.indices, K_upper.data, nsys, perm=perm
                )
                # Get inertia: count positive/negative in D (diagonal of L D L^T)
                D = fac.D  # assume qdldl exposes D
                num_pos = np.sum(D > 0)
                num_neg = np.sum(D < 0)
                num_zero = np.sum(np.abs(D) < 1e-12)  # approx zero
                if num_pos == n and num_neg == mE and num_zero == 0:
                    # Correct inertia
                    self._delta_w_last = delta_w
                    x = qd.solve(fac, rhs)
                    if refine_iters > 0:
                        x = qd.solve_refine(fac, rhs, refine_iters)
                    return x[:n], x[n:] if mE > 0 else (x, np.zeros(0))
                # Adjust deltas per IC
                if num_zero > 0:
                    delta_c = self.cfg.ip_delta_c_bar * self.mu**self.cfg.ip_kappa_c
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
                print("KKT factorization failed, adjusting deltas")
                # Fallback to SciPy as before
                sol = spla.spsolve(K.tocsc(), rhs)
                return sol[:n], sol[n:] if mE > 0 else (sol, np.zeros(0))
        raise ValueError(
            "Inertia correction failed after max attempts - switch to restoration"
        )

    def _compute_error(self, model: Model, x, lam, nu, z, mu: float = 0.0) -> float:
        # IPOPT's E_mu or E_0 (5)
        data = model.eval_all(x, components=["f", "g", "cI", "JI", "cE", "JE"])
        g = np.asarray(data["g"], float)
        cI = np.asarray(data["cI"], float) if data["cI"] is not None else np.zeros(0)
        cE = np.asarray(data["cE"], float) if data["cE"] is not None else np.zeros(0)
        JI = data["JI"]
        JE = data["JE"]
        r_d = (
            g
            + (JI.T @ lam if JI is not None else 0)
            + (JE.T @ nu if JE is not None else 0)
            - z
        )  # new: -z for bounds
        r_cI = cI
        r_cE = cE
        r_comp_bounds = x * z - mu * np.ones_like(x)  # new: comp for bounds
        r_comp_slacks = (
            self.s * lam - mu * np.ones_like(self.s) if self.mI > 0 else np.zeros(0)
        )
        s_max = getattr(self.cfg, "ip_s_max", 100.0)
        s_d = (
            max(
                s_max,
                (np.sum(np.abs(lam)) + np.sum(np.abs(nu)) + np.sum(np.abs(z)))
                / (self.mI + self.mE + len(x)),
            )
            / s_max
        )
        s_c = max(s_max, np.sum(np.abs(z)) / len(x)) / s_max
        err = max(
            np.linalg.norm(r_d, np.inf) / s_d,
            np.linalg.norm(r_cE, np.inf) if self.mE > 0 else 0.0,
            np.linalg.norm(r_cI, np.inf) if self.mI > 0 else 0.0,
            np.linalg.norm(r_comp_bounds, np.inf) / s_c,
            np.linalg.norm(r_comp_slacks, np.inf) / s_c,
        )
        return err

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
        Perform one iteration of the primal-dual interior-point method with filter line-search.
        Follows IPOPT's Algorithm A, integrating primal-dual equations, filter method,
        second-order corrections, and feasibility restoration.
        """
        # Initialize or retrieve state
        st = (
            ip_state
            if (ip_state and ip_state.initialized)
            else IPState.from_model(model, x, self.cfg)
        )
        n = model.n
        mI, mE = st.mI, st.mE
        s, lmb, nuv, z, mu = (
            st.s.copy(),
            st.lam.copy(),
            st.nu.copy(),
            st.z.copy(),
            float(st.mu),
        )
        tau = max(self.cfg.ip_tau_min, 1 - mu)  # Fraction-to-boundary parameter (8)

        # Evaluate model at current point
        data = model.eval_all(x, components=["f", "g", "cI", "JI", "cE", "JE"])
        f = float(data["f"])
        g = np.asarray(data["g"], float)
        cI = np.zeros(mI) if mI == 0 else np.asarray(data["cI"], float)
        cE = np.zeros(mE) if mE == 0 else np.asarray(data["cE"], float)
        JI = data["JI"] if mI > 0 else None
        JE = data["JE"] if mE > 0 else None
        theta = model.constraint_violation(x)  # ||cE|| + ||max(0, -cI)||

        self.mI = mI
        self.s = s
        self.mE = mE

        # Compute Hessian of Lagrangian
        H = (
            model.lagrangian_hessian(x, lmb, nuv)
            if self.cfg.ip_exact_hessian
            else self.hess.get_hessian(model, x, lmb, nuv)
        )
        H, _ = make_psd_advanced(H, self.reg, it)

        # Compute residuals for primal-dual equations (4)
        r_d = (
            g
            + (JI.T @ lmb if JI is not None else 0)
            + (JE.T @ nuv if JE is not None else 0)
            - z
        )
        r_pE = cE
        r_pI = cI + s if mI > 0 else np.zeros(0)
        r_comp_x = x * z - mu * np.ones(n)  # Complementarity for bounds
        r_comp_s = s * lmb - mu * np.ones(mI) if mI > 0 else np.zeros(0)

        # Compute optimality error E_mu (5)
        s_max = getattr(self.cfg, "ip_s_max", 100.0)
        s_d = (
            max(
                s_max,
                (np.sum(np.abs(lmb)) + np.sum(np.abs(nuv)) + np.sum(np.abs(z)))
                / (mI + mE + n),
            )
            / s_max
        )
        s_c = max(s_max, np.sum(np.abs(z)) / n) / s_max

        err_mu = max(
            np.linalg.norm(r_d, np.inf) / s_d,
            np.linalg.norm(r_pE, np.inf) if mE > 0 else 0.0,
            np.linalg.norm(r_pI, np.inf) if mI > 0 else 0.0,
            np.linalg.norm(r_comp_x, np.inf) / s_c,
            np.linalg.norm(r_comp_s, np.inf) / s_c,
        )
        err_0 = self._compute_error(model, x, lmb, nuv, z, 0.0)  # E_0 for convergence

        # Convergence checks
        tol = getattr(self.cfg, "tol", 1e-8)
        if err_0 <= tol:
            info = self._pack_info(
                step_norm=0.0,
                accepted=True,
                converged=True,
                f=f,
                theta=theta,
                kkt={
                    "stat": np.linalg.norm(r_d, np.inf),
                    "ineq": np.linalg.norm(np.maximum(0, -cI), np.inf),
                    "eq": np.linalg.norm(cE, np.inf),
                    "comp": np.linalg.norm(r_comp_x, np.inf)
                    + np.linalg.norm(r_comp_s, np.inf),
                },
                alpha=0.0,
                rho=0.0,
                mu=mu,
                ls_iters=0,
            )
            return x, lmb, nuv, info

        # Update mu if tolerance met (7)
        if err_mu <= self.cfg.ip_kappa_eps * mu:
            mu_new = max(
                tol / 10, min(self.cfg.ip_kappa_mu * mu, mu**self.cfg.ip_theta_mu)
            )
            st.mu = mu_new
            self.filter = Filter(self.cfg)  # Reset filter
            tau = max(self.cfg.ip_tau_min, 1 - mu_new)
            mu = mu_new

        # Solve barrier problem (11)
        Sigma_x = z / x  # Diagonal for bounds
        Sigma_s = lmb / s if mI > 0 else np.zeros(0)  # Diagonal for slacks
        W = H + _diag(Sigma_x) + (JI.T @ _diag(Sigma_s) @ JI if mI > 0 else 0)
        rhs_x = -r_d
        dx, dnu = self.solve_KKT(W, rhs_x, JE, r_pE)
        ds = -r_pI - (JI @ dx if JI is not None else 0) if mI > 0 else np.zeros(0)
        dlam = (
            -(lmb - mu / s) - (mu / (s**2) * ds) if mI > 0 else np.zeros(0)
        )  # Adjust for reference mode
        dz = mu / x - z - (z / x * dx)

        # Fraction-to-boundary step sizes (15)
        alpha_max_x = self._max_step_ftb(x, dx, tau)
        alpha_max_z = self._max_step_ftb(z, dz, tau)
        alpha_max_s = self._max_step_ftb(s, ds, tau) if mI > 0 else 1.0
        alpha_max = min(alpha_max_x, alpha_max_s)

        # Barrier objective and gradient
        phi_0 = f - mu * (
            np.sum(np.log(x)) + np.sum(np.log(s)) if mI > 0 else np.sum(np.log(x))
        )
        grad_phi = g - mu / x  # Approximate ∇ϕ_μ
        d_phi = grad_phi @ dx

        # Filter line-search (Section 2.3)
        alpha = alpha_max
        ls_iters = 0
        gamma_theta, gamma_phi, delta, s_theta, s_phi, eta_phi = (
            self.cfg.ip_gamma_theta,
            self.cfg.ip_gamma_phi,
            self.cfg.ip_delta,
            self.cfg.ip_s_theta,
            self.cfg.ip_s_phi,
            self.cfg.ip_eta_phi,
        )
        alpha, ls_iters, needs_restoration = self.ls.search_ip(
            model=model,
            x=x,
            dx=dx,
            ds=ds,
            s=s,
            mu=mu,
            d_phi=d_phi,
            theta0=theta,
            alpha_max=alpha_max
        )
        
        # Second-order correction if first trial would increase theta (Section 2.4)
        if ls_iters == 0 and alpha == 0.0 and needs_restoration:
            for soc_iter in range(self.cfg.ip_pmax_soc):
                cE_soc = cE + (JE @ dx if JE is not None else 0) if mE > 0 else np.zeros(0)
                cI_soc = cI + (JI @ dx if JI is not None else 0) if mI > 0 else np.zeros(0)
                rhs_soc = -np.concatenate([cE_soc, cI_soc]) if mE + mI > 0 else np.zeros(0)
                dx_soc, _ = self.solve_KKT(W, rhs_soc[:n], JE, rhs_soc[n:] if mE > 0 else None)
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
                        alpha_max=alpha_soc
                    )
                    break
                if soc_iter == self.cfg.ip_pmax_soc - 1:
                    break

        # Handle restoration if needed
        if alpha == 0.0 and needs_restoration:
            try:
                x_new = self._feasibility_restoration(model, x, mu, self.filter)
                # Recompute multipliers (approximate)
                data_new = model.eval_all(x_new, ["g", "cI", "JI", "cE", "JE"])
                g_new = data_new["g"]
                JI_new = data_new["JI"]
                JE_new = data_new["JE"]
                cI_new = data_new["cI"]
                cE_new = data_new["cE"]
                lmb_new = np.maximum(1e-8, lmb) if mI > 0 else lmb
                nu_new = np.zeros(mE) if mE > 0 else nuv
                z_new = np.maximum(1e-8, z)  # Reset z
                info = self._pack_info(
                    step_norm=np.linalg.norm(x_new - x), accepted=True, converged=False,
                    f=model.eval_all(x_new, ["f"])["f"], theta=model.constraint_violation(x_new),
                    kkt={}, alpha=0.0, rho=0.0, mu=mu, ls_iters=ls_iters
                )
                st.s, st.lam, st.nu, st.z = s, lmb_new, nu_new, z_new
                return x_new, lmb_new, nu_new, info
            except ValueError as e:
                # Infeasibility detected
                info = self._pack_info(
                    step_norm=0.0, accepted=False, converged=False, f=f, theta=theta,
                    kkt={"stat": safe_inf_norm(r_d), "ineq": safe_inf_norm(np.maximum(0, -cI)),
                        "eq": safe_inf_norm(cE), "comp": safe_inf_norm(r_comp_x) + safe_inf_norm(r_comp_s)},
                    alpha=0.0, rho=0.0, mu=mu, ls_iters=ls_iters
                )
                info["error"] = str(e)
                return x, lmb, nuv, info
            
        # Accept step
        x_new = x + alpha * dx
        s_new = s + alpha * ds if mI > 0 else s
        lmb_new = lmb + alpha * dlam if mI > 0 else lmb
        nu_new = nuv + alpha * dnu if mE > 0 else nuv
        z_new = z + min(alpha, alpha_max_z) * dz
        # Apply z correction (16)
        z_new = np.maximum(
            mu / self.cfg.ip_kappa_sigma / x_new,
            np.minimum(z_new, self.cfg.ip_kappa_sigma * mu / x_new),
        )

        # # Update filter (22)
        # if not (
        #     theta <= theta_min
        #     and switching
        #     and phi_trial <= phi_0 + eta_phi * alpha * d_phi
        # ):
        #     self.filter.add_if_acceptable((1 - gamma_theta) * theta, phi_0 - gamma_phi * theta)

        # Compute new residuals and check convergence
        data_new = model.eval_all(x_new, ["f", "cI", "cE"])
        f_new = float(data_new["f"])
        cI_new = np.asarray(data_new["cI"], float) if mI > 0 else np.zeros(0)
        cE_new = np.asarray(data_new["cE"], float) if mE > 0 else np.zeros(0)
        theta_new = model.constraint_violation(x_new)
        r_d_new = (
            g
            + (JI.T @ lmb_new if JI is not None else 0)
            + (JE.T @ nu_new if JE is not None else 0)
            - z_new
        )
        r_comp_x_new = x_new * z_new
        r_comp_s_new = s_new * lmb_new if mI > 0 else np.zeros(0)
        kkt_new = {
            "stat": np.linalg.norm(r_d_new, np.inf),
            "ineq": np.linalg.norm(np.maximum(0, -cI_new), np.inf) if mI > 0 else 0.0,
            "eq": np.linalg.norm(cE_new, np.inf) if mE > 0 else 0.0,
            "comp": np.linalg.norm(r_comp_x_new, np.inf)
            + np.linalg.norm(r_comp_s_new, np.inf),
        }
        converged = (
            kkt_new["stat"] <= tol
            and kkt_new["ineq"] <= tol
            and kkt_new["eq"] <= tol
            and kkt_new["comp"] <= tol
            and mu <= tol / 10
        )

        # Pack info
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
            #ls_iters=ls_iters,
        )
        st.s, st.lam, st.nu, st.z, st.mu = s_new, lmb_new, nu_new, z_new, mu
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
