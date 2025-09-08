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
        return IPState(mI=mI, mE=mE, s=s0, lam=lam, nu=nu, mu=mu0, initialized=True)

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
        self.filter = flt
        self.funnel = funnel
        self.ls = ls  # may be None; if None we fall back to internal LS
        self.soc = soc  # may be None; if None we skip SOC rescue
        self._ensure_cfg_defaults()

    def _ensure_cfg_defaults(self):
        def add(name, val):
            if not hasattr(self.cfg, name):
                setattr(self.cfg, name, val)

        # algebra / hessian mode
        add("ip_match_ref_form", True)
        add("ip_exact_hessian", True)
        add("ip_hess_reg0", 1e-4)
        add("ip_eq_reg", 1e-4)

        # barrier & predictor-corrector (used only if ip_match_ref_form=False)
        add("ip_mu_init", 1e-1)
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
        """
        Solve:
            [ W   J^T ] [dx] = [rhs_x]
            [ J   B22 ] [λ ]   [-rpE ]
        where B22 = ip_eq_reg * I (usually small, > 0).  If mE == 0, solves W dx = rhs_x.

        Internally:
        - Converts to upper CSC
        - (Optionally) orders with AMD/RCM
        - Uses qd.analyze / qd.refactorize for speed on repeated solves
        """
        n = Wmat.shape[0]
        mE = 0 if JE_mat is None else JE_mat.shape[0]

        if mE > 0:
            # Assemble KKT in sparse blocks
            Wcsr = Wmat if sp.isspmatrix_csr(Wmat) else sp.csr_matrix(Wmat)
            JEcsr = JE_mat if sp.isspmatrix_csr(JE_mat) else sp.csr_matrix(JE_mat)
            ip_eq = float(getattr(self.cfg, "ip_eq_reg", 1e-4))
            B22 = ip_eq * sp.eye(mE, format="csr")

            K = sp.vstack(
                [
                    sp.hstack([Wcsr, JEcsr.T], format="csr"),
                    sp.hstack([JEcsr, B22], format="csr"),
                ],
                format="csc",
            )
            rhs = np.concatenate([rhs_x, -rpE])
        else:
            # No equality constraints: just W
            K = sp.csc_matrix(Wmat)  # may be dense → sparse
            rhs = np.asarray(rhs_x)

        # QDLDL expects *upper* CSC (including diagonal)
        K_upper = _upper_csc(K)
        nsys = K_upper.shape[0]

        # Optional ordering
        perm = _ordering_perm(K_upper) if use_ordering else None

        # Cache symbolic to reuse between calls if pattern is unchanged
        # You can keep these attributes on `self` (or another long-lived object)
        have_cache = (
            reuse_symbolic
            and hasattr(self, "_qd_sym")
            and getattr(self, "_qd_sym_n", None) == nsys
        )
        try:
            if reuse_symbolic:
                if not have_cache:
                    self._qd_sym = qd.analyze(
                        K_upper.indptr,
                        K_upper.indices,
                        K_upper.data,
                        np.int64(nsys),
                        perm=perm,
                    )
                    self._qd_sym_n = nsys
                fac = qd.refactorize(
                    self._qd_sym,
                    K_upper.indptr,
                    K_upper.indices,
                    K_upper.data,
                    np.int64(nsys),
                )
            else:
                fac = qd.factorize(
                    K_upper.indptr,
                    K_upper.indices,
                    K_upper.data,
                    np.int64(nsys),
                    perm=perm,
                )

            x = qd.solve(fac, rhs.astype(np.float64, copy=False))
            if refine_iters and refine_iters > 0:
                x = qd.solve_refine(
                    fac, rhs.astype(np.float64, copy=False), int(refine_iters)
                )

            if mE > 0:
                return x[:n], x[n:]
            else:
                return x, np.zeros(0, dtype=x.dtype)

        except Exception as e:
            print(f"[QDLDL] KKT solve failed: {e}")
            # Fall back to SciPy if factorization fails
            # (use LU or dense solve depending on structure)
            try:
                sol = spla.spsolve(K.tocsc(), rhs)
            except Exception:
                sol = np.linalg.solve(K.toarray(), rhs)
            if mE > 0:
                return sol[:n], sol[n:]
            else:
                return sol, np.zeros(0, dtype=sol.dtype)

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
        # -------------------- setup & state --------------------
        n = model.n
        st = (
            ip_state
            if (ip_state and ip_state.initialized)
            else IPState.from_model(model, x, self.cfg)
        )
        mI, mE = st.mI, st.mE
        s, lmb, nuv, mu = st.s.copy(), st.lam.copy(), st.nu.copy(), float(st.mu)
        rho = float(getattr(self.cfg, "ip_rho_init", 1.0))
        use_soc = bool(getattr(self.cfg, "use_soc", False))
        soc_when = str(getattr(self.cfg, "soc_when", "after_ls_accept"))
        allow_null_accept = bool(getattr(self.cfg, "ip_allow_null_accept", True))
        # optional TR radius for SOC cap
        Delta = None
        if getattr(self, "tr", None) is not None and hasattr(self.tr, "radius"):
            try:
                Delta = float(self.tr.radius)
                if not np.isfinite(Delta) or Delta <= 0.0:
                    Delta = None
            except Exception:
                Delta = None

        # -------------------- evaluate base point --------------------
        data = model.eval_all(x, components=["f", "g", "cI", "JI", "cE", "JE"])
        f, g = float(data["f"]), np.asarray(data["g"], float)
        cI = (
            np.zeros(mI)
            if (mI == 0 or data["cI"] is None)
            else np.asarray(data["cI"], float)
        )
        cE = (
            np.zeros(mE)
            if (mE == 0 or data["cE"] is None)
            else np.asarray(data["cE"], float)
        )
        JI = None if mI == 0 else data["JI"]
        JE = None if mE == 0 else data["JE"]

        # -------------------- Hessian of L --------------------
        if bool(getattr(self.cfg, "ip_exact_hessian", True)):
            H = model.lagrangian_hessian(x, lmb, nuv)
            H, _ = make_psd_advanced(H, self.reg, it)
        else:
            H = self.hess.get_hessian(model, x, lmb, nuv)
            H, _ = make_psd_advanced(
                H, self.reg, it
            )

        # -------------------- residuals at x --------------------
        r_d = g.copy()
        if mI > 0:
            r_d += JI.T @ lmb if sp.isspmatrix(JI) else np.asarray(JI).T @ lmb
        if mE > 0:
            r_d += JE.T @ nuv if sp.isspmatrix(JE) else np.asarray(JE).T @ nuv
        r_pI = cI + s
        r_pE = cE

        # norms
        stat = float(np.linalg.norm(r_d, ord=np.inf))
        feasI = (
            0.0 if mI == 0 else float(np.linalg.norm(np.maximum(0.0, cI), ord=np.inf))
        )
        feasE = 0.0 if mE == 0 else float(np.linalg.norm(cE, ord=np.inf))
        comp = 0.0 if mI == 0 else float(np.linalg.norm(s * lmb - mu, ord=np.inf))
        theta = model.constraint_violation(x)

        # quick exit if already converged (including μ)
        if (
            stat <= getattr(self.cfg, "ip_tol_stat", 1e-8)
            and max(feasI, feasE) <= getattr(self.cfg, "ip_tol_feas", 1e-8)
            and (0.0 if mI == 0 else abs(s @ lmb) / max(1, mI))
            <= getattr(self.cfg, "ip_tol_comp", 1e-8)
            and mu <= getattr(self.cfg, "ip_mu_min", 1e-12)
        ):
            info = self._pack_info(
                0.0,
                True,
                True,
                f,
                theta,
                {"stat": stat, "ineq": feasI, "eq": feasE, "comp": comp},
                1.0,
                0.0,
                mu,
            )
            return x, lmb, nuv, info

        # -------------------- build W and RHS --------------------
        def _isspm(A):
            return sp.isspmatrix(A)

        def _csr(A, shape=None):
            if A is None:
                if shape is None:
                    raise ValueError("shape required when A is None")
                return sp.csr_matrix(shape)
            return A.tocsr() if _isspm(A) else sp.csr_matrix(A)

        def _sym(A):
            return (A + A.T) * 0.5 if _isspm(A) else 0.5 * (A + A.T)

        def _diag(v: np.ndarray):
            return sp.diags(v) if v.ndim == 1 else sp.diags(np.asarray(v).ravel())

        if mI > 0:
            s = np.maximum(s, 1e-12)
            lmb = np.maximum(lmb, 1e-12)
            if bool(getattr(self.cfg, "ip_match_ref_form", True)):
                # reference μ/s^2 algebra
                Dref = mu / (s**2)
                if _isspm(H) or _isspm(JI):
                    Hs = _csr(H, (n, n))
                    JIcsr = _csr(JI, (mI, n))
                    W = _sym(Hs + JIcsr.T @ _diag(Dref) @ JIcsr)
                else:
                    W = _sym(H + (np.asarray(JI).T * Dref) @ np.asarray(JI))
                r2 = lmb - mu / s
                r3 = r_pI
                rhs_core = -r2 + (mu / (s**2)) * r3
                rhs_x = -r_d - (
                    (JI.T @ rhs_core) if _isspm(JI) else (np.asarray(JI).T @ rhs_core)
                )
            else:
                # Mehrotra λ/s algebra
                D = lmb / s
                if _isspm(H) or _isspm(JI):
                    Hs = _csr(H, (n, n))
                    JIcsr = _csr(JI, (mI, n))
                    W = _sym(Hs + JIcsr.T @ _diag(D) @ JIcsr)
                else:
                    W = _sym(H + (np.asarray(JI).T * D) @ np.asarray(JI))
                rhs_x_aff = -r_d - (
                    (JI.T @ (D * r_pI))
                    if _isspm(JI)
                    else (np.asarray(JI).T @ (D * r_pI))
                )
        else:
            W = _sym(H)
            rhs_x = -r_d

        if mI == 0:
            dx, dnu = self.solve_KKT(W, rhs_x, JE, r_pE)
            ds = np.zeros(0)
            dlam = np.zeros(0)
        else:
            if bool(getattr(self.cfg, "ip_match_ref_form", True)):
                dx, dnu = self.solve_KKT(W, rhs_x, JE, r_pE)
                JI_dx = (JI @ dx) if not _isspm(JI) else JI.dot(dx)
                ds = -r_pI - JI_dx
                dlam = -(lmb - mu / s) - (mu / (s**2)) * ds
            else:
                dx_aff, dnu_aff = self.solve_KKT(W, rhs_x_aff, JE, r_pE)
                JI_dx_aff = (JI @ dx_aff) if not _isspm(JI) else JI.dot(dx_aff)
                ds_aff = -r_pI - JI_dx_aff
                dl_aff = (lmb / s) * (r_pI + JI_dx_aff) - lmb
                a_p_aff = self._max_step_ftb(
                    s, ds_aff, float(getattr(self.cfg, "ip_tau", 0.995))
                )
                a_d_aff = self._max_step_ftb(
                    lmb, dl_aff, float(getattr(self.cfg, "ip_tau", 0.995))
                )
                mu_aff = ((s + a_p_aff * ds_aff) @ (lmb + a_d_aff * dl_aff)) / max(
                    1, mI
                )
                sigma = (mu_aff / max(mu, 1e-16)) ** float(
                    getattr(self.cfg, "ip_sigma_power", 3.0)
                )
                rhs_corr = (lmb / s) * r_pI + ((sigma * mu) / s - lmb)
                rhs_x = -r_d - (
                    (JI.T @ rhs_corr) if _isspm(JI) else (np.asarray(JI).T @ rhs_corr)
                )
                dx, dnu = self.solve_KKT(W, rhs_x, JE, r_pE)
                JI_dx = (JI @ dx) if not _isspm(JI) else JI.dot(dx)
                ds = -r_pI - JI_dx
                dlam = (lmb / s) * (r_pI + JI_dx) + ((sigma * mu) / s - lmb)

        # -------------------- fraction-to-boundary & LS --------------------
        a_p = (
            self._max_step_ftb(s, ds, float(getattr(self.cfg, "ip_tau", 0.995)))
            if mI > 0
            else 1.0
        )
        a_d = (
            self._max_step_ftb(lmb, dlam, float(getattr(self.cfg, "ip_tau", 0.995)))
            if mI > 0
            else 1.0
        )
        alpha_max = float(min(float(getattr(self.cfg, "ip_alpha_max", 1.0)), a_p, a_d))

        ls_iters = 0
        if (self.ls is not None) and (mI > 0):
            theta0 = theta
            alpha, ls_iters, _needs_restoration = self.ls.search_ip(
                model=model,
                x=x,
                dx=dx,
                ds=ds,
                s=s,
                mu=mu,
                theta0=theta0,
                alpha_max=alpha_max,
            )
        else:
            # fallback simple Armijo on φ
            alpha = alpha_max
            phi0 = float(
                f
                - mu * (np.sum(np.log(np.maximum(s, 1e-16))) if s.size else 0.0)
                + rho
                * (
                    (np.sum(np.abs(cE)) if cE.size else 0.0)
                    + (np.sum(np.abs(cI + s)) if (cI.size or s.size) else 0.0)
                )
            )
            dphi = float(g @ dx) + (rho * float(np.sum(np.abs(cE))) if cE.size else 0.0)
            for _ in range(int(getattr(self.cfg, "ip_ls_max", 25))):
                if alpha < float(getattr(self.cfg, "ip_alpha_min", 1e-10)):
                    break
                s_trial = s + alpha * ds if mI > 0 else s
                if mI > 0 and np.any(s_trial <= 0):
                    alpha *= float(getattr(self.cfg, "ip_alpha_backtrack", 0.5))
                    ls_iters += 1
                    continue
                x_trial = x + alpha * dx
                dt = model.eval_all(x_trial, components=["f", "cI", "cE"])
                f_t = float(dt["f"])
                cI_t = (
                    np.zeros(mI)
                    if (mI == 0 or dt["cI"] is None)
                    else np.asarray(dt["cI"], float)
                )
                cE_t = (
                    np.zeros(mE)
                    if (mE == 0 or dt["cE"] is None)
                    else np.asarray(dt["cE"], float)
                )
                phi_t = float(
                    f_t
                    - mu
                    * (
                        np.sum(np.log(np.maximum(s_trial, 1e-16)))
                        if s_trial.size
                        else 0.0
                    )
                    + rho
                    * (
                        (np.sum(np.abs(cE_t)) if cE_t.size else 0.0)
                        + (
                            np.sum(np.abs(cI_t + s_trial))
                            if (cI_t.size or s_trial.size)
                            else 0.0
                        )
                    )
                )
                if (
                    phi_t
                    <= phi0
                    + float(getattr(self.cfg, "ip_armijo_coeff", 1e-4)) * alpha * dphi
                ):
                    break
                alpha *= float(getattr(self.cfg, "ip_alpha_backtrack", 0.5))
                ls_iters += 1

        # -------------------- α == 0 : try SOC rescue, or null-accept μ --------------------
        if alpha <= 0.0:
            # Attempt SOC rescue if enabled
            if (
                use_soc
                and (soc_when == "after_ls_reject")
                and (getattr(self, "soc", None) is not None)
            ):
                # predictions for Funnel/Filter
                pred_df = max(0.0, float(-(g @ dx)))

                def theta_lin_unit():
                    cE_lin = (
                        (cE + (JE @ dx if (mE > 0 and JE is not None) else 0.0))
                        if (mE > 0)
                        else None
                    )
                    inc_I = JI @ dx if (mI > 0 and JI is not None) else 0.0
                    rI_lin = (cI + s + inc_I + ds) if (mI > 0) else None
                    thE = float(np.sum(np.abs(cE_lin))) if cE_lin is not None else 0.0
                    thI = float(np.sum(np.abs(rI_lin))) if rI_lin is not None else 0.0
                    return thE + thI

                pred_dtheta = max(0.0, theta - theta_lin_unit())

                corr, needs_rest = self.soc.compute_correction(
                    model=model,
                    x_trial=x,
                    Delta=Delta,
                    s=(s if mI > 0 else None),
                    mu=mu,
                    funnel=getattr(self, "funnel", None),
                    filter=(
                        None
                        if getattr(self, "funnel", None) is not None
                        else getattr(self, "filter", None)
                    ),
                    theta0=theta,
                    f0=f,
                    pred_df=pred_df,
                    pred_dtheta=pred_dtheta,
                )
                if (np.linalg.norm(corr) > 0.0) and (not needs_rest):
                    # Accept SOC correction as the step
                    x_new = x + corr
                    s_new = s
                    if mI > 0 and s.size > 0:
                        inc_I = (JI @ corr) if (JI is not None) else 0.0
                        s_new = s - (cI + inc_I)  # keep rI ≈ 0 around x
                        if np.any(
                            s_new <= 0.0
                        ):  # stay interior; drop slack update if needed
                            s_new = s
                    l_new, nu_new = lmb, nuv  # multipliers unchanged on rescue

                    # μ schedule after accepted move
                    if mI > 0:
                        mu_cand = float(s_new @ l_new) / max(1, mI)
                        mu = max(
                            getattr(self.cfg, "ip_mu_min", 1e-12),
                            min(mu, 0.9 * mu_cand),
                        )
                    if (it + 1) % int(getattr(self.cfg, "ip_mu_reduce_every", 10)) == 0:
                        mu = max(
                            getattr(self.cfg, "ip_mu_min", 1e-12),
                            float(getattr(self.cfg, "ip_mu_reduce", 0.2)) * mu,
                        )

                    # pack and return
                    kkt = model.kkt_residuals(x_new, l_new, nu_new)
                    conv = (
                        kkt["stat"] <= getattr(self.cfg, "tol_stat", 1e-8)
                        and kkt["ineq"] <= getattr(self.cfg, "tol_feas", 1e-8)
                        and kkt["eq"] <= getattr(self.cfg, "tol_feas", 1e-8)
                        and kkt["comp"] <= getattr(self.cfg, "tol_comp", 1e-8)
                        and mu <= getattr(self.cfg, "ip_mu_min", 1e-12)
                    )
                    theta_new = model.constraint_violation(x_new)
                    f_new = float(model.eval_all(x_new, components=["f"])["f"])
                    info = self._pack_info(
                        step_norm=float(np.linalg.norm(x_new - x)),
                        accepted=True,
                        converged=conv,
                        f=f_new,
                        theta=theta_new,
                        kkt=kkt,
                        alpha=0.0,
                        rho=0.0,
                        mu=mu,
                    )
                    info["ls_iters"] = ls_iters
                    # write back
                    st.s, st.lam, st.nu, st.mu = s_new, l_new, nu_new, mu
                    return x_new, l_new, nu_new, info

            # Null-accept path to advance μ when already small KKT
            kkt_now = model.kkt_residuals(x, lmb, nuv)
            kkt_ok = (
                kkt_now["stat"] <= getattr(self.cfg, "tol_stat", 1e-8)
                and kkt_now["ineq"] <= getattr(self.cfg, "tol_feas", 1e-8)
                and kkt_now["eq"] <= getattr(self.cfg, "tol_feas", 1e-8)
                and kkt_now["comp"] <= getattr(self.cfg, "tol_comp", 1e-8)
            )
            if allow_null_accept and kkt_ok:
                # progress μ even without a move
                if mI > 0:
                    mu_cand = float(s @ lmb) / max(1, mI)
                    mu = max(
                        getattr(self.cfg, "ip_mu_min", 1e-12), min(mu, 0.9 * mu_cand)
                    )
                if (it + 1) % int(getattr(self.cfg, "ip_mu_reduce_every", 10)) == 0:
                    mu = max(
                        getattr(self.cfg, "ip_mu_min", 1e-12),
                        float(getattr(self.cfg, "ip_mu_reduce", 0.2)) * mu,
                    )

                # check full convergence with μ
                conv = (
                    kkt_now["stat"] <= getattr(self.cfg, "tol_stat", 1e-8)
                    and kkt_now["ineq"] <= getattr(self.cfg, "tol_feas", 1e-8)
                    and kkt_now["eq"] <= getattr(self.cfg, "tol_feas", 1e-8)
                    and kkt_now["comp"] <= getattr(self.cfg, "tol_comp", 1e-8)
                    and mu <= getattr(self.cfg, "ip_mu_min", 1e-12)
                )
                info = self._pack_info(
                    step_norm=0.0,
                    accepted=True,
                    converged=conv,
                    f=float(model.eval_all(x, components=["f"])["f"]),
                    theta=model.constraint_violation(x),
                    kkt=kkt_now,
                    alpha=0.0,
                    rho=0.0,
                    mu=mu,
                )
                info["ls_iters"] = ls_iters
                st.s, st.lam, st.nu, st.mu = s, lmb, nuv, mu
                return x, lmb, nuv, info

            # Otherwise: reject as before
            info = self._pack_info(
                step_norm=0.0,
                accepted=False,
                converged=False,
                f=f,
                theta=theta,
                kkt={"stat": stat, "ineq": feasI, "eq": feasE, "comp": comp},
                alpha=0.0,
                rho=0.0,
                mu=mu,
            )
            info["ls_iters"] = ls_iters
            return x, lmb, nuv, info

        # -------------------- α > 0 : form trial; optional SOC polish --------------------
        x_trial = x + alpha * dx
        s_trial = s + alpha * ds if mI > 0 else s
        l_trial = lmb + alpha * dlam if mI > 0 else lmb
        nu_trial = nuv + alpha * dnu if mE > 0 else nuv

        if (
            use_soc
            and (soc_when == "after_ls_accept")
            and (getattr(self, "soc", None) is not None)
        ):
            # predictions based on accepted step (scale by α)
            pred_df = max(0.0, float(-(g @ dx))) * float(alpha)

            def theta_lin_scaled(v_dx, v_ds):
                cE_lin = (
                    (cE + (JE @ v_dx if (mE > 0 and JE is not None) else 0.0))
                    if (mE > 0)
                    else None
                )
                inc_I = JI @ v_dx if (mI > 0 and JI is not None) else 0.0
                rI_lin = (
                    (cI + s + inc_I + (v_ds if mI > 0 else 0.0)) if (mI > 0) else None
                )
                thE = float(np.sum(np.abs(cE_lin))) if cE_lin is not None else 0.0
                thI = float(np.sum(np.abs(rI_lin))) if rI_lin is not None else 0.0
                return thE + thI

            theta_pred = theta_lin_scaled(alpha * dx, alpha * ds if mI > 0 else 0.0)
            pred_dtheta = max(0.0, theta - theta_pred)

            corr, needs_rest = self.soc.compute_correction(
                model=model,
                x_trial=x_trial,
                Delta=Delta,
                s=(s_trial if mI > 0 else None),
                mu=mu,
                funnel=getattr(self, "funnel", None),
                filter=(
                    None
                    if getattr(self, "funnel", None) is not None
                    else getattr(self, "filter", None)
                ),
                theta0=model.constraint_violation(x_trial),
                f0=float(model.eval_all(x_trial, components=["f"])["f"]),
                pred_df=pred_df,
                pred_dtheta=pred_dtheta,
            )
            if (np.linalg.norm(corr) > 0.0) and (not needs_rest):
                x_trial = x_trial + corr
                if mI > 0 and s_trial.size > 0:
                    # keep rI ≈ 0 around pre-polish point using JI @ corr
                    inc_I = (JI @ corr) if (JI is not None) else 0.0
                    s_trial = s_trial - inc_I
                    if np.any(s_trial <= 0.0):  # keep interior
                        s_trial = s_trial + inc_I  # undo

        # -------------------- accept, update μ, check convergence --------------------
        st.s, st.lam, st.nu = s_trial, l_trial, nu_trial

        if mI > 0:
            mu_cand = float(st.s @ st.lam) / max(1, mI)
            mu = max(getattr(self.cfg, "ip_mu_min", 1e-12), min(mu, 0.9 * mu_cand))
        if (it + 1) % int(getattr(self.cfg, "ip_mu_reduce_every", 10)) == 0:
            mu = max(
                getattr(self.cfg, "ip_mu_min", 1e-12),
                float(getattr(self.cfg, "ip_mu_reduce", 0.2)) * mu,
            )

        # pack
        kkt = model.kkt_residuals(x_trial, l_trial, nu_trial)
        theta_new = model.constraint_violation(x_trial)
        f_new = float(model.eval_all(x_trial, components=["f"])["f"])
        conv = (
            kkt["stat"] <= getattr(self.cfg, "tol_stat", 1e-8)
            and kkt["ineq"] <= getattr(self.cfg, "tol_feas", 1e-8)
            and kkt["eq"] <= getattr(self.cfg, "tol_feas", 1e-8)
            and kkt["comp"] <= getattr(self.cfg, "tol_comp", 1e-8)
            and mu <= getattr(self.cfg, "ip_mu_min", 1e-12)
        )

        info = self._pack_info(
            step_norm=float(np.linalg.norm(x_trial - x)),
            accepted=True,
            converged=conv,
            f=f_new,
            theta=theta_new,
            kkt=kkt,
            alpha=alpha,
            rho=0.0,
            mu=mu,
        )
        info["ls_iters"] = ls_iters

        st.mu = mu
        return x_trial, l_trial, nu_trial, info

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
