from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:
    import numba as nb

    NUMBA_OK = True
except Exception:
    NUMBA_OK = False


MatLike = Union[np.ndarray, sp.spmatrix, spla.LinearOperator]
Vec = np.ndarray


# ------------------------------- utilities -------------------------------- #


def _as_linear_operator(H: MatLike, n: int) -> spla.LinearOperator:
    """Wrap dense/sparse/callable into a LinearOperator with shape (n,n)."""
    if isinstance(H, spla.LinearOperator):
        return H
    if callable(H):
        return spla.LinearOperator((n, n), matvec=lambda d: H(d))
    if sp.issparse(H):
        return spla.aslinearoperator(H.tocsr())
    H = np.asarray(H)
    return spla.aslinearoperator(H)


def _safe_norm(x: Vec) -> float:
    v = float(np.linalg.norm(x))
    return v if np.isfinite(v) else np.inf


def _diag_or_floor(H: MatLike, floor: float) -> Vec:
    if sp.issparse(H):
        d = H.diagonal()
    elif isinstance(H, np.ndarray):
        d = np.diag(H)
    else:
        # fall back: probe along coordinate directions (cheaper than full)
        d = np.full((H.shape[0],), floor)
    d = np.asarray(d, dtype=float)
    m = np.mean(np.abs(d)) if d.size else 1.0
    d = np.maximum(d, max(floor, 1e-16, 0.1 * m))
    return d


# ----------------------------- configuration ------------------------------- #


@dataclass
class TRDefaults:
    # Trust region
    tr_delta0: float = 0.1
    tr_delta_min: float = 1e-12
    tr_delta_max: float = 1e6
    tr_eta_lo: float = 0.1
    tr_eta_hi: float = 0.9
    tr_gamma_dec: float = 0.5
    tr_gamma_inc: float = 2.0
    tr_boundary_trigger: float = 0.8
    tr_grow_requires_boundary: bool = True
    tr_norm_type: str = "2"  # {"ellip","2","scaled"}  (no "inf" solver path)

    # CG / GLTR (Steihaug)
    cg_tol: float = 1e-8
    cg_max_iter: int = 200
    cg_restart_every: int = 50
    cg_neg_curv_tol: float = 1e-14

    # Constraints (active set)
    act_tol: float = 1e-8
    max_ws_add: int = 10

    # Metric
    chol_jitter: float = 1e-10
    diag_floor: float = 1e-12
    metric_regularization: float = 1e-8
    metric_condition_threshold: float = 1e12

    # Adaptive
    rho_smooth: float = 0.3
    pr_threshold: float = 1e-16

    # Algorithm flavors
    dfo_mode: bool = False
    curvature_aware_growth: bool = True
    feasibility_weighted_updates: bool = True


# --------------------------- trust region manager -------------------------- #


class TrustRegionManager:
    """
    Cleaned-up TR manager with:
      • LinearOperator-unified matvecs
      • preconditioned CG-Steihaug
      • robust metric building
      • rank-aware nullspace for equalities
      • simple but reliable inequality working set
    """

    def __init__(self, cfg: Optional[object] = None):
        base = TRDefaults()
        if cfg is not None:
            for k, v in vars(base).items():
                if hasattr(cfg, k):
                    setattr(base, k, getattr(cfg, k))
        self.cfg: TRDefaults = base
        self.radius: float = float(self.cfg.tr_delta0)
        self.rej: int = 0
        self.norm_type: str = str(self.cfg.tr_norm_type)

        # metric L (ellipsoidal): ||p||_TR = ||L p||_2
        self.L: Optional[np.ndarray] = None
        self.Linv: Optional[np.ndarray] = None
        self._metric_cond: Optional[float] = None
        self._H_sig: Optional[int] = None

        # histories
        self._rho_s: Optional[float] = None
        self._recent_rho: list = []
        self._recent_bdry: list = []
        self._recent_curv: list = []
        self._hist_len: int = 10

        # stats
        self._cg_stats: Dict[str, int] = {"iters": 0, "neg_curv": 0, "bdry": 0}

        # internal
        self._n_last: Optional[int] = None

    # ------------------------------ metric --------------------------------- #

    def set_metric_from_H(self, H: Optional[MatLike], force: bool = False) -> None:
        if H is None:
            self._reset_metric()
            return
        n = H.shape[0]
        if n == 0:
            self._reset_metric()
            return

        # avoid rebuilds
        H_sig = id(H) if not hasattr(H, "tobytes") else hash(H.tobytes())
        if not force and self._H_sig == H_sig and self.L is not None:
            return
        self._H_sig = H_sig

        if isinstance(H, np.ndarray):
            if self._build_chol_metric(H):
                return
        # fallback: diagonal scaling
        self._build_diag_metric(H)

    def _build_chol_metric(self, H: np.ndarray) -> bool:
        Hs = 0.5 * (H + H.T)
        n = Hs.shape[0]
        # crude cond estimate via Gershgorin
        try:
            d = np.diag(Hs)
            off = np.sum(np.abs(Hs), axis=1) - np.abs(d)
            lam_min = float(np.min(d - off))
            lam_max = float(np.max(d + off))
            cond_est = lam_max / lam_min if lam_min > 0 else np.inf
        except Exception:
            cond_est = np.inf

        base = self.cfg.chol_jitter
        reg = (
            min(1e-4, (cond_est if np.isfinite(cond_est) else 1e16) * base)
            if cond_est > self.cfg.metric_condition_threshold
            else base
        )
        I = np.eye(n)
        for _ in range(8):
            try:
                M = Hs + reg * I
                L = la.cholesky(M, lower=True, check_finite=False)
                Linv = la.solve_triangular(L, I, lower=True, check_finite=False)
                self.L, self.Linv, self._metric_cond = L, Linv, cond_est
                return True
            except la.LinAlgError:
                reg *= 10.0
        return False

    def _build_diag_metric(self, H: MatLike) -> None:
        d = _diag_or_floor(H, self.cfg.diag_floor)
        s = np.sqrt(d)
        self.L = np.diag(s)
        self.Linv = np.diag(1.0 / s)
        self._metric_cond = float(np.max(s) / max(np.min(s), 1e-32))

    def _reset_metric(self) -> None:
        self.L = None
        self.Linv = None
        self._metric_cond = None
        self._H_sig = None

    # ------------------------------- norms --------------------------------- #

    def _tr_norm(self, p: Vec) -> float:
        if self.norm_type == "ellip" and self.L is not None:
            return _safe_norm(self.L @ p)
        if self.norm_type == "scaled" and self.L is not None:
            # scaled ≈ diagonal metric only
            return _safe_norm(np.diag(self.L) * p)
        # we don’t implement an actual ∞-norm TR solver path; be explicit:
        return _safe_norm(p)

    def _project_to_boundary(self, p: Vec, Delta: float) -> Vec:
        nrm = self._tr_norm(p)
        if nrm == 0.0 or nrm <= Delta:
            return p
        return (Delta / nrm) * p

    # ------------------------ adaptive update policy ----------------------- #

    def update(
        self,
        pred_red: float,
        act_red: float,
        step_norm: float,
        theta0: Optional[float] = None,
        kkt: Optional[Dict] = None,
        act_sz: Optional[int] = None,
        H: Optional[MatLike] = None,
        step: Optional[Vec] = None,
    ) -> bool:
        cfg = self.cfg
        rho = self._robust_ratio(pred_red, act_red)
        self._push_history(rho, step_norm, H, step)

        eta_lo, eta_hi, gdec, ginc = self._tuned_thresholds()
        used_bdry = step_norm >= cfg.tr_boundary_trigger * max(self.radius, 1e-32)
        curv = (
            self._curv_along(H, step) if (H is not None and step is not None) else None
        )

        if rho < eta_lo:
            self.rej += 1
            shrink = (
                gdec
                * (0.5 if rho < -0.5 else 1.0)
                * (0.5 if (curv is not None and curv > 1e-6) else 1.0)
            )
            self.radius = max(cfg.tr_delta_min, shrink * self.radius)
            return (
                self.rej >= cfg.tr_max_rejections
                if hasattr(cfg, "tr_max_rejections")
                else False
            )

        # accept
        self.rej = 0
        if rho >= eta_hi and (used_bdry or not cfg.tr_grow_requires_boundary):
            grow = ginc
            if self.cfg.curvature_aware_growth and curv is not None:
                if curv > 1e-6:
                    grow = min(2.5, 1.25 * ginc)
                elif abs(curv) < 1e-12:
                    grow = min(1.2, ginc)
            self.radius = min(cfg.tr_delta_max, grow * self.radius)

        self.radius = float(np.clip(self.radius, cfg.tr_delta_min, cfg.tr_delta_max))
        return False

    def _robust_ratio(self, pr: float, ar: float) -> float:
        if not (np.isfinite(pr) and np.isfinite(ar)):
            return -10.0
        if abs(pr) < self.cfg.pr_threshold:
            if ar > 0:
                return 10.0
            if abs(ar) < self.cfg.pr_threshold:
                return 1.0
            return -10.0
        rho = ar / pr
        if not np.isfinite(rho):
            return -10.0 if ar < 0 else 10.0
        return float(np.clip(rho, -1e6, 1e6))

    def _tuned_thresholds(self) -> Tuple[float, float, float, float]:
        if self.cfg.dfo_mode:
            return (
                max(0.01, 0.5 * self.cfg.tr_eta_lo),
                min(0.95, 1.1 * self.cfg.tr_eta_hi),
                0.8 * self.cfg.tr_gamma_dec,
                0.8 * self.cfg.tr_gamma_inc,
            )
        return (
            self.cfg.tr_eta_lo,
            self.cfg.tr_eta_hi,
            self.cfg.tr_gamma_dec,
            self.cfg.tr_gamma_inc,
        )

    def _push_history(
        self, rho: float, step_norm: float, H: Optional[MatLike], p: Optional[Vec]
    ) -> None:
        self._recent_rho.append(rho)
        self._recent_bdry.append(step_norm / max(self.radius, 1e-32))
        if H is not None and p is not None and p.size:
            try:
                Hop = _as_linear_operator(H, p.size)
                Hp = Hop @ p
                curv = float(np.dot(p, Hp) / max(np.dot(p, p), 1e-16))
                self._recent_curv.append(curv)
            except Exception:
                pass
        for li in (self._recent_rho, self._recent_bdry, self._recent_curv):
            while len(li) > self._hist_len:
                li.pop(0)
        a = self.cfg.rho_smooth
        self._rho_s = rho if self._rho_s is None else (a * rho + (1 - a) * self._rho_s)

    def _curv_along(self, H: MatLike, p: Vec) -> Optional[float]:
        try:
            Hop = _as_linear_operator(H, p.size)
            Hp = Hop @ p
            sn2 = float(np.dot(p, p))
            return float(np.dot(p, Hp) / max(sn2, 1e-16))
        except Exception:
            return None

    # ----------------------- top-level TRSP dispatcher ---------------------- #
    def _metric_matrix_action(self, p: np.ndarray) -> np.ndarray:
        """Return M p where M is the TR metric in stationarity:
        Euclidean: M = I; ellip: M = L^T L; scaled: diag(L)^2."""
        if self.norm_type == "ellip" and self.L is not None:
            # (L^T L) p
            return self.L.T @ (self.L @ p)
        if self.norm_type == "scaled" and self.L is not None:
            d = np.diag(self.L)
            return (d * d) * p
        return p  # Euclidean

    def _estimate_sigma(self, H, g, p: np.ndarray) -> float:
        """Estimate TR multiplier σ from (H + σ M) p + g ≈ 0 ⇒ σ = - pᵀ(Hp+g) / pᵀ(Mp)."""
        n = p.size
        Hop = _as_linear_operator(H, n)
        Hp = Hop @ p
        Mp = self._metric_matrix_action(p)
        num = float(p @ (Hp + g))
        den = float(p @ Mp)
        if den <= 1e-32:
            return 0.0
        sigma = -num / den
        # σ must be ≥ 0 if boundary is active; small negatives are numerical noise
        return float(max(0.0, sigma))

    def _recover_multipliers(
        self,
        H,
        g,
        p: np.ndarray,
        sigma: float,
        JE: Optional[np.ndarray],
        JI: Optional[np.ndarray],
        active_idx: Optional[list],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Recover (nu, lam) by solving JE^T nu + JI_A^T lam_A = -r, r := H p + g + σ M p.
        Always returns arrays:
        - lam: shape (mI,), zeros for inactive or if JI is None
        - nu : shape (mE,), zeros if JE is None
        """
        n = p.size
        Hop = _as_linear_operator(H, n)
        r = Hop @ p + g + sigma * self._metric_matrix_action(p)

        mE = 0 if JE is None else int(JE.shape[0])
        mI = 0 if JI is None else int(JI.shape[0])

        # default: zero arrays
        nu = np.zeros(mE, dtype=float)
        lam_full = np.zeros(mI, dtype=float)

        # assemble A^T y = -r with y = [nu; lam_A]
        blocks = []
        if mE > 0:
            blocks.append(JE.T)
        A_active = None
        if (mI > 0) and active_idx:
            idx = np.asarray(active_idx, dtype=int)
            A_active = JI[idx]
            blocks.append(A_active.T)

        if not blocks:
            # unconstrained: keep zeros
            return lam_full, nu

        AT = np.concatenate(blocks, axis=1)  # n × (mE + mA)
        y, *_ = la.lstsq(AT, -r, rcond=1e-12)
        # split into nu, lam_A
        if mE > 0:
            nu = y[:mE]
            y = y[mE:]
        if A_active is not None:
            lam_A = np.maximum(0.0, y)  # KKT: λ ≥ 0
            # refine ν after clipping λ: JE^T ν = -r - JI_A^T λ_A
            if mE > 0:
                rhs = -r - A_active.T @ lam_A
                nu, *_ = la.lstsq(JE.T, rhs, rcond=1e-12)
            lam_full[np.asarray(active_idx, dtype=int)] = lam_A

        return lam_full, nu

    def solve(
        self,
        H: MatLike,
        g: Vec,
        JI: Optional[np.ndarray] = None,
        cI: Optional[Vec] = None,
        JE: Optional[np.ndarray] = None,
        cE: Optional[Vec] = None,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        act_tol: Optional[float] = None,
        max_ws_add: Optional[int] = None,
    ) -> Tuple[Vec, float, Dict[str, Any]]:
        cfg = self.cfg
        n = int(g.size)
        self._n_last = n
        Delta = float(self.radius)

        if self.norm_type == "ellip":
            self.set_metric_from_H(H)

        tol = cfg.cg_tol if tol is None else float(tol)
        max_iter = cfg.cg_max_iter if max_iter is None else int(max_iter)
        act_tol = cfg.act_tol if act_tol is None else float(act_tol)
        max_ws_add = cfg.max_ws_add if max_ws_add is None else int(max_ws_add)

        g = np.asarray(g, dtype=float)
        JE = (
            None
            if (JE is None or getattr(JE, "size", 0) == 0)
            else np.asarray(JE, dtype=float)
        )
        cE = (
            None
            if (cE is None or getattr(cE, "size", 0) == 0)
            else np.asarray(cE, dtype=float).ravel()
        )
        JI = (
            None
            if (JI is None or getattr(JI, "size", 0) == 0)
            else np.asarray(JI, dtype=float)
        )
        cI = (
            None
            if (cI is None or getattr(cI, "size", 0) == 0)
            else np.asarray(cI, dtype=float).ravel()
        )

        if JE is not None:
            return self._solve_eq_then_ineq(
                H, g, JE, cE, JI, cI, Delta, tol, max_iter, act_tol, max_ws_add
            )
        elif JI is not None:
            return self._solve_ineq_only(
                H, g, JI, cI, Delta, tol, max_iter, act_tol, max_ws_add
            )
        else:
            p, p_norm, info = self._solve_unconstrained(H, g, Delta, tol, max_iter)
            sigma = self._estimate_sigma(H, g, p)
            info["tr_multiplier_sigma"] = sigma
            lam, nu = self._recover_multipliers(H, g, p, sigma, None, None, None)
            return p, p_norm, info, lam, nu

    # ------------------------ unconstrained subproblem ---------------------- #

    def _solve_unconstrained(
        self, H: MatLike, g: Vec, Delta: float, tol: float, max_iter: int
    ):
        n = g.size
        Hop = _as_linear_operator(H, n)

        # Preconditioner: Jacobi (diag of H) or metric-based diag
        if self.norm_type == "ellip" and self.L is not None:
            Mdiag = np.diag(self.L) ** 2
        else:
            Mdiag = _diag_or_floor(H, self.cfg.diag_floor)
        Minv = 1.0 / np.maximum(Mdiag, 1e-16)

        if self.norm_type == "ellip" and self.L is not None:
            # transform to y = L p, ||y||_2 ≤ Δ
            W = self.Linv
            WT = W.T

            def Ht(d):
                return WT @ (Hop @ (W @ d))

            gtil = WT @ g
            y, info = _pcg_steihaug(
                Ht, gtil, Delta, tol, max_iter, Minv=None
            )  # Minv in y-space often unhelpful
            p = W @ y
        else:
            p, info = _pcg_steihaug(Hop, g, Delta, tol, max_iter, Minv=Minv)

        p = self._project_to_boundary(p, Delta)
        return p, self._tr_norm(p), info

    # ------------------- equalities (nullspace reduction) ------------------- #
    def _solve_eq_then_ineq(
        self,
        H,
        g,
        JE,
        cE,
        JI,
        cI,
        Delta,
        tol,
        max_iter,
        act_tol,
        max_ws_add,
    ):
        """
        Equality handling via particular solution + nullspace reduction, then
        (optionally) inequality polishing/working-set. Returns:
        p, ||p||_TR, info, lam (full-sized), nu
        """
        n = g.size
        info: Dict[str, Any] = {}

        # 1) Particular solution to JE p = -cE (rank-aware)
        p_part = _min_norm_particular(JE, -cE)

        # 2) Nullspace basis (orthonormal)
        Z = _nullspace_basis(JE)

        if Z.shape[1] == 0:
            # No freedom: p = p_part
            p = p_part
            p = self._project_to_boundary(p, Delta)
            p_norm = self._tr_norm(p)

            active_idx = None
            if JI is not None and JI.size > 0:
                viol = JI @ p + cI
                if np.any(viol > act_tol):
                    info["status"] = "equality_infeasible"
                else:
                    info["status"] = "equality_constrained"
            else:
                info["status"] = "equality_constrained"

            # multipliers
            sigma = self._estimate_sigma(H, g, p)
            info["tr_multiplier_sigma"] = sigma
            lam, nu = self._recover_multipliers(H, g, p, sigma, JE, JI, active_idx)
            return p, p_norm, info, lam, nu

        # 3) Reduced TR in z (Euclidean TR on p = p_part + Z z is fine)
        Hop = _as_linear_operator(H, n)

        Hz = spla.LinearOperator(
            (Z.shape[1], Z.shape[1]),
            matvec=lambda z: Z.T @ (Hop @ (Z @ z)),
        )
        gz = Z.T @ (g + Hop @ p_part)

        z, cg_info = _pcg_steihaug(Hz, gz, Delta, tol, max_iter, Minv=None)
        p = p_part + Z @ z
        p = self._project_to_boundary(p, Delta)
        p_norm = self._tr_norm(p)
        info.update({"status": "nullspace_solved", **cg_info})

        # 4) Inequalities (optional): light projection / WS loop
        active_idx: Optional[list] = None
        if JI is not None and JI.size > 0:
            # mild projection first
            v = JI @ p + cI
            if np.any(v > act_tol):
                # local projection for small violations
                if float(np.max(v)) < 10 * act_tol:
                    pp = p.copy()
                    for _ in range(10):
                        vv = JI @ pp + cI
                        if not np.any(vv > act_tol):
                            break
                        k = int(np.argmax(vv))
                        jk = JI[k]; vk = float(vv[k])
                        s = float(jk @ jk)
                        if s > 1e-12:
                            pp = pp - (vk / s) * jk
                    pp = self._project_to_boundary(pp, Delta)
                    if not np.any(JI @ pp + cI > act_tol):
                        p = pp
                        p_norm = self._tr_norm(p)
                        info["status"] = "projected"

                # if still violated → working-set loop
                v = JI @ p + cI
                if np.any(v > act_tol):
                    p_ws, _, ws_info, lam_ws, nu_ws = self._solve_ineq_only(
                        H, g, JI, cI, Delta, tol, max_iter, act_tol, max_ws_add
                    )
                    # adopt WS result (nu_ws is only w.r.t. equalities in WS path; here JE also present)
                    p, p_norm = p_ws, self._tr_norm(p_ws)
                    info.update({k: v for k, v in ws_info.items() if k != "tr_multiplier_sigma"})
                    active_idx = ws_info.get("active_set_indices", None)

        # 5) Multipliers (sigma, lam, nu)
        sigma = self._estimate_sigma(H, g, p)
        info["tr_multiplier_sigma"] = sigma
        lam, nu = self._recover_multipliers(H, g, p, sigma, JE, JI, active_idx)
        return p, p_norm, info, lam, nu

    # ------------------------- inequalities (WS) ---------------------------- #
    def _solve_ineq_only(
        self,
        H,
        g,
        JI,
        cI,
        Delta,
        tol,
        max_iter,
        act_tol,
        max_ws_add,
    ):
        """
        Inequalities via a simple working-set loop:
        - promote worst violator to equalities,
        - solve equality-constrained TR,
        - iterate until feasible or max_ws_add.
        Returns:
        p, ||p||_TR, info, lam (full-sized), nu (None: no JE here)
        """
        active: set[int] = set()
        it = 0
        info: Dict[str, Any] = {}
        p = np.zeros_like(g)

        while it <= max_ws_add:
            if active:
                idx = sorted(active)
                JEa, cEa = JI[idx], cI[idx]
            else:
                JEa, cEa = None, None

            if JEa is None:
                # unconstrained solve
                p_uc, _, cg_info = self._solve_unconstrained(H, g, Delta, tol, max_iter)
                p = p_uc
                info.update({k: v for k, v in cg_info.items()})
            else:
                # equality solve on active set only (no additional ineqs here)
                p_eq, _, eq_info = self._solve_eq_then_ineq(
                    H, g, JEa, cEa, None, None, Delta, tol, max_iter, act_tol, 0
                )
                # _solve_eq_then_ineq returns 5-tuple; unpack just the p and info
                if isinstance(p_eq, tuple) and len(p_eq) == 5:
                    p, _, eqi, _, _ = p_eq
                    info.update({k: v for k, v in eqi.items()})
                else:
                    p = p_eq
                    info.update({k: v for k, v in eq_info.items()})

            v = JI @ p + cI
            violated = v > act_tol
            if not np.any(violated):
                break

            worst = int(np.argmax(v))
            if worst in active:   # stagnation guard
                break
            active.add(worst)
            it += 1

        info.update({
            "active_set_size": len(active),
            "active_set_iterations": it,
            "active_set_indices": sorted(active),
            "status": info.get("status", "feasible") if not np.any(JI @ p + cI > act_tol) else "infeasible_ws",
        })

        # multipliers for ineq-only case (nu=None, JE=None)
        sigma = self._estimate_sigma(H, g, p)
        info["tr_multiplier_sigma"] = sigma
        lam, nu = self._recover_multipliers(H, g, p, sigma, JE=None, JI=JI, active_idx=sorted(active) if active else None)
        return p, self._tr_norm(p), info, lam, nu

    def _ineq_polish(self, p, H, g, JI, cI, Delta, act_tol, max_ws_add):
        v = JI @ p + cI
        if not np.any(v > act_tol):
            return p, self._tr_norm(p), {"status": "feasible"}
        # local projection for mild violations
        if float(np.max(v)) < 10 * act_tol:
            pp = p.copy()
            for _ in range(10):
                v = JI @ pp + cI
                if not np.any(v > act_tol):
                    break
                k = int(np.argmax(v))
                jk = JI[k]
                vk = float(v[k])
                s = float(jk @ jk)
                if s > 1e-12:
                    pp = pp - (vk / s) * jk
            pp = self._project_to_boundary(pp, Delta)
            if not np.any(JI @ pp + cI > act_tol):
                return pp, self._tr_norm(pp), {"status": "projected"}
        # fall back to WS loop
        return self._solve_ineq_only(
            H, g, JI, cI, Delta, 1e-8, 100, act_tol, max_ws_add
        )

    # ------------------------------ public API ------------------------------ #

    def get_radius(self) -> float:
        return float(self.radius)

    def set_radius(self, r: float) -> None:
        self.radius = float(np.clip(r, self.cfg.tr_delta_min, self.cfg.tr_delta_max))

    def reset(self) -> None:
        self.radius = self.cfg.tr_delta0
        self.rej = 0
        self._rho_s = None
        self._recent_rho.clear()
        self._recent_bdry.clear()
        self._recent_curv.clear()
        self._reset_metric()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "radius": self.radius,
            "rejections": self.rej,
            "smoothed_ratio": self._rho_s,
            "cg_stats": dict(self._cg_stats),
            "metric_condition": self._metric_cond,
            "recent": {
                "avg_rho": (np.mean(self._recent_rho) if self._recent_rho else None),
                "avg_bdry": (np.mean(self._recent_bdry) if self._recent_bdry else None),
                "avg_curv": (np.mean(self._recent_curv) if self._recent_curv else None),
            },
        }

    def configure_for_algorithm(self, algorithm: str):
        alg = algorithm.lower()
        if alg == "dfo":
            self.cfg.dfo_mode = True
            self.cfg.tr_eta_lo = 0.01
            self.cfg.tr_eta_hi = 0.85
            self.cfg.tr_gamma_dec = 0.25
            self.cfg.tr_gamma_inc = 1.3
            self.cfg.rho_smooth = 0.1
        elif alg == "sqp":
            self.cfg.dfo_mode = False
            self.cfg.tr_eta_lo = 0.1
            self.cfg.tr_eta_hi = 0.9
            self.cfg.tr_gamma_dec = 0.5
            self.cfg.tr_gamma_inc = 2.0
            self.cfg.rho_smooth = 0.3


# ----------------------- PCG-Steihaug (TR, Euclidean) --------------------- #


def _pcg_steihaug(
    H: Union[MatLike, Callable[[Vec], Vec]],
    g: Vec,
    Delta: float,
    tol: float,
    max_iter: int,
    Minv: Optional[Vec],
) -> Tuple[Vec, Dict[str, Any]]:
    """
    Preconditioned CG-Steihaug for ½ pᵀ H p + gᵀ p subject to ||p||₂ ≤ Δ.

    H : LinearOperator or callable matvec
    Minv : diagonal of preconditioner inverse (Jacobi). If None, no precond.
    """
    n = g.size
    Hop = _as_linear_operator(H, n)
    p = np.zeros(n, dtype=float)
    r = -g.copy()
    z = r if Minv is None else (Minv * r)
    d = z.copy()
    rz = float(np.dot(r, z))

    if np.sqrt(max(rz, 0.0)) < tol:
        return p, {"status": "initial_convergence", "iterations": 0}

    for k in range(1, max_iter + 1):
        Hd = Hop @ d
        dHd = float(np.dot(d, Hd))
        dd = float(np.dot(d, d))

        # negative curvature → go to boundary
        if dHd <= 1e-14 * max(1.0, dd):
            tau = _tau_to_boundary(p, d, Delta)
            return p + tau * d, {"status": "negative_curvature", "iterations": k}

        alpha = rz / max(dHd, 1e-32)
        p_trial = p + alpha * d

        if np.linalg.norm(p_trial) >= Delta * (1 + 1e-12):
            tau = _tau_to_boundary(p, d, Delta)
            return p + tau * d, {"status": "boundary_hit", "iterations": k}

        r_new = r - alpha * Hd
        if Minv is None:
            z_new = r_new
        else:
            z_new = Minv * r_new
        rz_new = float(np.dot(r_new, z_new))
        if np.sqrt(max(rz_new, 0.0)) < tol:
            return p_trial, {"status": "converged", "iterations": k}

        beta = rz_new / max(rz, 1e-32)
        d = z_new + beta * d
        p, r, z, rz = p_trial, r_new, z_new, rz_new

    return p, {"status": "max_iterations", "iterations": max_iter}


def _tau_to_boundary(p: Vec, d: Vec, Delta: float) -> float:
    pp = float(np.dot(p, p))
    pd = float(np.dot(p, d))
    dd = float(np.dot(d, d))
    if dd <= 1e-32:
        return 0.0
    c = pp - Delta * Delta
    disc = max(0.0, pd * pd - dd * c)
    root = np.sqrt(disc)
    t1 = (-pd + root) / dd
    t2 = (-pd - root) / dd
    return max(t1, t2, 0.0)


# ----------------------- equalities helpers (rank-aware) ------------------- #


def _min_norm_particular(A: np.ndarray, b: Vec, rcond: float = 1e-12) -> Vec:
    """
    Solve min ||p||_2 s.t. A p = b (or least-squares if infeasible).
    Uses SVD for rank-awareness, with QR fallback.
    """
    try:
        U, s, VT = la.svd(A, full_matrices=False, check_finite=False)
        mask = s > rcond * (s[0] if s.size else 1.0)
        if not np.any(mask):
            return np.zeros(A.shape[1])
        s_inv = np.zeros_like(s)
        s_inv[mask] = 1.0 / s[mask]
        return VT.T @ (s_inv * (U.T @ b))
    except la.LinAlgError:
        return la.lstsq(A, b, rcond=rcond)[0]


def _nullspace_basis(A: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """
    Orthonormal basis for null(A) using SVD (robust rank revelation).
    """
    m, n = A.shape
    if m == 0 or n == 0:
        return np.eye(n)
    try:
        U, s, VT = la.svd(A, full_matrices=False, check_finite=False)
        rank = int(np.sum(s > rcond * (s[0] if s.size else 1.0)))
        if rank >= n:
            return np.zeros((n, 0))
        N = VT[rank:].T  # columns span nullspace
        # Orthonormalize (VT rows already orthonormal; but be safe)
        Q, _ = la.qr(N, mode="economic", pivoting=False)
        return Q
    except la.LinAlgError:
        # QR with pivoting fallback
        Q, R, P = la.qr(A.T, mode="economic", pivoting=True)
        rank = int(np.sum(np.abs(np.diag(R)) > rcond))
        N = Q[:, rank:]
        return N
