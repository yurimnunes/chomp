# --- kkt_core.py (put this near your linear algebra helpers) ---
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .blocks.reg import Regularizer, make_preconditioner_only, make_psd_advanced
from .ip_cg import *

try:
    import qdldl_cpp as qd

    _HAS_QDLDL = True
except Exception:
    qd = None
    _HAS_QDLDL = False


def _csr(A, shape=None):
    if sp.isspmatrix_csr(A):
        return A
    return A.tocsr() if shape is None else sp.csr_matrix(A, shape=shape)


def _upper_csc(K: sp.spmatrix) -> sp.csc_matrix:
    # 1) keep only upper triangle (incl. diagonal)
    coo = K.tocoo()
    mask = coo.row <= coo.col
    Kup = sp.csc_matrix(
        (coo.data[mask], (coo.row[mask], coo.col[mask])), shape=coo.shape
    )

    # 2) coalesce and sort
    Kup.sum_duplicates()
    Kup.sort_indices()

    # 3) dtypes qdldl likes: float64 + int32
    if Kup.dtype != np.float64:
        Kup = Kup.astype(np.float64)
    if Kup.indptr.dtype != np.int32:
        Kup.indptr = Kup.indptr.astype(np.int32, copy=False)
    if Kup.indices.dtype != np.int32:
        Kup.indices = Kup.indices.astype(np.int32, copy=False)

    # 4) ensure every diagonal exists and is finite (tiny ridge if needed)
    d = Kup.diagonal()  # returns 1D ndarray
    bad = (~np.isfinite(d)) | (np.abs(d) < 1e-300)
    if np.any(bad):
        eps = 1e-12
        Kup = Kup + sp.diags(np.where(bad, eps, 0.0), format="csc")

    # (Optional) sanity check: no lower entries remain
    # assert sp.tril(Kup, -1).nnz == 0

    return Kup


def _normest_rowsum_inf(A: sp.spmatrix) -> float:
    A = _csr(A)
    return float(np.max(np.abs(A).sum(axis=1))) if A.shape[0] else 1.0


def _cg_matfree(matvec, b, x0=None, tol=1e-10, maxit=200, M=None):
    n = b.size
    x = np.zeros(n) if x0 is None else x0.copy()
    r = b - matvec(x)
    z = r if M is None else M(r)
    p = z.copy()
    rz_old = float(r @ z)
    nrm0 = float(np.linalg.norm(r))
    stop = max(tol * nrm0, 0.0)
    if nrm0 <= stop:
        return x, 0
    for it in range(1, maxit + 1):
        Ap = matvec(p)
        pAp = float(p @ Ap)
        if pAp <= 0.0:
            break
        alpha = rz_old / pAp
        x += alpha * p
        r -= alpha * Ap
        nr = float(np.linalg.norm(r))
        if nr <= stop:
            return x, it
        z = r if M is None else M(r)
        rz_new = float(r @ z)
        beta = rz_new / rz_old
        rz_old = rz_new
        p = z + beta * p
    return x, it


# ---------------------- Reusable handle interface ----------------------
class KKTReusable(Protocol):
    def solve(
        self,
        r1: np.ndarray,
        r2: Optional[np.ndarray],
        cg_tol: float = 1e-8,
        cg_maxit: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray]: ...


# ---------------------- Config + Cache ----------------------
@dataclass
class KKTCache:
    # method-agnostic things we may reuse between calls
    prev_dx: Optional[np.ndarray] = None
    prev_dy: Optional[np.ndarray] = None
    delta_w_last: float = 0.0
    # strategy-specific caches if needed
    strategy_state: Dict[str, Any] = None


# ---------------------- Abstract strategy ----------------------
class KKTStrategy:
    name: str

    def factor_and_solve(
        self,
        W: sp.spmatrix,  # n×n
        G: Optional[sp.spmatrix],  # m×n or None
        r1: np.ndarray,
        r2: Optional[np.ndarray],
        cfg: SQPConfig,
        cache: KKTCache,
        *,
        refine_iters: int = 0,
        use_ordering: bool = False,
        cg_tol: float = 1e-10,
        cg_maxit: int = 200,
        gamma: Optional[float] = None,
        delta_c_lift: Optional[float] = None,
        reuse_symbolic: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[KKTReusable]]:
        raise NotImplementedError


# ---------------------- HYKKT ----------------------
class HYKKTStrategy(KKTStrategy):
    name = "hykkt"

    def __init__(self, cholmod_loader: Optional[Callable[[], Any]] = None):
        # late import hook (so environments w/o scikit-sparse still work)
        self._cholmod_loader = cholmod_loader

    def factor_and_solve(self, W, G, r1, r2, cfg, reg, cache, **kw):
        """
        HYKKT Schur-complement solve for equality-constrained KKT:
            K_gamma = W + δI + γ GᵀG
            S(y)    = G K_gamma^{-1} Gᵀ y
            rhs_s   = G K_gamma^{-1}(r1 + γ Gᵀ r2) - r2
            dy      = S^{-1} rhs_s
            dx      = K_gamma^{-1}(r1 + γ Gᵀ r2 - Gᵀ dy)

        Returns
        -------
        dx, dy, reusable
            reusable.solve(r1n, r2n, cg_tol, cg_maxit) -> (dxn, dyn)
        """
        assert G is not None and r2 is not None, "HYKKT requires equality constraints"

        # --- kwargs / cache ---
        cg_tol = kw.get("cg_tol", 1e-10)
        cg_maxit = kw.get("cg_maxit", 200)
        gamma = kw.get("gamma")
        delta_w_last = float(cache.delta_w_last or 0.0)

        # --- sizes ---
        nW = int(W.shape[0])
        m, p = G.shape  # p should equal nW

        # --- gamma heuristic if not provided ---
        if gamma is None:
            num = _normest_rowsum_inf(W) + delta_w_last
            den = max(_normest_rowsum_inf(G), 1.0)
            gamma = max(1.0, num / (den * den))

        # --- K_gamma assembly ---
        Kgam = (
            W + delta_w_last * sp.eye(nW, format="csr") + gamma * (G.T @ G).tocsr()
        ).tocsr()

        # --- select inner solver for K_gamma ---
        used_inner = None
        try:
            # Prefer CHOLMOD if available (fast SPD)
            if self._cholmod_loader:
                cholmod = self._cholmod_loader()
            else:
                import sksparse.cholmod as cholmod
            chol = cholmod.cholesky(Kgam.tocsc())
            inner_solve = lambda q: chol.solve_A(q)
            used_inner = "cholmod"
        except Exception:
            # Try QDLDL (LDLᵀ) next
            try:
                import qdldl as qd  # adjust import name if your binding differs

                K_up = _upper_csc(Kgam.tocsc())  # make upper-CSC view
                fac = qd.factorize(
                    K_up.indptr, K_up.indices, K_up.data, K_up.shape[0], perm=None
                )
                inner_solve = lambda q: qd.solve(fac, q)
                used_inner = "qdldl"
            except Exception:
                # PCG fallback on K_gamma (diagonal preconditioner)
                A = Kgam  # CSR
                indptr, indices, data = A.indptr, A.indices, A.data
                diag = np.maximum(A.diagonal().astype(float), 1e-12)

                def _pcg(q):
                    x0 = np.zeros_like(q)
                    x, _, _ = pcg_csr(
                        indptr,
                        indices,
                        data,
                        q,
                        x0,
                        diag,
                        tol=max(1e-2 * cg_tol, 1e-12),
                        maxit=max(10 * cg_maxit, 200),
                    )
                    return x

                inner_solve = _pcg
                used_inner = "pcg"

        # --- Schur matvec (operator): S(y) = G K_gamma^{-1} Gᵀ y ---
        def S_mv(y: np.ndarray) -> np.ndarray:
            return G @ inner_solve(G.T @ y)

        # --- Build diagonal A^{-1} used to assemble Ŝ for preconditioning ---
        #     A_diag ≈ diag(W) + δ + γ * diag(GᵀG)
        diagW = np.array(W.diagonal(), dtype=float)
        diagGtG = np.array((G.multiply(G)).sum(axis=0)).ravel()  # length p
        A_diag = np.maximum(diagW + delta_w_last + gamma * diagGtG, 1e-12)
        Ainv = sp.diags(1.0 / A_diag)

        # --- Schur approximation Ŝ = G A^{-1} Gᵀ (m x m, square) ---
        S_hat = (G @ Ainv @ G.T).tocsr()

        # --- Preconditioner for Ŝ (LinearOperator in original coordinates) ---
        _, _, S_M = make_preconditioner_only(S_hat, reg)

        # --- Right-hand side for Schur system ---
        svec = r1 + gamma * (G.T @ r2)
        rhs_s = (G @ inner_solve(svec)) - r2

        # --- Solve Schur system with CG (matrix-free), preconditioned ---
        x0 = cache.prev_dy if cache.prev_dy is not None else None

        # If your CG expects a callable, use: M_arg = S_M.matvec
        M_arg = S_M  # most CG implementations accept LinearOperator directly
        dy, _ = _cg_matfree(S_mv, rhs_s, x0=x0, tol=cg_tol, maxit=cg_maxit, M=M_arg)

        # --- Back-substitute for dx ---
        dx = inner_solve(svec - (G.T @ dy))

        # --- reusable closure ---
        class _Reusable(KKTReusable):
            def __init__(self, G, inner_solve, gamma, S_M):
                self.G = G
                self.inner_solve = inner_solve
                self.gamma = float(gamma)
                self.S_M = S_M  # LinearOperator

            def solve(self, r1n, r2n, cg_tol=1e-8, cg_maxit=200):
                svec_n = r1n + self.gamma * (self.G.T @ r2n)
                rhs_s_n = (self.G @ self.inner_solve(svec_n)) - r2n

                def S_mv_n(y):
                    return self.G @ self.inner_solve(self.G.T @ y)

                M_arg = self.S_M
                dyn, _ = _cg_matfree(
                    S_mv_n, rhs_s_n, x0=None, tol=cg_tol, maxit=cg_maxit, M=M_arg
                )
                dxn = self.inner_solve(svec_n - (self.G.T @ dyn))
                return dxn, dyn

        cache.prev_dx, cache.prev_dy = dx, dy
        return dx, dy, _Reusable(G, inner_solve, gamma, S_M)


# ---------------------- Lifted SPD ----------------------
class LiftedStrategy(KKTStrategy):
    name = "lifted"

    def factor_and_solve(self, W, G, r1, r2, cfg, cache, **kw):
        assert G is not None and r2 is not None, "Lifted requires equality constraints"
        delta_c_lift = kw.get("delta_c_lift")
        n = W.shape[0]
        delta_w_last = float(cache.delta_w_last or 0.0)

        if delta_c_lift is None:
            num = _normest_rowsum_inf(W) + delta_w_last
            den = max(_normest_rowsum_inf(G), 1.0)
            gamma_hat = max(1.0, num / (den * den))
            delta_c_lift = 1.0 / gamma_hat

        Kspd = (
            W
            + delta_w_last * sp.eye(n, format="csr")
            + (1.0 / delta_c_lift) * (G.T @ G)
        )
        solveA = None
        try:
            if globals().get("_HAS_LLDL", False):
                fac = lldl_factorize(Kspd.tocsc(), memory_limit=100)
                solveA = fac.solve
            else:
                solveA = spla.factorized(Kspd.tocsc())
        except Exception:
            pass

        if solveA is None:
            dx = spla.spsolve(Kspd.tocsc(), r1 - (1.0 / delta_c_lift) * (G.T @ r2))
        else:
            dx = solveA(r1 - (1.0 / delta_c_lift) * (G.T @ r2))
        dy = (G @ dx - r2) * (1.0 / delta_c_lift)

        class _Reusable(KKTReusable):
            def __init__(self, G, delta_c_lift, Kspd, solveA):
                self.G, self.delta_c_lift, self.Kspd, self.solveA = (
                    G,
                    delta_c_lift,
                    Kspd,
                    solveA,
                )

            def solve(self, r1n, r2n, **_):
                if self.solveA is None:
                    dxn = spla.spsolve(
                        self.Kspd.tocsc(),
                        r1n - (1.0 / self.delta_c_lift) * (self.G.T @ r2n),
                    )
                else:
                    dxn = self.solveA(
                        r1n - (1.0 / self.delta_c_lift) * (self.G.T @ r2n)
                    )
                dyn = (self.G @ dxn - r2n) * (1.0 / self.delta_c_lift)
                return dxn, dyn

        cache.prev_dx, cache.prev_dy = dx, dy
        return dx, dy, _Reusable(G, delta_c_lift, Kspd, solveA)


# ---------------------- Indefinite LDLᵀ ----------------------
class LDLStrategy(KKTStrategy):
    name = "ldl"  # covers 'qdldl' and 'lldl' choices

    def __init__(self, prefer_lldl: bool = False):
        self.prefer_lldl = prefer_lldl

    def factor_and_solve(self, W, G, r1, r2, cfg, regularizer, cache, **kw):
        refine_iters = kw.get("refine_iters", 0)
        method_req = kw.get("method", "qdldl")
        n = W.shape[0]
        mE = 0 if G is None else G.shape[0]

        delta_w_last = float(cache.delta_w_last or 0.0)
        delta_w = 0.0
        delta_c = 0.0
        max_attempts = 10

        def _build_K():
            if mE > 0:
                Wcsr = _csr(W + delta_w * sp.eye(n))
                B22 = -delta_c * sp.eye(mE) if delta_c > 0 else sp.csr_matrix((mE, mE))
                K = sp.vstack(
                    [sp.hstack([Wcsr, G.T]), sp.hstack([G, B22])], format="csc"
                )
                rhs = np.concatenate([r1, r2])
            else:
                K = _csr(W + delta_w * sp.eye(n))
                rhs = r1
            return K, rhs

        for attempt in range(max_attempts):
            K, rhs = _build_K()
            K, _ = make_psd_advanced(K, regularizer, attempt)
            K_up = _upper_csc(K)
            nsys = K_up.shape[0]
            perm = None  # (plug AMD if you want)

            try:
                used = None
                if (self.prefer_lldl or method_req == "lldl") and globals().get(
                    "_HAS_LLDL", False
                ):
                    fac = lldl_factorize(K_up, perm=perm)
                    D = fac.D
                    used = "lldl"

                    def do_solve(b):
                        return fac.solve(b)

                else:
                    if not globals().get("_HAS_QDLDL", False):
                        raise RuntimeError("qdldl unavailable")
                    fac = qd.factorize(
                        K_up.indptr, K_up.indices, K_up.data, nsys, perm=perm
                    )
                    D = fac.D
                    used = "qdldl"

                    def do_solve(b):
                        return (
                            qd.solve_refine(fac, b, refine_iters)
                            if refine_iters > 0
                            else qd.solve(fac, b)
                        )

                num_pos = int(np.sum(D > 0))
                num_neg = int(np.sum(D < 0))
                num_zero = int(np.sum(np.abs(D) < 1e-12))

                if mE > 0:
                    ok = num_pos == n and num_neg == mE and num_zero == 0
                else:
                    ok = num_pos == n and num_neg == 0 and num_zero == 0

                if ok:
                    sol = do_solve(rhs)
                    if mE > 0:
                        dx, dy = sol[:n], sol[n:]
                    else:
                        dx, dy = sol, np.zeros(0)

                    class _Reusable(KKTReusable):
                        def __init__(self, fac, used, n, mE):
                            self.fac, self.used, self.n, self.mE = fac, used, n, mE

                        def solve(self, r1n, r2n=None, **_):
                            if self.used == "qdldl":
                                rhsn = (
                                    r1n if self.mE == 0 else np.concatenate([r1n, r2n])
                                )
                                soln = qd.solve(self.fac, rhsn)
                            else:
                                rhsn = (
                                    r1n if self.mE == 0 else np.concatenate([r1n, r2n])
                                )
                                soln = self.fac.solve(rhsn)
                            return (
                                (soln[: self.n], soln[self.n :])
                                if self.mE > 0
                                else (soln, np.zeros(0))
                            )

                    cache.delta_w_last = delta_w
                    cache.prev_dx, cache.prev_dy = dx, dy
                    return dx, dy, _Reusable(fac, used, n, mE)

                # adjust regs
                if num_zero > 0 or (mE > 0 and num_neg != mE):
                    delta_c = max(delta_c, 1e-8)
                    if delta_w_last == 0:
                        delta_w = (
                            cfg.ip_hess_reg0
                            if attempt == 0
                            else 8.0 * max(cfg.ip_hess_reg0, delta_w)
                        )
                    else:
                        delta_w = (
                            max(1e-20, 1 / 3 * delta_w_last)
                            if attempt == 0
                            else 8.0 * max(1e-20, delta_w)
                        )
                    if delta_w > 1e40:
                        raise ValueError("Inertia correction failed")
            except Exception:
                # Sparse fallback
                sol = spla.spsolve(K.tocsc(), rhs)
                if mE > 0:
                    dx, dy = sol[:n], sol[n:]
                else:
                    dx, dy = sol, np.zeros(0)
                return dx, dy, None

        # give up
        return np.zeros(n), np.zeros(mE), None


# ---------------------- Registry and dispatcher ----------------------
class KKTSolverRegistry:
    def __init__(self):
        self._map: Dict[str, KKTStrategy] = {}

    def register(self, strategy: KKTStrategy):
        self._map[strategy.name] = strategy

    def get(self, name: str) -> KKTStrategy:
        if name not in self._map:
            raise KeyError(f"Unknown KKT method '{name}'")
        return self._map[name]


# Build a default registry you can import and reuse
def _load_cholmod():
    import sksparse.cholmod as cholmod

    return cholmod


DEFAULT_KKT_REGISTRY = KKTSolverRegistry()
DEFAULT_KKT_REGISTRY.register(HYKKTStrategy(cholmod_loader=_load_cholmod))
DEFAULT_KKT_REGISTRY.register(LiftedStrategy())
DEFAULT_KKT_REGISTRY.register(LDLStrategy(prefer_lldl=False))  # name "ldl"

# Optional aliases if you want:
DEFAULT_KKT_REGISTRY._map["qdldl"] = DEFAULT_KKT_REGISTRY._map["ldl"]
DEFAULT_KKT_REGISTRY._map["lldl"] = LDLStrategy(prefer_lldl=True)
