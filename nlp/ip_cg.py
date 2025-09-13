# hykkt_pcg_numba.py
from typing import Optional, Tuple

import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

# ---------- CSR SpMV (Numba) ----------
if _HAS_NUMBA:
    @njit(cache=True, fastmath=True)
    def _csr_matvec(indptr, indices, data, x, out):
        n = len(indptr) - 1
        for i in range(n):
            s = 0.0
            row_start = indptr[i]
            row_end   = indptr[i+1]
            for k in range(row_start, row_end):
                s += data[k] * x[indices[k]]
            out[i] = s

    @njit(cache=True, fastmath=True)
    def _axpy(a, x, y):  # y += a*x
        for i in range(y.size):
            y[i] += a * x[i]

    @njit(cache=True, fastmath=True)
    def _copy(x, y):  # y = x
        for i in range(y.size):
            y[i] = x[i]

    @njit(cache=True, fastmath=True)
    def _scal(a, x):  # x *= a
        for i in range(x.size):
            x[i] *= a

    @njit(cache=True, fastmath=True)
    def _dot(x, y):
        s = 0.0
        for i in range(x.size):
            s += x[i] * y[i]
        return s

    @njit(cache=True, fastmath=True)
    def _inv_diag(diag, out):
        for i in range(diag.size):
            out[i] = 1.0 / diag[i] if diag[i] != 0.0 else 1.0

    @njit(cache=True, fastmath=True)
    def pcg_csr(
        indptr, indices, data,
        b, x0, diag_precond,  # pass diag(H+σI) or block-diag approx
        tol=1e-8, maxit=10_000, atol=0.0
    ) -> Tuple[np.ndarray, int, float]:
        """
        Solve A x = b with A in CSR (SPD) using PCG.
        Preconditioner: M^{-1} = diag_precond^{-1}.
        """
        n = b.size
        x  = x0.copy()
        r  = np.empty(n, dtype=np.float64)
        Ap = np.empty(n, dtype=np.float64)
        z  = np.empty(n, dtype=np.float64)
        p  = np.empty(n, dtype=np.float64)
        Minv = np.empty(n, dtype=np.float64)

        # r = b - A x
        _csr_matvec(indptr, indices, data, x, Ap)
        for i in range(n): r[i] = b[i] - Ap[i]

        # z = M^{-1} r
        _inv_diag(diag_precond, Minv)
        for i in range(n): z[i] = Minv[i] * r[i]
        _copy(z, p)

        rz_old = _dot(r, z)
        norm0  = np.sqrt(_dot(r, r))
        if norm0 == 0.0:
            return x, 0, 0.0

        stop = max(tol*norm0, atol)

        for it in range(1, maxit+1):
            _csr_matvec(indptr, indices, data, p, Ap)
            pAp = _dot(p, Ap)
            if pAp <= 0.0:
                # Numerical trouble / not SPD
                return x, it, np.sqrt(_dot(r, r))
            alpha = rz_old / pAp

            # x += alpha p
            _axpy(alpha, p, x)
            # r -= alpha A p
            _axpy(-alpha, Ap, r)

            nr = np.sqrt(_dot(r, r))
            if nr <= stop:
                return x, it, nr

            # z = M^{-1} r
            for i in range(n): z[i] = Minv[i] * r[i]

            rz_new = _dot(r, z)
            beta   = rz_new / rz_old
            rz_old = rz_new

            # p = z + beta p
            for i in range(n): p[i] = z[i] + beta * p[i]

        return x, maxit, np.sqrt(_dot(r, r))
else:
    # Fallback (no numba): vectorized PCG with diag precond
    def pcg_csr(indptr, indices, data, b, x0, diag_precond, tol=1e-8, maxit=10_000, atol=0.0):
        import numpy as np

        def csr_mv(x):
            out = np.zeros_like(x)
            n = len(indptr)-1
            for i in range(n):
                start, end = indptr[i], indptr[i+1]
                out[i] = (data[start:end] * x[indices[start:end]]).sum()
            return out

        x = x0.copy()
        r = b - csr_mv(x)
        Minv = np.divide(1.0, diag_precond, out=np.ones_like(diag_precond), where=diag_precond!=0)
        z = Minv * r
        p = z.copy()
        rz_old = float(r @ z)
        norm0 = float(np.linalg.norm(r))
        stop = max(tol*norm0, atol)
        for it in range(1, maxit+1):
            Ap = csr_mv(p)
            pAp = float(p @ Ap)
            if pAp <= 0.0:  # not SPD / breakdown
                return x, it, float(np.linalg.norm(r))
            alpha = rz_old / pAp
            x += alpha * p
            r -= alpha * Ap
            nr = float(np.linalg.norm(r))
            if nr <= stop:
                return x, it, nr
            z = Minv * r
            rz_new = float(r @ z)
            beta = rz_new / rz_old
            rz_old = rz_new
            p = z + beta * p
        return x, maxit, float(np.linalg.norm(r))
