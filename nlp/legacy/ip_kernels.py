# ip_kernels.py
import numpy as np
from numba import njit

EPS_DIV = 1e-16


@njit(cache=True, fastmath=True)
def _div(a: np.ndarray, b: np.ndarray, eps: float = EPS_DIV) -> np.ndarray:
    out = np.empty_like(a)
    for i in range(a.size):
        denom = b[i] if b[i] > eps else eps
        out[i] = a[i] / denom
    return out


@njit(cache=True, fastmath=True)
def _safe_pos_div(num: np.ndarray, den: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    out = np.empty_like(num)
    for i in range(num.size):
        d = den[i] if den[i] > eps else eps
        out[i] = num[i] / d
    return out


@njit(cache=True, fastmath=True)
def k_max_step_ftb(z: np.ndarray, dz: np.ndarray, tau: float) -> float:
    # alpha ≤ -z_i / dz_i for dz_i < 0 ; then scaled by tau
    n = z.size
    a = 1.0
    for i in range(n):
        if dz[i] < 0.0:
            denom = dz[i] if dz[i] < -EPS_DIV else -EPS_DIV  # min to avoid tiny denom
            cand = -z[i] / denom
            if cand < a:
                a = cand
    a = tau * a
    if a < 0.0:
        a = 0.0
    if a > 1.0:
        a = 1.0
    return a


@njit(cache=True, fastmath=True)
def k_alpha_fraction_to_boundary(
    x, dx, s, ds, lam, dlam, lb, ub, hasL, hasU, tau_pri, tau_dual
) -> float:
    # primary (x/s) and dual (λ/z) combined minimum
    a_pri = 1.0
    a_dual = 1.0

    if s.size > 0:
        # s + α ds > 0
        for i in range(s.size):
            if ds[i] < 0.0:
                denom = ds[i] if ds[i] < -EPS_DIV else -EPS_DIV
                cand = -s[i] / denom
                if cand < a_pri:
                    a_pri = cand
        # λ + α dλ > 0
        for i in range(lam.size):
            if dlam[i] < 0.0:
                denom = dlam[i] if dlam[i] < -EPS_DIV else -EPS_DIV
                cand = -lam[i] / denom
                if cand < a_dual:
                    a_dual = cand

    # bounds: x within [lb, ub]
    n = x.size
    for i in range(n):
        if hasL[i] and dx[i] < 0.0:
            denom = dx[i] if dx[i] < -EPS_DIV else -EPS_DIV
            cand = -(x[i] - lb[i]) / denom
            if cand < a_pri:
                a_pri = cand
        if hasU[i] and dx[i] > 0.0:
            denom = dx[i] if dx[i] > EPS_DIV else EPS_DIV
            cand = (ub[i] - x[i]) / denom
            if cand < a_pri:
                a_pri = cand

    a = tau_pri * a_pri
    b = tau_dual * a_dual
    out = a if a < b else b
    if out < 0.0:
        out = 0.0
    if out > 1.0:
        out = 1.0
    return out


@njit(cache=True, fastmath=True)
def k_build_sigmas(
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
):
    n = sL.size
    Sigma_x_vec = np.zeros(n, dtype=np.float64)

    # Σ_x = zL/(sL+β) + zU/(sU+β) (masked), clipped
    for i in range(n):
        val = 0.0
        if hasL[i]:
            denomL = sL[i] + (bound_shift if use_shifted else 0.0)
            denomL = denomL if denomL > eps_abs else eps_abs
            val += zL[i] / denomL
        if hasU[i]:
            denomU = sU[i] + (bound_shift if use_shifted else 0.0)
            denomU = denomU if denomU > eps_abs else eps_abs
            val += zU[i] / denomU
        if val < 0.0:
            val = 0.0
        if val > cap_val:
            val = cap_val
        Sigma_x_vec[i] = val

    # Σ_s
    mI = s.size
    Sigma_s_vec = np.zeros(mI, dtype=np.float64)
    if mI > 0:
        if use_shifted:
            for i in range(mI):
                denom = s[i] + tau_shift
                denom = denom if denom > eps_abs else eps_abs
                val = lmb[i] / denom
                if val < 0.0:
                    val = 0.0
                if val > cap_val:
                    val = cap_val
                Sigma_s_vec[i] = val
        else:
            # s_safe = max(s, s_floor) with s_floor = clip(|cI|, 1e-8, 1.0)
            for i in range(mI):
                s_floor = abs(cI[i])
                if s_floor < 1e-8:
                    s_floor = 1e-8
                elif s_floor > 1.0:
                    s_floor = 1.0
                sv = s[i] if s[i] > s_floor else s_floor
                sv = sv if sv > eps_abs else eps_abs
                val = lmb[i] / sv
                if val < 0.0:
                    val = 0.0
                if val > cap_val:
                    val = cap_val
                Sigma_s_vec[i] = val

    return Sigma_x_vec, Sigma_s_vec


@njit(cache=True, fastmath=True)
def k_dz_bounds_from_dx(
    dx, zL, zU, sL, sU, hasL, hasU, bound_shift, use_shifted, mu, use_mu: int
):
    n = dx.size
    dzL = np.zeros(n, dtype=np.float64)
    dzU = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if hasL[i]:
            denomL = sL[i] + (bound_shift if use_shifted else 0.0)
            if denomL < EPS_DIV:
                denomL = EPS_DIV
            if use_mu:
                dzL[i] = (mu - denomL * zL[i] - zL[i] * dx[i]) / denomL
            else:
                dzL[i] = -(zL[i] * dx[i]) / denomL
        # upper
        if hasU[i]:
            denomU = sU[i] + (bound_shift if use_shifted else 0.0)
            if denomU < EPS_DIV:
                denomU = EPS_DIV
            if use_mu:
                dzU[i] = (mu - denomU * zU[i] + zU[i] * dx[i]) / denomU
            else:
                dzU[i] = (zU[i] * dx[i]) / denomU
    return dzL, dzU


@njit(cache=True, fastmath=True)
def k_cap_bound_duals_sigma_box(
    zL_new, zU_new, sL_new, sU_new, hasL, hasU, use_shifted, bound_shift, mu, ksig
):
    n = zL_new.size
    for i in range(n):
        if hasL[i]:
            sLc = sL_new[i] + (bound_shift if use_shifted else 0.0)
            if sLc < 1e-16:
                sLc = 1e-16
            lo = mu / (ksig * sLc)
            hi = (ksig * mu) / sLc
            if zL_new[i] < lo:
                zL_new[i] = lo
            elif zL_new[i] > hi:
                zL_new[i] = hi
        if hasU[i]:
            sUc = sU_new[i] + (bound_shift if use_shifted else 0.0)
            if sUc < 1e-16:
                sUc = 1e-16
            lo = mu / (ksig * sUc)
            hi = (ksig * mu) / sUc
            if zU_new[i] < lo:
                zU_new[i] = lo
            elif zU_new[i] > hi:
                zU_new[i] = hi
    # in-place clamp
    return


@njit(cache=True, fastmath=True)
def k_compute_complementarity(s, lam, mu, tau_shift, use_shifted: int) -> float:
    m = s.size
    if m == 0:
        return 0.0
    acc = 0.0
    if use_shifted:
        for i in range(m):
            acc += abs((s[i] + tau_shift) * lam[i] - mu)
    else:
        for i in range(m):
            acc += abs(s[i] * lam[i] - mu)
    return acc / m if m > 0 else 0.0
