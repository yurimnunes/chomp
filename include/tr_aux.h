#pragma once

// ====== Core Eigen ======
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCore>

// ====== STL ======
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

// ====== Project ======
#include "definitions.h"
#include "filter.h"
#include "model.h"

#include "tr_definitions.h"

enum class TRBackend { PCG, DOGLEG, GLTR };

// ---------- NEW: Dogleg (dense SPD) ----------
struct DoglegResult {
    dvec p;
    int iters = 1;
    const char *status = "dogleg";
};


DoglegResult dogleg_step_(const dmat &Hspd, const dvec &g, const Metric &M,
                          double delta) {
    // Symmetrize (cheap) for safety
    const dmat Hs = 0.5 * (Hspd + Hspd.transpose());

    // Cauchy direction
    const dvec Hg = Hs * g;
    const double gg = g.dot(g);
    const double gHg = g.dot(Hg);
    const double alpha_sd = (gHg > 0.0) ? (gg / gHg) : 0.0;
    dvec pC = -alpha_sd * g;

    // Newton step: H pN = -g
    dvec pN;
    {
        Eigen::LLT<dmat> llt(Hs);
        if (llt.info() == Eigen::Success) {
            pN = llt.solve(-g);
        } else {
            Eigen::LDLT<dmat> ldlt(Hs);
            pN = ldlt.solve(-g);
        }
    }

    auto normM = [&](const dvec &v) -> double {
        return M.valid ? M.norm(v) : v.norm();
    };

    const double nC = normM(pC);
    const double nN = normM(pN);

    if (nN <= delta) {
        return {pN, 1, "dogleg_newton"};
    }
    if (nC >= delta) {
        const double t = std::max(1e-16, nC);
        return {(delta / t) * pC, 1, "dogleg_cauchy_boundary"};
    }

    // interpolate on dogleg: p(τ) = pC + τ (pN - pC), ||p(τ)||_M = delta
    const dvec d = pN - pC;
    const double tau = detail::boundary_intersection_metric(M, pC, d, delta);
    return {pC + std::clamp(tau, 0.0, 1.0) * d, 1, "dogleg_interpolate"};
}

// -----------------------------------------------------------------------------
// CG (Steihaug-PCG)
// -----------------------------------------------------------------------------
struct CGResult {
    dvec p;
    TRStatus status;
    int iters;
};

[[nodiscard]] inline CGResult steihaug_pcg(const LinOp &H, const dvec &g,
                                           const Metric &M, double Delta,
                                           double tol, int maxiter,
                                           double neg_curv_tol, const Prec &P,
                                           TRWorkspace &W) {
    const int n = H.n;
    W.ensure(n);
    auto &p = W.p_try;
    p.setZero();
    auto &r = W.r;
    r = -g;
    auto &z = W.z;
    P.apply_into(r, z);
    auto &d = W.d;
    d = z;

    if (r.norm() <= tol)
        return {p, TRStatus::SUCCESS, 0};

    double rz = r.dot(z);
    for (int k = 0; k < maxiter; ++k) {
        H_apply(H, d, W.Hd);
        const double dTHd = d.dot(W.Hd);
        if (dTHd <= neg_curv_tol * std::max(1.0, d.squaredNorm())) {
            const double tau =
                detail::boundary_intersection_metric(M, p, d, Delta);
            p.noalias() += tau * d;
            return {p, TRStatus::NEG_CURV, k};
        }
        const double denom =
            (std::abs(dTHd) > detail::kTinyDen)
                ? dTHd
                : (dTHd >= 0 ? detail::kTinyDen : -detail::kTinyDen);
        const double alpha = rz / denom;

        W.tmp.noalias() = p + alpha * d;
        if (M.norm(W.tmp) >= Delta) {
            const double tau =
                detail::boundary_intersection_metric(M, p, d, Delta);
            p.noalias() += tau * d;
            return {p, TRStatus::BOUNDARY, k};
        }
        p.swap(W.tmp);

        r.noalias() -= alpha * W.Hd;
        if (r.norm() <= tol)
            return {p, TRStatus::SUCCESS, k + 1};

        P.apply_into(r, W.z_next);
        const double rz_next = r.dot(W.z_next);
        const double beta = rz_next / std::max(rz, 1e-32);
        d.noalias() = W.z_next + beta * d;
        rz = rz_next;
    }
    return {p, TRStatus::MAX_ITER, maxiter};
}
