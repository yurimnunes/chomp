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

// [[nodiscard]] GLTRResult gltr_step_(const LinOp &H, const dvec &g, const Metric &M, double delta, double tol, int maxiter) {
//     GLTRResult res;
//     int n = g.size();
//     if (n == 0 || g.norm() <= tol) {
//         res.p = dvec::Zero(n);
//         res.iters = 0;
//         res.status = TRStatus::SUCCESS;
//         return res;
//     }

//     bool use_metric = M.valid;
//     LinOp H_op = H;
//     dvec g_op = g;
//     double delta_op = delta;

//     if (use_metric) {
//         g_op = M.L.triangularView<Eigen::Lower>().solve(g);
//         LinOp new_H;
//         new_H.n = n;
//         new_H.mv = [&M, &H](const dvec &v, dvec &out) {
//             dvec tmp = M.L.transpose().triangularView<Eigen::Upper>().solve(v);
//             dvec Htmp;
//             H_apply(H, tmp, Htmp);
//             out = M.L.triangularView<Eigen::Lower>().solve(Htmp);
//         };
//         H_op = new_H;
//     }

//     double gnorm = g_op.norm();
//     std::vector<dvec> V;
//     dvec v = g_op / gnorm;
//     V.push_back(v);
//     dvec w;
//     H_apply(H_op, v, w);
//     double a = v.dot(w);
//     w -= a * v;
//     double b = w.norm();

//     dmat T(1,1);
//     T(0,0) = a;

//     double lambda = 0.0;
//     bool converged = false;
//     int k = 0;
//     dvec s = dvec::Zero(n);
//     bool interior = true;

//     while (k < maxiter && !converged) {
//         k++;

//         if (b < cfg_.neg_curv_tol) {
//             res.status = TRStatus::NEG_CURV;
//             break;
//         }

//         v = w / b;
//         V.push_back(v);
//         H_apply(H_op, v, w);
//         w -= b * V[k-1];
//         a = v.dot(w);
//         w -= a * v;
//         b = w.norm();

//         // Extend T
//         dmat T_new(k+1, k+1);
//         T_new.topLeftCorner(k, k) = T;
//         T_new(k-1, k) = b;
//         T_new(k, k-1) = b;
//         T_new(k, k) = a;
//         T = std::move(T_new);

//         // Solve reduced TR: min 1/2 h^T T h + gnorm h(0) s.t. ||h|| <= delta_op

//         Eigen::SelfAdjointEigenSolver<dmat> es(T);
//         double lambda_min = es.eigenvalues().minCoeff();
//         lambda = std::max(0.0, -lambda_min);
//         dvec h(k+1);
//         double phi = 0.0;
//         int newton_iter = 0;
//         const int max_newton = 50;
//         interior = true;
//         for (newton_iter = 0; newton_iter < max_newton; ++newton_iter) {
//             dmat Tp = T + lambda * dmat::Identity(k+1, k+1);
//             Eigen::LDLT<dmat> ldlt(Tp);
//             if (ldlt.info() != Eigen::Success) {
//                 lambda += 1e-8;
//                 continue;
//             }
//             dvec e1 = dvec::Zero(k+1);
//             e1(0) = gnorm;
//             h = -ldlt.solve(e1);
//             double hnorm = h.norm();
//             phi = hnorm - delta_op;
//             if (lambda == 0.0 && hnorm <= delta_op) {
//                 interior = true;
//                 break;
//             }
//             if (lambda > 0.0 && std::abs(hnorm - delta_op) <= 1e-6 * delta_op) {
//                 interior = false;
//                 break;
//             }
//             double dlambda = (hnorm^2) / (h.dot(ldlt.solve(h))) * (hnorm - delta_op) / delta_op;
//             lambda += dlambda;
//             lambda = std::max(lambda, -lambda_min + 1e-8);
//         }

//         if (interior == false && std::abs(phi) > 1e-6 * delta_op) {
//             h *= delta_op / h.norm();
//         }

//         // Check residual

//         double last = h(k);

//         double residual = b * std::abs(last);

//         if (residual <= tol * gnorm) {
//             converged = true;
//         }

//         // Update s

//         s = dvec::Zero(n);
//         for (size_t i = 0; i < V.size(); ++i) {
//             s += h(i) * V[i];
//         }

//     }

//     if (use_metric) {
//         res.p = M.L.transpose().triangularView<Eigen::Upper>().solve(s);
//     } else {
//         res.p = s;
//     }

//     res.iters = k;

//     res.status = converged ? TRStatus::SUCCESS : TRStatus::MAX_ITER;
//     if (b < cfg_.neg_curv_tol) res.status = TRStatus::NEG_CURV;
//     if (!interior) res.status = TRStatus::BOUNDARY;

//     return res;
// }