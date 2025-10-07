#pragma once
#include "definitions.h"
#include <Eigen/Core>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

// -----------------------------------------------------------------------------
// Info bundles
// -----------------------------------------------------------------------------
struct TRInfo {
    std::string status = "success";
    int iterations = 0;
    double step_norm = 0.0;
    double model_reduction = 0.0;
    double model_reduction_quad = 0.0;
    double sigma_est = 0.0;
    double predicted_reduction_cubic = 0.0;
    double constraint_violation = 0.0;
    bool preconditioned = false;
    bool preconditioned_reduced = false;
    int active_set_size = 0;
    int active_set_iterations = 0;
    std::vector<int> active_set_indices;
    std::string accepted_by = "no_filter";
    bool accepted = true;
    bool soc_applied = false;
    double theta0 = std::numeric_limits<double>::quiet_NaN();
    double theta1 = std::numeric_limits<double>::quiet_NaN();
    bool criticality = false;
    int criticality_shrinks = 0;
};

struct TRResult {
    dvec p;      // step
    TRInfo info; // diagnostics
    dvec lam;    // inequality multipliers (size mI or 0)
    dvec nu;     // equality multipliers   (size mE or 0)
    dvec zL;    // lower bound multipliers (size n or 0)
    dvec zU;    // upper bound multipliers (size n or 0)
};

enum class TRStatus { SUCCESS, BOUNDARY, NEG_CURV, MAX_ITER };

// -----------------------------------------------------------------------------
// Metric
// -----------------------------------------------------------------------------
struct Metric {
    dmat L; // Cholesky factor: M = L L^T
    dmat M; // Cached SPD metric
    bool valid = false;

    [[nodiscard]] inline double norm(const dvec &p) const noexcept {
        if (!valid) [[likely]]
            return p.norm();
        dvec y = L.transpose() * p;
        return y.norm();
    }
};


namespace detail {
constexpr double kTiny = 1e-18;
constexpr double kTinyDen = 1e-32;

[[nodiscard]] inline double
boundary_intersection_euclid(const dvec &p, const dvec &d,
                             double Delta) noexcept {
    const double pTp = p.squaredNorm();
    const double pTd = p.dot(d);
    const double dTd = d.squaredNorm();
    if (dTd <= kTiny)
        return 0.0;
    const double disc = std::max(0.0, pTd * pTd - dTd * (pTp - Delta * Delta));
    return (-pTd + std::sqrt(disc)) / dTd;
}

[[nodiscard]] inline double boundary_intersection_metric(const Metric &M,
                                                         const dvec &p,
                                                         const dvec &d,
                                                         double Delta) {
    if (!M.valid)
        return boundary_intersection_euclid(p, d, Delta);
    const dvec y0 = M.L.transpose() * p;
    const dvec yd = M.L.transpose() * d;
    const double a = yd.squaredNorm();
    const double b = 2.0 * y0.dot(yd);
    const double c = y0.squaredNorm() - Delta * Delta;
    if (a <= kTiny)
        return 0.0;
    const double disc = std::max(0.0, b * b - 4 * a * c);
    return (-b + std::sqrt(disc)) / (2 * a);
}
} // namespace detail


// -----------------------------------------------------------------------------
// Linear operator wrapper
// -----------------------------------------------------------------------------
struct LinOp {
    // y = Hx
    std::function<void(const dvec &x, dvec &y)> mv;
    int n = 0;

    inline void apply_into(const dvec &x, dvec &y) const {
        y.setZero();
        mv(x, y);
    }
};
inline void H_apply(const LinOp &H, const dvec &p, dvec &Hp) {
    H.apply_into(p, Hp);
}

// -----------------------------------------------------------------------------
// Workspace (avoid hot allocations)
// -----------------------------------------------------------------------------
struct TRWorkspace {
    dvec r, z, d, Hd, p_try, z_next, tmp, Hp, Px, Hx, PHx;
    dvec proj_buf1, proj_buf2;
    dvec rhs_small, y_small;

    inline void ensure(int n) {
        auto need = [&](dvec &v) {
            if (v.size() != n)
                v = dvec::Zero(n);
        };
        need(r);
        need(z);
        need(d);
        need(Hd);
        need(p_try);
        need(z_next);
        need(tmp);
        need(Hp);
        need(Px);
        need(Hx);
        need(PHx);
        need(proj_buf1);
        need(proj_buf2);
    }
    inline void ensure_small(int m) {
        if (rhs_small.size() != m)
            rhs_small = dvec::Zero(m);
        if (y_small.size() != m)
            y_small = dvec::Zero(m);
    }
};


// -----------------------------------------------------------------------------
// Preconditioners (factory)
// -----------------------------------------------------------------------------
struct Prec {
    std::function<void(const dvec &, dvec &)> apply;
    bool valid = false;
    inline void apply_into(const dvec &r, dvec &z) const {
        if (valid)
            apply(r, z);
        else
            z = r;
    }
};
struct PrecIdentity {
    [[nodiscard]] static Prec make(int) {
        Prec P;
        P.valid = true;
        P.apply = [](const dvec &r, dvec &z) { z = r; };
        return P;
    }
};

template <class Mat> dvec safe_diag(const Mat &H) {
    dvec d = H.diagonal();
    d = d.unaryExpr([](double v) { return (std::abs(v) > 0) ? v : 1.0; });
    return d;
}
template <> inline dvec safe_diag<spmat>(const spmat &H) {
    dvec d(H.rows());
    d.setOnes();
    for (int k = 0; k < H.outerSize(); ++k)
        for (spmat::InnerIterator it(H, k); it; ++it)
            if (it.row() == it.col())
                d[it.row()] = (std::abs(it.value()) > 0) ? it.value() : 1.0;
    return d;
}

struct PrecJacobi {
    template <class Mat> [[nodiscard]] static Prec from(const Mat &H) {
        Prec P;
        const dvec inv = safe_diag(H).unaryExpr(
            [](double v) { return (std::abs(v) > 0) ? 1.0 / v : 0.0; });
        P.valid = true;
        P.apply = [inv](const dvec &r, dvec &z) {
            z = inv.array() * r.array();
        };
        return P;
    }
};

struct PrecSSOR {
    [[nodiscard]] static Prec fromSparseSPD(const spmat &H,
                                            double omega = 1.0) {
        Prec P;
        const int n = (int)H.rows();
        if (n == 0)
            return P;
        dvec D(n);
        D.setZero();
        for (int k = 0; k < H.outerSize(); ++k)
            for (spmat::InnerIterator it(H, k); it; ++it)
                if (it.row() == it.col())
                    D[it.row()] = std::abs(it.value()) > 0 ? it.value() : 1.0;

        P.valid = true;
        P.apply = [=](const dvec &r, dvec &z) {
            dvec y = r; // forward
            for (int k = 0; k < H.outerSize(); ++k) {
                for (spmat::InnerIterator it(H, k); it; ++it) {
                    const int i = it.row(), j = it.col();
                    if (i > j)
                        y[i] -= omega * it.value() * y[j];
                }
                y[k] /= (std::abs(D[k]) > 1e-30 ? D[k] : 1.0);
            }
            z = y; // backward
            for (int k = n - 1; k >= 0; --k) {
                for (spmat::InnerIterator it(H, k); it; ++it) {
                    const int i = it.row(), j = it.col();
                    if (i < j)
                        z[i] -= omega * it.value() * z[j];
                }
                z[k] /= (std::abs(D[k]) > 1e-30 ? D[k] : 1.0);
            }
            const double scale = 1.0 / (omega * (2.0 - omega));
            if (std::isfinite(scale) && std::abs(scale) > 1e-16)
                z *= scale;
        };
        return P;
    }
};

template <class Mat>
[[nodiscard]] Prec make_prec(const Mat *H, const std::string &kind,
                             bool use_prec, double ssor_omega) {
    const int n = H ? (int)H->rows() : 0;
    if (!use_prec)
        return PrecIdentity::make(n);
    if (kind == "identity")
        return PrecIdentity::make(n);
    if constexpr (std::is_same_v<Mat, spmat>) {
        if (H && kind == "ssor")
            return PrecSSOR::fromSparseSPD(*H, ssor_omega);
    }
    if (H)
        return PrecJacobi::from(*H);
    return PrecIdentity::make(n);
}
