// tr_sota_optimized.cpp
// Trust-region (Steihaug–Toint) with Eigen + pybind11 + SOC correction.
// Optimized to reduce heap churn and tiny temporaries while keeping ALL
// features.
// - Dense or sparse H (via LinOp)
// - Equality + inequality constraints (active set)
// - Preconditioners: identity / Jacobi / SSOR
// - Ellipsoidal norm via cached Cholesky
// - SOC ported from your Python logic (model.eval_all, constraint_violation)
// - Criticality step, curvature-aware TR growth, filter-based acceptance
// - Box handling with mode switch: "projection" | "alpha"
// (fraction-to-boundary)
//
// Drop-in replacement for your previous file: same public API/signatures.
//
// Build with your existing CMake setup for pybind11.
//
// Notes on improvements:
//  - LinOp no longer allocates per-apply; callers pass output buffers
//  - A single TRWorkspace holds all hot-loop vectors to avoid repeated allocs
//  - PCG rewritten to reuse workspace buffers and avoid "+=" surprises
//  - Projected preconditioner reuses temporary buffers and projections
//  - All helper paths avoid ephemeral Eigen temporaries where possible
//  - Safer numerics: fewer divisions by tiny numbers, consistent clamps

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

using dvec = Eigen::VectorXd;
using dmat = Eigen::MatrixXd;
using spmat = Eigen::SparseMatrix<double, Eigen::ColMajor, long>;

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

inline double safe_norm(const dvec &x) { return x.size() ? x.norm() : 0.0; }

enum class TRStatus { SUCCESS, BOUNDARY, NEG_CURV, MAX_ITER };

struct Metric {
    dmat L; // lower-triangular, M = L L^T
    bool valid = false;

    // y = L^T p; ||y||
    double norm(const dvec &p) const {
        if (!valid)
            return p.norm();
        dvec y = L.transpose() * p;
        return y.norm();
    }
};

inline double boundary_intersection_euclid(const dvec &p, const dvec &d,
                                           double Delta) {
    const double pTp = p.squaredNorm();
    const double pTd = p.dot(d);
    const double dTd = d.squaredNorm();
    if (dTd <= 1e-18)
        return 0.0;
    const double disc = std::max(0.0, pTd * pTd - dTd * (pTp - Delta * Delta));
    return (-pTd + std::sqrt(disc)) / dTd;
}

inline double boundary_intersection_metric(const Metric &M, const dvec &p,
                                           const dvec &d, double Delta) {
    if (!M.valid)
        return boundary_intersection_euclid(p, d, Delta);
    dvec y0 = M.L.transpose() * p;
    dvec yd = M.L.transpose() * d;
    const double a = yd.squaredNorm();
    const double b = 2.0 * y0.dot(yd);
    const double c = y0.squaredNorm() - Delta * Delta;
    if (a <= 1e-18)
        return 0.0;
    const double disc = std::max(0.0, b * b - 4 * a * c);
    return (-b + std::sqrt(disc)) / (2 * a);
}

struct LinOp {
    // y = Hx (y is pre-sized and will be overwritten)
    std::function<void(const dvec &x, dvec &y)> mv;
    int n = 0;

    inline void apply_into(const dvec &x, dvec &y) const {
        y.setZero();
        mv(x, y);
    }
};

// ---------- Workspace: reused across hot paths to avoid allocations ----------
struct TRWorkspace {
    dvec r, z, d, Hd, p_try, z_next, tmp, Hp, Px, Hx, PHx; // general
    dvec proj_buf1, proj_buf2;
    dvec rhs_small, y_small; // small solves
    void ensure(int n) {
        auto need = [&](dvec &v) {
            if ((int)v.size() != n)
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
    void ensure_small(int m) {
        if ((int)rhs_small.size() != m)
            rhs_small = dvec::Zero(m);
        if ((int)y_small.size() != m)
            y_small = dvec::Zero(m);
    }
};

// -------------------- Preconditioners --------------------
struct Prec {
    std::function<void(const dvec &r, dvec &z)> apply;
    bool valid = false;
    inline void apply_into(const dvec &r, dvec &z) const {
        if (valid)
            apply(r, z);
        else
            z = r;
    }
};

struct PrecIdentity {
    static Prec make(int) {
        Prec P;
        P.valid = true;
        P.apply = [](const dvec &r, dvec &z) { z = r; };
        return P;
    }
};

struct PrecJacobi {
    static Prec fromDiag(const dvec &diag) {
        Prec P;
        if (diag.size() == 0)
            return P;
        dvec inv = diag.unaryExpr(
            [](double v) { return (std::abs(v) > 0) ? 1.0 / v : 0.0; });
        P.valid = true;
        P.apply = [inv](const dvec &r, dvec &z) {
            z = inv.array() * r.array();
        };
        return P;
    }
};

struct PrecSSOR {
    static Prec fromSparseSPD(const spmat &H, double omega = 1.0) {
        Prec P;
        const int n = (int)H.rows();
        if (n == 0)
            return P;

        dvec D(n);
        D.setZero();
        for (int k = 0; k < H.outerSize(); ++k)
            for (spmat::InnerIterator it(H, k); it; ++it)
                if (it.row() == it.col())
                    D[it.row()] = (std::abs(it.value()) > 0 ? it.value() : 1.0);

        P.valid = true;
        P.apply = [=](const dvec &r, dvec &z) {
            dvec y = r;
            // forward (D+ωL) y = r
            for (int k = 0; k < H.outerSize(); ++k) {
                for (spmat::InnerIterator it(H, k); it; ++it) {
                    int i = it.row(), j = it.col();
                    if (i > j)
                        y[i] -= omega * it.value() * y[j];
                }
                y[k] /= (std::abs(D[k]) > 1e-30 ? D[k] : 1.0);
            }
            // backward (D+ωU)^T z = y
            z = y;
            for (int k = n - 1; k >= 0; --k) {
                for (spmat::InnerIterator it(H, k); it; ++it) {
                    int i = it.row(), j = it.col();
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

inline dvec diag_from_H(const dmat &H) {
    return H.diagonal().unaryExpr(
        [](double v) { return (std::abs(v) > 0) ? v : 1.0; });
}
inline dvec diag_from_H(const spmat &H) {
    dvec d(H.rows());
    d.setOnes();
    for (int k = 0; k < H.outerSize(); ++k)
        for (spmat::InnerIterator it(H, k); it; ++it)
            if (it.row() == it.col())
                d[it.row()] = (std::abs(it.value()) > 0 ? it.value() : 1.0);
    return d;
}

// -------------------- Utilities --------------------
inline dmat symmetrize(const dmat &A) { return 0.5 * (A + A.transpose()); }

inline dmat psd_cholesky_with_shift(const dmat &S, double shift_min) {
    Eigen::LLT<dmat> llt(S);
    if (llt.info() == Eigen::Success)
        return llt.matrixL();
    Eigen::SelfAdjointEigenSolver<dmat> es(0.5 * (S + S.transpose()));
    const auto &vals = es.eigenvalues();
    const auto &V = es.eigenvectors();
    double add = shift_min;
    if (vals.size())
        add = std::max(shift_min, 1e-12 - vals.minCoeff());
    dvec pos = (vals.array() + add).max(shift_min).matrix();
    dmat Spos = V * pos.asDiagonal() * V.transpose();
    Eigen::LLT<dmat> llt2(Spos);
    if (llt2.info() != Eigen::Success)
        throw std::runtime_error("Cholesky failed after shift");
    return llt2.matrixL();
}

inline void H_apply(const LinOp &H, const dvec &p, dvec &Hp) {
    H.apply_into(p, Hp);
}

inline double model_reduction_quad_into(const LinOp &H, const dvec &g,
                                        const dvec &p, dvec &Hp) {
    H_apply(H, p, Hp);
    return -(g.dot(p) + 0.5 * p.dot(Hp));
}

inline double estimate_sigma_into(const LinOp &H, const Metric &M,
                                  const dvec &g, const dvec &p, dvec &Hp) {
    if (p.size() == 0)
        return 0.0;
    const double den = std::pow(M.norm(p), 2);
    if (den <= 1e-14)
        return 0.0;
    H_apply(H, p, Hp);
    return std::max(0.0, -(p.dot(Hp) + p.dot(g)) / den);
}

inline double pred_red_cubic_into(const LinOp &H, const Metric &M,
                                  const dvec &g, const dvec &p, double sigma,
                                  dvec &Hp) {
    H_apply(H, p, Hp);
    const double quad = g.dot(p) + 0.5 * p.dot(Hp);
    return -(quad + (sigma / 3.0) * std::pow(M.norm(p), 3));
}

// --- replace your current AAT_solve with this cached version ---
inline dvec AAT_solve(const dmat &A, const dvec &rhs,
                      double reg_floor = 1e-12) {
    // Lightweight single-entry cache keyed by (rows, cols, Fro-norm, diag-sum,
    // reg)
    struct Cache {
        int rows = -1, cols = -1;
        double fnorm = std::numeric_limits<double>::quiet_NaN();
        double dsum = std::numeric_limits<double>::quiet_NaN();
        double reg = std::numeric_limits<double>::quiet_NaN();
        dmat M;               // cached A A^T + reg I
        Eigen::LLT<dmat> llt; // cached factorization
        bool valid = false;
    };
    static thread_local Cache C;

    auto nearly_eq = [](double a, double b, double tol = 1e-10) {
        return std::abs(a - b) <=
               tol * (1.0 + std::max(std::abs(a), std::abs(b)));
    };

    const int m = (int)A.rows();
    const int n = (int)A.cols();
    const double reg = std::max(reg_floor, 1e-12);

    // Fast features to characterize A
    const double fnorm = A.norm();
    const double dsum =
        A.rowwise().squaredNorm().sum(); // cheap stable checksum

    bool reuse = C.valid && C.rows == m && C.cols == n &&
                 nearly_eq(C.fnorm, fnorm, 1e-12) &&
                 nearly_eq(C.dsum, dsum, 1e-12) && nearly_eq(C.reg, reg, 1e-14);

    if (!reuse) {
        // Rebuild M = A Aᵀ + reg I and (re)factor
        dmat M = A * A.transpose();
        M.diagonal().array() += reg;

        Eigen::LLT<dmat> llt(M);
        if (llt.info() != Eigen::Success) {
            // robust fallback
            Eigen::LDLT<dmat> ldlt(M);
            if (ldlt.info() == Eigen::Success) {
                return ldlt.solve(rhs);
            }
            // worst-case: dense solve
            return M.partialPivLu().solve(rhs);
        }

        // Update cache
        C.rows = m;
        C.cols = n;
        C.fnorm = fnorm;
        C.dsum = dsum;
        C.reg = reg;
        C.M = std::move(M);
        C.llt = std::move(llt);
        C.valid = true;
    }

    // Solve using cached LLT
    return C.llt.solve(rhs);
}

// --- replace your current project_tangent with this (same signature) ---
inline dvec project_tangent(const dvec &v, const dmat &Aeq) {
    if (Aeq.size() == 0)
        return v;
    dvec y = AAT_solve(Aeq, Aeq * v); // now uses cached factorization
    return v - Aeq.transpose() * y;
}

// NEW: lightweight wrapper used by code paths that want an "into" API.
inline void project_tangent_into(const dvec &v, const dmat &Aeq, dvec &out,
                                 TRWorkspace & /*W*/) {
    out = project_tangent(v, Aeq);
}

// -------------------- Steihaug–Toint PCG (workspace-based)
// --------------------
struct CGResult {
    dvec p;
    TRStatus status;
    int iters;
};

inline CGResult steihaug_pcg(const LinOp &H, const dvec &g, const Metric &M,
                             double Delta, double tol, int maxiter,
                             double neg_curv_tol, const Prec &P,
                             TRWorkspace &W) {
    const int n = H.n;
    W.ensure(n);

    dvec &p = W.p_try;
    p.setZero();
    dvec &r = W.r;
    r = -g;
    dvec &z = W.z;
    P.apply_into(r, z);
    dvec &d = W.d;
    d = z;

    if (r.norm() <= tol)
        return {p, TRStatus::SUCCESS, 0};

    double rz = r.dot(z);
    dvec &Hd = W.Hd;

    for (int k = 0; k < maxiter; ++k) {
        H_apply(H, d, Hd);
        const double dTHd = d.dot(Hd);
        if (dTHd <= neg_curv_tol * std::max(1.0, d.squaredNorm())) {
            const double tau = boundary_intersection_metric(M, p, d, Delta);
            p.noalias() += tau * d;
            return {p, TRStatus::NEG_CURV, k};
        }
        const double denom =
            (std::abs(dTHd) > 1e-32 ? dTHd : (dTHd >= 0 ? 1e-32 : -1e-32));
        const double alpha = rz / denom;

        W.tmp.noalias() = p + alpha * d; // candidate
        if (M.norm(W.tmp) >= Delta) {
            const double tau = boundary_intersection_metric(M, p, d, Delta);
            p.noalias() += tau * d;
            return {p, TRStatus::BOUNDARY, k};
        }
        p.swap(W.tmp); // accept

        r.noalias() -= alpha * Hd;
        if (r.norm() <= tol)
            return {p, TRStatus::SUCCESS, k + 1};

        dvec &z_next = W.z_next;
        P.apply_into(r, z_next);
        const double rz_next = r.dot(z_next);
        const double beta = rz_next / std::max(rz, 1e-32);
        d.noalias() = z_next + beta * d;
        rz = rz_next;
    }
    return {p, TRStatus::MAX_ITER, maxiter};
}

// -------------------- Config --------------------
struct TRConfig {
    double delta0 = 1.0, delta_min = 1e-10, delta_max = 1e3;
    double cg_tol = 1e-8, cg_tol_rel = 1e-4;
    int cg_maxiter = 200;
    double neg_curv_tol = 1e-14, rcond = 1e-12, metric_shift = 1e-10;
    double zeta = 0.8, constraint_tol = 1e-8;
    int max_active_set_iter = 8;
    bool use_prec = true;
    std::string prec_kind = "ssor";   // "auto_jacobi" | "ssor" | "identity"
    std::string norm_type = "euclid"; // "euclid" | "ellip"
    double ssor_omega = 1.0;
    double gamma1 = 0.5; // shrink when rho < eta1
    double gamma2 = 2.0; // growth when rho >= eta2 and near boundary
    double eta1 = 0.1;   // low acceptance bar
    double eta2 = 0.9;   // high acceptance bar
    bool curvature_aware = true;

    // criticality / globalization
    bool criticality_enabled = true;
    int max_crit_shrinks = 1;
    double kappa_g = 1e-2;   // ||P g|| <= kappa_g * Delta
    double theta_crit = 0.5; // shrink factor when in criticality

    std::string box_mode = "alpha"; // "alpha" | "projection"
    bool recover_lam_active_only =
        true; // NEW: restrict λ recovery to active inequalities
};

// -------------------- TrustRegionManager --------------------
class TrustRegionManager {
public:
    explicit TrustRegionManager(const TRConfig &cfg)
        : cfg_(cfg), delta_(cfg.delta0) {
        metric_.valid = false;
    }

    TrustRegionManager() : cfg_(), delta_(cfg_.delta0) {
        metric_.valid = false;
    }

    void set_metric_from_H_dense(const dmat &H) {
        if (cfg_.norm_type != "ellip") {
            metric_.valid = false;
            return;
        }
        dmat L = psd_cholesky_with_shift(
            symmetrize(H) +
                cfg_.metric_shift * dmat::Identity(H.rows(), H.cols()),
            cfg_.metric_shift);
        metric_.L = std::move(L);
        metric_.valid = true;
    }
    void set_metric_from_H_sparse(const spmat &H) {
        if (cfg_.norm_type != "ellip") {
            metric_.valid = false;
            return;
        }
        dmat Hd = dmat(H);
        set_metric_from_H_dense(Hd);
    }

    // Dense entry (SOC-aware)
    std::tuple<dvec, py::dict, dvec, dvec>
    solve_dense(const dmat &H, const dvec &g,
                const std::optional<dmat> &Aineq = std::nullopt,
                const std::optional<dvec> &bineq = std::nullopt,
                const std::optional<dmat> &Aeq = std::nullopt,
                const std::optional<dvec> &beq = std::nullopt,
                const std::optional<py::object> &model = std::nullopt,
                const std::optional<dvec> &x = std::nullopt,
                const std::optional<dvec> &lb = std::nullopt,
                const std::optional<dvec> &ub = std::nullopt, double mu = 0.0,
                const std::optional<py::object> &filter = std::nullopt,
                const std::optional<double> &f_old = std::nullopt) {
        LinOp Hop;
        Hop.n = (int)g.size();
        Hop.mv = [&H](const dvec &in, dvec &out) { out.noalias() = H * in; };
        auto [p, info, lam, nu] =
            core_solve_(Hop, g, Aineq, bineq, Aeq, beq, std::nullopt, H, model,
                        x, lb, ub, mu, filter, f_old);
        return {p, info, lam, nu};
    }

    // Sparse entry (SOC-aware)
    std::tuple<dvec, py::dict, dvec, dvec>
    solve_sparse(const spmat &H, const dvec &g,
                 const std::optional<dmat> &Aineq = std::nullopt,
                 const std::optional<dvec> &bineq = std::nullopt,
                 const std::optional<dmat> &Aeq = std::nullopt,
                 const std::optional<dvec> &beq = std::nullopt,
                 const std::optional<py::object> &model = std::nullopt,
                 const std::optional<dvec> &x = std::nullopt,
                 const std::optional<dvec> &lb = std::nullopt,
                 const std::optional<dvec> &ub = std::nullopt, double mu = 0.0,
                 const std::optional<py::object> &filter = std::nullopt,
                 const std::optional<double> &f_old = std::nullopt) {
        LinOp Hop;
        Hop.n = (int)g.size();
        Hop.mv = [&H](const dvec &in, dvec &out) { out.noalias() = H * in; };
        auto [p, info, lam, nu] =
            core_solve_(Hop, g, Aineq, bineq, Aeq, beq, H, std::nullopt, model,
                        x, lb, ub, mu, filter, f_old);
        return {p, info, lam, nu};
    }

    double get_delta() const { return delta_; }
    void set_delta(double d) {
        delta_ = std::min(std::max(d, cfg_.delta_min), cfg_.delta_max);
    }

    double delta_;

private:
    TRConfig cfg_;
    Metric metric_;
    mutable TRWorkspace W_; // <— single reusable workspace

    // --- transient box context (valid only inside a solve) ---
    mutable std::optional<dvec> box_x_;
    mutable std::optional<dvec> box_lb_;
    mutable std::optional<dvec> box_ub_;

    inline bool use_projection_() const {
        return (cfg_.box_mode == "projection");
    }

    double tr_norm_(const dvec &p) const { return metric_.norm(p); }
    double cg_tol_(double gnorm) const {
        return std::min(cfg_.cg_tol, cfg_.cg_tol_rel * std::max(gnorm, 1e-16));
    }

    // ---------- Gradient projected norm (box-aware like Python) ----------
    double projected_grad_norm_(const dvec &g,
                                const std::optional<dmat> &Aeq) const {
        dvec gp = g;
        if (box_x_ && (box_lb_ || box_ub_)) {
            const dvec &x = *box_x_;
            const dvec *lb = box_lb_ ? &*box_lb_ : nullptr;
            const dvec *ub = box_ub_ ? &*box_ub_ : nullptr;
            const double tol = 1e-12;
            for (int i = 0; i < gp.size(); ++i) {
                if (lb && i < (int)lb->size() && x[i] <= (*lb)[i] + tol &&
                    gp[i] > 0.0)
                    gp[i] = 0.0;
                if (ub && i < (int)ub->size() && x[i] >= (*ub)[i] - tol &&
                    gp[i] < 0.0)
                    gp[i] = 0.0;
            }
        }
        if (Aeq && Aeq->size()) {
            project_tangent_into(gp, *Aeq, W_.tmp, W_);
            return tr_norm_(W_.tmp);
        }
        return tr_norm_(gp);
    }

    // ---------- Criticality ----------
    std::pair<bool, int>
    maybe_apply_criticality_(const dvec &g, const std::optional<dmat> &Aeq) {
        bool used = false;
        int shr = 0;
        if (!cfg_.criticality_enabled)
            return {false, 0};
        for (int k = 0; k < cfg_.max_crit_shrinks; ++k) {
            const double pg = projected_grad_norm_(g, Aeq);
            if (pg <= cfg_.kappa_g * std::max(delta_, 1e-16)) {
                delta_ = std::max(cfg_.delta_min, cfg_.theta_crit * delta_);
                used = true;
                ++shr;
            } else
                break;
        }
        return {used, shr};
    }

    // ---------- Curvature-aware growth ----------
    double curvature_along_(const LinOp &H, const dvec &p) {
        double denom = tr_norm_(p);
        denom *= denom;
        if (denom <= 1e-16)
            return std::numeric_limits<double>::quiet_NaN();
        H_apply(H, p, W_.Hp);
        double num = p.dot(W_.Hp);
        return num / denom;
    }

    void update_tr_radius_(double predicted, double actual, double step_norm,
                           const LinOp &H, const dvec &p) {
        if (!(std::isfinite(predicted) && std::abs(predicted) > 1e-16))
            return;

        const double rho = actual / predicted;
        const double eta1 = cfg_.eta1;
        const double eta2 = cfg_.eta2;

        if (rho < eta1) {
            delta_ *= cfg_.gamma1;
        } else {
            const bool near_boundary = (step_norm >= 0.8 * delta_);
            if (near_boundary && rho >= eta2) {
                double g2 = cfg_.gamma2;
                if (cfg_.curvature_aware) {
                    const double curv = curvature_along_(H, p);
                    if (std::isfinite(curv)) {
                        if (curv <= -1e-10)
                            g2 = 1.0;
                        else if (curv < 1e-12)
                            g2 = std::min(1.2, g2);
                        else if (curv < 1e-4)
                            g2 = std::min(1.6, std::max(1.2, g2));
                        else if (curv < 1e-2)
                            g2 = std::min(2.0, std::max(1.4, g2));
                        else
                            g2 = std::min(2.5, std::max(1.6, 1.25 * g2));
                    }
                }
                delta_ = std::min(cfg_.delta_max, g2 * delta_);
            }
        }
        delta_ = std::min(std::max(delta_, cfg_.delta_min), cfg_.delta_max);
    }

    double choose_predicted_(const TRInfo &info) const {
        const double pc = info.predicted_reduction_cubic;
        if (std::isfinite(pc) && std::abs(pc) >= 1e-16)
            return pc;
        return info.model_reduction_quad;
    }

    // ---------- Preconditioners ----------
    Prec make_prec_(const std::optional<spmat> &Hs,
                    const std::optional<dmat> &Hd) const {
        const int n = (int)(Hs ? Hs->rows() : Hd->rows());
        if (!cfg_.use_prec)
            return PrecIdentity::make(n);
        if (cfg_.prec_kind == "identity")
            return PrecIdentity::make(n);
        if (cfg_.prec_kind == "ssor" && Hs)
            return PrecSSOR::fromSparseSPD(*Hs, cfg_.ssor_omega);
        if (Hd)
            return PrecJacobi::fromDiag(diag_from_H(*Hd));
        if (Hs)
            return PrecJacobi::fromDiag(diag_from_H(*Hs));
        return PrecIdentity::make(n);
    }

    Prec make_projected_prec_(const dmat &Aeq, const Prec &base) const {
        Prec Pproj;
        Pproj.valid = true;
        Pproj.apply = [this, Aeq, base](const dvec &r, dvec &z) {
            project_tangent_into(r, Aeq, W_.proj_buf1, W_); // rp
            if (base.valid) {
                base.apply(W_.proj_buf1, W_.proj_buf2);
                project_tangent_into(W_.proj_buf2, Aeq, z, W_);
            } else {
                z = W_.proj_buf1;
            }
        };
        return Pproj;
    }

    // ---------- Box handling ----------
    static double alpha_max_box_(const dvec &x, const dvec &d,
                                 const std::optional<dvec> &lb,
                                 const std::optional<dvec> &ub,
                                 double tau = 0.999999) {
        if ((!lb && !ub) || d.size() == 0)
            return 1.0;
        double amax = 1.0;
        if (lb)
            for (int i = 0; i < d.size(); ++i)
                if (d[i] < 0.0)
                    amax = std::min(amax, ((*lb)[i] - x[i]) / d[i]);
        if (ub)
            for (int i = 0; i < d.size(); ++i)
                if (d[i] > 0.0)
                    amax = std::min(amax, ((*ub)[i] - x[i]) / d[i]);
        amax = std::clamp(amax, 0.0, 1.0);
        return std::clamp(tau * std::max(0.0, amax), 0.0, 1.0);
    }

    dvec enforce_box_on_step(const dvec &x, const dvec &p,
                             const std::optional<dvec> &lb,
                             const std::optional<dvec> &ub,
                             const std::string &mode) const {
        if (!lb && !ub)
            return p;
        if (mode == "projection") {
            dvec xp = x + p;
            if (lb)
                xp = xp.cwiseMax(*lb);
            if (ub)
                xp = xp.cwiseMin(*ub);
            return xp - x;
        } else {
            const double a = alpha_max_box_(x, p, lb, ub);
            return a * p;
        }
    }

    // ---------- Solve core ----------
    std::tuple<dvec, py::dict, dvec, dvec> core_solve_(
        const LinOp &Hop, const dvec &g, const std::optional<dmat> &Aineq,
        const std::optional<dvec> &bineq, const std::optional<dmat> &Aeq,
        const std::optional<dvec> &beq, const std::optional<spmat> &sparseH,
        const std::optional<dmat> &denseH,
        const std::optional<py::object> &model, const std::optional<dvec> &x,
        const std::optional<dvec> &lb, const std::optional<dvec> &ub, double mu,
        const std::optional<py::object> &filter,
        const std::optional<double> &f_old_opt) {
        TRInfo info;
        box_x_ = x;
        box_lb_ = lb;
        box_ub_ = ub;

        dvec lam(Aineq ? (int)Aineq->rows() : 0);
        dvec nu(Aeq ? (int)Aeq->rows() : 0);

        dvec p; // final step

        if (Aeq) {
            auto [pE, infE] = solve_with_equalities_(
                Hop, g, *Aeq, *beq, Aineq, bineq, sparseH, denseH, x, lb, ub);
            p = pE;
            info = infE;
        } else if (Aineq) {
            auto [pI, infI] = solve_with_inequalities_(
                Hop, g, *Aineq, bineq.value_or(dvec::Zero(Aineq->rows())),
                sparseH, denseH, x, lb, ub);
            p = pI;
            info = infI;
        } else {
            // Unconstrained
            {
                auto [crit_used, shr] =
                    maybe_apply_criticality_(g, std::nullopt);
                info.criticality = crit_used;
                info.criticality_shrinks = shr;
            }
            auto P = make_prec_(sparseH, denseH);
            const double tol = cg_tol_(tr_norm_(g));
            auto cg = steihaug_pcg(Hop, g, metric_, delta_, tol,
                                   cfg_.cg_maxiter, cfg_.neg_curv_tol, P, W_);
            p = cg.p;
            if (x && (lb || ub))
                p = enforce_box_on_step(*x, p, lb, ub, cfg_.box_mode);

            info.iterations = cg.iters;
            info.status = status_to_string_(cg.status);
            info.step_norm = tr_norm_(p);
            info.model_reduction = model_reduction_quad_into(Hop, g, p, W_.Hp);
            info.model_reduction_quad = info.model_reduction;
            info.preconditioned = P.valid;
        }

        // Sigma + predicted cubic reduction
        const double sigma = estimate_sigma_into(Hop, metric_, g, p, W_.Hp);
        info.sigma_est = sigma;
        info.predicted_reduction_cubic =
            pred_red_cubic_into(Hop, metric_, g, p, sigma, W_.Hp);
        info.step_norm = tr_norm_(p);
        if (Aeq)
            info.constraint_violation = ((*Aeq) * p + (*beq)).norm();

        // Recover multipliers (+ box μL/μU)
        // --- inside core_solve_ (near end), replace the existing call ---
        dvec muL, muU;
        std::tie(lam, nu, muL, muU) = recover_multipliers_(
            Hop, g, p, sigma, Aeq, Aineq, x, lb, ub,
            info.active_set_size
                ? std::optional<std::vector<int>>(info.active_set_indices)
                : std::nullopt);

        // Criticality before SOC (so SOC sees shrunk Δ)
        if (model) {
            if (Aeq && Aeq->size()) {
                auto [c2, s2] = maybe_apply_criticality_(g, Aeq);
                (void)c2;
                (void)s2;
            } else {
                auto [c2, s2] = maybe_apply_criticality_(g, std::nullopt);
                (void)c2;
                (void)s2;
            }
        }

        // SOC
        if (model && x) {
            auto soc = soc_correction_(
                *model, *x, p, Hop, denseH, sparseH, lam, mu,
                /*wE=*/10.0, /*wI=*/1.0, /*tolE=*/1e-8, /*violI=*/0.0,
                /*reg=*/1e-12, /*sigma0=*/cfg_.metric_shift, lb, ub);
            if (soc.applied) {
                p += soc.q;
                info.soc_applied = true;
                info.theta0 = soc.theta0;
                info.theta1 = soc.theta1;
                info.step_norm = tr_norm_(p);
                info.model_reduction =
                    model_reduction_quad_into(Hop, g, p, W_.Hp);
                info.model_reduction_quad = info.model_reduction;
            }
        }

        // Evaluate trial (f, theta)
        std::optional<double> f_trial, theta_trial;
        if (model && (x || p.size())) {
            auto ft = eval_model_f_theta_(model, x, p);
            f_trial = ft.first;
            theta_trial = ft.second;
        }

        // Infer f_old if not provided
        std::optional<double> f_old = f_old_opt;
        if (!f_old && model) {
            try {
                if (py::hasattr(*model, "f_current")) {
                    f_old = py::float_((*model).attr("f_current"));
                } else if (x) {
                    auto ft0 =
                        eval_model_f_theta_(model, x, dvec::Zero(g.size()));
                    f_old = ft0.first;
                }
            } catch (...) {
            }
        }

        // Filter acceptance
        bool accepted = true;
        std::string accepted_by = "no_filter";
        if (filter && f_trial && theta_trial && std::isfinite(*f_trial) &&
            std::isfinite(*theta_trial)) {
            try {
                py::object is_acc = (*filter).attr("is_acceptable");
                bool ok = is_acc(*theta_trial, *f_trial,
                                 py::arg("trust_radius") = delta_)
                              .cast<bool>();
                if (ok) {
                    py::object add_if = (*filter).attr("add_if_acceptable");
                    add_if(*theta_trial, *f_trial,
                           py::arg("trust_radius") = delta_);
                    accepted_by = "filter";
                } else {
                    accepted = false;
                    accepted_by = "rejected_by_filter";
                }
            } catch (...) {
            }
        }

        // Backtracking if rejected
        if (!accepted) {
            auto [ok, p_new, by, f_bt, th_bt] = _backtrack_on_reject_(
                model, x, p, lb, ub, filter, delta_, f_old, /*max_tries=*/3);
            if (ok) {
                p = p_new;
                accepted = true;
                accepted_by = by;
                f_trial = f_bt;
                theta_trial = th_bt;
                info.step_norm = tr_norm_(p);
                info.model_reduction =
                    model_reduction_quad_into(Hop, g, p, W_.Hp);
                info.model_reduction_quad = info.model_reduction;
            }
        }

        info.accepted = accepted;
        info.accepted_by = accepted_by;

        // TR radius update on acceptance
        if (info.accepted) {
            const double predicted = choose_predicted_(info);
            double actual = std::numeric_limits<double>::quiet_NaN();
            if (f_old && f_trial && std::isfinite(*f_old) &&
                std::isfinite(*f_trial))
                actual = (*f_old) - (*f_trial);
            if (std::isfinite(predicted) && std::isfinite(actual))
                update_tr_radius_(predicted, actual, info.step_norm, Hop, p);
        }

        // Pack & return
        py::dict pyd = pack_info_(info);
        pyd["f_trial"] = f_trial ? py::float_(*f_trial) : py::float_(NAN);
        pyd["theta_trial"] =
            theta_trial ? py::float_(*theta_trial) : py::float_(NAN);

        box_x_.reset();
        box_lb_.reset();
        box_ub_.reset();
        return {p, pyd, lam, nu};
    }

    // ---------- Projected operator for tangent CG ----------
    LinOp make_projected_operator_(const LinOp &H, const dmat &Aeq) const {
        LinOp P;
        P.n = H.n;
        P.mv = [this, H, Aeq](const dvec &x, dvec &y) {
            project_tangent_into(x, Aeq, W_.Px, W_);
            H_apply(H, W_.Px, W_.Hx);
            project_tangent_into(W_.Hx, Aeq, W_.PHx, W_);
            y = W_.PHx;
        };
        return P;
    }

    // ---------- Normal step (metric min-norm) ----------
    dvec min_norm_normal_(const dmat &A, const dvec &b) const {
        if (A.size() == 0)
            return dvec::Zero(b.size());
        if (!metric_.valid) {
            dvec y = AAT_solve(A, -b);
            return A.transpose() * y;
        }
        auto Minv = [&](const dvec &r) {
            dvec y = metric_.L.triangularView<Eigen::Lower>().solve(r);
            return metric_.L.transpose().triangularView<Eigen::Upper>().solve(
                y);
        };
        dmat AMiAT = A * Minv(A.transpose());
        dvec lam = AAT_solve(AMiAT, -b);
        return Minv(A.transpose() * lam);
    }

    // ---------- Equality path ----------
    std::pair<dvec, TRInfo> solve_with_equalities_(
        const LinOp &Hop, const dvec &g, const dmat &Aeq, const dvec &beq,
        const std::optional<dmat> &Aineq, const std::optional<dvec> &bineq,
        const std::optional<spmat> &sparseH, const std::optional<dmat> &denseH,
        const std::optional<dvec> &x, const std::optional<dvec> &lb,
        const std::optional<dvec> &ub) {
        TRInfo info;

        // Criticality in tangent (pre-normal)
        {
            auto [c, s] = maybe_apply_criticality_(g, Aeq);
            info.criticality = c;
            info.criticality_shrinks = s;
        }

        // Normal step
        dvec p_n = min_norm_normal_(Aeq, beq);
        const double zeta = cfg_.zeta;
        double nn = tr_norm_(p_n);
        if (nn > zeta * delta_)
            p_n *= (zeta * delta_ / std::max(1e-16, nn));

        // BOX: ensure normal step respects bounds
        if (x && (lb || ub)) {
            if (use_projection_()) {
                dvec xp = *x + p_n;
                if (lb)
                    xp = xp.cwiseMax(*lb);
                if (ub)
                    xp = xp.cwiseMin(*ub);
                dvec shift = xp - (*x + p_n);
                dvec beq_new = beq + Aeq * shift;
                p_n = min_norm_normal_(Aeq, beq_new);
                nn = tr_norm_(p_n);
                if (nn > zeta * delta_)
                    p_n *= (zeta * delta_ / std::max(1e-16, nn));
            } else {
                double amax = alpha_max_box_(*x, p_n, lb, ub);
                if (amax < 1.0) {
                    dvec xp = *x + p_n;
                    if (lb)
                        xp = xp.cwiseMax(*lb);
                    if (ub)
                        xp = xp.cwiseMin(*ub);
                    dvec shift = xp - (*x + p_n);
                    dvec beq_new = beq + Aeq * shift;
                    p_n = min_norm_normal_(Aeq, beq_new);
                    nn = tr_norm_(p_n);
                    if (nn > zeta * delta_)
                        p_n *= (zeta * delta_ / std::max(1e-16, nn));
                }
            }
        }

        // Tangential step
        W_.tmp = g;
        Hop.apply_into(p_n, W_.Hp);
        W_.tmp.noalias() += W_.Hp; // gtilde = g + H p_n
        auto Htan = make_projected_operator_(Hop, Aeq);
        project_tangent_into(W_.tmp, Aeq, W_.proj_buf1, W_); // gtan
        double rem = std::sqrt(
            std::max(0.0, delta_ * delta_ - tr_norm_(p_n) * tr_norm_(p_n)));

        {
            auto [c2, s2] = maybe_apply_criticality_(W_.proj_buf1, Aeq);
            info.criticality = info.criticality || c2;
            info.criticality_shrinks += s2;
            rem = std::sqrt(
                std::max(0.0, delta_ * delta_ - tr_norm_(p_n) * tr_norm_(p_n)));
        }

        Prec Pbase = make_prec_(sparseH, denseH);
        Prec Pproj = make_projected_prec_(Aeq, Pbase);
        info.preconditioned_reduced = true;

        const double tol = cg_tol_(W_.proj_buf1.norm());
        auto cg = steihaug_pcg(Htan, W_.proj_buf1, metric_, rem, tol,
                               cfg_.cg_maxiter, cfg_.neg_curv_tol, Pproj, W_);
        dvec p_t = cg.p;

        // TR clip tangential
        double pt_norm = tr_norm_(p_t);
        if (pt_norm > rem)
            p_t *= (rem / std::max(1e-16, pt_norm));

        // BOX: keep x + p_n + p_t inside
        if (x && (lb || ub)) {
            if (use_projection_()) {
                dvec xt = *x + p_n + p_t;
                if (lb)
                    xt = xt.cwiseMax(*lb);
                if (ub)
                    xt = xt.cwiseMin(*ub);
                p_t = (xt - *x) - p_n;
            } else {
                double beta = alpha_max_box_(*x + p_n, p_t, lb, ub);
                if (beta < 1.0)
                    p_t *= beta;
            }
        }

        dvec p = p_n + p_t;

        // Inequalities via active set on top of equalities
        if (Aineq && Aineq->rows() > 0) {
            auto [p2, active, inf_add] =
                active_set_loop_(Hop, g, Aeq, beq, *Aineq,
                                 bineq.value_or(dvec::Zero(Aineq->rows())), p);
            if (box_x_ && (box_lb_ || box_ub_)) {
                const dvec &x0 = *box_x_;
                dvec xt = x0 + p2;
                if (box_lb_)
                    xt = xt.cwiseMax(*box_lb_);
                if (box_ub_)
                    xt = xt.cwiseMin(*box_ub_);
                p = xt - x0; // projection-only pullback
            } else {
                p = p2;
            }
            info.active_set_indices = active;
            info.active_set_size = (int)active.size();
            info.active_set_iterations = inf_add.first;
        }

        info.iterations = 0;
        info.status = "success";
        info.step_norm = tr_norm_(p);
        info.model_reduction = model_reduction_quad_into(Hop, g, p, W_.Hp);
        info.model_reduction_quad = info.model_reduction;
        info.constraint_violation = (Aeq * p + beq).norm();
        return {p, info};
    }

    // ---------- Inequality-only path ----------
    std::pair<dvec, TRInfo> solve_with_inequalities_(
        const LinOp &Hop, const dvec &g, const dmat &Aineq, const dvec &bineq,
        const std::optional<spmat> &sparseH, const std::optional<dmat> &denseH,
        const std::optional<dvec> &x, const std::optional<dvec> &lb,
        const std::optional<dvec> &ub) {
        TRInfo info;
        {
            auto [c, s] = maybe_apply_criticality_(g, std::nullopt);
            info.criticality = c;
            info.criticality_shrinks = s;
        }

        auto P = make_prec_(sparseH, denseH);
        const double tol = cg_tol_(tr_norm_(g));
        auto cg = steihaug_pcg(Hop, g, metric_, delta_, tol, cfg_.cg_maxiter,
                               cfg_.neg_curv_tol, P, W_);
        dvec p = cg.p;

        if (x && (lb || ub))
            p = enforce_box_on_step(*x, p, lb, ub, cfg_.box_mode);

        dvec viol = Aineq * p + bineq;
        if ((viol.array() > cfg_.constraint_tol).any()) {
            dmat Aeq0(0, (int)g.size());
            dvec beq0(0);
            auto [p2, active, inf_add] =
                active_set_loop_(Hop, g, Aeq0, beq0, Aineq, bineq, p);
            if (x && (lb || ub)) {
                p2 = enforce_box_on_step(
                    *x, p2, lb, ub, use_projection_() ? "projection" : "alpha");
            }
            p = p2;
            info.active_set_indices = active;
            info.active_set_size = (int)active.size();
            info.active_set_iterations = inf_add.first;
        }

        info.iterations = cg.iters;
        info.status = status_to_string_(cg.status);
        info.step_norm = tr_norm_(p);
        info.model_reduction = model_reduction_quad_into(Hop, g, p, W_.Hp);
        info.model_reduction_quad = info.model_reduction;
        info.preconditioned = P.valid;
        return {p, info};
    }

    // ---------- Active-set loop (equality-solve parity; box pullback only)
    // ----------
    std::tuple<dvec, std::vector<int>, std::pair<int, int>>
    active_set_loop_(const LinOp &Hop, const dvec &g,
                     const std::optional<dmat> &Aeq,
                     const std::optional<dvec> &beq, const dmat &Aineq,
                     const dvec &bineq, const dvec &p_init) {
        const int n = (int)g.size();
        dvec p = p_init;
        std::set<int> active;
        int it = 0;

        auto rank_of = [&](const dmat &A) -> int {
            if (!A.size())
                return 0;
            Eigen::FullPivLU<dmat> lu(A);
            lu.setThreshold(cfg_.rcond);
            return (int)lu.rank();
        };

        auto build_cur = [&](dmat &Acur, dvec &bcur) {
            if (active.empty()) {
                if (Aeq && beq) {
                    Acur = *Aeq;
                    bcur = *beq;
                } else {
                    Acur.resize(0, Aineq.cols());
                    bcur.resize(0);
                }
                return;
            }
            dmat Aact(active.size(), Aineq.cols());
            dvec bact(active.size());
            int r = 0;
            for (int idx : active) {
                Aact.row(r) = Aineq.row(idx);
                bact[r++] = bineq[idx];
            }
            if (Aeq && beq && Aeq->size()) {
                Acur.resize(Aeq->rows() + Aact.rows(), Aeq->cols());
                bcur.resize(beq->size() + bact.size());
                Acur << *Aeq, Aact;
                bcur << *beq, bact;
            } else {
                Acur = Aact;
                bcur = bact;
            }
        };

        const int rank_eq = (Aeq && Aeq->size()) ? rank_of(*Aeq) : 0;
        const int max_active =
            std::max(0, std::min<int>((int)Aineq.rows(), n - rank_eq));

        while (it < cfg_.max_active_set_iter) {
            dvec viol = Aineq * p + bineq;
            Eigen::Array<bool, Eigen::Dynamic, 1> mask =
                (viol.array() > cfg_.constraint_tol);
            if (!(mask.any()))
                break;

            dmat Acur;
            dvec bcur;
            build_cur(Acur, bcur);
            const int rcur = Acur.size() ? rank_of(Acur) : 0;
            if (rcur >= n || (int)active.size() >= max_active)
                break;

            std::vector<int> cand;
            cand.reserve((int)viol.size());
            for (int i = 0; i < viol.size(); ++i)
                if (mask[i])
                    cand.push_back(i);
            std::sort(cand.begin(), cand.end(),
                      [&](int a, int b) { return viol[a] > viol[b]; });

            bool added = false;
            int last = -1;
            for (int idx : cand) {
                if (active.count(idx))
                    continue;
                dmat Atest;
                if (!Acur.size())
                    Atest = Aineq.row(idx);
                else {
                    Atest.resize(Acur.rows() + 1, Acur.cols());
                    Atest << Acur, Aineq.row(idx);
                }
                if (rank_of(Atest) > rcur) {
                    active.insert(idx);
                    last = idx;
                    added = true;
                    break;
                }
            }
            if (!added)
                break;

            dmat A_aug;
            dvec b_aug;
            build_cur(A_aug, b_aug);

            dvec p_eq;
            TRInfo info_eq;
            try {
                std::tie(p_eq, info_eq) = solve_with_equalities_(
                    Hop, g, A_aug, b_aug, std::nullopt, std::nullopt,
                    std::nullopt, std::nullopt, box_x_, box_lb_, box_ub_);
            } catch (...) {
                if (last >= 0)
                    active.erase(last);
                break;
            }

            // Projection-only pullback (Python parity)
            if (box_x_ && (box_lb_ || box_ub_)) {
                const dvec &x0 = *box_x_;
                dvec xt = x0 + p_eq;
                if (box_lb_)
                    xt = xt.cwiseMax(*box_lb_);
                if (box_ub_)
                    xt = xt.cwiseMin(*box_ub_);
                p = xt - x0;
            } else {
                p = p_eq;
            }
            ++it;
        }

        dvec viol = Aineq * p + bineq;
        (void)viol;
        std::vector<int> active_idx(active.begin(), active.end());
        return {p, active_idx, {it, (int)active.size()}};
    }

    // ---------- Multipliers (+ box μ) ----------
    inline std::pair<std::vector<int>, std::vector<int>>
    detect_active_bounds_(const std::optional<dvec> &x, const dvec &p,
                          const std::optional<dvec> &lb,
                          const std::optional<dvec> &ub,
                          double tol = 1e-10) const {
        std::vector<int> idxL, idxU;
        if (!(x && (lb || ub)))
            return {idxL, idxU};
        const dvec xt = *x + p;
        const int n = (int)xt.size();
        idxL.reserve(n);
        idxU.reserve(n);
        for (int i = 0; i < n; ++i) {
            if (lb && i < (int)lb->size() && xt[i] <= (*lb)[i] + tol)
                idxL.push_back(i);
            if (ub && i < (int)ub->size() && xt[i] >= (*ub)[i] - tol)
                idxU.push_back(i);
        }
        return {idxL, idxU};
    }

    // --- replace your recover_multipliers_ with this version (NOTE the new
    // arg: active_idx_opt) ---
    std::tuple<dvec, dvec, dvec, dvec> recover_multipliers_(
        const LinOp &Hop, const dvec &g, const dvec &p, double sigma,
        const std::optional<dmat> &Aeq, const std::optional<dmat> &Aineq,
        const std::optional<dvec> &x, const std::optional<dvec> &lb,
        const std::optional<dvec> &ub,
        const std::optional<std::vector<int>> &active_idx_opt // NEW
    ) {
        const int n = (int)g.size();

        // residual r = H p + g + σ p
        dvec Hp(p.size());
        Hp.setZero();
        H_apply(Hop, p, Hp);
        dvec r = Hp + g + sigma * p;

        // Aeq^T block
        dmat AeqT(n, 0);
        if (Aeq && Aeq->size())
            AeqT = Aeq->transpose();

        // Aact^T block: either all inequalities (legacy) or only active ones
        // (NEW)
        dmat AactT(n, 0);
        if (Aineq && Aineq->size()) {
            if (cfg_.recover_lam_active_only && active_idx_opt &&
                !active_idx_opt->empty()) {
                const auto &ids = *active_idx_opt;
                dmat A_sel(ids.size(), Aineq->cols());
                for (size_t k = 0; k < ids.size(); ++k)
                    A_sel.row((int)k) = Aineq->row(ids[k]);
                AactT = A_sel.transpose();
            } else {
                // legacy: use all inequalities
                AactT = Aineq->transpose();
            }
        }

        // Box activity blocks
        auto [idxL, idxU] = detect_active_bounds_(x, p, lb, ub, 1e-10);
        dmat IL(n, (int)idxL.size()), IU(n, (int)idxU.size());
        if (IL.cols()) {
            IL.setZero();
            for (int j = 0; j < IL.cols(); ++j)
                IL(idxL[j], j) = 1.0;
        }
        if (IU.cols()) {
            IU.setZero();
            for (int j = 0; j < IU.cols(); ++j)
                IU(idxU[j], j) = 1.0;
        }

        // Concatenate [Aeq^T | Aact^T | I_L | -I_U]
        int mcols = AeqT.cols() + AactT.cols() + IL.cols() + IU.cols();
        if (mcols == 0) {
            dvec lam(Aineq ? (int)Aineq->rows() : 0);
            dvec nu(Aeq ? (int)Aeq->rows() : 0);
            dvec muL = dvec::Zero(n), muU = dvec::Zero(n);
            return {lam, nu, muL, muU};
        }
        dmat AT(n, mcols);
        int ofs = 0;
        if (AeqT.cols()) {
            AT.middleCols(ofs, AeqT.cols()) = AeqT;
            ofs += AeqT.cols();
        }
        if (AactT.cols()) {
            AT.middleCols(ofs, AactT.cols()) = AactT;
            ofs += AactT.cols();
        }
        if (IL.cols()) {
            AT.middleCols(ofs, IL.cols()) = IL;
            ofs += IL.cols();
        }
        if (IU.cols()) {
            AT.middleCols(ofs, IU.cols()) = -IU;
            ofs += IU.cols();
        }

        // Normal equations solve: (ATᵀ AT) y = -ATᵀ r
        dmat N = AT.transpose() * AT;
        dvec rhs = -AT.transpose() * r;
        dvec y = N.ldlt().solve(rhs);

        // Slice back into [nu | lam_block | yL | yU]
        ofs = 0;
        dvec nu(Aeq ? (int)Aeq->rows() : 0);
        if (AeqT.cols()) {
            nu = y.segment(ofs, AeqT.cols());
            ofs += AeqT.cols();
        }

        // λ mapping
        dvec lam; // full-sized if Aineq exists
        if (Aineq && Aineq->size()) {
            lam = dvec::Zero(Aineq->rows());
            int block_cols = AactT.cols();
            if (block_cols > 0) {
                dvec lam_block =
                    y.segment(ofs, block_cols).cwiseMax(0.0); // λ ≥ 0
                if (cfg_.recover_lam_active_only && active_idx_opt &&
                    !active_idx_opt->empty()) {
                    // scatter only to active rows
                    for (int j = 0; j < (int)active_idx_opt->size(); ++j) {
                        lam[(*active_idx_opt)[j]] = lam_block[j];
                    }
                } else {
                    // legacy: block corresponds to all inequalities in order
                    lam = lam_block;
                }
                ofs += block_cols;
            }
        } else {
            lam = dvec(0);
        }

        // Box multipliers μL, μU (projected to ≥0)
        dvec muL = dvec::Zero(n), muU = dvec::Zero(n);
        if (IL.cols()) {
            dvec yL = y.segment(ofs, IL.cols());
            for (int j = 0; j < IL.cols(); ++j)
                muL[idxL[j]] = std::max(0.0, -yL[j]);
            ofs += IL.cols();
        }
        if (IU.cols()) {
            dvec yU = y.segment(ofs, IU.cols());
            for (int j = 0; j < IU.cols(); ++j)
                muU[idxU[j]] = std::max(0.0, -yU[j]);
            ofs += IU.cols();
        }

        return {lam, nu, muL, muU};
    }

    // ---------- Eval (f, theta) ----------
    std::pair<std::optional<double>, std::optional<double>>
    eval_model_f_theta_(const std::optional<py::object> &model,
                        const std::optional<dvec> &x,
                        const dvec &s = dvec()) const {
        if (!model)
            return {std::nullopt, std::nullopt};
        dvec xt = x ? (*x + s) : s;
        try {
            py::dict out = (*model).attr("eval_all")(xt).cast<py::dict>();
            std::optional<double> f, th;
            if (out.contains("f"))
                f = py::float_(out["f"]);
            if (py::hasattr(*model, "constraint_violation"))
                th = py::float_((*model).attr("constraint_violation")(xt));
            return {f, th};
        } catch (...) {
            return {std::nullopt, std::nullopt};
        }
    }

    // ---------- Backtracking after filter rejection ----------
    std::tuple<bool, dvec, std::string, std::optional<double>,
               std::optional<double>>
    _backtrack_on_reject_(const std::optional<py::object> &model,
                          const std::optional<dvec> &x, const dvec &s,
                          const std::optional<dvec> &lb,
                          const std::optional<dvec> &ub,
                          const std::optional<py::object> &filter_obj,
                          double delta, std::optional<double> base_f,
                          int max_tries = 3) const {
        const double alphas_raw[3] = {0.5, 0.25, 0.125};
        const int tries = std::min(max_tries, 3);

        for (int k = 0; k < tries; ++k) {
            const double a = alphas_raw[k];
            dvec sa = a * s;
            if (x && (lb || ub))
                sa = enforce_box_on_step(*x, sa, lb, ub, cfg_.box_mode);

            auto [f_t, th_t] = eval_model_f_theta_(model, x, sa);

            if (filter_obj && f_t && th_t && std::isfinite(*f_t) &&
                std::isfinite(*th_t)) {
                try {
                    py::object is_acc = (*filter_obj).attr("is_acceptable");
                    bool ok =
                        is_acc(*th_t, *f_t, py::arg("trust_radius") = delta)
                            .cast<bool>();
                    if (ok) {
                        py::object add_if =
                            (*filter_obj).attr("add_if_acceptable");
                        add_if(*th_t, *f_t, py::arg("trust_radius") = delta);
                        return {true, sa, "ls-filter", f_t, th_t};
                    }
                } catch (...) {
                }
            }

            if (base_f && f_t && std::isfinite(*f_t) &&
                std::isfinite(*base_f)) {
                if (*f_t <= *base_f - 1e-4 * a * std::abs(*base_f))
                    return {true, sa, "ls-armijo", f_t, th_t};
            }
        }
        return {false, s, std::string("rejected"), std::nullopt, std::nullopt};
    }

    // ---------- Pack info ----------
    py::dict pack_info_(const TRInfo &inf) const {
        py::dict d;
        d["status"] = inf.status;
        d["iterations"] = inf.iterations;
        d["step_norm"] = inf.step_norm;
        d["model_reduction"] = inf.model_reduction;
        d["model_reduction_quad"] = inf.model_reduction_quad;
        d["sigma_est"] = inf.sigma_est;
        d["predicted_reduction_cubic"] = inf.predicted_reduction_cubic;
        d["constraint_violation"] = inf.constraint_violation;
        d["preconditioned"] = inf.preconditioned;
        d["preconditioned_reduced"] = inf.preconditioned_reduced;
        d["active_set_size"] = inf.active_set_size;
        d["active_set_iterations"] = inf.active_set_iterations;
        py::list act;
        for (int i : inf.active_set_indices)
            act.append(i);
        d["active_set_indices"] = act;
        d["accepted_by"] = inf.accepted_by;
        d["accepted"] = inf.accepted;
        d["soc_applied"] = inf.soc_applied;
        d["theta0"] = inf.theta0;
        d["theta1"] = inf.theta1;
        d["criticality"] = inf.criticality;
        d["criticality_shrinks"] = inf.criticality_shrinks;
        return d;
    }

    inline std::string status_to_string_(TRStatus s) const {
        switch (s) {
        case TRStatus::SUCCESS:
            return "success";
        case TRStatus::BOUNDARY:
            return "boundary";
        case TRStatus::NEG_CURV:
            return "negative_curvature";
        default:
            return "max_iterations";
        }
    }

    // -------------------- SOC (Second-Order Correction) --------------------
    struct SOCResult {
        dvec q;
        bool applied = false;
        double theta0 = NAN, theta1 = NAN;
    };

    static std::optional<dmat> to_dense_matrix_optional(const py::handle &obj) {
        if (!obj || obj.is_none())
            return std::nullopt;
        try {
            if (py::hasattr(obj, "toarray")) {
                py::object dense = obj.attr("toarray")();
                return dense.cast<dmat>();
            }
            return obj.cast<dmat>();
        } catch (...) {
            return std::nullopt;
        }
    }
    static std::optional<dvec> to_dense_vector_optional(const py::handle &obj) {
        if (!obj || obj.is_none())
            return std::nullopt;
        try {
            return obj.cast<dvec>();
        } catch (...) {
            return std::nullopt;
        }
    }

    dvec clip_correction_to_radius_(const dvec &s, const dvec &q) const {
        if (tr_norm_(s + q) <= delta_ + 1e-14 || tr_norm_(q) <= 1e-16)
            return q;
        double t = boundary_intersection_metric(metric_, s, q, delta_);
        t = std::clamp(t, 0.0, 1.0);
        return t * q;
    }

    SOCResult soc_correction_(const py::object &model, const dvec &x,
                              const dvec &p, const LinOp &Hop,
                              const std::optional<dmat> &Hdense,
                              const std::optional<spmat> &Hsparse,
                              const dvec &lam_ineq, double mu, double wE,
                              double wI, double tolE, double violI, double reg,
                              double sigma0, const std::optional<dvec> &lb,
                              const std::optional<dvec> &ub) {
        SOCResult R;
        R.q = dvec::Zero(p.size());
        dvec x_trial = x + p;
        py::dict out = model.attr("eval_all")(x_trial).cast<py::dict>();

        py::object cE_obj =
            out.contains("cE") ? py::object(out["cE"]) : py::none();
        py::object cI_obj =
            out.contains("cI") ? py::object(out["cI"]) : py::none();
        py::object JE_obj =
            out.contains("JE") ? py::object(out["JE"]) : py::none();
        py::object JI_obj =
            out.contains("JI") ? py::object(out["JI"]) : py::none();

        auto cE_ = to_dense_vector_optional(cE_obj);
        auto cI_ = to_dense_vector_optional(cI_obj);
        auto JE_ = to_dense_matrix_optional(JE_obj);
        auto JI_ = to_dense_matrix_optional(JI_obj);

        std::vector<dmat> rows;
        std::vector<dvec> rhs;
        std::vector<dvec> wts;

        if (cE_ && JE_ && JE_->size()) {
            Eigen::Array<bool, Eigen::Dynamic, 1> mask =
                (cE_->array().abs() > tolE);
            int m = (int)mask.count();
            if (m > 0) {
                dmat JEsel(m, JE_->cols());
                dvec rsel(m), w(m);
                for (int i = 0, r = 0; i < cE_->size(); ++i)
                    if (mask[i]) {
                        JEsel.row(r) = JE_->row(i);
                        rsel[r] = -(*cE_)[i];
                        w[r++] = wE;
                    }
                rows.push_back(JEsel);
                rhs.push_back(rsel);
                wts.push_back(w);
            }
        }
        if (cI_ && JI_ && JI_->size()) {
            Eigen::Array<bool, Eigen::Dynamic, 1> mask = (cI_->array() > violI);
            int m = (int)mask.count();
            if (m > 0) {
                dmat JIsel(m, JI_->cols());
                dvec rsel(m), w(m);
                for (int i = 0, r = 0; i < cI_->size(); ++i)
                    if (mask[i]) {
                        JIsel.row(r) = JI_->row(i);
                        double rhsI = -(*cI_)[i];
                        if (mu > 0.0 && lam_ineq.size() == cI_->size()) {
                            double lam = std::max(lam_ineq[i], 1e-12);
                            rhsI = -((*cI_)[i] + (mu / lam));
                        }
                        rsel[r] = rhsI;
                        w[r++] = wI;
                    }
                rows.push_back(JIsel);
                rhs.push_back(rsel);
                wts.push_back(w);
            }
        }
        if (rows.empty())
            return R;

        int mTot = 0, n = (int)p.size();
        for (auto &A : rows)
            mTot += (int)A.rows();
        dmat J(mTot, n);
        dvec r(mTot), w(mTot);
        for (int k = 0, ofs = 0; k < (int)rows.size(); ++k) {
            int mk = (int)rows[k].rows();
            J.middleRows(ofs, mk) = rows[k];
            r.segment(ofs, mk) = rhs[k];
            w.segment(ofs, mk) = wts[k].array().max(1e-16);
            ofs += mk;
        }
        dvec sqrtw = w.array().sqrt();
        dmat Jw = J.array().colwise() * sqrtw.array();
        dvec rw = r.array() * sqrtw.array();

        dvec q = dvec::Zero(n);

        bool used_metric = (cfg_.norm_type == "ellip");
        if (used_metric && (Hdense || Hsparse)) {
            dmat Hd = Hdense ? *Hdense : dmat(*Hsparse);
            dmat A = symmetrize(Hd) + sigma0 * dmat::Identity(n, n);
            dmat L = psd_cholesky_with_shift(A, cfg_.metric_shift);
            auto Minv = [&](const dvec &v) {
                dvec y = L.triangularView<Eigen::Lower>().solve(v);
                return L.transpose().triangularView<Eigen::Upper>().solve(y);
            };
            dmat JT = Jw.transpose();
            dmat MinvJT(JT.rows(), JT.cols());
            for (int i = 0; i < JT.cols(); ++i)
                MinvJT.col(i) = Minv(JT.col(i));
            dmat S = Jw * MinvJT;
            double lam_reg =
                reg * std::max(1.0, S.trace() / std::max(1, (int)S.rows()));
            S.diagonal().array() += lam_reg;
            dvec y = S.ldlt().solve(rw);
            q = -Minv(Jw.transpose() * y);
        } else {
            int m = (int)Jw.rows(), nJ = (int)Jw.cols();
            if (m >= nJ) {
                dmat JtJ = Jw.transpose() * Jw;
                double lam_reg = reg * (JtJ.trace() / std::max(1, nJ));
                JtJ.diagonal().array() += lam_reg;
                q = -(JtJ.ldlt().solve(Jw.transpose() * rw));
            } else {
                dmat JJt = Jw * Jw.transpose();
                double lam_reg = reg * (JJt.trace() / std::max(1, m));
                JJt.diagonal().array() += lam_reg;
                dvec y = JJt.ldlt().solve(rw);
                q = -(Jw.transpose() * y);
            }
        }

        q = clip_correction_to_radius_(p, q);

        if (cI_ && JI_ && JI_->size()) {
            dvec inc = (*JI_) * q;
            bool any_pos = false;
            for (int i = 0; i < inc.size(); ++i)
                if (inc[i] > 0.0) {
                    any_pos = true;
                    break;
                }
            if (any_pos) {
                double alpha = 1.0;
                for (int i = 0; i < inc.size(); ++i)
                    if (inc[i] > 0.0) {
                        double safe = std::max(0.0, -(*cI_)[i] / inc[i]);
                        alpha = std::min(alpha, 0.99 * safe);
                    }
                if (!std::isfinite(alpha))
                    return R;
                if (alpha < 1.0)
                    q *= alpha;
            }
        }

        if (lb || ub) {
            if (use_projection_()) {
                dvec xt = x_trial + q;
                if (lb)
                    xt = xt.cwiseMax(*lb);
                if (ub)
                    xt = xt.cwiseMin(*ub);
                q = xt - x_trial;
            } else {
                double amax = alpha_max_box_(x_trial, q, lb, ub);
                if (amax < 1.0)
                    q *= amax;
            }
        }

        double th0 = model.attr("constraint_violation")(x_trial).cast<double>();
        double th1 =
            model.attr("constraint_violation")(x_trial + q).cast<double>();
        R.q = (th1 < 0.9 * th0) ? q : dvec::Zero(n);
        R.applied = (th1 < 0.9 * th0);
        R.theta0 = th0;
        R.theta1 = th1;
        return R;
    }
};

// -------------------- PYBIND11 --------------------
PYBIND11_MODULE(tr_cpp, m) {
    m.doc() =
        "Trust-region (Steihaug–Toint) with Eigen + pybind11 + SOC correction";

    py::class_<TRConfig>(m, "TRConfig")
        .def(py::init<>())
        .def_readwrite("delta0", &TRConfig::delta0)
        .def_readwrite("delta_min", &TRConfig::delta_min)
        .def_readwrite("delta_max", &TRConfig::delta_max)
        .def_readwrite("cg_tol", &TRConfig::cg_tol)
        .def_readwrite("cg_tol_rel", &TRConfig::cg_tol_rel)
        .def_readwrite("cg_maxiter", &TRConfig::cg_maxiter)
        .def_readwrite("neg_curv_tol", &TRConfig::neg_curv_tol)
        .def_readwrite("rcond", &TRConfig::rcond)
        .def_readwrite("metric_shift", &TRConfig::metric_shift)
        .def_readwrite("zeta", &TRConfig::zeta)
        .def_readwrite("constraint_tol", &TRConfig::constraint_tol)
        .def_readwrite("max_active_set_iter", &TRConfig::max_active_set_iter)
        .def_readwrite("use_prec", &TRConfig::use_prec)
        .def_readwrite("prec_kind", &TRConfig::prec_kind)
        .def_readwrite("norm_type", &TRConfig::norm_type)
        .def_readwrite("ssor_omega", &TRConfig::ssor_omega);

    py::class_<TrustRegionManager>(m, "TrustRegionManager")
        .def(py::init<const TRConfig &>(), py::arg("config"))
        .def("set_metric_from_H_dense",
             &TrustRegionManager::set_metric_from_H_dense)
        .def_readwrite("delta", &TrustRegionManager::delta_)

        .def("set_metric_from_H_sparse",
             &TrustRegionManager::set_metric_from_H_sparse)
        .def("solve_dense", &TrustRegionManager::solve_dense, py::arg("H"),
             py::arg("g"), py::arg("Aineq") = py::none(),
             py::arg("bineq") = py::none(), py::arg("Aeq") = py::none(),
             py::arg("beq") = py::none(), py::arg("model") = py::none(),
             py::arg("x") = py::none(), py::arg("lb") = py::none(),
             py::arg("ub") = py::none(), py::arg("mu") = 0.0,
             py::arg("filter") = py::none(), // NEW
             py::arg("f_old") = py::none())
        .def("solve_sparse", &TrustRegionManager::solve_sparse, py::arg("H"),
             py::arg("g"), py::arg("Aineq") = py::none(),
             py::arg("bineq") = py::none(), py::arg("Aeq") = py::none(),
             py::arg("beq") = py::none(), py::arg("model") = py::none(),
             py::arg("x") = py::none(), py::arg("lb") = py::none(),
             py::arg("ub") = py::none(), py::arg("mu") = 0.0,
             py::arg("filter") = py::none(), // NEW
             py::arg("f_old") = py::none())

        // convenience overload with underscored names
        .def(
            "solve_dense",
            [](TrustRegionManager &self, const dmat &H, const dvec &g,
               py::object A_ineq, py::object b_ineq, py::object A_eq,
               py::object b_eq, py::object model, py::object x, py::object lb,
               py::object ub, double mu, py::object filter, py::object f_old) {
                auto optMat = [](py::object o) -> std::optional<dmat> {
                    return o.is_none() ? std::nullopt
                                       : std::optional<dmat>(o.cast<dmat>());
                };
                auto optVec = [](py::object o) -> std::optional<dvec> {
                    return o.is_none() ? std::nullopt
                                       : std::optional<dvec>(o.cast<dvec>());
                };
                auto optObj = [](py::object o) -> std::optional<py::object> {
                    return o.is_none() ? std::nullopt
                                       : std::optional<py::object>(o);
                };
                auto optF = [](py::object o) -> std::optional<double> {
                    return o.is_none()
                               ? std::nullopt
                               : std::optional<double>(o.cast<double>());
                };

                return self.solve_dense(
                    H, g, optMat(A_ineq), optVec(b_ineq), optMat(A_eq),
                    optVec(b_eq), optObj(model), optVec(x), optVec(lb),
                    optVec(ub), mu, optObj(filter), optF(f_old));
            },
            py::arg("H"), py::arg("g"), py::arg("A_ineq") = py::none(),
            py::arg("b_ineq") = py::none(), py::arg("A_eq") = py::none(),
            py::arg("b_eq") = py::none(), py::arg("model") = py::none(),
            py::arg("x") = py::none(), py::arg("lb") = py::none(),
            py::arg("ub") = py::none(), py::arg("mu") = 0.0,
            py::arg("filter") = py::none(), py::arg("f_old") = py::none());
}
