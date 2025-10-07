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
#include "tr_aux.h"

// If you expose nanobind types/errors in ModelC implementations
namespace nb = nanobind;

// QDLDL (modern C++ wrapper; adjust header as needed in your tree)
#include "qdldl.h"

// -----------------------------------------------------------------------------
// Box handling — unified
// -----------------------------------------------------------------------------
enum class BoxMode { Projection, Alpha };

struct BoxCtx {
    std::optional<dvec> x, lb, ub;
    BoxMode mode{BoxMode::Alpha};
    [[nodiscard]] inline bool active() const noexcept {
        return x && (lb || ub);
    }

    static double alpha_max(const dvec &x, const dvec &d,
                            const std::optional<dvec> &lb,
                            const std::optional<dvec> &ub,
                            double tau = 0.999999) {
        if ((!lb && !ub) || d.size() == 0)
            return 1.0;
        double amax = 1.0;
        if (lb) {
            for (int i = 0; i < d.size(); ++i)
                if (d[i] < 0.0)
                    amax = std::min(amax, ((*lb)[i] - x[i]) / d[i]);
        }
        if (ub) {
            for (int i = 0; i < d.size(); ++i)
                if (d[i] > 0.0)
                    amax = std::min(amax, ((*ub)[i] - x[i]) / d[i]);
        }
        amax = std::clamp(amax, 0.0, 1.0);
        return std::clamp(tau * amax, 0.0, 1.0);
    }

    static dvec pullback(const dvec &x, const dvec &p,
                         const std::optional<dvec> &lb,
                         const std::optional<dvec> &ub) {
        dvec xt = x + p;
        if (lb)
            xt = xt.cwiseMax(*lb);
        if (ub)
            xt = xt.cwiseMin(*ub);
        return xt - x;
    }

    [[nodiscard]] dvec enforce_step(const dvec &p) const {
        if (!active())
            return p;
        const dvec &xx = *x;
        if (mode == BoxMode::Projection)
            return pullback(xx, p, lb, ub);
        const double a = alpha_max(xx, p, lb, ub);
        return a * p;
    }

    [[nodiscard]] dvec enforce_correction(const dvec &x_trial,
                                          const dvec &q) const {
        if (!active())
            return q;
        if (mode == BoxMode::Projection) {
            dvec xc = x_trial + q;
            if (lb)
                xc = xc.cwiseMax(*lb);
            if (ub)
                xc = xc.cwiseMin(*ub);
            return xc - x_trial;
        }
        const double a = alpha_max(x_trial, q, lb, ub);
        return a * q;
    }
};

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------
[[nodiscard]] inline dmat symmetrize(const dmat &A) {
    return 0.5 * (A + A.transpose());
}

[[nodiscard]] inline dmat psd_cholesky_with_shift(const dmat &S,
                                                  double shift_min) {
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

[[nodiscard]] inline double model_reduction_quad_into(const LinOp &H,
                                                      const dvec &g,
                                                      const dvec &p, dvec &Hp) {
    H_apply(H, p, Hp);
    return -(g.dot(p) + 0.5 * p.dot(Hp));
}

[[nodiscard]] inline double estimate_sigma_into(const LinOp &H, const Metric &M,
                                                const dvec &g, const dvec &p,
                                                dvec &Hp) {
    if (p.size() == 0)
        return 0.0;
    double den = std::pow(M.norm(p), 2);
    if (den <= 1e-14)
        return 0.0;
    H_apply(H, p, Hp);
    return std::max(0.0, -(p.dot(Hp) + p.dot(g)) / den);
}

[[nodiscard]] inline double pred_red_cubic_into(const LinOp &H, const Metric &M,
                                                const dvec &g, const dvec &p,
                                                double sigma, dvec &Hp) {
    H_apply(H, p, Hp);
    const double quad = g.dot(p) + 0.5 * p.dot(Hp);
    return -(quad + (sigma / 3.0) * std::pow(M.norm(p), 3));
}

// -----------------------------------------------------------------------------
// AAT cached solve for projection — QDLDL version (robust; int32 guaranteed)
// -----------------------------------------------------------------------------
namespace {
inline dvec solve_sym_spd_qdldl(const dmat &N_in, const dvec &b_in) {
    using namespace qdldl23;
    using FloatT = double;
    using IntT = int32_t;

    const int m = static_cast<int>(N_in.rows());
    if (m == 0)
        return dvec(0);

    // Pack the (dense) upper triangle of N into CSC expected by QDLDL
    SparseD32 S;
    S.n = static_cast<IntT>(m);
    S.Ap.resize(static_cast<size_t>(m) + 1);
    S.Ap[0] = 0;
    for (IntT j = 0; j < (IntT)m; ++j)
        S.Ap[(size_t)j + 1] = S.Ap[(size_t)j] + (j + 1); // 0..j

    const size_t nnz = (size_t)S.Ap.back();
    S.Ai.resize(nnz);
    S.Ax.resize(nnz);

    for (IntT j = 0; j < (IntT)m; ++j) {
        IntT p = S.Ap[(size_t)j];
        for (IntT i = 0; i <= j; ++i, ++p) {
            S.Ai[(size_t)p] = i;
            S.Ax[(size_t)p] = N_in(i, j);
        }
    }
    S.finalize_upper_inplace(/*require_diag=*/true);

    try {
        const Symb32 sym = analyze_fast(S);
        auto F = refactorize<FloatT, IntT>(S, sym);

        dvec x = b_in; // will be overwritten with solution
        solve<FloatT, IntT>(F, x.data());

        // A couple of refinement sweeps (cheap for small m)
        refine<FloatT, IntT>(S, F, x.data(), b_in.data(), /*iters=*/2,
                             static_cast<const Ordering<IntT> *>(nullptr));

        return x;
    } catch (...) {
        // Dense fallbacks (should be rare)
        Eigen::LLT<dmat> llt(N_in);
        if (llt.info() == Eigen::Success)
            return llt.solve(b_in);
        Eigen::LDLT<dmat> ldlt(N_in);
        if (ldlt.info() == Eigen::Success)
            return ldlt.solve(b_in);
        return N_in.partialPivLu().solve(b_in);
    }
}
} // namespace

[[nodiscard]] inline dvec AAT_solve(const dmat &A, const dvec &rhs,
                                    double reg_floor = 1e-12) {
    using namespace qdldl23;
    using FloatT = double;
    using IntT = int32_t;

    struct Cache {
        int rows = -1, cols = -1;
        double fnorm = NAN, dsum = NAN, reg = NAN;
        Symb32 S;
        SparseD32 pattern; // upper CSC pattern
        bool analyzed = false;
        bool valid = false;
    };
    static thread_local Cache C;

    auto nearly = [](double a, double b, double t = 1e-10) {
        return std::abs(a - b) <=
               t * (1.0 + std::max(std::abs(a), std::abs(b)));
    };

    const int m = (int)A.rows();
    const int n = (int)A.cols();
    (void)n;

    const double reg = std::max(reg_floor, 1e-12);
    const double fnorm = A.norm();
    const double dsum = A.rowwise().squaredNorm().sum();

    const bool reuse = C.valid && C.rows == m && C.cols == (int)A.cols() &&
                       nearly(C.fnorm, fnorm) && nearly(C.dsum, dsum) &&
                       nearly(C.reg, reg, 1e-14);

    if (!reuse) {
        C.rows = m;
        C.cols = (int)A.cols();
        C.fnorm = fnorm;
        C.dsum = dsum;
        C.reg = reg;

        // Build full upper-triangular pattern (with diag)
        SparseD32 P;
        P.n = static_cast<IntT>(m);
        P.Ap.resize((size_t)m + 1);
        P.Ap[0] = 0;
        for (IntT j = 0; j < (IntT)m; ++j)
            P.Ap[(size_t)j + 1] = P.Ap[(size_t)j] + (j + 1);
        const size_t nnz = (size_t)P.Ap[(size_t)m];
        P.Ai.resize(nnz);
        P.Ax.resize(nnz);
        for (IntT j = 0; j < (IntT)m; ++j) {
            IntT p = P.Ap[(size_t)j];
            for (IntT i = 0; i <= j; ++i)
                P.Ai[(size_t)p++] = i;
        }
        P.finalize_upper_inplace(/*require_diag=*/true);

        C.S = analyze_fast(P);
        C.pattern = std::move(P);
        C.analyzed = true;
        C.valid = true;
    }

    // M = A*Aᵀ + reg I
    dmat M = A * A.transpose();
    M.diagonal().array() += reg;

    // Inject values into persistent CSC buffer (upper packed)
    {
        auto &Ap = C.pattern.Ap;
        auto &Ai = C.pattern.Ai;
        auto &Ax = C.pattern.Ax;
        for (int j = 0; j < m; ++j) {
            int p = Ap[(size_t)j];
            for (int i = 0; i <= j; ++i, ++p)
                Ax[(size_t)p] = M(i, j);
        }
    }

    auto F = qdldl23::refactorize<FloatT, IntT>(C.pattern, C.S);
    dvec x = rhs;
    qdldl23::solve<FloatT, IntT>(F, x.data());
    qdldl23::refine<FloatT, IntT>(
        C.pattern, F, x.data(), rhs.data(), /*iters=*/1,
        static_cast<const qdldl23::Ordering<IntT> *>(nullptr));
    return x;
}

inline void project_tangent_into(const dvec &v, const dmat &Aeq, dvec &out) {
    if (Aeq.size() == 0) {
        out = v;
        return;
    }
    const dvec rhs = Aeq * v;
    dvec y = AAT_solve(Aeq, rhs);
    out.noalias() = v - Aeq.transpose() * y;
}
inline dvec project_tangent(const dvec &v, const dmat &Aeq) {
    dvec out(v.size());
    project_tangent_into(v, Aeq, out);
    return out;
}

// -----------------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------------
struct TRConfig {

    std::string tr_method = "auto"; // "auto" | "pcg" | "dogleg" | "gltr"
    int dogleg_n_max = 300;         // use dogleg if n <= this and dense SPD
    int gltr_n_min = 200;           // use GLTR if n >= this
    double gltr_cond_trigger = 1e8; // or when cg stalls / high cond estimate
    bool assume_gn_spd = true;      // okay to treat GN H as SPD for dogleg

    double delta0 = 1.0, delta_min = 1e-10, delta_max = 1e3;
    double cg_tol = 1e-8, cg_tol_rel = 1e-4;
    int cg_maxiter = 200;
    double neg_curv_tol = 1e-14, rcond = 1e-12, metric_shift = 1e-10;
    double zeta = 0.8, constraint_tol = 1e-8;
    int max_active_set_iter = 8;
    bool use_prec = true;
    std::string prec_kind = "ssor";
    std::string norm_type = "euclid";
    double ssor_omega = 1.0;
    double gamma1 = 0.5, gamma2 = 2.0, eta1 = 0.1, eta2 = 0.9;
    bool curvature_aware = true;

    // criticality
    bool criticality_enabled = true;
    int max_crit_shrinks = 1;
    double kappa_g = 1e-2;
    double theta_crit = 0.5;
    double kkt_residual_tol = 1e-6, complementarity_tol = 1e-8,
           licq_threshold = 1e-10;
    bool use_kkt_criticality = true, use_curvature_criticality = true;
    double curvature_criticality_tol = 1e-6;

    // box
    std::string box_mode = "alpha"; // "alpha" | "projection"
    bool recover_lam_active_only = true;

    // filter
    bool use_filter = true;
    FilterConfig filter_cfg{};

    // SOC / funnel
    double soc_theta_reduction = 0.9;
    double soc_radius_fraction = 0.5;
    bool soc_use_funnel = true;
    double funnel_gamma = 0.01;

    // extras
    double jacobian_reg_min = 1e-12;
    double constraint_weight = 0.3;
    bool adaptive_eta_thresholds = true;
    double eta_adaptive_factor = 0.1;
    bool constraint_based_growth = false;
    double max_jacobian_condition = 1e10;
};

// -----------------------------------------------------------------------------
// TrustRegionManager
// -----------------------------------------------------------------------------
class TrustRegionManager {
public:
    explicit TrustRegionManager(const TRConfig &cfg)
        : cfg_(cfg), delta_(cfg.delta0), filter_(cfg.filter_cfg),
          filter_enabled_(cfg.use_filter) {
        metric_.valid = false;
    }
    TrustRegionManager()
        : cfg_(), delta_(cfg_.delta0), filter_(cfg_.filter_cfg),
          filter_enabled_(cfg_.use_filter) {
        metric_.valid = false;
    }

    TRBackend choose_backend_(int n, bool has_dense_H, bool spd_hint,
                              double cond_hint) const {
        if (cfg_.tr_method == "dogleg")
            return TRBackend::DOGLEG;
        if (cfg_.tr_method == "pcg")
            return TRBackend::PCG;
        if (cfg_.tr_method == "gltr")
            return TRBackend::GLTR;

        // auto:
        if (has_dense_H && n <= cfg_.dogleg_n_max && spd_hint)
            return TRBackend::DOGLEG;

        if (n >= cfg_.gltr_n_min && (cond_hint >= cfg_.gltr_cond_trigger))
            return TRBackend::GLTR;

        return TRBackend::PCG;
    }

    // ---------- NEW: dense nullspace basis for Aeq (thin, stable) ----------
    std::pair<dmat, int> nullspace_basis_(const dmat &Aeq) const {
        // Build Q2 s.t. Aeq * (Q2 * z) = 0 for any z
        if (Aeq.size() == 0) {
            return {dmat::Identity((int)Aeq.cols(), (int)Aeq.cols()),
                    (int)Aeq.cols()};
        }
        const int m = (int)Aeq.rows();
        const int n = (int)Aeq.cols();

        // QR of Aeq^T: Aeq^T = Q [R; 0]
        Eigen::HouseholderQR<dmat> qr(Aeq.transpose());
        dmat Q = qr.householderQ() * dmat::Identity(n, n);

        // rank via R diagonal (robust threshold already in config)
        const int r = std::min(m, n);
        // take "nullspace" block Q2 = Q.rightCols(n - rank)
        // Use a crude rank estimate from R (explicitly form R to inspect diag)
        dmat R = qr.matrixQR().template triangularView<Eigen::Upper>();
        int rank = 0;
        const double tol = cfg_.licq_threshold;
        const int diagN = std::min(r, (int)R.rows());
        for (int i = 0; i < diagN; ++i)
            if (std::abs(R(i, i)) > tol)
                ++rank;

        const int k = std::max(0, n - rank);
        if (k == 0) {
            // fully constrained: nullspace is {0}
            return {dmat(n, 0), 0};
        }
        dmat Q2 = Q.rightCols(k);
        return {Q2, k};
    }

    inline void enable_filter(bool on) { filter_enabled_ = on; }
    inline void reset_filter() { filter_.reset(); }

    // Unified metric APIs
    template <class Mat> void set_metric_from_H(const Mat &H) {
        if (cfg_.norm_type != "ellip") {
            metric_.valid = false;
            return;
        }
        set_metric_from_dense_spd_(to_metric_matrix_(H, cfg_.metric_shift));
    }
    template <class Mat> void update_metric_from_H(const Mat &H) {
        if (cfg_.norm_type != "ellip") {
            metric_.valid = false;
            return;
        }
        update_metric_from_dense_spd_(to_metric_matrix_(H, cfg_.metric_shift));
    }
    inline void set_metric_from_H_dense(const dmat &H) { set_metric_from_H(H); }
    inline void set_metric_from_H_sparse(const spmat &H) {
        set_metric_from_H(H);
    }
    inline void update_metric_from_H_dense(const dmat &H) {
        update_metric_from_H(H);
    }
    inline void update_metric_from_H_sparse(const spmat &H) {
        update_metric_from_H(H);
    }

    [[nodiscard]] inline double get_delta() const noexcept { return delta_; }
    inline void set_delta(double d) {
        delta_ = std::clamp(d, cfg_.delta_min, cfg_.delta_max);
    }

    struct HContext {
        std::optional<dmat> Hdense;
        std::optional<spmat> Hsparse;
    };

    // Primary entry
    TRResult solve(const LinOp &H, const dvec &g,
                   const std::optional<dmat> &Aineq = std::nullopt,
                   const std::optional<dvec> &bineq = std::nullopt,
                   const std::optional<dmat> &Aeq = std::nullopt,
                   const std::optional<dvec> &beq = std::nullopt,
                   ModelC *model = nullptr, const BoxCtx &box = {},
                   double mu = 0.0,
                   const std::optional<double> &f_old = std::nullopt,
                   const HContext &HC = {}) {
        box_ = box;
        return core_solve_(H, g, Aineq, bineq, Aeq, beq, HC.Hsparse, HC.Hdense,
                           model, box.x, box.lb, box.ub, mu, f_old);
    }

    ModelC *model_ = nullptr; // optional callback handle

    // Convenience wrappers
    TRResult solve_dense(const dmat &H, const dvec &g,
                         const std::optional<dmat> &Aineq = std::nullopt,
                         const std::optional<dvec> &bineq = std::nullopt,
                         const std::optional<dmat> &Aeq = std::nullopt,
                         const std::optional<dvec> &beq = std::nullopt,
                         ModelC *model = nullptr,
                         const std::optional<dvec> &x = std::nullopt,
                         const std::optional<dvec> &lb = std::nullopt,
                         const std::optional<dvec> &ub = std::nullopt,
                         double mu = 0.0,
                         const std::optional<double> &f_old = std::nullopt) {
        LinOp Hop{
            .mv = [&H](const dvec &in, dvec &out) { out.noalias() = H * in; },
            .n = (int)g.size()};
        BoxCtx box;
        box.x = x;
        box.lb = lb;
        box.ub = ub;
        box.mode = (cfg_.box_mode == "projection" ? BoxMode::Projection
                                                  : BoxMode::Alpha);
        return solve(Hop, g, Aineq, bineq, Aeq, beq, model, box, mu, f_old,
                     HContext{H, std::nullopt});
    }

    TRResult solve_sparse(const spmat &H, const dvec &g,
                          const std::optional<dmat> &Aineq = std::nullopt,
                          const std::optional<dvec> &bineq = std::nullopt,
                          const std::optional<dmat> &Aeq = std::nullopt,
                          const std::optional<dvec> &beq = std::nullopt,
                          ModelC *model = nullptr,
                          const std::optional<dvec> &x = std::nullopt,
                          const std::optional<dvec> &lb = std::nullopt,
                          const std::optional<dvec> &ub = std::nullopt,
                          double mu = 0.0,
                          const std::optional<double> &f_old = std::nullopt) {
        LinOp Hop{
            .mv = [&H](const dvec &in, dvec &out) { out.noalias() = H * in; },
            .n = (int)g.size()};
        BoxCtx box;
        box.x = x;
        box.lb = lb;
        box.ub = ub;
        box.mode = (cfg_.box_mode == "projection" ? BoxMode::Projection
                                                  : BoxMode::Alpha);
        return solve(Hop, g, Aineq, bineq, Aeq, beq, model, box, mu, f_old,
                     HContext{std::nullopt, H});
    }

private:
    // ---------------- state ----------------
    Filter filter_;
    bool filter_enabled_;
    TRConfig cfg_;
    Metric metric_;
    mutable TRWorkspace W_;
    mutable double current_theta_ = std::numeric_limits<double>::infinity();
    mutable double jacobian_condition_ = 1.0;
    mutable std::vector<double> recent_rhos_;
    mutable BoxCtx box_{};
    double delta_;
    mutable double current_kkt_residual_ =
        std::numeric_limits<double>::quiet_NaN();

    // --------------- metric core ---------------
    template <class Mat>
    static dmat to_metric_matrix_(const Mat &H, double shift) {
        dmat Msym = dmat(H);
        Msym = 0.5 * (Msym + Msym.transpose());
        Msym.diagonal().array() += shift;
        return Msym;
    }

    void set_metric_from_dense_spd_(const dmat &Mspd) {
        if (cfg_.norm_type != "ellip") {
            metric_.valid = false;
            return;
        }
        const dmat Msym = 0.5 * (Mspd + Mspd.transpose());
        metric_.L = psd_cholesky_with_shift(Msym, cfg_.metric_shift);
        metric_.M = Msym;
        metric_.valid = true;
    }

    void update_metric_from_dense_spd_(const dmat &M_new_in) {
        if (cfg_.norm_type != "ellip") {
            metric_.valid = false;
            return;
        }
        if (!metric_.valid || metric_.L.size() == 0 || metric_.M.size() == 0) {
            set_metric_from_dense_spd_(M_new_in);
            return;
        }
        const dmat M_new = 0.5 * (M_new_in + M_new_in.transpose());
        const dmat Delta =
            0.5 * ((M_new - metric_.M) + (M_new - metric_.M).transpose());
        const double base = std::max(1.0, metric_.M.norm());
        const double rel = Delta.norm() / base;
        if (!std::isfinite(rel) || rel <= 1e-12)
            return;

        dmat A = metric_.L * metric_.L.transpose();
        Eigen::LLT<dmat> llt(A);
        if (llt.info() != Eigen::Success) {
            set_metric_from_dense_spd_(M_new);
            return;
        }

        Eigen::SelfAdjointEigenSolver<dmat> es(Delta);
        if (es.info() != Eigen::Success) {
            set_metric_from_dense_spd_(M_new);
            return;
        }
        const dvec evals = es.eigenvalues();
        const dmat U = es.eigenvectors();
        std::vector<int> order((size_t)evals.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return std::abs(evals[a]) > std::abs(evals[b]);
        });
        const double tol = 1e-12 * (Delta.norm() + 1.0);
        for (int idx : order) {
            const double lam = evals[idx];
            if (std::abs(lam) <= tol)
                continue;
            const double sigma = (lam >= 0.0) ? +1.0 : -1.0;
            dvec w = std::sqrt(std::abs(lam)) * U.col(idx);
            llt.rankUpdate(w, sigma);
            if (llt.info() != Eigen::Success) {
                set_metric_from_dense_spd_(M_new);
                return;
            }
        }
        metric_.L = llt.matrixL();
        metric_.M = M_new;
        metric_.valid = true;
    }

    // --------------- helpers ---------------
    [[nodiscard]] inline bool use_projection_() const noexcept {
        return (cfg_.box_mode == "projection");
    }
    [[nodiscard]] inline double tr_norm_(const dvec &p) const {
        return metric_.norm(p);
    }
    [[nodiscard]] inline double cg_tol_(double gnorm) const {
        return std::min(cfg_.cg_tol, cfg_.cg_tol_rel * std::max(gnorm, 1e-16));
    }

    [[nodiscard]] double
    projected_grad_norm_(const dvec &g, const std::optional<dmat> &Aeq) const {
        dvec gp = g;
        if (box_.active()) {
            const dvec &x = *box_.x;
            constexpr double tol = 1e-12;
            if (box_.lb) {
                const dvec &lb = *box_.lb;
                const int n = static_cast<int>(gp.size());
                for (int i = 0; i < n; ++i)
                    if (i < lb.size() && x[i] <= lb[i] + tol && gp[i] > 0.0)
                        gp[i] = 0.0;
            }
            if (box_.ub) {
                const dvec &ub = *box_.ub;
                const int n = static_cast<int>(gp.size());
                for (int i = 0; i < n; ++i)
                    if (i < ub.size() && x[i] >= ub[i] - tol && gp[i] < 0.0)
                        gp[i] = 0.0;
            }
        }
        if (Aeq && Aeq->size()) {
            project_tangent_into(gp, *Aeq, W_.tmp);
            return tr_norm_(W_.tmp);
        }
        return tr_norm_(gp);
    }

    // ---------------- criticality machinery ----------------
    [[nodiscard]] std::pair<bool, int>
    maybe_apply_criticality_(const dvec &g, const std::optional<dmat> &Aeq,
                             const std::optional<dmat> &Aineq = std::nullopt,
                             const std::optional<dvec> &bineq = std::nullopt,
                             const std::optional<LinOp> &Hop = std::nullopt) {
        if (!cfg_.criticality_enabled)
            return {false, 0};
        bool criticality = false;
        int shrinks = 0;

        for (int k = 0; k < cfg_.max_crit_shrinks; ++k) {
            bool any = false;

            // projected gradient
            const double pg = projected_grad_norm_(g, Aeq);
            if (pg <= cfg_.kappa_g * std::max(delta_, 1e-16))
                any = true;

            // KKT residual
            if (cfg_.use_kkt_criticality && Hop && Aineq && bineq) {
                const double kkt =
                    compute_kkt_residual_(g, Aeq, *Aineq, *bineq, *Hop);
                current_kkt_residual_ = kkt;
                if (kkt <= cfg_.kkt_residual_tol)
                    any = true;
            }

            // curvature
            if (cfg_.use_curvature_criticality && Hop) {
                if (check_curvature_criticality_(g, Aeq, *Hop))
                    any = true;
            }

            // simple degeneracy check
            if (Aineq && bineq && check_constraint_degeneracy_(*Aineq, *bineq))
                any = true;

            if (any) {
                delta_ = std::max(cfg_.delta_min, cfg_.theta_crit * delta_);
                ++shrinks;
                criticality = true;
            } else
                break;
        }
        return {criticality, shrinks};
    }

    bool check_second_order_criticality(const LinOp &H, const dvec &g,
                                        const std::optional<dmat> &Aeq) {
        dvec pg = project_tangent(g, Aeq ? *Aeq : dmat(0, g.size()));
        if (tr_norm_(pg) > cfg_.kkt_residual_tol)
            return false;
        dvec d = dvec::Random(g.size()).normalized();
        if (Aeq && Aeq->size())
            project_tangent_into(d, *Aeq, d);
        dvec Hd;
        H_apply(H, d, Hd);
        if (Aeq && Aeq->size())
            project_tangent_into(Hd, *Aeq, Hd);
        return d.dot(Hd) >= -cfg_.curvature_criticality_tol;
    }

    [[nodiscard]] double compute_kkt_residual_(const dvec &g,
                                               const std::optional<dmat> &Aeq,
                                               const dmat &Aineq,
                                               const dvec &bineq,
                                               const LinOp &Hop) {
        const int n = (int)g.size();
        const dvec p0 = dvec::Zero(n);

        // multipliers at p=0
        auto [lam, nu, muL, muU] = recover_multipliers_(
            Hop, g, p0, 0.0, Aeq, std::make_optional(Aineq), box_.x, box_.lb,
            box_.ub, std::nullopt);

        dvec gradL = g;
        if (Aeq && nu.size())
            gradL += Aeq->transpose() * nu;
        if (lam.size())
            gradL += Aineq.transpose() * lam;
        if (muL.size())
            gradL -= muL;
        if (muU.size())
            gradL += muU;

        const double stationarity = tr_norm_(gradL);

        // Evaluate constraint residuals from model if available
        double cons = 0.0;
        if (global_model) {
            try {
                std::vector<std::string> need{"cI", "cE"};
                global_model->eval_all(box_.x.value_or(dvec::Zero(n)), need);

                if (Aeq && Aeq->size()) {
                    if (auto cE_ = global_model->get_cE(); cE_)
                        cons += cE_->norm();
                }
                if (auto cI_ = global_model->get_cI(); cI_)
                    cons += cI_->cwiseMax(0.0).norm();

                double compl_ = 0.0;
                if (auto cI_ = global_model->get_cI(); cI_ && lam.size() > 0) {
                    const int m = std::min<int>(lam.size(), cI_->size());
                    for (int i = 0; i < m; ++i)
                        compl_ += std::abs(lam[i] * std::max(0.0, (*cI_)[i]));
                }
                return std::max({stationarity, cons, compl_});
            } catch (...) {
                // fallthrough to structural norm
            }
        }

        // fallback: structural violation at p = 0
        const dvec vI = Aineq * p0 + bineq;
        const double vInorm = vI.cwiseMax(0.0).norm();
        return std::max(stationarity, vInorm);
    }

    [[nodiscard]] bool
    check_curvature_criticality_(const dvec &g, const std::optional<dmat> &Aeq,
                                 const LinOp &Hop) const {
        dvec pg = g;
        if (Aeq && Aeq->size()) {
            project_tangent_into(g, *Aeq, W_.tmp);
            pg = W_.tmp;
        }
        const double nrm = tr_norm_(pg);
        if (nrm <= 1e-12)
            return true;
        dvec d = pg / std::max(nrm, 1e-16);
        if (Aeq && Aeq->size()) {
            project_tangent_into(d, *Aeq, W_.tmp);
            d = W_.tmp;
            d /= std::max(tr_norm_(d), 1e-16);
        }
        dvec Hd(d.size());
        Hop.apply_into(d, Hd);
        if (Aeq && Aeq->size()) {
            project_tangent_into(Hd, *Aeq, W_.tmp);
            Hd = W_.tmp;
        }
        const double curv = d.dot(Hd);
        return curv <= cfg_.curvature_criticality_tol;
    }

    [[nodiscard]] bool check_constraint_degeneracy_(const dmat &Aineq,
                                                    const dvec &) const {
        if (Aineq.rows() <= 1)
            return false;
        try {
            Eigen::FullPivLU<dmat> lu(Aineq);
            lu.setThreshold(cfg_.licq_threshold);
            return lu.rank() < std::min(Aineq.rows(), Aineq.cols());
        } catch (...) {
            return true;
        }
    }

    [[nodiscard]] inline double curvature_along_(const LinOp &H,
                                                 const dvec &p) {
        double denom = tr_norm_(p);
        denom *= denom;
        if (denom <= 1e-16)
            return std::numeric_limits<double>::quiet_NaN();
        H_apply(H, p, W_.Hp);
        return p.dot(W_.Hp) / denom;
    }

    void update_tr_radius_(double predicted, double actual, double step_norm,
                           const LinOp &H, const dvec &p, double /*theta_old*/,
                           double /*theta_new*/) {
        if (!(std::isfinite(predicted) && std::abs(predicted) > 1e-16))
            return;
        const double rho = actual / predicted;
        recent_rhos_.push_back(rho);
        if (recent_rhos_.size() > 10)
            recent_rhos_.erase(recent_rhos_.begin());
        auto [eta1, eta2] = compute_adaptive_thresholds_();

        if (rho < eta1) {
            double shrink = cfg_.gamma1;
            if (jacobian_condition_ > cfg_.max_jacobian_condition / 10.0)
                shrink *= 0.5;
            delta_ *= shrink;
        } else {
            const bool near_bdry = (step_norm >= 0.8 * delta_);
            if (near_bdry && rho >= eta2) {
                double growth = cfg_.gamma2;
                if (cfg_.curvature_aware) {
                    const double curv = curvature_along_(H, p);
                    growth = adjust_growth_for_curvature_(growth, curv);
                }
                if (jacobian_condition_ > 100)
                    growth = std::min(growth, 1.5);
                delta_ = std::min(cfg_.delta_max, growth * delta_);
            }
        }
        delta_ = std::clamp(delta_, cfg_.delta_min, cfg_.delta_max);
    }

    [[nodiscard]] std::pair<double, double>
    compute_adaptive_thresholds_() const {
        if (!cfg_.adaptive_eta_thresholds || recent_rhos_.empty())
            return {cfg_.eta1, cfg_.eta2};
        const double mean =
            std::accumulate(recent_rhos_.begin(), recent_rhos_.end(), 0.0) /
            recent_rhos_.size();
        double e1 = cfg_.eta1, e2 = cfg_.eta2;
        if (mean < cfg_.eta1) {
            e1 *= (1 + cfg_.eta_adaptive_factor);
            e2 *= (1 + cfg_.eta_adaptive_factor);
        } else if (mean > cfg_.eta2) {
            e1 *= (1 - cfg_.eta_adaptive_factor);
            e2 *= (1 - cfg_.eta_adaptive_factor);
        }
        return {std::clamp(e1, 0.05, 0.25), std::clamp(e2, 0.7, 0.95)};
    }

    [[nodiscard]] inline double choose_predicted_(const TRInfo &info) const {
        if (std::isfinite(info.predicted_reduction_cubic) &&
            std::abs(info.predicted_reduction_cubic) >= 1e-16)
            return info.predicted_reduction_cubic;
        return info.model_reduction_quad;
    }

    [[nodiscard]] Prec make_projected_prec_(const dmat &Aeq,
                                            const Prec &base) const {
        Prec Pproj;
        Pproj.valid = true;
        Pproj.apply = [this, Aeq, base](const dvec &r, dvec &z) {
            project_tangent_into(r, Aeq, W_.proj_buf1);
            if (base.valid) {
                base.apply(W_.proj_buf1, W_.proj_buf2);
                project_tangent_into(W_.proj_buf2, Aeq, z);
            } else {
                z = W_.proj_buf1;
            }
        };
        return Pproj;
    }

    ModelC *global_model = nullptr;

    // ---------------- core solve ----------------
    [[nodiscard]] TRResult core_solve_(
        const LinOp &Hop, const dvec &g, const std::optional<dmat> &Aineq,
        const std::optional<dvec> &bineq, const std::optional<dmat> &Aeq,
        const std::optional<dvec> &beq, const std::optional<spmat> &sparseH,
        const std::optional<dmat> &denseH, ModelC *model,
        const std::optional<dvec> &x, const std::optional<dvec> &lb,
        const std::optional<dvec> &ub, double mu,
        const std::optional<double> &f_old_opt) {

        (void)mu; // not used here directly (placeholder for future penalties)

        global_model = model ? model : nullptr;

        TRInfo info;
        dvec lam(Aineq ? (int)Aineq->rows() : 0);
        dvec nu(Aeq ? (int)Aeq->rows() : 0);
        dvec p;

        // equality path
        if (Aeq && Aeq->rows() > 0) {
            auto [pE, infE] = solve_with_equalities_(
                Hop, g, *Aeq, beq.value_or(dvec::Zero(Aeq->rows())), Aineq,
                bineq, sparseH, denseH, x, lb, ub);
            p = pE;
            info = infE;
        } else if (Aineq && Aineq->rows() > 0) {
            auto [pI, infI] = solve_with_inequalities_(
                Hop, g, *Aineq, bineq.value_or(dvec::Zero(Aineq->rows())),
                sparseH, denseH, x, lb, ub);
            p = pI;
            info = infI;
        } else {
            // ---------- UNCONSTRAINED ----------
            {
                auto [crit_used, shr] =
                    maybe_apply_criticality_(g, std::nullopt);
                info.criticality = crit_used;
                info.criticality_shrinks = shr;
            }

            const int n = (int)g.size();
            const bool has_dense = (bool)denseH;
            const bool spd_hint =
                cfg_.assume_gn_spd; // or inspect H type if you track it
            const double cond_hint =
                jacobian_condition_; // updated in SOC; ok if NaN

            const TRBackend be =
                choose_backend_(n, has_dense, spd_hint,
                                std::isfinite(cond_hint) ? cond_hint : 0.0);

            bool took_step = false;

            if (be == TRBackend::DOGLEG && has_dense) {
                // Dense SPD dogleg
                auto dr = dogleg_step_(*denseH, g, metric_, delta_);
                dvec p_try = box_.active() ? box_.enforce_step(dr.p) : dr.p;

                info.iterations = dr.iters;
                info.status = dr.status;
                info.preconditioned = false;
                p = std::move(p_try);
                took_step = true;
            } else if (be == TRBackend::GLTR) {
                // If you don't have GLTR implemented, fall back to PCG.
                // (If you do, call your gltr_krylov_step_ here.)
            }

            if (!took_step) {
                // Default Steihaug-PCG
                const Prec P =
                    denseH ? make_prec(&*denseH, cfg_.prec_kind, cfg_.use_prec,
                                       cfg_.ssor_omega)
                    : sparseH ? make_prec(&*sparseH, cfg_.prec_kind,
                                          cfg_.use_prec, cfg_.ssor_omega)
                              : PrecIdentity::make(Hop.n);

                const double tol = cg_tol_(tr_norm_(g));
                const auto cg =
                    steihaug_pcg(Hop, g, metric_, delta_, tol, cfg_.cg_maxiter,
                                 cfg_.neg_curv_tol, P, W_);

                dvec p_try = box_.active() ? box_.enforce_step(cg.p) : cg.p;

                info.iterations = cg.iters;
                info.status = status_to_string_(cg.status);
                info.preconditioned = P.valid;
                p = std::move(p_try);
            }

            info.step_norm = tr_norm_(p);
            info.model_reduction = model_reduction_quad_into(Hop, g, p, W_.Hp);
            info.model_reduction_quad = info.model_reduction;
        }

        // cubic prediction
        const double sigma = estimate_sigma_into(Hop, metric_, g, p, W_.Hp);
        info.sigma_est = sigma;
        info.predicted_reduction_cubic =
            pred_red_cubic_into(Hop, metric_, g, p, sigma, W_.Hp);
        info.step_norm = tr_norm_(p);
        if (Aeq)
            info.constraint_violation =
                (Aeq->size()
                     ? ((*Aeq) * p + beq.value_or(dvec::Zero(Aeq->rows())))
                           .norm()
                     : 0.0);

        // multipliers
        dvec muL, muU;
        std::tie(lam, nu, muL, muU) = recover_multipliers_(
            Hop, g, p, sigma, Aeq, Aineq, x, lb, ub,
            info.active_set_size
                ? std::optional<std::vector<int>>(info.active_set_indices)
                : std::nullopt);

        // SOC
        if (model && box_.x) {
            auto soc = soc_correction_(
                model, *box_.x, p, Hop, denseH, sparseH, lam,
                /*mu*/ 0.0, /*wE*/ 10.0, /*wI*/ 1.0, /*tolE*/ 1e-8,
                /*violI*/ 0.0,
                /*reg*/ 1e-12, /*sigma0*/ cfg_.metric_shift, lb, ub);
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

        // model eval (optional)
        std::optional<double> f_trial, theta_trial;
        std::optional<double> f_old = f_old_opt;
        if (model) {
            if (!f_old) {
                try {
                    if (model->get_f())
                        f_old = model->get_f().value();
                    else if (box_.x)
                        f_old = eval_model_f_theta_(model, box_.x,
                                                    dvec::Zero(g.size()))
                                    .first;
                } catch (...) {
                }
            }
            auto ft = eval_model_f_theta_(model, box_.x, p);
            f_trial = ft.first;
            theta_trial = ft.second;
        }

        // filter
        bool accepted = true;
        std::string accepted_by = "no_filter";
        if (filter_enabled_ && f_trial && theta_trial &&
            std::isfinite(*f_trial) && std::isfinite(*theta_trial)) {
            if (filter_.is_acceptable(*theta_trial, *f_trial, delta_)) {
                (void)filter_.add_if_acceptable(*theta_trial, *f_trial, delta_);
                accepted_by = "filter";
            } else {
                accepted = false;
                accepted_by = "rejected_by_filter";
            }
        }

        // backtracking if rejected
        if (!accepted) {
            auto [ok, p_new, by, f_bt, th_bt] = _backtrack_on_reject_(
                model, box_.x, p, box_.lb, box_.ub, delta_, f_old, 3);
            if (ok) {
                if (filter_enabled_ && f_bt && th_bt && std::isfinite(*f_bt) &&
                    std::isfinite(*th_bt))
                    (void)filter_.add_if_acceptable(*th_bt, *f_bt, delta_);
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

        // radius update
        if (info.accepted) {
            const double predicted = choose_predicted_(info);
            double actual = std::numeric_limits<double>::quiet_NaN();
            if (f_old && f_trial && std::isfinite(*f_old) &&
                std::isfinite(*f_trial))
                actual = (*f_old) - (*f_trial);
            if (std::isfinite(predicted) && std::isfinite(actual))
                update_tr_radius_(predicted, actual, info.step_norm, Hop, p,
                                  NAN, NAN);
        }

        TRResult ret;
        ret.p = std::move(p);
        ret.info = std::move(info);
        ret.lam = std::move(lam);
        ret.nu = std::move(nu);
        ret.zL = std::move(muL);
        ret.zU = std::move(muU);
        return ret;
    }

    // --------------- projected operator / normal step ---------------
    [[nodiscard]] LinOp make_projected_operator_(const LinOp &H,
                                                 const dmat &Aeq) const {
        LinOp P;
        P.n = H.n;
        P.mv = [this, H, Aeq](const dvec &x, dvec &y) {
            project_tangent_into(x, Aeq, W_.Px);
            H_apply(H, W_.Px, W_.Hx);
            project_tangent_into(W_.Hx, Aeq, W_.PHx);
            y = W_.PHx;
        };
        return P;
    }

    [[nodiscard]] dvec min_norm_normal_(const dmat &A, const dvec &b) const {
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
        const dmat AMiAT = A * Minv(A.transpose());
        const dvec lam = AAT_solve(AMiAT, -b);
        return Minv(A.transpose() * lam);
    }

    // --------------- equality path ---------------
    [[nodiscard]] std::pair<dvec, TRInfo> solve_with_equalities_(
        const LinOp &Hop, const dvec &g, const dmat &Aeq, const dvec &beq,
        const std::optional<dmat> &Aineq, const std::optional<dvec> &bineq,
        const std::optional<spmat> &sparseH, const std::optional<dmat> &denseH,
        const std::optional<dvec> &x, const std::optional<dvec> &lb,
        const std::optional<dvec> &ub) {
        TRInfo info;
        {
            auto [c, s] = maybe_apply_criticality_(g, Aeq);
            info.criticality = c;
            info.criticality_shrinks = s;
        }

        dvec p_n = min_norm_normal_(Aeq, beq);
        const double zeta = cfg_.zeta;
        double nn = tr_norm_(p_n);
        if (nn > zeta * delta_)
            p_n *= (zeta * delta_ / std::max(1e-16, nn));

        if (box_.active()) {
            if (use_projection_()) {
                const dvec shift =
                    BoxCtx::pullback(*box_.x, p_n, box_.lb, box_.ub) - p_n;
                const dvec beqNew = beq + Aeq * shift;
                p_n = min_norm_normal_(Aeq, beqNew);
                nn = tr_norm_(p_n);
                if (nn > zeta * delta_)
                    p_n *= (zeta * delta_ / std::max(1e-16, nn));
            } else {
                const double amax =
                    BoxCtx::alpha_max(*box_.x, p_n, box_.lb, box_.ub);
                if (amax < 1.0) {
                    const dvec shift =
                        BoxCtx::pullback(*box_.x, p_n, box_.lb, box_.ub) - p_n;
                    const dvec beqNew = beq + Aeq * shift;
                    p_n = min_norm_normal_(Aeq, beqNew);
                    nn = tr_norm_(p_n);
                    if (nn > zeta * delta_)
                        p_n *= (zeta * delta_ / std::max(1e-16, nn));
                }
            }
        }

        // ---------- Tangential subproblem ----------
        W_.tmp = g;
        Hop.apply_into(p_n, W_.Hp);
        W_.tmp.noalias() += W_.Hp; // gtilde
        auto Htan = make_projected_operator_(Hop, Aeq);
        project_tangent_into(W_.tmp, Aeq, W_.proj_buf1); // g_tan
        double rem = std::sqrt(
            std::max(0.0, delta_ * delta_ - tr_norm_(p_n) * tr_norm_(p_n)));

        {
            auto [c2, s2] = maybe_apply_criticality_(W_.proj_buf1, Aeq);
            (void)c2;
            (void)s2;
            rem = std::sqrt(
                std::max(0.0, delta_ * delta_ - tr_norm_(p_n) * tr_norm_(p_n)));
        }

        // Choose backend for the tangential step
        const int n = (int)g.size();
        const bool has_dense = (bool)denseH;
        const bool spd_hint =
            cfg_.assume_gn_spd; // GN/H SPD-ish in tangent space
        const double cond_hint = jacobian_condition_;

        const TRBackend be = choose_backend_(
            n, has_dense, spd_hint, std::isfinite(cond_hint) ? cond_hint : 0.0);

        dvec p_t;
        bool took_step = false;

        if (be == TRBackend::DOGLEG && has_dense) {
            // Build dense reduced problem in the nullspace of Aeq
            auto [Q2, k] = nullspace_basis_(Aeq);
            if (k > 0) {
                // Reduced Hessian and gradient (tangent)
                const dmat Hs = 0.5 * ((*denseH) + denseH->transpose());
                dmat Hred = Q2.transpose() * Hs * Q2;

                // g_tan already in tangent space; reduce:
                dvec gred = Q2.transpose() * W_.proj_buf1;

                // Dogleg in reduced space with Euclidean metric (or pull metric
                // if you want)
                Metric Mred =
                    metric_; // ok: your boundary_intersection_metric uses M

                auto dr = dogleg_step_(Hred, gred, Mred, rem);
                dvec z = dr.p;        // reduced coords
                dvec pt_try = Q2 * z; // lift to full space

                // Enforce radius in metric and box after combining with normal
                // step:
                if (tr_norm_(pt_try) > rem) {
                    const double t = std::max(1e-16, tr_norm_(pt_try));
                    pt_try *= (rem / t);
                }
                if (box_.active()) {
                    if (use_projection_())
                        pt_try = box_.enforce_step(p_n + pt_try) - p_n;
                    else {
                        const double beta = BoxCtx::alpha_max(
                            *box_.x + p_n, pt_try, box_.lb, box_.ub);
                        if (beta < 1.0)
                            pt_try *= beta;
                    }
                }

                p_t = std::move(pt_try);
                info.preconditioned_reduced = false;
                info.iterations += 1;
                took_step = true;
            }
        } else if (be == TRBackend::GLTR) {
            // If you implement GLTR for the projected operator, call it here.
            // Otherwise, fall back to PCG below.
        }

        if (!took_step) {
            // Default: Projected Steihaug-PCG
            const Prec Pbase =
                has_dense ? make_prec(&*denseH, cfg_.prec_kind, cfg_.use_prec,
                                      cfg_.ssor_omega)
                          : (sparseH ? make_prec(&*sparseH, cfg_.prec_kind,
                                                 cfg_.use_prec, cfg_.ssor_omega)
                                     : PrecIdentity::make(Hop.n));
            const Prec Pproj = make_projected_prec_(Aeq, Pbase);
            info.preconditioned_reduced = true;

            const double tol = cg_tol_(W_.proj_buf1.norm());
            const auto cg =
                steihaug_pcg(Htan, W_.proj_buf1, metric_, rem, tol,
                             cfg_.cg_maxiter, cfg_.neg_curv_tol, Pproj, W_);
            p_t = cg.p;
            info.iterations += cg.iters;
            info.status = status_to_string_(cg.status);
        }

        // Clip to remaining radius if needed
        const double pt_norm = tr_norm_(p_t);
        if (pt_norm > rem)
            p_t *= (rem / std::max(1e-16, pt_norm));

        if (box_.active()) {
            if (use_projection_())
                p_t = box_.enforce_step(p_n + p_t) - p_n;
            else {
                const double beta =
                    BoxCtx::alpha_max(*box_.x + p_n, p_t, box_.lb, box_.ub);
                if (beta < 1.0)
                    p_t *= beta;
            }
        }

        dvec p = p_n + p_t;

        // inequalities atop equalities
        if (Aineq && Aineq->rows() > 0) {
            auto [p2, active, inf_add] =
                active_set_loop_(Hop, g, Aeq, beq, *Aineq,
                                 bineq.value_or(dvec::Zero(Aineq->rows())), p);
            p = box_.active() ? box_.enforce_step(p2) : p2;
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

    // --------------- inequality-only path ---------------
    [[nodiscard]] std::pair<dvec, TRInfo> solve_with_inequalities_(
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

        const Prec P = denseH    ? make_prec(&*denseH, cfg_.prec_kind,
                                             cfg_.use_prec, cfg_.ssor_omega)
                       : sparseH ? make_prec(&*sparseH, cfg_.prec_kind,
                                             cfg_.use_prec, cfg_.ssor_omega)
                                 : PrecIdentity::make(Hop.n);
        const double tol = cg_tol_(tr_norm_(g));
        const auto cg = steihaug_pcg(Hop, g, metric_, delta_, tol,
                                     cfg_.cg_maxiter, cfg_.neg_curv_tol, P, W_);
        dvec p = box_.active() ? box_.enforce_step(cg.p) : cg.p;

        dvec viol = Aineq * p + bineq;
        if ((viol.array() > cfg_.constraint_tol).any()) {
            dmat Aeq0(0, (int)g.size());
            dvec beq0(0);
            auto [p2, active, inf_add] =
                active_set_loop_(Hop, g, Aeq0, beq0, Aineq, bineq, p);
            p = box_.active() ? box_.enforce_step(p2) : p2;
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

    // --------------- active-set loop ---------------
    [[nodiscard]] std::tuple<dvec, std::vector<int>, std::pair<int, int>>
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

        auto build_augmented = [&](dmat &Acur, dvec &bcur) {
            if (active.empty()) {
                if (Aeq && beq && Aeq->size()) {
                    Acur = *Aeq;
                    bcur = *beq;
                } else {
                    Acur.resize(0, Aineq.cols());
                    bcur.resize(0);
                }
                return;
            }
            dmat Aact((int)active.size(), Aineq.cols());
            dvec bact((int)active.size());
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
                Acur = std::move(Aact);
                bcur = std::move(bact);
            }
        };

        const int rank_eq = (Aeq && Aeq->size()) ? rank_of(*Aeq) : 0;
        const int max_active =
            std::max(0, std::min<int>((int)Aineq.rows(), n - rank_eq));

        while (it < cfg_.max_active_set_iter) {
            const dvec viol = Aineq * p + bineq;
            const auto mask = (viol.array() > cfg_.constraint_tol);
            if (!mask.any())
                break;

            dmat Acur;
            dvec bcur;
            build_augmented(Acur, bcur);
            const int rcur = Acur.size() ? rank_of(Acur) : 0;
            if (rcur >= n || (int)active.size() >= max_active)
                break;

            std::vector<int> cand;
            cand.reserve((size_t)viol.size());
            for (int i = 0; i < viol.size(); ++i)
                if (mask[i])
                    cand.push_back(i);
            std::ranges::sort(cand,
                              [&](int a, int b) { return viol[a] > viol[b]; });

            bool added = false;
            int last = -1;
            for (int idx : cand) {
                if (active.contains(idx))
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
            build_augmented(A_aug, b_aug);
            try {
                auto [p_eq, _] = solve_with_equalities_(
                    Hop, g, A_aug, b_aug, std::nullopt, std::nullopt,
                    std::nullopt, std::nullopt, box_.x, box_.lb, box_.ub);
                p = box_.active() ? box_.enforce_step(p_eq) : p_eq;
            } catch (...) {
                if (last >= 0)
                    active.erase(last);
                break;
            }
            ++it;
        }

        std::vector<int> active_idx(active.begin(), active.end());
        return {p, active_idx, {it, (int)active.size()}};
    }

    // --------------- multipliers ---------------
    [[nodiscard]] std::pair<std::vector<int>, std::vector<int>>
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

    // ---------- Multipliers (clean/lean, fewer allocs) ----------
    [[nodiscard]] std::tuple<dvec, dvec, dvec, dvec> recover_multipliers_(
        const LinOp &Hop, const dvec &g, const dvec &p, double sigma,
        const std::optional<dmat> &Aeq, const std::optional<dmat> &Aineq,
        const std::optional<dvec> &x, const std::optional<dvec> &lb,
        const std::optional<dvec> &ub,
        const std::optional<std::vector<int>> &active_idx_opt) {

        const int n = (int)g.size();

        // r = H p + g + sigma p  (reuse workspace buffer)
        W_.Hp.resizeLike(p);
        W_.Hp.setZero();
        H_apply(Hop, p, W_.Hp);
        const dvec r = W_.Hp + g + sigma * p;

        // Transposed blocks that will compose AT = [Aeq^T | Aact^T | I_L |
        // -I_U]
        dmat AeqT(n, 0);
        if (Aeq && Aeq->size())
            AeqT = Aeq->transpose();

        dmat AactT(n, 0);
        if (Aineq && Aineq->size()) {
            if (cfg_.recover_lam_active_only && active_idx_opt &&
                !active_idx_opt->empty()) {
                const auto &ids = *active_idx_opt;
                const int k = (int)ids.size();
                dmat A_sel(k, Aineq->cols());
                for (int i = 0; i < k; ++i)
                    A_sel.row(i) = Aineq->row(ids[i]);
                AactT = A_sel.transpose();
            } else {
                AactT = Aineq->transpose();
            }
        }

        // Active bound indicator columns
        const auto [idxL, idxU] = detect_active_bounds_(x, p, lb, ub, 1e-10);
        const int nL = (int)idxL.size(), nU = (int)idxU.size();

        dmat IL(n, nL), IU(n, nU);
        if (nL) {
            IL.setZero();
            for (int j = 0; j < nL; ++j)
                IL(idxL[j], j) = 1.0;
        }
        if (nU) {
            IU.setZero();
            for (int j = 0; j < nU; ++j)
                IU(idxU[j], j) = 1.0;
        }

        const int mcols = AeqT.cols() + AactT.cols() + nL + nU;
        if (mcols == 0) {
            // No multipliers to recover
            dvec lam(Aineq ? (int)Aineq->rows() : 0);
            dvec nu(Aeq ? (int)Aeq->rows() : 0);
            return {lam, nu, dvec::Zero(n), dvec::Zero(n)};
        }

        // Build AT in one pass
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
        if (nL) {
            AT.middleCols(ofs, nL) = IL;
            ofs += nL;
        }
        if (nU) {
            AT.middleCols(ofs, nU) = -IU;
            ofs += nU;
        }

        // Solve (AT^T AT) y = -AT^T r  (small SPD; LDLT is fine)
        // const dmat N = AT.transpose() * AT;
        // const dvec rhs = -AT.transpose() * r;
        // const dvec y = N.ldlt().solve(rhs);

        // Solve (AT^T AT + reg I) y = -AT^T r  using QDLDL on the upper
        // triangle
        const double reg = 1e-12;
        dmat N = AT.transpose() * AT;
        N.diagonal().array() += reg;
        const dvec rhs = -AT.transpose() * r;

        const dvec y = solve_sym_spd_qdldl(N, rhs);

        // Unpack y -> (nu, lam, muL, muU)
        ofs = 0;
        dvec nu(Aeq ? (int)Aeq->rows() : 0);
        if (AeqT.cols()) {
            nu = y.segment(ofs, AeqT.cols());
            ofs += AeqT.cols();
        }

        dvec lam;
        if (Aineq && Aineq->size()) {
            lam = dvec::Zero(Aineq->rows());
            const int block = AactT.cols();
            if (block > 0) {
                const dvec lam_block = y.segment(ofs, block).cwiseMax(0.0);
                if (cfg_.recover_lam_active_only && active_idx_opt &&
                    !active_idx_opt->empty()) {
                    for (int j = 0; j < (int)active_idx_opt->size(); ++j)
                        lam[(*active_idx_opt)[j]] = lam_block[j];
                } else {
                    // All constraints used
                    // Note: if AactT included all rows, block == rows(Aineq)
                    lam.head(block) = lam_block;
                }
                ofs += block;
            }
        } else {
            lam = dvec(0);
        }

        dvec muL = dvec::Zero(n), muU = dvec::Zero(n);
        if (nL) {
            const dvec yL = y.segment(ofs, nL);
            for (int j = 0; j < nL; ++j)
                muL[idxL[j]] = std::max(0.0, -yL[j]);
            ofs += nL;
        }
        if (nU) {
            const dvec yU = y.segment(ofs, nU);
            for (int j = 0; j < nU; ++j)
                muU[idxU[j]] = std::max(0.0, -yU[j]);
            ofs += nU;
        }

        return {lam, nu, muL, muU};
    }

    // ---------- Model eval (tight, no GIL unless you add it) ----------
    [[nodiscard]] std::pair<std::optional<double>, std::optional<double>>
    eval_model_f_theta_(ModelC *model, const std::optional<dvec> &x,
                        const dvec &s = dvec()) const {
        if (!model)
            return {std::nullopt, std::nullopt};
        const dvec xt = x ? (*x + s) : s;
        try {
            std::vector<std::string> need{"f"}; // keep eval light
            model->eval_all(xt, need);
            auto f = model->get_f();
            auto th = model->constraint_violation(xt);
            return {f, th};
        } catch (const nb::python_error &) {
            return {std::nullopt, std::nullopt};
        }
    }

    // ---------- Backtracking (factor common checks, keep semantics) ----------
    [[nodiscard]] std::tuple<bool, dvec, std::string, std::optional<double>,
                             std::optional<double>>
    _backtrack_on_reject_(ModelC *model, const std::optional<dvec> &x,
                          const dvec &s, const std::optional<dvec> &lb,
                          const std::optional<dvec> &ub, double delta,
                          std::optional<double> base_f, int max_tries = 3) {
        static constexpr std::array<double, 3> alphas{0.5, 0.25, 0.125};
        const int tries = std::min(max_tries, (int)alphas.size());

        auto acceptable = [&](const std::optional<double> &f_t,
                              const std::optional<double> &th_t) -> bool {
            return filter_enabled_ && f_t && th_t && std::isfinite(*f_t) &&
                   std::isfinite(*th_t) &&
                   filter_.is_acceptable(*th_t, *f_t, delta);
        };

        for (int k = 0; k < tries; ++k) {
            dvec sa = alphas[(size_t)k] * s;
            if (box_.active())
                sa = box_.enforce_step(sa);

            auto [f_t, th_t] = eval_model_f_theta_(model, x, sa);

            if (acceptable(f_t, th_t)) {
                (void)filter_.add_if_acceptable(*th_t, *f_t, delta);
                return {true, sa, "ls-filter", f_t, th_t};
            }

            if (base_f && f_t && std::isfinite(*base_f) &&
                std::isfinite(*f_t)) {
                // Simple Armijo on f; keep constants as before
                if (*f_t <=
                    *base_f - 1e-4 * alphas[(size_t)k] * std::abs(*base_f)) {
                    if (filter_enabled_ && th_t && std::isfinite(*th_t))
                        (void)filter_.add_if_acceptable(*th_t, *f_t, delta);
                    return {true, sa, "ls-armijo", f_t, th_t};
                }
            }
        }
        return {false, s, std::string("rejected"), std::nullopt, std::nullopt};
    }

    // ---------- Status text ----------
    [[nodiscard]] static std::string status_to_string_(TRStatus s) {
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

    // ---------- SOC helpers ----------
    struct SOCResult {
        dvec q;
        bool applied = false;
        double theta0 = NAN;
        double theta1 = NAN;
    };

    [[nodiscard]] dvec clip_correction_to_radius_(const dvec &s,
                                                  const dvec &q) const {
        if (tr_norm_(s + q) <= delta_ + 1e-14 || tr_norm_(q) <= 1e-16)
            return q;
        const double t =
            detail::boundary_intersection_metric(metric_, s, q, delta_);
        return std::clamp(t, 0.0, 1.0) * q;
    }

    [[nodiscard]] SOCResult
    soc_correction_(ModelC *model, const dvec &x, const dvec &p,
                    const LinOp &Hop, const std::optional<dmat> &Hdense,
                    const std::optional<spmat> &Hsparse, const dvec &lam_ineq,
                    double mu, double wE, double wI, double tolE, double violI,
                    double reg, double sigma0, const std::optional<dvec> &lb,
                    const std::optional<dvec> &ub) {
        (void)Hop;
        (void)lam_ineq;
        (void)mu;
        (void)lb;
        (void)ub;

        SOCResult R{dvec::Zero(p.size()), false, NAN, NAN};

        const dvec x_trial = x + p;
        std::vector<std::string> need{"cE", "cI", "JE", "JI"};
        model->eval_all(x_trial, /*need*/ need);

        const double theta0 = model->constraint_violation(x_trial);
        R.theta0 = theta0;
        if (theta0 <= cfg_.constraint_tol)
            return R;

        const int mI = model->get_mI();
        const int mE = model->get_mE();
        const int n = model->get_n();

        const dmat JI_ = (mI > 0) ? model->get_JI().value() : dmat(mI, n);
        const dmat JE_ = (mE > 0) ? model->get_JE().value() : dmat(mE, n);
        const dvec cI_ = (mI > 0) ? model->get_cI().value() : dvec::Zero(mI);
        const dvec cE_ = (mE > 0) ? model->get_cE().value() : dvec::Zero(mE);

        jacobian_condition_ =
            estimate_jacobian_condition_(JE_, JI_, cI_, cfg_.constraint_tol);

        dvec q = compute_soc_step_(cE_, cI_, JE_, JI_, wE, wI, tolE, violI,
                                   std::max(reg, cfg_.jacobian_reg_min), sigma0,
                                   Hdense, Hsparse);

        // Bound within SOC radius and TR metric
        const double soc_radius = cfg_.soc_radius_fraction * delta_;
        const double qn = tr_norm_(q);
        if (qn > soc_radius && qn > 1e-16)
            q *= (soc_radius / qn);

        q = clip_correction_to_radius_(p, q);
        if (box_.active())
            q = box_.enforce_correction(x_trial, q);

        const double theta1 = model->constraint_violation(x_trial + q);
        R.theta1 = theta1;

        if (soc_acceptance_test_(theta0, theta1)) {
            R.q = q;
            R.applied = true;
            if (cfg_.soc_use_funnel)
                current_theta_ = theta1;
        }
        return R;
    }

    [[nodiscard]] double adjust_growth_for_curvature_(double growth,
                                                      double curv) const {
        if (!std::isfinite(curv))
            return growth;
        if (curv <= -1e-10)
            return 1.0;
        if (curv < 1e-12)
            return std::min(1.2, growth);
        if (curv < 1e-4)
            return std::min(2.0, std::max(1.2, growth));
        if (curv < 1e-2)
            return std::min(2.5, std::max(1.4, growth));
        return std::min(3.0, std::max(1.6, 1.25 * growth));
    }

    // ---------- Regularized LS (dense SPD small systems) ----------
    // [[nodiscard]] dvec solve_regularized_ls_(const dmat &A, const dvec &b,
    //                                          double reg) const {
    //     if (A.size() == 0)
    //         return dvec::Zero(b.size());
    //     dmat N = A.transpose() * A;
    //     N.diagonal().array() += reg;
    //     return N.ldlt().solve(A.transpose() * b);
    // }

    [[nodiscard]] dvec solve_regularized_ls_(const dmat &A, const dvec &b,
                                             double reg) const {
        if (A.size() == 0)
            return dvec::Zero(b.size());

        // Build normal equations with a tiny floor on reg for numerical safety
        const double reg_eff = std::max(reg, 1e-12);
        dmat N = A.transpose() * A;
        N.diagonal().array() += reg_eff;

        const dvec rhs = A.transpose() * b;

        // Use the QDLDL-based symmetric SPD solver on the upper triangle
        return solve_sym_spd_qdldl(N, rhs);
    }

    [[nodiscard]] dvec solve_regularized_metric_ls_(const dmat &A,
                                                    const dvec &b,
                                                    const dmat &M,
                                                    double reg) const {
        if (A.size() == 0)
            return dvec::Zero(M.cols()); // n=cols(M). (No rows in A ⇒ x=0)

        try {
            // Factor M = L Lᵀ (with minimal shift inside helper for robustness)
            const double reg_eff = std::max(reg, 1e-12);
            const dmat L =
                psd_cholesky_with_shift(M, reg_eff); // Lower-triangular

            // We need: (A M^{-1} Aᵀ + reg I) y = b, then x = M^{-1} Aᵀ y
            // Compute Minv Aᵀ via two triangular solves on a dense multi-RHS:
            //   Solve L * Y = Aᵀ          → Y = L^{-1} Aᵀ
            //   Solve Lᵀ * Z = Y          → Z = L^{-T} Y = M^{-1} Aᵀ
            const dmat At = A.transpose();
            dmat Y = L.triangularView<Eigen::Lower>().solve(At);
            dmat Z = L.transpose().triangularView<Eigen::Upper>().solve(
                Y); // Z = M^{-1}Aᵀ

            // Build S = A * Z = A * M^{-1} * Aᵀ, add Tikhonov reg on the
            // diagonal
            dmat S = A * Z;
            S.diagonal().array() += reg_eff;

            // Solve S y = b with QDLDL (upper triangle packed); fallback inside
            // helper
            const dvec y = solve_sym_spd_qdldl(S, b);

            // Recover x = M^{-1} Aᵀ y efficiently:
            //   t = Aᵀ y
            //   Solve L * w = t, then Lᵀ * x = w  → x = M^{-1} Aᵀ y
            const dvec t = At * y;
            dvec w = L.triangularView<Eigen::Lower>().solve(t);
            dvec x = L.transpose().triangularView<Eigen::Upper>().solve(w);
            return x;
        } catch (...) {
            // If anything goes wrong with metric factorization, fall back to
            // Euclidean LS
            return solve_regularized_ls_(A, b, reg);
        }
    }

    // [[nodiscard]] dvec solve_regularized_metric_ls_(const dmat &A,
    //                                                 const dvec &b,
    //                                                 const dmat &M,
    //                                                 double reg) const {
    //     try {
    //         const dmat L = psd_cholesky_with_shift(M, reg);
    //         auto Minv = [&](const dvec &v) {
    //             dvec y = L.triangularView<Eigen::Lower>().solve(v);
    //             return L.transpose().triangularView<Eigen::Upper>().solve(y);
    //         };
    //         const dmat At = A.transpose();

    //         // Build A Minv At without forming Minv explicitly
    //         dmat AMinvAt(A.rows(), A.rows());
    //         for (int i = 0; i < At.cols(); ++i)
    //             AMinvAt.col(i) = A * Minv(At.col(i));
    //         AMinvAt.diagonal().array() += reg;

    //         const dvec y = AMinvAt.ldlt().solve(b);
    //         return Minv(At * y);
    //     } catch (...) {
    //         // Fall back to Euclidean LS if metric factorization fails
    //         return solve_regularized_ls_(A, b, reg);
    //     }
    // }

    // ---------- Jacobian condition estimate (stacked blocks, tight) ----------
    [[nodiscard]] double estimate_jacobian_condition_(
        const std::optional<dmat> &JE, const std::optional<dmat> &JI,
        const std::optional<dvec> &cI, double tol) const {

        std::vector<dmat> blocks;
        blocks.reserve(2);

        if (JE && JE->size())
            blocks.push_back(*JE);

        if (JI && JI->size() && cI) {
            const auto mask = (cI->array().abs() <= 10 * tol);
            if (mask.any()) {
                const int m = (int)mask.count();
                dmat JI_act(m, JI->cols());
                for (int i = 0, k = 0; i < cI->size(); ++i)
                    if (mask[i])
                        JI_act.row(k++) = JI->row(i);
                blocks.push_back(std::move(JI_act));
            }
        }

        if (blocks.empty())
            return 1.0;

        int rows = 0, cols = blocks[0].cols();
        for (const auto &B : blocks)
            rows += B.rows();

        dmat J(rows, cols);
        for (int ofs = 0, i = 0; i < (int)blocks.size(); ++i) {
            J.middleRows(ofs, blocks[i].rows()) = blocks[i];
            ofs += blocks[i].rows();
        }

        try {
            Eigen::JacobiSVD<dmat> svd(J, Eigen::ComputeThinU |
                                              Eigen::ComputeThinV);
            const auto &s = svd.singularValues();
            if (s.size() == 0 || s.tail(1)[0] <= 1e-16)
                return cfg_.max_jacobian_condition;
            return std::min(s.head(1)[0] / s.tail(1)[0],
                            cfg_.max_jacobian_condition);
        } catch (...) {
            return cfg_.max_jacobian_condition;
        }
    }

    // ---------- SOC acceptance ----------
    [[nodiscard]] bool soc_acceptance_test_(double theta0,
                                            double theta1) const {
        if (!std::isfinite(theta0) || !std::isfinite(theta1))
            return false;
        if (cfg_.soc_use_funnel) {
            const double sufficient = cfg_.soc_theta_reduction * theta0;
            const double funnel = cfg_.funnel_gamma * current_theta_;
            return theta1 <= std::max(sufficient, funnel);
        }
        return theta1 <= cfg_.soc_theta_reduction * theta0;
    }

    // ---------- SOC step (single pass build, minimal temporaries) ----------
    [[nodiscard]] dvec compute_soc_step_(
        const std::optional<dvec> &cE, const std::optional<dvec> &cI,
        const std::optional<dmat> &JE, const std::optional<dmat> &JI, double wE,
        double wI, double tolE, double violI, double reg, double sigma0,
        const std::optional<dmat> &Hdense,
        const std::optional<spmat> &Hsparse) const {

        std::vector<dmat> Jblocks;
        Jblocks.reserve(2);
        std::vector<dvec> rblocks;
        rblocks.reserve(2);

        const double sE = std::sqrt(std::max(0.0, wE));
        const double sI = std::sqrt(std::max(0.0, wI));

        if (cE && JE && JE->size()) {
            const auto mask = (cE->array().abs() > tolE);
            if (mask.any()) {
                const int m = (int)mask.count();
                dmat J(m, JE->cols());
                dvec r(m);
                for (int i = 0, k = 0; i < cE->size(); ++i)
                    if (mask[i]) {
                        J.row(k) = sE * JE->row(i);
                        r[k++] = -sE * (*cE)[i];
                    }
                Jblocks.push_back(std::move(J));
                rblocks.push_back(std::move(r));
            }
        }

        if (cI && JI && JI->size()) {
            const auto mask = (cI->array() > violI);
            if (mask.any()) {
                const int m = (int)mask.count();
                dmat J(m, JI->cols());
                dvec r(m);
                for (int i = 0, k = 0; i < cI->size(); ++i)
                    if (mask[i]) {
                        J.row(k) = sI * JI->row(i);
                        r[k++] = -sI * (*cI)[i];
                    }
                Jblocks.push_back(std::move(J));
                rblocks.push_back(std::move(r));
            }
        }

        if (Jblocks.empty()) {
            const int n =
                Hdense ? Hdense->cols() : (Hsparse ? (int)Hsparse->cols() : 0);
            return dvec::Zero(n);
        }

        int rows = 0, cols = Jblocks[0].cols();
        for (const auto &J : Jblocks)
            rows += J.rows();

        dmat J(rows, cols);
        dvec r(rows);
        for (int ofs = 0, i = 0; i < (int)Jblocks.size(); ++i) {
            J.middleRows(ofs, Jblocks[i].rows()) = Jblocks[i];
            r.segment(ofs, Jblocks[i].rows()) = rblocks[i];
            ofs += Jblocks[i].rows();
        }

        if (cfg_.norm_type == "ellip" && (Hdense || Hsparse)) {
            const dmat H = Hdense ? *Hdense : dmat(*Hsparse);
            const dmat M = symmetrize(H) + sigma0 * dmat::Identity(cols, cols);
            return solve_regularized_metric_ls_(
                J, r, M, reg * std::max(1.0, jacobian_condition_));
        }
        return solve_regularized_ls_(J, r,
                                     reg * std::max(1.0, jacobian_condition_));
    }

}; // namespace trl