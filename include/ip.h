// ip_cpp.cpp — optimized, modernized C++23 version (behavior preserved)

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SparseCore>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nanobind/eigen/dense.h>  // dense Eigen
#include <nanobind/eigen/sparse.h> // sparse Eigen
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "definitions.h"
#include "funnel.h"
#include "ip_aux.h"
#include "kkt_core.h"
#include "linesearch.h"
#include "regularizer.h"

#include "model.h" // Model
#include <chrono>
#include <iomanip> // std::setprecision

#ifndef IP_PROFILE
#define IP_PROFILE 1
#endif

#if IP_PROFILE
struct ScopedTimer {
    using clock = std::chrono::high_resolution_clock;
    std::string name;
    clock::time_point t0, last;
    bool print_on_destroy;

    explicit ScopedTimer(std::string n, bool print_total = true)
        : name(std::move(n)), t0(clock::now()), last(t0),
          print_on_destroy(print_total) {
        std::cout << "[IP] >>> " << name << " begin\n";
    }
    ~ScopedTimer() {
        if (print_on_destroy) {
            auto dt =
                std::chrono::duration<double, std::milli>(clock::now() - t0)
                    .count();
            std::cout << "[IP] <<< " << name << " end  (total: " << std::fixed
                      << std::setprecision(3) << dt << " ms)\n";
        }
    }
    void lap(const char *label) {
        auto now = clock::now();
        auto dt = std::chrono::duration<double, std::milli>(now - last).count();
        std::cout << "      [lap] " << label << ": " << std::fixed
                  << std::setprecision(3) << dt << " ms\n";
        last = now;
    }
};
#define IP_TIMER(name) ScopedTimer __ip_timer__(name)
#define IP_LAP(label) __ip_timer__.lap(label)
#define IP_LOG(msg) std::cout << "[IP] " << msg << "\n"
#else
#define IP_TIMER(name)                                                         \
    struct {                                                                   \
        void lap(const char *) {}                                              \
    } __ip_timer__(name)
#define IP_LAP(label)                                                          \
    do {                                                                       \
    } while (0)
#define IP_LOG(msg)                                                            \
    do {                                                                       \
    } while (0)
#endif

// ---------- External structs expected from your headers ----------
struct StepResult {
    dvec x;   // new primal iterate
    dvec lam; // new inequality multipliers
    dvec nu;  // new equality multipliers (empty if no JE)
};
struct KKTResult {
    dvec dx; // primal search direction
    dvec dy; // equality multipliers (empty if no JE)
    std::shared_ptr<kkt::KKTReusable> reusable; // factorization handle
};

// ---------- Constants ----------
namespace consts {
constexpr double EPS_DIV = 1e-16;
constexpr double EPS_POS = 1e-12;
constexpr double INF = std::numeric_limits<double>::infinity();
} // namespace consts

// ---------- Small utilities ----------
template <class T> [[nodiscard]] constexpr T clamp(T v, T lo, T hi) noexcept {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}
template <class T> [[nodiscard]] constexpr T clamp_min(T v, T lo) noexcept {
    return (v < lo) ? lo : v;
}
template <class T> [[nodiscard]] constexpr T clamp_max(T v, T hi) noexcept {
    return (v > hi) ? hi : v;
}
template <class T> [[nodiscard]] constexpr T clamp01(T v) noexcept {
    return v < T(0) ? T(0) : (v > T(1) ? T(1) : v);
}
[[nodiscard]] inline double sdiv(double num, double den,
                                 double eps = consts::EPS_DIV) noexcept {
    const double d = (std::abs(den) < eps) ? (den < 0 ? -eps : eps) : den;
    return num / d;
}
[[nodiscard]] inline double safe_inf_norm(const dvec &v) noexcept {
    return (v.size() == 0) ? 0.0 : v.cwiseAbs().maxCoeff();
}

// nanobind alias
namespace nb = nanobind;

// ---------- Core data structures ----------
struct IPState {
    int mI = 0, mE = 0;
    dvec s, lam, nu, zL, zU;
    double mu = 1e-2;
    double tau_shift = 0.0;
    bool initialized = false;
};

struct Bounds {
    dvec lb, ub, sL, sU;
    std::vector<uint8_t> hasL, hasU;
};

struct Sigmas {
    dvec Sigma_x, Sigma_s;
};

// ---------- Helpers ----------
namespace detail {

[[nodiscard]] inline Sigmas
build_sigmas(const dvec &zL, const dvec &zU, const Bounds &B, const dvec &lmb,
             const dvec &s, const dvec &cI, double tau_shift,
             double bound_shift, bool use_shifted, double eps_abs, double cap) {
    const int n = static_cast<int>(zL.size());
    const int mI = static_cast<int>(s.size());
    Sigmas S;

    // Ultra-vectorized Sigma_x computation
    S.Sigma_x = dvec::Zero(n);

    if (n > 0) {
        const double shift = use_shifted ? bound_shift : 0.0;

        // Convert bounds to Eigen arrays for vectorization
        dvec hasL_float(n), hasU_float(n), sL_vec(n), sU_vec(n);

        // Single loop to extract all bound data
        for (int i = 0; i < n; ++i) {
            hasL_float[i] = B.hasL[i] ? 1.0 : 0.0;
            hasU_float[i] = B.hasU[i] ? 1.0 : 0.0;
            sL_vec[i] = B.sL[i];
            sU_vec[i] = B.sU[i];
        }

        // Fully vectorized computation
        dvec dL = (sL_vec.array() + shift).max(eps_abs);
        dvec dU = (sU_vec.array() + shift).max(eps_abs);

        dvec zL_contrib = hasL_float.cwiseProduct(zL.cwiseQuotient(dL));
        dvec zU_contrib = hasU_float.cwiseProduct(zU.cwiseQuotient(dU));

        S.Sigma_x = (zL_contrib + zU_contrib).array().max(0.0).min(cap);
    }

    // Same vectorized Sigma_s as above
    S.Sigma_s = dvec::Zero(mI);

    if (mI > 0) {
        if (use_shifted) {
            dvec d = (s.array() + tau_shift).max(eps_abs);
            S.Sigma_s = (lmb.cwiseQuotient(d)).array().max(0.0).min(cap);
        } else {
            dvec sf = cI.cwiseAbs().array().max(1e-8).min(1.0);
            dvec sv = s.cwiseMax(sf).array().max(eps_abs);
            S.Sigma_s = (lmb.cwiseQuotient(sv)).array().max(0.0).min(cap);
        }
    }

    return S;
}

[[nodiscard]] inline std::pair<dvec, dvec>
dz_bounds_from_dx_vec(const dvec &dx, const dvec &zL, const dvec &zU,
                      const Bounds &B, double bound_shift, bool use_shifted,
                      double mu, bool use_mu) {
    const int n = dx.size();
    dvec dzL = dvec::Zero(n), dzU = dvec::Zero(n);
    for (int i = 0; i < n; ++i) {
        if (B.hasL[i]) {
            const double d = clamp_min(
                B.sL[i] + (use_shifted ? bound_shift : 0.0), consts::EPS_DIV);
            dzL[i] = use_mu ? (mu - d * zL[i] - zL[i] * dx[i]) / d
                            : -(zL[i] * dx[i]) / d;
        }
        if (B.hasU[i]) {
            const double d = clamp_min(
                B.sU[i] + (use_shifted ? bound_shift : 0.0), consts::EPS_DIV);
            dzU[i] = use_mu ? (mu - d * zU[i] + zU[i] * dx[i]) / d
                            : (zU[i] * dx[i]) / d;
        }
    }
    return {dzL, dzU};
}

[[nodiscard]] inline double complementarity(const dvec &s, const dvec &lmb,
                                            double mu, double tau_shift,
                                            bool use_shifted) {
    const int m = static_cast<int>(s.size());
    if (!m)
        return 0.0;
    double acc = 0.0;
    if (use_shifted) {
        for (int i = 0; i < m; ++i)
            acc += std::abs((s[i] + tau_shift) * lmb[i] - mu);
    } else {
        for (int i = 0; i < m; ++i)
            acc += std::abs(s[i] * lmb[i] - mu);
    }
    return acc / clamp_min(m, 1);
}

[[nodiscard]] inline double alpha_ftb_vec(const dvec &x, const dvec &dx,
                                          const dvec &s, const dvec &ds,
                                          const dvec &lmb, const dvec &dlam,
                                          const Bounds &B, double tau_pri,
                                          double tau_dual) {

    double a_pri = 1.0, a_dual = 1.0;

    for (int i = 0; i < s.size(); ++i)
        if (ds[i] < 0.0)
            a_pri = std::min(a_pri, -s[i] / std::min(ds[i], -consts::EPS_DIV));

    for (int i = 0; i < lmb.size(); ++i)
        if (dlam[i] < 0.0)
            a_dual =
                std::min(a_dual, -lmb[i] / std::min(dlam[i], -consts::EPS_DIV));

    for (int i = 0; i < x.size(); ++i) {
        if (B.hasL[i] && dx[i] < 0.0)
            a_pri = std::min(a_pri, -(x[i] - B.lb[i]) /
                                        std::min(dx[i], -consts::EPS_DIV));
        if (B.hasU[i] && dx[i] > 0.0)
            a_pri = std::min(a_pri, (B.ub[i] - x[i]) /
                                        clamp_min(dx[i], consts::EPS_DIV));
    }
    return clamp(std::min(tau_pri * a_pri, tau_dual * a_dual), 0.0, 1.0);
}

[[nodiscard]] inline double comp_inf_norm(const dvec &s, const dvec &lam,
                                          const dvec &zL, const dvec &zU,
                                          const Bounds &B, double mu,
                                          bool use_shifted, double tau_shift,
                                          double bound_shift) {

    const int n = static_cast<int>(zL.size());
    double c_inf = 0.0;

    for (int i = 0; i < n; ++i) {
        if (B.hasL[i]) {
            const double sL = B.sL[i] + (use_shifted ? bound_shift : 0.0);
            c_inf = std::max(c_inf, std::abs(sL * zL[i] - mu));
        }
        if (B.hasU[i]) {
            const double sU = B.sU[i] + (use_shifted ? bound_shift : 0.0);
            c_inf = std::max(c_inf, std::abs(sU * zU[i] - mu));
        }
    }
    if (s.size()) {
        for (int i = 0; i < s.size(); ++i) {
            const double se = s[i] + (use_shifted ? tau_shift : 0.0);
            c_inf = std::max(c_inf, std::abs(se * lam[i] - mu));
        }
    }
    return c_inf;
}

inline void cap_bound_duals_sigma_box(dvec &zL, dvec &zU, const Bounds &B,
                                      bool use_shifted, double bound_shift,
                                      double mu, double ksig = 1e10) {
    for (int i = 0; i < zL.size(); ++i) {
        if (B.hasL[i]) {
            const double sLc = clamp_min(
                B.sL[i] + (use_shifted ? bound_shift : 0.0), consts::EPS_DIV);
            const double lo = mu / (ksig * sLc);
            const double hi = (ksig * mu) / sLc;
            zL[i] = clamp(zL[i], lo, hi);
        }
        if (B.hasU[i]) {
            const double sUc = clamp_min(
                B.sU[i] + (use_shifted ? bound_shift : 0.0), consts::EPS_DIV);
            const double lo = mu / (ksig * sUc);
            const double hi = (ksig * mu) / sUc;
            zU[i] = clamp(zU[i], lo, hi);
        }
    }
}

} // namespace detail

// ---------- Main Interior Point Stepper ----------
class InteriorPointStepper {
public:
    Bounds get_bounds(const dvec &x) {
        const int n = static_cast<int>(x.size());
        Bounds B;

        // lb / ub
        B.lb = m_->get_lb().value_or(dvec::Constant(n, -consts::INF));
        B.ub = m_->get_ub().value_or(dvec::Constant(n, consts::INF));

        B.hasL.assign(n, 0);
        B.hasU.assign(n, 0);
        B.sL.resize(n);
        B.sU.resize(n);

        for (int i = 0; i < n; ++i) {
            const bool hL = std::isfinite(B.lb[i]);
            const bool hU = std::isfinite(B.ub[i]);
            B.hasL[i] = static_cast<uint8_t>(hL);
            B.hasU[i] = static_cast<uint8_t>(hU);
            B.sL[i] = hL ? clamp_min(x[i] - B.lb[i], consts::EPS_POS) : 1.0;
            B.sU[i] = hU ? clamp_min(B.ub[i] - x[i], consts::EPS_POS) : 1.0;
        }
        return B;
    }

    IPState st{};
    std::shared_ptr<LineSearcher> ls_;
    std::shared_ptr<regx::Regularizer> regularizer_ =
        std::make_shared<regx::Regularizer>();

    nb::object model;
    RichardsonExtrapolator extrapolator_;
    AdaptiveBarrierManager mu_mgr_; // <<< NEW
    bool mu_mgr_inited_ = false;    // <<< NEW

    bool use_richardson_ = true;
    double richardson_tolerance_ = 1e-8;
    int max_richardson_order_ = 3;

    ModelC *m_ = nullptr;
    InteriorPointStepper(nb::object cfg, ModelC *cfg_model)
        : cfg_(std::move(cfg)), m_(cfg_model) {
        load_defaults_();
        load_gondzio_defaults_();
        std::shared_ptr<Funnel> funnel = std::shared_ptr<Funnel>();
        ls_ = std::make_shared<LineSearcher>(cfg_, nb::none(), funnel);
    }

    double colnorms(const std::optional<spmat> &J) {
        if (!J || J->nonZeros() == 0)
            return 1.0;
        dvec col_sums = dvec::Zero(J->cols());
        for (int k = 0; k < J->outerSize(); ++k) {
            double sum = 0.0;
            for (spmat::InnerIterator it(*J, k); it; ++it) {
                sum += it.value() * it.value();
            }
            col_sums(k) = std::sqrt(std::max(sum, 1e-16));
        }
        return std::max(
            1.0,
            col_sums.mean()); // Use mean instead of median (see next error)
    }

    std::optional<dvec> compute_equality_shift(const std::optional<spmat> &JE,
                                               const std::optional<dvec> &cE) {
        if (!cE || cE->size() == 0)
            return std::nullopt;
        double sigma_E = get_attr_or<double>(cfg_, "ip_sigma_E", 0.1) *
                         std::max(1.0, colnorms(JE));
        dvec rE = -sigma_E * (*cE);
        double max_shift = get_attr_or<double>(cfg_, "ip_max_shift", 1e-2);
        rE.cwiseMin(max_shift).cwiseMax(-max_shift);
        return rE;
    }

    double adaptive_shift_slack_(const dvec &s, const dvec &cI, const dvec &lam,
                                 int it) const {
        if (!s.size())
            return 0.0;
        double base = get_attr_or<double>(cfg_, "ip_shift_tau", 1e-3);
        double min_s = s.minCoeff();
        double max_v = (cI.size() > 0) ? cI.cwiseAbs().maxCoeff() : 0.0;
        double active_tol = get_attr_or<double>(cfg_, "ip_active_tol", 1e-3);
        double mu_target = get_attr_or<double>(cfg_, "ip_mu_target", 1e-4);
        double comp_shift = 0.0;
        if (cI.size() > 0 && lam.size() == cI.size()) {
            for (int i = 0; i < cI.size(); ++i) {
                if (cI[i] >= -active_tol) {
                    double lam_i = std::max(std::abs(lam[i]), 1e-12);
                    comp_shift = std::max(
                        comp_shift, std::max(0.0, mu_target / lam_i - cI[i]));
                }
            }
        }
        if (min_s < 1e-6 || max_v > 1e2)
            return std::min(1.0, base * (1.0 + 0.1 * it) + comp_shift);
        if (min_s > 1e-2 && max_v < 1e-2)
            return clamp_min(base * (1.0 - 0.05 * it) + comp_shift, 0.0);
        return base + comp_shift;
    }

    void init_mu_manager_from_cfg_() { // <<< NEW
        AdaptiveBarrierConfig acfg;

        // Map strings from cfg_ to enums (with safe defaults)
        const auto mu_strategy_str =
            get_attr_or<std::string>(cfg_, "mu_strategy", "adaptive");
        acfg.mu_strategy = (mu_strategy_str == "monotone")
                               ? MuStrategy::Monotone
                               : MuStrategy::Adaptive;

        const auto mu_oracle_str =
            get_attr_or<std::string>(cfg_, "mu_oracle", "quality-function");
        acfg.mu_oracle = (mu_oracle_str == "loqo") ? MuOracle::LOQO
                                                   : MuOracle::QualityFunction;

        // Typical IPOPT-style knobs (honor if present)
        acfg.mu_init = get_attr_or<double>(cfg_, "mu_init", 1e-1);
        acfg.mu_min = get_attr_or<double>(cfg_, "mu_min", 1e-14);
        acfg.mu_max = get_attr_or<double>(cfg_, "mu_max", 1e2);
        acfg.mu_target = get_attr_or<double>(cfg_, "mu_target", 0.0);

        acfg.sigma_min = get_attr_or<double>(cfg_, "sigma_min", 0.05);
        acfg.sigma_max = get_attr_or<double>(cfg_, "sigma_max", 0.95);
        acfg.sigma_pow = get_attr_or<double>(cfg_, "sigma_pow", 3.0);

        // Monotone mode helpers
        acfg.mu_linear_decrease_factor =
            get_attr_or<double>(cfg_, "mu_linear_decrease_factor", 0.2);
        acfg.barrier_tol_factor =
            get_attr_or<double>(cfg_, "barrier_tol_factor", 0.1);

        // Progress/near-solution (keep your defaults if set)
        acfg.fast_progress_thr =
            get_attr_or<double>(cfg_, "fast_progress_thr", 0.35);
        acfg.stall_thr = get_attr_or<double>(cfg_, "stall_thr", 0.05);
        acfg.stall_k_max = get_attr_or<int>(cfg_, "stall_k_max", 3);
        acfg.sigma_shrink_on_stall =
            get_attr_or<double>(cfg_, "sigma_shrink_on_stall", 0.5);
        acfg.sigma_boost_on_fast =
            get_attr_or<double>(cfg_, "sigma_boost_on_fast", 1.2);
        acfg.near_cap_c = get_attr_or<double>(cfg_, "near_cap_c", 1e-2);
        acfg.near_cap_enable_thr =
            get_attr_or<double>(cfg_, "near_cap_enable_thr", 1e-6);

        // Acceptable termination (outer solver uses; exposed for completeness)
        acfg.acceptable_tol = get_attr_or<double>(cfg_, "acceptable_tol", 1e-6);
        acfg.acceptable_iter = get_attr_or<int>(cfg_, "acceptable_iter", 15);
        acfg.acceptable_dual_inf_tol =
            get_attr_or<double>(cfg_, "acceptable_dual_inf_tol", 1e10);
        acfg.acceptable_constr_viol_tol =
            get_attr_or<double>(cfg_, "acceptable_constr_viol_tol", 1e-2);
        acfg.acceptable_compl_inf_tol =
            get_attr_or<double>(cfg_, "acceptable_compl_inf_tol", 1e-2);
        acfg.acceptable_obj_change_tol =
            get_attr_or<double>(cfg_, "acceptable_obj_change_tol", 1e-20);

        mu_mgr_ = AdaptiveBarrierManager(acfg);
        mu_mgr_.reset();
        mu_mgr_inited_ = true;
    }
    std::tuple<dvec, dvec, dvec, SolverInfo>
    step(const dvec &x, const dvec &lam, const dvec &nu, int it,
         std::optional<IPState> /*ip_state_opt*/ = std::nullopt) {

#if IP_PROFILE
        IP_TIMER("step()");
#endif

        // -------------------- Init / state --------------------
        if (!st.initialized)
            st = state_from_model_(x);
        if (!mu_mgr_inited_) {
            init_mu_manager_from_cfg_();
            const double mu_init = mu_mgr_.config().mu_init;
            if (st.mu <= 0)
                st.mu = mu_init;
        }

        const int n = static_cast<int>(x.size());
        const int mI = st.mI;
        const int mE = st.mE;

        dvec s = st.s;
        dvec lmb = st.lam;
        dvec nuv = st.nu;
        dvec zL = st.zL;
        dvec zU = st.zU;
        double mu = st.mu;

        const bool use_shifted =
            get_attr_or<bool>(cfg_, "ip_use_shifted_barrier", false);
        double tau_shift = use_shifted ? st.tau_shift : 0.0;
        const bool shift_adapt =
            get_attr_or<bool>(cfg_, "ip_shift_adaptive", true);

        IP_LAP("init/state");

        // -------------------- Small helpers (only de-duped bits)
        // --------------------
        struct EvalPack {
            double f{};
            dvec g, cI, cE;
            spmat JI, JE;
            double theta{};
        };

        auto eval_all_at = [&](const dvec &X, int n_, int mI_,
                               int mE_) -> EvalPack {
            m_->eval_all(X);
            EvalPack E;
            E.f = m_->get_f().value_or(0.0);
            E.g = m_->get_g().value_or(dvec::Zero(n_));
            E.cI = (mI_ > 0) ? m_->get_cI().value() : dvec::Zero(mI_);
            E.cE = (mE_ > 0) ? m_->get_cE().value() : dvec::Zero(mE_);
            E.JI = (mI_ > 0) ? m_->get_JI().value() : spmat(mI_, n_);
            E.JE = (mE_ > 0) ? m_->get_JE().value() : spmat(mE_, n_);
            E.theta = m_->constraint_violation(X);
            return E;
        };

        auto build_r_d = [&](const EvalPack &E, const dvec &lmb_in,
                             const dvec &nu_in, const dvec &zL_in,
                             const dvec &zU_in) -> dvec {
            dvec r_d = E.g;
            if (E.JI.nonZeros())
                r_d.noalias() += E.JI.transpose() * lmb_in;
            if (E.JE.nonZeros())
                r_d.noalias() += E.JE.transpose() * nu_in;
            r_d -= zL_in;
            r_d += zU_in;
            return r_d;
        };

        auto aggregated_error =
            [&](const EvalPack &E, const dvec &x_in, const dvec &lam_in,
                const dvec &nu_in, const dvec &zL_in, const dvec &zU_in,
                double mu_in, const dvec &s_in, const Bounds &B_in, int n_,
                int mE_, int mI_, bool use_shifted_, double tau_shift_,
                double bound_shift_) -> double {
            dvec r_d = build_r_d(E, lam_in, nu_in, zL_in, zU_in);

            const double s_max = get_attr_or<double>(cfg_, "ip_s_max", 100.0);
            const int denom_ct = static_cast<int>(s_in.size()) + mE_ + n_;
            const double sum_mults = lam_in.lpNorm<1>() + nu_in.lpNorm<1>() +
                                     zL_in.lpNorm<1>() + zU_in.lpNorm<1>();
            const double s_d =
                clamp_min(sum_mults / clamp_min(denom_ct, 1), s_max) / s_max;
            const double s_c =
                clamp_min((zL_in.lpNorm<1>() + zU_in.lpNorm<1>()) /
                              clamp_min(n_, 1),
                          s_max) /
                s_max;

            const double comp_box =
                detail::comp_inf_norm(s_in, lam_in, zL_in, zU_in, B_in, mu_in,
                                      use_shifted_, tau_shift_, bound_shift_);

            return std::max({safe_inf_norm(r_d) / s_d,
                             (mE_ > 0) ? safe_inf_norm(E.cE) : 0.0,
                             (mI_ > 0) ? safe_inf_norm(E.cI) : 0.0,
                             comp_box / s_c});
        };

        auto assemble_W = [&](const spmat &H_in, const dvec &Sigma_x,
                              const std::optional<spmat> &JI_opt,
                              const dvec &Sigma_s_opt) -> spmat {
            spmat W = H_in;
            if (Sigma_x.size())
                W.diagonal().array() += Sigma_x.array();
            if (JI_opt && JI_opt->rows() > 0 &&
                Sigma_s_opt.size() == JI_opt->rows()) {
                spmat JIc = *JI_opt;
                W += JIc.transpose() * Sigma_s_opt.asDiagonal() * JIc;
                W.makeCompressed();
            }
            return W;
        };

        // stabilizing clamp + floor (used pre-manager)
        auto preguard_mu = [&](double mu_in, double mu_aff, double sigma_pred,
                               double comp_now, int mI_) -> double {
            const double mu_min_cfg =
                get_attr_or<double>(cfg_, "ip_mu_min", 1e-12);
            const double comp_cap_factor =
                get_attr_or<double>(cfg_, "ip_mu_comp_cap_factor", 10.0);
            const bool use_mI_scaling =
                get_attr_or<bool>(cfg_, "ip_mu_comp_cap_scale_by_mI", true);
            const double hard_mu_upper =
                get_attr_or<double>(cfg_, "ip_mu_hard_upper", 10.0);

            const double mI_scale = use_mI_scaling ? clamp_min(mI_, 1) : 1.0;
            const double cap_from_comp =
                std::min(comp_now * mI_scale, hard_mu_upper);
            const double floor_from_aff =
                std::max(mu_min_cfg, sigma_pred * std::max(mu_aff, mu_min_cfg));

            double mu_pre = mu_in;
            if (comp_now * mI_scale > comp_cap_factor * mu_pre)
                mu_pre = cap_from_comp;
            mu_pre = std::max(mu_pre, floor_from_aff);
            mu_pre = std::clamp(mu_pre, mu_mgr_.config().mu_min,
                                mu_mgr_.config().mu_max);
            return std::min(mu_pre, cap_from_comp); // optional post-cap
        };

        auto call_manager = [&](double mu0, double mu_aff, double err_cur,
                                double err_prev_, double rp, double rd,
                                double comp_now, double a_pri, double a_dua,
                                int it_) -> double {
            AdaptiveBarrierManager::Inputs in{
                .mu = std::max(mu0, 1e-16),
                .mu_aff = std::max(mu_aff, 1e-16),
                .kkt_norm = std::max(err_cur, 1e-16),
                .kkt_norm_prev = std::max(err_prev_, 1e-16),
                .rp_norm = rp,
                .rd_norm = rd,
                .comp_mu = std::max(comp_now, 1e-16),
                .alpha_pri = a_pri,
                .alpha_dua = a_dua,
                .step_accepted = true,
                .iter = it_};
            return mu_mgr_.update(in).mu_new;
        };

        // -------------------- Evaluate model at x --------------------
        EvalPack E = eval_all_at(x, n, mI, mE);
        IP_LAP("eval_all(x)");

        Bounds B = get_bounds(x);
        IP_LAP("bounds(get)");

        if (use_shifted && shift_adapt) {
            tau_shift = adaptive_shift_slack_(s, E.cI, lam, it);
            st.tau_shift = tau_shift;
        }
        IP_LAP("bounds & shifts (A)");

        double bound_shift =
            use_shifted ? get_attr_or<double>(cfg_, "ip_shift_bounds", 0.0)
                        : 0.0;
        if (use_shifted && shift_adapt)
            bound_shift = adaptive_shift_bounds_(x, B, it);
        IP_LAP("bounds & shifts (B)");

        const double tol = get_attr_or<double>(cfg_, "tol", 1e-8);

        // -------------------- Quick convergence check --------------------
        const double err_prev =
            aggregated_error(E, x, lmb, nuv, zL, zU, mu, s, B, n, mE, mI,
                             use_shifted, tau_shift, bound_shift);
        IP_LAP("compute_error_(x)");

        if (err_prev <= tol) {
            SolverInfo info;
            info.mode = "ip";
            info.step_norm = 0.0;
            info.accepted = true;
            info.converged = true;
            info.f = E.f;
            info.theta = E.theta;
            info.stat = safe_inf_norm(E.g);
            info.ineq = (mI > 0)
                            ? safe_inf_norm((E.cI.array().max(0.0)).matrix())
                            : 0.0;
            info.eq = (mE > 0) ? safe_inf_norm(E.cE) : 0.0;
            info.comp =
                detail::complementarity(s, lmb, mu, tau_shift, use_shifted);
            info.ls_iters = 0;
            info.alpha = 0.0;
            info.rho = 0.0;
            info.tr_radius = 0.0;
            info.mu = mu;
            return {x, lmb, nuv, info};
        }

        // -------------------- Sigmas & Hessian --------------------
        const double eps_abs = get_attr_or<double>(cfg_, "sigma_eps_abs", 1e-8);
        const double cap_add = get_attr_or<double>(cfg_, "sigma_cap", 1e8);

        Sigmas Sg =
            detail::build_sigmas(zL, zU, B, lmb, s, E.cI, tau_shift,
                                 bound_shift, use_shifted, eps_abs, cap_add);
        IP_LAP("build_sigmas");

        auto H0 = m_->hess(x, lmb, nuv);
        IP_LAP("get Hessian");
        auto [H, reg_info] = regularizer_->regularize(H0, it);
        H.makeCompressed();
        IP_LAP("regularize(H)");

        spmat W =
            assemble_W(H, Sg.Sigma_x,
                       (mI > 0 && E.JI.nonZeros()) ? std::optional<spmat>(E.JI)
                                                   : std::nullopt,
                       Sg.Sigma_s);
        IP_LAP("assemble W");

        // -------------------- Residuals --------------------
        dvec r_d = build_r_d(E, lmb, nuv, zL, zU);
        dvec r_pE = (mE > 0) ? E.cE : dvec();
        dvec r_pI = (mI > 0) ? (E.cI + s) : dvec();
        IP_LAP("build residuals");

        std::optional<dvec> r_pE_shifted = r_pE;
        if (mE > 0) {
            auto rE = compute_equality_shift(
                E.JE.nonZeros() ? std::optional<spmat>(E.JE) : std::nullopt,
                r_pE);
            if (rE)
                r_pE_shifted = r_pE + *rE;
        }

        // -------------------- Mehrotra + Gondzio --------------------
        const auto [alpha_aff, mu_aff, sigma_pred, gondzio_step] =
            mehrotra_with_gondzio_corrections_(
                W, r_d,
                (mE > 0 && E.JE.nonZeros()) ? std::optional<spmat>(E.JE)
                                            : std::nullopt,
                r_pE_shifted,
                (mI > 0 && E.JI.nonZeros()) ? std::optional<spmat>(E.JI)
                                            : std::nullopt,
                (mI > 0 && r_pI.size()) ? std::optional<dvec>(r_pI)
                                        : std::nullopt,
                s, lmb, zL, zU, B, use_shifted, tau_shift, bound_shift, mu,
                E.theta, Sg);
        IP_LAP("mehrotra+gondzio");

        dvec dx = gondzio_step.dx;
        dvec dnu = (mE > 0 && gondzio_step.dnu.size() == mE) ? gondzio_step.dnu
                                                             : dvec::Zero(mE);
        dvec ds = gondzio_step.ds;
        dvec dlam = gondzio_step.dlam;
        dvec dzL = gondzio_step.dzL;
        dvec dzU = gondzio_step.dzU;

        // -------------------- μ pre-guard + manager (pre-LS)
        // --------------------
        {
            const double comp_now =
                detail::complementarity(s, lmb, mu, tau_shift, use_shifted);
            double mu_pre = preguard_mu(mu, mu_aff, sigma_pred, comp_now, mI);
            const double rp = std::max({(mE > 0) ? safe_inf_norm(E.cE) : 0.0,
                                        (mI > 0) ? safe_inf_norm(E.cI) : 0.0});
            const double rd = safe_inf_norm(E.g);
            mu = call_manager(mu_pre, mu_aff,
                              /*err_cur*/ err_prev, /*err_prev*/ err_prev, rp,
                              rd, comp_now,
                              /*alpha_pri*/ 1.0, /*alpha_dua*/ 1.0, it);
        }
        IP_LAP("barrier manager (pre)");

        // -------------------- Trust region clipping --------------------
        const double dx_cap = get_attr_or<double>(cfg_, "ip_dx_max", 1e3);
        const double nx = dx.norm();
        if (nx > dx_cap && nx > 0.0) {
            const double sc = dx_cap / nx;
            dx *= sc;
            dzL *= sc;
            dzU *= sc;
            if (mI > 0) {
                ds *= sc;
                dlam *= sc;
            }
        }
        IP_LAP("barrier & TR clip");

        // -------------------- Fraction-to-boundary + LS --------------------
        const double tau_pri = get_attr_or<double>(
            cfg_, "ip_tau_pri", get_attr_or<double>(cfg_, "ip_tau", 0.995));
        const double tau_dual = get_attr_or<double>(
            cfg_, "ip_tau_dual", get_attr_or<double>(cfg_, "ip_tau", 0.995));
        const double a_ftb = detail::alpha_ftb_vec(
            x, dx, (mI ? s : dvec()), (mI ? ds : dvec()), lmb,
            (mI ? dlam : dvec()), B, tau_pri, tau_dual);
        const double alpha_max =
            std::min(a_ftb, get_attr_or<double>(cfg_, "ip_alpha_max", 1.0));

        double alpha = std::min(1.0, alpha_max);
        int ls_iters = 0;
        bool needs_restoration = false;

        auto ls_res =
            ls_->search(m_, x, dx, (mI ? s : dvec()), (mI ? ds : dvec()), mu,
                        E.g.dot(dx), E.theta, alpha_max);
        alpha = std::get<0>(ls_res);
        ls_iters = std::get<1>(ls_res);
        needs_restoration = std::get<2>(ls_res);
        IP_LAP("line search");

        // -------------------- Restoration path --------------------
        const double ls_min_alpha = get_attr_or<double>(
            cfg_, "ls_min_alpha",
            get_attr_or<double>(cfg_, "ip_alpha_min", 1e-10));
        if (alpha <= ls_min_alpha && needs_restoration) {
            dvec dxf = -E.g;
            const double ng = dxf.norm();
            if (ng > 0)
                dxf /= ng;
            const double a_safe = std::min(alpha_max, 1e-2);
            dvec x_new = x + a_safe * dxf;

            SolverInfo info;
            info.mode = "ip";
            info.step_norm = (x_new - x).norm();
            info.accepted = true;
            info.converged = false;
            info.f = 0.0; // optionally eval f at x_new if needed
            info.theta = m_->constraint_violation(x_new);
            info.stat = 0.0;
            info.ineq = 0.0;
            info.eq = 0.0;
            info.comp = 0.0;
            info.ls_iters = ls_iters;
            info.alpha = 0.0;
            info.rho = 0.0;
            info.tr_radius = 0.0;
            info.mu = mu;
            return {x_new, lmb, nuv, info};
        }

        // -------------------- Accept step & evaluate new point
        // --------------------
        dvec x_new = x + alpha * dx;
        dvec s_new = mI ? (s + alpha * ds) : s;
        dvec lmb_new = mI ? (lmb + alpha * dlam) : lmb;
        dvec nu_new = mE ? (nuv + alpha * dnu) : nuv;
        dvec zL_new = zL + alpha * dzL;
        dvec zU_new = zU + alpha * dzU;

        Bounds Bn = get_bounds(x_new);
        detail::cap_bound_duals_sigma_box(zL_new, zU_new, Bn, use_shifted,
                                          bound_shift, mu, 1e10);
        IP_LAP("apply step & cap");

        EvalPack En = eval_all_at(x_new, n, mI, mE);
        IP_LAP("eval_all(x_new)");

        // -------------------- KKT residuals (new) --------------------
        dvec r_d_new = build_r_d(En, lmb_new, nu_new, zL_new, zU_new);

        KKT kkt_new;
        kkt_new.stat = safe_inf_norm(r_d_new);
        kkt_new.ineq =
            (mI > 0) ? safe_inf_norm((En.cI.array().max(0.0)).matrix()) : 0.0;
        kkt_new.eq = (mE > 0) ? safe_inf_norm(En.cE) : 0.0;
        kkt_new.comp =
            detail::comp_inf_norm(s_new, lmb_new, zL_new, zU_new, Bn, mu,
                                  use_shifted, tau_shift, bound_shift);
        const double tol_outer = tol;
        const bool converged =
            (kkt_new.stat <= tol_outer && kkt_new.ineq <= tol_outer &&
             kkt_new.eq <= tol_outer && kkt_new.comp <= tol_outer &&
             mu <= tol_outer / 10.0);
        IP_LAP("KKT new");

        // -------------------- Aggregated error (new) --------------------
        const double err_new = aggregated_error(
            En, x_new, lmb_new, nu_new, zL_new, zU_new, mu, s_new, Bn, n, mE,
            mI, use_shifted, tau_shift, bound_shift);
        IP_LAP("compute_error_(x_new)");

        // -------------------- Barrier manager update (post-LS)
        // --------------------
        {
            const double comp_now_new = detail::complementarity(
                s_new, lmb_new, mu, tau_shift, use_shifted);
            mu = call_manager(mu, std::max(mu_aff, 1e-16), err_new, err_prev,
                              std::max(kkt_new.ineq, kkt_new.eq), kkt_new.stat,
                              comp_now_new,
                              /*alpha_pri*/ alpha, /*alpha_dua*/ alpha, it);
        }
        IP_LAP("barrier manager (post)");

        // -------------------- Pack & update state --------------------
        SolverInfo info;
        info.mode = "ip";
        info.step_norm = (x_new - x).norm();
        info.accepted = true;
        info.converged = converged;
        info.f = En.f;
        info.theta = En.theta;
        info.stat = kkt_new.stat;
        info.ineq = kkt_new.ineq;
        info.eq = kkt_new.eq;
        info.comp = kkt_new.comp;
        info.ls_iters = ls_iters;
        info.alpha = alpha;
        info.rho = 0.0;
        info.tr_radius = 0.0;
        info.mu = mu;
        info.shifted_barrier = use_shifted;
        info.tau_shift = tau_shift;
        info.bound_shift = bound_shift;

        st.s = std::move(s_new);
        st.lam = std::move(lmb_new);
        st.nu = std::move(nu_new);
        st.zL = std::move(zL_new);
        st.zU = std::move(zU_new);
        st.mu = mu;
        st.tau_shift = tau_shift;

        return {x_new, st.lam, st.nu, info};
    }

private:
    nb::object cfg_, hess_;
    std::unordered_map<std::string, dvec> kkt_cache_;
    std::shared_ptr<kkt::KKTReusable> cached_kkt_solver_{};
    spmat cached_kkt_matrix_;
    bool kkt_factorization_valid_ = false;

    bool matrices_equal(const spmat &A, const spmat &B, double tol = 1e-4) {
        if (A.rows() != B.rows() || A.cols() != B.cols())
            return false;
        return (A - B).norm() < tol;
    }

    void load_defaults_() {
        auto set_if_missing = [&](const char *name, nb::object v) {
            if (!pyu::has_attr(cfg_, name))
                cfg_.attr(name) = v;
        };
        set_if_missing("ip_exact_hessian", nb::bool_(true));
        set_if_missing("ip_hess_reg0", nb::float_(1e-4));
        set_if_missing("ip_eq_reg", nb::float_(1e-4));
        set_if_missing("ip_use_shifted_barrier", nb::bool_(true));
        set_if_missing("ip_shift_tau", nb::float_(0.01));
        set_if_missing("ip_shift_bounds", nb::float_(0.1));
        set_if_missing("ip_shift_adaptive", nb::bool_(true));
        set_if_missing("ip_mu_init", nb::float_(1e-2));
        set_if_missing("ip_mu_min", nb::float_(1e-12));
        set_if_missing("ip_sigma_power", nb::float_(3.0));
        set_if_missing("ip_tau_pri", nb::float_(0.995));
        set_if_missing("ip_tau_dual", nb::float_(0.99));
        set_if_missing("ip_tau", nb::float_(0.995));
        set_if_missing("ip_alpha_max", nb::float_(1.0));
        set_if_missing("ip_dx_max", nb::float_(1e3));
        set_if_missing("ip_theta_clip", nb::float_(1e-2));
        set_if_missing("sigma_eps_abs", nb::float_(1e-8));
        set_if_missing("sigma_cap", nb::float_(1e8));
        set_if_missing("ip_kkt_method", nb::str("hykkt"));
        set_if_missing("tol", nb::float_(1e-6));
        set_if_missing("ls_backtrack", nb::float_(get_attr_or<double>(
                                           cfg_, "ip_alpha_backtrack", 0.5)));
        set_if_missing("ls_armijo_f", nb::float_(get_attr_or<double>(
                                          cfg_, "ip_armijo_coeff", 1e-4)));
        set_if_missing("ls_max_iter",
                       nb::int_(get_attr_or<int>(cfg_, "ip_ls_max", 5)));
        set_if_missing("ls_min_alpha", nb::float_(get_attr_or<double>(
                                           cfg_, "ip_alpha_min", 1e-10)));
    }

    // --- State bootstrap ---
    [[nodiscard]] IPState state_from_model_(const dvec &x) {
        IPState s{};
        s.mI = m_->get_mI();
        s.mE = m_->get_mE();

        m_->eval_all(x, std::vector<std::string>{"cI", "cE"});
        dvec cI = (s.mI > 0) ? m_->get_cI().value() : dvec::Zero(s.mI);

        const double mu0 =
            clamp_min(get_attr_or<double>(cfg_, "ip_mu_init", 1e-2), 1e-12);
        const bool use_shifted =
            get_attr_or<bool>(cfg_, "ip_use_shifted_barrier", true);
        const double tau_shift =
            use_shifted ? get_attr_or<double>(cfg_, "ip_shift_tau", 0.1) : 0.0;
        const double bound_shift =
            use_shifted ? get_attr_or<double>(cfg_, "ip_shift_bounds", 0.1)
                        : 0.0;

        if (s.mI > 0) {
            s.s.resize(s.mI);
            s.lam.resize(s.mI);
            for (int i = 0; i < s.mI; ++i) {
                s.s[i] = clamp_min(-cI[i] + 1e-3, 1.0);
                const double denom =
                    (tau_shift > 0.0) ? (s.s[i] + tau_shift) : s.s[i];
                s.lam[i] = clamp_min(mu0 / clamp_min(denom, 1e-12), 1e-8);
            }
        } else {
            s.s.resize(0);
            s.lam.resize(0);
        }

        s.nu = (s.mE > 0) ? dvec::Zero(s.mE) : dvec();

        Bounds B = get_bounds(x);
        s.zL = dvec::Zero(x.size());
        s.zU = dvec::Zero(x.size());
        for (int i = 0; i < x.size(); ++i) {
            if (B.hasL[i])
                s.zL[i] = clamp_min(
                    mu0 / clamp_min(B.sL[i] + bound_shift, 1e-12), 1e-8);
            if (B.hasU[i])
                s.zU[i] = clamp_min(
                    mu0 / clamp_min(B.sU[i] + bound_shift, 1e-12), 1e-8);
        }

        s.mu = mu0;
        s.tau_shift = tau_shift;
        s.initialized = true;
        return s;
    }

    [[nodiscard]] double adaptive_shift_bounds_(const dvec &x, const Bounds &B,
                                                int it) const {
        bool any = false;
        for (int i = 0; i < x.size(); ++i)
            if (B.hasL[i] || B.hasU[i]) {
                any = true;
                break;
            }
        if (!any)
            return 0.0;

        double min_gap = consts::INF;
        for (int i = 0; i < x.size(); ++i) {
            if (B.hasL[i])
                min_gap = std::min(min_gap, x[i] - B.lb[i]);
            if (B.hasU[i])
                min_gap = std::min(min_gap, B.ub[i] - x[i]);
        }
        double b0 = get_attr_or<double>(cfg_, "ip_shift_bounds", 0.0);
        if (min_gap < 1e-8)
            return std::min(1.0, clamp_min(b0, 1e-3) * (1 + 0.05 * it));
        if (min_gap > 1e-2)
            return clamp_min(b0 * 0.9, 0.0);
        return b0;
    }

    // --- KKT solve (with reuse) ---
    [[nodiscard]] KKTResult solve_KKT_(const spmat &W, const dvec &rhs_x,
                                       const std::optional<spmat> &JE,
                                       const std::optional<dvec> &rpE,
                                       std::string_view method_in) {
        const int mE = (JE && JE->rows() > 0) ? JE->rows() : 0;

        kkt::dvec r1 = rhs_x;
        std::optional<kkt::dvec> r2;
        if (mE > 0)
            r2 = rpE ? (-(*rpE)).eval() : kkt::dvec::Zero(mE);

        std::string method_cpp = std::string(method_in);
        if (mE == 0 && (method_cpp == "hykkt" || method_cpp == "hykkt_cholmod"))
            method_cpp = "ldl";

        const bool can_reuse = kkt_factorization_valid_ && cached_kkt_solver_ &&
                               matrices_equal(W, cached_kkt_matrix_);
        if (can_reuse) {
            auto [dx, dy] = cached_kkt_solver_->solve(rhs_x, r2, 1e-8, 200);
            return {std::move(dx), std::move(dy), cached_kkt_solver_};
        }

        kkt::ChompConfig conf;
        auto &reg = kkt::default_registry();
        auto strat = reg.get(method_cpp);
        auto [dx, dy, reusable] = strat->factor_and_solve(
            m_, W, (mE > 0 ? std::optional<spmat>(*JE) : std::nullopt), r1, r2,
            conf, std::nullopt, kkt_cache_, 0.0, std::nullopt, true, true);

        cached_kkt_matrix_ = W;
        cached_kkt_solver_ = reusable;
        kkt_factorization_valid_ = true;
        return {std::move(dx), std::move(dy), std::move(reusable)};
    }

    // --- Mehrotra (affine) ---
    [[nodiscard]] std::tuple<double, double, double> mehrotra_affine_predictor_(
        const spmat &W, const dvec &r_d, const std::optional<spmat> &JE,
        const std::optional<dvec> &r_pE, const std::optional<spmat> &JI,
        const std::optional<dvec> &r_pI, const dvec &s, const dvec &lmb,
        const dvec &zL, const dvec &zU, const Bounds &B, bool use_shifted,
        double tau_shift, double bound_shift, double mu, double theta) {

        const bool haveJE = JE && JE->rows() > 0 && JE->cols() > 0;

        auto res = solve_KKT_(
            W, -r_d, haveJE ? std::optional<spmat>(*JE) : std::nullopt,
            (r_pE && r_pE->size() > 0) ? std::optional<dvec>(*r_pE)
                                       : std::nullopt,
            get_attr_or<std::string>(cfg_, "ip_kkt_method", "hykkt"));

        dvec dx_aff = std::move(res.dx);
        const int mI = static_cast<int>(s.size());
        const int n = static_cast<int>(zL.size());

        dvec ds_aff, dlam_aff;
        if (mI > 0 && JI) {
            ds_aff = -(r_pI.value() + (*JI) * dx_aff);
            dlam_aff = dvec(mI);
            for (int i = 0; i < mI; ++i) {
                const double d = use_shifted ? (s[i] + tau_shift) : s[i];
                dlam_aff[i] = sdiv(-(d * lmb[i]) - lmb[i] * ds_aff[i], d);
            }
        } else {
            ds_aff.resize(0);
            dlam_aff.resize(0);
        }

        dvec dzL_aff = dvec::Zero(n), dzU_aff = dvec::Zero(n);
        for (int i = 0; i < n; ++i) {
            if (B.hasL[i]) {
                const double d = (use_shifted && bound_shift > 0.0)
                                     ? (B.sL[i] + bound_shift)
                                     : B.sL[i];
                dzL_aff[i] = sdiv(-(d * zL[i]) - zL[i] * dx_aff[i], d);
            }
            if (B.hasU[i]) {
                const double d = (use_shifted && bound_shift > 0.0)
                                     ? (B.sU[i] + bound_shift)
                                     : B.sU[i];
                dzU_aff[i] = sdiv(-(d * zU[i]) + zU[i] * dx_aff[i], d);
            }
        }

        // step length
        double alpha_aff = 1.0;
        if (mI > 0) {
            for (int i = 0; i < mI; ++i)
                if (ds_aff[i] < 0.0)
                    alpha_aff =
                        std::min(alpha_aff,
                                 -s[i] / std::min(ds_aff[i], -consts::EPS_DIV));
            for (int i = 0; i < mI; ++i)
                if (dlam_aff[i] < 0.0)
                    alpha_aff = std::min(
                        alpha_aff,
                        -lmb[i] / std::min(dlam_aff[i], -consts::EPS_DIV));
        }
        for (int i = 0; i < n; ++i)
            if (B.hasL[i] && dx_aff[i] < 0.0)
                alpha_aff =
                    std::min(alpha_aff,
                             -B.sL[i] / std::min(dx_aff[i], -consts::EPS_DIV));
        for (int i = 0; i < n; ++i)
            if (B.hasL[i] && dzL_aff[i] < 0.0)
                alpha_aff = std::min(
                    alpha_aff, -zL[i] / std::min(dzL_aff[i], -consts::EPS_DIV));
        for (int i = 0; i < n; ++i) {
            const double mdx = -dx_aff[i];
            if (B.hasU[i] && mdx < 0.0)
                alpha_aff = std::min(
                    alpha_aff, -B.sU[i] / std::min(mdx, -consts::EPS_DIV));
        }
        for (int i = 0; i < n; ++i)
            if (B.hasU[i] && dzU_aff[i] < 0.0)
                alpha_aff = std::min(
                    alpha_aff, -zU[i] / std::min(dzU_aff[i], -consts::EPS_DIV));
        alpha_aff = clamp01(alpha_aff);

        const double mu_min = get_attr_or<double>(cfg_, "ip_mu_min", 1e-12);
        double sum_parts = 0.0;
        int denom_cnt = 0;

        if (mI > 0) {
            for (int i = 0; i < mI; ++i) {
                const double s_aff = s[i] + alpha_aff * ds_aff[i];
                const double lam_aff = lmb[i] + alpha_aff * dlam_aff[i];
                const double s_eff = use_shifted ? (s_aff + tau_shift) : s_aff;
                sum_parts += s_eff * lam_aff;
                ++denom_cnt;
            }
        }
        for (int i = 0; i < n; ++i)
            if (B.hasL[i]) {
                const double sL_aff = B.sL[i] + alpha_aff * dx_aff[i];
                const double zL_af = zL[i] + alpha_aff * dzL_aff[i];
                const double s_eff = (use_shifted && bound_shift > 0.0)
                                         ? (sL_aff + bound_shift)
                                         : sL_aff;
                sum_parts += s_eff * zL_af;
                ++denom_cnt;
            }
        for (int i = 0; i < n; ++i)
            if (B.hasU[i]) {
                const double sU_aff = B.sU[i] - alpha_aff * dx_aff[i];
                const double zU_af = zU[i] + alpha_aff * dzU_aff[i];
                const double s_eff = (use_shifted && bound_shift > 0.0)
                                         ? (sU_aff + bound_shift)
                                         : sU_aff;
                sum_parts += s_eff * zU_af;
                ++denom_cnt;
            }

        const double mu_aff =
            (denom_cnt > 0)
                ? clamp_min(sum_parts / clamp_min(denom_cnt, 1), mu_min)
                : clamp_min(mu, mu_min);

        const double pwr = get_attr_or<double>(cfg_, "ip_sigma_power", 3.0);
        double sigma =
            (alpha_aff > 0.9)
                ? 0.0
                : clamp(std::pow(1.0 - alpha_aff, 2) *
                            std::pow(mu_aff / clamp_min(mu, mu_min), pwr),
                        0.0, 1.0);

        const double theta_clip =
            get_attr_or<double>(cfg_, "ip_theta_clip", 1e-2);
        if (theta > theta_clip)
            sigma = clamp_min(sigma, 0.5);

        return {alpha_aff, mu_aff, sigma};
    }

    [[nodiscard]] double tr_radius_() const { return 0.0; }

    // ---------------- Gondzio configuration & logic ----------------

    // --- Gondzio config/logic ---
    struct GondzioConfig {
        int max_corrections = 2;
        double gamma_a = 0.1, gamma_b = 10.0;
        double beta_min = 0.1, beta_max = 10.0;
        double tau_min = 0.005;
        bool use_adaptive_gamma = true;
        double progress_threshold = 0.1;
        double soc_threshold = 0.1; // Threshold for triggering SOC
        int max_soc_steps = 3;      // Max number of SOC iterations
    } gondzio_config_;

    void load_gondzio_defaults_() {
        gondzio_config_.max_corrections =
            get_attr_or<int>(cfg_, "gondzio_max_corrections", 2);
        gondzio_config_.gamma_a =
            get_attr_or<double>(cfg_, "gondzio_gamma_a", 0.1);
        gondzio_config_.gamma_b =
            get_attr_or<double>(cfg_, "gondzio_gamma_b", 10.0);
        gondzio_config_.beta_min =
            get_attr_or<double>(cfg_, "gondzio_beta_min", 0.1);
        gondzio_config_.beta_max =
            get_attr_or<double>(cfg_, "gondzio_beta_max", 10.0);
        gondzio_config_.tau_min =
            get_attr_or<double>(cfg_, "gondzio_tau_min", 0.005);
        gondzio_config_.use_adaptive_gamma =
            get_attr_or<bool>(cfg_, "gondzio_adaptive_gamma", true);
        gondzio_config_.progress_threshold =
            get_attr_or<double>(cfg_, "gondzio_progress_threshold", 0.1);
        gondzio_config_.soc_threshold =
            get_attr_or<double>(cfg_, "soc_threshold", 0.1);
        gondzio_config_.max_soc_steps =
            get_attr_or<int>(cfg_, "max_soc_steps", 3);
    }

    struct GondzioStepData {
        dvec dx, dnu, ds, dlam, dzL, dzU;
        double alpha_pri = 1.0, alpha_dual = 1.0;
        double mu_target = 0.0;
        int correction_count = 0;
        bool use_correction = false;
        int soc_count = 0; // Track SOC iterations
    };

    // New SOC helper function
    [[nodiscard]] GondzioStepData soc_correction_(
        const spmat &W, const std::optional<spmat> &JE,
        const std::optional<spmat> &JI, const dvec &s, const dvec &lam,
        const dvec &zL, const dvec &zU, const dvec &dx_aff, const dvec &ds_aff,
        const dvec &dlam_aff, const dvec &dzL_aff, const dvec &dzU_aff,
        const Bounds &B, double alpha_aff, double mu_target, bool use_shifted,
        double tau_shift, double bound_shift, const Sigmas &Sg) {
        IP_TIMER("soc_correction");
        const int n = static_cast<int>(dx_aff.size());
        const int mI = static_cast<int>(s.size());
        dvec rhs_x = dvec::Zero(n);
        dvec rhs_s = dvec::Zero(mI);
        dvec cI_new = dvec::Zero(mI);

        // Compute constraint violations at the affine step
        if (mI > 0 && JI) {
            dvec x_aff = dx_aff * alpha_aff;
            m_->eval_all(x_aff); // Evaluate at affine step point
            cI_new = (mI > 0) ? m_->get_cI().value() : dvec::Zero(mI);
            for (int i = 0; i < mI; ++i) {
                const double s_aff = s[i] + alpha_aff * ds_aff[i];
                const double lam_aff = lam[i] + alpha_aff * dlam_aff[i];
                const double s_eff = use_shifted ? (s_aff + tau_shift) : s_aff;
                rhs_s[i] = mu_target - s_eff * lam_aff - cI_new[i];
            }
            if (JI->nonZeros() && Sg.Sigma_s.size() == mI)
                rhs_x.noalias() +=
                    JI->transpose() * (Sg.Sigma_s.asDiagonal() * rhs_s);
        }

        // Bound complementarity corrections
        for (int i = 0; i < n; ++i) {
            double bound_corr = 0.0;
            if (B.hasL[i]) {
                const double sL_aff = B.sL[i] + alpha_aff * dx_aff[i];
                const double zL_aff = zL[i] + alpha_aff * dzL_aff[i];
                const double s_eff =
                    use_shifted ? (sL_aff + bound_shift) : sL_aff;
                const double corrL = mu_target - s_eff * zL_aff;
                bound_corr += corrL / clamp_min(s_eff, consts::EPS_POS);
            }
            if (B.hasU[i]) {
                const double sU_aff = B.sU[i] - alpha_aff * dx_aff[i];
                const double zU_aff = zU[i] + alpha_aff * dzU_aff[i];
                const double s_eff =
                    use_shifted ? (sU_aff + bound_shift) : sU_aff;
                const double corrU = mu_target - s_eff * zU_aff;
                bound_corr -= corrU / clamp_min(s_eff, consts::EPS_POS);
            }
            rhs_x[i] += bound_corr;
        }

        // Solve KKT system for SOC step (reuse factorization)
        auto soc_res = solve_KKT_(
            W, rhs_x, JE, std::nullopt,
            get_attr_or<std::string>(cfg_, "ip_kkt_method", "hykkt"));
        GondzioStepData soc_step;
        soc_step.dx = soc_res.dx;
        soc_step.dnu = (JE && soc_res.dy.size() > 0)
                           ? soc_res.dy
                           : dvec::Zero(JE ? JE->rows() : 0);
        if (mI > 0 && JI) {
            soc_step.ds = -(cI_new + (*JI) * soc_step.dx);
            soc_step.dlam = dvec(mI);
            for (int i = 0; i < mI; ++i) {
                const double d = use_shifted ? (s[i] + tau_shift) : s[i];
                soc_step.dlam[i] =
                    sdiv(mu_target - d * lam[i] - lam[i] * soc_step.ds[i], d);
            }
        } else {
            soc_step.ds.resize(0);
            soc_step.dlam.resize(0);
        }
        std::tie(soc_step.dzL, soc_step.dzU) = detail::dz_bounds_from_dx_vec(
            soc_step.dx, zL, zU, B, bound_shift, use_shifted, mu_target, true);
        const double tau_pri = get_attr_or<double>(cfg_, "ip_tau_pri", 0.995);
        const double tau_dual = get_attr_or<double>(cfg_, "ip_tau_dual", 0.995);
        std::tie(soc_step.alpha_pri, soc_step.alpha_dual) =
            gondzio_step_lengths_(s, soc_step.ds, lam, soc_step.dlam, zL,
                                  soc_step.dzL, zU, soc_step.dzU, soc_step.dx,
                                  B, tau_pri, tau_dual);
        soc_step.mu_target = mu_target;
        soc_step.soc_count = 1;
        return soc_step;
    }
    [[nodiscard]] std::pair<double, double>
    gondzio_step_lengths_(const dvec &s, const dvec &ds, const dvec &lam,
                          const dvec &dlam, const dvec &zL, const dvec &dzL,
                          const dvec &zU, const dvec &dzU, const dvec &dx,
                          const Bounds &B, double tau_pri,
                          double tau_dual) const {

        double a_pri = 1.0, a_dual = 1.0;
        for (int i = 0; i < s.size(); ++i)
            if (ds[i] < 0.0)
                a_pri =
                    std::min(a_pri, -s[i] / std::min(ds[i], -consts::EPS_DIV));
        for (int i = 0; i < lam.size(); ++i)
            if (dlam[i] < 0.0)
                a_dual = std::min(
                    a_dual, -lam[i] / std::min(dlam[i], -consts::EPS_DIV));

        const int n = static_cast<int>(dx.size());
        for (int i = 0; i < n; ++i) {
            if (B.hasL[i] && dx[i] < 0.0)
                a_pri = std::min(a_pri, -(B.sL[i]) /
                                            std::min(dx[i], -consts::EPS_DIV));
            if (B.hasU[i] && dx[i] > 0.0)
                a_pri = std::min(a_pri,
                                 (B.sU[i]) / clamp_min(dx[i], consts::EPS_DIV));
        }
        for (int i = 0; i < n && i < zL.size(); ++i)
            if (B.hasL[i] && dzL[i] < 0.0)
                a_dual = std::min(a_dual,
                                  -zL[i] / std::min(dzL[i], -consts::EPS_DIV));
        for (int i = 0; i < n && i < zU.size(); ++i)
            if (B.hasU[i] && dzU[i] < 0.0)
                a_dual = std::min(a_dual,
                                  -zU[i] / std::min(dzU[i], -consts::EPS_DIV));

        return {clamp01(tau_pri * a_pri), clamp01(tau_dual * a_dual)};
    }

    [[nodiscard]] double
    centrality_measure_(const dvec &s, const dvec &lam, const dvec &ds,
                        const dvec &dlam, const dvec &zL, const dvec &zU,
                        const dvec &dzL, const dvec &dzU, const dvec &dx,
                        const Bounds &B, double alpha_pri, double alpha_dual,
                        double mu_target, bool use_shifted, double tau_shift,
                        double bound_shift) const {

        double min_ratio = std::numeric_limits<double>::infinity();
        double max_ratio = 0.0;
        int count = 0;

        for (int i = 0; i < s.size(); ++i) {
            const double s_new = s[i] + alpha_pri * ds[i];
            const double l_new = lam[i] + alpha_dual * dlam[i];
            if (s_new > consts::EPS_POS && l_new > consts::EPS_POS) {
                const double prod =
                    (use_shifted ? (s_new + tau_shift) : s_new) * l_new;
                const double ratio = prod / mu_target;
                min_ratio = std::min(min_ratio, ratio);
                max_ratio = std::max(max_ratio, ratio);
                ++count;
            }
        }

        const int n = static_cast<int>(dx.size());
        for (int i = 0; i < n && i < zL.size(); ++i)
            if (B.hasL[i]) {
                const double sL_new = B.sL[i] + alpha_pri * dx[i];
                const double zL_new = zL[i] + alpha_dual * dzL[i];
                if (sL_new > consts::EPS_POS && zL_new > consts::EPS_POS) {
                    const double prod =
                        (use_shifted ? (sL_new + bound_shift) : sL_new) *
                        zL_new;
                    const double ratio = prod / mu_target;
                    min_ratio = std::min(min_ratio, ratio);
                    max_ratio = std::max(max_ratio, ratio);
                    ++count;
                }
            }
        for (int i = 0; i < n && i < zU.size(); ++i)
            if (B.hasU[i]) {
                const double sU_new = B.sU[i] - alpha_pri * dx[i];
                const double zU_new = zU[i] + alpha_dual * dzU[i];
                if (sU_new > consts::EPS_POS && zU_new > consts::EPS_POS) {
                    const double prod =
                        (use_shifted ? (sU_new + bound_shift) : sU_new) *
                        zU_new;
                    const double ratio = prod / mu_target;
                    min_ratio = std::min(min_ratio, ratio);
                    max_ratio = std::max(max_ratio, ratio);
                    ++count;
                }
            }
        if (count == 0)
            return 1.0;
        return (min_ratio > 0) ? (max_ratio / min_ratio)
                               : std::numeric_limits<double>::infinity();
    }

    [[nodiscard]] bool should_apply_gondzio_(double centrality_measure,
                                             double alpha_max) const {
        return (centrality_measure > gondzio_config_.gamma_b ||
                centrality_measure < gondzio_config_.gamma_a) &&
               alpha_max >= gondzio_config_.tau_min;
    }

    [[nodiscard]] std::pair<dvec, dvec> gondzio_corr_rhs_(
        const dvec &s, const dvec &lam, const dvec &ds, const dvec &dlam,
        const dvec &zL, const dvec &zU, const dvec &dzL, const dvec &dzU,
        const dvec &dx, const Bounds &B, const spmat &JI, double alpha_pri,
        double alpha_dual, double mu_target, double centrality_measure,
        bool use_shifted, double tau_shift, double bound_shift,
        const Sigmas &Sg) const {

        const int n = static_cast<int>(dx.size());
        const int mI = static_cast<int>(s.size());

        double beta = 1.0;
        if (centrality_measure > gondzio_config_.gamma_b)
            beta = clamp(2.0 * centrality_measure / gondzio_config_.gamma_b,
                         gondzio_config_.beta_min, gondzio_config_.beta_max);
        else if (centrality_measure < gondzio_config_.gamma_a)
            beta = clamp(gondzio_config_.gamma_a / (2.0 * centrality_measure),
                         gondzio_config_.beta_min, gondzio_config_.beta_max);

        dvec rhs_x = dvec::Zero(n);
        dvec rhs_s = dvec::Zero(mI);

        if (mI > 0) {
            for (int i = 0; i < mI; ++i) {
                const double s_pred = s[i] + alpha_pri * ds[i];
                const double lam_pred = lam[i] + alpha_dual * dlam[i];
                const double s_eff =
                    use_shifted ? (s_pred + tau_shift) : s_pred;
                const double rc =
                    -ds[i] * dlam[i] + beta * mu_target - s_eff * lam_pred;
                rhs_s[i] = rc;
            }
            if (JI.nonZeros() && Sg.Sigma_s.size() == mI)
                rhs_x.noalias() +=
                    JI.transpose() * (Sg.Sigma_s.asDiagonal() * rhs_s);
        }

        for (int i = 0; i < n; ++i) {
            double bound_corr = 0.0;
            if (i < zL.size() && B.hasL[i]) {
                const double sL_pred = B.sL[i] + alpha_pri * dx[i];
                const double zL_pred = zL[i] + alpha_dual * dzL[i];
                const double s_eff =
                    use_shifted ? (sL_pred + bound_shift) : sL_pred;
                const double corrL =
                    -dx[i] * dzL[i] + beta * mu_target - s_eff * zL_pred;
                bound_corr += corrL / clamp_min(s_eff, consts::EPS_POS);
            }
            if (i < zU.size() && B.hasU[i]) {
                const double sU_pred = B.sU[i] - alpha_pri * dx[i];
                const double zU_pred = zU[i] + alpha_dual * dzU[i];
                const double s_eff =
                    use_shifted ? (sU_pred + bound_shift) : sU_pred;
                const double corrU =
                    dx[i] * dzU[i] + beta * mu_target - s_eff * zU_pred;
                bound_corr -= corrU / clamp_min(s_eff, consts::EPS_POS);
            }
            rhs_x[i] += bound_corr;
        }
        return {rhs_x, rhs_s};
    }

    [[nodiscard]] GondzioStepData gondzio_multi_(
        const spmat &W, const dvec &r_d, const std::optional<spmat> &JE,
        const std::optional<dvec> &r_pE, const std::optional<spmat> &JI,
        const std::optional<dvec> &r_pI, const dvec &s, const dvec &lam,
        const dvec &zL, const dvec &zU, const Bounds &B, double mu_target,
        bool use_shifted, double tau_shift, double bound_shift,
        const Sigmas &Sg, const dvec &base_dx, const dvec &base_dnu) {
        IP_TIMER("gondzio_multi");
        GondzioStepData R;
        R.dx = base_dx;
        R.dnu = base_dnu;
        R.mu_target = mu_target;

        const int mI = static_cast<int>(s.size());
        if (mI > 0 && JI) {
            R.ds = -(r_pI.value() + (*JI) * base_dx);
            R.dlam.resize(mI);
            for (int i = 0; i < mI; ++i) {
                const double d = use_shifted ? (s[i] + tau_shift) : s[i];
                R.dlam[i] = sdiv(mu_target - d * lam[i] - lam[i] * R.ds[i], d);
            }
        } else {
            R.ds.resize(0);
            R.dlam.resize(0);
        }

        auto [dzL_base, dzU_base] = detail::dz_bounds_from_dx_vec(
            base_dx, zL, zU, B, bound_shift, use_shifted, mu_target, true);
        R.dzL = dzL_base;
        R.dzU = dzU_base;

        const double tau_pri = get_attr_or<double>(cfg_, "ip_tau_pri", 0.995);
        const double tau_dual = get_attr_or<double>(cfg_, "ip_tau_dual", 0.995);
        std::tie(R.alpha_pri, R.alpha_dual) =
            gondzio_step_lengths_(s, R.ds, lam, R.dlam, zL, R.dzL, zU, R.dzU,
                                  R.dx, B, tau_pri, tau_dual);

        // adaptive correction count
        int max_corr = gondzio_config_.max_corrections;
        static double prev_cent = std::numeric_limits<double>::infinity();
        const double centrality = centrality_measure_(
            s, lam, R.ds, R.dlam, zL, zU, R.dzL, R.dzU, R.dx, B, R.alpha_pri,
            R.alpha_dual, mu_target, use_shifted, tau_shift, bound_shift);
        if (centrality > prev_cent * 0.9 && R.correction_count > 0)
            max_corr = std::max(1, max_corr - 1);
        prev_cent = centrality;

        for (int k = 0; k < max_corr; ++k) {
            const double alpha_max = std::min(R.alpha_pri, R.alpha_dual);
            if (!should_apply_gondzio_(centrality, alpha_max))
                break;

            auto [rhs_corr_x, rhs_corr_s] = gondzio_corr_rhs_(
                s, lam, R.ds, R.dlam, zL, zU, R.dzL, R.dzU, R.dx, B,
                JI ? *JI : spmat(), R.alpha_pri, R.alpha_dual, mu_target,
                centrality, use_shifted, tau_shift, bound_shift, Sg);

            auto corr_res = solve_KKT_(
                W, rhs_corr_x, JE, std::nullopt,
                get_attr_or<std::string>(cfg_, "ip_kkt_method", "hykkt"));
            R.dx += corr_res.dx;
            if (JE && corr_res.dy.size() > 0)
                R.dnu += corr_res.dy;

            if (mI > 0 && JI) {
                R.ds = -(r_pI.value() + (*JI) * R.dx);
                for (int i = 0; i < mI; ++i) {
                    const double d = use_shifted ? (s[i] + tau_shift) : s[i];
                    R.dlam[i] =
                        sdiv(mu_target - d * lam[i] - lam[i] * R.ds[i], d);
                }
            }
            std::tie(R.dzL, R.dzU) = detail::dz_bounds_from_dx_vec(
                R.dx, zL, zU, B, bound_shift, use_shifted, mu_target, true);
            std::tie(R.alpha_pri, R.alpha_dual) =
                gondzio_step_lengths_(s, R.ds, lam, R.dlam, zL, R.dzL, zU,
                                      R.dzU, R.dx, B, tau_pri, tau_dual);

            R.correction_count++;
            R.use_correction = true;
        }
        return R;
    }

    // Updated mehrotra_with_gondzio_corrections_ with SOC
    [[nodiscard]] std::tuple<double, double, double, GondzioStepData>
    mehrotra_with_gondzio_corrections_(
        const spmat &W, const dvec &r_d, const std::optional<spmat> &JE,
        const std::optional<dvec> &r_pE, const std::optional<spmat> &JI,
        const std::optional<dvec> &r_pI, const dvec &s, const dvec &lam,
        const dvec &zL, const dvec &zU, const Bounds &B, bool use_shifted,
        double tau_shift, double bound_shift, double mu, double theta,
        const Sigmas &Sg) {
        auto [alpha_aff, mu_aff, sigma] = mehrotra_affine_predictor_(
            W, r_d, JE, r_pE, JI, r_pI, s, lam, zL, zU, B, use_shifted,
            tau_shift, bound_shift, mu, theta);
        const double mu_min = get_attr_or<double>(cfg_, "ip_mu_min", 1e-12);
        const double mu_target = clamp_min(sigma * mu_aff, mu_min);
        dvec rhs_x = -r_d;
        const int mI = static_cast<int>(s.size());
        const int n = static_cast<int>(zL.size());
        if (mI > 0 && JI && Sg.Sigma_s.size()) {
            dvec rc_s(mI);
            for (int i = 0; i < mI; ++i) {
                const double ds = use_shifted ? (s[i] + tau_shift) : s[i];
                rc_s[i] = mu_target - ds * lam[i];
            }
            dvec temp(mI);
            for (int i = 0; i < mI; ++i) {
                const double li =
                    (std::abs(lam[i]) < consts::EPS_POS)
                        ? ((lam[i] >= 0) ? consts::EPS_POS : -consts::EPS_POS)
                        : lam[i];
                temp[i] = rc_s[i] / li;
            }
            rhs_x.noalias() +=
                (*JI).transpose() * (Sg.Sigma_s.asDiagonal() * temp);
        }
        for (int i = 0; i < n; ++i)
            if (B.hasL[i]) {
                const double denom =
                    clamp_min(use_shifted ? (B.sL[i] + bound_shift) : B.sL[i],
                              consts::EPS_POS);
                rhs_x[i] += (mu_target - denom * zL[i]) / denom;
            }
        for (int i = 0; i < n; ++i)
            if (B.hasU[i]) {
                const double denom =
                    clamp_min(use_shifted ? (B.sU[i] + bound_shift) : B.sU[i],
                              consts::EPS_POS);
                rhs_x[i] -= (mu_target - denom * zU[i]) / denom;
            }
        auto base_res = solve_KKT_(
            W, rhs_x, JE, r_pE,
            get_attr_or<std::string>(cfg_, "ip_kkt_method", "hykkt"));
        GondzioStepData gondzio_result = gondzio_multi_(
            W, r_d, JE, r_pE, JI, r_pI, s, lam, zL, zU, B, mu_target,
            use_shifted, tau_shift, bound_shift, Sg, base_res.dx, base_res.dy);

        // Add SOC if alpha_aff is low
        if (alpha_aff < gondzio_config_.soc_threshold) {
            dvec ds_aff(mI), dlam_aff(mI), dzL_aff(n), dzU_aff(n);
            if (mI > 0 && JI) {
                ds_aff = -(r_pI.value() + (*JI) * gondzio_result.dx);
                for (int i = 0; i < mI; ++i) {
                    const double d = use_shifted ? (s[i] + tau_shift) : s[i];
                    dlam_aff[i] = sdiv(-(d * lam[i]) - lam[i] * ds_aff[i], d);
                }
            }
            for (int i = 0; i < n; ++i) {
                if (B.hasL[i]) {
                    const double d = (use_shifted && bound_shift > 0.0)
                                         ? (B.sL[i] + bound_shift)
                                         : B.sL[i];
                    dzL_aff[i] =
                        sdiv(-(d * zL[i]) - zL[i] * gondzio_result.dx[i], d);
                }
                if (B.hasU[i]) {
                    const double d = (use_shifted && bound_shift > 0.0)
                                         ? (B.sU[i] + bound_shift)
                                         : B.sU[i];
                    dzU_aff[i] =
                        sdiv(-(d * zU[i]) + zU[i] * gondzio_result.dx[i], d);
                }
            }
            GondzioStepData best_soc = gondzio_result;
            double best_alpha =
                std::min(gondzio_result.alpha_pri, gondzio_result.alpha_dual);
            for (int soc_k = 0; soc_k < gondzio_config_.max_soc_steps;
                 ++soc_k) {
                GondzioStepData soc = soc_correction_(
                    W, JE, JI, s, lam, zL, zU, gondzio_result.dx, ds_aff,
                    dlam_aff, dzL_aff, dzU_aff, B, alpha_aff, mu_target,
                    use_shifted, tau_shift, bound_shift, Sg);
                double soc_alpha = std::min(soc.alpha_pri, soc.alpha_dual);
                if (soc_alpha >
                    best_alpha *
                        1.1) { // Accept if improves step length by at least 10%
                    best_soc = soc;
                    best_alpha = soc_alpha;
                    gondzio_result.dx += soc.dx;
                    gondzio_result.dnu += soc.dnu;
                    gondzio_result.ds += soc.ds;
                    gondzio_result.dlam += soc.dlam;
                    gondzio_result.dzL += soc.dzL;
                    gondzio_result.dzU += soc.dzU;
                    gondzio_result.alpha_pri = soc.alpha_pri;
                    gondzio_result.alpha_dual = soc.alpha_dual;
                    gondzio_result.soc_count++;
                } else {
                    break; // Stop if no improvement
                }
            }
        }

        return {alpha_aff, mu_aff, sigma, gondzio_result};
    }

};
