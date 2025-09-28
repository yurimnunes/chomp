// ip_cpp.cpp — refactored and organized C++23 version
#pragma once
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

#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "definitions.h"
#include "model.h"

#include <chrono>
#include <iomanip>

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

namespace nb = nanobind;

// ---------- Constants ----------
namespace consts {
constexpr double EPS_DIV = 1e-16;
constexpr double EPS_POS = 1e-12;
constexpr double INF = std::numeric_limits<double>::infinity();
} // namespace consts

// ---------- External structs ----------
struct StepResult {
    dvec x, lam, nu;
};
struct KKTResult {
    dvec dx, dy;
    std::shared_ptr<kkt::KKTReusable> reusable;
};

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

static inline double percentile_(const dvec &v, double p01_99) {
    if (v.size() == 0)
        return 0.0;
    std::vector<double> a(v.data(), v.data() + v.size());
    std::sort(a.begin(), a.end());
    p01_99 = clamp(p01_99, 0.0, 1.0);
    const double idx = p01_99 * (a.size() - 1);
    const size_t i = static_cast<size_t>(idx);
    const double frac = idx - i;
    if (i + 1 < a.size())
        return a[i] * (1.0 - frac) + a[i + 1] * frac;
    return a.back();
}

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

struct EvalPack {
    double f{};
    dvec g, cI, cE;
    spmat JI, JE;
    double theta{};
};

// ---------- Adaptive Shift Manager ----------
// ---------- Adaptive Shift Manager (revised) ----------
class AdaptiveShiftManager {
public:
    explicit AdaptiveShiftManager(nb::object cfg) : cfg_(cfg) {}

    void reset() {
        tau_ema_init_ = bshift_ema_init_ = sigmaE_ema_init_ = false;
        tau_ema_ = bshift_ema_ = sigmaE_ema_ = 0.0;
        cE_prev_inf_ = 0.0;
        shift_stall_count_ = 0;
    }
    double compute_slack_shift(const dvec &s, const dvec &cI, const dvec &lam,
                               int it) {
        if (!s.size())
            return 0.0;
        assert(s.size() == cI.size() && lam.size() == cI.size());

        const double base = get_attr_or<double>(cfg_, "ip_shift_tau", 1e-3);
        const double beta = get_attr_or<double>(cfg_, "ip_shift_beta", 0.25);
        const double k_mu =
            get_attr_or<double>(cfg_, "ip_shift_tau_k_mu", 0.50);
        const double k_active =
            get_attr_or<double>(cfg_, "ip_shift_tau_k_active", 0.50);
        const double k_stall =
            get_attr_or<double>(cfg_, "ip_shift_tau_k_stall", 0.25);
        const double active_tol =
            get_attr_or<double>(cfg_, "ip_active_tol", 1e-3);
        const double mu_target =
            get_attr_or<double>(cfg_, "ip_mu_target", 1e-4);

        // Single pass to compute active set stats
        int near_active_ct = 0;
        double s_sum = 0.0, lam_sum = 0.0;
        int sample_count = 0;
        constexpr int max_samples = 1000; // Limit samples for large vectors
        const int step = std::max(1, static_cast<int>(s.size()) / max_samples);

        for (int i = 0; i < cI.size(); i += step) {
            if (cI[i] >= -active_tol) {
                s_sum += std::max(s[i], 1e-12);
                lam_sum += std::max(std::abs(lam[i]), 1e-12);
                ++near_active_ct;
                ++sample_count;
            }
        }

        const double frac_active =
            (cI.size() > 0) ? static_cast<double>(near_active_ct) / cI.size()
                            : 0.0;
        const double s_p10 =
            sample_count > 0
                ? s_sum / sample_count
                : percentile_(s, 0.10); // Fallback to full percentile if needed
        const double lam_med = sample_count > 0
                                   ? lam_sum / sample_count
                                   : percentile_(lam.cwiseAbs(), 0.50);

        // Complementarity-driven target τ
        const double tau_from_comp =
            (lam_med > 0.0) ? std::max(0.0, mu_target / lam_med - s_p10) : 0.0;

        // Stall-aware boost
        const bool stallish = (s_p10 < 1e-6) && (frac_active > 0.25);
        if (stallish)
            ++shift_stall_count_;
        const double stall_boost =
            stallish ? (k_stall * (1.0 + 0.05 * it)) : 0.0;

        // Compose and smooth τ
        double tau_raw = base + k_mu * std::sqrt(std::max(mu_target, 0.0)) +
                         k_active * frac_active * std::max(0.0, tau_from_comp) +
                         stall_boost;
        if (!tau_ema_init_) {
            tau_ema_ = tau_raw;
            tau_ema_init_ = true;
        }
        tau_ema_ = (1.0 - beta) * tau_ema_ + beta * tau_raw;
        const double tau_max =
            get_attr_or<double>(cfg_, "ip_shift_tau_max", 1.0);
        const double tau_min =
            get_attr_or<double>(cfg_, "ip_shift_tau_min", 0.0);
        return clamp(tau_ema_, tau_min, tau_max);
    }

    double compute_bound_shift(const dvec &x, const Bounds &B, int it) {
        bool any = false;
        for (int i = 0; i < x.size(); ++i) {
            if (B.hasL[i] || B.hasU[i]) {
                any = true;
                break;
            }
        }
        if (!any)
            return 0.0;

        const double base = get_attr_or<double>(cfg_, "ip_shift_bounds", 0.0);
        const double beta = get_attr_or<double>(cfg_, "ip_shift_beta", 0.25);
        const double near_tol =
            get_attr_or<double>(cfg_, "ip_bound_near_tol", 1e-6);
        const double mu_target =
            get_attr_or<double>(cfg_, "ip_mu_target", 1e-4);
        const double k_mu =
            get_attr_or<double>(cfg_, "ip_shift_bounds_k_mu", 0.30);
        const double k_frac =
            get_attr_or<double>(cfg_, "ip_shift_bounds_k_frac", 0.75);
        const double k_gap =
            get_attr_or<double>(cfg_, "ip_shift_bounds_k_gap", 0.50);

        // Single pass for gaps and near-active counts
        double gap_sum = 0.0;
        int gap_count = 0, near_active_ct = 0;
        constexpr int max_samples = 1000;
        const int step = std::max(1, static_cast<int>(x.size()) / max_samples);

        for (int i = 0; i < x.size(); i += step) {
            if (B.hasL[i]) {
                const double gL = std::max(x[i] - B.lb[i], 0.0);
                gap_sum += gL;
                ++gap_count;
                if (gL < near_tol)
                    ++near_active_ct;
            }
            if (B.hasU[i]) {
                const double gU = std::max(B.ub[i] - x[i], 0.0);
                gap_sum += gU;
                ++gap_count;
                if (gU < near_tol)
                    ++near_active_ct;
            }
        }

        const double frac_near =
            gap_count > 0 ? static_cast<double>(near_active_ct) / gap_count
                          : 0.0;
        const double g_p10 =
            gap_count > 0 ? gap_sum / gap_count : 1.0; // Approximate percentile

        // μ-scaled nudging
        const double raise =
            k_mu * std::sqrt(std::max(mu_target, 0.0)) + k_frac * frac_near +
            k_gap * std::max(0.0, (1e-3 - std::min(g_p10, 1e-3)));
        double kappa_raw =
            (frac_near > 0.15 ? base * (1.0 + 0.05 * it) : base * 0.9) + raise;

        if (!bshift_ema_init_) {
            bshift_ema_ = kappa_raw;
            bshift_ema_init_ = true;
        }
        bshift_ema_ = (1.0 - beta) * bshift_ema_ + beta * kappa_raw;
        const double kappa_max =
            get_attr_or<double>(cfg_, "ip_shift_bounds_max", 1.0);
        const double kappa_min =
            get_attr_or<double>(cfg_, "ip_shift_bounds_min", 0.0);
        return clamp(bshift_ema_, kappa_min, kappa_max);
    }

    // Adaptive RHS shift for equalities: returns Δr_E (same shape as cE)
    std::optional<dvec> compute_equality_shift(const std::optional<spmat> &JE,
                                               const std::optional<dvec> &cE) {
        if (!cE || cE->size() == 0)
            return std::nullopt;
        if (JE && JE->rows() != cE->size()) {
            // shape mismatch: be conservative and skip
            return std::nullopt;
        }

        const double base = get_attr_or<double>(cfg_, "ip_sigma_E", 0.10);
        const double sigma_min =
            get_attr_or<double>(cfg_, "ip_sigma_E_min", 0.02);
        const double sigma_max =
            get_attr_or<double>(cfg_, "ip_sigma_E_max", 0.50);
        const double beta = get_attr_or<double>(cfg_, "ip_shift_beta", 0.25);

        // Progress ratio
        const double cE_inf = cE->cwiseAbs().maxCoeff();
        const double prog_r = (cE_prev_inf_ > 0.0)
                                  ? clamp(cE_inf / cE_prev_inf_, 0.0, 10.0)
                                  : 1.0;
        cE_prev_inf_ = cE_inf;

        // Conditioning proxy from JE column norms (storage-agnostic)
        auto cs = colnorm_stats_(JE);
        const double cond_proxy = (cs.mx / cs.mn); // ≥ 1
        const double cond_term =
            (cond_proxy - 1.0) / (cond_proxy + 1.0); // ∈ [0,1)

        const double k_prog =
            get_attr_or<double>(cfg_, "ip_sigmaE_k_prog", 0.60);
        const double k_cond =
            get_attr_or<double>(cfg_, "ip_sigmaE_k_cond", 0.80);

        double sigmaE =
            base * (1.0 + k_prog * (prog_r - 1.0)) * (1.0 + k_cond * cond_term);
        sigmaE = clamp(sigmaE, sigma_min, sigma_max);

        if (!sigmaE_ema_init_) {
            sigmaE_ema_ = sigmaE;
            sigmaE_ema_init_ = true;
        }
        sigmaE_ema_ = (1.0 - beta) * sigmaE_ema_ + beta * sigmaE;

        // Δr_E = -σ_E * cE, with absolute & relative caps
        dvec rE = -sigmaE_ema_ * (*cE);
        const double max_shift_abs =
            get_attr_or<double>(cfg_, "ip_max_shift", 1e-2);
        const double max_shift_rel = get_attr_or<double>(
            cfg_, "ip_max_shift_rel", 0.10); // fraction of |cE_i|
        for (int i = 0; i < rE.size(); ++i) {
            const double cap_i =
                std::max(max_shift_abs, max_shift_rel * std::abs((*cE)[i]));
            rE[i] = clamp(rE[i], -cap_i, cap_i);
        }
        return rE;
    }

private:
    nb::object cfg_;

    // EMA states + init flags
    double tau_ema_ = 0.0;
    bool tau_ema_init_ = false;
    double bshift_ema_ = 0.0;
    bool bshift_ema_init_ = false;
    double sigmaE_ema_ = 0.0;
    bool sigmaE_ema_init_ = false;

    double cE_prev_inf_ = 0.0;
    int shift_stall_count_ = 0; // optional heuristic counter

    struct ColNormStats {
        double mean{1.0}, mn{1.0}, mx{1.0};
    };

    // Storage-agnostic column norm stats
    static ColNormStats colnorm_stats_(const std::optional<spmat> &J) {
        ColNormStats s;
        if (!J || J->nonZeros() == 0)
            return s;

        const int ncols = J->cols();
        std::vector<double> col_sq(ncols, 0.0);

        // Iterate over outer dimension; record per-column contributions
        for (int k = 0; k < J->outerSize(); ++k) {
            for (spmat::InnerIterator it(*J, k); it; ++it) {
                const int c =
                    it.col(); // true column index regardless of storage
                const double v = it.value();
                col_sq[c] += v * v;
            }
        }

        double sum = 0.0, mn = std::numeric_limits<double>::infinity(),
               mx = 0.0;
        for (int c = 0; c < ncols; ++c) {
            const double cn = std::sqrt(std::max(col_sq[c], 1e-16));
            sum += cn;
            mn = std::min(mn, cn);
            mx = std::max(mx, cn);
        }
        const double mean = sum / std::max(1, ncols);

        s.mean = std::max(1.0, mean);
        s.mn = std::max(1e-12, mn);
        s.mx = std::max(s.mn, mx);
        return s;
    }
};

#include "ip_soc.h"
// ---------- Mehrotra-Gondzio Solver with Fixes ----------
class MehrotraGondzioSolver {
public:
    enum class BarrierStrategy {
        Monotone,            // Standard Fiacco-McCormick
        Mehrotra,            // Classic Mehrotra
        MehrotraSafeguarded, // Mehrotra with corrector safeguards
        Probing,             // KNITRO-style probing
        QualityFunction      // Minimize quality function
    };

    struct GondzioConfig {
        int max_corrections = 3;
        double gamma_a = 0.1, gamma_b = 10.0;
        double beta_min = 0.1, beta_max = 10.0;
        double tau_min = 0.005;
        bool use_adaptive_gamma = true;
        double progress_threshold = 0.1;
        double soc_threshold = 0.1;
        int max_soc_steps = 3;

        // Additional features
        bool use_iterative_refinement = true;
        double safeguard_threshold = 0.1;
        BarrierStrategy barrier_strategy = BarrierStrategy::Monotone;
        bool adaptive_correction_count = true;
        double conditioning_threshold = 1e12;
    };

    struct StepData {
        dvec dx, dnu, ds, dlam, dzL, dzU;
        double alpha_pri = 1.0, alpha_dual = 1.0;
        double mu_target = 0.0;
        int correction_count = 0;
        bool use_correction = false;
        int soc_count = 0;
        double quality_function_value = 0.0;
        double centrality_measure = 1.0;
    };

    AdvancedSOC advanced_soc_;

    explicit MehrotraGondzioSolver(nb::object cfg, ModelC *model)
        : cfg_(cfg), model_(model) {
        load_config_();
        AdvancedSOC::Config soc_config;
    }

    std::tuple<double, double, double, StepData>
    solve(const spmat &W, const dvec &r_d, const std::optional<spmat> &JE,
          const std::optional<dvec> &r_pE, const std::optional<spmat> &JI,
          const std::optional<dvec> &r_pI, const dvec &s, const dvec &lam,
          const dvec &zL, const dvec &zU, const Bounds &B, bool use_shifted,
          double tau_shift, double bound_shift, double mu, double theta,
          const Sigmas &Sg) {
        IP_TIMER("solve()");

        // Mehrotra affine predictor with corrected bound handling
        auto [alpha_aff, mu_aff, sigma] = compute_affine_predictor_(
            W, r_d, JE, r_pE, JI, r_pI, s, lam, zL, zU, B, use_shifted,
            tau_shift, bound_shift, mu, theta);
        IP_LAP("affine predictor");

        // Barrier parameter selection with multiple strategies
        double mu_target = compute_barrier_parameter_(
            mu, mu_aff, sigma, alpha_aff, theta, s, lam, zL, zU, B);
        IP_LAP("barrier parameter");

        // Centering + corrector step
        auto base_step = compute_centering_step_(
            W, r_d, JE, r_pE, JI, r_pI, s, lam, zL, zU, B, mu_target,
            use_shifted, tau_shift, bound_shift, Sg);
        IP_LAP("centering step");

        // Gondzio corrections with safeguards
        auto final_step = apply_gondzio_corrections_(
            W, r_d, JE, r_pE, JI, r_pI, s, lam, zL, zU, B, mu_target,
            use_shifted, tau_shift, bound_shift, Sg, base_step);
        IP_LAP("Gondzio corrections");

        // // SOC with constraint violation handling
        // if (alpha_aff < config_.soc_threshold) {
        //     final_step = apply_soc_corrections_(
        //         W, JE, JI, s, lam, zL, zU, final_step, B, alpha_aff,
        //         mu_target, use_shifted, tau_shift, bound_shift, Sg);
        // }

        bool soc_applied = advanced_soc_.apply_soc(
            final_step, W, JE, JI, s, lam, zL, zU, B, alpha_aff, mu_target,
            use_shifted, tau_shift, bound_shift, Sg,
            [this](const auto &W, const auto &rhs, const auto &JE,
                   const auto &r_pE,
                   auto tag) { return solve_kkt_(W, rhs, JE, r_pE, tag); });
        IP_LAP("SOC step");

        return {alpha_aff, mu_aff, sigma, final_step};
    }

    void set_kkt_solver(
        std::function<KKTResult(const spmat &, const dvec &,
                                const std::optional<spmat> &,
                                const std::optional<dvec> &, std::string_view)>
            solver) {
        solve_kkt_ = std::move(solver);
    }

    std::string solving_method = "ldl";

private:
    nb::object cfg_;
    ModelC *model_;
    GondzioConfig config_;
    std::function<KKTResult(const spmat &, const dvec &,
                            const std::optional<spmat> &,
                            const std::optional<dvec> &, std::string_view)>
        solve_kkt_;

    // Cache for dx_aff to fix the affine mu calculation
    mutable dvec dx_aff_cache_;

    void load_config_() {
        config_.max_corrections =
            get_attr_or<int>(cfg_, "gondzio_max_corrections", 3);
        config_.gamma_a = get_attr_or<double>(cfg_, "gondzio_gamma_a", 0.1);
        config_.gamma_b = get_attr_or<double>(cfg_, "gondzio_gamma_b", 10.0);
        config_.beta_min = get_attr_or<double>(cfg_, "gondzio_beta_min", 0.1);
        config_.beta_max = get_attr_or<double>(cfg_, "gondzio_beta_max", 10.0);
        config_.tau_min = get_attr_or<double>(cfg_, "gondzio_tau_min", 0.005);
        config_.use_adaptive_gamma =
            get_attr_or<bool>(cfg_, "gondzio_adaptive_gamma", true);
        config_.progress_threshold =
            get_attr_or<double>(cfg_, "gondzio_progress_threshold", 0.1);
        config_.soc_threshold = get_attr_or<double>(cfg_, "soc_threshold", 0.1);
        config_.max_soc_steps = get_attr_or<int>(cfg_, "max_soc_steps", 3);
        config_.safeguard_threshold =
            get_attr_or<double>(cfg_, "mehrotra_safeguard_threshold", 0.1);
        config_.adaptive_correction_count =
            get_attr_or<bool>(cfg_, "adaptive_correction_count", true);

        // Barrier strategy selection
        std::string strategy_str = get_attr_or<std::string>(
            cfg_, "barrier_strategy", "mehrotra_safeguarded");
        if (strategy_str == "monotone")
            config_.barrier_strategy = BarrierStrategy::Monotone;
        else if (strategy_str == "mehrotra")
            config_.barrier_strategy = BarrierStrategy::Mehrotra;
        else if (strategy_str == "probing")
            config_.barrier_strategy = BarrierStrategy::Probing;
        else if (strategy_str == "quality_function")
            config_.barrier_strategy = BarrierStrategy::QualityFunction;
        else
            config_.barrier_strategy = BarrierStrategy::MehrotraSafeguarded;
    }

    // FIXED: Proper affine predictor with correct bound slack calculations
    std::tuple<double, double, double> compute_affine_predictor_(
        const spmat &W, const dvec &r_d, const std::optional<spmat> &JE,
        const std::optional<dvec> &r_pE, const std::optional<spmat> &JI,
        const std::optional<dvec> &r_pI, const dvec &s, const dvec &lam,
        const dvec &zL, const dvec &zU, const Bounds &B, bool use_shifted,
        double tau_shift, double bound_shift, double mu, double theta) {

        const bool haveJE = JE && JE->rows() > 0 && JE->cols() > 0;
        auto res = solve_kkt_(W, -r_d, haveJE ? JE : std::nullopt,
                              (r_pE && r_pE->size() > 0) ? r_pE : std::nullopt,
                              solving_method);

        dvec dx_aff = std::move(res.dx);
        dx_aff_cache_ = dx_aff; // Cache for later use in mu calculation

        const int mI = static_cast<int>(s.size());
        const int n = static_cast<int>(zL.size());

        // Compute affine slack and multiplier directions
        dvec ds_aff, dlam_aff;
        // --- inequalities ---
        if (mI > 0 && JI) {
            ds_aff = -(r_pI.value() + (*JI) * dx_aff);
            dlam_aff = dvec(mI);
            for (int i = 0; i < mI; ++i) {
                const double d = clamp_min(
                    use_shifted ? (s[i] + tau_shift) : s[i], consts::EPS_DIV);
                dlam_aff[i] = -(d * lam[i] + lam[i] * ds_aff[i]) / d;
            }
        } else {
            ds_aff.resize(0);
            dlam_aff.resize(0);
        }

        // --- bounds ---
        dvec dzL_aff = dvec::Zero(n), dzU_aff = dvec::Zero(n);
        for (int i = 0; i < n; ++i) {
            if (B.hasL[i]) {
                const double sL = clamp_min(B.sL[i], consts::EPS_DIV);
                const double dL = use_shifted ? (sL + bound_shift) : sL;
                dzL_aff[i] = -(dL * zL[i] + zL[i] * dx_aff[i]) / dL;
            }
            if (B.hasU[i]) {
                const double sU = clamp_min(B.sU[i], consts::EPS_DIV);
                const double dU = use_shifted ? (sU + bound_shift) : sU;
                dzU_aff[i] = -(dU * zU[i] - zU[i] * dx_aff[i]) / dU;
            }
        }

        // Compute affine step length
        double alpha_aff = compute_step_length_(
            s, ds_aff, lam, dlam_aff, zL, dzL_aff, zU, dzU_aff, dx_aff, B);

        // FIXED: Compute affine duality measure with correct slack updates
        double mu_aff = compute_affine_mu_(s, ds_aff, lam, dlam_aff, zL,
                                           dzL_aff, zU, dzU_aff, B, alpha_aff,
                                           use_shifted, tau_shift, bound_shift);

        // Centering parameter with multiple strategies
        double sigma =
            compute_centering_parameter_(alpha_aff, mu, mu_aff, theta);

        return {alpha_aff, mu_aff, sigma};
    }

    // FIXED: Proper affine mu calculation with correct bound slack evolution
    double compute_affine_mu_(const dvec &s, const dvec &ds, const dvec &lam,
                              const dvec &dlam, const dvec &zL, const dvec &dzL,
                              const dvec &zU, const dvec &dzU, const Bounds &B,
                              double alpha_aff, bool use_shifted,
                              double tau_shift, double bound_shift) const {
        double sum_parts = 0.0;
        int denom_cnt = 0;
        const int mI = static_cast<int>(s.size());

        // Inequality constraints
        if (mI > 0) {
            for (int i = 0; i < mI; ++i) {
                const double s_aff = s[i] + alpha_aff * ds[i];
                const double lam_aff = lam[i] + alpha_aff * dlam[i];
                const double s_eff = use_shifted ? (s_aff + tau_shift) : s_aff;
                sum_parts += s_eff * lam_aff;
                ++denom_cnt;
            }
        }

        // FIXED: Bound constraints with proper slack evolution
        const int n = static_cast<int>(zL.size());
        for (int i = 0; i < n; ++i) {
            if (B.hasL[i]) {
                // sL = x - lb evolves as sL_new = sL + alpha * dx
                const double sL_aff = B.sL[i] + alpha_aff * dx_aff_cache_[i];
                const double zL_aff = zL[i] + alpha_aff * dzL[i];
                const double s_eff =
                    use_shifted ? (sL_aff + bound_shift) : sL_aff;
                sum_parts += std::max(s_eff * zL_aff, 0.0);
                ++denom_cnt;
            }
            if (B.hasU[i]) {
                // sU = ub - x evolves as sU_new = sU - alpha * dx
                const double sU_aff = B.sU[i] - alpha_aff * dx_aff_cache_[i];
                const double zU_aff = zU[i] + alpha_aff * dzU[i];
                const double s_eff =
                    use_shifted ? (sU_aff + bound_shift) : sU_aff;
                sum_parts += std::max(s_eff * zU_aff, 0.0);
                ++denom_cnt;
            }
        }

        return (denom_cnt > 0) ? sum_parts / clamp_min(denom_cnt, 1) : 0.0;
    }

    // Barrier parameter selection with multiple strategies
    double compute_barrier_parameter_(double mu, double mu_aff, double sigma,
                                      double alpha_aff, double theta,
                                      const dvec &s, const dvec &lam,
                                      const dvec &zL, const dvec &zU,
                                      const Bounds &B) {
        const double mu_min = get_attr_or<double>(cfg_, "ip_mu_min", 1e-12);

        switch (config_.barrier_strategy) {
        case BarrierStrategy::Monotone:
            return std::max(mu_min, 0.1 * mu);

        case BarrierStrategy::Mehrotra:
            return clamp_min(sigma * mu_aff, mu_min);

        case BarrierStrategy::MehrotraSafeguarded: {
            double mu_target = clamp_min(sigma * mu_aff, mu_min);
            // Safeguard: prevent too aggressive reduction
            if (mu_target < config_.safeguard_threshold * mu &&
                alpha_aff < 0.9) {
                mu_target =
                    std::max(mu_target, config_.safeguard_threshold * mu);
            }
            return mu_target;
        }

        case BarrierStrategy::Probing: {
            // KNITRO-style probing approach
            double probe_factor = std::min(0.1, std::pow(alpha_aff, 2));
            return clamp_min(probe_factor * mu, mu_min);
        }

        case BarrierStrategy::QualityFunction: {
            // Minimize quality function approach
            double quality =
                compute_quality_function_(mu_aff, alpha_aff, theta);
            double scaling = std::min(1.0, quality);
            return clamp_min(scaling * sigma * mu_aff, mu_min);
        }

        default:
            return clamp_min(sigma * mu_aff, mu_min);
        }
    }

    // Centering parameter computation
    double compute_centering_parameter_(double alpha_aff, double mu,
                                        double mu_aff, double theta) {
        const double pwr = get_attr_or<double>(cfg_, "ip_sigma_power", 3.0);
        const double mu_min = get_attr_or<double>(cfg_, "ip_mu_min", 1e-12);

        if (alpha_aff > 0.95) {
            return 0.0; // Pure Newton step for good affine progress
        }

        // Standard Mehrotra formula
        double sigma = std::pow(1.0 - alpha_aff, 2) *
                       std::pow(mu_aff / clamp_min(mu, mu_min), pwr);

        // Constraint violation adjustment
        const double theta_clip =
            get_attr_or<double>(cfg_, "ip_theta_clip", 1e-2);
        if (theta > theta_clip) {
            sigma =
                std::max(sigma, 0.5); // More centering for infeasible points
        }

        // Problem conditioning adjustment
        if (config_.use_adaptive_gamma) {
            sigma *= (1.0 + std::min(theta / theta_clip, 2.0)) / 3.0;
        }

        return clamp(sigma, 0.0, 1.0);
    }

    // Quality function for barrier parameter selection
    double compute_quality_function_(double mu_aff, double alpha_aff,
                                     double theta) const {
        const double progress_term = 1.0 - alpha_aff;
        const double centrality_term = mu_aff / (mu_aff + 1.0);
        const double feasibility_term = 1.0 / (1.0 + theta);

        return 0.4 * progress_term + 0.4 * centrality_term +
               0.2 * feasibility_term;
    }

    // Step length computation with better numerics
    double compute_step_length_(const dvec &s, const dvec &ds, const dvec &lam,
                                const dvec &dlam, const dvec &zL,
                                const dvec &dzL, const dvec &zU,
                                const dvec &dzU, const dvec &dx,
                                const Bounds &B) const {
        double alpha = 1.0;
        const double eps_div = std::max(consts::EPS_DIV, 1e-16);

        // Slack step length
        for (int i = 0; i < s.size(); ++i) {
            if (ds[i] < -eps_div) {
                alpha = std::min(alpha, -s[i] / ds[i]);
            }
        }

        // Multiplier step length
        for (int i = 0; i < lam.size(); ++i) {
            if (dlam[i] < -eps_div) {
                alpha = std::min(alpha, -lam[i] / dlam[i]);
            }
        }

        // Bound step lengths with proper slack evolution
        const int n = static_cast<int>(dx.size());
        for (int i = 0; i < n; ++i) {
            if (B.hasL[i] && dx[i] < -eps_div) {
                alpha = std::min(alpha, -B.sL[i] / dx[i]);
            }
            if (B.hasU[i] && dx[i] > eps_div) {
                alpha = std::min(alpha, B.sU[i] / dx[i]);
            }
        }

        // Bound dual step lengths
        for (int i = 0; i < std::min(n, (int)zL.size()); ++i) {
            if (B.hasL[i] && dzL[i] < -eps_div) {
                alpha = std::min(alpha, -zL[i] / dzL[i]);
            }
        }
        for (int i = 0; i < std::min(n, (int)zU.size()); ++i) {
            if (B.hasU[i] && dzU[i] < -eps_div) {
                alpha = std::min(alpha, -zU[i] / dzU[i]);
            }
        }

        return clamp01(alpha);
    }

    // Centering step computation
    StepData compute_centering_step_(
        const spmat &W, const dvec &r_d, const std::optional<spmat> &JE,
        const std::optional<dvec> &r_pE, const std::optional<spmat> &JI,
        const std::optional<dvec> &r_pI, const dvec &s, const dvec &lam,
        const dvec &zL, const dvec &zU, const Bounds &B, double mu_target,
        bool use_shifted, double tau_shift, double bound_shift,
        const Sigmas &Sg) {

        dvec rhs_x = -r_d;
        const int mI = static_cast<int>(s.size());
        const int n = static_cast<int>(zL.size());

        // Add centering terms
        if (mI > 0 && JI && Sg.Sigma_s.size()) {
            dvec rc_s(mI);
            for (int i = 0; i < mI; ++i) {
                const double ds = clamp_min(
                    use_shifted ? (s[i] + tau_shift) : s[i], consts::EPS_POS);
                rc_s[i] = mu_target - ds * lam[i];
            }

            dvec temp(mI);
            for (int i = 0; i < mI; ++i) {
                const double li = clamp_min(std::abs(lam[i]), consts::EPS_POS);
                temp[i] = rc_s[i] / (lam[i] >= 0 ? li : -li);
            }
            rhs_x.noalias() +=
                (*JI).transpose() * (Sg.Sigma_s.asDiagonal() * temp);
        }

        // Bound centering terms
        for (int i = 0; i < n; ++i) {
            if (B.hasL[i]) {
                const double denom =
                    clamp_min(use_shifted ? (B.sL[i] + bound_shift) : B.sL[i],
                              consts::EPS_POS);
                rhs_x[i] += (mu_target - denom * zL[i]) / denom;
            }
            if (B.hasU[i]) {
                const double denom =
                    clamp_min(use_shifted ? (B.sU[i] + bound_shift) : B.sU[i],
                              consts::EPS_POS);
                rhs_x[i] -= (mu_target - denom * zU[i]) / denom;
            }
        }

        auto base_res = solve_kkt_(W, rhs_x, JE, r_pE, solving_method);

        StepData result;
        result.dx = std::move(base_res.dx);
        result.dnu = (JE && base_res.dy.size() > 0)
                         ? std::move(base_res.dy)
                         : dvec::Zero(JE ? JE->rows() : 0);
        result.mu_target = mu_target;

        // Compute remaining step components
        if (mI > 0 && JI) {
            result.ds = -(r_pI.value() + (*JI) * result.dx);
            result.dlam.resize(mI);
            for (int i = 0; i < mI; ++i) {
                const double d = clamp_min(
                    use_shifted ? (s[i] + tau_shift) : s[i], consts::EPS_DIV);
                result.dlam[i] =
                    (mu_target - d * lam[i] - lam[i] * result.ds[i]) / d;
            }
        } else {
            result.ds.resize(0);
            result.dlam.resize(0);
        }

        // Compute bound dual steps
        std::tie(result.dzL, result.dzU) = compute_bound_duals_(
            result.dx, zL, zU, B, bound_shift, use_shifted, mu_target, true);

        // Compute step lengths
        const double tau_pri = get_attr_or<double>(cfg_, "ip_tau_pri", 0.995);
        const double tau_dual = get_attr_or<double>(cfg_, "ip_tau_dual", 0.995);
        std::tie(result.alpha_pri, result.alpha_dual) = compute_step_lengths_(
            s, result.ds, lam, result.dlam, zL, result.dzL, zU, result.dzU,
            result.dx, B, tau_pri, tau_dual);

        return result;
    }

    // Bound dual computation
    std::pair<dvec, dvec> compute_bound_duals_(const dvec &dx, const dvec &zL,
                                               const dvec &zU, const Bounds &B,
                                               double bound_shift,
                                               bool use_shifted, double mu,
                                               bool use_mu) const {
        const int n = dx.size();
        dvec dzL = dvec::Zero(n), dzU = dvec::Zero(n);

        for (int i = 0; i < n; ++i) {
            if (B.hasL[i]) {
                const double d =
                    clamp_min(B.sL[i] + (use_shifted ? bound_shift : 0.0),
                              consts::EPS_DIV);
                dzL[i] = use_mu ? (mu - d * zL[i] - zL[i] * dx[i]) / d
                                : -(zL[i] * dx[i]) / d;
            }
            if (B.hasU[i]) {
                const double d =
                    clamp_min(B.sU[i] + (use_shifted ? bound_shift : 0.0),
                              consts::EPS_DIV);
                dzU[i] = use_mu ? (mu - d * zU[i] + zU[i] * dx[i]) / d
                                : (zU[i] * dx[i]) / d;
            }
        }
        return {dzL, dzU};
    }

    // Step length computation with separate primal/dual
    std::pair<double, double>
    compute_step_lengths_(const dvec &s, const dvec &ds, const dvec &lam,
                          const dvec &dlam, const dvec &zL, const dvec &dzL,
                          const dvec &zU, const dvec &dzU, const dvec &dx,
                          const Bounds &B, double tau_pri,
                          double tau_dual) const {

        double a_pri = 1.0, a_dual = 1.0;
        const double eps_div = std::max(consts::EPS_DIV, 1e-16);

        // Primal step length
        for (int i = 0; i < s.size(); ++i) {
            if (ds[i] < -eps_div) {
                a_pri = std::min(a_pri, -s[i] / ds[i]);
            }
        }

        const int n = static_cast<int>(dx.size());
        for (int i = 0; i < n; ++i) {
            if (B.hasL[i] && dx[i] < -eps_div) {
                a_pri = std::min(a_pri, -B.sL[i] / dx[i]);
            }
            if (B.hasU[i] && dx[i] > eps_div) {
                a_pri = std::min(a_pri, B.sU[i] / dx[i]);
            }
        }

        // Dual step length
        for (int i = 0; i < lam.size(); ++i) {
            if (dlam[i] < -eps_div) {
                a_dual = std::min(a_dual, -lam[i] / dlam[i]);
            }
        }

        for (int i = 0; i < std::min(n, (int)zL.size()); ++i) {
            if (B.hasL[i] && dzL[i] < -eps_div) {
                a_dual = std::min(a_dual, -zL[i] / dzL[i]);
            }
        }
        for (int i = 0; i < std::min(n, (int)zU.size()); ++i) {
            if (B.hasU[i] && dzU[i] < -eps_div) {
                a_dual = std::min(a_dual, -zU[i] / dzU[i]);
            }
        }

        return {clamp01(tau_pri * a_pri), clamp01(tau_dual * a_dual)};
    }

    // Gondzio corrections with safeguards
    StepData apply_gondzio_corrections_(
        const spmat &W, const dvec &r_d, const std::optional<spmat> &JE,
        const std::optional<dvec> &r_pE, const std::optional<spmat> &JI,
        const std::optional<dvec> &r_pI, const dvec &s, const dvec &lam,
        const dvec &zL, const dvec &zU, const Bounds &B, double mu_target,
        bool use_shifted, double tau_shift, double bound_shift,
        const Sigmas &Sg, StepData base_step) {

        StepData result = std::move(base_step);
        const double tau_pri = get_attr_or<double>(cfg_, "ip_tau_pri", 0.995);
        const double tau_dual = get_attr_or<double>(cfg_, "ip_tau_dual", 0.995);

        // Adaptive correction count
        int max_corr = config_.max_corrections;
        if (config_.adaptive_correction_count) {
            static double prev_cent = std::numeric_limits<double>::infinity();
            static int stall_count = 0;

            const double centrality = compute_centrality_measure_(
                s, lam, result.ds, result.dlam, zL, zU, result.dzL, result.dzU,
                result.dx, B, result.alpha_pri, result.alpha_dual, mu_target,
                use_shifted, tau_shift, bound_shift);

            result.centrality_measure = centrality;

            // Reduce corrections if not making progress
            if (centrality > prev_cent * 0.95) {
                stall_count++;
                if (stall_count > 2)
                    max_corr = std::max(1, max_corr - 1);
            } else {
                stall_count = 0;
                if (centrality < 0.1)
                    max_corr =
                        std::min(config_.max_corrections + 1, max_corr + 1);
            }
            prev_cent = centrality;
        }

        for (int k = 0; k < max_corr; ++k) {
            const double alpha_max =
                std::min(result.alpha_pri, result.alpha_dual);
            const double centrality_current = result.centrality_measure;

            if (!should_apply_gondzio_(centrality_current, alpha_max, k))
                break;

            auto [rhs_corr_x, rhs_corr_s] = compute_correction_rhs_(
                s, lam, result.ds, result.dlam, zL, zU, result.dzL, result.dzU,
                result.dx, B, JI ? *JI : spmat(), result.alpha_pri,
                result.alpha_dual, mu_target, centrality_current, use_shifted,
                tau_shift, bound_shift, Sg, k);

            auto corr_res =
                solve_kkt_(W, rhs_corr_x, JE, std::nullopt, solving_method);

            // Safeguard: limit correction magnitude
            double corr_norm = corr_res.dx.norm();
            double base_norm = result.dx.norm();
            if (corr_norm > config_.safeguard_threshold * base_norm &&
                base_norm > 0) {
                double scale =
                    config_.safeguard_threshold * base_norm / corr_norm;
                corr_res.dx *= scale;
                if (corr_res.dy.size() > 0)
                    corr_res.dy *= scale;
            }

            // Apply correction
            result.dx += corr_res.dx;
            if (JE && corr_res.dy.size() > 0)
                result.dnu += corr_res.dy;

            // Recompute slack and multiplier steps
            const int mI = static_cast<int>(s.size());
            if (mI > 0 && JI) {
                result.ds = -(r_pI.value() + (*JI) * result.dx);
                for (int i = 0; i < mI; ++i) {
                    const double d =
                        clamp_min(use_shifted ? (s[i] + tau_shift) : s[i],
                                  consts::EPS_DIV);
                    result.dlam[i] =
                        (mu_target - d * lam[i] - lam[i] * result.ds[i]) / d;
                }
            }

            std::tie(result.dzL, result.dzU) =
                compute_bound_duals_(result.dx, zL, zU, B, bound_shift,
                                     use_shifted, mu_target, true);

            std::tie(result.alpha_pri, result.alpha_dual) =
                compute_step_lengths_(s, result.ds, lam, result.dlam, zL,
                                      result.dzL, zU, result.dzU, result.dx, B,
                                      tau_pri, tau_dual);

            result.correction_count++;
            result.use_correction = true;

            // Early termination if correction doesn't help
            double new_alpha = std::min(result.alpha_pri, result.alpha_dual);
            if (new_alpha < alpha_max * 0.95)
                break;
        }

        return result;
    }

    // Centrality measure with better numerics
    double compute_centrality_measure_(
        const dvec &s, const dvec &lam, const dvec &ds, const dvec &dlam,
        const dvec &zL, const dvec &zU, const dvec &dzL, const dvec &dzU,
        const dvec &dx, const Bounds &B, double alpha_pri, double alpha_dual,
        double mu_target, bool use_shifted, double tau_shift,
        double bound_shift) const {

        std::vector<double> ratios;
        ratios.reserve(s.size() + zL.size() + zU.size());

        // Slack complementarity ratios
        for (int i = 0; i < s.size(); ++i) {
            const double s_new =
                std::max(s[i] + alpha_pri * ds[i], consts::EPS_POS);
            const double l_new =
                std::max(lam[i] + alpha_dual * dlam[i], consts::EPS_POS);
            const double prod =
                (use_shifted ? (s_new + tau_shift) : s_new) * l_new;
            ratios.push_back(prod / std::max(mu_target, consts::EPS_POS));
        }

        // FIXED: Bound complementarity ratios with correct slack evolution
        const int n = static_cast<int>(dx.size());
        for (int i = 0; i < std::min(n, (int)zL.size()); ++i) {
            if (B.hasL[i]) {
                // sL = x - lb evolves as sL_new = sL + alpha * dx
                const double sL_new =
                    std::max(B.sL[i] + alpha_pri * dx[i], consts::EPS_POS);
                const double zL_new =
                    std::max(zL[i] + alpha_dual * dzL[i], consts::EPS_POS);
                const double prod =
                    (use_shifted ? (sL_new + bound_shift) : sL_new) * zL_new;
                ratios.push_back(prod / std::max(mu_target, consts::EPS_POS));
            }
        }
        for (int i = 0; i < std::min(n, (int)zU.size()); ++i) {
            if (B.hasU[i]) {
                // sU = ub - x evolves as sU_new = sU - alpha * dx
                const double sU_new =
                    std::max(B.sU[i] - alpha_pri * dx[i], consts::EPS_POS);
                const double zU_new =
                    std::max(zU[i] + alpha_dual * dzU[i], consts::EPS_POS);
                const double prod =
                    (use_shifted ? (sU_new + bound_shift) : sU_new) * zU_new;
                ratios.push_back(prod / std::max(mu_target, consts::EPS_POS));
            }
        }

        if (ratios.empty())
            return 1.0;

        // Compute robust centrality measure
        std::sort(ratios.begin(), ratios.end());
        const double min_ratio = ratios.front();
        const double max_ratio = ratios.back();
        const double median_ratio = ratios[ratios.size() / 2];

        // Use geometric mean of min/max ratio and median deviation
        const double range_measure =
            (min_ratio > 0) ? (max_ratio / min_ratio) : 1e6;
        const double deviation_measure =
            std::max(std::abs(median_ratio - 1.0), 0.1);

        return std::sqrt(range_measure * (1.0 + deviation_measure));
    }

    // Gondzio acceptance criteria
    bool should_apply_gondzio_(double centrality_measure, double alpha_max,
                               int correction_index) const {
        // Basic Gondzio criteria
        bool basic_criteria = (centrality_measure > config_.gamma_b ||
                               centrality_measure < config_.gamma_a) &&
                              alpha_max >= config_.tau_min;

        if (!basic_criteria)
            return false;

        // Additional criteria for later corrections
        if (correction_index > 0) {
            double threshold = config_.gamma_b + correction_index * 0.5;
            if (centrality_measure < threshold &&
                centrality_measure > config_.gamma_a)
                return false;

            if (alpha_max < config_.tau_min * (1.0 + correction_index * 0.5))
                return false;
        }

        return true;
    }
    std::pair<dvec, dvec> compute_correction_rhs_(
        const dvec &s, const dvec &lam, const dvec &ds, const dvec &dlam,
        const dvec &zL, const dvec &zU, const dvec &dzL, const dvec &dzU,
        const dvec &dx, const Bounds &B, const spmat &JI, double alpha_pri,
        double alpha_dual, double mu_target, double centrality_measure,
        bool use_shifted, double tau_shift, double bound_shift,
        const Sigmas &Sg, int correction_index) const {
        const int n = static_cast<int>(dx.size());
        const int mI = static_cast<int>(s.size());

        // Beta computation
        double beta = 1.0;
        if (centrality_measure > config_.gamma_b) {
            beta = clamp(2.0 * centrality_measure / config_.gamma_b,
                         config_.beta_min, config_.beta_max);
        } else if (centrality_measure < config_.gamma_a) {
            beta = clamp(config_.gamma_a / (2.0 * centrality_measure),
                         config_.beta_min, config_.beta_max);
        }
        beta *= std::pow(0.8, correction_index);

        // Initialize result vectors without zero-initialization
        dvec rhs_x(n);
        dvec rhs_s(mI);
        rhs_x.setZero(); // Only if necessary; consider removing if all elements
                         // are assigned
        rhs_s.setZero();

        // Slack corrections (vectorized where possible)
        if (mI > 0) {
            Eigen::VectorXd s_pred = s + alpha_pri * ds;
            Eigen::VectorXd lam_pred = lam + alpha_dual * dlam;
            s_pred = s_pred.cwiseMax(consts::EPS_POS);
            lam_pred = lam_pred.cwiseMax(consts::EPS_POS);
            Eigen::VectorXd s_eff =
                use_shifted ? (s_pred.array() + tau_shift).matrix() : s_pred;
            rhs_s = (-ds.array() * dlam.array() + beta * mu_target -
                     s_eff.array() * lam_pred.array())
                        .matrix();

            if (JI.nonZeros() && Sg.Sigma_s.size() == mI) {
                rhs_x.noalias() +=
                    JI.transpose() * (Sg.Sigma_s.asDiagonal() * rhs_s);
            }
        }

        // Bound corrections
        for (int i = 0; i < n; ++i) {
            double bound_corr = 0.0;
            if (i < zL.size() && B.hasL[i]) {
                const double sL_pred =
                    std::max(B.sL[i] + alpha_pri * dx[i], consts::EPS_POS);
                const double zL_pred =
                    std::max(zL[i] + alpha_dual * dzL[i], consts::EPS_POS);
                const double s_eff =
                    use_shifted ? (sL_pred + bound_shift) : sL_pred;
                const double nonlinear_L = -dx[i] * dzL[i];
                const double centering_L = beta * mu_target - s_eff * zL_pred;
                bound_corr += (nonlinear_L + centering_L) /
                              clamp_min(s_eff, consts::EPS_POS);
            }
            if (i < zU.size() && B.hasU[i]) {
                const double sU_pred =
                    std::max(B.sU[i] - alpha_pri * dx[i], consts::EPS_POS);
                const double zU_pred =
                    std::max(zU[i] + alpha_dual * dzU[i], consts::EPS_POS);
                const double s_eff =
                    use_shifted ? (sU_pred + bound_shift) : sU_pred;
                const double nonlinear_U = dx[i] * dzU[i];
                const double centering_U = beta * mu_target - s_eff * zU_pred;
                bound_corr -= (nonlinear_U + centering_U) /
                              clamp_min(s_eff, consts::EPS_POS);
            }
            rhs_x[i] = bound_corr;
        }

        return {rhs_x, rhs_s};
    }

public:
    double
    average_complementarity(const dvec &s,   // size mI
                            const dvec &lam, // size mI
                            const dvec &zL,  // size n
                            const dvec &zU,  // size n
                            const Bounds &B, // hasL/hasU, sL/sU at current x
                            bool use_shifted, double tau_shift,
                            double bound_shift) {
        double sum = 0.0;
        int cnt = 0;

        // Inequalities: μ_i = (s_i(+τ)) * λ_i
        const int mI = static_cast<int>(s.size());
        for (int i = 0; i < mI; ++i) {
            const double si = std::max(s[i], consts::EPS_POS);
            const double li = std::max(lam[i], consts::EPS_POS);
            const double s_eff = use_shifted ? (si + tau_shift) : si;
            sum += std::max(s_eff * li, 0.0);
            ++cnt;
        }

        // Bounds: sL = x - lb = B.sL, sU = ub - x = B.sU  (already at x_new in
        // B)
        const int n = static_cast<int>(zL.size());
        for (int i = 0; i < n; ++i) {
            if (B.hasL[i]) {
                const double sL = std::max(B.sL[i], consts::EPS_POS);
                const double zLi = std::max(zL[i], consts::EPS_POS);
                const double s_eff = use_shifted ? (sL + bound_shift) : sL;
                sum += std::max(s_eff * zLi, 0.0);
                ++cnt;
            }
            if (B.hasU[i]) {
                const double sU = std::max(B.sU[i], consts::EPS_POS);
                const double zUi = std::max(zU[i], consts::EPS_POS);
                const double s_eff = use_shifted ? (sU + bound_shift) : sU;
                sum += std::max(s_eff * zUi, 0.0);
                ++cnt;
            }
        }

        return (cnt > 0) ? (sum / static_cast<double>(cnt)) : 0.0;
    }
};