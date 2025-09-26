// Enhanced Interior Point Stepper with SOTA improvements
// 1. Richardson extrapolation for step refinement
// 2. Adaptive barrier parameter strategies (Fiacco-McCormick + superlinear)
// 3. Aggressive μ reduction near solution
#pragma once
#include "definitions.h"
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SparseCore>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <vector>
#include <numeric>
class RichardsonExtrapolator {
private:
    struct StepHistory { dvec dx; double h; double err; int it; };
    std::vector<StepHistory> hist_;
    int max_hist_ = 3;
    double min_ratio_ = 1.5, max_ratio_ = 4.0; // require roughly geometric spacing
    double max_amplify_ = 50.0;                // cap (ratio^p - 1)^{-1}
    double p_lo_ = 0.5, p_hi_ = 8.0;

    static double safe_log(double v) { return std::log(std::max(v, 1e-300)); }

    static bool strictly_increasing(const std::vector<double>& v) {
        for (size_t i = 1; i < v.size(); ++i) if (!(v[i] > v[i-1])) return false;
        return true;
    }

    // Median of pairwise slopes when hs are roughly geometric
    static std::optional<double> estimate_p_pairwise(const std::vector<double>& h,
                                                     const std::vector<dvec>& f) {
        const int m = (int)h.size();
        if (m < 3) return std::nullopt;
        std::vector<double> pvals; pvals.reserve(m-1);
        for (int i = 1; i < m-1; ++i) {
            const double num = (f[i]   - f[i-1]).norm();
            const double den = (f[i+1] - f[i]).norm();
            if (num <= 0.0 || den <= 0.0) continue;
            const double r = h[i] / h[i-1];
            const double r2 = h[i+1] / h[i];
            if (!(r>1.0 && r2>1.0)) continue;
            // use adjacent with closest ratios to avoid scale drift
            const double ruse = 0.5*(r+r2);
            const double p = std::log(num/den) / safe_log(ruse);
            if (std::isfinite(p)) pvals.push_back(p);
        }
        if (pvals.empty()) return std::nullopt;
        std::nth_element(pvals.begin(), pvals.begin()+pvals.size()/2, pvals.end());
        return pvals[pvals.size()/2];
    }

    static bool ratios_reasonable(const std::vector<double>& h, double rmin, double rmax) {
        for (size_t i = 1; i < h.size(); ++i) {
            double r = h[i]/h[i-1];
            if (!(r >= rmin && r <= rmax)) return false;
        }
        return true;
    }

    struct TableauOut { dvec best, prev; bool have_prev; int order; double ampl; };
    TableauOut build_tableau_guarded(const std::vector<double>& h,
                                     const std::vector<dvec>& f,
                                     double p) const {
        const int m = (int)f.size();
        std::vector<std::vector<dvec>> R(m);
        for (int i = 0; i < m; ++i) { R[i].resize(i+1); R[i][0] = f[i]; }

        double worst_ampl = 1.0;
        for (int i = 1; i < m; ++i) {
            for (int j = 1; j <= i; ++j) {
                double ratio = std::pow(h[i] / h[i-j], p);
                double denom = ratio - 1.0;
                if (!std::isfinite(ratio) || std::abs(denom) < 1e-12) {
                    R[i][j] = R[i][j-1]; // fallback
                    continue;
                }
                double ampl = std::abs(1.0 / denom);
                if (ampl > max_amplify_) {
                    // stop escalating this column; copy previous
                    R[i][j] = R[i][j-1];
                } else {
                    R[i][j] = R[i][j-1] + (R[i][j-1] - R[i-1][j-1]) / denom;
                    worst_ampl = std::max(worst_ampl, ampl);
                }
            }
        }
        TableauOut out;
        out.best      = R[m-1][m-1];
        out.prev      = (m >= 2 ? R[m-1][m-2] : R[m-1][m-1]);
        out.have_prev = (m >= 2);
        out.order     = m;
        out.ampl      = worst_ampl;
        return out;
    }

public:
    struct ExtrapolatedStep {
        dvec   dx_refined;
        double error_estimate;
        bool   converged;
        int    order_achieved;
        double p_used;
        double ampl_factor; // diagnostic
    };

    void clear() { hist_.clear(); }

    // Reset history when direction changes or μ changes (caller can decide)
    void reset() { clear(); }

    // Add a step (direction dx at step size h); ignore incompatible sizes
    void add_step(const dvec& dx, double h, double err=std::numeric_limits<double>::quiet_NaN()) {
        if (dx.size() == 0) return;
        hist_.push_back({dx, h, err, (int)hist_.size()});
        if ((int)hist_.size() > max_hist_) hist_.erase(hist_.begin());
    }

    ExtrapolatedStep extrapolate_step(const dvec& current_dx,
                                      double current_h,
                                      double tol = 1e-8) {
        ExtrapolatedStep out;
        out.dx_refined     = current_dx;
        out.error_estimate = std::numeric_limits<double>::infinity();
        out.converged      = false;
        out.order_achieved = 1;
        out.p_used         = 2.0;
        out.ampl_factor    = 1.0;

        // Gather & sort by increasing h (smallest last)
        std::vector<double> h; std::vector<dvec> f;
        h.reserve(hist_.size()+1); f.reserve(hist_.size()+1);
        for (auto& s : hist_) if (s.dx.size()==current_dx.size()) { h.push_back(s.h); f.push_back(s.dx); }
        h.push_back(current_h); f.push_back(current_dx);

        if (f.size() < 2) return out;
        std::vector<size_t> idx(h.size()); std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b){ return h[a] < h[b]; });
        std::vector<double> hs; std::vector<dvec> fs; hs.reserve(h.size()); fs.reserve(f.size());
        for (auto k : idx) { hs.push_back(h[k]); fs.push_back(f[k]); }

        // Require monotone and roughly geometric spacing to proceed
        if (!strictly_increasing(hs) || !ratios_reasonable(hs, min_ratio_, max_ratio_)) {
            return out; // fall back to unrefined
        }

        // Estimate p: pairwise median first, then clamp
        double p_use = estimate_p_pairwise(hs, fs).value_or(2.0);
        if (!std::isfinite(p_use)) p_use = 2.0;
        p_use = std::clamp(p_use, p_lo_, p_hi_);
        out.p_used = p_use;

        // Build guarded tableau
        auto tab = build_tableau_guarded(hs, fs, p_use);
        out.dx_refined     = tab.best;
        out.order_achieved = tab.order;
        out.ampl_factor    = tab.ampl;

        // Error estimate: prefer scaled form if last ratio is clean
        double err = (tab.have_prev ? (tab.best - tab.prev).norm()
                                    : (tab.best - fs.back()).norm());
        double last_ratio_p = std::pow(hs.back()/hs[hs.size()-2], p_use);
        if (std::isfinite(last_ratio_p) && std::abs(last_ratio_p-1.0) > 1e-6)
            err = err / std::abs(last_ratio_p - 1.0);
        out.error_estimate = err;
        out.converged = (err < tol);
        return out;
    }
};



// --------------------------- Public knobs (IPOPT-style) -----------------------
enum class MuStrategy { Adaptive, Monotone };   // "mu_strategy"
enum class MuOracle  { QualityFunction, LOQO }; // "mu_oracle"

struct AdaptiveBarrierConfig {
    // Strategy selection (IPOPT-compatible surface)
    MuStrategy mu_strategy         = MuStrategy::Adaptive;        // adaptive | monotone
    MuOracle  mu_oracle            = MuOracle::QualityFunction;   // quality-function | loqo

    // μ bounds and targets
    double mu_init                 = 1e-1;   // initial μ (for solver to use externally)
    double mu_min                  = 1e-14;
    double mu_max                  = 1e+2;
    double mu_target               = 0.0;    // optional lower target in early/mid iterations (0=disabled)

    // Centering (σ) bounds and exponent
    double sigma_min               = 0.05;
    double sigma_max               = 0.95;
    double sigma_pow               = 3.0;    // p in σ_base = (μ_aff/μ)^p, classic 2..3

    // Quality-function extras (kept minimal and robust)
    double qf_centrality_weight    = 0.0;    // 0 disables penalty term (placeholder kept off)

    // Monotone strategy controls
    double mu_linear_decrease_factor = 0.2;  // τ in μ := max(μ_min, τ * μ) each iter

    // Subproblem tolerance scaling for monotone mode (caller uses this)
    double barrier_tol_factor      = 0.1;    // suggest inner tol ≈ μ * factor

    // Progress & stabilization heuristics
    double fast_progress_thr       = 0.35;   // relative drop in KKT to call "fast"
    double stall_thr               = 0.05;   // relative drop below this ⇒ stall candidate
    int    stall_k_max             = 3;      // consecutive stalls before damping σ
    double sigma_shrink_on_stall   = 0.5;    // damp σ when stalling
    double sigma_boost_on_fast     = 1.2;    // slightly more aggressive when fast

    // Near-solution safeguard: μ ≤ c * ||KKT||^2
    double near_cap_c              = 1e-2;
    double near_cap_enable_thr     = 1e-6;

    // History
    int    max_hist                = 10;

    // Acceptable termination knobs (outer solver should use these)
    double acceptable_tol             = 1e-6;
    int    acceptable_iter            = 15;
    double acceptable_dual_inf_tol    = 1e+10;
    double acceptable_constr_viol_tol = 1e-2;
    double acceptable_compl_inf_tol   = 1e-2;
    double acceptable_obj_change_tol  = 1e-20;

    // ---------------- Anti-loop / oscillation safeguards ----------------
    int    osc_window              = 4;      // sliding window to detect μ/KKT oscillation
    double osc_rel_mu_tol          = 1e-3;   // |μ_i - μ_{i-1}| / max(μ_i,1) <= tol
    double osc_rel_kkt_tol         = 1e-3;   // |KKT_i - KKT_{i-1}| / max(KKT_i,1) <= tol
    double emergency_tau           = 0.2;    // when oscillating: force μ_new <= τ*μ (monotone fallback)
    double min_shrink_on_stall     = 0.2;    // ensure at least this shrink when stalled (if not oscillating)
    int    freeze_max_consecutive  = 1;      // never freeze more than this many consecutive times
};

// --------------------------- Manager interface --------------------------------
class AdaptiveBarrierManager {
public:
    struct Inputs {
        double mu;          // current μ
        double mu_aff;      // affine μ from predictor: (x_aff^T s_aff)/m

        // Residuals (post-corrector, accepted trial):
        double kkt_norm;        // current full KKT norm (aggregated if desired)
        double kkt_norm_prev;   // previous full KKT norm
        double rp_norm;         // primal residual norm (diagnostics)
        double rd_norm;         // dual residual norm   (diagnostics)
        double comp_mu;         // current complementarity x^T s / m

        // Step info
        double alpha_pri;       // accepted primal step length
        double alpha_dua;       // accepted dual step length (can = alpha_pri)
        bool   step_accepted;   // line-search/filter accepted the step

        // Iteration counter (optional)
        int    iter;
    };

    struct Result {
        double mu_new;          // μ for next iteration
        double sigma_used;      // centering parameter σ used
        std::string strategy;   // e.g., "adaptive:quality-function+cap", "monotone", "anti-osc"
    };

    explicit AdaptiveBarrierManager(AdaptiveBarrierConfig cfg = {})
        : cfg_(cfg) {}

    Result update(const Inputs& in) {
        push_hist(in.mu, in.kkt_norm);

        const double mu  = clip(in.mu,     cfg_.mu_min, cfg_.mu_max);
        const double muA = std::max(in.mu_aff, cfg_.mu_min);
        const double kkt_prev = std::max(in.kkt_norm_prev, 1e-16);
        const double kkt_cur  = std::max(in.kkt_norm,       1e-16);

        // Progress metrics
        const double rel_drop = (kkt_prev - kkt_cur) / kkt_prev; // >0 is good
        const bool stalled = (rel_drop < cfg_.stall_thr) || !in.step_accepted;
        const bool fast    = (rel_drop > cfg_.fast_progress_thr) && in.step_accepted;
        if (stalled) ++stall_count_; else stall_count_ = 0;

        // Base σ selection
        double sigma = 0.0;
        std::string strategy;

        if (cfg_.mu_strategy == MuStrategy::Monotone) {
            sigma = cfg_.mu_linear_decrease_factor; // implicit centering
            strategy = "monotone";
        } else {
            if (cfg_.mu_oracle == MuOracle::QualityFunction) {
                sigma = std::pow(muA / mu, cfg_.sigma_pow);
                if (fast)   sigma *= cfg_.sigma_boost_on_fast;
                if (stall_count_ >= cfg_.stall_k_max) sigma *= cfg_.sigma_shrink_on_stall;
                strategy = "adaptive:quality-function";
            } else {
                const double xi = std::max(in.comp_mu, 1e-16);
                sigma = std::pow(xi / mu, cfg_.sigma_pow);
                strategy = "adaptive:loqo";
            }
            sigma = clip(sigma, cfg_.sigma_min, cfg_.sigma_max);
        }

        double mu_new = clip(sigma * mu, cfg_.mu_min, cfg_.mu_max);

        // Optional μ_target: avoid shrinking below target until near-solution
        if (cfg_.mu_target > 0.0 && kkt_cur > cfg_.near_cap_enable_thr) {
            mu_new = std::max(mu_new, cfg_.mu_target);
        }

        // Near-solution cap: μ ≤ c ||KKT||²
        if (kkt_cur < cfg_.near_cap_enable_thr) {
            const double cap = cfg_.near_cap_c * kkt_cur * kkt_cur;
            if (mu_new > cap) { mu_new = std::max(cfg_.mu_min, cap); strategy += "+cap"; }
        }

        // ---------------- Anti-loop safeguards ----------------
        bool would_freeze = false;
        if (stall_count_ >= 2 * cfg_.stall_k_max) {
            // Instead of always freezing, limit consecutive freezes
            if (consecutive_freezes_ < cfg_.freeze_max_consecutive) {
                mu_new = mu;              // freeze once
                strategy = "frozen_on_stall";
                ++consecutive_freezes_;
                would_freeze = true;
            } else {
                // Next time: emergency monotone shrink
                const double mu_em = clip(cfg_.emergency_tau * mu, cfg_.mu_min, cfg_.mu_max);
                mu_new = std::min(mu_new, mu_em);
                strategy = "anti-osc:emergency_monotone";
                consecutive_freezes_ = 0; // reset after emergency action
            }
        } else {
            consecutive_freezes_ = 0; // reset when not in deep stall
        }

        // Detect oscillation in a short window (μ ~ constant and KKT ~ constant)
        if (!would_freeze && is_oscillating_()) {
            const double mu_em = clip(cfg_.emergency_tau * mu, cfg_.mu_min, cfg_.mu_max);
            mu_new = std::min(mu_new, mu_em);
            strategy += (strategy.empty() ? "anti-osc" : "+anti-osc");
            // also damp σ to align with emergency action
            sigma = std::min(sigma, cfg_.emergency_tau);
        }

        // If stalled but not oscillating, enforce a minimum shrink to avoid no-op μ updates
        if (stalled && mu_new >= 0.98 * mu) { // almost unchanged
            const double mu_min_shrink = clip(cfg_.min_shrink_on_stall * mu, cfg_.mu_min, cfg_.mu_max);
            mu_new = std::min(mu_new, mu_min_shrink);
            strategy += (strategy.empty() ? "stall-min-shrink" : "+stall-min-shrink");
        }

        prev_mu_ = mu;
        prev_kkt_ = kkt_cur;
        last_sigma_used_ = sigma;

        return {mu_new, sigma, strategy.empty() ? "mehrotra" : strategy};
    }

    // Reset internal state (e.g., at the start of a solve)
    void reset() {
        mu_hist_.clear();
        kkt_hist_.clear();
        stall_count_ = 0;
        prev_mu_ = cfg_.mu_init;
        prev_kkt_ = std::numeric_limits<double>::infinity();
        last_sigma_used_ = 1.0;
        consecutive_freezes_ = 0;
    }

    // Diagnostics
    int stall_count() const { return stall_count_; }
    double last_sigma() const { return last_sigma_used_; }
    const std::vector<double>& mu_history() const { return mu_hist_; }
    const std::vector<double>& kkt_history() const { return kkt_hist_; }
    const AdaptiveBarrierConfig& config() const { return cfg_; }
    AdaptiveBarrierConfig& config() { return cfg_; }

private:
    AdaptiveBarrierConfig cfg_;
    int    stall_count_      = 0;
    int    consecutive_freezes_ = 0;
    double prev_mu_          = 1e-2;
    double prev_kkt_         = std::numeric_limits<double>::infinity();
    double last_sigma_used_  = 1.0;
    std::vector<double> mu_hist_, kkt_hist_;

    static double clip(double v, double lo, double hi) {
        return std::max(lo, std::min(v, hi));
    }
    void push_hist(double mu, double kkt) {
        mu_hist_.push_back(mu);
        kkt_hist_.push_back(kkt);
        if ((int)mu_hist_.size() > cfg_.max_hist)  mu_hist_.erase(mu_hist_.begin());
        if ((int)kkt_hist_.size() > cfg_.max_hist) kkt_hist_.erase(kkt_hist_.begin());
    }

    bool is_oscillating_() const {
        const int w = std::min(cfg_.osc_window, (int)mu_hist_.size());
        if (w < 2 || (int)kkt_hist_.size() < w) return false;
        // Check last w deltas are small
        for (int i = (int)mu_hist_.size() - w + 1; i < (int)mu_hist_.size(); ++i) {
            const double mu_i  = mu_hist_[i];
            const double mu_im = mu_hist_[i-1];
            const double k_i   = kkt_hist_[i];
            const double k_im  = kkt_hist_[i-1];
            const double rel_mu = std::abs(mu_i - mu_im) / std::max(1.0, std::max(mu_i, mu_im));
            const double rel_k  = std::abs(k_i  - k_im)  / std::max(1.0, std::max(k_i,  k_im));
            if (rel_mu > cfg_.osc_rel_mu_tol || rel_k > cfg_.osc_rel_kkt_tol)
                return false;
        }
        return true; // persistent tiny changes ⇒ oscillation
    }
};
