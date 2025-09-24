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

// Enhanced barrier parameter adaptation
struct AdaptiveBarrierConfig {
    // Fiacco-McCormick parameters
    double fm_reduction_factor = 0.1;   // Standard F-M reduction
    double fm_aggressive_factor = 0.01; // Aggressive reduction near solution

    // Superlinear reduction parameters
    double superlinear_threshold = 1e-4; // When to switch to superlinear
    double superlinear_exponent = 2.0;   // μ_new = μ^exponent

    // Aggressive reduction near solution
    double kkt_threshold = 1e-6;     // KKT error threshold for aggressive mode
    double aggressive_factor = 1e-3; // μ = aggressive_factor * ||KKT||²

    // Safety bounds
    double mu_min = 1e-12;
    double mu_max = 1e2;
};
class RichardsonExtrapolator {
private:
    struct StepHistory {
        dvec x, dx;     // approximation f(h_i)
        double alpha;   // step size h_i
        double error_estimate;
        int iteration;
    };
    std::vector<StepHistory> history_;
    int max_history_ = 3;

    static double safe_log(double v) {
        return std::log(std::max(v, 1e-300));
    }

    // Try to estimate p using log-log linear regression of ||f(h_i)-L|| vs h_i,
    // where L is a provisional limit estimate.
    static std::optional<double> estimate_p_regression(const std::vector<double>& hs,
                                                       const std::vector<dvec>& fs,
                                                       const dvec& L) {
        const int m = static_cast<int>(hs.size());
        if (m < 3) return std::nullopt;

        std::vector<double> xs, ys;
        xs.reserve(m);
        ys.reserve(m);
        for (int i = 0; i < m; ++i) {
            const double err = (fs[i] - L).norm();
            if (err <= 0.0) continue; // skip degenerate
            xs.push_back(std::log(hs[i]));
            ys.push_back(std::log(err));
        }
        if (xs.size() < 2) return std::nullopt; // need slope

        // Ordinary least squares slope
        double sx=0, sy=0, sxx=0, sxy=0;
        for (size_t i = 0; i < xs.size(); ++i) {
            sx  += xs[i];
            sy  += ys[i];
            sxx += xs[i]*xs[i];
            sxy += xs[i]*ys[i];
        }
        const double n = static_cast<double>(xs.size());
        const double denom = n*sxx - sx*sx;
        if (std::abs(denom) < 1e-14) return std::nullopt;

        const double slope = (n*sxy - sx*sy) / denom; // ~ p
        if (!std::isfinite(slope)) return std::nullopt;
        return slope;
    }

    // Fallback: estimate p from three consecutive levels without assuming geometric spacing:
    // delta0/delta1 ≈ (h0^p - h1^p)/(h1^p - h2^p).
    static std::optional<double> estimate_p_threepoint(double h0, double h1, double h2,
                                                       double delta0, double delta1) {
        if (delta0 <= 0.0 || delta1 <= 0.0) return std::nullopt;
        if (!(h2 < h1 && h1 < h0)) return std::nullopt;

        const double R = delta0 / delta1;
        auto F = [&](double p) {
            const double a0 = std::pow(h0, p), a1 = std::pow(h1, p), a2 = std::pow(h2, p);
            const double num = a0 - a1;
            const double den = a1 - a2;
            if (std::abs(den) < 1e-300) return std::numeric_limits<double>::infinity();
            return num/den - R;
        };

        // Bisection on p ∈ [p_min, p_max]
        double p_min = 1e-6, p_max = 12.0;
        double f_min = F(p_min), f_max = F(p_max);
        // Expand if needed (rudimentary)
        for (int k = 0; k < 5 && f_min*f_max > 0; ++k) { p_max *= 2.0; f_max = F(p_max); }
        if (!std::isfinite(f_min) || !std::isfinite(f_max) || f_min*f_max > 0) return std::nullopt;

        for (int it = 0; it < 80; ++it) {
            double pm = 0.5*(p_min + p_max);
            double fm = F(pm);
            if (!std::isfinite(fm)) { p_min = pm; continue; }
            if (std::abs(fm) < 1e-12) return pm;
            if (f_min * fm < 0) {
                p_max = pm; f_max = fm;
            } else {
                p_min = pm; f_min = fm;
            }
        }
        return 0.5*(p_min + p_max);
    }

    // Build a Richardson/Neville tableau for a given base order p.
    static void build_tableau(const std::vector<double>& hs,
                              const std::vector<dvec>& fs,
                              int& order_achieved,
                              dvec& best, dvec& prev, bool& have_prev,
                              double p) {
        const int m = static_cast<int>(fs.size());
        std::vector<std::vector<dvec>> R(m, std::vector<dvec>());
        for (int i = 0; i < m; ++i) { R[i].resize(i+1); R[i][0] = fs[i]; }

        for (int i = 1; i < m; ++i) {
            for (int j = 1; j <= i; ++j) {
                double ratio = std::pow(hs[i] / hs[i-j], p);
                double denom = ratio - 1.0;
                if (std::abs(denom) < 1e-14) {
                    R[i][j] = R[i][j-1]; // fallback
                } else {
                    R[i][j] = R[i][j-1] + (R[i][j-1] - R[i-1][j-1]) / denom;
                }
            }
        }
        best = R[m-1][m-1];
        if (m >= 2) { prev = R[m-1][m-2]; have_prev = true; } else { have_prev = false; }
        order_achieved = m; // eliminated up to m-1 levels of h^p
    }

public:
    struct ExtrapolatedStep {
        dvec   dx_refined;       // extrapolated limit ~ f(0)
        double error_estimate;   // ||last - prev|| as a-posteriori error
        bool   converged;
        int    order_achieved;   // number of levels used
        double p_used;           // estimated base order
    };

    void add_step(const dvec& x, const dvec& dx, double alpha, double error) {
        history_.push_back({x, dx, alpha, error, static_cast<int>(history_.size())});
        if (history_.size() > max_history_) history_.erase(history_.begin());
    }

    ExtrapolatedStep extrapolate_step(const dvec& current_dx,
                                      double current_alpha,
                                      double tol = 1e-8) {
        ExtrapolatedStep out;
        out.dx_refined      = current_dx;
        out.error_estimate  = std::numeric_limits<double>::infinity();
        out.converged       = false;
        out.order_achieved  = 1;
        out.p_used          = 2.0;

        // Gather compatible history
        std::vector<double> h;
        std::vector<dvec>   f;
        h.reserve(history_.size() + 1);
        f.reserve(history_.size() + 1);
        for (const auto& s : history_) {
            if (s.dx.size() == current_dx.size()) { h.push_back(s.alpha); f.push_back(s.dx); }
        }
        // Append current
        h.push_back(current_alpha);
        f.push_back(current_dx);

        // Need at least two levels to extrapolate
        if (f.size() < 2) return out;

        // Sort by increasing h (largest at front, smallest last)
        std::vector<size_t> idx(h.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b){ return h[a] < h[b]; });

        std::vector<double> hs; hs.reserve(h.size());
        std::vector<dvec>   fs; fs.reserve(f.size());
        for (size_t k : idx) { hs.push_back(h[k]); fs.push_back(f[k]); }

        const int m = static_cast<int>(fs.size());

        // Pass 1: provisional limit with p=2 (safe default)
        int ord_tmp = 1; dvec best_tmp = fs.back(), prev_tmp = fs.back(); bool have_prev_tmp=false;
        build_tableau(hs, fs, ord_tmp, best_tmp, prev_tmp, have_prev_tmp, /*p=*/2.0);

        // Estimate p via regression if possible
        std::optional<double> p_hat = estimate_p_regression(hs, fs, best_tmp);

        // Fallback: use last three to solve ratio equation if regression fails
        if (!p_hat && m >= 3) {
            const int i0 = m-3, i1 = m-2, i2 = m-1; // three smallest h
            const double d0 = (fs[i1] - fs[i0]).norm();
            const double d1 = (fs[i2] - fs[i1]).norm();
            p_hat = estimate_p_threepoint(hs[i0], hs[i1], hs[i2], d0, d1);
        }

        // Final clamp / default
        double p_use = p_hat.value_or(2.0);
        if (!std::isfinite(p_use)) p_use = 2.0;
        p_use = std::clamp(p_use, 0.5, 12.0);
        out.p_used = p_use;

        // Pass 2: final tableau with p_use
        int order_final = 1; dvec best = fs.back(), prev = fs.back(); bool have_prev=false;
        build_tableau(hs, fs, order_final, best, prev, have_prev, p_use);

        out.dx_refined      = best;
        out.order_achieved  = order_final;
        out.error_estimate  = have_prev ? (best - prev).norm() : (best - fs.back()).norm();
        out.converged       = (out.error_estimate < tol);
        return out;
    }

    void clear() { history_.clear(); }
};

// Enhanced adaptive barrier parameter manager
class AdaptiveBarrierManager {
private:
    AdaptiveBarrierConfig config_;
    double prev_mu_ = 1e-2;
    double prev_kkt_error_ = std::numeric_limits<double>::infinity();
    int stagnation_count_ = 0;
    bool near_solution_ = false;

    // History for trend analysis
    std::vector<double> mu_history_;
    std::vector<double> kkt_history_;
    int max_history_ = 10;

public:
    AdaptiveBarrierManager(const AdaptiveBarrierConfig &config = {})
        : config_(config) {}

    struct BarrierUpdate {
        double mu_new;
        std::string strategy_used;
        bool is_aggressive;
        double reduction_factor;
    };

    BarrierUpdate update_barrier_parameter(double current_mu, double kkt_error,
                                           double complementarity,
                                           double feasibility_error,
                                           double alpha_step,
                                           bool step_accepted, int iteration) {

        // Update history
        mu_history_.push_back(current_mu);
        kkt_history_.push_back(kkt_error);
        if (mu_history_.size() > max_history_) {
            mu_history_.erase(mu_history_.begin());
            kkt_history_.erase(kkt_history_.begin());
        }

        BarrierUpdate result;
        result.mu_new = current_mu;
        result.strategy_used = "fixed";
        result.is_aggressive = false;
        result.reduction_factor = 1.0;

        // Detect if we're near the solution
        near_solution_ = (kkt_error < config_.kkt_threshold) &&
                         (feasibility_error < config_.kkt_threshold) &&
                         (complementarity < 10 * current_mu);

        // Strategy 1: Aggressive reduction when near solution
        if (near_solution_ && kkt_error > 0) {
            double mu_aggressive =
                config_.aggressive_factor * kkt_error * kkt_error;
            mu_aggressive = std::max(mu_aggressive, config_.mu_min);

            if (mu_aggressive < current_mu) {
                result.mu_new = mu_aggressive;
                result.strategy_used = "aggressive_kkt_squared";
                result.is_aggressive = true;
                result.reduction_factor = mu_aggressive / current_mu;
                return result;
            }
        }

        // Strategy 2: Superlinear reduction for good progress
        if (kkt_error < config_.superlinear_threshold && step_accepted) {
            // Check if we're making consistent progress
            bool consistent_progress = true;
            if (kkt_history_.size() >= 3) {
                for (size_t i = kkt_history_.size() - 2;
                     i < kkt_history_.size(); ++i) {
                    if (kkt_history_[i] >= kkt_history_[i - 1]) {
                        consistent_progress = false;
                        break;
                    }
                }
            }

            if (consistent_progress) {
                double mu_superlinear =
                    std::pow(current_mu, config_.superlinear_exponent);
                mu_superlinear = std::max(mu_superlinear, config_.mu_min);

                result.mu_new = mu_superlinear;
                result.strategy_used = "superlinear";
                result.reduction_factor = mu_superlinear / current_mu;
                return result;
            }
        }

        // Strategy 3: Adaptive Fiacco-McCormick
        double reduction_factor = config_.fm_reduction_factor;

        // Adjust reduction based on step quality
        if (alpha_step > 0.9 && step_accepted) {
            // Good step - be more aggressive
            reduction_factor = config_.fm_aggressive_factor;
        } else if (alpha_step < 0.1 || !step_accepted) {
            // Poor step - be conservative
            reduction_factor = std::sqrt(config_.fm_reduction_factor);
        }

        // Adjust based on convergence rate
        if (kkt_history_.size() >= 2) {
            double improvement_rate = (prev_kkt_error_ - kkt_error) /
                                      std::max(prev_kkt_error_, 1e-12);

            if (improvement_rate > 0.5) {
                // Fast convergence - be more aggressive
                reduction_factor *= 0.5;
            } else if (improvement_rate < 0.1) {
                // Slow convergence - be more conservative
                reduction_factor *= 2.0;
                stagnation_count_++;
            } else {
                stagnation_count_ = 0;
            }
        }

        // Handle stagnation
        if (stagnation_count_ > 3) {
            reduction_factor = std::min(reduction_factor * 0.1, 0.01);
            stagnation_count_ = 0;
        }

        double mu_new = current_mu * reduction_factor;
        mu_new = std::clamp(mu_new, config_.mu_min, config_.mu_max);

        result.mu_new = mu_new;
        result.strategy_used = "adaptive_fiacco_mccormick";
        result.reduction_factor = reduction_factor;

        // Update state
        prev_mu_ = current_mu;
        prev_kkt_error_ = kkt_error;

        return result;
    }

    // Predict optimal μ based on current trajectory
    double predict_optimal_mu(double current_kkt_error) const {
        if (near_solution_) {
            return config_.aggressive_factor * current_kkt_error *
                   current_kkt_error;
        }

        // Use trend analysis to predict
        if (kkt_history_.size() >= 3) {
            double avg_improvement = 0.0;
            for (size_t i = 1; i < kkt_history_.size(); ++i) {
                avg_improvement += kkt_history_[i - 1] - kkt_history_[i];
            }
            avg_improvement /= (kkt_history_.size() - 1);

            if (avg_improvement > 0) {
                // Predict when KKT error will reach threshold
                double steps_to_threshold =
                    (current_kkt_error - config_.kkt_threshold) /
                    avg_improvement;

                if (steps_to_threshold < 5) {
                    // Switch to aggressive mode soon
                    return config_.aggressive_factor *
                           (current_kkt_error -
                            steps_to_threshold * avg_improvement) *
                           (current_kkt_error -
                            steps_to_threshold * avg_improvement);
                }
            }
        }

        return prev_mu_ * config_.fm_reduction_factor;
    }

    bool is_near_solution() const { return near_solution_; }

    void reset() {
        mu_history_.clear();
        kkt_history_.clear();
        stagnation_count_ = 0;
        near_solution_ = false;
        prev_kkt_error_ = std::numeric_limits<double>::infinity();
    }
};
