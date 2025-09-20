#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Optimized configuration for the Funnel acceptance logic
struct FunnelConfig {
    // Required parameters (no defaults)
    double funnel_initial_tau;    // > 0
    double funnel_sigma;          // in (0, 1)
    double funnel_delta;          // > 0
    double funnel_beta;           // in (0, 1)
    double funnel_kappa;          // in (0, 1)
    double funnel_min_tau;        // >= 0
    std::size_t funnel_max_history; // >= 0
    double funnel_kappa_initial;  // used at first accept
    
    // Optional parameters with defaults
    double funnel_sigma_rho_f = 0.10;      // mild ratio check when curvature used
    double funnel_theta_curv_scale = 1.0;  // scales the GN proxy term
    double funnel_phi_alpha = 0.10;        // controls tau update curvature coupling
    
    // Validation helper
    void validate() const {
        if (!(funnel_sigma > 0.0 && funnel_sigma < 1.0))
            throw std::invalid_argument("funnel_sigma must be in (0,1)");
        if (!(funnel_beta > 0.0 && funnel_beta < 1.0))
            throw std::invalid_argument("funnel_beta must be in (0,1)");
        if (!(funnel_kappa > 0.0 && funnel_kappa < 1.0))
            throw std::invalid_argument("funnel_kappa must be in (0,1)");
        if (!(funnel_delta > 0.0))
            throw std::invalid_argument("funnel_delta must be > 0");
        if (!(funnel_min_tau >= 0.0))
            throw std::invalid_argument("funnel_min_tau must be >= 0");
        if (!(funnel_initial_tau > 0.0))
            throw std::invalid_argument("funnel_initial_tau must be > 0");
    }
};

class Funnel {
public:
    explicit Funnel(FunnelConfig cfg) : cfg_(std::move(cfg)), tau_(cfg_.funnel_initial_tau), iter_(0) {
        cfg_.validate();
        history_.reserve(cfg_.funnel_max_history + 1); // Pre-allocate to avoid reallocations
    }

    // Default constructor with sensible defaults
    explicit Funnel() : Funnel(FunnelConfig{
        .funnel_initial_tau = 1.0,
        .funnel_sigma = 0.1,
        .funnel_delta = 1e-4,
        .funnel_beta = 0.9,
        .funnel_kappa = 0.9,
        .funnel_min_tau = 1e-8,
        .funnel_max_history = 20,
        .funnel_kappa_initial = 0.1
    }) {}

    // Optimized quadratic prediction computation
    [[nodiscard]] static constexpr std::optional<double>
    quad_pred_df(const std::optional<double> &gTs, const std::optional<double> &sTHs) noexcept {
        if (!gTs || !sTHs) return std::nullopt;
        return std::max(0.0, -(*gTs + 0.5 * (*sTHs)));
    }

    // Optimized ratio helper with better numerical stability
    [[nodiscard]] static constexpr double rho(double actual, double predicted, double eps = 1e-12) noexcept {
        return actual / std::max(predicted, eps);
    }

    // Main acceptance test with optimized control flow
    [[nodiscard]] bool is_acceptable(double current_theta, double current_f, double new_theta, double new_f,
                                   double predicted_df_lin, double predicted_dtheta_lin,
                                   std::optional<double> gTs = std::nullopt,
                                   std::optional<double> sTHs = std::nullopt,
                                   std::optional<double> JTJs_s2 = std::nullopt) const {
        
        // Fast early exits for invalid cases
        constexpr double eps = 1e-10;
        if (new_theta < 0.0 || current_theta < 0.0 || !std::isfinite(new_f) || new_theta > tau_ + eps) {
            return false;
        }

        // Compute actual decreases
        const double df_act = current_f - new_f;
        const double dtheta_act = current_theta - new_theta;

        // Determine prediction model and step type
        const auto pred_df_quad = quad_pred_df(gTs, sTHs);
        const double pred_df_use = pred_df_quad.value_or(predicted_df_lin);
        const bool f_type = (pred_df_use >= cfg_.funnel_delta * (current_theta * current_theta) - eps);

        if (f_type) {
            // f-type step: Armijo condition on objective
            const bool armijo_ok = (df_act >= cfg_.funnel_sigma * pred_df_use);
            
            if (!pred_df_quad) {
                return armijo_ok; // Legacy behavior without curvature
            }
            
            // With curvature: additional ratio check
            return armijo_ok && (rho(df_act, pred_df_use) >= cfg_.funnel_sigma_rho_f);
        } else {
            // h-type step: constraint reduction with optional curvature enhancement
            double pred_dtheta_use = predicted_dtheta_lin;
            if (JTJs_s2 && *JTJs_s2 > 0.0) {
                pred_dtheta_use += cfg_.funnel_theta_curv_scale * 0.5 * std::sqrt(*JTJs_s2);
            }
            
            return (new_theta <= cfg_.funnel_beta * tau_ + eps) && 
                   (dtheta_act >= cfg_.funnel_sigma * pred_dtheta_use);
        }
    }

    // Combined acceptance and update with optimized history management
    bool add_if_acceptable(double current_theta, double current_f, double new_theta, double new_f,
                          double predicted_df_lin, double predicted_dtheta_lin,
                          std::optional<double> gTs = std::nullopt,
                          std::optional<double> sTHs = std::nullopt,
                          std::optional<double> JTJs_s2 = std::nullopt) {
        
        // Initialize tau on first iteration
        if (iter_ == 0) {
            tau_ = std::max(tau_, cfg_.funnel_kappa_initial * std::max(current_theta, 0.0));
        }

        // Check acceptance
        if (!is_acceptable(current_theta, current_f, new_theta, new_f, 
                          predicted_df_lin, predicted_dtheta_lin, gTs, sTHs, JTJs_s2)) {
            return false;
        }

        // Efficient history management
        if (cfg_.funnel_max_history > 0) {
            if (history_.size() >= cfg_.funnel_max_history) {
                // Use move-based rotation instead of erase for better performance
                std::rotate(history_.begin(), history_.begin() + 1, history_.end());
                history_.back() = {new_theta, new_f};
            } else {
                history_.emplace_back(new_theta, new_f);
            }
        }

        // Determine step type for tau update (consistent with acceptance logic)
        const auto pred_df_quad = quad_pred_df(gTs, sTHs);
        const double pred_df_use = pred_df_quad.value_or(predicted_df_lin);
        const bool f_type = (pred_df_use >= cfg_.funnel_delta * (current_theta * current_theta) - 1e-10);

        // Update tau only for h-type steps
        if (!f_type) {
            const double curv = std::max(0.0, JTJs_s2.value_or(0.0));
            const double phi = 1.0 / (1.0 + cfg_.funnel_phi_alpha * std::sqrt(curv));
            const double kappa_eff = std::clamp(cfg_.funnel_kappa * phi, 0.0, 1.0);
            tau_ = std::max(cfg_.funnel_min_tau, 
                           (1.0 - kappa_eff) * new_theta + kappa_eff * tau_);
        }

        ++iter_;
        return true;
    }

    // Reset with efficient cleanup
    void reset() noexcept {
        tau_ = cfg_.funnel_initial_tau;
        history_.clear();
        iter_ = 0;
    }

    // Accessors
    [[nodiscard]] double tau() const noexcept { return tau_; }
    [[nodiscard]] int iter() const noexcept { return iter_; }
    [[nodiscard]] const std::vector<std::pair<double, double>>& history() const noexcept { 
        return history_; 
    }

    // Additional utility methods
    [[nodiscard]] bool empty() const noexcept { return history_.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return history_.size(); }
    
    // Get configuration (useful for debugging/analysis)
    [[nodiscard]] const FunnelConfig& config() const noexcept { return cfg_; }

private:
    FunnelConfig cfg_;
    double tau_;
    int iter_;
    std::vector<std::pair<double, double>> history_; // (theta, f)
};