#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct FilterConfig {
    double filter_theta_min = 1e-4;
    double filter_gamma_theta = 0.1;
    double filter_gamma_f = 0.1;
    double filter_margin_min = 1e-8;
    double iter_scale_factor = 10.0;
    std::size_t filter_max_size = 20;
    double switch_theta = 1e-4;
    double switch_f = 1e-4;
    double tr_delta0 = 1.0;

    void validate() const {
        if (filter_max_size < 1)
            throw std::invalid_argument("filter_max_size must be positive");
        if (!(filter_theta_min > 0.0))
            throw std::invalid_argument("filter_theta_min must be positive");
        if (!(tr_delta0 > 0.0))
            throw std::invalid_argument("tr_delta0 must be positive");
    }
};

class Filter {
private:
    struct Entry {
        double theta_neg;
        double f;

        // Comparison for heap operations (optimize frequent comparisons)
        bool operator<(const Entry &other) const noexcept {
            return theta_neg < other.theta_neg;
        }
    };

public:
    explicit Filter(const FilterConfig &cfg) : cfg_(cfg) {
        cfg_.validate();
        reset();
    }

    void reset() {
        entries_.clear();
        entries_.reserve(cfg_.filter_max_size +
                         1); // Pre-allocate for efficiency

        // Sentinel entry: (theta_max = 10 * filter_theta_min, f = -inf)
        entries_.emplace_back(Entry{-(cfg_.filter_theta_min * 10.0),
                                    -std::numeric_limits<double>::infinity()});

        iter_ = 0;
        initial_theta_.reset();
        initial_f_.reset();
    }

    [[nodiscard]] bool
    is_acceptable(double theta, double f,
                  std::optional<double> trust_radius = std::nullopt) const {
        // Fast early exits
        if (!std::isfinite(theta) || !std::isfinite(f) || theta < 0.0) {
            return false;
        }

        const auto [gθ, gf] = compute_margins_(trust_radius);

        // Compute scales once
        const double theta_scale =
            initial_theta_.value_or(std::max(theta, 1.0));
        const double f_scale = initial_f_.value_or(std::max(std::abs(f), 1.0));
        const double eps = 1e-8 * std::max(theta_scale, f_scale);

        // Check against all filter entries
        for (const auto &e : entries_) {
            const double t_i = -e.theta_neg;

            // Forbidden region check
            if (theta >= (1.0 - gθ) * t_i - eps &&
                f >= e.f - gf * theta - eps) {
                // Switching rescue mechanism
                const double swθ = cfg_.switch_theta * theta_scale;
                const double swf = cfg_.switch_f * f_scale;

                if (theta < swθ || f < e.f - swf) {
                    continue; // Rescue successful
                }
                return false; // In forbidden region, no rescue
            }
        }

        return true;
    }

    bool add_if_acceptable(double theta, double f,
                           std::optional<double> trust_radius = std::nullopt) {
        // Fast validation
        if (!std::isfinite(theta) || !std::isfinite(f) || theta < 0.0) {
            return false;
        }

        // Initialize scales on first call
        if (!initial_theta_)
            initial_theta_ = std::max(theta, 1e-8);
        if (!initial_f_)
            initial_f_ = std::max(std::abs(f), 1e-8);

        const auto [gθ, gf] = compute_margins_(trust_radius);
        const double eps = 1e-8 * std::max(*initial_theta_, *initial_f_);

        // Check dominance and build new filter
        std::vector<Entry> kept;
        kept.reserve(entries_.size() + 1);

        for (const auto &e : entries_) {
            const double t_i = -e.theta_neg;

            // Check if new point dominates this entry
            const bool dominated_by_new = (theta < (1.0 - gθ) * t_i + eps) ||
                                          (f < e.f - gf * theta + eps);

            if (!dominated_by_new) {
                return false; // New point is dominated, reject
            }

            kept.push_back(e);
        }

        // Add new entry
        kept.emplace_back(Entry{-theta, f});

        // Maintain heap property and size limit
        auto comp = [](const Entry &a, const Entry &b) {
            return a.theta_neg > b.theta_neg;
        };

        std::make_heap(kept.begin(), kept.end(), comp);

        if (kept.size() > cfg_.filter_max_size) {
            std::pop_heap(kept.begin(), kept.end(), comp);
            kept.pop_back();
        }

        entries_ = std::move(kept);
        ++iter_;
        return true;
    }

    // Debugging/inspection interface
    [[nodiscard]] std::vector<std::pair<double, double>> entries() const {
        std::vector<std::pair<double, double>> out;
        out.reserve(entries_.size());

        for (const auto &e : entries_) {
            out.emplace_back(-e.theta_neg, e.f);
        }

        return out;
    }

    // Accessors for inspection
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }
    [[nodiscard]] std::size_t iteration() const noexcept { return iter_; }
    [[nodiscard]] bool empty() const noexcept { return entries_.empty(); }
    [[nodiscard]] const FilterConfig &config() const noexcept { return cfg_; }

private:
    [[nodiscard]] std::pair<double, double>
    compute_margins_(std::optional<double> trust_radius) const {
        // Find maximum theta in filter
        double theta_max = 1.0;
        if (!entries_.empty()) {
            theta_max = 0.0;
            for (const auto &e : entries_) {
                theta_max = std::max(theta_max, -e.theta_neg);
            }
        }

        // Compute decay and iteration scaling
        const double decay =
            std::clamp(theta_max / cfg_.filter_theta_min, 0.1, 1.0);
        const double iter_scale =
            1.0 / (1.0 + static_cast<double>(iter_) / cfg_.iter_scale_factor);

        // Base margins
        double g_theta = std::max(cfg_.filter_margin_min,
                                  cfg_.filter_gamma_theta * decay * iter_scale);
        double g_f = std::max(cfg_.filter_margin_min,
                              cfg_.filter_gamma_f * decay * iter_scale);

        // Trust region scaling
        if (trust_radius) {
            const double scale = std::max(1.0, *trust_radius / cfg_.tr_delta0);
            g_theta *= scale;
            g_f *= scale;
        }

        // Ensure minimum margins
        g_theta = std::max(g_theta, 1e-8);
        g_f = std::max(g_f, 1e-8);

        return {g_theta, g_f};
    }

    FilterConfig cfg_;
    std::vector<Entry> entries_;
    std::size_t iter_ = 0;
    std::optional<double> initial_theta_;
    std::optional<double> initial_f_;
};