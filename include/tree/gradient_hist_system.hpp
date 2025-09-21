// include/hist/gradient_histogram_system.hpp
#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// Adjust include if your path differs
#include "data_binner.hpp"  // DataBinner, EdgeSet, _strict_increasing

namespace foretree {

// =============================== Config ======================================

struct HistogramConfig {
    std::string method = "grad_aware"; // "hist" | "quantile" | "grad_aware" | "adaptive"
    int   max_bins = 256;
    bool  use_missing_bin = true;

    // quantile / grad-aware
    int   coarse_bins = 64;      // base for grad_aware auto-tuning
    bool  density_aware = true;  // reserved for future

    // NEW: Adaptive binning parameters
    int   min_bins = 8;          // minimum bins per feature
    int   target_bins = 32;      // target bins for "normal" features
    bool  adaptive_binning = true; // enable per-feature adaptive bin counts
    double importance_threshold = 0.1; // features above this get more bins
    double complexity_threshold = 0.7; // features above this get more bins
    
    // NEW: Feature importance weighting
    bool  use_feature_importance = false;
    std::vector<double> feature_importance_weights; // if provided, override auto-detection

    // sketch-ish knobs (reserved for future approximations)
    double subsample_ratio = 0.3;
    int    min_sketch_size = 10000;

    // threading
    bool use_parallel = false;
    int  max_workers  = 8;

    // rng (for any sampling we might add)
    uint64_t rng_seed = 42;

    // regularization-esque eps
    double eps = 1e-12;

    int total_bins()     const { return max_bins + (use_missing_bin ? 1 : 0); }
    int missing_bin_id() const { return use_missing_bin ? max_bins : -1; }
};

// =============================== Feature analysis ========================

struct FeatureStats {
    double variance = 0.0;
    double gradient_variance = 0.0;
    double gradient_range = 0.0;
    double value_range = 0.0;
    int unique_count = 0;
    double complexity_score = 0.0;
    double importance_score = 0.0;
    bool is_categorical = false;
    
    // Computed bin allocation
    int suggested_bins = 32;
    std::string allocation_reason = "default";
};

// Analyze feature to determine optimal bin count
inline FeatureStats analyze_feature_importance(const std::vector<double>& values,
                                               const std::vector<double>& gradients,
                                               const std::vector<double>& hessians,
                                               const HistogramConfig& cfg) {
    FeatureStats stats;
    
    if (values.empty()) {
        stats.suggested_bins = cfg.min_bins;
        stats.allocation_reason = "empty_feature";
        return stats;
    }
    
    // Basic statistics
    double val_min = std::numeric_limits<double>::max();
    double val_max = std::numeric_limits<double>::lowest();
    double val_sum = 0.0, val_sq = 0.0;
    double grad_sum = 0.0, grad_sq = 0.0;
    double grad_min = std::numeric_limits<double>::max();
    double grad_max = std::numeric_limits<double>::lowest();
    
    std::vector<double> unique_vals;
    unique_vals.reserve(values.size());
    
    size_t finite_count = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        const double v = values[i];
        const double g = (i < gradients.size()) ? gradients[i] : 0.0;
        
        if (std::isfinite(v) && std::isfinite(g)) {
            finite_count++;
            val_min = std::min(val_min, v);
            val_max = std::max(val_max, v);
            val_sum += v;
            val_sq += v * v;
            
            grad_min = std::min(grad_min, g);
            grad_max = std::max(grad_max, g);
            grad_sum += g;
            grad_sq += g * g;
            
            unique_vals.push_back(v);
        }
    }
    
    if (finite_count == 0) {
        stats.suggested_bins = cfg.min_bins;
        stats.allocation_reason = "no_finite_values";
        return stats;
    }
    
    // Compute statistics
    const double n = static_cast<double>(finite_count);
    const double val_mean = val_sum / n;
    const double grad_mean = grad_sum / n;
    
    stats.variance = (val_sq - val_sum * val_mean) / (n - 1);
    stats.gradient_variance = (grad_sq - grad_sum * grad_mean) / (n - 1);
    stats.value_range = val_max - val_min;
    stats.gradient_range = grad_max - grad_min;
    
    // Count unique values
    std::sort(unique_vals.begin(), unique_vals.end());
    unique_vals.erase(std::unique(unique_vals.begin(), unique_vals.end()), unique_vals.end());
    stats.unique_count = static_cast<int>(unique_vals.size());
    
    // Categorical check
    stats.is_categorical = (stats.unique_count <= std::min(cfg.max_bins / 4, 32));
    
    // Compute complexity score (gradient variation)
    if (finite_count >= 3) {
        std::vector<size_t> order(values.size());
        std::iota(order.begin(), order.end(), 0);
        
        // Sort by values to compute gradient complexity
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return values[a] < values[b];
        });
        
        double complexity_sum = 0.0;
        int complexity_count = 0;
        for (size_t i = 2; i < order.size(); ++i) {
            const double g_curr = gradients[order[i]];
            const double g_prev = gradients[order[i-1]];
            const double g_prev2 = gradients[order[i-2]];
            
            if (std::isfinite(g_curr) && std::isfinite(g_prev) && std::isfinite(g_prev2)) {
                const double second_diff = g_curr - 2.0 * g_prev + g_prev2;
                complexity_sum += std::abs(second_diff);
                complexity_count++;
            }
        }
        
        if (complexity_count > 0) {
            stats.complexity_score = complexity_sum / complexity_count;
        }
    }
    
    // Compute importance score (normalized gradient variance * value range)
    const double norm_grad_var = stats.gradient_variance / (1.0 + std::abs(grad_mean));
    const double norm_val_range = stats.value_range / (1.0 + std::abs(val_mean));
    stats.importance_score = std::sqrt(norm_grad_var * norm_val_range);
    
    // Determine bin allocation
    if (stats.is_categorical) {
        stats.suggested_bins = std::min(stats.unique_count, cfg.max_bins);
        stats.allocation_reason = "categorical";
    } else if (stats.importance_score > cfg.importance_threshold && 
               stats.complexity_score > cfg.complexity_threshold) {
        // High importance + high complexity -> more bins
        const double factor = 1.5 + 0.5 * stats.importance_score + 0.3 * stats.complexity_score;
        stats.suggested_bins = std::min(cfg.max_bins, 
                                       std::max(cfg.min_bins, 
                                               static_cast<int>(cfg.target_bins * factor)));
        stats.allocation_reason = "high_importance_complex";
    } else if (stats.importance_score > cfg.importance_threshold) {
        // High importance -> more bins
        const double factor = 1.2 + 0.4 * stats.importance_score;
        stats.suggested_bins = std::min(cfg.max_bins,
                                       std::max(cfg.min_bins,
                                               static_cast<int>(cfg.target_bins * factor)));
        stats.allocation_reason = "high_importance";
    } else if (stats.complexity_score > cfg.complexity_threshold) {
        // High complexity -> more bins
        const double factor = 1.1 + 0.3 * stats.complexity_score;
        stats.suggested_bins = std::min(cfg.max_bins,
                                       std::max(cfg.min_bins,
                                               static_cast<int>(cfg.target_bins * factor)));
        stats.allocation_reason = "high_complexity";
    } else if (stats.unique_count < cfg.min_bins) {
        // Very few unique values
        stats.suggested_bins = std::max(2, stats.unique_count);
        stats.allocation_reason = "few_unique_values";
    } else {
        // Normal feature
        stats.suggested_bins = cfg.target_bins;
        stats.allocation_reason = "normal";
    }
    
    return stats;
}

// =============================== Feature bins =================================

struct FeatureBins {
    std::vector<double> edges; // strictly increasing, size = nb+1
    bool   is_uniform = false;
    std::string strategy = "uniform";  // "uniform"|"quantile"|"categorical"|"grad_aware"|"adaptive"
    double lo    = 0.0;  // for O(1) uniform binning
    double width = 1.0;  // for O(1) uniform binning
    
    // NEW: Analysis results
    FeatureStats stats;

    int n_bins() const {
        return static_cast<int>(edges.empty() ? 0 : (edges.size() - 1));
    }
};

inline void _check_uniform(FeatureBins& b, double tol = 1e-9) {
    const int nb = b.n_bins();
    if (nb <= 1) {
        b.is_uniform = true;
        b.lo    = b.edges.front();
        b.width = (nb == 1 ? (b.edges[1] - b.edges[0]) : 1.0);
        return;
    }
    double total = 0.0, max_dev = 0.0;
    for (int k = 0; k < nb; ++k) total += (b.edges[k + 1] - b.edges[k]);
    if (total <= 0.0) { b.is_uniform = false; return; }

    const double meanw = total / nb;
    for (int k = 0; k < nb; ++k) {
        const double w = (b.edges[k + 1] - b.edges[k]);
        max_dev = std::max(max_dev, std::abs(w - meanw));
    }
    b.is_uniform = (max_dev <= tol * std::max(1.0, std::abs(meanw)));
    if (b.is_uniform) { b.lo = b.edges.front(); b.width = meanw; }
}

// =============================== Utilities ===================================

inline std::vector<double> _midpoint_edges_of_unique(const std::vector<double>& uniq) {
    // 'uniq' must be sorted and unique
    if (uniq.empty())  return {0.0, 1.0};
    if (uniq.size() == 1) {
        const double x = uniq[0];
        return {x - 1e-12, x + 1e-12};
    }
    std::vector<double> e(uniq.size() + 1);
    e.front() = uniq.front() - 1e-12;
    e.back()  = uniq.back()  + 1e-12;
    for (size_t i = 1; i < uniq.size(); ++i) e[i] = 0.5 * (uniq[i - 1] + uniq[i]);
    _strict_increasing(e);
    return e;
}

inline std::vector<double> exact_quantile_edges(const std::vector<double>& vals, int nb) {
    if (vals.empty()) return {0.0, 1.0};
    nb = std::max(1, nb);
    std::vector<double> s = vals;
    std::sort(s.begin(), s.end());

    std::vector<double> e; e.reserve(nb + 1);
    const size_t n = s.size();
    for (int i = 0; i <= nb; ++i) {
        const double q   = static_cast<double>(i) / nb;
        const double pos = q * (n - 1);
        const size_t lo  = static_cast<size_t>(std::floor(pos));
        const size_t hi  = static_cast<size_t>(std::ceil(pos));
        const double w   = pos - lo;
        const double v   = (1.0 - w) * s[lo] + w * s[hi];
        e.push_back(v);
    }
    _strict_increasing(e);
    return e;
}

inline std::vector<double> weighted_quantile_edges(const std::vector<double>& vals,
                                                   const std::vector<double>& wts,
                                                   int nb) {
    nb = std::max(1, nb);
    struct Pair { double v, w; };
    std::vector<Pair> vw; vw.reserve(vals.size());
    for (size_t i = 0; i < vals.size(); ++i) {
        const double v = vals[i];
        const double w = (i < wts.size() ? wts[i] : 1.0);
        if (std::isfinite(v) && std::isfinite(w) && w > 0.0) vw.push_back({v, w});
    }
    if (vw.empty()) return {0.0, 1.0};
    std::sort(vw.begin(), vw.end(), [](const Pair& a, const Pair& b){ return a.v < b.v; });

    // Compress duplicates: one row per unique value with summed weight
    std::vector<double> uniq; uniq.reserve(vw.size());
    std::vector<double> wuniq; wuniq.reserve(vw.size());
    for (const auto& p : vw) {
        if (uniq.empty() || p.v != uniq.back()) {
            uniq.push_back(p.v);
            wuniq.push_back(p.w);
        } else {
            wuniq.back() += p.w;
        }
    }

    const int U = (int)uniq.size();
    if (U == 1) {
        const double x = uniq[0];
        return {x - 1e-12, x + 1e-12};
    }

    // CDF over uniques
    std::vector<double> cdf(U);
    cdf[0] = wuniq[0];
    for (int i = 1; i < U; ++i) cdf[i] = cdf[i-1] + wuniq[i];
    const double W = cdf.back();

    std::vector<double> e(nb + 1);
    e[0]     = uniq.front() - 1e-12;
    e[nb]    = uniq.back()  + 1e-12;

    // Internal edges at centered targets: (i-0.5)/nb
    for (int i = 1; i < nb; ++i) {
        const double target = ((double)i - 0.5) / (double)nb * W;
        auto it = std::lower_bound(cdf.begin(), cdf.end(), target);
        const int k = (it == cdf.end() ? (U - 1) : int(it - cdf.begin()));
        e[i] = uniq[k];
    }

    _strict_increasing(e);
    return e;
}

inline double gradient_complexity(const std::vector<double>& /*v_sorted*/,
                                  const std::vector<double>& g_sorted) {
    const size_t n = g_sorted.size();
    if (n < 3) return 0.45; // mild default
    double acc = 0.0;
    for (size_t i = 2; i < n; ++i) {
        const double d = g_sorted[i] - 2.0 * g_sorted[i - 1] + g_sorted[i - 2];
        acc += std::abs(d);
    }
    const double m = static_cast<double>(n - 2);
    const double local = (m > 0.0 ? acc / m : 0.0);
    const double smooth = 1.0 / (1.0 + local);
    // Map to a factor around 1.0 -> [0.1, 2.0] with center near 0.45~1.0
    const double comp = 0.20 * (1.0 - smooth);
    return std::min(2.0, std::max(0.1, 0.45 + comp));
}

// Downsample an edge vector to exactly 'new_nb' bins (=> new_nb+1 edges).
// Uses linear interpolation in edge index space; then enforces strict increase.
inline std::vector<double> downsample_edges(const std::vector<double>& edges, int new_nb) {
    const int nb = static_cast<int>(edges.empty() ? 0 : (edges.size() - 1));
    if (nb <= 0 || new_nb <= 0) return {0.0, 1.0};
    if (new_nb >= nb) return edges;

    std::vector<double> out; out.reserve((size_t)new_nb + 1);
    const int E = static_cast<int>(edges.size());
    for (int k = 0; k <= new_nb; ++k) {
        const double pos = (double)k * (double)nb / (double)new_nb; // in [0, nb]
        int    i = static_cast<int>(std::floor(pos));
        double t = pos - i;
        if (i < 0)      { i = 0;     t = 0.0; }
        if (i >= E - 1) { i = E - 2; t = 1.0; }
        const double v = (1.0 - t) * edges[(size_t)i] + t * edges[(size_t)(i + 1)];
        out.push_back(v);
    }
    _strict_increasing(out);
    return out;
}

// ============================ Strategies ======================================

struct IBinningStrategy {
    virtual ~IBinningStrategy() = default;
    virtual FeatureBins create_bins(const std::vector<double>& values,
                                    const std::vector<double>& gradients,
                                    const std::vector<double>& hessians,
                                    const HistogramConfig& cfg) = 0;
};

struct QuantileBinner final : IBinningStrategy {
    FeatureBins create_bins(const std::vector<double>& values,
                            const std::vector<double>& /*gradients*/,
                            const std::vector<double>& hessians,
                            const HistogramConfig& cfg) override {
        // Filter finite values
        std::vector<double> v; v.reserve(values.size());
        std::vector<double> w; w.reserve(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            const double vi = values[i];
            const double wi = (i < hessians.size() ? hessians[i] : 1.0);
            if (std::isfinite(vi) && std::isfinite(wi)) {
                v.push_back(vi);
                w.push_back(std::max(cfg.eps, wi));
            }
        }
        FeatureBins fb; fb.strategy = "quantile";
        if (v.empty()) {
            fb.edges = {0.0, 1.0};
            _check_uniform(fb);
            return fb;
        }

        // If few unique values, treat as categorical-like (midpoint edges)
        std::vector<double> u = v;
        std::sort(u.begin(), u.end());
        u.erase(std::unique(u.begin(), u.end()), u.end());
        if (static_cast<int>(u.size()) <= cfg.max_bins) {
            fb.edges = _midpoint_edges_of_unique(u);
            _check_uniform(fb);
            return fb;
        }

        // Weighted quantiles by Hessians (like XGBoost)
        fb.edges = weighted_quantile_edges(v, w, cfg.max_bins);
        _check_uniform(fb);
        return fb;
    }
};

struct GradientAwareBinner final : IBinningStrategy {
    FeatureBins create_bins(const std::vector<double>& values,
                            const std::vector<double>& gradients,
                            const std::vector<double>& hessians,
                            const HistogramConfig& cfg) override {
        // Clean masks
        std::vector<double> v, g, h;
        v.reserve(values.size());
        g.reserve(values.size());
        h.reserve(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            const double vi = values[i];
            const double gi = (i < gradients.size() ? gradients[i] : 0.0);
            const double hi = (i < hessians.size() ? hessians[i] : 1.0);
            if (std::isfinite(vi) && std::isfinite(gi) && std::isfinite(hi)) {
                v.push_back(vi);
                g.push_back(gi);
                h.push_back(std::max(cfg.eps, hi));
            }
        }
        FeatureBins fb; fb.strategy = "grad_aware";
        if (v.empty()) {
            fb.edges = {0.0, 1.0};
            _check_uniform(fb);
            return fb;
        }

        // Unique check → categorical-ish fallback
        std::vector<double> u = v;
        std::sort(u.begin(), u.end());
        u.erase(std::unique(u.begin(), u.end()), u.end());
        if (static_cast<int>(u.size()) <= std::min(cfg.max_bins, 32)) {
            fb.edges = _midpoint_edges_of_unique(u);
            _check_uniform(fb);
            return fb;
        }

        // Order by v to compute local gradient curvature
        std::vector<size_t> ord(v.size());
        std::iota(ord.begin(), ord.end(), size_t{0});
        std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b){ return v[a] < v[b]; });

        std::vector<double> v_s; v_s.reserve(v.size());
        std::vector<double> g_s; g_s.reserve(v.size());
        std::vector<double> h_s; h_s.reserve(v.size());
        for (size_t k : ord) { v_s.push_back(v[k]); g_s.push_back(g[k]); h_s.push_back(h[k]); }

        // Complexity adjusts bin count around coarse_bins
        const double comp = gradient_complexity(v_s, g_s);
        const int nb = std::max(16, std::min(cfg.max_bins,
                        (int)std::lround((double)cfg.coarse_bins * comp)));

        // Use weighted quantiles by Hessians (can mix |g| if desired)
        std::vector<double> w(v_s.size());
        for (size_t i = 0; i < v_s.size(); ++i) w[i] = std::max(cfg.eps, h_s[i]);
        fb.edges = weighted_quantile_edges(v_s, w, nb);
        _check_uniform(fb);
        return fb;
    }
};

// NEW: Adaptive binner that analyzes feature importance
struct AdaptiveBinner final : IBinningStrategy {
    FeatureBins create_bins(const std::vector<double>& values,
                            const std::vector<double>& gradients,
                            const std::vector<double>& hessians,
                            const HistogramConfig& cfg) override {
        // Clean data
        std::vector<double> v, g, h;
        v.reserve(values.size());
        g.reserve(values.size());
        h.reserve(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            const double vi = values[i];
            const double gi = (i < gradients.size() ? gradients[i] : 0.0);
            const double hi = (i < hessians.size() ? hessians[i] : 1.0);
            if (std::isfinite(vi) && std::isfinite(gi) && std::isfinite(hi)) {
                v.push_back(vi);
                g.push_back(gi);
                h.push_back(std::max(cfg.eps, hi));
            }
        }
        
        FeatureBins fb; 
        fb.strategy = "adaptive";
        
        if (v.empty()) {
            fb.edges = {0.0, 1.0};
            fb.stats.suggested_bins = cfg.min_bins;
            fb.stats.allocation_reason = "empty_feature";
            _check_uniform(fb);
            return fb;
        }

        // Analyze feature to determine optimal bin count
        fb.stats = analyze_feature_importance(v, g, h, cfg);
        const int target_bins = fb.stats.suggested_bins;

        // Handle categorical features
        if (fb.stats.is_categorical) {
            std::vector<double> u = v;
            std::sort(u.begin(), u.end());
            u.erase(std::unique(u.begin(), u.end()), u.end());
            fb.edges = _midpoint_edges_of_unique(u);
            _check_uniform(fb);
            return fb;
        }

        // Use weighted quantiles with adaptive bin count
        std::vector<double> w(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            // Weight by hessian + gradient magnitude for important features
            const double base_weight = h[i];
            const double grad_weight = (fb.stats.importance_score > cfg.importance_threshold) 
                                      ? (1.0 + std::abs(g[i])) : 1.0;
            w[i] = base_weight * grad_weight;
        }
        
        fb.edges = weighted_quantile_edges(v, w, target_bins);
        _check_uniform(fb);
        return fb;
    }
};

// ============================ Histogram system =================================

class GradientHistogramSystem {
public:
    explicit GradientHistogramSystem(HistogramConfig cfg)
        : cfg_(std::move(cfg)), rng_(cfg_.rng_seed), P_(0), N_(0) {}

    // ---- Fit bins per feature using the chosen strategy (quantile/grad_aware/adaptive) ----
    // X: row-major (N x P); g/h length N
    void fit_bins(const double* X, int N, int P, const double* g, const double* h) {
        if (N <= 0 || P <= 0) throw std::invalid_argument("fit_bins: invalid N or P");
        if (!X || !g || !h)   throw std::invalid_argument("fit_bins: null input");
        N_ = N; P_ = P;

        std::unique_ptr<IBinningStrategy> strat;
        if (cfg_.method == "quantile")           strat = std::make_unique<QuantileBinner>();
        else if (cfg_.method == "hist")          strat = std::make_unique<QuantileBinner>(); // baseline
        else if (cfg_.method == "adaptive")      strat = std::make_unique<AdaptiveBinner>();
        else                                     strat = std::make_unique<GradientAwareBinner>();

        feature_bins_.assign(P_, FeatureBins{});

        auto process_feature = [&](int j) {
            // Gather column j and reuse buffers
            std::vector<double> col(N_), gj(N_), hj(N_);
            for (int i = 0; i < N_; ++i) {
                const size_t off = (size_t)i * (size_t)P_ + (size_t)j;
                col[i] = X[off];
                gj[i]  = g[i];
                hj[i]  = h[i];
            }
            FeatureBins fb = strat->create_bins(col, gj, hj, cfg_);

            // For adaptive strategy, use the suggested bin count directly
            int max_bins_for_feature;
            if (cfg_.method == "adaptive" && cfg_.adaptive_binning) {
                max_bins_for_feature = fb.stats.suggested_bins;
            } else {
                max_bins_for_feature = cfg_.max_bins;
            }

            // Enforce capacity by downsampling edges if needed
            const int nb = fb.n_bins();
            if (nb > max_bins_for_feature) {
                fb.edges = downsample_edges(fb.edges, max_bins_for_feature);
                _check_uniform(fb);
            }
            return fb;
        };

        if (!cfg_.use_parallel || P_ == 1) {
            for (int j = 0; j < P_; ++j) feature_bins_[j] = process_feature(j);
        } else {
            const int workers = std::max(1, std::min(cfg_.max_workers, P_));
            (void)workers; // placeholder; std::async decides threads itself
            std::vector<std::future<FeatureBins>> futs; futs.reserve(P_);
            for (int j = 0; j < P_; ++j) {
                futs.emplace_back(std::async(std::launch::async, [&, j]{ return process_feature(j); }));
            }
            for (int j = 0; j < P_; ++j) feature_bins_[j] = futs[j].get();
        }

        // Publish into a DataBinner layout (mode "hist" for compatibility)
        std::vector<std::vector<double>> edges_per_feat(P_);
        for (int j = 0; j < P_; ++j) edges_per_feat[j] = feature_bins_[j].edges;

        EdgeSet es;
        es.edges_per_feat = std::move(edges_per_feat);
        // DataBinner::register_edges will compute per-feature bin counts automatically
        binner_ = std::make_unique<DataBinner>(P_);
        binner_->register_edges("hist", std::move(es));

        codes_.reset();
        miss_id_ = binner_->missing_bin_id("hist");
    }

    // ---- Prebin whole matrix X; caches codes internally. Returns (codes_ptr, missing_id). ----
    std::pair<std::shared_ptr<std::vector<uint16_t>>, int>
    prebin_dataset(const double* X, int N, int P) {
        if (!binner_) throw std::runtime_error("fit_bins must be called before prebin_dataset");
        if (N != N_ || P != P_) throw std::invalid_argument("prebin_dataset: shape mismatch vs fit_bins");
        auto pr = binner_->prebin(X, N, P, "hist", -1);
        codes_  = pr.first;
        miss_id_= pr.second;
        return pr;
    }

    // ---- Prebin ANY matrix X using fitted edges, without touching the internal cache. ----
    std::pair<std::shared_ptr<std::vector<uint16_t>>, int>
    prebin_matrix(const double* X, int N, int P) const {
        if (!binner_) throw std::runtime_error("fit_bins must be called before prebin_matrix");
        if (P != P_)  throw std::invalid_argument("prebin_matrix: P mismatch vs fit_bins");
        return binner_->prebin(X, N, P, "hist", -1);
    }

    // ---- Histogram builders (fast path from cached codes) ---------------------

    // Core accumulator used by both "with counts" and "no counts"
    template <bool WITH_COUNTS, class GFloat, class HFloat>
    void _accumulate_hist(const GFloat* g, const HFloat* h,
                          const int* sample_indices, int n_sub,
                          std::vector<double>& Hg, std::vector<double>& Hh,
                          std::vector<int>* Cptr) const {
        if (!codes_)  throw std::runtime_error("prebin_dataset must be called before build_histograms_fast");
        if (!binner_) throw std::runtime_error("fit_bins must be called before build_histograms_fast");

        const uint16_t* CODES = codes_->data();

        auto accumulate_one = [&](int i){
            const uint16_t* row = CODES + (size_t)i * (size_t)P_;
            const double gi = (double)g[i];
            const double hi = (double)h[i];
            for (int j = 0; j < P_; ++j) {
                const uint16_t b = row[j];
                const int feat_total_bins = binner_->total_bins("hist", j);
                if (b >= (uint16_t)feat_total_bins) continue; // safety per feature
                
                // Calculate offset in the histogram array
                // We need to compute cumulative offset for variable bin sizes
                size_t feat_offset = 0;
                for (int k = 0; k < j; ++k) {
                    feat_offset += (size_t)binner_->total_bins("hist", k);
                }
                const size_t off = feat_offset + b;
                
                Hg[off] += gi;
                Hh[off] += hi;
                if constexpr (WITH_COUNTS) { (*Cptr)[off] += 1; }
            }
        };

        if (!sample_indices || n_sub <= 0) {
            for (int i = 0; i < N_; ++i) accumulate_one(i);
        } else {
            for (int t = 0; t < n_sub; ++t) accumulate_one(sample_indices[t]);
        }
    }

    // With counts (G/H/C) - now handles variable bin sizes
    template <class GFloat = float, class HFloat = float>
    std::tuple<std::vector<double>, std::vector<double>, std::vector<int>>
    build_histograms_fast_with_counts(const GFloat* g, const HFloat* h,
                                      const int* sample_indices = nullptr,
                                      int n_sub = 0) const {
        // Calculate total histogram size for variable bin counts
        size_t total_hist_size = 0;
        for (int j = 0; j < P_; ++j) {
            total_hist_size += (size_t)binner_->total_bins("hist", j);
        }
        
        std::vector<double> Hg(total_hist_size, 0.0);
        std::vector<double> Hh(total_hist_size, 0.0);
        std::vector<int>    C (total_hist_size, 0);
        _accumulate_hist<true>(g, h, sample_indices, n_sub, Hg, Hh, &C);
        return {std::move(Hg), std::move(Hh), std::move(C)};
    }

    // No counts (G/H) - now handles variable bin sizes
    template <class GFloat = float, class HFloat = float>
    std::pair<std::vector<double>, std::vector<double>>
    build_histograms_fast(const GFloat* g, const HFloat* h,
                          const int* sample_indices = nullptr,
                          int n_sub = 0) const {
        // Calculate total histogram size for variable bin counts
        size_t total_hist_size = 0;
        for (int j = 0; j < P_; ++j) {
            total_hist_size += (size_t)binner_->total_bins("hist", j);
        }
        
        std::vector<double> Hg(total_hist_size, 0.0);
        std::vector<double> Hh(total_hist_size, 0.0);
        _accumulate_hist<false>(g, h, sample_indices, n_sub, Hg, Hh, nullptr);
        return {std::move(Hg), std::move(Hh)};
    }

    // NEW: Helper to extract histogram for a specific feature from the packed format
    std::tuple<std::vector<double>, std::vector<double>, std::vector<int>>
    extract_feature_histogram(const std::vector<double>& Hg, 
                             const std::vector<double>& Hh,
                             const std::vector<int>& C,
                             int feature) const {
        if (!binner_ || feature < 0 || feature >= P_) {
            throw std::invalid_argument("Invalid feature index");
        }
        
        // Calculate offset for this feature
        size_t feat_offset = 0;
        for (int k = 0; k < feature; ++k) {
            feat_offset += (size_t)binner_->total_bins("hist", k);
        }
        
        const int feat_bins = binner_->total_bins("hist", feature);
        
        std::vector<double> feat_Hg(feat_bins);
        std::vector<double> feat_Hh(feat_bins);
        std::vector<int> feat_C(feat_bins);
        
        for (int b = 0; b < feat_bins; ++b) {
            const size_t idx = feat_offset + b;
            feat_Hg[b] = (idx < Hg.size()) ? Hg[idx] : 0.0;
            feat_Hh[b] = (idx < Hh.size()) ? Hh[idx] : 0.0;
            feat_C[b] = (idx < C.size()) ? C[idx] : 0;
        }
        
        return {std::move(feat_Hg), std::move(feat_Hh), std::move(feat_C)};
    }

    // NEW: Get feature histogram offsets for manual indexing
    std::vector<size_t> get_feature_offsets() const {
        std::vector<size_t> offsets(P_ + 1, 0);
        for (int j = 0; j < P_; ++j) {
            offsets[j + 1] = offsets[j] + (size_t)binner_->total_bins("hist", j);
        }
        return offsets;
    }

    // ---- Accessors ------------------------------------------------------------
    int P() const { return P_; }
    int N() const { return N_; }
    int missing_bin_id() const { return miss_id_; }
    
    // NEW: Per-feature accessors
    int finite_bins(int feature) const { 
        return binner_ ? binner_->finite_bins("hist", feature) : cfg_.max_bins; 
    }
    int total_bins(int feature) const { 
        return binner_ ? binner_->total_bins("hist", feature) : cfg_.total_bins(); 
    }
    int missing_bin_id(int feature) const {
        return binner_ ? binner_->missing_bin_id("hist", feature) : cfg_.missing_bin_id();
    }
    
    // Legacy accessors (return max across features for compatibility)
    int finite_bins() const { return binner_ ? binner_->finite_bins("hist") : cfg_.max_bins; }
    int total_bins()  const { return binner_ ? binner_->total_bins("hist")  : cfg_.total_bins(); }

    // NEW: Get all bin counts at once
    std::vector<int> all_finite_bins() const {
        if (!binner_) return std::vector<int>(P_, cfg_.max_bins);
        return binner_->finite_bins_per_feat("hist");
    }
    std::vector<int> all_total_bins() const {
        std::vector<int> result(P_);
        for (int j = 0; j < P_; ++j) {
            result[j] = total_bins(j);
        }
        return result;
    }

    // NEW: Feature analysis results
    const FeatureStats& feature_stats(int j) const { 
        return feature_bins_.at(j).stats; 
    }
    
    // NEW: Get summary of bin allocation decisions
    std::vector<std::pair<int, std::string>> get_bin_allocation_summary() const {
        std::vector<std::pair<int, std::string>> summary(P_);
        for (int j = 0; j < P_; ++j) {
            summary[j] = {finite_bins(j), feature_bins_[j].stats.allocation_reason};
        }
        return summary;
    }

    const FeatureBins& feature_bins(int j) const { return feature_bins_.at(j); }
    const DataBinner*  binner() const { return binner_.get(); }
    std::shared_ptr<std::vector<uint16_t>> codes_view() const { return codes_; }

private:
    HistogramConfig              cfg_;
    std::mt19937_64              rng_;

    int P_ = 0, N_ = 0;
    int miss_id_ = -1;

    std::vector<FeatureBins>     feature_bins_;
    std::unique_ptr<DataBinner>  binner_;
    std::shared_ptr<std::vector<uint16_t>> codes_;
};

} // namespace foretree