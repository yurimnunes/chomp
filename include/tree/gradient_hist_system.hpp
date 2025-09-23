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

#ifdef __AVX2__
#include <immintrin.h>
#endif

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

    // Adaptive binning parameters
    int   min_bins = 8;          // minimum bins per feature
    int   target_bins = 32;      // target bins for "normal" features
    bool  adaptive_binning = true; // enable per-feature adaptive bin counts
    double importance_threshold = 0.1; // features above this get more bins
    double complexity_threshold = 0.7; // features above this get more bins
    
    // Feature importance weighting
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

// =============================== Layout for variable bins =================

class VariableBinLayout {
private:
    std::vector<size_t> feature_offsets_;
    std::vector<uint16_t> bins_per_feature_;
    size_t total_histogram_size_ = 0;

public:
    void initialize(const std::vector<int>& bins_per_feat) {
        const size_t P = bins_per_feat.size();
        feature_offsets_.resize(P + 1);
        bins_per_feature_.resize(P);
        
        feature_offsets_[0] = 0;
        for (size_t i = 0; i < P; ++i) {
            bins_per_feature_[i] = static_cast<uint16_t>(bins_per_feat[i]);
            feature_offsets_[i + 1] = feature_offsets_[i] + bins_per_feat[i];
        }
        total_histogram_size_ = feature_offsets_[P];
    }
    
    // O(1) offset calculation
    inline size_t get_offset(int feature, int bin) const {
        return feature_offsets_[feature] + bin;
    }
    
    inline size_t feature_offset(int feature) const {
        return feature_offsets_[feature];
    }
    
    inline uint16_t bins_for_feature(int feature) const {
        return bins_per_feature_[feature];
    }
    
    inline size_t total_size() const {
        return total_histogram_size_;
    }
    
    inline int num_features() const {
        return static_cast<int>(bins_per_feature_.size());
    }
};

// =============================== SIMD optimized accumulation ==============

class HistogramAccumulator {
private:
    const VariableBinLayout* layout_;
    std::vector<double> hist_g_, hist_h_;
    std::vector<int> hist_c_;

public:
    explicit HistogramAccumulator(const VariableBinLayout* layout) 
        : layout_(layout) {
        if (layout_) {
            const size_t size = layout_->total_size();
            hist_g_.resize(size, 0.0);
            hist_h_.resize(size, 0.0);
            hist_c_.resize(size, 0);
        }
    }
    
    void clear() {
        std::fill(hist_g_.begin(), hist_g_.end(), 0.0);
        std::fill(hist_h_.begin(), hist_h_.end(), 0.0);
        std::fill(hist_c_.begin(), hist_c_.end(), 0);
    }
    
    // Core accumulation - template for flexibility
    template<bool WITH_COUNTS, typename GFloat, typename HFloat>
    void accumulate_samples(const uint16_t* codes, const GFloat* g, const HFloat* h,
                           const int* sample_indices, int n_samples, int P) {
        if (!layout_ || !codes || hist_g_.size() != layout_->total_size()) return;
        
        auto accumulate_one = [&](int i) {
            const uint16_t* row = codes + static_cast<size_t>(i) * P;
            const double gi = static_cast<double>(g[i]);
            const double hi = static_cast<double>(h[i]);
            
            for (int j = 0; j < P; ++j) {
                const uint16_t bin = row[j];
                const uint16_t max_bins = layout_->bins_for_feature(j);
                if (bin >= max_bins) continue; // bounds check
                
                const size_t offset = layout_->get_offset(j, bin);
                if (offset >= hist_g_.size()) continue; // safety check
                
                hist_g_[offset] += gi;
                hist_h_[offset] += hi;
                if constexpr (WITH_COUNTS) {
                    hist_c_[offset] += 1;
                }
            }
        };
        
        if (!sample_indices) {
            // Process all samples - use vectorized version if available
            #ifdef __AVX2__
            accumulate_vectorized<WITH_COUNTS>(codes, g, h, n_samples, P);
            #else
            for (int i = 0; i < n_samples; ++i) {
                accumulate_one(i);
            }
            #endif
        } else {
            // Process subset
            for (int t = 0; t < n_samples; ++t) {
                const int sample_idx = sample_indices[t];
                if (sample_idx >= 0) { // bounds check for sample indices
                    accumulate_one(sample_idx);
                }
            }
        }
    }
    
#ifdef __AVX2__
    template<bool WITH_COUNTS, typename GFloat, typename HFloat>
    void accumulate_vectorized(const uint16_t* codes, const GFloat* g, const HFloat* h,
                              int n_samples, int P) {
        // Vectorized version processes multiple samples, but still needs scalar
        // bin updates due to sparse/irregular access patterns
        const int vec_width = 8; // AVX2 processes 8 floats
        const int vec_end = n_samples - (n_samples % vec_width);
        
        // Process vectorizable portion
        for (int i = 0; i < vec_end; i += vec_width) {
            // Load gradients and hessians
            __m256 g_vec, h_vec;
            if constexpr (std::is_same_v<GFloat, float>) {
                g_vec = _mm256_loadu_ps(&g[i]);
            } else {
                // Convert double to float for SIMD
                alignas(32) float g_temp[8];
                for (int k = 0; k < 8; ++k) g_temp[k] = static_cast<float>(g[i + k]);
                g_vec = _mm256_load_ps(g_temp);
            }
            
            if constexpr (std::is_same_v<HFloat, float>) {
                h_vec = _mm256_loadu_ps(&h[i]);
            } else {
                alignas(32) float h_temp[8];
                for (int k = 0; k < 8; ++k) h_temp[k] = static_cast<float>(h[i + k]);
                h_vec = _mm256_load_ps(h_temp);
            }
            
            // Extract and accumulate (still need scalar for irregular bin access)
            alignas(32) float g_array[8], h_array[8];
            _mm256_store_ps(g_array, g_vec);
            _mm256_store_ps(h_array, h_vec);
            
            for (int k = 0; k < 8; ++k) {
                const int idx = i + k;
                const uint16_t* row = codes + static_cast<size_t>(idx) * P;
                const double gi = static_cast<double>(g_array[k]);
                const double hi = static_cast<double>(h_array[k]);
                
                for (int j = 0; j < P; ++j) {
                    const uint16_t bin = row[j];
                    const uint16_t max_bins = layout_->bins_for_feature(j);
                    if (bin >= max_bins) continue;
                    
                    const size_t offset = layout_->get_offset(j, bin);
                    if (offset >= hist_g_.size()) continue; // safety check
                    
                    hist_g_[offset] += gi;
                    hist_h_[offset] += hi;
                    if constexpr (WITH_COUNTS) {
                        hist_c_[offset] += 1;
                    }
                }
            }
        }
        
        // Handle remaining samples
        for (int i = vec_end; i < n_samples; ++i) {
            const uint16_t* row = codes + static_cast<size_t>(i) * P;
            const double gi = static_cast<double>(g[i]);
            const double hi = static_cast<double>(h[i]);
            
            for (int j = 0; j < P; ++j) {
                const uint16_t bin = row[j];
                const uint16_t max_bins = layout_->bins_for_feature(j);
                if (bin >= max_bins) continue;
                
                const size_t offset = layout_->get_offset(j, bin);
                if (offset >= hist_g_.size()) continue; // safety check
                
                hist_g_[offset] += gi;
                hist_h_[offset] += hi;
                if constexpr (WITH_COUNTS) {
                    hist_c_[offset] += 1;
                }
            }
        }
    }
#endif
    
    // Getters
    const std::vector<double>& gradients() const { return hist_g_; }
    const std::vector<double>& hessians() const { return hist_h_; }
    const std::vector<int>& counts() const { return hist_c_; }
    
    // Move results out
    std::tuple<std::vector<double>, std::vector<double>, std::vector<int>> take_results() {
        return {std::move(hist_g_), std::move(hist_h_), std::move(hist_c_)};
    }
    
    std::pair<std::vector<double>, std::vector<double>> take_gh_results() {
        return {std::move(hist_g_), std::move(hist_h_)};
    }
};

// =============================== Streaming quantiles ======================

class StreamingQuantileBuilder {
private:
    struct Bucket {
        double value;
        double weight;
        int count;
        
        Bucket(double v, double w, int c) : value(v), weight(w), count(c) {}
    };
    
    std::vector<Bucket> buckets_;
    double total_weight_ = 0.0;
    int max_buckets_;
    
public:
    explicit StreamingQuantileBuilder(int max_buckets = 1000) 
        : max_buckets_(max_buckets) {}
    
    void add_point(double value, double weight = 1.0) {
        if (!std::isfinite(value) || !std::isfinite(weight) || weight <= 0.0) return;
        
        // Find insertion point
        auto it = std::lower_bound(buckets_.begin(), buckets_.end(), value,
                                  [](const Bucket& b, double v) { return b.value < v; });
        
        if (it != buckets_.end() && std::abs(it->value - value) < 1e-12) {
            // Merge with existing bucket
            it->weight += weight;
            it->count += 1;
        } else {
            // Insert new bucket
            buckets_.emplace(it, value, weight, 1);
        }
        
        total_weight_ += weight;
        
        // Compress if too many buckets
        if (static_cast<int>(buckets_.size()) > max_buckets_) {
            compress();
        }
    }
    
    std::vector<double> get_quantile_edges(int n_bins) const {
        if (buckets_.empty()) return {0.0, 1.0};
        if (n_bins <= 1) return {buckets_.front().value - 1e-12, buckets_.back().value + 1e-12};
        
        std::vector<double> edges(n_bins + 1);
        edges[0] = buckets_.front().value - 1e-12;
        edges[n_bins] = buckets_.back().value + 1e-12;
        
        // Build cumulative weights
        std::vector<double> cum_weights(buckets_.size());
        cum_weights[0] = buckets_[0].weight;
        for (size_t i = 1; i < buckets_.size(); ++i) {
            cum_weights[i] = cum_weights[i-1] + buckets_[i].weight;
        }
        
        // Find internal edges
        for (int i = 1; i < n_bins; ++i) {
            const double target = (static_cast<double>(i) - 0.5) / n_bins * total_weight_;
            auto it = std::lower_bound(cum_weights.begin(), cum_weights.end(), target);
            const size_t idx = (it == cum_weights.end()) ? (buckets_.size() - 1) 
                                                        : (it - cum_weights.begin());
            edges[i] = buckets_[idx].value;
        }
        
        // Ensure strict increasing
        for (int i = 1; i <= n_bins; ++i) {
            if (edges[i] <= edges[i-1]) {
                edges[i] = edges[i-1] + 1e-12;
            }
        }
        
        return edges;
    }
    
private:
    void compress() {
        if (buckets_.size() <= 2) return;
        
        // Simple compression: merge buckets with smallest weight differences
        std::vector<Bucket> new_buckets;
        new_buckets.reserve(max_buckets_);
        
        // Keep some samples at regular intervals
        const size_t step = buckets_.size() / (max_buckets_ / 2);
        for (size_t i = 0; i < buckets_.size(); i += std::max(size_t{1}, step)) {
            new_buckets.push_back(buckets_[i]);
        }
        
        // Always keep first and last
        if (new_buckets.front().value != buckets_.front().value) {
            new_buckets.insert(new_buckets.begin(), buckets_.front());
        }
        if (new_buckets.back().value != buckets_.back().value) {
            new_buckets.push_back(buckets_.back());
        }
        
        buckets_ = std::move(new_buckets);
    }
};

// =============================== Analyze feature importance ===============

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
    
    // Basic statistics with single pass
    double val_min = std::numeric_limits<double>::max();
    double val_max = std::numeric_limits<double>::lowest();
    double val_sum = 0.0, val_sq = 0.0;
    double grad_sum = 0.0, grad_sq = 0.0;
    double grad_min = std::numeric_limits<double>::max();
    double grad_max = std::numeric_limits<double>::lowest();
    
    std::vector<double> unique_vals;
    unique_vals.reserve(std::min(values.size(), size_t{1000})); // Cap for performance
    
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
            
            if (unique_vals.size() < 1000) {
                unique_vals.push_back(v);
            }
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
    
    stats.variance = (n > 1) ? (val_sq - val_sum * val_mean) / (n - 1) : 0.0;
    stats.gradient_variance = (n > 1) ? (grad_sq - grad_sum * grad_mean) / (n - 1) : 0.0;
    stats.value_range = val_max - val_min;
    stats.gradient_range = grad_max - grad_min;
    
    // Count unique values (approximate for large datasets)
    std::sort(unique_vals.begin(), unique_vals.end());
    unique_vals.erase(std::unique(unique_vals.begin(), unique_vals.end()), unique_vals.end());
    stats.unique_count = static_cast<int>(unique_vals.size());
    
    // Categorical check
    stats.is_categorical = (stats.unique_count <= std::min(cfg.max_bins / 4, 32));
    
    // Compute complexity score (simplified for performance)
    if (finite_count >= 10) {
        // Sample-based complexity estimation
        const size_t sample_size = std::min(finite_count, size_t{1000});
        double complexity_sum = 0.0;
        int complexity_count = 0;
        
        for (size_t i = 2; i < sample_size && i < values.size(); ++i) {
            const double g_curr = gradients[i];
            const double g_prev = gradients[i-1];
            const double g_prev2 = gradients[i-2];
            
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
        const double factor = 1.5 + 0.5 * stats.importance_score + 0.3 * stats.complexity_score;
        stats.suggested_bins = std::min(cfg.max_bins, 
                                       std::max(cfg.min_bins, 
                                               static_cast<int>(cfg.target_bins * factor)));
        stats.allocation_reason = "high_importance_complex";
    } else if (stats.importance_score > cfg.importance_threshold) {
        const double factor = 1.2 + 0.4 * stats.importance_score;
        stats.suggested_bins = std::min(cfg.max_bins,
                                       std::max(cfg.min_bins,
                                               static_cast<int>(cfg.target_bins * factor)));
        stats.allocation_reason = "high_importance";
    } else if (stats.complexity_score > cfg.complexity_threshold) {
        const double factor = 1.1 + 0.3 * stats.complexity_score;
        stats.suggested_bins = std::min(cfg.max_bins,
                                       std::max(cfg.min_bins,
                                               static_cast<int>(cfg.target_bins * factor)));
        stats.allocation_reason = "high_complexity";
    } else if (stats.unique_count < cfg.min_bins) {
        stats.suggested_bins = std::max(2, stats.unique_count);
        stats.allocation_reason = "few_unique_values";
    } else {
        stats.suggested_bins = cfg.target_bins;
        stats.allocation_reason = "normal";
    }
    
    return stats;
}

// =============================== Rest of existing code with minimal changes ===============

struct FeatureBins {
    std::vector<double> edges; // strictly increasing, size = nb+1
    bool   is_uniform = false;
    std::string strategy = "uniform";  // "uniform"|"quantile"|"categorical"|"grad_aware"|"adaptive"
    double lo    = 0.0;  // for O(1) uniform binning
    double width = 1.0;  // for O(1) uniform binning
    
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
    if (uniq.empty())  return {0.0, 1.0};
    if (uniq.size() == 1) {
        const double x = uniq[0];
        return {x - 1e-12, x + 1e-12};
    }
    std::vector<double> e(uniq.size() + 1);
    e.front() = uniq.front() - 1e-12;
    e.back()  = uniq.back()  + 1e-12;
    for (size_t i = 1; i < uniq.size(); ++i) e[i] = 0.5 * (uniq[i - 1] + uniq[i]);
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
    return e;
}

// Streaming-based weighted quantiles for better memory efficiency
inline std::vector<double> weighted_quantile_edges(const std::vector<double>& vals,
                                                   const std::vector<double>& wts,
                                                   int nb) {
    StreamingQuantileBuilder builder(std::min(10000, static_cast<int>(vals.size())));
    
    for (size_t i = 0; i < vals.size(); ++i) {
        const double v = vals[i];
        const double w = (i < wts.size() ? wts[i] : 1.0);
        if (std::isfinite(v) && std::isfinite(w) && w > 0.0) {
            builder.add_point(v, w);
        }
    }
    
    return builder.get_quantile_edges(nb);
}

inline double gradient_complexity(const std::vector<double>& /*v_sorted*/,
                                  const std::vector<double>& g_sorted) {
    const size_t n = g_sorted.size();
    if (n < 3) return 0.45; // mild default
    
    // Sample for performance on large datasets
    const size_t max_samples = 1000;
    const size_t step = std::max(size_t{1}, n / max_samples);
    
    double acc = 0.0;
    size_t count = 0;
    
    // Ensure we don't go out of bounds with the step size
    for (size_t i = 2 * step; i < n; i += step) {
        if (i >= step && i >= 2 * step) { // additional safety check
            const double d = g_sorted[i] - 2.0 * g_sorted[i - step] + g_sorted[i - 2 * step];
            acc += std::abs(d);
            count++;
        }
    }
    
    const double local = (count > 0 ? acc / count : 0.0);
    const double smooth = 1.0 / (1.0 + local);
    const double comp = 0.20 * (1.0 - smooth);
    return std::min(2.0, std::max(0.1, 0.45 + comp));
}

inline std::vector<double> downsample_edges(const std::vector<double>& edges, int new_nb) {
    const int nb = static_cast<int>(edges.empty() ? 0 : (edges.size() - 1));
    if (nb <= 0 || new_nb <= 0) return {0.0, 1.0};
    if (new_nb >= nb) return edges;

    std::vector<double> out; out.reserve(static_cast<size_t>(new_nb) + 1);
    const int E = static_cast<int>(edges.size());
    for (int k = 0; k <= new_nb; ++k) {
        const double pos = static_cast<double>(k) * static_cast<double>(nb) / static_cast<double>(new_nb);
        int    i = static_cast<int>(std::floor(pos));
        double t = pos - i;
        if (i < 0)      { i = 0;     t = 0.0; }
        if (i >= E - 1) { i = E - 2; t = 1.0; }
        const double v = (1.0 - t) * edges[static_cast<size_t>(i)] + t * edges[static_cast<size_t>(i + 1)];
        out.push_back(v);
    }
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
        std::vector<double> v, w;
        v.reserve(values.size());
        w.reserve(values.size());
        
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

        // If few unique values, treat as categorical-like
        std::vector<double> u = v;
        std::sort(u.begin(), u.end());
        u.erase(std::unique(u.begin(), u.end()), u.end());
        if (static_cast<int>(u.size()) <= cfg.max_bins) {
            fb.edges = _midpoint_edges_of_unique(u);
            _check_uniform(fb);
            return fb;
        }

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

        std::vector<double> v_s, g_s, h_s;
        v_s.reserve(v.size());
        g_s.reserve(v.size());
        h_s.reserve(v.size());
        for (size_t k : ord) { 
            v_s.push_back(v[k]); 
            g_s.push_back(g[k]); 
            h_s.push_back(h[k]); 
        }

        // Complexity adjusts bin count around coarse_bins
        const double comp = gradient_complexity(v_s, g_s);
        const int nb = std::max(16, std::min(cfg.max_bins,
                        static_cast<int>(std::lround(static_cast<double>(cfg.coarse_bins) * comp))));

        // Use weighted quantiles by Hessians
        std::vector<double> w(v_s.size());
        for (size_t i = 0; i < v_s.size(); ++i) w[i] = std::max(cfg.eps, h_s[i]);
        fb.edges = weighted_quantile_edges(v_s, w, nb);
        _check_uniform(fb);
        return fb;
    }
};

struct AdaptiveBinner final : IBinningStrategy {
    FeatureBins create_bins(const std::vector<double>& values,
                            const std::vector<double>& gradients,
                            const std::vector<double>& hessians,
                            const HistogramConfig& cfg) override {
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

// ============================ Main Histogram System =================================

class GradientHistogramSystem {
public:
    explicit GradientHistogramSystem(HistogramConfig cfg)
        : cfg_(std::move(cfg)), rng_(cfg_.rng_seed), P_(0), N_(0) {}

    // Fit bins per feature using the chosen strategy
    void fit_bins(const double* X, int N, int P, const double* g, const double* h) {
        if (N <= 0 || P <= 0) throw std::invalid_argument("fit_bins: invalid N or P");
        if (!X || !g || !h)   throw std::invalid_argument("fit_bins: null input");
        N_ = N; P_ = P;

        std::unique_ptr<IBinningStrategy> strat;
        if (cfg_.method == "quantile")           strat = std::make_unique<QuantileBinner>();
        else if (cfg_.method == "hist")          strat = std::make_unique<QuantileBinner>();
        else if (cfg_.method == "adaptive")      strat = std::make_unique<AdaptiveBinner>();
        else                                     strat = std::make_unique<GradientAwareBinner>();

        feature_bins_.assign(P_, FeatureBins{});

        auto process_feature = [&](int j) {
            std::vector<double> col(N_), gj(N_), hj(N_);
            for (int i = 0; i < N_; ++i) {
                const size_t off = static_cast<size_t>(i) * static_cast<size_t>(P_) + static_cast<size_t>(j);
                col[i] = X[off];
                gj[i]  = g[i];
                hj[i]  = h[i];
            }
            FeatureBins fb = strat->create_bins(col, gj, hj, cfg_);

            int max_bins_for_feature;
            if (cfg_.method == "adaptive" && cfg_.adaptive_binning) {
                max_bins_for_feature = fb.stats.suggested_bins;
            } else {
                max_bins_for_feature = cfg_.max_bins;
            }

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
            (void)workers;
            std::vector<std::future<FeatureBins>> futs; 
            futs.reserve(P_);
            for (int j = 0; j < P_; ++j) {
                futs.emplace_back(std::async(std::launch::async, [&, j]{ return process_feature(j); }));
            }
            for (int j = 0; j < P_; ++j) feature_bins_[j] = futs[j].get();
        }

        // Setup variable bin layout and DataBinner
        std::vector<int> bins_per_feat(P_);
        for (int j = 0; j < P_; ++j) {
            bins_per_feat[j] = feature_bins_[j].n_bins();
        }
        layout_.initialize(bins_per_feat);

        std::vector<std::vector<double>> edges_per_feat(P_);
        for (int j = 0; j < P_; ++j) edges_per_feat[j] = feature_bins_[j].edges;

        EdgeSet es;
        es.edges_per_feat = std::move(edges_per_feat);
        binner_ = std::make_unique<DataBinner>(P_);
        binner_->register_edges("hist", std::move(es));

        codes_.reset();
        miss_id_ = binner_->missing_bin_id("hist");
    }

    // Prebin whole matrix X; caches codes internally
    std::pair<std::shared_ptr<std::vector<uint16_t>>, int>
    prebin_dataset(const double* X, int N, int P) {
        if (!binner_) throw std::runtime_error("fit_bins must be called before prebin_dataset");
        if (N != N_ || P != P_) throw std::invalid_argument("prebin_dataset: shape mismatch vs fit_bins");
        auto pr = binner_->prebin(X, N, P, "hist", -1);
        codes_  = pr.first;
        miss_id_= pr.second;
        return pr;
    }

    // Prebin ANY matrix X using fitted edges, without touching the internal cache
    std::pair<std::shared_ptr<std::vector<uint16_t>>, int>
    prebin_matrix(const double* X, int N, int P) const {
        if (!binner_) throw std::runtime_error("fit_bins must be called before prebin_matrix");
        if (P != P_)  throw std::invalid_argument("prebin_matrix: P mismatch vs fit_bins");
        return binner_->prebin(X, N, P, "hist", -1);
    }

    // High-performance histogram builders using optimized accumulator
    template <class GFloat = float, class HFloat = float>
    std::tuple<std::vector<double>, std::vector<double>, std::vector<int>>
    build_histograms_with_counts(const GFloat* g, const HFloat* h,
                                 const int* sample_indices = nullptr,
                                 int n_sub = 0) const {
        HistogramAccumulator acc(&layout_);
        acc.accumulate_samples<true>(codes_->data(), g, h, sample_indices, 
                                    (sample_indices ? n_sub : N_), P_);
        return acc.take_results();
    }

    template <class GFloat = float, class HFloat = float>
    std::pair<std::vector<double>, std::vector<double>>
    build_histograms(const GFloat* g, const HFloat* h,
                     const int* sample_indices = nullptr,
                     int n_sub = 0) const {
        HistogramAccumulator acc(&layout_);
        acc.accumulate_samples<false>(codes_->data(), g, h, sample_indices, 
                                     (sample_indices ? n_sub : N_), P_);
        return acc.take_gh_results();
    }

    // Legacy methods for compatibility
    template <class GFloat = float, class HFloat = float>
    std::tuple<std::vector<double>, std::vector<double>, std::vector<int>>
    build_histograms_fast_with_counts(const GFloat* g, const HFloat* h,
                                      const int* sample_indices = nullptr,
                                      int n_sub = 0) const {
        return build_histograms_with_counts(g, h, sample_indices, n_sub);
    }

    template <class GFloat = float, class HFloat = float>
    std::pair<std::vector<double>, std::vector<double>>
    build_histograms_fast(const GFloat* g, const HFloat* h,
                          const int* sample_indices = nullptr,
                          int n_sub = 0) const {
        return build_histograms(g, h, sample_indices, n_sub);
    }

    // Helper to extract histogram for a specific feature from the packed format
    std::tuple<std::vector<double>, std::vector<double>, std::vector<int>>
    extract_feature_histogram(const std::vector<double>& Hg, 
                             const std::vector<double>& Hh,
                             const std::vector<int>& C,
                             int feature) const {
        if (!binner_ || feature < 0 || feature >= P_) {
            throw std::invalid_argument("Invalid feature index");
        }
        
        const size_t feat_offset = layout_.feature_offset(feature);
        const int feat_bins = layout_.bins_for_feature(feature);
        
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

    // Get feature histogram offsets for manual indexing
    std::vector<size_t> get_feature_offsets() const {
        std::vector<size_t> offsets(P_ + 1);
        for (int j = 0; j < P_; ++j) {
            offsets[j] = layout_.feature_offset(j);
        }
        offsets[P_] = layout_.total_size(); // Final offset
        return offsets;
    }

    // Accessors
    int P() const { return P_; }
    int N() const { return N_; }
    int missing_bin_id() const { return miss_id_; }
    
    // Per-feature accessors
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

    // Get all bin counts at once
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

    // Feature analysis results
    const FeatureStats& feature_stats(int j) const { 
        return feature_bins_.at(j).stats; 
    }
    
    // Get summary of bin allocation decisions
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

    VariableBinLayout            layout_;
    std::vector<FeatureBins>     feature_bins_;
    std::unique_ptr<DataBinner>  binner_;
    std::shared_ptr<std::vector<uint16_t>> codes_;
};

} // namespace foretree