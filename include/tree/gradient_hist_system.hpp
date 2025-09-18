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
    std::string method = "grad_aware"; // "hist" | "quantile" | "grad_aware"
    int   max_bins = 256;
    bool  use_missing_bin = true;

    // quantile / grad-aware
    int   coarse_bins = 64;      // base for grad_aware auto-tuning
    bool  density_aware = true;  // reserved for future

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

// =============================== Feature bins =================================

struct FeatureBins {
    std::vector<double> edges; // strictly increasing, size = nb+1
    bool   is_uniform = false;
    std::string strategy = "uniform";  // "uniform"|"quantile"|"categorical"|"grad_aware"
    double lo    = 0.0;  // for O(1) uniform binning
    double width = 1.0;  // for O(1) uniform binning

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
    // Filter non-finite values and non-positive weights
    struct Pair { double v, w; };
    std::vector<Pair> vw; vw.reserve(vals.size());
    for (size_t i = 0; i < vals.size(); ++i) {
        const double v = vals[i];
        const double w = (i < wts.size() ? wts[i] : 1.0);
        if (std::isfinite(v) && std::isfinite(w) && w > 0.0) vw.push_back({v, w});
    }
    if (vw.empty()) return {0.0, 1.0};
    std::sort(vw.begin(), vw.end(), [](const Pair& a, const Pair& b){ return a.v < b.v; });

    const size_t n = vw.size();
    std::vector<double> cdf(n);
    cdf[0] = vw[0].w;
    for (size_t i = 1; i < n; ++i) cdf[i] = cdf[i-1] + vw[i].w;
    const double W = cdf.back();
    if (!(W > 0.0)) return exact_quantile_edges(vals, nb);

    nb = std::max(1, nb);
    std::vector<double> e; e.reserve(nb + 1);
    for (int i = 0; i <= nb; ++i) {
        const double target = (double)i / (double)nb * W;
        auto it = std::lower_bound(cdf.begin(), cdf.end(), target);
        const size_t idx = (it == cdf.end() ? (n - 1) : (size_t)std::distance(cdf.begin(), it));
        e.push_back(vw[idx].v); // stepwise weighted quantile (robust & simple)
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

// ============================ Histogram system =================================

class GradientHistogramSystem {
public:
    explicit GradientHistogramSystem(HistogramConfig cfg)
        : cfg_(std::move(cfg)), rng_(cfg_.rng_seed), P_(0), N_(0) {}

    // ---- Fit bins per feature using the chosen strategy (quantile/grad_aware) ----
    // X: row-major (N x P); g/h length N
    void fit_bins(const double* X, int N, int P, const double* g, const double* h) {
        if (N <= 0 || P <= 0) throw std::invalid_argument("fit_bins: invalid N or P");
        if (!X || !g || !h)   throw std::invalid_argument("fit_bins: null input");
        N_ = N; P_ = P;

        std::unique_ptr<IBinningStrategy> strat;
        if (cfg_.method == "quantile")      strat = std::make_unique<QuantileBinner>();
        else if (cfg_.method == "hist")     strat = std::make_unique<QuantileBinner>(); // baseline
        else                                strat = std::make_unique<GradientAwareBinner>();

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

            // Enforce capacity (nb <= max_bins) by downsampling edges
            const int nb = fb.n_bins();
            if (nb > cfg_.max_bins) {
                fb.edges = downsample_edges(fb.edges, cfg_.max_bins);
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
        // DataBinner::register_edges will recompute finite_bins/missing_bin_id
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

        const int    tot_bins    = this->total_bins();      // <-- renamed to avoid shadow/call clash
        const size_t feat_stride = (size_t)tot_bins;
        const uint16_t* CODES = codes_->data();

        auto accumulate_one = [&](int i){
            const uint16_t* row = CODES + (size_t)i * (size_t)P_;
            const double gi = (double)g[i];
            const double hi = (double)h[i];
            for (int j = 0; j < P_; ++j) {
                const uint16_t b = row[j];
                if (b >= (uint16_t)tot_bins) continue; // safety
                const size_t off = (size_t)j * feat_stride + b;
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

    // With counts (G/H/C)
    template <class GFloat = float, class HFloat = float>
    std::tuple<std::vector<double>, std::vector<double>, std::vector<int>>
    build_histograms_fast_with_counts(const GFloat* g, const HFloat* h,
                                      const int* sample_indices = nullptr,
                                      int n_sub = 0) const {
        const size_t stride = (size_t)this->total_bins();
        std::vector<double> Hg((size_t)P_ * stride, 0.0);
        std::vector<double> Hh((size_t)P_ * stride, 0.0);
        std::vector<int>    C ((size_t)P_ * stride, 0);
        _accumulate_hist<true>(g, h, sample_indices, n_sub, Hg, Hh, &C);
        return {std::move(Hg), std::move(Hh), std::move(C)};
    }

    // No counts (G/H)
    template <class GFloat = float, class HFloat = float>
    std::pair<std::vector<double>, std::vector<double>>
    build_histograms_fast(const GFloat* g, const HFloat* h,
                          const int* sample_indices = nullptr,
                          int n_sub = 0) const {
        const size_t stride = (size_t)this->total_bins();
        std::vector<double> Hg((size_t)P_ * stride, 0.0);
        std::vector<double> Hh((size_t)P_ * stride, 0.0);
        _accumulate_hist<false>(g, h, sample_indices, n_sub, Hg, Hh, nullptr);
        return {std::move(Hg), std::move(Hh)};
    }

    // ---- Accessors ------------------------------------------------------------
    int P() const { return P_; }
    int N() const { return N_; }
    int missing_bin_id() const { return miss_id_; }
    int finite_bins() const { return binner_ ? binner_->finite_bins("hist") : cfg_.max_bins; }
    int total_bins()  const { return binner_ ? binner_->total_bins("hist")  : cfg_.total_bins(); }

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
