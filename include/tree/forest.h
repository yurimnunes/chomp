#pragma once
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <execution>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "gradient_hist_system.hpp"
#include "unified_tree.hpp"

namespace foretree {

// ============================================================================
// ForeForest — bagging (Random Forest) + boosting (GBDT) with optional DART
// ============================================================================

struct ForeForestConfig {
    enum class Mode { Bagging, GBDT } mode = Mode::Bagging;

    // --- Common ---
    int n_estimators = 100;
    double learning_rate = 0.1; // used in GBDT/DART
    uint64_t rng_seed = 123456789ULL;

    // Allow trees to look at raw X (float32) + missing mask for Exact/Hybrid
    // splits
    bool use_raw_matrix_for_exact = false;

    // Shared systems/configs
    HistogramConfig hist_cfg{}; // binning
    TreeConfig tree_cfg{};      // tree growth/splits/posting

    // --- Column sampling (wire through UnifiedTree when ready) ---
    double colsample_bytree = 1.0; // fraction of features per tree
    double colsample_bynode = 1.0; // fraction per node (split-time)

    // --- Bagging (Random Forest) ---
    double rf_row_subsample = 1.0; // fraction of rows per tree (bootstrap size)
    bool rf_bootstrap = true;      // with replacement
    bool rf_bootstrap_dedup = false; // keep multiplicities by default
    bool rf_parallel = true;       // parallel tree building (requires PSTL)

    // --- Boosting (GBDT) ---
    double gbdt_row_subsample = 1.0; // stochastic boosting fraction
    bool gbdt_use_subsample = false;

    // --- DART (Dropout for trees) ---
    bool dart_enabled = false;
    double dart_drop_rate = 0.1; // prob of dropping each existing tree per iter
    int dart_max_drop = 3;       // cap number of dropped trees
    bool dart_normalize = true;  // normalize weights by 1/(k+1)
};

class ForeForest {
public:
    explicit ForeForest(ForeForestConfig cfg)
        : cfg_(std::move(cfg)),
          rng_(cfg_.rng_seed),
          ghs_(std::make_unique<GradientHistogramSystem>(cfg_.hist_cfg)) {}

    // Provide raw matrix if you want Exact/Hybrid splits to use float values +
    // missing mask Xraw: pointer to N * P row-major (float). Xmiss: N*P mask
    // (0=no-miss, nonzero=miss) or nullptr.
    void set_raw_matrix(const float *Xraw, const uint8_t *Xmiss_or_null) {
        Xraw_ = Xraw;
        Xmiss_ = Xmiss_or_null;
        cfg_.use_raw_matrix_for_exact = (Xraw_ != nullptr);
    }

    // Main fit API (dense double). Bins once, then trains trees for the chosen
    // mode.
    void fit_complete(const double *X, int N, int P, const double *y) {
        if (!X || !y)
            throw std::invalid_argument("ForeForest::fit_complete: null X or y");
        if (N <= 0 || P <= 0)
            throw std::invalid_argument("ForeForest::fit_complete: invalid N or P");

        N_ = N;
        P_ = P;

        // Base score for squared loss (mean of y); used by GBDT
        base_score_ = 0.0;
        for (int i = 0; i < N_; ++i) base_score_ += y[i];
        base_score_ /= std::max(1, N_);

        // 1) Fit binning with dummy gradients/hessians (for quantiles/sketch).
        std::vector<double> dummy_g(N_, 0.0), dummy_h(N_, 1.0);
        ghs_->fit_bins(X, N_, P_, dummy_g.data(), dummy_h.data());

        // 2) Pre-bin dataset and keep a shared handle to codes (N x P).
        auto pr = ghs_->prebin_dataset(X, N_, P_);
        codes_ = pr.first;    // shared_ptr<std::vector<uint16_t>>
        miss_id_ = pr.second; // missing bin id if used

        // 3) Shared per-sample grad/hess buffers (float for hist speed)
        std::vector<float> g(N_), h(N_, 1.0f);

        // 4) Train
        trees_.clear();
        tree_weights_.clear();
        trees_.reserve(cfg_.n_estimators);
        tree_weights_.reserve(cfg_.n_estimators);

        if (cfg_.mode == ForeForestConfig::Mode::Bagging) {
            train_bagging_(g, h, y);
        } else {
            train_gbdt_(g, h, y);
        }
    }

    // Overload using vectors
    void fit_complete(const std::vector<double> &X, int N, int P,
                      const std::vector<double> &y) {
        if ((int)X.size() != N * P)
            throw std::invalid_argument("ForeForest::fit_complete: X.size != N*P");
        if ((int)y.size() != N)
            throw std::invalid_argument("ForeForest::fit_complete: y.size != N");
        fit_complete(X.data(), N, P, y.data());
    }

    // Predict on dense matrix (double). Uses the forest’s binner; returns
    // per-row predictions.
    std::vector<double> predict(const double *X, int N, int P) const {
        if (!ghs_->binner())
            throw std::runtime_error("ForeForest::predict: model not fitted");
        if (P != P_)
            throw std::invalid_argument("ForeForest::predict: P mismatch");
        auto pr = ghs_->prebin_matrix(X, N, P);
        auto out = predict_from_binned_(*pr.first, N, P);
        if (cfg_.mode == ForeForestConfig::Mode::GBDT) {
            for (double &v : out) v += base_score_;
        }
        return out;
    }
    std::vector<double> predict(const std::vector<double> &X, int N,
                                int P) const {
        return predict(X.data(), N, P);
    }

    // Feature importance (gain) aggregated across trees, weighted by per-tree
    // weights.
    std::vector<double> feature_importance_gain() const {
        std::vector<double> agg(P_, 0.0);
        for (size_t t = 0; t < trees_.size(); ++t) {
            const double wt = tree_weights_[t];
            const auto &g = trees_[t].feature_importance_gain();
            const int m = std::min<int>(P_, (int)g.size());
            for (int j = 0; j < m; ++j)
                agg[j] += wt * g[j];
        }
        return agg;
    }

    int size() const { return (int)trees_.size(); }
    void clear() {
        trees_.clear();
        tree_weights_.clear();
        codes_.reset();
        N_ = P_ = 0;
        base_score_ = 0.0;
        ghs_ = std::make_unique<GradientHistogramSystem>(cfg_.hist_cfg);
    }

private:
    // ===================== Bagging (Random Forest) ===========================
    void train_bagging_(std::vector<float> &g, std::vector<float> &h,
                        const double *y) {
        // CART-like regression via gradient trees: set g=-y, h=1 so leaves
        // learn local mean(y).
        for (int i = 0; i < N_; ++i) {
            g[i] = (float)(-y[i]);
            h[i] = 1.0f;
        }

        const int M = std::max(1, cfg_.n_estimators);

        // Pre-generate row samples per tree
        std::vector<std::vector<int>> boot_rows((size_t)M);

        // Per-thread RNGs to avoid data races under parallel generation
        std::vector<std::mt19937_64> rngs((size_t)M);
        for (int t = 0; t < M; ++t)
            rngs[(size_t)t] =
                std::mt19937_64(cfg_.rng_seed + 0x9E3779B97F4A7C15ULL * (t + 1));

        auto gen_sample = [&](int t) {
            auto &rows = boot_rows[(size_t)t];
            auto &rng  = rngs[(size_t)t];
            std::uniform_int_distribution<int> J(0, N_ - 1);

            const int k = std::max(1, (int)std::round(cfg_.rf_row_subsample * N_));
            if (cfg_.rf_bootstrap) {
                // with replacement: keep multiplicities unless user forces dedup
                rows.resize((size_t)k);
                for (int i = 0; i < k; ++i) rows[(size_t)i] = J(rng);
                if (cfg_.rf_bootstrap_dedup) {
                    std::sort(rows.begin(), rows.end());
                    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
                    if (rows.empty()) rows.push_back(J(rng));
                }
            } else {
                // without replacement via shuffle
                std::vector<int> all(N_);
                std::iota(all.begin(), all.end(), 0);
                std::shuffle(all.begin(), all.end(), rng);
                rows.assign(all.begin(), all.begin() + std::min(k, N_));
                std::sort(rows.begin(), rows.end());
                rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
            }
        };

        if (cfg_.rf_parallel) {
            std::vector<int> tids(M);
            std::iota(tids.begin(), tids.end(), 0);
            std::for_each(std::execution::par, tids.begin(), tids.end(),
                          [&](int t) { gen_sample(t); });
        } else {
            for (int t = 0; t < M; ++t) gen_sample(t);
        }

        trees_.resize((size_t)M);
        tree_weights_.assign((size_t)M, 1.0); // we'll average at predict time

        auto build_one = [&](int t) {
            UnifiedTree T(cfg_.tree_cfg, ghs_.get());
            if (cfg_.use_raw_matrix_for_exact)
                T.set_raw_matrix(Xraw_, Xmiss_);
            // TODO: plumb colsample_bytree / bynode into T if supported
            T.fit_with_row_ids(*codes_, N_, P_, g, h, boot_rows[(size_t)t]);
            trees_[(size_t)t] = std::move(T);
        };

        if (cfg_.rf_parallel) {
            std::vector<int> tids(M);
            std::iota(tids.begin(), tids.end(), 0);
            std::for_each(std::execution::par, tids.begin(), tids.end(),
                          [&](int t) { build_one(t); });
        } else {
            for (int t = 0; t < M; ++t) build_one(t);
        }
    }

    // ===================== Boosting (GBDT, optional DART) ====================
    void train_gbdt_(std::vector<float> &g, std::vector<float> &h,
                     const double *y) {
        const int M = std::max(1, cfg_.n_estimators);
        // Start from base_score_ (mean(y)) for squared loss
        std::vector<double> F(N_, base_score_);

        trees_.clear();
        tree_weights_.clear();
        trees_.reserve(M);
        tree_weights_.reserve(M);

        std::uniform_real_distribution<double> U(0.0, 1.0);

        auto compute_grad_hess = [&](const std::vector<double> &Fbase) {
            // Squared loss: L = 0.5*(y - F)^2; grad wrt F is (F - y); hess is 1
            for (int i = 0; i < N_; ++i) {
                g[i] = (float)(Fbase[i] - y[i]); // F - y
                h[i] = 1.0f;
            }
        };

        for (int m = 0; m < M; ++m) {
            // ----- DART: choose a dropout set over existing trees -----
            std::vector<int> dropped;
            if (cfg_.dart_enabled && !trees_.empty()) {
                for (int t = 0; t < (int)trees_.size(); ++t) {
                    if ((int)dropped.size() >= cfg_.dart_max_drop) break;
                    if (U(rng_) < cfg_.dart_drop_rate) dropped.push_back(t);
                }
                if (dropped.empty()) {
                    std::uniform_int_distribution<int> D(0, (int)trees_.size() - 1);
                    dropped.push_back(D(rng_));
                }
            }

            // Build F_base = ensemble prediction excluding dropped trees (if any)
            std::vector<double> Fbase = F;
            if (!dropped.empty())
                subtract_dropped_contrib_(Fbase, dropped);

            // Compute gradient on F_base (DART) or on F (plain GBDT)
            compute_grad_hess(cfg_.dart_enabled ? Fbase : F);

            // Optional stochastic subsampling for this iteration
            std::vector<int> rows;
            if (cfg_.gbdt_use_subsample && cfg_.gbdt_row_subsample < 1.0) {
                rows.reserve(N_);
                for (int i = 0; i < N_; ++i)
                    if (U(rng_) < cfg_.gbdt_row_subsample)
                        rows.push_back(i);
                if (rows.empty()) {
                    std::uniform_int_distribution<int> D(0, N_ - 1);
                    rows.push_back(D(rng_));
                }
                std::sort(rows.begin(), rows.end());
                rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
            } else {
                rows.resize(N_);
                std::iota(rows.begin(), rows.end(), 0);
            }

            // Train tree on current (g,h)
            UnifiedTree T(cfg_.tree_cfg, ghs_.get());
            if (cfg_.use_raw_matrix_for_exact)
                T.set_raw_matrix(Xraw_, Xmiss_);
            // TODO: plumb colsample_bytree / bynode into T if supported
            T.fit_with_row_ids(*codes_, N_, P_, g, h, rows);

            // Weight for new tree (with DART normalization if needed)
            double wt_new = cfg_.learning_rate;

            // === DART reweighting: resync F to reflect normalization of dropped trees ===
            if (cfg_.dart_enabled && cfg_.dart_normalize && !dropped.empty()) {
                const int k = (int)dropped.size();
                const double scale = 1.0 / double(k + 1);

                // For each dropped tree: scale its weight and adjust F accordingly
                std::vector<double> pred(N_, 0.0);
                for (int t : dropped) {
                    predict_tree_on_binned_(trees_[(size_t)t], *codes_, N_, P_, pred);
                    const double old_w = tree_weights_[(size_t)t];
                    const double new_w = old_w * scale;
                    const double delta = new_w - old_w; // negative increment
                    for (int i = 0; i < N_; ++i)
                        F[i] += delta * pred[i];
                    tree_weights_[(size_t)t] = new_w;
                }

                // New tree receives lr/(k+1)
                wt_new = cfg_.learning_rate * scale;
            }

            trees_.push_back(std::move(T));
            tree_weights_.push_back(wt_new);

            // Update F with the new tree contribution
            add_tree_contrib_(F, trees_.back(), wt_new);
        }
    }

    // Subtract contributions of dropped trees from Fbase (used by DART)
    void subtract_dropped_contrib_(std::vector<double> &Fbase,
                                   const std::vector<int> &dropped) const {
        std::vector<double> pred(N_, 0.0);
        for (int t : dropped) {
            predict_tree_on_binned_(trees_[(size_t)t], *codes_, N_, P_, pred);
            const double wt = tree_weights_[(size_t)t];
            for (int i = 0; i < N_; ++i)
                Fbase[i] -= wt * pred[i];
        }
    }

    // Add contribution of one tree (with weight wt) to F
    void add_tree_contrib_(std::vector<double> &F, const UnifiedTree &T,
                           double wt) const {
        std::vector<double> pred(N_, 0.0);
        predict_tree_on_binned_(T, *codes_, N_, P_, pred);
        for (int i = 0; i < N_; ++i)
            F[i] += wt * pred[i];
    }

    // Predict for pre-binned matrix; for Bagging we average tree outputs
    std::vector<double> predict_from_binned_(const std::vector<uint16_t> &Xb,
                                             int N, int P) const {
        (void)P; // already validated earlier
        std::vector<double> out(N, 0.0);

        // Simple serial accumulation; can be parallelized by trees if desired
        std::vector<double> pred(N, 0.0);
        for (size_t t = 0; t < trees_.size(); ++t) {
            const double wt = tree_weights_[t];
            predict_tree_on_binned_(trees_[t], Xb, N, P_, pred);
            for (int i = 0; i < N; ++i)
                out[i] += wt * pred[i];
        }

        if (cfg_.mode == ForeForestConfig::Mode::Bagging && !trees_.empty()) {
            const double invT = 1.0 / (double)trees_.size();
            for (double &v : out)
                v *= invT;
        }
        return out;
    }

    // Predict a single tree on pre-binned matrix into 'dst'
    static void predict_tree_on_binned_(const UnifiedTree &T,
                                        const std::vector<uint16_t> &Xb, int N,
                                        int P, std::vector<double> &dst) {
        auto v = T.predict(Xb, N, P);
        if ((int)dst.size() != N)
            dst.resize(N);
        std::copy(v.begin(), v.end(), dst.begin());
    }

private:
    ForeForestConfig cfg_;
    std::mt19937_64 rng_;

    // Shared across trees
    std::unique_ptr<GradientHistogramSystem> ghs_;
    std::shared_ptr<std::vector<uint16_t>> codes_;
    int miss_id_ = -1;

    // Raw matrix access for exact/hybrid splits
    const float *Xraw_ = nullptr;
    const uint8_t *Xmiss_ = nullptr;

    // Learned ensemble
    std::vector<UnifiedTree> trees_;
    std::vector<double> tree_weights_; // per-tree weights (GBDT uses lr;
                                       // Bagging uses 1.0 then we average)

    // Data shape
    int N_ = 0, P_ = 0;

    // Bias for boosting (squared loss)
    double base_score_ = 0.0;
};

} // namespace foretree
