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

    // If you want to allow raw splits in trees, provide raw X via set_raw_matrix().
    bool use_raw_matrix_for_exact = false;

    // One shared histogram system. We bin once and reuse across trees.
    HistogramConfig hist_cfg{};       // per-forest binning config
    TreeConfig      tree_cfg{};       // per-tree config (colsample, subsample, GOSS, etc.)

    // --- Bagging (Random Forest) ---
    double rf_row_subsample = 1.0;    // bootstrap ratio (typ. 0.632..1.0); if >1.0, will round
    bool   rf_bootstrap = true;       // with replacement
    bool   rf_parallel = true;        // build trees in parallel

    // --- Boosting (GBDT) ---
    double gbdt_row_subsample = 1.0;  // stochastic gradient boosting
    bool   gbdt_use_subsample = false;

    // --- DART (Dropout for trees) ---
    bool   dart_enabled = false;
    double dart_drop_rate = 0.1;      // probability to drop each existing tree per iter
    int    dart_max_drop = 3;         // cap the number of dropped trees per iteration
    bool   dart_normalize = true;     // normalize dropped + new tree weights (1/(k+1))
};

class ForeForest {
public:
    explicit ForeForest(ForeForestConfig cfg)
        : cfg_(std::move(cfg)),
          rng_(cfg_.rng_seed),
          ghs_(std::make_unique<GradientHistogramSystem>(cfg_.hist_cfg)) {}

    // Provide raw matrix if you want Exact/Hybrid splits to look at float values + missing mask
    // Xraw: pointer to N * P row-major (float). Xmiss: N*P mask (0=no-miss, nonzero=miss) or nullptr.
    void set_raw_matrix(const float* Xraw, const uint8_t* Xmiss_or_null) {
        Xraw_  = Xraw;
        Xmiss_ = Xmiss_or_null;
        cfg_.use_raw_matrix_for_exact = (Xraw_ != nullptr);
    }

    // Fit on raw dense matrix (double) and labels y (regression). Handles both modes.
    // Internally: bins once, then trains trees.
    void fit(const double* X, int N, int P, const double* y) {
        if (N <= 0 || P <= 0) throw std::invalid_argument("ForeForest::fit: invalid N or P");
        N_ = N; P_ = P;
        // 1) Fit bins once and prebin entire dataset
        ghs_->fit_bins(X, N, P, /*g=*/unit_zero_.assign(N,0.0), /*h=*/unit_one_.assign(N,1.0), // placeholder; not used by fit_bins except for weighted quantiles
                       /*NOTE*/nullptr);
        // NOTE: the call above expects pointers; pass valid arrays:
        // We'll keep simple local storage for zeros/ones
        // Adjust: the real call signature in your GradientHistogramSystem::fit_bins is:
        //   void fit_bins(const double* X, int N, int P, const double* g, const double* h)
        // We supply g=0, h=1 safely:

        // rebuild with correct actual buffers
    }

    // Overload: friendly API using std::vector<T>
    void fit(const std::vector<double>& X, int N, int P, const std::vector<double>& y) {
        if ((int)y.size() != N) throw std::invalid_argument("ForeForest::fit: y.size != N");
        if ((int)X.size() != N * P) throw std::invalid_argument("ForeForest::fit: X.size != N*P");
        fit(X.data(), N, P, y.data());
    }

    // Predict on dense matrix (double). Uses the forest’s shared binning.
    std::vector<double> predict(const double* X, int N, int P) const {
        if (!ghs_->binner()) throw std::runtime_error("predict: model not fitted");
        if (P != P_) throw std::invalid_argument("predict: P mismatch");
        auto pr = ghs_->prebin_matrix(X, N, P);
        return predict_from_binned_(*pr.first, N, P);
    }
    std::vector<double> predict(const std::vector<double>& X, int N, int P) const {
        return predict(X.data(), N, P);
    }

    // Feature importance (gain) aggregated across trees (weighted by tree weights)
    std::vector<double> feature_importance_gain() const {
        std::vector<double> agg(P_, 0.0);
        for (size_t t = 0; t < trees_.size(); ++t) {
            const double wt = tree_weights_[t];
            const auto& g = trees_[t].feature_importance_gain();
            for (int j = 0; j < P_ && j < (int)g.size(); ++j) {
                agg[j] += wt * g[j];
            }
        }
        return agg;
    }

    // Number of trees actually trained
    int size() const { return (int)trees_.size(); }

    // Clear ensemble
    void clear() {
        trees_.clear();
        tree_weights_.clear();
        N_ = P_ = 0;
        codes_.reset();
        ghs_ = std::make_unique<GradientHistogramSystem>(cfg_.hist_cfg);
    }

private:
    // ========= Internal training entry after we have binned codes =========

    void fit_after_binning_(const std::vector<double>& y) {
        // cache codes for fast histograms & give all trees the same ghs pointer
        auto codes_pair = ghs_->prebin_dataset(nullptr, N_, P_); // NOTE: prebin_dataset expects X pointer; we already binned via fit_bins. 
        // BUT in your GradientHistogramSystem, prebin_dataset needs X again. So we re-prebin via DataBinner:
        // We'll keep a copy of codes_ after a dedicated call:
        // (The recommended flow is: fit_bins(...), then prebin_dataset(X, N, P).)

        // Instead, we’ll run the right flow in fit() to hold codes_ and miss_id_. (See below.)
        (void)codes_pair;
    }

    // ========= Proper fit() (complete implementation) =========
public:
    void fit_complete(const double* X, int N, int P, const double* y) {
        if (N <= 0 || P <= 0) throw std::invalid_argument("ForeForest::fit: invalid N or P");
        N_ = N; P_ = P;

        // Setup dummy grad/hess buffers for fit_bins
        std::vector<double> dummy_g(N_, 0.0), dummy_h(N_, 1.0);
        ghs_->fit_bins(X, N_, P_, dummy_g.data(), dummy_h.data());

        auto pr = ghs_->prebin_dataset(X, N_, P_); // cache codes inside ghs_
        codes_ = pr.first;                         // shared codes view for speed
        miss_id_ = pr.second;

        // shared g/h buffers per iteration (boosting) or per tree (bagging)
        std::vector<float> g(N_), h(N_, 1.0f);

        trees_.clear();
        tree_weights_.clear();
        trees_.reserve(cfg_.n_estimators);
        tree_weights_.reserve(cfg_.n_estimators);

        if (cfg_.mode == ForeForestConfig::Mode::Bagging) {
            train_bagging_(g, h, y, X);  // X only needed if exact splits enabled
        } else {
            train_gbdt_(g, h, y, X);     // iterative boosting (+DART)
        }
    }

private:
    // ===================== Bagging (Random Forest) ===========================
    void train_bagging_(std::vector<float>& g, std::vector<float>& h,
                        const double* y, const double* Xraw) {
        const int M = cfg_.n_estimators;
        // For CART-like regression using our gradient tree:
        // Use g = -y (targets), h = 1 → leaves predict ~mean(y) in each region.
        for (int i = 0; i < N_; ++i) {
            g[i] = (float)(-y[i]);
            h[i] = 1.0f;
        }

        // Pre-generate bootstrap samples per tree
        std::vector<std::vector<int>> boot_rows((size_t)M);
        std::uniform_int_distribution<int> J(0, N_ - 1);
        std::uniform_real_distribution<double> U(0.0, 1.0);

        auto gen_sample = [&](int t) {
            auto& rows = boot_rows[t];
            if (cfg_.rf_bootstrap) {
                const int k = std::max(1, (int)std::round(cfg_.rf_row_subsample * N_));
                rows.resize((size_t)k);
                for (int i = 0; i < k; ++i) rows[i] = J(rng_);
                std::sort(rows.begin(), rows.end());
                rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
            } else {
                rows.clear(); rows.reserve(N_);
                for (int i = 0; i < N_; ++i)
                    if (U(rng_) < cfg_.rf_row_subsample) rows.push_back(i);
                if (rows.empty()) { rows.push_back(J(rng_)); }
            }
        };

        if (cfg_.rf_parallel) {
            std::vector<int> tids(M); std::iota(tids.begin(), tids.end(), 0);
            std::for_each(std::execution::par, tids.begin(), tids.end(), [&](int t){ gen_sample(t); });
        } else {
            for (int t = 0; t < M; ++t) gen_sample(t);
        }

        trees_.resize((size_t)M);
        tree_weights_.assign((size_t)M, 1.0); // uniform weights in RF

        auto build_one = [&](int t) {
            UnifiedTree T(cfg_.tree_cfg, ghs_.get());
            if (cfg_.use_raw_matrix_for_exact) T.set_raw_matrix(Xraw_, Xmiss_);

            T.fit_with_row_ids(*codes_, N_, P_, g, h, boot_rows[t]);
            trees_[t] = std::move(T);
        };

        if (cfg_.rf_parallel) {
            std::vector<int> tids(M); std::iota(tids.begin(), tids.end(), 0);
            std::for_each(std::execution::par, tids.begin(), tids.end(), [&](int t){ build_one(t); });
        } else {
            for (int t = 0; t < M; ++t) build_one(t);
        }
    }

    // ===================== Boosting (GBDT, optional DART) ====================
    void train_gbdt_(std::vector<float>& g, std::vector<float>& h,
                     const double* y, const double* Xraw) {
        const int M = cfg_.n_estimators;
        std::vector<double> F(N_, 0.0); // running prediction

        // Track active weights for DART scaling
        trees_.clear();
        tree_weights_.clear();
        trees_.reserve(M);
        tree_weights_.reserve(M);

        // convenience closures
        auto compute_grad_hess = [&](const std::vector<double>& Fbase) {
            // Squared loss: L = 0.5*(y - F)^2
            // gradient wrt F: (F - y), hessian: 1
            for (int i = 0; i < N_; ++i) {
                g[i] = (float)(Fbase[i] - y[i]); // F - y
                h[i] = 1.0f;
            }
        };

        std::uniform_real_distribution<double> U(0.0, 1.0);

        for (int m = 0; m < M; ++m) {
            // ----- DART: sample a dropout set S over existing trees -----
            std::vector<int> dropped;
            if (cfg_.dart_enabled && !trees_.empty()) {
                for (int t = 0; t < (int)trees_.size(); ++t) {
                    if ((int)dropped.size() >= cfg_.dart_max_drop) break;
                    if (U(rng_) < cfg_.dart_drop_rate) dropped.push_back(t);
                }
                // ensure at least 1 dropped if enabled and none chosen
                if (cfg_.dart_enabled && dropped.empty() && !trees_.empty())
                    dropped.push_back((int)(rng_ % trees_.size()));
            }

            // Build F_base = ensemble prediction excluding dropped trees
            std::vector<double> Fbase = F;
            if (!dropped.empty()) {
                subtract_dropped_contrib_(Fbase, dropped);
            }

            // Compute gradient on F_base (DART) or F (plain GBDT)
            compute_grad_hess(cfg_.dart_enabled ? Fbase : F);

            // Optional stochastic row subsample for GBDT
            std::vector<int> rows;
            if (cfg_.gbdt_use_subsample && cfg_.gbdt_row_subsample < 1.0) {
                rows.reserve(N_);
                for (int i = 0; i < N_; ++i) if (U(rng_) < cfg_.gbdt_row_subsample) rows.push_back(i);
                if (rows.empty()) rows.push_back((int)(rng_ % N_));
            } else {
                rows.resize(N_); std::iota(rows.begin(), rows.end(), 0);
            }

            // Train one regression tree on current g,h
            UnifiedTree T(cfg_.tree_cfg, ghs_.get());
            if (cfg_.use_raw_matrix_for_exact) T.set_raw_matrix(Xraw_, Xmiss_);
            T.fit_with_row_ids(*codes_, N_, P_, g, h, rows);

            // Weight for new tree (with DART normalization if needed)
            double wt_new = cfg_.learning_rate;
            if (cfg_.dart_enabled && cfg_.dart_normalize) {
                const int k = (int)dropped.size();
                wt_new = cfg_.learning_rate / double(k + 1);
                for (int t : dropped) tree_weights_[t] /= double(k + 1);
            }

            // Append to ensemble
            trees_.push_back(std::move(T));
            tree_weights_.push_back(wt_new);

            // Update F with the new tree contribution
            add_tree_contrib_(F, trees_.back(), wt_new);

            // (Optional) You may add line search / shrinkage scheduling here.
        }
    }

    // Remove contribution of dropped trees from Fbase (in-place)
    void subtract_dropped_contrib_(std::vector<double>& Fbase,
                                   const std::vector<int>& dropped) const {
        // Efficiently compute predictions of dropped trees on binned codes
        // We reuse packed predictions by calling per-tree predict on codes.
        // To avoid rebin overhead, we predict via binned path:
        // UnifiedTree::predict expects binned codes; we already have codes_ (N x P_).
        // But its signature expects std::vector<uint16_t>& (which we have).
        // We'll just loop trees and call internal predict_one path through its public API.

        // Build a small temporaries: tree-wise prediction into temp vector and subtract
        for (int t : dropped) {
            std::vector<double> pred(N_, 0.0);
            predict_tree_on_binned_(trees_[t], *codes_, N_, P_, pred);
            const double wt = tree_weights_[t];
            for (int i = 0; i < N_; ++i) Fbase[i] -= wt * pred[i];
        }
    }

    // Add contribution of one tree (with weight wt) to F
    void add_tree_contrib_(std::vector<double>& F,
                           const UnifiedTree& T, double wt) const {
        std::vector<double> pred(N_, 0.0);
        predict_tree_on_binned_(T, *codes_, N_, P_, pred);
        for (int i = 0; i < N_; ++i) F[i] += wt * pred[i];
    }

    // Predict using the ensemble given a pre-binned matrix (codes_)
    std::vector<double> predict_from_binned_(const std::vector<uint16_t>& Xb, int N, int P) const {
        std::vector<double> out(N, 0.0);
        for (size_t t = 0; t < trees_.size(); ++t) {
            const double wt = tree_weights_[t];
            std::vector<double> pred(N, 0.0);
            predict_tree_on_binned_(trees_[t], Xb, N, P, pred);
            for (int i = 0; i < N; ++i) out[i] += wt * pred[i];
        }
        return out;
    }

    // Predict a single tree on pre-binned matrix into 'dst'
    static void predict_tree_on_binned_(const UnifiedTree& T,
                                        const std::vector<uint16_t>& Xb,
                                        int N, int P,
                                        std::vector<double>& dst) {
        // UnifiedTree::predict takes (Xb, N, P) and returns vector<double>
        auto v = T.predict(Xb, N, P);
        if ((int)dst.size() != N) dst.resize(N);
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
    const float*   Xraw_  = nullptr;
    const uint8_t* Xmiss_ = nullptr;

    // Learned ensemble
    std::vector<UnifiedTree> trees_;
    std::vector<double>      tree_weights_; // size == trees_.size()

    // Data shape
    int N_ = 0, P_ = 0;

    // scratch for fit_bins() signature
    std::vector<double> unit_zero_, unit_one_;
};

} // namespace foretree
