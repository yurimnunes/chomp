#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <utility>
#include <vector>

#include "gradient_hist_system.hpp"
#include "split_engine.hpp"
#include "split_finder.hpp"

namespace foretree {

// --------------------------- Config & Node -----------------------------------

struct TreeConfig {
    int max_depth = 6;
    int max_leaves = 31;
    int min_samples_split = 10;
    int min_samples_leaf = 5;

    double min_child_weight = 1e-3;
    double lambda_ = 1.0;
    double alpha_ = 0.0;
    double gamma_ = 0.0;
    double max_delta_step = 0.0;

    int n_bins = 256;

    enum class Growth { LeafWise, LevelWise } growth = Growth::LeafWise;

    // Leaf-wise priority tweaks
    double leaf_gain_eps = 0.0;
    bool allow_zero_gain = false;
    double leaf_depth_penalty = 0.0;
    double leaf_hess_boost = 0.0;

    // Feature sampling
    int feature_bagging_k = -1;
    bool feature_bagging_with_replacement = false;
    int colsample_bytree_percent = 100;
    int colsample_bylevel_percent = 100;
    int colsample_bynode_percent = 100;

    bool use_sibling_subtract = true;

    enum class MissingPolicy { Learn, AlwaysLeft, AlwaysRight };
    MissingPolicy missing_policy = MissingPolicy::Learn;

    std::vector<int8_t> monotone_constraints;

    enum class SplitMode { Histogram, Exact, Hybrid };
    SplitMode split_mode = SplitMode::Histogram;
    int exact_cutover = 2048;

    // Row subsampling
    double subsample_bytree = 1.0;
    double subsample_bylevel = 1.0;
    double subsample_bynode = 1.0;
    bool subsample_with_replacement = true;
    bool subsample_importance_scale = false;

    // GOSS configuration
    struct GOSS {
        bool enabled = false;
        double top_rate = 0.2;       // a
        double other_rate = 0.1;     // b
        bool scale_hessian = true;   // <- default true: H scales like weights
        int min_node_size = 2000;
    } goss;

    // Histogram management
    bool cache_histograms = true;
    bool presort_goss = true;
    int cache_threshold = 1000;

    // --- On-tree (pre) pruning ------------------------------------------------
    struct OnTree {
        bool  enabled = false;

        // Cost-complexity alpha applied during growth: accept split only if
        // (gain - ccp_alpha*(Δleaves)) > 0. For a binary split, Δleaves = +1.
        double ccp_alpha = 0.0;

        // Extra safety thresholds (applied after ccp_alpha):
        double min_gain = 0.0;                 // absolute minimum gain
        double min_gain_rel = 0.0;             // relative to |parent leaf objective|
        double min_impurity_decrease = 0.0;    // alias for API parity

        double eps = 1e-12;                    // numerical guard
    } on_tree;
};

struct Node {
    int id = -1;
    bool is_leaf = true;
    int depth = 0;

    int feature = -1;
    int thr = -1;
    bool miss_left = true;

    int left = -1, right = -1, sibling = -1;

    double G = 0.0, H = 0.0;
    int C = 0;

    int lo = 0, hi = 0;

    double best_gain = -1e300;
    double leaf_value = 0.0;

    // Histogram caching
    mutable std::vector<double> hist_G, hist_H;
    mutable std::vector<int> hist_C;
    mutable std::vector<int> hist_features;
    mutable bool hist_valid = false;
};

struct HistPair {
    std::vector<double> G, H;
    std::vector<int> C;

    void resize(size_t size) {
        G.resize(size);
        H.resize(size);
        C.resize(size);
    }

    void clear() {
        std::fill(G.begin(), G.end(), 0.0);
        std::fill(H.begin(), H.end(), 0.0);
        std::fill(C.begin(), C.end(), 0);
    }

    void subtract(const HistPair &other) {
        for (size_t i = 0; i < G.size(); ++i) {
            G[i] -= other.G[i];
            H[i] -= other.H[i];
            C[i] -= other.C[i];
        }
    }
};

// Memory pool for histogram reuse
class HistogramPool {
private:
    std::vector<std::unique_ptr<HistPair>> pool_;
    std::queue<size_t> available_;
    size_t hist_size_;

public:
    explicit HistogramPool(size_t size) : hist_size_(size) {}

    std::unique_ptr<HistPair> get() {
        if (available_.empty()) {
            auto hist = std::make_unique<HistPair>();
            hist->resize(hist_size_);
            return hist;
        }

        size_t idx = available_.front();
        available_.pop();
        auto hist = std::move(pool_[idx]);
        hist->clear();
        return hist;
    }

    void return_histogram(std::unique_ptr<HistPair> hist) {
        if (hist && hist->G.size() == hist_size_) {
            size_t idx = pool_.size();
            pool_.push_back(std::move(hist));
            available_.push(idx);
        }
    }

    void clear() {
        pool_.clear();
        while (!available_.empty())
            available_.pop();
    }
};

// ------------------------------- UnifiedTree ---------------------------------

class UnifiedTree {
public:
    explicit UnifiedTree(TreeConfig cfg = {},
                         const GradientHistogramSystem *ghs = nullptr)
        : cfg_(std::move(cfg)), ghs_(ghs), P_(0) {}

    void set_raw_matrix(const float *Xraw, const uint8_t *miss_mask_or_null) {
        Xraw_ = Xraw;
        Xmiss_ = miss_mask_or_null;
    }

    void fit(const std::vector<uint16_t> &Xb, int N, int P,
             const std::vector<float> &g, const std::vector<float> &h) {
        std::vector<int> all(N);
        std::iota(all.begin(), all.end(), 0);
        fit_with_row_ids(Xb, N, P, g, h, all);
    }

    void fit_with_row_ids(const std::vector<uint16_t> &Xb, int N, int P,
                          const std::vector<float> &g,
                          const std::vector<float> &h,
                          const std::vector<int> &root_rows) {
        Xb_ = &Xb;
        g_ = &g;
        h_ = &h;
        N_ = N;
        P_ = P;

        initialize_bin_info_();
        reset_();

        nodes_.reserve(std::max(2 * cfg_.max_leaves + 5, 64));
        id2pos_.reserve(std::max(2 * cfg_.max_leaves + 5, 64));
        build_feature_pool_();

        std::vector<int> seed = root_rows;
        apply_tree_level_row_subsample_(seed);
        index_pool_ = std::move(seed);

        initialize_caching_();

        Node r;
        r.id = next_id_++;
        r.depth = 0;
        r.lo = 0;
        r.hi = (int)index_pool_.size();
        accum_(r);
        nodes_.push_back(std::move(r));
        register_pos_(nodes_.back());

        if (cfg_.growth == TreeConfig::Growth::LeafWise)
            grow_leaf_();
        else
            grow_level_();

        // Finalize leaves (use GOSS-weighted totals if GOSS was active for node)
        for (auto &n : nodes_)
            if (n.is_leaf) {
                auto [GG, HH] = node_totals_for_leaf_(n);
                n.leaf_value = leaf_value_(GG, HH);
            }
        pack_();

        cleanup_caching_();
    }

    std::vector<double> predict(const std::vector<uint16_t> &Xb, int N,
                                int P) const {
        std::vector<double> out(N, 0.0);
        (void)P;
        if (!packed_)
            return out;

        for (int i = 0; i < N; ++i) {
            const uint16_t *row = Xb.data() + (size_t)i * (size_t)P_;
            out[i] = predict_one_(row);
        }
        return out;
    }

    const std::vector<double> &feature_importance_gain() const {
        return feat_gain_;
    }
    int n_nodes() const { return (int)nodes_.size(); }
    int n_leaves() const {
        int c = 0;
        for (auto &n : nodes_)
            if (n.is_leaf)
                ++c;
        return c;
    }
    int depth() const {
        int d = 0;
        for (auto &n : nodes_)
            d = std::max(d, n.depth);
        return d;
    }

private:
    // Data & state
    const std::vector<uint16_t> *Xb_ = nullptr;
    const std::vector<float> *g_ = nullptr;
    const std::vector<float> *h_ = nullptr;
    int N_ = 0, P_ = 0;

    TreeConfig cfg_;
    const GradientHistogramSystem *ghs_ = nullptr;

    // Bin information
    std::vector<int> finite_bins_per_feat_;
    std::vector<int> missing_ids_per_feat_;
    std::vector<size_t> feature_offsets_;
    size_t total_hist_size_ = 0;
    int miss_id_ = -1;

    std::vector<Node> nodes_;
    std::vector<int> id2pos_;
    int next_id_ = 0;
    bool packed_ = false;
    int root_id_ = 0;

    std::vector<int> index_pool_;
    std::vector<int> feat_pool_;
    mutable std::mt19937 rng_{1234567u};
    std::vector<double> feat_gain_;

    // Prediction arrays
    std::vector<int> feat_, thr_, left_, right_;
    std::vector<uint8_t> miss_left_, is_leaf_;
    std::vector<double> leaf_val_;

    // Raw matrix for exact splits
    const float *Xraw_ = nullptr;
    const uint8_t *Xmiss_ = nullptr;

    // Caching structures
    std::unique_ptr<HistogramPool> hist_pool_;
    std::unique_ptr<HistPair> tree_histogram_;
    std::vector<int> tree_features_;

    void initialize_bin_info_() {
        if (ghs_ && P_ > 0) {
            finite_bins_per_feat_ = ghs_->all_finite_bins();
            missing_ids_per_feat_.resize(P_);
            feature_offsets_.resize(P_ + 1, 0);

            for (int j = 0; j < P_; ++j) {
                missing_ids_per_feat_[j] = ghs_->total_bins(j) - 1;
                feature_offsets_[j + 1] =
                    feature_offsets_[j] + ghs_->total_bins(j);
            }
            total_hist_size_ = feature_offsets_[P_];

            cfg_.n_bins = ghs_->finite_bins();
            miss_id_ = ghs_->missing_bin_id();
        } else {
            finite_bins_per_feat_.assign(P_, cfg_.n_bins);
            missing_ids_per_feat_.assign(P_, cfg_.n_bins);
            miss_id_ = cfg_.n_bins;

            feature_offsets_.resize(P_ + 1);
            for (int j = 0; j <= P_; ++j) {
                feature_offsets_[j] = j * (cfg_.n_bins + 1);
            }
            total_hist_size_ = P_ * (cfg_.n_bins + 1);
        }
    }

    void initialize_caching_() {
        if (cfg_.cache_histograms) {
            hist_pool_ = std::make_unique<HistogramPool>(total_hist_size_);
        }

        // Build a tree-level histogram only when NOT using GOSS (keeps parent=left+right valid)
        if (cfg_.cache_histograms && !cfg_.goss.enabled &&
            index_pool_.size() >= cfg_.cache_threshold) {
            build_tree_histogram_();
        }
    }

    void cleanup_caching_() {
        if (hist_pool_) {
            hist_pool_->clear();
            hist_pool_.reset();
        }
        tree_histogram_.reset();
        tree_features_.clear();
    }

    void build_tree_histogram_() {
        tree_features_ = feat_pool_;
        if (tree_features_.empty()) {
            tree_features_.resize(P_);
            std::iota(tree_features_.begin(), tree_features_.end(), 0);
        }

        tree_histogram_ = std::make_unique<HistPair>();
        tree_histogram_->resize(total_hist_size_);
        tree_histogram_->clear();

        for (size_t i = 0; i < index_pool_.size(); ++i) {
            const int r = index_pool_[i];
            const uint16_t *row = Xb_->data() + (size_t)r * (size_t)P_;
            const double gr = (double)(*g_)[r];
            const double hr = (double)(*h_)[r];

            for (int f : tree_features_) {
                uint16_t b = row[f];
                if (b >= (uint16_t)missing_ids_per_feat_[f]) {
                    b = (uint16_t)(missing_ids_per_feat_[f]);
                }
                const size_t off = feature_offsets_[f] + (size_t)b;
                if (off < tree_histogram_->G.size()) {
                    tree_histogram_->G[off] += gr;
                    tree_histogram_->H[off] += hr;
                    tree_histogram_->C[off] += 1;
                }
            }
        }
    }

    void reset_() {
        nodes_.clear();
        id2pos_.clear();
        index_pool_.clear();
        next_id_ = 0;
        packed_ = false;
        feat_gain_.assign(P_, 0.0);
        feat_pool_.clear();
    }

    inline void register_pos_(const Node &n) {
        if ((int)id2pos_.size() <= n.id)
            id2pos_.resize(n.id + 1, -1);
        id2pos_[n.id] = (int)nodes_.size() - 1;
    }

    inline Node *by_id_(int id) const {
        if (id < 0 || id >= (int)id2pos_.size())
            return nullptr;
        int pos = id2pos_[id];
        if (pos < 0 || pos >= (int)nodes_.size())
            return nullptr;
        return const_cast<Node *>(&nodes_[pos]);
    }

    void accum_(Node &n) {
        double G = 0.0, H = 0.0;
        for (int i = n.lo; i < n.hi; ++i) {
            int r = index_pool_[i];
            G += (*g_)[r];
            H += (*h_)[r];
        }
        n.G = G;
        n.H = H;
        n.C = n.hi - n.lo;
    }

    double leaf_value_(double G, double H) const {
        double denom = H + cfg_.lambda_;
        if (denom <= 0.0)
            return 0.0;

        double gg = G;
        if (cfg_.alpha_ > 0.0) {
            if (gg > cfg_.alpha_)
                gg -= cfg_.alpha_;
            else if (gg < -cfg_.alpha_)
                gg += cfg_.alpha_;
            else
                gg = 0.0;
        }

        double step = -gg / denom;
        if (cfg_.max_delta_step > 0.0) {
            step = std::clamp(step, -cfg_.max_delta_step, cfg_.max_delta_step);
        }
        return step;
    }

    // NOTE: integrates on-tree pruning into priority used for leaf-wise growth
    double priority_(double gain, const Node &nd) const {
        if (cfg_.on_tree.enabled)
            gain -= cfg_.on_tree.ccp_alpha; // Δleaves = +1 for a binary split

        double pr = gain;
        if (cfg_.leaf_depth_penalty > 0.0)
            pr /= (1.0 + cfg_.leaf_depth_penalty * double(nd.depth));
        if (cfg_.leaf_hess_boost > 0.0)
            pr *= (1.0 + cfg_.leaf_hess_boost * std::max(0.0, nd.H));
        return pr;
    }

    void build_feature_pool_() {
        std::vector<int> all(P_);
        std::iota(all.begin(), all.end(), 0);

        if (cfg_.feature_bagging_k > 0) {
            const int k = std::min(std::max(1, cfg_.feature_bagging_k), P_);
            feat_pool_ =
                sample_k_(all, k, cfg_.feature_bagging_with_replacement);
            return;
        }

        const int pct = std::clamp(cfg_.colsample_bytree_percent, 1, 100);
        if (pct >= 100) {
            feat_pool_ = std::move(all);
            return;
        }

        const int k = std::max(1, P_ * pct / 100);
        feat_pool_ = sample_k_(all, k, false);
    }

    std::vector<int> sample_k_(const std::vector<int> &pool, int k,
                               bool with_replacement) const {
        std::vector<int> out;
        if (k <= 0)
            return out;
        if ((int)pool.size() <= k && !with_replacement)
            return pool;

        out.reserve(k);
        if (with_replacement) {
            std::uniform_int_distribution<int> dist(0, (int)pool.size() - 1);
            for (int i = 0; i < k; ++i)
                out.push_back(pool[dist(rng_)]);
            std::sort(out.begin(), out.end());
            out.erase(std::unique(out.begin(), out.end()), out.end());
        } else {
            std::vector<int> cp = pool;
            std::shuffle(cp.begin(), cp.end(), rng_);
            cp.resize(k);
            std::sort(cp.begin(), cp.end());
            out = std::move(cp);
        }
        return out;
    }

    std::vector<int> select_features_(int depth) const {
        std::vector<int> pool = feat_pool_;
        if (pool.empty()) {
            pool.resize(P_);
            std::iota(pool.begin(), pool.end(), 0);
        }

        const int base = (cfg_.growth == TreeConfig::Growth::LeafWise
                              ? cfg_.colsample_bynode_percent
                              : cfg_.colsample_bylevel_percent);
        const int pct = std::clamp(base, 1, 100);
        if (pct >= 100)
            return pool;

        const int k = std::max(1, (int)pool.size() * pct / 100);
        return sample_k_(pool, k, false);
    }

    void apply_tree_level_row_subsample_(std::vector<int> &rows) {
        const double rate = cfg_.subsample_bytree;
        if (rate >= 1.0 || rows.empty())
            return;

        std::vector<int> out;
        out.reserve((size_t)std::ceil(rate * rows.size()));
        std::uniform_real_distribution<double> U(0.0, 1.0);

        if (!cfg_.subsample_with_replacement) {
            for (int r : rows)
                if (U(rng_) < rate)
                    out.push_back(r);
        } else {
            const int k = std::max(1, (int)std::round(rate * rows.size()));
            std::uniform_int_distribution<int> J(0, (int)rows.size() - 1);
            for (int i = 0; i < k; ++i)
                out.push_back(rows[J(rng_)]);
            std::sort(out.begin(), out.end());
            out.erase(std::unique(out.begin(), out.end()), out.end());
        }

        if (!out.empty())
            rows.swap(out);
    }

    SplitHyper make_hyper_() const {
        SplitHyper hyp;
        hyp.lambda_ = cfg_.lambda_;
        hyp.alpha_ = cfg_.alpha_;
        hyp.gamma_ = cfg_.gamma_;
        hyp.min_samples_leaf = cfg_.min_samples_leaf;
        hyp.min_child_weight = cfg_.min_child_weight;

        switch (cfg_.missing_policy) {
        case TreeConfig::MissingPolicy::Learn:
            hyp.missing_policy = 0;
            break;
        case TreeConfig::MissingPolicy::AlwaysLeft:
            hyp.missing_policy = 1;
            break;
        case TreeConfig::MissingPolicy::AlwaysRight:
            hyp.missing_policy = 2;
            break;
        }

        hyp.leaf_gain_eps = cfg_.leaf_gain_eps;
        hyp.allow_zero_gain = cfg_.allow_zero_gain;
        return hyp;
    }

    const std::vector<int8_t> *maybe_monotone_() const {
        if (cfg_.monotone_constraints.size() == (size_t)P_)
            return &cfg_.monotone_constraints;
        return nullptr;
    }

    bool use_exact_for_(const Node &nd) const {
        if (!Xraw_)
            return false;
        if (cfg_.split_mode == TreeConfig::SplitMode::Exact)
            return true;
        if (cfg_.split_mode == TreeConfig::SplitMode::Hybrid)
            return nd.C <= cfg_.exact_cutover;
        return false;
    }

    // Helper: does this node use GOSS?
    bool node_uses_goss_(const Node& nd) const {
        return cfg_.goss.enabled && (nd.hi - nd.lo) >= cfg_.goss.min_node_size;
    }

    // Histogram Provider with efficient caching
    struct HistogramProvider {
        const UnifiedTree &T;
        const std::vector<int> &index_pool;

        explicit HistogramProvider(const UnifiedTree &tree,
                                   const std::vector<int> &arena)
            : T(tree), index_pool(arena) {}

        std::unique_ptr<HistPair>
        build_histogram(const Node &nd, const std::vector<int> &feats) const {
            // If we have a valid per-node cache for these features, use it
            if (nd.hist_valid && nd.hist_features == feats) {
                auto hist = T.hist_pool_ ? T.hist_pool_->get()
                                         : std::make_unique<HistPair>();
                hist->resize(T.total_hist_size_);
                hist->G = nd.hist_G;
                hist->H = nd.hist_H;
                hist->C = nd.hist_C;
                return hist;
            }

            auto hist = T.hist_pool_ ? T.hist_pool_->get()
                                     : std::make_unique<HistPair>();
            hist->resize(T.total_hist_size_);
            hist->clear();

            const bool goss_here = T.node_uses_goss_(nd);

            // Avoid deriving from tree-level histogram when GOSS is active
            if (!goss_here && T.tree_histogram_ && feats == T.tree_features_ &&
                nd.C >= T.cfg_.cache_threshold) {
                derive_from_tree_histogram_(nd, feats, *hist);
            } else {
                build_from_rows_(nd, feats, *hist, goss_here);
            }

            // Cache the result on the node (safe regardless of GOSS)
            if (T.cfg_.cache_histograms) {
                nd.hist_G = hist->G;
                nd.hist_H = hist->H;
                nd.hist_C = hist->C;
                nd.hist_features = feats;
                nd.hist_valid = true;
            }

            return hist;
        }

    private:
        void derive_from_tree_histogram_(const Node &nd,
                                         const std::vector<int> &feats,
                                         HistPair &hist) const {
            // NOTE: This simplified version falls back to direct build for correctness.
            build_from_rows_(nd, feats, hist, /*use_goss=*/false);
        }

        void build_from_rows_(const Node &nd, const std::vector<int> &feats,
                              HistPair &hist, bool use_goss) const {
            std::vector<int> work_rows = subsample_for_node_(nd);
            if (use_goss) {
                build_with_goss_(nd, feats, hist);
            } else if (!work_rows.empty()) {
                build_for_rows_(work_rows, feats, hist);
            } else {
                build_for_range_(nd.lo, nd.hi, feats, hist);
            }
        }

        std::vector<int> subsample_for_node_(const Node &nd) const {
            double rate = T.cfg_.subsample_bynode;
            if (T.cfg_.growth == TreeConfig::Growth::LevelWise &&
                T.cfg_.subsample_bylevel < 1.0) {
                rate = T.cfg_.subsample_bylevel;
            }
            if (rate >= 1.0)
                return {};

            std::vector<int> rows;
            rows.reserve((size_t)(nd.hi - nd.lo));
            for (int i = nd.lo; i < nd.hi; ++i)
                rows.push_back(index_pool[i]);

            std::vector<int> out;
            std::uniform_real_distribution<double> U(0.0, 1.0);
            for (int r : rows)
                if (U(T.rng_) < rate)
                    out.push_back(r);
            return out;
        }

        void build_with_goss_(const Node &nd, const std::vector<int> &feats,
                              HistPair &hist) const {
            const double a = std::clamp(T.cfg_.goss.top_rate,   0.01, 1.0);
            const double b = std::clamp(T.cfg_.goss.other_rate, 0.0,  1.0);
            const int total = nd.hi - nd.lo;

            if (total < T.cfg_.goss.min_node_size) {
                build_for_range_(nd.lo, nd.hi, feats, hist);
                return;
            }

            // Build (|g|, row) for this node
            std::vector<std::pair<double,int>> node_samples;
            node_samples.reserve(total);
            for (int i = nd.lo; i < nd.hi; ++i) {
                const int r = index_pool[i];
                node_samples.emplace_back(std::abs((*T.g_)[r]), r);
            }

            // Sort descending by |g|
            std::sort(node_samples.begin(), node_samples.end(),
                      [](const auto &A, const auto &B){ return A.first > B.first; });

            // Select top and rest deterministically
            int k_top  = std::max(1, (int)std::round(a * total));
            int k_rest = std::max(0, (int)std::round(b * total));
            k_top  = std::min(k_top, total);
            k_rest = std::min(k_rest, total - k_top);

            std::vector<int> top_rows;  top_rows.reserve(k_top);
            std::vector<int> rest_rows; rest_rows.reserve(k_rest);

            for (int i = 0; i < k_top; ++i)
                top_rows.push_back(node_samples[i].second);

            // Deterministic rest: take next k_rest
            for (int i = k_top; i < k_top + k_rest; ++i)
                rest_rows.push_back(node_samples[i].second);

            // Build separate histograms
            HistPair top_hist, rest_hist;
            top_hist.resize(T.total_hist_size_); top_hist.clear();
            rest_hist.resize(T.total_hist_size_); rest_hist.clear();

            build_for_rows_(top_rows,  feats, top_hist);
            if (!rest_rows.empty())
                build_for_rows_(rest_rows, feats, rest_hist);

            // Combine with proper GOSS weighting
            const double rest_scale = (1.0 - a) / std::max(b, 1e-15);

            for (size_t i = 0; i < hist.G.size(); ++i) {
                hist.G[i] = top_hist.G[i] + rest_scale * rest_hist.G[i];
                hist.H[i] = T.cfg_.goss.scale_hessian
                          ? (top_hist.H[i] + rest_scale * rest_hist.H[i])
                          : (top_hist.H[i] + rest_hist.H[i]);
                // counts are unweighted
                hist.C[i] = top_hist.C[i] + rest_hist.C[i];
            }
        }

        void build_for_rows_(const std::vector<int> &rows,
                             const std::vector<int> &feats,
                             HistPair &hist) const {
            for (int r : rows) {
                const uint16_t *row = T.Xb_->data() + (size_t)r * (size_t)T.P_;
                const double gr = (double)(*T.g_)[r];
                const double hr = (double)(*T.h_)[r];

                for (int f : feats) {
                    uint16_t b = row[f];
                    if (b >= (uint16_t)T.missing_ids_per_feat_[f]) {
                        b = (uint16_t)(T.missing_ids_per_feat_[f]);
                    }
                    const size_t off = T.feature_offsets_[f] + (size_t)b;
                    if (off < hist.G.size()) {
                        hist.G[off] += gr;
                        hist.H[off] += hr;
                        hist.C[off] += 1;
                    }
                }
            }
        }

        void build_for_range_(int lo, int hi, const std::vector<int> &feats,
                              HistPair &hist) const {
            for (int i = lo; i < hi; ++i) {
                const int r = index_pool[i];
                const uint16_t *row = T.Xb_->data() + (size_t)r * (size_t)T.P_;
                const double gr = (double)(*T.g_)[r];
                const double hr = (double)(*T.h_)[r];

                for (int f : feats) {
                    uint16_t b = row[f];
                    if (b >= (uint16_t)T.missing_ids_per_feat_[f]) {
                        b = (uint16_t)(T.missing_ids_per_feat_[f]);
                    }
                    const size_t off = T.feature_offsets_[f] + (size_t)b;
                    if (off < hist.G.size()) {
                        hist.G[off] += gr;
                        hist.H[off] += hr;
                        hist.C[off] += 1;
                    }
                }
            }
        }

    public:
        foretree::splitx::Candidate best_split(const Node &nd, const SplitHyper &hyp,
                             const std::vector<int8_t> *mono) {
            auto feats = T.select_features_(nd.depth);
            auto hist = build_histogram(nd, feats);

            // Compute parent totals consistently from histogram (match sampling/weights)
            double Gtot = 0.0, Htot = 0.0;
            if (!feats.empty()) {
                const int f0 = feats[0];
                const size_t off0 = T.feature_offsets_[f0];
                const int n_bins0 = T.finite_bins_per_feat_[f0] + 1;
                for (int b = 0; b < n_bins0; ++b) {
                    Gtot += hist->G[off0 + (size_t)b];
                    Htot += hist->H[off0 + (size_t)b];
                }
            } else {
                Gtot = nd.G; Htot = nd.H;
            }

            foretree::splitx::SplitContext ctx{&hist->G, &hist->H, &hist->C, T.P_, 0,
                             Gtot, Htot, nd.C, mono, hyp};

            ctx.variable_bins = true;
            ctx.feature_offsets = T.feature_offsets_.data();
            ctx.finite_bins_per_feat = T.finite_bins_per_feat_.data();
            ctx.missing_ids_per_feat = T.missing_ids_per_feat_.data();

            auto result = Splitter::best_split(ctx, SplitEngine::Histogram);

            if (T.hist_pool_) {
                T.hist_pool_->return_histogram(std::move(hist));
            }

            return result;
        }
    };

    // Exact provider for raw matrix splits
    struct ExactProvider {
        const UnifiedTree &T;
        const std::vector<int> &index_pool;

        explicit ExactProvider(const UnifiedTree &tree,
                               const std::vector<int> &arena)
            : T(tree), index_pool(arena) {}

        void build_missing_aggregates_(const Node &nd,
                                       std::vector<double> &Gmiss,
                                       std::vector<double> &Hmiss,
                                       std::vector<int> &Cmiss) const {
            Gmiss.assign(T.P_, 0.0);
            Hmiss.assign(T.P_, 0.0);
            Cmiss.assign(T.P_, 0);

            if (!T.Xraw_)
                return;

            const bool has_mask = (T.Xmiss_ != nullptr);
            for (int i = nd.lo; i < nd.hi; ++i) {
                const int r = index_pool[i];
                const size_t row = (size_t)r * (size_t)T.P_;
                for (int f = 0; f < T.P_; ++f) {
                    bool miss = has_mask
                                    ? (T.Xmiss_[row + (size_t)f] != 0)
                                    : !std::isfinite(T.Xraw_[row + (size_t)f]);
                    if (miss) {
                        Gmiss[f] += (double)(*T.g_)[r];
                        Hmiss[f] += (double)(*T.h_)[r];
                        Cmiss[f] += 1;
                    }
                }
            }
        }

        foretree::splitx::Candidate best_split(const Node &nd, const SplitHyper &hyp,
                             const std::vector<int8_t> *mono) {
            foretree::splitx::SplitContext ctx{nullptr, nullptr, nullptr, T.P_, 0,
                             nd.G,    nd.H,    nd.C,    mono, hyp};
            ctx.row_g = T.g_->data();
            ctx.row_h = T.h_->data();

            std::vector<double> Gmiss, Hmiss;
            std::vector<int> Cmiss;
            build_missing_aggregates_(nd, Gmiss, Hmiss, Cmiss);
            ctx.has_missing = true;
            ctx.Gmiss = Gmiss.data();
            ctx.Hmiss = Hmiss.data();
            ctx.Cmiss = Cmiss.data();

            std::vector<int> rows;
            rows.reserve((size_t)(nd.hi - nd.lo));
            for (int i = nd.lo; i < nd.hi; ++i) {
                rows.push_back(index_pool[i]);
            }

            return Splitter::best_split(ctx, SplitEngine::Exact, T.Xraw_, T.P_,
                                        rows.data(), (int)rows.size(),
                                        hyp.missing_policy, T.Xmiss_);
        }
    };

    // Core evaluation logic
    template <class Provider>
    bool eval_with_provider_(Provider &prov, Node &nd, foretree::splitx::Candidate &out) const {
        if (nd.C < cfg_.min_samples_split || nd.depth >= cfg_.max_depth)
            return false;

        const auto *mono = maybe_monotone_();
        const SplitHyper hyp = make_hyper_();

        foretree::splitx::Candidate cand = prov.best_split(nd, hyp, mono);
        if (cand.thr < 0)
            return false;

        out = cand;
        return true;
    }

    bool eval_node_split_(Node &nd, foretree::splitx::Candidate &out) const {
        const bool exact = use_exact_for_(nd);
        if (exact) {
            ExactProvider prov(*this, index_pool_);
            return eval_with_provider_(prov, nd, out);
        } else {
            HistogramProvider prov(*this, index_pool_);
            return eval_with_provider_(prov, nd, out);
        }
    }

    // Partitioning with per-feature missing IDs
    int partition_hist_(Node &nd, int feat, int thr, bool miss_left) {
        const uint16_t miss = (uint16_t)missing_ids_per_feat_[feat];
        int i = nd.lo, j = nd.hi - 1;

        while (i <= j) {
            int ri = index_pool_[i];
            uint16_t bi = (*Xb_)[(size_t)ri * (size_t)P_ + (size_t)feat];
            bool go_left_i = (bi == miss) ? miss_left : (bi <= (uint16_t)thr);
            if (go_left_i) {
                ++i;
                continue;
            }

            int rj = index_pool_[j];
            uint16_t bj = (*Xb_)[(size_t)rj * (size_t)P_ + (size_t)feat];
            bool go_left_j = (bj == miss) ? miss_left : (bj <= (uint16_t)thr);
            if (!go_left_j) {
                --j;
                continue;
            }

            std::swap(index_pool_[i++], index_pool_[j--]);
        }
        return i;
    }

    int partition_exact_(Node &nd, int feat, double split_value,
                         bool miss_left) {
        const size_t stride = (size_t)P_;
        const size_t off = (size_t)feat;
        int i = nd.lo, j = nd.hi - 1;

        while (i <= j) {
            int ri = index_pool_[i];
            float vi = Xraw_[(size_t)ri * stride + off];
            bool miss_i = !(std::isfinite(vi));
            bool go_left_i = miss_i ? miss_left : (double(vi) <= split_value);
            if (go_left_i) {
                ++i;
                continue;
            }

            int rj = index_pool_[j];
            float vj = Xraw_[(size_t)rj * stride + off];
            bool miss_j = !(std::isfinite(vj));
            bool go_left_j = miss_j ? miss_left : (double(vj) <= split_value);
            if (!go_left_j) {
                --j;
                continue;
            }

            std::swap(index_pool_[i++], index_pool_[j--]);
        }
        return i;
    }

    void apply_split_(Node &nd, const foretree::splitx::Candidate &sp) {
        int mid;
        if (std::isfinite(sp.split_value) && Xraw_) {
            mid = partition_exact_(nd, sp.feat, sp.split_value, sp.miss_left);
        } else {
            mid = partition_hist_(nd, sp.feat, sp.thr, sp.miss_left);
        }

        Node ln, rn;
        ln.id = next_id_++;
        rn.id = next_id_++;
        ln.depth = nd.depth + 1;
        rn.depth = nd.depth + 1;
        ln.lo = nd.lo;
        ln.hi = mid;
        rn.lo = mid;
        rn.hi = nd.hi;

        accum_(ln);
        accum_(rn);

        nd.is_leaf = false;
        nd.feature = sp.feat;
        nd.thr = sp.thr;
        nd.miss_left = sp.miss_left;
        nd.left = ln.id;
        nd.right = rn.id;
        nd.best_gain = sp.gain;
        ln.sibling = rn.id;
        rn.sibling = ln.id;

        if (sp.feat >= 0 && sp.feat < (int)feat_gain_.size() &&
            std::isfinite(sp.gain))
            feat_gain_[sp.feat] += sp.gain;

        // Sibling subtraction optimization (skip on GOSS nodes)
        if (cfg_.use_sibling_subtract && !node_uses_goss_(nd)) {
            auto features = select_features_(nd.depth);
            HistogramProvider prov(*this, index_pool_);

            // Build parent histogram
            auto parent_hist = prov.build_histogram(nd, features);

            // Choose smaller child to build, derive larger by subtraction
            Node &smaller = (ln.C <= rn.C) ? ln : rn;
            Node &larger = (ln.C <= rn.C) ? rn : ln;

            auto small_hist = prov.build_histogram(smaller, features);

            // Cache larger child histogram by subtraction
            if (cfg_.cache_histograms) {
                larger.hist_G = parent_hist->G;
                larger.hist_H = parent_hist->H;
                larger.hist_C = parent_hist->C;

                for (size_t i = 0; i < larger.hist_G.size(); ++i) {
                    larger.hist_G[i] -= small_hist->G[i];
                    larger.hist_H[i] -= small_hist->H[i];
                    larger.hist_C[i] -= small_hist->C[i];
                }
                larger.hist_features = features;
                larger.hist_valid = true;
            }

            if (hist_pool_) {
                hist_pool_->return_histogram(std::move(parent_hist));
                hist_pool_->return_histogram(std::move(small_hist));
            }
        }

        nodes_.push_back(std::move(ln));
        register_pos_(nodes_.back());
        nodes_.push_back(std::move(rn));
        register_pos_(nodes_.back());
    }

    // --- On-tree acceptance ---------------------------------------------------
    inline double parent_leaf_objective_(const Node& nd) const {
        return leaf_objective_optimal_(nd.G, nd.H);
    }

    inline bool accept_split_(const Node& nd, const foretree::splitx::Candidate& sp) const {
        if (!cfg_.on_tree.enabled) return true;

        double g = sp.gain;

        // Cost-complexity penalty: binary split creates +1 new leaf
        if (cfg_.on_tree.ccp_alpha > 0.0)
            g -= cfg_.on_tree.ccp_alpha;

        // Absolute minimum gain (honor alias)
        const double min_abs = std::max(cfg_.on_tree.min_gain,
                                        cfg_.on_tree.min_impurity_decrease);
        if (min_abs > 0.0 && g < (min_abs - cfg_.on_tree.eps))
            return false;

        // Relative threshold
        if (cfg_.on_tree.min_gain_rel > 0.0) {
            const double base = std::abs(parent_leaf_objective_(nd));
            const double rel_thresh = cfg_.on_tree.min_gain_rel * base;
            if (g < (rel_thresh - cfg_.on_tree.eps))
                return false;
        }

        return (g > cfg_.on_tree.eps);
    }

    // Growth algorithms
    void grow_leaf_() {
        struct QItem {
            double pr;
            int id;
            int uid;
            bool operator<(const QItem &o) const {
                if (pr != o.pr)
                    return pr > o.pr;
                return uid > o.uid;
            }
        };

        int uid = 0;
        std::priority_queue<QItem> heap;

        auto push_node = [&](int id) {
            Node &n = *by_id_(id);
            foretree::splitx::Candidate sp;
            if (!eval_node_split_(n, sp) || !accept_split_(n, sp)) {
                auto [GG, HH] = node_totals_for_leaf_(n);
                n.leaf_value = leaf_value_(GG, HH);
                return;
            }
            double pr = priority_(sp.gain, n);
            heap.push(QItem{-pr, id, uid++});
        };

        push_node(nodes_[0].id);
        int leaves = 1;
        const int max_leaves = std::max(1, cfg_.max_leaves);

        while (!heap.empty() && leaves < max_leaves) {
            auto it = heap.top();
            heap.pop();
            Node &n = *by_id_(it.id);

            foretree::splitx::Candidate sp;
            if (!eval_node_split_(n, sp) || !accept_split_(n, sp)) {
                auto [GG, HH] = node_totals_for_leaf_(n);
                n.leaf_value = leaf_value_(GG, HH);
                continue;
            }

            apply_split_(n, sp);

            const Node *n_after = by_id_(it.id);
            const int left_id = (n_after ? n_after->left : -1);
            const int right_id = (n_after ? n_after->right : -1);

            ++leaves;
            if (left_id >= 0)
                push_node(left_id);
            if (right_id >= 0)
                push_node(right_id);
        }
    }

    void grow_level_() {
        std::vector<int> q;
        q.push_back(nodes_[0].id);
        int current_depth = 0;

        while (!q.empty()) {
            int sz = (int)q.size();
            for (int i = 0; i < sz; ++i) {
                int id = q.front();
                q.erase(q.begin());
                Node &n = *by_id_(id);

                foretree::splitx::Candidate sp;
                if (!eval_node_split_(n, sp) || !accept_split_(n, sp)) {
                    auto [GG, HH] = node_totals_for_leaf_(n);
                    n.leaf_value = leaf_value_(GG, HH);
                    continue;
                }

                apply_split_(n, sp);

                const Node *n_after = by_id_(id);
                if (n_after) {
                    if (n_after->left >= 0)
                        q.push_back(n_after->left);
                    if (n_after->right >= 0)
                        q.push_back(n_after->right);
                }
            }
            current_depth++;
            if (current_depth >= cfg_.max_depth)
                break;
        }
    }

    void pack_() {
        int maxid = -1;
        for (auto &n : nodes_)
            maxid = std::max(maxid, n.id);
        int Nn = maxid + 1;

        feat_.assign(Nn, -1);
        thr_.assign(Nn, -1);
        miss_left_.assign(Nn, 0);
        left_.assign(Nn, -1);
        right_.assign(Nn, -1);
        is_leaf_.assign(Nn, 1);
        leaf_val_.assign(Nn, 0.0);

        for (auto &n : nodes_) {
            if (n.is_leaf) {
                is_leaf_[n.id] = 1;
                leaf_val_[n.id] = n.leaf_value;
            } else {
                is_leaf_[n.id] = 0;
                feat_[n.id] = n.feature;
                thr_[n.id] = n.thr;
                miss_left_[n.id] = n.miss_left ? 1 : 0;
                left_[n.id] = n.left;
                right_[n.id] = n.right;
            }
        }
        root_id_ = nodes_[0].id;
        packed_ = true;
    }

    inline double predict_one_(const uint16_t *row) const {
        int id = root_id_;
        while (id >= 0 && is_leaf_[id] == 0) {
            const int f = feat_[id];
            const int t = thr_[id];
            const bool ml = (miss_left_[id] != 0);
            const uint16_t b = row[f];

            const uint16_t feat_miss = (uint16_t)missing_ids_per_feat_[f];
            const bool is_miss = (b == feat_miss);

            id = is_miss ? (ml ? left_[id] : right_[id])
                         : (b <= (uint16_t)t ? left_[id] : right_[id]);
        }
        return id >= 0 ? leaf_val_[id] : 0.0;
    }

public:
    // CART-style cost-complexity post-pruning
    void post_prune_ccp(double ccp_alpha) {
        if (nodes_.empty())
            return;

        std::vector<Node *> by_id;
        by_id.reserve(nodes_.size());
        for (auto &n : nodes_) {
            if ((int)by_id.size() <= n.id)
                by_id.resize(n.id + 1, nullptr);
            by_id[n.id] = &n;
        }
        Node *root = by_id[root_id_];
        if (!root)
            return;

        struct Stats {
            int leaves = 0;
            int internal = 0;
            double R_sub = 0.0;
            double R_collapse = 0.0;
            double alpha_star = std::numeric_limits<double>::infinity();
        };
        std::vector<Stats> S(by_id.size());

        std::function<void(Node *)> acc = [&](Node *nd) {
            if (!nd)
                return;
            if (nd->is_leaf) {
                S[nd->id].leaves = 1;
                S[nd->id].internal = 0;
                const double Rleaf = leaf_objective_optimal_(nd->G, nd->H);
                S[nd->id].R_sub = Rleaf;
                S[nd->id].R_collapse = Rleaf;
                S[nd->id].alpha_star = std::numeric_limits<double>::infinity();
                return;
            }
            acc(by_id[nd->left]);
            acc(by_id[nd->right]);
            const auto &L = S[nd->left];
            const auto &R = S[nd->right];
            auto &dst = S[nd->id];
            dst.leaves = L.leaves + R.leaves;
            dst.internal = L.internal + R.internal + 1;
            dst.R_sub = L.R_sub + R.R_sub - cfg_.gamma_;
            dst.R_collapse = leaf_objective_optimal_(nd->G, nd->H);
            const int denom = std::max(dst.leaves - 1, 1);
            dst.alpha_star = (dst.R_collapse - dst.R_sub) / double(denom);
        };
        acc(root);

        std::function<void(Node *)> apply = [&](Node *nd) {
            if (!nd || nd->is_leaf)
                return;
            apply(by_id[nd->left]);
            apply(by_id[nd->right]);
            const auto alpha_star = S[nd->id].alpha_star;
            if (alpha_star <= ccp_alpha) {
                nd->is_leaf = true;
                nd->left = -1;
                nd->right = -1;
                nd->feature = -1;
                nd->thr = -1;
                nd->miss_left = true;
                nd->best_gain = -std::numeric_limits<double>::infinity();
                nd->leaf_value = leaf_value_(nd->G, nd->H);
            }
        };
        apply(root);
        pack_();
    }

    inline double leaf_objective_optimal_(double G, double H) const {
        double gg = G;
        if (cfg_.alpha_ > 0.0) {
            if (gg > cfg_.alpha_)
                gg -= cfg_.alpha_;
            else if (gg < -cfg_.alpha_)
                gg += cfg_.alpha_;
            else
                gg = 0.0;
        }
        const double denom = H + cfg_.lambda_;
        if (!(denom > 0.0))
            return 0.0;
        double step = -gg / denom;
        if (cfg_.max_delta_step > 0.0)
            step = std::clamp(step, -cfg_.max_delta_step, cfg_.max_delta_step);
        const double w = step;
        return 0.5 * denom * w * w + w * G + std::abs(w) * cfg_.alpha_;
    }

private:
    // Compute (G,H) for leaves consistent with GOSS if active for the node.
    std::pair<double,double> node_totals_for_leaf_(const Node& nd) const {
        if (!node_uses_goss_(nd)) return {nd.G, nd.H};

        const double a = std::clamp(cfg_.goss.top_rate,   0.01, 1.0);
        const double b = std::clamp(cfg_.goss.other_rate, 0.0,  1.0);
        const int total = nd.hi - nd.lo;

        int k_top  = std::max(1, (int)std::round(a * total));
        int k_rest = std::max(0, (int)std::round(b * total));
        k_top  = std::min(k_top, total);
        k_rest = std::min(k_rest, total - k_top);

        std::vector<std::pair<double,int>> v;
        v.reserve(total);
        for (int i = nd.lo; i < nd.hi; ++i) {
            const int r = index_pool_[i];
            v.emplace_back(std::abs((*g_)[r]), r);
        }
        std::sort(v.begin(), v.end(), [](auto &A, auto &B){ return A.first > B.first; });

        double Gt=0.0, Ht=0.0, Gr=0.0, Hr=0.0;
        for (int i = 0; i < k_top; ++i) {
            const int r = v[i].second; Gt += (*g_)[r]; Ht += (*h_)[r];
        }
        for (int i = k_top; i < k_top + k_rest; ++i) {
            const int r = v[i].second; Gr += (*g_)[r]; Hr += (*h_)[r];
        }

        const double rest_scale = (1.0 - a) / std::max(b, 1e-15);
        const double GG = Gt + rest_scale * Gr;
        const double HH = cfg_.goss.scale_hessian ? (Ht + rest_scale * Hr)
                                                  : (Ht + Hr);
        return {GG, HH};
    }
};

} // namespace foretree
