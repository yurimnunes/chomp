#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>      // std::abs, std::isfinite
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <utility>
#include <vector>

#include "gradient_hist_system.hpp" // GradientHistogramSystem
#include "split_engine.hpp"         // Splitter, SplitEngine
#include "split_finder.hpp"         // Candidate, SplitContext, SplitHyper

// -----------------------------------------------------------------------------
// Unified, cleaner tree with provider/policy-based split evaluation.
// - Eliminates repetition across histogram/exact paths
// - Encapsulates GOSS, feature masking, sibling-subtract inside HistogramProvider
// - Encapsulates missing-aggregate logic inside ExactProvider
// - Single templated core: eval_with_provider_()
// -----------------------------------------------------------------------------

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

    // IMPORTANT: n_bins is the number of FINITE bins (missing is mapped to n_bins)
    int n_bins = 256;

    enum class Growth { LeafWise, LevelWise } growth = Growth::LeafWise;

    // Leaf-wise priority tweaks
    double leaf_gain_eps = 0.0;
    bool allow_zero_gain = false;
    double leaf_depth_penalty = 0.0;
    double leaf_hess_boost = 0.0;

    // ---------------- Feature sampling (bagging & per-level/node sampling)
    int  feature_bagging_k = -1; // if >0, choose exactly K features per tree
    bool feature_bagging_with_replacement = false;
    int  colsample_bytree_percent = 100;
    int  colsample_bylevel_percent = 100;
    int  colsample_bynode_percent  = 100;

    bool use_sibling_subtract = true;

    // missing policy for axis splits
    enum class MissingPolicy { Learn, AlwaysLeft, AlwaysRight };
    MissingPolicy missing_policy = MissingPolicy::Learn;

    // monotone constraints (size P; -1/0/+1). May be empty -> none
    std::vector<int8_t> monotone_constraints;

    // split engine selection
    enum class SplitMode { Histogram, Exact, Hybrid };
    SplitMode split_mode = SplitMode::Histogram;
    int exact_cutover = 2048; // try exact when node.C <= exact_cutover

    // --------------------------- GOSS configuration --------------------------
    struct GOSS {
        bool   enabled = false;     // turn on GOSS
        double top_rate = 0.2;      // keep top a fraction by |g|
        double other_rate = 0.1;    // sample b fraction from the rest
        bool   scale_hessian = false; // LightGBM default: false
        int    min_node_size = 2000;  // only apply if node.C >= this
    } goss;
};

struct Node {
    int  id = -1;
    bool is_leaf = true;
    int  depth = 0;

    int  feature = -1; // chosen feature
    int  thr = -1;     // threshold bin id (<= thr goes left); for exact, rank
    bool miss_left = true;

    int left = -1, right = -1, sibling = -1;

    double G = 0.0, H = 0.0;
    int    C = 0;            // sample count at node
    std::vector<int> idx;    // training indices at node

    double best_gain = -1e300;
    double leaf_value = 0.0;

    // cached parent histogram (for sibling subtract)
    std::vector<double> parentG, parentH;
    std::vector<int>    parentC;
    bool has_parent = false;

    // node's own histogram (enables sibling-subtract for siblings)
    std::vector<double> selfG, selfH;
    std::vector<int>    selfC;
    bool has_self = false;
};

// -------------------------- Minimal histogram shims --------------------------
struct HistPair {
    std::vector<double> G, H;
    std::vector<int>    C;
};

// ------------------------------- UnifiedTree ---------------------------------

class UnifiedTree {
public:
    explicit UnifiedTree(TreeConfig cfg = {}, const GradientHistogramSystem* ghs = nullptr)
    : cfg_(std::move(cfg)), ghs_(ghs) {
        if (ghs_) {
            cfg_.n_bins = ghs_->finite_bins();   // finite capacity (per mode)
            miss_id_    = ghs_->missing_bin_id();// reserved missing id == finite_bins
        } else {
            miss_id_    = cfg_.n_bins;           // if no GHS, missing := n_bins (finite)
        }
    }

    // Optional: raw matrix for exact splitting
    void set_raw_matrix(const float* Xraw, const uint8_t* miss_mask_or_null) {
        Xraw_  = Xraw;
        Xmiss_ = miss_mask_or_null;
    }

    // Xb: prebinned matrix (row-major), codes in [0..n_bins] where n_bins == missing id
    void fit(const std::vector<uint16_t>& Xb, int N, int P,
             const std::vector<float>& g, const std::vector<float>& h) {
        Xb_ = &Xb; g_ = &g; h_ = &h; N_ = N; P_ = P;
        reset_();
        nodes_.reserve(std::max(2 * cfg_.max_leaves + 5, 64));
        id2pos_.reserve(std::max(2 * cfg_.max_leaves + 5, 64));
        build_feature_pool_();

        Node r;
        r.id = next_id_++;
        r.depth = 0;
        r.idx.resize(N);
        std::iota(r.idx.begin(), r.idx.end(), 0);
        accum_(r);
        nodes_.push_back(std::move(r));
        register_pos_(nodes_.back());

        if (cfg_.growth == TreeConfig::Growth::LeafWise) grow_leaf_();
        else                                             grow_level_();

        // finalize leaves
        for (auto& n : nodes_) if (n.is_leaf) n.leaf_value = leaf_value_(n.G, n.H);
        pack_();
    }

    std::vector<double> predict(const std::vector<uint16_t>& Xb, int N, int P) const {
        std::vector<double> out(N, 0.0);
        (void)P;
        if (!packed_) return out;
        for (int i = 0; i < N; ++i) {
            const uint16_t* row = Xb.data() + (size_t)i * (size_t)P_;
            out[i] = predict_one_(row);
        }
        return out;
    }

    const std::vector<double>& feature_importance_gain() const { return feat_gain_; }
    int n_nodes()  const { return (int)nodes_.size(); }
    int n_leaves() const { int c=0; for (auto& n : nodes_) if (n.is_leaf) ++c; return c; }
    int depth()    const { int d=0; for (auto& n : nodes_) d = std::max(d, n.depth); return d; }

    // CART-style cost-complexity post-pruning
    void post_prune_ccp(double ccp_alpha) {
        if (nodes_.empty()) return;

        std::vector<Node*> by_id; by_id.reserve(nodes_.size());
        for (auto& n : nodes_) {
            if ((int)by_id.size() <= n.id) by_id.resize(n.id + 1, nullptr);
            by_id[n.id] = &n;
        }
        Node* root = by_id[root_id_]; if (!root) return;

        struct Stats {
            int    leaves = 0;
            int    internal = 0;
            double R_sub = 0.0;
            double R_collapse = 0.0;
            double alpha_star = std::numeric_limits<double>::infinity();
        };
        std::vector<Stats> S(by_id.size());

        std::function<void(Node*)> acc = [&](Node* nd){
            if (!nd) return;
            if (nd->is_leaf) {
                S[nd->id].leaves   = 1;
                S[nd->id].internal = 0;
                const double Rleaf = leaf_objective_optimal_(nd->G, nd->H);
                S[nd->id].R_sub = Rleaf;
                S[nd->id].R_collapse = Rleaf;
                S[nd->id].alpha_star = std::numeric_limits<double>::infinity();
                return;
            }
            acc(by_id[nd->left]);
            acc(by_id[nd->right]);
            const auto& L = S[nd->left];
            const auto& R = S[nd->right];
            auto& dst = S[nd->id];
            dst.leaves   = L.leaves + R.leaves;
            dst.internal = L.internal + R.internal + 1;
            dst.R_sub    = L.R_sub + R.R_sub - cfg_.gamma_;
            dst.R_collapse = leaf_objective_optimal_(nd->G, nd->H);
            const int denom = std::max(dst.leaves - 1, 1);
            dst.alpha_star  = (dst.R_collapse - dst.R_sub) / double(denom);
        };
        acc(root);

        std::function<void(Node*)> apply = [&](Node* nd){
            if (!nd || nd->is_leaf) return;
            apply(by_id[nd->left]);
            apply(by_id[nd->right]);
            const auto alpha_star = S[nd->id].alpha_star;
            if (alpha_star <= ccp_alpha) {
                nd->is_leaf   = true;
                nd->left      = -1;
                nd->right     = -1;
                nd->feature   = -1;
                nd->thr       = -1;
                nd->miss_left = true;
                nd->best_gain = -std::numeric_limits<double>::infinity();
                nd->leaf_value = leaf_value_(nd->G, nd->H);
            }
        };
        apply(root);
        pack_();
    }

private:
    // ----------------------------- Data & state --------------------------------
    const std::vector<uint16_t>* Xb_ = nullptr;
    const std::vector<float>*    g_  = nullptr;
    const std::vector<float>*    h_  = nullptr;
    int N_ = 0, P_ = 0;

    TreeConfig cfg_;
    const GradientHistogramSystem* ghs_ = nullptr;
    int miss_id_ = -1;

    std::vector<Node> nodes_;
    std::vector<int>  id2pos_; // id -> position in nodes_
    int  next_id_  = 0;
    bool packed_   = false;
    int  root_id_  = 0;

    std::vector<int> feat_pool_; // by-tree feature subset (bagging)
    mutable std::mt19937 rng_{std::random_device{}()};

    // feature importance (sum of gains per feature)
    std::vector<double> feat_gain_;

    // packed arrays for fast prediction
    std::vector<int>     feat_, thr_, left_, right_;
    std::vector<uint8_t> miss_left_, is_leaf_;
    std::vector<double>  leaf_val_;

    // raw matrix (for exact)
    const float*   Xraw_  = nullptr;    // N_ x P_ row-major, optional
    const uint8_t* Xmiss_ = nullptr;    // optional missing mask (N_*P_)

    // ------------------------------- Utilities ---------------------------------
    void reset_() {
        nodes_.clear(); id2pos_.clear(); next_id_ = 0; packed_ = false;
        feat_gain_.assign(P_, 0.0);
        feat_pool_.clear();
    }

    inline void register_pos_(const Node& n) {
        if ((int)id2pos_.size() <= n.id) id2pos_.resize(n.id + 1, -1);
        id2pos_[n.id] = (int)nodes_.size() - 1;
    }

    inline Node* by_id_(int id) const {
        if (id < 0) return nullptr;
        if (id >= (int)id2pos_.size()) return nullptr;
        int pos = id2pos_[id];
        if (pos < 0 || pos >= (int)nodes_.size()) return nullptr;
        return const_cast<Node*>(&nodes_[pos]);
    }

    void accum_(Node& n) {
        double G = 0.0, H = 0.0;
        for (int r : n.idx) { G += (*g_)[r]; H += (*h_)[r]; }
        n.G = G; n.H = H; n.C = (int)n.idx.size();
    }

    // soft-threshold & leaf value
    double leaf_value_(double G, double H) const {
        double denom = H + cfg_.lambda_;
        if (denom <= 0.0) return 0.0;
        double gg = G;
        if (cfg_.alpha_ > 0.0) {
            if      (gg >  cfg_.alpha_) gg -= cfg_.alpha_;
            else if (gg < -cfg_.alpha_) gg += cfg_.alpha_;
            else gg = 0.0;
        }
        double step = -gg / denom;
        if (cfg_.max_delta_step > 0.0) {
            step = std::clamp(step, -cfg_.max_delta_step, cfg_.max_delta_step);
        }
        return step;
    }
    inline double leaf_objective_optimal_(double G, double H) const {
        double gg = G;
        if (cfg_.alpha_ > 0.0) {
            if      (gg >  cfg_.alpha_) gg -= cfg_.alpha_;
            else if (gg < -cfg_.alpha_) gg += cfg_.alpha_;
            else gg = 0.0;
        }
        const double denom = H + cfg_.lambda_;
        if (!(denom > 0.0)) return 0.0;
        double step = -gg / denom;
        if (cfg_.max_delta_step > 0.0)
            step = std::clamp(step, -cfg_.max_delta_step, cfg_.max_delta_step);
        const double w = step;
        return 0.5 * denom * w * w + w * G + std::abs(w) * cfg_.alpha_;
    }

    double priority_(double gain, const Node& nd) const {
        double pr = gain;
        if (cfg_.leaf_depth_penalty > 0.0)
            pr /= (1.0 + cfg_.leaf_depth_penalty * double(nd.depth));
        if (cfg_.leaf_hess_boost > 0.0)
            pr *= (1.0 + cfg_.leaf_hess_boost * std::max(0.0, nd.H));
        return pr;
    }

    // ------------------------------- Feature bagging ---------------------------
    std::vector<int> sample_k_(const std::vector<int>& pool, int k, bool with_replacement) const {
        std::vector<int> out;
        if (k <= 0) return out;
        if ((int)pool.size() <= k && !with_replacement) return pool;
        out.reserve(k);
        if (with_replacement) {
            std::uniform_int_distribution<int> dist(0, (int)pool.size() - 1);
            for (int i = 0; i < k; ++i) out.push_back(pool[dist(rng_)]);
            std::sort(out.begin(), out.end());
            out.erase(std::unique(out.begin(), out.end()), out.end());
            if ((int)out.size() < k) {
                std::vector<int> cp = pool;
                std::shuffle(cp.begin(), cp.end(), rng_);
                for (int v : cp) { out.push_back(v); if ((int)out.size() == k) break; }
                std::sort(out.begin(), out.end());
                out.erase(std::unique(out.begin(), out.end()), out.end());
            }
            return out;
        }
        std::vector<int> cp = pool;
        std::shuffle(cp.begin(), cp.end(), rng_);
        cp.resize(k);
        std::sort(cp.begin(), cp.end());
        return cp;
    }

    void build_feature_pool_() {
        std::vector<int> all(P_); std::iota(all.begin(), all.end(), 0);
        if (cfg_.feature_bagging_k > 0) {
            const int k = std::min(std::max(1, cfg_.feature_bagging_k), P_);
            feat_pool_ = sample_k_(all, k, cfg_.feature_bagging_with_replacement);
            return;
        }
        const int pct = std::clamp(cfg_.colsample_bytree_percent, 1, 100);
        if (pct >= 100) { feat_pool_ = std::move(all); return; }
        const int k = std::max(1, P_ * pct / 100);
        feat_pool_ = sample_k_(all, k, false);
    }

    std::vector<int> select_features_(int depth) const {
        std::vector<int> pool = feat_pool_;
        if (pool.empty()) { pool.resize(P_); std::iota(pool.begin(), pool.end(), 0); }
        (void)depth;
        const int base = (cfg_.growth == TreeConfig::Growth::LeafWise
                         ? cfg_.colsample_bynode_percent
                         : cfg_.colsample_bylevel_percent);
        const int pct = std::clamp(base, 1, 100);
        if (pct >= 100) return pool;
        const int k = std::max(1, (int)pool.size() * pct / 100);
        return sample_k_(pool, k, false);
    }

    // ------------------------------ Helpers (non-repetitive) -------------------
    SplitHyper make_hyper_() const {
        SplitHyper hyp;
        hyp.lambda_ = cfg_.lambda_;
        hyp.alpha_  = cfg_.alpha_;
        hyp.gamma_  = cfg_.gamma_;
        hyp.min_samples_leaf = cfg_.min_samples_leaf;
        hyp.min_child_weight = cfg_.min_child_weight;
        switch (cfg_.missing_policy) {
            case TreeConfig::MissingPolicy::Learn:       hyp.missing_policy = 0; break;
            case TreeConfig::MissingPolicy::AlwaysLeft:  hyp.missing_policy = 1; break;
            case TreeConfig::MissingPolicy::AlwaysRight: hyp.missing_policy = 2; break;
        }
        return hyp;
    }

    const std::vector<int8_t>* maybe_monotone_() const {
        if (cfg_.monotone_constraints.size() == (size_t)P_) return &cfg_.monotone_constraints;
        return nullptr;
    }

    bool use_exact_for_(const Node& nd) const {
        if (!Xraw_) return false;
        if (cfg_.split_mode == TreeConfig::SplitMode::Exact)  return true;
        if (cfg_.split_mode == TreeConfig::SplitMode::Hybrid) return nd.C <= cfg_.exact_cutover;
        return false;
    }

    // ------------------------------ Providers ---------------------------------
    // Histogram provider encapsulates: sibling-subtract, GOSS, feature masking.
    struct HistogramProvider {
        const UnifiedTree& T;
        int B; // total bins (finite + missing)

        // Scratch buffers
        HistPair HP;
        std::vector<char> feat_keep;

        explicit HistogramProvider(const UnifiedTree& tree)
        : T(tree), B(tree.ghs_ ? tree.ghs_->total_bins() : (tree.cfg_.n_bins + 1)) {}

        // Build histogram for row-index subset
        void build_hist_for_indices_(const std::vector<int>& rows, HistPair& dst) const {
            if (!T.ghs_) throw std::runtime_error("UnifiedTree: GHS not set");
            const size_t K = (size_t)T.P_ * (size_t)B;
            if (rows.empty()) {
                dst.G.assign(K, 0.0); dst.H.assign(K, 0.0); dst.C.assign(K, 0);
                return;
            }
            auto triple = T.ghs_->build_histograms_fast_with_counts(
                T.g_->data(), T.h_->data(), rows.data(), (int)rows.size());
            dst.G = std::move(std::get<0>(triple));
            dst.H = std::move(std::get<1>(triple));
            dst.C = std::move(std::get<2>(triple));
            if (dst.G.size() != K || dst.H.size() != K || dst.C.size() != K) {
                dst.G.assign(K, 0.0); dst.H.assign(K, 0.0); dst.C.assign(K, 0);
            }
        }

        // Node histogram from caches or fallback scan
        void get_node_hist_(const Node& nd, HistPair& dst) const {
            if (!T.ghs_) throw std::runtime_error("UnifiedTree: GHS not set");

            if (nd.has_self) { dst.G = nd.selfG; dst.H = nd.selfH; dst.C = nd.selfC; return; }

            const Node* sib = T.by_id_(nd.sibling);
            if (nd.has_parent && sib && sib->has_self) {
                const auto& PG = nd.parentG; const auto& PH = nd.parentH; const auto& PC = nd.parentC;
                const auto& SG = sib->selfG; const auto& SH = sib->selfH; const auto& SC = sib->selfC;
                const size_t K = PG.size();
                dst.G.resize(K); dst.H.resize(K); dst.C.resize(K);
                for (size_t i = 0; i < K; ++i) {
                    dst.G[i] = PG[i] - SG[i];
                    dst.H[i] = PH[i] - SH[i];
                    dst.C[i] = PC[i] - SC[i];
                }
                return;
            }

            // Fallback: scan node rows
            build_hist_for_indices_(nd.idx, dst);
        }

        // GOSS path (build combined histogram)
        void build_goss_hist_(const Node& nd, HistPair& dst) const {
            // 1) rank by |g|
            std::vector<int> order = nd.idx;
            std::sort(order.begin(), order.end(), [&](int a, int b){
                return std::abs((*T.g_)[a]) > std::abs((*T.g_)[b]);
            });

            const int C = nd.C;
            const double a = std::clamp(T.cfg_.goss.top_rate,   0.01, 1.0);
            const double b0= std::clamp(T.cfg_.goss.other_rate, 0.0,  1.0);

            int k_top = std::max(1, (int)std::round(a * (double)C));
            k_top = std::min(k_top, C);

            std::vector<int> idx_top; idx_top.reserve(k_top);
            idx_top.assign(order.begin(), order.begin() + k_top);

            std::vector<int> rest;
            if (k_top < (int)order.size())
                rest.assign(order.begin() + k_top, order.end());

            const int rest_n = (int)rest.size();
            int k_rest = (rest_n > 0) ? std::max(0, (int)std::round(b0 * (double)C)) : 0;
            k_rest = std::min(k_rest, rest_n);

            if (k_rest > 0 && rest_n > 0) {
                std::random_device rd; std::mt19937 local_rng(rd());
                std::shuffle(rest.begin(), rest.end(), local_rng);
                rest.resize(k_rest);
            } else { rest.clear(); }

            HistPair Htop, Hrest;
            build_hist_for_indices_(idx_top, Htop);
            build_hist_for_indices_(rest,   Hrest);

            // combine with scale s = (1-a)/b
            double b = b0; if (rest.empty()) b = 1.0; // avoid div by 0
            const double s = (1.0 - a) / b;

            const size_t K = Htop.G.size();
            if (Hrest.G.size() != K || Hrest.H.size() != K || Hrest.C.size() != K) {
                // zero hist with correct shape
                HistPair zero; build_hist_for_indices_({}, zero);
                Hrest = std::move(zero);
            }

            dst.G.resize(K); dst.H.resize(K); dst.C.resize(K);
            for (size_t i = 0; i < K; ++i) {
                dst.G[i] = Htop.G[i] + s * Hrest.G[i];
                dst.H[i] = T.cfg_.goss.scale_hessian ? (Htop.H[i] + s * Hrest.H[i])
                                                     : (Htop.H[i] +     Hrest.H[i]);
                dst.C[i] = Htop.C[i] + Hrest.C[i];
                if (!std::isfinite(dst.G[i])) dst.G[i] = 0.0;
                if (!std::isfinite(dst.H[i])) dst.H[i] = 0.0;
                if (dst.C[i] < 0) dst.C[i] = 0;
            }
        }

        // Build final masked histogram for this node
        void prepare_histogram(const Node& nd) {
            const bool use_goss = (T.cfg_.goss.enabled && nd.C >= T.cfg_.goss.min_node_size);
            if (use_goss) build_goss_hist_(nd, HP);
            else          get_node_hist_(nd, HP);

            // Feature sampling -> mask columns
            auto feats = T.select_features_(nd.depth);
            if ((int)feats.size() < T.P_) {
                feat_keep.assign(T.P_, 0);
                for (int f : feats) if (f>=0 && f<T.P_) feat_keep[f] = 1;
                for (int f = 0; f < T.P_; ++f) if (!feat_keep[f]) {
                    const size_t base = (size_t)f * (size_t)B;
                    for (int b = 0; b < B; ++b) {
                        HP.G[base + b] = 0.0;
                        HP.H[base + b] = 0.0;
                        HP.C[base + b] = 0;
                    }
                }
            }
        }

        Candidate best_split(const Node& nd, const SplitHyper& hyp,
                             const std::vector<int8_t>* mono) {
            prepare_histogram(nd);

            SplitContext ctx{
                /*HG*/ &HP.G, /*HH*/ &HP.H, /*HC*/ &HP.C,
                T.P_, B,
                nd.G, nd.H, nd.C,
                mono, hyp
            };
            return Splitter::best_split(ctx, SplitEngine::Histogram);
        }
    };

    // Exact provider encapsulates: per-feature missing aggregates, raw access
    struct ExactProvider {
        const UnifiedTree& T;

        explicit ExactProvider(const UnifiedTree& tree) : T(tree) {}

        void build_missing_aggregates_(const Node& nd,
                                       std::vector<double>& Gmiss,
                                       std::vector<double>& Hmiss,
                                       std::vector<int>&    Cmiss) const {
            Gmiss.assign(T.P_, 0.0); Hmiss.assign(T.P_, 0.0); Cmiss.assign(T.P_, 0);
            if (!T.Xraw_) return;
            const bool has_mask = (T.Xmiss_ != nullptr);
            for (int r : nd.idx) {
                const size_t row = (size_t)r * (size_t)T.P_;
                for (int f = 0; f < T.P_; ++f) {
                    bool miss = has_mask ? (T.Xmiss_[row + (size_t)f] != 0)
                                         : !std::isfinite(T.Xraw_[row + (size_t)f]);
                    if (miss) {
                        Gmiss[f] += (double)(*T.g_)[r];
                        Hmiss[f] += (double)(*T.h_)[r];
                        Cmiss[f] += 1;
                    }
                }
            }
        }

        Candidate best_split(const Node& nd, const SplitHyper& hyp,
                             const std::vector<int8_t>* mono) {
            SplitContext ctx{
                /*HG*/ nullptr, /*HH*/ nullptr, /*HC*/ nullptr,
                T.P_, /*B*/ 0,
                nd.G, nd.H, nd.C,
                mono, hyp
            };
            ctx.row_g = T.g_->data();
            ctx.row_h = T.h_->data();

            std::vector<double> Gmiss, Hmiss;
            std::vector<int>    Cmiss;
            build_missing_aggregates_(nd, Gmiss, Hmiss, Cmiss);
            ctx.has_missing = true;
            ctx.Gmiss = Gmiss.data();
            ctx.Hmiss = Hmiss.data();
            ctx.Cmiss = Cmiss.data();

            // Splitter API for Exact uses raw matrix & mask
            return Splitter::best_split(ctx, SplitEngine::Exact,
                                        T.Xraw_, T.P_,
                                        nd.idx.data(), nd.C,
                                        hyp.missing_policy, T.Xmiss_);
        }
    };

    // --------------------------- Core: templated eval --------------------------
    template <class Provider>
    bool eval_with_provider_(Provider& prov, Node& nd, Candidate& out) const {
        if (nd.C < cfg_.min_samples_split || nd.depth >= cfg_.max_depth)
            return false;
        const auto* mono = maybe_monotone_();
        const SplitHyper hyp = make_hyper_();

        Candidate cand = prov.best_split(nd, hyp, mono);
        if (cand.thr < 0) return false;
        out = cand;
        return true;
    }

    // Dispatcher
    bool eval_node_split_(Node& nd, Candidate& out) const {
        const bool exact = use_exact_for_(nd);
        if (exact) {
            ExactProvider prov(*this);
            return eval_with_provider_(prov, nd, out);
        } else {
            HistogramProvider prov(*this);
            return eval_with_provider_(prov, nd, out);
        }
    }

    // ----------------------------- Apply split --------------------------------
    void apply_split_(Node& nd, const Candidate& sp) {
        const uint16_t miss = (uint16_t)miss_id_;

        std::vector<int> L; L.reserve(nd.C);
        std::vector<int> R; R.reserve(nd.C);

        if (std::isfinite(sp.split_value) && Xraw_) {
            // exact partition by numeric threshold
            for (int r : nd.idx) {
                const float v = Xraw_[(size_t)r * (size_t)P_ + (size_t)sp.feat];
                const bool is_miss = !(std::isfinite(v));
                bool go_left = is_miss ? sp.miss_left : (double(v) <= sp.split_value);
                (go_left ? L : R).push_back(r);
            }
        } else {
            // histogram partition
            for (int r : nd.idx) {
                const uint16_t b = (*Xb_)[(size_t)r * (size_t)P_ + (size_t)sp.feat];
                const bool go_left = (b == miss) ? sp.miss_left : (b <= (uint16_t)sp.thr);
                (go_left ? L : R).push_back(r);
            }
        }

        Node ln, rn;
        ln.id = next_id_++; rn.id = next_id_++;
        ln.depth = nd.depth + 1; rn.depth = nd.depth + 1;
        ln.idx.swap(L); rn.idx.swap(R);
        accum_(ln); accum_(rn);

        nd.is_leaf = false;
        nd.feature = sp.feat;
        nd.thr = sp.thr; // for exact, thr is rank; kept for debugging
        nd.miss_left = sp.miss_left;
        nd.left = ln.id; nd.right = rn.id;
        nd.best_gain = sp.gain;
        ln.sibling = rn.id; rn.sibling = ln.id;

        if (sp.feat >= 0 && sp.feat < (int)feat_gain_.size() && std::isfinite(sp.gain))
            feat_gain_[sp.feat] += sp.gain;

        // Cache parent histogram on children (for sibling-subtract)
        if (cfg_.use_sibling_subtract && ghs_) {
            // Build parent hist once
            HistPair parentHP;
            {
                auto triple = ghs_->build_histograms_fast_with_counts(
                    g_->data(), h_->data(), nd.idx.data(), (int)nd.idx.size());
                parentHP.G = std::move(std::get<0>(triple));
                parentHP.H = std::move(std::get<1>(triple));
                parentHP.C = std::move(std::get<2>(triple));
                const int BB = ghs_ ? ghs_->total_bins() : (cfg_.n_bins + 1);
                const size_t KK = (size_t)P_ * (size_t)BB;
                if (parentHP.G.size() != KK || parentHP.H.size() != KK || parentHP.C.size() != KK) {
                    parentHP.G.assign(KK, 0.0); parentHP.H.assign(KK, 0.0); parentHP.C.assign(KK, 0);
                }
            }
            ln.parentG = parentHP.G; ln.parentH = parentHP.H; ln.parentC = parentHP.C; ln.has_parent = true;
            rn.parentG = parentHP.G; rn.parentH = parentHP.H; rn.parentC = parentHP.C; rn.has_parent = true;

            // Build hist for the smaller child only; derive the other by subtract
            Node& small = (ln.C <= rn.C) ? ln : rn;
            Node& big   = (ln.C <= rn.C) ? rn : ln;

            {
                auto triple = ghs_->build_histograms_fast_with_counts(
                    g_->data(), h_->data(), small.idx.data(), (int)small.idx.size());
                small.selfG = std::move(std::get<0>(triple));
                small.selfH = std::move(std::get<1>(triple));
                small.selfC = std::move(std::get<2>(triple));
                const int BB = ghs_ ? ghs_->total_bins() : (cfg_.n_bins + 1);
                const size_t KK = (size_t)P_ * (size_t)BB;
                if (small.selfG.size() != KK || small.selfH.size() != KK || small.selfC.size() != KK) {
                    small.selfG.assign(KK, 0.0); small.selfH.assign(KK, 0.0); small.selfC.assign(KK, 0);
                }
                small.has_self = true;
            }
            {
                const auto& PG = parentHP.G; const auto& PH = parentHP.H; const auto& PC = parentHP.C;
                const size_t K = PG.size();
                big.selfG.resize(K); big.selfH.resize(K); big.selfC.resize(K);
                for (size_t i = 0; i < K; ++i) {
                    big.selfG[i] = PG[i] - small.selfG[i];
                    big.selfH[i] = PH[i] - small.selfH[i];
                    big.selfC[i] = PC[i] - small.selfC[i];
                }
                big.has_self = true;
            }
        }

        // Push children and register positions
        nodes_.push_back(std::move(ln)); register_pos_(nodes_.back());
        nodes_.push_back(std::move(rn)); register_pos_(nodes_.back());
    }

    // ------------------------------- Growth modes ------------------------------
    void grow_leaf_() {
        struct QItem { double pr; int id; int uid;
            bool operator<(const QItem& o) const {
                if (pr != o.pr) return pr > o.pr; // min-heap by pr via reversed compare
                return uid > o.uid;
            }
        };

        int uid = 0;
        std::priority_queue<QItem> heap;

        auto push_node = [&](int id) {
            Node& n = *by_id_(id);
            Candidate sp;
            if (!eval_node_split_(n, sp)) { n.leaf_value = leaf_value_(n.G, n.H); return; }
            double pr = priority_(sp.gain, n);
            heap.push(QItem{-pr, id, uid++});
        };

        push_node(nodes_[0].id);
        int leaves = 1;
        const int max_leaves = std::max(1, cfg_.max_leaves);

        while (!heap.empty() && leaves < max_leaves) {
            auto it = heap.top(); heap.pop();
            Node& n = *by_id_(it.id);

            Candidate sp;
            if (!eval_node_split_(n, sp)) { n.leaf_value = leaf_value_(n.G, n.H); continue; }

            apply_split_(n, sp);

            // nodes_ may have reallocated; don't touch 'n' anymore.
            const Node* n_after = by_id_(it.id);
            const int left_id  = (n_after ? n_after->left  : -1);
            const int right_id = (n_after ? n_after->right : -1);

            ++leaves;
            if (left_id  >= 0) push_node(left_id);
            if (right_id >= 0) push_node(right_id);
        }
    }

    void grow_level_() {
        std::vector<int> q; q.push_back(nodes_[0].id);
        while (!q.empty()) {
            int id = q.front(); q.erase(q.begin());
            Node& n = *by_id_(id);

            Candidate sp;
            if (!eval_node_split_(n, sp)) { n.leaf_value = leaf_value_(n.G, n.H); continue; }

            apply_split_(n, sp);

            // nodes_ may have reallocated; don't touch 'n' anymore.
            const Node* n_after = by_id_(id);
            if (n_after) {
                if (n_after->left  >= 0) q.push_back(n_after->left);
                if (n_after->right >= 0) q.push_back(n_after->right);
            }
        }
    }

    // ----------------------------- Packing & Predict ---------------------------
    void pack_() {
        int maxid = -1; for (auto& n : nodes_) maxid = std::max(maxid, n.id);
        int Nn = maxid + 1;

        feat_.assign(Nn, -1);
        thr_.assign(Nn, -1);
        miss_left_.assign(Nn, 0);
        left_.assign(Nn, -1);
        right_.assign(Nn, -1);
        is_leaf_.assign(Nn, 1);
        leaf_val_.assign(Nn, 0.0);

        for (auto& n : nodes_) {
            if (n.is_leaf) {
                is_leaf_[n.id] = 1;
                leaf_val_[n.id] = n.leaf_value;
            } else {
                is_leaf_[n.id] = 0;
                feat_[n.id] = n.feature;
                thr_[n.id]  = n.thr;
                miss_left_[n.id] = n.miss_left ? 1 : 0;
                left_[n.id]  = n.left;
                right_[n.id] = n.right;
            }
        }
        root_id_ = nodes_[0].id;
        packed_  = true;
    }

    inline double predict_one_(const uint16_t* row) const {
        int id = root_id_;
        const uint16_t miss = (uint16_t)miss_id_;
        while (id >= 0 && is_leaf_[id] == 0) {
            const int  f  = feat_[id];
            const int  t  = thr_[id];
            const bool ml = (miss_left_[id] != 0);
            const uint16_t b = row[f];
            const bool is_miss = (b == miss);
            id = is_miss ? (ml ? left_[id] : right_[id])
                         : (b <= (uint16_t)t ? left_[id] : right_[id]);
        }
        return id >= 0 ? leaf_val_[id] : 0.0;
    }
};

} // namespace foretree
