#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

/*
  Unified CART for boosting/bagging (binned histogram path)

  Key points:
    - Input is *pre-binned* matrix Xb (row-major, uint16_t) with N rows, P cols.
      Bins in [0, n_bins] where bin==n_bins is the "missing" bin.
    - fit(Xb, N, P, g, h): builds a tree using gradients/hessians (Newton).
    - predict(Xb, N, P): returns double predictions for all rows.
    - Leaf value: soft-thresholded Newton step (L2 "lambda_", optional L1 "alpha_"),
      optional max_delta_step_ clipping (as in XGBoost/LightGBM).
    - Growth policies: leaf-wise (priority queue) and level-wise (queue).
    - Missing handling: per-node learns whether missing goes left or right.
    - Feature importance: gain-based counter.
    - Thread-safe prediction after fit (immutable arrays).

  Notes:
    - This file focuses on the *binned* path (fast, memory-friendly).
    - Hooks are left for exact/oblique/categorical if you want parity with your Python.
*/

namespace foretree {

struct TreeConfig {
  // Topology / stopping
  int   max_depth         = 6;
  int   max_leaves        = 31;
  int   min_samples_split = 10;
  int   min_samples_leaf  = 5;

  // Regularization
  double min_child_weight = 1e-3;   // sum(h) threshold
  double lambda_          = 1.0;    // L2
  double alpha_           = 0.0;    // L1
  double gamma_           = 0.0;    // split penalty
  double max_delta_step   = 0.0;    // optional clipping on leaf value magnitude

  // Binning
  int   n_bins            = 256;    // usable bins: 0..n_bins-1, missing := n_bins
  bool  use_hist_subtract = true;   // sibling histogram subtraction (future)

  // Growth policy: "leaf" or "level"
  enum class Growth { LeafWise, LevelWise } growth = Growth::LeafWise;

  // Optional prioritization tweaks
  double leaf_gain_eps     = 0.0;   // ignore micro-gains
  bool   allow_zero_gain   = false; // accept zero gain splits
  double leaf_depth_penalty = 0.0;  // priority := gain / (1 + alpha*depth)
  double leaf_hess_boost    = 0.0;  // priority := gain*(1 + beta*H_parent)
};

struct Node {
  int id = -1;
  bool is_leaf = true;

  // split info (axis-aligned, binned)
  int feature = -1;            // global feature id
  int threshold_bin = -1;      // go left if bin_id <= threshold_bin
  bool missing_go_left = true; // routing for missing bin

  // structure
  int left = -1;
  int right = -1;
  int depth = 0;

  // sufficient stats at node
  double g_sum = 0.0;
  double h_sum = 0.0;
  int    n_samples = 0;

  // best split metadata
  double best_gain = -std::numeric_limits<double>::infinity();

  // leaf value
  double leaf_value = 0.0;

  // training index slice (only during fit)
  std::vector<int> indices;
};

class UnifiedTree {
public:
  explicit UnifiedTree(TreeConfig cfg = {}) : cfg_(cfg) {}

  // Fit with binned features matrix (row-major), gradients g, hessians h
  // Xb.size() must be N*P. Bins in [0..cfg_.n_bins], where missing_bin_id() == cfg_.n_bins.
  void fit(const std::vector<uint16_t>& Xb, int N, int P,
           const std::vector<float>& g, const std::vector<float>& h) {
    assert((int)g.size() == N && (int)h.size() == N);
    N_ = N; P_ = P;
    Xb_ = &Xb; g_ = &g; h_ = &h;

    reset_state_();

    // Create root
    Node root;
    root.id = next_node_id_++;
    root.depth = 0;
    root.is_leaf = true;
    root.indices.resize(N);
    std::iota(root.indices.begin(), root.indices.end(), 0);
    accum_node_stats_(root);
    nodes_.push_back(std::move(root));

    // Choose growth policy
    if (cfg_.growth == TreeConfig::Growth::LeafWise) {
      grow_leaf_wise_();
    } else {
      grow_level_wise_();
    }

    // finalize leaves w/o leaf_value
    for (auto& nd : nodes_) {
      if (nd.is_leaf) nd.leaf_value = calc_leaf_value_(nd.g_sum, nd.h_sum);
    }

    // Build compressed arrays for fast prediction
    build_pred_arrays_();
  }

  // Predict with binned features (same binning as training)
  std::vector<double> predict(const std::vector<uint16_t>& Xb, int N, int P) const {
    assert(P == P_);
    std::vector<double> out(N, 0.0);
    if (nodes_.empty()) return out;
    if (!packed_valid_) {
      // Fallback slow traversal (after move/serialization you should rebuild)
      for (int i = 0; i < N; ++i) out[i] = predict_one_slow_(Xb.data() + i*P);
      return out;
    }
    const uint16_t* row = nullptr;
    for (int i = 0; i < N; ++i) {
      row = Xb.data() + i*P;
      out[i] = predict_one_packed_(row);
    }
    return out;
  }

  // Feature importance (gain)
  const std::vector<double>& feature_importance_gain() const { return feat_gain_; }

  // Introspection
  int n_nodes() const { return (int)nodes_.size(); }
  int depth()   const {
    int d = 0;
    for (auto& n : nodes_) d = std::max(d, n.depth);
    return d;
  }
  int n_leaves() const {
    int c = 0; for (auto& n : nodes_) if (n.is_leaf) ++c; return c;
  }

  // Missing bin id (reserved)
  int missing_bin_id() const { return cfg_.n_bins; }

private:
  // --- config/state ---
  TreeConfig cfg_;
  int N_ = 0, P_ = 0;
  const std::vector<uint16_t>* Xb_ = nullptr;
  const std::vector<float>*    g_  = nullptr;
  const std::vector<float>*    h_  = nullptr;

  std::vector<Node> nodes_;
  int next_node_id_ = 0;

  // feature importance
  std::vector<double> feat_gain_;

  // packed prediction arrays
  std::vector<int>    p_feature_;
  std::vector<int>    p_thresh_;
  std::vector<uint8_t> p_missing_left_;
  std::vector<int>    p_left_;
  std::vector<int>    p_right_;
  std::vector<uint8_t> p_is_leaf_;
  std::vector<double> p_leaf_value_;
  int root_id_ = 0;
  bool packed_valid_ = false;

private:
  void reset_state_() {
    nodes_.clear();
    next_node_id_ = 0;
    feat_gain_.assign(P_, 0.0);
    packed_valid_ = false;
  }

  // Accumulate g/h and sample count for a node
  void accum_node_stats_(Node& nd) const {
    double G = 0.0, H = 0.0;
    for (int idx : nd.indices) {
      G += (*g_)[idx];
      H += (*h_)[idx];
    }
    nd.g_sum = G; nd.h_sum = H; nd.n_samples = (int)nd.indices.size();
  }

  // Newton leaf value with L1/L2 and optional clipping
  double calc_leaf_value_(double G, double H) const {
    double denom = H + cfg_.lambda_;
    if (denom <= 0.0) return 0.0;

    // L1 soft-threshold on gradient
    double g = G;
    if (cfg_.alpha_ > 0.0) {
      if      (g >  cfg_.alpha_) g -= cfg_.alpha_;
      else if (g < -cfg_.alpha_) g += cfg_.alpha_;
      else                       g  = 0.0;
    }
    double step = -g / denom;
    if (cfg_.max_delta_step > 0.0) {
      if (step >  cfg_.max_delta_step) step =  cfg_.max_delta_step;
      if (step < -cfg_.max_delta_step) step = -cfg_.max_delta_step;
    }
    return step;
  }

  // Gain function for splitting (XGBoost-style)
  static inline double leaf_obj_(double G, double H, double lambda, double alpha, double mds) {
    // value = calc_leaf_value_; objective = -0.5 * G'*inv(H+λ)*G + L1 handled inside
    // We approximate the improvement using the closed form with soft-threshold:
    // score = 0.5 * g^2 / (H + λ) after soft-threshold on g; (no - sign, we compare gains)
    if (H + lambda <= 0.0) return 0.0;
    double g = G;
    if (alpha > 0.0) {
      if      (g >  alpha) g -= alpha;
      else if (g < -alpha) g += alpha;
      else                 g  = 0.0;
    }
    double val = (g*g) / (H + lambda);
    // max_delta_step affects value indirectly via clipping; we ignore here for gain (common practice)
    return 0.5 * val;
  }

  static inline bool finite_pos_gain_(double gain, double eps, bool allow_zero) {
    if (!std::isfinite(gain)) return false;
    return allow_zero ? (gain >= eps) : (gain > std::max(0.0, eps));
  }

  // Compute best axis-aligned binned split for a node
  struct BestSplit {
    int feature = -1;
    int thresh_bin = -1;
    bool missing_left = true;
    double gain = -std::numeric_limits<double>::infinity();
  };

  BestSplit eval_best_split_(const Node& nd) const {
    BestSplit best;
    if (nd.n_samples < cfg_.min_samples_split) return best;

    const int B = cfg_.n_bins + 1; // include missing bin
    std::vector<double> Hg(B, 0.0), Hh(B, 0.0);

    // Per-feature loop
    for (int f = 0; f < P_; ++f) {
      std::fill(Hg.begin(), Hg.end(), 0.0);
      std::fill(Hh.begin(), Hh.end(), 0.0);

      // Build histograms over bins (including missing bin at id==cfg_.n_bins)
      for (int idx : nd.indices) {
        const uint16_t b = (*Xb_)[(size_t)idx * (size_t)P_ + (size_t)f];
        Hg[b] += (*g_)[idx];
        Hh[b] += (*h_)[idx];
      }

      // Prefix sums for 0..B-2 (usable bins), missing handled separately
      const int M = cfg_.n_bins;  // usable bins
      // Try both directions for missing assignment: left or right
      // We'll compute two best thresholds and keep the max.

      auto scan_direction = [&](bool missing_left) -> std::pair<int,double> {
        double G_L = 0.0, H_L = 0.0;
        double G_missing = Hg[M];  // bin n_bins is missing
        double H_missing = Hh[M];

        int best_t = -1;
        double best_gain = -std::numeric_limits<double>::infinity();

        for (int t = 0; t < M; ++t) {
          // include current bin into left
          G_L += Hg[t];
          H_L += Hh[t];

          double G_left = G_L + (missing_left ? G_missing : 0.0);
          double H_left = H_L + (missing_left ? H_missing : 0.0);
          double G_right = (nd.g_sum - G_L) + (missing_left ? 0.0 : G_missing);
          double H_right = (nd.h_sum - H_L) + (missing_left ? 0.0 : H_missing);

          // Guards
          const int n_left_min  = (t+1) * 1; // proxy for count; we don't track counts per bin here
          (void)n_left_min; // not used; we enforce via Hessian & child weight below
          if (H_left  < cfg_.min_child_weight) continue;
          if (H_right < cfg_.min_child_weight) continue;

          // Optional samples guard (cheap approximation): skip if either side likely tiny
          // (You can maintain per-bin counts alongside Hg/Hh if you want exact sample guards.)
          // Gain
          const double gain_left  = leaf_obj_(G_left,  H_left,  cfg_.lambda_, cfg_.alpha_, cfg_.max_delta_step);
          const double gain_right = leaf_obj_(G_right, H_right, cfg_.lambda_, cfg_.alpha_, cfg_.max_delta_step);
          const double gain_parent= leaf_obj_(nd.g_sum, nd.h_sum, cfg_.lambda_, cfg_.alpha_, cfg_.max_delta_step);

          double gain = gain_left + gain_right - gain_parent - cfg_.gamma_;
          if (gain > best_gain) {
            best_gain = gain;
            best_t = t;
          }
        }
        return {best_t, best_gain};
      };

      auto [tL, gL] = scan_direction(true);
      auto [tR, gR] = scan_direction(false);

      bool miss_left_pick = (gL >= gR);
      int t_pick = miss_left_pick ? tL : tR;
      double gain_pick = miss_left_pick ? gL : gR;

      if (t_pick >= 0 && finite_pos_gain_(gain_pick, cfg_.leaf_gain_eps, cfg_.allow_zero_gain)) {
        double priority = compose_priority_(gain_pick, nd);
        // compare by actual gain for tree quality, but we could also store priority if needed
        if (gain_pick > best.gain) {
          best.feature = f;
          best.thresh_bin = t_pick;
          best.missing_left = miss_left_pick;
          best.gain = gain_pick;
        }
      }
    }
    return best;
  }

  double compose_priority_(double gain, const Node& nd) const {
    double pr = gain;
    if (cfg_.leaf_depth_penalty > 0.0) {
      pr = pr / (1.0 + cfg_.leaf_depth_penalty * double(nd.depth));
    }
    if (cfg_.leaf_hess_boost > 0.0) {
      pr = pr * (1.0 + cfg_.leaf_hess_boost * std::max(0.0, nd.h_sum));
    }
    return pr;
  }

  // Apply chosen split (partition indices; create children)
  // Returns -1 if split rejected.
  int apply_split_(int node_id, const BestSplit& sp) {
    if (sp.feature < 0 || sp.thresh_bin < 0) return -1;
    Node& parent = nodes_[node_id];

    // Partition
    std::vector<int> L; L.reserve(parent.indices.size());
    std::vector<int> R; R.reserve(parent.indices.size());

    const int f = sp.feature;
    const uint16_t t = static_cast<uint16_t>(sp.thresh_bin);
    const uint16_t miss = static_cast<uint16_t>(missing_bin_id());

    for (int idx : parent.indices) {
      const uint16_t b = (*Xb_)[(size_t)idx * (size_t)P_ + (size_t)f];
      bool go_left;
      if (b == miss) {
        go_left = sp.missing_left;
      } else {
        go_left = (b <= t);
      }
      (go_left ? L : R).push_back(idx);
    }

    if ((int)L.size() < cfg_.min_samples_leaf || (int)R.size() < cfg_.min_samples_leaf) {
      return -1;
    }

    Node ln, rn;
    ln.id = next_node_id_++; rn.id = next_node_id_++;
    ln.depth = parent.depth + 1; rn.depth = parent.depth + 1;
    ln.indices = std::move(L); rn.indices = std::move(R);
    accum_node_stats_(ln); accum_node_stats_(rn);

    if (ln.h_sum < cfg_.min_child_weight || rn.h_sum < cfg_.min_child_weight) {
      return -1;
    }

    // Commit
    parent.is_leaf = false;
    parent.feature = f;
    parent.threshold_bin = sp.thresh_bin;
    parent.missing_go_left = sp.missing_left;
    parent.left = ln.id;
    parent.right = rn.id;
    parent.best_gain = sp.gain;

    // Update feature importance by actual gain
    if (f >= 0 && f < (int)feat_gain_.size() && std::isfinite(sp.gain)) {
      feat_gain_[f] += sp.gain;
    }

    nodes_.push_back(std::move(ln));
    nodes_.push_back(std::move(rn));
    return parent.left; // any valid child id
  }

  // Leaf-wise growth with priority queue
  void grow_leaf_wise_() {
    struct QItem {
      double pr; // priority (higher better) stored as negative for min-heap
      int node_id;
      // stable tiebreaker
      int uid;
      bool operator<(const QItem& other) const {
        if (pr != other.pr) return pr > other.pr; // min-heap on pr
        return uid > other.uid;
      }
    };

    int uid = 0;
    auto push_if_good = [&](std::priority_queue<QItem>& heap, int nid) {
      Node& nd = nodes_[nid];
      if (nd.depth >= cfg_.max_depth) return;
      if (nd.is_leaf == false) return;
      auto sp = eval_best_split_(nd);
      if (sp.feature < 0) return;
      double pr = compose_priority_(sp.gain, nd);
      heap.push(QItem{ -pr, nid, uid++ });
      // stash provisional split params on node (so we don't recompute twice)
      // Here we recompute at pop to keep this simple; for speed, store on side map.
      (void)sp;
    };

    std::priority_queue<QItem> heap;
    push_if_good(heap, nodes_[0].id);
    int leaves = 1;
    const int max_leaves = std::max(1, cfg_.max_leaves);

    while (!heap.empty() && leaves < max_leaves) {
      auto [neg_pr, nid, _] = heap.top(); heap.pop();
      Node& nd = nodes_[nid];

      // Re-evaluate best split now (stats might have changed from siblings)
      auto sp = eval_best_split_(nd);
      if (sp.feature < 0) {
        // finalize as leaf
        nd.leaf_value = calc_leaf_value_(nd.g_sum, nd.h_sum);
        continue;
      }
      // Apply
      int left_child_id = apply_split_(nid, sp);
      if (left_child_id < 0) {
        nd.leaf_value = calc_leaf_value_(nd.g_sum, nd.h_sum);
        continue;
      }
      // one leaf -> two leaves
      ++leaves;

      // enqueue children
      push_if_good(heap, nd.left);
      push_if_good(heap, nd.right);
    }
  }

  // Level-wise growth (queue)
  void grow_level_wise_() {
    std::vector<int> q; q.push_back(nodes_[0].id);
    while (!q.empty()) {
      int nid = q.front(); q.erase(q.begin());
      Node& nd = nodes_[nid];
      if (nd.depth >= cfg_.max_depth) {
        nd.leaf_value = calc_leaf_value_(nd.g_sum, nd.h_sum);
        continue;
      }
      auto sp = eval_best_split_(nd);
      if (sp.feature < 0) {
        nd.leaf_value = calc_leaf_value_(nd.g_sum, nd.h_sum);
        continue;
      }
      if (apply_split_(nid, sp) < 0) {
        nd.leaf_value = calc_leaf_value_(nd.g_sum, nd.h_sum);
        continue;
      }
      q.push_back(nd.left);
      q.push_back(nd.right);
    }
  }

  // Build packed arrays for fast prediction
  void build_pred_arrays_() {
    int max_id = -1;
    for (auto& n : nodes_) max_id = std::max(max_id, n.id);
    const int N = max_id + 1;
    p_feature_.assign(N, -1);
    p_thresh_.assign(N, -1);
    p_missing_left_.assign(N, 0);
    p_left_.assign(N, -1);
    p_right_.assign(N, -1);
    p_is_leaf_.assign(N, 1);
    p_leaf_value_.assign(N, 0.0);

    for (auto& n : nodes_) {
      if (n.is_leaf) {
        p_is_leaf_[n.id] = 1;
        p_leaf_value_[n.id] = n.leaf_value;
      } else {
        p_is_leaf_[n.id] = 0;
        p_feature_[n.id] = n.feature;
        p_thresh_[n.id] = n.threshold_bin;
        p_missing_left_[n.id] = n.missing_go_left ? 1 : 0;
        p_left_[n.id] = n.left;
        p_right_[n.id] = n.right;
      }
    }
    root_id_ = nodes_.empty() ? -1 : nodes_[0].id;
    packed_valid_ = true;
  }

  // Traversal (packed)
  inline double predict_one_packed_(const uint16_t* row) const {
    int nid = root_id_;
    while (nid >= 0 && p_is_leaf_[nid] == 0) {
      const int f = p_feature_[nid];
      const int t = p_thresh_[nid];
      const bool miss_left = p_missing_left_[nid] != 0;
      const uint16_t b = row[f];
      const bool is_missing = (b == (uint16_t)missing_bin_id());
      const bool go_left = is_missing ? miss_left : (b <= (uint16_t)t);
      nid = go_left ? p_left_[nid] : p_right_[nid];
    }
    return (nid >= 0) ? p_leaf_value_[nid] : 0.0;
  }

  // Slow traversal (for safety)
  inline double predict_one_slow_(const uint16_t* row) const {
    const Node* nd = nodes_.empty() ? nullptr : &nodes_[0];
    while (nd && !nd->is_leaf) {
      const uint16_t b = row[nd->feature];
      const bool is_missing = (b == (uint16_t)missing_bin_id());
      const bool go_left = is_missing ? nd->missing_go_left : (b <= (uint16_t)nd->threshold_bin);
      const int next_id = go_left ? nd->left : nd->right;
      nd = (next_id >= 0) ? &nodes_[id2pos_(next_id)] : nullptr;
    }
    return nd ? nd->leaf_value : 0.0;
  }

  // map sparse ids to vector positions (linear scan fallback)
  int id2pos_(int id) const {
    for (int i = 0; i < (int)nodes_.size(); ++i) if (nodes_[i].id == id) return i;
    return -1;
  }
};

} // namespace foretree
