#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>
#include <cmath>

#include "split_finder.hpp"  // AxisSplitFinder, CategoricalKWaySplitFinder, ObliqueSplitFinder
                             // SplitHyper, SplitContext, Candidate, SplitKind

namespace foretree {

// ------------------------------ Engines --------------------------------------
enum class SplitEngine { Histogram, Exact };

// ---------------------------- Shared utils -----------------------------------
struct SplitUtils {
  static inline double soft(double g, double alpha) {
    if (alpha <= 0.0) return g;
    if (g >  alpha) return g - alpha;
    if (g < -alpha) return g + alpha;
    return 0.0;
  }
  static inline double leaf_obj(double G, double H, double lambda, double alpha) {
    const double denom = H + lambda;
    if (!(denom > 0.0)) return 0.0;
    const double gs = soft(G, alpha);
    const double w  = -gs / denom;
    return 0.5 * denom * w * w + w * G + std::abs(w) * alpha;
  }
  static inline double split_gain(double Gp, double Hp,
                                  double GL, double HL,
                                  int nL, int nR,
                                  const SplitHyper& hyp) {
    if (nL < hyp.min_samples_leaf || nR < hyp.min_samples_leaf)
      return -std::numeric_limits<double>::infinity();
    if (HL < hyp.min_child_weight || (Hp - HL) < hyp.min_child_weight)
      return -std::numeric_limits<double>::infinity();

    const double Rpar = leaf_obj(Gp, Hp, hyp.lambda_, hyp.alpha_);
    const double Rl   = leaf_obj(GL, HL, hyp.lambda_, hyp.alpha_);
    const double Rr   = leaf_obj(Gp - GL, Hp - HL, hyp.lambda_, hyp.alpha_);
    return (Rpar - (Rl + Rr)) - hyp.gamma_;
  }
  static inline bool monotone_ok(int8_t mono, double GL, double HL,
                                 double GR, double HR, const SplitHyper& hyp) {
    if (mono == 0) return true;
    const double denomL = HL + hyp.lambda_;
    const double denomR = HR + hyp.lambda_;
    if (!(denomL > 0.0) || !(denomR > 0.0)) return true;

    auto wt = [&](double Gx, double Hx){
      const double gs = soft(Gx, hyp.alpha_);
      return -gs / (Hx + hyp.lambda_);
    };
    const double wL = wt(GL, HL), wR = wt(GR, HR);
    if (mono > 0 && wL > wR) return false;
    if (mono < 0 && wL < wR) return false;
    return true;
  }
};

// ------------------------- Config for extra modes -----------------------------
struct SplitEngineConfig {
  bool enable_axis    = true;
  bool enable_kway    = false;   // histogram backend only
  bool enable_oblique = false;

  // Axis vs Oblique guard: if axis_gain * guard >= oblique_gain, keep axis
  double axis_vs_oblique_guard = 1.02;

  // K-way knob
  int kway_max_groups = 8;

  // Oblique knobs
  int    oblique_k_features = 2;
  double oblique_ridge      = 1e-3;
};

// ============================ Backend Policies ===============================
//
// We use two tiny policy types exposing a uniform static API:
//
//   struct Backend {
//     static constexpr bool supports_kway    = ...;
//     static constexpr bool supports_oblique = ...;
//
//     static Candidate best_axis(const SplitContext&);
//
//     // Only if supports_kway == true
//     static Candidate best_kway(const SplitContext&, int max_groups);
//
//     // Only if supports_oblique == true
//     static Candidate best_oblique(const SplitContext&,
//                                   int k_features, double ridge,
//                                   double axis_guard_gain /*may be < 0*/,
//                                   // Exact-only extras (ignored by Histogram):
//                                   const float* Xraw = nullptr, int P = 0,
//                                   const int* node_idx = nullptr, int nidx = 0,
//                                   int missing_policy = 0,
//                                   const uint8_t* miss_mask = nullptr);
//   }
//
// This avoids repetitive switch/if ladders in the high-level Splitter.
// ============================================================================

// --------------------------- Histogram backend --------------------------------
struct HistogramBackend {
  static constexpr bool supports_kway    = true;
  static constexpr bool supports_oblique = true;

  static Candidate best_axis(const SplitContext& ctx) {
    AxisSplitFinder finder;
    return finder.best_axis(ctx);
  }

  static Candidate best_kway(const SplitContext& ctx, int max_groups) {
    CategoricalKWaySplitFinder finder;
    finder.max_groups = std::max(2, max_groups);
    return finder.best_kway(ctx);
  }

  static Candidate best_oblique(const SplitContext& ctx,
                                int k_features,
                                double ridge,
                                double axis_guard_gain = -1.0,
                                const float* = nullptr, int = 0,
                                const int* = nullptr,  int = 0,
                                int = 0, const uint8_t* = nullptr) {
    // Works if ctx.Xcols is set; otherwise ObliqueSplitFinder will no-op return
    ObliqueSplitFinder finder;
    finder.k_features = std::max(2, k_features);
    finder.ridge      = ridge;
    return finder.best_oblique(ctx, axis_guard_gain);
  }
};

// ------------------------------ Exact backend ---------------------------------
struct ExactBackend {
  static constexpr bool supports_kway    = false;
  static constexpr bool supports_oblique = true;

  static Candidate best_axis(const SplitContext& ctx,
                             const float* Xraw, int P,
                             const int* node_idx, int nidx,
                             int missing_policy,
                             const uint8_t* miss_mask) {
    Candidate best; best.kind = SplitKind::Axis; best.thr = -1;
    best.gain = -std::numeric_limits<double>::infinity();
    if (!Xraw || !ctx.row_g || !ctx.row_h || nidx <= 1) return best;

    std::vector<std::pair<float,int>> col; col.reserve(nidx);

    for (int f = 0; f < P; ++f) {
      col.clear();
      for (int ii = 0; ii < nidx; ++ii) {
        const int r = node_idx[ii];
        const size_t off = (size_t)r*(size_t)P + (size_t)f;
        const bool miss = miss_mask ? (miss_mask[off] != 0)
                                    : !std::isfinite(Xraw[off]);
        if (!miss) col.emplace_back(Xraw[off], r);
      }
      const int n_valid = (int)col.size();
      if (n_valid < ctx.hyp.min_samples_leaf || n_valid < 2) continue;

      std::sort(col.begin(), col.end(),
                [](const auto& a, const auto& b){ return a.first < b.first; });

      const double Gm = (ctx.Gmiss ? ctx.Gmiss[f] : 0.0);
      const double Hm = (ctx.Hmiss ? ctx.Hmiss[f] : 0.0);
      const int    Cm = (ctx.Cmiss ? ctx.Cmiss[f] : 0);
      const bool has_miss = ctx.has_missing && (Cm > 0);

      double GL = 0.0, HL = 0.0; int nL = 0;
      const int8_t mono = (ctx.monotone && f < (int)ctx.monotone->size()) ? (*ctx.monotone)[f] : 0;

      for (int k = 0; k < n_valid - 1; ++k) {
        const int r = col[k].second;
        GL += (double)ctx.row_g[r];
        HL += (double)ctx.row_h[r];
        ++nL;

        const float v  = col[k].first;
        const float vp = col[k+1].first;
        if (!(v < vp)) continue;

        auto eval_dir = [&](bool miss_left){
          double GLL = GL, HLL = HL; int nLL = nL;
          if (miss_left && has_miss) { GLL += Gm; HLL += Hm; nLL += Cm; }
          const double GRR = ctx.Gp - GLL;
          const double HRR = ctx.Hp - HLL;
          const int    nRR = (has_miss ? (n_valid + Cm) : n_valid) - nLL;
          if (!SplitUtils::monotone_ok(mono, GLL, HLL, GRR, HRR, ctx.hyp))
            return -std::numeric_limits<double>::infinity();
          return SplitUtils::split_gain(ctx.Gp, ctx.Hp, GLL, HLL, nLL, nRR, ctx.hyp);
        };

        double gain; bool miss_left_pick = true;
        if (missing_policy == 1) {
          gain = eval_dir(true);
          miss_left_pick = true;
        } else if (missing_policy == 2) {
          gain = eval_dir(false);
          miss_left_pick = false;
        } else {
          const double gL = eval_dir(true);
          const double gR = eval_dir(false);
          if (gL >= gR) { gain = gL; miss_left_pick = true; }
          else          { gain = gR; miss_left_pick = false; }
        }

        if (gain > best.gain) {
          best.gain     = gain;
          best.kind     = SplitKind::Axis;
          best.feat     = f;
          best.thr      = k;           // rank in the sorted valid list
          best.miss_left= miss_left_pick;
          // Optionally: best.split_value = 0.5*(double(v)+double(vp));
        }
      }
    }
    return best;
  }

  static Candidate best_oblique(const SplitContext& ctx,
                                int k_features,
                                double ridge,
                                double axis_guard_gain = -1.0,
                                const float* Xraw = nullptr, int /*P*/ = 0,
                                const int* /*node_idx*/ = nullptr, int /*nidx*/ = 0,
                                int /*missing_policy*/ = 0, const uint8_t* /*miss*/ = nullptr) {
    // Exact-oblique relies on ctx.Xcols/row access prepared by caller.
    if (!ctx.Xcols || !ctx.row_g || !ctx.row_h || ctx.N <= 0) {
      Candidate c; c.kind = SplitKind::Oblique;
      c.gain = -std::numeric_limits<double>::infinity();
      return c;
    }
    (void)Xraw;
    ObliqueSplitFinder finder;
    finder.k_features = std::max(2, k_features);
    finder.ridge      = ridge;
    return finder.best_oblique(ctx, axis_guard_gain);
  }
};

// ============================ Splitter (templated) ============================

struct Splitter {

  // --------------------- Minimal legacy APIs (axis-only) ---------------------
  static inline Candidate best_split(const SplitContext& ctx, SplitEngine eng) {
    if (eng == SplitEngine::Histogram) return HistogramBackend::best_axis(ctx);
    // For Exact legacy call we cannot run without the raw arrays; return invalid.
    Candidate c; c.kind = SplitKind::Axis;
    c.gain = -std::numeric_limits<double>::infinity();
    return c;
  }

  static inline Candidate best_split(const SplitContext& ctx, SplitEngine eng,
                                     const float* Xraw, int P,
                                     const int* node_idx, int nidx,
                                     int missing_policy, const uint8_t* miss_mask) {
    if (eng == SplitEngine::Histogram) return HistogramBackend::best_axis(ctx);
    return ExactBackend::best_axis(ctx, Xraw, P, node_idx, nidx, missing_policy, miss_mask);
  }

  // --------------------------- New unified API --------------------------------
  template <class Backend>
  static Candidate best_split_with_backend(const SplitContext& ctx,
                                           const SplitEngineConfig& cfg,
                                           // Exact-only extras (ignored by Histogram):
                                           const float* Xraw = nullptr, int P = 0,
                                           const int* node_idx = nullptr, int nidx = 0,
                                           int missing_policy = 0,
                                           const uint8_t* miss_mask = nullptr)
  {
    Candidate best; best.gain = -std::numeric_limits<double>::infinity();

    // 1) Axis
    Candidate axis; axis.gain = -std::numeric_limits<double>::infinity();
    if (cfg.enable_axis) {
      if constexpr (std::is_same_v<Backend, HistogramBackend>) {
        axis = Backend::best_axis(ctx);
      } else {
        axis = Backend::best_axis(ctx, Xraw, P, node_idx, nidx, missing_policy, miss_mask);
      }
    }

    // 2) K-way (only if supported by backend)
    Candidate kway; kway.gain = -std::numeric_limits<double>::infinity();
    if constexpr (Backend::supports_kway) {
      if (cfg.enable_kway) {
        kway = Backend::best_kway(ctx, cfg.kway_max_groups);
      }
    }

    // 3) Oblique (optional)
    Candidate obli; obli.gain = -std::numeric_limits<double>::infinity();
    if constexpr (Backend::supports_oblique) {
      if (cfg.enable_oblique) {
        const double guard = (axis.gain > 0.0) ? axis.gain : -1.0;
        obli = Backend::best_oblique(ctx,
                                     cfg.oblique_k_features,
                                     cfg.oblique_ridge,
                                     guard,
                                     Xraw, P, node_idx, nidx, missing_policy, miss_mask);
        // Optional guard: if axis clearly wins, prefer axis
        if (axis.gain > 0.0 && obli.gain > 0.0 &&
            axis.gain * cfg.axis_vs_oblique_guard >= obli.gain) {
          obli.gain = -std::numeric_limits<double>::infinity();
        }
      }
    }

    // 4) Pick max-gain among enabled candidates
    best = axis;
    if (kway.gain > best.gain) best = kway;
    if (obli.gain > best.gain) best = obli;

    if (!std::isfinite(best.gain)) {
      best.kind = SplitKind::Axis;
      best.gain = -std::numeric_limits<double>::infinity();
    }
    return best;
  }

  // Convenience frontends mirroring your original signature set
  static Candidate best_split(const SplitContext& ctx,
                              SplitEngine backend,
                              const SplitEngineConfig& cfg,
                              const float* Xraw = nullptr, int P = 0,
                              const int* node_idx = nullptr, int nidx = 0,
                              int missing_policy = 0,
                              const uint8_t* miss_mask = nullptr)
  {
    if (backend == SplitEngine::Histogram) {
      return best_split_with_backend<HistogramBackend>(ctx, cfg);
    } else {
      return best_split_with_backend<ExactBackend>(ctx, cfg, Xraw, P, node_idx, nidx, missing_policy, miss_mask);
    }
  }
};

} // namespace foretree
