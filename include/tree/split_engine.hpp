#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>
#include <cmath>
#include <type_traits>

#include "split_finder.hpp"  // AxisSplitFinder, CategoricalKWaySplitFinder, ObliqueSplitFinder
                             // SplitHyper, foretree::splitx::SplitContext, foretree::splitx::Candidate, foretree::splitx::SplitKind

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
  // match the objective used in split_finder.hpp (gain math):
  static inline double leaf_obj(double G, double H, double lambda, double alpha) {
    const double denom = H + lambda;
    if (!(denom > 0.0)) return 0.0;
    const double gs = soft(G, alpha);
    return 0.5 * (gs * gs) / denom;
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
    return (Rl + Rr - Rpar) - hyp.gamma_; // children minus parent, minus gamma
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
    if (mono > 0 && wL > wR) return false; // non-decreasing: wL <= wR
    if (mono < 0 && wL < wR) return false; // non-increasing: wL >= wR
    return true;
  }
};

enum class ObliqueMode : uint8_t {
  Off,               // never try oblique
  Full,              // ObliqueSplitFinder (k-feature ridge WLS)
  InteractionSeeded, // 2-feature interaction-seeded
  Auto               // try both, keep the better (with axis guard)
};

// ------------------------- Config for extra modes -----------------------------
struct SplitEngineConfig {
  bool enable_axis    = true;
  bool enable_kway    = false;   // histogram backend only
  bool enable_oblique = false;   // set true to try oblique splits

  // Axis vs Oblique guard: if axis_gain * guard >= oblique_gain, keep axis
  double axis_vs_oblique_guard = 1.02;

  // K-way knob
  int kway_max_groups = 8;

  // Oblique knobs
  int    oblique_k_features = 2;
  double oblique_ridge      = 1e-3;

  // oblique mode
  ObliqueMode oblique_mode = ObliqueMode::Off;

  // interaction-seeded params
  InteractionSeededConfig iseed; // from your file; default ctor is fine
};

// ============================ Backend Policies ===============================
//
// We use two policy types exposing a uniform static API:
//
//   struct Backend {
//     static constexpr bool supports_kway    = ...;
//     static constexpr bool supports_oblique = ...;
//
//     static foretree::splitx::Candidate best_axis(const foretree::splitx::SplitContext&);
//
//     // Only if supports_kway == true
//     static foretree::splitx::Candidate best_kway(const foretree::splitx::SplitContext&, int max_groups);
//
//     // Only if supports_oblique == true
//     static foretree::splitx::Candidate best_oblique(const foretree::splitx::SplitContext&,
//                                   int k_features, double ridge,
//                                   double axis_guard_gain /*may be < 0*/,
//                                   // Exact-only extras (ignored by Histogram):
//                                   const float* Xraw = nullptr, int P = 0,
//                                   const int* node_idx = nullptr, int nidx = 0,
//                                   int missing_policy = 0,
//                                   const uint8_t* miss_mask = nullptr);
//   }
//
// ============================================================================

// --------------------------- Histogram backend --------------------------------
struct HistogramBackend {
  static constexpr bool supports_kway    = true;
  static constexpr bool supports_oblique = true;  // histogram-backed oblique via bin centers

  static foretree::splitx::Candidate best_axis(const foretree::splitx::SplitContext& ctx) {
    AxisSplitFinder finder;
    return finder.best_axis(ctx);
  }

  static foretree::splitx::Candidate best_kway(const foretree::splitx::SplitContext& ctx, int max_groups) {
    CategoricalKWaySplitFinder finder;
    finder.max_groups = std::max(2, max_groups);
    return finder.best_kway(ctx);
  }

  static foretree::splitx::Candidate best_oblique(const foretree::splitx::SplitContext& ctx,
                                int k_features,
                                double ridge,
                                double /*axis_guard_gain*/ = -1.0,
                                const float* = nullptr, int = 0,
                                const int* = nullptr,  int = 0,
                                int = 0, const uint8_t* = nullptr) {
    // Uses oblique trained on bin centers (fast approx).
    ObliqueSplitFinder finder;
    finder.k_features = std::max(2, k_features);
    finder.ridge      = ridge;
    return finder.best_oblique_hist(ctx);
  }
};

// ------------------------------ Exact backend ---------------------------------
struct ExactBackend {
  static constexpr bool supports_kway    = false;
  static constexpr bool supports_oblique = true;

  // Now fully delegated to AxisSplitFinder exact-mode API
  static foretree::splitx::Candidate best_axis(const foretree::splitx::SplitContext& ctx,
                             const float* Xraw, int P,
                             const int* node_idx, int nidx,
                             int missing_policy,
                             const uint8_t* miss_mask) {
    AxisSplitFinder finder;
    return finder.best_axis_exact(ctx, Xraw, P, node_idx, nidx, missing_policy, miss_mask);
  }

  // Oblique remains delegated to ObliqueSplitFinder (exact uses ctx columns)
  static foretree::splitx::Candidate best_oblique(const foretree::splitx::SplitContext& ctx,
                                int k_features,
                                double ridge,
                                double axis_guard_gain = -1.0,
                                const float* /*Xraw*/ = nullptr, int /*P*/ = 0,
                                const int* /*node_idx*/ = nullptr, int /*nidx*/ = 0,
                                int /*missing_policy*/ = 0, const uint8_t* /*miss*/ = nullptr) {
    if (!ctx.Xcols || !ctx.row_g || !ctx.row_h || ctx.N <= 0) {
      foretree::splitx::Candidate c; c.kind = foretree::splitx::SplitKind::Oblique;
      c.gain = -std::numeric_limits<double>::infinity();
      return c;
    }
    ObliqueSplitFinder finder;
    finder.k_features = std::max(2, k_features);
    finder.ridge      = ridge;
    return finder.best_oblique(ctx, axis_guard_gain);
  }
};

// ============================ Splitter (templated) ============================

struct Splitter {

  // --------------------- PRESERVE ALL EXISTING APIs -------------------------
  
  // Legacy API 1: Simple 2-parameter call
  static inline foretree::splitx::Candidate best_split(const foretree::splitx::SplitContext& ctx, SplitEngine eng) {
    if (eng == SplitEngine::Histogram) return HistogramBackend::best_axis(ctx);
    // For Exact legacy call we cannot run without the raw arrays; return invalid.
    foretree::splitx::Candidate c; c.kind = foretree::splitx::SplitKind::Axis;
    c.gain = -std::numeric_limits<double>::infinity();
    return c;
  }

  // Legacy API 2: Exact with full raw parameters (PRESERVE EXACTLY)
  static inline foretree::splitx::Candidate best_split(const foretree::splitx::SplitContext& ctx, SplitEngine eng,
                                     const float* Xraw, int P,
                                     const int* node_idx, int nidx,
                                     int missing_policy, const uint8_t* miss_mask) {
    if (eng == SplitEngine::Histogram) return HistogramBackend::best_axis(ctx);
    return ExactBackend::best_axis(ctx, Xraw, P, node_idx, nidx, missing_policy, miss_mask);
  }

  // --------------------------- NEW Enhanced APIs --------------------------------
  template <class Backend>
  static foretree::splitx::Candidate best_split_with_backend(const foretree::splitx::SplitContext& ctx,
                                           const SplitEngineConfig& cfg,
                                           // Exact-only extras (ignored by Histogram):
                                           const float* Xraw = nullptr, int P = 0,
                                           const int* node_idx = nullptr, int nidx = 0,
                                           int missing_policy = 0,
                                           const uint8_t* miss_mask = nullptr)
  {
    foretree::splitx::Candidate best; best.gain = -std::numeric_limits<double>::infinity();

    // 1) Axis
    foretree::splitx::Candidate axis; axis.gain = -std::numeric_limits<double>::infinity();
    if (cfg.enable_axis) {
      if constexpr (std::is_same_v<Backend, HistogramBackend>) {
        axis = Backend::best_axis(ctx);
      } else {
        axis = Backend::best_axis(ctx, Xraw, P, node_idx, nidx, missing_policy, miss_mask);
      }
    }

    // 2) K-way (only if supported by backend)
    foretree::splitx::Candidate kway; kway.gain = -std::numeric_limits<double>::infinity();
    if constexpr (Backend::supports_kway) {
      if (cfg.enable_kway) {
        kway = Backend::best_kway(ctx, cfg.kway_max_groups);
      }
    }

    // 3) Oblique (optional; only if supported)
    foretree::splitx::Candidate obli; obli.gain = -std::numeric_limits<double>::infinity();
    if constexpr (Backend::supports_oblique) {
      if (cfg.enable_oblique) {
        const double guard = (axis.gain > 0.0) ? axis.gain : -1.0;
        obli = Backend::best_oblique(ctx,
                                     cfg.oblique_k_features,
                                     cfg.oblique_ridge,
                                     guard,
                                     Xraw, P, node_idx, nidx, missing_policy, miss_mask);
        // Prefer axis if gains are essentially tied (guard >= 1.0 keeps axis)
        if (axis.gain > 0.0 && obli.gain > 0.0 &&
            axis.gain * cfg.axis_vs_oblique_guard >= obli.gain) {
          obli.gain = -std::numeric_limits<double>::infinity();
        }
      }
    }

    // 4) Pick max-gain among enabled foretree::splitx::Candidates
    best = axis;
    if (kway.gain > best.gain) best = kway;
    if (obli.gain > best.gain) best = obli;

    if (!std::isfinite(best.gain)) {
      best.kind = foretree::splitx::SplitKind::Axis;
      best.gain = -std::numeric_limits<double>::infinity();
    }
    return best;
  }

  // New API: Enhanced with config
  static foretree::splitx::Candidate best_split(const foretree::splitx::SplitContext& ctx,
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
