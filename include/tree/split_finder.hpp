#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace foretree {

// ============================================================================
// Split hyper-parameters (unchanged API)
// ============================================================================
struct SplitHyper {
    double lambda_ = 1.0;
    double alpha_ = 0.0;
    double gamma_ = 0.0;
    double max_delta_step = 0.0; // reserved for future use in leaf_obj
    int min_samples_leaf = 5;
    double min_child_weight = 1e-3;
    // 0=Learn, 1=AlwaysLeft, 2=AlwaysRight
    int missing_policy = 0;
    double leaf_gain_eps = 0.0;
    bool allow_zero_gain = false;
};

// ============================================================================
// Small numerics helpers
// ============================================================================
namespace detail {
constexpr double NEG_INF = -1.0 * std::numeric_limits<double>::infinity();
constexpr double EPS = 1e-12;

inline double soft(double g, double alpha) {
    if (alpha <= 0.0)
        return g;
    if (g > alpha)
        return g - alpha;
    if (g < -alpha)
        return g + alpha;
    return 0.0;
}

inline bool pass_monotone_guard(int8_t mono, double wL, double wR) {
    // mono > 0: non-decreasing (wL <= wR), mono < 0: non-increasing (wL >= wR)
    if (mono == 0)
        return true;
    return mono > 0 ? (wL <= wR) : (wL >= wR);
}

inline double weight_from_GRH(double G, double H, const SplitHyper &hyp) {
    const double denom = H + hyp.lambda_;
    if (!(denom > 0.0))
        return 0.0;
    return -soft(G, hyp.alpha_) / denom;
}
} // namespace detail

// ============================================================================
// Leaf objective (kept compatible with your version here)
//  NOTE: This is 0.5 * soft(G)^2 / (H + lambda). It differs from a form that
//  also adds |w|*alpha, but matches the rest of this file's logic.
// ============================================================================
inline double leaf_obj(double G, double H, double lambda, double alpha) {
    const double denom = H + lambda;
    if (!(denom > 0.0))
        return 0.0;
    const double gs = detail::soft(G, alpha);
    return 0.5 * (gs * gs) / denom;
}

// ============================================================================
// Candidate: supports axis, k-way, oblique
// ============================================================================
enum class SplitKind : uint8_t { Axis = 0, KWay = 1, Oblique = 2 };

struct Candidate {
    // Common
    SplitKind kind = SplitKind::Axis;
    double gain = detail::NEG_INF;

    // --- Axis fields ---
    int feat = -1; // global/local feat id depending on caller
    int thr = -1;  // histogram bin threshold (finite bins)
    bool miss_left = true;
    double split_value =
        std::numeric_limits<double>::quiet_NaN(); // optional exact

    // --- K-way (categorical via histogram bins) ---
    std::vector<int> left_groups; // finite bin ids that go left
    int missing_group = -1; // which group collects missing (if multi-child)

    // --- Oblique ---
    std::vector<int> oblique_features;   // indices into features
    std::vector<double> oblique_weights; // same size as oblique_features
    double oblique_bias = 0.0;
    double oblique_threshold = std::numeric_limits<double>::quiet_NaN();
    bool oblique_missing_left = true;
};

// ============================================================================
// UPDATED SplitContext: Enhanced for variable bin sizes per feature
// ============================================================================
struct SplitContext {
    // Histogram views (required for axis/k-way)
    const std::vector<double> *G = nullptr; // size depends on variable_bins flag
    const std::vector<double> *H = nullptr; // size depends on variable_bins flag
    const std::vector<int> *C = nullptr;    // size depends on variable_bins flag

    int P = 0;                 // #features
    int B = 0;                 // #bins per feature (uniform mode only)
    double Gp = 0.0, Hp = 0.0; // parent aggregates
    int Cp = 0;

    // Monotone constraints (either pointer can be set; accessor below)
    const std::vector<int8_t> *monotone = nullptr;

    SplitHyper hyp;

    // NEW: Variable bin size support
    bool variable_bins = false;                   // enable variable bin mode
    const size_t* feature_offsets = nullptr;     // cumulative offsets [P+1]
    const int* finite_bins_per_feat = nullptr;   // finite bins per feature [P]
    const int* missing_ids_per_feat = nullptr;   // missing bin ID per feature [P]



    // --- exact-mode (optional) ---
    const float *row_g = nullptr; // size N_total (or N at node)
    const float *row_h = nullptr; // size N_total (or N at node)
    // Optional per-feature missing aggregates (for exact)
    const double *Gmiss = nullptr; // size P
    const double *Hmiss = nullptr; // size P
    const int *Cmiss = nullptr;    // size P
    bool has_missing = false;

    // Row-level feature access for oblique (optional)
    int N = 0; // #rows in the node
    const double *const *Xcols =
        nullptr; // Xcols[f] -> pointer to N doubles for feature f

    // --- binned-matrix path for oblique (histogram-based) ---
    const uint16_t *Xb = nullptr;   // prebinned matrix (row-major, N_total x P)
    const int *row_index = nullptr; // pointer to node's row indices (length N)
    int miss_bin_id = -1;           // usually B-1 (uniform) or per-feature (variable)
    const double *bin_centers = nullptr; // size varies based on variable_bins
    int Bz = 256;                        // #bins for projection z histogram

    // Helper accessor: choose whichever monotone vector is set
    const std::vector<int8_t> *mono_ptr() const { return monotone; }

    
    // NEW: Helper methods for variable bin access
    size_t get_histogram_offset(int feature, int bin) const {
        if (variable_bins) {
            return feature_offsets[feature] + static_cast<size_t>(bin);
        } else {
            return static_cast<size_t>(feature) * static_cast<size_t>(B) + static_cast<size_t>(bin);
        }
    }

    int get_feature_bins(int feature) const {
        if (variable_bins) {
            return finite_bins_per_feat[feature];
        } else {
            return B - 1; // B includes missing bin
        }
    }

    int get_missing_bin_id(int feature) const {
        if (variable_bins) {
            return missing_ids_per_feat[feature];
        } else {
            return B - 1; // uniform: missing is last bin
        }
    }
};

// ============================================================================
// 1) UPDATED Axis (histogram) — compact & monotone-aware with variable bins
// ============================================================================
class AxisSplitFinder {
public:
    Candidate best_axis(const SplitContext &ctx) const {
        Candidate best;
        best.kind = SplitKind::Axis;
        best.gain = detail::NEG_INF;

        const int P = ctx.P;
        if (!ctx.G || !ctx.H || !ctx.C || P <= 0)
            return best;

        const auto &G = *ctx.G;
        const auto &H = *ctx.H;
        const auto &C = *ctx.C;
        const auto *mono_vec = ctx.mono_ptr();

        const double parent_gain =
            leaf_obj(ctx.Gp, ctx.Hp, ctx.hyp.lambda_, ctx.hyp.alpha_);

        for (int f = 0; f < P; ++f) {
            // NEW: Get per-feature bin information
            const int finite_bins = ctx.get_feature_bins(f);
            const int miss_id = ctx.get_missing_bin_id(f);
            
            if (finite_bins <= 0) continue;

            // Get missing bin statistics
            const size_t miss_offset = ctx.get_histogram_offset(f, miss_id);
            const double Gm = (miss_offset < G.size()) ? G[miss_offset] : 0.0;
            const double Hm = (miss_offset < H.size()) ? H[miss_offset] : 0.0;
            const int Cm = (miss_offset < C.size()) ? C[miss_offset] : 0;
            
            const int8_t mono =
                (mono_vec && f < (int)mono_vec->size()) ? (*mono_vec)[f] : 0;

            auto scan_dir = [&](bool miss_left) -> Candidate {
                Candidate cand;
                cand.kind = SplitKind::Axis;
                cand.feat = f;
                cand.thr = -1;
                cand.miss_left = miss_left;
                cand.gain = detail::NEG_INF;

                double GL = 0.0, HL = 0.0;
                int CL = 0;

                for (int t = 0; t < finite_bins; ++t) {
                    const size_t bin_offset = ctx.get_histogram_offset(f, t);
                    if (bin_offset >= G.size()) continue;
                    
                    GL += G[bin_offset];
                    HL += H[bin_offset];
                    CL += C[bin_offset];

                    // Left child: finite-left (+ missing if miss_left)
                    const double GLx = GL + (miss_left ? Gm : 0.0);
                    const double HLx = HL + (miss_left ? Hm : 0.0);
                    const int CLx = CL + (miss_left ? Cm : 0);

                    // Right child
                    const double GRx = (ctx.Gp - GL) - (miss_left ? Gm : 0.0);
                    const double HRx = (ctx.Hp - HL) - (miss_left ? Hm : 0.0);
                    const int CRx = (ctx.Cp - CL) - (miss_left ? Cm : 0);

                    // Guards
                    if (CLx < ctx.hyp.min_samples_leaf ||
                        CRx < ctx.hyp.min_samples_leaf)
                        continue;
                    if (HLx < ctx.hyp.min_child_weight ||
                        HRx < ctx.hyp.min_child_weight)
                        continue;

                    // Gain
                    const double gainL =
                        leaf_obj(GLx, HLx, ctx.hyp.lambda_, ctx.hyp.alpha_);
                    const double gainR =
                        leaf_obj(GRx, HRx, ctx.hyp.lambda_, ctx.hyp.alpha_);
                    double gain =
                        (gainL + gainR - parent_gain) - ctx.hyp.gamma_;

                    // Monotone constraint check (if any)
                    if (mono != 0) {
                        const double wL =
                            detail::weight_from_GRH(GLx, HLx, ctx.hyp);
                        const double wR =
                            detail::weight_from_GRH(GRx, HRx, ctx.hyp);
                        if (!detail::pass_monotone_guard(mono, wL, wR))
                            continue;
                    }

                    if (gain > cand.gain) {
                        cand.gain = gain;
                        cand.thr = t;
                    }
                }
                return cand;
            };

            Candidate cand = [&]() {
                if (ctx.hyp.missing_policy == 1)
                    return scan_dir(true);
                if (ctx.hyp.missing_policy == 2)
                    return scan_dir(false);
                Candidate cL = scan_dir(true), cR = scan_dir(false);
                return (cL.gain >= cR.gain) ? cL : cR;
            }();

            if (cand.thr >= 0 && cand.gain > best.gain)
                best = cand;
        }
        return best;
    }
};

// ============================================================================
// 2) UPDATED Categorical K-way from histograms with variable bins
// ============================================================================
class CategoricalKWaySplitFinder {
public:
    int max_groups = 8; // cap #groups we assemble internally

    Candidate best_kway(const SplitContext &ctx) const {
        Candidate best;
        best.kind = SplitKind::KWay;
        best.gain = detail::NEG_INF;

        if (!ctx.G || !ctx.H || !ctx.C || ctx.P <= 0)
            return best;

        const int P = ctx.P;
        const auto &G = *ctx.G;
        const auto &H = *ctx.H;

        const double lam = ctx.hyp.lambda_;
        const double gamma = ctx.hyp.gamma_;

        // Parent is node-level, independent of feature
        const double parent = (ctx.Gp * ctx.Gp) / (ctx.Hp + lam);

        for (int f = 0; f < P; ++f) {
            // NEW: Get per-feature bin information
            const int finite_bins = ctx.get_feature_bins(f);
            const int miss_id = ctx.get_missing_bin_id(f);
            
            if (finite_bins <= 1) continue;

            // count non-empty finite bins
            int non_empty = 0;
            for (int t = 0; t < finite_bins; ++t) {
                const size_t bin_offset = ctx.get_histogram_offset(f, t);
                if (bin_offset >= G.size()) continue;
                
                const double g = G[bin_offset];
                const double h = H[bin_offset];
                if (g != 0.0 || h > 0.0)
                    ++non_empty;
            }
            if (non_empty < 2)
                continue;

            // Heuristic: many active bins => likely continuous, skip
            if (non_empty > 32 && non_empty > finite_bins / 2)
                continue;

            // Score bins by |G|/(H+lam)
            std::vector<std::pair<double, int>> scored;
            scored.reserve(static_cast<size_t>(finite_bins));
            for (int t = 0; t < finite_bins; ++t) {
                const size_t bin_offset = ctx.get_histogram_offset(f, t);
                if (bin_offset >= G.size()) continue;
                
                const double g = G[bin_offset];
                const double h = H[bin_offset];
                if (g != 0.0 || h > 0.0) {
                    const double s = std::abs(g) / (h + lam + detail::EPS);
                    scored.emplace_back(s, t);
                }
            }
            if (scored.size() < 2)
                continue;

            std::sort(
                scored.begin(), scored.end(),
                [](const auto &a, const auto &b) { return a.first > b.first; });

            // Build groups: a few strong singletons + tail
            std::vector<std::vector<int>> groups;
            const int cap = std::max(2, max_groups);
            const int singles = (int)std::min<size_t>(cap - 1, scored.size());
            for (int i = 0; i < singles - 1; ++i)
                groups.push_back({scored[i].second});

            std::vector<int> tail;
            for (size_t k = std::max(0, singles - 1); k < scored.size(); ++k)
                tail.push_back(scored[k].second);
            if (!tail.empty())
                groups.push_back(std::move(tail));

            if (groups.size() < 2)
                continue;

            // Choose left group = strongest singleton group
            const std::vector<int> &left = groups[0];

            // Missing side: choose group with max Hessian to host missing
            std::vector<double> Hgroup(groups.size(), 0.0);
            for (size_t gi = 0; gi < groups.size(); ++gi) {
                double hs = 0.0;
                for (int t : groups[gi]) {
                    const size_t bin_offset = ctx.get_histogram_offset(f, t);
                    if (bin_offset < H.size()) {
                        hs += H[bin_offset];
                    }
                }
                Hgroup[gi] = hs;
            }
            const int mg = (int)std::distance(
                Hgroup.begin(), std::max_element(Hgroup.begin(), Hgroup.end()));
            const bool missing_left = (mg == 0);

            // Compute aggregates
            double GL = 0.0, HL = 0.0;
            for (int t : left) {
                const size_t bin_offset = ctx.get_histogram_offset(f, t);
                if (bin_offset < G.size()) {
                    GL += G[bin_offset];
                    HL += H[bin_offset];
                }
            }
            
            // Add missing bin statistics
            const size_t miss_offset = ctx.get_histogram_offset(f, miss_id);
            const double Gm = (miss_offset < G.size()) ? G[miss_offset] : 0.0;
            const double Hm = (miss_offset < H.size()) ? H[miss_offset] : 0.0;
            
            if (missing_left) {
                GL += Gm;
                HL += Hm;
            }

            const double GR = ctx.Gp - GL;
            const double HR = ctx.Hp - HL;

            if (HL < ctx.hyp.min_child_weight || HR < ctx.hyp.min_child_weight)
                continue;

            const double child =
                (GL * GL) / (HL + lam) + (GR * GR) / (HR + lam);
            const double gain = 0.5 * (child - parent) - gamma;

            if (gain > best.gain) {
                best.kind = SplitKind::KWay;
                best.gain = gain;
                best.feat = f;
                best.left_groups = left; // finite bin ids
                best.missing_group = mg;
                best.miss_left = missing_left; // for binary emulation
            }
        }
        return best;
    }
};

// ============================================================================
// Helper function for variable bin centers in oblique splitting
// ============================================================================
inline double x_from_code_variable(int f, uint16_t code, const SplitContext &ctx) {
    if (ctx.variable_bins) {
        // Variable bin mode: use feature offsets to find correct bin center
        const int finite_bins = ctx.get_feature_bins(f);
        const int miss_id = ctx.get_missing_bin_id(f);
        
        if (code == static_cast<uint16_t>(miss_id)) {
            // Missing value - caller should handle this
            return std::numeric_limits<double>::quiet_NaN();
        }
        
        if (code >= static_cast<uint16_t>(finite_bins)) {
            // Invalid bin - treat as missing
            return std::numeric_limits<double>::quiet_NaN();
        }
        
        const size_t center_offset = ctx.feature_offsets[f] + static_cast<size_t>(code);
        return ctx.bin_centers[center_offset];
    } else {
        // Uniform bin mode: original logic
        if (code == static_cast<uint16_t>(ctx.B - 1)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return ctx.bin_centers[static_cast<size_t>(f) * static_cast<size_t>(ctx.B) + static_cast<size_t>(code)];
    }
}

// ============================================================================
// Rest of the classes (Oblique, etc.) remain largely unchanged but should use
// the new helper functions and SplitContext methods for bin access
// ============================================================================

// Tiny linear-algebra helpers for oblique (Cholesky SPD)
inline bool cholesky_spd(std::vector<double> &A, int n) {
    // A is row-major, overwritten with lower-tri L
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = A[i * n + j];
            for (int k = 0; k < j; ++k)
                sum -= A[i * n + k] * A[j * n + k];
            if (i == j) {
                if (sum <= 0.0)
                    return false;
                A[i * n + j] = std::sqrt(sum);
            } else
                A[i * n + j] = sum / A[j * n + j];
        }
        for (int j = i + 1; j < n; ++j)
            A[i * n + j] = 0.0; // zero upper
    }
    return true;
}

inline void chol_solve_inplace(const std::vector<double> &L, int n,
                               const std::vector<double> &b,
                               std::vector<double> &x) {
    x = b;
    // forward: L y = b
    for (int i = 0; i < n; ++i) {
        double s = x[i];
        for (int k = 0; k < i; ++k)
            s -= L[i * n + k] * x[k];
        x[i] = s / L[i * n + i];
    }
    // backward: L^T x = y
    for (int i = n - 1; i >= 0; --i) {
        double s = x[i];
        for (int k = i + 1; k < n; ++k)
            s -= L[k * n + i] * x[k];
        x[i] = s / L[i * n + i];
    }
}

// Build normal equations (X^T diag(h) X + (ridge+λ)I) w = - X^T g
// XS is a list of column pointers for selected features
inline void build_normal_eq_cols(const std::vector<const double *> &XS,
                                 const float *g, const float *h, int N,
                                 double ridge_plus_lambda,
                                 std::vector<double> &A,
                                 std::vector<double> &b) {
    const int k = (int)XS.size();
    A.assign((size_t)k * (size_t)k, 0.0);
    b.assign((size_t)k, 0.0);

    // b = - X^T g
    for (int i = 0; i < N; ++i) {
        const double gi = (double)g[i];
        for (int c = 0; c < k; ++c) {
            const double xic = XS[c][i];
            if (std::isfinite(xic))
                b[(size_t)c] -= xic * gi;
        }
    }

    // A = X^T diag(h) X  (upper triangle)
    for (int i = 0; i < N; ++i) {
        const double hi = (double)h[i];
        for (int r = 0; r < k; ++r) {
            const double xir = XS[r][i];
            if (!std::isfinite(xir))
                continue;
            const double hrx = hi * xir;
            for (int c = r; c < k; ++c) {
                const double xic = XS[c][i];
                if (std::isfinite(xic))
                    A[(size_t)r * (size_t)k + (size_t)c] += hrx * xic;
            }
        }
    }
    // symmetrize + ridge
    for (int r = 0; r < k; ++r) {
        for (int c = r + 1; c < k; ++c)
            A[(size_t)c * (size_t)k + (size_t)r] =
                A[(size_t)r * (size_t)k + (size_t)c];
        A[(size_t)r * (size_t)k + (size_t)r] += ridge_plus_lambda;
    }
}

// ============================================================================
// UPDATED: Build normal equations from binned matrix using variable bin centers
// ============================================================================
inline void build_normal_eq_from_codes(
    const std::vector<int> &S, const uint16_t *Xb, const int *rows, int nrows,
    int P, const SplitContext &ctx, const float *g, const float *h,
    double ridge_plus_lambda, std::vector<double> &A, std::vector<double> &b) {
    const int k = (int)S.size();
    A.assign((size_t)k * (size_t)k, 0.0);
    b.assign((size_t)k, 0.0);
    std::vector<double> xi((size_t)k, 0.0);

    // b = - X^T g  and A = X^T diag(h) X (upper)
    for (int rr = 0; rr < nrows; ++rr) {
        const int i = rows[rr];
        bool miss = false;
        for (int c = 0; c < k; ++c) {
            const int f = S[c];
            const uint16_t code = Xb[(size_t)i * (size_t)P + (size_t)f];
            const double x_val = x_from_code_variable(f, code, ctx);
            if (!std::isfinite(x_val)) {
                miss = true;
                break;
            }
            xi[(size_t)c] = x_val;
        }
        if (miss)
            continue;

        const double gi = (double)g[i];
        for (int c = 0; c < k; ++c)
            b[(size_t)c] -= xi[(size_t)c] * gi;

        const double hi = (double)h[i];
        for (int r = 0; r < k; ++r) {
            const double hrx = hi * xi[(size_t)r];
            for (int c = r; c < k; ++c)
                A[(size_t)r * (size_t)k + (size_t)c] += hrx * xi[(size_t)c];
        }
    }
    // symmetrize + ridge
    for (int r = 0; r < k; ++r) {
        for (int c = r + 1; c < k; ++c)
            A[(size_t)c * (size_t)k + (size_t)r] =
                A[(size_t)r * (size_t)k + (size_t)c];
        A[(size_t)r * (size_t)k + (size_t)r] += ridge_plus_lambda;
    }
}

// ============================================================================
// Best split along projection z = w^T x with missing-policy search
// ============================================================================
inline std::tuple<double, double, bool> best_split_on_projection(
    const std::vector<double> &z, const float *g, const float *h, int N,
    const std::vector<uint8_t> &miss_mask, const SplitHyper &hyp) {
    // Collect finite indices & missing aggregates
    std::vector<int> idx;
    idx.reserve(N);
    double gm = 0.0, hm = 0.0;
    for (int i = 0; i < N; ++i) {
        if (miss_mask[i]) {
            gm += (double)g[i];
            hm += (double)h[i];
        } else {
            idx.push_back(i);
        }
    }
    const int nf = (int)idx.size();
    if (nf < 4)
        return {detail::NEG_INF, 0.0, true};

    // sort by z
    std::sort(idx.begin(), idx.end(),
              [&](int i, int j) { return z[i] < z[j]; });

    // prefix on sorted finite
    std::vector<double> pg(nf), ph(nf);
    for (int k = 0; k < nf; ++k) {
        const int i = idx[k];
        pg[k] = (double)g[i] + (k ? pg[k - 1] : 0.0);
        ph[k] = (double)h[i] + (k ? ph[k - 1] : 0.0);
    }

    const double gtot = pg[nf - 1] + gm;
    const double htot = ph[nf - 1] + hm;
    const double parent = (gtot * gtot) / (htot + hyp.lambda_);

    double best_gain = detail::NEG_INF, best_thr = 0.0;
    bool best_mleft = true;

    for (int k = 1; k < nf; ++k) {
        const int iL = idx[k - 1], iR = idx[k];
        if (!(z[iR] > z[iL] + 1e-15))
            continue;
        const double thr = 0.5 * (z[iL] + z[iR]);

        // finite L/R
        const double glb = pg[k - 1];
        const double hlb = ph[k - 1];
        const double grb = pg[nf - 1] - glb;
        const double hrb = ph[nf - 1] - hlb;

        // missing -> left
        {
            const double HL = hlb + hm, HR = hrb;
            if (HL >= hyp.min_child_weight && HR >= hyp.min_child_weight) {
                const double GL = glb + gm, GR = grb;
                const double child = (GL * GL) / (HL + hyp.lambda_) +
                                     (GR * GR) / (HR + hyp.lambda_);
                const double gain = 0.5 * (child - parent) - hyp.gamma_;
                if (gain > best_gain) {
                    best_gain = gain;
                    best_thr = thr;
                    best_mleft = true;
                }
            }
        }
        // missing -> right
        {
            const double HL = hlb, HR = hrb + hm;
            if (HL >= hyp.min_child_weight && HR >= hyp.min_child_weight) {
                const double GL = glb, GR = grb + gm;
                const double child = (GL * GL) / (HL + hyp.lambda_) +
                                     (GR * GR) / (HR + hyp.lambda_);
                const double gain = 0.5 * (child - parent) - hyp.gamma_;
                if (gain > best_gain) {
                    best_gain = gain;
                    best_thr = thr;
                    best_mleft = false;
                }
            }
        }
    }
    return {best_gain, best_thr, best_mleft};
}

// ============================================================================
// UPDATED: Build 1-D histogram on z = w^T x using variable bin centers
// ============================================================================
inline void build_projection_hist_from_codes(
    const std::vector<int> &S, const std::vector<double> &w, const uint16_t *Xb,
    const int *rows, int nrows, int P, const SplitContext &ctx, 
    const float *g, const float *h, std::vector<double> &Gz,
    std::vector<double> &Hz, std::vector<int> &Cz, double &Gm, double &Hm,
    int &Cm, double &zmin, double &zmax) {
    Gm = Hm = 0.0;
    Cm = 0;
    const int Bz = (ctx.Bz > 0) ? ctx.Bz : 256;
    
    if (nrows <= 0 || S.empty()) {
        Gz.assign(Bz, 0.0);
        Hz.assign(Bz, 0.0);
        Cz.assign(Bz, 0);
        zmin = 0.0;
        zmax = 1.0;
        return;
    }

    // pass 1: compute z, find zmin/zmax
    std::vector<double> zbuf((size_t)nrows, 0.0);
    std::vector<uint8_t> finite((size_t)nrows, 1u);
    zmin = std::numeric_limits<double>::infinity();
    zmax = -std::numeric_limits<double>::infinity();

    for (int rr = 0; rr < nrows; ++rr) {
        const int i = rows[rr];
        bool miss = false;
        double acc = 0.0;
        for (size_t j = 0; j < S.size(); ++j) {
            const int f = S[j];
            const uint16_t code = Xb[(size_t)i * (size_t)P + (size_t)f];
            const double x_val = x_from_code_variable(f, code, ctx);
            if (!std::isfinite(x_val)) {
                miss = true;
                break;
            }
            acc += w[j] * x_val;
        }
        if (miss) {
            finite[(size_t)rr] = 0u;
            Gm += (double)g[i];
            Hm += (double)h[i];
            ++Cm;
        } else {
            zbuf[(size_t)rr] = acc;
            zmin = std::min(zmin, acc);
            zmax = std::max(zmax, acc);
        }
    }

    if (!(zmax > zmin)) {
        Gz.assign(Bz, 0.0);
        Hz.assign(Bz, 0.0);
        Cz.assign(Bz, 0);
        return;
    }

    // pass 2: bin finite z and accumulate G/H
    Gz.assign((size_t)Bz, 0.0);
    Hz.assign((size_t)Bz, 0.0);
    Cz.assign((size_t)Bz, 0);
    const double invw = (double)Bz / (zmax - zmin);

    for (int rr = 0; rr < nrows; ++rr) {
        if (!finite[(size_t)rr])
            continue;
        const int i = rows[rr];
        const double z = zbuf[(size_t)rr];
        int bz = (int)std::floor((z - zmin) * invw);
        if (bz < 0)
            bz = 0;
        else if (bz >= Bz)
            bz = Bz - 1;
        Gz[(size_t)bz] += (double)g[i];
        Hz[(size_t)bz] += (double)h[i];
        Cz[(size_t)bz] += 1;
    }
}

// ============================================================================
// 3) UPDATED Oblique splitter with variable bin support
// ============================================================================
class ObliqueSplitFinder {
public:
    int k_features = 6; // pick top-k by |corr(x,g)|
    double ridge = 1e-3;

    // Row-wise (exact) oblique - unchanged since it doesn't use bins
    Candidate best_oblique(const SplitContext &ctx,
                           double axis_guard_gain = -1.0 /*optional*/) const {
        Candidate out;
        out.kind = SplitKind::Oblique;
        out.gain = detail::NEG_INF;

        if (!ctx.Xcols || ctx.N <= 0 || !ctx.row_g || !ctx.row_h)
            return out;

        const int P = ctx.P, N = ctx.N;
        if (P <= 1)
            return out;

        // 1) Rank features by |corr(x, g)|
        std::vector<double> corr(P, 0.0);
        double mg = 0.0;
        for (int i = 0; i < N; ++i)
            mg += (double)ctx.row_g[i];
        mg /= std::max(1, N);

        for (int f = 0; f < P; ++f) {
            const double *x = ctx.Xcols[f];
            double cnt = 0.0, sx = 0.0;
            for (int i = 0; i < N; ++i) {
                const double xi = x[i];
                if (std::isfinite(xi)) {
                    sx += xi;
                    cnt += 1.0;
                }
            }
            if (cnt < 2.0) {
                corr[f] = 0.0;
                continue;
            }
            const double mx = sx / cnt;
            double sxx = 0.0, sgg = 0.0, sxg = 0.0;
            for (int i = 0; i < N; ++i) {
                const double xi = x[i];
                if (!std::isfinite(xi))
                    continue;
                const double dx = xi - mx, dg = (double)ctx.row_g[i] - mg;
                sxx += dx * dx;
                sgg += dg * dg;
                sxg += dx * dg;
            }
            const double denom = std::sqrt(sxx * sgg) + detail::EPS;
            corr[f] = (denom <= detail::EPS) ? 0.0 : std::abs(sxg / denom);
        }

        std::vector<int> ord(P);
        std::iota(ord.begin(), ord.end(), 0);
        const int k = std::min(k_features, P);
        std::partial_sort(ord.begin(), ord.begin() + k, ord.end(),
                          [&](int a, int b) { return corr[a] > corr[b]; });
        std::vector<int> S(ord.begin(), ord.begin() + k);
        if ((int)S.size() < 2)
            return out;

        // 2) Build normal equations on S and solve for w
        std::vector<const double *> XS;
        XS.reserve(S.size());
        for (int f : S)
            XS.push_back(ctx.Xcols[f]);

        std::vector<double> A, b, w;
        build_normal_eq_cols(XS, ctx.row_g, ctx.row_h, N,
                             ridge + ctx.hyp.lambda_, A, b);
        if (!cholesky_spd(A, (int)S.size()))
            return out;
        chol_solve_inplace(A, (int)S.size(), b, w);

        // 3) Project z = w^T x and mark missing
        std::vector<double> z(N, 0.0);
        std::vector<uint8_t> miss(N, 0);
        for (int i = 0; i < N; ++i) {
            double acc = 0.0;
            bool finite = true;
            for (size_t j = 0; j < S.size(); ++j) {
                const double xi = XS[j][i];
                if (!std::isfinite(xi)) {
                    finite = false;
                    break;
                }
                acc += xi * w[j];
            }
            z[i] = acc;
            miss[i] = finite ? 0u : 1u;
        }

        // 4) best threshold along z (try missing left/right)
        auto [bgain, bthr, bmleft] =
            best_split_on_projection(z, ctx.row_g, ctx.row_h, N, miss, ctx.hyp);
        if (!(bgain > 0.0))
            return out;

        // 5) Package
        out.kind = SplitKind::Oblique;
        out.gain = bgain;
        out.oblique_features = std::move(S);
        out.oblique_weights = std::move(w);
        out.oblique_threshold = bthr;
        out.oblique_bias = 0.0;
        out.oblique_missing_left = bmleft;
        return out;
    }

    // UPDATED: Histogram-backed oblique with variable bin support
    Candidate best_oblique_hist(const SplitContext &ctx) const {
        Candidate out;
        out.kind = SplitKind::Oblique;
        out.gain = detail::NEG_INF;

        // Need: Xb, row_index, bin_centers, row_g/h
        if (!ctx.Xb || !ctx.row_index || !ctx.bin_centers || !ctx.row_g ||
            !ctx.row_h)
            return out;

        const int P = ctx.P, Nn = ctx.N;
        if (P <= 1 || Nn <= 0)
            return out;

        // 1) rank features by |corr(x, g)| approx using bin centers
        std::vector<double> score(P, 0.0);
        for (int f = 0; f < P; ++f) {
            double sx = 0.0, sxx = 0.0, sg = 0.0, sgg = 0.0, sxg = 0.0;
            int n = 0;
            for (int rr = 0; rr < Nn; ++rr) {
                const int i = ctx.row_index[rr];
                const uint16_t code = ctx.Xb[(size_t)i * (size_t)P + (size_t)f];
                const double x = x_from_code_variable(f, code, ctx);
                if (!std::isfinite(x))
                    continue;
                const double gi = (double)ctx.row_g[i];
                sx += x;
                sxx += x * x;
                sg += gi;
                sgg += gi * gi;
                sxg += x * gi;
                ++n;
            }
            if (n >= 2) {
                const double invn = 1.0 / (double)n;
                const double mx = sx * invn, mg = sg * invn;
                const double cov = sxg * invn - mx * mg;
                const double varx = sxx * invn - mx * mx;
                const double varg = sgg * invn - mg * mg;
                const double denom =
                    std::sqrt(std::max(0.0, varx) * std::max(0.0, varg)) +
                    detail::EPS;
                score[(size_t)f] = std::abs(cov) / denom;
            }
        }

        std::vector<int> ord(P);
        std::iota(ord.begin(), ord.end(), 0);
        const int k = std::min(k_features, P);
        std::partial_sort(
            ord.begin(), ord.begin() + k, ord.end(),
            [&](int a, int b) { return score[(size_t)a] > score[(size_t)b]; });
        std::vector<int> S(ord.begin(), ord.begin() + k);
        if ((int)S.size() < 2)
            return out;

        // 2) normal equations on codes (ridge includes lambda)
        std::vector<double> A, b, w;
        build_normal_eq_from_codes(S, ctx.Xb, ctx.row_index, Nn, P, ctx,
                                   ctx.row_g, ctx.row_h,
                                   ridge + ctx.hyp.lambda_, A, b);
        if (!cholesky_spd(A, (int)S.size()))
            return out;
        chol_solve_inplace(A, (int)S.size(), b, w);

        // 3) build z-hist (1D) and missing aggregate
        std::vector<double> Gz, Hz;
        std::vector<int> Cz;
        double Gm = 0.0, Hm = 0.0;
        int Cm = 0;
        double zmin = 0.0, zmax = 1.0;
        
        build_projection_hist_from_codes(
            S, w, ctx.Xb, ctx.row_index, Nn, P, ctx,
            ctx.row_g, ctx.row_h, Gz, Hz, Cz, Gm, Hm, Cm, zmin, zmax);

        // If no spread along z, bail
        bool allG0 = std::all_of(Gz.begin(), Gz.end(),
                                 [](double v) { return v == 0.0; });
        bool allH0 = std::all_of(Hz.begin(), Hz.end(),
                                 [](double v) { return v == 0.0; });
        if (allG0 && allH0)
            return out;

        // 4) axis-like scan along z bins (treat missing left/right)
        const double Gtot = std::accumulate(Gz.begin(), Gz.end(), 0.0) + Gm;
        const double Htot = std::accumulate(Hz.begin(), Hz.end(), 0.0) + Hm;
        const int Ctot = std::accumulate(Cz.begin(), Cz.end(), 0);
        const double parent = (Gtot * Gtot) / (Htot + ctx.hyp.lambda_);
        const int Bz = static_cast<int>(Gz.size());

        auto scan_dir = [&](bool miss_left) -> std::pair<double, int> {
            double GL = 0.0, HL = 0.0;
            int CL = 0;
            double best_gain = detail::NEG_INF;
            int best_t = -1;

            for (int t = 0; t < Bz - 1; ++t) {
                GL += Gz[(size_t)t];
                HL += Hz[(size_t)t];
                CL += Cz[(size_t)t];

                const double GLx = GL + (miss_left ? Gm : 0.0);
                const double HLx = HL + (miss_left ? Hm : 0.0);
                const double GRx = Gtot - GLx;
                const double HRx = Htot - HLx;

                const int CLx = CL + (miss_left ? Cm : 0);
                const int CRx = (Ctot - CL) + (miss_left ? 0 : Cm);

                if (CLx < ctx.hyp.min_samples_leaf ||
                    CRx < ctx.hyp.min_samples_leaf)
                    continue;
                if (HLx < ctx.hyp.min_child_weight ||
                    HRx < ctx.hyp.min_child_weight)
                    continue;

                const double child = (GLx * GLx) / (HLx + ctx.hyp.lambda_) +
                                     (GRx * GRx) / (HRx + ctx.hyp.lambda_);
                const double gain = 0.5 * (child - parent) - ctx.hyp.gamma_;
                if (gain > best_gain) {
                    best_gain = gain;
                    best_t = t;
                }
            }
            return {best_gain, best_t};
        };

        auto [gL, tL] = scan_dir(true);
        auto [gR, tR] = scan_dir(false);
        const bool miss_left = (gL >= gR);
        const double gain = std::max(gL, gR);
        const int t = miss_left ? tL : tR;
        if (t < 0 || !(gain > 0.0))
            return out;

        // 5) convert bin id -> numeric threshold
        const double dz = (zmax - zmin) / (double)Bz;
        const double thr = zmin + dz * (t + 1); // boundary between t and t+1

        // 6) package candidate
        out.kind = SplitKind::Oblique;
        out.gain = gain;
        out.oblique_features = std::move(S);
        out.oblique_weights = std::move(w);
        out.oblique_threshold = thr;
        out.oblique_bias = 0.0;
        out.oblique_missing_left = miss_left;
        return out;
    }
};

// ============================================================================
// Rest of the classes can remain largely unchanged since they use the same
// helper functions and SplitContext methods
// ============================================================================

struct InteractionSeededConfig {
    int pairs = 5;               // max pairs to evaluate
    int max_top_features = 8;    // among corr-ranked features
    int max_var_candidates = 16; // first stage variance pre-filter
    int first_i_cap = 4;         // i < first_i_cap in pair loop
    int second_j_cap = 8;        // j < second_j_cap in pair loop
    double ridge = 1e-3;         // ridge added to ctx.hyp.lambda_
    double axis_guard_factor =
        1.02; // skip if axis_gain * guard >= oblique_gain
    bool use_axis_guard = true;
};

// The rest of the InteractionSeededObliqueFinder class remains unchanged
// since it operates on row-level data and doesn't use histogram bins

namespace detail {

inline bool is_fin(double x) { return std::isfinite(x); }

// variance of a column ignoring NaNs; returns {count, mean, var}
inline std::tuple<int, double, double> col_var_ignore_nan(const double *x,
                                                          int n) {
    int cnt = 0;
    double sx = 0.0;
    for (int i = 0; i < n; ++i) {
        const double xi = x[i];
        if (is_fin(xi)) {
            ++cnt;
            sx += xi;
        }
    }
    if (cnt < 2)
        return {cnt, 0.0, 0.0};
    const double mx = sx / (double)cnt;
    double sxx = 0.0;
    for (int i = 0; i < n; ++i) {
        const double xi = x[i];
        if (!is_fin(xi))
            continue;
        const double dx = xi - mx;
        sxx += dx * dx;
    }
    const double var = sxx / (double)cnt;
    return {cnt, mx, var};
}

// |corr(x,g)| per column, ignoring NaNs in x;
// use global mean of g (like the Python kernel), but accumulate only where x is
// finite
inline std::vector<double> abs_corr_cols_ignore_nan(const double *const *Xcols,
                                                    int N, int P,
                                                    const float *g) {
    std::vector<double> out((size_t)P, 0.0);

    // global mean of g
    double sg = 0.0;
    for (int i = 0; i < N; ++i)
        sg += (double)g[i];
    const double mg = (N > 0 ? sg / (double)N : 0.0);

    // first pass: counts and means
    std::vector<int> cnt((size_t)P, 0);
    std::vector<double> sx((size_t)P, 0.0);
    for (int j = 0; j < P; ++j) {
        const double *x = Xcols[j];
        int c = 0;
        double s = 0.0;
        for (int i = 0; i < N; ++i) {
            const double xi = x[i];
            if (is_fin(xi)) {
                ++c;
                s += xi;
            }
        }
        cnt[(size_t)j] = c;
        sx[(size_t)j] = s;
    }

    // second pass: corr components
    for (int j = 0; j < P; ++j) {
        const int c = cnt[(size_t)j];
        if (c < 2) {
            out[(size_t)j] = 0.0;
            continue;
        }
        const double *x = Xcols[j];
        const double mx = sx[(size_t)j] / (double)c;
        double sxx = 0.0, sgg = 0.0, sxg = 0.0;
        for (int i = 0; i < N; ++i) {
            const double xi = x[i];
            if (!is_fin(xi))
                continue;
            const double dx = xi - mx;
            const double dg = (double)g[i] - mg;
            sxx += dx * dx;
            sgg += dg * dg;
            sxg += dx * dg;
        }
        const double denom = std::sqrt(sxx * sgg) + 1e-12;
        out[(size_t)j] = (denom < 1e-12 ? 0.0 : std::abs(sxg / denom));
    }
    return out;
}

// Build 2x2 normal equations for (X^T H X + (ridge+λ)I)w = -X^T g,
// ignoring rows where xi or yi is non-finite.
inline void build_2x2_Ab(const double *x1, const double *x2, const float *g,
                         const float *h, int N, double ridge_plus_lambda,
                         double &A00, double &A01, double &A11, double &b0,
                         double &b1, int &n_finite_rows) {
    A00 = A01 = A11 = 0.0;
    b0 = b1 = 0.0;
    n_finite_rows = 0;
    for (int i = 0; i < N; ++i) {
        const double xi = x1[i], yi = x2[i];
        if (!is_fin(xi) && !is_fin(yi))
            continue; // both missing => skip
        const double gi = (double)g[i];
        const double hi = (double)h[i];
        if (is_fin(xi)) {
            b0 -= xi * gi;
            A00 += hi * xi * xi;
        }
        if (is_fin(yi)) {
            b1 -= yi * gi;
            A11 += hi * yi * yi;
        }
        if (is_fin(xi) && is_fin(yi)) {
            A01 += hi * xi * yi;
            ++n_finite_rows;
        }
    }
    A00 += ridge_plus_lambda;
    A11 += ridge_plus_lambda;
}

// Solve 2x2 system; returns false if nearly singular.
inline bool solve_2x2(double A00, double A01, double A11, double b0, double b1,
                      double &w0, double &w1) {
    const double det = A00 * A11 - A01 * A01;
    if (!(std::abs(det) > 1e-18))
        return false;
    const double inv = 1.0 / det;
    w0 = (A11 * b0 - A01 * b1) * inv;
    w1 = (-A01 * b0 + A00 * b1) * inv;
    return std::isfinite(w0) && std::isfinite(w1);
}

// Best split along z with missing→left/right search and child guards.
// Returns {gain, thr, miss_left}.
inline std::tuple<double, double, bool>
best_split_on_projection(const std::vector<double> &z,
                         const std::vector<uint8_t> &miss, const float *g,
                         const float *h, int N, const SplitHyper &hyp) {
    // Collect finite slice and missing aggregates
    std::vector<int> idx;
    idx.reserve((size_t)N);
    double gm = 0.0, hm = 0.0;
    int Cm = 0;
    for (int i = 0; i < N; ++i) {
        if (miss[(size_t)i]) {
            gm += (double)g[i];
            hm += (double)h[i];
            ++Cm;
        } else {
            idx.push_back(i);
        }
    }
    const int nf = (int)idx.size();
    if (nf < 4)
        return {-1.0, 0.0, true};

    // sort by z (finite only)
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return z[(size_t)a] < z[(size_t)b]; });

    // prefix sums on sorted finite
    std::vector<double> pg((size_t)nf), ph((size_t)nf);
    for (int k = 0; k < nf; ++k) {
        const int i = idx[(size_t)k];
        pg[(size_t)k] = (double)g[i] + (k ? pg[(size_t)k - 1] : 0.0);
        ph[(size_t)k] = (double)h[i] + (k ? ph[(size_t)k - 1] : 0.0);
    }

    const double gtot = pg[(size_t)nf - 1] + gm;
    const double htot = ph[(size_t)nf - 1] + hm;
    const double parent = (gtot * gtot) / (htot + hyp.lambda_);

    double best_gain = -1.0, best_thr = 0.0;
    bool best_mleft = true;

    for (int k = 1; k < nf; ++k) {
        const int il = idx[(size_t)k - 1], ir = idx[(size_t)k];
        const double zl = z[(size_t)il], zr = z[(size_t)ir];
        if (!(zr > zl + 1e-15))
            continue; // tie

        const double thr = 0.5 * (zl + zr);
        const double glb = pg[(size_t)k - 1], hlb = ph[(size_t)k - 1];
        const double grb = pg[(size_t)nf - 1] - glb;
        const double hrb = ph[(size_t)nf - 1] - hlb;

        auto eval_dir = [&](bool miss_left) {
            const double HL = hlb + (miss_left ? hm : 0.0);
            const double HR = hrb + (miss_left ? 0.0 : hm);
            const int nL = k + (miss_left ? Cm : 0);
            const int nR = (nf + Cm) - nL;
            if (nL < hyp.min_samples_leaf || nR < hyp.min_samples_leaf)
                return -std::numeric_limits<double>::infinity();
            if (HL < hyp.min_child_weight || HR < hyp.min_child_weight)
                return -std::numeric_limits<double>::infinity();
            const double GL = glb + (miss_left ? gm : 0.0);
            const double GR = grb + (miss_left ? 0.0 : gm);
            const double child =
                (GL * GL) / (HL + hyp.lambda_) + (GR * GR) / (HR + hyp.lambda_);
            return 0.5 * (child - parent) - hyp.gamma_;
        };

        const double gL = eval_dir(true);
        const double gR = eval_dir(false);
        if (gL > best_gain) {
            best_gain = gL;
            best_thr = thr;
            best_mleft = true;
        }
        if (gR > best_gain) {
            best_gain = gR;
            best_thr = thr;
            best_mleft = false;
        }
    }
    return {best_gain, best_thr, best_mleft};
}

} // namespace detail

class InteractionSeededObliqueFinder {
public:
    InteractionSeededObliqueFinder() = default;

    // Returns the best oblique candidate formed from 2-feature interactions.
    // If no valid candidate, returns gain = -inf and thr < 0.
    Candidate best_oblique_interaction(const SplitContext &ctx,
                                       const InteractionSeededConfig &cfg,
                                       double axis_guard_gain = -1.0) const {
        Candidate best;
        best.kind = SplitKind::Oblique;
        best.gain = -std::numeric_limits<double>::infinity();

        // Need row-level access
        if (!ctx.Xcols || !ctx.row_g || !ctx.row_h || ctx.N <= 0 || ctx.P <= 1)
            return best;

        const int N = ctx.N, P = ctx.P;

        // 1) variance pre-filter (top max_var_candidates)
        std::vector<int> cand;
        cand.reserve((size_t)P);
        cand.resize((size_t)P);
        std::iota(cand.begin(), cand.end(), 0);

        // compute variance per feature ignoring NaNs
        std::vector<double> var((size_t)P, 0.0);
        for (int f = 0; f < P; ++f) {
            const double *x = ctx.Xcols[f];
            int cnt;
            double mx, v;
            std::tie(cnt, mx, v) = detail::col_var_ignore_nan(x, N);
            var[(size_t)f] = (cnt < 2 ? 0.0 : v);
        }
        std::partial_sort(
            cand.begin(), cand.begin() + std::min(cfg.max_var_candidates, P),
            cand.end(),
            [&](int a, int b) { return var[(size_t)a] > var[(size_t)b]; });
        cand.resize(std::min(cfg.max_var_candidates, P));

        if ((int)cand.size() < 2)
            return best;

        // 2) correlation ranking on the shortlisted features
        //    compute |corr(x,g)| for all columns, then keep order among 'cand'
        const auto corr_all =
            detail::abs_corr_cols_ignore_nan(ctx.Xcols, N, P, ctx.row_g);

        std::sort(cand.begin(), cand.end(), [&](int a, int b) {
            return corr_all[(size_t)a] > corr_all[(size_t)b];
        });

        // final shortlist for pair formation
        const int shortlist = std::min(cfg.max_top_features, (int)cand.size());
        if (shortlist < 2)
            return best;

        // 3) generate limited pairs (like Python: i in [0..min(4,..)), j in
        // (i+1..min(8,..))
        std::vector<std::pair<int, int>> pairs;
        pairs.reserve((size_t)cfg.pairs);
        const int i_cap = std::min(cfg.first_i_cap, shortlist);
        const int j_cap = std::min(cfg.second_j_cap, shortlist);
        for (int ii = 0; ii < i_cap && (int)pairs.size() < cfg.pairs; ++ii) {
            for (int jj = ii + 1; jj < j_cap && (int)pairs.size() < cfg.pairs;
                 ++jj) {
                pairs.emplace_back(cand[(size_t)ii], cand[(size_t)jj]);
            }
        }
        if (pairs.empty())
            return best;

        const double ridge_plus_lambda = cfg.ridge + ctx.hyp.lambda_;

        // 4) evaluate each pair
        for (auto [fa, fb] : pairs) {
            const double *x1 = ctx.Xcols[fa];
            const double *x2 = ctx.Xcols[fb];

            double A00, A01, A11, b0, b1;
            int nfinite = 0;
            detail::build_2x2_Ab(x1, x2, ctx.row_g, ctx.row_h, N,
                                 ridge_plus_lambda, A00, A01, A11, b0, b1,
                                 nfinite);
            if (nfinite < 2)
                continue;

            double w0, w1;
            if (!detail::solve_2x2(A00, A01, A11, b0, b1, w0, w1))
                continue;

            // project z and build missing mask (missing if any coord
            // non-finite)
            std::vector<double> z((size_t)N, 0.0);
            std::vector<uint8_t> miss((size_t)N, 0u);
            for (int i = 0; i < N; ++i) {
                const double xi = x1[i], yi = x2[i];
                const bool finite = detail::is_fin(xi) && detail::is_fin(yi);
                miss[(size_t)i] = finite ? 0u : 1u;
                if (finite)
                    z[(size_t)i] = w0 * xi + w1 * yi;
            }

            auto [gain, thr, mleft] = detail::best_split_on_projection(
                z, miss, ctx.row_g, ctx.row_h, N, ctx.hyp);

            if (gain <= 0.0)
                continue;

            // optional axis guard
            if (cfg.use_axis_guard && axis_guard_gain > 0.0 &&
                axis_guard_gain * cfg.axis_guard_factor >= gain) {
                continue;
            }

            if (gain > best.gain) {
                best.kind = SplitKind::Oblique;
                best.gain = gain;
                best.oblique_features = {fa, fb};
                best.oblique_weights = {w0, w1};
                best.oblique_bias = 0.0;
                best.oblique_threshold = thr;
                best.oblique_missing_left = mleft;
            }
        }

        return best;
    }
};

} // namespace foretree