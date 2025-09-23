#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "split_helpers.h"

namespace foretree {

// ============================================================================
// 1) Axis (histogram/exact) — uses splitx providers & scanners
// ============================================================================
class AxisSplitFinder {
public:
    // Histogram-backed axis split (variable bins supported)
    splitx::Candidate best_axis(const splitx::SplitContext &ctx) const {
        using namespace splitx;
        Candidate best; best.kind = SplitKind::Axis; best.gain = NEG_INF;
        if (!ctx.G || !ctx.H || !ctx.C || ctx.P <= 0) return best;

        const auto &G = *ctx.G; const auto &H = *ctx.H; const auto &C = *ctx.C;
        const auto *mono_vec = ctx.mono_ptr();
        const double parent_gain = leaf_obj(ctx.Gp, ctx.Hp, ctx.hyp.lambda_, ctx.hyp.alpha_);

        for (int f = 0; f < ctx.P; ++f) {
            const int finite_bins = ctx.get_feature_bins(f);
            if (finite_bins <= 0) continue;

            const int8_t mono =
                (mono_vec && f < (int)mono_vec->size()) ? (*mono_vec)[f] : 0;

            HistProvider prov(ctx, f, G, H, C);

            double Gm=0.0, Hm=0.0; int Cm=0; bool has_miss=false;
            prov.missing(Gm,Hm,Cm,has_miss);
            const int totalC = prov.total_count();
            const int steps  = prov.steps();

            Candidate cand = scan_axis_with_policy(
                ctx, f, prov, steps, mono, ctx.hyp.missing_policy,
                parent_gain, totalC, Gm, Hm, Cm, has_miss);

            if (cand.thr >= 0 && cand.gain > best.gain) best = cand;
        }
        return best;
    }

    // Exact-mode axis split (raw X + node index + missing policy)
    splitx::Candidate best_axis_exact(const splitx::SplitContext &ctx,
                              const float *Xraw, int P,
                              const int *node_idx, int nidx,
                              int missing_policy,
                              const uint8_t *miss_mask) const
    {
        using namespace splitx;
        Candidate best; best.kind = SplitKind::Axis; best.gain = NEG_INF; best.thr = -1;
        if (!Xraw || !ctx.row_g || !ctx.row_h || !node_idx || nidx <= 1 || P <= 0) return best;

        const auto *mono_vec = ctx.mono_ptr();
        const double parent_gain = leaf_obj(ctx.Gp, ctx.Hp, ctx.hyp.lambda_, ctx.hyp.alpha_);
        ExactProvider prov(ctx, Xraw, P, node_idx, nidx, miss_mask);

        for (int f = 0; f < P; ++f) {
            const int n_valid = prov.prepare_feature(f);
            if (n_valid < 2) continue;

            double Gm=0.0, Hm=0.0; int Cm=0; bool has_miss=false;
            prov.missing(Gm,Hm,Cm,has_miss);
            if (ctx.Gmiss) Gm = ctx.Gmiss[f];
            if (ctx.Hmiss) Hm = ctx.Hmiss[f];

            const int totalC = prov.total_count_for_nvalid(n_valid);
            const int8_t mono =
                (mono_vec && f < (int)mono_vec->size()) ? (*mono_vec)[f] : 0;
            const int steps = prov.steps_for_nvalid(n_valid);

            const int mpol = (missing_policy == 0) ? ctx.hyp.missing_policy : missing_policy;
            Candidate cand = scan_axis_with_policy(
                ctx, f, prov, steps, mono, mpol,
                parent_gain, totalC, Gm, Hm, Cm, has_miss);

            if (cand.thr >= 0 && cand.gain > best.gain) best = cand;
        }
        return best;
    }
};

// ============================================================================
// 2) Categorical K-way from histograms with variable bins
// ============================================================================
class CategoricalKWaySplitFinder {
public:
    int max_groups = 8; // cap #groups we assemble internally

    splitx::Candidate best_kway(const splitx::SplitContext &ctx) const {
        using namespace splitx;
        Candidate best; best.kind = SplitKind::KWay; best.gain = NEG_INF;
        if (!ctx.G || !ctx.H || !ctx.C || ctx.P <= 0) return best;

        const auto &G = *ctx.G; const auto &H = *ctx.H;
        const double lam = ctx.hyp.lambda_, gamma = ctx.hyp.gamma_;
        const double parent = (ctx.Gp * ctx.Gp) / (ctx.Hp + lam);

        for (int f = 0; f < ctx.P; ++f) {
            const int finite_bins = ctx.get_feature_bins(f);
            const int miss_id     = ctx.get_missing_bin_id(f);
            if (finite_bins <= 1) continue;

            int non_empty = 0;
            for (int t = 0; t < finite_bins; ++t) {
                const size_t bin_off = ctx.get_histogram_offset(f, t);
                if (bin_off >= G.size()) continue;
                const double g = G[bin_off], h = H[bin_off];
                if (g != 0.0 || h > 0.0) ++non_empty;
            }
            if (non_empty < 2) continue;
            if (non_empty > 32 && non_empty > finite_bins / 2) continue; // heuristic

            std::vector<std::pair<double, int>> scored;
            scored.reserve((size_t)finite_bins);
            for (int t = 0; t < finite_bins; ++t) {
                const size_t off = ctx.get_histogram_offset(f, t);
                if (off >= G.size()) continue;
                const double g = G[off], h = H[off];
                if (g != 0.0 || h > 0.0) {
                    const double s = std::abs(g) / (h + lam + EPS);
                    scored.emplace_back(s, t);
                }
            }
            if (scored.size() < 2) continue;
            std::sort(scored.begin(), scored.end(),
                      [](const auto &a, const auto &b) { return a.first > b.first; });

            std::vector<std::vector<int>> groups;
            const int cap = std::max(2, max_groups);
            const int singles = (int)std::min<size_t>(cap - 1, scored.size());
            for (int i = 0; i < singles - 1; ++i) groups.push_back({scored[i].second});

            std::vector<int> tail;
            for (size_t k = std::max(0, singles - 1); k < scored.size(); ++k)
                tail.push_back(scored[k].second);
            if (!tail.empty()) groups.push_back(std::move(tail));
            if (groups.size() < 2) continue;

            const std::vector<int> &left = groups[0];

            std::vector<double> Hgroup(groups.size(), 0.0);
            for (size_t gi = 0; gi < groups.size(); ++gi) {
                double hs = 0.0;
                for (int t : groups[gi]) {
                    const size_t off = ctx.get_histogram_offset(f, t);
                    if (off < H.size()) hs += H[off];
                }
                Hgroup[gi] = hs;
            }
            const int mg = (int)std::distance(
                Hgroup.begin(), std::max_element(Hgroup.begin(), Hgroup.end()));
            const bool missing_left = (mg == 0);

            double GL = 0.0, HL = 0.0;
            for (int t : left) {
                const size_t off = ctx.get_histogram_offset(f, t);
                if (off < G.size()) { GL += G[off]; HL += H[off]; }
            }

            const size_t miss_off = ctx.get_histogram_offset(f, miss_id);
            const double Gm = (miss_off < G.size()) ? G[miss_off] : 0.0;
            const double Hm = (miss_off < H.size()) ? H[miss_off] : 0.0;
            if (missing_left) { GL += Gm; HL += Hm; }

            const double GR = ctx.Gp - GL, HR = ctx.Hp - HL;
            if (HL < ctx.hyp.min_child_weight || HR < ctx.hyp.min_child_weight) continue;

            const double child = (GL * GL) / (HL + lam) + (GR * GR) / (HR + lam);
            const double gain  = 0.5 * (child - parent) - gamma;

            if (gain > best.gain) {
                best.kind = SplitKind::KWay; best.gain = gain; best.feat = f;
                best.left_groups = left; best.missing_group = mg; best.miss_left = missing_left;
            }
        }
        return best;
    }
};

// ============================================================================
// 3) Oblique splitter (row-wise and hist) with variable bin support
// ============================================================================
class ObliqueSplitFinder {
public:
    int k_features = 6; // pick top-k by |corr(x,g)|
    double ridge = 1e-3;

    // Row-wise (exact) oblique
    splitx::Candidate best_oblique(const splitx::SplitContext &ctx,
                                   double /*axis_guard_gain*/ = -1.0) const {
        using namespace splitx;
        Candidate out; out.kind = SplitKind::Oblique; out.gain = NEG_INF;
        if (!ctx.Xcols || ctx.N <= 0 || !ctx.row_g || !ctx.row_h) return out;

        const int P = ctx.P, N = ctx.N;
        if (P <= 1) return out;

        // 1) Rank features by |corr(x, g)|
        std::vector<double> corr(P, 0.0);
        double mg = 0.0; for (int i = 0; i < N; ++i) mg += (double)ctx.row_g[i];
        mg /= std::max(1, N);

        for (int f = 0; f < P; ++f) {
            const double *x = ctx.Xcols[f];
            double cnt = 0.0, sx = 0.0;
            for (int i = 0; i < N; ++i) { const double xi = x[i]; if (std::isfinite(xi)) { sx += xi; cnt += 1.0; } }
            if (cnt < 2.0) { corr[f] = 0.0; continue; }
            const double mx = sx / cnt;
            double sxx = 0.0, sgg = 0.0, sxg = 0.0;
            for (int i = 0; i < N; ++i) {
                const double xi = x[i]; if (!std::isfinite(xi)) continue;
                const double dx = xi - mx, dg = (double)ctx.row_g[i] - mg;
                sxx += dx * dx; sgg += dg * dg; sxg += dx * dg;
            }
            const double denom = std::sqrt(sxx * sgg) + EPS;
            corr[f] = (denom <= EPS) ? 0.0 : std::abs(sxg / denom);
        }

        std::vector<int> ord(P); std::iota(ord.begin(), ord.end(), 0);
        const int k = std::min(k_features, P);
        std::partial_sort(ord.begin(), ord.begin() + k, ord.end(),
                          [&](int a, int b) { return corr[a] > corr[b]; });
        std::vector<int> S(ord.begin(), ord.begin() + k);
        if ((int)S.size() < 2) return out;

        // 2) Build normal equations on S and solve for w
        std::vector<const double *> XS; XS.reserve(S.size());
        for (int f : S) XS.push_back(ctx.Xcols[f]);

        std::vector<double> A, b, w;
        build_normal_eq_cols(XS, ctx.row_g, ctx.row_h, N,
                             ridge + ctx.hyp.lambda_, A, b);
        if (!cholesky_spd(A, (int)S.size())) return out;
        chol_solve_inplace(A, (int)S.size(), b, w);

        // 3) Project z = w^T x and mark missing
        std::vector<double> z(N, 0.0);
        std::vector<uint8_t> miss(N, 0);
        for (int i = 0; i < N; ++i) {
            double acc = 0.0; bool finite = true;
            for (size_t j = 0; j < S.size(); ++j) {
                const double xi = XS[j][i];
                if (!std::isfinite(xi)) { finite = false; break; }
                acc += xi * w[j];
            }
            z[i] = acc; miss[i] = finite ? 0u : 1u;
        }

        // 4) best threshold along z
        auto [bgain, bthr, bmleft] = best_split_on_projection(z, ctx.row_g, ctx.row_h, N, miss, ctx.hyp);
        if (!(bgain > 0.0)) return out;

        out.gain = bgain;
        out.oblique_features = std::move(S);
        out.oblique_weights  = std::move(w);
        out.oblique_threshold = bthr;
        out.oblique_bias = 0.0;
        out.oblique_missing_left = bmleft;
        return out;
    }

    // Histogram-backed oblique with variable bin support
    splitx::Candidate best_oblique_hist(const splitx::SplitContext &ctx) const {
        using namespace splitx;
        Candidate out; out.kind = SplitKind::Oblique; out.gain = NEG_INF;

        if (!ctx.Xb || !ctx.row_index || !ctx.bin_centers || !ctx.row_g || !ctx.row_h) return out;
        const int P = ctx.P, Nn = ctx.N;
        if (P <= 1 || Nn <= 0) return out;

        // 1) rank features by |corr(x, g)| approx using bin centers
        std::vector<double> score(P, 0.0);
        for (int f = 0; f < P; ++f) {
            double sx=0.0, sxx=0.0, sg=0.0, sgg=0.0, sxg=0.0; int n = 0;
            for (int rr = 0; rr < Nn; ++rr) {
                const int i = ctx.row_index[rr];
                const uint16_t code = ctx.Xb[(size_t)i * (size_t)P + (size_t)f];
                const double x = x_from_code_variable(f, code, ctx);
                if (!std::isfinite(x)) continue;
                const double gi = (double)ctx.row_g[i];
                sx += x; sxx += x * x; sg += gi; sgg += gi * gi; sxg += x * gi; ++n;
            }
            if (n >= 2) {
                const double invn = 1.0 / (double)n;
                const double mx = sx * invn, mg = sg * invn;
                const double cov = sxg * invn - mx * mg;
                const double varx = sxx * invn - mx * mx;
                const double varg = sgg * invn - mg * mg;
                const double denom = std::sqrt(std::max(0.0, varx) * std::max(0.0, varg)) + EPS;
                score[(size_t)f] = std::abs(cov) / denom;
            }
        }

        std::vector<int> ord(P); std::iota(ord.begin(), ord.end(), 0);
        const int k = std::min(k_features, P);
        std::partial_sort(ord.begin(), ord.begin() + k, ord.end(),
                          [&](int a, int b) { return score[(size_t)a] > score[(size_t)b]; });
        std::vector<int> S(ord.begin(), ord.begin() + k);
        if ((int)S.size() < 2) return out;

        // 2) normal equations on codes (ridge includes lambda)
        std::vector<double> A, b, w;
        build_normal_eq_from_codes(S, ctx.Xb, ctx.row_index, Nn, P, ctx,
                                   ctx.row_g, ctx.row_h,
                                   ridge + ctx.hyp.lambda_, A, b);
        if (!cholesky_spd(A, (int)S.size())) return out;
        chol_solve_inplace(A, (int)S.size(), b, w);

        // 3) build z-hist (1D) and missing aggregate
        std::vector<double> Gz, Hz; std::vector<int> Cz;
        double Gm = 0.0, Hm = 0.0; int Cm = 0; double zmin = 0.0, zmax = 1.0;

        build_projection_hist_from_codes(S, w, ctx.Xb, ctx.row_index, Nn, P,
                                         ctx, ctx.row_g, ctx.row_h, Gz, Hz, Cz,
                                         Gm, Hm, Cm, zmin, zmax);

        bool allG0 = std::all_of(Gz.begin(), Gz.end(), [](double v){ return v == 0.0; });
        bool allH0 = std::all_of(Hz.begin(), Hz.end(), [](double v){ return v == 0.0; });
        if (allG0 && allH0) return out;

        const double Gtot = std::accumulate(Gz.begin(), Gz.end(), 0.0) + Gm;
        const double Htot = std::accumulate(Hz.begin(), Hz.end(), 0.0) + Hm;
        const int    Ctot = std::accumulate(Cz.begin(), Cz.end(), 0);
        const double parent = (Gtot * Gtot) / (Htot + ctx.hyp.lambda_);
        const int Bz = (int)Gz.size();

        auto scan_dir = [&](bool mleft) -> std::pair<double,int> {
            double GL=0.0, HL=0.0; int CL=0;
            double best_gain = splitx::NEG_INF; int best_t = -1;
            for (int t = 0; t < Bz - 1; ++t) {
                GL += Gz[(size_t)t]; HL += Hz[(size_t)t]; CL += Cz[(size_t)t];
                const double GLx = GL + (mleft ? Gm : 0.0);
                const double HLx = HL + (mleft ? Hm : 0.0);
                const double GRx = Gtot - GLx;
                const double HRx = Htot - HLx;
                const int    CLx = CL + (mleft ? Cm : 0);
                const int    CRx = (Ctot - CL) + (mleft ? 0 : Cm);
                if (CLx < ctx.hyp.min_samples_leaf || CRx < ctx.hyp.min_samples_leaf) continue;
                if (HLx < ctx.hyp.min_child_weight || HRx < ctx.hyp.min_child_weight) continue;
                const double child = (GLx * GLx) / (HLx + ctx.hyp.lambda_) +
                                     (GRx * GRx) / (HRx + ctx.hyp.lambda_);
                const double gain = 0.5 * (child - parent) - ctx.hyp.gamma_;
                if (gain > best_gain) { best_gain = gain; best_t = t; }
            }
            return {best_gain, best_t};
        };

        auto [gL, tL] = scan_dir(true);
        auto [gR, tR] = scan_dir(false);
        const bool miss_left = (gL >= gR);
        const double gain = std::max(gL, gR);
        const int t = miss_left ? tL : tR;
        if (t < 0 || !(gain > 0.0)) return out;

        const double dz = (zmax - zmin) / (double)Bz;
        const double thr = zmin + dz * (t + 1);

        out.gain = gain;
        out.oblique_features = std::move(S);
        out.oblique_weights  = std::move(w);
        out.oblique_threshold = thr;
        out.oblique_bias = 0.0;
        out.oblique_missing_left = miss_left;
        return out;
    }
};

// ============================================================================
// 4) Interaction-seeded oblique (2-feature)
// ============================================================================
struct InteractionSeededConfig {
    int pairs = 5;
    int max_top_features = 8;
    int max_var_candidates = 16;
    int first_i_cap = 4;
    int second_j_cap = 8;
    double ridge = 1e-3;
    double axis_guard_factor = 1.02;
    bool use_axis_guard = true;
};

class InteractionSeededObliqueFinder {
public:
    InteractionSeededObliqueFinder() = default;

    splitx::Candidate best_oblique_interaction(const splitx::SplitContext &ctx,
                                               const InteractionSeededConfig &cfg,
                                               double axis_guard_gain = -1.0) const
    {
        using namespace splitx;
        Candidate best; best.kind = SplitKind::Oblique; best.gain = -std::numeric_limits<double>::infinity();

        if (!ctx.Xcols || !ctx.row_g || !ctx.row_h || ctx.N <= 0 || ctx.P <= 1) return best;

        const int N = ctx.N, P = ctx.P;

        std::vector<int> cand((size_t)P);
        std::iota(cand.begin(), cand.end(), 0);

        std::vector<double> var((size_t)P, 0.0);
        for (int f = 0; f < P; ++f) {
            const double *x = ctx.Xcols[f];
            int cnt; double mx, v;
            std::tie(cnt, mx, v) = col_var_ignore_nan(x, N);
            var[(size_t)f] = (cnt < 2 ? 0.0 : v);
        }
        std::partial_sort(
            cand.begin(), cand.begin() + std::min(cfg.max_var_candidates, P),
            cand.end(),
            [&](int a, int b) { return var[(size_t)a] > var[(size_t)b]; });
        cand.resize(std::min(cfg.max_var_candidates, P));
        if ((int)cand.size() < 2) return best;

        const auto corr_all = abs_corr_cols_ignore_nan(ctx.Xcols, N, P, ctx.row_g);
        std::sort(cand.begin(), cand.end(),
                  [&](int a, int b) { return corr_all[(size_t)a] > corr_all[(size_t)b]; });

        const int shortlist = std::min(cfg.max_top_features, (int)cand.size());
        if (shortlist < 2) return best;

        std::vector<std::pair<int, int>> pairs;
        pairs.reserve((size_t)cfg.pairs);
        const int i_cap = std::min(cfg.first_i_cap, shortlist);
        const int j_cap = std::min(cfg.second_j_cap, shortlist);
        for (int ii = 0; ii < i_cap && (int)pairs.size() < cfg.pairs; ++ii) {
            for (int jj = ii + 1; jj < j_cap && (int)pairs.size() < cfg.pairs; ++jj) {
                pairs.emplace_back(cand[(size_t)ii], cand[(size_t)jj]);
            }
        }
        if (pairs.empty()) return best;

        const double ridge_plus_lambda = cfg.ridge + ctx.hyp.lambda_;

        for (auto [fa, fb] : pairs) {
            const double *x1 = ctx.Xcols[fa];
            const double *x2 = ctx.Xcols[fb];

            double A00, A01, A11, b0, b1;
            int nfinite = 0;
            splitx::build_2x2_Ab(x1, x2, ctx.row_g, ctx.row_h, N,
                                 ridge_plus_lambda, A00, A01, A11, b0, b1,
                                 nfinite);
            if (nfinite < 2) continue;

            double w0, w1;
            if (!splitx::solve_2x2(A00, A01, A11, b0, b1, w0, w1)) continue;

            std::vector<double> z((size_t)N, 0.0);
            std::vector<uint8_t> miss((size_t)N, 0u);
            for (int i = 0; i < N; ++i) {
                const double xi = x1[i], yi = x2[i];
                const bool finite = splitx::is_fin(xi) && splitx::is_fin(yi);
                miss[(size_t)i] = finite ? 0u : 1u;
                if (finite) z[(size_t)i] = w0 * xi + w1 * yi;
            }

            auto [gain, thr, mleft] =
                splitx::best_split_on_projection_interact(z, miss, ctx.row_g, ctx.row_h, N, ctx.hyp);

            if (gain <= 0.0) continue;
            if (cfg.use_axis_guard && axis_guard_gain > 0.0 &&
                axis_guard_gain * cfg.axis_guard_factor >= gain) {
                continue;
            }

            if (gain > best.gain) {
                best.kind = SplitKind::Oblique; best.gain = gain;
                best.oblique_features = {fa, fb};
                best.oblique_weights  = {w0, w1};
                best.oblique_bias = 0.0;
                best.oblique_threshold = thr;
                best.oblique_missing_left = mleft;
            }
        }
        return best;
    }
};

} // namespace foretree
