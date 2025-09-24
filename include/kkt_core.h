#pragma once
// Optimized C++23 KKT core (HYKKT + LDL) with Eigen
// AVX guard fixed; ILU removed; Schur solves use CG + (Jacobi|SSOR).

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/OrderingMethods>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>

#ifdef EIGEN_CHOLMOD_SUPPORT
#include <Eigen/CholmodSupport> // CHOLMOD (SuiteSparse) via Eigen
#endif

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "qdldl.h"

#include "kkt_helper.h"

namespace kkt {

// ------------------------------ typedefs ------------------------------
using dvec = Eigen::VectorXd;
using dmat = Eigen::MatrixXd;
using spmat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

// ------------------------------ KKT assembly --------------------------

/**
 * ReusedK
 * --------
 * Keeps:
 *  - AMD permutation (optional)
 *  - Symbolic analysis for LLT or LDLT (chosen once, with automatic fallback)
 *
 * Usage:
 *   ReusedK RK;
 *   RK.analyze_once(K0, use_amd=true);   // first iteration
 *   ...
 *   RK.refactor(Kt);                         // next iterations (same sparsity)
 *   x = RK.solve(b);                         // solve K x = b
 *
 * Notes:
 *  - If LLT fails during analyze_once(), we switch to LDLT and lock into it.
 *  - During refactor(), if the locked solver fails, we throw (pattern likely
 * changed).
 *  - If you *know* SPD can sometimes break, just start with LDLT by calling
 *    analyze_once(..., ..., force_ldlt=true).
 */
struct ReusedK {
    // configuration
    bool use_amd = false;
    bool locked_ldlt = false; // once we fall back, we keep LDLT forever

    // permutation
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> P;

    // solvers (one active at a time)
    std::shared_ptr<Eigen::SimplicialLLT<spmat>> llt;
    std::shared_ptr<Eigen::SimplicialLDLT<spmat>> ldlt;

    // quick status flags
    bool analyzed = false;
    bool is_spd() const { return analyzed && !locked_ldlt; }

    // Clears all state (forces re-analyze next time)
    void reset() {
        use_amd = false;
        locked_ldlt = false;
        P.resize(0);
        llt.reset();
        ldlt.reset();
        analyzed = false;
    }

    // Optional: switch ordering mode before analyze
    void set_use_amd(bool v) { use_amd = v; }

    /**
     * analyze_once: run ordering + symbolic analysis exactly once.
     * @param K          matrix to analyze (pattern reference)
     * @param amd        whether to use AMD ordering
     * @param force_ldlt force LDLT (skip LLT attempt)
     */
    void analyze_once(const spmat &K, bool amd = true,
                      bool force_ldlt = false) {
        if (analyzed)
            return; // already done
        use_amd = amd;

        const int n = K.rows();
        if (K.cols() != n)
            throw std::invalid_argument("ReusedK: K must be square");

        // Build permutation
        if (use_amd) {
            Eigen::AMDOrdering<int> ord;
            ord(K, P);
        } else {
            P.setIdentity(n);
        }

        // Permuted pattern
        spmat Kp = use_amd ? spmat((P.transpose() * K) * P) : K;
        Kp.makeCompressed();

        if (!force_ldlt) {
            // Try LLT (SPD)
            llt = std::make_shared<Eigen::SimplicialLLT<spmat>>();
            llt->analyzePattern(Kp);
            llt->factorize(Kp);
            if (llt->info() == Eigen::Success) {
                ldlt.reset();
                locked_ldlt = false;
                analyzed = true;
                return;
            }
        }

        // Fallback to LDLT (indefinite-safe)
        ldlt = std::make_shared<Eigen::SimplicialLDLT<spmat>>();
        ldlt->analyzePattern(Kp);
        ldlt->factorize(Kp);
        if (ldlt->info() != Eigen::Success) {
            llt.reset();
            ldlt.reset();
            analyzed = false;
            throw std::runtime_error(
                "ReusedK::analyze_once: factorization failed");
        }

        llt.reset();
        locked_ldlt = true;
        analyzed = true;
    }

    /**
     * refactor: numeric refactorization for the same sparsity pattern.
     * Must call analyze_once() first; throws if numeric factorization fails.
     */
    void refactor(const spmat &K) {
        if (!analyzed)
            throw std::logic_error(
                "ReusedK::refactor: analyze_once() not called");

        spmat Kp = use_amd ? spmat((P.transpose() * K) * P) : K;
        Kp.makeCompressed();

        if (!locked_ldlt) {
            // LLT path
            llt->factorize(Kp);
            if (llt->info() == Eigen::Success)
                return;

            // If LLT failed now (SPD broke), try switching permanently to LDLT
            ldlt = std::make_shared<Eigen::SimplicialLDLT<spmat>>();
            ldlt->analyzePattern(
                Kp); // same pattern, but LDLT needs its own symbolic
            ldlt->factorize(Kp);
            if (ldlt->info() != Eigen::Success) {
                ldlt.reset();
                throw std::runtime_error(
                    "ReusedK::refactor: LLT failed and LDLT failed");
            }
            llt.reset();
            locked_ldlt = true;
            return;
        }

        // LDLT path (locked)
        ldlt->factorize(Kp);
        if (ldlt->info() != Eigen::Success)
            throw std::runtime_error(
                "ReusedK::refactor: LDLT factorization failed");
    }

    /**
     * solve: y = K^{-1} b  (uses last numeric factors)
     */
    dvec solve(const dvec &b) const {
        if (!analyzed)
            throw std::logic_error("ReusedK::solve: analyze_once() not called");

        if (use_amd) {
            dvec bp = P.transpose() * b;
            if (!locked_ldlt) {
                return P * llt->solve(bp);
            } else {
                return P * ldlt->solve(bp);
            }
        } else {
            if (!locked_ldlt)
                return llt->solve(b);
            else
                return ldlt->solve(b);
        }
    }

    /**
     * solve_inplace: b ← K^{-1} b  (overwrites b)
     * Note: Eigen’s SimplicialLLT/LDLT don’t have solveInPlace for sparse by
     * default, so we just assign the returned vector back to b.
     */
    void solve_inplace(dvec &b) const { b = solve(b); }

    /**
     * Multi-RHS solve: X = K^{-1} B
     * (Columns solved independently; no extra symbolic cost.)
     */
    // Put this in ReusedK (replacing the previous dmat overload)
    // Efficient multi-RHS solve: X = K^{-1} B
    dmat solve(const dmat &B) const {
        if (!analyzed)
            throw std::logic_error(
                "ReusedK::solve(B): analyze_once() not called");

        dmat X;
        if (!locked_ldlt) {
            // SPD: K = P * (L L^T) * P^T
            // 1) permute RHS once: Bp = P^T * B
            dmat Bp = use_amd ? (P.transpose() * B).eval() : B;

            // 2) Y = L^{-1} Bp   (forward solve on all columns)
            dmat Y = Bp;
            llt->matrixL().solveInPlace(Y);

            // 3) Xp = L^{-T} Y    (backward solve on all columns)
            dmat Xp = Y;
            llt->matrixL().transpose().solveInPlace(Xp);

            // 4) undo permutation: X = P * Xp
            X = use_amd ? (P * Xp).eval() : Xp;
        } else {
            // Indefinite: K = P * (L D L^T) * P^T
            dmat Bp = use_amd ? (P.transpose() * B).eval() : B;

            // Forward: Y = L^{-1} Bp
            dmat Y = Bp;
            ldlt->matrixL().solveInPlace(Y);

            // Scale by D^{-1}
            const dvec &D = ldlt->vectorD();
            for (int i = 0; i < Y.rows(); ++i) {
                const double invd = std::abs(D[i]) > 1e-30 ? 1.0 / D[i] : 0.0;
                Y.row(i) *= invd;
            }

            // Backward: Xp = L^{-T} Y
            dmat Xp = Y;
            ldlt->matrixL().transpose().solveInPlace(Xp);

            X = use_amd ? (P * Xp).eval() : Xp;
        }
        return X;
    }
};

class HYKKTFlow {
public:
    explicit HYKKTFlow(ChompConfig cfg) : cfg_(std::move(cfg)) {}

    // Reset persistent state between *problems* (not between IP steps)
    void reset() {
        rk_.reset();
        dy_prev_.reset();
        gamma_.reset();
        G_cached_.resize(0, 0);
        analyzed_ = false;
    }
    std::pair<dvec, dvec>
    step(const spmat &W, const spmat &G, const dvec &r1, const dvec &r2,
         double delta, std::optional<double> gamma_user = std::nullopt) {
        const int n = W.rows(), m = G.rows();
        if (m == 0 || G.cols() != n)
            throw std::invalid_argument("HYKKTFlow::step: invalid G");

        // 1) gamma
        if (gamma_user)
            gamma_ = *gamma_user;
        if (!gamma_)
            gamma_ = compute_gamma_heuristic(W, G, delta);
        const double gamma = *gamma_;

        // 2) K
        spmat K = build_augmented_system_inplace(W, G, delta, gamma);

        // 3) analyze once / refactor
        if (!rk_)
            rk_ = std::make_shared<ReusedK>();
        if (!analyzed_) {
            rk_->analyze_once(K, /*amd=*/true);
            analyzed_ = true;
        } else {
            rk_->refactor(K);
        }
        auto Ks = [&](const dvec &b) -> dvec { return rk_->solve(b); };

        // 4) Schur RHS
        const dvec s = r1 + gamma * (G.transpose() * r2);
        const dvec rhs_s = G * Ks(s) - r2;

        // 5) Dense or iterative Schur
        const bool small_m =
            (m <= std::max(1, int(0.05 * n))); // default cutoff

        dvec dy;
        if (small_m) {
            const spmat Gt = G.transpose();
            dmat Z(n, m);
            Z.setZero();
            dvec rhs(n);
            for (int j = 0; j < m; ++j) {
                rhs.setZero();
                for (spmat::InnerIterator it(Gt, j); it; ++it)
                    rhs[it.row()] = it.value();
                Z.col(j) = Ks(rhs);
            }
            const dmat S = G * Z;
            dy = Eigen::LLT<dmat>(S.selfadjointView<Eigen::Lower>())
                     .solve(rhs_s);
        } else {
            // Light Jacobi preconditioner by default
            dvec diagKinv = K.diagonal().cwiseMax(1e-12).cwiseInverse();
            dvec dS = schur_diag_hat(G, diagKinv);
            for (int i = 0; i < dS.size(); ++i)
                dS[i] = 1.0 / std::max(dS[i], 1e-18);
            std::optional<dvec> JacobiMinv = std::move(dS);

            LinOp S_op{m, [&](const dvec &y, dvec &out) {
                           out = G * Ks(G.transpose() * y);
                       }};
            auto [dy_sol, _] = cg(S_op, rhs_s, JacobiMinv, 1e-8, 200, dy_prev_,
                                  std::nullopt, true);
            dy = std::move(dy_sol);
        }
        dy_prev_ = dy;

        // 6) Back-substitute
        const dvec dx = Ks(s - G.transpose() * dy);
        return {dx, dy};
    }

    // Accessors (optional)
    const std::shared_ptr<ReusedK> &reusedK() const { return rk_; }
    std::optional<double> gamma() const { return gamma_; }

private:
    ChompConfig cfg_;
    std::shared_ptr<ReusedK> rk_;
    bool analyzed_ = false;

    std::optional<double> gamma_;
    std::optional<dvec> dy_prev_;
    spmat G_cached_; // kept if your Schur op needs a persistent G
};

// ------------------------------ HYKKT ---------------------------------
class HYKKTStrategy final : public KKTStrategy {
public:
    HYKKTStrategy() { name = "hykkt"; }

    std::tuple<dvec, dvec, std::shared_ptr<KKTReusable>>
    factor_and_solve(const spmat &W, const std::optional<spmat> &Gopt,
                     const dvec &r1, const std::optional<dvec> &r2opt,
                     const ChompConfig &cfg,
                     std::optional<double> /*regularizer*/,
                     std::unordered_map<std::string, dvec> &cache, double delta,
                     std::optional<double> gamma_user,
                     bool assemble_schur_if_m_small, bool use_prec) override {

        if (!Gopt || !r2opt) {
            throw std::invalid_argument("HYKKT requires equality constraints");
        }

        const auto &G = *Gopt;
        const auto &r2 = *r2opt;
        const int n = W.rows(), m = G.rows();

        // ---------- gamma selection with small cache ----------
        const double gamma = [&] {
            if (gamma_user)
                return *gamma_user;
            if (!cfg.adaptive_gamma)
                return compute_gamma_heuristic(W, G, delta);
            const std::string key =
                "gamma_" + std::to_string(n) + "_" + std::to_string(m);
            auto it = cache.find(key);
            if (it != cache.end())
                return it->second[0];
            double g = compute_adaptive_gamma(W, G, delta);
            cache[key] = dvec::Constant(1, g);
            return g;
        }();

        // ---------- build K = W + δI + γ GᵀG (in-place, no temp I, fewer
        // prunes) ----------
        spmat K = build_augmented_system_inplace(W, G, delta, gamma);

        // ---------- analyze+factor with optional AMD ----------
        auto [solver, is_spd] = create_solver(K, cfg);
        (void)is_spd;

        // ---------- solve Schur & primal ----------
        auto [dx, dy] = solve_schur_system(G, K, r1, r2, gamma, solver, cfg,
                                           assemble_schur_if_m_small, use_prec);

        // ---------- reusable wrapper ----------
        auto reusable = create_reusable_solver(G, solver, gamma, cfg);
        return std::make_tuple(dx, dy, reusable);
    }

private:
    // ---------------- gamma heuristics ----------------
    static double rowsum_inf_norm(const spmat &A) {
        double mx = 0.0;
        for (int j = 0; j < A.outerSize(); ++j) {
            for (spmat::InnerIterator it(A, j); it; ++it)
                mx = std::max(mx, std::abs(it.value()));
        }
        return mx;
    }

    double compute_gamma_heuristic(const spmat &W, const spmat &G,
                                   double delta) const {
        const double Wn = rowsum_inf_norm(W) + delta;
        const double Gn = rowsum_inf_norm(G);
        return std::max(1.0, Wn / std::max(1.0, Gn * Gn));
    }

    double estimate_condition_number(const spmat &W, double delta) const {
        // Very cheap proxy: ratio of max/min diagonal after δ; robust to zeros
        dvec d = W.diagonal();
        if (d.size() == 0)
            return 1.0;
        double dmin = std::numeric_limits<double>::infinity();
        double dmax = 0.0;
        for (int i = 0; i < d.size(); ++i) {
            double v = std::abs(d[i]) + delta;
            dmin = std::min(dmin, v);
            dmax = std::max(dmax, v);
        }
        dmin = std::max(dmin, 1e-18);
        return dmax / dmin;
    }

    double compute_adaptive_gamma(const spmat &W, const spmat &G,
                                  double delta) const {
        const double Wn = rowsum_inf_norm(W) + delta;
        const double Gn = rowsum_inf_norm(G);
        const double cond_est = estimate_condition_number(W, delta);
        double g = std::max(1.0, Wn / std::max(1.0, Gn * Gn));
        if (cond_est > 1e12)
            g *= 10.0;
        else if (cond_est < 1e6)
            g *= 0.1;
        return g;
    }

    // ------------- fast build of K = W + δI + γ GᵀG -------------
    spmat build_augmented_system_inplace(const spmat &W, const spmat &G,
                                         double delta, double gamma) const {
        spmat K = W; // copy structure+values
        K.makeCompressed();

        // Add delta on diagonal without forming I
        if (std::abs(delta) > 0.0) {
            // Ensure diagonal entries exist; create if structurally missing
            K.reserve(Eigen::VectorXi::Constant(K.cols(), 1));
            for (int i = 0; i < K.rows(); ++i)
                K.coeffRef(i, i) += delta;
        }

        // Add γ GᵀG efficiently; Eigen's sparse product is OK but we avoid
        // extra prune()
        if (std::abs(gamma) > 0.0) {
            spmat GtG = (G.transpose() * G).eval();
            GtG.makeCompressed();
            // K ← K + γ GᵀG
            K += (gamma * GtG).pruned();
        }

        K.prune(1e-300); // keep structure clean
        K.makeCompressed();
        return K;
    }

    // ------------- solver factory (analyzePattern + factorize) -------------
    std::pair<std::function<dvec(const dvec &)>, bool>
    create_solver(const spmat &K, const ChompConfig &cfg) const {
        const int n = K.rows();

        // Ordering
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> P;
        const bool use_amd = (cfg.sym_ordering == "amd");
        if (use_amd) {
            Eigen::AMDOrdering<int> amd;
            amd(K, P);
        } else {
            P.setIdentity(n);
        }

        spmat Kp = use_amd ? spmat((P.transpose() * K) * P) : K;
        Kp.makeCompressed();

        // Try LLT
        auto llt = std::make_shared<Eigen::SimplicialLLT<spmat>>();
        llt->analyzePattern(Kp);
        llt->factorize(Kp);
        if (llt->info() == Eigen::Success) {
            auto f = [P, llt, use_amd](const dvec &b) -> dvec {
                if (use_amd) {
                    dvec bp = P.transpose() * b;
                    dvec yp = llt->solve(bp);
                    return P * yp;
                } else {
                    return llt->solve(b);
                }
            };
            return {f, true};
        }

        // Fallback: LDLT
        auto ldlt = std::make_shared<Eigen::SimplicialLDLT<spmat>>();
        ldlt->analyzePattern(Kp);
        ldlt->factorize(Kp);
        if (ldlt->info() != Eigen::Success) {
            throw std::runtime_error("HYKKT: K factorization failed");
        }
        auto f = [P, ldlt, use_amd](const dvec &b) -> dvec {
            if (use_amd) {
                dvec bp = P.transpose() * b;
                dvec yp = ldlt->solve(bp);
                return P * yp;
            } else {
                return ldlt->solve(b);
            }
        };
        return {f, false};
    }

    // ------------- Schur solve -------------
    std::pair<dvec, dvec>
    solve_schur_system(const spmat &G, const spmat &K, const dvec &r1,
                       const dvec &r2, double gamma,
                       const std::function<dvec(const dvec &)> &Ks,
                       const ChompConfig &cfg, bool assemble_schur_if_m_small,
                       bool use_prec) const {
        const int n = K.rows();
        const int m = G.rows();

        // s = r1 + γ Gᵀ r2
        const dvec s = r1 + gamma * (G.transpose() * r2);
        // rhs for Schur: rhs_s = G K^{-1} s − r2
        const dvec rhs_s = G * Ks(s) - r2;

        dvec dy;
        const bool small_m =
            assemble_schur_if_m_small &&
            (m <= std::max(1, int(cfg.schur_dense_cutoff * n)));

        if (small_m) {
            // Dense Schur path (batched solves, minimal conversions)
            // Build Gᵀ once (cheap transpose view); stream RHS columns
            const spmat Gt = G.transpose();
            dmat Z(n, m);
            Z.setZero();

            // Solve K Z = Gᵀ in blocks (cache-friendly)
            // Use a single dense temporary per column to avoid repeated allocs
            dvec rhs(n);
            for (int j = 0; j < m; ++j) {
                rhs.setZero();
                for (spmat::InnerIterator it(Gt, j); it; ++it)
                    rhs[it.row()] = it.value();
                Z.col(j) = Ks(rhs);
            }
            const dmat S = G * Z; // m × m dense Schur
            dy = Eigen::LLT<dmat>(S.selfadjointView<Eigen::Lower>())
                     .solve(rhs_s);
        } else {
            // Iterative Schur with optional (Jacobi|SSOR) preconditioner
            dy = solve_schur_iterative(G, Ks, rhs_s, K, cfg, use_prec);
        }

        const dvec dx = Ks(s - G.transpose() * dy);
        return {dx, dy};
    }

    // Diagonal of S_hat = G * diag(d) * Gᵀ, computed without forming S_hat
    static dvec schur_diag_hat(const spmat &G, const dvec &d) {
        const int m = G.rows();
        dvec diag = dvec::Zero(m);
        for (int j = 0; j < G.outerSize(); ++j) {
            for (spmat::InnerIterator it(G, j); it; ++it) {
                const int i = it.row();
                const double gij = it.value();
                diag[i] += (gij * gij) * d[j];
            }
        }
        return diag;
    }

    // Build S_hat = G * diag(d) * Gᵀ efficiently (rank-1 accumulation)
    static spmat build_S_hat_fast(const spmat &G, const dvec &d) {
        const int m = G.rows();
        std::vector<Eigen::Triplet<double>> T;
        // Rough reserve: nnz(S_hat) ≤ sum over cols of nnz(g_k)^2
        size_t cap = 0;
        for (int j = 0; j < G.outerSize(); ++j) {
            int nnz = 0;
            for (spmat::InnerIterator it(G, j); it; ++it)
                ++nnz;
            cap += size_t(nnz) * size_t(nnz);
        }
        T.reserve(std::min<size_t>(cap, size_t(4) * size_t(G.nonZeros())));

        // For each column k, add d[k] * g_k g_kᵀ
        for (int j = 0; j < G.outerSize(); ++j) {
            std::vector<int> rows;
            std::vector<double> vals;
            for (spmat::InnerIterator it(G, j); it; ++it) {
                rows.push_back(it.row());
                vals.push_back(it.value());
            }
            const double dj = d[j];
            const int t = (int)rows.size();
            for (int a = 0; a < t; ++a) {
                const int ra = rows[a];
                const double va = vals[a];
                for (int b = a; b < t; ++b) {
                    const int rb = rows[b];
                    const double vb = vals[b];
                    T.emplace_back(ra, rb, dj * va * vb);
                    if (rb != ra)
                        T.emplace_back(rb, ra, dj * va * vb);
                }
            }
        }
        spmat S(m, m);
        S.setFromTriplets(T.begin(), T.end());
        S.makeCompressed();
        return S;
    }

    // ------------- iterative Schur with optional preconditioner -------------
    dvec solve_schur_iterative(const spmat &G,
                               const std::function<dvec(const dvec &)> &Ks,
                               const dvec &rhs_s, const spmat &K,
                               const ChompConfig &cfg, bool use_prec) const {

        const int m = G.rows();

        LinOp S_op{m, [&](const dvec &y, dvec &out) {
                       // out = G * K^{-1} * Gᵀ * y
                       out = G * Ks(G.transpose() * y);
                   }};

        std::optional<dvec> JacobiMinv;
        std::optional<SSORPrecond> ssor;

        if (use_prec) {
            // diag(K)^{-1}
            dvec diagKinv = K.diagonal().cwiseMax(1e-12).cwiseInverse();

            if (cfg.prec_type == "jacobi") {
                // Jacobi on Schur ≈ diag(G diag(K^{-1}) Gᵀ)^{-1}
                dvec dS = schur_diag_hat(G, diagKinv);
                // Guard and invert (avoid divide-by-zero)
                for (int i = 0; i < dS.size(); ++i)
                    dS[i] = 1.0 / std::max(dS[i], 1e-18);
                JacobiMinv = std::move(dS);
            } else if (cfg.prec_type == "ssor") {
                // SSOR on S_hat = G diag(K^{-1}) Gᵀ
                const spmat S_hat = build_S_hat_fast(G, diagKinv);
                ssor.emplace(S_hat, cfg.ssor_omega);
            } // "none" → no preconditioner
        }

        auto [dy, _info] = cg(S_op, rhs_s, JacobiMinv, cfg.cg_tol, cfg.cg_maxit,
                              std::nullopt, ssor, cfg.use_simd);
        return std::move(dy);
    }

    // ------------- reusable wrapper -------------
    std::shared_ptr<KKTReusable>
    create_reusable_solver(const spmat &G,
                           const std::function<dvec(const dvec &)> &Ks,
                           double gamma, const ChompConfig &cfg) const {
        struct Reuse final : KKTReusable {
            spmat G;
            std::function<dvec(const dvec &)> Ks;
            double gamma;
            ChompConfig config;

            Reuse(spmat Gin, std::function<dvec(const dvec &)> Ksin, double g,
                  ChompConfig cfg)
                : G(std::move(Gin)), Ks(std::move(Ksin)), gamma(g),
                  config(std::move(cfg)) {}

            std::pair<dvec, dvec> solve(const dvec &r1n,
                                        const std::optional<dvec> &r2n,
                                        double tol, int maxit) override {
                if (!r2n)
                    throw std::invalid_argument("HYKKT::Reuse needs r2");

                const dvec s = r1n + gamma * (G.transpose() * (*r2n));
                const dvec rhs_s = G * Ks(s) - (*r2n);

                LinOp S_op{static_cast<int>(G.rows()),
                           [&](const dvec &y, dvec &out) {
                               out = G * Ks(G.transpose() * y);
                           }};

                auto [dy, _] = cg(S_op, rhs_s, std::nullopt, tol, maxit,
                                  std::nullopt, std::nullopt, config.use_simd);
                const dvec dx = Ks(s - G.transpose() * dy);
                return {dx, dy};
            }
        };

        return std::make_shared<Reuse>(G, Ks, gamma, cfg);
    }
};

// ----------------------- HYKKT via CHOLMOD (SuiteSparse) --------------
class HYKKTCholmodStrategy final : public KKTStrategy {
public:
    HYKKTCholmodStrategy() { name = "hykkt_cholmod"; }

    std::tuple<dvec, dvec, std::shared_ptr<KKTReusable>>
    factor_and_solve(const spmat &W, const std::optional<spmat> &Gopt,
                     const dvec &r1, const std::optional<dvec> &r2opt,
                     const ChompConfig &cfg,
                     std::optional<double> /*regularizer*/,
                     std::unordered_map<std::string, dvec> &cache, double delta,
                     std::optional<double> gamma_user,
                     bool assemble_schur_if_m_small, bool use_prec) override {
        std::cout << "Using HYKKTCholmodStrategy" << std::endl;
#ifndef EIGEN_CHOLMOD_SUPPORT
        (void)W;
        (void)Gopt;
        (void)r1;
        (void)r2opt;
        (void)cfg;
        (void)cache;
        (void)delta;
        (void)gamma_user;
        (void)assemble_schur_if_m_small;
        (void)use_prec;
        throw std::runtime_error(
            "HYKKT CHOLMOD requested but Eigen was built without CHOLMOD "
            "support. Rebuild with EIGEN_CHOLMOD_SUPPORT and link CHOLMOD.");
#else
        if (!Gopt || !r2opt)
            throw std::invalid_argument(
                "HYKKT (cholmod) requires equality constraints");

        const auto &G = *Gopt;
        const auto &r2 = *r2opt;
        const int n = W.rows(), m = G.rows();

        // gamma selection (mirrors HYKKTStrategy)
        double gamma;
        if (gamma_user) {
            gamma = *gamma_user;
        } else if (cfg.adaptive_gamma) {
            const std::string cache_key =
                "gamma_" + std::to_string(n) + "_" + std::to_string(m);
            if (auto it = cache.find(cache_key); it != cache.end()) {
                gamma = it->second[0];
            } else {
                gamma = compute_adaptive_gamma(W, G, delta);
                cache[cache_key] = dvec::Constant(1, gamma);
            }
        } else {
            gamma = compute_gamma_heuristic(W, G, delta);
        }

        // augmented SPD block
        spmat K = build_augmented_system(W, G, delta, gamma);

        // factor K with CHOLMOD (prefer supernodal LLT; fall back to LDLT)
        auto solver_pair = create_solver_cholmod(K, cfg);
        auto &solveK = solver_pair.first;

        // Schur solve (dense or iterative as in HYKKTStrategy)
        auto [dx, dy] =
            solve_schur_system(G, K, r1, r2, gamma, solveK, cfg,
                               assemble_schur_if_m_small, use_prec, n, m);

        auto reusable = create_reusable_solver(G, solveK, gamma, cfg);
        return std::make_tuple(dx, dy, reusable);
#endif
    }

private:
#ifdef EIGEN_CHOLMOD_SUPPORT
    static double compute_gamma_heuristic(const spmat &W, const spmat &G,
                                          double delta) {
        const double W_norm = rowsum_inf_norm(W) + delta;
        const double G_norm = rowsum_inf_norm(G);
        return std::max(1.0, W_norm / std::max(1.0, G_norm * G_norm));
    }
    static double compute_adaptive_gamma(const spmat &W, const spmat &G,
                                         double delta) {
        const double W_norm = rowsum_inf_norm(W) + delta;
        const double G_norm = rowsum_inf_norm(G);
        const double cond_est = estimate_condition_number(W, delta);
        double base_gamma =
            std::max(1.0, W_norm / std::max(1.0, G_norm * G_norm));
        if (cond_est > 1e12)
            base_gamma *= 10.0;
        else if (cond_est < 1e6)
            base_gamma *= 0.1;
        return base_gamma;
    }
    static spmat build_augmented_system(const spmat &W, const spmat &G,
                                        double delta, double gamma) {
        spmat K = W;
        if (delta != 0.0) {
            spmat I(W.rows(), W.rows());
            I.setIdentity();
            K = (K + delta * I).pruned();
        }
        if (gamma != 0.0) {
            K = (K + gamma * (G.transpose() * G).pruned()).pruned();
        }
        K.makeCompressed();
        return K;
    }

    std::pair<std::function<dvec(const dvec &)>, bool>
    create_solver_cholmod(const spmat &K, const ChompConfig &cfg) const {
        const int n = K.rows();

        // Allow user to force identity ordering; otherwise let CHOLMOD choose
        const bool force_identity = (cfg.sym_ordering == "none");
        spmat Kp = K;
        Kp.makeCompressed();
        if (force_identity) {
            // nothing to do; CHOLMOD will still analyze but with natural
            // ordering
        }

        // Try supernodal LLT
        auto llt = std::make_shared<Eigen::CholmodSupernodalLLT<spmat>>();
        llt->compute(Kp);
        if (llt->info() == Eigen::Success) {
            auto solver = [llt](const dvec &b) -> dvec {
                return llt->solve(b);
            };
            return {solver, true};
        }

        // Try simplicial LLT
        auto llt2 = std::make_shared<Eigen::CholmodSimplicialLLT<spmat>>();
        llt2->compute(Kp);
        if (llt2->info() == Eigen::Success) {
            auto solver = [llt2](const dvec &b) -> dvec {
                return llt2->solve(b);
            };
            return {solver, true};
        }

        // Fall back to simplicial LDLT
        auto ldlt = std::make_shared<Eigen::CholmodSimplicialLDLT<spmat>>();
        ldlt->compute(Kp);
        if (ldlt->info() != Eigen::Success)
            throw std::runtime_error(
                "HYKKT (cholmod): K factorization failed (LLT/LDLT)");

        auto solver = [ldlt](const dvec &b) -> dvec { return ldlt->solve(b); };
        return {solver, false};
    }

    std::pair<dvec, dvec>
    solve_schur_system(const spmat &G, const spmat &K, const dvec &r1,
                       const dvec &r2, double gamma,
                       const std::function<dvec(const dvec &)> &solveK,
                       const ChompConfig &cfg, bool assemble_schur_if_m_small,
                       bool use_prec, int n, int m) const {

        const dvec svec = r1 + gamma * (G.transpose() * r2);
        const dvec rhs_s = G * solveK(svec) - r2;

        dvec dy;
        const bool small_m =
            assemble_schur_if_m_small &&
            (m <= std::max(1, int(cfg.schur_dense_cutoff * n)));

        if (small_m) {
            dmat Z(n, m);
            for (int j = 0; j < m; ++j)
                Z.col(j) = solveK(G.transpose().col(j));
            const dmat S = G * Z;
            dy = Eigen::LLT<dmat>(S).solve(rhs_s);
        } else {
            LinOp S_op{m, [&](const dvec &y, dvec &out) {
                           out = G * solveK(G.transpose() * y);
                       }};
            std::optional<dvec> JacobiMinv;
            std::optional<SSORPrecond> ssor;

            if (use_prec) {
                dvec diagKinv = K.diagonal().cwiseMax(1e-12).cwiseInverse();
                if (cfg.prec_type == "jacobi") {
                    JacobiMinv = schur_diag_hat(G, diagKinv).cwiseInverse();
                } else if (cfg.prec_type == "ssor") {
                    const spmat S_hat = build_S_hat(G, diagKinv);
                    ssor.emplace(S_hat, cfg.ssor_omega);
                }
            }
            auto cg_res = cg(S_op, rhs_s, JacobiMinv, cfg.cg_tol, cfg.cg_maxit,
                             std::nullopt, ssor, cfg.use_simd);
            dy = std::move(cg_res.first);
        }

        const dvec dx = solveK(svec - G.transpose() * dy);
        return {dx, dy};
    }

    std::shared_ptr<KKTReusable>
    create_reusable_solver(const spmat &G,
                           const std::function<dvec(const dvec &)> &solveK,
                           double gamma, const ChompConfig &cfg) const {
        struct Reuse final : KKTReusable {
            spmat G;
            std::function<dvec(const dvec &)> Ks;
            double gamma;
            ChompConfig config;
            Reuse(spmat G_, std::function<dvec(const dvec &)> Ks_, double g,
                  ChompConfig cfg)
                : G(std::move(G_)), Ks(std::move(Ks_)), gamma(g), config(cfg) {}
            std::pair<dvec, dvec> solve(const dvec &r1n,
                                        const std::optional<dvec> &r2n,
                                        double tol, int maxit) override {
                if (!r2n)
                    throw std::invalid_argument(
                        "HYKKT::Reuse (cholmod) needs r2");
                const dvec svec_n = r1n + gamma * (G.transpose() * (*r2n));
                const dvec rhs_s_n = G * Ks(svec_n) - (*r2n);
                LinOp S_op{static_cast<int>(G.rows()),
                           [&](const dvec &y, dvec &out) {
                               out = G * Ks(G.transpose() * y);
                           }};
                auto cg_res2 = cg(S_op, rhs_s_n, std::nullopt, tol, maxit,
                                  std::nullopt, std::nullopt, config.use_simd);
                dvec dy_n = std::move(cg_res2.first);
                const dvec dx_n = Ks(svec_n - G.transpose() * dy_n);
                return {dx_n, dy_n};
            }
        };
        return std::make_shared<Reuse>(G, solveK, gamma, cfg);
    }
#endif // EIGEN_CHOLMOD_SUPPORT
};

// ------------------------------ LDL (QDLDL) ---------------------------
class LDLStrategy final : public KKTStrategy {
public:
    LDLStrategy() { name = "ldl"; }

    std::tuple<dvec, dvec, std::shared_ptr<KKTReusable>>
    factor_and_solve(const spmat &W, const std::optional<spmat> &Gopt,
                     const dvec &r1, const std::optional<dvec> &r2opt,
                     const ChompConfig &, std::optional<double> /*regularizer*/,
                     std::unordered_map<std::string, dvec> & /*cache*/,
                     double delta, std::optional<double> /*gamma*/,
                     bool /*assemble_schur_if_m_small*/,
                     bool /*use_prec*/) override {

        bool hasE = false;
        spmat K = assemble_KKT(W, delta, Gopt, &hasE);

        const int n = W.rows();
        const int m = hasE ? Gopt->rows() : 0;
        const int nsys = hasE ? (n + m) : n;

        dvec rhs(nsys);
        if (hasE) {
            if (!r2opt)
                throw std::runtime_error("LDLStrategy: missing r2");
            rhs.head(n) = r1;
            rhs.tail(m) = *r2opt;
        } else {
            rhs = r1;
        }

        auto U = eigen_to_upper_csc(K, 1e-12);
        auto F = qdldl23::factorize(U);

        dvec x = rhs;
        qdldl23::solve(F, x.data());

        dvec dx, dy;
        if (hasE) {
            dx = x.head(n);
            dy = x.tail(m);
        } else {
            dx = x;
            dy.resize(0);
        }

        struct Reuse final : KKTReusable {
            qdldl23::SparseD32 U;
            qdldl23::LDL32 F;
            int n, m;

            Reuse(qdldl23::SparseD32 U_, qdldl23::LDL32 F_, int n_, int m_)
                : U(std::move(U_)), F(std::move(F_)), n(n_), m(m_) {}

            std::pair<dvec, dvec> solve(const dvec &r1n,
                                        const std::optional<dvec> &r2n,
                                        double /*cg_tol*/,
                                        int /*cg_maxit*/) override {
                const int nsys = n + m;
                dvec rhs(nsys);

                if (m > 0) {
                    if (!r2n)
                        throw std::runtime_error("LDL::Reuse: missing r2");
                    rhs.head(n) = r1n;
                    rhs.tail(m) = *r2n;
                } else {
                    rhs = r1n;
                }

                qdldl23::solve(F, rhs.data());

                if (m > 0)
                    return {rhs.head(n), rhs.tail(m)};
                return {rhs, dvec()};
            }
        };

        auto res = std::make_shared<Reuse>(std::move(U), std::move(F), n, m);
        return std::make_tuple(dx, dy, res);
    }
};

// ------------------------------ registry ------------------------------
class KKTSolverRegistry {
    std::unordered_map<std::string, std::shared_ptr<KKTStrategy>> strategies_;

public:
    void register_strategy(std::shared_ptr<KKTStrategy> s) {
        strategies_[s->name] = std::move(s);
    }

    [[nodiscard]] std::shared_ptr<KKTStrategy>
    get(std::string_view name) const {
        auto it = strategies_.find(std::string(name));
        if (it == strategies_.end()) {
            throw std::runtime_error("Unknown KKT strategy: " +
                                     std::string(name));
        }
        return it->second;
    }

    [[nodiscard]] const auto &all() const noexcept { return strategies_; }
};

[[nodiscard]] inline KKTSolverRegistry &default_registry() {
    static KKTSolverRegistry registry;
    static std::once_flag init_flag;

    std::call_once(init_flag, []() {
        registry.register_strategy(std::make_shared<HYKKTStrategy>());
#ifdef EIGEN_CHOLMOD_SUPPORT
        registry.register_strategy(std::make_shared<HYKKTCholmodStrategy>());
#endif
        registry.register_strategy(std::make_shared<LDLStrategy>());
    });

    return registry;
}

} // namespace kkt
