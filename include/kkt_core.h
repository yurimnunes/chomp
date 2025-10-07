#pragma once
// Optimized C++23 KKT core (HYKKT + LDL) with Eigen
// AVX guard fixed; ILU removed; Schur solves use CG + (Jacobi|SSOR).

#include "model.h"
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/OrderingMethods>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <deque>
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

#include "amd.h"
#include "kkt_core_beta.h"
#include "kkt_helper.h"
#include "qdldl.h"

namespace kkt {

// ------------------------------ typedefs ------------------------------
using dvec = Eigen::VectorXd;
using dmat = Eigen::MatrixXd;
using spmat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
// ------------------------------ HYKKT ---------------------------------

class HYKKTStrategy final : public KKTStrategy {
public:
    HYKKTStrategy() { name = "hykkt"; }

    ModelC *model = nullptr; // To access model parameters if needed
    std::tuple<dvec, dvec, std::shared_ptr<KKTReusable>>
    factor_and_solve(ModelC *model_in, const spmat &W,
                     const std::optional<spmat> &Gopt, const dvec &r1,
                     const std::optional<dvec> &r2opt, const ChompConfig &cfg,
                     std::optional<double> /*regularizer*/,
                     std::unordered_map<std::string, dvec> &cache, double delta,
                     std::optional<double> gamma_user,
                     bool assemble_schur_if_m_small, bool use_prec) override {
        if (!Gopt || !r2opt)
            throw std::invalid_argument("HYKKT requires equality constraints");

        model = model_in; // store model pointer

        const auto &G = *Gopt;
        const auto &r2 = *r2opt;
        const int n = W.rows(), m = G.rows();

        // ---------- gamma selection (same logic you had) ----------
        const double gamma = [&] {
            if (gamma_user)
                return *gamma_user;
            if (!cfg.adaptive_gamma)
                return compute_gamma_heuristic(W, G, delta);
            const std::string key =
                "gamma_" + std::to_string(n) + "_" + std::to_string(m);
            if (auto it = cache.find(key); it != cache.end())
                return it->second[0];
            double g = compute_adaptive_gamma(W, G, delta);
            cache[key] = dvec::Constant(1, g);
            return g;
        }();

        // ---------- δ₁ loop: make Hδ SPD or at least factorizable ----------
        double delta1 = delta;
        if (delta1 == 0.0)
            delta1 = std::max(1e-12, cfg.delta_min); // gentle start
        std::function<dvec(const dvec &)> Ks;        // K^{-1}(.)
        bool used_llt = false;
        spmat K_final;

        for (int tries = 0; tries < 10; ++tries) {
            // Use HVP optimization if enabled and model available
            auto [solver_fn, is_spd] = create_or_refactor_solver_qdldl(
                build_augmented_system_inplace(W, G, delta1, gamma), cfg);

            if (solver_fn) {
                Ks = std::move(solver_fn);
                used_llt = is_spd;

                // Keep the final matrix for preconditioner (only needed for
                // non-HVP path)
                if (!cfg.use_hvp || !model) {
                    spmat K =
                        build_augmented_system_inplace(W, G, delta1, gamma);
                    K.makeCompressed();
                    K_final.swap(K);
                }
                break;
            }
            delta1 = std::min(cfg.delta_max, std::max(2.0 * delta1, 1e-9));
        }

        if (!Ks) {
            throw std::runtime_error(
                "HYKKT: failed to factor H_delta after δ₁ ramp.");
        }

        // ---------- Schur RHS ----------
        // s = r1 + γ Gᵀ r2
        const dvec s = r1 + gamma * (G.transpose() * r2);
        // rhs for Schur: rhs_s = G K^{-1} s − r2
        const dvec rhs_s = G * Ks(s) - r2;

        // ---------- Solve Schur (dense for small m, else iterative) ----------
        dvec dy;
        const bool small_m =
            assemble_schur_if_m_small &&
            (m <= std::max(1, int(cfg.schur_dense_cutoff * n)));

        if (small_m) {
            dy = solve_schur_dense_with_delta2(G, Ks, rhs_s, cfg);
        } else {
            dy = solve_schur_iterative_with_delta2(G, Ks, rhs_s, K_final, cfg,
                                                   use_prec);
        }

        // ---------- Back-substitution ----------
        const dvec dx = Ks(s - G.transpose() * dy);

        // ---------- reusable wrapper ----------
        auto reusable = create_reusable_solver(G, Ks, gamma, cfg);
        return std::make_tuple(dx, dy, reusable);
    }

private:
    // ======================= symbolic cache ==========================
    struct SymbolicCache {
        int n{-1};
        bool use_amd{false};
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> P;
        // Keep both to allow fallback without re-analyze
        std::shared_ptr<Eigen::SimplicialLLT<spmat>> llt;
        std::shared_ptr<Eigen::SimplicialLDLT<spmat>> ldlt;
        bool analyzed_llt{false}, analyzed_ldlt{false};
    };

    struct QDLDLCache {
        int n{-1};
        bool use_amd{false};

        // ordering
        qdldl23::Ordering<int32_t> ord; // identity or AMD
        bool have_ord{false};

        // symbolic (for permuted matrix)
        qdldl23::Symb32 Sperm;
        bool have_S{false};

        // numeric factors for permuted matrix
        qdldl23::LDL32 F;
        bool have_F{false};
    };

    mutable std::optional<QDLDLCache> qcache_;
    mutable std::optional<SymbolicCache> sym_; // persisted across calls

    // ======================= heuristics (yours) ======================
    static double rowsum_inf_norm(const spmat &A) {
        double mx = 0.0;
        for (int j = 0; j < A.outerSize(); ++j)
            for (spmat::InnerIterator it(A, j); it; ++it)
                mx = std::max(mx, std::abs(it.value()));
        return mx;
    }
    double compute_gamma_heuristic(const spmat &W, const spmat &G,
                                   double delta) const {
        const double Wn = rowsum_inf_norm(W) + delta;
        const double Gn = rowsum_inf_norm(G);
        return std::max(1.0, Wn / std::max(1.0, Gn * Gn));
    }
    double estimate_condition_number(const spmat &W, double delta) const {
        dvec d = W.diagonal();
        if (d.size() == 0)
            return 1.0;
        double dmin = std::numeric_limits<double>::infinity(), dmax = 0.0;
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

    // =================== build K = W + δI + γ GᵀG ====================
    spmat build_augmented_system_inplace(const spmat &W, const spmat &G,
                                         double delta, double gamma) const {
        spmat K = W; // copy values+structure
        K.makeCompressed();

        if (std::abs(delta) > 0.0) {
            // ensure diag exists; coeffRef creates it if missing
            for (int i = 0; i < K.rows(); ++i)
                K.coeffRef(i, i) += delta;
        }
        if (std::abs(gamma) > 0.0) {
            spmat GtG = (G.transpose() * G).eval();
            GtG.makeCompressed();
            K += (gamma * GtG).pruned();
        }
        K.prune(1e-300);
        K.makeCompressed();
        return K;
    }

    struct Work {
        dvec v, Kv, Gv, Gtv, z, p, rhs, tmpn;
        void ensure(int n, int m) {
            if (v.size() != n) {
                v.resize(n);
                Kv.resize(n);
                z.resize(n);
                p.resize(n);
                tmpn.resize(n);
            }
            if (Gv.size() != m) {
                Gv.resize(m);
                Gtv.resize(m);
            }
            if (rhs.size() != n)
                rhs.resize(n);
        }
    };
    mutable Work wk_;

    // --------- New: stronger diagonal Schur preconditioner ----------
    // Probe diag(K^{-1}) only on columns that actually appear in G
    static dvec
    schur_diagKinv_probe(const spmat &G,
                         const std::function<dvec(const dvec &)> &Kinv, int n) {
        std::vector<char> mark((size_t)n, 0);
        for (int j = 0; j < G.outerSize(); ++j)
            for (spmat::InnerIterator it(G, j); it; ++it)
                mark[(size_t)it.col()] = 1;

        dvec dinv = dvec::Zero(n);
        // parallelize if you wish
        for (int j = 0; j < n; ++j) {
            if (!mark[(size_t)j])
                continue;
            dvec ej = dvec::Zero(n);
            ej[j] = 1.0;
            dvec Kj = Kinv(ej);
            dinv[j] = std::max(Kj[j], 1e-16);
        }
        return dinv;
    }
    // diag(S) from diag(K^{-1}): S ≈ G·diag(K^{-1})·Gᵀ  (take diagonal only)
    static dvec schur_diag_from_diagKinv(const spmat &G, const dvec &diagKinv) {
        const int m = G.rows();
        dvec diagS = dvec::Zero(m);
        for (int j = 0; j < G.outerSize(); ++j)
            for (spmat::InnerIterator it(G, j); it; ++it)
                diagS[it.row()] += (it.value() * it.value()) * diagKinv[j];
        return diagS;
    }

    // =============== Helper Functions ===============
    bool should_use_smw_direct(const spmat &G, double gamma) const {
        // Use SMW for small constraint matrices or small gamma
        const int m = G.rows();
        const int n = G.cols();

        // Heuristic: use SMW if constraint matrix is relatively small
        // or if gamma is small enough that γ*G^T*G is not dominant
        return (m < n / 4) || (gamma < 1e-3) || (G.nonZeros() < n);
    }

    // ====================== Original factorization methods =================
    std::pair<std::function<dvec(const dvec &)>, bool>
    create_or_refactor_solver(const spmat &K, const ChompConfig &cfg) const {
        const int n = K.rows();
        const bool want_amd = (cfg.sym_ordering == "amd");

        // init / (re)build cache if shape/order changed
        if (!sym_ || sym_->n != n || sym_->use_amd != want_amd) {
            sym_.emplace();
            sym_->n = n;
            sym_->use_amd = want_amd;
            sym_->P.setIdentity(n);
            if (want_amd) {
                Eigen::AMDOrdering<int> amd;
                amd(K, sym_->P);
            }
            sym_->llt = std::make_shared<Eigen::SimplicialLLT<spmat>>();
            sym_->ldlt = std::make_shared<Eigen::SimplicialLDLT<spmat>>();
            sym_->analyzed_llt = false;
            sym_->analyzed_ldlt = false;
        }

        spmat Kp =
            sym_->use_amd ? spmat((sym_->P.transpose() * K) * sym_->P) : K;
        Kp.makeCompressed();

        // Try LLT first
        if (!sym_->analyzed_llt) {
            sym_->llt->analyzePattern(Kp);
            sym_->analyzed_llt = true;
        }
        sym_->llt->factorize(Kp);
        if (sym_->llt->info() == Eigen::Success) {
            auto f = [P = sym_->P, L = sym_->llt,
                      use_amd = sym_->use_amd](const dvec &b) -> dvec {
                if (use_amd) {
                    dvec bp = P.transpose() * b;
                    dvec yp = L->solve(bp);
                    return P * yp;
                }
                return L->solve(b);
            };
            return {f, true};
        }

        // Fallback to LDLT (same symbolics reused)
        if (!sym_->analyzed_ldlt) {
            sym_->ldlt->analyzePattern(Kp);
            sym_->analyzed_ldlt = true;
        }
        sym_->ldlt->factorize(Kp);
        if (sym_->ldlt->info() != Eigen::Success) {
            return {nullptr, false}; // let caller bump δ₁ and retry
        }
        auto f = [P = sym_->P, L = sym_->ldlt,
                  use_amd = sym_->use_amd](const dvec &b) -> dvec {
            if (use_amd) {
                dvec bp = P.transpose() * b;
                dvec yp = L->solve(bp);
                return P * yp;
            }
            return L->solve(b);
        };
        return {f, false};
    }

    std::pair<std::function<dvec(const dvec &)>, bool>
    create_or_refactor_solver_qdldl(const spmat &K,
                                    const ChompConfig &cfg) const {
        using namespace qdldl23;
        const int n = K.rows();

        const bool use_eigen_amd = (cfg.sym_ordering == "amd");
        const bool use_my_amd = (cfg.sym_ordering == "amd_custom");

        // (Re)alloc cache on size/order change
        if (!qcache_ || qcache_->n != n ||
            qcache_->use_amd != (use_eigen_amd || use_my_amd)) {
            qcache_.emplace();
            qcache_->n = n;
            qcache_->use_amd = (use_eigen_amd || use_my_amd);
            qcache_->have_ord = false;
            qcache_->have_S = false;
            qcache_->have_F = false;
        }

        // Convert to upper CSC for qdldl
        SparseD32 A = eigen_to_upper_csc(K);

        // ---------- Ordering: your AMD (preferred), Eigen AMD, or identity
        // ----------
        if (!qcache_->have_ord) {
            if (use_my_amd) {
                qcache_->ord = qdldl_order_from_my_amd(
                    K,
                    /*symmetrize_union=*/true,
                    /*dense_cutoff=*/
                    (cfg.amd_dense_cutoff_has_value ? cfg.amd_dense_cutoff
                                                    : -1));
            } else if (use_eigen_amd) {
                Eigen::AMDOrdering<int> amd;
                Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> P(
                    n);
                P.setIdentity(n);
                amd(K, P);
                std::vector<int32_t> perm_old2new((size_t)n);
                for (int i = 0; i < n; ++i)
                    perm_old2new[(size_t)P.indices()[i]] = i;
                qcache_->ord =
                    Ordering<int32_t>::from_perm(std::move(perm_old2new));
            } else {
                qcache_->ord = Ordering<int32_t>::identity((int32_t)n);
            }
            qcache_->have_ord = true;
        }

        // Analyze once on permuted structure
        if (!qcache_->have_S) {
            const auto B = permute_symmetric_upper(A, qcache_->ord);
            qcache_->Sperm = analyze_fast(B);
            qcache_->have_S = true;
        }

        // Numeric refactorization with current values
        qcache_->F = refactorize_with_ordering(A, qcache_->Sperm, qcache_->ord);
        qcache_->have_F = true;

        // Solve closure using the cached factors + ordering
        auto f = [F = qcache_->F, ord = qcache_->ord](const dvec &b) -> dvec {
            dvec x = b;
            qdldl23::solve_with_ordering(F, ord, x.data());
            return x;
        };
        return {f, false};
    }

    // =================== Original Schur methods ===============
    dvec solve_schur_dense_with_delta2(
        const spmat &G, const std::function<dvec(const dvec &)> &Ks,
        const dvec &rhs_s, const ChompConfig &cfg) const {
        const int n = G.cols(), m = G.rows();
        const spmat Gt = G.transpose();

        dmat Z(n, m);
        Z.setZero();
        dvec rhs(n);

        // Solve K Z = Gᵀ by columns
        for (int j = 0; j < m; ++j) {
            rhs.setZero();
            for (spmat::InnerIterator it(Gt, j); it; ++it)
                rhs[it.row()] = it.value();
            Z.col(j) = Ks(rhs);
        }
        dmat S = (G * Z).selfadjointView<Eigen::Lower>();

        // Try LLT; add δ₂ I if needed
        double delta2 = 0.0;
        for (int tries = 0; tries < 5; ++tries) {
            if (delta2 > 0.0)
                S.diagonal().array() += delta2;
            Eigen::LLT<dmat> llt(S.selfadjointView<Eigen::Lower>());
            if (llt.info() == Eigen::Success) {
                return llt.solve(rhs_s);
            }
            // bump δ₂
            delta2 = (delta2 == 0.0)
                         ? std::max(1e-12, cfg.schur_delta2_min)
                         : std::min(cfg.schur_delta2_max, 10.0 * delta2);
        }
        throw std::runtime_error("HYKKT: dense Schur LLT failed even with δ₂.");
    }

    // ============== Schur (iterative) with δ₂ & optional prec =========
    dvec solve_schur_iterative_with_delta2(
        const spmat &G, const std::function<dvec(const dvec &)> &Ks,
        const dvec &rhs_s, const spmat &K, const ChompConfig &cfg,
        bool use_prec) const {
        const int m = G.rows();

        // Base operator: S y = G K^{-1} Gᵀ y
        LinOp S_op{m, [&](const dvec &y, dvec &out) {
                       out = G * Ks(G.transpose() * y);
                   }};

        std::optional<dvec> JacobiMinv;
        std::optional<SSORPrecond> ssor;

        if (use_prec) {
            // Build preconditioner for Ŝ = G * diag(K^{-1}) * Gᵀ
            // 1) approximate diag(K^{-1}) via 1 / max(diag(K), eps)
            dvec diagK = K.diagonal();
            for (int i = 0; i < diagK.size(); ++i)
                diagK[i] = 1.0 / std::max(std::abs(diagK[i]), 1e-12);

            if (cfg.prec_type == "jacobi") {
                // Jacobi on Ŝ -> M^{-1} = diag(Ŝ)^{-1}
                dvec dS = schur_diag_hat(G, diagK);
                for (int i = 0; i < dS.size(); ++i)
                    dS[i] = 1.0 / std::max(dS[i], 1e-18);
                JacobiMinv = std::move(dS);
            } else if (cfg.prec_type == "ssor") {
                // SSOR on Ŝ
                spmat S_hat = build_S_hat(G, diagK);
                ssor.emplace(S_hat, cfg.ssor_omega);
            }
            // else "none" -> no preconditioner
        }

        // First CG attempt
        auto [y, info] = cg(S_op, rhs_s, JacobiMinv, cfg.cg_tol, cfg.cg_maxit,
                            std::nullopt, ssor, cfg.use_simd);
        if (info.converged)
            return y;

        // δ₂ shift and retry: (S + δ₂ I) y = rhs
        double delta2 = std::max(1e-12, cfg.schur_delta2_min);
        for (int tries = 0; tries < 5; ++tries) {
            LinOp Sshift{m, [&](const dvec &v, dvec &out) {
                             out = G * Ks(G.transpose() * v);
                             out.noalias() += delta2 * v;
                         }};
            std::tie(y, info) =
                cg(Sshift, rhs_s, JacobiMinv, cfg.cg_tol, cfg.cg_maxit2,
                   std::nullopt, ssor, cfg.use_simd);
            if (info.converged)
                return y;
            delta2 = std::min(cfg.schur_delta2_max, 10.0 * delta2);
        }

        throw std::runtime_error("HYKKT: CG on Schur failed even with δ₂.");
    }

    // ==================== helpers reused from your code =====================
    // Diagonal of S_hat = G * diag(d) * Gᵀ
    static dvec schur_diag_hat(const spmat &G, const dvec &d) {
        const int m = G.rows();
        dvec diag = dvec::Zero(m);
        for (int j = 0; j < G.outerSize(); ++j)
            for (spmat::InnerIterator it(G, j); it; ++it)
                diag[it.row()] += (it.value() * it.value()) * d[j];
        return diag;
    }

    // Build full S_hat matrix for SSOR preconditioner
    static spmat build_S_hat(const spmat &G, const dvec &d) {
        const int m = G.rows();
        const int n = G.cols();

        // Create diagonal matrix from d
        spmat D(n, n);
        D.reserve(n);
        for (int i = 0; i < n; ++i) {
            D.insert(i, i) = d[i];
        }
        D.makeCompressed();

        // Compute S_hat = G * D * G^T
        spmat Gt = G.transpose();
        return (G * D * Gt).pruned();
    }

    // ==================== reusable wrapper ================
    std::shared_ptr<KKTReusable>
    create_reusable_solver(const spmat &G,
                           const std::function<dvec(const dvec &)> &Ks,
                           double gamma, const ChompConfig &cfg) const {
        struct Reuse final : KKTReusable {
            spmat G;
            std::function<dvec(const dvec &)> Ks;
            double gamma;
            ChompConfig cfg;
            ModelC *model_ptr;

            Reuse(spmat Gin, std::function<dvec(const dvec &)> Ksin, double g,
                  ChompConfig c, ModelC *m)
                : G(std::move(Gin)), Ks(std::move(Ksin)), gamma(g),
                  cfg(std::move(c)), model_ptr(m) {}

            std::pair<dvec, dvec> solve(const dvec &r1n,
                                        const std::optional<dvec> &r2n,
                                        double tol, int maxit) override {
                if (!r2n)
                    throw std::invalid_argument("HYKKT::Reuse needs r2");

                const dvec s = r1n + gamma * (G.transpose() * (*r2n));
                const dvec rhs_s = G * Ks(s) - (*r2n);

                // Use more relaxed tolerances for the reusable path
                double reuse_tol = std::max(tol, 1e-8);
                int reuse_maxit = std::max(maxit, 100);

                LinOp S_op{static_cast<int>(G.rows()),
                           [&](const dvec &y, dvec &out) {
                               out = G * Ks(G.transpose() * y);
                           }};

                auto [dy, info] =
                    cg(S_op, rhs_s, std::nullopt, reuse_tol, reuse_maxit,
                       std::nullopt, std::nullopt, cfg.use_simd);
                if (info.converged) {
                    const dvec dx = Ks(s - G.transpose() * dy);
                    return {dx, dy};
                }

                // First fallback: try with even more relaxed tolerance
                reuse_tol = std::max(reuse_tol * 10, 1e-6);
                std::tie(dy, info) =
                    cg(S_op, rhs_s, std::nullopt, reuse_tol, reuse_maxit * 2,
                       std::nullopt, std::nullopt, cfg.use_simd);
                if (info.converged) {
                    const dvec dx = Ks(s - G.transpose() * dy);
                    return {dx, dy};
                }

                // Second fallback: δ₂ shift with relaxed tolerance
                double d2 = std::max(1e-10, cfg.schur_delta2_min);
                for (int shift_tries = 0; shift_tries < 3; ++shift_tries) {
                    LinOp Sshift{static_cast<int>(G.rows()),
                                 [&](const dvec &v, dvec &out) {
                                     out = G * Ks(G.transpose() * v);
                                     out.noalias() += d2 * v;
                                 }};
                    std::tie(dy, info) =
                        cg(Sshift, rhs_s, std::nullopt, reuse_tol, reuse_maxit,
                           std::nullopt, std::nullopt, cfg.use_simd);
                    if (info.converged) {
                        const dvec dx = Ks(s - G.transpose() * dy);
                        return {dx, dy};
                    }
                    d2 *= 100.0; // Increase regularization more aggressively
                    reuse_tol *= 10.0; // Further relax tolerance
                }

                // Final fallback: try direct dense solve if problem is small
                const int m = G.rows();
                if (m <= 500) { // Only for reasonably small problems
                    try {
                        // Build Schur complement explicitly
                        const spmat Gt = G.transpose();
                        dmat Z(G.cols(), m);
                        dvec temp(G.cols());

                        for (int j = 0; j < m; ++j) {
                            temp.setZero();
                            for (spmat::InnerIterator it(Gt, j); it; ++it) {
                                temp[it.row()] = it.value();
                            }
                            Z.col(j) = Ks(temp);
                        }

                        dmat S = (G * Z).selfadjointView<Eigen::Lower>();
                        S.diagonal().array() +=
                            std::max(d2, 1e-8); // Add regularization

                        Eigen::LDLT<dmat> ldlt(S);
                        if (ldlt.info() == Eigen::Success) {
                            dy = ldlt.solve(rhs_s);
                            const dvec dx = Ks(s - G.transpose() * dy);
                            return {dx, dy};
                        }
                    } catch (...) {
                        // Dense fallback failed, continue to error
                    }
                }

                throw std::runtime_error(
                    "HYKKT::Reuse failed after all fallback attempts. "
                    "Try increasing tolerances or disabling HVP optimization.");
            }
        };
        return std::make_shared<Reuse>(G, Ks, gamma, cfg, model);
    }
};

// ------------------------------ LDL (QDLDL) ---------------------------
// ------------------------------ LDL (QDLDL) ---------------------------
class LDLStrategy final : public KKTStrategy {
public:
    LDLStrategy() { name = "ldl"; }

    // Optional knobs (could be pulled from ChompConfig)
    struct Options {
        bool use_symbolic_cache = true;
        bool use_ordering = false;          // set true if you have/compute a perm
        int  refine_iters = 0;              // 0 = off, 1-2 usually enough
    } opts;

    // Lightweight cache keyed by sparsity pattern of U (Ap/Ai)
    struct PatternKey {
        std::size_t h1{0}, h2{0};
        bool operator==(const PatternKey& o) const noexcept {
            return h1 == o.h1 && h2 == o.h2;
        }
    };
    struct PatternKeyHasher {
        std::size_t operator()(const PatternKey& k) const noexcept {
            // simple mix
            return k.h1 ^ (k.h2 + 0x9e3779b97f4a7c15ULL + (k.h1<<6) + (k.h1>>2));
        }
    };

    struct CachedSymb {
        qdldl23::Symb32 S;                 // symbolic (on possibly permuted pattern)
        std::optional<qdldl23::Ordering<int32_t>> ord;
        int n{0};
    };

    // tiny LRU-ish (size <= 4) implemented as an unordered_map + clock
    std::unordered_map<PatternKey, CachedSymb, PatternKeyHasher> symb_cache_;
    std::deque<PatternKey> cache_order_; // maintain recency
    static constexpr size_t kCacheCap = 4;

    ModelC *model = nullptr;

    static PatternKey make_key(const qdldl23::SparseD32& U) {
        // Hash Ap and first/last few Ai to be cheap yet robust.
        // For rock-solid uniqueness, hash the whole Ai; adjust as you like.
        auto mix = [](std::size_t seed, uint64_t v){
            v ^= v >> 33; v *= 0xff51afd7ed558ccdULL;
            v ^= v >> 33; v *= 0xc4ceb9fe1a85ec53ULL;
            v ^= v >> 33; return seed ^ v;
        };
        std::size_t hAp = 0;
        for (int i = 0; i < static_cast<int>(U.Ap.size()); ++i)
            hAp = mix(hAp, static_cast<uint64_t>(U.Ap[static_cast<size_t>(i)]) + 0x9e37U*i);

        std::size_t hAi = 0;
        const size_t nnz = U.Ai.size();
        const int take = 64; // sample
        for (int i = 0; i < std::min<int>(take, (int)nnz); ++i)
            hAi = mix(hAi, static_cast<uint64_t>(U.Ai[static_cast<size_t>(i)]) + 0x85ebU*i);
        for (int i = (int)nnz - 1, k = 0; i >= 0 && k < take; --i, ++k)
            hAi = mix(hAi, static_cast<uint64_t>(U.Ai[static_cast<size_t>(i)]) + 0xc2b2U*k);
        return {hAp, hAi};
    }

    void touch_key_(const PatternKey& k) {
        // move to back; drop old if > cap
        auto it = std::find(cache_order_.begin(), cache_order_.end(), k);
        if (it != cache_order_.end()) cache_order_.erase(it);
        cache_order_.push_back(k);
        while (cache_order_.size() > kCacheCap) {
            auto victim = cache_order_.front(); cache_order_.pop_front();
            symb_cache_.erase(victim);
        }
    }

    // Hook point: provide a symmetric ordering if you have one (AMD, RCM, etc.)
    // Return std::nullopt for identity.
    std::optional<qdldl23::Ordering<int32_t>>
    maybe_build_ordering_(const qdldl23::SparseD32& /*U*/, bool enable) const {
        if (!enable) return std::nullopt;
        // Example placeholder: identity
        // return qdldl23::Ordering<int32_t>::identity((int32_t)/*U.n*/ U.n);
    }

    std::tuple<dvec, dvec, std::shared_ptr<KKTReusable>> factor_and_solve(
        ModelC *model_in, const spmat &W, const std::optional<spmat> &Gopt,
        const dvec &r1, const std::optional<dvec> &r2opt, const ChompConfig &cfg,
        std::optional<double> /*regularizer*/,
        std::unordered_map<std::string, dvec> & /*cache*/, double delta,
        std::optional<double> /*gamma*/, bool /*assemble_schur_if_m_small*/,
        bool /*use_prec*/) override
    {
        (void)cfg;
        bool hasE = false;
        spmat K = assemble_KKT(W, delta, Gopt, &hasE);

        const int n = W.rows();
        const int m = hasE ? Gopt->rows() : 0;
        const int N = hasE ? (n + m) : n;

        dvec rhs(N);
        if (hasE) {
            if (!r2opt)
                throw std::runtime_error("LDLStrategy: missing r2");
            rhs.head(n) = r1;
            rhs.tail(m) = *r2opt;
        } else {
            rhs = r1;
        }

        // Convert to strict upper CSC for qdldl23
        // Prefer to enforce diagonal in K assembly; if not, keep small diag_eps.
        auto U = eigen_to_upper_csc(K, 1e-12);

        // ---- Symbolic (and optional ordering) reuse path ----
        qdldl23::LDL32 F;
        qdldl23::Symb32 S;
        std::optional<qdldl23::Ordering<int32_t>> ord;

        const PatternKey key = make_key(U);
        bool used_cached_symbolic = false;

        if (opts.use_symbolic_cache) {
            auto it = symb_cache_.find(key);
            if (it != symb_cache_.end()) {
                // Found cached symbolic (already matches ordering choice)
                const auto &C = it->second;
                if (C.n == U.n) {
                    S = C.S; ord = C.ord;
                    used_cached_symbolic = true;
                    touch_key_(key);
                }
            }
        }

        if (!used_cached_symbolic) {
            // Decide ordering
            ord = maybe_build_ordering_(U, opts.use_ordering);
            if (ord) {
                // Analyze permuted pattern once
                auto Up = qdldl23::permute_symmetric_upper(U, *ord);
                S = qdldl23::analyze_fast(Up);
                // Numeric on permuted A
                F = qdldl23::refactorize(Up, S);
            } else {
                // No ordering
                S = qdldl23::analyze_fast(U);
                F = qdldl23::refactorize(U, S);
            }

            if (opts.use_symbolic_cache) {
                CachedSymb C; C.S = S; C.ord = ord; C.n = U.n;
                symb_cache_[key] = C; touch_key_(key);
            }
        } else {
            // Have S (and maybe ord). Just numeric refactorize on the same pattern.
            if (ord) {
                auto Up = qdldl23::permute_symmetric_upper(U, *ord);
                F = qdldl23::refactorize(Up, S);
            } else {
                F = qdldl23::refactorize(U, S);
            }
        }

        // Solve
        dvec x = rhs;
        if (ord) {
            qdldl23::solve_with_ordering(F, *ord, x.data());
        } else {
            qdldl23::solve(F, x.data());
        }

        // if (opts.refine_iters > 0) {
        //     // Refinement expects the same matrix that was factorized
        //     if (ord) {
        //         // Build permuted U for residual (same as used in factorization)
        //         auto Up = qdldl23::permute_symmetric_upper(U, *ord);
        //         qdldl23::refine(Up, F, x.data(), rhs.data(), opts.refine_iters, &(*ord));
        //     } else {
        //         qdldl23::refine(U, F, x.data(), rhs.data(), opts.refine_iters, nullptr);
        //     }
        // }

        dvec dx, dy;
        if (hasE) {
            dx = x.head(n);
            dy = x.tail(m);
        } else {
            dx = x;
            dy.resize(0);
        }

        // Reusable object: keep U, S, ord, and F for fast multi-RHS solves
        struct Reuse final : KKTReusable {
            qdldl23::SparseD32 U;
            qdldl23::Symb32    S;
            std::optional<qdldl23::Ordering<int32_t>> ord;
            qdldl23::LDL32     F;
            int n, m;
            int refine_iters{0};

            Reuse(qdldl23::SparseD32 U_, qdldl23::Symb32 S_,
                  std::optional<qdldl23::Ordering<int32_t>> ord_,
                  qdldl23::LDL32 F_, int n_, int m_, int refine_it)
                : U(std::move(U_)), S(std::move(S_)), ord(std::move(ord_)),
                  F(std::move(F_)), n(n_), m(m_), refine_iters(refine_it) {}

            std::pair<dvec, dvec> solve(const dvec &r1n,
                                        const std::optional<dvec> &r2n,
                                        double /*cg_tol*/,
                                        int /*cg_maxit*/) override {
                const int N = n + m;
                dvec rhs(N);

                if (m > 0) {
                    if (!r2n)
                        throw std::runtime_error("LDL::Reuse: missing r2");
                    rhs.head(n) = r1n;
                    rhs.tail(m) = *r2n;
                } else {
                    rhs = r1n;
                }

                if (ord) {
                    qdldl23::solve_with_ordering(F, *ord, rhs.data());
                } else {
                    qdldl23::solve(F, rhs.data());
                }
                // if (refine_iters > 0) {
                //     if (ord) {
                //         auto Up = qdldl23::permute_symmetric_upper(U, *ord);
                //         qdldl23::refine(Up, F, rhs.data(), rhs.data(), refine_iters, &(*ord));
                //     } else {
                //         qdldl23::refine(U, F, rhs.data(), rhs.data(), refine_iters, nullptr);
                //     }
                // }

                if (m > 0)
                    return {rhs.head(n), rhs.tail(m)};
                return {rhs, dvec()};
            }
        };

        auto res = std::make_shared<Reuse>(qdldl23::SparseD32(U), S, ord, F, n, m, opts.refine_iters);
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
        registry.register_strategy(std::make_shared<LDLStrategy>());
        registry.register_strategy(std::make_shared<IPEnhancedHYKKTStrategy>());
    });

    return registry;
}

} // namespace kkt
