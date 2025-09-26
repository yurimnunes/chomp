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

#include "amd.h"
#include "qdldl.h"

#include "kkt_helper.h"

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
            auto [solver_fn, is_spd] = cfg.use_hvp && model ?
                create_hvp_based_solver(W, G, delta1, gamma, cfg) :
                create_or_refactor_solver_qdldl(build_augmented_system_inplace(W, G, delta1, gamma), cfg);
                
            if (solver_fn) {
                Ks = std::move(solver_fn);
                used_llt = is_spd;

                // Keep the final matrix for preconditioner (only needed for non-HVP path)
                if (!cfg.use_hvp || !model) {
                    spmat K = build_augmented_system_inplace(W, G, delta1, gamma);
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
            dy = cfg.use_hvp && model ?
                solve_schur_dense_hvp_optimized(G, Ks, rhs_s, gamma, cfg) :
                solve_schur_dense_with_delta2(G, Ks, rhs_s, cfg);
        } else {
            dy = cfg.use_hvp && model ?
                solve_schur_iterative_hvp_optimized(G, Ks, rhs_s, gamma, cfg, use_prec) :
                solve_schur_iterative_with_delta2(G, Ks, rhs_s, K_final, cfg, use_prec);
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

    // HVP-specific cache
    struct HVPCache {
        int n{-1};
        double last_delta{-1.0};
        std::function<dvec(const dvec &)> W_solver;
        bool have_W_solver{false};
    };

    mutable std::optional<QDLDLCache> qcache_;
    mutable std::optional<SymbolicCache> sym_; // persisted across calls
    mutable std::optional<HVPCache> hvp_cache_; // HVP solver cache

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

    // =============== HVP-Based Solver Creation ===============
    std::pair<std::function<dvec(const dvec &)>, bool>
    create_hvp_based_solver(const spmat &W, const spmat &G, double delta1, 
                            double gamma, const ChompConfig &cfg) const {
        if (!model) {
            // Fallback to original method if no model available
            return create_or_refactor_solver_qdldl(
                build_augmented_system_inplace(W, G, delta1, gamma), cfg);
        }
        
        const int n = W.rows();
        
        // Check if we need to rebuild W solver cache
        if (!hvp_cache_ || hvp_cache_->n != n || 
            std::abs(hvp_cache_->last_delta - delta1) > 1e-12 || !hvp_cache_->have_W_solver) {
            
            hvp_cache_.emplace();
            hvp_cache_->n = n;
            hvp_cache_->last_delta = delta1;
            
            // Factor the base system W + δI (much sparser than full K)
            spmat W_delta = W;
            W_delta.makeCompressed();
            for (int i = 0; i < n; ++i) {
                W_delta.coeffRef(i, i) += delta1;
            }
            W_delta.makeCompressed();
            
            // Use optimized QDLDL for W + δI
            auto [W_solver, _] = create_or_refactor_solver_qdldl(W_delta, cfg);
            if (!W_solver) {
                // Fallback if W + δI factorization fails
                return create_or_refactor_solver_qdldl(
                    build_augmented_system_inplace(W, G, delta1, gamma), cfg);
            }
            
            hvp_cache_->W_solver = std::move(W_solver);
            hvp_cache_->have_W_solver = true;
        }
        
        // Create matrix-free K^(-1) solver using cached W solver
        auto Ks = [this, W = W, G = G, delta1, gamma, cfg](const dvec &b) -> dvec {
            return solve_augmented_iterative(hvp_cache_->W_solver, W, G, b, delta1, gamma, cfg);
        };
        
        return {Ks, false};
    }
// K = Wop + δ1 I + γ GᵀG
dvec solve_augmented_iterative(
    const std::function<dvec(const dvec &)> &W_solver, // ≈ (W+δ1 I)^{-1} preconditioner
    const spmat &W_unused,                              // no longer used here
    const spmat &G, const dvec &b,
    double delta1, double gamma, const ChompConfig &cfg) const
{
    (void)W_unused; // avoid unused warning

    // 1) Get compiled W operator (this applies full W: H + diag(Sx) + JIᵀ diag(Ss) JI (+ optional sigma))
    auto Wop = model->getCompiledWOp();   // <-- ensure API name matches your ModelC

    auto apply_K = [&](const dvec &v, dvec &Kv) {
        Kv.resize(v.size());
        // Kv = Wop * v
        Wop.perform_op(v, Kv);
        // + δ1 * v
        Kv.noalias() += delta1 * v;
        // + γ * Gᵀ(G v)
        dvec Gv = G * v;
        Kv.noalias() += gamma * (G.transpose() * Gv);
    };

    // 2) PCG with M^{-1} ≈ (W+δ1 I)^{-1}
    const double tol   = cfg.hvp_iterative_tol.value_or(1e-10);
    const int    itmax = cfg.hvp_iterative_maxiter.value_or(std::max(20, std::min(80, (int)b.size()/8)));

    dvec x = W_solver(b);                 // warm start
    dvec Kx(b.size()); apply_K(x, Kx);
    dvec r = b - Kx;

    const double bnorm = std::max(1e-30, b.norm());
    dvec z = W_solver(r);
    dvec p = z;
    double rz_old = r.dot(z);

    for (int k = 0; k < itmax; ++k) {
        if (r.norm() <= tol * bnorm) break;
        dvec Kp(b.size()); apply_K(p, Kp);
        double denom = std::max(1e-30, p.dot(Kp));
        double alpha = rz_old / denom;

        x.noalias() += alpha * p;
        r.noalias() -= alpha * Kp;

        dvec z_new = W_solver(r);
        double rz_new = r.dot(z_new);
        double beta   = rz_new / std::max(1e-30, rz_old);
        p = z_new + beta * p;
        z.swap(z_new);
        rz_old = rz_new;
    }
    return x;
}


    // =============== Sherman-Morrison-Woodbury Direct Formula ===============
    dvec solve_via_smw_formula(const std::function<dvec(const dvec &)> &W_solver,
                              const spmat &G, const dvec &b, double gamma) const {
        // For K = W + δI + γ*G^T*G, use SMW:
        // K^(-1) = W^(-1) - γ * W^(-1) * G^T * (I + γ*G*W^(-1)*G^T)^(-1) * G * W^(-1)
        
        if (gamma < 1e-12) {
            return W_solver(b);  // No regularization needed
        }
        
        const int m = G.rows();
        const spmat Gt = G.transpose();
        
        // Step 1: w = W^(-1) * b
        dvec w = W_solver(b);
        
        // Step 2: Solve W * Z = G^T (Z is n x m)
        dmat Z(G.cols(), m);
        dvec temp(G.cols());
        
        for (int j = 0; j < m; ++j) {
            temp.setZero();
            for (spmat::InnerIterator it(Gt, j); it; ++it) {
                temp[it.row()] = it.value();
            }
            Z.col(j) = W_solver(temp);
        }
        
        // Step 3: Form S = I + γ * G * Z  (m x m matrix)
        dmat S = dmat::Identity(m, m) + gamma * (G * Z);
        
        // Step 4: Solve S * y = G * w
        dvec Gw = G * w;
        Eigen::LDLT<dmat> ldlt(S);
        if (ldlt.info() != Eigen::Success) {
            throw std::runtime_error("SMW: Failed to factor correction matrix");
        }
        dvec y = ldlt.solve(Gw);
        
        // Step 5: x = w - γ * Z * y
        return w - gamma * (Z * y);
    }

    // =============== Optimized Dense Schur with HVP ===============
    dvec solve_schur_dense_hvp_optimized(
        const spmat &G, const std::function<dvec(const dvec &)> &Ks,
        const dvec &rhs_s, double gamma, const ChompConfig &cfg) const {
        
        const int n = G.cols(), m = G.rows();
        
        // Build Schur complement matrix more efficiently
        dmat Z(n, m);
        dvec temp(n);
        
        // Solve K * Z = G^T by columns (using the matrix-free Ks)
        const spmat Gt = G.transpose();
        for (int j = 0; j < m; ++j) {
            temp.setZero();
            for (spmat::InnerIterator it(Gt, j); it; ++it) {
                temp[it.row()] = it.value();
            }
            Z.col(j) = Ks(temp);
        }
        
        // Form Schur complement S = G * Z
        dmat S = (G * Z).selfadjointView<Eigen::Lower>();
        
        // Solve with δ₂ regularization if needed
        double delta2 = 0.0;
        for (int tries = 0; tries < 5; ++tries) {
            if (delta2 > 0.0) {
                S.diagonal().array() += delta2;
            }
            
            Eigen::LLT<dmat> llt(S.selfadjointView<Eigen::Lower>());
            if (llt.info() == Eigen::Success) {
                return llt.solve(rhs_s);
            }
            
            delta2 = (delta2 == 0.0) ? std::max(1e-12, cfg.schur_delta2_min)
                                     : std::min(cfg.schur_delta2_max, 10.0 * delta2);
        }
        
        throw std::runtime_error("HVP-optimized dense Schur failed with δ₂");
    }

    // =============== Optimized Iterative Schur with HVP ===============
    dvec solve_schur_iterative_hvp_optimized(
        const spmat &G, const std::function<dvec(const dvec &)> &Ks,
        const dvec &rhs_s, double gamma, const ChompConfig &cfg, bool use_prec) const {
        
        const int m = G.rows();
        
        // Matrix-free Schur operator: S y = G * K^(-1) * G^T * y
        LinOp S_op{m, [&](const dvec &y, dvec &out) {
            dvec Gty = G.transpose() * y;
            dvec temp = Ks(Gty);
            out = G * temp;
        }};
        
        std::optional<dvec> precond;
        
        if (use_prec && cfg.prec_type == "jacobi") {
            // Improved preconditioner using diagonal approximation
            dvec diag_approx(m);
            for (int i = 0; i < m; ++i) {
                dvec ei = dvec::Zero(G.cols());
                for (spmat::InnerIterator it(G, i); it; ++it) {
                    ei[it.col()] = it.value();
                }
                dvec Ki_ei = Ks(ei);
                diag_approx[i] = ei.dot(Ki_ei);
            }
            
            for (int i = 0; i < m; ++i) {
                diag_approx[i] = 1.0 / std::max(diag_approx[i], 1e-12);
            }
            precond = std::move(diag_approx);
        }
        
        // Solve with CG
        auto [y, info] = cg(S_op, rhs_s, precond, cfg.cg_tol, cfg.cg_maxit,
                           std::nullopt, std::nullopt, cfg.use_simd);
        
        if (info.converged) return y;
        
        // Fallback with δ₂ shift
        double delta2 = std::max(1e-12, cfg.schur_delta2_min);
        LinOp S_shifted{m, [&](const dvec &v, dvec &out) {
            dvec Gtv = G.transpose() * v;
            dvec temp = Ks(Gtv);
            out = G * temp + delta2 * v;
        }};
        
        std::tie(y, info) = cg(S_shifted, rhs_s, precond, cfg.cg_tol, cfg.cg_maxit2,
                              std::nullopt, std::nullopt, cfg.use_simd);
        
        if (!info.converged) {
            throw std::runtime_error("HVP-optimized iterative Schur failed");
        }
        
        return y;
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

        // ---------- Ordering: your AMD (preferred), Eigen AMD, or identity ----------
        if (!qcache_->have_ord) {
            if (use_my_amd) {
                qcache_->ord = qdldl_order_from_my_amd(
                    K,
                    /*symmetrize_union=*/true,
                    /*dense_cutoff=*/
                    (cfg.amd_dense_cutoff_has_value ? cfg.amd_dense_cutoff : -1));
            } else if (use_eigen_amd) {
                Eigen::AMDOrdering<int> amd;
                Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> P(n);
                P.setIdentity(n);
                amd(K, P);
                std::vector<int32_t> perm_old2new((size_t)n);
                for (int i = 0; i < n; ++i)
                    perm_old2new[(size_t)P.indices()[i]] = i;
                qcache_->ord = Ordering<int32_t>::from_perm(std::move(perm_old2new));
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
                           
                auto [dy, info] = cg(S_op, rhs_s, std::nullopt, reuse_tol, reuse_maxit,
                                     std::nullopt, std::nullopt, cfg.use_simd);
                if (info.converged) {
                    const dvec dx = Ks(s - G.transpose() * dy);
                    return {dx, dy};
                }

                // First fallback: try with even more relaxed tolerance
                reuse_tol = std::max(reuse_tol * 10, 1e-6);
                std::tie(dy, info) = cg(S_op, rhs_s, std::nullopt, reuse_tol, reuse_maxit * 2,
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
                if (m <= 500) {  // Only for reasonably small problems
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
                        S.diagonal().array() += std::max(d2, 1e-8); // Add regularization
                        
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

                throw std::runtime_error("HYKKT::Reuse failed after all fallback attempts. "
                                        "Try increasing tolerances or disabling HVP optimization.");
            }
        };
        return std::make_shared<Reuse>(G, Ks, gamma, cfg, model);
    }
};

// ----------------------- HYKKT via CHOLMOD (SuiteSparse) --------------
class HYKKTCholmodStrategy final : public KKTStrategy {
public:
    HYKKTCholmodStrategy() { name = "hykkt_cholmod"; }

    std::tuple<dvec, dvec, std::shared_ptr<KKTReusable>>
    factor_and_solve(ModelC* model_in, const spmat &W, const std::optional<spmat> &Gopt,
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

    ModelC *model = nullptr; // To access model parameters if needed

    std::tuple<dvec, dvec, std::shared_ptr<KKTReusable>>
    factor_and_solve(ModelC* model_in, const spmat &W, const std::optional<spmat> &Gopt,
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
