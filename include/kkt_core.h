#pragma once
// Optimized C++23 KKT core (HYKKT + LDL) with Eigen
// AVX guard fixed; ILU removed; Schur solves use CG + (Jacobi|SSOR).

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/OrderingMethods>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>

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

namespace kkt {

// ------------------------------ typedefs ------------------------------
using dvec = Eigen::VectorXd;
using dmat = Eigen::MatrixXd;
using spmat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

// ------------------------------ config -------------------------------
struct ChompConfig {
    double cg_tol = 1e-6;
    int cg_maxit = 200;
    double ip_hess_reg0 = 1e-8;
    double schur_dense_cutoff = 0.25;
    std::string prec_type = "jacobi"; // "jacobi" | "ssor" | "none"
    double ssor_omega = 1.0;
    std::string sym_ordering = "none"; // "amd" | "none"
    bool use_simd = false;
    int block_size = 256;
    bool adaptive_gamma = true;
};

// ------------------------------ SIMD optimized helpers
// ------------------------------
#if defined(__AVX2__)
[[nodiscard]] inline double simd_dot_product(const double *a, const double *b,
                                             size_t n) noexcept {
    __m256d acc = _mm256_setzero_pd();
    size_t i = 0;

    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
#if defined(__FMA__)
        acc = _mm256_fmadd_pd(va, vb, acc);
#else
        acc = _mm256_add_pd(acc, _mm256_mul_pd(va, vb));
#endif
    }

    // horizontal sum
    __m128d lo = _mm256_castpd256_pd128(acc);
    __m128d hi = _mm256_extractf128_pd(acc, 1);
    __m128d sum128 = _mm_add_pd(lo, hi);
    __m128d hi64 = _mm_unpackhi_pd(sum128, sum128);
    double s = _mm_cvtsd_f64(_mm_add_sd(sum128, hi64));

    for (; i < n; ++i)
        s += a[i] * b[i];
    return s;
}

[[nodiscard]] inline void simd_axpy(double alpha, const double *x, double *y,
                                    size_t n) noexcept {
    __m256d va = _mm256_set1_pd(alpha);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d vx = _mm256_loadu_pd(x + i);
        __m256d vy = _mm256_loadu_pd(y + i);
#if defined(__FMA__)
        __m256d r = _mm256_fmadd_pd(va, vx, vy);
#else
        __m256d r = _mm256_add_pd(vy, _mm256_mul_pd(va, vx));
#endif
        _mm256_storeu_pd(y + i, r);
    }
    for (; i < n; ++i)
        y[i] += alpha * x[i];
}
#endif

// ------------------------------ optimized helpers
// ------------------------------
[[nodiscard]] inline double rowsum_inf_norm(const spmat &A) noexcept {
    double mx = 0.0;
    for (int j = 0; j < A.outerSize(); ++j) {
        double col = 0.0;
        for (spmat::InnerIterator it(A, j); it; ++it) {
            col += std::abs(it.value());
        }
        mx = std::max(mx, col);
    }
    return mx;
}

[[nodiscard]] inline dvec diag_GtG(const spmat &G) {
    dvec d = dvec::Zero(G.cols());
    for (int j = 0; j < G.outerSize(); ++j) {
        for (spmat::InnerIterator it(G, j); it; ++it) {
            const double val = it.value();
            d[it.col()] += val * val;
        }
    }
    return d;
}

[[nodiscard]] inline dvec schur_diag_hat(const spmat &G, const dvec &diagKinv) {
    const int m = G.rows();
    dvec out = dvec::Zero(m);

    for (int j = 0; j < G.outerSize(); ++j) {
        const double invjj = diagKinv[j];
        for (spmat::InnerIterator it(G, j); it; ++it) {
            const double gij = it.value();
            out[it.row()] += gij * gij * invjj;
        }
    }
    return out.unaryExpr([](double x) { return std::max(x, 1e-12); });
}

[[nodiscard]] inline spmat build_S_hat(const spmat &G, const dvec &diagKinv) {
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(G.nonZeros());

    for (int j = 0; j < G.outerSize(); ++j) {
        const double s = std::sqrt(std::max(1e-18, diagKinv[j]));
        for (spmat::InnerIterator it(G, j); it; ++it) {
            triplets.emplace_back(it.row(), it.col(), it.value() * s);
        }
    }

    spmat Gs(G.rows(), G.cols());
    Gs.setFromTriplets(triplets.begin(), triplets.end());
    Gs.makeCompressed();

    spmat S = (Gs * Gs.transpose()).pruned();
    S.makeCompressed();
    return S;
}

[[nodiscard]] inline double estimate_condition_number(const spmat &A,
                                                      double reg = 0.0) {
    const int n = A.rows();
    if (n == 0)
        return 1.0;

    dvec v = dvec::Random(n).normalized();
    dvec Av(n);

    // Power iteration for largest eigenvalue
    double lambda_max = 0.0;
    for (int i = 0; i < 5; ++i) {
        Av = A * v;
        if (reg != 0.0)
            Av += reg * v;
        lambda_max = v.dot(Av);
        const double norm = Av.norm();
        if (norm > 1e-16)
            v = Av / norm;
    }

    // crude lower bound via diagonal
    const dvec diag = A.diagonal().array().abs();
    const double lambda_min = std::max(diag.minCoeff() + reg, 1e-16);

    return std::abs(lambda_max) / lambda_min;
}

// ------------------------- linear operator & CG -----------------------
struct LinOp {
    int n = 0;
    std::function<void(const dvec &, dvec &)> mv;

    void apply(const dvec &x, dvec &y) const {
        y.setZero();
        mv(x, y);
    }
};

struct CGInfo {
    int iters = 0;
    bool converged = false;
    double final_residual = 0.0;
};

// SSOR (symmetric) preconditioner (valid for SPD)
class SSORPrecond {
private:
    spmat Lw, Uw;
    dvec D;
    double scale = 1.0;

public:
    SSORPrecond() = default;

    SSORPrecond(const spmat &S_hat, double omega) { compute(S_hat, omega); }

    void compute(const spmat &S_hat, double omega) {
        omega = std::clamp(omega, 1e-6, 2.0 - 1e-6);
        const int m = S_hat.rows();

        D = S_hat.diagonal().cwiseMax(1e-18);

        std::vector<Eigen::Triplet<double>> TL, TU;
        const size_t reserve_size = S_hat.nonZeros();
        TL.reserve(reserve_size);
        TU.reserve(reserve_size);

        for (int j = 0; j < S_hat.outerSize(); ++j) {
            for (spmat::InnerIterator it(S_hat, j); it; ++it) {
                const int i = it.row();
                const double v = it.value();

                if (i == j) {
                    TL.emplace_back(i, j, D[i]);
                    TU.emplace_back(i, j, D[i]);
                } else if (i > j) {
                    TL.emplace_back(i, j, omega * v);
                } else {
                    TU.emplace_back(i, j, omega * v);
                }
            }
        }

        Lw.resize(m, m);
        Uw.resize(m, m);
        Lw.setFromTriplets(TL.begin(), TL.end());
        Uw.setFromTriplets(TU.begin(), TU.end());
        Lw.makeCompressed();
        Uw.makeCompressed();

        scale = (2.0 - omega) / omega;
    }

    void apply(const dvec &r, dvec &y) const {
        dvec z = Lw.triangularView<Eigen::Lower>().solve(r);
        dvec t = D.cwiseProduct(z);
        y = Uw.triangularView<Eigen::Upper>().solve(t);
        y *= scale;
    }
};

// Optimized CG for SPD with symmetric preconditioning (Jacobi/SSOR)
[[nodiscard]] inline std::pair<dvec, CGInfo>
cg(const LinOp &A, const dvec &b,
   const std::optional<dvec> &JacobiMinvDiag = std::nullopt, double tol = 1e-10,
   int maxit = 200, const std::optional<dvec> &x0 = std::nullopt,
   const std::optional<SSORPrecond> &ssor = std::nullopt,
   bool use_simd = true) {

    const int n = A.n;
    dvec x = x0.value_or(dvec::Zero(n));
    dvec Ax(n), r(n), z(n), p(n);

    A.apply(x, Ax);
    r = b - Ax;

    // preconditioned residual
    if (ssor) {
        ssor->apply(r, z);
    } else if (JacobiMinvDiag) {
        z = r.cwiseProduct(*JacobiMinvDiag);
    } else {
        z = r;
    }

    p = z;

    double rz;
#if defined(__AVX2__)
    if (use_simd && n >= 4)
        rz = simd_dot_product(r.data(), z.data(), n);
    else
        rz = r.dot(z);
#else
    rz = r.dot(z);
#endif

    const double nrm0 = r.norm();
    const double stop = std::max(tol * nrm0, 1e-16);

    CGInfo info{};
    if (nrm0 <= stop) {
        info.converged = true;
        info.final_residual = nrm0;
        return {x, info};
    }

    double prev_residual = nrm0;
    int stagnation_count = 0;

    for (int k = 1; k <= maxit; ++k) {
        A.apply(p, Ax);

        double pAp;
#if defined(__AVX2__)
        if (use_simd && n >= 4)
            pAp = simd_dot_product(p.data(), Ax.data(), n);
        else
            pAp = p.dot(Ax);
#else
        pAp = p.dot(Ax);
#endif
        pAp = std::max(pAp, 1e-300);
        const double alpha = rz / pAp;

#if defined(__AVX2__)
        if (use_simd && n >= 4) {
            simd_axpy(alpha, p.data(), x.data(), n);
            simd_axpy(-alpha, Ax.data(), r.data(), n);
        } else {
            x.noalias() += alpha * p;
            r.noalias() -= alpha * Ax;
        }
#else
        x.noalias() += alpha * p;
        r.noalias() -= alpha * Ax;
#endif

        const double current_residual = r.norm();
        info.final_residual = current_residual;

        if (current_residual <= stop) {
            info.converged = true;
            info.iters = k;
            return {x, info};
        }

        // Apply preconditioner
        if (ssor) {
            ssor->apply(r, z);
        } else if (JacobiMinvDiag) {
            z = r.cwiseProduct(*JacobiMinvDiag);
        } else {
            z = r;
        }

        double rz_new;
#if defined(__AVX2__)
        if (use_simd && n >= 4)
            rz_new = simd_dot_product(r.data(), z.data(), n);
        else
            rz_new = r.dot(z);
#else
        rz_new = r.dot(z);
#endif

        const double beta = rz_new / std::max(rz, 1e-300);
        p = z + beta * p;
        rz = rz_new;

        // mild restart heuristic
        if (k > 5 && current_residual / prev_residual > 0.98) {
            stagnation_count++;
            if (stagnation_count > 5) {
                p = z; // restart
                stagnation_count = 0;
            }
        } else {
            stagnation_count = 0;
        }
        prev_residual = current_residual;
        info.iters = k;
    }

    return {x, info};
}

// ---------------------------- reusable API ----------------------------
struct KKTReusable {
    virtual ~KKTReusable() = default;
    virtual std::pair<dvec, dvec> solve(const dvec &r1,
                                        const std::optional<dvec> &r2,
                                        double cg_tol = 1e-8,
                                        int cg_maxit = 200) = 0;
};

// ---------------------------- strategy base ---------------------------
struct KKTStrategy {
    virtual ~KKTStrategy() = default;
    virtual std::tuple<dvec, dvec, std::shared_ptr<KKTReusable>>
    factor_and_solve(const spmat &W, const std::optional<spmat> &G,
                     const dvec &r1, const std::optional<dvec> &r2,
                     const ChompConfig &cfg, std::optional<double> regularizer,
                     std::unordered_map<std::string, dvec> &cache,
                     double delta = 0.0,
                     std::optional<double> gamma = std::nullopt,
                     bool assemble_schur_if_m_small = true,
                     bool use_prec = true) = 0;
    std::string name;
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

        // Gamma selection with caching
        double gamma;
        if (gamma_user) {
            gamma = *gamma_user;
        } else if (cfg.adaptive_gamma) {
            std::string cache_key =
                "gamma_" + std::to_string(n) + "_" + std::to_string(m);
            if (cache.find(cache_key) != cache.end()) {
                gamma = cache[cache_key][0];
            } else {
                gamma = compute_adaptive_gamma(W, G, delta);
                cache[cache_key] = dvec::Constant(1, gamma);
            }
        } else {
            gamma = compute_gamma_heuristic(W, G, delta);
        }

        // Build augmented system
        spmat K = build_augmented_system(W, G, delta, gamma);

        // Factorization with ordering
        auto [solver, is_spd] = create_solver(K, cfg);
        (void)is_spd; // not used downstream

        // Solve Schur system
        auto [dx, dy] =
            solve_schur_system(G, K, r1, r2, gamma, solver, cfg,
                               assemble_schur_if_m_small, use_prec, n, m);

        // Create reusable solver
        auto reusable = create_reusable_solver(G, solver, gamma, cfg);

        return std::make_tuple(dx, dy, reusable);
    }

private:
    double compute_gamma_heuristic(const spmat &W, const spmat &G,
                                   double delta) const {
        const double W_norm = rowsum_inf_norm(W) + delta;
        const double G_norm = rowsum_inf_norm(G);
        return std::max(1.0, W_norm / std::max(1.0, G_norm * G_norm));
    }

    double compute_adaptive_gamma(const spmat &W, const spmat &G,
                                  double delta) const {
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

    spmat build_augmented_system(const spmat &W, const spmat &G, double delta,
                                 double gamma) const {
        spmat K = W;

        if (delta != 0.0) {
            spmat I(W.rows(), W.rows());
            I.setIdentity();
            K = (K + delta * I).pruned();
        }

        if (gamma != 0.0) {
            K = (K + gamma * (G.transpose() * G).pruned()).pruned();
        }

        return K;
    }

    std::pair<std::function<dvec(const dvec &)>, bool>
    create_solver(const spmat &K, const ChompConfig &cfg) const {
        const int n = K.rows();

        // Ordering selection
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> P;
        bool use_ordering = (cfg.sym_ordering == "amd");

        if (use_ordering) {
            Eigen::AMDOrdering<int> amd;
            amd(K, P);
        } else {
            P.setIdentity(n);
        }

        spmat Kp;
        if (use_ordering) {
            spmat temp = P.transpose() * K;
            Kp = (temp * P).eval();
            Kp.pruned();
        } else {
            Kp = K;
        }
        Kp.makeCompressed();

        // Try SPD factorization first
        auto llt_ptr = std::make_shared<Eigen::SimplicialLLT<spmat>>();
        llt_ptr->compute(Kp);

        if (llt_ptr->info() == Eigen::Success) {
            auto solver = [P, llt_ptr, use_ordering](const dvec &b) -> dvec {
                if (use_ordering) {
                    const dvec bp = P.transpose() * b;
                    const dvec yp = llt_ptr->solve(bp);
                    return P * yp;
                } else {
                    return llt_ptr->solve(b);
                }
            };
            return {solver, true};
        }

        // Fall back to LDLT
        auto ldlt_ptr = std::make_shared<Eigen::SimplicialLDLT<spmat>>();
        ldlt_ptr->compute(Kp);

        if (ldlt_ptr->info() != Eigen::Success) {
            throw std::runtime_error("HYKKT: K factorization failed");
        }

        auto solver = [P, ldlt_ptr, use_ordering](const dvec &b) -> dvec {
            if (use_ordering) {
                const dvec bp = P.transpose() * b;
                const dvec yp = ldlt_ptr->solve(bp);
                return P * yp;
            } else {
                return ldlt_ptr->solve(b);
            }
        };

        return {solver, false};
    }

    std::pair<dvec, dvec>
    solve_schur_system(const spmat &G, const spmat &K, const dvec &r1,
                       const dvec &r2, double gamma,
                       const std::function<dvec(const dvec &)> &solver,
                       const ChompConfig &cfg, bool assemble_schur_if_m_small,
                       bool use_prec, int n, int m) const {

        const dvec svec = r1 + gamma * (G.transpose() * r2);
        const dvec rhs_s = G * solver(svec) - r2;

        dvec dy;
        const bool small_m =
            assemble_schur_if_m_small &&
            (m <= std::max(1, int(cfg.schur_dense_cutoff * n)));

        if (small_m) {
            // Dense Schur computation
            dmat Z(n, m);
            for (int j = 0; j < m; ++j)
                Z.col(j) = solver(G.transpose().col(j));
            const dmat S = G * Z;
            dy = Eigen::LLT<dmat>(S).solve(rhs_s);
        } else {
            // Iterative Schur solve (CG + Jacobi/SSOR)
            dy = solve_schur_iterative(G, solver, rhs_s, K, cfg, use_prec, m);
        }

        const dvec dx = solver(svec - G.transpose() * dy);
        return {dx, dy};
    }

    dvec solve_schur_iterative(const spmat &G,
                               const std::function<dvec(const dvec &)> &solver,
                               const dvec &rhs_s, const spmat &K,
                               const ChompConfig &cfg, bool use_prec,
                               int m) const {

        LinOp S_op{m, [&](const dvec &y, dvec &out) {
                       out = G * solver(G.transpose() * y);
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
            } // "none" -> no preconditioner
        }

        auto cg_res = cg(S_op, rhs_s, JacobiMinv, cfg.cg_tol, cfg.cg_maxit,
                         std::nullopt, ssor, cfg.use_simd);
        dvec dy_sol = std::move(cg_res.first);
        return dy_sol;
    }

    std::shared_ptr<KKTReusable>
    create_reusable_solver(const spmat &G,
                           const std::function<dvec(const dvec &)> &solver,
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
                    throw std::invalid_argument("HYKKT::Reuse needs r2");
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

        return std::make_shared<Reuse>(G, solver, gamma, cfg);
    }
};

// Optimized CSC conversion (upper triangle) for QDLDL
[[nodiscard]] inline qdldl23::SparseD32
eigen_to_upper_csc(const spmat &A, double diag_eps = 0.0) {
    const int n = A.rows();
    if (A.cols() != n)
        throw std::invalid_argument(
            "eigen_to_upper_csc: matrix must be square");

    std::vector<int> rows, cols;
    std::vector<double> vals;
    const size_t est_nnz = A.nonZeros() / 2 + n;
    rows.reserve(est_nnz);
    cols.reserve(est_nnz);
    vals.reserve(est_nnz);

    std::vector<bool> has_diag(n, false);

    for (int j = 0; j < A.outerSize(); ++j) {
        for (spmat::InnerIterator it(A, j); it; ++it) {
            if (it.row() <= j) {
                rows.push_back(it.row());
                cols.push_back(j);
                vals.push_back(it.value());
                if (it.row() == j)
                    has_diag[j] = true;
            }
        }
    }

    if (diag_eps > 0.0) {
        for (int j = 0; j < n; ++j) {
            if (!has_diag[j]) {
                rows.push_back(j);
                cols.push_back(j);
                vals.push_back(diag_eps);
            }
        }
    }

    std::vector<int> Ap(n + 1, 0);
    for (int c : cols)
        ++Ap[c + 1];
    for (int j = 0; j < n; ++j)
        Ap[j + 1] += Ap[j];

    const size_t nnz = Ap[n];
    std::vector<int> Ai(nnz);
    std::vector<double> Ax(nnz);
    std::vector<int> next = Ap;

    for (size_t k = 0; k < cols.size(); ++k) {
        const size_t p = next[cols[k]]++;
        Ai[p] = rows[k];
        Ax[p] = vals[k];
    }

    return qdldl23::SparseD32(n, std::move(Ap), std::move(Ai), std::move(Ax));
}

// Optimized KKT assembly
[[nodiscard]] inline spmat assemble_KKT(const spmat &W, double delta,
                                        const std::optional<spmat> &Gopt,
                                        bool *out_hasE) {
    const int n = W.rows();
    if (W.cols() != n)
        throw std::invalid_argument("assemble_KKT: W not square");

    const bool hasE = Gopt && (Gopt->rows() > 0);
    if (out_hasE)
        *out_hasE = hasE;

    if (!hasE) {
        if (delta == 0.0)
            return W;
        spmat I(n, n);
        I.setIdentity();
        return (W + delta * I).pruned();
    }

    const auto &G = *Gopt;
    const int m = G.rows();

    std::vector<Eigen::Triplet<double>> triplets;
    const size_t est_triplets =
        W.nonZeros() + 2 * G.nonZeros() + (delta != 0.0 ? n : 0);
    triplets.reserve(est_triplets);

    std::vector<bool> has_diag(n, false);
    for (int j = 0; j < W.outerSize(); ++j) {
        for (spmat::InnerIterator it(W, j); it; ++it) {
            double val = it.value();
            if (delta != 0.0 && it.row() == it.col()) {
                val += delta;
                has_diag[it.row()] = true;
            }
            triplets.emplace_back(it.row(), it.col(), val);
        }
    }
    if (delta != 0.0) {
        for (int j = 0; j < n; ++j)
            if (!has_diag[j])
                triplets.emplace_back(j, j, delta);
    }

    for (int j = 0; j < G.outerSize(); ++j) {
        for (spmat::InnerIterator it(G, j); it; ++it) {
            triplets.emplace_back(it.col(), n + it.row(), it.value()); // G^T
            triplets.emplace_back(n + it.row(), it.col(), it.value()); // G
        }
    }

    spmat K(n + m, n + m);
    K.setFromTriplets(triplets.begin(), triplets.end());
    K.makeCompressed();
    return K;
}

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
        registry.register_strategy(std::make_shared<LDLStrategy>());
    });

    return registry;
}

} // namespace kkt
