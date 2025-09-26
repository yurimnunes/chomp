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

// --- tiny CSR builder from Eigen (pattern-only)
static CSR eigen_to_csr_pattern(const spmat &A) {
    const int n = A.rows();
    std::vector<std::vector<i32>> rows(n);
    for (int j = 0; j < A.outerSize(); ++j) {
        for (spmat::InnerIterator it(A, j); it; ++it) {
            rows[it.row()].push_back(it.col());
        }
    }
    CSR C(n);
    C.indptr[0] = 0;
    for (int i = 0; i < n; ++i) {
        auto &r = rows[i];
        std::sort(r.begin(), r.end());
        r.erase(std::unique(r.begin(), r.end()), r.end());
        C.indptr[i + 1] = C.indptr[i] + (i32)r.size();
    }
    C.indices.resize(C.indptr.back());
    for (int i = 0, w = 0; i < n; ++i)
        for (int v : rows[i])
            C.indices[(size_t)w++] = (i32)v;
    return C;
}

// --- build qdldl23::Ordering from your AMD
static qdldl23::Ordering<int32_t>
qdldl_order_from_my_amd(const spmat &K, bool symmetrize_union = true,
                        int dense_cutoff = -1) {
    // 1) pattern to CSR
    CSR A = eigen_to_csr_pattern(K);

    // 2) run your AMD → p_new2old
    AMDReorderingArray my_amd(/*aggressive_absorption=*/true, dense_cutoff);
    std::vector<i32> p_new2old =
        my_amd.amd_order(A, /*symmetrize=*/symmetrize_union);

    // 3) invert to old->new as qdldl expects
    std::vector<int32_t> perm_old2new(p_new2old.size());
    for (int32_t newi = 0; newi < (int32_t)p_new2old.size(); ++newi) {
        int32_t oldi = p_new2old[(size_t)newi];
        perm_old2new[(size_t)oldi] = newi;
    }

    return qdldl23::Ordering<int32_t>::from_perm(std::move(perm_old2new));
}

namespace kkt {
// Faster KKT assembly: single copy of W, no extra diagonal pass, streamed
// inserts.
// ====================== assemble_KKT (row-major right block)
// ======================
[[nodiscard]] inline spmat assemble_KKT(spmat W, double delta,
                                        const std::optional<spmat> &Gopt,
                                        bool *out_hasE) {
    const int n = W.rows();
    if (W.cols() != n)
        throw std::invalid_argument("assemble_KKT: W not square");

    W.makeCompressed();

    // In-place diagonal regularization
    if (delta != 0.0) {
        for (int i = 0; i < n; ++i)
            W.coeffRef(i, i) += delta;
        W.makeCompressed();
    }

    const bool hasE = Gopt && (Gopt->rows() > 0);
    if (out_hasE)
        *out_hasE = hasE;
    if (!hasE)
        return W;

    const spmat &G = *Gopt;
    if (G.cols() != n)
        throw std::invalid_argument("assemble_KKT: G.cols != n");
    const int m = G.rows();
    const int N = n + m;

    // Row-major view for fast row iteration on G
    using spmatR = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;
    spmatR G_r = G; // copies structure+values into row-major

    Eigen::VectorXi reserve(N);
    reserve.setZero();

    // Left block (0..n-1): W col nnz + G col nnz
    for (int j = 0; j < n; ++j) {
        const int wj = W.outerIndexPtr()[j + 1] - W.outerIndexPtr()[j];
        int gj = 0;
        for (spmat::InnerIterator it(G, j); it; ++it)
            ++gj;
        reserve[j] = wj + gj;
    }
    // Right block (n..n+m-1): G row nnz
    for (int i = 0; i < G_r.outerSize(); ++i) {
        int nnz = 0;
        for (spmatR::InnerIterator it(G_r, i); it; ++it)
            ++nnz;
        reserve[n + i] = nnz;
    }

    spmat K(N, N);
    K.reserve(reserve);

    // Stream columns 0..n-1
    for (int j = 0; j < n; ++j) {
        K.startVec(j);
        // W block
        for (spmat::InnerIterator it(W, j); it; ++it)
            K.insertBack(it.row(), j) = it.value();
        // G block (bottom-left)
        for (spmat::InnerIterator it(G, j); it; ++it)
            K.insertBack(n + it.row(), j) = it.value();
    }

    // Stream columns n..n+m-1 via row-major G (top-right = G^T)
    for (int i = 0; i < G_r.outerSize(); ++i) {
        const int col = n + i;
        K.startVec(col);
        for (spmatR::InnerIterator it(G_r, i); it; ++it) {
            K.insertBack(it.col(), col) = it.value();
        }
        // bottom-right is structurally zero
    }

    K.finalize();
    return K;
}

// ------------------------------ QDLDL helpers -------------------------
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

// ------------------------------ SIMD optimized helpers ----------------
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

// ------------------------------ optimized helpers --------------------
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

[[nodiscard]] inline spmat build_S_hat(const spmat &G, const dvec &diagKinv) {
    const int m = G.rows();
    const int n = G.cols();
    if (diagKinv.size() != n)
        throw std::invalid_argument("build_S_hat: diagKinv size mismatch");

    spmat Gs = G; // copy structure+values
    Gs.makeCompressed();

    // Scale each column j by s = sqrt(max(1e-18, diagKinv[j]))
    for (int j = 0; j < Gs.outerSize(); ++j) {
        const double s = std::sqrt(std::max(1e-18, diagKinv[j]));
        if (s == 1.0)
            continue;
        for (spmat::InnerIterator it(Gs, j); it; ++it)
            it.valueRef() *= s;
    }

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
class SSORPrecond {
private:
    spmat Lw, Uw; // (D + ωL), (D + ωU)
    dvec D;       // diag(S_hat) clamped
    double scale = 1.0;

public:
    SSORPrecond() = default;
    SSORPrecond(const spmat &S_hat, double omega) { compute(S_hat, omega); }

    void compute(const spmat &S_hat, double omega) {
        // Clamp ω to (0, 2)
        omega = std::clamp(omega, 1e-6, 2.0 - 1e-6);
        const int m = S_hat.rows();
        if (S_hat.cols() != m)
            throw std::invalid_argument("SSORPrecond: S not square");

        // Diagonal (guarded)
        D = S_hat.diagonal().cwiseMax(1e-18);

        // Reserve per column (faster than a single big reserve for many
        // problems)
        Lw.resize(m, m);
        Uw.resize(m, m);
        {
            Eigen::VectorXi perCol(m);
            perCol.setZero();
            for (int j = 0; j < S_hat.outerSize(); ++j) {
                int nnz = 0;
                for (spmat::InnerIterator it(S_hat, j); it; ++it) {
                    const int i = it.row();
                    // One nonzero will go to Lw or Uw; also ensure diagonal
                    if (i != j)
                        ++nnz;
                    else
                        ++nnz;
                }
                perCol[j] = nnz;
            }
            Lw.reserve(perCol);
            Uw.reserve(perCol);
        }

        // Fill Lw / Uw in a single pass
        std::vector<Eigen::Triplet<double>> TL;
        TL.reserve(S_hat.nonZeros());
        std::vector<Eigen::Triplet<double>> TU;
        TU.reserve(S_hat.nonZeros());
        TL.emplace_back(0, 0, 0.0);
        TU.emplace_back(0, 0, 0.0); // ensure capacity pattern (removed later)

        for (int j = 0; j < S_hat.outerSize(); ++j) {
            bool wrote_diag = false;
            for (spmat::InnerIterator it(S_hat, j); it; ++it) {
                const int i = it.row();
                const double v = it.value();

                if (i == j) {
                    TL.emplace_back(i, j, D[i]); // diag into both
                    TU.emplace_back(i, j, D[i]);
                    wrote_diag = true;
                } else if (i > j) {
                    // strictly lower → Lw gets ω*v
                    TL.emplace_back(i, j, omega * v);
                } else {
                    // strictly upper → Uw gets ω*v
                    TU.emplace_back(i, j, omega * v);
                }
            }
            if (!wrote_diag) { // extremely rare if input had missing diagonal
                               // structurally
                TL.emplace_back(j, j, D[j]);
                TU.emplace_back(j, j, D[j]);
            }
        }

        // Build sparse factors
        Lw.setFromTriplets(TL.begin() + 1,
                           TL.end()); // skip the dummy (0,0) seed
        Uw.setFromTriplets(TU.begin() + 1, TU.end());
        Lw.makeCompressed();
        Uw.makeCompressed();

        // Classic SSOR scaling factor
        // M^{-1} r = (ω / (2-ω)) (D/ω + U)^{-1} D (D/ω + L)^{-1} r
        // With our (D + ωL) / (D + ωU) factors and in-place solves we
        // compensate via:
        scale = omega / (2.0 - omega);
    }

    // Apply y ← M^{-1} r using in-place triangular solves and a single diagonal
    // scale
    void apply(const dvec &r, dvec &y) const {
        y = r; // in-place rhs buffer

        // Solve (D + ωL) z = r  → y holds z
        Lw.triangularView<Eigen::Lower>().solveInPlace(y);

        // y ← D ∘ y  (elementwise)
        y.array() *= D.array();

        // Solve (D + ωU) y = y  → y holds final vector
        Uw.triangularView<Eigen::Upper>().solveInPlace(y);

        // Final SSOR scale
        y *= scale;
    }
};
// Helper to pick dot path once
inline double maybe_simd_dot(const double *a, const double *b, int n,
                             bool use_simd) {
#if defined(__AVX2__)
    if (use_simd && n >= 4)
        return simd_dot_product(a, b, n);
#endif
    return Eigen::Map<const dvec>(a, n).dot(Eigen::Map<const dvec>(b, n));
}

// Optimized CG for SPD with symmetric preconditioning (Jacobi / SSOR)
[[nodiscard]] inline std::pair<dvec, CGInfo>
cg(const LinOp &A, const dvec &b,
   const std::optional<dvec> &JacobiMinvDiag = std::nullopt, double tol = 1e-10,
   int maxit = 200, const std::optional<dvec> &x0 = std::nullopt,
   const std::optional<SSORPrecond> &ssor = std::nullopt,
   bool use_simd = true) {
    const int n = A.n;
    dvec x = x0.value_or(dvec::Zero(n));

    dvec r(n), z(n), p(n), Ap(n);

    // r = b - A x
    A.apply(x, Ap);
    r = b - Ap;

    // z = M^{-1} r
    if (ssor) {
        z.resizeLike(r);
        ssor->apply(r, z);
    } else if (JacobiMinvDiag) {
        z = r.cwiseProduct(*JacobiMinvDiag);
    } else {
        z = r;
    }

    p = z;

    const double b2 = b.squaredNorm();
    const double rel_stop2 = std::max(tol * tol * b2, 1e-32);

    CGInfo info{};
    info.iters = 0;
    info.final_residual = r.norm();

    double rz = maybe_simd_dot(r.data(), z.data(), n, use_simd);
    double prev_r2 = r.squaredNorm();
    int stagnation_count = 0;

    // Early exit
    if ((ssor || JacobiMinvDiag) ? (rz <= rel_stop2) : (prev_r2 <= rel_stop2)) {
        info.converged = true;
        return {x, info};
    }

    for (int k = 1; k <= maxit; ++k) {
        // Ap = A p
        A.apply(p, Ap);

        double pAp = maybe_simd_dot(p.data(), Ap.data(), n, use_simd);
        pAp = std::max(pAp, 1e-300);
        const double alpha = rz / pAp;

#if defined(__AVX2__)
        if (use_simd && n >= 4) {
            simd_axpy(alpha, p.data(), x.data(), n);   // x += α p
            simd_axpy(-alpha, Ap.data(), r.data(), n); // r -= α Ap
        } else {
            x.noalias() += alpha * p;
            r.noalias() -= alpha * Ap;
        }
#else
        x.noalias() += alpha * p;
        r.noalias() -= alpha * Ap;
#endif

        const double r2 = r.squaredNorm();
        info.final_residual = std::sqrt(r2);

        // Recompute z only if needed for test / next step
        if (ssor) {
            ssor->apply(r, z);
            const double rMr = maybe_simd_dot(r.data(), z.data(), n, use_simd);
            if (rMr <= rel_stop2) {
                info.converged = true;
                info.iters = k;
                return {x, info};
            }
        } else if (JacobiMinvDiag) {
            z = r.cwiseProduct(*JacobiMinvDiag);
            const double rMr = maybe_simd_dot(r.data(), z.data(), n, use_simd);
            if (rMr <= rel_stop2) {
                info.converged = true;
                info.iters = k;
                return {x, info};
            }
        } else {
            if (r2 <= rel_stop2) {
                info.converged = true;
                info.iters = k;
                return {x, info};
            }
            z = r; // no preconditioner
        }

        const double rz_new = maybe_simd_dot(r.data(), z.data(), n, use_simd);
        const double beta = rz_new / std::max(rz, 1e-300);
        rz = rz_new;

        // p = z + β p
        p.noalias() = z + beta * p;

        // Mild stagnation guard
        if (k > 5 && r2 >= 0.9604 * prev_r2) { // 0.98^2
            if (++stagnation_count > 5) {
                p = z; // restart
                stagnation_count = 0;
            }
        } else {
            stagnation_count = 0;
        }

        prev_r2 = r2;
        info.iters = k;
    }

    return {x, info};
}

// K = W + δI + γ GᵀG  (values only; sparsity same as W + GᵀG)
inline spmat build_augmented_system_inplace(const spmat &W, const spmat &G,
                                            double delta, double gamma) {
    spmat K = W;
    K.makeCompressed();

    if (delta != 0.0) {
        // add to diagonal without building I
        for (int i = 0; i < K.rows(); ++i)
            K.coeffRef(i, i) += delta;
    }

    if (gamma != 0.0) {
        spmat GtG = (G.transpose() * G).eval();
        GtG.makeCompressed();
        K += (gamma * GtG).pruned();
    }
    K.prune(1e-300);
    K.makeCompressed();
    return K;
}

inline double compute_gamma_heuristic(const spmat &W, const spmat &G,
                                      double delta) {
    const double Wn = rowsum_inf_norm(W) + delta;
    const double Gn = rowsum_inf_norm(G);
    return std::max(1.0, Wn / std::max(1.0, Gn * Gn));
}

inline dvec schur_diag_hat(const spmat &G, const dvec &d) {
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

// ------------------------------ Small Dense Solvers -------------------
template <int N> struct SmallDenseSolver {
    static dvec solve(const Eigen::Matrix<double, N, N> &A,
                      const Eigen::Matrix<double, N, 1> &b) {
        if constexpr (N <= 4) {
            return A.llt().solve(b); // Very fast for tiny systems
        } else if constexpr (N <= 12) {
            return A.partialPivLu().solve(b);
        } else {
            return A.llt().solve(b); // Assume SPD for larger small systems
        }
    }
};

// Specialization for 2x2
template <> struct SmallDenseSolver<2> {
    static dvec solve(const Eigen::Matrix2d &A, const Eigen::Vector2d &b) {
        const double det = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
        if (std::abs(det) < 1e-16) {
            return A.llt().solve(b); // Fallback
        }

        dvec x(2);
        x(0) = (A(1, 1) * b(0) - A(0, 1) * b(1)) / det;
        x(1) = (A(0, 0) * b(1) - A(1, 0) * b(0)) / det;
        return x;
    }
};

// Specialization for 3x3
template <> struct SmallDenseSolver<3> {
    static dvec solve(const Eigen::Matrix3d &A, const Eigen::Vector3d &b) {
        const double det = A(0, 0) * (A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1)) -
                           A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0)) +
                           A(0, 2) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0));

        if (std::abs(det) < 1e-14) {
            return A.llt().solve(b); // Fallback
        }

        dvec x(3);
        x(0) = (b(0) * (A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1)) -
                A(0, 1) * (b(1) * A(2, 2) - A(1, 2) * b(2)) +
                A(0, 2) * (b(1) * A(2, 1) - A(1, 1) * b(2))) /
               det;
        x(1) = (A(0, 0) * (b(1) * A(2, 2) - A(1, 2) * b(2)) -
                b(0) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0)) +
                A(0, 2) * (A(1, 0) * b(2) - b(1) * A(2, 0))) /
               det;
        x(2) = (A(0, 0) * (A(1, 1) * b(2) - b(1) * A(2, 1)) -
                A(0, 1) * (A(1, 0) * b(2) - b(1) * A(2, 0)) +
                b(0) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0))) /
               det;
        return x;
    }
};

} // namespace kkt