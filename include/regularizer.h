#pragma once
// regularizer_advanced.hpp
// C++23, Eigen, header-only. No external AMG deps.
// Provides: RegInfo, Regularizer (dense/sparse), Mini preconditioners and
// caching.

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <Eigen/src/IterativeLinearSolvers/IncompleteLUT.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace regx {

using dvec = Eigen::VectorXd;
using dmat = Eigen::MatrixXd;
using spmat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

struct RegInfo {
    std::string mode{"TIKHONOV"};
    double sigma{1e-8};
    double cond_before{std::numeric_limits<double>::infinity()};
    double cond_after{std::numeric_limits<double>::infinity()};
    double min_eig_before{0.0};
    double min_eig_after{0.0};
    int rank_def{0};
    std::tuple<int, int, int> inertia_before{0, 0, 0};
    std::tuple<int, int, int> inertia_after{0, 0, 0};
    int nnz_before{0};
    int nnz_after{0};
    std::string precond_type{"none"};
    double precond_setup_time{0.0};
    int eigensolve_iterations{0};
    bool eigensolve_converged{true};
};

// -------------------- Config --------------------
struct RegularizerConfig {
    // Mode
    std::string reg_mode{
        "AUTO"}; // AUTO|EIGEN_MOD|INERTIA_FIX|SPECTRAL|TIKHONOV
    double reg_sigma{1e-8}, reg_sigma_min{1e-12}, reg_sigma_max{1e6};
    double reg_target_cond{1e12};
    double reg_min_eig_thresh{1e-8};

    // Adapt
    double reg_adapt_factor{2.0};
    double iter_tol{1e-6};
    int max_iter{500};
    int k_eigs{16};

    // Permutation + scaling
    bool reg_use_rcm{false};
    bool reg_use_ruiz{false};
    int reg_ruiz_iters{3};

    // Preconditioning
    bool use_preconditioning{false};
    std::string precond_type{
        "auto"};               // auto|none|jacobi|bjacobi|ssor|ilu|shift_invert
                               // (shift_invert becomes sparseLU)
    double ilu_drop_tol{1e-3}; // (Informative; Eigen::IncompleteLUT uses
                               // "fillfactor"/"droptol" differently)
    double ilu_fill_factor{10.0};

    int precond_small_n{350};
    double precond_cond_gate{5e3};
    double precond_density_gate{0.02};
    int precond_fail_backoff{2};

    int bjacobi_block_size{64};
    double ssor_omega{1.0};

    // TR-aware
    bool reg_tr_aware{true};
    double reg_tr_c{1e-2};

    // Shift-invert behavior (sparseLU factorization of H - sigma I)
    std::string shift_invert_mode{"buckling"}; // nominal (kept for parity)

    // In RegularizerConfig (add these fields)
    double amg_theta{0.08};       // strength threshold
    double amg_jacobi_omega{0.7}; // Jacobi smoother weight
    int amg_presmooth{1};
    int amg_postsmooth{1};
    int amg_max_levels{10};
    int amg_min_coarse{40}; // stop when n <= this
};

// ---------- Minimal Smoothed-Aggregation AMG (SPD, educational) ----------
struct MiniAMG {
    struct Level {
        spmat A; // system on this level (CSR-like ops via InnerIterator)
        dvec D;  // diagonal of A (positive for SPD)
        spmat P; // prolongator to finer level (absent on coarsest)
        spmat R; // restriction (P^T for SPD)
        bool hasP{false};
    };

    std::vector<Level> L;
    double jw{0.7};
    int presmooth{1}, postsmooth{1};

    MiniAMG(const spmat &Ain, double theta = 0.08, double jacobi_omega = 0.7,
            int pres = 1, int posts = 1, int max_levels = 10,
            int min_coarse = 40)
        : jw(jacobi_omega), presmooth(pres), postsmooth(posts) {
        spmat A = Ain;
        A.makeCompressed();

        for (int lev = 0; lev < max_levels; ++lev) {
            const int n = A.rows();
            Level lvl;
            lvl.A = A;
            lvl.D = A.diagonal();
            for (int i = 0; i < n; ++i)
                if (std::abs(lvl.D(i)) < 1e-30)
                    lvl.D(i) = 1.0;
            L.push_back(std::move(lvl));

            if (n <= min_coarse || lev == max_levels - 1)
                break;

            // Strength graph: |a_ij| >= theta * sqrt(a_ii a_jj), off-diagonal
            spmat S = strength_graph(A, theta);

            // Greedy aggregation
            Eigen::VectorXi agg = greedy_aggregate(S); // size n, 0..Nc-1
            const int Nc = agg.size() > 0 ? (agg.maxCoeff() + 1) : 0;
            if (Nc <= 0 || Nc >= n)
                break; // failed coarsening

            // Tentative prolongator T: (n x Nc) with one 1 per row
            spmat T = tentative_prolongator(agg, n, Nc);

            // Smoothed prolongator: P = (I - ω D^{-1} A) T
            const dvec Dinv = L.back().D.cwiseInverse();
            // Compute A*T (dense cols; robust & simple)
            dmat AT = dmat::Zero(n, Nc);
            for (int j = 0; j < Nc; ++j) {
                // column j of T is indicator of aggregate j
                for (int r = 0; r < n; ++r) {
                    if (T.coeff(r, j) != 0.0) { // r belongs to aggregate j
                        for (spmat::InnerIterator it(A, r); it; ++it) {
                            AT(it.row(), j) += it.value();
                        }
                    }
                }
            }
            // P = T - ω * diag(Dinv) * (A*T)
            dmat Pden = dmat::Zero(n, Nc);
            for (int j = 0; j < Nc; ++j) {
                for (int i = 0; i < n; ++i) {
                    double Tij = T.coeff(i, j);
                    double corr = jw * Dinv(i) * AT(i, j);
                    Pden(i, j) = Tij - corr;
                }
            }
            spmat P = Pden.sparseView();
            P.makeCompressed();

            // Coarse operator: Ac = P^T A P
            spmat Ac = (P.transpose() * A * P).pruned();
            Ac.makeCompressed();

            // Store P/R on this level and continue
            L.back().P = P;
            L.back().R = spmat(P.transpose());
            L.back().hasP = true;

            A = Ac; // next level
        }
    }

    // Apply one V-cycle to approximately solve A0 x = rhs (zero initial guess)
    dvec solve(const dvec &rhs) const {
        dvec x = dvec::Zero(rhs.size());
        return vcycle(0, x, rhs);
    }

private:
    static spmat strength_graph(const spmat &A, double theta) {
        const int n = A.rows();
        const dvec D = A.diagonal();
        std::vector<Eigen::Triplet<double, int>> trips;
        trips.reserve(A.nonZeros());
        for (int j = 0; j < A.outerSize(); ++j) {
            for (spmat::InnerIterator it(A, j); it; ++it) {
                const int i = it.row();
                if (i == j)
                    continue;
                const double denom =
                    std::sqrt(std::max(D(i), 1e-32) * std::max(D(j), 1e-32));
                if (denom <= 0.0)
                    continue;
                const double s = std::abs(it.value()) / denom;
                if (s >= theta)
                    trips.emplace_back(i, j, 1.0);
            }
        }
        spmat S(n, n);
        S.setFromTriplets(trips.begin(), trips.end());
        // symmetrize
        S = (S + spmat(S.transpose())).pruned();
        S.makeCompressed();
        return S;
    }

    static Eigen::VectorXi greedy_aggregate(const spmat &S) {
        const int n = S.rows();
        Eigen::VectorXi agg = Eigen::VectorXi::Constant(n, -1);
        std::vector<char> vis(n, 0);
        int cur = 0;

        // Materialize transpose once (needed to iterate row-neighbors with
        // ColMajor storage)
        spmat ST = S.transpose();
        ST.makeCompressed();

        for (int i = 0; i < n; ++i) {
            if (vis[i])
                continue;

            // seed
            agg(i) = cur;
            vis[i] = 1;

            // attach strong neighbors in the same column (neighbors j with
            // S(i,j) ≠ 0)
            for (spmat::InnerIterator it(S, i); it; ++it) {
                const int r = it.row(); // row index in column i
                if (!vis[r]) {
                    agg(r) = cur;
                    vis[r] = 1;
                }
            }

            // attach strong neighbors in the same row (neighbors j with S(j,i)
            // ≠ 0)
            for (spmat::InnerIterator it2(ST, i); it2; ++it2) {
                const int r =
                    it2.row(); // row index in column i of S^T ⇒ neighbor index
                if (!vis[r]) {
                    agg(r) = cur;
                    vis[r] = 1;
                }
            }

            ++cur;
        }

        // any isolated leftovers → singleton aggregates
        for (int i = 0; i < n; ++i) {
            if (agg(i) < 0) {
                agg(i) = cur;
                ++cur;
            }
        }

        return agg;
    }

    static spmat tentative_prolongator(const Eigen::VectorXi &agg, int n,
                                       int Nc) {
        std::vector<Eigen::Triplet<double, int>> trips;
        trips.reserve(n);
        for (int i = 0; i < n; ++i) {
            trips.emplace_back(i, agg(i), 1.0);
        }
        spmat T(n, Nc);
        T.setFromTriplets(trips.begin(), trips.end());
        T.makeCompressed();
        return T;
    }

    static dvec jacobi(const spmat &A, const dvec &D, dvec x, const dvec &b,
                       double w, int iters) {
        const dvec Dinv = D.cwiseInverse();
        for (int k = 0; k < std::max(iters, 0); ++k) {
            dvec r = b - A * x;
            x += w * (Dinv.cwiseProduct(r));
        }
        return x;
    }

    dvec vcycle(int k, dvec x, const dvec &b) const {
        const Level &lev = L[k];
        // pre-smooth
        x = jacobi(lev.A, lev.D, x, b, jw, presmooth);

        if (!lev.hasP) {
            // coarsest: dense solve fallback
            Eigen::SimplicialLDLT<spmat> ldlt;
            ldlt.compute(lev.A);
            if (ldlt.info() == Eigen::Success) {
                return ldlt.solve(b);
            }
            // fallback: extra smoothing
            return jacobi(lev.A, lev.D, x, b, jw, 12);
        }

        // residual
        dvec r = b - lev.A * x;
        // restrict
        const dvec bc = lev.R * r;
        // coarse solve
        dvec xc = dvec::Zero(bc.size());
        xc = vcycle(k + 1, xc, bc);
        // prolongate & correct
        x += lev.P * xc;
        // post-smooth
        x = jacobi(lev.A, lev.D, x, b, jw, postsmooth);
        return x;
    }
};

// -------------------- hashing for sparsity patterns --------------------
namespace detail {
inline std::string pattern_signature(const spmat &A) {
    // compact 64-bit xor hash over index arrays + size
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t v) {
        h ^= v;
        h *= 1099511628211ull;
    };
    mix(static_cast<uint64_t>(A.rows()));
    mix(static_cast<uint64_t>(A.cols()));
    // outerIndexPtr size = cols+1 (ColMajor)
    for (int i = 0; i < A.outerSize() + 1; ++i)
        mix(static_cast<uint64_t>(A.outerIndexPtr()[i]));
    for (int k = 0; k < A.nonZeros(); ++k)
        mix(static_cast<uint64_t>(A.innerIndexPtr()[k]));
    return std::to_string(h);
}

inline bool is_sparse(const auto &) { return false; }
inline bool is_sparse(const spmat &) { return true; }

inline int nnz(const dmat &A) { return static_cast<int>(A.size()); }
inline int nnz(const spmat &A) { return A.nonZeros(); }

inline dmat sym(const dmat &A) { return 0.5 * (A + A.transpose()); }
inline spmat sym(const spmat &A) {
    spmat S = A;
    spmat AT = spmat(A.transpose());
    // average structurally (A + A^T)/2
    spmat B = S;
    B = (S + AT) * 0.5;
    B.makeCompressed();
    return B;
}

inline dmat tikhonov(const dmat &H, double s) {
    dmat I = dmat::Identity(H.rows(), H.cols());
    return H + s * I;
}
inline spmat tikhonov(const spmat &H, double s) {
    spmat I(H.rows(), H.cols());
    I.setIdentity();
    return H + s * I;
}

inline std::pair<double, double> eig_extents_dense(const dmat &H) {
    Eigen::SelfAdjointEigenSolver<dmat> es(H);
    const auto &w = es.eigenvalues();
    return {static_cast<double>(w(0)), static_cast<double>(w(w.size() - 1))};
}

inline double cond_from_extents(double lmin, double lmax) {
    if (lmax <= 0.0)
        return std::numeric_limits<double>::infinity();
    double denom = std::max(std::abs(lmin), 1e-16);
    return lmax / denom;
}

// Symmetric Ruiz equilibration: Hs = D H D
inline std::pair<spmat, dvec> ruiz_equilibrate(spmat H, int iters) {
    const int n = H.rows();
    dvec d = dvec::Ones(n);
    for (int it = 0; it < std::max(1, iters); ++it) {
        dvec r = dvec::Zero(n);
        // compute row 2-norms of H (approx via |.|^2 accumulation)
        for (int j = 0; j < H.outerSize(); ++j) {
            for (spmat::InnerIterator it(H, j); it; ++it) {
                const int i = it.row();
                const double v = it.value();
                r(i) += v * v;
            }
        }
        for (int i = 0; i < n; ++i)
            r(i) = std::sqrt(std::max(r(i), 1e-16));
        dvec s = r.array().inverse().sqrt(); // 1/sqrt(r)
        // H <- D H D
        spmat D(n, n);
        D.reserve(Eigen::VectorXi::Constant(n, 1));
        for (int i = 0; i < n; ++i)
            D.insert(i, i) = s(i);
        H = D * H * D;
        d = d.cwiseProduct(s);
    }
    H.makeCompressed();
    return {std::move(H), std::move(d)};
}

inline spmat apply_diag_scaling(const spmat &H, const dvec &d) {
    const int n = H.rows();
    spmat D(n, n);
    D.reserve(Eigen::VectorXi::Constant(n, 1));
    for (int i = 0; i < n; ++i)
        D.insert(i, i) = d(i);
    spmat out = D * H * D;
    out.makeCompressed();
    return out;
}

// Reverse Cuthill–McKee using Eigen's AMD is not available;
// we provide a small RCM via Eigen::Matrix market? Not included.
// We fall back to identity if not available.
inline std::optional<Eigen::VectorXi> rcm_permutation(const spmat &H) {
    // Placeholder: return std::nullopt to avoid risky custom code here.
    return std::nullopt;
}

// ------------ tiny Lanczos (symmetric) for extremal eigenvalues ------------
struct LanczosResult {
    double value{0.0};
    int iters{0};
    bool ok{false};
};

// Compute largest magnitude (LA) or smallest algebraic (SA) approximately.
// For SA we run Lanczos on (H + shift*I)^{-1} if requested via functor OPinv,
// else do inverse iteration fallback (cheap single linear solve per step via
// SparseLU).
inline LanczosResult lanczos_extreme(
    const spmat &H,
    std::string_view which, // "LA" or "SA"
    int maxit, double tol,
    std::optional<std::function<dvec(const dvec &)>> OPinv = std::nullopt,
    std::optional<double> sigma = std::nullopt) {
    const int n = H.rows();
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    dvec v = dvec::NullaryExpr(n, [&](auto) { return U(rng); });
    v.normalize();
    dvec w = dvec::Zero(n), v_old = dvec::Zero(n);

    double alpha = 0.0, beta = 0.0, beta_old = 0.0, theta = 0.0;
    double theta_old = 0.0;

    // Optional inverse operator
    std::function<dvec(const dvec &)> Apply = [&](const dvec &x) -> dvec {
        if (OPinv)
            return (*OPinv)(x);
        return H * x;
    };

    // If SA and no OPinv given, build SparseLU on (H - sigma I)
    std::unique_ptr<Eigen::SparseLU<spmat>> lu;
    if (which == "SA" && !OPinv) {
        spmat A = H;
        double s = sigma.value_or(0.0);
        if (s != 0.0) {
            spmat I(n, n);
            I.setIdentity();
            A = H - s * I;
        }
        lu = std::make_unique<Eigen::SparseLU<spmat>>();
        lu->analyzePattern(A);
        lu->factorize(A);
        if (lu->info() != Eigen::Success) {
            // fallback to plain power on H for smallest (weak)
            lu.reset();
        } else {
            Apply = [&](const dvec &x) -> dvec {
                dvec y = x;
                // solve (H - sI) y = x  -> y
                y = lu->solve(x);
                return y;
            };
        }
    }

    for (int k = 0; k < maxit; ++k) {
        w = Apply(v);
        if (which == "SA" && !lu && !OPinv) {
            // cheap inverse iteration: one gradient step
            // this is weak, but prevents total failure
            w = (H * v);
        }
        alpha = v.dot(w);
        w = w - alpha * v - beta * v_old;
        beta_old = beta;
        beta = w.norm();
        if (beta < 1e-14) {
            theta = alpha; // converged
            return {theta, k + 1, true};
        }
        v_old = v;
        v = w / beta;

        // Rayleigh quotient approximation oscillates; use alpha as Ritz on the
        // fly
        theta_old = theta;
        theta = alpha;
        if (k > 10 && std::abs(theta - theta_old) <
                          tol * std::max(1.0, std::abs(theta_old))) {
            return {theta, k + 1, true};
        }
    }
    return {theta, maxit, false};
}

// Low rank PSD bump: H + V diag(delta) V^T, for small (dense) V
inline dmat low_rank_bump(const dmat &H, const dmat &V, const dvec &delta) {
    return detail::sym(H + V * delta.asDiagonal() * V.transpose());
}
inline spmat low_rank_bump(const spmat &H, const dmat &V, const dvec &delta) {
    dmat upd = V * delta.asDiagonal() * V.transpose();
    spmat U = upd.sparseView();
    dmat Hd = dmat(H);
    dmat Ud = dmat(U);
    dmat sym_dense = detail::sym(Hd + Ud);
    spmat out = sym_dense.sparseView();
    out.makeCompressed();
    return out;
}

inline std::pair<dmat, dvec> smallest_k_dense(const dmat &H, int k) {
    Eigen::SelfAdjointEigenSolver<dmat> es(H);
    const auto &w = es.eigenvalues();
    const dmat &V = es.eigenvectors();
    k = std::min(k, static_cast<int>(w.size()));
    return {V.leftCols(k), w.head(k)};
}

} // namespace detail

// -------------------- Preconditioners --------------------
class Preconditioner {
public:
    using Matvec = std::function<dvec(const dvec &)>;
    Preconditioner() = default;
    explicit Preconditioner(Matvec mv, int n) : mv_(std::move(mv)), n_(n) {}
    dvec operator()(const dvec &x) const { return mv_ ? mv_(x) : x; }
    explicit operator bool() const { return static_cast<bool>(mv_); }
    int rows() const { return n_; }

private:
    Matvec mv_{};
    int n_{0};
};

class PrecondFactory {
public:
    static Preconditioner none(int n) {
        return Preconditioner([](const dvec &x) { return x; }, n);
    }

    // In class PrecondFactory:
    static Preconditioner amg(const spmat &H, double theta = 0.08,
                              double jw = 0.7, int pres = 1, int posts = 1,
                              int max_levels = 10, int min_coarse = 40) {
        using AMG = MiniAMG; // <-- if you defined it as regx::MiniAMG
        // using AMG = detail::MiniAMG; // <-- use this instead if the class is
        // in regx::detail

        auto mg = std::make_shared<AMG>(H, theta, jw, pres, posts, max_levels,
                                        min_coarse);
        const int n = H.rows();
        return Preconditioner([mg](const dvec &x) { return mg->solve(x); }, n);
    }

    static Preconditioner jacobi(const spmat &H) {
        const int n = H.rows();
        dvec d(n);
        for (int i = 0; i < n; ++i) {
            double diag = H.coeff(i, i);
            if (std::abs(diag) < 1e-16)
                diag = 1.0;
            d(i) = 1.0 / diag;
        }
        return Preconditioner([d](const dvec &x) { return d.cwiseProduct(x); },
                              n);
    }
    static Preconditioner bjacobi(const spmat &H, int block) {
        const int n = H.rows();
        struct Block {
            int s, e;
            std::variant<dmat, dvec> inv;
        };
        std::vector<Block> blocks;
        for (int s = 0; s < n; s += block) {
            int e = std::min(s + block, n);
            // extract dense block
            dmat B = dmat::Zero(e - s, e - s);
            for (int j = s; j < e; ++j) {
                for (spmat::InnerIterator it(H, j); it; ++it) {
                    int i = it.row();
                    if (i >= s && i < e)
                        B(i - s, j - s) = it.value();
                }
            }
            Eigen::FullPivLU<dmat> lu(B);
            if (lu.isInvertible()) {
                dmat Binv = lu.inverse();
                blocks.push_back({s, e, std::move(Binv)});
            } else {
                dvec d = B.diagonal();
                for (int i = 0; i < d.size(); ++i)
                    if (std::abs(d(i)) < 1e-16)
                        d(i) = 1.0;
                d = d.cwiseInverse();
                blocks.push_back({s, e, std::move(d)});
            }
        }
        return Preconditioner(
            [blocks](const dvec &x) {
                dvec y = dvec::Zero(x.size());
                for (auto &blk : blocks) {
                    const int s = blk.s, e = blk.e;
                    dvec xi = x.segment(s, e - s);
                    if (std::holds_alternative<dmat>(blk.inv)) {
                        const dmat &Binv = std::get<dmat>(blk.inv);
                        y.segment(s, e - s) = Binv * xi;
                    } else {
                        const dvec &d = std::get<dvec>(blk.inv);
                        y.segment(s, e - s) = d.cwiseProduct(xi);
                    }
                }
                return y;
            },
            n);
    }
    static Preconditioner ssor(const spmat &H, double omega) {
        const int n = H.rows();
        spmat Hc = H;
        Hc.makeCompressed();
        dvec D(n);
        for (int i = 0; i < n; ++i) {
            double diag = Hc.coeff(i, i);
            if (std::abs(diag) < 1e-16)
                diag = 1.0;
            D(i) = diag;
        }
        return Preconditioner(
            [Hc, D, omega, n](const dvec &x) {
                // forward
                dvec y = dvec::Zero(n);
                for (int i = 0; i < n; ++i) {
                    double s = x(i);
                    for (spmat::InnerIterator it(Hc, i); it; ++it) {
                        int j = it.row();
                        if (j < i)
                            s -= omega * it.value() * y(j);
                    }
                    y(i) = s / D(i);
                }
                // backward
                dvec z = dvec::Zero(n);
                dvec yprime = y.cwiseQuotient(D);
                for (int i = n - 1; i >= 0; --i) {
                    double s = yprime(i);
                    for (spmat::InnerIterator it(Hc, i); it; ++it) {
                        int j = it.row();
                        if (j > i)
                            s -= omega * it.value() * z(j);
                    }
                    z(i) = s / D(i);
                }
                return z;
            },
            n);
    }
    // REPLACE your ilu() with this version
    static Preconditioner ilu(const spmat &H) {
        const int n = H.rows();
        auto ilu = std::make_shared<Eigen::IncompleteLUT<double>>();
        ilu->setDroptol(1e-4);
        ilu->setFillfactor(10);
        ilu->compute(H);
        return Preconditioner([ilu](const dvec &x) { return ilu->solve(x); },
                              n);
    }

    static Preconditioner shift_invert(const spmat &H, double sigma) {
        const int n = H.rows();
        spmat A = H;
        if (sigma != 0.0) {
            spmat I(n, n);
            I.setIdentity();
            A = H - sigma * I;
        }
        auto lu = std::make_shared<Eigen::SparseLU<spmat>>();
        lu->analyzePattern(A);
        lu->factorize(A);
        return Preconditioner([lu](const dvec &x) { return lu->solve(x); }, n);
    }
};

// -------------------- Regularizer --------------------
class Regularizer {
public:
    explicit Regularizer(RegularizerConfig cfg = {}) : cfg_(std::move(cfg)) {}

    // Wrap preconditioner into original coordinates after perm/scaling:
    // y = P^T D^{-1/2}  M'( D^{1/2} P x )
    static Preconditioner
    wrap_preconditioner(const Preconditioner &Minner,
                        const std::optional<Eigen::VectorXi> &perm,
                        const std::optional<dvec> &dscale, int n) {
        if (!perm && !dscale)
            return Minner;
        Eigen::VectorXi ip;
        if (perm) {
            ip.resize(perm->size());
            for (int i = 0; i < perm->size(); ++i)
                ip((*perm)(i)) = i;
        }
        dvec ds, ids;
        if (dscale) {
            ds = dscale->array().sqrt().matrix();
            ids = ds.array().inverse().matrix();
        }
        return Preconditioner(
            [Minner, perm, ip, ds, ids, dscale](const dvec &x) {
                dvec v = x;
                if (perm) {
                    dvec xp(v.size());
                    for (int i = 0; i < perm->size(); ++i)
                        xp(i) = v((*perm)(i));
                    v.swap(xp);
                }
                if (dscale)
                    v = ds.cwiseProduct(v);
                v = Minner(v);
                if (dscale)
                    v = ids.cwiseProduct(v);
                if (perm) {
                    dvec xo(v.size());
                    for (int i = 0; i < perm->size(); ++i)
                        xo(ip(i)) = v(i);
                    v.swap(xo);
                }
                return v;
            },
            n);
    }

    // -------- public API: SPARSE --------
    std::pair<spmat, RegInfo>
    regularize(const spmat &H_in, int iteration = 0,
               std::optional<double> model_quality = std::nullopt,
               int constraint_count = 0,
               std::optional<double> grad_norm = std::nullopt,
               std::optional<double> tr_radius = std::nullopt,
               bool precond_only = false,
               std::optional<Preconditioner> *out_precond = nullptr) {
        RegInfo info;
        spmat H = detail::sym(H_in);
        const int n = H.rows();
        info.nnz_before = H.nonZeros();

        // Optional RCM
        std::optional<Eigen::VectorXi> perm;
        // if (cfg_.reg_use_rcm && n >= 200) {
        //     perm = detail::rcm_permutation(H);
        //     if (perm) {
        //         H = permute(H, *perm);
        //     }
        // }

        // Optional Ruiz (symmetric)
        std::optional<dvec> dscale;
        if (cfg_.reg_use_ruiz) {
            auto [Hs, d] = detail::ruiz_equilibrate(H, cfg_.reg_ruiz_iters);
            H = std::move(Hs);
            dscale = d;
        }

        // analyze
        auto analysis = analyze_sparse_(H);
        info.min_eig_before = analysis.min_eig;
        info.cond_before = analysis.cond;
        info.inertia_before = analysis.inertia;
        info.precond_type = analysis.precond;

        // precond-only path
        if (precond_only) {
            Preconditioner Minner = build_precond_(H, analysis.precond);
            Preconditioner M = wrap_preconditioner(Minner, perm, dscale, n);
            if (out_precond)
                *out_precond = M;

            auto post = post_analyze_sparse_(H);
            info.mode = "PRECOND_ONLY";
            info.sigma = sigma_;
            info.min_eig_after = post.min_eig;
            info.cond_after = post.cond;
            info.rank_def = post.rank_def;
            info.inertia_after = post.inertia;
            info.nnz_after = H.nonZeros();

            // Undo transforms for returned H
            spmat H_out = H;
            if (dscale) {
                dvec invd = dscale->array().inverse().matrix();
                H_out = detail::apply_diag_scaling(H_out, invd);
            }
            if (perm) {
                H_out = inverse_permute(H_out, *perm);
            }

            return {H_out, info};
        }

        adapt_sigma_(analysis, iteration, grad_norm, tr_radius);

        // choose mode
        std::string mode = cfg_.reg_mode;
        if (mode == "AUTO") {
            if (std::get<1>(analysis.inertia) > 0 && constraint_count > 0)
                mode = "INERTIA_FIX";
            else if ((std::abs(analysis.min_eig) < 1e-10 &&
                      analysis.cond > cfg_.reg_target_cond) ||
                     analysis.cond > 100.0 * cfg_.reg_target_cond)
                mode = "SPECTRAL";
            else if (analysis.min_eig < -1e-10 ||
                     analysis.cond > cfg_.reg_target_cond)
                mode = "EIGEN_MOD";
            else
                mode = "TIKHONOV";
        }

        if (mode == "EIGEN_MOD") {
            H = eigen_bump_sparse_(H, analysis);
        } else if (mode == "INERTIA_FIX") {
            H = inertia_fix_sparse_(H, analysis, constraint_count);
        } else if (mode == "SPECTRAL") {
            H = spectral_floor_sparse_(H, analysis);
        } else {
            H = detail::tikhonov(H, choose_floor_(analysis));
        }
        info.mode = mode;
        info.sigma = sigma_;

        // Undo transforms
        if (dscale) {
            dvec invd = dscale->array().inverse().matrix();
            H = detail::apply_diag_scaling(H, invd);
        }
        if (perm) {
            H = inverse_permute(H, *perm);
        }

        // auto post = post_analyze_sparse_(H);
        // info.min_eig_after = post.min_eig;
        // info.cond_after = post.cond;
        // info.rank_def = post.rank_def;
        // info.inertia_after = post.inertia;
        // info.nnz_after = H.nonZeros();

        return {H, info};
    }

    // -------- public API: DENSE (compact, for n<=800 typical) --------
    std::pair<dmat, RegInfo>
    regularize(const dmat &H_in, int iteration = 0,
               std::optional<double> /*model_quality*/ = std::nullopt,
               int constraint_count = 0,
               std::optional<double> grad_norm = std::nullopt,
               std::optional<double> tr_radius = std::nullopt) {
        RegInfo info;
        dmat H = detail::sym(H_in);
        info.nnz_before = H.size();

        auto [lmin, lmax] = detail::eig_extents_dense(H);
        info.min_eig_before = lmin;
        info.cond_before = detail::cond_from_extents(lmin, lmax);
        info.inertia_before = {0, 0, 0};

        adapt_sigma_({lmin, lmax, info.cond_before, {0, 0, 0}, "none"},
                     iteration, grad_norm, tr_radius);

        std::string mode = cfg_.reg_mode;
        if (mode == "AUTO") {
            if (lmin < -1e-10 || info.cond_before > cfg_.reg_target_cond)
                mode = "EIGEN_MOD";
            else
                mode = "TIKHONOV";
        }

        if (mode == "EIGEN_MOD") {
            int k = std::min(cfg_.k_eigs,
                             std::max(1, static_cast<int>(H.rows() / 20)));
            auto [V, w] = detail::smallest_k_dense(H, k);
            const double floor = std::max(sigma_, cfg_.reg_min_eig_thresh);
            Eigen::ArrayXd mask = (w.array() < floor).cast<double>();
            if (mask.any()) {
                dvec delta = (floor - w.array()).max(0.0).matrix();
                H = detail::low_rank_bump(H, V, delta.head(V.cols()));
            }
        } else {
            H = detail::tikhonov(
                H, choose_floor_(
                       {lmin, lmax, info.cond_before, {0, 0, 0}, "none"}));
        }
        info.mode = mode;
        info.sigma = sigma_;
        auto [plmin, plmax] = detail::eig_extents_dense(H);
        info.min_eig_after = plmin;
        info.cond_after = detail::cond_from_extents(plmin, plmax);
        info.nnz_after = H.size();
        return {H, info};
    }

private:
    struct Analysis {
        double min_eig{0.0}, max_eig{0.0},
            cond{std::numeric_limits<double>::infinity()};
        std::tuple<int, int, int> inertia{0, 0, 0};
        std::string precond{"none"};
        int rank_def{0}; // <--- add this
    };

    RegularizerConfig cfg_;
    double sigma_{1e-8};
    int precond_fail_streak_{0};

    // -------- helpers: permutation --------
    static spmat permute(const spmat &H, const Eigen::VectorXi &p) {
        using Perm =
            Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int>;
        Perm P(p);
        spmat out = P * H * P.transpose();
        out.makeCompressed();
        return out;
    }

    static spmat inverse_permute(const spmat &H, const Eigen::VectorXi &p) {
        using Perm =
            Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int>;
        Eigen::VectorXi ip(p.size());
        for (int i = 0; i < p.size(); ++i)
            ip(p(i)) = i;
        Perm Pinv(ip);
        spmat out = Pinv * H * Pinv.transpose();
        out.makeCompressed();
        return out;
    }

    // -------- eigen extents (sparse): tiny Lanczos --------
    Analysis analyze_sparse_(const spmat &H) {
        Analysis A;
        const int n = H.rows();
        const double density =
            static_cast<double>(H.nonZeros()) / std::max(1, n * n);
        // choose precond "auto"
        A.precond = select_precond_(H, {0, 0, 0}, density);
        // lmax via power/lanczos
        auto rLA =
            detail::lanczos_extreme(H, "LA", cfg_.max_iter, cfg_.iter_tol);
        // lmin via inverse iteration or shift-invert
        auto rSA =
            detail::lanczos_extreme(H, "SA", cfg_.max_iter, cfg_.iter_tol);
        // fallback to diagonal if unstable
        if (!rLA.ok || !std::isfinite(rLA.value)) {
            double mx = -1e300;
            for (int i = 0; i < n; ++i)
                mx = std::max(mx, std::abs(H.coeff(i, i)));
            A.max_eig = std::max(mx, 1.0);
        } else
            A.max_eig = rLA.value;

        if (!rSA.ok || !std::isfinite(rSA.value)) {
            double mn = 1e300;
            for (int i = 0; i < n; ++i)
                mn = std::min(mn, H.coeff(i, i));
            A.min_eig = std::isfinite(mn) ? mn : 1.0;
        } else
            A.min_eig = rSA.value;

        A.cond = detail::cond_from_extents(A.min_eig, A.max_eig);
        A.inertia = {0, 0, 0}; // hook here if you wire qdldl later
        A.rank_def = 0;        // placeholder

        return A;
    }
    Analysis post_analyze_sparse_(const spmat &H) { return analyze_sparse_(H); }

    // -------- precond selection --------
    std::string select_precond_(const spmat &H,
                                std::tuple<int, int, int> /*inertia_hint*/,
                                double density) {
        if (!cfg_.use_preconditioning || cfg_.precond_type == "none")
            return "none";
        if (cfg_.precond_type != "auto")
            return cfg_.precond_type;

        const int n = H.rows();
        if (n <= cfg_.precond_small_n)
            return "none";

        // Heuristics:
        // Prefer AMG for big, very sparse, diagonally reasonable matrices
        // (SPD-ish).
        double diag_nz = 0.0, weak_diag = 0.0;
        {
            dvec d = H.diagonal().cwiseAbs();
            diag_nz = (d.array() > 1e-14).cast<double>().mean();
            weak_diag = (d.array() < 1e-8).cast<double>().mean();
        }
        if (n >= 1500 && density < 0.01 && diag_nz > 0.9 && weak_diag < 0.2)
            return "amg";

        if (density < 0.01)
            return "ilu";
        if (density < 0.10)
            return "ilu";

        // bandwidth-ish -> SSOR
        double band_mean = 0.0;
        {
            long long cnt = 0;
            for (int j = 0; j < H.outerSize(); ++j)
                for (spmat::InnerIterator it(H, j); it; ++it) {
                    band_mean += std::abs(it.row() - j);
                    ++cnt;
                }
            band_mean = (cnt > 0) ? band_mean / double(cnt) : 0.0;
        }
        if (band_mean < 0.12 * n)
            return "ssor";
        if (density < 0.35)
            return "bjacobi";
        return "jacobi";
    }

    bool should_use_precond_(const spmat &H, const Analysis &A) {
        const int n = H.rows();
        const double density =
            static_cast<double>(H.nonZeros()) / std::max(1, n * n);
        if (!cfg_.use_preconditioning || cfg_.precond_type == "none")
            return false;
        if (n <= cfg_.precond_small_n)
            return false;
        if (precond_fail_streak_ > 0)
            return false;
        if (A.cond >= cfg_.precond_cond_gate)
            return true;
        return density <= cfg_.precond_density_gate;
    }

    Preconditioner build_precond_(const spmat &H, const std::string &type) {
        try {
            if (type == "none")
                return PrecondFactory::none(H.rows());
            if (type == "jacobi")
                return PrecondFactory::jacobi(H);
            if (type == "bjacobi")
                return PrecondFactory::bjacobi(H, cfg_.bjacobi_block_size);
            if (type == "ssor")
                return PrecondFactory::ssor(H, cfg_.ssor_omega);
            if (type == "ilu")
                return PrecondFactory::ilu(H);
            if (type == "shift_invert")
                return PrecondFactory::shift_invert(H, 0.0);
            // In Regularizer::build_precond_
            if (type == "amg")
                return PrecondFactory::amg(
                    H, cfg_.amg_theta, cfg_.amg_jacobi_omega,
                    cfg_.amg_presmooth, cfg_.amg_postsmooth,
                    cfg_.amg_max_levels, cfg_.amg_min_coarse);

            return PrecondFactory::none(H.rows());
        } catch (...) {
            // backoff on failure
            precond_fail_streak_ =
                std::min(cfg_.precond_fail_backoff, precond_fail_streak_ + 1);
            return PrecondFactory::jacobi(H);
        }
    }

    // -------- mode implementations (sparse) --------
    spmat eigen_bump_sparse_(const spmat &H, const Analysis &A) {
        const int n = H.rows();
        const double floor = std::max(sigma_, cfg_.reg_min_eig_thresh);
        // cheap approach: compute k Ritz vectors using dense on small n
        if (n <= 800) {
            dmat Hd = dmat(H);
            auto [V, w] = detail::smallest_k_dense(
                Hd, std::min(cfg_.k_eigs, std::max(1, n / 20)));
            dvec delta = dvec::Zero(V.cols());
            for (int i = 0; i < V.cols(); ++i) {
                double wi = w(i);
                if (wi < floor)
                    delta(i) = (floor - wi);
            }
            if (delta.lpNorm<Eigen::Infinity>() == 0.0)
                return H;
            return detail::low_rank_bump(H, V, delta);
        }
        // large n: fallback to Tikhonov floor (still robust without Spectra)
        return detail::tikhonov(H, floor);
    }

    spmat inertia_fix_sparse_(const spmat &H, const Analysis &A, int m_eq) {
        const int n = H.rows();
        const int target_pos = std::max(1, n - std::max(0, m_eq));
        // approximate with k small eigs; if not enough positive, bump
        if (n <= 800) {
            dmat Hd = dmat(H);
            auto [V, w] = detail::smallest_k_dense(
                Hd, std::min(cfg_.k_eigs, std::max(2, n / 15)));
            int pos_now = static_cast<int>((w.array() > 1e-12).count());
            if (pos_now >= target_pos)
                return H;
            const double floor = std::max(sigma_, cfg_.reg_min_eig_thresh);
            dvec delta = dvec::Zero(V.cols());
            for (int i = 0; i < V.cols(); ++i) {
                double wi = w(i);
                if (wi < floor)
                    delta(i) = (floor - wi);
            }
            return detail::low_rank_bump(H, V, delta);
        }
        return detail::tikhonov(H, std::max(sigma_, cfg_.reg_min_eig_thresh));
    }

    spmat spectral_floor_sparse_(const spmat &H, const Analysis &A) {
        const int n = H.rows();
        const double floor = std::max(sigma_, cfg_.reg_min_eig_thresh);
        if (n <= 800) {
            dmat Hd = dmat(H);
            Eigen::SelfAdjointEigenSolver<dmat> es(Hd);
            dmat V = es.eigenvectors();
            dvec s = es.eigenvalues();
            for (int i = 0; i < s.size(); ++i)
                s(i) = std::max(s(i), floor);
            dmat S = s.asDiagonal();
            dmat Hreg = detail::sym(V * S * V.transpose());
            return Hreg.sparseView();
        }
        return detail::tikhonov(H, floor);
    }

    // -------- adapt & floors --------
    void adapt_sigma_(const Analysis &A, int iteration,
                      std::optional<double> grad_norm,
                      std::optional<double> tr_radius) {
        sigma_ = std::clamp(sigma_, cfg_.reg_sigma_min, cfg_.reg_sigma_max);
        if (A.cond > cfg_.reg_target_cond) {
            sigma_ =
                std::min(cfg_.reg_sigma_max, sigma_ * cfg_.reg_adapt_factor);
        } else if (A.cond < std::max(1.0, cfg_.reg_target_cond * 1e-2) &&
                   A.min_eig >= -1e-12) {
            sigma_ =
                std::max(cfg_.reg_sigma_min, sigma_ / cfg_.reg_adapt_factor);
        }
        if (cfg_.reg_tr_aware && grad_norm && tr_radius && *tr_radius > 0.0) {
            sigma_ =
                std::max(sigma_, cfg_.reg_tr_c * (*grad_norm) / (*tr_radius));
        }
        if (iteration < 3)
            sigma_ = std::max(sigma_, 1e-8);
    }

    double choose_floor_(const Analysis &A) const {
        double floor = std::max(sigma_, cfg_.reg_min_eig_thresh);
        if (A.cond > cfg_.reg_target_cond && A.max_eig > 0.0) {
            double target_min = A.max_eig / cfg_.reg_target_cond;
            floor = std::max(floor, target_min);
        }
        return floor;
    }
};

} // namespace regx
