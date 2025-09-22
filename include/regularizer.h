#pragma once
// regularizer_.hpp — C++23 regularizer (Spectra-powered)
// Keeps public API. Uses Spectra for extremal eigenvalues and k-smallest eigenpairs.
// Falls back gracefully if Spectra is not present.

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

#ifdef __AVX2__
#include <immintrin.h>
#endif

// ---------------- Spectra detection ----------------
#ifndef AD_HAVE_SPECTRA
#  if defined(__has_include)
#    if __has_include(<Spectra/SymEigsSolver.h>)
#      define AD_HAVE_SPECTRA 1
#    else
#      define AD_HAVE_SPECTRA 0
#    endif
#  else
#    define AD_HAVE_SPECTRA 0
#  endif
#endif

#if AD_HAVE_SPECTRA
  #include <Spectra/SymEigsSolver.h>
  #include <Spectra/SymEigsShiftSolver.h>
  #include <Spectra/MatOp/DenseSymMatProd.h>
  #include <Spectra/MatOp/DenseSymShiftSolve.h>
  #include <Spectra/MatOp/SparseSymMatProd.h>
  #include <Spectra/MatOp/SparseSymShiftSolve.h>
#endif

namespace regx {

using dvec = Eigen::VectorXd;
using dmat = Eigen::MatrixXd;
using spmat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

// ----------------------------- Config / Info -----------------------------
struct RegConfig {
    double sigma{1e-8}, sigma_min{1e-12}, sigma_max{1e6};
    double target_cond{1e12}, min_eig_thresh{1e-8};
    double adapt_factor{2.0};
    int    max_lanczos_iter{50};
    double lanczos_tol{1e-4};
    bool   use_amg{true};
    int    k_eigs{8};          // # of smallest eigenpairs for modification
    double tr_c{1e-2};

    // Spectra knobs
    int spectra_ncv_factor{3}; // ncv ≈ spectra_ncv_factor * nev + 1
    int spectra_maxit{2000};
    double spectra_tol{1e-8};
};

struct RegInfo {
    std::string mode{"AUTO"};
    double sigma{1e-8};
    double cond_before{1e16}, cond_after{1e16};
    double min_eig_before{0.0}, min_eig_after{0.0};
    int nnz_before{0}, nnz_after{0};
    bool converged{true};
};

// ==================== - Minimal AMG (unchanged) ====================
class AMG {
private:
    struct Level {
        spmat A;
        dvec D_inv;
        spmat P, R;
        bool is_coarsest{false};
    };

    std::vector<Level> levels_;
    const double jw_{0.6};

    thread_local static std::vector<double> scratch_vec_;
    thread_local static std::vector<int> scratch_int_;

    static void ensure_scratch(size_t n) {
        if (scratch_vec_.size() < n) {
            scratch_vec_.resize(n * 2);
            scratch_int_.resize(n * 2);
        }
    }

public:
    explicit AMG(const spmat& A_in) {
        spmat A = A_in;
        A.makeCompressed();

        constexpr int MAX_LEVELS = 6;
        constexpr int MIN_COARSE = 50;
        constexpr double STRENGTH_THRESH = 0.1;

        for (int lev = 0; lev < MAX_LEVELS; ++lev) {
            const int n = A.rows();
            if (n <= MIN_COARSE) {
                Level lvl;
                lvl.A = std::move(A);
                lvl.D_inv = compute_diag_inv(lvl.A);
                lvl.is_coarsest = true;
                levels_.push_back(std::move(lvl));
                break;
            }

            auto agg = _aggregate(A, STRENGTH_THRESH);
            const int nc = agg.maxCoeff() + 1;
            if (nc >= n * 0.8) break;

            spmat P = build_tentative_prolongation(agg, n, nc);
            spmat R = spmat(P.transpose());
            spmat Ac = (R * A * P).pruned();
            Ac.makeCompressed();

            Level lvl;
            lvl.A = A;
            lvl.D_inv = compute_diag_inv(A);
            lvl.P = std::move(P);
            lvl.R = std::move(R);
            levels_.push_back(std::move(lvl));

            A = std::move(Ac);
        }
    }

    dvec solve(const dvec& b) const {
        dvec x = dvec::Zero(b.size());
        return vcycle(0, x, b);
    }

private:
    static dvec compute_diag_inv(const spmat& A) {
        const int n = A.rows();
        dvec d_inv(n);
        #pragma omp simd
        for (int i = 0; i < n; ++i) {
            double diag = A.coeff(i, i);
            d_inv(i) = (std::abs(diag) > 1e-16) ? (1.0 / diag) : 1.0;
        }
        return d_inv;
    }

    static Eigen::VectorXi _aggregate(const spmat& A, double theta) {
        const int n = A.rows();
        ensure_scratch(n);

        Eigen::VectorXi agg = Eigen::VectorXi::Constant(n, -1);
        std::vector<char>& marked = reinterpret_cast<std::vector<char>&>(scratch_int_);
        marked.assign(n, 0);

        int cur_agg = 0;
        const dvec diag = A.diagonal();

        for (int i = 0; i < n; ++i) {
            if (marked[i]) continue;
            agg(i) = cur_agg;
            marked[i] = 1;

            const double thresh = theta * std::sqrt(std::abs(diag(i)));
            for (spmat::InnerIterator it(A, i); it; ++it) {
                const int j = it.row();
                if (j == i || marked[j]) continue;
                const double strength = std::abs(it.value()) /
                  std::sqrt(std::abs(diag(i)) * std::abs(diag(j)) + 1e-16);
                if (strength >= thresh) {
                    agg(j) = cur_agg;
                    marked[j] = 1;
                }
            }
            ++cur_agg;
        }
        return agg;
    }

    static spmat build_tentative_prolongation(const Eigen::VectorXi& agg, int n, int nc) {
        std::vector<Eigen::Triplet<double, int>> triplets;
        triplets.reserve(n);
        for (int i = 0; i < n; ++i) triplets.emplace_back(i, agg(i), 1.0);
        spmat P(n, nc);
        P.setFromTriplets(triplets.begin(), triplets.end());
        P.makeCompressed();
        return P;
    }

    static dvec jacobi_sweep(const spmat& A, const dvec& D_inv, dvec x,
                             const dvec& b, double omega, int iters) {
        const int n = A.rows();
        ensure_scratch(n);
        for (int iter = 0; iter < iters; ++iter) {
            dvec& r = reinterpret_cast<dvec&>(scratch_vec_);
            r.resize(n);
            #pragma omp parallel for simd
            for (int i = 0; i < n; ++i) {
                double sum = 0.0;
                for (spmat::InnerIterator it(A, i); it; ++it) {
                    sum += it.value() * x(it.row());
                }
                r(i) = b(i) - sum;
            }
            #pragma omp simd
            for (int i = 0; i < n; ++i) x(i) += omega * D_inv(i) * r(i);
        }
        return x;
    }

    dvec vcycle(int level, dvec x, const dvec& b) const {
        const Level& lev = levels_[level];
        x = jacobi_sweep(lev.A, lev.D_inv, x, b, jw_, 1);
        if (lev.is_coarsest) return jacobi_sweep(lev.A, lev.D_inv, x, b, jw_, 10);
        dvec residual = b - lev.A * x;
        dvec rhs_coarse = lev.R * residual;
        dvec corr_coarse = dvec::Zero(rhs_coarse.size());
        corr_coarse = vcycle(level + 1, corr_coarse, rhs_coarse);
        x += lev.P * corr_coarse;
        x = jacobi_sweep(lev.A, lev.D_inv, x, b, jw_, 1);
        return x;
    }
};

thread_local std::vector<double> AMG::scratch_vec_;
thread_local std::vector<int> AMG::scratch_int_;

// ---------------- Fallback Lanczos (kept for no-Spectra path) ---------------
struct LanczosResult { double eigenvalue{0.0}; bool converged{false}; int iterations{0}; };

class Lanczos {
public:
    static LanczosResult estimate_extreme_eigenvalue(
        const spmat& A, bool largest = true, int max_iter = 30, double tol = 1e-4) {
        const int n = A.rows();
        if(n==0) return {0.0,true,0};
        dvec v = dvec::Ones(n);
        for(int i=0;i<n;i+=2) v(i) = -1.0;
        v.normalize();

        dvec v_prev = dvec::Zero(n);
        double alpha = 0.0, beta = 0.0, theta = 0.0, theta_prev = 0.0;

        for (int k = 0; k < max_iter; ++k) {
            dvec w = A * v;
            alpha = v.dot(w);
            w = w - alpha * v - beta * v_prev;
            beta = w.norm();
            if (beta < 1e-14) return {alpha, true, k + 1};
            v_prev = v; v = w / beta;
            theta_prev = theta;
            theta = alpha;
            if (k > 5 && std::abs(theta - theta_prev) < tol * (std::abs(theta)+1e-16))
                return {theta, true, k + 1};
        }
        return {theta, false, max_iter};
    }
};

// ====================  REGULARIZER (Spectra-powered) ====================
class Regularizer {
private:
    RegConfig cfg_;
    mutable double sigma_{1e-8};
    mutable std::unordered_map<uint64_t, std::unique_ptr<AMG>> amg_cache_;

public:
    explicit Regularizer(RegConfig cfg = {}) : cfg_(std::move(cfg)) {}

    std::pair<spmat, RegInfo> regularize(
        const spmat& H_in,
        int iteration = 0,
        std::optional<double> grad_norm = std::nullopt,
        std::optional<double> tr_radius = std::nullopt) const
    {
        RegInfo info;
        info.nnz_before = H_in.nonZeros();

        spmat H = make_symmetric_(H_in);

        auto [min_eig, max_eig, converged] = spectral_bounds_sparse_(H);
        const double cond = (min_eig > 1e-16) ? (max_eig / min_eig) : 1e16;
        info.min_eig_before = min_eig;
        info.cond_before = cond;
        info.converged = converged;

        adapt_sigma_(min_eig, max_eig, cond, iteration, grad_norm, tr_radius);
        info.sigma = sigma_;

        spmat H_reg;
        if (min_eig < -1e-10) {
            info.mode = "EIGEN_MOD";
            H_reg = eigen_modification_sparse_(H, min_eig, max_eig);
        } else if (cond > cfg_.target_cond) {
            info.mode = "TIKHONOV";
            H_reg = tikhonov_(H, compute_regularization_parameter(min_eig, max_eig, cond));
        } else {
            info.mode = "MINIMAL";
            H_reg = tikhonov_(H, sigma_);
        }

        info.nnz_after = H_reg.nonZeros();

        // post info (optional: estimate after)
        // auto [min2, max2, conv2] = spectral_bounds_sparse_(H_reg);
        // info.min_eig_after = min2;
        // info.cond_after = (min2 > 1e-16) ? (max2 / min2) : info.cond_before;
        // info.converged = info.converged && conv2;

        return {std::move(H_reg), info};
    }

    // Dense variant (unchanged except small polish)
    std::pair<dmat, RegInfo> regularize_auto(const dmat& H_in) const {
        RegInfo info;
        info.nnz_before = static_cast<int>(H_in.size());
        dmat H = 0.5 * (H_in + H_in.transpose());
        auto [min_eig, max_eig] = power_iteration_bounds(H);
        const double cond = (min_eig > 1e-16) ? (max_eig / min_eig) : 1e16;
        info.min_eig_before = min_eig;
        info.cond_before = cond;

        dmat H_reg;
        if (min_eig < -1e-10) {
            info.mode = "EIGEN_MOD";
            H_reg = eigen_floor_dense(H, sigma_);
        } else if (cond > cfg_.target_cond) {
            info.mode = "TIKHONOV";
            H_reg = H + compute_regularization_parameter(min_eig, max_eig, cond) *
                        dmat::Identity(H.rows(), H.cols());
        } else {
            info.mode = "MINIMAL";
            H_reg = H + sigma_ * dmat::Identity(H.rows(), H.cols());
        }

        info.nnz_after = static_cast<int>(H_reg.size());
        info.sigma = sigma_;
        return {std::move(H_reg), info};
    }

private:
    // -------------------- Symmetrize --------------------
    static spmat make_symmetric_(const spmat& A) {
        if (A.rows() != A.cols()) return A;
        if (A.rows() > 1000) {
            bool likely_symmetric = true;
            const int samples = std::min(20, static_cast<int>(A.rows()));
            for (int k = 0; k < samples && likely_symmetric; ++k) {
                int i = k * A.rows() / samples;
                int j = (k + 1) * A.rows() / (samples + 1);
                if (std::abs(A.coeff(i, j) - A.coeff(j, i)) > 1e-12) {
                    likely_symmetric = false;
                }
            }
            if (likely_symmetric) return A;
        }
        spmat AT = spmat(A.transpose());
        return ((A + AT) * 0.5).pruned();
    }

    // -------------------- Spectral bounds (sparse) --------------------
    std::tuple<double,double,bool> spectral_bounds_sparse_(const spmat& H) const {
        const int n = H.rows();
        if (n == 0) return {0.0, 0.0, true};

        double min_ev = 0.0, max_ev = 0.0;
        bool ok_min = false, ok_max = false;

    #if AD_HAVE_SPECTRA
        try {
            // Largest algebraic eigenvalue
            {
                Spectra::SparseSymMatProd<double> op(H);
                int nev = 1;
                int ncv = std::min(n, std::max(2 * nev + 1, cfg_.spectra_ncv_factor * nev + 1));
                Spectra::SymEigsSolver<Spectra::SparseSymMatProd<double>> eigs(op, nev, ncv);
                eigs.init();
                eigs.compute(Spectra::SortRule::LargestAlge, cfg_.spectra_maxit, cfg_.spectra_tol);
                if (eigs.info() == Spectra::CompInfo::Successful) {
                    max_ev = eigs.eigenvalues()(0);
                    ok_max = true;
                }
            }
            // Smallest algebraic eigenvalue via shift-invert with shift 0
            {
                // H must be factorizable; Spectra uses internal LDLT/SparseLU
                Spectra::SparseSymShiftSolve<double> op_shift(H);
                int nev = 1;
                int ncv = std::min(n, std::max(2 * nev + 1, cfg_.spectra_ncv_factor * nev + 1));
                Spectra::SymEigsShiftSolver<Spectra::SparseSymShiftSolve<double>> eigs(op_shift, nev, ncv, 0.0);
                eigs.init();
                eigs.compute(Spectra::SortRule::SmallestAlge, cfg_.spectra_maxit, cfg_.spectra_tol, Spectra::SortRule::SmallestAlge);
                if (eigs.info() == Spectra::CompInfo::Successful) {
                    min_ev = eigs.eigenvalues()(0);
                    ok_min = true;
                }
            }
        } catch (...) {
            // fall back below
        }
    #endif

        if (!(ok_min && ok_max)) {
            // Fallback: lightweight Lanczos/diagonal bounds
            auto max_result = Lanczos::estimate_extreme_eigenvalue(
                H, true, cfg_.max_lanczos_iter, cfg_.lanczos_tol);
            auto min_result = Lanczos::estimate_extreme_eigenvalue(
                H, false, cfg_.max_lanczos_iter, cfg_.lanczos_tol);

            bool conv = max_result.converged && min_result.converged;
            if (!conv) {
                double min_diag = 1e30, max_diag = -1e30;
                const int n = H.rows();
                #pragma omp parallel for reduction(min:min_diag) reduction(max:max_diag)
                for (int i = 0; i < n; ++i) {
                    double d = H.coeff(i, i);
                    min_diag = std::min(min_diag, d);
                    max_diag = std::max(max_diag, d);
                }
                return {min_diag, max_diag, false};
            }
            return {min_result.eigenvalue, max_result.eigenvalue, true};
        }
        return {min_ev, max_ev, true};
    }

    // -------------------- Sigma tuning --------------------
    void adapt_sigma_(double min_eig, double max_eig, double cond,
                      int iteration, std::optional<double> grad_norm,
                      std::optional<double> tr_radius) const {
        if (grad_norm && tr_radius && *tr_radius > 0.0) {
            double tr_sigma = cfg_.tr_c * (*grad_norm) / (*tr_radius);
            sigma_ = std::max(sigma_, tr_sigma);
        }
        if (cond > cfg_.target_cond * 2.0) {
            sigma_ = std::min(cfg_.sigma_max, sigma_ * cfg_.adapt_factor);
        } else if (cond < cfg_.target_cond * 0.1 && min_eig > cfg_.min_eig_thresh) {
            sigma_ = std::max(cfg_.sigma_min, sigma_ / cfg_.adapt_factor);
        }
        if (iteration < 3) sigma_ = std::max(sigma_, 1e-8);
        sigma_ = std::clamp(sigma_, cfg_.sigma_min, cfg_.sigma_max);
    }

    double compute_regularization_parameter(double min_eig, double max_eig, double cond) const {
        if (max_eig <= 0.0) return sigma_;
        double target_min = max_eig / cfg_.target_cond;
        double reg_param = std::max({sigma_, cfg_.min_eig_thresh, target_min - min_eig});
        return std::clamp(reg_param, cfg_.sigma_min, cfg_.sigma_max);
    }

    static spmat tikhonov_(const spmat& H, double sigma) {
        if (sigma == 0.0) return H;
        spmat I(H.rows(), H.cols());
        I.setIdentity();
        return (H + sigma * I).pruned();
    }

    // -------------------- Eigenvalue modification (sparse) --------------------
    spmat eigen_modification_sparse_(const spmat& H, double min_eig, double max_eig) const {
        const int n = H.rows();
        const double target_min = std::max(sigma_, cfg_.min_eig_thresh);
        if (n > 2000) { // keep it lean for huge problems; tune as needed
            double shift = std::max(0.0, target_min - min_eig);
            return tikhonov_(H, shift);
        }

        const int nev = std::clamp(cfg_.k_eigs, 1, std::max(1, n / 4));
        dmat V;        // n x nev
        dvec lambda;   // nev

        bool ok = false;

    #if AD_HAVE_SPECTRA
        try {
            // Smallest eigenpairs via shift-invert at shift=0
            Spectra::SparseSymShiftSolve<double> op_shift(H);
            int ncv = std::min(n, std::max(2 * nev + 1, cfg_.spectra_ncv_factor * nev + 1));
            Spectra::SymEigsShiftSolver<Spectra::SparseSymShiftSolve<double>> eigs(op_shift, nev, ncv, 0.0);
            eigs.init();
            eigs.compute(Spectra::SortRule::SmallestAlge, cfg_.spectra_maxit, cfg_.spectra_tol, Spectra::SortRule::SmallestAlge);
            if (eigs.info() == Spectra::CompInfo::Successful) {
                lambda = eigs.eigenvalues();
                V = eigs.eigenvectors(); // n x nev
                ok = true;
            }
        } catch (...) {
            ok = false;
        }
    #endif

        if (!ok) {
            // Dense fallback for small n
            dmat Hd = dmat(H);
            Eigen::SelfAdjointEigenSolver<dmat> es;
            es.compute(Hd);
            lambda = es.eigenvalues().head(nev);
            V = es.eigenvectors().leftCols(nev);
        }

        dvec delta = dvec::Zero(nev);
        for (int i = 0; i < nev; ++i)
            if (lambda(i) < target_min) delta(i) = target_min - lambda(i);

        if (delta.lpNorm<Eigen::Infinity>() <= 0.0) return H; // nothing to fix

        dmat update = V * delta.asDiagonal() * V.transpose();
        dmat Hdense = dmat(H) + update;
        return Hdense.sparseView().pruned();
    }

    // -------------------- Dense helpers (kept) --------------------
    static std::pair<double, double> power_iteration_bounds(const dmat& H, int max_iter = 20) {
        const int n = H.rows();
        if (n == 0) return {0.0, 0.0};
        dvec v = dvec::Random(n); v.normalize();
        double max_eig = 0.0;
        for (int i = 0; i < max_iter; ++i) {
            v = H * v;
            max_eig = v.norm();
            if (max_eig > 1e-16) v /= max_eig;
        }
        double shift = max_eig * 0.1;
        dmat Hs = H - shift * dmat::Identity(n, n);
        Eigen::FullPivLU<dmat> lu(Hs);
        if (lu.isInvertible()) {
            dvec u = dvec::Random(n); u.normalize();
            double min_inv_eig = 0.0;
            for (int i = 0; i < max_iter; ++i) {
                u = lu.solve(u);
                min_inv_eig = u.norm();
                if (min_inv_eig > 1e-16) u /= min_inv_eig;
            }
            double min_eig = (min_inv_eig > 1e-16) ? (1.0 / min_inv_eig + shift) : shift;
            return {min_eig, max_eig};
        } else {
            return {H.diagonal().minCoeff(), H.diagonal().maxCoeff()};
        }
    }

    static dmat eigen_floor_dense(const dmat& H, double floor) {
        Eigen::SelfAdjointEigenSolver<dmat> es(H);
        dvec ev = es.eigenvalues();
        for (int i = 0; i < ev.size(); ++i) ev(i) = std::max(ev(i), floor);
        return es.eigenvectors() * ev.asDiagonal() * es.eigenvectors().transpose();
    }
};

} // namespace regx
