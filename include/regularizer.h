#pragma once
// regularizer_.hpp — C++23 regularizer (Spectra-powered, optimized)
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
    
    // Cache settings
    bool enable_cache{true};
    size_t max_cache_size{16};
};

struct RegInfo {
    std::string mode{"AUTO"};
    double sigma{1e-8};
    double cond_before{1e16}, cond_after{1e16};
    double min_eig_before{0.0}, min_eig_after{0.0};
    int nnz_before{0}, nnz_after{0};
    bool converged{true};
};

// ==================== Optimized AMG ====================
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

    // Pre-allocated working vectors (avoid repeated allocations)
    mutable std::vector<double> work_vec_;
    mutable std::vector<int> work_int_;

public:
    explicit AMG(const spmat& A_in) {
        spmat A = A_in;
        A.makeCompressed();

        // Pre-allocate working memory
        work_vec_.reserve(A.rows() * 2);
        work_int_.reserve(A.rows() * 2);

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
        
        // Vectorized diagonal extraction
        #pragma omp parallel for simd
        for (int i = 0; i < n; ++i) {
            double diag = A.coeff(i, i);
            d_inv(i) = (std::abs(diag) > 1e-16) ? (1.0 / diag) : 1.0;
        }
        return d_inv;
    }

    Eigen::VectorXi _aggregate(const spmat& A, double theta) const {
        const int n = A.rows();
        
        // Ensure working memory
        if (work_int_.size() < n) {
            work_int_.resize(n * 2);
        }

        Eigen::VectorXi agg = Eigen::VectorXi::Constant(n, -1);
        std::vector<char> marked(n, 0);

        int cur_agg = 0;
        const dvec diag = A.diagonal();

        // Pre-compute square roots for all diagonal elements
        dvec sqrt_diag(n);
        #pragma omp simd
        for (int i = 0; i < n; ++i) {
            sqrt_diag(i) = std::sqrt(std::abs(diag(i)));
        }

        for (int i = 0; i < n; ++i) {
            if (marked[i]) continue;
            agg(i) = cur_agg;
            marked[i] = 1;

            const double thresh = theta * sqrt_diag(i);
            const double sqrt_diag_i = sqrt_diag(i);
            
            for (spmat::InnerIterator it(A, i); it; ++it) {
                const int j = it.row();
                if (j == i || marked[j]) continue;
                
                const double strength = std::abs(it.value()) /
                  (sqrt_diag_i * sqrt_diag(j) + 1e-16);
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
        for (int i = 0; i < n; ++i) {
            triplets.emplace_back(i, agg(i), 1.0);
        }
        spmat P(n, nc);
        P.setFromTriplets(triplets.begin(), triplets.end());
        P.makeCompressed();
        return P;
    }

    dvec jacobi_sweep(const spmat& A, const dvec& D_inv, dvec x,
                     const dvec& b, double omega, int iters) const {
        const int n = A.rows();
        
        // Use pre-allocated working memory
        if (work_vec_.size() < n) {
            work_vec_.resize(n * 2);
        }
        
        for (int iter = 0; iter < iters; ++iter) {
            // Compute residual: r = b - A*x
            std::fill_n(work_vec_.data(), n, 0.0);
            
            #pragma omp parallel for schedule(dynamic, 64)
            for (int i = 0; i < n; ++i) {
                double sum = 0.0;
                for (spmat::InnerIterator it(A, i); it; ++it) {
                    sum += it.value() * x(it.row());
                }
                work_vec_[i] = b(i) - sum;
            }
            
            // Update: x += omega * D_inv * r
            #pragma omp simd
            for (int i = 0; i < n; ++i) {
                x(i) += omega * D_inv(i) * work_vec_[i];
            }
        }
        return x;
    }

    dvec vcycle(int level, dvec x, const dvec& b) const {
        const Level& lev = levels_[level];
        x = jacobi_sweep(lev.A, lev.D_inv, x, b, jw_, 1);
        
        if (lev.is_coarsest) {
            return jacobi_sweep(lev.A, lev.D_inv, x, b, jw_, 10);
        }
        
        dvec residual = b - lev.A * x;
        dvec rhs_coarse = lev.R * residual;
        dvec corr_coarse = dvec::Zero(rhs_coarse.size());
        corr_coarse = vcycle(level + 1, corr_coarse, rhs_coarse);
        x += lev.P * corr_coarse;
        x = jacobi_sweep(lev.A, lev.D_inv, x, b, jw_, 1);
        return x;
    }
};

// ---------------- Optimized Lanczos ---------------
struct LanczosResult { 
    double eigenvalue{0.0}; 
    bool converged{false}; 
    int iterations{0}; 
};

class Lanczos {
private:
    // Cache for working vectors
    static thread_local dvec work1_, work2_, work3_;
    
public:
    static LanczosResult estimate_extreme_eigenvalue(
        const spmat& A, bool largest = true, int max_iter = 30, double tol = 1e-4) {
        
        const int n = A.rows();
        if(n == 0) return {0.0, true, 0};
        
        // Use cached working vectors
        if (work1_.size() != n) {
            work1_.resize(n);
            work2_.resize(n);
            work3_.resize(n);
        }
        
        dvec& v = work1_;
        dvec& v_prev = work2_;
        dvec& w = work3_;
        
        // Better initialization
        v.setRandom();
        for(int i = 0; i < n; i += 2) v(i) = -v(i);
        v.normalize();
        
        v_prev.setZero();
        double alpha = 0.0, beta = 0.0, theta = 0.0, theta_prev = 0.0;

        for (int k = 0; k < max_iter; ++k) {
            w.noalias() = A * v;  // Use noalias for better performance
            alpha = v.dot(w);
            w = w - alpha * v - beta * v_prev;
            beta = w.norm();
            
            if (beta < 1e-14) return {alpha, true, k + 1};
            
            v_prev.swap(v);  // Avoid copying
            v = w / beta;
            
            theta_prev = theta;
            theta = alpha;
            
            if (k > 5 && std::abs(theta - theta_prev) < tol * (std::abs(theta) + 1e-16)) {
                return {theta, true, k + 1};
            }
        }
        return {theta, false, max_iter};
    }
};

// Thread-local storage for Lanczos working vectors
thread_local dvec Lanczos::work1_;
thread_local dvec Lanczos::work2_;
thread_local dvec Lanczos::work3_;

// ==================== Optimized REGULARIZER ====================
class Regularizer {
private:
    RegConfig cfg_;
    mutable double sigma_{1e-8};
    mutable std::unordered_map<uint64_t, std::unique_ptr<AMG>> amg_cache_;
    
    // Symmetry check cache
    mutable std::unordered_map<uint64_t, bool> symmetry_cache_;

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

        spmat H = make_symmetric_optimized_(H_in);

        auto [min_eig, max_eig, converged] = spectral_bounds_sparse_optimized_(H);
        const double cond = (min_eig > 1e-16) ? (max_eig / min_eig) : 1e16;
        info.min_eig_before = min_eig;
        info.cond_before = cond;
        info.converged = converged;

        adapt_sigma_(min_eig, max_eig, cond, iteration, grad_norm, tr_radius);
        info.sigma = sigma_;

        spmat H_reg;
        if (min_eig < -1e-10) {
            info.mode = "EIGEN_MOD";
            H_reg = eigen_modification_sparse_optimized_(H, min_eig, max_eig);
        } else if (cond > cfg_.target_cond) {
            info.mode = "TIKHONOV";
            H_reg = tikhonov_optimized_(H, compute_regularization_parameter(min_eig, max_eig, cond));
        } else {
            info.mode = "MINIMAL";
            H_reg = tikhonov_optimized_(H, sigma_);
        }

        info.nnz_after = H_reg.nonZeros();
        return {std::move(H_reg), info};
    }

    // Dense variant (unchanged except small polish)
    std::pair<dmat, RegInfo> regularize_auto(const dmat& H_in) const {
        RegInfo info;
        info.nnz_before = static_cast<int>(H_in.size());
        dmat H = 0.5 * (H_in + H_in.transpose());
        auto [min_eig, max_eig] = power_iteration_bounds_optimized(H);
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
    // -------------------- Optimized Symmetrize --------------------
    spmat make_symmetric_optimized_(const spmat& A) const {
        if (A.rows() != A.cols()) return A;
        
        // Hash-based caching for symmetry check
        if (cfg_.enable_cache) {
            uint64_t hash = compute_matrix_hash_(A);
            auto it = symmetry_cache_.find(hash);
            if (it != symmetry_cache_.end()) {
                if (it->second) return A;  // Already symmetric
            } else {
                bool is_sym = check_symmetry_fast_(A);
                symmetry_cache_[hash] = is_sym;
                if (symmetry_cache_.size() > cfg_.max_cache_size) {
                    symmetry_cache_.clear(); // Simple cache eviction
                }
                if (is_sym) return A;
            }
        } else {
            if (check_symmetry_fast_(A)) return A;
        }
        
        // Make symmetric
        spmat AT = spmat(A.transpose());
        return ((A + AT) * 0.5).pruned();
    }
    
    bool check_symmetry_fast_(const spmat& A) const {
        if (A.rows() != A.cols()) return false;
        
        const int n = A.rows();
        if (n > 1000) {
            // Sample-based check for large matrices
            const int samples = std::min(50, n / 20);  // Increased samples
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, n - 1);
            
            for (int k = 0; k < samples; ++k) {
                int i = dis(gen);
                int j = dis(gen);
                if (i != j && std::abs(A.coeff(i, j) - A.coeff(j, i)) > 1e-12) {
                    return false;
                }
            }
            return true;
        } else {
            // Full check for small matrices
            for (int k = 0; k < A.outerSize(); ++k) {
                for (spmat::InnerIterator it(A, k); it; ++it) {
                    if (std::abs(it.value() - A.coeff(it.col(), it.row())) > 1e-12) {
                        return false;
                    }
                }
            }
            return true;
        }
    }
    
    uint64_t compute_matrix_hash_(const spmat& A) const {
        // Simple hash based on structure and a few values
        uint64_t hash = A.rows();
        hash = hash * 31 + A.cols();
        hash = hash * 31 + A.nonZeros();
        
        // Sample a few values
        if (A.nonZeros() > 0) {
            const int samples = std::min(16, static_cast<int>(A.nonZeros()));
            for (int i = 0; i < samples; ++i) {
                hash = hash * 31 + std::hash<double>{}(A.valuePtr()[i * A.nonZeros() / samples]);
            }
        }
        return hash;
    }

    // -------------------- Optimized Spectral bounds --------------------
    std::tuple<double,double,bool> spectral_bounds_sparse_optimized_(const spmat& H) const {
        const int n = H.rows();
        if (n == 0) return {0.0, 0.0, true};

        double min_ev = 0.0, max_ev = 0.0;
        bool ok_min = false, ok_max = false;

    #if AD_HAVE_SPECTRA
        try {
            // Use more aggressive convergence settings for speed
            const double fast_tol = std::max(cfg_.spectra_tol * 10, 1e-6);
            const int fast_maxit = std::min(cfg_.spectra_maxit / 2, 1000);
            
            // Largest algebraic eigenvalue
            {
                Spectra::SparseSymMatProd<double> op(H);
                int nev = 1;
                int ncv = std::min(n, std::max(4, cfg_.spectra_ncv_factor * nev + 1));
                Spectra::SymEigsSolver<Spectra::SparseSymMatProd<double>> eigs(op, nev, ncv);
                eigs.init();
                eigs.compute(Spectra::SortRule::LargestAlge, fast_maxit, fast_tol);
                if (eigs.info() == Spectra::CompInfo::Successful) {
                    max_ev = eigs.eigenvalues()(0);
                    ok_max = true;
                }
            }
            
            // Smallest algebraic eigenvalue
            if (ok_max) {  // Only compute if max succeeded
                Spectra::SparseSymShiftSolve<double> op_shift(H);
                int nev = 1;
                int ncv = std::min(n, std::max(4, cfg_.spectra_ncv_factor * nev + 1));
                Spectra::SymEigsShiftSolver<Spectra::SparseSymShiftSolve<double>> eigs(op_shift, nev, ncv, 0.0);
                eigs.init();
                eigs.compute(Spectra::SortRule::SmallestAlge, fast_maxit, fast_tol, Spectra::SortRule::SmallestAlge);
                if (eigs.info() == Spectra::CompInfo::Successful) {
                    min_ev = eigs.eigenvalues()(0);
                    ok_min = true;
                }
            }
        } catch (...) {
            // fall back below
            ok_min = ok_max = false;
        }
    #endif

        if (!(ok_min && ok_max)) {
            // Fast diagonal bounds first
            double min_diag = 1e30, max_diag = -1e30;
            #pragma omp parallel for reduction(min:min_diag) reduction(max:max_diag)
            for (int i = 0; i < n; ++i) {
                double d = H.coeff(i, i);
                min_diag = std::min(min_diag, d);
                max_diag = std::max(max_diag, d);
            }
            
            // Use Lanczos only if diagonal bounds suggest we need more accuracy
            if (max_diag / std::max(min_diag, 1e-16) > cfg_.target_cond * 0.1) {
                auto max_result = Lanczos::estimate_extreme_eigenvalue(
                    H, true, cfg_.max_lanczos_iter / 2, cfg_.lanczos_tol * 2);
                auto min_result = Lanczos::estimate_extreme_eigenvalue(
                    H, false, cfg_.max_lanczos_iter / 2, cfg_.lanczos_tol * 2);
                
                if (max_result.converged && min_result.converged) {
                    return {min_result.eigenvalue, max_result.eigenvalue, true};
                }
            }
            
            return {min_diag, max_diag, false};
        }
        return {min_ev, max_ev, true};
    }

    // -------------------- Optimized Tikhonov --------------------
    spmat tikhonov_optimized_(const spmat& H, double sigma) const {
        if (sigma == 0.0) return H;
        
        const int n = H.rows();
        
        // For small sigma or sparse matrices, add directly to diagonal
        if (sigma < 1e-6 || H.nonZeros() < n * n / 4) {
            spmat result = H;
            
            // Add to diagonal in-place
            for (int i = 0; i < n; ++i) {
                result.coeffRef(i, i) += sigma;
            }
            return result.pruned();
        }
        
        // For larger sigma, use identity addition
        spmat I(n, n);
        I.setIdentity();
        return (H + sigma * I).pruned();
    }

    // -------------------- Sigma tuning (unchanged) --------------------
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

    // -------------------- Optimized Eigenvalue modification --------------------
    spmat eigen_modification_sparse_optimized_(const spmat& H, double min_eig, double max_eig) const {
        const int n = H.rows();
        const double target_min = std::max(sigma_, cfg_.min_eig_thresh);
        
        // Early exit for large problems or when shift is sufficient
        if (n > 2000 || min_eig > target_min * 0.9) {
            double shift = std::max(0.0, target_min - min_eig);
            return tikhonov_optimized_(H, shift);
        }

        const int nev = std::clamp(cfg_.k_eigs, 1, std::max(1, n / 4));
        dmat V;        // n x nev
        dvec lambda;   // nev

        bool ok = false;

    #if AD_HAVE_SPECTRA
        try {
            // Use faster settings for eigenvalue modification
            const double fast_tol = std::max(cfg_.spectra_tol * 5, 1e-6);
            const int fast_maxit = std::min(cfg_.spectra_maxit / 2, 1000);
            
            Spectra::SparseSymShiftSolve<double> op_shift(H);
            int ncv = std::min(n, std::max(2 * nev + 1, cfg_.spectra_ncv_factor * nev + 1));
            Spectra::SymEigsShiftSolver<Spectra::SparseSymShiftSolve<double>> eigs(op_shift, nev, ncv, 0.0);
            eigs.init();
            eigs.compute(Spectra::SortRule::SmallestAlge, fast_maxit, fast_tol, Spectra::SortRule::SmallestAlge);
            if (eigs.info() == Spectra::CompInfo::Successful) {
                lambda = eigs.eigenvalues();
                V = eigs.eigenvectors();
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
        bool needs_modification = false;
        for (int i = 0; i < nev; ++i) {
            if (lambda(i) < target_min) {
                delta(i) = target_min - lambda(i);
                needs_modification = true;
            }
        }

        if (!needs_modification) return H;

        // Use low-rank update: H_new = H + V * Delta * V^T
        dmat update = V * delta.asDiagonal() * V.transpose();
        dmat Hdense = dmat(H) + update;
        return Hdense.sparseView().pruned();
    }

    // -------------------- Optimized Dense helpers --------------------
    std::pair<double, double> power_iteration_bounds_optimized(const dmat& H, int max_iter = 15) const {
        const int n = H.rows();
        if (n == 0) return {0.0, 0.0};
        
        // Use fewer iterations for speed
        dvec v = dvec::Random(n); 
        v.normalize();
        double max_eig = 0.0;
        
        for (int i = 0; i < max_iter; ++i) {
            v.noalias() = H * v;  // Use noalias
            max_eig = v.norm();
            if (max_eig > 1e-16) v /= max_eig;
            else break;
        }
        
        // For minimum eigenvalue, use shift-and-invert with adaptive shift
        double shift = max_eig * 0.05;  // Smaller shift for better conditioning
        dmat Hs = H - shift * dmat::Identity(n, n);
        
        Eigen::FullPivLU<dmat> lu(Hs);
        if (lu.isInvertible()) {
            dvec u = dvec::Random(n); 
            u.normalize();
            double min_inv_eig = 0.0;
            
            for (int i = 0; i < max_iter; ++i) {
                u = lu.solve(u);
                min_inv_eig = u.norm();
                if (min_inv_eig > 1e-16) u /= min_inv_eig;
                else break;
            }
            double min_eig = (min_inv_eig > 1e-16) ? (1.0 / min_inv_eig + shift) : shift;
            return {min_eig, max_eig};
        } else {
            // Fallback to diagonal bounds
            return {H.diagonal().minCoeff(), H.diagonal().maxCoeff()};
        }
    }

    static dmat eigen_floor_dense(const dmat& H, double floor) {
        Eigen::SelfAdjointEigenSolver<dmat> es(H);
        dvec ev = es.eigenvalues();
        
        // Vectorized eigenvalue flooring
        #pragma omp simd
        for (int i = 0; i < ev.size(); ++i) {
            ev(i) = std::max(ev(i), floor);
        }
        
        return es.eigenvectors() * ev.asDiagonal() * es.eigenvectors().transpose();
    }
};

} // namespace regx