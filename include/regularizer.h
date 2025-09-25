#pragma once
// regularizer_v2.hpp — High-performance C++23 regularizer
// Key improvements: SIMD, better caching, randomized eigenvalue estimation, memory management

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
#include <execution>

#ifdef __AVX2__
#include <immintrin.h>
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

// ========================= PERFORMANCE UTILITIES =========================

// SIMD-accelerated vector operations
class VectorOps {
public:
    #ifdef __AVX2__
    static double dot_product_avx(const double* a, const double* b, size_t n) {
        __m256d sum = _mm256_setzero_pd();
        size_t i = 0;
        
        // Process 4 doubles at a time
        for (; i + 3 < n; i += 4) {
            __m256d va = _mm256_loadu_pd(&a[i]);
            __m256d vb = _mm256_loadu_pd(&b[i]);
            sum = _mm256_fmadd_pd(va, vb, sum);
        }
        
        // Horizontal sum
        double result[4];
        _mm256_storeu_pd(result, sum);
        double total = result[0] + result[1] + result[2] + result[3];
        
        // Handle remaining elements
        for (; i < n; ++i) {
            total += a[i] * b[i];
        }
        return total;
    }
    
    static void axpy_avx(double alpha, const double* x, double* y, size_t n) {
        __m256d va = _mm256_set1_pd(alpha);
        size_t i = 0;
        
        for (; i + 3 < n; i += 4) {
            __m256d vx = _mm256_loadu_pd(&x[i]);
            __m256d vy = _mm256_loadu_pd(&y[i]);
            __m256d result = _mm256_fmadd_pd(va, vx, vy);
            _mm256_storeu_pd(&y[i], result);
        }
        
        for (; i < n; ++i) {
            y[i] += alpha * x[i];
        }
    }
    #endif
    
    static double robust_dot(const dvec& a, const dvec& b) {
        #ifdef __AVX2__
        if (a.size() >= 8) {
            return dot_product_avx(a.data(), b.data(), a.size());
        }
        #endif
        return a.dot(b);
    }
};

// Memory pool for frequent allocations
class MemoryPool {
private:
    mutable std::vector<std::unique_ptr<dvec>> vector_pool_;
    mutable std::vector<std::unique_ptr<dmat>> matrix_pool_;
    mutable size_t pool_size_{16};
    
public:
    dvec* get_vector(int size) const {
        for (auto& ptr : vector_pool_) {
            if (ptr && ptr->size() == size) {
                auto* result = ptr.release();
                return result;
            }
        }
        return new dvec(size);
    }
    
    void return_vector(dvec* vec) const {
        if (vector_pool_.size() < pool_size_) {
            vector_pool_.emplace_back(vec);
        } else {
            delete vec;
        }
    }
    
    dmat* get_matrix(int rows, int cols) const {
        for (auto& ptr : matrix_pool_) {
            if (ptr && ptr->rows() == rows && ptr->cols() == cols) {
                auto* result = ptr.release();
                return result;
            }
        }
        return new dmat(rows, cols);
    }
    
    void return_matrix(dmat* mat) const {
        if (matrix_pool_.size() < pool_size_) {
            matrix_pool_.emplace_back(mat);
        } else {
            delete mat;
        }
    }
};

// Randomized eigenvalue estimation using modern techniques
class SpectralAnalysis {
private:
    static thread_local MemoryPool pool_;
    
public:
    // Randomized power iteration for extreme eigenvalues
    static std::pair<double, double> randomized_bounds(
        const spmat& A, int num_samples = 8, int power_iters = 4) {
        
        const int n = A.rows();
        if (n == 0) return {0.0, 0.0};
        
        // Use diagonal bounds as initial estimates
        double min_eig = 1e30, max_eig = -1e30;
        
        // #pragma omp parallel for reduction(min:min_eig) reduction(max:max_eig)
        for (int i = 0; i < n; ++i) {
            double d = A.coeff(i, i);
            min_eig = std::min(min_eig, d);
            max_eig = std::max(max_eig, d);
        }
        
        // For small matrices or when diagonal bounds are good enough
        if (n < 200 || (max_eig / std::max(min_eig, 1e-16) < 1e6)) {
            return {min_eig, max_eig};
        }
        
        // Randomized power iteration
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dist(0.0, 1.0);
        
        dvec v(n);
        double est_max = -1e30, est_min = 1e30;
        
        for (int trial = 0; trial < num_samples; ++trial) {
            // Random initialization
            for (int i = 0; i < n; ++i) {
                v(i) = dist(gen);
            }
            v.normalize();
            
            // Power iteration for largest eigenvalue
            for (int iter = 0; iter < power_iters; ++iter) {
                dvec Av = A * v;
                double rayleigh = VectorOps::robust_dot(v, Av);
                est_max = std::max(est_max, rayleigh);
                
                double norm = Av.norm();
                if (norm > 1e-16) v = Av / norm;
            }
            
            // Inverse power iteration for smallest eigenvalue
            // Use simple Richardson iteration instead of solve
            dvec u = v;
            for (int iter = 0; iter < power_iters; ++iter) {
                dvec Au = A * u;
                double rayleigh = VectorOps::robust_dot(u, Au);
                est_min = std::min(est_min, rayleigh);
                
                // Richardson step: u = u - tau * (A*u - lambda*u)
                double tau = 0.1 / std::max(est_max, 1.0);
                u = u - tau * (Au - rayleigh * u);
                u.normalize();
            }
        }
        
        return {std::max(est_min, min_eig), std::max(est_max, max_eig)};
    }
    
    // Block Lanczos for better convergence
    static std::tuple<double, double, bool> block_lanczos_bounds(
        const spmat& A, int block_size = 4, int max_iter = 20) {
        
        const int n = A.rows();
        if (n == 0) return {0.0, 0.0, true};
        
        block_size = std::min(block_size, n / 4);
        if (block_size < 1) block_size = 1;
        
        dmat Q(n, block_size);
        Q.setRandom();
        
        // QR factorization for orthogonalization
        Eigen::HouseholderQR<dmat> qr(Q);
        Q = qr.householderQ() * dmat::Identity(n, block_size);
        
        dmat T = dmat::Zero(block_size * max_iter, block_size * max_iter);
        
        for (int iter = 0; iter < max_iter; ++iter) {
            dmat AQ = A * Q;
            dmat alpha = Q.transpose() * AQ;
            
            // Store in tridiagonal form
            int start = iter * block_size;
            T.block(start, start, block_size, block_size) = alpha;
            
            if (iter > 0) {
                T.block(start - block_size, start, block_size, block_size) = 
                    T.block(start, start - block_size, block_size, block_size).transpose();
            }
            
            if (iter == max_iter - 1) break;
            
            dmat beta_full = AQ - Q * alpha;
            if (iter > 0) {
                // Subtract previous Q contribution
                beta_full -= Q * T.block(start, start - block_size, block_size, block_size);
            }
            
            Eigen::HouseholderQR<dmat> qr_beta(beta_full);
            dmat beta = qr_beta.matrixQR().triangularView<Eigen::Upper>();
            
            if (beta.norm() < 1e-12) break;
            
            Q = qr_beta.householderQ() * dmat::Identity(n, block_size);
            T.block(start + block_size, start, block_size, block_size) = beta;
        }
        
        // Eigenvalues of tridiagonal matrix
        Eigen::SelfAdjointEigenSolver<dmat> es(T.topLeftCorner(
            std::min((int)T.rows(), block_size * max_iter),
            std::min((int)T.cols(), block_size * max_iter)));
        
        if (es.info() == Eigen::Success) {
            dvec evals = es.eigenvalues();
            return {evals.minCoeff(), evals.maxCoeff(), true};
        }
        
        return {0.0, 0.0, false};
    }
};

thread_local MemoryPool SpectralAnalysis::pool_;

// Configuration with adaptive parameters
struct RegConfig {
    double sigma{1e-8}, sigma_min{1e-12}, sigma_max{1e6};
    double target_cond{1e12}, min_eig_thresh{1e-8};
    double adapt_factor{1.5};  // Reduced for stability
    
    // Spectral analysis
    bool use_randomized_bounds{true};
    int randomized_samples{6};  // Reduced for speed
    int randomized_power_iters{3};
    
    // Fallback settings
    int max_lanczos_iter{20};   // Reduced
    double lanczos_tol{1e-3};   // Relaxed
    
    // Spectra settings (when available)
    int spectra_ncv_factor{2};  // Reduced
    int spectra_maxit{500};     // Reduced
    double spectra_tol{1e-6};   // Relaxed
    
    // Eigenvalue modification
    int k_eigs{4};              // Reduced
    bool use_low_rank_update{true};
    
    // Caching
    bool enable_cache{true};
    size_t max_cache_size{32};  // Increased
    
    // Threading
    bool enable_parallel{true};
    
    // Trust region
    double tr_c{1e-2};
    
    // Matrix size thresholds
    int small_matrix_threshold{500};
    int large_matrix_threshold{5000};
};

struct RegInfo {
    std::string mode{"AUTO"};
    double sigma{1e-8};
    double cond_before{1e16}, cond_after{1e16};
    double min_eig_before{0.0}, min_eig_after{0.0};
    int nnz_before{0}, nnz_after{0};
    bool converged{true};
};

// ========================= MAIN REGULARIZER CLASS =========================

class Regularizer {
private:
    RegConfig cfg_;
    mutable double sigma_{1e-8};
    mutable std::unordered_map<uint64_t, std::pair<double, double>> bounds_cache_;
    mutable std::unordered_map<uint64_t, bool> symmetry_cache_;
    mutable MemoryPool pool_;

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
        
        // Step 1: Symmetrization
        spmat H = make_symmetric_(H_in);
        
        // Step 2: Spectral bounds
        auto [min_eig, max_eig, converged] = get_spectral_bounds_(H);
        const double cond = (min_eig > 1e-16) ? (max_eig / min_eig) : 1e16;
        
        info.min_eig_before = min_eig;
        info.cond_before = cond;
        info.converged = converged;
        
        // Step 3: Adaptive sigma tuning
        adapt_sigma_(min_eig, max_eig, cond, iteration, grad_norm, tr_radius);
        info.sigma = sigma_;
        
        // Step 4: Choose regularization strategy
        spmat H_reg;
        if (min_eig < -cfg_.min_eig_thresh) {
            info.mode = "EIGEN_MOD";
            H_reg = eigen_modification_(H, min_eig, max_eig);
        } else if (cond > cfg_.target_cond) {
            info.mode = "TIKHONOV";
            double reg_param = compute_regularization_parameter_(min_eig, max_eig, cond);
            H_reg = tikhonov_(H, reg_param);
        } else {
            info.mode = "MINIMAL";
            H_reg = tikhonov_(H, sigma_);
        }
        
        info.nnz_after = H_reg.nonZeros();
        return {std::move(H_reg), info};
    }

    // Dense variant (for compatibility)
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
            H_reg = H + compute_regularization_parameter_(min_eig, max_eig, cond) *
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
    // Symmetrization with better caching
    spmat make_symmetric_(const spmat& A) const {
        if (A.rows() != A.cols()) return A;
        
        const int n = A.rows();
        
        // For very small matrices, just check directly
        if (n < 50) {
            if (is_symmetric_small_(A)) return A;
            return ((A + spmat(A.transpose())) * 0.5).pruned();
        }
        
        // Hash-based caching
        if (cfg_.enable_cache) {
            uint64_t hash = compute_hash_(A);
            auto it = symmetry_cache_.find(hash);
            if (it != symmetry_cache_.end()) {
                return it->second ? A : ((A + spmat(A.transpose())) * 0.5).pruned();
            }
            
            bool is_sym = is_symmetric_sampled_(A);
            symmetry_cache_[hash] = is_sym;
            
            // Simple cache management
            if (symmetry_cache_.size() > cfg_.max_cache_size) {
                symmetry_cache_.clear();
            }
            
            if (is_sym) return A;
        } else {
            if (is_symmetric_sampled_(A)) return A;
        }
        
        return ((A + spmat(A.transpose())) * 0.5).pruned();
    }
    
    bool is_symmetric_small_(const spmat& A) const {
        for (int k = 0; k < A.outerSize(); ++k) {
            for (spmat::InnerIterator it(A, k); it; ++it) {
                if (std::abs(it.value() - A.coeff(it.col(), it.row())) > 1e-12) {
                    return false;
                }
            }
        }
        return true;
    }
    
    bool is_symmetric_sampled_(const spmat& A) const {
        const int n = A.rows();
        const int samples = std::min(100, n / 10);  // More samples but capped
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, n - 1);
        
        for (int k = 0; k < samples; ++k) {
            int i = dis(gen);
            int j = dis(gen);
            if (i != j && std::abs(A.coeff(i, j) - A.coeff(j, i)) > 1e-11) {
                return false;
            }
        }
        return true;
    }
    
    uint64_t compute_hash_(const spmat& A) const {
        uint64_t hash = A.rows();
        hash = hash * 31 + A.cols();
        hash = hash * 31 + A.nonZeros();
        
        // Sample diagonal and a few off-diagonal elements
        const int n = std::min(16, static_cast<int>(A.rows()));
        for (int i = 0; i < n; ++i) {
            hash = hash * 31 + std::hash<double>{}(A.coeff(i, i));
        }
        
        return hash;
    }
    
    // Spectral bounds with caching
    std::tuple<double, double, bool> get_spectral_bounds_(const spmat& H) const {
        if (cfg_.enable_cache) {
            uint64_t hash = compute_hash_(H);
            auto it = bounds_cache_.find(hash);
            if (it != bounds_cache_.end()) {
                return {it->second.first, it->second.second, true};
            }
        }
        
        auto result = compute_spectral_bounds_(H);
        
        if (cfg_.enable_cache && std::get<2>(result)) {
            uint64_t hash = compute_hash_(H);
            bounds_cache_[hash] = {std::get<0>(result), std::get<1>(result)};
            
            if (bounds_cache_.size() > cfg_.max_cache_size) {
                bounds_cache_.clear();
            }
        }
        
        return result;
    }
    
    std::tuple<double, double, bool> compute_spectral_bounds_(const spmat& H) const {
        const int n = H.rows();
        if (n == 0) return {0.0, 0.0, true};
        
        // Strategy selection based on matrix size
        if (n < cfg_.small_matrix_threshold) {
            // Small matrices: use exact methods
            return compute_exact_bounds_(H);
        } else if (n < cfg_.large_matrix_threshold && cfg_.use_randomized_bounds) {
            // Medium matrices: randomized methods
            auto [min_eig, max_eig] = SpectralAnalysis::randomized_bounds(
                H, cfg_.randomized_samples, cfg_.randomized_power_iters);
            return {min_eig, max_eig, true};
        } else {
            // Large matrices: approximations
            return compute_approximate_bounds_(H);
        }
    }
    
    std::tuple<double, double, bool> compute_exact_bounds_(const spmat& H) const {
        #if AD_HAVE_SPECTRA
        try {
            // Use Spectra for exact computation with tight tolerances
            const int n = H.rows();
            
            // Largest eigenvalue
            Spectra::SparseSymMatProd<double> op(H);
            int nev = 1;
            int ncv = std::min(n, std::max(4, cfg_.spectra_ncv_factor * nev + 1));
            Spectra::SymEigsSolver<Spectra::SparseSymMatProd<double>> eigs_max(op, nev, ncv);
            eigs_max.init();
            eigs_max.compute(Spectra::SortRule::LargestAlge, cfg_.spectra_maxit, cfg_.spectra_tol);
            
            if (eigs_max.info() != Spectra::CompInfo::Successful) {
                return compute_approximate_bounds_(H);
            }
            
            double max_eig = eigs_max.eigenvalues()(0);
            
            // Smallest eigenvalue
            Spectra::SparseSymShiftSolve<double> op_shift(H);
            Spectra::SymEigsShiftSolver<Spectra::SparseSymShiftSolve<double>> eigs_min(op_shift, nev, ncv, 0.0);
            eigs_min.init();
            eigs_min.compute(Spectra::SortRule::SmallestAlge, cfg_.spectra_maxit, cfg_.spectra_tol);
            
            if (eigs_min.info() != Spectra::CompInfo::Successful) {
                return compute_approximate_bounds_(H);
            }
            
            double min_eig = eigs_min.eigenvalues()(0);
            return {min_eig, max_eig, true};
        } catch (...) {
            return compute_approximate_bounds_(H);
        }
        #else
        return compute_approximate_bounds_(H);
        #endif
    }
    
    std::tuple<double, double, bool> compute_approximate_bounds_(const spmat& H) const {
        // Use diagonal bounds as baseline
        const int n = H.rows();
        double min_diag = 1e30, max_diag = -1e30;
        
        // #pragma omp parallel for reduction(min:min_diag) reduction(max:max_diag) if(cfg_.enable_parallel)
        for (int i = 0; i < n; ++i) {
            double d = H.coeff(i, i);
            min_diag = std::min(min_diag, d);
            max_diag = std::max(max_diag, d);
        }
        
        // Use Gershgorin circles for bounds refinement
        double min_gershgorin = 1e30, max_gershgorin = -1e30;
        
        // #pragma omp parallel for reduction(min:min_gershgorin) reduction(max:max_gershgorin) if(cfg_.enable_parallel)
        for (int i = 0; i < n; ++i) {
            double diag = H.coeff(i, i);
            double radius = 0.0;
            
            for (spmat::InnerIterator it(H, i); it; ++it) {
                if (it.row() != i) {
                    radius += std::abs(it.value());
                }
            }
            
            min_gershgorin = std::min(min_gershgorin, diag - radius);
            max_gershgorin = std::max(max_gershgorin, diag + radius);
        }
        
        // Combine bounds
        double min_eig = std::max(min_diag, min_gershgorin);
        double max_eig = std::min(max_diag, max_gershgorin);
        
        return {min_eig, max_eig, false};  // Mark as approximate
    }
    
    // Sigma adaptation
    void adapt_sigma_(double min_eig, double max_eig, double cond,
                      int iteration, std::optional<double> grad_norm,
                      std::optional<double> tr_radius) const {
        
        // Trust region adjustment
        if (grad_norm && tr_radius && *tr_radius > 0.0) {
            double tr_sigma = cfg_.tr_c * (*grad_norm) / (*tr_radius);
            sigma_ = std::max(sigma_, tr_sigma);
        }
        
        // Condition number based adaptation
        if (cond > cfg_.target_cond * 2.0) {
            sigma_ = std::min(cfg_.sigma_max, sigma_ * cfg_.adapt_factor);
        } else if (cond < cfg_.target_cond * 0.1 && min_eig > cfg_.min_eig_thresh) {
            sigma_ = std::max(cfg_.sigma_min, sigma_ / cfg_.adapt_factor);
        }
        
        // Early iteration safeguard
        if (iteration < 5) {
            sigma_ = std::max(sigma_, 1e-8);
        }
        
        sigma_ = std::clamp(sigma_, cfg_.sigma_min, cfg_.sigma_max);
    }
    
    double compute_regularization_parameter_(double min_eig, double max_eig, double cond) const {
        if (max_eig <= 0.0) return sigma_;
        
        double target_min = max_eig / cfg_.target_cond;
        double reg_param = std::max({sigma_, cfg_.min_eig_thresh, target_min - min_eig});
        
        return std::clamp(reg_param, cfg_.sigma_min, cfg_.sigma_max);
    }
    
    // Tikhonov regularization
    spmat tikhonov_(const spmat& H, double sigma) const {
        if (sigma == 0.0) return H;
        
        const int n = H.rows();
        spmat result = H;
        
        // In-place diagonal addition for sparse matrices
        // #pragma omp parallel for if(cfg_.enable_parallel && n > 1000)
        for (int i = 0; i < n; ++i) {
            result.coeffRef(i, i) += sigma;
        }
        
        return result.pruned();
    }
    
    // Eigenvalue modification
    spmat eigen_modification_(const spmat& H, double min_eig, double max_eig) const {
        const int n = H.rows();
        const double target_min = std::max(sigma_, cfg_.min_eig_thresh);
        
        // For large matrices or small negative eigenvalues, use simple shift
        if (n > cfg_.large_matrix_threshold || min_eig > target_min * 0.8) {
            double shift = std::max(0.0, target_min - min_eig);
            return tikhonov_(H, shift);
        }
        
        // For smaller matrices, use low-rank eigenvalue modification
        if (cfg_.use_low_rank_update) {
            return eigen_modification_low_rank_(H, target_min);
        } else {
            double shift = target_min - min_eig;
            return tikhonov_(H, shift);
        }
    }
    
    spmat eigen_modification_low_rank_(const spmat& H, double target_min) const {
        const int n = H.rows();
        const int nev = std::min(cfg_.k_eigs, n / 8);  // Conservative choice
        
        #if AD_HAVE_SPECTRA
        try {
            Spectra::SparseSymShiftSolve<double> op_shift(H);
            int ncv = std::min(n, std::max(2 * nev + 1, cfg_.spectra_ncv_factor * nev + 1));
            Spectra::SymEigsShiftSolver<Spectra::SparseSymShiftSolve<double>> eigs(
                op_shift, nev, ncv, 0.0);
            
            eigs.init();
            eigs.compute(Spectra::SortRule::SmallestAlge, cfg_.spectra_maxit / 2, cfg_.spectra_tol * 2);
            
            if (eigs.info() == Spectra::CompInfo::Successful) {
                dvec lambda = eigs.eigenvalues();
                dmat V = eigs.eigenvectors();
                
                dvec delta = dvec::Zero(nev);
                bool needs_mod = false;
                
                for (int i = 0; i < nev; ++i) {
                    if (lambda(i) < target_min) {
                        delta(i) = target_min - lambda(i);
                        needs_mod = true;
                    }
                }
                
                if (needs_mod) {
                    // Low-rank update: H + V * Delta * V^T
                    dmat update = V * delta.asDiagonal() * V.transpose();
                    dmat H_dense = dmat(H) + update;
                    return H_dense.sparseView(1e-14).pruned();
                }
            }
        } catch (...) {
            // Fall back to shift
        }
        #endif
        
        double shift = std::max(0.0, target_min - H.diagonal().minCoeff());
        return tikhonov_(H, shift);
    }

    // Dense matrix helper functions for compatibility
    std::pair<double, double> power_iteration_bounds(const dmat& H, int max_iter = 15) const {
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

// ==================== AMG (Unchanged but cleaned up) ====================
class AMG {
private:
    struct Level {
        spmat A;
        dvec D_inv;
        spmat P, R;
        bool is_coarsest{false};
    };

    std::vector<Level> levels_;
    const double jacobi_weight_{0.6};

    // Pre-allocated working vectors
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

            auto agg = aggregate_(A, STRENGTH_THRESH);
            const int nc = agg.maxCoeff() + 1;
            if (nc >= n * 0.8) break;

            spmat P = build_prolongation_(agg, n, nc);
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
        return vcycle_(0, x, b);
    }

private:
    static dvec compute_diag_inv(const spmat& A) {
        const int n = A.rows();
        dvec d_inv(n);
        
        // #pragma omp parallel for simd
        for (int i = 0; i < n; ++i) {
            double diag = A.coeff(i, i);
            d_inv(i) = (std::abs(diag) > 1e-16) ? (1.0 / diag) : 1.0;
        }
        return d_inv;
    }

    Eigen::VectorXi aggregate_(const spmat& A, double theta) const {
        const int n = A.rows();
        
        if (work_int_.size() < n) {
            work_int_.resize(n * 2);
        }

        Eigen::VectorXi agg = Eigen::VectorXi::Constant(n, -1);
        std::vector<char> marked(n, 0);

        int cur_agg = 0;
        const dvec diag = A.diagonal();

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

    static spmat build_prolongation_(const Eigen::VectorXi& agg, int n, int nc) {
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

    dvec jacobi_sweep_(const spmat& A, const dvec& D_inv, dvec x,
                      const dvec& b, double omega, int iters) const {
        const int n = A.rows();
        
        if (work_vec_.size() < n) {
            work_vec_.resize(n * 2);
        }
        
        for (int iter = 0; iter < iters; ++iter) {
            std::fill_n(work_vec_.data(), n, 0.0);
            
            // #pragma omp parallel for schedule(dynamic, 64)
            for (int i = 0; i < n; ++i) {
                double sum = 0.0;
                for (spmat::InnerIterator it(A, i); it; ++it) {
                    sum += it.value() * x(it.row());
                }
                work_vec_[i] = b(i) - sum;
            }
            
            #pragma omp simd
            for (int i = 0; i < n; ++i) {
                x(i) += omega * D_inv(i) * work_vec_[i];
            }
        }
        return x;
    }

    dvec vcycle_(int level, dvec x, const dvec& b) const {
        const Level& lev = levels_[level];
        x = jacobi_sweep_(lev.A, lev.D_inv, x, b, jacobi_weight_, 1);
        
        if (lev.is_coarsest) {
            return jacobi_sweep_(lev.A, lev.D_inv, x, b, jacobi_weight_, 10);
        }
        
        dvec residual = b - lev.A * x;
        dvec rhs_coarse = lev.R * residual;
        dvec corr_coarse = dvec::Zero(rhs_coarse.size());
        corr_coarse = vcycle_(level + 1, corr_coarse, rhs_coarse);
        x += lev.P * corr_coarse;
        x = jacobi_sweep_(lev.A, lev.D_inv, x, b, jacobi_weight_, 1);
        return x;
    }
};

// ==================== Lanczos (Unchanged but cleaned) ====================
struct LanczosResult { 
    double eigenvalue{0.0}; 
    bool converged{false}; 
    int iterations{0}; 
};

class Lanczos {
private:
    static thread_local dvec work1_, work2_, work3_;
    
public:
    static LanczosResult estimate_extreme_eigenvalue(
        const spmat& A, bool largest = true, int max_iter = 30, double tol = 1e-4) {
        
        const int n = A.rows();
        if(n == 0) return {0.0, true, 0};
        
        if (work1_.size() != n) {
            work1_.resize(n);
            work2_.resize(n);
            work3_.resize(n);
        }
        
        dvec& v = work1_;
        dvec& v_prev = work2_;
        dvec& w = work3_;
        
        v.setRandom();
        for(int i = 0; i < n; i += 2) v(i) = -v(i);
        v.normalize();
        
        v_prev.setZero();
        double alpha = 0.0, beta = 0.0, theta = 0.0, theta_prev = 0.0;

        for (int k = 0; k < max_iter; ++k) {
            w.noalias() = A * v;
            alpha = v.dot(w);
            w = w - alpha * v - beta * v_prev;
            beta = w.norm();
            
            if (beta < 1e-14) return {alpha, true, k + 1};
            
            v_prev.swap(v);
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

thread_local dvec Lanczos::work1_;
thread_local dvec Lanczos::work2_;
thread_local dvec Lanczos::work3_;

} // namespace regx