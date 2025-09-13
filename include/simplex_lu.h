#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>  // needed for sparse backend
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

// ======================================================
// Markowitz + rook LU with permutations and refinement
// ======================================================
class MarkowitzLU {
public:
    MarkowitzLU() = default;
    MarkowitzLU(const Eigen::MatrixXd &A, double pivot_rel = 1e-12,
                double abs_floor = 1e-16, int rook_iters = 2) {
        factor(A, pivot_rel, abs_floor, rook_iters);
    }

    void factor(const Eigen::MatrixXd &A, double pivot_rel = 1e-12,
                double abs_floor = 1e-16, int rook_iters = 2) {
        if (A.rows() != A.cols())
            throw std::invalid_argument("MarkowitzLU: square only");
        n_ = static_cast<int>(A.rows());
        pivot_rel_ = pivot_rel;
        abs_floor_ = abs_floor;
        rook_iters_ = rook_iters;
        L_ = Eigen::MatrixXd::Zero(n_, n_);
        U_ = A;
        Pr_.resize(n_);
        Pc_.resize(n_);
        std::iota(Pr_.begin(), Pr_.end(), 0);
        std::iota(Pc_.begin(), Pc_.end(), 0);
        factorize_();
    }

    Eigen::VectorXd solve(const Eigen::VectorXd &b) const {
        if (b.size() != n_)
            throw std::invalid_argument("MarkowitzLU::solve size mismatch");

        // P_r * b
        Eigen::VectorXd Pb = apply_Pr_(b);

        // Forward/back solves
        Eigen::VectorXd z = forward_sub_(L_, Pb);
        Eigen::VectorXd w = back_sub_(U_, z);

        // iterative refinement in permuted space
        const int max_refinements = 3;
        const double refinement_tol = 1e-14;

        for (int iter = 0; iter < max_refinements; ++iter) {
            // r = Pb - L*(U*w)
            Eigen::VectorXd r = Pb - L_ * (U_ * w);
            double denom = std::max(1.0, Pb.lpNorm<Eigen::Infinity>());
            double backward_err = r.lpNorm<Eigen::Infinity>() / denom;
            if (backward_err < refinement_tol) break;

            Eigen::VectorXd dz = forward_sub_(L_, r);
            Eigen::VectorXd dw = back_sub_(U_, dz);
            if (!dw.array().isFinite().all() ||
                dw.lpNorm<Eigen::Infinity>() < 1e-16) {
                break;
            }
            w += dw;
        }

        // x = P_c * w
        return apply_Pc_(w);
    }

    Eigen::VectorXd solveT(const Eigen::VectorXd &c) const {
        if (c.size() != n_)
            throw std::invalid_argument("MarkowitzLU::solveT size mismatch");

        // P_c^T * c
        Eigen::VectorXd PcTc = apply_PcT_(c);

        // Forward/back solves on transposed
        Eigen::VectorXd t = forward_sub_(U_.transpose(), PcTc);
        Eigen::VectorXd s = back_sub_(L_.transpose(), t);

        // iterative refinement for transpose
        const int max_refinements = 3;
        const double refinement_tol = 1e-14;

        for (int iter = 0; iter < max_refinements; ++iter) {
            // r = PcTc - U^T*(L^T*s)
            Eigen::VectorXd r = PcTc - U_.transpose() * (L_.transpose() * s);
            double denom = std::max(1.0, PcTc.lpNorm<Eigen::Infinity>());
            double backward_err = r.lpNorm<Eigen::Infinity>() / denom;
            if (backward_err < refinement_tol) break;

            Eigen::VectorXd dt = forward_sub_(U_.transpose(), r);
            Eigen::VectorXd ds = back_sub_(L_.transpose(), dt);
            if (!ds.array().isFinite().all() ||
                ds.lpNorm<Eigen::Infinity>() < 1e-16) {
                break;
            }
            s += ds;
        }

        // return P_r^T * s
        return apply_PrT_inv_(s);
    }

    int n() const noexcept { return n_; }

    // Expose L/U for local in-place updates by FTBasis (dense path only)
    Eigen::MatrixXd& L() { return L_; }
    Eigen::MatrixXd& U() { return U_; }

private:
    static constexpr double kSingFloor_ = 1e-18;

    static Eigen::VectorXd forward_sub_(const Eigen::MatrixXd &L,
                                        const Eigen::VectorXd &b) {
        // Unit-lower triangular solve (L has ones on diagonal by construction)
        // Use manual code to keep exact behavior and avoid dependencies.
        const int n = static_cast<int>(L.rows());
        Eigen::VectorXd x = b;
        for (int i = 0; i < n; ++i) {
            double s = L.row(i).head(i).dot(x.head(i));
            double piv = L(i, i); // should be 1.0
            if (std::abs(piv) < kSingFloor_ || !std::isfinite(piv))
                throw std::runtime_error("Singular lower triangular");
            x(i) = (x(i) - s) / piv;
        }
        return x;
    }
    static Eigen::VectorXd back_sub_(const Eigen::MatrixXd &U,
                                     const Eigen::VectorXd &b) {
        const int n = static_cast<int>(U.rows());
        Eigen::VectorXd x = b;
        for (int i = n - 1; i >= 0; --i) {
            double s = U.row(i)
                           .segment(i + 1, n - (i + 1))
                           .dot(x.segment(i + 1, n - (i + 1)));
            double piv = U(i, i);
            if (std::abs(piv) < kSingFloor_ || !std::isfinite(piv))
                throw std::runtime_error("Singular upper triangular");
            x(i) = (x(i) - s) / piv;
        }
        return x;
    }

    Eigen::VectorXd apply_Pr_(const Eigen::VectorXd &v) const {
        Eigen::VectorXd out(n_);
        for (int i = 0; i < n_; ++i)
            out(i) = v(Pr_[i]);
        return out;
    }
    Eigen::VectorXd apply_PrT_inv_(const Eigen::VectorXd &y) const {
        // returns P_r^T * y (i.e., apply inverse row permutation)
        Eigen::VectorXd out(n_);
        for (int i = 0; i < n_; ++i)
            out(Pr_[i]) = y(i);
        return out;
    }
    Eigen::VectorXd apply_Pc_(const Eigen::VectorXd &x) const {
        Eigen::VectorXd out(n_);
        for (int i = 0; i < n_; ++i)
            out(Pc_[i]) = x(i);
        return out;
    }
    Eigen::VectorXd apply_PcT_(const Eigen::VectorXd &c) const {
        Eigen::VectorXd out(n_);
        for (int i = 0; i < n_; ++i)
            out(i) = c(Pc_[i]);
        return out;
    }

    void swap_rows_(int i, int j) {
        if (i == j)
            return;
        U_.row(i).swap(U_.row(j));
        L_.row(i).head(i).swap(L_.row(j).head(i));
        std::swap(Pr_[i], Pr_[j]);
    }
    void swap_cols_(int i, int j) {
        if (i == j)
            return;
        U_.col(i).swap(U_.col(j));
        std::swap(Pc_[i], Pc_[j]);
    }

    static std::pair<std::vector<int>, std::vector<int>>
    nnz_row_col_(const Eigen::MatrixXd &M, double eps = 1e-16) {
        const int r = (int)M.rows(), c = (int)M.cols();
        std::vector<int> rn(r, 0), cn(c, 0);
        for (int i = 0; i < r; ++i)
            for (int j = 0; j < c; ++j)
                if (std::abs(M(i, j)) > eps) {
                    rn[i]++;
                    cn[j]++;
                }
        return {rn, cn};
    }

    std::tuple<int, int, double> choose_pivot_(int k,
                                               const Eigen::VectorXd &colmax) {
        int best_i = -1, best_j = -1;
        double best_val = 0.0;
        long best_score = -1;
        Eigen::MatrixXd sub = U_.block(k, k, n_ - k, n_ - k);
        auto [rn, cn] = nnz_row_col_(sub);
        for (int i = k; i < n_; ++i) {
            for (int j = k; j < n_; ++j) {
                double aij = U_(i, j);
                if (std::abs(aij) >=
                    pivot_rel_ * std::max(colmax(j - k), abs_floor_)) {
                    long score = (long)(rn[i - k] - 1) * (long)(cn[j - k] - 1);
                    if (score > best_score ||
                        (score == best_score &&
                         std::abs(aij) > std::abs(best_val))) {
                        best_score = score;
                        best_i = i;
                        best_j = j;
                        best_val = aij;
                    }
                }
            }
        }
        if (best_i >= 0)
            return {best_i, best_j, best_val};

        // rook pivoting (early exit if no movement)
        int i_idx;
        U_.col(k).segment(k, n_ - k).cwiseAbs().maxCoeff(&i_idx);
        int i = k + i_idx;
        int j_idx;
        U_.row(i).segment(k, n_ - k).cwiseAbs().maxCoeff(&j_idx);
        int j = k + j_idx;
        for (int t = 0; t < std::max(0, rook_iters_); ++t) {
            int prev_i = i, prev_j = j;
            U_.col(j).segment(k, n_ - k).cwiseAbs().maxCoeff(&i_idx);
            i = k + i_idx;
            U_.row(i).segment(k, n_ - k).cwiseAbs().maxCoeff(&j_idx);
            j = k + j_idx;
            if (i == prev_i && j == prev_j) break;
        }
        double val = U_(i, j);
        double col_abs_max = U_.col(j).segment(k, n_ - k).cwiseAbs().maxCoeff();
        if (std::abs(val) >= std::max(abs_floor_, pivot_rel_ * col_abs_max))
            return {i, j, val};
        return {-1, -1, 0.0};
    }

    void factorize_() {
        const double inf_norm = L1_inf_norm_(U_);
        for (int k = 0; k < n_; ++k) {
            Eigen::VectorXd colmax = Eigen::VectorXd::Ones(n_ - k);
            if (k < n_) {
                colmax = U_.block(k, k, n_ - k, n_ - k)
                             .cwiseAbs()
                             .colwise()
                             .maxCoeff()
                             .transpose();
                for (int t = 0; t < colmax.size(); ++t)
                    if (colmax(t) < abs_floor_)
                        colmax(t) = 1.0;
            }
            auto [pi, pj, pval] = choose_pivot_(k, colmax);
            if (pi < 0) {
                Eigen::MatrixXd sub = U_.block(k, k, n_ - k, n_ - k).cwiseAbs();
                Eigen::Index rr, cc;
                sub.maxCoeff(&rr, &cc);
                pi = k + (int)rr;
                pj = k + (int)cc;
                const double floor_adapt =
                    std::max(abs_floor_, 10 * std::numeric_limits<double>::epsilon() * inf_norm);
                if (std::abs(U_(pi, pj)) < floor_adapt) {
                    throw std::runtime_error(
                        "MarkowitzLU: singular matrix (no acceptable pivot)");
                }
            }
            swap_rows_(k, pi);
            swap_cols_(k, pj);
            double piv = U_(k, k);
            const double floor_adapt =
                std::max(abs_floor_, 10 * std::numeric_limits<double>::epsilon() * inf_norm);
            if (std::abs(piv) < floor_adapt || !std::isfinite(piv)) {
                throw std::runtime_error("MarkowitzLU: numerically singular pivot");
            }
            L_(k, k) = 1.0;
            for (int i = k + 1; i < n_; ++i) {
                double lik = U_(i, k);
                if (lik != 0.0) {
                    L_(i, k) = lik / piv;
                    U_.row(i).segment(k, n_ - k) -=
                        L_(i, k) * U_.row(k).segment(k, n_ - k);
                }
            }
        }
    }

    static double L1_inf_norm_(const Eigen::MatrixXd &A) {
        if (A.size() == 0)
            return 0.0;
        double maxrow = 0.0;
        for (int i = 0; i < A.rows(); ++i) {
            double s = A.row(i).cwiseAbs().sum();
            if (s > maxrow)
                maxrow = s;
        }
        return maxrow;
    }

private:
    int n_{0};
    double pivot_rel_{1e-12}, abs_floor_{1e-16};
    int rook_iters_{2};
    Eigen::MatrixXd L_, U_;
    std::vector<int> Pr_, Pc_;
};

// ======================================================
// Forrest–Tomlin product-form updates over LU backends
//   - Dense: MarkowitzLU (+ optional FT in-place updates)
//   - Sparse: Eigen::SparseLU with cached B and B^T factorizations
// ======================================================
class FTBasis {
public:
    using DenseMat = Eigen::MatrixXd;
    using SparseMat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

    struct Options {
        int refactor_every = 64;
        int compress_every = 32;
        double pivot_rel = 1e-12;
        double abs_floor = 1e-16;
        double alpha_tol = 1e-10;
        double z_inf_guard = 1e6;
        bool sparse_amd = true;
        double sparse_drop_tol = 0.0;

        // New: update mode and FT locality cap (dense only)
        enum class UpdateMode { EtaStack, ForrestTomlin };
        UpdateMode update_mode = UpdateMode::ForrestTomlin;
        int ft_bandwidth_cap = 16; // 0 = unlimited, else local band half-width around pivot col
    };

    struct Eta {
        int j;
        Eigen::VectorXd u; // new_col - old_col
        Eigen::VectorXd z; // B^{-1} u
        Eigen::VectorXd w; // B^{-T} e_j
        double alpha;      // 1 + z(j)
    };

    FTBasis(const DenseMat &A, const std::vector<int> &basis)
        : FTBasis(A, basis, Options{}) {}

    FTBasis(const DenseMat &A, const std::vector<int> &basis,
            const Options &opt)
        : A_dense_(std::cref(A)), Bcols_dense_(), lu_dense_(),
          A_sparse_(*(const SparseMat *)nullptr), Bcols_sparse_(),
          A_is_sparse_(false), m_((int)A.rows()), basis_(basis), opt_(opt),
          etas_(), update_count_(0), B_sparse_(), BT_sparse_(), solver_B_(),
          solver_BT_() {
        if ((int)basis_.size() != m_)
            throw std::invalid_argument("FTBasis: basis size must equal m");
        Bcols_dense_.resize(m_);
        for (int i = 0; i < m_; ++i)
            Bcols_dense_[i] = A_dense_.get().col(basis_[i]);
        dense_refactor_();
    }

    FTBasis(const SparseMat &A, const std::vector<int> &basis)
        : FTBasis(A, basis, Options{}) {}

    FTBasis(const SparseMat &A, const std::vector<int> &basis,
            const Options &opt)
        : A_dense_(*(const DenseMat *)nullptr), Bcols_dense_(), lu_dense_(),
          A_sparse_(std::cref(A)), Bcols_sparse_(), A_is_sparse_(true),
          m_((int)A.rows()), basis_(basis), opt_(opt), etas_(),
          update_count_(0), B_sparse_(), BT_sparse_(), solver_B_(),
          solver_BT_() {
        if ((int)basis_.size() != m_)
            throw std::invalid_argument("FTBasis: basis size must equal m");
        Bcols_sparse_.resize(m_);
        for (int i = 0; i < m_; ++i)
            Bcols_sparse_[i] = A_sparse_.get().col(basis_[i]);
        sparse_refactor_();
    }

    // Back-compat delegating ctor (old 8-arg signature on dense A)
    FTBasis(const DenseMat &A, const std::vector<int> &basis,
            int refactor_every, int compress_every, double pivot_rel,
            double abs_floor, double alpha_tol, double z_inf_guard)
        : FTBasis(A, basis,
                  Options{refactor_every, compress_every, pivot_rel, abs_floor,
                          alpha_tol, z_inf_guard,
                          /*sparse_amd=*/true, /*sparse_drop_tol=*/0.0,
                          Options::UpdateMode::EtaStack, 16}) {}

    // -------- Basic info --------
    int rows() const noexcept { return m_; }
    const std::vector<int> &basis() const noexcept { return basis_; }
    const std::vector<Eta> &etas() const noexcept { return etas_; }

    // -------- Solves --------
    Eigen::VectorXd solve_B(const Eigen::VectorXd &b) const {
        Eigen::VectorXd x = A_is_sparse_ ? sparse_solve_(b) : dense_solve_(b);
        if (!etas_.empty())
            x = apply_etas_solve_(x);
        return x;
    }

    Eigen::VectorXd solve_BT(const Eigen::VectorXd &c) const {
        Eigen::VectorXd y = A_is_sparse_ ? sparse_solveT_(c) : dense_solveT_(c);
        if (!etas_.empty())
            y = apply_etas_solve_T_(y);
        return y;
    }

    // -------- Column replacement (pivot) --------
    void replace_column(int j, const Eigen::VectorXd &new_col_dense) {
        replace_column_impl_(j, new_col_dense);
    }

    template <typename Derived>
    void replace_column(int j,
                        const Eigen::SparseMatrixBase<Derived> &new_col_sparse) {
        Eigen::SparseMatrix<double> tmp = new_col_sparse.derived().eval(); // m x 1
        Eigen::VectorXd nd(m_);
        nd.setZero();
        for (Eigen::SparseMatrix<double>::InnerIterator it(tmp, 0); it; ++it) {
            nd[it.row()] = it.value();
        }
        replace_column_impl_(j, nd);
    }

    // -------- Force full refactor (clears eta stack) --------
    void refactor() {
        if (A_is_sparse_)
            sparse_refactor_();
        else
            dense_refactor_();
    }

private:
    // ===== Dense backend =====
    void dense_refactor_() {
        Eigen::MatrixXd B(m_, m_);
        for (int k = 0; k < m_; ++k)
            B.col(k) = Bcols_dense_[k];
        lu_dense_.factor(B, opt_.pivot_rel, opt_.abs_floor);
        etas_.clear();
        update_count_ = 0;
    }
    Eigen::VectorXd dense_solve_(const Eigen::VectorXd &b) const {
        return lu_dense_.solve(b);
    }
    Eigen::VectorXd dense_solveT_(const Eigen::VectorXd &c) const {
        return lu_dense_.solveT(c);
    }

    // ===== Sparse backend =====
    void sparse_build_B_(Eigen::SparseMatrix<double> &B) const {
        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve((size_t)std::max(1, m_) * 8);
        for (int k = 0; k < m_; ++k) {
            const auto &col = Bcols_sparse_[k];
            for (SparseMat::InnerIterator it(col, 0); it; ++it) {
                trips.emplace_back(it.row(), k, it.value());
            }
        }
        B.resize(m_, m_);
        B.setFromTriplets(trips.begin(), trips.end());
        if (opt_.sparse_drop_tol > 0.0)
            B.prune(opt_.sparse_drop_tol);
        B.makeCompressed();
    }

    void sparse_factorize_(const Eigen::SparseMatrix<double> &B) {
        B_sparse_ = B;
        if (opt_.sparse_amd)
            solver_B_.analyzePattern(B_sparse_);
        solver_B_.factorize(B_sparse_);
        if (solver_B_.info() != Eigen::Success)
            throw std::runtime_error("SparseLU factorization failed for B");

        BT_sparse_ = B.transpose();
        if (opt_.sparse_amd)
            solver_BT_.analyzePattern(BT_sparse_);
        solver_BT_.factorize(BT_sparse_);
        if (solver_BT_.info() != Eigen::Success)
            throw std::runtime_error("SparseLU factorization failed for B^T");
    }

    void sparse_refactor_() {
        Eigen::SparseMatrix<double> B;
        sparse_build_B_(B);
        sparse_factorize_(B);
        etas_.clear();
        update_count_ = 0;
    }

    Eigen::VectorXd sparse_solve_(const Eigen::VectorXd &b) const {
        Eigen::VectorXd x = solver_B_.solve(b);
        if (solver_B_.info() != Eigen::Success)
            throw std::runtime_error("Sparse solve failed (B x = b)");
        return x;
    }
    Eigen::VectorXd sparse_solveT_(const Eigen::VectorXd &c) const {
        Eigen::VectorXd y = solver_BT_.solve(c);
        if (solver_BT_.info() != Eigen::Success)
            throw std::runtime_error("Sparse solve failed (B^T y = c)");
        return y;
    }

    // ===== FT product-form update path (dense only) =====
    // Minimal, safe implementation guarded by alpha/norm/limits.
    // If anything looks unstable, we fall back to a full refactor.
    void forrest_tomlin_update_dense_(int j,
                                      const Eigen::VectorXd& /*u*/,
                                      const Eigen::VectorXd& z,
                                      const Eigen::VectorXd& /*w*/,
                                      double alpha)
    {
        // We use a *local* update that keeps U upper and L unit-lower by:
        // 1) Adjusting U's column j (diagonal scaling by alpha)
        // 2) Moving subdiagonal parts of column j into L
        // 3) Chasing fill below the diagonal in col j with row ops
        // This is a compact/safe variant; if checks fail, we refactor().

        Eigen::MatrixXd& L = lu_dense_.L();
        Eigen::MatrixXd& U = lu_dense_.U();
        const int n = (int)U.rows();

        // Band limits (optional)
        const int band = opt_.ft_bandwidth_cap > 0 ? opt_.ft_bandwidth_cap : n;
        const int i_lo = std::max(0, j - band);
        const int i_hi = std::min(n - 1, j + band);

        // Guard small alpha
        if (!(std::isfinite(alpha)) || std::abs(alpha) < opt_.alpha_tol) {
            throw std::runtime_error("FT update: alpha too small");
        }

        // 1) Scale U(j,j) by alpha
        double old_piv = U(j, j);
        double new_piv = alpha * old_piv;
        if (!std::isfinite(new_piv) ||
            std::abs(new_piv) < std::max(opt_.abs_floor,
                                         10 * std::numeric_limits<double>::epsilon() * std::abs(old_piv)))
            throw std::runtime_error("FT update: unstable new pivot");
        U(j, j) = new_piv;

        // 2) Move subdiagonal contrib of column j into L via z(i)
        //    L(i,j) += z(i) for i>j (within band)
        for (int i = std::max(j + 1, i_lo); i <= i_hi; ++i) {
            double zi = z(i);
            if (zi != 0.0) {
                L(i, j) += zi;
            }
        }

        // 3) Chase the subdiagonal entries back to zero by subtracting multiples of row j of U
        //    For i>j: U(i, j:end) -= L(i,j) * U(j, j:end); then set L(i,j)=0
        const int row_tail = n - j;
        for (int i = std::max(j + 1, i_lo); i <= i_hi; ++i) {
            double lij = L(i, j);
            if (lij != 0.0) {
                U.row(i).segment(j, row_tail).noalias() -= lij * U.row(j).segment(j, row_tail);
                L(i, j) = 0.0;
            }
        }

        // L's diagonal must stay 1
        if (!std::isfinite(L(j, j)) || std::abs(L(j, j) - 1.0) > 1e-14)
            L(j, j) = 1.0;

        // Final pivot guard
        if (!std::isfinite(U(j, j)) || std::abs(U(j, j)) < opt_.abs_floor)
            throw std::runtime_error("FT update: pivot guard triggered");
    }

    // ===== FT product-form update framework (both paths) =====
    template <typename ColSetter>
    void replace_column_impl_core_(int j, const Eigen::VectorXd &new_col_dense,
                                   ColSetter &&set_col) {
        // Build old col and delta
        Eigen::VectorXd old(m_);
        old.setZero();
        if (A_is_sparse_) {
            const auto &scol = Bcols_sparse_[j];
            for (SparseMat::InnerIterator it(scol, 0); it; ++it)
                old[it.row()] = it.value();
        } else {
            old = Bcols_dense_[j];
        }
        Eigen::VectorXd u = new_col_dense - old;

        // Solve z = B^{-1} u  and w = B^{-T} e_j
        Eigen::VectorXd z = solve_B(u);
        Eigen::VectorXd ej = Eigen::VectorXd::Zero(m_); ej(j) = 1.0;
        Eigen::VectorXd w = solve_BT(ej);

        // alpha = 1 + z_j
        double alpha = 1.0 + z(j);

        // decide update path
        const bool try_ft =
            !A_is_sparse_ &&
            opt_.update_mode == Options::UpdateMode::ForrestTomlin &&
            std::abs(alpha) >= opt_.alpha_tol &&
            update_count_ < opt_.refactor_every &&
            z.cwiseAbs().maxCoeff() <= opt_.z_inf_guard;

        // Always keep cached column views in sync
        set_col(j, new_col_dense);

        if (try_ft) {
            // Attempt FT in-place; if anything fails -> refactor
            bool ok = true;
            try {
                forrest_tomlin_update_dense_(j, u, z, w, alpha);
            } catch (...) {
                ok = false;
            }
            if (!ok) {
                refactor();
                return;
            }
            // success
            ++update_count_;
            if (need_compress_())
                refactor();
            return;
        }

        // Fallback / sparse path: push eta and optionally compress
        // (keeps exact former behavior)
        const bool refactor_now = (std::abs(alpha) < opt_.alpha_tol) ||
                                  (update_count_ >= opt_.refactor_every);
        if (refactor_now) {
            refactor();
            return;
        }

        etas_.push_back(Eta{j, std::move(u), std::move(z), std::move(w), alpha});
        ++update_count_;
        if (need_compress_())
            refactor();
    }

    void replace_column_impl_(int j, const Eigen::VectorXd &new_col_dense) {
        if (A_is_sparse_) {
            auto set_col = [&](int col_j, const Eigen::VectorXd &dense) {
                std::vector<Eigen::Triplet<double>> tr;
                tr.reserve((size_t)std::min<int>((int)dense.size(), 16));
                for (int r = 0; r < dense.size(); ++r) {
                    double v = dense[r];
                    if (v != 0.0)
                        tr.emplace_back(r, 0, v);
                }
                SparseMat col(m_, 1);
                if (!tr.empty())
                    col.setFromTriplets(tr.begin(), tr.end());
                col.makeCompressed();
                Bcols_sparse_[col_j] = std::move(col);
            };
            replace_column_impl_core_(j, new_col_dense, set_col);
        } else {
            auto set_col = [&](int col_j, const Eigen::VectorXd &dense) {
                Bcols_dense_[col_j] = dense;
            };
            replace_column_impl_core_(j, new_col_dense, set_col);
        }
    }

    // Apply product-form etas after base solve
    Eigen::VectorXd apply_etas_solve_(Eigen::VectorXd x) const {
        for (const auto &eta : etas_) {
            double xj = x(eta.j);
            if (xj != 0.0)
                x.noalias() -= eta.z * (xj / eta.alpha);
        }
        return x;
    }
    Eigen::VectorXd apply_etas_solve_T_(Eigen::VectorXd y) const {
        for (const auto &eta : etas_) {
            double uy = eta.u.dot(y);
            if (uy != 0.0)
                y.noalias() -= eta.w * (uy / eta.alpha);
        }
        return y;
    }

    bool need_compress_() const noexcept {
        if ((int)etas_.size() >= opt_.compress_every)
            return true;
        double maxabsz = 0.0;
        for (const auto &e : etas_) {
            double v = e.z.cwiseAbs().maxCoeff();
            if (v > maxabsz)
                maxabsz = v;
        }
        if (maxabsz > opt_.z_inf_guard)
            return true;
        if (A_is_sparse_ && !etas_.empty() &&
            update_count_ > opt_.compress_every / 2)
            return true;
        return false;
    }

private:
    // ---- Members (order matters; keep in sync with init lists) ----
    // Dense storage
    std::reference_wrapper<const DenseMat> A_dense_;
    std::vector<Eigen::VectorXd> Bcols_dense_;
    MarkowitzLU lu_dense_;

    // Sparse storage
    std::reference_wrapper<const SparseMat> A_sparse_;
    std::vector<SparseMat> Bcols_sparse_;
    bool A_is_sparse_;

    // Problem size and options
    int m_;
    std::vector<int> basis_;
    Options opt_;
    std::vector<Eta> etas_;
    int update_count_;

    // Cached sparse factorizations (only used when A_is_sparse_)
    Eigen::SparseMatrix<double> B_sparse_, BT_sparse_;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver_B_, solver_BT_;
};
