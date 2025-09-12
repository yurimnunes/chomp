#pragma once

#include "presolver.h" // namespace presolve
#include "simplex_aux.h"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

// ==============================
// Public result container
// ==============================
struct LPSolution {
    enum class Status {
        Optimal,
        Unbounded,
        Infeasible,
        IterLimit,
        Singular,
        NeedPhase1
    };
    Status status{};
    Eigen::VectorXd x;
    double obj = std::numeric_limits<double>::quiet_NaN();
    std::vector<int> basis;
    int iters = 0;
    std::unordered_map<std::string, std::string> info;
};

inline const char *to_string(LPSolution::Status s) {
    switch (s) {
    case LPSolution::Status::Optimal:
        return "optimal";
    case LPSolution::Status::Unbounded:
        return "unbounded";
    case LPSolution::Status::Infeasible:
        return "infeasible";
    case LPSolution::Status::IterLimit:
        return "iterlimit";
    case LPSolution::Status::Singular:
        return "singular";
    case LPSolution::Status::NeedPhase1:
        return "need_phase1";
    }
    return "unknown";
}

// ==============================
// Options
// ==============================
struct RevisedSimplexOptions {
    int max_iters = 50'000;
    double tol = 1e-9;
    bool bland = false;
    double svd_tol = 1e-8;
    double ratio_delta = 1e-12;
    double ratio_eta = 1e-7;
    double deg_step_tol = 1e-12;
    double epsilon_cost = 1e-10;
    int rng_seed = 13;

    // Basis / LU
    int refactor_every = 128; // FT hard cap
    int compress_every = 64;  // FT soft cap
    double lu_pivot_rel = 1e-12;
    double lu_abs_floor = 1e-16;
    double alpha_tol = 1e-10;
    double z_inf_guard = 1e6;

    // Pricing
    int devex_reset = 200;
    std::string pricing_rule = "adaptive"; // or "devex"
    int steepest_edge_reset_freq = 1000;

    int max_basis_rebuilds = 3;
};

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

        Eigen::VectorXd Pb = apply_Pr_(b);
        Eigen::VectorXd z = forward_sub_(L_, Pb);
        Eigen::VectorXd w = back_sub_(U_, z);
        Eigen::VectorXd x = apply_Pc_(w);

        // Enhanced iterative refinement
        const int max_refinements = 3;
        const double refinement_tol = 1e-14;

        for (int iter = 0; iter < max_refinements; ++iter) {
            // Compute residual r = Pb - L*U*w (in permuted space)
            Eigen::VectorXd LUw = L_ * (U_ * w);
            Eigen::VectorXd r = Pb - LUw;

            double residual_norm = r.lpNorm<Eigen::Infinity>();
            if (residual_norm < refinement_tol)
                break;

            // Solve L*U*dw = r for correction
            Eigen::VectorXd dz = forward_sub_(L_, r);
            Eigen::VectorXd dw = back_sub_(U_, dz);

            // Apply correction
            w += dw;

            // Check for numerical breakdown
            if (!dw.array().isFinite().all() ||
                dw.lpNorm<Eigen::Infinity>() < 1e-16) {
                break;
            }
        }

        return apply_Pc_(w);
    }
    Eigen::VectorXd solveT(const Eigen::VectorXd &c) const {
        if (c.size() != n_)
            throw std::invalid_argument("MarkowitzLU::solveT size mismatch");

        Eigen::VectorXd PcTc = apply_PcT_(c);
        Eigen::VectorXd t = forward_sub_(U_.transpose(), PcTc);
        Eigen::VectorXd s = back_sub_(L_.transpose(), t);

        // Enhanced iterative refinement for transpose solve
        const int max_refinements = 3;
        const double refinement_tol = 1e-14;

        for (int iter = 0; iter < max_refinements; ++iter) {
            // Compute residual r = PcTc - U^T * L^T * s (in permuted space)
            Eigen::VectorXd UTLTs = U_.transpose() * (L_.transpose() * s);
            Eigen::VectorXd r = PcTc - UTLTs;

            double residual_norm = r.lpNorm<Eigen::Infinity>();
            if (residual_norm < refinement_tol)
                break;

            // Solve U^T * L^T * ds = r for correction
            Eigen::VectorXd dt = forward_sub_(U_.transpose(), r);
            Eigen::VectorXd ds = back_sub_(L_.transpose(), dt);

            // Apply correction
            s += ds;

            // Check for numerical breakdown
            if (!ds.array().isFinite().all() ||
                ds.lpNorm<Eigen::Infinity>() < 1e-16) {
                break;
            }
        }

        return apply_PrT_inv_(s);
    }

    int n() const { return n_; }

private:
    static constexpr double kSingFloor_ = 1e-18;

    static Eigen::VectorXd forward_sub_(const Eigen::MatrixXd &L,
                                        const Eigen::VectorXd &b) {
        const int n = static_cast<int>(L.rows());
        Eigen::VectorXd x = b;
        for (int i = 0; i < n; ++i) {
            double s = L.row(i).head(i).dot(x.head(i));
            double piv = L(i, i);
            if (std::abs(piv) < kSingFloor_)
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
            if (std::abs(piv) < kSingFloor_)
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

        // rook pivoting
        int i_idx;
        U_.col(k).segment(k, n_ - k).cwiseAbs().maxCoeff(&i_idx);
        int i = k + i_idx;
        int j_idx;
        U_.row(i).segment(k, n_ - k).cwiseAbs().maxCoeff(&j_idx);
        int j = k + j_idx;
        for (int t = 0; t < rook_iters_; ++t) {
            U_.col(j).segment(k, n_ - k).cwiseAbs().maxCoeff(&i_idx);
            i = k + i_idx;
            U_.row(i).segment(k, n_ - k).cwiseAbs().maxCoeff(&j_idx);
            j = k + j_idx;
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
                if (std::abs(U_(pi, pj)) <
                    std::max(abs_floor_,
                             10 * std::numeric_limits<double>::epsilon() *
                                 inf_norm)) {
                    throw std::runtime_error(
                        "MarkowitzLU: singular matrix (no acceptable pivot)");
                }
            }
            swap_rows_(k, pi);
            swap_cols_(k, pj);
            double piv = U_(k, k);
            if (std::abs(piv) <
                std::max(abs_floor_,
                         10 * std::numeric_limits<double>::epsilon() *
                             inf_norm)) {
                throw std::runtime_error(
                    "MarkowitzLU: numerically singular pivot");
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
//   - Dense: MarkowitzLU
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
                          /*sparse_amd=*/true, /*sparse_drop_tol=*/0.0}) {}
    // -------- Basic info --------
    int rows() const { return m_; }
    const std::vector<int> &basis() const { return basis_; }
    const std::vector<Eta> &etas() const { return etas_; }

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

    // Accepts any sparse column expression (A.col(k), SparseVector, etc.)
    template <typename Derived>
    void
    replace_column(int j,
                   const Eigen::SparseMatrixBase<Derived> &new_col_sparse) {
        Eigen::SparseMatrix<double> tmp =
            new_col_sparse.derived().eval(); // m x 1
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
        // Reserve a heuristic amount to avoid many reallocs
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

    // ===== FT product-form update path =====
    template <typename ColSetter>
    void replace_column_impl_core_(int j, const Eigen::VectorXd &new_col_dense,
                                   ColSetter &&set_col) {
        // u = new_col - old_col
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

        // z = B^{-1} u
        Eigen::VectorXd z = solve_B(u);

        // alpha and refactor guard
        double alpha = 1.0 + z(j);
        bool refactor_now = (std::abs(alpha) < opt_.alpha_tol) ||
                            (update_count_ >= opt_.refactor_every);

        // w = B^{-T} e_j
        Eigen::VectorXd ej = Eigen::VectorXd::Zero(m_);
        ej(j) = 1.0;
        Eigen::VectorXd w = solve_BT(ej);

        if (refactor_now) {
            set_col(j, new_col_dense);
            refactor();
            return;
        }

        // Push eta (lazy product-form)
        etas_.push_back(
            Eta{j, std::move(u), std::move(z), std::move(w), alpha});
        set_col(j, new_col_dense);
        ++update_count_;
        if (need_compress_())
            refactor();
    }

    void replace_column_impl_(int j, const Eigen::VectorXd &new_col_dense) {
        if (A_is_sparse_) {
            auto set_col = [&](int col_j, const Eigen::VectorXd &dense) {
                std::vector<Eigen::Triplet<double>> tr;
                tr.reserve((size_t)std::min<int>(dense.size(), 16));
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

    bool need_compress_() const {
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
        // crude heuristic for sparse drift
        if (A_is_sparse_ && etas_.size() > 0 &&
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

// ==============================
// RevisedSimplex
// ==============================
class RevisedSimplex {
public:
    explicit RevisedSimplex(RevisedSimplexOptions opt = {})
        : opt_(std::move(opt)), rng_(opt_.rng_seed), degen_(opt_.rng_seed) {}

    LPSolution solve(const Eigen::MatrixXd &A_in, const Eigen::VectorXd &b_in,
                     const Eigen::VectorXd &c_in,
                     std::optional<std::vector<int>> basis_opt = std::nullopt) {
        const int n = static_cast<int>(A_in.cols());

        // ---- 0) Build presolve LP: Ax=b, default bounds, costs = c_in ----
        presolve::LP lp;
        lp.A = A_in;
        lp.b = b_in;
        lp.sense.assign(static_cast<int>(A_in.rows()), presolve::RowSense::EQ);
        lp.c = c_in;
        lp.l = Eigen::VectorXd::Zero(n);
        lp.u = Eigen::VectorXd::Constant(n, presolve::inf());
        lp.c0 = 0.0;

        // ---- 1) Presolve ----
        presolve::Presolver::Options popt;
        popt.enable_rowreduce = true;
        popt.enable_scaling = true;
        popt.max_passes = 10;
        if (A_in.cols() > static_cast<int>(A_in.rows() * 1.2)) {
            popt.conservative_mode = true;
        }

        presolve::Presolver P(popt);
        const auto pres = P.run(lp);

        if (pres.proven_infeasible) {
            return make_solution(LPSolution::Status::Infeasible,
                                 Eigen::VectorXd::Zero(n),
                                 std::numeric_limits<double>::infinity(), {}, 0,
                                 {{"presolve", "infeasible"}});
        }
        if (pres.proven_unbounded) {
            auto xnan = Eigen::VectorXd::Constant(
                n, std::numeric_limits<double>::quiet_NaN());
            return make_solution(LPSolution::Status::Unbounded, xnan,
                                 -std::numeric_limits<double>::infinity(), {},
                                 0, {{"presolve", "unbounded"}});
        }

        const Eigen::MatrixXd &Atil = pres.reduced.A;
        const Eigen::VectorXd &btil = pres.reduced.b;
        const Eigen::VectorXd &ctil = pres.reduced.c;
        const Eigen::VectorXd &lred = pres.reduced.l;
        const Eigen::VectorXd &ured = pres.reduced.u;

        // ---- 2) m==0 fast path: optimize over bounds only ----
        if (Atil.rows() == 0) {
            Eigen::VectorXd vred =
                Eigen::VectorXd::Zero(static_cast<int>(ctil.size()));
            bool is_bounded = true;
            for (int j = 0; j < static_cast<int>(ctil.size()); ++j) {
                if (ctil(j) > opt_.tol) {
                    vred(j) = std::isfinite(lred(j)) ? lred(j) : 0.0;
                } else if (ctil(j) < -opt_.tol) {
                    if (std::isfinite(ured(j)))
                        vred(j) = ured(j);
                    else {
                        is_bounded = false;
                        break;
                    }
                } else {
                    vred(j) = std::isfinite(lred(j)) ? lred(j) : 0.0;
                }
            }
            if (!is_bounded) {
                auto xnan = Eigen::VectorXd::Constant(
                    n, std::numeric_limits<double>::quiet_NaN());
                return make_solution(
                    LPSolution::Status::Unbounded, xnan,
                    -std::numeric_limits<double>::infinity(), {}, 0,
                    {{"presolve", "m=0 neg cost & +inf upper"}});
            }
            auto [x_full, obj_corr] = P.postsolve(vred);
            double total_obj = c_in.dot(x_full) + obj_corr;
            return make_solution(LPSolution::Status::Optimal, x_full, total_obj,
                                 {}, 0,
                                 {{"presolve", "m=0 optimized over bounds"}});
        }

        // ---- 3) Augment with reasonable upper-bounds (extra rows/cols) ----
        auto build_with_upper_bounds = [&](const Eigen::MatrixXd &A0,
                                           const Eigen::VectorXd &b0,
                                           const Eigen::VectorXd &c0)
            -> std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd,
                          std::vector<int>, std::vector<int>> {
            const int m0 = static_cast<int>(A0.rows());
            const int n0 = static_cast<int>(A0.cols());

            std::vector<int> ub_idx;
            ub_idx.reserve(n0);
            const double max_reasonable_bound = 1e6;
            for (int j = 0; j < n0; ++j)
                if (std::isfinite(ured(j)) && ured(j) <= max_reasonable_bound)
                    ub_idx.push_back(j);

            const int p = static_cast<int>(ub_idx.size());
            if (p == 0) {
                std::vector<int> col_orig_map(n0);
                for (int j = 0; j < n0; ++j)
                    col_orig_map[j] = pres.orig_col_index[j];
                std::vector<int> row_orig_map(m0);
                std::iota(row_orig_map.begin(), row_orig_map.end(), 0);
                return {A0, b0, c0, col_orig_map, row_orig_map};
            }

            Eigen::MatrixXd A(m0 + p, n0 + p);
            A.setZero();
            A.topLeftCorner(m0, n0) = A0;

            Eigen::VectorXd b(m0 + p);
            b.head(m0) = b0;

            Eigen::VectorXd c(n0 + p);
            c.head(n0) = c0;
            c.tail(p).setZero();

            for (int k = 0; k < p; ++k) {
                const int j = ub_idx[k];
                const int row = m0 + k;
                const int slack_col = n0 + k;
                A(row, j) = 1.0;
                A(row, slack_col) = 1.0;
                b(row) = ured(j);
            }

            std::vector<int> col_orig_map(n0 + p, -1);
            for (int j = 0; j < n0; ++j)
                col_orig_map[j] = pres.orig_col_index[j];
            std::vector<int> row_orig_map(m0 + p);
            std::iota(row_orig_map.begin(), row_orig_map.end(), 0);
            return {A, b, c, col_orig_map, row_orig_map};
        };

        Eigen::MatrixXd Ared = Atil;
        Eigen::VectorXd bred = btil;
        Eigen::VectorXd cred = ctil;
        std::vector<int> col_orig_map, row_orig_map;
        std::tie(Ared, bred, cred, col_orig_map, row_orig_map) =
            build_with_upper_bounds(Ared, bred, cred);

        const int m_eff = static_cast<int>(Ared.rows());
        const int n_eff = static_cast<int>(Ared.cols());

        // --- Effective bounds for reduced problem ---
        Eigen::VectorXd l_eff = Eigen::VectorXd::Zero(n_eff);
        Eigen::VectorXd u_eff =
            Eigen::VectorXd::Constant(n_eff, presolve::inf());
        for (int jr = 0; jr < n_eff; ++jr) {
            int jorig = col_orig_map[jr];
            if (jorig >= 0) {
                l_eff(jr) = lred(jorig);
                u_eff(jr) = ured(jorig);
            }
        }

        // ---- 4) Map incoming basis to reduced space (optional) ----
        std::optional<std::vector<int>> red_basis_opt = std::nullopt;
        if (basis_opt && !basis_opt->empty()) {
            std::unordered_map<int, int> orig2red;
            orig2red.reserve(n_eff);
            for (int jr = 0; jr < n_eff; ++jr) {
                int jorig = col_orig_map[jr];
                if (jorig >= 0)
                    orig2red[jorig] = jr;
            }
            std::vector<int> cand;
            cand.reserve(std::min(m_eff, (int)basis_opt->size()));
            for (int jorig : *basis_opt) {
                auto it = orig2red.find(jorig);
                if (it != orig2red.end()) {
                    cand.push_back(it->second);
                    if ((int)cand.size() == m_eff)
                        break;
                }
            }
            if ((int)cand.size() == m_eff)
                red_basis_opt = std::move(cand);
        }

        // ---- 5) Try Phase II directly on reduced problem ----
        std::vector<int> basis_guess;
        if (red_basis_opt && (int)red_basis_opt->size() == m_eff) {
            basis_guess = *red_basis_opt;
        } else {
            auto maybe = find_initial_basis(Ared, bred);
            if (maybe)
                basis_guess = *maybe;
        }

        auto add_info = [&](std::unordered_map<std::string, std::string> info) {
            info["presolve_actions"] = std::to_string(pres.stack.size());
            info["reduced_m"] = std::to_string(m_eff);
            info["reduced_n"] = std::to_string(n_eff);
            info["obj_shift"] = std::to_string(pres.obj_shift);
            return info;
        };

        if ((int)basis_guess.size() == m_eff) {
            auto [st, v2, red_basis2, it2, info2] =
                phase(Ared, bred, cred, basis_guess, l_eff, u_eff);
            if (st == LPSolution::Status::Optimal ||
                st == LPSolution::Status::Unbounded ||
                st == LPSolution::Status::IterLimit) {

                auto [x_full, obj_correction] = P.postsolve(v2);
                double total_obj = c_in.dot(x_full) + obj_correction;

                std::vector<int> basis_full;
                basis_full.reserve(red_basis2.size());
                for (int jr : red_basis2) {
                    if (jr >= 0 && jr < (int)col_orig_map.size()) {
                        int jorig = col_orig_map[jr];
                        if (jorig >= 0)
                            basis_full.push_back(jorig);
                    }
                }
                auto info = add_info(std::move(info2));
                return make_solution(st, x_full, total_obj, basis_full, it2,
                                     std::move(info));
            }
            if (st == LPSolution::Status::Singular) {
                auto info = add_info({});
                return make_solution(LPSolution::Status::Singular,
                                     Eigen::VectorXd::Zero(n),
                                     std::numeric_limits<double>::quiet_NaN(),
                                     {}, 0, std::move(info));
            }
        }

        // ---- 6) Phase I on reduced problem ----
        auto [A1, b1, c1, basis1, n_orig_eff, m_rows] = make_phase1(Ared, bred);
        auto [status1, v1, basis1_out, it1, info1] =
            phase(A1, b1, c1, basis1,
                  /*l=*/Eigen::VectorXd::Zero(A1.cols()),
                  /*u=*/Eigen::VectorXd::Constant(A1.cols(), presolve::inf()));

        if (status1 != LPSolution::Status::Optimal || c1.dot(v1) > opt_.tol) {
            auto info = add_info({{"phase1_status", to_string(status1)}});
            auto e = degen_.get_statistics();
            info.insert(e.begin(), e.end());
            return make_solution(LPSolution::Status::Infeasible,
                                 Eigen::VectorXd::Zero(n),
                                 std::numeric_limits<double>::infinity(), {},
                                 it1, std::move(info));
        }

        // Warm-start Phase II
        std::vector<int> red_basis2;
        red_basis2.reserve(m_rows);
        for (int j : basis1_out)
            if (j < n_orig_eff)
                red_basis2.push_back(j);

        // Basis completion if needed
        if ((int)red_basis2.size() < m_rows) {
            for (int j = 0; j < n_orig_eff; ++j) {
                if ((int)red_basis2.size() == m_rows)
                    break;
                if (std::find(red_basis2.begin(), red_basis2.end(), j) !=
                    red_basis2.end())
                    continue;
                std::vector<int> cand = red_basis2;
                cand.push_back(j);
                if ((int)cand.size() > m_rows)
                    continue;
                Eigen::MatrixXd B =
                    Ared(Eigen::all,
                         Eigen::VectorXi::Map(cand.data(), (int)cand.size()));
                Eigen::FullPivLU<Eigen::MatrixXd> lu(B);
                if (lu.rank() == (int)cand.size() && lu.isInvertible())
                    red_basis2 = std::move(cand);
            }
        }

        LPSolution::Status status2;
        Eigen::VectorXd v2;
        std::vector<int> red_basis_out;
        int it2 = 0;
        std::unordered_map<std::string, std::string> info2;

        if ((int)red_basis2.size() == m_rows) {
            std::tie(status2, v2, red_basis_out, it2, info2) =
                phase(Ared, bred, cred, red_basis2, l_eff, u_eff);
        } else {
            std::tie(status2, v2, red_basis_out, it2, info2) =
                phase(Ared, bred, cred, std::nullopt, l_eff, u_eff);
            if (status2 == LPSolution::Status::NeedPhase1) {
                status2 = LPSolution::Status::Singular;
                info2["note"] = "reduced matrix cannot form a proper basis";
            }
        }

        int total_iters = it1 + it2;
        auto merged_info = add_info(std::move(info2));
        merged_info.insert({"phase1_iters", std::to_string(it1)});

        auto [x_full, obj_correction] = P.postsolve(v2);
        double total_obj = c_in.dot(x_full) + obj_correction;

        std::vector<int> basis_full;
        basis_full.reserve(red_basis_out.size());
        for (int jr : red_basis_out) {
            if (jr >= 0 && jr < (int)col_orig_map.size()) {
                int jorig = col_orig_map[jr];
                if (jorig >= 0)
                    basis_full.push_back(jorig);
            }
        }

        if (status2 == LPSolution::Status::Optimal) {
            return make_solution(LPSolution::Status::Optimal, x_full, total_obj,
                                 basis_full, total_iters,
                                 std::move(merged_info));
        }
        if (status2 == LPSolution::Status::Unbounded) {
            return make_solution(LPSolution::Status::Unbounded, x_full,
                                 -std::numeric_limits<double>::infinity(),
                                 basis_full, total_iters,
                                 std::move(merged_info));
        }

        double obj_fallback = x_full.array().isFinite().all()
                                  ? total_obj
                                  : std::numeric_limits<double>::quiet_NaN();
        return make_solution(status2, x_full, obj_fallback, basis_full,
                             total_iters, std::move(merged_info));
    }

private:
    // ---------- helpers ----------
    static Eigen::VectorXd clip_small(Eigen::VectorXd x, double tol = 1e-12) {
        for (int i = 0; i < x.size(); ++i)
            if (std::abs(x(i)) < tol)
                x(i) = 0.0;
        return x;
    }

    static std::optional<std::vector<int>>
    find_initial_basis(const Eigen::MatrixXd &A, const Eigen::VectorXd &b) {
        const int m = static_cast<int>(A.rows());
        const int n = static_cast<int>(A.cols());
        std::vector<int> basis;
        basis.reserve(m);
        std::vector<bool> used_row(m, false);

        for (int j = 0; j < n; ++j) {
            Eigen::VectorXd col = A.col(j);
            int one_idx = -1, ones = 0;
            bool zeros_ok = true;
            for (int i = 0; i < m; ++i) {
                double v = col(i);
                if (std::abs(v - 1.0) <= 1e-12) {
                    one_idx = i;
                    ++ones;
                } else if (std::abs(v) > 1e-12) {
                    zeros_ok = false;
                    break;
                }
            }
            if (ones == 1 && zeros_ok && !used_row[one_idx] &&
                b(one_idx) >= -1e-12) {
                basis.push_back(j);
                used_row[one_idx] = true;
                if ((int)basis.size() == m)
                    break;
            }
        }
        if ((int)basis.size() == m)
            return basis;
        return std::nullopt;
    }

    static std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd,
                      std::vector<int>, int, int>
    make_phase1(const Eigen::MatrixXd &A, const Eigen::VectorXd &b) {
        const int m = static_cast<int>(A.rows());
        const int n = static_cast<int>(A.cols());
        Eigen::MatrixXd A1 = A;
        Eigen::VectorXd b1 = b;
        for (int i = 0; i < m; ++i)
            if (b1(i) < 0) {
                A1.row(i) *= -1.0;
                b1(i) *= -1.0;
            }

        Eigen::MatrixXd A_aux(m, n + m);
        A_aux.leftCols(n) = A1;
        A_aux.rightCols(m) = Eigen::MatrixXd::Identity(m, m);
        Eigen::VectorXd c_aux(n + m);
        c_aux.setZero();
        c_aux.tail(m).setOnes();
        std::vector<int> basis(m);
        std::iota(basis.begin(), basis.end(), n);
        return {A_aux, b1, c_aux, basis, n, m};
    }

    // Harris ratio (improved) — returns (leaving_row, theta_B)
    std::pair<std::optional<int>, double>
    harris_ratio(const Eigen::VectorXd &xB, const Eigen::VectorXd &dB,
                 double delta, double eta) const {
        std::vector<int> pos;
        pos.reserve(dB.size());
        for (int i = 0; i < dB.size(); ++i)
            if (dB(i) > delta)
                pos.push_back(i);
        if (pos.empty())
            return {std::nullopt, std::numeric_limits<double>::infinity()};

        double theta_star = std::numeric_limits<double>::infinity();
        for (int idx : pos)
            theta_star = std::min(theta_star, xB(idx) / dB(idx));

        double max_resid = 0.0;
        std::vector<int> candidates;
        for (int idx : pos) {
            double ratio = xB(idx) / dB(idx);
            if (std::abs(ratio - theta_star) <= 1e-10)
                candidates.push_back(idx);
            double resid = xB(idx) - theta_star * dB(idx);
            max_resid = std::max(max_resid, std::max(0.0, resid));
        }
        if (!candidates.empty()) {
            int best = candidates.front();
            for (int idx : candidates)
                if (idx < best)
                    best = idx;
            return {best, theta_star};
        }

        double kappa = std::max(eta, eta * max_resid);
        std::vector<int> eligible;
        for (int idx : pos) {
            double resid = xB(idx) - theta_star * dB(idx);
            if (resid <= kappa)
                eligible.push_back(idx);
        }
        if (!eligible.empty()) {
            int best = eligible.front();
            for (int idx : eligible)
                if (idx < best)
                    best = idx;
            return {best, theta_star};
        }

        int best = pos.front();
        double best_ratio = xB(best) / dB(best);
        for (int i = 1; i < (int)pos.size(); ++i) {
            int idx = pos[i];
            double r = xB(idx) / dB(idx);
            if (r < best_ratio) {
                best_ratio = r;
                best = idx;
            }
        }
        return {best, best_ratio};
    }

    // --- BFRT helper: permissible step to nearest active bound for entering
    // var
    struct BFRTStep {
        double theta_e = std::numeric_limits<double>::infinity();
        bool to_upper = false;
    };
    BFRTStep entering_bound_step(double x_e, double l_e, double u_e,
                                 double rc_e, double tol) const {
        BFRTStep out;
        // Primal simplex (minimization): rc_e < 0 ⇒ objective improves as x_e
        // increases
        if (rc_e < -tol) {
            if (std::isfinite(u_e)) {
                out.theta_e = std::max(0.0, u_e - x_e);
                out.to_upper = true;
            }
        } else if (rc_e > tol) { // move downwards
            if (std::isfinite(l_e)) {
                out.theta_e = std::max(0.0, x_e - l_e);
                out.to_upper = false;
            }
        }
        return out;
    }

    // ---------- phase (bound-aware via BFRT) ----------
    std::tuple<LPSolution::Status, Eigen::VectorXd, std::vector<int>, int,
               std::unordered_map<std::string, std::string>>
    phase(const Eigen::MatrixXd &A, const Eigen::VectorXd &b,
          const Eigen::VectorXd &c, std::optional<std::vector<int>> basis_opt,
          const Eigen::VectorXd &l, const Eigen::VectorXd &u) {
        const int m = static_cast<int>(A.rows());
        const int n = static_cast<int>(A.cols());
        int iters = 0;

        std::vector<int> basis;
        if (basis_opt) {
            basis = *basis_opt;
            if ((int)basis.size() != m)
                return {LPSolution::Status::NeedPhase1,
                        Eigen::VectorXd::Zero(n),
                        {},
                        0,
                        {{"reason", "basis size != m"}}};
        } else {
            auto maybe = find_initial_basis(A, b);
            if (!maybe) {
                return {LPSolution::Status::NeedPhase1,
                        Eigen::VectorXd::Zero(n),
                        {},
                        0,
                        {{"reason", "no_trivial_basis"}}};
            }
            basis = *maybe;
        }

        std::vector<int> N;
        N.reserve(n - m);
        {
            std::vector<char> inB(n, 0);
            for (int j : basis) {
                if (j < 0 || j >= n)
                    return {LPSolution::Status::Singular,
                            Eigen::VectorXd::Zero(n),
                            basis,
                            0,
                            {{"where", "initial basis index out of range"}}};
                inB[j] = 1;
            }
            for (int j = 0; j < n; ++j)
                if (!inB[j])
                    N.push_back(j);
        }

        FTBasis B(A, basis, opt_.refactor_every, opt_.compress_every,
                  opt_.lu_pivot_rel, opt_.lu_abs_floor, opt_.alpha_tol,
                  opt_.z_inf_guard);

        AdaptivePricer::PricingOptions pricing_opts;
        pricing_opts.steepest_pool_max = 0;
        pricing_opts.steepest_reset_freq = opt_.steepest_edge_reset_freq;
        adaptive_pricer_ = AdaptivePricer(n, pricing_opts);
        if (opt_.pricing_rule == "adaptive") {
            adaptive_pricer_.build_pools(B, A, N);
        }

        int rebuild_attempts = 0;

        while (iters < opt_.max_iters) {
            ++iters;

            // xB = B^{-1} b
            Eigen::VectorXd xB;
            try {
                xB = B.solve_B(b);
            } catch (...) {
                if (rebuild_attempts < opt_.max_basis_rebuilds) {
                    ++rebuild_attempts;
                    B.refactor();
                    if (opt_.pricing_rule == "adaptive") {
                        adaptive_pricer_.build_pools(B, A, N);
                        adaptive_pricer_.clear_rebuild_flag();
                    }
                    continue;
                }
                return {LPSolution::Status::Singular,
                        Eigen::VectorXd::Zero(n),
                        basis,
                        iters,
                        {{"where", "solve(B,b) repair failed"}}};
            }

            if ((xB.array() < -opt_.tol).any()) {
                return {LPSolution::Status::NeedPhase1,
                        Eigen::VectorXd::Zero(n),
                        basis,
                        iters,
                        {{"reason", "negative_basic_vars"}}};
            }
            xB = xB.cwiseMax(0.0);

            // y = B^{-T} c_B
            Eigen::VectorXd cB(m);
            for (int i = 0; i < m; ++i)
                cB(i) = c(basis[i]);
            Eigen::VectorXd y;
            try {
                y = B.solve_BT(cB);
            } catch (...) {
                B.refactor();
                y = B.solve_BT(cB);
                if (opt_.pricing_rule == "steepest_edge") {
                    adaptive_pricer_.build_pools(B, A, N);
                    adaptive_pricer_.clear_rebuild_flag();
                }
            }

            // reduced costs on nonbasics
            Eigen::VectorXd rN(N.size());
            for (int k = 0; k < (int)N.size(); ++k) {
                int j = N[k];
                rN(k) = c(j) - A.col(j).dot(y);
            }

            // pricing ⇒ choose entering rel index e_rel
            std::optional<int> e_rel;
            if (opt_.bland) {
                int idx = -1;
                for (int k = 0; k < (int)N.size(); ++k)
                    if (rN(k) < -opt_.tol) {
                        idx = k;
                        break;
                    }
                if (idx == -1) {
                    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
                    for (int i = 0; i < m; ++i)
                        x(basis[i]) = xB(i);
                    return {LPSolution::Status::Optimal, clip_small(x), basis,
                            iters, degen_.get_statistics()};
                }
                e_rel = idx;
            } else {
                if (opt_.pricing_rule == "adaptive") {
                    // Build current solution
                    Eigen::VectorXd x_current = Eigen::VectorXd::Zero(n);
                    for (int i = 0; i < m; ++i)
                        x_current(basis[i]) = xB(i);
                    double current_obj = c.dot(x_current);

                    e_rel = adaptive_pricer_.choose_entering(
                        rN, N, opt_.tol, iters, current_obj, B, A);
                } else {
                    int idx = -1;
                    double best = 0.0;
                    for (int k = 0; k < (int)N.size(); ++k)
                        if (rN(k) < -opt_.tol) {
                            if (idx < 0 || rN(k) < best) {
                                best = rN(k);
                                idx = k;
                            }
                        }
                    if (idx >= 0)
                        e_rel = idx;
                }
                if (!e_rel) {
                    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
                    for (int i = 0; i < m; ++i)
                        x(basis[i]) = xB(i);
                    return {LPSolution::Status::Optimal, clip_small(x), basis,
                            iters, degen_.get_statistics()};
                }
            }

            const int e = N[*e_rel];
            const Eigen::VectorXd a_e = A.col(e);

            // s = dB = B^{-1} a_e
            Eigen::VectorXd dB;
            try {
                dB = B.solve_B(a_e);
            } catch (...) {
                B.refactor();
                dB = B.solve_B(a_e);
                if (opt_.pricing_rule == "steepest_edge") {
                    adaptive_pricer_.build_pools(B, A, N);
                    adaptive_pricer_.clear_rebuild_flag();
                }
            }

            // Harris (leaving step from basics)
            auto [leave_rel_opt, theta_B] =
                harris_ratio(xB, dB, opt_.ratio_delta, opt_.ratio_eta);

            // BFRT step for entering variable (respect its bounds)
            const int rN_idx = *e_rel;
            const double rc_e = rN(rN_idx);
            const double l_e = (e >= 0 && e < l.size()) ? l(e) : 0.0;
            const double u_e =
                (e >= 0 && e < u.size()) ? u(e) : presolve::inf();
            const double x_e =
                std::isfinite(l_e) ? l_e : 0.0; // nonbasic sits at bound
            BFRTStep bfrt = entering_bound_step(x_e, l_e, u_e, rc_e, opt_.tol);

            double step = std::min(theta_B, bfrt.theta_e);

            if (!std::isfinite(step)) {
                Eigen::VectorXd x = Eigen::VectorXd::Constant(
                    n, std::numeric_limits<double>::quiet_NaN());
                return {LPSolution::Status::Unbounded, x, basis, iters,
                        degen_.get_statistics()};
            }

            // If BFRT wins strictly, flip the entering direction locally:
            const bool flip_entering = (bfrt.theta_e + 1e-14 < theta_B);
            if (flip_entering) {
                dB = -dB;
                const_cast<Eigen::VectorXd &>(rN)(rN_idx) = -rc_e;
            }

            if (!leave_rel_opt) {
                Eigen::VectorXd x = Eigen::VectorXd::Constant(
                    n, std::numeric_limits<double>::quiet_NaN());
                return {LPSolution::Status::Unbounded, x, basis, iters,
                        degen_.get_statistics()};
            }

            const int r = *leave_rel_opt;
            const double alpha = dB(r);
            const int old_abs = basis[r];
            const int e_abs = e;

            // Degeneracy hooks
            bool is_degenerate =
                degen_.detect_degeneracy(step, opt_.deg_step_tol);
            if (is_degenerate && degen_.should_apply_perturbation()) {
                auto [Ap, bp, cp] =
                    degen_.apply_perturbation(A, b, c, basis, iters);
                (void)Ap;
                (void)bp;
                (void)cp;
            } else {
                (void)degen_.reset_perturbation();
            }

            // Update pricer state
            if (opt_.pricing_rule == "steepest_edge") {
                adaptive_pricer_.update_after_pivot(r, e_abs, old_abs, dB,
                                                    alpha, step, A, N);
            }

            // Pivot in indices
            basis[r] = e_abs;
            N[*e_rel] = old_abs;

            // Update basis matrix col
            try {
                B.replace_column(r, a_e);
            } catch (...) {
                B.refactor();
                if (opt_.pricing_rule == "steepest_edge") {
                    adaptive_pricer_.build_pools(B, A, N);
                    adaptive_pricer_.clear_rebuild_flag();
                }
            }

            if (opt_.pricing_rule == "steepest_edge" &&
                adaptive_pricer_.needs_rebuild()) {
                adaptive_pricer_.build_pools(B, A, N);
                adaptive_pricer_.clear_rebuild_flag();
            }
        }

        return {LPSolution::Status::IterLimit, Eigen::VectorXd::Zero(n), basis,
                iters, degen_.get_statistics()};
    }

    static LPSolution
    make_solution(LPSolution::Status st, Eigen::VectorXd x, double obj,
                  std::vector<int> basis, int iters,
                  std::unordered_map<std::string, std::string> info) {
        LPSolution sol;
        sol.status = st;
        sol.x = std::move(x);
        sol.obj = obj;
        sol.basis = std::move(basis);
        sol.iters = iters;
        sol.info = std::move(info);
        return sol;
    }

private:
    RevisedSimplexOptions opt_;
    std::mt19937 rng_;
    DegeneracyManager degen_;
    AdaptivePricer adaptive_pricer_{1}; // Add this - initialize with dummy size
};
