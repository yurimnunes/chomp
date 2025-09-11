#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>
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
#include "presolver.h"  // namespace presolve

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
    std::string pricing_rule = "steepest_edge"; // or "devex"
    int steepest_edge_reset_freq = 1000;

    int max_basis_rebuilds = 3;
};

// ==============================
// Degeneracy hooks (no-op)
// ==============================
class DegeneracyManager {
public:
    explicit DegeneracyManager(int rng_seed = 13)
        : rng_(rng_seed), consecutive_degenerate_(0),
          perturbation_active_(false) {}

    bool detect_degeneracy(double step, double deg_step_tol) {
        bool is_degenerate = (step <= deg_step_tol);
        if (is_degenerate) {
            ++consecutive_degenerate_;
        } else {
            consecutive_degenerate_ = 0;
        }
        return is_degenerate;
    }

    bool should_apply_perturbation() const {
        return consecutive_degenerate_ > 10 && !perturbation_active_;
    }

    std::tuple<std::optional<Eigen::MatrixXd>, std::optional<Eigen::VectorXd>,
               std::optional<Eigen::VectorXd>>
    reset_perturbation() {
        if (perturbation_active_) {
            perturbation_active_ = false;
            consecutive_degenerate_ = 0;
            return {A_orig_, b_orig_, c_orig_};
        }
        return {std::nullopt, std::nullopt, std::nullopt};
    }

    std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd>
    apply_perturbation(const Eigen::MatrixXd &A, const Eigen::VectorXd &b,
                       const Eigen::VectorXd &c, const std::vector<int> &basis,
                       int iters) {
        if (!perturbation_active_) {
            // Store originals
            A_orig_ = A;
            b_orig_ = b;
            c_orig_ = c;
            perturbation_active_ = true;
        }

        // Apply small random perturbations
        const double pert_scale = 1e-8 * (1.0 + iters * 1e-10);

        Eigen::MatrixXd A_pert = A;
        Eigen::VectorXd b_pert = b;
        Eigen::VectorXd c_pert = c;

        // Perturb RHS slightly
        std::uniform_real_distribution<double> dist(-pert_scale, pert_scale);
        for (int i = 0; i < b_pert.size(); ++i) {
            b_pert(i) += dist(rng_);
        }

        return {A_pert, b_pert, c_pert};
    }

    std::unordered_map<std::string, std::string> get_statistics() const {
        return {
            {"degeneracy_detected", std::to_string(consecutive_degenerate_)},
            {"perturbation_active", perturbation_active_ ? "yes" : "no"}};
    }

private:
    std::mt19937 rng_;
    int consecutive_degenerate_;
    bool perturbation_active_;
    std::optional<Eigen::MatrixXd> A_orig_;
    std::optional<Eigen::VectorXd> b_orig_;
    std::optional<Eigen::VectorXd> c_orig_;
};

// ==============================
// Minimal Steepest-edge pricer
// ==============================
// ==============================
// True Steepest-Edge (FT-consistent, no B^{-1})
// ==============================
class SteepestEdgePricer {
public:
    struct Entry {
        int jN; // the absolute column index (in N) for this nonbasic var
        Eigen::VectorXd t; // cached t = B^{-1} a_j
        double weight;     // w = 1 + ||t||^2
    };

    SteepestEdgePricer(int pool_max = 0, int reset_frequency = 1000)
        : pool_max_(pool_max), reset_freq_(reset_frequency), iter_count_(0) {}

    // Build/refresh the pricing pool by doing FTRAN once per chosen nonbasic
    // If pool_max_ == 0, we include all N (simple but robust).
    template <class BasisLike>
    void build_pool(const BasisLike &B, const Eigen::MatrixXd &A,
                    const std::vector<int> &N) {
        pool_.clear();
        pos_.clear();
        pool_.reserve(pool_max_ > 0 ? std::min<int>(pool_max_, (int)N.size())
                                    : (int)N.size());

        const int take = (pool_max_ > 0)
                             ? std::min<int>(pool_max_, (int)N.size())
                             : (int)N.size();
        for (int k = 0; k < take; ++k) {
            const int j = N[k];
            Entry e;
            e.jN = j;
            e.t = B.solve_B(A.col(j));
            e.weight = 1.0 + e.t.squaredNorm();
            pos_[j] = (int)pool_.size();
            pool_.push_back(std::move(e));
        }
        iter_count_ = 0;
    }

    // Choose entering: arg max |rc_j| / sqrt(w_j) among negatives
    std::optional<int> choose_entering(const Eigen::VectorXd &rcN,
                                       const std::vector<int> &N, double tol) {
        ++iter_count_;
        int best_rel = -1;
        double best_score = -1.0;

        for (int k = 0; k < (int)N.size(); ++k) {
            if (rcN(k) >= -tol)
                continue;
            const int j = N[k];
            double w = 1.0; // default if not cached (shouldn't happen if pool
                            // covers N)
            auto it = pos_.find(j);
            if (it != pos_.end()) {
                w = pool_[it->second].weight;
            } else {
                // If not in pool, fall back to unit weight (Devex-like grace)
                // or you can lazily compute one FTRAN here (tradeoff).
                w = 1.0;
            }
            const double score = (rcN(k) * rcN(k)) / w;
            if (score > best_score) {
                best_score = score;
                best_rel = k;
            }
        }
        if (best_rel < 0)
            return std::nullopt;
        return best_rel;
    }

    // After a pivot (replace row 'leave_rel' with entering 'e_rel'):
    // Inputs:
    //   leave_rel: pivot row r in B (0..m-1)
    //   e_abs    : absolute entering column index e
    //   old_abs  : absolute leaving variable index (just left the basis)
    //   s        : dB = B^{-1} a_e computed *before* updating B
    //   alpha    : s(leave_rel)  (pivot element in FTRAN space)
    //
    // We update all cached t_j and weights via:
    //   t_j' = t_j - s * ( t_j[r] / alpha )
    //   w_j' = 1 + ||t_j'||^2
    //
    // We also:
    //   - remove 'e_abs' from the pool (it just became basic),
    //   - optionally insert the new nonbasic 'old_abs' with t_old' = e_r -
    //   s*(1/alpha)
    //     (since t_old before update equals e_r).
    void update_after_pivot(int leave_rel, int e_abs, int old_abs,
                            const Eigen::VectorXd &s, double alpha,
                            const Eigen::MatrixXd & /*A*/,
                            const std::vector<int> & /*N*/,
                            bool insert_leaver_into_pool = true) {
        if (std::abs(alpha) < 1e-14) {
            // numerically unsafe: rebuild pool on next call
            need_rebuild_ = true;
            return;
        }

        // 1) Update every cached t_j and weight
        const double inv_alpha = 1.0 / alpha;
        for (auto &E : pool_) {
            const double tr = E.t(leave_rel);
            if (tr != 0.0) {
                E.t.noalias() -= s * (tr * inv_alpha);
                E.weight = 1.0 + E.t.squaredNorm();
            }
        }

        // 2) Remove the now-basic entering var from the pool (if present)
        auto itE = pos_.find(e_abs);
        if (itE != pos_.end()) {
            int idx = itE->second;
            // swap-erase
            int last = (int)pool_.size() - 1;
            if (idx != last) {
                pos_[pool_[last].jN] = idx;
                std::swap(pool_[idx], pool_[last]);
            }
            pool_.pop_back();
            pos_.erase(itE);
        }

        // 3) Optionally insert the leaving variable (now nonbasic) with updated
        // t_old'
        if (insert_leaver_into_pool) {
            // t_old (before update) = e_r, so t_old' = e_r - s*(1/alpha)
            Entry E;
            E.jN = old_abs;
            E.t = Eigen::VectorXd::Zero(s.size());
            E.t(leave_rel) = 1.0;           // e_r
            E.t.noalias() -= s * inv_alpha; // - s / alpha
            E.weight = 1.0 + E.t.squaredNorm();

            // If pool has a max size, keep size bounded with a simple strategy:
            if (pool_max_ > 0 && (int)pool_.size() >= pool_max_) {
                // evict the smallest impact (e.g., largest weight => least
                // promising), or just evict the last to keep it simple. Here we
                // evict the largest weight to bias for more "violating"
                // columns.
                int evict = largest_weight_index_();
                pos_.erase(pool_[evict].jN);
                pool_[evict] = std::move(E);
                pos_[pool_[evict].jN] = evict;
            } else {
                pos_[E.jN] = (int)pool_.size();
                pool_.push_back(std::move(E));
            }
        }

        // 4) Periodic full rebuild if requested/aged
        ++iter_count_;
        if (need_rebuild_ || iter_count_ >= reset_freq_) {
            need_rebuild_ =
                true; // signal caller to rebuild at a convenient time
        }
    }

    bool needs_rebuild() const { return need_rebuild_; }
    void clear_rebuild_flag() { need_rebuild_ = false; }

private:
    int largest_weight_index_() const {
        int idx = 0;
        double w = pool_[0].weight;
        for (int i = 1; i < (int)pool_.size(); ++i) {
            if (pool_[i].weight > w) {
                w = pool_[i].weight;
                idx = i;
            }
        }
        return idx;
    }

private:
    std::vector<Entry> pool_;
    std::unordered_map<int, int> pos_; // absolute column -> index in pool_
    int pool_max_{0};                  // 0 == “all N”
    int reset_freq_{1000};
    int iter_count_{0};
    bool need_rebuild_{false};
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
        U_ = A; // working copy
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

        // one refinement step in PLU space
        Eigen::VectorXd r = Pb - (L_ * (U_ * w));
        if (r.lpNorm<Eigen::Infinity>() > 0) {
            Eigen::VectorXd dz = forward_sub_(L_, r);
            Eigen::VectorXd dw = back_sub_(U_, dz);
            x = apply_Pc_(w + dw);
        }
        return x;
    }

    Eigen::VectorXd solveT(const Eigen::VectorXd &c) const {
        if (c.size() != n_)
            throw std::invalid_argument("MarkowitzLU::solveT size mismatch");
        Eigen::VectorXd PcTc = apply_PcT_(c);
        Eigen::VectorXd t = forward_sub_(U_.transpose(), PcTc);
        Eigen::VectorXd s = back_sub_(L_.transpose(), t);
        Eigen::VectorXd y = apply_PrT_inv_(s);

        // one refinement step in PLU^T space
        Eigen::VectorXd r = PcTc - (U_.transpose() * (L_.transpose() * s));
        if (r.lpNorm<Eigen::Infinity>() > 0) {
            Eigen::VectorXd dt = forward_sub_(U_.transpose(), r);
            Eigen::VectorXd ds = back_sub_(L_.transpose(), dt);
            y = apply_PrT_inv_(s + ds);
        }
        return y;
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
        const int r = static_cast<int>(M.rows());
        const int c = static_cast<int>(M.cols());
        std::vector<int> rn(r, 0), cn(c, 0);
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < c; ++j) {
                if (std::abs(M(i, j)) > eps) {
                    rn[i]++;
                    cn[j]++;
                }
            }
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
                    long score = static_cast<long>(rn[i - k] - 1) *
                                 static_cast<long>(cn[j - k] - 1);
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
        if (std::abs(val) >= std::max(abs_floor_, pivot_rel_ * col_abs_max)) {
            return {i, j, val};
        }
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
                pi = k + static_cast<int>(rr);
                pj = k + static_cast<int>(cc);
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
/* Forrest–Tomlin product-form updates over MarkowitzLU */
// ======================================================
class FTBasis {
public:
    struct Eta {
        int j;
        Eigen::VectorXd u;
        Eigen::VectorXd z;
        Eigen::VectorXd w;
        double alpha;
    };

    FTBasis(const Eigen::MatrixXd &A, const std::vector<int> &basis,
            int refactor_every = 64, int compress_every = 32,
            double pivot_rel = 1e-12, double abs_floor = 1e-16,
            double alpha_tol = 1e-10, double z_inf_guard = 1e6)
        : A_(A), m_(static_cast<int>(A.rows())), basis_(basis),
          refactor_every_(refactor_every), compress_every_(compress_every),
          pivot_rel_(pivot_rel), abs_floor_(abs_floor), alpha_tol_(alpha_tol),
          z_inf_guard_(z_inf_guard) {
        if ((int)basis_.size() != m_)
            throw std::invalid_argument("FTBasis: basis size must equal m");
        Bcols_.resize(m_);
        for (int i = 0; i < m_; ++i)
            Bcols_[i] = A_.get().col(basis_[i]);
        base_refactor_();
    }

    void refactor() { base_refactor_(); }

    Eigen::VectorXd solve_B(const Eigen::VectorXd& b) const {
        Eigen::VectorXd x;
        try { x = lu_.solve(b); }
        catch (...) { const_cast<FTBasis*>(this)->base_refactor_(); x = lu_.solve(b); }
        if (!etas_.empty()) x = apply_etas_solve_(x);
        return x;
    }

    Eigen::VectorXd solve_BT(const Eigen::VectorXd& c) const {
        Eigen::VectorXd y;
        try { y = lu_.solveT(c); }
        catch (...) { const_cast<FTBasis*>(this)->base_refactor_(); y = lu_.solveT(c); }
        if (!etas_.empty()) y = apply_etas_solve_T_(y);
        return y;
    }

    void replace_column(int j, const Eigen::VectorXd &new_col) {
        Eigen::VectorXd u = new_col - Bcols_[j];

        Eigen::VectorXd z;
        try {
            z = solve_B(u);
        } catch (...) {
            base_refactor_();
            z = solve_B(u);
        }

        double alpha = 1.0 + z(j);
        if (std::abs(alpha) < alpha_tol_ || update_count_ >= refactor_every_) {
            Bcols_[j] = new_col;
            base_refactor_();
            return;
        }

        Eigen::VectorXd ej = Eigen::VectorXd::Zero(m_);
        ej(j) = 1.0;
        Eigen::VectorXd w;
        try {
            w = solve_BT(ej);
        } catch (...) {
            base_refactor_();
            w = solve_BT(ej);
        }

        etas_.push_back(Eta{j, u, z, w, alpha});
        Bcols_[j] = new_col;
        update_count_++;

        if (need_compress_())
            base_refactor_();
    }

private:
    void base_refactor_() {
        Eigen::MatrixXd B(m_, m_);
        for (int k = 0; k < m_; ++k)
            B.col(k) = Bcols_[k];
        lu_.factor(B, pivot_rel_, abs_floor_);
        etas_.clear();
        update_count_ = 0;
    }

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
        if ((int)etas_.size() >= compress_every_)
            return true;
        double maxabsz = 0.0;
        for (const auto &e : etas_) {
            double v = e.z.cwiseAbs().maxCoeff();
            if (v > maxabsz)
                maxabsz = v;
        }
        return maxabsz > z_inf_guard_;
    }

private:
    std::reference_wrapper<const Eigen::MatrixXd> A_;
    int m_;
    std::vector<int> basis_;
    std::vector<Eigen::VectorXd> Bcols_;
    mutable MarkowitzLU lu_;
    mutable std::vector<Eta> etas_;
    mutable int update_count_{0};

    int refactor_every_{64};
    int compress_every_{32};
    double pivot_rel_{1e-12};
    double abs_floor_{1e-16};
    double alpha_tol_{1e-10};
    double z_inf_guard_{1e6};
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
                 std::optional<std::vector<int>> basis_opt = std::nullopt)
{
    const int n = static_cast<int>(A_in.cols());

    // ---- 0) Build presolve LP: Ax=b, nonnegativity defaults; costs = c_in ----
    presolve::LP lp;
    lp.A = A_in;
    lp.b = b_in;
    lp.sense.assign(static_cast<int>(A_in.rows()), presolve::RowSense::EQ);
    lp.c = c_in;
    lp.l = Eigen::VectorXd::Zero(n);
    lp.u = Eigen::VectorXd::Constant(n, presolve::inf());
    lp.c0 = 0.0;

    // ---- 1) Run presolve (reversible) with proper degeneracy handling ----
    presolve::Presolver::Options popt;
    popt.enable_rowreduce = true;
    popt.enable_scaling   = true;
    popt.max_passes       = 10;

    // Enable conservative mode if problem is skinny (more vars than ~1.2x constraints)
    if (A_in.cols() > static_cast<int>(A_in.rows() * 1.2)) {
        popt.conservative_mode = true;
        // popt.svd_tol = 1e-8; // optional: slightly tighter SVD tol for degeneracy
    }

    presolve::Presolver P(popt);
    const auto pres = P.run(lp);

    // Handle presolve outcomes
    if (pres.proven_infeasible) {
        return make_solution(LPSolution::Status::Infeasible,
                             Eigen::VectorXd::Zero(n),
                             std::numeric_limits<double>::infinity(),
                             {}, 0, {{"presolve","infeasible"}});
    }
    if (pres.proven_unbounded) {
        auto xnan = Eigen::VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
        return make_solution(LPSolution::Status::Unbounded,
                             xnan,
                             -std::numeric_limits<double>::infinity(),
                             {}, 0, {{"presolve","unbounded"}});
    }

    const Eigen::MatrixXd &Atil = pres.reduced.A;
    const Eigen::VectorXd &btil = pres.reduced.b;
    const Eigen::VectorXd &ctil = pres.reduced.c;
    const Eigen::VectorXd &lred = pres.reduced.l;
    const Eigen::VectorXd &ured = pres.reduced.u;

    // ---- 2) m==0 fast path: optimize over bounds only ----
    if (Atil.rows() == 0) {
        Eigen::VectorXd vred = Eigen::VectorXd::Zero(static_cast<int>(ctil.size()));
        bool is_bounded = true;

        for (int j = 0; j < static_cast<int>(ctil.size()); ++j) {
            if (ctil(j) > opt_.tol) {
                // Minimize positive cost -> choose lower bound (default 0 if -inf)
                vred(j) = std::isfinite(lred(j)) ? lred(j) : 0.0;
            } else if (ctil(j) < -opt_.tol) {
                // Minimize negative cost -> choose upper bound; if +inf, unbounded
                if (std::isfinite(ured(j))) {
                    vred(j) = ured(j);
                } else {
                    is_bounded = false;
                    break;
                }
            } else {
                // Zero cost -> any feasible value; pick lower if finite, else 0
                vred(j) = std::isfinite(lred(j)) ? lred(j) : 0.0;
            }
        }

        if (!is_bounded) {
            auto xnan = Eigen::VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
            return make_solution(LPSolution::Status::Unbounded,
                                 xnan,
                                 -std::numeric_limits<double>::infinity(),
                                 {}, 0, {{"presolve","m=0 with negative cost & infinite upper bound"}});
        }

        auto [x_full, obj_correction] = P.postsolve(vred);
        // IMPORTANT: postsolve's obj_correction already includes presolve shifts.
        double total_obj = c_in.dot(x_full) + obj_correction;

        return make_solution(LPSolution::Status::Optimal,
                             x_full, total_obj,
                             {}, 0, {{"presolve","m=0 optimized over bounds"}});
    }

    // ---- 3) Selective upper-bound augmentation (add slacks only for "reasonable" U) ----
    auto build_with_upper_bounds =
        [&](const Eigen::MatrixXd& A0, const Eigen::VectorXd& b0, const Eigen::VectorXd& c0)
        -> std::tuple<Eigen::MatrixXd,Eigen::VectorXd,Eigen::VectorXd,std::vector<int>,std::vector<int>>
    {
        const int m0 = static_cast<int>(A0.rows());
        const int n0 = static_cast<int>(A0.cols());

        std::vector<int> ub_idx;
        ub_idx.reserve(n0);
        const double max_reasonable_bound = 1e6;  // heuristic to avoid huge slacks

        for (int j = 0; j < n0; ++j) {
            if (std::isfinite(ured(j)) && ured(j) <= max_reasonable_bound) {
                ub_idx.push_back(j);
            }
        }

        const int p = static_cast<int>(ub_idx.size());
        if (p == 0) {
            std::vector<int> col_orig_map(n0);
            for (int j = 0; j < n0; ++j) col_orig_map[j] = pres.orig_col_index[j];
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

        // Add constraints: x_j + s_k = u_j
        for (int k = 0; k < p; ++k) {
            const int j = ub_idx[k];
            const int row = m0 + k;
            const int slack_col = n0 + k;
            A(row, j) = 1.0;
            A(row, slack_col) = 1.0;
            b(row) = ured(j);
        }

        std::vector<int> col_orig_map(n0 + p, -1);
        for (int j = 0; j < n0; ++j) col_orig_map[j] = pres.orig_col_index[j];

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

    // ---- 4) Map incoming basis to reduced space ----
    std::optional<std::vector<int>> red_basis_opt = std::nullopt;
    if (basis_opt && !basis_opt->empty()) {
        std::unordered_map<int,int> orig2red;
        orig2red.reserve(static_cast<size_t>(n_eff));
        for (int jr = 0; jr < n_eff; ++jr) {
            int jorig = col_orig_map[jr];
            if (jorig >= 0) orig2red[jorig] = jr;
        }
        std::vector<int> cand;
        cand.reserve(std::min(m_eff, static_cast<int>(basis_opt->size())));
        for (int jorig : *basis_opt) {
            auto it = orig2red.find(jorig);
            if (it != orig2red.end()) {
                cand.push_back(it->second);
                if (static_cast<int>(cand.size()) == m_eff) break;
            }
        }
        if (static_cast<int>(cand.size()) == m_eff) red_basis_opt = std::move(cand);
    }

    // ---- 5) Try Phase II directly on reduced problem ----
    std::vector<int> basis_guess;
    if (red_basis_opt && static_cast<int>(red_basis_opt->size()) == m_eff) {
        basis_guess = *red_basis_opt;
    } else {
        auto maybe = find_initial_basis(Ared, bred);
        if (maybe) basis_guess = *maybe;
    }

    auto add_info = [&](std::unordered_map<std::string,std::string> info){
        info["presolve_actions"] = std::to_string(pres.stack.size());
        info["reduced_m"] = std::to_string(m_eff);
        info["reduced_n"] = std::to_string(n_eff);
        // pres.obj_shift is already accounted for in postsolve's obj_correction;
        // we still expose it for diagnostics.
        info["obj_shift"] = std::to_string(pres.obj_shift);
        return info;
    };

    if (static_cast<int>(basis_guess.size()) == m_eff) {
        auto [st, v2, red_basis2, it2, info2] = phase(Ared, bred, cred, basis_guess);
        if (st == LPSolution::Status::Optimal ||
            st == LPSolution::Status::Unbounded ||
            st == LPSolution::Status::IterLimit)
        {
            auto [x_full, obj_correction] = P.postsolve(v2);
            double total_obj = c_in.dot(x_full) + obj_correction; // no + pres.obj_shift here

            // Map basis back to original indices
            std::vector<int> basis_full;
            basis_full.reserve(red_basis2.size());
            for (int jr : red_basis2) {
                if (jr >= 0 && jr < static_cast<int>(col_orig_map.size())) {
                    int jorig = col_orig_map[jr];
                    if (jorig >= 0) basis_full.push_back(jorig);
                }
            }

            auto info = add_info(std::move(info2));
            return make_solution(st, x_full, total_obj, basis_full, it2, std::move(info));
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
    auto [status1, v1, basis1_out, it1, info1] = phase(A1, b1, c1, basis1);

    if (status1 != LPSolution::Status::Optimal || c1.dot(v1) > opt_.tol) {
        auto info = add_info({{"phase1_status", to_string(status1)}});
        auto e = degen_.get_statistics();
        info.insert(e.begin(), e.end());
        return make_solution(LPSolution::Status::Infeasible,
                             Eigen::VectorXd::Zero(n),
                             std::numeric_limits<double>::infinity(),
                             {}, it1, std::move(info));
    }

    // Warm-start Phase II
    std::vector<int> red_basis2;
    red_basis2.reserve(m_rows);
    for (int j : basis1_out) if (j < n_orig_eff) red_basis2.push_back(j);

    // Basis completion if needed
    if (static_cast<int>(red_basis2.size()) < m_rows) {
        for (int j = 0; j < n_orig_eff; ++j) {
            if (static_cast<int>(red_basis2.size()) == m_rows) break;
            if (std::find(red_basis2.begin(), red_basis2.end(), j) != red_basis2.end()) continue;

            std::vector<int> cand = red_basis2;
            cand.push_back(j);
            if (static_cast<int>(cand.size()) > m_rows) continue;

            Eigen::MatrixXd B = Ared(Eigen::all,
                                     Eigen::VectorXi::Map(cand.data(), static_cast<int>(cand.size())));
            Eigen::FullPivLU<Eigen::MatrixXd> lu(B);
            if (lu.rank() == static_cast<int>(cand.size()) && lu.isInvertible()) {
                red_basis2 = std::move(cand);
            }
        }
    }

    LPSolution::Status status2;
    Eigen::VectorXd v2;
    std::vector<int> red_basis_out;
    int it2 = 0;
    std::unordered_map<std::string, std::string> info2;

    if (static_cast<int>(red_basis2.size()) == m_rows) {
        std::tie(status2, v2, red_basis_out, it2, info2) = phase(Ared, bred, cred, red_basis2);
    } else {
        std::tie(status2, v2, red_basis_out, it2, info2) = phase(Ared, bred, cred, std::nullopt);
        if (status2 == LPSolution::Status::NeedPhase1) {
            status2 = LPSolution::Status::Singular;
            info2["note"] = "reduced matrix cannot form a proper basis";
        }
    }

    int total_iters = it1 + it2;
    auto merged_info = add_info(std::move(info2));
    merged_info.insert({"phase1_iters", std::to_string(it1)});

    auto [x_full, obj_correction] = P.postsolve(v2);
    double total_obj = c_in.dot(x_full) + obj_correction; // no + pres.obj_shift

    std::vector<int> basis_full;
    basis_full.reserve(red_basis_out.size());
    for (int jr : red_basis_out) {
        if (jr >= 0 && jr < static_cast<int>(col_orig_map.size())) {
            int jorig = col_orig_map[jr];
            if (jorig >= 0) basis_full.push_back(jorig);
        }
    }

    if (status2 == LPSolution::Status::Optimal) {
        return make_solution(LPSolution::Status::Optimal,
                             x_full, total_obj,
                             basis_full, total_iters, std::move(merged_info));
    }
    if (status2 == LPSolution::Status::Unbounded) {
        return make_solution(LPSolution::Status::Unbounded,
                             x_full, -std::numeric_limits<double>::infinity(),
                             basis_full, total_iters, std::move(merged_info));
    }

    double obj_fallback = x_full.array().isFinite().all()
                          ? total_obj
                          : std::numeric_limits<double>::quiet_NaN();

    return make_solution(status2, x_full, obj_fallback,
                         basis_full, total_iters, std::move(merged_info));
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
            int one_idx = -1;
            int ones = 0;
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

    static std::tuple<bool, Eigen::MatrixXd, Eigen::VectorXd, std::string>
    presolve_row_reduce(const Eigen::MatrixXd &A, const Eigen::VectorXd &b,
                        double svd_tol, double tol) {
        using SVD = Eigen::BDCSVD<Eigen::MatrixXd>;
        SVD svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

        const auto &S = svd.singularValues();
        const int m = static_cast<int>(A.rows());
        const int n = static_cast<int>(A.cols());

        int r = 0;
        for (int i = 0; i < S.size(); ++i)
            if (S(i) > svd_tol)
                ++r;

        if (r == 0) {
            if (b.lpNorm<Eigen::Infinity>() <= tol)
                return {true, Eigen::MatrixXd(0, n), Eigen::VectorXd(0), "r=0"};
            return {false, {}, {}, "A≈0, b≠0"};
        }

        const Eigen::MatrixXd Ur = svd.matrixU().leftCols(r); // m x r
        if (r < m) {
            const Eigen::VectorXd resid = b - Ur * (Ur.transpose() * b);
            if (resid.lpNorm<Eigen::Infinity>() > 1e3 * tol)
                return {false, {}, {}, "inconsistent equalities"};
        }

        const Eigen::VectorXd Sr = S.head(r);
        const Eigen::MatrixXd Vr = svd.matrixV().leftCols(r); // n x r
        const Eigen::MatrixXd A_tilde =
            Sr.asDiagonal() * Vr.transpose();               // r x n
        const Eigen::VectorXd b_tilde = Ur.transpose() * b; // r
        return {true, A_tilde, b_tilde, "ok"};
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

    // Improved Harris ratio test with numerical safeguards
    std::pair<std::optional<int>, double>
    harris_ratio(const Eigen::VectorXd &xB, const Eigen::VectorXd &dB,
                 double delta, double eta) const {
        std::vector<int> pos;
        pos.reserve(dB.size());
        for (int i = 0; i < dB.size(); ++i) {
            if (dB(i) > delta)
                pos.push_back(i);
        }
        if (pos.empty()) {
            return {std::nullopt, std::numeric_limits<double>::infinity()};
        }

        // Compute minimum ratio
        double theta_star = std::numeric_limits<double>::infinity();
        for (int idx : pos) {
            theta_star = std::min(theta_star, xB(idx) / dB(idx));
        }

        // Harris test: collect candidates within tolerance
        double max_resid = 0.0;
        std::vector<int> candidates;

        for (int idx : pos) {
            double ratio = xB(idx) / dB(idx);
            if (std::abs(ratio - theta_star) <=
                1e-10) { // Tight tolerance for exact ties
                candidates.push_back(idx);
            }
            double resid = xB(idx) - theta_star * dB(idx);
            max_resid = std::max(max_resid, std::max(0.0, resid));
        }

        if (!candidates.empty()) {
            // Among exact ties, choose using secondary criterion (e.g.,
            // smallest index for stability)
            int best = candidates.front();
            for (int idx : candidates) {
                if (idx < best)
                    best = idx;
            }
            return {best, theta_star};
        }

        // Second pass with Harris tolerance
        double kappa = std::max(eta, eta * max_resid);
        std::vector<int> eligible;

        for (int idx : pos) {
            double resid = xB(idx) - theta_star * dB(idx);
            if (resid <= kappa) {
                eligible.push_back(idx);
            }
        }

        if (!eligible.empty()) {
            // Among eligible, choose smallest index for deterministic behavior
            int best = eligible.front();
            for (int idx : eligible) {
                if (idx < best)
                    best = idx;
            }
            return {best, theta_star};
        }

        // Fallback: strict minimum ratio
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
    // ---------- phase (patched with true steepest-edge without B^{-1}) ----------
std::tuple<LPSolution::Status, Eigen::VectorXd, std::vector<int>, int,
           std::unordered_map<std::string, std::string>>
phase(const Eigen::MatrixXd& A,
                      const Eigen::VectorXd& b,
                      const Eigen::VectorXd& c,
                      std::optional<std::vector<int>> basis_opt)
{
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    int iters = 0;

    std::vector<int> basis;
    if (basis_opt) {
        basis = *basis_opt;
        if ((int)basis.size() != m)
            return {LPSolution::Status::NeedPhase1, Eigen::VectorXd::Zero(n), {}, 0,
                    {{"reason","basis size != m"}}};
    } else {
        auto maybe = find_initial_basis(A, b);
        if (!maybe) {
            return {LPSolution::Status::NeedPhase1, Eigen::VectorXd::Zero(n), {}, 0,
                    {{"reason","no_trivial_basis"}}};
        }
        basis = *maybe;
    }

    std::vector<int> N;
    N.reserve(n - m);
    {
        std::vector<char> inB(n, 0);
        for (int j : basis) {
            if (j < 0 || j >= n)
                return {LPSolution::Status::Singular, Eigen::VectorXd::Zero(n), basis, 0,
                        {{"where","initial basis index out of range"}}};
            inB[j] = 1;
        }
        for (int j = 0; j < n; ++j) if (!inB[j]) N.push_back(j);
    }

    // Initialize FT basis solver (dense)
    FTBasis B(A, basis,
              opt_.refactor_every, opt_.compress_every,
              opt_.lu_pivot_rel, opt_.lu_abs_floor,
              opt_.alpha_tol, opt_.z_inf_guard);

    // ---- NEW: True steepest-edge pricer using FT-consistent updates ----
    SteepestEdgePricer pricer(/*pool_max=*/0, opt_.steepest_edge_reset_freq);
    if (opt_.pricing_rule == "steepest_edge") {
        pricer.build_pool(B, A, N); // seed t_j = B^{-1} a_j via FTRAN (no B^{-1} formed)
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
                ++rebuild_attempts; B.refactor();
                if (opt_.pricing_rule == "steepest_edge") {
                    pricer.build_pool(B, A, N);
                    pricer.clear_rebuild_flag();
                }
                continue;
            }
            return {LPSolution::Status::Singular, Eigen::VectorXd::Zero(n), basis, iters,
                    {{"where","solve(B,b) repair failed"}}};
        }

        // Strict negativity test (keeps behavior aligned with previous version)
        if ((xB.array() < -opt_.tol).any()) {
            return {LPSolution::Status::NeedPhase1, Eigen::VectorXd::Zero(n), basis, iters,
                    {{"reason","negative_basic_vars"}}};
        }
        xB = xB.cwiseMax(0.0);

        // y: B^T y = c_B
        Eigen::VectorXd cB(m);
        for (int i = 0; i < m; ++i) cB(i) = c(basis[i]);
        Eigen::VectorXd y;
        try { y = B.solve_BT(cB); }
        catch (...) { B.refactor(); y = B.solve_BT(cB);
            if (opt_.pricing_rule == "steepest_edge") {
                pricer.build_pool(B, A, N);
                pricer.clear_rebuild_flag();
            }
        }

        // reduced costs rN = c_N - A_N^T y
        Eigen::VectorXd rN(N.size());
        for (int k = 0; k < (int)N.size(); ++k) {
            int j = N[k];
            rN(k) = c(j) - A.col(j).dot(y);
        }

        // Choose entering
        std::optional<int> e_rel;
        if (opt_.bland) {
            int idx = -1;
            for (int k = 0; k < (int)N.size(); ++k) if (rN(k) < -opt_.tol) { idx = k; break; }
            if (idx == -1) {
                Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
                for (int i = 0; i < m; ++i) x(basis[i]) = xB(i);
                return {LPSolution::Status::Optimal, clip_small(x), basis, iters, degen_.get_statistics()};
            }
            e_rel = idx;
        } else {
            if (opt_.pricing_rule == "steepest_edge") {
                e_rel = pricer.choose_entering(rN, N, opt_.tol);
            } else {
                int idx = -1; double best = 0.0;
                for (int k = 0; k < (int)N.size(); ++k) if (rN(k) < -opt_.tol) {
                    if (idx < 0 || rN(k) < best) { best = rN(k); idx = k; }
                }
                if (idx >= 0) e_rel = idx;
            }
            if (!e_rel) {
                Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
                for (int i = 0; i < m; ++i) x(basis[i]) = xB(i);
                return {LPSolution::Status::Optimal, clip_small(x), basis, iters, degen_.get_statistics()};
            }
        }

        const int e = N[*e_rel];
        const Eigen::VectorXd a_e = A.col(e);

        // s = dB = B^{-1} a_e
        Eigen::VectorXd dB;
        try { dB = B.solve_B(a_e); }
        catch (...) { B.refactor(); dB = B.solve_B(a_e);
            if (opt_.pricing_rule == "steepest_edge") {
                pricer.build_pool(B, A, N);
                pricer.clear_rebuild_flag();
            }
        }

        // Harris ratio
        auto [leave_rel_opt, step] = harris_ratio(xB, dB, opt_.ratio_delta, opt_.ratio_eta);
        if (!leave_rel_opt) {
            Eigen::VectorXd x = Eigen::VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
            return {LPSolution::Status::Unbounded, x, basis, iters, degen_.get_statistics()};
        }
        const int r = *leave_rel_opt;
        const double alpha = dB(r);
        const int old_abs = basis[r];
        const int e_abs   = e;

        // Degeneracy hooks
        bool is_degenerate = degen_.detect_degeneracy(step, opt_.deg_step_tol);
        if (is_degenerate && degen_.should_apply_perturbation()) {
            auto [Ap, bp, cp] = degen_.apply_perturbation(A, b, c, basis, iters);
            (void)Ap; (void)bp; (void)cp;
        } else {
            (void)degen_.reset_perturbation();
        }

        // ---- NEW: update steepest-edge weights and cached FTRANs using FT-consistent rank-1 ----
        if (opt_.pricing_rule == "steepest_edge") {
            pricer.update_after_pivot(r, e_abs, old_abs, dB /*s*/, alpha, A, N, /*insert_leaver=*/true);
        }

        // Pivot in the combinatorics
        basis[r]   = e_abs;
        N[*e_rel]  = old_abs;

        // Update B with new column, handle refactor and pool rebuild signals
        try {
            B.replace_column(r, a_e);
        } catch (...) {
            B.refactor();
            if (opt_.pricing_rule == "steepest_edge") {
                pricer.build_pool(B, A, N);
                pricer.clear_rebuild_flag();
            }
        }

        if (opt_.pricing_rule == "steepest_edge" && pricer.needs_rebuild()) {
            pricer.build_pool(B, A, N);
            pricer.clear_rebuild_flag();
        }
    }

    return {LPSolution::Status::IterLimit, Eigen::VectorXd::Zero(n), basis, iters, degen_.get_statistics()};
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
};
