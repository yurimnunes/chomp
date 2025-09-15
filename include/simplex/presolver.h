#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace presolve {

// ------------------------------
// Problem container
// ------------------------------
enum class RowSense : int { LE = -1, EQ = 0, GE = 1 };

struct LP {
    Eigen::MatrixXd A;           // m x n
    Eigen::VectorXd b;           // m
    std::vector<RowSense> sense; // m
    Eigen::VectorXd c;           // n
    Eigen::VectorXd l;           // n  (can be -inf)
    Eigen::VectorXd u;           // n  (can be +inf)
    double c0 = 0.0;
};

inline double inf() { return std::numeric_limits<double>::infinity(); }
inline double ninf() { return -std::numeric_limits<double>::infinity(); }
inline bool is_finite(double v) { return std::isfinite(v); }

struct ActRowReduce {
    Eigen::MatrixXd U;
    Eigen::VectorXi keep;
    int old_m = 0;
};
struct ActRemoveRow {
    int i;
    RowSense sense;
    double rhs;
    Eigen::VectorXd row;
};
struct ActRemoveCol {
    int j;
    double c_j;
    double l_j, u_j;
    Eigen::VectorXd col;
};
struct ActFixVar {
    int j;
    double x_fix;
    double c_j;
    Eigen::VectorXd col;
};
struct ActTightenBound {
    int j;
    double old_l, old_u;
};
struct ActScaleRow {
    int i;
    double scale;
};
struct ActScaleCol {
    int j;
    double scale;
};
struct ActSingletonRowElim {
    int i;
    int j;
    RowSense sense;
    double rhs;
    double aij;
    Eigen::VectorXd row;
};
struct ActSingletonColElim {
    int j;
    int i;
    double aij;
    Eigen::VectorXd col;
};
struct ActDualFix {
    int j;
    double old_l, old_u;
    double x_fix;
};

using Action =
    std::variant<ActRowReduce, ActRemoveRow, ActRemoveCol, ActFixVar,
                 ActTightenBound, ActScaleRow, ActScaleCol, ActSingletonRowElim,
                 ActSingletonColElim, ActDualFix>;

struct PresolveResult {
    LP reduced;
    std::vector<Action> stack;
    std::vector<int> orig_col_index;
    std::vector<int> orig_row_index;
    double obj_shift = 0.0;
    bool proven_infeasible = false;
    bool proven_unbounded = false;
};

struct ActivityBounds {
    double min_act = 0.0, max_act = 0.0;
};

inline ActivityBounds row_activity_bounds(const Eigen::RowVectorXd &a,
                                          const Eigen::VectorXd &l,
                                          const Eigen::VectorXd &u) {
    ActivityBounds ab{0.0, 0.0};
    const int n = (int)a.size();
    for (int j = 0; j < n; ++j) {
        const double coeff = a(j);
        if (coeff >= 0.0) {
            ab.min_act +=
                coeff *
                (is_finite(l(j)) ? l(j) : (coeff == 0.0 ? 0.0 : -inf()));
            ab.max_act += coeff * (is_finite(u(j)) ? u(j) : inf());
        } else {
            ab.min_act += coeff * (is_finite(u(j)) ? u(j) : inf());
            ab.max_act += coeff * (is_finite(l(j)) ? l(j) : -inf());
        }
    }
    return ab;
}

inline bool nearly_zero(double v, double tol = 1e-12) {
    return std::abs(v) <= tol;
}

enum class RowReduceMethod { RRQR, SVD, Auto };

class Presolver {
public:
    struct Options {
        // numerics
        double svd_tol = 1e-8;
        double zero_tol = 1e-12;
        double infeas_tol = 1e-9;
        double rrqr_pivot_tol = 1e-8;
        double rr_infeas_mult = 1e3;
        double cond_max = 1e10;

        // passes
        int max_passes = 5;
        bool enable_rowreduce = true;
        bool enable_scaling = true;     // row scaling only
        bool enable_col_scaling = true; // OFF in non-destructive mode

        // behavior
        bool non_destructive = false; // keep ranking stable (default)
        bool allow_structural_changes =
            true; // if true, can remove columns / change c

        // row-reduce
        int max_ruiz_iters = 10;
        RowReduceMethod row_reduce_method = RowReduceMethod::Auto;
        bool use_iter_refine = true;

        // conservative extras
        bool classic_row_reduce = true;
        bool conservative_mode = false;
    };

    Presolver() : opt_() { domprop_min_delta_ = 1e3 * opt_.infeas_tol; }
    explicit Presolver(const Options &opt) : opt_(opt) {
        domprop_min_delta_ = 1e3 * opt_.infeas_tol;
        if (opt_.non_destructive) {
            opt_.enable_col_scaling = false;
        }
    }

    using Result = PresolveResult;

    PresolveResult run(const LP &in) {
        res_.stack.clear();
        res_.obj_shift = 0.0;
        res_.proven_infeasible = false;
        res_.proven_unbounded = false;

        LP P = in;
        const int m0 = (int)P.A.rows();
        const int n0 = (int)P.A.cols();
        res_.orig_row_index.resize(m0);
        std::iota(res_.orig_row_index.begin(), res_.orig_row_index.end(), 0);
        res_.orig_col_index.resize(n0);
        std::iota(res_.orig_col_index.begin(), res_.orig_col_index.end(), 0);

        // sanity
        if ((int)P.sense.size() != (int)P.b.size())
            throw std::invalid_argument("presolve: sense size mismatch with b");
        if ((int)P.l.size() != n0 || (int)P.u.size() != n0 ||
            (int)P.c.size() != n0)
            throw std::invalid_argument("presolve: vector sizes must equal n");

        if (!check_and_fix_bounds(P)) {
            res_.reduced = std::move(P);
            res_.proven_infeasible = true;
            return res_;
        }
        if (detect_unboundedness(P)) {
            res_.reduced = std::move(P);
            res_.proven_unbounded = true;
            return res_;
        }

        if (opt_.enable_scaling)
            scale_rows_unit_inf(P);
        if (opt_.enable_col_scaling && !opt_.non_destructive)
            scale_cols_unit_inf(P);

        if (opt_.enable_rowreduce) {
            if (!row_reduce(P)) {
                res_.reduced = std::move(P);
                res_.proven_infeasible = true;
                return res_;
            }
        }

        int pass = 0;
        bool changed = true;
        while (changed && pass < opt_.max_passes) {
            changed = false;

            if (detect_unboundedness(P)) {
                res_.reduced = std::move(P);
                res_.proven_unbounded = true;
                return res_;
            }

            // Zero-rows only (safe)
            changed |= remove_free_zero_rows(P);
            if (res_.proven_infeasible)
                break;

            // Fixed variable handling:
            //  - non_destructive => "fix-and-zero": keep column, zero A(:,j),
            //  keep c_j, set l=u=x*
            //  - structural      => erase column and shift
            changed |= fixed_variable_detection(P);

            // Tighten bounds by row activities (no c changes)
            changed |= tighten_bounds_by_rows(P);
            if (res_.proven_infeasible)
                break;

            // Single guarded domain propagation (tiny-change threshold)
            changed |= domain_propagation_once(P);

            // Exact duplicate row removal (safe)
            changed |= redundancy_duplicate_rows(P);

            // No structural or objective-changing passes unless explicitly
            // enabled
            ++pass;
        }

        if (detect_unboundedness(P)) {
            res_.reduced = std::move(P);
            res_.proven_unbounded = true;
            return res_;
        }

        prune_zero_rows(P);

        res_.reduced = std::move(P);
        return res_;
    }

    std::pair<Eigen::VectorXd, double>
    postsolve(const Eigen::VectorXd &x_red) const {
        const int n_full_guess = (int)res_.orig_col_index.size();
        Eigen::VectorXd x_full = Eigen::VectorXd::Constant(
            n_full_guess, std::numeric_limits<double>::quiet_NaN());
        for (int jr = 0; jr < (int)x_red.size(); ++jr) {
            int jorig = res_.orig_col_index[jr];
            if (jorig >= 0 && jorig < n_full_guess)
                x_full(jorig) = x_red(jr);
        }
        double obj_correction = res_.obj_shift;

        for (int k = (int)res_.stack.size() - 1; k >= 0; --k) {
            const auto &act = res_.stack[k];
            std::visit(
                [&](auto const &a) { undo_action(a, x_full, obj_correction); },
                act);
        }

        return {x_full, obj_correction};
    }

    const PresolveResult &result() const noexcept { return res_; }

private:
    // ---------- numerics helpers ----------
    static double cond2_estimate_upper(const Eigen::MatrixXd &R11) {
        if (R11.size() == 0)
            return 0.0;
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(R11, Eigen::ComputeThinU |
                                                       Eigen::ComputeThinV);
        const auto s = svd.singularValues();
        const double smax = s(0);
        const double smin = s.tail(1)(0);
        return (smin > 0.0) ? (smax / smin)
                            : std::numeric_limits<double>::infinity();
    }

    static std::pair<Eigen::VectorXd, Eigen::VectorXd>
    ruiz_balance(Eigen::MatrixXd &A, int iters, double floor = 1e-12) {
        const int m = (int)A.rows(), n = (int)A.cols();
        Eigen::VectorXd Dr = Eigen::VectorXd::Ones(m),
                        Dc = Eigen::VectorXd::Ones(n);
        for (int k = 0; k < iters; ++k) {
            for (int i = 0; i < m; ++i) {
                double s = std::sqrt(A.row(i).cwiseAbs().mean());
                if (!std::isfinite(s) || s < floor)
                    s = 1.0;
                A.row(i) /= s;
                Dr(i) *= s;
            }
            for (int j = 0; j < n; ++j) {
                double s = std::sqrt(A.col(j).cwiseAbs().mean());
                if (!std::isfinite(s) || s < floor)
                    s = 1.0;
                A.col(j) /= s;
                Dc(j) *= s;
            }
        }
        return {Dr, Dc};
    }
    bool detect_unboundedness(const LP &P) const {
        const int n = (int)P.A.cols(), m = (int)P.A.rows();
        for (int j = 0; j < n; ++j) {
            const bool can_inc = !is_finite(P.u(j));
            const bool can_dec = !is_finite(P.l(j));
            if (!can_inc && !can_dec)
                continue;

            // guard: if column is numerically zero, don’t use it to declare
            // unbounded
            if (P.A.col(j).cwiseAbs().maxCoeff() <= opt_.zero_tol) {
                if ((can_inc || can_dec) && std::abs(P.c(j)) > opt_.zero_tol) {
                    // only suspicious if literally free and improving, but we
                    // still skip declaring here
                    continue; // let the solver (not presolve) decide
                }
                continue;
            }

            bool blocks_plus = false, blocks_minus = false;
            for (int i = 0; i < m; ++i) {
                const double aij = P.A(i, j);
                if (std::abs(aij) <= opt_.zero_tol)
                    continue;
                if (P.sense[i] == RowSense::EQ) {
                    blocks_plus = blocks_minus = true;
                    break;
                }
                if (P.sense[i] == RowSense::LE) {
                    if (aij > 0)
                        blocks_plus = true;
                    if (aij < 0)
                        blocks_minus = true;
                } else if (P.sense[i] == RowSense::GE) {
                    if (aij < 0)
                        blocks_plus = true;
                    if (aij > 0)
                        blocks_minus = true;
                }
            }
            if (P.c(j) < -opt_.zero_tol && can_inc && !blocks_plus)
                return true;
            if (P.c(j) > opt_.zero_tol && can_dec && !blocks_minus)
                return true;
        }
        return false;
    }

    void scale_rows_unit_inf(LP &P) {
        const int m = (int)P.A.rows();
        for (int i = 0; i < m; ++i) {
            const double s = P.A.row(i).cwiseAbs().maxCoeff();
            if (s > 0 && !nearly_zero(s, opt_.zero_tol) && std::isfinite(s)) {
                P.A.row(i) /= s;
                P.b(i) /= s;
                res_.stack.emplace_back(ActScaleRow{i, s});
            }
        }
    }
    void scale_cols_unit_inf(LP &P) {
        // disabled by default to preserve ranking
        if (opt_.non_destructive)
            return;
        const int n = (int)P.A.cols();
        for (int j = 0; j < n; ++j) {
            const double s = P.A.col(j).cwiseAbs().maxCoeff();
            if (s > 0 && !nearly_zero(s, opt_.zero_tol) && std::isfinite(s)) {
                P.A.col(j) /= s;
                P.c(j) /= s;
                if (is_finite(P.l(j)))
                    P.l(j) *= s;
                if (is_finite(P.u(j)))
                    P.u(j) *= s;
                res_.stack.emplace_back(ActScaleCol{j, s});
            }
        }
    }

    // --- row reduction: choose RRQR by default, fallback to SVD ---
    bool row_reduce(LP &P) {
        if (!opt_.enable_rowreduce)
            return true;
        switch (opt_.row_reduce_method) {
        case RowReduceMethod::RRQR:
            return row_reduce_rrqr(P);
        case RowReduceMethod::SVD:
            return row_reduce_svd(P);
        case RowReduceMethod::Auto:
        default:
            return row_reduce_rrqr(P);
        }
    }

    bool row_reduce_svd(LP &P) {
        using SVD = Eigen::BDCSVD<Eigen::MatrixXd>;
        const int m = (int)P.A.rows(), n = (int)P.A.cols();
        if (m == 0)
            return true;

        SVD svd(P.A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const auto &S = svd.singularValues();
        const double eps = std::numeric_limits<double>::epsilon();
        const double smax = (S.size() > 0 ? S(0) : 0.0);
        const double thr = std::max(opt_.svd_tol * smax, 100.0 * eps * smax);

        int r = 0;
        for (int i = 0; i < S.size(); ++i)
            if (S(i) > thr)
                ++r;

        const double ainfn = P.A.cwiseAbs().rowwise().sum().maxCoeff();
        const bool matrix_not_tiny = (smax > 1e3 * eps * std::max(1.0, ainfn));
        if (r == 0) {
            if (!matrix_not_tiny) {
                if (P.b.lpNorm<Eigen::Infinity>() <= opt_.infeas_tol) {
                    res_.stack.emplace_back(ActRowReduce{
                        Eigen::MatrixXd::Zero(m, 0), Eigen::VectorXi(), m});
                    P.A.resize(0, n);
                    P.b.resize(0);
                    P.sense.clear();
                    res_.orig_row_index.clear();
                    return true;
                } else {
                    res_.proven_infeasible = true;
                    return false;
                }
            } else
                return true;
        }

        const Eigen::MatrixXd Ur = svd.matrixU().leftCols(r);
        if (r < m) {
            const Eigen::VectorXd resid = P.b - Ur * (Ur.transpose() * P.b);
            const double res_inf = resid.lpNorm<Eigen::Infinity>();
            const double allowed = opt_.rr_infeas_mult * opt_.infeas_tol *
                                   std::max(1.0, P.b.lpNorm<Eigen::Infinity>());
            if (res_inf > allowed) {
                res_.proven_infeasible = true;
                return false;
            }
        }

        const Eigen::VectorXd Sr = S.head(r);
        const Eigen::MatrixXd Vr = svd.matrixV().leftCols(r);
        const Eigen::MatrixXd Atil = Sr.asDiagonal() * Vr.transpose();
        const Eigen::VectorXd btil = Ur.transpose() * P.b;

        Eigen::VectorXi keep(r);
        for (int i = 0; i < r; ++i)
            keep(i) = i;
        res_.stack.emplace_back(ActRowReduce{Ur, keep, m});
        P.A = Atil;
        P.b = btil;
        P.sense.assign(r, RowSense::EQ);
        res_.orig_row_index.resize(r);
        std::iota(res_.orig_row_index.begin(), res_.orig_row_index.end(), 0);
        return true;
    }

    bool row_reduce_rrqr(LP &P) {
        const int m = (int)P.A.rows(), n = (int)P.A.cols();
        if (m == 0)
            return true;

        Eigen::MatrixXd A = P.A;
        Eigen::VectorXd b = P.b;
        if (opt_.max_ruiz_iters > 0) {
            auto [Dr, Dc] = ruiz_balance(A, opt_.max_ruiz_iters);
            for (int i = 0; i < m; ++i)
                b(i) /= Dr(i);
            (void)Dc;
        }

        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qrA(A);
        if (qrA.info() != Eigen::Success)
            return true;

        const int kmax = std::min(m, n);
        Eigen::MatrixXd R = qrA.matrixR()
                                .topLeftCorner(kmax, kmax)
                                .template triangularView<Eigen::Upper>();
        Eigen::VectorXd diagR = R.diagonal().cwiseAbs();
        const double eps = std::numeric_limits<double>::epsilon();
        const double dmax = (diagR.size() ? diagR.maxCoeff() : 0.0);
        const double rthr = std::max(
            {opt_.rrqr_pivot_tol, opt_.svd_tol * dmax, 100.0 * eps * dmax});

        int r = 0;
        for (; r < diagR.size(); ++r)
            if (diagR(r) < rthr)
                break;
        if (r == 0) {
            const double ainfn = P.A.cwiseAbs().rowwise().sum().maxCoeff();
            if (ainfn > 1e3 * eps)
                return true;
            if (P.b.lpNorm<Eigen::Infinity>() <= opt_.infeas_tol) {
                res_.stack.emplace_back(ActRowReduce{
                    Eigen::MatrixXd::Zero(m, 0), Eigen::VectorXi(), m});
                P.A.resize(0, n);
                P.b.resize(0);
                P.sense.clear();
                res_.orig_row_index.clear();
                return true;
            } else {
                res_.proven_infeasible = true;
                return false;
            }
        }

        auto cond_ok = [&](int rr) -> bool {
            if (rr <= 0)
                return false;
            Eigen::MatrixXd R11 = R.topLeftCorner(rr, rr);
            const double kappa = cond2_estimate_upper(R11);
            return (kappa <= opt_.cond_max || !std::isfinite(kappa));
        };
        int r_try = r;
        while (r_try <= (int)diagR.size()) {
            if (!cond_ok(r_try))
                break;
            ++r_try;
        }
        r = std::max(1, std::min(r_try - 1, (int)diagR.size()));

        Eigen::HouseholderQR<Eigen::MatrixXd> qra_full(P.A);
        Eigen::MatrixXd Ur = Eigen::MatrixXd::Identity(m, r);
        Ur = qra_full.householderQ() * Ur;

        Eigen::VectorXd resid = P.b - Ur * (Ur.transpose() * P.b);
        const double allowed = opt_.rr_infeas_mult * opt_.infeas_tol *
                               std::max(1.0, P.b.lpNorm<Eigen::Infinity>());
        if (resid.lpNorm<Eigen::Infinity>() > allowed)
            return row_reduce_svd(P);

        Eigen::MatrixXd Atil = Ur.transpose() * P.A;
        Eigen::VectorXd btil = Ur.transpose() * P.b;
        if (opt_.use_iter_refine) {
            Eigen::VectorXd corr = Ur.transpose() * (P.b - Ur * btil);
            btil += corr;
        }
        Eigen::VectorXi keep(r);
        for (int i = 0; i < r; ++i)
            keep(i) = i;
        res_.stack.emplace_back(ActRowReduce{Ur, keep, m});
        P.A = std::move(Atil);
        P.b = std::move(btil);
        P.sense.assign(r, RowSense::EQ);
        res_.orig_row_index.resize(r);
        std::iota(res_.orig_row_index.begin(), res_.orig_row_index.end(), 0);
        return true;
    }

    // ---------- passes that do NOT touch columns or c ----------
    bool remove_free_zero_rows(LP &P) {
        bool changed = false;
        for (int i = 0; i < (int)P.A.rows();) {
            if (i >= (int)P.A.rows())
                break;
            const auto row = P.A.row(i);
            if (row.cwiseAbs().maxCoeff() <= opt_.zero_tol) {
                const double rhs = P.b(i);
                if ((P.sense[i] == RowSense::EQ &&
                     std::abs(rhs) > opt_.infeas_tol) ||
                    (P.sense[i] == RowSense::LE && rhs < -opt_.infeas_tol) ||
                    (P.sense[i] == RowSense::GE && rhs > opt_.infeas_tol)) {
                    res_.proven_infeasible = true;
                    return true;
                }
                res_.stack.emplace_back(
                    ActRemoveRow{i, P.sense[i], rhs, row.transpose()});
                erase_row(P, i);
                changed = true;
            } else
                ++i;
        }
        return changed;
    }

    // non_destructive fix-and-zero
    bool fixed_variable_detection(LP &P) {
        bool changed = false;
        for (int j = 0; j < (int)P.A.cols(); ++j) {
            const double lj = P.l(j), uj = P.u(j);
            if (!(is_finite(lj) && is_finite(uj) &&
                  std::abs(lj - uj) <= opt_.zero_tol))
                continue;

            const double xfix = 0.5 * (lj + uj);
            // b <- b - A(:,j)*xfix
            P.b.noalias() -= P.A.col(j) * xfix;

            if (opt_.allow_structural_changes && !opt_.non_destructive) {
                // old behavior: remove column, shift objective
                res_.obj_shift += P.c(j) * xfix;
                res_.stack.emplace_back(ActFixVar{j, xfix, P.c(j), P.A.col(j)});
                erase_col(P, j);
                --j;
                changed = true;
            } else {
                // keep column for ranking: zero it out, keep c_j, set l=u=xfix
                res_.stack.emplace_back(ActTightenBound{j, P.l(j), P.u(j)});
                P.A.col(j).setZero();
                P.l(j) = xfix;
                P.u(j) = xfix;
                changed = true;
            }
        }
        return changed;
    }

    bool tighten_bounds_by_rows(LP &P) {
        bool changed = false;
        const int m = (int)P.A.rows(), n = (int)P.A.cols();
        for (int i = 0; i < m; ++i) {
            const auto ab = row_activity_bounds(P.A.row(i), P.l, P.u);
            const double rhs = P.b(i);
            if (P.sense[i] == RowSense::LE) {
                if (ab.min_act > rhs + opt_.infeas_tol) {
                    res_.proven_infeasible = true;
                    return true;
                }
            } else if (P.sense[i] == RowSense::GE) {
                if (ab.max_act < rhs - opt_.infeas_tol) {
                    res_.proven_infeasible = true;
                    return true;
                }
            } else {
                if (ab.min_act > rhs + opt_.infeas_tol ||
                    ab.max_act < rhs - opt_.infeas_tol) {
                    res_.proven_infeasible = true;
                    return true;
                }
            }

            for (int j = 0; j < n; ++j) {
                const double aij = P.A(i, j);
                if (std::abs(aij) <= opt_.zero_tol)
                    continue;

                auto contrib = [&](bool take_min) {
                    if (aij >= 0)
                        return (take_min
                                    ? aij * (is_finite(P.l(j)) ? P.l(j) : 0.0)
                                    : aij * (is_finite(P.u(j)) ? P.u(j) : 0.0));
                    else
                        return (take_min
                                    ? aij * (is_finite(P.u(j)) ? P.u(j) : 0.0)
                                    : aij * (is_finite(P.l(j)) ? P.l(j) : 0.0));
                };
                const double other_min = ab.min_act - contrib(true);
                const double other_max = ab.max_act - contrib(false);

                double L = P.l(j), U = P.u(j);
                if (P.sense[i] == RowSense::LE) {
                    if (aij > 0 && is_finite(U))
                        U = std::min(U, (rhs - other_min) / aij);
                    else if (aij < 0 && is_finite(L))
                        L = std::max(L, (rhs - other_min) / aij);
                } else if (P.sense[i] == RowSense::GE) {
                    if (aij > 0 && is_finite(L))
                        L = std::max(L, (rhs - other_max) / aij);
                    else if (aij < 0 && is_finite(U))
                        U = std::min(U, (rhs - other_max) / aij);
                } else {
                    if (aij > 0) {
                        if (is_finite(U))
                            U = std::min(U, (rhs - other_min) / aij);
                        if (is_finite(L))
                            L = std::max(L, (rhs - other_max) / aij);
                    } else {
                        if (is_finite(L))
                            L = std::max(L, (rhs - other_min) / aij);
                        if (is_finite(U))
                            U = std::min(U, (rhs - other_max) / aij);
                    }
                }
                if (L > U + opt_.infeas_tol) {
                    res_.proven_infeasible = true;
                    return true;
                }
                if ((is_finite(L) && L > P.l(j) + opt_.zero_tol) ||
                    (is_finite(U) && U < P.u(j) - opt_.zero_tol)) {
                    res_.stack.emplace_back(ActTightenBound{j, P.l(j), P.u(j)});
                    P.l(j) = L;
                    P.u(j) = U;
                    changed = true;
                }
            }
        }
        return changed;
    }

    bool domain_propagation_once(LP &P) {
        bool changed_any = false;
        const int m = (int)P.A.rows(), n = (int)P.A.cols();
        for (int i = 0; i < m; ++i) {
            const Eigen::RowVectorXd row = P.A.row(i);
            const double rhs = P.b(i);
            ActivityBounds ab = row_activity_bounds(row, P.l, P.u);

            if (P.sense[i] == RowSense::LE) {
                if (ab.min_act > rhs + opt_.infeas_tol) {
                    res_.proven_infeasible = true;
                    return true;
                }
            } else if (P.sense[i] == RowSense::GE) {
                if (ab.max_act < rhs - opt_.infeas_tol) {
                    res_.proven_infeasible = true;
                    return true;
                }
            } else {
                if (ab.min_act > rhs + opt_.infeas_tol ||
                    ab.max_act < rhs - opt_.infeas_tol) {
                    res_.proven_infeasible = true;
                    return true;
                }
            }

            for (int j = 0; j < n; ++j) {
                const double aij = row(j);
                if (std::abs(aij) <= opt_.zero_tol)
                    continue;

                double L = P.l(j), U = P.u(j);
                double other_min =
                    (aij >= 0 ? ab.min_act - aij * (is_finite(L) ? L : 0.0)
                              : ab.min_act - aij * (is_finite(U) ? U : 0.0));
                double other_max =
                    (aij >= 0 ? ab.max_act - aij * (is_finite(U) ? U : 0.0)
                              : ab.max_act - aij * (is_finite(L) ? L : 0.0));
                double newL = L, newU = U;
                if (P.sense[i] == RowSense::LE) {
                    if (aij > 0 && is_finite(newU))
                        newU = std::min(newU, (rhs - other_min) / aij);
                    else if (aij < 0 && is_finite(newL))
                        newL = std::max(newL, (rhs - other_min) / aij);
                } else if (P.sense[i] == RowSense::GE) {
                    if (aij > 0 && is_finite(newL))
                        newL = std::max(newL, (rhs - other_max) / aij);
                    else if (aij < 0 && is_finite(newU))
                        newU = std::min(newU, (rhs - other_max) / aij);
                } else {
                    if (aij > 0) {
                        if (is_finite(newU))
                            newU = std::min(newU, (rhs - other_min) / aij);
                        if (is_finite(newL))
                            newL = std::max(newL, (rhs - other_max) / aij);
                    } else {
                        if (is_finite(newL))
                            newL = std::max(newL, (rhs - other_min) / aij);
                        if (is_finite(newU))
                            newU = std::min(newU, (rhs - other_max) / aij);
                    }
                }
                if (newL > newU + opt_.infeas_tol) {
                    res_.proven_infeasible = true;
                    return true;
                }

                const bool bigL = (is_finite(L) && is_finite(newL) &&
                                   (newL - L) > domprop_min_delta_);
                const bool bigU = (is_finite(U) && is_finite(newU) &&
                                   (U - newU) > domprop_min_delta_);
                if ((bigL && newL > L) || (bigU && newU < U)) {
                    res_.stack.emplace_back(ActTightenBound{j, P.l(j), P.u(j)});
                    P.l(j) = std::max(L, newL);
                    P.u(j) = std::min(U, newU);
                    changed_any = true;
                }
            }
        }
        return changed_any;
    }

    bool redundancy_duplicate_rows(LP &P) {
        bool changed = false;
        for (int i = 0; i < (int)P.A.rows();) {
            if (i >= (int)P.A.rows())
                break;
            bool removed = false;
            for (int k = i + 1; k < (int)P.A.rows(); ++k) {
                if ((P.sense[i] == P.sense[k]) &&
                    (P.A.row(i) - P.A.row(k)).cwiseAbs().maxCoeff() <=
                        opt_.zero_tol &&
                    std::abs(P.b(i) - P.b(k)) <= opt_.infeas_tol) {
                    res_.stack.emplace_back(ActRemoveRow{
                        k, P.sense[k], P.b(k), P.A.row(k).transpose()});
                    erase_row(P, k);
                    removed = true;
                    changed = true;
                    break;
                }
            }
            if (!removed)
                ++i;
        }
        return changed;
    }

    void prune_zero_rows(LP &P) {
        for (int i = 0; i < (int)P.A.rows();) {
            if (P.A.row(i).cwiseAbs().maxCoeff() <= opt_.zero_tol) {
                res_.stack.emplace_back(ActRemoveRow{i, P.sense[i], P.b(i),
                                                     P.A.row(i).transpose()});
                erase_row(P, i);
            } else
                ++i;
        }
    }

    // ---------- maintenance ----------
    bool check_and_fix_bounds(LP &P) const {
        const int n = (int)P.A.cols();
        for (int j = 0; j < n; ++j) {
            if (is_finite(P.l(j)) && is_finite(P.u(j)) &&
                P.l(j) > P.u(j) + opt_.infeas_tol)
                return false;
            if (is_finite(P.l(j)) && is_finite(P.u(j)) && P.l(j) > P.u(j)) {
                const double mid = 0.5 * (P.l(j) + P.u(j));
                P.l(j) = P.u(j) = mid;
            }
        }
        return true;
    }

    void erase_row(LP &P, int i) {
        const int m = (int)P.A.rows(), n = (int)P.A.cols();
        if (i < m - 1) {
            P.A.block(i, 0, m - i - 1, n) = P.A.block(i + 1, 0, m - i - 1, n);
            P.b.segment(i, m - i - 1) = P.b.segment(i + 1, m - i - 1);
            for (int k = i; k < m - 1; ++k)
                P.sense[k] = P.sense[k + 1];
        }
        P.A.conservativeResize(m - 1, n);
        P.b.conservativeResize(m - 1);
        P.sense.pop_back();
        if ((int)res_.orig_row_index.size() == m)
            res_.orig_row_index.erase(res_.orig_row_index.begin() + i);
    }

    void erase_col(LP &P, int j) {
        const int m = (int)P.A.rows(), n = (int)P.A.cols();
        if (j < n - 1) {
            P.A.block(0, j, m, n - j - 1) = P.A.block(0, j + 1, m, n - j - 1);
            P.c.segment(j, n - j - 1) = P.c.segment(j + 1, n - j - 1);
            P.l.segment(j, n - j - 1) = P.l.segment(j + 1, n - j - 1);
            P.u.segment(j, n - j - 1) = P.u.segment(j + 1, n - j - 1);
        }
        P.A.conservativeResize(m, n - 1);
        P.c.conservativeResize(n - 1);
        P.l.conservativeResize(n - 1);
        P.u.conservativeResize(n - 1);
        if ((int)res_.orig_col_index.size() == n)
            res_.orig_col_index.erase(res_.orig_col_index.begin() + j);
    }

    // ---------- undo ----------
    static void undo_action(const ActScaleRow &, Eigen::VectorXd &, double &) {}
    static void undo_action(const ActScaleCol &a, Eigen::VectorXd &x,
                            double &) {
        if (a.j < (int)x.size() && std::isfinite(x(a.j)))
            x(a.j) /= a.scale;
    }
    static void undo_action(const ActRemoveRow &, Eigen::VectorXd &, double &) {
    }
    static void undo_action(const ActRowReduce &, Eigen::VectorXd &, double &) {
    }
    static void undo_action(const ActFixVar &a, Eigen::VectorXd &x,
                            double &obj) {
        if (a.j >= (int)x.size()) {
            Eigen::VectorXd xnew(a.j + 1);
            xnew.setConstant(std::numeric_limits<double>::quiet_NaN());
            xnew.head(x.size()) = x;
            x.swap(xnew);
        }
        x(a.j) = a.x_fix;
        obj += a.c_j * a.x_fix;
    }
    static void undo_action(const ActTightenBound &, Eigen::VectorXd &,
                            double &) {}
    static void undo_action(const ActSingletonRowElim &a, Eigen::VectorXd &x,
                            double &) {
        if (a.j >= (int)x.size()) {
            Eigen::VectorXd xnew(a.j + 1);
            xnew.setConstant(std::numeric_limits<double>::quiet_NaN());
            xnew.head(x.size()) = x;
            x.swap(xnew);
        }
        if (a.sense == RowSense::EQ && std::abs(a.aij) > 1e-12) {
            double other = 0.0;
            for (int k = 0; k < (int)a.row.size(); ++k)
                if (k != a.j && k < (int)x.size() && std::isfinite(x(k)))
                    other += a.row(k) * x(k);
            x(a.j) = (a.rhs - other) / a.aij;
        } else if (!std::isfinite(x(a.j)))
            x(a.j) = 0.0;
    }
    static void undo_action(const ActSingletonColElim &a, Eigen::VectorXd &x,
                            double &) {
        if (a.j >= (int)x.size()) {
            Eigen::VectorXd xnew(a.j + 1);
            xnew.setConstant(std::numeric_limits<double>::quiet_NaN());
            xnew.head(x.size()) = x;
            x.swap(xnew);
        }
        if (!std::isfinite(x(a.j)))
            x(a.j) = 0.0;
    }
    static void undo_action(const ActDualFix &a, Eigen::VectorXd &x, double &) {
        if (a.j >= (int)x.size()) {
            Eigen::VectorXd xnew(a.j + 1);
            xnew.setConstant(std::numeric_limits<double>::quiet_NaN());
            xnew.head(x.size()) = x;
            x.swap(xnew);
        }
        if (!std::isfinite(x(a.j)))
            x(a.j) = a.x_fix;
    }
    static void undo_action(const ActRemoveCol &a, Eigen::VectorXd &x,
                            double &) {
        if (a.j >= (int)x.size()) {
            Eigen::VectorXd xnew(a.j + 1);
            xnew.setConstant(std::numeric_limits<double>::quiet_NaN());
            xnew.head(x.size()) = x;
            x.swap(xnew);
        }
        if (!std::isfinite(x(a.j))) {
            if (is_finite(a.l_j) && is_finite(a.u_j))
                x(a.j) = 0.5 * (a.l_j + a.u_j);
            else if (is_finite(a.l_j))
                x(a.j) = a.l_j;
            else if (is_finite(a.u_j))
                x(a.j) = a.u_j;
            else
                x(a.j) = 0.0;
        }
    }

private:
    Options opt_;
    PresolveResult res_;
    double domprop_min_delta_{1e-6};
};

} // namespace presolve
