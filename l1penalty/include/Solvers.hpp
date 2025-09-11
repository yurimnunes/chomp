#pragma once
#include "../../include/piqp.h"
#include "../../include/simplex.h"
#include "SolversBase.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <vector>
// include fmt
#include <fmt/core.h>
#include <fmt/ostream.h>
class GenericSolver : public SolverBase {
public:
    static void   initializeEnvironment();

    Eigen::VectorXd solveLinearProblem(const Eigen::VectorXd &f, const Eigen::MatrixXd &Aineq,
                                       const Eigen::VectorXd &bineq, const Eigen::MatrixXd &Aeq,
                                       const Eigen::VectorXd &beq, const Eigen::VectorXd &lb, const Eigen::VectorXd &ub,
                                       const Eigen::VectorXd &x0 = Eigen::VectorXd()) override {
        // ---- Build standard form: minimize c^T v  s.t.  Ā v = b~,  v ≥ 0
        // Variable modeling:
        //  - If lb_j is finite: x_j = lb_j + y_j, y_j ≥ 0
        //      If ub_j finite:   y_j ≤ (ub_j - lb_j)  --> add equality: y_j + s^U_j = (ub_j - lb_j), s^U_j ≥ 0
        //  - If lb_j = -inf: x_j = y^+_j - y^-_j, y^+, y^- ≥ 0
        //      If ub_j finite:   y^+_j - y^-_j ≤ ub_j  --> add inequality + slack
        //
        // Inequalities Aineq x ≤ bineq become equalities via slack s^I ≥ 0.
        //
        // Objective:
        //  - Shift constants f^T lb added to obj_shift (ignored for x return).
        //  - y_j gets cost f_j; (y^+_j, y^-_j) get ( +f_j, -f_j ); slacks cost 0.

        const int n          = static_cast<int>(f.size());
        auto      is_neg_inf = [](double v) { return !std::isfinite(v) && v < 0; };
        auto      is_pos_inf = [](double v) { return !std::isfinite(v) && v > 0; };

        // Maps from original var j to indices in standard vars
        struct VarMap {
            bool   shifted = false;                                   // true: x = lb + y
            int    y       = -1;                                      // index of y (if shifted)
            int    y_pos = -1, y_neg = -1;                            // indices if split
            bool   has_ub  = false;                                   // whether an upper bound row was created
            double shift   = 0.0;                                     // lb if shifted, else 0
            double ub_span = std::numeric_limits<double>::infinity(); // (ub - lb) if shifted and finite
        };

        std::vector<VarMap> vm(n);
        int                 nv        = 0;   // number of standard variables (grows)
        double              obj_shift = 0.0; // f^T * lb (and other substitutions)

        // 1) Create modeling vars
        for (int j = 0; j < n; ++j) {
            const double lj = lb.size() == n ? lb(j) : -std::numeric_limits<double>::infinity();
            const double uj = ub.size() == n ? ub(j) : std::numeric_limits<double>::infinity();

            if (std::isfinite(lj) && std::isfinite(uj) && uj < lj) {
                // Infeasible bound; return a vector of NaNs (or throw)
                return Eigen::VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
            }

            if (!is_neg_inf(lj)) {
                // x = lb + y, y >= 0
                vm[j].shifted = true;
                vm[j].y       = nv++;
                vm[j].shift   = lj;
                obj_shift += f(j) * lj;

                if (!is_pos_inf(uj)) {
                    vm[j].has_ub  = true;
                    vm[j].ub_span = uj - lj; // RHS for the upper-bound equality with slack
                    if (vm[j].ub_span < 0.0) {
                        return Eigen::VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
                    }
                    // we'll add a slack for this later as an equality row
                }
            } else {
                // Free to -inf on lower: split x = y^+ - y^-, y^+, y^- >= 0
                vm[j].shifted = false;
                vm[j].y_pos   = nv++;
                vm[j].y_neg   = nv++;
                vm[j].shift   = 0.0;

                if (!is_pos_inf(uj)) {
                    vm[j].has_ub  = true;
                    vm[j].ub_span = uj; // inequality RHS for y^+ - y^- ≤ ub
                }
            }
        }

        // Count rows to assemble Ā v = b~
        int m_eq       = static_cast<int>(Aeq.rows());
        int m_ineq     = static_cast<int>(Aineq.rows());
        int rows_upper = 0; // for upper bound equalities
        for (int j = 0; j < n; ++j)
            if (vm[j].has_ub && vm[j].shifted) rows_upper++;

        // slack counts
        int n_slack_ineq        = m_ineq; // one slack per ≤ row
        int n_slack_upper_split = 0;      // for split-vars' ub inequalities
        for (int j = 0; j < n; ++j)
            if (vm[j].has_ub && !vm[j].shifted) n_slack_upper_split++;

        // Total variables so far: nv + slacks (ineq + upper-split + upper-shifted)
        int slack_ineq_offset          = nv;
        int slack_upper_split_offset   = slack_ineq_offset + n_slack_ineq;
        int slack_upper_shifted_offset = slack_upper_split_offset + n_slack_upper_split;
        int nv_total                   = slack_upper_shifted_offset + rows_upper;

        // Total equality rows:
        //   - original equalities m_eq
        //   - converted inequalities m_ineq  (with slacks)
        //   - upper bound equalities for shifted vars rows_upper
        //   - upper bound inequalities for split vars become equalities with slacks (count = n_slack_upper_split)
        int m_total = m_eq + m_ineq + rows_upper + n_slack_upper_split;

        Eigen::MatrixXd Astd = Eigen::MatrixXd::Zero(m_total, nv_total);
        Eigen::VectorXd bstd = Eigen::VectorXd::Zero(m_total);
        Eigen::VectorXd cstd = Eigen::VectorXd::Zero(nv_total);

        // 2) Objective coefficients into standard vars
        for (int j = 0; j < n; ++j) {
            if (vm[j].shifted) {
                cstd(vm[j].y) += f(j);
            } else {
                cstd(vm[j].y_pos) += f(j);
                cstd(vm[j].y_neg) += -f(j);
            }
        }
        // slacks have zero cost

        // 3) Equality rows: Aeq (shifted RHS)
        int row = 0;
        if (m_eq > 0) {
            for (int i = 0; i < m_eq; ++i, ++row) {
                double rhs = beq(i);
                // account for shifts: subtract Aeq(i,j)*lb_j
                for (int j = 0; j < n; ++j) {
                    const double aij = Aeq(i, j);
                    if (vm[j].shifted) {
                        rhs -= aij * vm[j].shift;
                        if (aij != 0.0) Astd(row, vm[j].y) += aij;
                    } else {
                        if (aij != 0.0) {
                            Astd(row, vm[j].y_pos) += aij;
                            Astd(row, vm[j].y_neg) += -aij;
                        }
                    }
                }
                bstd(row) = rhs;
            }
        }

        // 4) Inequality rows become equalities with slack ≥ 0: Aineq x + s = bineq
        if (m_ineq > 0) {
            for (int i = 0; i < m_ineq; ++i, ++row) {
                double rhs = bineq(i);
                for (int j = 0; j < n; ++j) {
                    const double aij = Aineq(i, j);
                    if (vm[j].shifted) {
                        rhs -= aij * vm[j].shift;
                        if (aij != 0.0) Astd(row, vm[j].y) += aij;
                    } else {
                        if (aij != 0.0) {
                            Astd(row, vm[j].y_pos) += aij;
                            Astd(row, vm[j].y_neg) += -aij;
                        }
                    }
                }
                // inequality slack
                Astd(row, slack_ineq_offset + i) = 1.0; // s^I_i
                bstd(row)                        = rhs;
            }
        }

        // 5) Upper bounds:
        //    (a) shifted: y_j + s^U = (ub - lb)
        int upper_shift_row = 0;
        for (int j = 0; j < n; ++j)
            if (vm[j].has_ub && vm[j].shifted) {
                Astd(row, vm[j].y)                                      = 1.0;
                Astd(row, slack_upper_shifted_offset + upper_shift_row) = 1.0;           // s^U (>=0)
                bstd(row)                                               = vm[j].ub_span; // ub-lb
                ++upper_shift_row;
                ++row;
            }

        //    (b) split: y^+ - y^- + s = ub
        int upper_split_row = 0;
        for (int j = 0; j < n; ++j)
            if (vm[j].has_ub && !vm[j].shifted) {
                Astd(row, vm[j].y_pos)                                = 1.0;
                Astd(row, vm[j].y_neg)                                = -1.0;
                Astd(row, slack_upper_split_offset + upper_split_row) = 1.0;           // s^U_split (>=0)
                bstd(row)                                             = vm[j].ub_span; // ub
                ++upper_split_row;
                ++row;
            }

        // 6) Call your solver (public API)
        RevisedSimplex simplex; // (optionally pass options)
        LPSolution     sol = simplex.solve(Astd, bstd, cstd);

        if (sol.status != LPSolution::Status::Optimal) {
            // Choose your failure policy (throw / NaNs). Here: NaNs.
            return Eigen::VectorXd::Constant((int)f.size(), std::numeric_limits<double>::quiet_NaN());
        }
        const Eigen::VectorXd &v = sol.x; // solution in standard variables

        // 7) Map back to original x
        Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
        for (int j = 0; j < n; ++j) {
            // `vm[j]` is the VarMap built in the standard-form step
            if (vm[j].shifted) {
                const int    iy = vm[j].y;
                const double y  = (iy >= 0 && iy < v.size()) ? v(iy) : 0.0;
                x(j)            = vm[j].shift + y;
            } else {
                const int    ip = vm[j].y_pos, in = vm[j].y_neg;
                const double yp = (ip >= 0 && ip < v.size()) ? v(ip) : 0.0;
                const double yn = (in >= 0 && in < v.size()) ? v(in) : 0.0;
                x(j)            = yp - yn;
            }
        }

        (void)x0; // warm-start currently unused
        return x;
    }

    std::tuple<Eigen::VectorXd, double, bool>
    solveQuadraticProblem(const Eigen::MatrixXd &H, const Eigen::VectorXd &g, double c, const Eigen::MatrixXd &Aineq,
                          const Eigen::VectorXd &bineq, const Eigen::MatrixXd &Aeq, const Eigen::VectorXd &beq,
                          const Eigen::VectorXd &lb, const Eigen::VectorXd &ub, const Eigen::VectorXd &x0) override {
        const int n = static_cast<int>(x0.size());

        // Regularize Hessian for positive semi-definiteness
        Eigen::MatrixXd                                H_reg = H;
        double                                         reg   = 1e-8 * H.norm();
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
        if (es.info() == Eigen::Success && es.eigenvalues().minCoeff() < 0) {
            reg = std::max(reg, -es.eigenvalues().minCoeff() + 1e-8);
            H_reg.diagonal().array() += reg;
        }

        // Convert H_reg to sparse matrix
        Eigen::SparseMatrix<double>         P(n, n);
        std::vector<Eigen::Triplet<double>> P_triplets;
        P_triplets.reserve(n * (n + 1) / 2); // Lower triangular part
        for (int j = 0; j < n; ++j) {
            for (int i = j; i < n; ++i) {            // Lower triangle including diagonal
                if (std::abs(H_reg(i, j)) > 1e-12) { // Skip near-zero entries
                    P_triplets.emplace_back(i, j, H_reg(i, j));
                }
            }
        }
        P.setFromTriplets(P_triplets.begin(), P_triplets.end());
        P.makeCompressed();

        // Set q = g
        Eigen::VectorXd q = g;

        // Set up equality constraints
        std::optional<Eigen::SparseMatrix<double>> A = std::nullopt;
        std::optional<Eigen::VectorXd>             b = std::nullopt;
        if (Aeq.rows() > 0) {
            Eigen::SparseMatrix<double>         A_sparse(Aeq.rows(), Aeq.cols());
            std::vector<Eigen::Triplet<double>> A_triplets;
            A_triplets.reserve(Aeq.rows() * Aeq.cols());
            for (int i = 0; i < Aeq.rows(); ++i) {
                for (int j = 0; j < Aeq.cols(); ++j) {
                    if (std::abs(Aeq(i, j)) > 1e-12) { A_triplets.emplace_back(i, j, Aeq(i, j)); }
                }
            }
            A_sparse.setFromTriplets(A_triplets.begin(), A_triplets.end());
            A_sparse.makeCompressed();
            A = A_sparse;
            b = beq;
        }

        // Set up inequality constraints: Aineq x <= bineq, lb <= x <= ub
        int                                 num_ineq = Aineq.rows() + 2 * n;
        Eigen::SparseMatrix<double>         G(num_ineq, n);
        std::vector<Eigen::Triplet<double>> G_triplets;
        G_triplets.reserve(Aineq.rows() * n + 2 * n);

        // Add Aineq x <= bineq
        for (int i = 0; i < Aineq.rows(); ++i) {
            for (int j = 0; j < n; ++j) {
                if (std::abs(Aineq(i, j)) > 1e-12) { G_triplets.emplace_back(i, j, Aineq(i, j)); }
            }
        }

        // Add bound constraints: x >= lb, -x >= -ub
        for (int i = 0; i < n; ++i) {
            G_triplets.emplace_back(Aineq.rows() + i, i, 1.0);      // x_i >= lb_i
            G_triplets.emplace_back(Aineq.rows() + n + i, i, -1.0); // -x_i >= -ub_i
        }
        G.setFromTriplets(G_triplets.begin(), G_triplets.end());
        G.makeCompressed();

        Eigen::VectorXd h(num_ineq);
        h.head(Aineq.rows())       = bineq;
        h.segment(Aineq.rows(), n) = lb;
        h.tail(n)                  = -ub;

        // Handle non-finite bounds
        for (int i = 0; i < num_ineq; ++i) {
            h(i) = std::isfinite(h(i)) ? h(i) : (i < Aineq.rows() ? 1e30 : (i < Aineq.rows() + n ? -1e30 : 1e30));
        }

        // Set up PIQPSolver
        piqp::PIQPSettings settings;
        settings.eps_abs  = 1e-8;
        settings.eps_rel  = 1e-8;
        settings.max_iter = 100;
        settings.verbose  = false; // Enable for debugging
        piqp::PIQPSolver solver(settings);

        // Setup and solve
        solver.setup(P, q, A, b, G, h);
        solver.warm_start(x0);
        piqp::PIQPResult result = solver.solve();

        // Extract solution
        Eigen::VectorXd x       = result.x;
        double          obj_val = 0.5 * x.dot(H * x) + g.dot(x) + c; // Use original H for objective
        bool            success = (result.status == "solved");

        return {x, obj_val, success};
    }
};

std::tuple<Eigen::VectorXd, bool, int, double, double>
solveQuadraticNonnegADMM(const Eigen::MatrixXd  &H_in,      // n x n
                         const Eigen::VectorXd  &g,         // n
                         const std::vector<int> &bound_idx, // indices with λ_i >= 0
                         double rho = 1e-1, double eps_abs = 1e-6, double eps_rel = 1e-4, int max_iter = 500,
                         double diag_reg = 0.0);