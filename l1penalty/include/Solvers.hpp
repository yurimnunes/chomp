#pragma once
#include "gurobi_c++.h"
#include "SolversBase.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "../../include/piqp.h"
class GurobiSolver : public SolverBase {
public:
    static GRBEnv env; // Static environment shared by all instances
    static void initializeEnvironment();

    Eigen::VectorXd solveLinearProblem(const Eigen::VectorXd &f, const Eigen::MatrixXd &Aineq,
                                       const Eigen::VectorXd &bineq, const Eigen::MatrixXd &Aeq,
                                       const Eigen::VectorXd &beq, const Eigen::VectorXd &lb,
                                       const Eigen::VectorXd &ub, const Eigen::VectorXd &x0 = Eigen::VectorXd()) override {
        initializeEnvironment();

        GRBModel model(env);

        int     n = lb.size();
        GRBVar *x = model.addVars(lb.data(), ub.data(), nullptr, nullptr, nullptr, n);

        // Set initial solution
        if (x0.size() == n) {
            for (int i = 0; i < n; ++i) { x[i].set(GRB_DoubleAttr_Start, x0[i]); }
        }

        // Set objective
        GRBLinExpr expr = 0;
        for (int i = 0; i < n; ++i) { expr += f[i] * x[i]; }
        model.setObjective(expr, GRB_MINIMIZE);

        // Add inequality constraints
        if (Aineq.rows() > 0) {
            for (int i = 0; i < Aineq.rows(); ++i) {
                GRBLinExpr aineq_expr = 0;
                for (int j = 0; j < n; ++j) { aineq_expr += Aineq(i, j) * x[j]; }
                model.addConstr(aineq_expr <= bineq[i]);
            }
        }

        // Add equality constraints
        if (Aeq.rows() > 0) {
            for (int i = 0; i < Aeq.rows(); ++i) {
                GRBLinExpr aeq_expr = 0;
                for (int j = 0; j < n; ++j) { aeq_expr += Aeq(i, j) * x[j]; }
                model.addConstr(aeq_expr == beq[i]);
            }
        }

        model.set(GRB_IntParam_OutputFlag, 0);
        model.optimize();

        Eigen::VectorXd result(n);
        for (int i = 0; i < n; ++i) { result[i] = x[i].get(GRB_DoubleAttr_X); }
        delete[] x;

        return result;
    }

    std::tuple<Eigen::VectorXd, double, bool> solveQuadraticProblem(
    const Eigen::MatrixXd &H, const Eigen::VectorXd &g, double c,
    const Eigen::MatrixXd &Aineq, const Eigen::VectorXd &bineq,
    const Eigen::MatrixXd &Aeq, const Eigen::VectorXd &beq,
    const Eigen::VectorXd &lb, const Eigen::VectorXd &ub,
    const Eigen::VectorXd &x0) override {
    const int n = static_cast<int>(x0.size());

    // Regularize Hessian for positive semi-definiteness
    Eigen::MatrixXd H_reg = H;
    double reg = 1e-8 * H.norm();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
    if (es.info() == Eigen::Success && es.eigenvalues().minCoeff() < 0) {
        reg = std::max(reg, -es.eigenvalues().minCoeff() + 1e-8);
        H_reg.diagonal().array() += reg;
    }

    // Convert H_reg to sparse matrix
    Eigen::SparseMatrix<double> P(n, n);
    std::vector<Eigen::Triplet<double>> P_triplets;
    P_triplets.reserve(n * (n + 1) / 2); // Lower triangular part
    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) { // Lower triangle including diagonal
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
    std::optional<Eigen::VectorXd> b = std::nullopt;
    if (Aeq.rows() > 0) {
        Eigen::SparseMatrix<double> A_sparse(Aeq.rows(), Aeq.cols());
        std::vector<Eigen::Triplet<double>> A_triplets;
        A_triplets.reserve(Aeq.rows() * Aeq.cols());
        for (int i = 0; i < Aeq.rows(); ++i) {
            for (int j = 0; j < Aeq.cols(); ++j) {
                if (std::abs(Aeq(i, j)) > 1e-12) {
                    A_triplets.emplace_back(i, j, Aeq(i, j));
                }
            }
        }
        A_sparse.setFromTriplets(A_triplets.begin(), A_triplets.end());
        A_sparse.makeCompressed();
        A = A_sparse;
        b = beq;
    }

    // Set up inequality constraints: Aineq x <= bineq, lb <= x <= ub
    int num_ineq = Aineq.rows() + 2 * n;
    Eigen::SparseMatrix<double> G(num_ineq, n);
    std::vector<Eigen::Triplet<double>> G_triplets;
    G_triplets.reserve(Aineq.rows() * n + 2 * n);

    // Add Aineq x <= bineq
    for (int i = 0; i < Aineq.rows(); ++i) {
        for (int j = 0; j < n; ++j) {
            if (std::abs(Aineq(i, j)) > 1e-12) {
                G_triplets.emplace_back(i, j, Aineq(i, j));
            }
        }
    }

    // Add bound constraints: x >= lb, -x >= -ub
    for (int i = 0; i < n; ++i) {
        G_triplets.emplace_back(Aineq.rows() + i, i, 1.0);        // x_i >= lb_i
        G_triplets.emplace_back(Aineq.rows() + n + i, i, -1.0);  // -x_i >= -ub_i
    }
    G.setFromTriplets(G_triplets.begin(), G_triplets.end());
    G.makeCompressed();

    Eigen::VectorXd h(num_ineq);
    h.head(Aineq.rows()) = bineq;
    h.segment(Aineq.rows(), n) = lb;
    h.tail(n) = -ub;

    // Handle non-finite bounds
    for (int i = 0; i < num_ineq; ++i) {
        h(i) = std::isfinite(h(i)) ? h(i) : (i < Aineq.rows() ? 1e30 : (i < Aineq.rows() + n ? -1e30 : 1e30));
    }

    // Set up PIQPSolver
    piqp::PIQPSettings settings;
    settings.eps_abs = 1e-8;
    settings.eps_rel = 1e-8;
    settings.max_iter = 100;
    settings.verbose = false; // Enable for debugging
    piqp::PIQPSolver solver(settings);

    // Setup and solve
    solver.setup(P, q, A, b, G, h);
    solver.warm_start(x0);
    piqp::PIQPResult result = solver.solve();

    // Extract solution
    Eigen::VectorXd x = result.x;
    double obj_val = 0.5 * x.dot(H * x) + g.dot(x) + c; // Use original H for objective
    bool success = (result.status == "solved");

    return {x, obj_val, success};
}
};

                                std::tuple<Eigen::VectorXd, bool, int, double, double>
solveQuadraticNonnegADMM(
    const Eigen::MatrixXd &H_in,           // n x n
    const Eigen::VectorXd &g,              // n
    const std::vector<int> &bound_idx,     // indices with λ_i >= 0
    double rho       = 1e-1,
    double eps_abs   = 1e-6,
    double eps_rel   = 1e-4,
    int    max_iter  = 500,
    double diag_reg  = 0.0  );