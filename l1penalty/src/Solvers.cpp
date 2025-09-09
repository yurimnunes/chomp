#include "../include/Solvers.hpp"
#include "gurobi_c++.h"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <limits>
#include <tuple>
#include <vector>

// Definition
GRBEnv GurobiSolver::env;

void GurobiSolver::initializeEnvironment() {
    static bool initialized = false;
    if (!initialized) {
        try {
            GRBEnv env = GRBEnv();
            env.start();
            initialized = true;
        } catch (GRBException &e) {
            std::cerr << "Error code = " << e.getErrorCode() << std::endl;
            std::cerr << e.getMessage() << std::endl;
            throw;
        }
    }
}

// Solve:  minimize 0.5 * λ^T H λ + g^T λ  s.t. λ_i >= 0 for i in bound_idx.
// H should be SPD or PSD; a small diagonal regularization is added for stability.
// Returns: (lambda, converged, iters, r_norm, s_norm)
std::tuple<Eigen::VectorXd, bool, int, double, double>
solveQuadraticNonnegADMM(const Eigen::MatrixXd  &H_in,      // n x n
                         const Eigen::VectorXd  &g,         // n
                         const std::vector<int> &bound_idx, // indices with λ_i >= 0
                         double rho, double eps_abs, double eps_rel, int max_iter,
                         double diag_reg // if <=0, choose automatically
) {
    using std::abs;
    const int n = static_cast<int>(g.size());
    const int m = static_cast<int>(bound_idx.size());

    // Handle trivial unconstrained case quickly
    if (m == 0) {
        // Solve (H + δI) λ = -g
        Eigen::MatrixXd H         = H_in;
        double          mean_diag = H.diagonal().cwiseAbs().mean();
        double delta = (diag_reg > 0.0) ? diag_reg : std::max(1e-12, 1e-8 * (mean_diag > 0.0 ? mean_diag : 1.0));
        H.diagonal().array() += delta;

        Eigen::VectorXd lambda;
        // Try LLT first
        Eigen::LLT<Eigen::MatrixXd> llt(H);
        if (llt.info() == Eigen::Success) {
            lambda = llt.solve(-g);
        } else {
            Eigen::LDLT<Eigen::MatrixXd> ldlt(H);
            lambda = ldlt.solve(-g);
        }
        return {lambda, true, 1, 0.0, 0.0};
    }

    // Build diagonal selector D (n-vector of 0/1) for constrained subset
    Eigen::VectorXd D = Eigen::VectorXd::Zero(n);
    for (int j = 0; j < m; ++j) {
        int i = bound_idx[j];
        if (i >= 0 && i < n) D(i) = 1.0;
    }

    // System matrix W = H + rho * D (diagonal)
    Eigen::MatrixXd W = H_in;
    // Add baseline regularization for numerical stability
    double mean_diag = W.diagonal().cwiseAbs().mean();
    double delta     = (diag_reg > 0.0) ? diag_reg : std::max(1e-12, 1e-8 * (mean_diag > 0.0 ? mean_diag : 1.0));
    W.diagonal().array() += delta + rho * D.array();

    // Pre-factorize W once (we keep rho fixed; you can add rho-adaptation if you want)
    Eigen::LLT<Eigen::MatrixXd>  llt(W);
    bool                         useLLT = (llt.info() == Eigen::Success);
    Eigen::LDLT<Eigen::MatrixXd> ldlt;
    if (!useLLT) {
        ldlt.compute(W);
        if (ldlt.info() != Eigen::Success) {
            // Last resort: nuke with more regularization
            Eigen::MatrixXd W2 = W;
            W2.diagonal().array() += 1e-6;
            ldlt.compute(W2);
        }
    }

    // ADMM variables
    Eigen::VectorXd lambda = Eigen::VectorXd::Zero(n); // primal
    Eigen::VectorXd z      = Eigen::VectorXd::Zero(m); // split var on constrained subset
    Eigen::VectorXd u      = Eigen::VectorXd::Zero(m); // scaled dual

    // helpers to extract/accumulate on constrained subset
    auto take_S = [&](const Eigen::VectorXd &v) {
        Eigen::VectorXd vs(m);
        for (int j = 0; j < m; ++j) vs(j) = v(bound_idx[j]);
        return vs;
    };
    auto add_ST = [&](Eigen::VectorXd &out, const Eigen::VectorXd &w) {
        // out += S^T w  (S selects the bound_idx rows)
        for (int j = 0; j < m; ++j) out(bound_idx[j]) += w(j);
    };

    // stopping thresholds (updated each iter)
    double r_norm = 0.0, s_norm = 0.0;

    // ADMM loop
    for (int it = 1; it <= max_iter; ++it) {
        // λ-update: solve (H + rho*D) λ = -g + rho*S^T(z - u)
        Eigen::VectorXd rhs = -g;
        add_ST(rhs, rho * (z - u));
        if (useLLT)
            lambda = llt.solve(rhs);
        else
            lambda = ldlt.solve(rhs);

        // z-update: projection onto R_+^m
        Eigen::VectorXd lambda_S = take_S(lambda);
        Eigen::VectorXd z_prev   = z;
        z                        = (lambda_S + u).cwiseMax(0.0);

        // u-update
        u += lambda_S - z;

        // Residuals
        r_norm = (lambda_S - z).norm();       // primal residual ||Sλ - z||
        s_norm = (rho * (z - z_prev)).norm(); // dual residual  ||ρ (z - z_prev)||

        // Tolerances
        const double eps_pri = std::sqrt((double)m) * eps_abs + eps_rel * std::max(lambda_S.norm(), z.norm());
        const double eps_dua = std::sqrt((double)n) * eps_abs + eps_rel * (rho * u).norm();

        if (r_norm <= eps_pri && s_norm <= eps_dua) { return {lambda, true, it, r_norm, s_norm}; }
    }
    return {lambda, false, max_iter, r_norm, s_norm};
}