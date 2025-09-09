#pragma once
#include "gurobi_c++.h"
#include "SolversBase.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <vector>

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

    std::tuple<Eigen::VectorXd, double, bool>
    solveQuadraticProblem(const Eigen::MatrixXd &H, const Eigen::VectorXd &g, double c, const Eigen::MatrixXd &Aineq,
                          const Eigen::VectorXd &bineq, const Eigen::MatrixXd &Aeq, const Eigen::VectorXd &beq,
                          const Eigen::VectorXd &lb, const Eigen::VectorXd &ub, const Eigen::VectorXd &x0) override {

        initializeEnvironment();

        GRBModel model(env);
        int      n = x0.size();

        GRBVar *x = model.addVars(lb.data(), ub.data(), nullptr, nullptr, nullptr, n);

        // Set objective
        GRBQuadExpr obj = c;
        for (int i = 0; i < n; ++i) {
            obj += g[i] * x[i];
            for (int j = 0; j < n; ++j) { obj += 0.5 * x[i] * H(i, j) * x[j]; }
        }
        model.setObjective(obj, GRB_MINIMIZE);

        // Add inequality constraints
        if (Aineq.rows() > 0) {
            for (int i = 0; i < Aineq.rows(); ++i) {
                GRBLinExpr constr_expr = 0;
                for (int j = 0; j < n; ++j) { constr_expr += Aineq(i, j) * x[j]; }
                model.addConstr(constr_expr <= bineq[i]);
            }
        }

        // Add equality constraints
        if (Aeq.rows() > 0) {
            for (int i = 0; i < Aeq.rows(); ++i) {
                GRBLinExpr constr_expr = 0;
                for (int j = 0; j < n; ++j) { constr_expr += Aeq(i, j) * x[j]; }
                model.addConstr(constr_expr == beq[i]);
            }
        }

        model.set(GRB_IntParam_OutputFlag, 0);
        model.optimize();

        Eigen::VectorXd result(n);
        for (int i = 0; i < n; ++i) { result[i] = x[i].get(GRB_DoubleAttr_X); }
        double objVal  = model.get(GRB_DoubleAttr_ObjVal);
        bool   success = model.get(GRB_IntAttr_Status) == GRB_OPTIMAL;

        delete[] x;

        return {result, objVal, success};
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