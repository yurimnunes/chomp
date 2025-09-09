#pragma once
#include <Eigen/Dense>
#include <tuple>

class SolverBase {
public:
    virtual ~SolverBase() = default;

    virtual Eigen::VectorXd solveLinearProblem(const Eigen::VectorXd &f, const Eigen::MatrixXd &Aineq,
                                               const Eigen::VectorXd &bineq, const Eigen::MatrixXd &Aeq,
                                               const Eigen::VectorXd &beq, const Eigen::VectorXd &lb,
                                               const Eigen::VectorXd &ub, const Eigen::VectorXd &x0 = Eigen::VectorXd()) = 0;

    virtual std::tuple<Eigen::VectorXd, double, bool>
    solveQuadraticProblem(const Eigen::MatrixXd &H, const Eigen::VectorXd &g, double c, const Eigen::MatrixXd &Aineq,
                          const Eigen::VectorXd &bineq, const Eigen::MatrixXd &Aeq, const Eigen::VectorXd &beq,
                          const Eigen::VectorXd &lb, const Eigen::VectorXd &ub, const Eigen::VectorXd &x0) = 0;
};
