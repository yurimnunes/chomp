#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>

using dvec = Eigen::VectorXd;
using dmat = Eigen::MatrixXd;
using spmat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

struct KKT {
    double stat{0.0}, eq{0.0}, ineq{0.0}, comp{0.0};
};

struct SolverInfo {
    std::string mode{"ip"}; // algorithmic mode (ip, sqp, etc.)
    double step_norm{0.0};
    bool accepted{true};
    bool converged{true};

    double f{0.0};     // objective value
    double theta{0.0}; // merit or filter measure
    double stat{0.0};  // gradient norm (safe_inf_norm(g))
    double ineq{0.0};  // infinity norm of violated inequalities
    double eq{0.0};    // infinity norm of equality violations
    double comp{0.0};  // complementarity measure

    int ls_iters{0};   // line search iterations
    double alpha{0.0}; // step size
    double rho{0.0};   // trust region ratio
    double tr_radius{0.0};
    double delta{0.0}; // trust region radius
    double mu{0.0};    // barrier parameter

    bool shifted_barrier{false}; // whether the barrier was shifted
    double tau_shift{0.0};       // amount of barrier shift
    double bound_shift{0.0};     // amount of bound shift
};