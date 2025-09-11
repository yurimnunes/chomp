#include "gurobi_c++.h"

#include "../include/Definitions.hpp"
#include "../include/Helpers.hpp"
#include "../include/PolynomialVector.hpp"
#include "../include/Solvers.hpp"
#include "../include/l1Fun.hpp"
#include "Eigen/Core"

#include "../include/TRModel.hpp"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <fmt/core.h>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

// =================== CONSTRAINT IDENTIFICATION ===================

std::tuple<std::vector<bool>, std::vector<bool>, std::vector<bool>>
l1_identify_constraints(CModel &cmodel, const Eigen::VectorXd &x, const Eigen::VectorXd &lb, 
                       const Eigen::VectorXd &ub, double epsilon) {
    const int ncon = cmodel->size();
    const int nvar = static_cast<int>(x.size());

    std::vector<bool> is_eactive(ncon);
    std::vector<bool> lb_active(nvar);
    std::vector<bool> ub_active(nvar);

    const double tol_bounds = std::sqrt(std::numeric_limits<double>::epsilon());
    const double epsilon_bounds = std::min(tol_bounds, epsilon / 16.0);

    for (int k = 0; k < ncon; ++k) {
        is_eactive[k] = std::abs((*cmodel)[k].c) < epsilon;
    }
    
    for (int i = 0; i < nvar; ++i) {
        lb_active[i] = (x(i) - lb(i)) < epsilon_bounds;
        ub_active[i] = (ub(i) - x(i)) < epsilon_bounds;
    }
    
    return {is_eactive, lb_active, ub_active};
}

// =================== GRADIENT AND HESSIAN COMPUTATION ===================

Eigen::VectorXd l1_pseudo_gradient_general(
    TRModelPtr &fmodel,
    CModel &cmodel,
    double mu,
    const Eigen::VectorXd &s,
    const Eigen::VectorXd &is_eactive
) {
    const int dim = static_cast<int>(fmodel->g().size());
    Eigen::VectorXd vg = Eigen::VectorXd::Zero(dim);
    const double tol = 1e-12;

    for (int n = 0; n < cmodel->size(); ++n) {
        if (is_eactive(n) <= 0.5) continue;

        const auto &cn = (*cmodel)[n];
        const double c_value = cn.c + cn.g.dot(s) + 0.5 * (s.transpose() * (cn.H * s)).value();

        if (c_value > tol) {
            vg.noalias() += (cn.g + cn.H * s);
        }
    }

    const Eigen::VectorXd pg_base = fmodel->g() + fmodel->H() * s;
    return pg_base + mu * vg;
}

Eigen::MatrixXd l1Hessian(TRModelPtr &fmodel, CModel &cmodel, double mu, const Eigen::VectorXd &s) {
    Eigen::MatrixXd Hfx = fmodel->H();
    Eigen::MatrixXd Hc = Eigen::MatrixXd::Zero(Hfx.rows(), Hfx.cols());
    
    for (int n = 0; n < cmodel->size(); ++n) {
        const auto &cn = (*cmodel)[n];
        const double c_value = cn.c + cn.g.dot(s) + 0.5 * (s.transpose() * (cn.H * s)).value();
        if (c_value > 0) {
            Hc += cn.H;
        }
    }
    return Hfx + mu * Hc;
}

Eigen::MatrixXd l1PseudoHessian(TRModelPtr &fmodel, CModel &cmodel, double mu, 
                               const Eigen::VectorXd &is_eactive, const Eigen::VectorXd &multipliers) {
    const int dim = fmodel->H().rows();
    Eigen::MatrixXd Hfx = fmodel->H();
    Eigen::MatrixXd Hc = Eigen::MatrixXd::Zero(dim, dim);
    Eigen::MatrixXd Hm = Eigen::MatrixXd::Zero(dim, dim);
    
    int mult_i = 0;
    for (int n = 0; n < cmodel->size(); ++n) {
        if (is_eactive(n) == 1) {
            Hm += multipliers(mult_i) * (*cmodel)[n].H;
            mult_i++;
        } else if ((*cmodel)[n].c > 0) {
            Hc += (*cmodel)[n].H;
        }
    }
    return Hfx + Hm + (mu * Hc);
}

// =================== BOUNDS AND BREAKPOINTS ===================

std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>
boundsBreakpoints(const Eigen::VectorXd &x, const Eigen::VectorXd &lb, 
                 const Eigen::VectorXd &ub, const Eigen::VectorXd &d) {
    const int n = static_cast<int>(x.size());
    Eigen::VectorXd t_lb(n), t_ub(n);

    for (int i = 0; i < n; ++i) {
        t_lb(i) = (d(i) < 0.0) ? (lb(i) - x(i)) / d(i) : std::numeric_limits<double>::infinity();
        t_ub(i) = (d(i) > 0.0) ? (ub(i) - x(i)) / d(i) : std::numeric_limits<double>::infinity();
        
        // Keep only forward (t > 0) breakpoints
        if (!(t_lb(i) > 0.0) || !std::isfinite(t_lb(i))) t_lb(i) = std::numeric_limits<double>::infinity();
        if (!(t_ub(i) > 0.0) || !std::isfinite(t_ub(i))) t_ub(i) = std::numeric_limits<double>::infinity();
    }
    
    return {t_lb, t_ub, t_lb.cwiseMin(t_ub)};
}

double infinityTRRadiusBreakpoint(const Eigen::VectorXd &d, double radius, 
                                 const Eigen::VectorXd &s0) {
    assert(radius > 0.0);
    
    const Eigen::VectorXd s0_eff = (s0.size() == 0) ? Eigen::VectorXd::Zero(d.size()) : s0;
    
    if (s0_eff.lpNorm<Eigen::Infinity>() > radius + 1e-12) {
        std::cerr << "Warning: Initial point out of TR radius\n";
    }

    const auto inc = d.array() > 0.0;
    const auto dec = d.array() < 0.0;

    const Eigen::ArrayXd inc_max = (radius - s0_eff.array()) / d.array();
    const Eigen::ArrayXd dec_max = (-radius - s0_eff.array()) / d.array();

    double t_inc = std::numeric_limits<double>::infinity();
    for (int i = 0; i < inc_max.size(); ++i) {
        if (inc(i)) t_inc = std::min(t_inc, inc_max(i));
    }

    double t_dec = std::numeric_limits<double>::infinity();
    for (int i = 0; i < dec_max.size(); ++i) {
        if (dec(i)) t_dec = std::min(t_dec, dec_max(i));
    }

    const double t = std::min(t_inc, t_dec);
    return std::max(0.0, t);
}

// =================== DESCENT PREDICTION ===================

double predictDescent(TRModelPtr &fmodel, CModel &con_model, const Eigen::VectorXd &s, 
                     double mu, const std::vector<bool> &ind_eactive) {
    const int n_constraints = con_model->size();
    std::vector<bool> local_ind_eactive(n_constraints, false);

    if (!ind_eactive.empty()) {
        local_ind_eactive = ind_eactive;
    }

    double f_change = 0.5 * (s.transpose() * fmodel->H() * s).value() + 
                     (fmodel->g().transpose() * s).value();
    double c_change = 0;
    
    for (int k = 0; k < n_constraints; ++k) {
        if (!local_ind_eactive[k]) {
            const auto &ck = (*con_model)[k];
            double this_change = 0.5 * (s.transpose() * ck.H * s).value() + 
                                ck.g.transpose() * s;

            if (ck.c > 0) {
                this_change = std::max(this_change, -ck.c);
                c_change += this_change;
            } else if (ck.c + this_change > 0) {
                c_change += (ck.c + this_change);
            }
        }
    }
    return -(f_change + mu * c_change);
}

// =================== CONSTRAINT UTILITIES ===================

std::tuple<Eigen::VectorXd, Eigen::MatrixXd>
constraintValuesAndGradientsAtPoint(CModel &cmodelUnfiltered, const std::vector<bool> &is_eactive, 
                                   const Eigen::VectorXd &s) {
    auto cmodel = cmodelUnfiltered->filter(is_eactive);
    const int n_vars = s.size();
    const int n_considered = cmodel.size();

    Eigen::VectorXd nphi(n_considered);
    Eigen::MatrixXd nA(n_vars, n_considered);
    nA.setZero();

    for (int n = 0; n < cmodel.size(); ++n) {
        const auto &cn = cmodel[n];
        nphi(n) = cn.c + cn.g.dot(s) + 0.5 * (s.transpose() * cn.H * s).value();
        nA.col(n) = cn.g + cn.H * s;
    }
    return {nphi, nA};
}

// =================== FEASIBILITY CORRECTION ===================

Eigen::VectorXd l1FeasibilityCorrection(CModel &cmodel, const std::vector<bool> &is_eactive,
                                       const Eigen::VectorXd &tr_center, const Eigen::VectorXd &xh,
                                       double radius, const Eigen::VectorXd &lb, const Eigen::VectorXd &ub) {
    Eigen::VectorXd h = xh - tr_center;
    Eigen::VectorXd is_eactive_vec = active2eigen(is_eactive);

    if (is_eactive_vec.sum() == 0) {
        return xh;
    }

    auto [c, A] = constraintValuesAndGradientsAtPoint(cmodel, is_eactive, h);
    Eigen::MatrixXd H = A * A.transpose();
    Eigen::VectorXd g = A * c;
    
    Eigen::VectorXd v_lb = (Eigen::VectorXd::Constant(h.size(), -radius) - h)
                          .cwiseMax(lb - tr_center - h);
    Eigen::VectorXd v_ub = (Eigen::VectorXd::Constant(h.size(), radius) - h)
                          .cwiseMin(ub - tr_center - h);
    Eigen::VectorXd v0 = 0.5 * (v_lb + v_ub);

    GurobiSolver solver;
    auto [v, obj, optimal] = solver.solveQuadraticProblem(
        H, g, 0, 
        Eigen::VectorXd::Zero(0), Eigen::VectorXd::Zero(0),
        Eigen::VectorXd::Zero(0), Eigen::VectorXd::Zero(0), 
        v_lb, v_ub, v0
    );
    
    return xh + v;
}

// =================== MULTIPLIER ESTIMATION ===================

Eigen::VectorXd estimateMultipliers(TRModelPtr &fmodel, CModel &cmodel, const Eigen::VectorXd &x,
                                   double mu, const std::vector<bool> &is_eactive,
                                   const Eigen::VectorXd &lb, const Eigen::VectorXd &ub) {
    Eigen::VectorXd is_act_vec = active2eigen(is_eactive);
    const int dim = static_cast<int>(x.size());

    if (is_act_vec.sum() == 0) {
        return Eigen::VectorXd::Zero(0);
    }

    // Pseudo-gradient at current point
    Eigen::VectorXd s0 = Eigen::VectorXd::Zero(dim);
    Eigen::VectorXd pg = l1_pseudo_gradient_general(fmodel, cmodel, mu, s0, is_act_vec);

    // Active nonlinear constraints
    auto filtered = (*cmodel)[is_eactive];
    Eigen::MatrixXd A = filtered.g();
    const int n_act_nl = static_cast<int>(A.rows());

    if (n_act_nl == 0) {
        return Eigen::VectorXd::Zero(0);
    }

    // Bound activity detection
    const double maxDiff = std::max(1.0, (ub - lb).array().abs().maxCoeff());
    const double tol_bounds = std::min(1e-5, 1e-3 * maxDiff);
    const double grad_norm = std::max(pg.norm(), 1.0);
    const double adaptive_tol = std::max(tol_bounds, 1e-8 * grad_norm);
    const double abs_floor = 1e-12;

    std::vector<int> lb_idxs, ub_idxs;
    for (int i = 0; i < dim; ++i) {
        if (x(i) - lb(i) < std::max(adaptive_tol, abs_floor)) lb_idxs.push_back(i);
        if (ub(i) - x(i) < std::max(adaptive_tol, abs_floor)) ub_idxs.push_back(i);
    }

    const int n_lb = static_cast<int>(lb_idxs.size());
    const int n_ub = static_cast<int>(ub_idxs.size());
    const int n_rows = n_act_nl + n_lb + n_ub;

    // Extended constraint matrix
    Eigen::MatrixXd A_ext = Eigen::MatrixXd::Zero(n_rows, dim);
    if (n_act_nl > 0) A_ext.topRows(n_act_nl) = A;

    int r = n_act_nl;
    for (int i : lb_idxs) { A_ext(r, i) = -1.0; ++r; }
    for (int i : ub_idxs) { A_ext(r, i) = 1.0; ++r; }

    // Solve least-squares problem
    Eigen::VectorXd lambda;
    if (A_ext.rows() == 0) {
        lambda = Eigen::VectorXd::Zero(n_rows);
    } else {
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(
            A_ext.transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV
        );
        const double smax = (svd.singularValues().size() > 0) ? svd.singularValues()(0) : 1.0;
        const double threshold = std::max(1e-12, 1e-8 * smax);
        svd.setThreshold(threshold);

        lambda = -svd.solve(pg);

        // Enforce nonnegativity for bound multipliers
        for (int i = n_act_nl; i < n_rows; ++i) {
            lambda(i) = std::max(0.0, lambda(i));
        }
    }

    return lambda.head(n_act_nl);
}

bool validateMultipliers(const Eigen::VectorXd &multipliers, double mu, double tolerance) {
    if (tolerance < 0) {
        tolerance = 10 * std::max(1e-12, 1e-10 * mu);
    }

    bool has_negative = (multipliers.array() < -tolerance).any();
    bool has_excessive = (multipliers.array() > mu + tolerance).any();

    return !has_negative && !has_excessive;
}

// =================== MEASURE COMPUTATION ===================

std::tuple<double, Eigen::VectorXd, Eigen::VectorXd> 
correctMeasureComputation(const Eigen::VectorXd &pg, const Eigen::MatrixXd &G,
                         double mu, const Eigen::VectorXd &lb, const Eigen::VectorXd &ub,
                         const Eigen::VectorXd &dt) {
    const int dim = static_cast<int>(pg.size());
    const int n_aconstraints = (G.rows() > 0) ? static_cast<int>(G.rows()) : 0;

    if (dt.size() == 0) {
        return {0.0, Eigen::VectorXd::Zero(0), Eigen::VectorXd::Zero(0)};
    }

    assert(dt.size() == dim + n_aconstraints);
    const Eigen::VectorXd d = dt.head(dim);
    Eigen::VectorXd theta;
    if (n_aconstraints > 0) {
        theta = (G * d).cwiseMax(0.0);
    } else {
        theta = Eigen::VectorXd::Zero(0);
    }

    double measure = -(pg.dot(d) + mu * theta.lpNorm<1>());
    return {measure, d, theta};
}

// =================== LINE SEARCH ===================

std::tuple<double, Eigen::VectorXd> 
lineSearchCG(TRModelPtr &fmodel, CModel &cmodel, double mu, const Eigen::VectorXd &s0, double max_t) {
    const int dim = static_cast<int>(fmodel->g().size());
    const int n_constraints = static_cast<int>(cmodel->size());
    const double eps_a = 1e-14;
    const double eps_t = 1e-12;
    const double T_MAX = std::max(0.0, max_t);

    // Initial active set
    Eigen::VectorXd gv = Eigen::VectorXd::Zero(dim);
    Eigen::MatrixXd Bv = Eigen::MatrixXd::Zero(dim, dim);
    std::vector<char> active(n_constraints, 0);

    auto quad_coeffs = [&](int n) {
        const auto &cn = (*cmodel)[n];
        const double a = 0.5 * (s0.transpose() * cn.H * s0).value();
        const double b = cn.g.dot(s0);
        const double c = cn.c;
        return std::tuple<double, double, double>(a, b, c);
    };

    for (int n = 0; n < n_constraints; ++n) {
        auto [a, b, c] = quad_coeffs(n);
        const bool act0 = (c > 0.0) || (std::abs(c) <= eps_a && 
                         (b > 0.0 || (std::abs(b) <= eps_a && 2.0 * a > 0.0)));
        active[n] = act0 ? 1 : 0;
        if (act0) {
            gv.noalias() += (*cmodel)[n].g;
            Bv.noalias() += (*cmodel)[n].H;
        }
    }

    Eigen::VectorXd g = fmodel->g() + mu * gv;
    Eigen::MatrixXd B = fmodel->H() + mu * Bv;

    // Collect crossing events
    std::vector<std::pair<double, int>> events;
    events.reserve(n_constraints * 2);

    for (int n = 0; n < n_constraints; ++n) {
        auto [a, b, c] = quad_coeffs(n);
        
        if (std::abs(a) <= eps_a) {
            if (std::abs(b) > eps_a) {
                const double t = -c / b;
                if (t > 0.0 && t < T_MAX && std::isfinite(t)) {
                    events.emplace_back(t, n);
                }
            }
        } else {
            double disc = b * b - 4.0 * a * c;
            if (disc >= 0.0) {
                const double rdisc = std::sqrt(std::max(0.0, disc));
                const double t1 = (-b - rdisc) / (2.0 * a);
                const double t2 = (-b + rdisc) / (2.0 * a);
                if (t1 > 0.0 && t1 < T_MAX && std::isfinite(t1)) events.emplace_back(t1, n);
                if (t2 > 0.0 && t2 < T_MAX && std::isfinite(t2)) events.emplace_back(t2, n);
            }
        }
    }
    
    std::sort(events.begin(), events.end());

    // Scan segments
    std::vector<double> candidates;
    candidates.reserve(events.size() + 1);

    double t_l = 0.0;
    size_t k = 0;
    
    while (k < events.size()) {
        const double t_u = events[k].first;
        const double group_tol = std::max(eps_t, eps_t * std::abs(t_u));
        const double t_group_hi = t_u + group_tol;

        // Analyze segment [t_l, t_u]
        const double den = s0.dot(B.selfadjointView<Eigen::Upper>() * s0);
        const double num = g.dot(s0);

        if (num > 0.0) {
            candidates.push_back(t_u);
        } else if (den < 0.0 || (std::abs(den) <= eps_a && num < 0.0)) {
            candidates.push_back(t_u);
        } else {
            const double tau = (std::abs(den) <= eps_a) ? t_u : -num / den;
            if (tau > t_l && tau < t_u && std::isfinite(tau)) {
                candidates.push_back(tau);
            } else {
                candidates.push_back(t_u);
            }
        }

        // Apply crossings at ~t_u
        while (k < events.size() && events[k].first <= t_group_hi) {
            const int idx = events[k].second;
            const auto &cn = (*cmodel)[idx];
            if (active[idx]) {
                B.noalias() -= mu * cn.H;
                g.noalias() -= mu * cn.g;
                active[idx] = 0;
            } else {
                B.noalias() += mu * cn.H;
                g.noalias() += mu * cn.g;
                active[idx] = 1;
            }
            ++k;
        }
        t_l = t_u;
    }

    // Last segment
    const double den = s0.dot(B.selfadjointView<Eigen::Upper>() * s0);
    const double num = g.dot(s0);
    if (!(num > 0.0)) {
        const double tau = (std::abs(den) <= eps_a) ? T_MAX : -num / den;
        if (tau > t_l && tau < T_MAX && std::isfinite(tau)) {
            candidates.push_back(tau);
        } else {
            candidates.push_back(T_MAX);
        }
    }

    // Choose best candidate
    double best_t = 0.0;
    double best_pred = -std::numeric_limits<double>::infinity();
    
    for (double cand : candidates) {
        if (!(cand > 0.0) || cand > T_MAX || !std::isfinite(cand)) continue;
        Eigen::VectorXd s_cand = cand * s0;
        const double pred = predictDescent(fmodel, cmodel, s_cand, mu);
        if (pred > best_pred) {
            best_pred = pred;
            best_t = cand;
        }
    }

    // Build path points
    std::vector<double> path_points;
    path_points.reserve(events.size() + 1);
    const double best_tol = std::max(eps_t, eps_t * std::abs(best_t));

    for (const auto &ev : events) {
        if (ev.first + std::max(eps_t, eps_t * std::abs(ev.first)) < best_t) {
            path_points.push_back(ev.first);
        }
    }
    
    if (best_t > 0.0 && std::isfinite(best_t)) {
        if (path_points.empty() || std::abs(path_points.back() - best_t) > best_tol) {
            path_points.push_back(best_t);
        }
    }

    Eigen::VectorXd path_points_eigen(static_cast<int>(path_points.size()));
    for (int i = 0; i < path_points_eigen.size(); ++i) {
        path_points_eigen(i) = path_points[i];
    }

    return {best_t, path_points_eigen};
}

// =================== MAIN ALGORITHMS ===================

std::tuple<double, Eigen::VectorXd, std::vector<bool>>
l1CriticalityMeasureAndDescentDirection(TRModelPtr &fmodel, CModel &cmodel, const Eigen::VectorXd &x,
                                       double mu, double epsilon, const Eigen::VectorXd &lb,
                                       const Eigen::VectorXd &ub, const Eigen::VectorXd &centerIn,
                                       bool giveRadius) {
    double radius = std::numeric_limits<double>::infinity();
    Eigen::VectorXd center = x;
    
    if (centerIn.size() > 0) center = centerIn;
    if (giveRadius) radius = fmodel->getRadius();

    const int dim = static_cast<int>(x.size());
    auto [is_eactive, lb_active, ub_active] = l1_identify_constraints(cmodel, x, lb, ub, epsilon);
    
    Eigen::VectorXd s0 = Eigen::VectorXd::Zero(fmodel->g().size());
    Eigen::VectorXd is_active_vec = active2eigen(is_eactive);
    Eigen::VectorXd pg = l1_pseudo_gradient_general(fmodel, cmodel, mu, s0, is_active_vec);

    const int n_eactive = static_cast<int>(is_active_vec.sum());

    // Assemble LP problem
    Eigen::VectorXd f_assembled(dim + n_eactive);
    f_assembled << pg, mu * Eigen::VectorXd::Ones(n_eactive);

    Eigen::MatrixXd G;
    if (n_eactive > 0) {
        G.resize(n_eactive, dim);
        G.setZero();
        int idx = 0;
        for (int i = 0; i < cmodel->size(); ++i) {
            if (is_eactive[i]) {
                G.row(idx) = (*cmodel)[i].g;
                ++idx;
            }
        }
    }

    Eigen::MatrixXd Aineq_assembled;
    Eigen::VectorXd bineq_assembled;
    if (n_eactive > 0) {
        Aineq_assembled.resize(n_eactive, dim + n_eactive);
        Aineq_assembled << G, -Eigen::MatrixXd::Identity(n_eactive, n_eactive);
        bineq_assembled = Eigen::VectorXd::Zero(n_eactive);
    }

    // Trust region bounds
    Eigen::VectorXd dlb = (lb.array() - x.array()).cwiseMax((center.array() - x.array() - radius));
    Eigen::VectorXd dub = (ub.array() - x.array()).cwiseMin((center.array() - x.array() + radius));

    Eigen::VectorXd plb_assembled(dim + n_eactive);
    plb_assembled << dlb, Eigen::VectorXd::Zero(n_eactive);

    Eigen::VectorXd pub_assembled(dim + n_eactive);
    pub_assembled << dub, Eigen::VectorXd::Constant(n_eactive, std::numeric_limits<double>::infinity());

    // Solve LP
    GurobiSolver solver;
    auto dt = solver.solveLinearProblem(
        f_assembled, Aineq_assembled, bineq_assembled,
        Eigen::MatrixXd::Zero(0, 0), Eigen::VectorXd::Zero(0),
        plb_assembled, pub_assembled, Eigen::VectorXd::Zero(dim + n_eactive)
    );

    auto [measureOut, dOut, thetaOut] = correctMeasureComputation(pg, G, mu, lb, ub, dt);

    // Safeguarded re-solve
    Eigen::VectorXd dt2(dim + n_eactive);
    dt2 << dOut, thetaOut;

    auto dt2Out = solver.solveLinearProblem(
        f_assembled, Aineq_assembled, bineq_assembled,
        Eigen::MatrixXd::Zero(0, 0), Eigen::VectorXd::Zero(0),
        plb_assembled, pub_assembled, dt2
    );

    auto [measure2Out, d2Out, theta2Out] = correctMeasureComputation(pg, G, mu, lb, ub, dt2Out);
    return {measure2Out, d2Out, is_eactive};
}

Eigen::VectorXd descentDirectionOnePass(TRModelPtr &fmodel, CModel &cmodel, double mu, 
                                       const Eigen::VectorXd &x0, const Eigen::VectorXd &d,
                                       double radius, const Eigen::VectorXd &lb, const Eigen::VectorXd &ub) {
    Eigen::VectorXd x = projectToBounds(x0, lb, ub);
    auto [t_lb, t_ub, tmax_bounds] = boundsBreakpoints(x, lb, ub, d);

    const double tmax_tr = infinityTRRadiusBreakpoint(d, radius);
    const double tmax = std::min(tmax_bounds.minCoeff(), tmax_tr);

    auto [t, brpoints_crossed] = lineSearchCG(fmodel, cmodel, mu, d, tmax);

    const Eigen::ArrayXd lower_hits = (t_lb.array() == t).cast<double>();
    const Eigen::ArrayXd upper_hits = (t_ub.array() == t).cast<double>();

    x = x + t * d;

    for (int i = 0; i < x.size(); ++i) {
        if (lower_hits(i) > 0.0) x(i) = lb(i);
        else if (upper_hits(i) > 0.0) x(i) = ub(i);
    }
    return x;
}

Eigen::VectorXd conjugateGradientNewMeasure(TRModelPtr &fmodel, CModel &cmodel, const Eigen::VectorXd &x0,
                                           double mu, double epsilon, double radius,
                                           const Eigen::VectorXd &lb, const Eigen::VectorXd &ub,
                                           const std::vector<bool> &is_eactive) {
    const int dim = x0.size();
    const int n_constraints = cmodel->size();
    const int max_iters = dim + n_constraints;
    const double tol_d = 2.2204e-16;
    const double tol_h = 2.2204e-16;

    TRModelPtr fmodel_d = std::make_shared<TRModel>(*fmodel);
    CModel cmodel_d = std::make_shared<CModelClass>(*cmodel);

    Eigen::VectorXd x = x0;
    double pred = 0;

    if ((x.array() > ub.array()).any() || (x.array() < lb.array()).any()) {
        throw std::runtime_error("Point already out of bounds");
    }

    auto [_, d, __] = l1CriticalityMeasureAndDescentDirection(fmodel_d, cmodel_d, x, mu, epsilon, lb, ub);

    for (int iter = 0; iter < max_iters; ++iter) {
        if (d.norm() < tol_d) break;

        auto [t_lb, t_ub, tmax_bounds] = boundsBreakpoints(x, lb, ub, d);
        auto toInfinity = x - x0;
        double tmax_tr = infinityTRRadiusBreakpoint(d, radius, toInfinity);
        double tmax = std::min(tmax_bounds.minCoeff(), tmax_tr);

        auto [t, brpoints_crossed] = lineSearchCG(fmodel_d, cmodel_d, mu, d, tmax);

        const double hit_tol = 1e-12;
        Eigen::ArrayXd lower_bound_hits = ((t_lb.array() - t).abs() <= hit_tol).cast<double>();
        Eigen::ArrayXd upper_bound_hits = ((t_ub.array() - t).abs() <= hit_tol).cast<double>();

        Eigen::VectorXd x_prev = x;
        double pred_prev = pred;

        x = x + (t * d.array()).matrix();

        for (int i = 0; i < lower_bound_hits.size(); ++i) {
            if (lower_bound_hits(i) > 0) x(i) = lb(i);
        }
        for (int i = 0; i < upper_bound_hits.size(); ++i) {
            if (upper_bound_hits(i) > 0) x(i) = ub(i);
        }

        Eigen::VectorXd s = x - x0;
        pred = predictDescent(fmodel, cmodel, s, mu);
        if (pred < pred_prev) {
            x = x_prev;
            break;
        }
        if (radius - s.lpNorm<Eigen::Infinity>() < 0.05 * radius) break;

        fmodel_d = shiftModel(fmodel, s);

        for (int k = 0; k < cmodel->size(); ++k) {
            Constraint tempC = cmodel->at(k).shift(s);
            cmodel_d->setC(k, tempC);
        }

        auto [___, pseudo_steepest_descent, ____] = 
            l1CriticalityMeasureAndDescentDirection(fmodel_d, cmodel_d, x, mu, epsilon, lb, ub);

        if ((upper_bound_hits.array() > 0.0).any() || (lower_bound_hits.array() > 0.0).any()) {
            d = pseudo_steepest_descent;
        } else {
            double bpoint_prev = 0.0;
            Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dim, dim);

            for (int k = 0; k < brpoints_crossed.size(); ++k) {
                const double bpoint = brpoints_crossed[k];
                const double interval_length = bpoint - bpoint_prev;
                if (interval_length > 0.0) {
                    const double mid_t = bpoint_prev + 0.5 * interval_length;
                    Eigen::VectorXd s_mid = (x - x0) + mid_t * d;
                    H += (interval_length / t) * l1Hessian(fmodel, cmodel, mu, s_mid);
                }
                bpoint_prev = bpoint;
            }

            if (bpoint_prev < t) {
                const double interval_length = t - bpoint_prev;
                if (interval_length > 0.0) {
                    const double mid_t = bpoint_prev + 0.5 * interval_length;
                    Eigen::VectorXd s_mid = (x - x0) + mid_t * d;
                    H += (interval_length / t) * l1Hessian(fmodel, cmodel, mu, s_mid);
                }
            }

            double dHd = d.dot(H * d);
            if (std::abs(dHd) < tol_h) break;
            double beta = -((pseudo_steepest_descent.transpose() * H * d) / dHd).value();
            d = pseudo_steepest_descent + beta * d;
        }
    }
    return x;
}

std::pair<Eigen::VectorXd, double> 
tryToMakeActivitiesExact(TRModelPtr &fmodel, CModel &cmodel, double mu,
                        const std::vector<bool> &is_eactive, const Eigen::VectorXd &tr_center,
                        const Eigen::VectorXd &xh, double radius,
                        const Eigen::VectorXd &lb, const Eigen::VectorXd &ub, bool guard_descent) {
    Eigen::VectorXd xv = l1FeasibilityCorrection(cmodel, is_eactive, tr_center, xh, radius, lb, ub);

    Eigen::VectorXd xv_tr = xv - tr_center;
    Eigen::VectorXd xh_tr = xh - tr_center;
    double pred_v = predictDescent(fmodel, cmodel, xv_tr, mu);

    if (guard_descent) {
        double pred_h = predictDescent(fmodel, cmodel, xh_tr, mu);
        if (pred_v >= pred_h) {
            return {xv, pred_v};
        } else {
            return {xh, pred_h};
        }
    } else {
        return {xv, pred_v};
    }
}

Eigen::VectorXd l1StepWithMultipliers(TRModelPtr &fmodel_x, CModel &cmodel_x, Eigen::VectorXd &x,
                                     Eigen::VectorXd &multipliers, Eigen::VectorXd &is_eactive,
                                     double mu, Eigen::VectorXd &tr_center, double radius,
                                     const Eigen::VectorXd &lb, const Eigen::VectorXd &ub) {
    const int dim = static_cast<int>(x.size());

    Eigen::VectorXd s0 = Eigen::VectorXd::Zero(static_cast<int>(fmodel_x->g().size()));
    Eigen::VectorXd g = l1_pseudo_gradient_general(fmodel_x, cmodel_x, mu, s0, is_eactive);
    Eigen::MatrixXd H = l1PseudoHessian(fmodel_x, cmodel_x, mu, is_eactive, multipliers);

    // Step bounds
    Eigen::VectorXd lb_shifted = (lb - x).cwiseMax((tr_center - x) - radius * Eigen::VectorXd::Ones(dim));
    Eigen::VectorXd ub_shifted = (ub - x).cwiseMin((tr_center - x) + radius * Eigen::VectorXd::Ones(dim));

    // Equality Jacobian for active constraints
    const int n_eactive = static_cast<int>(is_eactive.sum());
    Eigen::MatrixXd J(n_eactive, dim);
    int idx = 0;
    for (int i = 0; i < is_eactive.size(); ++i) {
        if (is_eactive(i)) {
            J.row(idx) = (*cmodel_x)[i].g;
            ++idx;
        }
    }
    Eigen::VectorXd z = Eigen::VectorXd::Zero(n_eactive);

    GurobiSolver solver;
    auto [h, _, __] = solver.solveQuadraticProblem(
        H, g, 0.0, J, z,
        Eigen::MatrixXd::Zero(0, 0), Eigen::VectorXd::Zero(0),
        lb_shifted, ub_shifted, Eigen::VectorXd::Zero(dim)
    );

    Eigen::VectorXd xm = x + h;
    return projectToBounds(xm, lb, ub);
}

std::tuple<Eigen::VectorXd, double, double> 
l1TrustRegionStep(TRModelPtr &fmodel, CModel &cmodel, Eigen::VectorXd &x,
                 double epsilon, double lambda, double mu, double radius,
                 const Eigen::VectorXd &lb, const Eigen::VectorXd &ub) {

    auto [measure, d, is_eactive_out] = 
        l1CriticalityMeasureAndDescentDirection(fmodel, cmodel, x, mu, epsilon, lb, ub);
    
    Eigen::VectorXd is_eactive = active2eigen(is_eactive_out);

    Eigen::VectorXd xh = descentDirectionOnePass(fmodel, cmodel, mu, x, d, radius, lb, ub);
    Eigen::VectorXd x_cg = conjugateGradientNewMeasure(fmodel, cmodel, x, mu, epsilon, radius, lb, ub, is_eactive_out);
    
    Eigen::VectorXd xh_x = xh - x;
    Eigen::VectorXd x_cg_x = x_cg - x;
    double pred_h = predictDescent(fmodel, cmodel, xh_x, mu);
    double pred_cg = predictDescent(fmodel, cmodel, x_cg_x, mu);

    if (pred_cg - pred_h < -std::numeric_limits<double>::epsilon()) {
        throw std::runtime_error("Wrong computation in descent prediction comparison");
    }

    Eigen::VectorXd xused = x_cg;
    Eigen::VectorXd s = xused - x;
    auto fmodel_shifted = shiftModel(fmodel, s);
    auto cmodel_shifted = shiftListOfModels(cmodel, s);

    auto [is_eactive_h_out, _, __] = l1_identify_constraints(cmodel_shifted, xused, lb, ub, epsilon);

    auto [x_trial, pred] = tryToMakeActivitiesExact(fmodel, cmodel, mu, is_eactive_h_out, x, xused, radius, lb, ub, true);

    if (measure < lambda) {
        auto [x_other, ___] = tryToMakeActivitiesExact(fmodel, cmodel, mu, is_eactive_out, x, x, radius, lb, ub, false);

        s = x_other - x;
        fmodel_shifted = shiftModel(fmodel, s);
        cmodel_shifted = shiftListOfModels(cmodel, s);

        Eigen::VectorXd multipliers = estimateMultipliers(fmodel_shifted, cmodel_shifted, x, mu, is_eactive_out, lb, ub);

        double tol_multipliers = 10 * eps(mu);
        const bool negative_multiplier = (multipliers.array() < -tol_multipliers).any();
        const bool high_multiplier = (multipliers.array() > mu + tol_multipliers).any();

        if (!negative_multiplier && !high_multiplier) {
            Eigen::VectorXd xm = l1StepWithMultipliers(fmodel_shifted, cmodel_shifted, x_other, multipliers, 
                                                      is_eactive, mu, x, radius, lb, ub);
            Eigen::VectorXd xm_x = xm - x;
            double pred_xm = predictDescent(fmodel, cmodel, xm_x, mu);
            if (pred_xm >= pred) {
                pred = pred_xm;
                x_trial = xm;
            } else {
                lambda *= 0.5;
            }
        }
    }

    return {x_trial, pred, lambda};
}

// =================== L1 FUNCTION EVALUATION ===================

std::tuple<double, Eigen::VectorXd> 
l1_function(Funcao &func, const Eigen::VectorXd &con_lb, const Eigen::VectorXd &con_ub,
           double mu, const Eigen::VectorXd &x) {
    auto f = func.obj;
    auto phi = func.con;

    const int n_constraints = phi.size();
    double f_val = f(x);
    Eigen::VectorXd con_vals(n_constraints);
    
    for (int n = 0; n < n_constraints; ++n) {
        con_vals(n) = phi[n](x);
    }
    
    Eigen::VectorXd viol_lb = (con_lb - con_vals).cwiseMax(0.0);
    Eigen::VectorXd viol_ub = (con_vals - con_ub).cwiseMax(0.0);
    double p = f_val + mu * (viol_lb + viol_ub).sum();
    
    Eigen::VectorXd fvalues(n_constraints + 1);
    fvalues << f_val, con_vals;
    
    return {p, fvalues};
}