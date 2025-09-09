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

std::tuple<std::vector<bool>, std::vector<bool>, std::vector<bool>>
l1_identify_constraints(CModel &cmodel, Eigen::VectorXd &x, Eigen::VectorXd &lb, Eigen::VectorXd &ub, double epsilon) {
    const int ncon = cmodel->size();
    const int nvar = static_cast<int>(x.size());

    std::vector<bool> is_eactive(ncon);
    std::vector<bool> lb_active(nvar);
    std::vector<bool> ub_active(nvar);

    const double tol_bounds     = std::sqrt(std::numeric_limits<double>::epsilon());
    const double epsilon_bounds = std::min(tol_bounds, epsilon / 16.0);

    for (int k = 0; k < ncon; ++k) { is_eactive[k] = std::abs((*cmodel)[k].c) < epsilon; }
    for (int i = 0; i < nvar; ++i) {
        lb_active[i] = (x(i) - lb(i)) < epsilon_bounds;
        ub_active[i] = (ub(i) - x(i)) < epsilon_bounds;
    }
    return {is_eactive, lb_active, ub_active};
}

Eigen::VectorXd l1_pseudo_gradient_general(
    TRModelPtr &fmodel,
    CModel &cmodel,
    double mu,
    const Eigen::VectorXd &s,           // displacement from expansion point
    const Eigen::VectorXd &is_eactive   // 1.0 = active, 0.0 = inactive
) {
    const int dim = static_cast<int>(fmodel->g().size());
    Eigen::VectorXd vg = Eigen::VectorXd::Zero(dim);

    // small tolerance to avoid chatter near zero
    const double tol = 1e-12;

    for (int n = 0; n < cmodel->size(); ++n) {
        if (is_eactive(n) <= 0.5) continue;  // only use currently active constraints

        const auto &cn = (*cmodel)[n];

        // c_n(s) = c + g^T s + 0.5 s^T H s  (quadratic model of constraint)
        const double c_value = cn.c + cn.g.dot(s)
                             + 0.5 * (s.transpose() * (cn.H * s)).value();

        if (c_value > tol) {
            // ∂/∂s max(0, c_n(s)) = g_n + H_n s  when c_n(s) > 0
            vg.noalias() += (cn.g + cn.H * s);
        }
    }

    // Base model gradient: ∇f(s) = g_f + H_f s
    const Eigen::VectorXd pg_base = fmodel->g() + fmodel->H() * s;

    // Pseudo-gradient of exact-L1 merit: ∇f + μ * sum_active ∂max(0, c_n)
    return pg_base + mu * vg;
}

std::tuple<double, Eigen::VectorXd, std::vector<bool>>
l1CriticalityMeasureAndDescentDirection(TRModelPtr &fmodel, CModel &cmodel, Eigen::VectorXd &x, double mu,
                                        double epsilon, Eigen::VectorXd &lb, Eigen::VectorXd &ub,
                                        const Eigen::VectorXd &centerIn, bool giveRadius) {

    double          radius = std::numeric_limits<double>::infinity();
    Eigen::VectorXd center = x;
    if (centerIn.size() > 0) { center = centerIn; }
    if (giveRadius) { radius = fmodel->getRadius(); }

    const int dim = static_cast<int>(x.size());

    auto [is_eactive, lb_active, ub_active] = l1_identify_constraints(cmodel, x, lb, ub, epsilon);
    const int gDim                          = static_cast<int>(fmodel->g().size());

    Eigen::VectorXd s0            = Eigen::VectorXd::Zero(gDim);
    Eigen::VectorXd is_active_vec = active2eigen(is_eactive);

    // pseudo-gradient at s = 0
    Eigen::VectorXd pg = l1_pseudo_gradient_general(fmodel, cmodel, mu, s0, is_active_vec);

    const int n_eactive = static_cast<int>(is_active_vec.sum());

    // Linear objective vector [pg; mu*1]
    Eigen::VectorXd f_assembled(dim + n_eactive);
    f_assembled << pg, mu * Eigen::VectorXd::Ones(n_eactive);

    // Active constraint gradients G (n_eactive x dim)
    Eigen::MatrixXd G;
    if (n_eactive > 0) {
        G.resize(n_eactive, dim);
        G.setZero(); // important: make deterministic
        int idx = 0;
        for (int i = 0; i < cmodel->size(); ++i) {
            if (is_eactive[i]) {
                G.row(idx) = (*cmodel)[i].g;
                ++idx;
            }
        }
    } else {
        G.resize(0, 0);
    }

    // Aineq = [ G  -I ], bineq = 0  (enforce G d <= θ, θ >= 0)
    Eigen::MatrixXd Aineq_assembled;
    Eigen::VectorXd bineq_assembled;
    if (n_eactive > 0) {
        Aineq_assembled.resize(n_eactive, dim + n_eactive);
        Aineq_assembled << G, -Eigen::MatrixXd::Identity(n_eactive, n_eactive);
        bineq_assembled = Eigen::VectorXd::Zero(n_eactive);
    } else {
        Aineq_assembled.resize(0, 0);
        bineq_assembled.resize(0);
    }

    // Trust-region box on the step around center: d ∈ [dlb, dub]
    // NOTE: removed ad-hoc ±1 clamps
    Eigen::VectorXd dlb = (lb.array() - x.array()).cwiseMax((center.array() - x.array() - radius));
    Eigen::VectorXd dub = (ub.array() - x.array()).cwiseMin((center.array() - x.array() + radius));

    // Variable bounds on [d; θ]
    Eigen::VectorXd plb_assembled(dim + n_eactive);
    plb_assembled << dlb, Eigen::VectorXd::Zero(n_eactive); // θ ≥ 0

    Eigen::VectorXd pub_assembled(dim + n_eactive);
    pub_assembled << dub, Eigen::VectorXd::Constant(n_eactive, std::numeric_limits<double>::infinity());

    // Solve LP: minimize f_assembled^T [d;θ]  s.t. Aineq [d;θ] ≤ 0, bounds
    GurobiSolver solver;
    auto dt = solver.solveLinearProblem(f_assembled, Aineq_assembled, bineq_assembled, Eigen::MatrixXd::Zero(0, 0),
                                        Eigen::VectorXd::Zero(0), // no equalities
                                        plb_assembled, pub_assembled, Eigen::VectorXd::Zero(dim + n_eactive));

    // Recompute measure with proper (d,θ)
    auto [measureOut, dOut, thetaOut] = correctMeasureComputation(pg, G, mu, lb, ub, dt);

    // One safeguarded re-solve starting from [dOut; thetaOut]
    Eigen::VectorXd dt2(dim + n_eactive);
    dt2 << dOut, thetaOut;

    auto dt2Out = solver.solveLinearProblem(f_assembled, Aineq_assembled, bineq_assembled, Eigen::MatrixXd::Zero(0, 0),
                                            Eigen::VectorXd::Zero(0), plb_assembled, pub_assembled, dt2);

    auto [measure2Out, d2Out, theta2Out] = correctMeasureComputation(pg, G, mu, lb, ub, dt2Out);
    return {measure2Out, d2Out, is_eactive};
}

std::tuple<double, Eigen::VectorXd, Eigen::VectorXd> correctMeasureComputation(Eigen::VectorXd &pg, Eigen::MatrixXd &G,
                                                                               double mu, Eigen::VectorXd &lb,
                                                                               Eigen::VectorXd &ub,
                                                                               Eigen::VectorXd &dt) {
    const int dim            = static_cast<int>(pg.size());
    const int n_aconstraints = (G.rows() > 0) ? static_cast<int>(G.rows()) : 0;

    double          measure = 0.0;
    Eigen::VectorXd d, theta;

    if (dt.size() == 0) {
        d     = Eigen::VectorXd::Zero(0);
        theta = Eigen::VectorXd::Zero(0);
        return {measure, d, theta};
    }

    assert(dt.size() == dim + n_aconstraints);
    const Eigen::VectorXd d_ot     = dt.head(dim);
    const Eigen::VectorXd theta_ot = dt.tail(n_aconstraints);

    // No projection here: d_ot already satisfies the step bounds from the LP
    d = d_ot;

    if (n_aconstraints > 0) {
        theta = (G * d).cwiseMax(0.0);
    } else {
        theta = Eigen::VectorXd::Zero(0);
    }

    measure = -(pg.dot(d) + mu * theta.lpNorm<1>());
    (void)theta_ot; // optional: compare if you want

    return {measure, d, theta};
}

// define l1TrustRegionStep
std::tuple<Eigen::VectorXd, double, double> l1TrustRegionStep(TRModelPtr &fmodel, CModel &cmodel, Eigen::VectorXd &x,
                                                              double epsilon, double lambda, double mu, double radius,
                                                              Eigen::VectorXd &lb, Eigen::VectorXd &ub) {

    auto [measure, d, is_eactive_out] = l1CriticalityMeasureAndDescentDirection(fmodel, cmodel, x, mu, epsilon, lb, ub);
    auto is_eactive                   = active2eigen(is_eactive_out);

    Eigen::VectorXd xh   = descentDirectionOnePass(fmodel, cmodel, mu, x, d, radius, lb, ub);
    Eigen::VectorXd x_cg = conjugateGradientNewMeasure(fmodel, cmodel, x, mu, epsilon, radius, lb, ub, is_eactive_out);
    Eigen::VectorXd xh_x = xh - x;
    Eigen::VectorXd x_cg_x  = x_cg - x;
    double          pred_h  = predictDescent(fmodel, cmodel, xh_x, mu);
    double          pred_cg = predictDescent(fmodel, cmodel, x_cg_x, mu);

    if (pred_cg - pred_h < -std::numeric_limits<double>::epsilon()) {
        throw std::runtime_error("Wrong computation in descent prediction comparison");
    }

    Eigen::VectorXd xused = x_cg; // Using deep copy for safety

    Eigen::VectorXd s              = xused - x;
    auto            fmodel_shifted = shiftModel(fmodel, s);
    auto            cmodel_shifted = shiftListOfModels(cmodel, s);

    auto [is_eactive_h_out, _, __] = l1_identify_constraints(cmodel_shifted, xused, lb, ub, epsilon);

    Eigen::VectorXd is_eactive_h = active2eigen(is_eactive_h_out);
    auto [x_trial, pred] =
        tryToMakeActivitiesExact(fmodel, cmodel, mu, is_eactive_h_out, x, xused, radius, lb, ub, true);

    if (measure < lambda) {
        auto [x_other, _] = tryToMakeActivitiesExact(fmodel, cmodel, mu, is_eactive_out, x, x, radius, lb, ub, false);

        s              = x_other - x;
        fmodel_shifted = shiftModel(fmodel, s);
        cmodel_shifted = shiftListOfModels(cmodel, s);

        Eigen::VectorXd multipliers =
            estimateMultipliers(fmodel_shifted, cmodel_shifted, x, mu, is_eactive_out, lb, ub);

        double     tol_multipliers     = 10 * eps(mu);
        const bool negative_multiplier = (multipliers.array() < -tol_multipliers).any();
        const bool high_multiplier     = (multipliers.array() > mu + tol_multipliers).any();

        //if (negative_multiplier) { std::cout << "Warning: Negative Lagrange multiplier\n"; }
        //if (high_multiplier) { std::cout << "Warning: Overly high Lagrange multiplier\n"; }

        if (!negative_multiplier && !high_multiplier) {
            Eigen::VectorXd xm = l1StepWithMultipliers(fmodel_shifted, cmodel_shifted, x_other, multipliers, is_eactive,
                                                       mu, x, radius, lb, ub);
            Eigen::VectorXd xm_x    = xm - x;
            double          pred_xm = predictDescent(fmodel, cmodel, xm_x, mu);
            if (pred_xm >= pred) {
                pred    = pred_xm;
                x_trial = xm;
            } else {
                lambda *= 0.5;
            }
        }
    }

    return {x_trial, pred, lambda};
}

Eigen::VectorXd descentDirectionOnePass(TRModelPtr &fmodel, CModel &cmodel, double mu, Eigen::VectorXd &x0,
                                        Eigen::VectorXd &d, double radius, Eigen::VectorXd &lb, Eigen::VectorXd &ub) {
    Eigen::VectorXd x = projectToBounds(x0, lb, ub); // Ensure feasibility
    Eigen::VectorXd t_lb, t_ub, tmax_bounds;
    std::tie(t_lb, t_ub, tmax_bounds) = boundsBreakpoints(x, lb, ub, d);

    const double tmax_tr = infinityTRRadiusBreakpoint(d, radius);
    const double tmax    = std::min(tmax_bounds.minCoeff(), tmax_tr);

    double          t = 0.0;
    Eigen::VectorXd brpoints_crossed;
    std::tie(t, brpoints_crossed) = lineSearchCG(fmodel, cmodel, mu, d, tmax);

    // Which bounds hit?
    const Eigen::ArrayXd lower_hits = (t_lb.array() == t).cast<double>();
    const Eigen::ArrayXd upper_hits = (t_ub.array() == t).cast<double>();

    x = x + t * d;

    for (int i = 0; i < x.size(); ++i) {
        if (lower_hits(i) > 0.0)
            x(i) = lb(i);
        else if (upper_hits(i) > 0.0)
            x(i) = ub(i);
    }
    return x;
}

double predictDescent(TRModelPtr &fmodel, CModel &con_model, Eigen::VectorXd &s, double mu,
                      const std::vector<bool> &ind_eactive) {

    int               n_constraints = con_model->size();
    std::vector<bool> local_ind_eactive(n_constraints, false);

    if (ind_eactive.size() > 0) { local_ind_eactive = ind_eactive; }

    double f_change = 0.5 * (s.transpose() * fmodel->H() * s).value() + (fmodel->g().transpose() * s).value();
    double c_change = 0;
    for (int k = 0; k < n_constraints; ++k) {
        if (local_ind_eactive[k] == false) {
            double this_change =
                0.5 * (s.transpose() * (*con_model)[k].H * s).value() + (*con_model)[k].g.transpose() * s;

            if ((*con_model)[k].c > 0) {
                if (this_change + (*con_model)[k].c < 0) { this_change = -(*con_model)[k].c; }
                c_change += this_change;
            } else if ((*con_model)[k].c + this_change > 0) {
                this_change = (*con_model)[k].c + this_change;
                c_change += this_change;
            }
        }
    }
    return -(f_change + mu * c_change);
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>
boundsBreakpoints(Eigen::VectorXd &x, Eigen::VectorXd &lb, Eigen::VectorXd &ub, Eigen::VectorXd &d) {
    const int       n = static_cast<int>(x.size());
    Eigen::VectorXd t_lb(n), t_ub(n);

    for (int i = 0; i < n; ++i) {
        if (d(i) < 0.0)
            t_lb(i) = (lb(i) - x(i)) / d(i);
        else
            t_lb(i) = std::numeric_limits<double>::infinity();
        if (d(i) > 0.0)
            t_ub(i) = (ub(i) - x(i)) / d(i);
        else
            t_ub(i) = std::numeric_limits<double>::infinity();
    }
    // keep only forward (t>0) breakpoints; else set inf
    for (int i = 0; i < n; ++i) {
        if (!(t_lb(i) > 0.0) || !std::isfinite(t_lb(i))) t_lb(i) = std::numeric_limits<double>::infinity();
        if (!(t_ub(i) > 0.0) || !std::isfinite(t_ub(i))) t_ub(i) = std::numeric_limits<double>::infinity();
    }
    return {t_lb, t_ub, t_lb.cwiseMin(t_ub)};
}

double infinityTRRadiusBreakpoint(const Eigen::VectorXd &d, double radius, const Eigen::VectorXd &s0) {
    assert(radius > 0.0);

    const Eigen::VectorXd s0_eff = (s0.size() == 0) ? Eigen::VectorXd::Zero(d.size()) : s0;

    if (s0_eff.lpNorm<Eigen::Infinity>() > radius + 1e-12) { std::cerr << "Warning: Initial point out of TR radius\n"; }

    const auto inc = d.array() > 0.0;
    const auto dec = d.array() < 0.0;

    // Avoid dividing by zero: we will only consider entries where inc/dec is true
    const Eigen::ArrayXd inc_max = (radius - s0_eff.array()) / d.array();
    const Eigen::ArrayXd dec_max = (-radius - s0_eff.array()) / d.array();

    double t_inc = std::numeric_limits<double>::infinity();
    for (int i = 0; i < inc_max.size(); ++i)
        if (inc(i)) t_inc = std::min(t_inc, inc_max(i));

    double t_dec = std::numeric_limits<double>::infinity();
    for (int i = 0; i < dec_max.size(); ++i)
        if (dec(i)) t_dec = std::min(t_dec, dec_max(i));

    const double t = std::min(t_inc, t_dec);
    assert(std::isfinite(t) || std::isinf(t));
    return std::max(0.0, t);
}

std::pair<Eigen::VectorXd, double> tryToMakeActivitiesExact(TRModelPtr &fmodel, CModel &cmodel, double mu,
                                                            std::vector<bool> &is_eactive, Eigen::VectorXd &tr_center,
                                                            Eigen::VectorXd &xh, double radius, Eigen::VectorXd &lb,
                                                            Eigen::VectorXd &ub, bool guard_descent) {
    Eigen::VectorXd xv = l1FeasibilityCorrection(cmodel, is_eactive, tr_center, xh, radius, lb, ub);

    Eigen::VectorXd xv_tr  = xv - tr_center;
    Eigen::VectorXd xh_tr  = xh - tr_center;
    double          pred_v = predictDescent(fmodel, cmodel, xv_tr, mu);
    // double          pred_h = predictDescent(fmodel, cmodel, xh_tr, mu);
    double          pred, pred_h;
    Eigen::VectorXd x;
    if (guard_descent) {
        pred_h = predictDescent(fmodel, cmodel, xh_tr, mu);
        if (pred_v >= pred_h) {
            pred = pred_v;
            x    = xv;
        } else {
            pred = pred_h;
            x    = xh;
        }
    } else {
        x    = xv;
        pred = pred_v;
    }
    return {x, pred};
}

Eigen::VectorXd l1FeasibilityCorrection(CModel &cmodel, std::vector<bool> &is_eactive, Eigen::VectorXd &tr_center,
                                        Eigen::VectorXd &xh, double radius, Eigen::VectorXd &lb, Eigen::VectorXd &ub) {
    Eigen::VectorXd h = xh - tr_center;
    Eigen::VectorXd xv;

    auto is_eactive_vec = active2eigen(is_eactive);

    if (is_eactive_vec.sum() == 0) {
        xv = xh;
    } else {
        Eigen::VectorXd c;
        Eigen::MatrixXd A;
        std::tie(c, A)       = constraintValuesAndGradientsAtPoint(cmodel, is_eactive, h);
        Eigen::MatrixXd H    = A * A.transpose();
        Eigen::VectorXd g    = A * c;
        Eigen::VectorXd v_lb = (Eigen::VectorXd::Constant(h.size(), -radius) - h).cwiseMax(lb - tr_center - h);
        Eigen::VectorXd v_ub = (Eigen::VectorXd::Constant(h.size(), radius) - h).cwiseMin(ub - tr_center - h);
        Eigen::VectorXd v0   = 0.5 * (v_lb + v_ub);

        auto solver = GurobiSolver();
        auto [v, obj, optimal] =
            solver.solveQuadraticProblem(H, g, 0, Eigen::VectorXd::Zero(0), Eigen::VectorXd::Zero(0),
                                         Eigen::VectorXd::Zero(0), Eigen::VectorXd::Zero(0), v_lb, v_ub, v0);
        xv = xh + v;
    }
    return xv;
}

// define constraintValuesAndGradientsAtPoint
std::tuple<Eigen::VectorXd, Eigen::MatrixXd>
constraintValuesAndGradientsAtPoint(CModel &cmodelUnfiltered, std::vector<bool> &is_eactive, Eigen::VectorXd s) {

    auto cmodel = cmodelUnfiltered->filter(is_eactive);

    auto n_vars       = s.size();
    auto n_considered = cmodel.size();

    Eigen::VectorXd nphi(n_considered);
    Eigen::MatrixXd nA(n_vars, n_considered);
    nA.setZero();

    for (int n = 0; n < cmodel.size(); ++n) {
        nphi(n)   = cmodel[n].c + cmodel[n].g.dot(s) + 0.5 * (s.transpose() * cmodel[n].H * s).value();
        nA.col(n) = cmodel[n].g + cmodel[n].H * s;
    }
    return {nphi, nA};
}

// define conjugateGradientNewMeasure
Eigen::VectorXd conjugateGradientNewMeasure(TRModelPtr &fmodel, CModel &cmodel, Eigen::VectorXd &x0, double mu,
                                            double epsilon, double radius, Eigen::VectorXd &lb, Eigen::VectorXd &ub,
                                            std::vector<bool> &is_eactive) {
    auto   dim           = x0.size();
    auto   n_constraints = cmodel->size();
    auto   max_iters     = dim + n_constraints;
    double tol_d         = 2.2204e-16;
    double tol_h         = 2.2204e-16;
    double tol_con       = 1e-6;

    TRModelPtr fmodel_d = std::make_shared<TRModel>(*fmodel);
    CModel     cmodel_d = std::make_shared<CModelClass>(*cmodel);

    Eigen::VectorXd x    = x0;
    double          pred = 0;

    auto [_, d, __] = l1CriticalityMeasureAndDescentDirection(fmodel_d, cmodel_d, x, mu, epsilon, lb, ub);

    if ((x.array() > ub.array()).any() || (x.array() < lb.array()).any()) {
        throw std::runtime_error("Point already out of bounds");
    }

    for (int iter = 0; iter < max_iters; ++iter) {
        // print separator line
        if (d.norm() < tol_d) { break; }

        Eigen::VectorXd t_lb, t_ub, tmax_bounds;
        std::tie(t_lb, t_ub, tmax_bounds) = boundsBreakpoints(x, lb, ub, d);

        auto   toInfinity = x - x0;
        double tmax_tr    = infinityTRRadiusBreakpoint(d, radius, toInfinity);

        double          tmax = std::min(tmax_bounds.minCoeff(), tmax_tr);
        double          t;
        Eigen::VectorXd brpoints_crossed;
        try {
            std::tie(t, brpoints_crossed) = lineSearchCG(fmodel_d, cmodel_d, mu, d, tmax);
        } catch (std::exception &e) { throw e; }

        // Convert boolean conditions to a numeric type that can be used in conditional indexing
        const double hit_tol = 1e-12;
        Eigen::ArrayXd lower_bound_hits = ( (t_lb.array() - t).abs() <= hit_tol ).cast<double>();
        Eigen::ArrayXd upper_bound_hits = ( (t_ub.array() - t).abs() <= hit_tol ).cast<double>();

        // Store previous values if needed for rollback or comparison
        Eigen::VectorXd x_prev    = x;
        double          pred_prev = pred;

        // Update position x based on t and d
        x = x + (t * d.array()).matrix();

        // Apply lower bounds
        for (int i = 0; i < lower_bound_hits.size(); ++i) {
            if (lower_bound_hits(i) > 0) { // Check as double (1.0 if true, 0.0 if false)
                x(i) = lb(i);
            }
        }

        // Apply upper bounds
        for (int i = 0; i < upper_bound_hits.size(); ++i) {
            if (upper_bound_hits(i) > 0) { // Check as double (1.0 if true, 0.0 if false)
                x(i) = ub(i);
            }
        }

        Eigen::VectorXd s = x - x0;
        pred              = predictDescent(fmodel, cmodel, s, mu);
        if (pred < pred_prev) {
            x = x_prev;
            break;
        }
        if (radius - s.lpNorm<Eigen::Infinity>() < 0.05 * radius) { break; }

        auto fmodel_d = shiftModel(fmodel, s);

        // update the working copy of cmodel_d in-place:
        for (int k = 0; k < cmodel->size(); ++k) {
            Constraint tempC = cmodel->at(k).shift(s);
            cmodel_d->setC(k, tempC);
        }

        auto [__, pseudo_steepest_descent, ___] =
            l1CriticalityMeasureAndDescentDirection(fmodel_d, cmodel_d, x, mu, epsilon, lb, ub);

        if ((upper_bound_hits.array() > 0.0).any() || (lower_bound_hits.array() > 0.0).any()) {
            d = pseudo_steepest_descent;
        } else {
            double bpoint_prev = 0.0;
            Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dim, dim);

            // Segments for all crossed breakpoints: [bpoint_prev, bpoint]
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

            // --- Tail segment: [last_bpoint, t] if line search stopped past the last breakpoint ---
            if (bpoint_prev < t) {
                const double interval_length = t - bpoint_prev;
                if (interval_length > 0.0) {
                    const double mid_t = bpoint_prev + 0.5 * interval_length;
                     Eigen::VectorXd s_mid = (x - x0) + mid_t * d;
                    H += (interval_length / t) * l1Hessian(fmodel, cmodel, mu, s_mid);
                }
            }

            double dHd = d.dot(H * d); // or (d.transpose() * H * d)(0,0)
            if (std::abs(dHd) < tol_h) break;
            double beta = -((pseudo_steepest_descent.transpose() * H * d) / dHd).value();

            d = pseudo_steepest_descent + beta * d;
        }
    }
    return x;
}

std::tuple<double, Eigen::VectorXd> lineSearchCG(TRModelPtr &fmodel, CModel &cmodel, double mu, Eigen::VectorXd &s0,
                                                 double max_t) {
    using std::abs;
    using std::sqrt;

    const int    dim           = static_cast<int>(fmodel->g().size());
    const int    n_constraints = static_cast<int>(cmodel->size());
    const double eps_a         = 1e-14;                // quad/lin detect
    const double eps_t         = 1e-12;                // event grouping tol
    const double T_MAX         = std::max(0.0, max_t); // guard

    // ---- Initial active set at t = 0
    Eigen::VectorXd   gv = Eigen::VectorXd::Zero(dim);
    Eigen::MatrixXd   Bv = Eigen::MatrixXd::Zero(dim, dim);
    std::vector<char> active(n_constraints, 0);

    auto quad_coeffs = [&](int n) {
        const auto  &cn = (*cmodel)[n];
        const double a  = 0.5 * (s0.transpose() * cn.H * s0).value(); // t^2 coeff
        const double b  = cn.g.dot(s0);                                // t coeff
        const double c  = cn.c;                                        // const
        return std::tuple<double, double, double>(a, b, c);
    };

    for (int n = 0; n < n_constraints; ++n) {
        auto [a, b, c]  = quad_coeffs(n);
        const bool act0 = (c > 0.0) || (abs(c) <= eps_a && (b > 0.0 || (abs(b) <= eps_a && 2.0 * a > 0.0)));
        active[n]       = act0 ? 1 : 0;
        if (act0) {
            gv.noalias() += (*cmodel)[n].g;
            Bv.noalias() += (*cmodel)[n].H;
        }
    }

    Eigen::VectorXd g = fmodel->g() + mu * gv;
    Eigen::MatrixXd B = fmodel->H() + mu * Bv;
    // Optional: enforce symmetry once for better den stability
    // B = 0.5 * (B + B.transpose());

    // ---- Collect all positive crossings in (0, T_MAX)
    std::vector<std::pair<double, int>> events;
    events.reserve(n_constraints * 2);

    for (int n = 0; n < n_constraints; ++n) {
        auto [a, b, c] = quad_coeffs(n);
        const double A = a, Bq = b, Cq = c; // A t^2 + Bq t + Cq = 0

        if (abs(A) <= eps_a) {
            if (abs(Bq) > eps_a) {
                const double t = -Cq / Bq;
                if (t > 0.0 && t < T_MAX && std::isfinite(t)) events.emplace_back(t, n);
            }
        } else {
            double disc = Bq * Bq - 4.0 * A * Cq;
            if (disc >= 0.0) {
                disc               = std::max(0.0, disc); // clamp tiny negatives
                const double rdisc = sqrt(disc);
                const double t1    = (-Bq - rdisc) / (2.0 * A);
                const double t2    = (-Bq + rdisc) / (2.0 * A);
                if (t1 > 0.0 && t1 < T_MAX && std::isfinite(t1)) events.emplace_back(t1, n);
                if (t2 > 0.0 && t2 < T_MAX && std::isfinite(t2)) events.emplace_back(t2, n);
            }
        }
    }
    std::sort(events.begin(), events.end(),
              [](const auto &L, const auto &R) { return L.first < R.first; });

    // ---- Scan segments; between grouped events, (g,B) is constant
    std::vector<double> candidates;
    candidates.reserve(events.size() + 1);

    double t_l = 0.0;
    size_t k   = 0;
    while (k < events.size()) {
        const double t_u        = events[k].first;
        const double group_tol  = std::max(eps_t, eps_t * abs(t_u));
        const double t_group_hi = t_u + group_tol;

        // Segment [t_l, t_u]
        // If B is symmetric, use selfadjointView for stability:
        const double den = s0.dot(B.selfadjointView<Eigen::Upper>() * s0); // s0' B s0
        const double num = g.dot(s0);                                      // g' s0

        if (num > 0.0) {
            // ascent at left; best at right boundary
            candidates.push_back(t_u);
        } else if (den < 0.0 || (abs(den) <= eps_a && num < 0.0)) {
            // negative curvature or flat descent; boundary again
            candidates.push_back(t_u);
        } else {
            // possible stationary point in (t_l, t_u)
            const double tau = (abs(den) <= eps_a) ? t_u : -num / den;
            if (tau > t_l && tau < t_u && std::isfinite(tau))
                candidates.push_back(tau);
            else
                candidates.push_back(t_u);
        }

        // Apply all crossings at ~t_u (toggle active sets)
        while (k < events.size() && events[k].first <= t_group_hi) {
            const int   idx = events[k].second;
            const auto &cn  = (*cmodel)[idx];
            if (active[idx]) { // turning inactive
                B.noalias() -= mu * cn.H;
                g.noalias() -= mu * cn.g;
                active[idx] = 0;
            } else { // turning active
                B.noalias() += mu * cn.H;
                g.noalias() += mu * cn.g;
                active[idx] = 1;
            }
            ++k;
        }

        t_l = t_u;
    }

    // ---- Last segment [t_l, T_MAX]
    {
        const double den = s0.dot(B.selfadjointView<Eigen::Upper>() * s0);
        const double num = g.dot(s0);
        if (!(num > 0.0)) {
            const double tau = (abs(den) <= eps_a) ? T_MAX : -num / den;
            if (tau > t_l && tau < T_MAX && std::isfinite(tau))
                candidates.push_back(tau);
            else
                candidates.push_back(T_MAX);
        }
        // else ascent; no interior min — skip
    }

    // ---- Choose best t by predicted descent
    double best_t    = 0.0;
    double best_pred = -std::numeric_limits<double>::infinity();
    for (double cand : candidates) {
        if (!(cand > 0.0) || cand > T_MAX || !std::isfinite(cand)) continue;
        Eigen::VectorXd s_cand = cand * s0;
        const double    pred   = predictDescent(fmodel, cmodel, s_cand, mu);
        if (pred > best_pred) {
            best_pred = pred;
            best_t    = cand;
        }
    }

    // ---- Build path points: events strictly before best_t (+ tol), plus best_t if distinct
    std::vector<double> path_points;
    path_points.reserve(events.size() + 1);
    const double best_tol = std::max(eps_t, eps_t * abs(best_t));

    for (const auto &ev : events) {
        if (ev.first + std::max(eps_t, eps_t * abs(ev.first)) < best_t) {
            path_points.push_back(ev.first);
        }
    }
    if (best_t > 0.0 && std::isfinite(best_t)) {
        if (path_points.empty() || abs(path_points.back() - best_t) > best_tol) {
            path_points.push_back(best_t);
        }
    }

    Eigen::VectorXd path_points_eigen(static_cast<int>(path_points.size()));
    for (int i = 0; i < path_points_eigen.size(); ++i) path_points_eigen(i) = path_points[i];

    return {best_t, path_points_eigen};
}


// Improved multiplier estimation with better numerical handling
Eigen::VectorXd estimateMultipliers(TRModelPtr &fmodel, CModel &cmodel, Eigen::VectorXd &x, double mu,
                                    std::vector<bool> &is_eactive, Eigen::VectorXd &lb, Eigen::VectorXd &ub) {
    Eigen::VectorXd is_act_vec = active2eigen(is_eactive);
    const int dim = static_cast<int>(x.size());

    if (is_act_vec.sum() == 0) {
        return Eigen::VectorXd::Zero(0);
    }

    // Pseudo-gradient at current point (s = 0)
    Eigen::VectorXd s0 = Eigen::VectorXd::Zero(dim);
    Eigen::VectorXd pg = l1_pseudo_gradient_general(fmodel, cmodel, mu, s0, is_act_vec);

    // Active nonlinear constraints
    auto            filtered = (*cmodel)[is_eactive];
    Eigen::MatrixXd A        = filtered.g();            // n_act_nl x dim
    const int       n_act_nl = static_cast<int>(A.rows());

    if (n_act_nl == 0) {
        return Eigen::VectorXd::Zero(0);
    }

    // Adaptive bound activity tolerance
    const double maxDiff      = std::max(1.0, (ub - lb).array().abs().maxCoeff());
    const double tol_bounds   = std::min(1e-5, 1e-3 * maxDiff);
    const double grad_norm    = std::max(pg.norm(), 1.0);          // avoid 0
    const double adaptive_tol = std::max(tol_bounds, 1e-8 * grad_norm);
    const double abs_floor    = 1e-12;

    std::vector<int> lb_idxs, ub_idxs;
    lb_idxs.reserve(dim);
    ub_idxs.reserve(dim);

    for (int i = 0; i < dim; ++i) {
        if (x(i) - lb(i) < std::max(adaptive_tol, abs_floor)) lb_idxs.push_back(i);
        if (ub(i) - x(i) < std::max(adaptive_tol, abs_floor)) ub_idxs.push_back(i);
    }

    const int n_lb   = static_cast<int>(lb_idxs.size());
    const int n_ub   = static_cast<int>(ub_idxs.size());
    const int n_rows = n_act_nl + n_lb + n_ub;

    // Extended constraint matrix A_ext (n_rows x dim)
    Eigen::MatrixXd A_ext = Eigen::MatrixXd::Zero(n_rows, dim);
    if (n_act_nl > 0) A_ext.topRows(n_act_nl) = A;

    int r = n_act_nl;
    for (int i : lb_idxs) { A_ext(r, i) = -1.0; ++r; } // lb: lb - x <= 0 -> grad = -e_i
    for (int i : ub_idxs) { A_ext(r, i) =  1.0; ++r; } // ub: x - ub <= 0 -> grad = +e_i

    // Decide QP vs LS via crude condition estimate
    // SVD on A_ext^T (dim x n_rows)
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_check(
        A_ext.transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV
    );

    double cond_num = std::numeric_limits<double>::infinity();
    if (svd_check.singularValues().size() > 0) {
        const double smax = svd_check.singularValues()(0);
        const double smin = svd_check.singularValues().tail(1)(0);
        if (smin > 0.0 && std::isfinite(smax)) cond_num = smax / smin;
    }
    const bool use_qp = (cond_num < 1e12) && (n_rows <= dim + 5);

    Eigen::VectorXd lambda; // n_rows

    if (use_qp) {
        // Minimize 0.5||A_ext^T λ + pg||^2  -> 0.5 λ^T (A_ext A_ext^T) λ + (A_ext pg)^T λ + const
        Eigen::MatrixXd H = A_ext * A_ext.transpose(); // n_rows x n_rows (PSD)
        // Regularize
        const double diag_mean = (H.diagonal().array().abs().mean() > 0.0) ? H.diagonal().array().abs().mean() : 1.0;
        const double reg = std::max(1e-12, 1e-6 * diag_mean);
        H.diagonal().array() += reg;

        Eigen::VectorXd g = A_ext * pg;

        // Bounds: nonlinear multipliers free; bound multipliers >= 0
        Eigen::VectorXd l_lb = Eigen::VectorXd::Constant(n_rows, -1e30);
        Eigen::VectorXd u_ub = Eigen::VectorXd::Constant(n_rows,  1e30);
        for (int i = n_act_nl; i < n_rows; ++i) l_lb(i) = 0.0;

        GurobiSolver solver;
        Eigen::VectorXd lambda0 = Eigen::VectorXd::Zero(n_rows);

        try {
            auto [lambda_qp, obj, optimal] =
                solver.solveQuadraticProblem(
                    H, g, 0.0,
                    Eigen::VectorXd(), Eigen::VectorXd(), // no equalities
                    Eigen::VectorXd(), Eigen::VectorXd(), // no inequalities
                    l_lb, u_ub, lambda0
                );
            if (lambda_qp.size() == n_rows) lambda = lambda_qp;
        } catch (...) {
            // fall through to LS
        }
    }

    if (lambda.size() == 0) {
        // Least-squares: solve A_ext^T λ ≈ −pg with SVD
        if (A_ext.rows() == 0) {
            lambda = Eigen::VectorXd::Zero(n_rows);
        } else {
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(
                A_ext.transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV
            );
            const double smax = (svd.singularValues().size() > 0) ? svd.singularValues()(0) : 1.0;
            const double threshold = std::max(1e-12, 1e-8 * smax);
            svd.setThreshold(threshold);

            lambda = -svd.solve(pg); // dim eqs, n_rows unknowns (minimum-norm λ)

            // Enforce nonnegativity for bound multipliers
            for (int i = n_act_nl; i < n_rows; ++i) {
                if (!std::isfinite(lambda(i)) || lambda(i) < 0.0) lambda(i) = std::max(0.0, lambda(i));
            }
            // Optional: zero tiny noise
            for (int i = 0; i < n_act_nl; ++i) {
                if (!std::isfinite(lambda(i))) lambda(i) = 0.0;
            }
        }
    }

    // Return only nonlinear multipliers
    if (lambda.size() >= n_act_nl) {
        return lambda.head(n_act_nl);
    } else {
        return Eigen::VectorXd::Zero(n_act_nl);
    }
}


// Helper function to validate multiplier estimates
bool validateMultipliers(const Eigen::VectorXd &multipliers, double mu, double tolerance) {
    if (tolerance < 0) { tolerance = 10 * std::max(1e-12, 1e-10 * mu); }

    // Check for negative multipliers (allowing small numerical errors)
    bool has_negative = (multipliers.array() < -tolerance).any();

    // Check for excessively large multipliers
    bool has_excessive = (multipliers.array() > mu + tolerance).any();

    return !has_negative && !has_excessive;
}
Eigen::VectorXd l1StepWithMultipliers(TRModelPtr &fmodel_x, CModel &cmodel_x, Eigen::VectorXd &x,
                                      Eigen::VectorXd &multipliers, Eigen::VectorXd &is_eactive, double mu,
                                      Eigen::VectorXd &tr_center, double radius, Eigen::VectorXd &lb,
                                      Eigen::VectorXd &ub) {
    const int dim = static_cast<int>(x.size());

    Eigen::VectorXd s0 = Eigen::VectorXd::Zero(static_cast<int>(fmodel_x->g().size()));

    Eigen::VectorXd g = l1_pseudo_gradient_general(fmodel_x, cmodel_x, mu, s0, is_eactive);
    Eigen::MatrixXd H = l1PseudoHessian(fmodel_x, cmodel_x, mu, is_eactive, multipliers);

    // step bounds around x within TR and variable bounds
    Eigen::VectorXd lb_shifted = (lb - x).cwiseMax((tr_center - x) - radius * Eigen::VectorXd::Ones(dim));
    Eigen::VectorXd ub_shifted = (ub - x).cwiseMin((tr_center - x) + radius * Eigen::VectorXd::Ones(dim));

    // Equality Jacobian for active constraints
    const int       n_eactive = static_cast<int>(is_eactive.sum());
    Eigen::MatrixXd J(n_eactive, dim);
    int             idx = 0;
    for (int i = 0; i < is_eactive.size(); ++i) {
        if (is_eactive(i)) {
            J.row(idx) = (*cmodel_x)[i].g;
            ++idx;
        }
    }
    Eigen::VectorXd z = Eigen::VectorXd::Zero(n_eactive);

    GurobiSolver solver;
    // NOTE: J,z go in Aeq,beq (not inequalities)
    auto [h, _, __] =
        solver.solveQuadraticProblem(H, g, 0.0, J, z, // Aeq, beq  (enforce J h = 0)
                                     Eigen::MatrixXd::Zero(0, 0), Eigen::VectorXd::Zero(0), // Aineq, bineq
                                     lb_shifted, ub_shifted, Eigen::VectorXd::Zero(dim));

    Eigen::VectorXd xm = x + h;
    return projectToBounds(xm, lb, ub);
}

Eigen::MatrixXd l1Hessian(TRModelPtr &fmodel, CModel &cmodel, double mu, Eigen::VectorXd &s) {
    Eigen::MatrixXd Hfx = fmodel->H();
    Eigen::MatrixXd Hc  = Eigen::MatrixXd::Zero(Hfx.rows(), Hfx.cols());
    for (int n = 0; n < cmodel->size(); ++n) {
        if ((*cmodel)[n].c + (*cmodel)[n].g.dot(s) + 0.5 * (s.dot((*cmodel)[n].H * s)) > 0) { Hc += (*cmodel)[n].H; }
    }
    return Hfx + mu * Hc;
}

// define l1_function
std::tuple<double, Eigen::VectorXd> l1_function(Funcao &func, Eigen::VectorXd &con_lb, Eigen::VectorXd &con_ub,
                                                double mu, Eigen::VectorXd &x) {

    auto f   = func.obj;
    auto phi = func.con;

    auto            n_constraints = phi.size();
    double          f_val         = f(x);
    Eigen::VectorXd con_vals(n_constraints);
    double          sum_violations = 0;
    for (int n = 0; n < n_constraints; ++n) { con_vals(n) = phi[n](x); }
    Eigen::VectorXd viol_lb = (Eigen::VectorXd::Zero(n_constraints).cwiseMax(con_lb - con_vals));
    Eigen::VectorXd viol_ub = (Eigen::VectorXd::Zero(n_constraints).cwiseMax(con_vals - con_ub));
    double          p       = f_val + mu * (viol_lb + viol_ub).sum();
    Eigen::VectorXd fvalues(n_constraints + 1);
    fvalues << f_val, con_vals;
    return {p, fvalues};
}

// define l1PseudoHessian
Eigen::MatrixXd l1PseudoHessian(TRModelPtr &fmodel, CModel &cmodel, double mu, Eigen::VectorXd &is_eactive,
                                Eigen::VectorXd &multipliers) {
    auto            dim    = fmodel->H().rows();
    Eigen::MatrixXd Hfx    = fmodel->H();
    Eigen::MatrixXd Hc     = Eigen::MatrixXd::Zero(dim, dim);
    Eigen::MatrixXd Hm     = Eigen::MatrixXd::Zero(dim, dim);
    int             mult_i = 0;
    for (int n = 0; n < cmodel->size(); ++n) {
        if (is_eactive(n) == 1) {
            auto doubleH = multipliers(mult_i) * (*cmodel)[n].H;
            mult_i++;
            Hm += doubleH;
        } else if ((*cmodel)[n].c > 0) {
            Hc += (*cmodel)[n].H;
        }
    }
    return Hfx + Hm + (mu * Hc);
}
