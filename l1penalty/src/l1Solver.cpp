#include "../include/l1Solver.hpp"
#include "../include/Definitions.hpp"
#include "../include/Helpers.hpp"
#include "../include/PolynomialVector.hpp"
#include "../include/Polynomials.hpp"
#include "../include/Solvers.hpp"
#include "../include/TRModel.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <fmt/core.h>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <vector>

std::tuple<Eigen::VectorXd, double> l1_penalty_solve(
    Funcao &func,
    Eigen::MatrixXd &initial_points,
    double mu, double epsilon, double delta, double lambda,
    Eigen::VectorXd &bl, Eigen::VectorXd &bu,
    Eigen::VectorXd &con_bl, Eigen::VectorXd &con_bu,
    Options &options)
{
    // --- Basic sizes ---
    const int dim = static_cast<int>(initial_points.rows());
    int n_initial_points = static_cast<int>(initial_points.cols());
    const int n_functions = 1 + static_cast<int>(func.con.size());

    // --- Ensure at least 2 initial points (second inside bounds or near x0) ---
    if (n_initial_points == 1) {
        initial_points.conservativeResize(dim, 2);
        n_initial_points = 2;

        std::mt19937 gen(std::random_device{}());
        const Eigen::VectorXd x0 = initial_points.col(0);
        Eigen::VectorXd x1(dim);

        for (int i = 0; i < dim; ++i) {
            const double lbi = (bl.size() ? bl(i) : -std::numeric_limits<double>::infinity());
            const double ubi = (bu.size() ? bu(i) :  std::numeric_limits<double>::infinity());
            const bool has_lb = std::isfinite(lbi);
            const bool has_ub = std::isfinite(ubi);

            if (has_lb && has_ub && lbi < ubi) {
                std::uniform_real_distribution<double> dist(lbi, ubi);
                x1(i) = dist(gen);
            } else {
                std::uniform_real_distribution<double> dist(-0.25, 0.25);
                x1(i) = x0(i) + dist(gen);
            }
        }
        initial_points.col(1) = projectToBounds(x1, bl, bu);
    }

    // --- Evaluate f,con at the initial points (project to bounds first) ---
    Eigen::MatrixXd initial_f_values(n_functions, n_initial_points);
    for (int k = 0; k < n_initial_points; ++k) {
        initial_points.col(k) = projectToBounds(initial_points.col(k), bl, bu);
        initial_f_values.col(k) = func.calcAll(initial_points.col(k));
    }

    // --- Build initial TR model ---
    TRModelPtr trmodel = std::make_shared<TRModel>(initial_points, initial_f_values, options);
    trmodel->rebuildModel(options);
    trmodel->computePolynomialModels();

    // --- State at current center x ---
    Eigen::VectorXd x = initial_points.col(0);

    // Initial penalty/objective at x (true, not model)
    Eigen::VectorXd x_copy0 = x;
    auto [p0, fvals0] = l1_function(func, con_bl, con_bu, mu, x_copy0);
    double px = p0;                   // penalty value
    // double fx_true = fvals0(0);    // true objective if you want it here as well

    // --- Housekeeping ---
    double fx_model = std::numeric_limits<double>::quiet_NaN(); // model "fx" (c from polynomial)
    Eigen::VectorXd fmodel_g;
    Eigen::MatrixXd fmodel_H;
    CModel          cmodel;

    double measure = std::numeric_limits<double>::quiet_NaN();
    Eigen::VectorXd d;
    std::vector<bool> is_eactive;

    int    iter = 0;
    bool   finish = false;
    bool   tr_criticality_step_executed = false;
    int    exchange_counts = 0;
    int    mchange_flag = 0;

    // Options shorthands
    const double eps_c       = options.eps_c;
    const double tol_measure = options.tol_measure;
    const double eta_1       = options.eta_1;
    const double eta_2       = options.eta_2;
    const double gamma_1     = options.gamma_dec;
    const double gamma_2     = options.gamma_inc;
    const double radius_max  = options.radius_max;
    const double tol_radius  = options.tol_radius;
    const int    max_iter    = options.max_iter;
    const double divergence_threshold = options.divergence_threshold;

    double ared = std::numeric_limits<double>::quiet_NaN();
    double pred = std::numeric_limits<double>::quiet_NaN();

    double epsilon_decrease_measure_threshold = 1e3 * tol_measure;
    double epsilon_decrease_radius_threshold  = options.initial_radius;

    Funcao fphi = func; // copy as in your original

    // Gurobi env once
    GurobiSolver::initializeEnvironment();

    // Log header
    fmt::print("\n");
    fmt::print("|{:^12}|{:^12}|{:^12}|{:^12}|{:^12}|{:^12}|\n",
               "iter", "fx(model)", "measure", "pred", "rho", "radius");
    fmt::print("|{:-^12}|{:-^12}|{:-^12}|{:-^12}|{:-^12}|{:-^12}|\n",
               "", "", "", "", "", "");

    while (!finish && iter < max_iter) {
        if (std::abs(px) > divergence_threshold) break;

        // Refresh models at current TR state
        trmodel->computePolynomialModels();
        std::tie(fx_model, fmodel_g, fmodel_H) = trmodel->getModelMatrices(0);
        cmodel = trmodel->extractConstraintsFromTRModel(con_bl, con_bu);

        trmodel->log(iter);

        // Criticality measure & direction
        std::tie(measure, d, is_eactive) =
            l1CriticalityMeasureAndDescentDirection(trmodel, cmodel, x, mu, epsilon, bl, bu);

        if (measure <= eps_c) {
            tr_criticality_step_executed = true;
            std::tie(epsilon, epsilon_decrease_measure_threshold, epsilon_decrease_radius_threshold) =
                trmodel->trCriticalityStep(fphi, mu, epsilon, bl, bu, con_bl, con_bu,
                                           epsilon_decrease_measure_threshold, epsilon_decrease_radius_threshold,
                                           options);

            // Rebuild polynomials after criticality step
            trmodel->computePolynomialModels();
            std::tie(fx_model, fmodel_g, fmodel_H) = trmodel->getModelMatrices(0);
            cmodel = trmodel->extractConstraintsFromTRModel(con_bl, con_bu);

            // Recompute measure/direction
            std::tie(measure, d, is_eactive) =
                l1CriticalityMeasureAndDescentDirection(trmodel, cmodel, x, mu, epsilon, bl, bu);
        } else {
            tr_criticality_step_executed = false;
        }

        // Stopping on first-order stationarity (penalty stationarity)
        if (measure < tol_measure) break;

        const bool geometry_ok = trmodel->isLambdaPoised(options);

        // Trust-region step
        Eigen::VectorXd x_step;
        std::tie(x_step, pred, lambda) =
            l1TrustRegionStep(trmodel, cmodel, x, epsilon, lambda, mu, trmodel->radius, bl, bu);

        // Project to bounds
        Eigen::VectorXd x_trial = projectToBounds(x_step, bl, bu);
        const Eigen::VectorXd s = x_trial - x;

        double rho = std::numeric_limits<double>::quiet_NaN();
        bool   evaluate_step = (pred > 0);

        if (evaluate_step) {
            // True penalty at trial point
            Eigen::VectorXd x_trial_copy = x_trial;
            auto [p_trial, trial_fvalues] = l1_function(func, con_bl, con_bu, mu, x_trial_copy);

            if (!trial_fvalues.array().isFinite().all()) {
                rho  = -std::numeric_limits<double>::infinity();
                ared = std::numeric_limits<double>::quiet_NaN();
            } else {
                // ARED using TR model’s stored fValues at center vs. trial fvalues
                ared = evaluatePDescent(trmodel->fValues.col(trmodel->trCenter),
                                        trial_fvalues, con_bl, con_bu, mu);
                rho  = ared / pred;
            }

            if ((rho >= eta_2) || (rho > eta_1 && geometry_ok)) {
                px = p_trial;
                x  = x_trial;

                try {
                    mchange_flag = changeTRCenter(trmodel, x_trial, trial_fvalues, options);
                } catch (const std::exception &e) {
                    std::cerr << e.what() << std::endl;
                    throw;
                }
            } else if (std::isinf(rho)) {
                mchange_flag = ensureImprovement(trmodel, fphi, bl, bu, options);
            } else {
                mchange_flag = try2addPoint(trmodel, x_trial, trial_fvalues, fphi, bl, bu, options);
            }
        } else {
            rho  = -std::numeric_limits<double>::infinity();
            ared = 0.0;
            mchange_flag = ensureImprovement(trmodel, fphi, bl, bu, options);
        }

        // Radius update
        if (rho < eta_2 && (geometry_ok || mchange_flag == 4)) {
            trmodel->radius *= gamma_1;
        } else if (rho >= eta_2 && std::isfinite(rho)) {
            double s_norm_inf = std::min(trmodel->radius, s.lpNorm<Eigen::Infinity>());
            double radius_inc = std::max(1.0, gamma_2 * (s_norm_inf / trmodel->radius));
            trmodel->radius   = std::min(radius_inc * trmodel->radius, radius_max);
        }

        // Exchange throttling
        if (mchange_flag == 2 && rho >= eta_2) {
            if (++exchange_counts > 2 * dim) {
                mchange_flag    = ensureImprovement(trmodel, fphi, bl, bu, options);
                exchange_counts = 0;
            }
        } else {
            exchange_counts = 0;
        }

        // Geometry sanity: detect bad pivots (inf or NaN)
        if (!trmodel->pivotValues.array().isFinite().all()) {
            std::cerr << "Warning: Non-finite pivot detected at iter " << iter
                      << ", radius: " << trmodel->radius << std::endl;
            trmodel->rebuildModel(options);
        }

        // Log (fx_model is the model constant term; true f(x) is printed at the end)
        fmt::print("|{:^12}|{:^12.3f}|{:^12.3f}|{:^12.3f}|{:^12.3f}|{:^12.6f}|\n",
                   iter, fx_model, measure, pred, rho, trmodel->radius);

        if (trmodel->radius < tol_radius) finish = true;
        ++iter;
    }

    // Final reporting
    fmt::print("{}\n", x.transpose());
    fmt::print("|{:-^12}|{:-^12}|{:-^12}|{:-^12}|{:-^12}|{:-^12}|\n", "", "", "", "", "", "");
    fmt::print("\n");

    // Evaluate ORIGINAL objective at final x and return it (not the model's fx)
    Eigen::VectorXd x_final_copy = x;
    auto [p_final, fvals_final] = l1_function(func, con_bl, con_bu, mu, x_final_copy);
    return {x, fvals_final(0)};
}
