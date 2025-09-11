#pragma once
#include "Eigen/Core"
#include "PolynomialVector.hpp"

#include "Helpers.hpp"
#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

bool validateMultipliers(const Eigen::VectorXd& multipliers, double mu, double tolerance = -1);

// define l1CriticalityMeasureAndDescentDirection
std::tuple<double, Eigen::VectorXd, std::vector<bool>>
l1CriticalityMeasureAndDescentDirection(TRModelPtr &fmodel, CModel &cmodel, const Eigen::VectorXd &x,
                                       double mu, double epsilon, const Eigen::VectorXd &lb,
                                       const Eigen::VectorXd &ub, const Eigen::VectorXd &centerIn = Eigen::VectorXd(),
                                       bool giveRadius = false);
// define getGradientAndHessian
std::tuple<Eigen::VectorXd, Eigen::VectorXd> getGradientAndHessian(Eigen::VectorXd &fx, CModel &cmodel,
                                                                   Eigen::VectorXd &x, Eigen::VectorXd &lb,
                                                                   Eigen::VectorXd &ub);

std::tuple<double, Eigen::VectorXd, Eigen::VectorXd> correctMeasureComputation(Eigen::VectorXd &pg, Eigen::MatrixXd &G,
                                                                               double mu, Eigen::VectorXd &lb,
                                                                               Eigen::VectorXd &ub,
                                                                               Eigen::VectorXd &dt);

std::tuple<std::vector<bool>, std::vector<bool>, std::vector<bool>>
l1_identify_constraints(CModel &cmodel, const Eigen::VectorXd &x, const Eigen::VectorXd &lb, 
                       const Eigen::VectorXd &ub, double epsilon);

std::tuple<Eigen::VectorXd, double, double> l1TrustRegionStep(TRModelPtr &fmodel, CModel &cmodel, Eigen::VectorXd &x,
                                                              double epsilon, double lambda, double mu, double radius,
                                                              const Eigen::VectorXd &lb, const Eigen::VectorXd &ub);

Eigen::VectorXd descentDirectionOnePass(TRModelPtr &fmodel, CModel &cmodel, double mu, const Eigen::VectorXd &x0,
                                        const Eigen::VectorXd &d, double radius,const Eigen::VectorXd &lb,const Eigen::VectorXd &ub);

double infinityTRRadiusBreakpoint(const Eigen::VectorXd &d, double radius,
                                  const Eigen::VectorXd &s0 = Eigen::VectorXd());

std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>
boundsBreakpoints(Eigen::VectorXd &x, Eigen::VectorXd &lb, Eigen::VectorXd &ub, Eigen::VectorXd &d);
// Header file with the declaration including default arguments
double predictDescent(TRModelPtr &fmodel, CModel &con_model, const Eigen::VectorXd &s, double mu,
                      const std::vector<bool> &ind_eactive = std::vector<bool>());

Eigen::VectorXd conjugateGradientNewMeasure(TRModelPtr &fmodel, CModel &cmodel, const Eigen::VectorXd &x0,
                                           double mu, double epsilon, double radius,
                                           const Eigen::VectorXd &lb, const Eigen::VectorXd &ub,
                                           const std::vector<bool> &is_eactive) ;

std::tuple<Eigen::VectorXd, Eigen::MatrixXd>
constraintValuesAndGradientsAtPoint(CModel &sharedPtr, std::vector<bool> &matrix, Eigen::VectorXd matrix1);

std::tuple<double, Eigen::VectorXd> 
lineSearchCG(TRModelPtr &fmodel, CModel &cmodel, double mu, const Eigen::VectorXd &s0, double max_t) ;

std::tuple<double, Eigen::VectorXd> 
l1_function(Funcao &func, const Eigen::VectorXd &con_lb, const Eigen::VectorXd &con_ub,
           double mu, const Eigen::VectorXd &x);

Eigen::VectorXd l1StepWithMultipliers(TRModelPtr &fmodel_x, CModel &cmodel_x, Eigen::VectorXd &x,
                                      Eigen::VectorXd &multipliers, Eigen::VectorXd &is_eactive, double mu,
                                      Eigen::VectorXd &tr_center, double radius, const Eigen::VectorXd &lb,
                                      const Eigen::VectorXd &ub);

Eigen::VectorXd l1FeasibilityCorrection(CModel &cmodel, std::vector<bool> &is_eactive, Eigen::VectorXd &tr_center,
                                        Eigen::VectorXd &xh, double radius, Eigen::VectorXd &lb, Eigen::VectorXd &ub);
std::pair<Eigen::VectorXd, double> 
tryToMakeActivitiesExact(TRModelPtr &fmodel, CModel &cmodel, double mu,
                        const std::vector<bool> &is_eactive, const Eigen::VectorXd &tr_center,
                        const Eigen::VectorXd &xh, double radius,
                        const Eigen::VectorXd &lb, const Eigen::VectorXd &ub, bool guard_descent) ;

Eigen::VectorXd estimateMultipliers(TRModelPtr &fmodel, CModel &cmodel, const Eigen::VectorXd &x, double mu,
                                   const std::vector<bool> &is_eactive, const Eigen::VectorXd &lb, const Eigen::VectorXd &ub);

Eigen::MatrixXd l1Hessian(TRModelPtr &fmodel, CModel &cmodel, double mu, const Eigen::VectorXd &s);

Eigen::MatrixXd l1PseudoHessian(TRModelPtr &fmodel, CModel &cmodel, double mu, 
                               const Eigen::VectorXd &is_eactive, const Eigen::VectorXd &multipliers);

