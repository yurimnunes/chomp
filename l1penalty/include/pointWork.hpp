#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <tuple>
#include <vector>

#include "Definitions.hpp"
#include "TRModel.hpp"

class PolynomialVector;

int addPoint(TRModelPtr &model, const Eigen::VectorXd &newPoint, const Eigen::VectorXd &newFValues,
             double relativePivotThreshold);

bool chooseAndReplacePoint(TRModelPtr &model, const Funcao &funcs, Eigen::VectorXd &bl, Eigen::VectorXd &bu,
                           const Options &options);

bool improveModelNFP(TRModelPtr &model, const Funcao &funcs, Eigen::VectorXd &bl, Eigen::VectorXd &bu,
                     const Options &options);

int ensureImprovement(TRModelPtr &model, const Funcao &funcs, Eigen::VectorXd &bl, Eigen::VectorXd &bu,
                      const Options &options);

std::tuple<bool, int> exchangePoint(TRModelPtr &model, const Eigen::VectorXd &newPoint,
                                    const Eigen::VectorXd &newFValues, double relativePivotThreshold,
                                    bool allowExchangeCenter);

Eigen::MatrixXd nfp_finite_differences(const Eigen::MatrixXd &points, const Eigen::MatrixXd &fvalues,
                                       const PolynomialVector &polynomials);

bool hasDistantPoints(TRModelPtr &model, const Options &options);

// bool hasNonFinite(const Eigen::VectorXd &vec);

int try2addPoint(TRModelPtr &model, const Eigen::VectorXd &newPoint, const Eigen::VectorXd &newFValues,
                 const Funcao &funcs, Eigen::VectorXd &bl, Eigen::VectorXd &bu, const Options &options);

double evaluatePDescent(const Eigen::VectorXd &oldValues, const Eigen::VectorXd &newValues,
                        const Eigen::VectorXd &conLb, const Eigen::VectorXd &conUb, double mu);

int changeTRCenter(TRModelPtr &model, Eigen::VectorXd &newPoint, Eigen::VectorXd &newFValues,
                   const Options &options);

std::tuple<Eigen::VectorXd, double, int> minimizeTr(const PolynomialPtr &polynomial, const Eigen::VectorXd &xTrCenter,
                                                    double radius, const Eigen::VectorXd &bl,
                                                    const Eigen::VectorXd &bu);