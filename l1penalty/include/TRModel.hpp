#pragma once

#include "Definitions.hpp"
#include "PolynomialVector.hpp"

#include "Helpers.hpp"
#include "Polynomials.hpp"
#include "pointWork.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>
// include iota library
#include "l1Fun.hpp"
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <numeric>

class TRModel : public std::enable_shared_from_this<TRModel> {
private:
    double          lambdaVal;
    double          c_val;
    Eigen::VectorXd g_val;
    Eigen::MatrixXd H_val;

public:
    Eigen::MatrixXd pointsAbs;
    Eigen::MatrixXd fValues;
    double          radius;
    int             trCenter = 0;
    size_t          cacheMax;
    Eigen::MatrixXd pointsShifted;
    Eigen::MatrixXd cachedPoints;
    Eigen::MatrixXd cachedFValues;
    Eigen::VectorXd pivotValues;
    Options         options;

    PolynomialVector pivotPolynomials;
    PolynomialVector modelingPolynomials;

    // Constructor
    TRModel(Eigen::MatrixXd &points, Eigen::MatrixXd &fvalues, Options &options)
        : pointsAbs(points), fValues(fvalues), options(options), radius(options.initial_radius) {
        size_t dim  = points.rows();
        cacheMax    = 3 * dim * dim;
        size_t fdim = fValues.rows();
        cachedPoints.conservativeResize(dim, 0);
        cachedFValues.conservativeResize(fdim, 0);
        // initializeCaches();
    }

    double          c() const;
    Eigen::VectorXd g() const;
    Eigen::MatrixXd H() const;
    void            set_c(double c);
    void            set_g(Eigen::VectorXd g);

    void setLambda(double lambda);

    double lambda() const;

    // define hasDistantPoints
    bool hasDistantPoints(const Options &options);
    int  getCenter() const;

    double getRadius() const;

    // Access center point
    Eigen::VectorXd centerPoint() const;

    std::tuple<double, Eigen::VectorXd, Eigen::MatrixXd> getModelMatrices(int m);

    // Access first point
    Eigen::VectorXd firstPoint() const;

    // Get function values at the center
    Eigen::VectorXd centerFValues(size_t fvalInd = 0) const;

    // Number of points
    size_t numberOfPoints() const;

    // Shifting points relative to the center
    void shiftPoints();

    // define updatePoint
    void updatePoint(int index, Eigen::VectorXd &point, Eigen::VectorXd fValue);

    bool isLambdaPoised(const Options &options);

    bool isComplete();

    bool isOld(const Options &options);

    void rebuildModel(const Options &options);


    // define computePolynomialModels
    void computePolynomialModels();

    // define extractConstraintsFromTRModel
    CModel extractConstraintsFromTRModel(Eigen::VectorXd &con_lb, Eigen::VectorXd &con_ub);
    // define trCriticalityStep
    std::tuple<double, double, double> trCriticalityStep(Funcao &funcs, double p_mu, double epsilon,
                                                         Eigen::VectorXd &lb, Eigen::VectorXd &ub,
                                                         Eigen::VectorXd &con_lb, Eigen::VectorXd &con_ub,
                                                         double epsilon_decrease_measure_threshold,
                                                         double epsilon_decrease_radius_threshold, Options &options);

    void log(int iteration);

    Eigen::VectorXd findBestPoint(Eigen::VectorXd &bl, Eigen::VectorXd &bu, std::function<double(Eigen::VectorXd &)> f);
};
