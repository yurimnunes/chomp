#pragma once

#include "Definitions.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <memory> // Add missing include statement
#include <stdexcept>
#include <vector>

#include "Definitions.hpp"
#include "PolynomialVector.hpp"
#include "Polynomials.hpp"

#include "TRModel.hpp"

class PolynomialVector;

PolynomialPtr operator*(PolynomialPtr &poly, double scalar);

PolynomialPtr operator*(double scalar, PolynomialPtr &poly);

PolynomialPtr operator+(PolynomialPtr &poly1, PolynomialPtr &poly2);

PolynomialPtr combinePolynomials(PolynomialVector &polynomials, const Eigen::VectorXd &coefficients);

PolynomialVector orthogonalizeBlock(const PolynomialVector &polynomials, const int np, const Eigen::VectorXd &point,
                                    int orthBeginning, int orthEnd);

PolynomialPtr shiftPolynomial(PolynomialPtr &poly, const Eigen::VectorXd &shift);

PolynomialVector computeQuadraticMnPolynomials(const Eigen::MatrixXd &points, int center_i,
                                               const Eigen::MatrixXd &fvalues);

PolynomialVector computeQuadraticMFNPolynomials(const Eigen::MatrixXd &pointsAbs, int trCenter,
                                                const Eigen::MatrixXd &fValues);

PolynomialVector computeQuadraticAdaptivePolynomials(const Eigen::MatrixXd &pointsAbs, int trCenter,
                                                     const Eigen::MatrixXd &fValues);
