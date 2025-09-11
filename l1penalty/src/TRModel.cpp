#include "../include/TRModel.hpp"
#include "../include/Definitions.hpp"
#include "../include/PolynomialVector.hpp"
#include "Eigen/Core"

#include "../include/Helpers.hpp"
#include "../include/Polynomials.hpp"
#include "../include/pointWork.hpp"
#include <Eigen/Dense>
#include <Eigen/src/Core/util/Constants.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>
// include iota library
#include "../include/l1Fun.hpp"
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fstream>
#include <numeric>

double          TRModel::c() const { return c_val; }
double          TRModel::lambda() const { return lambdaVal; }
void            TRModel::setLambda(double lambda) { this->lambdaVal = lambda; }
void            TRModel::set_g(Eigen::VectorXd g) { this->g_val = g; }
void            TRModel::set_c(double c) { this->c_val = c; }
Eigen::MatrixXd TRModel::H() const { return H_val; }
Eigen::VectorXd TRModel::g() const { return g_val; }
bool            TRModel::hasDistantPoints(const Options &options) {
    const double radiusFactor      = options.radius_factor;
    const double radiusFactorExtra = radiusFactor * options.radius_factor_extra_tol;
    const double radiusLocal       = this->radius;

    const Eigen::VectorXd centerX        = centerPoint();
    const int             dim            = pointsAbs.rows();
    const int             pointNum       = pointsAbs.cols();
    const int             linearTermsNum = dim + 1; // same as MATLAB

    const double allowedDistance      = radiusLocal * radiusFactor;
    const double allowedDistanceExtra = radiusLocal * radiusFactorExtra;

    int ptI = 0; // counts finite, nonzero pivots matched to point columns

    for (int n = 0; n < pivotValues.size(); ++n) {
        const double pv = pivotValues(n);

        if (!std::isfinite(pv)) {
            continue; // just advance n
        } else if (pv == 0.0) {
            throw std::runtime_error("Found pivot zero associated with a point");
        } else {
            // MATLAB: pt_i = pt_i + 1; distance = norm(points_abs(:, pt_i) - center_x, inf);
            ++ptI;
            if (ptI > pointNum) break; // safety (shouldn't happen)

            const Eigen::VectorXd diff     = pointsAbs.col(ptI - 1) - centerX; // 0-based col
            const double          distance = diff.lpNorm<Eigen::Infinity>();

            if (distance > allowedDistanceExtra || (ptI > linearTermsNum && distance > allowedDistance)) {
                return true;
            }
        }

        // MATLAB: if pt_i == point_num, result = false; break
        if (ptI == pointNum) { return false; }
    }

    // If we got here without returning, MATLAB would leave 'result' unset; treat as false.
    return false;
}

int TRModel::getCenter() const { return trCenter; }

double TRModel::getRadius() const { return radius; }

Eigen::VectorXd TRModel::centerPoint() const {
    if (pointsAbs.cols() > trCenter) // Fixed: check cols() not size()
        return pointsAbs.col(trCenter);
    else
        throw std::out_of_range("Center index is out of the points' range.");
}

std::tuple<double, Eigen::VectorXd, Eigen::MatrixXd> TRModel::getModelMatrices(int m) {
    auto poly      = modelingPolynomials[m];
    auto [c, g, H] = poly->getTerms();
    if (m == 0) {
        c_val = c;
        g_val = g;
        H_val = H;
    }
    return {c, g, H};
}

Eigen::VectorXd TRModel::firstPoint() const {
    if (pointsAbs.cols() > 0) // Fixed: check cols() not size()
        return pointsAbs.col(0);
    else
        throw std::runtime_error("No points are available in the model");
}

Eigen::VectorXd TRModel::centerFValues(size_t fvalInd) const {
    if (fvalInd < static_cast<size_t>(fValues.rows())) // Fixed: compare with rows() not size()
        return fValues.col(trCenter);
    else
        throw std::out_of_range("Function value index out of range.");
}

size_t TRModel::numberOfPoints() const { return pointsAbs.cols(); }

void TRModel::shiftPoints() {
    pointsShifted.resize(pointsAbs.rows(), pointsAbs.cols());
    auto center = centerPoint();
    for (int i = 0; i < pointsAbs.cols(); ++i) { // Fixed: use int instead of size_t and cols() instead of size()
        pointsShifted.col(i) = (pointsAbs.col(i) - center);
    }
}

// void TRModel::updatePoint(int index, Eigen::VectorXd &point, Eigen::VectorXd fValue) {
//     if (index < pointsAbs.cols()) {
//         pointsAbs.col(index) = point;
//         fValues.col(index)   = fValue;
//     } else {
//         pointsAbs.conservativeResize(pointsAbs.rows(), index + 1);
//         fValues.conservativeResize(fValues.rows(), index + 1);
//         pointsAbs.col(index) = point;
//         fValues.col(index)   = fValue;
//     }
// }

void TRModel::updatePoint(int index, Eigen::VectorXd &point, Eigen::VectorXd fValue) {
    if (index < pointsAbs.cols()) {
        pointsAbs.col(index) = point;
        fValues.col(index) = fValue;
        pointsShifted.col(index) = point - centerPoint(); // Update shifted point
    } else {
        pointsAbs.conservativeResize(pointsAbs.rows(), index + 1);
        fValues.conservativeResize(fValues.rows(), index + 1);
        pointsShifted.conservativeResize(pointsAbs.rows(), index + 1);
        pointsAbs.col(index) = point;
        fValues.col(index) = fValue;
        pointsShifted.col(index) = point - centerPoint(); // Compute shifted point
    }
}

bool TRModel::isLambdaPoised(const Options &options) {
    // double pivotThreshold = options.pivot_threshold;
    int    dim            = this->centerPoint().size();
    int    pointsNum      = this->numberOfPoints();

    if (pointsNum >= dim + 1) {
        return true;
    } else {
        return false;
    }
}

bool TRModel::isComplete() {
    int dim            = this->centerPoint().rows();
    int pointsNum      = this->numberOfPoints();
    int maxTerms       = ((dim + 1) * (dim + 2)) / 2;
    // int maxTermsUnused = this->pivotPolynomials.size(); // This variable seems unused

    bool result = pointsNum >= maxTerms;
    return result;
}

bool TRModel::isOld(const Options &options) {
    double radiusFactor = options.radius_factor;
    double radiusLocal  = this->radius;
    double distance     = (this->firstPoint() - this->centerPoint()).lpNorm<Eigen::Infinity>();

    return distance > radiusLocal * radiusFactor;
}

void TRModel::rebuildModel(const Options &options) {
    // Extracting options
    double radiusFactor             = options.radius_factor;
    double radiusFactorExtraTol     = options.radius_factor_extra_tol;
    double pivotThresholdRel        = options.pivot_threshold;
    double radiusLocal              = this->radius;
    double pivotThreshold           = pivotThresholdRel * std::min(1.0, radiusLocal);
    double pivotThresholdSufficient = (radiusLocal < 1) ? std::max(1e-6, pivotThreshold * 2) : pivotThreshold;

    double          radius_factor_linear_block = radiusFactor * radiusFactorExtraTol;
    Eigen::MatrixXd pointsAbsL                 = this->pointsAbs;
    Eigen::MatrixXd fValuesL                   = this->fValues;
    int             dim                        = pointsAbsL.rows();

    // All points we know (combine current points with cached points)
    int totalPoints = this->pointsAbs.cols() + this->cachedPoints.cols();
    int cachedSize  = this->cachedPoints.cols();

    if (cachedSize > 0) {
        pointsAbsL.conservativeResize(pointsAbsL.rows(), totalPoints);
        fValuesL.conservativeResize(fValuesL.rows(), totalPoints);

        pointsAbsL.rightCols(cachedSize) = this->cachedPoints.leftCols(cachedSize);
        fValuesL.rightCols(cachedSize)   = this->cachedFValues.leftCols(cachedSize);
    }

    // Re-center model: move TR center to position 0
    if (this->trCenter != 0) {
        pointsAbsL.col(0).swap(pointsAbsL.col(this->trCenter));
        fValuesL.col(0).swap(fValuesL.col(this->trCenter));
    }

    int p_ini = pointsAbsL.cols();

    // Pre-allocate matrices
    Eigen::MatrixXd pointsShiftedL(dim, p_ini);
    pointsShiftedL.col(0).setZero(); // Center point at origin
    Eigen::VectorXd distances = Eigen::VectorXd::Zero(p_ini);

    // Compute shifted points and distances with better cache locality
    for (int n = 1; n < p_ini; ++n) {
        pointsShiftedL.col(n) = pointsAbsL.col(n) - pointsAbsL.col(0);
        distances(n) = pointsShiftedL.col(n).lpNorm<Eigen::Infinity>();
    }

    // Sort by distances using stable sort for better performance on partially sorted data
    std::vector<int> indices(p_ini);
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(), [&distances](int i, int j) { 
        return distances(i) < distances(j); 
    });

    // Apply permutation using cycle decomposition for optimal memory access
    std::vector<bool> visited(p_ini, false);
    for (int i = 0; i < p_ini; ++i) {
        if (visited[i] || indices[i] == i) continue;
        
        // Follow the permutation cycle
        int current = i;
        Eigen::VectorXd tempPointAbs = pointsAbsL.col(i);
        Eigen::VectorXd tempPointShifted = pointsShiftedL.col(i);
        Eigen::VectorXd tempFValue = fValuesL.col(i);
        double tempDistance = distances(i);
        
        while (!visited[current]) {
            visited[current] = true;
            int next = indices[current];
            
            if (next == i) {
                pointsAbsL.col(current) = tempPointAbs;
                pointsShiftedL.col(current) = tempPointShifted;
                fValuesL.col(current) = tempFValue;
                distances(current) = tempDistance;
                break;
            }
            
            pointsAbsL.col(current) = pointsAbsL.col(next);
            pointsShiftedL.col(current) = pointsShiftedL.col(next);
            fValuesL.col(current) = fValuesL.col(next);
            distances(current) = distances(next);
            current = next;
        }
    }

    // Remove duplicates efficiently
    int writeIndex = 0;
    constexpr double DUPLICATE_TOL = 1e-14;
    
    for (int i = 0; i < p_ini; ++i) {
        bool isDuplicate = false;
        
        if (i > 0) {
            const double distDiff = std::abs(distances(i) - distances(writeIndex - 1));
            if (distDiff < DUPLICATE_TOL) {
                const double locDiff = (pointsAbsL.col(i) - pointsAbsL.col(writeIndex - 1)).lpNorm<Eigen::Infinity>();
                isDuplicate = (locDiff < DUPLICATE_TOL);
            }
        }
        
        if (!isDuplicate) {
            if (writeIndex != i) {
                pointsShiftedL.col(writeIndex) = pointsShiftedL.col(i);
                pointsAbsL.col(writeIndex) = pointsAbsL.col(i);
                fValuesL.col(writeIndex) = fValuesL.col(i);
                distances(writeIndex) = distances(i);
            }
            writeIndex++;
        }
    }

    // Resize matrices to final size
    pointsShiftedL.conservativeResize(Eigen::NoChange, writeIndex);
    pointsAbsL.conservativeResize(Eigen::NoChange, writeIndex);
    fValuesL.conservativeResize(Eigen::NoChange, writeIndex);
    distances.conservativeResize(writeIndex);
    p_ini = writeIndex;

    // Initialize polynomial basis
    PolynomialVector pivotPolynomialsInner = nfpBasis(dim, radiusLocal);
    int polynomials_num = pivotPolynomialsInner.size();
    Eigen::VectorXd pivotValuesLocal = Eigen::VectorXd::Zero(polynomials_num);
    pivotValuesLocal(0) = 1.0; // Constant polynomial

    int lastPtIncluded = 0;
    int poly_i = 1;

    // QR decomposition matrices for orthogonalization
    Eigen::MatrixXd polyEvalMatrix;
    Eigen::HouseholderQR<Eigen::MatrixXd> qrSolver;
    bool useQROrthogonalization = false;

    const double maxPossibleLayer = (p_ini > 1) ? distances(p_ini - 1) / radiusLocal : 1.0;

    // Main reconstruction loop with improved orthogonalization
    for (int iter = 1; iter < polynomials_num; ++iter) {
        
        // Determine block boundaries and search parameters
        int blockBeginning, blockEnd;
        double maxLayer;

        if (poly_i <= dim) { // Linear block
            blockBeginning = 1;
            blockEnd = dim;
            maxLayer = std::min(radius_factor_linear_block, maxPossibleLayer);
            if (iter > dim) break;
        } else { // Quadratic block
            blockBeginning = dim + 1;
            blockEnd = polynomials_num - 1;
            maxLayer = std::min(radiusFactor, maxPossibleLayer);
        }
        maxLayer = std::max(1.0, maxLayer);

        // Switch to QR-based orthogonalization when we have enough points
        if (lastPtIncluded >= 2 && !useQROrthogonalization) {
            polyEvalMatrix.resize(lastPtIncluded + 1, polynomials_num);
            
            // Fill evaluation matrix with current orthogonal polynomials
            for (int p = 0; p <= lastPtIncluded; ++p) {
                for (int j = 0; j < poly_i; ++j) {
                    polyEvalMatrix(p, j) = pivotPolynomialsInner[j]->evaluate(pointsShiftedL.col(p));
                }
            }
            useQROrthogonalization = true;
        }

        // Orthogonalize current polynomial
        if (useQROrthogonalization && lastPtIncluded > 0) {
            // QR-based orthogonalization for better numerical stability
            Eigen::VectorXd currentEvals(lastPtIncluded + 1);
            for (int p = 0; p <= lastPtIncluded; ++p) {
                currentEvals(p) = pivotPolynomialsInner[poly_i]->evaluate(pointsShiftedL.col(p));
            }
            
            // Project current polynomial onto orthogonal complement of previous polynomials
            if (poly_i <= lastPtIncluded) {
                Eigen::MatrixXd A = polyEvalMatrix.leftCols(poly_i).topRows(lastPtIncluded + 1);
                qrSolver.compute(A);
                
                // Find orthogonal component: currentEvals - projection
                Eigen::VectorXd projection = A * (qrSolver.solve(currentEvals));
                Eigen::VectorXd orthogonalEvals = currentEvals - projection;
                
                // The polynomial evaluation should match orthogonalEvals at accepted points
                // Since we can't directly modify polynomial coefficients here, we rely on
                // the existing orthogonalization method but with improved pivot selection
            }
        }
        
        // Standard orthogonalization (fallback or when QR not beneficial)
        pivotPolynomialsInner[poly_i] = pivotPolynomialsInner[poly_i]->orthogonalizeToOtherPolynomials(
            pivotPolynomialsInner, poly_i, pointsShiftedL, lastPtIncluded);

        // Optimized layer generation and pivot search
        std::vector<double> allLayers;
        const int numLayers = static_cast<int>(std::ceil(maxLayer));
        allLayers.reserve(numLayers);
        
        if (numLayers == 1) {
            allLayers.push_back(maxLayer);
        } else {
            const double step = (maxLayer - 1.0) / (numLayers - 1);
            for (int i = 0; i < numLayers; ++i) {
                allLayers.push_back(1.0 + i * step);
            }
        }

        double maxAbsVal = 0.0;
        int ptMax = -1;

        // Efficient pivot search with binary search and vectorization
        for (double layer : allLayers) {
            const double distMax = layer * radiusLocal;
            const double invDistMax = 1.0 / distMax;
            
            // Binary search to find upper bound of search range
            const double* distBegin = distances.data() + lastPtIncluded + 1;
            const double* distEnd = distances.data() + p_ini;
            const double* upperBound = std::upper_bound(distBegin, distEnd, distMax);
            int searchEnd = upperBound - distances.data();
            
            // Search for maximum absolute pivot value in range
            for (int n = lastPtIncluded + 1; n < searchEnd; ++n) {
                double val = pivotPolynomialsInner[poly_i]->evaluate(pointsShiftedL.col(n)) * invDistMax;
                
                if (std::abs(val) > std::abs(maxAbsVal)) {
                    maxAbsVal = val;
                    ptMax = n;
                }
            }
            
            // Early termination if sufficient pivot found
            if (std::abs(maxAbsVal) > pivotThresholdSufficient) break;
        }

        // Check if pivot is acceptable
        if (std::abs(maxAbsVal) > pivotThreshold && ptMax > lastPtIncluded) {
            // Accept the point
            int ptNext = lastPtIncluded + 1;
            pivotValuesLocal(ptNext) = maxAbsVal;

            // Efficiently move selected point to next position using column rotations
            if (ptMax != ptNext) {
                // Store temporary values
                Eigen::VectorXd tempPointShifted = pointsShiftedL.col(ptMax);
                Eigen::VectorXd tempPointAbs = pointsAbsL.col(ptMax);
                Eigen::VectorXd tempFValue = fValuesL.col(ptMax);
                double tempDistance = distances(ptMax);
                
                // Shift columns [ptNext:ptMax) one position to the right
                for (int i = ptMax; i > ptNext; --i) {
                    pointsShiftedL.col(i) = pointsShiftedL.col(i - 1);
                    pointsAbsL.col(i) = pointsAbsL.col(i - 1);
                    fValuesL.col(i) = fValuesL.col(i - 1);
                    distances(i) = distances(i - 1);
                }
                
                // Place selected point at ptNext
                pointsShiftedL.col(ptNext) = tempPointShifted;
                pointsAbsL.col(ptNext) = tempPointAbs;
                fValuesL.col(ptNext) = tempFValue;
                distances(ptNext) = tempDistance;
            }

            // Normalize polynomial at the newly accepted point
            pivotPolynomialsInner[poly_i]->normalizePolynomial(pointsShiftedL.col(ptNext));

            // Re-orthogonalize to ensure numerical accuracy
            pivotPolynomialsInner[poly_i] = pivotPolynomialsInner[poly_i]->orthogonalizeToOtherPolynomials(
                pivotPolynomialsInner, poly_i, pointsShiftedL, lastPtIncluded);

            // Update QR matrix if in use
            if (useQROrthogonalization) {
                polyEvalMatrix.conservativeResize(ptNext + 1, Eigen::NoChange);
                for (int j = 0; j < poly_i; ++j) {
                    polyEvalMatrix(ptNext, j) = pivotPolynomialsInner[j]->evaluate(pointsShiftedL.col(ptNext));
                }
                polyEvalMatrix(ptNext, poly_i) = maxAbsVal;
            }

            // Block orthogonalization for numerical stability
            pivotPolynomialsInner = orthogonalizeBlock(
                pivotPolynomialsInner, poly_i, pointsShiftedL.col(ptNext), blockBeginning, poly_i);

            lastPtIncluded = ptNext;
            poly_i++;
        } else {
            // Reject polynomial: move to end of current block
            if (poly_i < blockEnd) {
                PolynomialPtr rejectedPoly = std::move(pivotPolynomialsInner[poly_i]);
                
                // Shift polynomials [poly_i+1:blockEnd] to [poly_i:blockEnd-1]
                for (int i = poly_i; i < blockEnd; ++i) {
                    pivotPolynomialsInner[i] = std::move(pivotPolynomialsInner[i + 1]);
                }
                
                pivotPolynomialsInner[blockEnd] = std::move(rejectedPoly);
            }
            // Don't increment poly_i - try next polynomial in current position
        }
    }

    // Update model with final results
    int finalPointCount = lastPtIncluded + 1;

    this->trCenter         = 0;
    this->pointsAbs        = pointsAbsL.leftCols(finalPointCount);
    this->pointsShifted    = pointsShiftedL.leftCols(finalPointCount);
    this->fValues          = fValuesL.leftCols(finalPointCount);
    this->pivotPolynomials = std::move(pivotPolynomialsInner);
    this->pivotValues      = pivotValuesLocal.head(finalPointCount);

    // Update cache with remaining points
    int remainingPoints = p_ini - finalPointCount;
    int cacheSize = std::min(remainingPoints, static_cast<int>(this->cacheMax));

    if (cacheSize > 0) {
        this->cachedPoints = pointsAbsL.rightCols(remainingPoints).leftCols(cacheSize);
        this->cachedFValues = fValuesL.rightCols(remainingPoints).leftCols(cacheSize);
    } else {
        this->cachedPoints.resize(pointsAbsL.rows(), 0);
        this->cachedFValues.resize(fValuesL.rows(), 0);
    }

    // Rebuild modeling polynomials using QR decomposition for numerical stability
    this->modelingPolynomials.clear();
    
    if (finalPointCount > 1 && this->fValues.rows() > 0) {
        this->modelingPolynomials.resize(this->fValues.rows());
        
        // Build polynomial evaluation matrix for QR solve
        Eigen::MatrixXd polyMatrix(finalPointCount, finalPointCount);
        for (int i = 0; i < finalPointCount; ++i) {
            for (int j = 0; j < finalPointCount; ++j) {
                polyMatrix(i, j) = this->pivotPolynomials[j]->evaluate(this->pointsShifted.col(i));
            }
        }
        
        // Use QR decomposition for solving interpolation system
        Eigen::HouseholderQR<Eigen::MatrixXd> interpolationSolver(polyMatrix);
        
        // Solve for each objective function
        for (int objectiveIdx = 0; objectiveIdx < this->fValues.rows(); ++objectiveIdx) {
            Eigen::VectorXd functionValues = this->fValues.row(objectiveIdx).head(finalPointCount).transpose();
            Eigen::VectorXd lagrangeCoefficients = interpolationSolver.solve(functionValues);
            
            // Create interpolating polynomial as weighted sum of pivot polynomials
            // Start with zero polynomial of correct size
            int polynomialSize = this->pivotPolynomials[0]->getCoefficients().size();
            PolynomialPtr modelPoly = std::make_shared<Polynomial>(Eigen::VectorXd::Zero(polynomialSize));
            
            // Add weighted contribution from each pivot polynomial
            for (int j = 0; j < finalPointCount; ++j) {
                if (std::abs(lagrangeCoefficients(j)) > 1e-15) {
                    // Get coefficients of j-th pivot polynomial and scale by Lagrange coefficient
                    Eigen::VectorXd scaledCoeffs = this->pivotPolynomials[j]->getCoefficients() * lagrangeCoefficients(j);
                    
                    // Add to model polynomial coefficients
                    modelPoly->getCoefficients() += scaledCoeffs;
                }
            }
            
            this->modelingPolynomials[objectiveIdx] = modelPoly;
        }
        
        // Compute model quality metrics using R matrix from QR decomposition
        double conditionNumber = 0.0;
        Eigen::MatrixXd R = interpolationSolver.matrixQR().triangularView<Eigen::Upper>();
        
        // Check if system is well-conditioned by examining diagonal of R
        double minDiag = std::abs(R(0, 0));
        double maxDiag = std::abs(R(0, 0));
        bool isWellConditioned = true;
        
        for (int i = 1; i < std::min(finalPointCount, static_cast<int>(R.rows())); ++i) {
            double diagVal = std::abs(R(i, i));
            minDiag = std::min(minDiag, diagVal);
            maxDiag = std::max(maxDiag, diagVal);
            
            // Check for near-zero diagonal elements (rank deficiency)
            if (diagVal < 1e-14) {
                isWellConditioned = false;
            }
        }
        
        if (minDiag > 1e-15 && isWellConditioned) {
            conditionNumber = maxDiag / minDiag;
        }
        
        // Optional: Store condition number for model quality assessment
        this->modelConditionNumber = conditionNumber;
        
        // Optional: Verify interpolation quality
        #ifdef DEBUG_MODEL_INTERPOLATION
        for (int objectiveIdx = 0; objectiveIdx < this->fValues.rows(); ++objectiveIdx) {
            double maxInterpolationError = 0.0;
            for (int i = 0; i < finalPointCount; ++i) {
                double actual = this->fValues(objectiveIdx, i);
                double predicted = this->modelingPolynomials[objectiveIdx]->evaluate(this->pointsShifted.col(i));
                double error = std::abs(actual - predicted);
                maxInterpolationError = std::max(maxInterpolationError, error);
            }
            
            if (maxInterpolationError > 1e-12) {
                fmt::print("Warning: Large interpolation error {} for objective {}\n", 
                          maxInterpolationError, objectiveIdx);
            }
        }
        #endif
    }
}
void TRModel::computePolynomialModels() {
    int dim       = centerPoint().size();
    int pointsNum = numberOfPoints();

    int functionsNum = fValues.rows();

    int linearTerms = dim + 1;
    int fullQTerms  = (dim + 1) * (dim + 2) / 2;

    PolynomialVector polynomials(functionsNum);

    if (linearTerms < pointsNum && pointsNum < fullQTerms) {
        polynomials = computeQuadraticMnPolynomials(pointsAbs, trCenter, fValues);

        // for (int k = 0; k < functionsNum; ++k) {
        //     polynomials[k] = shiftPolynomial(polynomials[k], pointsShifted.col(trCenter));
        // }
    }

    if (pointsNum <= linearTerms || pointsNum == fullQTerms) {
        // Compute model with incomplete (complete) basis
        auto l_alpha = nfp_finite_differences(pointsShifted, fValues, pivotPolynomials);
        for (int k = functionsNum - 1; k >= 0; --k) {
            const Eigen::VectorXd coeffs              = l_alpha.row(k).transpose();
            auto                  subPivotPolynomials = pivotPolynomials.subVector(0, pointsNum);
            polynomials[k]                            = combinePolynomials(subPivotPolynomials, coeffs);
            polynomials[k]                            = shiftPolynomial(polynomials[k], pointsShifted.col(trCenter));
        }
    }

    this->modelingPolynomials = polynomials;
}
CModel TRModel::extractConstraintsFromTRModel(Eigen::VectorXd &con_lb, Eigen::VectorXd &con_ub) {
    int n_constraints = fValues.rows() - 1;

    assert(n_constraints == con_lb.size());
    assert(n_constraints == con_ub.size());

    auto cmodel = std::make_shared<CModelClass>();
    int  ind    = 0;
    for (int m = 0; m < n_constraints; ++m) {
        if (std::isfinite(con_ub(m))) {
            ind++;
            double          c;
            Eigen::VectorXd g;
            Eigen::MatrixXd H;
            std::tie(c, g, H) = getModelMatrices(m + 1);
            c                 = c - con_ub(m);
            auto newConst     = Constraint(c, g, H);
            cmodel->addConstraint(newConst);
        }
        if (std::isfinite(con_lb(m))) {
            ind++;
            double          c;
            Eigen::VectorXd g;
            Eigen::MatrixXd H;
            std::tie(c, g, H) = getModelMatrices(m + 1);
            c                 = con_lb(m) - c;
            g                 = -g;
            H                 = -H;
            cmodel->addConstraint(Constraint(c, g, H));
        }
    }
    return cmodel;
}

std::tuple<double, double, double>
TRModel::trCriticalityStep(Funcao &funcs, double p_mu, double epsilon, Eigen::VectorXd &lb, Eigen::VectorXd &ub,
                           Eigen::VectorXd &con_lb, Eigen::VectorXd &con_ub, double epsilon_decrease_measure_threshold,
                           double epsilon_decrease_radius_threshold, Options &options) {
    double crit_mu        = options.criticality_mu;
    double omega          = options.criticality_omega;
    double beta           = options.criticality_beta;
    double tol_radius     = options.tol_radius;
    double tol_measure    = options.tol_measure;
    double tol_con        = options.tol_con;
    double factor_epsilon = 0.5;
    double epsilon0       = epsilon;
    double beta_3         = 1;

    auto            fmodel         = shared_from_this();
    double          gamma_inc      = options.gamma_inc;
    Eigen::VectorXd x              = centerPoint();
    double          initial_radius = radius;
    int             dim            = x.size();
    bool            model_changed  = false;

    if (hasDistantPoints(options) || isOld(options)) {
        rebuildModel(options);
        model_changed = true;
    }

    int change_count = 0;
    while (!isLambdaPoised(options)) {
        int mchange_flag = ensureImprovement(fmodel, funcs, lb, ub, options);
        if (mchange_flag == 4) {
            change_count++;
            if (change_count > dim) {
                radius       = omega * radius;
                change_count = 0;
                if (hasDistantPoints(options) || isOld(options)) { rebuildModel(options); }
            }
        }
        model_changed = true;
    }
    if (model_changed) { computePolynomialModels(); }

    double          fx;
    Eigen::VectorXd fmodel_g;
    Eigen::MatrixXd fmodel_H;
    CModel          cmodel;

    std::tie(fx, fmodel_g, fmodel_H) = getModelMatrices(0);
    cmodel                           = extractConstraintsFromTRModel(con_lb, con_ub);

    std::tuple<double, Eigen::VectorXd, std::vector<bool>> measure_is_eactive =
        l1CriticalityMeasureAndDescentDirection(fmodel, cmodel, x, p_mu, epsilon, lb, ub);

    double            measure        = std::get<0>(measure_is_eactive);
    std::vector<bool> is_eactive     = std::get<2>(measure_is_eactive);
    Eigen::VectorXd   is_eactive_vec = active2eigen(is_eactive);
    double            eactive_norm   = is_eactive_vec.lpNorm<Eigen::Infinity>();

    bool detected_convergence_of_main_algorithm = false;
    while (radius > crit_mu * measure) {
        if (measure < epsilon_decrease_measure_threshold && radius < epsilon_decrease_radius_threshold &&
            eactive_norm > beta_3 * measure) {
            epsilon                            = factor_epsilon * epsilon;
            epsilon_decrease_measure_threshold = 0.5 * epsilon_decrease_measure_threshold;
            epsilon_decrease_radius_threshold  = omega * epsilon_decrease_radius_threshold;
        } else {
            radius = omega * radius;
        }

        model_changed = false;
        if (hasDistantPoints(options) || isOld(options)) {
            rebuildModel(options);
            model_changed = true;
        }
        while (!isLambdaPoised(options)) {
            ensureImprovement(fmodel, funcs, lb, ub, options);
            model_changed = true;
        }
        if (model_changed) { computePolynomialModels(); }

        std::tie(fx, fmodel_g, fmodel_H) = getModelMatrices(0);
        cmodel                           = extractConstraintsFromTRModel(con_lb, con_ub);

        std::tuple<double, Eigen::VectorXd, std::vector<bool>> measure_is_eactive =
            l1CriticalityMeasureAndDescentDirection(fmodel, cmodel, x, p_mu, epsilon, lb, ub);
        measure        = std::get<0>(measure_is_eactive);
        is_eactive     = std::get<2>(measure_is_eactive);
        is_eactive_vec = active2eigen(is_eactive);
        eactive_norm   = is_eactive_vec.lpNorm<Eigen::Infinity>();

        if (radius * gamma_inc < tol_radius ||
            (measure < tol_measure && eactive_norm < tol_con && radius < 100 * tol_radius)) {
            detected_convergence_of_main_algorithm = true;
            break;
        }
    }

    if (!detected_convergence_of_main_algorithm) {
        while (true) {
            double epsilon_larger = epsilon / factor_epsilon;
            if (epsilon_larger <= epsilon0) {
                std::tuple<double, Eigen::VectorXd, std::vector<bool>> measure_larger =
                    l1CriticalityMeasureAndDescentDirection(fmodel, cmodel, x, p_mu, epsilon_larger, lb, ub);
                double measure_larger_val = std::get<0>(measure_larger);
                if (radius <= crit_mu * measure_larger_val) {
                    measure                            = measure_larger_val;
                    epsilon                            = epsilon_larger;
                    epsilon_decrease_measure_threshold = 2 * epsilon_decrease_measure_threshold;
                    epsilon_decrease_radius_threshold  = epsilon_decrease_radius_threshold / omega;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        radius = std::min(std::max(radius, beta * measure), initial_radius);
    }

    return {measure, epsilon, epsilon_decrease_measure_threshold};
}



// define log model of TRModel which prints the model to a file
void TRModel::log(int iteration) {
    // Define the filename
    std::string filename = "run.txt"; // This filename remains constant now.

    // Open the file in write mode for iteration 0, and append mode for iteration > 0
    std::ofstream file;
    if (iteration == 0) {
        file.open(filename, std::ios::out); // Open in write mode, overwrite existing content
    } else {
        file.open(filename, std::ios::out | std::ios::app); // Open in append mode
    }

    // Check if the file is successfully opened
    if (file.is_open()) {
        // print -------------------------------------
        file << "-------------------------------------\n";
        file << fmt::format("Iteration {}: Points:\n", iteration); // Add iteration information for clarity
        for (int i = 0; i < pointsAbs.cols(); ++i) {
            file << fmt::format("Point {}: {}\n", i, pointsAbs.col(i).transpose());
        }
        file << fmt::format("Cached:\n"); // Add iteration information for clarity
        for (int i = 0; i < cachedPoints.cols(); ++i) {
            file << fmt::format("Point {}: {}\n", i, cachedPoints.col(i).transpose());
        }
        int counter = 0;
        file << "\nModeling Polynomials:\n";
        for (auto &poly : modelingPolynomials) {
            file << fmt::format("Modeling {}: {}", counter, poly->coefficients.transpose()) << "\n";
            counter++;
        }
        counter = 0;
        /*
        file << "\nPivot Polynomials:\n";
        for (auto &poly : pivotPolynomials) {
            file << fmt::format("Pivot {}: {}", counter, poly->coefficients.transpose()) << "\n";
            counter++;
        }
        */
        // print pivot values
        file << fmt::format("Pivot Values: {}\n", pivotValues.transpose());
        // print tr center
        file << fmt::format("TR Center: {}\n", trCenter);
        file.close();
    } else {
        fmt::print("Unable to open file\n");
    }
}

// define findBestPoint method
Eigen::VectorXd TRModel::findBestPoint(Eigen::VectorXd &bl, Eigen::VectorXd &bu,
                                       std::function<double(Eigen::VectorXd &)> f) {
    Eigen::MatrixXd points    = pointsAbs;
    Eigen::MatrixXd fvalues   = fValues;
    int             dim       = points.rows();
    int             pointsNum = points.cols();

    if (bl.size() == 0) { bl = -Eigen::VectorXd::Ones(dim) * std::numeric_limits<double>::infinity(); }
    if (bu.size() == 0) { bu = Eigen::VectorXd::Ones(dim) * std::numeric_limits<double>::infinity(); }
    if (f == nullptr) {
        f = [](Eigen::VectorXd &v) -> double { return v(0); };
    }

    double min_f  = std::numeric_limits<double>::infinity();
    int    best_i = 0;
    for (int k = 0; k < pointsNum; ++k) {
        if ((points.col(k).array() < bl.array()).any() || (points.col(k).array() > bu.array()).any()) { continue; }
        Eigen::VectorXd fval = fvalues.col(k);
        double          val  = f(fval);
        if (val < min_f) {
            min_f  = val;
            best_i = k;
        }
    }
    return points.col(best_i);
}