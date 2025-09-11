// === Enhanced TR Model Operations (Compatible) ===
// Focused SOTA improvements while maintaining interface compatibility

#include "../include/PolynomialVector.hpp"
#include "../include/Solvers.hpp"
#include "Eigen/Core"
#include <Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "../include/Definitions.hpp"
#include "../include/Helpers.hpp"
#include "../include/Polynomials.hpp"
#include "../include/TRModel.hpp"
#include "../include/l1Solver.hpp"

namespace {
// Enhanced constants for better numerical stability
constexpr double NUMERICAL_TOLERANCE   = 1e-15;
constexpr double QR_DIAGONAL_TOLERANCE = 1e-12;
constexpr double SVD_TOLERANCE         = 1e-14;
constexpr double GEOMETRY_THRESHOLD    = 1e-8;

inline bool isIndexInRange(int i, int n) { return (i >= 0) && (i < n); }

inline int lastPointIdx(int p) { return std::max(0, p - 1); }

inline bool hasNonFinite(const Eigen::VectorXd &v) { return !v.allFinite(); }


std::pair<int, int> getBlockBounds(int position, int dim, int totalPolynomials) {
    if (position <= dim) {
        return {1, std::min(dim, totalPolynomials - 1)};
    } else {
        return {dim + 1, totalPolynomials - 1};
    }
}

void updateCache(TRModelPtr &model, const Eigen::VectorXd &point, const Eigen::VectorXd &fValues) {
    const int cacheCap = static_cast<int>(model->cacheMax);
    const int curCache = static_cast<int>(model->cachedPoints.cols());
    const int newSize  = std::min(curCache + 1, cacheCap);

    Eigen::MatrixXd newCachePts(model->cachedPoints.rows(), newSize);
    Eigen::MatrixXd newCacheF(model->cachedFValues.rows(), newSize);

    newCachePts.col(0) = point;
    newCacheF.col(0)   = fValues;

    const int copyCnt = std::min(curCache, newSize - 1);
    if (copyCnt > 0) {
        newCachePts.rightCols(copyCnt) = model->cachedPoints.leftCols(copyCnt);
        newCacheF.rightCols(copyCnt)   = model->cachedFValues.leftCols(copyCnt);
    }

    model->cachedPoints  = std::move(newCachePts);
    model->cachedFValues = std::move(newCacheF);
}

void updateModelStructures(TRModelPtr &model, int position, const Eigen::VectorXd &newPointAbs,
                           const Eigen::VectorXd &newPointShifted, const Eigen::VectorXd &newFValues,
                           const PolynomialVector &polynomials, double pivotValue) {
    model->pointsAbs.col(position)     = newPointAbs;
    model->pointsShifted.col(position) = newPointShifted;
    model->fValues.col(position)       = newFValues;
    model->pivotPolynomials            = polynomials;
    model->pivotValues(position)       = pivotValue;
    model->modelingPolynomials.clear();
}

} // namespace

[[nodiscard]] std::tuple<PolynomialVector, double, bool> choosePivotPolynomial(PolynomialVector      &pivotPolynomials,
                                                                               const Eigen::MatrixXd &points,
                                                                               int initialIndex, int finalIndex,
                                                                               double tolerance) {
    using Eigen::MatrixXd;
    using Eigen::VectorXd;

    const int numPolys  = static_cast<int>(pivotPolynomials.size());
    const int numPoints = static_cast<int>(points.cols());

    // Basic guardrails
    if (initialIndex < 0 || finalIndex < 0 || initialIndex >= numPolys || finalIndex >= numPolys ||
        initialIndex > finalIndex || initialIndex >= numPoints) {
        return {pivotPolynomials, 0.0, false};
    }

    const int m     = std::min(initialIndex + 1, numPoints); // #rows (points used so far, incl. incumbent)
    const int nCand = finalIndex - initialIndex + 1;         // #candidate polynomials

    // Build evaluation matrix incrementally: A(:, j) = p_{initialIndex+j}(x_0..x_initialIndex)
    MatrixXd            A(m, nCand);
    std::vector<double> pivotValues(nCand, 0.0);

    // Evaluate all candidates on all rows once (cheap relative to factorizations)
    for (int j = 0; j < nCand; ++j) {
        const int polyIdx = initialIndex + j;
        for (int i = 0; i < m; ++i) { A(i, j) = pivotPolynomials[polyIdx]->evaluate(points.col(i)); }
        // incumbent-point value (last row in A)
        pivotValues[j] = A(m - 1, j);
    }

    auto geometryScoreQR = [&](const MatrixXd &Ablock) -> std::pair<double, int> {
        // Column-scale to unit 2-norm to reduce scale bias between basis functions
        MatrixXd As = Ablock;
        for (int j = 0; j < As.cols(); ++j) {
            const double cn = As.col(j).norm();
            if (cn > 0.0) As.col(j) /= cn;
        }
        Eigen::ColPivHouseholderQR<MatrixXd> qr(As);
        const int                            r = qr.rank();

        // rmin/rmax from |diag(R)|
        const int k = std::min<int>(As.rows(), As.cols());
        if (k == 0) return {1.0, r};

        VectorXd     rdiag = qr.matrixR().topLeftCorner(k, k).diagonal().cwiseAbs();
        const double rmax  = std::max(1e-16, rdiag.maxCoeff());
        const double rmin  = rdiag.minCoeff();

        double score = rmin / rmax; // in (0,1]; higher = better geometry
        return {score, r};
    };

    double bestScore  = -1.0;
    int    bestOffset = -1;
    double bestPivot  = 0.0;

    for (int j = 0; j < nCand; ++j) {
        // Reject tiny incumbent value upfront
        const double pmag = std::abs(pivotValues[j]);
        if (pmag <= tolerance) continue;

        // Use A(:, 0..j) block to assess geometry *if we were to accept this j-th column now*
        MatrixXd Ablock      = A.leftCols(j + 1);
        auto [geom, rankEst] = geometryScoreQR(Ablock);

        // Require full rank growth: rank should be j+1 (bounded by m)
        const int targetRank = std::min(j + 1, m);
        if (rankEst < targetRank) continue; // rank collapsed; reject

        // Combined score: incumbent magnitude × geometry
        const double score = pmag * geom;
        if (score > bestScore) {
            bestScore  = score;
            bestOffset = j;
            bestPivot  = pivotValues[j];
        }
    }

    if (bestOffset < 0) { return {pivotPolynomials, 0.0, false}; }

    const int bestPolyIndex = initialIndex + bestOffset;

    pivotPolynomials[bestPolyIndex]->orthogonalizeToOtherPolynomials(pivotPolynomials, bestPolyIndex, points,
                                                                     /*maxPointIndex=*/initialIndex - 1);
    std::swap(pivotPolynomials[initialIndex], pivotPolynomials[bestPolyIndex]);

    return {pivotPolynomials, bestPivot, true};
}

// Enhanced point addition with better error handling (keeps original name)
int addPoint(TRModelPtr &model, const Eigen::VectorXd &newPoint, const Eigen::VectorXd &newFValues,
             double relativePivotThreshold) {

    if (model->pointsAbs.cols() == 0 || model->isComplete()) {
        throw std::runtime_error("Model either empty or full. Should not be calling addPoint");
    }

    const double pivotThreshold = std::min(1.0, model->radius) * relativePivotThreshold;
    const int    dim            = static_cast<int>(model->pointsAbs.rows());
    const int    currentPoints  = static_cast<int>(model->pointsAbs.cols());

    const Eigen::VectorXd shiftCenter     = model->pointsAbs.col(0);
    const Eigen::VectorXd newPointShifted = newPoint - shiftCenter;

    // Extend point matrices
    Eigen::MatrixXd extendedPointsShifted = model->pointsShifted;
    extendedPointsShifted.conservativeResize(Eigen::NoChange, currentPoints + 1);
    extendedPointsShifted.col(currentPoints) = newPointShifted;

    const int nextPosition      = currentPoints;
    auto [blockBegin, blockEnd] = getBlockBounds(nextPosition, dim, static_cast<int>(model->pivotPolynomials.size()));

    // Enhanced polynomial selection
    PolynomialVector workingPolynomials = model->pivotPolynomials;
    auto [updatedPolynomials, pivotValue, success] =
        choosePivotPolynomial(workingPolynomials, extendedPointsShifted, nextPosition, blockEnd, pivotThreshold);

    if (!success) return 0;

    // Robust normalization and orthogonalization
    updatedPolynomials[nextPosition]->normalizePolynomial(newPointShifted);
    updatedPolynomials[nextPosition]->orthogonalizeToOtherPolynomials(
        updatedPolynomials, nextPosition, extendedPointsShifted, lastPointIdx(nextPosition));

    // Block orthogonalization
    updatedPolynomials =
        orthogonalizeBlock(updatedPolynomials, nextPosition, newPointShifted, blockBegin, nextPosition - 1);

    // Update model with extended structures
    model->pointsShifted = extendedPointsShifted;
    model->fValues.conservativeResize(Eigen::NoChange, nextPosition + 1);
    model->fValues.col(nextPosition) = newFValues;
    model->pointsAbs.conservativeResize(Eigen::NoChange, nextPosition + 1);
    model->pointsAbs.col(nextPosition) = newPoint;
    model->pivotValues.conservativeResize(nextPosition + 1);

    updateModelStructures(model, nextPosition, newPoint, newPointShifted, newFValues, updatedPolynomials, pivotValue);

    return 1;
}

// Enhanced model improvement with adaptive radius (keeps original name)
bool improveModelNFP(TRModelPtr &model, const Funcao &funcs, Eigen::VectorXd &bl, Eigen::VectorXd &bu,
                     const Options &options) {

    if (model->pointsShifted.cols() == 0 || model->isComplete()) {
        throw std::runtime_error("Model either empty or full. Should not be calling improve_model_nfp");
    }

    if (model->isOld(options) && model->isLambdaPoised(options)) {
        throw std::runtime_error("Model too old. Should be calling rebuild model");
    }

    const double pivotThreshold = options.pivot_threshold * std::min(1.0, model->radius);
    const double tolRadius      = options.tol_radius;
    const int    dim            = static_cast<int>(model->pointsShifted.rows());
    const int    currentPoints  = static_cast<int>(model->pointsShifted.cols());

    if (bl.size() == 0) bl = Eigen::VectorXd::Constant(dim, -std::numeric_limits<double>::infinity());
    if (bu.size() == 0) bu = Eigen::VectorXd::Constant(dim, std::numeric_limits<double>::infinity());

    const Eigen::VectorXd shiftCenter     = model->pointsAbs.col(0);
    const Eigen::VectorXd trCenterShifted = model->pointsShifted.col(model->trCenter);
    const Eigen::VectorXd blShifted       = bl - shiftCenter;
    const Eigen::VectorXd buShifted       = bu - shiftCenter;

    const int nextPosition   = currentPoints;
    const int blockBeginning = (currentPoints < dim + 1) ? 1 : (dim + 1);

    double currentRadius = model->radius;

    // Enhanced multi-attempt loop with better radius strategy
    for (int attempts = 0; attempts < 5; ++attempts) { // Increased attempts for robustness
        // Robust orthogonalization
        PolynomialVector workingPolynomials = model->pivotPolynomials;
        workingPolynomials[nextPosition]->orthogonalizeToOtherPolynomials(
            workingPolynomials, nextPosition, model->pointsShifted, lastPointIdx(currentPoints));
        auto [candidates, pivotValues, candidatesAbs, successFlags] =
            workingPolynomials[nextPosition]->maximizePolynomialAbs(trCenterShifted, shiftCenter, currentRadius,
                                                                    blShifted, buShifted);

        // Enhanced candidate evaluation with geometry check
        std::vector<std::tuple<int, double, double>> validCandidates; // index, pivot_value, abs_pivot

        for (int j = 0; j < candidates.cols(); ++j) {
            if (!successFlags[j]) continue;

            const double absPivotValue = std::abs(pivotValues(j));
            if (absPivotValue < pivotThreshold) continue;

            validCandidates.emplace_back(j, pivotValues(j), absPivotValue);
        }

        // Sort by absolute pivot value (descending) for better selection
        std::sort(validCandidates.begin(), validCandidates.end(),
                  [](const auto &a, const auto &b) { return std::get<2>(a) > std::get<2>(b); });

        for (const auto &[j, pivotValue, absPivotValue] : validCandidates) {
            Eigen::VectorXd candidatePoint = candidatesAbs.col(j);
            auto [fValues, evalSuccess]    = evaluateNewFValues(funcs, candidatePoint);
            if (!evalSuccess || hasNonFinite(fValues)) continue;

            Eigen::MatrixXd extendedPointsShifted = model->pointsShifted;
            extendedPointsShifted.conservativeResize(Eigen::NoChange, nextPosition + 1);
            extendedPointsShifted.col(nextPosition) = candidates.col(j);

            // Enhanced normalization and orthogonalization
            workingPolynomials[nextPosition]->normalizePolynomial(candidates.col(j));
            workingPolynomials[nextPosition]->orthogonalizeToOtherPolynomials(
                workingPolynomials, nextPosition, extendedPointsShifted, lastPointIdx(nextPosition));

            if (currentPoints > 0) {
                workingPolynomials = orthogonalizeBlock(workingPolynomials, nextPosition, candidates.col(j),
                                                        blockBeginning, nextPosition - 1);
            }

            // Update model structures
            model->pointsShifted = extendedPointsShifted;
            model->pointsAbs.conservativeResize(Eigen::NoChange, nextPosition + 1);
            model->pointsAbs.col(nextPosition) = candidatePoint;
            model->fValues.conservativeResize(Eigen::NoChange, nextPosition + 1);
            model->fValues.col(nextPosition) = fValues;
            model->pivotValues.conservativeResize(nextPosition + 1);

            updateModelStructures(model, nextPosition, candidatePoint, candidates.col(j), fValues, workingPolynomials,
                                  pivotValue);
            return true;
        }

        // Enhanced radius reduction strategy
        if (currentRadius > tolRadius) {
            currentRadius *= (attempts < 2) ? 0.7 : 0.5; // Slower reduction initially
        } else {
            break;
        }
    }

    return false;
}

// Enhanced trust region subproblem with better regularization (keeps original name)
std::tuple<Eigen::VectorXd, double, int> minimizeTr(const PolynomialPtr &polynomial, const Eigen::VectorXd &xTrCenter,
                                                    double radius, const Eigen::VectorXd &bl,
                                                    const Eigen::VectorXd &bu) {

    const int dim = static_cast<int>(xTrCenter.size());

    const Eigen::VectorXd blLocal =
        bl.size() == 0 ? Eigen::VectorXd::Constant(dim, -std::numeric_limits<double>::infinity()) : bl;
    const Eigen::VectorXd buLocal =
        bu.size() == 0 ? Eigen::VectorXd::Constant(dim, std::numeric_limits<double>::infinity()) : bu;

    const Eigen::VectorXd blTr  = xTrCenter.array() - radius;
    const Eigen::VectorXd buTr  = xTrCenter.array() + radius;
    const Eigen::VectorXd blMod = blLocal.cwiseMax(blTr);
    const Eigen::VectorXd buMod = buLocal.cwiseMin(buTr);

    Eigen::VectorXd x0 = xTrCenter.cwiseMax(blLocal).cwiseMin(buLocal);

    auto [c, g, H] = polynomial->getBalancedTerms();

    // Enhanced stationary point handling with eigenvalue analysis
    const double gradNorm = (g + H * x0).norm();
    if (gradNorm < 1e-6) {
        x0 = 0.5 * (blMod + buMod);

        if ((g + H * x0).norm() < 1e-6) {
            // Enhanced eigenvector-based perturbation
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(H);
            const auto                                    &eigenvalues  = eigenSolver.eigenvalues();
            const auto                                    &eigenvectors = eigenSolver.eigenvectors();

            int          minEigenIndex;
            const double minEigenval = eigenvalues.minCoeff(&minEigenIndex);

            if (minEigenval < -1e-8) { // Negative eigenvalue exists
                const Eigen::VectorXd minEigenVector = eigenvectors.col(minEigenIndex).real();

                if (minEigenVector.norm() > 1e-10) {
                    // Compute maximum feasible step along negative curvature direction
                    double maxStep = radius;

                    for (int i = 0; i < dim; ++i) {
                        if (std::abs(minEigenVector(i)) > 1e-10) {
                            const double stepLower       = (blMod(i) - x0(i)) / minEigenVector(i);
                            const double stepUpper       = (buMod(i) - x0(i)) / minEigenVector(i);
                            const double maxFeasibleStep = std::min(std::abs(stepLower), std::abs(stepUpper));
                            maxStep                      = std::min(maxStep, maxFeasibleStep);
                        }
                    }

                    if (maxStep > 1e-10) {
                        // Step along negative curvature direction
                        const double stepSize = std::min(maxStep, 0.5 * radius);
                        x0 += stepSize * minEigenVector.normalized();
                        x0 = x0.cwiseMax(blMod).cwiseMin(buMod);
                    }
                }
            }
        }
    }

    // Enhanced Hessian regularization for indefinite matrices
    Eigen::MatrixXd                                      regularizedH = H;
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(H);
    const double                                         minEigenval = eigenSolver.eigenvalues().minCoeff();

    if (minEigenval < 1e-8) {
        const double regularization = std::max(1e-8, -1.5 * minEigenval + 1e-8);
        regularizedH.diagonal().array() += regularization;
    }

    // Try enhanced solver first
    try {
        GurobiSolver solver;
        auto [solution, objectiveValue, success] =
            solver.solveQuadraticProblem(regularizedH, g, c, Eigen::VectorXd(), Eigen::VectorXd(), Eigen::VectorXd(),
                                         Eigen::VectorXd(), blMod, buMod, x0);

        if (success) {
            solution = projectToBounds(solution, blLocal, buLocal);
            return {solution, objectiveValue, 1};
        }
    } catch (...) {
        // Fall through to backup method
    }

    // Backup: simple projected gradient descent
    Eigen::VectorXd x     = x0;
    const double    alpha = 0.1;

    for (int iter = 0; iter < 20; ++iter) {
        const Eigen::VectorXd grad = g + regularizedH * (x - xTrCenter);
        if (grad.norm() < 1e-8) break;

        x -= alpha * grad;
        x = x.cwiseMax(blMod).cwiseMin(buMod);
    }

    x                     = projectToBounds(x, blLocal, buLocal);
    const double objValue = c + g.dot(x - xTrCenter) + 0.5 * (x - xTrCenter).dot(H * (x - xTrCenter));

    return {x, objValue, 1};
}

// Keep all other functions unchanged for compatibility
bool chooseAndReplacePoint(TRModelPtr &model, const Funcao &funcs, Eigen::VectorXd &bl, Eigen::VectorXd &bu,
                           const Options &options) {

    const double pivotThreshold = options.exchange_threshold;
    const double radius         = model->radius;
    const int    dim            = static_cast<int>(model->pointsShifted.rows());
    const int    pointsNum      = static_cast<int>(model->pointsShifted.cols());

    if (bl.size() == 0) bl = Eigen::VectorXd::Constant(dim, -std::numeric_limits<double>::infinity());
    if (bu.size() == 0) bu = Eigen::VectorXd::Constant(dim, std::numeric_limits<double>::infinity());

    const Eigen::VectorXd shiftCenter     = model->pointsAbs.col(0);
    const int             trCenter        = model->trCenter;
    const Eigen::VectorXd trCenterShifted = model->pointsShifted.col(trCenter);
    const Eigen::VectorXd blShifted       = bl - shiftCenter;
    const Eigen::VectorXd buShifted       = bu - shiftCenter;

    const Eigen::VectorXd absValues  = model->pivotValues.head(pointsNum).cwiseAbs();
    std::vector<int>      pivotOrder = argsort(absValues);

    int       targetPosition = -1;
    const int linearTerms    = dim + 1;

    for (int pos : pivotOrder) {
        if (pos == 0 || pos == trCenter || (pos < linearTerms && pointsNum > linearTerms)) { continue; }
        targetPosition = pos;
        break;
    }

    if (targetPosition == -1) return false;

    auto [candidates, pivotValues, candidatesAbs, successFlags] =
        model->pivotPolynomials[targetPosition]->maximizePolynomialAbs(trCenterShifted, shiftCenter, radius, blShifted,
                                                                       buShifted);

    for (int j = 0; j < candidates.cols(); ++j) {
        if (!successFlags[j] || std::abs(pivotValues(j)) < pivotThreshold) continue;

        Eigen::VectorXd candidatePoint = candidatesAbs.col(j);
        auto [fValues, evalSuccess]    = evaluateNewFValues(funcs, candidatePoint);
        if (!evalSuccess) continue;

        updateCache(model, model->pointsAbs.col(targetPosition), model->fValues.col(targetPosition));

        const Eigen::VectorXd newPointShifted = candidates.col(j);
        auto [blockBegin, blockEnd]           = getBlockBounds(targetPosition, dim, pointsNum);

        PolynomialVector updatedPolynomials = model->pivotPolynomials;
        updatedPolynomials[targetPosition]->normalizePolynomial(newPointShifted);

        updatedPolynomials[targetPosition]->orthogonalizeToOtherPolynomials(
            updatedPolynomials, targetPosition, model->pointsShifted, lastPointIdx(targetPosition));

        updatedPolynomials =
            orthogonalizeBlock(updatedPolynomials, targetPosition, newPointShifted, blockBegin, pointsNum - 1);

        const double oldPivotValue = model->pivotValues(targetPosition);
        double       newPivotValue = pivotValues(j) * oldPivotValue;

        if (!std::isfinite(newPivotValue) && std::isfinite(pivotValues(j)) && std::isfinite(oldPivotValue)) {
            newPivotValue = std::copysign(std::numeric_limits<double>::max(), newPivotValue);
            std::cerr << "Warning: Bad geometry of interpolation set for machine precision\n";
        }

        updateModelStructures(model, targetPosition, candidatePoint, newPointShifted, fValues, updatedPolynomials,
                              newPivotValue);
        return true;
    }

    return false;
}

std::tuple<bool, int> exchangePoint(TRModelPtr &model, const Eigen::VectorXd &newPoint,
                                    const Eigen::VectorXd &newFValues, double relativePivotThreshold,
                                    bool allowExchangeCenter) {

    const double pivotThreshold = std::min(1.0, model->radius) * relativePivotThreshold;
    const int    pointsCount    = static_cast<int>(model->pointsAbs.cols());

    if (pointsCount < 2) { throw std::runtime_error("Error: not enough points for exchange"); }

    const int             dim             = static_cast<int>(model->pointsAbs.rows());
    const int             centerIndex     = model->trCenter;
    const Eigen::VectorXd shiftCenter     = model->pointsAbs.col(0);
    const Eigen::VectorXd newPointShifted = newPoint - shiftCenter;

    auto [blockBegin, blockEnd] = getBlockBounds(pointsCount, dim, pointsCount);

    double maxScore      = 0.0;
    int    bestPolyIndex = -1;

    for (int polyIndex = blockEnd; polyIndex >= blockBegin; --polyIndex) {
        if (!allowExchangeCenter && polyIndex == centerIndex) continue;

        const double evalValue = model->pivotPolynomials[polyIndex]->evaluate(newPointShifted);
        const double score     = model->pivotValues(polyIndex) * evalValue;

        if (std::abs(score) > std::abs(maxScore)) {
            maxScore      = score;
            bestPolyIndex = polyIndex;
        }
    }

    if (bestPolyIndex < 0 || std::abs(maxScore) <= pivotThreshold) { return {false, 0}; }

    double       newPivotValue = maxScore;
    const double oldPivotValue = model->pivotValues(bestPolyIndex);
    const double evalValue =
        (oldPivotValue != 0.0) ? (maxScore / oldPivotValue) : std::numeric_limits<double>::infinity();

    if (!std::isfinite(newPivotValue) && std::isfinite(evalValue) && std::isfinite(oldPivotValue)) {
        newPivotValue = std::copysign(std::numeric_limits<double>::max(), newPivotValue);
        std::cerr << "Warning: Bad geometry of interpolation set for machine precision\n";
    }

    updateCache(model, model->pointsAbs.col(bestPolyIndex), model->fValues.col(bestPolyIndex));

    Eigen::MatrixXd updatedPointsShifted    = model->pointsShifted;
    updatedPointsShifted.col(bestPolyIndex) = newPointShifted;

    PolynomialVector updatedPolynomials = model->pivotPolynomials;
    updatedPolynomials[bestPolyIndex]->normalizePolynomial(newPointShifted);

    updatedPolynomials[bestPolyIndex]->orthogonalizeToOtherPolynomials(
        updatedPolynomials, bestPolyIndex, updatedPointsShifted, lastPointIdx(bestPolyIndex));
    updatedPolynomials =
        orthogonalizeBlock(updatedPolynomials, bestPolyIndex, newPointShifted, blockBegin, pointsCount - 1);

    model->pointsShifted = updatedPointsShifted;
    updateModelStructures(model, bestPolyIndex, newPoint, newPointShifted, newFValues, updatedPolynomials,
                          newPivotValue);

    return {true, bestPolyIndex};
}

int ensureImprovement(TRModelPtr &model, const Funcao &funcs, Eigen::VectorXd &bl, Eigen::VectorXd &bu,
                      const Options &options) {

    enum StatusCode { POINT_ADDED = 1, POINT_REPLACED = 2, OLD_MODEL_REBUILT = 3, MODEL_REBUILT = 4 };

    const bool modelComplete = model->isComplete();
    const bool modelPoised   = model->isLambdaPoised(options);
    const bool modelOld      = model->isOld(options);

    bool success = false;
    int  status  = 0;

    if (!modelComplete && (!modelOld || !modelPoised)) {
        success = improveModelNFP(model, funcs, bl, bu, options);
        if (success) status = POINT_ADDED;
    } else if (modelComplete && !modelOld) {
        success = chooseAndReplacePoint(model, funcs, bl, bu, options);
        if (success) status = POINT_REPLACED;
    }

    if (!success) {
        model->rebuildModel(options);
        status = modelOld ? OLD_MODEL_REBUILT : MODEL_REBUILT;
    }

    return status;
}

Eigen::MatrixXd nfp_finite_differences(const Eigen::MatrixXd &points, const Eigen::MatrixXd &fvalues,
                                       const PolynomialVector &polynomials) {
    const int       dim       = static_cast<int>(points.rows());
    const int       pointsNum = static_cast<int>(points.cols());
    Eigen::MatrixXd result    = fvalues;

    for (int m = 1; m < pointsNum; ++m) {
        const double val = polynomials[0]->evaluate(points.col(m));
        result.col(m) -= result.col(0) * val;
    }

    for (int m = dim + 1; m < pointsNum; ++m) {
        for (int n = 1; n <= dim; ++n) {
            const double val = polynomials[n]->evaluate(points.col(m));
            result.col(m) -= result.col(n) * val;
        }
    }

    return result;
}

int try2addPoint(TRModelPtr &model, const Eigen::VectorXd &newPoint, const Eigen::VectorXd &newFValues,
                 const Funcao &funcs, Eigen::VectorXd &bl, Eigen::VectorXd &bu, const Options &options) {

    constexpr int POINT_ADDED = 1;
    int           status      = 0;
    bool          pointAdded  = false;

    if (newFValues.allFinite()) {
        if (!model->isComplete()) {
            pointAdded = (addPoint(model, newPoint, newFValues, options.add_threshold) != 0);
            if (pointAdded) status = POINT_ADDED;
        }

        if (model->isComplete() || !pointAdded) { updateCache(model, newPoint, newFValues); }
    }

    if (!newFValues.allFinite() || model->isComplete() || !pointAdded) {
        status = ensureImprovement(model, funcs, bl, bu, options);
    }

    return status;
}

int changeTRCenter(TRModelPtr &model, Eigen::VectorXd &newPoint, Eigen::VectorXd &newFValues, const Options &options) {

    enum StatusCode { POINT_ADDED = 1, POINT_EXCHANGED = 2, MODEL_REBUILT = 4 };

    if (!model->isComplete()) {
        if (addPoint(model, newPoint, newFValues, options.pivot_threshold) != 0) {
            model->trCenter = static_cast<int>(model->pointsAbs.cols()) - 1;
            return POINT_ADDED;
        }
    }

    auto [exchanged, ptIndex] = exchangePoint(model, newPoint, newFValues, options.pivot_threshold, false);
    if (exchanged) {
        model->trCenter = ptIndex;
        return POINT_EXCHANGED;
    }

    const int currentSize = static_cast<int>(model->pointsAbs.cols());
    model->pointsAbs.conservativeResize(Eigen::NoChange, currentSize + 1);
    model->pointsAbs.col(currentSize) = newPoint;
    model->fValues.conservativeResize(Eigen::NoChange, currentSize + 1);
    model->fValues.col(currentSize) = newFValues;

    const Eigen::VectorXd shiftCenter = model->pointsAbs.col(0);
    model->pointsShifted.conservativeResize(Eigen::NoChange, currentSize + 1);
    model->pointsShifted.col(currentSize) = newPoint - shiftCenter;

    model->trCenter = currentSize;
    model->rebuildModel(options);
    return MODEL_REBUILT;
}

double evaluatePDescent(const Eigen::VectorXd &oldValues, const Eigen::VectorXd &newValues,
                        const Eigen::VectorXd &conLb, const Eigen::VectorXd &conUb, double mu) {

    const double objectiveImprovement = oldValues(0) - newValues(0);

    Eigen::VectorXd constraintChange(conLb.size() + conUb.size());
    constraintChange.setZero();

    for (int i = 0; i < conLb.size(); ++i) {
        if (std::isfinite(conLb(i))) {
            const double oldViolation = std::max(0.0, conLb(i) - oldValues(i + 1));
            const double newViolation = std::max(0.0, conLb(i) - newValues(i + 1));
            constraintChange(i)       = oldViolation - newViolation;
        }
    }

    for (int i = 0; i < conUb.size(); ++i) {
        if (std::isfinite(conUb(i))) {
            const double oldViolation          = std::max(0.0, oldValues(i + 1) - conUb(i));
            const double newViolation          = std::max(0.0, newValues(i + 1) - conUb(i));
            constraintChange(i + conLb.size()) = oldViolation - newViolation;
        }
    }

    return objectiveImprovement + mu * constraintChange.sum();
}