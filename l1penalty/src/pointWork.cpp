// === cleaned_tr_model_ops.cpp ===
// Drop-in replacements for your functions. Behavior preserved, edge cases fixed.

#include "../include/PolynomialVector.hpp"
#include "../include/Solvers.hpp"
#include "Eigen/Core"
#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "../include/Definitions.hpp"
#include "../include/Helpers.hpp"
#include "../include/Polynomials.hpp"
#include "../include/TRModel.hpp"
#include "../include/l1Solver.hpp"

namespace {

// Small helpers
inline bool isIndexInRange(int i, int n) { return (i >= 0) && (i < n); }

// Inclusive “last index” of points to orthogonalize against,
// given a current number of points p (0..p-1 are valid).
inline int lastPointIdx(int p) { return std::max(0, p - 1); }

} // namespace

std::tuple<PolynomialVector, double, bool>
choosePivotPolynomial(PolynomialVector      &pivotPolynomials,
                      const Eigen::MatrixXd &points,
                      int                    initialI,
                      int                    finalI,
                      double                 tol) {
    // Defensive checks
    const int numPolys  = static_cast<int>(pivotPolynomials.size());
    const int numPoints = static_cast<int>(points.cols());
    if (initialI < 0 || finalI < 0 || initialI >= numPolys || finalI >= numPolys || initialI > finalI) {
        return {pivotPolynomials, 0.0, false};
    }
    if (initialI >= numPoints) { return {pivotPolynomials, 0.0, false}; }

    // Work on a copy (matches your MATLAB semantics)
    PolynomialVector out = pivotPolynomials;

    // Number of points to orthogonalize against: 0..(initialI-1).
    // IMPORTANT: allow -1 to mean "no previous points".
    const int lastPt = initialI - 1;

    const Eigen::VectorXd incumbentPoint = points.col(initialI);

    bool          success    = false;
    double        pivotValue = 0.0;
    int           bestK      = -1;
    double        bestAbsVal = 0.0;
    PolynomialPtr bestPoly   = nullptr;

    // Search for the BEST pivot (max |val| above tol), not just the first acceptable one.
    for (int k = initialI; k <= finalI; ++k) {
        // Orthogonalize polynomial k to points 0..lastPt using the ORIGINAL set.
        // Contract: orthogonalizeToOtherPolynomials must treat lastIdx < 0 as "no-op".
        auto poly = pivotPolynomials[k]->orthogonalizeToOtherPolynomials(pivotPolynomials, k, points, lastPt);

        const double val    = poly->evaluate(incumbentPoint);
        const double absVal = std::abs(val);

        if (absVal > tol && absVal > bestAbsVal) {
            bestAbsVal = absVal;
            bestK      = k;
            bestPoly   = poly;
            pivotValue = val;     // store the value corresponding to bestPoly
            success    = true;
        }
    }

    // Apply the best pivot found (if any)
    if (success && bestK >= 0) {
        // Accept: swap in the orthogonalized candidate at initialI
        // Move original initialI polynomial to bestK
        out[bestK]    = pivotPolynomials[initialI];
        out[initialI] = bestPoly;
    }

    return {out, pivotValue, success};
}


int addPoint(TRModelPtr &model, const Eigen::VectorXd &newPoint, const Eigen::VectorXd &newFValues,
             double relativePivotThreshold) {
    const double pivotThreshold = std::min(1.0, model->radius) * relativePivotThreshold;

    const int dim   = static_cast<int>(model->pointsAbs.rows());
    const int lastP = static_cast<int>(model->pointsAbs.cols()); // current number of points

    if (lastP == 0 || model->isComplete()) {
        throw std::runtime_error("Model either empty or full. Should not be calling addPoint");
    }

    PolynomialVector pivotPolynomials = model->pivotPolynomials;
    const int        polynomialsNum   = static_cast<int>(pivotPolynomials.size());

    // Shifted new point
    const Eigen::VectorXd shiftCenter     = model->pointsAbs.col(0);
    const Eigen::VectorXd newPointShifted = newPoint - shiftCenter;

    // Extend shifted matrix
    Eigen::MatrixXd pointsShifted = model->pointsShifted;
    pointsShifted.conservativeResize(Eigen::NoChange, lastP + 1);
    pointsShifted.col(lastP) = newPointShifted;

    const int nextPosition = lastP;

    // Decide block (linear or quadratic)
    int blockBeginning = 1; // skip constant at 0
    int blockEnd       = 0; // inclusive
    if (nextPosition <= dim) {
        // linear block indices: 1..dim (inclusive)
        blockEnd = dim;
    } else {
        // quadratic block indices: (dim+1)..(polynomialsNum-1) (inclusive)
        blockBeginning = dim + 1;
        blockEnd       = polynomialsNum - 1;
    }
    blockBeginning = std::clamp(blockBeginning, 0, polynomialsNum - 1);
    blockEnd       = std::clamp(blockEnd, 0, polynomialsNum - 1);

    auto [pivotPolynomialsOut, pivotValue, success] =
        choosePivotPolynomial(pivotPolynomials, pointsShifted, nextPosition, blockEnd, pivotThreshold);

    if (!success) return 0;

    pivotPolynomials = std::move(pivotPolynomialsOut);

    // Normalize and re-orthogonalize new pivot at nextPosition
    pivotPolynomials[nextPosition]->normalizePolynomial(newPointShifted);
    pivotPolynomials[nextPosition] = pivotPolynomials[nextPosition]->orthogonalizeToOtherPolynomials(
        pivotPolynomials, nextPosition, pointsShifted, lastPointIdx(nextPosition));

    // Orthogonalize the block against the new point
    pivotPolynomials =
        orthogonalizeBlock(pivotPolynomials, nextPosition, newPointShifted, blockBeginning, nextPosition - 1);

    // Update model structures (extend once each, then assign)
    model->pointsShifted = pointsShifted;

    model->fValues.conservativeResize(Eigen::NoChange, nextPosition + 1);
    model->fValues.col(nextPosition) = newFValues;

    model->pointsAbs.conservativeResize(Eigen::NoChange, nextPosition + 1);
    model->pointsAbs.col(nextPosition) = newPoint;

    model->modelingPolynomials.clear();
    model->pivotPolynomials = pivotPolynomials;

    model->pivotValues.conservativeResize(nextPosition + 1);
    model->pivotValues(nextPosition) = pivotValue;

    return 1;
}

bool chooseAndReplacePoint(TRModelPtr &model, const Funcao &funcs, Eigen::VectorXd &bl, Eigen::VectorXd &bu,
                           const Options &options) {
    const double pivotThreshold = options.exchange_threshold;
    const double radius         = model->radius;

    const Eigen::MatrixXd pointsShifted = model->pointsShifted;
    const int             dim           = static_cast<int>(pointsShifted.rows());
    const int             pointsNum     = static_cast<int>(pointsShifted.cols());

    const Eigen::VectorXd shiftCenter = model->pointsAbs.col(0);
    const int             trCenter    = model->trCenter;
    const Eigen::VectorXd trCenterX   = model->pointsShifted.col(trCenter);

    Eigen::VectorXd  pivotValues      = model->pivotValues;
    PolynomialVector pivotPolynomials = model->pivotPolynomials;

    const int linearTerms = dim + 1;

    // Default bounds if empty
    if (bl.size() == 0) bl = Eigen::VectorXd::Constant(dim, -std::numeric_limits<double>::infinity());
    if (bu.size() == 0) bu = Eigen::VectorXd::Constant(dim, std::numeric_limits<double>::infinity());

    const Eigen::VectorXd blShifted = bl - shiftCenter;
    const Eigen::VectorXd buShifted = bu - shiftCenter;

    // Smallest |pivot| position among current points
    Eigen::VectorXd  absValues  = pivotValues.head(pointsNum).cwiseAbs();
    std::vector<int> pivotOrder = argsort(absValues);
    const int        pos        = pivotOrder.empty() ? 0 : pivotOrder[0];

    bool success = false;

    // Guard against replacing constant (0), TR center, or protected early linear terms
    if (pos == 0 || pos == trCenter || (pos < linearTerms && pointsNum > linearTerms)) {
        return false; // rebuild is better
    }

    auto [candShifted, candPivots, candAbs, okFlags] =
        pivotPolynomials[pos]->maximizePolynomialAbs(trCenterX, shiftCenter, radius, blShifted, buShifted);

    bool            pointFound    = false;
    bool            fSucceeded    = false;
    double          newPivotValue = 0.0;
    Eigen::VectorXd newPointShifted, newPointAbs, newFValues;

    for (int j = 0; j < candShifted.cols(); ++j) {
        if (!okFlags[j]) continue;

        const double v = candPivots(j);
        if (std::abs(v) < pivotThreshold) continue;

        pointFound      = true;
        newPivotValue   = v;
        newPointShifted = candShifted.col(j);
        newPointAbs     = candAbs.col(j);

        auto [fVals, ok] = evaluateNewFValues(funcs, newPointAbs);
        fSucceeded       = ok;
        newFValues       = fVals;
        if (fSucceeded) break;
    }

    success = pointFound && fSucceeded;
    if (!success) return false;

    // Compute block range for orthogonalization
    int blockBeginning = 1;
    int blockEnd       = pointsNum - 1; // inclusive last existing point index
    if (pos <= dim) {
        blockBeginning = 1;
        blockEnd       = std::min(pointsNum - 1, dim);
    } else {
        blockBeginning = dim + 1;
        blockEnd       = pointsNum - 1;
    }

    // Normalize/orthogonalize pos
    pivotPolynomials[pos]->normalizePolynomial(newPointShifted);
    pivotPolynomials[pos] =
        pivotPolynomials[pos]->orthogonalizeToOtherPolynomials(pivotPolynomials, pos, pointsShifted, blockEnd);

    pivotPolynomials = orthogonalizeBlock(pivotPolynomials, pos, newPointShifted, blockBeginning, pointsNum - 1);

    // Cache the point being replaced at pos
    {
        const int cacheCap = static_cast<int>(model->cacheMax);
        const int curCache = static_cast<int>(model->cachedPoints.cols());
        const int newSize  = std::min(curCache + 1, cacheCap);

        Eigen::MatrixXd newCachePts(model->cachedPoints.rows(), newSize);
        Eigen::MatrixXd newCacheF(model->cachedFValues.rows(), newSize);

        // Insert old pos at front
        newCachePts.col(0) = model->pointsAbs.col(pos);
        newCacheF.col(0)   = model->fValues.col(pos);

        // Shift old cache (truncate if needed)
        const int copyCnt = std::min(curCache, newSize - 1);
        if (copyCnt > 0) {
            newCachePts.rightCols(copyCnt) = model->cachedPoints.leftCols(copyCnt);
            newCacheF.rightCols(copyCnt)   = model->cachedFValues.leftCols(copyCnt);
        }
        model->cachedPoints  = std::move(newCachePts);
        model->cachedFValues = std::move(newCacheF);
    }

    // Commit replacement
    model->pointsAbs.col(pos)     = newPointAbs;
    model->pointsShifted.col(pos) = newPointShifted;
    model->fValues.col(pos)       = newFValues;
    model->pivotPolynomials       = pivotPolynomials;

    // Update pivot value (keep MATLAB semantics: multiply by old)
    const double oldPivotValue = model->pivotValues(pos);
    double       newPV         = newPivotValue * oldPivotValue;
    if (!std::isfinite(newPV) && std::isfinite(newPivotValue) && std::isfinite(oldPivotValue)) {
        newPV = std::copysign(std::numeric_limits<double>::max(), newPV);
        std::cerr << "Warning: Bad geometry of interpolation set for machine precision\n";
    }
    model->pivotValues(pos) = newPV;

    model->modelingPolynomials.clear();
    return true;
}
bool improveModelNFP(TRModelPtr &model, const Funcao &funcs, Eigen::VectorXd &bl, Eigen::VectorXd &bu,
                     const Options &options) {
    const double relPivotThreshold = options.pivot_threshold;
    const double tolRadius         = options.tol_radius;
    const double radius            = model->radius;

    const double pivotThreshold = relPivotThreshold * std::min(1.0, radius);

    Eigen::MatrixXd pointsShifted = model->pointsShifted;
    const int       dim           = static_cast<int>(pointsShifted.rows());
    const int       pIni          = static_cast<int>(pointsShifted.cols()); // current #points (0-based count)

    if (pIni == 0 || model->isComplete()) {
        throw std::runtime_error("Model either empty or full. Should not be calling improve_model_nfp");
    }
    if (model->isOld(options) && model->isLambdaPoised(options)) {
        throw std::runtime_error("Model too old. Should be calling rebuild model");
    }

    const Eigen::VectorXd shiftCenter = model->pointsAbs.col(0);
    const int             trCenter    = model->trCenter;
    const Eigen::VectorXd trCenterPt  = pointsShifted.col(trCenter);

    // Default bounds (MATLAB: set to +/-inf if empty)
    if (bl.size() == 0) bl = Eigen::VectorXd::Constant(dim, -std::numeric_limits<double>::infinity());
    if (bu.size() == 0) bu = Eigen::VectorXd::Constant(dim, std::numeric_limits<double>::infinity());

    const Eigen::VectorXd blShifted = bl - shiftCenter;
    const Eigen::VectorXd buShifted = bu - shiftCenter;

    PolynomialVector pivotPolynomials = model->pivotPolynomials;

    // Block beginning as in MATLAB (1-based: 2 or dim+2) → (0-based: 1 or dim+1)
    int blockBeginning = (pIni < dim + 1) ? 1 : (dim + 1);

    // In MATLAB: next_position = p_ini + 1 (1-based index of the new slot)
    // Here (0-based): nextPosition = pIni
    const int nextPosition = pIni;

    const int blockEndForSearch = nextPosition;

    double radiusUsed = radius;

    bool            pointFound = false, fSucceeded = false;
    double          newPivotValue = 0.0; // RAW value from maximize (no extra multiply)
    Eigen::VectorXd newFValues(model->fValues.rows());
    Eigen::VectorXd newPointAbs(dim), newPointShifted(dim);

    for (int attempts = 0; attempts < 3; ++attempts) {
        // polyI runs only at nextPosition to mirror MATLAB "not really iterating"
        for (int polyI = nextPosition; polyI <= blockEndForSearch; ++polyI) {
            // Orthogonalize polyI against existing points 0..(pIni-1)
            auto poly = pivotPolynomials[polyI]->orthogonalizeToOtherPolynomials(
                pivotPolynomials, polyI, pointsShifted, /*last existing idx*/ lastPointIdx(pIni));

            // In MATLAB maximize_polynomial_abs takes shift/unshift lambdas; here we use the
            // implementation that returns shifted, abs, and raw pivot values consistently.
            auto [candShifted, candPivots, candAbs, okFlags] =
                poly->maximizePolynomialAbs(trCenterPt, shiftCenter, radiusUsed, blShifted, buShifted);

            pointFound = false;
            fSucceeded = false;

            for (int j = 0; j < candShifted.cols(); ++j) {
                if (!okFlags[j]) continue;

                // MATLAB uses the raw new_pivot_value returned by maximize (no extra multiply)
                const double rawPivot = candPivots(j);
                if (std::abs(rawPivot) < pivotThreshold) {
                    continue; // below threshold, skip
                } else {
                    pointFound = true;
                }

                newPivotValue   = rawPivot; // store RAW pivot for this slot
                newPointShifted = candShifted.col(j);
                newPointAbs     = candAbs.col(j);

                auto [fVals, ok] = evaluateNewFValues(funcs, newPointAbs);
                fSucceeded       = ok;
                newFValues       = fVals;

                if (fSucceeded) break; // accept this candidate
            }

            if (pointFound && fSucceeded) {
                // Update this polynomial into the set, then swap into nextPosition
                pivotPolynomials[polyI] = poly;
                pivotPolynomials.swapElements(nextPosition, polyI);

                // Add point at nextPosition
                pointsShifted.conservativeResize(Eigen::NoChange, nextPosition + 1);
                pointsShifted.col(nextPosition) = newPointShifted;

                break; // stop trying other polynomials (there aren't, by construction)
            }
            // else: try another (none in this mapping)
        }

        if (pointFound && fSucceeded) {
            break; // success
        } else if (pointFound && radiusUsed > tolRadius) {
            // Found candidates above threshold but evaluation failed → shrink radius and retry
            radiusUsed *= 0.5;
        } else {
            break; // nothing acceptable or radius too small → give up
        }
    }

    const bool success = (pointFound && fSucceeded);
    if (!success) return false;

    // Post-acceptance: normalize at nextPosition and re-orthogonalize
    pivotPolynomials[nextPosition]->normalizePolynomial(newPointShifted);
    pivotPolynomials[nextPosition] = pivotPolynomials[nextPosition]->orthogonalizeToOtherPolynomials(
        pivotPolynomials, nextPosition, pointsShifted, /*last existing idx*/ lastPointIdx(pIni));

    // Orthogonalize the present block (deferring subsequent ones)
    // MATLAB passes [block_beginning, p_ini] (1-based). Here use [blockBeginning, pIni-1] (0-based).
    if (pIni > 0) {
        pivotPolynomials = orthogonalizeBlock(pivotPolynomials, nextPosition, newPointShifted, blockBeginning,
                                              /*end*/ nextPosition - 1);
    }

    // Update model and store RAW new pivot value at nextPosition (no extra multiply)
    model->pointsAbs.conservativeResize(Eigen::NoChange, nextPosition + 1);
    model->pointsAbs.col(nextPosition) = newPointAbs;

    model->pointsShifted = pointsShifted;

    model->fValues.conservativeResize(Eigen::NoChange, nextPosition + 1);
    model->fValues.col(nextPosition) = newFValues;

    model->pivotPolynomials = pivotPolynomials;

    model->pivotValues.conservativeResize(nextPosition + 1);
    model->pivotValues(nextPosition) = newPivotValue; // RAW pivot from maximize

    model->modelingPolynomials.clear();
    return true;
}

int ensureImprovement(TRModelPtr &model, const Funcao &funcs, Eigen::VectorXd &bl, Eigen::VectorXd &bu,
                      const Options &options) {
    // Status constants
    constexpr int STATUS_POINT_ADDED       = 1;
    constexpr int STATUS_POINT_REPLACED    = 2;
    constexpr int STATUS_OLD_MODEL_REBUILT = 3;
    constexpr int STATUS_MODEL_REBUILT     = 4;

    const bool modelComplete = model->isComplete();
    const bool modelFl       = model->isLambdaPoised(options);
    const bool modelOld      = model->isOld(options);

    bool success  = false;
    int  exitflag = 0;

    if (!modelComplete && (!modelOld || !modelFl)) {
        success = improveModelNFP(model, funcs, bl, bu, options);
        if (success) exitflag = STATUS_POINT_ADDED;
    } else if (modelComplete && !modelOld) {
        success = chooseAndReplacePoint(model, funcs, bl, bu, options);
        if (success) exitflag = STATUS_POINT_REPLACED;
    }

    if (!success) {
        model->rebuildModel(options);
        exitflag = modelOld ? STATUS_OLD_MODEL_REBUILT : STATUS_MODEL_REBUILT;
    }

    return exitflag;
}
std::tuple<bool, int>
exchangePoint(TRModelPtr &model,
              const Eigen::VectorXd &newPoint,
              const Eigen::VectorXd &newFValues,
              double                 relativePivotThreshold,
              bool                   allowExchangeCenter) {
    const double pivotThreshold = std::min(1.0, model->radius) * relativePivotThreshold;

    const int dim   = static_cast<int>(model->pointsAbs.rows());
    const int lastP = static_cast<int>(model->pointsAbs.cols());
    if (lastP < 2) { throw std::runtime_error("Error: not enough points for exchange"); }

    PolynomialVector pivotPolynomials = model->pivotPolynomials;
    const int        centerI          = model->trCenter;

    const Eigen::VectorXd shiftCenter     = model->pointsAbs.col(0);
    const Eigen::VectorXd newPointShifted = newPoint - shiftCenter;
    Eigen::MatrixXd       pointsShifted   = model->pointsShifted;

    // Compute block range (same policy as before)
    int blockBeginning = 1;
    int blockEnd       = 0; // inclusive
    if (lastP <= dim + 1) {
        blockBeginning = 1;
        blockEnd       = std::min(dim, lastP - 1);
    } else {
        blockBeginning = dim + 1;
        blockEnd       = lastP - 1;
    }

    // Select polynomial by maximizing |pivotValues[i] * evaluate_i(newPointShifted)|
    double maxScore = 0.0;
    int    maxPolyI = -1;

    for (int polyI = blockEnd; polyI >= blockBeginning; --polyI) {
        if (!allowExchangeCenter && polyI == centerI) continue;

        const double valEval = pivotPolynomials[polyI]->evaluate(newPointShifted);
        const double score   = model->pivotValues(polyI) * valEval; // composed score ONCE
        if (std::abs(score) > std::abs(maxScore)) {
            maxScore = score;
            maxPolyI = polyI;
        }
    }

    // New pivot value is exactly the composed score (no double multiply).
    double newPivotVal = 0.0;
    if (maxPolyI >= 0) {
        newPivotVal = maxScore;
        // Robustness warning if composition blew up
        const double oldPV   = model->pivotValues(maxPolyI);
        const double valEval = (oldPV != 0.0) ? (maxScore / oldPV) : std::numeric_limits<double>::infinity();
        if (!std::isfinite(newPivotVal) && std::isfinite(valEval) && std::isfinite(oldPV)) {
            newPivotVal = std::copysign(std::numeric_limits<double>::max(), newPivotVal);
            std::cerr << "Warning: Bad geometry of interpolation set for machine precision\n";
        }
    }

    bool succeeded = false;
    int  ptI       = -1;

    if (maxPolyI >= 0 && std::abs(newPivotVal) > pivotThreshold) {
        // Insert new shifted point at maxPolyI
        pointsShifted.col(maxPolyI) = newPointShifted;

        // Normalize and orthogonalize the chosen polynomial in place
        pivotPolynomials[maxPolyI]->normalizePolynomial(newPointShifted);
        pivotPolynomials[maxPolyI] =
            pivotPolynomials[maxPolyI]->orthogonalizeToOtherPolynomials(
                pivotPolynomials, maxPolyI, pointsShifted, lastP - 1);

        // Orthogonalize the block against the new point
        pivotPolynomials =
            orthogonalizeBlock(pivotPolynomials, maxPolyI, newPointShifted, blockBeginning, lastP - 1);

        // Cache old point at front (bounded size)
        const int cacheCap = static_cast<int>(model->cacheMax);
        const int curCache = static_cast<int>(model->cachedPoints.cols());
        const int newSize  = std::min(curCache + 1, cacheCap);

        Eigen::MatrixXd newCachePts(model->cachedPoints.rows(), newSize);
        Eigen::MatrixXd newCacheF(model->cachedFValues.rows(), newSize);

        newCachePts.col(0) = model->pointsAbs.col(maxPolyI);
        newCacheF.col(0)   = model->fValues.col(maxPolyI);

        const int copyCnt = std::min(curCache, newSize - 1);
        if (copyCnt > 0) {
            newCachePts.rightCols(copyCnt) = model->cachedPoints.leftCols(copyCnt);
            newCacheF.rightCols(copyCnt)   = model->cachedFValues.leftCols(copyCnt);
        }
        model->cachedPoints  = std::move(newCachePts);
        model->cachedFValues = std::move(newCacheF);

        // Commit new point
        model->pointsAbs.col(maxPolyI) = newPoint;
        model->fValues.col(maxPolyI)   = newFValues;

        model->pointsShifted         = pointsShifted;
        model->pivotPolynomials      = pivotPolynomials;
        model->pivotValues(maxPolyI) = newPivotVal; // store composed score ONCE
        model->modelingPolynomials.clear();

        succeeded = true;
        ptI       = maxPolyI;
    }

    return {succeeded, ptI < 0 ? 0 : ptI};
}


Eigen::MatrixXd nfp_finite_differences(const Eigen::MatrixXd &points, const Eigen::MatrixXd &fvalues,
                                       const PolynomialVector &polynomials) {
    const int dim       = static_cast<int>(points.rows());
    const int pointsNum = static_cast<int>(points.cols());

    Eigen::MatrixXd lAlpha = fvalues;

    // Remove constant polynomial contribution
    for (int m = 1; m < pointsNum; ++m) {
        const double val = polynomials[0]->evaluate(points.col(m));
        lAlpha.col(m) -= lAlpha.col(0) * val;
    }

    // Remove degree-1 (linear) contributions
    for (int m = dim + 1; m < pointsNum; ++m) {
        for (int n = 1; n <= dim; ++n) {
            const double val = polynomials[n]->evaluate(points.col(m));
            lAlpha.col(m) -= lAlpha.col(n) * val;
        }
    }

    return lAlpha;
}

inline bool hasNonFinite(const Eigen::VectorXd &v) {
    for (int i = 0; i < v.size(); ++i)
        if (!std::isfinite(v(i))) return true;
    return false;
}

int try2addPoint(TRModelPtr &model, const Eigen::VectorXd &newPoint, const Eigen::VectorXd &newFValues,
                 const Funcao &funcs, Eigen::VectorXd &bl, Eigen::VectorXd &bu, const Options &options) {
    constexpr int STATUS_POINT_ADDED = 1;

    int  exitflag   = 0;
    bool pointAdded = false;

    const bool goodFValues = !hasNonFinite(newFValues);

    if (goodFValues) {
        const bool modelIsComplete = model->isComplete();

        if (!modelIsComplete) {
            const double relativePivotThreshold = options.add_threshold;
            pointAdded                          = (addPoint(model, newPoint, newFValues, relativePivotThreshold) != 0);
            exitflag                            = STATUS_POINT_ADDED;
        }

        if (modelIsComplete || !pointAdded) {
            // Push into cache (prepend; bounded capacity handled elsewhere)
            const int       cacheSize = static_cast<int>(model->cachedPoints.cols());
            Eigen::MatrixXd newCachePts(model->cachedPoints.rows(), cacheSize + 1);
            Eigen::MatrixXd newCacheF(model->cachedFValues.rows(), cacheSize + 1);

            newCachePts.col(0) = newPoint;
            newCacheF.col(0)   = newFValues;

            if (cacheSize > 0) {
                newCachePts.rightCols(cacheSize) = model->cachedPoints;
                newCacheF.rightCols(cacheSize)   = model->cachedFValues;
            }
            model->cachedPoints  = std::move(newCachePts);
            model->cachedFValues = std::move(newCacheF);
        }
    }

    if (!goodFValues || model->isComplete() || !pointAdded) {
        exitflag = ensureImprovement(model, funcs, bl, bu, options);
    }

    return exitflag;
}

int changeTRCenter(TRModelPtr &model, Eigen::VectorXd &newPoint, Eigen::VectorXd &newFValues, const Options &options) {
    constexpr int STATUS_POINT_ADDED     = 1;
    constexpr int STATUS_POINT_EXCHANGED = 2;
    constexpr int STATUS_MODEL_REBUILT   = 4;

    bool pointAdded = false;
    int  exitflag   = 0;

    const bool modelIsComplete = model->isComplete();

    if (!modelIsComplete) {
        const double relativePivotThreshold = options.pivot_threshold;
        pointAdded                          = (addPoint(model, newPoint, newFValues, relativePivotThreshold) != 0);
        if (pointAdded) {
            model->trCenter = static_cast<int>(model->pointsAbs.cols()) - 1; // last
            exitflag        = STATUS_POINT_ADDED;
        }
    }

    if (modelIsComplete || !pointAdded) {
        const double relativePivotThreshold = options.pivot_threshold;
        auto [exchanged, ptIndex]           = exchangePoint(model, newPoint, newFValues, relativePivotThreshold, false);

        if (exchanged) {
            model->trCenter = ptIndex;
            exitflag        = STATUS_POINT_EXCHANGED;
        } else {
            // Force-insert then rebuild (keep center last)
            const int currentSize = static_cast<int>(model->pointsAbs.cols());

            model->pointsAbs.conservativeResize(Eigen::NoChange, currentSize + 1);
            model->pointsAbs.col(currentSize) = newPoint;

            model->fValues.conservativeResize(Eigen::NoChange, currentSize + 1);
            model->fValues.col(currentSize) = newFValues;

            const Eigen::VectorXd shiftCenter = model->pointsAbs.col(0);
            model->pointsShifted.conservativeResize(Eigen::NoChange, currentSize + 1);
            model->pointsShifted.col(currentSize) = newPoint - shiftCenter;

            model->trCenter = static_cast<int>(model->pointsAbs.cols()) - 1;
            model->rebuildModel(options);
            exitflag = STATUS_MODEL_REBUILT;
        }
    }

    return exitflag;
}

double evaluatePDescent(const Eigen::VectorXd &oldValues, const Eigen::VectorXd &newValues,
                        const Eigen::VectorXd &conLb, const Eigen::VectorXd &conUb, double mu) {
    // Objective improvement
    const double fChange = oldValues(0) - newValues(0);

    // Build constraint violation vectors (lb slack >=0, ub slack >=0)
    Eigen::VectorXd oldC(conLb.size() + conUb.size());
    Eigen::VectorXd newC(conLb.size() + conUb.size());
    oldC.setZero();
    newC.setZero();

    for (int i = 0; i < conLb.size(); ++i) {
        if (std::isfinite(conLb(i))) {
            oldC(i) = conLb(i) - oldValues(i + 1);
            newC(i) = conLb(i) - newValues(i + 1);
        }
    }
    for (int i = 0; i < conUb.size(); ++i) {
        if (std::isfinite(conUb(i))) {
            oldC(i + conLb.size()) = oldValues(i + 1) - conUb(i);
            newC(i + conLb.size()) = newValues(i + 1) - conUb(i);
        }
    }

    // Positive parts only
    const Eigen::ArrayXd posOld = oldC.array().max(0.0);
    const Eigen::ArrayXd posNew = newC.array().max(0.0);

    // NOTE: sorting before sum was a no-op; removed for correctness & speed.
    const double cChange = (posOld - posNew).sum();

    return fChange + mu * cChange;
}

std::tuple<Eigen::VectorXd, double, int> minimizeTr(const PolynomialPtr &polynomial, const Eigen::VectorXd &xTrCenter,
                                                    double radius, const Eigen::VectorXd &bl,
                                                    const Eigen::VectorXd &bu) {
    const int dim = static_cast<int>(xTrCenter.size());

    // Local bounds with defaults if empty
    const Eigen::VectorXd blLocal =
        (bl.size() == 0) ? Eigen::VectorXd::Constant(dim, -std::numeric_limits<double>::infinity()) : bl;
    const Eigen::VectorXd buLocal =
        (bu.size() == 0) ? Eigen::VectorXd::Constant(dim, std::numeric_limits<double>::infinity()) : bu;

    // Trust region box
    const Eigen::VectorXd blTr = xTrCenter.array() - radius;
    const Eigen::VectorXd buTr = xTrCenter.array() + radius;

    const Eigen::VectorXd blMod = blLocal.cwiseMax(blTr);
    const Eigen::VectorXd buMod = buLocal.cwiseMin(buTr);

    // Restore feasibility at TR center (copy)
    Eigen::VectorXd x0 = xTrCenter;
    for (int i = 0; i < dim; ++i) {
        if (x0(i) < blLocal(i)) x0(i) = blLocal(i);
        if (x0(i) > buLocal(i)) x0(i) = buLocal(i);
    }

    auto [c, g, H] = polynomial->getBalancedTerms(); // (c, g, H), H should be symmetric

    // Step away from stationary point
    if ((g + H * x0).norm() < 1e-4) {
        // Try mid of box
        x0 = 0.5 * (blMod + buMod);

        if ((g + H * x0).norm() < 1e-4) {
            // Move along the min-eig direction
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
            es.compute(H);
            const auto &eigvals = es.eigenvalues();
            const auto &eigvecs = es.eigenvectors();

            int    minIdx = 0;
            double minVal = eigvals(0);
            for (int i = 1; i < eigvals.size(); ++i) {
                if (eigvals(i) < minVal) {
                    minVal = eigvals(i);
                    minIdx = i;
                }
            }
            const Eigen::VectorXd v = eigvecs.col(minIdx).real();
            if (v.norm() > 1e-8) {
                double step = std::numeric_limits<double>::infinity();
                for (int i = 0; i < dim; ++i) {
                    if (std::abs(v(i)) < 1e-8) continue;
                    const double aL = (blMod(i) - x0(i)) / v(i);
                    const double aU = (buMod(i) - x0(i)) / v(i);
                    step            = std::min(step, std::min(std::abs(aL), std::abs(aU)));
                }
                if (std::isfinite(step)) x0 += step * v;
            }
        }
    }

    // Solve bound-constrained QP
    GurobiSolver solver;
    auto [x, fval, ok] = solver.solveQuadraticProblem(H, g, c, Eigen::VectorXd(), Eigen::VectorXd(), Eigen::VectorXd(),
                                                      Eigen::VectorXd(), blMod, buMod, x0);

    // Final projection to (original) bounds
    x = projectToBounds(x, blLocal, buLocal);

    return {x, fval, ok ? 1 : 0};
}
