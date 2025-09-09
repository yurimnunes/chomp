#pragma once

#include "Definitions.hpp"
#include "Eigen/Core"
#include "PolynomialVector.hpp"
#include "pointWork.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <fmt/core.h>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "Overloads.hpp"
#include "TRModel.hpp"

class Polynomial : public std::enable_shared_from_this<Polynomial> {
public:
    Eigen::VectorXd coefficients;

    // Constructors
    Polynomial() = default;
    Polynomial(double c, const Eigen::VectorXd &g, const Eigen::MatrixXd &H) { matricesToPolynomial(c, g, H); }
    explicit Polynomial(const Eigen::VectorXd &coeffs) : coefficients(coeffs) {}

    static Polynomial Zero(int size) { return Polynomial(Eigen::VectorXd::Zero(size)); }

    Eigen::VectorXd       &getCoefficients() { return coefficients; }
    const Eigen::VectorXd &getCoefficients() const { return coefficients; }

    void normalizePolynomial(const Eigen::VectorXd &point, double eps = 1e-14) {
        const double val = evaluate(point);
        if (std::abs(val) <= eps) { throw std::runtime_error("normalizePolynomial: value is ~0 at point"); }
        coefficients /= val;
    }

    // Make this zero at x by adding a multiple of p2
    // Make (this)(x) ≈ 0 by subtracting (px/p2x) * p2, up to 2 corrective passes.
    // Returns a new polynomial; leaves original unchanged.
    PolynomialPtr zeroAtPoint(const PolynomialPtr &p2, const Eigen::VectorXd &x, double eps = 1e-12,
                              int max_iters = 2) const {
        // Start from a copy of *this
        PolynomialPtr p = std::make_shared<Polynomial>(*this);
        if (!p2) return p;

        const double p2x = p2->evaluate(x);
        if (std::abs(p2x) <= eps) {
            // MATLAB version doesn’t guard; we do to avoid division-by-zero blowups.
            // You could warn instead of returning unchanged.
            return p;
        }

        for (int iter = 0; iter < max_iters; ++iter) {
            const double px = p->evaluate(x);
            if (std::abs(px) <= eps) break;               // already ~0 at x
            const double factor = -px / p2x;              // subtract multiple of p2
            p->coefficients += p2->coefficients * factor; // add_p(multiply_p(...))
        }
        return p;
    }

    PolynomialPtr orthogonalizeToOtherPolynomials(const PolynomialVector &allPolynomials, int polyIndex,
                                                  const Eigen::MatrixXd &points, int lastPt) const {
        PolynomialPtr polynomial = std::make_shared<Polynomial>(*this);
        const int     P          = points.cols();
        lastPt                   = std::max(-1, std::min(lastPt, P - 1));
        for (int n = 0; n <= lastPt; ++n) {
            if (n != polyIndex) { polynomial = polynomial->zeroAtPoint(allPolynomials[n], points.col(n)); }
        }
        return polynomial;
    }

    PolynomialPtr orthogonalizeRange(const PolynomialVector &allPolynomials, int polyIndex,
                                     const Eigen::MatrixXd &points, int firstPt, int lastPt) const {
        PolynomialPtr polynomial = std::make_shared<Polynomial>(*this);
        const int     P          = points.cols();
        firstPt                  = std::max(0, std::min(firstPt, P - 1));
        lastPt                   = std::max(firstPt, std::min(lastPt, P - 1));
        for (int n = firstPt; n <= lastPt; ++n) {
            if (n != polyIndex) { polynomial = polynomial->zeroAtPoint(allPolynomials[n], points.col(n)); }
        }
        return polynomial;
    }

    // Robustly infer dimension from number of coefficients
    static int dimensionFromCoeffsCount(Eigen::Index N) {
        if (N < 1) throw std::runtime_error("dimensionFromCoeffsCount: invalid size");
        // Find d s.t. T(d+1) = (d+1)(d+2)/2 = N  -> quadratic in d
        // Solve in integers by scanning around the double root.
        const double dd  = std::floor((-3.0 + std::sqrt(1.0 + 8.0 * static_cast<double>(N))) / 2.0 + 1e-12);
        int          d   = static_cast<int>(dd);
        auto         tri = [](long long k) { return (k * (k + 1)) / 2; };
        // Check neighbors in case of tiny FP drift
        for (int delta : {0, -1, 1}) {
            int cand = d + delta;
            if (cand >= 0) {
                long long terms = tri(cand + 1);
                if (terms == N) return cand;
            }
        }
        throw std::runtime_error("Coefficient vector has inconsistent size for any dimension");
    }

    std::tuple<double, Eigen::VectorXd, Eigen::MatrixXd> getTerms() const {
        const int dimension = dimensionFromCoeffsCount(coefficients.size());
        const int n_terms   = (dimension + 1) * (dimension + 2) / 2;
        if (coefficients.size() != n_terms) { throw std::runtime_error("getTerms: wrong coefficient size"); }

        const double    c = coefficients(0);
        Eigen::VectorXd g = coefficients.segment(1, dimension);
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dimension, dimension);

        Eigen::Index idx = dimension + 1;
        for (int k = 0; k < dimension; ++k) {
            for (int m = 0; m <= k; ++m) {
                H(k, m) = coefficients(idx);
                H(m, k) = H(k, m);
                ++idx;
            }
        }
        return {c, g, H};
    }

    std::tuple<double, Eigen::VectorXd, Eigen::MatrixXd> getBalancedTerms() const {
        auto [c, g, H] = getTerms();

        Eigen::MatrixXd off = H;
        off.diagonal().setZero();
        const int degree = static_cast<int>(H.rows());

        const double gamma = 0.9;
        bool         changed;
        do {
            changed = false;
            for (int i = 0; i < degree; ++i) {
                const double row1 = off.row(i).lpNorm<1>();
                const double col1 = off.col(i).lpNorm<1>();
                if (row1 == 0.0 || col1 == 0.0) continue;
                int exp2 = 0;
                std::frexp(row1 / col1, &exp2);
                exp2 /= 2;
                if (exp2 != 0) {
                    const double col_new = std::ldexp(col1, exp2);
                    const double row_new = std::ldexp(row1, -exp2);
                    if (col_new + row_new < gamma * (col1 + row1)) {
                        changed = true;
                        off.row(i) *= std::ldexp(1.0, -exp2);
                        off.col(i) *= std::ldexp(1.0, exp2);
                    }
                }
            }
        } while (changed);

        off.diagonal() = H.diagonal();
        return {c, g, off};
    }

    double evaluate(const Eigen::VectorXd &point) const {
        auto [c, g, H]         = getTerms();
        const double linear    = g.dot(point);
        const double quadratic = 0.5 * (point.transpose() * H * point).value();
        return c + linear + quadratic;
    }

    double operator()(const Eigen::VectorXd &point) const { return evaluate(point); }
    std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, std::vector<bool>>
    maximizePolynomialAbs(const Eigen::VectorXd &trCenter, const Eigen::VectorXd &shiftCenter, double radius,
                          const Eigen::VectorXd &lb, const Eigen::VectorXd &ub) {
        // Use an owned copy to be safe even if *this isn’t managed by a shared_ptr.
        PolynomialPtr thisPoly = std::make_shared<Polynomial>(*this);

        // MIN side
        auto [newPointMin, pivotMin, ExitFlagMin] = minimizeTr(thisPoly, trCenter, radius, lb, ub);

        const Eigen::VectorXd bl_unshifted = lb + shiftCenter;
        const Eigen::VectorXd bu_unshifted = ub + shiftCenter;

        auto shift   = [&](const Eigen::VectorXd &x) { return x - shiftCenter; };
        auto unshift = [&](const Eigen::VectorXd &x) {
            Eigen::VectorXd y = x + shiftCenter;
            for (int i = 0; i < y.size(); ++i) { y[i] = std::max(std::min(y[i], bu_unshifted(i)), bl_unshifted(i)); }
            return y;
        };

        Eigen::VectorXd newPointMinAbs = unshift(newPointMin);
        newPointMin                    = shift(newPointMinAbs);
        pivotMin                       = thisPoly->evaluate(newPointMin);

        // MAX side (minimize -p)
        PolynomialPtr polynomialMax = std::make_shared<Polynomial>(*thisPoly);
        polynomialMax->coefficients *= -1.0; // negate the polynomial

        auto [newPointMax, pivotMax, ExitFlagMax] = minimizeTr(polynomialMax, trCenter, radius, lb, ub);

        Eigen::VectorXd newPointMaxAbs = unshift(newPointMax);
        newPointMax                    = shift(newPointMaxAbs);
        pivotMax                       = thisPoly->evaluate(newPointMax);

        const int       n = static_cast<int>(newPointMax.size());
        Eigen::MatrixXd newPoints(n, 2);
        newPoints.setZero();
        Eigen::MatrixXd newPointsAbs(n, 2);
        newPointsAbs.setZero();
        Eigen::VectorXd newPivotValues(2);
        newPivotValues.setZero();
        std::vector<bool> exitFlags(2, false);

        if (ExitFlagMin >= 0) {
            if (ExitFlagMax >= 0) {
                if (std::abs(pivotMax) >= std::abs(pivotMin)) {
                    newPoints.col(0)    = newPointMax;
                    newPoints.col(1)    = newPointMin;
                    newPivotValues(0)   = pivotMax;
                    newPivotValues(1)   = pivotMin;
                    newPointsAbs.col(0) = newPointMaxAbs;
                    newPointsAbs.col(1) = newPointMinAbs;
                    exitFlags[0] = exitFlags[1] = true;
                } else {
                    newPoints.col(0)    = newPointMin;
                    newPoints.col(1)    = newPointMax;
                    newPivotValues(0)   = pivotMin;
                    newPivotValues(1)   = pivotMax;
                    newPointsAbs.col(0) = newPointMinAbs;
                    newPointsAbs.col(1) = newPointMaxAbs;
                    exitFlags[0] = exitFlags[1] = true;
                }
            } else {
                newPoints.col(0)    = newPointMin;
                newPivotValues(0)   = pivotMin;
                newPointsAbs.col(0) = newPointMinAbs;
                exitFlags[0]        = true;
            }
        } else if (ExitFlagMax >= 0) {
            newPoints.col(0)    = newPointMax;
            newPivotValues(0)   = pivotMax;
            newPointsAbs.col(0) = newPointMaxAbs;
            exitFlags[0]        = true;
        }

        return {newPoints, newPivotValues, newPointsAbs, exitFlags};
    }

    void matricesToPolynomial(double c, const Eigen::VectorXd &g, const Eigen::MatrixXd &H, bool symmetrizeH = true) {
        const int dimension = static_cast<int>(g.size());
        if (H.rows() != dimension || H.cols() != dimension) {
            throw std::runtime_error("matricesToPolynomial: H has wrong shape");
        }
        Eigen::MatrixXd Hsym = H;
        if (symmetrizeH) { Hsym = 0.5 * (H + H.transpose()); }

        const int       n_terms = (dimension + 1) * (dimension + 2) / 2;
        Eigen::VectorXd coeffs(n_terms);
        coeffs(0)                    = c;
        coeffs.segment(1, dimension) = g;

        Eigen::Index idx = dimension + 1;
        for (int k = 0; k < dimension; ++k) {
            for (int m = 0; m <= k; ++m) { coeffs(idx++) = Hsym(k, m); }
        }
        coefficients = std::move(coeffs);
    }

    void toString() const { fmt::print("Coefs: {}\n", coefficients.transpose()); }

    double directional_derivative(const Eigen::VectorXd &x, const Eigen::VectorXd &u) const {
        const double uinf = u.lpNorm<Eigen::Infinity>();
        if (!(uinf > 0.0)) throw std::runtime_error("directional_derivative: zero direction");
        auto [c, g, H] = getTerms();
        return (g + H * x).dot(u / uinf);
    }
};
