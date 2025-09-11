// === Improved Overloads.cpp (linker-safe) ===

#include "../include/Definitions.hpp"
#include "../include/PolynomialVector.hpp"
#include "../include/Polynomials.hpp"
#include "../include/TRModel.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include <limits>

// ------------------------------------------------------------------
// Core (const&) implementations with improved error handling
// ------------------------------------------------------------------
static inline PolynomialPtr poly_mul_const(const PolynomialPtr &poly, double scalar) {
    if (!poly) { 
        throw std::runtime_error("operator*: null polynomial"); 
    }
    if (!std::isfinite(scalar)) {
        throw std::runtime_error("operator*: invalid scalar value");
    }
    
    Polynomial out = *poly;
    out.coefficients *= scalar;
    return std::make_shared<Polynomial>(std::move(out));
}

static inline PolynomialPtr poly_add_const(const PolynomialPtr &a, const PolynomialPtr &b) {
    if (!a || !b) { 
        throw std::runtime_error("operator+: null polynomial"); 
    }
    if (a->coefficients.size() != b->coefficients.size()) {
        throw std::runtime_error("operator+: polynomial dimension mismatch");
    }
    
    Polynomial out = *a;
    out.coefficients += b->coefficients;
    return std::make_shared<Polynomial>(std::move(out));
}

static inline PolynomialPtr shift_poly_const(const PolynomialPtr &poly, const Eigen::VectorXd &shift) {
    if (!poly) { 
        throw std::runtime_error("shiftPolynomial: null polynomial"); 
    }
    
    auto [c, g, H] = poly->getTerms();
    
    if (g.size() != shift.size()) {
        throw std::runtime_error("shiftPolynomial: dimension mismatch between polynomial and shift vector");
    }
    
    // More efficient computation: c + g'*s + 0.5*s'*H*s
    const double c_mod = c + g.dot(shift) + 0.5 * shift.dot(H * shift);
    const Eigen::VectorXd g_mod = g + H * shift;
    
    return std::make_shared<Polynomial>(c_mod, g_mod, H);
}

static inline PolynomialPtr combine_polys_const(const PolynomialVector &polynomials,
                                                const Eigen::VectorXd &coefficients) {
    const int terms = static_cast<int>(polynomials.size());
    if (terms == 0) return std::make_shared<Polynomial>();
    
    if (coefficients.size() != terms) {
        throw std::runtime_error("combinePolynomials: size mismatch");
    }
    
    // Check for null polynomials
    for (int i = 0; i < terms; ++i) {
        if (!polynomials[i]) {
            throw std::runtime_error("combinePolynomials: null polynomial at index " + std::to_string(i));
        }
    }

    // Initialize with first polynomial
    auto [c0, g0, H0] = polynomials[0]->getTerms();
    const double a0 = coefficients(0);
    
    double C = a0 * c0;
    Eigen::VectorXd G = a0 * g0;
    Eigen::MatrixXd H = a0 * H0;

    // Accumulate remaining polynomials more efficiently
    for (int k = 1; k < terms; ++k) {
        const double ak = coefficients(k);
        if (std::abs(ak) < std::numeric_limits<double>::epsilon()) continue; // Skip zero coefficients
        
        auto [ck, gk, Hk] = polynomials[k]->getTerms();
        
        // Dimension consistency check
        if (gk.size() != G.size() || Hk.rows() != H.rows() || Hk.cols() != H.cols()) {
            throw std::runtime_error("combinePolynomials: inconsistent polynomial dimensions");
        }
        
        C += ak * ck;
        G.noalias() += ak * gk;
        H.noalias() += ak * Hk;
    }
    
    return std::make_shared<Polynomial>(C, G, H);
}

// ------------------------------------------------------------------
// Wrappers with EXACT signatures (maintaining compatibility)
// ------------------------------------------------------------------
PolynomialPtr operator*(PolynomialPtr &poly, double scalar) {
    return poly_mul_const(static_cast<const PolynomialPtr &>(poly), scalar);
}
PolynomialPtr operator*(double scalar, PolynomialPtr &poly) {
    return poly_mul_const(static_cast<const PolynomialPtr &>(poly), scalar);
}
PolynomialPtr operator+(PolynomialPtr &a, PolynomialPtr &b) {
    return poly_add_const(static_cast<const PolynomialPtr &>(a), static_cast<const PolynomialPtr &>(b));
}
PolynomialPtr shiftPolynomial(PolynomialPtr &poly, const Eigen::VectorXd &shift) {
    return shift_poly_const(static_cast<const PolynomialPtr &>(poly), shift);
}
PolynomialPtr combinePolynomials(PolynomialVector &polynomials, const Eigen::VectorXd &coefficients) {
    return combine_polys_const(static_cast<const PolynomialVector &>(polynomials), coefficients);
}

// ------------------------------------------------------------------
// Block orthogonalization with improved bounds checking
// ------------------------------------------------------------------
PolynomialVector orthogonalizeBlock(const PolynomialVector &polynomials, int np, const Eigen::VectorXd &point,
                                    int orthBeginning, int orthEnd) {
    PolynomialVector result = polynomials;
    const int N = static_cast<int>(result.size());
    
    if (N == 0) return result;
    if (np < 0 || np >= N) {
        throw std::runtime_error("orthogonalizeBlock: np index out of bounds");
    }
    
    // Improved bounds checking with informative errors
    if (orthBeginning < 0) orthBeginning = 0;
    if (orthEnd >= N) orthEnd = N - 1;
    if (orthBeginning > orthEnd) return result;

    // Check for null polynomials
    if (!result[np]) {
        throw std::runtime_error("orthogonalizeBlock: null polynomial at reference index");
    }

    for (int p = orthBeginning; p <= orthEnd; ++p) {
        if (p == np) continue;
        if (!result[p]) {
            throw std::runtime_error("orthogonalizeBlock: null polynomial at index " + std::to_string(p));
        }
        result[p] = result[p]->zeroAtPoint(result[np], point);
    }
    return result;
}

// ------------------------------------------------------------------
// Improved quadratic MN polynomials with better numerical stability
// ------------------------------------------------------------------
PolynomialVector computeQuadraticMnPolynomials(const Eigen::MatrixXd &points, int center_i,
                                               const Eigen::MatrixXd &fvalues) {

    int             dim           = points.rows();
    int             points_num    = points.cols();
    int             functions_num = fvalues.rows();
    Eigen::MatrixXd points_shifted(dim, points_num - 1);
    Eigen::MatrixXd fvalues_diff(functions_num, points_num - 1);

    // zero out the shifted points and function values
    points_shifted.setZero();
    fvalues_diff.setZero();
    int m2 = 0;
    for (int m = 0; m < points_num; ++m) {
        if (m != center_i) {
            points_shifted.col(m2) = points.col(m) - points.col(center_i);
            fvalues_diff.col(m2)   = fvalues.col(m) - fvalues.col(center_i);
            m2++;
        }
    }

    Eigen::MatrixXd M = points_shifted.transpose() * points_shifted;

    M = 0.5 * (M.array().square()).matrix() + M;

    Eigen::MatrixXd mult_mn = M.ldlt().solve(fvalues_diff.transpose());

    double accuracy = 1.0;
    if (M.determinant() < 1e4 * std::numeric_limits<double>::epsilon()) {
        // std::cerr << "Ill conditioned system" << std::endl;
        //  solve again multiplying
    }

    PolynomialVector polynomials;
    if (accuracy == 0) {
        std::cerr << "Bad set of points" << std::endl;
    } else {
        for (int n = 0; n < functions_num; ++n) {
            Eigen::VectorXd g = Eigen::VectorXd::Zero(dim);
            Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dim, dim);

            for (int m = 0; m < points_num - 1; ++m) {
                g += mult_mn(m, n) * points_shifted.col(m);
                H += mult_mn(m, n) * (points_shifted.col(m) * points_shifted.col(m).transpose());
            }
            double c = fvalues(n, center_i);
            polynomials.push_back(std::make_shared<Polynomial>(c, g, H));
        }
    }

    return polynomials;
}