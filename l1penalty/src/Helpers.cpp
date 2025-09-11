#include "../include/Definitions.hpp"
#include "../include/PolynomialVector.hpp"

#include <Eigen/Dense>
#include <algorithm> // <-- needed for std::sort
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "../include/Polynomials.hpp"
#include "../include/TRModel.hpp"
#include <limits> // for std::numeric_limits

// NOTE: forward-decl of PolynomialVector not needed (already included)

Eigen::VectorXd active2eigen(const std::vector<bool> &is_eactive) {
    // Build a plain double buffer then copy into an owning Eigen vector.
    const Eigen::Index n = static_cast<Eigen::Index>(is_eactive.size());
    Eigen::VectorXd    out(n);
    for (Eigen::Index i = 0; i < n; ++i) { out(i) = static_cast<double>(is_eactive[static_cast<size_t>(i)]); }
    return out;
}

void swapElements(Eigen::MatrixXd &matrix, int col1, int col2) {
    if (col1 < 0 || col1 >= matrix.cols() || col2 < 0 || col2 >= matrix.cols()) {
        throw std::out_of_range("Column index out of range");
    }
    matrix.col(col1).swap(matrix.col(col2));
}

Eigen::VectorXd shiftPoint(const Eigen::VectorXd &x, const Eigen::VectorXd &shift_center) { return x - shift_center; }

Eigen::VectorXd unshiftPoint(const Eigen::VectorXd &x, const Eigen::VectorXd &shift_center, const Eigen::VectorXd &bl,
                             const Eigen::VectorXd &bu) {
    return (x + shift_center).cwiseMin(bu).cwiseMax(bl);
}

// Keep your const-ref definition
std::pair<Eigen::VectorXd, bool> evaluateNewFValues(const Funcao &funcs, const Eigen::VectorXd &point) {
    const int       functions_num = funcs.size();
    Eigen::VectorXd fvalues       = Eigen::VectorXd::Zero(functions_num);
    bool            succeeded     = true;

    for (int nf = 0; nf < functions_num; ++nf) {
        try {
            if (nf == 0) {
                fvalues(nf) = funcs.evaluateObjective(point);
            } else {
                fvalues(nf) = funcs.con[nf - 1](point);
            }
        } catch (const std::exception &) { fvalues(nf) = std::numeric_limits<double>::quiet_NaN(); }
        if (!std::isfinite(fvalues(nf))) {
            succeeded = false;
            break;
        }
    }
    return {fvalues, succeeded};
}

// Overload to satisfy callers compiled against non-const-ref
std::pair<Eigen::VectorXd, bool> evaluateNewFValues(const Funcao &funcs, Eigen::VectorXd &point) {
    return evaluateNewFValues(funcs, static_cast<const Eigen::VectorXd &>(point));
}

Eigen::VectorXd projectToBounds(const Eigen::VectorXd &x, const Eigen::VectorXd &bl, const Eigen::VectorXd &bu) {
    return x.cwiseMax(bl).cwiseMin(bu);
}

std::vector<int> argsort(const Eigen::VectorXd &v) {
    std::vector<int> idx(static_cast<size_t>(v.size()));
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&v](int i1, int i2) { return v(i1) < v(i2); });
    return idx;
}
// Optimized basis generation with compile-time constants and efficient memory allocation

PolynomialVector nfpBasis(int dimension, double radius) {
    const int dim = dimension;
    constexpr auto poly_count = [](int d) constexpr { return (d + 1) * (d + 2) / 2; };
    const int poly_num = poly_count(dim);
    const int linear_size = dim + 1;
    
    // Pre-allocate with exact capacity to avoid reallocations
    PolynomialVector basis;
    basis.reserve(poly_num);
    
    // Cache common values
    const double inv_radius = 1.0 / radius;
    const double inv_radius_sq = inv_radius * inv_radius;
    const Eigen::VectorXd g0 = Eigen::VectorXd::Zero(dim);
    
    // Linear terms: use move semantics and direct construction
    for (int k = 0; k < linear_size; ++k) {
        basis.push_back(std::make_shared<Polynomial>());
        Eigen::VectorXd coeffs = Eigen::VectorXd::Zero(poly_num);
        coeffs(k) = 1.0;
        basis[k]->coefficients = std::move(coeffs);
    }
    
    // Quadratic terms: vectorized Hessian construction
    const int quad_terms = poly_num - linear_size;
    std::vector<std::pair<int, int>> indices;
    indices.reserve(quad_terms);
    
    // Generate index pairs efficiently
    for (int m = 0; m < dim; ++m) {
        for (int n = m; n < dim; ++n) {
            indices.emplace_back(m, n);
        }
    }
    
    // Batch create quadratic polynomials
    for (int i = 0; i < quad_terms; ++i) {
        const auto& [m, n] = indices[i];
        
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dim, dim);
        const double scale = (m == n) ? 2.0 * inv_radius_sq : inv_radius;
        
        H(m, n) = scale;
        if (m != n) H(n, m) = scale;  // Symmetric fill
        
        basis.push_back(std::make_shared<Polynomial>(0.0, g0, std::move(H)));
    }
    
    return basis;
}
