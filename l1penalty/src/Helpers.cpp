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

// PolynomialVector nfpBasis(int dimension) {
//     const int poly_num    = (dimension + 1) * (dimension + 2) / 2;
//     const int linear_size = dimension + 1; // constant (0) + linear terms (1..dimension)

//     PolynomialVector basis(poly_num);

//     // Constant + linear as one-hot coefficient vectors
//     for (int k = 0; k < linear_size; ++k) {
//         basis[k]               = std::make_shared<Polynomial>();
//         Eigen::VectorXd coeffs = Eigen::VectorXd::Zero(poly_num);
//         coeffs(k)              = 1.0;
//         basis[k]->coefficients = std::move(coeffs);
//     }

//     // Quadratic block (c, g, H) representation
//     int                   m = 0, n = 0;
//     const double          c0 = 0.0;
//     const Eigen::VectorXd g0 = Eigen::VectorXd::Zero(dimension);

//     for (int poly_i = dimension + 1; poly_i < poly_num; ++poly_i) {
//         Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dimension, dimension);
//         if (m == n) {
//             // For 1/2 x^T H x, set H(mm)=2 to represent x_m^2
//             H(m, n) = 2.0;
//         } else {
//             // Cross term x_m x_n
//             H(m, n) = 1.0;
//             H(n, m) = 1.0;
//         }
//         basis[poly_i] = std::make_shared<Polynomial>(c0, g0, H);

//         if (n < dimension - 1) {
//             ++n;
//         } else {
//             ++m;
//             n = m;
//         }
//     }
//     return basis;
// }

PolynomialVector nfpBasis(int dimension, double radius = 1.0) {
    const int poly_num = (dimension + 1) * (dimension + 2) / 2;
    const int linear_size = dimension + 1;
    PolynomialVector basis(poly_num);
    for (int k = 0; k < linear_size; ++k) {
        basis[k] = std::make_shared<Polynomial>();
        Eigen::VectorXd coeffs = Eigen::VectorXd::Zero(poly_num);
        coeffs(k) = 1.0;
        basis[k]->coefficients = std::move(coeffs);
    }
    int m = 0, n = 0;
    const double c0 = 0.0;
    const Eigen::VectorXd g0 = Eigen::VectorXd::Zero(dimension);
    for (int poly_i = dimension + 1; poly_i < poly_num; ++poly_i) {
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dimension, dimension);
        double scale = (m == n) ? 2.0 / (radius * radius) : 1.0 / radius;
        if (m == n) {
            H(m, n) = scale;
        } else {
            H(m, n) = scale;
            H(n, m) = scale;
        }
        basis[poly_i] = std::make_shared<Polynomial>(c0, g0, H);
        if (n < dimension - 1) {
            ++n;
        } else {
            ++m;
            n = m;
        }
    }
    return basis;
}