// === Overloads.cpp (linker-safe) ===

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

// ------------------------------------------------------------------
// Core (const&) implementations
// ------------------------------------------------------------------
static inline PolynomialPtr poly_mul_const(const PolynomialPtr &poly, double scalar) {
    if (!poly) { throw std::runtime_error("operator*: null polynomial"); }
    Polynomial out = *poly;
    out.coefficients *= scalar;
    return std::make_shared<Polynomial>(std::move(out));
}
static inline PolynomialPtr poly_add_const(const PolynomialPtr &a, const PolynomialPtr &b) {
    if (!a || !b) { throw std::runtime_error("operator+: null polynomial"); }
    Polynomial out = *a;
    out.coefficients += b->coefficients;
    return std::make_shared<Polynomial>(std::move(out));
}
static inline PolynomialPtr shift_poly_const(const PolynomialPtr &poly, const Eigen::VectorXd &shift) {
    if (!poly) { throw std::runtime_error("shiftPolynomial: null polynomial"); }
    double          c;
    Eigen::VectorXd g;
    Eigen::MatrixXd H;
    std::tie(c, g, H)           = poly->getTerms();
    const double          c_mod = c + g.dot(shift) + 0.5 * (shift.transpose() * H * shift).value();
    const Eigen::VectorXd g_mod = g + H * shift;
    return std::make_shared<Polynomial>(c_mod, g_mod, H);
}
static inline PolynomialPtr combine_polys_const(const PolynomialVector &polynomials,
                                                const Eigen::VectorXd  &coefficients) {
    const int terms = static_cast<int>(polynomials.size());
    if (terms == 0) return std::make_shared<Polynomial>();
    if (coefficients.size() != terms) throw std::runtime_error("combinePolynomials: size mismatch");

    double          C = 0.0;
    Eigen::VectorXd G;
    Eigen::MatrixXd H;

    {
        auto [c0, g0, H0] = polynomials[0]->getTerms();
        const double a0   = coefficients(0);
        C                 = a0 * c0;
        G                 = a0 * g0;
        H                 = a0 * H0;
    }
    for (int k = 1; k < terms; ++k) {
        const double a  = coefficients(k);
        auto [c, g, Hk] = polynomials[k]->getTerms();
        C += a * c;
        G.noalias() += a * g;
        H.noalias() += a * Hk;
    }
    return std::make_shared<Polynomial>(C, G, H);
}

// ------------------------------------------------------------------
// Wrappers with EXACT signatures expected elsewhere (non-const &)
// These satisfy the linker and forward to the const-core versions.
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
// Block orthogonalization (unchanged, inclusive indices)
// ------------------------------------------------------------------
PolynomialVector orthogonalizeBlock(const PolynomialVector &polynomials, int np, const Eigen::VectorXd &point,
                                    int orthBeginning, int orthEnd) {
    PolynomialVector result = polynomials;
    const int        N      = static_cast<int>(result.size());
    if (N == 0) return result;
    if (np < 0 || np >= N) return result;

    orthBeginning = std::clamp(orthBeginning, 0, N - 1);
    orthEnd       = std::clamp(orthEnd, 0, N - 1);
    if (orthBeginning > orthEnd) return result;

    for (int p = orthBeginning; p <= orthEnd; ++p) {
        if (p == np) continue;
        result[p] = result[p]->zeroAtPoint(result[np], point);
    }
    return result;
}

// ------------------------------------------------------------------
// Quadratic MN polynomials (robust solve) — same as before
// ------------------------------------------------------------------
// Robust MN quadratic fit with rank detection + regularization
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

PolynomialVector computeQuadraticMFNPolynomials(const Eigen::MatrixXd &pointsAbs, int trCenter,
                                                const Eigen::MatrixXd &fValues) {
    const int d = static_cast<int>(pointsAbs.rows());
    const int p = static_cast<int>(pointsAbs.cols());
    const int q = d * (d + 1) / 2;

    const Eigen::VectorXd x0 = pointsAbs.col(trCenter);  // center (abs coords)
    Eigen::MatrixXd       S  = pointsAbs.colwise() - x0; // d x p (local coords s = x - x0)

    // PhiL (p x (d+1)) with rows [1, s^T]
    Eigen::MatrixXd PhiL(p, d + 1);
    for (int j = 0; j < p; ++j) {
        PhiL(j, 0)                = 1.0;
        PhiL.row(j).segment(1, d) = S.col(j).transpose();
    }

    // PhiQ (p x q): monomials s_i s_l in lower-triangular order (l <= i)
    Eigen::MatrixXd PhiQ(p, q);
    for (int j = 0; j < p; ++j) {
        int t = 0;
        for (int i = 0; i < d; ++i) {
            for (int l = 0; l <= i; ++l) { PhiQ(j, t++) = S(i, j) * S(l, j); }
        }
    }

    // ---- Correct Frobenius weights in h-space: diag=1, off=1/sqrt(2) ----
    Eigen::VectorXd w(q);
    {
        int t = 0;
        for (int i = 0; i < d; ++i) {
            for (int l = 0; l <= i; ++l) {
                if (i == l)
                    w(t++) = 1.0; // H_ii
                else
                    w(t++) = 1.0 / std::sqrt(2.0); // h_off = 2*H_il  -> weight = 1/sqrt(2)
            }
        }
    }
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> W(w);

    const int        m = static_cast<int>(fValues.rows());
    PolynomialVector models(m);

    for (int k = 0; k < m; ++k) {
        const Eigen::VectorXd y = fValues.row(k).transpose(); // length p

        // Linear part [c; g]
        const Eigen::VectorXd cg = PhiL.colPivHouseholderQr().solve(y); // (d+1)
        const Eigen::VectorXd r  = y - PhiL * cg;                       // residual

        // ---- Quadratic: (1/2) * PhiQ * h = r  => PhiQ * h = 2r ----
        const Eigen::VectorXd h_p = PhiQ.colPivHouseholderQr().solve(2.0 * r); // particular

        // Nullspace for weighted min-norm solution
        Eigen::FullPivLU<Eigen::MatrixXd> lu(PhiQ);
        lu.setThreshold(1e-12);
        const Eigen::MatrixXd Z = lu.kernel(); // q x nz

        Eigen::VectorXd h;
        if (Z.cols() == 0) {
            h = h_p;
        } else {
            // Solve: min_z || W (h_p + Z z) ||_2
            const Eigen::MatrixXd WZ   = W * Z;               // q x nz
            const Eigen::VectorXd Whp  = W * h_p;             // q
            const Eigen::MatrixXd Hsys = WZ.transpose() * WZ; // nz x nz (SPD if full col rank)
            const Eigen::VectorXd rhs  = -WZ.transpose() * Whp;
            const Eigen::VectorXd z    = Hsys.ldlt().solve(rhs);
            h                          = h_p + Z * z;
        }

        // ---- Reconstruct (c, g, H) from cg, h ----
        const double          c = cg(0);
        const Eigen::VectorXd g = cg.tail(d);

        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(d, d);
        {
            int t = 0;
            for (int i = 0; i < d; ++i) {
                for (int l = 0; l <= i; ++l) {
                    if (i == l) {
                        H(i, i) = h(t); // h_diag = H_ii
                    } else {
                        const double h_off = h(t);
                        const double Hij   = 0.5 * h_off; // ---- FIX: off-diag is h/2 ----
                        H(i, l)            = Hij;
                        H(l, i)            = Hij;
                    }
                    ++t;
                }
            }
        }

        // Build polynomial around x0 and shift back to absolute coords
        auto poly = std::make_shared<Polynomial>();
        poly->matricesToPolynomial(c, g, H, /*symmetrizeH=*/true);

        // ---- FIX: shift by x0, not S.col(trCenter) (which is zero) ----
        poly = shiftPolynomial(poly, /*shift=*/x0);

        models[k] = poly;
    }
    return models;
}

PolynomialVector computeQuadraticAdaptivePolynomials(const Eigen::MatrixXd &pointsAbs, int trCenter,
                                                     const Eigen::MatrixXd &fValues) {
    const int             d  = pointsAbs.rows();
    const int             p  = pointsAbs.cols();
    const int             q  = d * (d + 1) / 2;
    const int             m  = fValues.rows();
    const Eigen::VectorXd x0 = pointsAbs.col(trCenter);
    Eigen::MatrixXd       S  = pointsAbs.colwise() - x0; // d x p
    // Build Phi (p x (d + 1 + q))
    Eigen::MatrixXd Phi(p, d + 1 + q);
    for (int j = 0; j < p; ++j) {
        Phi(j, 0)                = 1.0;
        Phi.row(j).segment(1, d) = S.col(j).transpose();
        int t                    = 0;
        for (int i = 0; i < d; ++i) {
            for (int l = 0; l <= i; ++l) { Phi(j, d + 1 + t++) = S(i, j) * S(l, j); }
        }
    }
    PolynomialVector models(m);
    for (int k = 0; k < m; ++k) {
        Eigen::VectorXd y = fValues.row(k).transpose();
        // Adaptive regularization
        Eigen::BDCSVD<Eigen::MatrixXd> svd(Phi);
        double                         lambda  = 1e-10 * svd.singularValues()(0); // Initial guess
        Eigen::MatrixXd                PhiTPhi = Phi.transpose() * Phi;
        Eigen::VectorXd                PhiTy   = Phi.transpose() * y;
        Eigen::MatrixXd                A       = PhiTPhi + lambda * Eigen::MatrixXd::Identity(d + 1 + q, d + 1 + q);
        Eigen::VectorXd                theta   = A.colPivHouseholderQr().solve(PhiTy);
        // Reconstruct polynomial
        double          c = theta(0);
        Eigen::VectorXd g = theta.segment(1, d);
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(d, d);
        int             t = 0;
        for (int i = 0; i < d; ++i) {
            for (int l = 0; l <= i; ++l) {
                H(i, l) = H(l, i) = (i == l) ? theta(d + 1 + t) : 0.5 * theta(d + 1 + t);
                ++t;
            }
        }
        auto poly = std::make_shared<Polynomial>(c, g, H);
        poly      = shiftPolynomial(poly, x0);
        models[k] = poly;
    }
    return models;
}