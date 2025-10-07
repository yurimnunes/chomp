#pragma once
// dopt_stabilizer.hpp
// Gill–Saunders-style residual shifting for QP linearizations.
//   JE p = -(cE + rE),  JI p <= -(cI + rI)
// rE, rI are small, scaled shifts to stabilize multipliers / rank issues.

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <tuple>
#include <vector>

namespace dopt {

using dvec = Eigen::VectorXd;
using dmat = Eigen::MatrixXd;

enum class ScaleMode { None, Ruiz };

struct DOptConfig {
    // Scaling policy
    ScaleMode dopt_scaling = ScaleMode::Ruiz;

    // Shift gains (multiplied by magnitude proxy from Jacobian)
    double dopt_sigma_E = 1e-2;  // equality linear pull
    double dopt_sigma_I = 1e-2;  // inequality linear pull

    // Caps / thresholds
    double dopt_max_shift   = 1e2;     // absolute cap for |rE_i| and rI_i
    double dopt_active_tol  = 1e-8;    // cI >= -tol considered near-active
    double dopt_mu_target   = 1e-4;    // target complementarity μ
};

struct DOptMeta {
    double ce_scale = 1.0;   // median column-2-norm proxy (equalities)
    double ci_scale = 1.0;   // median column-2-norm proxy (inequalities)
    double sE       = 1.0;   // row-equilibration scale (equalities)
    double sI       = 1.0;   // row-equilibration scale (inequalities)
    double sigmaE   = 0.0;   // effective equality sigma used
    double sigmaI   = 0.0;   // effective inequality sigma used
};

class DOptStabilizer {
public:
    explicit DOptStabilizer(DOptConfig cfg = {}) : cfg_(cfg) {}

    // Inputs (pass empty matrices/vectors when absent):
    //  - JE, JI: Jacobians (mE×n, mI×n), possibly 0×n
    //  - cE, cI: residual vectors (length mE/mI), possibly length 0
    //  - lam   : inequality multipliers (length mI), possibly length 0
    //  - nu    : equality multipliers (length mE), currently unused
    //
    // Outputs:
    //  - rE (opt): equality shift (same size as cE)  or std::nullopt if mE=0
    //  - rI (opt): inequality shift (same size as cI) or std::nullopt if mI=0
    //  - meta     : diagnostics/scales used
    std::tuple<std::optional<dvec>, std::optional<dvec>, DOptMeta>
    compute_shifts(const dmat& JE,
                   const dmat& JI,
                   const dvec& cE,
                   const dvec& cI,
                   const dvec& lam,
                   const dvec& /*nu*/) const
    {
        DOptMeta meta{};
        std::optional<dvec> rE, rI;

        // --- Equalities ---
        if (cE.size() > 0) {
            double sE = 1.0;
            if (cfg_.dopt_scaling == ScaleMode::Ruiz && JE.size() > 0) {
                sE = row_equilibrate_scale_(JE);  // median row-2-norm
            }
            meta.sE = sE;

            const double ce_scale = (cfg_.dopt_scaling == ScaleMode::Ruiz && JE.size() > 0)
                                        ? colnorms_median_(JE / sE)
                                        : colnorms_median_(JE);
            meta.ce_scale = ce_scale;

            const double sigmaE = cfg_.dopt_sigma_E * std::max(1.0, ce_scale);
            meta.sigmaE = sigmaE;

            dvec r = -sigmaE * cE;
            clip_inplace_(r, -cfg_.dopt_max_shift, cfg_.dopt_max_shift);
            rE = std::move(r);
        }

        // --- Inequalities ---
        if (cI.size() > 0) {
            double sI = 1.0;
            if (cfg_.dopt_scaling == ScaleMode::Ruiz && JI.size() > 0) {
                sI = row_equilibrate_scale_(JI);
            }
            meta.sI = sI;

            const double ci_scale = (cfg_.dopt_scaling == ScaleMode::Ruiz && JI.size() > 0)
                                        ? colnorms_median_(JI / sI)
                                        : colnorms_median_(JI);
            meta.ci_scale = ci_scale;

            const double sigmaI = cfg_.dopt_sigma_I * std::max(1.0, ci_scale);
            meta.sigmaI = sigmaI;

            const Eigen::Index nI = cI.size();
            dvec r_base = dvec::Zero(nI);
            dvec r_comp = dvec::Zero(nI);

            // Active set: cI >= -active_tol
            for (Eigen::Index i = 0; i < nI; ++i) {
                if (cI(i) >= -cfg_.dopt_active_tol) {
                    r_base(i) = -sigmaI * cI(i);
                }
            }

            // Complementarity-shaped term uses λ (inequality multipliers)
            if (lam.size() == nI) {
                constexpr double tiny = 1e-12;
                for (Eigen::Index i = 0; i < nI; ++i) {
                    if (cI(i) >= -cfg_.dopt_active_tol) {
                        const double li = std::max(std::abs(lam(i)), tiny);
                        const double target = cfg_.dopt_mu_target / li;
                        r_comp(i) = std::max(0.0, target - cI(i));
                    }
                }
            }

            dvec r = r_base + r_comp;
            // Keep inequality shift conservative: rI ∈ [0, max]
            clip_inplace_(r, 0.0, cfg_.dopt_max_shift);
            rI = std::move(r);
        }

        return {rE, rI, meta};
    }

private:
    DOptConfig cfg_;

    // Median column 2-norm as a magnitude proxy; returns 1.0 if J empty.
    static double colnorms_median_(const dmat& J) {
        if (J.size() == 0) return 1.0;
        const auto m = J.rows();
        const auto n = J.cols();
        if (n == 0) return 1.0;

        std::vector<double> norms;
        norms.reserve(static_cast<std::size_t>(n));
        for (Eigen::Index j = 0; j < n; ++j) {
            double s = 0.0;
            for (Eigen::Index i = 0; i < m; ++i) {
                const double v = J(i, j);
                s += v * v;
            }
            norms.push_back(std::sqrt(std::max(s, 1e-16)));
        }
        std::nth_element(norms.begin(), norms.begin() + norms.size() / 2, norms.end());
        return norms[norms.size() / 2];
    }

    // Median row 2-norm scale (used for "Ruiz-like" row equilibration proxy).
    static double row_equilibrate_scale_(const dmat& J) {
        if (J.size() == 0) return 1.0;
        const auto m = J.rows();
        std::vector<double> norms;
        norms.reserve(static_cast<std::size_t>(m));
        for (Eigen::Index i = 0; i < m; ++i) {
            double s = 0.0;
            for (Eigen::Index j = 0; j < J.cols(); ++j) {
                const double v = J(i, j);
                s += v * v;
            }
            norms.push_back(std::sqrt(std::max(s, 1e-16)));
        }
        std::nth_element(norms.begin(), norms.begin() + norms.size() / 2, norms.end());
        const double med = norms[norms.size() / 2];
        return (std::isfinite(med) && med > 0.0) ? med : 1.0;
    }

    static void clip_inplace_(dvec& v, double lo, double hi) {
        for (Eigen::Index i = 0; i < v.size(); ++i) {
            if (v(i) < lo) v(i) = lo;
            else if (v(i) > hi) v(i) = hi;
        }
    }
};

} // namespace dopt
