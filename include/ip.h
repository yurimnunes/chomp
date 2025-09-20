// ip_cpp.cpp — optimized & cleaned version
// Behavior: identical functionality, but faster & cleaner implementation

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SparseCore>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "definitions.h"
#include "funnel.h"
#include "kkt_core.h"
#include "linesearch.h"
#include "regularizer.h"

namespace py = pybind11;

struct KKTResult {
    dvec dx; // primal search direction
    dvec dy; // equality multipliers (empty if no JE)
    std::shared_ptr<kkt::KKTReusable> reusable; // factorization handle
};

namespace consts {
constexpr double EPS_DIV = 1e-16;
constexpr double EPS_POS = 1e-12;
constexpr double INF = std::numeric_limits<double>::infinity();
} // namespace consts

// ---------- Utility functions ----------
template <class T> [[nodiscard]] constexpr T clamp(T v, T lo, T hi) noexcept {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

template <class T> [[nodiscard]] constexpr T clamp_min(T v, T lo) noexcept {
    return (v < lo) ? lo : v;
}

template <class T> [[nodiscard]] constexpr T clamp_max(T v, T hi) noexcept {
    return (v > hi) ? hi : v;
}

[[nodiscard]] inline double safe_div(double a, double b,
                                     double eps = consts::EPS_DIV) noexcept {
    return a / clamp_min(std::abs(b), eps);
}

[[nodiscard]] inline double sdiv(double a, double b,
                                 double eps = consts::EPS_DIV) noexcept {
    return a / ((b > eps) ? b : eps);
}

[[nodiscard]] inline double spdiv(double a, double b,
                                  double eps = 1e-12) noexcept {
    return a / clamp_min(b, eps);
}

[[nodiscard]] inline double safe_inf_norm(const dvec &v) noexcept {
    if (v.size() == 0)
        return 0.0;
    return v.cwiseAbs().maxCoeff();
}

// ---------- Python attribute helpers ----------
namespace pyu {
[[nodiscard]] inline bool has_attr(const py::object &o,
                                   const char *name) noexcept {
    return o && PyObject_HasAttrString(o.ptr(), name);
}

template <class T>
[[nodiscard]] T getattr_or(const py::object &o, const char *name,
                           const T &fallback) {
    if (!o || !has_attr(o, name))
        return fallback;
    try {
        return o.attr(name).cast<T>();
    } catch (...) {
        return fallback;
    }
}
} // namespace pyu

// ---------- Python ↔ Eigen conversions ----------
namespace pyconv {
[[nodiscard]] inline dvec to_vec(const py::object &arr) {
    if (arr.is_none())
        return dvec();

    py::array_t<double, py::array::c_style | py::array::forcecast> a(arr);

    if (a.ndim() == 0 || a.size() == 0)
        return dvec();
    if (a.ndim() != 1) {
        throw std::runtime_error("to_vec: Expected 1-D array, got " +
                                 std::to_string(a.ndim()) + "-D array");
    }

    auto r = a.unchecked<1>();
    dvec v(r.shape(0));
    for (ssize_t i = 0; i < r.shape(0); ++i) {
        v[i] = r(i);
    }
    return v;
}

[[nodiscard]] inline spmat to_sparse(const py::object &obj) {
    if (obj.is_none())
        return spmat();

    // Check for scipy sparse matrix
    if (PyObject_HasAttrString(obj.ptr(), "tocoo")) {
        py::object coo = obj.attr("tocoo")();
        auto shape =
            coo.attr("shape").cast<std::pair<py::ssize_t, py::ssize_t>>();

        auto row =
            coo.attr("row").cast<py::array_t<long long, py::array::c_style>>();
        auto col =
            coo.attr("col").cast<py::array_t<long long, py::array::c_style>>();
        auto dat =
            coo.attr("data").cast<py::array_t<double, py::array::c_style>>();

        auto R = row.unchecked<1>();
        auto C = col.unchecked<1>();
        auto X = dat.unchecked<1>();

        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(X.shape(0));
        for (ssize_t k = 0; k < X.shape(0); ++k) {
            triplets.emplace_back(static_cast<int>(R(k)),
                                  static_cast<int>(C(k)), X(k));
        }

        spmat A(shape.first, shape.second);
        A.setFromTriplets(triplets.begin(), triplets.end());
        A.makeCompressed();
        return A;
    }

    // Dense numpy array fallback
    py::array_t<double, py::array::c_style | py::array::forcecast> a(obj);
    if (a.ndim() != 2)
        throw std::runtime_error("Expected 2D array");

    const int rows = static_cast<int>(a.shape(0));
    const int cols = static_cast<int>(a.shape(1));
    dmat M(rows, cols);
    auto r = a.unchecked<2>();
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            M(i, j) = r(i, j);
        }
    }
    return M.sparseView();
}
} // namespace pyconv

// ---------- Core data structures ----------
struct IPState {
    int mI = 0, mE = 0;
    dvec s, lam, nu, zL, zU;
    double mu = 1e-2;
    double tau_shift = 0.0;
    bool initialized = false;
};

struct Bounds {
    dvec lb, ub, sL, sU;
    std::vector<uint8_t> hasL, hasU; // Keep uint8_t for compatibility
};

struct Sigmas {
    dvec Sigma_x, Sigma_s;
};

// ---------- Helper functions ----------
namespace detail {

[[nodiscard]] inline Bounds get_bounds(const py::object &model, const dvec &x) {
    const int n = static_cast<int>(x.size());
    Bounds B;

    // Handle lower bounds
    if (pyu::has_attr(model, "lb") && !model.attr("lb").is_none()) {
        dvec lb_vec = pyconv::to_vec(model.attr("lb"));
        B.lb = (lb_vec.size() == n) ? std::move(lb_vec)
                                    : dvec::Constant(n, -consts::INF);
    } else {
        B.lb = dvec::Constant(n, -consts::INF);
    }

    // Handle upper bounds
    if (pyu::has_attr(model, "ub") && !model.attr("ub").is_none()) {
        dvec ub_vec = pyconv::to_vec(model.attr("ub"));
        B.ub = (ub_vec.size() == n) ? std::move(ub_vec)
                                    : dvec::Constant(n, +consts::INF);
    } else {
        B.ub = dvec::Constant(n, +consts::INF);
    }

    // Initialize bound indicators and slack variables
    B.hasL.assign(n, 0);
    B.hasU.assign(n, 0);
    B.sL.resize(n);
    B.sU.resize(n);

    for (int i = 0; i < n; ++i) {
        const bool hL = std::isfinite(B.lb[i]);
        const bool hU = std::isfinite(B.ub[i]);
        B.hasL[i] = static_cast<uint8_t>(hL);
        B.hasU[i] = static_cast<uint8_t>(hU);
        B.sL[i] = hL ? clamp_min(x[i] - B.lb[i], consts::EPS_POS) : 1.0;
        B.sU[i] = hU ? clamp_min(B.ub[i] - x[i], consts::EPS_POS) : 1.0;
    }

    return B;
}

[[nodiscard]] inline Sigmas
build_sigmas(const dvec &zL, const dvec &zU, const Bounds &B, const dvec &lmb,
             const dvec &s, const dvec &cI, double tau_shift,
             double bound_shift, bool use_shifted, double eps_abs, double cap) {
    const int n = static_cast<int>(zL.size());
    const int mI = static_cast<int>(s.size());

    Sigmas S;
    S.Sigma_x = dvec::Zero(n);

    for (int i = 0; i < n; ++i) {
        double v = 0.0;
        if (B.hasL[i]) {
            double d = B.sL[i] + (use_shifted ? bound_shift : 0.0);
            v += zL[i] / clamp_min(d, eps_abs);
        }
        if (B.hasU[i]) {
            double d = B.sU[i] + (use_shifted ? bound_shift : 0.0);
            v += zU[i] / clamp_min(d, eps_abs);
        }
        S.Sigma_x[i] = clamp(v, 0.0, cap);
    }

    S.Sigma_s = dvec::Zero(mI);
    if (mI > 0) {
        if (use_shifted) {
            for (int i = 0; i < mI; ++i) {
                double d = s[i] + tau_shift;
                S.Sigma_s[i] = clamp(lmb[i] / clamp_min(d, eps_abs), 0.0, cap);
            }
        } else {
            for (int i = 0; i < mI; ++i) {
                double sf = clamp(std::abs(cI[i]), 1e-8, 1.0);
                double sv = clamp_min(s[i], sf);
                S.Sigma_s[i] = clamp(lmb[i] / clamp_min(sv, eps_abs), 0.0, cap);
            }
        }
    }
    return S;
}

[[nodiscard]] inline std::pair<dvec, dvec>
dz_bounds_from_dx(const dvec &dx, const dvec &zL, const dvec &zU,
                  const Bounds &B, double bound_shift, bool use_shifted,
                  double mu, bool use_mu) {

    const int n = dx.size();
    dvec dzL = dvec::Zero(n);
    dvec dzU = dvec::Zero(n);

    for (int i = 0; i < n; ++i) {
        if (B.hasL[i]) {
            double d = clamp_min(B.sL[i] + (use_shifted ? bound_shift : 0.0),
                                 consts::EPS_DIV);
            dzL[i] = use_mu ? (mu - d * zL[i] - zL[i] * dx[i]) / d
                            : -(zL[i] * dx[i]) / d;
        }
        if (B.hasU[i]) {
            double d = clamp_min(B.sU[i] + (use_shifted ? bound_shift : 0.0),
                                 consts::EPS_DIV);
            dzU[i] = use_mu ? (mu - d * zU[i] + zU[i] * dx[i]) / d
                            : (zU[i] * dx[i]) / d;
        }
    }

    return {dzL, dzU};
}

[[nodiscard]] inline double complementarity(const dvec &s, const dvec &lmb,
                                            double mu, double tau_shift,
                                            bool use_shifted) {
    const int m = static_cast<int>(s.size());
    if (!m)
        return 0.0;

    double acc = 0.0;
    if (use_shifted) {
        for (int i = 0; i < m; ++i) {
            acc += std::abs((s[i] + tau_shift) * lmb[i] - mu);
        }
    } else {
        for (int i = 0; i < m; ++i) {
            acc += std::abs(s[i] * lmb[i] - mu);
        }
    }
    return acc / clamp_min(m, 1);
}

[[nodiscard]] inline double alpha_ftb(const dvec &x, const dvec &dx,
                                      const dvec &s, const dvec &ds,
                                      const dvec &lmb, const dvec &dlam,
                                      const Bounds &B, double tau_pri,
                                      double tau_dual) {
    double a_pri = 1.0, a_dual = 1.0;

    // Inequalities
    for (int i = 0; i < s.size(); ++i) {
        if (ds[i] < 0.0) {
            a_pri = std::min(a_pri, -s[i] / std::min(ds[i], -consts::EPS_DIV));
        }
    }

    for (int i = 0; i < lmb.size(); ++i) {
        if (dlam[i] < 0.0) {
            a_dual =
                std::min(a_dual, -lmb[i] / std::min(dlam[i], -consts::EPS_DIV));
        }
    }

    // Box constraints
    for (int i = 0; i < x.size(); ++i) {
        if (B.hasL[i] && dx[i] < 0.0) {
            a_pri = std::min(a_pri, -(x[i] - B.lb[i]) /
                                        std::min(dx[i], -consts::EPS_DIV));
        }
        if (B.hasU[i] && dx[i] > 0.0) {
            a_pri = std::min(a_pri, (B.ub[i] - x[i]) /
                                        clamp_min(dx[i], consts::EPS_DIV));
        }
    }

    double a = std::min(tau_pri * a_pri, tau_dual * a_dual);
    return clamp(a, 0.0, 1.0);
}

inline void cap_bound_duals_sigma_box(dvec &zL, dvec &zU, const Bounds &B,
                                      bool use_shifted, double bound_shift,
                                      double mu, double ksig = 1e10) {
    for (int i = 0; i < zL.size(); ++i) {
        if (B.hasL[i]) {
            double sLc = clamp_min(B.sL[i] + (use_shifted ? bound_shift : 0.0),
                                   consts::EPS_DIV);
            double lo = mu / (ksig * sLc);
            double hi = (ksig * mu) / sLc;
            zL[i] = clamp(zL[i], lo, hi);
        }
        if (B.hasU[i]) {
            double sUc = clamp_min(B.sU[i] + (use_shifted ? bound_shift : 0.0),
                                   consts::EPS_DIV);
            double lo = mu / (ksig * sUc);
            double hi = (ksig * mu) / sUc;
            zU[i] = clamp(zU[i], lo, hi);
        }
    }
}

} // namespace detail

// ---------- Main Interior Point Stepper ----------
class InteriorPointStepper {
public:
    IPState st{};
    std::shared_ptr<LineSearcher> ls_;
    std::shared_ptr<regx::Regularizer> regularizer_ =
        std::make_shared<regx::Regularizer>();

    InteriorPointStepper(py::object cfg, py::object hess)
        : cfg_(std::move(cfg)), hess_(std::move(hess)) {
        load_defaults_();
        load_gondzio_defaults_();
        std::shared_ptr<Funnel> funnel = std::shared_ptr<Funnel>();
        ls_ = std::make_shared<LineSearcher>(cfg_, py::none(), funnel);
    }

    std::tuple<dvec, dvec, dvec, py::dict>
    step(py::object model, const dvec &x, const dvec &lam, const dvec &nu,
         int it, std::optional<IPState> ip_state_opt = std::nullopt) {

        if (!st.initialized) {
            st = state_from_model_(model, x);
        }

        const int n = static_cast<int>(x.size());
        const int mI = st.mI;
        const int mE = st.mE;

        dvec s = st.s;
        dvec lmb = st.lam;
        dvec nuv = st.nu;
        dvec zL = st.zL;
        dvec zU = st.zU;
        double mu = st.mu;

        const bool use_shifted =
            pyu::getattr_or<bool>(cfg_, "ip_use_shifted_barrier", false);
        double tau_shift = use_shifted ? st.tau_shift : 0.0;
        const bool shift_adapt =
            pyu::getattr_or<bool>(cfg_, "ip_shift_adaptive", true);

        // Evaluate model
        auto comps = py::make_tuple("f", "g", "cI", "JI", "cE", "JE");
        py::dict d0 = model.attr("eval_all")(x, comps);
        const double f = d0["f"].cast<double>();
        dvec g = pyconv::to_vec(d0["g"]);
        dvec cI = (mI > 0 && !d0["cI"].is_none()) ? pyconv::to_vec(d0["cI"])
                                                  : dvec::Zero(mI);
        dvec cE = (mE > 0 && !d0["cE"].is_none()) ? pyconv::to_vec(d0["cE"])
                                                  : dvec::Zero(mE);

        spmat JI, JE;
        if (mI > 0 && !d0["JI"].is_none()) {
            JI = pyconv::to_sparse(d0["JI"]);
            if (JI.rows() != mI || JI.cols() != n) {
                throw std::runtime_error("JI dimension mismatch");
            }
        }
        if (mE > 0 && !d0["JE"].is_none()) {
            JE = pyconv::to_sparse(d0["JE"]);
            if (JE.rows() != mE || JE.cols() != n) {
                throw std::runtime_error("JE dimension mismatch");
            }
        }

        double theta = model.attr("constraint_violation")(x).cast<double>();

        // Bounds with adaptive shifts
        Bounds B = detail::get_bounds(model, x);
        if (use_shifted && shift_adapt) {
            tau_shift = adaptive_shift_slack_(s, cI, it);
            st.tau_shift = tau_shift;
        }
        double bound_shift =
            use_shifted ? pyu::getattr_or<double>(cfg_, "ip_shift_bounds", 0.0)
                        : 0.0;
        if (use_shifted && shift_adapt) {
            bound_shift = adaptive_shift_bounds_(x, B, it);
        }

        // Quick convergence check
        const double tol = pyu::getattr_or<double>(cfg_, "tol", 1e-8);
        const double err_0 =
            compute_error_(model, x, lmb, nuv, zL, zU, 0.0, s, mI);
        if (err_0 <= tol) {
            py::dict info;
            info["mode"] = "ip";
            info["step_norm"] = 0.0;
            info["accepted"] = true;
            info["converged"] = true;
            info["f"] = f;
            info["theta"] = theta;
            info["stat"] = safe_inf_norm(g);
            info["ineq"] =
                (mI > 0) ? safe_inf_norm((cI.array().max(0.0)).matrix()) : 0.0;
            info["eq"] = (mE > 0) ? safe_inf_norm(cE) : 0.0;
            info["comp"] = 0.0;
            info["ls_iters"] = 0;
            info["alpha"] = 0.0;
            info["rho"] = 0.0;
            info["tr_radius"] = tr_radius_();
            info["mu"] = mu;
            return {x, lmb, nuv, info};
        }

        // Build diagonal regularization
        const double eps_abs =
            pyu::getattr_or<double>(cfg_, "sigma_eps_abs", 1e-8);
        const double cap = pyu::getattr_or<double>(cfg_, "sigma_cap", 1e8);
        Sigmas Sg =
            detail::build_sigmas(zL, zU, B, lmb, s, cI, tau_shift, bound_shift,
                                 use_shifted, eps_abs, cap);

        // Get and regularize Hessian
        py::object H_obj = pyu::getattr_or<bool>(cfg_, "ip_exact_hessian", true)
                               ? model.attr("lagrangian_hessian")(
                                     x, py::cast(lmb), py::cast(nuv))
                               : hess_.attr("get_hessian")(
                                     model, x, py::cast(lmb), py::cast(nuv));

        spmat H_obj_sparse = pyconv::to_sparse(H_obj);
        auto [H, reg_info] = regularizer_->regularize(H_obj_sparse, it);

        // Assemble KKT matrix: W = H + diag(Sigma_x) + JI^T diag(Sigma_s) JI
        spmat W = H;
        for (int i = 0; i < std::min<int>(W.rows(), Sg.Sigma_x.size()); ++i) {
            W.coeffRef(i, i) += Sg.Sigma_x[i];
        }

        if (mI > 0 && JI.size() && Sg.Sigma_s.size()) {
            spmat JIw = Sg.Sigma_s.asDiagonal() * JI;
            W += JI.transpose() * JIw;
        }

        // Build residuals
        dvec r_d = g;
        if (mI > 0 && JI.size())
            r_d.noalias() += JI.transpose() * lmb;
        if (mE > 0 && JE.size())
            r_d.noalias() += JE.transpose() * nuv;
        r_d -= zL;
        r_d += zU;

        dvec r_pE = (mE > 0) ? cE : dvec();
        dvec r_pI = (mI > 0) ? (cI + s) : dvec();

        // Mehrotra affine predictor
        // const auto [alpha_aff, mu_aff, sigma] = mehrotra_affine_predictor_(
        //     W, r_d, (mE > 0) ? std::optional<spmat>(JE) : std::nullopt,
        //     (mE > 0) ? std::optional<dvec>(r_pE) : std::nullopt,
        //     (mI > 0) ? std::optional<spmat>(JI) : std::nullopt,
        //     (mI > 0) ? std::optional<dvec>(r_pI) : std::nullopt, s, lmb, zL,
        //     zU, B, use_shifted, tau_shift, bound_shift, mu, theta);

        const auto [alpha_aff, mu_aff, sigma, gondzio_step] =
            mehrotra_with_gondzio_corrections_(
                W, r_d, (mE > 0) ? std::optional<spmat>(JE) : std::nullopt,
                (mE > 0) ? std::optional<dvec>(r_pE) : std::nullopt,
                (mI > 0) ? std::optional<spmat>(JI) : std::nullopt,
                (mI > 0) ? std::optional<dvec>(r_pI) : std::nullopt, s, lmb, zL,
                zU, B, use_shifted, tau_shift, bound_shift, mu, theta, Sg);

        // Then use gondzio_step instead of solving the corrector system again
        dvec dx = std::move(gondzio_step.dx);
        dvec dnu = (mE > 0 && gondzio_step.dnu.size() == mE)
                       ? std::move(gondzio_step.dnu)
                       : dvec::Zero(mE);
        dvec ds = std::move(gondzio_step.ds);
        dvec dlam = std::move(gondzio_step.dlam);
        dvec dzL = std::move(gondzio_step.dzL);
        dvec dzU = std::move(gondzio_step.dzU);

        // Update barrier parameter
        const double comp =
            detail::complementarity(s, lmb, mu, tau_shift, use_shifted);
        if (comp * clamp_min(mI, 1) > 10.0 * mu) {
            mu = std::min(comp * clamp_min(mI, 1), 10.0);
        }
        mu = clamp_min(sigma * mu_aff,
                       pyu::getattr_or<double>(cfg_, "ip_mu_min", 1e-12));

        // // Build corrector RHS
        // dvec rhs_x = -r_d;

        // if (mI > 0 && JI.size() && Sg.Sigma_s.size()) {
        //     dvec rc_s(mI);
        //     for (int i = 0; i < mI; ++i) {
        //         const double ds = use_shifted ? (s[i] + tau_shift) : s[i];
        //         rc_s[i] = mu - ds * lmb[i];
        //     }
        //     dvec temp(mI);
        //     for (int i = 0; i < mI; ++i) {
        //         const double lam_safe =
        //             (std::abs(lmb[i]) < consts::EPS_POS)
        //                 ? ((lmb[i] >= 0) ? consts::EPS_POS :
        //                 -consts::EPS_POS) : lmb[i];
        //         temp[i] = rc_s[i] / lam_safe;
        //     }
        //     rhs_x.noalias() +=
        //         JI.transpose() * (Sg.Sigma_s.asDiagonal() * temp);
        // }

        // // Add bound terms to RHS
        // for (int i = 0; i < n; ++i) {
        //     if (B.hasL[i]) {
        //         double denom =
        //             clamp_min(use_shifted ? (B.sL[i] + bound_shift) :
        //             B.sL[i],
        //                       consts::EPS_POS);
        //         rhs_x[i] += (mu - denom * zL[i]) / denom;
        //     }
        // }
        // for (int i = 0; i < n; ++i) {
        //     if (B.hasU[i]) {
        //         double denom =
        //             clamp_min(use_shifted ? (B.sU[i] + bound_shift) :
        //             B.sU[i],
        //                       consts::EPS_POS);
        //         rhs_x[i] -= (mu - denom * zU[i]) / denom;
        //     }
        // }

        // // Solve KKT system
        // auto res = solve_KKT_(
        //     W, rhs_x, (mE > 0) ? std::optional<spmat>(JE) : std::nullopt,
        //     (mE > 0) ? std::optional<dvec>(cE) : std::nullopt,
        //     pyu::getattr_or<std::string>(cfg_, "ip_kkt_method", "hykkt"));

        // dvec dx = std::move(res.dx);
        // dvec dnu = (mE > 0 && res.dy.size() == mE) ? std::move(res.dy)
        //                                            : dvec::Zero(mE);

        // // Recover ds, dλ, dz
        // dvec ds, dlam;
        // if (mI > 0) {
        //     ds = -(r_pI + JI * dx);
        //     dlam = dvec(mI);
        //     for (int i = 0; i < mI; ++i) {
        //         double d = use_shifted ? (s[i] + tau_shift) : s[i];
        //         dlam[i] = sdiv(mu - d * lmb[i] - lmb[i] * ds[i], d);
        //     }
        // }
        // auto [dzL, dzU] = detail::dz_bounds_from_dx(dx, zL, zU, B,
        // bound_shift,
        //                                             use_shifted, mu, true);

        // Trust region clipping
        const double dx_cap = pyu::getattr_or<double>(cfg_, "ip_dx_max", 1e3);
        const double nx = dx.norm();
        if (nx > dx_cap && nx > 0.0) {
            const double sc = dx_cap / nx;
            dx *= sc;
            dzL *= sc;
            dzU *= sc;
            if (mI > 0) {
                ds *= sc;
                dlam *= sc;
            }
        }

        // Fraction-to-boundary and line search
        const double tau_pri = pyu::getattr_or<double>(
            cfg_, "ip_tau_pri", pyu::getattr_or<double>(cfg_, "ip_tau", 0.995));
        const double tau_dual = pyu::getattr_or<double>(
            cfg_, "ip_tau_dual",
            pyu::getattr_or<double>(cfg_, "ip_tau", 0.995));

        const double a_ftb =
            detail::alpha_ftb(x, dx, (mI ? s : dvec()), (mI ? ds : dvec()), lmb,
                              (mI ? dlam : dvec()), B, tau_pri, tau_dual);

        const double alpha_max =
            std::min(a_ftb, pyu::getattr_or<double>(cfg_, "ip_alpha_max", 1.0));

        double alpha = std::min(1.0, alpha_max);
        int ls_iters = 0;
        bool needs_restoration = false;

        // Line search
        auto ls_res =
            ls_->search(model, x, dx, (mI ? s : dvec()), (mI ? ds : dvec()), mu,
                        g.dot(dx), theta, alpha_max);
        alpha = std::get<0>(ls_res);
        ls_iters = std::get<1>(ls_res);
        needs_restoration = std::get<2>(ls_res);

        // Handle restoration
        const double ls_min_alpha = pyu::getattr_or<double>(
            cfg_, "ls_min_alpha",
            pyu::getattr_or<double>(cfg_, "ip_alpha_min", 1e-10));
        if (alpha <= ls_min_alpha && needs_restoration) {
            dvec dxf = -g;
            const double ng = dxf.norm();
            if (ng > 0)
                dxf /= ng;
            const double a_safe = std::min(alpha_max, 1e-2);
            dvec x_new = x + a_safe * dxf;

            py::dict info;
            info["mode"] = "ip";
            info["step_norm"] = (x_new - x).norm();
            info["accepted"] = true;
            info["converged"] = false;
            info["f"] = model.attr("eval_all")(x_new, py::make_tuple("f"))["f"]
                            .cast<double>();
            info["theta"] =
                model.attr("constraint_violation")(x_new).cast<double>();
            info["stat"] = 0.0;
            info["ineq"] = 0.0;
            info["eq"] = 0.0;
            info["comp"] = 0.0;
            info["ls_iters"] = ls_iters;
            info["alpha"] = 0.0;
            info["rho"] = 0.0;
            info["tr_radius"] = tr_radius_();
            info["mu"] = mu;
            return {x_new, lmb, nuv, info};
        }

        // Take the step
        dvec x_new = x + alpha * dx;
        dvec s_new = (mI ? (s + alpha * ds) : s);
        dvec lmb_new = (mI ? (lmb + alpha * dlam) : lmb);
        dvec nu_new = (mE ? (nuv + alpha * dnu) : nuv);
        dvec zL_new = zL + alpha * dzL;
        dvec zU_new = zU + alpha * dzU;

        // Update bounds and cap duals
        Bounds Bn = detail::get_bounds(model, x_new);
        detail::cap_bound_duals_sigma_box(zL_new, zU_new, Bn, use_shifted,
                                          bound_shift, mu, 1e10);

        // Evaluate at new point
        auto dN = model.attr("eval_all")(
            x_new, py::make_tuple("f", "g", "cI", "cE", "JI", "JE"));
        double f_new = dN["f"].cast<double>();
        dvec g_new = pyconv::to_vec(dN["g"]);
        dvec cI_new = (mI > 0 && !dN["cI"].is_none()) ? pyconv::to_vec(dN["cI"])
                                                      : dvec::Zero(mI);
        dvec cE_new = (mE > 0 && !dN["cE"].is_none()) ? pyconv::to_vec(dN["cE"])
                                                      : dvec::Zero(mE);

        spmat JI_new, JE_new;
        if (mI > 0 && !dN["JI"].is_none())
            JI_new = pyconv::to_sparse(dN["JI"]);
        if (mE > 0 && !dN["JE"].is_none())
            JE_new = pyconv::to_sparse(dN["JE"]);

        double theta_new =
            model.attr("constraint_violation")(x_new).cast<double>();

        // Compute KKT residuals at new point
        dvec r_d_new = g_new;
        if (mI > 0 && JI_new.size())
            r_d_new.noalias() += JI_new.transpose() * lmb_new;
        if (mE > 0 && JE_new.size())
            r_d_new.noalias() += JE_new.transpose() * nu_new;
        r_d_new -= zL_new;
        r_d_new += zU_new;

        // Complementarity residuals at new point
        dvec r_comp_L_new(n), r_comp_U_new(n), r_comp_s_new;
        if (use_shifted) {
            for (int i = 0; i < n; ++i) {
                r_comp_L_new[i] =
                    Bn.hasL[i] ? ((Bn.sL[i] + bound_shift) * zL_new[i] - mu)
                               : 0.0;
                r_comp_U_new[i] =
                    Bn.hasU[i] ? ((Bn.sU[i] + bound_shift) * zU_new[i] - mu)
                               : 0.0;
            }
            if (mI > 0) {
                r_comp_s_new = dvec(mI);
                for (int i = 0; i < mI; ++i) {
                    r_comp_s_new[i] = (s_new[i] + tau_shift) * lmb_new[i] - mu;
                }
            }
        } else {
            for (int i = 0; i < n; ++i) {
                r_comp_L_new[i] =
                    Bn.hasL[i] ? (Bn.sL[i] * zL_new[i] - mu) : 0.0;
                r_comp_U_new[i] =
                    Bn.hasU[i] ? (Bn.sU[i] * zU_new[i] - mu) : 0.0;
            }
            if (mI > 0) {
                r_comp_s_new = dvec(mI);
                for (int i = 0; i < mI; ++i) {
                    r_comp_s_new[i] = s_new[i] * lmb_new[i] - mu;
                }
            }
        }

        py::dict kkt_new;
        kkt_new["stat"] = safe_inf_norm(r_d_new);
        kkt_new["ineq"] =
            (mI > 0) ? safe_inf_norm((cI_new.array().max(0.0)).matrix()) : 0.0;
        kkt_new["eq"] = (mE > 0) ? safe_inf_norm(cE_new) : 0.0;

        double comp_val =
            std::max(safe_inf_norm(r_comp_L_new), safe_inf_norm(r_comp_U_new));
        if (mI > 0 && r_comp_s_new.size()) {
            comp_val = std::max(comp_val, safe_inf_norm(r_comp_s_new));
        }
        kkt_new["comp"] = comp_val;

        const bool converged =
            (kkt_new["stat"].cast<double>() <= tol &&
             kkt_new["ineq"].cast<double>() <= tol &&
             kkt_new["eq"].cast<double>() <= tol &&
             kkt_new["comp"].cast<double>() <= tol && mu <= tol / 10.0);

        // Update barrier parameter
        mu = update_mu_(mu, s_new, lmb_new, theta_new, kkt_new, true,
                        std::numeric_limits<double>::quiet_NaN(), sigma, mu_aff,
                        use_shifted, tau_shift);

        // Build result
        py::dict info;
        info["mode"] = "ip";
        info["step_norm"] = (x_new - x).norm();
        info["accepted"] = true;
        info["converged"] = converged;
        info["f"] = f_new;
        info["theta"] = theta_new;
        info["stat"] = kkt_new["stat"];
        info["ineq"] = kkt_new["ineq"];
        info["eq"] = kkt_new["eq"];
        info["comp"] = kkt_new["comp"];
        info["ls_iters"] = ls_iters;
        info["alpha"] = alpha;
        info["rho"] = 0.0;
        info["tr_radius"] = tr_radius_();
        info["mu"] = mu;
        info["shifted_barrier"] = use_shifted;
        info["tau_shift"] = tau_shift;
        info["bound_shift"] = bound_shift;

        // Update state
        st.s = std::move(s_new);
        st.lam = std::move(lmb_new);
        st.nu = std::move(nu_new);
        st.zL = std::move(zL_new);
        st.zU = std::move(zU_new);
        st.mu = mu;
        st.tau_shift = tau_shift;

        return {x_new, st.lam, st.nu, info};
    }

private:
    py::object cfg_, hess_;
    std::unordered_map<std::string, dvec> kkt_cache_;
    spmat prev_kkt_matrix_;
    std::shared_ptr<kkt::KKTReusable> prev_factorization_{};

    void load_defaults_() {
        auto set_if_missing = [&](const char *name, py::object v) {
            if (!pyu::has_attr(cfg_, name))
                cfg_.attr(name) = v;
        };

        set_if_missing("ip_exact_hessian", py::bool_(true));
        set_if_missing("ip_hess_reg0", py::float_(1e-4));
        set_if_missing("ip_eq_reg", py::float_(1e-4));
        set_if_missing("ip_use_shifted_barrier", py::bool_(false));
        set_if_missing("ip_shift_tau", py::float_(0.1));
        set_if_missing("ip_shift_bounds", py::float_(0.1));
        set_if_missing("ip_shift_adaptive", py::bool_(true));
        set_if_missing("ip_mu_init", py::float_(1e-2));
        set_if_missing("ip_mu_min", py::float_(1e-12));
        set_if_missing("ip_sigma_power", py::float_(3.0));
        set_if_missing("ip_tau_pri", py::float_(0.995));
        set_if_missing("ip_tau_dual", py::float_(0.99));
        set_if_missing("ip_tau", py::float_(0.995));
        set_if_missing("ip_alpha_max", py::float_(1.0));
        set_if_missing("ip_dx_max", py::float_(1e3));
        set_if_missing("ip_theta_clip", py::float_(1e-2));
        set_if_missing("sigma_eps_abs", py::float_(1e-8));
        set_if_missing("sigma_cap", py::float_(1e8));
        set_if_missing("ip_kkt_method", py::str("hykkt"));
        set_if_missing("tol", py::float_(1e-8));
        set_if_missing("ls_backtrack", py::float_(pyu::getattr_or<double>(
                                           cfg_, "ip_alpha_backtrack", 0.5)));
        set_if_missing("ls_armijo_f", py::float_(pyu::getattr_or<double>(
                                          cfg_, "ip_armijo_coeff", 1e-4)));
        set_if_missing("ls_max_iter",
                       py::int_(pyu::getattr_or<int>(cfg_, "ip_ls_max", 30)));
        set_if_missing("ls_min_alpha", py::float_(pyu::getattr_or<double>(
                                           cfg_, "ip_alpha_min", 1e-10)));
    }

    [[nodiscard]] IPState state_from_model_(const py::object &model,
                                            const dvec &x) {
        IPState s{};
        s.mI = pyu::getattr_or<int>(model, "m_ineq", 0);
        s.mE = pyu::getattr_or<int>(model, "m_eq", 0);

        py::dict d = model.attr("eval_all")(x, py::make_tuple("cI", "cE"));
        dvec cI = (s.mI > 0 && !d["cI"].is_none()) ? pyconv::to_vec(d["cI"])
                                                   : dvec::Zero(s.mI);

        const double mu0 =
            clamp_min(pyu::getattr_or<double>(cfg_, "ip_mu_init", 1e-2), 1e-12);
        const bool use_shifted =
            pyu::getattr_or<bool>(cfg_, "ip_use_shifted_barrier", true);
        const double tau_shift =
            use_shifted ? pyu::getattr_or<double>(cfg_, "ip_shift_tau", 0.1)
                        : 0.0;
        const double bound_shift =
            use_shifted ? pyu::getattr_or<double>(cfg_, "ip_shift_bounds", 0.1)
                        : 0.0;

        if (s.mI > 0) {
            s.s = dvec(s.mI);
            s.lam = dvec(s.mI);
            for (int i = 0; i < s.mI; ++i) {
                s.s[i] = clamp_min(-cI[i] + 1e-3, 1.0);
                double denom =
                    (tau_shift > 0.0) ? (s.s[i] + tau_shift) : s.s[i];
                s.lam[i] = clamp_min(mu0 / clamp_min(denom, 1e-12), 1e-8);
            }
        } else {
            s.s.resize(0);
            s.lam.resize(0);
        }

        s.nu = (s.mE > 0) ? dvec::Zero(s.mE) : dvec();

        // Initialize bound duals
        Bounds B = detail::get_bounds(model, x);
        s.zL = dvec::Zero(x.size());
        s.zU = dvec::Zero(x.size());
        for (int i = 0; i < x.size(); ++i) {
            if (B.hasL[i]) {
                s.zL[i] = clamp_min(
                    mu0 / clamp_min(B.sL[i] + bound_shift, 1e-12), 1e-8);
            }
            if (B.hasU[i]) {
                s.zU[i] = clamp_min(
                    mu0 / clamp_min(B.sU[i] + bound_shift, 1e-12), 1e-8);
            }
        }

        s.mu = mu0;
        s.tau_shift = tau_shift;
        s.initialized = true;
        return s;
    }

    [[nodiscard]] double adaptive_shift_slack_(const dvec &s, const dvec &cI,
                                               int it) const {
        if (!s.size())
            return 0.0;
        double base = pyu::getattr_or<double>(cfg_, "ip_shift_tau", 1e-3);
        double min_s = s.minCoeff();
        double max_v = (cI.size() > 0) ? cI.cwiseAbs().maxCoeff() : 0.0;
        if (min_s < 1e-6 || max_v > 1e2) {
            return std::min(1.0, base * (1.0 + 0.1 * it));
        }
        if (min_s > 1e-2 && max_v < 1e-2) {
            return clamp_min(base * (1.0 - 0.05 * it), 0.0);
        }
        return base;
    }

    [[nodiscard]] double adaptive_shift_bounds_(const dvec &x, const Bounds &B,
                                                int it) const {
        bool any_bounds = false;
        for (int i = 0; i < x.size(); ++i) {
            if (B.hasL[i] || B.hasU[i]) {
                any_bounds = true;
                break;
            }
        }
        if (!any_bounds)
            return 0.0;

        double min_gap = consts::INF;
        for (int i = 0; i < x.size(); ++i) {
            if (B.hasL[i])
                min_gap = std::min(min_gap, x[i] - B.lb[i]);
            if (B.hasU[i])
                min_gap = std::min(min_gap, B.ub[i] - x[i]);
        }

        double b0 = pyu::getattr_or<double>(cfg_, "ip_shift_bounds", 0.0);
        if (min_gap < 1e-8) {
            return std::min(1.0, clamp_min(b0, 1e-3) * (1 + 0.05 * it));
        }
        if (min_gap > 1e-2) {
            return clamp_min(b0 * 0.9, 0.0);
        }
        return b0;
    }

    [[nodiscard]] KKTResult solve_KKT_(const spmat &W, const dvec &rhs_x,
                                       const std::optional<spmat> &JE,
                                       const std::optional<dvec> &rpE,
                                       std::string_view method_in) {
        const int n = W.rows();
        int mE = 0;
        std::optional<spmat> G;
        if (JE && JE->rows() > 0) {
            mE = JE->rows();
            G = *JE;
        }

        kkt::dvec r1 = rhs_x;
        std::optional<kkt::dvec> r2;
        if (mE > 0) {
            r2 = rpE ? (-(*rpE)).eval() : kkt::dvec::Zero(mE);
        }

        std::string method_cpp = (method_in == "hykkt") ? "hykkt" : "ldl";
        if (mE == 0 && method_cpp == "hykkt")
            method_cpp = "ldl";

        kkt::ChompConfig conf;
        conf.cg_tol = 1e-6;
        conf.cg_maxit = 200;
        conf.ip_hess_reg0 = 1e-8;
        conf.schur_dense_cutoff = 0.25;

        auto &reg = kkt::default_registry();
        auto strat = reg.get(method_cpp);

        auto [dx, dy, reusable] =
            strat->factor_and_solve(W, G, r1, r2, conf, std::nullopt,
                                    kkt_cache_, 0.0, std::nullopt, true, true);

        prev_kkt_matrix_ = W;
        prev_factorization_ = (method_cpp == "ldl") ? reusable : nullptr;

        return KKTResult{std::move(dx), std::move(dy), std::move(reusable)};
    }

    [[nodiscard]] std::tuple<double, double, double> mehrotra_affine_predictor_(
        const spmat &W, const dvec &r_d, const std::optional<spmat> &JE,
        const std::optional<dvec> &r_pE, const std::optional<spmat> &JI,
        const std::optional<dvec> &r_pI, const dvec &s, const dvec &lmb,
        const dvec &zL, const dvec &zU, const Bounds &B, bool use_shifted,
        double tau_shift, double bound_shift, double mu, double theta) {

        const bool haveJE = JE && JE->rows() > 0 && JE->cols() > 0;

        auto res = solve_KKT_(
            W, -r_d, haveJE ? std::optional<spmat>(*JE) : std::nullopt,
            (r_pE && r_pE->size() > 0) ? std::optional<dvec>(*r_pE)
                                       : std::nullopt,
            pyu::getattr_or<std::string>(cfg_, "ip_kkt_method", "hykkt"));

        dvec dx_aff = std::move(res.dx);
        const int mI = static_cast<int>(s.size());
        const int n = static_cast<int>(zL.size());

        dvec ds_aff, dlam_aff;
        if (mI > 0) {
            ds_aff = -(r_pI.value() + (*JI) * dx_aff);
            dlam_aff = dvec(mI);
            for (int i = 0; i < mI; ++i) {
                double d = use_shifted ? (s[i] + tau_shift) : s[i];
                dlam_aff[i] = sdiv(-(d * lmb[i]) - lmb[i] * ds_aff[i], d);
            }
        }

        dvec dzL_aff = dvec::Zero(n), dzU_aff = dvec::Zero(n);
        for (int i = 0; i < n; ++i) {
            if (B.hasL[i]) {
                double d = (use_shifted && bound_shift > 0.0)
                               ? (B.sL[i] + bound_shift)
                               : B.sL[i];
                dzL_aff[i] = sdiv(-(d * zL[i]) - zL[i] * dx_aff[i], d);
            }
            if (B.hasU[i]) {
                double d = (use_shifted && bound_shift > 0.0)
                               ? (B.sU[i] + bound_shift)
                               : B.sU[i];
                dzU_aff[i] = sdiv(-(d * zU[i]) + zU[i] * dx_aff[i], d);
            }
        }

        const double tau_pri = pyu::getattr_or<double>(
            cfg_, "ip_tau_pri", pyu::getattr_or<double>(cfg_, "ip_tau", 0.995));
        const double tau_dual = pyu::getattr_or<double>(
            cfg_, "ip_tau_dual",
            pyu::getattr_or<double>(cfg_, "ip_tau", 0.995));
        double alpha_aff = 1.0;

        if (mI > 0) {
            for (int i = 0; i < mI; ++i) {
                if (ds_aff[i] < 0.0) {
                    alpha_aff =
                        std::min(alpha_aff,
                                 -s[i] / std::min(ds_aff[i], -consts::EPS_DIV));
                }
            }
            for (int i = 0; i < mI; ++i) {
                if (dlam_aff[i] < 0.0) {
                    alpha_aff = std::min(
                        alpha_aff,
                        -lmb[i] / std::min(dlam_aff[i], -consts::EPS_DIV));
                }
            }
        }

        for (int i = 0; i < n; ++i) {
            if (B.hasL[i] && dx_aff[i] < 0.0) {
                alpha_aff = std::min(alpha_aff,
                                     -(B.sL[i]) /
                                         std::min(dx_aff[i], -consts::EPS_DIV));
            }
        }
        for (int i = 0; i < n; ++i) {
            if (B.hasL[i] && dzL_aff[i] < 0.0) {
                alpha_aff =
                    std::min(alpha_aff,
                             -(zL[i]) / std::min(dzL_aff[i], -consts::EPS_DIV));
            }
        }
        for (int i = 0; i < n; ++i) {
            const double mdx = -dx_aff[i];
            if (B.hasU[i] && mdx < 0.0) {
                alpha_aff = std::min(
                    alpha_aff, -(B.sU[i]) / std::min(mdx, -consts::EPS_DIV));
            }
        }
        for (int i = 0; i < n; ++i) {
            if (B.hasU[i] && dzU_aff[i] < 0.0) {
                alpha_aff =
                    std::min(alpha_aff,
                             -(zU[i]) / std::min(dzU_aff[i], -consts::EPS_DIV));
            }
        }

        alpha_aff = clamp(alpha_aff, 0.0, 1.0);

        const double mu_min = pyu::getattr_or<double>(cfg_, "ip_mu_min", 1e-12);
        double sum_parts = 0.0;
        int denom_cnt = 0;

        if (mI > 0) {
            for (int i = 0; i < mI; ++i) {
                const double s_aff = s[i] + alpha_aff * ds_aff[i];
                const double lam_aff = lmb[i] + alpha_aff * dlam_aff[i];
                const double s_eff = use_shifted ? (s_aff + tau_shift) : s_aff;
                sum_parts += s_eff * lam_aff;
                ++denom_cnt;
            }
        }
        for (int i = 0; i < n; ++i) {
            if (B.hasL[i]) {
                const double sL_aff = B.sL[i] + alpha_aff * dx_aff[i];
                const double zL_af = zL[i] + alpha_aff * dzL_aff[i];
                const double s_eff = (use_shifted && bound_shift > 0.0)
                                         ? (sL_aff + bound_shift)
                                         : sL_aff;
                sum_parts += s_eff * zL_af;
                ++denom_cnt;
            }
        }
        for (int i = 0; i < n; ++i) {
            if (B.hasU[i]) {
                const double sU_aff = B.sU[i] - alpha_aff * dx_aff[i];
                const double zU_af = zU[i] + alpha_aff * dzU_aff[i];
                const double s_eff = (use_shifted && bound_shift > 0.0)
                                         ? (sU_aff + bound_shift)
                                         : sU_aff;
                sum_parts += s_eff * zU_af;
                ++denom_cnt;
            }
        }

        const double mu_aff =
            (denom_cnt > 0)
                ? clamp_min(sum_parts / clamp_min(denom_cnt, 1), mu_min)
                : clamp_min(mu, mu_min);

        const double pwr = pyu::getattr_or<double>(cfg_, "ip_sigma_power", 3.0);
        double sigma =
            (alpha_aff > 0.9)
                ? 0.0
                : clamp(std::pow(1.0 - alpha_aff, 2) *
                            std::pow(mu_aff / clamp_min(mu, mu_min), pwr),
                        0.0, 1.0);

        const double theta_clip =
            pyu::getattr_or<double>(cfg_, "ip_theta_clip", 1e-2);
        if (theta > theta_clip) {
            sigma = clamp_min(sigma, 0.5);
        }

        return {alpha_aff, mu_aff, sigma};
    }

    [[nodiscard]] double compute_error_(const py::object &model, const dvec &x,
                                        const dvec &lam, const dvec &nu,
                                        const dvec &zL, const dvec &zU,
                                        double mu, const dvec &s, int mI_decl) {
        auto data = model
                        .attr("eval_all")(
                            x, py::make_tuple("g", "cI", "cE", "JI", "JE"))
                        .cast<py::dict>();
        dvec g = pyconv::to_vec(data["g"]);
        int mI = pyu::getattr_or<int>(model, "m_ineq", 0);
        int mE = pyu::getattr_or<int>(model, "m_eq", 0);

        dvec cI = (mI > 0 && !data["cI"].is_none()) ? pyconv::to_vec(data["cI"])
                                                    : dvec::Zero(mI);
        dvec cE = (mE > 0 && !data["cE"].is_none()) ? pyconv::to_vec(data["cE"])
                                                    : dvec::Zero(mE);

        spmat JI, JE;
        if (!data["JI"].is_none())
            JI = pyconv::to_sparse(data["JI"]);
        if (!data["JE"].is_none())
            JE = pyconv::to_sparse(data["JE"]);

        Bounds B = detail::get_bounds(model, x);
        const int n = static_cast<int>(x.size());

        // Stationarity residual
        dvec r_d = g;
        if (JI.size())
            r_d.noalias() += JI.transpose() * lam;
        if (JE.size())
            r_d.noalias() += JE.transpose() * nu;
        r_d -= zL;
        r_d += zU;

        // Scaling factors
        const double s_max = pyu::getattr_or<double>(cfg_, "ip_s_max", 100.0);
        const int denom_ct = mI + mE + n;
        const double sum_mults =
            lam.lpNorm<1>() + nu.lpNorm<1>() + zL.lpNorm<1>() + zU.lpNorm<1>();
        const double s_d =
            clamp_min(sum_mults / clamp_min(denom_ct, 1), s_max) / s_max;
        const double s_c =
            clamp_min((zL.lpNorm<1>() + zU.lpNorm<1>()) / clamp_min(n, 1),
                      s_max) /
            s_max;

        // Complementarity residuals
        const bool use_shifted =
            pyu::getattr_or<bool>(cfg_, "ip_use_shifted_barrier", false);
        const double tau_shift = use_shifted ? st.tau_shift : 0.0;
        const double bshift =
            use_shifted ? pyu::getattr_or<double>(cfg_, "ip_shift_bounds", 0.0)
                        : 0.0;

        dvec r_comp_L(n), r_comp_U(n), r_comp_s;
        if (use_shifted) {
            for (int i = 0; i < n; ++i) {
                r_comp_L[i] =
                    B.hasL[i] ? ((B.sL[i] + bshift) * zL[i] - mu) : 0.0;
                r_comp_U[i] =
                    B.hasU[i] ? ((B.sU[i] + bshift) * zU[i] - mu) : 0.0;
            }
            if (mI > 0 && s.size()) {
                r_comp_s.resize(s.size());
                for (int i = 0; i < s.size(); ++i) {
                    r_comp_s[i] = (s[i] + tau_shift) * lam[i] - mu;
                }
            }
        } else {
            for (int i = 0; i < n; ++i) {
                r_comp_L[i] = B.hasL[i] ? (B.sL[i] * zL[i] - mu) : 0.0;
                r_comp_U[i] = B.hasU[i] ? (B.sU[i] * zU[i] - mu) : 0.0;
            }
            if (mI > 0 && s.size()) {
                r_comp_s.resize(s.size());
                for (int i = 0; i < s.size(); ++i) {
                    r_comp_s[i] = s[i] * lam[i] - mu;
                }
            }
        }

        return std::max(
            {safe_inf_norm(r_d) / s_d, (mE > 0) ? safe_inf_norm(cE) : 0.0,
             (mI > 0) ? safe_inf_norm(cI) : 0.0,
             std::max(safe_inf_norm(r_comp_L), safe_inf_norm(r_comp_U)) / s_c,
             (r_comp_s.size() > 0) ? (safe_inf_norm(r_comp_s) / s_c) : 0.0});
    }

    [[nodiscard]] double update_mu_(double mu, const dvec &s, const dvec &lam,
                                    double theta, py::dict &kkt, bool accepted,
                                    double cond_H, double sigma, double mu_aff,
                                    bool use_shifted, double tau_shift) {
        const double mu_min = pyu::getattr_or<double>(cfg_, "ip_mu_min", 1e-12);
        const double kappa = pyu::getattr_or<double>(cfg_, "kappa_mu", 1.5);
        const double theta_tol =
            pyu::getattr_or<double>(cfg_, "tol_feas", 1e-6);
        const double comp_tol = pyu::getattr_or<double>(cfg_, "tol_comp", 1e-6);
        const double cond_max =
            pyu::getattr_or<double>(cfg_, "cond_threshold", 1e6);

        const double comp =
            detail::complementarity(s, lam, mu, tau_shift, use_shifted);
        const bool good =
            (accepted && theta <= theta_tol && comp <= comp_tol &&
             kkt["stat"].cast<double>() <=
                 pyu::getattr_or<double>(cfg_, "tol_stat", 1e-6) &&
             (std::isnan(cond_H) || cond_H <= cond_max));

        const double comp_ratio = comp / clamp_min(mu, 1e-12);
        const double mu_base = clamp_min(sigma * mu_aff, mu_min);

        double mu_new;
        if (good && comp_ratio < 0.1) {
            mu_new = mu_base *
                     std::min(0.1, std::pow(mu_aff / clamp_min(mu, 1e-12), 2));
        } else if (comp_ratio > 10.0 || theta > 10 * theta_tol) {
            mu_new = std::min(10.0 * mu, mu_base * 1.2);
        } else {
            mu_new = std::min(mu_base, std::pow(mu, kappa));
        }
        return clamp_min(mu_new, mu_min);
    }

    [[nodiscard]] double tr_radius_() const { return 0.0; }

    // Gondzio Multiple Centrality Corrections Implementation - Fixed Version
    // Add this to your InteriorPointStepper class

    struct GondzioConfig {
        int max_corrections = 3;         // Maximum number of corrector steps
        double gamma_a = 0.1;            // Lower bound for centrality measure
        double gamma_b = 10.0;           // Upper bound for centrality measure
        double beta_min = 0.1;           // Minimum centering parameter
        double beta_max = 10.0;          // Maximum centering parameter
        double tau_min = 0.005;          // Minimum step length for correction
        bool use_adaptive_gamma = true;  // Adapt gamma bounds based on progress
        double progress_threshold = 0.1; // Threshold for progress detection
    };

private:
    GondzioConfig gondzio_config_;

    // Initialize Gondzio configuration in constructor or load_defaults_()
    void load_gondzio_defaults_() {
        gondzio_config_.max_corrections =
            pyu::getattr_or<int>(cfg_, "gondzio_max_corrections", 2);
        gondzio_config_.gamma_a =
            pyu::getattr_or<double>(cfg_, "gondzio_gamma_a", 0.1);
        gondzio_config_.gamma_b =
            pyu::getattr_or<double>(cfg_, "gondzio_gamma_b", 10.0);
        gondzio_config_.beta_min =
            pyu::getattr_or<double>(cfg_, "gondzio_beta_min", 0.1);
        gondzio_config_.beta_max =
            pyu::getattr_or<double>(cfg_, "gondzio_beta_max", 10.0);
        gondzio_config_.tau_min =
            pyu::getattr_or<double>(cfg_, "gondzio_tau_min", 0.005);
        gondzio_config_.use_adaptive_gamma =
            pyu::getattr_or<bool>(cfg_, "gondzio_adaptive_gamma", true);
        gondzio_config_.progress_threshold =
            pyu::getattr_or<double>(cfg_, "gondzio_progress_threshold", 0.1);
    }

    struct GondzioStepData {
        dvec dx, dnu, ds, dlam, dzL, dzU;
        double alpha_pri, alpha_dual;
        double mu_target;
        int correction_count;
        bool use_correction;
    };

    // Compute step lengths for Gondzio corrections
    [[nodiscard]] std::pair<double, double> compute_gondzio_step_lengths(
        const dvec &s, const dvec &ds, const dvec &lam, const dvec &dlam,
        const dvec &zL, const dvec &dzL, const dvec &zU, const dvec &dzU,
        const dvec &dx, const Bounds &B, double tau_pri,
        double tau_dual) const {

        double alpha_pri = 1.0;
        double alpha_dual = 1.0;

        // Primal step length from inequality slacks
        if (s.size() > 0) {
            for (int i = 0; i < s.size(); ++i) {
                if (ds[i] < 0.0) {
                    alpha_pri = std::min(
                        alpha_pri, -s[i] / std::min(ds[i], -consts::EPS_DIV));
                }
            }
        }

        // Dual step length from inequality multipliers
        if (lam.size() > 0) {
            for (int i = 0; i < lam.size(); ++i) {
                if (dlam[i] < 0.0) {
                    alpha_dual =
                        std::min(alpha_dual,
                                 -lam[i] / std::min(dlam[i], -consts::EPS_DIV));
                }
            }
        }

        // Primal step length from bound constraints
        const int n = static_cast<int>(dx.size());
        for (int i = 0; i < n; ++i) {
            if (B.hasL[i] && dx[i] < 0.0) {
                alpha_pri = std::min(
                    alpha_pri, -(B.sL[i]) / std::min(dx[i], -consts::EPS_DIV));
            }
            if (B.hasU[i] && dx[i] > 0.0) {
                alpha_pri = std::min(
                    alpha_pri, (B.sU[i]) / clamp_min(dx[i], consts::EPS_DIV));
            }
        }

        // Dual step length from bound multipliers
        for (int i = 0; i < n && i < zL.size(); ++i) {
            if (B.hasL[i] && dzL[i] < 0.0) {
                alpha_dual = std::min(
                    alpha_dual, -zL[i] / std::min(dzL[i], -consts::EPS_DIV));
            }
        }
        for (int i = 0; i < n && i < zU.size(); ++i) {
            if (B.hasU[i] && dzU[i] < 0.0) {
                alpha_dual = std::min(
                    alpha_dual, -zU[i] / std::min(dzU[i], -consts::EPS_DIV));
            }
        }

        alpha_pri = clamp(tau_pri * alpha_pri, 0.0, 1.0);
        alpha_dual = clamp(tau_dual * alpha_dual, 0.0, 1.0);

        return {alpha_pri, alpha_dual};
    }

    // Compute centrality measure for complementarity pairs
    [[nodiscard]] double compute_centrality_measure(
        const dvec &s, const dvec &lam, const dvec &ds, const dvec &dlam,
        const dvec &zL, const dvec &zU, const dvec &dzL, const dvec &dzU,
        const dvec &dx, const Bounds &B, double alpha_pri, double alpha_dual,
        double mu_target, bool use_shifted, double tau_shift,
        double bound_shift) const {
        double min_ratio = std::numeric_limits<double>::infinity();
        double max_ratio = 0.0;
        int count = 0;

        // Inequality constraints centrality
        if (s.size() > 0) {
            for (int i = 0; i < s.size(); ++i) {
                const double s_new = s[i] + alpha_pri * ds[i];
                const double lam_new = lam[i] + alpha_dual * dlam[i];

                if (s_new > consts::EPS_POS && lam_new > consts::EPS_POS) {
                    const double s_eff =
                        use_shifted ? (s_new + tau_shift) : s_new;
                    const double product = s_eff * lam_new;
                    const double ratio = product / mu_target;

                    min_ratio = std::min(min_ratio, ratio);
                    max_ratio = std::max(max_ratio, ratio);
                    count++;
                }
            }
        }

        // Bound constraints centrality
        const int n = static_cast<int>(dx.size());
        for (int i = 0; i < n && i < zL.size(); ++i) {
            if (B.hasL[i]) {
                // For lower bounds: s_L = x - lb, so ds_L = dx
                const double sL_new = B.sL[i] + alpha_pri * dx[i];
                const double zL_new = zL[i] + alpha_dual * dzL[i];

                if (sL_new > consts::EPS_POS && zL_new > consts::EPS_POS) {
                    const double s_eff =
                        use_shifted ? (sL_new + bound_shift) : sL_new;
                    const double product = s_eff * zL_new;
                    const double ratio = product / mu_target;

                    min_ratio = std::min(min_ratio, ratio);
                    max_ratio = std::max(max_ratio, ratio);
                    count++;
                }
            }
        }

        for (int i = 0; i < n && i < zU.size(); ++i) {
            if (B.hasU[i]) {
                // For upper bounds: s_U = ub - x, so ds_U = -dx
                const double sU_new = B.sU[i] + alpha_pri * (-dx[i]);
                const double zU_new = zU[i] + alpha_dual * dzU[i];

                if (sU_new > consts::EPS_POS && zU_new > consts::EPS_POS) {
                    const double s_eff =
                        use_shifted ? (sU_new + bound_shift) : sU_new;
                    const double product = s_eff * zU_new;
                    const double ratio = product / mu_target;

                    min_ratio = std::min(min_ratio, ratio);
                    max_ratio = std::max(max_ratio, ratio);
                    count++;
                }
            }
        }

        if (count == 0)
            return 1.0; // Perfect centrality if no constraints

        // Return the ratio of max to min - closer to 1.0 means better
        // centrality
        return (min_ratio > 0) ? (max_ratio / min_ratio)
                               : std::numeric_limits<double>::infinity();
    }

    // Check if Gondzio correction should be applied
    [[nodiscard]] bool
    should_apply_gondzio_correction(double centrality_measure, double alpha_max,
                                    const GondzioConfig &config) const {
        return (centrality_measure > config.gamma_b ||
                centrality_measure < config.gamma_a) &&
               alpha_max >= config.tau_min;
    }

    // Compute corrector RHS for Gondzio step
    [[nodiscard]] std::pair<dvec, dvec> compute_gondzio_corrector_rhs(
        const dvec &s, const dvec &lam, const dvec &ds, const dvec &dlam,
        const dvec &zL, const dvec &zU, const dvec &dzL, const dvec &dzU,
        const dvec &dx, const Bounds &B, const spmat &JI, double alpha_pri,
        double alpha_dual, double mu_target, double centrality_measure,
        bool use_shifted, double tau_shift, double bound_shift,
        const Sigmas &Sg) const {

        const int n = static_cast<int>(dx.size());
        const int mI = static_cast<int>(s.size());

        // Adaptive centering parameter based on centrality measure
        double beta = 1.0;
        if (centrality_measure > gondzio_config_.gamma_b) {
            beta = clamp(2.0 * centrality_measure / gondzio_config_.gamma_b,
                         gondzio_config_.beta_min, gondzio_config_.beta_max);
        } else if (centrality_measure < gondzio_config_.gamma_a) {
            beta = clamp(gondzio_config_.gamma_a / (2.0 * centrality_measure),
                         gondzio_config_.beta_min, gondzio_config_.beta_max);
        }

        dvec rhs_x = dvec::Zero(n);
        dvec rhs_s = dvec::Zero(mI);

        // Corrector for inequality constraints
        if (mI > 0) {
            for (int i = 0; i < mI; ++i) {
                const double s_pred = s[i] + alpha_pri * ds[i];
                const double lam_pred = lam[i] + alpha_dual * dlam[i];
                const double s_eff =
                    use_shifted ? (s_pred + tau_shift) : s_pred;

                const double rc =
                    -ds[i] * dlam[i] + beta * mu_target - s_eff * lam_pred;

                // If Sigma_s == Λ^{-1}, do NOT divide by λ here; push Λ^{-1}
                // once via Sigma_s
                rhs_s[i] = rc;
            }
            if (JI.size() > 0 && Sg.Sigma_s.size() == mI) {
                rhs_x.noalias() +=
                    JI.transpose() * (Sg.Sigma_s.asDiagonal() * rhs_s);
            }
        }

        // Corrector for bound constraints
        for (int i = 0; i < n; ++i) {
            double bound_correction = 0.0;

            if (i < zL.size() && B.hasL[i]) {
                const double sL_pred = B.sL[i] + alpha_pri * dx[i];
                const double zL_pred = zL[i] + alpha_dual * dzL[i];
                const double s_eff =
                    use_shifted ? (sL_pred + bound_shift) : sL_pred;

                // Gondzio corrector for lower bounds
                const double corrector_L =
                    -dx[i] * dzL[i] + beta * mu_target - s_eff * zL_pred;
                bound_correction +=
                    corrector_L / clamp_min(s_eff, consts::EPS_POS);
            }

            if (i < zU.size() && B.hasU[i]) {
                const double sU_pred = B.sU[i] - alpha_pri * dx[i];
                const double zU_pred = zU[i] + alpha_dual * dzU[i];
                const double s_eff =
                    use_shifted ? (sU_pred + bound_shift) : sU_pred;

                // Gondzio corrector for upper bounds
                const double corrector_U =
                    dx[i] * dzU[i] + beta * mu_target - s_eff * zU_pred;
                bound_correction -=
                    corrector_U / clamp_min(s_eff, consts::EPS_POS);
            }

            rhs_x[i] += bound_correction;
        }

        return {rhs_x, rhs_s};
    }

    // Main Gondzio multiple centrality corrections
    [[nodiscard]] GondzioStepData gondzio_multiple_corrections(
        const spmat &W, const dvec &r_d, const std::optional<spmat> &JE,
        const std::optional<dvec> &r_pE, const std::optional<spmat> &JI,
        const std::optional<dvec> &r_pI, const dvec &s, const dvec &lam,
        const dvec &zL, const dvec &zU, const Bounds &B, double mu_target,
        bool use_shifted, double tau_shift, double bound_shift,
        const Sigmas &Sg, const dvec &base_dx, const dvec &base_dnu) {

        GondzioStepData result;
        result.dx = base_dx;
        result.dnu = base_dnu;
        result.correction_count = 0;
        result.use_correction = false;
        result.mu_target = mu_target;

        const int mI = static_cast<int>(s.size());
        const int n = static_cast<int>(zL.size());

        // Recover base ds, dlam, dzL, dzU from base solution
        if (mI > 0 && JI) {
            result.ds = -(r_pI.value() + (*JI) * base_dx);
            result.dlam = dvec(mI);
            for (int i = 0; i < mI; ++i) {
                double d = use_shifted ? (s[i] + tau_shift) : s[i];
                result.dlam[i] =
                    sdiv(mu_target - d * lam[i] - lam[i] * result.ds[i], d);
            }
        } else {
            result.ds = dvec::Zero(mI);
            result.dlam = dvec::Zero(mI);
        }

        auto [dzL_base, dzU_base] = detail::dz_bounds_from_dx(
            base_dx, zL, zU, B, bound_shift, use_shifted, mu_target, true);
        result.dzL = dzL_base;
        result.dzU = dzU_base;

        // Compute initial step lengths
        const double tau_pri =
            pyu::getattr_or<double>(cfg_, "ip_tau_pri", 0.995);
        const double tau_dual =
            pyu::getattr_or<double>(cfg_, "ip_tau_dual", 0.995);

        auto [alpha_pri_init, alpha_dual_init] = compute_gondzio_step_lengths(
            s, result.ds, lam, result.dlam, zL, result.dzL, zU, result.dzU,
            result.dx, B, tau_pri, tau_dual);
        result.alpha_pri = alpha_pri_init;
        result.alpha_dual = alpha_dual_init;

        // Apply Gondzio corrections
        for (int k = 0; k < gondzio_config_.max_corrections; ++k) {
            // Compute centrality measure
            double centrality = compute_centrality_measure(
                s, lam, result.ds, result.dlam, zL, zU, result.dzL, result.dzU,
                result.dx, B, result.alpha_pri, result.alpha_dual, mu_target,
                use_shifted, tau_shift, bound_shift);

            // Check if correction is needed
            const double alpha_max =
                std::min(result.alpha_pri, result.alpha_dual);
            if (!should_apply_gondzio_correction(centrality, alpha_max,
                                                 gondzio_config_)) {
                break; // Satisfactory centrality achieved
            }

            // Compute corrector RHS
            auto [rhs_corr_x, rhs_corr_s] = compute_gondzio_corrector_rhs(
                s, lam, result.ds, result.dlam, zL, zU, result.dzL, result.dzU,
                result.dx, B, JI ? *JI : spmat(), result.alpha_pri,
                result.alpha_dual, mu_target, centrality, use_shifted,
                tau_shift, bound_shift, Sg);

            // Solve corrector system
            auto corr_res = solve_KKT_(
                W, rhs_corr_x, JE, std::nullopt, // Use same factorization
                pyu::getattr_or<std::string>(cfg_, "ip_kkt_method", "hykkt"));

            // Add correction to current direction
            result.dx += corr_res.dx;
            if (JE && corr_res.dy.size() > 0) {
                result.dnu += corr_res.dy;
            }

            // Recompute ds, dlam, dzL, dzU with corrected dx
            if (mI > 0 && JI) {
                result.ds = -(r_pI.value() + (*JI) * result.dx);
                for (int i = 0; i < mI; ++i) {
                    double d = use_shifted ? (s[i] + tau_shift) : s[i];
                    result.dlam[i] =
                        sdiv(mu_target - d * lam[i] - lam[i] * result.ds[i], d);
                }
            }

            auto [dzL_corr, dzU_corr] =
                detail::dz_bounds_from_dx(result.dx, zL, zU, B, bound_shift,
                                          use_shifted, mu_target, true);
            result.dzL = dzL_corr;
            result.dzU = dzU_corr;

            // Recompute step lengths
            auto [alpha_pri_new, alpha_dual_new] = compute_gondzio_step_lengths(
                s, result.ds, lam, result.dlam, zL, result.dzL, zU, result.dzU,
                result.dx, B, tau_pri, tau_dual);
            result.alpha_pri = alpha_pri_new;
            result.alpha_dual = alpha_dual_new;

            result.correction_count++;
            result.use_correction = true;

            // Adaptive gamma bounds based on progress
            if (gondzio_config_.use_adaptive_gamma && k > 0) {
                double prev_centrality = centrality;
                centrality = compute_centrality_measure(
                    s, lam, result.ds, result.dlam, zL, zU, result.dzL,
                    result.dzU, result.dx, B, result.alpha_pri,
                    result.alpha_dual, mu_target, use_shifted, tau_shift,
                    bound_shift);

                // If not making sufficient progress, stop corrections
                if (std::abs(centrality - prev_centrality) <
                    gondzio_config_.progress_threshold) {
                    break;
                }
            }
        }

        return result;
    }

    // Modified Mehrotra predictor to use Gondzio corrections
    [[nodiscard]] std::tuple<double, double, double, GondzioStepData>
    mehrotra_with_gondzio_corrections_(
        const spmat &W, const dvec &r_d, const std::optional<spmat> &JE,
        const std::optional<dvec> &r_pE, const std::optional<spmat> &JI,
        const std::optional<dvec> &r_pI, const dvec &s, const dvec &lam,
        const dvec &zL, const dvec &zU, const Bounds &B, bool use_shifted,
        double tau_shift, double bound_shift, double mu, double theta,
        const Sigmas &Sg) {

        // Standard affine predictor step (same as before)
        auto [alpha_aff, mu_aff, sigma] = mehrotra_affine_predictor_(
            W, r_d, JE, r_pE, JI, r_pI, s, lam, zL, zU, B, use_shifted,
            tau_shift, bound_shift, mu, theta);

        // Compute target mu for centering
        const double mu_min = pyu::getattr_or<double>(cfg_, "ip_mu_min", 1e-12);
        double mu_target = clamp_min(sigma * mu_aff, mu_min);

        // Solve basic corrector system (Mehrotra)
        dvec rhs_x = -r_d;

        const int mI = static_cast<int>(s.size());
        const int n = static_cast<int>(zL.size());

        // Add inequality constraint corrector terms
        if (mI > 0 && JI && Sg.Sigma_s.size()) {
            dvec rc_s(mI);
            for (int i = 0; i < mI; ++i) {
                const double ds = use_shifted ? (s[i] + tau_shift) : s[i];
                rc_s[i] = mu_target - ds * lam[i];
            }
            dvec temp(mI);
            for (int i = 0; i < mI; ++i) {
                const double lam_safe =
                    (std::abs(lam[i]) < consts::EPS_POS)
                        ? ((lam[i] >= 0) ? consts::EPS_POS : -consts::EPS_POS)
                        : lam[i];
                temp[i] = rc_s[i] / lam_safe;
            }
            rhs_x.noalias() +=
                (*JI).transpose() * (Sg.Sigma_s.asDiagonal() * temp);
        }

        // Add bound constraint corrector terms
        for (int i = 0; i < n; ++i) {
            if (B.hasL[i]) {
                double denom =
                    clamp_min(use_shifted ? (B.sL[i] + bound_shift) : B.sL[i],
                              consts::EPS_POS);
                rhs_x[i] += (mu_target - denom * zL[i]) / denom;
            }
        }
        for (int i = 0; i < n; ++i) {
            if (B.hasU[i]) {
                double denom =
                    clamp_min(use_shifted ? (B.sU[i] + bound_shift) : B.sU[i],
                              consts::EPS_POS);
                rhs_x[i] -= (mu_target - denom * zU[i]) / denom;
            }
        }

        auto base_res = solve_KKT_(
            W, rhs_x, JE, r_pE,
            pyu::getattr_or<std::string>(cfg_, "ip_kkt_method", "hykkt"));

        // Apply Gondzio multiple corrections
        GondzioStepData gondzio_result = gondzio_multiple_corrections(
            W, r_d, JE, r_pE, JI, r_pI, s, lam, zL, zU, B, mu_target,
            use_shifted, tau_shift, bound_shift, Sg, base_res.dx, base_res.dy);

        return {alpha_aff, mu_aff, sigma, gondzio_result};
    }
};