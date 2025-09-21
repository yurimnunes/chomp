// ip_cpp.cpp — cleaned & modernized (drop-in)
// Behavior: identical to your working version, but faster & tidier.

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

#include "../include/definitions.h"
#include "../include/funnel.h"
#include "../include/kkt_core.h"
#include "../include/linesearch.h"
#include "../include/regularizer.h"

namespace py = pybind11;

struct KKTResult {
    dvec dx; // primal search direction
    dvec dy; // equality multipliers (empty if no JE)
    std::shared_ptr<kkt::KKTReusable>
        reusable; // factorization handle (set for LDL, optional otherwise)
};

namespace consts {
constexpr double EPS_DIV = 1e-16;
constexpr double EPS_POS = 1e-12;
constexpr double ONE = 1.0;
constexpr double ZERO = 0.0;
} // namespace consts

// ---------- Small value clamps ----------
template <class T> [[nodiscard]] inline T clamp_min(T v, T lo) noexcept {
    return (v < lo) ? lo : v;
}

template <class T> [[nodiscard]] inline T clamp_max(T v, T hi) noexcept {
    return (v > hi) ? hi : v;
}

template <class T> [[nodiscard]] inline T clamp(T v, T lo, T hi) noexcept {
    return (v < lo) ? lo : (v > hi) ? hi : v;
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

// ---------- Python ↔ Eigen conversions (fast paths) ----------
namespace pyconv {

[[nodiscard]] inline dvec to_vec(const py::object &arr) {
    if (arr.is_none())
        return dvec(); // Return empty vector for None

    py::array_t<double, py::array::c_style | py::array::forcecast> a(arr);

    // Handle 0-dimensional arrays (scalars) - return empty array
    if (a.ndim() == 0) {
        return dvec(); // Return empty vector for scalars
    }

    // Handle 1-dimensional arrays (including empty ones)
    if (a.ndim() == 1) {
        // Check for empty array
        if (a.size() == 0) {
            return dvec(); // Return empty vector for empty arrays
        }

        auto r = a.unchecked<1>();
        dvec v(r.shape(0));
        for (ssize_t i = 0; i < r.shape(0); ++i)
            v[i] = r(i);
        return v;
    }

    // Reject higher-dimensional arrays
    throw std::runtime_error("to_vec: Expected 0-D or 1-D array, got " +
                             std::to_string(a.ndim()) + "-D array");
}
[[nodiscard]] inline spmat from_dense(const py::array &a_in) {
    py::array_t<double, py::array::c_style | py::array::forcecast> a(a_in);
    if (a.ndim() != 2)
        throw std::runtime_error("expected 2D dense array");
    const int rows = static_cast<int>(a.shape(0));
    const int cols = static_cast<int>(a.shape(1));
    dmat M(rows, cols);
    auto r = a.unchecked<2>();
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            M(i, j) = r(i, j);
    spmat S = M.sparseView();
    S.makeCompressed();
    return S;
}

[[nodiscard]] inline bool is_scipy_sparse(const py::object &o) noexcept {
    return !o.is_none() && PyObject_HasAttrString(o.ptr(), "tocoo") &&
           PyObject_HasAttrString(o.ptr(), "shape");
}

[[nodiscard]] inline spmat any_scipy_to_coo(const py::object &sp) {
    py::object coo = sp.attr("tocoo")();
    auto shape = coo.attr("shape").cast<std::pair<py::ssize_t, py::ssize_t>>();
    const int rows = static_cast<int>(shape.first);
    const int cols = static_cast<int>(shape.second);

    auto row = coo.attr("row")
                   .cast<py::array_t<long long, py::array::c_style |
                                                    py::array::forcecast>>();
    auto col = coo.attr("col")
                   .cast<py::array_t<long long, py::array::c_style |
                                                    py::array::forcecast>>();
    auto dat = coo.attr("data")
                   .cast<py::array_t<double, py::array::c_style |
                                                 py::array::forcecast>>();

    auto R = row.unchecked<1>();
    auto C = col.unchecked<1>();
    auto X = dat.unchecked<1>();

    std::vector<Eigen::Triplet<double>> trip;
    trip.reserve(static_cast<size_t>(X.shape(0)));
    for (ssize_t k = 0; k < X.shape(0); ++k)
        trip.emplace_back(static_cast<int>(R(k)), static_cast<int>(C(k)), X(k));

    spmat A(rows, cols);
    A.setFromTriplets(trip.begin(), trip.end());
    A.makeCompressed();
    return A;
}

[[nodiscard]] inline spmat to_sparse(const py::object &obj) {
    if (obj.is_none())
        return spmat();
    if (is_scipy_sparse(obj))
        return any_scipy_to_coo(obj);
    // Fallback: dense numpy array
    py::array a = py::array(obj);
    return from_dense(a);
}

} // namespace pyconv

// ---------- State & small structs ----------
struct IPState {
    int mI = 0, mE = 0;
    dvec s, lam, nu, zL, zU;
    double mu = 1e-2;
    double tau_shift = 0.0;
    bool initialized = false;
};

struct Bounds {
    dvec lb, ub, sL, sU;
    std::vector<uint8_t> hasL, hasU; // 0/1 to be trivially copyable
};

struct Sigmas {
    dvec Sigma_x, Sigma_s; // x-diagonal and inequality diagonal
};

// ---------- math helpers ----------
namespace detail {

[[nodiscard]] inline double safe_inf_norm(const dvec &v) noexcept {
    double m = 0.0;
    for (int i = 0; i < v.size(); ++i)
        m = std::max(m, std::abs(v[i]));
    return m;
}

[[nodiscard]] inline double sdiv(double a, double b,
                                 double eps = consts::EPS_DIV) noexcept {
    return a / ((b > eps) ? b : eps);
}

[[nodiscard]] inline double spdiv(double a, double b,
                                  double eps = 1e-12) noexcept {
    const double d = (b > eps) ? b : eps;
    return a / d;
}

[[nodiscard]] inline dvec zeros_like(int n) { return dvec::Zero(n); }

[[nodiscard]] inline Bounds get_bounds(const py::object &model, const dvec &x) {
    const int n = static_cast<int>(x.size());
    Bounds B;

    // Handle lower bounds - check for empty array instead of None
    if (pyu::has_attr(model, "lb") && !model.attr("lb").is_none()) {
        dvec lb_vec = pyconv::to_vec(model.attr("lb"));
        // cout lb_vec size
        // contents
        if (lb_vec.size() == n) {
            // Non-empty array with correct size
            B.lb = std::move(lb_vec);
        } else if (lb_vec.size() == 0) {
            // Empty array - treat as no bounds
            B.lb = dvec::Constant(n, -std::numeric_limits<double>::infinity());
        } else {
            // Size mismatch - this shouldn't happen if constructor is correct
            throw std::runtime_error(
                "Lower bounds size mismatch in get_bounds");
        }
    } else {
        B.lb = dvec::Constant(n, -std::numeric_limits<double>::infinity());
    }

    // Handle upper bounds - check for empty array instead of None
    if (pyu::has_attr(model, "ub") && !model.attr("ub").is_none()) {
        dvec ub_vec = pyconv::to_vec(model.attr("ub"));
        if (ub_vec.size() == n) {
            // Non-empty array with correct size
            B.ub = std::move(ub_vec);
        } else if (ub_vec.size() == 0) {
            // Empty array - treat as no bounds
            B.ub = dvec::Constant(n, +std::numeric_limits<double>::infinity());
        } else {
            // Size mismatch - this shouldn't happen if constructor is correct
            throw std::runtime_error(
                "Upper bounds size mismatch in get_bounds");
        }
    } else {
        B.ub = dvec::Constant(n, +std::numeric_limits<double>::infinity());
    }

    // Initialize bound indicators and slack variables
    B.hasL.assign(n, 0);
    B.hasU.assign(n, 0);
    B.sL = dvec(n);
    B.sU = dvec(n);

    for (int i = 0; i < n; ++i) {
        const bool hL = std::isfinite(B.lb[i]);
        const bool hU = std::isfinite(B.ub[i]);
        B.hasL[i] = static_cast<uint8_t>(hL);
        B.hasU[i] = static_cast<uint8_t>(hU);
        B.sL[i] = hL ? std::max(consts::EPS_POS, x[i] - B.lb[i]) : 1.0;
        B.sU[i] = hU ? std::max(consts::EPS_POS, B.ub[i] - x[i]) : 1.0;
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
            v += zL[i] / std::max(d, eps_abs);
        }
        if (B.hasU[i]) {
            double d = B.sU[i] + (use_shifted ? bound_shift : 0.0);
            v += zU[i] / std::max(d, eps_abs);
        }
        S.Sigma_x[i] = clamp(v, 0.0, cap);
    }

    S.Sigma_s = dvec::Zero(mI);
    if (mI > 0) {
        if (use_shifted) {
            for (int i = 0; i < mI; ++i) {
                double d = s[i] + tau_shift;
                S.Sigma_s[i] = clamp(lmb[i] / std::max(d, eps_abs), 0.0, cap);
            }
        } else {
            for (int i = 0; i < mI; ++i) {
                double sf = std::abs(cI[i]);
                sf = clamp(sf, 1e-8, 1.0);
                double sv = std::max(s[i], sf);
                S.Sigma_s[i] = clamp(lmb[i] / std::max(sv, eps_abs), 0.0, cap);
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
    dvec dzL = dvec::Zero(n), dzU = dvec::Zero(n);
    for (int i = 0; i < n; ++i) {
        if (B.hasL[i]) {
            double d = B.sL[i] + (use_shifted ? bound_shift : 0.0);
            d = std::max(d, consts::EPS_DIV);
            dzL[i] = use_mu ? ((mu - d * zL[i] - zL[i] * dx[i]) / d)
                            : (-(zL[i] * dx[i]) / d);
        }
        if (B.hasU[i]) {
            double d = B.sU[i] + (use_shifted ? bound_shift : 0.0);
            d = std::max(d, consts::EPS_DIV);
            dzU[i] = use_mu ? ((mu - d * zU[i] + zU[i] * dx[i]) / d)
                            : ((zU[i] * dx[i]) / d);
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
        for (int i = 0; i < m; ++i)
            acc += std::abs((s[i] + tau_shift) * lmb[i] - mu);
    } else {
        for (int i = 0; i < m; ++i)
            acc += std::abs(s[i] * lmb[i] - mu);
    }
    return acc / std::max(1, m);
}

[[nodiscard]] inline double alpha_ftb(const dvec &x, const dvec &dx,
                                      const dvec &s, const dvec &ds,
                                      const dvec &lmb, const dvec &dlam,
                                      const Bounds &B, double tau_pri,
                                      double tau_dual) {
    using namespace consts;
    double a_pri = 1.0, a_dual = 1.0;
    // Inequalities
    for (int i = 0; i < s.size(); ++i)
        if (ds[i] < 0.0)
            a_pri = std::min(a_pri, -s[i] / std::min(ds[i], -EPS_DIV));
    for (int i = 0; i < lmb.size(); ++i)
        if (dlam[i] < 0.0)
            a_dual = std::min(a_dual, -lmb[i] / std::min(dlam[i], -EPS_DIV));
    // Box
    for (int i = 0; i < x.size(); ++i) {
        if (B.hasL[i] && dx[i] < 0.0)
            a_pri =
                std::min(a_pri, -(x[i] - B.lb[i]) / std::min(dx[i], -EPS_DIV));
        if (B.hasU[i] && dx[i] > 0.0)
            a_pri =
                std::min(a_pri, (B.ub[i] - x[i]) / std::max(dx[i], EPS_DIV));
    }
    double a = std::min(tau_pri * a_pri, tau_dual * a_dual);
    return clamp(a, 0.0, 1.0);
}

inline void cap_bound_duals_sigma_box(dvec &zL, dvec &zU, const Bounds &B,
                                      bool use_shifted, double bound_shift,
                                      double mu, double ksig = 1e10) {
    for (int i = 0; i < zL.size(); ++i) {
        if (B.hasL[i]) {
            double sLc = B.sL[i] + (use_shifted ? bound_shift : 0.0);
            sLc = clamp_min(sLc, consts::EPS_DIV);
            double lo = mu / (ksig * sLc);
            double hi = (ksig * mu) / sLc;
            zL[i] = clamp(zL[i], lo, hi);
        }
        if (B.hasU[i]) {
            double sUc = B.sU[i] + (use_shifted ? bound_shift : 0.0);
            sUc = clamp_min(sUc, consts::EPS_DIV);
            double lo = mu / (ksig * sUc);
            double hi = (ksig * mu) / sUc;
            zU[i] = clamp(zU[i], lo, hi);
        }
    }
}

} // namespace detail

// ---------- InteriorPointStepper ----------
class InteriorPointStepper {
public:
    IPState st{};

    std::shared_ptr<LineSearcher> ls_;
    std::shared_ptr<regx::Regularizer> regularizer_ =
        std::make_shared<regx::Regularizer>();

    InteriorPointStepper(py::object cfg, py::object hess)
        : cfg_(std::move(cfg)), hess_(std::move(hess)) {
        load_defaults_();
        std::shared_ptr<Funnel> funnel = std::shared_ptr<Funnel>();
        ls_ = std::make_shared<LineSearcher>(cfg_, py::none(), funnel);
    }

    // --- main step (API unchanged) ---
    std::tuple<dvec, dvec, dvec, py::dict>
    step(py::object model, const dvec &x, const dvec &lam, const dvec &nu,
         int it, std::optional<IPState> ip_state_opt = std::nullopt) {
        using namespace consts;

        if (!st.initialized)
            st = state_from_model_(model, x);

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

        // eval
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
            if (JI.rows() != mI || JI.cols() != n)
                throw std::runtime_error("JI dimension mismatch");
        }
        if (mE > 0 && !d0["JE"].is_none()) {
            JE = pyconv::to_sparse(d0["JE"]);
            if (JE.rows() != mE || JE.cols() != n)
                throw std::runtime_error("JE dimension mismatch");
        }

        double theta = model.attr("constraint_violation")(x).cast<double>();

        // bounds (with adaptive shifts)
        Bounds B = detail::get_bounds(model, x);
        if (use_shifted && shift_adapt) {
            tau_shift = adaptive_shift_slack_(s, cI, it);
            st.tau_shift = tau_shift;
        }
        double bound_shift =
            use_shifted ? pyu::getattr_or<double>(cfg_, "ip_shift_bounds", 0.0)
                        : 0.0;
        if (use_shifted && shift_adapt)
            bound_shift = adaptive_shift_bounds_(x, B, it);

        // quick exit
        const double err_0 =
            compute_error_(model, x, lmb, nuv, zL, zU, 0.0, s, mI);
        const double tol = pyu::getattr_or<double>(cfg_, "tol", 1e-8);
        if (err_0 <= tol) {
            py::dict info;
            info["mode"] = "ip";
            info["step_norm"] = 0.0;
            info["accepted"] = true;
            info["converged"] = true;
            info["f"] = f;
            info["theta"] = theta;
            info["stat"] = detail::safe_inf_norm(g);
            info["ineq"] =
                (mI > 0) ? detail::safe_inf_norm((cI.array().max(0.0)).matrix())
                         : 0.0;
            info["eq"] = (mE > 0) ? detail::safe_inf_norm(cE) : 0.0;
            info["comp"] = 0.0;
            info["ls_iters"] = 0;
            info["alpha"] = 0.0;
            info["rho"] = 0.0;
            info["tr_radius"] = 0.0;
            info["mu"] = mu;
            return {x, lmb, nuv, info};
        }

        // Sigma/diagonal
        const double eps_abs =
            pyu::getattr_or<double>(cfg_, "sigma_eps_abs", 1e-8);
        const double cap = pyu::getattr_or<double>(cfg_, "sigma_cap", 1e8);
        Sigmas Sg =
            detail::build_sigmas(zL, zU, B, lmb, s, cI, tau_shift, bound_shift,
                                 use_shifted, eps_abs, cap);

        // Hessian + PSD
        py::object H_obj = pyu::getattr_or<bool>(cfg_, "ip_exact_hessian", true)
                               ? model.attr("lagrangian_hessian")(
                                     x, py::cast(lmb), py::cast(nuv))
                               : hess_.attr("get_hessian")(
                                     model, x, py::cast(lmb), py::cast(nuv));

        spmat H_obj_sparse = pyconv::to_sparse(H_obj);
        auto [H, reg_info] = regularizer_->regularize(H_obj_sparse, it);

        // W assembly: H + diag(Sigma_x) + JI^T diag(Sigma_s) JI
        spmat W = H; // copy
        for (int i = 0; i < std::min<int>(W.rows(), Sg.Sigma_x.size()); ++i)
            W.coeffRef(i, i) += Sg.Sigma_x[i];

        if (mI > 0 && JI.size() && Sg.Sigma_s.size()) {
            spmat JIw = Sg.Sigma_s.asDiagonal() * JI;
            W += JI.transpose() * JIw;
        }

        // residuals
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
        const auto [alpha_aff, mu_aff, sigma] = mehrotra_affine_predictor_(
            W, r_d, (mE > 0) ? std::optional<spmat>(JE) : std::nullopt,
            (mE > 0) ? std::optional<dvec>(r_pE) : std::nullopt,
            (mI > 0) ? std::optional<spmat>(JI) : std::nullopt,
            (mI > 0) ? std::optional<dvec>(r_pI) : std::nullopt, s, lmb, zL, zU,
            B, use_shifted, tau_shift, bound_shift, mu, theta);

        // μ pre-corrector
        const double comp =
            detail::complementarity(s, lmb, mu, tau_shift, use_shifted);
        if (comp * std::max(1, mI) > 10.0 * mu)
            mu = std::min(comp * std::max(1, mI), 10.0);
        // print mu_aff
        // std::cout << "mu_aff: " << mu_aff << std::endl;
        // print sigma
        // std::cout << "sigma: " << sigma << std::endl;
        // print sigma * mu_aff
        // std::cout << "sigma * mu_aff: " << sigma * mu_aff << std::endl;
        mu = std::max(pyu::getattr_or<double>(cfg_, "ip_mu_min", 1e-12),
                      sigma * mu_aff);

        // corrector RHS: rhs_x = -r_d + inequality/bounds terms
        dvec rhs_x = -r_d;

        if (mI > 0 && JI.size() && Sg.Sigma_s.size()) {
            dvec rc_s(mI);
            for (int i = 0; i < mI; ++i) {
                const double ds = use_shifted ? (s[i] + tau_shift) : s[i];
                rc_s[i] = mu - ds * lmb[i];
            }
            dvec temp(mI);
            for (int i = 0; i < mI; ++i) {
                const double lam_safe =
                    (std::abs(lmb[i]) < EPS_POS)
                        ? ((lmb[i] >= 0) ? EPS_POS : -EPS_POS)
                        : lmb[i];
                temp[i] = rc_s[i] / lam_safe;
            }
            rhs_x.noalias() +=
                JI.transpose() * (Sg.Sigma_s.asDiagonal() * temp);
        }

        for (int i = 0; i < n; ++i)
            if (B.hasL[i])
                rhs_x[i] +=
                    (mu - (use_shifted ? (B.sL[i] + bound_shift) : B.sL[i]) *
                              zL[i]) /
                    std::max(EPS_POS,
                             (use_shifted ? (B.sL[i] + bound_shift) : B.sL[i]));
        for (int i = 0; i < n; ++i)
            if (B.hasU[i])
                rhs_x[i] -=
                    (mu - (use_shifted ? (B.sU[i] + bound_shift) : B.sU[i]) *
                              zU[i]) /
                    std::max(EPS_POS,
                             (use_shifted ? (B.sU[i] + bound_shift) : B.sU[i]));

        auto res = solve_KKT_(
            W, rhs_x, (mE > 0) ? std::optional<spmat>(JE) : std::nullopt,
            (mE > 0) ? std::optional<dvec>(cE) : std::nullopt,
            pyu::getattr_or<std::string>(cfg_, "ip_kkt_method", "hykkt"));

        dvec dx = std::move(res.dx);
        dvec dnu = (mE > 0 && res.dy.size() == mE) ? std::move(res.dy)
                                                   : dvec::Zero(mE);
        // recover ds, dλ, dz
        dvec ds, dlam;
        if (mI > 0) {
            ds = -(r_pI + JI * dx);
            dlam = dvec(mI);
            for (int i = 0; i < mI; ++i) {
                double d = use_shifted ? (s[i] + tau_shift) : s[i];
                dlam[i] = detail::sdiv(mu - d * lmb[i] - lmb[i] * ds[i], d);
            }
        }
        auto [dzL, dzU] = detail::dz_bounds_from_dx(dx, zL, zU, B, bound_shift,
                                                    use_shifted, mu, true);

        // trust region cap on direction vector family
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

        // fraction-to-boundary & LS
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

        // optional Python LS
        // auto ls_res = ls_->search(model, x, dx, (mI ? s : dvec()),
        //                           (mI ? ds : dvec()), mu, g.dot(dx), theta,
        //                           alpha_max); // reset
        // alpha = std::get<0>(ls_res);
        ls_iters = 0; // std::get<1>(ls_res);
        needs_restoration = false; //std::get<2>(ls_res);

        // early restoration (light fallback)
        const double ls_min_alpha = pyu::getattr_or<double>(
            cfg_, "ls_min_alpha",
            pyu::getattr_or<double>(cfg_, "ip_alpha_min", 1e-10));
        if (alpha <= ls_min_alpha && needs_restoration) {
            // std::cout << "IP step: light restoration" << std::endl;
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

        dvec x_new = x + alpha * dx;
        dvec s_new = (mI ? (s + alpha * ds) : s);
        dvec lmb_new = (mI ? (lmb + alpha * dlam) : lmb);
        dvec nu_new = (mE ? (nuv + alpha * dnu) : nuv);
        dvec zL_new = zL + alpha * dzL;
        dvec zU_new = zU + alpha * dzU;

        // recompute sL/sU on new x for capping
        Bounds Bn = detail::get_bounds(model, x_new);
        detail::cap_bound_duals_sigma_box(zL_new, zU_new, Bn, use_shifted,
                                          bound_shift, mu, 1e10);

        // report
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

        dvec r_d_new = g_new;
        if (mI > 0 && JI_new.size())
            r_d_new.noalias() += JI_new.transpose() * lmb_new;
        if (mE > 0 && JE_new.size())
            r_d_new.noalias() += JE_new.transpose() * nu_new;
        r_d_new -= zL_new;
        r_d_new += zU_new;

        // complementarity pieces at new point
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
                for (int i = 0; i < mI; ++i)
                    r_comp_s_new[i] = (s_new[i] + tau_shift) * lmb_new[i] - mu;
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
                for (int i = 0; i < mI; ++i)
                    r_comp_s_new[i] = s_new[i] * lmb_new[i] - mu;
            }
        }

        py::dict kkt_new;
        kkt_new["stat"] = detail::safe_inf_norm(r_d_new);
        kkt_new["ineq"] =
            (mI > 0) ? detail::safe_inf_norm((cI_new.array().max(0.0)).matrix())
                     : 0.0;
        kkt_new["eq"] = (mE > 0) ? detail::safe_inf_norm(cE_new) : 0.0;

        double comp_val = std::max(detail::safe_inf_norm(r_comp_L_new),
                                   detail::safe_inf_norm(r_comp_U_new));
        if (mI > 0 && r_comp_s_new.size())
            comp_val = std::max(comp_val, detail::safe_inf_norm(r_comp_s_new));
        kkt_new["comp"] = comp_val;

        const bool converged =
            (kkt_new["stat"].cast<double>() <= tol &&
             kkt_new["ineq"].cast<double>() <= tol &&
             kkt_new["eq"].cast<double>() <= tol &&
             kkt_new["comp"].cast<double>() <= tol && mu <= tol / 10.0);

        // update mu (unchanged logic)
        mu = update_mu_(mu, s_new, lmb_new, theta_new, kkt_new, true,
                        std::numeric_limits<double>::quiet_NaN(), sigma, mu_aff,
                        use_shifted, tau_shift);

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

        // persist state
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
    py::object cfg_, hess_, reg_, tr_, filter_, funnel_, soc_;
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
        set_if_missing("ip_kkt_method", py::str("ldl"));
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
            std::max(pyu::getattr_or<double>(cfg_, "ip_mu_init", 1e-2), 1e-12);
        const bool use_shifted =
            pyu::getattr_or<bool>(cfg_, "ip_use_shifted_barrier", true);
        // std::cout << "use_shifted: " << use_shifted << std::endl;
        const double tau_shift =
            use_shifted ? pyu::getattr_or<double>(cfg_, "ip_shift_tau", 0.1)
                        : 0.0;
        const double bound_shift =
            use_shifted ? pyu::getattr_or<double>(cfg_, "ip_shift_bounds", 0.1)
                        : 0.0;

        if (s.mI > 0) {
            s.s = dvec(s.mI);
            for (int i = 0; i < s.mI; ++i)
                s.s[i] = std::max(1.0, -cI[i] + 1e-3);
            s.lam = dvec(s.mI);
            for (int i = 0; i < s.mI; ++i) {
                const double denom =
                    (tau_shift > 0.0) ? (s.s[i] + tau_shift) : s.s[i];
                s.lam[i] = std::max(1e-8, mu0 / std::max(denom, 1e-12));
            }
        } else {
            s.s.resize(0);
            s.lam.resize(0);
        }

        s.nu = (s.mE > 0) ? dvec::Zero(s.mE) : dvec();

        // bounds initialize zL/zU
        Bounds B = detail::get_bounds(model, x);
        s.zL = dvec::Zero(x.size());
        s.zU = dvec::Zero(x.size());
        for (int i = 0; i < x.size(); ++i) {
            if (B.hasL[i])
                s.zL[i] = std::max(
                    1e-8, mu0 / std::max(B.sL[i] + bound_shift, 1e-12));
            if (B.hasU[i])
                s.zU[i] = std::max(
                    1e-8, mu0 / std::max(B.sU[i] + bound_shift, 1e-12));
        }

        s.mu = mu0;
        s.tau_shift = tau_shift;
        s.initialized = true;
        return s;
    }

    // adaptive shifts
    [[nodiscard]] double adaptive_shift_slack_(const dvec &s, const dvec &cI,
                                               int it) const {
        if (!s.size())
            return 0.0;
        double base = pyu::getattr_or<double>(cfg_, "ip_shift_tau", 1e-3);
        double min_s = s.minCoeff();
        double max_v = (cI.size() > 0) ? cI.cwiseAbs().maxCoeff() : 0.0;
        if (min_s < 1e-6 || max_v > 1e2)
            return std::min(1.0, base * (1.0 + 0.1 * it));
        if (min_s > 1e-2 && max_v < 1e-2)
            return std::max(0.0, base * (1.0 - 0.05 * it));
        return base;
    }

    [[nodiscard]] double adaptive_shift_bounds_(const dvec &x, const Bounds &B,
                                                int it) const {
        bool any =
            std::any_of(B.hasL.begin(), B.hasL.end(),
                        [](auto b) { return b; }) ||
            std::any_of(B.hasU.begin(), B.hasU.end(), [](auto b) { return b; });
        if (!any)
            return 0.0;
        double min_gap = std::numeric_limits<double>::infinity();
        for (int i = 0; i < x.size(); ++i) {
            if (B.hasL[i])
                min_gap = std::min(min_gap, x[i] - B.lb[i]);
            if (B.hasU[i])
                min_gap = std::min(min_gap, B.ub[i] - x[i]);
        }
        double b0 = pyu::getattr_or<double>(cfg_, "ip_shift_bounds", 0.0);
        if (min_gap < 1e-8)
            return std::min(1.0, std::max(b0, 1e-3) * (1 + 0.05 * it));
        if (min_gap > 1e-2)
            return std::max(0.0, b0 * 0.9);
        return b0;
    }

    // KKT solve bridge (unchanged semantics)
    [[nodiscard]]
    KKTResult solve_KKT_(const spmat &W, const dvec &rhs_x,
                         const std::optional<spmat> &JE,
                         const std::optional<dvec> &rpE,
                         std::string_view method_in) {
        // Dimensions
        const int n = W.rows();

        int mE = 0;
        std::optional<spmat> G;
        if (JE && JE->rows() > 0) {
            mE = JE->rows();
            G = *JE; // copy (or std::move if your call-site grants it)
        }

        // r1
        kkt::dvec r1 = rhs_x;

        // r2 = Zero(mE) if no rpE; otherwise r2 = -(rpE) (same sign as your py
        // version)
        std::optional<kkt::dvec> r2;
        if (mE > 0) {
            if (rpE) {
                r2 = (-(*rpE)).eval();
            } else {
                r2 = kkt::dvec::Zero(mE);
            }
        }

        // Choose method; if no equalities, force LDL
        std::string method_cpp = (method_in == "hykkt") ? "hykkt" : "ldl";
        if (mE == 0 && method_cpp == "hykkt")
            method_cpp = "ldl";

        // Default config (same as before)
        kkt::ChompConfig conf;
        conf.cg_tol = 1e-6;
        conf.cg_maxit = 200;
        conf.ip_hess_reg0 = 1e-8;
        conf.schur_dense_cutoff = 0.25;

        // Obtain strategy and factor/solve
        auto &reg = kkt::default_registry();
        auto strat = reg.get(method_cpp);

        auto [dx, dy, reusable] = strat->factor_and_solve(
            W, G, r1, r2, conf,
            /*regularizer*/ std::nullopt, kkt_cache_, /*delta*/ 0.0,
            /*gamma*/ std::nullopt,
            /*assemble_schur_if_m_small*/ true,
            /*jacobi_schur_prec*/ true);

        // Cache for potential reuse (LDL path only in this setup)
        prev_kkt_matrix_ = W;
        prev_factorization_ = (method_cpp == "ldl") ? reusable : nullptr;

        return KKTResult{std::move(dx), std::move(dy), std::move(reusable)};
    }

    // affine predictor (logic unchanged, cleaner arguments)
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

        dvec ds_aff, dlam_aff;
        const int mI = static_cast<int>(s.size());
        const int n = static_cast<int>(zL.size());

        if (mI > 0) {
            ds_aff = -(r_pI.value() + (*JI) * dx_aff);
            dlam_aff = dvec(mI);
            for (int i = 0; i < mI; ++i) {
                double d = use_shifted ? (s[i] + tau_shift) : s[i];
                dlam_aff[i] =
                    detail::sdiv(-(d * lmb[i]) - lmb[i] * ds_aff[i], d);
            }
        }

        dvec dzL_aff = dvec::Zero(n), dzU_aff = dvec::Zero(n);
        for (int i = 0; i < n; ++i) {
            if (B.hasL[i]) {
                double d = (use_shifted && bound_shift > 0.0)
                               ? (B.sL[i] + bound_shift)
                               : B.sL[i];
                dzL_aff[i] = detail::sdiv(-(d * zL[i]) - zL[i] * dx_aff[i], d);
            }
            if (B.hasU[i]) {
                double d = (use_shifted && bound_shift > 0.0)
                               ? (B.sU[i] + bound_shift)
                               : B.sU[i];
                dzU_aff[i] = detail::sdiv(-(d * zU[i]) + zU[i] * dx_aff[i], d);
            }
        }

        const double tau_pri = pyu::getattr_or<double>(
            cfg_, "ip_tau_pri", pyu::getattr_or<double>(cfg_, "ip_tau", 0.995));
        const double tau_dual = pyu::getattr_or<double>(
            cfg_, "ip_tau_dual",
            pyu::getattr_or<double>(cfg_, "ip_tau", 0.995));
        double alpha_aff = 1.0;

        if (mI > 0) {
            for (int i = 0; i < mI; ++i)
                if (ds_aff[i] < 0.0)
                    alpha_aff =
                        std::min(alpha_aff,
                                 -s[i] / std::min(ds_aff[i], -consts::EPS_DIV));
            for (int i = 0; i < mI; ++i)
                if (dlam_aff[i] < 0.0)
                    alpha_aff = std::min(
                        alpha_aff,
                        -lmb[i] / std::min(dlam_aff[i], -consts::EPS_DIV));
        }
        for (int i = 0; i < n; ++i)
            if (B.hasL[i] && dx_aff[i] < 0.0) {
                alpha_aff = std::min(alpha_aff,
                                     -(B.sL[i]) /
                                         std::min(dx_aff[i], -consts::EPS_DIV));
            }
        for (int i = 0; i < n; ++i)
            if (B.hasL[i] && dzL_aff[i] < 0.0) {
                alpha_aff =
                    std::min(alpha_aff,
                             -(zL[i]) / std::min(dzL_aff[i], -consts::EPS_DIV));
            }
        for (int i = 0; i < n; ++i) {
            const double mdx = -dx_aff[i];
            if (B.hasU[i] && mdx < 0.0)
                alpha_aff = std::min(
                    alpha_aff, -(B.sU[i]) / std::min(mdx, -consts::EPS_DIV));
        }
        for (int i = 0; i < n; ++i)
            if (B.hasU[i] && dzU_aff[i] < 0.0) {
                alpha_aff =
                    std::min(alpha_aff,
                             -(zU[i]) / std::min(dzU_aff[i], -consts::EPS_DIV));
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
        for (int i = 0; i < n; ++i)
            if (B.hasL[i]) {
                const double sL_aff = B.sL[i] + alpha_aff * dx_aff[i];
                const double zL_af = zL[i] + alpha_aff * dzL_aff[i];
                const double s_eff = (use_shifted && bound_shift > 0.0)
                                         ? (sL_aff + bound_shift)
                                         : sL_aff;
                sum_parts += s_eff * zL_af;
                ++denom_cnt;
            }
        for (int i = 0; i < n; ++i)
            if (B.hasU[i]) {
                const double sU_aff = B.sU[i] - alpha_aff * dx_aff[i];
                const double zU_af = zU[i] + alpha_aff * dzU_aff[i];
                const double s_eff = (use_shifted && bound_shift > 0.0)
                                         ? (sU_aff + bound_shift)
                                         : sU_aff;
                sum_parts += s_eff * zU_af;
                ++denom_cnt;
            }

        // print denom_cnt, sum_parts
        const double mu_aff =
            (denom_cnt > 0)
                ? std::max(mu_min, sum_parts / std::max(1, denom_cnt))
                : std::max(mu_min, mu);

        const double pwr = pyu::getattr_or<double>(cfg_, "ip_sigma_power", 3.0);

        double sigma =
            (alpha_aff > 0.9)
                ? 0.0
                : clamp(std::pow(1.0 - alpha_aff, 2) *
                            std::pow(mu_aff / std::max(mu, mu_min), pwr),
                        0.0, 1.0);

        const double theta_clip =
            pyu::getattr_or<double>(cfg_, "ip_theta_clip", 1e-2);
        if (theta > theta_clip)
            sigma = std::max(sigma, 0.5);

        return {alpha_aff, mu_aff, sigma};
    }
    // ---- error metric (same semantics, cleaner) ----
    [[nodiscard]] double compute_error_(const py::object &model, const dvec &x,
                                        const dvec &lam, const dvec &nu,
                                        const dvec &zL, const dvec &zU,
                                        double mu, const dvec &s, int mI_decl) {
        // Pull core pieces
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

        // stationarity
        dvec r_d = g;
        if (JI.size())
            r_d.noalias() += JI.transpose() * lam;
        if (JE.size())
            r_d.noalias() += JE.transpose() * nu;
        r_d -= zL;
        r_d += zU;

        // scaling (unchanged logic)
        const double s_max = pyu::getattr_or<double>(cfg_, "ip_s_max", 100.0);
        const int denom_ct = mI + mE + n;
        const double sum_mults =
            lam.lpNorm<1>() + nu.lpNorm<1>() + zL.lpNorm<1>() + zU.lpNorm<1>();
        const double s_d =
            std::max(s_max, sum_mults / std::max(1, denom_ct)) / s_max;
        const double s_c = std::max(s_max, (zL.lpNorm<1>() + zU.lpNorm<1>()) /
                                               std::max(1, n)) /
                           s_max;

        // complementarity residuals
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
                for (int i = 0; i < s.size(); ++i)
                    r_comp_s[i] = (s[i] + tau_shift) * lam[i] - mu;
            }
        } else {
            for (int i = 0; i < n; ++i) {
                r_comp_L[i] = B.hasL[i] ? (B.sL[i] * zL[i] - mu) : 0.0;
                r_comp_U[i] = B.hasU[i] ? (B.sU[i] * zU[i] - mu) : 0.0;
            }
            if (mI > 0 && s.size()) {
                r_comp_s.resize(s.size());
                for (int i = 0; i < s.size(); ++i)
                    r_comp_s[i] = s[i] * lam[i] - mu;
            }
        }

        return std::max({detail::safe_inf_norm(r_d) / s_d,
                         (mE > 0) ? detail::safe_inf_norm(cE) : 0.0,
                         (mI > 0) ? detail::safe_inf_norm(cI) : 0.0,
                         std::max(detail::safe_inf_norm(r_comp_L),
                                  detail::safe_inf_norm(r_comp_U)) /
                             s_c,
                         (r_comp_s.size() > 0)
                             ? (detail::safe_inf_norm(r_comp_s) / s_c)
                             : 0.0});
    }

    // ---- mu update (same policy as before) ----
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

        const double comp_ratio = comp / std::max(mu, 1e-12);
        const double mu_base = std::max(mu_min, sigma * mu_aff);

        double mu_new;
        if (good && comp_ratio < 0.1) {
            mu_new = mu_base *
                     std::min(0.1, std::pow(mu_aff / std::max(mu, 1e-12), 2));
        } else if (comp_ratio > 10.0 || theta > 10 * theta_tol) {
            mu_new = std::min(10.0 * mu, mu_base * 1.2);
        } else {
            mu_new = std::min(mu_base, std::pow(mu, kappa));
        }
        return std::max(mu_new, mu_min);
    }

    // ---- TR bridge (same behavior) ----
    [[nodiscard]] double tr_radius_() const { return 0.0; }
}; // class InteriorPointStepper

// ---------- pybind ----------
PYBIND11_MODULE(ip_cpp, m) {
    py::class_<IPState>(m, "IPState")
        .def(py::init<>())
        .def_readwrite("mI", &IPState::mI)
        .def_readwrite("mE", &IPState::mE)
        .def_readwrite("s", &IPState::s)
        .def_readwrite("lam", &IPState::lam)
        .def_readwrite("nu", &IPState::nu)
        .def_readwrite("zL", &IPState::zL)
        .def_readwrite("zU", &IPState::zU)
        .def_readwrite("mu", &IPState::mu)
        .def_readwrite("tau_shift", &IPState::tau_shift)
        .def_readwrite("initialized", &IPState::initialized);

    py::class_<InteriorPointStepper>(m, "InteriorPointStepper")
        .def(py::init<py::object, py::object, py::object>(), py::arg("cfg"),
             py::arg("hess"), py::arg("regularizer"))
        .def("step", &InteriorPointStepper::step, py::arg("model"),
             py::arg("x"), py::arg("lam"), py::arg("nu"), py::arg("it"),
             py::arg("ip_state") = std::optional<IPState>{});
}