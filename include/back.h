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
#include "model.h"
#include "regularizer.h"

namespace py = pybind11;

struct StepResult {
    dvec x;   // new primal iterate
    dvec lam; // new inequality multipliers
    dvec nu;  // new equality multipliers (empty if no JE)
};

struct KKTResult {
    dvec dx; // primal search direction
    dvec dy; // equality multipliers (empty if no JE)
    std::shared_ptr<kkt::KKTReusable> reusable; // factorization handle
};

spmat to_csr(const dmat& A, double prune_eps = 0.0) {
    dmat B = A;
    if (prune_eps > 0.0) {
        // Optionally drop tiny entries before sparsifying
        for (int i = 0; i < B.rows(); ++i)
            for (int j = 0; j < B.cols(); ++j)
                if (std::abs(B(i,j)) < prune_eps) B(i,j) = 0.0;
    }
    spmat S = B.sparseView(); // builds a sparse view
    S.makeCompressed();       // ensure CSR is compressed
    return S;
}

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
[[nodiscard]] inline Bounds get_bounds(Model* model, const dvec& x) {
    const int n = static_cast<int>(x.size());
    Bounds B;

    // Pull lb/ub from the C++ Model (fallback to ±INF if sizes mismatch)
    const dvec& lb_m = model->lb();
    const dvec& ub_m = model->ub();

    if (lb_m.size() == n) {
        B.lb = lb_m;
    } else {
        B.lb = dvec::Constant(n, -consts::INF);
    }

    if (ub_m.size() == n) {
        B.ub = ub_m;
    } else {
        B.ub = dvec::Constant(n, +consts::INF);
    }

    // Indicators and slacks
    B.hasL.assign(n, 0);
    B.hasU.assign(n, 0);
    B.sL.resize(n);
    B.sU.resize(n);

    for (int i = 0; i < n; ++i) {
        const bool hL = std::isfinite(B.lb[i]);
        const bool hU = std::isfinite(B.ub[i]);
        B.hasL[i] = static_cast<uint8_t>(hL);
        B.hasU[i] = static_cast<uint8_t>(hU);
        B.sL[i]   = hL ? clamp_min(x[i] - B.lb[i], consts::EPS_POS) : 1.0;
        B.sU[i]   = hU ? clamp_min(B.ub[i] - x[i], consts::EPS_POS) : 1.0;
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

    Model *model = nullptr; // Pointer to the Python model
    Bounds B;

    InteriorPointStepper(Model *model, py::object cfg, py::object hess)
        : model(std::move(model)), cfg_(std::move(cfg)),
          hess_(std::move(hess)) {
        load_defaults_();
        load_gondzio_defaults_();
        std::shared_ptr<Funnel> funnel = std::shared_ptr<Funnel>();
        ls_ = std::make_shared<LineSearcher>(cfg_, py::none(), funnel);
    }

    std::tuple<dvec, dvec, dvec, SolverInfo>
    step(const dvec &x, const dvec &lam, const dvec &nu, int it,
         std::optional<IPState> ip_state_opt = std::nullopt) {

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

        // Ask for the same components (C++ vector now, not a Python tuple)
        std::vector<std::string> comps = {"f", "g", "cI", "JI", "cE", "JE"};
        auto d0 = model->eval_all(x, comps);
        auto has = [&](const char *k) { return d0.find(k) != d0.end(); };

        // Small helpers
        auto getd = [&](const char *k) -> double {
            return std::get<double>(d0.at(k));
        };
        auto getv = [&](const char *k) -> const dvec & {
            return std::get<dvec>(d0.at(k));
        };

        // Pull scalars/vectors
        const double f =
            has("f") ? getd("f") : std::numeric_limits<double>::infinity();
        const dvec &g =
            has("g") ? getv("g")
                     : (*(new dvec(dvec::Zero(n)))); // or keep a local scratch
        dvec cI = (mI > 0 && has("cI")) ? getv("cI") : dvec::Zero(mI);
        dvec cE = (mE > 0 && has("cE")) ? getv("cE") : dvec::Zero(mE);

        // Pull Jacobians (your solver wants CSR). eval_all returns
        //  - spmat for "JI"/"JE" when model.use_sparse()==true
        //  - dmat otherwise. Convert to spmat if needed.
        spmat JI, JE;

        if (mI > 0 && has("JI")) {
            if (std::holds_alternative<spmat>(d0.at("JI"))) {
                JI = std::get<spmat>(d0.at("JI"));
            } else {
                const dmat &Jdense = std::get<dmat>(d0.at("JI"));
                JI = spmat(Jdense.sparseView());
                JI.makeCompressed();
            }
            if (JI.rows() != mI || JI.cols() != n) {
                throw std::runtime_error("JI dimension mismatch");
            }
        }

        if (mE > 0 && has("JE")) {
            if (std::holds_alternative<spmat>(d0.at("JE"))) {
                JE = std::get<spmat>(d0.at("JE"));
            } else {
                const dmat &Jdense = std::get<dmat>(d0.at("JE"));
                JE = spmat(Jdense.sparseView());
                JE.makeCompressed();
            }
            if (JE.rows() != mE || JE.cols() != n) {
                throw std::runtime_error("JE dimension mismatch");
            }
        }

        double theta = model->constraint_violation(x);

        // Bounds with adaptive shifts
        // Bounds B = detail::get_bounds(model, x);
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
        auto compute_error_ = [&](const dvec &x, const dvec &lmb,
                                  const dvec &nu, const dvec &zL,
                                  const dvec &zU, double mu, const dvec &s,
                                  int mI) {
            // Primal residual
            // Stationarity residual
            dvec r_d = g;
            if (JI.size())
                r_d.noalias() += JI.transpose() * lam;
            if (JE.size())
                r_d.noalias() += JE.transpose() * nu;
            r_d -= zL;
            r_d += zU;

            // Scaling factors
            const double s_max =
                pyu::getattr_or<double>(cfg_, "ip_s_max", 100.0);
            const int denom_ct = mI + mE + n;
            const double sum_mults = lam.lpNorm<1>() + nu.lpNorm<1>() +
                                     zL.lpNorm<1>() + zU.lpNorm<1>();
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
                use_shifted
                    ? pyu::getattr_or<double>(cfg_, "ip_shift_bounds", 0.0)
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
                 std::max(safe_inf_norm(r_comp_L), safe_inf_norm(r_comp_U)) /
                     s_c,
                 (r_comp_s.size() > 0) ? (safe_inf_norm(r_comp_s) / s_c)
                                       : 0.0});
        };
        auto err_0 = compute_error_(x, lmb, nuv, zL, zU, mu, s, mI);
        if (err_0 <= tol) {
            struct SolverInfo info;
            info.mode = "ip";
            info.step_norm = 0.0;
            info.accepted = true;
            info.converged = true;
            info.f = f;
            info.theta = theta;
            info.stat = safe_inf_norm(g);
            info.ineq =
                (mI > 0) ? safe_inf_norm((cI.array().max(0.0)).matrix()) : 0.0;
            info.eq = (mE > 0) ? safe_inf_norm(cE) : 0.0;
            info.comp =
                detail::complementarity(s, lmb, mu, tau_shift, use_shifted);
            info.ls_iters = 0;
            info.alpha = 0.0;
            info.rho = 0.0;
            info.tr_radius = 0.0;
            info.mu = mu;
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
        auto H_obj = model->lagrangian_hessian(x, lmb, nuv);

        // convert H_obj to sparse if needed
        // convert dmat to spmat
        spmat H_obj_sp = to_csr(H_obj);

        auto [H, reg_info] = regularizer_->regularize(H_obj_sp, it);

        // // Assemble KKT matrix: W = H + diag(Sigma_x) + JI^T diag(Sigma_s) JI
        // spmat W = H;
        // for (int i = 0; i < std::min<int>(W.rows(), Sg.Sigma_x.size()); ++i)
        // {
        //     W.coeffRef(i, i) += Sg.Sigma_x[i];
        // }

        // if (mI > 0 && JI.size() && Sg.Sigma_s.size()) {
        //     spmat JIw = Sg.Sigma_s.asDiagonal() * JI;
        //     W += JI.transpose() * JIw;
        // }

        auto build_W_efficient = [](const spmat &H, const dvec &Sigma_x,
                                    const spmat &JI, const dvec &Sigma_s) {
            const int n = H.rows();
            const int mI = Sigma_s.size();

            // Estimate non-zeros: H + diagonal + JI^T*JI structure
            const size_t est_nnz =
                H.nonZeros() + n + (mI > 0 ? 2 * JI.nonZeros() : 0);

            std::vector<Eigen::Triplet<double>> triplets;
            triplets.reserve(est_nnz);

            // Add H entries
            for (int j = 0; j < H.outerSize(); ++j) {
                for (spmat::InnerIterator it(H, j); it; ++it) {
                    triplets.emplace_back(it.row(), it.col(), it.value());
                }
            }

            // Add diagonal regularization
            for (int i = 0; i < std::min(n, static_cast<int>(Sigma_x.size()));
                 ++i) {
                if (Sigma_x[i] != 0.0) {
                    triplets.emplace_back(i, i, Sigma_x[i]);
                }
            }

            // Add JI^T * diag(Sigma_s) * JI efficiently
            if (mI > 0 && JI.size() > 0) {
                for (int j = 0; j < JI.outerSize(); ++j) {
                    for (spmat::InnerIterator it_j(JI, j); it_j; ++it_j) {
                        const int row_j = it_j.row();
                        const double val_j = it_j.value() * Sigma_s[row_j];

                        // Inner product with column j of JI
                        for (spmat::InnerIterator it_k(JI, j); it_k; ++it_k) {
                            const int row_k = it_k.row();
                            const double val_k = it_k.value() * Sigma_s[row_k];

                            triplets.emplace_back(j, j, val_j * val_k);
                        }
                    }
                }
            }

            spmat W(n, n);
            W.setFromTriplets(triplets.begin(), triplets.end());
            W.makeCompressed();
            return W;
        };
        spmat W = build_W_efficient(H, Sg.Sigma_x, JI, Sg.Sigma_s);

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

            // py::dict info;
            // info["mode"] = "ip";
            // info["step_norm"] = (x_new - x).norm();
            // info["accepted"] = true;
            // info["converged"] = false;
            // info["f"] = model.attr("eval_all")(x_new,
            // py::make_tuple("f"))["f"]
            //                 .cast<double>();
            // info["theta"] =
            //     model.attr("constraint_violation")(x_new).cast<double>();
            // info["stat"] = 0.0;
            // info["ineq"] = 0.0;
            // info["eq"] = 0.0;
            // info["comp"] = 0.0;
            // info["ls_iters"] = ls_iters;
            // info["alpha"] = 0.0;
            // info["rho"] = 0.0;
            // info["tr_radius"] = tr_radius_();
            // info["mu"] = mu;
            struct SolverInfo info;
            info.mode = "ip";
            info.step_norm = (x_new - x).norm();
            info.accepted = true;
            info.converged = false;
            info.f =
                 0.0;
            info.theta =
                model->constraint_violation(x_new);
            info.stat = 0.0;
            info.ineq = 0.0;
            info.eq = 0.0;
            info.comp = 0.0;
            info.ls_iters = ls_iters;
            info.alpha = 0.0;
            info.rho = 0.0;
            info.tr_radius = 0.0;
            info.mu = mu;
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
        // Request all components at once (no Python objects involved)
        auto dN = model->eval_all(x_new, comps);
        auto has_new = [&](const char *k) { return dN.find(k) != dN.end(); };

        // Small helpers
        auto getd_new = [&](const char *k) -> double {
            return std::get<double>(dN.at(k));
        };
        auto getv_new = [&](const char *k) -> const dvec & {
            return std::get<dvec>(dN.at(k));
        };
        // helpers

        // scalars / vectors
        double f_new = has_new("f") ? std::get<double>(dN.at("f"))
                                : std::numeric_limits<double>::infinity();

        dvec g_new = has_new("g") ? std::get<dvec>(dN.at("g")) : dvec::Zero(n);

        dvec cI_new = (mI > 0 && has_new("cI")) ? std::get<dvec>(dN.at("cI"))
                                            : dvec::Zero(mI);

        dvec cE_new = (mE > 0 && has_new("cE")) ? std::get<dvec>(dN.at("cE"))
                                            : dvec::Zero(mE);

        // Jacobians: handle both dense and sparse variants
        spmat JI_new, JE_new;

        if (mI > 0 && has("JI")) {
            const auto &v = dN.at("JI");
            if (std::holds_alternative<spmat>(v)) {
                JI_new = std::get<spmat>(v);
            } else {
                const dmat &Jdense = std::get<dmat>(v);
                JI_new = spmat(Jdense.sparseView());
                JI_new.makeCompressed();
            }
            if (JI_new.rows() != mI || JI_new.cols() != n)
                throw std::runtime_error("JI dimension mismatch");
        }

        if (mE > 0 && has("JE")) {
            const auto &v = dN.at("JE");
            if (std::holds_alternative<spmat>(v)) {
                JE_new = std::get<spmat>(v);
            } else {
                const dmat &Jdense = std::get<dmat>(v);
                JE_new = spmat(Jdense.sparseView());
                JE_new.makeCompressed();
            }
            if (JE_new.rows() != mE || JE_new.cols() != n)
                throw std::runtime_error("JE dimension mismatch");
        }

        double theta_new = model->constraint_violation(x_new);

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

        struct KKT kkt_new;
        kkt_new.stat = safe_inf_norm(r_d_new);
        kkt_new.ineq =
            (mI > 0) ? safe_inf_norm((cI_new.array().max(0.0)).matrix()) : 0.0;
        kkt_new.eq = (mE > 0) ? safe_inf_norm(cE_new) : 0.0;
        double comp_val_new =
            std::max(safe_inf_norm(r_comp_L_new), safe_inf_norm(r_comp_U_new));
        if (mI > 0 && r_comp_s_new.size()) {
            comp_val_new = std::max(comp_val_new, safe_inf_norm(r_comp_s_new));
        }
        kkt_new.comp = comp_val_new;

        const bool converged =
            (kkt_new.stat <= tol && kkt_new.ineq <= tol && kkt_new.eq <= tol &&
             kkt_new.comp <= tol && mu <= tol / 10.0);

        // Update barrier parameter
        mu = update_mu_(mu, s_new, lmb_new, theta_new, kkt_new, true,
                        std::numeric_limits<double>::quiet_NaN(), sigma, mu_aff,
                        use_shifted, tau_shift);

        struct SolverInfo info;
        info.mode = "ip";
        info.step_norm = (x_new - x).norm();
        info.accepted = true;
        info.converged = converged;
        info.f = f_new;
        info.theta = theta_new;
        info.stat = kkt_new.stat;
        info.ineq = kkt_new.ineq;
        info.eq = kkt_new.eq;
        info.comp = kkt_new.comp;
        info.ls_iters = ls_iters;
        info.alpha = alpha;
        info.rho = 0.0;
        info.tr_radius = 0.0;
        info.mu = mu;
        info.shifted_barrier = use_shifted;
        info.tau_shift = tau_shift;
        info.bound_shift = bound_shift;

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
    std::shared_ptr<kkt::KKTReusable> cached_kkt_solver_{};

    spmat cached_kkt_matrix_;
    bool kkt_factorization_valid_ = false;

    // Add matrix comparison helper
    bool matrices_equal(const spmat &A, const spmat &B, double tol = 1e-14) {
        if (A.rows() != B.rows() || A.cols() != B.cols())
            return false;
        return (A - B).norm() < tol;
    }

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
        set_if_missing("ip_shift_bounds", py::float_(1e-3));
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
                       py::int_(pyu::getattr_or<int>(cfg_, "ip_ls_max", 5)));
        set_if_missing("ls_min_alpha", py::float_(pyu::getattr_or<double>(
                                           cfg_, "ip_alpha_min", 1e-10)));
    }
    [[nodiscard]] IPState state_from_model_(Model *model, const dvec &x) {
        IPState s{};
        s.mI = model->m_ineq();
        s.mE = model->m_eq();

        // Ask the Eigen-native eval_all for cI/cE
        const std::vector<std::string> comps = {"cI", "cE"};
        auto d = model->eval_all(x, comps);

        auto has = [&](const char *k) { return d.find(k) != d.end(); };
        auto get_vec = [&](const char *k) -> const dvec & {
            return std::get<dvec>(d.at(k));
        };

        dvec cI = (s.mI > 0 && has("cI")) ? get_vec("cI") : dvec::Zero(s.mI);
        // (cE not needed below, but available as:)
        // dvec cE = (s.mE > 0 && has("cE")) ? get_vec("cE") : dvec::Zero(s.mE);

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
                const double denom =
                    (tau_shift > 0.0) ? (s.s[i] + tau_shift) : s.s[i];
                s.lam[i] = clamp_min(mu0 / clamp_min(denom, 1e-12), 1e-8);
            }
        } else {
            s.s.resize(0);
            s.lam.resize(0);
        }

        s.nu = (s.mE > 0) ? dvec::Zero(s.mE) : dvec();

        // Initialize bound duals
        B = detail::get_bounds(model, x);
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

        const bool can_reuse = kkt_factorization_valid_ && cached_kkt_solver_ &&
                               matrices_equal(W, cached_kkt_matrix_);

        if (can_reuse) {
            // Reuse existing factorization
            std::optional<dvec> r2 =
                rpE ? std::optional<dvec>(-(*rpE)) : std::nullopt;
            auto [dx, dy] = cached_kkt_solver_->solve(rhs_x, r2, 1e-8, 200);

            return KKTResult{std::move(dx), std::move(dy), cached_kkt_solver_};
        } else {
            kkt::ChompConfig conf;
            conf.cg_tol = 1e-6;
            conf.cg_maxit = 200;
            conf.ip_hess_reg0 = 1e-8;
            conf.schur_dense_cutoff = 0.25;

            auto &reg = kkt::default_registry();
            auto strat = reg.get(method_cpp);

            // Need fresh factorization
            auto [dx, dy, reusable] = strat->factor_and_solve(
                W, G, r1, r2, conf, std::nullopt, kkt_cache_, 0.0, std::nullopt,
                true, true);

            // Cache the results
            cached_kkt_matrix_ = W;
            cached_kkt_solver_ = reusable;
            kkt_factorization_valid_ = true;

            return KKTResult{std::move(dx), std::move(dy), std::move(reusable)};
        }

        // auto [dx, dy, reusable] =
        //     strat->factor_and_solve(W, G, r1, r2, conf, std::nullopt,
        //                             kkt_cache_, 0.0, std::nullopt, true,
        //                             true);

        // prev_kkt_matrix_ = W;
        // prev_factorization_ = (method_cpp == "ldl") ? reusable : nullptr;

        // return KKTResult{std::move(dx), std::move(dy), std::move(reusable)};
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

    [[nodiscard]] double update_mu_(double mu, const dvec &s, const dvec &lam,
                                    double theta, KKT &kkt, bool accepted,
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
             kkt.stat <= pyu::getattr_or<double>(cfg_, "tol_stat", 1e-6) &&
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


// line_searcher_pybind.cc
#include <Eigen/Core>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

#include "funnel.h" // provides FunnelConfig and Funnel
#include "model.h"
namespace py = pybind11;
using dvec = Eigen::VectorXd;

// ---------------- config mirror (lightweight) ----------------
struct LSConfig {
    double ls_backtrack{0.5};
    double ls_armijo_f{1e-4};
    int ls_max_iter{20};
    double ls_min_alpha{1e-12};
    double ls_wolfe_c{0.9}; // not used here, kept for parity
    double ip_fraction_to_boundary_tau{0.995};
    double ls_theta_restoration{1e3};

    // SOC tuning
    int max_soc{4};
    double kappa_soc_min{0.1};  // Minimum kappa_soc for adaptive strategy
    double kappa_soc_max{0.99}; // Maximum kappa_soc for adaptive strategy
    double kappa_soc_base{0.5}; // Base kappa_soc when theta_t/theta0 is large

    double soc_active_tol{1e-8}; // when to consider a constraint active
    double soc_gamma{0.1};      // fraction of margin to use as gamma
};

// ---------------- small helpers ----------------
static inline double getattr_or_double(const py::object &obj, const char *name,
                                       double fallback) {
    if (!obj || obj.is_none())
        return fallback;
    if (py::hasattr(obj, name))
        return py::cast<double>(obj.attr(name));
    return fallback;
}

static inline int getattr_or_int(const py::object &obj, const char *name,
                                 int fallback) {
    if (!obj || obj.is_none())
        return fallback;
    if (py::hasattr(obj, name))
        return py::cast<int>(obj.attr(name));
    return fallback;
}

static inline bool getattr_or_bool(const py::object &obj, const char *name,
                                   bool fallback) {
    if (!obj || obj.is_none())
        return fallback;
    if (py::hasattr(obj, name))
        return py::cast<bool>(obj.attr(name));
    return fallback;
}

static inline py::object dict_get(const py::dict &d, const char *k) {
    if (d.contains(k))
        return d[py::str(k)];
    return py::none();
}

static inline dvec to_vec_opt(const py::object &o) {
    if (!o || o.is_none())
        return dvec();
    return py::cast<dvec>(o);
}

static inline dvec matvec(const py::object &M, const dvec &v) {
    if (!M || M.is_none())
        return dvec();
    if (py::hasattr(M, "dot")) { // scipy sparse prefers .dot
        py::object mv = M.attr("dot")(py::cast(v));
        return py::cast<dvec>(mv);
    }
    // dense @
    py::object mv = M.attr("__matmul__")(py::cast(v));
    return py::cast<dvec>(mv);
}

// ---------------- line searcher (holds Python model, optional Python filter,
// C++ funnel) ----------------
class LineSearcher {
public:
    LineSearcher(py::object cfg, py::object filter = py::none(),
                 std::shared_ptr<Funnel> funnel = nullptr)
        : cfg_obj_(std::move(cfg)), filter_(std::move(filter)),
          funnel_(std::move(funnel)) {
        // sanitize/load cfg once (mirror python guards)
        cfg_.ls_backtrack = std::clamp(
            getattr_or_double(cfg_obj_, "ls_backtrack", 0.5), 1e-4, 0.99);
        cfg_.ls_armijo_f =
            std::max(1e-12, getattr_or_double(cfg_obj_, "ls_armijo_f", 1e-4));
        cfg_.ls_max_iter =
            std::max(1, getattr_or_int(cfg_obj_, "ls_max_iter", 20));
        cfg_.ls_min_alpha =
            std::max(0.0, getattr_or_double(cfg_obj_, "ls_min_alpha", 1e-12));
        (void)getattr_or_bool(cfg_obj_, "ls_use_wolfe", false);
        cfg_.ls_wolfe_c =
            std::clamp(getattr_or_double(cfg_obj_, "ls_wolfe_c", 0.9),
                       cfg_.ls_armijo_f, 0.999);
        cfg_.ip_fraction_to_boundary_tau =
            getattr_or_double(cfg_obj_, "ip_fraction_to_boundary_tau", 0.995);
        cfg_.ls_theta_restoration =
            getattr_or_double(cfg_obj_, "ls_theta_restoration", 1e3);

        cfg_.max_soc = std::max(0, getattr_or_int(cfg_obj_, "max_soc", 4));
        cfg_.kappa_soc_min = std::clamp(
            getattr_or_double(cfg_obj_, "kappa_soc_min", 0.1), 0.01, 0.5);
        cfg_.kappa_soc_max = std::clamp(
            getattr_or_double(cfg_obj_, "kappa_soc_max", 0.99), 0.5, 0.99);
        cfg_.kappa_soc_base =
            std::clamp(getattr_or_double(cfg_obj_, "kappa_soc_base", 0.5),
                       cfg_.kappa_soc_min, cfg_.kappa_soc_max);
    }

    // Returns (alpha, iters, needs_restoration, dx_cor, ds_cor)
    // If dx_cor is empty, use ori
    // ginal dx, ds with alpha; else use alpha with
    // dx_cor, ds_cor
 // Eigen-native line search using Model::eval_all (no py::dict).
// Assumes members: mI, mE, cfg_, funnel_ (C++ ptr or nullptr), filter_ (py::object or None),
// and a helper matvec(spmat, dvec) that returns zero-sized on empty matrices.
// If you don't have matvec helpers, a simple lambda is included below.

std::tuple<double, int, bool, dvec, dvec>
search(Model* model_,
       const dvec& x,
       const dvec& dx,
       const dvec& ds,
       const dvec& s,
       double mu,
       double d_phi,
       std::optional<double> theta0_opt = std::nullopt,
       double alpha_max = 1.0) const
{
    const int n  = static_cast<int>(x.size());
    const int mI = model_->m_ineq();
    const int mE = model_->m_eq();

    // --- base eval (single call; request only needed pieces)
    std::vector<std::string> comps0 = {"f","g","cE","cI","JE","JI"};
    auto d0 = model_->eval_all(x, comps0);

    auto has = [&](const char* k){ return d0.find(k) != d0.end(); };

    // Scalars / vectors
    const double f0 = has("f") ? std::get<double>(d0.at("f"))
                               : std::numeric_limits<double>::infinity();

    const dvec g0   = has("g")  ? std::get<dvec>(d0.at("g"))
                                : dvec::Zero(n);

    const dvec cE0  = (mE > 0 && has("cE")) ? std::get<dvec>(d0.at("cE"))
                                            : dvec::Zero(mE);
    const dvec cI0  = (mI > 0 && has("cI")) ? std::get<dvec>(d0.at("cI"))
                                            : dvec::Zero(mI);

    // Jacobians (accept dense or sparse from eval_all; convert to CSR if needed)
    spmat JE, JI;
    if (mE > 0 && has("JE")) {
        const auto& v = d0.at("JE");
        if (std::holds_alternative<spmat>(v)) {
            JE = std::get<spmat>(v);
        } else {
            const dmat& JEd = std::get<dmat>(v);
            JE = spmat(JEd.sparseView()); JE.makeCompressed();
        }
    }
    if (mI > 0 && has("JI")) {
        const auto& v = d0.at("JI");
        if (std::holds_alternative<spmat>(v)) {
            JI = std::get<spmat>(v);
        } else {
            const dmat& JId = std::get<dmat>(v);
            JI = spmat(JId.sparseView()); JI.makeCompressed();
        }
    }

    // quick slack checks
    if (s.size() == 0 || (s.array() <= 0.0).any()) {
        return {alpha_max, 0, true, dvec(), dvec()};
    }

    // φ0 and θ0
    const double barrier_eps = std::max(1e-8 * mu, 1e-16);
    const double phi0 = f0 - mu * (s.array().unaryExpr(
        [&](double v){ return std::log(std::max(v, barrier_eps)); }).sum());

    double theta0 = 0.0;
    if (theta0_opt) {
        theta0 = *theta0_opt;
    } else {
        const double thE = (mE ? cE0.array().abs().sum() : 0.0);
        const double thI = (mI ? (cI0.array() + s.array()).abs().sum() : 0.0);
        theta0 = thE + thI;
    }

    // descent check on φ (tolerant to tiny FP noise)
    if (d_phi >= -1e-12) {
        return {alpha_max, 0, true, dvec(), dvec()};
    }

    // fraction-to-boundary α_max (original direction)
    if (ds.size() == s.size()) {
        for (Eigen::Index i = 0; i < ds.size(); ++i) {
            if (ds[i] < 0.0) {
                const double am = (1.0 - cfg_.ip_fraction_to_boundary_tau) * s[i] / (-ds[i]);
                if (am < alpha_max) alpha_max = am;
            }
        }
    }
    if (alpha_max < 1.0e-16) alpha_max = 1.0e-16;

    // Funnel predictions at unit step
    double pred_df = 0.0; // max(0, -(g^T dx))
    if (g0.size() == dx.size()) {
        pred_df = -g0.dot(dx);
        if (pred_df < 0.0) pred_df = 0.0;
    }

    // matvec helpers
    auto matvec_sp = [](const spmat& A, const dvec& v)->dvec {
        if (A.rows()==0 || A.cols()==0) return dvec();
        return A * v;
    };

    // θ linear prediction: JE@dx and JI@dx
    const dvec je_dx = matvec_sp(JE, dx);
    const dvec ji_dx = matvec_sp(JI, dx);

    double thE_lin = 0.0, thI_lin = 0.0;
    if (mE) {
        thE_lin = (je_dx.size() ? (cE0 + je_dx).array().abs().sum()
                                : cE0.array().abs().sum());
    }
    if (mI) {
        dvec rI_lin = cI0 + s + ds;
        if (ji_dx.size()) rI_lin += ji_dx;
        thI_lin = rI_lin.array().abs().sum();
    }
    const double theta_pred = thE_lin + thI_lin;
    double pred_dtheta = theta0 - theta_pred;
    if (pred_dtheta < 0.0) pred_dtheta = 0.0;

    // ---- line search loop ----
    double alpha = (alpha_max > 1.0) ? 1.0 : alpha_max;
    int it = 0;
    double theta_t = 0.0;

    while (it < cfg_.ls_max_iter) {
        const dvec x_t = x + alpha * dx;
        const dvec s_t = s + alpha * ds;

        if ((s_t.array() <= 0.0).any()) {
            alpha *= cfg_.ls_backtrack;
            ++it;
            continue;
        }

        try {
            std::vector<std::string> comps_t = {"f","cE","cI"};
            auto d_t = model_->eval_all(x_t, comps_t);

            const double f_t = std::get<double>(d_t.at("f"));
            if (!std::isfinite(f_t)) {
                alpha *= cfg_.ls_backtrack; ++it; continue;
            }

            const double phi_t = f_t - mu * (s_t.array().unaryExpr(
                [&](double v){ return std::log(std::max(v, barrier_eps)); }).sum());
            if (!std::isfinite(phi_t)) {
                alpha *= cfg_.ls_backtrack; ++it; continue;
            }

            // Armijo on φ for original direction
            if (phi_t <= phi0 + cfg_.ls_armijo_f * alpha * d_phi) {
                const dvec cE_t = (mE && d_t.find("cE")!=d_t.end())
                                  ? std::get<dvec>(d_t.at("cE")) : dvec::Zero(mE);
                const dvec cI_t = (mI && d_t.find("cI")!=d_t.end())
                                  ? std::get<dvec>(d_t.at("cI")) : dvec::Zero(mI);

                const double thE_t = (mE ? cE_t.array().abs().sum() : 0.0);
                const double thI_t = (mI ? (cI_t.array() + s_t.array()).abs().sum() : 0.0);
                theta_t = thE_t + thI_t;

                bool acceptable_ok = true;
                if (funnel_) {
                    acceptable_ok = funnel_->is_acceptable(theta0, f0, theta_t, f_t, pred_df, pred_dtheta);
                } else if (filter_ && !filter_.is_none()) {
                    acceptable_ok = py::cast<bool>(filter_.attr("is_acceptable")(theta_t, f_t));
                }

                if (acceptable_ok) {
                    if (funnel_) {
                        (void)funnel_->add_if_acceptable(theta0, f0, theta_t, f_t, pred_df, pred_dtheta);
                    } else if (filter_ && !filter_.is_none()) {
                        (void)filter_.attr("add_if_acceptable")(theta_t, f_t);
                    }
                    return {alpha, it, false, dvec(), dvec()};
                }

                // stash for SOC base below
            }
        } catch (...) {
            // robust backtrack on any evaluation failure
            alpha *= cfg_.ls_backtrack;
            ++it;
            continue;
        }

        // ---- Second-Order Correction (SOC) ----
        {
            dvec x_t_current = x + alpha * dx;
            dvec s_t_current = s + alpha * ds;

            // Evaluate cE/cI at SOC base (use cached x inside model)
            std::vector<std::string> comps_base = {"cE","cI","g"};
            auto d_base = model_->eval_all(x_t_current, comps_base);

            const dvec cE_base = (mE && d_base.find("cE")!=d_base.end())
                                 ? std::get<dvec>(d_base.at("cE")) : dvec::Zero(mE);
            const dvec cI_base = (mI && d_base.find("cI")!=d_base.end())
                                 ? std::get<dvec>(d_base.at("cI")) : dvec::Zero(mI);
            dvec g_t_current    = (d_base.find("g")!=d_base.end())
                                 ? std::get<dvec>(d_base.at("g")) : g0;

            // Compute theta_t at base if we haven't yet
            if (theta_t == 0.0) {
                const double thE_t = (mE ? cE_base.array().abs().sum() : 0.0);
                const double thI_t = (mI ? (cI_base.array() + s_t_current.array()).abs().sum() : 0.0);
                theta_t = thE_t + thI_t;
            }

            double theta_last = theta_t;
            int soc_count = 0;

            while (soc_count < cfg_.max_soc) {
                ++soc_count;

                // Adaptive kappa_soc
                double kappa_soc = cfg_.kappa_soc_base;
                if (theta0 > 1e-8) {
                    const double theta_ratio = theta_last / theta0;
                    if (theta_ratio > 10.0) {
                        kappa_soc = cfg_.kappa_soc_min; // stricter when far off
                    } else if (theta_ratio > 1.0) {
                        kappa_soc = cfg_.kappa_soc_min +
                                    (cfg_.kappa_soc_max - cfg_.kappa_soc_min) *
                                    (1.0 - (theta_ratio - 1.0) / 9.0);
                    } else {
                        kappa_soc = cfg_.kappa_soc_max; // near feasible
                    }
                }
                kappa_soc = std::min(kappa_soc + 0.1 * (soc_count - 1), cfg_.kappa_soc_max);

                // Residuals at SOC base
                dvec rE = (mE ? cE_base : dvec());
                dvec rI = (mI ? (cI_base + s_t_current) : dvec());

                // Ask model for SOC step (note: Model::compute_soc_step takes optionals of py objects)
                std::pair<dvec,dvec> soc;
                try {
                    soc = model_->compute_soc_step(
                        (mE ? std::optional<py::object>(py::cast(rE)) : std::nullopt),
                        (mI ? std::optional<py::object>(py::cast(rI)) : std::nullopt),
                        /*mu*/ mu,
                        /*active_tol*/ cfg_.soc_active_tol,
                        /*w_eq*/       1.0,
                        /*w_ineq*/     1.0,
                        /*gamma*/      cfg_.soc_gamma);
                } catch (...) {
                    break; // cannot form SOC
                }

                dvec dx_cor = soc.first;
                dvec ds_cor = soc.second;
                if (dx_cor.size() == 0 || ds_cor.size() == 0) break;

                // Fraction-to-boundary from SOC base
                double alpha_soc = 1.0;
                for (Eigen::Index i = 0; i < ds_cor.size(); ++i) {
                    if (ds_cor[i] < 0.0) {
                        const double am = (1.0 - cfg_.ip_fraction_to_boundary_tau) *
                                          s_t_current[i] / (-ds_cor[i]);
                        if (am < alpha_soc) alpha_soc = am;
                    }
                }
                alpha_soc = std::max(alpha_soc, cfg_.ls_min_alpha);

                // Trial point for SOC
                dvec x_t_soc = x_t_current + alpha_soc * dx_cor;
                dvec s_t_soc = s_t_current + alpha_soc * ds_cor;
                if ((s_t_soc.array() <= 0.0).any()) break;

                // Eval at SOC trial
                std::vector<std::string> comps_soc = {"f","cE","cI"};
                auto d_soc = model_->eval_all(x_t_soc, comps_soc);

                const double f_t_soc = std::get<double>(d_soc.at("f"));
                if (!std::isfinite(f_t_soc)) continue;

                const double phi_t_soc = f_t_soc - mu * (s_t_soc.array().unaryExpr(
                    [&](double v){ return std::log(std::max(v, barrier_eps)); }).sum());
                if (!std::isfinite(phi_t_soc)) continue;

                const dvec cE_soc = (mE && d_soc.find("cE")!=d_soc.end())
                                    ? std::get<dvec>(d_soc.at("cE")) : dvec::Zero(mE);
                const dvec cI_soc = (mI && d_soc.find("cI")!=d_soc.end())
                                    ? std::get<dvec>(d_soc.at("cI")) : dvec::Zero(mI);

                const double thE_t_soc = (mE ? cE_soc.array().abs().sum() : 0.0);
                const double thI_t_soc = (mI ? (cI_soc.array() + s_t_soc.array()).abs().sum() : 0.0);
                const double theta_t_soc = thE_t_soc + thI_t_soc;

                // Require feasibility improvement
                if (theta_t_soc >= kappa_soc * theta_last) break;

                // Update base for potential further SOC iters
                theta_last   = theta_t_soc;
                x_t_current  = x_t_soc;
                s_t_current  = s_t_soc;
                // Update residuals at the new base
                // (cE_base/cI_base not strictly needed beyond theta_t updates)

                // Armijo on φ using corrected direction & α_soc
                double dphi_cor = 0.0;
                if (g_t_current.size() == dx_cor.size())
                    dphi_cor = g_t_current.dot(dx_cor);
                dphi_cor -= mu * (ds_cor.array() / s_t_current.array()).sum();

                if (phi_t_soc > phi0 + cfg_.ls_armijo_f * alpha_soc * dphi_cor)
                    continue;

                // Acceptability (funnel/filter)
                bool acceptable_ok_soc = true;
                if (funnel_) {
                    acceptable_ok_soc = funnel_->is_acceptable(
                        theta0, f0, theta_t_soc, f_t_soc, pred_df, pred_dtheta);
                } else if (filter_ && !filter_.is_none()) {
                    acceptable_ok_soc = py::cast<bool>(
                        filter_.attr("is_acceptable")(theta_t_soc, f_t_soc));
                }

                if (acceptable_ok_soc) {
                    if (funnel_) {
                        (void)funnel_->add_if_acceptable(
                            theta0, f0, theta_t_soc, f_t_soc, pred_df, pred_dtheta);
                    } else if (filter_ && !filter_.is_none()) {
                        (void)filter_.attr("add_if_acceptable")(theta_t_soc, f_t_soc);
                    }
                    // Return the corrected direction and its step length
                    return {alpha_soc, it + soc_count, false, dx_cor, ds_cor};
                }
            } // SOC loop
        } // SOC block

        // Backtrack and retry
        alpha *= cfg_.ls_backtrack;
        ++it;
    }

    const bool needs_restoration = (theta0 > cfg_.ls_theta_restoration);
    return {alpha_max, it, needs_restoration, dvec(), dvec()};
}


private:
    py::object cfg_obj_;
    py::object filter_;
    std::shared_ptr<Funnel> funnel_;
    LSConfig cfg_;
};
