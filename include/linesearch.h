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
    }

    // Returns (alpha, iters, needs_restoration)
    std::tuple<double, int, bool>
    search(py::object model_, const dvec &x, const dvec &dx, const dvec &ds,
           const dvec &s, double mu, double d_phi,
           std::optional<double> theta0_opt = std::nullopt,
           double alpha_max = 1.0) const {
        // --- base eval (single call; request only needed pieces)
        py::tuple comps(6);
        comps[0] = py::str("f");
        comps[1] = py::str("g");
        comps[2] = py::str("cE");
        comps[3] = py::str("cI");
        comps[4] = py::str("JE");
        comps[5] = py::str("JI");

        py::dict d0 =
            model_.attr("eval_all")(py::cast(x), py::arg("components") = comps)
                .cast<py::dict>();
        const double f0 = py::cast<double>(d0["f"]);
        const dvec g0 = to_vec_opt(d0["g"]);
        const py::object cE0 = dict_get(d0, "cE");
        const py::object cI0 = dict_get(d0, "cI");
        const py::object JE = dict_get(d0, "JE");
        const py::object JI = dict_get(d0, "JI");

        // quick slack checks
        if (s.size() == 0 || (s.array() <= 0.0).any()) {
            return {alpha_max, 0, true};
        }

        // φ0 and θ0
        const double barrier_eps = std::max(1e-8 * mu, 1e-16);
        const double phi0 =
            f0 - mu * (s.array()
                           .unaryExpr([&](double v) {
                               return std::log(std::max(v, barrier_eps));
                           })
                           .sum());

        double theta0 = 0.0;
        if (theta0_opt) {
            theta0 = *theta0_opt;
        } else {
            double thE = 0.0, thI = 0.0;
            if (!cE0.is_none()) {
                dvec vE = py::cast<dvec>(cE0);
                thE = vE.array().abs().sum();
            }
            if (!cI0.is_none()) {
                dvec vI = py::cast<dvec>(cI0);
                thI = (vI.array() + s.array()).abs().sum();
            }
            theta0 = thE + thI;
        }

        // descent check on φ
        if (d_phi >= -1e-12) {
            return {alpha_max, 0, true};
        }

        // fraction-to-boundary α_max
        if (ds.size() == s.size()) {
            for (Eigen::Index i = 0; i < ds.size(); ++i) {
                if (ds[i] < 0.0) {
                    const double am = (1.0 - cfg_.ip_fraction_to_boundary_tau) *
                                      s[i] / (-ds[i]);
                    if (am < alpha_max)
                        alpha_max = am;
                }
            }
        }
        if (alpha_max < 1.0e-16)
            alpha_max = 1.0e-16;

        // Funnel predictions at unit step
        double pred_df = 0.0; // max(0, -(g^T dx))
        if (g0.size() == dx.size()) {
            pred_df = -g0.dot(dx);
            if (pred_df < 0.0)
                pred_df = 0.0;
        }

        // θ linear prediction: JE@dx and JI@dx (support scipy.sparse)
        const dvec je_dx = matvec(JE, dx);
        const dvec ji_dx = matvec(JI, dx);

        double thE_lin = 0.0, thI_lin = 0.0;
        if (!cE0.is_none()) {
            dvec vE = py::cast<dvec>(cE0);
            thE_lin = (je_dx.size() ? (vE + je_dx).array().abs().sum()
                                    : vE.array().abs().sum());
        }
        if (!cI0.is_none()) {
            dvec vI = py::cast<dvec>(cI0);
            dvec rI_lin = vI + s + ds;
            if (ji_dx.size())
                rI_lin += ji_dx;
            thI_lin = rI_lin.array().abs().sum();
        }
        const double theta_pred = thE_lin + thI_lin;
        double pred_dtheta = theta0 - theta_pred;
        if (pred_dtheta < 0.0)
            pred_dtheta = 0.0;

        // line search loop
        double alpha = (alpha_max > 1.0) ? 1.0 : alpha_max;
        int it = 0;

        while (it < cfg_.ls_max_iter) {
            const dvec x_t = x + alpha * dx;
            const dvec s_t = s + alpha * ds;

            if ((s_t.array() <= 0.0).any()) {
                alpha *= cfg_.ls_backtrack;
                ++it;
                continue;
            }

            try {
                py::tuple comps_t(3);
                comps_t[0] = py::str("f");
                comps_t[1] = py::str("cE");
                comps_t[2] = py::str("cI");
                py::dict d_t =
                    model_
                        .attr("eval_all")(py::cast(x_t),
                                          py::arg("components") = comps_t)
                        .cast<py::dict>();
                const double f_t = py::cast<double>(d_t["f"]);
                if (!std::isfinite(f_t)) {
                    alpha *= cfg_.ls_backtrack;
                    ++it;
                    continue;
                }
                const double phi_t =
                    f_t -
                    mu * (s_t.array()
                              .unaryExpr([&](double v) {
                                  return std::log(std::max(v, barrier_eps));
                              })
                              .sum());
                if (!std::isfinite(phi_t)) {
                    alpha *= cfg_.ls_backtrack;
                    ++it;
                    continue;
                }

                // Armijo on φ
                if (phi_t <= phi0 + cfg_.ls_armijo_f * alpha * d_phi) {
                    const py::object cE_t = dict_get(d_t, "cE");
                    const py::object cI_t = dict_get(d_t, "cI");

                    const double thE_t =
                        cE_t.is_none()
                            ? 0.0
                            : py::cast<dvec>(cE_t).array().abs().sum();

                    double thI_t = 0.0;
                    if (!cI_t.is_none()) {
                        dvec vI_t = py::cast<dvec>(cI_t);
                        thI_t = (vI_t.array() + s_t.array()).abs().sum();
                    }
                    const double theta_t = thE_t + thI_t;

                    bool acceptable_ok = true;
                    if (funnel_) {
                        // C++ funnel: direct call (pure C++, no pyattr)
                        acceptable_ok = funnel_->is_acceptable(
                            theta0, f0, theta_t, f_t, pred_df, pred_dtheta);
                    } else if (filter_ && !filter_.is_none()) {
                        acceptable_ok = py::cast<bool>(
                            filter_.attr("is_acceptable")(theta_t, f_t));
                    }

                    if (acceptable_ok) {
                        if (funnel_) {
                            (void)funnel_->add_if_acceptable(
                                theta0, f0, theta_t, f_t, pred_df, pred_dtheta);
                        } else if (filter_ && !filter_.is_none()) {
                            (void)filter_.attr("add_if_acceptable")(theta_t,
                                                                    f_t);
                        }
                        return {alpha, it, false};
                    }
                }
            } catch (const std::exception &) {
                // robust backtrack on any evaluation failure
            }

            alpha *= cfg_.ls_backtrack;
            ++it;
        }

        const bool needs_restoration = (theta0 > cfg_.ls_theta_restoration);
        (void)cfg_.ls_min_alpha; // parity with original; logging omitted here
        return {alpha_max, it, needs_restoration};
    }

private:
    py::object cfg_obj_;
    py::object model_;
    py::object filter_;
    std::shared_ptr<Funnel> funnel_;
    LSConfig cfg_;
};
