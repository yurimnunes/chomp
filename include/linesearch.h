// line_searcher_nb.cc
// nanobind + Eigen port of line_searcher_pybind.cc
//
// Requires:
//   - nanobind (with Eigen bridge): <nanobind/nanobind.h>, <nanobind/eigen/dense.h>
//   - Eigen
//   - Your Funnel classes in "funnel.h"

#include "funnel.h" // provides FunnelConfig and Funnel
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;
using dvec = Eigen::VectorXd;

// ---------------- config mirror (lightweight) ----------------
struct LSConfig {
    double ls_backtrack{0.5};
    double ls_armijo_f{1e-4};
    int    ls_max_iter{20};
    double ls_min_alpha{1e-12};
    double ls_wolfe_c{0.9}; // not used here, kept for parity
    double ip_fraction_to_boundary_tau{0.995};
    double ls_theta_restoration{1e3};
    int    max_soc{4};
    double kappa_soc_min{0.1};   // Minimum kappa_soc for adaptive strategy
    double kappa_soc_max{0.99};  // Maximum kappa_soc for adaptive strategy
    double kappa_soc_base{0.5};  // Base kappa_soc when theta_t/theta0 is large
};

// ---------------- small helpers ----------------
static inline double getattr_or_double(const nb::object &obj, const char *name,
                                       double fallback) {
    if (!obj || obj.is_none())
        return fallback;
    if (nb::hasattr(obj, name))
        return nb::cast<double>(obj.attr(name));
    return fallback;
}

static inline int getattr_or_int(const nb::object &obj, const char *name,
                                 int fallback) {
    if (!obj || obj.is_none())
        return fallback;
    if (nb::hasattr(obj, name))
        return nb::cast<int>(obj.attr(name));
    return fallback;
}

static inline bool getattr_or_bool(const nb::object &obj, const char *name,
                                   bool fallback) {
    if (!obj || obj.is_none())
        return fallback;
    if (nb::hasattr(obj, name))
        return nb::cast<bool>(obj.attr(name));
    return fallback;
}

static inline nb::object dict_get(const nb::dict &d, const char *k) {
    if (d.contains(k))
        return d[nb::str(k)];
    return nb::none();
}

static inline dvec to_vec_opt(const nb::object &o) {
    if (!o || o.is_none())
        return dvec();
    return nb::cast<dvec>(o);
}

static inline dvec matvec(const nb::object &M, const dvec &v) {
    if (!M || M.is_none())
        return dvec();
    if (nb::hasattr(M, "dot")) { // scipy.sparse prefers .dot
        nb::object mv = M.attr("dot")(v);
        return nb::cast<dvec>(mv);
    }
    // dense @
    nb::object mv = M.attr("__matmul__")(v);
    return nb::cast<dvec>(mv);
}

// ---------------- line searcher (holds Python filter, C++ funnel) ----------------
class LineSearcher {
public:
    LineSearcher(nb::object cfg, nb::object filter = nb::none(),
                 std::shared_ptr<Funnel> funnel = nullptr)
        : cfg_obj_(std::move(cfg)),
          filter_(std::move(filter)),
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
    // If dx_cor is empty, use original dx, ds with alpha; else use alpha with
    // dx_cor, ds_cor
    std::tuple<double, int, bool, dvec, dvec>
    search(nb::object model, const dvec &x, const dvec &dx, const dvec &ds,
           const dvec &s, double mu, double d_phi,
           std::optional<double> theta0_opt = std::nullopt,
           double alpha_max = 1.0) const {
        // --- base eval (single call; request only needed pieces)
        nb::list comps;
        comps.append(nb::str("f"));
        comps.append(nb::str("g"));
        comps.append(nb::str("cE"));
        comps.append(nb::str("cI"));
        comps.append(nb::str("JE"));
        comps.append(nb::str("JI"));

        nb::dict d0 = nb::cast<nb::dict>(
            model.attr("eval_all")(x, nb::arg("components") = comps));
        const double f0 = nb::cast<double>(d0["f"]);
        const dvec g0   = to_vec_opt(d0["g"]);
        const nb::object cE0 = dict_get(d0, "cE");
        const nb::object cI0 = dict_get(d0, "cI");
        const nb::object JE  = dict_get(d0, "JE");
        const nb::object JI  = dict_get(d0, "JI");

        // quick slack checks
        if (s.size() == 0 || (s.array() <= 0.0).any()) {
            return {alpha_max, 0, true, dvec(), dvec()};
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
                dvec vE = nb::cast<dvec>(cE0);
                thE = vE.array().abs().sum();
            }
            if (!cI0.is_none()) {
                dvec vI = nb::cast<dvec>(cI0);
                thI = (vI.array() + s.array()).abs().sum();
            }
            theta0 = thE + thI;
        }

        // descent check on φ
        if (d_phi >= -1e-12) {
            return {alpha_max, 0, true, dvec(), dvec()};
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
            dvec vE = nb::cast<dvec>(cE0);
            thE_lin = (je_dx.size() ? (vE + je_dx).array().abs().sum()
                                    : vE.array().abs().sum());
        }
        if (!cI0.is_none()) {
            dvec vI = nb::cast<dvec>(cI0);
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
        double theta_t = 0.0; // Declare theta_t outside try block
        nb::dict d_t;         // Declare d_t outside try block
        while (it < cfg_.ls_max_iter) {
            const dvec x_t = x + alpha * dx;
            const dvec s_t = s + alpha * ds;
            if ((s_t.array() <= 0.0).any()) {
                alpha *= cfg_.ls_backtrack;
                ++it;
                continue;
            }
            try {
                nb::list comps_t;
                comps_t.append(nb::str("f"));
                comps_t.append(nb::str("cE"));
                comps_t.append(nb::str("cI"));
                d_t = nb::cast<nb::dict>(
                    model.attr("eval_all")(x_t, nb::arg("components") = comps_t));
                const double f_t = nb::cast<double>(d_t["f"]);
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
                    const nb::object cE_t = dict_get(d_t, "cE");
                    const nb::object cI_t = dict_get(d_t, "cI");
                    const double thE_t =
                        cE_t.is_none()
                            ? 0.0
                            : nb::cast<dvec>(cE_t).array().abs().sum();
                    double thI_t = 0.0;
                    if (!cI_t.is_none()) {
                        dvec vI_t = nb::cast<dvec>(cI_t);
                        thI_t = (vI_t.array() + s_t.array()).abs().sum();
                    }
                    theta_t = thE_t + thI_t; // Update theta_t
                    bool acceptable_ok = true;
                    if (funnel_) {
                        acceptable_ok = funnel_->is_acceptable(
                            theta0, f0, theta_t, f_t, pred_df, pred_dtheta);
                    } else if (filter_ && !filter_.is_none()) {
                        acceptable_ok = nb::cast<bool>(
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
                        return {alpha, it, false, dvec(), dvec()};
                    }
                }
            } catch (const std::exception &) {
                // robust backtrack on any evaluation failure
                alpha *= cfg_.ls_backtrack;
                ++it;
                continue;
            }
            // Try second-order corrections if conditions met
            if (it == 0 && theta_t >= theta0) {
                double theta_last = theta_t;
                nb::object cE_t_current = dict_get(d_t, "cE");
                nb::object cI_t_current = dict_get(d_t, "cI");
                dvec s_t_current = s_t;
                int soc_count = 0;
                while (soc_count < cfg_.max_soc) {
                    ++soc_count;
                    // Compute adaptive kappa_soc
                    double kappa_soc = cfg_.kappa_soc_base;
                    if (theta0 > 1e-8) { // Avoid division by near-zero
                        const double theta_ratio = theta_t / theta0;
                        if (theta_ratio > 10.0) {
                            kappa_soc = cfg_.kappa_soc_min; // strict reduction
                        } else if (theta_ratio > 1.0) {
                            // Linear interpolation between min and max
                            kappa_soc =
                                cfg_.kappa_soc_min +
                                (cfg_.kappa_soc_max - cfg_.kappa_soc_min) *
                                    (1.0 - (theta_ratio - 1.0) / 9.0);
                        } else {
                            kappa_soc = cfg_.kappa_soc_max; // loose reduction
                        }
                    }
                    // Relax kappa_soc for later SOC iterations
                    kappa_soc = std::min(kappa_soc + 0.1 * (soc_count - 1),
                                         cfg_.kappa_soc_max);
                    // Compute c_soc
                    dvec c_soc_E = dvec();
                    if (!cE0.is_none()) {
                        c_soc_E = alpha * nb::cast<dvec>(cE0) +
                                  nb::cast<dvec>(cE_t_current);
                    }
                    dvec c_soc_I = dvec();
                    if (!cI0.is_none()) {
                        dvec ci0_plus_s = nb::cast<dvec>(cI0) + s;
                        dvec ci_t_plus_s_t =
                            nb::cast<dvec>(cI_t_current) + s_t_current;
                        c_soc_I = alpha * ci0_plus_s + ci_t_plus_s_t;
                    }
                    // Call model.compute_soc_step(c_soc_E, c_soc_I, mu)
                    nb::tuple soc_res;
                    try {
                        soc_res = nb::cast<nb::tuple>(
                            model.attr("compute_soc_step")(c_soc_E, c_soc_I, mu));
                    } catch (const std::exception &) {
                        break;
                    }
                    dvec dx_cor = to_vec_opt(soc_res[0]);
                    dvec ds_cor = to_vec_opt(soc_res[1]);
                    if (dx_cor.size() == 0 || ds_cor.size() == 0)
                        break;
                    // Fraction-to-boundary for corrected direction
                    double alpha_soc = 1.0;
                    for (Eigen::Index i = 0; i < ds_cor.size(); ++i) {
                        if (ds_cor[i] < 0.0) {
                            const double am =
                                (1.0 - cfg_.ip_fraction_to_boundary_tau) *
                                s[i] / (-ds_cor[i]);
                            if (am < alpha_soc)
                                alpha_soc = am;
                        }
                    }
                    alpha_soc = std::max(alpha_soc, cfg_.ls_min_alpha);
                    // Trial point for SOC
                    dvec x_t_soc = x + alpha_soc * dx_cor;
                    dvec s_t_soc = s + alpha_soc * ds_cor;
                    if ((s_t_soc.array() <= 0.0).any())
                        break;
                    // Eval at SOC trial
                    nb::dict d_t_soc;
                    try {
                        nb::list comps_t;
                        comps_t.append(nb::str("f"));
                        comps_t.append(nb::str("cE"));
                        comps_t.append(nb::str("cI"));
                        d_t_soc = nb::cast<nb::dict>(
                            model.attr("eval_all")(x_t_soc,
                                                   nb::arg("components") = comps_t));
                    } catch (const std::exception &) {
                        continue;
                    }
                    const double f_t_soc = nb::cast<double>(d_t_soc["f"]);
                    if (!std::isfinite(f_t_soc))
                        continue;
                    const double phi_t_soc =
                        f_t_soc -
                        mu * (s_t_soc.array()
                                  .unaryExpr([&](double v) {
                                      return std::log(std::max(v, barrier_eps));
                                  })
                                  .sum());
                    if (!std::isfinite(phi_t_soc))
                        continue;
                    const nb::object cE_t_soc = dict_get(d_t_soc, "cE");
                    const nb::object cI_t_soc = dict_get(d_t_soc, "cI");
                    const double thE_t_soc =
                        cE_t_soc.is_none()
                            ? 0.0
                            : nb::cast<dvec>(cE_t_soc).array().abs().sum();
                    double thI_t_soc = 0.0;
                    if (!cI_t_soc.is_none()) {
                        dvec vI_t_soc = nb::cast<dvec>(cI_t_soc);
                        thI_t_soc =
                            (vI_t_soc.array() + s_t_soc.array()).abs().sum();
                    }
                    const double theta_t_soc = thE_t_soc + thI_t_soc;
                    // Check sufficient reduction in theta for continuing SOC
                    if (theta_t_soc >= kappa_soc * theta_last)
                        break;
                    // Update for next iteration
                    theta_last = theta_t_soc;
                    cE_t_current = cE_t_soc;
                    cI_t_current = cI_t_soc;
                    s_t_current = s_t_soc;
                    // Armijo on φ using original alpha
                    if (phi_t_soc > phi0 + cfg_.ls_armijo_f * alpha * d_phi)
                        continue;
                    // Check acceptability
                    bool acceptable_ok_soc = true;
                    if (funnel_) {
                        acceptable_ok_soc = funnel_->is_acceptable(
                            theta0, f0, theta_t_soc, f_t_soc, pred_df,
                            pred_dtheta);
                    } else if (filter_ && !filter_.is_none()) {
                        acceptable_ok_soc = nb::cast<bool>(
                            filter_.attr("is_acceptable")(theta_t_soc, f_t_soc));
                    }
                    if (acceptable_ok_soc) {
                        if (funnel_) {
                            (void)funnel_->add_if_acceptable(
                                theta0, f0, theta_t_soc, f_t_soc, pred_df,
                                pred_dtheta);
                        } else if (filter_ && !filter_.is_none()) {
                            (void)filter_.attr("add_if_acceptable")(theta_t_soc,
                                                                    f_t_soc);
                        }
                        return {alpha_soc, it + soc_count, false, dx_cor,
                                ds_cor};
                    }
                }
            }
            alpha *= cfg_.ls_backtrack;
            ++it;
        }
        const bool needs_restoration = (theta0 > cfg_.ls_theta_restoration);
        (void)cfg_.ls_min_alpha; // parity with original; logging omitted here
        return {alpha_max, it, needs_restoration, dvec(), dvec()};
    }

private:
    nb::object cfg_obj_;
    nb::object filter_;
    std::shared_ptr<Funnel> funnel_;
    LSConfig cfg_;
};
