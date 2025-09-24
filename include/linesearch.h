// line_searcher_nb.cc
// nanobind + Eigen port of line_searcher_pybind.cc
//
// Requires:
//   - nanobind (with Eigen bridge): <nanobind/nanobind.h>,
//   <nanobind/eigen/dense.h>
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

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "model.h"

namespace nb = nanobind;
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
    int max_soc{4};
    double kappa_soc_min{0.1};  // Minimum kappa_soc for adaptive strategy
    double kappa_soc_max{0.99}; // Maximum kappa_soc for adaptive strategy
    double kappa_soc_base{0.5}; // Base kappa_soc when theta_t/theta0 is large
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

// ---------------- line searcher (holds Python filter, C++ funnel)
// ----------------
class LineSearcher {
public:
    LineSearcher(nb::object cfg, nb::object filter = nb::none(),
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
    // If dx_cor is empty, use original dx, ds with alpha; else use alpha with
    // dx_cor, ds_cor
    std::tuple<double, int, bool, dvec, dvec>
    search(ModelC *model, const dvec &x, const dvec &dx, const dvec &ds,
           const dvec &s, double mu, double d_phi,
           std::optional<double> theta0_opt = std::nullopt,
           double alpha_max = 1.0) const {
        // --- base eval (single call; request only needed pieces)
        std::vector<std::string> comp_names{"f", "g", "cE", "cI", "JE", "JI"};

        model->eval_all(x, comp_names);
        const double f0 = model->get_f().value();
        const dvec g0 = model->get_g().value();
        const std::optional<dvec> cE0 = model->get_cE();
        const std::optional<dvec> cI0 = model->get_cI();
        const std::optional<spmat> JE0 = model->get_JE();
        const std::optional<spmat> JI0 = model->get_JI();

        std::cout << "Line search eval done.\n";

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
            if (cE0) {
                dvec vE = cE0.value();
                thE = vE.array().abs().sum();
            }
            if (cI0) {
                dvec vI = cI0.value();
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

        std::cout << "Line search precomputes done.\n";
        // θ linear prediction: JE@dx and JI@dx (support scipy.sparse)
        const dvec je_dx = JE0 ? JE0.value() * dx : dvec();
        const dvec ji_dx = JI0 ? JI0.value() * dx : dvec();
        double thE_lin = 0.0, thI_lin = 0.0;
        if (cE0 && !cE0->size() == 0) {
            dvec vE = cE0.value();
            thE_lin = (je_dx.size() ? (vE + je_dx).array().abs().sum()
                                    : vE.array().abs().sum());
        }
        if (cI0 && !cI0->size() == 0) {
            dvec vI = cI0.value();
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
        while (it < cfg_.ls_max_iter) {
            const dvec x_t = x + alpha * dx;
            const dvec s_t = s + alpha * ds;
            if ((s_t.array() <= 0.0).any()) {
                alpha *= cfg_.ls_backtrack;
                ++it;
                continue;
            }
            try {
                std::vector<std::string> comps_t{"f", "cE", "cI"};
                model->eval_all(x_t, comps_t);
                const double f_t = model->get_f().value();
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
                    auto cE_t = model->get_cE();
                    auto cI_t = model->get_cI();
                    const double thE_t = cE_t ? cE_t.value().array().abs().sum()
                                              : 0.0; // cE_t.is_none() ? 0.0
                    double thI_t = 0.0;
                    if (cI_t) {
                        dvec vI_t = cI_t.value();
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
            // Try second-order corrections if conditions met (use ModelC API)
            if (it == 0 && theta_t >= theta0 && cfg_.max_soc > 0) {
                // Current residuals at (x_t, s_t)
                dvec cE_t = cE0 ? model->get_cE().value() : dvec();
                dvec cI_t = cI0 ? model->get_cI().value() : dvec();
                dvec s_t_cur = s + alpha * ds;

                // Guard sizes
                auto nnzE = (cE_t.size() ? cE_t.size() : 0);
                auto nnzI = (cI_t.size() ? cI_t.size() : 0);

                double theta_last = theta_t;
                int soc_count = 0;

                while (soc_count < cfg_.max_soc) {
                    ++soc_count;

                    // --- Adaptive kappa_soc (like before, but entirely local)
                    double kappa_soc = cfg_.kappa_soc_base;
                    if (theta0 > 1e-8) {
                        const double ratio = theta_last / theta0;
                        if (ratio > 10.0) {
                            kappa_soc = cfg_.kappa_soc_min; // strict
                        } else if (ratio > 1.0) {
                            // Linear interp in (1,10] from min→max
                            kappa_soc =
                                cfg_.kappa_soc_min +
                                (cfg_.kappa_soc_max - cfg_.kappa_soc_min) *
                                    (1.0 - (ratio - 1.0) / 9.0);
                        } else {
                            kappa_soc = cfg_.kappa_soc_max; // loose
                        }
                    }
                    // Relax for later SOC iterations
                    kappa_soc = std::min(kappa_soc + 0.1 * (soc_count - 1),
                                         cfg_.kappa_soc_max);

                    // --- Build/refresh Jacobians at x_t
                    // We use a fresh eval so J* correspond to x_t (not x).
                    {
                        std::vector<std::string> compsJ{"JE", "JI"};
                        model->eval_all(x + alpha * dx, compsJ);
                    }
                    dmat JE_t = (JE0 && nnzE) ? model->get_JE().value()
                                              : dmat((Eigen::Index)0,
                                                     (Eigen::Index)x.size());
                    dmat JI_t = (JI0 && nnzI) ? model->get_JI().value()
                                              : dmat((Eigen::Index)0,
                                                     (Eigen::Index)x.size());

                    // Right-hand side residuals for LS:
                    // rE = cE(x_t), rI = cI(x_t) + s_t
                    dvec rE = cE_t;
                    dvec rI = (nnzI ? (cI_t + s_t_cur) : dvec());

                    // If both empty, nothing to correct
                    if ((rE.size() == 0) && (rI.size() == 0))
                        break;

                    // --- Form normal equations: (A^T A) dx_cor = -A^T r
                    // where A = [JE_t; JI_t], r = [rE; rI]
                    const Eigen::Index n = static_cast<Eigen::Index>(x.size());
                    dmat AtA = dmat::Zero(n, n);
                    dvec At_r = dvec::Zero(n);

                    if (rE.size() && JE_t.rows() == rE.size()) {
                        AtA.noalias() += JE_t.transpose() * JE_t;
                        At_r.noalias() += JE_t.transpose() * rE;
                    }
                    if (rI.size() && JI_t.rows() == rI.size()) {
                        AtA.noalias() += JI_t.transpose() * JI_t;
                        At_r.noalias() += JI_t.transpose() * rI;
                    }

                    // Small Tikhonov if needed for numerical safety
                    const double eps_reg = 1e-12;
                    AtA.diagonal().array() += eps_reg;

                    // Solve normal equations
                    dvec dx_cor;
                    {
                        Eigen::LDLT<dmat> ldlt(AtA);
                        if (ldlt.info() != Eigen::Success) {
                            // Fallback to QR if LDLT fails
                            dx_cor = (-AtA).colPivHouseholderQr().solve(At_r);
                        } else {
                            dx_cor = ldlt.solve(-At_r);
                        }
                    }

                    // Slack correction: ds_cor = -(cI_t + s_t + JI_t dx_cor)
                    dvec ds_cor;
                    if (rI.size()) {
                        dvec JI_dx = (JI_t.rows() ? (JI_t * dx_cor) : dvec());
                        ds_cor = -(rI + JI_dx);
                    } else {
                        ds_cor = dvec(); // no inequality side
                    }

                    // Fraction-to-boundary for corrected direction (from
                    // current s)
                    double alpha_soc = 1.0;
                    if (ds_cor.size()) {
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
                    }

                    // Trial with SOC step
                    const dvec x_t_soc = x + alpha_soc * dx_cor;
                    const dvec s_t_soc = s + alpha_soc * ds_cor;
                    if (s_t_soc.size() && (s_t_soc.array() <= 0.0).any())
                        break;

                    // Evaluate f, cE, cI at (x_t_soc)
                    double f_t_soc = 0.0;
                    double theta_t_soc = 0.0;
                    try {
                        std::vector<std::string> comps_soc{"f", "cE", "cI"};
                        model->eval_all(x_t_soc, comps_soc);

                        f_t_soc = model->get_f().value();
                        if (!std::isfinite(f_t_soc))
                            continue;

                        dvec cE_soc =
                            model->get_cE() ? model->get_cE().value() : dvec();
                        dvec cI_soc =
                            model->get_cI() ? model->get_cI().value() : dvec();

                        const double thE_soc =
                            cE_soc.size() ? cE_soc.array().abs().sum() : 0.0;
                        double thI_soc = 0.0;
                        if (cI_soc.size()) {
                            thI_soc =
                                (cI_soc.array() + s_t_soc.array()).abs().sum();
                        }
                        theta_t_soc = thE_soc + thI_soc;
                    } catch (const std::exception &) {
                        continue;
                    }

                    // Sufficient infeasibility reduction?
                    if (theta_t_soc >= kappa_soc * theta_last)
                        break;

                    // Armijo on φ with corrected step (use same d_phi slope)
                    const double phi_t_soc =
                        f_t_soc -
                        mu * (s_t_soc.array()
                                  .unaryExpr([&](double v) {
                                      return std::log(std::max(v, barrier_eps));
                                  })
                                  .sum());
                    if (!std::isfinite(phi_t_soc) ||
                        phi_t_soc > phi0 + cfg_.ls_armijo_f * alpha * d_phi) {
                        // Not acceptable by merit decrease → try next SOC or
                        // quit
                        continue;
                    }

                    // Acceptability (filter/funnel)
                    bool acceptable_ok_soc = true;
                    if (funnel_) {
                        acceptable_ok_soc = funnel_->is_acceptable(
                            theta0, f0, theta_t_soc, f_t_soc, pred_df,
                            pred_dtheta);
                    } else if (filter_ && !filter_.is_none()) {
                        acceptable_ok_soc = nb::cast<bool>(filter_.attr(
                            "is_acceptable")(theta_t_soc, f_t_soc));
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
                        // Return corrected step
                        return {alpha_soc, it + soc_count, false, dx_cor,
                                ds_cor};
                    }

                    // Prepare for a possible next SOC round:
                    theta_last = theta_t_soc;
                    // Refresh residuals at (x_t_soc) for next iteration
                    try {
                        std::vector<std::string> comps_next{"cE", "cI"};
                        model->eval_all(x_t_soc, comps_next);
                        cE_t =
                            model->get_cE() ? model->get_cE().value() : dvec();
                        cI_t =
                            model->get_cI() ? model->get_cI().value() : dvec();
                        s_t_cur = s_t_soc;
                    } catch (const std::exception &) {
                        break;
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
