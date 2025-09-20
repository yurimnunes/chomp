// sqp_stepper.cpp
// C++23 + pybind11 implementation of the Python SQPStepper
// - Instantiates native TrustRegionManager (no TR passed from Python)
// - Calls your Python Model/Regularizer/QP via pybind11
// - Depends on Eigen + pybind11
//
// Build target: module name `sqp_cpp` exposing class `SQPStepper`

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Core>
#include <optional>
#include <string>
#include <unordered_map>

#include "../include/tr.h" // TrustRegionManager, TRConfig, TRResult, TRInfo
#include "definitions.h"

namespace py = pybind11;
using dvec = Eigen::VectorXd;
using dmat = Eigen::MatrixXd;

// TR types are in the global namespace in your header
using TRConfig = ::TRConfig;
using TrustRegionManager = ::TrustRegionManager;
using TRResult = ::TRResult;
using TRInfo = ::TRInfo;

// ------------------------ Small helpers ------------------------ //
static inline std::optional<dvec> opt_vec_from_py(const py::object &o) {
    if (o.is_none())
        return std::nullopt;
    try {
        return o.cast<dvec>();
    } catch (...) {
        return std::nullopt;
    }
}

static inline std::optional<dmat> opt_mat_from_py(const py::object &o) {
    if (o.is_none())
        return std::nullopt;
    try {
        return o.cast<dmat>();
    } catch (...) {
        return std::nullopt;
    }
}

static inline dvec clip_box(const dvec &x, const std::optional<dvec> &lb,
                            const std::optional<dvec> &ub) {
    dvec y = x;
    const int n = (int)x.size();
    if (lb)
        for (int i = 0; i < n; ++i)
            y[i] = std::max(y[i], (*lb)[i]);
    if (ub)
        for (int i = 0; i < n; ++i)
            y[i] = std::min(y[i], (*ub)[i]);
    return y;
}

template <class T>
static T get_attr_or(const py::object &obj, const char *name,
                     const T &fallback) {
    if (!py::hasattr(obj, name))
        return fallback;
    try {
        return obj.attr(name).cast<T>();
    } catch (...) {
        return fallback;
    }
}

static py::object get_or_none(const py::dict &d, const char *k) {
    if (d.contains(k))
        return d[k];
    return py::none();
}

// ------------------------ SQP Stepper (C++) ------------------------ //
class SQPStepper {
public:
    // Single constructor: builds native TrustRegionManager from cfg
    explicit SQPStepper(py::object cfg, py::object hess_mgr,
                        py::object qp_solver, py::object regularizer,
                        py::object restoration)
        : cfg_(std::move(cfg)), hess_(std::move(hess_mgr)),
          qp_(std::move(qp_solver)), reg_(std::move(regularizer)),
          rest_(std::move(restoration)), tr_(build_tr_config_from_cfg_(cfg_)) {
        init_misc_();
    }

    // step(model, x, lam, nu, it) -> (x_trial, lam_user, nu_new, info_dict)
    std::tuple<dvec, dvec, dvec, SolverInfo> step(py::object model,
                                                  const dvec &x_in,
                                                  const dvec &lam_in,
                                                  const dvec &nu_in, int it) {
        // ---- 0) Box clipping ----
        std::optional<dvec> lb = get_opt_vec(model, "lb");
        std::optional<dvec> ub = get_opt_vec(model, "ub");
        dvec x = clip_box(x_in, lb, ub);

        // ---- 1) Evaluate once at x ----
        py::list need;
        need.append("f");
        need.append("g");
        need.append("cI");
        need.append("JI");
        need.append("cE");
        need.append("JE");
        py::dict d0 = model.attr("eval_all")(x, py::arg("components") = need)
                          .cast<py::dict>();
        const double f0 = d0["f"].cast<double>();
        const dvec g0 = d0["g"].cast<dvec>();
        auto JI = opt_mat_from_py(get_or_none(d0, "JI"));
        auto JE = opt_mat_from_py(get_or_none(d0, "JE"));
        auto cI = opt_vec_from_py(get_or_none(d0, "cI"));
        auto cE = opt_vec_from_py(get_or_none(d0, "cE"));
        const double theta0 =
            model.attr("constraint_violation")(x).cast<double>();

        // ---- 2) Lagrangian Hessian + regularization ----
        py::object H_raw_py =
            model.attr("lagrangian_hessian")(x, lam_in, nu_in);
        const double tr_delta_at_call = tr_.get_delta();
        const double gnorm = g0.norm();
        py::object reg_out =
            reg_.attr("regularize")(H_raw_py, it, gnorm, tr_delta_at_call);
        py::object H_psd_py = py::none();
        if (py::isinstance<py::tuple>(reg_out)) {
            py::tuple t = reg_out.cast<py::tuple>();
            if (t.size() >= 1)
                H_psd_py = t[0];
        } else {
            H_psd_py = reg_out; // fallback
        }
        dmat H = H_psd_py.cast<dmat>();

        // If your TR supports ellipse metric from H, set it here:
        // check if norm is "ellip" in cfg
        const std::string norm_type =
            get_attr_or<std::string>(cfg_, "norm_type", std::string("2"));
        if (norm_type == "ellip")
            tr_.set_metric_from_H_dense(H); // uncomment if available/desired

        // dopt::DOptConfig cfg;
        // cfg.dopt_sigma_E = 1e-2;
        // cfg.dopt_sigma_I = 1e-2;
        // cfg.dopt_mu_target = 1e-4;
        // cfg.dopt_scaling = dopt::ScaleMode::Ruiz;

        // dopt::DOptStabilizer stab(cfg);

        // const dopt::dmat *JE_ptr = haveJE ? &JE : nullptr;
        // const dopt::dmat *JI_ptr = haveJI ? &JI : nullptr;
        // const dopt::dvec *cE_ptr = havecE ? &cE : nullptr;
        // const dopt::dvec *cI_ptr = havecI ? &cI : nullptr;
        // const dopt::dvec *lam_ptr = haveLam ? &lam : nullptr;

        // auto [rE_opt, rI_opt, meta] =
        //     stab.compute_shifts(JE_ptr, JI_ptr, cE_ptr, cI_ptr, lam_ptr);

        // ---- 3) Native TR solve (dense path) ----
        TRResult out =
            tr_.solve_dense(H, g0,
                            /*A_ineq*/ JI, /*b_ineq*/ cI,
                            /*A_eq*/ JE, /*b_eq*/ cE,
                            /*model*/ std::optional<py::object>(model),
                            /*x*/ std::optional<dvec>(x),
                            /*lb*/ lb,
                            /*ub*/ ub,
                            /*mu*/ get_attr_or<double>(model, "mu", 0.0),
                            /*f_old*/ std::optional<double>(f0));

        const dvec &p = out.p;
        const dvec &lam_user = out.lam;
        const dvec &nu_new = out.nu;
        const TRInfo &tinfo = out.info;

        if (!p.allFinite()) {
            return std::make_tuple(
                x, dvec(), dvec(),
                make_info_dict(0.0, false, false, f0, theta0, KKT{}, 0, 0.0,
                               std::numeric_limits<double>::quiet_NaN(),
                               tr_.get_delta()));
        }

        // ---- 4) Trial evaluation ----
        const dvec x_trial = x + p;
        py::dict d1 =
            model.attr("eval_all")(x_trial, py::arg("components") = need)
                .cast<py::dict>();
        const double f1 = d1["f"].cast<double>();
        const dvec g1 = d1["g"].cast<dvec>();
        auto JE1 = opt_mat_from_py(get_or_none(d1, "JE"));
        auto JI1 = opt_mat_from_py(get_or_none(d1, "JI"));
        auto cE1 = opt_vec_from_py(get_or_none(d1, "cE"));
        auto cI1 = opt_vec_from_py(get_or_none(d1, "cI"));
        const double theta1 =
            model.attr("constraint_violation")(x_trial).cast<double>();

        // Bound multipliers from TR: not returned; default to zeros for KKT
        // (fine).
        dvec zL = dvec::Zero(x.size());
        dvec zU = dvec::Zero(x.size());

        // ---- 5) KKT residuals at trial ----
        auto kkt = compute_kkt(g1, JE1, JI1, cE1, cI1, x_trial, nu_new,
                               lam_user, zL, zU, lb, ub);

        // ---- 6) Convergence test ----
        const double cEn = cE ? cE->norm() : 0.0;
        const double cIn =
            cI ? cI->cwiseMax(0.0).lpNorm<Eigen::Infinity>() : 0.0;
        const double g_scale = std::max(1.0, g0.norm() + std::max(cEn, cIn));
        const bool converged =
            ((kkt.stat / g_scale) <=
                 get_attr_or<double>(cfg_, "tol_stat", 1e-6) &&
             (kkt.ineq / g_scale) <=
                 get_attr_or<double>(cfg_, "tol_feas", 1e-6) &&
             (kkt.eq / g_scale) <=
                 get_attr_or<double>(cfg_, "tol_feas", 1e-6) &&
             ((kkt.comp / g_scale) <=
                  get_attr_or<double>(cfg_, "tol_comp", 1e-6) ||
              kkt.ineq <= 1e-10)) ||
            (std::abs(f1 - f0) <
             get_attr_or<double>(cfg_, "tol_obj_change", 1e-12));

        // ---- 7) Optional quasi-Newton update ----
        const std::string hmode = get_attr_or<std::string>(
            cfg_, "hessian_mode", std::string("exact"));
        if (hmode == "bfgs" || hmode == "lbfgs" || hmode == "hybrid") {
            if (py::hasattr(hess_, "update")) {
                hess_.attr("update")(p, g1 - g0);
            }
        }

        // ---- 8) Pack & return ----
        const double step_norm = tinfo.step_norm; // TR-provided norm
        const double rho =
            std::numeric_limits<double>::quiet_NaN(); // not exposed by TR
        const bool accepted = tinfo.accepted;

        // py::dict kkt_dict = kkt.to_dict();
        // py::dict tr_info = tr_info_to_dict(tinfo);

        SolverInfo info =
            make_info_dict(step_norm, accepted, converged, f1, theta1, kkt, 0,
                           1.0, rho, tr_.get_delta());
        info.delta = tr_.get_delta();
        // info["tr"] = tr_info;

        return std::make_tuple(x_trial, lam_user, nu_new, info);
    }

private:
    // config and collaborators
    py::object cfg_, hess_, ls_, qp_, soc_, reg_, rest_;
    TrustRegionManager tr_;
    bool requires_dense_{};

    static TRConfig build_tr_config_from_cfg_(const py::object &cfg) {
        TRConfig tc; // defaults from your C++ TRConfig (../include/tr.h)
        // Map selected fields if present.
        if (py::hasattr(cfg, "delta0"))
            tc.delta0 = cfg.attr("delta0").cast<double>();
        if (py::hasattr(cfg, "delta_min"))
            tc.delta_min = cfg.attr("delta_min").cast<double>();
        if (py::hasattr(cfg, "delta_max"))
            tc.delta_max = cfg.attr("delta_max").cast<double>();
        if (py::hasattr(cfg, "eta1"))
            tc.eta1 = cfg.attr("eta1").cast<double>();
        if (py::hasattr(cfg, "eta2"))
            tc.eta2 = cfg.attr("eta2").cast<double>();
        if (py::hasattr(cfg, "gamma1"))
            tc.gamma1 = cfg.attr("gamma1").cast<double>();
        if (py::hasattr(cfg, "gamma2"))
            tc.gamma2 = cfg.attr("gamma2").cast<double>();
        if (py::hasattr(cfg, "zeta"))
            tc.zeta = cfg.attr("zeta").cast<double>();
        if (py::hasattr(cfg, "norm_type"))
            tc.norm_type = cfg.attr("norm_type").cast<std::string>();
        if (py::hasattr(cfg, "metric_shift"))
            tc.metric_shift = cfg.attr("metric_shift").cast<double>();
        if (py::hasattr(cfg, "curvature_aware"))
            tc.curvature_aware = cfg.attr("curvature_aware").cast<bool>();
        if (py::hasattr(cfg, "criticality_enabled"))
            tc.criticality_enabled =
                cfg.attr("criticality_enabled").cast<bool>();

        // Optional extras (if present in your TRConfig):
        if (py::hasattr(cfg, "cg_tol"))
            tc.cg_tol = get_attr_or<double>(cfg, "cg_tol", tc.cg_tol);
        if (py::hasattr(cfg, "cg_tol_rel"))
            tc.cg_tol_rel =
                get_attr_or<double>(cfg, "cg_tol_rel", tc.cg_tol_rel);
        if (py::hasattr(cfg, "cg_maxiter"))
            tc.cg_maxiter = get_attr_or<int>(cfg, "cg_maxiter", tc.cg_maxiter);
        if (py::hasattr(cfg, "neg_curv_tol"))
            tc.neg_curv_tol =
                get_attr_or<double>(cfg, "neg_curv_tol", tc.neg_curv_tol);
        if (py::hasattr(cfg, "box_mode"))
            tc.box_mode =
                get_attr_or<std::string>(cfg, "box_mode", tc.box_mode);
        if (py::hasattr(cfg, "use_prec"))
            tc.use_prec = get_attr_or<bool>(cfg, "use_prec", tc.use_prec);
        if (py::hasattr(cfg, "prec_kind"))
            tc.prec_kind =
                get_attr_or<std::string>(cfg, "prec_kind", tc.prec_kind);
        if (py::hasattr(cfg, "ssor_omega"))
            tc.ssor_omega =
                get_attr_or<double>(cfg, "ssor_omega", tc.ssor_omega);

        // Filter config if exposed under cfg.filter_cfg (optional)
        if (py::hasattr(cfg, "use_filter"))
            tc.use_filter = get_attr_or<bool>(cfg, "use_filter", tc.use_filter);

        return tc;
    }

    void init_misc_() {
        requires_dense_ = py::hasattr(qp_, "requires_dense")
                              ? get_attr_or<bool>(qp_, "requires_dense", false)
                              : false;
        ensure_default("tol_feas", 1e-6);
        ensure_default("tol_stat", 1e-6);
        ensure_default("tol_comp", 1e-6);
        ensure_default("tol_obj_change", 1e-12);
        ensure_default("tr_delta0", 1.0);
        ensure_default("filter_theta_min", 1e-8);
        ensure_default("hessian_mode", std::string("exact"));
        ensure_default("soc_violation_ratio", 0.9);
    }

    template <class T> void ensure_default(const char *name, const T &val) {
        if (!py::hasattr(cfg_, name))
            cfg_.attr(name) = val;
    }

    static std::optional<dvec> get_opt_vec(const py::object &obj,
                                           const char *name) {
        if (!py::hasattr(obj, name))
            return std::nullopt;
        py::object v = obj.attr(name);
        if (v.is_none())
            return std::nullopt;
        try {
            return v.cast<dvec>();
        } catch (...) {
            return std::nullopt;
        }
    }

    static KKT compute_kkt(const dvec &g, const std::optional<dmat> &JE,
                           const std::optional<dmat> &JI,
                           const std::optional<dvec> &cE,
                           const std::optional<dvec> &cI, const dvec &xval,
                           const dvec &nu, const dvec &lam_user, const dvec &zL,
                           const dvec &zU, const std::optional<dvec> &lb,
                           const std::optional<dvec> &ub) {
        dvec r = g;
        if (JE && JE->size() > 0)
            r += JE->transpose() * nu;
        if (JI && JI->size() > 0 && lam_user.size() > 0)
            r += JI->transpose() * lam_user;
        if (zU.size() == r.size())
            r += zU; // +z_U
        if (zL.size() == r.size())
            r -= zL; // -z_L
        const double stat = r.lpNorm<Eigen::Infinity>();

        const double feas_eq = cE ? cE->lpNorm<Eigen::Infinity>() : 0.0;
        double feas_in = 0.0;
        if (cI) {
            dvec tmp = *cI;
            for (int i = 0; i < tmp.size(); ++i)
                tmp[i] = std::max(0.0, tmp[i]);
            feas_in = tmp.lpNorm<Eigen::Infinity>();
        }
        double feas_box = 0.0;
        if (lb) {
            dvec t = (*lb) - xval;
            for (int i = 0; i < t.size(); ++i)
                t[i] = std::max(0.0, t[i]);
            feas_box = std::max(feas_box, t.lpNorm<Eigen::Infinity>());
        }
        if (ub) {
            dvec t = xval - (*ub);
            for (int i = 0; i < t.size(); ++i)
                t[i] = std::max(0.0, t[i]);
            feas_box = std::max(feas_box, t.lpNorm<Eigen::Infinity>());
        }
        const double ineq = std::max(feas_in, feas_box);

        double comp = 0.0;
        if (cI && lam_user.size() > 0) {
            dvec t = *cI;
            for (int i = 0; i < t.size(); ++i)
                t[i] = std::max(0.0, t[i]) * lam_user[i];
            comp = std::max(comp, t.lpNorm<Eigen::Infinity>());
        }
        if (lb && zL.size() == xval.size()) {
            dvec t = xval - (*lb);
            for (int i = 0; i < t.size(); ++i)
                t[i] = std::max(0.0, t[i]) * zL[i];
            comp = std::max(comp, t.lpNorm<Eigen::Infinity>());
        }
        if (ub && zU.size() == xval.size()) {
            dvec t = (*ub) - xval;
            for (int i = 0; i < t.size(); ++i)
                t[i] = std::max(0.0, t[i]) * zU[i];
            comp = std::max(comp, t.lpNorm<Eigen::Infinity>());
        }
        return {stat, feas_eq, ineq, comp};
    }

    SolverInfo make_info_dict(double step_norm, bool accepted, bool converged,
                              double f_val, double theta, const KKT &kkt,
                              int ls_iters, double alpha, double rho,
                              double tr_delta) {
        SolverInfo d;
        d.mode = "sqp";
        d.step_norm = step_norm;
        d.accepted = accepted;
        d.converged = converged;
        d.f = f_val;
        d.theta = theta;
        d.stat = kkt.stat;
        d.ineq = kkt.ineq;
        d.eq = kkt.eq;
        d.comp = kkt.comp;
        d.ls_iters = ls_iters;
        d.alpha = alpha;
        d.rho = rho;
        d.tr_radius = tr_delta;
        // Alternative: return as py::dict
        // py::dict d;
        // d["step_norm"] = step_norm;
        // d["accepted"] = accepted;
        // d["converged"] = converged;
        // d["f"] = f_val;
        // d["theta"] = theta;
        // d["stat"] = kkt["stat"];
        // d["ineq"] = kkt["ineq"];
        // d["eq"] = kkt["eq"];
        // d["comp"] = kkt["comp"];
        // d["ls_iters"] = ls_iters;
        // d["alpha"] = alpha;
        // d["rho"] = rho;
        // d["tr_radius"] = tr_delta;
        return d;
    }
};

// ------------------------ PYBIND MODULE ------------------------ //
// PYBIND11_MODULE(sqp_cpp, m) {
//     m.doc() = "C++23 SQPStepper with pybind11 (native TrustRegionManager)";

//     py::class_<SQPStepper>(m, "SQPStepper")
//         .def(py::init<py::object, py::object, py::object, py::object,
//                       py::object>(),
//              py::arg("cfg"), py::arg("hessian_manager"),
//              py::arg("qp_solver"), py::arg("regularizer"),
//              py::arg("restoration"))
//         .def("step", &SQPStepper::step, py::arg("model"), py::arg("x"),
//              py::arg("lam"), py::arg("nu"), py::arg("it"),
//              R"doc(
//                  Perform one SQP step.
//                  Args:
//                    model: Python model object with eval_all,
//                    constraint_violation, lagrangian_hessian, and optional
//                    lb/ub attributes x: current point (n,) lam:
//                    user-inequality multipliers at x (mI,) nu: equality
//                    multipliers at x (mE,) it: iteration index
//                  Returns: (x_trial, lam_user, nu_new, info_dict)
//              )doc");
// }
