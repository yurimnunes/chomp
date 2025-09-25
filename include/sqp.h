// sqp_stepper.cpp
// C++23 + nanobind implementation of the Python SQPStepper
// - Instantiates native TrustRegionManager (no TR passed from Python)
// - Calls your Python Model/Regularizer/QP via nanobind
// - Depends on Eigen + nanobind

#include <cstddef>
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <Eigen/Core>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

#include "definitions.h"
#include "dopt.h"
#include "model.h"
#include "tr.h" // TrustRegionManager, TRConfig, TRResult, TRInfo

namespace nb = nanobind;
using dvec = Eigen::VectorXd;
using dmat = Eigen::MatrixXd;

// TR types are in the global namespace in your header
using TRConfig = ::TRConfig;
using TrustRegionManager = ::TrustRegionManager;
using TRResult = ::TRResult;
using TRInfo = ::TRInfo;

// Vectorized box clipping: y = min(max(x, lb), ub) where lb/ub may be absent
static inline dvec clip_box(const dvec &x, const std::optional<dvec> &lb,
                            const std::optional<dvec> &ub) {
    dvec y = x;
    if (lb) {
        if (NB_UNLIKELY(lb->size() != x.size()))
            throw std::invalid_argument("clip_box: lb size mismatch");
        y = y.cwiseMax(*lb);
    }
    if (ub) {
        if (NB_UNLIKELY(ub->size() != x.size()))
            throw std::invalid_argument("clip_box: ub size mismatch");
        y = y.cwiseMin(*ub);
    }
    return y;
}

// ------------------------ SQP Stepper (C++) ------------------------ //
class SQPStepper {
public:
    // Single constructor: builds native TrustRegionManager from cfg
    explicit SQPStepper(nb::object cfg, nb::object qp_solver,
                        ModelC *model = nullptr)
        : cfg_(std::move(cfg)), qp_(std::move(qp_solver)),
          tr_(build_tr_config_from_cfg_(cfg_)), model_(model) {
        if (!model_)
            throw std::invalid_argument(
                "SQPStepper: ModelC* model must not be null");
        init_misc_();
        auto dopt_cfg = dopt::DOptConfig();
        dopt = new dopt::DOptStabilizer(dopt_cfg);
    }

    std::shared_ptr<regx::Regularizer> regularizer_ =
        std::make_shared<regx::Regularizer>();

    dopt::DOptStabilizer *dopt = nullptr;
    ModelC *model_ = nullptr; // optional ModelC pointer for AD/LBFGS Hessians

    std::tuple<dvec, dvec, dvec, SolverInfo>
    step(const dvec &x_in, const dvec &lam_in, const dvec &nu_in, int it) {
        // ---- 0) Box clipping ----
        std::optional<dvec> lb = model_->get_lb();
        std::optional<dvec> ub = model_->get_ub();
        const int mI = model_->get_mI();
        const int mE = model_->get_mE();
        const int n = model_->get_n();

        dvec x = clip_box(x_in, lb, ub);

        // ---- 1) Evaluate once at x ----
        const std::vector<std::string> need_strs = {"f",  "g",  "JI",
                                                    "JE", "cI", "cE"};
        model_->eval_all(x, need_strs);

        const double f0 = model_->get_f().value();
        const dvec g0 = model_->get_g().value();

        spmat JI = (mI > 0) ? model_->get_JI().value() : spmat(mI, n);
        spmat JE = (mE > 0) ? model_->get_JE().value() : spmat(mE, n);
        dvec cI = (mI > 0) ? model_->get_cI().value() : dvec::Zero(mI);
        dvec cE = (mE > 0) ? model_->get_cE().value() : dvec::Zero(mE);

        const double theta0 = model_->constraint_violation(x);

        // ---- 2) Compute DOpt shifts (matrices unchanged; dense only for
        // stats) ----
        Eigen::MatrixXd JE_dense = JE.toDense();
        Eigen::MatrixXd JI_dense = JI.toDense();

        auto [rE_opt, rI_opt, dopt_meta] =
            dopt->compute_shifts(JE_dense, JI_dense, cE, cI, lam_in, nu_in);

        // ---- 2b) Adaptive damping & relative caps (gentle stabilization) ----
        // Config knobs (with safe defaults)
        const double k_theta = get_attr_or<double>(cfg_, "dopt_k_theta", 0.5);
        const double k_delta = get_attr_or<double>(cfg_, "dopt_k_delta", 0.5);
        const double delta_ref =
            get_attr_or<double>(cfg_, "dopt_delta_ref", 1.0);
        const double tol_eq_off =
            get_attr_or<double>(cfg_, "dopt_tol_eq_off", 1e-8);
        const double alpha_comp =
            get_attr_or<double>(cfg_, "dopt_alpha_comp", 0.5);
        const double rel_cap = get_attr_or<double>(cfg_, "dopt_rel_cap", 0.5);
        const bool ap_plus_b_le0 =
            get_attr_or<bool>(cfg_, "qp_aplusb_le0", true);
        // If your TR expects A p ≤ b, set qp_aplusb_le0=false in cfg.

        const double delta_now = tr_.get_delta();
        const double w_theta = std::min(1.0, k_theta * theta0);
        const double w_delta =
            std::min(1.0, k_delta * (delta_now / std::max(1e-16, delta_ref)));
        const double w_base = std::min(1.0, std::max(0.0, w_theta * w_delta));

        // Start from unshifted residuals
        dvec cE_shifted = cE;
        dvec cI_shifted = cI;

        // Equality shift (turn off if already very tight)
        if (rE_opt && rE_opt->size() > 0) {
            dvec rE = *rE_opt;
            // Relative cap: |rE_i| <= rel_cap * (|cE_i| + eps)
            for (Eigen::Index i = 0; i < rE.size(); ++i) {
                const double cap = rel_cap * (std::abs(cE(i)) + 1e-12);
                rE(i) = std::clamp(rE(i), -cap, cap);
            }
            const double cEnorm = cE.norm();
            const double w_eq = (cEnorm <= tol_eq_off) ? 0.0 : w_base;
            cE_shifted.noalias() += w_eq * rE;
        }

        // Inequality shift (nonnegative, smaller comp push)
        if (rI_opt && rI_opt->size() > 0) {
            dvec rI = *rI_opt;
            // Clip to [0, rel_cap * (|cI_i| + eps)]
            for (Eigen::Index i = 0; i < rI.size(); ++i) {
                const double cap = rel_cap * (std::abs(cI(i)) + 1e-12);
                rI(i) = std::clamp(rI(i), 0.0, cap);
            }
            cI_shifted.noalias() += w_base * (alpha_comp * rI);
        }

        // ---- 3) Lagrangian Hessian (exact or LBFGS) + regularization ----
        const std::string hmode = get_attr_or<std::string>(
            cfg_, "hessian_mode", std::string("lbfgs"));

        spmat H0;
        if (hmode == "exact") {
            H0 = model_->hess(x, lam_in, nu_in);
        } else if (hmode == "lbfgs") {
            model_->set_use_lbfgs_hess(true);
            H0 = model_->lbfgs_matrix(
                get_attr_or<double>(cfg_, "lbfgs_sparse_threshold", 1e-12));
        } else {
            throw std::runtime_error("Unknown hessian_mode: " + hmode);
        }

        auto [H, reg_info] = regularizer_->regularize(H0, it);
        H.makeCompressed();

        // Optional: set TR metric from H
        const std::string norm_type =
            get_attr_or<std::string>(cfg_, "norm_type", std::string("2"));
        if (norm_type == "ellip") {
            tr_.set_metric_from_H_dense(H);
        }

        // ---- 4) Build RHS for TR/QP per convention ----
        dvec b_eq = ap_plus_b_le0
                        ? cE_shifted
                        : -cE_shifted; // JE p + b_eq = 0  OR  JE p = b_eq
        dvec b_in = ap_plus_b_le0
                        ? cI_shifted
                        : -cI_shifted; // JI p + b_in ≤ 0 OR  JI p ≤ b_in

        // ---- 5) Native TR solve (dense path) ----
        TRResult out = tr_.solve_dense(H, g0,
                                       /*A_ineq*/ JI, /*b_ineq*/ b_in,
                                       /*A_eq*/ JE, /*b_eq*/ b_eq,
                                       /*model*/ model_,
                                       /*x*/ std::optional<dvec>(x),
                                       /*lb*/ lb, /*ub*/ ub,
                                       /*mu*/ 0.0,
                                       /*f_old*/ std::optional<double>(f0));

        const dvec &p = out.p;
        const dvec &lam_usr = out.lam;
        const dvec &nu_new = out.nu;
        const TRInfo &tinfo = out.info;

        if (!p.allFinite()) {
            return std::make_tuple(
                x, dvec(), dvec(),
                make_info_dict(0.0, false, false, f0, theta0, KKT{}, 0, 0.0,
                               std::numeric_limits<double>::quiet_NaN(),
                               tr_.get_delta()));
        }

        // ---- 6) Trial evaluation (true, unshifted problem) ----
        const dvec x_trial = clip_box(x + p, lb, ub);
        model_->eval_all(x_trial, need_strs);

        const double f1 = model_->get_f().value();
        const dvec g1 = model_->get_g().value();
        spmat JI1 = (mI > 0) ? model_->get_JI().value() : spmat(mI, n);
        spmat JE1 = (mE > 0) ? model_->get_JE().value() : spmat(mE, n);
        dvec cI1 = (mI > 0) ? model_->get_cI().value() : dvec::Zero(mI);
        dvec cE1 = (mE > 0) ? model_->get_cE().value() : dvec::Zero(mE);

        const double theta1 = model_->constraint_violation(x_trial);

        // Bound multipliers (not returned by TR): zeros for KKT
        dvec zL = dvec::Zero(x.size());
        dvec zU = dvec::Zero(x.size());

        // ---- 7) KKT residuals at trial (unshifted) ----
        auto kkt = compute_kkt(g1, JE1, JI1, cE1, cI1, x_trial, nu_new, lam_usr,
                               zL, zU, lb, ub);

        // ---- 8) Convergence test ----
        const double cEn = (model_->get_cE() ? model_->get_cE()->norm() : 0.0);
        const double cIn =
            (model_->get_cI()
                 ? model_->get_cI()->cwiseMax(0.0).lpNorm<Eigen::Infinity>()
                 : 0.0);
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

        // ---- 9) LBFGS memory update *only if accepted* ----
        if (hmode == "lbfgs" && tinfo.accepted) {
            model_->lbfgs_update(x_trial, lam_usr, nu_new);
        }

        // ---- 10) Pack & return ----
        const double step_norm = tinfo.step_norm;
        const double rho =
            std::numeric_limits<double>::quiet_NaN(); // not exposed by TR
        const bool accepted = tinfo.accepted;

        SolverInfo info =
            make_info_dict(step_norm, accepted, converged, f1, theta1, kkt, 0,
                           1.0, rho, tr_.get_delta());
        info.delta = tr_.get_delta();

        return std::make_tuple(x_trial, lam_usr, nu_new, info);
    }

private:
    // config and collaborators
    nb::object cfg_, hess_, ls_, qp_, soc_, reg_, rest_;
    TrustRegionManager tr_;
    bool requires_dense_{};

    static TRConfig build_tr_config_from_cfg_(const nb::object &cfg) {
        TRConfig tc; // start with C++ defaults

        // Core TR params
        tc.delta0 = get_attr_or<double>(cfg, "delta0", tc.delta0);
        tc.delta_min = get_attr_or<double>(cfg, "delta_min", tc.delta_min);
        tc.delta_max = get_attr_or<double>(cfg, "delta_max", tc.delta_max);
        tc.eta1 = get_attr_or<double>(cfg, "eta1", tc.eta1);
        tc.eta2 = get_attr_or<double>(cfg, "eta2", tc.eta2);
        tc.gamma1 = get_attr_or<double>(cfg, "gamma1", tc.gamma1);
        tc.gamma2 = get_attr_or<double>(cfg, "gamma2", tc.gamma2);
        tc.zeta = get_attr_or<double>(cfg, "zeta", tc.zeta);
        tc.norm_type = get_attr_or<std::string>(cfg, "norm_type", tc.norm_type);
        tc.metric_shift =
            get_attr_or<double>(cfg, "metric_shift", tc.metric_shift);
        tc.curvature_aware =
            get_attr_or<bool>(cfg, "curvature_aware", tc.curvature_aware);
        tc.criticality_enabled = get_attr_or<bool>(cfg, "criticality_enabled",
                                                   tc.criticality_enabled);

        // Optional extras (present in your TRConfig)
        tc.cg_tol = get_attr_or<double>(cfg, "cg_tol", tc.cg_tol);
        tc.cg_tol_rel = get_attr_or<double>(cfg, "cg_tol_rel", tc.cg_tol_rel);
        tc.cg_maxiter = get_attr_or<int>(cfg, "cg_maxiter", tc.cg_maxiter);
        tc.neg_curv_tol =
            get_attr_or<double>(cfg, "neg_curv_tol", tc.neg_curv_tol);
        tc.box_mode = get_attr_or<std::string>(cfg, "box_mode", tc.box_mode);
        tc.use_prec = get_attr_or<bool>(cfg, "use_prec", tc.use_prec);
        tc.prec_kind = get_attr_or<std::string>(cfg, "prec_kind", tc.prec_kind);
        tc.ssor_omega = get_attr_or<double>(cfg, "ssor_omega", tc.ssor_omega);

        // Filter knob
        tc.use_filter = get_attr_or<bool>(cfg, "use_filter", tc.use_filter);

        return tc;
    }

    void init_misc_() {
        requires_dense_ = nb::hasattr(qp_, "requires_dense")
                              ? get_attr_or<bool>(qp_, "requires_dense", false)
                              : false;
        ensure_default("tol_feas", 1e-6);
        ensure_default("tol_stat", 1e-6);
        ensure_default("tol_comp", 1e-6);
        ensure_default("tol_obj_change", 1e-12);
        ensure_default("tr_delta0", 1.0);
        ensure_default("filter_theta_min", 1e-8);
        ensure_default("hessian_mode",
                       std::string("lbfgs")); // "exact" or "lbfgs"
        ensure_default("lbfgs_sparse_threshold", 1e-12);
        ensure_default("soc_violation_ratio", 0.9);
    }

    template <class T> void ensure_default(const char *name, const T &val) {
        if (!nb::hasattr(cfg_, name))
            cfg_.attr(name) = val;
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
        return d;
    }
};
