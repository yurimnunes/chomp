// ip_cpp.cpp — optimized, modernized C++23 version (behavior preserved)

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SparseCore>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nanobind/eigen/dense.h>  // dense Eigen
#include <nanobind/eigen/sparse.h> // sparse Eigen
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "definitions.h"
#include "funnel.h"
#include "ip_aux.h"
#include "kkt_core.h"
#include "linesearch.h"
#include "model.h" // Model
#include "regularizer.h"

#include "ip_helpers.h"
#include <chrono>
#include <iomanip> // std::setprecision

// nanobind alias
namespace nb = nanobind;

// ---------- Helpers ----------
namespace detail {

[[nodiscard]] inline Sigmas
build_sigmas(const dvec &zL, const dvec &zU, const Bounds &B, const dvec &lmb,
             const dvec &s, const dvec &cI, double tau_shift,
             double bound_shift, bool use_shifted, double eps_abs, double cap) {
    const int n = static_cast<int>(zL.size());
    const int mI = static_cast<int>(s.size());
    Sigmas S;

    // Ultra-vectorized Sigma_x computation
    S.Sigma_x = dvec::Zero(n);

    if (n > 0) {
        const double shift = use_shifted ? bound_shift : 0.0;

        // Convert bounds to Eigen arrays for vectorization
        dvec hasL_float(n), hasU_float(n), sL_vec(n), sU_vec(n);

        // Single loop to extract all bound data
        for (int i = 0; i < n; ++i) {
            hasL_float[i] = B.hasL[i] ? 1.0 : 0.0;
            hasU_float[i] = B.hasU[i] ? 1.0 : 0.0;
            sL_vec[i] = B.sL[i];
            sU_vec[i] = B.sU[i];
        }

        // Fully vectorized computation
        dvec dL = (sL_vec.array() + shift).max(eps_abs);
        dvec dU = (sU_vec.array() + shift).max(eps_abs);

        dvec zL_contrib = hasL_float.cwiseProduct(zL.cwiseQuotient(dL));
        dvec zU_contrib = hasU_float.cwiseProduct(zU.cwiseQuotient(dU));

        S.Sigma_x = (zL_contrib + zU_contrib).array().max(0.0).min(cap);
    }

    // Same vectorized Sigma_s as above
    S.Sigma_s = dvec::Zero(mI);

    if (mI > 0) {
        if (use_shifted) {
            dvec d = (s.array() + tau_shift).max(eps_abs);
            S.Sigma_s = (lmb.cwiseQuotient(d)).array().max(0.0).min(cap);
        } else {
            dvec sf = cI.cwiseAbs().array().max(1e-8).min(1.0);
            dvec sv = s.cwiseMax(sf).array().max(eps_abs);
            S.Sigma_s = (lmb.cwiseQuotient(sv)).array().max(0.0).min(cap);
        }
    }

    return S;
}

[[nodiscard]] inline std::pair<dvec, dvec>
dz_bounds_from_dx_vec(const dvec &dx, const dvec &zL, const dvec &zU,
                      const Bounds &B, double bound_shift, bool use_shifted,
                      double mu, bool use_mu) {
    const int n = dx.size();
    dvec dzL = dvec::Zero(n), dzU = dvec::Zero(n);
    for (int i = 0; i < n; ++i) {
        if (B.hasL[i]) {
            const double d = clamp_min(
                B.sL[i] + (use_shifted ? bound_shift : 0.0), consts::EPS_DIV);
            dzL[i] = use_mu ? (mu - d * zL[i] - zL[i] * dx[i]) / d
                            : -(zL[i] * dx[i]) / d;
        }
        if (B.hasU[i]) {
            const double d = clamp_min(
                B.sU[i] + (use_shifted ? bound_shift : 0.0), consts::EPS_DIV);
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
        for (int i = 0; i < m; ++i)
            acc += std::abs((s[i] + tau_shift) * lmb[i] - mu);
    } else {
        for (int i = 0; i < m; ++i)
            acc += std::abs(s[i] * lmb[i] - mu);
    }
    return acc / clamp_min(m, 1);
}

[[nodiscard]] inline double alpha_ftb_vec(const dvec &x, const dvec &dx,
                                          const dvec &s, const dvec &ds,
                                          const dvec &lmb, const dvec &dlam,
                                          const Bounds &B, double tau_pri,
                                          double tau_dual) {

    double a_pri = 1.0, a_dual = 1.0;

    for (int i = 0; i < s.size(); ++i)
        if (ds[i] < 0.0)
            a_pri = std::min(a_pri, -s[i] / std::min(ds[i], -consts::EPS_DIV));

    for (int i = 0; i < lmb.size(); ++i)
        if (dlam[i] < 0.0)
            a_dual =
                std::min(a_dual, -lmb[i] / std::min(dlam[i], -consts::EPS_DIV));

    for (int i = 0; i < x.size(); ++i) {
        if (B.hasL[i] && dx[i] < 0.0)
            a_pri = std::min(a_pri, -(x[i] - B.lb[i]) /
                                        std::min(dx[i], -consts::EPS_DIV));
        if (B.hasU[i] && dx[i] > 0.0)
            a_pri = std::min(a_pri, (B.ub[i] - x[i]) /
                                        clamp_min(dx[i], consts::EPS_DIV));
    }
    return clamp(std::min(tau_pri * a_pri, tau_dual * a_dual), 0.0, 1.0);
}

[[nodiscard]] inline double comp_inf_norm(const dvec &s, const dvec &lam,
                                          const dvec &zL, const dvec &zU,
                                          const Bounds &B, double mu,
                                          bool use_shifted, double tau_shift,
                                          double bound_shift) {

    const int n = static_cast<int>(zL.size());
    double c_inf = 0.0;

    for (int i = 0; i < n; ++i) {
        if (B.hasL[i]) {
            const double sL = B.sL[i] + (use_shifted ? bound_shift : 0.0);
            c_inf = std::max(c_inf, std::abs(sL * zL[i] - mu));
        }
        if (B.hasU[i]) {
            const double sU = B.sU[i] + (use_shifted ? bound_shift : 0.0);
            c_inf = std::max(c_inf, std::abs(sU * zU[i] - mu));
        }
    }
    if (s.size()) {
        for (int i = 0; i < s.size(); ++i) {
            const double se = s[i] + (use_shifted ? tau_shift : 0.0);
            c_inf = std::max(c_inf, std::abs(se * lam[i] - mu));
        }
    }
    return c_inf;
}

inline void cap_bound_duals_sigma_box(dvec &zL, dvec &zU, const Bounds &B,
                                      bool use_shifted, double bound_shift,
                                      double mu, double ksig = 1e10) {
    for (int i = 0; i < zL.size(); ++i) {
        if (B.hasL[i]) {
            const double sLc = clamp_min(
                B.sL[i] + (use_shifted ? bound_shift : 0.0), consts::EPS_DIV);
            const double lo = mu / (ksig * sLc);
            const double hi = (ksig * mu) / sLc;
            zL[i] = clamp(zL[i], lo, hi);
        }
        if (B.hasU[i]) {
            const double sUc = clamp_min(
                B.sU[i] + (use_shifted ? bound_shift : 0.0), consts::EPS_DIV);
            const double lo = mu / (ksig * sUc);
            const double hi = (ksig * mu) / sUc;
            zU[i] = clamp(zU[i], lo, hi);
        }
    }
}

} // namespace detail

// ---------- Main Interior Point Stepper ----------
class InteriorPointStepper {
public:
    Bounds get_bounds(const dvec &x) {
        const int n = static_cast<int>(x.size());
        Bounds B;

        // lb / ub
        B.lb = m_->get_lb().value_or(dvec::Constant(n, -consts::INF));
        B.ub = m_->get_ub().value_or(dvec::Constant(n, consts::INF));

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

    IPState st{};
    std::shared_ptr<LineSearcher> ls_;
    std::shared_ptr<regx::Regularizer> regularizer_ =
        std::make_shared<regx::Regularizer>();
    MehrotraGondzioSolver *mg_solver_ = nullptr;
    AdaptiveShiftManager *shift_mgr_ = nullptr; // <<< NEW

    ModelC *m_ = nullptr;
    InteriorPointStepper(nb::object cfg, ModelC *cfg_model)
        : cfg_(std::move(cfg)), m_(cfg_model) {
        load_defaults_();
        std::shared_ptr<Funnel> funnel = std::shared_ptr<Funnel>();
        ls_ = std::make_shared<LineSearcher>(cfg_, nullptr, funnel);
        mg_solver_ = new MehrotraGondzioSolver(cfg_, m_);
        mg_solver_->set_kkt_solver([this](const spmat &W, const dvec &r1,
                                          const std::optional<spmat> &JE,
                                          const std::optional<dvec> &r_pE,
                                          std::string_view method) {
            return this->solve_KKT_(W, r1, JE, r_pE, method);
        });
        shift_mgr_ = new AdaptiveShiftManager(cfg_);
    }

    double penalty_rho_ = 1.0;   // cubic penalty ρ parameter
    double penalty_sigma_ = 0.1; // cubic penalty σ parameter
    double last_theta_ = std::numeric_limits<double>::max();
    std::deque<double> theta_history_; // Fixed: use deque for pop_front()

    std::tuple<dvec, dvec, dvec, SolverInfo>
    step(const dvec &x, const dvec &lam, const dvec &nu, int it,
         std::optional<IPState> /*ip_state_opt*/ = std::nullopt) {
#if IP_PROFILE
        IP_TIMER("step()");
#endif
        // -------------------- Init / state --------------------
        if (!st.initialized)
            st = state_from_model_(x);

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
            get_attr_or<bool>(cfg_, "ip_use_shifted_barrier", false);
        const bool shift_adapt =
            get_attr_or<bool>(cfg_, "ip_shift_adaptive", true);
        IP_LAP("init/state");

        // -------------------- Trust region initialization --------------------
        if (!tr_inited_) {
            tr_radius_ = get_attr_or<double>(cfg_, "ip_tr_init", 1e3);
            tr_ema_ = tr_radius_;
            tr_inited_ = true;

            // Initialize cubic penalty parameters
            penalty_rho_ =
                get_attr_or<double>(cfg_, "ip_penalty_rho_init", 1.0);
            penalty_sigma_ =
                get_attr_or<double>(cfg_, "ip_penalty_sigma_init", 0.1);
            last_theta_ = std::numeric_limits<double>::max();
            theta_history_.clear();
        }

        // -------------------- Enhanced helper functions --------------------
        struct EvalPack {
            double f{};
            dvec g, cI, cE;
            spmat JI, JE;
            double theta{};
            double theta_E{}, theta_I{}; // Separate violation measures
        };

        auto eval_all_at = [&](const dvec &X, int n_, int mI_,
                               int mE_) -> EvalPack {
            m_->eval_all(X);
            EvalPack E;
            E.f = m_->get_f().value_or(0.0);
            E.g = m_->get_g().value_or(dvec::Zero(n_));
            E.cI = (mI_ > 0) ? m_->get_cI().value() : dvec::Zero(mI_);
            E.cE = (mE_ > 0) ? m_->get_cE().value() : dvec::Zero(mE_);
            E.JI = (mI_ > 0) ? m_->get_JI().value() : spmat(mI_, n_);
            E.JE = (mE_ > 0) ? m_->get_JE().value() : spmat(mE_, n_);
            E.theta = m_->constraint_violation(X);

            // Separate constraint violations for penalty tuning
            E.theta_E = (mE_ > 0) ? E.cE.lpNorm<1>() : 0.0;
            E.theta_I =
                (mI_ > 0) ? (E.cI.array().max(0.0)).matrix().lpNorm<1>() : 0.0;

            return E;
        };

        auto build_r_d = [&](const EvalPack &E, const dvec &lmb_in,
                             const dvec &nu_in, const dvec &zL_in,
                             const dvec &zU_in) -> dvec {
            dvec r_d = E.g;
            if (E.JI.nonZeros())
                r_d.noalias() += E.JI.transpose() * lmb_in;
            if (E.JE.nonZeros())
                r_d.noalias() += E.JE.transpose() * nu_in;
            r_d -= zL_in;
            r_d += zU_in;
            return r_d;
        };

        auto aggregated_error =
            [&](const EvalPack &E, const dvec &x_in, const dvec &lam_in,
                const dvec &nu_in, const dvec &zL_in, const dvec &zU_in,
                double mu_in, const dvec &s_in, const Bounds &B_in, int n_,
                int mE_, int mI_, bool use_shifted_, double tau_shift_,
                double bound_shift_) -> double {
            dvec r_d = build_r_d(E, lam_in, nu_in, zL_in, zU_in);
            const double s_max = get_attr_or<double>(cfg_, "ip_s_max", 100.0);
            const int denom_ct = static_cast<int>(s_in.size()) + mE_ + n_;
            const double sum_mults = lam_in.lpNorm<1>() + nu_in.lpNorm<1>() +
                                     zL_in.lpNorm<1>() + zU_in.lpNorm<1>();
            const double s_d =
                clamp_min(sum_mults / clamp_min(denom_ct, 1), s_max) / s_max;
            const double s_c =
                clamp_min((zL_in.lpNorm<1>() + zU_in.lpNorm<1>()) /
                              clamp_min(n_, 1),
                          s_max) /
                s_max;
            const double comp_box =
                detail::comp_inf_norm(s_in, lam_in, zL_in, zU_in, B_in, mu_in,
                                      use_shifted_, tau_shift_, bound_shift_);
            return std::max({safe_inf_norm(r_d) / s_d,
                             (mE_ > 0) ? safe_inf_norm(E.cE) : 0.0,
                             (mI_ > 0) ? safe_inf_norm(E.cI) : 0.0,
                             comp_box / s_c});
        };

        auto assemble_W = [&](const spmat &H_in, const dvec &Sigma_x,
                              const std::optional<spmat> &JI_opt,
                              const dvec &Sigma_s_opt) -> spmat {
            spmat W = H_in;
            if (Sigma_x.size())
                W.diagonal().array() += Sigma_x.array();
            if (JI_opt && JI_opt->rows() > 0 &&
                Sigma_s_opt.size() == JI_opt->rows()) {
                spmat JIc = *JI_opt;
                W += JIc.transpose() * Sigma_s_opt.asDiagonal() * JIc;
                W.makeCompressed();
            }
            return W;
        };

        // -------------------- Trust region helpers --------------------
        auto step_norm_W = [&](const dvec &dx, const dvec &ds, const Sigmas &Sg,
                               dvec &Wdx_scratch) -> double {
            Wdx_scratch.setZero(dx.size());
            m_->getCompiledWOp().perform_op(dx, Wdx_scratch);
            const double qx = dx.dot(Wdx_scratch);
            const double qs = (Sg.Sigma_s.size() && ds.size())
                                  ? ds.cwiseProduct(Sg.Sigma_s).dot(ds)
                                  : 0.0;
            return std::sqrt(std::max(0.0, qx + qs));
        };

        auto scale_full_step = [&](double sc, dvec &dx, dvec &dnu, dvec &ds,
                                   dvec &dlam, dvec &dzL, dvec &dzU) {
            if (sc >= 1.0)
                return;
            dx *= sc;
            if (dnu.size())
                dnu *= sc;
            if (ds.size())
                ds *= sc;
            if (dlam.size())
                dlam *= sc;
            dzL *= sc;
            dzU *= sc;
        };

        auto barrier_sum = [&](const dvec &sv) {
            constexpr double eps = 1e-12;
            double v = 0.0;
            for (int i = 0; i < sv.size(); ++i)
                v += std::log(std::max(sv[i], eps));
            return v;
        };

        // -------------------- Adaptive Cubic Penalty Management
        // --------------------
        auto update_penalty_parameters = [&](double theta_current,
                                             double theta_reduction_rate,
                                             double mu_current, int iteration) {
            const double rho_min =
                get_attr_or<double>(cfg_, "ip_penalty_rho_min", 1e-6);
            const double rho_max =
                get_attr_or<double>(cfg_, "ip_penalty_rho_max", 1e6);
            const double sigma_min =
                get_attr_or<double>(cfg_, "ip_penalty_sigma_min", 1e-8);
            const double sigma_max =
                get_attr_or<double>(cfg_, "ip_penalty_sigma_max", 1e3);

            // Track constraint violation history for trend analysis
            theta_history_.push_back(theta_current);
            if (theta_history_.size() > 5)
                theta_history_.pop_front();

            // Compute average reduction rate over recent history
            double avg_reduction = 0.0;
            if (theta_history_.size() >= 3) {
                for (size_t i = 1; i < theta_history_.size(); ++i) {
                    if (theta_history_[i - 1] > 1e-16) {
                        avg_reduction +=
                            (theta_history_[i - 1] - theta_history_[i]) /
                            theta_history_[i - 1];
                    }
                }
                avg_reduction /= (theta_history_.size() - 1);
            }

            // Adaptive linear penalty parameter ρ
            if (avg_reduction < 0.05 && theta_current > 1e-5) {
                // Slow constraint progress - increase linear penalty
                penalty_rho_ = std::min(penalty_rho_ * 1.5, rho_max);
            } else if (avg_reduction > 0.3 && theta_current < 1e-7) {
                // Fast constraint progress - can reduce penalty
                penalty_rho_ = std::max(penalty_rho_ * 0.9, rho_min);
            }

            // Adaptive cubic penalty parameter σ
            const double theta_threshold =
                get_attr_or<double>(cfg_, "ip_theta_cubic_threshold", 1e-2);
            if (theta_current > theta_threshold) {
                // Significant violations - increase cubic penalty for
                // superlinear effect
                penalty_sigma_ = std::min(penalty_sigma_ * 1.2, sigma_max);
            } else if (theta_current < theta_threshold * 0.01) {
                // Very small violations - reduce cubic penalty to avoid
                // over-penalization
                penalty_sigma_ = std::max(penalty_sigma_ * 0.95, sigma_min);
            }

            // Keep penalty parameters proportional to barrier parameter
            const double min_rho_relative =
                get_attr_or<double>(cfg_, "ip_penalty_rho_relative", 10.0);
            penalty_rho_ =
                std::max(penalty_rho_, min_rho_relative * mu_current);

            last_theta_ = theta_current;
        };

        // -------------------- Cubic Penalty Merit Function (for TR only)
        // --------------------
        auto compute_cubic_merit = [&](double f_val, const dvec &s_val,
                                       double mu_val,
                                       double theta_val) -> double {
            const double barrier_term = mu_val * barrier_sum(s_val);
            const double linear_penalty = penalty_rho_ * theta_val;
            const double cubic_penalty =
                penalty_sigma_ * std::pow(theta_val, 3.0);

            return f_val - barrier_term + linear_penalty + cubic_penalty;
        };

        // -------------------- Enhanced Cubic Predicted Reduction
        // --------------------
        auto predicted_reduction_cubic =
            [&](const dvec &dx, const dvec &r_d, dvec &Wdx_scratch,
                const Sigmas &Sg, const dvec &s, const dvec &ds, double mu,
                const EvalPack &E) -> double {
            // Standard quadratic model terms
            m_->getCompiledWOp().perform_op(dx, Wdx_scratch);
            const double q = 0.5 * dx.dot(Wdx_scratch);
            const double lin = r_d.size() ? r_d.dot(dx) : 0.0;

            // Barrier change prediction
            double dphi = 0.0;
            if (s.size() && ds.size()) {
                constexpr double eps = 1e-12;
                for (int i = 0; i < s.size(); ++i) {
                    const double si = std::max(s[i], eps);
                    const double sip = std::max(s[i] + ds[i], eps);
                    dphi += std::log(si) - std::log(sip);
                }
            }

            // Constraint linearization for penalty terms
            double theta_pred_linear = 0.0;

            if (mE > 0 && E.cE.size() && E.JE.nonZeros()) {
                dvec JE_dx = E.JE * dx;
                theta_pred_linear += E.cE.dot(JE_dx);
            }

            if (mI > 0 && E.cI.size() && E.JI.nonZeros()) {
                dvec JI_dx = E.JI * dx;
                for (int i = 0; i < E.cI.size(); ++i) {
                    const double ci_new = E.cI[i] + JI_dx[i];
                    if (ci_new > 0) { // Only positive violations contribute
                        theta_pred_linear += ci_new - std::max(0.0, E.cI[i]);
                    }
                }
            }

            // Predicted new constraint violation
            const double theta_new = std::max(0.0, E.theta + theta_pred_linear);

            // Linear penalty contribution
            const double linear_penalty_change =
                penalty_rho_ * theta_pred_linear;

            // Cubic penalty contribution
            const double cubic_penalty_old =
                penalty_sigma_ * std::pow(E.theta, 3.0);
            const double cubic_penalty_new =
                penalty_sigma_ * std::pow(theta_new, 3.0);
            const double cubic_penalty_change =
                cubic_penalty_new - cubic_penalty_old;

            // Total predicted reduction: -(linear + quadratic) -
            // mu*barrier_change - penalty_changes
            return -(lin + q) - mu * dphi - linear_penalty_change -
                   cubic_penalty_change;
        };

        // ------------------------------------------------------------------------

        // -------------------- Evaluate model at x --------------------
        EvalPack E = eval_all_at(x, n, mI, mE);
        IP_LAP("eval_all(x)");

        // Update penalty parameters based on current progress
        const double theta_reduction_rate =
            (last_theta_ > 1e-16) ? (last_theta_ - E.theta) / last_theta_ : 0.0;
        update_penalty_parameters(E.theta, theta_reduction_rate, mu, it);

        Bounds B = get_bounds(x);
        IP_LAP("bounds(get)");

        double tau_shift =
            use_shifted ? shift_mgr_->compute_slack_shift(s, E.cI, lmb, it)
                        : 0.0;
        st.tau_shift = tau_shift;
        double bound_shift =
            use_shifted ? shift_mgr_->compute_bound_shift(x, B, it) : 0.0;
        IP_LAP("bounds & shifts (A)");

        const double tol = get_attr_or<double>(cfg_, "tol", 1e-8);

        // -------------------- Enhanced convergence check --------------------
        const double err_prev =
            aggregated_error(E, x, lmb, nuv, zL, zU, mu, s, B, n, mE, mI,
                             use_shifted, tau_shift, bound_shift);
        IP_LAP("compute_error_(x)");

        // Check penalty-aware convergence
        const bool penalty_converged =
            (penalty_rho_ * E.theta < tol) &&
            (penalty_sigma_ * std::pow(E.theta, 3.0) < tol);

        if (err_prev <= tol && penalty_converged) {
            SolverInfo info;
            info.mode = "ip";
            info.step_norm = 0.0;
            info.accepted = true;
            info.converged = true;
            info.f = E.f;
            info.theta = E.theta;
            info.stat = safe_inf_norm(E.g);
            info.ineq = (mI > 0)
                            ? safe_inf_norm((E.cI.array().max(0.0)).matrix())
                            : 0.0;
            info.eq = (mE > 0) ? safe_inf_norm(E.cE) : 0.0;
            info.comp =
                detail::complementarity(s, lmb, mu, tau_shift, use_shifted);
            info.ls_iters = 0;
            info.alpha = 0.0;
            info.rho = 0.0;
            info.tr_radius = tr_radius_;
            info.mu = mu;
            info.penalty_rho = penalty_rho_;
            info.penalty_sigma = penalty_sigma_;
            return {x, lmb, nuv, info};
        }

        // -------------------- Sigmas & Hessian --------------------
        const double eps_abs = get_attr_or<double>(cfg_, "sigma_eps_abs", 1e-8);
        const double cap_add = get_attr_or<double>(cfg_, "sigma_cap", 1e8);
        Sigmas Sg =
            detail::build_sigmas(zL, zU, B, lmb, s, E.cI, tau_shift,
                                 bound_shift, use_shifted, eps_abs, cap_add);
        IP_LAP("build_sigmas");

        auto H0 = m_->hess(x, lmb, nuv);
        IP_LAP("get Hessian");
        auto [H, reg_info] = regularizer_->regularize(H0, it);
        H.makeCompressed();
        IP_LAP("regularize(H)");

        // Compile W operator for fast products
        m_->compileWOp(
            Sg.Sigma_x, (mI > 0) ? std::optional<spmat>(E.JI) : std::nullopt,
            (mI > 0) ? std::optional<dvec>(Sg.Sigma_s) : std::nullopt);

        spmat W =
            assemble_W(H, Sg.Sigma_x,
                       (mI > 0 && E.JI.nonZeros()) ? std::optional<spmat>(E.JI)
                                                   : std::nullopt,
                       Sg.Sigma_s);
        IP_LAP("assemble W");

        // -------------------- Residuals --------------------
        dvec r_d = build_r_d(E, lmb, nuv, zL, zU);
        dvec r_pI = (mI > 0) ? (E.cI + s) : dvec();
        IP_LAP("build residuals");

        std::optional<spmat> JE_eff = (mE > 0 && E.JE.nonZeros())
                                          ? std::optional<spmat>(E.JE)
                                          : std::nullopt;

        dvec r_pE = (mE > 0) ? E.cE : dvec();

        if (mE > 0) {
            if (auto d_rE = shift_mgr_->compute_equality_shift(
                    JE_eff,
                    (E.cE.size() ? std::optional<dvec>(E.cE) : std::nullopt))) {
                if (d_rE->size() == r_pE.size()) {
                    r_pE.noalias() += *d_rE;
                }
            }
        }

        m_->set_params(Sg.Sigma_x, (mI > 0) ? std::optional<dvec>(Sg.Sigma_s)
                                                  : std::nullopt, mu, E.theta, it);
        // -------------------- Solve for step --------------------
        auto [alpha_aff, mu_aff, sigma_pred, step] = mg_solver_->solve(
            W, r_d, JE_eff, r_pE,
            (mI > 0 && E.JI.nonZeros()) ? std::optional<spmat>(E.JI)
                                        : std::nullopt,
            r_pI, s, lmb, zL, zU, B, use_shifted, tau_shift, bound_shift, mu,
            E.theta, Sg);

        dvec dx = step.dx;
        dvec dnu =
            (mE > 0 && step.dnu.size() == mE) ? step.dnu : dvec::Zero(mE);
        dvec ds = step.ds;
        dvec dlam = step.dlam;
        dvec dzL = step.dzL;
        dvec dzU = step.dzU;

        mu = step.mu_target;
        IP_LAP("solve step");

        // -------------------- Cubic Penalty Trust Region Management
        // --------------------
        static int tr_clip_streak = 0;
        static int tr_good_streak = 0;

        dvec Wdx_scratch(dx.size());
        const double stepW =
            step_norm_W(dx, (mI ? ds : dvec()), Sg, Wdx_scratch);

        double Delta = tr_radius_;
        double sc_TR = 1.0;
        bool was_clipped = false;

        // Enhanced step quality assessment
        const double min_step_norm =
            get_attr_or<double>(cfg_, "ip_min_step_norm", 1e-15);
        const double step_quality_ratio = stepW / std::max(Delta, 1e-12);

        if (stepW < min_step_norm) {
            // Degenerate step - moderate shrinkage
            tr_radius_ *= 0.75;
            tr_ema_ = tr_radius_;
            tr_clip_streak = 0;
            tr_good_streak = 0;
        } else if (stepW > Delta && stepW > 0.0) {
            sc_TR = Delta / stepW;
            scale_full_step(sc_TR, dx, dnu, ds, dlam, dzL, dzU);
            was_clipped = true;
            tr_clip_streak++;
            tr_good_streak = 0;
        } else {
            tr_clip_streak = std::max(0, tr_clip_streak - 1);
            if (step_quality_ratio > 0.7) {
                tr_good_streak++;
            } else {
                tr_good_streak = std::max(0, tr_good_streak - 1);
            }
        }
        IP_LAP("cubic penalty TR management");

        // -------------------- Fraction-to-boundary + Funnel Line Search
        // --------------------
        const double tau_pri = get_attr_or<double>(
            cfg_, "ip_tau_pri", get_attr_or<double>(cfg_, "ip_tau", 0.995));
        const double tau_dual = get_attr_or<double>(
            cfg_, "ip_tau_dual", get_attr_or<double>(cfg_, "ip_tau", 0.995));

        const double a_ftb = detail::alpha_ftb_vec(
            x, dx, (mI ? s : dvec()), (mI ? ds : dvec()), lmb,
            (mI ? dlam : dvec()), B, tau_pri, tau_dual);

        const double alpha_max =
            std::min(a_ftb, get_attr_or<double>(cfg_, "ip_alpha_max", 1.0));
        double alpha = std::min(1.0, alpha_max);
        int ls_iters = 0;
        bool needs_restoration = false;

        // Use your existing funnel line search (unchanged)
        auto ls_res =
            ls_->search(m_, x, dx, (mI ? s : dvec()), (mI ? ds : dvec()), mu,
                        E.g.dot(dx), E.theta, alpha_max);
        alpha = std::get<0>(ls_res);
        ls_iters = std::get<1>(ls_res);
        needs_restoration = std::get<2>(ls_res);
        IP_LAP("funnel line search");

        // -------------------- Restoration path --------------------
        const double ls_min_alpha = get_attr_or<double>(
            cfg_, "ls_min_alpha",
            get_attr_or<double>(cfg_, "ip_alpha_min", 1e-10));

        if (alpha <= ls_min_alpha && needs_restoration) {
            dvec dxf = -E.g;
            const double ng = dxf.norm();
            if (ng > tr_ema_ && ng > 0.0)
                dxf *= (tr_ema_ / ng);
            if (ng > 0)
                dxf /= ng;
            const double a_safe = std::min(alpha_max, 1e-2);
            dvec x_new = x + a_safe * dxf;

            // Shrink TR on restoration
            tr_radius_ *= get_attr_or<double>(cfg_, "ip_tr_gamma_shrink", 0.5);
            tr_ema_ = tr_radius_;

            SolverInfo info;
            info.mode = "ip_restoration";
            info.step_norm = (x_new - x).norm();
            info.accepted = true;
            info.converged = false;
            info.f = 0.0;
            info.theta = m_->constraint_violation(x_new);
            info.stat = 0.0;
            info.ineq = 0.0;
            info.eq = 0.0;
            info.comp = 0.0;
            info.ls_iters = ls_iters;
            info.alpha = 0.0;
            info.rho = 0.0;
            info.tr_radius = tr_radius_;
            info.mu = mu;
            info.penalty_rho = penalty_rho_;
            info.penalty_sigma = penalty_sigma_;
            return {x_new, lmb, nuv, info};
        }

        // -------------------- Accept step & evaluate new point
        // --------------------
        dvec x_new = x + alpha * dx;
        dvec s_new = mI ? (s + alpha * ds) : s;
        dvec lmb_new = mI ? (lmb + alpha * dlam) : lmb;
        dvec nu_new = mE ? (nuv + alpha * dnu) : nuv;
        dvec zL_new = zL + alpha * dzL;
        dvec zU_new = zU + alpha * dzU;

        Bounds Bn = get_bounds(x_new);
        detail::cap_bound_duals_sigma_box(zL_new, zU_new, Bn, use_shifted,
                                          bound_shift, mu, 1e10);
        IP_LAP("apply step & cap");

        EvalPack En = eval_all_at(x_new, n, mI, mE);
        IP_LAP("eval_all(x_new)");

        // -------------------- KKT residuals (new) --------------------
        dvec r_d_new = build_r_d(En, lmb_new, nu_new, zL_new, zU_new);
        KKT kkt_new;
        kkt_new.stat = safe_inf_norm(r_d_new);
        kkt_new.ineq =
            (mI > 0) ? safe_inf_norm((En.cI.array().max(0.0)).matrix()) : 0.0;
        kkt_new.eq = (mE > 0) ? safe_inf_norm(En.cE) : 0.0;
        kkt_new.comp =
            detail::comp_inf_norm(s_new, lmb_new, zL_new, zU_new, Bn, mu,
                                  use_shifted, tau_shift, bound_shift);

        const double tol_outer = tol;
        const bool converged =
            (kkt_new.stat <= tol_outer && kkt_new.ineq <= tol_outer &&
             kkt_new.eq <= tol_outer && kkt_new.comp <= tol_outer &&
             mu <= tol_outer / 10.0);
        IP_LAP("KKT new");

        // -------------------- Cubic Penalty Trust Region Update
        // -------------------- Predicted reduction using cubic penalty model
        double pred = predicted_reduction_cubic(dx, r_d, Wdx_scratch, Sg, s,
                                                (mI ? ds : dvec()), mu, E);

        // Actual reduction using cubic penalty merit function
        double merit_old = compute_cubic_merit(E.f, s, mu, E.theta);
        double merit_new = compute_cubic_merit(En.f, s_new, mu, En.theta);
        double ared = merit_old - merit_new;

        // Robust ratio calculation
        const double pred_threshold =
            get_attr_or<double>(cfg_, "ip_pred_threshold", 1e-16);
        double rho;
        if (std::abs(pred) < pred_threshold) {
            // Use constraint progress as fallback indicator
            const double theta_progress =
                (E.theta > 1e-16)
                    ? std::max(0.0, (E.theta - En.theta) / E.theta)
                    : 1.0;
            rho = (ared > 0 && alpha > 0.1)
                      ? std::min(0.9, 0.5 + theta_progress)
                      : 0.1;
        } else {
            rho = ared / pred;
        }

        // Adaptive trust region parameters based on penalty state
        const double eta_excellent =
            get_attr_or<double>(cfg_, "ip_tr_eta_excellent", 0.9);
        const double eta_good =
            get_attr_or<double>(cfg_, "ip_tr_eta_good", 0.7);
        const double eta_poor =
            get_attr_or<double>(cfg_, "ip_tr_eta_poor", 0.15);

        // Adaptive expansion/contraction based on constraint violation level
        double gam_expand =
            get_attr_or<double>(cfg_, "ip_tr_gamma_expand", 2.0);
        double gam_shrink =
            get_attr_or<double>(cfg_, "ip_tr_gamma_shrink", 0.6);

        // Be more aggressive when constraints are well-satisfied
        if (E.theta < 1e-6) {
            gam_expand *= 1.2; // More aggressive expansion in feasible region
        } else if (E.theta > 1e-3) {
            gam_shrink *=
                0.8; // More aggressive shrinkage when highly infeasible
        }

        const double beta = get_attr_or<double>(cfg_, "ip_tr_beta", 0.25);

        double tr_new = tr_radius_;
        const bool near_full_alpha = (alpha > 0.9 * alpha_max);
        const bool unclipped = (sc_TR >= 0.999);
        const bool good_constraint_progress = (En.theta < 0.9 * E.theta);

        // Enhanced trust region update logic
        if (rho > eta_excellent && near_full_alpha && unclipped &&
            tr_good_streak >= 2) {
            // Excellent step with consistent good performance
            tr_new *= gam_expand;
            tr_good_streak++;
        } else if (rho > eta_good && near_full_alpha &&
                   good_constraint_progress) {
            // Good step with constraint progress
            tr_new *= std::sqrt(gam_expand);
            tr_good_streak++;
        } else if (rho < eta_poor || alpha <= ls_min_alpha ||
                   tr_clip_streak >= 3) {
            // Poor step or persistent clipping
            tr_new *= gam_shrink;
            tr_good_streak = 0;
        } else if (tr_clip_streak >= 2 && !good_constraint_progress) {
            // Multiple clips without constraint progress
            tr_new *= std::sqrt(gam_shrink);
            tr_good_streak = 0;
        }
        // else: maintain current radius

        const double tr_min = get_attr_or<double>(cfg_, "ip_tr_min", 1e-8);
        const double tr_max = get_attr_or<double>(cfg_, "ip_tr_max", 1e6);
        tr_new = std::clamp(tr_new, tr_min, tr_max);

        // Adaptive EMA smoothing - faster response when penalties are active
        double adaptive_beta = beta;
        if (penalty_rho_ * E.theta > 1e-4 ||
            penalty_sigma_ * std::pow(E.theta, 3.0) > 1e-6) {
            adaptive_beta *=
                1.5; // Faster adaptation when penalties are significant
        }
        adaptive_beta = std::min(adaptive_beta, 0.8);

        tr_ema_ = (1.0 - adaptive_beta) * tr_ema_ + adaptive_beta * tr_new;
        tr_radius_ = tr_ema_;

        // Update measured μ for next iteration
        const double mu_next_meas = mg_solver_->average_complementarity(
            s_new, lmb_new, zL_new, zU_new, Bn, use_shifted, tau_shift,
            bound_shift);
        st.mu = std::max(mu_next_meas,
                         get_attr_or<double>(cfg_, "ip_mu_min", 1e-12));

        // -------------------- Pack & update state --------------------
        SolverInfo info;
        info.mode = "ip";
        info.step_norm = (x_new - x).norm();
        info.accepted = true;
        info.converged = converged;
        info.f = En.f;
        info.theta = En.theta;
        info.stat = kkt_new.stat;
        info.ineq = kkt_new.ineq;
        info.eq = kkt_new.eq;
        info.comp = kkt_new.comp;
        info.ls_iters = ls_iters;
        info.alpha = alpha;
        info.rho = rho;
        info.tr_radius = tr_radius_;
        info.mu = st.mu;
        info.shifted_barrier = use_shifted;
        info.tau_shift = tau_shift;
        info.bound_shift = bound_shift;

        // Enhanced diagnostics for cubic penalty method
        info.penalty_rho = penalty_rho_;
        info.penalty_sigma = penalty_sigma_;
        info.step_quality_ratio = stepW / std::max(Delta, 1e-12);
        info.was_clipped = was_clipped;
        info.clip_streak = tr_clip_streak;
        info.good_streak = tr_good_streak;
        info.theta_reduction =
            (E.theta > 1e-16) ? (E.theta - En.theta) / E.theta : 0.0;

        st.s = std::move(s_new);
        st.lam = std::move(lmb_new);
        st.nu = std::move(nu_new);
        st.zL = std::move(zL_new);
        st.zU = std::move(zU_new);
        st.mu = mu;
        st.tau_shift = tau_shift;

        return {x_new, st.lam, st.nu, info};
    }

    double tr_radius_ = 0.0; // Current radius
    double tr_ema_ = 0.0;    // EMA for smoothing
    bool tr_inited_ = false;

private:
    nb::object cfg_, hess_;
    std::unordered_map<std::string, dvec> kkt_cache_;
    std::shared_ptr<kkt::KKTReusable> cached_kkt_solver_{};
    spmat cached_kkt_matrix_;
    bool kkt_factorization_valid_ = false;

    bool matrices_equal(const spmat &A, const spmat &B, double tol = 1e-4) {
        if (A.rows() != B.rows() || A.cols() != B.cols())
            return false;
        return (A - B).norm() < tol;
    }

    void load_defaults_() {
        auto set_if_missing = [&](const char *name, nb::object v) {
            if (!pyu::has_attr(cfg_, name))
                cfg_.attr(name) = v;
        };
        set_if_missing("ip_exact_hessian", nb::bool_(true));
        set_if_missing("ip_hess_reg0", nb::float_(1e-4));
        set_if_missing("ip_eq_reg", nb::float_(1e-4));
        set_if_missing("ip_use_shifted_barrier", nb::bool_(true));
        set_if_missing("ip_shift_tau", nb::float_(0.01));
        set_if_missing("ip_shift_bounds", nb::float_(0.1));
        set_if_missing("ip_shift_adaptive", nb::bool_(true));
        set_if_missing("ip_mu_init", nb::float_(1e-2));
        set_if_missing("ip_mu_min", nb::float_(1e-12));
        set_if_missing("ip_sigma_power", nb::float_(3.0));
        set_if_missing("ip_tau_pri", nb::float_(0.995));
        set_if_missing("ip_tau_dual", nb::float_(0.99));
        set_if_missing("ip_tau", nb::float_(0.995));
        set_if_missing("ip_alpha_max", nb::float_(1.0));
        set_if_missing("ip_dx_max", nb::float_(1e3));
        set_if_missing("ip_theta_clip", nb::float_(1e-2));
        set_if_missing("sigma_eps_abs", nb::float_(1e-8));
        set_if_missing("sigma_cap", nb::float_(1e8));
        set_if_missing("ip_kkt_method", nb::str("hykkt"));
        set_if_missing("tol", nb::float_(1e-6));
        set_if_missing("ls_backtrack", nb::float_(get_attr_or<double>(
                                           cfg_, "ip_alpha_backtrack", 0.5)));
        set_if_missing("ls_armijo_f", nb::float_(get_attr_or<double>(
                                          cfg_, "ip_armijo_coeff", 1e-4)));
        set_if_missing("ls_max_iter",
                       nb::int_(get_attr_or<int>(cfg_, "ip_ls_max", 5)));
        set_if_missing("ls_min_alpha", nb::float_(get_attr_or<double>(
                                           cfg_, "ip_alpha_min", 1e-10)));
    }

    // --- State bootstrap ---
    [[nodiscard]] IPState state_from_model_(const dvec &x) {
        IPState s{};
        s.mI = m_->get_mI();
        s.mE = m_->get_mE();

        m_->eval_all(x, std::vector<std::string>{"cI", "cE"});
        dvec cI = (s.mI > 0) ? m_->get_cI().value() : dvec::Zero(s.mI);

        const double mu0 =
            clamp_min(get_attr_or<double>(cfg_, "ip_mu_init", 1e-2), 1e-12);
        const bool use_shifted =
            get_attr_or<bool>(cfg_, "ip_use_shifted_barrier", true);
        const double tau_shift =
            use_shifted ? get_attr_or<double>(cfg_, "ip_shift_tau", 0.1) : 0.0;
        const double bound_shift =
            use_shifted ? get_attr_or<double>(cfg_, "ip_shift_bounds", 0.1)
                        : 0.0;

        if (s.mI > 0) {
            s.s.resize(s.mI);
            s.lam.resize(s.mI);
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

        Bounds B = get_bounds(x);
        s.zL = dvec::Zero(x.size());
        s.zU = dvec::Zero(x.size());
        for (int i = 0; i < x.size(); ++i) {
            if (B.hasL[i])
                s.zL[i] = clamp_min(
                    mu0 / clamp_min(B.sL[i] + bound_shift, 1e-12), 1e-8);
            if (B.hasU[i])
                s.zU[i] = clamp_min(
                    mu0 / clamp_min(B.sU[i] + bound_shift, 1e-12), 1e-8);
        }

        s.mu = mu0;
        s.tau_shift = tau_shift;
        s.initialized = true;
        return s;
    }

    // --- KKT solve (with reuse) ---
    [[nodiscard]] KKTResult solve_KKT_(const spmat &W, const dvec &rhs_x,
                                       const std::optional<spmat> &JE,
                                       const std::optional<dvec> &rpE,
                                       std::string_view method_in) {
        const int mE = (JE && JE->rows() > 0) ? JE->rows() : 0;

        kkt::dvec r1 = rhs_x;
        std::optional<kkt::dvec> r2;
        if (mE > 0)
            r2 = rpE ? (-(*rpE)).eval() : kkt::dvec::Zero(mE);

        std::string method_cpp = std::string(method_in);
        if (mE == 0 && (method_cpp == "hykkt" || method_cpp == "hykkt_cholmod"))
            method_cpp = "ldl";

        const bool can_reuse = kkt_factorization_valid_ && cached_kkt_solver_ &&
                               matrices_equal(W, cached_kkt_matrix_);
        if (can_reuse) {
            auto [dx, dy] = cached_kkt_solver_->solve(rhs_x, r2, 1e-8, 200);
            return {std::move(dx), std::move(dy), cached_kkt_solver_};
        }

        kkt::ChompConfig conf;
        auto &reg = kkt::default_registry();
        auto strat = reg.get(method_cpp);
        auto [dx, dy, reusable] = strat->factor_and_solve(
            m_, W, (mE > 0 ? std::optional<spmat>(*JE) : std::nullopt), r1, r2,
            conf, std::nullopt, kkt_cache_, 0.0, std::nullopt, true, true);

        cached_kkt_matrix_ = W;
        cached_kkt_solver_ = reusable;
        kkt_factorization_valid_ = true;
        return {std::move(dx), std::move(dy), std::move(reusable)};
    }
};
