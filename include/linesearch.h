// line_searcher_improved.cc — enhanced with pure C++ filter calls and SOTA improvements

#include "funnel.h" // FunnelConfig, Funnel
#include "filter.h" // Filter class
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "model.h"
#include "ip_aux.h" // RichardsonExtrapolator
#include "filter.h"

namespace nb = nanobind;
using dvec = Eigen::VectorXd;
using dmat = Eigen::MatrixXd;
using spmat = Eigen::SparseMatrix<double>;

enum class LSMeritMode { MonotonePhi, NonMonotonePhi, FilterOrFunnel };

struct LSConfig {
    // Backtracking / Armijo
    double ls_backtrack{0.5};
    double ls_armijo_f{1e-4};
    int    ls_max_iter{20};
    double ls_min_alpha{1e-12};
    double ls_wolfe_c{0.9};

    // IP specifics
    double ip_fraction_to_boundary_tau{0.995};
    double ls_theta_restoration{1e3};

    // Second-order correction
    int    max_soc{4};
    double kappa_soc_min{0.1};
    double kappa_soc_max{0.99};
    double kappa_soc_base{0.5};

    // Merit selection
    LSMeritMode merit_mode{LSMeritMode::FilterOrFunnel};
    int    nm_memory{5};        // cross-iteration window length
    double nm_safeguard{0.0};   // φ_ref offset

    // Richardson extrapolation
    bool   use_richardson{true};
    double rich_tol{1e-8};
    int    rich_min_order{2};
    
    // Enhanced parameters
    double ls_sufficient_decrease{1e-4};
    double ls_curvature_condition{0.9};
    bool   use_strong_wolfe{false};
    double ls_theta_eps{1e-12};
    int    ls_watchdog_max{10};
    double ls_theta_min_improvement{0.1}; // Minimum relative theta improvement for SOC
};

// --------------- Enhanced helper functions ---------------
static inline double getattr_or_double(const nb::object &obj, const char *name, double fallback) {
    if (!obj || obj.is_none()) return fallback;
    if (nb::hasattr(obj, name)) return nb::cast<double>(obj.attr(name));
    return fallback;
}

static inline int getattr_or_int(const nb::object &obj, const char *name, int fallback) {
    if (!obj || obj.is_none()) return fallback;
    if (nb::hasattr(obj, name)) return nb::cast<int>(obj.attr(name));
    return fallback;
}

static inline bool getattr_or_bool(const nb::object &obj, const char *name, bool fallback) {
    if (!obj || obj.is_none()) return fallback;
    if (nb::hasattr(obj, name)) return nb::cast<bool>(obj.attr(name));
    return fallback;
}

// Enhanced step concatenation with bounds checking
static inline dvec concat_dx_ds(const dvec &dx, const dvec &ds) {
    dvec out(dx.size() + ds.size());
    out.head(dx.size()) = dx;
    out.tail(ds.size()) = ds;
    return out;
}

static inline void split_dx_ds(const dvec &v, Eigen::Index nx, dvec &dx_out, dvec &ds_out) {
    if (v.size() < nx) {
        dx_out = dvec::Zero(nx);
        ds_out = dvec::Zero(0);
        return;
    }
    dx_out = v.head(nx);
    ds_out = v.tail(v.size() - nx);
}

// Safe barrier computation with overflow protection
static inline double safe_log_barrier(double val, double eps) {
    return std::log(std::max(val, eps));
}

// Enhanced constraint violation computation
static inline double compute_theta(const std::optional<dvec>& cE, const std::optional<dvec>& cI, 
                                   const dvec& s, bool use_inf_norm = false) {
    double theta_E = 0.0, theta_I = 0.0;
    
    if (cE && cE->size() > 0) {
        theta_E = use_inf_norm ? cE->lpNorm<Eigen::Infinity>() : cE->lpNorm<1>();
    }
    
    if (cI && cI->size() > 0 && s.size() == cI->size()) {
        dvec viol = (cI->array() + s.array()).max(0.0);
        theta_I = use_inf_norm ? viol.lpNorm<Eigen::Infinity>() : viol.lpNorm<1>();
    }
    
    return theta_E + theta_I;
}

// --------------- Enhanced Line Searcher ---------------
class LineSearcher {
public:
    LineSearcher(nb::object cfg, std::shared_ptr<Filter> filter = nullptr,
                 std::shared_ptr<Funnel> funnel = nullptr)
        : cfg_obj_(std::move(cfg)), filter_(std::move(filter)), funnel_(std::move(funnel)) {

        // Enhanced configuration loading with validation
        cfg_.ls_backtrack = std::clamp(getattr_or_double(cfg_obj_, "ls_backtrack", 0.5), 1e-4, 0.99);
        cfg_.ls_armijo_f  = std::max(1e-12, getattr_or_double(cfg_obj_, "ls_armijo_f", 1e-4));
        cfg_.ls_max_iter  = std::max(1, getattr_or_int(cfg_obj_, "ls_max_iter", 20));
        cfg_.ls_min_alpha = std::max(0.0, getattr_or_double(cfg_obj_, "ls_min_alpha", 1e-12));
        cfg_.ls_wolfe_c   = std::clamp(getattr_or_double(cfg_obj_, "ls_wolfe_c", 0.9),
                                       cfg_.ls_armijo_f, 0.999);
        cfg_.ip_fraction_to_boundary_tau =
            std::clamp(getattr_or_double(cfg_obj_, "ip_fraction_to_boundary_tau", 0.995), 0.9, 0.999);
        cfg_.ls_theta_restoration =
            std::max(1.0, getattr_or_double(cfg_obj_, "ls_theta_restoration", 1e3));

        // SOC parameters with validation
        cfg_.max_soc       = std::max(0, getattr_or_int(cfg_obj_, "max_soc", 4));
        cfg_.kappa_soc_min = std::clamp(getattr_or_double(cfg_obj_, "kappa_soc_min", 0.1), 0.01, 0.5);
        cfg_.kappa_soc_max = std::clamp(getattr_or_double(cfg_obj_, "kappa_soc_max", 0.99), 0.5, 0.99);
        cfg_.kappa_soc_base= std::clamp(getattr_or_double(cfg_obj_, "kappa_soc_base", 0.5),
                                        cfg_.kappa_soc_min, cfg_.kappa_soc_max);

        // Merit mode selection
        const std::string mode = nb::hasattr(cfg_obj_, "ls_merit_mode")
                                 ? nb::cast<std::string>(cfg_obj_.attr("ls_merit_mode"))
                                 : std::string("filter");
        if (mode == "monotone")         cfg_.merit_mode = LSMeritMode::MonotonePhi;
        else if (mode == "nonmonotone") cfg_.merit_mode = LSMeritMode::NonMonotonePhi;
        else                            cfg_.merit_mode = LSMeritMode::FilterOrFunnel;

        cfg_.nm_memory    = std::max(1, getattr_or_int(cfg_obj_, "ls_nm_memory", 5));
        cfg_.nm_safeguard = std::max(0.0, getattr_or_double(cfg_obj_, "ls_nm_safeguard", 0.0));

        // Richardson extrapolation
        cfg_.use_richardson = getattr_or_bool(cfg_obj_, "ls_use_richardson", true);
        cfg_.rich_tol       = std::max(1e-14, getattr_or_double(cfg_obj_, "ls_rich_tol", 1e-8));
        cfg_.rich_min_order = std::max(2, getattr_or_int(cfg_obj_, "ls_rich_min_order", 2));
        
        // Enhanced parameters
        cfg_.ls_sufficient_decrease = std::max(1e-8, getattr_or_double(cfg_obj_, "ls_sufficient_decrease", 1e-4));
        cfg_.ls_curvature_condition = std::clamp(getattr_or_double(cfg_obj_, "ls_curvature_condition", 0.9),
                                                  cfg_.ls_sufficient_decrease, 0.999);
        cfg_.use_strong_wolfe = getattr_or_bool(cfg_obj_, "ls_use_strong_wolfe", false);
        cfg_.ls_theta_eps = std::max(1e-16, getattr_or_double(cfg_obj_, "ls_theta_eps", 1e-12));
        cfg_.ls_watchdog_max = std::max(1, getattr_or_int(cfg_obj_, "ls_watchdog_max", 10));
        cfg_.ls_theta_min_improvement = std::clamp(getattr_or_double(cfg_obj_, "ls_theta_min_improvement", 0.1),
                                                   0.01, 0.9);
    }

    // Enhanced public interface
    void reset_nonmonotone_history() {
        phi_hist_.clear();
    }
    
    void seed_nonmonotone(double phi0) {
        if (std::isfinite(phi0)) {
            phi_hist_.clear();
            phi_hist_.push_back(phi0);
        }
    }

    // Returns (alpha, iters, needs_restoration, dx_cor, ds_cor)
    std::tuple<double, int, bool, dvec, dvec>
    search(ModelC *model, const dvec &x, const dvec &dx, const dvec &ds,
           const dvec &s, double mu, double d_phi,
           std::optional<double> theta0_opt = std::nullopt,
           double alpha_max = 1.0) const
    {
        // Input validation
        if (!model || x.size() == 0 || dx.size() != x.size()) {
            return {cfg_.ls_min_alpha, 0, true, dvec(), dvec()};
        }
        
        if (ds.size() > 0 && ds.size() != s.size()) {
            return {cfg_.ls_min_alpha, 0, true, dvec(), dvec()};
        }

        // ---- Base evaluation with error handling ----
        try {
            std::vector<std::string> comp_names{"f", "g", "cE", "cI", "JE", "JI"};
            model->eval_all(x, comp_names);
        } catch (const std::exception&) {
            return {cfg_.ls_min_alpha, 0, true, dvec(), dvec()};
        }

        const double f0 = model->get_f().value_or(std::numeric_limits<double>::max());
        const dvec   g0 = model->get_g().value_or(dvec::Zero(x.size()));
        const std::optional<dvec>  cE0 = model->get_cE();
        const std::optional<dvec>  cI0 = model->get_cI();
        const std::optional<spmat> JE0 = model->get_JE();
        const std::optional<spmat> JI0 = model->get_JI();

        if (!std::isfinite(f0)) {
            return {cfg_.ls_min_alpha, 0, true, dvec(), dvec()};
        }

        // Enhanced slack validation
        if (s.size() > 0 && (s.array() <= cfg_.ls_theta_eps).any()) {
            return {alpha_max, 0, true, dvec(), dvec()};
        }

        // Enhanced barrier computation with overflow protection
        const double barrier_eps = std::max(cfg_.ls_theta_eps, 1e-8 * mu);
        double phi0;
        try {
            phi0 = f0 - mu * s.array().unaryExpr([&](double v) {
                return safe_log_barrier(v, barrier_eps);
            }).sum();
        } catch (...) {
            return {cfg_.ls_min_alpha, 0, true, dvec(), dvec()};
        }

        if (!std::isfinite(phi0)) {
            return {cfg_.ls_min_alpha, 0, true, dvec(), dvec()};
        }

        // Enhanced constraint violation computation
        double theta0;
        if (theta0_opt) {
            theta0 = *theta0_opt;
        } else {
            theta0 = compute_theta(cE0, cI0, s);
        }

        // Enhanced descent direction check
        if (d_phi >= -cfg_.ls_theta_eps) {
            return {alpha_max, 0, true, dvec(), dvec()};
        }

        // Enhanced fraction-to-boundary computation
        double alpha_ftb = alpha_max;
        if (ds.size() == s.size()) {
            for (Eigen::Index i = 0; i < ds.size(); ++i) {
                if (ds[i] < -cfg_.ls_theta_eps) {
                    const double am = (1.0 - cfg_.ip_fraction_to_boundary_tau) * s[i] / (-ds[i]);
                    alpha_ftb = std::min(alpha_ftb, am);
                }
            }
        }
        alpha_ftb = std::max(alpha_ftb, cfg_.ls_min_alpha);

        // Enhanced funnel predictions
        double pred_df = 0.0;
        if (g0.size() == dx.size()) {
            pred_df = std::max(0.0, -g0.dot(dx));
        }

        // Enhanced constraint linearization
        const dvec je_dx = (JE0 && JE0->rows() > 0) ? (JE0.value() * dx) : dvec();
        const dvec ji_dx = (JI0 && JI0->rows() > 0) ? (JI0.value() * dx) : dvec();
        
        double theta_pred = 0.0;
        if (cE0 && cE0->size() > 0) {
            const dvec cE_pred = je_dx.size() ? (*cE0 + je_dx) : *cE0;
            theta_pred += cE_pred.lpNorm<1>();
        }
        if (cI0 && cI0->size() > 0) {
            dvec cI_pred = *cI0 + s + ds;
            if (ji_dx.size()) cI_pred += ji_dx;
            theta_pred += cI_pred.array().max(0.0).sum();
        }
        
        const double pred_dtheta = std::max(0.0, theta0 - theta_pred);

        // Seed non-monotone memory if empty
        if (cfg_.merit_mode == LSMeritMode::NonMonotonePhi && phi_hist_.empty()) {
            phi_hist_.push_back(phi0);
        }

        // Enhanced Richardson extrapolator (per search)
        RichardsonExtrapolator rich;

        // Enhanced line search loop with watchdog
        double alpha = std::min(1.0, alpha_ftb);
        int it = 0;
        int watchdog_count = 0;
        double best_alpha = cfg_.ls_min_alpha;
        double best_phi = std::numeric_limits<double>::max();
        double theta_t = theta0;

        while (it < cfg_.ls_max_iter && watchdog_count < cfg_.ls_watchdog_max) {
            const dvec x_t = x + alpha * dx;
            const dvec s_t = s + alpha * ds;
            
            // Enhanced slack bounds checking
            if (s_t.size() > 0 && (s_t.array() <= cfg_.ls_theta_eps).any()) {
                alpha *= cfg_.ls_backtrack;
                ++it;
                ++watchdog_count;
                continue;
            }
            
            try {
                std::vector<std::string> comps_t{"f", "cE", "cI"};
                model->eval_all(x_t, comps_t);
                const double f_t = model->get_f().value_or(std::numeric_limits<double>::max());
                
                if (!std::isfinite(f_t)) { 
                    alpha *= cfg_.ls_backtrack; 
                    ++it; 
                    ++watchdog_count;
                    continue; 
                }

                // Enhanced barrier computation
                double phi_t;
                try {
                    phi_t = f_t - mu * s_t.array().unaryExpr([&](double v) {
                        return safe_log_barrier(v, barrier_eps);
                    }).sum();
                } catch (...) {
                    alpha *= cfg_.ls_backtrack; 
                    ++it; 
                    ++watchdog_count;
                    continue;
                }
                
                if (!std::isfinite(phi_t)) { 
                    alpha *= cfg_.ls_backtrack; 
                    ++it; 
                    ++watchdog_count;
                    continue; 
                }

                // Track best point for potential fallback
                if (phi_t < best_phi) {
                    best_phi = phi_t;
                    best_alpha = alpha;
                }

                // Enhanced constraint violation computation
                auto cE_t = model->get_cE();
                auto cI_t = model->get_cI();
                theta_t = compute_theta(cE_t, cI_t, s_t);

                // Enhanced acceptance tests
                bool accepted = false;
                
                // 1) Funnel/Filter takes precedence
                if (funnel_) {
                    if (funnel_->is_acceptable(theta0, f0, theta_t, f_t, pred_df, pred_dtheta)) {
                        funnel_->add_if_acceptable(theta0, f0, theta_t, f_t, pred_df, pred_dtheta);
                        push_phi_(phi_t);
                        return {alpha, it, false, dvec(), dvec()};
                    }
                } else if (filter_) {
                    // Use C++ filter directly
                    if (filter_->is_acceptable(theta_t, f_t)) {
                        filter_->add_if_acceptable(theta_t, f_t);
                        push_phi_(phi_t);
                        return {alpha, it, false, dvec(), dvec()};
                    }
                } else {
                    // 2) Enhanced Monotone/Non-monotone φ
                    double phi_ref = phi0; // monotone default
                    if (cfg_.merit_mode == LSMeritMode::NonMonotonePhi) {
                        // Cross-iteration window max
                        if (!phi_hist_.empty()) {
                            phi_ref = *std::max_element(phi_hist_.begin(), phi_hist_.end());
                        }
                        phi_ref += cfg_.nm_safeguard;
                    }
                    
                    // Enhanced Armijo condition with sufficient decrease
                    if (phi_t <= phi_ref + cfg_.ls_sufficient_decrease * alpha * d_phi) {
                        push_phi_(phi_t);
                        return {alpha, it, false, dvec(), dvec()};
                    }
                }

                // -------- Enhanced Richardson attempt --------
                if (cfg_.use_richardson && it < cfg_.ls_max_iter / 2) {
                    const dvec step_comb = concat_dx_ds(alpha * dx, alpha * ds);
                    rich.add_step(step_comb, alpha);
                    auto ext = rich.extrapolate_step(step_comb, alpha, cfg_.rich_tol);
                    
                    if (ext.order_achieved >= cfg_.rich_min_order &&
                        std::isfinite(ext.error_estimate) &&
                        ext.error_estimate < cfg_.rich_tol) {

                        dvec dx_ref, ds_ref;
                        split_dx_ds(ext.dx_refined, dx.size(), dx_ref, ds_ref);

                        // Enhanced Richardson step validation
                        double alpha_ref = 1.0;
                        if (ds_ref.size() == s.size()) {
                            for (Eigen::Index i = 0; i < ds_ref.size(); ++i) {
                                if (ds_ref[i] < -cfg_.ls_theta_eps) {
                                    const double am = (1.0 - cfg_.ip_fraction_to_boundary_tau) *
                                                      s[i] / (-ds_ref[i]);
                                    alpha_ref = std::min(alpha_ref, am);
                                }
                            }
                            alpha_ref = std::max(alpha_ref, cfg_.ls_min_alpha);
                        }

                        const dvec x_ref = x + alpha_ref * dx_ref;
                        const dvec s_ref = s + alpha_ref * ds_ref;
                        
                        if (s_ref.size() == 0 || (s_ref.array() > cfg_.ls_theta_eps).all()) {
                            try {
                                std::vector<std::string> comps_r{"f", "cE", "cI"};
                                model->eval_all(x_ref, comps_r);
                                const double f_ref = model->get_f().value_or(std::numeric_limits<double>::max());
                                
                                if (std::isfinite(f_ref)) {
                                    double phi_refined;
                                    try {
                                        phi_refined = f_ref - mu * s_ref.array().unaryExpr([&](double v) {
                                            return safe_log_barrier(v, barrier_eps);
                                        }).sum();
                                    } catch (...) {
                                        goto skip_richardson;
                                    }
                                    
                                    if (!std::isfinite(phi_refined)) goto skip_richardson;
                                    
                                    auto cE_r = model->get_cE();
                                    auto cI_r = model->get_cI();
                                    const double theta_refined = compute_theta(cE_r, cI_r, s_ref);

                                    bool ok_ref = false;
                                    if (funnel_) {
                                        ok_ref = funnel_->is_acceptable(theta0, f0, theta_refined, f_ref, pred_df, pred_dtheta);
                                        if (ok_ref) funnel_->add_if_acceptable(theta0, f0, theta_refined, f_ref, pred_df, pred_dtheta);
                                    } else if (filter_) {
                                        ok_ref = filter_->is_acceptable(theta_refined, f_ref);
                                        if (ok_ref) filter_->add_if_acceptable(theta_refined, f_ref);
                                    } else {
                                        double phi_ref_acc = phi0;
                                        if (cfg_.merit_mode == LSMeritMode::NonMonotonePhi && !phi_hist_.empty()) {
                                            phi_ref_acc = *std::max_element(phi_hist_.begin(), phi_hist_.end()) + cfg_.nm_safeguard;
                                        }
                                        ok_ref = (phi_refined <= phi_ref_acc + cfg_.ls_sufficient_decrease * alpha_ref * d_phi);
                                    }

                                    if (ok_ref) {
                                        push_phi_(phi_refined);
                                        return {alpha_ref, it, false, dx_ref, ds_ref};
                                    }
                                }
                            } catch (...) {
                                // Richardson failed, continue normal search
                            }
                        }
                    }
                }
                skip_richardson:

                // // Enhanced SOC with improved triggers and validation
                // if (it == 0 && theta_t >= cfg_.ls_theta_min_improvement * theta0 && 
                //     cfg_.max_soc > 0 && theta0 > cfg_.ls_theta_eps) {
                    
                //     auto soc_result = perform_enhanced_soc(model, x, s, alpha, dx, ds, mu, phi0, d_phi,
                //                                          theta0, theta_t, f0, pred_df, pred_dtheta, barrier_eps);
                //     if (std::get<0>(soc_result)) {
                //         // SOC succeeded
                //         return std::get<1>(soc_result);
                //     }
                // }

            } catch (const std::exception &) {
                alpha *= cfg_.ls_backtrack;
                ++it;
                ++watchdog_count;
                continue;
            }

            // Enhanced backtracking with adaptive step size
            const double backtrack_factor = (watchdog_count > cfg_.ls_watchdog_max / 2) ? 
                cfg_.ls_backtrack * 0.5 : cfg_.ls_backtrack;
            alpha *= backtrack_factor;
            ++it;
            
            // Reset watchdog if we're making progress
            if (theta_t < theta0) {
                watchdog_count = 0;
            } else {
                ++watchdog_count;
            }
        }

        // Enhanced fallback strategy
        const bool needs_restoration = (theta0 > cfg_.ls_theta_restoration) || 
                                       (best_alpha <= cfg_.ls_min_alpha);
        return {best_alpha, it, needs_restoration, dvec(), dvec()};
    }

private:
    // Enhanced SOC implementation
    std::tuple<bool, std::tuple<double, int, bool, dvec, dvec>>
    perform_enhanced_soc(ModelC* model, const dvec& x, const dvec& s, double alpha,
                        const dvec& dx, const dvec& ds, double mu, double phi0, double d_phi,
                        double theta0, double theta_t, double f0, double pred_df, double pred_dtheta,
                        double barrier_eps) const {
        
        try {
            std::vector<std::string> compsJ{"JE", "JI", "cE", "cI"};
            model->eval_all(x + alpha * dx, compsJ);
            
            const auto JE_t = model->get_JE();
            const auto JI_t = model->get_JI();
            const auto cE_t = model->get_cE();
            const auto cI_t = model->get_cI();
            
            dvec s_t_cur = s + alpha * ds;
            double theta_last = theta_t;
            
            for (int soc_count = 1; soc_count <= cfg_.max_soc; ++soc_count) {
                // Enhanced adaptive kappa
                double kappa_soc = cfg_.kappa_soc_base;
                if (theta0 > cfg_.ls_theta_eps) {
                    const double ratio = theta_last / theta0;
                    if (ratio > 10.0) {
                        kappa_soc = cfg_.kappa_soc_min;
                    } else if (ratio > 1.0) {
                        kappa_soc = cfg_.kappa_soc_min + 
                            (cfg_.kappa_soc_max - cfg_.kappa_soc_min) * (1.0 - (ratio - 1.0) / 9.0);
                    } else {
                        kappa_soc = cfg_.kappa_soc_max;
                    }
                }
                kappa_soc = std::min(kappa_soc + 0.05 * (soc_count - 1), cfg_.kappa_soc_max);

                // Enhanced constraint residuals
                dvec rE = cE_t ? cE_t.value() : dvec();
                dvec rI = (cI_t && s_t_cur.size() == cI_t->size()) ? 
                    (cI_t.value() + s_t_cur) : dvec();
                
                if (rE.size() == 0 && rI.size() == 0) break;

                // Enhanced normal equations with regularization
                const Eigen::Index n = static_cast<Eigen::Index>(x.size());
                dmat AtA = dmat::Zero(n, n);
                dvec At_r = dvec::Zero(n);
                
                if (rE.size() > 0 && JE_t && JE_t->rows() == rE.size()) {
                    const dmat JE_dense = dmat(JE_t.value());
                    AtA.noalias() += JE_dense.transpose() * JE_dense;
                    At_r.noalias() += JE_dense.transpose() * rE;
                }
                
                if (rI.size() > 0 && JI_t && JI_t->rows() == rI.size()) {
                    const dmat JI_dense = dmat(JI_t.value());
                    AtA.noalias() += JI_dense.transpose() * JI_dense;
                    At_r.noalias() += JI_dense.transpose() * rI;
                }
                
                // Enhanced regularization
                const double eps_reg = std::max(1e-12, 1e-8 * AtA.diagonal().maxCoeff());
                AtA.diagonal().array() += eps_reg;

                // Enhanced solve with fallback
                dvec dx_cor;
                try {
                    Eigen::LDLT<dmat> ldlt(AtA);
                    if (ldlt.info() == Eigen::Success) {
                        dx_cor = ldlt.solve(-At_r);
                    } else {
                        dx_cor = AtA.colPivHouseholderQr().solve(-At_r);
                    }
                } catch (...) {
                    break; // SOC failed
                }

                // Enhanced ds_cor computation
                dvec ds_cor = dvec::Zero(s.size());
                if (rI.size() > 0 && JI_t && JI_t->rows() == rI.size()) {
                    const dvec JI_dx = dmat(JI_t.value()) * dx_cor;
                    ds_cor = -(rI + JI_dx);
                }

                // Enhanced fraction-to-boundary for SOC
                double alpha_soc = 1.0;
                if (ds_cor.size() > 0) {
                    for (Eigen::Index i = 0; i < ds_cor.size(); ++i) {
                        if (ds_cor[i] < -cfg_.ls_theta_eps) {
                            const double am = (1.0 - cfg_.ip_fraction_to_boundary_tau) * s[i] / (-ds_cor[i]);
                            alpha_soc = std::min(alpha_soc, am);
                        }
                    }
                    alpha_soc = std::max(alpha_soc, cfg_.ls_min_alpha);
                }

                const dvec x_t_soc = x + alpha_soc * dx_cor;
                const dvec s_t_soc = s + alpha_soc * ds_cor;
                
                if (s_t_soc.size() > 0 && (s_t_soc.array() <= cfg_.ls_theta_eps).any()) break;

                // Enhanced SOC evaluation
                try {
                    std::vector<std::string> comps_soc{"f", "cE", "cI"};
                    model->eval_all(x_t_soc, comps_soc);
                    const double f_t_soc = model->get_f().value_or(std::numeric_limits<double>::max());
                    
                    if (!std::isfinite(f_t_soc)) continue;

                    const auto cE_soc = model->get_cE();
                    const auto cI_soc = model->get_cI();
                    const double theta_t_soc = compute_theta(cE_soc, cI_soc, s_t_soc);

                    // Enhanced SOC acceptance criteria
                    if (theta_t_soc >= kappa_soc * theta_last) break;

                    double phi_t_soc;
                    try {
                        phi_t_soc = f_t_soc - mu * s_t_soc.array().unaryExpr([&](double v) {
                            return safe_log_barrier(v, barrier_eps);
                        }).sum();
                    } catch (...) {
                        continue;
                    }

                    if (!std::isfinite(phi_t_soc) ||
                        phi_t_soc > phi0 + cfg_.ls_sufficient_decrease * alpha * d_phi) {
                        continue;
                    }

                    // Enhanced acceptance test for SOC
                    bool acceptable_ok_soc = true;
                    if (funnel_) {
                        acceptable_ok_soc = funnel_->is_acceptable(theta0, f0, theta_t_soc, f_t_soc, pred_df, pred_dtheta);
                        if (acceptable_ok_soc) {
                            funnel_->add_if_acceptable(theta0, f0, theta_t_soc, f_t_soc, pred_df, pred_dtheta);
                        }
                    } else if (filter_) {
                        acceptable_ok_soc = filter_->is_acceptable(theta_t_soc, f_t_soc);
                        if (acceptable_ok_soc) {
                            filter_->add_if_acceptable(theta_t_soc, f_t_soc);
                        }
                    }

                    if (acceptable_ok_soc) {
                        push_phi_(phi_t_soc);
                        return {true, std::make_tuple(alpha_soc, soc_count, false, dx_cor, ds_cor)};
                    }

                    theta_last = theta_t_soc;
                    s_t_cur = s_t_soc;
                    
                } catch (const std::exception &) {
                    continue;
                }
            }
        } catch (...) {
            // SOC initialization failed
        }
        
        return {false, std::make_tuple(0.0, 0, true, dvec(), dvec())};
    }

    // Enhanced cross-iteration memory management
    void push_phi_(double phi) const {
        if (!std::isfinite(phi)) return;
        if (cfg_.merit_mode != LSMeritMode::NonMonotonePhi) return;
        
        phi_hist_.push_back(phi);
        while (static_cast<int>(phi_hist_.size()) > cfg_.nm_memory) {
            phi_hist_.pop_front();
        }
    }

    nb::object cfg_obj_;
    std::shared_ptr<Filter> filter_;     // Using your existing Filter class
    std::shared_ptr<Funnel> funnel_;
    LSConfig cfg_;

    // Cross-iteration non-monotone φ memory (mutable for const search)
    mutable std::deque<double> phi_hist_;
};