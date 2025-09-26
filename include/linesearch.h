// line_searcher_nb.cc — enhanced (Richardson + monotone/nonmonotone φ with cross-iteration memory)

#include "funnel.h" // FunnelConfig, Funnel
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "model.h"

// Bring your RichardsonExtrapolator declaration
#include "ip_aux.h"

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
    double ls_wolfe_c{0.9}; // placeholder

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
};

// --------------- small helpers ---------------
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
static inline dvec concat_dx_ds(const dvec &dx, const dvec &ds) {
    dvec out(dx.size() + ds.size());
    out.head(dx.size()) = dx;
    out.tail(ds.size()) = ds;
    return out;
}
static inline void split_dx_ds(const dvec &v, Eigen::Index nx, dvec &dx_out, dvec &ds_out) {
    dx_out = v.head(nx);
    ds_out = v.tail(v.size() - nx);
}

// --------------- line searcher ---------------
class LineSearcher {
public:
    LineSearcher(nb::object cfg, nb::object filter = nb::none(),
                 std::shared_ptr<Funnel> funnel = nullptr)
        : cfg_obj_(std::move(cfg)), filter_(std::move(filter)), funnel_(std::move(funnel)) {

        // cfg load
        cfg_.ls_backtrack = std::clamp(getattr_or_double(cfg_obj_, "ls_backtrack", 0.5), 1e-4, 0.99);
        cfg_.ls_armijo_f  = std::max(1e-12, getattr_or_double(cfg_obj_, "ls_armijo_f", 1e-4));
        cfg_.ls_max_iter  = std::max(1, getattr_or_int(cfg_obj_, "ls_max_iter", 20));
        cfg_.ls_min_alpha = std::max(0.0, getattr_or_double(cfg_obj_, "ls_min_alpha", 1e-12));
        (void)getattr_or_bool(cfg_obj_, "ls_use_wolfe", false);
        cfg_.ls_wolfe_c   = std::clamp(getattr_or_double(cfg_obj_, "ls_wolfe_c", 0.9),
                                       cfg_.ls_armijo_f, 0.999);
        cfg_.ip_fraction_to_boundary_tau =
            getattr_or_double(cfg_obj_, "ip_fraction_to_boundary_tau", 0.995);
        cfg_.ls_theta_restoration =
            getattr_or_double(cfg_obj_, "ls_theta_restoration", 1e3);

        cfg_.max_soc       = std::max(0, getattr_or_int(cfg_obj_, "max_soc", 4));
        cfg_.kappa_soc_min = std::clamp(getattr_or_double(cfg_obj_, "kappa_soc_min", 0.1), 0.01, 0.5);
        cfg_.kappa_soc_max = std::clamp(getattr_or_double(cfg_obj_, "kappa_soc_max", 0.99), 0.5, 0.99);
        cfg_.kappa_soc_base= std::clamp(getattr_or_double(cfg_obj_, "kappa_soc_base", 0.5),
                                        cfg_.kappa_soc_min, cfg_.kappa_soc_max);

        // Merit mode selection
        const std::string mode = nb::hasattr(cfg_obj_, "ls_merit_mode")
                                 ? nb::cast<std::string>(cfg_obj_.attr("ls_merit_mode"))
                                 : std::string("filter"); // "monotone", "nonmonotone", "filter"
        if (mode == "monotone")      cfg_.merit_mode = LSMeritMode::MonotonePhi;
        else if (mode == "nonmonotone") cfg_.merit_mode = LSMeritMode::NonMonotonePhi;
        else                          cfg_.merit_mode = LSMeritMode::FilterOrFunnel;

        cfg_.nm_memory    = std::max(1, getattr_or_int(cfg_obj_, "ls_nm_memory", 5));
        cfg_.nm_safeguard = std::max(0.0, getattr_or_double(cfg_obj_, "ls_nm_safeguard", 0.0));

        // Richardson
        cfg_.use_richardson = getattr_or_bool(cfg_obj_, "ls_use_richardson", true);
        cfg_.rich_tol       = std::max(1e-14, getattr_or_double(cfg_obj_, "ls_rich_tol", 1e-8));
        cfg_.rich_min_order = std::max(2, getattr_or_int(cfg_obj_, "ls_rich_min_order", 2));
    }

    // --- public helpers for outer control (optional) ---
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
        // ---- base eval
        std::vector<std::string> comp_names{"f", "g", "cE", "cI", "JE", "JI"};
        model->eval_all(x, comp_names);
        const double f0 = model->get_f().value();
        const dvec   g0 = model->get_g().value();
        const std::optional<dvec>  cE0 = model->get_cE();
        const std::optional<dvec>  cI0 = model->get_cI();
        const std::optional<spmat> JE0 = model->get_JE();
        const std::optional<spmat> JI0 = model->get_JI();

        if (s.size() == 0 || (s.array() <= 0.0).any())
            return {alpha_max, 0, true, dvec(), dvec()};

        const double barrier_eps = std::max(1e-8 * mu, 1e-16);
        const double phi0 = f0 - mu * (s.array().unaryExpr([&](double v) {
            return std::log(std::max(v, barrier_eps));
        }).sum());

        double theta0 = 0.0;
        if (theta0_opt) {
            theta0 = *theta0_opt;
        } else {
            double thE = 0.0, thI = 0.0;
            if (cE0 && cE0->size() != 0) thE = cE0->array().abs().sum();
            if (cI0 && cI0->size() != 0) thI = (cI0->array() + s.array()).abs().sum();
            theta0 = thE + thI;
        }

        if (d_phi >= -1e-12)
            return {alpha_max, 0, true, dvec(), dvec()};

        // fraction-to-boundary α_max
        if (ds.size() == s.size()) {
            for (Eigen::Index i = 0; i < ds.size(); ++i) {
                if (ds[i] < 0.0) {
                    const double am = (1.0 - cfg_.ip_fraction_to_boundary_tau) * s[i] / (-ds[i]);
                    if (am < alpha_max) alpha_max = am;
                }
            }
        }
        if (alpha_max < 1.0e-16) alpha_max = 1.0e-16;

        // Funnel predictions
        double pred_df = 0.0;
        if (g0.size() == dx.size()) {
            pred_df = -g0.dot(dx);
            if (pred_df < 0.0) pred_df = 0.0;
        }

        const dvec je_dx = JE0 ? (JE0.value() * dx) : dvec();
        const dvec ji_dx = JI0 ? (JI0.value() * dx) : dvec();
        double thE_lin = 0.0, thI_lin = 0.0;
        if (cE0 && cE0->size() != 0) {
            const dvec &vE = *cE0;
            thE_lin = (je_dx.size() ? (vE + je_dx).array().abs().sum() : vE.array().abs().sum());
        }
        if (cI0 && cI0->size() != 0) {
            const dvec &vI = *cI0;
            dvec rI_lin = vI + s + ds;
            if (ji_dx.size()) rI_lin += ji_dx;
            thI_lin = rI_lin.array().abs().sum();
        }
        const double theta_pred = thE_lin + thI_lin;
        double pred_dtheta = theta0 - theta_pred;
        if (pred_dtheta < 0.0) pred_dtheta = 0.0;

        // Seed non-monotone memory if empty
        if (cfg_.merit_mode == LSMeritMode::NonMonotonePhi && phi_hist_.empty()) {
            phi_hist_.push_back(phi0);
        }

        // Richardson (per search)
        RichardsonExtrapolator rich;

        // line search loop
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
                std::vector<std::string> comps_t{"f", "cE", "cI"};
                model->eval_all(x_t, comps_t);
                const double f_t = model->get_f().value();
                if (!std::isfinite(f_t)) { alpha *= cfg_.ls_backtrack; ++it; continue; }

                const double phi_t =
                    f_t - mu * (s_t.array().unaryExpr([&](double v) {
                        return std::log(std::max(v, barrier_eps));
                    }).sum());
                if (!std::isfinite(phi_t)) { alpha *= cfg_.ls_backtrack; ++it; continue; }

                auto cE_t = model->get_cE();
                auto cI_t = model->get_cI();
                const double thE_t = (cE_t && cE_t->size()!=0) ? cE_t->array().abs().sum() : 0.0;
                double thI_t = 0.0;
                if (cI_t && cI_t->size()!=0) {
                    dvec vI_t = *cI_t;
                    thI_t = (vI_t.array() + s_t.array()).abs().sum();
                }
                theta_t = thE_t + thI_t;

                // Acceptance
                // 1) Funnel/Filter takes precedence (keeps original semantics)
                if (funnel_) {
                    if (funnel_->is_acceptable(theta0, f0, theta_t, f_t, pred_df, pred_dtheta)) {
                        (void)funnel_->add_if_acceptable(theta0, f0, theta_t, f_t, pred_df, pred_dtheta);
                        // Update cross-iteration memory
                        push_phi_(phi_t);
                        return {alpha, it, false, dvec(), dvec()};
                    }
                } else if (filter_ && !filter_.is_none()) {
                    bool ok = nb::cast<bool>(filter_.attr("is_acceptable")(theta_t, f_t));
                    if (ok) {
                        (void)filter_.attr("add_if_acceptable")(theta_t, f_t);
                        push_phi_(phi_t);
                        return {alpha, it, false, dvec(), dvec()};
                    }
                } else {
                    // 2) Monotone/Non-monotone φ
                    double phi_ref = phi0; // monotone default
                    if (cfg_.merit_mode == LSMeritMode::NonMonotonePhi) {
                        // Cross-iteration window max
                        double phi_max = phi_hist_.empty() ? phi0 : phi_hist_.front();
                        for (double v : phi_hist_) phi_max = std::max(phi_max, v);
                        phi_ref = phi_max + cfg_.nm_safeguard;
                    }
                    if (phi_t <= phi_ref + cfg_.ls_armijo_f * alpha * d_phi) {
                        push_phi_(phi_t);
                        return {alpha, it, false, dvec(), dvec()};
                    }
                }

                // -------- Richardson attempt (fail-soft) --------
                if (cfg_.use_richardson) {
                    const dvec step_comb = concat_dx_ds(alpha * dx, alpha * ds);
                    rich.add_step(step_comb, /*h=*/alpha);
                    auto ext = rich.extrapolate_step(step_comb, alpha, cfg_.rich_tol);
                    if (ext.order_achieved >= cfg_.rich_min_order &&
                        std::isfinite(ext.error_estimate) &&
                        ext.error_estimate < cfg_.rich_tol) {

                        dvec dx_ref, ds_ref;
                        split_dx_ds(ext.dx_refined, dx.size(), dx_ref, ds_ref);

                        double alpha_ref = 1.0;
                        if (ds_ref.size() == s.size()) {
                            for (Eigen::Index i = 0; i < ds_ref.size(); ++i) {
                                if (ds_ref[i] < 0.0) {
                                    const double am = (1.0 - cfg_.ip_fraction_to_boundary_tau) *
                                                      s[i] / (-ds_ref[i]);
                                    if (am < alpha_ref) alpha_ref = am;
                                }
                            }
                            alpha_ref = std::max(alpha_ref, cfg_.ls_min_alpha);
                        }

                        const dvec x_ref = x + alpha_ref * dx_ref;
                        const dvec s_ref = s + alpha_ref * ds_ref;
                        if (!(s_ref.size() && (s_ref.array() <= 0.0).any())) {
                            double f_ref = 0.0, phi_refined = 0.0, theta_refined = 0.0;
                            try {
                                std::vector<std::string> comps_r{"f", "cE", "cI"};
                                model->eval_all(x_ref, comps_r);
                                f_ref = model->get_f().value();
                                if (std::isfinite(f_ref)) {
                                    phi_refined = f_ref - mu * (s_ref.array().unaryExpr([&](double v) {
                                        return std::log(std::max(v, barrier_eps));
                                    }).sum());
                                    auto cE_r = model->get_cE();
                                    auto cI_r = model->get_cI();
                                    const double thE_r = (cE_r && cE_r->size()!=0)
                                                           ? cE_r->array().abs().sum() : 0.0;
                                    double thI_r = 0.0;
                                    if (cI_r && cI_r->size()!=0) {
                                        dvec vI_r = *cI_r;
                                        thI_r = (vI_r.array() + s_ref.array()).abs().sum();
                                    }
                                    theta_refined = thE_r + thI_r;

                                    bool ok_ref = true;
                                    if (funnel_) {
                                        ok_ref = funnel_->is_acceptable(theta0, f0, theta_refined, f_ref, pred_df, pred_dtheta);
                                        if (ok_ref) (void)funnel_->add_if_acceptable(theta0, f0, theta_refined, f_ref, pred_df, pred_dtheta);
                                    } else if (filter_ && !filter_.is_none()) {
                                        ok_ref = nb::cast<bool>(filter_.attr("is_acceptable")(theta_refined, f_ref));
                                        if (ok_ref) (void)filter_.attr("add_if_acceptable")(theta_refined, f_ref);
                                    } else {
                                        double phi_ref_acc = phi0;
                                        if (cfg_.merit_mode == LSMeritMode::NonMonotonePhi) {
                                            double phi_max = phi_hist_.empty() ? phi0 : phi_hist_.front();
                                            for (double v : phi_hist_) phi_max = std::max(phi_max, v);
                                            phi_ref_acc = phi_max + cfg_.nm_safeguard;
                                        }
                                        ok_ref = (phi_refined <= phi_ref_acc + cfg_.ls_armijo_f * alpha_ref * d_phi);
                                    }

                                    if (ok_ref) {
                                        push_phi_(phi_refined);
                                        return {alpha_ref, it, false, dx_ref, ds_ref};
                                    }
                                }
                            } catch (...) {
                                // ignore
                            }
                        }
                    }
                }

            } catch (const std::exception &) {
                alpha *= cfg_.ls_backtrack;
                ++it;
                continue;
            }

            // ---------------- SOC block (unchanged except memory update on accept) ------------
            if (it == 0 && theta_t >= theta0 && cfg_.max_soc > 0) {
                dvec cE_t = cE0 ? model->get_cE().value() : dvec();
                dvec cI_t = cI0 ? model->get_cI().value() : dvec();
                dvec s_t_cur = s + alpha * ds;

                auto nnzE = (cE_t.size() ? cE_t.size() : 0);
                auto nnzI = (cI_t.size() ? cI_t.size() : 0);

                double theta_last = theta_t;
                int soc_count = 0;

                while (soc_count < cfg_.max_soc) {
                    ++soc_count;

                    double kappa_soc = cfg_.kappa_soc_base;
                    if (theta0 > 1e-8) {
                        const double ratio = theta_last / theta0;
                        if (ratio > 10.0)      kappa_soc = cfg_.kappa_soc_min;
                        else if (ratio > 1.0)  kappa_soc = cfg_.kappa_soc_min +
                            (cfg_.kappa_soc_max - cfg_.kappa_soc_min) * (1.0 - (ratio - 1.0) / 9.0);
                        else                   kappa_soc = cfg_.kappa_soc_max;
                    }
                    kappa_soc = std::min(kappa_soc + 0.1 * (soc_count - 1), cfg_.kappa_soc_max);

                    { std::vector<std::string> compsJ{"JE", "JI"};
                      model->eval_all(x + alpha * dx, compsJ); }
                    dmat JE_t = (JE0 && nnzE) ? model->get_JE().value()
                                              : dmat((Eigen::Index)0,(Eigen::Index)x.size());
                    dmat JI_t = (JI0 && nnzI) ? model->get_JI().value()
                                              : dmat((Eigen::Index)0,(Eigen::Index)x.size());

                    dvec rE = cE_t;
                    dvec rI = (nnzI ? (cI_t + s_t_cur) : dvec());
                    if ((rE.size() == 0) && (rI.size() == 0)) break;

                    const Eigen::Index n = static_cast<Eigen::Index>(x.size());
                    dmat AtA = dmat::Zero(n, n);
                    dvec At_r = dvec::Zero(n);
                    if (rE.size() && JE_t.rows() == rE.size()) { AtA.noalias() += JE_t.transpose() * JE_t; At_r.noalias() += JE_t.transpose() * rE; }
                    if (rI.size() && JI_t.rows() == rI.size()) { AtA.noalias() += JI_t.transpose() * JI_t; At_r.noalias() += JI_t.transpose() * rI; }
                    const double eps_reg = 1e-12; AtA.diagonal().array() += eps_reg;

                    dvec dx_cor;
                    {   Eigen::LDLT<dmat> ldlt(AtA);
                        if (ldlt.info() != Eigen::Success)
                            dx_cor = (-AtA).colPivHouseholderQr().solve(At_r);
                        else
                            dx_cor = ldlt.solve(-At_r);
                    }
                    dvec ds_cor;
                    if (rI.size()) {
                        dvec JI_dx = (JI_t.rows() ? (JI_t * dx_cor) : dvec());
                        ds_cor = -(rI + JI_dx);
                    } else ds_cor = dvec();

                    double alpha_soc = 1.0;
                    if (ds_cor.size()) {
                        for (Eigen::Index i = 0; i < ds_cor.size(); ++i)
                            if (ds_cor[i] < 0.0) {
                                const double am = (1.0 - cfg_.ip_fraction_to_boundary_tau) * s[i] / (-ds_cor[i]);
                                if (am < alpha_soc) alpha_soc = am;
                            }
                        alpha_soc = std::max(alpha_soc, cfg_.ls_min_alpha);
                    }

                    const dvec x_t_soc = x + alpha_soc * dx_cor;
                    const dvec s_t_soc = s + alpha_soc * ds_cor;
                    if (s_t_soc.size() && (s_t_soc.array() <= 0.0).any()) break;

                    double f_t_soc = 0.0, theta_t_soc = 0.0;
                    try {
                        std::vector<std::string> comps_soc{"f", "cE", "cI"};
                        model->eval_all(x_t_soc, comps_soc);
                        f_t_soc = model->get_f().value();
                        if (!std::isfinite(f_t_soc)) continue;

                        dvec cE_soc = model->get_cE() ? model->get_cE().value() : dvec();
                        dvec cI_soc = model->get_cI() ? model->get_cI().value() : dvec();
                        const double thE_soc = cE_soc.size() ? cE_soc.array().abs().sum() : 0.0;
                        double thI_soc = 0.0;
                        if (cI_soc.size()) thI_soc = (cI_soc.array() + s_t_soc.array()).abs().sum();
                        theta_t_soc = thE_soc + thI_soc;
                    } catch (const std::exception &) { continue; }

                    const double phi_t_soc =
                        f_t_soc - mu * (s_t_soc.array().unaryExpr([&](double v) {
                            return std::log(std::max(v, barrier_eps));
                        }).sum());

                    // Require infeasibility improvement and Armijo (your policy)
                    if (!(theta_t_soc < kappa_soc * theta_last)) { break; }
                    if (!std::isfinite(phi_t_soc) ||
                        phi_t_soc > phi0 + cfg_.ls_armijo_f * alpha * d_phi) {
                        continue;
                    }

                    bool acceptable_ok_soc = true;
                    if (funnel_)        acceptable_ok_soc = funnel_->is_acceptable(theta0, f0, theta_t_soc, f_t_soc, pred_df, pred_dtheta);
                    else if (filter_ && !filter_.is_none())
                                        acceptable_ok_soc = nb::cast<bool>(filter_.attr("is_acceptable")(theta_t_soc, f_t_soc));

                    if (acceptable_ok_soc) {
                        if (funnel_) (void)funnel_->add_if_acceptable(theta0, f0, theta_t_soc, f_t_soc, pred_df, pred_dtheta);
                        else if (filter_ && !filter_.is_none())
                            (void)filter_.attr("add_if_acceptable")(theta_t_soc, f_t_soc);
                        // Update cross-iteration memory
                        push_phi_(phi_t_soc);
                        return {alpha_soc, it + soc_count, false, dx_cor, ds_cor};
                    }

                    theta_last = theta_t_soc;
                }
            }

            // backtrack
            alpha *= cfg_.ls_backtrack;
            ++it;
        }

        const bool needs_restoration = (theta0 > cfg_.ls_theta_restoration);
        return {alpha_max, it, needs_restoration, dvec(), dvec()};
    }

private:
    // push into cross-iteration window (trim to nm_memory)
    void push_phi_(double phi) const {
        if (!std::isfinite(phi)) return;
        if (cfg_.merit_mode != LSMeritMode::NonMonotonePhi) return;
        phi_hist_.push_back(phi);
        while ((int)phi_hist_.size() > cfg_.nm_memory) phi_hist_.pop_front();
    }

    nb::object cfg_obj_;
    nb::object filter_;
    std::shared_ptr<Funnel> funnel_;
    LSConfig cfg_;

    // Cross-iteration non-monotone φ memory (mutable to update inside const search)
    mutable std::deque<double> phi_hist_;
};
