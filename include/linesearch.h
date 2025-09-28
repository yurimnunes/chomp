// line_searcher_improved.cc — faster, allocation-light line search with mask/view fast paths
// Compile-time toggles (define if your ModelC supports them):
//   - MODEL_HAS_EVAL_MASK : enables eval_all_mask(x, mask) instead of string vectors
//   - MODEL_HAS_VIEW      : enables get_view() to avoid optionals/copies
//
// Notes:
// - SOC uses sparse least-squares (SparseQR) on [JE; JI] without dense conversion.
// - Fraction-to-boundary and slack checks are vectorized.
// - Non-monotone memory uses a fixed-size ring buffer (no per-iteration allocs).

#include "funnel.h"   // FunnelConfig, Funnel
#include "filter.h"   // Filter class
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/QR>
#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <deque>

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "model.h"
#include "ip_aux.h"  // RichardsonExtrapolator

namespace nb = nanobind;
using dvec  = Eigen::VectorXd;
using dmat  = Eigen::MatrixXd;
using spmat = Eigen::SparseMatrix<double>;

enum class LSMeritMode { MonotonePhi, NonMonotonePhi, FilterOrFunnel };

// ---- Optional fast-path interface (define in model.h if you can) ----------
#ifdef MODEL_HAS_EVAL_MASK
enum EvalMask : uint32_t {
    EV_F  = 1u << 0,
    EV_G  = 1u << 1,
    EV_cE = 1u << 2,
    EV_cI = 1u << 3,
    EV_JE = 1u << 4,
    EV_JI = 1u << 5
};
#endif

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
    int    nm_memory{5};
    double nm_safeguard{0.0};

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
    double ls_theta_min_improvement{0.1};
};

// ----------------- Helpers (no allocations on hot path) -----------------
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

static inline double safe_log_barrier(double val, double eps) {
    return std::log(std::max(val, eps));
}

static inline double compute_theta(const dvec* cE, const dvec* cI, const dvec& s, bool inf_norm=false) {
    double tE = 0.0, tI = 0.0;
    if (cE && cE->size() > 0) tE = inf_norm ? cE->lpNorm<Eigen::Infinity>() : cE->lpNorm<1>();
    if (cI && cI->size() > 0 && s.size() == cI->size()) {
        dvec viol = (cI->array() + s.array()).max(0.0);
        tI = inf_norm ? viol.lpNorm<Eigen::Infinity>() : viol.lpNorm<1>();
    }
    return tE + tI;
}

// Fixed-size ring buffer (no alloc after reserve)
class RingBuffer {
public:
    explicit RingBuffer(int cap=5) : cap_(std::max(1, cap)), buf_(cap_, 0.0) {}
    void reset(int cap) {
        cap_ = std::max(1, cap);
        buf_.assign(cap_, 0.0);
        size_ = 0; head_ = 0;
    }
    void push(double v) {
        buf_[head_] = v;
        head_ = (head_ + 1) % cap_;
        size_ = std::min(size_ + 1, cap_);
    }
    bool empty() const { return size_ == 0; }
    double max() const {
        double m = -std::numeric_limits<double>::infinity();
        for (int i=0;i<size_;++i) m = std::max(m, buf_[i]);
        return m;
    }
private:
    int cap_;
    std::vector<double> buf_;
    int size_ = 0;
    int head_ = 0;
};

// ----------------- Line Searcher -----------------
class LineSearcher {
public:
    LineSearcher(nb::object cfg, std::shared_ptr<Filter> filter = nullptr,
                 std::shared_ptr<Funnel> funnel = nullptr)
        : cfg_obj_(std::move(cfg)), filter_(std::move(filter)), funnel_(std::move(funnel)) {

        cfg_.ls_backtrack = std::clamp(getattr_or_double(cfg_obj_, "ls_backtrack", 0.5), 1e-4, 0.99);
        cfg_.ls_armijo_f  = std::max(1e-12, getattr_or_double(cfg_obj_, "ls_armijo_f", 1e-4));
        cfg_.ls_max_iter  = std::max(1, getattr_or_int(cfg_obj_, "ls_max_iter", 20));
        cfg_.ls_min_alpha = std::max(0.0, getattr_or_double(cfg_obj_, "ls_min_alpha", 1e-12));
        cfg_.ls_wolfe_c   = std::clamp(getattr_or_double(cfg_obj_, "ls_wolfe_c", 0.9), cfg_.ls_armijo_f, 0.999);

        cfg_.ip_fraction_to_boundary_tau =
            std::clamp(getattr_or_double(cfg_obj_, "ip_fraction_to_boundary_tau", 0.995), 0.9, 0.999);
        cfg_.ls_theta_restoration =
            std::max(1.0, getattr_or_double(cfg_obj_, "ls_theta_restoration", 1e3));

        cfg_.max_soc       = std::max(0, getattr_or_int(cfg_obj_, "max_soc", 4));
        cfg_.kappa_soc_min = std::clamp(getattr_or_double(cfg_obj_, "kappa_soc_min", 0.1), 0.01, 0.5);
        cfg_.kappa_soc_max = std::clamp(getattr_or_double(cfg_obj_, "kappa_soc_max", 0.99), 0.5, 0.99);
        cfg_.kappa_soc_base= std::clamp(getattr_or_double(cfg_obj_, "kappa_soc_base", 0.5),
                                        cfg_.kappa_soc_min, cfg_.kappa_soc_max);

        const std::string mode = nb::hasattr(cfg_obj_, "ls_merit_mode")
                                 ? nb::cast<std::string>(cfg_obj_.attr("ls_merit_mode"))
                                 : std::string("filter");
        if (mode == "monotone")         cfg_.merit_mode = LSMeritMode::MonotonePhi;
        else if (mode == "nonmonotone") cfg_.merit_mode = LSMeritMode::NonMonotonePhi;
        else                            cfg_.merit_mode = LSMeritMode::FilterOrFunnel;

        cfg_.nm_memory    = std::max(1, getattr_or_int(cfg_obj_, "ls_nm_memory", 5));
        cfg_.nm_safeguard = std::max(0.0, getattr_or_double(cfg_obj_, "ls_nm_safeguard", 0.0));

        cfg_.use_richardson = getattr_or_bool(cfg_obj_, "ls_use_richardson", true);
        cfg_.rich_tol       = std::max(1e-14, getattr_or_double(cfg_obj_, "ls_rich_tol", 1e-8));
        cfg_.rich_min_order = std::max(2, getattr_or_int(cfg_obj_, "ls_rich_min_order", 2));

        cfg_.ls_sufficient_decrease = std::max(1e-8, getattr_or_double(cfg_obj_, "ls_sufficient_decrease", 1e-4));
        cfg_.ls_curvature_condition = std::clamp(getattr_or_double(cfg_obj_, "ls_curvature_condition", 0.9),
                                                 cfg_.ls_sufficient_decrease, 0.999);
        cfg_.use_strong_wolfe = getattr_or_bool(cfg_obj_, "ls_use_strong_wolfe", false);
        cfg_.ls_theta_eps = std::max(1e-16, getattr_or_double(cfg_obj_, "ls_theta_eps", 1e-12));
        cfg_.ls_watchdog_max = std::max(1, getattr_or_int(cfg_obj_, "ls_watchdog_max", 10));
        cfg_.ls_theta_min_improvement =
            std::clamp(getattr_or_double(cfg_obj_, "ls_theta_min_improvement", 0.1), 0.01, 0.9);

        // Prepare small scratch buffers; resized on first use.
        z0_.resize(0);
        phi_hist_.reset(cfg_.nm_memory);
    }

    void reset_nonmonotone_history() { phi_hist_.reset(cfg_.nm_memory); }
    void seed_nonmonotone(double phi0) {
        if (std::isfinite(phi0)) { phi_hist_.reset(cfg_.nm_memory); phi_hist_.push(phi0); }
    }

    // Returns (alpha, iters, needs_restoration, dx_cor, ds_cor)
    std::tuple<double, int, bool, dvec, dvec>
    search(ModelC *model, const dvec &x, const dvec &dx, const dvec &ds,
           const dvec &s, double mu, double d_phi,
           std::optional<double> theta0_opt = std::nullopt,
           double alpha_max = 1.0) const
    {
        if (!model || x.size() == 0 || dx.size() != x.size()) {
            return {cfg_.ls_min_alpha, 0, true, dvec(), dvec()};
        }
        if (ds.size() > 0 && ds.size() != s.size()) {
            return {cfg_.ls_min_alpha, 0, true, dvec(), dvec()};
        }

        // ---- Base evaluation (fast path if available) ----
        if (!eval_base_(model, x)) {
            return {cfg_.ls_min_alpha, 0, true, dvec(), dvec()};
        }

        // Accessors
        double f0 = base_f_;
        const dvec& g0 = base_g_;
        const dvec* cE0 = base_cE_;
        const dvec* cI0 = base_cI_;
        const spmat* JE0 = base_JE_;
        const spmat* JI0 = base_JI_;

        if (!std::isfinite(f0)) return {cfg_.ls_min_alpha, 0, true, dvec(), dvec()};

        // Slack validation (vectorized)
        if (s.size() > 0) {
            if ((s.array() <= cfg_.ls_theta_eps).any()) {
                return {alpha_max, 0, true, dvec(), dvec()};
            }
        }

        const double barrier_eps = std::max(cfg_.ls_theta_eps, 1e-8 * mu);
        double phi0 = f0;
        if (s.size() > 0) {
            // phi0 = f0 - mu * sum(log(s))
            phi0 -= mu * (s.array().unaryExpr([&](double v){return safe_log_barrier(v, barrier_eps);})).sum();
        }
        if (!std::isfinite(phi0)) return {cfg_.ls_min_alpha, 0, true, dvec(), dvec()};

        double theta0 = theta0_opt ? *theta0_opt : compute_theta(cE0, cI0, s);

        // Descent check
        if (d_phi >= -cfg_.ls_theta_eps) {
            return {alpha_max, 0, true, dvec(), dvec()};
        }

        // Fraction-to-boundary (vectorized)
        double alpha_ftb = alpha_max;
        if (ds.size() == s.size() && s.size() > 0) {
            // For ds < 0: (1 - tau) * s / (-ds)
            Eigen::ArrayXd mask = (ds.array() < -cfg_.ls_theta_eps).cast<double>();
            if (mask.any()) {
                Eigen::ArrayXd ratio = ((1.0 - cfg_.ip_fraction_to_boundary_tau) * s.array()) / (-ds.array());
                double min_ratio = std::numeric_limits<double>::infinity();
                for (Eigen::Index i=0;i<ratio.size();++i)
                    if (mask[i] != 0.0) min_ratio = std::min(min_ratio, ratio[i]);
                if (std::isfinite(min_ratio)) alpha_ftb = std::min(alpha_ftb, min_ratio);
            }
        }
        alpha_ftb = std::max(alpha_ftb, cfg_.ls_min_alpha);

        // Predicted df, dtheta
        double pred_df = 0.0;
        if (g0.size() == dx.size()) pred_df = std::max(0.0, -g0.dot(dx));

        double theta_pred = 0.0;
        if (cE0 && cE0->size() > 0) {
            dvec je_dx = (JE0 && JE0->rows() > 0) ? (*JE0) * dx : Z_(cE0->size());
            dvec cE_pred = *cE0 + je_dx;
            theta_pred += cE_pred.lpNorm<1>();
        }
        if (cI0 && cI0->size() > 0) {
            dvec ji_dx = (JI0 && JI0->rows() > 0) ? (*JI0) * dx : Z_(cI0->size());
            dvec cI_pred = *cI0 + s + ds + ji_dx;
            theta_pred += cI_pred.array().max(0.0).sum();
        }
        const double pred_dtheta = std::max(0.0, theta0 - theta_pred);

        // Seed non-monotone memory if needed
        if (cfg_.merit_mode == LSMeritMode::NonMonotonePhi && phi_hist_.empty()) {
            phi_hist_.push(phi0);
        }

        RichardsonExtrapolator rich;

        double alpha = std::min(1.0, alpha_ftb);
        int it = 0;
        int watchdog = 0;
        double best_alpha = cfg_.ls_min_alpha;
        double best_phi   = std::numeric_limits<double>::max();
        double theta_t    = theta0;

        while (it < cfg_.ls_max_iter && watchdog < cfg_.ls_watchdog_max) {
            // Trial slack vector (vectorized check)
            dvec s_t;
            if (ds.size() == s.size()) {
                s_t = s + alpha * ds;
                if ((s_t.array() <= cfg_.ls_theta_eps).any()) {
                    alpha *= cfg_.ls_backtrack;
                    ++it; ++watchdog;
                    continue;
                }
            }

            // Evaluate f, cE, cI at x + α dx
            dvec x_t = x + alpha * dx;
            if (!eval_step_(model, x_t)) { // step eval failed
                alpha *= cfg_.ls_backtrack; ++it; ++watchdog; continue;
            }

            const double f_t = step_f_;
            if (!std::isfinite(f_t)) { alpha *= cfg_.ls_backtrack; ++it; ++watchdog; continue; }

            double phi_t = f_t;
            if (s_t.size() > 0) {
                phi_t -= mu * (s_t.array().unaryExpr([&](double v){return safe_log_barrier(v, barrier_eps);})).sum();
                if (!std::isfinite(phi_t)) { alpha *= cfg_.ls_backtrack; ++it; ++watchdog; continue; }
            }

            if (phi_t < best_phi) { best_phi = phi_t; best_alpha = alpha; }

            const dvec* cE_t = step_cE_;
            const dvec* cI_t = step_cI_;
            theta_t = compute_theta(cE_t, cI_t, s_t.size() ? s_t : s);

            // 1) Funnel/Filter acceptance
            if (funnel_) {
                if (funnel_->is_acceptable(theta0, f0, theta_t, f_t, pred_df, pred_dtheta)) {
                    funnel_->add_if_acceptable(theta0, f0, theta_t, f_t, pred_df, pred_dtheta);
                    push_phi_(phi_t);
                    return {alpha, it, false, dvec(), dvec()};
                }
            } else if (filter_) {
                if (filter_->is_acceptable(theta_t, f_t)) {
                    filter_->add_if_acceptable(theta_t, f_t);
                    push_phi_(phi_t);
                    return {alpha, it, false, dvec(), dvec()};
                }
            } else {
                // 2) Monotone / Non-monotone phi
                double phi_ref = phi0;
                if (cfg_.merit_mode == LSMeritMode::NonMonotonePhi && !phi_hist_.empty()) {
                    phi_ref = phi_hist_.max() + cfg_.nm_safeguard;
                }
                if (phi_t <= phi_ref + cfg_.ls_sufficient_decrease * alpha * d_phi) {
                    push_phi_(phi_t);
                    return {alpha, it, false, dvec(), dvec()};
                }
            }

            // 3) (Optional) Richardson extrapolation early in the search
            if (cfg_.use_richardson && it < cfg_.ls_max_iter / 2) {
                // concatenate without allocating via head/tail map
                if (scratch_dxds_.size() != dx.size() + ds.size()) scratch_dxds_.resize(dx.size() + ds.size());
                scratch_dxds_.head(dx.size()) = alpha * dx;
                scratch_dxds_.tail(ds.size()) = alpha * ds;
                rich.add_step(scratch_dxds_, alpha);
                auto ext = rich.extrapolate_step(scratch_dxds_, alpha, cfg_.rich_tol);
                if (ext.order_achieved >= cfg_.rich_min_order &&
                    std::isfinite(ext.error_estimate) &&
                    ext.error_estimate < cfg_.rich_tol) {

                    // split
                    if (scratch_refined_.size() != ext.dx_refined.size())
                        scratch_refined_.resize(ext.dx_refined.size());
                    scratch_refined_ = ext.dx_refined;

                    dvec dx_ref = scratch_refined_.head(dx.size());
                    dvec ds_ref = scratch_refined_.tail(ds.size());

                    double alpha_ref = 1.0;
                    if (ds_ref.size() == s.size() && s.size() > 0) {
                        Eigen::ArrayXd mask = (ds_ref.array() < -cfg_.ls_theta_eps).cast<double>();
                        if (mask.any()) {
                            Eigen::ArrayXd ratio = ((1.0 - cfg_.ip_fraction_to_boundary_tau) * s.array())
                                                 / (-ds_ref.array());
                            double min_ratio = std::numeric_limits<double>::infinity();
                            for (Eigen::Index i=0;i<ratio.size();++i)
                                if (mask[i] != 0.0) min_ratio = std::min(min_ratio, ratio[i]);
                            if (std::isfinite(min_ratio)) alpha_ref = std::min(alpha_ref, min_ratio);
                        }
                        alpha_ref = std::max(alpha_ref, cfg_.ls_min_alpha);
                    }

                    dvec x_ref = x + alpha_ref * dx_ref;
                    dvec s_ref = s.size() ? (s + alpha_ref * ds_ref) : dvec();

                    if (s_ref.size()==0 || (s_ref.array() > cfg_.ls_theta_eps).all()) {
                        if (eval_step_(model, x_ref)) {
                            const double f_ref = step_f_;
                            if (std::isfinite(f_ref)) {
                                double phi_ref_acc = f_ref;
                                if (s_ref.size() > 0) {
                                    phi_ref_acc -= mu * (s_ref.array().unaryExpr(
                                        [&](double v){return safe_log_barrier(v, barrier_eps);})).sum();
                                }
                                if (std::isfinite(phi_ref_acc)) {
                                    const dvec* cE_r = step_cE_;
                                    const dvec* cI_r = step_cI_;
                                    const double theta_ref = compute_theta(cE_r, cI_r, s_ref.size()?s_ref:s);
                                    bool ok = false;
                                    if (funnel_) {
                                        ok = funnel_->is_acceptable(theta0, f0, theta_ref, f_ref, pred_df, pred_dtheta);
                                        if (ok) funnel_->add_if_acceptable(theta0, f0, theta_ref, f_ref, pred_df, pred_dtheta);
                                    } else if (filter_) {
                                        ok = filter_->is_acceptable(theta_ref, f_ref);
                                        if (ok) filter_->add_if_acceptable(theta_ref, f_ref);
                                    } else {
                                        double phi_ref_cmp = phi0;
                                        if (cfg_.merit_mode == LSMeritMode::NonMonotonePhi && !phi_hist_.empty())
                                            phi_ref_cmp = phi_hist_.max() + cfg_.nm_safeguard;
                                        ok = (phi_ref_acc <= phi_ref_cmp + cfg_.ls_sufficient_decrease * alpha_ref * d_phi);
                                    }
                                    if (ok) {
                                        push_phi_(phi_ref_acc);
                                        return {alpha_ref, it, false, dx_ref, ds_ref};
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Backtrack (slightly stronger when watchdog is high)
            const double bt = (watchdog > cfg_.ls_watchdog_max/2) ? (cfg_.ls_backtrack * 0.5) : cfg_.ls_backtrack;
            alpha *= bt;
            ++it;
            if (theta_t < theta0) watchdog = 0; else ++watchdog;
        }

        const bool needs_restoration = (theta0 > cfg_.ls_theta_restoration) || (best_alpha <= cfg_.ls_min_alpha);
        return {best_alpha, it, needs_restoration, dvec(), dvec()};
    }

private:
    // ---- Fast-path base eval (f, g, cE, cI, JE, JI) ----
    bool eval_base_(ModelC* model, const dvec& x) const {
#ifdef MODEL_HAS_EVAL_MASK
        constexpr uint32_t BASE_MASK = EV_F | EV_G | EV_cE | EV_cI | EV_JE | EV_JI;
        if (!model->eval_all_mask(x, BASE_MASK)) return false;
#else
        try {
            static const std::vector<std::string> comps{"f","g","cE","cI","JE","JI"};
            model->eval_all(x, comps);
        } catch (...) { return false; }
#endif

#ifdef MODEL_HAS_VIEW
        auto v = model->get_view();
        if (!v.f) return false;
        base_f_  = *v.f;
        base_g_  = v.g ? *v.g : Z_(x.size());
        base_cE_ = v.cE;
        base_cI_ = v.cI;
        base_JE_ = v.JE;
        base_JI_ = v.JI;
#else
        {
            auto fopt = model->get_f();
            if (!fopt || !std::isfinite(*fopt)) return false;
            base_f_ = *fopt;
        }
        {
            auto gopt = model->get_g();
            if (gopt) base_g_ = *gopt; else base_g_ = Z_(x.size());
        }
        cE_store_ = model->get_cE();
        cI_store_ = model->get_cI();
        JE_store_ = model->get_JE();
        JI_store_ = model->get_JI();

        base_cE_ = cE_store_ ? &(*cE_store_) : nullptr;
        base_cI_ = cI_store_ ? &(*cI_store_) : nullptr;
        base_JE_ = JE_store_ ? &(*JE_store_) : nullptr;
        base_JI_ = JI_store_ ? &(*JI_store_) : nullptr;
#endif
        return true;
    }

    // ---- Fast-path step eval (f, cE, cI) ----
    bool eval_step_(ModelC* model, const dvec& x) const {
#ifdef MODEL_HAS_EVAL_MASK
        constexpr uint32_t STEP_MASK = EV_F | EV_cE | EV_cI;
        if (!model->eval_all_mask(x, STEP_MASK)) return false;
#else
        try {
            static const std::vector<std::string> comps{"f","cE","cI"};
            model->eval_all(x, comps);
        } catch (...) { return false; }
#endif

#ifdef MODEL_HAS_VIEW
        auto v = model->get_view();
        if (!v.f) return false;
        step_f_  = *v.f;
        step_cE_ = v.cE;
        step_cI_ = v.cI;
#else
        {
            auto fopt = model->get_f();
            if (!fopt || !std::isfinite(*fopt)) return false;
            step_f_ = *fopt;
        }
        step_cE_store_ = model->get_cE();
        step_cI_store_ = model->get_cI();
        step_cE_ = step_cE_store_ ? &(*step_cE_store_) : nullptr;
        step_cI_ = step_cI_store_ ? &(*step_cI_store_) : nullptr;
#endif
        return true;
    }

    // Sparse SOC least-squares on J = [JE; JI], r = [cE; (cI+s)]
    // Solves min ||J dx + r||_2 via SparseQR; no dense conversions.
    bool soc_least_squares_(const spmat* JE, const spmat* JI,
                            const dvec* rE, const dvec* rI,
                            dvec& dx_cor) const
    {
        const bool hasE = JE && rE && JE->rows() == rE->size() && JE->rows() > 0;
        const bool hasI = JI && rI && JI->rows() == rI->size() && JI->rows() > 0;
        if (!hasE && !hasI) return false;

        // Build stacked sparse J and rhs
        spmat J;
        dvec rhs;
        if (hasE && hasI) {
            J.resize(JE->rows() + JI->rows(), JE->cols());
            rhs.resize(J.rows());
            // reserve
            J.reserve(JE->nonZeros() + JI->nonZeros());
            // Top: JE
            for (int k=0;k<JE->outerSize();++k)
                for (spmat::InnerIterator it(*JE, k); it; ++it)
                    J.insert(it.row(), it.col()) = it.value();
            // Bottom: JI (row-shifted)
            const int rshift = JE->rows();
            for (int k=0;k<JI->outerSize();++k)
                for (spmat::InnerIterator it(*JI, k); it; ++it)
                    J.insert(it.row()+rshift, it.col()) = it.value();
            // rhs
            rhs.head(rE->size()) = *rE;
            rhs.tail(rI->size()) = *rI;
        } else if (hasE) {
            J = *JE;
            rhs = *rE;
        } else { // hasI
            J = *JI;
            rhs = *rI;
        }
        J.makeCompressed();

        // Solve J dx = -rhs in least-squares sense
        try {
            Eigen::SparseQR<spmat, Eigen::COLAMDOrdering<int>> qr;
            qr.compute(J);
            if (qr.info() != Eigen::Success) return false;
            dx_cor = qr.solve(-rhs);
            return (qr.info() == Eigen::Success) && (dx_cor.array().isFinite().all());
        } catch (...) {
            return false;
        }
    }

    void push_phi_(double phi) const {
        if (!std::isfinite(phi)) return;
        if (cfg_.merit_mode != LSMeritMode::NonMonotonePhi) return;
        phi_hist_.push(phi);
    }

    // Zero vector scratch (avoid allocating new zero vectors)
    const dvec& Z_(Eigen::Index n) const {
        if (z0_.size() != n) { z0_.resize(n); z0_.setZero(); }
        return z0_;
    }

private:
    nb::object cfg_obj_;
    std::shared_ptr<Filter> filter_;
    std::shared_ptr<Funnel> funnel_;
    LSConfig cfg_;

    // Cross-iteration non-monotone memory
    mutable RingBuffer phi_hist_;

    // Scratch (mutable for const search)
    mutable dvec z0_;
    mutable dvec scratch_dxds_;
    mutable dvec scratch_refined_;

    // Cached base step (fast-path storage)
    mutable double base_f_{std::numeric_limits<double>::infinity()};
    mutable dvec   base_g_;

#ifdef MODEL_HAS_VIEW
    mutable const dvec*  base_cE_{nullptr};
    mutable const dvec*  base_cI_{nullptr};
    mutable const spmat* base_JE_{nullptr};
    mutable const spmat* base_JI_{nullptr};
#else
    mutable std::optional<dvec>  cE_store_;
    mutable std::optional<dvec>  cI_store_;
    mutable std::optional<spmat> JE_store_;
    mutable std::optional<spmat> JI_store_;
    mutable const dvec*  base_cE_{nullptr};
    mutable const dvec*  base_cI_{nullptr};
    mutable const spmat* base_JE_{nullptr};
    mutable const spmat* base_JI_{nullptr};
#endif

    // Cached trial step (fast-path storage)
    mutable double step_f_{std::numeric_limits<double>::infinity()};
#ifdef MODEL_HAS_VIEW
    mutable const dvec* step_cE_{nullptr};
    mutable const dvec* step_cI_{nullptr};
#else
    mutable std::optional<dvec> step_cE_store_;
    mutable std::optional<dvec> step_cI_store_;
    mutable const dvec* step_cE_{nullptr};
    mutable const dvec* step_cI_{nullptr};
#endif
};
