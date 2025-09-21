// nlp_hybrid_cpp_nb.cpp
// C++23 + nanobind hybrid NLP solver wrapper (IP + SQP)
// Build target name suggestion: `chomp`
//
// Requires:
//   - nanobind (with Eigen support)
//   - Eigen
//   - fmt
//   - Your C++ IP/SQP headers: ../include/ip.h, ../include/sqp.h
//
// Python side expectations (unchanged):
//   - nlp.blocks.aux: Model, HessianManager, RestorationManager, SQPConfig
//   - nlp.blocks.reg: Regularizer
//   - nlp.blocks.qp : QPSolver
//
// Notes:
//   * We accept/return Eigen vectors (nanobind auto-converts to/from NumPy).
//   * Bounds (lb/ub) can be None or 1-D arrays; we forward as Eigen vectors.

#include <fmt/core.h>
#include <fmt/color.h>
#include <fmt/format.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/eigen/dense.h>

#include <Eigen/Core>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "../include/ip.h"
#include "../include/sqp.h"

namespace nb = nanobind;
using nb::arg;
using dvec = Eigen::VectorXd;

// -------------------- small helpers --------------------

inline bool has_attr(const nb::object &o, const char *name) {
    return nb::hasattr(o, name);
}

template <typename T>
inline void ensure_attr(nb::object &cfg, const char *name, T value) {
    if (!nb::hasattr(cfg, name)) {
        cfg.attr(name) = nb::cast(std::move(value));
    }
}

inline double get_float_attr_or(const nb::object &o, const char *name, double defval) {
    if (!nb::hasattr(o, name)) return defval;
    try { return nb::cast<double>(o.attr(name)); }
    catch (...) { return defval; }
}

inline std::string get_str_attr_or(const nb::object &o, const char *name, const std::string &defval) {
    if (!nb::hasattr(o, name)) return defval;
    try { return nb::cast<std::string>(o.attr(name)); }
    catch (...) { return defval; }
}

inline bool get_bool_attr_or(const nb::object &o, const char *name, bool defval) {
    if (!nb::hasattr(o, name)) return defval;
    try { return nb::cast<bool>(o.attr(name)); }
    catch (...) { return defval; }
}

inline nb::object import_attr(const char *mod, const char *name) {
    nb::module_ m = nb::module_::import_(mod);
    return m.attr(name);
}

// -------------------- Printer --------------------
static void print_iteration_row(int k, const SolverInfo &info,
                                const std::string &mode, int last_header_ref,
                                bool force_header = false) {
    using fmt::color;
    using fmt::emphasis;
    using fmt::fg;

    static bool banner_printed = false;
    static std::optional<double> f_prev;
    static std::optional<double> theta_prev;

    if (!banner_printed) {
        fmt::print(fg(color::cyan) | emphasis::bold, "\nCHOMP");
        fmt::print(fg(color::light_gray) | emphasis::bold, " — made by ");
        fmt::print(fg(color::white) | emphasis::bold, "L. O. Seman\n");
        banner_printed = true;
    }

    if (force_header || k == 0 || (k - last_header_ref) >= 20) {
        fmt::print(fg(color::light_gray) | emphasis::bold,
                   " {:>3s} {:>3s} {:>12s} {:>13s} {:>11s} {:>9s} {:>9s}\n",
                   "k", "st", "step", "f", "theta", "alpha", "Δ");
    }

    auto trend_color = [](std::optional<double> prev, double val) -> fmt::color {
        if (!prev.has_value() || std::isnan(val)) return color::white;
        return color::white; // (simple neutral trend color for now)
    };
    const auto f_col     = trend_color(f_prev, info.f);
    const auto theta_col = trend_color(theta_prev, info.theta);
    const auto st_col    = info.accepted ? color::green : color::red;

    fmt::print(" {:>3d} ", k);
    fmt::print(fg(st_col) | emphasis::bold, "{:>3s} ", info.accepted ? "A" : "R");

    fmt::print("{:>12.3e} ", info.step_norm);
    fmt::print(fg(f_col),     "{:>13.6e} ", info.f);
    fmt::print(fg(theta_col), "{:>11.3e} ", info.theta);
    fmt::print("{:>9.2e} {:>9.2e}\n", info.alpha, info.tr_radius);

    if (!std::isnan(info.f))     f_prev     = info.f;
    if (!std::isnan(info.theta)) theta_prev = info.theta;
}

// ==================== NLPSolver (nanobind) ====================

class NLPSolverCPP {
public:
    NLPSolverCPP(nb::object f,
                 nb::object c_ineq_list,   // list[Callable] or None
                 nb::object c_eq_list,     // list[Callable] or None
                 nb::object lb_or_none,    // None or 1-D array-like
                 nb::object ub_or_none,    // None or 1-D array-like
                 const dvec &x0,           // 1-D
                 nb::object cfg_or_none)   // None or SQPConfig-like
    {
        // --- config ---
        cfg_ = cfg_or_none.is_none()
                   ? nb::module_::import_("nlp.blocks.aux").attr("SQPConfig")()
                   : cfg_or_none;
        ensure_auto_defaults_(cfg_);

        // --- x0 (state) ---
        x_ = x0;
        n_ = static_cast<int>(x_.size());

        // Default empty bounds
        nb::object lb_py = nb::cast(dvec()); // size 0
        nb::object ub_py = nb::cast(dvec()); // size 0

        // lb
        if (!lb_or_none.is_none()) {
            try {
                dvec lb_vec = nb::cast<dvec>(lb_or_none);
                if (lb_vec.size() > 0) lb_py = nb::cast(std::move(lb_vec));
            } catch (...) {
                // keep empty if cast fails
            }
        }
        // ub
        if (!ub_or_none.is_none()) {
            try {
                dvec ub_vec = nb::cast<dvec>(ub_or_none);
                if (ub_vec.size() > 0) ub_py = nb::cast(std::move(ub_vec));
            } catch (...) {
                // keep empty if cast fails
            }
        }

        // --- Model: Model(f, c_ineq, c_eq, n, lb, ub) ---
        auto Model = import_attr("nlp.blocks.aux", "Model");
        nb::object cI = c_ineq_list.is_none() ? nb::none() : c_ineq_list;
        nb::object cE = c_eq_list.is_none() ? nb::none() : c_eq_list;

        auto t0 = std::chrono::high_resolution_clock::now();
        model_ = Model(f, cI, cE, n_, lb_py, ub_py);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = t1 - t0;
        std::cout << "Model creation took " << elapsed.count() << " seconds.\n";

        // Managers (Python side)
        auto HessianManager      = import_attr("nlp.blocks.aux", "HessianManager");
        auto RestorationManager  = import_attr("nlp.blocks.aux", "RestorationManager");
        auto Regularizer         = import_attr("nlp.blocks.reg", "Regularizer");
        auto QPSolver            = import_attr("nlp.blocks.qp",  "QPSolver");

        hess_ = HessianManager(n_, cfg_);
        rest_ = RestorationManager(cfg_);
        reg_  = Regularizer(cfg_);
        qp_   = QPSolver(cfg_, Regularizer(cfg_));

        // Equalities/inequalities sizes
        mI_ = cI.is_none() ? 0 : static_cast<int>(nb::len(cI));
        mE_ = cE.is_none() ? 0 : static_cast<int>(nb::len(cE));

        // Multipliers
        lam_ = dvec::Zero(mI_);
        nu_  = dvec::Zero(mE_);

        // Mode selection
        std::string mode = get_str_attr_or(cfg_, "mode", "auto");
        if (mode != "ip" && mode != "sqp" && mode != "auto" && mode != "dfo")
            mode = "auto";
        mode_ = mode;

        if (mode_ == "auto") {
            double theta0 = nb::cast<double>(model_.attr("constraint_violation")(x_));
            double ip_switch_theta = get_float_attr_or(cfg_, "ip_switch_theta", 1e-3);
            double tol_feas        = get_float_attr_or(cfg_, "tol_feas",       1e-6);
            mode_ = (theta0 > std::max(ip_switch_theta, 10.0 * tol_feas)) ? "ip" : "sqp";
        }

        // IP stepper + state
        ip_state_   = IPState();                    // default-init
        ip_stepper_ = new InteriorPointStepper(cfg_, hess_);

        // SQP stepper
        // sqp_stepper_ = new SQPStepper(cfg_, hess_, qp_, reg_, rest_);

        // Trackers
        last_header_row_ = -1;
        last_switch_iter_ = -1000000000;
        prev_theta_.reset();
        reject_streak_ = small_alpha_streak_ = no_progress_streak_ = 0;
    }

    // Solve; returns the final x as an Eigen vector (NumPy array in Python)
    dvec solve(int max_iter = 100, double tol = 1e-8, bool verbose = true) {
        const double tol_stat = get_float_attr_or(cfg_, "tol_stat", 1e-6);
        const double tol_feas = get_float_attr_or(cfg_, "tol_feas", 1e-6);
        const double tol_comp = get_float_attr_or(cfg_, "tol_comp", 1e-6);

        for (int k = 0; k < max_iter; ++k) {
            SolverInfo info;
            if (mode_ == "ip") {
                info = ip_step_(k);
            } else if (mode_ == "sqp") {
                // info = sqp_step_(k);
            } else if (mode_ == "dfo") {
                // DFO not wired here; keep placeholder behavior
                info.accepted = false;
                info.mode = "dfo";
            } else {
                throw std::runtime_error("Unknown mode: " + mode_);
            }

            if (verbose) {
                print_iteration_row(k, info, mode_, last_header_row_,
                                    (k == 0 || (k - last_header_row_) >= 20));
                if (k == 0 || (k - last_header_row_) >= 20)
                    last_header_row_ = k;
            }

            const double k_stat = info.stat;
            const double k_ineq = info.ineq;
            const double k_eq   = info.eq;
            const double k_comp = info.comp;

            if (k_stat <= tol_stat && k_ineq <= tol_feas && k_eq <= tol_feas && k_comp <= tol_comp) {
                if (verbose) fmt::print("✓ Converged at iteration {}\n", k);
                break;
            }
        }
        return x_;
    }

private:
    // ---- steps --------------------------------------------------------------
    SolverInfo ip_step_(int it) {
        auto [x_out, lam_out, nu_out, info] =
            ip_stepper_->step(model_, x_, lam_, nu_, it, ip_state_);
        if (info.accepted) {
            x_   = std::move(x_out);
            lam_ = std::move(lam_out);
            nu_  = std::move(nu_out);
        }
        return info;
    }

    // SolverInfo sqp_step_(int it) {
    //     auto [x_out, lam_out, nu_out, info] =
    //         sqp_stepper_->step(model_, x_, lam_, nu_, it);

    //     if (info.accepted) {
    //         if (get_bool_attr_or(cfg_, "use_watchdog", false)) {
    //             watchdog_update_(x_out); // stub (no-op)
    //         }
    //         x_   = std::move(x_out);
    //         lam_ = std::move(lam_out);
    //         nu_  = std::move(nu_out);
    //     }
    //     return info;
    // }

    // ---- config defaults ----------------------------------------------------
    void ensure_auto_defaults_(nb::object &cfg) {
        // Initial mode decision
        ensure_attr(cfg, "ip_switch_theta", 1e-3);

        // IP → SQP switch
        ensure_attr(cfg, "auto_ip2sqp_theta_cut", 5e-5);
        ensure_attr(cfg, "auto_ip2sqp_mu_cut",    1e-6);
        ensure_attr(cfg, "auto_ip_min_iters",     3);

        // SQP → IP switch
        ensure_attr(cfg, "auto_sqp2ip_theta_blowup",      1e-2);
        ensure_attr(cfg, "auto_sqp2ip_stall_iters",       3);
        ensure_attr(cfg, "auto_sqp2ip_reject_streak",     2);
        ensure_attr(cfg, "auto_sqp2ip_small_alpha_streak",3);
        ensure_attr(cfg, "auto_sqp_min_iters",            3);

        // Hysteresis and spacing
        ensure_attr(cfg, "auto_hysteresis_factor",            2.0);
        ensure_attr(cfg, "auto_min_iter_between_switches",    2);

        // Small step detection
        ensure_attr(cfg, "auto_small_alpha", 1e-6);
    }

    void watchdog_update_(const dvec &x_cand) {
        (void)x_cand; // stub no-op (port your Python watchdog if/when needed)
    }

private:
    // Core problem bits
    nb::object cfg_;
    nb::object model_;
    nb::object hess_;
    nb::object rest_;
    nb::object reg_;
    nb::object qp_;

    // Steppers
    InteriorPointStepper *ip_stepper_ = nullptr;
    IPState               ip_state_;
    SQPStepper           *sqp_stepper_ = nullptr;

    // State
    dvec x_;
    dvec lam_;
    dvec nu_;
    int  n_{0}, mI_{0}, mE_{0};

    // Mode + trackers
    std::string mode_;
    int last_header_row_{-1};
    int last_switch_iter_{-1000000000};
    std::optional<double> prev_theta_;
    int reject_streak_{0};
    int small_alpha_streak_{0};
    int no_progress_streak_{0};
};

// -------------------- nanobind module --------------------

NB_MODULE(chomp, m) {
    m.doc() = "Hybrid NLP Solver (IP + SQP) — nanobind wrapper";

    nb::class_<NLPSolverCPP>(m, "NLPSolver")
        .def(nb::init<nb::object, nb::object, nb::object,
                      nb::object, nb::object,
                      const dvec&, nb::object>(),
             arg("f"),
             arg("c_ineq") = nb::none(),
             arg("c_eq")   = nb::none(),
             arg("lb")     = nb::none(),
             arg("ub")     = nb::none(),
             arg("x0"),
             arg("config") = nb::none())
        .def("solve", &NLPSolverCPP::solve,
             arg("max_iter") = 100,
             arg("tol")      = 1e-8,
             arg("verbose")  = true,
             "Run hybrid solve; returns the final x (Eigen/NumPy).");
}
