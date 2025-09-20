// nlp_hybrid_cpp.cpp
// C++23 + pybind11 hybrid NLP solver wrapper mirroring nlp_hybrid.py
//
// Build: compile into a module (e.g., target name `nlp_hybrid_cpp`)
// Requires: pybind11, Eigen, and your existing ip_cpp / sqp_cpp Python modules.

#include <fmt/core.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Core>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "fmt/color.h"
#include "fmt/format.h"

#include "../include/ip.h"
#include "../include/sqp.h"

namespace py = pybind11;
using dvec = Eigen::VectorXd;

namespace {

// ---- small helpers ---------------------------------------------------------

inline bool has_attr(const py::object &o, const char *name) {
    return py::hasattr(o, name);
}

template <typename T>
inline void ensure_attr(py::object &cfg, const char *name, T value) {
    if (!py::hasattr(cfg, name)) {
        cfg.attr(name) = py::cast(value);
    }
}

inline double get_float_attr_or(const py::object &o, const char *name,
                                double defval) {
    if (!py::hasattr(o, name))
        return defval;
    try {
        return py::cast<double>(o.attr(name));
    } catch (...) {
        return defval;
    }
}

inline std::string get_str_attr_or(const py::object &o, const char *name,
                                   const std::string &defval) {
    if (!py::hasattr(o, name))
        return defval;
    try {
        return py::cast<std::string>(o.attr(name));
    } catch (...) {
        return defval;
    }
}

inline int get_int_attr_or(const py::object &o, const char *name, int defval) {
    if (!py::hasattr(o, name))
        return defval;
    try {
        return py::cast<int>(o.attr(name));
    } catch (...) {
        return defval;
    }
}

inline py::object import_attr(const char *mod, const char *name) {
    py::module m = py::module::import(mod);
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
        // Banner
        fmt::print(fg(color::cyan) | emphasis::bold, "\nCHOMP");
        fmt::print(fg(color::light_gray) | emphasis::bold, " — made by ");
        fmt::print(fg(color::yellow) | emphasis::bold, "L. O. Seman\n");

        // Preamble
        // if (info.n >= 0) {
        //     fmt::print(fg(color::light_gray), "Problem: ");
        //     fmt::print("n={}  mI={}  mE={}\n", info.n, info.mI, info.mE);
        // }
        // if (!mode.empty() || !info.hess.empty() || !info.reg.empty() || !info.backend.empty()) {
        //     fmt::print(fg(color::light_gray), "Config:  ");
        //     if (!mode.empty())        fmt::print("mode={}  ", mode);
        //     if (!info.hess.empty())   fmt::print("hess={}  ", info.hess);
        //     if (!info.reg.empty())    fmt::print("reg={}  ", info.reg);
        //     if (!info.backend.empty())fmt::print("lin.solver={}  ", info.backend);
        //     fmt::print("\n");
        // }
        // if (info.max_iter >= 0 || !std::isnan(info.tol)) {
        //     fmt::print(fg(color::light_gray), "Limits:  ");
        //     if (info.max_iter >= 0) fmt::print("max_iter={}  ", info.max_iter);
        //     if (!std::isnan(info.tol)) fmt::print("tol={:.2e}  ", info.tol);
        //     fmt::print("\n");
        // }

        // Header
        // fmt::print(fg(color::light_gray) | emphasis::bold,
        //            "\n {:>3s} {:>3s} {:>5s} {:>8s} {:>12s} {:>13s} {:>11s} {:>9s} {:>9s}\n",
        //            "k", "st", "mode", "sw", "step", "f", "theta", "alpha", "Δ");
        banner_printed = true;
    }

    // Periodic header
    if (force_header || k == 0 || (k - last_header_ref) >= 20) {
        fmt::print(fg(color::light_gray) | emphasis::bold,
                   " {:>3s} {:>3s} {:>5s} {:>8s} {:>12s} {:>13s} {:>11s} {:>9s} {:>9s}\n",
                   "k", "st", "mode", "sw", "step", "f", "theta", "alpha", "Δ");
    }

    // Trend-based color selection
    auto trend_color = [](std::optional<double> prev, double val) -> fmt::color {
        if (!prev.has_value() || std::isnan(val)) return color::white;
        if (val < prev.value()) return color::green;
        if (val > prev.value()) return color::red;
        return color::white;
    };
    const auto f_col     = trend_color(f_prev, info.f);
    const auto theta_col = trend_color(theta_prev, info.theta);
    const auto st_col    = info.accepted ? color::green : color::red;

    // Row
    fmt::print(" {:>3d} ", k);
    fmt::print(fg(st_col) | emphasis::bold, "{:>3s} ", info.accepted ? "A" : "R");
    const std::string mode3 = mode.size() >= 3 ? mode.substr(0, 3) : mode;
    // fmt::print("[{:>3s}] {:>8s} ", mode3, info.switched_to);

    fmt::print("{:>12.3e} ", info.step_norm);
    fmt::print(fg(f_col),     "{:>13.6e} ", info.f);
    fmt::print(fg(theta_col), "{:>11.3e} ", info.theta);
    fmt::print("{:>9.2e} {:>9.2e}\n", info.alpha, info.tr_radius);

    // Update trends
    if (!std::isnan(info.f))     f_prev     = info.f;
    if (!std::isnan(info.theta)) theta_prev = info.theta;
}

} // namespace

class NLPSolverCPP {
public:
    NLPSolverCPP(py::object f,
                 py::object c_ineq_list, // list[Callable] or None
                 py::object c_eq_list,   // list[Callable] or None
                 py::array_t<double> lb_or_none, py::array_t<double> ub_or_none,
                 py::array_t<double> x0, py::object cfg_or_none) {

        // --- config ---
        cfg_ = cfg_or_none.is_none()
                   ? py::module::import("nlp.blocks.aux").attr("SQPConfig")()
                   : cfg_or_none;
        ensure_auto_defaults_(cfg_);

        // --- x0 (state) ---
        x_ = eigen_vec_from(x0);
        n_ = static_cast<int>(x_.size());

        // --- arrays lb/ub (may be None) ---
        auto is_empty_bounds =
            [](const py::array_t<double> &arr_param) -> bool {
            try {
                if (arr_param.is_none())
                    return true;

                py::array_t<double> arr =
                    py::cast<py::array_t<double>>(arr_param);
                return arr.size() == 0;
            } catch (...) {
                return true; // If we can't cast it, treat as None
            }
        };

        // --- arrays lb/ub (may be None or empty) ---
        // Helper function to create empty numpy array
        auto create_empty_array = []() -> py::array_t<double> {
            return py::array_t<double>(0); // Create empty 1D array
        };

        // Default to empty arrays instead of None
        py::object lb_py = create_empty_array();
        py::object ub_py = create_empty_array();

        // Handle lb
        if (!lb_or_none.is_none()) {
            py::array_t<double> lb_arr =
                py::cast<py::array_t<double>>(lb_or_none);
            if (lb_arr.size() > 0) {
                // Use the provided non-empty array
                lb_py = py::reinterpret_borrow<py::object>(lb_or_none);
            }
            // If array is empty, keep the default empty array
        }

        // Handle ub
        if (!ub_or_none.is_none()) {
            py::array_t<double> ub_arr =
                py::cast<py::array_t<double>>(ub_or_none);
            if (ub_arr.size() > 0) {
                // Use the provided non-empty array
                ub_py = py::reinterpret_borrow<py::object>(ub_or_none);
            }
            // If array is empty, keep the default empty array
        }

        // --- Model: Model(f, c_ineq, c_eq, n, lb, ub) ---
        auto Model = import_attr("nlp.blocks.aux", "Model");
        py::object cI = c_ineq_list.is_none() ? py::none() : c_ineq_list;
        py::object cE = c_eq_list.is_none() ? py::none() : c_eq_list;
        auto start_time = std::chrono::high_resolution_clock::now();
        model_ = Model(f, cI, cE, n_, lb_py, ub_py);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        std::cout << "Model creation took " << elapsed.count() << " seconds.\n";
        // Managers (Python-side re-use)
        auto HessianManager = import_attr("nlp.blocks.aux", "HessianManager");
        auto RestorationManager =
            import_attr("nlp.blocks.aux", "RestorationManager");
        auto Regularizer = import_attr("nlp.blocks.reg", "Regularizer");
        auto QPSolver = import_attr("nlp.blocks.qp", "QPSolver");

        hess_ = HessianManager(n_, cfg_);
        rest_ = RestorationManager(cfg_);
        reg_ = Regularizer(cfg_);

        qp_ = QPSolver(cfg_, Regularizer(cfg_)); // matches your Python ctor

        // Equalities/inequalities sizes
        mI_ = cI.is_none() ? 0 : py::len(cI);
        mE_ = cE.is_none() ? 0 : py::len(cE);

        // Multipliers
        lam_ = dvec::Zero(mI_);
        nu_ = dvec::Zero(mE_);

        // Mode selection
        std::string mode = get_str_attr_or(cfg_, "mode", "auto");
        if (mode != "ip" && mode != "sqp" && mode != "auto" && mode != "dfo")
            mode = "auto";
        mode_ = mode;

        if (mode_ == "auto") {
            double theta0 = py::cast<double>(
                model_.attr("constraint_violation")(eigen_to_numpy(x_)));
            double ip_switch_theta =
                get_float_attr_or(cfg_, "ip_switch_theta", 1e-3);
            double tol_feas = get_float_attr_or(cfg_, "tol_feas", 1e-6);
            mode_ = (theta0 > std::max(ip_switch_theta, 10.0 * tol_feas))
                        ? "ip"
                        : "sqp";
        }

        // IP stepper + state
        // auto ip_mod = py::module::import("ip_cpp");
        // py::object IPState = ip_mod.attr("IPState");
        // py::object InteriorPointStepper =
        // ip_mod.attr("InteriorPointStepper");

        ip_state_ = IPState(); // empty init, like Python
        ip_stepper_ = new InteriorPointStepper(model_, cfg_, hess_);

        // SQP stepper
        sqp_stepper_ = new SQPStepper(cfg_, hess_, qp_, reg_, rest_);

        // Watchdog & autoswitch trackers (kept minimal here)
        last_header_row_ = -1;
        last_switch_iter_ = -1000000000;
        prev_theta_ = std::nullopt;
        reject_streak_ = small_alpha_streak_ = no_progress_streak_ = 0;
    }

    // Returns (x: np.ndarray, hist: list[dict])
    py::array_t<double> solve(int max_iter = 100, double tol = 1e-8,
                              bool verbose = true) {
        py::list hist;

        // for this port we force starting in IP (matching your current Python
        // snippet)
        // add timer and init time count
        // create totaltime var
        const double tol_stat = get_float_attr_or(cfg_, "tol_stat", 1e-6);
        const double tol_feas = get_float_attr_or(cfg_, "tol_feas", 1e-6);
        const double tol_comp = get_float_attr_or(cfg_, "tol_comp", 1e-6);
        double k_stat = std::numeric_limits<double>::infinity();
        double k_ineq = std::numeric_limits<double>::infinity();
        double k_eq = std::numeric_limits<double>::infinity();
        double k_comp = std::numeric_limits<double>::infinity();
        // auto total_start_time = std::chrono::high_resolution_clock::now();
        for (int k = 0; k < max_iter; ++k) {
            // auto start_time = std::chrono::high_resolution_clock::now();
            SolverInfo info;
            if (mode_ == "ip") {
                info = ip_step_(k);
            } else if (mode_ == "sqp") {
                info = sqp_step_(k);
            } else if (mode_ == "dfo") {
                // Not implemented here; keep behavior similar to your Python
                // snippet. You can wire your DFO path from C++ later if needed.
                py::dict info;
                info["accepted"] = false;
                info["mode"] = "dfo";
            } else {
                throw std::runtime_error("Unknown mode: " + mode_);
            }
            // add elapsed time to info
            // auto end_time = std::chrono::high_resolution_clock::now();
            // std::chrono::duration<double> elapsed = end_time - start_time;
            // std::cout << "Iteration " << k << " took " << elapsed.count()
            //           << " seconds.\n";
            // hist.append(info);

            if (verbose) {
                print_iteration_row(
                    k, info, mode_, last_header_row_,
                    /*force_header*/ (k == 0 || (k - last_header_row_) >=
                    20));
                if (k == 0 || (k - last_header_row_) >= 20)
                    last_header_row_ = k;
            }

            // auto time_to_convert = std::chrono::high_resolution_clock::now();
            k_stat = info.stat;
            k_ineq = info.ineq;
            k_eq = info.eq;
            k_comp = info.comp;
            // auto time_after_convert = std::chrono::high_resolution_clock::now();
            // std::chrono::duration<double> convert_elapsed =
                // time_after_convert - time_to_convert;
            // std::cout << "Conversion took " << convert_elapsed.count()
            //           << " seconds.\n";

            if (k_stat <= tol_stat && k_ineq <= tol_feas && k_eq <= tol_feas &&
                k_comp <= tol_comp) {
                if (verbose)
                    py::print("✓ Converged at iteration", k);
                break;
            }

            // Optional: auto-switch logic can be re-enabled here if desired.
        }
        // auto total_end_time = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> total_elapsed =
        //     total_end_time - total_start_time;
        // std::cout << "Total solve time: " << total_elapsed.count()
        //           << " seconds.\n";

        return eigen_to_numpy(x_);
    }

private:
    // ---- steps --------------------------------------------------------------

    SolverInfo ip_step_(int it) {
        auto step_ret = ip_stepper_->step( // model
            x_,                            // x
            lam_,                          // lam
            nu_,                           // nu
            it,                            // iteration
            ip_state_                      // state
        );
        // Python returns: (x_out, lam_out, nu_out, info)
        auto [x_out, lam_out, nu_out, info] = step_ret;

        const bool accepted = info.accepted;
        if (accepted) {
            x_ = std::move(x_out);
            lam_ = std::move(lam_out);
            nu_ = std::move(nu_out);
        }
        return info;
    }

    SolverInfo sqp_step_(int it) {
        auto step_ret = sqp_stepper_->step(model_, x_, lam_, nu_, it);

        auto [x_out, lam_out, nu_out, info] = step_ret;

        // update_streaks_(info);

        const bool accepted = info.accepted;
        if (accepted) {
            if (get_bool_attr_or(cfg_, "use_watchdog", false)) {
                watchdog_update_(
                    x_out); // safe no-op stub here; can be expanded
            }
            x_ = std::move(x_out);
            lam_ = std::move(lam_out);
            nu_ = std::move(nu_out);
        }
        return info;
    }

    // ---- helpers ------------------------------------------------------------

    static dvec eigen_vec_from(const py::object &arr_like) {
        // Accept numpy arrays (1-D) and cast to VectorXd
        py::array a = py::cast<py::array>(arr_like);
        py::buffer_info bi = a.request();
        if (bi.ndim != 1)
            throw std::runtime_error("Expected 1-D array");
        auto *data = static_cast<const double *>(bi.ptr);
        dvec v(bi.size);
        for (ssize_t i = 0; i < bi.size; ++i)
            v[i] = data[i];
        return v;
    }

    static py::array_t<double> eigen_to_numpy(const dvec &v) {
        return py::array_t<double>(static_cast<ssize_t>(v.size()), v.data());
    }

    static bool get_bool_attr_or(const py::object &o, const char *name,
                                 bool defval) {
        if (!py::hasattr(o, name))
            return defval;
        try {
            return py::cast<bool>(o.attr(name));
        } catch (...) {
            return defval;
        }
    }

    void ensure_auto_defaults_(py::object &cfg) {
        // Initial mode decision
        ensure_attr(cfg, "ip_switch_theta", 1e-3);

        // IP → SQP switch
        ensure_attr(cfg, "auto_ip2sqp_theta_cut", 5e-5);
        ensure_attr(cfg, "auto_ip2sqp_mu_cut", 1e-6);
        ensure_attr(cfg, "auto_ip_min_iters", 3);

        // SQP → IP switch
        ensure_attr(cfg, "auto_sqp2ip_theta_blowup", 1e-2);
        ensure_attr(cfg, "auto_sqp2ip_stall_iters", 3);
        ensure_attr(cfg, "auto_sqp2ip_reject_streak", 2);
        ensure_attr(cfg, "auto_sqp2ip_small_alpha_streak", 3);
        ensure_attr(cfg, "auto_sqp_min_iters", 3);

        // Hysteresis and spacing
        ensure_attr(cfg, "auto_hysteresis_factor", 2.0);
        ensure_attr(cfg, "auto_min_iter_between_switches", 2);

        // Small step detection
        ensure_attr(cfg, "auto_small_alpha", 1e-6);
    }

    void update_streaks_(const py::dict &info) {
        bool accepted = info.contains("accepted")
                            ? py::cast<bool>(info["accepted"])
                            : false;
        double alpha =
            info.contains("alpha") ? py::cast<double>(info["alpha"]) : 1.0;
        double theta = info.contains("theta")
                           ? py::cast<double>(info["theta"])
                           : py::cast<double>(model_.attr(
                                 "constraint_violation")(eigen_to_numpy(x_)));

        reject_streak_ = accepted ? 0 : (reject_streak_ + 1);

        const double small_alpha_cut =
            get_float_attr_or(cfg_, "auto_small_alpha", 1e-6);
        small_alpha_streak_ =
            (alpha <= small_alpha_cut) ? (small_alpha_streak_ + 1) : 0;

        if (!prev_theta_.has_value()) {
            no_progress_streak_ = 0;
        } else {
            bool improved = (theta <= prev_theta_.value() * 0.99) ||
                            (theta <= prev_theta_.value() - 1e-14);
            no_progress_streak_ = improved ? 0 : (no_progress_streak_ + 1);
        }
        prev_theta_ = theta;
    }

    void watchdog_update_(const dvec &x_cand) {
        // Minimal safe stub (your Python has full logic; you can port it if
        // desired)
        (void)x_cand;
    }

private:
    // Core problem bits
    py::object cfg_;
    py::object model_;
    py::object hess_;
    py::object rest_;
    py::object reg_;
    py::object qp_;

    // Steppers
    InteriorPointStepper *ip_stepper_ = nullptr;
    IPState ip_state_;
    SQPStepper *sqp_stepper_ = nullptr;

    // State
    dvec x_;
    dvec lam_;
    dvec nu_;
    int n_{0}, mI_{0}, mE_{0};

    // Mode + trackers
    std::string mode_;
    int last_header_row_{-1};
    int last_switch_iter_{-1000000000};
    std::optional<double> prev_theta_;
    int reject_streak_{0};
    int small_alpha_streak_{0};
    int no_progress_streak_{0};
};

// ---- pybind module ----------------------------------------------------------

PYBIND11_MODULE(chomp, m) {
    m.doc() = "Hybrid NLP Solver (IP + SQP) — modern C++ wrapper for your "
              "Python ecosystem";

    py::class_<NLPSolverCPP>(m, "NLPSolver")
        .def(py::init<py::object, py::object, py::object, py::array_t<double>,
                      py::array_t<double>, py::array_t<double>, py::object>(),
             py::arg("f"), py::arg("c_ineq") = py::none(),
             py::arg("c_eq") = py::none(), py::arg("lb") = py::none(),
             py::arg("ub") = py::none(), py::arg("x0"),
             py::arg("config") = py::none())
        .def("solve", &NLPSolverCPP::solve, py::arg("max_iter") = 2,
             py::arg("tol") = 1e-8, py::arg("verbose") = true,
             "Run hybrid solve; returns (x, hist).");
}
