// src/osqp_bindings.cpp
#include "../include/osqp.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <optional>

namespace py = pybind11;
using namespace sosqp;

// ------------------------------
// Python → Eigen helpers
// ------------------------------
static SpMat py_to_sparse(const py::object &obj, int rows_hint = -1, int cols_hint = -1) {
    if (obj.is_none()) {
        int r = std::max(0, rows_hint);
        int c = std::max(0, cols_hint);
        SpMat S(r, c);
        S.makeCompressed();
        return S;
    }

    // SciPy sparse? use .tocoo()
    if (py::hasattr(obj, "tocoo")) {
        py::object coo = obj.attr("tocoo")();
        py::array data = coo.attr("data").cast<py::array>();
        py::array row  = coo.attr("row").cast<py::array>();
        py::array col  = coo.attr("col").cast<py::array>();
        py::tuple shape = coo.attr("shape").cast<py::tuple>();
        int rows = shape[0].cast<int>();
        int cols = shape[1].cast<int>();

        auto d = data.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        auto r = row .cast<py::array_t<long long, py::array::c_style | py::array::forcecast>>();
        auto c = col .cast<py::array_t<long long, py::array::c_style | py::array::forcecast>>();

        const double* dptr = d.data();
        const long long* rptr = r.data();
        const long long* cptr = c.data();
        ssize_t nnz = d.shape(0);

        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(static_cast<size_t>(nnz));
        for (ssize_t k=0; k<nnz; ++k) {
            trips.emplace_back(static_cast<int>(rptr[k]),
                               static_cast<int>(cptr[k]),
                               dptr[k]);
        }
        SpMat S(rows, cols);
        S.setFromTriplets(trips.begin(), trips.end());
        S.makeCompressed();
        return S;
    }

    // Otherwise assume dense array-like
    Eigen::MatrixXd dense = obj.cast<Eigen::MatrixXd>();
    SpMat S = dense.sparseView();
    S.makeCompressed();
    return S;
}

static Vec py_to_vec_opt(const py::object &obj, int size_hint = -1) {
    if (obj.is_none()) {
        return (size_hint > 0) ? Vec::Zero(size_hint) : Vec(); // empty if unknown
    }
    return obj.cast<Eigen::VectorXd>();
}

// ------------------------------
// Small stateful wrapper (convenience)
// ------------------------------
class OSQPProblem {
public:
    explicit OSQPProblem(const Settings& s = Settings{}) : settings_(s) {
        // back-compat: ensure rho0 mirrors rho if user set it in Python
        // (pybind will have copied values already)
        settings_.rho0 = settings_.rho;
    }

    // Set both P and A (initial)
    OSQPProblem& set_matrices(const SpMat& P, const SpMat& A) {
        P_ = P;
        A_ = A;
        return *this;
    }

    // Update either matrix; if arg is None on Python side, it stays unchanged.
    OSQPProblem& update_P_A(std::optional<SpMat> P_new,
                            std::optional<SpMat> A_new) {
        if (P_new) P_ = std::move(*P_new);
        if (A_new) A_ = std::move(*A_new);
        return *this;
    }

    // One solve using stored P/A and provided q,l,u (+ optional warm start)
    Result solve(const Vec& q,
                 const Vec& l, const Vec& u,
                 const Vec* x0=nullptr, const Vec* z0=nullptr, const Vec* y0=nullptr) const {
        // If A is not set, build an empty 0×n with the correct n
        SpMat A_use = A_;
        if (A_use.rows()==0 && A_use.cols()==0) {
            A_use.resize(0, static_cast<int>(P_.cols()));
            A_use.makeCompressed();
        }
        SparseOSQPSolver solver(settings_);
        return solver.solve(P_, q, A_use, l, u, x0, z0, y0);
    }

    const SpMat& P() const { return P_; }
    const SpMat& A() const { return A_; }
    const Settings& settings() const { return settings_; }

private:
    Settings settings_;
    SpMat P_;
    SpMat A_;
};

// ------------------------------
// Module
// ------------------------------
PYBIND11_MODULE(osqp_cpp, m) {
    m.doc() = "Sparse-only OSQP-like ADMM solver (Eigen + pybind11)";

    // --- Settings ---
    py::class_<Settings>(m, "Settings")
        .def(py::init<>())
        // new canonical field
        .def_readwrite("rho0", &Settings::rho0)
        // deprecated mirror (kept for compatibility)
        .def_readwrite("rho", &Settings::rho)
        .def_readwrite("rho_eq_scale", &Settings::rho_eq_scale)
        .def_readwrite("sigma", &Settings::sigma)
        .def_readwrite("alpha", &Settings::alpha)
        .def_readwrite("adaptive_rho", &Settings::adaptive_rho)
        .def_readwrite("eps_abs", &Settings::eps_abs)
        .def_readwrite("eps_rel", &Settings::eps_rel)
        .def_readwrite("eps_pinf", &Settings::eps_pinf)
        .def_readwrite("eps_dinf", &Settings::eps_dinf)
        .def_readwrite("max_iter", &Settings::max_iter)
        .def_readwrite("check_every", &Settings::check_every)
        .def_readwrite("diag_reg", &Settings::diag_reg)
        .def_readwrite("eq_tol", &Settings::eq_tol)
        .def_readwrite("verbose", &Settings::verbose)
        .def_readwrite("polish", &Settings::polish)
        .def_readwrite("polish_delta", &Settings::polish_delta)
        .def_readwrite("polish_refine_steps", &Settings::polish_refine_steps)
        // legacy knobs retained
        .def_readwrite("rho_min", &Settings::rho_min)
        .def_readwrite("rho_max", &Settings::rho_max)
        .def_readwrite("explode_refactor", &Settings::explode_refactor)
        .def_readwrite("max_refactor", &Settings::max_refactor)
        .def("__repr__", [](const Settings &s) {
            return "<Settings(rho0=" + std::to_string(s.rho0) +
                   ", sigma=" + std::to_string(s.sigma) +
                   ", alpha=" + std::to_string(s.alpha) +
                   ", eps_abs=" + std::to_string(s.eps_abs) +
                   ", eps_rel=" + std::to_string(s.eps_rel) + ")>";
        });

    // --- Residuals ---
    py::class_<Residuals>(m, "Residuals")
        .def_readonly("pri_inf", &Residuals::pri_inf)
        .def_readonly("dua_inf", &Residuals::dua_inf)
        .def("__repr__", [](const Residuals &r) {
            return "<Residuals(pri_inf=" + std::to_string(r.pri_inf) +
                   ", dua_inf=" + std::to_string(r.dua_inf) + ")>";
        });

    // --- Result ---
    py::class_<Result>(m, "Result")
        .def_readonly("status", &Result::status)
        .def_readonly("iters", &Result::iters)
        .def_readonly("obj_val", &Result::obj_val)
        .def_readonly("x", &Result::x)
        .def_readonly("z", &Result::z)
        .def_readonly("y", &Result::y)
        .def_readonly("residuals", &Result::res)
        .def_readonly("primal_infeasible", &Result::primal_infeasible)
        .def_readonly("dual_infeasible", &Result::dual_infeasible)
        .def_readonly("y_cert", &Result::y_cert)
        .def_readonly("x_cert", &Result::x_cert)
        .def_readonly("x_polish", &Result::x_polish)
        .def("__repr__", [](const Result &r) {
            return "<Result(status='" + r.status + "', iters=" +
                   std::to_string(r.iters) + ", obj_val=" +
                   std::to_string(r.obj_val) + ")>";
        });

    // --- Stateless solver class ---
    py::class_<SparseOSQPSolver>(m, "SparseOSQPSolver")
        .def(py::init<const Settings &>(), py::arg("settings") = Settings{})
        .def("solve",
            [](SparseOSQPSolver &solver,
               const py::object &P,
               const Eigen::Ref<const Vec> &q,
               const py::object &A,
               const py::object &l,
               const py::object &u,
               const py::object &x0,
               const py::object &z0,
               const py::object &y0) -> Result {

                SpMat Ps = py_to_sparse(P);
                const int n = static_cast<int>(Ps.rows());

                SpMat As;
                int m = 0;
                if (!A.is_none()) {
                    As = py_to_sparse(A, -1, n);
                    m = static_cast<int>(As.rows());
                } else {
                    As.resize(0, n);
                    As.makeCompressed();
                }

                Vec lv = py_to_vec_opt(l, m);
                Vec uv = py_to_vec_opt(u, m);

                // Warm starts (optional)
                Vec x0v = x0.is_none() ? Vec() : x0.cast<Vec>();
                Vec z0v = z0.is_none() ? Vec() : z0.cast<Vec>();
                Vec y0v = y0.is_none() ? Vec() : y0.cast<Vec>();

                return solver.solve(Ps, q, As, lv, uv,
                                    x0.is_none()? nullptr : &x0v,
                                    z0.is_none()? nullptr : &z0v,
                                    y0.is_none()? nullptr : &y0v);
            },
            py::arg("P"),
            py::arg("q"),
            py::arg("A") = py::none(),
            py::arg("l") = py::none(),
            py::arg("u") = py::none(),
            py::arg("x0") = py::none(),
            py::arg("z0") = py::none(),
            py::arg("y0") = py::none()
        )
        // small ergonomic alias for explicit warm-start call
        .def("warm_start_solve",
            [](SparseOSQPSolver &solver,
               const py::object &P,
               const Eigen::Ref<const Vec> &q,
               const py::object &A,
               const py::object &l,
               const py::object &u,
               const Eigen::Ref<const Vec> &x0,
               const py::object &z0,
               const py::object &y0) -> Result {
                return py::cast(solver).attr("solve")(P, q, A, l, u, x0, z0, y0).cast<Result>();
            },
            py::arg("P"),
            py::arg("q"),
            py::arg("A") = py::none(),
            py::arg("l") = py::none(),
            py::arg("u") = py::none(),
            py::arg("x0"),
            py::arg("z0") = py::none(),
            py::arg("y0") = py::none()
        );

    // --- One-shot convenience function (stateless) ---
    m.def("solve",
        [](const py::object &P,
           const Eigen::Ref<const Vec> &q,
           const py::object &A,
           const py::object &l,
           const py::object &u,
           const Settings &settings,
           const py::object &x0,
           const py::object &z0,
           const py::object &y0) -> Result {

            SparseOSQPSolver solver(settings);

            SpMat Ps = py_to_sparse(P);
            const int n = static_cast<int>(Ps.rows());

            SpMat As;
            int m = 0;
            if (!A.is_none()) {
                As = py_to_sparse(A, -1, n);
                m = static_cast<int>(As.rows());
            } else {
                As.resize(0, n);
                As.makeCompressed();
            }

            Vec lv = py_to_vec_opt(l, m);
            Vec uv = py_to_vec_opt(u, m);

            Vec x0v = x0.is_none() ? Vec() : x0.cast<Vec>();
            Vec z0v = z0.is_none() ? Vec() : z0.cast<Vec>();
            Vec y0v = y0.is_none() ? Vec() : y0.cast<Vec>();

            return solver.solve(Ps, q, As, lv, uv,
                                x0.is_none()? nullptr : &x0v,
                                z0.is_none()? nullptr : &z0v,
                                y0.is_none()? nullptr : &y0v);
        },
        py::arg("P"),
        py::arg("q"),
        py::arg("A") = py::none(),
        py::arg("l") = py::none(),
        py::arg("u") = py::none(),
        py::arg("settings") = Settings{},
        py::arg("x0") = py::none(),
        py::arg("z0") = py::none(),
        py::arg("y0") = py::none()
    );

    // --- Stateful convenience holder (stores P/A/settings; still refactorizes internally) ---
    py::class_<OSQPProblem>(m, "OSQPProblem")
        .def(py::init<const Settings &>(), py::arg("settings") = Settings{})
        .def("set_matrices",
             [](OSQPProblem &prob, const py::object &P, const py::object &A) {
                 SpMat Ps = py_to_sparse(P);
                 SpMat As = py_to_sparse(A, -1, static_cast<int>(Ps.cols()));
                 prob.set_matrices(Ps, As);
                 return &prob;
             }, py::return_value_policy::reference_internal)
        .def("update_P_A",
             [](OSQPProblem &prob, const py::object &P, const py::object &A) {
                 std::optional<SpMat> Pn, An;
                 if (!P.is_none()) Pn = py_to_sparse(P);
                 if (!A.is_none()) {
                     // if P not provided, use existing n for A hint
                     int n_hint = Pn ? static_cast<int>(Pn->cols())
                                     : static_cast<int>(prob.P().cols());
                     An = py_to_sparse(A, -1, n_hint);
                 }
                 prob.update_P_A(Pn, An);
                 return &prob;
             }, py::arg("P") = py::none(), py::arg("A") = py::none(),
             py::return_value_policy::reference_internal)
        .def("solve",
             [](const OSQPProblem &prob,
                const Eigen::Ref<const Vec> &q,
                const py::object &l, const py::object &u,
                const py::object &x0, const py::object &z0, const py::object &y0) -> Result {

                 int m = prob.A().rows();
                 Vec lv = py_to_vec_opt(l, m);
                 Vec uv = py_to_vec_opt(u, m);

                 Vec x0v = x0.is_none() ? Vec() : x0.cast<Vec>();
                 Vec z0v = z0.is_none() ? Vec() : z0.cast<Vec>();
                 Vec y0v = y0.is_none() ? Vec() : y0.cast<Vec>();

                 return prob.solve(q, lv, uv,
                                   x0.is_none()? nullptr : &x0v,
                                   z0.is_none()? nullptr : &z0v,
                                   y0.is_none()? nullptr : &y0v);
             },
             py::arg("q"), py::arg("l") = py::none(), py::arg("u") = py::none(),
             py::arg("x0") = py::none(), py::arg("z0") = py::none(), py::arg("y0") = py::none()
        )
        .def_property_readonly("P", &OSQPProblem::P, py::return_value_policy::reference_internal)
        .def_property_readonly("A", &OSQPProblem::A, py::return_value_policy::reference_internal)
        .def_property_readonly("settings", &OSQPProblem::settings, py::return_value_policy::reference_internal);
}
