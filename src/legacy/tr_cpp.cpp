// tr_sota_optimized.cpp
// Trust-region (Steihaug–Toint) with Eigen + pybind11 + SOC correction.
// Optimized to reduce heap churn and tiny temporaries while keeping ALL
// features.
// - Dense or sparse H (via LinOp)
// - Equality + inequality constraints (active set)
// - Preconditioners: identity / Jacobi / SSOR
// - Ellipsoidal norm via cached Cholesky
// - SOC ported from your Python logic (model.eval_all, constraint_violation)
// - Criticality step, curvature-aware TR growth, filter-based acceptance
// - Box handling with mode switch: "projection" | "alpha"
// (fraction-to-boundary)
//
// Drop-in replacement for your previous file: same public API/signatures.
//
// Build with your existing CMake setup for pybind11.
//
// Notes on improvements:
//  - LinOp no longer allocates per-apply; callers pass output buffers
//  - A single TRWorkspace holds all hot-loop vectors to avoid repeated allocs
//  - PCG rewritten to reuse workspace buffers and avoid "+=" surprises
//  - Projected preconditioner reuses temporary buffers and projections
//  - All helper paths avoid ephemeral Eigen temporaries where possible
//  - Safer numerics: fewer divisions by tiny numbers, consistent clamps

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../include/tr.h"

// -------------------- PYBIND11 --------------------
PYBIND11_MODULE(tr_cpp, m) {
    m.doc() =
        "Trust-region (Steihaug–Toint) with Eigen + pybind11 + SOC correction";

    py::class_<TRConfig>(m, "TRConfig")
        .def(py::init<>())
        .def_readwrite("delta0", &TRConfig::delta0)
        .def_readwrite("delta_min", &TRConfig::delta_min)
        .def_readwrite("delta_max", &TRConfig::delta_max)
        .def_readwrite("cg_tol", &TRConfig::cg_tol)
        .def_readwrite("cg_tol_rel", &TRConfig::cg_tol_rel)
        .def_readwrite("cg_maxiter", &TRConfig::cg_maxiter)
        .def_readwrite("neg_curv_tol", &TRConfig::neg_curv_tol)
        .def_readwrite("rcond", &TRConfig::rcond)
        .def_readwrite("metric_shift", &TRConfig::metric_shift)
        .def_readwrite("zeta", &TRConfig::zeta)
        .def_readwrite("constraint_tol", &TRConfig::constraint_tol)
        .def_readwrite("max_active_set_iter", &TRConfig::max_active_set_iter)
        .def_readwrite("use_prec", &TRConfig::use_prec)
        .def_readwrite("prec_kind", &TRConfig::prec_kind)
        .def_readwrite("norm_type", &TRConfig::norm_type)
        .def_readwrite("ssor_omega", &TRConfig::ssor_omega);

    py::class_<TrustRegionManager>(m, "TrustRegionManager")
        .def(py::init<const TRConfig &>(), py::arg("config"))
        .def("set_metric_from_H_dense",
             &TrustRegionManager::set_metric_from_H_dense)
        .def_readwrite("delta", &TrustRegionManager::delta_)

        .def("set_metric_from_H_sparse",
             &TrustRegionManager::set_metric_from_H_sparse)
        .def("solve_dense", &TrustRegionManager::solve_dense, py::arg("H"),
             py::arg("g"), py::arg("Aineq") = py::none(),
             py::arg("bineq") = py::none(), py::arg("Aeq") = py::none(),
             py::arg("beq") = py::none(), py::arg("model") = py::none(),
             py::arg("x") = py::none(), py::arg("lb") = py::none(),
             py::arg("ub") = py::none(), py::arg("mu") = 0.0,
             py::arg("f_old") = py::none())
        .def("solve_sparse", &TrustRegionManager::solve_sparse, py::arg("H"),
             py::arg("g"), py::arg("Aineq") = py::none(),
             py::arg("bineq") = py::none(), py::arg("Aeq") = py::none(),
             py::arg("beq") = py::none(), py::arg("model") = py::none(),
             py::arg("x") = py::none(), py::arg("lb") = py::none(),
             py::arg("ub") = py::none(), py::arg("mu") = 0.0,
             py::arg("f_old") = py::none())

        // convenience overload with underscored names
        .def(
            "solve_dense",
            [](TrustRegionManager &self, const dmat &H, const dvec &g,
               py::object A_ineq, py::object b_ineq, py::object A_eq,
               py::object b_eq, py::object model, py::object x, py::object lb,
               py::object ub, double mu, py::object f_old) {
                auto optMat = [](py::object o) -> std::optional<dmat> {
                    return o.is_none() ? std::nullopt
                                       : std::optional<dmat>(o.cast<dmat>());
                };
                auto optVec = [](py::object o) -> std::optional<dvec> {
                    return o.is_none() ? std::nullopt
                                       : std::optional<dvec>(o.cast<dvec>());
                };
                auto optObj = [](py::object o) -> std::optional<py::object> {
                    return o.is_none() ? std::nullopt
                                       : std::optional<py::object>(o);
                };
                auto optF = [](py::object o) -> std::optional<double> {
                    return o.is_none()
                               ? std::nullopt
                               : std::optional<double>(o.cast<double>());
                };

                return self.solve_dense(H, g, optMat(A_ineq), optVec(b_ineq),
                                        optMat(A_eq), optVec(b_eq),
                                        optObj(model), optVec(x), optVec(lb),
                                        optVec(ub), mu, optF(f_old));
            },
            py::arg("H"), py::arg("g"), py::arg("A_ineq") = py::none(),
            py::arg("b_ineq") = py::none(), py::arg("A_eq") = py::none(),
            py::arg("b_eq") = py::none(), py::arg("model") = py::none(),
            py::arg("x") = py::none(), py::arg("lb") = py::none(),
            py::arg("ub") = py::none(), py::arg("mu") = 0.0,
            py::arg("f_old") = py::none());
}
