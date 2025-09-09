// src/l1_bindings.cpp
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/complex.h>
#include "../include/Definitions.hpp"
#include "../include/Helpers.hpp"
#include "../include/PolynomialVector.hpp"
#include "../include/Polynomials.hpp"
#include "../include/Solvers.hpp"
#include "../include/TRModel.hpp"
#include "../include/l1Solver.hpp" // declares l1_function, evaluatePDescent, etc.

namespace py = pybind11;

PYBIND11_MODULE(l1core, m) {
    m.doc() = "Bindings for L1/TR DFO pieces to run outer loop in Python";

    py::class_<Options>(m, "Options")
        .def(py::init<>())
        .def_readwrite("tol_radius", &Options::tol_radius)
        .def_readwrite("tol_f", &Options::tol_f)
        .def_readwrite("tol_measure", &Options::tol_measure)
        .def_readwrite("tol_con", &Options::tol_con)
        .def_readwrite("pivot_threshold", &Options::pivot_threshold)
        .def_readwrite("initial_radius", &Options::initial_radius)
        .def_readwrite("radius_max", &Options::radius_max)
        .def_readwrite("max_it", &Options::max_it)
        .def_readwrite("verbose", &Options::verbose)
        .def_readwrite("initial_radius", &Options::initial_radius)
        .def_readwrite("radius_max", &Options::radius_max)
        .def_readwrite("gamma_inc", &Options::gamma_inc)
        .def_readwrite("gamma_dec", &Options::gamma_dec)
        .def_readwrite("eta_1", &Options::eta_1)
        .def_readwrite("eta_2", &Options::eta_2)
        .def_readwrite("eps_c", &Options::eps_c)
        .def_readwrite("tol_measure", &Options::tol_measure)
        .def_readwrite("tol_radius", &Options::tol_radius)
        .def_readwrite("max_iter", &Options::max_iter)
        .def_readwrite("divergence_threshold", &Options::divergence_threshold)
        .def_readwrite("pivot_threshold", &Options::pivot_threshold)
        .def_readwrite("radius_factor", &Options::radius_factor)
        .def_readwrite("radius_factor_extra_tol", &Options::radius_factor_extra_tol);

    // -------- Funcao (objective + constraints) --------
    py::class_<Funcao>(m, "Funcao")
        .def(py::init<>())
        .def("addObjective", &Funcao::addObjective)
        .def("addConstraint", &Funcao::addConstraint)
        .def("evaluateObjective", &Funcao::evaluateObjective, py::arg("x"))
        .def("evaluateConstraint", &Funcao::evaluateConstraint, py::arg("x"), py::arg("index"))
        .def("calcAll", &Funcao::calcAll, py::arg("x"))
        .def("size", &Funcao::size)
        // Optional: expose fields (read/write) if you like
        .def_readwrite("obj", &Funcao::obj)
        .def_readwrite("con", &Funcao::con);
py::class_<CModelClass, std::shared_ptr<CModelClass>>(m, "CModelClass")
    .def(py::init<>())
    // Container-ish helpers
    .def("addConstraint", &CModelClass::addConstraint, py::arg("constraint"))
    .def("print", &CModelClass::print)
    .def("c", &CModelClass::c) // returns Eigen::VectorXd of all c's
    // Expose a python-friendly filter (mask) that returns a new CModelClass
    .def("filter_mask",
         [](const CModelClass& self, const std::vector<bool>& mask) {
             // Reuse your FilterProxy -> operator CModelClass()
             return self[mask]; // uses your FilterProxy conversion
         },
         py::arg("mask"))
    // Optional: access by index (read-only copy)
    .def("__len__", [](const CModelClass& self){ return self.size(); })
    .def("__getitem__", [](const CModelClass& self, size_t i) {
         if (i >= self.size()) throw py::index_error();
         return self[i]; // copies Constraint (ok)
    })
    // Optional: setC (mutates existing)
    .def("setC", &CModelClass::setC, py::arg("index"), py::arg("constraint"))
;
    // -------- TRModel --------
    py::class_<TRModel, std::shared_ptr<TRModel>>(m, "TRModel")
        .def(py::init<Eigen::MatrixXd &, Eigen::MatrixXd &, Options &>(), py::arg("initial_points"),
             py::arg("initial_f_values"), py::arg("options"))
        .def("rebuildModel", &TRModel::rebuildModel, py::arg("options"))
        .def("computePolynomialModels", &TRModel::computePolynomialModels)
        .def(
            "getModelMatrices",
            [](TRModel &self, int which) {
                double          fx;
                Eigen::VectorXd g;
                Eigen::MatrixXd H;
                std::tie(fx, g, H) = self.getModelMatrices(which);
                return py::make_tuple(fx, g, H);
            },
            py::arg("which"))
        .def("extractConstraintsFromTRModel", &TRModel::extractConstraintsFromTRModel, py::arg("con_bl"),
             py::arg("con_bu"))
        .def("isLambdaPoised", &TRModel::isLambdaPoised, py::arg("options"))
        .def(
            "trCriticalityStep",
            [](TRModel &self, Funcao &fphi, double mu, double epsilon, Eigen::VectorXd &bl,
               Eigen::VectorXd &bu, Eigen::VectorXd &con_bl, Eigen::VectorXd &con_bu,
               double eps_measure_thr, double eps_radius_thr, Options &options) {
                double eps_out, thr_out1, thr_out2;
                std::tie(eps_out, thr_out1, thr_out2) = self.trCriticalityStep(
                    fphi, mu, epsilon, bl, bu, con_bl, con_bu, eps_measure_thr, eps_radius_thr, options);
                return py::make_tuple(eps_out, thr_out1, thr_out2);
            },
            py::arg("fphi"), py::arg("mu"), py::arg("epsilon"), py::arg("bl"), py::arg("bu"), py::arg("con_bl"),
            py::arg("con_bu"), py::arg("eps_measure_thr"), py::arg("eps_radius_thr"), py::arg("options"))
        .def("log", &TRModel::log, py::arg("iter"))
        .def_readwrite("radius", &TRModel::radius)
        .def_readonly("pivotValues", &TRModel::pivotValues)
        .def_readonly("fValues", &TRModel::fValues)
        .def_readonly("trCenter", &TRModel::trCenter);

    // -------- Free helpers you call in the loop --------
    m.def("projectToBounds", &projectToBounds, py::arg("x"), py::arg("bl"), py::arg("bu"), "Project x to [bl, bu]");

    m.def(
        "l1_function",
        [](Funcao &func, Eigen::VectorXd &con_bl, Eigen::VectorXd &con_bu, double mu, Eigen::VectorXd x) {
            // NOTE: your original signature took x by non-const ref; we pass a copy and return both
            double          p;
            Eigen::VectorXd fvals;
            std::tie(p, fvals) = l1_function(func, con_bl, con_bu, mu, x);
            return py::make_tuple(p, fvals);
        },
        py::arg("func"), py::arg("con_bl"), py::arg("con_bu"), py::arg("mu"), py::arg("x"),
        "Compute L1 exact-penalty and stacked f/cons at x");

    m.def("evaluatePDescent", &evaluatePDescent, py::arg("fvals_center"), py::arg("fvals_trial"), py::arg("con_bl"),
          py::arg("con_bu"), py::arg("mu"));

    m.def("l1CriticalityMeasureAndDescentDirection", &l1CriticalityMeasureAndDescentDirection, py::arg("trmodel"),
          py::arg("cmodel"), py::arg("x"), py::arg("mu"), py::arg("epsilon"), py::arg("bl"), py::arg("bu"), py::arg("centerIn") = Eigen::VectorXd(),
          py::arg("giveRadius") = false,
          "Return (measure, d, is_eactive)");

    m.def("l1TrustRegionStep", &l1TrustRegionStep, py::arg("trmodel"), py::arg("cmodel"), py::arg("x"),
          py::arg("epsilon"), py::arg("lambda"), py::arg("mu"), py::arg("radius"), py::arg("bl"), py::arg("bu"),
          "Return (x_step, pred, lambda_out)");

    m.def("changeTRCenter", &changeTRCenter, py::arg("trmodel"), py::arg("x_trial"), py::arg("trial_fvalues"),
          py::arg("options"));

    m.def("ensureImprovement", &ensureImprovement, py::arg("trmodel"), py::arg("fphi"), py::arg("bl"), py::arg("bu"),
          py::arg("options"));

    m.def("try2addPoint", &try2addPoint, py::arg("trmodel"), py::arg("x_trial"), py::arg("trial_fvalues"),
          py::arg("fphi"), py::arg("bl"), py::arg("bu"), py::arg("options"));

    // Optional: expose Gurobi environment bootstrap if you call it once.
    m.def("initialize_gurobi_env", &GurobiSolver::initializeEnvironment);
}
