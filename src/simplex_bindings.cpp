// simplex_bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "../include/simplex.h"

namespace py = pybind11;

static std::string status_to_str(const LPSolution::Status s) {
    return to_string(s);
}

// Convert LPSolution to a Python-facing object with .status as string
struct PyLPSolution {
    std::string status;                  // 'optimal', 'unbounded', ...
    Eigen::VectorXd x;
    double obj;
    std::vector<int> basis;
    int iters;
    std::unordered_map<std::string, std::string> info;

    static PyLPSolution from_cpp(const LPSolution& s) {
        return PyLPSolution{
            status_to_str(s.status),
            s.x, s.obj, s.basis, s.iters, s.info
        };
    }
};

static RevisedSimplexOptions make_opts_from_kwargs(const py::kwargs& kw) {
    RevisedSimplexOptions o;
    auto get = [&](const char* k){ return kw.contains(k) ? kw[k] : py::none(); };

    if (!get("max_iters").is_none())                o.max_iters = py::cast<int>(kw["max_iters"]);
    if (!get("tol").is_none())                      o.tol = py::cast<double>(kw["tol"]);
    if (!get("bland").is_none())                    o.bland = py::cast<bool>(kw["bland"]);
    if (!get("svd_tol").is_none())                  o.svd_tol = py::cast<double>(kw["svd_tol"]);
    if (!get("ratio_delta").is_none())              o.ratio_delta = py::cast<double>(kw["ratio_delta"]);
    if (!get("ratio_eta").is_none())                o.ratio_eta = py::cast<double>(kw["ratio_eta"]);
    if (!get("deg_step_tol").is_none())             o.deg_step_tol = py::cast<double>(kw["deg_step_tol"]);
    if (!get("epsilon_cost").is_none())             o.epsilon_cost = py::cast<double>(kw["epsilon_cost"]);
    if (!get("rng_seed").is_none())                 o.rng_seed = py::cast<int>(kw["rng_seed"]);
    if (!get("refactor_every").is_none())           o.refactor_every = py::cast<int>(kw["refactor_every"]);
    if (!get("pricing_rule").is_none())             o.pricing_rule = py::cast<std::string>(kw["pricing_rule"]);
    if (!get("max_basis_rebuilds").is_none())       o.max_basis_rebuilds = py::cast<int>(kw["max_basis_rebuilds"]);
    return o;
}

PYBIND11_MODULE(simplex_core, m) {
    m.doc() = "Revised Simplex (modern C++) Python bindings";

    // Expose options (nice for discoverability)
    py::class_<RevisedSimplexOptions>(m, "RevisedSimplexOptions")
        .def(py::init<>())
        .def_readwrite("max_iters", &RevisedSimplexOptions::max_iters)
        .def_readwrite("tol", &RevisedSimplexOptions::tol)
        .def_readwrite("bland", &RevisedSimplexOptions::bland)
        .def_readwrite("svd_tol", &RevisedSimplexOptions::svd_tol)
        .def_readwrite("ratio_delta", &RevisedSimplexOptions::ratio_delta)
        .def_readwrite("ratio_eta", &RevisedSimplexOptions::ratio_eta)
        .def_readwrite("deg_step_tol", &RevisedSimplexOptions::deg_step_tol)
        .def_readwrite("epsilon_cost", &RevisedSimplexOptions::epsilon_cost)
        .def_readwrite("rng_seed", &RevisedSimplexOptions::rng_seed)
        .def_readwrite("refactor_every", &RevisedSimplexOptions::refactor_every)
        .def_readwrite("pricing_rule", &RevisedSimplexOptions::pricing_rule)
        .def_readwrite("max_basis_rebuilds", &RevisedSimplexOptions::max_basis_rebuilds)
        ;

    // Python-facing LPSolution (status is string)
    py::class_<PyLPSolution>(m, "LPSolution")
        .def_property_readonly("status", [](const PyLPSolution& s){ return s.status; })
        .def_readonly("x", &PyLPSolution::x)
        .def_readonly("obj", &PyLPSolution::obj)
        .def_readonly("basis", &PyLPSolution::basis)
        .def_readonly("iters", &PyLPSolution::iters)
        .def_readonly("info", &PyLPSolution::info)
        ;

    // RevisedSimplex class
    py::class_<RevisedSimplex>(m, "RevisedSimplex")
        // __init__(**kwargs) optional; defaults match C++
        .def(py::init([&](const py::kwargs& kw){
            if (kw.size() == 0) return RevisedSimplex{};
            return RevisedSimplex{ make_opts_from_kwargs(kw) };
        }),
        R"doc(
            RevisedSimplex(**kwargs)
            Keyword options include:
              - max_iters, tol, bland, svd_tol, ratio_delta, ratio_eta,
                deg_step_tol, epsilon_cost, rng_seed, refactor_every,
                lu_pivot_rel, lu_abs_floor, devex_reset, pricing_rule,
                steepest_edge_reset_freq, max_basis_rebuilds
        )doc")
        // solve(A, b, c, basis=None) -> LPSolution
        .def("solve",
            [](RevisedSimplex& self,
               const Eigen::MatrixXd& A,
               const Eigen::VectorXd& b,
               const Eigen::VectorXd& c,
               py::object basis_opt) -> PyLPSolution
            {
                std::optional<std::vector<int>> basis = std::nullopt;
                if (!basis_opt.is_none()) {
                    basis = py::cast<std::vector<int>>(basis_opt);
                }
                LPSolution raw = self.solve(A, b, c, basis);
                return PyLPSolution::from_cpp(raw);
            },
            py::arg("A"), py::arg("b"), py::arg("c"), py::arg("basis") = py::none(),
            R"doc(
                Solve standard-form LP:
                    min c^T x
                    s.t. A x = b, x >= 0

                Returns LPSolution with fields:
                    status (str), x (np.ndarray), obj (float),
                    basis (List[int]), iters (int), info (dict)
            )doc")
        ;
}
