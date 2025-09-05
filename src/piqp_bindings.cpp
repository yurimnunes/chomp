#include "../include/piqp.h"
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace piqp;

// Convert Python (NumPy or SciPy sparse) to Eigen::SparseMatrix<double>
static SparseMatrix py_to_sparse(const py::object &obj) {
    if (obj.is_none())
        return SparseMatrix(0, 0);

    // If it looks like a SciPy sparse matrix, use .tocoo() to pull triplets
    if (py::hasattr(obj, "tocoo")) {
        py::object coo = obj.attr("tocoo")();
        py::array data = coo.attr("data").cast<py::array>();
        py::array row = coo.attr("row").cast<py::array>();
        py::array col = coo.attr("col").cast<py::array>();
        py::tuple shape = coo.attr("shape").cast<py::tuple>();
        int rows = shape[0].cast<int>();
        int cols = shape[1].cast<int>();

        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(static_cast<size_t>(data.shape(0)));

        auto d = data.cast<
            py::array_t<double, py::array::c_style | py::array::forcecast>>();
        auto r = row.cast<py::array_t<long long, py::array::c_style |
                                                     py::array::forcecast>>();
        auto c = col.cast<py::array_t<long long, py::array::c_style |
                                                     py::array::forcecast>>();

        const double *dptr = d.data();
        const long long *rptr = r.data();
        const long long *cptr = c.data();
        ssize_t nnz = d.shape(0);
        for (ssize_t k = 0; k < nnz; ++k) {
            trips.emplace_back(static_cast<int>(rptr[k]),
                               static_cast<int>(cptr[k]), dptr[k]);
        }
        SparseMatrix S(rows, cols);
        S.setFromTriplets(trips.begin(), trips.end());
        S.makeCompressed();
        return S;
    }

    // Else assume dense array-likes (NumPy) and convert to sparse
    Eigen::MatrixXd dense = obj.cast<Eigen::MatrixXd>();
    SparseMatrix S = dense.sparseView();
    S.makeCompressed();
    return S;
}

PYBIND11_MODULE(piqp_cpp, m) {
    m.doc() = "PIQP (sparse-only): Proximal Interior-Point QP Solver";

    py::class_<PIQPSettings>(m, "PIQPSettings")
        .def(py::init<>())
        .def_readwrite("eps_abs", &PIQPSettings::eps_abs)
        .def_readwrite("eps_rel", &PIQPSettings::eps_rel)
        .def_readwrite("max_iter", &PIQPSettings::max_iter)
        .def_readwrite("rho_init", &PIQPSettings::rho_init)
        .def_readwrite("delta_init", &PIQPSettings::delta_init)
        .def_readwrite("rho_floor", &PIQPSettings::rho_floor)
        .def_readwrite("delta_floor", &PIQPSettings::delta_floor)
        .def_readwrite("tau", &PIQPSettings::tau)
        .def_readwrite("reg_eps", &PIQPSettings::reg_eps)
        .def_readwrite("verbose", &PIQPSettings::verbose)
        .def_readwrite("min_slack", &PIQPSettings::min_slack)
        .def_readwrite("scale", &PIQPSettings::scale)
        .def_readwrite("ruiz_iters", &PIQPSettings::ruiz_iters)
        .def_readwrite("scale_eps", &PIQPSettings::scale_eps)
        .def_readwrite("cost_scaling", &PIQPSettings::cost_scaling)
        .def("__repr__", [](const PIQPSettings &s) {
            return "<PIQPSettings(eps_abs=" + std::to_string(s.eps_abs) +
                   ", eps_rel=" + std::to_string(s.eps_rel) +
                   ", max_iter=" + std::to_string(s.max_iter) + ")>";
        });

    // Mirror of the struct in header
    using Residuals = PIQPResiduals;
    py::class_<Residuals>(m, "PIQPResiduals")
        .def_readonly("eq_inf", &Residuals::eq_inf)
        .def_readonly("ineq_inf", &Residuals::ineq_inf)
        .def_readonly("stat_inf", &Residuals::stat_inf)
        .def_readonly("gap", &Residuals::gap);

    py::class_<PIQPResult>(m, "PIQPResult")
        .def_readonly("status", &PIQPResult::status)
        .def_readonly("iterations", &PIQPResult::iterations)
        .def_readonly("x", &PIQPResult::x)
        .def_readonly("s", &PIQPResult::s) // if s is also non-optional
        .def_readonly("y", &PIQPResult::y) // λ (equalities)
        .def_readonly("z", &PIQPResult::z) // μ (inequalities)
        .def_readonly("obj_val", &PIQPResult::obj_val)
        .def_readonly("residuals", &PIQPResult::residuals)
        .def("__repr__", [](const PIQPResult &r) {
            return "<PIQPResult(status='" + r.status +
                   "', iterations=" + std::to_string(r.iterations) +
                   ", obj_val=" + std::to_string(r.obj_val) + ")>";
        });

    py::class_<PIQPSolver>(m, "PIQPSolver")
        .def(py::init<>())
        .def(py::init<const PIQPSettings &>())
        .def(
            "setup",
            [](PIQPSolver &solver, const py::object &P,
               const Eigen::Ref<const Vector> &q, const py::object &A,
               const py::object &b, const py::object &G,
               const py::object &h) -> PIQPSolver & {
                SparseMatrix Ps = py_to_sparse(P);
                std::optional<SparseMatrix> As, Gs;
                std::optional<Vector> bs, hs;
                if (!A.is_none())
                    As = py_to_sparse(A);
                if (!b.is_none())
                    bs = b.cast<Eigen::Ref<const Vector>>();
                if (!G.is_none())
                    Gs = py_to_sparse(G);
                if (!h.is_none())
                    hs = h.cast<Eigen::Ref<const Vector>>();
                return solver.setup(Ps, q, As, bs, Gs, hs);
            },
            py::arg("P"), py::arg("q"), py::arg("A") = py::none(),
            py::arg("b") = py::none(), py::arg("G") = py::none(),
            py::arg("h") = py::none(),
            py::return_value_policy::reference_internal)
        .def("solve", &PIQPSolver::solve)
        .def("get_settings", &PIQPSolver::getSettings,
             py::return_value_policy::reference_internal)
        .def("get_x", &PIQPSolver::getX,
             py::return_value_policy::reference_internal)
        .def("get_s", &PIQPSolver::getS,
             py::return_value_policy::reference_internal)
        .def("get_y", &PIQPSolver::getY,
             py::return_value_policy::reference_internal)
        .def("get_z", &PIQPSolver::getZ,
             py::return_value_policy::reference_internal)
        .def("get_iterations", &PIQPSolver::getIterations)
        .def("get_status", &PIQPSolver::getStatus,
             py::return_value_policy::reference_internal)
        // Warm start: x,y,z,s (+ optional prox centers)
        .def(
            "warm_start",
            [](PIQPSolver &solver, py::object x, py::object y, py::object z,
               py::object s, bool copy_to_prox_centers) -> PIQPSolver & {
                std::optional<Vector> xo, yo, zo, so;
                if (!x.is_none())
                    xo = x.cast<Eigen::VectorXd>();
                if (!y.is_none())
                    yo = y.cast<Eigen::VectorXd>();
                if (!z.is_none())
                    zo = z.cast<Eigen::VectorXd>();
                if (!s.is_none())
                    so = s.cast<Eigen::VectorXd>();
                return solver.warm_start(xo, yo, zo, so, copy_to_prox_centers);
            },
            py::arg("x") = py::none(), py::arg("y") = py::none(),
            py::arg("z") = py::none(), py::arg("s") = py::none(),
            py::arg("copy_to_prox_centers") = true,
            py::return_value_policy::reference_internal)

        // Prox centers only
        .def(
            "set_prox_centers",
            [](PIQPSolver &solver, py::object xi, py::object lambda,
               py::object nu) -> PIQPSolver & {
                std::optional<Vector> xio, lo, nuo;
                if (!xi.is_none())
                    xio = xi.cast<Eigen::VectorXd>();
                if (!lambda.is_none())
                    lo = lambda.cast<Eigen::VectorXd>();
                if (!nu.is_none())
                    nuo = nu.cast<Eigen::VectorXd>();
                return solver.set_prox_centers(xio, lo, nuo);
            },
            py::arg("xi") = py::none(), py::arg("lambda") = py::none(),
            py::arg("nu") = py::none(),
            py::return_value_policy::reference_internal)

        // Reuse last (optionally as prox centers)
        .def("use_last_as_warm_start", &PIQPSolver::use_last_as_warm_start,
             py::arg("also_prox") = true,
             py::return_value_policy::reference_internal)

        // Reset rho/delta
        .def("set_prox_params", &PIQPSolver::set_prox_params, py::arg("rho"),
             py::arg("delta"), py::return_value_policy::reference_internal)

        // Fast numeric update (pattern-stable by default)
        .def(
            "update_values",
            [](PIQPSolver &solver, py::object P,
               const Eigen::Ref<const Vector> &q, py::object A, py::object b,
               py::object G, py::object h, bool same_pattern) -> PIQPSolver & {
                SparseMatrix Ps = py_to_sparse(P);
                std::optional<SparseMatrix> As, Gs;
                std::optional<Vector> bs, hs;
                if (!A.is_none())
                    As = py_to_sparse(A);
                if (!b.is_none())
                    bs = b.cast<Eigen::Ref<const Vector>>();
                if (!G.is_none())
                    Gs = py_to_sparse(G);
                if (!h.is_none())
                    hs = h.cast<Eigen::Ref<const Vector>>();
                return solver.update_values(Ps, q, As, bs, Gs, hs,
                                            same_pattern);
            },
            py::arg("P"), py::arg("q"), py::arg("A") = py::none(),
            py::arg("b") = py::none(), py::arg("G") = py::none(),
            py::arg("h") = py::none(), py::arg("same_pattern") = true,
            py::return_value_policy::reference_internal);
    // One-shot convenience
    m.def(
        "solve",
        [](const py::object &P, const Eigen::Ref<const Vector> &q,
           const py::object &A, const py::object &b, const py::object &G,
           const py::object &h, const PIQPSettings &settings) -> PIQPResult {
            PIQPSolver solver(settings);
            SparseMatrix Ps = py_to_sparse(P);
            std::optional<SparseMatrix> As, Gs;
            std::optional<Vector> bs, hs;
            if (!A.is_none())
                As = py_to_sparse(A);
            if (!b.is_none())
                bs = b.cast<Eigen::Ref<const Vector>>();
            if (!G.is_none())
                Gs = py_to_sparse(G);
            if (!h.is_none())
                hs = h.cast<Eigen::Ref<const Vector>>();
            solver.setup(Ps, q, As, bs, Gs, hs);
            return solver.solve();
        },
        py::arg("P"), py::arg("q"), py::arg("A") = py::none(),
        py::arg("b") = py::none(), py::arg("G") = py::none(),
        py::arg("h") = py::none(), py::arg("settings") = PIQPSettings{});
}
