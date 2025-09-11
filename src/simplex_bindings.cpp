// python_bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include "simplex.h"

// Include the simplex solver headers
// #include "simplex_solver.h" // This would be the header for our C++ implementation

// For this example, I'll include the key parts inline
namespace py = pybind11;
using namespace simplex;

// Helper functions for numpy array conversion
template<typename T>
std::vector<T> numpy_to_vector(py::array_t<T> input) {
    py::buffer_info buf_info = input.request();
    T* ptr = static_cast<T*>(buf_info.ptr);
    return std::vector<T>(ptr, ptr + buf_info.size);
}

template<typename T>
py::array_t<T> vector_to_numpy(const std::vector<T>& vec) {
    return py::array_t<T>(
        vec.size(),
        vec.data(),
        py::cast(vec) // Keep the vector alive
    );
}

// Python wrapper for SparseMatrix
class PySparseMatrix {
private:
    std::unique_ptr<SparseMatrix> m_matrix;

public:
    PySparseMatrix(py::array_t<Real> data, py::array_t<Index> indices, 
                   py::array_t<LargeIndex> indptr, Index rows, Index cols) {
        auto data_vec = numpy_to_vector(data);
        auto indices_vec = numpy_to_vector(indices);
        auto indptr_vec = numpy_to_vector(indptr);
        
        m_matrix = std::make_unique<SparseMatrix>(
            rows, cols, std::move(indptr_vec), 
            std::move(indices_vec), std::move(data_vec)
        );
    }

    // Create from scipy.sparse matrix
    static PySparseMatrix from_scipy_csc(py::object scipy_matrix) {
        // Extract data from scipy.sparse.csc_matrix
        auto data = scipy_matrix.attr("data").cast<py::array_t<Real>>();
        auto indices = scipy_matrix.attr("indices").cast<py::array_t<Index>>();
        auto indptr = scipy_matrix.attr("indptr").cast<py::array_t<LargeIndex>>();
        auto shape = scipy_matrix.attr("shape").cast<py::tuple>();
        
        Index rows = shape[0].cast<Index>();
        Index cols = shape[1].cast<Index>();
        
        return PySparseMatrix(data, indices, indptr, rows, cols);
    }

    // Matrix-vector multiplication
    py::array_t<Real> multiply(py::array_t<Real> x) {
        auto x_vec = numpy_to_vector(x);
        std::vector<Real> y(m_matrix->rows(), 0.0);
        m_matrix->multiply(x_vec, y);
        return vector_to_numpy(y);
    }

    py::array_t<Real> multiply_transpose(py::array_t<Real> x) {
        auto x_vec = numpy_to_vector(x);
        std::vector<Real> y(m_matrix->cols(), 0.0);
        m_matrix->multiplyTranspose(x_vec, y);
        return vector_to_numpy(y);
    }

    // Properties
    Index rows() const { return m_matrix->rows(); }
    Index cols() const { return m_matrix->cols(); }
    LargeIndex nnz() const { return m_matrix->nnz(); }

    // Get underlying matrix for solver
    SparseMatrix* get_matrix() { return m_matrix.get(); }
};

// Python wrapper for LPData
class PyLPData {
private:
    std::unique_ptr<LPData> m_data;

public:
    PyLPData(py::array_t<Real> c, py::array_t<Real> xlb, py::array_t<Real> xub,
             py::array_t<Real> l, py::array_t<Real> u, PySparseMatrix& A) {
        
        auto c_vec = numpy_to_vector(c);
        auto xlb_vec = numpy_to_vector(xlb);
        auto xub_vec = numpy_to_vector(xub);
        auto l_vec = numpy_to_vector(l);
        auto u_vec = numpy_to_vector(u);
        
        // Create a copy of the sparse matrix
        // In practice, you'd want to move or share the matrix
        auto matrix_copy = std::make_unique<SparseMatrix>(*A.get_matrix());
        
        m_data = createLPData(c_vec, xlb_vec, xub_vec, l_vec, u_vec, std::move(matrix_copy));
    }

    // Factory method from numpy arrays
    static PyLPData from_arrays(py::array_t<Real> c, py::array_t<Real> A_data,
                               py::array_t<Index> A_indices, py::array_t<LargeIndex> A_indptr,
                               py::array_t<Real> xlb, py::array_t<Real> xub,
                               py::array_t<Real> l, py::array_t<Real> u,
                               Index rows, Index cols) {
        
        PySparseMatrix A(A_data, A_indices, A_indptr, rows, cols);
        return PyLPData(c, xlb, xub, l, u, A);
    }

    // Properties
    Index num_vars() const { return m_data->numVars(); }
    Index num_cons() const { return m_data->numCons(); }
    Index total_vars() const { return m_data->totalVars(); }

    // Get underlying data for solver
    LPData* get_data() { return m_data.get(); }
};

// Python wrapper for Timings
class PyTimings {
private:
    const Timings* m_timings;

public:
    PyTimings(const Timings& timings) : m_timings(&timings) {}

    Real matvec() const { return m_timings->matvec; }
    Real ratiotest() const { return m_timings->ratiotest; }
    Real scan() const { return m_timings->scan; }
    Real ftran() const { return m_timings->ftran; }
    Real btran() const { return m_timings->btran; }
    Real ftran2() const { return m_timings->ftran2; }
    Real factor() const { return m_timings->factor; }
    Real updatefactor() const { return m_timings->updatefactor; }
    Real updateiters() const { return m_timings->updateiters; }
    Real extra() const { return m_timings->extra; }
    
    py::dict to_dict() const {
        py::dict result;
        result["matvec"] = m_timings->matvec;
        result["ratiotest"] = m_timings->ratiotest;
        result["scan"] = m_timings->scan;
        result["ftran"] = m_timings->ftran;
        result["btran"] = m_timings->btran;
        result["ftran2"] = m_timings->ftran2;
        result["factor"] = m_timings->factor;
        result["updatefactor"] = m_timings->updatefactor;
        result["updateiters"] = m_timings->updateiters;
        result["extra"] = m_timings->extra;
        return result;
    }
};

// Python wrapper for DualSimplexSolver
class PyDualSimplexSolver {
private:
    std::unique_ptr<DualSimplexSolver> m_solver;

public:
    PyDualSimplexSolver(PyLPData& data) {
        // Move the LPData into the solver
        auto lp_data = std::unique_ptr<LPData>(data.get_data());
        m_solver = std::make_unique<DualSimplexSolver>(std::move(lp_data));
    }

    // Main solve method
    void solve() {
        m_solver->solve();
    }

    // Solution access
    Real objective_value() const {
        return m_solver->getObjectiveValue();
    }

    py::array_t<Real> solution() const {
        const auto& sol = m_solver->getSolution();
        return vector_to_numpy(sol);
    }

    SolverStatus status() const {
        return m_solver->getStatus();
    }

    PyTimings timings() const {
        return PyTimings(m_solver->getTimings());
    }

    // Status checking methods
    bool is_optimal() const {
        return m_solver->getStatus() == SolverStatus::Optimal;
    }

    bool is_infeasible() const {
        return m_solver->getStatus() == SolverStatus::Infeasible;
    }

    bool is_unbounded() const {
        return m_solver->getStatus() == SolverStatus::Unbounded;
    }

    // Get solution for structural variables only
    py::array_t<Real> primal_solution() const {
        const auto& full_sol = m_solver->getSolution();
        // We need to get the number of structural variables from the LP data
        // For now, return the full solution - this would be refined in actual implementation
        return vector_to_numpy(full_sol);
    }
};

// Utility functions
namespace utils {
    // Convert scipy.sparse matrix to our format
    PySparseMatrix scipy_to_sparse(py::object scipy_matrix) {
        return PySparseMatrix::from_scipy_csc(scipy_matrix);
    }

    // Create standard form LP from general form
    PyLPData create_standard_form(py::array_t<Real> c, py::object A_scipy,
                                 py::array_t<Real> b, py::array_t<Real> bounds_lower,
                                 py::array_t<Real> bounds_upper) {
        // Convert scipy matrix
        auto A = PySparseMatrix::from_scipy_csc(A_scipy);
        
        // Create bounds for constraints (equality constraints)
        auto b_vec = numpy_to_vector(b);
        auto l = vector_to_numpy(b_vec);  // l = b
        auto u = vector_to_numpy(b_vec);  // u = b (equality)
        
        return PyLPData(c, bounds_lower, bounds_upper, l, u, A);
    }

    // Solve simple LP from numpy arrays
    py::dict solve_lp(py::array_t<Real> c, py::object A_scipy, py::array_t<Real> b,
                     py::array_t<Real> bounds_lower = py::array_t<Real>(),
                     py::array_t<Real> bounds_upper = py::array_t<Real>()) {
        
        // Set default bounds if not provided
        Index n_vars = c.size();
        std::vector<Real> lb(n_vars, 0.0);  // Default: x >= 0
        std::vector<Real> ub(n_vars, std::numeric_limits<Real>::infinity());
        
        if (bounds_lower.size() > 0) {
            lb = numpy_to_vector(bounds_lower);
        }
        if (bounds_upper.size() > 0) {
            ub = numpy_to_vector(bounds_upper);
        }
        
        auto lb_array = vector_to_numpy(lb);
        auto ub_array = vector_to_numpy(ub);
        
        // Create LP data
        auto lp_data = create_standard_form(c, A_scipy, b, lb_array, ub_array);
        
        // Solve
        PyDualSimplexSolver solver(lp_data);
        solver.solve();
        
        // Return results as dictionary
        py::dict result;
        result["success"] = solver.is_optimal();
        result["status"] = static_cast<int>(solver.status());
        result["fun"] = solver.objective_value();
        result["x"] = solver.primal_solution();
        result["timings"] = solver.timings().to_dict();
        
        return result;
    }
}

// Module definition
PYBIND11_MODULE(simplex_core, m) {
    m.doc() = "Modern C++ Dual Simplex Solver with Python Bindings";
    
    // Enums
    py::enum_<BoundType>(m, "BoundType")
        .value("LowerBound", BoundType::LowerBound)
        .value("UpperBound", BoundType::UpperBound)
        .value("Range", BoundType::Range)
        .value("Fixed", BoundType::Fixed)
        .value("Free", BoundType::Free);
    
    py::enum_<VariableState>(m, "VariableState")
        .value("Basic", VariableState::Basic)
        .value("AtLower", VariableState::AtLower)
        .value("AtUpper", VariableState::AtUpper)
        .value("AtFixed", VariableState::AtFixed);
    
    py::enum_<SolverStatus>(m, "SolverStatus")
        .value("Uninitialized", SolverStatus::Uninitialized)
        .value("Initialized", SolverStatus::Initialized)
        .value("PrimalFeasible", SolverStatus::PrimalFeasible)
        .value("DualFeasible", SolverStatus::DualFeasible)
        .value("Optimal", SolverStatus::Optimal)
        .value("Unbounded", SolverStatus::Unbounded)
        .value("Infeasible", SolverStatus::Infeasible);
    
    // SparseMatrix class
    py::class_<PySparseMatrix>(m, "SparseMatrix")
        .def(py::init<py::array_t<Real>, py::array_t<Index>, py::array_t<LargeIndex>, Index, Index>(),
             "Create sparse matrix from data, indices, indptr arrays")
        .def_static("from_scipy", &PySparseMatrix::from_scipy_csc,
                   "Create from scipy.sparse.csc_matrix")
        .def("multiply", &PySparseMatrix::multiply, "Matrix-vector multiplication")
        .def("multiply_transpose", &PySparseMatrix::multiply_transpose, 
             "Transpose matrix-vector multiplication")
        .def_property_readonly("rows", &PySparseMatrix::rows)
        .def_property_readonly("cols", &PySparseMatrix::cols)
        .def_property_readonly("nnz", &PySparseMatrix::nnz)
        .def("__repr__", [](const PySparseMatrix& m) {
            return "<SparseMatrix " + std::to_string(m.rows()) + "x" + 
                   std::to_string(m.cols()) + " nnz=" + std::to_string(m.nnz()) + ">";
        });
    
    // LPData class
    py::class_<PyLPData>(m, "LPData")
        .def(py::init<py::array_t<Real>, py::array_t<Real>, py::array_t<Real>,
                     py::array_t<Real>, py::array_t<Real>, PySparseMatrix&>())
        .def_static("from_arrays", &PyLPData::from_arrays,
                   "Create LP data from numpy arrays")
        .def_property_readonly("num_vars", &PyLPData::num_vars)
        .def_property_readonly("num_cons", &PyLPData::num_cons)
        .def_property_readonly("total_vars", &PyLPData::total_vars);
    
    // Timings class
    py::class_<PyTimings>(m, "Timings")
        .def_property_readonly("matvec", &PyTimings::matvec)
        .def_property_readonly("ratiotest", &PyTimings::ratiotest)
        .def_property_readonly("scan", &PyTimings::scan)
        .def_property_readonly("ftran", &PyTimings::ftran)
        .def_property_readonly("btran", &PyTimings::btran)
        .def_property_readonly("ftran2", &PyTimings::ftran2)
        .def_property_readonly("factor", &PyTimings::factor)
        .def_property_readonly("updatefactor", &PyTimings::updatefactor)
        .def_property_readonly("updateiters", &PyTimings::updateiters)
        .def_property_readonly("extra", &PyTimings::extra)
        .def("to_dict", &PyTimings::to_dict, "Convert to dictionary")
        .def("__repr__", [](const PyTimings& t) {
            return "<Timings total=" + 
                   std::to_string(t.matvec() + t.ratiotest() + t.scan() + 
                                 t.ftran() + t.btran() + t.factor()) + "s>";
        });
    
    // DualSimplexSolver class
    py::class_<PyDualSimplexSolver>(m, "DualSimplexSolver")
        .def(py::init<PyLPData&>())
        .def("solve", &PyDualSimplexSolver::solve, "Solve the linear program")
        .def_property_readonly("objective_value", &PyDualSimplexSolver::objective_value)
        .def_property_readonly("solution", &PyDualSimplexSolver::solution)
        .def_property_readonly("primal_solution", &PyDualSimplexSolver::primal_solution)
        .def_property_readonly("status", &PyDualSimplexSolver::status)
        .def_property_readonly("timings", &PyDualSimplexSolver::timings)
        .def("is_optimal", &PyDualSimplexSolver::is_optimal)
        .def("is_infeasible", &PyDualSimplexSolver::is_infeasible)
        .def("is_unbounded", &PyDualSimplexSolver::is_unbounded)
        .def("__repr__", [](const PyDualSimplexSolver& s) {
            return "<DualSimplexSolver status=" + 
                   std::to_string(static_cast<int>(s.status())) + ">";
        });
    
    // Utility functions
    py::module utils_module = m.def_submodule("utils", "Utility functions");
    utils_module.def("scipy_to_sparse", &utils::scipy_to_sparse,
                    "Convert scipy.sparse matrix to SparseMatrix");
    utils_module.def("create_standard_form", &utils::create_standard_form,
                    "Create standard form LP from general form");
    utils_module.def("solve_lp", &utils::solve_lp,
                    "Solve linear program from numpy arrays",
                    py::arg("c"), py::arg("A"), py::arg("b"),
                    py::arg("bounds_lower") = py::array_t<Real>(),
                    py::arg("bounds_upper") = py::array_t<Real>());
    
    // Module-level solve function for convenience
    m.def("solve", &utils::solve_lp,
          "Solve linear program: min c^T x s.t. Ax = b, bounds_lower <= x <= bounds_upper",
          py::arg("c"), py::arg("A"), py::arg("b"),
          py::arg("bounds_lower") = py::array_t<Real>(),
          py::arg("bounds_upper") = py::array_t<Real>());
    
    // Version info
    m.attr("__version__") = "1.0.0";
    m.attr("__author__") = "Modern Simplex Team";
}