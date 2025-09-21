#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Sparse>

using dvec = Eigen::VectorXd;
using dmat = Eigen::MatrixXd;
using spmat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

struct KKT {
    double stat{0.0}, eq{0.0}, ineq{0.0}, comp{0.0};
};

struct SolverInfo {
    std::string mode{"ip"}; // algorithmic mode (ip, sqp, etc.)
    double step_norm{0.0};
    bool accepted{true};
    bool converged{true};

    double f{0.0};     // objective value
    double theta{0.0}; // merit or filter measure
    double stat{0.0};  // gradient norm (safe_inf_norm(g))
    double ineq{0.0};  // infinity norm of violated inequalities
    double eq{0.0};    // infinity norm of equality violations
    double comp{0.0};  // complementarity measure

    int ls_iters{0};   // line search iterations
    double alpha{0.0}; // step size
    double rho{0.0};   // trust region ratio
    double tr_radius{0.0};
    double delta{0.0}; // trust region radius
    double mu{0.0};    // barrier parameter

    bool shifted_barrier{false}; // whether the barrier was shifted
    double tau_shift{0.0};       // amount of barrier shift
    double bound_shift{0.0};     // amount of bound shift
};
// csr_map_pybind.h

// csr_map_pybind.h
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Sparse>

namespace py = pybind11;

using f64 = double;
using i32 = int32_t;
using SparseRow = Eigen::SparseMatrix<f64, Eigen::RowMajor, i32>;
using SparseRowMap = Eigen::Map<const SparseRow>;

struct PyCSRHandle {
    int rows = 0, cols = 0, nnz = 0;
    py::array_t<i32> indptr;   // 1D, length rows+1
    py::array_t<i32> indices;  // 1D, length nnz
    py::array_t<f64> data;     // 1D, length nnz

    SparseRowMap map() const {
        const i32* p_indptr  = indptr.data();
        const i32* p_indices = indices.data();
        const f64* p_data    = data.data();
        return SparseRowMap(rows, cols, nnz, p_indptr, p_indices, p_data);
    }
};

// Strict zero-copy parse: requires exact dtypes (np.int32 / np.float64)
inline PyCSRHandle parse_csr_tuple(const py::handle& obj) {
    if (!py::isinstance<py::tuple>(obj))
        throw py::type_error("Expected tuple (shape, indptr, indices, data)");
    py::tuple t = obj.cast<py::tuple>();
    if (t.size() != 4)
        throw py::value_error("CSR tuple must have 4 elements");

    auto shape = t[0].cast<std::pair<int,int>>();
    const int rows = shape.first;
    const int cols = shape.second;

    // Use ensure() to verify dtype without forcing a copy
    py::array indptr_obj  = t[1].cast<py::array>();
    py::array indices_obj = t[2].cast<py::array>();
    py::array data_obj    = t[3].cast<py::array>();

    auto indptr  = py::array_t<i32>::ensure(indptr_obj);
    auto indices = py::array_t<i32>::ensure(indices_obj);
    auto data    = py::array_t<f64>::ensure(data_obj);

    if (!indptr || !indices || !data)
        throw py::type_error("Expected dtypes: indptr=int32, indices=int32, data=float64");

    if (indptr.ndim() != 1 || indices.ndim() != 1 || data.ndim() != 1)
        throw py::value_error("indptr/indices/data must be 1D arrays");
    if (indptr.shape(0) != static_cast<py::ssize_t>(rows + 1))
        throw py::value_error("len(indptr) must be rows+1");

    const int nnz = static_cast<int>(data.shape(0));
    if (indices.shape(0) != static_cast<py::ssize_t>(nnz))
        throw py::value_error("len(indices) must equal len(data)");

    PyCSRHandle h;
    h.rows = rows; h.cols = cols; h.nnz = nnz;
    h.indptr = std::move(indptr);
    h.indices = std::move(indices);
    h.data = std::move(data);
    return h;
}
