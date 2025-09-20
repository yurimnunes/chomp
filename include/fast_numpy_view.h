// --- fast_numpy_views.h ------------------------------------------------------
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Core>

namespace py = pybind11;

// Create a zero-copy NumPy view over an Eigen vector (read-only)
inline py::array eigen_vec_view(const Eigen::Ref<const Eigen::VectorXd>& x) {
    // shape: (n,), strides: (sizeof(double),)
    return py::array(
        py::buffer_info(
            const_cast<double*>(x.data()),                // ptr (const ignored by NumPy)
            sizeof(double),                               // itemsize
            py::format_descriptor<double>::format(),      // format
            1,                                            // ndim
            { static_cast<size_t>(x.size()) },            // shape
            { sizeof(double) }                            // strides
        ),
        // keep-alive: tie lifetime to a capsule holding no-op deleter
        py::cast(nullptr)
    );
}

// Map a Python 1D ndarray (C-contiguous, float64) to Eigen::Map<VectorXd> without copy
inline Eigen::Map<Eigen::VectorXd> map_vec_nocopy(const py::array& a) {
    // REQUIRE: dtype=float64, ndim=1, C contiguous
    auto arr = py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(a);
    auto buf = arr.request();
    if (buf.ndim != 1)
        throw std::runtime_error("Expected 1-D array.");
    return Eigen::Map<Eigen::VectorXd>(static_cast<double*>(buf.ptr),
                                       static_cast<Eigen::Index>(buf.shape[0]));
}

// Map a Python 2D ndarray (C-contiguous, float64) to Eigen::Map<MatrixXd> without copy
inline Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
map_mat_nocopy(const py::array& a, int rows=-1, int cols=-1) {
    using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    auto arr = py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(a);
    auto buf = arr.request();
    if (buf.ndim != 2) throw std::runtime_error("Expected 2-D array.");
    const int r = static_cast<int>(buf.shape[0]);
    const int c = static_cast<int>(buf.shape[1]);
    if ((rows>=0 && rows!=r) || (cols>=0 && cols!=c))
        throw std::runtime_error("Matrix shape mismatch.");
    return Eigen::Map<RowMat>(static_cast<double*>(buf.ptr), r, c);
}
