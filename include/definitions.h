#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <variant>
#include <unordered_map>
#include <string>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <cstring> // std::memcpy

using dvec = Eigen::VectorXd;
using dmat = Eigen::MatrixXd;
using spmat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

// The value type your map stores (adjust if yours differs)
using Val = std::variant<double, dvec, dmat, spmat>;
using Dict = std::unordered_map<std::string, Val>;

// ---------- small helpers ----------
template <class T>
inline const T& must_get(const Dict& d, std::string_view key) {
    auto it = d.find(std::string(key));
    if (it == d.end())
        throw std::out_of_range(std::string("key not found: ") + std::string(key));
    return std::get<T>(it->second);            // throws if wrong alternative
}

inline bool has_key(const Dict& d, std::string_view key) {
    return d.find(std::string(key)) != d.end();
}

// Accept either dense or sparse and always return sparse
inline spmat to_spmat(const Val& v, int rows, int cols) {
    return std::visit([&](auto&& M) -> spmat {
        using T = std::decay_t<decltype(M)>;
        if constexpr (std::is_same_v<T, spmat>) {
            return M;
        } else if constexpr (std::is_same_v<T, dmat>) {
            spmat S(rows, cols);
            // Eigen: use sparseView for efficient conversion
            S = M.sparseView();
            return S;
        } else {
            throw std::runtime_error("Expected (dense|sparse) matrix variant");
        }
    }, v);
}

namespace nb = nanobind;
using namespace nb::literals;

inline bool is_numpy_array(const nb::handle &h) {
    // Fast check: presence of __array_interface__ (covers ndarray &
    // array-likes)
    return h && nb::hasattr(h, "__array_interface__");
}

template <class T> inline std::optional<T> try_cast_opt(const nb::handle &h) {
    try {
        return nb::cast<T>(h);
    } catch (const nb::cast_error &) {
        return std::nullopt;
    }
}

// Map a 1D/2D ndarray<float64> to dvec/dmat (copy into Eigen-owned result)
inline std::optional<dvec>
ndarray_to_dvec(const nb::ndarray<nb::ro, double> &a) {
    if (a.ndim() == 1) {
        const auto n = (Eigen::Index)a.shape(0);
        dvec v(n);
        std::memcpy(v.data(), a.data(), sizeof(double) * (size_t)n);
        return v;
    } else if (a.ndim() == 2) {
        // Accept (n,1) or (1,n)
        Eigen::Index r = (Eigen::Index)a.shape(0);
        Eigen::Index c = (Eigen::Index)a.shape(1);
        if (c == 1) {
            dvec v(r);
            std::memcpy(v.data(), a.data(), sizeof(double) * (size_t)r);
            return v;
        } else if (r == 1) {
            dvec v(c);
            // row vector -> copy contiguous row
            std::memcpy(v.data(), a.data(), sizeof(double) * (size_t)c);
            return v;
        }
    }
    return std::nullopt;
}

inline std::optional<dmat>
ndarray_to_dmat(const nb::ndarray<nb::ro, double> &a) {
    if (a.ndim() != 2)
        return std::nullopt;
    const auto r = (Eigen::Index)a.shape(0);
    const auto c = (Eigen::Index)a.shape(1);
    dmat M(r, c);
    // Regardless of C/F order, memcpy row-major linear layout then let Eigen
    // own it. (If you want to preserve strides/orders, you can branch on
    // a.strides().) Here we copy element-by-element if strides are
    // non-contiguous.
    const auto s0 = a.stride(0) / (ptrdiff_t)sizeof(double);
    const auto s1 = a.stride(1) / (ptrdiff_t)sizeof(double);
    const double *ptr = a.data();

    if ((s1 == 1 && s0 == (ptrdiff_t)c) || (s0 == 1 && s1 == (ptrdiff_t)r)) {
        // Contiguous in either C or F order -> bulk copy then reshape
        // appropriately. We'll do a safe element copy to avoid order headaches:
        for (Eigen::Index i = 0; i < r; ++i)
            for (Eigen::Index j = 0; j < c; ++j)
                M(i, j) = ptr[i * s0 + j * s1];
    } else {
        // General strided copy
        for (Eigen::Index i = 0; i < r; ++i)
            for (Eigen::Index j = 0; j < c; ++j)
                M(i, j) = ptr[i * s0 + j * s1];
    }
    return M;
}

static inline nb::dict call_eval_all_dict(const nb::object &model,
                                          const Eigen::VectorXd &x,
                                          const nb::list &need) {
    if (!model || !nb::hasattr(model, "eval_all")) {
        throw std::invalid_argument(
            "call_eval_all_dict: model has no 'eval_all'.");
    }

    // nb::gil_scoped_acquire gil; // calling into Python

    nb::object out_obj = model.attr("eval_all")(x, "components"_a = need);

    if (!nb::isinstance<nb::dict>(out_obj)) {
        std::string typ = "<unknown>";
        try {
            typ = nb::cast<std::string>(
                out_obj.attr("__class__").attr("__name__"));
        } catch (...) {
            // ignore; keep "<unknown>"
        }
        throw std::runtime_error(
            "eval_all(x, components=...) must return a dict; got: " + typ);
    }
    return nb::cast<nb::dict>(out_obj);
}

template <class T>
static std::optional<T> to_dense_optional(const nb::handle &obj) {
    if (!obj || obj.is_none())
        return std::nullopt;

    try {
        // 1) If scipy.sparse or any object exposing .toarray(), prefer that
        if (nb::hasattr(obj, "toarray")) {
            nb::object arr = obj.attr("toarray")();
            return nb::cast<T>(arr);
        }

        // 2) Fast path for NumPy ndarray (avoids intermediate Python
        // conversions)
        if (is_numpy_array(obj)) {
            // Only float64 is supported here; adjust if you want to accept
            // other dtypes.
            nb::ndarray<nb::ro, double> a =
                nb::ndarray<nb::ro, double>(obj.ptr());
            if constexpr (std::is_same_v<T, dvec>) {
                if (auto v = ndarray_to_dvec(a))
                    return v;
                // Fall back to generic cast for odd shapes
                return nb::cast<T>(obj);
            } else if constexpr (std::is_same_v<T, dmat>) {
                if (auto M = ndarray_to_dmat(a))
                    return M;
                return nb::cast<T>(obj);
            } else {
                return nb::cast<T>(obj);
            }
        }

        // 3) Generic cast (handles Python lists/tuples/Eigen types already
        // bound, etc.)
        return nb::cast<T>(obj);
    } catch (const nb::cast_error &) {
        return std::nullopt;
    }
}

// Safe: return d[k] if present, else nb::none()
static inline nb::object nb_dict_get(const nb::dict &d, const nb::handle &k) {
    return d.contains(k) ? d[k] : nb::none();
}
// ---------- Python attribute helpers ----------
namespace pyu {
[[nodiscard]] inline bool has_attr(const nb::object &o,
                                   const char *name) noexcept {
    return o.is_valid() && PyObject_HasAttrString(o.ptr(), name);
}

template <class T>
[[nodiscard]] T getattr_or(const nb::object &o, const char *name,
                           const T &fallback) {
    if (!o.is_valid() || !has_attr(o, name)) [[likely]]
        return fallback;
    try {
        return nb::cast<T>(o.attr(name));
    } catch (...) {
        return fallback;
    }
}
} // namespace pyu

// ---------- Optimized Python ↔ Eigen conversions ----------
namespace pyconv {

// Cache frequently used numpy module
inline nb::object &get_numpy_module() {
    static nb::object numpy = nb::module_::import_("numpy");
    return numpy;
}

inline bool is_numpy_c_contig(const nb::ndarray<nb::numpy> &a) noexcept {
    if (!a.is_valid()) [[unlikely]]
        return false;
    const auto ndim = a.ndim();
    if (ndim == 1) [[likely]]
        return a.stride(0) == sizeof(double);
    if (ndim == 2)
        return a.stride(1) == sizeof(double) &&
               a.stride(0) == a.stride(1) * a.shape(1);
    return false;
}

// Helper function to check if object is numpy array (avoid template issues)
[[nodiscard]] inline bool is_numpy_array(const nb::object &obj) noexcept {
    return PyObject_HasAttrString(obj.ptr(), "__array_interface__") ||
           PyObject_HasAttrString(obj.ptr(), "__array__");
}

// Helper to get array dtype as string - optimized with string_view
[[nodiscard]] inline std::string_view
get_array_dtype(const nb::object &obj) noexcept {
    try {
        static thread_local std::string dtype_str;
        dtype_str = nb::cast<std::string>(obj.attr("dtype").attr("name"));
        return dtype_str;
    } catch (...) {
        return "unknown";
    }
}

// Helper to get array shape - optimized with span-like interface
[[nodiscard]] inline std::vector<size_t>
get_array_shape(const nb::object &obj) {
    try {
        nb::object shape_tuple = obj.attr("shape");
        nb::tuple shape_tup = nb::cast<nb::tuple>(shape_tuple);
        const size_t size = shape_tup.size();

        std::vector<size_t> shape;
        shape.reserve(size); // Pre-allocate

        for (size_t i = 0; i < size; ++i) {
            shape.emplace_back(nb::cast<size_t>(shape_tup[i]));
        }
        return shape;
    } catch (...) {
        return {};
    }
}
// Ultra-fast optimized vector conversion
[[nodiscard]] inline dvec to_vec_safe(const nb::object &obj) {
    if (!obj.is_valid() || obj.is_none()) [[unlikely]]
        return dvec{};

    // Fast path: Direct numpy array conversion (most common case)
    if (is_numpy_array(obj)) [[likely]] {
        try {
            auto a = nb::cast<nb::ndarray<nb::numpy>>(obj);

            // Handle scalar case early
            if (a.ndim() == 0) [[unlikely]] {
                return dvec::Constant(1,
                                      *static_cast<const double *>(a.data()));
            }

            if (a.ndim() != 1) [[unlikely]]
                throw std::runtime_error("Expected 1-D array");

            const size_t n = a.shape(0);
            if (n == 0) [[unlikely]]
                return dvec{};

            // Check if already contiguous double array
            if (a.dtype() == nb::dtype<double>() &&
                a.stride(0) == sizeof(double)) [[likely]] {
                // Direct memory copy - fastest possible
                dvec v(n);
                std::memcpy(v.data(), a.data(), n * sizeof(double));
                return v;
            } else {
                // Need conversion - only do it once
                auto &numpy = get_numpy_module();
                nb::object arr = numpy.attr("ascontiguousarray")(
                    obj, nb::arg("dtype") = "float64");
                auto converted = nb::cast<nb::ndarray<nb::numpy>>(arr);

                dvec v(n);
                std::memcpy(v.data(), converted.data(), n * sizeof(double));
                return v;
            }
        } catch (...) {
            // Fall through to slower methods
        }
    }

    // Fast sequence path: Single conversion + pre-size optimization
    if (PySequence_Check(obj.ptr()) && !PyUnicode_Check(obj.ptr())) [[likely]] {
        // Single call to PySequence_Fast
        PyObject *fast_seq = PySequence_Fast(obj.ptr(), "");
        if (!fast_seq) [[unlikely]]
            throw std::runtime_error("Unable to convert to vector");

        const Py_ssize_t size = PySequence_Fast_GET_SIZE(fast_seq);
        if (size <= 0) [[unlikely]] {
            Py_DECREF(fast_seq);
            return dvec{};
        }

        dvec v(size);
        PyObject **items = PySequence_Fast_ITEMS(fast_seq);

        // Vectorized conversion with minimal error checking
        for (Py_ssize_t i = 0; i < size; ++i) {
            v[i] = PyFloat_AsDouble(items[i]);
            // Optional: Check PyErr_Occurred() only once at end if needed
        }

        Py_DECREF(fast_seq);
        return v;
    }

    throw std::runtime_error("Unable to convert to vector");
}
// ---------- Zero-copy sparse views (CSR for RowMajor / CSC for ColMajor)
// ----------
template <class SparseT>
inline constexpr bool kRowMajor = (SparseT::Options & Eigen::RowMajor) != 0;

template <class SparseT> using SparseIndexT = typename SparseT::StorageIndex;

struct SciPyCompressed {
    nb::object mat;
    nb::ndarray<nb::numpy> indptr, indices, data;
    int rows = 0, cols = 0;
    long long nnz = 0;
};

template <class SparseT>
bool extract_compressed_for(const nb::object &obj, SciPyCompressed &out) {
    if (obj.is_none()) [[unlikely]]
        return false;
    nb::object m = obj;

    constexpr const char *want = kRowMajor<SparseT> ? "tocsr" : "tocsc";
    if (!PyObject_HasAttrString(m.ptr(), "indptr") ||
        !PyObject_HasAttrString(m.ptr(), "indices") ||
        !PyObject_HasAttrString(m.ptr(), "data")) {
        if (PyObject_HasAttrString(m.ptr(), want)) [[likely]]
            m = m.attr(want)();
        else
            return false;
    }

    try {
        auto shape = nb::cast<std::pair<size_t, size_t>>(m.attr("shape"));
        out.rows = static_cast<int>(shape.first);
        out.cols = static_cast<int>(shape.second);

        // Extract arrays
        nb::object indptr_obj = m.attr("indptr");
        nb::object indices_obj = m.attr("indices");
        nb::object data_obj = m.attr("data");

        // Check if they're array-like
        if (!is_numpy_array(indptr_obj) || !is_numpy_array(indices_obj) ||
            !is_numpy_array(data_obj)) [[unlikely]] {
            return false;
        }

        // Get as generic arrays
        out.indptr = nb::cast<nb::ndarray<nb::numpy>>(indptr_obj);
        out.indices = nb::cast<nb::ndarray<nb::numpy>>(indices_obj);
        out.data = nb::cast<nb::ndarray<nb::numpy>>(data_obj);

        // Optimized: Check data type and convert only if needed
        if (out.data.dtype() != nb::dtype<double>()) [[unlikely]] {
            auto &numpy = get_numpy_module(); // Use cached module
            nb::object converted = numpy.attr("ascontiguousarray")(
                data_obj, nb::arg("dtype") = "float64");
            out.data = nb::cast<nb::ndarray<nb::numpy>>(std::move(converted));
        }

        // Optimized: Direct dtype check instead of string comparison
        constexpr size_t int32_size = sizeof(int32_t);
        constexpr size_t uint32_size = sizeof(uint32_t);
        constexpr size_t int64_size = sizeof(int64_t);
        constexpr size_t uint64_size = sizeof(uint64_t);

        const size_t indptr_itemsize = out.indptr.itemsize();
        const size_t indices_itemsize = out.indices.itemsize();

        const bool indptr_ok =
            (indptr_itemsize == int32_size || indptr_itemsize == uint32_size ||
             indptr_itemsize == int64_size || indptr_itemsize == uint64_size);
        const bool indices_ok =
            (indices_itemsize == int32_size ||
             indices_itemsize == uint32_size ||
             indices_itemsize == int64_size || indices_itemsize == uint64_size);

        if (!indptr_ok || !indices_ok) [[unlikely]]
            return false;

        // Get data size - direct shape access
        out.nnz = static_cast<long long>(out.data.shape(0));

        out.mat = std::move(m);
        return true;

    } catch (...) {
        return false;
    }
}

template <class SparseT> struct PySparseView {
    using MapT = Eigen::Map<const SparseT>;
    nb::object owner;
    nb::ndarray<nb::numpy> indptr, indices, data;
    MapT map;

    PySparseView(nb::object o, nb::ndarray<nb::numpy> ip,
                 nb::ndarray<nb::numpy> idx, nb::ndarray<nb::numpy> d, int r,
                 int c, long long nnz)
        : owner(std::move(o)), indptr(std::move(ip)), indices(std::move(idx)),
          data(std::move(d)),
          map(r, c, static_cast<SparseIndexT<SparseT>>(nnz),
              reinterpret_cast<const SparseIndexT<SparseT> *>(indptr.data()),
              reinterpret_cast<const SparseIndexT<SparseT> *>(indices.data()),
              reinterpret_cast<const double *>(data.data())) {}
};

template <class SparseT>
[[nodiscard]] std::optional<PySparseView<SparseT>>
to_sparse_view_any(const nb::object &o) {
    SciPyCompressed s;
    if (!extract_compressed_for<SparseT>(o, s)) [[unlikely]]
        return std::nullopt;
    if (s.indptr.itemsize() != sizeof(SparseIndexT<SparseT>) ||
        s.indices.itemsize() != sizeof(SparseIndexT<SparseT>)) [[unlikely]]
        return std::nullopt;
    return PySparseView<SparseT>(std::move(s.mat), std::move(s.indptr),
                                 std::move(s.indices), std::move(s.data),
                                 s.rows, s.cols, s.nnz);
}

// Optimized version - same interface, better performance
[[nodiscard]] inline spmat to_sparse(const nb::object &obj) {
    if (obj.is_none()) [[unlikely]]
        return spmat{};

    // Try SciPy compressed
    if (auto view = to_sparse_view_any<spmat>(obj)) [[likely]] {
        spmat A = view->map;
        A.makeCompressed();
        return A;
    }

    // Dense fallback - optimized
    try {
        auto &numpy = get_numpy_module(); // Use cached module

        // Optimized: Check if conversion is needed
        nb::object arr_obj;
        if (is_numpy_array(obj)) [[likely]] {
            auto candidate = nb::cast<nb::ndarray<nb::numpy>>(obj);
            // Only convert if not already float64 and C-contiguous
            bool is_contiguous =
                candidate.ndim() == 2 &&
                candidate.stride(1) == sizeof(double) &&
                candidate.stride(0) == candidate.shape(1) * sizeof(double);

            if (candidate.dtype() == nb::dtype<double>() && is_contiguous)
                [[likely]] {
                arr_obj = obj; // No conversion needed
            } else {
                arr_obj = numpy.attr("ascontiguousarray")(
                    obj, nb::arg("dtype") = "float64");
            }
        } else {
            arr_obj = numpy.attr("array")(obj, nb::arg("dtype") = "float64");
            arr_obj = numpy.attr("ascontiguousarray")(arr_obj);
        }

        // Get the array and check dimensions
        auto a = nb::cast<nb::ndarray<nb::numpy>>(arr_obj);
        if (a.ndim() != 2) [[unlikely]]
            throw std::runtime_error("to_sparse: expected 2-D array");

        const int rows = static_cast<int>(a.shape(0));
        const int cols = static_cast<int>(a.shape(1));

        // Early exit for empty matrices
        if (rows == 0 || cols == 0) [[unlikely]]
            return spmat(rows, cols);

        // Same mapping as original
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::RowMajor>>
            M(static_cast<const double *>(a.data()), rows, cols);
        spmat S = M.sparseView(0.0, 1.0);
        S.makeCompressed();
        return S;

    } catch (const std::exception &e) {
        throw std::runtime_error("to_sparse conversion failed: " +
                                 std::string(e.what()));
    }
}

} // namespace pyconv

struct KKT {
    double stat{0.0}, eq{0.0}, ineq{0.0}, comp{0.0};
};

namespace kkt {
// ---------------------------- reusable API ----------------------------

// ------------------------------ config -------------------------------
struct ChompConfig {
    double cg_tol = 1e-6;
    int cg_maxit = 50;
    double ip_hess_reg0 = 1e-8;
    double schur_dense_cutoff = 0.25;
    std::string prec_type = "ssor"; // "jacobi" | "ssor" | "none"
    double ssor_omega = 1.0;
    std::string sym_ordering = "none"; // "amd" | "none"
    bool use_simd = true;
    int block_size = 256;
    bool adaptive_gamma = false;
};

struct KKTReusable {
    virtual ~KKTReusable() = default;
    virtual std::pair<dvec, dvec> solve(const dvec &r1,
                                        const std::optional<dvec> &r2,
                                        double cg_tol = 1e-8,
                                        int cg_maxit = 200) = 0;
};

// ---------------------------- strategy base ---------------------------
struct KKTStrategy {
    virtual ~KKTStrategy() = default;
    virtual std::tuple<dvec, dvec, std::shared_ptr<KKTReusable>>
    factor_and_solve(const spmat &W, const std::optional<spmat> &G,
                     const dvec &r1, const std::optional<dvec> &r2,
                     const ChompConfig &cfg, std::optional<double> regularizer,
                     std::unordered_map<std::string, dvec> &cache,
                     double delta = 0.0,
                     std::optional<double> gamma = std::nullopt,
                     bool assemble_schur_if_m_small = true,
                     bool use_prec = true) = 0;
    std::string name;
};

} // namespace kkt

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
#include <Eigen/Sparse>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using f64 = double;
using i32 = int32_t;
using SparseRow = Eigen::SparseMatrix<f64, Eigen::RowMajor, i32>;
using SparseRowMap = Eigen::Map<const SparseRow>;

struct PyCSRHandle {
    int rows = 0, cols = 0, nnz = 0;
    py::array_t<i32> indptr;  // 1D, length rows+1
    py::array_t<i32> indices; // 1D, length nnz
    py::array_t<f64> data;    // 1D, length nnz

    SparseRowMap map() const {
        const i32 *p_indptr = indptr.data();
        const i32 *p_indices = indices.data();
        const f64 *p_data = data.data();
        return SparseRowMap(rows, cols, nnz, p_indptr, p_indices, p_data);
    }
};

// Strict zero-copy parse: requires exact dtypes (np.int32 / np.float64)
inline PyCSRHandle parse_csr_tuple(const py::handle &obj) {
    if (!py::isinstance<py::tuple>(obj))
        throw py::type_error("Expected tuple (shape, indptr, indices, data)");
    py::tuple t = obj.cast<py::tuple>();
    if (t.size() != 4)
        throw py::value_error("CSR tuple must have 4 elements");

    auto shape = t[0].cast<std::pair<int, int>>();
    const int rows = shape.first;
    const int cols = shape.second;

    // Use ensure() to verify dtype without forcing a copy
    py::array indptr_obj = t[1].cast<py::array>();
    py::array indices_obj = t[2].cast<py::array>();
    py::array data_obj = t[3].cast<py::array>();

    auto indptr = py::array_t<i32>::ensure(indptr_obj);
    auto indices = py::array_t<i32>::ensure(indices_obj);
    auto data = py::array_t<f64>::ensure(data_obj);

    if (!indptr || !indices || !data)
        throw py::type_error(
            "Expected dtypes: indptr=int32, indices=int32, data=float64");

    if (indptr.ndim() != 1 || indices.ndim() != 1 || data.ndim() != 1)
        throw py::value_error("indptr/indices/data must be 1D arrays");
    if (indptr.shape(0) != static_cast<py::ssize_t>(rows + 1))
        throw py::value_error("len(indptr) must be rows+1");

    const int nnz = static_cast<int>(data.shape(0));
    if (indices.shape(0) != static_cast<py::ssize_t>(nnz))
        throw py::value_error("len(indices) must equal len(data)");

    PyCSRHandle h;
    h.rows = rows;
    h.cols = cols;
    h.nnz = nnz;
    h.indptr = std::move(indptr);
    h.indices = std::move(indices);
    h.data = std::move(data);
    return h;
}
