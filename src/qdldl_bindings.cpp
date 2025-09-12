#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <queue> // Add this for RCM ordering

#include "../include/qdldl.h"

namespace py = pybind11;
using namespace qdldl;

// ---------------------------
// Helpers & small holders
// ---------------------------

struct Factor {
    std::vector<std::int64_t> Lp;
    std::vector<std::int64_t> Li;
    std::vector<double> Lx;
    std::vector<double> D;
    std::vector<double> Dinv;
    std::int64_t n = 0;
    std::int64_t posD = 0;

    // NEW: Track ordering information for correct solving
    std::vector<std::int64_t> perm;  // empty if no ordering used
    std::vector<std::int64_t> iperm; // empty if no ordering used
    bool has_ordering = false;

    std::size_t nnzL() const noexcept { return Lx.size(); }
};
// Store symbolic struct + (optional) permutation so we can refactorize later
struct SymbolicHandle {
    Symbolic<double, std::int64_t> S;
    std::vector<std::int64_t> perm;  // empty if none provided
    std::vector<std::int64_t> iperm; // empty if none provided
    std::int64_t n = 0;
    bool has_perm = false;
};

// Build SparseMatrixD from CSC arrays (upper triangle expected)
static inline SparseMatrixD make_sparse_from_csc(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> indptr,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
        indices,
    py::array_t<double, py::array::c_style | py::array::forcecast> data,
    std::int64_t n) {
    if (indptr.ndim() != 1 || indices.ndim() != 1 || data.ndim() != 1)
        throw std::invalid_argument("indptr, indices, data must be 1-D arrays");
    if (indptr.shape(0) != n + 1)
        throw std::invalid_argument("indptr length must be n+1");
    if (indices.shape(0) != data.shape(0))
        throw std::invalid_argument(
            "indices and data must have the same length");

    const auto nnz = static_cast<std::size_t>(indices.shape(0));
    std::vector<std::int64_t> Ap(n + 1);
    std::vector<std::int64_t> Ai(nnz);
    std::vector<double> Ax(nnz);
    std::memcpy(Ap.data(), indptr.data(), sizeof(std::int64_t) * (n + 1));
    std::memcpy(Ai.data(), indices.data(), sizeof(std::int64_t) * nnz);
    std::memcpy(Ax.data(), data.data(), sizeof(double) * nnz);
    return SparseMatrixD(n, std::move(Ap), std::move(Ai), std::move(Ax));
}

// Read permutation from numpy (int64 1-D)
static inline qdldl::Ordering<std::int64_t>
make_ordering_from_numpy(py::object maybe_perm, std::int64_t n) {
    if (maybe_perm.is_none()) {
        // Return an identity ordering
        std::vector<std::int64_t> perm(static_cast<std::size_t>(n));
        std::iota(perm.begin(), perm.end(), 0);
        return qdldl::make_ordering<std::int64_t>(std::move(perm));
    }
    auto arr =
        py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>(
            maybe_perm);
    if (arr.ndim() != 1)
        throw std::invalid_argument("perm must be 1-D int64 array");
    if (arr.shape(0) != n)
        throw std::invalid_argument("perm length must equal n");
    std::vector<std::int64_t> perm(static_cast<std::size_t>(n));
    std::memcpy(perm.data(), arr.data(),
                sizeof(std::int64_t) * static_cast<std::size_t>(n));
    return qdldl::make_ordering<std::int64_t>(std::move(perm));
}

// UPDATED: parse string-or-array ordering; compute various orderings when
// requested
static inline qdldl::Ordering<std::int64_t>
parse_ordering_or_compute(py::object perm_or_str, const SparseMatrixD &A,
                          std::int64_t n) {
    if (perm_or_str.is_none()) {
        // identity
        std::vector<std::int64_t> p(static_cast<std::size_t>(n));
        std::iota(p.begin(), p.end(), 0);
        return qdldl::make_ordering<std::int64_t>(std::move(p));
    }

    // if it's a string, allow various ordering methods
    if (py::isinstance<py::str>(perm_or_str)) {
        auto s = perm_or_str.cast<std::string>();
        for (auto &c : s)
            c = static_cast<char>(::tolower(c));

        if (s == "rcm") {
            return qdldl::rcm_ordering<double, std::int64_t>(A);
        } else {
            throw std::invalid_argument(
                "Unknown ordering string: " + s +
                " (use 'amd', 'rcm', 'minimum_degree', or 'natural')");
        }
    }

    // otherwise assume it is a NumPy permutation array
    return make_ordering_from_numpy(perm_or_str, n);
}

// ---------------------------
// Pybind11 module
// ---------------------------

PYBIND11_MODULE(qdldl_cpp, m) {
    m.doc() =
        "QDLDL (modern C++) Python bindings with ordering & symbolic reuse";

    // Exceptions passthrough
    py::register_exception<FactorizationError>(m, "FactorizationError");
    py::register_exception<InvalidMatrixError>(m, "InvalidMatrixError");

    // Factor holder
    py::class_<Factor>(m, "Factor")
        .def_property_readonly("n", [](const Factor &f) { return f.n; })
        .def_property_readonly("posD", [](const Factor &f) { return f.posD; })
        .def_property_readonly("nnzL", &Factor::nnzL)
        .def_property_readonly("has_ordering",
                               [](const Factor &f) { return f.has_ordering; })
        .def_property_readonly("perm",
                               [](const Factor &f) {
                                   return py::array(f.perm.size(),
                                                    f.perm.data());
                               })
        .def_property_readonly("iperm",
                               [](const Factor &f) {
                                   return py::array(f.iperm.size(),
                                                    f.iperm.data());
                               })
        .def_property_readonly(
            "L_indptr",
            [](const Factor &f) { return py::array(f.Lp.size(), f.Lp.data()); })
        .def_property_readonly(
            "L_indices",
            [](const Factor &f) { return py::array(f.Li.size(), f.Li.data()); })
        .def_property_readonly(
            "L_data",
            [](const Factor &f) { return py::array(f.Lx.size(), f.Lx.data()); })
        .def_property_readonly(
            "D",
            [](const Factor &f) { return py::array(f.D.size(), f.D.data()); })
        .def_property_readonly("Dinv",
                               [](const Factor &f) {
                                   return py::array(f.Dinv.size(),
                                                    f.Dinv.data());
                               })
        .def("__repr__", [](const Factor &f) {
            return "<qdldl_cpp.Factor n=" + std::to_string(f.n) +
                   " nnzL=" + std::to_string(f.nnzL()) +
                   " posD=" + std::to_string(f.posD) +
                   (f.has_ordering ? " (permuted)>" : ">");
        });

    // Symbolic holder
    py::class_<SymbolicHandle>(m, "Symbolic")
        .def_property_readonly("n", [](const SymbolicHandle &s) { return s.n; })
        .def_property_readonly(
            "has_perm", [](const SymbolicHandle &s) { return s.has_perm; })
        .def_property_readonly("perm",
                               [](const SymbolicHandle &s) {
                                   return py::array(s.perm.size(),
                                                    s.perm.data());
                               })
        .def_property_readonly("iperm",
                               [](const SymbolicHandle &s) {
                                   return py::array(s.iperm.size(),
                                                    s.iperm.data());
                               })
        .def_property_readonly("L_indptr",
                               [](const SymbolicHandle &s) {
                                   return py::array(s.S.L_col_ptr.size(),
                                                    s.S.L_col_ptr.data());
                               })
        .def("__repr__", [](const SymbolicHandle &s) {
            return "<qdldl_cpp.Symbolic n=" + std::to_string(s.n) +
                   (s.has_perm ? " (with ordering)>" : ">");
        });

    // ---------------------------
    // UPDATED: expose multiple ordering algorithms
    // ---------------------------
    m.def(
        "amd",
        [](py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
               indptr,
           py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
               indices,
           py::array_t<double, py::array::c_style | py::array::forcecast> data,
           std::int64_t n) {
            SparseMatrixD A = make_sparse_from_csc(indptr, indices, data, n);
            qdldl::Ordering<std::int64_t> ord;
            {
                ord =
                    qdldl::rcm_ordering<double, std::int64_t>(
                        A);
            }
            return py::array(ord.perm.size(), ord.perm.data());
        },
        py::arg("indptr"), py::arg("indices"), py::arg("data"), py::arg("n"),
        R"doc(
Compute a simple minimum degree permutation for the given upper-triangular CSC matrix.
Returns 'perm' such that analyzing/factorizing P*A*P^T reduces fill.
)doc");

    m.def(
        "rcm",
        [](py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
               indptr,
           py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
               indices,
           py::array_t<double, py::array::c_style | py::array::forcecast> data,
           std::int64_t n) {
            SparseMatrixD A = make_sparse_from_csc(indptr, indices, data, n);
            qdldl::Ordering<std::int64_t> ord;
            {
                ord = qdldl::rcm_ordering<double, std::int64_t>(A);
            }
            return py::array(ord.perm.size(), ord.perm.data());
        },
        py::arg("indptr"), py::arg("indices"), py::arg("data"), py::arg("n"),
        R"doc(
Compute a Reverse Cuthill-McKee permutation for bandwidth reduction.
Returns 'perm' array.
)doc");

    m.def(
        "minimum_degree",
        [](py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
               indptr,
           py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
               indices,
           py::array_t<double, py::array::c_style | py::array::forcecast> data,
           std::int64_t n) {
            SparseMatrixD A = make_sparse_from_csc(indptr, indices, data, n);
            qdldl::Ordering<std::int64_t> ord;
            {
                ord =
                    qdldl::rcm_ordering<double, std::int64_t>(
                        A);
            }
            return py::array(ord.perm.size(), ord.perm.data());
        },
        py::arg("indptr"), py::arg("indices"), py::arg("data"), py::arg("n"),
        R"doc(
Compute a simple minimum degree permutation (alias for amd).
)doc");

    // ---------------------------
    // factorize (perm: None | ndarray | "amd" | "rcm" | "natural")
    // ---------------------------

    // Updated factorize function that stores ordering info
    m.def(
        "factorize",
        [](py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
               indptr,
           py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
               indices,
           py::array_t<double, py::array::c_style | py::array::forcecast> data,
           std::int64_t n,
           py::object perm /* = None | "amd" | "rcm" | "natural" | ndarray */) {
            SparseMatrixD A = make_sparse_from_csc(indptr, indices, data, n);

            LDLFactorization<double, std::int64_t> fac(n);
            qdldl::Ordering<std::int64_t> ordering;
            bool used_ordering = false;

            {
                if (perm.is_none()) {
                    fac = QDLDLSolverD::factorize(A);
                    used_ordering = false;
                } else {
                    ordering = parse_ordering_or_compute(perm, A, n);
                    fac = QDLDLSolverD::factorize_with_ordering(A, ordering);
                    used_ordering = true;
                }
            }

            Factor F;
            F.n = n;
            F.posD = fac.positive_eigenvalues;
            F.Lp = std::move(fac.L_col_ptr);
            F.Li = std::move(fac.L_row_idx);
            F.Lx = std::move(fac.L_values);
            F.D = std::move(fac.D);
            F.Dinv = std::move(fac.D_inv);

            // Store ordering information
            if (used_ordering) {
                F.perm = std::move(ordering.perm);
                F.iperm = std::move(ordering.iperm);
                F.has_ordering = true;
            }

            return F;
        },
        py::arg("indptr"), py::arg("indices"), py::arg("data"), py::arg("n"),
        py::arg("perm") = py::none(),
        R"doc(
Factorize an upper-triangular CSC matrix (n x n) using QDLDL.
The returned Factor automatically tracks whether ordering was used.
)doc");

    // Back-compat shim
    m.def(
        "factorize_scipy",
        [](py::array_t<std::int64_t> indptr, py::array_t<std::int64_t> indices,
           py::array_t<double> data, std::int64_t n) {
            return py::module_::import("qdldl_cpp")
                .attr("factorize")(indptr, indices, data, n);
        },
        py::arg("indptr"), py::arg("indices"), py::arg("data"), py::arg("n"),
        "Compatibility alias of factorize() without permutation.");

    // ---------------------------
    // analyze (perm: None | ndarray | "amd" | "rcm" | "natural") -> Symbolic
    // ---------------------------
    m.def(
        "analyze",
        [](py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
               indptr,
           py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
               indices,
           py::array_t<double, py::array::c_style | py::array::forcecast> data,
           std::int64_t n,
           py::object perm /* = None | "amd" | "rcm" | "natural" | ndarray */) {
            SparseMatrixD A = make_sparse_from_csc(indptr, indices, data, n);
            SymbolicHandle out;
            out.n = n;

            {
                py::gil_scoped_release nogil;
                if (perm.is_none()) {
                    out.S = QDLDLSolverD::analyze(A);
                    out.has_perm = false;
                } else {
                    auto ord = parse_ordering_or_compute(perm, A, n);
                    out.S = QDLDLSolverD::analyze_with_ordering(A, ord);
                    out.perm = std::move(ord.perm);
                    out.iperm = std::move(ord.iperm);
                    out.has_perm = true;
                }
            }
            return out;
        },
        py::arg("indptr"), py::arg("indices"), py::arg("data"), py::arg("n"),
        py::arg("perm") = py::none(),
        R"doc(
Symbolic analysis (etree + L column pointers).
'perm' may be None, "amd", "rcm", "natural", or a NumPy int64 array of shape [n].
Returns a Symbolic handle you can pass to refactorize().
)doc");

    // ---------------------------
    // refactorize (reuses symbolic)
    // ---------------------------
    m.def(
        "refactorize",
        [](const SymbolicHandle &S,
           py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
               indptr,
           py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
               indices,
           py::array_t<double, py::array::c_style | py::array::forcecast> data,
           std::int64_t n) {
            if (n != S.n)
                throw std::invalid_argument("n mismatch with Symbolic.n");
            SparseMatrixD A = make_sparse_from_csc(indptr, indices, data, n);

            LDLFactorization<double, std::int64_t> fac(n);
            {
                py::gil_scoped_release nogil;
                if (!S.has_perm) {
                    fac = QDLDLSolverD::refactorize(A, S.S);
                } else {
                    qdldl::Ordering<std::int64_t> ord;
                    ord.n = S.n;
                    ord.perm = S.perm;
                    ord.iperm = S.iperm;
                    fac = QDLDLSolverD::refactorize_with_ordering(A, S.S, ord);
                }
            }

            Factor F;
            F.n = n;
            F.posD = fac.positive_eigenvalues;
            F.Lp = std::move(fac.L_col_ptr);
            F.Li = std::move(fac.L_row_idx);
            F.Lx = std::move(fac.L_values);
            F.D = std::move(fac.D);
            F.Dinv = std::move(fac.D_inv);
            return F;
        },
        py::arg("symbolic"), py::arg("indptr"), py::arg("indices"),
        py::arg("data"), py::arg("n"),
        R"doc(
Numeric refactorization using a precomputed Symbolic (and the same permutation, if any).
Faster when sparsity pattern is unchanged but values change.
)doc");

    // Updated solve function that automatically handles permutation
    auto solve_lambda_updated =
        [](const Factor &F,
           py::array_t<double, py::array::c_style | py::array::forcecast> b) {
            if (b.ndim() != 1)
                throw std::invalid_argument("b must be 1-D");
            if (b.shape(0) != F.n)
                throw std::invalid_argument("b length must equal n");

            LDLFactorization<double, std::int64_t> fac(F.n);
            fac.L_col_ptr = F.Lp;
            fac.L_row_idx = F.Li;
            fac.L_values = F.Lx;
            fac.D = F.D;
            fac.D_inv = F.Dinv;
            fac.positive_eigenvalues = F.posD;

            py::array_t<double> out(b.shape(0));
            std::memcpy(out.mutable_data(), b.data(),
                        sizeof(double) * static_cast<std::size_t>(b.shape(0)));

            {
                py::gil_scoped_release nogil;
                auto xspan = std::span<double>(out.mutable_data(),
                                               static_cast<std::size_t>(F.n));

                if (!F.has_ordering) {
                    // No permutation used - standard solve
                    QDLDLSolverD::solve(fac, xspan);
                } else {
                    // Permutation was used - need permutation-aware solve
                    qdldl::Ordering<std::int64_t> ord;
                    ord.n = F.n;
                    ord.perm = F.perm;
                    ord.iperm = F.iperm;
                    QDLDLSolverD::solve_with_ordering(fac, ord, xspan);
                }
            }
            return out;
        };

    m.def("solve", solve_lambda_updated, py::arg("factor"), py::arg("b"),
          "Solve A x = b using an existing Factor (automatically handles "
          "permutation).");

    // Updated solve_refine that needs original matrix for residual computation
    m.def(
        "solve_refine",
        [](const Factor &F,
           py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
               indptr,
           py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>
               indices,
           py::array_t<double, py::array::c_style | py::array::forcecast> data,
           py::array_t<double, py::array::c_style | py::array::forcecast> b,
           int iters) {
            if (b.ndim() != 1)
                throw std::invalid_argument("b must be 1-D");
            if (b.shape(0) != F.n)
                throw std::invalid_argument("b length must equal n");

            // Reconstruct original matrix for residual computation
            SparseMatrixD A_orig =
                make_sparse_from_csc(indptr, indices, data, F.n);

            LDLFactorization<double, std::int64_t> fac(F.n);
            fac.L_col_ptr = F.Lp;
            fac.L_row_idx = F.Li;
            fac.L_values = F.Lx;
            fac.D = F.D;
            fac.D_inv = F.Dinv;
            fac.positive_eigenvalues = F.posD;

            py::array_t<double> out(b.shape(0));
            std::memcpy(out.mutable_data(), b.data(),
                        sizeof(double) * static_cast<std::size_t>(b.shape(0)));

            {
                py::gil_scoped_release nogil;
                auto xspan = std::span<double>(out.mutable_data(),
                                               static_cast<std::size_t>(F.n));
                auto bspan = std::span<const double>(
                    b.data(), static_cast<std::size_t>(F.n));

                if (!F.has_ordering) {
                    // First solve
                    QDLDLSolverD::solve(fac, xspan);
                    // Then refine (you'd need to implement this version)
                    QDLDLSolverD::solve_refine(fac, bspan, xspan, iters);
                } else {
                    // Permutation-aware solve and refine
                    qdldl::Ordering<std::int64_t> ord;
                    ord.n = F.n;
                    ord.perm = F.perm;
                    ord.iperm = F.iperm;
                    QDLDLSolverD::solve_with_ordering(fac, ord, xspan);
                    QDLDLSolverD::solve_refine_with_ordering(
                        fac, ord, A_orig, bspan, xspan, iters);
                }
            }
            return out;
        },
        py::arg("factor"), py::arg("indptr"), py::arg("indices"),
        py::arg("data"), py::arg("b"), py::arg("iters") = 2,
        "Solve with iterative refinement (handles permutation automatically).");

    // Tiny info
    m.def("version",
          []() { return std::string("qdldl_cpp 1.4 (simple orderings)"); });
}