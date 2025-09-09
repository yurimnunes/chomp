// amdqg_pybind.cpp
// Build: see CMakeLists.txt below

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "../include/amd.h"  // <-- put your C++ AMD/CSR code here (without main())

namespace py = pybind11;
using i32 = int32_t;

// ---- helpers ----

static inline std::vector<i32> to_vec_i32(const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &a) {
    if (a.ndim() != 1) throw std::runtime_error("Expected a 1-D int32 array");
    const i32* ptr = a.data();
    return std::vector<i32>(ptr, ptr + a.size());
}

static inline py::array_t<i32> to_numpy_i32(const std::vector<i32>& v) {
    auto out = py::array_t<i32>(v.size());
    std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(i32));
    return out;
}

// Factory that validates shapes lightly and builds a CSR
static CSR make_csr(
    i32 n,
    const py::array_t<i32, py::array::c_style | py::array::forcecast>& indptr,
    const py::array_t<i32, py::array::c_style | py::array::forcecast>& indices
) {
    if (indptr.ndim() != 1 || indices.ndim() != 1)
        throw std::runtime_error("indptr and indices must be 1-D int32 arrays");
    if (indptr.size() != static_cast<size_t>(n + 1))
        throw std::runtime_error("indptr length must be n+1");
    CSR A(n);
    A.indptr = to_vec_i32(indptr);
    A.indices = to_vec_i32(indices);
    return A;
}

PYBIND11_MODULE(amdqg, m) {
    m.doc() = "Array-based AMD with quotient-graph (C++17) — pybind11 bindings";

    // -------- CSR ----------
    py::class_<CSR>(m, "CSR")
        .def(py::init<>())
        .def_readwrite("n", &CSR::n)
        .def_readwrite("indptr", &CSR::indptr)
        .def_readwrite("indices", &CSR::indices)
        .def("nnz", &CSR::nnz)
        .def("strict_upper_union_transpose", [](const CSR& A){
            return A.strict_upper_union_transpose();
        })
        .def("to_numpy", [](const CSR& A){
            py::dict d;
            d["n"] = A.n;
            d["indptr"] = to_numpy_i32(A.indptr);
            d["indices"] = to_numpy_i32(A.indices);
            return d;
        });

    // Factory from NumPy arrays (easier when calling from SciPy)
    m.def("make_csr", &make_csr,
          py::arg("n"), py::arg("indptr"), py::arg("indices"),
          "Create CSR(n, indptr, indices) from int32 NumPy arrays");

    // -------- AMD ----------
    py::class_<AMDReorderingArray>(m, "AMDReorderingArray")
        .def(py::init<bool,int>(), py::arg("aggressive_absorption")=true, py::arg("dense_cutoff")=-1)
        .def("amd_order",
             [](AMDReorderingArray& self, const CSR& A, bool symmetrize){
                 return to_numpy_i32(self.amd_order(A, symmetrize));
             }, py::arg("A"), py::arg("symmetrize")=true,
             "Return AMD permutation as int32 NumPy array")
        .def("compute_fill_reducing_permutation",
             [](AMDReorderingArray& self, const CSR& A, bool symmetrize){
                 auto pr = self.compute_fill_reducing_permutation(A, symmetrize);
                 auto perm = to_numpy_i32(pr.first);
                 const AMDStats& st = pr.second;
                 py::dict d;
                 d["original_nnz"] = st.original_nnz;
                 d["original_bandwidth"] = st.original_bandwidth;
                 d["reordered_bandwidth"] = st.reordered_bandwidth;
                 d["bandwidth_reduction"] = st.bandwidth_reduction;
                 d["matrix_size"] = st.matrix_size;
                 d["inverse_permutation"] = to_numpy_i32(st.inverse_permutation);
                 d["absorbed_elements"] = st.absorbed_elements;
                 d["coalesced_variables"] = st.coalesced_variables;
                 d["iw_capacity_peak"] = st.iw_capacity_peak;
                 return py::make_tuple(perm, d);
             }, py::arg("A"), py::arg("symmetrize")=true,
             "Return (perm, stats) where perm is int32 array and stats is a dict");

    // Convenience: pass raw arrays from Python (e.g. SciPy CSR arrays)
    m.def("amd_order_from_csr_arrays",
          [](i32 n,
             const py::array_t<i32, py::array::c_style | py::array::forcecast>& indptr,
             const py::array_t<i32, py::array::c_style | py::array::forcecast>& indices,
             bool symmetrize,
             bool aggressive_absorption,
             int dense_cutoff){
              AMDReorderingArray amd(aggressive_absorption, dense_cutoff);
              auto A = make_csr(n, indptr, indices);
              auto p = amd.amd_order(A, symmetrize);
              return to_numpy_i32(p);
          },
          py::arg("n"), py::arg("indptr"), py::arg("indices"),
          py::arg("symmetrize")=true,
          py::arg("aggressive_absorption")=true,
          py::arg("dense_cutoff")=-1,
          "One-shot AMD on (n, indptr, indices)");
}
