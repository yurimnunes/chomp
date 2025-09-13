// amdqg_pybind.cpp
// Build: see CMakeLists.txt below

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>  // for std::memcpy

#include "../include/amd.h"  // <-- put your C++ AMD/CSR + Supernode code here (no main())

namespace py = pybind11;
using i32 = int32_t;

// ---- helpers ----

static inline std::vector<i32>
to_vec_i32(const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &a) {
    if (a.ndim() != 1) throw std::runtime_error("Expected a 1-D int32 array");
    const i32* ptr = a.data();
    return std::vector<i32>(ptr, ptr + a.size());
}

static inline py::array_t<i32> to_numpy_i32(const std::vector<i32>& v) {
    auto out = py::array_t<i32>(v.size());
    std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(i32));
    return out;
}

static inline py::array_t<i32> to_numpy_i32_from_ptr(const i32* data, size_t n) {
    auto out = py::array_t<i32>(n);
    std::memcpy(out.mutable_data(), data, n * sizeof(i32));
    return out;
}

static inline py::array_t<i32> ranges_to_numpy_2col(const std::vector<std::pair<i32,i32>>& ranges) {
    const size_t m = ranges.size();
    py::array_t<i32> arr({(py::ssize_t)m, (py::ssize_t)2});
    auto buf = arr.mutable_unchecked<2>();
    for (size_t i = 0; i < m; ++i) {
        buf(i,0) = ranges[i].first;
        buf(i,1) = ranges[i].second;
    }
    return arr;
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

// Convert SupernodeInfo -> Python dict
static py::dict supernodes_to_pydict(const SupernodeInfo& sn) {
    py::dict d;
    d["ranges"] = ranges_to_numpy_2col(sn.ranges); // shape [ns, 2], inclusive start/end
    d["col2sn"] = to_numpy_i32(sn.col2sn);         // shape [n]
    d["etree"]  = to_numpy_i32(sn.etree);          // shape [n], -1 is root
    d["post"]   = to_numpy_i32(sn.post);           // shape [n]
    return d;
}

PYBIND11_MODULE(amdqg, m) {
    m.doc() = "Array-based AMD with quotient-graph + supernode detection (C++17) — pybind11 bindings";

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
             "Return (perm, stats) where perm is int32 array and stats is a dict")
        // New: convenience that returns (perm, stats, supernodes_dict)
        .def("compute_perm_and_supernodes",
             [](AMDReorderingArray& self,
                const CSR& A,
                bool symmetrize,
                int relax,
                double tau,
                int max_size) {
                 auto pr = self.compute_fill_reducing_permutation(A, symmetrize);
                 auto perm_vec = pr.first;
                 const AMDStats& st = pr.second;
                 SupernodeInfo sn = identify_supernodes(A, perm_vec, relax, tau, max_size);
                 py::dict stats;
                 stats["original_nnz"] = st.original_nnz;
                 stats["original_bandwidth"] = st.original_bandwidth;
                 stats["reordered_bandwidth"] = st.reordered_bandwidth;
                 stats["bandwidth_reduction"] = st.bandwidth_reduction;
                 stats["matrix_size"] = st.matrix_size;
                 stats["inverse_permutation"] = to_numpy_i32(st.inverse_permutation);
                 stats["absorbed_elements"] = st.absorbed_elements;
                 stats["coalesced_variables"] = st.coalesced_variables;
                 stats["iw_capacity_peak"] = st.iw_capacity_peak;
                 return py::make_tuple(to_numpy_i32(perm_vec), stats, supernodes_to_pydict(sn));
             },
             py::arg("A"),
             py::arg("symmetrize") = true,
             py::arg("relax") = 0,
             py::arg("tau") = 1.0,
             py::arg("max_size") = std::numeric_limits<int>::max(),
             "Return (perm, stats, supernodes) with relaxed amalgamation options");

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

    // -------- Supernode identification (free functions) ----------
    // identify_supernodes(A: CSR, p: np.int32[1D], relax=0, tau=1.0, max_size=INT_MAX) -> dict
    m.def("identify_supernodes",
          [](const CSR& A,
             const py::array_t<i32, py::array::c_style | py::array::forcecast>& p,
             int relax,
             double tau,
             int max_size) {
              auto pvec = to_vec_i32(p);
              SupernodeInfo sn = identify_supernodes(A, pvec, relax, tau, max_size);
              return supernodes_to_pydict(sn);
          },
          py::arg("A"),
          py::arg("p"),
          py::arg("relax") = 0,
          py::arg("tau") = 1.0,
          py::arg("max_size") = std::numeric_limits<int>::max(),
          "Identify structural supernodes on the pattern of A[p,p].\n"
          "Returns dict with:\n"
          "  ranges: int32 array [ns,2] of inclusive column ranges in permuted space\n"
          "  col2sn: int32 array [n] mapping column->supernode id\n"
          "  etree : int32 array [n] elimination tree parents (-1 = root)\n"
          "  post  : int32 array [n] postorder of the etree");
}
