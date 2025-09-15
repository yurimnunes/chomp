// qdldl_bindings.cpp — Python bindings with fast cached-permutation path
#include <numeric>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../include/qdldl.h"

namespace py = pybind11;
using namespace qdldl23;

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

    // external ordering (optional)
    std::vector<std::int64_t> perm;
    std::vector<std::int64_t> iperm;
    bool has_ordering = false;

    std::size_t nnzL() const noexcept { return Lx.size(); }
};

struct SymbolicHandle {
    Symb64 S; // etree, Lp, Lnz
    std::vector<std::int64_t> perm;
    std::vector<std::int64_t> iperm;
    std::int64_t n = 0;
    bool has_perm = false;

    // Cached permuted structural CSC and A→B mapping
    std::vector<std::int64_t> Bp;   // size n+1
    std::vector<std::int64_t> Bi;   // size nnzB
    std::vector<std::int64_t> A2B;  // size nnzA (maps A entry order to B slot)

    // Etree orders (optional but handy)
    std::vector<std::int64_t> preorder, postorder, first_child, next_sib;
};

// Build Sparse from raw CSC (upper+diag expected; validated by ctor)
static inline SparseD64 make_sparse_from_csc(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> indptr,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> indices,
    py::array_t<double,        py::array::c_style | py::array::forcecast> data,
    std::int64_t n) {

    if (indptr.ndim() != 1 || indices.ndim() != 1 || data.ndim() != 1)
        throw std::invalid_argument("indptr, indices, data must be 1-D arrays");
    if (indptr.shape(0) != n + 1)
        throw std::invalid_argument("indptr length must be n+1");
    if (indices.shape(0) != data.shape(0))
        throw std::invalid_argument("indices and data must have the same length");

    const auto nnz = static_cast<std::size_t>(indices.shape(0));
    std::vector<std::int64_t> Ap(static_cast<std::size_t>(n + 1));
    std::vector<std::int64_t> Ai(nnz);
    std::vector<double>       Ax(nnz);
    std::memcpy(Ap.data(), indptr.data(), sizeof(std::int64_t) * (n + 1));
    std::memcpy(Ai.data(), indices.data(), sizeof(std::int64_t) * nnz);
    std::memcpy(Ax.data(), data.data(),    sizeof(double)       * nnz);

    return SparseD64(static_cast<std::int64_t>(n), std::move(Ap), std::move(Ai), std::move(Ax));
}

// Build Ordering from a user-supplied permutation vector (no built-ins)
static inline Ordering<std::int64_t>
make_ordering_from_perm(const std::vector<std::int64_t>& p) {
    const std::int64_t n = static_cast<std::int64_t>(p.size());
    Ordering<std::int64_t> ord;
    ord.n = n;
    ord.perm = p;
    ord.iperm.resize(static_cast<std::size_t>(n));
    // validate and build inverse
    std::vector<char> seen(static_cast<std::size_t>(n), 0);
    for (std::int64_t i = 0; i < n; ++i) {
        auto pi = p[static_cast<std::size_t>(i)];
        if (pi < 0 || pi >= n) throw std::invalid_argument("perm contains out-of-range index");
        if (seen[static_cast<std::size_t>(pi)]) throw std::invalid_argument("perm is not a permutation (duplicates)");
        seen[static_cast<std::size_t>(pi)] = 1;
        ord.iperm[static_cast<std::size_t>(pi)] = i;
    }
    return ord;
}

// Parse optional external permutation (None -> no ordering)
static inline bool parse_external_perm(py::object perm_obj,
                                       std::int64_t n,
                                       Ordering<std::int64_t>* out_ord) {
    if (perm_obj.is_none()) return false;
    auto arr = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>(perm_obj);
    if (arr.ndim() != 1) throw std::invalid_argument("perm must be a 1-D int64 array or None");
    if (arr.shape(0) != n) throw std::invalid_argument("perm length must equal n");
    std::vector<std::int64_t> p(static_cast<std::size_t>(n));
    std::memcpy(p.data(), arr.data(), sizeof(std::int64_t) * static_cast<std::size_t>(n));
    *out_ord = make_ordering_from_perm(p);
    return true;
}

// Convert Factor -> LDL64
static inline LDL64 to_ldl64(const Factor &F) {
    LDL64 fac;
    fac.n       = static_cast<std::int64_t>(F.n);
    fac.num_pos = static_cast<std::int64_t>(F.posD);
    fac.Lp      = F.Lp;
    fac.Li      = F.Li;
    fac.Lx      = F.Lx;
    fac.D       = F.D;
    fac.Dinv    = F.Dinv;
    return fac;
}

// ---- etree orders (pre/post/child/sibling) for convenience (optional) ----
template <std::signed_integral IntT=int32_t>
static inline void etree_orders(const std::vector<IntT>& parent,
                                std::vector<IntT>& preorder,
                                std::vector<IntT>& postorder,
                                std::vector<IntT>& first_child,
                                std::vector<IntT>& next_sib) {
    const IntT n = (IntT)parent.size();
    first_child.assign((size_t)n, (IntT)-1);
    next_sib.assign((size_t)n, (IntT)-1);
    for (IntT v=0; v<n; ++v) {
        IntT p = parent[(size_t)v];
        if (p>=0) {
            next_sib[(size_t)v] = first_child[(size_t)p];
            first_child[(size_t)p] = v;
        }
    }
    preorder.clear(); preorder.reserve((size_t)n);
    postorder.clear(); postorder.reserve((size_t)n);
    std::vector<char> entered((size_t)n, 0);

    for (IntT r=0; r<n; ++r) if (parent[(size_t)r]==-1) {
        std::vector<IntT> st; st.push_back(r);
        while (!st.empty()) {
            IntT u = st.back();
            if (!entered[(size_t)u]) {
                entered[(size_t)u]=1; preorder.push_back(u);
                // push children in reverse sibling order to visit first_child first
                std::vector<IntT> kids;
                for (IntT c=first_child[(size_t)u]; c!=-1; c=next_sib[(size_t)c]) kids.push_back(c);
                for (auto it=kids.begin(); it!=kids.end(); ++it) st.push_back(*it);
            } else {
                st.pop_back(); postorder.push_back(u);
            }
        }
    }
}

// ---- Build cached permuted structure Bp/Bi and A→B value map in one pass ----
static inline void build_permuted_structure_and_map(
    const SparseD64& A, const Ordering<std::int64_t>* ord,
    std::vector<std::int64_t>& Bp,
    std::vector<std::int64_t>& Bi,
    std::vector<std::int64_t>& A2B)
{
    const auto n = A.n;

    // count per permuted column
    std::vector<std::int64_t> cnt((size_t)n, 0);
    {
        for (std::int64_t j=0;j<n;++j) {
            const auto pj = ord ? ord->perm[(size_t)j] : j;
            for (auto p=A.Ap[(size_t)j]; p<A.Ap[(size_t)j+1]; ++p) {
                const auto i  = A.Ai[(size_t)p];
                const auto pi = ord ? ord->perm[(size_t)i] : i;
                const auto col = (pi>pj)?pi:pj;
                ++cnt[(size_t)col];
            }
        }
    }
    Bp.assign((size_t)n+1, 0);
    for (std::int64_t j=0;j<n;++j) Bp[(size_t)j+1]=Bp[(size_t)j]+cnt[(size_t)j];
    Bi.resize((size_t)Bp[(size_t)n]);
    A2B.resize((size_t)A.nnz());

    // bucket per target column (row, src_idx) and then sort rows
    std::vector<std::vector<std::pair<std::int64_t, std::size_t>>> buckets((size_t)n);
    buckets.shrink_to_fit();
    std::size_t src_idx = 0;
    for (std::int64_t j=0;j<n;++j) {
        const auto pj = ord ? ord->perm[(size_t)j] : j;
        for (auto p=A.Ap[(size_t)j]; p<A.Ap[(size_t)j+1]; ++p, ++src_idx) {
            auto i = A.Ai[(size_t)p];
            auto pi = ord ? ord->perm[(size_t)i] : i;
            auto col = pj, row = pi;
            if (row>col) std::swap(row,col);
            buckets[(size_t)col].emplace_back(row, src_idx);
        }
    }
    // write sorted
    for (std::int64_t j=0;j<n;++j) {
        auto& b = buckets[(size_t)j];
        std::sort(b.begin(), b.end(),
                  [](auto& a, auto& b){ return a.first < b.first; });
        auto w = Bp[(size_t)j];
        for (auto& [row, src] : b) {
            Bi[(size_t)w]   = row;
            A2B[(size_t)src]= w;
            ++w;
        }
        // enforce diagonal existence check: Bi is sorted, must see j somewhere
        if (!std::binary_search(b.begin(), b.end(), std::pair<std::int64_t,std::size_t>{j,0},
                                [](auto& a, auto& b){ return a.first < b.first; })) {
            throw InvalidMatrixError("Missing diagonal after permutation at column " + std::to_string(j));
        }
    }
}

// ---- Fill Bx using A2B map (O(nnz)) ----
static inline void fill_Bx_from_map(
    const py::array_t<double>& dataA,
    const std::vector<std::int64_t>& A2B,
    std::vector<double>& Bx)
{
    const std::size_t nnzA = (std::size_t)A2B.size();
    Bx.assign(Bx.size(), 0.0);
    const double* Ax = dataA.data();
    for (std::size_t s=0; s<nnzA; ++s) {
        const auto dst = (std::size_t)A2B[s];
        Bx[dst] += Ax[s]; // coalesce duplicates if any (should be none if upper-only)
    }
}

// ---------------------------
// Module
// ---------------------------

PYBIND11_MODULE(qdldl_cpp, m) {
    m.doc() = "QDLDL-style LDL^T (header-only, C++23) Python bindings with external-only permutation and cached fast refactorization";

    // Exceptions
    py::register_exception<InvalidMatrixError>(m, "InvalidMatrixError");
    py::register_exception<FactorizationError>(m, "FactorizationError");

    // Factor holder
    py::class_<Factor>(m, "Factor")
        .def_property_readonly("n",        [](const Factor &f) { return f.n; })
        .def_property_readonly("posD",     [](const Factor &f) { return f.posD; })
        .def_property_readonly("nnzL",     &Factor::nnzL)
        .def_property_readonly("has_ordering", [](const Factor &f) { return f.has_ordering; })
        .def_property_readonly("perm",     [](const Factor &f) { return py::array(f.perm.size(),  f.perm.data()); })
        .def_property_readonly("iperm",    [](const Factor &f) { return py::array(f.iperm.size(), f.iperm.data()); })
        .def_property_readonly("L_indptr", [](const Factor &f) { return py::array(f.Lp.size(),    f.Lp.data()); })
        .def_property_readonly("L_indices",[](const Factor &f) { return py::array(f.Li.size(),    f.Li.data()); })
        .def_property_readonly("L_data",   [](const Factor &f) { return py::array(f.Lx.size(),    f.Lx.data()); })
        .def_property_readonly("D",        [](const Factor &f) { return py::array(f.D.size(),     f.D.data()); })
        .def_property_readonly("Dinv",     [](const Factor &f) { return py::array(f.Dinv.size(),  f.Dinv.data()); })
        .def("__repr__", [](const Factor &f) {
            return "<qdldl_cpp.Factor n=" + std::to_string(f.n) +
                   " nnzL=" + std::to_string(f.nnzL()) +
                   " posD=" + std::to_string(f.posD) +
                   (f.has_ordering ? " (permuted)>" : ">");
        });

    // Symbolic holder
    py::class_<SymbolicHandle>(m, "Symbolic")
        .def_property_readonly("n",        [](const SymbolicHandle &s) { return s.n; })
        .def_property_readonly("has_perm", [](const SymbolicHandle &s) { return s.has_perm; })
        .def_property_readonly("perm",     [](const SymbolicHandle &s) { return py::array(s.perm.size(),  s.perm.data()); })
        .def_property_readonly("iperm",    [](const SymbolicHandle &s) { return py::array(s.iperm.size(), s.iperm.data()); })
        .def_property_readonly("L_indptr", [](const SymbolicHandle &s) { return py::array(s.S.Lp.size(),   s.S.Lp.data()); })
        .def_property_readonly("Lnz",      [](const SymbolicHandle &s) { return py::array(s.S.Lnz.size(),  s.S.Lnz.data()); })
        .def_property_readonly("Bp",       [](const SymbolicHandle &s) { return py::array(s.Bp.size(),     s.Bp.data()); })
        .def_property_readonly("Bi",       [](const SymbolicHandle &s) { return py::array(s.Bi.size(),     s.Bi.data()); })
        .def_property_readonly("A2B",      [](const SymbolicHandle &s) { return py::array(s.A2B.size(),    s.A2B.data()); })
        .def("__repr__", [](const SymbolicHandle &s) {
            return "<qdldl_cpp.Symbolic n=" + std::to_string(s.n) +
                   (s.has_perm ? " (with external perm; cached structure)>" : " (cached structure)>");
        });

    // ---------------------------
    // analyze (symbolic) — builds cached permuted structure + A→B map
    // ---------------------------
    m.def(
        "analyze",
        [](py::array_t<std::int64_t> indptr,
           py::array_t<std::int64_t> indices,
           py::array_t<double>        data,
           std::int64_t               n,
           py::object                 perm /* None | np.array[int64] */) {
            // normalize A (upper+diag) once
            auto A = make_sparse_from_csc(indptr, indices, data, n);

            SymbolicHandle out;
            out.n = n;

            Ordering<std::int64_t> ord;
            const bool use_perm = parse_external_perm(perm, n, &ord);

            // Build cached structure + map (B is structure only here; values ignored)
            build_permuted_structure_and_map(A, use_perm ? &ord : nullptr,
                                             out.Bp, out.Bi, out.A2B);

            // Analyze on the cached structure by creating a temporary Sparse with zeros
            {
                std::vector<double> zeros(out.Bi.size(), 0.0);
                SparseD64 B(n, out.Bp, out.Bi, zeros);
                out.S = analyze_fast(B);
            }

            // etree orders (optional)
            etree_orders(out.S.etree, out.preorder, out.postorder, out.first_child, out.next_sib);

            out.has_perm = use_perm;
            if (use_perm) {
                out.perm  = std::move(ord.perm);
                out.iperm = std::move(ord.iperm);
            }
            return out;
        },
        py::arg("indptr"), py::arg("indices"), py::arg("data"), py::arg("n"),
        py::arg("perm") = py::none(),
        R"doc(Symbolic analysis with cached permuted structure and A→B map. If `perm` is provided, it is used; otherwise identity.)doc"
    );

    // ---------------------------
    // factorize — external permutation (one-shot, no cache reuse)
    // ---------------------------
    m.def(
        "factorize",
        [](py::array_t<std::int64_t> indptr,
           py::array_t<std::int64_t> indices,
           py::array_t<double>        data,
           std::int64_t               n,
           py::object                 perm /* None | np.array[int64] */) {
            auto A = make_sparse_from_csc(indptr, indices, data, n);

            Ordering<std::int64_t> ord;
            const bool use_perm = parse_external_perm(perm, n, &ord);

            LDL64 fac = (!use_perm) ? factorize(A)
                                    : factorize_with_ordering(A, ord);

            Factor F;
            F.n    = n;
            F.posD = fac.num_pos;
            F.Lp   = std::move(fac.Lp);
            F.Li   = std::move(fac.Li);
            F.Lx   = std::move(fac.Lx);
            F.D    = std::move(fac.D);
            F.Dinv = std::move(fac.Dinv);
            if (use_perm) {
                F.perm = std::move(ord.perm);
                F.iperm= std::move(ord.iperm);
                F.has_ordering = true;
            }
            return F;
        },
        py::arg("indptr"), py::arg("indices"), py::arg("data"), py::arg("n"),
        py::arg("perm") = py::none(),
        R"doc(Numeric LDL^T factorization. If `perm` is provided, it is used; otherwise no permutation.)doc"
    );

    // Compatibility alias (kept)
    m.def("factorize_scipy",
          [](py::array_t<std::int64_t> indptr,
             py::array_t<std::int64_t> indices,
             py::array_t<double>        data,
             std::int64_t               n) {
              return py::module_::import("qdldl_cpp")
                         .attr("factorize")(indptr, indices, data, n);
          });

    // ---------------------------
    // refactorize — reuse cached structure (Bp/Bi/A2B) for O(nnz) scatter
    // ---------------------------
    m.def(
        "refactorize",
        [](const SymbolicHandle &S,
           py::array_t<std::int64_t> indptr,
           py::array_t<std::int64_t> indices,
           py::array_t<double>        data,
           std::int64_t               n) {
            if (n != S.n)
                throw std::invalid_argument("n mismatch with Symbolic.n");

            // Incoming A is re-validated (upper+diag); we only reuse its values
            auto A = make_sparse_from_csc(indptr, indices, data, n);

            // Build B values quickly using cached map/structure
            std::vector<double> Bx(S.Bi.size(), 0.0);
            fill_Bx_from_map(data, S.A2B, Bx);

            SparseD64 B;
            B.n  = n;
            B.Ap = S.Bp; // reuse cached
            B.Ai = S.Bi; // reuse cached
            B.Ax = std::move(Bx);

            // Numeric factorization on the cached structure
            LDL64 fac = refactorize(B, S.S);

            Factor F;
            F.n    = n;
            F.posD = fac.num_pos;
            F.Lp   = std::move(fac.Lp);
            F.Li   = std::move(fac.Li);
            F.Lx   = std::move(fac.Lx);
            F.D    = std::move(fac.D);
            F.Dinv = std::move(fac.Dinv);
            if (S.has_perm) {
                F.perm = S.perm;
                F.iperm= S.iperm;
                F.has_ordering = true;
            }
            return F;
        },
        py::arg("symbolic"), py::arg("indptr"), py::arg("indices"),
        py::arg("data"), py::arg("n"),
        R"doc(Numeric refactorization using cached permuted structure (Bp/Bi) and A→B map. O(nnz) scatter; no re-permutation.)doc"
    );

    // ---------------------------
    // solve — applies stored external permutation if present
    // ---------------------------
    m.def(
        "solve",
        [](const Factor &F,
           py::array_t<double, py::array::c_style | py::array::forcecast> b) {
            if (b.ndim() != 1)
                throw std::invalid_argument("b must be 1-D");
            if (b.shape(0) != F.n)
                throw std::invalid_argument("b length must equal n");

            auto out = py::array_t<double>(b.shape(0));
            std::memcpy(out.mutable_data(), b.data(),
                        sizeof(double) * static_cast<std::size_t>(b.shape(0)));

            auto fac = to_ldl64(F);
            double *x = out.mutable_data();

            if (!F.has_ordering) {
                solve(fac, x);
            } else {
                Ordering<std::int64_t> ord;
                ord.n    = F.n;
                ord.perm = F.perm;
                ord.iperm= F.iperm;
                solve_with_ordering(fac, ord, x);
            }
            return out;
        },
        py::arg("factor"), py::arg("b"),
        "Solve A x = b using an existing Factor (uses external permutation if present)."
    );

    // ---------------------------
    // solve_refine — residual-based refinement against original A
    // ---------------------------
    m.def(
        "solve_refine",
        [](const Factor &F,
           py::array_t<std::int64_t> indptr,
           py::array_t<std::int64_t> indices,
           py::array_t<double>        data,
           py::array_t<double>        b,
           int iters) {
            if (b.ndim() != 1)
                throw std::invalid_argument("b must be 1-D");
            if (b.shape(0) != F.n)
                throw std::invalid_argument("b length must equal n");

            auto A = make_sparse_from_csc(indptr, indices, data, F.n);
            auto fac = to_ldl64(F);

            py::array_t<double> out(b.shape(0));
            std::memcpy(out.mutable_data(), b.data(),
                        sizeof(double) * static_cast<std::size_t>(b.shape(0)));
            double *x = out.mutable_data();

            if (!F.has_ordering) {
                solve(fac, x);
                qdldl23::refine<double, std::int64_t>(
                    A, fac, x, b.data(), iters,
                    static_cast<const qdldl23::Ordering<std::int64_t>*>(nullptr));
            } else {
                Ordering<std::int64_t> ord;
                ord.n    = F.n;
                ord.perm = F.perm;
                ord.iperm= F.iperm;
                solve_with_ordering(fac, ord, x);
                qdldl23::refine<double, std::int64_t>(A, fac, x, b.data(), iters, &ord);
            }
            return out;
        },
        py::arg("factor"), py::arg("indptr"), py::arg("indices"),
        py::arg("data"), py::arg("b"), py::arg("iters") = 2,
        "Solve with iterative refinement against the original A (uses external permutation if present)."
    );

    m.def("version", []() {
        return std::string("qdldl_cpp 3.0 — cached-permutation fast refactorization");
    });
}
