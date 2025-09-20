#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Eigen/SparseCore>
#include <Eigen/Core>
#include <memory>
#include "../include/kkt_core.h"

namespace py = pybind11;
using kkt::dvec; using kkt::dmat; using kkt::spmat;

// ---- helpers: build CSR from numpy buffers ----
static spmat csr_from_buffers(py::array_t<int,   py::array::c_style | py::array::forcecast> indptr,
                              py::array_t<int,   py::array::c_style | py::array::forcecast> indices,
                              py::array_t<double,py::array::c_style | py::array::forcecast> data,
                              int nrows, int ncols) {
    const int* ip = indptr.data();
    const int* ci = indices.data();
    const double* dd = data.data();
    const int nnz = static_cast<int>(data.size());

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(nnz);
    for (int r = 0; r < nrows; ++r) {
        for (int k = ip[r]; k < ip[r+1]; ++k) {
            trips.emplace_back(r, ci[k], dd[k]);
        }
    }
    spmat A(nrows, ncols);
    A.setFromTriplets(trips.begin(), trips.end());
    A.makeCompressed();
    return A;
}

// ---- A lightweight wrapper around the reusable C++ object we return to Python ----
struct ReusableWrapper {
    std::shared_ptr<kkt::KKTReusable> impl;
    int n{0}, m{0};
    ReusableWrapper(std::shared_ptr<kkt::KKTReusable> p, int n_, int m_) : impl(std::move(p)), n(n_), m(m_) {}
    std::pair<py::array_t<double>, py::array_t<double>>
    solve(py::array_t<double, py::array::c_style | py::array::forcecast> r1,
          py::object r2_opt,
          double cg_tol=1e-8, int cg_maxit=200)
    {
        if (!impl) throw std::runtime_error("reusable handle is null");
        if (r1.ndim() != 1 || static_cast<int>(r1.shape(0)) != n)
            throw std::runtime_error("r1 has wrong shape");
        dvec r1d(n);
        std::memcpy(r1d.data(), r1.data(), sizeof(double)*n);

        std::optional<dvec> r2d;
        if (!r2_opt.is_none()) {
            auto r2 = r2_opt.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
            if (r2.ndim() != 1 || static_cast<int>(r2.shape(0)) != m)
                throw std::runtime_error("r2 has wrong shape");
            r2d.emplace(m);
            std::memcpy(r2d->data(), r2.data(), sizeof(double)*m);
        }

        auto [dx, dy] = impl->solve(r1d, r2d, cg_tol, cg_maxit);

        py::array_t<double> dx_py(n); std::memcpy(dx_py.mutable_data(), dx.data(), sizeof(double)*n);
        py::array_t<double> dy_py(m); if (m>0) std::memcpy(dy_py.mutable_data(), dy.data(), sizeof(double)*m);
        return {dx_py, dy_py};
    }
};

// ---- The main binding entry that accepts CSR buffers (SciPy-compatible) ----
static py::tuple solve_kkt_csr(
    // W
    py::array_t<int,    py::array::c_style | py::array::forcecast> W_indptr,
    py::array_t<int,    py::array::c_style | py::array::forcecast> W_indices,
    py::array_t<double, py::array::c_style | py::array::forcecast> W_data,
    int n,
    // G (may be None from python; in that case pass empty arrays & m=0)
    py::object G_indptr_obj,
    py::object G_indices_obj,
    py::object G_data_obj,
    int mE,
    // right-hand sides
    py::array_t<double, py::array::c_style | py::array::forcecast> r1,
    py::object r2_opt,
    // options
    std::string method,
    double delta,
    py::object gamma_opt,
    bool assemble_schur_if_m_small,
    bool jacobi_schur_prec,
    double cg_tol,
    int cg_maxit
) {
    if (r1.ndim()!=1 || static_cast<int>(r1.shape(0))!=n) throw std::runtime_error("rhs_x has wrong shape");

    spmat W = csr_from_buffers(W_indptr, W_indices, W_data, n, n);

    std::optional<spmat> G;
    std::optional<dvec> r2;
    if (mE > 0) {
        auto G_indptr = G_indptr_obj.cast<py::array_t<int,    py::array::c_style | py::array::forcecast>>();
        auto G_indices= G_indices_obj.cast<py::array_t<int,    py::array::c_style | py::array::forcecast>>();
        auto G_data   = G_data_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        G = csr_from_buffers(G_indptr, G_indices, G_data, mE, n);

        if (!r2_opt.is_none()) {
            auto r2_py = r2_opt.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
            if (r2_py.ndim()!=1 || static_cast<int>(r2_py.shape(0))!=mE) throw std::runtime_error("rpE has wrong shape");
            r2.emplace(mE);
            std::memcpy(r2->data(), r2_py.data(), sizeof(double)*mE);
        } else {
            r2.emplace(dvec::Zero(mE));
        }
    }

    dvec r1d(n);
    std::memcpy(r1d.data(), r1.data(), sizeof(double)*n);

    kkt::SQPConfig cfg;
    cfg.cg_tol   = cg_tol;
    cfg.cg_maxit = cg_maxit;

    auto reg = std::optional<double>{}; // placeholder
    std::unordered_map<std::string, dvec> dummy_cache;

    // choose strategy
    auto strat = kkt::default_registry().get(
        (method=="hykkt") ? "hykkt" : "ldl"
    );

    std::optional<double> gamma = gamma_opt.is_none() ? std::nullopt
                                                      : std::optional<double>(gamma_opt.cast<double>());

    auto [dx, dy, handle] = strat->factor_and_solve(
        W, G, r1d, r2, cfg, reg, dummy_cache,
        delta, gamma, assemble_schur_if_m_small, jacobi_schur_prec
    );

    py::array_t<double> dx_py(n); std::memcpy(dx_py.mutable_data(), dx.data(), sizeof(double)*n);
    py::array_t<double> dy_py(mE); if (mE>0) std::memcpy(dy_py.mutable_data(), dy.data(), sizeof(double)*mE);

    py::object handle_obj = py::none();
    if (handle) {
        auto wrap = std::make_shared<ReusableWrapper>(handle, n, mE);
        handle_obj = py::cast(wrap);
    }
    return py::make_tuple(dx_py, dy_py, handle_obj);
}

PYBIND11_MODULE(kkt_core_cpp, m) {
    py::class_<ReusableWrapper, std::shared_ptr<ReusableWrapper>>(m, "KKTReusable")
        .def("solve", &ReusableWrapper::solve,
             py::arg("r1"), py::arg("r2") = py::none(),
             py::arg("cg_tol")=1e-8, py::arg("cg_maxit")=200);

    m.def("solve_kkt_csr", &solve_kkt_csr,
          py::arg("W_indptr"), py::arg("W_indices"), py::arg("W_data"), py::arg("n"),
          py::arg("G_indptr")=py::none(), py::arg("G_indices")=py::none(), py::arg("G_data")=py::none(), py::arg("mE")=0,
          py::arg("rhs_x"),
          py::arg("rpE")=py::none(),
          py::arg("method")="hykkt",
          py::arg("delta")=0.0,
          py::arg("gamma")=py::none(),
          py::arg("assemble_schur_if_m_small")=true,
          py::arg("jacobi_schur_prec")=true,
          py::arg("cg_tol")=1e-6,
          py::arg("cg_maxit")=200);
}
