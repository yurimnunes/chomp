// model.h — Precompiled-from-Python AD model (hot path is pure C++)
#pragma once
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/SparseCore>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace py = pybind11;
using dvec = Eigen::VectorXd;
using dmat = Eigen::MatrixXd;
using spmat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

namespace fast {

// ---------- NumPy ↔ Eigen zero-copy helpers ---------------------------------
inline py::array eigen_vec_view(const Eigen::Ref<const dvec> &x) {
    return py::array(
        py::buffer_info(const_cast<double *>(x.data()), sizeof(double),
                        py::format_descriptor<double>::format(), 1,
                        {static_cast<size_t>(x.size())}, {sizeof(double)}),
        py::none());
}

inline Eigen::Map<dvec> map_vec_nocopy(const py::array &a) {
    auto arr = py::cast<
        py::array_t<double, py::array::c_style | py::array::forcecast>>(a);
    auto buf = arr.request();
    if (buf.ndim != 1)
        throw std::runtime_error("Expected 1-D float64 array.");
    return Eigen::Map<dvec>(static_cast<double *>(buf.ptr),
                            static_cast<Eigen::Index>(buf.shape[0]));
}

using RowMat =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
inline Eigen::Map<RowMat> map_mat_nocopy(const py::array &a, int rows = -1,
                                         int cols = -1) {
    auto arr = py::cast<
        py::array_t<double, py::array::c_style | py::array::forcecast>>(a);
    auto buf = arr.request();
    if (buf.ndim != 2)
        throw std::runtime_error("Expected 2-D float64 array.");
    const int r = static_cast<int>(buf.shape[0]);
    const int c = static_cast<int>(buf.shape[1]);
    if ((rows >= 0 && rows != r) || (cols >= 0 && cols != c))
        throw std::runtime_error("Matrix shape mismatch.");
    return Eigen::Map<RowMat>(static_cast<double *>(buf.ptr), r, c);
}

// ---------- SciPy CSR → Eigen CSR (and dense) --------------------------------
inline bool is_scipy_sparse(const py::object &obj) {
    return obj && !obj.is_none() && py::hasattr(obj, "tocsr");
}

inline spmat csr_to_eigen_csr(const py::object &csr_like, int rows = -1,
                              int cols = -1) {
    py::object csr = csr_like.attr("tocsr")();
    auto shape = csr.attr("shape");
    const int m =
        rows >= 0 ? rows : py::cast<int>(shape.attr("__getitem__")(0));
    const int n =
        cols >= 0 ? cols : py::cast<int>(shape.attr("__getitem__")(1));

    auto data = csr.attr("data")
                    .cast<py::array_t<double, py::array::c_style |
                                                  py::array::forcecast>>();
    auto indices =
        csr.attr("indices")
            .cast<
                py::array_t<int, py::array::c_style | py::array::forcecast>>();
    auto indptr =
        csr.attr("indptr")
            .cast<
                py::array_t<int, py::array::c_style | py::array::forcecast>>();

    using RowSp = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;
    RowSp rowCsr(m, n);

    std::vector<Eigen::Triplet<double, int>> trips;
    trips.reserve(static_cast<size_t>(data.size()));

    auto D = data.unchecked<1>();
    auto I = indices.unchecked<1>();
    auto P = indptr.unchecked<1>();

    for (int r = 0; r < m; ++r) {
        const int start = P(r);
        const int end = P(r + 1);
        for (int k = start; k < end; ++k) {
            const double v = D(k);
            if (std::isfinite(v))
                trips.emplace_back(r, I(k), v);
        }
    }
    rowCsr.setFromTriplets(trips.begin(), trips.end());
    rowCsr.makeCompressed();

    spmat colCsr = rowCsr; // convert storage
    colCsr.makeCompressed();
    return colCsr;
}

inline dmat to_dense_any(const py::object &obj, int rows = -1, int cols = -1) {
    if (is_scipy_sparse(obj)) {
        spmat S = csr_to_eigen_csr(obj, rows, cols);
        return dmat(S);
    }
    auto A = py::cast<py::array>(obj);
    auto Mrow = map_mat_nocopy(A, rows, cols);
    dmat M = Mrow;
    return M;
}

inline spmat to_csr_any(const py::object &obj, int rows = -1, int cols = -1) {
    if (is_scipy_sparse(obj))
        return csr_to_eigen_csr(obj, rows, cols);
    auto A = py::cast<py::array>(obj);
    auto Mrow = map_mat_nocopy(A, rows, cols);
    dmat M = Mrow;
    spmat S(M.sparseView());
    S.makeCompressed();
    return S;
}

// ---------- small sanitizers --------------------------------------------------
inline void ensure_finite_inplace(dvec &v) {
    for (int i = 0; i < v.size(); ++i)
        if (!std::isfinite(v[i]))
            v[i] = 0.0;
}

inline void ensure_finite_inplace(dmat &M, double clip = 1e12) {
    for (int i = 0; i < M.rows(); ++i)
        for (int j = 0; j < M.cols(); ++j) {
            double &x = M(i, j);
            if (!std::isfinite(x))
                x = 0.0;
            if (std::abs(x) > clip)
                x = (x > 0 ? clip : -clip);
        }
}

inline void symmetrize_inplace(dmat &H) { H = 0.5 * (H + H.transpose()); }

} // namespace fast

// ============================================================================
// Native AD functors (compiled once via Python `ad` module)
struct ADCompiled {
    std::function<double (const dvec&)> val;
    std::function<dvec   (const dvec&)> grad;
    std::function<dmat   (const dvec&)> hess;
};

// ============================================================================

class Model {
public:
    // New full-args ctor (Python callables → compiled native functors)
    Model(py::object f, std::optional<py::iterable> c_ineq,
          std::optional<py::iterable> c_eq, int n, std::optional<py::object> lb,
          std::optional<py::object> ub, bool use_sparse,
          std::optional<py::object> ad_module = std::nullopt)
        : n_(n), use_sparse_(use_sparse) {

        if (n_ <= 0)
            throw std::invalid_argument("n must be positive.");
        if (!py::isinstance<py::function>(f))
            throw std::invalid_argument("f must be a Python callable.");

        if (lb.has_value() && !lb->is_none())
            lb_ = fast::map_vec_nocopy(py::cast<py::array>(*lb)).eval();
        else
            lb_ = dvec::Constant(n_, -std::numeric_limits<double>::infinity());

        if (ub.has_value() && !ub->is_none())
            ub_ = fast::map_vec_nocopy(py::cast<py::array>(*ub)).eval();
        else
            ub_ = dvec::Constant(n_, std::numeric_limits<double>::infinity());

        if (lb_.size() != n_ || ub_.size() != n_)
            throw std::invalid_argument("lb/ub size must be n.");

        // Import `ad` and compile f/cI/cE into native functors
        py::object AD;
        {
            py::gil_scoped_acquire gil;
            AD = ad_module.has_value() ? *ad_module : py::module::import("ad");
        }
        compile_objective_(AD, f);

        if (c_ineq.has_value())
            for (auto &&obj : *c_ineq)
                cI_compiled_.push_back(compile_constraint_(AD, py::reinterpret_borrow<py::object>(obj)));
        if (c_eq.has_value())
            for (auto &&obj : *c_eq)
                cE_compiled_.push_back(compile_constraint_(AD, py::reinterpret_borrow<py::object>(obj)));

        mI_ = static_cast<int>(cI_compiled_.size());
        mE_ = static_cast<int>(cE_compiled_.size());
    }

    // Legacy ctor (keeps older call sites compiling)
    Model(py::object f, py::object c_ineq, py::object c_eq, int n,
          py::object lb, py::object ub)
        : Model(std::move(f),
                c_ineq.is_none()
                    ? std::optional<py::iterable>{}
                    : std::optional<py::iterable>{py::reinterpret_borrow<
                          py::iterable>(c_ineq)},
                c_eq.is_none()
                    ? std::optional<py::iterable>{}
                    : std::optional<py::iterable>{py::reinterpret_borrow<
                          py::iterable>(c_eq)},
                n,
                lb.is_none() ? std::optional<py::object>{}
                             : std::optional<py::object>{lb},
                ub.is_none() ? std::optional<py::object>{}
                             : std::optional<py::object>{ub},
                /*use_sparse=*/false,
                /*ad_module=*/std::nullopt) {}

    // --- meta ----------------------------------------------------------------
    int n() const noexcept { return n_; }
    int m_ineq() const noexcept { return mI_; }
    int m_eq() const noexcept { return mE_; }
    bool use_sparse() const noexcept { return use_sparse_; }

    // define method lb that returns the lower bounds as an Eigen vector
    const dvec &lb() const noexcept { return lb_; }
    // define method ub that returns the upper bounds as an Eigen vector
    const dvec &ub() const noexcept { return ub_; }

    // --- objective -----------------------------------------------------------
    double f_val(const Eigen::Ref<const dvec> &x) const {
        set_last_x_(x);
        double v = F_.val(x);
        return std::isfinite(v) ? v : std::numeric_limits<double>::infinity();
    }

    dvec grad(const Eigen::Ref<const dvec> &x) const {
        set_last_x_(x);
        dvec g = F_.grad(x);
        fast::ensure_finite_inplace(g);
        return g;
    }

    dmat hess(const Eigen::Ref<const dvec> &x) const {
        set_last_x_(x);
        dmat H = F_.hess(x);
        fast::symmetrize_inplace(H);
        fast::ensure_finite_inplace(H);
        return H;
    }

    spmat hess_csr(const Eigen::Ref<const dvec> &x) const {
        dmat Hd = hess(x);
        spmat S(Hd.sparseView());
        S.makeCompressed();
        return S;
    }

    // --- inequalities --------------------------------------------------------
    dvec cI(const Eigen::Ref<const dvec> &x) const {
        set_last_x_(x);
        dvec out(mI_);
        if (mI_ == 0) return out;
        for (int i = 0; i < mI_; ++i) {
            double v = cI_compiled_[i].val(x);
            out[i] = std::isfinite(v) ? v : 0.0;
        }
        return out;
    }

    dmat JI(const Eigen::Ref<const dvec> &x) const {
        set_last_x_(x);
        if (mI_ == 0) return dmat(mI_, n_);
        fast::RowMat Jrow(mI_, n_);
        for (int i = 0; i < mI_; ++i) {
            dvec gi = cI_compiled_[i].grad(x);
            if (gi.size() != n_) throw std::runtime_error("JI row size mismatch.");
            Jrow.row(i) = gi.transpose();
        }
        dmat J = Jrow;
        fast::ensure_finite_inplace(J);
        // cache
        ji_cache_ = J;
        have_ji_ = true;
        return J;
    }

    spmat JI_csr(const Eigen::Ref<const dvec> &x) const {
        dmat J = JI(x);
        spmat S(J.sparseView());
        S.makeCompressed();
        return S;
    }

    // --- equalities ----------------------------------------------------------
    dvec cE(const Eigen::Ref<const dvec> &x) const {
        set_last_x_(x);
        dvec out(mE_);
        if (mE_ == 0) return out;
        for (int j = 0; j < mE_; ++j) {
            double v = cE_compiled_[j].val(x);
            out[j] = std::isfinite(v) ? v : 0.0;
        }
        return out;
    }

    dmat JE(const Eigen::Ref<const dvec> &x) const {
        set_last_x_(x);
        if (mE_ == 0) return dmat(mE_, n_);
        fast::RowMat Jrow(mE_, n_);
        for (int j = 0; j < mE_; ++j) {
            dvec gj = cE_compiled_[j].grad(x);
            if (gj.size() != n_) throw std::runtime_error("JE row size mismatch.");
            Jrow.row(j) = gj.transpose();
        }
        dmat J = Jrow;
        fast::ensure_finite_inplace(J);
        // cache
        je_cache_ = J;
        have_je_ = true;
        return J;
    }

    spmat JE_csr(const Eigen::Ref<const dvec> &x) const {
        dmat J = JE(x);
        spmat S(J.sparseView());
        S.makeCompressed();
        return S;
    }

    // --- Lagrangian Hessian (dense) ------------------------------------------
    dmat lagrangian_hessian(const Eigen::Ref<const dvec> &x,
                            const Eigen::Ref<const dvec> &lam,
                            const Eigen::Ref<const dvec> &nu,
                            double clip = 1e12, double diag_floor = 0.0,
                            double multiplier_threshold = 1e-8) const {
        if (lam.size() != mI_ || nu.size() != mE_)
            throw std::runtime_error("lam/nu size mismatch.");
        dmat H = hess(x);

        for (int i = 0; i < mI_; ++i) {
            const double w = lam[i];
            if (std::abs(w) <= multiplier_threshold) continue;
            dmat Hi = cI_compiled_[i].hess(x);
            fast::symmetrize_inplace(Hi);
            H.noalias() += (w * Hi);
        }
        for (int j = 0; j < mE_; ++j) {
            const double w = nu[j];
            if (std::abs(w) <= multiplier_threshold) continue;
            dmat Hj = cE_compiled_[j].hess(x);
            fast::symmetrize_inplace(Hj);
            H.noalias() += (w * Hj);
        }
        if (diag_floor > 0.0) {
            for (int k = 0; k < n_; ++k) {
                double &d = H(k, k);
                if (!std::isfinite(d) || std::abs(d) < diag_floor)
                    d = diag_floor;
            }
        }
        fast::ensure_finite_inplace(H, clip);
        return H;
    }

    // --- constraint_violation ------------------------------------------------
    double constraint_violation(const Eigen::Ref<const dvec> &x) const {
        const int mI = mI_, mE = mE_;
        dvec cI_v, cE_v;
        if (mI) cI_v = cI(x);
        if (mE) cE_v = cE(x);

        const double scale =
            std::max<double>(1.0, std::max<double>(n_, mI + mE));
        double theta = 0.0;

        if (mI) theta += (cI_v.array().max(0.0)).sum() / scale;
        if (mE) theta += cE_v.array().abs().sum() / scale;

        if (!std::isfinite(theta))
            return std::numeric_limits<double>::infinity();
        return theta;
    }

    // --- kkt_residuals -------------------------------------------------------
    std::map<std::string, double>
    kkt_residuals(const Eigen::Ref<const dvec> &x,
                  const Eigen::Ref<const dvec> &lam,
                  const Eigen::Ref<const dvec> &nu) const {
        if (x.size() != n_ || lam.size() != mI_ || nu.size() != mE_)
            throw std::runtime_error("Incompatible shapes for kkt_residuals.");

        dvec g = grad(x);
        double scale_g = std::max(1.0, g.lpNorm<Eigen::Infinity>());

        dmat rL_add = dmat::Zero(n_, 1);
        if (mI_) {
            dmat JI_d = have_ji_ && last_x_.has_value() &&
                        (*last_x_ - x).lpNorm<Eigen::Infinity>() == 0.0
                        ? ji_cache_ : JI(x);
            dvec lam_pos = lam.cwiseMax(0.0);
            rL_add.col(0) += JI_d.transpose() * lam_pos;
        }
        if (mE_) {
            dmat JE_d = have_je_ && last_x_.has_value() &&
                        (*last_x_ - x).lpNorm<Eigen::Infinity>() == 0.0
                        ? je_cache_ : JE(x);
            rL_add.col(0) += JE_d.transpose() * nu;
        }

        dvec rL = g + rL_add;
        double stat_inf = rL.lpNorm<Eigen::Infinity>() / scale_g;

        double ineq_inf = 0.0, comp_inf = 0.0, eq_inf = 0.0;

        if (mI_) {
            dvec cI_v = cI(x);
            dvec cI_plus = cI_v.cwiseMax(0.0);
            ineq_inf = cI_plus.lpNorm<Eigen::Infinity>() / scale_g;

            dvec lam_pos = lam.cwiseMax(0.0);
            dvec comp = (lam_pos.array() * cI_v.array()).abs().matrix();
            comp_inf = (comp.size() ? comp.maxCoeff() : 0.0) / scale_g;
        }
        if (mE_) {
            dvec cE_v = cE(x);
            eq_inf = cE_v.lpNorm<Eigen::Infinity>() / scale_g;
        }

        std::map<std::string, double> res{{"stat", stat_inf},
                                          {"ineq", ineq_inf},
                                          {"eq", eq_inf},
                                          {"comp", comp_inf}};
        for (auto &kv : res)
            if (!std::isfinite(kv.second))
                kv.second = std::numeric_limits<double>::infinity();
        return res;
    }

    // --- reset_cache ---------------------------------------------------------
    void reset_cache() {
        have_je_ = false;
        have_ji_ = false;
        last_x_.reset();
        je_cache_.resize(0, 0);
        ji_cache_.resize(0, 0);
    }

    // --- compute_soc_step (uses cached JE/JI at last base point) -------------
    std::pair<dvec, dvec>
    compute_soc_step(std::optional<py::object> rE_py,
                     std::optional<py::object> rI_py, double /*mu*/,
                     double active_tol = 1e-6, double w_eq = 1.0,
                     double w_ineq = 1.0, double gamma = 1e-8) const {
        const int n = n_, mE = mE_, mI = mI_;

        // Ensure JE/JI caches exist for base x
        if (!have_je_ && mE > 0) {
            if (!last_x_)
                throw std::runtime_error(
                    "compute_soc_step: JE not cached and no base x available.");
            const_cast<Model *>(this)->JE(*last_x_);
        }
        if (!have_ji_ && mI > 0) {
            if (!last_x_)
                throw std::runtime_error(
                    "compute_soc_step: JI not cached and no base x available.");
            const_cast<Model *>(this)->JI(*last_x_);
        }

        // Optional residuals → vectors
        dvec rE = dvec::Zero(mE);
        dvec rI = dvec::Zero(mI);
        bool have_rE = false, have_rI = false;

        if (mE && rE_py.has_value() && !rE_py->is_none()) {
            rE = fast::map_vec_nocopy(py::cast<py::array>(*rE_py)).eval();
            have_rE = (rE.size() == mE);
        }
        if (mI && rI_py.has_value() && !rI_py->is_none()) {
            rI = fast::map_vec_nocopy(py::cast<py::array>(*rI_py)).eval();
            have_rI = (rI.size() == mI);
        }

        // Build normal equations: (AᵀA + γI) dx = -Aᵀb
        dmat AtA = dmat::Identity(n, n) * gamma;
        dvec ATb = dvec::Zero(n);

        if (mE && have_rE) {
            const dmat &JE_d = je_cache_;
            if (JE_d.size()) {
                const dmat Aeq = w_eq * JE_d;
                AtA.noalias() += Aeq.transpose() * Aeq;
                ATb.noalias() += Aeq.transpose() * (w_eq * rE);
            }
        }
        if (mI && have_rI) {
            const dmat &JI_d = ji_cache_;
            if (JI_d.size()) {
                // W selects rows: violations or |rI| >= active_tol
                dvec W = dvec::Zero(mI);
                for (int i = 0; i < mI; ++i)
                    if (rI[i] > 0.0 || std::abs(rI[i]) >= active_tol)
                        W[i] = 1.0;

                dmat WJ = W.asDiagonal() * JI_d; // row scaling
                dvec Wr = W.array() * rI.array();

                dmat Aineq = w_ineq * WJ;
                dvec bineq = w_ineq * Wr;

                AtA.noalias() += Aineq.transpose() * Aineq;
                ATb.noalias() += Aineq.transpose() * bineq;
            }
        }

        // Solve
        dvec dx;
        Eigen::LDLT<dmat> ldlt(AtA);
        if (ldlt.info() == Eigen::Success) {
            dx = -ldlt.solve(ATb);
        } else {
            dx = -AtA.colPivHouseholderQr().solve(ATb);
        }
        for (int i = 0; i < dx.size(); ++i)
            if (!std::isfinite(dx[i]))
                dx[i] = 0.0;

        // ds = -(rI + JI dx)
        dvec ds = dvec::Zero(mI);
        if (mI && have_rI && ji_cache_.size()) {
            ds = -(rI + ji_cache_ * dx);
            for (int i = 0; i < ds.size(); ++i)
                if (!std::isfinite(ds[i]))
                    ds[i] = 0.0;
        }
        return {dx, ds};
    }

private:
    int n_{0}, mI_{0}, mE_{0};
    bool use_sparse_{false};

    dvec lb_, ub_;

    // Compiled functors (native; no Python in hot path)
    ADCompiled F_;
    std::vector<ADCompiled> cI_compiled_, cE_compiled_;

    // Keep compiled Python-side objects (ValFn/GradFn/HessFn) alive.
    std::vector<py::object> keepalive_;

    // Tiny cache to support SOC & cheaper KKT assembly
    mutable std::optional<dvec> last_x_;
    mutable dmat je_cache_{0, 0}, ji_cache_{0, 0};
    mutable bool have_je_{false}, have_ji_{false};

    // --- Compilation helpers -------------------------------------------------
    void compile_objective_(const py::object& AD, const py::object& f) {
        py::object valfn, gradfn, hessfn;
        {
            py::gil_scoped_acquire gil;
            valfn  = AD.attr("sym_val")(f, n_, true);   // ValFn
            gradfn = AD.attr("sym_grad")(f, n_, true);  // GradFn
            hessfn = AD.attr("sym_hess")(f, n_, true);  // HessFn
        }
        F_.val  = [valfn, this](const dvec& x)->double {
            py::gil_scoped_acquire gil;
            return py::cast<double>(valfn.attr("__call__")(fast::eigen_vec_view(x)));
        };
        F_.grad = [gradfn, this](const dvec& x)->dvec {
            py::gil_scoped_acquire gil;
            py::object arr = gradfn.attr("__call__")(fast::eigen_vec_view(x));
            auto a = py::cast<py::array>(arr);
            auto buf = a.request();
            dvec g(buf.shape[0]);
            std::memcpy(g.data(), buf.ptr, sizeof(double)*g.size());
            return g;
        };
        F_.hess = [hessfn, this](const dvec& x)->dmat {
            py::gil_scoped_acquire gil;
            py::object arr = hessfn.attr("__call__")(fast::eigen_vec_view(x));
            auto A = py::cast<py::array>(arr);
            auto buf = A.request();
            if (buf.ndim != 2 || buf.shape[0] != n_ || buf.shape[1] != n_)
                throw std::runtime_error("Hessian shape mismatch.");
            using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
            Eigen::Map<RowMat> M(static_cast<double*>(buf.ptr), n_, n_);
            return dmat(M);
        };
        keepalive_.push_back(std::move(valfn));
        keepalive_.push_back(std::move(gradfn));
        keepalive_.push_back(std::move(hessfn));
    }

    ADCompiled compile_constraint_(const py::object& AD, const py::object& c) {
        py::object valfn, gradfn, hessfn;
        {
            py::gil_scoped_acquire gil;
            valfn  = AD.attr("sym_val")(c, n_, true);
            gradfn = AD.attr("sym_grad")(c, n_, true);
            hessfn = AD.attr("sym_hess")(c, n_, true);
        }
        ADCompiled out;
        out.val  = [valfn, this](const dvec& x)->double {
            py::gil_scoped_acquire gil;
            return py::cast<double>(valfn.attr("__call__")(fast::eigen_vec_view(x)));
        };
        out.grad = [gradfn, this](const dvec& x)->dvec {
            py::gil_scoped_acquire gil;
            py::object arr = gradfn.attr("__call__")(fast::eigen_vec_view(x));
            auto a = py::cast<py::array>(arr);
            auto buf = a.request();
            dvec g(buf.shape[0]);
            std::memcpy(g.data(), buf.ptr, sizeof(double)*g.size());
            return g;
        };
        out.hess = [hessfn, this](const dvec& x)->dmat {
            py::gil_scoped_acquire gil;
            py::object arr = hessfn.attr("__call__")(fast::eigen_vec_view(x));
            auto A = py::cast<py::array>(arr);
            auto buf = A.request();
            if (buf.ndim != 2 || buf.shape[0] != n_ || buf.shape[1] != n_)
                throw std::runtime_error("Constraint Hessian shape mismatch.");
            using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
            Eigen::Map<RowMat> M(static_cast<double*>(buf.ptr), n_, n_);
            return dmat(M);
        };
        keepalive_.push_back(std::move(valfn));
        keepalive_.push_back(std::move(gradfn));
        keepalive_.push_back(std::move(hessfn));
        return out;
    }

    inline void set_last_x_(const Eigen::Ref<const dvec> &x) const {
        if (!last_x_.has_value() || last_x_->size() != x.size() ||
            (*last_x_ - x).lpNorm<Eigen::Infinity>() != 0.0) {
            last_x_ = x;
            // Invalidate jacobian caches if x changes
            have_je_ = false;
            have_ji_ = false;
            je_cache_.resize(0, 0);
            ji_cache_.resize(0, 0);
        }
    }

    public:
    // Add inside class Model (public:)
std::map<std::string, std::variant<double, dvec, dmat, spmat>>
eval_all(const Eigen::Ref<const dvec> &x,
         std::optional<std::vector<std::string>> components = std::nullopt) const {
    using Ret = std::map<std::string, std::variant<double, dvec, dmat, spmat>>;

    // Default set of components
    const std::vector<std::string> want =
        (components && !components->empty())
            ? *components
            : std::vector<std::string>{"f","g","H","cI","JI","cE","JE"};

    auto wants = [&](const char* k){
        for (const auto& s : want) if (s == k) return true;
        return false;
    };

    const bool need_f  = wants("f");
    const bool need_g  = wants("g");
    const bool need_H  = wants("H");
    const bool need_cI = wants("cI");
    const bool need_JI = wants("JI");
    const bool need_cE = wants("cE");
    const bool need_JE = wants("JE");

    // keep track of x and invalidate jac caches if it changed
    set_last_x_(x);

    Ret out;

    // ---------- f ----------
    if (need_f) {
        double fv;
        try {
            fv = f_val(x);
            if (!std::isfinite(fv)) fv = std::numeric_limits<double>::infinity();
        } catch (...) {
            fv = std::numeric_limits<double>::infinity();
        }
        out.emplace("f", fv);
    }

    // ---------- g ----------
    if (need_g) {
        dvec gvec;
        try {
            gvec = grad(x);
        } catch (...) {
            gvec.setZero(n_);
        }
        out.emplace("g", std::move(gvec));
    }

    // ---------- H ----------
    if (need_H) {
        if (use_sparse_) {
            try {
                spmat Hs = hess_csr(x);
                out.emplace("H", std::move(Hs));
            } catch (...) {
                spmat I(n_, n_); I.setIdentity();
                out.emplace("H", std::move(I));
            }
        } else {
            dmat Hd;
            try {
                Hd = hess(x);
            } catch (...) {
                Hd.setIdentity(n_, n_);
            }
            out.emplace("H", std::move(Hd));
        }
    }

    // ---------- Inequalities ----------
    const bool haveI = (mI_ > 0);
    if (need_cI && haveI) {
        dvec cIv;
        try {
            cIv = cI(x);
            for (int i = 0; i < cIv.size(); ++i)
                if (!std::isfinite(cIv[i])) cIv[i] = 0.0;
        } catch (...) {
            cIv.setZero(mI_);
        }
        out.emplace("cI", std::move(cIv));
    }

    if (need_JI && haveI) {
        if (use_sparse_) {
            try {
                // Build from dense once; also fills cache
                dmat Jd = JI(x);
                spmat Js(Jd.sparseView());
                Js.makeCompressed();
                out.emplace("JI", std::move(Js));
            } catch (...) {
                spmat Z(mI_, n_);
                out.emplace("JI", std::move(Z));
            }
        } else {
            dmat Jd;
            try {
                Jd = JI(x); // updates ji_cache_
            } catch (...) {
                Jd.setZero(mI_, n_);
            }
            out.emplace("JI", std::move(Jd));
        }
    }

    // ---------- Equalities ----------
    const bool haveE = (mE_ > 0);
    if (need_cE && haveE) {
        dvec cEv;
        try {
            cEv = cE(x);
            for (int i = 0; i < cEv.size(); ++i)
                if (!std::isfinite(cEv[i])) cEv[i] = 0.0;
        } catch (...) {
            cEv.setZero(mE_);
        }
        out.emplace("cE", std::move(cEv));
    }

    if (need_JE && haveE) {
        if (use_sparse_) {
            try {
                dmat Jd = JE(x);
                spmat Js(Jd.sparseView());
                Js.makeCompressed();
                out.emplace("JE", std::move(Js));
            } catch (...) {
                spmat Z(mE_, n_);
                out.emplace("JE", std::move(Z));
            }
        } else {
            dmat Jd;
            try {
                Jd = JE(x); // updates je_cache_
            } catch (...) {
                Jd.setZero(mE_, n_);
            }
            out.emplace("JE", std::move(Jd));
        }
    }

    return out;
}

};

