// model.h
#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <optional>
#include <variant>
#include <functional>
#include <unordered_map>
#include <vector>
#include <string>
#include <limits>
#include <stdexcept>
#include <cmath>
#include <cassert>

namespace mdl {

using dvec  = Eigen::VectorXd;
using mat   = Eigen::MatrixXd;
using spmat = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;

// ---------- Helpers ----------
inline bool is_finite(const dvec& v) { return (v.array().isFinite()).all(); }
inline bool is_finite(const mat& A)  { return (A.array().isFinite()).all(); }
inline bool is_finite(const spmat& A){ return (A.toDense().array().isFinite()).all(); }

inline mat symmetrize(const mat& A) { return 0.5 * (A + A.transpose()); }
inline spmat symmetrize(const spmat& A) {
    spmat B = A;
    spmat AT = spmat(A.transpose());
    // (A + A^T)/2
    spmat S = (B + AT);
    S *= 0.5;
    return S;
}

inline void clip_inplace(mat& A, double clip_max) {
    if (!std::isfinite(clip_max) || clip_max <= 0) return;
    A = A.array().max(-clip_max).min(clip_max).matrix();
}
inline spmat clip_copy(const spmat& A, double clip_max) {
    if (!std::isfinite(clip_max) || clip_max <= 0) return A;
    spmat B = A;
    B.makeCompressed();
    auto* d = B.valuePtr();
    for (int i = 0; i < B.nonZeros(); ++i) {
        if (d[i] >  clip_max) d[i] =  clip_max;
        if (d[i] < -clip_max) d[i] = -clip_max;
        if (!std::isfinite(d[i])) d[i] = 0.0;
    }
    return B;
}

// ---------- Return containers ----------
struct EvalAll {
    // always filled if requested
    std::optional<double> f;
    std::optional<dvec>   g;
    // H, JI, JE may be dense or sparse depending on use_sparse
    std::optional<std::variant<mat, spmat>> H;
    std::optional<dvec>   cI;   // size m_ineq or empty
    std::optional<std::variant<mat, spmat>> JI;
    std::optional<dvec>   cE;   // size m_eq or empty
    std::optional<std::variant<mat, spmat>> JE;
};

struct KKTResiduals {
    double stat{std::numeric_limits<double>::infinity()};
    double ineq{0.0};
    double eq{0.0};
    double comp{0.0};
};

// ---------- Model ----------
class Model {
public:
    // AD hooks (set these after construction — like your AD.sym_* products)
    // Objective
    std::function<double(const dvec&)>      f_val;
    std::function<dvec(const dvec&)>        f_grad;
    // Provide either dense OR sparse Hess functor (or both; dense takes precedence if both set)
    std::function<mat(const dvec&)>         f_hess_dense;
    std::function<spmat(const dvec&)>       f_hess_sparse;

    // Inequalities: c_i(x) >= ? (use same semantics as Python code)
    std::vector<std::function<double(const dvec&)>> cI_vals;
    std::vector<std::function<dvec  (const dvec&)>> cI_grads;
    std::vector<std::function<mat   (const dvec&)>> cI_hess_dense;
    std::vector<std::function<spmat (const dvec&)>> cI_hess_sparse;

    // Equalities
    std::vector<std::function<double(const dvec&)>> cE_vals;
    std::vector<std::function<dvec  (const dvec&)>> cE_grads;
    std::vector<std::function<mat   (const dvec&)>> cE_hess_dense;
    std::vector<std::function<spmat (const dvec&)>> cE_hess_sparse;

    // Public config-like knobs (mirrors your Python cfg usage where needed)
    double multiplier_threshold{1e-8};
    double hess_clip_max{1e12};
    double hess_diag_floor{0.0};

    // Bounds (not used directly here, but stored like in Python class)
    dvec lb, ub;

public:
    Model(int n_,
          std::vector<std::function<double(const dvec&)>> cI = {},
          std::vector<std::function<double(const dvec&)>> cE = {},
          const dvec& lb_ = dvec(), const dvec& ub_ = dvec(),
          bool use_sparse_ = false)
    : n_(n_),
      m_ineq_(static_cast<int>(cI.size())),
      m_eq_(static_cast<int>(cE.size())),
      lb(lb_), ub(ub_),
      use_sparse_(use_sparse_) {
        if (n_ <= 0) throw std::invalid_argument("n must be positive");
        cI_vals = std::move(cI);
        cE_vals = std::move(cE);
        // grads/hessians must be wired via setters or direct access by the caller
    }

    int n() const { return n_; }
    int m_ineq() const { return m_ineq_; }
    int m_eq() const { return m_eq_; }
    bool use_sparse() const { return use_sparse_; }
    void set_use_sparse(bool v) { use_sparse_ = v; }

    // ---------- eval_all ----------
    // components: {"f","g","H","cI","JI","cE","JE"}  (if empty → all)
    EvalAll eval_all(const dvec& x,
                     const std::vector<std::string>& components = {}) {
        guard_x_(x);

        const bool want_all = components.empty();
        auto wants = [&](std::string_view k) {
            if (want_all) return true;
            for (auto& s : components) if (s == k) return true;
            return false;
        };

        // Cache fast path (exact x match)
        if (last_x_ && (*last_x_).size() == x.size() && (*last_x_).isApprox(x, 0.0)) {
            if (want_all || have_all_(components, last_eval_)) {
                return subset_(last_eval_, components);
            }
        }

        EvalAll out;

        // --- Objective value ---
        if (wants("f")) {
            try {
                double fv = f_val ? f_val(x) : std::numeric_limits<double>::infinity();
                if (!std::isfinite(fv)) fv = std::numeric_limits<double>::infinity();
                out.f = fv;
            } catch (...) {
                out.f = std::numeric_limits<double>::infinity();
            }
        }

        // --- Gradient ---
        if (wants("g")) {
            try {
                dvec g = f_grad ? f_grad(x) : dvec::Zero(n_);
                if (!is_finite(g)) g = dvec::Zero(n_);
                out.g = std::move(g);
            } catch (...) {
                out.g = dvec::Zero(n_);
            }
        }

        // --- Hessian (symmetrize + sanitize + dense<->sparse as requested) ---
        if (wants("H")) {
            try {
                if (f_hess_dense) {
                    mat H = symmetrize(f_hess_dense(x));
                    if (!is_finite(H)) H = mat::Identity(n_, n_);
                    clip_inplace(H, hess_clip_max);
                    if (use_sparse_) {
                        spmat S = H.sparseView();
                        out.H = std::variant<mat, spmat>(std::in_place_type<spmat>, std::move(S));
                    } else {
                        out.H = std::variant<mat, spmat>(std::in_place_type<mat>, std::move(H));
                    }
                } else if (f_hess_sparse) {
                    spmat Hs = symmetrize(f_hess_sparse(x));
                    Hs = clip_copy(Hs, hess_clip_max);
                    if (use_sparse_) {
                        out.H = std::variant<mat, spmat>(std::in_place_type<spmat>, std::move(Hs));
                    } else {
                        out.H = std::variant<mat, spmat>(std::in_place_type<mat>, Hs.toDense());
                    }
                } else {
                    // fallback
                    if (use_sparse_)
                        out.H = std::variant<mat, spmat>(std::in_place_type<spmat>, spmat(n_, n_));
                    else
                        out.H = std::variant<mat, spmat>(std::in_place_type<mat>, mat::Zero(n_, n_));
                }
            } catch (...) {
                if (use_sparse_)
                    out.H = std::variant<mat, spmat>(std::in_place_type<spmat>, spmat::Identity(n_, n_));
                else
                    out.H = std::variant<mat, spmat>(std::in_place_type<mat>, mat::Identity(n_, n_));
            }
        }

        const bool haveI = m_ineq_ > 0;
        const bool haveE = m_eq_   > 0;

        // --- Inequalities cI ---
        if (wants("cI")) {
            if (haveI) {
                dvec cI(m_ineq_);
                bool ok = true;
                for (int i = 0; i < m_ineq_; ++i) {
                    try {
                        cI[i] = cI_vals[i] ? cI_vals[i](x) : 0.0;
                        if (!std::isfinite(cI[i])) ok = false;
                    } catch (...) { ok = false; break; }
                }
                if (!ok || !is_finite(cI)) cI.setZero();
                out.cI = std::move(cI);
            } else {
                out.cI = std::nullopt;
            }
        }

        // --- JI ---
        if (wants("JI")) {
            if (haveI) {
                try {
                    if (use_sparse_) {
                        // build dense then convert (simple and safe)
                        mat J = mat::Zero(m_ineq_, n_);
                        for (int i = 0; i < m_ineq_; ++i) {
                            dvec row = (i < (int)cI_grads.size() && cI_grads[i]) ? cI_grads[i](x) : dvec::Zero(n_);
                            if (!is_finite(row)) row.setZero();
                            J.row(i) = row.transpose();
                        }
                        spmat Js = J.sparseView();
                        out.JI = std::variant<mat, spmat>(std::in_place_type<spmat>, std::move(Js));
                    } else {
                        mat J = mat::Zero(m_ineq_, n_);
                        for (int i = 0; i < m_ineq_; ++i) {
                            dvec row = (i < (int)cI_grads.size() && cI_grads[i]) ? cI_grads[i](x) : dvec::Zero(n_);
                            if (!is_finite(row)) row.setZero();
                            J.row(i) = row.transpose();
                        }
                        out.JI = std::variant<mat, spmat>(std::in_place_type<mat>, std::move(J));
                    }
                } catch (...) {
                    if (use_sparse_)
                        out.JI = std::variant<mat, spmat>(std::in_place_type<spmat>, spmat(m_ineq_, n_));
                    else
                        out.JI = std::variant<mat, spmat>(std::in_place_type<mat>, mat::Zero(m_ineq_, n_));
                }
            } else {
                out.JI = std::nullopt;
            }
        }

        // --- Equalities cE ---
        if (wants("cE")) {
            if (haveE) {
                dvec cE(m_eq_);
                bool ok = true;
                for (int j = 0; j < m_eq_; ++j) {
                    try {
                        cE[j] = cE_vals[j] ? cE_vals[j](x) : 0.0;
                        if (!std::isfinite(cE[j])) ok = false;
                    } catch (...) { ok = false; break; }
                }
                if (!ok || !is_finite(cE)) cE.setZero();
                out.cE = std::move(cE);
            } else {
                out.cE = std::nullopt;
            }
        }

        // --- JE ---
        if (wants("JE")) {
            if (haveE) {
                try {
                    if (use_sparse_) {
                        mat J = mat::Zero(m_eq_, n_);
                        for (int j = 0; j < m_eq_; ++j) {
                            dvec row = (j < (int)cE_grads.size() && cE_grads[j]) ? cE_grads[j](x) : dvec::Zero(n_);
                            if (!is_finite(row)) row.setZero();
                            J.row(j) = row.transpose();
                        }
                        spmat Js = J.sparseView();
                        out.JE = std::variant<mat, spmat>(std::in_place_type<spmat>, std::move(Js));
                    } else {
                        mat J = mat::Zero(m_eq_, n_);
                        for (int j = 0; j < m_eq_; ++j) {
                            dvec row = (j < (int)cE_grads.size() && cE_grads[j]) ? cE_grads[j](x) : dvec::Zero(n_);
                            if (!is_finite(row)) row.setZero();
                            J.row(j) = row.transpose();
                        }
                        out.JE = std::variant<mat, spmat>(std::in_place_type<mat>, std::move(J));
                    }
                } catch (...) {
                    if (use_sparse_)
                        out.JE = std::variant<mat, spmat>(std::in_place_type<spmat>, spmat(m_eq_, n_));
                    else
                        out.JE = std::variant<mat, spmat>(std::in_place_type<mat>, mat::Zero(m_eq_, n_));
                }
            } else {
                out.JE = std::nullopt;
            }
        }

        // cache
        last_x_    = x;
        last_eval_ = out;
        return subset_(last_eval_, components);
    }

    // ---------- Lagrangian Hessian ----------
    // Returns dense or sparse depending on use_sparse()
    std::variant<mat, spmat> lagrangian_hessian(const dvec& x,
                                                const dvec& lam,
                                                const dvec& nu) {
        guard_x_(x);
        dvec lamv = lam;
        dvec nuv  = nu;
        if (lamv.size() != m_ineq_) lamv.conservativeResize(m_ineq_), lamv.setZero();
        if (nuv.size()  != m_eq_)   nuv.conservativeResize(m_eq_),   nuv.setZero();
        if (!is_finite(x) || !is_finite(lamv) || !is_finite(nuv))
            throw std::invalid_argument("Non-finite x/lam/nu");

        // Base H
        std::variant<mat, spmat> H;
        if (f_hess_dense) {
            mat Hd = symmetrize(f_hess_dense(x));
            if (!is_finite(Hd)) Hd.setZero();
            clip_inplace(Hd, hess_clip_max);
            H = use_sparse_ ? std::variant<mat, spmat>(spmat(Hd.sparseView()))
                            : std::variant<mat, spmat>(std::move(Hd));
        } else if (f_hess_sparse) {
            spmat Hs = symmetrize(f_hess_sparse(x));
            if (!is_finite(Hs)) Hs.setZero();
            Hs = clip_copy(Hs, hess_clip_max);
            H = use_sparse_ ? std::variant<mat, spmat>(std::move(Hs))
                            : std::variant<mat, spmat>(Hs.toDense());
        } else {
            H = use_sparse_ ? std::variant<mat, spmat>(spmat(n_, n_))
                            : std::variant<mat, spmat>(mat::Zero(n_, n_));
        }

        auto add_piece_dense = [&](mat& Acc, double w, const auto& fun) {
            if (!fun || std::abs(w) <= multiplier_threshold) return;
            try {
                mat A = symmetrize(fun(x));
                clip_inplace(A, hess_clip_max);
                if (!is_finite(A)) return;
                Acc.noalias() += (w * A);
            } catch (...) { /* ignore */ }
        };
        auto add_piece_sparse = [&](spmat& Acc, double w, const auto& fun) {
            if (!fun || std::abs(w) <= multiplier_threshold) return;
            try {
                spmat A = symmetrize(fun(x));
                A = clip_copy(A, hess_clip_max);
                if (!is_finite(A)) return;
                Acc = Acc + (w * A);
            } catch (...) { /* ignore */ }
        };

        if (use_sparse_) {
            spmat Acc = std::holds_alternative<spmat>(H) ? std::get<spmat>(H)
                                                          : spmat(std::get<mat>(H).sparseView());
            for (int i = 0; i < m_ineq_ && i < (int)std::max(cI_hess_dense.size(), cI_hess_sparse.size()); ++i) {
                if (i < (int)cI_hess_sparse.size() && cI_hess_sparse[i]) add_piece_sparse(Acc, lamv[i], cI_hess_sparse[i]);
                else if (i < (int)cI_hess_dense.size() && cI_hess_dense[i]) {
                    // densify then sparse (simple)
                    mat tmp = symmetrize(cI_hess_dense[i](x));
                    clip_inplace(tmp, hess_clip_max);
                    Acc = Acc + (lamv[i] * tmp.sparseView());
                }
            }
            for (int j = 0; j < m_eq_ && j < (int)std::max(cE_hess_dense.size(), cE_hess_sparse.size()); ++j) {
                if (j < (int)cE_hess_sparse.size() && cE_hess_sparse[j]) add_piece_sparse(Acc, nuv[j], cE_hess_sparse[j]);
                else if (j < (int)cE_hess_dense.size() && cE_hess_dense[j]) {
                    mat tmp = symmetrize(cE_hess_dense[j](x));
                    clip_inplace(tmp, hess_clip_max);
                    Acc = Acc + (nuv[j] * tmp.sparseView());
                }
            }
            // diag floor
            if (hess_diag_floor > 0.0) {
                spmat D(n_, n_);
                D.setIdentity();
                D *= hess_diag_floor;
                // add only where |diag| < floor (simple: just add everywhere, matches Python "tiny floor" behavior)
                Acc = Acc + D;
            }
            if (!is_finite(Acc)) {
                Acc.setIdentity();
            }
            return Acc;
        } else {
            mat Acc = std::holds_alternative<mat>(H) ? std::get<mat>(H)
                                                     : std::get<spmat>(H).toDense();
            for (int i = 0; i < m_ineq_ && i < (int)std::max(cI_hess_dense.size(), cI_hess_sparse.size()); ++i) {
                if (i < (int)cI_hess_dense.size() && cI_hess_dense[i]) add_piece_dense(Acc, lamv[i], cI_hess_dense[i]);
                else if (i < (int)cI_hess_sparse.size() && cI_hess_sparse[i]) {
                    spmat tmp = symmetrize(cI_hess_sparse[i](x));
                    tmp = clip_copy(tmp, hess_clip_max);
                    Acc.noalias() += (lamv[i] * tmp.toDense());
                }
            }
            for (int j = 0; j < m_eq_ && j < (int)std::max(cE_hess_dense.size(), cE_hess_sparse.size()); ++j) {
                if (j < (int)cE_hess_dense.size() && cE_hess_dense[j]) add_piece_dense(Acc, nuv[j], cE_hess_dense[j]);
                else if (j < (int)cE_hess_sparse.size() && cE_hess_sparse[j]) {
                    spmat tmp = symmetrize(cE_hess_sparse[j](x));
                    tmp = clip_copy(tmp, hess_clip_max);
                    Acc.noalias() += (nuv[j] * tmp.toDense());
                }
            }
            if (hess_diag_floor > 0.0) {
                for (int i = 0; i < n_; ++i) {
                    if (!std::isfinite(Acc(i,i)) || std::abs(Acc(i,i)) < hess_diag_floor)
                        Acc(i,i) = hess_diag_floor;
                }
            }
            if (!is_finite(Acc)) {
                Acc.setIdentity();
            }
            return Acc;
        }
    }

    // ---------- Constraint violation (L1 of cI+ + |cE|)/scale ----------
    double constraint_violation(const dvec& x) {
        // mirrors your Python’s robust behavior
        auto d = eval_all(x, {"cI","cE"});

        const int mI = m_ineq_, mE = m_eq_;
        auto scale = static_cast<double>(std::max({1, n_, mI + mE}));

        double theta = 0.0;
        if (mI && d.cI) {
            const dvec& cI = *d.cI;
            double s = 0.0;
            for (int i = 0; i < cI.size(); ++i) s += std::max(0.0, cI[i]);
            theta += s / scale;
        }
        if (mE && d.cE) {
            const dvec& cE = *d.cE;
            double s = 0.0;
            for (int i = 0; i < cE.size(); ++i) s += std::abs(cE[i]);
            theta += s / scale;
        }
        if (!std::isfinite(theta)) return std::numeric_limits<double>::infinity();
        return theta;
    }

    // ---------- KKT residuals ----------
    KKTResiduals kkt_residuals(const dvec& x, const dvec& lam, const dvec& nu) {
        if (x.size() != n_ || lam.size() != m_ineq_ || nu.size() != m_eq_)
            throw std::invalid_argument("shape mismatch in kkt_residuals");
        if (!is_finite(x) || !is_finite(lam) || !is_finite(nu))
            throw std::invalid_argument("non-finite x/lam/nu");

        std::vector<std::string> need = {"g"};
        if (m_ineq_) { need.push_back("JI"); need.push_back("cI"); }
        if (m_eq_)   { need.push_back("JE"); need.push_back("cE"); }
        auto d = eval_all(x, need);

        const dvec g = (d.g ? *d.g : dvec::Zero(n_));
        double scale_g = std::max(1.0, g.lpNorm<Eigen::Infinity>());

        // rL = g + JI^T max(lam,0) + JE^T nu
        dvec rL = g;
        if (m_ineq_ && d.JI) {
            dvec lam_pos = lam.cwiseMax(0.0);
            if (std::holds_alternative<mat>(*d.JI))
                rL.noalias() += std::get<mat>(*d.JI).transpose() * lam_pos;
            else
                rL.noalias() += std::get<spmat>(*d.JI).transpose() * lam_pos;
        }
        if (m_eq_ && d.JE) {
            if (std::holds_alternative<mat>(*d.JE))
                rL.noalias() += std::get<mat>(*d.JE).transpose() * nu;
            else
                rL.noalias() += std::get<spmat>(*d.JE).transpose() * nu;
        }
        double stat_inf = rL.lpNorm<Eigen::Infinity>() / scale_g;

        // Feasibility & complementarity
        double ineq_inf = 0.0, comp_inf = 0.0, eq_inf = 0.0;
        if (m_ineq_ && d.cI) {
            const dvec& cI = *d.cI;
            dvec cI_plus = cI.cwiseMax(0.0);
            ineq_inf = cI_plus.lpNorm<Eigen::Infinity>() / scale_g;

            dvec comp = (lam.cwiseMax(0.0).array() * cI.array()).abs().matrix();
            comp_inf = (comp.size() ? comp.maxCoeff() : 0.0) / scale_g;
        }
        if (m_eq_ && d.cE) {
            eq_inf = d.cE->lpNorm<Eigen::Infinity>() / scale_g;
        }

        KKTResiduals out{stat_inf, ineq_inf, eq_inf, comp_inf};
        if (!std::isfinite(out.stat) || !std::isfinite(out.ineq) ||
            !std::isfinite(out.eq)   || !std::isfinite(out.comp)) {
            out = {std::numeric_limits<double>::infinity(),0,0,0};
        }
        return out;
    }

    void reset_cache() {
        last_x_.reset();
        last_eval_ = EvalAll{};
    }

private:
    int  n_{0};
    int  m_ineq_{0};
    int  m_eq_{0};
    bool use_sparse_{false};

    std::optional<dvec> last_x_;
    EvalAll             last_eval_;

    void guard_x_(const dvec& x) const {
        if (x.size() != n_)
            throw std::invalid_argument("x dimension mismatch");
        if (!is_finite(x))
            throw std::invalid_argument("x has non-finite entries");
    }

    static bool have_all_(const std::vector<std::string>& keys, const EvalAll& e) {
        for (auto& k : keys) if (!has_key_(e, k)) return false;
        return true;
    }
    static bool has_key_(const EvalAll& e, const std::string& k) {
        if (k=="f")  return e.f.has_value();
        if (k=="g")  return e.g.has_value();
        if (k=="H")  return e.H.has_value();
        if (k=="cI") return e.cI.has_value();
        if (k=="JI") return e.JI.has_value();
        if (k=="cE") return e.cE.has_value();
        if (k=="JE") return e.JE.has_value();
        return false;
    }
    static EvalAll subset_(const EvalAll& e, const std::vector<std::string>& keys) {
        if (keys.empty()) return e;
        EvalAll out;
        for (auto& k : keys) {
            if (k=="f")  out.f  = e.f;
            if (k=="g")  out.g  = e.g;
            if (k=="H")  out.H  = e.H;
            if (k=="cI") out.cI = e.cI;
            if (k=="JI") out.JI = e.JI;
            if (k=="cE") out.cE = e.cE;
            if (k=="JE") out.JE = e.JE;
        }
        return out;
    }
};

} // namespace mdl
