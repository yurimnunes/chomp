// model.h — Precompiled-from-Python AD model (hot path is pure C++)
#pragma once
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/SparseCore>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "ad/ADBindings.h" // GradFn, LagHessFn, compile_*
#include "definitions.h"

namespace detail {

// ----- Limited-memory BFGS for the Lagrangian Hessian (compact form) -----
// ----- Limited-memory BFGS for the Lagrangian Hessian (advanced) -----
struct LBFGSLagHess {
    // ===== Configuration =====
    int m_mem = 20;                   // memory size
    double curvature_eps = 1e-8;      // relative guard
    double densify_threshold = 1e-12; // sparseView threshold
    bool use_damping = true;          // Powell damping
    bool allow_indefinite = false;    // if true, can blend SR1
    double sr1_blend = 0.0;           // in [0,1]; 0 = pure BFGS
    double collinear_tol = 1e-3;      // merge if angle < ~0.057 rad
    double scale_floor = 1e-16;       // floor on H0 scale gamma

    // ===== Storage =====
    int n = 0;
    std::vector<dvec> S;     // s_k
    std::vector<dvec> Y;     // y_k
    std::vector<double> rho; // 1/(y^T s) after damping
    double gamma = 1.0;      // H0 = gamma * I

    // Cached compact factors (for materialization path of B)
    Eigen::MatrixXd M; // 2m x 2m
    Eigen::LDLT<Eigen::MatrixXd> M_ldlt;
    Eigen::MatrixXd STY, STS;
    bool need_factor = true;

    void reset(int n_new) {
        n = n_new;
        S.clear();
        Y.clear();
        rho.clear();
        gamma = 1.0;
        need_factor = true;
        M.resize(0, 0);
        STY.resize(0, 0);
        STS.resize(0, 0);
        M_ldlt = Eigen::LDLT<Eigen::MatrixXd>();
    }
    bool has_pairs() const { return !S.empty(); }
    int mem() const { return (int)S.size(); }

    // ---- Helper: robust H0 scale (BB + safeguards) ----
    static double robust_gamma(const dvec &s, const dvec &y, double floor_v) {
        const double sty = s.dot(y);
        const double yty = y.squaredNorm();
        const double val = (yty > floor_v) ? (sty / yty) : 1.0;
        return (std::isfinite(val) && val > floor_v) ? val : 1.0;
    }

    // ---- Helper: near-collinearity merge (replace the closest s_i) ----
    void maybe_aggregate(const dvec &s_new, const dvec &y_new) {
        if (S.empty())
            return;
        // Find most collinear existing s_i
        int best = -1;
        double best_abs_cos = -1.0;
        const double ns = s_new.norm();
        if (ns <= 0)
            return;
        for (int i = 0; i < mem(); ++i) {
            const double c =
                std::abs(S[i].dot(s_new) / (S[i].norm() * ns + 1e-32));
            if (c > best_abs_cos) {
                best_abs_cos = c;
                best = i;
            }
        }
        if (best_abs_cos >= 1.0 - 0.5 * collinear_tol * collinear_tol) {
            // Merge: simple weighted sum (keep scale similar)
            const double ws = 0.5, wy = 0.5;
            S[best] = ws * S[best] + (1.0 - ws) * s_new;
            Y[best] = wy * Y[best] + (1.0 - wy) * y_new;
            need_factor = true;
        } else {
            // no-op; caller will push_back
        }
    }

    // ---- Update with (x_{k+1}-x_k, gL_{k+1}-gL_k), with damping/caution ----
    void update(const dvec &x_prev, const dvec &g_prev, const dvec &x_curr,
                const dvec &g_curr) {
        if (x_prev.size() == 0 || g_prev.size() == 0)
            return;
        if (x_curr.size() != x_prev.size() || g_curr.size() != g_prev.size())
            throw std::runtime_error("LBFGSLagHess::update: size mismatch");
        if (n == 0)
            reset((int)x_curr.size());

        dvec s = x_curr - x_prev;
        dvec y = g_curr - g_prev;

        const double s_norm = s.norm(), y_norm = y.norm();
        if (s_norm < 1e-14 || y_norm < 1e-14)
            return;

        const double sty = s.dot(y);
        const double yty = y.squaredNorm();
        const double sts = s.squaredNorm();

        if (!(std::isfinite(sty) && std::isfinite(yty) && std::isfinite(sts)))
            return;

        // Relative curvature guard (Nocedal-Wright style)
        const double rel_guard = curvature_eps * std::sqrt(sts * yty);
        if (sty <= rel_guard) {
            // Try Powell damping: y <- theta*y + (1-theta) * (B s)
            if (use_damping) {
                // Approximate B s with simple scaled identity (pre-update)
                const double inv_gamma0 =
                    1.0 /
                    std::max(robust_gamma(s, y, scale_floor), scale_floor);
                const dvec Bs = inv_gamma0 * s;
                const double sBs = s.dot(Bs);
                const double theta =
                    0.8 * (sBs - rel_guard) / (sBs - sty + 1e-32);
                if (theta > 0.0 && theta < 1.0) {
                    y = theta * y + (1.0 - theta) * Bs;
                }
            }
        }

        // Recompute after damping
        const double sty2 = s.dot(y);
        if (sty2 <= rel_guard) {
            // still bad curvature: either skip or aggregate
            if (mem() > 0) {
                maybe_aggregate(s, y);
            } else {
                return;
            } // nothing to merge with; skip
        } else {
            // Update gamma (H0 scale) using robust BB rule
            gamma = robust_gamma(s, y, scale_floor);

            // Manage memory / collinearity
            if (mem() == m_mem) {
                S.erase(S.begin());
                Y.erase(Y.begin());
                rho.erase(rho.begin());
            }
            // Try to aggregate before pushing if highly collinear with existing
            int before = mem();
            maybe_aggregate(s, y);
            if (mem() == before) {
                S.emplace_back(std::move(s));
                Y.emplace_back(std::move(y));
            }

            // rho = 1 / (y^T s) for two-loop
            const double r = 1.0 / std::max(1e-32, Y.back().dot(S.back()));
            rho.emplace_back(r);
            need_factor = true;
        }
    }

    // ===== Two-loop recursion for H v (inverse Hessian action) =====
    dvec apply_H(const dvec &v) const {
        const int m = mem();
        if (m == 0)
            return gamma * v; // H0 v
        std::vector<double> alpha(m);
        dvec q = v;
        // First loop
        for (int i = m - 1; i >= 0; --i) {
            alpha[i] = rho[i] * S[i].dot(q);
            q.noalias() -= alpha[i] * Y[i];
        }
        // Apply H0
        q *= gamma;
        // Second loop
        for (int i = 0; i < m; ++i) {
            const double beta = rho[i] * Y[i].dot(q);
            q.noalias() += (alpha[i] - beta) * S[i];
        }
        return q;
    }

    // Optional: SR1 blending on top of H action (captures indefiniteness)
    dvec apply_H_sr1(const dvec &v) const {
        if (sr1_blend <= 0.0)
            return apply_H(v);
        const dvec Hv = apply_H(v);
        if (!allow_indefinite)
            return Hv;
        // One-pass limited SR1 correction using the most recent pair
        if (mem() == 0)
            return Hv;
        const dvec &s = S.back();
        const dvec &y = Y.back();
        const dvec Hy = apply_H(y);
        const dvec u = (s - Hy);
        const double denom = u.dot(y);
        if (std::abs(denom) < 1e-12)
            return Hv;
        const double coeff = sr1_blend * (u.dot(v) / denom);
        return Hv + coeff * u;
    }

    // Convenience aliases
    dvec inv_hess_vec(const dvec &v) const { return apply_H_sr1(v); } // H v
    dvec hess_vec(const dvec &v) const {
        // Use B = H^{-1} via CG if needed; approximate with symmetric secant:
        // Here we provide a cheap fallback: B ≈ (1/gamma) I - secant low-rank.
        // For high accuracy, prefer build_sparse_matrix() or a matrix-free CG.
        const double inv_gamma = 1.0 / std::max(gamma, scale_floor);
        // Use compact formula if pairs exist:
        if (!has_pairs())
            return inv_gamma * v;
        // Build W^T v and solve with M (same as your prior apply_B)
        // Use factor_if_needed() path:
        return apply_B(v); // reuses compact route below
    }

    // ===== Materialization path for B (kept from your previous version) =====
    void factor_if_needed() const {
        if (!need_factor)
            return;
        auto *self = const_cast<LBFGSLagHess *>(this);
        const int m = mem();
        self->STY.resize(m, m);
        self->STS.resize(m, m);
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < m; ++j) {
                self->STY(i, j) = S[i].dot(Y[j]);
                self->STS(i, j) = S[i].dot(S[j]);
            }
        const double inv_gamma = 1.0 / std::max(gamma, scale_floor);
        self->M.resize(2 * m, 2 * m);
        self->M.setZero();

        self->M.block(0, 0, m, m) = inv_gamma * self->STS; // S^T B0 S
        Eigen::MatrixXd L = self->STY.triangularView<Eigen::StrictlyLower>();
        self->M.block(0, m, m, m) = L;
        self->M.block(m, 0, m, m) = L.transpose();

        Eigen::VectorXd d = self->STY.diagonal().array().abs().max(1e-12);
        self->M.block(m, m, m, m).setZero();
        self->M.block(m, m, m, m).diagonal() = -d;

        // tiny ridge
        self->M.diagonal().array() += 1e-12;
        self->M_ldlt.compute(self->M.selfadjointView<Eigen::Lower>());
        if (self->M_ldlt.info() != Eigen::Success) {
            Eigen::MatrixXd M2 = self->M;
            M2.diagonal().array() += 1e-9;
            self->M_ldlt.compute(M2.selfadjointView<Eigen::Lower>());
        }
        self->need_factor = false;
    }

    dvec apply_B(const dvec &v) const {
        const int m = mem();
        const double inv_gamma = 1.0 / std::max(gamma, scale_floor);
        if (m == 0)
            return inv_gamma * v;

        factor_if_needed();
        Eigen::VectorXd Wt_v(2 * m);
        for (int j = 0; j < m; ++j) {
            Wt_v(j) = inv_gamma * S[j].dot(v);
            Wt_v(m + j) = Y[j].dot(v);
        }
        Eigen::VectorXd z = M_ldlt.solve(Wt_v);
        if (M_ldlt.info() != Eigen::Success) {
            Eigen::MatrixXd M2 = M;
            M2.diagonal().array() += 1e-9;
            Eigen::LDLT<Eigen::MatrixXd> bk;
            bk.compute(M2.selfadjointView<Eigen::Lower>());
            z = bk.solve(Wt_v);
        }
        dvec Wz = dvec::Zero(n);
        for (int j = 0; j < m; ++j)
            Wz.noalias() += (inv_gamma * z(j)) * S[j];
        for (int j = 0; j < m; ++j)
            Wz.noalias() += z(m + j) * Y[j];
        return inv_gamma * v - Wz;
    }

    // Build a (dense->sparse) B matrix for downstream solvers
    spmat build_sparse_matrix(int n_build, double thresh) const {
        const double inv_gamma = 1.0 / std::max(gamma, scale_floor);
        if (!has_pairs()) {
            spmat D(n_build, n_build);
            D.reserve(Eigen::ArrayXi::Constant(n_build, 1));
            for (int i = 0; i < n_build; ++i)
                D.insert(i, i) = inv_gamma;
            D.makeCompressed();
            return D;
        }
        Eigen::MatrixXd Bdense(n_build, n_build);
        for (int i = 0; i < n_build; ++i) {
            dvec ei = dvec::Zero(n_build);
            ei[i] = 1.0;
            Bdense.col(i) = apply_B(ei);
        }
        Eigen::MatrixXd Bsym = 0.5 * (Bdense + Bdense.transpose());
        auto SPM = Bsym.sparseView(thresh, 1.0);
        return SPM;
    }
};

} // namespace detail

// ============================================================================
// Native AD functors (compiled once via Python `ad` module)
struct ADCompiled {
    std::function<double(const dvec &)> val;
    std::function<dvec(const dvec &)> grad;
    std::function<spmat(const spmat &)> hess;
};

// ============================================================================
// Cached results for a given x (keep original structure)
struct EvalEntry {
    dvec x;
    std::size_t hash{0};

    std::optional<double> f;
    std::optional<dvec> g, cI, cE;
    std::optional<spmat> JI, JE;
    std::optional<spmat> H; // Hessian

    mutable int access_order{0};

    EvalEntry() = default;
    EvalEntry(EvalEntry &&other) noexcept = default;
    EvalEntry &operator=(EvalEntry &&other) noexcept = default;
};

// ============================================================================
// Improved LRU cache with better performance
class EvalCache {
public:
    explicit EvalCache(std::size_t capacity = 16)
        : capacity_(capacity), current_access_(0) {
        entries_.reserve(capacity);
    }

    EvalEntry *find(const Eigen::Ref<const dvec> &x) {
        std::size_t h = hash_vec(x);
        for (auto &entry : entries_) {
            if (entry.hash == h && entry.x.size() == x.size() &&
                entry.x.isApprox(x, 1e-14)) {
                entry.access_order = ++current_access_;
                return &entry;
            }
        }
        return nullptr;
    }

    EvalEntry &insert(const Eigen::Ref<const dvec> &x) {
        std::size_t h = hash_vec(x);
        if (entries_.size() < capacity_) {
            entries_.emplace_back();
            EvalEntry &entry = entries_.back();
            entry.x = x;
            entry.hash = h;
            entry.access_order = ++current_access_;
            return entry;
        }
        auto lru_it =
            std::min_element(entries_.begin(), entries_.end(),
                             [](const EvalEntry &a, const EvalEntry &b) {
                                 return a.access_order < b.access_order;
                             });
        lru_it->x = x;
        lru_it->hash = h;
        lru_it->access_order = ++current_access_;

        lru_it->f.reset();
        lru_it->g.reset();
        lru_it->cI.reset();
        lru_it->cE.reset();
        lru_it->JI.reset();
        lru_it->JE.reset();
        lru_it->H.reset();
        return *lru_it;
    }

    static std::size_t hash_vec(const Eigen::Ref<const dvec> &v) {
        constexpr std::size_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
        constexpr std::size_t FNV_PRIME = 1099511628211ULL;
        std::size_t hash = FNV_OFFSET_BASIS;
        const double *data = v.data();
        const std::size_t size = v.size() * sizeof(double);
        const uint8_t *bytes = reinterpret_cast<const uint8_t *>(data);
        for (std::size_t i = 0; i < size; ++i) {
            hash ^= bytes[i];
            hash *= FNV_PRIME;
        }
        return hash;
    }

    std::size_t capacity() const { return capacity_; }
    void set_capacity(std::size_t new_capacity) {
        capacity_ = new_capacity;
        if (entries_.size() > capacity_) {
            std::sort(entries_.begin(), entries_.end(),
                      [](const EvalEntry &a, const EvalEntry &b) {
                          return a.access_order > b.access_order;
                      });
            entries_.resize(capacity_);
        }
        entries_.reserve(capacity_);
    }

private:
    std::size_t capacity_;
    std::vector<EvalEntry> entries_;
    int current_access_;
};

// ============================================================================
// ModelC with LRU multi-point cache + optional L-BFGS Lagrangian Hessian
// ============================================================================
class ModelC {
public:
    EvalEntry current_vals = EvalEntry();

    std::optional<double> get_f() const { return current_vals.f; }
    std::optional<dvec> get_g() const { return current_vals.g; }
    std::optional<dvec> get_cI() const { return current_vals.cI; }
    std::optional<dvec> get_cE() const { return current_vals.cE; }
    std::optional<spmat> get_JI() const { return current_vals.JI; }
    std::optional<spmat> get_JE() const { return current_vals.JE; }
    std::optional<dvec> get_lb() const { return lb_; }
    std::optional<dvec> get_ub() const { return ub_; }

    std::shared_ptr<GradFn> f_grad_;
    std::shared_ptr<LagHessFn> f_hess_;
    std::vector<std::shared_ptr<GradFn>> cI_compiled_, cE_compiled_;

    int get_mI() const { return mI_; }
    int get_mE() const { return mE_; }
    int get_n() const { return n_; }

    // New toggles
    void set_use_lbfgs_hess(bool v) { use_lbfgs_hess_ = v; }
    void set_lbfgs_mem(int m) { lbfgs_.m_mem = std::max(1, m); }
    void set_lbfgs_sparse_threshold(double t) {
        lbfgs_.densify_threshold = std::max(0.0, t);
    }

    // Materialize current LBFGS Hessian without updating memory
    spmat lbfgs_matrix(double threshold = 1e-12) const {
        return lbfgs_.build_sparse_matrix(n_, threshold);
    }

    ModelC(const ModelC &) = default;
    ModelC(ModelC &&) = default;
    ModelC &operator=(const ModelC &) = default;
    ModelC &operator=(ModelC &&) = default;

    explicit ModelC(nb::object f, std::optional<nb::object> c_ineq,
                    std::optional<nb::object> c_eq, int n,
                    std::optional<nb::object> lb, std::optional<nb::object> ub,
                    bool use_sparse = false,
                    bool use_lbfgs_hess = false, int lbfgs_mem = 20,
                    double lbfgs_sparse_threshold = 1e-12)
        : n_(n), use_sparse_(use_sparse), cache_(16),
          use_lbfgs_hess_(use_lbfgs_hess) {
        f_grad_ = compile_objective_(f);
        f_hess_ = compile_lag_hess_(f, c_ineq, c_eq);

        if (c_ineq) {
            auto ineq_list = nb::cast<std::vector<nb::object>>(*c_ineq);
            for (auto &obj : ineq_list)
                cI_compiled_.push_back(compile_constraint_(obj));
        }
        if (c_eq) {
            auto eq_list = nb::cast<std::vector<nb::object>>(*c_eq);
            for (auto &obj : eq_list)
                cE_compiled_.push_back(compile_constraint_(obj));
        }

        if (lb && !lb->is_none()) {
            lb_ = nb::cast<dvec>(*lb);
            if (lb_.size() != n_)
                throw std::runtime_error("ModelC: lb size mismatch");
        } else {
            lb_ = dvec::Constant(n_, -std::numeric_limits<double>::infinity());
        }
        if (ub && !ub->is_none()) {
            ub_ = nb::cast<dvec>(*ub);
            if (ub_.size() != n_)
                throw std::runtime_error("ModelC: ub size mismatch");
        } else {
            ub_ = dvec::Constant(n_, std::numeric_limits<double>::infinity());
        }

        mI_ = (int)cI_compiled_.size();
        mE_ = (int)cE_compiled_.size();

        // L-BFGS init
        lbfgs_.reset(n_);
        lbfgs_.m_mem = std::max(1, lbfgs_mem);
        lbfgs_.densify_threshold = std::max(0.0, lbfgs_sparse_threshold);
        last_x_.resize(0);
        last_gradL_.resize(0);
    }

    // -------------------------------------------------------------------------
    // Evaluate (exact or LBFGS) Lagrangian Hessian
    spmat hess(const Eigen::Ref<const dvec> &x,
               const Eigen::Ref<const dvec> &lam,
               const Eigen::Ref<const dvec> &nu) const {
        if (!use_lbfgs_hess_) {
            // Exact AD path (unchanged)
            EvalEntry *e = cache_.find(x);
            if (!e)
                e = &cache_.insert(x);
            if (!e->H)
                e->H = f_hess_->hess_sparse(x, lam, nu);
            return *e->H;
        }

        // LBFGS path: DO NOT update here; just materialize from current memory.
        // Any (s,y) updates should be driven externally via lbfgs_update(...)
        return lbfgs_.build_sparse_matrix(n_, lbfgs_.densify_threshold);
    }

    CompiledWOp hvp;
    CompiledWOp getCompiledWOp() { return hvp; }

    void compileWOp(dvec &Sigma_x,    // size n OR empty
                    std::optional<spmat> JI,     // may be std::nullopt
                    std::optional<dvec> Sigma_s, // may be std::nullopt
                    double sigma_isotropic = 0.0) {
        auto compiled_hpv =
            CompiledWOp(f_hess_, Sigma_x, JI, Sigma_s, sigma_isotropic);
        hvp = compiled_hpv;
    }

    // -------------------------------------------------------------------------
    void eval_all(
        const Eigen::Ref<const dvec> &x,
        std::optional<std::vector<std::string>> components = std::nullopt) {
        EvalEntry *e = cache_.find(x);
        if (!e)
            e = &cache_.insert(x);

        const std::vector<std::string> want =
            (components && !components->empty())
                ? *components
                : std::vector<std::string>{"f", "g", "cI", "JI", "cE", "JE"};

        auto wants = [&](const char *k) {
            return std::find(want.begin(), want.end(), k) != want.end();
        };

        // f and g
        if ((wants("f") || wants("g")) && (!e->f || !e->g)) {
            auto [fv, fg] = f_grad_->value_grad_eigen(x);
            e->f = fv;
            e->g = fg;
            current_vals.f = e->f;
            current_vals.g = e->g;
        }

        // inequalities
        if ((wants("cI") || wants("JI")) && (!e->cI || !e->JI) && mI_ > 0) {
            auto [vals, J] =
                batch_value_grad_from_gradfns_sparse(cI_compiled_, x);
            e->cI = vals;
            e->JI = J;
            current_vals.cI = e->cI;
            current_vals.JI = e->JI;
        }

        // equalities
        if ((wants("cE") || wants("JE")) && (!e->cE || !e->JE) && mE_ > 0) {
            auto [vals, J] =
                batch_value_grad_from_gradfns_sparse(cE_compiled_, x);
            e->cE = vals;
            e->JE = J;
            current_vals.cE = e->cE;
            current_vals.JE = e->JE;
        }
    }

    // -------------------------------------------------------------------------
    double constraint_violation(const Eigen::Ref<const dvec> &x) {
        eval_all(x, std::vector<std::string>{"cI", "cE"});
        dvec cI_v, cE_v;
        if (mI_)
            cI_v = current_vals.cI.value();
        if (mE_)
            cE_v = current_vals.cE.value();

        const double scale =
            std::max<double>(1.0, std::max<double>(n_, mI_ + mE_));
        double theta = 0.0;
        if (mI_)
            theta += (cI_v.array().max(0.0)).sum() / scale;
        if (mE_)
            theta += cE_v.array().abs().sum() / scale;

        return std::isfinite(theta) ? theta
                                    : std::numeric_limits<double>::infinity();
    }

    // Call once per accepted step to update L-BFGS memory (no materialization)
    void lbfgs_update(const Eigen::Ref<const dvec> &x,
                      const Eigen::Ref<const dvec> &lam,
                      const Eigen::Ref<const dvec> &nu) {
        // Ensure gradient of Lagrangian is available
        eval_all(x, std::vector<std::string>{"g", (mI_ ? "JI" : "none"),
                                             (mE_ ? "JE" : "none")});

        dvec gradL = current_vals.g.value();
        if (mI_)
            gradL.noalias() += current_vals.JI.value().transpose() * lam;
        if (mE_)
            gradL.noalias() += current_vals.JE.value().transpose() * nu;

        if (last_x_.size() == x.size() && last_gradL_.size() == gradL.size()) {
            lbfgs_.update(last_x_, last_gradL_, x, gradL);
        }
        last_x_ = x;
        last_gradL_ = std::move(gradL);
    }

    // Optional: clear memory between phases
    void lbfgs_reset() {
        lbfgs_.reset(n_);
        last_x_.resize(0);
        last_gradL_.resize(0);
    }

private:
    int n_{0}, mI_{0}, mE_{0};
    bool use_sparse_{false};
    dvec lb_, ub_;

    mutable EvalCache cache_; // global per-model multipoint cache

    // L-BFGS state
    bool use_lbfgs_hess_{false};
    mutable detail::LBFGSLagHess lbfgs_;
    mutable dvec last_x_;
    mutable dvec last_gradL_;

    // Compilation helpers
    std::shared_ptr<LagHessFn>
    compile_lag_hess_(const nb::object &f,
                      const std::optional<nb::object> &c_ineq = nb::none(),
                      const std::optional<nb::object> &c_eq = nb::none()) {
        auto ineq_list = c_ineq ? nb::cast<std::vector<nb::object>>(*c_ineq)
                                : std::vector<nb::object>{};
        auto eq_list = c_eq ? nb::cast<std::vector<nb::object>>(*c_eq)
                            : std::vector<nb::object>{};
        return std::make_shared<LagHessFn>(f, ineq_list, eq_list, (size_t)n_,
                                           true);
    }

    std::shared_ptr<GradFn> compile_objective_(const nb::object f) {
        return std::make_shared<GradFn>(f, (size_t)n_, true);
    }

    std::shared_ptr<GradFn> compile_constraint_(const nb::object &c,
                                                bool vector_input = true) {
        return std::make_shared<GradFn>(c, (size_t)n_, vector_input);
    }
};
