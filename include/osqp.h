// include/osqp.h
#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace sosqp {

using Scalar = double;
using Vec = Eigen::VectorXd;
using SpMat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int>;
using Triplet = Eigen::Triplet<Scalar>;

// -------- Efficient in-place column/row scaling for SpMat --------
inline void scale_cols_inplace(SpMat &M, const Vec &s) {
    const int n = M.cols();
    for (int j = 0; j < n; ++j) {
        Scalar sj = (j < s.size() ? s[j] : 1.0);
        if (sj == 1.0)
            continue;
        for (SpMat::InnerIterator it(M, j); it; ++it) {
            it.valueRef() *= sj; // scale column j
        }
    }
}
inline void scale_rows_inplace(SpMat &M, const Vec &s) {
    const int n = M.cols();
    for (int j = 0; j < n; ++j) {
        for (SpMat::InnerIterator it(M, j); it; ++it) {
            int i = it.row();
            Scalar si = (i < s.size() ? s[i] : 1.0);
            if (si != 1.0)
                it.valueRef() *= si; // scale row i
        }
    }
}

// ---------- Division-free LDLᵀ solve wrapper for Eigen::SimplicialLDLT
struct LDLtDivFree {
    Eigen::SimplicialLDLT<SpMat> ldlt;
    Vec invD; // diag(D)^{-1}
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> P;
    SpMat L; // unit-lower
    bool factorized = false;

    void compute(const SpMat &M) {
        ldlt.compute(M);
        factorized = (ldlt.info() == Eigen::Success);
        if (!factorized)
            return;
        P = ldlt.permutationP();
        L = SpMat(ldlt.matrixL());
        Eigen::VectorXd D = ldlt.vectorD();
        invD = D.cwiseInverse();
    }

    // x = M^{-1} b
    inline Vec solve(const Vec &b) const {
        Vec pb = P * b;
        Vec y = pb;
        L.template triangularView<Eigen::Lower>().solveInPlace(y);
        Vec z = invD.cwiseProduct(y); // division-free middle step
        Vec w = z;
        L.transpose().template triangularView<Eigen::Upper>().solveInPlace(w);
        return P.transpose() * w;
    }

    // X = M^{-1} B  (apply to multiple RHS columns)
    template <typename DenseMat>
    inline DenseMat solveMat(const DenseMat &B) const {
        DenseMat PB = P * B;
        DenseMat Y = PB;
        // forward
        L.template triangularView<Eigen::Lower>().solveInPlace(Y);
        // middle: D^{-1} * Y
        for (int j = 0; j < Y.cols(); ++j)
            Y.col(j) = invD.cwiseProduct(Y.col(j));
        // backward
        DenseMat W = Y;
        L.transpose().template triangularView<Eigen::Upper>().solveInPlace(W);
        return P.transpose() * W;
    }
};

// ---------- Quasi-definite KKT solver via SPD Schur complement (batched build)
// ----------
// ---------- Fast Quasi-definite KKT solver via sparse Schur complement
// ----------
// ---------- Fast KKT solver with cached factorizations and efficient rho
// updates ----------
class KKTQuasiDef {
public:
    // Build from P, A, rho, sigma - initial setup
    bool build(const SpMat &P, const SpMat &A, const Vec &rho, Scalar sigma,
               Scalar diag_reg) {
        n_ = int(P.rows());
        m_ = int(A.rows());
        if (P.cols() != n_ || A.cols() != n_ || rho.size() != m_)
            return false;

        A_ = A;
        rho_base_ = rho;

        // Build and cache H = 0.5(P+P^T) + (sigma+diag_reg)I
        SpMat H = Scalar(0.5) * (P + SpMat(P.transpose()));
        SpMat I(n_, n_);
        I.setIdentity();
        H += (sigma + diag_reg) * I;

        // Cache H factorization (this won't change)
        H_.compute(H);
        if (!H_.factorized)
            return false;

        // Precompute and cache A H^{-1} A^T (the expensive part)
        if (!build_AHA_cache())
            return false;

        // Build initial Schur complement with current rho
        return update_schur_complement(rho);
    }

    // Fast update when only rho changes (no refactorization of H needed)
    bool update_rho(const Vec &new_rho) {
        if (new_rho.size() != m_)
            return false;
        return update_schur_complement(new_rho);
    }

    // Solve K [x; y] = [rhs1; rhs2]
    std::pair<Vec, Vec> solve(const Vec &rhs1, const Vec &rhs2) const {
        Vec t = H_.solve(rhs1);  // t = H^{-1} rhs1
        Vec At = A_times_vec(t); // At = A t
        Vec rhs_y = At - rhs2;
        Vec y = S_.solve(rhs_y);
        Vec ATy = AT_times_vec(y);
        Vec x = H_.solve(rhs1 - ATy);
        return {x, y};
    }

private:
    int n_{0}, m_{0};
    SpMat A_;
    Vec rho_base_;
    LDLtDivFree H_, S_;

    // Cached A H^{-1} A^T computation
    std::vector<std::vector<std::pair<int, Scalar>>>
        AHA_sparse_; // sparse storage
    bool analyzed_ = false;

    bool build_AHA_cache() {
        // Build sparse A^T for efficient column access
        SpMat AT = A_.transpose();
        AT.makeCompressed();

        AHA_sparse_.resize(m_);

        // For each constraint i, compute row i of A H^{-1} A^T
        for (int i = 0; i < m_; ++i) {
            // Extract row i of A (column i of A^T)
            Vec ai = Vec::Zero(n_);
            for (SpMat::InnerIterator it(AT, i); it; ++it) {
                ai[it.row()] = it.value();
            }

            if (ai.norm() < 1e-15)
                continue; // skip empty rows

            // Solve H * yi = ai
            Vec yi = H_.solve(ai);

            // Compute A * yi to get row i of A H^{-1} A^T
            Vec Ayi = A_times_vec(yi);

            // Store sparse entries for row i
            AHA_sparse_[i].clear();
            for (int j = 0; j < m_; ++j) {
                if (std::abs(Ayi[j]) > 1e-15) {
                    AHA_sparse_[i].emplace_back(j, Ayi[j]);
                }
            }
        }
        return true;
    }

    bool update_schur_complement(const Vec &rho) {
        // Build S = cached_AHA + diag(1/rho) using cached sparse structure
        std::vector<Triplet> S_triplets;
        S_triplets.reserve(m_ * 20);

        // Add cached A H^{-1} A^T entries
        for (int i = 0; i < m_; ++i) {
            for (const auto &entry : AHA_sparse_[i]) {
                int j = entry.first;
                Scalar val = entry.second;
                S_triplets.emplace_back(i, j, val);
            }

            // Add diagonal 1/rho[i] term
            Scalar diag_val = Scalar(1.0) / std::max(rho[i], Scalar(1e-30));
            S_triplets.emplace_back(i, i, diag_val);
        }

        SpMat S(m_, m_);
        S.setFromTriplets(S_triplets.begin(), S_triplets.end());
        S.makeCompressed();

        // Check if rho changed significantly - if so, force new symbolic
        // analysis
        static Vec prev_rho = rho;
        bool rho_changed_significantly = false;
        if (prev_rho.size() == rho.size()) {
            for (int i = 0; i < rho.size(); ++i) {
                if (std::abs(rho[i] / prev_rho[i] - 1.0) >
                    0.5) { // 50% change threshold
                    rho_changed_significantly = true;
                    break;
                }
            }
        } else {
            rho_changed_significantly = true;
        }
        prev_rho = rho;

        if (rho_changed_significantly) {
            analyzed_ = false; // Force new symbolic analysis
        }

        if (!analyzed_) {
            S_.ldlt.analyzePattern(S);
            analyzed_ = true;
        }

        S_.ldlt.factorize(S);
        if (S_.ldlt.info() != Eigen::Success)
            return false;

        // Update cached components
        S_.P = S_.ldlt.permutationP();
        S_.L = SpMat(S_.ldlt.matrixL());
        S_.invD = S_.ldlt.vectorD().cwiseInverse();
        S_.factorized = true;

        return true;
    }

    inline Vec A_times_vec(const Vec &v) const {
        Vec out = Vec::Zero(m_);
        for (int j = 0; j < A_.outerSize(); ++j) {
            const Scalar vj = v[j];
            if (std::abs(vj) < 1e-15)
                continue;
            for (SpMat::InnerIterator it(A_, j); it; ++it)
                out[it.row()] += it.value() * vj;
        }
        return out;
    }

    inline Vec AT_times_vec(const Vec &w) const {
        Vec out = Vec::Zero(n_);
        for (int j = 0; j < A_.outerSize(); ++j)
            for (SpMat::InnerIterator it(A_, j); it; ++it)
                out[j] += it.value() * w[it.row()];
        return out;
    }
};

// -------- create sqrt(rho) * A (scale rows) --------
inline SpMat create_scaled_A(const SpMat &A, const Vec &rho) {
    if (A.rows() == 0)
        return A;
    SpMat scaled = A;
    Vec sqrt_rho = rho.cwiseSqrt();
    scale_rows_inplace(scaled, sqrt_rho);
    scaled.makeCompressed();
    return scaled;
}
// ============================ Small utilities ============================
inline Scalar inf_norm(const Vec &v) {
    return (v.size() ? v.lpNorm<Eigen::Infinity>() : 0.0);
}

// ---------- Fast Normal-Equations cache with symbolic reuse ----------
struct NEFast {
    // M(w) = H + w * G
    SpMat H;          // constant while P, sigma, diag_reg unchanged
    SpMat G;          // A^T diag(rho0) A built once using the *base* rho
    SpMat M;          // working matrix = H + w * G
    LDLtDivFree ldlt; // division-free LDLᵀ wrapper
    bool analyzed = false;
    int n = 0;

    // Build H and G once. rho_base is your current rho vector.
    bool build_once(const SpMat &P_sym, Scalar sigma, Scalar diag_reg,
                    const SpMat &A, const Vec &rho_base) {
        n = int(P_sym.rows());
        if (P_sym.cols() != n || A.cols() != n || rho_base.size() != A.rows())
            return false;

        // H = P_sym + (sigma + diag_reg) I
        H = P_sym;
        SpMat I(n, n);
        I.setIdentity();
        H += (sigma + diag_reg) * I;
        H.makeCompressed();

        // G = A^T diag(rho_base) A
        SpMat As = create_scaled_A(A, rho_base); // scales rows by sqrt(rho)
        G = As.transpose() * As;
        G.makeCompressed();

        // Initial M = H + 1 * G (we'll treat "1" as the initial weight)
        M = H;
        M += G;
        M.makeCompressed();

        // One-time symbolic analysis; then numerical factorize
        ldlt.ldlt.analyzePattern(M);
        analyzed = true;

        // Numeric factorization
        ldlt.compute(M);
        return ldlt.factorized;
    }

    // Change weight w: M = H + w * G (numeric refactor only)
    bool refactor_with_weight(Scalar w) {
        if (!analyzed)
            return false;
        // Rebuild values: M = H + w*G (pattern doesn't change)
        M = H;
        if (w != Scalar(0))
            M += w * G;
        M.makeCompressed();
        ldlt.ldlt.factorize(M); // numeric-only refactor
        if (ldlt.ldlt.info() != Eigen::Success)
            return false;

        // refresh cached pieces for division-free solves
        ldlt.P = ldlt.ldlt.permutationP();
        ldlt.L = SpMat(ldlt.ldlt.matrixL());
        ldlt.invD = ldlt.ldlt.vectorD().cwiseInverse();
        ldlt.factorized = true;
        return true;
    }

    Vec solve(const Vec &rhs) const { return ldlt.solve(rhs); }
};

// =============================== Settings ===============================
struct Settings {
    // ---- Algorithm params (OSQP-style defaults) ----
    Scalar sigma = 1e-6; // x-regularization
    Scalar alpha = 1.6;  // over-relaxation
    Scalar rho0 = 1e-1;  // base rho for inequalities

    // Back-compat with bindings (mirror fields)
    Scalar rho =
        1e-1; // DEPRECATED: kept for pybind repr/API; used to init rho0

    Scalar rho_eq_scale = 1e3; // multiplier for equality rows (l==u)
    bool adaptive_rho = true;

    // Termination (OSQP Sec. 3.4)
    Scalar eps_abs = 1e-3;
    Scalar eps_rel = 1e-3;

    // Infeasibility detection thresholds
    Scalar eps_pinf = 1e-4;
    Scalar eps_dinf = 1e-4;

    // Iterations / checks
    int max_iter = 4000;
    int check_every = 5;

    // Numerical tolerances
    Scalar diag_reg = 1e-12; // tiny I added to SPD system
    Scalar eq_tol = 1e-9;    // equality detect |l-u| <= eq_tol
    bool verbose = false;

    // Polishing
    bool polish = true;
    Scalar polish_delta = 1e-6;
    int polish_refine_steps = 3;

    // Back-compat knobs (not critical internally)
    Scalar rho_min = 1e-6;
    Scalar rho_max = 1e6;
    Scalar explode_refactor = 1e3;

    Scalar max_refactor = 3; // UNUSED internally; kept for API compatibility

    // --------- Ruiz scaling (optional; disabled by default) ---------
    bool enable_ruiz = true;    // enable modified Ruiz equilibration
    int ruiz_max_iter = 20;     // max iterations (few tens typical)
    Scalar ruiz_tol = 1e-3;     // convergence of per-iter deltas
    bool check_unscaled = true; // check stopping on original problem

    Settings() = default;
};

// =============================== Results ===============================
struct Residuals {
    Scalar pri_inf = 0.0; // ||Ax - z||_inf
    Scalar dua_inf = 0.0; // ||Px + q + A^T y||_inf
};

struct Result {
    std::string status = "max_iter_reached";
    int iters = 0;
    Scalar obj_val = std::numeric_limits<Scalar>::quiet_NaN();
    Vec x, z, y;
    Residuals res;

    // Infeasibility certificates (optional)
    bool primal_infeasible = false;
    bool dual_infeasible = false;
    Vec y_cert, x_cert; // dy and dx direction certificates

    // Optional polished x
    std::optional<Vec> x_polish;
};

// -------- build M = P + sigma*I + A^T diag(rho) A --------
inline SpMat build_M(const SpMat &P, const SpMat &A, const Vec &rho,
                     Scalar sigma, Scalar diag_reg) {
    const int n = static_cast<int>(P.rows());
    SpMat M = Scalar(0.5) * (P + SpMat(P.transpose()));

    // sigma*I + tiny regularization
    SpMat I(n, n);
    I.setIdentity();
    M += (sigma + diag_reg) * I;

    if (A.rows() > 0) {
        SpMat As = create_scaled_A(A, rho);
        M += As.transpose() * As;
    }
    M.makeCompressed();
    return M;
}

// -------- residuals & thresholds (original space) --------
struct Termination {
    Scalar eps_pri = 0.0;
    Scalar eps_dua = 0.0;
};

inline Termination compute_thresholds(const SpMat &P, const SpMat &A,
                                      const Vec &x, const Vec &z, const Vec &q,
                                      const Vec &y, const Settings &cfg) {
    Vec Ax, ATy, Pxq;
    if (A.rows() > 0) {
        Ax = A * x;
        ATy = A.transpose() * y;
    } else {
        Ax.resize(0);
        ATy = Vec::Zero(x.size());
    }
    Pxq = P * x + q;

    const Scalar eps_pri =
        cfg.eps_abs + cfg.eps_rel * std::max(inf_norm(Ax), inf_norm(z));
    const Scalar eps_dua =
        cfg.eps_abs + cfg.eps_rel * std::max(inf_norm(Pxq), inf_norm(ATy));
    return {eps_pri, eps_dua};
}

inline Residuals compute_residuals(const SpMat &P, const SpMat &A, const Vec &x,
                                   const Vec &z, const Vec &q, const Vec &y) {
    Vec r_p, ATy;
    if (A.rows() > 0) {
        r_p = A * x - z;
        ATy = A.transpose() * y;
    } else {
        r_p.resize(0);
        ATy = Vec::Zero(x.size());
    }
    Vec r_d = P * x + q + ATy;
    return {inf_norm(r_p), inf_norm(r_d)};
}

// =============================== Ruiz scaling ===============================
struct Scaling {
    // x̄ = D^{-1} x,  z̄ = E z,  ȳ = c E^{-1} y
    Vec Dx; // length n (positive)
    Vec Ez; // length m (positive)
    Scalar c = 1.0;
    bool enabled = false;

    void reset(int n, int m) {
        Dx = Vec::Ones(n);
        Ez = Vec::Ones(m);
        c = 1.0;
        enabled = false;
    }
};

inline Scalar col_inf_norm(const SpMat &M, int j) {
    Scalar v = 0.0;
    for (SpMat::InnerIterator it(M, j); it; ++it)
        v = std::max(v, std::abs(it.value()));
    return v;
}

inline Scalar row_inf_norm(const SpMat &M, int i) {
    Scalar v = 0.0;
    for (int k = 0; k < M.outerSize(); ++k)
        for (SpMat::InnerIterator it(M, k); it; ++it)
            if (it.row() == i)
                v = std::max(v, std::abs(it.value()));
    return v;
}

// Modified Ruiz equilibration (block-wise; paper Alg. 2)
inline void ruiz_equilibrate_modified(const SpMat &P, const Vec &q,
                                      const SpMat &A, const Vec &l,
                                      const Vec &u, int max_iter, Scalar tol,
                                      SpMat &Pbar, Vec &qbar, SpMat &Abar,
                                      Vec &lbar, Vec &ubar, Scaling &S) {
    const int n = int(P.rows());
    const int m = int(A.rows());

    S.reset(n, m);
    Pbar = P;
    Abar = A;
    qbar = q;
    lbar = l;
    ubar = u;

    Vec delta_x = Vec::Ones(n);
    Vec delta_z = Vec::Ones(m);

    for (int it = 0; it < max_iter; ++it) {
        Scalar max_dev = 0.0;

        // Columns for x-block (P cols and A cols)
        for (int j = 0; j < n; ++j) {
            Scalar nP = col_inf_norm(Pbar, j);
            Scalar nA = col_inf_norm(Abar, j);
            Scalar s = std::max(nP, nA);
            Scalar dj = (s > 0) ? 1.0 / std::sqrt(s) : 1.0;
            delta_x[j] = dj;
            max_dev = std::max(max_dev, std::abs(1.0 - dj));
        }

        // Rows for z-block = rows of A
        for (int i = 0; i < m; ++i) {
            Scalar nAr = row_inf_norm(Abar, i);
            Scalar di = (nAr > 0) ? 1.0 / std::sqrt(nAr) : 1.0;
            delta_z[i] = di;
            max_dev = std::max(max_dev, std::abs(1.0 - di));
        }

        // Apply: P̄ ← D P̄ D;  Ā ← E Ā D;  q̄ ← D q̄;  l̄,ū ← E l̄, E ū
        scale_cols_inplace(Pbar, delta_x);
        scale_rows_inplace(Pbar, delta_x);
        // Abar rows (E), then cols (D)
        scale_rows_inplace(Abar, delta_z);
        scale_cols_inplace(Abar, delta_x);
        // qbar
        qbar = delta_x.cwiseProduct(qbar);
        // bounds
        if (lbar.size())
            lbar = delta_z.cwiseProduct(lbar);
        if (ubar.size())
            ubar = delta_z.cwiseProduct(ubar);

        // Accumulate
        S.Dx = S.Dx.cwiseProduct(delta_x);
        S.Ez = S.Ez.cwiseProduct(delta_z);

        // Cost scaling γ
        Scalar meanPcol = 0.0;
        if (n) {
            Scalar sum = 0.0;
            for (int j = 0; j < n; ++j)
                sum += col_inf_norm(Pbar, j);
            meanPcol = sum / n;
        }
        Scalar qinf = (qbar.size() ? qbar.lpNorm<Eigen::Infinity>() : 1.0);
        Scalar gamma = 1.0 / std::max(meanPcol, qinf);
        if (!std::isfinite(gamma) || gamma <= 0)
            gamma = 1.0;

        Pbar *= gamma;
        qbar *= gamma;
        S.c *= gamma;

        if (max_dev <= tol)
            break;
    }

    S.enabled = true;
}

// Map initial guesses to scaled space
inline void map_initial_to_scaled(const Scaling &S, Vec &x0, Vec &z0, Vec &y0) {
    if (!S.enabled)
        return;
    if (x0.size())
        x0 = x0.cwiseQuotient(S.Dx); // x̄0 = D^{-1} x0
    if (z0.size())
        z0 = S.Ez.cwiseProduct(z0); // z̄0 = E z0
    if (y0.size())
        y0 = (S.c) * y0.cwiseQuotient(S.Ez); // ȳ0 = c E^{-1} y0
}

// Map solution back to original space
inline void map_solution_from_scaled(const Scaling &S, Vec &x, Vec &z, Vec &y) {
    if (!S.enabled)
        return;
    if (x.size())
        x = S.Dx.cwiseProduct(x); // x = D x̄
    if (z.size())
        z = z.cwiseQuotient(S.Ez); // z = E^{-1} z̄
    if (y.size())
        y = (1.0 / S.c) * S.Ez.cwiseProduct(y); // y = c^{-1} E ȳ
}

// Unscaled thresholds from scaled iterates (paper §5.1)
inline Termination compute_thresholds_unscaled(
    const SpMat &Pbar, const SpMat &Abar, const Vec &xbar, const Vec &zbar,
    const Vec &qbar, const Vec &ybar, const Scaling &S, const Settings &cfg) {
    Vec Ax_bar = (Abar.rows() ? Abar * xbar : Vec());
    Scalar term_pri = 0.0;
    if (zbar.size()) {
        Scalar tA =
            (Ax_bar.size()
                 ? (Ax_bar.cwiseQuotient(S.Ez)).lpNorm<Eigen::Infinity>()
                 : 0.0);
        Scalar tz = (zbar.cwiseQuotient(S.Ez)).lpNorm<Eigen::Infinity>();
        term_pri = std::max(tA, tz);
    }

    Vec Px_bar = (Pbar.rows() ? Pbar * xbar : Vec());
    Vec ATy_bar = (Abar.rows() ? Abar.transpose() * ybar : Vec());
    Scalar t1 =
        (Px_bar.size() ? (Px_bar.cwiseQuotient(S.Dx)).lpNorm<Eigen::Infinity>()
                       : 0.0);
    Scalar t2 = (ATy_bar.size()
                     ? (ATy_bar.cwiseQuotient(S.Dx)).lpNorm<Eigen::Infinity>()
                     : 0.0);
    Scalar t3 =
        (qbar.size() ? (qbar.cwiseQuotient(S.Dx)).lpNorm<Eigen::Infinity>()
                     : 0.0);
    Scalar term_dua = (1.0 / S.c) * std::max({t1, t2, t3});

    return {cfg.eps_abs + cfg.eps_rel * term_pri,
            cfg.eps_abs + cfg.eps_rel * term_dua};
}

// Unscaled residuals from scaled vars (paper §5.1)
inline Residuals compute_residuals_unscaled(const SpMat &Pbar,
                                            const SpMat &Abar, const Vec &xbar,
                                            const Vec &zbar, const Vec &qbar,
                                            const Vec &ybar, const Scaling &S) {
    Vec r_p, r_d;
    if (Abar.rows()) {
        Vec tmp = Abar * xbar - zbar;
        r_p = tmp.cwiseQuotient(S.Ez);
    } else
        r_p.resize(0);

    Vec Px = (Pbar.rows() ? Pbar * xbar : Vec());
    Vec ATy = (Abar.rows() ? Abar.transpose() * ybar : Vec());
    Vec sum = Vec::Zero(xbar.size());
    if (Px.size())
        sum += Px;
    if (qbar.size())
        sum += qbar;
    if (ATy.size())
        sum += ATy;
    r_d = (1.0 / S.c) * sum.cwiseQuotient(S.Dx);

    return {inf_norm(r_p), inf_norm(r_d)};
}

// ======================= Main Sparse OSQP-like Solver =======================
class SparseOSQPSolver {
public:
    explicit SparseOSQPSolver(Settings s = Settings{}) : cfg_(s) {
        // back-compat: if user set Settings.rho (deprecated), propagate to rho0
        cfg_.rho0 = cfg_.rho;
    }

    // Solve:
    //   minimize 0.5 x^T P x + q^T x
    //   s.t.     l <= A x <= u
    Result solve(const SpMat &P, const Vec &q, const SpMat &A, const Vec &l,
                 const Vec &u, const Vec *x0 = nullptr, const Vec *z0 = nullptr,
                 const Vec *y0 = nullptr) {
        const int n = static_cast<int>(P.rows());
        const int m = static_cast<int>(A.rows());

        // ---- Dimension checks
        if (P.cols() != n)
            throw std::invalid_argument("P must be square");
        if (q.size() != n)
            throw std::invalid_argument("q dimension mismatch");
        if (A.cols() != n)
            throw std::invalid_argument("A column dim mismatch");
        if (l.size() != m || u.size() != m)
            throw std::invalid_argument("l,u dim mismatch");

        // ---- Prepare (possibly scaled) problem
        SpMat P_use = P, A_use = A;
        Vec q_use = q, l_use = l, u_use = u;

        // Back-compat: allow deprecated Settings.rho to seed rho0
        Settings cfg = cfg_;
        cfg.rho0 = cfg_.rho;

        // ---- Optional modified Ruiz scaling
        Scaling S;
        S.reset(n, m);
        if (cfg.enable_ruiz) {
            ruiz_equilibrate_modified(P, q, A, l, u, cfg.ruiz_max_iter,
                                      cfg.ruiz_tol, P_use, q_use, A_use, l_use,
                                      u_use, S);
        }

        // ---- Initialize scaled iterates (bar-variables)
        Vec xbar = (x0 && x0->size() == n) ? *x0 : Vec::Zero(n);
        Vec zbar, ybar;

        if (S.enabled) {
            if (xbar.size())
                xbar = xbar.cwiseQuotient(S.Dx); // x̄0 = D^{-1} x0
        }
        if (z0 && z0->size() == m) {
            zbar = *z0;
            if (S.enabled)
                zbar = S.Ez.cwiseProduct(zbar); // z̄0 = E z0
            zbar = project_box(zbar, l_use, u_use);
        } else {
            if (m > 0)
                zbar = project_box(A_use * xbar, l_use, u_use);
            else
                zbar.resize(0);
        }
        if (y0 && y0->size() == m) {
            ybar = *y0;
            if (S.enabled)
                ybar = (S.c) * ybar.cwiseQuotient(S.Ez); // ȳ0 = c E^{-1} y0
        } else {
            ybar = (m ? Vec::Zero(m) : Vec());
        }

        // ---- Output holder (final values will be ORIGINAL-space)
        Result out;
        out.status = "max_iter_reached";
        out.iters = 0;

        auto map_finalize_and_return =
            [&](const std::string &status) -> Result {
            Vec x_ret = xbar, z_ret = zbar, y_ret = ybar;
            map_solution_from_scaled(S, x_ret, z_ret, y_ret);
            out.x = x_ret;
            out.z = z_ret;
            out.y = y_ret;
            out.status = status;
            return finalize(out, P, A, q);
        };

        // ---- Per-row rho (equality rows boosted) computed in *scaled* bounds
        Vec rho = Vec::Constant(m, cfg.rho0);
        if (m > 0) {
            for (int i = 0; i < m; ++i) {
                const bool finite_l = std::isfinite(l_use[i]);
                const bool finite_u = std::isfinite(u_use[i]);
                if (finite_l && finite_u &&
                    std::abs(l_use[i] - u_use[i]) <= cfg.eq_tol) {
                    rho[i] = cfg.rho0 * cfg.rho_eq_scale;
                }
            }
        }

        // ---- Choose linear system path
        // If KKT was slower for your workloads, feel free to set use_kkt =
        // false;
        const bool use_kkt = false; // (m > 0) && (n > 80 || m > 80); //
                                   // heuristic; tune or force false

        // Builders / factorizations
        int refactor_count = 0;

        // ---------- Normal-Equations fast cache ----------
        NEFast NEF;
        Scalar rho_weight =
            1.0; // cumulative scaling of rho since NEF.build_once(...)

        // Precompute symmetric P once (for H)
        SpMat P_sym = Scalar(0.5) * (P_use + SpMat(P_use.transpose()));

        auto build_nefast = [&](int refc) -> bool {
            Scalar bump = std::pow(10.0, refc);
            return NEF.build_once(P_sym, cfg.sigma, cfg.diag_reg * bump, A_use,
                                  rho);
        };

        // ---------- Optional KKT path ----------
        KKTQuasiDef KKT;
        auto build_kkt = [&](int refc) -> bool {
            Scalar bump = std::pow(10.0, refc);
            return KKT.build(P_use, A_use, rho, cfg.sigma, cfg.diag_reg * bump);
        };

        if (use_kkt) {
            // Full rebuild of KKT when rho changes
            refactor_count = 0;
            while (!build_kkt(refactor_count)) {
                if (cfg.verbose)
                    std::cerr << "[sOSQP] KKT rebuild failed; "
                                 "increase diag_reg\n";
                if (++refactor_count > cfg.max_refactor)
                    return map_finalize_and_return("factorization_failed");
            }
        } else {
            while (!build_nefast(refactor_count)) {
                if (cfg.verbose)
                    std::cerr << "[sOSQP] NE build failed; increase diag_reg\n";
                if (++refactor_count > cfg.max_refactor)
                    return map_finalize_and_return("factorization_failed");
            }
        }

        // ---- Main loop variables (scaled space)
        Vec x_prev = xbar, z_prev2 = zbar, y_prev = ybar;
        bool primal_inf_flag = false, dual_inf_flag = false;
        Vec dy_cert, dx_cert;

        // Stagnation trackers (for restoration)
        Scalar best_pri = std::numeric_limits<Scalar>::infinity();
        Scalar best_dua = std::numeric_limits<Scalar>::infinity();
        int stall_cnt = 0;

        constexpr Scalar kRhoFloor = 1e-30;

        for (int k = 0; k < cfg.max_iter; ++k) {
            // ===== x-update (either NEFast or KKT)
            Vec x_new;
            if (use_kkt) {
                Vec rhs1 = -q_use + cfg.sigma * xbar;
                if (m > 0)
                    rhs1 += A_use.transpose() * (rho.cwiseProduct(zbar) - ybar);
                Vec rhs2 = (m > 0 ? zbar : Vec());
                x_new = KKT.solve(rhs1, rhs2).first; // keep ADMM y-update below
            } else {
                Vec rhs = -q_use + cfg.sigma * xbar;
                if (m > 0)
                    rhs += A_use.transpose() * (rho.cwiseProduct(zbar) - ybar);
                x_new = NEF.solve(rhs);
#if 1
                Vec r_lin = rhs - NEF.M * x_new;
                if (r_lin.lpNorm<Eigen::Infinity>() > 0) {
                    x_new += NEF.solve(r_lin); // 1 correction step
                }
#endif
            }

            // ===== Over-relaxation and (z,y) updates (scaled)
            Vec zt = (m > 0 ? A_use * x_new : Vec()); // Ā x̄^{k+1}
            Vec x_hat = cfg.alpha * x_new + (1.0 - cfg.alpha) * xbar;

            Vec v; // for z update
            if (m > 0) {
                v = cfg.alpha * zt + (1.0 - cfg.alpha) * zbar +
                    ybar.cwiseQuotient(rho.cwiseMax(kRhoFloor));
            }

            Vec z_new = (m > 0 ? project_box(v, l_use, u_use) : Vec());
            Vec y_new =
                (m > 0
                     ? ybar + rho.cwiseProduct(cfg.alpha * zt +
                                               (1.0 - cfg.alpha) * zbar - z_new)
                     : Vec());

            // Roll
            xbar = x_hat;
            zbar = z_new;
            ybar = y_new;

            // ===== Check termination / certificates / adaptive rho
            const bool do_check =
                ((k % cfg.check_every) == 0) || (k == cfg.max_iter - 1);
            if (!do_check)
                continue;

            Residuals r;
            Termination t;
            if (S.enabled && cfg.check_unscaled) {
                r = compute_residuals_unscaled(P_use, A_use, xbar, zbar, q_use,
                                               ybar, S);
                t = compute_thresholds_unscaled(P_use, A_use, xbar, zbar, q_use,
                                                ybar, S, cfg);
            } else {
                r = compute_residuals(P_use, A_use, xbar, zbar, q_use, ybar);
                t = compute_thresholds(P_use, A_use, xbar, zbar, q_use, ybar,
                                       cfg);
            }

            if (cfg.verbose) {
                std::cerr << "[sOSQP] it " << (k + 1) << "  r_pri " << r.pri_inf
                          << " (eps " << t.eps_pri << ")"
                          << "  r_dual " << r.dua_inf << " (eps " << t.eps_dua
                          << ")\n";
            }

            if (r.pri_inf <= t.eps_pri && r.dua_inf <= t.eps_dua) {
                // Map back and finalize as SOLVED
                Vec x_ret = xbar, z_ret = zbar, y_ret = ybar;
                map_solution_from_scaled(S, x_ret, z_ret, y_ret);
                out.status = "solved";
                out.iters = k + 1;
                out.x = x_ret;
                out.z = z_ret;
                out.y = y_ret;
                out.res = compute_residuals(P, A, out.x, out.z, q, out.y);
                out.obj_val = 0.5 * out.x.dot(P * out.x) + q.dot(out.x);
                if (cfg.polish)
                    try_polish(out, P, q, A, l, u);
                out.primal_infeasible = primal_inf_flag;
                out.dual_infeasible = dual_inf_flag;
                out.y_cert = dy_cert;
                out.x_cert = dx_cert;
                return out;
            }

            // --- Stagnation / restoration heuristic (paper-inspired) ---
            bool improved =
                (r.pri_inf < 0.9 * best_pri) || (r.dua_inf < 0.9 * best_dua);
            best_pri = std::min(best_pri, r.pri_inf);
            best_dua = std::min(best_dua, r.dua_inf);
            stall_cnt = improved ? 0 : (stall_cnt + 1);

            const bool bad_numeric =
                (!std::isfinite(r.pri_inf) || !std::isfinite(r.dua_inf));
            if (bad_numeric || stall_cnt >= 3) {
                // Soft restart: boost rho, reset dual, recenter z, cheap
                // refactor
                const Scalar rho_boost = 2.0; // tune 1.5–5.0
                if (m > 0) {
                    rho *= rho_boost;
                    rho = rho.cwiseMax(cfg.rho_min).cwiseMin(cfg.rho_max);
                }

                // Reset dual; recenter z on current Ax
                ybar.setZero();
                if (m > 0) {
                    Vec Ax_now = A_use * xbar;
                    zbar = project_box(Ax_now, l_use, u_use);
                }

                // Cheap refactorization of linear system
                if (!use_kkt) {
                    // NEFast: only change the weight and do numeric refactor
                    rho_weight *= rho_boost;

                    refactor_count = 0;
                    while (!NEF.refactor_with_weight(rho_weight)) {
                        if (cfg.verbose)
                            std::cerr << "[sOSQP] NE restore refactor failed; "
                                         "bump diag_reg\n";
                        if (++refactor_count > cfg.max_refactor)
                            return map_finalize_and_return(
                                "factorization_failed");

                        // Rebuild once with higher reg using CURRENT rho as new
                        // base
                        if (!NEF.build_once(P_sym, cfg.sigma,
                                            cfg.diag_reg *
                                                std::pow(10.0, refactor_count),
                                            A_use, rho)) {
                            continue; // try larger bump
                        }
                        rho_weight =
                            1.0; // reset because G now reflects current rho
                        break;
                    }
                } else {
                    // KKT path: rebuild
                    refactor_count = 0;
                    if (!KKT.update_rho(rho)) {
                        refactor_count = 0;
                        while (!build_kkt(refactor_count)) {
                            if (cfg.verbose)
                                std::cerr
                                    << "[sOSQP] KKT restore rebuild failed; "
                                       "bump diag_reg\n";
                            if (++refactor_count > cfg.max_refactor)
                                return map_finalize_and_return(
                                    "factorization_failed");
                        }
                    }
                }

                stall_cnt = 0; // reset window
                continue; // skip certs/ρ update this check and iterate again
            }

            // ---- Infeasibility certificates (ORIGINAL space)
            Vec dx_bar = xbar - x_prev;
            Vec dz_bar = zbar - z_prev2;
            Vec dy_bar = ybar - y_prev;

            Vec dx_orig = dx_bar;
            Vec dy_orig = dy_bar;
            if (S.enabled) {
                dx_orig = S.Dx.cwiseProduct(dx_bar);
                dy_orig = (1.0 / S.c) * S.Ez.cwiseProduct(dy_bar);
            }

            x_prev = xbar;
            z_prev2 = zbar;
            y_prev = ybar;

            if (m > 0) {
                Vec ATdy = A.transpose() * dy_orig;
                const Scalar dy_norm =
                    std::max<Scalar>(1e-30, inf_norm(dy_orig));
                bool pinf1 = (inf_norm(ATdy) <= cfg.eps_pinf * dy_norm);
                Scalar s2 = 0.0;
                for (int i = 0; i < m; ++i) {
                    Scalar ui = std::isfinite(u[i]) ? u[i] : 0.0;
                    Scalar li = std::isfinite(l[i]) ? l[i] : 0.0;
                    const Scalar pos = std::max(dy_orig[i], 0.0);
                    const Scalar neg = std::min(dy_orig[i], 0.0);
                    s2 += ui * pos + li * neg;
                }
                bool pinf2 = (s2 <= cfg.eps_pinf * dy_norm);
                if (pinf1 && pinf2) {
                    primal_inf_flag = true;
                    dy_cert = dy_orig;
                }
            }

            {
                const Scalar dx_norm =
                    std::max<Scalar>(1e-30, inf_norm(dx_orig));
                Vec Pdx = P * dx_orig;
                bool dinf1 = (inf_norm(Pdx) <= cfg.eps_dinf * dx_norm);
                bool dinf2 = (q.dot(dx_orig) <= cfg.eps_dinf * dx_norm);

                bool dinf3 = true;
                if (m > 0) {
                    Vec Adx = A * dx_orig;
                    for (int i = 0; i < m; ++i) {
                        if (std::isfinite(l[i]) && std::isfinite(u[i])) {
                            if (std::fabs(Adx[i]) > cfg.eps_dinf * dx_norm) {
                                dinf3 = false;
                                break;
                            }
                        } else if (!std::isfinite(u[i])) { // only lower bound
                            if (Adx[i] < -cfg.eps_dinf * dx_norm) {
                                dinf3 = false;
                                break;
                            }
                        } else if (!std::isfinite(l[i])) { // only upper bound
                            if (Adx[i] > cfg.eps_dinf * dx_norm) {
                                dinf3 = false;
                                break;
                            }
                        }
                    }
                }
                if (dinf1 && dinf2 && dinf3) {
                    dual_inf_flag = true;
                    dx_cert = dx_orig;
                }
            }

            if (primal_inf_flag || dual_inf_flag) {
                Vec x_ret = xbar, z_ret = zbar, y_ret = ybar;
                map_solution_from_scaled(S, x_ret, z_ret, y_ret);
                out.status =
                    primal_inf_flag ? "primal_infeasible" : "dual_infeasible";
                out.iters = k + 1;
                out.x = x_ret;
                out.z = z_ret;
                out.y = y_ret;
                out.res = compute_residuals(P, A, out.x, out.z, q, out.y);
                out.obj_val = 0.5 * out.x.dot(P * out.x) + q.dot(out.x);
                out.primal_infeasible = primal_inf_flag;
                out.dual_infeasible = dual_inf_flag;
                out.y_cert = dy_cert;
                out.x_cert = dx_cert;
                return out;
            }

            // ---- Adaptive rho (global rescale). Cheap refactor with NEFast.
            if (cfg.adaptive_rho && m > 0) {
                Scalar npri = 0.0, ndua = 0.0;

                if (S.enabled && cfg.check_unscaled) {
                    Vec x_tmp = xbar, z_tmp = zbar, y_tmp = ybar;
                    map_solution_from_scaled(S, x_tmp, z_tmp, y_tmp);
                    const Scalar denom_pri = std::max<Scalar>(
                        1e-30, std::max(inf_norm(A * x_tmp), inf_norm(z_tmp)));
                    Vec Pxq = P * x_tmp + q;
                    const Scalar denom_dua = std::max<Scalar>(
                        1e-30, std::max(inf_norm(Pxq),
                                        inf_norm(A.transpose() * y_tmp)));
                    npri = r.pri_inf / denom_pri;
                    ndua = r.dua_inf / denom_dua;
                } else {
                    const Scalar denom_pri =
                        std::max<Scalar>(1e-30, std::max(inf_norm(A_use * xbar),
                                                         inf_norm(zbar)));
                    Vec Pxq = P_use * xbar + q_use;
                    const Scalar denom_dua = std::max<Scalar>(
                        1e-30, std::max(inf_norm(Pxq),
                                        inf_norm(A_use.transpose() * ybar)));
                    npri = r.pri_inf / denom_pri;
                    ndua = r.dua_inf / denom_dua;
                }

                if (ndua > 0) {
                    const Scalar ratio = std::sqrt(npri / ndua);
                    if (ratio >= 5.0 || ratio <= 0.2) {
                        const Scalar scale =
                            std::clamp(ratio, Scalar(0.2), Scalar(5.0));

                        // Update algorithmic rho vector
                        rho *= scale;
                        rho = rho.cwiseMax(cfg.rho_min).cwiseMin(cfg.rho_max);

                        if (use_kkt) {
                            // Full rebuild of KKT when rho changes
                            refactor_count = 0;
                            if (!KKT.update_rho(rho)) {
                                refactor_count = 0;
                                while (!build_kkt(refactor_count)) {
                                    if (cfg.verbose)
                                        std::cerr << "[sOSQP] KKT restore "
                                                     "rebuild failed; "
                                                     "bump diag_reg\n";
                                    if (++refactor_count > cfg.max_refactor)
                                        return map_finalize_and_return(
                                            "factorization_failed");
                                }
                            }
                        } else {
                            // Numeric-only refactor via updated weight
                            rho_weight *= scale;

                            refactor_count = 0;
                            while (!NEF.refactor_with_weight(rho_weight)) {
                                if (cfg.verbose)
                                    std::cerr << "[sOSQP] NE refactor failed; "
                                                 "increasing regularization\n";
                                if (++refactor_count > cfg.max_refactor)
                                    return map_finalize_and_return(
                                        "factorization_failed");

                                // Rebuild once with higher regularization,
                                // using CURRENT rho as the new base
                                if (!NEF.build_once(
                                        P_sym, cfg.sigma,
                                        cfg.diag_reg *
                                            std::pow(10.0, refactor_count),
                                        A_use, rho)) {
                                    continue; // try larger bump
                                }
                                rho_weight = 1.0; // base reset
                                break;
                            }
                        }
                    }
                }
            }
        }

        // ---- Max iters reached: map back and finalize
        {
            Vec x_ret = xbar, z_ret = zbar, y_ret = ybar;
            map_solution_from_scaled(S, x_ret, z_ret, y_ret);
            out.x = x_ret;
            out.z = z_ret;
            out.y = y_ret;
            out.res = compute_residuals(P, A, out.x, out.z, q, out.y);
            out.obj_val = 0.5 * out.x.dot(P * out.x) + q.dot(out.x);
            return out;
        }
    }

private:
    Settings cfg_;

    Result &finalize(Result &out, const SpMat &P, const SpMat &A,
                     const Vec &q) const {
        out.res = compute_residuals(P, A, out.x, out.z, q, out.y);
        out.obj_val = 0.5 * out.x.dot(P * out.x) + q.dot(out.x);
        return out;
    }

    // -------- Polishing: quasi-definite KKT with delta, solve via SparseLU +
    // refinement
    // -------- Polishing: reduced KKT with active-set detection + refinement
    void try_polish(Result &out, const SpMat &P, const Vec &q, const SpMat &A,
                    const Vec &l, const Vec &u) const {
        const int n = int(P.rows());
        const int m = int(A.rows());
        if (m == 0)
            return;
        if (out.x.size() != n || out.z.size() != m || out.y.size() != m)
            return;

        // ---- Active set selection
        // Include a row in L if: y_i < 0 OR (z_i is near l_i)
        // Include a row in U if: y_i > 0 OR (z_i is near u_i)
        // Note: if both sides hit (rare, due to eq rows), prefer sign(y).
        const Scalar tau_act = 1e-6; // proximity tolerance to bounds
        std::vector<int> L, U;
        L.reserve(m);
        U.reserve(m);

        for (int i = 0; i < m; ++i) {
            const bool hasL = std::isfinite(l[i]);
            const bool hasU = std::isfinite(u[i]);

            const bool nearL = hasL && (out.z[i] - l[i] <= tau_act);
            const bool nearU = hasU && (u[i] - out.z[i] <= tau_act);

            if (out.y[i] < 0) {
                L.push_back(i);
            } else if (out.y[i] > 0) {
                U.push_back(i);
            } else {
                if (nearL && !nearU)
                    L.push_back(i);
                else if (nearU && !nearL)
                    U.push_back(i);
                // if neither, leave inactive
            }
        }

        if (L.empty() && U.empty())
            return; // nothing to polish

        // ---- Helper: take selected rows from A (keeps sparse structure)
        auto take_rows = [&](const SpMat &Mx,
                             const std::vector<int> &idx) -> SpMat {
            if (idx.empty())
                return SpMat(0, Mx.cols());
            std::vector<int> sorted = idx;
            std::sort(sorted.begin(), sorted.end());
            SpMat R(int(sorted.size()), Mx.cols());
            std::vector<Triplet> T;
            T.reserve(size_t(Mx.nonZeros()) * 1u); // rough upper bound
            for (int k = 0; k < Mx.outerSize(); ++k) {
                for (SpMat::InnerIterator it(Mx, k); it; ++it) {
                    int r = it.row();
                    auto itp =
                        std::lower_bound(sorted.begin(), sorted.end(), r);
                    if (itp != sorted.end() && *itp == r) {
                        int rr = int(itp - sorted.begin());
                        T.emplace_back(rr, it.col(), it.value());
                    }
                }
            }
            R.setFromTriplets(T.begin(), T.end());
            R.makeCompressed();
            return R;
        };

        // ---- Build reduced KKT
        std::vector<int> Ls = L, Us = U;
        std::sort(Ls.begin(), Ls.end());
        std::sort(Us.begin(), Us.end());
        SpMat AL = take_rows(A, Ls);
        SpMat AU = take_rows(A, Us);

        // Symmetrize P and add small delta on x-block
        SpMat Pp = Scalar(0.5) * (P + SpMat(P.transpose()));
        SpMat I_n(n, n);
        I_n.setIdentity();
        const Scalar delta = cfg_.polish_delta;
        Pp += delta * I_n;

        // Assemble:
        // [  Pp      AL^T     AU^T ] [x]   [ -q ]
        // [  AL     -delta I   0   ] [λL]  [  l_L ]
        // [  AU       0      -delta I] [λU] [  u_U ]
        const int nL = int(Ls.size());
        const int nU = int(Us.size());
        const int nK = n + nL + nU;

        std::vector<Triplet> T;
        T.reserve(size_t(Pp.nonZeros()) + size_t(AL.nonZeros()) * 2 +
                  size_t(AU.nonZeros()) * 2 + size_t(nL + nU));

        // Top-left: Pp
        for (int k = 0; k < Pp.outerSize(); ++k)
            for (SpMat::InnerIterator it(Pp, k); it; ++it)
                T.emplace_back(it.row(), it.col(), it.value());

        // Top-right: A_L^T at (0, n), A_U^T at (0, n+nL)
        auto add_block_AT = [&](const SpMat &B, int row_off, int col_off) {
            for (int k = 0; k < B.outerSize(); ++k)
                for (SpMat::InnerIterator it(B, k); it; ++it)
                    T.emplace_back(row_off + it.col(), col_off + it.row(),
                                   it.value()); // B^T
        };
        const int offL = n;
        const int offU = n + nL;
        add_block_AT(AL, 0, offL);
        add_block_AT(AU, 0, offU);

        // Middle-left: A_L at (n,0) and -delta*I at (n,n)
        for (int k = 0; k < AL.outerSize(); ++k)
            for (SpMat::InnerIterator it(AL, k); it; ++it)
                T.emplace_back(offL + it.row(), it.col(), it.value());
        for (int i = 0; i < nL; ++i)
            T.emplace_back(offL + i, offL + i, -delta);

        // Bottom-left: A_U at (n+nL,0) and -delta*I at (n+nL, n+nL)
        for (int k = 0; k < AU.outerSize(); ++k)
            for (SpMat::InnerIterator it(AU, k); it; ++it)
                T.emplace_back(offU + it.row(), it.col(), it.value());
        for (int i = 0; i < nU; ++i)
            T.emplace_back(offU + i, offU + i, -delta);

        SpMat KKT(nK, nK);
        KKT.setFromTriplets(T.begin(), T.end());
        KKT.makeCompressed();

        // RHS
        Vec rhs(nK);
        rhs.setZero();
        rhs.head(n) = -q;
        for (int i = 0; i < nL; ++i)
            rhs[offL + i] = l[Ls[i]];
        for (int i = 0; i < nU; ++i)
            rhs[offU + i] = u[Us[i]];

        // ---- Factor & solve (SparseLU is robust for quasi-definite with small
        // -delta)
        Eigen::SparseLU<SpMat> slu;
        slu.analyzePattern(KKT);
        slu.factorize(KKT);
        if (slu.info() != Eigen::Success)
            return;

        Vec sol = slu.solve(rhs);
        if (slu.info() != Eigen::Success || !sol.allFinite())
            return;

        // ---- Iterative refinement (a few steps)
        for (int t = 0; t < cfg_.polish_refine_steps; ++t) {
            Vec r = rhs - KKT * sol;
            if (!r.allFinite())
                break;
            const Scalar r_inf = r.lpNorm<Eigen::Infinity>();
            if (r_inf <= 1e-12)
                break;
            Vec d = slu.solve(r);
            if (slu.info() != Eigen::Success || !d.allFinite())
                break;
            sol += d;
        }

        // ---- Extract polished x and check
        Vec xhat = sol.head(n);
        Vec zhat = (m > 0 ? A * xhat : Vec());
        Vec zproj = project_box(zhat, l, u);

        Residuals r_pol = compute_residuals(P, A, xhat, zproj, q, out.y);
        Termination t_pol =
            compute_thresholds(P, A, xhat, zproj, q, out.y, cfg_);

        // Accept if meets tolerances and (optionally) improves objective
        if (r_pol.pri_inf <= t_pol.eps_pri && r_pol.dua_inf <= t_pol.eps_dua) {
            out.x_polish = xhat;
            Scalar obj_pol = 0.5 * xhat.dot(P * xhat) + q.dot(xhat);
            if (std::isfinite(obj_pol)) {
                if (!std::isfinite(out.obj_val) ||
                    obj_pol <= out.obj_val + 1e-12) {
                    out.obj_val = obj_pol;
                }
            }
        }
    }

    // -------- box projection --------
    static Vec project_box(const Vec &v, const Vec &l, const Vec &u) {
        if (v.size() == 0)
            return v;
        Vec out = v;
        for (int i = 0; i < out.size(); ++i) {
            if (std::isfinite(l[i]))
                out[i] = std::max(out[i], l[i]);
            if (std::isfinite(u[i]))
                out[i] = std::min(out[i], u[i]);
        }
        return out;
    }
};

} // namespace sosqp
