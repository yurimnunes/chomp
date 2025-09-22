#pragma once
// Schur KKT solver using CG with lanes-backed HVP.
// - By default uses a robust single-RHS CG (cg_single) that calls only mv (HVP).
// - If KKT_USE_BLOCK_CG is defined, multi-RHS block CG (block_cg) is used where helpful.
//
// Requirements from your AD/bindings (WOp is provided elsewhere):
//   LagHessFn::set_state_eigen(x,lam,nu)
//   LagHessFn::hvp_into_nogil(const double* v, double* y)       // single-RHS
//   LagHessFn::hvp_multi_into(const dmat& V, dmat& Y)           // block-RHS

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include <omp.h>

#include "kkt_core.h" // dvec, dmat, spmat, ChompConfig, cg(), LinOp, SSORPrecond, helpers like diag_GtG, schur_diag_hat, build_S_hat
//#define KKT_USE_BLOCK_CG
namespace kkt {

// --------------------------- Linear operator glue ----------------------------

struct BlockLinOp {
    int n = 0;

    // Single-RHS y = A x
    std::function<void(const dvec&, dvec&)> mv;

    // Optional: Block-RHS Y = A X (n×k)
    std::function<void(const dmat&, dmat&)> bmv;

    void apply(const dvec& x, dvec& y) const { mv(x, y); }

    void apply_block(const dmat& X, dmat& Y) const {
        if (bmv) { bmv(X, Y); return; }
        const int nloc = static_cast<int>(X.rows());
        const int k    = static_cast<int>(X.cols());
        if (Y.rows()!=nloc || Y.cols()!=k) Y.resize(nloc, k);
        dvec yj(nloc);
        for (int j = 0; j < k; ++j) {
            mv(X.col(j), yj);
            Y.col(j).noalias() = yj;
        }
    }
};

// ------------------------------- Helpers -------------------------------------

// BLAS-3 Gram helper: k×k = AᵀB
inline dmat gramT(const dmat& A, const dmat& B) { return A.transpose() * B; }

// Scale-aware SPD “jitter” (keeps LLT stable in finite precision)
inline void add_jitter_spd(dmat& M) {
    M = 0.5 * (M + M.transpose().eval());
    const double t = M.diagonal().array().abs().mean();
    M.diagonal().array() += std::max(1e-14 * std::max(1.0, t), 1e-18);
}

// --------------------------- Simple (single-RHS) CG --------------------------

struct CGInfoBlock {
    int    iters          = 0;
    bool   converged      = false;
    double max_rel_resid  = std::numeric_limits<double>::infinity();
};

struct ScalarJacobi {
    dvec inv_diag; // 1./diag(A) or any robust positive diagonal
    void apply(const dvec& r, dvec& z) const {
        if (z.size()!=r.size()) z.resize(r.size());
        z.array() = r.array() * inv_diag.array();
    }
};

inline std::pair<dvec, CGInfoBlock>
cg_single(const BlockLinOp& A,
          const dvec& b,
          const ScalarJacobi& Minv,
          double tol, int maxit,
          const std::optional<dvec>& x0 = std::nullopt)
{
    CGInfoBlock info;
    const int n = static_cast<int>(b.size());
    if (n==0) return {dvec(), info};

    dvec x = x0 ? *x0 : dvec::Zero(n);
    dvec Ax(n), r(n), z(n), p(n), Ap(n);

    // r = b - A x
    A.apply(x, Ax);
    r.noalias() = b - Ax;

    const double bnorm = b.norm();
    double rnorm = r.norm();
    info.max_rel_resid = (bnorm > 0.0) ? (rnorm / bnorm) : rnorm;
    if (info.max_rel_resid <= tol) { info.converged = true; return {x, info}; }

    // z = M^{-1} r
    Minv.apply(r, z);
    p = z;
    double rz_old = r.dot(z);

    for (int it = 0; it < maxit; ++it) {
        A.apply(p, Ap);

        const double pAp   = std::max(p.dot(Ap), 1e-300);
        const double alpha = rz_old / pAp;

        x.noalias() += alpha * p;
        r.noalias() -= alpha * Ap;

        rnorm = r.norm();
        info.max_rel_resid = (bnorm > 0.0) ? (rnorm / bnorm) : rnorm;
        info.iters = it + 1;
        if (info.max_rel_resid <= tol) { info.converged = true; break; }

        Minv.apply(r, z);
        const double rz_new = r.dot(z);
        const double beta   = rz_new / std::max(rz_old, 1e-300);
        p.noalias() = z + beta * p;
        rz_old = rz_new;
    }

    return {x, info};
}

// ------------------------------ Block CG (opt) -------------------------------

struct BlockJacobi {
    dvec inv_diag; // 1./diag(A) on each row
    void apply(const dvec& r, dvec& z) const {
        if (z.size()!=r.size()) z.resize(r.size());
        z.array() = r.array() * inv_diag.array();
    }
    void apply(const dmat& R, dmat& Z) const {
        const int n = static_cast<int>(R.rows());
        const int k = static_cast<int>(R.cols());
        if (Z.rows()!=n || Z.cols()!=k) Z.resize(n,k);
        Z.noalias() = R;
        Z.array().colwise() *= inv_diag.array();
    }
};

inline std::pair<dmat, CGInfoBlock>
block_cg(const BlockLinOp& A, const dmat& B, const BlockJacobi& Minv,
         double tol, int maxit, const std::optional<dmat>& X0 = std::nullopt,
         bool /*use_simd*/ = true)
{
    CGInfoBlock info;
    const int n  = static_cast<int>(B.rows());
    const int k0 = static_cast<int>(B.cols());
    if (n == 0 || k0 == 0) return {dmat(n, k0), info};

    // Workspaces
    dmat X = X0 ? *X0 : dmat::Zero(n, k0);
    dmat R(n, k0), AX(n, k0), Z, Znew, P, AP;
    dmat RtZ, RtZ_old, RtZ_new, PtAP, alpha, beta;

    // Initial residual
    A.apply_block(X, AX);
    R.noalias() = B - AX;

    // Per-column norms & done flags
    std::vector<double> bnorm(k0), rnorm(k0);
    std::vector<char> done(k0, 0);
    for (int j = 0; j < k0; ++j) {
        bnorm[j] = B.col(j).norm();
        rnorm[j] = R.col(j).norm();
        if ((bnorm[j] == 0.0 && rnorm[j] == 0.0) ||
            (bnorm[j] > 0.0 && rnorm[j] / bnorm[j] <= tol))
            done[j] = 1;
    }

    auto build_active = [&]() {
        std::vector<int> idx; idx.reserve(k0);
        for (int j = 0; j < k0; ++j) if (!done[j]) idx.push_back(j);
        return idx;
    };
    auto pack_cols_inplace = [&](dmat& M, const std::vector<int>& act) {
        const int kk = static_cast<int>(act.size());
        for (int j_new = 0; j_new < kk; ++j_new) {
            const int j_old = act[j_new];
            if (j_old != j_new) M.col(j_new).noalias() = M.col(j_old);
        }
        M.conservativeResize(n, kk);
    };
    auto pack_norms = [&](std::vector<double>& v, const std::vector<int>& act) {
        std::vector<double> w; w.reserve(act.size());
        for (int j : act) w.push_back(v[j]);
        v.swap(w);
    };
    auto max_rel = [&](const std::vector<double>& rn, const std::vector<double>& bn) {
        double mr = 0.0;
        for (int j = 0; j < (int)rn.size(); ++j) {
            const double denom = (bn[j] > 0.0) ? bn[j] : 1.0;
            mr = std::max(mr, rn[j] / denom);
        }
        return rn.empty() ? 0.0 : mr;
    };

    // Initial compaction
    std::vector<int> active = build_active();
    if (active.size() < (size_t)k0) {
        pack_cols_inplace(X,  active);
        pack_cols_inplace(R,  active);
        pack_cols_inplace(AX, active);
        pack_norms(bnorm, active);
        pack_norms(rnorm, active);
    }

    info.max_rel_resid = max_rel(rnorm, bnorm);
    if (rnorm.empty() || info.max_rel_resid <= tol) { info.converged = true; return {X, info}; }

    // Allocate skinny blocks to current kk
    const int kk_init = static_cast<int>(rnorm.size());
    Z.resize(n, kk_init); Znew.resize(n, kk_init);
    P.resize(n, kk_init); AP.resize(n, kk_init);
    RtZ.resize(kk_init, kk_init); RtZ_old.resize(kk_init, kk_init);
    RtZ_new.resize(kk_init, kk_init); PtAP.resize(kk_init, kk_init);
    alpha.resize(kk_init, kk_init); beta.resize(kk_init, kk_init);

    // Precondition: Z = M^{-1} R; P = Z; RtZ = RᵀZ
    Minv.apply(R, Z);
    P.noalias() = Z;
    RtZ.noalias() = gramT(R, Z);

    const int comp_freq = 4;

    for (int it = 0; it < maxit; ++it) {
        // AP = A P
        A.apply_block(P, AP);

        // PtAP = Pᵀ A P  (stabilized SPD)
        PtAP.noalias() = gramT(P, AP);
        add_jitter_spd(PtAP);
        Eigen::LLT<dmat> llt(PtAP);
        alpha.noalias() = llt.solve(RtZ);

        // X += P alpha; R -= AP alpha
        X.noalias() += P * alpha;
        R.noalias() -= AP * alpha;

        // Residual norms
        for (int j = 0; j < (int)rnorm.size(); ++j) rnorm[j] = R.col(j).norm();

        info.max_rel_resid = max_rel(rnorm, bnorm);
        info.iters = it + 1;
        if (info.max_rel_resid <= tol) { info.converged = true; break; }

        // Znew = M^{-1} R;  RtZ_old = RtZ;  RtZ_new = Rᵀ Znew
        Minv.apply(R, Znew);
        RtZ_old = RtZ; add_jitter_spd(RtZ_old);
        RtZ_new.noalias() = gramT(R, Znew);

        // beta from RtZ_old * beta = RtZ_new
        Eigen::LLT<dmat> llt2(RtZ_old);
        beta.noalias() = llt2.solve(RtZ_new);

        // P = Znew + P beta; swap Z <- Znew; RtZ <- RtZ_new
        P.noalias() = Znew + P * beta;
        Z.swap(Znew);
        RtZ.swap(RtZ_new);

        // Early drop of converged columns
        bool need_compact = false;
        std::vector<int> keep_idx; keep_idx.reserve(rnorm.size());
        std::vector<double> b2, r2; b2.reserve(rnorm.size()); r2.reserve(rnorm.size());
        for (int j = 0; j < (int)rnorm.size(); ++j) {
            const double denom = (bnorm[j] > 0.0) ? bnorm[j] : 1.0;
            if (rnorm[j] / denom > tol) { keep_idx.push_back(j); b2.push_back(bnorm[j]); r2.push_back(rnorm[j]); }
            else need_compact = true;
        }
        if (need_compact && (it % comp_freq == 0 || (int)keep_idx.size() < (int)rnorm.size())) {
            auto pack_to_keep = [&](dmat& M) {
                const int kk = (int)keep_idx.size();
                for (int j_new = 0; j_new < kk; ++j_new) {
                    int j_old = keep_idx[j_new];
                    if (j_old != j_new) M.col(j_new).noalias() = M.col(j_old);
                }
                M.conservativeResize(n, kk);
            };
            pack_to_keep(X);  pack_to_keep(R);
            pack_to_keep(Z);  pack_to_keep(P);
            pack_to_keep(AX); pack_to_keep(AP);

            const int kk = (int)keep_idx.size();
            auto shrink = [&](dmat& M){ M.conservativeResize(kk, kk); };
            shrink(RtZ); shrink(RtZ_old); shrink(RtZ_new);
            shrink(PtAP); shrink(alpha);  shrink(beta);

            bnorm.swap(b2); rnorm.swap(r2);
            if (rnorm.empty()) { info.converged = true; break; }
        }
    }

    return {X, info};
}

// Optional wrapper: force block CG to k=1 by looping columns.
// Enable with -DKKT_BLOCKCG_FORCE_K1 (compile flag).
inline std::pair<dmat, CGInfoBlock>
block_cg_k1(const BlockLinOp& A, const dmat& B, const BlockJacobi& Minv,
            double tol, int maxit, const std::optional<dmat>& X0 = std::nullopt,
            bool use_simd = true)
{
#ifndef KKT_BLOCKCG_FORCE_K1
    // Normal multi-RHS block CG
    return block_cg(A, B, Minv, tol, maxit, X0, use_simd);
#else
    // Column loop with k=1 runs the same block CG code path but with one RHS
    const int n = static_cast<int>(B.rows());
    const int k = static_cast<int>(B.cols());
    dmat X(n, k);
    CGInfoBlock agg; agg.converged = true; agg.iters = 0; agg.max_rel_resid = 0.0;

    for (int j = 0; j < k; ++j) {
        dmat Bj(n,1); Bj.col(0) = B.col(j);
        std::optional<dmat> X0j;
        if (X0 && X0->cols() == k) { X0j.emplace(n,1); X0j->col(0) = X0->col(j); }
        auto [Xj, infoj] = block_cg(A, Bj, Minv, tol, maxit, X0j, use_simd);
        X.col(j).noalias() = Xj.col(0);
        agg.converged       = agg.converged && infoj.converged;
        agg.iters           = std::max(agg.iters, infoj.iters);
        agg.max_rel_resid   = std::max(agg.max_rel_resid, infoj.max_rel_resid);
    }
    return {X, agg};
#endif
}


// ------------------------- K = W + δI + γ GᵀG operator ----------------------

struct KOp {
    int n = 0;
    const WOp* W = nullptr;
    double delta = 0.0;
    double gamma = 0.0;
    const spmat* G = nullptr;

    void apply(const dvec& x, dvec& y) const {
        if (y.size()!=x.size()) y.resize(x.size());
        y.setZero();

        W->apply(x, y);
        if (delta != 0.0) y.noalias() += delta * x;

        if (G && G->rows() > 0) {
            static thread_local dvec t;
            const int m = static_cast<int>(G->rows());
            if (t.size()!=m) t.resize(m);
            t.noalias() = (*G) * x;
            y.noalias() += gamma * (G->transpose() * t);
        }
    }

    void apply_block(const dmat& X, dmat& Y) const {
        const int nX = static_cast<int>(X.rows());
        const int k  = static_cast<int>(X.cols());
        if (Y.rows()!=nX || Y.cols()!=k) Y.resize(nX, k);
        Y.setZero();

        W->apply_block(X, Y);
        if (delta != 0.0) Y.noalias() += delta * X;

        if (G && G->rows() > 0 && k > 0) {
            dmat T = (*G) * X;               // (m × k)
            Y.noalias() += gamma * (G->transpose() * T);
        }
    }

    BlockLinOp as_block_op() const {
        BlockLinOp A;
        A.n  = n;
        A.mv = [this](const dvec& x, dvec& y){ this->apply(x,y); };
        A.bmv= [this](const dmat& X, dmat& Y){ this->apply_block(X,Y); };
        return A;
    }
};

// --------------------------- Schur HVP KKT (CG) -----------------------------

class SchurHVPKKT {
public:
    struct SolveInfo {
        int    schur_cg_iters      = 0;
        bool   schur_cg_converged  = false;
        double schur_cg_residual   = 0.0;
        int    formed_s_cols       = 0;
    };

    // Single RHS solve
    static std::tuple<dvec, dvec, SolveInfo>
    solve_once(const WOp& W,
               const std::optional<spmat> &Gopt,
               const dvec& r1, const dvec& r2,
               double delta, double gamma,
               const ChompConfig& cfg,
               bool assemble_schur_if_m_small = true,
               bool use_prec = true,
               const std::optional<dvec>& Kdiag_hint = std::nullopt)
    {
        const int n = static_cast<int>(r1.size());
        const int m = (Gopt && Gopt->rows() > 0) ? static_cast<int>(Gopt->rows()) : 0;

        // Prepare primals (enables lanes re-use)
        W.prepare();

        // Build K and operator
        KOp Kop{n, &W, delta, gamma, (Gopt && Gopt->rows()>0) ? &(*Gopt) : nullptr};
        BlockLinOp A = Kop.as_block_op();

        // --------- Diagonal approx & Jacobi preconds ----------
        dvec diagGtG = dvec::Zero(n);
        if (Kop.G && Kop.G->rows() > 0) diagGtG = diag_GtG(*Kop.G);

        dvec diagK = dvec::Constant(n, std::max(1e-12, Kop.delta));
        if (Kop.G && Kop.G->rows() > 0) diagK.noalias() += Kop.gamma * diagGtG;

        // If W carries a diagonal Sigma_x, you can add it here:
        // diagK += Sigma_x;

        dvec Kdiag_inv = diagK.cwiseMax(1e-12).cwiseInverse();
        if (Kdiag_hint && Kdiag_hint->size() == n)
            Kdiag_inv = Kdiag_hint->cwiseMax(1e-12).cwiseInverse();

        ScalarJacobi Minv_scalar{Kdiag_inv};
        BlockJacobi  Minv_block{Kdiag_inv};

        // Two tolerances for inner solves
        const double assemble_tol = std::max(1e-2, cfg.cg_tol) * 1e1;
        const int    assemble_it  = std::max(10, cfg.cg_maxit / 2);
        const double final_tol    = cfg.cg_tol;
        const int    final_it     = cfg.cg_maxit;

        SolveInfo sinfo{};
        dvec dy;

        if (m == 0) {
            // dx = K^{-1} r1
#ifdef KKT_USE_BLOCK_CG
            dmat B(n,1); B.col(0) = r1;
            auto [Xsol, info0] = block_cg_k1(A, B, Minv_block, final_tol, final_it, std::nullopt, cfg.use_simd);
            (void)info0;
            return {Xsol.col(0), dvec(), sinfo};
#else
            auto [dx, info0] = cg_single(A, r1, Minv_scalar, final_tol, final_it, std::nullopt);
            (void)info0;
            return {dx, dvec(), sinfo};
#endif
        }

        // svec = r1 + γ Gᵀ r2
        const dvec svec = r1 + gamma * (Kop.G->transpose() * r2);

        // K^{-1} svec (looser tol)
#ifdef KKT_USE_BLOCK_CG
        dmat B1(n,1); B1.col(0) = svec;
        auto [Kinv_s_mat, info_s] = block_cg_k1(A, B1, Minv_block, assemble_tol, assemble_it, std::nullopt, cfg.use_simd);
        (void)info_s;
        const dvec Kinvsvec = Kinv_s_mat.col(0);
#else
        auto [Kinvsvec, info_s] = cg_single(A, svec, Minv_scalar, assemble_tol, assemble_it, std::nullopt);
        (void)info_s;
#endif

        // rhs_s = G K^{-1} svec - r2
        const dvec rhs_s = (*Kop.G) * Kinvsvec - r2;

        const bool small_m =
            assemble_schur_if_m_small &&
            (m <= std::max(1, int(cfg.schur_dense_cutoff * n)));

        if (small_m) {
            // Dense Schur: S = G K^{-1} Gᵀ
#ifdef KKT_USE_BLOCK_CG
            dmat B(n, m); B = Kop.G->transpose();
            auto [Z, infoZ] = block_cg_k1(A, B, Minv_block, assemble_tol, assemble_it, std::nullopt, cfg.use_simd);
            (void)infoZ;
            dmat S = (*Kop.G) * Z; // (m×m)
#else
            dmat Z(n, m);
            for (int j = 0; j < m; ++j) {
                dvec rhs = Kop.G->transpose().col(j);
                auto [zj, infoZ] = cg_single(A, rhs, Minv_scalar, assemble_tol, assemble_it, std::nullopt);
                (void)infoZ;
                Z.col(j).noalias() = zj;
            }
            dmat S = (*Kop.G) * Z; // (m×m)
#endif
            add_jitter_spd(S);
            dy = Eigen::LLT<dmat>(S).solve(rhs_s);
            sinfo.formed_s_cols = m;
        } else {
            // Iterative Schur (single-rhs CG outer), apply y ↦ G K^{-1} Gᵀ y
            LinOp S_op{m, [&](const dvec& y, dvec& out) {
#ifdef KKT_USE_BLOCK_CG
                dmat BY(n,1); BY.col(0) = Kop.G->transpose() * y;
                auto [Zy, infoY] = block_cg_k1(A, BY, Minv_block, assemble_tol, assemble_it, std::nullopt, cfg.use_simd);
                (void)infoY;
                out.noalias() = (*Kop.G) * Zy.col(0);
#else
                dvec rhs = Kop.G->transpose() * y;
                auto [z, infoY] = cg_single(A, rhs, Minv_scalar, assemble_tol, assemble_it, std::nullopt);
                (void)infoY;
                out.noalias() = (*Kop.G) * z;
#endif
            }};

            // Optional Schur preconditioners (Jacobi/SSOR)
            std::optional<dvec> SchurJacobiMinv;
            std::optional<SSORPrecond> SchurSSOR;
            if (use_prec) {
                if (cfg.prec_type == "jacobi") {
                    SchurJacobiMinv = schur_diag_hat(*Kop.G, Kdiag_inv).cwiseInverse();
                } else if (cfg.prec_type == "ssor") {
                    const spmat S_hat = build_S_hat(*Kop.G, Kdiag_inv);
                    SchurSSOR.emplace(S_hat, cfg.ssor_omega);
                }
            }

            auto cg_res = cg(S_op, rhs_s,
                             SchurJacobiMinv,
                             cfg.cg_tol, cfg.cg_maxit,
                             std::nullopt,
                             SchurSSOR,
                             cfg.use_simd);

            dy = std::move(cg_res.first);
            sinfo.schur_cg_iters     = cg_res.second.iters;
            sinfo.schur_cg_converged = cg_res.second.converged;
            sinfo.schur_cg_residual  = cg_res.second.final_residual;
        }

        // Final dx: K dx = svec - Gᵀ dy, warm-start with K^{-1}svec
        const dvec rhs_final = svec - Kop.G->transpose() * dy;
#ifdef KKT_USE_BLOCK_CG
        dmat X0(n,1); X0.col(0) = Kinvsvec;
        dmat Bf(n,1); Bf.col(0) = rhs_final;
        auto [Xfin, info_fin] = block_cg_k1(A, Bf, Minv_block, final_tol, final_it, X0, cfg.use_simd);
        (void)info_fin;
        return {Xfin.col(0), dy, sinfo};
#else
        auto [dx, info_fin] = cg_single(A, rhs_final, ScalarJacobi{Kdiag_inv}, final_tol, final_it, Kinvsvec);
        (void)info_fin;
        return {dx, dy, sinfo};
#endif
    }

    // Reusable wrapper
    class Reusable : public KKTReusable {
    public:
        Reusable(WOp W_,
                 std::optional<spmat> G_,
                 double delta_, double gamma_, ChompConfig cfg_)
            : W(std::move(W_)), G(std::move(G_)),
              delta(delta_), gamma(gamma_), cfg(std::move(cfg_)) {}

        std::pair<dvec, dvec> solve(const dvec& r1,
                                    const std::optional<dvec>& r2,
                                    double cg_tol, int cg_maxit) override {
            if (!r2) throw std::invalid_argument("SchurHVPKKT::Reusable needs r2");
            ChompConfig local = cfg;
            local.cg_tol   = cg_tol;
            local.cg_maxit = std::max(1, cg_maxit);

            auto [dx, dy, _] =
                SchurHVPKKT::solve_once(W, G, r1, *r2, delta, gamma,
                                        local, /*assemble small*/ true,
                                        /*use prec*/ true, std::nullopt);
            return {dx, dy};
        }

        WOp W;
        std::optional<spmat> G;  // equality Jacobian
        double delta = 0.0, gamma = 0.0;
        ChompConfig cfg;
    };
};

} // namespace kkt
