#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <vector>

namespace piqp {

using Vector = Eigen::VectorXd;
using SparseMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

struct PIQPSettings {
    double eps_abs = 1e-6;
    double eps_rel = 1e-6;
    int max_iter = 300;

    // Prox/PMM
    double rho_init = 1e-2;
    double delta_init = 1e-1;
    double rho_floor = 1e-9;
    double delta_floor = 1e-9;

    // Fraction-to-boundary
    double tau = 0.995;

    // Numerical / logging
    double reg_eps = 1e-12;
    bool verbose = false;

    // Safety
    double min_slack = 1e-16;

    // (knobs kept for compatibility; unused here)
    bool scale = false;
    int ruiz_iters = 4;
    double scale_eps = 1e-3;
    bool cost_scaling = false;
};

struct PIQPResiduals {
    double eq_inf = 0.0;
    double ineq_inf = 0.0;
    double stat_inf = 0.0;
    double gap = 0.0;
};

struct PIQPResult {
    std::string status;
    int iterations = 0;
    Vector x;
    Vector s; // slacks
    Vector y; // λ = equality multipliers
    Vector z; // μ = inequality multipliers

    double obj_val = 0.0;
    PIQPResiduals residuals;
};

class PIQPSolver {
public:
    explicit PIQPSolver(const PIQPSettings &settings = PIQPSettings{})
        : settings_(settings) {}

    // Sparse-only setup
    PIQPSolver &setup(const SparseMatrix &P, const Vector &q,
                      const std::optional<SparseMatrix> &A = std::nullopt,
                      const std::optional<Vector> &b = std::nullopt,
                      const std::optional<SparseMatrix> &G = std::nullopt,
                      const std::optional<Vector> &h = std::nullopt) {
        n_ = static_cast<int>(q.size());
        P_ = P; // assumed symmetric; Ψ uses lower triangle via LLT
        q_ = q;

        if (A && b) {
            A_ = *A;
            b_ = *b;
            p_ = static_cast<int>(A_.rows());
        } else {
            A_.resize(0, n_);
            A_.data().squeeze();
            b_.resize(0);
            p_ = 0;
        }

        if (G && h) {
            G_ = *G;
            h_ = *h;
            m_ = static_cast<int>(G_.rows());
        } else {
            G_.resize(0, n_);
            G_.data().squeeze();
            h_.resize(0);
            m_ = 0;
        }

        // Parameters
        delta_ = settings_.delta_init;
        rho_ = settings_.rho_init;

        // Iterates
        x_.setZero(n_);
        if (p_ > 0)
            y_.setZero(p_);
        else
            y_.resize(0);
        if (m_ > 0) {
            s_ = Vector::Constant(m_, 1.0);
            z_ = Vector::Constant(m_, 1.0);
        } else {
            s_.resize(0);
            z_.resize(0);
        }

        // Prox centers
        xi_ = x_;
        if (p_ > 0)
            lambda_ = y_;
        else
            lambda_.resize(0);
        if (m_ > 0)
            nu_ = z_;
        else
            nu_.resize(0);

        // Precompute AᵀA (pattern and values fixed)
        if (p_ > 0) {
            AtA_ = A_.transpose() * A_;
            AtA_.makeCompressed();
        } else {
            AtA_.resize(0, 0);
            AtA_.data().squeeze();
        }

        // Save originals for objective and final residual report
        P_orig_ = P_;
        q_orig_ = q_;
        A_orig_ = A_;
        b_orig_ = b_;
        G_orig_ = G_;
        h_orig_ = h_;

        // Prepare Ψ pattern once (pattern = P ∪ AᵀA ∪ GᵀG ∪ I)
        buildPsiPattern_();

        return *this;
    }

    PIQPResult solve() {
        std::string status = "max_iter";
        int iters = settings_.max_iter;

        auto [p_prev, d_prev] = measuresPkDk();
        double sz_prev = (m_ > 0) ? s_.dot(z_) : 0.0;

        // Reused temporaries
        Vector rx_aff(n_), ry_aff((p_ > 0) ? p_ : 0), rz_aff((m_ > 0) ? m_ : 0),
            rs_aff((m_ > 0) ? m_ : 0);
        Vector dx_aff(n_), dy_aff((p_ > 0) ? p_ : 0), dz_aff((m_ > 0) ? m_ : 0),
            ds_aff((m_ > 0) ? m_ : 0);
        Vector dx_cor(n_), dy_cor((p_ > 0) ? p_ : 0), dz_cor((m_ > 0) ? m_ : 0),
            ds_cor((m_ > 0) ? m_ : 0);
        Vector Ax((p_ > 0) ? p_ : 0), Gx((m_ > 0) ? m_ : 0);
        Vector lambda_weights((m_ > 0) ? m_ : 0);
        Vector ones_m;
        if (m_ > 0)
            ones_m = Vector::Ones(m_);
        else
            ones_m.resize(0);
        Vector s_a((m_ > 0) ? m_ : 0), z_a((m_ > 0) ? m_ : 0);

        for (iter_ = 0; iter_ < settings_.max_iter; ++iter_) {
            const double inv_delta = (p_ > 0) ? (1.0 / delta_) : 0.0;

            // Reduced weights W = diag( (s./z + δ)^{-1} )
            if (m_ > 0)
                lambda_weights = buildReducedWeights(s_, z_);

            // Cache Ax, Gx
            if (p_ > 0)
                Ax.noalias() = A_ * x_;
            if (m_ > 0)
                Gx.noalias() = G_ * x_;

            // -------- Affine RHS --------
            rx_aff.noalias() = -(P_ * x_ + q_ + rho_ * (x_ - xi_));
            if (p_ > 0)
                rx_aff.noalias() -= A_.transpose() * y_;
            if (m_ > 0)
                rx_aff.noalias() -= G_.transpose() * z_;

            if (p_ > 0)
                ry_aff = -(Ax + delta_ * (lambda_ - y_) - b_);
            else
                ry_aff.resize(0);

            if (m_ > 0) {
                rz_aff = -(Gx + delta_ * (nu_ - z_) - h_ + s_);
                rs_aff = -(s_.cwiseProduct(z_));
            } else {
                rz_aff.resize(0);
                rs_aff.resize(0);
            }

            // Predictor (reuse symbolic Ψ)
            solvePsiNewton_(rx_aff, ry_aff, rz_aff, rs_aff, lambda_weights,
                            inv_delta, dx_aff, dy_aff, dz_aff, ds_aff);

            // Fraction-to-boundary & sigma
            const double alpha_aff = fractionToBoundary(s_, ds_aff, z_, dz_aff);
            const double mu = (m_ > 0) ? (s_.dot(z_) / std::max(1, m_)) : 0.0;
            double sigma = 0.0;
            if (m_ > 0) {
                s_a.noalias() = s_ + alpha_aff * ds_aff;
                z_a.noalias() = z_ + alpha_aff * dz_aff;
                const double num = s_a.dot(z_a);
                const double den = std::max(1e-30, s_.dot(z_));
                const double eta = num / den;
                sigma = std::max(0.0, std::min(1.0, std::pow(eta, 3)));
            }

            // -------- Corrector RHS (only rs changes) --------
            Vector rs_cor((m_ > 0) ? m_ : 0);
            if (m_ > 0) {
                rs_cor.noalias() = -(s_.cwiseProduct(z_)) -
                                   ds_aff.cwiseProduct(dz_aff) +
                                   sigma * mu * ones_m;
            } else {
                rs_cor.resize(0);
            }

            // Corrector (same Ψ pattern, new numeric)
            solvePsiNewton_(rx_aff, ry_aff, rz_aff, rs_cor, lambda_weights,
                            inv_delta, dx_cor, dy_cor, dz_cor, ds_cor);

            // -------- Take step --------
            const double alpha = fractionToBoundary(s_, ds_cor, z_, dz_cor);
            x_ += alpha * dx_cor;
            if (p_ > 0)
                y_ += alpha * dy_cor;
            if (m_ > 0) {
                s_ += alpha * ds_cor;
                z_ += alpha * dz_cor;
                s_ = s_.array().max(settings_.min_slack);
                z_ = z_.array().max(settings_.min_slack);
            }

            // Termination
            if (checkTermination()) {
                status = "solved";
                iters = iter_ + 1;
                break;
            }

            // Algorithm 2 updates
            auto [p_new, d_new] = measuresPkDk();
            const double sz_new = (m_ > 0) ? s_.dot(z_) : sz_prev;
            const double r = (m_ == 0) ? 0.0
                                       : std::abs(sz_prev - sz_new) /
                                             std::max(std::abs(sz_prev), 1e-30);

            if (p_new <= 0.95 * p_prev) {
                if (p_ > 0)
                    lambda_ = y_;
                if (m_ > 0)
                    nu_ = z_;
                delta_ = (1.0 - r) * delta_;
            } else {
                delta_ = (1.0 - r / 3.0) * delta_;
            }

            if (d_new <= 0.95 * d_prev) {
                xi_ = x_;
                rho_ = (1.0 - r) * rho_;
            } else {
                rho_ = (1.0 - r / 3.0) * rho_;
            }

            delta_ = std::max(delta_, settings_.delta_floor);
            rho_ = std::max(rho_, settings_.rho_floor);

            p_prev = p_new;
            d_prev = d_new;
            sz_prev = sz_new;

            if (settings_.verbose) {
                std::cout << "[PIQP] it " << iter_ << "  rho=" << rho_
                          << "  delta=" << delta_ << std::endl;
            }
        }

        // Objective
        double obj_val = 0.0;
        if (n_ > 0) {
            Vector Px = P_orig_ * x_;
            obj_val = 0.5 * x_.dot(Px) + q_orig_.dot(x_);
        }

        // Residuals for report
        auto [r_eq_final, r_ineq_final] = primalBlock();
        Vector r_stat_final = dualStationarity();

        PIQPResult R;
        R.status = status;
        R.iterations = iters;
        R.x = x_;
        R.s = s_;
        R.y = y_;
        R.z = z_;
        R.obj_val = obj_val;
        if (m_ > 0)
            R.s = s_;
        if (p_ > 0)
            R.y = y_;
        if (m_ > 0)
            R.z = z_;
        R.obj_val = obj_val;
        R.residuals.eq_inf =
            (p_ > 0) ? r_eq_final.lpNorm<Eigen::Infinity>() : 0.0;
        R.residuals.ineq_inf =
            (m_ > 0) ? r_ineq_final.lpNorm<Eigen::Infinity>() : 0.0;
        R.residuals.stat_inf = r_stat_final.lpNorm<Eigen::Infinity>();
        R.residuals.gap = std::abs(dualityGap());

        status_ = status;
        return R;
    }

    // Getters
    const PIQPSettings &getSettings() const { return settings_; }
    const Vector &getX() const { return x_; }
    const Vector &getS() const { return s_; }
    const Vector &getY() const { return y_; }
    const Vector &getZ() const { return z_; }
    int getIterations() const { return iter_; }
    const std::string &getStatus() const { return status_; }

public:
    // ---- Warm start API ----
    // Set initial iterates; any std::nullopt leaves current values in place.
    // If copy_to_prox_centers==true, also sets (xi, lambda, nu) to the same.
    PIQPSolver &warm_start(const std::optional<Vector> &x = std::nullopt,
                           const std::optional<Vector> &y = std::nullopt,
                           const std::optional<Vector> &z = std::nullopt,
                           const std::optional<Vector> &s = std::nullopt,
                           bool copy_to_prox_centers = true) {
        if (x) {
            assert(int(x->size()) == n_);
            x_ = *x;
        }
        if (p_ > 0 && y) {
            assert(int(y->size()) == p_);
            y_ = *y;
        }
        if (m_ > 0 && z) {
            assert(int(z->size()) == m_);
            z_ = z->array().max(settings_.min_slack);
        }
        if (m_ > 0 && s) {
            assert(int(s->size()) == m_);
            s_ = s->array().max(settings_.min_slack);
        }
        if (copy_to_prox_centers) {
            xi_ = x_;
            if (p_ > 0)
                lambda_ = y_;
            if (m_ > 0)
                nu_ = z_;
        }
        return *this;
    }

    // Explicit prox-center warm start (does not touch x,y,z,s)
    PIQPSolver &set_prox_centers(const std::optional<Vector> &xi,
                                 const std::optional<Vector> &lambda,
                                 const std::optional<Vector> &nu) {
        if (xi) {
            assert(int(xi->size()) == n_);
            xi_ = *xi;
        }
        if (p_ > 0 && lambda) {
            assert(int(lambda->size()) == p_);
            lambda_ = *lambda;
        }
        if (m_ > 0 && nu) {
            assert(int(nu->size()) == m_);
            nu_ = nu->array().max(settings_.min_slack);
        }
        return *this;
    }

    // Convenience: prime next solve with the solver's current iterates.
    // (also primes prox centers unless also_prox=false)
    PIQPSolver &use_last_as_warm_start(bool also_prox = true) {
        if (also_prox) {
            xi_ = x_;
            if (p_ > 0)
                lambda_ = y_;
            if (m_ > 0)
                nu_ = z_;
        }
        return *this;
    }

    // Quick knobs to reset proximal penalties before a new solve
    PIQPSolver &set_prox_params(double rho, double delta) {
        rho_ = std::max(rho, settings_.rho_floor);
        delta_ = std::max(delta, settings_.delta_floor);
        return *this;
    }

    // ---- Numeric update for SQP loops (pattern-stable fast path) ----
    // Update values of P,q,A,b,G,h. If same_pattern=true (default), we REUSE
    // the analyzed pattern of Ψ; else we rebuild it (one-time analyzePattern).
    PIQPSolver &
    update_values(const SparseMatrix &P, const Vector &q,
                  const std::optional<SparseMatrix> &A = std::nullopt,
                  const std::optional<Vector> &b = std::nullopt,
                  const std::optional<SparseMatrix> &G = std::nullopt,
                  const std::optional<Vector> &h = std::nullopt,
                  bool same_pattern = true) {
        assert(int(q.size()) == n_);
        P_ = P;
        q_ = q;

        if (A) {
            assert(A->cols() == n_);
            A_ = *A;
            p_ = int(A_.rows());
        }
        if (b) {
            b_ = *b;
        }
        if (G) {
            assert(G->cols() == n_);
            G_ = *G;
            m_ = int(G_.rows());
        }
        if (h) {
            h_ = *h;
        }

        // Recompute AᵀA numerically; pattern unchanged if A's pattern
        // unchanged.
        if (p_ > 0) {
            AtA_ = A_.transpose() * A_;
            AtA_.makeCompressed();
        } else {
            AtA_.resize(0, 0);
            AtA_.data().squeeze();
        }

        // Optionally rebuild Ψ pattern if sparsity changed
        if (!same_pattern) {
            psi_symbolic_done_ = false;
            buildPsiPattern_();
        }

        // Ensure sizes of iterates & prox centers remain consistent
        resize_iterates_if_needed_();

        return *this;
    }

private:
    void resize_iterates_if_needed_() {
        if (int(x_.size()) != n_) x_.setZero(n_);
        if (p_ > 0) {
            if (int(y_.size()) != p_) y_.setZero(p_);
            if (int(lambda_.size()) != p_) lambda_.setZero(p_);
        } else {
            y_.resize(0); lambda_.resize(0);
        }
        if (m_ > 0) {
            if (int(s_.size()) != m_) s_ = Vector::Constant(m_, 1.0);
            if (int(z_.size()) != m_) z_ = Vector::Constant(m_, 1.0);
            s_ = s_.array().max(settings_.min_slack);
            z_ = z_.array().max(settings_.min_slack);
            if (int(nu_.size()) != m_) nu_ = z_;
        } else {
            s_.resize(0); z_.resize(0); nu_.resize(0);
        }
        if (int(xi_.size()) != n_) xi_ = x_;
    }

    // ---------- data ----------
    PIQPSettings settings_;

    // problem data (sparse only)
    SparseMatrix P_, A_, G_;
    Vector q_, b_, h_;
    int n_ = 0, p_ = 0, m_ = 0;

    // copies for objective/residuals
    SparseMatrix P_orig_, A_orig_, G_orig_;
    Vector q_orig_, b_orig_, h_orig_;

    // precomputed
    SparseMatrix AtA_; // AᵀA (pattern/values fixed)

    // iterates
    Vector x_, s_, y_, z_;
    Vector xi_, lambda_, nu_;
    double delta_ = 0.0, rho_ = 0.0;

    // stats
    int iter_ = 0;
    std::string status_ = "unknown";

    // ---------- Ψ factorization with analyzePattern ----------
    bool psi_symbolic_done_ = false;
    Eigen::SimplicialLLT<SparseMatrix, Eigen::Lower> chol_;
    SparseMatrix psi_pattern_; // pattern-only matrix (no numerical values)

    void buildPsiPattern_() {
        // Pattern = P ∪ AᵀA ∪ GᵀG ∪ I
        SparseMatrix I(n_, n_);
        I.setIdentity();

        SparseMatrix GtG;
        if (m_ > 0) {
            GtG = G_.transpose() * G_;
            GtG.makeCompressed();
        } else {
            GtG.resize(0, 0);
            GtG.data().squeeze();
        }

        psi_pattern_.resize(n_, n_);
        psi_pattern_.setZero();
        // Use triplets to form union of patterns
        std::vector<Eigen::Triplet<double, int>> T;
        T.reserve(P_.nonZeros() + AtA_.nonZeros() + GtG.nonZeros() + n_);

        // helper to append pattern (values = 1.0)
        auto push_pattern = [&](const SparseMatrix &M) {
            for (int k = 0; k < M.outerSize(); ++k) {
                for (SparseMatrix::InnerIterator it(M, k); it; ++it) {
                    T.emplace_back(it.row(), it.col(), 1.0);
                }
            }
        };
        if (P_.rows() == n_)
            push_pattern(P_);
        if (AtA_.rows() == n_)
            push_pattern(AtA_);
        if (GtG.rows() == n_)
            push_pattern(GtG);
        push_pattern(I);

        psi_pattern_.setFromTriplets(T.begin(), T.end());
        psi_pattern_.makeCompressed();

        // Run analyzePattern once on this pattern
        chol_.analyzePattern(psi_pattern_.selfadjointView<Eigen::Lower>());
        psi_symbolic_done_ = true;
    }

    // Assemble numeric Ψ and solve reduced Newton system with reused symbolic
    // factor
    void solvePsiNewton_(const Vector &rx, const Vector &ry, const Vector &rz,
                         const Vector &rs, const Vector &lambda_weights,
                         double inv_delta, Vector &dx, Vector &dy, Vector &dz,
                         Vector &ds) {
        // ---- assemble Ψ numerically (pattern fixed) ----
        SparseMatrix Psi(n_, n_);
        Psi = P_;

        // + rho I + reg I
        {
            SparseMatrix I(n_, n_);
            I.setIdentity();
            if (rho_ != 0.0)
                Psi += rho_ * I;
            if (settings_.reg_eps != 0.0)
                Psi += settings_.reg_eps * I;
        }

        // + (1/δ) AᵀA
        if (p_ > 0 && inv_delta > 0.0) {
            Psi += inv_delta * AtA_;
        }

        // + Gᵀ W G (W diagonal from lambda_weights)
        if (m_ > 0 && lambda_weights.size() == m_) {
            // build B = sqrt(W) * G and add BᵀB
            SparseMatrix B = G_;
            // Row scaling by sqrt(w_i)
            for (int k = 0; k < B.outerSize(); ++k) {
                for (SparseMatrix::InnerIterator it(B, k); it; ++it) {
                    const int r = it.row();
                    const double s =
                        std::sqrt(std::max(0.0, lambda_weights(r)));
                    it.valueRef() *= s;
                }
            }
            Psi += B.transpose() * B;
        }

        Psi.makeCompressed();

        // ---- factorize with reused symbolic structure ----
        if (!psi_symbolic_done_) {
            chol_.analyzePattern(Psi.selfadjointView<Eigen::Lower>());
            psi_symbolic_done_ = true;
        }
        chol_.factorize(Psi.selfadjointView<Eigen::Lower>());

        // ---- build RHS: rx + (1/δ)Aᵀ ry + Gᵀ (W (rz - rs./z)) ----
        Vector rhs = rx;
        if (p_ > 0 && inv_delta > 0.0)
            rhs.noalias() += inv_delta * (A_.transpose() * ry);
        if (m_ > 0) {
            Vector rbar = rz - rs.cwiseQuotient(z_);
            rhs.noalias() +=
                G_.transpose() * (lambda_weights.cwiseProduct(rbar));
        }

        // ---- solve Ψ dx = rhs ----
        dx = chol_.solve(rhs);

        // ---- recover dy, dz, ds ----
        if (p_ > 0)
            dy = inv_delta * (A_ * dx - ry);
        else
            dy.resize(0);

        if (m_ > 0) {
            Vector rbar = rz - rs.cwiseQuotient(z_);
            dz = lambda_weights.cwiseProduct(G_ * dx - rbar);
            ds = rs.cwiseQuotient(z_) - s_.cwiseQuotient(z_).cwiseProduct(dz);
        } else {
            dz.resize(0);
            ds.resize(0);
        }
    }

    // Core pieces used by termination and reporting
    std::pair<Vector, Vector> primalBlock(const Vector *x = nullptr,
                                          const Vector *s = nullptr) const {
        const Vector &xu = x ? *x : x_;
        const Vector &su = s ? *s : s_;
        Vector r_eq((p_ > 0) ? p_ : 0), r_in((m_ > 0) ? m_ : 0);
        if (p_ > 0)
            r_eq = A_ * xu - b_;
        else
            r_eq.resize(0);
        if (m_ > 0)
            r_in = G_ * xu - h_ + su;
        else
            r_in.resize(0);
        return {r_eq, r_in};
    }

    Vector dualStationarity(const Vector *x = nullptr,
                            const Vector *y = nullptr,
                            const Vector *z = nullptr) const {
        const Vector &xu = x ? *x : x_;
        Vector r_stat = P_ * xu + q_;
        if (p_ > 0) {
            const Vector &yu = y ? *y : y_;
            r_stat.noalias() += A_.transpose() * yu;
        }
        if (m_ > 0) {
            const Vector &zu = z ? *z : z_;
            r_stat.noalias() += G_.transpose() * zu;
        }
        return r_stat;
    }

    std::pair<double, double> measuresPkDk(const Vector *x = nullptr,
                                           const Vector *y = nullptr,
                                           const Vector *z = nullptr,
                                           const Vector *s = nullptr) const {
        auto [r_eq, r_in] = primalBlock(x, s);
        Vector r_stat = dualStationarity(x, y, z);
        double p_k = 0.0;
        if (r_eq.size() > 0)
            p_k = std::max(p_k, r_eq.lpNorm<Eigen::Infinity>());
        if (r_in.size() > 0)
            p_k = std::max(p_k, r_in.lpNorm<Eigen::Infinity>());
        double d_k =
            (r_stat.size() > 0) ? r_stat.lpNorm<Eigen::Infinity>() : 0.0;
        return {p_k, d_k};
    }

    double dualityGap(const Vector *x = nullptr, const Vector *y = nullptr,
                      const Vector *z = nullptr) const {
        const Vector &xu = x ? *x : x_;
        const Vector &yu = y ? *y : y_;
        const Vector &zu = z ? *z : z_;
        double t1 = (n_ > 0) ? xu.dot(P_ * xu) : 0.0;
        double t2 = (n_ > 0) ? q_.dot(xu) : 0.0;
        double t3 = (p_ > 0) ? b_.dot(yu) : 0.0;
        double t4 = (m_ > 0) ? h_.dot(zu) : 0.0;
        return t1 + t2 + t3 + t4;
    }

    bool checkTermination() {
        auto [r_eq, r_in] = primalBlock();
        Vector r_stat = dualStationarity();

        // primal
        double lhs_pri = 0.0;
        if (p_ > 0)
            lhs_pri = std::max(lhs_pri, r_eq.lpNorm<Eigen::Infinity>());
        if (m_ > 0)
            lhs_pri = std::max(lhs_pri, r_in.lpNorm<Eigen::Infinity>());

        Vector Ax((p_ > 0) ? p_ : 0), Gx((m_ > 0) ? m_ : 0);
        if (p_ > 0)
            Ax = A_ * x_;
        if (m_ > 0)
            Gx = G_ * x_;

        double scale_pri = 1.0;
        if (p_ > 0) {
            scale_pri = std::max(scale_pri, Ax.lpNorm<Eigen::Infinity>());
            scale_pri = std::max(scale_pri, b_.lpNorm<Eigen::Infinity>());
        }
        if (m_ > 0) {
            scale_pri = std::max(scale_pri, Gx.lpNorm<Eigen::Infinity>());
            scale_pri = std::max(scale_pri, h_.lpNorm<Eigen::Infinity>());
            scale_pri = std::max(scale_pri, s_.lpNorm<Eigen::Infinity>());
        }
        const double rhs_pri =
            settings_.eps_abs + settings_.eps_rel * scale_pri;

        // dual
        const double lhs_dual =
            (r_stat.size() > 0) ? r_stat.lpNorm<Eigen::Infinity>() : 0.0;

        double Aty_inf = 0.0, Gtz_inf = 0.0, Px_inf = 0.0, q_inf = 0.0;
        if (n_ > 0) {
            Vector Px = P_ * x_;
            Px_inf = Px.lpNorm<Eigen::Infinity>();
            q_inf = q_.lpNorm<Eigen::Infinity>();
        }
        if (p_ > 0) {
            Vector Aty = A_.transpose() * y_;
            Aty_inf = Aty.lpNorm<Eigen::Infinity>();
        }
        if (m_ > 0) {
            Vector Gtz = G_.transpose() * z_;
            Gtz_inf = Gtz.lpNorm<Eigen::Infinity>();
        }
        const double scale_dual =
            std::max({1.0, Px_inf, Aty_inf, Gtz_inf, q_inf});
        const double rhs_dual =
            settings_.eps_abs + settings_.eps_rel * scale_dual;

        // gap
        const double gap = std::abs(dualityGap());
        double scale_gap = 1e-30;
        if (n_ > 0) {
            Vector Px = P_ * x_;
            scale_gap = std::max(scale_gap, std::abs(x_.dot(Px)));
            scale_gap = std::max(scale_gap, std::abs(q_.dot(x_)));
        }
        if (p_ > 0)
            scale_gap = std::max(scale_gap, std::abs(b_.dot(y_)));
        if (m_ > 0)
            scale_gap = std::max(scale_gap, std::abs(h_.dot(z_)));
        const double rhs_gap =
            settings_.eps_abs + settings_.eps_rel * scale_gap;

        const bool ok =
            (lhs_pri <= rhs_pri) && (lhs_dual <= rhs_dual) && (gap <= rhs_gap);

        if (settings_.verbose) {
            std::cout << "[PIQP] it " << iter_ << "  pri " << lhs_pri << "/"
                      << rhs_pri << "  dual " << lhs_dual << "/" << rhs_dual
                      << "  gap " << gap << "/" << rhs_gap << std::endl;
        }
        return ok;
    }

    Vector buildReducedWeights(const Vector &s, const Vector &z) const {
        if (m_ == 0)
            return Vector(0);
        Vector w = s.cwiseQuotient(z);              // s ./ z
        return (w.array() + delta_).cwiseInverse(); // 1 ./ (s./z + δ)
    }

    double fractionToBoundary(const Vector &s, const Vector &ds,
                              const Vector &z, const Vector &dz) const {
        if (m_ == 0)
            return 1.0;
        double alpha = 1.0;
        const double tau = settings_.tau;
        for (int i = 0; i < m_; ++i)
            if (ds(i) < 0.0)
                alpha = std::min(alpha, -tau * s(i) / ds(i));
        for (int i = 0; i < m_; ++i)
            if (dz(i) < 0.0)
                alpha = std::min(alpha, -tau * z(i) / dz(i));
        return std::max(0.0, std::min(1.0, alpha));
    }
};

} // namespace piqp
