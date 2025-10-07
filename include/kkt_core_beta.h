#pragma once
// Enhanced KKT solver with IP-specific optimizations

#include "model.h"
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "kkt_helper.h"
#include "qdldl.h"

namespace kkt {

using dvec = Eigen::VectorXd;
using dmat = Eigen::MatrixXd;
using spmat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

// ==================== IP-Specific Configuration ====================
struct IPRegularizationConfig {
    bool use_ip_regularization{true};
    double mu_dependent_reg{1e-2};     // μ-dependent regularization factor
    double constraint_scaling{1e-3};   // scale with constraint violation
    double sigma_scaling{1e-2};        // scale with barrier weights
    bool use_primal_dual_reg{true};    // separate primal/dual regularization
    double min_regularization{1e-12};  // minimum regularization
    double max_regularization{1e6};    // maximum regularization
    bool use_inertia_correction{true}; // enable inertia correction
    int max_inertia_corrections{5};    // max attempts to fix inertia
    double inertia_regularization_factor{
        10.0}; // factor for inertia corrections
};

struct InertiaInfo {
    int positive{0};
    int negative{0};
    int zero{0};
    bool is_valid() const { return zero == 0; }
    bool has_correct_ip_inertia(int n, int m) const {
        return positive == n && negative == m && zero == 0;
    }
};

// ==================== Barrier-Aware Preconditioner ====================
class BarrierPreconditioner {
public:
    static dvec compute_barrier_weights(const dvec &s, const dvec &z, double mu,
                                        bool use_sqrt = true) {
        dvec weights(s.size());
        const double eps = std::max(1e-12, mu * 1e-3);

        for (int i = 0; i < s.size(); ++i) {
            const double si = std::max(s[i], eps);
            const double zi = std::max(z[i], eps);

            if (use_sqrt) {
                weights[i] = std::sqrt(zi / si);
            } else {
                weights[i] = zi / si;
            }
        }
        return weights;
    }

    static dvec compute_schur_barrier_preconditioner(const spmat &G,
                                                     const dvec &Sigma_x,
                                                     const dvec &Sigma_s,
                                                     double mu) {
        const int m = G.rows();
        dvec diag_prec(m);

        // More sophisticated barrier-aware diagonal approximation
        for (int i = 0; i < m; ++i) {
            double sum = 0.0;
            for (spmat::InnerIterator it(G, i); it; ++it) {
                const int col = it.col();
                const double gij = it.value();
                const double sigma_inv = 1.0 / std::max(Sigma_x[col], 1e-12);
                sum += gij * gij * sigma_inv;
            }

            // Add contribution from slack variables
            if (i < Sigma_s.size()) {
                sum += 1.0 / std::max(Sigma_s[i], 1e-12);
            }

            // Include μ-dependent scaling
            const double mu_factor = 1.0 + 1.0 / std::max(mu, 1e-8);
            diag_prec[i] = 1.0 / std::max(sum * mu_factor, 1e-16);
        }

        return diag_prec;
    }
};

// ==================== Inertia Detection and Correction ====================
class InertiaCorrector {
public:
    static InertiaInfo compute_ldlt_inertia(const qdldl23::LDL32 &factors) {
        InertiaInfo info;
        const auto &D = factors.D;

        for (size_t i = 0; i < D.size(); ++i) {
            const double d = D[i];
            if (std::abs(d) < 1e-14) {
                info.zero++;
            } else if (d > 0) {
                info.positive++;
            } else {
                info.negative++;
            }
        }

        return info;
    }

    static double compute_inertia_regularization(const InertiaInfo &current,
                                                 int target_positive,
                                                 int target_negative,
                                                 double base_reg,
                                                 double factor) {
        // Determine how much regularization is needed
        const int excess_negative =
            std::max(0, current.negative - target_negative);
        const int insufficient_positive =
            std::max(0, target_positive - current.positive);

        if (excess_negative > 0 || insufficient_positive > 0 ||
            current.zero > 0) {
            return base_reg * factor *
                   std::max(1.0, double(excess_negative +
                                        insufficient_positive + current.zero));
        }

        return 0.0; // Inertia is correct
    }
};

// ==================== IP-Enhanced Regularization Strategy ====================
class IPRegularizer {
public:
    IPRegularizationConfig config_;

private:
    mutable double last_mu_{1.0};
    mutable double last_theta_{1.0};
    mutable double adaptive_reg_{0.0};

public:
    IPRegularizer(const IPRegularizationConfig &config = {})
        : config_(config) {}

    double compute_ip_regularization(double mu, double theta,
                                     const dvec &Sigma_x, const dvec &Sigma_s,
                                     int iteration = 0) const {
        if (!config_.use_ip_regularization)
            return 0.0;

        // Base μ-dependent regularization
        double mu_reg = config_.mu_dependent_reg * std::max(mu, 1e-12);

        // Constraint violation scaling
        double theta_reg = config_.constraint_scaling * std::min(theta, 1e3);

        // Barrier weight scaling
        double sigma_reg = 0.0;
        if (config_.sigma_scaling > 0.0) {
            double avg_sigma_x = Sigma_x.size() > 0 ? Sigma_x.mean() : 0.0;
            double avg_sigma_s = Sigma_s.size() > 0 ? Sigma_s.mean() : 0.0;
            sigma_reg =
                config_.sigma_scaling * std::max(avg_sigma_x, avg_sigma_s);
        }

        // Adaptive component based on progress
        double progress_factor = 1.0;
        if (iteration > 0) {
            const double mu_ratio = last_mu_ / std::max(mu, 1e-16);
            const double theta_ratio = last_theta_ / std::max(theta, 1e-16);

            // Increase regularization if progress is slow
            if (mu_ratio < 1.1 && theta_ratio < 1.1) {
                progress_factor = 2.0;
            } else if (mu_ratio > 2.0 || theta_ratio > 2.0) {
                progress_factor = 0.5;
            }
        }

        last_mu_ = mu;
        last_theta_ = theta;

        double total_reg = (mu_reg + theta_reg + sigma_reg) * progress_factor;
        total_reg = std::clamp(total_reg, config_.min_regularization,
                               config_.max_regularization);

        adaptive_reg_ = total_reg;
        return total_reg;
    }

    double get_last_regularization() const { return adaptive_reg_; }
};

// ==================== Enhanced HYKKT Strategy with IP Features
// ====================
class IPEnhancedHYKKTStrategy final : public KKTStrategy {
public:
    IPEnhancedHYKKTStrategy() {
        name = "ip_hykkt";
        regularizer_.emplace();
    }

    // Enhanced interface for IP-specific data
    std::tuple<dvec, dvec, std::shared_ptr<KKTReusable>> factor_and_solve_ip(
        ModelC *model_in, const spmat &W, const std::optional<spmat> &Gopt,
        const dvec &r1, const std::optional<dvec> &r2opt,
        const ChompConfig &cfg, dvec &Sigma_x, dvec &Sigma_s,
        double mu, double theta, int iteration = 0) {

        if (!Gopt || !r2opt) {
            throw std::invalid_argument(
                "IP-HYKKT requires equality constraints");
        }

        model = model_in;
        const auto &G = *Gopt;
        const auto &r2 = *r2opt;
        const int n = W.rows(), m = G.rows();

        // Compute IP-specific regularization
        double ip_reg = regularizer_->compute_ip_regularization(
            mu, theta, Sigma_x, Sigma_s, iteration);

        // Enhanced γ selection for IP
        double gamma = compute_ip_gamma(W, G, Sigma_x, Sigma_s, mu, theta, cfg);

        // Build system with inertia correction
        auto [Ks, final_reg] = create_ip_system_with_inertia_correction(
            W, G, Sigma_x, Sigma_s, ip_reg, gamma, n, m, cfg);

        if (!Ks) {
            throw std::runtime_error(
                "IP-HYKKT: failed to create system with correct inertia");
        }

        // Enhanced Schur complement solution
        const dvec s = r1 + gamma * (G.transpose() * r2);
        const dvec rhs_s = G * Ks(s) - r2;

        dvec dy =
            solve_ip_schur_complement(G, Ks, rhs_s, Sigma_x, Sigma_s, mu, cfg);
        const dvec dx = Ks(s - G.transpose() * dy);

        // Create IP-aware reusable solver
        auto reusable = create_ip_reusable_solver(G, Ks, gamma, mu, cfg);
        return std::make_tuple(dx, dy, reusable);
    }

    // Fallback to standard interface
    std::tuple<dvec, dvec, std::shared_ptr<KKTReusable>> factor_and_solve(
        ModelC *model_in, const spmat &W, const std::optional<spmat> &Gopt,
        const dvec &r1, const std::optional<dvec> &r2opt,
        const ChompConfig &cfg, std::optional<double> /*regularizer*/,
        std::unordered_map<std::string, dvec> & /*cache*/, double delta,
        std::optional<double> gamma_user, bool /*assemble_schur_if_m_small*/,
        bool /*use_prec*/) override {

        // Convert to IP interface with dummy values
        const int n = W.rows();
        auto [Sigma_x, Sigma_s, mu, theta, it] = model_in->get_parms();
        return factor_and_solve_ip(model_in, W, Gopt, r1, r2opt, cfg, Sigma_x,
                                   Sigma_s, mu, theta, it);
    }

private:
    ModelC *model = nullptr;
    mutable std::optional<IPRegularizer> regularizer_;
    mutable std::optional<qdldl23::LDL32> cached_factors_;

    double compute_ip_gamma(const spmat &W, const spmat &G, const dvec &Sigma_x,
                            const dvec &Sigma_s, double mu, double theta,
                            const ChompConfig &cfg) const {
        // Base heuristic
        double base_gamma = compute_gamma_heuristic(W, G, 0.0);

        // IP-specific adjustments
        double mu_scaling = std::max(1.0, std::sqrt(1.0 / std::max(mu, 1e-12)));
        double constraint_scaling = 1.0 + std::min(theta, 10.0);

        // Barrier weight scaling
        double sigma_scaling = 1.0;
        if (Sigma_s.size() > 0) {
            double avg_sigma = Sigma_s.mean();
            sigma_scaling = 1.0 + std::sqrt(avg_sigma);
        }

        double gamma =
            base_gamma * mu_scaling * constraint_scaling * sigma_scaling;

        // Clamp to reasonable range
        return std::clamp(gamma, 1e-3, 1e6);
    }

    std::pair<std::function<dvec(const dvec &)>, double>
    create_ip_system_with_inertia_correction(const spmat &W, const spmat &G,
                                             const dvec &Sigma_x,
                                             const dvec &Sigma_s,
                                             double base_reg, double gamma,
                                             int n, int m,
                                             const ChompConfig &cfg) const {

        double current_reg = base_reg;

        for (int attempt = 0;
             attempt < regularizer_->config_.max_inertia_corrections;
             ++attempt) {
            // Build augmented system with current regularization
            spmat K =
                build_ip_augmented_system(W, G, Sigma_x, current_reg, gamma);

            // Try factorization
            auto U = eigen_to_upper_csc(K);
            try {
                auto F = qdldl23::factorize(U);

                // Check inertia if enabled
                if (regularizer_->config_.use_inertia_correction) {
                    InertiaInfo inertia =
                        InertiaCorrector::compute_ldlt_inertia(F);

                    if (!inertia.has_correct_ip_inertia(n, m)) {
                        // Compute additional regularization needed
                        double add_reg =
                            InertiaCorrector::compute_inertia_regularization(
                                inertia, n, m, current_reg,
                                regularizer_->config_
                                    .inertia_regularization_factor);

                        current_reg += add_reg;
                        continue; // Try again with more regularization
                    }
                }

                // Success - create solver
                cached_factors_ = std::move(F);
                auto solver = [this, U = std::move(U)](const dvec &b) -> dvec {
                    dvec x = b;
                    qdldl23::solve(*cached_factors_, x.data());
                    return x;
                };

                return {solver, current_reg};

            } catch (const std::exception &) {
                // Factorization failed - increase regularization
                current_reg *=
                    regularizer_->config_.inertia_regularization_factor;
            }
        }

        return {nullptr, current_reg};
    }

    spmat build_ip_augmented_system(const spmat &W, const spmat &G,
                                    const dvec &Sigma_x, double reg,
                                    double gamma) const {
        spmat K = W;
        K.makeCompressed();

        // Add regularization and barrier terms
        for (int i = 0; i < K.rows(); ++i) {
            K.coeffRef(i, i) += reg + Sigma_x[i];
        }

        // Add constraint terms
        if (gamma > 0.0) {
            spmat GtG = (G.transpose() * G).eval();
            K += (gamma * GtG).pruned();
        }

        K.prune(1e-16);
        K.makeCompressed();
        return K;
    }

    dvec solve_ip_schur_complement(const spmat &G,
                                   const std::function<dvec(const dvec &)> &Ks,
                                   const dvec &rhs_s, const dvec &Sigma_x,
                                   const dvec &Sigma_s, double mu,
                                   const ChompConfig &cfg) const {

        const int m = G.rows();

        // Use barrier-aware preconditioner
        dvec barrier_prec =
            BarrierPreconditioner::compute_schur_barrier_preconditioner(
                G, Sigma_x, Sigma_s, mu);

        // Enhanced Schur operator with barrier awareness
        LinOp S_op{m, [&](const dvec &y, dvec &out) {
                       dvec Gty = G.transpose() * y;
                       dvec KGty = Ks(Gty);
                       out.noalias() = G * KGty;
                   }};

        // Solve with barrier-aware preconditioning
        auto [dy, info] =
            cg(S_op, rhs_s, barrier_prec,
               cfg.cg_tol * 0.1, // Tighter tolerance for IP
               cfg.cg_maxit, std::nullopt, std::nullopt, cfg.use_simd);

        if (!info.converged) {
            // Fallback with regularization
            double delta2 = std::max(1e-10, mu * 1e-2);
            LinOp S_reg{m, [&](const dvec &v, dvec &out) {
                            dvec Gtv = G.transpose() * v;
                            dvec KGtv = Ks(Gtv);
                            out.noalias() = G * KGtv;
                            out.noalias() += delta2 * v;
                        }};

            std::tie(dy, info) =
                cg(S_reg, rhs_s, barrier_prec, cfg.cg_tol, cfg.cg_maxit2,
                   std::nullopt, std::nullopt, cfg.use_simd);
        }

        if (!info.converged) {
            throw std::runtime_error("IP-HYKKT: Schur complement solve failed");
        }

        return dy;
    }

    std::shared_ptr<KKTReusable> create_ip_reusable_solver(
        const spmat &G, const std::function<dvec(const dvec &)> &Ks,
        double gamma, double mu, const ChompConfig &cfg) const {

        struct IPReusableSolver final : KKTReusable {
            spmat G;
            std::function<dvec(const dvec &)> Ks;
            double gamma, mu;
            ChompConfig cfg;
            mutable dvec last_dx, last_dy;
            mutable bool has_previous_solution{false};

            IPReusableSolver(spmat G_, std::function<dvec(const dvec &)> Ks_,
                             double gamma_, double mu_, ChompConfig cfg_)
                : G(std::move(G_)), Ks(std::move(Ks_)), gamma(gamma_), mu(mu_),
                  cfg(std::move(cfg_)) {}

            std::pair<dvec, dvec> solve(const dvec &r1n,
                                        const std::optional<dvec> &r2n,
                                        double tol, int maxit) override {
                if (!r2n)
                    throw std::invalid_argument("IP reusable solver needs r2");

                const dvec s = r1n + gamma * (G.transpose() * (*r2n));
                const dvec rhs_s = G * Ks(s) - (*r2n);

                // Use previous solution as warm start if available
                dvec dy_init = dvec::Zero(G.rows());
                if (has_previous_solution && last_dy.size() == G.rows()) {
                    // Scale previous solution (simple warm-starting)
                    dy_init = last_dy * 0.8;
                }

                LinOp S_op{static_cast<int>(G.rows()),
                           [&](const dvec &y, dvec &out) {
                               out = G * Ks(G.transpose() * y);
                           }};

                // More relaxed tolerance for reusable solver
                double reuse_tol = std::max(tol * 10, 1e-6);
                int reuse_maxit = std::max(maxit, 50);

                auto [dy, info] =
                    cg(S_op, rhs_s, std::nullopt, reuse_tol, reuse_maxit,
                       has_previous_solution ? std::optional<dvec>(dy_init)
                                             : std::nullopt,
                       std::nullopt, cfg.use_simd);

                if (!info.converged) {
                    // Fallback with μ-dependent regularization
                    double reg = std::max(1e-10, mu * 1e-3);
                    LinOp S_reg{static_cast<int>(G.rows()),
                                [&](const dvec &v, dvec &out) {
                                    out = G * Ks(G.transpose() * v);
                                    out.noalias() += reg * v;
                                }};

                    std::tie(dy, info) =
                        cg(S_reg, rhs_s, std::nullopt, reuse_tol * 10,
                           reuse_maxit * 2, std::nullopt, std::nullopt,
                           cfg.use_simd);
                }

                if (!info.converged) {
                    throw std::runtime_error(
                        "IP reusable solver failed to converge");
                }

                const dvec dx = Ks(s - G.transpose() * dy);

                // Store for next warm start
                last_dx = dx;
                last_dy = dy;
                has_previous_solution = true;

                return {dx, dy};
            }
        };

        return std::make_shared<IPReusableSolver>(G, Ks, gamma, mu, cfg);
    }

    // Helper functions (simplified versions of your existing ones)
    static double rowsum_inf_norm(const spmat &A) {
        double mx = 0.0;
        for (int j = 0; j < A.outerSize(); ++j)
            for (spmat::InnerIterator it(A, j); it; ++it)
                mx = std::max(mx, std::abs(it.value()));
        return mx;
    }

    double compute_gamma_heuristic(const spmat &W, const spmat &G,
                                   double delta) const {
        const double Wn = rowsum_inf_norm(W) + delta;
        const double Gn = rowsum_inf_norm(G);
        return std::max(1.0, Wn / std::max(1.0, Gn * Gn));
    }
};

// ==================== Factory Function ====================
inline std::shared_ptr<KKTStrategy>
create_ip_enhanced_strategy(const IPRegularizationConfig &ip_config = {}) {
    auto strategy = std::make_shared<IPEnhancedHYKKTStrategy>();
    // Configure IP-specific parameters through the strategy
    return strategy;
}

} // namespace kkt