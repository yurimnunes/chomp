#pragma once

#include "presolver.h" // namespace presolve
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

// ==============================
// Degeneracy hooks (no-op)
// ==============================
#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <deque>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

class DegeneracyManager {
public:
    explicit DegeneracyManager(int rng_seed = 13)
        : rng_(rng_seed), consecutive_degenerate_(0),
          perturbation_active_(false), total_degeneracies_(0),
          adaptive_threshold_(10), base_perturbation_scale_(1e-8),
          current_perturbation_scale_(1e-8), learning_rate_(0.1),
          success_decay_(0.95), failure_penalty_(1.2), min_threshold_(5),
          max_threshold_(50), step_history_size_(20),
          perturbation_success_rate_(0.0), total_perturbations_(0),
          successful_perturbations_(0) {
        // Note: std::deque doesn't have reserve(), but that's fine for
        // performance
    }

    // Main interface methods (maintain compatibility)
    bool detect_degeneracy(double step, double deg_step_tol) {
        // Store step history for trend analysis
        step_history_.push_back(step);
        if (step_history_.size() > step_history_size_) {
            step_history_.pop_front();
        }

        bool is_degenerate = is_step_degenerate(step, deg_step_tol);

        if (is_degenerate) {
            ++consecutive_degenerate_;
            ++total_degeneracies_;
            update_degeneracy_pattern();
        } else {
            if (perturbation_active_) {
                // Track perturbation success
                ++successful_perturbations_;
                perturbation_success_rate_ =
                    static_cast<double>(successful_perturbations_) /
                    total_perturbations_;
                adapt_parameters_on_success();
            }
            consecutive_degenerate_ = 0;
        }

        return is_degenerate;
    }

    bool should_apply_perturbation() const {
        return consecutive_degenerate_ > adaptive_threshold_ &&
               !perturbation_active_;
    }

    std::tuple<std::optional<Eigen::MatrixXd>, std::optional<Eigen::VectorXd>,
               std::optional<Eigen::VectorXd>>
    reset_perturbation() {
        if (perturbation_active_) {
            perturbation_active_ = false;
            consecutive_degenerate_ = 0;
            return {A_orig_, b_orig_, c_orig_};
        }
        return {std::nullopt, std::nullopt, std::nullopt};
    }

    std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd>
    apply_perturbation(const Eigen::MatrixXd &A, const Eigen::VectorXd &b,
                       const Eigen::VectorXd &c, const std::vector<int> &basis,
                       int iters) {
        if (!perturbation_active_) {
            A_orig_ = A;
            b_orig_ = b;
            c_orig_ = c;
            perturbation_active_ = true;
            ++total_perturbations_;

            // Adapt perturbation strategy based on problem characteristics
            analyze_problem_structure(A, b, c);
            current_perturbation_scale_ = compute_adaptive_scale(iters);
        }

        return apply_intelligent_perturbation(A, b, c, basis, iters);
    }

    std::unordered_map<std::string, std::string> get_statistics() const {
        return {
            {"degeneracy_detected", std::to_string(consecutive_degenerate_)},
            {"perturbation_active", perturbation_active_ ? "yes" : "no"},
            {"total_degeneracies", std::to_string(total_degeneracies_)},
            {"adaptive_threshold", std::to_string(adaptive_threshold_)},
            {"perturbation_success_rate",
             std::to_string(perturbation_success_rate_)},
            {"current_scale", std::to_string(current_perturbation_scale_)},
            {"problem_condition", std::to_string(estimated_condition_number_)},
            {"degeneracy_trend", get_trend_description()}};
    }

private:
    // Core state (maintain compatibility)
    std::mt19937 rng_;
    int consecutive_degenerate_;
    bool perturbation_active_;
    std::optional<Eigen::MatrixXd> A_orig_;
    std::optional<Eigen::VectorXd> b_orig_;
    std::optional<Eigen::VectorXd> c_orig_;

    // Advanced adaptive components
    int total_degeneracies_;
    int adaptive_threshold_;
    double base_perturbation_scale_;
    double current_perturbation_scale_;
    double learning_rate_;
    double success_decay_;
    double failure_penalty_;
    int min_threshold_;
    int max_threshold_;

    // Problem analysis
    double estimated_condition_number_ = 1.0;
    double problem_scale_ = 1.0;
    int problem_size_ = 0;

    // History tracking
    std::deque<double> step_history_;
    size_t step_history_size_;

    // Performance tracking
    double perturbation_success_rate_;
    int total_perturbations_;
    int successful_perturbations_;

    // Degeneracy pattern detection
    std::vector<int> degeneracy_positions_;
    int cycling_detected_ = 0;

    bool is_step_degenerate(double step, double deg_step_tol) const {
        // Enhanced degeneracy detection with relative tolerance
        double adaptive_tol =
            std::max(deg_step_tol, deg_step_tol * problem_scale_ * 1e-6);
        return step <= adaptive_tol;
    }

    void update_degeneracy_pattern() {
        degeneracy_positions_.push_back(consecutive_degenerate_);

        // Detect cycling patterns
        if (degeneracy_positions_.size() > 6) {
            detect_cycling_pattern();
        }

        // Keep only recent history
        if (degeneracy_positions_.size() > 100) {
            degeneracy_positions_.erase(degeneracy_positions_.begin(),
                                        degeneracy_positions_.begin() + 50);
        }
    }

    void detect_cycling_pattern() {
        // Simple cycle detection - look for repeating patterns
        int n = degeneracy_positions_.size();
        for (int cycle_len = 2; cycle_len <= 6 && cycle_len * 3 < n;
             ++cycle_len) {
            bool is_cycle = true;
            for (int i = 0; i < cycle_len; ++i) {
                if (degeneracy_positions_[n - 1 - i] !=
                        degeneracy_positions_[n - 1 - i - cycle_len] ||
                    degeneracy_positions_[n - 1 - i] !=
                        degeneracy_positions_[n - 1 - i - 2 * cycle_len]) {
                    is_cycle = false;
                    break;
                }
            }
            if (is_cycle) {
                cycling_detected_ = cycle_len;
                // More aggressive adaptation for cycling
                adaptive_threshold_ =
                    std::max(min_threshold_, adaptive_threshold_ / 2);
                break;
            }
        }
    }

    void adapt_parameters_on_success() {
        // Gradually increase threshold on success (less aggressive
        // perturbations)
        adaptive_threshold_ = std::min(
            max_threshold_, static_cast<int>(adaptive_threshold_ * 1.1));

        // Reduce perturbation scale slightly
        current_perturbation_scale_ *= success_decay_;
        current_perturbation_scale_ = std::max(current_perturbation_scale_,
                                               base_perturbation_scale_ * 0.1);
    }

    void analyze_problem_structure(const Eigen::MatrixXd &A,
                                   const Eigen::VectorXd &b,
                                   const Eigen::VectorXd &c) {
        problem_size_ = A.rows();

        // Estimate problem scale
        problem_scale_ =
            std::max({A.lpNorm<Eigen::Infinity>(), b.lpNorm<Eigen::Infinity>(),
                      c.lpNorm<Eigen::Infinity>()});

        // Rough condition number estimate using matrix norms
        if (A.rows() <= A.cols()) {
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(A.leftCols(A.rows()));
            auto singular_values = svd.singularValues();
            if (singular_values.size() > 0 &&
                singular_values(singular_values.size() - 1) > 1e-15) {
                estimated_condition_number_ =
                    singular_values(0) /
                    singular_values(singular_values.size() - 1);
            }
        }
    }

    double compute_adaptive_scale(int iters) const {
        double scale = base_perturbation_scale_;

        // Scale based on problem conditioning
        scale *= std::min(
            10.0, std::log10(std::max(1.0, estimated_condition_number_)) + 1.0);

        // Scale based on problem size
        scale *= std::sqrt(static_cast<double>(problem_size_)) / 10.0;

        // Scale based on iteration count (diminishing perturbations)
        scale *= (1.0 + iters * 1e-10);

        // Scale based on success rate
        if (total_perturbations_ > 3) {
            if (perturbation_success_rate_ < 0.3) {
                scale *= failure_penalty_; // Increase if not working well
            } else if (perturbation_success_rate_ > 0.7) {
                scale *= 0.8; // Decrease if working too well
            }
        }

        // Scale based on cycling detection
        if (cycling_detected_ > 0) {
            scale *= (1.0 + cycling_detected_ * 0.5);
        }

        return std::clamp(scale, base_perturbation_scale_ * 0.01,
                          base_perturbation_scale_ * 100.0);
    }

    std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd>
    apply_intelligent_perturbation(const Eigen::MatrixXd &A,
                                   const Eigen::VectorXd &b,
                                   const Eigen::VectorXd &c,
                                   const std::vector<int> &basis, int iters) {
        Eigen::MatrixXd A_pert = A;
        Eigen::VectorXd b_pert = b;
        Eigen::VectorXd c_pert = c;

        // Multi-strategy perturbation
        std::uniform_real_distribution<double> uniform_dist(-1.0, 1.0);
        std::normal_distribution<double> normal_dist(0.0, 1.0);

        // Strategy 1: Structured perturbation based on basis
        if (!basis.empty()) {
            // Perturb RHS more for basic variables
            for (int i = 0; i < b_pert.size(); ++i) {
                bool is_basic =
                    std::find(basis.begin(), basis.end(), i) != basis.end();
                double local_scale = current_perturbation_scale_;
                if (is_basic) {
                    local_scale *= 2.0; // More perturbation for basic variables
                }
                b_pert(i) += local_scale * normal_dist(rng_);
            }
        } else {
            // Fallback: uniform perturbation
            for (int i = 0; i < b_pert.size(); ++i) {
                b_pert(i) += current_perturbation_scale_ * uniform_dist(rng_);
            }
        }

        // Strategy 2: Objective perturbation for highly degenerate cases
        if (consecutive_degenerate_ > adaptive_threshold_ * 2) {
            for (int i = 0; i < c_pert.size(); ++i) {
                c_pert(i) +=
                    current_perturbation_scale_ * 0.1 * normal_dist(rng_);
            }
        }

        // Strategy 3: Matrix perturbation for extreme cases
        if (consecutive_degenerate_ > adaptive_threshold_ * 3) {
            for (int i = 0; i < A_pert.rows(); ++i) {
                for (int j = 0; j < A_pert.cols(); ++j) {
                    if (std::abs(A_pert(i, j)) > 1e-12) {
                        A_pert(i, j) += current_perturbation_scale_ * 0.01 *
                                        uniform_dist(rng_) *
                                        std::abs(A_pert(i, j));
                    }
                }
            }
        }

        return {A_pert, b_pert, c_pert};
    }

    std::string get_trend_description() const {
        if (step_history_.size() < 5)
            return "insufficient_data";

        // Simple trend analysis
        double recent_avg = 0.0, older_avg = 0.0;
        int half = step_history_.size() / 2;

        for (int i = 0; i < half; ++i) {
            older_avg += step_history_[i];
        }
        for (size_t i = half; i < step_history_.size(); ++i) {
            recent_avg += step_history_[i];
        }

        older_avg /= half;
        recent_avg /= (step_history_.size() - half);

        double trend_ratio = recent_avg / (older_avg + 1e-15);

        if (trend_ratio > 1.2)
            return "improving";
        else if (trend_ratio < 0.8)
            return "degrading";
        else
            return "stable";
    }
};
// ==============================
// True Steepest-Edge pricer (FT-consistent)
// ==============================
class SteepestEdgePricer {
public:
    struct Entry {
        int jN;
        Eigen::VectorXd t; // t = B^{-1} a_j
        double weight;     // 1 + ||t||^2
    };

    SteepestEdgePricer(int pool_max = 0, int reset_frequency = 1000)
        : pool_max_(pool_max), reset_freq_(reset_frequency), iter_count_(0) {}

    template <class BasisLike>
    void build_pool(const BasisLike &B, const Eigen::MatrixXd &A,
                    const std::vector<int> &N) {
        pool_.clear();
        pos_.clear();
        pool_.reserve(pool_max_ > 0 ? std::min<int>(pool_max_, (int)N.size())
                                    : (int)N.size());
        const int take = (pool_max_ > 0)
                             ? std::min<int>(pool_max_, (int)N.size())
                             : (int)N.size();
        for (int k = 0; k < take; ++k) {
            const int j = N[k];
            Entry e;
            e.jN = j;
            e.t = B.solve_B(A.col(j));
            e.weight = 1.0 + e.t.squaredNorm();
            pos_[j] = (int)pool_.size();
            pool_.push_back(std::move(e));
        }
        iter_count_ = 0;
    }

    std::optional<int> choose_entering(const Eigen::VectorXd &rcN,
                                       const std::vector<int> &N, double tol) {
        ++iter_count_;
        int best_rel = -1;
        double best_score = -1.0;
        for (int k = 0; k < (int)N.size(); ++k) {
            if (rcN(k) >= -tol)
                continue;
            const int j = N[k];
            double w = 1.0;
            auto it = pos_.find(j);
            if (it != pos_.end())
                w = pool_[it->second].weight;
            const double score = (rcN(k) * rcN(k)) / w;
            if (score > best_score) {
                best_score = score;
                best_rel = k;
            }
        }
        if (best_rel < 0)
            return std::nullopt;
        return best_rel;
    }

    void update_after_pivot(int leave_rel, int e_abs, int old_abs,
                            const Eigen::VectorXd &s, double alpha,
                            const Eigen::MatrixXd & /*A*/,
                            const std::vector<int> & /*N*/,
                            bool insert_leaver_into_pool = true) {
        if (std::abs(alpha) < 1e-14) {
            need_rebuild_ = true;
            return;
        }
        const double inv_alpha = 1.0 / alpha;
        for (auto &E : pool_) {
            const double tr = E.t(leave_rel);
            if (tr != 0.0) {
                E.t.noalias() -= s * (tr * inv_alpha);
                E.weight = 1.0 + E.t.squaredNorm();
            }
        }
        auto itE = pos_.find(e_abs);
        if (itE != pos_.end()) {
            int idx = itE->second;
            int last = (int)pool_.size() - 1;
            if (idx != last) {
                pos_[pool_[last].jN] = idx;
                std::swap(pool_[idx], pool_[last]);
            }
            pool_.pop_back();
            pos_.erase(itE);
        }
        if (insert_leaver_into_pool) {
            Entry E;
            E.jN = old_abs;
            E.t = Eigen::VectorXd::Zero(s.size());
            E.t(leave_rel) = 1.0;
            E.t.noalias() -= s * inv_alpha;
            E.weight = 1.0 + E.t.squaredNorm();
            if (pool_max_ > 0 && (int)pool_.size() >= pool_max_) {
                int evict = 0;
                double w = pool_[0].weight;
                for (int i = 1; i < (int)pool_.size(); ++i)
                    if (pool_[i].weight > w) {
                        w = pool_[i].weight;
                        evict = i;
                    }
                pos_.erase(pool_[evict].jN);
                pool_[evict] = std::move(E);
                pos_[pool_[evict].jN] = evict;
            } else {
                pos_[E.jN] = (int)pool_.size();
                pool_.push_back(std::move(E));
            }
        }
        ++iter_count_;
        if (need_rebuild_ || iter_count_ >= reset_freq_)
            need_rebuild_ = true;
    }

    bool needs_rebuild() const { return need_rebuild_; }
    void clear_rebuild_flag() { need_rebuild_ = false; }

private:
    std::vector<Entry> pool_;
    std::unordered_map<int, int> pos_;
    int pool_max_{0};
    int reset_freq_{1000};
    int iter_count_{0};
    bool need_rebuild_{false};
};

#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <deque>
#include <numeric>
#include <optional>
#include <random>
#include <unordered_map>
#include <vector>

// ==============================
// Devex Pricer (inherits from SteepestEdgePricer pattern)
// ==============================

class DevexPricer {
public:
    DevexPricer(double threshold = 0.99, int reset_frequency = 1000)
        : threshold_(threshold), reset_freq_(reset_frequency), iter_count_(0),
          rng_(std::random_device{}()) {}

    template <class BasisLike>
    void build_pool(const BasisLike &B, const Eigen::MatrixXd &A,
                    const std::vector<int> &N) {
        // Initialize devex weights to 1.0
        weights_.clear();
        for (int j : N) {
            weights_[j] = 1.0;
        }
        iter_count_ = 0;
    }

    std::optional<int> choose_entering(const Eigen::VectorXd &rcN,
                                       const std::vector<int> &N, double tol) {
        ++iter_count_;

        // Reset weights periodically
        if (iter_count_ % reset_freq_ == 0) {
            for (auto &[j, weight] : weights_) {
                weight = 1.0;
            }
        }

        int best_rel = -1;
        double best_criterion = -1.0;

        for (int k = 0; k < (int)N.size(); ++k) {
            if (rcN(k) >= -tol)
                continue;

            const int j = N[k];
            double weight = 1.0;
            auto it = weights_.find(j);
            if (it != weights_.end()) {
                weight = it->second;
            }

            // Devex criterion: |reduced_cost|^2 / weight
            double criterion = (rcN(k) * rcN(k)) / weight;

            if (criterion > best_criterion) {
                best_criterion = criterion;
                best_rel = k;
            }
        }

        return (best_rel >= 0) ? std::optional<int>(best_rel) : std::nullopt;
    }

    void update_after_pivot(int leave_rel, int e_abs, int old_abs,
                            const Eigen::VectorXd &pivot_column, double alpha,
                            const Eigen::MatrixXd & /*A*/,
                            const std::vector<int> &N,
                            bool /*insert_leaver_into_pool*/ = true) {
        if (std::abs(alpha) < 1e-14)
            return;

        // Update devex weight for entering variable
        if (weights_.find(e_abs) != weights_.end()) {
            weights_[e_abs] = alpha * alpha;
        }

        // Update weights for all nonbasic variables based on pivot column
        for (int k = 0; k < (int)N.size(); ++k) {
            int j = N[k];
            if (j == e_abs)
                continue;

            auto it = weights_.find(j);
            if (it != weights_.end() && leave_rel < pivot_column.size()) {
                double gamma = pivot_column(leave_rel) / alpha;
                double new_weight = it->second + gamma * gamma;
                it->second = std::max(new_weight, threshold_ * it->second);
            }
        }

        // Add leaving variable to weights if needed
        if (weights_.find(old_abs) == weights_.end()) {
            weights_[old_abs] = 1.0;
        }
    }

    bool needs_rebuild() const { return false; } // Devex doesn't need rebuilds
    void clear_rebuild_flag() {}

private:
    std::unordered_map<int, double> weights_;
    double threshold_;
    int reset_freq_;
    int iter_count_;
    std::mt19937 rng_;
};

// ==============================
// Dual Steepest Edge Pricer
// ==============================

class DualSteepestEdgePricer {
public:
    struct DualEntry {
        int jN;
        Eigen::VectorXd w;  // w = B^{-T} e_j (dual direction)
        double dual_weight; // ||w||^2
    };

    DualSteepestEdgePricer(int pool_max = 0, int reset_frequency = 1000)
        : pool_max_(pool_max), reset_freq_(reset_frequency), iter_count_(0) {}

    template <class BasisLike>
    void build_pool(const BasisLike &B, const Eigen::MatrixXd &A,
                    const std::vector<int> &N) {
        dual_pool_.clear();
        dual_pos_.clear();

        const int m = B.rows();
        const int take = (pool_max_ > 0)
                             ? std::min<int>(pool_max_, (int)N.size())
                             : (int)N.size();

        dual_pool_.reserve(take);

        for (int k = 0; k < take; ++k) {
            const int j = N[k];
            DualEntry e;
            e.jN = j;

            // Compute dual direction w = B^{-T} * A_j^T (simplified)
            Eigen::VectorXd Aj = A.col(j);
            e.w = B.solve_BT(Aj);
            e.dual_weight = e.w.squaredNorm();

            dual_pos_[j] = (int)dual_pool_.size();
            dual_pool_.push_back(std::move(e));
        }
        iter_count_ = 0;
    }

    std::optional<int> choose_entering(const Eigen::VectorXd &rcN,
                                       const std::vector<int> &N, double tol) {
        ++iter_count_;

        int best_rel = -1;
        double best_score = -1.0;

        for (int k = 0; k < (int)N.size(); ++k) {
            if (rcN(k) >= -tol)
                continue;

            const int j = N[k];
            double dual_weight = 1.0;

            auto it = dual_pos_.find(j);
            if (it != dual_pos_.end()) {
                dual_weight = std::max(1.0, dual_pool_[it->second].dual_weight);
            }

            // Dual steepest edge criterion
            double score = (rcN(k) * rcN(k)) / dual_weight;

            if (score > best_score) {
                best_score = score;
                best_rel = k;
            }
        }

        return (best_rel >= 0) ? std::optional<int>(best_rel) : std::nullopt;
    }

    void update_after_pivot(int leave_rel, int e_abs, int old_abs,
                            const Eigen::VectorXd &s, double alpha,
                            const Eigen::MatrixXd & /*A*/,
                            const std::vector<int> & /*N*/,
                            bool insert_leaver_into_pool = true) {
        if (std::abs(alpha) < 1e-14) {
            need_rebuild_ = true;
            return;
        }

        const double inv_alpha = 1.0 / alpha;

        // Update dual directions
        for (auto &E : dual_pool_) {
            if (leave_rel < E.w.size()) {
                const double wr = E.w(leave_rel);
                if (wr != 0.0) {
                    // Update dual direction: w = w - (w_r / alpha) *
                    // dual_pivot_col Simplified update - in practice would need
                    // proper dual pivot column
                    E.w(leave_rel) = -wr * inv_alpha;
                    E.dual_weight = E.w.squaredNorm();
                }
            }
        }

        // Remove entering variable from pool
        auto itE = dual_pos_.find(e_abs);
        if (itE != dual_pos_.end()) {
            int idx = itE->second;
            int last = (int)dual_pool_.size() - 1;
            if (idx != last) {
                dual_pos_[dual_pool_[last].jN] = idx;
                std::swap(dual_pool_[idx], dual_pool_[last]);
            }
            dual_pool_.pop_back();
            dual_pos_.erase(itE);
        }

        // Add leaving variable to pool
        if (insert_leaver_into_pool) {
            DualEntry E;
            E.jN = old_abs;
            E.w = Eigen::VectorXd::Zero(s.size());
            if (leave_rel < E.w.size()) {
                E.w(leave_rel) = 1.0;
            }
            E.dual_weight = E.w.squaredNorm();

            if (pool_max_ > 0 && (int)dual_pool_.size() >= pool_max_) {
                // Evict entry with largest weight
                int evict = 0;
                double max_weight = dual_pool_[0].dual_weight;
                for (int i = 1; i < (int)dual_pool_.size(); ++i) {
                    if (dual_pool_[i].dual_weight > max_weight) {
                        max_weight = dual_pool_[i].dual_weight;
                        evict = i;
                    }
                }
                dual_pos_.erase(dual_pool_[evict].jN);
                dual_pool_[evict] = std::move(E);
                dual_pos_[dual_pool_[evict].jN] = evict;
            } else {
                dual_pos_[E.jN] = (int)dual_pool_.size();
                dual_pool_.push_back(std::move(E));
            }
        }

        ++iter_count_;
        if (iter_count_ >= reset_freq_) {
            need_rebuild_ = true;
        }
    }

    bool needs_rebuild() const { return need_rebuild_; }
    void clear_rebuild_flag() { need_rebuild_ = false; }

private:
    std::vector<DualEntry> dual_pool_;
    std::unordered_map<int, int> dual_pos_;
    int pool_max_;
    int reset_freq_;
    int iter_count_;
    bool need_rebuild_ = false;
};

// ==============================
// Enhanced Adaptive Pricer
// ==============================

class AdaptivePricer {
public:
    enum Strategy {
        STEEPEST_EDGE,
        DEVEX,
        PARTIAL_PRICING,
        DUAL_STEEPEST,
        MOST_NEGATIVE
    };

    struct PricingOptions {
        Strategy initial_strategy = STEEPEST_EDGE;
        int switch_threshold = 100;
        int performance_window = 50;
        double improvement_factor = 1.2;
        int partial_block_factor = 10;
        int min_partial_block = 10;
        bool enable_adaptive_switching = true;
        int steepest_pool_max = 0;
        int steepest_reset_freq = 1000;
        int devex_reset_freq = 1000;
        int dual_steepest_pool_max = 0;
        int dual_steepest_reset_freq = 1000;

        // Default constructor
        PricingOptions() = default;
    };

    struct PricingStats {
        int total_pricing_calls = 0;
        int strategy_switches = 0;
        double avg_improvement_per_iteration = 0.0;
        std::vector<int> strategy_usage_count;

        PricingStats() : strategy_usage_count(5, 0) {}
    };

private:
    Strategy current_strategy_;
    PricingOptions options_;
    std::deque<double> performance_history_;
    std::deque<double> recent_objectives_;
    int n_;
    mutable PricingStats stats_;

    // Pricer instances
    SteepestEdgePricer steepest_pricer_;
    DevexPricer devex_pricer_;
    DualSteepestEdgePricer dual_steepest_pricer_;

    // Performance tracking
    int iterations_since_switch_;
    double last_objective_;
    bool first_call_;

public:
    explicit AdaptivePricer(int n) : AdaptivePricer(n, PricingOptions()) {}

    AdaptivePricer(int n, const PricingOptions &opts)
        : current_strategy_(opts.initial_strategy), options_(opts), n_(n),
          steepest_pricer_(opts.steepest_pool_max, opts.steepest_reset_freq),
          devex_pricer_(0.99, opts.devex_reset_freq),
          dual_steepest_pricer_(opts.dual_steepest_pool_max,
                                opts.dual_steepest_reset_freq),
          iterations_since_switch_(0), last_objective_(0.0), first_call_(true) {
        // Note: std::deque doesn't have reserve(), but manages memory
        // efficiently
        stats_.strategy_usage_count.resize(5, 0);
    }

    // Main pricing interface
    template <typename BasisLike>
    std::optional<int>
    choose_entering(const Eigen::VectorXd &rN, const std::vector<int> &N,
                    double tol, int iteration, double current_objective,
                    const BasisLike &basis, const Eigen::MatrixXd &A) {
        ++stats_.total_pricing_calls;
        ++stats_.strategy_usage_count[current_strategy_];

        // Track performance for adaptive switching
        track_performance(current_objective);

        // Adaptive strategy switching
        if (options_.enable_adaptive_switching &&
            should_switch_strategy(iteration)) {
            adapt_strategy();
            // Rebuild pools when switching strategies
            rebuild_pools(basis, A, N);
        }

        std::optional<int> result;

        switch (current_strategy_) {
        case STEEPEST_EDGE:
            result = steepest_pricer_.choose_entering(rN, N, tol);
            break;
        case DEVEX:
            result = devex_pricer_.choose_entering(rN, N, tol);
            break;
        case PARTIAL_PRICING:
            result = partial_pricing(rN, N, tol, iteration);
            break;
        case DUAL_STEEPEST:
            result = dual_steepest_pricer_.choose_entering(rN, N, tol);
            break;
        case MOST_NEGATIVE:
            result = most_negative_pricing(rN, N, tol);
            break;
        }

        return result;
    }

    // Initialize/rebuild pools for current strategy
    template <typename BasisLike>
    void build_pools(const BasisLike &basis, const Eigen::MatrixXd &A,
                     const std::vector<int> &N) {
        steepest_pricer_.build_pool(basis, A, N);
        devex_pricer_.build_pool(basis, A, N);
        dual_steepest_pricer_.build_pool(basis, A, N);
    }

    // Update method to be called after each pivot
    void update_after_pivot(int leaving_rel, int entering_abs, int old_abs,
                            const Eigen::VectorXd &pivot_column, double alpha,
                            double step_size, const Eigen::MatrixXd &A,
                            const std::vector<int> &N) {
        // Update all pricers
        steepest_pricer_.update_after_pivot(leaving_rel, entering_abs, old_abs,
                                            pivot_column, alpha, A, N, true);
        devex_pricer_.update_after_pivot(leaving_rel, entering_abs, old_abs,
                                         pivot_column, alpha, A, N, true);
        dual_steepest_pricer_.update_after_pivot(leaving_rel, entering_abs,
                                                 old_abs, pivot_column, alpha,
                                                 A, N, true);

        // Track step size for performance evaluation
        if (performance_history_.size() >= options_.performance_window) {
            performance_history_.pop_front();
        }
        performance_history_.push_back(step_size);
    }

    // Check if any pricer needs rebuilding
    bool needs_rebuild() const {
        switch (current_strategy_) {
        case STEEPEST_EDGE:
            return steepest_pricer_.needs_rebuild();
        case DUAL_STEEPEST:
            return dual_steepest_pricer_.needs_rebuild();
        default:
            return false;
        }
    }

    void clear_rebuild_flag() {
        steepest_pricer_.clear_rebuild_flag();
        dual_steepest_pricer_.clear_rebuild_flag();
    }

    const char *get_current_strategy_name() const {
        switch (current_strategy_) {
        case STEEPEST_EDGE:
            return "steepest_edge";
        case DEVEX:
            return "devex";
        case PARTIAL_PRICING:
            return "partial_pricing";
        case DUAL_STEEPEST:
            return "dual_steepest";
        case MOST_NEGATIVE:
            return "most_negative";
        }
        return "unknown";
    }

    const PricingStats &get_stats() const { return stats_; }

    void reset(int new_n) {
        n_ = new_n;
        current_strategy_ = options_.initial_strategy;
        performance_history_.clear();
        recent_objectives_.clear();
        iterations_since_switch_ = 0;
        first_call_ = true;
        stats_ = PricingStats();
        stats_.strategy_usage_count.resize(5, 0);
    }

private:
    template <typename BasisLike>
    void rebuild_pools(const BasisLike &basis, const Eigen::MatrixXd &A,
                       const std::vector<int> &N) {
        switch (current_strategy_) {
        case STEEPEST_EDGE:
            steepest_pricer_.build_pool(basis, A, N);
            break;
        case DEVEX:
            devex_pricer_.build_pool(basis, A, N);
            break;
        case DUAL_STEEPEST:
            dual_steepest_pricer_.build_pool(basis, A, N);
            break;
        default:
            break; // No pools needed for other strategies
        }
    }

    void track_performance(double current_objective) {
        if (!first_call_) {
            double improvement = std::abs(current_objective - last_objective_);
            if (recent_objectives_.size() >= 10) {
                recent_objectives_.pop_front();
            }
            recent_objectives_.push_back(improvement);
        }
        last_objective_ = current_objective;
        first_call_ = false;
    }

    void adapt_strategy() {
        if (recent_objectives_.size() < 20)
            return;

        ++stats_.strategy_switches;

        double recent_avg = std::accumulate(recent_objectives_.end() - 10,
                                            recent_objectives_.end(), 0.0) /
                            10.0;
        double older_avg =
            std::accumulate(recent_objectives_.begin(),
                            recent_objectives_.begin() + 10, 0.0) /
            10.0;

        if (recent_avg < older_avg / options_.improvement_factor) {
            if (n_ > 10000) {
                current_strategy_ = (current_strategy_ == PARTIAL_PRICING)
                                        ? DEVEX
                                        : PARTIAL_PRICING;
            } else if (performance_history_.size() > 0) {
                double avg_step =
                    std::accumulate(performance_history_.begin(),
                                    performance_history_.end(), 0.0) /
                    performance_history_.size();
                if (avg_step < 1e-10) {
                    current_strategy_ = STEEPEST_EDGE;
                } else {
                    current_strategy_ =
                        static_cast<Strategy>((current_strategy_ + 1) % 5);
                }
            } else {
                current_strategy_ =
                    static_cast<Strategy>((current_strategy_ + 1) % 5);
            }
        }

        iterations_since_switch_ = 0;
    }

    bool should_switch_strategy(int iteration) {
        ++iterations_since_switch_;
        return iterations_since_switch_ >= options_.switch_threshold;
    }

    std::optional<int> partial_pricing(const Eigen::VectorXd &rN,
                                       const std::vector<int> &N, double tol,
                                       int iteration) {
        int block_size = std::max(options_.min_partial_block,
                                  static_cast<int>(N.size()) /
                                      options_.partial_block_factor);
        int start_idx = (iteration * block_size) % N.size();

        int best_idx = -1;
        double best_rc = 0.0;

        for (int k = 0; k < block_size && k < (int)N.size(); ++k) {
            int idx = (start_idx + k) % N.size();
            if (rN(idx) < -tol && rN(idx) < best_rc) {
                best_rc = rN(idx);
                best_idx = idx;
            }
        }

        return (best_idx >= 0) ? std::optional<int>(best_idx) : std::nullopt;
    }

    std::optional<int> most_negative_pricing(const Eigen::VectorXd &rN,
                                             const std::vector<int> &N,
                                             double tol) {
        int best_idx = -1;
        double best_rc = 0.0;

        for (int k = 0; k < (int)N.size(); ++k) {
            if (rN(k) < -tol && rN(k) < best_rc) {
                best_rc = rN(k);
                best_idx = k;
            }
        }

        return (best_idx >= 0) ? std::optional<int>(best_idx) : std::nullopt;
    }
};