#pragma once
// Degeneracy manager + pricers (header-only, drop-in compatible)

#include "presolver.h" // namespace presolve (used by callers; not required here)

// STL
#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

// Eigen
#include <Eigen/Dense>
#include <Eigen/SVD>

// -----------------------------------------------------------------------------
// Forward declaration (to decouple from AdaptivePricer below)
// -----------------------------------------------------------------------------
struct AdaptivePricerLike {
    enum Strategy {
        STEEPEST_EDGE,
        DEVEX,
        PARTIAL_PRICING,
        DUAL_STEEPEST,
        MOST_NEGATIVE
    };
};

// ============================================================================
// DegeneracyManager
//  - API preserved exactly
//  - Internals tidied up, comments clarified
// ============================================================================
class DegeneracyManager {
public:
    // Signals sent to the pricer (nudge only; pricer remains authoritative)
    struct DegeneracySignals {
        std::optional<AdaptivePricerLike::Strategy> preferred_strategy{};
        bool request_pool_rebuild{false};
        std::vector<int> forbid_rel_candidates; // relative indices in N
        std::unordered_map<int, double> weight_overrides; // ABS N indices
        std::unordered_map<int, std::vector<double>> lex_order; // ABS N indices
        bool encourage_partial_pricing{false};
        bool cycling_alert{false};
        int epoch{0};
    };

    enum class Method {
        PERTURBATION,
        LEXICOGRAPHIC,
        BLAND,
        STEEPEST_EDGE,
        DEVEX,
        DUAL_SIMPLEX,
        PRIMAL_DUAL,
        HYBRID
    };

    explicit DegeneracyManager(int rng_seed = 13,
                               Method default_method = Method::HYBRID)
        : rng_(rng_seed),
          default_method_(default_method),
          current_method_(default_method) {
        reset();
        method_perf_.reserve(8);
    }

    // ---------------- Backward-compatible core ----------------

    // Heuristic: a very small primal step is degeneracy
    bool detect_degeneracy(double step, double deg_step_tol) {
        push_step_(step);
        const double tol = std::max(deg_step_tol, 1e-16 * scale_hint_);
        const bool deg = (step <= tol);
        if (deg) {
            ++deg_streak_;
            ++deg_total_;
            update_cycle_signal_();
        } else {
            if (deg_streak_ > 0) ++successes_recent_;
            deg_streak_ = 0;
            cycling_len_ = 0;
        }
        return deg;
    }

    // Legacy toggle (kept for compatibility – prefer pricer-based anti-cycling)
    bool should_apply_perturbation() const {
        return (deg_streak_ > std::max(10, adaptive_deg_threshold_) && !perturb_on_);
    }

    // Compatibility: does not modify A,b,c anymore
    std::tuple<std::optional<Eigen::MatrixXd>, std::optional<Eigen::VectorXd>,
               std::optional<Eigen::VectorXd>>
    reset_perturbation() {
        perturb_on_ = false;
        return {std::nullopt, std::nullopt, std::nullopt};
    }

    // Compatibility: returns inputs (no-op)
    std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd>
    apply_perturbation(const Eigen::MatrixXd &A,
                       const Eigen::VectorXd &b,
                       const Eigen::VectorXd &c,
                       const std::vector<int> &/*basis*/,
                       int /*iters*/) {
        return {A, b, c};
    }

    // ---------------- New hooks for the pricer loop ----------------

    // Call at start of pricing
    const DegeneracySignals &
    begin_pricing(double objective,
                  int iter,
                  int n_nonbasic,
                  std::optional<double> illcond_hint = std::nullopt) {
        ++epoch_;
        iter_ = iter;
        last_obj_impr_ = std::max(0.0, last_obj_ - objective);
        last_obj_ = objective;
        nN_ = n_nonbasic;
        if (illcond_hint) cond_est_ = *illcond_hint;

        signals_ = {};
        signals_.epoch = epoch_;

        // Cycling suspicion → nudge for partial pricing & robust criteria
        if (cycling_len_ >= 3 || (deg_streak_ >= 8 && small_recent_improvement_())) {
            signals_.cycling_alert = true;
            signals_.encourage_partial_pricing = true;

            if (cond_est_ > 1e9) {
                signals_.preferred_strategy = AdaptivePricerLike::DUAL_STEEPEST;
            } else {
                signals_.preferred_strategy =
                    (iter % 2 == 0) ? AdaptivePricerLike::DEVEX
                                    : AdaptivePricerLike::STEEPEST_EDGE;
            }
            signals_.request_pool_rebuild = true;
        }

        // Lex tie-break seeds when degeneracy is ongoing
        if (deg_streak_ >= 3) {
            std::uniform_real_distribution<double> eps(-1e-14, 1e-14);
            for (int jAbs = 0; jAbs < nN_; ++jAbs) {
                signals_.lex_order[jAbs] = {static_cast<double>(jAbs), eps(rng_)};
            }
        }

        // Ill-conditioning → gently increase weights (more conservative)
        if (cond_est_ > 1e10) {
            for (int jAbs = 0; jAbs < nN_; ++jAbs) {
                signals_.weight_overrides[jAbs] = 2.0; // heavier
            }
        }

        // Recurrent suspects (light forbid), only if degeneracy persists
        trim_repeat_window_();
        if (!repeat_rel_block_.empty() && deg_streak_ >= 4) {
            signals_.forbid_rel_candidates.assign(repeat_rel_block_.begin(),
                                                  repeat_rel_block_.end());
        }

        return signals_;
    }

    // Optional filter (in-place) for relative candidate indices
    void filter_candidates_in_place(std::vector<int> &rel_candidates) const {
        if (signals_.forbid_rel_candidates.empty()) return;
        std::set<int> forbid(signals_.forbid_rel_candidates.begin(),
                             signals_.forbid_rel_candidates.end());
        std::vector<int> keep; keep.reserve(rel_candidates.size());
        for (int k : rel_candidates) if (!forbid.count(k)) keep.push_back(k);
        rel_candidates.swap(keep);
    }

    // Call right after pivot is applied
    void after_pivot(int /*leaving_rel*/,
                     int entering_abs,
                     double step_alpha,
                     double rc_improvement,
                     double step_norm) {
        last_rc_impr_   = rc_improvement;
        last_step_norm_ = step_norm;

        if (std::abs(step_alpha) <= 1e-14) { // degenerate pivot
            ++deg_streak_;
            ++deg_total_;
            push_repeat_(entering_abs);
            update_cycle_signal_();
        } else {
            ++successes_recent_;
            deg_streak_ = 0;
            cycling_len_ = 0;
            repeat_rel_block_.clear();
        }
        tune_thresholds_();
        scale_hint_ = std::max(scale_hint_, last_step_norm_);
    }

    // Lightweight stats for logging
    struct Stats {
        int degeneracy_streak{0};
        int degeneracy_total{0};
        int suspected_cycling{0};
        double cond_est{0.0};
        int adaptive_deg_threshold{10};
        int epoch{0};
    };
    Stats get_stats() const {
        return Stats{deg_streak_, deg_total_, cycling_len_, cond_est_,
                     adaptive_deg_threshold_, epoch_};
    }

    // Manual knobs
    void set_method(Method m) { current_method_ = m; }
    Method method() const { return current_method_; }

    void reset() {
        step_hist_.clear();
        deg_streak_ = 0;
        deg_total_ = 0;
        cycling_len_ = 0;
        successes_recent_ = 0;
        adaptive_deg_threshold_ = 10;
        scale_hint_ = 1.0;
        cond_est_ = 1.0;
        epoch_ = 0;
        last_obj_ = 0.0;
        last_obj_impr_ = 0.0;
        last_rc_impr_ = 0.0;
        last_step_norm_ = 0.0;
        repeat_window_.clear();
        repeat_rel_block_.clear();
        nN_ = 0;
        perturb_on_ = false;
        current_method_ = default_method_;
        signals_ = {};
    }

private:
    // --- small helpers ---
    void push_step_(double s) {
        if (step_hist_.size() >= 32) step_hist_.pop_front();
        step_hist_.push_back(s);
    }
    bool small_recent_improvement_() const {
        const bool obj_stalled = (last_obj_impr_ < 1e-10);
        const bool rc_stalled  = (std::abs(last_rc_impr_) < 1e-12);
        return obj_stalled && rc_stalled;
    }
    void push_repeat_(int entering_abs) {
        if (repeat_window_.size() >= 16) repeat_window_.pop_front();
        repeat_window_.push_back(entering_abs);
        std::unordered_map<int,int> freq;
        for (int j : repeat_window_) ++freq[j];
        repeat_rel_block_.clear();
        for (auto &kv : freq) if (kv.second >= 3) repeat_rel_block_.push_back(kv.first);
    }
    void update_cycle_signal_() {
        if (deg_streak_ >= 6) cycling_len_ = std::max(cycling_len_, 3);
    }
    void tune_thresholds_() {
        if (successes_recent_ >= 5) {
            adaptive_deg_threshold_ = std::min(50, int(adaptive_deg_threshold_ * 1.1 + 1));
            successes_recent_ = 0;
        } else if (deg_streak_ >= adaptive_deg_threshold_) {
            adaptive_deg_threshold_ = std::max(4, int(adaptive_deg_threshold_ * 0.8));
        }
    }
    void trim_repeat_window_() {
        if (deg_streak_ == 0 && !repeat_window_.empty() && repeat_window_.size() > 8) {
            repeat_window_.erase(repeat_window_.begin(),
                                 repeat_window_.begin() + int(repeat_window_.size())/2);
        }
    }

private:
    // RNG & config
    std::mt19937 rng_;
    Method       default_method_;
    Method       current_method_;

    // State
    int iter_{0};
    int epoch_{0};
    int nN_{0};

    // Degeneracy / cycling
    int  deg_streak_{0};
    int  deg_total_{0};
    int  cycling_len_{0};
    int  adaptive_deg_threshold_{10};
    int  successes_recent_{0};
    bool perturb_on_{false};

    // Scales & telemetry
    double scale_hint_{1.0};
    double cond_est_{1.0};
    double last_obj_{0.0};
    double last_obj_impr_{0.0};
    double last_rc_impr_{0.0};
    double last_step_norm_{0.0};

    // Histories
    std::deque<double> step_hist_;
    std::deque<int>    repeat_window_;
    std::vector<int>   repeat_rel_block_;

    // (future) method performance bookkeeping
    struct Perf { int tries{0}, wins{0}; double delta{0.0}; };
    std::vector<Perf> method_perf_;

    // Outgoing signals
    DegeneracySignals signals_;
};

// ============================================================================
// DegeneracyPricerBridge
//  - Thin adapter to thread DegeneracyManager signals into your pricer
// ============================================================================
template <class AdaptivePricer>
struct DegeneracyPricerBridge {
    DegeneracyManager &dm;
    AdaptivePricer    &pricer;

    DegeneracyPricerBridge(DegeneracyManager &dm_, AdaptivePricer &pr_)
        : dm(dm_), pricer(pr_) {}

    template <class BasisLike>
    std::optional<int>
    choose_entering(const Eigen::VectorXd &rN, const std::vector<int> &N,
                    double tol, int iteration, double current_objective,
                    const BasisLike &basis, const Eigen::MatrixXd &A) {
        const auto &sig = dm.begin_pricing(current_objective, iteration, int(N.size()));

        if (sig.request_pool_rebuild) {
            pricer.build_pools(basis, A, N);
        }

        auto entering_rel =
            pricer.choose_entering(rN, N, tol, iteration, current_objective, basis, A);

        // If chosen entry is on forbid list, try a quick local fallback
        if (entering_rel && !sig.forbid_rel_candidates.empty()) {
            std::set<int> forbid(sig.forbid_rel_candidates.begin(),
                                 sig.forbid_rel_candidates.end());
            if (forbid.count(*entering_rel)) {
                int best = -1; double best_rc = 0.0;
                for (int k = 0; k < (int)N.size(); ++k) {
                    if (rN(k) < -tol && !forbid.count(k) && rN(k) < best_rc) {
                        best_rc = rN(k); best = k;
                    }
                }
                if (best >= 0) entering_rel = best;
            }
        }
        return entering_rel;
    }

    void after_pivot(int leaving_rel, int entering_abs, int old_abs,
                     const Eigen::VectorXd &pivot_column, double alpha,
                     double step_size, const Eigen::MatrixXd &A,
                     const std::vector<int> &N, double rc_improvement = 0.0) {
        pricer.update_after_pivot(leaving_rel, entering_abs, old_abs,
                                  pivot_column, alpha, step_size, A, N);
        dm.after_pivot(leaving_rel, entering_abs, alpha, rc_improvement, step_size);

        if (pricer.needs_rebuild()) {
            pricer.clear_rebuild_flag();
        }
    }
};

// ============================================================================
// SteepestEdgePricer (true steepest-edge; FT-consistent update)
// ============================================================================
class SteepestEdgePricer {
public:
    struct Entry {
        int jN;                 // absolute col index
        Eigen::VectorXd t;      // B^{-1} a_j
        double          weight; // 1 + ||t||^2
    };

    explicit SteepestEdgePricer(int pool_max = 0, int reset_frequency = 1000)
        : pool_max_(pool_max), reset_freq_(reset_frequency) {}

    template <class BasisLike>
    void build_pool(const BasisLike &B,
                    const Eigen::MatrixXd &A,
                    const std::vector<int> &N) {
        pool_.clear(); pos_.clear();
        const int take = (pool_max_ > 0) ? std::min<int>(pool_max_, (int)N.size())
                                         : (int)N.size();
        pool_.reserve(take);
        for (int k = 0; k < take; ++k) {
            const int j = N[k];
            Entry e;
            e.jN = j;
            e.t = B.solve_B(A.col(j));   // caller-provided
            e.weight = 1.0 + e.t.squaredNorm();
            pos_[j] = (int)pool_.size();
            pool_.push_back(std::move(e));
        }
        iter_count_ = 0;
        need_rebuild_ = false;
    }

    std::optional<int>
    choose_entering(const Eigen::VectorXd &rcN,
                    const std::vector<int> &N,
                    double tol) {
        ++iter_count_;
        int best_rel = -1;
        double best_score = -1.0;

        for (int k = 0; k < (int)N.size(); ++k) {
            if (rcN(k) >= -tol) continue;
            const int j = N[k];
            double w = 1.0;
            if (auto it = pos_.find(j); it != pos_.end()) w = pool_[it->second].weight;
            const double score = (rcN(k) * rcN(k)) / w;
            if (score > best_score) { best_score = score; best_rel = k; }
        }
        return (best_rel >= 0) ? std::optional<int>(best_rel) : std::nullopt;
    }

    void update_after_pivot(int leave_rel, int e_abs, int old_abs,
                            const Eigen::VectorXd &s, double alpha,
                            const Eigen::MatrixXd &/*A*/,
                            const std::vector<int> &/*N*/,
                            bool insert_leaver_into_pool = true) {
        if (std::abs(alpha) < 1e-14) { need_rebuild_ = true; return; }

        const double inv_alpha = 1.0 / alpha;

        // Update t, weight
        for (auto &E : pool_) {
            if (leave_rel < E.t.size()) {
                const double tr = E.t(leave_rel);
                if (tr != 0.0) {
                    E.t.noalias() -= s * (tr * inv_alpha);
                    E.weight = 1.0 + E.t.squaredNorm();
                }
            }
        }

        // Remove entering from pool
        if (auto itE = pos_.find(e_abs); itE != pos_.end()) {
            const int idx = itE->second;
            const int last = (int)pool_.size() - 1;
            if (idx != last) {
                pos_[pool_[last].jN] = idx;
                std::swap(pool_[idx], pool_[last]);
            }
            pool_.pop_back();
            pos_.erase(itE);
        }

        // Optionally add leaving
        if (insert_leaver_into_pool) {
            Entry E;
            E.jN = old_abs;
            E.t  = Eigen::VectorXd::Zero(s.size());
            if (leave_rel < E.t.size()) E.t(leave_rel) = 1.0;
            E.t.noalias() -= s * inv_alpha;
            E.weight = 1.0 + E.t.squaredNorm();

            if (pool_max_ > 0 && (int)pool_.size() >= pool_max_) {
                // Evict largest-weight entry
                int evict = 0; double wmax = pool_[0].weight;
                for (int i = 1; i < (int)pool_.size(); ++i) {
                    if (pool_[i].weight > wmax) { wmax = pool_[i].weight; evict = i; }
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
        if (need_rebuild_ || iter_count_ >= reset_freq_) need_rebuild_ = true;
    }

    bool needs_rebuild() const { return need_rebuild_; }
    void clear_rebuild_flag()  { need_rebuild_ = false; }

private:
    std::vector<Entry>              pool_;
    std::unordered_map<int, int>    pos_;
    int   pool_max_{0};
    int   reset_freq_{1000};
    int   iter_count_{0};
    bool  need_rebuild_{false};
};

// ============================================================================
// DevexPricer (lightweight weights; API preserved)
// ============================================================================
class DevexPricer {
public:
    explicit DevexPricer(double threshold = 0.99, int reset_frequency = 1000)
        : threshold_(threshold), reset_freq_(reset_frequency) {}

    template <class BasisLike>
    void build_pool(const BasisLike &/*B*/,
                    const Eigen::MatrixXd &/*A*/,
                    const std::vector<int> &N) {
        weights_.clear();
        for (int j : N) weights_[j] = 1.0;
        iter_count_ = 0;
    }

    std::optional<int>
    choose_entering(const Eigen::VectorXd &rcN,
                    const std::vector<int> &N,
                    double tol) {
        ++iter_count_;
        if (iter_count_ % reset_freq_ == 0) {
            for (auto &p : weights_) p.second = 1.0;
        }

        int best_rel = -1; double best_crit = -1.0;
        for (int k = 0; k < (int)N.size(); ++k) {
            if (rcN(k) >= -tol) continue;
            const int j = N[k];
            const double w = (weights_.count(j) ? weights_.at(j) : 1.0);
            const double crit = (rcN(k) * rcN(k)) / w;
            if (crit > best_crit) { best_crit = crit; best_rel = k; }
        }
        return (best_rel >= 0) ? std::optional<int>(best_rel) : std::nullopt;
    }

    void update_after_pivot(int leave_rel, int e_abs, int old_abs,
                            const Eigen::VectorXd &pivot_column, double alpha,
                            const Eigen::MatrixXd &/*A*/,
                            const std::vector<int> &N,
                            bool /*insert_leaver_into_pool*/ = true) {
        if (std::abs(alpha) < 1e-14) return;

        // Entering weight
        weights_[e_abs] = alpha * alpha;

        // Update others
        if (leave_rel < pivot_column.size()) {
            const double gamma_over_alpha = pivot_column(leave_rel) / alpha;
            for (int k = 0; k < (int)N.size(); ++k) {
                const int j = N[k];
                if (j == e_abs) continue;
                double &w = weights_[j];
                const bool had = (weights_.find(j) != weights_.end());
                if (!had) w = 1.0;
                const double nw = w + gamma_over_alpha * gamma_over_alpha;
                w = std::max(nw, threshold_ * w);
            }
        }

        // Ensure leaving has a slot
        (void)old_abs;
        if (!weights_.count(old_abs)) weights_[old_abs] = 1.0;
    }

    bool needs_rebuild() const { return false; }
    void clear_rebuild_flag()  {}

private:
    std::unordered_map<int,double> weights_;
    double threshold_{0.99};
    int    reset_freq_{1000};
    int    iter_count_{0};
};

// ============================================================================
// DualSteepestEdgePricer (simplified dual-direction maintenance)
// ============================================================================
class DualSteepestEdgePricer {
public:
    struct DualEntry {
        int jN;
        Eigen::VectorXd w;   // approx B^{-T} a_j
        double dual_weight;  // ||w||^2
    };

    explicit DualSteepestEdgePricer(int pool_max = 0, int reset_frequency = 1000)
        : pool_max_(pool_max), reset_freq_(reset_frequency) {}

    template <class BasisLike>
    void build_pool(const BasisLike &B,
                    const Eigen::MatrixXd &A,
                    const std::vector<int> &N) {
        dual_pool_.clear(); dual_pos_.clear();
        const int take = (pool_max_ > 0) ? std::min<int>(pool_max_, (int)N.size())
                                         : (int)N.size();
        dual_pool_.reserve(take);
        for (int k = 0; k < take; ++k) {
            const int j = N[k];
            DualEntry e;
            e.jN = j;
            const Eigen::VectorXd Aj = A.col(j);
            e.w = B.solve_BT(Aj);   // caller-provided
            e.dual_weight = e.w.squaredNorm();
            dual_pos_[j] = (int)dual_pool_.size();
            dual_pool_.push_back(std::move(e));
        }
        iter_count_ = 0;
        need_rebuild_ = false;
    }

    std::optional<int>
    choose_entering(const Eigen::VectorXd &rcN,
                    const std::vector<int> &N,
                    double tol) {
        ++iter_count_;
        int best_rel = -1; double best_score = -1.0;
        for (int k = 0; k < (int)N.size(); ++k) {
            if (rcN(k) >= -tol) continue;
            const int j = N[k];
            double dw = 1.0;
            if (auto it = dual_pos_.find(j); it != dual_pos_.end()) {
                dw = std::max(1.0, dual_pool_[it->second].dual_weight);
            }
            const double score = (rcN(k) * rcN(k)) / dw;
            if (score > best_score) { best_score = score; best_rel = k; }
        }
        return (best_rel >= 0) ? std::optional<int>(best_rel) : std::nullopt;
    }

    void update_after_pivot(int leave_rel, int e_abs, int old_abs,
                            const Eigen::VectorXd &s, double alpha,
                            const Eigen::MatrixXd &/*A*/,
                            const std::vector<int> &/*N*/,
                            bool insert_leaver_into_pool = true) {
        if (std::abs(alpha) < 1e-14) { need_rebuild_ = true; return; }

        const double inv_alpha = 1.0 / alpha;

        // Crude maintenance of dual directions (placeholder; robust impl needs dual pivot col)
        for (auto &E : dual_pool_) {
            if (leave_rel < E.w.size()) {
                const double wr = E.w(leave_rel);
                if (wr != 0.0) {
                    E.w(leave_rel) = -wr * inv_alpha;
                    E.dual_weight = E.w.squaredNorm();
                }
            }
        }

        // Remove entering
        if (auto itE = dual_pos_.find(e_abs); itE != dual_pos_.end()) {
            const int idx = itE->second, last = (int)dual_pool_.size() - 1;
            if (idx != last) {
                dual_pos_[dual_pool_[last].jN] = idx;
                std::swap(dual_pool_[idx], dual_pool_[last]);
            }
            dual_pool_.pop_back();
            dual_pos_.erase(itE);
        }

        // Add leaving
        if (insert_leaver_into_pool) {
            DualEntry E;
            E.jN = old_abs;
            E.w  = Eigen::VectorXd::Zero(s.size());
            if (leave_rel < E.w.size()) E.w(leave_rel) = 1.0;
            E.dual_weight = E.w.squaredNorm();

            if (pool_max_ > 0 && (int)dual_pool_.size() >= pool_max_) {
                int evict = 0; double wmax = dual_pool_[0].dual_weight;
                for (int i = 1; i < (int)dual_pool_.size(); ++i) {
                    if (dual_pool_[i].dual_weight > wmax) { wmax = dual_pool_[i].dual_weight; evict = i; }
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
        if (iter_count_ >= reset_freq_) need_rebuild_ = true;
    }

    bool needs_rebuild() const { return need_rebuild_; }
    void clear_rebuild_flag()  { need_rebuild_ = false; }

private:
    std::vector<DualEntry>           dual_pool_;
    std::unordered_map<int,int>      dual_pos_;
    int   pool_max_{0};
    int   reset_freq_{1000};
    int   iter_count_{0};
    bool  need_rebuild_{false};
};

// ============================================================================
// AdaptivePricer (strategy orchestration; API preserved)
// ============================================================================
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
        int     switch_threshold  = 100;
        int     performance_window = 50;
        double  improvement_factor = 1.2;
        int     partial_block_factor = 10;
        int     min_partial_block = 10;
        bool    enable_adaptive_switching = true;
        int     steepest_pool_max = 0;
        int     steepest_reset_freq = 1000;
        int     devex_reset_freq = 1000;
        int     dual_steepest_pool_max = 0;
        int     dual_steepest_reset_freq = 1000;
    };

    struct PricingStats {
        int total_pricing_calls{0};
        int strategy_switches{0};
        double avg_improvement_per_iteration{0.0};
        std::vector<int> strategy_usage_count{5, 0};
    };

    explicit AdaptivePricer(int n) : AdaptivePricer(n, PricingOptions{}) {}

    AdaptivePricer(int n, const PricingOptions &opts)
        : current_strategy_(opts.initial_strategy),
          options_(opts),
          n_(n),
          steepest_pricer_(opts.steepest_pool_max, opts.steepest_reset_freq),
          devex_pricer_(0.99, opts.devex_reset_freq),
          dual_steepest_pricer_(opts.dual_steepest_pool_max, opts.dual_steepest_reset_freq),
          iterations_since_switch_(0),
          last_objective_(0.0),
          first_call_(true) {
        stats_.strategy_usage_count.assign(5, 0);
    }

    // Main pricing entry
    template <typename BasisLike>
    std::optional<int>
    choose_entering(const Eigen::VectorXd &rN, const std::vector<int> &N,
                    double tol, int iteration, double current_objective,
                    const BasisLike &basis, const Eigen::MatrixXd &A) {
        ++stats_.total_pricing_calls;
        ++stats_.strategy_usage_count[current_strategy_];

        track_performance_(current_objective);

        if (options_.enable_adaptive_switching && should_switch_strategy_(iteration)) {
            adapt_strategy_();
            rebuild_pools_(basis, A, N);
        }

        switch (current_strategy_) {
            case STEEPEST_EDGE:  return steepest_pricer_.choose_entering(rN, N, tol);
            case DEVEX:          return devex_pricer_.choose_entering(rN, N, tol);
            case PARTIAL_PRICING:return partial_pricing_(rN, N, tol, iteration);
            case DUAL_STEEPEST:  return dual_steepest_pricer_.choose_entering(rN, N, tol);
            case MOST_NEGATIVE:  return most_negative_pricing_(rN, N, tol);
        }
        return std::nullopt;
    }

    // Build pools for all (cheaper than switching on demand, and preserves API)
    template <typename BasisLike>
    void build_pools(const BasisLike &basis, const Eigen::MatrixXd &A,
                     const std::vector<int> &N) {
        steepest_pricer_.build_pool(basis, A, N);
        devex_pricer_.build_pool(basis, A, N);
        dual_steepest_pricer_.build_pool(basis, A, N);
    }

    void update_after_pivot(int leaving_rel, int entering_abs, int old_abs,
                            const Eigen::VectorXd &pivot_column, double alpha,
                            double step_size, const Eigen::MatrixXd &A,
                            const std::vector<int> &N) {
        steepest_pricer_.update_after_pivot(leaving_rel, entering_abs, old_abs,
                                            pivot_column, alpha, A, N, true);
        devex_pricer_.update_after_pivot(leaving_rel, entering_abs, old_abs,
                                         pivot_column, alpha, A, N, true);
        dual_steepest_pricer_.update_after_pivot(leaving_rel, entering_abs,
                                                 old_abs, pivot_column, alpha,
                                                 A, N, true);

        if ((int)performance_history_.size() >= options_.performance_window)
            performance_history_.pop_front();
        performance_history_.push_back(step_size);
    }

    bool needs_rebuild() const {
        switch (current_strategy_) {
            case STEEPEST_EDGE: return steepest_pricer_.needs_rebuild();
            case DUAL_STEEPEST: return dual_steepest_pricer_.needs_rebuild();
            default:            return false;
        }
    }

    void clear_rebuild_flag() {
        steepest_pricer_.clear_rebuild_flag();
        dual_steepest_pricer_.clear_rebuild_flag();
    }

    const char *get_current_strategy_name() const {
        switch (current_strategy_) {
            case STEEPEST_EDGE:  return "steepest_edge";
            case DEVEX:          return "devex";
            case PARTIAL_PRICING:return "partial_pricing";
            case DUAL_STEEPEST:  return "dual_steepest";
            case MOST_NEGATIVE:  return "most_negative";
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
        stats_ = PricingStats{};
        stats_.strategy_usage_count.assign(5, 0);
    }

private:
    template <typename BasisLike>
    void rebuild_pools_(const BasisLike &basis, const Eigen::MatrixXd &A,
                        const std::vector<int> &N) {
        switch (current_strategy_) {
            case STEEPEST_EDGE:  steepest_pricer_.build_pool(basis, A, N); break;
            case DEVEX:          devex_pricer_.build_pool(basis, A, N); break;
            case DUAL_STEEPEST:  dual_steepest_pricer_.build_pool(basis, A, N); break;
            default: break;
        }
    }

    void track_performance_(double current_objective) {
        if (!first_call_) {
            double improvement = std::abs(current_objective - last_objective_);
            if ((int)recent_objectives_.size() >= 10) recent_objectives_.pop_front();
            recent_objectives_.push_back(improvement);
        }
        last_objective_ = current_objective;
        first_call_ = false;
        // Optional: maintain average
        double sum = std::accumulate(recent_objectives_.begin(),
                                     recent_objectives_.end(), 0.0);
        const int cnt = (int)recent_objectives_.size();
        stats_.avg_improvement_per_iteration = (cnt > 0) ? (sum / cnt) : 0.0;
    }

    void adapt_strategy_() {
        if ((int)recent_objectives_.size() < 20) return;

        ++stats_.strategy_switches;

        const double recent_avg =
            std::accumulate(recent_objectives_.end() - 10, recent_objectives_.end(), 0.0) / 10.0;
        const double older_avg  =
            std::accumulate(recent_objectives_.begin(), recent_objectives_.begin() + 10, 0.0) / 10.0;

        if (recent_avg < older_avg / options_.improvement_factor) {
            if (n_ > 10000) {
                current_strategy_ =
                    (current_strategy_ == PARTIAL_PRICING) ? DEVEX : PARTIAL_PRICING;
            } else if (!performance_history_.empty()) {
                const double avg_step =
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

    bool should_switch_strategy_(int /*iteration*/) {
        return (++iterations_since_switch_) >= options_.switch_threshold;
    }

    std::optional<int>
    partial_pricing_(const Eigen::VectorXd &rN,
                     const std::vector<int> &N,
                     double tol,
                     int iteration) {
        const int block_size = std::max(options_.min_partial_block,
                                        (int)N.size() / std::max(1, options_.partial_block_factor));
        const int start_idx  = (block_size > 0) ? ((iteration * block_size) % std::max(1, (int)N.size()))
                                                : 0;

        int best_idx = -1; double best_rc = 0.0;
        const int limit = std::min(block_size, (int)N.size());
        for (int k = 0; k < limit; ++k) {
            const int idx = (start_idx + k) % N.size();
            if (rN(idx) < -tol && rN(idx) < best_rc) { best_rc = rN(idx); best_idx = idx; }
        }
        return (best_idx >= 0) ? std::optional<int>(best_idx) : std::nullopt;
    }

    std::optional<int>
    most_negative_pricing_(const Eigen::VectorXd &rN,
                           const std::vector<int> &N,
                           double tol) {
        (void)N; // rN is already aligned with N
        int best_idx = -1; double best_rc = 0.0;
        for (int k = 0; k < rN.size(); ++k) {
            if (rN(k) < -tol && rN(k) < best_rc) { best_rc = rN(k); best_idx = k; }
        }
        return (best_idx >= 0) ? std::optional<int>(best_idx) : std::nullopt;
    }

private:
    Strategy       current_strategy_;
    PricingOptions options_;
    int            n_{0};
    mutable PricingStats stats_;

    // Sub-pricers
    SteepestEdgePricer      steepest_pricer_;
    DevexPricer             devex_pricer_;
    DualSteepestEdgePricer  dual_steepest_pricer_;

    // Switching/perf state
    int    iterations_since_switch_{0};
    double last_objective_{0.0};
    bool   first_call_{true};
    std::deque<double> performance_history_;
    std::deque<double> recent_objectives_;
};
