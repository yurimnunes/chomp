#pragma once
#include <optional>
#include <vector>
// Advanced Second-Order Correction (SOC) Class
// Designed to integrate with existing MehrotraGondzioSolver
// Advanced Second-Order Correction (SOC) Class
// Designed to integrate with existing MehrotraGondzioSolver
class AdvancedSOC {
public:
    enum class Strategy {
        Basic,                // Standard SOC
        FeasibilityRestoring, // Target constraint violations
        OptimalityImproving,  // Target complementarity errors
        Hybrid,               // Adaptive combination
        TrustRegion,          // Trust-region limited
        HigherOrder           // Third-order corrections
    };

    enum class TriggerMode {
        ThresholdBased,   // Simple alpha threshold
        ViolationBased,   // Constraint violation magnitude
        StagnationBased,  // Progress detection
        AdaptiveComposite // Multiple criteria
    };
    struct Config {
        double alpha_threshold = 0.3; // Less frequent SOC triggering
        int max_soc_iterations = 3;   // Fewer iterations to avoid overuse
        double min_improvement = 0.1; // Stricter improvement requirement

        TriggerMode trigger_mode = TriggerMode::AdaptiveComposite;
        double violation_tolerance = 1e-5;       // Relaxed tolerance
        double complementarity_tolerance = 1e-7; // Relaxed tolerance
        double stagnation_threshold = 0.9;
        int stagnation_lookback = 5;

        std::vector<Strategy> strategy_sequence = {
            Strategy::Basic, Strategy::FeasibilityRestoring,
            Strategy::Hybrid};  // Exclude HigherOrder, TrustRegion for now
        int max_candidates = 2; // Fewer candidates

        double trust_radius_initial = 2.0; // Larger initial radius
        double trust_radius_max = 20.0;
        double trust_shrink_factor = 0.7;
        double trust_expand_factor = 1.5;

        double acceptance_threshold = 0.2; // Stricter acceptance
        double merit_improvement_threshold = 0.05;
        bool use_filter = false; // Disable filter temporarily
        int filter_max_size = 10;

        double alpha_weight = 0.3;
        double complementarity_weight = 0.3;
        double feasibility_weight = 0.4; // Prioritize feasibility
    };

    struct Analysis {
        // Constraint violations
        dvec violation_magnitudes;
        std::vector<int> most_violated_indices;
        double max_violation = 0.0;
        double violation_norm = 0.0;

        // Complementarity errors
        dvec complementarity_errors;
        std::vector<int> worst_complementarity_indices;
        double max_comp_error = 0.0;
        double comp_error_norm = 0.0;
        double average_complementarity = 0.0;

        bool has_significant_violations = false;
        bool has_poor_complementarity = false;
    };

    struct Candidate {
        dvec dx, dnu, ds, dlam, dzL, dzU;
        double alpha_pri = 1.0, alpha_dual = 1.0;

        Strategy strategy_used = Strategy::Basic;
        double merit_value = 0.0;
        double alpha_improvement = 0.0;
        double violation_reduction = 0.0;
        double complementarity_improvement = 0.0;
        double trust_region_violation = 0.0;
        bool is_acceptable = false;

        // Convert to your StepData format
        template <typename StepData>
        void fill_step_data(StepData &step, double mu_target) const {
            step.dx = dx;
            step.dnu = dnu;
            step.ds = ds;
            step.dlam = dlam;
            step.dzL = dzL;
            step.dzU = dzU;
            step.alpha_pri = alpha_pri;
            step.alpha_dual = alpha_dual;
            step.mu_target = mu_target;
            step.soc_count = 1;
            step.use_correction = true;
        }
    };

    AdvancedSOC() : config_(), trust_radius_(config_.trust_radius_initial) {}

    explicit AdvancedSOC(const Config &config)
        : config_(config), trust_radius_(config.trust_radius_initial) {}

    // Main interface - integrate this into your solver
    template <typename StepData, typename Bounds, typename Sigmas>
    bool apply_soc(
        StepData &step, const spmat &W, const std::optional<spmat> &JE,
        const std::optional<spmat> &JI, const dvec &s, const dvec &lam,
        const dvec &zL, const dvec &zU, const Bounds &B, double alpha_aff,
        double mu_target, bool use_shifted, double tau_shift,
        double bound_shift, const Sigmas &Sg,
        std::function<KKTResult(const spmat &, const dvec &,
                                const std::optional<spmat> &,
                                const std::optional<dvec> &, std::string_view)>
            solve_kkt) {

        // Check if SOC should be triggered
        if (!should_trigger_soc(step, alpha_aff, s, lam, zL, zU, B,
                                mu_target)) {
            return false;
        }

        // Analyze current state
        Analysis analysis =
            analyze_current_state(step, s, lam, zL, zU, B, alpha_aff, mu_target,
                                  use_shifted, tau_shift, bound_shift);

        // Generate SOC candidates
        std::vector<Candidate> candidates = generate_candidates(
            W, JE, JI, s, lam, zL, zU, step, B, alpha_aff, mu_target,
            use_shifted, tau_shift, bound_shift, Sg, analysis, solve_kkt);

        // Select best candidate
        auto best = select_best_candidate(candidates, step);

        if (best.is_acceptable) {
            best.fill_step_data(step, mu_target);
            update_state(best, analysis);
            return true;
        }

        return false;
    }

    // Getters for diagnostics
    double get_trust_radius() const { return trust_radius_; }
    int get_stagnation_count() const { return stagnation_counter_; }
    const std::vector<double> &get_progress_history() const {
        return progress_history_;
    }

    template <typename StepData, typename Bounds>
    Analysis
    analyze_current_state(const StepData &step, const dvec &s, const dvec &lam,
                          const dvec &zL, const dvec &zU, const Bounds &B,
                          double alpha_aff, double mu_target, bool use_shifted,
                          double tau_shift, double bound_shift) const {

        Analysis analysis;
        const int mI = s.size();
        const int n = zL.size();

        // Analyze constraint violations
        if (mI > 0) {
            analysis.violation_magnitudes.resize(mI);

            for (int i = 0; i < mI; ++i) {
                double s_pred = s[i] + alpha_aff * step.ds[i];
                if (use_shifted)
                    s_pred += tau_shift;

                double violation = std::min(s_pred, 0.0);
                analysis.violation_magnitudes[i] = std::abs(violation);
                analysis.max_violation =
                    std::max(analysis.max_violation, std::abs(violation));
            }

            analysis.violation_norm = analysis.violation_magnitudes.norm();
            analysis.has_significant_violations =
                analysis.violation_norm > config_.violation_tolerance;

            // Find most violated constraints
            std::vector<std::pair<double, int>> violations_indexed;
            for (int i = 0; i < mI; ++i) {
                violations_indexed.push_back(
                    {analysis.violation_magnitudes[i], i});
            }
            std::sort(violations_indexed.rbegin(), violations_indexed.rend());

            for (int i = 0; i < std::min(5, mI); ++i) {
                if (violations_indexed[i].first > config_.violation_tolerance) {
                    analysis.most_violated_indices.push_back(
                        violations_indexed[i].second);
                }
            }
        }

        // Analyze complementarity errors
        int total_pairs = mI;
        for (int i = 0; i < n; ++i) {
            if (B.hasL[i])
                total_pairs++;
            if (B.hasU[i])
                total_pairs++;
        }

        analysis.complementarity_errors.resize(total_pairs);
        int idx = 0;
        double sum_comp = 0.0;

        // Inequality complementarity
        for (int i = 0; i < mI; ++i) {
            double s_pred = s[i] + alpha_aff * step.ds[i];
            double l_pred = lam[i] + alpha_aff * step.dlam[i];
            if (use_shifted)
                s_pred += tau_shift;

            double comp_product = s_pred * l_pred;
            double error = std::abs(comp_product - mu_target);
            analysis.complementarity_errors[idx] = error;
            analysis.max_comp_error = std::max(analysis.max_comp_error, error);
            sum_comp += comp_product;
            idx++;
        }

        // Bound complementarity
        for (int i = 0; i < n; ++i) {
            if (B.hasL[i]) {
                double sL_pred = B.sL[i] + alpha_aff * step.dx[i];
                double zL_pred = zL[i] + alpha_aff * step.dzL[i];
                if (use_shifted)
                    sL_pred += bound_shift;

                double comp_product = sL_pred * zL_pred;
                double error = std::abs(comp_product - mu_target);
                analysis.complementarity_errors[idx] = error;
                analysis.max_comp_error =
                    std::max(analysis.max_comp_error, error);
                sum_comp += comp_product;
                idx++;
            }

            if (B.hasU[i]) {
                double sU_pred = B.sU[i] - alpha_aff * step.dx[i];
                double zU_pred = zU[i] + alpha_aff * step.dzU[i];
                if (use_shifted)
                    sU_pred += bound_shift;

                double comp_product = sU_pred * zU_pred;
                double error = std::abs(comp_product - mu_target);
                analysis.complementarity_errors[idx] = error;
                analysis.max_comp_error =
                    std::max(analysis.max_comp_error, error);
                sum_comp += comp_product;
                idx++;
            }
        }

        analysis.comp_error_norm = analysis.complementarity_errors.norm();
        analysis.average_complementarity = sum_comp / std::max(1, total_pairs);
        analysis.has_poor_complementarity =
            analysis.comp_error_norm > config_.complementarity_tolerance;

        // Find worst complementarity pairs
        std::vector<std::pair<double, int>> errors_indexed;
        for (int i = 0; i < idx; ++i) {
            errors_indexed.push_back({analysis.complementarity_errors[i], i});
        }
        std::sort(errors_indexed.rbegin(), errors_indexed.rend());

        for (int i = 0; i < std::min(5, idx); ++i) {
            if (errors_indexed[i].first > config_.complementarity_tolerance) {
                analysis.worst_complementarity_indices.push_back(
                    errors_indexed[i].second);
            }
        }

        return analysis;
    }

private:
    Config config_;

    // State tracking
    mutable std::vector<double> progress_history_;
    mutable int stagnation_counter_ = 0;
    mutable double trust_radius_;
    mutable std::vector<std::pair<double, double>>
        filter_; // (violation, objective) pairs

    template <typename StepData, typename Bounds>
    bool should_trigger_soc(const StepData &step, double alpha_aff,
                            const dvec &s, const dvec &lam, const dvec &zL,
                            const dvec &zU, const Bounds &B,
                            double mu_target) const {

        switch (config_.trigger_mode) {
        case TriggerMode::ThresholdBased:
            return alpha_aff < config_.alpha_threshold;

        case TriggerMode::ViolationBased: {
            double violation_norm = compute_violation_norm(s, B);
            return violation_norm > config_.violation_tolerance;
        }

        case TriggerMode::StagnationBased: {
            if (progress_history_.size() >= config_.stagnation_lookback) {
                double current_progress =
                    std::min(step.alpha_pri, step.alpha_dual);
                double avg_recent = 0.0;
                for (int i = 0; i < config_.stagnation_lookback; ++i) {
                    avg_recent +=
                        progress_history_[progress_history_.size() - 1 - i];
                }
                avg_recent /= config_.stagnation_lookback;
                return current_progress <
                       config_.stagnation_threshold * avg_recent;
            }
            return alpha_aff < config_.alpha_threshold;
        }

        case TriggerMode::AdaptiveComposite: {
            bool threshold_trigger = alpha_aff < config_.alpha_threshold;
            double violation_norm = compute_violation_norm(s, B);
            bool violation_trigger =
                violation_norm > config_.violation_tolerance;
            double comp_error =
                compute_complementarity_error(s, lam, zL, zU, B, mu_target);
            bool comp_trigger = comp_error > config_.complementarity_tolerance;

            return threshold_trigger || violation_trigger || comp_trigger;
        }
        }

        return alpha_aff < config_.alpha_threshold;
    }

    template <typename StepData, typename Bounds, typename Sigmas>
    std::vector<Candidate> generate_candidates(
        const spmat &W, const std::optional<spmat> &JE,
        const std::optional<spmat> &JI, const dvec &s, const dvec &lam,
        const dvec &zL, const dvec &zU, const StepData &base_step,
        const Bounds &B, double alpha_aff, double mu_target, bool use_shifted,
        double tau_shift, double bound_shift, const Sigmas &Sg,
        const Analysis &analysis,
        std::function<KKTResult(const spmat &, const dvec &,
                                const std::optional<spmat> &,
                                const std::optional<dvec> &, std::string_view)>
            solve_kkt) const {

        std::vector<Candidate> candidates;

        for (Strategy strategy : config_.strategy_sequence) {
            if (candidates.size() >= config_.max_candidates)
                break;

            Candidate candidate;
            candidate.strategy_used = strategy;

            dvec rhs_x = compute_soc_rhs(
                strategy, s, lam, zL, zU, base_step, B, alpha_aff, mu_target,
                use_shifted, tau_shift, bound_shift, Sg, analysis, JI);

            auto soc_res = solve_kkt(W, rhs_x, JE, std::nullopt, "hykkt");

            // Handle trust region constraint
            if (strategy == Strategy::TrustRegion) {
                double step_norm = soc_res.dx.norm();
                if (step_norm > trust_radius_) {
                    double scale = trust_radius_ / step_norm;
                    soc_res.dx *= scale;
                    if (soc_res.dy.size() > 0)
                        soc_res.dy *= scale;
                    candidate.trust_region_violation =
                        step_norm - trust_radius_;
                }
            }

            // Fill candidate data
            candidate.dx = soc_res.dx;
            candidate.dnu = (JE && soc_res.dy.size() > 0)
                                ? soc_res.dy
                                : dvec::Zero(JE ? JE->rows() : 0);

            // Compute remaining components
            complete_candidate(candidate, JI, s, lam, zL, zU, B, mu_target,
                               use_shifted, tau_shift, bound_shift);

            // Evaluate candidate
            evaluate_candidate(candidate, base_step, s, lam, zL, zU, B,
                               mu_target);

            candidates.push_back(candidate);
        }

        return candidates;
    }

    template <typename StepData, typename Bounds, typename Sigmas>
    dvec compute_soc_rhs(Strategy strategy, const dvec &s, const dvec &lam,
                         const dvec &zL, const dvec &zU,
                         const StepData &base_step, const Bounds &B,
                         double alpha_aff, double mu_target, bool use_shifted,
                         double tau_shift, double bound_shift, const Sigmas &Sg,
                         const Analysis &analysis,
                         const std::optional<spmat> &JI) const {

        const int n = base_step.dx.size();
        const int mI = s.size();
        dvec rhs_x = dvec::Zero(n);

        switch (strategy) {
        case Strategy::Basic:
            return compute_basic_soc_rhs(s, lam, zL, zU, base_step, B,
                                         alpha_aff, mu_target, use_shifted,
                                         tau_shift, bound_shift, Sg, JI);

        case Strategy::FeasibilityRestoring:
            return compute_feasibility_rhs(
                s, lam, zL, zU, base_step, B, alpha_aff, mu_target, use_shifted,
                tau_shift, bound_shift, Sg, analysis, JI);

        case Strategy::OptimalityImproving:
            return compute_optimality_rhs(s, lam, zL, zU, base_step, B,
                                          alpha_aff, mu_target, use_shifted,
                                          tau_shift, bound_shift, analysis);

        case Strategy::Hybrid:
            return compute_hybrid_rhs(s, lam, zL, zU, base_step, B, alpha_aff,
                                      mu_target, use_shifted, tau_shift,
                                      bound_shift, Sg, analysis, JI);

        case Strategy::TrustRegion:
            return compute_basic_soc_rhs(s, lam, zL, zU, base_step, B,
                                         alpha_aff, mu_target, use_shifted,
                                         tau_shift, bound_shift, Sg, JI);

        case Strategy::HigherOrder:
            return compute_higher_order_rhs(s, lam, zL, zU, base_step, B,
                                            alpha_aff, mu_target, use_shifted,
                                            tau_shift, bound_shift, Sg, JI);
        }

        return rhs_x;
    }

    // Specific RHS computation methods
    template <typename StepData, typename Bounds, typename Sigmas>
    dvec compute_basic_soc_rhs(const dvec &s, const dvec &lam, const dvec &zL,
                               const dvec &zU, const StepData &base_step,
                               const Bounds &B, double alpha_aff,
                               double mu_target, bool use_shifted,
                               double tau_shift, double bound_shift,
                               const Sigmas &Sg,
                               const std::optional<spmat> &JI) const {

        const int n = base_step.dx.size();
        const int mI = s.size();
        dvec rhs_x = dvec::Zero(n);

        // Standard SOC for inequalities
        if (mI > 0 && JI) {
            for (int i = 0; i < mI; ++i) {
                const double s_aff =
                    std::max(s[i] + alpha_aff * base_step.ds[i], 1e-12);
                const double lam_aff =
                    std::max(lam[i] + alpha_aff * base_step.dlam[i], 1e-12);
                const double s_eff = use_shifted ? (s_aff + tau_shift) : s_aff;

                double soc_correction = mu_target - s_eff * lam_aff;

                if (JI->nonZeros() && i < Sg.Sigma_s.size()) {
                    dvec contrib = (*JI).row(i).transpose() *
                                   (Sg.Sigma_s[i] * soc_correction);
                    rhs_x += contrib;
                }
            }
        }

        // Bound complementarity corrections
        for (int i = 0; i < n; ++i) {
            double bound_corr = 0.0;

            if (B.hasL[i]) {
                const double sL_aff =
                    std::max(B.sL[i] + alpha_aff * base_step.dx[i], 1e-12);
                const double zL_aff =
                    std::max(zL[i] + alpha_aff * base_step.dzL[i], 1e-12);
                const double s_eff =
                    use_shifted ? (sL_aff + bound_shift) : sL_aff;
                bound_corr += (mu_target - s_eff * zL_aff) / s_eff;
            }

            if (B.hasU[i]) {
                const double sU_aff =
                    std::max(B.sU[i] - alpha_aff * base_step.dx[i], 1e-12);
                const double zU_aff =
                    std::max(zU[i] + alpha_aff * base_step.dzU[i], 1e-12);
                const double s_eff =
                    use_shifted ? (sU_aff + bound_shift) : sU_aff;
                bound_corr -= (mu_target - s_eff * zU_aff) / s_eff;
            }

            rhs_x[i] += bound_corr;
        }

        return rhs_x;
    }

    template <typename StepData, typename Bounds, typename Sigmas>
    dvec compute_feasibility_rhs(const dvec &s, const dvec &lam, const dvec &zL,
                                 const dvec &zU, const StepData &base_step,
                                 const Bounds &B, double alpha_aff,
                                 double mu_target, bool use_shifted,
                                 double tau_shift, double bound_shift,
                                 const Sigmas &Sg, const Analysis &analysis,
                                 const std::optional<spmat> &JI) const {

        const int n = base_step.dx.size();
        dvec rhs_x = dvec::Zero(n);

        // Focus on violated constraints with aggressive correction
        if (analysis.has_significant_violations && JI) {
            for (int idx : analysis.most_violated_indices) {
                double s_pred = s[idx] + alpha_aff * base_step.ds[idx];
                if (use_shifted)
                    s_pred += tau_shift;

                if (s_pred < 0) {
                    double correction_factor = 2.0; // More aggressive
                    double violation_correction = -correction_factor * s_pred;

                    if (JI->nonZeros() && idx < Sg.Sigma_s.size()) {
                        dvec contrib = (*JI).row(idx).transpose() *
                                       (Sg.Sigma_s[idx] * violation_correction);
                        rhs_x += contrib;
                    }
                }
            }
        }

        // Minimal complementarity correction (focus on feasibility)
        double comp_weight = 0.1;
        for (int i = 0; i < n; ++i) {
            double bound_corr = 0.0;

            if (B.hasL[i]) {
                const double sL_pred =
                    std::max(B.sL[i] + alpha_aff * base_step.dx[i], 1e-12);
                const double zL_pred =
                    std::max(zL[i] + alpha_aff * base_step.dzL[i], 1e-12);
                const double s_eff =
                    use_shifted ? (sL_pred + bound_shift) : sL_pred;
                bound_corr +=
                    comp_weight * (mu_target - s_eff * zL_pred) / s_eff;
            }

            if (B.hasU[i]) {
                const double sU_pred =
                    std::max(B.sU[i] - alpha_aff * base_step.dx[i], 1e-12);
                const double zU_pred =
                    std::max(zU[i] + alpha_aff * base_step.dzU[i], 1e-12);
                const double s_eff =
                    use_shifted ? (sU_pred + bound_shift) : sU_pred;
                bound_corr -=
                    comp_weight * (mu_target - s_eff * zU_pred) / s_eff;
            }

            rhs_x[i] += bound_corr;
        }

        return rhs_x;
    }

    template <typename StepData, typename Bounds>
    dvec compute_optimality_rhs(const dvec &s, const dvec &lam, const dvec &zL,
                                const dvec &zU, const StepData &base_step,
                                const Bounds &B, double alpha_aff,
                                double mu_target, bool use_shifted,
                                double tau_shift, double bound_shift,
                                const Analysis &analysis) const {

        const int n = base_step.dx.size();
        dvec rhs_x = dvec::Zero(n);

        // Enhanced complementarity correction
        double comp_weight = 1.5;

        // Priority bounds from worst complementarity pairs
        std::set<int> priority_bounds;
        for (int err_idx : analysis.worst_complementarity_indices) {
            if (err_idx >= s.size()) {
                priority_bounds.insert(err_idx - s.size());
            }
        }

        for (int i = 0; i < n; ++i) {
            double bound_corr = 0.0;
            double weight =
                priority_bounds.count(i) > 0 ? comp_weight * 2.0 : comp_weight;

            if (B.hasL[i]) {
                const double sL_pred =
                    std::max(B.sL[i] + alpha_aff * base_step.dx[i], 1e-12);
                const double zL_pred =
                    std::max(zL[i] + alpha_aff * base_step.dzL[i], 1e-12);
                const double s_eff =
                    use_shifted ? (sL_pred + bound_shift) : sL_pred;

                double comp_error = s_eff * zL_pred - mu_target;
                bound_corr += -weight * comp_error / s_eff;
            }

            if (B.hasU[i]) {
                const double sU_pred =
                    std::max(B.sU[i] - alpha_aff * base_step.dx[i], 1e-12);
                const double zU_pred =
                    std::max(zU[i] + alpha_aff * base_step.dzU[i], 1e-12);
                const double s_eff =
                    use_shifted ? (sU_pred + bound_shift) : sU_pred;

                double comp_error = s_eff * zU_pred - mu_target;
                bound_corr += weight * comp_error / s_eff;
            }

            rhs_x[i] += bound_corr;
        }

        return rhs_x;
    }

    template <typename StepData, typename Bounds, typename Sigmas>
    dvec compute_hybrid_rhs(const dvec &s, const dvec &lam, const dvec &zL,
                            const dvec &zU, const StepData &base_step,
                            const Bounds &B, double alpha_aff, double mu_target,
                            bool use_shifted, double tau_shift,
                            double bound_shift, const Sigmas &Sg,
                            const Analysis &analysis,
                            const std::optional<spmat> &JI) const {

        // Adaptive weighting based on problem state
        double violation_severity = analysis.violation_norm;
        double comp_severity = analysis.comp_error_norm;
        double total_severity = violation_severity + comp_severity;

        double feasibility_weight =
            (total_severity > 0) ? violation_severity / total_severity : 0.5;
        double optimality_weight = 1.0 - feasibility_weight;

        auto feas_rhs = compute_feasibility_rhs(
            s, lam, zL, zU, base_step, B, alpha_aff, mu_target, use_shifted,
            tau_shift, bound_shift, Sg, analysis, JI);

        auto opt_rhs = compute_optimality_rhs(s, lam, zL, zU, base_step, B,
                                              alpha_aff, mu_target, use_shifted,
                                              tau_shift, bound_shift, analysis);

        return feasibility_weight * feas_rhs + optimality_weight * opt_rhs;
    }

    template <typename StepData, typename Bounds, typename Sigmas>
    dvec compute_higher_order_rhs(const dvec &s, const dvec &lam,
                                  const dvec &zL, const dvec &zU,
                                  const StepData &base_step, const Bounds &B,
                                  double alpha_aff, double mu_target,
                                  bool use_shifted, double tau_shift,
                                  double bound_shift, const Sigmas &Sg,
                                  const std::optional<spmat> &JI) const {

        auto basic_rhs = compute_basic_soc_rhs(
            s, lam, zL, zU, base_step, B, alpha_aff, mu_target, use_shifted,
            tau_shift, bound_shift, Sg, JI);

        // Add third-order terms
        const int n = base_step.dx.size();
        const int mI = s.size();
        dvec third_order_rhs = dvec::Zero(n);

        // Third-order complementarity terms for inequalities
        if (mI > 0 && JI) {
            for (int i = 0; i < mI; ++i) {
                // Third-order: ds_aff * dlam_aff * ds_soc (would need SOC step,
                // so approximate)
                double third_order_term =
                    base_step.ds[i] * base_step.dlam[i] * base_step.ds[i] * 0.1;

                if (JI->nonZeros() && i < Sg.Sigma_s.size()) {
                    dvec contrib = (*JI).row(i).transpose() *
                                   (Sg.Sigma_s[i] * third_order_term);
                    third_order_rhs += contrib;
                }
            }
        }

        // Third-order bound terms
        for (int i = 0; i < n; ++i) {
            double bound_third = 0.0;

            if (B.hasL[i]) {
                double third_L =
                    base_step.dx[i] * base_step.dzL[i] * base_step.dx[i] * 0.1;
                const double sL_eff = std::max(
                    use_shifted ? (B.sL[i] + bound_shift) : B.sL[i], 1e-12);
                bound_third += third_L / sL_eff;
            }

            if (B.hasU[i]) {
                double third_U =
                    -base_step.dx[i] * base_step.dzU[i] * base_step.dx[i] * 0.1;
                const double sU_eff = std::max(
                    use_shifted ? (B.sU[i] + bound_shift) : B.sU[i], 1e-12);
                bound_third += third_U / sU_eff;
            }

            third_order_rhs[i] += bound_third;
        }

        return basic_rhs +
               0.5 * third_order_rhs; // Conservative third-order weight
    }

    template <typename Bounds>
    void complete_candidate(Candidate &candidate,
                            const std::optional<spmat> &JI, const dvec &s,
                            const dvec &lam, const dvec &zL, const dvec &zU,
                            const Bounds &B, double mu_target, bool use_shifted,
                            double tau_shift, double bound_shift) const {

        const int mI = s.size();
        const int n = candidate.dx.size();

        // Compute slack steps
        if (mI > 0 && JI) {
            dvec r_pI_zero = dvec::Zero(mI);
            candidate.ds = -(r_pI_zero + (*JI) * candidate.dx);

            candidate.dlam.resize(mI);
            for (int i = 0; i < mI; ++i) {
                const double d =
                    std::max(use_shifted ? (s[i] + tau_shift) : s[i], 1e-12);
                candidate.dlam[i] =
                    (mu_target - d * lam[i] - lam[i] * candidate.ds[i]) / d;
            }
        } else {
            candidate.ds.resize(0);
            candidate.dlam.resize(0);
        }

        // Compute bound dual steps
        candidate.dzL = dvec::Zero(n);
        candidate.dzU = dvec::Zero(n);

        for (int i = 0; i < n; ++i) {
            if (B.hasL[i]) {
                const double d = std::max(
                    use_shifted ? (B.sL[i] + bound_shift) : B.sL[i], 1e-12);
                candidate.dzL[i] =
                    (mu_target - d * zL[i] - zL[i] * candidate.dx[i]) / d;
            }
            if (B.hasU[i]) {
                const double d = std::max(
                    use_shifted ? (B.sU[i] + bound_shift) : B.sU[i], 1e-12);
                candidate.dzU[i] =
                    (mu_target - d * zU[i] + zU[i] * candidate.dx[i]) / d;
            }
        }

        // Compute step lengths
        double tau_pri = 0.995, tau_dual = 0.995;
        candidate.alpha_pri = tau_pri;
        candidate.alpha_dual = tau_dual;

        // Primal step length
        for (int i = 0; i < mI; ++i) {
            if (candidate.ds[i] < -1e-16) {
                candidate.alpha_pri =
                    std::min(candidate.alpha_pri, -s[i] / candidate.ds[i]);
            }
        }

        for (int i = 0; i < n; ++i) {
            if (B.hasL[i] && candidate.dx[i] < -1e-16) {
                candidate.alpha_pri =
                    std::min(candidate.alpha_pri, -B.sL[i] / candidate.dx[i]);
            }
            if (B.hasU[i] && candidate.dx[i] > 1e-16) {
                candidate.alpha_pri =
                    std::min(candidate.alpha_pri, B.sU[i] / candidate.dx[i]);
            }
        }

        // Dual step length
        for (int i = 0; i < mI; ++i) {
            if (candidate.dlam[i] < -1e-16) {
                candidate.alpha_dual =
                    std::min(candidate.alpha_dual, -lam[i] / candidate.dlam[i]);
            }
        }

        for (int i = 0; i < n; ++i) {
            if (B.hasL[i] && candidate.dzL[i] < -1e-16) {
                candidate.alpha_dual =
                    std::min(candidate.alpha_dual, -zL[i] / candidate.dzL[i]);
            }
            if (B.hasU[i] && candidate.dzU[i] < -1e-16) {
                candidate.alpha_dual =
                    std::min(candidate.alpha_dual, -zU[i] / candidate.dzU[i]);
            }
        }

        candidate.alpha_pri =
            std::max(0.0, std::min(1.0, tau_pri * candidate.alpha_pri));
        candidate.alpha_dual =
            std::max(0.0, std::min(1.0, tau_dual * candidate.alpha_dual));
    }

    template <typename StepData, typename Bounds>
    void evaluate_candidate(Candidate &candidate, const StepData &base_step,
                            const dvec &s, const dvec &lam, const dvec &zL,
                            const dvec &zU, const Bounds &B,
                            double mu_target) const {

        // Step length improvement
        double base_alpha = std::min(base_step.alpha_pri, base_step.alpha_dual);
        double candidate_alpha =
            std::min(candidate.alpha_pri, candidate.alpha_dual);
        candidate.alpha_improvement = candidate_alpha - base_alpha;

        // Enhanced merit function
        candidate.merit_value =
            compute_merit_function(candidate, s, lam, zL, zU, B, mu_target);

        // Violation and complementarity measures (simplified)
        candidate.violation_reduction =
            std::max(0.0, candidate.alpha_improvement * 0.1);
        candidate.complementarity_improvement =
            std::max(0.0, candidate.alpha_improvement * 0.1);

        // Acceptance criteria
        candidate.is_acceptable = evaluate_acceptance(candidate);
    }

    template <typename Bounds>
    double compute_merit_function(const Candidate &candidate, const dvec &s,
                                  const dvec &lam, const dvec &zL,
                                  const dvec &zU, const Bounds &B,
                                  double mu_target) const {

        // Step length component
        double alpha_merit =
            1.0 - std::min(candidate.alpha_pri, candidate.alpha_dual);

        // Complementarity component
        double comp_merit = 0.0;
        int comp_count = 0;

        // Inequality complementarity
        for (int i = 0; i < s.size(); ++i) {
            double s_new = s[i] + candidate.alpha_pri * candidate.ds[i];
            double l_new = lam[i] + candidate.alpha_dual * candidate.dlam[i];
            comp_merit += std::abs(s_new * l_new - mu_target);
            comp_count++;
        }

        // Bound complementarity
        const int n = candidate.dx.size();
        for (int i = 0; i < n; ++i) {
            if (B.hasL[i]) {
                double sL_new = B.sL[i] + candidate.alpha_pri * candidate.dx[i];
                double zL_new = zL[i] + candidate.alpha_dual * candidate.dzL[i];
                comp_merit += std::abs(sL_new * zL_new - mu_target);
                comp_count++;
            }
            if (B.hasU[i]) {
                double sU_new = B.sU[i] - candidate.alpha_pri * candidate.dx[i];
                double zU_new = zU[i] + candidate.alpha_dual * candidate.dzU[i];
                comp_merit += std::abs(sU_new * zU_new - mu_target);
                comp_count++;
            }
        }

        comp_merit /= std::max(1, comp_count);

        // Feasibility component
        double feas_merit = 0.0;
        for (int i = 0; i < s.size(); ++i) {
            double s_new = s[i] + candidate.alpha_pri * candidate.ds[i];
            feas_merit += std::max(0.0, -s_new);
        }

        // Combined merit
        return config_.alpha_weight * alpha_merit +
               config_.complementarity_weight * comp_merit /
                   std::max(mu_target, 1e-12) +
               config_.feasibility_weight * feas_merit;
    }

    bool evaluate_acceptance(const Candidate &candidate) const {
        bool alpha_improvement =
            candidate.alpha_improvement > config_.acceptance_threshold;
        bool merit_improvement = candidate.merit_value < 1.0;

        switch (candidate.strategy_used) {
        case Strategy::FeasibilityRestoring:
            return alpha_improvement || candidate.violation_reduction >
                                            config_.merit_improvement_threshold;

        case Strategy::OptimalityImproving:
            return alpha_improvement || candidate.complementarity_improvement >
                                            config_.merit_improvement_threshold;

        case Strategy::TrustRegion:
            return (alpha_improvement || merit_improvement) &&
                   candidate.trust_region_violation < 1e-6;

        case Strategy::Hybrid:
            return alpha_improvement ||
                   (candidate.violation_reduction > 0 &&
                    candidate.complementarity_improvement > 0);

        case Strategy::HigherOrder:
            return alpha_improvement && merit_improvement;

        default:
            return alpha_improvement;
        }
    }

    Candidate select_best_candidate(const std::vector<Candidate> &candidates,
                                    const auto &base_step) const {

        Candidate best;
        best.is_acceptable = false;

        double best_score = std::numeric_limits<double>::infinity();

        for (const auto &candidate : candidates) {
            if (!candidate.is_acceptable)
                continue;

            double score = candidate.merit_value;

            // Bonuses and penalties
            if (candidate.alpha_improvement > 0) {
                score -= 0.5 * candidate.alpha_improvement;
            }
            if (candidate.violation_reduction > 0) {
                score -= 0.3 * candidate.violation_reduction;
            }
            if (candidate.trust_region_violation > 0) {
                score += 2.0 * candidate.trust_region_violation;
            }

            if (score < best_score) {
                best_score = score;
                best = candidate;
            }
        }

        // Filter check
        if (config_.use_filter && best.is_acceptable) {
            best.is_acceptable = passes_filter(best);
        }

        return best;
    }

    bool passes_filter(const Candidate &candidate) const {
        double violation_measure = candidate.violation_reduction; // Simplified
        double objective_measure = candidate.merit_value;

        for (const auto &filter_point : filter_) {
            if (violation_measure >= filter_point.first &&
                objective_measure >= filter_point.second) {
                return false;
            }
        }
        return true;
    }

    void update_state(const Candidate &applied_candidate,
                      const Analysis &analysis) const {
        // Update progress history
        double current_progress =
            std::min(applied_candidate.alpha_pri, applied_candidate.alpha_dual);
        progress_history_.push_back(current_progress);
        if (progress_history_.size() > 10) {
            progress_history_.erase(progress_history_.begin());
        }

        // Update stagnation counter
        if (progress_history_.size() >= 2) {
            double prev_progress =
                progress_history_[progress_history_.size() - 2];
            if (current_progress <
                config_.stagnation_threshold * prev_progress) {
                stagnation_counter_++;
            } else {
                stagnation_counter_ = 0;
            }
        }

        // Update trust region
        if (applied_candidate.strategy_used == Strategy::TrustRegion) {
            if (applied_candidate.alpha_improvement >
                config_.acceptance_threshold) {
                trust_radius_ =
                    std::min(trust_radius_ * config_.trust_expand_factor,
                             config_.trust_radius_max);
            } else if (applied_candidate.alpha_improvement < 0) {
                trust_radius_ *= config_.trust_shrink_factor;
            }
        }

        // Update filter
        if (config_.use_filter) {
            filter_.push_back({applied_candidate.violation_reduction,
                               applied_candidate.merit_value});

            // Remove dominated points and limit size
            filter_.erase(
                std::remove_if(filter_.begin(), filter_.end(),
                               [this](const auto &point) {
                                   return std::any_of(
                                       filter_.begin(), filter_.end(),
                                       [&point](const auto &other) {
                                           return &other != &point &&
                                                  other.first <= point.first &&
                                                  other.second <= point.second;
                                       });
                               }),
                filter_.end());

            if (filter_.size() > config_.filter_max_size) {
                filter_.erase(filter_.begin());
            }
        }
    }

    // Helper functions
    template <typename Bounds>
    double compute_violation_norm(const dvec &s, const Bounds &B) const {
        double norm = 0.0;

        for (int i = 0; i < s.size(); ++i) {
            norm += std::max(0.0, -s[i]) * std::max(0.0, -s[i]);
        }

        const int n = B.sL.size();
        for (int i = 0; i < n; ++i) {
            if (B.hasL[i]) {
                norm += std::max(0.0, -B.sL[i]) * std::max(0.0, -B.sL[i]);
            }
            if (B.hasU[i]) {
                norm += std::max(0.0, -B.sU[i]) * std::max(0.0, -B.sU[i]);
            }
        }

        return std::sqrt(norm);
    }

    template <typename Bounds>
    double compute_complementarity_error(const dvec &s, const dvec &lam,
                                         const dvec &zL, const dvec &zU,
                                         const Bounds &B,
                                         double mu_target) const {
        double error = 0.0;
        int count = 0;

        for (int i = 0; i < s.size(); ++i) {
            error += std::abs(s[i] * lam[i] - mu_target);
            count++;
        }

        const int n = zL.size();
        for (int i = 0; i < n; ++i) {
            if (B.hasL[i]) {
                error += std::abs(B.sL[i] * zL[i] - mu_target);
                count++;
            }
            if (B.hasU[i]) {
                error += std::abs(B.sU[i] * zU[i] - mu_target);
                count++;
            }
        }

        return count > 0 ? error / count : 0.0;
    }
};