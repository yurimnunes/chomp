#pragma once

// -----------------------------------------------------------------------------
// Revised Simplex (header-only, drop-in compatible) — now with Dual Simplex
// Public API preserved: LPSolution, to_string, RevisedSimplexOptions,
//                       RevisedSimplex{ ctor, solve(...) }.
// Internals tidied without behavioral changes, plus a dual simplex phase:
//   - Options::mode = {Auto, Primal, Dual}
//   - Auto tries primal, and if primal reports negative basic variables,
//     falls back to dual before Phase I.
//   - You can force Dual by setting options.mode = SimplexMode::Dual.
// -----------------------------------------------------------------------------

#include "presolver.h"   // presolve::LP, Presolver
#include "simplex_aux.h" // FTBasis helpers declared by your project
#include "simplex_lu.h" // FTBasis implementation (solve_B, solve_BT, replace_column, refactor)

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Sparse>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

// ============================================================================
// Public result container
// ============================================================================
struct LPSolution {
    enum class Status {
        Optimal,
        Unbounded,
        Infeasible,
        IterLimit,
        Singular,
        NeedPhase1
    };

    Status status{};
    Eigen::VectorXd x; // primal solution (original space)
    double obj = std::numeric_limits<double>::quiet_NaN();
    std::vector<int> basis; // basis indices in original problem
    int iters = 0;          // total iterations (Phase I + II)
    std::unordered_map<std::string, std::string> info; // telemetry
    Eigen::VectorXd farkas_y; // Farkas certificate of infeasibility (if any)
    bool farkas_has_cert = false; // whether farkas_y is valid
};

inline const char *to_string(LPSolution::Status s) {
    switch (s) {
    case LPSolution::Status::Optimal:
        return "optimal";
    case LPSolution::Status::Unbounded:
        return "unbounded";
    case LPSolution::Status::Infeasible:
        return "infeasible";
    case LPSolution::Status::IterLimit:
        return "iterlimit";
    case LPSolution::Status::Singular:
        return "singular";
    case LPSolution::Status::NeedPhase1:
        return "need_phase1";
    }
    return "unknown";
}

// ============================================================================
// Options
// ============================================================================
enum class SimplexMode { Auto, Primal, Dual };

struct RevisedSimplexOptions {
    // Global
    int max_iters = 50'000;
    double tol = 1e-9;
    bool bland = false;
    double svd_tol = 1e-8;
    double ratio_delta = 1e-12;
    double ratio_eta = 1e-7;
    double deg_step_tol = 1e-12;
    double epsilon_cost = 1e-10;
    int rng_seed = 13;

    // Basis / LU
    int refactor_every = 128; // FT hard cap
    int compress_every = 64;  // FT soft cap
    double lu_pivot_rel = 1e-12;
    double lu_abs_floor = 1e-16;
    double alpha_tol = 1e-10;
    double z_inf_guard = 1e6;

    // Pricing
    int devex_reset = 200;
    std::string pricing_rule = "adaptive"; // or "devex" / "most_negative"
    int adaptive_reset_freq = 1000;

    // Recovery
    int max_basis_rebuilds = 3;

    // Algorithm selection/tuning (append to your options)
    bool dual_allow_bound_flip = true;  // enable Beale bound-flipping
    double dual_flip_pivot_tol = 1e-10; // |pN(e)| below this ⇒ consider flip
    double dual_flip_rc_tol = 1e-10;    // |rN(e)| “near dual-feasible”
    int dual_flip_max_per_iter = 2;     // avoid pathological flip storms

    // Algorithm selection
    SimplexMode mode = SimplexMode::Auto; // Auto | Primal | Dual
};

// Forward decls from degeneracy/pricer header (kept external)
static inline std::unordered_map<std::string, std::string>
dm_stats_to_map(const DegeneracyManager::Stats &s) {
    std::unordered_map<std::string, std::string> info;
    info["deg_streak"] = std::to_string(s.degeneracy_streak);
    info["deg_total"] = std::to_string(s.degeneracy_total);
    info["cycle_len"] = std::to_string(s.suspected_cycling);
    info["cond_est"] = std::to_string(s.cond_est);
    info["deg_thresh"] = std::to_string(s.adaptive_deg_threshold);
    info["deg_epoch"] = std::to_string(s.epoch);
    return info;
}

// ============================================================================
// RevisedSimplex
// ============================================================================
class RevisedSimplex {
public:
    explicit RevisedSimplex(RevisedSimplexOptions opt = {})
        : opt_(std::move(opt)), rng_(opt_.rng_seed), degen_(opt_.rng_seed),
          adaptive_pricer_(1) // initialized to a dummy size; rebuilt per solve
    {}

    // Main entry (drop-in compatible)
    LPSolution solve(const Eigen::MatrixXd &A_in, const Eigen::VectorXd &b_in,
                     const Eigen::VectorXd &c_in,
                     std::optional<std::vector<int>> basis_opt = std::nullopt) {
        const int n = static_cast<int>(A_in.cols());

        // ---- (0) Wrap into presolve LP: Ax=b, default bounds, costs=c ----
        presolve::LP lp;
        lp.A = A_in;
        lp.b = b_in;
        lp.sense.assign(static_cast<int>(A_in.rows()), presolve::RowSense::EQ);
        lp.c = c_in;
        lp.l = Eigen::VectorXd::Zero(n);
        lp.u = Eigen::VectorXd::Constant(n, presolve::inf());
        lp.c0 = 0.0;

        // ---- (1) Presolve ----
        presolve::Presolver::Options popt;
        popt.enable_rowreduce = true;
        popt.enable_scaling = true;
        popt.max_passes = 10;
        if (A_in.cols() > static_cast<int>(A_in.rows() * 1.2)) {
            popt.conservative_mode = true;
        }

        presolve::Presolver P(popt);
        const auto pres = P.run(lp);

        if (pres.proven_infeasible) {
            return make_solution_(LPSolution::Status::Infeasible,
                                  Eigen::VectorXd::Zero(n),
                                  std::numeric_limits<double>::infinity(), {},
                                  0, {{"presolve", "infeasible"}});
        }
        if (pres.proven_unbounded) {
            Eigen::VectorXd xnan = Eigen::VectorXd::Constant(
                n, std::numeric_limits<double>::quiet_NaN());
            return make_solution_(LPSolution::Status::Unbounded, xnan,
                                  -std::numeric_limits<double>::infinity(), {},
                                  0, {{"presolve", "unbounded"}});
        }

        const Eigen::MatrixXd &Atil = pres.reduced.A;
        const Eigen::VectorXd &btil = pres.reduced.b;
        const Eigen::VectorXd &ctil = pres.reduced.c;
        const Eigen::VectorXd &lred = pres.reduced.l;
        const Eigen::VectorXd &ured = pres.reduced.u;

        // ---- (2) m==0 fast path: optimize over bounds only ----
        if (Atil.rows() == 0) {
            Eigen::VectorXd vred =
                Eigen::VectorXd::Zero(static_cast<int>(ctil.size()));
            bool is_bounded = true;
            for (int j = 0; j < static_cast<int>(ctil.size()); ++j) {
                if (ctil(j) > opt_.tol) {
                    vred(j) = std::isfinite(lred(j)) ? lred(j) : 0.0;
                } else if (ctil(j) < -opt_.tol) {
                    if (std::isfinite(ured(j)))
                        vred(j) = ured(j);
                    else {
                        is_bounded = false;
                        break;
                    }
                } else {
                    vred(j) = std::isfinite(lred(j)) ? lred(j) : 0.0;
                }
            }
            if (!is_bounded) {
                Eigen::VectorXd xnan = Eigen::VectorXd::Constant(
                    n, std::numeric_limits<double>::quiet_NaN());
                return make_solution_(
                    LPSolution::Status::Unbounded, xnan,
                    -std::numeric_limits<double>::infinity(), {}, 0,
                    {{"presolve", "m=0 neg cost & +inf upper"}});
            }
            auto [x_full, obj_corr] = P.postsolve(vred);
            const double total_obj = c_in.dot(x_full) + obj_corr;
            return make_solution_(LPSolution::Status::Optimal, x_full,
                                  total_obj, {}, 0,
                                  {{"presolve", "m=0 optimized over bounds"}});
        }

        // ---- (3) Add explicit finite-UB rows/slacks when "reasonable" ----
        Eigen::MatrixXd Ared = Atil;
        Eigen::VectorXd bred = btil;
        Eigen::VectorXd cred = ctil;
        std::vector<int> col_orig_map, row_orig_map;

        std::tie(Ared, bred, cred, col_orig_map, row_orig_map) =
            build_with_upper_bounds_(Ared, bred, cred, pres, ured);

        const int m_eff = static_cast<int>(Ared.rows());
        const int n_eff = static_cast<int>(Ared.cols());

        // Effective bounds (reduced space)
        Eigen::VectorXd l_eff = Eigen::VectorXd::Zero(n_eff);
        Eigen::VectorXd u_eff =
            Eigen::VectorXd::Constant(n_eff, presolve::inf());
        for (int jr = 0; jr < n_eff; ++jr) {
            const int jorig = col_orig_map[jr];
            if (jorig >= 0) {
                l_eff(jr) = lred(jorig);
                u_eff(jr) = ured(jorig);
            }
        }

        // ---- (4) Map incoming basis into reduced space (optional) ----
        std::optional<std::vector<int>> red_basis_opt = std::nullopt;
        if (basis_opt && !basis_opt->empty()) {
            std::unordered_map<int, int> orig2red;
            orig2red.reserve(n_eff);
            for (int jr = 0; jr < n_eff; ++jr) {
                const int jorig = col_orig_map[jr];
                if (jorig >= 0)
                    orig2red[jorig] = jr;
            }
            std::vector<int> cand;
            cand.reserve(std::min(m_eff, (int)basis_opt->size()));
            for (int jorig : *basis_opt) {
                auto it = orig2red.find(jorig);
                if (it != orig2red.end()) {
                    cand.push_back(it->second);
                    if ((int)cand.size() == m_eff)
                        break;
                }
            }
            if ((int)cand.size() == m_eff)
                red_basis_opt = std::move(cand);
        }

        // ---- (5) Try Phase II directly on reduced problem (Primal/Dual per
        // mode) ----
        std::vector<int> basis_guess;
        if (red_basis_opt && (int)red_basis_opt->size() == m_eff) {
            basis_guess = *red_basis_opt;
        } else {
            if (auto maybe = find_initial_basis_(Ared, bred))
                basis_guess = *maybe;
        }

        const auto add_info =
            [&](std::unordered_map<std::string, std::string> info) {
                info["presolve_actions"] = std::to_string(pres.stack.size());
                info["reduced_m"] = std::to_string(m_eff);
                info["reduced_n"] = std::to_string(n_eff);
                info["obj_shift"] = std::to_string(pres.obj_shift);
                return info;
            };

        if ((int)basis_guess.size() == m_eff) {
            LPSolution::Status st;
            Eigen::VectorXd v2;
            std::vector<int> red_basis2;
            int it2;
            std::unordered_map<std::string, std::string> info2;

            auto run_primal = [&] {
                return phase_(Ared, bred, cred, basis_guess, l_eff, u_eff);
            };
            auto run_dual = [&] {
                return dual_phase_(Ared, bred, cred, basis_guess, l_eff, u_eff);
            };

            if (opt_.mode == SimplexMode::Dual) {
                std::tie(st, v2, red_basis2, it2, info2) = run_dual();
            } else if (opt_.mode == SimplexMode::Primal) {
                std::tie(st, v2, red_basis2, it2, info2) = run_primal();
            } else {
                // Auto: primal first; if primal reports negative basics → dual
                std::tie(st, v2, red_basis2, it2, info2) = run_primal();
                if (st == LPSolution::Status::NeedPhase1 &&
                    info2.count("reason") &&
                    info2.at("reason") == std::string("negative_basic_vars")) {
                    std::tie(st, v2, red_basis2, it2, info2) = run_dual();
                }
            }

            if (st == LPSolution::Status::Optimal ||
                st == LPSolution::Status::Unbounded ||
                st == LPSolution::Status::IterLimit) {
                auto [x_full, obj_corr] = P.postsolve(v2);
                const double total_obj = c_in.dot(x_full) + obj_corr;

                std::vector<int> basis_full;
                basis_full.reserve(red_basis2.size());
                for (int jr : red_basis2) {
                    if (jr >= 0 && jr < (int)col_orig_map.size()) {
                        const int jorig = col_orig_map[jr];
                        if (jorig >= 0)
                            basis_full.push_back(jorig);
                    }
                }
                auto info = add_info(std::move(info2));
                return make_solution_(st, x_full, total_obj, basis_full, it2,
                                      std::move(info));
            }
            if (st == LPSolution::Status::Singular) {
                auto info = add_info({});
                return make_solution_(LPSolution::Status::Singular,
                                      Eigen::VectorXd::Zero(n),
                                      std::numeric_limits<double>::quiet_NaN(),
                                      {}, 0, std::move(info));
            }
        }

        // ---- (6) Phase I on reduced problem ----
        auto [A1, b1, c1, basis1, n_orig_eff, m_rows] =
            make_phase1_(Ared, bred);
        auto [status1, v1, basis1_out, it1, info1] =
            phase_(A1, b1, c1, basis1, Eigen::VectorXd::Zero(A1.cols()),
                   Eigen::VectorXd::Constant(A1.cols(), presolve::inf()));

        // If phase I fails or artificial cost > tol ⇒ infeasible
        if (status1 != LPSolution::Status::Optimal || c1.dot(v1) > opt_.tol) {
            auto info = add_info({{"phase1_status", to_string(status1)}});
            const auto s = degen_.get_stats();
            auto more = dm_stats_to_map(s);
            info.insert(more.begin(), more.end());
            return make_solution_(LPSolution::Status::Infeasible,
                                  Eigen::VectorXd::Zero(n),
                                  std::numeric_limits<double>::infinity(), {},
                                  it1, std::move(info));
        }

        // Warm-start Phase II basis by removing artificials
        std::vector<int> red_basis2;
        red_basis2.reserve(m_rows);
        for (int j : basis1_out)
            if (j < (int)n_orig_eff)
                red_basis2.push_back(j);

        // Basis completion if needed
        if ((int)red_basis2.size() < m_rows) {
            for (int j = 0; j < (int)n_orig_eff; ++j) {
                if ((int)red_basis2.size() == m_rows)
                    break;
                if (std::find(red_basis2.begin(), red_basis2.end(), j) !=
                    red_basis2.end())
                    continue;
                std::vector<int> cand = red_basis2;
                cand.push_back(j);
                if ((int)cand.size() > m_rows)
                    continue;
                const Eigen::MatrixXd Btest =
                    Ared(Eigen::all,
                         Eigen::VectorXi::Map(cand.data(), (int)cand.size()));
                Eigen::FullPivLU<Eigen::MatrixXd> lu(Btest);
                if (lu.rank() == (int)cand.size() && lu.isInvertible())
                    red_basis2 = std::move(cand);
            }
        }

        // Final Phase II on reduced problem (respect mode)
        LPSolution::Status status2;
        Eigen::VectorXd v2;
        std::vector<int> red_basis_out;
        int it2 = 0;
        std::unordered_map<std::string, std::string> info2;

        if ((int)red_basis2.size() == m_rows) {
            if (opt_.mode == SimplexMode::Dual) {
                std::tie(status2, v2, red_basis_out, it2, info2) =
                    dual_phase_(Ared, bred, cred, red_basis2, l_eff, u_eff);
                if (status2 == LPSolution::Status::Infeasible) {
                    auto it = info2.find("farkas_has_cert");
                    if (it != info2.end() && it->second == "1") {
                        // parse CSV into a vector
                        Eigen::VectorXd yF(m_eff);
                        {
                            std::vector<double> vals;
                            vals.reserve(m_eff);
                            std::stringstream ss(info2["farkas_y"]);
                            std::string tok;
                            while (std::getline(ss, tok, ','))
                                vals.push_back(std::stod(tok));
                            yF = Eigen::Map<const Eigen::VectorXd>(
                                vals.data(), (int)vals.size());
                        }
                        return make_solution_(
                            LPSolution::Status::Infeasible,
                            Eigen::VectorXd::Zero(n),
                            std::numeric_limits<double>::infinity(), {}, it2,
                            add_info(std::move(info2)), yF, true);
                    }
                }

            } else if (opt_.mode == SimplexMode::Primal) {
                std::tie(status2, v2, red_basis_out, it2, info2) =
                    phase_(Ared, bred, cred, red_basis2, l_eff, u_eff);
            } else {
                // Auto: primal first; if negative basics → dual
                std::tie(status2, v2, red_basis_out, it2, info2) =
                    phase_(Ared, bred, cred, red_basis2, l_eff, u_eff);
                if (status2 == LPSolution::Status::NeedPhase1 &&
                    info2.count("reason") &&
                    info2.at("reason") == std::string("negative_basic_vars")) {
                    std::tie(status2, v2, red_basis_out, it2, info2) =
                        dual_phase_(Ared, bred, cred, red_basis2, l_eff, u_eff);
                }
            }
        } else {
            // Fall back to find a basis internally
            std::tie(status2, v2, red_basis_out, it2, info2) =
                phase_(Ared, bred, cred, std::nullopt, l_eff, u_eff);
            if (status2 == LPSolution::Status::NeedPhase1) {
                status2 = LPSolution::Status::Singular;
                info2["note"] = "reduced matrix cannot form a proper basis";
            }
        }

        const int total_iters = it1 + it2;
        auto merged_info = add_info(std::move(info2));
        merged_info.insert({"phase1_iters", std::to_string(it1)});

        auto [x_full, obj_correction] = P.postsolve(v2);
        const double total_obj = c_in.dot(x_full) + obj_correction;

        std::vector<int> basis_full;
        basis_full.reserve(red_basis_out.size());
        for (int jr : red_basis_out) {
            if (jr >= 0 && jr < (int)col_orig_map.size()) {
                const int jorig = col_orig_map[jr];
                if (jorig >= 0)
                    basis_full.push_back(jorig);
            }
        }

        if (status2 == LPSolution::Status::Optimal) {
            return make_solution_(LPSolution::Status::Optimal, x_full,
                                  total_obj, basis_full, total_iters,
                                  std::move(merged_info));
        }
        if (status2 == LPSolution::Status::Unbounded) {
            return make_solution_(LPSolution::Status::Unbounded, x_full,
                                  -std::numeric_limits<double>::infinity(),
                                  basis_full, total_iters,
                                  std::move(merged_info));
        }

        const double obj_fallback =
            x_full.array().isFinite().all()
                ? total_obj
                : std::numeric_limits<double>::quiet_NaN();
        return make_solution_(status2, x_full, obj_fallback, basis_full,
                              total_iters, std::move(merged_info));
    }

private:
    // =========================================================================
    // Helpers (private; signatures preserved where externally referenced)
    // =========================================================================

    static Eigen::VectorXd clip_small_(Eigen::VectorXd x, double tol = 1e-12) {
        for (int i = 0; i < x.size(); ++i)
            if (std::abs(x(i)) < tol)
                x(i) = 0.0;
        return x;
    }

    static std::optional<std::vector<int>>
    find_initial_basis_(const Eigen::MatrixXd &A, const Eigen::VectorXd &b) {
        const int m = static_cast<int>(A.rows());
        const int n = static_cast<int>(A.cols());
        std::vector<int> basis;
        basis.reserve(m);
        std::vector<bool> used_row(m, false);

        for (int j = 0; j < n; ++j) {
            const Eigen::VectorXd col = A.col(j);
            int one_idx = -1, ones = 0;
            bool zeros_ok = true;
            for (int i = 0; i < m; ++i) {
                const double v = col(i);
                if (std::abs(v - 1.0) <= 1e-12) {
                    one_idx = i;
                    ++ones;
                } else if (std::abs(v) > 1e-12) {
                    zeros_ok = false;
                    break;
                }
            }
            if (ones == 1 && zeros_ok && !used_row[one_idx] &&
                b(one_idx) >= -1e-12) {
                basis.push_back(j);
                used_row[one_idx] = true;
                if ((int)basis.size() == m)
                    break;
            }
        }
        if ((int)basis.size() == m)
            return basis;
        return std::nullopt;
    }

    static std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd,
                      std::vector<int>, std::size_t, int>
    make_phase1_(const Eigen::MatrixXd &A, const Eigen::VectorXd &b) {
        const int m = static_cast<int>(A.rows());
        const int n = static_cast<int>(A.cols());

        Eigen::MatrixXd A1 = A;
        Eigen::VectorXd b1 = b;
        for (int i = 0; i < m; ++i)
            if (b1(i) < 0) {
                A1.row(i) *= -1.0;
                b1(i) *= -1.0;
            }

        Eigen::MatrixXd A_aux(m, n + m);
        A_aux.leftCols(n) = A1;
        A_aux.rightCols(m) = Eigen::MatrixXd::Identity(m, m);

        Eigen::VectorXd c_aux(n + m);
        c_aux.setZero();
        c_aux.tail(m).setOnes();

        std::vector<int> basis(m);
        std::iota(basis.begin(), basis.end(), n);

        return {A_aux, b1, c_aux, basis, static_cast<std::size_t>(n), m};
    }

    // Add finite-UB rows/slacks for "reasonable" uppers; return maps
    static std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd,
                      std::vector<int>, std::vector<int>>
    build_with_upper_bounds_(const Eigen::MatrixXd &A0,
                             const Eigen::VectorXd &b0,
                             const Eigen::VectorXd &c0,
                             const presolve::PresolveResult &pres,
                             const Eigen::VectorXd &u_in) {
        const int m0 = static_cast<int>(A0.rows());
        const int n0 = static_cast<int>(A0.cols());

        std::vector<int> ub_idx;
        ub_idx.reserve(n0);
        constexpr double kMaxReasonableUB = 1e6;
        for (int j = 0; j < n0; ++j)
            if (std::isfinite(u_in(j)) && u_in(j) <= kMaxReasonableUB)
                ub_idx.push_back(j);

        const int p = static_cast<int>(ub_idx.size());
        if (p == 0) {
            std::vector<int> col_orig_map(n0);
            for (int j = 0; j < n0; ++j)
                col_orig_map[j] = pres.orig_col_index[j];
            std::vector<int> row_orig_map(m0);
            std::iota(row_orig_map.begin(), row_orig_map.end(), 0);
            return {A0, b0, c0, col_orig_map, row_orig_map};
        }

        Eigen::MatrixXd A(m0 + p, n0 + p);
        A.setZero();
        A.topLeftCorner(m0, n0) = A0;

        Eigen::VectorXd b(m0 + p);
        b.head(m0) = b0;

        Eigen::VectorXd c(n0 + p);
        c.head(n0) = c0;
        c.tail(p).setZero(); // slacks are free (cost 0)

        for (int k = 0; k < p; ++k) {
            const int j = ub_idx[k];
            const int row = m0 + k;
            const int slack_col = n0 + k;
            A(row, j) = 1.0;
            A(row, slack_col) = 1.0;
            b(row) = u_in(j);
        }

        std::vector<int> col_orig_map(n0 + p, -1);
        for (int j = 0; j < n0; ++j)
            col_orig_map[j] = pres.orig_col_index[j];
        std::vector<int> row_orig_map(m0 + p);
        std::iota(row_orig_map.begin(), row_orig_map.end(), 0);

        return {A, b, c, col_orig_map, row_orig_map};
    }

    // Harris ratio (improved): returns (leaving_row, theta_B) for primal
    std::pair<std::optional<int>, double>
    harris_ratio_(const Eigen::VectorXd &xB, const Eigen::VectorXd &dB,
                  double delta, double eta) const {
        std::vector<int> pos;
        pos.reserve(dB.size());
        for (int i = 0; i < dB.size(); ++i)
            if (dB(i) > delta)
                pos.push_back(i);
        if (pos.empty())
            return {std::nullopt, std::numeric_limits<double>::infinity()};

        double theta_star = std::numeric_limits<double>::infinity();
        for (int idx : pos)
            theta_star = std::min(theta_star, xB(idx) / dB(idx));

        double max_resid = 0.0;
        std::vector<int> candidates;
        for (int idx : pos) {
            const double ratio = xB(idx) / dB(idx);
            if (std::abs(ratio - theta_star) <= 1e-10)
                candidates.push_back(idx);
            const double resid = xB(idx) - theta_star * dB(idx);
            max_resid = std::max(max_resid, std::max(0.0, resid));
        }
        if (!candidates.empty()) {
            int best = candidates.front();
            for (int idx : candidates)
                if (idx < best)
                    best = idx;
            return {best, theta_star};
        }

        const double kappa = std::max(eta, eta * max_resid);
        std::vector<int> eligible;
        for (int idx : pos) {
            const double resid = xB(idx) - theta_star * dB(idx);
            if (resid <= kappa)
                eligible.push_back(idx);
        }
        if (!eligible.empty()) {
            int best = eligible.front();
            for (int idx : eligible)
                if (idx < best)
                    best = idx;
            return {best, theta_star};
        }

        int best = pos.front();
        double best_ratio = xB(best) / dB(best);
        for (int i = 1; i < (int)pos.size(); ++i) {
            const int idx = pos[i];
            const double r = xB(idx) / dB(idx);
            if (r < best_ratio) {
                best_ratio = r;
                best = idx;
            }
        }
        return {best, best_ratio};
    }

    // BFRT: permissible step for entering variable to nearest active bound
    // (primal)
    struct BFRTStep {
        double theta_e = std::numeric_limits<double>::infinity();
        bool to_upper = false;
    };

    BFRTStep entering_bound_step_(double x_e, double l_e, double u_e,
                                  double rc_e, double tol) const {
        BFRTStep out;
        // Primal simplex (minimization): rc_e < 0 ⇒ objective improves as x_e
        // increases
        if (rc_e < -tol) {
            if (std::isfinite(u_e)) {
                out.theta_e = std::max(0.0, u_e - x_e);
                out.to_upper = true;
            }
        } else if (rc_e > tol) { // move downward
            if (std::isfinite(l_e)) {
                out.theta_e = std::max(0.0, x_e - l_e);
                out.to_upper = false;
            }
        }
        return out;
    }

    // --------------------------- PRIMAL PHASE ---------------------------
    std::tuple<LPSolution::Status, Eigen::VectorXd, std::vector<int>, int,
               std::unordered_map<std::string, std::string>>
    phase_(const Eigen::MatrixXd &A, const Eigen::VectorXd &b,
           const Eigen::VectorXd &c, std::optional<std::vector<int>> basis_opt,
           const Eigen::VectorXd &l, const Eigen::VectorXd &u) {
        const int m = static_cast<int>(A.rows());
        const int n = static_cast<int>(A.cols());
        int iters = 0;

        // --- Basis initialization ---
        std::vector<int> basis;
        if (basis_opt) {
            basis = *basis_opt;
            if ((int)basis.size() != m)
                return {LPSolution::Status::NeedPhase1,
                        Eigen::VectorXd::Zero(n),
                        {},
                        0,
                        {{"reason", "basis size != m"}}};
        } else {
            auto maybe = find_initial_basis_(A, b);
            if (!maybe)
                return {LPSolution::Status::NeedPhase1,
                        Eigen::VectorXd::Zero(n),
                        {},
                        0,
                        {{"reason", "no_trivial_basis"}}};
            basis = *maybe;
        }

        // Nonbasic list N
        std::vector<int> N;
        N.reserve(n - m);
        {
            std::vector<char> inB(n, 0);
            for (int j : basis) {
                if (j < 0 || j >= n)
                    return {LPSolution::Status::Singular,
                            Eigen::VectorXd::Zero(n),
                            basis,
                            0,
                            {{"where", "initial basis index out of range"}}};
                inB[j] = 1;
            }
            for (int j = 0; j < n; ++j)
                if (!inB[j])
                    N.push_back(j);
        }

        // Basis factor
        FTBasis B(A, basis, opt_.refactor_every, opt_.compress_every,
                  opt_.lu_pivot_rel, opt_.lu_abs_floor, opt_.alpha_tol,
                  opt_.z_inf_guard);

        // Adaptive pricing setup (if enabled)
        if (opt_.pricing_rule == "adaptive") {
            AdaptivePricer::PricingOptions popts;
            popts.steepest_pool_max = 0;
            popts.steepest_reset_freq = opt_.adaptive_reset_freq;
            adaptive_pricer_ = AdaptivePricer(n, popts);
            adaptive_pricer_.build_pools(B, A, N);
            bridge_ = std::make_unique<DegeneracyPricerBridge<AdaptivePricer>>(
                degen_, adaptive_pricer_);
        }

        int rebuild_attempts = 0;

        // --- Main loop ---
        while (iters < opt_.max_iters) {
            ++iters;

            // xB = B^{-1} b
            Eigen::VectorXd xB;
            try {
                xB = B.solve_B(b);
            } catch (...) {
                if (rebuild_attempts < opt_.max_basis_rebuilds) {
                    ++rebuild_attempts;
                    B.refactor();
                    if (opt_.pricing_rule == "adaptive") {
                        adaptive_pricer_.build_pools(B, A, N);
                        adaptive_pricer_.clear_rebuild_flag();
                    }
                    continue;
                }
                return {LPSolution::Status::Singular,
                        Eigen::VectorXd::Zero(n),
                        basis,
                        iters,
                        {{"where", "solve(B,b) repair failed"}}};
            }

            // Basic feasibility
            if ((xB.array() < -opt_.tol).any()) {
                // Signal to caller that primal is infeasible for this basis
                return {LPSolution::Status::NeedPhase1,
                        Eigen::VectorXd::Zero(n),
                        basis,
                        iters,
                        {{"reason", "negative_basic_vars"}}};
            }
            xB = xB.cwiseMax(0.0);

            // y = B^{-T} c_B
            Eigen::VectorXd cB(m);
            for (int i = 0; i < m; ++i)
                cB(i) = c(basis[i]);

            Eigen::VectorXd y;
            try {
                y = B.solve_BT(cB);
            } catch (...) {
                B.refactor();
                y = B.solve_BT(cB);
                if (opt_.pricing_rule == "adaptive") {
                    adaptive_pricer_.build_pools(B, A, N);
                    adaptive_pricer_.clear_rebuild_flag();
                }
            }

            // Reduced costs rN on nonbasics
            Eigen::VectorXd rN(N.size());
            for (int k = 0; k < (int)N.size(); ++k) {
                const int j = N[k];
                rN(k) = c(j) - A.col(j).dot(y);
            }

            // Choose entering
            std::optional<int> e_rel;

            if (opt_.bland) {
                int idx = -1;
                for (int k = 0; k < (int)N.size(); ++k)
                    if (rN(k) < -opt_.tol) {
                        idx = k;
                        break;
                    }
                if (idx < 0) {
                    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
                    for (int i = 0; i < m; ++i)
                        x(basis[i]) = xB(i);
                    return {LPSolution::Status::Optimal, clip_small_(x), basis,
                            iters, dm_stats_to_map(degen_.get_stats())};
                }
                e_rel = idx;
            } else {
                if (opt_.pricing_rule == "adaptive") {
                    // Current objective for adaptive signals
                    Eigen::VectorXd xcur = Eigen::VectorXd::Zero(n);
                    for (int i = 0; i < m; ++i)
                        xcur(basis[i]) = xB(i);
                    const double current_obj = c.dot(xcur);

                    e_rel = bridge_->choose_entering(rN, N, opt_.tol, iters,
                                                     current_obj, B, A);
                } else {
                    // Most negative rc
                    int idx = -1;
                    double best = 0.0;
                    for (int k = 0; k < (int)N.size(); ++k)
                        if (rN(k) < -opt_.tol) {
                            if (idx < 0 || rN(k) < best) {
                                best = rN(k);
                                idx = k;
                            }
                        }
                    if (idx >= 0)
                        e_rel = idx;
                }

                if (!e_rel) {
                    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
                    for (int i = 0; i < m; ++i)
                        x(basis[i]) = xB(i);
                    return {LPSolution::Status::Optimal, clip_small_(x), basis,
                            iters, dm_stats_to_map(degen_.get_stats())};
                }
            }

            const int e = N[*e_rel];
            const auto aE = A.col(e);

            // dB = B^{-1} a_e
            Eigen::VectorXd dB;
            try {
                dB = B.solve_B(aE);
            } catch (...) {
                B.refactor();
                dB = B.solve_B(aE);
                if (opt_.pricing_rule == "adaptive") {
                    adaptive_pricer_.build_pools(B, A, N);
                    adaptive_pricer_.clear_rebuild_flag();
                }
            }

            // Harris ratio (leaving) from basics
            auto [leave_rel_opt, theta_B] =
                harris_ratio_(xB, dB, opt_.ratio_delta, opt_.ratio_eta);

            // BFRT step for entering bound
            const int idxN = *e_rel;
            const double rc_e = rN(idxN);
            const double l_e = (e >= 0 && e < l.size()) ? l(e) : 0.0;
            const double u_e =
                (e >= 0 && e < u.size()) ? u(e) : presolve::inf();
            const double x_e =
                std::isfinite(l_e) ? l_e : 0.0; // nonbasic at bound
            const BFRTStep bfrt =
                entering_bound_step_(x_e, l_e, u_e, rc_e, opt_.tol);

            double step = std::min(theta_B, bfrt.theta_e);
            if (!std::isfinite(step)) {
                Eigen::VectorXd x = Eigen::VectorXd::Constant(
                    n, std::numeric_limits<double>::quiet_NaN());
                return {LPSolution::Status::Unbounded, x, basis, iters,
                        dm_stats_to_map(degen_.get_stats())};
            }

            // If BFRT wins strictly, flip enter direction locally
            const bool flip_entering = (bfrt.theta_e + 1e-14 < theta_B);
            if (flip_entering) {
                dB = -dB;
                const_cast<Eigen::VectorXd &>(rN)(idxN) = -rc_e;
            }

            if (!leave_rel_opt) {
                Eigen::VectorXd x = Eigen::VectorXd::Constant(
                    n, std::numeric_limits<double>::quiet_NaN());
                return {LPSolution::Status::Unbounded, x, basis, iters,
                        dm_stats_to_map(degen_.get_stats())};
            }

            const int r = *leave_rel_opt;
            const double alpha = dB(r);
            const int oldAbs = basis[r];
            const int eAbs = e;

            // Degeneracy signals
            const bool is_degenerate =
                degen_.detect_degeneracy(step, opt_.deg_step_tol);
            if (is_degenerate && degen_.should_apply_perturbation()) {
                auto [Ap, bp, cp] =
                    degen_.apply_perturbation(A, b, c, basis, iters);
                (void)Ap;
                (void)bp;
                (void)cp; // no-op by default, preserves API
            } else {
                (void)degen_.reset_perturbation();
            }

            // Pricer updates
            if (opt_.pricing_rule == "adaptive") {
                const double rc_impr = -rN(idxN);
                bridge_->after_pivot(r, eAbs, oldAbs, dB, alpha, step, A, N,
                                     rc_impr);
            }

            // Pivot indices
            basis[r] = eAbs;
            N[idxN] = oldAbs;

            // Update basis matrix column
            try {
                B.replace_column(r, aE);
            } catch (...) {
                B.refactor();
                if (opt_.pricing_rule == "adaptive") {
                    adaptive_pricer_.build_pools(B, A, N);
                    adaptive_pricer_.clear_rebuild_flag();
                }
            }

            // Rebuild pricing pools if requested
            if (opt_.pricing_rule == "adaptive" &&
                adaptive_pricer_.needs_rebuild()) {
                adaptive_pricer_.build_pools(B, A, N);
                adaptive_pricer_.clear_rebuild_flag();
            }
        }

        return {LPSolution::Status::IterLimit, Eigen::VectorXd::Zero(n), basis,
                iters, dm_stats_to_map(degen_.get_stats())};
    }

    // --------------------------- DUAL PHASE ---------------------------
    // Dual Harris two-pass ratio test: entering e for leaving row r.
    // Inputs:
    //   r: leaving basic row (xB(r) < 0)
    //   rN: reduced costs on N (dual-feasible target rN >= -tol)
    //   pN: row r of B^{-1}A_N
    // Eligibility: pN(k) < -delta (so increasing τ ≥ 0 increases x_r by -τ
    // pN(k))
    struct DualChoose {
        std::optional<int> e_rel;
        double tau = std::numeric_limits<double>::infinity();
    };

    DualChoose dual_harris_choose_(const Eigen::VectorXd &rN,
                                   const Eigen::VectorXd &pN, double delta,
                                   double eta) const {
        std::vector<int> E;
        E.reserve((int)pN.size());
        for (int k = 0; k < pN.size(); ++k)
            if (pN(k) < -delta)
                E.push_back(k);
        if (E.empty())
            return {};

        // First pass τ* = min_k rN_k / -pN_k
        double tau_star = std::numeric_limits<double>::infinity();
        for (int k : E)
            tau_star = std::min(tau_star, rN(k) / (-pN(k)));

        // Second pass: allow slight relaxation
        const double kappa = std::max(eta, eta * std::abs(tau_star));
        std::vector<int> candidates;
        for (int k : E) {
            if ((rN(k) / (-pN(k))) <= tau_star + kappa)
                candidates.push_back(k);
        }
        if (!candidates.empty()) {
            // Bland-ish tie-break
            int best = candidates.front();
            double best_ratio = rN(best) / (-pN(best));
            for (int kk : candidates) {
                const double val = rN(kk) / (-pN(kk));
                if ((val < best_ratio - 1e-16) ||
                    (std::abs(val - best_ratio) <= 1e-16 && kk < best)) {
                    best = kk;
                    best_ratio = val;
                }
            }
            return {best, std::max(0.0, best_ratio)};
        }

        // Fallback to strict minimum
        int best = E.front();
        double best_ratio = rN(best) / (-pN(best));
        for (int i = 1; i < (int)E.size(); ++i) {
            const int k = E[i];
            const double val = rN(k) / (-pN(k));
            if (val < best_ratio) {
                best_ratio = val;
                best = k;
            }
        }
        return {best, std::max(0.0, best_ratio)};
    }
    std::tuple<LPSolution::Status, Eigen::VectorXd, std::vector<int>, int,
               std::unordered_map<std::string, std::string>>
    dual_phase_(const Eigen::MatrixXd &A, const Eigen::VectorXd &b,
                const Eigen::VectorXd &c,
                std::optional<std::vector<int>> basis_opt,
                const Eigen::VectorXd &l, const Eigen::VectorXd &u) {
        const int m = static_cast<int>(A.rows());
        const int n = static_cast<int>(A.cols());
        int iters = 0;

        // --- Basis initialization ---
        std::vector<int> basis;
        if (basis_opt) {
            basis = *basis_opt;
            if ((int)basis.size() != m)
                return {LPSolution::Status::NeedPhase1,
                        Eigen::VectorXd::Zero(n),
                        {},
                        0,
                        {{"reason", "basis size != m"}}};
        } else {
            auto maybe = find_initial_basis_(A, b);
            if (!maybe)
                return {LPSolution::Status::NeedPhase1,
                        Eigen::VectorXd::Zero(n),
                        {},
                        0,
                        {{"reason", "no_trivial_basis"}}};
            basis = *maybe;
        }

        // Nonbasic list N
        std::vector<int> N;
        N.reserve(n - m);
        {
            std::vector<char> inB(n, 0);
            for (int j : basis) {
                if (j < 0 || j >= n)
                    return {LPSolution::Status::Singular,
                            Eigen::VectorXd::Zero(n),
                            basis,
                            0,
                            {{"where", "initial basis index out of range"}}};
                inB[j] = 1;
            }
            for (int j = 0; j < n; ++j)
                if (!inB[j])
                    N.push_back(j);
        }

        // Basis factor
        FTBasis B(A, basis, opt_.refactor_every, opt_.compress_every,
                  opt_.lu_pivot_rel, opt_.lu_abs_floor, opt_.alpha_tol,
                  opt_.z_inf_guard);

        // Adaptive pricing setup (reused hooks)
        if (opt_.pricing_rule == "adaptive") {
            AdaptivePricer::PricingOptions popts;
            popts.steepest_pool_max = 0;
            popts.steepest_reset_freq = opt_.adaptive_reset_freq;
            adaptive_pricer_ = AdaptivePricer(n, popts);
            adaptive_pricer_.build_pools(B, A, N);
            bridge_ = std::make_unique<DegeneracyPricerBridge<AdaptivePricer>>(
                degen_, adaptive_pricer_);
        }

        // σ-orientation for nonbasics: +1 ⇒ lower-bound view, −1 ⇒ upper-bound
        // view.
        std::vector<int> sigma(n, +1);

        int rebuild_attempts = 0;
        int total_flips = 0;

        // helper: serialize a vector to CSV for info map
        auto serialize_vec = [](const Eigen::VectorXd &v) {
            std::ostringstream oss;
            oss.setf(std::ios::scientific);
            oss << std::setprecision(17);
            for (int i = 0; i < v.size(); ++i) {
                if (i)
                    oss << ",";
                oss << v(i);
            }
            return oss.str();
        };

        // --- Main dual loop ---
        while (iters < opt_.max_iters) {
            ++iters;
            int flips_this_iter = 0;

            // Solve xB = B^{-1} b (to find primal infeasibilities)
            Eigen::VectorXd xB;
            try {
                xB = B.solve_B(b);
            } catch (...) {
                if (rebuild_attempts < opt_.max_basis_rebuilds) {
                    ++rebuild_attempts;
                    B.refactor();
                    if (opt_.pricing_rule == "adaptive") {
                        adaptive_pricer_.build_pools(B, A, N);
                        adaptive_pricer_.clear_rebuild_flag();
                    }
                    continue;
                }
                return {LPSolution::Status::Singular,
                        Eigen::VectorXd::Zero(n),
                        basis,
                        iters,
                        {{"where", "dual: solve(B,b) repair failed"}}};
            }

            // Detect most negative basic (primal infeasibility)
            int r_leave = -1;
            double most_neg = -opt_.tol;
            for (int i = 0; i < m; ++i) {
                if (xB(i) < most_neg) {
                    most_neg = xB(i);
                    r_leave = i;
                }
            }

            if (r_leave < 0) {
                // All basics ≥ 0 → primal feasible. Check dual feasibility.
                Eigen::VectorXd cB(m);
                for (int i = 0; i < m; ++i)
                    cB(i) = c(basis[i]);
                Eigen::VectorXd y;
                try {
                    y = B.solve_BT(cB);
                } catch (...) {
                    B.refactor();
                    y = B.solve_BT(cB);
                }

                bool dual_feasible = true;
                for (int k = 0; k < (int)N.size(); ++k) {
                    const int j = N[k];
                    const double r_orig = c(j) - A.col(j).dot(y);
                    const double r_eff = (double)sigma[j] * r_orig;
                    if (r_eff < -opt_.tol) {
                        dual_feasible = false;
                        break;
                    }
                }
                if (dual_feasible) {
                    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
                    for (int i = 0; i < m; ++i)
                        x(basis[i]) = std::max(0.0, xB(i));
                    auto info_map = dm_stats_to_map(degen_.get_stats());
                    info_map["dual_beale_flips"] = std::to_string(total_flips);
                    return {LPSolution::Status::Optimal, clip_small_(x), basis,
                            iters, std::move(info_map)};
                }
                return {LPSolution::Status::NeedPhase1,
                        Eigen::VectorXd::Zero(n),
                        basis,
                        iters,
                        {{"reason", "dual_infeasible_at_primal_feasible"}}};
            }

            // Build dual pricing row: w = B^{-T} e_r; then pN = σ_j * w^T a_j
            Eigen::VectorXd e_r = Eigen::VectorXd::Zero(m);
            e_r(r_leave) = 1.0;
            Eigen::VectorXd w;
            try {
                w = B.solve_BT(e_r);
            } catch (...) {
                B.refactor();
                w = B.solve_BT(e_r);
                if (opt_.pricing_rule == "adaptive") {
                    adaptive_pricer_.build_pools(B, A, N);
                    adaptive_pricer_.clear_rebuild_flag();
                }
            }

            Eigen::VectorXd pN(N.size());
            for (int k = 0; k < (int)N.size(); ++k) {
                const int j = N[k];
                pN(k) = (double)sigma[j] * w.dot(A.col(j));
            }

            // Compute y and effective reduced costs rN = σ_j * (c_j − a_j^T y)
            Eigen::VectorXd cB(m);
            for (int i = 0; i < m; ++i)
                cB(i) = c(basis[i]);
            Eigen::VectorXd y;
            try {
                y = B.solve_BT(cB);
            } catch (...) {
                B.refactor();
                y = B.solve_BT(cB);
            }
            Eigen::VectorXd rN(N.size());
            for (int k = 0; k < (int)N.size(); ++k) {
                const int j = N[k];
                rN(k) = (double)sigma[j] * (c(j) - A.col(j).dot(y));
            }

            // Choose entering with dual Harris rule on (rN, pN)
            DualChoose dc =
                dual_harris_choose_(rN, pN, opt_.ratio_delta, opt_.ratio_eta);
            if (!dc.e_rel) {
                if (rebuild_attempts < opt_.max_basis_rebuilds) {
                    ++rebuild_attempts;
                    B.refactor();
                    if (opt_.pricing_rule == "adaptive") {
                        adaptive_pricer_.build_pools(B, A, N);
                        adaptive_pricer_.clear_rebuild_flag();
                    }
                    continue;
                }
                return {LPSolution::Status::Singular,
                        Eigen::VectorXd::Zero(n),
                        basis,
                        iters,
                        {{"where", "dual: no eligible entering"}}};
            }
            const int e_rel = *dc.e_rel;
            const int eAbs = N[e_rel];

            // ------------------ Beale’s bound flip (safe, surgical)
            // ------------------
            if (opt_.dual_allow_bound_flip &&
                flips_this_iter < opt_.dual_flip_max_per_iter) {
                const bool hasL = std::isfinite(l(eAbs));
                const bool hasU = std::isfinite(u(eAbs));
                const bool can_flip = hasL && hasU;

                if (can_flip &&
                    std::abs(pN(e_rel)) <= opt_.dual_flip_pivot_tol &&
                    std::abs(rN(e_rel)) <= opt_.dual_flip_rc_tol) {
                    sigma[eAbs] = -sigma[eAbs];
                    ++flips_this_iter;
                    ++total_flips;

                    const double r_orig_e = c(eAbs) - A.col(eAbs).dot(y);
                    rN(e_rel) = (double)sigma[eAbs] * r_orig_e;
                    pN(e_rel) = (double)sigma[eAbs] * w.dot(A.col(eAbs));

                    if (!(pN(e_rel) < -opt_.ratio_delta)) {
                        DualChoose dc2 = dual_harris_choose_(
                            rN, pN, opt_.ratio_delta, opt_.ratio_eta);
                        if (!dc2.e_rel) {
                            if (rebuild_attempts < opt_.max_basis_rebuilds) {
                                ++rebuild_attempts;
                                B.refactor();
                                if (opt_.pricing_rule == "adaptive") {
                                    adaptive_pricer_.build_pools(B, A, N);
                                    adaptive_pricer_.clear_rebuild_flag();
                                }
                                continue;
                            }
                            return {LPSolution::Status::Singular,
                                    Eigen::VectorXd::Zero(n),
                                    basis,
                                    iters,
                                    {{"where", "dual: post-flip no entering"}}};
                        }
                        dc = dc2;
                    }
                }
            }
            // ------------------------------------------------------------------------

            const double tau = dc.tau;
            if (!std::isfinite(tau)) {
                // Dual unbounded ⇒ primal infeasible. Build a Farkas
                // certificate yF = B^{-T} e_r.
                Eigen::VectorXd yF = w; // already B^{-T} e_r
                // Adjust sign: we want yF^T b < 0 and yF^T A ≥ 0 componentwise
                // if possible.
                if (yF.dot(b) >= 0)
                    yF = -yF;

                auto info_map = dm_stats_to_map(degen_.get_stats());
                info_map["where"] = "dual: infinite step";
                info_map["dual_beale_flips"] = std::to_string(total_flips);
                info_map["farkas_has_cert"] = "1";
                info_map["farkas_dim"] = std::to_string(m);
                info_map["farkas_y"] = serialize_vec(yF);

                return {LPSolution::Status::Infeasible,
                        Eigen::VectorXd::Zero(n), basis, iters,
                        std::move(info_map)};
            }

            // Degeneracy tracking
            const bool is_degenerate =
                degen_.detect_degeneracy(tau, opt_.deg_step_tol);
            if (is_degenerate && degen_.should_apply_perturbation()) {
                auto [Ap, bp, cp] =
                    degen_.apply_perturbation(A, b, c, basis, iters);
                (void)Ap;
                (void)bp;
                (void)cp;
            } else {
                (void)degen_.reset_perturbation();
            }

            // Pricer feedback (optional)
            if (opt_.pricing_rule == "adaptive") {
                const double rc_impr = tau;
                bridge_->after_pivot(r_leave, N[e_rel], basis[r_leave], pN,
                                     pN(e_rel), tau, A, N, rc_impr);
            }

            // Perform pivot: replace column r_leave by *effective* a_e with
            // orientation σ
            Eigen::VectorXd aE_eff = A.col(eAbs);
            if (sigma[eAbs] < 0)
                aE_eff = -aE_eff;

            const int oldAbs = basis[r_leave];
            basis[r_leave] = eAbs;
            N[e_rel] = oldAbs;

            try {
                B.replace_column(r_leave, aE_eff);
            } catch (...) {
                B.refactor();
                if (opt_.pricing_rule == "adaptive") {
                    adaptive_pricer_.build_pools(B, A, N);
                    adaptive_pricer_.clear_rebuild_flag();
                }
            }

            // Orientation updates:
            sigma[eAbs] = +1; // entering becomes basic
            sigma[oldAbs] =
                +1; // leaving becomes nonbasic (default to lower view)

            // Rebuild pricing pools if requested
            if (opt_.pricing_rule == "adaptive" &&
                adaptive_pricer_.needs_rebuild()) {
                adaptive_pricer_.build_pools(B, A, N);
                adaptive_pricer_.clear_rebuild_flag();
            }
        }

        auto info_map = dm_stats_to_map(degen_.get_stats());
        info_map["dual_beale_flips"] = std::to_string(total_flips);
        return {LPSolution::Status::IterLimit, Eigen::VectorXd::Zero(n), basis,
                iters, std::move(info_map)};
    }

    // --------------------------- Utilities ---------------------------
    static LPSolution
    make_solution_(LPSolution::Status st, Eigen::VectorXd x, double obj,
                   std::vector<int> basis, int iters,
                   std::unordered_map<std::string, std::string> info,
                   std::optional<Eigen::VectorXd> farkas_y = std::nullopt,
                   std::optional<bool> farkas_has_cert = std::nullopt) {
        LPSolution sol;
        sol.status = st;
        sol.x = std::move(x);
        sol.obj = obj;
        sol.basis = std::move(basis);
        sol.iters = iters;
        sol.info = std::move(info);
        sol.farkas_y = Eigen::VectorXd();
        sol.farkas_has_cert = false;
        return sol;
    }

private:
    // Options and state
    RevisedSimplexOptions opt_;
    std::mt19937 rng_;

    // Degeneracy + pricing
    DegeneracyManager degen_;
    AdaptivePricer adaptive_pricer_{1};
    std::unique_ptr<DegeneracyPricerBridge<AdaptivePricer>> bridge_;
};
