#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstring> // std::memcpy
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>

using dvec = Eigen::VectorXd;
using dmat = Eigen::MatrixXd;
using spmat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

// The value type your map stores (adjust if yours differs)
using Val = std::variant<double, dvec, dmat, spmat>;
using Dict = std::unordered_map<std::string, Val>;

// ---------- small helpers ----------

namespace nb = nanobind;
using namespace nb::literals;

// Get attribute with fallback; works with any castable T.
template <class T>
static inline T get_attr_or(const nb::handle &obj, const char *name,
                            const T &fallback) {
    if (!obj || !nb::hasattr(obj, name))
        return fallback;
    try {
        return nb::cast<T>(obj.attr(name));
    } catch (const nb::cast_error &) {
        return fallback;
    }
}

// ---------- Python attribute helpers ----------
namespace pyu {
[[nodiscard]] inline bool has_attr(const nb::object &o,
                                   const char *name) noexcept {
    return o.is_valid() && PyObject_HasAttrString(o.ptr(), name);
}

} // namespace pyu

struct KKT {
    double stat{0.0}, eq{0.0}, ineq{0.0}, comp{0.0};
};

namespace kkt {
// ---------------------------- reusable API ----------------------------

// ------------------------------ config -------------------------------
struct ChompConfig {
    double cg_tol = 1e-6;
    int cg_maxit = 50;
    double ip_hess_reg0 = 1e-8;
    double schur_dense_cutoff = 0.25;
    std::string prec_type = "ssor"; // "jacobi" | "ssor" | "none"
    double ssor_omega = 1.0;
    std::string sym_ordering = "false"; // "amd" | "none"
    bool use_simd = true;
    int block_size = 256;
    bool adaptive_gamma = false;
    bool assemble_schur_if_m_small = true;
    bool use_prec = true;
};

struct KKTReusable {
    virtual ~KKTReusable() = default;
    virtual std::pair<dvec, dvec> solve(const dvec &r1,
                                        const std::optional<dvec> &r2,
                                        double cg_tol = 1e-8,
                                        int cg_maxit = 200) = 0;
};

// ---------------------------- strategy base ---------------------------
struct KKTStrategy {
    virtual ~KKTStrategy() = default;
    virtual std::tuple<dvec, dvec, std::shared_ptr<KKTReusable>>
    factor_and_solve(const spmat &W, const std::optional<spmat> &G,
                     const dvec &r1, const std::optional<dvec> &r2,
                     const ChompConfig &cfg, std::optional<double> regularizer,
                     std::unordered_map<std::string, dvec> &cache,
                     double delta = 0.0,
                     std::optional<double> gamma = std::nullopt,
                     bool assemble_schur_if_m_small = true,
                     bool use_prec = true) = 0;
    std::string name;
};

} // namespace kkt

struct SolverInfo {
    std::string mode{"ip"}; // algorithmic mode (ip, sqp, etc.)
    double step_norm{0.0};
    bool accepted{true};
    bool converged{true};

    double f{0.0};     // objective value
    double theta{0.0}; // merit or filter measure
    double stat{0.0};  // gradient norm (safe_inf_norm(g))
    double ineq{0.0};  // infinity norm of violated inequalities
    double eq{0.0};    // infinity norm of equality violations
    double comp{0.0};  // complementarity measure

    int ls_iters{0};   // line search iterations
    double alpha{0.0}; // step size
    double rho{0.0};   // trust region ratio
    double tr_radius{0.0};
    double delta{0.0}; // trust region radius
    double mu{0.0};    // barrier parameter

    bool shifted_barrier{false}; // whether the barrier was shifted
    double tau_shift{0.0};       // amount of barrier shift
    double bound_shift{0.0};     // amount of bound shift
};
