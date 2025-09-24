// ad.cpp — nanobind + NumPy fast paths + GIL release (C++23, de-duplicated)
#pragma once
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "../include/ad/ADGraph.h"
#include "../include/ad/Definitions.h"
#include "../include/ad/Expression.h"
#include "../include/ad/Variable.h"

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <expected>
#include <functional>
#include <limits>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

// ---------- Memory pool for hot allocations ----------
static std::pmr::unsynchronized_pool_resource g_pool{
    {.max_blocks_per_chunk = 256, .largest_required_pool_block = 8192}};
thread_local std::pmr::vector<double> tl_scratch{&g_pool};

// ---------- Concepts ----------
template <typename T>
concept Numeric = std::is_arithmetic_v<T>;

template <typename T>
concept PyHandle = std::same_as<T, nb::handle> || std::same_as<T, nb::object>;

// ---------- ndarray aliases ----------
using Arr1D = nb::ndarray<double, nb::shape<-1>, nb::c_contig>;
using Arr2D = nb::ndarray<double, nb::shape<-1, -1>, nb::c_contig>;

// ---------- small hot helpers ----------
[[gnu::always_inline, gnu::hot]]
static inline bool is_number(const nb::handle &h) noexcept {
    return nb::isinstance<nb::float_>(h) || nb::isinstance<nb::int_>(h);
}

[[gnu::always_inline, gnu::hot]]
static inline bool is_sequence(const nb::handle &h) noexcept {
    return nb::isinstance<nb::list>(h) || nb::isinstance<nb::tuple>(h);
}

[[gnu::always_inline, gnu::hot]]
static inline std::span<const double> as_span_1d(const Arr1D &a) {
    if (a.ndim() != 1) [[unlikely]]
        throw std::invalid_argument("expected 1D float64 array");
    return {a.data(), (size_t)a.shape(0)};
}

[[gnu::always_inline]]
static inline std::pair<ssize_t, ssize_t> shape_2d(const Arr2D &a) {
    if (a.ndim() != 2) [[unlikely]]
        throw std::invalid_argument("expected 2D float64 array");
    return {a.shape(0), a.shape(1)};
}

[[gnu::always_inline]]
static inline Arr1D create_zeros_1d(ssize_t n) {
    auto numpy = nb::module_::import_("numpy");
    return nb::cast<Arr1D>(
        numpy.attr("zeros")(nb::make_tuple(n), "dtype"_a = "float64"));
}

[[gnu::always_inline]]
static inline Arr2D create_zeros_2d(ssize_t m, ssize_t n) {
    auto numpy = nb::module_::import_("numpy");
    return nb::cast<Arr2D>(
        numpy.attr("zeros")(nb::make_tuple(m, n), "dtype"_a = "float64"));
}

[[gnu::always_inline]]
static inline nb::tuple to_tuple(const std::vector<nb::object> &vec) {
    nb::list temp;
    for (const auto &item : vec)
        temp.append(item);
    return nb::tuple(temp);
}

// ---------- AD builders ----------
[[gnu::always_inline]]
static inline ADNodePtr make_const_node(const ADGraphPtr &g, double v) {
    auto n = std::make_shared<ADNode>();
    n->type = Operator::cte;
    n->value = v;
    if (g) [[likely]]
        g->addNode(n);
    return n;
}

[[gnu::always_inline]]
static inline std::shared_ptr<Expression> make_const_expr(double val,
                                                          const ADGraphPtr &g) {
    auto n = make_const_node(g, val);
    return std::make_shared<Expression>(n, g);
}

[[gnu::always_inline]]
static inline std::shared_ptr<Expression>
make_expr_from_variable(const std::shared_ptr<Variable> &v,
                        const ADGraphPtr &g) {
    return std::make_shared<Expression>(v, 1.0, g);
}

[[gnu::always_inline]]
static inline std::shared_ptr<Expression>
make_expr_from_number(double val, const ADGraphPtr &g) {
    return make_const_expr(val, g);
}

[[gnu::hot]]
static inline std::shared_ptr<Expression> as_expression(const nb::handle &h,
                                                        const ADGraphPtr &g) {
    if (nb::isinstance<Expression>(h)) [[likely]]
        return nb::cast<std::shared_ptr<Expression>>(h);
    if (nb::isinstance<Variable>(h)) [[likely]] {
        auto v = nb::cast<std::shared_ptr<Variable>>(h);
        return make_expr_from_variable(v, g);
    }
    if (is_number(h)) [[likely]]
        return make_expr_from_number(nb::cast<double>(h), g);
    throw std::invalid_argument(
        "Argument must be Expression, Variable, int, or float.");
}

[[gnu::hot]]
static inline std::shared_ptr<Expression>
ensure_expression(const nb::handle &ret, const ADGraphPtr &g) {
    if (nb::isinstance<Expression>(ret)) [[likely]]
        return nb::cast<std::shared_ptr<Expression>>(ret);
    if (is_number(ret)) [[likely]]
        return make_expr_from_number(nb::cast<double>(ret), g);
    throw std::invalid_argument(
        "Function must return Expression or a numeric value.");
}

// ---------- Python call helpers (tuple/list policy) ----------
enum class ArgPolicy : uint8_t { Tuple, List };

template <ArgPolicy P, typename Callable>
[[gnu::always_inline]]
static inline nb::object call_py_fn(Callable &&f,
                                    const std::vector<nb::object> &args) {
    if constexpr (P == ArgPolicy::Tuple) {
        nb::tuple t = to_tuple(args);
        return f(*t);
    } else {
        nb::list lst;
        for (auto &a : args)
            lst.append(a);
        return f(lst);
    }
}

// ---------- Input setters (templated) ----------
template <typename Vec, typename Span>
[[gnu::always_inline, gnu::hot]]
static inline void set_inputs_from_span(Vec &nodes, Span x) {
    if (x.size() != nodes.size()) [[unlikely]]
        throw std::invalid_argument("wrong input length");
    for (size_t i = 0; i < nodes.size(); ++i)
        nodes[i]->value = x[i];
}

template <typename Vec>
[[gnu::always_inline, gnu::hot]]
static inline void set_inputs_from_arr(Vec &nodes, const Arr1D &xin) {
    set_inputs_from_span(nodes, as_span_1d(xin));
}

template <typename Vec>
[[gnu::always_inline, gnu::hot]]
static inline void set_inputs_from_seq(Vec &nodes, const nb::object &x) {
    if (!is_sequence(x)) [[unlikely]]
        throw std::invalid_argument("expected a list/tuple");
    nb::sequence sx = nb::cast<nb::sequence>(x);
    if ((size_t)nb::len(sx) != nodes.size()) [[unlikely]]
        throw std::invalid_argument("wrong input length");
    for (size_t i = 0; i < nodes.size(); ++i)
        nodes[i]->value = nb::cast<double>(sx[i]);
}

// ---------- Unary ops ----------
[[gnu::always_inline]]
static inline double apply_unary_op(Operator op, double x) noexcept {
    switch (op) {
    case Operator::Sin:
        return std::sin(x);
    case Operator::Cos:
        return std::cos(x);
    case Operator::Tan:
        return std::tan(x);
    case Operator::Exp:
        return std::exp(x);
    case Operator::Log:
        return std::log(x);
    case Operator::Tanh:
        return std::tanh(x);
    case Operator::Relu:
        return x > 0.0 ? x : 0.0;
    case Operator::Silu:
        return x / (1.0 + std::exp(-x));
    case Operator::Gelu: {
        // tanh approx (kept to match your symbolic)
        constexpr double c = 0.7978845608028654; // sqrt(2/pi)
        return 0.5 * x * (1.0 + std::tanh(c * (x + 0.044715 * x * x * x)));
    }
    default:
        return x;
    }
}

[[gnu::always_inline]]
static inline std::shared_ptr<Expression>
unary_from_expression(Operator op, const std::shared_ptr<Expression> &x) {
    auto g = x->graph;
    auto e = std::make_shared<Expression>(g);
    e->node->type = op;
    e->node->addInput(x->node);
    if (g) [[likely]]
        g->addNode(e->node);
    return e;
}

[[gnu::always_inline]]
static inline std::shared_ptr<Expression>
unary_from_variable(Operator op, const std::shared_ptr<Variable> &v) {
    auto g = std::make_shared<ADGraph>();
    auto x = make_expr_from_variable(v, g);
    return unary_from_expression(op, x);
}

[[gnu::hot]]
static inline nb::object unary_dispatch(nb::object x, Operator op) {
    if (is_number(x)) [[likely]]
        return nb::float_(apply_unary_op(op, nb::cast<double>(x)));
    if (nb::isinstance<Expression>(x)) [[likely]]
        return nb::cast(unary_from_expression(
            op, nb::cast<std::shared_ptr<Expression>>(x)));
    if (nb::isinstance<Variable>(x))
        return nb::cast(
            unary_from_variable(op, nb::cast<std::shared_ptr<Variable>>(x)));
    throw std::invalid_argument(
        "Argument must be Expression, Variable, int, or float.");
}

// ---------- pow helpers ----------
[[gnu::always_inline]]
static inline bool _is_effectively_int(double p, double &pr_out) noexcept {
    const double pr = std::round(p);
    if (std::isfinite(pr) &&
        std::fabs(p - pr) <= 1e-12 * std::max(1.0, std::fabs(p))) {
        pr_out = pr;
        return true;
    }
    return false;
}

[[gnu::always_inline]]
static inline ADNodePtr mul_node(const ADGraphPtr &g, const ADNodePtr &a,
                                 const ADNodePtr &b) {
    auto m = std::make_shared<ADNode>();
    m->type = Operator::Multiply;
    m->addInput(a);
    m->addInput(b);
    g->addNode(m);
    return m;
}

[[gnu::hot]]
static inline ADNodePtr pow_pos_node(const ADGraphPtr &g, const ADNodePtr &base,
                                     long long e) {
    if (e == 1)
        return base;
    if (e == 2)
        return mul_node(g, base, base);
    ADNodePtr result = base, cur = base;
    e >>= 1;
    while (e > 0) {
        cur = mul_node(g, cur, cur);
        if (e & 1)
            result = mul_node(g, result, cur);
        e >>= 1;
    }
    return result;
}

[[gnu::hot]]
static inline ADNodePtr powi_node(const ADGraphPtr &g, const ADNodePtr &base,
                                  long long e) {
    if (e == 0)
        return make_const_node(g, 1.0);
    if (e > 0)
        return pow_pos_node(g, base, e);
    if (base->type == Operator::cte && base->value == 0.0)
        throw std::domain_error(
            "x**p: base == 0 and integer p < 0 is undefined.");
    auto den = pow_pos_node(g, base, -e);
    auto num = make_const_node(g, 1.0);
    auto div = std::make_shared<ADNode>();
    div->type = Operator::Divide;
    div->addInput(num);
    div->addInput(den);
    g->addNode(div);
    return div;
}

[[gnu::hot]]
static inline std::shared_ptr<Expression> expr_pow_any(nb::object x, double p) {
    ADGraphPtr g;
    std::shared_ptr<Expression> ex;
    if (nb::isinstance<Expression>(x)) {
        ex = nb::cast<std::shared_ptr<Expression>>(x);
        g = ex->graph ? ex->graph : std::make_shared<ADGraph>();
    } else {
        g = std::make_shared<ADGraph>();
        ex = as_expression(x, g);
    }
    if (ex->node)
        g->adoptSubgraph(ex->node);

    double pr = 0.0;
    if (_is_effectively_int(p, pr)) {
        auto out = std::make_shared<Expression>(g);
        out->node = powi_node(g, ex->node, (long long)pr);
        return out;
    }

    if (ex->node && ex->node->type == Operator::cte && ex->node->value <= 0.0)
        throw std::domain_error("x**p with non-integer p requires base > 0.");

    auto e_log = std::make_shared<Expression>(g);
    e_log->node->type = Operator::Log;
    e_log->node->addInput(ex->node);
    g->addNode(e_log->node);

    auto e_mul = std::make_shared<Expression>(g);
    e_mul->node->type = Operator::Multiply;
    e_mul->node->addInput(e_log->node);
    e_mul->node->addInput(make_const_node(g, p));
    g->addNode(e_mul->node);

    auto e_exp = std::make_shared<Expression>(g);
    e_exp->node->type = Operator::Exp;
    e_exp->node->addInput(e_mul->node);
    g->addNode(e_exp->node);
    return e_exp;
}

[[gnu::always_inline]]
static inline std::shared_ptr<Expression>
scalar_pow_expr(double s, const std::shared_ptr<Expression> &x) {
    if (s <= 0.0)
        throw std::domain_error("scalar ** Expression requires base > 0");
    ADGraphPtr g = x->graph ? x->graph : std::make_shared<ADGraph>();
    if (x->node)
        g->adoptSubgraph(x->node);
    auto e_mul = std::make_shared<Expression>(g);
    e_mul->node->type = Operator::Multiply;
    e_mul->node->addInput(x->node);
    e_mul->node->addInput(make_const_node(g, std::log(s)));
    g->addNode(e_mul->node);
    auto e_exp = std::make_shared<Expression>(g);
    e_exp->node->type = Operator::Exp;
    e_exp->node->addInput(e_mul->node);
    g->addNode(e_exp->node);
    return e_exp;
}

// ---------- max ----------
[[gnu::hot]]
static inline nb::object binary_max_dispatch(nb::object x, nb::object y) {
    const bool nx = is_number(x), ny = is_number(y);
    if (nx && ny) {
        const double a = nb::cast<double>(x), b = nb::cast<double>(y);
        return nb::float_(a >= b ? a : b);
    }
    ADGraphPtr g = std::make_shared<ADGraph>();
    auto ex = as_expression(x, g), ey = as_expression(y, g);
    if (ex->graph)
        g = ex->graph;
    if (ey->graph)
        g = ey->graph;
    g->adoptSubgraph(ex->node);
    g->adoptSubgraph(ey->node);
    auto out = std::make_shared<Expression>(g);
    out->node->type = Operator::Max;
    out->node->addInput(ex->node);
    out->node->addInput(ey->node);
    g->addNode(out->node);
    return nb::cast(out);
}

// ---------- Common compiler for Grad/Hess ----------
struct Compiled {
    ADGraphPtr g;
    ADNodePtr root;
    std::vector<ADNodePtr> vars;
};

template <ArgPolicy P>
[[gnu::hot]]
static inline Compiled compile_to_graph(nb::object f, size_t arity,
                                        bool vector_input) {
    Compiled out;
    out.g = std::make_shared<ADGraph>();
    std::vector<nb::object> expr_args;
    expr_args.reserve(arity);
    out.vars.reserve(arity);
    for (size_t i = 0; i < arity; ++i) {
        auto v = std::make_shared<Variable>("", 0.0);
        auto e = std::make_shared<Expression>(v, 1.0, out.g);
        out.vars.push_back(e->node);
        expr_args.emplace_back(nb::cast(e));
    }
    nb::object ret = vector_input ? call_py_fn<ArgPolicy::List>(f, expr_args)
                                  : call_py_fn<P>(f, expr_args);
    if (ret.is_none()) [[unlikely]]
        throw std::invalid_argument("compile: function returned None");
    auto expr = nb::cast<std::shared_ptr<Expression>>(ret);
    out.g->adoptSubgraph(expr->node);
    // out.g->simplifyExpression();
    out.root = expr->node;
    return out;
}

// ---------- GradFn ----------
class GradFn {
public:
    ADGraphPtr g;
    ADNodePtr expr_root;
    std::vector<ADNodePtr> var_nodes;
    bool vector_mode{};
    nb::object python_func;

    [[gnu::pure]] std::string expr_str() const {
        return (g && expr_root) ? g->getExpression(expr_root) : std::string{};
    }

    GradFn(nb::object f, size_t arity, bool vector_input)
        : vector_mode(vector_input), python_func(f) {
        auto C = compile_to_graph<ArgPolicy::Tuple>(f, arity, vector_input);
        g = C.g;
        g->simplifyGraph();
        expr_root = C.root;
        var_nodes = std::move(C.vars);
    }

    // Fast path: write (f, grad) into external buffers; no Python, no GIL.
    [[gnu::hot]]
    void value_grad_into_nogil(const double *x, std::size_t n, double *f_out,
                               double *g_out) {
        if (n != var_nodes.size())
            throw std::invalid_argument(
                "value_grad_into_nogil: wrong input length");

        // Set inputs
        for (std::size_t i = 0; i < n; ++i)
            var_nodes[i]->value = x[i];

        // Pure C++ AD; safe without GIL
        g->resetGradients();     // also bumps grad epoch
        g->resetForwardPass();   // bump value epoch
        g->computeForwardPass(); // compute primals
        const double fval = expr_root->value;

        set_epoch_value(expr_root->gradient, expr_root->grad_epoch,
                        g->cur_grad_epoch_, 1.0);
        g->initiateBackwardPass(expr_root);

        if (g_out) {
            for (std::size_t i = 0; i < n; ++i)
                g_out[i] = var_nodes[i]->gradient;
        }
        if (f_out)
            *f_out = fval;
    }

    [[gnu::hot]] Arr1D call_numpy(Arr1D x_in) {
        set_inputs_from_arr(var_nodes, x_in);
        {
            nb::gil_scoped_release nogil;
            g->resetGradients();
            g->computeForwardPass();
            set_epoch_value(expr_root->gradient, expr_root->grad_epoch,
                            g->cur_grad_epoch_, 1.0);
            g->initiateBackwardPass(expr_root);
        }
        const ssize_t n = (ssize_t)var_nodes.size();
        Arr1D out = create_zeros_1d(n);
        double *om = out.data();
        for (ssize_t i = 0; i < n; ++i)
            om[i] = var_nodes[(size_t)i]->gradient;
        return out;
    }

    [[gnu::hot]] double value_numpy(Arr1D x_in) {
        set_inputs_from_arr(var_nodes, x_in);
        double fval;
        {
            nb::gil_scoped_release nogil;
            g->resetForwardPass();
            g->computeForwardPass();
            fval = expr_root->value;
        }
        return fval;
    }

    [[gnu::hot]] double value_eigen(const Eigen::VectorXd &x) {
        if ((size_t)x.size() != var_nodes.size()) [[unlikely]]
            throw std::invalid_argument("GradFn.value: wrong input length");
        for (size_t i = 0; i < var_nodes.size(); ++i)
            var_nodes[i]->value = x[i]; // implicit cast double <- scalar

        double fval;
        {
            nb::gil_scoped_release nogil;
            g->resetForwardPass();
            g->computeForwardPass();
            fval = expr_root->value;
        }
        return fval;
    }

    [[gnu::hot]] std::pair<double, Arr1D> value_grad_numpy(Arr1D x_in) {
        auto x = as_span_1d(x_in);
        if (x.size() != var_nodes.size()) [[unlikely]]
            throw std::invalid_argument(
                "GradFn.value_grad: wrong input length");
        for (size_t i = 0; i < x.size(); ++i)
            var_nodes[i]->value = x[i];

        double fval;
        {
            nb::gil_scoped_release nogil;
            g->resetGradients();
            g->resetForwardPass();
            g->computeForwardPass();
            fval = expr_root->value;
            set_epoch_value(expr_root->gradient, expr_root->grad_epoch,
                            g->cur_grad_epoch_, 1.0);
            g->initiateBackwardPass(expr_root);
        }
        Arr1D grad = create_zeros_1d((ssize_t)var_nodes.size());
        double *gd = grad.data();
        for (size_t i = 0; i < var_nodes.size(); ++i)
            gd[i] = var_nodes[i]->gradient;
        return {fval, std::move(grad)};
    }

    [[gnu::hot]] std::pair<double, Eigen::VectorXd>
    value_grad_eigen(const Eigen::VectorXd &x_in) {
        // convert x_in to span 1d
        auto x = std::span<const double>(x_in.data(), (size_t)x_in.size());
        if ((size_t)x.size() != var_nodes.size()) [[unlikely]]
            throw std::invalid_argument(
                "GradFn.value_grad: wrong input length");
        for (size_t i = 0; i < var_nodes.size(); ++i)
            var_nodes[i]->value = x[i]; // implicit cast double <- scalar

        double fval;
        {
            nb::gil_scoped_release nogil;
            g->resetGradients();
            g->resetForwardPass();
            g->computeForwardPass();
            fval = expr_root->value;
            set_epoch_value(expr_root->gradient, expr_root->grad_epoch,
                            g->cur_grad_epoch_, 1.0);
            g->initiateBackwardPass(expr_root);
        }
        Eigen::VectorXd grad(var_nodes.size());
        for (size_t i = 0; i < var_nodes.size(); ++i)
            grad[i] = var_nodes[i]->gradient;
        return {fval, std::move(grad)};
    }

    nb::list operator()(nb::object x) {
        set_inputs_from_seq(var_nodes, x);
        {
            nb::gil_scoped_release nogil;
            g->resetGradients();
            g->computeForwardPass();
            set_epoch_value(expr_root->gradient, expr_root->grad_epoch,
                            g->cur_grad_epoch_, 1.0);
            g->initiateBackwardPass(expr_root);
        }
        nb::list out;
        for (auto &nd : var_nodes)
            out.append(nb::float_(nd->gradient));
        return out;
    }
};

// ---------- HessFn ----------
class HessFn {
public:
    ADGraphPtr g;
    ADNodePtr expr_root;
    std::vector<ADNodePtr> var_nodes;
    bool vector_mode{};
    std::pmr::vector<double> seed{&g_pool};
    nb::object python_func;

    [[gnu::pure]] std::string expr_str() const {
        return (g && expr_root) ? g->getExpression(expr_root) : std::string{};
    }

    HessFn(nb::object f, size_t arity, bool vector_input)
        : vector_mode(vector_input), python_func(f) {
        auto C = compile_to_graph<ArgPolicy::Tuple>(f, arity, vector_input);
        g = C.g;
        expr_root = C.root;
        var_nodes = std::move(C.vars);
        seed.assign(arity, 0.0);
        g->initializeNodeVariables();
    }

    void set_inputs_seq(const nb::object &x) {
        set_inputs_from_seq(var_nodes, x);
    }
    void set_inputs_arr(const Arr1D &x) { set_inputs_from_arr(var_nodes, x); }

    [[gnu::hot]]
    std::pmr::vector<double> hvp_once(std::span<const double> v) {
        const size_t n = var_nodes.size();
        if (v.size() != n) [[unlikely]]
            throw std::invalid_argument("HessFn.hvp_once: v wrong length");

        {
            nb::gil_scoped_release nogil;
            g->resetTangents();
            for (size_t i = 0; i < n; ++i)
                set_epoch_value(var_nodes[i]->dot, var_nodes[i]->dot_epoch,
                                g->cur_dot_epoch_, v[i]);
            g->resetForwardPass();
            g->computeForwardPassWithDotLanes();
            g->resetGradients();
            g->resetGradDot();
            set_epoch_value(expr_root->gradient, expr_root->grad_epoch,
                            g->cur_grad_epoch_, 1.0);
            set_epoch_value(expr_root->grad_dot, expr_root->gdot_epoch,
                            g->cur_gdot_epoch_, 0.0);
            g->initiateBackwardPassHVP();
        }

        std::pmr::vector<double> Hv(n, 0.0, &g_pool);
        for (size_t i = 0; i < n; ++i)
            Hv[i] = var_nodes[i]->grad_dot;
        return Hv;
    }

    [[gnu::hot]]
    Arr2D call_numpy(Arr1D x_in) {
        set_inputs_arr(x_in);
        const size_t n = var_nodes.size();
        {
            nb::gil_scoped_release nogil;
            g->resetForwardPass();
            g->computeForwardPass();
        }

        Arr2D H = create_zeros_2d((ssize_t)n, (ssize_t)n);
        double *data = H.data();
        for (size_t j = 0; j < n; ++j) {
            std::fill(seed.begin(), seed.end(), 0.0);
            seed[j] = 1.0;
            auto col = hvp_once(seed);
            for (size_t i = 0; i < n; ++i)
                data[i * n + j] = col[i];
        }
        return H;
    }

    nb::list operator()(nb::object x) {
        set_inputs_seq(x);
        const size_t n = var_nodes.size();
        {
            nb::gil_scoped_release nogil;
            g->resetForwardPass();
            g->computeForwardPass();
        }

        // PMR-backed matrix to limit churn
        std::pmr::vector<std::pmr::vector<double>> H(
            n, std::pmr::vector<double>(n, 0.0, &g_pool), &g_pool);

        for (size_t j = 0; j < n; ++j) {
            std::fill(seed.begin(), seed.end(), 0.0);
            seed[j] = 1.0;
            auto col = hvp_once(seed);
            for (size_t i = 0; i < n; ++i)
                H[i][j] = col[i];
        }
        nb::list mat;
        for (size_t i = 0; i < n; ++i) {
            nb::list row;
            for (size_t j = 0; j < n; ++j)
                row.append(nb::float_(H[i][j]));
            mat.append(row);
        }
        return mat;
    }

    [[gnu::hot]]
    Arr1D hvp_numpy(Arr1D x_in, Arr1D v_in) {
        set_inputs_arr(x_in);
        auto v = as_span_1d(v_in);
        if (v.size() != var_nodes.size()) [[unlikely]]
            throw std::invalid_argument("HessFn.hvp: wrong vector length");
        auto Hv = hvp_once(v);
        Arr1D out = create_zeros_1d((ssize_t)v.size());
        std::memcpy(out.data(), Hv.data(), Hv.size() * sizeof(double));
        return out;
    }

    nb::list hvp_seq(nb::object x, nb::object v) {
        set_inputs_seq(x);
        if (!is_sequence(v)) [[unlikely]]
            throw std::invalid_argument(
                "HessFn.hvp: expected list/tuple for v");
        nb::sequence sv = nb::cast<nb::sequence>(v);
        const size_t n = var_nodes.size();
        if ((size_t)nb::len(sv) != n) [[unlikely]]
            throw std::invalid_argument("HessFn.hvp: wrong vector length");

        tl_scratch.clear();
        tl_scratch.reserve(n);
        for (size_t i = 0; i < n; ++i)
            tl_scratch.push_back(nb::cast<double>(sv[i]));
        auto Hv = hvp_once(tl_scratch);
        nb::list out;
        for (size_t i = 0; i < n; ++i)
            out.append(nb::float_(Hv[i]));
        return out;
    }
};

// ---------- Lagrangian Hessian (unchanged logic; minor DRY) ----------
class LagHessFn {
public:
    using dvec = Eigen::VectorXd;
    std::vector<int> x2g_, g2x_;
    ADGraphPtr g;
    ADNodePtr L_root;
    std::vector<ADNodePtr> x_nodes, lam_nodes, nu_nodes;
    nb::object f_fun;
    std::vector<nb::object> cI_funs, cE_funs;
    bool vector_mode{};
    bool order_ready_{false};

    void build_permutations_once_() {
        if (order_ready_)
            return;
        g->initializeNodeVariables();
        const size_t n = x_nodes.size();
        x2g_.assign(n, -1);
        g2x_.assign(n, -1);
        for (size_t i = 0; i < n; ++i) {
            const int k = x_nodes[i]->order;
            if (k < 0 || (size_t)k >= n)
                throw std::runtime_error("bad order");
            x2g_[i] = k;
            g2x_[k] = (int)i;
        }
        order_ready_ = true;
    }

    LagHessFn(nb::object f, const std::vector<nb::object> &cI,
              const std::vector<nb::object> &cE, size_t arity,
              bool vector_input)
        : g(std::make_shared<ADGraph>()), f_fun(f), cI_funs(cI), cE_funs(cE),
          vector_mode(vector_input) {
        // build placeholders
        std::vector<nb::object> args;
        args.reserve(arity);
        x_nodes.reserve(arity);
        for (size_t i = 0; i < arity; ++i) {
            auto v = std::make_shared<Variable>("x" + std::to_string(i), 0.0);
            auto e = std::make_shared<Expression>(v, 1.0, g);
            x_nodes.push_back(e->node);
            args.emplace_back(nb::cast(e));
        }

        auto f_ret = vector_mode ? call_py_fn<ArgPolicy::List>(f_fun, args)
                                 : call_py_fn<ArgPolicy::Tuple>(f_fun, args);
        if (f_ret.is_none())
            throw std::invalid_argument("LagHessFn: f returned None");
        auto f_expr = nb::cast<std::shared_ptr<Expression>>(f_ret);
        g->adoptSubgraph(f_expr->node);
        ADNodePtr acc = f_expr->node;

        auto add_lin = [&](const std::vector<nb::object> &funs,
                           std::vector<ADNodePtr> &coeffs) {
            coeffs.reserve(funs.size());
            for (auto &fi : funs) {
                if (!fi.is_valid())
                    continue;
                nb::object ci_ret =
                    vector_mode ? call_py_fn<ArgPolicy::List>(fi, args)
                                : call_py_fn<ArgPolicy::Tuple>(fi, args);
                if (ci_ret.is_none())
                    throw std::invalid_argument(
                        "LagHessFn: constraint returned None");
                auto ce = nb::cast<std::shared_ptr<Expression>>(ci_ret);
                g->adoptSubgraph(ce->node);
                auto lam = std::make_shared<ADNode>();
                lam->type = Operator::cte;
                lam->value = 0.0;
                g->addNode(lam);
                coeffs.push_back(lam);

                auto mul = std::make_shared<ADNode>();
                mul->type = Operator::Multiply;
                mul->addInput(lam);
                mul->addInput(ce->node);
                g->addNode(mul);
                auto add = std::make_shared<ADNode>();
                add->type = Operator::Add;
                add->addInput(acc);
                add->addInput(mul);
                g->addNode(add);
                acc = add;
            }
        };
        add_lin(cI_funs, lam_nodes);
        add_lin(cE_funs, nu_nodes);

        L_root = acc;
        build_permutations_once_();

        g->initializeNodeVariables();
        g->simplifyGraph();

        {
            nb::gil_scoped_release nogil;
            g->resetForwardPass();
            g->computeForwardPass();
        }
    }

    void set_state_eigen(const dvec &x, const dvec &lam, const dvec &nu) {
        if ((size_t)x.size() != x_nodes.size())
            throw std::invalid_argument("set_state_eigen: x size mismatch");
        if ((size_t)lam.size() != lam_nodes.size() ||
            (size_t)nu.size() != nu_nodes.size())
            throw std::invalid_argument(
                "set_state_eigen: multiplier size mismatch");
        for (size_t i = 0; i < x_nodes.size(); ++i)
            x_nodes[i]->value = x[i];
        for (size_t i = 0; i < lam_nodes.size(); ++i)
            lam_nodes[i]->value = lam[i];
        for (size_t i = 0; i < nu_nodes.size(); ++i)
            nu_nodes[i]->value = nu[i];
        nb::gil_scoped_release nogil;
        g->resetForwardPass();
        g->computeForwardPass();
    }

    void set_state_numpy(Arr1D x_in, Arr1D lam_in, Arr1D nu_in) {
        auto span1 = [](const Arr1D &a) -> std::span<const double> {
            if (a.ndim() != 1)
                throw std::invalid_argument("expected 1D float64");
            return {a.data(), (size_t)a.shape(0)};
        };
        auto x = span1(x_in), l = span1(lam_in), n = span1(nu_in);
        dvec xe((int)x.size()), le((int)l.size()), ne((int)n.size());
        std::memcpy(xe.data(), x.data(), x.size() * sizeof(double));
        std::memcpy(le.data(), l.data(), l.size() * sizeof(double));
        std::memcpy(ne.data(), n.data(), n.size() * sizeof(double));
        set_state_eigen(xe, le, ne);
    }

    void refresh_orders_() const { g->initializeNodeVariables(); }

    std::vector<double> x_to_graph_order_(std::span<const double> v_x) const {
        const size_t n = x_nodes.size();
        std::vector<double> v_g(n);
        for (size_t i = 0; i < n; ++i)
            v_g[(size_t)x2g_[i]] = v_x[i];
        return v_g;
    }
    std::vector<double>
    graph_to_x_order_(const std::vector<double> &w_g) const {
        const size_t n = x_nodes.size();
        std::vector<double> w_x(n);
        for (size_t k = 0; k < n; ++k)
            w_x[(size_t)g2x_[k]] = w_g[k];
        return w_x;
    }

    Arr1D hvp_numpy(Arr1D x_in, Arr1D lam_in, Arr1D nu_in, Arr1D v_in) {
        set_state_numpy(x_in, lam_in, nu_in);
        build_permutations_once_();
        std::span<const double> vx{v_in.data(), (size_t)v_in.shape(0)};
        auto v_g = x_to_graph_order_(vx);
        std::vector<double> Hv_g;
        {
            nb::gil_scoped_release nogil;
            Hv_g = g->hessianVectorProduct(L_root, v_g);
        }
        auto Hv_x = graph_to_x_order_(Hv_g);
        Arr1D out = create_zeros_1d((ssize_t)Hv_x.size());
        std::memcpy(out.data(), Hv_x.data(), Hv_x.size() * sizeof(double));
        return out;
    }

    Arr2D hess_numpy(Arr1D x_in, Arr1D lam_in, Arr1D nu_in) {
        set_state_numpy(x_in, lam_in, nu_in);
        const size_t n = x_nodes.size();
        Arr2D H = create_zeros_2d((ssize_t)n, (ssize_t)n);
        double *Hd = H.data();

        build_permutations_once_();
        const size_t L = 16;
        std::vector<double> V(n * L, 0.0), Y(n * L, 0.0);

        for (size_t base = 0; base < n; base += L) {
            const size_t k = std::min(L, n - base);
            std::fill(V.begin(), V.end(), 0.0);
            for (size_t j = 0; j < k; ++j) {
                const size_t xj = base + j;
                const int gij = x2g_[xj];
                V[(size_t)gij * L + j] = 1.0;
            }
            {
                nb::gil_scoped_release nogil;
                g->hessianMultiVectorProduct(L_root, V.data(), L, Y.data(), L,
                                             k);
            }
            for (size_t j = 0; j < k; ++j) {
                const size_t col_x = base + j;
                for (size_t gi = 0; gi < n; ++gi) {
                    const size_t xi = (size_t)g2x_[gi];
                    Hd[xi * n + col_x] = Y[gi * L + j];
                }
            }
        }
        return H;
    }

    Eigen::MatrixXd hess_eigen(const Eigen::Ref<const Eigen::VectorXd> &x_in,
                               const Eigen::Ref<const Eigen::VectorXd> &lam_in,
                               const Eigen::Ref<const Eigen::VectorXd> &nu_in) {
        // If you already have a native (non-numpy) set_state(), call it here.
        // Otherwise, keep this call, or add a thin overload that forwards to
        // internal buffers.
        set_state_eigen(
            x_in, lam_in,
            nu_in); // implement this or replace with set_state_numpy wrapper

        const size_t n = x_nodes.size();
        Eigen::MatrixXd H(n, n); // column-major by default

        build_permutations_once_();

        constexpr size_t L = 16; // block size (columns per sweep)
        // Row-major scratch to match your existing "ld = L" layout and indexing
        // (gi*L + j)
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            V(n, L);
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            Y(n, L);

        for (size_t base = 0; base < n; base += L) {
            const size_t k = std::min(L, n - base);

            // Build a block of k basis vectors in graph-order (g) with
            // row-major layout
            V.setZero();
            for (size_t j = 0; j < k; ++j) {
                const size_t xj = base + j; // x-index of this column
                const int gij = x2g_[xj];   // corresponding graph index
                V(static_cast<Eigen::Index>(gij),
                  static_cast<Eigen::Index>(j)) = 1.0;
            }

            {
                // Perform k Hessian·v products at once (no GIL, native code)
                nb::gil_scoped_release nogil;
                // Arguments mirror your original:
                //   (root, V.data(), ldV=L, Y.data(), ldY=L, k_cols)
                g->hessianMultiVectorProduct(L_root, V.data(), L, Y.data(), L,
                                             k);
            }

            // Scatter back to H columns in x-order using g2x_ permutation
            for (size_t j = 0; j < k; ++j) {
                const size_t col_x = base + j;
                for (size_t gi = 0; gi < n; ++gi) {
                    const size_t xi = static_cast<size_t>(g2x_[gi]);
                    H(static_cast<Eigen::Index>(xi),
                      static_cast<Eigen::Index>(col_x)) =
                        Y(static_cast<Eigen::Index>(gi),
                          static_cast<Eigen::Index>(j));
                }
            }
        }
        return H;
    }

    Eigen::SparseMatrix<double>
    hess_sparse(const Eigen::Ref<const Eigen::VectorXd> &x_in,
                const Eigen::Ref<const Eigen::VectorXd> &lam_in,
                const Eigen::Ref<const Eigen::VectorXd> &nu_in,
                double tol = 1e-12) {
        set_state_eigen(x_in, lam_in, nu_in);

        const size_t n = x_nodes.size();
        build_permutations_once_();

        constexpr size_t L = 16; // block width
        // Row-major scratch to match (gi*L + j) addressing
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            V(n, L);
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            Y(n, L);

        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(
            std::min<size_t>(n * 16, n * n)); // rough guess; grows if needed

        for (size_t base = 0; base < n; base += L) {
            const size_t k = std::min(L, n - base);

            // Build k basis columns in graph index space
            V.setZero();
            for (size_t j = 0; j < k; ++j) {
                const size_t xj = base + j;
                const int gij = x2g_[xj];
                V(static_cast<Eigen::Index>(gij),
                  static_cast<Eigen::Index>(j)) = 1.0;
            }

            {
                nb::gil_scoped_release nogil;
                g->hessianMultiVectorProduct(L_root, V.data(), L, Y.data(), L,
                                             k);
            }

            // Scatter: we only take i <= col_x (upper triangle)
            for (size_t j = 0; j < k; ++j) {
                const size_t col_x = base + j;
                for (size_t gi = 0; gi < n; ++gi) {
                    const size_t xi = static_cast<size_t>(g2x_[gi]);
                    if (xi > col_x)
                        continue; // keep to upper triangle

                    const double val = Y(static_cast<Eigen::Index>(gi),
                                         static_cast<Eigen::Index>(j));
                    if (std::abs(val) >= tol) {
                        trips.emplace_back(static_cast<int>(xi),
                                           static_cast<int>(col_x), val);
                    }
                }
            }
        }

        // Build upper-triangular sparse
        Eigen::SparseMatrix<double> H_upper(static_cast<int>(n),
                                            static_cast<int>(n));
        H_upper.setFromTriplets(trips.begin(), trips.end());
        H_upper.makeCompressed();

        // Symmetrize: H = H_upper + H_upper.transpose() - diag(H_upper)
        // (avoid double-counting diagonal)
        Eigen::SparseMatrix<double> H = H_upper.selfadjointView<Eigen::Upper>();
        return H; // CSC by default
    }

    Arr2D hvp_multi_numpy(Arr1D x_in, Arr1D lam_in, Arr1D nu_in, Arr2D V_in) {
        set_state_numpy(x_in, lam_in, nu_in);
        auto [n, k] = shape_2d(V_in);
        if ((size_t)n != x_nodes.size())
            throw std::invalid_argument("hvp_multi: V.rows() != nvars");

        Arr2D Y = create_zeros_2d(n, k);
        double *Yd = Y.data();
        const double *Vd = V_in.data();
        build_permutations_once_();

        std::vector<double> v_x((size_t)n), v_g, Hy_g, Hy_x;
        for (ssize_t j = 0; j < k; ++j) {
            for (ssize_t i = 0; i < n; ++i)
                v_x[(size_t)i] = Vd[(size_t)i * (size_t)k + (size_t)j];
            v_g = x_to_graph_order_(
                std::span<const double>(v_x.data(), (size_t)n));
            {
                nb::gil_scoped_release nogil;
                Hy_g = g->hessianVectorProduct(L_root, v_g);
            }
            Hy_x = graph_to_x_order_(Hy_g);
            for (ssize_t i = 0; i < n; ++i)
                Yd[(size_t)i * (size_t)k + (size_t)j] = Hy_x[(size_t)i];
        }
        return Y;
    }
};

// ---------- Caches ----------
struct FnKey {
    PyObject *f;
    size_t arity;
    bool vector;
    constexpr bool operator==(const FnKey &o) const noexcept {
        return f == o.f && arity == o.arity && vector == o.vector;
    }
};
struct FnKeyHash {
    [[gnu::always_inline]] constexpr size_t
    operator()(const FnKey &k) const noexcept {
        size_t h = std::bit_cast<size_t>(k.f);
        h ^= h >> 30;
        h *= 0xbf58476d1ce4e5b9ULL;
        h ^= h >> 27;
        h *= 0x94d049bb133111ebULL;
        h ^= h >> 31;
        h ^= k.arity + 0x9e3779b97f4a7c15ULL;
        h ^= (size_t)k.vector << 1;
        return h;
    }
};

thread_local tsl::robin_map<FnKey, std::shared_ptr<GradFn>, FnKeyHash>
    tl_grad_cache;
thread_local tsl::robin_map<FnKey, std::shared_ptr<HessFn>, FnKeyHash>
    tl_hess_cache;
static std::shared_mutex g_cache_mtx;
static tsl::robin_map<FnKey, std::weak_ptr<GradFn>, FnKeyHash> g_grad_cache;
static tsl::robin_map<FnKey, std::weak_ptr<HessFn>, FnKeyHash> g_hess_cache;

template <class T, class TLMap, class GlobalMap, class Maker>
[[gnu::hot]]
static inline std::shared_ptr<T>
cache_get_or_make(TLMap &tl_map, GlobalMap &g_map, const FnKey &k,
                  Maker &&make) {
    if (auto it = tl_map.find(k); it != tl_map.end()) [[likely]]
        return it->second;
    {
        std::shared_lock rlk(g_cache_mtx);
        if (auto it = g_map.find(k); it != g_map.end())
            if (auto sp = it->second.lock()) {
                tl_map[k] = sp;
                return sp;
            }
    }
    std::unique_lock wlk(g_cache_mtx);
    if (auto it = g_map.find(k); it != g_map.end())
        if (auto sp = it->second.lock()) {
            tl_map[k] = sp;
            return sp;
        }
    auto sp = make();
    g_map[k] = sp;
    tl_map[k] = sp;
    return sp;
}

[[gnu::always_inline]]
static inline std::shared_ptr<GradFn> get_or_make_grad(nb::object f, size_t n,
                                                       bool vec) {
    FnKey k{f.ptr(), n, vec};
    return cache_get_or_make<GradFn>(tl_grad_cache, g_grad_cache, k, [&] {
        return std::make_shared<GradFn>(f, n, vec);
    });
}

[[gnu::always_inline]]
static inline std::shared_ptr<HessFn> get_or_make_hess(nb::object f, size_t n,
                                                       bool vec) {
    FnKey k{f.ptr(), n, vec};
    return cache_get_or_make<HessFn>(tl_hess_cache, g_hess_cache, k, [&] {
        return std::make_shared<HessFn>(f, n, vec);
    });
}

// ---------- Immediate / high-level APIs ----------
[[gnu::hot]]
static nb::list py_gradient(nb::object f, nb::args xs) {
    auto g = std::make_shared<ADGraph>();

    auto push_expr_from_item = [&](nb::handle item, size_t idx,
                                   std::vector<nb::object> &dst,
                                   std::vector<std::string> &names) {
        if (nb::isinstance<Variable>(item)) {
            auto v = nb::cast<std::shared_ptr<Variable>>(item);
            names[idx] = v->getName();
            dst.emplace_back(nb::cast(make_expr_from_variable(v, g)));
        } else if (is_number(item)) {
            const double val = nb::cast<double>(item);
            auto vx =
                std::make_shared<Variable>("x" + std::to_string(idx), val);
            names[idx] = vx->getName();
            dst.emplace_back(nb::cast(make_expr_from_variable(vx, g)));
        } else if (nb::isinstance<Expression>(item)) {
            auto e = nb::cast<std::shared_ptr<Expression>>(item);
            g->adoptSubgraph(e->node);
            dst.emplace_back(nb::cast(e));
        } else {
            throw std::invalid_argument(
                "gradient: args must be Variable / Expression / number.");
        }
    };

    std::vector<nb::object> expr_args;
    std::vector<std::string> var_names;

    if (nb::len(xs) == 1 && is_sequence(xs[0])) [[likely]] {
        nb::sequence seq = nb::borrow<nb::sequence>(xs[0]);
        const size_t n = nb::len(seq);
        expr_args.reserve(n);
        var_names.resize(n);
        for (size_t i = 0; i < n; ++i)
            push_expr_from_item(seq[i], i, expr_args, var_names);
        auto ret = call_py_fn<ArgPolicy::List>(f, expr_args);
        auto expr = ensure_expression(ret, g);
        tsl::robin_map<std::string, double> grad_map;
        {
            nb::gil_scoped_release nogil;
            grad_map = expr->computeGradient();
        }
        nb::list out;
        for (size_t i = 0; i < n; ++i) {
            double gi = 0.0;
            if (!var_names[i].empty())
                if (auto it = grad_map.find(var_names[i]); it != grad_map.end())
                    gi = it->second;
            out.append(nb::float_(gi));
        }
        return out;
    }

    // positional case
    const size_t n = nb::len(xs);
    expr_args.reserve(n);
    var_names.resize(n);
    for (size_t i = 0; i < n; ++i)
        push_expr_from_item(xs[i], i, expr_args, var_names);

    auto ret = call_py_fn<ArgPolicy::Tuple>(f, expr_args);
    auto expr = ensure_expression(ret, g);
    tsl::robin_map<std::string, double> grad_map;
    {
        nb::gil_scoped_release nogil;
        grad_map = expr->computeGradient();
    }

    nb::list out;
    for (size_t i = 0; i < n; ++i) {
        double gi = 0.0;
        if (!var_names[i].empty())
            if (auto it = grad_map.find(var_names[i]); it != grad_map.end())
                gi = it->second;
        out.append(nb::float_(gi));
    }
    return out;
}

// fused value+grad (cached)
[[gnu::hot]]
static std::pair<double, Arr1D> py_value_grad_numpy(nb::object f, Arr1D x_in) {
    auto x = as_span_1d(x_in);
    auto gf = get_or_make_grad(f, x.size(), /*vector_input=*/true);
    const double fval = gf->value_numpy(x_in);
    auto grad = gf->call_numpy(x_in);
    return {fval, std::move(grad)};
}

[[gnu::hot]]
static Arr1D py_gradient_numpy(nb::object f, Arr1D x_in) {
    auto x = as_span_1d(x_in);
    auto gf = get_or_make_grad(f, x.size(), /*vector_input=*/true);
    return gf->call_numpy(x_in);
}

[[gnu::hot]]
static Arr2D py_hessian_numpy(nb::object f, Arr1D x_in) {
    auto x = as_span_1d(x_in);
    auto hf = get_or_make_hess(f, x.size(), /*vector_input=*/true);
    return hf->call_numpy(x_in);
}

// ---------- Batch fused value+grad over many GradFn ----------

// Overload 1: list of compiled GradFn
[[gnu::hot]]
static std::pair<Arr1D, Arr2D>
batch_value_grad_from_gradfns(const std::vector<std::shared_ptr<GradFn>> &gfs,
                              Arr1D x_in) {
    const auto x = as_span_1d(x_in);
    const ssize_t m = (ssize_t)gfs.size();
    if (m == 0) {
        return {create_zeros_1d(0), create_zeros_2d(0, (ssize_t)x.size())};
    }
    // sanity: all arities must match x.size()
    for (const auto &gf : gfs) {
        if (!gf)
            throw std::invalid_argument("batch_valgrad: null GradFn");
        if (gf->var_nodes.size() != (size_t)x.size())
            throw std::invalid_argument("batch_valgrad: inconsistent arity");
    }

    // outputs
    Arr1D vals = create_zeros_1d(m);
    Arr2D J = create_zeros_2d(m, (ssize_t)x.size());
    double *vd = vals.data();
    double *Jd = J.data();
    const ssize_t n = (ssize_t)x.size();

    {
        nb::gil_scoped_release nogil;

        // We still need one forward+reverse per function (distinct graphs),
        // but we keep everything in C++ and reuse the same x buffer.
        for (ssize_t j = 0; j < m; ++j) {
            auto &gf = gfs[(size_t)j];

            // set inputs
            for (ssize_t i = 0; i < n; ++i)
                gf->var_nodes[(size_t)i]->value = x[(size_t)i];

            // fused value+grad
            gf->g->resetGradients();
            gf->g->resetForwardPass();
            gf->g->computeForwardPass();
            vd[j] = gf->expr_root->value;

            set_epoch_value(gf->expr_root->gradient, gf->expr_root->grad_epoch,
                            gf->g->cur_grad_epoch_, 1.0);
            gf->g->initiateBackwardPass(gf->expr_root);

            // copy row j
            for (ssize_t i = 0; i < n; ++i)
                Jd[(size_t)j * (size_t)n + (size_t)i] =
                    gf->var_nodes[(size_t)i]->gradient;
        }
    }

    return {std::move(vals), std::move(J)};
}
using dvec = Eigen::VectorXd;
using dmat = Eigen::MatrixXd;

static inline std::pair<dvec, dmat> batch_value_grad_from_gradfns_eigen(
    const std::vector<std::shared_ptr<GradFn>> &gfs, const dvec &x) {
    const Eigen::Index m = static_cast<Eigen::Index>(gfs.size());
    const Eigen::Index n = x.size();

    if (m == 0) {
        return {dvec(0), dmat(0, n)}; // zero-row Jacobian with n cols
    }

    // sanity: all arities must match x.size()
    for (const auto &gf : gfs) {
        if (!gf)
            throw std::invalid_argument("batch_valgrad: null GradFn");
        if (static_cast<Eigen::Index>(gf->var_nodes.size()) != n)
            throw std::invalid_argument("batch_valgrad: inconsistent arity");
    }

    dvec vals(m);
    dmat J(m, n);

    {
        nb::gil_scoped_release nogil;

        for (Eigen::Index j = 0; j < m; ++j) {
            auto &gf = gfs[static_cast<size_t>(j)];

            // set inputs
            for (Eigen::Index i = 0; i < n; ++i)
                gf->var_nodes[static_cast<size_t>(i)]->value = x[i];

            // fused value+grad
            gf->g->resetGradients();
            gf->g->resetForwardPass();
            gf->g->computeForwardPass();
            vals[j] = gf->expr_root->value;

            set_epoch_value(gf->expr_root->gradient, gf->expr_root->grad_epoch,
                            gf->g->cur_grad_epoch_, 1.0);
            gf->g->initiateBackwardPass(gf->expr_root);

            // copy gradient row j
            for (Eigen::Index i = 0; i < n; ++i)
                J(j, i) = gf->var_nodes[static_cast<size_t>(i)]->gradient;
        }
    }

    return {std::move(vals), std::move(J)};
}

// Overload 2: list of Python callables (we compile/cache GradFn on the fly)
[[gnu::hot]]
static std::pair<Arr1D, Arr2D> batch_value_grad_from_callables(nb::list funcs,
                                                               Arr1D x_in) {
    const auto x = as_span_1d(x_in);
    const ssize_t m = nb::len(funcs);
    std::vector<std::shared_ptr<GradFn>> gfs;
    gfs.reserve((size_t)m);
    for (ssize_t j = 0; j < m; ++j) {
        nb::object f = funcs[(size_t)j];
        // cached compiler keyed by (f, n, vector=True)
        gfs.emplace_back(get_or_make_grad(f, (size_t)x.size(), /*vec=*/true));
    }
    return batch_value_grad_from_gradfns(gfs, x_in);
}

[[gnu::hot]]
static std::pair<Arr1D, Arr2D> batch_value_grad_numpy(nb::sequence grads,
                                                      Arr1D x_in) {
    const std::size_t m = static_cast<std::size_t>(nb::len(grads));
    auto x = as_span_1d(x_in);
    const std::size_t n = x.size();

    // Collect raw pointers to compiled GradFn to avoid Python in parallel loop
    std::vector<GradFn *> gptrs;
    gptrs.reserve(m);
    for (std::size_t j = 0; j < m; ++j) {
        nb::handle h = grads[j];
        if (!nb::isinstance<GradFn>(h))
            throw std::invalid_argument(
                "batch_valgrad: grads must be GradFn objects");
        gptrs.push_back(nb::cast<std::shared_ptr<GradFn>>(h).get());
    }

    // Allocate outputs (C-contiguous row-major)
    Arr1D vals = create_zeros_1d(static_cast<ssize_t>(m));
    Arr2D J = create_zeros_2d(static_cast<ssize_t>(m), static_cast<ssize_t>(n));
    double *vals_d = vals.data();
    double *J_d = J.data(); // row j starts at J_d + j*n

    // Parallelize over rows (each GradFn has its own ADGraph -> thread-safe)
    {
        nb::gil_scoped_release nogil;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t j = 0; j < static_cast<std::ptrdiff_t>(m); ++j) {
            GradFn *gf = gptrs[static_cast<std::size_t>(j)];
            double *rowj = J_d + static_cast<std::size_t>(j) * n;
            double fj = 0.0;
            gf->value_grad_into_nogil(x.data(), n, &fj, rowj);
            vals_d[j] = fj;
        }
    }

    return {std::move(vals), std::move(J)};
}