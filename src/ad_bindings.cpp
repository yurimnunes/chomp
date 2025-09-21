// ad.cpp — nanobind + NumPy fast paths + GIL release (C++23 optimized)

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/shared_ptr.h>

#include "../include/ad/ADGraph.h"
#include "../include/ad/Definitions.h"
#include "../include/ad/Expression.h"
#include "../include/ad/Variable.h"

#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <expected>
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
#include <functional>

namespace nb = nanobind;
using namespace nb::literals;

// ---------- Memory pool for hot allocations ----------
static std::pmr::unsynchronized_pool_resource g_pool{
    {.max_blocks_per_chunk = 256, .largest_required_pool_block = 8192}
};
thread_local std::pmr::vector<double> tl_scratch{&g_pool};

// ---------- Compile-time type checks ----------
template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

template<typename T>
concept PyHandle = std::same_as<T, nb::handle> || std::same_as<T, nb::object>;

// ---------- Hot path helpers (force inline + likely/unlikely) ----------
[[gnu::always_inline, gnu::hot]]
static inline bool is_number(const nb::handle &h) noexcept {
    return nb::isinstance<nb::float_>(h) || nb::isinstance<nb::int_>(h);
}

[[gnu::always_inline, gnu::hot]]
static inline bool is_sequence(const nb::handle &h) noexcept {
    // Accept list/tuple as "sequence"; explicitly exclude str
    return nb::isinstance<nb::list>(h) || nb::isinstance<nb::tuple>(h);
}

[[gnu::always_inline]]
static inline nb::tuple to_tuple(const std::vector<nb::object> &vec) {
    nb::list temp;
    for (const auto& item : vec) {
        temp.append(item);
    }
    return nb::tuple(temp);
}

// Optimized epoch bumping with memory ordering
template<Numeric T>
[[gnu::always_inline]] 
static inline void bump_epoch(T &e) noexcept { 
    ++e; 
}

template<Numeric T>
[[gnu::always_inline]]
static inline void bump_epoch(std::atomic<T> &e) noexcept {
    e.fetch_add(1, std::memory_order_acq_rel);
}

// ---------- NumPy ndarray aliases (C-contiguous float64) ----------
using Arr1D = nb::ndarray<double, nb::shape<-1>,     nb::c_contig>;
using Arr2D = nb::ndarray<double, nb::shape<-1, -1>, nb::c_contig>;

[[gnu::always_inline, gnu::hot]]
static inline std::span<const double> as_span_1d(const Arr1D &a) {
    if (a.ndim() != 1) [[unlikely]]
        throw std::invalid_argument("expected 1D float64 array");
    return {a.data(), static_cast<size_t>(a.shape(0))};
}

[[gnu::always_inline]]
static inline std::pair<ssize_t, ssize_t> shape_2d(const Arr2D &a) {
    if (a.ndim() != 2) [[unlikely]]
        throw std::invalid_argument("expected 2D float64 array");
    return {a.shape(0), a.shape(1)};
}

// Helper to create numpy arrays
[[gnu::always_inline]]
static inline Arr1D create_zeros_1d(ssize_t n) {
    auto numpy = nb::module_::import_("numpy");
    return nb::cast<Arr1D>(numpy.attr("zeros")(nb::make_tuple(n), "dtype"_a="float64"));
}

[[gnu::always_inline]]
static inline Arr2D create_zeros_2d(ssize_t m, ssize_t n) {
    auto numpy = nb::module_::import_("numpy");
    return nb::cast<Arr2D>(numpy.attr("zeros")(nb::make_tuple(m, n), "dtype"_a="float64"));
}

// ---------- Expression construction (optimized allocations) ----------
[[gnu::always_inline]]
static inline ADNodePtr make_const_node(const ADGraphPtr &g, double v) noexcept {
    auto n = std::make_shared<ADNode>();
    n->type = Operator::cte;
    n->value = v;
    if (g) [[likely]] g->addNode(n);
    return n;
}

[[gnu::always_inline]]
static inline std::shared_ptr<Expression> make_const_expr(double val, const ADGraphPtr &g) {
    auto n = make_const_node(g, val);
    return std::make_shared<Expression>(n, g);
}

[[gnu::always_inline]]
static inline std::shared_ptr<Expression>
make_expr_from_variable(const std::shared_ptr<Variable> &v, const ADGraphPtr &g) {
    return std::make_shared<Expression>(v, 1.0, g);
}

[[gnu::always_inline]]
static inline std::shared_ptr<Expression>
make_expr_from_number(double val, const ADGraphPtr &g) {
    return make_const_expr(val, g);
}

// Optimized conversion with early returns
[[gnu::hot]]
static inline std::shared_ptr<Expression>
as_expression(const nb::handle &h, const ADGraphPtr &g) {
    if (nb::isinstance<Expression>(h)) [[likely]]
        return nb::cast<std::shared_ptr<Expression>>(h);
    
    if (nb::isinstance<Variable>(h)) [[likely]] {
        auto v = nb::cast<std::shared_ptr<Variable>>(h);
        return make_expr_from_variable(v, g);
    }
    
    if (is_number(h)) [[likely]]
        return make_expr_from_number(nb::cast<double>(h), g);
    
    throw std::invalid_argument("Argument must be Expression, Variable, int, or float.");
}

[[gnu::hot]]
static inline std::shared_ptr<Expression>
ensure_expression(const nb::handle &ret, const ADGraphPtr &g) {
    if (nb::isinstance<Expression>(ret)) [[likely]]
        return nb::cast<std::shared_ptr<Expression>>(ret);
    
    if (is_number(ret)) [[likely]]
        return make_expr_from_number(nb::cast<double>(ret), g);
    
    throw std::invalid_argument("Function must return Expression or a numeric value.");
}

// ---------- Optimized unary ops with fast dispatch ----------
[[gnu::always_inline]]
static inline double apply_unary_op(Operator op, double x) noexcept {
    switch (op) {
        case Operator::Sin:  return std::sin(x);
        case Operator::Cos:  return std::cos(x);
        case Operator::Tan:  return std::tan(x);
        case Operator::Exp:  return std::exp(x);
        case Operator::Log:  return std::log(x);
        case Operator::Tanh: return std::tanh(x);
        case Operator::Relu: return x > 0.0 ? x : 0.0;
        case Operator::Silu: return x / (1.0 + std::exp(-x));
        case Operator::Gelu: {
            constexpr double c = 0.7978845608028654;  // sqrt(2/π)
            return 0.5 * x * (1.0 + std::tanh(c * (x + 0.044715 * x * x * x)));
        }
        default: return x;  // Identity fallback
    }
}

[[gnu::always_inline]]
static inline std::shared_ptr<Expression>
unary_from_expression(Operator op, const std::shared_ptr<Expression> &x) {
    auto g = x->graph;
    auto e = std::make_shared<Expression>(g);
    e->node->type = op;
    e->node->addInput(x->node);
    if (g) [[likely]] g->addNode(e->node);
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
    // Fast path for numbers with optimized switch
    if (is_number(x)) [[likely]] {
        const double a = nb::cast<double>(x);
        return nb::float_(apply_unary_op(op, a));
    }
    
    if (nb::isinstance<Expression>(x)) [[likely]]
        return nb::cast(unary_from_expression(op, nb::cast<std::shared_ptr<Expression>>(x)));
    
    if (nb::isinstance<Variable>(x))
        return nb::cast(unary_from_variable(op, nb::cast<std::shared_ptr<Variable>>(x)));
    
    throw std::invalid_argument("Argument must be Expression, Variable, int, or float.");
}

// ---------- Optimized pow helpers with bit manipulation ----------
[[gnu::always_inline]]
static inline bool _is_effectively_int(double p, double &pr_out) noexcept {
    const double pr = std::round(p);
    if (std::isfinite(pr) && std::fabs(p - pr) <= 1e-12 * std::max(1.0, std::fabs(p))) [[likely]] {
        pr_out = pr; 
        return true;
    }
    return false;
}

[[gnu::always_inline]]
static inline ADNodePtr mul_node(const ADGraphPtr &g, const ADNodePtr &a, const ADNodePtr &b) {
    auto m = std::make_shared<ADNode>();
    m->type = Operator::Multiply;
    m->addInput(a);
    m->addInput(b);
    g->addNode(m);
    return m;
}

// Optimized exponentiation using binary exponentiation
[[gnu::hot]]
static inline ADNodePtr pow_pos_node(const ADGraphPtr &g, const ADNodePtr &base, long long e) {
    if (e == 1) [[likely]] return base;
    if (e == 2) [[likely]] return mul_node(g, base, base);
    
    ADNodePtr result = base;
    ADNodePtr cur = base;
    
    // Skip the first bit since we already have base^1
    e >>= 1;
    
    while (e > 0) {
        cur = mul_node(g, cur, cur);  // Square
        if (e & 1) [[unlikely]]
            result = mul_node(g, result, cur);
        e >>= 1;
    }
    
    return result;
}

[[gnu::hot]]
static inline ADNodePtr powi_node(const ADGraphPtr &g, const ADNodePtr &base, long long e) {
    if (e == 0) [[unlikely]] return make_const_node(g, 1.0);
    if (e > 0) [[likely]] return pow_pos_node(g, base, e);
    
    // e < 0: Check for division by zero at compile time if possible
    if (base->type == Operator::cte && base->value == 0.0) [[unlikely]]
        throw std::domain_error("x**p: base == 0 and integer p < 0 is undefined.");
    
    auto den = pow_pos_node(g, base, -e);
    auto num = make_const_node(g, 1.0);
    auto div = std::make_shared<ADNode>();
    div->type = Operator::Divide;
    div->addInput(num);
    div->addInput(den);
    g->addNode(div);
    return div;
}

// Optimized pow with precomputed constants
[[gnu::hot]]
static inline std::shared_ptr<Expression> expr_pow_any(nb::object x, double p) {
    ADGraphPtr g;
    std::shared_ptr<Expression> ex;

    if (nb::isinstance<Expression>(x)) [[likely]] {
        ex = nb::cast<std::shared_ptr<Expression>>(x);
        g = ex->graph ? ex->graph : std::make_shared<ADGraph>();
    } else {
        g = std::make_shared<ADGraph>();
        ex = as_expression(x, g);
    }
    
    if (ex->node) [[likely]] g->adoptSubgraph(ex->node);

    double pr = 0.0;
    if (_is_effectively_int(p, pr)) [[likely]] {
        auto out = std::make_shared<Expression>(g);
        out->node = powi_node(g, ex->node, static_cast<long long>(pr));
        return out;
    }

    // Non-integer: exp(p*log(x)) with domain check
    if (ex->node && ex->node->type == Operator::cte && ex->node->value <= 0.0) [[unlikely]]
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
    if (s <= 0.0) [[unlikely]] 
        throw std::domain_error("scalar ** Expression requires base > 0");
    
    ADGraphPtr g = x->graph ? x->graph : std::make_shared<ADGraph>();
    if (x->node) [[likely]] g->adoptSubgraph(x->node);

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

// ---------- max dispatch ----------
[[gnu::hot]]
static inline nb::object binary_max_dispatch(nb::object x, nb::object y) {
    const bool nx = is_number(x), ny = is_number(y);
    if (nx && ny) [[likely]] {
        const double a = nb::cast<double>(x), b = nb::cast<double>(y);
        return nb::float_(a >= b ? a : b);
    }
    
    ADGraphPtr g = std::make_shared<ADGraph>();
    auto ex = as_expression(x, g);
    auto ey = as_expression(y, g);
    if (ex->graph) g = ex->graph;
    if (ey->graph) g = ey->graph;
    g->adoptSubgraph(ex->node);
    g->adoptSubgraph(ey->node);

    auto out = std::make_shared<Expression>(g);
    out->node->type = Operator::Max;
    out->node->addInput(ex->node);
    out->node->addInput(ey->node);
    g->addNode(out->node);
    return nb::cast(out);
}

// ---------- Function call helpers ----------
template<typename Callable>
static inline nb::object call_python_function(Callable&& f, const std::vector<nb::object>& args) {
    nb::tuple args_tuple = to_tuple(args);
    return f(*args_tuple);
}

template<typename Callable>
static inline nb::object call_python_function_with_list(Callable&& f, const std::vector<nb::object>& args) {
    nb::list args_list;
    for (const auto& arg : args) {
        args_list.append(arg);
    }
    return f(args_list);
}

// ============================================================================
//          Compiled Value / Gradient / Hessian (NumPy fast paths)
// ============================================================================
class ValFn {
public:
    ADGraphPtr g;
    ADNodePtr expr_root;
    std::vector<ADNodePtr> var_nodes;
    bool vector_mode;
    nb::object python_func;

    [[gnu::pure]]
    std::string expr_str() const {
        return (g && expr_root) ? g->getExpression(expr_root) : std::string{};
    }

    ValFn(nb::object f, size_t arity, bool vector_input)
        : g(std::make_shared<ADGraph>()), vector_mode(vector_input), python_func(f) {
        std::vector<nb::object> expr_args;
        expr_args.reserve(arity);
        var_nodes.reserve(arity);
        
        for (size_t i = 0; i < arity; ++i) {
            auto vx = std::make_shared<Variable>("", 0.0);
            auto ex = std::make_shared<Expression>(vx, 1.0, g);
            var_nodes.push_back(ex->node);
            expr_args.emplace_back(nb::cast(ex));
        }
        
        nb::object ret = vector_mode ? 
            call_python_function_with_list(f, expr_args) :
            call_python_function(f, expr_args);
            
        if (ret.is_none()) [[unlikely]]
            throw std::invalid_argument("compile_value: function returned None");
        
        auto expr = nb::cast<std::shared_ptr<Expression>>(ret);
        g->adoptSubgraph(expr->node);
        expr_root = expr->node;
        g->initializeNodeVariables();
    }

    [[gnu::hot]]
    double operator()(Arr1D x_in) {
        auto x = as_span_1d(x_in);
        if (x.size() != var_nodes.size()) [[unlikely]]
            throw std::invalid_argument("ValFn: wrong input length");
            
        for (size_t i = 0; i < x.size(); ++i) 
            var_nodes[i]->value = x[i];
            
        double fval;
        {
            nb::gil_scoped_release nogil;
            g->resetForwardPass();
            g->computeForwardPass();
            fval = expr_root->value;
        }
        return fval;
    }

    double call_seq(nb::object xseq) {
        if (!is_sequence(xseq)) [[unlikely]]
            throw std::invalid_argument("ValFn: expected list/tuple");
            
        nb::sequence s = nb::cast<nb::sequence>(xseq);
        if (static_cast<size_t>(nb::len(s)) != var_nodes.size()) [[unlikely]]
            throw std::invalid_argument("ValFn: wrong input length");
            
        for (size_t i = 0; i < var_nodes.size(); ++i)
            var_nodes[i]->value = nb::cast<double>(s[i]);
            
        double fval;
        {
            nb::gil_scoped_release nogil;
            g->resetForwardPass();
            g->computeForwardPass();
            fval = expr_root->value;
        }
        return fval;
    }
};

class GradFn {
public:
    ADGraphPtr g;
    ADNodePtr expr_root;
    std::vector<ADNodePtr> var_nodes;
    bool vector_mode;
    nb::object python_func;

    [[gnu::pure]]
    std::string expr_str() const {
        return (g && expr_root) ? g->getExpression(expr_root) : std::string{};
    }

    GradFn(nb::object f, size_t arity, bool vector_input)
        : g(std::make_shared<ADGraph>()), vector_mode(vector_input), python_func(f) {
        std::vector<nb::object> expr_args;
        expr_args.reserve(arity);
        var_nodes.reserve(arity);

        for (size_t i = 0; i < arity; ++i) {
            auto vx = std::make_shared<Variable>("", 0.0);
            auto ex = std::make_shared<Expression>(vx, 1.0, g);
            var_nodes.push_back(ex->node);
            expr_args.emplace_back(nb::cast(ex));
        }
        nb::object ret = vector_mode ?
            call_python_function_with_list(f, expr_args) :
            call_python_function(f, expr_args);
            
        if (ret.is_none()) [[unlikely]]
            throw std::invalid_argument("compile_gradient: function returned None");

        auto expr = nb::cast<std::shared_ptr<Expression>>(ret);
        g->adoptSubgraph(expr->node);
        expr_root = expr->node;
    }

    [[gnu::hot]]
    Arr1D call_numpy(Arr1D x_in) {
        auto x = as_span_1d(x_in);
        const ssize_t n = static_cast<ssize_t>(x.size());
        if (static_cast<size_t>(n) != var_nodes.size()) [[unlikely]]
            throw std::invalid_argument("GradFn: wrong input length");
            
        for (ssize_t i = 0; i < n; ++i) 
            var_nodes[static_cast<size_t>(i)]->value = x[static_cast<size_t>(i)];

        {
            nb::gil_scoped_release nogil;
            g->resetGradients();
            g->computeForwardPass();
            set_epoch_value(expr_root->gradient, expr_root->grad_epoch, g->cur_grad_epoch_, 1.0);
            g->initiateBackwardPass(expr_root);
        }

        Arr1D out = create_zeros_1d(n);
        double* om = out.data();
        for (ssize_t i = 0; i < n; ++i) 
            om[i] = var_nodes[static_cast<size_t>(i)]->gradient;
        return out;
    }

    [[gnu::hot]]
    double value_numpy(Arr1D x_in) {
        auto x = as_span_1d(x_in);
        if (x.size() != var_nodes.size()) [[unlikely]]
            throw std::invalid_argument("GradFn.value: wrong input length");
            
        for (size_t i = 0; i < x.size(); ++i) 
            var_nodes[i]->value = x[i];
            
        double fval;
        {
            nb::gil_scoped_release nogil;
            g->resetForwardPass();
            g->computeForwardPass();
            fval = expr_root->value;
        }
        return fval;
    }

    nb::list operator()(nb::object x) {
        if (!is_sequence(x)) [[unlikely]]
            throw std::invalid_argument("GradFn: expected a list/tuple of numbers");
            
        nb::sequence seq = nb::cast<nb::sequence>(x);
        if (static_cast<size_t>(nb::len(seq)) != var_nodes.size()) [[unlikely]]
            throw std::invalid_argument("GradFn: wrong length");
            
        for (size_t i = 0; i < var_nodes.size(); ++i)
            var_nodes[i]->value = nb::cast<double>(seq[i]);
            
        {
            nb::gil_scoped_release nogil;
            g->resetGradients();
            g->computeForwardPass();
            set_epoch_value(expr_root->gradient, expr_root->grad_epoch, g->cur_grad_epoch_, 1.0);
            g->initiateBackwardPass(expr_root);
        }
        
        nb::list out;
        for (size_t i = 0; i < var_nodes.size(); ++i)
            out.append(nb::float_(var_nodes[i]->gradient));
        return out;
    }
};

class HessFn {
public:
    ADGraphPtr g;
    ADNodePtr expr_root;
    std::vector<ADNodePtr> var_nodes;
    bool vector_mode;
    std::pmr::vector<double> seed;
    nb::object python_func;

    [[gnu::pure]]
    std::string expr_str() const {
        return (g && expr_root) ? g->getExpression(expr_root) : std::string{};
    }

    HessFn(nb::object f, size_t arity, bool vector_input)
        : g(std::make_shared<ADGraph>()), vector_mode(vector_input), seed(arity, 0.0, &g_pool), python_func(f) {
        std::vector<nb::object> expr_args;
        expr_args.reserve(arity);
        var_nodes.reserve(arity);
        
        for (size_t i = 0; i < arity; ++i) {
            auto vx = std::make_shared<Variable>("", 0.0);
            auto ex = std::make_shared<Expression>(vx, 1.0, g);
            var_nodes.push_back(ex->node);
            expr_args.emplace_back(nb::cast(ex));
        }
        
        nb::object ret = vector_mode ?
            call_python_function_with_list(f, expr_args) :
            call_python_function(f, expr_args);
            
        if (ret.is_none()) [[unlikely]]
            throw std::invalid_argument("compile_hessian: function returned None");
            
        auto expr = nb::cast<std::shared_ptr<Expression>>(ret);
        g->adoptSubgraph(expr->node);
        expr_root = expr->node;
        g->initializeNodeVariables();
    }

    void set_inputs_seq(const nb::object &x) {
        if (!is_sequence(x)) [[unlikely]]
            throw std::invalid_argument("HessFn: expected a list/tuple");
            
        nb::sequence sx = nb::cast<nb::sequence>(x);
        if (static_cast<size_t>(nb::len(sx)) != var_nodes.size()) [[unlikely]]
            throw std::invalid_argument("HessFn: wrong input length");
            
        for (size_t i = 0; i < var_nodes.size(); ++i)
            var_nodes[i]->value = nb::cast<double>(sx[i]);
    }
    
    void set_inputs_arr(const Arr1D &x_in) {
        auto x = as_span_1d(x_in);
        if (x.size() != var_nodes.size()) [[unlikely]]
            throw std::invalid_argument("HessFn: wrong input length");
            
        for (size_t i = 0; i < var_nodes.size(); ++i) 
            var_nodes[i]->value = x[i];
    }

    [[gnu::hot]]
    std::pmr::vector<double> hvp_once(const std::span<const double> v) {
        const size_t n = var_nodes.size();
        if (v.size() != n) [[unlikely]] 
            throw std::invalid_argument("HessFn.hvp_once: v wrong length");

        {
            nb::gil_scoped_release nogil;
            g->resetTangents();
            for (size_t i = 0; i < n; ++i) {
                auto &nd = var_nodes[i];
                set_epoch_value(nd->dot, nd->dot_epoch, g->cur_dot_epoch_, v[i]);
            }
            g->resetForwardPass();
            g->computeForwardPassWithDot();
            g->resetGradients();
            g->resetGradDot();
            set_epoch_value(expr_root->gradient, expr_root->grad_epoch, g->cur_grad_epoch_, 1.0);
            set_epoch_value(expr_root->grad_dot, expr_root->gdot_epoch,   g->cur_gdot_epoch_, 0.0);
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

        Arr2D H = create_zeros_2d(static_cast<ssize_t>(n), static_cast<ssize_t>(n));
        double* data = H.data();
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

        std::pmr::vector<std::pmr::vector<double>> Hm(n, std::pmr::vector<double>(n, 0.0, &g_pool), &g_pool);
        
        for (size_t j = 0; j < n; ++j) {
            std::fill(seed.begin(), seed.end(), 0.0);
            seed[j] = 1.0;
            auto col = hvp_once(seed);
            for (size_t i = 0; i < n; ++i) 
                Hm[i][j] = col[i];
        }
        
        nb::list mat;
        for (size_t i = 0; i < n; ++i) {
            nb::list row;
            for (size_t j = 0; j < n; ++j) 
                row.append(nb::float_(Hm[i][j]));
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
        Arr1D out = create_zeros_1d(static_cast<ssize_t>(v.size()));
        double* om = out.data();
        for (ssize_t i = 0; i < static_cast<ssize_t>(v.size()); ++i) 
            om[i] = Hv[static_cast<size_t>(i)];
        return out;
    }

    nb::list hvp_seq(nb::object x, nb::object v) {
        set_inputs_seq(x);
        if (!is_sequence(v)) [[unlikely]]
            throw std::invalid_argument("HessFn.hvp: expected list/tuple for v");
            
        nb::sequence sv = nb::cast<nb::sequence>(v);
        const size_t n = var_nodes.size();
        if (static_cast<size_t>(nb::len(sv)) != n) [[unlikely]]
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

// ---------- Optimized cache with better hash and reduced contention ----------
struct FnKey {
    PyObject *f;
    size_t arity;
    bool vector;
    
    constexpr bool operator==(const FnKey &o) const noexcept {
        return f == o.f && arity == o.arity && vector == o.vector;
    }
};

struct FnKeyHash {
    [[gnu::always_inline]]
    constexpr size_t operator()(const FnKey &k) const noexcept {
        size_t h = std::bit_cast<size_t>(k.f);
        h ^= h >> 30;
        h *= 0xbf58476d1ce4e5b9ULL;
        h ^= h >> 27;
        h *= 0x94d049bb133111ebULL;
        h ^= h >> 31;
        h ^= k.arity + 0x9e3779b97f4a7c15ULL;
        h ^= static_cast<size_t>(k.vector) << 1;
        return h;
    }
};

thread_local tsl::robin_map<FnKey, std::shared_ptr<ValFn>,  FnKeyHash> tl_val_cache;
thread_local tsl::robin_map<FnKey, std::shared_ptr<GradFn>, FnKeyHash> tl_grad_cache;
thread_local tsl::robin_map<FnKey, std::shared_ptr<HessFn>, FnKeyHash> tl_hess_cache;

static std::shared_mutex g_cache_mtx;
static tsl::robin_map<FnKey, std::weak_ptr<ValFn>,  FnKeyHash> g_val_cache;
static tsl::robin_map<FnKey, std::weak_ptr<GradFn>, FnKeyHash> g_grad_cache;
static tsl::robin_map<FnKey, std::weak_ptr<HessFn>, FnKeyHash> g_hess_cache;

template <class T, class TLMap, class GlobalMap, class Maker>
[[gnu::hot]]
static inline std::shared_ptr<T> cache_get_or_make(TLMap &tl_map, GlobalMap &g_map, const FnKey &k, Maker &&make) {
    if (auto it = tl_map.find(k); it != tl_map.end()) [[likely]]
        return it->second;
    
    {
        std::shared_lock rlk(g_cache_mtx);
        if (auto it = g_map.find(k); it != g_map.end()) {
            if (auto sp = it->second.lock()) {
                tl_map[k] = sp;
                return sp;
            }
        }
    }
    
    std::unique_lock wlk(g_cache_mtx);
    if (auto it = g_map.find(k); it != g_map.end()) {
        if (auto sp = it->second.lock()) {
            tl_map[k] = sp;
            return sp;
        }
    }
    
    auto sp = make();
    g_map[k] = sp;
    tl_map[k] = sp;
    return sp;
}

[[gnu::always_inline]]
static inline std::shared_ptr<ValFn> get_or_make_val(nb::object f, size_t n, bool vec) {
    FnKey k{f.ptr(), n, vec};
    return cache_get_or_make<ValFn>(tl_val_cache, g_val_cache, k, 
        [&]{ return std::make_shared<ValFn>(f, n, vec); });
}

[[gnu::always_inline]]
static inline std::shared_ptr<GradFn> get_or_make_grad(nb::object f, size_t n, bool vec) {
    FnKey k{f.ptr(), n, vec};
    return cache_get_or_make<GradFn>(tl_grad_cache, g_grad_cache, k, 
        [&]{ return std::make_shared<GradFn>(f, n, vec); });
}

[[gnu::always_inline]]
static inline std::shared_ptr<HessFn> get_or_make_hess(nb::object f, size_t n, bool vec) {
    FnKey k{f.ptr(), n, vec};
    return cache_get_or_make<HessFn>(tl_hess_cache, g_hess_cache, k, 
        [&]{ return std::make_shared<HessFn>(f, n, vec); });
}

// ---------- Immediate / high-level APIs (optimized) ----------
[[gnu::hot]]
static double py_value(nb::object f, nb::args xs) {
    auto g = std::make_shared<ADGraph>();
    
    if (nb::len(xs) == 1 && is_sequence(xs[0])) [[likely]] {
        nb::sequence seq = nb::borrow<nb::sequence>(xs[0]);
        const size_t n = nb::len(seq);
        std::vector<nb::object> expr_elems;
        expr_elems.reserve(n);
        
        for (size_t i = 0; i < n; ++i)
            expr_elems.emplace_back(nb::cast(as_expression(seq[i], g)));
            
        auto ret = call_python_function_with_list(f, expr_elems);
        auto expr = ensure_expression(ret, g);
        
        nb::gil_scoped_release nogil;
        return expr->evaluate();
    }
    
    std::vector<nb::object> expr_args;
    expr_args.reserve(nb::len(xs));
    for (size_t i = 0; i < nb::len(xs); ++i) 
        expr_args.emplace_back(nb::cast(as_expression(xs[i], g)));
        
    auto ret = call_python_function(f, expr_args);
    auto expr = ensure_expression(ret, g);
    
    nb::gil_scoped_release nogil;
    return expr->evaluate();
}

[[gnu::hot]]
static double py_value_numpy(nb::object f, Arr1D x_in) {
    auto x = as_span_1d(x_in);
    auto vf = get_or_make_val(f, x.size(), /*vector_input=*/true);
    return (*vf)(x_in);
}

[[gnu::hot]]
static nb::list py_gradient(nb::object f, nb::args xs) {
    auto g = std::make_shared<ADGraph>();
    
    if (nb::len(xs) == 1 && is_sequence(xs[0])) [[likely]] {
        nb::sequence seq = nb::borrow<nb::sequence>(xs[0]);
        const size_t n = nb::len(seq);
        std::vector<nb::object> expr_elems; 
        expr_elems.reserve(n);
        std::vector<std::string> var_names(n);
        
        for (size_t i = 0; i < n; ++i) {
            nb::handle item = seq[i];
            if (nb::isinstance<Variable>(item)) [[likely]] {
                auto v = nb::cast<std::shared_ptr<Variable>>(item);
                var_names[i] = v->getName();
                expr_elems.emplace_back(nb::cast(make_expr_from_variable(v, g)));
            } else if (is_number(item)) [[likely]] {
                const double val = nb::cast<double>(item);
                auto vx = std::make_shared<Variable>("x"+std::to_string(i), val);
                var_names[i] = vx->getName();
                expr_elems.emplace_back(nb::cast(make_expr_from_variable(vx, g)));
            } else if (nb::isinstance<Expression>(item)) {
                auto e = nb::cast<std::shared_ptr<Expression>>(item);
                g->adoptSubgraph(e->node);
                expr_elems.emplace_back(nb::cast(e));
            } else [[unlikely]] {
                throw std::invalid_argument("gradient: elements must be Variable / Expression / number.");
            }
        }
        
        auto ret = call_python_function_with_list(f, expr_elems);
        auto expr = ensure_expression(ret, g);
        tsl::robin_map<std::string, double> grad_map;
        
        {
            nb::gil_scoped_release nogil;
            grad_map = expr->computeGradient();
        }
        
        nb::list out;
        for (size_t i = 0; i < n; ++i) {
            double gi = 0.0;
            if (!var_names[i].empty()) [[likely]] {
                if (auto it = grad_map.find(var_names[i]); it != grad_map.end())
                    gi = it->second;
            }
            out.append(nb::float_(gi));
        }
        return out;
    }

    // Handle positional arguments case
    const size_t n = nb::len(xs);
    std::vector<nb::object> expr_args; 
    expr_args.reserve(n);
    std::vector<std::string> var_names(n);
    
    for (size_t i = 0; i < n; ++i) {
        const auto &a = xs[i];
        if (nb::isinstance<Variable>(a)) [[likely]] {
            auto v = nb::cast<std::shared_ptr<Variable>>(a);
            var_names[i] = v->getName();
            expr_args.emplace_back(nb::cast(make_expr_from_variable(v, g)));
        } else if (is_number(a)) [[likely]] {
            const double val = nb::cast<double>(a);
            auto vx = std::make_shared<Variable>("x"+std::to_string(i), val);
            var_names[i] = vx->getName();
            expr_args.emplace_back(nb::cast(make_expr_from_variable(vx, g)));
        } else if (nb::isinstance<Expression>(a)) {
            auto e = nb::cast<std::shared_ptr<Expression>>(a);
            g->adoptSubgraph(e->node);
            expr_args.emplace_back(nb::cast(e));
        } else [[unlikely]] {
            throw std::invalid_argument("gradient: args must be Variable / Expression / number.");
        }
    }
    
    auto ret = call_python_function(f, expr_args);
    auto expr = ensure_expression(ret, g);
    tsl::robin_map<std::string, double> grad_map;
    
    {
        nb::gil_scoped_release nogil;
        grad_map = expr->computeGradient();
    }
    
    nb::list out;
    for (size_t i = 0; i < n; ++i) {
        double gi = 0.0;
        if (!var_names[i].empty()) [[likely]] {
            if (auto it = grad_map.find(var_names[i]); it != grad_map.end())
                gi = it->second;
        }
        out.append(nb::float_(gi));
    }
    return out;
}

// Fused value+grad (ndarray) - more efficient than separate calls
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

// ============================================================================
//                                 Module
// ============================================================================
NB_MODULE(ad, m) {
    m.doc() = "ad optimization and autodiff module (nanobind, NumPy fast paths, cached, C++23 optimized)";

    // Variable
nb::class_<Variable>(m, "Variable")
        .def(nb::init<const std::string &, double, double, double>(),
             "name"_a, "value"_a = 0.0,
             "lb"_a = -std::numeric_limits<double>::infinity(),
             "ub"_a =  std::numeric_limits<double>::infinity())
        .def_prop_ro("name", &Variable::getName)
        .def_prop_rw("value", &Variable::getValue, &Variable::setValue)
        .def_prop_rw("gradient", &Variable::getGradient, &Variable::setGradient)
        .def("getName", &Variable::getName)
        .def("getValue",&Variable::getValue)
        .def("setValue",&Variable::setValue)
        .def("getGradient",&Variable::getGradient)
        .def("__pow__", [](const std::shared_ptr<Variable> &v, double p) {
            auto g = std::make_shared<ADGraph>();
            auto ex = make_expr_from_variable(v, g);
            return expr_pow_any(nb::cast(ex), p);
        })
        .def("__rpow__", [](const std::shared_ptr<Variable> &v, double s) {
            auto g = std::make_shared<ADGraph>();
            auto ex = make_expr_from_variable(v, g);
            return scalar_pow_expr(s, ex);
        });

    // Expression
nb::class_<Expression>(m, "Expression")
        .def("computeGradient", &Expression::computeGradient)
        .def("computeHessian",  &Expression::computeHessian)
        .def("evaluate", (double (Expression::*)() const) &Expression::evaluate)
        .def("__str__", [](Expression &e) { return e.toString(); })
        .def("__add__",      [](const Expression &a, const Expression &b) { return a + b; })
        .def("__sub__",      [](const Expression &a, const Expression &b) { return a - b; })
        .def("__mul__",      [](const Expression &a, const Expression &b) { return a * b; })
        .def("__truediv__",  [](const Expression &a, const Expression &b) { return a / b; })
        .def("__add__",      [](const Expression &a, double s) { return a + s; })
        .def("__sub__",      [](const Expression &a, double s) { return a - s; })
        .def("__mul__",      [](const Expression &a, double s) { return a * s; })
        .def("__truediv__",  [](const Expression &a, double s) { return a / s; })
        .def("__radd__",     [](const Expression &a, double s) { return a + s; })
        .def("__rsub__",     [](const Expression &a, double s) {
            auto lhs = make_expr_from_number(s, a.graph);
            return (*lhs) - a;
        })
        .def("__rmul__",     [](const Expression &a, double s) { return a * s; })
        .def("__rtruediv__", [](const Expression &a, double s) { return s / a; })
        .def("__neg__",      [](const Expression &a) { return -a; })
        .def("__pow__",      [](const std::shared_ptr<Expression> &a, double p) {
            return expr_pow_any(nb::cast(a), p);
        })
        .def("__pow__",      [](const std::shared_ptr<Expression> &, const std::shared_ptr<Expression> &) {
            throw std::invalid_argument("Expression ** Expression not supported; exponent must be scalar.");
        })
        .def("__rpow__",     [](const std::shared_ptr<Expression> &a, double s) {
            return scalar_pow_expr(s, a);
        });

    // Immediate helpers
    m.def("val", [](nb::object f, nb::args xs) { 
        return py_value(f, xs); 
    }, "Evaluate f(*xs) or f([xs])");

    // NumPy overloads (cached)
    m.def("val", [](nb::object f, Arr1D x) { 
        return py_value_numpy(f, x); 
    }, "f"_a, "x"_a, "Evaluate f(x: ndarray) fast path (cached)");
    
    m.def("valgrad", [](nb::object f, Arr1D x) { 
        return py_value_grad_numpy(f, x); 
    }, "f"_a, "x"_a, "Return (f(x), grad f(x)) fast path for ndarray (cached).");
    
    m.def("hess", [](nb::object f, Arr1D x) { 
        return py_hessian_numpy(f, x); 
    }, "f"_a, "x"_a, "Return Hessian at x (ndarray) as ndarray[n,n] (cached).");

    // Grad (compat + ndarray)
    m.def("grad", [](nb::object f, nb::args xs) { 
        return py_gradient(f, xs); 
    }, "Return gradient of f as a list of floats in input order.");
    
    m.def("grad", [](nb::object f, Arr1D x) { 
        return py_gradient_numpy(f, x); 
    }, "f"_a, "x"_a, "Return gradient at x (ndarray[float64]) as an ndarray (cached).");

    // Unary operations with optimized dispatch
    m.def("sin",   [](nb::object x){ return unary_dispatch(x, Operator::Sin);  }, "sin(x)");
    m.def("cos",   [](nb::object x){ return unary_dispatch(x, Operator::Cos);  }, "cos(x)");
    m.def("tan",   [](nb::object x){ return unary_dispatch(x, Operator::Tan);  }, "tan(x)");
    m.def("exp",   [](nb::object x){ return unary_dispatch(x, Operator::Exp);  }, "exp(x)");
    m.def("log",   [](nb::object x){ return unary_dispatch(x, Operator::Log);  }, "log(x)");
    m.def("tanh",  [](nb::object x){ return unary_dispatch(x, Operator::Tanh); }, "tanh(x)");
    m.def("silu",  [](nb::object x){ return unary_dispatch(x, Operator::Silu); }, "silu(x) = x * sigmoid(x)");
    m.def("relu",  [](nb::object x){ return unary_dispatch(x, Operator::Relu); }, "relu(x) = max(0,x)");
    m.def("softmax", [](nb::object x){ return unary_dispatch(x, Operator::Softmax); },
          "softmax(x) over provided inputs");
    m.def("gelu",  [](nb::object x){ return unary_dispatch(x, Operator::Gelu); }, "gelu(x)");

    // max / pow
    m.def("max", [](nb::object a, nb::object b){ return binary_max_dispatch(a, b); },
          "max(a,b) with subgradient 0.5/0.5 at ties");
    m.def("pow", [](nb::object x, double p){ return expr_pow_any(x, p); },
          "pow(x,p) builds exp(p*log(x)) symbolically when p not integer");

    // Compiled value
    nb::class_<ValFn>(m, "ValFn")
        .def("__call__", &ValFn::operator(), "x"_a, "Evaluate f(x) (ndarray) -> float")
        .def("expr_str", &ValFn::expr_str)
        .def("call_seq", &ValFn::call_seq, "x_list"_a, "Evaluate f(x) (list/tuple) -> float")
        .def("__repr__", [](const ValFn &self){ return "<ValFn expr=" + self.expr_str() + ">"; });
    
    m.def("sym_val", [](nb::object f, ssize_t arity, bool vector_input){
            return std::make_shared<ValFn>(f, static_cast<size_t>(arity), vector_input);
        }, "f"_a, "arity"_a, "vector_input"_a = true,
        "Compile a value function once; returns ValFn.", 
        nb::rv_policy::take_ownership);
    
    m.def("sym_val_cached", [](nb::object f, ssize_t arity, bool vector_input){
            return get_or_make_val(f, static_cast<size_t>(arity), vector_input);
        }, "f"_a, "arity"_a, "vector_input"_a = true,
        "Get or compile & cache a ValFn keyed by (f,arity,vector_input).",
        nb::rv_policy::take_ownership);

    // Compiled gradient
    nb::class_<GradFn>(m, "GradFn")
        .def("__call__", &GradFn::call_numpy, "x"_a,
             "Evaluate gradient at x (ndarray) -> ndarray")
        .def("__call__", &GradFn::operator(), "x_list"_a,
             "Evaluate gradient at x (list/tuple) -> list")
        .def("expr_str", &GradFn::expr_str)
        .def("value", &GradFn::value_numpy, "x"_a,
             "Evaluate f(x) (ndarray) -> float")
        .def("__repr__", [](const GradFn &self){ return "<GradFn expr=" + self.expr_str() + ">"; });
    
    m.def("sym_grad", [](nb::object f, ssize_t arity, bool vector_input){
            return std::make_shared<GradFn>(f, static_cast<size_t>(arity), vector_input);
        }, "f"_a, "arity"_a, "vector_input"_a = true,
        "Compile a gradient function once; returns GradFn.",
        nb::rv_policy::take_ownership);
    
    m.def("gradient_from_example", [](nb::object f, nb::handle example){
            if (!is_sequence(example)) 
                throw std::invalid_argument("gradient_from_example: pass a list/tuple example");
            nb::sequence seq = nb::cast<nb::sequence>(example);
            return std::make_shared<GradFn>(f, static_cast<size_t>(nb::len(seq)), true);
        }, "f"_a, "example"_a,
        "Compile a gradient using a list/tuple example to infer arity.",
        nb::rv_policy::take_ownership);
    
    m.def("sym_grad_cached", [](nb::object f, ssize_t arity, bool vector_input){
            return get_or_make_grad(f, static_cast<size_t>(arity), vector_input);
        }, "f"_a, "arity"_a, "vector_input"_a = true,
        "Get or compile & cache a GradFn keyed by (f,arity,vector_input).",
        nb::rv_policy::take_ownership);

    // Compiled Hessian
    nb::class_<HessFn>(m, "HessFn")
        .def("__call__", &HessFn::call_numpy, "x"_a,
             "Evaluate Hessian at x (ndarray) -> ndarray[n,n]")
        .def("__call__", &HessFn::operator(), "x_list"_a,
             "Evaluate Hessian at x (list[list])")
        .def("hvp", &HessFn::hvp_numpy, "x"_a, "v"_a,
             "HVP at x with v (ndarray) -> ndarray")
        .def("expr_str", &HessFn::expr_str)
        .def("__repr__", [](const HessFn &self){ return "<HessFn expr=" + self.expr_str() + ">"; });
    
    m.def("sym_hess", [](nb::object f, ssize_t arity, bool vector_input){
            return std::make_shared<HessFn>(f, static_cast<size_t>(arity), vector_input);
        }, "f"_a, "arity"_a, "vector_input"_a = true,
        "Compile a Hessian function once; returns HessFn.",
        nb::rv_policy::take_ownership);
    
    m.def("hessian_from_example", [](nb::object f, nb::handle example){
            if (!is_sequence(example)) 
                throw std::invalid_argument("hessian_from_example: pass a list/tuple example");
            nb::sequence seq = nb::cast<nb::sequence>(example);
            return std::make_shared<HessFn>(f, static_cast<size_t>(nb::len(seq)), true);
        }, "f"_a, "example"_a,
        "Compile a Hessian using a list/tuple example to infer arity.",
        nb::rv_policy::take_ownership);
    
    m.def("sym_hess_cached", [](nb::object f, ssize_t arity, bool vector_input){
            return get_or_make_hess(f, static_cast<size_t>(arity), vector_input);
        }, "f"_a, "arity"_a, "vector_input"_a = true,
        "Get or compile & cache a HessFn keyed by (f,arity,vector_input).",
        nb::rv_policy::take_ownership);
}