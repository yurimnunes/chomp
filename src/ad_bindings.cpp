// ad.cpp  —  PyBind11 + NumPy fast paths + GIL release
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../include/ad/ADGraph.h"
#include "../include/ad/Definitions.h" // ensure_epoch_ & epoch fields
#include "../include/ad/Expression.h"
#include "../include/ad/Variable.h"

#include <cmath>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

// ========= Small helpers =========
static inline bool is_number(const py::handle &h) {
    return py::isinstance<py::int_>(h) || py::isinstance<py::float_>(h);
}
static inline bool is_sequence(const py::handle &h) {
    return py::isinstance<py::sequence>(h) && !py::isinstance<py::str>(h);
}
static py::tuple to_tuple(const std::vector<py::object> &vec) {
    py::tuple t(vec.size());
    for (size_t i = 0; i < vec.size(); ++i)
        t[i] = vec[i];
    return t;
}

// NumPy helpers
using Arr1D = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Arr2D = py::array_t<double, py::array::c_style | py::array::forcecast>;

static std::span<const double> as_span_1d(const Arr1D &a) {
    if (a.ndim() != 1)
        throw std::invalid_argument("expected 1D float64 array");
    return {a.data(), static_cast<size_t>(a.shape(0))};
}
static std::pair<ssize_t, ssize_t> shape_2d(const Arr2D &a) {
    if (a.ndim() != 2)
        throw std::invalid_argument("expected 2D float64 array");
    return {a.shape(0), a.shape(1)};
}

// ========= Expression construction utils =========
static ADNodePtr make_const_node(const ADGraphPtr &g, double v) {
    auto n = std::make_shared<ADNode>();
    n->type = Operator::cte;
    n->value = v;
    if (g)
        g->addNode(n);
    return n;
}
static std::shared_ptr<Expression> make_const_expr(double val,
                                                   const ADGraphPtr &g) {
    auto n = make_const_node(g, val);
    return std::make_shared<Expression>(n, g);
}
static std::shared_ptr<Expression>
make_expr_from_variable(const std::shared_ptr<Variable> &v,
                        const ADGraphPtr &g) {
    return std::make_shared<Expression>(v, 1.0, g);
}
static std::shared_ptr<Expression> make_expr_from_number(double val,
                                                         const ADGraphPtr &g) {
    return make_const_expr(val, g);
}
static std::shared_ptr<Expression> as_expression(const py::handle &h,
                                                 const ADGraphPtr &g) {
    if (py::isinstance<Expression>(h)) {
        return h.cast<std::shared_ptr<Expression>>();
    }
    if (py::isinstance<Variable>(h)) {
        auto v = h.cast<std::shared_ptr<Variable>>();
        return make_expr_from_variable(v, g);
    }
    if (is_number(h)) {
        return make_expr_from_number(py::cast<double>(h), g);
    }
    throw std::invalid_argument(
        "Argument must be Expression, Variable, int, or float.");
}
static std::shared_ptr<Expression> ensure_expression(const py::handle &ret,
                                                     const ADGraphPtr &g) {
    if (py::isinstance<Expression>(ret))
        return ret.cast<std::shared_ptr<Expression>>();
    if (is_number(ret))
        return make_expr_from_number(py::cast<double>(ret), g);
    throw std::invalid_argument(
        "Function must return Expression or a numeric value.");
}

// ========= Unary ops dispatch =========
static std::shared_ptr<Expression>
unary_from_expression(Operator op, const std::shared_ptr<Expression> &x) {
    auto g = x->graph;
    auto e = std::make_shared<Expression>(g);
    e->node->type = op;
    e->node->addInput(x->node);
    if (g)
        g->addNode(e->node);
    return e;
}
static std::shared_ptr<Expression>
unary_from_variable(Operator op, const std::shared_ptr<Variable> &v) {
    auto g = std::make_shared<ADGraph>();
    auto x = make_expr_from_variable(v, g);
    return unary_from_expression(op, x);
}
static py::object unary_dispatch(py::object x, Operator op) {
    if (py::isinstance<Expression>(x)) {
        auto ex = x.cast<std::shared_ptr<Expression>>();
        return py::cast(unary_from_expression(op, ex));
    }
    if (py::isinstance<Variable>(x)) {
        auto v = x.cast<std::shared_ptr<Variable>>();
        return py::cast(unary_from_variable(op, v));
    }
    if (is_number(x)) {
        const double a = py::cast<double>(x);
        switch (op) {
        case Operator::Sin:
            return py::float_(std::sin(a));
        case Operator::Cos:
            return py::float_(std::cos(a));
        case Operator::Tan:
            return py::float_(std::tan(a));
        case Operator::Exp:
            return py::float_(std::exp(a));
        case Operator::Log:
            return py::float_(std::log(a));
        default:
            return py::float_(a);
        }
    }
    throw std::invalid_argument(
        "Argument must be Expression, Variable, int, or float.");
}

// ========= pow helpers =========
static ADNodePtr powi_node(const ADGraphPtr &g, const ADNodePtr &base,
                           long long e) {
    if (e == 0) {
        auto one = std::make_shared<ADNode>();
        one->type = Operator::cte;
        one->value = 1.0;
        g->addNode(one);
        return one;
    }
    auto mul_node = [&](const ADNodePtr &a, const ADNodePtr &b) -> ADNodePtr {
        auto m = std::make_shared<ADNode>();
        m->type = Operator::Multiply;
        m->addInput(a);
        m->addInput(b);
        g->addNode(m);
        return m;
    };
    auto pow_pos = [&](long long k) -> ADNodePtr {
        ADNodePtr result;
        ADNodePtr cur = base;
        bool have_result = false;
        while (k > 0) {
            if (k & 1LL) {
                result = have_result ? mul_node(result, cur) : cur;
                have_result = true;
            }
            cur = mul_node(cur, cur);
            k >>= 1LL;
        }
        return have_result ? result : base;
    };
    if (e > 0)
        return pow_pos(e);
    auto num = std::make_shared<ADNode>();
    num->type = Operator::cte;
    num->value = 1.0;
    g->addNode(num);
    auto den = pow_pos(-e);
    auto div = std::make_shared<ADNode>();
    div->type = Operator::Divide;
    div->addInput(num);
    div->addInput(den);
    g->addNode(div);
    return div;
}

static std::shared_ptr<Expression> expr_pow_any(py::object x, double p) {
    ADGraphPtr g = std::make_shared<ADGraph>();
    auto ex = as_expression(x, g);
    if (ex->node)
        g->adoptSubgraph(ex->node);

    double pr = std::round(p);
    bool is_int = std::fabs(p - pr) < 1e-12 && std::isfinite(pr);
    if (is_int) {
        long long e = static_cast<long long>(pr);
        auto out = std::make_shared<Expression>(g);
        out->node = powi_node(g, ex->node, e);
        return out;
    }
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

static std::shared_ptr<Expression>
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

// ========= max dispatch =========
static py::object binary_max_dispatch(py::object x, py::object y) {
    const bool nx = is_number(x), ny = is_number(y);
    if (nx && ny) {
        double a = py::cast<double>(x), b = py::cast<double>(y);
        return py::float_(a >= b ? a : b);
    }
    ADGraphPtr g = std::make_shared<ADGraph>();
    auto ex = as_expression(x, g);
    auto ey = as_expression(y, g);
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
    return py::cast(out);
}

// ========= Call adapters =========
static py::object
call_with_positional(py::function f, const std::vector<py::object> &expr_args) {
    return f(*to_tuple(expr_args));
}
static py::object
call_with_single_sequence(py::function f,
                          const std::vector<py::object> &expr_list) {
    py::list l(expr_list.size());
    for (size_t i = 0; i < expr_list.size(); ++i)
        l[i] = expr_list[i];
    return f(l);
}

// ========= High-level immediate APIs =========

// value: list/tuple compatibility
static double py_value(py::function f, py::args xs) {
    auto g = std::make_shared<ADGraph>();
    if (xs.size() == 1 && is_sequence(xs[0])) {
        py::sequence seq = py::reinterpret_borrow<py::sequence>(xs[0]);
        const size_t n = seq.size();
        std::vector<py::object> expr_elems;
        expr_elems.reserve(n);
        for (size_t i = 0; i < n; ++i)
            expr_elems.emplace_back(py::cast(as_expression(seq[i], g)));
        auto ret = call_with_single_sequence(f, expr_elems);
        auto expr = ensure_expression(ret, g);
        py::gil_scoped_release nogil;
        return expr->evaluate();
    }
    std::vector<py::object> expr_args;
    expr_args.reserve(xs.size());
    for (auto &a : xs)
        expr_args.emplace_back(py::cast(as_expression(a, g)));
    auto ret = call_with_positional(f, expr_args);
    auto expr = ensure_expression(ret, g);
    py::gil_scoped_release nogil;
    return expr->evaluate();
}

// FAST value: NumPy 1D
static double py_value_numpy(py::function f, Arr1D x_in) {
    // Build a graph with Variables seeded from ndarray
    auto g = std::make_shared<ADGraph>();
    auto x = as_span_1d(x_in);
    std::vector<py::object> expr_elems;
    expr_elems.reserve(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        auto vx = std::make_shared<Variable>("x" + std::to_string(i), x[i]);
        expr_elems.emplace_back(py::cast(make_expr_from_variable(vx, g)));
    }
    auto ret = call_with_single_sequence(f, expr_elems);
    auto expr = ensure_expression(ret, g);
    py::gil_scoped_release nogil;
    return expr->evaluate();
}

// gradient: list/tuple compatibility (kept)
static py::list py_gradient(py::function f, py::args xs) {
    auto g = std::make_shared<ADGraph>();
    // vector input
    if (xs.size() == 1 && is_sequence(xs[0])) {
        py::sequence seq = py::reinterpret_borrow<py::sequence>(xs[0]);
        const size_t n = seq.size();
        std::vector<py::object> expr_elems;
        expr_elems.reserve(n);
        std::vector<std::string> var_names(n, "");
        for (size_t i = 0; i < n; ++i) {
            py::handle item = seq[i];
            if (py::isinstance<Variable>(item)) {
                auto v = item.cast<std::shared_ptr<Variable>>();
                var_names[i] = v->getName();
                expr_elems.emplace_back(
                    py::cast(make_expr_from_variable(v, g)));
            } else if (is_number(item)) {
                double v = py::cast<double>(item);
                auto vx =
                    std::make_shared<Variable>("x" + std::to_string(i), v);
                var_names[i] = vx->getName();
                expr_elems.emplace_back(
                    py::cast(make_expr_from_variable(vx, g)));
            } else if (py::isinstance<Expression>(item)) {
                auto e = item.cast<std::shared_ptr<Expression>>();
                g->adoptSubgraph(e->node);
                expr_elems.emplace_back(py::cast(e));
            } else {
                throw std::invalid_argument(
                    "gradient: sequence elements must be Variable / Expression "
                    "/ number.");
            }
        }
        auto ret = call_with_single_sequence(f, expr_elems);
        auto expr = ensure_expression(ret, g);
        std::unordered_map<std::string, double> grad_map;
        {
            py::gil_scoped_release nogil;
            grad_map = expr->computeGradient();
        }
        py::list out(n);
        for (size_t i = 0; i < n; ++i) {
            double gi = 0.0;
            if (!var_names[i].empty()) {
                auto it = grad_map.find(var_names[i]);
                if (it != grad_map.end())
                    gi = it->second;
            }
            out[i] = py::float_(gi);
        }
        return out;
    }
    // positional
    const size_t n = xs.size();
    std::vector<py::object> expr_args;
    expr_args.reserve(n);
    std::vector<std::string> var_names(n, "");
    for (size_t i = 0; i < n; ++i) {
        const auto &a = xs[i];
        if (py::isinstance<Variable>(a)) {
            auto v = a.cast<std::shared_ptr<Variable>>();
            var_names[i] = v->getName();
            expr_args.emplace_back(py::cast(make_expr_from_variable(v, g)));
        } else if (is_number(a)) {
            double v = py::cast<double>(a);
            auto vx = std::make_shared<Variable>("x" + std::to_string(i), v);
            var_names[i] = vx->getName();
            expr_args.emplace_back(py::cast(make_expr_from_variable(vx, g)));
        } else if (py::isinstance<Expression>(a)) {
            auto e = a.cast<std::shared_ptr<Expression>>();
            g->adoptSubgraph(e->node);
            expr_args.emplace_back(py::cast(e));
        } else {
            throw std::invalid_argument(
                "gradient: args must be Variable / Expression / number.");
        }
    }
    auto ret = call_with_positional(f, expr_args);
    auto expr = ensure_expression(ret, g);
    std::unordered_map<std::string, double> grad_map;
    {
        py::gil_scoped_release nogil;
        grad_map = expr->computeGradient();
    }
    py::list out(n);
    for (size_t i = 0; i < n; ++i) {
        double gi = 0.0;
        if (!var_names[i].empty()) {
            auto it = grad_map.find(var_names[i]);
            if (it != grad_map.end())
                gi = it->second;
        }
        out[i] = py::float_(gi);
    }
    return out;
}

// ---- FAST immediate gradient: grad(f, x: ndarray) -> ndarray ----
static py::array_t<double> py_gradient_numpy(py::function f, Arr1D x_in) {
    auto x = as_span_1d(x_in);
    const size_t n = x.size();

    // 1) Build a graph and symbolic inputs (Variables) seeded from x
    auto g = std::make_shared<ADGraph>();
    std::vector<py::object> expr_elems;
    expr_elems.reserve(n);
    std::vector<ADNodePtr> var_nodes;
    var_nodes.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        auto v = std::make_shared<Variable>("x" + std::to_string(i), x[i]);
        auto ex = std::make_shared<Expression>(v, 1.0, g);
        expr_elems.emplace_back(py::cast(ex));
        var_nodes.push_back(ex->node);
    }

    // 2) Call user function and ensure we got an Expression on our graph
    auto ret = call_with_single_sequence(f, expr_elems);
    auto expr = ensure_expression(ret, g);
    g->adoptSubgraph(expr->node);

    // 3) Do forward + reverse without the GIL
    {
        py::gil_scoped_release nogil;
        g->resetGradients();        // zero gradients
        g->computeForwardPass();    // compute .value across graph
        expr->node->gradient = 1.0; // seed
        g->initiateBackwardPass(expr->node);
    }

    // 4) Pack result into a 1D float64 ndarray
    py::array_t<double> out((ssize_t)n);
    auto om = out.mutable_unchecked<1>();
    for (ssize_t i = 0; i < (ssize_t)n; ++i)
        om(i) = var_nodes[(size_t)i]->gradient;

    return out;
}

// ---- add near the other NumPy helpers ----
static py::array_t<double> py_hessian_numpy(py::function f, Arr1D x_in) {
    auto x = as_span_1d(x_in);
    const size_t n = x.size();

    // Build graph and symbolic inputs once
    auto g = std::make_shared<ADGraph>();
    std::vector<py::object> expr_elems;
    expr_elems.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        auto vx = std::make_shared<Variable>("x" + std::to_string(i), x[i]);
        expr_elems.emplace_back(py::cast(make_expr_from_variable(vx, g)));
    }

    // Call user f and get Expression
    auto ret = call_with_single_sequence(f, expr_elems);
    auto expr = ensure_expression(ret, g);

    // Compute H via HVP columns, no GIL
    py::array_t<double> H({(ssize_t)n, (ssize_t)n});
    auto Hm = H.mutable_unchecked<2>();

    // We rely on g->initializeNodeVariables() if your HVP uses it
    g->initializeNodeVariables();

    std::vector<ADNodePtr> var_nodes;
    var_nodes.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        // Reuse the nodes from the variables we created above
        auto vx = std::static_pointer_cast<Expression>(
            expr_elems[i].cast<std::shared_ptr<Expression>>());
        var_nodes.push_back(vx->node);
    }

    auto hvp_once = [&](const std::vector<double> &v) {
        std::vector<double> Hv(n, 0.0);
        py::gil_scoped_release nogil;

        g->resetTangents();
        for (size_t i = 0; i < n; ++i) {
            auto &nd = var_nodes[i];
            set_epoch_value(nd->dot, nd->dot_epoch, g->cur_dot_epoch_, v[i]);
        }
        g->resetForwardPass();
        g->computeForwardPassWithDot();
        g->resetGradients();
        g->resetGradDot();
        set_epoch_value(expr->node->gradient, expr->node->grad_epoch,
                        g->cur_grad_epoch_, 1.0);
        set_epoch_value(expr->node->grad_dot, expr->node->gdot_epoch,
                        g->cur_gdot_epoch_, 0.0);
        g->initiateBackwardPassHVP();

        for (size_t i = 0; i < n; ++i)
            Hv[i] = var_nodes[i]->grad_dot;
        return Hv;
    };

    std::vector<double> e(n, 0.0);
    for (size_t j = 0; j < n; ++j) {
        std::fill(e.begin(), e.end(), 0.0);
        e[j] = 1.0;
        auto col = hvp_once(e);
        for (size_t i = 0; i < n; ++i)
            Hm((ssize_t)i, (ssize_t)j) = col[i];
    }
    return H;
}

// ========= Compiled Gradient (fast ndarray path) =========
class GradFn {
public:
    ADGraphPtr g;
    ADNodePtr expr_root;
    std::vector<ADNodePtr> var_nodes; // input order
    bool vector_mode;

    std::string expr_str() const {
        if (!g || !expr_root)
            return std::string();
        return g->getExpression(expr_root);
    }

    GradFn(py::function f, size_t arity, bool vector_input)
        : g(std::make_shared<ADGraph>()), vector_mode(vector_input) {
        std::vector<py::object> expr_args;
        expr_args.reserve(arity);
        var_nodes.reserve(arity);
        for (size_t i = 0; i < arity; ++i) {
            auto vx = std::make_shared<Variable>("x" + std::to_string(i), 0.0);
            auto ex = std::make_shared<Expression>(vx, 1.0, g);
            var_nodes.push_back(ex->node);
            expr_args.emplace_back(py::cast(ex));
        }
        py::object ret;
        if (vector_mode) {
            py::list l(arity);
            for (size_t i = 0; i < arity; ++i)
                l[i] = expr_args[i];
            ret = f(l);
        } else {
            py::tuple args(expr_args.size());
            for (size_t i = 0; i < expr_args.size(); ++i)
                args[i] = expr_args[i];
            ret = f(*args);
        }
        if (ret.is_none())
            throw std::invalid_argument(
                "compile_gradient: function returned None");
        auto expr = ret.cast<std::shared_ptr<Expression>>();
        g->adoptSubgraph(expr->node);
        expr_root = expr->node;
    }

    // FAST path: __call__(x: ndarray) -> ndarray
    Arr1D call_numpy(Arr1D x_in) {
        auto x = as_span_1d(x_in);
        if (x.size() != var_nodes.size())
            throw std::invalid_argument("GradFn: wrong input length");
        for (size_t i = 0; i < x.size(); ++i)
            var_nodes[i]->value = x[i];

        {
            py::gil_scoped_release nogil;
            g->resetGradients();
            g->computeForwardPass();
            expr_root->gradient = 1.0;
            g->initiateBackwardPass(expr_root);
        }
        Arr1D out(x_in.request().size);
        auto om = out.mutable_unchecked<1>();
        for (ssize_t i = 0; i < static_cast<ssize_t>(x.size()); ++i)
            om(i) = var_nodes[i]->gradient;
        return out;
    }

    // Backwards compatibility: list/tuple
    py::list operator()(py::object x) {
        if (!is_sequence(x))
            throw std::invalid_argument(
                "GradFn: expected a list/tuple of numbers");
        py::sequence seq = x.cast<py::sequence>();
        if ((size_t)seq.size() != var_nodes.size())
            throw std::invalid_argument("GradFn: wrong length");
        for (size_t i = 0; i < var_nodes.size(); ++i)
            var_nodes[i]->value = py::cast<double>(seq[i]);
        {
            py::gil_scoped_release nogil;
            g->resetGradients();
            g->computeForwardPass();
            expr_root->gradient = 1.0;
            g->initiateBackwardPass(expr_root);
        }
        py::list out(var_nodes.size());
        for (size_t i = 0; i < var_nodes.size(); ++i)
            out[i] = py::float_(var_nodes[i]->gradient);
        return out;
    }
};

static py::object gradfn_call_positional(GradFn &self, py::args xs) {
    if (self.vector_mode)
        throw std::invalid_argument(
            "GradFn expects a single array/list; use f([x0,...])");
    if (xs.size() != self.var_nodes.size())
        throw std::invalid_argument("GradFn: wrong number of positional args");
    py::list v(xs.size());
    for (size_t i = 0; i < xs.size(); ++i)
        v[i] = py::float_(py::cast<double>(xs[i]));
    return self(v);
}

static std::shared_ptr<GradFn> py_compile_gradient(py::function f, size_t arity,
                                                   bool vector_input) {
    return std::make_shared<GradFn>(f, arity, vector_input);
}
static std::shared_ptr<GradFn> py_gradient_from_example(py::function f,
                                                        py::handle example) {
    if (!(is_sequence(example)))
        throw std::invalid_argument(
            "gradient_from_example: pass a list/tuple example");
    py::sequence seq = example.cast<py::sequence>();
    return std::make_shared<GradFn>(f, (size_t)seq.size(), true);
}

// ========= Compiled Hessian (HVP) with fast ndarray paths =========
class HessFn {
public:
    ADGraphPtr g;
    ADNodePtr expr_root;
    std::vector<ADNodePtr> var_nodes;
    bool vector_mode;

    std::string expr_str() const {
        if (!g || !expr_root)
            return std::string();
        return g->getExpression(expr_root);
    }

    HessFn(py::function f, size_t arity, bool vector_input)
        : g(std::make_shared<ADGraph>()), vector_mode(vector_input) {
        std::vector<py::object> expr_args;
        expr_args.reserve(arity);
        var_nodes.reserve(arity);
        for (size_t i = 0; i < arity; ++i) {
            auto vx = std::make_shared<Variable>("x" + std::to_string(i), 0.0);
            auto ex = std::make_shared<Expression>(vx, 1.0, g);
            var_nodes.push_back(ex->node);
            expr_args.emplace_back(py::cast(ex));
        }
        py::object ret;
        if (vector_mode) {
            py::list l(arity);
            for (size_t i = 0; i < arity; ++i)
                l[i] = expr_args[i];
            ret = f(l);
        } else {
            py::tuple args(expr_args.size());
            for (size_t i = 0; i < expr_args.size(); ++i)
                args[i] = expr_args[i];
            ret = f(*args);
        }
        if (ret.is_none())
            throw std::invalid_argument(
                "compile_hessian: function returned None");
        auto expr = ret.cast<std::shared_ptr<Expression>>();
        g->adoptSubgraph(expr->node);
        expr_root = expr->node;

        g->initializeNodeVariables();
    }

    void set_inputs_seq(const py::object &x) {
        if (!py::isinstance<py::sequence>(x) || py::isinstance<py::str>(x))
            throw std::invalid_argument("HessFn: expected a list/tuple");
        py::sequence sx = x.cast<py::sequence>();
        if ((size_t)sx.size() != var_nodes.size())
            throw std::invalid_argument("HessFn: wrong input length");
        for (size_t i = 0; i < var_nodes.size(); ++i)
            var_nodes[i]->value = py::cast<double>(sx[i]);
    }
    void set_inputs_arr(const Arr1D &x_in) {
        auto x = as_span_1d(x_in);
        if (x.size() != var_nodes.size())
            throw std::invalid_argument("HessFn: wrong input length");
        for (size_t i = 0; i < var_nodes.size(); ++i)
            var_nodes[i]->value = x[i];
    }

    std::vector<double> hvp_once(const std::vector<double> &v) {
        const size_t n = var_nodes.size();
        if (v.size() != n)
            throw std::invalid_argument("HessFn.hvp_once: v wrong length");

        {
            py::gil_scoped_release nogil;
            g->resetTangents();
            for (size_t i = 0; i < n; ++i) {
                auto &nd = var_nodes[i];
                set_epoch_value(nd->dot, nd->dot_epoch, g->cur_dot_epoch_,
                                v[i]);
            }
            g->resetForwardPass();
            g->computeForwardPassWithDot();
            g->resetGradients();
            g->resetGradDot();
            set_epoch_value(expr_root->gradient, expr_root->grad_epoch,
                            g->cur_grad_epoch_, 1.0);
            set_epoch_value(expr_root->grad_dot, expr_root->gdot_epoch,
                            g->cur_gdot_epoch_, 0.0);
            g->initiateBackwardPassHVP();
        }
        std::vector<double> Hv(n, 0.0);
        for (size_t i = 0; i < n; ++i)
            Hv[i] = var_nodes[i]->grad_dot;
        return Hv;
    }

    // FAST: __call__(x: ndarray) -> ndarray(n,n)
    Arr2D call_numpy(Arr1D x_in) {
        set_inputs_arr(x_in);
        const size_t n = var_nodes.size();
        Arr2D H({(ssize_t)n, (ssize_t)n});
        auto Hm = H.mutable_unchecked<2>();
        std::vector<double> e(n, 0.0), col;
        for (size_t j = 0; j < n; ++j) {
            std::fill(e.begin(), e.end(), 0.0);
            e[j] = 1.0;
            col = hvp_once(e);
            for (size_t i = 0; i < n; ++i)
                Hm((ssize_t)i, (ssize_t)j) = col[i];
        }
        return H;
    }

    // Compatibility: list/tuple -> list[list]
    py::list operator()(py::object x) {
        set_inputs_seq(x);
        const size_t n = var_nodes.size();
        std::vector<std::vector<double>> Hm(n, std::vector<double>(n, 0.0));
        for (size_t j = 0; j < n; ++j) {
            std::vector<double> ej(n, 0.0);
            ej[j] = 1.0;
            auto col = hvp_once(ej);
            for (size_t i = 0; i < n; ++i)
                Hm[i][j] = col[i];
        }
        py::list mat(n);
        for (size_t i = 0; i < n; ++i) {
            py::list row(n);
            for (size_t j = 0; j < n; ++j)
                row[j] = py::float_(Hm[i][j]);
            mat[i] = std::move(row);
        }
        return mat;
    }

    // FAST: hvp(x: ndarray, v: ndarray) -> ndarray
    Arr1D hvp_numpy(Arr1D x_in, Arr1D v_in) {
        set_inputs_arr(x_in);
        auto v = as_span_1d(v_in);
        if (v.size() != var_nodes.size())
            throw std::invalid_argument("HessFn.hvp: wrong vector length");
        std::vector<double> vv(v.begin(), v.end());
        auto Hv = hvp_once(vv);
        Arr1D out(v_in.request().size);
        auto om = out.mutable_unchecked<1>();
        for (ssize_t i = 0; i < static_cast<ssize_t>(v.size()); ++i)
            om(i) = Hv[i];
        return out;
    }

    // Compatibility: hvp(list, list) -> list
    py::list hvp_seq(py::object x, py::object v) {
        set_inputs_seq(x);
        if (!py::isinstance<py::sequence>(v) || py::isinstance<py::str>(v))
            throw std::invalid_argument(
                "HessFn.hvp: expected list/tuple for v");
        py::sequence sv = v.cast<py::sequence>();
        const size_t n = var_nodes.size();
        if ((size_t)sv.size() != n)
            throw std::invalid_argument("HessFn.hvp: wrong vector length");
        std::vector<double> vv(n, 0.0);
        for (size_t i = 0; i < n; ++i)
            vv[i] = py::cast<double>(sv[i]);
        auto Hv = hvp_once(vv);
        py::list out(n);
        for (size_t i = 0; i < n; ++i)
            out[i] = py::float_(Hv[i]);
        return out;
    }
};

static py::object hessfn_call_positional(HessFn &self, py::args xs) {
    if (self.vector_mode)
        throw std::invalid_argument(
            "HessFn expects a single array/list; use f([x0,...])");
    const size_t n = self.var_nodes.size();
    if (xs.size() != n)
        throw std::invalid_argument("HessFn: wrong number of positional args");
    py::list v(n);
    for (size_t i = 0; i < n; ++i)
        v[i] = py::float_(py::cast<double>(xs[i]));
    return self(v);
}

static std::shared_ptr<HessFn> py_compile_hessian(py::function f, size_t arity,
                                                  bool vector_input) {
    return std::make_shared<HessFn>(f, arity, vector_input);
}
static std::shared_ptr<HessFn> py_hessian_from_example(py::function f,
                                                       py::handle example) {
    if (!is_sequence(example))
        throw std::invalid_argument(
            "hessian_from_example: pass a list/tuple example");
    py::sequence seq = example.cast<py::sequence>();
    return std::make_shared<HessFn>(f, (size_t)seq.size(), true);
}

// ========= Module =========
PYBIND11_MODULE(ad, m) {
    m.doc() =
        "ad optimization and autodiff module (pybind11, NumPy fast paths)";

    // Variable
    py::class_<Variable, std::shared_ptr<Variable>>(m, "Variable")
        .def(py::init<const std::string &, double, double, double>(),
             py::arg("name"), py::arg("value") = 0.0,
             py::arg("lb") = -std::numeric_limits<double>::infinity(),
             py::arg("ub") = std::numeric_limits<double>::infinity())
        .def_property_readonly("name", &Variable::getName)
        .def_property("value", &Variable::getValue, &Variable::setValue)
        .def_property("gradient", &Variable::getGradient,
                      &Variable::setGradient)
        .def("getName", &Variable::getName)
        .def("getValue", &Variable::getValue)
        .def("setValue", &Variable::setValue)
        .def("getGradient", &Variable::getGradient)
        .def("__pow__",
             [](const std::shared_ptr<Variable> &v, double p) {
                 auto g = std::make_shared<ADGraph>();
                 auto ex = make_expr_from_variable(v, g);
                 return expr_pow_any(py::cast(ex), p);
             })
        .def("__rpow__", [](const std::shared_ptr<Variable> &v, double s) {
            auto g = std::make_shared<ADGraph>();
            auto ex = make_expr_from_variable(v, g);
            return scalar_pow_expr(s, ex);
        });

    // Expression
    py::class_<Expression, std::shared_ptr<Expression>>(m, "Expression")
        .def("computeGradient", &Expression::computeGradient)
        .def("computeHessian", &Expression::computeHessian)
        .def("evaluate",
             (double (Expression::*)() const) & Expression::evaluate)
        .def("__str__", [](Expression &e) { return e.toString(); })
        .def("__add__",
             [](const Expression &a, const Expression &b) { return a + b; })
        .def("__sub__",
             [](const Expression &a, const Expression &b) { return a - b; })
        .def("__mul__",
             [](const Expression &a, const Expression &b) { return a * b; })
        .def("__truediv__",
             [](const Expression &a, const Expression &b) { return a / b; })
        .def("__add__", [](const Expression &a, double s) { return a + s; })
        .def("__sub__", [](const Expression &a, double s) { return a - s; })
        .def("__mul__", [](const Expression &a, double s) { return a * s; })
        .def("__truediv__", [](const Expression &a, double s) { return a / s; })
        .def("__radd__", [](const Expression &a, double s) { return a + s; })
        .def("__rsub__",
             [](const Expression &a, double s) {
                 auto lhs = make_expr_from_number(s, a.graph);
                 return (*lhs) - a;
             })
        .def("__rmul__", [](const Expression &a, double s) { return a * s; })
        .def("__rtruediv__",
             [](const Expression &a, double s) { return s / a; })
        .def("__neg__", [](const Expression &a) { return -a; })
        .def("__pow__", [](const std::shared_ptr<Expression> &a,
                           double p) { return expr_pow_any(py::cast(a), p); })
        .def("__pow__",
             [](const std::shared_ptr<Expression> &,
                const std::shared_ptr<Expression> &) {
                 throw std::invalid_argument(
                     "Expression ** Expression not supported; exponent must be "
                     "scalar.");
             })
        .def("__rpow__", [](const std::shared_ptr<Expression> &a, double s) {
            return scalar_pow_expr(s, a);
        });

    // Immediate helpers
    m.def("val", &py_value, "Evaluate f(*xs) or f([xs])");
    // NumPy overload for value
    m.def("val", &py_value_numpy, py::arg("f"), py::arg("x"),
          "Evaluate f(x: ndarray) fast path");

    // --- existing immediate list/tuple/variadic API (kept for compatibility)
    m.def(
        "grad",
        [](py::function f, py::args xs) {
            return py_gradient(f, xs); // your existing function
        },
        "Return gradient of f as a list of floats in input order.");

    // NEW: fast NumPy overload
    m.def("hess", &py_hessian_numpy, py::arg("f"), py::arg("x"),
          "Return Hessian at x (ndarray) as an ndarray[n,n].");
    // --- fast NumPy overload
    m.def("grad", &py_gradient_numpy, py::arg("f"), py::arg("x"),
          "Return gradient at x (ndarray[float64]) as an ndarray.");

    // Unary math
    m.def(
        "sin", [](py::object x) { return unary_dispatch(x, Operator::Sin); },
        "sin(x)");
    m.def(
        "cos", [](py::object x) { return unary_dispatch(x, Operator::Cos); },
        "cos(x)");
    m.def(
        "tan", [](py::object x) { return unary_dispatch(x, Operator::Tan); },
        "tan(x)");
    m.def(
        "exp", [](py::object x) { return unary_dispatch(x, Operator::Exp); },
        "exp(x)");
    m.def(
        "log", [](py::object x) { return unary_dispatch(x, Operator::Log); },
        "log(x)");
    m.def("tanh",
          [](py::object x) { return unary_dispatch(x, Operator::Tanh); },
          "tanh(x)");
    m.def("gelu",
          [](py::object x) { return unary_dispatch(x, Operator::Gelu); },
          "gelu(x)");
    m.def("relu",
          [](py::object x) { return unary_dispatch(x, Operator::Relu); },
          "relu(x)");
    m.def("silu",
          [](py::object x) { return unary_dispatch(x, Operator::Silu); },
          "silu(x)");
    m.def("softmax",
          [](py::object x) { return unary_dispatch(x, Operator::Softmax); },
          "softmax(x) with subgradient 0.5 at ties");

    // max / pow
    m.def(
        "max",
        [](py::object a, py::object b) { return binary_max_dispatch(a, b); },
        "max(a,b) with subgradient 0.5/0.5 at ties");
    m.def(
        "pow", [](py::object x, double p) { return expr_pow_any(x, p); },
        "pow(x,p) builds exp(p*log(x)) symbolically");

    // Compiled gradient & hessian
    py::class_<GradFn, std::shared_ptr<GradFn>>(m, "GradFn")
        .def("__call__", &GradFn::call_numpy, py::arg("x"),
             "Evaluate gradient at x (ndarray) -> ndarray")
        .def("__call__", &GradFn::operator(), py::arg("x_list"),
             "Evaluate gradient at x (list/tuple) -> list")
        .def("__call__", &gradfn_call_positional,
             "Call as g(x0, x1, ...) (compat)")
        .def("expr_str", &GradFn::expr_str)
        .def("__repr__", [](const GradFn &self) {
            return "<GradFn expr=" + self.expr_str() + ">";
        });

    py::class_<HessFn, std::shared_ptr<HessFn>>(m, "HessFn")
        .def("__call__", &HessFn::call_numpy, py::arg("x"),
             "Evaluate Hessian at x (ndarray) -> ndarray[n,n]")
        .def("__call__", &HessFn::operator(), py::arg("x_list"),
             "Evaluate Hessian at x (list/tuple) -> list[list]")
        .def("__call__", &hessfn_call_positional,
             "Call as H(x0, x1, ...) (compat)")
        .def("hvp", &HessFn::hvp_numpy, py::arg("x"), py::arg("v"),
             "HVP at x with v (ndarray) -> ndarray")
        .def("hvp", &HessFn::hvp_seq, py::arg("x_list"), py::arg("v_list"),
             "HVP at x with v (list/tuple) -> list")
        .def("expr_str", &HessFn::expr_str)
        .def("__repr__", [](const HessFn &self) {
            return "<HessFn expr=" + self.expr_str() + ">";
        });

    m.def("sym_grad", &py_compile_gradient, py::arg("f"), py::arg("arity"),
          py::arg("vector_input") = true,
          "Compile a gradient function once; returns GradFn.");
    m.def("gradient_from_example", &py_gradient_from_example, py::arg("f"),
          py::arg("example"),
          "Compile a gradient using a list/tuple example to infer arity.");
    m.def("sym_hess", &py_compile_hessian, py::arg("f"), py::arg("arity"),
          py::arg("vector_input") = true,
          "Compile a Hessian function once; returns HessFn.");
    m.def("hessian_from_example", &py_hessian_from_example, py::arg("f"),
          py::arg("example"),
          "Compile a Hessian using a list/tuple example to infer arity.");
}
