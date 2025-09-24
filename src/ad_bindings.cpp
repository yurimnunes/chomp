// ad.cpp — nanobind + NumPy fast paths + GIL release (C++23, de-duplicated)
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

// if you use tsl::robin_* elsewhere, keep includes in your project build.
#include "../../third_party/robin_map.h"
#include "../../third_party/robin_set.h"
#include "../include/ad/ADBindings.h" // pyu, pyconv

// ---------- Module ----------
NB_MODULE(ad, m) {
    m.doc() = "ad optimization and autodiff (nanobind, NumPy fast paths, "
              "cached, C++23, DRY)";

    // Variable
    nb::class_<Variable>(m, "Variable")
        .def(nb::init<const std::string &, double, double, double>(), "name"_a,
             "value"_a = 0.0, "lb"_a = -std::numeric_limits<double>::infinity(),
             "ub"_a = std::numeric_limits<double>::infinity())
        .def_prop_ro("name", &Variable::getName)
        .def_prop_rw("value", &Variable::getValue, &Variable::setValue)
        .def_prop_rw("gradient", &Variable::getGradient, &Variable::setGradient)
        .def("getName", &Variable::getName)
        .def("getValue", &Variable::getValue)
        .def("setValue", &Variable::setValue)
        .def("getGradient", &Variable::getGradient)
        .def("__pow__",
             [](const std::shared_ptr<Variable> &v, double p) {
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
                 auto lhs = make_const_expr(s, a.graph);
                 return (*lhs) - a;
             })
        .def("__rmul__", [](const Expression &a, double s) { return a * s; })
        .def("__rtruediv__",
             [](const Expression &a, double s) { return s / a; })
        .def("__neg__", [](const Expression &a) { return -a; })
        .def("__pow__", [](const std::shared_ptr<Expression> &a,
                           double p) { return expr_pow_any(nb::cast(a), p); })
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

    // immediate APIs
    m.def(
        "valgrad",
        [](nb::object f, Arr1D x) { return py_value_grad_numpy(f, x); }, "f"_a,
        "x"_a, "Return (f(x), grad f(x)) fast path (ndarray, cached).");

    m.def(
        "hess", [](nb::object f, Arr1D x) { return py_hessian_numpy(f, x); },
        "f"_a, "x"_a, "Return Hessian at x (ndarray[n,n], cached).");

    m.def(
        "grad", [](nb::object f, nb::args xs) { return py_gradient(f, xs); },
        "Return gradient of f as a list of floats in input order.");

    m.def(
        "grad", [](nb::object f, Arr1D x) { return py_gradient_numpy(f, x); },
        "f"_a, "x"_a, "Gradient at x (ndarray[float64], cached).");

    // unary registrations (DRY via lambda)
    auto reg_unary = [&](const char *name, Operator op, const char *doc) {
        m.def(
            name,
            [op](nb::object x) { return unary_dispatch(std::move(x), op); },
            doc);
    };
    reg_unary("sin", Operator::Sin, "sin(x)");
    reg_unary("cos", Operator::Cos, "cos(x)");
    reg_unary("tan", Operator::Tan, "tan(x)");
    reg_unary("exp", Operator::Exp, "exp(x)");
    reg_unary("log", Operator::Log, "log(x)");
    reg_unary("tanh", Operator::Tanh, "tanh(x)");
    reg_unary("silu", Operator::Silu, "silu(x) = x * sigmoid(x)");
    reg_unary("relu", Operator::Relu, "relu(x) = max(0,x)");
    reg_unary("softmax", Operator::Softmax, "softmax(x) over provided inputs");
    reg_unary("gelu", Operator::Gelu, "gelu(x)");

    // max / pow
    m.def(
        "max",
        [](nb::object a, nb::object b) { return binary_max_dispatch(a, b); },
        "max(a,b) with subgradient 0.5/0.5 at ties");
    m.def(
        "pow", [](nb::object x, double p) { return expr_pow_any(x, p); },
        "pow(x,p) builds exp(p*log(x)) when p is not integer");

    // Compiled gradient
    nb::class_<GradFn>(m, "GradFn")
        .def("__call__", &GradFn::call_numpy, "x"_a,
             "Evaluate gradient at x (ndarray)->ndarray")
        .def("__call__", &GradFn::operator(), "x_list"_a,
             "Evaluate gradient at x (list/tuple)->list")
        .def("expr_str", &GradFn::expr_str)
        .def("value", &GradFn::value_numpy, "x"_a,
             "Evaluate f(x) (ndarray)->float")
        .def("value_grad", &GradFn::value_grad_numpy, "x"_a,
             "Return (f(x), grad f(x)) (ndarray)")
        .def("__repr__", [](const GradFn &self) {
            return "<GradFn expr=" + self.expr_str() + ">";
        });

    m.def(
        "sym_grad",
        [](nb::object f, ssize_t arity, bool vector_input) {
            return std::make_shared<GradFn>(f, (size_t)arity, vector_input);
        },
        "f"_a, "arity"_a, "vector_input"_a = true,
        "Compile a gradient function once; returns GradFn.",
        nb::rv_policy::take_ownership);

    m.def(
        "gradient_from_example",
        [](nb::object f, nb::handle example) {
            if (!is_sequence(example))
                throw std::invalid_argument(
                    "gradient_from_example: pass a list/tuple example");
            nb::sequence seq = nb::cast<nb::sequence>(example);
            return std::make_shared<GradFn>(f, (size_t)nb::len(seq), true);
        },
        "f"_a, "example"_a,
        "Compile a gradient using a list/tuple example to infer arity.",
        nb::rv_policy::take_ownership);

    m.def(
        "sym_grad_cached",
        [](nb::object f, ssize_t arity, bool vector_input) {
            return get_or_make_grad(f, (size_t)arity, vector_input);
        },
        "f"_a, "arity"_a, "vector_input"_a = true,
        "Get or compile & cache a GradFn keyed by (f,arity,vector_input).",
        nb::rv_policy::take_ownership);

    // Compiled Hessian
    nb::class_<HessFn>(m, "HessFn")
        .def("__call__", &HessFn::call_numpy, "x"_a,
             "Evaluate Hessian at x (ndarray)->ndarray[n,n]")
        .def("__call__", &HessFn::operator(), "x_list"_a,
             "Evaluate Hessian at x (list[list])")
        .def("hvp", &HessFn::hvp_numpy, "x"_a, "v"_a,
             "HVP at x with v (ndarray)->ndarray")
        .def("expr_str", &HessFn::expr_str)
        .def("__repr__", [](const HessFn &self) {
            return "<HessFn expr=" + self.expr_str() + ">";
        });

    m.def(
        "sym_hess",
        [](nb::object f, ssize_t arity, bool vector_input) {
            return std::make_shared<HessFn>(f, (size_t)arity, vector_input);
        },
        "f"_a, "arity"_a, "vector_input"_a = true,
        "Compile a Hessian function once; returns HessFn.",
        nb::rv_policy::take_ownership);

    m.def(
        "hessian_from_example",
        [](nb::object f, nb::handle example) {
            if (!is_sequence(example))
                throw std::invalid_argument(
                    "hessian_from_example: pass a list/tuple example");
            nb::sequence seq = nb::cast<nb::sequence>(example);
            return std::make_shared<HessFn>(f, (size_t)nb::len(seq), true);
        },
        "f"_a, "example"_a,
        "Compile a Hessian using a list/tuple example to infer arity.",
        nb::rv_policy::take_ownership);

    m.def(
        "sym_hess_cached",
        [](nb::object f, ssize_t arity, bool vector_input) {
            return get_or_make_hess(f, (size_t)arity, vector_input);
        },
        "f"_a, "arity"_a, "vector_input"_a = true,
        "Get or compile & cache a HessFn keyed by (f,arity,vector_input).",
        nb::rv_policy::take_ownership);

    m.def(
        "sym_laghess",
        [](nb::object f, std::vector<nb::object> cI, std::vector<nb::object> cE,
           ssize_t arity, bool vec) {
            return std::make_shared<LagHessFn>(f, cI, cE, (size_t)arity, vec);
        },
        "f"_a, "cI"_a, "cE"_a, "arity"_a, "vector_input"_a = true);

    // Lagrangian Hessian
    nb::class_<LagHessFn>(m, "LagHessFn")
        .def(nb::init<nb::object, const std::vector<nb::object> &,
                      const std::vector<nb::object> &, size_t, bool>(),
             "f"_a, "cI"_a, "cE"_a, "arity"_a, "vector_input"_a = true)
        .def("set_state", &LagHessFn::set_state_numpy, "x"_a, "lam"_a, "nu"_a)
        .def("hess", &LagHessFn::hess_numpy, "x"_a, "lam"_a, "nu"_a,
             "Return full Hessian (ndarray[n,n])")
        .def("hvp_numpy", &LagHessFn::hvp_numpy, "x"_a, "lam"_a, "nu"_a, "v"_a,
             "Return H·v (ndarray[n])")
        .def("hvp_multi_numpy", &LagHessFn::hvp_multi_numpy, "x"_a, "lam"_a,
             "nu"_a, "V"_a, "Return H·V (loops over columns; ndarray[n,k])");
    // batch fused val+grad
    m.def(
        "batch_valgrad",
        [](const std::vector<std::shared_ptr<GradFn>> &gfs, Arr1D x) {
            return batch_value_grad_from_gradfns(gfs, x);
        },
        "gfs"_a, "x"_a,
        "Fused (vals, J) for many compiled GradFn at the same x. "
        "Returns (vals: ndarray[m], J: ndarray[m,n]).");

    m.def(
        "batch_valgrad",
        [](nb::list funcs, Arr1D x) {
            return batch_value_grad_from_callables(funcs, x);
        },
        "funcs"_a, "x"_a,
        "Fused (vals, J) for many Python functions at the same x. "
        "Functions are compiled/cached to GradFn internally.");

    m.def(
        "batch_valgrad",
        [](nb::sequence grads, Arr1D x) {
            return batch_value_grad_numpy(grads, x);
        },
        "grads"_a, "x"_a,
        "Fused, parallel (f_j(x), grad f_j(x)) for a list of compiled GradFn.\n"
        "Returns (vals: ndarray[m], J: ndarray[m,n]).");
}
