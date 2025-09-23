// OpTraits.h  — FIRST-ORDER ONLY (optimized)
// Drop-in replacement: removes forward_dot / hvp_backward.
// Keeps fast + numerically stable forward/backward.

#pragma once
#include "ADGraph.h"
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

// ---- tiny helpers ----
inline double _safe_div(double a, double b) {
    return (b != 0.0) ? (a / b) : 0.0;
}
inline bool _unary_ok(const ADNode &n)  { return n.inputs.size() == 1 && n.inputs[0] != nullptr; }
inline bool _binary_ok(const ADNode &n) { return n.inputs.size() == 2 && n.inputs[0] && n.inputs[1]; }
inline bool _nary_ok(const ADNode &n)   { return !n.inputs.empty(); }

// ---- default/no-op base ----
template <Operator Op> struct OpTraits {
    static constexpr const char *name = "unknown";
    static inline void forward(ADNode &, ADGraph &) {}
    static inline void backward(ADNode &, ADGraph &) {}
};

// ===== Nullary: cte / var =====
template <> struct OpTraits<Operator::cte> {
    static constexpr const char *name = "cte";
    static inline void forward(ADNode &n, ADGraph &g) {
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, n.value);
    }
    static inline void backward(ADNode &, ADGraph &) {}
};

template <> struct OpTraits<Operator::Var> {
    static constexpr const char *name = "var";
    static inline void forward(ADNode &n, ADGraph &g) {
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, n.value);
    }
    static inline void backward(ADNode &, ADGraph &) {}
};

// =====================================================================
//                       GENERIC UNARY OP PLUMBING
// =====================================================================
template <class Rule, const char *NameLiteral> struct UnaryOp {
    static constexpr const char *name = NameLiteral;

    static inline void forward(ADNode &n, ADGraph &g) {
        if (!_unary_ok(n)) return;
        auto a = n.inputs[0];
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, Rule::f(a->value));
    }

    static inline void backward(ADNode &n, ADGraph &g) {
        if (!_unary_ok(n)) return;
        auto a = n.inputs[0];
        const double df = Rule::df(a->value);
        ensure_epoch_zero(a->gradient, a->grad_epoch, g.cur_grad_epoch_) += n.gradient * df;
    }
};

// Name literals
namespace _op_names {
static constexpr char SIN[]    = "sin";
static constexpr char COS[]    = "cos";
static constexpr char TAN[]    = "tan";
static constexpr char EXP[]    = "exp";
static constexpr char LOG[]    = "log";
static constexpr char ADD[]    = "add";
static constexpr char SUB[]    = "subtract";
static constexpr char MUL[]    = "multiply";
static constexpr char DIV[]    = "divide";
static constexpr char MAXS[]   = "max";
static constexpr char TANH[]   = "tanh";
static constexpr char SILU[]   = "silu";
static constexpr char GELU[]   = "gelu";
static constexpr char SOFTMAX[] = "softmax";
static constexpr char RELU[]   = "relu";
} // namespace _op_names

// ---- Concrete unary rules ----
struct SinRule  { static double f(double x){return std::sin(x);}  static double df(double x){return std::cos(x);} };
struct CosRule  { static double f(double x){return std::cos(x);}  static double df(double x){return -std::sin(x);} };
struct ExpRule  { static double f(double x){return std::exp(x);}  static double df(double x){return std::exp(x);} };

struct LogRule {
    static double f (double x){ return (x > 0.0) ? std::log(x) : std::log(1e-16); }
    static double df(double x){ return (x > 0.0) ? (1.0 / x)   : 0.0; }
};

struct TanRule {
    static double f (double x){ return std::tan(x); }
    static double df(double x){
        const double c = std::cos(x);
        return (std::abs(c) > 1e-12) ? (1.0 / (c * c)) : 0.0;
    }
};

struct TanhRule {
    static double f (double x){ return std::tanh(x); }
    static double df(double x){ const double t = std::tanh(x); return 1.0 - t*t; }
};

inline double _sigmoid(double x) {
    if (x >= 0.0) {
        const double z = std::exp(-x);
        return 1.0 / (1.0 + z);
    } else {
        const double z = std::exp(x);
        return z / (1.0 + z);
    }
}

struct SiLURule {
    static double f (double x){ const double s=_sigmoid(x); return x*s; }
    static double df(double x){ const double s=_sigmoid(x); return s * (1.0 + x * (1.0 - s)); }
};

struct GELURule {
    static double f (double x){
        const double z = x * M_SQRT1_2;
        return 0.5 * x * (1.0 + std::erf(z));
    }
    static double df(double x){
        const double z = x * M_SQRT1_2;
        const double A = std::sqrt(2.0 / M_PI) * std::exp(-0.5 * x * x);
        return 0.5 * (1.0 + std::erf(z)) + 0.5 * x * A;
    }
};

struct ReluRule {
    static double f (double x){ return (x > 0.0) ? x : 0.0; }
    static double df(double x){ return (x > 0.0) ? 1.0 : 0.0; }
};

// =====================================================================
//                    SOFTMAX (component implementation)
// y = softmax(x)[0] — gradient: ∂y/∂x_k = y * (δ_{k0} - y_k)
// =====================================================================
template <> struct OpTraits<Operator::Softmax> {
    static constexpr const char *name = _op_names::SOFTMAX;

    static inline std::vector<double> &tls_vals() { thread_local std::vector<double> v; return v; }
    static inline std::vector<double> &tls_y()    { thread_local std::vector<double> v; return v; }

    static inline void forward(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n)) return;
        const size_t m = n.inputs.size();
        auto &x = tls_vals(); x.resize(m);
        for (size_t i=0;i<m;++i) x[i] = n.inputs[i]->value;

        double xmax = -std::numeric_limits<double>::infinity();
        for (double xi : x) if (xi > xmax) xmax = xi;

        double Z = 0.0;
        for (double xi : x) Z += std::exp(xi - xmax);
        if (Z <= 0.0) Z = 1.0;

        const double yi = std::exp(x[0] - xmax) / Z;
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, yi);
    }

    static inline void backward(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n)) return;
        const size_t m = n.inputs.size();
        auto &x = tls_vals(); x.resize(m);
        for (size_t i=0;i<m;++i) x[i] = n.inputs[i]->value;

        double xmax = -std::numeric_limits<double>::infinity();
        for (double xi : x) if (xi > xmax) xmax = xi;

        auto &y = tls_y(); y.resize(m);
        double Z = 0.0;
        for (size_t i=0;i<m;++i){ y[i] = std::exp(x[i] - xmax); Z += y[i]; }
        if (Z <= 0.0) Z = 1.0;
        for (size_t i=0;i<m;++i) y[i] /= Z;

        const double yi = y[0];
        const double w  = n.gradient;

        for (size_t k=0;k<m;++k) {
            const double dfk = yi * ((k==0)? 1.0 : 0.0) - yi * y[k];
            ensure_epoch_zero(n.inputs[k]->gradient, n.inputs[k]->grad_epoch, g.cur_grad_epoch_) += w * dfk;
        }
    }
};

// =====================================================================
//                    GENERIC BINARY OP PLUMBING
// =====================================================================
template <class Rule, const char *NameLiteral> struct BinaryOp {
    static constexpr const char *name = NameLiteral;

    static inline void forward(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n)) return;
        auto &a = *n.inputs[0], &b = *n.inputs[1];
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, Rule::f(a.value, b.value));
    }

    static inline void backward(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n)) return;
        auto &a = *n.inputs[0], &b = *n.inputs[1];
        const double A = a.value, B = b.value, w = n.gradient;
        ensure_epoch_zero(a.gradient, a.grad_epoch, g.cur_grad_epoch_) += w * Rule::dfa(A, B);
        ensure_epoch_zero(b.gradient, b.grad_epoch, g.cur_grad_epoch_) += w * Rule::dfb(A, B);
    }
};

// Binary rules
struct AddRule {
    static double f(double a, double b)  { return a + b; }
    static double dfa(double, double)    { return 1.0; }
    static double dfb(double, double)    { return 1.0; }
};

struct SubRule {
    static double f(double a, double b)  { return a - b; }
    static double dfa(double, double)    { return 1.0; }
    static double dfb(double, double)    { return -1.0; }
};

struct DivRule {
    static double f  (double a, double b) { return _safe_div(a, b); }
    static double dfa(double, double b)   { return (b != 0.0) ? (1.0 / b) : 0.0; }
    static double dfb(double a, double b) { return (b != 0.0) ? (-a / (b * b)) : 0.0; }
};

template <> struct OpTraits<Operator::Subtract> : BinaryOp<SubRule, _op_names::SUB> {};
template <> struct OpTraits<Operator::Divide>   : BinaryOp<DivRule, _op_names::DIV> {};

// =====================================================================
//                          N-ARY ADD / MUL / MAX
// =====================================================================
template <> struct OpTraits<Operator::Add> : BinaryOp<AddRule, _op_names::ADD> {
    static inline void forward(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n)) return;
        double s = 0.0;
        for (auto &a : n.inputs) s += a->value;
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, s);
    }

    static inline void backward(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n)) return;
        for (auto &a : n.inputs)
            ensure_epoch_zero(a->gradient, a->grad_epoch, g.cur_grad_epoch_) += n.gradient;
    }
};

template <> struct OpTraits<Operator::Multiply> {
    static constexpr const char *name = _op_names::MUL;

    static inline void forward(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n)) return;
        size_t zc = 0, zi = (size_t)-1;
        double prod_nz = 1.0;
        for (size_t i = 0; i < n.inputs.size(); ++i) {
            const double v = n.inputs[i]->value;
            if (v == 0.0) { if (++zc == 1) zi = i; }
            else prod_nz *= v;
        }
        const double y = (zc == 0) ? prod_nz : 0.0;
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, y);
    }

    static inline void backward(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n)) return;

        const size_t m = n.inputs.size();
        size_t zc = 0, zi = (size_t)-1;
        double prod_nz = 1.0;

        for (size_t i = 0; i < m; ++i) {
            const double v = n.inputs[i]->value;
            if (v == 0.0) { if (++zc == 1) zi = i; }
            else prod_nz *= v;
        }

        const double w = n.gradient;
        if (zc >= 2) return;

        if (zc == 1) {
            auto &gacc = ensure_epoch_zero(n.inputs[zi]->gradient, n.inputs[zi]->grad_epoch, g.cur_grad_epoch_);
            gacc += w * prod_nz;
            return;
        }

        // zc == 0
        for (size_t i = 0; i < m; ++i) {
            const double xi = n.inputs[i]->value;
            auto &gacc = ensure_epoch_zero(n.inputs[i]->gradient, n.inputs[i]->grad_epoch, g.cur_grad_epoch_);
            gacc += w * (prod_nz / xi);
        }
    }
};

template <> struct OpTraits<Operator::Max> {
    static constexpr const char *name = _op_names::MAXS;

    static inline void forward(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n)) return;
        const double a = n.inputs[0]->value, b = n.inputs[1]->value;
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, (a >= b ? a : b));
    }

    static inline void backward(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n)) return;
        auto &a = *n.inputs[0], &b = *n.inputs[1];
        if (a.value >= b.value)
            ensure_epoch_zero(a.gradient, a.grad_epoch, g.cur_grad_epoch_) += n.gradient;
        else
            ensure_epoch_zero(b.gradient, b.grad_epoch, g.cur_grad_epoch_) += n.gradient;
    }
};

// =====================================================================
//                     Plug unary ops into OpTraits
// =====================================================================
template <> struct OpTraits<Operator::Sin>   : UnaryOp<SinRule,  _op_names::SIN> {};
template <> struct OpTraits<Operator::Cos>   : UnaryOp<CosRule,  _op_names::COS> {};
template <> struct OpTraits<Operator::Exp>   : UnaryOp<ExpRule,  _op_names::EXP> {};
template <> struct OpTraits<Operator::Log>   : UnaryOp<LogRule,  _op_names::LOG> {};
template <> struct OpTraits<Operator::Tan>   : UnaryOp<TanRule,  _op_names::TAN> {};
template <> struct OpTraits<Operator::Tanh>  : UnaryOp<TanhRule, _op_names::TANH> {};
template <> struct OpTraits<Operator::Silu>  : UnaryOp<SiLURule, _op_names::SILU> {};
template <> struct OpTraits<Operator::Gelu>  : UnaryOp<GELURule, _op_names::GELU> {};
template <> struct OpTraits<Operator::Relu>  : UnaryOp<ReluRule, _op_names::RELU> {};

// Map names function
inline const char *op_name(Operator op) {
    switch (op) {
    case Operator::Add:      return OpTraits<Operator::Add>::name;
    case Operator::Subtract: return OpTraits<Operator::Subtract>::name;
    case Operator::Multiply: return OpTraits<Operator::Multiply>::name;
    case Operator::Divide:   return OpTraits<Operator::Divide>::name;
    case Operator::Sin:      return OpTraits<Operator::Sin>::name;
    case Operator::Cos:      return OpTraits<Operator::Cos>::name;
    case Operator::Tan:      return OpTraits<Operator::Tan>::name;
    case Operator::Exp:      return OpTraits<Operator::Exp>::name;
    case Operator::Log:      return OpTraits<Operator::Log>::name;
    case Operator::Max:      return OpTraits<Operator::Max>::name;
    case Operator::Var:      return OpTraits<Operator::Var>::name;
    case Operator::cte:      return OpTraits<Operator::cte>::name;
    case Operator::Tanh:     return OpTraits<Operator::Tanh>::name;
    case Operator::Silu:     return OpTraits<Operator::Silu>::name;
    case Operator::Gelu:     return OpTraits<Operator::Gelu>::name;
    case Operator::Softmax:  return OpTraits<Operator::Softmax>::name;
    case Operator::Relu:     return OpTraits<Operator::Relu>::name;
    default:                 return "unknown";
    }
}
