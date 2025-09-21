// OpTraits.h
#pragma once
#include "ADGraph.h" // declares ADNode, ADGraph, set_epoch_value, touch_epoch, ensure_epoch_zero
#include <cmath>
#include <cstddef>
#include <vector>

// enum class Operator { ... }  // assumed declared elsewhere

// ---- tiny helpers (unchanged) ----
inline double _safe_div(double a, double b) {
    return (b != 0.0) ? (a / b) : 0.0;
}
inline bool _unary_ok(const ADNode &n) {
    return n.inputs.size() == 1 && n.inputs[0] != nullptr;
}
inline bool _binary_ok(const ADNode &n) {
    return n.inputs.size() == 2 && n.inputs[0] && n.inputs[1];
}
inline bool _nary_ok(const ADNode &n) { return !n.inputs.empty(); }

// ---- default/no-op base ----
template <Operator Op> struct OpTraits {
    static constexpr const char *name = "unknown";
    static inline void forward(ADNode &, ADGraph &) {}
    static inline void forward_dot(ADNode &, ADGraph &) {}
    static inline void backward(ADNode &, ADGraph &) {}
    static inline void hvp_backward(ADNode &, ADGraph &) {}
};

// ===== Nullary: cte / var (kept minimal) =====
template <> struct OpTraits<Operator::cte> {
    static constexpr const char *name = "cte";
    static inline void forward(ADNode &n, ADGraph &g) {
        touch_epoch(n.val_epoch, g.cur_val_epoch_);
    }
    static inline void forward_dot(ADNode &n, ADGraph &g) {
        touch_epoch(n.dot_epoch, g.cur_dot_epoch_);
        touch_epoch(n.val_epoch, g.cur_val_epoch_);
    }
    static inline void backward(ADNode &, ADGraph &) {}
    static inline void hvp_backward(ADNode &, ADGraph &) {}
};
template <> struct OpTraits<Operator::Var> {
    static constexpr const char *name = "var";
    static inline void forward(ADNode &n, ADGraph &g) {
        touch_epoch(n.val_epoch, g.cur_val_epoch_);
    }
    static inline void forward_dot(ADNode &n, ADGraph &g) {
        touch_epoch(n.dot_epoch, g.cur_dot_epoch_);
        touch_epoch(n.val_epoch, g.cur_val_epoch_);
    }
    static inline void backward(ADNode &, ADGraph &) {}
    static inline void hvp_backward(ADNode &, ADGraph &) {}
};

// =====================================================================
//                       GENERIC UNARY OP PLUMBING
// =====================================================================
//
// Define a Rule with:
//   static double f(double x);
//   static double df(double x);
//   static double d2(double x);         // for HVP
// Optional:
//   static inline void forward_dot(ADNode&, ADGraph&);  // custom (e.g.,
//   tan/log with guards)
//
// Then:   template<> struct OpTraits<Op::Sin> : UnaryOp<SinRule, "sin"> {};
//
template <class Rule, const char *NameLiteral> struct UnaryOp {
    static constexpr const char *name = NameLiteral;

    static inline void forward(ADNode &n, ADGraph &g) {
        if (!_unary_ok(n))
            return;
        auto a = n.inputs[0];
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_,
                        Rule::f(a->value));
    }

    static inline void forward_dot(ADNode &n, ADGraph &g) {
        if (!_unary_ok(n))
            return;
        if constexpr (requires(ADNode &nn, ADGraph &gg) {
                          Rule::forward_dot(nn, gg);
                      }) {
            Rule::forward_dot(n, g); // custom fast-path
        } else {
            auto a = n.inputs[0];
            set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_,
                            Rule::df(a->value) * a->dot);
            touch_epoch(n.val_epoch, g.cur_val_epoch_);
        }
    }

    static inline void backward(ADNode &n, ADGraph &g) {
        if (!_unary_ok(n))
            return;
        auto a = n.inputs[0];
        ensure_epoch_zero(a->gradient, a->grad_epoch, g.cur_grad_epoch_) +=
            n.gradient * Rule::df(a->value);
    }

    static inline void hvp_backward(ADNode &n, ADGraph &g) {
        if (!_unary_ok(n))
            return;
        auto a = n.inputs[0];
        auto &gacc =
            ensure_epoch_zero(a->gradient, a->grad_epoch, g.cur_grad_epoch_);
        auto &gdacc =
            ensure_epoch_zero(a->grad_dot, a->gdot_epoch, g.cur_gdot_epoch_);
        const double x = a->value, xdot = a->dot, df = Rule::df(x),
                     d2 = Rule::d2(x);
        gacc += n.gradient * df;
        gdacc += n.grad_dot * df + n.gradient * d2 * xdot;
    }
};

// Name literals (addresses must be stable)
namespace _op_names {
static constexpr char SIN[] = "sin";
static constexpr char COS[] = "cos";
static constexpr char TAN[] = "tan";
static constexpr char EXP[] = "exp";
static constexpr char LOG[] = "log";
static constexpr char ADD[] = "add";
static constexpr char SUB[] = "subtract";
static constexpr char MUL[] = "multiply";
static constexpr char DIV[] = "divide";
static constexpr char MAXS[] = "max";
static constexpr char TANH[] = "tanh";
static constexpr char SILU[] = "silu";
static constexpr char GELU[] = "gelu";
static constexpr char SOFTMAX[] = "softmax";
static constexpr char RELU[] = "relu";

} // namespace _op_names

// ---- Concrete unary rules ----
struct SinRule {
    static double f(double x) { return std::sin(x); }
    static double df(double x) { return std::cos(x); }
    static double d2(double x) { return -std::sin(x); }
};
struct CosRule {
    static double f(double x) { return std::cos(x); }
    static double df(double x) { return -std::sin(x); }
    static double d2(double x) { return -std::cos(x); }
};
struct ExpRule {
    static double f(double x) { return std::exp(x); }
    static double df(double x) { return std::exp(x); }
    static double d2(double x) { return std::exp(x); }
};

// Log has domain guard + custom dot to avoid repeated exp/log
struct LogRule {
    static double f(double x) { return std::log(x); }
    static double df(double x) { return (x != 0.0) ? (1.0 / x) : 0.0; }
    static double d2(double x) { return (x != 0.0) ? (-1.0 / (x * x)) : 0.0; }
    static inline void forward_dot(ADNode &n, ADGraph &g) {
        auto a = n.inputs[0];
        const double x = a->value;
        set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_,
                        x != 0.0 ? (a->dot / x) : 0.0);
        touch_epoch(n.val_epoch, g.cur_val_epoch_);
    }
};

// Tan with stable cos guard + custom dot
struct TanRule {
    static double f(double x) { return std::tan(x); }
    static double df(double x) {
        const double c = std::cos(x);
        return (c != 0.0) ? (1.0 / (c * c)) : 0.0;
    } // sec^2
    static double d2(double x) {
        const double s = std::sin(x), c = std::cos(x);
        return (c != 0.0) ? (2.0 * s / (c * c * c)) : 0.0;
    } // 2 sec^2 tan = 2 sin/cos^3
    static inline void forward_dot(ADNode &n, ADGraph &g) {
        auto a = n.inputs[0];
        const double c = std::cos(a->value);
        set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_,
                        (c != 0.0) ? (a->dot / (c * c)) : 0.0);
        touch_epoch(n.val_epoch, g.cur_val_epoch_);
    }
};

// -------- tanh --------
struct TanhRule {
    static double f(double x)  { return std::tanh(x); }
    static double df(double x) { const double t = std::tanh(x); return 1.0 - t*t; }                 // sech^2
    static double d2(double x) { const double t = std::tanh(x); const double s2 = 1.0 - t*t; return -2.0*t*s2; }
};

// Helper: stable sigmoid
inline double _sigmoid(double x) {
    // numerically stable sigmoid
    if (x >= 0.0) {
        const double z = std::exp(-x);
        return 1.0 / (1.0 + z);
    } else {
        const double z = std::exp(x);
        return z / (1.0 + z);
    }
}

// -------- SiLU (Swish) : x * sigmoid(x) --------
struct SiLURule {
    static double f(double x)  { const double s = _sigmoid(x); return x * s; }
    static double df(double x) {
        const double s = _sigmoid(x);                // s = σ
        return s * (1.0 + x * (1.0 - s));            // σ + x σ(1-σ)
    }
    static double d2(double x) {
        const double s  = _sigmoid(x);
        const double sp = s * (1.0 - s);             // σ'
        // d2 = 2σ(1-σ) + x σ(1-σ)(1 - 2σ)
        return sp * (2.0 + x * (1.0 - 2.0 * s));
    }
};

// -------- GELU (exact, erf-based) --------
//   gelu(x) = 0.5 x (1 + erf(x / sqrt(2)))
//   d  = 0.5 (1 + erf(z)) + 0.5 x * sqrt(2/pi) * exp(-x^2/2)
//   d2 = sqrt(2/pi) * exp(-x^2/2) * (1 - x^2/2)
struct GELURule {
    static double f(double x)  {
        const double z = x * M_SQRT1_2;              // x / sqrt(2)
        return 0.5 * x * (1.0 + std::erf(z));
    }
    static double df(double x) {
        const double z  = x * M_SQRT1_2;
        const double A  = std::sqrt(2.0 / M_PI) * std::exp(-0.5 * x * x);
        return 0.5 * (1.0 + std::erf(z)) + 0.5 * x * A;
    }
    static double d2(double x) {
        const double A = std::sqrt(2.0 / M_PI) * std::exp(-0.5 * x * x);
        return A * (1.0 - 0.5 * x * x);
    }
};


struct ReluRule {
    static double f(double x)  { return (x > 0.0) ? x : 0.0; }
    static double df(double x) { return (x > 0.0) ? 1.0 : 0.0; }
    static double d2(double x) { return 0.0; }
};

// =====================================================================
// SOFTMAX (component): value = softmax(inputs)[0]
// Assumes first input is the "component-of-interest" x_i; the full set
// of inputs represents the vector x. Uses a max-shift for stability.
// Gradients/HVP propagate to all inputs.
// =====================================================================
template <> struct OpTraits<Operator::Softmax> {
    static constexpr const char *name = _op_names::SOFTMAX;

    static inline std::vector<double> &tls_vals(){ thread_local std::vector<double> v; return v; }
    static inline std::vector<double> &tls_dots(){ thread_local std::vector<double> v; return v; }
    static inline std::vector<double> &tls_y()   { thread_local std::vector<double> v; return v; }

    static inline void forward(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n)) return;
        const size_t m = n.inputs.size();
        auto &x = tls_vals(); x.resize(m);
        double xmax = -std::numeric_limits<double>::infinity();
        for (size_t i=0;i<m;++i){ x[i]=n.inputs[i]->value; if (x[i] > xmax) xmax = x[i]; }
        double Z = 0.0;
        for (size_t i=0;i<m;++i) Z += std::exp(x[i] - xmax);
        const double yi = std::exp(x[0] - xmax) / (Z > 0.0 ? Z : 1.0); // component for first input
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, yi);
    }

    static inline void forward_dot(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n)) return;
        const size_t m = n.inputs.size();
        auto &x  = tls_vals(); x.resize(m);
        auto &xd = tls_dots(); xd.resize(m);
        for (size_t i=0;i<m;++i){ x[i]=n.inputs[i]->value; xd[i]=n.inputs[i]->dot; }

        double xmax = -std::numeric_limits<double>::infinity();
        for (size_t i=0;i<m;++i) if (x[i] > xmax) xmax = x[i];

        auto &y = tls_y(); y.resize(m);
        double Z = 0.0;
        for (size_t i=0;i<m;++i){ y[i] = std::exp(x[i] - xmax); Z += y[i]; }
        if (Z <= 0.0) Z = 1.0;
        for (size_t i=0;i<m;++i) y[i] /= Z;

        const double yi   = y[0];
        double sdot = 0.0; for (size_t j=0;j<m;++j) sdot += y[j] * xd[j]; // y · xdot
        const double dot = yi * (xd[0] - sdot);

        set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_, dot);
        touch_epoch(n.val_epoch, g.cur_val_epoch_);
    }

    static inline void backward(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n)) return;
        const size_t m = n.inputs.size();
        auto &x = tls_vals(); x.resize(m);
        for (size_t i=0;i<m;++i) x[i]=n.inputs[i]->value;

        double xmax = -std::numeric_limits<double>::infinity();
        for (size_t i=0;i<m;++i) if (x[i] > xmax) xmax = x[i];

        auto &y = tls_y(); y.resize(m);
        double Z = 0.0; for (size_t i=0;i<m;++i){ y[i]=std::exp(x[i]-xmax); Z+=y[i]; }
        if (Z <= 0.0) Z = 1.0;
        for (size_t i=0;i<m;++i) y[i] /= Z;

        const double yi = y[0];
        const double w  = n.gradient;

        // ∂y_i/∂x_k = y_i (δ_{ik} - y_k)
        for (size_t k=0;k<m;++k) {
            const double dfk = yi * ((k==0)? 1.0 : 0.0) - yi * y[k];
            ensure_epoch_zero(n.inputs[k]->gradient, n.inputs[k]->grad_epoch, g.cur_grad_epoch_) += w * dfk;
        }
    }

    static inline void hvp_backward(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n)) return;
        const size_t m = n.inputs.size();
        auto &x  = tls_vals(); x.resize(m);
        auto &xd = tls_dots(); xd.resize(m);
        for (size_t i=0;i<m;++i){ x[i]=n.inputs[i]->value; xd[i]=n.inputs[i]->dot; }

        double xmax = -std::numeric_limits<double>::infinity();
        for (size_t i=0;i<m;++i) if (x[i] > xmax) xmax = x[i];

        auto &y = tls_y(); y.resize(m);
        double Z = 0.0; for (size_t i=0;i<m;++i){ y[i]=std::exp(x[i]-xmax); Z += y[i]; }
        if (Z <= 0.0) Z = 1.0;
        for (size_t i=0;i<m;++i) y[i] /= Z;

        const double yi = y[0];
        const double w  = n.gradient;
        const double wd = n.grad_dot;

        // sdot = Σ_j y_j * xdot_j
        double sdot = 0.0; for (size_t j=0;j<m;++j) sdot += y[j] * xd[j];

        for (size_t k=0;k<m;++k) {
            // df_k = y_i (δ_{ik} - y_k)
            const double dfk = yi * ((k==0)? 1.0 : 0.0) - yi * y[k];

            // (H · xdot)_k :
            // if k == i: yi * (1 - 2 yi) * (xd_i - sdot)
            // else     : yi * y_k * (2 sdot - xd_i - xd_k)
            double Hv_k;
            if (k == 0) {
                Hv_k = yi * (1.0 - 2.0 * yi) * (xd[0] - sdot);
            } else {
                Hv_k = yi * y[k] * (2.0 * sdot - xd[0] - xd[k]);
            }

            auto &gacc  = ensure_epoch_zero(n.inputs[k]->gradient, n.inputs[k]->grad_epoch, g.cur_grad_epoch_);
            auto &gdacc = ensure_epoch_zero(n.inputs[k]->grad_dot, n.inputs[k]->gdot_epoch, g.cur_gdot_epoch_);
            gacc  += w  * dfk;
            gdacc += wd * dfk + w * Hv_k;
        }
    }
};


// ---- Plug unary ops into OpTraits via the generic plumbing ----
template <>
struct OpTraits<Operator::Sin> : UnaryOp<SinRule, _op_names::SIN> {};
template <>
struct OpTraits<Operator::Cos> : UnaryOp<CosRule, _op_names::COS> {};
template <>
struct OpTraits<Operator::Exp> : UnaryOp<ExpRule, _op_names::EXP> {};
template <>
struct OpTraits<Operator::Log> : UnaryOp<LogRule, _op_names::LOG> {};
template <>
struct OpTraits<Operator::Tan> : UnaryOp<TanRule, _op_names::TAN> {};
template <>
struct OpTraits<Operator::Tanh> : UnaryOp<TanhRule, _op_names::TANH> {};
template <>
struct OpTraits<Operator::Silu> : UnaryOp<SiLURule, _op_names::SILU> {};
template <>
struct OpTraits<Operator::Gelu> : UnaryOp<GELURule, _op_names::GELU> {};
template <>
struct OpTraits<Operator::Relu> : UnaryOp<ReluRule, _op_names::RELU> {};

// =====================================================================
//                      GENERIC BINARY OP PLUMBING
// =====================================================================
//
// A BinaryRule must provide:
//   static double f(double a,double b);
//   static double dfa(double a,double b);   // ∂f/∂a
//   static double dfb(double a,double b);   // ∂f/∂b
//   static double d2aa(double a,double b);  // ∂²f/∂a²
//   static double d2ab(double a,double b);  // ∂²f/∂a∂b
//   static double d2bb(double a,double b);  // ∂²f/∂b²
// Optional:
//   static inline void forward_dot(ADNode&, ADGraph&);  // custom fast path
//
template <class Rule, const char *NameLiteral> struct BinaryOp {
    static constexpr const char *name = NameLiteral;

    static inline void forward(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n))
            return;
        auto &a = *n.inputs[0], &b = *n.inputs[1];
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_,
                        Rule::f(a.value, b.value));
    }

    static inline void forward_dot(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n))
            return;
        auto &a = *n.inputs[0], &b = *n.inputs[1];
        if constexpr (requires(ADNode &nn, ADGraph &gg) {
                          Rule::forward_dot(nn, gg);
                      }) {
            Rule::forward_dot(n, g);
        } else {
            const double A = a.value, B = b.value;
            set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_,
                            Rule::dfa(A, B) * a.dot + Rule::dfb(A, B) * b.dot);
            touch_epoch(n.val_epoch, g.cur_val_epoch_);
        }
    }

    static inline void backward(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n))
            return;
        auto &a = *n.inputs[0], &b = *n.inputs[1];
        const double A = a.value, B = b.value, w = n.gradient;
        ensure_epoch_zero(a.gradient, a.grad_epoch, g.cur_grad_epoch_) +=
            w * Rule::dfa(A, B);
        ensure_epoch_zero(b.gradient, b.grad_epoch, g.cur_grad_epoch_) +=
            w * Rule::dfb(A, B);
    }

    static inline void hvp_backward(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n))
            return;
        auto &a = *n.inputs[0], &b = *n.inputs[1];
        const double A = a.value, B = b.value, Ad = a.dot, Bd = b.dot;
        auto &ga =
            ensure_epoch_zero(a.gradient, a.grad_epoch, g.cur_grad_epoch_);
        auto &gb =
            ensure_epoch_zero(b.gradient, b.grad_epoch, g.cur_grad_epoch_);
        auto &gda =
            ensure_epoch_zero(a.grad_dot, a.gdot_epoch, g.cur_gdot_epoch_);
        auto &gdb =
            ensure_epoch_zero(b.grad_dot, b.gdot_epoch, g.cur_gdot_epoch_);
        const double w = n.gradient, wd = n.grad_dot;

        // First-order adjoints
        ga += w * Rule::dfa(A, B);
        gb += w * Rule::dfb(A, B);

        // HVP propagation: gdot_i += wdot * dfi + w * (H_iA * A_dot + H_iB *
        // B_dot)
        gda += wd * Rule::dfa(A, B) +
               w * (Rule::d2aa(A, B) * Ad + Rule::d2ab(A, B) * Bd);
        gdb += wd * Rule::dfb(A, B) +
               w * (Rule::d2ab(A, B) * Ad + Rule::d2bb(A, B) * Bd);
    }
};

// ---- Concrete binary rules ----
// Add: f = a + b
struct AddRule {
    static double f(double a, double b) { return a + b; }
    static double dfa(double, double) { return 1.0; }
    static double dfb(double, double) { return 1.0; }
    static double d2aa(double, double) { return 0.0; }
    static double d2ab(double, double) { return 0.0; }
    static double d2bb(double, double) { return 0.0; }
};
// Subtract: f = a - b
struct SubRule {
    static double f(double a, double b) { return a - b; }
    static double dfa(double, double) { return 1.0; }
    static double dfb(double, double) { return -1.0; }
    static double d2aa(double, double) { return 0.0; }
    static double d2ab(double, double) { return 0.0; }
    static double d2bb(double, double) { return 0.0; }
};
// Divide: f = a / b
struct DivRule {
    static double f(double a, double b) { return _safe_div(a, b); }
    static double dfa(double, double b) { return (b != 0.0) ? (1.0 / b) : 0.0; }
    static double dfb(double a, double b) {
        return (b != 0.0) ? (-a / (b * b)) : 0.0;
    }
    static double d2aa(double, double) { return 0.0; }
    static double d2ab(double, double b) {
        return (b != 0.0) ? (-1.0 / (b * b)) : 0.0;
    }
    static double d2bb(double a, double b) {
        return (b != 0.0) ? (2.0 * a / (b * b * b)) : 0.0;
    }
    static inline void forward_dot(ADNode &n, ADGraph &g) {
        auto &a = *n.inputs[0], &b = *n.inputs[1];
        const double d = b.value;
        set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_,
                        d != 0.0 ? ((a.dot * d - a.value * b.dot) / (d * d))
                                 : 0.0);
        touch_epoch(n.val_epoch, g.cur_val_epoch_);
    }
};

// ---- Hook rules to OpTraits ----
// template <> struct OpTraits<Operator::Add>      : BinaryOp<AddRule,
// _op_names::ADD> {};
template <>
struct OpTraits<Operator::Subtract> : BinaryOp<SubRule, _op_names::SUB> {};
template <>
struct OpTraits<Operator::Divide> : BinaryOp<DivRule, _op_names::DIV> {};

// =====================================================================
//                    N-ARY SUM (generic)  — stays tiny
// =====================================================================
template <> struct OpTraits<Operator::Add> : BinaryOp<AddRule, _op_names::ADD> {
    // n-ary sum specialization (overrides BinaryOp’s forward/backward/hvp)
    static inline void forward(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n))
            return;
        double s = 0.0;
        for (auto &a : n.inputs)
            s += a->value;
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, s);
    }
    static inline void forward_dot(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n))
            return;
        double sd = 0.0;
        for (auto &a : n.inputs)
            sd += a->dot;
        set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_, sd);
        touch_epoch(n.val_epoch, g.cur_val_epoch_);
    }
    static inline void backward(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n))
            return;
        for (auto &a : n.inputs)
            ensure_epoch_zero(a->gradient, a->grad_epoch, g.cur_grad_epoch_) +=
                n.gradient;
    }
    static inline void hvp_backward(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n))
            return;
        for (auto &a : n.inputs) {
            ensure_epoch_zero(a->gradient, a->grad_epoch, g.cur_grad_epoch_) +=
                n.gradient;
            ensure_epoch_zero(a->grad_dot, a->gdot_epoch, g.cur_gdot_epoch_) +=
                n.grad_dot;
        }
    }
};

// =====================================================================
//                 MULTIPLY (n-ary) — keep your optimized version
// =====================================================================
// =====================================================================
//                 MULTIPLY (n-ary) — O(m) with zero-aware math
// =====================================================================
template <> struct OpTraits<Operator::Multiply> {
    static constexpr const char *name = _op_names::MUL;

    // Thread-local scratch (kept tiny; we only need indices)
    static inline std::vector<double> &tls_vals() { thread_local std::vector<double> v; return v; }
    static inline std::vector<double> &tls_dots() { thread_local std::vector<double> v; return v; }

    // Helper: count zeros, track index of the (unique) zero if any, and product of nonzeros
    static inline void analyze_inputs(const std::vector<std::shared_ptr<ADNode>> &ins,
                                      size_t &zero_count, size_t &zero_idx,
                                      double &prod_nz) {
        zero_count = 0; zero_idx = (size_t)-1; prod_nz = 1.0;
        const size_t m = ins.size();
        for (size_t i=0;i<m;++i) {
            const double v = ins[i]->value;
            if (v == 0.0) {
                if (++zero_count == 1) zero_idx = i;
            } else {
                prod_nz *= v;
            }
        }
    }

    // y = Π x_i
    static inline void forward(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n)) return;
        size_t zc, zi; double prod_nz;
        analyze_inputs(n.inputs, zc, zi, prod_nz);
        const double y = (zc == 0) ? prod_nz : 0.0;
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, y);
    }

    // ẏ = y * Σ (ẋ_i / x_i), but handle zeros:
    //  - if zc>=2 -> y=0 and local Jacobian is zero ⇒ ẏ=0
    //  - if zc==1 -> only term from the zero index survives: ẏ = ẋ_z * Π_{j≠z} x_j
    //  - if zc==0 -> formula above
    static inline void forward_dot(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n)) return;
        const size_t m = n.inputs.size();

        size_t zc, zi; double prod_nz;
        analyze_inputs(n.inputs, zc, zi, prod_nz);

        double ydot = 0.0;
        if (zc >= 2) {
            ydot = 0.0;
        } else if (zc == 1) {
            // y = 0 but dy/dx_z = Π_{j≠z} x_j
            ydot = n.inputs[zi]->dot * prod_nz;
        } else {
            // regular case
            double sum = 0.0;
            for (size_t i=0;i<m;++i) {
                const double xi = n.inputs[i]->value;
                // xi != 0 here
                sum += n.inputs[i]->dot / xi;
            }
            ydot = prod_nz * sum;
        }

        set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_, ydot);
        touch_epoch(n.val_epoch, g.cur_val_epoch_);
    }

    // ∂y/∂x_i:
    //  - zc>=2 -> 0
    //  - zc==1 -> only i==zi gets Π_{j≠zi} x_j
    //  - zc==0 -> y / x_i
    static inline void backward(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n)) return;
        const size_t m = n.inputs.size();

        size_t zc, zi; double prod_nz;
        analyze_inputs(n.inputs, zc, zi, prod_nz);

        const double w = n.gradient;
        if (zc >= 2) {
            // all zero; nothing to do
            return;
        } else if (zc == 1) {
            auto &gacc = ensure_epoch_zero(n.inputs[zi]->gradient, n.inputs[zi]->grad_epoch, g.cur_grad_epoch_);
            gacc += w * prod_nz;
            return;
        } else {
            // zc==0
            for (size_t i=0;i<m;++i) {
                const double xi = n.inputs[i]->value; // nonzero
                auto &gacc = ensure_epoch_zero(n.inputs[i]->gradient, n.inputs[i]->grad_epoch, g.cur_grad_epoch_);
                gacc += w * (prod_nz / xi);
            }
        }
    }

    // H·v column for product:
    // if zc>=2 -> 0
    // if zc==1 -> only the zero index survives; (H·v)_i = 0 for i≠zi and for i==zi: sum_{j≠zi} v_j * Π_{ℓ≠{zi,j}} x_ℓ
    // if zc==0 -> (H·v)_i = (y/xi)*( Σ_j v_j/x_j - v_i/x_i )
    static inline void hvp_backward(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n)) return;
        const size_t m = n.inputs.size();

        // Fast binary specialization (kept: already optimal)
        if (m == 2) {
            auto* a = n.inputs[0].get();
            auto* b = n.inputs[1].get();
            const double aval=a->value, bval=b->value;
            const double adot=a->dot,   bdot=b->dot;
            const double ybar=n.gradient, ybdot=n.grad_dot;

            auto &ga  = ensure_epoch_zero(a->gradient, a->grad_epoch, g.cur_grad_epoch_);
            auto &gb  = ensure_epoch_zero(b->gradient, b->grad_epoch, g.cur_grad_epoch_);
            auto &gda = ensure_epoch_zero(a->grad_dot, a->gdot_epoch, g.cur_gdot_epoch_);
            auto &gdb = ensure_epoch_zero(b->grad_dot, b->gdot_epoch, g.cur_gdot_epoch_);

            ga  += ybar * bval;                gb  += ybar * aval;
            gda += ybdot * bval + ybar * bdot; gdb += ybdot * aval + ybar * adot;
            return;
        }

        // General case
        size_t zc, zi; double prod_nz;
        analyze_inputs(n.inputs, zc, zi, prod_nz);

        const double w  = n.gradient;
        const double wd = n.grad_dot;

        if (zc >= 2) {
            // everything is zero; nothing to accumulate
            return;
        }

        if (zc == 1) {
            // Only i==zi receives first-order and second-order contributions
            // First-order: dy/dx_zi = Π_{j≠zi} x_j = prod_nz
            {
                auto &gacc = ensure_epoch_zero(n.inputs[zi]->gradient, n.inputs[zi]->grad_epoch, g.cur_grad_epoch_);
                gacc += w * prod_nz;
            }

            // Second-order column:
            // (H·v)_zi = Σ_{j≠zi} v_j * Π_{ℓ≠{zi,j}} x_ℓ   ; others zero
            double sum_Hv_zi = 0.0;
            for (size_t j=0;j<m;++j) {
                if (j == zi) continue;
                // product without zi and j: prod_nz / x_j (since prod_nz = Π_{ℓ≠zi} x_ℓ and x_j≠0 here)
                const double xj = n.inputs[j]->value;
                sum_Hv_zi += n.inputs[j]->dot * (prod_nz / xj);
            }
            auto &gdacc = ensure_epoch_zero(n.inputs[zi]->grad_dot, n.inputs[zi]->gdot_epoch, g.cur_gdot_epoch_);
            gdacc += wd * prod_nz + w * sum_Hv_zi;
            return;
        }

        // zc == 0 : fully nonzero, use O(m) formulas
        // y = prod_nz
        const double y = prod_nz;

        // Precompute S = Σ (v_j / x_j)
        double S = 0.0;
        for (size_t j=0;j<m;++j) {
            S += n.inputs[j]->dot / n.inputs[j]->value;
        }

        for (size_t i=0;i<m;++i) {
            const double xi  = n.inputs[i]->value;
            const double vi  = n.inputs[i]->dot;
            const double dfi = y / xi;                  // ∂y/∂x_i
            const double Hvi = (y / xi) * (S - vi/xi);  // (H·v)_i

            auto &gacc  = ensure_epoch_zero(n.inputs[i]->gradient,  n.inputs[i]->grad_epoch,  g.cur_grad_epoch_);
            auto &gdacc = ensure_epoch_zero(n.inputs[i]->grad_dot,  n.inputs[i]->gdot_epoch,  g.cur_gdot_epoch_);
            gacc  += w  * dfi;
            gdacc += wd * dfi + w * Hvi;
        }
    }


};

// =====================================================================
//                         MAX (nonsmooth) — keep
// =====================================================================
template <> struct OpTraits<Operator::Max> {
    static constexpr const char *name = _op_names::MAXS;
    static inline void forward(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n))
            return;
        double a = n.inputs[0]->value, b = n.inputs[1]->value;
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_,
                        (a >= b ? a : b)); // tie -> a
    }
    static inline void forward_dot(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n))
            return;
        auto &a = *n.inputs[0], &b = *n.inputs[1];
        set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_,
                        (a.value >= b.value ? a.dot : b.dot));
        touch_epoch(n.val_epoch, g.cur_val_epoch_);
    }
    static inline void backward(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n))
            return;
        auto &a = *n.inputs[0], &b = *n.inputs[1];
        if (a.value >= b.value)
            ensure_epoch_zero(a.gradient, a.grad_epoch, g.cur_grad_epoch_) +=
                n.gradient;
        else
            ensure_epoch_zero(b.gradient, b.grad_epoch, g.cur_grad_epoch_) +=
                n.gradient;
    }
    static inline void hvp_backward(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n))
            return;
        auto &a = *n.inputs[0], &b = *n.inputs[1];
        if (a.value >= b.value) {
            ensure_epoch_zero(a.gradient, a.grad_epoch, g.cur_grad_epoch_) +=
                n.gradient;
            ensure_epoch_zero(a.grad_dot, a.gdot_epoch, g.cur_gdot_epoch_) +=
                n.grad_dot;
        } else {
            ensure_epoch_zero(b.gradient, b.grad_epoch, g.cur_grad_epoch_) +=
                n.gradient;
            ensure_epoch_zero(b.grad_dot, b.gdot_epoch, g.cur_gdot_epoch_) +=
                n.grad_dot;
        }
    }
};

// =====================================================================
//                          MAP NAMES (optional)
// =====================================================================
inline const char *op_name(Operator op) {
    switch (op) {
    case Operator::Add:
        return OpTraits<Operator::Add>::name;
    case Operator::Subtract:
        return OpTraits<Operator::Subtract>::name;
    case Operator::Multiply:
        return OpTraits<Operator::Multiply>::name;
    case Operator::Divide:
        return OpTraits<Operator::Divide>::name;
    case Operator::Sin:
        return OpTraits<Operator::Sin>::name;
    case Operator::Cos:
        return OpTraits<Operator::Cos>::name;
    case Operator::Tan:
        return OpTraits<Operator::Tan>::name;
    case Operator::Exp:
        return OpTraits<Operator::Exp>::name;
    case Operator::Log:
        return OpTraits<Operator::Log>::name;
    case Operator::Max:
        return OpTraits<Operator::Max>::name;
    case Operator::Var:
        return OpTraits<Operator::Var>::name;
    case Operator::cte:
        return OpTraits<Operator::cte>::name;
    case Operator::Tanh:
        return OpTraits<Operator::Tanh>::name;
    case Operator::Silu:
        return OpTraits<Operator::Silu>::name;
    case Operator::Gelu:
        return OpTraits<Operator::Gelu>::name;
    case Operator::Softmax:
        return OpTraits<Operator::Softmax>::name;
    case Operator::Relu:
        return OpTraits<Operator::Relu>::name;
    default:
        return "unknown";
    }
}
