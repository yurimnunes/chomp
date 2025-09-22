// OpTraits.h  — FIXED for multi-lane HVP (complete corrected version)
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

// ===== Nullary: cte / var =====
template <> struct OpTraits<Operator::cte> {
    static constexpr const char *name = "cte";
    static inline void forward(ADNode &n, ADGraph &g) {
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, n.value);
    }
    static inline void forward_dot(ADNode &n, ADGraph &g) {
        set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_, 0.0);
    }
    static inline void backward(ADNode &, ADGraph &) {}
    static inline void hvp_backward(ADNode &, ADGraph &) {}
};

template <> struct OpTraits<Operator::Var> {
    static constexpr const char *name = "var";
    static inline void forward(ADNode &n, ADGraph &g) {
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, n.value);
    }
    static inline void forward_dot(ADNode &n, ADGraph &g) {
        touch_epoch(n.dot_epoch, g.cur_dot_epoch_);
    }
    static inline void backward(ADNode &, ADGraph &) {}
    static inline void hvp_backward(ADNode &, ADGraph &) {}
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

    static inline void forward_dot(ADNode &n, ADGraph &g) {
        if (!_unary_ok(n)) return;
        
        if constexpr (requires(ADNode &nn, ADGraph &gg) { Rule::forward_dot(nn, gg); }) {
            Rule::forward_dot(n, g); // custom fast-path
        } else {
            auto a = n.inputs[0];
            set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_, Rule::df(a->value) * a->dot);
        }
    }

    static inline void backward(ADNode &n, ADGraph &g) {
        if (!_unary_ok(n)) return;
        auto a = n.inputs[0];
        ensure_epoch_zero(a->gradient, a->grad_epoch, g.cur_grad_epoch_) += n.gradient * Rule::df(a->value);
    }

    static inline void hvp_backward(ADNode &n, ADGraph &g) {
        if (!_unary_ok(n)) return;
        auto a = n.inputs[0];
        
        auto &gacc = ensure_epoch_zero(a->gradient, a->grad_epoch, g.cur_grad_epoch_);
        auto &gdacc = ensure_epoch_zero(a->grad_dot, a->gdot_epoch, g.cur_gdot_epoch_);
        
        const double x = a->value;
        const double xdot = a->dot;
        const double df = Rule::df(x);
        const double d2 = Rule::d2(x);

        if (g.hvp_add_first_order_) {
            gacc += n.gradient * df;
        }

        gdacc += n.grad_dot * df + n.gradient * d2 * xdot;
    }
};

// Name literals
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

struct LogRule {
    static double f(double x) { return (x > 0.0) ? std::log(x) : std::log(1e-16); }
    static double df(double x) { return (x > 0.0) ? (1.0 / x) : 0.0; }
    static double d2(double x) { return (x > 0.0) ? (-1.0 / (x * x)) : 0.0; }
    
    static inline void forward_dot(ADNode &n, ADGraph &g) {
        auto a = n.inputs[0];
        const double x = a->value;
        const double adot = a->dot;
        set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_, (x > 0.0) ? (adot / x) : 0.0);
    }
};

struct TanRule {
    static double f(double x) { return std::tan(x); }
    static double df(double x) {
        const double c = std::cos(x);
        return (std::abs(c) > 1e-12) ? (1.0 / (c * c)) : 0.0;
    }
    static double d2(double x) {
        const double s = std::sin(x), c = std::cos(x);
        return (std::abs(c) > 1e-12) ? (2.0 * s / (c * c * c)) : 0.0;
    }
    
    static inline void forward_dot(ADNode &n, ADGraph &g) {
        auto a = n.inputs[0];
        const double c = std::cos(a->value);
        const double adot = a->dot;
        set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_, 
                        (std::abs(c) > 1e-12) ? (adot / (c * c)) : 0.0);
    }
};

struct TanhRule {
    static double f(double x)  { return std::tanh(x); }
    static double df(double x) { const double t = std::tanh(x); return 1.0 - t*t; }
    static double d2(double x) { const double t = std::tanh(x); const double s2 = 1.0 - t*t; return -2.0*t*s2; }
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
    static double f(double x)  { const double s = _sigmoid(x); return x * s; }
    static double df(double x) {
        const double s = _sigmoid(x);
        return s * (1.0 + x * (1.0 - s));
    }
    static double d2(double x) {
        const double s  = _sigmoid(x);
        const double sp = s * (1.0 - s);
        return sp * (2.0 + x * (1.0 - 2.0 * s));
    }
};

struct GELURule {
    static double f(double x)  {
        const double z = x * M_SQRT1_2;
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
//                    SOFTMAX (component implementation)
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
        const double yi = std::exp(x[0] - xmax) / (Z > 0.0 ? Z : 1.0);
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
        double sdot = 0.0; for (size_t j=0;j<m;++j) sdot += y[j] * xd[j];
        const double dot = yi * (xd[0] - sdot);

        set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_, dot);
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

        double sdot = 0.0; for (size_t j=0;j<m;++j) sdot += y[j] * xd[j];

        for (size_t k=0;k<m;++k) {
            const double dfk = yi * ((k==0)? 1.0 : 0.0) - yi * y[k];

            double Hv_k;
            if (k == 0) {
                Hv_k = yi * (1.0 - 2.0 * yi) * (xd[0] - sdot);
            } else {
                Hv_k = yi * y[k] * (2.0 * sdot - xd[0] - xd[k]);
            }

            auto &gacc  = ensure_epoch_zero(n.inputs[k]->gradient, n.inputs[k]->grad_epoch, g.cur_grad_epoch_);
            auto &gdacc = ensure_epoch_zero(n.inputs[k]->grad_dot, n.inputs[k]->gdot_epoch, g.cur_gdot_epoch_);

            if (g.hvp_add_first_order_)
                gacc  += w  * dfk;

            gdacc += wd * dfk + w * Hv_k;
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

    static inline void forward_dot(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n)) return;
        auto &a = *n.inputs[0], &b = *n.inputs[1];
        
        if constexpr (requires(ADNode &nn, ADGraph &gg) { Rule::forward_dot(nn, gg); }) {
            Rule::forward_dot(n, g);
        } else {
            const double A = a.value, B = b.value;
            set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_, 
                            Rule::dfa(A, B) * a.dot + Rule::dfb(A, B) * b.dot);
        }
    }

    static inline void backward(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n)) return;
        auto &a = *n.inputs[0], &b = *n.inputs[1];
        const double A = a.value, B = b.value, w = n.gradient;
        ensure_epoch_zero(a.gradient, a.grad_epoch, g.cur_grad_epoch_) += w * Rule::dfa(A, B);
        ensure_epoch_zero(b.gradient, b.grad_epoch, g.cur_grad_epoch_) += w * Rule::dfb(A, B);
    }

    static inline void hvp_backward(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n)) return;
        auto &a = *n.inputs[0], &b = *n.inputs[1];
        const double A = a.value, B = b.value;
        const double Ad = a.dot, Bd = b.dot;
        
        auto &ga = ensure_epoch_zero(a.gradient, a.grad_epoch, g.cur_grad_epoch_);
        auto &gb = ensure_epoch_zero(b.gradient, b.grad_epoch, g.cur_grad_epoch_);
        auto &gda = ensure_epoch_zero(a.grad_dot, a.gdot_epoch, g.cur_gdot_epoch_);
        auto &gdb = ensure_epoch_zero(b.grad_dot, b.gdot_epoch, g.cur_gdot_epoch_);
        const double w = n.gradient, wd = n.grad_dot;

        if (g.hvp_add_first_order_) {
            ga += w * Rule::dfa(A, B);
            gb += w * Rule::dfb(A, B);
        }

        gda += wd * Rule::dfa(A, B) + w * (Rule::d2aa(A, B) * Ad + Rule::d2ab(A, B) * Bd);
        gdb += wd * Rule::dfb(A, B) + w * (Rule::d2ab(A, B) * Ad + Rule::d2bb(A, B) * Bd);
    }
};

// Binary rules
struct AddRule {
    static double f(double a, double b) { return a + b; }
    static double dfa(double, double) { return 1.0; }
    static double dfb(double, double) { return 1.0; }
    static double d2aa(double, double) { return 0.0; }
    static double d2ab(double, double) { return 0.0; }
    static double d2bb(double, double) { return 0.0; }
};

struct SubRule {
    static double f(double a, double b) { return a - b; }
    static double dfa(double, double) { return 1.0; }
    static double dfb(double, double) { return -1.0; }
    static double d2aa(double, double) { return 0.0; }
    static double d2ab(double, double) { return 0.0; }
    static double d2bb(double, double) { return 0.0; }
};

struct DivRule {
    static double f(double a, double b) { return _safe_div(a, b); }
    static double dfa(double, double b) { return (b != 0.0) ? (1.0 / b) : 0.0; }
    static double dfb(double a, double b) { return (b != 0.0) ? (-a / (b * b)) : 0.0; }
    static double d2aa(double, double) { return 0.0; }
    static double d2ab(double, double b) { return (b != 0.0) ? (-1.0 / (b * b)) : 0.0; }
    static double d2bb(double a, double b) { return (b != 0.0) ? (2.0 * a / (b * b * b)) : 0.0; }
    
    static inline void forward_dot(ADNode &n, ADGraph &g) {
        auto &a = *n.inputs[0], &b = *n.inputs[1];
        const double d = b.value;
        const double adot = a.dot;
        const double bdot = b.dot;
        set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_,
                        (d != 0.0) ? ((adot * d - a.value * bdot) / (d * d)) : 0.0);
    }
};

template <> struct OpTraits<Operator::Subtract> : BinaryOp<SubRule, _op_names::SUB> {};
template <> struct OpTraits<Operator::Divide> : BinaryOp<DivRule, _op_names::DIV> {};

// N-ARY ADD
template <> struct OpTraits<Operator::Add> : BinaryOp<AddRule, _op_names::ADD> {
    static inline void forward(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n)) return;
        double s = 0.0;
        for (auto &a : n.inputs) s += a->value;
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, s);
    }
    
    static inline void forward_dot(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n)) return;
        double sd = 0.0;
        for (auto &a : n.inputs) {
            sd += a->dot;
        }
        set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_, sd);
    }
    
    static inline void backward(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n)) return;
        for (auto &a : n.inputs) {
            ensure_epoch_zero(a->gradient, a->grad_epoch, g.cur_grad_epoch_) += n.gradient;
        }
    }
    
    static inline void hvp_backward(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n)) return;
        for (auto &a : n.inputs) {
            if (g.hvp_add_first_order_) {
                ensure_epoch_zero(a->gradient, a->grad_epoch, g.cur_grad_epoch_) += n.gradient;
            }
            ensure_epoch_zero(a->grad_dot, a->gdot_epoch, g.cur_gdot_epoch_) += n.grad_dot;
        }
    }
};

// MULTIPLY (n-ary)
template <> struct OpTraits<Operator::Multiply> {
    static constexpr const char *name = _op_names::MUL;

    static inline std::vector<double> &tls_vals() { thread_local std::vector<double> v; return v; }
    static inline std::vector<double> &tls_dots() { thread_local std::vector<double> v; return v; }

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

    static inline void forward(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n)) return;
        size_t zc, zi; double prod_nz;
        analyze_inputs(n.inputs, zc, zi, prod_nz);
        const double y = (zc == 0) ? prod_nz : 0.0;
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, y);
    }

    static inline void forward_dot(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n)) return;
        const size_t m = n.inputs.size();

        size_t zc, zi; double prod_nz;
        analyze_inputs(n.inputs, zc, zi, prod_nz);

        double ydot = 0.0;
        if (zc >= 2) {
            ydot = 0.0;
        } else if (zc == 1) {
            ydot = n.inputs[zi]->dot * prod_nz;
        } else {
            double sum = 0.0;
            for (size_t i=0;i<m;++i) {
                const double xi = n.inputs[i]->value;
                sum += n.inputs[i]->dot / xi;
            }
            ydot = prod_nz * sum;
        }

        set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_, ydot);
    }

    static inline void backward(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n)) return;
        const size_t m = n.inputs.size();

        size_t zc, zi; double prod_nz;
        analyze_inputs(n.inputs, zc, zi, prod_nz);

        const double w = n.gradient;
        if (zc >= 2) {
            return;
        } else if (zc == 1) {
            auto &gacc = ensure_epoch_zero(n.inputs[zi]->gradient, n.inputs[zi]->grad_epoch, g.cur_grad_epoch_);
            gacc += w * prod_nz;
            return;
        } else {
            for (size_t i=0;i<m;++i) {
                const double xi = n.inputs[i]->value;
                auto &gacc = ensure_epoch_zero(n.inputs[i]->gradient, n.inputs[i]->grad_epoch, g.cur_grad_epoch_);
                gacc += w * (prod_nz / xi);
            }
        }
    }

    static inline void hvp_backward(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n)) return;
        const size_t m = n.inputs.size();

        // Fast binary specialization
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

            if (g.hvp_add_first_order_) {
                ga  += ybar * bval;
                gb  += ybar * aval;
            }
            gda += ybdot * bval + ybar * bdot;
            gdb += ybdot * aval + ybar * adot;
            return;
        }

        // General case
        size_t zc, zi; double prod_nz;
        analyze_inputs(n.inputs, zc, zi, prod_nz);

        const double w  = n.gradient;
        const double wd = n.grad_dot;

        if (zc >= 2) {
            return;
        }

        if (zc == 1) {
            if (g.hvp_add_first_order_) {
                auto &gacc = ensure_epoch_zero(n.inputs[zi]->gradient, n.inputs[zi]->grad_epoch, g.cur_grad_epoch_);
                gacc += w * prod_nz;
            }

            double sum_Hv_zi = 0.0;
            for (size_t j=0;j<m;++j) {
                if (j == zi) continue;
                const double xj = n.inputs[j]->value;
                sum_Hv_zi += n.inputs[j]->dot * (prod_nz / xj);
            }
            auto &gdacc = ensure_epoch_zero(n.inputs[zi]->grad_dot, n.inputs[zi]->gdot_epoch, g.cur_gdot_epoch_);
            gdacc += wd * prod_nz + w * sum_Hv_zi;
            return;
        }

        // zc == 0 : fully nonzero
        const double y = prod_nz;

        double S = 0.0;
        for (size_t j=0;j<m;++j) {
            S += n.inputs[j]->dot / n.inputs[j]->value;
        }

        for (size_t i=0;i<m;++i) {
            const double xi  = n.inputs[i]->value;
            const double vi  = n.inputs[i]->dot;
            const double dfi = y / xi;
            const double Hvi = (y / xi) * (S - vi/xi);

            auto &gacc  = ensure_epoch_zero(n.inputs[i]->gradient,  n.inputs[i]->grad_epoch,  g.cur_grad_epoch_);
            auto &gdacc = ensure_epoch_zero(n.inputs[i]->grad_dot,  n.inputs[i]->gdot_epoch,  g.cur_gdot_epoch_);

            if (g.hvp_add_first_order_)
                gacc  += w  * dfi;

            gdacc += wd * dfi + w * Hvi;
        }
    }
};

// MAX (nonsmooth)
template <> struct OpTraits<Operator::Max> {
    static constexpr const char *name = _op_names::MAXS;
    
    static inline void forward(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n)) return;
        double a = n.inputs[0]->value, b = n.inputs[1]->value;
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, (a >= b ? a : b));
    }
    
    static inline void forward_dot(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n)) return;
        auto &a = *n.inputs[0], &b = *n.inputs[1];
        set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_, (a.value >= b.value ? a.dot : b.dot));
    }
    
    static inline void backward(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n)) return;
        auto &a = *n.inputs[0], &b = *n.inputs[1];
        if (a.value >= b.value)
            ensure_epoch_zero(a.gradient, a.grad_epoch, g.cur_grad_epoch_) += n.gradient;
        else
            ensure_epoch_zero(b.gradient, b.grad_epoch, g.cur_grad_epoch_) += n.gradient;
    }
    
    static inline void hvp_backward(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n)) return;
        auto &a = *n.inputs[0], &b = *n.inputs[1];
        if (a.value >= b.value) {
            if (g.hvp_add_first_order_)
                ensure_epoch_zero(a.gradient, a.grad_epoch, g.cur_grad_epoch_) += n.gradient;
            ensure_epoch_zero(a.grad_dot, a.gdot_epoch, g.cur_gdot_epoch_) += n.grad_dot;
        } else {
            if (g.hvp_add_first_order_)
                ensure_epoch_zero(b.gradient, b.grad_epoch, g.cur_grad_epoch_) += n.gradient;
            ensure_epoch_zero(b.grad_dot, b.gdot_epoch, g.cur_gdot_epoch_) += n.grad_dot;
        }
    }
};

// Plug unary ops into OpTraits
template <> struct OpTraits<Operator::Sin> : UnaryOp<SinRule, _op_names::SIN> {};
template <> struct OpTraits<Operator::Cos> : UnaryOp<CosRule, _op_names::COS> {};
template <> struct OpTraits<Operator::Exp> : UnaryOp<ExpRule, _op_names::EXP> {};
template <> struct OpTraits<Operator::Log> : UnaryOp<LogRule, _op_names::LOG> {};
template <> struct OpTraits<Operator::Tan> : UnaryOp<TanRule, _op_names::TAN> {};
template <> struct OpTraits<Operator::Tanh> : UnaryOp<TanhRule, _op_names::TANH> {};
template <> struct OpTraits<Operator::Silu> : UnaryOp<SiLURule, _op_names::SILU> {};
template <> struct OpTraits<Operator::Gelu> : UnaryOp<GELURule, _op_names::GELU> {};
template <> struct OpTraits<Operator::Relu> : UnaryOp<ReluRule, _op_names::RELU> {};

// Map names function
inline const char *op_name(Operator op) {
    switch (op) {
    case Operator::Add: return OpTraits<Operator::Add>::name;
    case Operator::Subtract: return OpTraits<Operator::Subtract>::name;
    case Operator::Multiply: return OpTraits<Operator::Multiply>::name;
    case Operator::Divide: return OpTraits<Operator::Divide>::name;
    case Operator::Sin: return OpTraits<Operator::Sin>::name;
    case Operator::Cos: return OpTraits<Operator::Cos>::name;
    case Operator::Tan: return OpTraits<Operator::Tan>::name;
    case Operator::Exp: return OpTraits<Operator::Exp>::name;
    case Operator::Log: return OpTraits<Operator::Log>::name;
    case Operator::Max: return OpTraits<Operator::Max>::name;
    case Operator::Var: return OpTraits<Operator::Var>::name;
    case Operator::cte: return OpTraits<Operator::cte>::name;
    case Operator::Tanh: return OpTraits<Operator::Tanh>::name;
    case Operator::Silu: return OpTraits<Operator::Silu>::name;
    case Operator::Gelu: return OpTraits<Operator::Gelu>::name;
    case Operator::Softmax: return OpTraits<Operator::Softmax>::name;
    case Operator::Relu: return OpTraits<Operator::Relu>::name;
    default: return "unknown";
    }
}