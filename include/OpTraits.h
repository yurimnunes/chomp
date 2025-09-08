// OpTraits.h
#pragma once
#include "ADGraph.h" // declares ADNode, ADGraph, set_epoch_value, touch_epoch, ensure_epoch_zero
#include <cmath>
#include <cstddef>
#include <vector>

// enum class Operator { ... }  // assumed declared elsewhere

// ---- tiny helpers (unchanged) ----
inline double _safe_div(double a, double b) { return (b != 0.0) ? (a / b) : 0.0; }
inline bool _unary_ok(const ADNode &n)  { return n.inputs.size() == 1 && n.inputs[0] != nullptr; }
inline bool _binary_ok(const ADNode &n) { return n.inputs.size() == 2 && n.inputs[0] && n.inputs[1]; }
inline bool _nary_ok(const ADNode &n)   { return !n.inputs.empty(); }

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
    static inline void forward(ADNode &n, ADGraph &g)      { touch_epoch(n.val_epoch, g.cur_val_epoch_); }
    static inline void forward_dot(ADNode &n, ADGraph &g)  { touch_epoch(n.dot_epoch, g.cur_dot_epoch_); touch_epoch(n.val_epoch, g.cur_val_epoch_); }
    static inline void backward(ADNode &, ADGraph &) {}
    static inline void hvp_backward(ADNode &, ADGraph &) {}
};
template <> struct OpTraits<Operator::Var> {
    static constexpr const char *name = "var";
    static inline void forward(ADNode &n, ADGraph &g)      { touch_epoch(n.val_epoch, g.cur_val_epoch_); }
    static inline void forward_dot(ADNode &n, ADGraph &g)  { touch_epoch(n.dot_epoch, g.cur_dot_epoch_); touch_epoch(n.val_epoch, g.cur_val_epoch_); }
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
//   static inline void forward_dot(ADNode&, ADGraph&);  // custom (e.g., tan/log with guards)
//
// Then:   template<> struct OpTraits<Op::Sin> : UnaryOp<SinRule, "sin"> {};
//
template <class Rule, const char *NameLiteral>
struct UnaryOp {
    static constexpr const char *name = NameLiteral;

    static inline void forward(ADNode &n, ADGraph &g) {
        if (!_unary_ok(n)) return;
        auto a = n.inputs[0];
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, Rule::f(a->value));
    }

    static inline void forward_dot(ADNode &n, ADGraph &g) {
        if (!_unary_ok(n)) return;
        if constexpr (requires(ADNode &nn, ADGraph &gg){ Rule::forward_dot(nn, gg); }) {
            Rule::forward_dot(n, g); // custom fast-path
        } else {
            auto a = n.inputs[0];
            set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_, Rule::df(a->value) * a->dot);
            touch_epoch(n.val_epoch, g.cur_val_epoch_);
        }
    }

    static inline void backward(ADNode &n, ADGraph &g) {
        if (!_unary_ok(n)) return;
        auto a = n.inputs[0];
        ensure_epoch_zero(a->gradient, a->grad_epoch, g.cur_grad_epoch_) +=
            n.gradient * Rule::df(a->value);
    }

    static inline void hvp_backward(ADNode &n, ADGraph &g) {
        if (!_unary_ok(n)) return;
        auto a = n.inputs[0];
        auto &gacc  = ensure_epoch_zero(a->gradient, a->grad_epoch, g.cur_grad_epoch_);
        auto &gdacc = ensure_epoch_zero(a->grad_dot, a->gdot_epoch, g.cur_gdot_epoch_);
        const double x = a->value, xdot = a->dot, df = Rule::df(x), d2 = Rule::d2(x);
        gacc  += n.gradient * df;
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
}

// ---- Concrete unary rules ----
struct SinRule { static double f(double x){return std::sin(x);} static double df(double x){return std::cos(x);} static double d2(double x){return -std::sin(x);} };
struct CosRule { static double f(double x){return std::cos(x);} static double df(double x){return -std::sin(x);} static double d2(double x){return -std::cos(x);} };
struct ExpRule { static double f(double x){return std::exp(x);} static double df(double x){return std::exp(x);} static double d2(double x){return std::exp(x);} };

// Log has domain guard + custom dot to avoid repeated exp/log
struct LogRule {
    static double f(double x)  { return std::log(x); }
    static double df(double x) { return (x != 0.0) ? (1.0 / x) : 0.0; }
    static double d2(double x) { return (x != 0.0) ? (-1.0 / (x * x)) : 0.0; }
    static inline void forward_dot(ADNode &n, ADGraph &g) {
        auto a = n.inputs[0];
        const double x = a->value;
        set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_, x != 0.0 ? (a->dot / x) : 0.0);
        touch_epoch(n.val_epoch, g.cur_val_epoch_);
    }
};

// Tan with stable cos guard + custom dot
struct TanRule {
    static double f(double x)  { return std::tan(x); }
    static double df(double x) { const double c=std::cos(x); return (c!=0.0)? (1.0/(c*c)) : 0.0; }           // sec^2
    static double d2(double x) { const double s=std::sin(x), c=std::cos(x); return (c!=0.0)? (2.0*s/(c*c*c)) : 0.0; } // 2 sec^2 tan = 2 sin/cos^3
    static inline void forward_dot(ADNode &n, ADGraph &g) {
        auto a = n.inputs[0];
        const double c = std::cos(a->value);
        set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_, (c!=0.0) ? (a->dot / (c*c)) : 0.0);
        touch_epoch(n.val_epoch, g.cur_val_epoch_);
    }
};

// ---- Plug unary ops into OpTraits via the generic plumbing ----
template <> struct OpTraits<Operator::Sin> : UnaryOp<SinRule, _op_names::SIN> {};
template <> struct OpTraits<Operator::Cos> : UnaryOp<CosRule, _op_names::COS> {};
template <> struct OpTraits<Operator::Exp> : UnaryOp<ExpRule, _op_names::EXP> {};
template <> struct OpTraits<Operator::Log> : UnaryOp<LogRule, _op_names::LOG> {};
template <> struct OpTraits<Operator::Tan> : UnaryOp<TanRule, _op_names::TAN> {};

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
template <class Rule, const char *NameLiteral>
struct BinaryOp {
    static constexpr const char *name = NameLiteral;

    static inline void forward(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n)) return;
        auto &a = *n.inputs[0], &b = *n.inputs[1];
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, Rule::f(a.value, b.value));
    }

    static inline void forward_dot(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n)) return;
        auto &a = *n.inputs[0], &b = *n.inputs[1];
        if constexpr (requires(ADNode &nn, ADGraph &gg){ Rule::forward_dot(nn, gg); }) {
            Rule::forward_dot(n, g);
        } else {
            const double A=a.value, B=b.value;
            set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_,
                Rule::dfa(A,B) * a.dot + Rule::dfb(A,B) * b.dot);
            touch_epoch(n.val_epoch, g.cur_val_epoch_);
        }
    }

    static inline void backward(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n)) return;
        auto &a = *n.inputs[0], &b = *n.inputs[1];
        const double A=a.value, B=b.value, w=n.gradient;
        ensure_epoch_zero(a.gradient, a.grad_epoch, g.cur_grad_epoch_) += w * Rule::dfa(A,B);
        ensure_epoch_zero(b.gradient, b.grad_epoch, g.cur_grad_epoch_) += w * Rule::dfb(A,B);
    }

    static inline void hvp_backward(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n)) return;
        auto &a = *n.inputs[0], &b = *n.inputs[1];
        const double A=a.value, B=b.value, Ad=a.dot, Bd=b.dot;
        auto &ga = ensure_epoch_zero(a.gradient, a.grad_epoch, g.cur_grad_epoch_);
        auto &gb = ensure_epoch_zero(b.gradient, b.grad_epoch, g.cur_grad_epoch_);
        auto &gda= ensure_epoch_zero(a.grad_dot, a.gdot_epoch, g.cur_gdot_epoch_);
        auto &gdb= ensure_epoch_zero(b.grad_dot, b.gdot_epoch, g.cur_gdot_epoch_);
        const double w = n.gradient, wd = n.grad_dot;

        // First-order adjoints
        ga += w * Rule::dfa(A,B);
        gb += w * Rule::dfb(A,B);

        // HVP propagation: gdot_i += wdot * dfi + w * (H_iA * A_dot + H_iB * B_dot)
        gda += wd * Rule::dfa(A,B) + w * (Rule::d2aa(A,B) * Ad + Rule::d2ab(A,B) * Bd);
        gdb += wd * Rule::dfb(A,B) + w * (Rule::d2ab(A,B) * Ad + Rule::d2bb(A,B) * Bd);
    }
};

// ---- Concrete binary rules ----
// Add: f = a + b
struct AddRule {
    static double f(double a,double b){ return a + b; }
    static double dfa(double,double){ return 1.0; }
    static double dfb(double,double){ return 1.0; }
    static double d2aa(double,double){ return 0.0; }
    static double d2ab(double,double){ return 0.0; }
    static double d2bb(double,double){ return 0.0; }
};
// Subtract: f = a - b
struct SubRule {
    static double f(double a,double b){ return a - b; }
    static double dfa(double,double){ return 1.0; }
    static double dfb(double,double){ return -1.0; }
    static double d2aa(double,double){ return 0.0; }
    static double d2ab(double,double){ return 0.0; }
    static double d2bb(double,double){ return 0.0; }
};
// Divide: f = a / b
struct DivRule {
    static double f(double a,double b){ return _safe_div(a,b); }
    static double dfa(double,double b){ return (b!=0.0) ? (1.0/b) : 0.0; }
    static double dfb(double a,double b){ return (b!=0.0) ? (-a/(b*b)) : 0.0; }
    static double d2aa(double,double){ return 0.0; }
    static double d2ab(double,double b){ return (b!=0.0) ? (-1.0/(b*b)) : 0.0; }
    static double d2bb(double a,double b){ return (b!=0.0) ? (2.0*a/(b*b*b)) : 0.0; }
    static inline void forward_dot(ADNode &n, ADGraph &g) {
        auto &a=*n.inputs[0], &b=*n.inputs[1];
        const double d = b.value;
        set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_,
            d!=0.0 ? ((a.dot * d - a.value * b.dot) / (d*d)) : 0.0);
        touch_epoch(n.val_epoch, g.cur_val_epoch_);
    }
};

// ---- Hook rules to OpTraits ----
//template <> struct OpTraits<Operator::Add>      : BinaryOp<AddRule, _op_names::ADD> {};
template <> struct OpTraits<Operator::Subtract> : BinaryOp<SubRule, _op_names::SUB> {};
template <> struct OpTraits<Operator::Divide>   : BinaryOp<DivRule, _op_names::DIV> {};

// =====================================================================
//                    N-ARY SUM (generic)  — stays tiny
// =====================================================================
template <> struct OpTraits<Operator::Add> : BinaryOp<AddRule, _op_names::ADD> {
    // n-ary sum specialization (overrides BinaryOp’s forward/backward/hvp)
    static inline void forward(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n)) return;
        double s = 0.0; for (auto &a : n.inputs) s += a->value;
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, s);
    }
    static inline void forward_dot(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n)) return;
        double sd = 0.0; for (auto &a : n.inputs) sd += a->dot;
        set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_, sd);
        touch_epoch(n.val_epoch, g.cur_val_epoch_);
    }
    static inline void backward(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n)) return;
        for (auto &a : n.inputs)
            ensure_epoch_zero(a->gradient, a->grad_epoch, g.cur_grad_epoch_) += n.gradient;
    }
    static inline void hvp_backward(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n)) return;
        for (auto &a : n.inputs) {
            ensure_epoch_zero(a->gradient, a->grad_epoch, g.cur_grad_epoch_) += n.gradient;
            ensure_epoch_zero(a->grad_dot, a->gdot_epoch, g.cur_gdot_epoch_) += n.grad_dot;
        }
    }
};

// =====================================================================
//                 MULTIPLY (n-ary) — keep your optimized version
// =====================================================================
template <> struct OpTraits<Operator::Multiply> {
    static constexpr const char *name = _op_names::MUL;

    static inline std::vector<double> &tls_vals(){ thread_local std::vector<double> v; return v; }
    static inline std::vector<double> &tls_dots(){ thread_local std::vector<double> v; return v; }
    static inline std::vector<double> &tls_pre() { thread_local std::vector<double> v; return v; }
    static inline std::vector<double> &tls_suf() { thread_local std::vector<double> v; return v; }

    static inline void build_prefix_suffix(const std::vector<double> &vals,
                                           std::vector<double> &pre,
                                           std::vector<double> &suf) {
        const size_t m = vals.size();
        pre.assign(m + 1, 1.0); suf.assign(m + 1, 1.0);
        for (size_t i=0;i<m;++i) pre[i+1] = pre[i] * vals[i];
        for (size_t i=m;i-- >0;) suf[i]   = suf[i+1] * vals[i];
    }

    static inline void forward(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n)) return;
        double p = 1.0; for (auto &a : n.inputs) p *= a->value;
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, p);
    }
    static inline void forward_dot(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n)) return;
        const size_t m = n.inputs.size();
        auto &vals=tls_vals(); auto &dots=tls_dots(); vals.resize(m); dots.resize(m);
        for (size_t i=0;i<m;++i){ vals[i]=n.inputs[i]->value; dots[i]=n.inputs[i]->dot; }
        auto &pre=tls_pre(); auto &suf=tls_suf(); build_prefix_suffix(vals, pre, suf);
        double ds=0.0; for (size_t i=0;i<m;++i) ds += dots[i] * pre[i] * suf[i+1];
        set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_, ds);
        touch_epoch(n.val_epoch, g.cur_val_epoch_);
    }
    static inline void backward(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n)) return;
        const size_t m = n.inputs.size();
        auto &vals=tls_vals(); vals.resize(m);
        for (size_t i=0;i<m;++i) vals[i]=n.inputs[i]->value;
        auto &pre=tls_pre(); auto &suf=tls_suf(); build_prefix_suffix(vals, pre, suf);
        for (size_t i=0;i<m;++i){
            const double P_wo_i = pre[i] * suf[i+1];
            ensure_epoch_zero(n.inputs[i]->gradient, n.inputs[i]->grad_epoch, g.cur_grad_epoch_) += n.gradient * P_wo_i;
        }
    }
    static inline void hvp_backward(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n)) return;
        const size_t m = n.inputs.size();
        auto &vals=tls_vals(); auto &dots=tls_dots(); vals.resize(m); dots.resize(m);
        for (size_t i=0;i<m;++i){ vals[i]=n.inputs[i]->value; dots[i]=n.inputs[i]->dot; }
        auto &pre=tls_pre(); auto &suf=tls_suf(); build_prefix_suffix(vals, pre, suf);
        for (size_t i=0;i<m;++i){
            const double P_wo_i = pre[i]*suf[i+1];
            double sum_term = 0.0;
            for (size_t k=0;k<m;++k){ if (k==i) continue;
                const size_t a=(i<k? i:k), b=(i<k? k:i);
                const double left=pre[a], mid=(pre[b]/pre[a+1]), right=suf[b+1];
                sum_term += dots[k] * (left * mid * right);
            }
            auto &gacc = ensure_epoch_zero(n.inputs[i]->gradient, n.inputs[i]->grad_epoch, g.cur_grad_epoch_);
            auto &gdacc= ensure_epoch_zero(n.inputs[i]->grad_dot, n.inputs[i]->gdot_epoch, g.cur_gdot_epoch_);
            gacc  += n.gradient * P_wo_i;
            gdacc += n.grad_dot * P_wo_i + n.gradient * sum_term;
        }
    }
};

// =====================================================================
//                         MAX (nonsmooth) — keep
// =====================================================================
template <> struct OpTraits<Operator::Max> {
    static constexpr const char *name = _op_names::MAXS;
    static inline void forward(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n)) return;
        double a=n.inputs[0]->value, b=n.inputs[1]->value;
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, (a>=b? a:b)); // tie -> a
    }
    static inline void forward_dot(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n)) return;
        auto &a=*n.inputs[0], &b=*n.inputs[1];
        set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_, (a.value>=b.value? a.dot:b.dot));
        touch_epoch(n.val_epoch, g.cur_val_epoch_);
    }
    static inline void backward(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n)) return;
        auto &a=*n.inputs[0], &b=*n.inputs[1];
        if (a.value>=b.value)
            ensure_epoch_zero(a.gradient,a.grad_epoch,g.cur_grad_epoch_)+=n.gradient;
        else
            ensure_epoch_zero(b.gradient,b.grad_epoch,g.cur_grad_epoch_)+=n.gradient;
    }
    static inline void hvp_backward(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n)) return;
        auto &a=*n.inputs[0], &b=*n.inputs[1];
        if (a.value>=b.value){
            ensure_epoch_zero(a.gradient,a.grad_epoch,g.cur_grad_epoch_)+=n.gradient;
            ensure_epoch_zero(a.grad_dot,a.gdot_epoch,g.cur_gdot_epoch_)+=n.grad_dot;
        } else {
            ensure_epoch_zero(b.gradient,b.grad_epoch,g.cur_grad_epoch_)+=n.gradient;
            ensure_epoch_zero(b.grad_dot,b.gdot_epoch,g.cur_gdot_epoch_)+=n.grad_dot;
        }
    }
};

// =====================================================================
//                          MAP NAMES (optional)
// =====================================================================
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
    default:                 return "unknown";
    }
}
