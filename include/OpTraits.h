// OpTraits.h
#pragma once
#include "ADGraph.h" // must declare ADNode, ADGraph, set_epoch_value, touch_epoch, ensure_epoch_zero
#include <cmath>
#include <cstddef>
#include <vector>
// and enum class Operator

// ---- small helpers ----
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

// A “trait” per Operator centralizes all math for:
//   forward(), forward_dot(), backward(), hvp_backward()
// Each function receives ADNode& n and ADGraph& g so epoch helpers are
// available.
//
// IMPORTANT: Functions must be no-ops if inputs are invalid (keep your
// diagnostics elsewhere).
template <Operator Op> struct OpTraits {
    static inline void forward(ADNode &, ADGraph &) {}
    static inline void forward_dot(ADNode &, ADGraph &) {}
    static inline void backward(ADNode &, ADGraph &) {}
    static inline void hvp_backward(ADNode &, ADGraph &) {}
};

// ===== cte / Var (nullary) =====
// OpTraits.h (patch: add empty backward/hvp_backward for Var/cte)

// ===== cte / Var (nullary) =====
template <> struct OpTraits<Operator::cte> {

    // add name
    static constexpr const char *name = "cte";
    static inline void forward(ADNode &n, ADGraph &g) {
        touch_epoch(n.val_epoch, g.cur_val_epoch_);
    }
    static inline void forward_dot(ADNode &n, ADGraph &g) {
        touch_epoch(n.dot_epoch, g.cur_dot_epoch_);
        touch_epoch(n.val_epoch, g.cur_val_epoch_);
    }
    // NEW: nullary has no parents to accumulate into
    static inline void backward(ADNode &, ADGraph &) {}
    static inline void hvp_backward(ADNode &, ADGraph &) {}
};

template <> struct OpTraits<Operator::Var> {

    // add name
    static constexpr const char *name = "var";
    static inline void forward(ADNode &n, ADGraph &g) {
        touch_epoch(n.val_epoch, g.cur_val_epoch_);
    }
    static inline void forward_dot(ADNode &n, ADGraph &g) {
        touch_epoch(n.dot_epoch, g.cur_dot_epoch_);
        touch_epoch(n.val_epoch, g.cur_val_epoch_);
    }
    // NEW: variables are leafs — nothing to push further during reverse passes
    static inline void backward(ADNode &, ADGraph &) {}
    static inline void hvp_backward(ADNode &, ADGraph &) {}
};

// ===== Unary template helpers =====
template <typename F>
static inline void _unary_forward(ADNode &n, ADGraph &g, F f) {
    if (!_unary_ok(n))
        return;
    auto a = n.inputs[0];
    set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, f(a->value));
}
template <typename DF>
static inline void _unary_backward(ADNode &n, ADGraph &g, DF df) {
    if (!_unary_ok(n))
        return;
    auto a = n.inputs[0];
    ensure_epoch_zero(a->gradient, a->grad_epoch, g.cur_grad_epoch_) +=
        n.gradient * df(a->value);
}
template <typename DF, typename D2>
static inline void _unary_hvp(ADNode &n, ADGraph &g, DF df, D2 d2) {
    if (!_unary_ok(n))
        return;
    auto a = n.inputs[0];
    // grad
    ensure_epoch_zero(a->gradient, a->grad_epoch, g.cur_grad_epoch_) +=
        n.gradient * df(a->value);
    // grad_dot
    auto &gd = ensure_epoch_zero(a->grad_dot, a->gdot_epoch, g.cur_gdot_epoch_);
    gd += n.grad_dot * df(a->value) + n.gradient * d2(a->value) * a->dot;
}

// ===== Sin / Cos / Tan / Exp / Log =====
template <> struct OpTraits<Operator::Sin> {

    // add name
    static constexpr const char *name = "sin";
    static inline void forward(ADNode &n, ADGraph &g) {
        _unary_forward(n, g, [](double x) { return std::sin(x); });
    }
    static inline void forward_dot(ADNode &n, ADGraph &g) {
        if (!_unary_ok(n))
            return;
        auto a = n.inputs[0];
        set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_,
                        std::cos(a->value) * a->dot);
        touch_epoch(n.val_epoch,
                    g.cur_val_epoch_); // value already set in forward()
    }
    static inline void backward(ADNode &n, ADGraph &g) {
        _unary_backward(n, g, [](double x) { return std::cos(x); });
    }
    static inline void hvp_backward(ADNode &n, ADGraph &g) {
        _unary_hvp(
            n, g, [](double x) { return std::cos(x); },
            [](double x) { return -std::sin(x); });
    }
};

template <> struct OpTraits<Operator::Cos> {
    static constexpr const char *name = "cos";

    static inline void forward(ADNode &n, ADGraph &g) {
        _unary_forward(n, g, [](double x) { return std::cos(x); });
    }
    static inline void forward_dot(ADNode &n, ADGraph &g) {
        if (!_unary_ok(n))
            return;
        auto a = n.inputs[0];
        set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_,
                        -std::sin(a->value) * a->dot);
        touch_epoch(n.val_epoch, g.cur_val_epoch_);
    }
    static inline void backward(ADNode &n, ADGraph &g) {
        _unary_backward(n, g, [](double x) { return -std::sin(x); });
    }
    static inline void hvp_backward(ADNode &n, ADGraph &g) {
        _unary_hvp(
            n, g, [](double x) { return -std::sin(x); },
            [](double x) { return -std::cos(x); });
    }
};

template <> struct OpTraits<Operator::Tan> {
    static constexpr const char *name = "tan";

    static inline void forward(ADNode &n, ADGraph &g) {
        _unary_forward(n, g, [](double x) { return std::tan(x); });
    }
    static inline void forward_dot(ADNode &n, ADGraph &g) {
        if (!_unary_ok(n))
            return;
        auto a = n.inputs[0];
        double c = std::cos(a->value);
        set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_,
                        (c != 0.0) ? (a->dot / (c * c)) : 0.0);
        touch_epoch(n.val_epoch, g.cur_val_epoch_);
    }
    static inline void backward(ADNode &n, ADGraph &g) {
        if (!_unary_ok(n))
            return;
        auto a = n.inputs[0];
        double c = std::cos(a->value);
        if (c == 0.0)
            return;
        ensure_epoch_zero(a->gradient, a->grad_epoch, g.cur_grad_epoch_) +=
            n.gradient * (1.0 / (c * c));
    }
    static inline void hvp_backward(ADNode &n, ADGraph &g) {
        if (!_unary_ok(n))
            return;
        auto a = n.inputs[0];
        double c = std::cos(a->value);
        if (c == 0.0)
            return;
        const double sec2 = 1.0 / (c * c);
        const double t = std::tan(a->value);
        ensure_epoch_zero(a->gradient, a->grad_epoch, g.cur_grad_epoch_) +=
            n.gradient * sec2;
        ensure_epoch_zero(a->grad_dot, a->gdot_epoch, g.cur_gdot_epoch_) +=
            n.grad_dot * sec2 + n.gradient * (2.0 * sec2 * t * a->dot);
    }
};

template <> struct OpTraits<Operator::Exp> {
    static constexpr const char *name = "exp";

    static inline void forward(ADNode &n, ADGraph &g) {
        _unary_forward(n, g, [](double x) { return std::exp(x); });
    }
    static inline void forward_dot(ADNode &n, ADGraph &g) {
        if (!_unary_ok(n))
            return;
        auto a = n.inputs[0];
        double ev = std::exp(a->value);
        set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_, ev * a->dot);
        touch_epoch(n.val_epoch, g.cur_val_epoch_);
    }
    static inline void backward(ADNode &n, ADGraph &g) {
        _unary_backward(n, g, [](double x) { return std::exp(x); });
    }
    static inline void hvp_backward(ADNode &n, ADGraph &g) {
        _unary_hvp(
            n, g, [](double x) { return std::exp(x); },
            [](double x) { return std::exp(x); });
    }
};

template <> struct OpTraits<Operator::Log> {
    static constexpr const char *name = "log";

    static inline void forward(ADNode &n, ADGraph &g) {
        _unary_forward(n, g, [](double x) { return std::log(x); });
    }
    static inline void forward_dot(ADNode &n, ADGraph &g) {
        if (!_unary_ok(n))
            return;
        auto a = n.inputs[0];
        set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_,
                        (a->value != 0.0) ? (a->dot / a->value) : 0.0);
        touch_epoch(n.val_epoch, g.cur_val_epoch_);
    }
    static inline void backward(ADNode &n, ADGraph &g) {
        if (!_unary_ok(n))
            return;
        auto a = n.inputs[0];
        if (a->value == 0.0)
            return;
        ensure_epoch_zero(a->gradient, a->grad_epoch, g.cur_grad_epoch_) +=
            n.gradient * (1.0 / a->value);
    }
    static inline void hvp_backward(ADNode &n, ADGraph &g) {
        if (!_unary_ok(n))
            return;
        auto a = n.inputs[0];
        if (a->value == 0.0)
            return;
        double inv = 1.0 / a->value, inv2 = inv * inv;
        ensure_epoch_zero(a->gradient, a->grad_epoch, g.cur_grad_epoch_) +=
            n.gradient * inv;
        ensure_epoch_zero(a->grad_dot, a->gdot_epoch, g.cur_gdot_epoch_) +=
            n.grad_dot * inv + n.gradient * (-a->dot * inv2);
    }
};

// ===== Add / Subtract (n-ary & binary) =====
template <> struct OpTraits<Operator::Add> {
    static constexpr const char *name = "add";

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

template <> struct OpTraits<Operator::Subtract> {
    static constexpr const char *name = "subtract";

    static inline void forward(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n))
            return;
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_,
                        n.inputs[0]->value - n.inputs[1]->value);
    }
    static inline void forward_dot(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n))
            return;
        set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_,
                        n.inputs[0]->dot - n.inputs[1]->dot);
        touch_epoch(n.val_epoch, g.cur_val_epoch_);
    }
    static inline void backward(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n))
            return;
        ensure_epoch_zero(n.inputs[0]->gradient, n.inputs[0]->grad_epoch,
                          g.cur_grad_epoch_) += n.gradient;
        ensure_epoch_zero(n.inputs[1]->gradient, n.inputs[1]->grad_epoch,
                          g.cur_grad_epoch_) -= n.gradient;
    }
    static inline void hvp_backward(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n))
            return;
        ensure_epoch_zero(n.inputs[0]->gradient, n.inputs[0]->grad_epoch,
                          g.cur_grad_epoch_) += n.gradient;
        ensure_epoch_zero(n.inputs[1]->gradient, n.inputs[1]->grad_epoch,
                          g.cur_grad_epoch_) -= n.gradient;
        ensure_epoch_zero(n.inputs[0]->grad_dot, n.inputs[0]->gdot_epoch,
                          g.cur_gdot_epoch_) += n.grad_dot;
        ensure_epoch_zero(n.inputs[1]->grad_dot, n.inputs[1]->gdot_epoch,
                          g.cur_gdot_epoch_) -= n.grad_dot;
    }
};
// ===== Multiply (n-ary) — optimized with thread_local scratch buffers =====
template <> struct OpTraits<Operator::Multiply> {
    static constexpr const char *name = "multiply";

    // Reusable scratch buffers (per-thread, no cross-graph contention).
    // We resize/assign each time but reuse capacity to avoid heap churn.
    static inline std::vector<double> &tls_vals() {
        thread_local std::vector<double> v;
        return v;
    }
    static inline std::vector<double> &tls_dots() {
        thread_local std::vector<double> v;
        return v;
    }
    static inline std::vector<double> &tls_pre() {
        thread_local std::vector<double> v;
        return v;
    }
    static inline std::vector<double> &tls_suf() {
        thread_local std::vector<double> v;
        return v;
    }

    static inline void build_prefix_suffix(const std::vector<double> &vals,
                                           std::vector<double> &pre,
                                           std::vector<double> &suf) {
        const size_t m = vals.size();
        pre.assign(m + 1, 1.0);
        suf.assign(m + 1, 1.0);
        for (size_t i = 0; i < m; ++i)
            pre[i + 1] = pre[i] * vals[i];
        for (size_t i = m; i-- > 0;)
            suf[i] = suf[i + 1] * vals[i];
    }

    static inline void forward(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n))
            return;
        double p = 1.0;
        for (auto &a : n.inputs)
            p *= a->value;
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, p);
    }

    static inline void forward_dot(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n))
            return;
        const size_t m = n.inputs.size();
        auto &vals = tls_vals();
        vals.resize(m);
        auto &dots = tls_dots();
        dots.resize(m);

        for (size_t i = 0; i < m; ++i) {
            vals[i] = n.inputs[i]->value;
            dots[i] = n.inputs[i]->dot;
        }

        auto &pre = tls_pre();
        auto &suf = tls_suf();
        build_prefix_suffix(vals, pre, suf);

        double ds = 0.0;
        for (size_t i = 0; i < m; ++i)
            ds += dots[i] * pre[i] * suf[i + 1];

        set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_, ds);
        touch_epoch(n.val_epoch, g.cur_val_epoch_);
    }

    static inline void backward(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n))
            return;
        const size_t m = n.inputs.size();
        auto &vals = tls_vals();
        vals.resize(m);
        for (size_t i = 0; i < m; ++i)
            vals[i] = n.inputs[i]->value;

        auto &pre = tls_pre();
        auto &suf = tls_suf();
        build_prefix_suffix(vals, pre, suf);

        for (size_t i = 0; i < m; ++i) {
            const double P_wo_i = pre[i] * suf[i + 1];
            ensure_epoch_zero(n.inputs[i]->gradient, n.inputs[i]->grad_epoch,
                              g.cur_grad_epoch_) += n.gradient * P_wo_i;
        }
    }

    static inline void hvp_backward(ADNode &n, ADGraph &g) {
        if (!_nary_ok(n))
            return;
        const size_t m = n.inputs.size();
        auto &vals = tls_vals();
        vals.resize(m);
        auto &dots = tls_dots();
        dots.resize(m);

        for (size_t i = 0; i < m; ++i) {
            vals[i] = n.inputs[i]->value;
            dots[i] = n.inputs[i]->dot;
        }

        auto &pre = tls_pre();
        auto &suf = tls_suf();
        build_prefix_suffix(vals, pre, suf);

        // NOTE: This remains O(n^2) for the Σ term, but with zero allocations.
        for (size_t i = 0; i < m; ++i) {
            const double P_wo_i = pre[i] * suf[i + 1];

            double sum_term = 0.0;
            for (size_t k = 0; k < m; ++k) {
                if (k == i)
                    continue;
                // Π_{j≠i,k} x_j using prefix/suffix segments:
                const size_t a = (i < k ? i : k);
                const size_t b = (i < k ? k : i);
                const double left = pre[a];
                const double mid = (pre[b] / pre[a + 1]); // product (a+1..b-1)
                const double right = suf[b + 1];
                const double P_wo_i_k = left * mid * right;
                sum_term += dots[k] * P_wo_i_k;
            }

            auto &gacc =
                ensure_epoch_zero(n.inputs[i]->gradient,
                                  n.inputs[i]->grad_epoch, g.cur_grad_epoch_);
            auto &gdacc =
                ensure_epoch_zero(n.inputs[i]->grad_dot,
                                  n.inputs[i]->gdot_epoch, g.cur_gdot_epoch_);

            gacc += n.gradient * P_wo_i;
            gdacc += n.grad_dot * P_wo_i + n.gradient * sum_term;
        }
    }
};

// ===== Divide (binary) =====
template <> struct OpTraits<Operator::Divide> {
    static constexpr const char *name = "divide";

    static inline void forward(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n))
            return;
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_,
                        _safe_div(n.inputs[0]->value, n.inputs[1]->value));
    }
    static inline void forward_dot(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n))
            return;
        auto &a = *n.inputs[0], &b = *n.inputs[1];
        double d = b.value;
        set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_,
                        d != 0.0 ? ((a.dot * d - a.value * b.dot) / (d * d))
                                 : 0.0);
        touch_epoch(n.val_epoch, g.cur_val_epoch_);
    }
    static inline void backward(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n))
            return;
        auto &a = *n.inputs[0], &b = *n.inputs[1];
        const double d = b.value;
        if (d == 0.0)
            return;
        ensure_epoch_zero(a.gradient, a.grad_epoch, g.cur_grad_epoch_) +=
            n.gradient * (1.0 / d);
        ensure_epoch_zero(b.gradient, b.grad_epoch, g.cur_grad_epoch_) +=
            n.gradient * (-a.value / (d * d));
    }
    static inline void hvp_backward(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n))
            return;
        auto &a = *n.inputs[0], &b = *n.inputs[1];
        const double d = b.value;
        if (d == 0.0)
            return;
        const double inv = 1.0 / d, inv2 = inv * inv, inv3 = inv2 * inv;
        ensure_epoch_zero(a.gradient, a.grad_epoch, g.cur_grad_epoch_) +=
            n.gradient * inv;
        ensure_epoch_zero(b.gradient, b.grad_epoch, g.cur_grad_epoch_) +=
            n.gradient * (-a.value * inv2);
        ensure_epoch_zero(a.grad_dot, a.gdot_epoch, g.cur_gdot_epoch_) +=
            n.grad_dot * inv + n.gradient * (-b.dot * inv2);
        ensure_epoch_zero(b.grad_dot, b.gdot_epoch, g.cur_gdot_epoch_) +=
            n.grad_dot * (-a.value * inv2) +
            n.gradient * (-a.dot * inv2 + 2.0 * a.value * b.dot * inv3);
    }
};

template <> struct OpTraits<Operator::Max> {
    static constexpr const char *name = "max";

    static inline void forward(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n)) return;
        double a = n.inputs[0]->value, b = n.inputs[1]->value;
        // tie-break toward 'a' (a >= b)
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, (a >= b ? a : b));
    }

    static inline void forward_dot(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n)) return;
        auto &a = *n.inputs[0], &b = *n.inputs[1];
        if (a.value > b.value) {
            set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_, a.dot);
        } else if (a.value < b.value) {
            set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_, b.dot);
        } else {
            // tie: take gradient of 'a'
            set_epoch_value(n.dot, n.dot_epoch, g.cur_dot_epoch_, a.dot);
        }
        touch_epoch(n.val_epoch, g.cur_val_epoch_);
    }

    static inline void backward(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n)) return;
        auto &a = *n.inputs[0], &b = *n.inputs[1];
        if (a.value > b.value) {
            ensure_epoch_zero(a.gradient, a.grad_epoch, g.cur_grad_epoch_) += n.gradient;
        } else if (a.value < b.value) {
            ensure_epoch_zero(b.gradient, b.grad_epoch, g.cur_grad_epoch_) += n.gradient;
        } else {
            // tie: route all adjoint to 'a'
            ensure_epoch_zero(a.gradient, a.grad_epoch, g.cur_grad_epoch_) += n.gradient;
        }
    }

    static inline void hvp_backward(ADNode &n, ADGraph &g) {
        if (!_binary_ok(n)) return;
        auto &a = *n.inputs[0], &b = *n.inputs[1];
        if (a.value > b.value) {
            ensure_epoch_zero(a.gradient, a.grad_epoch, g.cur_grad_epoch_) += n.gradient;
            ensure_epoch_zero(a.grad_dot, a.gdot_epoch, g.cur_gdot_epoch_) += n.grad_dot;
        } else if (a.value < b.value) {
            ensure_epoch_zero(b.gradient, b.grad_epoch, g.cur_grad_epoch_) += n.gradient;
            ensure_epoch_zero(b.grad_dot, b.gdot_epoch, g.cur_gdot_epoch_) += n.grad_dot;
        } else {
            // tie: route to 'a'
            ensure_epoch_zero(a.gradient, a.grad_epoch, g.cur_grad_epoch_) += n.gradient;
            ensure_epoch_zero(a.grad_dot, a.gdot_epoch, g.cur_gdot_epoch_) += n.grad_dot;
        }
    }
};


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
    default:
        return "unknown";
    }
}
