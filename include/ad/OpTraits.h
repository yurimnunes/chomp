// OpTraits.h — COMPLETE IMPLEMENTATION with Lane Support
#pragma once
#include "ADGraph.h"
#include <cmath>
#include <cstddef>
#include <limits>
#include <algorithm>
#include <memory>

// ---- Optimized helpers ----
inline double _safe_div(double a, double b) noexcept {
    return (b != 0.0) ? (a / b) : 0.0;
}

#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

// Forward declarations for helper access
namespace {
    extern thread_local std::vector<double> g_scratch_values;
    extern thread_local std::vector<size_t> g_scratch_bases;
    
    inline void ensure_scratch_size(size_t n) {
        if (g_scratch_values.size() < n)
            g_scratch_values.resize(n * 2);
    }
    
    inline void ensure_base_size(size_t n) {
        if (g_scratch_bases.size() < n)
            g_scratch_bases.resize(n * 2);
    }
}

// ---- Extended base template ----
template <Operator Op> struct OpTraits {
    static constexpr const char *name = "unknown";
    static inline void forward(ADNode &, ADGraph &) noexcept {}
    static inline void backward(ADNode &, ADGraph &) noexcept {}
    static inline void forward_dot_lanes(ADNode &, ADGraph &, size_t, size_t) noexcept {}
    static inline void backward_lanes(ADNode &, ADGraph &, size_t, size_t) noexcept {}
    static inline void fused_forward(ADNode &, ADGraph &, size_t, size_t) noexcept {}
};

// ===== NULLARY: cte / var =====
template <> struct OpTraits<Operator::cte> {
    static constexpr const char *name = "cte";
    static inline void forward(ADNode &n, ADGraph &g) noexcept {
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, n.value);
    }
    static inline void backward(ADNode &, ADGraph &) noexcept {}
    static inline void forward_dot_lanes(ADNode &, ADGraph &, size_t, size_t) noexcept {}
    static inline void backward_lanes(ADNode &, ADGraph &, size_t, size_t) noexcept {}
    static inline void fused_forward(ADNode &n, ADGraph &g, size_t, size_t) noexcept {
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, n.value);
    }
};

template <> struct OpTraits<Operator::Var> {
    static constexpr const char *name = "var";
    static inline void forward(ADNode &n, ADGraph &g) noexcept {
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, n.value);
    }
    static inline void backward(ADNode &, ADGraph &) noexcept {}
    static inline void forward_dot_lanes(ADNode &, ADGraph &, size_t, size_t) noexcept {}
    static inline void backward_lanes(ADNode &, ADGraph &, size_t, size_t) noexcept {}
    static inline void fused_forward(ADNode &n, ADGraph &g, size_t, size_t) noexcept {
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, n.value);
    }
};

// ===== UNARY OPERATORS =====
template <> struct OpTraits<Operator::Sin> {
    static constexpr const char *name = "sin";
    static inline void forward(ADNode &n, ADGraph &g) noexcept {
        const double x = n.inputs[0]->value;
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, std::sin(x));
    }
    static inline void backward(ADNode &n, ADGraph &g) noexcept {
        const double x = n.inputs[0]->value;
        const double df = std::cos(x);
        ensure_epoch_zero(n.inputs[0]->gradient, n.inputs[0]->grad_epoch, g.cur_grad_epoch_) += n.gradient * df;
    }
    
    static inline void forward_dot_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 1) return;
        const int ix = n.inputs[0]->id;
        const size_t xbase = g.lanes_.base(ix);
        const double cosx = std::cos(n.inputs[0]->value);
        for (size_t l = 0; l < L; ++l)
            g.lanes_.dot[ybase + l] = cosx * g.lanes_.dot[xbase + l];
    }
    
    static inline void backward_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 1) return;
        const size_t xbase = g.lanes_.base(n.inputs[0]->id);
        const double x = n.inputs[0]->value;
        const double c = std::cos(x), s = std::sin(x);
        const double w = n.gradient;
        for (size_t l = 0; l < L; ++l) {
            const double gu = g.lanes_.gdot[ybase + l];
            const double dx = g.lanes_.dot[xbase + l];
            g.lanes_.gdot[xbase + l] += gu * c + w * (-s) * dx;
        }
    }
    
    static inline void fused_forward(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 1) return;
        const double x = n.inputs[0]->value;
        const double sinx = std::sin(x);
        const double cosx = std::cos(x);
        
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, sinx);
        
        const size_t xbase = g.lanes_.base(n.inputs[0]->id);
        for (size_t l = 0; l < L; ++l)
            g.lanes_.dot[ybase + l] = cosx * g.lanes_.dot[xbase + l];
    }
};

template <> struct OpTraits<Operator::Cos> {
    static constexpr const char *name = "cos";
    static inline void forward(ADNode &n, ADGraph &g) noexcept {
        const double x = n.inputs[0]->value;
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, std::cos(x));
    }
    static inline void backward(ADNode &n, ADGraph &g) noexcept {
        const double x = n.inputs[0]->value;
        const double df = -std::sin(x);
        ensure_epoch_zero(n.inputs[0]->gradient, n.inputs[0]->grad_epoch, g.cur_grad_epoch_) += n.gradient * df;
    }
    
    static inline void forward_dot_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 1) return;
        const int ix = n.inputs[0]->id;
        const size_t xbase = g.lanes_.base(ix);
        const double sinx = -std::sin(n.inputs[0]->value);
        for (size_t l = 0; l < L; ++l)
            g.lanes_.dot[ybase + l] = sinx * g.lanes_.dot[xbase + l];
    }
    
    static inline void backward_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 1) return;
        const size_t xbase = g.lanes_.base(n.inputs[0]->id);
        const double x = n.inputs[0]->value;
        const double s = std::sin(x), c = std::cos(x);
        const double w = n.gradient;
        for (size_t l = 0; l < L; ++l) {
            const double gu = g.lanes_.gdot[ybase + l];
            const double dx = g.lanes_.dot[xbase + l];
            g.lanes_.gdot[xbase + l] += gu * (-s) + w * (-c) * dx;
        }
    }
    
    static inline void fused_forward(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 1) return;
        const double x = n.inputs[0]->value;
        const double cosx = std::cos(x);
        const double sinx = std::sin(x);
        
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, cosx);
        
        const size_t xbase = g.lanes_.base(n.inputs[0]->id);
        for (size_t l = 0; l < L; ++l)
            g.lanes_.dot[ybase + l] = -sinx * g.lanes_.dot[xbase + l];
    }
};

template <> struct OpTraits<Operator::Tan> {
    static constexpr const char *name = "tan";
    static inline void forward(ADNode &n, ADGraph &g) noexcept {
        const double x = n.inputs[0]->value;
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, std::tan(x));
    }
    static inline void backward(ADNode &n, ADGraph &g) noexcept {
        const double x = n.inputs[0]->value;
        const double c = std::cos(x);
        const double df = LIKELY(std::abs(c) > 1e-12) ? (1.0 / (c * c)) : 0.0;
        ensure_epoch_zero(n.inputs[0]->gradient, n.inputs[0]->grad_epoch, g.cur_grad_epoch_) += n.gradient * df;
    }
    
    static inline void forward_dot_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 1) return;
        const int ix = n.inputs[0]->id;
        const size_t xbase = g.lanes_.base(ix);
        const double tx = std::tan(n.inputs[0]->value);
        const double sec2 = 1.0 + tx * tx;
        for (size_t l = 0; l < L; ++l)
            g.lanes_.dot[ybase + l] = sec2 * g.lanes_.dot[xbase + l];
    }
    
    static inline void backward_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 1) return;
        const size_t xbase = g.lanes_.base(n.inputs[0]->id);
        const double x = n.inputs[0]->value;
        const double t = std::tan(x);
        const double sec2 = 1.0 + t * t;
        const double w = n.gradient;
        for (size_t l = 0; l < L; ++l) {
            const double gu = g.lanes_.gdot[ybase + l];
            const double dx = g.lanes_.dot[xbase + l];
            g.lanes_.gdot[xbase + l] += gu * sec2 + w * (2.0 * t * sec2) * dx;
        }
    }
    
    static inline void fused_forward(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 1) return;
        const double x = n.inputs[0]->value;
        const double tanx = std::tan(x);
        
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, tanx);
        
        const double sec2 = 1.0 + tanx * tanx;
        const size_t xbase = g.lanes_.base(n.inputs[0]->id);
        for (size_t l = 0; l < L; ++l)
            g.lanes_.dot[ybase + l] = sec2 * g.lanes_.dot[xbase + l];
    }
};

template <> struct OpTraits<Operator::Exp> {
    static constexpr const char *name = "exp";
    static inline void forward(ADNode &n, ADGraph &g) noexcept {
        const double x = n.inputs[0]->value;
        const double result = std::exp(x);
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, result);
    }
    static inline void backward(ADNode &n, ADGraph &g) noexcept {
        const double x = n.inputs[0]->value;
        const double df = std::exp(x);
        ensure_epoch_zero(n.inputs[0]->gradient, n.inputs[0]->grad_epoch, g.cur_grad_epoch_) += n.gradient * df;
    }
    
    static inline void forward_dot_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 1) return;
        const int ix = n.inputs[0]->id;
        const size_t xbase = g.lanes_.base(ix);
        const double ex = std::exp(n.inputs[0]->value);
        for (size_t l = 0; l < L; ++l)
            g.lanes_.dot[ybase + l] = ex * g.lanes_.dot[xbase + l];
    }
    
    static inline void backward_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 1) return;
        const size_t xbase = g.lanes_.base(n.inputs[0]->id);
        const double ex = std::exp(n.inputs[0]->value);
        const double w = n.gradient;
        for (size_t l = 0; l < L; ++l) {
            const double gu = g.lanes_.gdot[ybase + l];
            const double dx = g.lanes_.dot[xbase + l];
            g.lanes_.gdot[xbase + l] += gu * ex + w * ex * dx;
        }
    }
    
    static inline void fused_forward(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 1) return;
        const double x = n.inputs[0]->value;
        const double ex = std::exp(x);
        
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, ex);
        
        const size_t xbase = g.lanes_.base(n.inputs[0]->id);
        for (size_t l = 0; l < L; ++l)
            g.lanes_.dot[ybase + l] = ex * g.lanes_.dot[xbase + l];
    }
};

template <> struct OpTraits<Operator::Log> {
    static constexpr const char *name = "log";
    static inline void forward(ADNode &n, ADGraph &g) noexcept {
        const double x = n.inputs[0]->value;
        const double result = LIKELY(x > 0.0) ? std::log(x) : std::log(1e-16);
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, result);
    }
    static inline void backward(ADNode &n, ADGraph &g) noexcept {
        const double x = n.inputs[0]->value;
        if (LIKELY(x > 0.0)) {
            ensure_epoch_zero(n.inputs[0]->gradient, n.inputs[0]->grad_epoch, g.cur_grad_epoch_) += n.gradient / x;
        }
    }
    
    static inline void forward_dot_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 1) return;
        const int ix = n.inputs[0]->id;
        const size_t xbase = g.lanes_.base(ix);
        const double x = n.inputs[0]->value;
        if (x > 0.0) {
            for (size_t l = 0; l < L; ++l)
                g.lanes_.dot[ybase + l] = g.lanes_.dot[xbase + l] / x;
        } else {
            std::fill_n(&g.lanes_.dot[ybase], L, 0.0);
        }
    }
    
    static inline void backward_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 1) return;
        const size_t xbase = g.lanes_.base(n.inputs[0]->id);
        const double x = n.inputs[0]->value;
        if (x > 0.0) {
            const double invx = 1.0 / x;
            const double invx2 = invx * invx;
            const double w = n.gradient;
            for (size_t l = 0; l < L; ++l) {
                const double gu = g.lanes_.gdot[ybase + l];
                const double dx = g.lanes_.dot[xbase + l];
                g.lanes_.gdot[xbase + l] += gu * invx + w * (-invx2) * dx;
            }
        }
    }
    
    static inline void fused_forward(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 1) return;
        const double x = n.inputs[0]->value;
        
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, std::log(x));
        
        const size_t xbase = g.lanes_.base(n.inputs[0]->id);
        if (x > 0.0) {
            const double invx = 1.0 / x;
            for (size_t l = 0; l < L; ++l)
                g.lanes_.dot[ybase + l] = invx * g.lanes_.dot[xbase + l];
        } else {
            std::fill_n(&g.lanes_.dot[ybase], L, 0.0);
        }
    }
};

template <> struct OpTraits<Operator::Tanh> {
    static constexpr const char *name = "tanh";
    static inline void forward(ADNode &n, ADGraph &g) noexcept {
        const double x = n.inputs[0]->value;
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, std::tanh(x));
    }
    static inline void backward(ADNode &n, ADGraph &g) noexcept {
        const double x = n.inputs[0]->value;
        const double t = std::tanh(x);
        const double df = 1.0 - t * t;
        ensure_epoch_zero(n.inputs[0]->gradient, n.inputs[0]->grad_epoch, g.cur_grad_epoch_) += n.gradient * df;
    }
    
    static inline void forward_dot_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 1) return;
        const int ix = n.inputs[0]->id;
        const size_t xbase = g.lanes_.base(ix);
        const double th = std::tanh(n.inputs[0]->value);
        const double sech2 = 1.0 - th * th;
        for (size_t l = 0; l < L; ++l)
            g.lanes_.dot[ybase + l] = sech2 * g.lanes_.dot[xbase + l];
    }
    
    static inline void backward_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 1) return;
        const size_t xbase = g.lanes_.base(n.inputs[0]->id);
        const double x = n.inputs[0]->value;
        const double t = std::tanh(x);
        const double sech2 = 1.0 - t * t;
        const double fpp = -2.0 * t * sech2;
        const double w = n.gradient;
        for (size_t l = 0; l < L; ++l) {
            const double gu = g.lanes_.gdot[ybase + l];
            const double dx = g.lanes_.dot[xbase + l];
            g.lanes_.gdot[xbase + l] += gu * sech2 + w * fpp * dx;
        }
    }
    
    static inline void fused_forward(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 1) return;
        const double x = n.inputs[0]->value;
        const double th = std::tanh(x);
        
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, th);
        
        const double sech2 = 1.0 - th * th;
        const size_t xbase = g.lanes_.base(n.inputs[0]->id);
        for (size_t l = 0; l < L; ++l)
            g.lanes_.dot[ybase + l] = sech2 * g.lanes_.dot[xbase + l];
    }
};

// Optimized sigmoid helper
inline double fast_sigmoid(double x) noexcept {
    if (x >= 0.0) [[likely]] {
        const double z = std::exp(-x);
        return 1.0 / (1.0 + z);
    } else {
        const double z = std::exp(x);
        return z / (1.0 + z);
    }
}

template <> struct OpTraits<Operator::Silu> {
    static constexpr const char *name = "silu";
    static inline void forward(ADNode &n, ADGraph &g) noexcept {
        const double x = n.inputs[0]->value;
        const double s = fast_sigmoid(x);
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, x * s);
    }
    static inline void backward(ADNode &n, ADGraph &g) noexcept {
        const double x = n.inputs[0]->value;
        const double s = fast_sigmoid(x);
        const double df = s * (1.0 + x * (1.0 - s));
        ensure_epoch_zero(n.inputs[0]->gradient, n.inputs[0]->grad_epoch, g.cur_grad_epoch_) += n.gradient * df;
    }
    
    static inline void forward_dot_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 1) return;
        const int ix = n.inputs[0]->id;
        const size_t xbase = g.lanes_.base(ix);
        const double x = n.inputs[0]->value;
        const double s = fast_sigmoid(x);
        const double sp = s * (1.0 - s);
        const double f1 = s + x * sp;
        for (size_t l = 0; l < L; ++l)
            g.lanes_.dot[ybase + l] = f1 * g.lanes_.dot[xbase + l];
    }
    
    static inline void backward_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 1) return;
        const size_t xbase = g.lanes_.base(n.inputs[0]->id);
        const double x = n.inputs[0]->value;
        const double s = fast_sigmoid(x);
        const double sp = s * (1.0 - s);
        const double f1 = s + x * sp;
        const double f2 = 2.0 * sp + x * sp * (1.0 - 2.0 * s);
        const double w = n.gradient;
        for (size_t l = 0; l < L; ++l) {
            const double gu = g.lanes_.gdot[ybase + l];
            const double dx = g.lanes_.dot[xbase + l];
            g.lanes_.gdot[xbase + l] += gu * f1 + w * f2 * dx;
        }
    }
    
    static inline void fused_forward(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 1) return;
        const double x = n.inputs[0]->value;
        const double sigmoid = fast_sigmoid(x);
        
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, x * sigmoid);
        
        const double sigmoid_prime = sigmoid * (1.0 - sigmoid);
        const double f1 = sigmoid + x * sigmoid_prime;
        const size_t xbase = g.lanes_.base(n.inputs[0]->id);
        for (size_t l = 0; l < L; ++l)
            g.lanes_.dot[ybase + l] = f1 * g.lanes_.dot[xbase + l];
    }
};

template <> struct OpTraits<Operator::Gelu> {
    static constexpr const char *name = "gelu";
    static inline void forward(ADNode &n, ADGraph &g) noexcept {
        const double x = n.inputs[0]->value;
        const double z = x * M_SQRT1_2;
        const double result = 0.5 * x * (1.0 + std::erf(z));
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, result);
    }
    static inline void backward(ADNode &n, ADGraph &g) noexcept {
        const double x = n.inputs[0]->value;
        const double z = x * M_SQRT1_2;
        const double A = std::sqrt(2.0 / M_PI) * std::exp(-0.5 * x * x);
        const double df = 0.5 * (1.0 + std::erf(z)) + 0.5 * x * A;
        ensure_epoch_zero(n.inputs[0]->gradient, n.inputs[0]->grad_epoch, g.cur_grad_epoch_) += n.gradient * df;
    }
    
    static inline void forward_dot_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 1) return;
        const int ix = n.inputs[0]->id;
        const size_t xbase = g.lanes_.base(ix);
        constexpr double inv_sqrt2 = 0.70710678118654752440;
        constexpr double inv_sqrt2pi = 0.39894228040143267794;
        const double x = n.inputs[0]->value;
        const double Phi = 0.5 * (1.0 + std::erf(x * inv_sqrt2));
        const double phi = inv_sqrt2pi * std::exp(-0.5 * x * x);
        const double f1 = Phi + x * phi;
        for (size_t l = 0; l < L; ++l)
            g.lanes_.dot[ybase + l] = f1 * g.lanes_.dot[xbase + l];
    }
    
    static inline void backward_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 1) return;
        const size_t xbase = g.lanes_.base(n.inputs[0]->id);
        constexpr double inv_sqrt2 = 0.70710678118654752440;
        constexpr double inv_sqrt2pi = 0.39894228040143267794;
        const double x = n.inputs[0]->value;
        const double Phi = 0.5 * (1.0 + std::erf(x * inv_sqrt2));
        const double phi = inv_sqrt2pi * std::exp(-0.5 * x * x);
        const double f1 = Phi + x * phi;
        const double f2 = phi * (2.0 - x * x);
        const double w = n.gradient;
        for (size_t l = 0; l < L; ++l) {
            const double gu = g.lanes_.gdot[ybase + l];
            const double dx = g.lanes_.dot[xbase + l];
            g.lanes_.gdot[xbase + l] += gu * f1 + w * f2 * dx;
        }
    }
    
    static inline void fused_forward(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 1) return;
        constexpr double inv_sqrt2 = 0.70710678118654752440;
        constexpr double inv_sqrt2pi = 0.39894228040143267794;
        const double x = n.inputs[0]->value;
        const double erf_term = std::erf(x * inv_sqrt2);
        const double Phi = 0.5 * (1.0 + erf_term);
        const double phi = inv_sqrt2pi * std::exp(-0.5 * x * x);
        
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, x * Phi);
        
        const double f1 = Phi + x * phi;
        const size_t xbase = g.lanes_.base(n.inputs[0]->id);
        for (size_t l = 0; l < L; ++l)
            g.lanes_.dot[ybase + l] = f1 * g.lanes_.dot[xbase + l];
    }
};

template <> struct OpTraits<Operator::Relu> {
    static constexpr const char *name = "relu";
    static inline void forward(ADNode &n, ADGraph &g) noexcept {
        const double x = n.inputs[0]->value;
        const double result = (x > 0.0) ? x : 0.0;
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, result);
    }
    static inline void backward(ADNode &n, ADGraph &g) noexcept {
        const double x = n.inputs[0]->value;
        if (x > 0.0) {
            ensure_epoch_zero(n.inputs[0]->gradient, n.inputs[0]->grad_epoch, g.cur_grad_epoch_) += n.gradient;
        }
    }
    
    static inline void forward_dot_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 1) return;
        const int ix = n.inputs[0]->id;
        const size_t xbase = g.lanes_.base(ix);
        const double xv = n.inputs[0]->value;
        if (xv > 0.0) {
            std::copy_n(&g.lanes_.dot[xbase], L, &g.lanes_.dot[ybase]);
        } else {
            std::fill_n(&g.lanes_.dot[ybase], L, 0.0);
        }
    }
    
    static inline void backward_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 1) return;
        const size_t xbase = g.lanes_.base(n.inputs[0]->id);
        const double x = n.inputs[0]->value;
        if (x > 0.0) {
            for (size_t l = 0; l < L; ++l)
                g.lanes_.gdot[xbase + l] += g.lanes_.gdot[ybase + l];
        }
    }
    
    static inline void fused_forward(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 1) return;
        const double x = n.inputs[0]->value;
        const size_t xbase = g.lanes_.base(n.inputs[0]->id);
        
        if (x > 0.0) {
            set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, x);
            std::copy_n(&g.lanes_.dot[xbase], L, &g.lanes_.dot[ybase]);
        } else {
            set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, 0.0);
            std::fill_n(&g.lanes_.dot[ybase], L, 0.0);
        }
    }
};

// ===== BINARY OPERATORS =====
template <> struct OpTraits<Operator::Subtract> {
    static constexpr const char *name = "subtract";
    static inline void forward(ADNode &n, ADGraph &g) noexcept {
        const double a = n.inputs[0]->value;
        const double b = n.inputs[1]->value;
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, a - b);
    }
    static inline void backward(ADNode &n, ADGraph &g) noexcept {
        const double w = n.gradient;
        ensure_epoch_zero(n.inputs[0]->gradient, n.inputs[0]->grad_epoch, g.cur_grad_epoch_) += w;
        ensure_epoch_zero(n.inputs[1]->gradient, n.inputs[1]->grad_epoch, g.cur_grad_epoch_) -= w;
    }
    
    static inline void forward_dot_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 2) return;
        const int ia = n.inputs[0]->id, ib = n.inputs[1]->id;
        const size_t abase = g.lanes_.base(ia), bbase = g.lanes_.base(ib);
        for (size_t l = 0; l < L; ++l)
            g.lanes_.dot[ybase + l] = g.lanes_.dot[abase + l] - g.lanes_.dot[bbase + l];
    }
    
    static inline void backward_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 2) return;
        const size_t abase = g.lanes_.base(n.inputs[0]->id);
        const size_t bbase = g.lanes_.base(n.inputs[1]->id);
        for (size_t l = 0; l < L; ++l) {
            const double gu = g.lanes_.gdot[ybase + l];
            g.lanes_.gdot[abase + l] += gu;
            g.lanes_.gdot[bbase + l] -= gu;
        }
    }
    
    static inline void fused_forward(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 2) return;
        const double a = n.inputs[0]->value;
        const double b = n.inputs[1]->value;
        const int ia = n.inputs[0]->id, ib = n.inputs[1]->id;
        const size_t abase = g.lanes_.base(ia), bbase = g.lanes_.base(ib);
        
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, a - b);
        
        for (size_t l = 0; l < L; ++l)
            g.lanes_.dot[ybase + l] = g.lanes_.dot[abase + l] - g.lanes_.dot[bbase + l];
    }
};

template <> struct OpTraits<Operator::Divide> {
    static constexpr const char *name = "divide";
    static inline void forward(ADNode &n, ADGraph &g) noexcept {
        const double a = n.inputs[0]->value;
        const double b = n.inputs[1]->value;
        const double result = _safe_div(a, b);
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, result);
    }
    static inline void backward(ADNode &n, ADGraph &g) noexcept {
        const double a = n.inputs[0]->value;
        const double b = n.inputs[1]->value;
        const double w = n.gradient;
        if (LIKELY(b != 0.0)) {
            const double inv_b = 1.0 / b;
            ensure_epoch_zero(n.inputs[0]->gradient, n.inputs[0]->grad_epoch, g.cur_grad_epoch_) += w * inv_b;
            ensure_epoch_zero(n.inputs[1]->gradient, n.inputs[1]->grad_epoch, g.cur_grad_epoch_) += w * (-a * inv_b * inv_b);
        }
    }
    
    static inline void forward_dot_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 2) return;
        const int ia = n.inputs[0]->id, ib = n.inputs[1]->id;
        const size_t abase = g.lanes_.base(ia), bbase = g.lanes_.base(ib);
        const double aval = n.inputs[0]->value, bval = n.inputs[1]->value;
        if (bval == 0.0) {
            std::fill_n(&g.lanes_.dot[ybase], L, 0.0);
        } else {
            const double invb2 = 1.0 / (bval * bval);
            for (size_t l = 0; l < L; ++l) {
                const double ad = g.lanes_.dot[abase + l];
                const double bd = g.lanes_.dot[bbase + l];
                g.lanes_.dot[ybase + l] = (ad * bval - aval * bd) * invb2;
            }
        }
    }
    
    static inline void backward_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 2) return;
        const size_t abase = g.lanes_.base(n.inputs[0]->id);
        const size_t bbase = g.lanes_.base(n.inputs[1]->id);
        const double aval = n.inputs[0]->value, bval = n.inputs[1]->value;
        const double w = n.gradient;
        if (bval != 0.0) {
            const double invb = 1.0 / bval;
            const double invb2 = invb * invb;
            const double invb3 = invb2 * invb;
            for (size_t l = 0; l < L; ++l) {
                const double gu = g.lanes_.gdot[ybase + l];
                const double ad = g.lanes_.dot[abase + l];
                const double bd = g.lanes_.dot[bbase + l];
                g.lanes_.gdot[abase + l] += gu * invb + w * (-bd * invb2);
                g.lanes_.gdot[bbase + l] += gu * (-aval * invb2) + w * ((-ad * bval + 2.0 * aval * bd) * invb3);
            }
        }
    }
    
    static inline void fused_forward(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 2) return;
        const double aval = n.inputs[0]->value;
        const double bval = n.inputs[1]->value;
        const int ia = n.inputs[0]->id, ib = n.inputs[1]->id;
        const size_t abase = g.lanes_.base(ia), bbase = g.lanes_.base(ib);
        
        if (bval == 0.0) {
            set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, std::numeric_limits<double>::infinity());
            std::fill_n(&g.lanes_.dot[ybase], L, 0.0);
        } else {
            set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, aval / bval);
            const double invb2 = 1.0 / (bval * bval);
            for (size_t l = 0; l < L; ++l) {
                const double ad = g.lanes_.dot[abase + l];
                const double bd = g.lanes_.dot[bbase + l];
                g.lanes_.dot[ybase + l] = (ad * bval - aval * bd) * invb2;
            }
        }
    }
};

template <> struct OpTraits<Operator::Max> {
    static constexpr const char *name = "max";
    static inline void forward(ADNode &n, ADGraph &g) noexcept {
        const double a = n.inputs[0]->value;
        const double b = n.inputs[1]->value;
        const double result = (a >= b) ? a : b;
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, result);
    }
    static inline void backward(ADNode &n, ADGraph &g) noexcept {
        const double a = n.inputs[0]->value;
        const double b = n.inputs[1]->value;
        if (a >= b) {
            ensure_epoch_zero(n.inputs[0]->gradient, n.inputs[0]->grad_epoch, g.cur_grad_epoch_) += n.gradient;
        } else {
            ensure_epoch_zero(n.inputs[1]->gradient, n.inputs[1]->grad_epoch, g.cur_grad_epoch_) += n.gradient;
        }
    }
    
    static inline void forward_dot_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 2) return;
        const int ia = n.inputs[0]->id, ib = n.inputs[1]->id;
        const size_t abase = g.lanes_.base(ia), bbase = g.lanes_.base(ib);
        const double a = n.inputs[0]->value, b = n.inputs[1]->value;
        if (a >= b) {
            std::copy_n(&g.lanes_.dot[abase], L, &g.lanes_.dot[ybase]);
        } else {
            std::copy_n(&g.lanes_.dot[bbase], L, &g.lanes_.dot[ybase]);
        }
    }
    
    static inline void backward_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 2) return;
        const size_t abase = g.lanes_.base(n.inputs[0]->id);
        const size_t bbase = g.lanes_.base(n.inputs[1]->id);
        const double a = n.inputs[0]->value, b = n.inputs[1]->value;
        if (a >= b) {
            for (size_t l = 0; l < L; ++l)
                g.lanes_.gdot[abase + l] += g.lanes_.gdot[ybase + l];
        } else {
            for (size_t l = 0; l < L; ++l)
                g.lanes_.gdot[bbase + l] += g.lanes_.gdot[ybase + l];
        }
    }
    
    static inline void fused_forward(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        if (n.inputs.size() != 2) return;
        const double aval = n.inputs[0]->value;
        const double bval = n.inputs[1]->value;
        const int ia = n.inputs[0]->id, ib = n.inputs[1]->id;
        const size_t abase = g.lanes_.base(ia), bbase = g.lanes_.base(ib);
        
        if (aval >= bval) {
            set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, aval);
            std::copy_n(&g.lanes_.dot[abase], L, &g.lanes_.dot[ybase]);
        } else {
            set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, bval);
            std::copy_n(&g.lanes_.dot[bbase], L, &g.lanes_.dot[ybase]);
        }
    }
};

// ===== N-ARY OPERATORS =====
template <> struct OpTraits<Operator::Add> {
    static constexpr const char *name = "add";
    static inline void forward(ADNode &n, ADGraph &g) noexcept {
        const size_t m = n.inputs.size();
        double sum = 0.0;
        
        // Unroll for small cases
        switch (m) {
            case 2:
                sum = n.inputs[0]->value + n.inputs[1]->value;
                break;
            case 3:
                sum = n.inputs[0]->value + n.inputs[1]->value + n.inputs[2]->value;
                break;
            case 4:
                sum = n.inputs[0]->value + n.inputs[1]->value + 
                      n.inputs[2]->value + n.inputs[3]->value;
                break;
            default:
                for (const auto& input : n.inputs) {
                    sum += input->value;
                }
                break;
        }
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, sum);
    }

    static inline void backward(ADNode &n, ADGraph &g) noexcept {
        const double w = n.gradient;
        for (const auto& input : n.inputs) {
            ensure_epoch_zero(input->gradient, input->grad_epoch, g.cur_grad_epoch_) += w;
        }
    }
    
    static inline void forward_dot_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        const size_t m = n.inputs.size();
        if (m == 0) return;
        
        std::fill_n(&g.lanes_.dot[ybase], L, 0.0);
        for (const auto &inp : n.inputs) {
            const size_t ibase = g.lanes_.base(inp->id);
            for (size_t l = 0; l < L; ++l)
                g.lanes_.dot[ybase + l] += g.lanes_.dot[ibase + l];
        }
    }
    
    static inline void backward_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        for (const auto &inp : n.inputs) {
            const size_t ibase = g.lanes_.base(inp->id);
            for (size_t l = 0; l < L; ++l)
                g.lanes_.gdot[ibase + l] += g.lanes_.gdot[ybase + l];
        }
    }
    
    static inline void fused_forward(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        const size_t m = n.inputs.size();
        if (m == 0) return;
        
        // FUSED: Compute primal AND dot lanes together
        double sum_val = 0.0;
        std::fill_n(&g.lanes_.dot[ybase], L, 0.0);

        for (const auto& inp : n.inputs) {
            sum_val += inp->value; // Primal computation
            
            // Dot lane computation
            const size_t ibase = g.lanes_.base(inp->id);
            for (size_t l = 0; l < L; ++l)
                g.lanes_.dot[ybase + l] += g.lanes_.dot[ibase + l];
        }
        
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, sum_val);
    }
};
template <> struct OpTraits<Operator::Multiply> {
    static constexpr const char *name = "multiply";
    
    static inline void forward(ADNode &n, ADGraph &g) noexcept {
        const size_t m = n.inputs.size();
        
        // Fast binary path
        if (m == 2) [[likely]] {
            const double a = n.inputs[0]->value;
            const double b = n.inputs[1]->value;
            set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, a * b);
            return;
        }
        
        // N-ary with early zero detection
        size_t zc = 0;
        double prod_nz = 1.0;
        for (const auto& input : n.inputs) {
            const double v = input->value;
            if (UNLIKELY(v == 0.0)) {
                if (++zc > 1) break; // Multiple zeros, result is 0
            } else {
                prod_nz *= v;
            }
        }
        const double result = (zc == 0) ? prod_nz : 0.0;
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, result);
    }

    static inline void backward(ADNode &n, ADGraph &g) noexcept {
        const size_t m = n.inputs.size();
        
        // Fast binary path
        if (m == 2) [[likely]] {
            const double a = n.inputs[0]->value;
            const double b = n.inputs[1]->value;
            const double w = n.gradient;
            ensure_epoch_zero(n.inputs[0]->gradient, n.inputs[0]->grad_epoch, g.cur_grad_epoch_) += w * b;
            ensure_epoch_zero(n.inputs[1]->gradient, n.inputs[1]->grad_epoch, g.cur_grad_epoch_) += w * a;
            return;
        }
        
        // N-ary zero handling
        size_t zc = 0, zi = SIZE_MAX;
        double prod_nz = 1.0;

        for (size_t i = 0; i < m; ++i) {
            const double v = n.inputs[i]->value;
            if (UNLIKELY(v == 0.0)) {
                if (++zc == 1) zi = i;
                else return; // Multiple zeros -> all gradients zero
            } else {
                prod_nz *= v;
            }
        }

        const double w = n.gradient;
        if (zc == 1) {
            // Single zero case
            ensure_epoch_zero(n.inputs[zi]->gradient, n.inputs[zi]->grad_epoch, g.cur_grad_epoch_) += w * prod_nz;
        } else {
            // No zeros case
            for (size_t i = 0; i < m; ++i) {
                const double xi = n.inputs[i]->value;
                ensure_epoch_zero(n.inputs[i]->gradient, n.inputs[i]->grad_epoch, g.cur_grad_epoch_) += w * (prod_nz / xi);
            }
        }
    }
    
    static inline void forward_dot_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        const size_t m = n.inputs.size();
        if (m == 0) return;
        
        if (m == 2) [[likely]] {
            const int ia = n.inputs[0]->id, ib = n.inputs[1]->id;
            const size_t abase = g.lanes_.base(ia), bbase = g.lanes_.base(ib);
            const double aval = n.inputs[0]->value, bval = n.inputs[1]->value;
            
            for (size_t l = 0; l < L; ++l) {
                const double ad = g.lanes_.dot[abase + l];
                const double bd = g.lanes_.dot[bbase + l];
                g.lanes_.dot[ybase + l] = ad * bval + aval * bd;
            }
        } else {
            // N-ary multiply lane forward
            ensure_scratch_size(m);
            ensure_base_size(m);
            
            size_t zc = 0, zidx = 0;
            for (size_t j = 0; j < m; ++j) {
                const double vj = n.inputs[j]->value;
                g_scratch_values[j] = vj;
                g_scratch_bases[j] = g.lanes_.base(n.inputs[j]->id);
                if (vj == 0.0 && (++zc == 1))
                    zidx = j;
            }

            if (zc >= 2) {
                std::fill_n(&g.lanes_.dot[ybase], L, 0.0);
                return;
            }

            if (zc == 1) {
                double prod_nz = 1.0;
                for (size_t j = 0; j < m; ++j)
                    if (j != zidx)
                        prod_nz *= g_scratch_values[j];
                const size_t zb = g_scratch_bases[zidx];
                for (size_t l = 0; l < L; ++l)
                    g.lanes_.dot[ybase + l] = g.lanes_.dot[zb + l] * prod_nz;
                return;
            }

            // Use prefix/suffix products for n-ary case
            ensure_scratch_size(3 * m);
            double *prefix = g_scratch_values.data();
            double *suffix = g_scratch_values.data() + m;
            double *contrib = g_scratch_values.data() + 2 * m;

            prefix[0] = 1.0;
            for (size_t i = 1; i < m; ++i)
                prefix[i] = prefix[i - 1] * n.inputs[i - 1]->value;
            suffix[m - 1] = 1.0;
            for (size_t i = m - 1; i-- > 0;)
                suffix[i] = suffix[i + 1] * n.inputs[i + 1]->value;

            for (size_t i = 0; i < m; ++i)
                contrib[i] = prefix[i] * suffix[i];

            for (size_t l = 0; l < L; ++l) {
                double yd = 0.0;
                for (size_t i = 0; i < m; ++i)
                    yd += contrib[i] * g.lanes_.dot[g_scratch_bases[i] + l];
                g.lanes_.dot[ybase + l] = yd;
            }
        }
    }
    
    static inline void backward_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        const size_t m = n.inputs.size();
        if (m == 0) return;
        const double w = n.gradient;

        if (m == 2) [[likely]] {
            const size_t abase = g.lanes_.base(n.inputs[0]->id);
            const size_t bbase = g.lanes_.base(n.inputs[1]->id);
            const double aval = n.inputs[0]->value, bval = n.inputs[1]->value;
            
            for (size_t l = 0; l < L; ++l) {
                const double gu = g.lanes_.gdot[ybase + l];
                const double ad = g.lanes_.dot[abase + l];
                const double bd = g.lanes_.dot[bbase + l];
                g.lanes_.gdot[abase + l] += gu * bval + w * bd;
                g.lanes_.gdot[bbase + l] += gu * aval + w * ad;
            }
        } else {
            // N-ary multiply lane backward
            ensure_scratch_size(m);
            ensure_base_size(m);

            size_t zero_count = 0, zero_idx = 0;
            double prod_nz = 1.0;
            for (size_t j = 0; j < m; ++j) {
                const double vj = n.inputs[j]->value;
                g_scratch_values[j] = vj;
                g_scratch_bases[j] = g.lanes_.base(n.inputs[j]->id);
                if (vj == 0.0) {
                    if (++zero_count == 1)
                        zero_idx = j;
                } else {
                    prod_nz *= vj;
                }
            }

            for (size_t l = 0; l < L; ++l) {
                const double gu = g.lanes_.gdot[ybase + l];

                if (zero_count == 0) {
                    // sum over dj / vj
                    double sum_d_over_v = 0.0;
                    for (size_t j = 0; j < m; ++j)
                        sum_d_over_v += g.lanes_.dot[g_scratch_bases[j] + l] / g_scratch_values[j];

                    for (size_t i = 0; i < m; ++i) {
                        const double vi = g_scratch_values[i];
                        const double di = g.lanes_.dot[g_scratch_bases[i] + l];
                        g.lanes_.gdot[g_scratch_bases[i] + l] +=
                            gu * (prod_nz / vi) + w * ((prod_nz / vi) * (sum_d_over_v - di / vi));
                    }
                } else if (zero_count == 1) {
                    const double dz = g.lanes_.dot[g_scratch_bases[zero_idx] + l];
                    g.lanes_.gdot[g_scratch_bases[zero_idx] + l] += gu * prod_nz;

                    for (size_t i = 0; i < m; ++i)
                        if (i != zero_idx) {
                            const double vi = g_scratch_values[i];
                            g.lanes_.gdot[g_scratch_bases[i] + l] += w * (dz * (prod_nz / vi));
                        }
                }
                // if >=2 zeros: nothing to do
            }
        }
    }
    
    static inline void fused_forward(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        const size_t m = n.inputs.size();
        if (m == 0) return;

        if (m == 2) [[likely]] {
            const double aval = n.inputs[0]->value;
            const double bval = n.inputs[1]->value;
            const size_t abase = g.lanes_.base(n.inputs[0]->id);
            const size_t bbase = g.lanes_.base(n.inputs[1]->id);

            // FUSED: Primal and dot computation
            set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, aval * bval);

            for (size_t l = 0; l < L; ++l) {
                const double ad = g.lanes_.dot[abase + l];
                const double bd = g.lanes_.dot[bbase + l];
                g.lanes_.dot[ybase + l] = ad * bval + aval * bd;
            }
        } else {
            // N-ary fused forward (combine primal + dot logic)
            ensure_scratch_size(m);
            ensure_base_size(m);

            size_t zero_count = 0, zero_idx = 0;
            double prod_val = 1.0;
            double prod_nz = 1.0;

            // Single pass: collect values, bases, and compute primal
            for (size_t j = 0; j < m; ++j) {
                const double vj = n.inputs[j]->value;
                g_scratch_values[j] = vj;
                g_scratch_bases[j] = g.lanes_.base(n.inputs[j]->id);

                prod_val *= vj; // Primal product

                if (vj == 0.0) {
                    if (++zero_count == 1)
                        zero_idx = j;
                } else {
                    prod_nz *= vj;
                }
            }

            set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, prod_val);

            // Dot computation using precomputed values
            for (size_t l = 0; l < L; ++l) {
                double yd = 0.0;
                if (zero_count == 0) {
                    for (size_t j = 0; j < m; ++j) {
                        const double dj = g.lanes_.dot[g_scratch_bases[j] + l];
                        yd += dj * (prod_nz / g_scratch_values[j]);
                    }
                } else if (zero_count == 1) {
                    const double dz = g.lanes_.dot[g_scratch_bases[zero_idx] + l];
                    yd = dz * prod_nz;
                }
                g.lanes_.dot[ybase + l] = yd;
            }
        }
    }
};

// ===== SOFTMAX =====
template <> struct OpTraits<Operator::Softmax> {
    static constexpr const char *name = "softmax";
    static constexpr size_t STACK_THRESHOLD = 32;

    static inline void forward(ADNode &n, ADGraph &g) noexcept {
        const size_t m = n.inputs.size();
        
        // Stack allocation for small arrays
        double stack_x[STACK_THRESHOLD];
        std::unique_ptr<double[]> heap_x;
        double* x = (m <= STACK_THRESHOLD) ? stack_x : 
                   (heap_x = std::make_unique<double[]>(m)).get();

        // Collect inputs and find max in one pass
        double xmax = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < m; ++i) {
            x[i] = n.inputs[i]->value;
            if (x[i] > xmax) xmax = x[i];
        }

        // Compute stable softmax
        double Z = 0.0;
        for (size_t i = 0; i < m; ++i) {
            Z += std::exp(x[i] - xmax);
        }
        if (UNLIKELY(Z <= 0.0)) Z = 1.0;

        const double result = std::exp(x[0] - xmax) / Z;
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, result);
    }

    static inline void backward(ADNode &n, ADGraph &g) noexcept {
        const size_t m = n.inputs.size();
        
        // Stack allocation for small arrays
        double stack_x[STACK_THRESHOLD], stack_y[STACK_THRESHOLD];
        std::unique_ptr<double[]> heap_x, heap_y;
        double* x = (m <= STACK_THRESHOLD) ? stack_x : 
                   (heap_x = std::make_unique<double[]>(m)).get();
        double* y = (m <= STACK_THRESHOLD) ? stack_y : 
                   (heap_y = std::make_unique<double[]>(m)).get();

        // Recompute forward pass values
        double xmax = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < m; ++i) {
            x[i] = n.inputs[i]->value;
            if (x[i] > xmax) xmax = x[i];
        }

        double Z = 0.0;
        for (size_t i = 0; i < m; ++i) {
            y[i] = std::exp(x[i] - xmax);
            Z += y[i];
        }
        if (UNLIKELY(Z <= 0.0)) Z = 1.0;
        
        for (size_t i = 0; i < m; ++i) {
            y[i] /= Z;
        }

        // Compute gradients: ∂y₀/∂xₖ = y₀ * (δ_{k0} - yₖ)
        const double y0 = y[0];
        const double w = n.gradient;

        for (size_t k = 0; k < m; ++k) {
            const double delta_k0 = (k == 0) ? 1.0 : 0.0;
            const double grad_k = y0 * (delta_k0 - y[k]);
            ensure_epoch_zero(n.inputs[k]->gradient, n.inputs[k]->grad_epoch, g.cur_grad_epoch_) += w * grad_k;
        }
    }
    
    static inline void forward_dot_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        const size_t m = n.inputs.size();
        if (m == 0) return;

        ensure_scratch_size(m);
        // stable softmax over AoS primals
        double mmax = -std::numeric_limits<double>::infinity();
        for (size_t j = 0; j < m; ++j)
            mmax = std::max(mmax, n.inputs[j]->value);
        double Z = 0.0;
        for (size_t j = 0; j < m; ++j) {
            g_scratch_values[j] = std::exp(n.inputs[j]->value - mmax);
            Z += g_scratch_values[j];
        }
        for (size_t j = 0; j < m; ++j)
            g_scratch_values[j] /= Z; // s_j

        const size_t i = 0; // component index (first input)
        const double si = g_scratch_values[i];

        for (size_t l = 0; l < L; ++l) {
            double avg = 0.0;
            for (size_t j = 0; j < m; ++j)
                avg += g_scratch_values[j] * g.lanes_.dot[g.lanes_.base(n.inputs[j]->id) + l];
            const size_t ibase = g.lanes_.base(n.inputs[i]->id);
            const double di = g.lanes_.dot[ibase + l];
            g.lanes_.dot[ybase + l] = si * (di - avg);
        }
    }
    
    static inline void backward_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        const size_t m = n.inputs.size();
        if (m == 0) return;
        const double w = n.gradient;

        ensure_scratch_size(m);
        // recompute softmax s_j from AoS primals
        double mmax = -std::numeric_limits<double>::infinity();
        for (size_t j = 0; j < m; ++j)
            mmax = std::max(mmax, n.inputs[j]->value);
        double Z = 0.0;
        for (size_t j = 0; j < m; ++j) {
            g_scratch_values[j] = std::exp(n.inputs[j]->value - mmax);
            Z += g_scratch_values[j];
        }
        for (size_t j = 0; j < m; ++j)
            g_scratch_values[j] /= Z;
        const size_t i = 0;
        const double si = g_scratch_values[i];

        for (size_t l = 0; l < L; ++l) {
            const double gu = g.lanes_.gdot[ybase + l];

            double avg = 0.0;
            for (size_t j = 0; j < m; ++j)
                avg += g_scratch_values[j] * g.lanes_.dot[g.lanes_.base(n.inputs[j]->id) + l];

            const size_t ibase = g.lanes_.base(n.inputs[i]->id);
            const double di = g.lanes_.dot[ibase + l];
            const double sdot_i = si * (di - avg);

            for (size_t j = 0; j < m; ++j) {
                const size_t jbase = g.lanes_.base(n.inputs[j]->id);
                const double sj = g_scratch_values[j];
                const double dj = g.lanes_.dot[jbase + l];
                const double sdot_j = sj * (dj - avg);

                // gu * J + w * J'[d]
                double add = gu * si * ((j == i) ? (1.0 - sj) : -sj);
                if (j == i)
                    add += w * sdot_i * (1.0 - 2.0 * si);
                else
                    add += w * (-sdot_i * sj - si * sdot_j);

                g.lanes_.gdot[jbase + l] += add;
            }
        }
    }
    
    static inline void fused_forward(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        const size_t m = n.inputs.size();
        if (m == 0) return;

        ensure_scratch_size(m);

        // FUSED: Compute softmax and prepare for dot computation
        // First pass: find max for numerical stability
        double mmax = -std::numeric_limits<double>::infinity();
        for (size_t j = 0; j < m; ++j)
            mmax = std::max(mmax, n.inputs[j]->value);

        // Second pass: compute exp(x_i - max) and sum
        double Z = 0.0;
        for (size_t j = 0; j < m; ++j) {
            g_scratch_values[j] = std::exp(n.inputs[j]->value - mmax);
            Z += g_scratch_values[j];
        }

        // Third pass: normalize to get softmax values
        for (size_t j = 0; j < m; ++j)
            g_scratch_values[j] /= Z; // s_j = softmax values

        // Store primal result (assuming this is component 0)
        const size_t i = 0; // component index (first input)
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, g_scratch_values[i]);

        // Dot computation for softmax derivative
        const double si = g_scratch_values[i];

        for (size_t l = 0; l < L; ++l) {
            double avg = 0.0;
            for (size_t j = 0; j < m; ++j)
                avg += g_scratch_values[j] * g.lanes_.dot[g.lanes_.base(n.inputs[j]->id) + l];

            const size_t ibase = g.lanes_.base(n.inputs[i]->id);
            const double di = g.lanes_.dot[ibase + l];
            g.lanes_.dot[ybase + l] = si * (di - avg);
        }
    }
};

// ===== UNARY: ABS =====
template <> struct OpTraits<Operator::Abs> {
    static constexpr const char *name = "abs";
    static inline void forward(ADNode &n, ADGraph &g) noexcept {
        const double x = n.inputs[0]->value;
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, std::abs(x));
    }
    static inline void backward(ADNode &n, ADGraph &g) noexcept {
        const double x = n.inputs[0]->value;
        const double df = (x > 0.0) - (x < 0.0); // sign, 0 at 0
        if (df != 0.0)
            ensure_epoch_zero(n.inputs[0]->gradient, n.inputs[0]->grad_epoch, g.cur_grad_epoch_) += n.gradient * df;
    }
    static inline void forward_dot_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        const double x = n.inputs[0]->value;
        const size_t xbase = g.lanes_.base(n.inputs[0]->id);
        const double s = (x > 0.0) - (x < 0.0);
        if (s == 0.0) { std::fill_n(&g.lanes_.dot[ybase], L, 0.0); return; }
        for (size_t l = 0; l < L; ++l) g.lanes_.dot[ybase + l] = s * g.lanes_.dot[xbase + l];
    }
    static inline void backward_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        const double x = n.inputs[0]->value;
        const size_t xbase = g.lanes_.base(n.inputs[0]->id);
        const double s = (x > 0.0) - (x < 0.0);
        if (s == 0.0) return;
        for (size_t l = 0; l < L; ++l) g.lanes_.gdot[xbase + l] += s * g.lanes_.gdot[ybase + l];
    }
    static inline void fused_forward(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        const double x = n.inputs[0]->value;
        const size_t xbase = g.lanes_.base(n.inputs[0]->id);
        const double s = (x > 0.0) - (x < 0.0);
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, std::abs(x));
        if (s == 0.0) { std::fill_n(&g.lanes_.dot[ybase], L, 0.0); return; }
        for (size_t l = 0; l < L; ++l) g.lanes_.dot[ybase + l] = s * g.lanes_.dot[xbase + l];
    }
};

// ===== UNARY: SQRT =====
template <> struct OpTraits<Operator::Sqrt> {
    static constexpr const char *name = "sqrt";
    static inline void forward(ADNode &n, ADGraph &g) noexcept {
        double x = n.inputs[0]->value;
        x = (x > 0.0) ? x : 0.0; // clamp for stability
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, std::sqrt(x));
    }
    static inline void backward(ADNode &n, ADGraph &g) noexcept {
        const double x = n.inputs[0]->value;
        if (x > 0.0) {
            const double df = 0.5 / std::sqrt(x);
            ensure_epoch_zero(n.inputs[0]->gradient, n.inputs[0]->grad_epoch, g.cur_grad_epoch_) += n.gradient * df;
        }
    }
    static inline void forward_dot_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        const double x = n.inputs[0]->value;
        const size_t xbase = g.lanes_.base(n.inputs[0]->id);
        if (x <= 0.0) { std::fill_n(&g.lanes_.dot[ybase], L, 0.0); return; }
        const double df = 0.5 / std::sqrt(x);
        for (size_t l = 0; l < L; ++l) g.lanes_.dot[ybase + l] = df * g.lanes_.dot[xbase + l];
    }
    static inline void backward_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        const double x = n.inputs[0]->value;
        if (x <= 0.0) return;
        const size_t xbase = g.lanes_.base(n.inputs[0]->id);
        const double df = 0.5 / std::sqrt(x);
        const double fpp = -0.25 / (x * std::sqrt(x)); // d(0.5 x^-1/2)/dx
        const double w = n.gradient;
        for (size_t l = 0; l < L; ++l) {
            const double gu = g.lanes_.gdot[ybase + l];
            const double dx = g.lanes_.dot[xbase + l];
            g.lanes_.gdot[xbase + l] += gu * df + w * fpp * dx;
        }
    }
    static inline void fused_forward(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        double x = n.inputs[0]->value;
        const size_t xbase = g.lanes_.base(n.inputs[0]->id);
        if (x <= 0.0) {
            set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, 0.0);
            std::fill_n(&g.lanes_.dot[ybase], L, 0.0);
            return;
        }
        const double y = std::sqrt(x);
        const double df = 0.5 / y;
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, y);
        for (size_t l = 0; l < L; ++l) g.lanes_.dot[ybase + l] = df * g.lanes_.dot[xbase + l];
    }
};

// ===== BINARY: POW (a^b) =====
// NOTE: well-defined gradients for a>0. For a<=0 we zero-out lane/backward
// to avoid NaNs when b is non-integer. This mirrors the defensive style in Log.
template <> struct OpTraits<Operator::Pow> {
    static constexpr const char *name = "pow";
    static inline void forward(ADNode &n, ADGraph &g) noexcept {
        const double a = n.inputs[0]->value;
        const double b = n.inputs[1]->value;
        double y;
        if (a > 0.0) {
            y = std::pow(a, b);
        } else if (a == 0.0) {
            y = (b > 0.0) ? 0.0 : std::numeric_limits<double>::infinity();
        } else {
            // negative base: still compute value (might be NaN for non-integer b)
            y = std::pow(a, b);
            if (!std::isfinite(y)) y = 0.0; // defensive clamp
        }
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, y);
    }
    static inline void backward(ADNode &n, ADGraph &g) noexcept {
        const double a = n.inputs[0]->value;
        const double b = n.inputs[1]->value;
        const double w = n.gradient;

        if (a > 0.0) {
            const double y = std::pow(a, b);
            // dy/da = b*a^{b-1} = y * b / a
            ensure_epoch_zero(n.inputs[0]->gradient, n.inputs[0]->grad_epoch, g.cur_grad_epoch_) += w * (y * b / a);
            // dy/db = a^b * ln(a) = y * ln(a)
            ensure_epoch_zero(n.inputs[1]->gradient, n.inputs[1]->grad_epoch, g.cur_grad_epoch_) += w * (y * std::log(a));
        } else {
            // Undefined / non-smooth; keep stable (no contribution)
        }
    }
    static inline void forward_dot_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        const double a = n.inputs[0]->value;
        const double b = n.inputs[1]->value;
        const size_t abase = g.lanes_.base(n.inputs[0]->id);
        const size_t bbase = g.lanes_.base(n.inputs[1]->id);

        if (a <= 0.0) { std::fill_n(&g.lanes_.dot[ybase], L, 0.0); return; }

        const double y = std::pow(a, b);
        const double da = y * b / a;
        const double db = y * std::log(a);
        for (size_t l = 0; l < L; ++l) {
            const double ad = g.lanes_.dot[abase + l];
            const double bd = g.lanes_.dot[bbase + l];
            g.lanes_.dot[ybase + l] = da * ad + db * bd;
        }
    }
    static inline void backward_lanes(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        const double a = n.inputs[0]->value;
        const double b = n.inputs[1]->value;
        if (a <= 0.0) return;

        const size_t abase = g.lanes_.base(n.inputs[0]->id);
        const size_t bbase = g.lanes_.base(n.inputs[1]->id);
        const double y = std::pow(a, b);
        const double da = y * b / a;
        const double db = y * std::log(a);

        // second derivatives (for gdot transport)
        const double ddaa = y * (b * (b - 1.0)) / (a * a);  // ∂(da)/∂a
        const double ddab = y * (std::log(a) + b / a);      // ∂(da)/∂b
        const double ddba = db * (b / a);                   // ∂(db)/∂a = y*ln(a)*b/a
        const double ddbb = y * std::log(a) * std::log(a);  // ∂(db)/∂b

        const double w = n.gradient;
        for (size_t l = 0; l < L; ++l) {
            const double gu = g.lanes_.gdot[ybase + l];
            const double ad = g.lanes_.dot[abase + l];
            const double bd = g.lanes_.dot[bbase + l];

            // chain: g_x += gu * J + w * J'[d]
            g.lanes_.gdot[abase + l] += gu * da + w * (ddaa * ad + ddab * bd);
            g.lanes_.gdot[bbase + l] += gu * db + w * (ddba * ad + ddbb * bd);
        }
    }
    static inline void fused_forward(ADNode &n, ADGraph &g, size_t L, size_t ybase) noexcept {
        const double a = n.inputs[0]->value;
        const double b = n.inputs[1]->value;
        const size_t abase = g.lanes_.base(n.inputs[0]->id);
        const size_t bbase = g.lanes_.base(n.inputs[1]->id);

        if (a <= 0.0) {
            set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, 0.0);
            std::fill_n(&g.lanes_.dot[ybase], L, 0.0);
            return;
        }

        const double y = std::pow(a, b);
        set_epoch_value(n.value, n.val_epoch, g.cur_val_epoch_, y);

        const double da = y * b / a;
        const double db = y * std::log(a);
        for (size_t l = 0; l < L; ++l) {
            const double ad = g.lanes_.dot[abase + l];
            const double bd = g.lanes_.dot[bbase + l];
            g.lanes_.dot[ybase + l] = da * ad + db * bd;
        }
    }
};


// =====================================================================
//                     Name mapping (simplified)
// =====================================================================
inline const char *op_name(Operator op) noexcept {
    // Use static array for O(1) lookup instead of switch
    static constexpr const char* names[] = {
        "var", "cte", "add", "subtract", "multiply", "divide",
        "sin", "cos", "tan", "exp", "log", "tanh", "relu", 
        "max", "gelu", "silu", "softmax", "abs", "sqrt", "pow"
    };
    
    const int idx = static_cast<int>(op);
    return (idx >= 0 && idx < 17) ? names[idx] : "unknown";
}