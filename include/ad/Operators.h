#pragma once
#include "ADGraph.h"
#include "Expression.h"
#include "Variable.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// If you have robin_map/robin_set etc., include them above as needed.

// ============================================================================
// Local type aliases (match your codebase)
using VariablePtr   = std::shared_ptr<Variable>;
using ExpressionPtr = std::shared_ptr<Expression>;
using ADNodePtr     = std::shared_ptr<ADNode>;
using ADGraphPtr    = std::shared_ptr<ADGraph>;

// ============================================================================
// Minimal hot helpers (self-contained)
// ============================================================================

// Fast constant detection with optional out value
[[gnu::always_inline, gnu::hot]]
inline bool is_const(const ADNodePtr& n, double* out = nullptr) noexcept {
    if (n && n->type == Operator::cte) [[likely]] {
        if (out) *out = n->value;
        return true;
    }
    return false;
}

// Minimal constant-node helper (no dependency on ADGraph internals)
[[gnu::always_inline, gnu::hot]]
inline ADNodePtr make_cte(const ADGraphPtr &g, double v) {
    auto c = std::make_shared<ADNode>();
    c->type  = Operator::cte;
    c->value = v;
    if (g) [[likely]] g->addNode(c);
    return c;
}

// Optimized input attachment with early returns
[[gnu::always_inline, gnu::hot]]
inline ADNodePtr attach_input(const ExpressionPtr &e, const ADGraphPtr &g) {
    if (!e) [[unlikely]] return nullptr;

    // Fast path: check rootNode first (more specific)
    if (e->rootNode) [[likely]] {
        if (g) g->adoptSubgraph(e->rootNode);
        return e->rootNode;
    }
    if (e->node) [[likely]] {
        if (g) g->adoptSubgraph(e->node);
        return e->node;
    }
    return nullptr;
}

// Optimized flattening with capacity hints
[[gnu::always_inline]]
inline void flatten_into(Operator op, const ADNodePtr& child, std::vector<ADNodePtr>& dst) {
    if (child && child->type == op) [[likely]] {
        dst.insert(dst.end(), child->inputs.begin(), child->inputs.end());
    } else {
        dst.push_back(child);
    }
}

// Fast n-ary node construction
[[gnu::always_inline]]
inline ADNodePtr build_nary_node(const ADGraphPtr& g, Operator op,
                                 std::vector<ADNodePtr>&& ins) {
    auto out  = std::make_shared<ADNode>();
    out->type = op;
    out->inputs = std::move(ins);
    if (g) g->addNode(out);
    return out;
}

// Consolidate all constant children in an n-ary Add/Multiply and drop identities.
// Also short-circuits Multiply if a zero is found.
[[gnu::always_inline]]
inline void combine_constants_in_nary(Operator op, std::vector<ADNodePtr>& ins,
                                      const ADGraphPtr& g) {
    if (ins.empty()) return;

    // Remove nulls early (defensive)
    ins.erase(std::remove(ins.begin(), ins.end(), nullptr), ins.end());
    if (ins.empty()) return;

    double acc = (op == Operator::Add) ? 0.0 : 1.0;

    // Accumulate & remove constants; short-circuit for *0
    for (int i = (int)ins.size() - 1; i >= 0; --i) {
        double v;
        if (is_const(ins[i], &v)) {
            if (op == Operator::Multiply && v == 0.0) {
                ins.clear();
                ins.push_back(make_cte(g, 0.0));
                return;
            }
            acc = (op == Operator::Add) ? (acc + v) : (acc * v);
            ins.erase(ins.begin() + i);
        }
    }

    // Reinsert consolidated constant if not identity
    const bool is_identity = (op == Operator::Add ? (acc == 0.0) : (acc == 1.0));
    if (!is_identity) ins.push_back(make_cte(g, acc));
}

// Lightweight structural canonicalization (keeps constant at the back)
[[gnu::always_inline]]
inline void light_canon_for_cse(std::vector<ADNodePtr>& ins) {
    // Drop nulls (should already be gone)
    ins.erase(std::remove(ins.begin(), ins.end(), nullptr), ins.end());
    if (ins.size() <= 1) return;

    // Ensure a single constant (if any) sits at the back.
    int k = -1; double _;
    for (int i = 0; i < (int)ins.size(); ++i) {
        if (is_const(ins[i], &_)) { k = i; break; }
    }
    if (k >= 0 && k != (int)ins.size() - 1)
        std::swap(ins[k], ins.back());
}

// ============================================================================
// Expression ⊕ Expression
// ============================================================================

[[gnu::hot]]
inline ExpressionPtr operator+(const ExpressionPtr &lhs, const ExpressionPtr &rhs) {
    auto g = pick_graph(lhs ? lhs->graph : nullptr, rhs ? rhs->graph : nullptr);
    auto a = attach_input(lhs, g);
    auto b = attach_input(rhs, g);

    // Fast constant folding
    double av, bv;
    if (is_const(a, &av) && is_const(b, &bv)) [[unlikely]]
        return std::make_shared<Expression>(make_cte(g, av + bv), g);

    // Identity
    if (is_const(a, &av) && av == 0.0) [[unlikely]] return std::make_shared<Expression>(b, g);
    if (is_const(b, &bv) && bv == 0.0) [[unlikely]] return std::make_shared<Expression>(a, g);

    // n-ary Add + constant sinking
    const bool a_is_add = (a && a->type == Operator::Add);
    const bool b_is_add = (b && b->type == Operator::Add);
    const size_t est_sz  = (a_is_add ? a->inputs.size() : 1) +
                           (b_is_add ? b->inputs.size() : 1);
    std::vector<ADNodePtr> ins;
    ins.reserve(est_sz + 2);

    flatten_into(Operator::Add, a, ins);
    flatten_into(Operator::Add, b, ins);

    combine_constants_in_nary(Operator::Add, ins, g);
    light_canon_for_cse(ins);

    if (ins.empty())  return std::make_shared<Expression>(make_cte(g, 0.0), g);
    if (ins.size()==1) return std::make_shared<Expression>(ins[0], g);
    return std::make_shared<Expression>(build_nary_node(g, Operator::Add, std::move(ins)), g);
}

[[gnu::hot]]
inline ExpressionPtr operator-(const ExpressionPtr &lhs, const ExpressionPtr &rhs) {
    auto g = pick_graph(lhs ? lhs->graph : nullptr, rhs ? rhs->graph : nullptr);
    auto a = attach_input(lhs, g);
    auto b = attach_input(rhs, g);

    // Constant folding
    double av, bv;
    if (is_const(a, &av) && is_const(b, &bv)) [[unlikely]]
        return std::make_shared<Expression>(make_cte(g, av - bv), g);

    // Identities
    if (is_const(b, &bv) && bv == 0.0) [[unlikely]]
        return std::make_shared<Expression>(a, g);

    if (is_const(a, &av) && av == 0.0) [[unlikely]] {
        // 0 - b -> (-1) * b (flatten-friendly)
        std::vector<ADNodePtr> ins;
        ins.reserve((b && b->type == Operator::Multiply ? b->inputs.size() : 1) + 1);
        flatten_into(Operator::Multiply, b, ins);
        ins.push_back(make_cte(g, -1.0));
        combine_constants_in_nary(Operator::Multiply, ins, g);
        light_canon_for_cse(ins);
        if (ins.size()==1) return std::make_shared<Expression>(ins[0], g);
        return std::make_shared<Expression>(build_nary_node(g, Operator::Multiply, std::move(ins)), g);
    }

    // Keep binary Subtract (don’t normalize in-graph)
    auto out = std::make_shared<ADNode>();
    out->type = Operator::Subtract;
    out->addInput(a);
    out->addInput(b);
    if (g) g->addNode(out);
    return std::make_shared<Expression>(out, g);
}

[[gnu::hot]]
inline ExpressionPtr operator*(const ExpressionPtr &lhs, const ExpressionPtr &rhs) {
    auto g = pick_graph(lhs ? lhs->graph : nullptr, rhs ? rhs->graph : nullptr);
    auto a = attach_input(lhs, g);
    auto b = attach_input(rhs, g);

    // Constant folding
    double av, bv;
    if (is_const(a, &av) && is_const(b, &bv)) [[unlikely]]
        return std::make_shared<Expression>(make_cte(g, av * bv), g);

    // Zero / identity
    if ((is_const(a, &av) && av == 0.0) || (is_const(b, &bv) && bv == 0.0)) [[unlikely]]
        return std::make_shared<Expression>(make_cte(g, 0.0), g);
    if (is_const(a, &av) && av == 1.0) [[unlikely]] return std::make_shared<Expression>(b, g);
    if (is_const(b, &bv) && bv == 1.0) [[unlikely]] return std::make_shared<Expression>(a, g);

    // n-ary Multiply + constant sinking
    const bool a_is_mul = (a && a->type == Operator::Multiply);
    const bool b_is_mul = (b && b->type == Operator::Multiply);
    const size_t est_sz  = (a_is_mul ? a->inputs.size() : 1) +
                           (b_is_mul ? b->inputs.size() : 1);
    std::vector<ADNodePtr> ins;
    ins.reserve(est_sz + 2);

    flatten_into(Operator::Multiply, a, ins);
    flatten_into(Operator::Multiply, b, ins);

    combine_constants_in_nary(Operator::Multiply, ins, g);
    light_canon_for_cse(ins);

    if (ins.empty())   return std::make_shared<Expression>(make_cte(g, 1.0), g);
    if (ins.size()==1) return std::make_shared<Expression>(ins[0], g);
    return std::make_shared<Expression>(build_nary_node(g, Operator::Multiply, std::move(ins)), g);
}

[[gnu::hot]]
inline ExpressionPtr operator/(const ExpressionPtr &lhs, const ExpressionPtr &rhs) {
    auto g = pick_graph(lhs ? lhs->graph : nullptr, rhs ? rhs->graph : nullptr);
    auto a = attach_input(lhs, g);
    auto b = attach_input(rhs, g);

    // If denominator is constant, multiply by reciprocal (flatten-friendly)
    double bv;
    if (is_const(b, &bv)) [[likely]] {
        if (bv == 1.0) [[unlikely]] return std::make_shared<Expression>(a, g);
        if (bv == 0.0) [[unlikely]] {
            // Keep an explicit Divide node instead of folding to 0.0
            auto out = std::make_shared<ADNode>();
            out->type = Operator::Divide;
            out->addInput(a);
            out->addInput(b);
            if (g) g->addNode(out);
            return std::make_shared<Expression>(out, g);
        }

        // a / c -> a * (1/c)
        const double inv = 1.0 / bv;
        std::vector<ADNodePtr> ins;
        ins.reserve((a && a->type == Operator::Multiply ? a->inputs.size() : 1) + 1);
        flatten_into(Operator::Multiply, a, ins);
        ins.push_back(make_cte(g, inv));
        combine_constants_in_nary(Operator::Multiply, ins, g);
        light_canon_for_cse(ins);
        if (ins.size()==1) return std::make_shared<Expression>(ins[0], g);
        return std::make_shared<Expression>(build_nary_node(g, Operator::Multiply, std::move(ins)), g);
    }

    // Non-constant denominator: keep Divide node
    auto out = std::make_shared<ADNode>();
    out->type = Operator::Divide;
    out->addInput(a);
    out->addInput(b);
    if (g) g->addNode(out);
    return std::make_shared<Expression>(out, g);
}

// ============================================================================
// Expression ⊕ double
// ============================================================================

[[gnu::hot]]
inline ExpressionPtr operator+(const ExpressionPtr &expr, double scalar) {
    if (scalar == 0.0) [[unlikely]] return expr;
    auto g = pick_graph(expr ? expr->graph : nullptr);
    auto a = attach_input(expr, g);

    std::vector<ADNodePtr> ins;
    ins.reserve((a && a->type == Operator::Add ? a->inputs.size() : 1) + 1);
    flatten_into(Operator::Add, a, ins);
    ins.push_back(make_cte(g, scalar));

    combine_constants_in_nary(Operator::Add, ins, g);
    light_canon_for_cse(ins);
    if (ins.size()==1) return std::make_shared<Expression>(ins[0], g);
    return std::make_shared<Expression>(build_nary_node(g, Operator::Add, std::move(ins)), g);
}

[[gnu::hot]]
inline ExpressionPtr operator-(const ExpressionPtr &expr, double scalar) {
    if (scalar == 0.0) [[unlikely]] return expr;
    auto g = pick_graph(expr ? expr->graph : nullptr);
    auto a = attach_input(expr, g);
    auto c = make_cte(g, scalar);

    auto out = std::make_shared<ADNode>();
    out->type = Operator::Subtract;
    out->addInput(a);
    out->addInput(c);
    if (g) g->addNode(out);
    return std::make_shared<Expression>(out, g);
}

[[gnu::hot]]
inline ExpressionPtr operator*(const ExpressionPtr &expr, double scalar) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    if (scalar == 0.0) [[unlikely]] return std::make_shared<Expression>(make_cte(g, 0.0), g);
    if (scalar == 1.0) [[unlikely]] return expr;

    auto a = attach_input(expr, g);

    std::vector<ADNodePtr> ins;
    ins.reserve((a && a->type == Operator::Multiply ? a->inputs.size() : 1) + 1);
    flatten_into(Operator::Multiply, a, ins);
    ins.push_back(make_cte(g, scalar));

    combine_constants_in_nary(Operator::Multiply, ins, g);
    light_canon_for_cse(ins);
    if (ins.size()==1) return std::make_shared<Expression>(ins[0], g);
    return std::make_shared<Expression>(build_nary_node(g, Operator::Multiply, std::move(ins)), g);
}

[[gnu::hot]]
inline ExpressionPtr operator/(const ExpressionPtr &expr, double scalar) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    auto a = attach_input(expr, g);

    if (scalar == 1.0) [[unlikely]] return expr;

    if (scalar == 0.0) [[unlikely]] {
        // Keep explicit Divide node; do not fold to 0.0
        auto c = make_cte(g, 0.0);
        auto out = std::make_shared<ADNode>();
        out->type = Operator::Divide;
        out->addInput(a);
        out->addInput(c);
        if (g) g->addNode(out);
        return std::make_shared<Expression>(out, g);
    }

    // Multiply by reciprocal
    const double inv = 1.0 / scalar;
    std::vector<ADNodePtr> ins;
    ins.reserve((a && a->type == Operator::Multiply ? a->inputs.size() : 1) + 1);
    flatten_into(Operator::Multiply, a, ins);
    ins.push_back(make_cte(g, inv));

    combine_constants_in_nary(Operator::Multiply, ins, g);
    light_canon_for_cse(ins);
    if (ins.size()==1) return std::make_shared<Expression>(ins[0], g);
    return std::make_shared<Expression>(build_nary_node(g, Operator::Multiply, std::move(ins)), g);
}

// ============================================================================
// double ⊕ Expression (use commutativity where valid)
// ============================================================================

[[gnu::always_inline]]
inline ExpressionPtr operator+(double scalar, const ExpressionPtr &expr) {
    return expr + scalar;
}

[[gnu::hot]]
inline ExpressionPtr operator-(double scalar, const ExpressionPtr &expr) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    auto b = attach_input(expr, g);

    if (scalar == 0.0) [[unlikely]] {
        // 0 - b => (-1)*b, flatten-friendly
        std::vector<ADNodePtr> ins;
        ins.reserve((b && b->type == Operator::Multiply ? b->inputs.size() : 1) + 1);
        flatten_into(Operator::Multiply, b, ins);
        ins.push_back(make_cte(g, -1.0));
        combine_constants_in_nary(Operator::Multiply, ins, g);
        light_canon_for_cse(ins);
        if (ins.size()==1) return std::make_shared<Expression>(ins[0], g);
        return std::make_shared<Expression>(build_nary_node(g, Operator::Multiply, std::move(ins)), g);
    }

    auto c = make_cte(g, scalar);
    auto out = std::make_shared<ADNode>();
    out->type = Operator::Subtract;
    out->addInput(c);
    out->addInput(b);
    if (g) g->addNode(out);
    return std::make_shared<Expression>(out, g);
}

[[gnu::always_inline]]
inline ExpressionPtr operator*(double scalar, const ExpressionPtr &expr) {
    return expr * scalar;
}

[[gnu::hot]]
inline ExpressionPtr operator/(double scalar, const ExpressionPtr &expr) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    auto b = attach_input(expr, g);

    if (scalar == 0.0) [[unlikely]]
        return std::make_shared<Expression>(make_cte(g, 0.0), g);

    // Non-constant denominator here; keep Divide
    auto c = make_cte(g, scalar);
    auto out = std::make_shared<ADNode>();
    out->type = Operator::Divide;
    out->addInput(c);
    out->addInput(b);
    if (g) g->addNode(out);
    return std::make_shared<Expression>(out, g);
}

// ============================================================================
// ADNodePtr ⊕ double (wrap node in an Expression with a fresh/adopted graph)
// NOTE: If you can access node->graph, use that instead of creating a new one.
// ============================================================================

[[gnu::always_inline]]
inline ExpressionPtr wrap_with_graph(const ADNodePtr& node) {
    auto g = std::make_shared<ADGraph>();
    if (node) g->adoptSubgraph(node);
    return std::make_shared<Expression>(node, g);
}

[[gnu::always_inline]] inline ExpressionPtr operator+(const ADNodePtr &node, double s) { return wrap_with_graph(node) + s; }
[[gnu::always_inline]] inline ExpressionPtr operator-(const ADNodePtr &node, double s) { return wrap_with_graph(node) - s; }
[[gnu::always_inline]] inline ExpressionPtr operator*(const ADNodePtr &node, double s) { return wrap_with_graph(node) * s; }
[[gnu::always_inline]] inline ExpressionPtr operator/(const ADNodePtr &node, double s) { return wrap_with_graph(node) / s; }

// ============================================================================
// VariablePtr ⊕ ExpressionPtr
// ============================================================================

[[gnu::hot]]
inline ExpressionPtr operator+(const VariablePtr &var, const ExpressionPtr &expr) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    auto v = std::make_shared<Expression>(var, 1.0, g);
    auto a = attach_input(v, g);
    auto b = attach_input(expr, g);

    const bool a_is_add = (a && a->type == Operator::Add);
    const bool b_is_add = (b && b->type == Operator::Add);
    const size_t est_sz  = (a_is_add ? a->inputs.size() : 1) +
                           (b_is_add ? b->inputs.size() : 1);
    std::vector<ADNodePtr> ins;
    ins.reserve(est_sz + 2);

    flatten_into(Operator::Add, a, ins);
    flatten_into(Operator::Add, b, ins);

    combine_constants_in_nary(Operator::Add, ins, g);
    light_canon_for_cse(ins);

    if (ins.empty())   return std::make_shared<Expression>(make_cte(g, 0.0), g);
    if (ins.size()==1) return std::make_shared<Expression>(ins[0], g);
    return std::make_shared<Expression>(build_nary_node(g, Operator::Add, std::move(ins)), g);
}

[[gnu::hot]]
inline ExpressionPtr operator-(const VariablePtr &var, const ExpressionPtr &expr) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    auto v = std::make_shared<Expression>(var, 1.0, g);
    auto a = attach_input(v, g);
    auto b = attach_input(expr, g);

    auto out = std::make_shared<ADNode>();
    out->type = Operator::Subtract;
    out->addInput(a);
    out->addInput(b);
    if (g) g->addNode(out);
    return std::make_shared<Expression>(out, g);
}

[[gnu::hot]]
inline ExpressionPtr operator*(const VariablePtr &var, const ExpressionPtr &expr) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    auto v = std::make_shared<Expression>(var, 1.0, g);
    auto a = attach_input(v, g);
    auto b = attach_input(expr, g);

    const bool a_is_mul = (a && a->type == Operator::Multiply);
    const bool b_is_mul = (b && b->type == Operator::Multiply);
    const size_t est_sz  = (a_is_mul ? a->inputs.size() : 1) +
                           (b_is_mul ? b->inputs.size() : 1);
    std::vector<ADNodePtr> ins;
    ins.reserve(est_sz + 2);

    flatten_into(Operator::Multiply, a, ins);
    flatten_into(Operator::Multiply, b, ins);

    combine_constants_in_nary(Operator::Multiply, ins, g);
    light_canon_for_cse(ins);

    if (ins.empty())   return std::make_shared<Expression>(make_cte(g, 1.0), g);
    if (ins.size()==1) return std::make_shared<Expression>(ins[0], g);
    return std::make_shared<Expression>(build_nary_node(g, Operator::Multiply, std::move(ins)), g);
}

[[gnu::hot]]
inline ExpressionPtr operator/(const VariablePtr &var, const ExpressionPtr &expr) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    auto v = std::make_shared<Expression>(var, 1.0, g);
    auto a = attach_input(v, g);
    auto b = attach_input(expr, g);

    auto out = std::make_shared<ADNode>();
    out->type = Operator::Divide;
    out->addInput(a);
    out->addInput(b);
    if (g) g->addNode(out);
    return std::make_shared<Expression>(out, g);
}

// ============================================================================
// Unary minus (as multiply by -1 with n-ary flattening)
// ============================================================================

[[gnu::hot]]
inline ExpressionPtr operator-(const ExpressionPtr &expr) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    auto a = attach_input(expr, g);

    std::vector<ADNodePtr> ins;
    ins.reserve((a && a->type == Operator::Multiply ? a->inputs.size() : 1) + 1);
    flatten_into(Operator::Multiply, a, ins);
    ins.push_back(make_cte(g, -1.0));

    combine_constants_in_nary(Operator::Multiply, ins, g);
    light_canon_for_cse(ins);

    if (ins.size()==1) return std::make_shared<Expression>(ins[0], g);
    return std::make_shared<Expression>(build_nary_node(g, Operator::Multiply, std::move(ins)), g);
}

// ============================================================================
// maximum (with cheap folds)
// ============================================================================

[[gnu::hot]]
inline ExpressionPtr maximum(const ExpressionPtr &lhs, const ExpressionPtr &rhs) {
    auto g = pick_graph(lhs ? lhs->graph : nullptr, rhs ? rhs->graph : nullptr);
    auto a = attach_input(lhs, g);
    auto b = attach_input(rhs, g);

    if (a.get() == b.get()) [[unlikely]] return std::make_shared<Expression>(a, g);

    double av, bv;
    if (is_const(a, &av) && is_const(b, &bv)) [[unlikely]]
        return std::make_shared<Expression>(make_cte(g, std::max(av, bv)), g);

    auto out = std::make_shared<ADNode>();
    out->type = Operator::Max;
    out->addInput(a);
    out->addInput(b);
    if (g) g->addNode(out);
    return std::make_shared<Expression>(out, g);
}
