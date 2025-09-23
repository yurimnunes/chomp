#pragma once
#include "ADGraph.h"
#include "Expression.h"
#include "Variable.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>

// ============================================================================
// Optimized Helper Functions
// ============================================================================

// Fast constant node creation with inline hint
[[gnu::always_inline, gnu::hot]]
inline ADNodePtr make_const_node(const ADGraphPtr &g, double v) {
    auto c = std::make_shared<ADNode>();
    c->type = Operator::cte;
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
        g->adoptSubgraph(e->rootNode);
        return e->rootNode;
    }
    
    if (e->node) [[likely]] {
        g->adoptSubgraph(e->node);
        return e->node;
    }
    
    return nullptr;
}

// Fast constant detection with branch prediction hints
[[gnu::always_inline, gnu::hot]]
inline bool is_const(const ADNodePtr& n, double* out = nullptr) {
    if (n && n->type == Operator::cte) [[likely]] {
        if (out) [[likely]] *out = n->value;
        return true;
    }
    return false;
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
inline ADNodePtr build_nary_node(const ADGraphPtr& g, Operator op, std::vector<ADNodePtr>&& ins) {
    auto out = std::make_shared<ADNode>();
    out->type = op;
    out->inputs = std::move(ins);
    g->addNode(out);
    return out;
}

// ============================================================================
// Template-Based Optimized Binary Operations
// ============================================================================

// Template for common constant folding patterns
template<typename ConstFolder, typename NodeBuilder>
[[gnu::hot]]
inline ExpressionPtr create_binary_op(const ADNodePtr& a, const ADNodePtr& b, 
                                     const ADGraphPtr& g, ConstFolder folder, 
                                     NodeBuilder builder) {
    double av, bv;
    if (is_const(a, &av) && is_const(b, &bv)) [[unlikely]] {
        return std::make_shared<Expression>(make_const_node(g, folder(av, bv)), g);
    }
    return builder(a, b, g);
}

// ============================================================================
// ExpressionPtr ⊕ ExpressionPtr (Optimized)
// ============================================================================

[[gnu::hot]]
inline ExpressionPtr operator+(const ExpressionPtr &lhs, const ExpressionPtr &rhs) {
    auto g = pick_graph(lhs ? lhs->graph : nullptr, rhs ? rhs->graph : nullptr);
    auto a = attach_input(lhs, g);
    auto b = attach_input(rhs, g);

    // Fast constant folding
    double av, bv;
    if (is_const(a, &av) && is_const(b, &bv)) [[unlikely]]
        return std::make_shared<Expression>(make_const_node(g, av + bv), g);
    
    // Identity optimizations
    if (is_const(a, &av) && av == 0.0) [[unlikely]] return std::make_shared<Expression>(b, g);
    if (is_const(b, &bv) && bv == 0.0) [[unlikely]] return std::make_shared<Expression>(a, g);

    // Optimized n-ary Add with better capacity estimation
    const bool a_is_add = (a && a->type == Operator::Add);
    const bool b_is_add = (b && b->type == Operator::Add);
    const size_t est_size = (a_is_add ? a->inputs.size() : 1) + 
                           (b_is_add ? b->inputs.size() : 1);
    
    std::vector<ADNodePtr> ins;
    ins.reserve(est_size);
    flatten_into(Operator::Add, a, ins);
    flatten_into(Operator::Add, b, ins);
    
    return std::make_shared<Expression>(build_nary_node(g, Operator::Add, std::move(ins)), g);
}

[[gnu::hot]]
inline ExpressionPtr operator-(const ExpressionPtr &lhs, const ExpressionPtr &rhs) {
    auto g = pick_graph(lhs ? lhs->graph : nullptr, rhs ? rhs->graph : nullptr);
    auto a = attach_input(lhs, g);
    auto b = attach_input(rhs, g);
    
    // Fast constant folding
    double av, bv;
    if (is_const(a, &av) && is_const(b, &bv)) [[unlikely]]
        return std::make_shared<Expression>(make_const_node(g, av - bv), g);
    
    // Identity optimization
    if (is_const(b, &bv) && bv == 0.0) [[unlikely]]
        return std::make_shared<Expression>(a, g);

    // Keep binary subtract (simpler than n-ary conversion)
    auto out = std::make_shared<ADNode>();
    out->type = Operator::Subtract;
    out->addInput(a);
    out->addInput(b);
    g->addNode(out);
    return std::make_shared<Expression>(out, g);
}

[[gnu::hot]]
inline ExpressionPtr operator*(const ExpressionPtr &lhs, const ExpressionPtr &rhs) {
    auto g = pick_graph(lhs ? lhs->graph : nullptr, rhs ? rhs->graph : nullptr);
    auto a = attach_input(lhs, g);
    auto b = attach_input(rhs, g);

    // Fast constant folding with early returns
    double av, bv;
    if (is_const(a, &av) && is_const(b, &bv)) [[unlikely]]
        return std::make_shared<Expression>(make_const_node(g, av * bv), g);
    
    // Zero optimization
    if ((is_const(a, &av) && av == 0.0) || (is_const(b, &bv) && bv == 0.0)) [[unlikely]]
        return std::make_shared<Expression>(make_const_node(g, 0.0), g);
    
    // Identity optimizations
    if (is_const(a, &av) && av == 1.0) [[unlikely]] return std::make_shared<Expression>(b, g);
    if (is_const(b, &bv) && bv == 1.0) [[unlikely]] return std::make_shared<Expression>(a, g);

    // Optimized n-ary Multiply
    const bool a_is_mult = (a && a->type == Operator::Multiply);
    const bool b_is_mult = (b && b->type == Operator::Multiply);
    const size_t est_size = (a_is_mult ? a->inputs.size() : 1) + 
                           (b_is_mult ? b->inputs.size() : 1);
    
    std::vector<ADNodePtr> ins;
    ins.reserve(est_size);
    flatten_into(Operator::Multiply, a, ins);
    flatten_into(Operator::Multiply, b, ins);
    
    return std::make_shared<Expression>(build_nary_node(g, Operator::Multiply, std::move(ins)), g);
}

[[gnu::hot]]
inline ExpressionPtr operator/(const ExpressionPtr &lhs, const ExpressionPtr &rhs) {
    auto g = pick_graph(lhs ? lhs->graph : nullptr, rhs ? rhs->graph : nullptr);
    auto a = attach_input(lhs, g);
    auto b = attach_input(rhs, g);
    
    // Fast constant folding with division by zero handling
    double av, bv;
    if (is_const(a, &av) && is_const(b, &bv)) [[unlikely]]
        return std::make_shared<Expression>(make_const_node(g, (bv != 0.0) ? (av / bv) : 0.0), g);
    
    // Identity and zero optimizations
    if (is_const(b, &bv) && bv == 1.0) [[unlikely]]
        return std::make_shared<Expression>(a, g);
    if (is_const(a, &av) && av == 0.0) [[unlikely]]
        return std::make_shared<Expression>(make_const_node(g, 0.0), g);

    auto out = std::make_shared<ADNode>();
    out->type = Operator::Divide;
    out->addInput(a);
    out->addInput(b);
    g->addNode(out);
    return std::make_shared<Expression>(out, g);
}

// ============================================================================
// ExpressionPtr ⊕ double (Optimized with Early Returns)
// ============================================================================

[[gnu::hot]]
inline ExpressionPtr operator+(const ExpressionPtr &expr, double scalar) {
    if (scalar == 0.0) [[unlikely]] return expr; // Early return for identity
    
    auto g = pick_graph(expr ? expr->graph : nullptr);
    auto a = attach_input(expr, g);
    auto c = make_const_node(g, scalar);

    // Optimized n-ary Add flattening
    const size_t est_size = (a && a->type == Operator::Add ? a->inputs.size() : 1) + 1;
    std::vector<ADNodePtr> ins;
    ins.reserve(est_size);
    flatten_into(Operator::Add, a, ins);
    ins.push_back(c);
    
    return std::make_shared<Expression>(build_nary_node(g, Operator::Add, std::move(ins)), g);
}

[[gnu::hot]]
inline ExpressionPtr operator-(const ExpressionPtr &expr, double scalar) {
    if (scalar == 0.0) [[unlikely]] return expr; // Early return for identity
    
    auto g = pick_graph(expr ? expr->graph : nullptr);
    auto a = attach_input(expr, g);
    auto c = make_const_node(g, scalar);
    
    auto out = std::make_shared<ADNode>();
    out->type = Operator::Subtract;
    out->addInput(a);
    out->addInput(c);
    g->addNode(out);
    return std::make_shared<Expression>(out, g);
}

[[gnu::hot]]
inline ExpressionPtr operator*(const ExpressionPtr &expr, double scalar) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    
    // Fast special case handling
    if (scalar == 0.0) [[unlikely]] return std::make_shared<Expression>(make_const_node(g, 0.0), g);
    if (scalar == 1.0) [[unlikely]] return expr;
    
    auto a = attach_input(expr, g);
    auto c = make_const_node(g, scalar);
    
    // Optimized n-ary Multiply flattening
    const size_t est_size = (a && a->type == Operator::Multiply ? a->inputs.size() : 1) + 1;
    std::vector<ADNodePtr> ins;
    ins.reserve(est_size);
    flatten_into(Operator::Multiply, a, ins);
    ins.push_back(c);
    
    return std::make_shared<Expression>(build_nary_node(g, Operator::Multiply, std::move(ins)), g);
}

[[gnu::hot]]
inline ExpressionPtr operator/(const ExpressionPtr &expr, double scalar) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    
    // Handle special cases
    if (scalar == 0.0) [[unlikely]] return std::make_shared<Expression>(make_const_node(g, 0.0), g);
    if (scalar == 1.0) [[unlikely]] return expr;
    
    auto a = attach_input(expr, g);
    auto c = make_const_node(g, scalar);
    
    auto out = std::make_shared<ADNode>();
    out->type = Operator::Divide;
    out->addInput(a);
    out->addInput(c);
    g->addNode(out);
    return std::make_shared<Expression>(out, g);
}

// ============================================================================
// double ⊕ ExpressionPtr (Reverse scalar ops, leveraging commutativity)
// ============================================================================

[[gnu::always_inline]]
inline ExpressionPtr operator+(double scalar, const ExpressionPtr &expr) {
    return expr + scalar; // Leverage commutativity
}

inline ExpressionPtr operator-(double scalar, const ExpressionPtr &expr) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    
    if (scalar == 0.0) [[unlikely]] {
        // Optimized 0 - expr = -1 * expr
        auto a = attach_input(expr, g);
        auto m1 = make_const_node(g, -1.0);
        
        // Use n-ary multiply for consistency
        const size_t est_size = (a && a->type == Operator::Multiply ? a->inputs.size() : 1) + 1;
        std::vector<ADNodePtr> ins;
        ins.reserve(est_size);
        flatten_into(Operator::Multiply, a, ins);
        ins.push_back(m1);
        
        return std::make_shared<Expression>(build_nary_node(g, Operator::Multiply, std::move(ins)), g);
    }
    
    auto b = attach_input(expr, g);
    auto c = make_const_node(g, scalar);
    
    auto out = std::make_shared<ADNode>();
    out->type = Operator::Subtract;
    out->addInput(c);
    out->addInput(b);
    g->addNode(out);
    return std::make_shared<Expression>(out, g);
}

[[gnu::always_inline]]
inline ExpressionPtr operator*(double scalar, const ExpressionPtr &expr) {
    return expr * scalar; // Leverage commutativity
}

inline ExpressionPtr operator/(double scalar, const ExpressionPtr &expr) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    auto b = attach_input(expr, g);
    
    if (scalar == 0.0) [[unlikely]]
        return std::make_shared<Expression>(make_const_node(g, 0.0), g);
    
    auto c = make_const_node(g, scalar);
    auto out = std::make_shared<ADNode>();
    out->type = Operator::Divide;
    out->addInput(c);
    out->addInput(b);
    g->addNode(out);
    return std::make_shared<Expression>(out, g);
}

// ============================================================================
// ADNodePtr operations (simplified with function reuse)
// ============================================================================

[[gnu::always_inline]]
inline ExpressionPtr operator+(const ADNodePtr &node, double scalar) {
    auto g = std::make_shared<ADGraph>();
    g->adoptSubgraph(node);
    return std::make_shared<Expression>(node, g) + scalar;
}

[[gnu::always_inline]]
inline ExpressionPtr operator-(const ADNodePtr &node, double scalar) {
    auto g = std::make_shared<ADGraph>();
    g->adoptSubgraph(node);
    return std::make_shared<Expression>(node, g) - scalar;
}

[[gnu::always_inline]]
inline ExpressionPtr operator*(const ADNodePtr &node, double scalar) {
    auto g = std::make_shared<ADGraph>();
    g->adoptSubgraph(node);
    return std::make_shared<Expression>(node, g) * scalar;
}

[[gnu::always_inline]]
inline ExpressionPtr operator/(const ADNodePtr &node, double scalar) {
    auto g = std::make_shared<ADGraph>();
    g->adoptSubgraph(node);
    return std::make_shared<Expression>(node, g) / scalar;
}

// ============================================================================
// VariablePtr operations (optimized with capacity hints)
// ============================================================================

[[gnu::hot]]
inline ExpressionPtr operator+(const VariablePtr &var, const ExpressionPtr &expr) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    auto v = std::make_shared<Expression>(var, 1.0, g);
    auto a = attach_input(v, g);
    auto b = attach_input(expr, g);

    // Optimized n-ary Add with better size estimation
    const bool a_is_add = (a && a->type == Operator::Add);
    const bool b_is_add = (b && b->type == Operator::Add);
    const size_t est_size = (a_is_add ? a->inputs.size() : 1) + 
                           (b_is_add ? b->inputs.size() : 1);
    
    std::vector<ADNodePtr> ins;
    ins.reserve(est_size);
    flatten_into(Operator::Add, a, ins);
    flatten_into(Operator::Add, b, ins);
    
    return std::make_shared<Expression>(build_nary_node(g, Operator::Add, std::move(ins)), g);
}

inline ExpressionPtr operator-(const VariablePtr &var, const ExpressionPtr &expr) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    auto v = std::make_shared<Expression>(var, 1.0, g);
    auto a = attach_input(v, g);
    auto b = attach_input(expr, g);
    
    auto out = std::make_shared<ADNode>();
    out->type = Operator::Subtract;
    out->addInput(a);
    out->addInput(b);
    g->addNode(out);
    return std::make_shared<Expression>(out, g);
}

[[gnu::hot]]
inline ExpressionPtr operator*(const VariablePtr &var, const ExpressionPtr &expr) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    auto v = std::make_shared<Expression>(var, 1.0, g);
    auto a = attach_input(v, g);
    auto b = attach_input(expr, g);

    // Optimized n-ary Multiply with better size estimation
    const bool a_is_mult = (a && a->type == Operator::Multiply);
    const bool b_is_mult = (b && b->type == Operator::Multiply);
    const size_t est_size = (a_is_mult ? a->inputs.size() : 1) + 
                           (b_is_mult ? b->inputs.size() : 1);
    
    std::vector<ADNodePtr> ins;
    ins.reserve(est_size);
    flatten_into(Operator::Multiply, a, ins);
    flatten_into(Operator::Multiply, b, ins);
    
    return std::make_shared<Expression>(build_nary_node(g, Operator::Multiply, std::move(ins)), g);
}

inline ExpressionPtr operator/(const VariablePtr &var, const ExpressionPtr &expr) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    auto v = std::make_shared<Expression>(var, 1.0, g);
    auto a = attach_input(v, g);
    auto b = attach_input(expr, g);
    
    auto out = std::make_shared<ADNode>();
    out->type = Operator::Divide;
    out->addInput(a);
    out->addInput(b);
    g->addNode(out);
    return std::make_shared<Expression>(out, g);
}

// ============================================================================
// Unary minus (optimized with n-ary multiply consistency)
// ============================================================================

inline ExpressionPtr operator-(const ExpressionPtr &expr) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    auto a = attach_input(expr, g);

    // Optimized multiply by -1 using n-ary flattening
    auto m1 = make_const_node(g, -1.0);
    const size_t est_size = (a && a->type == Operator::Multiply ? a->inputs.size() : 1) + 1;
    
    std::vector<ADNodePtr> ins;
    ins.reserve(est_size);
    flatten_into(Operator::Multiply, a, ins);
    ins.push_back(m1);
    
    return std::make_shared<Expression>(build_nary_node(g, Operator::Multiply, std::move(ins)), g);
}

// ============================================================================
// maximum function (unchanged but with optimization hints)
// ============================================================================

[[gnu::hot]]
inline ExpressionPtr maximum(const ExpressionPtr &lhs, const ExpressionPtr &rhs) {
    auto g = pick_graph(lhs ? lhs->graph : nullptr, rhs ? rhs->graph : nullptr);
    auto a = attach_input(lhs, g);
    auto b = attach_input(rhs, g);
    
    auto out = std::make_shared<ADNode>();
    out->type = Operator::Max;
    out->addInput(a);
    out->addInput(b);
    g->addNode(out);
    return std::make_shared<Expression>(out, g);
}