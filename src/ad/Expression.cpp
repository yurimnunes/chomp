// Expression.cpp - Optimized version with full API compatibility
#include "../../include/ad/Expression.h"
#include "../../include/ad/ADGraph.h"
#include "../../include/ad/Operators.h"  // make_const_node(g, double)
#include "../../include/ad/Variable.h"

#include <memory>
#include <utility>

// ============================================================================
// Optimized Helper Functions
// ============================================================================

namespace {
    // Fast graph selection with move semantics
    [[gnu::always_inline, gnu::hot]]
    inline ADGraphPtr pick_graph_fast(const ADGraphPtr& g1, const ADGraphPtr& g2) noexcept {
        return g1 ? g1 : (g2 ? g2 : std::make_shared<ADGraph>());
    }

    // Template for creating binary operations (reduces code duplication)
    template<Operator Op>
    [[gnu::hot]]
    inline ExpressionPtr create_binary_op(const ADNodePtr& left, const ADNodePtr& right, 
                                         const ADGraphPtr& graph) {
        auto result = std::make_shared<Expression>(graph);
        result->node->type = Op;
        result->node->addInput(left);
        result->node->addInput(right);
        graph->addNode(result->node);
        return result;
    }

    // Template for creating unary operations
    template<Operator Op>
    [[gnu::hot]]
    inline ExpressionPtr create_unary_op(const ADNodePtr& input, const ADGraphPtr& graph) {
        auto result = std::make_shared<Expression>(graph);
        result->node->type = Op;
        result->node->addInput(input);
        graph->addNode(result->node);
        return result;
    }

    // Optimized constant node creation with value caching for common values
    [[gnu::always_inline, gnu::hot]]
    inline ADNodePtr make_const_node_fast(const ADGraphPtr& g, double value) {
        // For now, just use the regular make_const_node - caching can be added later
        // when we understand the ADNode structure better
        return make_const_node(g, value);
    }

    // Optimized subgraph adoption - simplified version
    [[gnu::always_inline, gnu::hot]]
    inline void adopt_if_needed(const ADGraphPtr& target_graph, const ADNodePtr& node) {
        if (node) [[likely]] {
            target_graph->adoptSubgraph(node);
        }
    }
}

// ============================================================================
// Expression ⊕ Expression (Optimized)
// ============================================================================

ExpressionPtr Expression::operator+(const Expression& other) const {
    auto g = pick_graph_fast(graph, other.graph);
    adopt_if_needed(g, node);
    adopt_if_needed(g, other.node);
    
    return create_binary_op<Operator::Add>(node, other.node, g);
}

ExpressionPtr Expression::operator-(const Expression& other) const {
    auto g = pick_graph_fast(graph, other.graph);
    adopt_if_needed(g, node);
    adopt_if_needed(g, other.node);
    
    return create_binary_op<Operator::Subtract>(node, other.node, g);
}

ExpressionPtr Expression::operator*(const Expression& other) const {
    auto g = pick_graph_fast(graph, other.graph);
    adopt_if_needed(g, node);
    adopt_if_needed(g, other.node);
    
    return create_binary_op<Operator::Multiply>(node, other.node, g);
}

ExpressionPtr Expression::operator/(const Expression& other) const {
    auto g = pick_graph_fast(graph, other.graph);
    adopt_if_needed(g, node);
    adopt_if_needed(g, other.node);
    
    return create_binary_op<Operator::Divide>(node, other.node, g);
}

// ============================================================================
// Expression ⊕ scalar (Optimized with constant folding)
// ============================================================================

ExpressionPtr Expression::operator+(double scalar) const {
    // Constant folding: if scalar is 0, return copy of this expression
    if (scalar == 0.0) [[unlikely]] {
        auto g = graph ? graph : std::make_shared<ADGraph>();
        adopt_if_needed(g, node);
        auto result = std::make_shared<Expression>(g);
        result->node = node; // Share the node
        return result;
    }
    
    auto g = graph ? graph : std::make_shared<ADGraph>();
    adopt_if_needed(g, node);
    
    return create_binary_op<Operator::Add>(node, make_const_node_fast(g, scalar), g);
}

ExpressionPtr Expression::operator-(double scalar) const {
    // Constant folding: if scalar is 0, return copy of this expression
    if (scalar == 0.0) [[unlikely]] {
        auto g = graph ? graph : std::make_shared<ADGraph>();
        adopt_if_needed(g, node);
        auto result = std::make_shared<Expression>(g);
        result->node = node; // Share the node
        return result;
    }
    
    auto g = graph ? graph : std::make_shared<ADGraph>();
    adopt_if_needed(g, node);
    
    return create_binary_op<Operator::Subtract>(node, make_const_node_fast(g, scalar), g);
}

ExpressionPtr Expression::operator*(double scalar) const {
    auto g = graph ? graph : std::make_shared<ADGraph>();
    adopt_if_needed(g, node);
    
    // Constant folding optimizations
    if (scalar == 1.0) [[unlikely]] {
        auto result = std::make_shared<Expression>(g);
        result->node = node; // Share the node
        return result;
    }
    
    if (scalar == 0.0) [[unlikely]] {
        auto result = std::make_shared<Expression>(g);
        result->node = make_const_node_fast(g, 0.0);
        return result;
    }
    
    if (scalar == -1.0) [[unlikely]] {
        // Use unary minus optimization
        return create_binary_op<Operator::Multiply>(make_const_node_fast(g, -1.0), node, g);
    }
    
    return create_binary_op<Operator::Multiply>(node, make_const_node_fast(g, scalar), g);
}

ExpressionPtr Expression::operator/(double scalar) const {
    if (scalar == 0.0) [[unlikely]] {
        throw std::domain_error("Division by zero in expression");
    }
    
    auto g = graph ? graph : std::make_shared<ADGraph>();
    adopt_if_needed(g, node);
    
    // Constant folding: division by 1
    if (scalar == 1.0) [[unlikely]] {
        auto result = std::make_shared<Expression>(g);
        result->node = node; // Share the node
        return result;
    }
    
    // Optimize division by -1
    if (scalar == -1.0) [[unlikely]] {
        return create_binary_op<Operator::Multiply>(make_const_node_fast(g, -1.0), node, g);
    }
    
    return create_binary_op<Operator::Divide>(node, make_const_node_fast(g, scalar), g);
}

// ============================================================================
// Expression ⊕ VariablePtr (Optimized)
// ============================================================================

ExpressionPtr Expression::operator+(const VariablePtr& var) const {
    auto g = graph ? graph : std::make_shared<ADGraph>();
    adopt_if_needed(g, node);
    
    // Create variable expression once and reuse
    auto vexpr = std::make_shared<Expression>(var, 1.0, g);
    return create_binary_op<Operator::Add>(node, vexpr->node, g);
}

ExpressionPtr Expression::operator-(const VariablePtr& var) const {
    auto g = graph ? graph : std::make_shared<ADGraph>();
    adopt_if_needed(g, node);
    
    auto vexpr = std::make_shared<Expression>(var, 1.0, g);
    return create_binary_op<Operator::Subtract>(node, vexpr->node, g);
}

ExpressionPtr Expression::operator*(const VariablePtr& var) const {
    auto g = graph ? graph : std::make_shared<ADGraph>();
    adopt_if_needed(g, node);
    
    auto vexpr = std::make_shared<Expression>(var, 1.0, g);
    return create_binary_op<Operator::Multiply>(node, vexpr->node, g);
}

ExpressionPtr Expression::operator/(const VariablePtr& var) const {
    auto g = graph ? graph : std::make_shared<ADGraph>();
    adopt_if_needed(g, node);
    
    auto vexpr = std::make_shared<Expression>(var, 1.0, g);
    return create_binary_op<Operator::Divide>(node, vexpr->node, g);
}

// ============================================================================
// Unary minus (Optimized)
// ============================================================================

ExpressionPtr Expression::operator-() const {
    auto g = graph ? graph : std::make_shared<ADGraph>();
    adopt_if_needed(g, node);
    
    return create_binary_op<Operator::Multiply>(make_const_node_fast(g, -1.0), node, g);
}

// ============================================================================
// Reverse scalar ops (Optimized)
// ============================================================================

ExpressionPtr operator+(double lhs, const Expression& rhs) {
    return rhs + lhs; // Leverage commutativity
}

ExpressionPtr operator-(double lhs, const Expression& rhs) {
    auto g = rhs.graph ? rhs.graph : std::make_shared<ADGraph>();
    adopt_if_needed(g, rhs.node);
    
    return create_binary_op<Operator::Subtract>(make_const_node_fast(g, lhs), rhs.node, g);
}

ExpressionPtr operator*(double lhs, const Expression& rhs) {
    return rhs * lhs; // Leverage commutativity
}

ExpressionPtr operator/(double lhs, const Expression& rhs) {
    auto g = rhs.graph ? rhs.graph : std::make_shared<ADGraph>();
    adopt_if_needed(g, rhs.node);
    
    return create_binary_op<Operator::Divide>(make_const_node_fast(g, lhs), rhs.node, g);
}

// ============================================================================
// Convenience functions (Optimized)
// ============================================================================

ExpressionPtr square(const Expression& x) {
    // Optimize x^2 as x * x for better cache locality and reuse
    return x * x;
}

ExpressionPtr reciprocal(const Expression& x) {
    auto g = x.graph ? x.graph : std::make_shared<ADGraph>();
    adopt_if_needed(g, x.node);
    
    return create_binary_op<Operator::Divide>(make_const_node_fast(g, 1.0), x.node, g);
}

ExpressionPtr pow(const Expression& x, double p) {
    // General case: pow(x, p) = exp(p * log(x))
    auto g = x.graph ? x.graph : std::make_shared<ADGraph>();
    adopt_if_needed(g, x.node);

    // log(x)
    auto e_log = std::make_shared<Expression>(g);
    e_log->node->type = Operator::Log;
    e_log->node->addInput(x.node);
    g->addNode(e_log->node);

    // p * log(x)
    auto e_scale = std::make_shared<Expression>(g);
    e_scale->node->type = Operator::Multiply;
    e_scale->node->addInput(e_log->node);
    e_scale->node->addInput(make_const_node_fast(g, p));
    g->addNode(e_scale->node);

    // exp(p * log(x))
    auto e_exp = std::make_shared<Expression>(g);
    e_exp->node->type = Operator::Exp;
    e_exp->node->addInput(e_scale->node);
    g->addNode(e_exp->node);

    return e_exp;
}

// ============================================================================
// Mathematical functions (Optimized with template)
// ============================================================================

template<Operator Op>
[[gnu::hot]]
static ExpressionPtr create_unary_math_op(const Expression& x) {
    auto g = x.graph ? x.graph : std::make_shared<ADGraph>();
    adopt_if_needed(g, x.node);
    return create_unary_op<Op>(x.node, g);
}

ExpressionPtr sin(const Expression& x) {
    return create_unary_math_op<Operator::Sin>(x);
}

ExpressionPtr cos(const Expression& x) {
    return create_unary_math_op<Operator::Cos>(x);
}

ExpressionPtr tan(const Expression& x) {
    return create_unary_math_op<Operator::Tan>(x);
}

ExpressionPtr exp(const Expression& x) {
    return create_unary_math_op<Operator::Exp>(x);
}

ExpressionPtr log(const Expression& x) {
    return create_unary_math_op<Operator::Log>(x);
}

ExpressionPtr tanh(const Expression& x) {
    return create_unary_math_op<Operator::Tanh>(x);
}
ExpressionPtr silu(const Expression& x) {
    return create_unary_math_op<Operator::Silu>(x);
}
ExpressionPtr gelu(const Expression& x) {
    return create_unary_math_op<Operator::Gelu>(x);
}
ExpressionPtr relu(const Expression& x) {
    return create_unary_math_op<Operator::Relu>(x);
}



ExpressionPtr maximum(const Expression& a, const Expression& b) {
    auto g = pick_graph_fast(a.graph, b.graph);
    adopt_if_needed(g, a.node);
    adopt_if_needed(g, b.node);
    
    return create_binary_op<Operator::Max>(a.node, b.node, g);
}

// ============================================================================
// Constructor (Optimized)
// ============================================================================

Expression::Expression(const ADGraphPtr& g)
    : graph(g ? g : std::make_shared<ADGraph>()) {
    initializeNode();
}