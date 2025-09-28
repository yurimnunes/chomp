// Expression.cpp — faster drop-in (API compatible, no Operator::Neg required)
#include "../../include/ad/Expression.h"
#include "../../include/ad/ADGraph.h"
#include "../../include/ad/Operators.h"   // enum Operator
#include "../../include/ad/Variable.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

// ============================================================================
// Helpers
// ============================================================================

namespace {

using ADNodePtr  = std::shared_ptr<ADNode>;
using ADGraphPtr = std::shared_ptr<ADGraph>;
using ExpressionPtr = std::shared_ptr<Expression>;

[[gnu::always_inline]] inline bool is_pos1(double v) noexcept {
    return std::bit_cast<std::uint64_t>(v) == std::bit_cast<std::uint64_t>(1.0);
}
[[gnu::always_inline]] inline bool is_neg1(double v) noexcept {
    return std::bit_cast<std::uint64_t>(v) == std::bit_cast<std::uint64_t>(-1.0);
}
[[gnu::always_inline]] inline bool is_pos0(double v) noexcept {
    // Treat +0 and -0 as zero
    return (std::bit_cast<std::uint64_t>(v) & ~std::uint64_t(1ULL << 63))
           == std::bit_cast<std::uint64_t>(0.0);
}

// Prefer reusing an existing graph; otherwise create a fresh one.
[[gnu::always_inline]] inline ADGraphPtr pick_graph_fast(const ADGraphPtr& g1,
                                                         const ADGraphPtr& g2) {
    if (g1) return g1;
    if (g2) return g2;
    return std::make_shared<ADGraph>();
}

// Adopt subgraph only when the source node belongs to a different graph.
[[gnu::always_inline]] inline void adopt_if_needed(const ADGraphPtr& target,
                                                   const ADGraphPtr& srcg,
                                                   const ADNodePtr& node) {
    if (node && target.get() != srcg.get()) [[likely]] {
        target->adoptSubgraph(node);
    }
}

// Minimal constant-node helper using make_cte from the operators file.
[[gnu::always_inline]] inline ADNodePtr make_const_node_fast(const ADGraphPtr& g, double v) {
    return make_cte(g, v);
}

// Node constructors with reserved inputs to avoid reallocs.
template<Operator Op>
[[gnu::hot]] inline ADNodePtr make_node_1in(const ADGraphPtr& g, const ADNodePtr& a) {
    auto n = std::make_shared<ADNode>();
    n->type = Op;
    n->inputs.reserve(1);
    n->addInput(a);
    g->addNode(n);
    return n;
}

template<Operator Op>
[[gnu::hot]] inline ADNodePtr make_node_2in(const ADGraphPtr& g, const ADNodePtr& a, const ADNodePtr& b) {
    auto n = std::make_shared<ADNode>();
    n->type = Op;
    n->inputs.reserve(2);
    n->addInput(a);
    n->addInput(b);
    g->addNode(n);
    return n;
}

// Wrap existing node into an Expression on graph g.
[[gnu::always_inline]] inline ExpressionPtr alias_expr(const ADGraphPtr& g, const ADNodePtr& n) {
    auto e = std::make_shared<Expression>(g);
    e->node = n;
    return e;
}

// Negation helper: uses dedicated op if available; otherwise (-1) * x
[[gnu::hot]] inline ADNodePtr negate_node_raw(const ADGraphPtr& g, const ADNodePtr& x) {
#ifdef AD_HAS_NEG_OP
    return make_node_1in<Operator::Neg>(g, x);
#else
    return make_node_2in<Operator::Multiply>(g, make_const_node_fast(g, -1.0), x);
#endif
}
[[gnu::always_inline]] inline ExpressionPtr negate_expr(const ADGraphPtr& g, const ADNodePtr& x) {
    return alias_expr(g, negate_node_raw(g, x));
}

} // namespace

// ============================================================================
// Expression ⊕ Expression
// ============================================================================

ExpressionPtr Expression::operator+(const Expression& o) const {
    auto g = pick_graph_fast(graph, o.graph);
    adopt_if_needed(g, graph,   node);
    adopt_if_needed(g, o.graph, o.node);
    return alias_expr(g, make_node_2in<Operator::Add>(g, node, o.node));
}

ExpressionPtr Expression::operator-(const Expression& o) const {
    auto g = pick_graph_fast(graph, o.graph);
    adopt_if_needed(g, graph,   node);
    adopt_if_needed(g, o.graph, o.node);
    return alias_expr(g, make_node_2in<Operator::Subtract>(g, node, o.node));
}

ExpressionPtr Expression::operator*(const Expression& o) const {
    auto g = pick_graph_fast(graph, o.graph);
    adopt_if_needed(g, graph,   node);
    adopt_if_needed(g, o.graph, o.node);
    return alias_expr(g, make_node_2in<Operator::Multiply>(g, node, o.node));
}

ExpressionPtr Expression::operator/(const Expression& o) const {
    auto g = pick_graph_fast(graph, o.graph);
    adopt_if_needed(g, graph,   node);
    adopt_if_needed(g, o.graph, o.node);
    return alias_expr(g, make_node_2in<Operator::Divide>(g, node, o.node));
}

// ============================================================================
// Expression ⊕ scalar (constant folding)
// ============================================================================

ExpressionPtr Expression::operator+(double s) const {
    auto g = graph ? graph : std::make_shared<ADGraph>();
    if (is_pos0(s)) [[unlikely]] return alias_expr(g, node); // x + 0 → x
    adopt_if_needed(g, graph, node);
    return alias_expr(g, make_node_2in<Operator::Add>(g, node, make_const_node_fast(g, s)));
}

ExpressionPtr Expression::operator-(double s) const {
    auto g = graph ? graph : std::make_shared<ADGraph>();
    if (is_pos0(s)) [[unlikely]] return alias_expr(g, node); // x - 0 → x
    adopt_if_needed(g, graph, node);
    return alias_expr(g, make_node_2in<Operator::Subtract>(g, node, make_const_node_fast(g, s)));
}

ExpressionPtr Expression::operator*(double s) const {
    auto g = graph ? graph : std::make_shared<ADGraph>();
    adopt_if_needed(g, graph, node);
    if (is_pos1(s)) [[unlikely]] return alias_expr(g, node);                     // x*1 → x
    if (is_pos0(s)) [[unlikely]] return alias_expr(g, make_const_node_fast(g, 0.0)); // x*0 → 0
    if (is_neg1(s)) [[unlikely]] return negate_expr(g, node);                    // x*(-1) → -x
    return alias_expr(g, make_node_2in<Operator::Multiply>(g, node, make_const_node_fast(g, s)));
}

ExpressionPtr Expression::operator/(double s) const {
    if (is_pos0(s)) [[unlikely]] throw std::domain_error("Division by zero in expression");
    auto g = graph ? graph : std::make_shared<ADGraph>();
    adopt_if_needed(g, graph, node);
    if (is_pos1(s)) [[unlikely]] return alias_expr(g, node); // x/1 → x
    if (is_neg1(s)) [[unlikely]] return negate_expr(g, node);
    return alias_expr(g, make_node_2in<Operator::Divide>(g, node, make_const_node_fast(g, s)));
}

// ============================================================================
// Expression ⊕ VariablePtr
// ============================================================================

ExpressionPtr Expression::operator+(const VariablePtr& v) const {
    auto g = graph ? graph : std::make_shared<ADGraph>();
    adopt_if_needed(g, graph, node);
    auto vexpr = std::make_shared<Expression>(v, 1.0, g);
    return alias_expr(g, make_node_2in<Operator::Add>(g, node, vexpr->node));
}

ExpressionPtr Expression::operator-(const VariablePtr& v) const {
    auto g = graph ? graph : std::make_shared<ADGraph>();
    adopt_if_needed(g, graph, node);
    auto vexpr = std::make_shared<Expression>(v, 1.0, g);
    return alias_expr(g, make_node_2in<Operator::Subtract>(g, node, vexpr->node));
}

ExpressionPtr Expression::operator*(const VariablePtr& v) const {
    auto g = graph ? graph : std::make_shared<ADGraph>();
    adopt_if_needed(g, graph, node);
    auto vexpr = std::make_shared<Expression>(v, 1.0, g);
    return alias_expr(g, make_node_2in<Operator::Multiply>(g, node, vexpr->node));
}

ExpressionPtr Expression::operator/(const VariablePtr& v) const {
    auto g = graph ? graph : std::make_shared<ADGraph>();
    adopt_if_needed(g, graph, node);
    auto vexpr = std::make_shared<Expression>(v, 1.0, g);
    return alias_expr(g, make_node_2in<Operator::Divide>(g, node, vexpr->node));
}

// ============================================================================
// Unary minus
// ============================================================================

ExpressionPtr Expression::operator-() const {
    auto g = graph ? graph : std::make_shared<ADGraph>();
    adopt_if_needed(g, graph, node);
    return negate_expr(g, node);
}

// ============================================================================
// Reverse scalar ops
// ============================================================================

ExpressionPtr operator+(double lhs, const Expression& rhs) {
    auto g = rhs.graph ? rhs.graph : std::make_shared<ADGraph>();
    if (is_pos0(lhs)) [[unlikely]] {
        adopt_if_needed(g, rhs.graph, rhs.node);
        return alias_expr(g, rhs.node);          // 0 + x → x
    }
    adopt_if_needed(g, rhs.graph, rhs.node);
    return alias_expr(g, make_node_2in<Operator::Add>(g, make_const_node_fast(g, lhs), rhs.node));
}

ExpressionPtr operator-(double lhs, const Expression& rhs) {
    auto g = rhs.graph ? rhs.graph : std::make_shared<ADGraph>();
    adopt_if_needed(g, rhs.graph, rhs.node);
    if (is_pos0(lhs)) [[unlikely]] {
        return negate_expr(g, rhs.node);         // 0 - x → -x
    }
    return alias_expr(g, make_node_2in<Operator::Subtract>(g, make_const_node_fast(g, lhs), rhs.node));
}

ExpressionPtr operator*(double lhs, const Expression& rhs) {
    // reuse rhs * lhs for symmetry and shared fast-paths
    return rhs * lhs;
}

ExpressionPtr operator/(double lhs, const Expression& rhs) {
    auto g = rhs.graph ? rhs.graph : std::make_shared<ADGraph>();
    adopt_if_needed(g, rhs.graph, rhs.node);
    if (is_pos0(lhs)) [[unlikely]] return alias_expr(g, make_const_node_fast(g, 0.0)); // 0/x → 0
    if (is_pos1(lhs)) [[unlikely]]
        return alias_expr(g, make_node_2in<Operator::Divide>(g, make_const_node_fast(g, 1.0), rhs.node));
    return alias_expr(g, make_node_2in<Operator::Divide>(g, make_const_node_fast(g, lhs), rhs.node));
}

// ============================================================================
// Convenience
// ============================================================================

ExpressionPtr square(const Expression& x) {
    auto g = x.graph ? x.graph : std::make_shared<ADGraph>();
    adopt_if_needed(g, x.graph, x.node);
    return alias_expr(g, make_node_2in<Operator::Multiply>(g, x.node, x.node));
}

ExpressionPtr reciprocal(const Expression& x) {
    auto g = x.graph ? x.graph : std::make_shared<ADGraph>();
    adopt_if_needed(g, x.graph, x.node);
    return alias_expr(g, make_node_2in<Operator::Divide>(g, make_const_node_fast(g, 1.0), x.node));
}

ExpressionPtr pow(const Expression& x, double p) {
    auto g = x.graph ? x.graph : std::make_shared<ADGraph>();
    adopt_if_needed(g, x.graph, x.node);

    if (is_pos1(p))  [[unlikely]] return alias_expr(g, x.node);                // x^1 → x
    if (is_pos0(p))  [[unlikely]] return alias_expr(g, make_const_node_fast(g, 1.0)); // x^0 → 1
    if (p == 2.0)    [[unlikely]] return square(x);                            // x^2
    if (p == -1.0)   [[unlikely]] return reciprocal(x);                        // x^-1

    // Generic: x^p = exp(p * log(x))
    auto ln  = make_node_1in<Operator::Log>(g, x.node);
    auto scl = make_node_2in<Operator::Multiply>(g, ln, make_const_node_fast(g, p));
    auto ex  = make_node_1in<Operator::Exp>(g, scl);
    return alias_expr(g, ex);
}

// ============================================================================
// Math functions
// ============================================================================

template<Operator Op>
[[gnu::hot]] static ExpressionPtr create_unary_math_op(const Expression& x) {
    auto g = x.graph ? x.graph : std::make_shared<ADGraph>();
    adopt_if_needed(g, x.graph, x.node);
    return alias_expr(g, make_node_1in<Op>(g, x.node));
}

ExpressionPtr sin (const Expression& x) { return create_unary_math_op<Operator::Sin >(x); }
ExpressionPtr cos (const Expression& x) { return create_unary_math_op<Operator::Cos >(x); }
ExpressionPtr tan (const Expression& x) { return create_unary_math_op<Operator::Tan >(x); }
ExpressionPtr exp (const Expression& x) { return create_unary_math_op<Operator::Exp >(x); }
ExpressionPtr log (const Expression& x) { return create_unary_math_op<Operator::Log >(x); }
ExpressionPtr tanh(const Expression& x) { return create_unary_math_op<Operator::Tanh>(x); }
ExpressionPtr silu(const Expression& x) { return create_unary_math_op<Operator::Silu>(x); }
ExpressionPtr gelu(const Expression& x) { return create_unary_math_op<Operator::Gelu>(x); }
ExpressionPtr relu(const Expression& x) { return create_unary_math_op<Operator::Relu>(x); }

// ============================================================================
// Constructor
// ============================================================================

Expression::Expression(const ADGraphPtr& g)
    : graph(g ? g : std::make_shared<ADGraph>()) {
    initializeNode();
}
