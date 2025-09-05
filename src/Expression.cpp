// Expression.cpp
#include "../include/Expression.h"
#include "../include/ADGraph.h"
#include "../include/Operators.h"  // make_const_node(g, double)
#include "../include/Variable.h"

#include <memory>

// -------------------------------
// Expression ⊕ Expression
// -------------------------------

ExpressionPtr Expression::operator+(const Expression& other) const {
    auto g = pick_graph(graph, other.graph);
    if (node)       g->adoptSubgraph(node);
    if (other.node) g->adoptSubgraph(other.node);

    auto result = std::make_shared<Expression>(g);
    result->node->type = Operator::Add;
    result->node->addInput(node);
    result->node->addInput(other.node);
    return result;
}

ExpressionPtr Expression::operator-(const Expression& other) const {
    auto g = pick_graph(graph, other.graph);
    if (node)       g->adoptSubgraph(node);
    if (other.node) g->adoptSubgraph(other.node);

    auto result = std::make_shared<Expression>(g);
    result->node->type = Operator::Subtract;
    result->node->addInput(node);
    result->node->addInput(other.node);
    return result;
}

ExpressionPtr Expression::operator*(const Expression& other) const {
    auto g = pick_graph(graph, other.graph);
    if (node)       g->adoptSubgraph(node);
    if (other.node) g->adoptSubgraph(other.node);

    auto result = std::make_shared<Expression>(g);
    result->node->type = Operator::Multiply;
    result->node->addInput(node);
    result->node->addInput(other.node);
    return result;
}

ExpressionPtr Expression::operator/(const Expression& other) const {
    auto g = pick_graph(graph, other.graph);
    if (node)       g->adoptSubgraph(node);
    if (other.node) g->adoptSubgraph(other.node);

    auto result = std::make_shared<Expression>(g);
    result->node->type = Operator::Divide;
    result->node->addInput(node);
    result->node->addInput(other.node);
    return result;
}

// -------------------------------
// Expression ⊕ scalar
// -------------------------------

ExpressionPtr Expression::operator+(double scalar) const {
    auto g = graph ? graph : std::make_shared<ADGraph>();
    if (node) g->adoptSubgraph(node);

    auto result = std::make_shared<Expression>(g);
    result->node->type = Operator::Add;
    result->node->addInput(node);
    result->node->addInput(make_const_node(g, scalar));
    return result;
}

ExpressionPtr Expression::operator-(double scalar) const {
    auto g = graph ? graph : std::make_shared<ADGraph>();
    if (node) g->adoptSubgraph(node);

    auto result = std::make_shared<Expression>(g);
    result->node->type = Operator::Subtract;
    result->node->addInput(node);
    result->node->addInput(make_const_node(g, scalar));
    return result;
}

ExpressionPtr Expression::operator*(double scalar) const {
    auto g = graph ? graph : std::make_shared<ADGraph>();
    if (node) g->adoptSubgraph(node);

    auto result = std::make_shared<Expression>(g);
    result->node->type = Operator::Multiply;
    result->node->addInput(node);
    result->node->addInput(make_const_node(g, scalar));
    return result;
}

ExpressionPtr Expression::operator/(double scalar) const {
    auto g = graph ? graph : std::make_shared<ADGraph>();
    if (node) g->adoptSubgraph(node);

    auto result = std::make_shared<Expression>(g);
    result->node->type = Operator::Divide;
    result->node->addInput(node);
    result->node->addInput(make_const_node(g, scalar));
    return result;
}

// -------------------------------
// Expression ⊕ VariablePtr
// -------------------------------

ExpressionPtr Expression::operator+(const VariablePtr& var) const {
    auto g = graph ? graph : std::make_shared<ADGraph>();
    if (node) g->adoptSubgraph(node);

    auto vexpr = std::make_shared<Expression>(var, 1.0, g); // var node on g

    auto result = std::make_shared<Expression>(g);
    result->node->type = Operator::Add;
    result->node->addInput(node);
    result->node->addInput(vexpr->node);
    return result;
}

ExpressionPtr Expression::operator-(const VariablePtr& var) const {
    auto g = graph ? graph : std::make_shared<ADGraph>();
    if (node) g->adoptSubgraph(node);

    auto vexpr = std::make_shared<Expression>(var, 1.0, g);

    auto result = std::make_shared<Expression>(g);
    result->node->type = Operator::Subtract;
    result->node->addInput(node);
    result->node->addInput(vexpr->node);
    return result;
}

ExpressionPtr Expression::operator*(const VariablePtr& var) const {
    auto g = graph ? graph : std::make_shared<ADGraph>();
    if (node) g->adoptSubgraph(node);

    auto vexpr = std::make_shared<Expression>(var, 1.0, g);

    auto result = std::make_shared<Expression>(g);
    result->node->type = Operator::Multiply;
    result->node->addInput(node);
    result->node->addInput(vexpr->node);
    return result;
}

ExpressionPtr Expression::operator/(const VariablePtr& var) const {
    auto g = graph ? graph : std::make_shared<ADGraph>();
    if (node) g->adoptSubgraph(node);

    auto vexpr = std::make_shared<Expression>(var, 1.0, g);

    auto result = std::make_shared<Expression>(g);
    result->node->type = Operator::Divide;
    result->node->addInput(node);
    result->node->addInput(vexpr->node);
    return result;
}

// -------------------------------
// Unary minus
// -------------------------------

ExpressionPtr Expression::operator-() const {
    // -(x) = (-1) * x
    auto g = graph ? graph : std::make_shared<ADGraph>();
    if (node) g->adoptSubgraph(node);

    auto result = std::make_shared<Expression>(g);
    result->node->type = Operator::Multiply;
    result->node->addInput(make_const_node(g, -1.0));
    result->node->addInput(node);
    return result;
}

// -------------------------------
// Reverse scalar ops (free functions)
// -------------------------------

ExpressionPtr operator+(double lhs, const Expression& rhs) {
    return rhs + lhs;
}

ExpressionPtr operator-(double lhs, const Expression& rhs) {
    auto g = rhs.graph ? rhs.graph : std::make_shared<ADGraph>();
    if (rhs.node) g->adoptSubgraph(rhs.node);

    auto result = std::make_shared<Expression>(g);
    result->node->type = Operator::Subtract;
    result->node->addInput(make_const_node(g, lhs));
    result->node->addInput(rhs.node);
    return result;
}

ExpressionPtr operator*(double lhs, const Expression& rhs) {
    return rhs * lhs;
}

ExpressionPtr operator/(double lhs, const Expression& rhs) {
    auto g = rhs.graph ? rhs.graph : std::make_shared<ADGraph>();
    if (rhs.node) g->adoptSubgraph(rhs.node);

    auto result = std::make_shared<Expression>(g);
    result->node->type = Operator::Divide;
    result->node->addInput(make_const_node(g, lhs));
    result->node->addInput(rhs.node);
    return result;
}

// -------------------------------
// Convenience functions (no new opcodes)
// -------------------------------

ExpressionPtr square(const Expression& x) {
    return x * x;
}

ExpressionPtr reciprocal(const Expression& x) {
    auto g = x.graph ? x.graph : std::make_shared<ADGraph>();
    if (x.node) g->adoptSubgraph(x.node);

    auto e = std::make_shared<Expression>(g);
    e->node->type = Operator::Divide;
    e->node->addInput(make_const_node(g, 1.0));
    e->node->addInput(x.node);
    return e;
}

ExpressionPtr pow(const Expression& x, double p) {
    // pow(x, p) = exp(p * log(x))
    auto g = x.graph ? x.graph : std::make_shared<ADGraph>();
    if (x.node) g->adoptSubgraph(x.node);

    // log(x)
    auto e_log = std::make_shared<Expression>(g);
    e_log->node->type = Operator::Log;
    e_log->node->addInput(x.node);
    g->addNode(e_log->node);

    // p * log(x)
    auto e_scale = std::make_shared<Expression>(g);
    e_scale->node->type = Operator::Multiply;
    e_scale->node->addInput(e_log->node);
    e_scale->node->addInput(make_const_node(g, p));
    g->addNode(e_scale->node);

    // exp(p * log(x))
    auto e_exp = std::make_shared<Expression>(g);
    e_exp->node->type = Operator::Exp;
    e_exp->node->addInput(e_scale->node);
    g->addNode(e_exp->node);

    return e_exp;
}

// -------------------------------
// Trig/exp/log helpers (C++ side)
// -------------------------------

ExpressionPtr sin(const Expression& x) {
    auto g = x.graph ? x.graph : std::make_shared<ADGraph>();
    if (x.node) g->adoptSubgraph(x.node);

    auto e = std::make_shared<Expression>(g);
    e->node->type = Operator::Sin;
    e->node->addInput(x.node);
    g->addNode(e->node);
    return e;
}

ExpressionPtr cos(const Expression& x) {
    auto g = x.graph ? x.graph : std::make_shared<ADGraph>();
    if (x.node) g->adoptSubgraph(x.node);

    auto e = std::make_shared<Expression>(g);
    e->node->type = Operator::Cos;
    e->node->addInput(x.node);
    g->addNode(e->node);
    return e;
}

ExpressionPtr tan(const Expression& x) {
    auto g = x.graph ? x.graph : std::make_shared<ADGraph>();
    if (x.node) g->adoptSubgraph(x.node);

    auto e = std::make_shared<Expression>(g);
    e->node->type = Operator::Tan;
    e->node->addInput(x.node);
    g->addNode(e->node);
    return e;
}

ExpressionPtr exp(const Expression& x) {
    auto g = x.graph ? x.graph : std::make_shared<ADGraph>();
    if (x.node) g->adoptSubgraph(x.node);

    auto e = std::make_shared<Expression>(g);
    e->node->type = Operator::Exp;
    e->node->addInput(x.node);
    g->addNode(e->node);
    return e;
}

ExpressionPtr log(const Expression& x) {
    auto g = x.graph ? x.graph : std::make_shared<ADGraph>();
    if (x.node) g->adoptSubgraph(x.node);

    auto e = std::make_shared<Expression>(g);
    e->node->type = Operator::Log;
    e->node->addInput(x.node);
    g->addNode(e->node);
    return e;
}

ExpressionPtr maximum(const Expression& a, const Expression& b) {
    auto g = (a.graph ? a.graph : (b.graph ? b.graph : std::make_shared<ADGraph>()));
    if (a.node) g->adoptSubgraph(a.node);
    if (b.node) g->adoptSubgraph(b.node);

    auto e = std::make_shared<Expression>(g);
    e->node->type = Operator::Max;
    e->node->addInput(a.node);
    e->node->addInput(b.node);
    g->addNode(e->node);
    return e;
}


// -------------------------------
// ctor
// -------------------------------

Expression::Expression(const ADGraphPtr& g)
    : graph(g ? g : std::make_shared<ADGraph>()) {
    initializeNode();
}
