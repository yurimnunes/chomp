#pragma once
#include "ADGraph.h"
#include "Expression.h"
#include "Variable.h"

#include <memory>
#include <string>
#include <utility>

// ---------- Small helpers ----------
inline ADNodePtr make_const_node(const ADGraphPtr &g, double v) {
    auto c = std::make_shared<ADNode>();
    c->type = Operator::cte;
    c->value = v;
    if (g)
        g->addNode(c);
    return c;
}

// Get a node to connect, adopting its subgraph into g.
inline ADNodePtr attach_input(const ExpressionPtr &e, const ADGraphPtr &g) {
    if (!e)
        return nullptr;
    if (e->node)
        g->adoptSubgraph(e->node);
    return e->rootNode ? e->rootNode : e->node;
}

// ---------- ExpressionPtr ⊕ ExpressionPtr ----------
inline ExpressionPtr operator+(const ExpressionPtr &lhs,
                               const ExpressionPtr &rhs) {
    auto g = pick_graph(lhs ? lhs->graph : nullptr, rhs ? rhs->graph : nullptr);
    auto a = attach_input(lhs, g);
    auto b = attach_input(rhs, g);

    auto out = std::make_shared<Expression>(g);
    out->node->type = Operator::Add;
    out->node->addInput(a);
    out->node->addInput(b);
    g->addNode(out->node);
    return out;
}

inline ExpressionPtr operator-(const ExpressionPtr &lhs,
                               const ExpressionPtr &rhs) {
    auto g = pick_graph(lhs ? lhs->graph : nullptr, rhs ? rhs->graph : nullptr);
    auto a = attach_input(lhs, g);
    auto b = attach_input(rhs, g);

    auto out = std::make_shared<Expression>(g);
    out->node->type = Operator::Subtract;
    out->node->addInput(a);
    out->node->addInput(b);
    g->addNode(out->node);
    return out;
}

inline ExpressionPtr operator*(const ExpressionPtr &lhs,
                               const ExpressionPtr &rhs) {
    auto g = pick_graph(lhs ? lhs->graph : nullptr, rhs ? rhs->graph : nullptr);
    auto a = attach_input(lhs, g);
    auto b = attach_input(rhs, g);

    auto out = std::make_shared<Expression>(g);
    out->node->type = Operator::Multiply;
    out->node->addInput(a);
    out->node->addInput(b);
    g->addNode(out->node);
    return out;
}

inline ExpressionPtr operator/(const ExpressionPtr &lhs,
                               const ExpressionPtr &rhs) {
    auto g = pick_graph(lhs ? lhs->graph : nullptr, rhs ? rhs->graph : nullptr);
    auto a = attach_input(lhs, g);
    auto b = attach_input(rhs, g);

    auto out = std::make_shared<Expression>(g);
    out->node->type = Operator::Divide;
    out->node->addInput(a);
    out->node->addInput(b);
    g->addNode(out->node);
    return out;
}

// ---------- ExpressionPtr ⊕ double ----------
inline ExpressionPtr operator+(const ExpressionPtr &expr, double scalar) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    auto a = attach_input(expr, g);
    auto c = make_const_node(g, scalar);

    auto out = std::make_shared<Expression>(g);
    out->node->type = Operator::Add;
    out->node->addInput(a);
    out->node->addInput(c);
    g->addNode(out->node);
    return out;
}

inline ExpressionPtr operator-(const ExpressionPtr &expr, double scalar) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    auto a = attach_input(expr, g);
    auto c = make_const_node(g, scalar);

    auto out = std::make_shared<Expression>(g);
    out->node->type = Operator::Subtract;
    out->node->addInput(a);
    out->node->addInput(c);
    g->addNode(out->node);
    return out;
}

inline ExpressionPtr operator*(const ExpressionPtr &expr, double scalar) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    auto a = attach_input(expr, g);
    auto c = make_const_node(g, scalar);

    auto out = std::make_shared<Expression>(g);
    out->node->type = Operator::Multiply;
    out->node->addInput(a);
    out->node->addInput(c);
    g->addNode(out->node);
    return out;
}

inline ExpressionPtr operator/(const ExpressionPtr &expr, double scalar) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    auto a = attach_input(expr, g);
    auto c = make_const_node(g, scalar);

    auto out = std::make_shared<Expression>(g);
    out->node->type = Operator::Divide;
    out->node->addInput(a);
    out->node->addInput(c);
    g->addNode(out->node);
    return out;
}

// ---------- double ⊕ ExpressionPtr (reverse scalar ops) ----------
inline ExpressionPtr operator+(double scalar, const ExpressionPtr &expr) {
    return expr + scalar;
}

inline ExpressionPtr operator-(double scalar, const ExpressionPtr &expr) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    auto b = attach_input(expr, g);
    auto c = make_const_node(g, scalar);

    auto out = std::make_shared<Expression>(g);
    out->node->type = Operator::Subtract;
    out->node->addInput(c);
    out->node->addInput(b);
    g->addNode(out->node);
    return out;
}

inline ExpressionPtr operator*(double scalar, const ExpressionPtr &expr) {
    return expr * scalar;
}

inline ExpressionPtr operator/(double scalar, const ExpressionPtr &expr) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    auto b = attach_input(expr, g);
    auto c = make_const_node(g, scalar);

    auto out = std::make_shared<Expression>(g);
    out->node->type = Operator::Divide;
    out->node->addInput(c);
    out->node->addInput(b);
    g->addNode(out->node);
    return out;
}

// ---------- ADNodePtr (+/-/*//) double (wrap in a graph-aware Expression)
// ----------
inline ExpressionPtr operator+(const ADNodePtr &node, double scalar) {
    auto g = std::make_shared<ADGraph>();
    auto e = std::make_shared<Expression>(node, g);
    g->adoptSubgraph(node);
    return e + scalar;
}
inline ExpressionPtr operator-(const ADNodePtr &node, double scalar) {
    auto g = std::make_shared<ADGraph>();
    auto e = std::make_shared<Expression>(node, g);
    g->adoptSubgraph(node);
    return e - scalar;
}
inline ExpressionPtr operator*(const ADNodePtr &node, double scalar) {
    auto g = std::make_shared<ADGraph>();
    auto e = std::make_shared<Expression>(node, g);
    g->adoptSubgraph(node);
    return e * scalar;
}
inline ExpressionPtr operator/(const ADNodePtr &node, double scalar) {
    auto g = std::make_shared<ADGraph>();
    auto e = std::make_shared<Expression>(node, g);
    g->adoptSubgraph(node);
    return e / scalar;
}

// ---------- VariablePtr (+/-/*//) ExpressionPtr ----------
inline ExpressionPtr operator+(const VariablePtr &var,
                               const ExpressionPtr &expr) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    auto v = std::make_shared<Expression>(var, 1.0, g);
    auto a = attach_input(v, g);
    auto b = attach_input(expr, g);

    auto out = std::make_shared<Expression>(g);
    out->node->type = Operator::Add;
    out->node->addInput(a);
    out->node->addInput(b);
    g->addNode(out->node);
    return out;
}

inline ExpressionPtr operator-(const VariablePtr &var,
                               const ExpressionPtr &expr) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    auto v = std::make_shared<Expression>(var, 1.0, g);
    auto a = attach_input(v, g);
    auto b = attach_input(expr, g);

    auto out = std::make_shared<Expression>(g);
    out->node->type = Operator::Subtract;
    out->node->addInput(a);
    out->node->addInput(b);
    g->addNode(out->node);
    return out;
}

inline ExpressionPtr operator*(const VariablePtr &var,
                               const ExpressionPtr &expr) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    auto v = std::make_shared<Expression>(var, 1.0, g);
    auto a = attach_input(v, g);
    auto b = attach_input(expr, g);

    auto out = std::make_shared<Expression>(g);
    out->node->type = Operator::Multiply;
    out->node->addInput(a);
    out->node->addInput(b);
    g->addNode(out->node);
    return out;
}

inline ExpressionPtr operator/(const VariablePtr &var,
                               const ExpressionPtr &expr) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    auto v = std::make_shared<Expression>(var, 1.0, g);
    auto a = attach_input(v, g);
    auto b = attach_input(expr, g);

    auto out = std::make_shared<Expression>(g);
    out->node->type = Operator::Divide;
    out->node->addInput(a);
    out->node->addInput(b);
    g->addNode(out->node);
    return out;
}

// ---------- Unary minus ----------
inline ExpressionPtr operator-(const ExpressionPtr &expr) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    auto a = attach_input(expr, g);
    auto m1 = make_const_node(g, -1.0);

    auto out = std::make_shared<Expression>(g);
    out->node->type = Operator::Multiply;
    out->node->addInput(m1);
    out->node->addInput(a);
    g->addNode(out->node);
    return out;
}

// ---------- maximum(lhs, rhs) ----------
inline ExpressionPtr maximum(const ExpressionPtr &lhs,
                             const ExpressionPtr &rhs) {
    auto g = pick_graph(lhs ? lhs->graph : nullptr, rhs ? rhs->graph : nullptr);
    auto a = attach_input(lhs, g);
    auto b = attach_input(rhs, g);

    auto out = std::make_shared<Expression>(g);
    out->node->type =
        Operator::Max; // requires Operator::Max and ADGraph support
    out->node->addInput(a);
    out->node->addInput(b);
    g->addNode(out->node);
    return out;
}
