#pragma once
#include "ADGraph.h"
#include "Expression.h"
#include "Variable.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

// ---------- Small helpers ----------
inline ADNodePtr make_const_node(const ADGraphPtr &g, double v) {
    auto c = std::make_shared<ADNode>();
    c->type = Operator::cte;
    c->value = v;
    if (g) g->addNode(c);
    return c;
}

// Adopt both possible roots from an Expression and return the node to use.
inline ADNodePtr attach_input(const ExpressionPtr &e, const ADGraphPtr &g) {
    if (!e) return nullptr;
    if (e->node)      g->adoptSubgraph(e->node);
    if (e->rootNode)  g->adoptSubgraph(e->rootNode);
    return e->rootNode ? e->rootNode : e->node;
}

// Helpers to recognize constants quickly.
inline bool is_const(const ADNodePtr& n, double* out = nullptr) {
    if (n && n->type == Operator::cte) {
        if (out) *out = n->value;
        return true;
    }
    return false;
}

// Flatten inputs if the child already matches the same n-ary op.
inline void flatten_into(Operator op, const ADNodePtr& child, std::vector<ADNodePtr>& dst) {
    if (child && child->type == op) {
        dst.insert(dst.end(), child->inputs.begin(), child->inputs.end());
    } else {
        dst.push_back(child);
    }
}

// Build an n-ary node with already-materialized input vector.
inline ADNodePtr build_nary_node(const ADGraphPtr& g, Operator op, std::vector<ADNodePtr>&& ins) {
    auto out = std::make_shared<ADNode>();
    out->type = op;
    out->inputs = std::move(ins);
    g->addNode(out);
    return out;
}

// ======================= ExpressionPtr ⊕ ExpressionPtr =======================

inline ExpressionPtr operator+(const ExpressionPtr &lhs, const ExpressionPtr &rhs) {
    auto g = pick_graph(lhs ? lhs->graph : nullptr, rhs ? rhs->graph : nullptr);
    auto a = attach_input(lhs, g);
    auto b = attach_input(rhs, g);

    // Constant folding
    double av, bv;
    if (is_const(a, &av) && is_const(b, &bv))
        return std::make_shared<Expression>(build_nary_node(g, Operator::cte, std::vector<ADNodePtr>{ make_const_node(g, av + bv) }), g);
    if (is_const(a, &av) && av == 0.0) return std::make_shared<Expression>(b, g);
    if (is_const(b, &bv) && bv == 0.0) return std::make_shared<Expression>(a, g);

    // Flatten n-ary Add
    std::vector<ADNodePtr> ins;
    ins.reserve( (a && a->type==Operator::Add ? a->inputs.size() : 1)
               + (b && b->type==Operator::Add ? b->inputs.size() : 1) );
    flatten_into(Operator::Add, a, ins);
    flatten_into(Operator::Add, b, ins);
    return std::make_shared<Expression>( build_nary_node(g, Operator::Add, std::move(ins)), g );
}

inline ExpressionPtr operator-(const ExpressionPtr &lhs, const ExpressionPtr &rhs) {
    auto g = pick_graph(lhs ? lhs->graph : nullptr, rhs ? rhs->graph : nullptr);
    auto a = attach_input(lhs, g);
    auto b = attach_input(rhs, g);

    double av, bv;
    if (is_const(a, &av) && is_const(b, &bv))
        return std::make_shared<Expression>(make_const_node(g, av - bv), g);
    if (is_const(b, &bv) && bv == 0.0)
        return std::make_shared<Expression>(a, g);

    // a - b (keep binary; n-ary “Add + (-1*rhs)” would also work)
    auto out = std::make_shared<ADNode>();
    out->type = Operator::Subtract;
    out->addInput(a);
    out->addInput(b);
    g->addNode(out);
    return std::make_shared<Expression>(out, g);
}

inline ExpressionPtr operator*(const ExpressionPtr &lhs, const ExpressionPtr &rhs) {
    auto g = pick_graph(lhs ? lhs->graph : nullptr, rhs ? rhs->graph : nullptr);
    auto a = attach_input(lhs, g);
    auto b = attach_input(rhs, g);

    // Constant folding
    double av, bv;
    if (is_const(a, &av) && is_const(b, &bv))
        return std::make_shared<Expression>(make_const_node(g, av * bv), g);
    if ((is_const(a, &av) && av == 0.0) || (is_const(b, &bv) && bv == 0.0))
        return std::make_shared<Expression>(make_const_node(g, 0.0), g);
    if (is_const(a, &av) && av == 1.0) return std::make_shared<Expression>(b, g);
    if (is_const(b, &bv) && bv == 1.0) return std::make_shared<Expression>(a, g);

    // Flatten n-ary Multiply
    std::vector<ADNodePtr> ins;
    ins.reserve( (a && a->type==Operator::Multiply ? a->inputs.size() : 1)
               + (b && b->type==Operator::Multiply ? b->inputs.size() : 1) );
    flatten_into(Operator::Multiply, a, ins);
    flatten_into(Operator::Multiply, b, ins);
    return std::make_shared<Expression>( build_nary_node(g, Operator::Multiply, std::move(ins)), g );
}

inline ExpressionPtr operator/(const ExpressionPtr &lhs, const ExpressionPtr &rhs) {
    auto g = pick_graph(lhs ? lhs->graph : nullptr, rhs ? rhs->graph : nullptr);
    auto a = attach_input(lhs, g);
    auto b = attach_input(rhs, g);

    double av, bv;
    if (is_const(a, &av) && is_const(b, &bv))
        return std::make_shared<Expression>(make_const_node(g, (bv != 0.0) ? (av / bv) : 0.0), g);
    if (is_const(b, &bv) && bv == 1.0)
        return std::make_shared<Expression>(a, g);
    if (is_const(a, &av) && av == 0.0)
        return std::make_shared<Expression>(make_const_node(g, 0.0), g);

    auto out = std::make_shared<ADNode>();
    out->type = Operator::Divide;
    out->addInput(a);
    out->addInput(b);
    g->addNode(out);
    return std::make_shared<Expression>(out, g);
}

// ======================= ExpressionPtr ⊕ double =======================

inline ExpressionPtr operator+(const ExpressionPtr &expr, double scalar) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    if (scalar == 0.0) return expr;

    auto a = attach_input(expr, g);
    auto c = make_const_node(g, scalar);

    // (a + scalar) with flatten
    std::vector<ADNodePtr> ins;
    ins.reserve((a && a->type==Operator::Add ? a->inputs.size() : 1) + 1);
    flatten_into(Operator::Add, a, ins);
    ins.push_back(c);
    return std::make_shared<Expression>( build_nary_node(g, Operator::Add, std::move(ins)), g );
}

inline ExpressionPtr operator-(const ExpressionPtr &expr, double scalar) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    if (scalar == 0.0) return expr;

    auto a = attach_input(expr, g);
    auto c = make_const_node(g, scalar);

    auto out = std::make_shared<ADNode>();
    out->type = Operator::Subtract;
    out->addInput(a);
    out->addInput(c);
    g->addNode(out);
    return std::make_shared<Expression>(out, g);
}

inline ExpressionPtr operator*(const ExpressionPtr &expr, double scalar) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    if (scalar == 0.0) return std::make_shared<Expression>(make_const_node(g, 0.0), g);
    if (scalar == 1.0) return expr;

    auto a = attach_input(expr, g);
    auto c = make_const_node(g, scalar);

    std::vector<ADNodePtr> ins;
    ins.reserve((a && a->type==Operator::Multiply ? a->inputs.size() : 1) + 1);
    flatten_into(Operator::Multiply, a, ins);
    ins.push_back(c);
    return std::make_shared<Expression>( build_nary_node(g, Operator::Multiply, std::move(ins)), g );
}

inline ExpressionPtr operator/(const ExpressionPtr &expr, double scalar) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    if (scalar == 0.0) return std::make_shared<Expression>(make_const_node(g, 0.0), g); // your Div rule guards zero
    if (scalar == 1.0) return expr;

    auto a = attach_input(expr, g);
    auto c = make_const_node(g, scalar);

    auto out = std::make_shared<ADNode>();
    out->type = Operator::Divide;
    out->addInput(a);
    out->addInput(c);
    g->addNode(out);
    return std::make_shared<Expression>(out, g);
}

// ======================= double ⊕ ExpressionPtr (reverse scalar ops) =======================

inline ExpressionPtr operator+(double scalar, const ExpressionPtr &expr) {
    return expr + scalar;
}

inline ExpressionPtr operator-(double scalar, const ExpressionPtr &expr) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    if (scalar == 0.0) {
        // 0 - expr
        auto a = attach_input(expr, g);
        auto m1 = make_const_node(g, -1.0);
        auto out = std::make_shared<ADNode>();
        out->type = Operator::Multiply;
        out->addInput(m1);
        out->addInput(a);
        g->addNode(out);
        return std::make_shared<Expression>(out, g);
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

inline ExpressionPtr operator*(double scalar, const ExpressionPtr &expr) {
    return expr * scalar;
}

inline ExpressionPtr operator/(double scalar, const ExpressionPtr &expr) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    auto b = attach_input(expr, g);
    auto c = make_const_node(g, scalar);

    if (scalar == 0.0)
        return std::make_shared<Expression>(make_const_node(g, 0.0), g);

    auto out = std::make_shared<ADNode>();
    out->type = Operator::Divide;
    out->addInput(c);
    out->addInput(b);
    g->addNode(out);
    return std::make_shared<Expression>(out, g);
}

// ======================= ADNodePtr (+/-/*//) double =======================
// Wraps in a fresh graph, adopts the subgraph, then applies the scalar op.
inline ExpressionPtr operator+(const ADNodePtr &node, double scalar) {
    auto g = std::make_shared<ADGraph>();
    g->adoptSubgraph(node);
    auto e = std::make_shared<Expression>(node, g);
    return e + scalar;
}
inline ExpressionPtr operator-(const ADNodePtr &node, double scalar) {
    auto g = std::make_shared<ADGraph>();
    g->adoptSubgraph(node);
    auto e = std::make_shared<Expression>(node, g);
    return e - scalar;
}
inline ExpressionPtr operator*(const ADNodePtr &node, double scalar) {
    auto g = std::make_shared<ADGraph>();
    g->adoptSubgraph(node);
    auto e = std::make_shared<Expression>(node, g);
    return e * scalar;
}
inline ExpressionPtr operator/(const ADNodePtr &node, double scalar) {
    auto g = std::make_shared<ADGraph>();
    g->adoptSubgraph(node);
    auto e = std::make_shared<Expression>(node, g);
    return e / scalar;
}

// ======================= VariablePtr (+/-/*//) ExpressionPtr =======================

inline ExpressionPtr operator+(const VariablePtr &var, const ExpressionPtr &expr) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    auto v = std::make_shared<Expression>(var, 1.0, g);
    auto a = attach_input(v, g);
    auto b = attach_input(expr, g);

    // Flatten Add
    std::vector<ADNodePtr> ins;
    ins.reserve((a && a->type==Operator::Add ? a->inputs.size() : 1)
              + (b && b->type==Operator::Add ? b->inputs.size() : 1));
    flatten_into(Operator::Add, a, ins);
    flatten_into(Operator::Add, b, ins);
    return std::make_shared<Expression>( build_nary_node(g, Operator::Add, std::move(ins)), g );
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

inline ExpressionPtr operator*(const VariablePtr &var, const ExpressionPtr &expr) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    auto v = std::make_shared<Expression>(var, 1.0, g);
    auto a = attach_input(v, g);
    auto b = attach_input(expr, g);

    // Flatten Multiply
    std::vector<ADNodePtr> ins;
    ins.reserve((a && a->type==Operator::Multiply ? a->inputs.size() : 1)
              + (b && b->type==Operator::Multiply ? b->inputs.size() : 1));
    flatten_into(Operator::Multiply, a, ins);
    flatten_into(Operator::Multiply, b, ins);
    return std::make_shared<Expression>( build_nary_node(g, Operator::Multiply, std::move(ins)), g );
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

// ======================= Unary minus =======================

inline ExpressionPtr operator-(const ExpressionPtr &expr) {
    auto g = pick_graph(expr ? expr->graph : nullptr);
    auto a = attach_input(expr, g);

    // Prefer multiply by -1 to keep Multiply n-ary flattening benefits.
    auto m1 = make_const_node(g, -1.0);
    std::vector<ADNodePtr> ins;
    ins.reserve((a && a->type==Operator::Multiply ? a->inputs.size() : 1) + 1);
    flatten_into(Operator::Multiply, a, ins);
    ins.push_back(m1);
    return std::make_shared<Expression>( build_nary_node(g, Operator::Multiply, std::move(ins)), g );
}

// ======================= maximum(lhs, rhs) =======================

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
