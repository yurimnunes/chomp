#pragma once

#include "ADGraph.h"
#include "Definitions.h"
#include "Variable.h"

#include <algorithm>
#include <limits>
#include <map> // <-- needed for std::map in getVariables()
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class Expression;
using ExpressionPtr = std::shared_ptr<Expression>;
using VariablePtr = std::shared_ptr<Variable>;
using ADGraphPtr = std::shared_ptr<ADGraph>;

class Expression : public std::enable_shared_from_this<Expression> {
public:
    ADGraphPtr graph{};
    ADNodePtr node{};
    ADNodePtr rootNode{};

    // Construct with graph (creates an empty node placeholder)
    explicit Expression(const ADGraphPtr &graphIn);

    // Constant expression (placeholder variant)
    explicit Expression(double /*constante*/,
                        const ADGraphPtr &graphIn = nullptr)
        : graph(graphIn ? graphIn : std::make_shared<ADGraph>()) {
        initializeNode();
    }

    Expression &operator=(const Expression &) = default;

    // From a (placeholder) Variable object; creates an empty node
    explicit Expression(const Variable) {
        graph = std::make_shared<ADGraph>();
        initializeNode();
    }

    // From a VariablePtr: build a Var node with meta (name/value/order/bounds)
    Expression(const VariablePtr &variable, double /*coeff*/ = 1.0,
               const ADGraphPtr &graphIn = nullptr)
        : graph(graphIn ? graphIn : std::make_shared<ADGraph>()) {
        initializeNode(Operator::Var, variable->getName(), variable->getValue(),
                       variable->getOrder(), variable->getLowerBound(),
                       variable->getUpperBound());
        graph->nodeVariables[variable->getName()] = node;
    }

    // From an existing node
    Expression(const ADNodePtr &n, const ADGraphPtr &graphIn = nullptr)
        : graph(graphIn ? graphIn : std::make_shared<ADGraph>()), node(n) {
        if (!node) {
            initializeNode();
        } else {
            graph->addNode(node);
        }
    }

    // Deep-copy ctor (optional)
    Expression(const Expression &other)
        : graph(std::make_shared<ADGraph>(*other.graph)), node(other.node),
          rootNode(other.rootNode), expVariables(other.expVariables) {}

    // -------- calculus APIs --------
    std::string toString() {
        prepare();
        return graph->getExpression(rootNode);
    }

    tsl::robin_map<std::string, double> computeGradient() {
        prepare();
        return graph->computePartialDerivatives(rootNode);
    }

    // Dense Hessian via n HVPs (column-by-column)
    std::unordered_map<std::string, std::unordered_map<std::string, double>>
    computeHessian() {
        prepare();

        std::unordered_map<std::string, std::unordered_map<std::string, double>>
            H;
        if (!rootNode)
            return H;

        graph->initializeNodeVariables();
        const int n = static_cast<int>(graph->nodeVariables.size());
        if (n == 0)
            return H;

        std::vector<std::string> idx2name(n);
        for (auto &kv : graph->nodeVariables) {
            if (kv.second && kv.second->order >= 0 && kv.second->order < n) {
                idx2name[kv.second->order] = kv.first;
            }
        }

        for (int j = 0; j < n; ++j) {
            std::vector<double> e(n, 0.0);
            e[j] = 1.0;
            auto col = graph->hessianVectorProduct(rootNode, e);

            const std::string &cj = idx2name[j];
            for (int i = 0; i < n; ++i) {
                const std::string &ri = idx2name[i];
                H[ri][cj] = (i < static_cast<int>(col.size())) ? col[i] : 0.0;
            }
        }
        return H;
    }

    std::vector<std::pair<std::string, double>> getGradient() {
        std::vector<std::pair<std::string, double>> out;
        auto m = computeGradient();
        out.reserve(m.size());
        for (auto &kv : m)
            out.emplace_back(kv.first, kv.second);
        return out;
    }

    double wrt(const VariablePtr &var) {
        prepare();
        return graph->getGradientOfVariable(var, rootNode);
    }

    std::vector<double> getJacobian() const {
        const_cast<Expression *>(this)->prepare();
        return graph->getGradientVector(rootNode);
    }

    ADNodePtr getRootNode() const {
        const_cast<Expression *>(this)->prepare();
        return rootNode;
    }

    double evaluate() const {
        const_cast<Expression *>(this)->prepare();
        return graph->evaluate(rootNode);
    }

    double evaluate(std::unordered_map<std::string, double> &values) {
        prepare();
        for (auto &kv : values) {
            auto it = expVariables.find(kv.first);
            if (it != expVariables.end() && it->second) {
                it->second->value = kv.second;
            }
        }
        return graph->evaluate(rootNode);
    }

    // Return true if a variable with that name exists in the graph and was
    // updated.
    bool setVariable(const VariablePtr &variable, double /*coeff*/ = 1.0) {
        if (!graph || !variable)
            return false;

        // Prefer O(1) lookup
        auto it = graph->nodeVariables.find(variable->getName());
        if (it != graph->nodeVariables.end() && it->second) {
            ADNodePtr n = it->second;
            n->value = variable->getValue();
            // keep metadata in sync (optional, but handy)
            n->lb = variable->getLowerBound();
            n->ub = variable->getUpperBound();
            return true;
        }

        // Fallback: rare cases where map isn’t populated yet
        for (auto &n : graph->nodes) {
            if (n && n->name == variable->getName()) {
                n->value = variable->getValue();
                n->lb = variable->getLowerBound();
                n->ub = variable->getUpperBound();
                // Also update the map so future lookups are O(1)
                graph->nodeVariables[variable->getName()] = n;
                return true;
            }
        }

        return false;
    }

    void printGraph() {
        prepare();
        graph->printTree(rootNode);
    }

    // -------- operator overloads (implemented in Expression.cpp) ----
    // Expression ⊕ Expression
    ExpressionPtr operator+(const Expression &other) const;
    ExpressionPtr operator-(const Expression &other) const;
    ExpressionPtr operator*(const Expression &other) const;
    ExpressionPtr operator/(const Expression &other) const; // NEW

    // Expression ⊕ scalar
    ExpressionPtr operator+(double scalar) const;
    ExpressionPtr operator-(double scalar) const;
    ExpressionPtr operator*(double scalar) const;
    ExpressionPtr operator/(double scalar) const; // NEW

    // Expression ⊕ VariablePtr
    ExpressionPtr operator+(const VariablePtr &other) const;
    ExpressionPtr operator-(const VariablePtr &other) const;
    ExpressionPtr operator*(const VariablePtr &other) const;
    ExpressionPtr operator/(const VariablePtr &other) const; // NEW

    // Unary minus
    ExpressionPtr operator-() const; // NEW

    // Variable access helpers (stubs)
    std::map<VariablePtr, double> getVariables() const { return {}; }
    std::vector<VariablePtr> getVariablesUnique() { return {}; }

    // Update a variable’s current value by name
    void setVar(std::string name, double value) {
        auto it = expVariables.find(name);
        if (it != expVariables.end() && it->second)
            it->second->value = value;
    }

private:
    tsl::robin_map<std::string, ADNodePtr> expVariables;

    void initializeNode(Operator type = Operator::NA, std::string name = "",
                        double value = 0.0, int order = -1,
                        double lb = -std::numeric_limits<double>::infinity(),
                        double ub = std::numeric_limits<double>::infinity()) {
        node = std::make_shared<ADNode>();
        node->type = type;
        node->name = std::move(name);
        node->value = value;
        node->order = order;
        node->lb = lb;
        node->ub = ub;
        if (graph)
            graph->addNode(node);
    }

    // Prepare the graph for evaluation/derivatives
    void prepare() {
        if (!graph)
            graph = std::make_shared<ADGraph>();
        if (node)
            graph->addNode(node);

        auto rebuilt = graph->rebuildGraphWithUniqueVariables(node);
        graph = std::get<0>(rebuilt);
        expVariables = std::get<1>(rebuilt);

        graph->initializeNodeVariables();

        auto roots = graph->findRootNodes();
        rootNode = !roots.empty() ? roots.back() : node;
    }
};

// -------- free operators (reverse scalar ops) ----
ExpressionPtr operator+(double lhs, const Expression &rhs);
ExpressionPtr operator-(double lhs, const Expression &rhs);
ExpressionPtr operator*(double lhs, const Expression &rhs);
ExpressionPtr operator/(double lhs, const Expression &rhs); // NEW

// -------- convenience functions (no new opcodes) ----
ExpressionPtr square(const Expression &x);        // NEW
ExpressionPtr reciprocal(const Expression &x);    // NEW
ExpressionPtr pow(const Expression &x, double p); // NEW

// -------- C++ trig/exp/log helpers ----
ExpressionPtr sin(const Expression &x);                          // NEW
ExpressionPtr cos(const Expression &x);                          // NEW
ExpressionPtr tan(const Expression &x);                          // NEW
ExpressionPtr exp(const Expression &x);                          // NEW
ExpressionPtr log(const Expression &x);                          // NEW
ExpressionPtr maximum(const Expression &a, const Expression &b); // NEW
