// ADGraph.cpp - COMPLETE FIXED VERSION WITH PROPER EPOCH MANAGEMENT
#include "../include/ADGraph.h"
#include "../include/Definitions.h"
#include "../include/Variable.h"

#include "../include/OpDispatch.h"
#include "../include/OpTraits.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using VariablePtr = std::shared_ptr<Variable>;

// ===== ADNode =====
void ADNode::addInput(const ADNodePtr &inputNode) {
    inputs.push_back(inputNode);
}

// ===== Helpers for epoch-based access =====

// Thread-safe variants if needed
static std::mutex epoch_mutex;
static inline double &ensure_epoch_zero_safe(double &x, unsigned &e,
                                             unsigned cur) {
    std::lock_guard<std::mutex> lock(epoch_mutex);
    return ensure_epoch_zero(x, e, cur);
}

// ===== Local helpers =====
namespace {

inline bool validate_nary_inputs(const std::vector<ADNodePtr> &inputs,
                                 const std::string &op_name,
                                 size_t min_inputs = 1) {
    if (inputs.size() < min_inputs) {
        std::cerr << "Error: " << op_name << " operation needs at least "
                  << min_inputs << " inputs, got " << inputs.size()
                  << std::endl;
        return false;
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!inputs[i]) {
            std::cerr << "Error: " << op_name
                      << " operation has null input at index " << i
                      << std::endl;
            return false;
        }
    }
    return true;
}
} // namespace

// ===== ADGraph: Dirty marking & cache rebuild =====
void ADGraph::markDirty_() { cache_.dirty = true; }
void ADGraph::rebuildCache_() {
    // Enforce uniqueness before building indexes and topo
    makeNodesUnique();

    // Build node ID mapping
    cache_.id_of.clear();
    cache_.id_of.reserve(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i) {
        cache_.id_of[nodes[i].get()] = i;
    }

    // DFS topo with cycle detection
    if (!buildTopoOrderDFS_()) {
        cache_.topo.clear();
        return;
    }

    // Build output adjacency (optional)
    cache_.out.assign(nodes.size(), {});
    for (auto &sp : nodes) {
        ADNode *n = sp.get();
        for (auto &in : n->inputs) {
            if (!in)
                continue;
            auto it = cache_.id_of.find(in.get());
            if (it != cache_.id_of.end()) {
                cache_.out[it->second].push_back(n);
            }
        }
    }

    cache_.dirty = false;
}

// ===== Core graph maintenance =====
void ADGraph::deleteNode(const ADNodePtr &node) {
    if (!node)
        return;

    // Remove from index
    if (!node->name.empty()) {
        nodeIndex_.erase(node->name);
    }

    // Remove from nodes vector and update references
    for (auto &n : nodes) {
        auto &ins = n->inputs;
        ins.erase(std::remove(ins.begin(), ins.end(), node), ins.end());
    }
    nodes.erase(std::remove(nodes.begin(), nodes.end(), node), nodes.end());

    for (auto it = nodeVariables.begin(); it != nodeVariables.end();) {
        if (it->second == node)
            it = nodeVariables.erase(it);
        else
            ++it;
    }
    markDirty_();
}

void ADGraph::addNode(const ADNodePtr &node) {
    if (!node)
        return;

    if (std::find(nodes.begin(), nodes.end(), node) == nodes.end()) {
        nodes.push_back(node);
        if (!node->name.empty()) {
            nodeIndex_[node->name] =
                node; // last-in wins in the index (dedup will fix later)
        }
        markDirty_();
    }
}

void ADGraph::makeNodesUnique() {
    // Map of canonical node by name (first-seen wins)
    std::unordered_map<std::string, ADNodePtr> by_name;

    // Pointer remap: every original pointer -> canonical pointer
    std::unordered_map<ADNodePtr, ADNodePtr> remap;
    remap.reserve(nodes.size());

    std::vector<ADNodePtr> unique_nodes;
    unique_nodes.reserve(nodes.size());

    // 1) Choose canonicals and build pointer remap
    for (const auto &n : nodes) {
        if (!n)
            continue;

        if (!n->name.empty()) {
            auto it = by_name.find(n->name);
            if (it == by_name.end()) {
                // First occurrence becomes canonical
                by_name.emplace(n->name, n);
                remap[n] = n;
                unique_nodes.push_back(n);
            } else {
                // Duplicate name: point to canonical
                remap[n] = it->second;
            }
        } else {
            // Unnamed nodes are always unique by identity
            remap[n] = n;
            unique_nodes.push_back(n);
        }
    }

    // 2) Rewire all inputs to canonical pointers
    for (auto &n : unique_nodes) {
        if (!n)
            continue;
        for (auto &in : n->inputs) {
            if (!in)
                continue;
            auto it = remap.find(in);
            if (it != remap.end()) {
                in = it->second;
            }
            // else: input not in graph? leave as-is (dangling external)
        }
    }

    // 3) Rebuild nodeIndex_ to point to canonicals only
    nodeIndex_.clear();
    nodeIndex_.reserve(by_name.size());
    for (const auto &kv : by_name) {
        nodeIndex_[kv.first] = kv.second;
    }

    // 4) Update nodeVariables to canonical pointers
    for (auto it = nodeVariables.begin(); it != nodeVariables.end();) {
        const std::string &nm = it->first;
        auto idx = nodeIndex_.find(nm);
        if (idx != nodeIndex_.end() && idx->second) {
            it->second = idx->second; // canonical
            ++it;
        } else {
            // Variable name without a canonical node -> drop it
            it = nodeVariables.erase(it);
        }
    }

    // 5) Replace nodes with deduplicated list (order preserved by first-seen)
    nodes = std::move(unique_nodes);

    markDirty_();
}
// ===== Introspection / printing =====
std::string ADGraph::getExpression(const ADNodePtr &node) {
    if (!node)
        return "";

    auto expr = [&](const ADNodePtr &n) { return getExpression(n); };

    switch (node->type) {
    case Operator::Var:
        return node->name;

    case Operator::cte:
        return std::to_string(node->value);

    case Operator::Sin:
    case Operator::Cos:
    case Operator::Tan:
    case Operator::Exp:
    case Operator::Log: {
        // unary
        if (validate_unary_inputs(node->inputs, op_name(node->type))) {
            return std::string(op_name(node->type)) + "(" +
                   expr(node->inputs[0]) + ")";
        }
        return std::string(op_name(node->type)) + "(?)";
    }

    case Operator::Add:
    case Operator::Multiply: {
        // n-ary infix
        const char *sym = (node->type == Operator::Add) ? "+" : "*";
        if (!validate_nary_inputs(node->inputs, op_name(node->type), 1))
            return std::string(op_name(node->type)) + "(?)";
        std::string out = "(";
        for (size_t i = 0; i < node->inputs.size(); ++i) {
            if (i)
                out += std::string(" ") + sym + " ";
            out += expr(node->inputs[i]);
        }
        out += ")";
        return out;
    }

    case Operator::Subtract:
    case Operator::Divide:
    case Operator::Max: {
        // binary infix or min/max
        const char *sym = (node->type == Operator::Subtract) ? "-"
                          : (node->type == Operator::Divide)
                              ? "/"
                              : ","; // for max/min
        if (!validate_binary_inputs(node->inputs, op_name(node->type)))
            return std::string(op_name(node->type)) + "(?)";

        if (node->type == Operator::Max) {
            // reflect the new semantics: min(...) instead of max(...)
            return std::string("min(") + expr(node->inputs[0]) + ", " +
                   expr(node->inputs[1]) + ")";
        } else {
            return "(" + expr(node->inputs[0]) + " " + sym + " " +
                   expr(node->inputs[1]) + ")";
        }
    }

    default:
        return "Unsupported_Operator(" + std::string(op_name(node->type)) + ")";
    }
}
void ADGraph::printTree(const ADNodePtr &node, int depth) {
    if (!node)
        return;

    // Use static thread_local with proper cleanup
    static thread_local std::unordered_set<const ADNode *> seen;

    // RAII guard for cleanup at top level
    struct SeenGuard {
        bool is_top_level;
        std::unordered_set<const ADNode *> &seen_ref;

        SeenGuard(bool top, std::unordered_set<const ADNode *> &s)
            : is_top_level(top), seen_ref(s) {
            if (is_top_level)
                seen_ref.clear();
        }

        ~SeenGuard() {
            if (is_top_level)
                seen_ref.clear(); // Always cleanup at top level
        }
    };

    SeenGuard guard(depth == 0, seen);

    if (seen.count(node.get())) {
        std::string indent(depth * 4, ' ');
        std::cout << indent << "|- [visited] " << node.get() << " ("
                  << node->name << ")\n";
        return;
    }
    seen.insert(node.get());

    std::string indent(depth * 4, ' ');
    std::cout << indent << "|- Node: " << node.get() << "\n"
              << indent << "|    Name: " << node->name << "\n"
              << indent << "|    Expression: " << getExpression(node) << "\n"
              << indent << "|    Value: " << node->value << "\n"
              << indent << "|    Operator: " << op_name(node->type) << "\n"
              << indent << "|    Inputs:";

    if (node->inputs.empty()) {
        std::cout << " None\n\n";
    } else {
        std::cout << "\n";
        for (const auto &in : node->inputs)
            printTree(in, depth + 1);
        std::cout << "\n";
    }
}

std::vector<ADNodePtr> ADGraph::findRootNodes() const {
    std::vector<ADNodePtr> roots;
    std::unordered_set<ADNodePtr> inputs;
    inputs.reserve(nodes.size());
    for (const auto &n : nodes)
        for (const auto &in : n->inputs)
            inputs.insert(in);
    for (const auto &n : nodes)
        if (!inputs.count(n))
            roots.push_back(n);
    if (roots.empty())
        std::cerr << "Warning: No root nodes found\n";
    return roots;
}

void ADGraph::computeForwardPass() {
    if (cache_.dirty)
        rebuildCache_();
    for (ADNode *u : cache_.topo) {
        _dispatch_op(*u, ForwardFunctor{*this, *u});
    }
    if (cache_.topo.size() != nodes.size()) {
        std::cerr
            << "Warning: cycle detected or dangling inputs in AD graph.\n";
    }
}
// Replace the big switch version with:
void ADGraph::computeNodeValue(const ADNodePtr &node,
                               std::unordered_set<ADNodePtr> &visited) {
    if (!node || visited.count(node))
        return;
    for (const auto &in : node->inputs)
        computeNodeValue(in, visited);
    _dispatch_op(*node, ForwardFunctor{*this, *node});
    visited.insert(node);
}

void ADGraph::resetForwardPass() {
    // Lazy reset: just bump epoch counter
    ++cur_val_epoch_;
    if (cur_val_epoch_ == 0)
        ++cur_val_epoch_; // avoid 0 wrap
}

void ADGraph::initiateBackwardPass(const ADNodePtr & /*outputNode*/) {
    if (cache_.dirty)
        rebuildCache_();
    for (auto it = cache_.topo.rbegin(); it != cache_.topo.rend(); ++it) {
        ADNode *n = *it;
        _dispatch_op(*n, BackwardFunctor{*this, *n});
    }
}

void ADGraph::resetGradients() {
    ++cur_grad_epoch_;
    if (cur_grad_epoch_ == 0)
        ++cur_grad_epoch_;
}

// ===== Public gradient API =====
std::unordered_map<std::string, double>
ADGraph::computePartialDerivatives(const ADNodePtr &expr) {
    std::unordered_map<std::string, double> partials;
    resetGradients();
    resetForwardPass();
    computeForwardPass();
    if (expr)
        set_epoch_value(expr->gradient, expr->grad_epoch, cur_grad_epoch_, 1.0);
    initiateBackwardPass(expr);
    for (auto &kv : nodeVariables)
        if (kv.second)
            partials[kv.first] = kv.second->gradient;
    return partials;
}

ADNodePtr ADGraph::getNode(const std::string &name) {
    auto it = nodeIndex_.find(name);
    return (it != nodeIndex_.end()) ? it->second : nullptr;
}

double ADGraph::getGradientOfVariable(const VariablePtr &var,
                                      const ADNodePtr &expr) {
    auto n = getNode(var->getName());
    if (!n || !expr)
        return 0.0;
    resetGradients();
    resetForwardPass();
    computeForwardPass();
    set_epoch_value(expr->gradient, expr->grad_epoch, cur_grad_epoch_, 1.0);
    initiateBackwardPass(expr);
    return n->gradient;
}

// ===== Eval & misc =====
double ADGraph::evaluate(const ADNodePtr &expr) {
    resetGradients();
    resetForwardPass();
    computeForwardPass();
    return expr ? expr->value : 0.0;
}

void ADGraph::initializeNodeVariables() {
    int order = 0;
    for (auto &kv : nodeVariables)
        if (kv.second)
            kv.second->order = order++;
}

std::vector<double> ADGraph::getGradientVector(const ADNodePtr &expr) {
    initializeNodeVariables();
    size_t varSize = nodeVariables.size();
    computePartialDerivatives(expr);
    std::vector<double> g(varSize, 0.0);
    for (const auto &n : nodes)
        if (n->type == Operator::Var && n->order >= 0 &&
            static_cast<size_t>(n->order) < varSize)
            g[n->order] = n->gradient;
    return g;
}

// ===== Rebuild graph (unique variables) =====
std::tuple<ADGraphPtr, std::unordered_map<std::string, ADNodePtr>>
ADGraph::rebuildGraphWithUniqueVariables(const ADNodePtr &rootNode) {
    std::unordered_map<std::string, ADNodePtr> coll;
    std::unordered_set<ADNodePtr> vis;
    std::unordered_map<std::string, ADNodePtr> vars;
    collectNodes(rootNode, coll, vis, vars);
    auto newG = std::make_shared<ADGraph>();
    for (const auto &p : coll)
        newG->addNode(p.second);
    newG->nodeVariables = vars;
    newG->makeNodesUnique();
    return {newG, vars};
}

void ADGraph::collectNodes(const ADNodePtr &start,
                           std::unordered_map<std::string, ADNodePtr> &coll,
                           std::unordered_set<ADNodePtr> &vis,
                           std::unordered_map<std::string, ADNodePtr> &vars) {
    if (!start || vis.count(start))
        return;
    vis.insert(start);
    if (!start->name.empty()) {
        if (!coll.count(start->name))
            coll[start->name] = start;
        if (start->type == Operator::Var)
            vars[start->name] = start;
    } else {
        std::string key = "node_" + std::to_string(coll.size());
        coll.emplace(key, start);
    }
    for (const auto &in : start->inputs)
        collectNodes(in, coll, vis, vars);
}

// ===== HVP (forward-over-reverse) =====
void ADGraph::resetTangents() {
    ++cur_dot_epoch_;
    if (cur_dot_epoch_ == 0)
        ++cur_dot_epoch_;
}

void ADGraph::resetGradDot() {
    ++cur_gdot_epoch_;
    if (cur_gdot_epoch_ == 0)
        ++cur_gdot_epoch_;
}
void ADGraph::computeForwardPassWithDot() {
    if (cache_.dirty)
        rebuildCache_();
    for (ADNode *u : cache_.topo) {
        _dispatch_op(*u, ForwardFunctor{*this, *u});    // values
        _dispatch_op(*u, ForwardDotFunctor{*this, *u}); // tangents
    }
}

void ADGraph::initiateBackwardPassHVP() {
    if (cache_.dirty)
        rebuildCache_();
    for (auto it = cache_.topo.rbegin(); it != cache_.topo.rend(); ++it) {
        ADNode *n = *it;
        _dispatch_op(*n, HVPBackwardFunctor{*this, *n});
    }
}

std::vector<double>
ADGraph::hessianVectorProduct(const ADNodePtr &outputNode,
                              const std::vector<double> &v) {
    size_t nvars = nodeVariables.size();
    std::vector<double> Hv(nvars, 0.0);
    if (!outputNode || nvars == 0)
        return Hv;

    // seed tangents from v
    resetTangents();
    for (auto &kv : nodeVariables) {
        auto &varNode = kv.second;
        if (!varNode)
            continue;
        int k = varNode->order;
        if (k >= 0 && static_cast<size_t>(k) < v.size())
            set_epoch_value(varNode->dot, varNode->dot_epoch, cur_dot_epoch_,
                            v[k]);
    }

    // forward: values + dot
    resetForwardPass();
    computeForwardPassWithDot();

    // reverse: gradients + grad_dot
    resetGradients();
    resetGradDot();
    set_epoch_value(outputNode->gradient, outputNode->grad_epoch,
                    cur_grad_epoch_, 1.0);
    set_epoch_value(outputNode->grad_dot, outputNode->gdot_epoch,
                    cur_gdot_epoch_, 0.0);
    initiateBackwardPassHVP();

    // extract (H v) in variable order
    for (auto &kv : nodeVariables) {
        auto &varNode = kv.second;
        if (!varNode)
            continue;
        int k = varNode->order;
        if (k >= 0 && static_cast<size_t>(k) < Hv.size())
            Hv[k] = varNode->grad_dot;
    }
    return Hv;
}

std::vector<std::vector<double>>
ADGraph::computeHessianDense(const ADNodePtr &y) {
    const size_t n = nodeVariables.size();
    std::vector<std::vector<double>> H(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        std::vector<double> e(n, 0.0);
        e[i] = 1.0;
        auto col = hessianVectorProduct(y, e);
        for (size_t r = 0; r < n; ++r)
            H[r][i] = col[r];
    }
    return H;
}

void ADGraph::adoptSubgraph(const ADNodePtr &root) {
    if (!root)
        return;
    std::unordered_set<ADNodePtr> visited;
    std::function<void(const ADNodePtr &)> dfs = [&](const ADNodePtr &n) {
        if (!n || visited.count(n))
            return;
        visited.insert(n);
        addNode(n);
        if (n->type == Operator::Var && !n->name.empty())
            nodeVariables[n->name] = n;
        for (const auto &in : n->inputs)
            dfs(in);
    };
    dfs(root);
    markDirty_();
}
void ADGraph::updateNodeIndex_() {
    nodeIndex_.clear();
    nodeIndex_.reserve(nodes.size());
    for (const auto &node : nodes) {
        if (node && !node->name.empty()) {
            nodeIndex_[node->name] = node;
        }
    }
}
bool ADGraph::dfsTopoSort_(int nodeId, std::vector<NodeColor> &colors,
                           std::vector<ADNode *> &topoOrder,
                           const std::vector<std::vector<int>> &adjList) {
    colors[nodeId] = NodeColor::GRAY;

    // Visit all neighbors
    for (int neighbor : adjList[nodeId]) {
        if (colors[neighbor] == NodeColor::GRAY) {
            // Back edge found - cycle detected!
            return false;
        }
        if (colors[neighbor] == NodeColor::WHITE &&
            !dfsTopoSort_(neighbor, colors, topoOrder, adjList)) {
            return false;
        }
    }

    colors[nodeId] = NodeColor::BLACK;
    topoOrder.push_back(nodes[nodeId].get());
    return true;
}
bool ADGraph::hasCycles() const {
    const size_t n = nodes.size();
    std::unordered_map<ADNode *, int> tempIdMap;
    for (size_t i = 0; i < n; ++i) {
        tempIdMap[nodes[i].get()] = i;
    }

    std::vector<NodeColor> colors(n, NodeColor::WHITE);

    std::function<bool(int)> dfsCheck = [&](int nodeId) -> bool {
        colors[nodeId] = NodeColor::GRAY;

        for (const auto &input : nodes[nodeId]->inputs) {
            if (!input)
                continue;
            auto it = tempIdMap.find(input.get());
            if (it == tempIdMap.end())
                continue;

            int inputId = it->second;
            if (colors[inputId] == NodeColor::GRAY) {
                return true; // Cycle found
            }
            if (colors[inputId] == NodeColor::WHITE && dfsCheck(inputId)) {
                return true;
            }
        }

        colors[nodeId] = NodeColor::BLACK;
        return false;
    };

    for (size_t i = 0; i < n; ++i) {
        if (colors[i] == NodeColor::WHITE && dfsCheck(i)) {
            return true;
        }
    }
    return false;
}
bool ADGraph::buildTopoOrderDFS_() {
    const size_t n = nodes.size();

    // Build adjacency list
    std::vector<std::vector<int>> adjList(n);
    for (size_t i = 0; i < n; ++i) {
        ADNode *node = nodes[i].get();
        for (const auto &input : node->inputs) {
            if (!input)
                continue;
            auto it = cache_.id_of.find(input.get());
            if (it != cache_.id_of.end()) {
                adjList[it->second].push_back(i);
            }
        }
    }

    // DFS with cycle detection
    std::vector<NodeColor> colors(n, NodeColor::WHITE);
    cache_.topo.clear();
    cache_.topo.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        if (colors[i] == NodeColor::WHITE) {
            if (!dfsTopoSort_(i, colors, cache_.topo, adjList)) {
                std::cerr << "Error: Cycle detected in computation graph!"
                          << std::endl;
                return false;
            }
        }
    }

    // Reverse to get correct topological order
    std::reverse(cache_.topo.begin(), cache_.topo.end());
    return true;
}
