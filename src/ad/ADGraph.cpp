// ADGraph.cpp — FAST C++23 VERSION (epoch-managed, cache-optimized)
#include "../../include/ad/ADGraph.h"
#include "../../include/ad/Definitions.h"
#include "../../include/ad/Variable.h"

#include "../../include/ad/OpDispatch.h"
#include "../../include/ad/OpTraits.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// Keep alias consistent with headers
using VariablePtr = std::shared_ptr<Variable>;

// ===== ADNode ================================================================
void ADNode::addInput(const ADNodePtr &inputNode) {
    inputs.push_back(inputNode);
}

// ===== Local helpers =========================================================
namespace {

inline bool validate_nary_inputs(const std::vector<ADNodePtr> &inputs,
                                 std::string_view op_name,
                                 size_t min_inputs = 1) {
    if (inputs.size() < min_inputs) [[unlikely]] {
        std::cerr << "Error: " << op_name << " needs at least "
                  << min_inputs << " inputs, got " << inputs.size() << '\n';
        return false;
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!inputs[i]) [[unlikely]] {
            std::cerr << "Error: " << op_name
                      << " has null input at index " << i << '\n';
            return false;
        }
    }
    return true;
}

// Raw-pointer keyed set (faster than shared_ptr hash)
struct PtrHash { size_t operator()(const ADNode *p) const noexcept {
    return std::hash<const void*>{}(p);
}};
struct PtrEq { bool operator()(const ADNode *a, const ADNode *b) const noexcept {
    return a == b;
}};

} // namespace

// ===== ADGraph: Dirty marking & cache rebuild ================================
void ADGraph::markDirty_() { cache_.dirty = true; }

void ADGraph::rebuildCache_() {
    // Enforce uniqueness before building indexes and topo
    makeNodesUnique();

    const size_t N = nodes.size();
    cache_.id_of.clear();
    cache_.id_of.reserve(N);

    // Map: raw* -> id
    for (size_t i = 0; i < N; ++i) {
        cache_.id_of[nodes[i].get()] = static_cast<int>(i);
    }

    // Build adjacency (inputs→this)
    std::vector<std::vector<int>> adjList;
    adjList.resize(N);
    for (size_t i = 0; i < N; ++i) {
        ADNode *node = nodes[i].get();
        const auto &ins = node->inputs;
        for (const auto &in : ins) {
            if (!in) [[unlikely]] continue;
            auto it = cache_.id_of.find(in.get());
            if (it != cache_.id_of.end()) {
                adjList[it->second].push_back(static_cast<int>(i));
            }
        }
    }

    // DFS topo with cycle detection
    std::vector<NodeColor> colors(N, NodeColor::WHITE);
    cache_.topo.clear();
    cache_.topo.reserve(N);

    std::function<bool(int)> dfs = [&](int u) -> bool {
        colors[u] = NodeColor::GRAY;
        for (int v : adjList[u]) {
            auto c = colors[v];
            if (c == NodeColor::GRAY) [[unlikely]] return false; // cycle
            if (c == NodeColor::WHITE) {
                if (!dfs(v)) [[unlikely]] return false;
            }
        }
        colors[u] = NodeColor::BLACK;
        cache_.topo.push_back(nodes[u].get());
        return true;
    };

    for (int i = 0; i < static_cast<int>(N); ++i)
        if (colors[i] == NodeColor::WHITE)
            if (!dfs(i)) {
                std::cerr << "Error: Cycle detected in computation graph!\n";
                cache_.topo.clear();
                cache_.dirty = false;
                return;
            }

    std::reverse(cache_.topo.begin(), cache_.topo.end());

    // // Build output adjacency for reverse sweeps (optional, kept for parity)
    // cache_.out.assign(N, {});
    // for (size_t i = 0; i < N; ++i) {
    //     ADNode *n = nodes[i].get();
    //     for (auto &in : n->inputs) {
    //         if (!in) continue;
    //         auto it = cache_.id_of.find(in.get());
    //         if (it != cache_.id_of.end()) {
    //             cache_.out[it->second].push_back(n);
    //         }
    //     }
    // }

    cache_.dirty = false;
}

// ===== Core graph maintenance ===============================================
void ADGraph::deleteNode(const ADNodePtr &node) {
    if (!node) return;

    // remove from name index
    if (!node->name.empty())
        nodeIndex_.erase(node->name);

    // remove edges pointing to node
    for (auto &n : nodes) {
        auto &ins = n->inputs;
        ins.erase(std::remove(ins.begin(), ins.end(), node), ins.end());
    }
    // remove node itself
    nodes.erase(std::remove(nodes.begin(), nodes.end(), node), nodes.end());

    // scrub from variable map
    for (auto it = nodeVariables.begin(); it != nodeVariables.end(); ) {
        if (it->second == node) it = nodeVariables.erase(it);
        else ++it;
    }

    markDirty_();
}

void ADGraph::addNode(const ADNodePtr &node) {
    if (!node) return;

    // avoid duplicates by identity
    if (std::find(nodes.begin(), nodes.end(), node) == nodes.end()) {
        nodes.push_back(node);
        if (!node->name.empty())
            nodeIndex_[node->name] = node; // last-in wins; dedup fixes later
        markDirty_();
    }
}

void ADGraph::makeNodesUnique() {
    // first-seen name is canonical
    tsl::robin_map<std::string, ADNodePtr> by_name;
    by_name.reserve(nodes.size());

    tsl::robin_map<ADNodePtr, ADNodePtr> remap;
    remap.reserve(nodes.size());

    std::vector<ADNodePtr> unique_nodes;
    unique_nodes.reserve(nodes.size());

    // 1) Pick canonicals and build remap
    for (const auto &n : nodes) {
        if (!n) continue;

        if (!n->name.empty()) {
            auto [it, inserted] = by_name.emplace(n->name, n);
            if (inserted) {
                remap[n] = n;
                unique_nodes.push_back(n);
            } else {
                remap[n] = it->second;
            }
        } else {
            remap[n] = n;
            unique_nodes.push_back(n);
        }
    }

    // 2) Rewire inputs to canonical nodes
    for (auto &n : unique_nodes) {
        if (!n) continue;
        for (auto &in : n->inputs) {
            if (!in) continue;
            auto it = remap.find(in);
            if (it != remap.end()) in = it->second;
        }
    }

    // 3) Rebuild name index to only canonicals
    nodeIndex_.clear();
    nodeIndex_.reserve(by_name.size());
    for (const auto &kv : by_name)
        nodeIndex_[kv.first] = kv.second;

    // 4) Update variable map to canonical nodes (drop orphans)
    for (auto it = nodeVariables.begin(); it != nodeVariables.end(); ) {
        auto jt = nodeIndex_.find(it->first);
        if (jt != nodeIndex_.end() && jt->second) {
            it.value() = jt->second;
            ++it;
        } else {
            it = nodeVariables.erase(it);
        }
    }

    // 5) Replace nodes with unique list
    nodes = std::move(unique_nodes);

    markDirty_();
}

// ===== Introspection / printing =============================================
std::string ADGraph::getExpression(const ADNodePtr &node) {
    if (!node) return {};

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
    case Operator::Log:
    case Operator::Tanh:
    case Operator::Silu:
    case Operator::Gelu:
    case Operator::Relu: {
        if (validate_unary_inputs(node->inputs, op_name(node->type)))
            return std::string(op_name(node->type)) + "(" + expr(node->inputs[0]) + ")";
        return std::string(op_name(node->type)) + "(?)";
    }

    case Operator::Add:
    case Operator::Multiply: {
        const char *sym = (node->type == Operator::Add) ? "+" : "*";
        if (!validate_nary_inputs(node->inputs, op_name(node->type), 1))
            return std::string(op_name(node->type)) + "(?)";
        std::string out = "(";
        for (size_t i = 0; i < node->inputs.size(); ++i) {
            if (i) out += ' ', out += sym, out += ' ';
            out += expr(node->inputs[i]);
        }
        out += ')';
        return out;
    }

    case Operator::Subtract:
    case Operator::Divide:
    case Operator::Max: {
        const char *sym = (node->type == Operator::Subtract) ? "-"
                          : (node->type == Operator::Divide) ? "/" : ",";
        if (!validate_binary_inputs(node->inputs, op_name(node->type)))
            return std::string(op_name(node->type)) + "(?)";
        if (node->type == Operator::Max) {
            return std::string("max(") + expr(node->inputs[0]) + ", " +
                   expr(node->inputs[1]) + ")";
        }
        return "(" + expr(node->inputs[0]) + " " + sym + " " +
               expr(node->inputs[1]) + ")";
    }

    case Operator::Softmax: {
        if (!validate_nary_inputs(node->inputs, op_name(node->type), 1))
            return std::string(op_name(node->type)) + "(?)";
        // Render as softmax(x0 | x0, x1, ..., xN)
        std::string out = std::string(op_name(node->type)) + "(" +
                          expr(node->inputs[0]) + " | ";
        for (size_t i = 0; i < node->inputs.size(); ++i) {
            if (i) out += ", ";
            out += expr(node->inputs[i]);
        }
        out += ")";
        return out;
    }

    default:
        return "Unsupported_Operator(" + std::string(op_name(node->type)) + ")";
    }
}

void ADGraph::printTree(const ADNodePtr &node, int depth) {
    if (!node) return;

    static thread_local std::unordered_set<const ADNode*, PtrHash, PtrEq> seen;

    struct SeenGuard {
        bool top;
        std::unordered_set<const ADNode*, PtrHash, PtrEq> &S;
        SeenGuard(bool t, decltype(S) &s) : top(t), S(s) { if (top) S.clear(); }
        ~SeenGuard() { if (top) S.clear(); }
    } guard(depth == 0, seen);

    auto raw = node.get();
    std::string indent(static_cast<size_t>(depth) * 4, ' ');

    if (seen.count(raw)) {
        std::cout << indent << "|- [visited] " << raw << " (" << node->name << ")\n";
        return;
    }
    seen.insert(raw);

    std::cout << indent << "|- Node: " << raw << '\n'
              << indent << "|    Name: " << node->name << '\n'
              << indent << "|    Expression: " << getExpression(node) << '\n'
              << indent << "|    Value: " << node->value << '\n'
              << indent << "|    Operator: " << op_name(node->type) << '\n'
              << indent << "|    Inputs:";

    if (node->inputs.empty()) {
        std::cout << " None\n\n";
    } else {
        std::cout << '\n';
        for (const auto &in : node->inputs)
            printTree(in, depth + 1);
        std::cout << '\n';
    }
}

std::vector<ADNodePtr> ADGraph::findRootNodes() const {
    std::vector<ADNodePtr> roots;
    roots.reserve(nodes.size());
    std::unordered_set<const ADNode*, PtrHash, PtrEq> inputs;
    inputs.reserve(nodes.size());

    for (const auto &n : nodes)
        for (const auto &in : n->inputs)
            if (in) inputs.insert(in.get());

    for (const auto &n : nodes)
        if (!inputs.count(n.get()))
            roots.push_back(n);

    if (roots.empty())
        std::cerr << "Warning: No root nodes found\n";
    return roots;
}

// ===== Forward ==============================================================

void ADGraph::computeForwardPass() {
    if (cache_.dirty) rebuildCache_();
    for (ADNode *u : cache_.topo) {
        _dispatch_op(*u, ForwardFunctor{*this, *u});
    }
    if (cache_.topo.size() != nodes.size()) [[unlikely]] {
        std::cerr << "Warning: cycle or dangling inputs in AD graph.\n";
    }
}

void ADGraph::computeNodeValue(const ADNodePtr &node,
                               std::unordered_set<ADNodePtr> &visited) {
    if (!node || visited.count(node)) return;
    for (const auto &in : node->inputs)
        computeNodeValue(in, visited);
    _dispatch_op(*node, ForwardFunctor{*this, *node});
    visited.insert(node);
}

void ADGraph::resetForwardPass() {
    ++cur_val_epoch_;
    if (cur_val_epoch_ == 0) ++cur_val_epoch_;
}

// ===== Reverse (gradients) ==================================================
void ADGraph::initiateBackwardPass(const ADNodePtr & /*outputNode*/) {
    if (cache_.dirty) rebuildCache_();
    for (auto it = cache_.topo.rbegin(); it != cache_.topo.rend(); ++it) {
        ADNode *n = *it;
        _dispatch_op(*n, BackwardFunctor{*this, *n});
    }
}

void ADGraph::resetGradients() {
    ++cur_grad_epoch_;
    if (cur_grad_epoch_ == 0) ++cur_grad_epoch_;
}

// ===== Public gradient API ===================================================
tsl::robin_map<std::string, double>
ADGraph::computePartialDerivatives(const ADNodePtr &expr) {
    tsl::robin_map<std::string, double> partials;
    partials.reserve(nodeVariables.size());

    resetGradients();
    resetForwardPass();

    computeForwardPass();

    if (expr)
        set_epoch_value(expr->gradient, expr->grad_epoch, cur_grad_epoch_, 1.0);

    initiateBackwardPass(expr);

    for (auto &kv : nodeVariables)
        if (kv.second)
            partials.emplace(kv.first, kv.second->gradient);

    return partials;
}

ADNodePtr ADGraph::getNode(const std::string &name) {
    auto it = nodeIndex_.find(name);
    return (it != nodeIndex_.end()) ? it->second : nullptr;
}

double ADGraph::getGradientOfVariable(const VariablePtr &var,
                                      const ADNodePtr &expr) {
    auto n = var ? getNode(var->getName()) : nullptr;
    if (!n || !expr) return 0.0;

    resetGradients();
    resetForwardPass();

    computeForwardPass();
    set_epoch_value(expr->gradient, expr->grad_epoch, cur_grad_epoch_, 1.0);
    initiateBackwardPass(expr);

    return n->gradient;
}

// ===== Eval & misc ==========================================================
double ADGraph::evaluate(const ADNodePtr &expr) {
    resetGradients();   // ensures clean grad state if user calls backward later
    resetForwardPass(); // bump value epoch
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
    const size_t varSize = nodeVariables.size();

    auto partials = computePartialDerivatives(expr);

    std::vector<double> g(varSize, 0.0);
    for (const auto &n : nodes) {
        if (n->type == Operator::Var && n->order >= 0 &&
            static_cast<size_t>(n->order) < varSize) {
            // prefer direct gradient on node; fall back to map if needed
            g[n->order] = n->gradient;
        }
    }
    return g;
}

// ===== Rebuild graph (unique variables) =====================================
std::tuple<ADGraphPtr, tsl::robin_map<std::string, ADNodePtr>>
ADGraph::rebuildGraphWithUniqueVariables(const ADNodePtr &rootNode) {
    tsl::robin_map<std::string, ADNodePtr> coll;
    std::unordered_set<ADNodePtr> vis;
    tsl::robin_map<std::string, ADNodePtr> vars;
    collectNodes(rootNode, coll, vis, vars);
    auto newG = std::make_shared<ADGraph>();
    newG->nodes.reserve(coll.size());
    for (const auto &p : coll) newG->addNode(p.second);
    newG->nodeVariables = std::move(vars);
    newG->makeNodesUnique();
    return {newG, newG->nodeVariables};
}

void ADGraph::collectNodes(const ADNodePtr &start,
                           tsl::robin_map<std::string, ADNodePtr> &coll,
                           std::unordered_set<ADNodePtr> &vis,
                           tsl::robin_map<std::string, ADNodePtr> &vars) {
    if (!start || vis.count(start)) return;
    vis.insert(start);

    if (!start->name.empty()) {
        coll.emplace(start->name, start);
        if (start->type == Operator::Var) vars[start->name] = start;
    } else {
        // stable synthetic key
        std::string key = "node_" + std::to_string(coll.size());
        coll.emplace(std::move(key), start);
    }
    for (const auto &in : start->inputs)
        collectNodes(in, coll, vis, vars);
}

// ===== HVP (forward-over-reverse) ===========================================
void ADGraph::resetTangents() {
    ++cur_dot_epoch_;
    if (cur_dot_epoch_ == 0) ++cur_dot_epoch_;
}

void ADGraph::resetGradDot() {
    ++cur_gdot_epoch_;
    if (cur_gdot_epoch_ == 0) ++cur_gdot_epoch_;
}

void ADGraph::computeForwardPassWithDot() {
    if (cache_.dirty) rebuildCache_();
    for (ADNode *u : cache_.topo) {
        _dispatch_op(*u, ForwardFunctor{*this, *u});      // values
        _dispatch_op(*u, ForwardDotFunctor{*this, *u});   // tangents
    }
}

void ADGraph::initiateBackwardPassHVP() {
    if (cache_.dirty) rebuildCache_();
    for (auto it = cache_.topo.rbegin(); it != cache_.topo.rend(); ++it) {
        ADNode *n = *it;
        _dispatch_op(*n, HVPBackwardFunctor{*this, *n});
    }
}

std::vector<double>
ADGraph::hessianVectorProduct(const ADNodePtr &outputNode,
                              const std::vector<double> &v) {
    const size_t nvars = nodeVariables.size();
    std::vector<double> Hv(nvars, 0.0);
    if (!outputNode || nvars == 0) return Hv;

    // Ensure consistent ordering
    initializeNodeVariables();

    // Seed tangents
    resetTangents();
    for (auto &kv : nodeVariables) {
        auto &varNode = kv.second;
        if (!varNode) continue;
        int k = varNode->order;
        if (k >= 0 && static_cast<size_t>(k) < v.size())
            set_epoch_value(varNode->dot, varNode->dot_epoch, cur_dot_epoch_, v[k]);
    }

    // forward (values + dot)
    resetForwardPass();
    computeForwardPassWithDot();

    // reverse (grads + grad_dot)
    resetGradients();
    resetGradDot();
    set_epoch_value(outputNode->gradient,  outputNode->grad_epoch, cur_grad_epoch_, 1.0);
    set_epoch_value(outputNode->grad_dot,  outputNode->gdot_epoch, cur_gdot_epoch_, 0.0);
    initiateBackwardPassHVP();

    for (auto &kv : nodeVariables) {
        auto &varNode = kv.second;
        if (!varNode) continue;
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
        for (size_t r = 0; r < n; ++r) H[r][i] = col[r];
    }
    return H;
}

// ===== Subgraph adoption / indexing =========================================
void ADGraph::adoptSubgraph(const ADNodePtr &root) {
    if (!root) return;
    std::unordered_set<const ADNode*, PtrHash, PtrEq> visited;
    visited.reserve(64);

    std::function<void(const ADNodePtr &)> dfs = [&](const ADNodePtr &n) {
        if (!n) return;
        auto raw = n.get();
        if (visited.count(raw)) return;
        visited.insert(raw);

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
    for (const auto &node : nodes)
        if (node && !node->name.empty())
            nodeIndex_[node->name] = node;
}
