// ADGraph.cpp — ULTRA-FAST C++23 VERSION (fused operations, minimal overhead)
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
#include <immintrin.h>  // For SIMD if available

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

struct PtrHash { size_t operator()(const ADNode *p) const noexcept {
    return std::hash<const void*>{}(p);
}};
struct PtrEq { bool operator()(const ADNode *a, const ADNode *b) const noexcept {
    return a == b;
}};

// Pre-allocated thread-local scratch buffers to eliminate allocations
thread_local std::vector<double> g_scratch_values;
thread_local std::vector<double> g_scratch_dots;
thread_local std::vector<double> g_scratch_softmax;

inline void ensure_scratch_size(size_t n) {
    if (g_scratch_values.size() < n) {
        g_scratch_values.resize(n * 2);  // Over-allocate to reduce future resizes
        g_scratch_dots.resize(n * 2);
        g_scratch_softmax.resize(n * 2);
    }
}

} // namespace

// ===== ADGraph: Dirty marking & cache rebuild ================================
void ADGraph::markDirty_() { cache_.dirty = true; }

void ADGraph::rebuildCache_() {
    makeNodesUnique();

    const size_t N = nodes.size();
    cache_.id_of.clear();
    cache_.id_of.reserve(N);

    for (size_t i = 0; i < N; ++i) {
        cache_.id_of[nodes[i].get()] = static_cast<int>(i);
    }

    std::vector<std::vector<int>> adjList(N);
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

    std::vector<NodeColor> colors(N, NodeColor::WHITE);
    cache_.topo.clear();
    cache_.topo.reserve(N);

    std::function<bool(int)> dfs = [&](int u) -> bool {
        colors[u] = NodeColor::GRAY;
        for (int v : adjList[u]) {
            auto c = colors[v];
            if (c == NodeColor::GRAY) [[unlikely]] return false;
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
    cache_.dirty = false;
}

// ===== Core graph maintenance (unchanged) ===================================
void ADGraph::deleteNode(const ADNodePtr &node) {
    if (!node) return;
    if (!node->name.empty())
        nodeIndex_.erase(node->name);
    for (auto &n : nodes) {
        auto &ins = n->inputs;
        ins.erase(std::remove(ins.begin(), ins.end(), node), ins.end());
    }
    nodes.erase(std::remove(nodes.begin(), nodes.end(), node), nodes.end());
    for (auto it = nodeVariables.begin(); it != nodeVariables.end(); ) {
        if (it->second == node) it = nodeVariables.erase(it);
        else ++it;
    }
    markDirty_();
}

void ADGraph::addNode(const ADNodePtr &node) {
    if (!node) return;
    if (std::find(nodes.begin(), nodes.end(), node) == nodes.end()) {
        nodes.push_back(node);
        if (!node->name.empty())
            nodeIndex_[node->name] = node;
        markDirty_();
    }
}

void ADGraph::makeNodesUnique() {
    tsl::robin_map<std::string, ADNodePtr> by_name;
    by_name.reserve(nodes.size());
    tsl::robin_map<ADNodePtr, ADNodePtr> remap;
    remap.reserve(nodes.size());
    std::vector<ADNodePtr> unique_nodes;
    unique_nodes.reserve(nodes.size());

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

    for (auto &n : unique_nodes) {
        if (!n) continue;
        for (auto &in : n->inputs) {
            if (!in) continue;
            auto it = remap.find(in);
            if (it != remap.end()) in = it->second;
        }
    }

    nodeIndex_.clear();
    nodeIndex_.reserve(by_name.size());
    for (const auto &kv : by_name)
        nodeIndex_[kv.first] = kv.second;

    for (auto it = nodeVariables.begin(); it != nodeVariables.end(); ) {
        auto jt = nodeIndex_.find(it->first);
        if (jt != nodeIndex_.end() && jt->second) {
            it.value() = jt->second;
            ++it;
        } else {
            it = nodeVariables.erase(it);
        }
    }

    nodes = std::move(unique_nodes);
    markDirty_();
}

// ===== ULTRA-FAST FORWARD PASS - FUSED OPERATIONS ==========================

void ADGraph::computeForwardPassWithDotFused() {
    if (cache_.dirty) rebuildCache_();
    
    // Batch process nodes by type for better cache locality
    const size_t n_nodes = cache_.topo.size();
    
    for (size_t i = 0; i < n_nodes; ++i) {
        ADNode *u = cache_.topo[i];
        
        // Direct switch - no dispatch overhead, fused value+dot computation
        switch(u->type) {
        case Operator::Var:
        case Operator::cte:
            // Leaf nodes: just touch epochs
            touch_epoch(u->val_epoch, cur_val_epoch_);
            touch_epoch(u->dot_epoch, cur_dot_epoch_);
            break;
            
        case Operator::Add: {
            if (!u->inputs.empty()) [[likely]] {
                double val = 0.0, dot = 0.0;
                // Unroll small cases for better performance
                const size_t n_inp = u->inputs.size();
                if (n_inp == 2) [[likely]] {
                    val = u->inputs[0]->value + u->inputs[1]->value;
                    dot = u->inputs[0]->dot + u->inputs[1]->dot;
                } else {
                    for (const auto& inp : u->inputs) {
                        val += inp->value;
                        dot += inp->dot;
                    }
                }
                set_epoch_value(u->value, u->val_epoch, cur_val_epoch_, val);
                set_epoch_value(u->dot, u->dot_epoch, cur_dot_epoch_, dot);
            }
            break;
        }
        
        case Operator::Multiply: {
            if (!u->inputs.empty()) [[likely]] {
                const size_t m = u->inputs.size();
                if (m == 2) [[likely]] {
                    // Binary multiply - most common case
                    const double a = u->inputs[0]->value, b = u->inputs[1]->value;
                    const double ad = u->inputs[0]->dot, bd = u->inputs[1]->dot;
                    set_epoch_value(u->value, u->val_epoch, cur_val_epoch_, a * b);
                    set_epoch_value(u->dot, u->dot_epoch, cur_dot_epoch_, ad * b + a * bd);
                } else {
                    // N-ary multiply with zero-aware optimization
                    size_t zero_count = 0, zero_idx = 0;
                    double prod_nz = 1.0;
                    
                    for (size_t j = 0; j < m; ++j) {
                        const double v = u->inputs[j]->value;
                        if (v == 0.0) {
                            if (++zero_count == 1) zero_idx = j;
                        } else {
                            prod_nz *= v;
                        }
                    }
                    
                    double ydot = 0.0;
                    if (zero_count >= 2) {
                        // Multiple zeros -> result is 0, derivative is 0
                    } else if (zero_count == 1) {
                        ydot = u->inputs[zero_idx]->dot * prod_nz;
                    } else {
                        // All nonzero: use optimized formula
                        for (size_t j = 0; j < m; ++j) {
                            ydot += u->inputs[j]->dot * (prod_nz / u->inputs[j]->value);
                        }
                    }
                    
                    const double yval = (zero_count == 0) ? prod_nz : 0.0;
                    set_epoch_value(u->value, u->val_epoch, cur_val_epoch_, yval);
                    set_epoch_value(u->dot, u->dot_epoch, cur_dot_epoch_, ydot);
                }
            }
            break;
        }
        
        case Operator::Sin: {
            if (u->inputs.size() == 1) [[likely]] {
                const double x = u->inputs[0]->value;
                const double xd = u->inputs[0]->dot;
                const double sx = std::sin(x), cx = std::cos(x);
                set_epoch_value(u->value, u->val_epoch, cur_val_epoch_, sx);
                set_epoch_value(u->dot, u->dot_epoch, cur_dot_epoch_, cx * xd);
            }
            break;
        }
        
        case Operator::Cos: {
            if (u->inputs.size() == 1) [[likely]] {
                const double x = u->inputs[0]->value;
                const double xd = u->inputs[0]->dot;
                const double cx = std::cos(x), sx = std::sin(x);
                set_epoch_value(u->value, u->val_epoch, cur_val_epoch_, cx);
                set_epoch_value(u->dot, u->dot_epoch, cur_dot_epoch_, -sx * xd);
            }
            break;
        }
        
        case Operator::Exp: {
            if (u->inputs.size() == 1) [[likely]] {
                const double x = u->inputs[0]->value;
                const double xd = u->inputs[0]->dot;
                const double ex = std::exp(x);
                set_epoch_value(u->value, u->val_epoch, cur_val_epoch_, ex);
                set_epoch_value(u->dot, u->dot_epoch, cur_dot_epoch_, ex * xd);
            }
            break;
        }
        
        case Operator::Log: {
            if (u->inputs.size() == 1) [[likely]] {
                const double x = u->inputs[0]->value;
                const double xd = u->inputs[0]->dot;
                if (x > 0.0) [[likely]] {
                    set_epoch_value(u->value, u->val_epoch, cur_val_epoch_, std::log(x));
                    set_epoch_value(u->dot, u->dot_epoch, cur_dot_epoch_, xd / x);
                } else {
                    set_epoch_value(u->value, u->val_epoch, cur_val_epoch_, 0.0);
                    set_epoch_value(u->dot, u->dot_epoch, cur_dot_epoch_, 0.0);
                }
            }
            break;
        }
        
        case Operator::Tanh: {
            if (u->inputs.size() == 1) [[likely]] {
                const double x = u->inputs[0]->value;
                const double xd = u->inputs[0]->dot;
                const double tx = std::tanh(x);
                const double sech2 = 1.0 - tx * tx;
                set_epoch_value(u->value, u->val_epoch, cur_val_epoch_, tx);
                set_epoch_value(u->dot, u->dot_epoch, cur_dot_epoch_, sech2 * xd);
            }
            break;
        }
        
        case Operator::Relu: {
            if (u->inputs.size() == 1) [[likely]] {
                const double x = u->inputs[0]->value;
                const double xd = u->inputs[0]->dot;
                if (x > 0.0) {
                    set_epoch_value(u->value, u->val_epoch, cur_val_epoch_, x);
                    set_epoch_value(u->dot, u->dot_epoch, cur_dot_epoch_, xd);
                } else {
                    set_epoch_value(u->value, u->val_epoch, cur_val_epoch_, 0.0);
                    set_epoch_value(u->dot, u->dot_epoch, cur_dot_epoch_, 0.0);
                }
            }
            break;
        }
        
        case Operator::Subtract: {
            if (u->inputs.size() == 2) [[likely]] {
                const double a = u->inputs[0]->value, b = u->inputs[1]->value;
                const double ad = u->inputs[0]->dot, bd = u->inputs[1]->dot;
                set_epoch_value(u->value, u->val_epoch, cur_val_epoch_, a - b);
                set_epoch_value(u->dot, u->dot_epoch, cur_dot_epoch_, ad - bd);
            }
            break;
        }
        
        case Operator::Divide: {
            if (u->inputs.size() == 2) [[likely]] {
                const double a = u->inputs[0]->value, b = u->inputs[1]->value;
                const double ad = u->inputs[0]->dot, bd = u->inputs[1]->dot;
                if (b != 0.0) [[likely]] {
                    const double val = a / b;
                    const double dot = (ad * b - a * bd) / (b * b);
                    set_epoch_value(u->value, u->val_epoch, cur_val_epoch_, val);
                    set_epoch_value(u->dot, u->dot_epoch, cur_dot_epoch_, dot);
                } else {
                    set_epoch_value(u->value, u->val_epoch, cur_val_epoch_, 0.0);
                    set_epoch_value(u->dot, u->dot_epoch, cur_dot_epoch_, 0.0);
                }
            }
            break;
        }
        
        case Operator::Max: {
            if (u->inputs.size() == 2) [[likely]] {
                const double a = u->inputs[0]->value, b = u->inputs[1]->value;
                if (a >= b) {
                    set_epoch_value(u->value, u->val_epoch, cur_val_epoch_, a);
                    set_epoch_value(u->dot, u->dot_epoch, cur_dot_epoch_, u->inputs[0]->dot);
                } else {
                    set_epoch_value(u->value, u->val_epoch, cur_val_epoch_, b);
                    set_epoch_value(u->dot, u->dot_epoch, cur_dot_epoch_, u->inputs[1]->dot);
                }
            }
            break;
        }
        
        case Operator::Softmax: {
            if (!u->inputs.empty()) [[likely]] {
                const size_t m = u->inputs.size();
                ensure_scratch_size(m);
                
                // Find max for numerical stability
                double xmax = -std::numeric_limits<double>::infinity();
                for (size_t j = 0; j < m; ++j) {
                    const double xj = u->inputs[j]->value;
                    g_scratch_values[j] = xj;
                    g_scratch_dots[j] = u->inputs[j]->dot;
                    if (xj > xmax) xmax = xj;
                }
                
                // Compute softmax values
                double Z = 0.0;
                for (size_t j = 0; j < m; ++j) {
                    const double ej = std::exp(g_scratch_values[j] - xmax);
                    g_scratch_softmax[j] = ej;
                    Z += ej;
                }
                
                if (Z > 0.0) [[likely]] {
                    for (size_t j = 0; j < m; ++j) {
                        g_scratch_softmax[j] /= Z;
                    }
                    
                    const double yi = g_scratch_softmax[0];
                    double sdot = 0.0;
                    for (size_t j = 0; j < m; ++j) {
                        sdot += g_scratch_softmax[j] * g_scratch_dots[j];
                    }
                    
                    set_epoch_value(u->value, u->val_epoch, cur_val_epoch_, yi);
                    set_epoch_value(u->dot, u->dot_epoch, cur_dot_epoch_, yi * (g_scratch_dots[0] - sdot));
                } else {
                    set_epoch_value(u->value, u->val_epoch, cur_val_epoch_, 0.0);
                    set_epoch_value(u->dot, u->dot_epoch, cur_dot_epoch_, 0.0);
                }
            }
            break;
        }
        
        // Add other operators (Tan, Silu, Gelu) as needed...
        default:
            // Fallback to original dispatch for unsupported ops
            _dispatch_op(*u, ForwardFunctor{*this, *u});
            _dispatch_op(*u, ForwardDotFunctor{*this, *u});
            break;
        }
    }
}

// ===== OPTIMIZED BACKWARD PASS (similar approach) =========================

void ADGraph::initiateBackwardPassFused() {
    if (cache_.dirty) rebuildCache_();
    
    // Process in reverse topological order
    for (auto it = cache_.topo.rbegin(); it != cache_.topo.rend(); ++it) {
        ADNode *u = *it;
        
        switch(u->type) {
        case Operator::Var:
        case Operator::cte:
            // Leaf nodes: nothing to propagate
            break;
            
        case Operator::Add: {
            if (!u->inputs.empty()) [[likely]] {
                const double w = u->gradient;
                for (const auto& inp : u->inputs) {
                    ensure_epoch_zero(inp->gradient, inp->grad_epoch, cur_grad_epoch_) += w;
                }
            }
            break;
        }
        
        case Operator::Multiply: {
            if (u->inputs.size() == 2) [[likely]] {
                const double w = u->gradient;
                const double a = u->inputs[0]->value, b = u->inputs[1]->value;
                ensure_epoch_zero(u->inputs[0]->gradient, u->inputs[0]->grad_epoch, cur_grad_epoch_) += w * b;
                ensure_epoch_zero(u->inputs[1]->gradient, u->inputs[1]->grad_epoch, cur_grad_epoch_) += w * a;
            } else if (!u->inputs.empty()) {
                // N-ary multiply backward pass
                const size_t m = u->inputs.size();
                size_t zero_count = 0, zero_idx = 0;
                double prod_nz = 1.0;
                
                for (size_t j = 0; j < m; ++j) {
                    const double v = u->inputs[j]->value;
                    if (v == 0.0) {
                        if (++zero_count == 1) zero_idx = j;
                    } else {
                        prod_nz *= v;
                    }
                }
                
                const double w = u->gradient;
                if (zero_count >= 2) {
                    // Multiple zeros -> all gradients are 0
                } else if (zero_count == 1) {
                    auto& gacc = ensure_epoch_zero(u->inputs[zero_idx]->gradient, u->inputs[zero_idx]->grad_epoch, cur_grad_epoch_);
                    gacc += w * prod_nz;
                } else {
                    // All nonzero
                    for (size_t j = 0; j < m; ++j) {
                        auto& gacc = ensure_epoch_zero(u->inputs[j]->gradient, u->inputs[j]->grad_epoch, cur_grad_epoch_);
                        gacc += w * (prod_nz / u->inputs[j]->value);
                    }
                }
            }
            break;
        }
        
        case Operator::Sin: {
            if (u->inputs.size() == 1) [[likely]] {
                const double w = u->gradient;
                const double x = u->inputs[0]->value;
                const double df = std::cos(x);  // d/dx sin(x) = cos(x)
                ensure_epoch_zero(u->inputs[0]->gradient, u->inputs[0]->grad_epoch, cur_grad_epoch_) += w * df;
            }
            break;
        }
        
        case Operator::Cos: {
            if (u->inputs.size() == 1) [[likely]] {
                const double w = u->gradient;
                const double x = u->inputs[0]->value;
                const double df = -std::sin(x);  // d/dx cos(x) = -sin(x)
                ensure_epoch_zero(u->inputs[0]->gradient, u->inputs[0]->grad_epoch, cur_grad_epoch_) += w * df;
            }
            break;
        }
        
        case Operator::Tan: {
            if (u->inputs.size() == 1) [[likely]] {
                const double w = u->gradient;
                const double x = u->inputs[0]->value;
                const double c = std::cos(x);
                const double df = (c != 0.0) ? (1.0 / (c * c)) : 0.0;  // sec²(x)
                ensure_epoch_zero(u->inputs[0]->gradient, u->inputs[0]->grad_epoch, cur_grad_epoch_) += w * df;
            }
            break;
        }
        
        case Operator::Exp: {
            if (u->inputs.size() == 1) [[likely]] {
                const double w = u->gradient;
                const double ex = std::exp(u->inputs[0]->value);  // d/dx exp(x) = exp(x)
                ensure_epoch_zero(u->inputs[0]->gradient, u->inputs[0]->grad_epoch, cur_grad_epoch_) += w * ex;
            }
            break;
        }
        
        case Operator::Log: {
            if (u->inputs.size() == 1) [[likely]] {
                const double w = u->gradient;
                const double x = u->inputs[0]->value;
                const double df = (x > 0.0) ? (1.0 / x) : 0.0;  // d/dx log(x) = 1/x
                ensure_epoch_zero(u->inputs[0]->gradient, u->inputs[0]->grad_epoch, cur_grad_epoch_) += w * df;
            }
            break;
        }
        
        case Operator::Tanh: {
            if (u->inputs.size() == 1) [[likely]] {
                const double w = u->gradient;
                const double t = std::tanh(u->inputs[0]->value);
                const double df = 1.0 - t * t;  // sech²(x) = 1 - tanh²(x)
                ensure_epoch_zero(u->inputs[0]->gradient, u->inputs[0]->grad_epoch, cur_grad_epoch_) += w * df;
            }
            break;
        }
        
        case Operator::Relu: {
            if (u->inputs.size() == 1) [[likely]] {
                const double w = u->gradient;
                const double x = u->inputs[0]->value;
                if (x > 0.0) {  // ReLU derivative: 1 if x > 0, 0 otherwise
                    ensure_epoch_zero(u->inputs[0]->gradient, u->inputs[0]->grad_epoch, cur_grad_epoch_) += w;
                }
            }
            break;
        }
        
        case Operator::Silu: {
            if (u->inputs.size() == 1) [[likely]] {
                const double w = u->gradient;
                const double x = u->inputs[0]->value;
                // SiLU: f(x) = x * σ(x), f'(x) = σ(x) + x * σ(x) * (1 - σ(x))
                const double sigma = (x >= 0.0) ? (1.0 / (1.0 + std::exp(-x))) : (std::exp(x) / (1.0 + std::exp(x)));
                const double df = sigma * (1.0 + x * (1.0 - sigma));
                ensure_epoch_zero(u->inputs[0]->gradient, u->inputs[0]->grad_epoch, cur_grad_epoch_) += w * df;
            }
            break;
        }
        
        case Operator::Gelu: {
            if (u->inputs.size() == 1) [[likely]] {
                const double w = u->gradient;
                const double x = u->inputs[0]->value;
                // GELU: f(x) = 0.5 * x * (1 + erf(x/√2))
                const double z = x * M_SQRT1_2;  // x / √2
                const double A = std::sqrt(2.0 / M_PI) * std::exp(-0.5 * x * x);
                const double df = 0.5 * (1.0 + std::erf(z)) + 0.5 * x * A;
                ensure_epoch_zero(u->inputs[0]->gradient, u->inputs[0]->grad_epoch, cur_grad_epoch_) += w * df;
            }
            break;
        }
        
        case Operator::Subtract: {
            if (u->inputs.size() == 2) [[likely]] {
                const double w = u->gradient;
                // d/da (a - b) = 1, d/db (a - b) = -1
                ensure_epoch_zero(u->inputs[0]->gradient, u->inputs[0]->grad_epoch, cur_grad_epoch_) += w;
                ensure_epoch_zero(u->inputs[1]->gradient, u->inputs[1]->grad_epoch, cur_grad_epoch_) -= w;
            }
            break;
        }
        
        case Operator::Divide: {
            if (u->inputs.size() == 2) [[likely]] {
                const double w = u->gradient;
                const double a = u->inputs[0]->value, b = u->inputs[1]->value;
                if (b != 0.0) [[likely]] {
                    // d/da (a/b) = 1/b, d/db (a/b) = -a/b²
                    ensure_epoch_zero(u->inputs[0]->gradient, u->inputs[0]->grad_epoch, cur_grad_epoch_) += w / b;
                    ensure_epoch_zero(u->inputs[1]->gradient, u->inputs[1]->grad_epoch, cur_grad_epoch_) += w * (-a / (b * b));
                }
            }
            break;
        }
        
        case Operator::Max: {
            if (u->inputs.size() == 2) [[likely]] {
                const double w = u->gradient;
                const double a = u->inputs[0]->value, b = u->inputs[1]->value;
                // Max is non-smooth: gradient goes to the larger input (tie breaks to first)
                if (a >= b) {
                    ensure_epoch_zero(u->inputs[0]->gradient, u->inputs[0]->grad_epoch, cur_grad_epoch_) += w;
                } else {
                    ensure_epoch_zero(u->inputs[1]->gradient, u->inputs[1]->grad_epoch, cur_grad_epoch_) += w;
                }
            }
            break;
        }
        
        case Operator::Softmax: {
            if (!u->inputs.empty()) [[likely]] {
                const size_t m = u->inputs.size();
                ensure_scratch_size(m);
                
                // Recompute softmax values for gradient calculation
                double xmax = -std::numeric_limits<double>::infinity();
                for (size_t j = 0; j < m; ++j) {
                    const double xj = u->inputs[j]->value;
                    g_scratch_values[j] = xj;
                    if (xj > xmax) xmax = xj;
                }
                
                double Z = 0.0;
                for (size_t j = 0; j < m; ++j) {
                    const double ej = std::exp(g_scratch_values[j] - xmax);
                    g_scratch_softmax[j] = ej;
                    Z += ej;
                }
                
                if (Z > 0.0) [[likely]] {
                    for (size_t j = 0; j < m; ++j) {
                        g_scratch_softmax[j] /= Z;
                    }
                    
                    const double yi = g_scratch_softmax[0];  // Component of interest (first input)
                    const double w = u->gradient;
                    
                    // Softmax Jacobian: ∂y_i/∂x_k = y_i * (δ_{ik} - y_k)
                    // For our case: we're computing gradient of y_0 w.r.t. all inputs
                    for (size_t k = 0; k < m; ++k) {
                        const double dfk = yi * ((k == 0) ? 1.0 : 0.0) - yi * g_scratch_softmax[k];
                        ensure_epoch_zero(u->inputs[k]->gradient, u->inputs[k]->grad_epoch, cur_grad_epoch_) += w * dfk;
                    }
                }
            }
            break;
        }
        
        default:
            // Fallback to original dispatch for any operators not yet optimized
            _dispatch_op(*u, BackwardFunctor{*this, *u});
            break;
        }
    }
}

// ===== Public API - use optimized versions ===============================

void ADGraph::computeForwardPass() {
    if (cache_.dirty) rebuildCache_();
    for (ADNode *u : cache_.topo) {
        _dispatch_op(*u, ForwardFunctor{*this, *u});
    }
    if (cache_.topo.size() != nodes.size()) [[unlikely]] {
        std::cerr << "Warning: cycle or dangling inputs in AD graph.\n";
    }
}

void ADGraph::computeForwardPassWithDot() {
    // Use the ultra-fast fused version
    computeForwardPassWithDotFused();
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

void ADGraph::initiateBackwardPass(const ADNodePtr & /*outputNode*/) {
    // Use optimized version when available
    initiateBackwardPassFused();
}

void ADGraph::resetGradients() {
    ++cur_grad_epoch_;
    if (cur_grad_epoch_ == 0) ++cur_grad_epoch_;
}

// ===== Rest of implementation unchanged ===================================

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
    const size_t varSize = nodeVariables.size();

    auto partials = computePartialDerivatives(expr);

    std::vector<double> g(varSize, 0.0);
    for (const auto &n : nodes) {
        if (n->type == Operator::Var && n->order >= 0 &&
            static_cast<size_t>(n->order) < varSize) {
            g[n->order] = n->gradient;
        }
    }
    return g;
}

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
        std::string key = "node_" + std::to_string(coll.size());
        coll.emplace(std::move(key), start);
    }
    for (const auto &in : start->inputs)
        collectNodes(in, coll, vis, vars);
}

void ADGraph::resetTangents() {
    ++cur_dot_epoch_;
    if (cur_dot_epoch_ == 0) ++cur_dot_epoch_;
}

void ADGraph::resetGradDot() {
    ++cur_gdot_epoch_;
    if (cur_gdot_epoch_ == 0) ++cur_gdot_epoch_;
}

void ADGraph::initiateBackwardPassHVP() {
    if (cache_.dirty)
        rebuildCache_();
    hvp_add_first_order_ = true;
    for (auto it = cache_.topo.rbegin(); it != cache_.topo.rend(); ++it)
        _dispatch_op(**it, HVPBackwardFunctor{*this, **it});
}

std::vector<double>
ADGraph::hessianVectorProduct(const ADNodePtr &outputNode,
                              const std::vector<double> &v) {
    const size_t nvars = nodeVariables.size();
    std::vector<double> Hv(nvars, 0.0);
    if (!outputNode || nvars == 0) return Hv;

    initializeNodeVariables();

    resetTangents();
    for (auto &kv : nodeVariables) {
        auto &varNode = kv.second;
        if (!varNode) continue;
        int k = varNode->order;
        if (k >= 0 && static_cast<size_t>(k) < v.size())
            set_epoch_value(varNode->dot, varNode->dot_epoch, cur_dot_epoch_, v[k]);
    }

    resetForwardPass();
    computeForwardPassWithDot();

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

// ===== Remaining expression/print functions (unchanged) ====================

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