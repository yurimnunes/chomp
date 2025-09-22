#pragma once
#include "Definitions.h"
#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct Variable;
using VariablePtr = std::shared_ptr<Variable>;

struct ADNode;
using ADNodePtr = std::shared_ptr<ADNode>;

// Input validation helpers
inline bool validate_unary_inputs(const std::vector<ADNodePtr> &inputs,
                                  const std::string &op_name) {
    if (inputs.size() != 1) {
        std::cerr << "Warning: " << op_name
                  << " operation expects exactly 1 input, got " << inputs.size()
                  << std::endl;
        return false;
    }
    if (!inputs[0]) {
        std::cerr << "Error: " << op_name << " operation has null input"
                  << std::endl;
        return false;
    }
    return true;
}

inline bool validate_binary_inputs(const std::vector<ADNodePtr> &inputs,
                                   const std::string &op_name) {
    if (inputs.size() != 2) {
        std::cerr << "Warning: " << op_name
                  << " operation expects exactly 2 inputs, got "
                  << inputs.size() << std::endl;
        return false;
    }
    if (!inputs[0] || !inputs[1]) {
        std::cerr << "Error: " << op_name << " operation has null input(s)"
                  << std::endl;
        return false;
    }
    return true;
}

// Reset-on-epoch-change (for accumulators like gradients)
static inline double &ensure_epoch_zero(double &x, unsigned &e, unsigned cur) {
    if (e != cur) {
        x = 0.0;
        e = cur;
    }
    return x;
}

// Just advance epoch, do NOT modify x (for cached values you want to keep)
static inline void touch_epoch(unsigned &e, unsigned cur) {
    if (e != cur)
        e = cur;
}

// Helper for setting values with epoch update (for computed values)
static inline double &set_epoch_value(double &x, unsigned &e, unsigned cur,
                                      double new_val) {
    x = new_val;
    e = cur;
    return x;
}

#include "../../third_party/robin_map.h"

struct ADNode {
    Operator type = Operator::NA;
    std::string name;
    double value = 0.0;
    double gradient = 0.0;
    double dot = 0.0;      // tangent for forward-mode
    double grad_dot = 0.0; // derivative of gradient along dot
    int order = -1;        // for vectorized gradient output

    // Epochs for lazy reset
    unsigned val_epoch = 0;
    unsigned grad_epoch = 0;
    unsigned dot_epoch = 0;
    unsigned gdot_epoch = 0;

    std::vector<ADNodePtr> inputs;

    void addInput(const ADNodePtr &inputNode);

    // (Legacy) function slots kept for compatibility; unused now.
    std::function<void()> backwardOperation = nullptr;
    std::function<void()> backwardOperationHVP = nullptr;

    // define lb
    double lb = -std::numeric_limits<double>::infinity();
    // define ub
    double ub = std::numeric_limits<double>::infinity();

    NLP nlpType = NLP::NA; // for classifying nonlinear types
};

struct ADGraph;
using ADGraphPtr = std::shared_ptr<ADGraph>;

struct ADGraph {
    bool hvp_add_first_order_ = true; // default for single-lane path

    // ---- Construction / maintenance
    void deleteNode(const ADNodePtr &node);
    void addNode(const ADNodePtr &node);
    void makeNodesUnique();

    // ---- Introspection
    std::string getExpression(const ADNodePtr &node);
    void printTree(const ADNodePtr &node, int depth = 0);
    std::vector<ADNodePtr> findRootNodes() const;

    // ---- Forward
    void computeForwardPass();
    void computeNodeValue(const ADNodePtr &node,
                          std::unordered_set<ADNodePtr> &visited);
    void resetForwardPass();

    // ---- Reverse (gradients)
    void initiateBackwardPass(const ADNodePtr &outputNode);
    void resetGradients();

    // ---- Public Gradient API
    tsl::robin_map<std::string, double>
    computePartialDerivatives(const ADNodePtr &expr);

    ADNodePtr getNode(const std::string &name);
    double getGradientOfVariable(const VariablePtr &var, const ADNodePtr &expr);
    double evaluate(const ADNodePtr &expr);
    void initializeNodeVariables();
    std::vector<double> getGradientVector(const ADNodePtr &expr);

    // ---- Rebuild graph (unique variables)
    std::tuple<ADGraphPtr, tsl::robin_map<std::string, ADNodePtr>>
    rebuildGraphWithUniqueVariables(const ADNodePtr &rootNode);
    void collectNodes(const ADNodePtr &start,
                      tsl::robin_map<std::string, ADNodePtr> &coll,
                      std::unordered_set<ADNodePtr> &vis,
                      tsl::robin_map<std::string, ADNodePtr> &vars);

    // ---- HVP (forward-over-reverse)
    void resetTangents();
    void resetGradDot();
    void computeForwardPassWithDot();
    void initiateBackwardPassHVP();
    void initiateBackwardPassFused();
    std::vector<double> hessianVectorProduct(const ADNodePtr &outputNode,
                                             const std::vector<double> &v);
    std::vector<std::vector<double>> computeHessianDense(const ADNodePtr &y);

    // ---- Adopt external subgraph (registers variables)
    void adoptSubgraph(const ADNodePtr &root);

    // ---- Public fields (existing)
    std::vector<ADNodePtr> nodes;
    tsl::robin_map<std::string, ADNodePtr> nodeVariables;

    // ===== Epoch counters for lazy reset =====
    unsigned cur_val_epoch_ = 1;
    unsigned cur_grad_epoch_ = 1;
    unsigned cur_dot_epoch_ = 1;
    unsigned cur_gdot_epoch_ = 1;

    void computeForwardPassWithDotFused() ;

private:
    // ===== Cached topology (rebuilt only when graph mutates) =====
    struct Cache {
        std::vector<ADNode *> topo;             // forward order
        std::vector<std::vector<ADNode *>> out; // children (by index)
        // std::unordered_map<ADNode *, int> id_of; // raw* -> id
        tsl::robin_map<ADNode *, int> id_of; // or as a member
    std::vector<std::vector<ADNode*>> levels; // NEW: levelized topological layers

        bool dirty = true;
    } cache_;

    void fused_forward_dot_kernel_(ADNode* u);
    void fused_backward_kernel_(ADNode* u);

    void markDirty_();
    void rebuildCache_();

    // Additional optimization: check for cycles without full rebuild
    bool hasCycles() const;
    // Add a comprehensive node name index
    std::unordered_map<std::string, ADNodePtr> nodeIndex_;

    void updateNodeIndex_();

    enum class NodeColor { WHITE, GRAY, BLACK };

    // DFS-based topological sort with cycle detection
    bool dfsTopoSort_(int nodeId, std::vector<NodeColor> &colors,
                      std::vector<ADNode *> &topoOrder,
                      const std::vector<std::vector<int>> &adjList);

    bool buildTopoOrderDFS_();

    void executeUnaryOp(ADNode *node, std::function<double(double)> forward_fn,
                        std::function<double(double)> backward_fn = nullptr) {
        if (!validate_unary_inputs(node->inputs, "unary op"))
            return;
        auto &input = node->inputs[0];
        set_epoch_value(node->value, node->val_epoch, cur_val_epoch_,
                        forward_fn(input->value));
    }

    void executeUnaryBackward(ADNode *node,
                              std::function<double(double)> derivative_fn) {
        if (!validate_unary_inputs(node->inputs, "unary backward"))
            return;
        auto &input = node->inputs[0];
        ensure_epoch_zero(input->gradient, input->grad_epoch,
                          cur_grad_epoch_) +=
            node->gradient * derivative_fn(input->value);
    }
};

// Prefer the first non-null graph, else create one.
inline ADGraphPtr pick_graph(const ADGraphPtr &a,
                             const ADGraphPtr &b = nullptr) {
    if (a)
        return a;
    if (b)
        return b;
    return std::make_shared<ADGraph>();
}
