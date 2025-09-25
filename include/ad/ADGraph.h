#pragma once
#include "Definitions.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../third_party/robin_map.h"
#include "../../third_party/robin_set.h"

// ============================== Forward decls ================================
struct Variable;
using VariablePtr = std::shared_ptr<Variable>;

struct ADNode;
using ADNodePtr = std::shared_ptr<ADNode>;

struct ADGraph;
using ADGraphPtr = std::shared_ptr<ADGraph>;

// ============================ Validation helpers =============================
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

// ============================== Epoch helpers ================================
static inline double &ensure_epoch_zero(double &x, unsigned &e, unsigned cur) {
    if (e != cur) {
        x = 0.0;
        e = cur;
    }
    return x;
}
static inline void touch_epoch(unsigned &e, unsigned cur) {
    if (e != cur)
        e = cur;
}
static inline double &set_epoch_value(double &x, unsigned &e, unsigned cur,
                                      double new_val) {
    x = new_val;
    e = cur;
    return x;
}

// ================================= ADNode ====================================
struct ADNode {
    Operator type = Operator::NA;
    std::string name;

    // AoS “legacy” scalar slots — kept for compatibility
    double value = 0.0;
    double gradient = 0.0;
    double dot = 0.0;      // single-lane legacy view (lane 0)
    double grad_dot = 0.0; // single-lane legacy view (lane 0)
    int order = -1;

    // Epochs (lazy reset)
    unsigned val_epoch = 0;
    unsigned grad_epoch = 0;
    unsigned dot_epoch = 0;  // (lane 0)
    unsigned gdot_epoch = 0; // (lane 0)

    std::vector<ADNodePtr> inputs;

    void addInput(const ADNodePtr &inputNode) { inputs.push_back(inputNode); }

    // Legacy compatibility placeholders
    std::function<void()> backwardOperation = nullptr;
    std::function<void()> backwardOperationHVP = nullptr;

    // Optional variable bounds
    double lb = -std::numeric_limits<double>::infinity();
    double ub = std::numeric_limits<double>::infinity();

    NLP nlpType = NLP::NA;
    int id = -1; // unique id in graph (debug)
};

// ============================== Graph cache ==================================
struct GraphCache {
    bool dirty = true;

    // Stable integer ids per node
    tsl::robin_map<const ADNode *, int> id_of; // node* -> id
    std::vector<ADNode *> by_id;               // id   -> node*
    std::vector<int> free_ids;
    int next_id = 0;

    // Adjacency (parents -> children) and reverse
    std::vector<tsl::robin_set<int>> adj;
    std::vector<tsl::robin_set<int>> radj;
    std::vector<int> indeg;

    // Global topological order (ids)
    std::vector<ADNode *> topo;

    // Incremental maintenance
    std::unordered_set<const ADNode *> dirty_nodes;

    // Heuristics
    size_t full_rebuild_threshold_nodes = 10000;
    double full_rebuild_dirty_ratio = 0.25;
};

// ================================= ADGraph ===================================
struct ADGraph {
    // ----------------------- Config / cache / public fields
    // -------------------
    bool hvp_add_first_order_ = true; // keep single-lane behavior
    GraphCache cache_;

    std::vector<ADNodePtr> nodes;
    tsl::robin_map<std::string, ADNodePtr> nodeVariables;

    // Global epochs
    unsigned cur_val_epoch_ = 1;
    unsigned cur_grad_epoch_ = 1;
    unsigned cur_dot_epoch_ = 1;
    unsigned cur_gdot_epoch_ = 1;

    // --------------------------- LANE STORAGE (minimal)
    // ----------------------- Only the hot lane data for HVP batching: dot and
    // gdot
    struct Lanes {
        size_t n = 0, L = 1;           // nodes, lanes
        std::vector<double> dot;       // size n*L, layout: l + L*i
        std::vector<double> gdot;      // size n*L
        std::vector<unsigned> dot_ep;  // size n*L (lazy reset)
        std::vector<unsigned> gdot_ep; // size n*L

        inline size_t idx(size_t i, size_t l) const { return l + L * i; }

        void allocate(size_t n_, size_t L_) {
            n = n_;
            L = std::max<size_t>(1, L_);
            dot.assign(n * L, 0.0);
            gdot.assign(n * L, 0.0);
            dot_ep.assign(n * L, 0);
            gdot_ep.assign(n * L, 0);
        }
        inline size_t base(int uid) const noexcept {
            // uid is guaranteed >= 0 after rebuild; if not, guard or cast
            // carefully
            return static_cast<size_t>(uid) * L;
        }

    } lanes_; // hot multi-RHS buffers

    size_t lanes_count_ = 1;

    // ------------------------ Lane API (inline helpers)
    // -----------------------
    inline void set_num_lanes(size_t k) {
        lanes_count_ = std::max<size_t>(1, k);
        ensureLaneBuffers_();
    }
    inline size_t lanes() const { return lanes_count_; }

    inline void resetTangentsLane(size_t l) {
        if (l >= lanes_.L)
            return;
        const size_t n = lanes_.n;
        const size_t off = lanes_.idx(0, l);
        std::fill_n(lanes_.dot.begin() + off, n, 0.0);
        std::fill_n(lanes_.dot_ep.begin() + off, n, cur_dot_epoch_);
    }
    inline void resetGradDotLane(size_t l) {
        if (l >= lanes_.L)
            return;
        const size_t n = lanes_.n;
        const size_t off = lanes_.idx(0, l);
        std::fill_n(lanes_.gdot.begin() + off, n, 0.0);
        std::fill_n(lanes_.gdot_ep.begin() + off, n, cur_gdot_epoch_);
    }

    void resetTangents() { resetTangentsAll(); }
    void resetGradDot() { resetGradDotAll(); }
    void initiateBackwardPassHVP() { initiateBackwardPassFusedLanes(); }
    inline void resetTangentsAll() {
        std::fill(lanes_.dot.begin(), lanes_.dot.end(), 0.0);
        std::fill(lanes_.dot_ep.begin(), lanes_.dot_ep.end(), cur_dot_epoch_);
    }
    inline void resetGradDotAll() {
        std::fill(lanes_.gdot.begin(), lanes_.gdot.end(), 0.0);
        std::fill(lanes_.gdot_ep.begin(), lanes_.gdot_ep.end(),
                  cur_gdot_epoch_);
    }

    // ----------------------------- Topology APIs
    // ------------------------------
    int ensure_id_(const ADNode *n);
    void release_id_(int id);
    void link_edge_(int pid, int cid);
    void unlink_edge_(int pid, int cid);

    void markDirty_();
    void markDirty_(ADNode *n);
    void rebuildCacheFull_();
    void refreshIncremental_();

    void collectForward_(const std::vector<int> &starts,
                         std::vector<char> &in_aff,
                         std::vector<int> &aff) const;
    void removeFromTopo_(const std::vector<char> &in_aff);
    bool topoForAffected_(const std::vector<int> &affected,
                          std::vector<int> &topo_aff) const;
    void spliceAffected_(const std::vector<int> &affected,
                         const std::vector<int> &topo_aff);

    // ------------------------- Graph construction APIs
    // ------------------------
    void deleteNode(const ADNodePtr &node);
    void addNode(const ADNodePtr &node);
    void makeNodesUnique();
    // In ADGraph.h
    void simplifyExpression(std::vector<ADNodePtr> &outputs);
    inline void simplifyExpression(ADNodePtr &root) {
        std::vector<ADNodePtr> outs{root};
        simplifyExpression(outs);
        root = outs[0];
    }

    ADNodePtr adoptSubgraphAndReturnRoot(const ADNodePtr &root_src);

    void computeForwardPassAndDotLanesTogether();

    void peepholeSimplify_();

    // ---------------------------- Introspection
    // -------------------------------
    std::string getExpression(const ADNodePtr &node);
    void printTree(const ADNodePtr &node, int depth = 0);
    std::vector<ADNodePtr> findRootNodes() const;

    // -------------------------- Scalar forward/grad
    // ---------------------------
    void computeForwardPass(); // AoS compat
    void computeNodeValue(const ADNodePtr &node,
                          std::unordered_set<ADNodePtr> &visited);
    void resetForwardPass();

    void initiateBackwardPass(const ADNodePtr &outputNode);
    void resetGradients();

    // void computeForwardPassWithDot(); // AoS compat
    void initiateBackwardPassFused(); // AoS compat

    // --------------------------- Gradient utilities
    // ---------------------------
    tsl::robin_map<std::string, double>
    computePartialDerivatives(const ADNodePtr &expr);
    ADNodePtr getNode(const std::string &name);
    double getGradientOfVariable(const VariablePtr &var, const ADNodePtr &expr);
    double evaluate(const ADNodePtr &expr);
    void initializeNodeVariables();
    std::vector<double> getGradientVector(const ADNodePtr &expr);

    // ------------------------------ Subgraphs --------------------------------
    std::tuple<ADGraphPtr, tsl::robin_map<std::string, ADNodePtr>>
    rebuildGraphWithUniqueVariables(const ADNodePtr &rootNode);
    void collectNodes(const ADNodePtr &start,
                      tsl::robin_map<std::string, ADNodePtr> &coll,
                      std::unordered_set<ADNodePtr> &vis,
                      tsl::robin_map<std::string, ADNodePtr> &vars);

    void adoptSubgraph(const ADNodePtr &root);

    // =========================== HVP / Hessian API
    // ============================ Legacy single-RHS entry
    std::vector<double> hessianVectorProduct(const ADNodePtr &outputNode,
                                             const std::vector<double> &v);

    // Dense Hessian (slow path)
    std::vector<std::vector<double>> computeHessianDense(const ADNodePtr &y);

    // NEW: multi-RHS entry for batched HVP
    // Layout expected for V_ptr/Y_ptr: lane-inner contiguous for each node:
    // element (node i, lane l) is at address base + (l + L*i).
    void hessianMultiVectorProduct(const ADNodePtr &outputNode,
                                   const double *V_ptr, size_t ldV, // ldV == L
                                   double *Y_ptr, size_t ldY,       // ldY == L
                                   size_t k);

    // ====================== Lane-aware fused passes (impl in .cpp) ===========
    // Forward: assumes primals (AoS) are up-to-date; propagates all lanes' dot.
    void computeForwardPassWithDotLanes();
    // Reverse: seeds grad[root]=1, gdot[root,l]=0; accumulates into gdot lanes.
    void initiateBackwardPassFusedLanes();

private:
    // Allocate lane buffers to current graph size/lanes_count_
    inline void ensureLaneBuffers_() {
        const size_t n =
            cache_.by_id.empty() ? nodes.size() : cache_.by_id.size();
        lanes_.allocate(n, lanes_count_);
    }

    // Node kernels (per-node outer loop, inner loop over lanes)
    void fused_forward_dot_kernel_lanes_(int uid);
    void fused_backward_kernel_lanes_(int uid);

    // Rebuild & utilities
    void rebuildCache_();
    bool hasCycles() const;

    std::unordered_map<std::string, ADNodePtr> nodeIndex_;
    void updateNodeIndex_();

    enum class NodeColor { WHITE, GRAY, BLACK };
    bool dfsTopoSort_(int nodeId, std::vector<NodeColor> &colors,
                      std::vector<ADNode *> &topoOrder,
                      const std::vector<std::vector<int>> &adjList);
    bool buildTopoOrderDFS_();
    void compactNodeIds_();

    // Constant pooling and fast replace helpers
    std::unordered_map<double, ADNodePtr> constant_pool_;
    std::unordered_map<const ADNode *, std::vector<ADNode *>> uses_;
    bool uses_valid_ = false;
    void buildUseListsOnce_();
    void markGraphMutated_();
    void canonicalizeOperands_(ADNode &node);

    // Convenience exec helpers (scalar/AoS — unchanged)
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

public:
    // New simplification methods
    void simplifyGraph();
    bool peepholeSimplify();
    void eliminateDeadCode();
    void constantFolding();
    void algebraicSimplification();

private:
    // Simplification helpers
    ADNodePtr createConstantNode(double value);
    ADNodePtr applyAlgebraicRule(const ADNodePtr &node);
    bool isConstant(const ADNodePtr &node) const;
    bool isZero(const ADNodePtr &node) const;
    bool isOne(const ADNodePtr &node) const;
    void replaceNodeReferences(const ADNodePtr &oldNode,
                               const ADNodePtr &newNode);
    std::vector<ADNodePtr> findNodesWithoutForwardReferences() const;

    // Simplification cache/flags
    mutable bool simplification_needed_ = false;
    size_t last_simplification_size_ = 0;


    // Quantize doubles for constant key stability (same helper you had)
    static inline double qfp(double x, double s = 1e12) {
        return std::nearbyint(x * s) / s;
    }

    inline bool is_commutative_node_(Operator op) const {
        return op == Operator::Add || op == Operator::Multiply;
    }

    struct CSEKey {
        Operator op = Operator::NA;
        uint32_t lanes = 1;
        bool is_cte = false;
        double cval = 0.0; // only if is_cte == true

        // Children identities (by pointer). For commutative ops, this is
        // sorted.
        std::vector<const ADNode *> kids;

        bool operator==(const CSEKey &o) const {
            if (op != o.op || lanes != o.lanes || is_cte != o.is_cte)
                return false;
            if (is_cte)
                return qfp(cval) == qfp(o.cval);
            if (kids.size() != o.kids.size())
                return false;
            for (size_t i = 0; i < kids.size(); ++i)
                if (kids[i] != o.kids[i])
                    return false;
            return true;
        }
    };

    struct CSEKeyHash {
        size_t operator()(const CSEKey &k) const noexcept {
            auto mix = [](size_t a, size_t b) {
                return a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2));
            };
            size_t h = std::hash<int>{}(int(k.op));
            h = mix(h, std::hash<uint32_t>{}(k.lanes));
            h = mix(h, std::hash<bool>{}(k.is_cte));
            if (k.is_cte) {
                uint64_t bits;
                std::memcpy(&bits, &k.cval, sizeof(bits));
                h = mix(h, std::hash<uint64_t>{}(bits));
            } else {
                for (auto *p : k.kids)
                    h = mix(h, std::hash<const void *>{}(p));
            }
            return h;
        }
    };

    // Collect additive terms: Add = sum(+inputs), Sub = first - rest
    void collectAddTerms_(const ADNode *n, int sgn,
                          std::unordered_map<const ADNode *, int> &mult,
                          double &csum) const {
        if (!n)
            return;
        if (n->type == Operator::Add) {
            for (auto &in : n->inputs)
                collectAddTerms_(in.get(), sgn, mult, csum);
            return;
        }
        if (n->type == Operator::Subtract) {
            if (!n->inputs.empty())
                collectAddTerms_(n->inputs[0].get(), sgn, mult, csum);
            for (size_t i = 1; i < n->inputs.size(); ++i)
                collectAddTerms_(n->inputs[i].get(), -sgn, mult, csum);
            return;
        }
        if (n->type == Operator::cte) {
            csum += sgn * n->value;
            return;
        }
        mult[n] += sgn; // record signed multiplicity
    }

    // Collect multiplicative factors: Mul = product, Div = first / rest
    void
    collectMulFactors_(const ADNode *n, int exp,
                       std::unordered_map<const ADNode *, int> &powmap) const {
        if (!n)
            return;
        if (n->type == Operator::Multiply) {
            for (auto &in : n->inputs)
                collectMulFactors_(in.get(), exp, powmap);
            return;
        }
        if (n->type == Operator::Divide) {
            if (!n->inputs.empty())
                collectMulFactors_(n->inputs[0].get(), exp, powmap);
            for (size_t i = 1; i < n->inputs.size(); ++i)
                collectMulFactors_(n->inputs[i].get(), -exp, powmap);
            return;
        }
        // Treat constants as normal factors—safe & simple (no 1/0 surprises)
        powmap[n] += exp;
    }

    CSEKey makeCSEKey_(const ADNode &n) const {
        CSEKey k;
        k.op = n.type;
        k.lanes = static_cast<uint32_t>(this->lanes());

        if (n.type == Operator::cte) {
            k.is_cte = true;
            k.cval = qfp(n.value);
            return k;
        }

        // Exact structure: preserve arity and (for non-commutative) order.
        // For commutative ops (Add/Multiply), sort children by pointer ONLY.
        k.kids.reserve(n.inputs.size());
        for (auto &in : n.inputs)
            k.kids.push_back(in.get());

        if (is_commutative_node_(n.type)) {
            std::sort(k.kids.begin(), k.kids.end(),
                      [](const ADNode *a, const ADNode *b) { return a < b; });
        }
        return k;
    }

    bool cseByKey_();
};

// ============================ pick_graph helper ==============================
inline ADGraphPtr pick_graph(const ADGraphPtr &a,
                             const ADGraphPtr &b = nullptr) {
    if (a)
        return a;
    if (b)
        return b;
    return std::make_shared<ADGraph>();
}
