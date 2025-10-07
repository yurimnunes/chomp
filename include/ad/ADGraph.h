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
#include "egraph.h"

// ============================================================================
// Forward declarations
// ============================================================================
struct Variable;
using VariablePtr = std::shared_ptr<Variable>;

struct ADNode;
using ADNodePtr = std::shared_ptr<ADNode>;

struct ADGraph;
using ADGraphPtr = std::shared_ptr<ADGraph>;

// ============================================================================
// Validation helpers (tiny, header-inline)
// ============================================================================
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

// ============================================================================
// Epoch helpers (tiny, header-inline)
// ============================================================================
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

// ============================================================================
// ADNode (POD-ish node record)
// ============================================================================
struct ADNode {
    Operator type = Operator::NA;
    std::string name;

    // Legacy scalar slots
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

    // Legacy placeholders
    std::function<void()> backwardOperation = nullptr;
    std::function<void()> backwardOperationHVP = nullptr;

    // Optional variable bounds
    double lb = -std::numeric_limits<double>::infinity();
    double ub = std::numeric_limits<double>::infinity();

    NLP nlpType = NLP::NA;
    int id = -1; // unique id in graph (debug)
};

// ============================================================================
// GraphCache (ids, adjacency, topo, incremental dirty tracking)
// ============================================================================
struct GraphCache {
    bool dirty = true;

    // Stable integer ids per node
    tsl::robin_map<const ADNode *, int> id_of; // node* -> id
    std::vector<ADNode *> by_id;               // id    -> node*
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

// ============================================================================
// ADGraph
// ============================================================================
struct ADGraph {
    // ------------------------------------------------------------------------
    // Configuration / state
    // ------------------------------------------------------------------------
    bool hvp_add_first_order_ = true; // keep single-lane behavior
    GraphCache cache_;

    std::vector<ADNodePtr> nodes;
    tsl::robin_map<std::string, ADNodePtr> nodeVariables;

    // Global epochs
    unsigned cur_val_epoch_ = 1;
    unsigned cur_grad_epoch_ = 1;
    unsigned cur_dot_epoch_ = 1;
    unsigned cur_gdot_epoch_ = 1;

    // Hot multi-RHS (lane) buffers: dot / gdot
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
            return static_cast<size_t>(uid) * L;
        }
    } lanes_;
    size_t lanes_count_ = 1;

    // ------------------------------------------------------------------------
    // PUBLIC: Construction / mutation
    // ------------------------------------------------------------------------
    ADNodePtr createNode(Operator op, const std::vector<ADNodePtr> &kids) {
        auto n = std::make_shared<ADNode>();
        n->type = op;
        n->inputs = kids;
        n->id = ensure_id_(n.get());
        nodes.push_back(n);
        for (auto &k : kids)
            if (k)
                link_edge_(n->id, k->id);
        markGraphMutated_();
        return n;
    }
    ADNodePtr createConstantNode(double value);

    void addNode(const ADNodePtr &node);
    void deleteNode(const ADNodePtr &node);
    void adoptSubgraph(const ADNodePtr &root);
    ADNodePtr adoptSubgraphAndReturnRoot(const ADNodePtr &root_src);
    void makeNodesUnique();

    // Lane control
    inline void set_num_lanes(size_t k) {
        lanes_count_ = std::max<size_t>(1, k);
        ensureLaneBuffers_();
    }
    inline size_t lanes() const { return lanes_count_; }

    inline void resetTangentsLane(size_t l) {
        if (l >= lanes_.L)
            return;
        const size_t n = lanes_.n, off = lanes_.idx(0, l);
        std::fill_n(lanes_.dot.begin() + off, n, 0.0);
        std::fill_n(lanes_.dot_ep.begin() + off, n, cur_dot_epoch_);
    }
    inline void resetGradDotLane(size_t l) {
        if (l >= lanes_.L)
            return;
        const size_t n = lanes_.n, off = lanes_.idx(0, l);
        std::fill_n(lanes_.gdot.begin() + off, n, 0.0);
        std::fill_n(lanes_.gdot_ep.begin() + off, n, cur_gdot_epoch_);
    }

    // Keep the public inline helpers that your code calls:
    inline void resetTangentsAll() {
        std::fill(lanes_.dot.begin(), lanes_.dot.end(), 0.0);
        std::fill(lanes_.dot_ep.begin(), lanes_.dot_ep.end(), cur_dot_epoch_);
    }
    inline void resetGradDotAll() {
        std::fill(lanes_.gdot.begin(), lanes_.gdot.end(), 0.0);
        std::fill(lanes_.gdot_ep.begin(), lanes_.gdot_ep.end(),
                  cur_gdot_epoch_);
    }
    void resetTangents() { resetTangentsAll(); }
    void resetGradDot() { resetGradDotAll(); }
    void initiateBackwardPassHVP() { initiateBackwardPassFusedLanes(); }

    // ------------------------------------------------------------------------
    // PUBLIC: Evaluation (scalar & fused)
    // ------------------------------------------------------------------------
    void computeForwardPass(); // AoS compatibility
    void computeNodeValue(const ADNodePtr &node,
                          std::unordered_set<ADNodePtr> &visited);
    void resetForwardPass();

    void initiateBackwardPass(const ADNodePtr &outputNode);
    void resetGradients();

    // AoS-compatible fused reverse (scalar lanes)
    void initiateBackwardPassFused();

    // Fused (lanes): forward (prop dot) / reverse (accumulate gdot)
    void computeForwardPassWithDotLanes();
    void initiateBackwardPassFusedLanes();

    // Utility entry: forward + dot lanes in one go (implementation in .cpp)
    void computeForwardPassAndDotLanesTogether();

    // ------------------------------------------------------------------------
    // PUBLIC: Simplification / canonicalization / e-graph
    // ------------------------------------------------------------------------
    void simplifyExpression(std::vector<ADNodePtr> &outputs);
    inline void simplifyExpression(ADNodePtr &root) {
        std::vector<ADNodePtr> outs{root};
        simplifyExpression(outs);
        root = outs[0];
    }

    void simplifyGraph();    // pipeline orchestrator
    bool peepholeSimplify(); // DECLARED ONCE (public)
    void eliminateDeadCode();
    void constantFolding();
    void algebraicSimplification();

    // e-graph toggle/budget (public for tuning)
    bool enable_egraph_ = true;
    EGraphBudget egraph_budget_{5, 100000, 4};

    // ------------------------------------------------------------------------
    // PUBLIC: HVP / Hessian API
    // ------------------------------------------------------------------------
    std::vector<double> hessianVectorProduct(const ADNodePtr &outputNode,
                                             const std::vector<double> &v);
    std::vector<std::vector<double>> computeHessianDense(const ADNodePtr &y);
    void hessianMultiVectorProduct(const ADNodePtr &outputNode,
                                   const double *V_ptr, size_t ldV, // == L
                                   double *Y_ptr, size_t ldY,       // == L
                                   size_t k);

    // ------------------------------------------------------------------------
    // PUBLIC: Introspection / utilities
    // ------------------------------------------------------------------------
    std::string getExpression(const ADNodePtr &node);
    void printTree(const ADNodePtr &node, int depth = 0);
    std::vector<ADNodePtr> findRootNodes() const;

    tsl::robin_map<std::string, double>
    computePartialDerivatives(const ADNodePtr &expr);

    ADNodePtr getNode(const std::string &name);
    double getGradientOfVariable(const VariablePtr &var, const ADNodePtr &expr);
    double evaluate(const ADNodePtr &expr);
    void initializeNodeVariables();
    std::vector<double> getGradientVector(const ADNodePtr &expr);

    std::tuple<ADGraphPtr, tsl::robin_map<std::string, ADNodePtr>>
    rebuildGraphWithUniqueVariables(const ADNodePtr &rootNode);

    void collectNodes(const ADNodePtr &start,
                      tsl::robin_map<std::string, ADNodePtr> &coll,
                      std::unordered_set<ADNodePtr> &vis,
                      tsl::robin_map<std::string, ADNodePtr> &vars);

    // ------------------------------------------------------------------------
    // PUBLIC: Topology / ids (thin public surface; full helpers private)
    // ------------------------------------------------------------------------
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

private:
    tsl::robin_map<double, ADNodePtr> constant_pool_;

    // ========================================================================
    // Private: lane buffers management
    // ========================================================================
    inline void ensureLaneBuffers_() {
        const size_t n =
            cache_.by_id.empty() ? nodes.size() : cache_.by_id.size();
        lanes_.allocate(n, lanes_count_);
    }

    // ========================================================================
    // Private: fused kernels (per-node, loop over lanes)
    // ========================================================================
    void fused_forward_dot_kernel_lanes_(int uid);
    void fused_backward_kernel_lanes_(int uid);

    // ========================================================================
    // Private: rebuild / topo / indices
    // ========================================================================
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

    // ========================================================================
    // Private: uses-lists / mutation bookkeeping / canonicalization
    // ========================================================================
    std::unordered_map<const ADNode *, std::vector<ADNode *>> uses_;
    bool uses_valid_ = false;

    void buildUseListsOnce_();
    void markGraphMutated_();
    void canonicalizeOperands_(ADNode &node); // AC-aware local canon

    // Tiny execution helpers (AoS scalar path)
    void executeUnaryOp(ADNode *node, std::function<double(double)> forward_fn,
                        std::function<double(double)> backward_fn = nullptr) {
        (void)backward_fn; // silence -Wunused-parameter
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

    // ========================================================================
    // Private: simplification support (no duplicate public decls)
    // ========================================================================
    ADNodePtr applyAlgebraicRule(const ADNodePtr &node);
    bool isConstant(const ADNodePtr &node) const;
    bool isZero(const ADNodePtr &node) const;
    bool isOne(const ADNodePtr &node) const;
    void replaceNodeReferences(const ADNodePtr &oldNode,
                               const ADNodePtr &newNode);
    std::vector<ADNodePtr> findNodesWithoutForwardReferences() const;

    mutable bool simplification_needed_ = false;
    size_t last_simplification_size_ = 0;

    // quantized-fp for CTE key stability

    bool is_commutative_node_(Operator op) const {
        return op == Operator::Add || op == Operator::Multiply;
    }

    struct CSEKey {
        Operator op = Operator::NA;

        // Canonical children ids (ordered for non-commutative; sorted for
        // commutative)
        std::vector<uint64_t> kids_ids;

        // For AC-normalized ops only (otherwise empty):
        // - Add/Sub  : coeffs[i] = signed multiplicity of kids_ids[i]
        // - Mul/Div  : exponents[i] = signed integer exponent of kids_ids[i]
        std::vector<int> coeffs;    // used for Add/Sub
        std::vector<int> exponents; // used for Mul/Div

        // Constants: either constant node itself, or absorbed constant for
        // sums/products
        bool is_cte = false;
        uint64_t cbits = 0; // quantized/normalized bits (qfp, -0→+0, fixed NaN)

        // Pow(base,k) with small integer k
        int small_exp = 0; // 0 if not used

        bool operator==(const CSEKey &o) const {
            return op == o.op && is_cte == o.is_cte && cbits == o.cbits &&
                   small_exp == o.small_exp && kids_ids == o.kids_ids &&
                   coeffs == o.coeffs && exponents == o.exponents;
        }
    };

    struct CSEKeyHash {
        size_t operator()(const CSEKey &k) const noexcept {
            auto mix = [](size_t a, size_t b) {
                return a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2));
            };
            size_t h = std::hash<int>{}(int(k.op));
            h = mix(h, std::hash<bool>{}(k.is_cte));
            h = mix(h, std::hash<uint64_t>{}(k.cbits));
            h = mix(h, std::hash<int>{}(k.small_exp));
            // kids
            for (auto id : k.kids_ids)
                h = mix(h, std::hash<uint64_t>{}(id));
            // payloads (usually empty for non-AC ops)
            for (auto c : k.coeffs)
                h = mix(h, std::hash<int>{}(c));
            for (auto e : k.exponents)
                h = mix(h, std::hash<int>{}(e));
            return h;
        }
    };
    // AC-aware CSE
    bool cseByKey_();
    CSEKey makeCSEKey_(const ADNode &n) const;

    // e-graph entry (roots are replaced in-place if improved)
    bool egraphSimplify_(std::vector<ADNodePtr> &roots);

    // ========================================================================
    // Private: HVP preparation / reuse
    // ========================================================================
    struct HVPPrepared {
        uint64_t val_epoch = 0;
        uint64_t grad_epoch = 0;
        int y_id = -1;
        bool valid = false;
    };
    HVPPrepared hvp_prep_;

    void hessianMultiVectorProductReuseScalar(const ADNodePtr &y,
                                              const double *V, size_t ldV,
                                              double *Y, size_t ldY, size_t k);
    void ensurePreparedForHVP_(const ADNodePtr &y) {
        if (!y)
            return;
        const bool need_scalar = !hvp_prep_.valid || hvp_prep_.y_id != y->id ||
                                 hvp_prep_.val_epoch != cur_val_epoch_;
        if (need_scalar) {
            resetForwardPass();
            computeForwardPass();
            resetGradients();
            set_epoch_value(y->gradient, y->grad_epoch, cur_grad_epoch_, 1.0);
            initiateBackwardPassFused();
            hvp_prep_.y_id = y->id;
            hvp_prep_.val_epoch = cur_val_epoch_;
            hvp_prep_.grad_epoch = cur_grad_epoch_;
            hvp_prep_.valid = true;
        } else if (hvp_prep_.grad_epoch != cur_grad_epoch_) {
            resetGradients();
            set_epoch_value(y->gradient, y->grad_epoch, cur_grad_epoch_, 1.0);
            initiateBackwardPassFused();
            hvp_prep_.grad_epoch = cur_grad_epoch_;
        }
    }
};

// ============================================================================
// pick_graph helper
// ============================================================================
inline ADGraphPtr pick_graph(const ADGraphPtr &a,
                             const ADGraphPtr &b = nullptr) {
    if (a)
        return a;
    if (b)
        return b;
    return std::make_shared<ADGraph>();
}
