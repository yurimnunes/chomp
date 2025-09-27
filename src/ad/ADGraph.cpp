// ADGraph.cpp — ULTRA-FAST C++23 VERSION (lanes, fused kernels, minimal allocs)
#include "../../include/ad/ADGraph.h"
#include "../../include/ad/Definitions.h"
#include "../../include/ad/Variable.h"

#include "../../include/ad/OpDispatch.h"
#include "../../include/ad/OpTraits.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <immintrin.h> // AVX2 helpers (optional)
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

using VariablePtr = std::shared_ptr<Variable>;

// ============================== Local helpers ================================
namespace {

thread_local std::vector<size_t> g_scratch_bases; // new
// inline void ensure_base_size(size_t n) {
//     if (g_scratch_bases.size() < n)
//         g_scratch_bases.resize(n * 2);
// }

inline bool validate_nary_inputs(const std::vector<ADNodePtr> &inputs,
                                 std::string_view op_name,
                                 size_t min_inputs = 1) {
    if (inputs.size() < min_inputs) [[unlikely]] {
        std::cerr << "Error: " << op_name << " needs at least " << min_inputs
                  << " inputs, got " << inputs.size() << '\n';
        return false;
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!inputs[i]) [[unlikely]] {
            std::cerr << "Error: " << op_name << " has null input at index "
                      << i << '\n';
            return false;
        }
    }
    return true;
}

struct PtrHash {
    size_t operator()(const ADNode *p) const noexcept {
        return std::hash<const void *>{}(p);
    }
};
struct PtrEq {
    bool operator()(const ADNode *a, const ADNode *b) const noexcept {
        return a == b;
    }
};

// Pre-allocated thread-local scratch buffers to eliminate tiny allocs
thread_local std::vector<double> g_scratch_values;
thread_local std::vector<double> g_scratch_dots;
thread_local std::vector<double> g_scratch_softmax;

// inline void ensure_scratch_size(size_t n) {
//     if (g_scratch_values.size() < n) {
//         g_scratch_values.resize(n * 2);
//         // g_scratch_dots.resize(n * 2);
//         // g_scratch_softmax.resize(n * 2);
//     }
// }

#if defined(__AVX2__)
inline double hsum256_pd(__m256d v) {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d sum2 = _mm_add_pd(lo, hi);
    __m128d shuf = _mm_unpackhi_pd(sum2, sum2);
    __m128d sum = _mm_add_sd(sum2, shuf);
    double out;
    _mm_store_sd(&out, sum);
    return out;
}
#endif

} // namespace

// ==================== ADGraph: IDs / adjacency helpers =======================
int ADGraph::ensure_id_(const ADNode *n) {
    auto it = cache_.id_of.find(n);
    if (it != cache_.id_of.end())
        return it->second;

    int id;
    if (!cache_.free_ids.empty()) {
        id = cache_.free_ids.back();
        cache_.free_ids.pop_back();
        if (id >= (int)cache_.by_id.size()) {
            cache_.by_id.resize(id + 1, nullptr);
            cache_.adj.resize(id + 1);
            cache_.radj.resize(id + 1);
            cache_.indeg.resize(id + 1, 0);
        }
        cache_.by_id[id] = const_cast<ADNode *>(n);
    } else {
        id = cache_.next_id++;
        cache_.by_id.push_back(const_cast<ADNode *>(n));
        cache_.adj.emplace_back();
        cache_.radj.emplace_back();
        cache_.indeg.emplace_back(0);
    }
    cache_.id_of.emplace(n, id);
    const_cast<ADNode *>(n)->id = id;
    return id;
}

void ADGraph::release_id_(int id) {
    if (id < 0 || id >= (int)cache_.by_id.size())
        return;
    ADNode *node = cache_.by_id[id];
    if (node)
        node->id = -1;

    for (int c : cache_.adj[id]) {
        if (c >= 0 && c < (int)cache_.indeg.size())
            cache_.indeg[c] = std::max(0, cache_.indeg[c] - 1);
        cache_.radj[c].erase(id);
    }
    cache_.adj[id].clear();

    for (int p : cache_.radj[id])
        cache_.adj[p].erase(id);
    cache_.radj[id].clear();

    cache_.by_id[id] = nullptr;
    cache_.free_ids.push_back(id);
}
void ADGraph::link_edge_(int pid, int cid) {
    if (pid == cid)
        return;
    const int need = std::max(pid, cid) + 1;
    if ((int)cache_.adj.size() < need)
        cache_.adj.resize(need);
    if ((int)cache_.radj.size() < need)
        cache_.radj.resize(need);
    if ((int)cache_.indeg.size() < need)
        cache_.indeg.resize(need, 0);

    auto [it, inserted] = cache_.adj[pid].insert(cid);
    if (inserted) {
        cache_.radj[cid].insert(pid);
        cache_.indeg[cid]++; // only if new
    }
}

void ADGraph::unlink_edge_(int pid, int cid) {
    if (cache_.adj[pid].erase(cid)) {
        cache_.radj[cid].erase(pid);
        cache_.indeg[cid] = std::max(0, cache_.indeg[cid] - 1);
    }
}

// ================================ Dirty flags ================================
void ADGraph::markDirty_() { cache_.dirty = true; }
void ADGraph::markDirty_(ADNode *n) {
    cache_.dirty = true;
    if (n)
        cache_.dirty_nodes.insert(n);
}

// ============================ Graph maintenance ==============================
void ADGraph::addNode(const ADNodePtr &node) {
    if (!node)
        return;
    if (cache_.id_of.find(node.get()) != cache_.id_of.end())
        return; // fast dup check

    nodes.push_back(node);
    if (!node->name.empty())
        nodeIndex_[node->name] = node;

    const int u = ensure_id_(node.get());
    for (auto &in : node->inputs) {
        if (!in)
            continue;
        const int p = ensure_id_(in.get());
        link_edge_(p, u);
    }
    markDirty_(node.get());
}

void ADGraph::deleteNode(const ADNodePtr &node) {
    if (!node)
        return;
    if (!node->name.empty())
        nodeIndex_.erase(node->name);

    for (auto it = nodeVariables.begin(); it != nodeVariables.end();)
        it = (it->second == node) ? nodeVariables.erase(it) : std::next(it);

    // Unlink from all parents via radj
    auto itid = cache_.id_of.find(node.get());
    if (itid != cache_.id_of.end()) {
        int uid = itid->second;

        // For each parent p in radj[uid], erase uid from its adj and from its
        // inputs vector.
        std::vector<int> parents(cache_.radj[uid].begin(),
                                 cache_.radj[uid].end());
        for (int p : parents) {
            cache_.adj[p].erase(uid);
            if (p >= 0 && p < (int)cache_.by_id.size() && cache_.by_id[p]) {
                ADNode *parent = cache_.by_id[p];
                auto &ins = parent->inputs;
                ins.erase(std::remove(ins.begin(), ins.end(), node), ins.end());
                markDirty_(parent);
            }
            cache_.indeg[uid] = std::max(0, cache_.indeg[uid] - 1);
        }
        cache_.radj[uid].clear();

        // Finally release own id (will also unlink children & fix indegrees)
        cache_.id_of.erase(itid);
        release_id_(uid);
    }

    nodes.erase(std::remove(nodes.begin(), nodes.end(), node), nodes.end());
    markDirty_();
}

// === makeNodesUnique (canonicalize by name, rewire inputs, keep topo sane) ===
void ADGraph::makeNodesUnique() {
    tsl::robin_map<std::string, ADNodePtr> first_by_name;
    first_by_name.reserve(nodes.size());
    tsl::robin_map<ADNodePtr, ADNodePtr> canonical;
    canonical.reserve(nodes.size());
    std::vector<ADNodePtr> unique;
    unique.reserve(nodes.size());

    for (auto &n : nodes) {
        if (!n)
            continue;
        if (!n->name.empty()) {
            auto [it, inserted] = first_by_name.emplace(n->name, n);
            if (inserted) {
                canonical[n] = n;
                unique.push_back(n);
            } else {
                canonical[n] = it->second;
            }
        } else {
            canonical[n] = n;
            unique.push_back(n);
        }
    }

    for (auto &n : unique) {
        if (!n)
            continue;
        int u = ensure_id_(n.get());
        auto &ins = n->inputs;
        for (auto &in : ins) {
            if (!in)
                continue;
            auto itc = canonical.find(in);
            ADNodePtr new_in = (itc != canonical.end()) ? itc->second : in;
            if (new_in != in) {
                int p_old = ensure_id_(in.get());
                unlink_edge_(p_old, u);
                int p_new = ensure_id_(new_in.get());
                link_edge_(p_new, u);
                in = new_in;
                markDirty_(n.get());
            } else {
                int p = ensure_id_(in.get());
                link_edge_(p, u);
            }
        }
    }

    if (unique.size() != nodes.size()) {
        std::unordered_set<ADNodePtr> kept(unique.begin(), unique.end());
        kept.reserve(unique.size() * 2);
        for (auto &n : nodes) {
            if (!n)
                continue;
            if (!kept.count(n)) {
                auto it = cache_.id_of.find(n.get());
                if (it != cache_.id_of.end()) {
                    int id = it->second;
                    for (int c : cache_.adj[id]) {
                        if (c >= 0 && c < (int)cache_.by_id.size() &&
                            cache_.by_id[c])
                            markDirty_(cache_.by_id[c]);
                    }
                    cache_.id_of.erase(it);
                    release_id_(id);
                }
            }
        }
        nodes.swap(unique);
    } else {
        nodes = std::move(unique);
    }

    nodeIndex_.clear();
    nodeIndex_.reserve(first_by_name.size());
    for (const auto &kv : first_by_name)
        nodeIndex_[kv.first] = kv.second;

    for (auto it = nodeVariables.begin(); it != nodeVariables.end();) {
        auto jt = nodeIndex_.find(it->first);
        if (jt != nodeIndex_.end() && jt->second) {
            it.value() = jt->second;
            ++it;
        } else {
            it = nodeVariables.erase(it);
        }
    }

    cache_.dirty = true;
}

// ==================== Incremental topo maintenance (ids) =====================
void ADGraph::collectForward_(const std::vector<int> &starts,
                              std::vector<char> &in_aff,
                              std::vector<int> &aff) const {
    std::vector<int> q;
    q.reserve(starts.size());
    for (int s : starts)
        if (s >= 0) {
            q.push_back(s);
            in_aff[s] = 1;
            aff.push_back(s);
        }
    for (size_t i = 0; i < q.size(); ++i) {
        int u = q[i];
        for (int v : cache_.adj[u])
            if (!in_aff[v]) {
                in_aff[v] = 1;
                q.push_back(v);
                aff.push_back(v);
            }
    }
}

bool ADGraph::topoForAffected_(const std::vector<int> &affected,
                               std::vector<int> &topo_aff) const {
    if (affected.empty()) {
        topo_aff.clear();
        return true;
    }

    tsl::robin_map<int, int> idx;
    idx.reserve(affected.size());
    for (int i = 0; i < (int)affected.size(); ++i)
        idx[affected[i]] = i;

    std::vector<int> indeg_local(affected.size(), 0);
    for (int id : affected)
        for (int v : cache_.adj[id]) {
            auto it = idx.find(v);
            if (it != idx.end())
                indeg_local[it->second]++;
        }

    std::vector<int> q;
    q.reserve(affected.size());
    for (int i = 0; i < (int)affected.size(); ++i)
        if (indeg_local[i] == 0)
            q.push_back(i);

    topo_aff.clear();
    topo_aff.reserve(affected.size());
    for (size_t h = 0; h < q.size(); ++h) {
        int ii = q[h];
        int u = affected[ii];
        topo_aff.push_back(u);
        for (int v : cache_.adj[u]) {
            auto it = idx.find(v);
            if (it != idx.end()) {
                int j = it->second;
                if (--indeg_local[j] == 0)
                    q.push_back(j);
            }
        }
    }
    return topo_aff.size() == affected.size();
}

void ADGraph::refreshIncremental_() {
    if (nodes.size() < cache_.full_rebuild_threshold_nodes ||
        cache_.dirty_nodes.size() >
            cache_.full_rebuild_dirty_ratio * nodes.size()) {
        rebuildCacheFull_();
        return;
    }

    std::vector<int> starts;
    starts.reserve(cache_.dirty_nodes.size());
    for (auto *n : cache_.dirty_nodes) {
        if (!n)
            continue;
        starts.push_back((n->id >= 0) ? n->id : ensure_id_(n));
    }

    std::vector<char> in_aff(cache_.by_id.size(), 0);
    std::vector<int> affected;
    affected.reserve(starts.size() * 4);
    collectForward_(starts, in_aff, affected);

    std::vector<int> topo_ids;
    topo_ids.reserve(cache_.topo.size());
    for (auto *p : cache_.topo) {
        if (!p)
            continue;
        topo_ids.push_back((p->id >= 0) ? p->id : ensure_id_(p));
    }

    size_t w = 0;
    for (size_t r = 0; r < topo_ids.size(); ++r) {
        int id = topo_ids[r];
        if (id < (int)in_aff.size() && in_aff[id])
            continue;
        topo_ids[w++] = id;
    }
    topo_ids.resize(w);

    if (!affected.empty()) {
        std::vector<int> topo_aff;
        if (!topoForAffected_(affected, topo_aff)) {
            rebuildCacheFull_();
            return;
        }
        tsl::robin_map<int, int> pos;
        pos.reserve(topo_ids.size());
        for (int i = 0; i < (int)topo_ids.size(); ++i)
            pos[topo_ids[i]] = i;

        int insert_at = 0;
        for (int u : affected) {
            for (int p : cache_.radj[u]) {
                auto it = pos.find(p);
                if (it != pos.end())
                    insert_at = std::max(insert_at, it->second + 1);
            }
        }
        insert_at = std::min<int>(insert_at, (int)topo_ids.size());
        topo_ids.insert(topo_ids.begin() + insert_at, topo_aff.begin(),
                        topo_aff.end());
    }

    cache_.topo.clear();
    cache_.topo.reserve(topo_ids.size());
    for (int id : topo_ids) {
        if (id >= 0 && id < (int)cache_.by_id.size()) {
            ADNode *p = cache_.by_id[id];
            if (p)
                cache_.topo.push_back(p);
        }
    }

    cache_.dirty_nodes.clear();
    cache_.dirty = false;

    // Keep lane buffers sized to current graph
    ensureLaneBuffers_();
}

// ========================== Full rebuild (topo) ==============================
void ADGraph::rebuildCacheFull_() {
    makeNodesUnique();

    cache_.id_of.clear();
    cache_.by_id.clear();
    cache_.free_ids.clear();
    cache_.next_id = 0;
    cache_.adj.clear();
    cache_.radj.clear();
    cache_.indeg.clear();
    cache_.topo.clear();

    for (auto &n : nodes)
        if (n)
            (void)ensure_id_(n.get());
    cache_.adj.resize(cache_.by_id.size());
    cache_.radj.resize(cache_.by_id.size());
    cache_.indeg.assign(cache_.by_id.size(), 0);

    for (auto &n : nodes) {
        if (!n)
            continue;
        const int u = n->id;
        for (auto &in : n->inputs) {
            if (!in)
                continue;
            const int p = (in->id >= 0) ? in->id : ensure_id_(in.get());
            link_edge_(p, u);
        }
    }

    std::vector<int> indeg = cache_.indeg;
    std::vector<int> q;
    q.reserve(cache_.by_id.size());
    for (int i = 0; i < (int)cache_.by_id.size(); ++i)
        if (cache_.by_id[i] && indeg[i] == 0)
            q.push_back(i);

    std::vector<int> topo_ids;
    topo_ids.reserve(cache_.by_id.size());
    for (size_t h = 0; h < q.size(); ++h) {
        int u = q[h];
        topo_ids.push_back(u);
        for (int v : cache_.adj[u])
            if (--indeg[v] == 0)
                q.push_back(v);
    }

    cache_.topo.clear();
    cache_.topo.reserve(topo_ids.size());
    for (int id : topo_ids) {
        if (id >= 0 && id < (int)cache_.by_id.size()) {
            ADNode *p = cache_.by_id[id];
            if (p)
                cache_.topo.push_back(p);
        }
    }

    size_t live_nodes = 0;
    for (auto *p : cache_.by_id)
        if (p)
            ++live_nodes;
    if (cache_.topo.size() != live_nodes) {
        std::cerr << "Error: cycle detected in computation graph!\n";
        cache_.topo.clear();
    }

    cache_.dirty_nodes.clear();
    cache_.dirty = false;

    // Keep lane buffers sized to current graph
    ensureLaneBuffers_();
}

void ADGraph::rebuildCache_() {
    if (!cache_.dirty_nodes.empty())
        refreshIncremental_();
    else
        rebuildCacheFull_();
}

// =================== Scalar forward (AoS) & backward (AoS) ===================
void ADGraph::computeForwardPass() {
    if (cache_.dirty)
        rebuildCache_();
    for (ADNode *u : cache_.topo) {
        if (!u)
            continue;
        dispatch_op(*u, ForwardFunctor{*this, *u});
    }
    if (cache_.topo.size() != nodes.size()) [[unlikely]]
        std::cerr << "Warning: cycle or dangling inputs in AD graph.\n";
}

void ADGraph::initiateBackwardPassFused() {
    if (cache_.dirty)
        rebuildCache_();
    for (auto it = cache_.topo.rbegin(); it != cache_.topo.rend(); ++it) {
        ADNode *u = *it;
        if (!u)
            continue;
        dispatch_op(*u, BackwardFunctor{*this, *u});
    }
}

// ======================== Lane-aware forward/backward ========================
// ADGraph.cpp - Updated methods using dispatcher

void ADGraph::computeForwardPassWithDotLanes() {
    if (cache_.dirty)
        rebuildCache_();

    const size_t L = lanes(); // use public lane count

    for (ADNode *u : cache_.topo) {
        if (!u)
            continue;
        const int uid = u->id;
        const size_t ybase = lanes_.base(uid);

        // Replace the entire giant switch statement with:
        dispatch_op(*u, ForwardDotLanesFunctor{*this, *u, L, ybase});
    }
}

void ADGraph::initiateBackwardPassFusedLanes() {
    if (cache_.dirty)
        rebuildCache_();
    const size_t L = lanes(); // use public lane count

    // NOTE: we rely on caller (e.g., hessianMultiVectorProduct) to zero gdot.
    // No per-lane epoch seeding here.

    for (auto it = cache_.topo.rbegin(); it != cache_.topo.rend(); ++it) {
        ADNode *u = *it;
        if (!u)
            continue;

        const int uid = u->id;
        const size_t ybase = lanes_.base(uid);

        // Replace the entire giant switch statement with:
        dispatch_op(*u, BackwardLanesFunctor{*this, *u, L, ybase});
    }
}

void ADGraph::computeForwardPassAndDotLanesTogether() {
    if (cache_.dirty)
        rebuildCache_();

    const size_t L = lanes();

    for (ADNode *u : cache_.topo) {
        if (!u)
            continue;

        const int uid = u->id;
        const size_t ybase = lanes_.base(uid);

        // Replace the massive switch statement with:
        dispatch_op(*u, FusedForwardFunctor{*this, *u, L, ybase});
    }

    if (cache_.topo.size() != nodes.size()) [[unlikely]]
        std::cerr << "Warning: cycle or dangling inputs in AD graph.\n";
}

void ADGraph::resetForwardPass() {
    ++cur_val_epoch_;
    if (cur_val_epoch_ == 0)
        ++cur_val_epoch_;
}

void ADGraph::initiateBackwardPass(const ADNodePtr & /*outputNode*/) {
    initiateBackwardPassFused();
}

void ADGraph::resetGradients() {
    ++cur_grad_epoch_;
    if (cur_grad_epoch_ == 0)
        ++cur_grad_epoch_;
}

// ------------------------------ Derivative APIs
// ------------------------------
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
    (void)computePartialDerivatives(expr);
    std::vector<double> g(varSize, 0.0);
    for (const auto &n : nodes) {
        if (n->type == Operator::Var && n->order >= 0 &&
            (size_t)n->order < varSize)
            g[n->order] = n->gradient;
    }
    return g;
}

// ------------------------- Multi-RHS HVP (lanes)
// -----------------------------
// void ADGraph::hessianMultiVectorProduct(const ADNodePtr &y, const double *V,
//                                         size_t ldV, double *Y, size_t ldY,
//                                         size_t k) {
//     if (!y || k == 0)
//         return;
//     if (cache_.dirty)
//         rebuildCache_();
//     initializeNodeVariables();
//     const size_t n = nodeVariables.size();
//     if (n == 0)
//         return;

//     set_num_lanes(k);
//     ensureLaneBuffers_();

//     // 1) zero lanes and load V into variable rows
//     std::fill(lanes_.dot.begin(), lanes_.dot.end(), 0.0);
//     std::fill(lanes_.gdot.begin(), lanes_.gdot.end(), 0.0);
//     for (const auto &kv : nodeVariables) {
//         const auto &var = kv.second;
//         if (!var)
//             continue;
//         const int ord = var->order;
//         if (ord < 0)
//             continue;
//         const size_t vbase = lanes_.base(var->id);
//         for (size_t l = 0; l < k; ++l)
//             lanes_.dot[vbase + l] = V[size_t(ord) * ldV + l];
//     }

//     // 2) fused scalar forward + lane forward
//     resetForwardPass();
//     computeForwardPassAndDotLanesTogether();

//     // 3) scalar reverse to get w = ∂y/∂x
//     resetGradients();
//     set_epoch_value(y->gradient, y->grad_epoch, cur_grad_epoch_, 1.0);
//     initiateBackwardPassFused();

//     // 4) lane reverse using w and per-lane dots
//     resetGradDotAll();
//     initiateBackwardPassFusedLanes();

//     // 5) gather Y
//     for (const auto &kv : nodeVariables) {
//         const auto &var = kv.second;
//         if (!var)
//             continue;
//         const int ord = var->order;
//         if (ord < 0)
//             continue;
//         const size_t gbase = lanes_.base(var->id);
//         for (size_t l = 0; l < k; ++l)
//             Y[size_t(ord) * ldY + l] = lanes_.gdot[gbase + l];
//     }
// }

// --- (Optional) refit your existing method to call the reuse path safely ---
void ADGraph::hessianMultiVectorProduct(const ADNodePtr &y, const double *V,
                                        size_t ldV, double *Y, size_t ldY,
                                        size_t k) {
    // Keep the original behavior for first call,
    // then all subsequent calls can use the reuseScalar API.
    hessianMultiVectorProductReuseScalar(y, V, ldV, Y, ldY, k);
}


// --- New public convenience that reuses scalar state across calls ---
void ADGraph::hessianMultiVectorProductReuseScalar(const ADNodePtr &y,
                                                   const double *V, size_t ldV,
                                                   double *Y, size_t ldY,
                                                   size_t k) {
    if (!y || k == 0) return;
    if (cache_.dirty) rebuildCache_();
    initializeNodeVariables();
    const size_t n = nodeVariables.size();
    if (n == 0) return;

    // 0) Ensure scalar forward/reverse are current for (x, y)
    ensurePreparedForHVP_(y);

    // 1) lanes: size/zero + load V into variable rows
    set_num_lanes(k);
    ensureLaneBuffers_();
    std::fill(lanes_.dot.begin(),  lanes_.dot.end(),  0.0);
    std::fill(lanes_.gdot.begin(), lanes_.gdot.end(), 0.0);

    for (const auto &kv : nodeVariables) {
        const auto &var = kv.second;
        if (!var) continue;
        const int ord = var->order;
        if (ord < 0) continue;
        const size_t vbase = lanes_.base(var->id);
        for (size_t l = 0; l < k; ++l)
            lanes_.dot[vbase + l] = V[size_t(ord) * ldV + l];
    }

    // 2) dot-only forward using cached primal values (no scalar forward)
    computeForwardPassWithDotLanes();

    // 3) lane reverse (uses already-computed scalar adjoints)
    resetGradDotAll();
    initiateBackwardPassFusedLanes();

    // 4) gather Y
    for (const auto &kv : nodeVariables) {
        const auto &var = kv.second;
        if (!var) continue;
        const int ord = var->order;
        if (ord < 0) continue;
        const size_t gbase = lanes_.base(var->id);
        for (size_t l = 0; l < k; ++l)
            Y[size_t(ord) * ldY + l] = lanes_.gdot[gbase + l];
    }
}


std::vector<double>
ADGraph::hessianVectorProduct(const ADNodePtr &outputNode,
                              const std::vector<double> &v) {
    const size_t nvars = nodeVariables.size();
    std::vector<double> Hv(nvars, 0.0);
    if (!outputNode || nvars == 0)
        return Hv;

    // Bridge to multi-RHS implementation with k=1
    set_num_lanes(1);
    ensureLaneBuffers_();
    hessianMultiVectorProduct(outputNode, v.data(), /*ldV=*/1, Hv.data(),
                              /*ldY=*/1,
                              /*k=*/1);
    return Hv;
}

std::vector<std::vector<double>>
ADGraph::computeHessianDense(const ADNodePtr &y) {
    const size_t n = nodeVariables.size();
    std::vector<std::vector<double>> H(n, std::vector<double>(n, 0.0));
    if (n == 0 || !y)
        return H;

    // Batch columns in groups of L lanes for throughput
    const size_t L =
        std::min<size_t>(std::max<size_t>(lanes(), 1), 32); // small cap
    std::vector<double> V(n * L, 0.0), Y(n * L, 0.0);

    for (size_t base = 0; base < n; base += L) {
        const size_t k = std::min(L, n - base);
        // Build V with k unit vectors
        std::fill(V.begin(), V.end(), 0.0);
        for (size_t j = 0; j < k; ++j)
            V[(base + j) * L + j] = 1.0;

        hessianMultiVectorProduct(y, V.data(), /*ldV=*/L, Y.data(),
                                  /*ldY=*/L, k);

        // Scatter into H
        for (size_t j = 0; j < k; ++j) {
            for (size_t r = 0; r < n; ++r) {
                H[r][base + j] = Y[r * L + j];
            }
        }
    }
    return H;
}

// ------------------------------- Subgraph ops
// --------------------------------
void ADGraph::adoptSubgraph(const ADNodePtr &root) {
    if (!root)
        return;

    std::unordered_set<const ADNode *, PtrHash, PtrEq> visited;
    visited.reserve(256);

    std::vector<ADNodePtr> stack;
    stack.reserve(256);
    stack.push_back(root);

    while (!stack.empty()) {
        ADNodePtr n = std::move(stack.back());
        stack.pop_back();
        if (!n)
            continue;
        const ADNode *raw = n.get();
        if (!visited.insert(raw).second)
            continue;

        addNode(n);
        if (n->type == Operator::Var && !n->name.empty())
            nodeVariables[n->name] = n;

        // push inputs
        for (const auto &in : n->inputs)
            if (in)
                stack.push_back(in);
    }
    // peepholeSimplify_();

    markDirty_();
    // simplifyGraph(); // Add this line at the end
}

// In ADGraph.cpp

void ADGraph::updateNodeIndex_() {
    nodeIndex_.clear();
    nodeIndex_.reserve(nodes.size());
    for (const auto &node : nodes)
        if (node && !node->name.empty())
            nodeIndex_[node->name] = node;
}

// ===================== Expression / debug printing
// ===========================
std::string ADGraph::getExpression(const ADNodePtr &node) {
    return "Unsupported in this build";
}

std::vector<ADNodePtr> ADGraph::findRootNodes() const {
    // We need a non-const rebuild check; keep this function non-const if you
    // want it to rebuild. If you must keep it const, drop the rebuild check (or
    // mark members mutable).
    if (cache_.dirty) {
        const_cast<ADGraph *>(this)->rebuildCache_();
    }

    std::vector<ADNodePtr> roots;
    roots.reserve(nodes.size());

    // O(N): any node whose indegree is zero is a root.
    for (const auto &n : nodes) {
        if (!n)
            continue;
        int id = (n->id >= 0)
                     ? n->id
                     : const_cast<ADGraph *>(this)->ensure_id_(n.get());
        // guard: cache_ vectors sized during rebuild
        if (id >= 0 && id < (int)cache_.indeg.size() && cache_.indeg[id] == 0) {
            roots.push_back(n);
        }
    }

    // (Optional) keep the warning for visibility, but it’s not an error.
    if (roots.empty())
        std::cerr << "Warning: No root nodes found\n";

    return roots;
}

// -------------------------- Unique rebuild helpers
// --------------------------
std::tuple<ADGraphPtr, tsl::robin_map<std::string, ADNodePtr>>
ADGraph::rebuildGraphWithUniqueVariables(const ADNodePtr &rootNode) {
    tsl::robin_map<std::string, ADNodePtr> coll;
    tsl::robin_map<std::string, ADNodePtr> vars;

    if (!rootNode) {
        auto newG = std::make_shared<ADGraph>();
        return {newG, newG->nodeVariables};
    }

    coll.reserve(128);
    vars.reserve(64);

    // visited on raw pointers (cheaper hash & no shared_ptr refcount churn)
    struct RawPtrHash {
        size_t operator()(const ADNode *p) const noexcept {
            return std::hash<const void *>{}(p);
        }
    };
    struct RawPtrEq {
        bool operator()(const ADNode *a, const ADNode *b) const noexcept {
            return a == b;
        }
    };
    std::unordered_set<const ADNode *, RawPtrHash, RawPtrEq> vis;
    vis.reserve(256);

    // iterative DFS stack
    std::vector<ADNodePtr> stack;
    stack.reserve(256);
    stack.push_back(rootNode);

    // For unnamed nodes we still need a key; avoid multiple std::to_string()
    // by keeping a running counter and a small stack buffer for formatting.
    size_t unnamed_counter = 0;

    while (!stack.empty()) {
        ADNodePtr n = std::move(stack.back());
        stack.pop_back();
        if (!n)
            continue;
        const ADNode *raw = n.get();
        if (!vis.insert(raw).second)
            continue;

        if (!n->name.empty()) {
            // only first occurrence matters (like your previous code)
            coll.emplace(n->name, n);
            if (n->type == Operator::Var)
                vars.emplace(n->name, n);
        } else {
            // generate compact key: "n_<counter>"
            // This avoids materializing potentially long decimal strings
            // multiple times.
            char buf[24];
            int len =
                std::snprintf(buf, sizeof(buf), "n_%zu", unnamed_counter++);
            coll.emplace(std::string(buf, (size_t)len), n);
        }

        // push inputs
        for (const auto &in : n->inputs)
            if (in)
                stack.push_back(in);
    }

    auto newG = std::make_shared<ADGraph>();
    newG->nodes.reserve(coll.size());
    for (const auto &kv : coll)
        newG->addNode(kv.second);
    newG->nodeVariables = std::move(vars);
    newG->makeNodesUnique();
    return {newG, newG->nodeVariables};
}

////////////////////////////////////////////////////

ADNodePtr ADGraph::createConstantNode(double value) {
    // Quantize a bit for stable hashing (tweak if you like)
    double q = std::nearbyint(value * 1e12) * 1e-12;

    auto it = constant_pool_.find(q);
    if (it != constant_pool_.end())
        return it->second;

    auto node = std::make_shared<ADNode>();
    node->type = Operator::cte;
    node->value = q;
    node->val_epoch = cur_val_epoch_;
    addNode(node);

    constant_pool_[q] = node;
    uses_valid_ = false; // graph changed
    return node;
}

void ADGraph::buildUseListsOnce_() {
    if (uses_valid_)
        return;
    uses_.clear();
    uses_.reserve(nodes.size() * 2);
    for (const auto &n : nodes) {
        if (!n)
            continue;
        for (const auto &in : n->inputs) {
            if (in)
                uses_[in.get()].push_back(n.get());
        }
    }
    uses_valid_ = true;
}

void ADGraph::markGraphMutated_() {
    uses_valid_ = false;
    simplification_needed_ = true;
    markDirty_();
}
void ADGraph::simplifyGraph() {
    if (!simplification_needed_ && nodes.size() == last_simplification_size_)
        return;

    size_t initial_size = nodes.size();
    bool changed = true;
    int iterations = 0;
    const int max_iterations = 10;

    buildUseListsOnce_(); // NEW: enables fast rewrites

    while (changed && iterations < max_iterations) {
        changed = false;

        constantFolding();
        algebraicSimplification();

        if (peepholeSimplify())
            changed = true;

        // NEW: non-destructive, AC-aware CSE
        if (cseByKey_())
            changed = true;

        size_t before_dce = nodes.size();
        eliminateDeadCode();
        if (nodes.size() < before_dce) {
            changed = true;
            uses_valid_ = false;
        }

        iterations++;
    }

    if (nodes.size() != initial_size) {
        markDirty_();
        makeNodesUnique();
    }
    last_simplification_size_ = nodes.size();
    simplification_needed_ = false;
    // std::cout << "Simplification: " << initial_size << " -> " << nodes.size()
    //           << " nodes (" << iterations << " iterations)\n";
}

void ADGraph::constantFolding() {
    std::vector<ADNodePtr> to_replace;

    for (const auto &node : nodes) {
        if (!node)
            continue;

        // Skip if already constant or variable
        if (node->type == Operator::cte || node->type == Operator::Var) {
            continue;
        }

        // Check if all inputs are constants
        bool all_constant = true;
        for (const auto &input : node->inputs) {
            if (!input || !isConstant(input)) {
                all_constant = false;
                break;
            }
        }

        if (all_constant) {
            // Evaluate the operation with constant inputs
            double result = 0.0;

            switch (node->type) {
            case Operator::Add:
                if (node->inputs.size() >= 2) {
                    result = node->inputs[0]->value + node->inputs[1]->value;
                }
                break;

            case Operator::Subtract:
                if (node->inputs.size() >= 2) {
                    result = node->inputs[0]->value - node->inputs[1]->value;
                }
                break;

            case Operator::Multiply:
                if (node->inputs.size() >= 2) {
                    result = node->inputs[0]->value * node->inputs[1]->value;
                }
                break;

            case Operator::Divide:
                if (node->inputs.size() >= 2 && node->inputs[1]->value != 0.0) {
                    result = node->inputs[0]->value / node->inputs[1]->value;
                }
                break;

            case Operator::Sin:
                if (!node->inputs.empty()) {
                    result = std::sin(node->inputs[0]->value);
                }
                break;

            case Operator::Cos:
                if (!node->inputs.empty()) {
                    result = std::cos(node->inputs[0]->value);
                }
                break;

            case Operator::Tan:
                if (!node->inputs.empty()) {
                    result = std::tan(node->inputs[0]->value);
                }
                break;

            case Operator::Exp:
                if (!node->inputs.empty()) {
                    result = std::exp(node->inputs[0]->value);
                }
                break;

            case Operator::Log:
                if (!node->inputs.empty() && node->inputs[0]->value > 0.0) {
                    result = std::log(node->inputs[0]->value);
                }
                break;

            case Operator::Tanh:
                if (!node->inputs.empty()) {
                    result = std::tanh(node->inputs[0]->value);
                }
                break;

            case Operator::Relu:
                if (!node->inputs.empty()) {
                    result = std::max(0.0, node->inputs[0]->value);
                }
                break;

            case Operator::Max:
                if (node->inputs.size() >= 2) {
                    result = std::max(node->inputs[0]->value,
                                      node->inputs[1]->value);
                }
                break;

            case Operator::Silu:
                if (!node->inputs.empty()) {
                    double x = node->inputs[0]->value;
                    result = x / (1.0 + std::exp(-x)); // x * sigmoid(x)
                }
                break;

            case Operator::Gelu:
                if (!node->inputs.empty()) {
                    double x = node->inputs[0]->value;
                    result = 0.5 * x *
                             (1.0 + std::tanh(std::sqrt(2.0 / M_PI) *
                                              (x + 0.044715 * x * x * x)));
                }
                break;

            default:
                all_constant = false; // Skip unsupported operations
                break;
            }

            if (all_constant) {
                auto constant_node = createConstantNode(result);
                constant_node->name = node->name; // Preserve name if any
                to_replace.push_back(node);
                replaceNodeReferences(node, constant_node);
            }
        }
    }
}

void ADGraph::algebraicSimplification() {
    for (const auto &node : nodes) {
        if (!node)
            continue;

        auto simplified = applyAlgebraicRule(node);
        if (simplified && simplified != node) {
            replaceNodeReferences(node, simplified);
        }
    }
}

ADNodePtr ADGraph::applyAlgebraicRule(const ADNodePtr &node) {
    if (!node || node->inputs.empty())
        return nullptr;

    switch (node->type) {
    case Operator::Add:
        if (node->inputs.size() >= 2) {
            // x + 0 = x
            if (isZero(node->inputs[1]))
                return node->inputs[0];
            if (isZero(node->inputs[0]))
                return node->inputs[1];

            // x + x = 2*x (if inputs are the same)
            if (node->inputs[0] == node->inputs[1]) {
                auto two = createConstantNode(2.0);
                // Would need to create multiplication node here
                // This requires access to node creation methods
            }
        }
        break;

    case Operator::Subtract:
        if (node->inputs.size() >= 2) {
            // x - 0 = x
            if (isZero(node->inputs[1]))
                return node->inputs[0];
            // x - x = 0
            if (node->inputs[0] == node->inputs[1])
                return createConstantNode(0.0);
        }
        break;

    case Operator::Multiply:
        if (node->inputs.size() >= 2) {
            // x * 0 = 0
            if (isZero(node->inputs[0]) || isZero(node->inputs[1])) {
                return createConstantNode(0.0);
            }
            // x * 1 = x
            if (isOne(node->inputs[1]))
                return node->inputs[0];
            if (isOne(node->inputs[0]))
                return node->inputs[1];
        }
        break;

    case Operator::Divide:
        if (node->inputs.size() >= 2) {
            // x / 1 = x
            if (isOne(node->inputs[1]))
                return node->inputs[0];
            // 0 / x = 0 (assuming x != 0)
            if (isZero(node->inputs[0]) && !isZero(node->inputs[1])) {
                return createConstantNode(0.0);
            }
        }
        break;

    case Operator::Relu:
        // relu(0) = 0, relu(positive_constant) = constant
        if (!node->inputs.empty() && isConstant(node->inputs[0])) {
            double val = node->inputs[0]->value;
            if (val <= 0.0) {
                return createConstantNode(0.0);
            } else {
                return node->inputs[0]; // relu(positive) = positive
            }
        }
        break;

    case Operator::Tanh:
        // tanh(0) = 0
        if (!node->inputs.empty() && isZero(node->inputs[0])) {
            return createConstantNode(0.0);
        }
        break;

    case Operator::Silu:
        // silu(0) = 0
        if (!node->inputs.empty() && isZero(node->inputs[0])) {
            return createConstantNode(0.0);
        }
        break;

    case Operator::Gelu:
        // gelu(0) = 0
        if (!node->inputs.empty() && isZero(node->inputs[0])) {
            return createConstantNode(0.0);
        }
        break;

    case Operator::Exp:
        // exp(0) = 1
        if (!node->inputs.empty() && isZero(node->inputs[0])) {
            return createConstantNode(1.0);
        }
        break;

    case Operator::Log:
        // log(1) = 0
        if (!node->inputs.empty() && isOne(node->inputs[0])) {
            return createConstantNode(0.0);
        }
        break;

    default:
        break;
    }

    return nullptr; // No simplification found
}
void ADGraph::compactNodeIds_() {
    // Build new contiguous ids keyed by raw pointer
    std::unordered_map<const ADNode *, int> new_id;
    new_id.reserve(nodes.size());
    int next = 0;
    for (const auto &sp : nodes)
        if (sp)
            new_id[sp.get()] = next++;

    // Assign new ids on the nodes (do NOT touch inputs' ids directly)
    for (const auto &sp : nodes)
        if (sp)
            sp->id = new_id[sp.get()];

    // Rebuild cache so adj/radj/indeg/topo match new ids
    rebuildCacheFull_();
}

void ADGraph::eliminateDeadCode() {
    // 1) Mark reachable from variables and roots (no rebuild up front)
    std::unordered_set<const ADNode *> reachable;
    std::queue<const ADNode *> q;

    // Variables are always live
    for (const auto &kv : nodeVariables) {
        if (kv.second) {
            const ADNode *p = kv.second.get();
            if (reachable.insert(p).second)
                q.push(p);
        }
    }

    // Roots = nodes with no forward references (your helper is fine)
    auto roots =
        findNodesWithoutForwardReferences(); // may rebuild if cache_.dirty; ok
    for (const auto &r : roots) {
        if (r) {
            const ADNode *p = r.get();
            if (reachable.insert(p).second)
                q.push(p);
        }
    }

    // BFS over inputs using the node's own inputs (no cache needed here)
    while (!q.empty()) {
        const ADNode *u = q.front();
        q.pop();
        for (const auto &in : u->inputs) {
            if (in) {
                const ADNode *v = in.get();
                if (reachable.insert(v).second)
                    q.push(v);
            }
        }
    }

    // 2) Collect & delete unreachable
    std::vector<ADNodePtr> to_remove;
    to_remove.reserve(nodes.size());
    for (const auto &n : nodes)
        if (!n || !reachable.count(n.get()))
            to_remove.push_back(n);

    for (const auto &n : to_remove)
        if (n)
            deleteNode(n);

    // 3) Compact ids + single rebuild to keep adjacency consistent
    compactNodeIds_();
}

bool ADGraph::peepholeSimplify() {
    bool changed = false;

    // Look for specific patterns that can be optimized
    for (const auto &node : nodes) {
        if (!node)
            continue;
        // canonicalizeOperands_(*node); // NEW

        // Pattern: (x + y) - y = x
        if (node->type == Operator::Subtract && node->inputs.size() >= 2) {
            auto left = node->inputs[0];
            auto right = node->inputs[1];

            if (left && left->type == Operator::Add &&
                left->inputs.size() >= 2) {
                // Check if right operand of subtraction matches either operand
                // of addition
                if (left->inputs[1] == right) {
                    replaceNodeReferences(node, left->inputs[0]);
                    changed = true;
                    continue;
                }
                if (left->inputs[0] == right) {
                    replaceNodeReferences(node, left->inputs[1]);
                    changed = true;
                    continue;
                }
            }
        }

        // Pattern: x * (y / x) = y (assuming x != 0)
        if (node->type == Operator::Multiply && node->inputs.size() >= 2) {
            auto left = node->inputs[0];
            auto right = node->inputs[1];

            if (right && right->type == Operator::Divide &&
                right->inputs.size() >= 2) {
                if (left == right->inputs[1]) { // x * (y / x)
                    replaceNodeReferences(node, right->inputs[0]);
                    changed = true;
                    continue;
                }
            }
        }
    }

    return changed;
}

inline static bool is_commutative_(Operator t) {
    return t == Operator::Add || t == Operator::Multiply || t == Operator::Max;
}
inline static bool is_associative_(Operator t) {
    return t == Operator::Add || t == Operator::Multiply;
}

void ADGraph::canonicalizeOperands_(ADNode &n) {
    if (is_associative_(n.type)) {
        std::vector<ADNodePtr> flat;
        flat.reserve(n.inputs.size());
        for (auto &in : n.inputs) {
            if (in && in->type == n.type) {
                flat.insert(flat.end(), in->inputs.begin(), in->inputs.end());
            } else {
                flat.push_back(in);
            }
        }
        n.inputs.swap(flat);
    }
    if (is_commutative_(n.type)) {
        std::stable_sort(n.inputs.begin(), n.inputs.end(),
                         [&](const ADNodePtr &a, const ADNodePtr &b) {
                             const bool ca = a && a->type == Operator::cte;
                             const bool cb = b && b->type == Operator::cte;
                             if (ca != cb)
                                 return !ca && cb; // non-consts first
                             return a.get() < b.get();
                         });
    }
}

// Helper methods

bool ADGraph::isConstant(const ADNodePtr &node) const {
    return node && node->type == Operator::cte;
}

bool ADGraph::isZero(const ADNodePtr &node) const {
    return isConstant(node) && std::abs(node->value) < 1e-12;
}

bool ADGraph::isOne(const ADNodePtr &node) const {
    return isConstant(node) && std::abs(node->value - 1.0) < 1e-12;
}
void ADGraph::replaceNodeReferences(const ADNodePtr &oldNode,
                                    const ADNodePtr &newNode) {
    if (!oldNode || !newNode || oldNode == newNode)
        return;

    // Try the fast path with use-lists
    buildUseListsOnce_();
    auto it = uses_.find(oldNode.get());
    if (it != uses_.end()) {
        auto &users = it->second; // vector<ADNode*>

        for (ADNode *u : users) {
            for (auto &inp : u->inputs) {
                if (inp.get() == oldNode.get()) {
                    inp = newNode;
                    markDirty_(u);
                }
            }
            uses_[newNode.get()].push_back(u);
        }
        users.clear();
    } else {
        // Fallback: full scan if not found (keeps compatibility)
        for (const auto &node : nodes) {
            if (!node)
                continue;
            for (auto &input : node->inputs) {
                if (input == oldNode) {
                    input = newNode;
                    markDirty_(node.get());
                }
            }
        }
        // Also update uses_ map lazily (we’ll rebuild later anyway)
        uses_valid_ = false;
    }

    // Keep your name/index and variable bookkeeping exactly as before
    if (!oldNode->name.empty()) {
        nodeIndex_[oldNode->name] = newNode;
        newNode->name = oldNode->name;
    }
    std::vector<std::string> keys_to_update;
    for (const auto &kv : nodeVariables)
        if (kv.second == oldNode)
            keys_to_update.push_back(kv.first);
    for (const auto &key : keys_to_update)
        nodeVariables[key] = newNode;

    markGraphMutated_(); // maintains your existing flags
}

std::vector<ADNodePtr> ADGraph::findNodesWithoutForwardReferences() const {
    std::unordered_set<ADNodePtr> referenced;

    // Collect all nodes that are referenced as inputs
    for (const auto &node : nodes) {
        if (!node)
            continue;
        for (const auto &input : node->inputs) {
            if (input) {
                referenced.insert(input);
            }
        }
    }

    // Find nodes that are not referenced by others
    std::vector<ADNodePtr> roots;
    for (const auto &node : nodes) {
        if (node && referenced.find(node) == referenced.end()) {
            roots.push_back(node);
        }
    }

    return roots;
}

bool ADGraph::cseByKey_() {
    // Build use-lists if you have them; it speeds replacement a lot.
    buildUseListsOnce_();

    std::unordered_map<CSEKey, ADNodePtr, CSEKeyHash> seen;
    seen.reserve(nodes.size() * 2);

    bool changed = false;
    for (auto &sp : nodes) {
        if (!sp)
            continue;

        // Skip pure variables and raw constants: you likely already pool
        // constants.
        if (sp->type == Operator::Var)
            continue;

        CSEKey key = makeCSEKey_(*sp);

        // Be conservative: avoid families with tricky NaN semantics
        // (Min/Max). We didn't include them in a family.
        auto it = seen.find(key);
        if (it == seen.end()) {
            seen.emplace(std::move(key), sp);
        } else {
            // Redirect users of this node to the first representative
            replaceNodeReferences(sp, it->second);
            changed = true;
        }
    }

    return changed;
}