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
inline void ensure_base_size(size_t n) {
    if (g_scratch_bases.size() < n)
        g_scratch_bases.resize(n * 2);
}

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

inline void ensure_scratch_size(size_t n) {
    if (g_scratch_values.size() < n) {
        g_scratch_values.resize(n * 2);
        // g_scratch_dots.resize(n * 2);
        // g_scratch_softmax.resize(n * 2);
    }
}

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
        _dispatch_op(*u, ForwardFunctor{*this, *u});
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
        _dispatch_op(*u, BackwardFunctor{*this, *u});
    }
}

// ======================== Lane-aware forward/backward ========================
// Forward lanes: propagate per-lane dot using node AoS primals.
void ADGraph::computeForwardPassWithDotLanes() {
    if (cache_.dirty)
        rebuildCache_();

    const size_t L = lanes(); // use public lane count

    for (ADNode *u : cache_.topo) {
        if (!u)
            continue;
        const int uid = u->id;
        const size_t ybase = lanes_.base(uid);

        switch (u->type) {
        case Operator::Var:
        case Operator::cte:
            // variables already have per-lane dots loaded; constants keep 0
            break;

        case Operator::Add: {
            const size_t m = u->inputs.size();
            if (m == 0)
                break;
            std::fill_n(&lanes_.dot[ybase], L, 0.0);
            for (const auto &inp : u->inputs) {
                const size_t ibase = lanes_.base(inp->id);
                for (size_t l = 0; l < L; ++l)
                    lanes_.dot[ybase + l] += lanes_.dot[ibase + l];
            }
            break;
        }

        case Operator::Subtract: {
            if (u->inputs.size() != 2)
                break;
            const int ia = u->inputs[0]->id, ib = u->inputs[1]->id;
            const size_t abase = lanes_.base(ia), bbase = lanes_.base(ib);
            for (size_t l = 0; l < L; ++l)
                lanes_.dot[ybase + l] =
                    lanes_.dot[abase + l] - lanes_.dot[bbase + l];
            break;
        }
        case Operator::Multiply: {
            const size_t m = u->inputs.size();
            if (m == 0)
                break;

            if (m == 2) [[likely]] {
                const int ia = u->inputs[0]->id, ib = u->inputs[1]->id;
                const size_t abase = lanes_.base(ia), bbase = lanes_.base(ib);
                const double aval = u->inputs[0]->value,
                             bval = u->inputs[1]->value;
                for (size_t l = 0; l < L; ++l) {
                    const double ad = lanes_.dot[abase + l];
                    const double bd = lanes_.dot[bbase + l];
                    lanes_.dot[ybase + l] = ad * bval + aval * bd;
                }
            } else {
                // General n-ary product with zero-count optimization
                ensure_scratch_size(m);
                ensure_base_size(m);

                size_t zero_count = 0, zero_idx = 0;
                double prod_nz = 1.0;
                for (size_t j = 0; j < m; ++j) {
                    const double vj = u->inputs[j]->value;
                    g_scratch_values[j] = vj;
                    g_scratch_bases[j] = lanes_.base(u->inputs[j]->id);
                    if (vj == 0.0) {
                        if (++zero_count == 1)
                            zero_idx = j;
                    } else {
                        prod_nz *= vj;
                    }
                }

                for (size_t l = 0; l < L; ++l) {
                    double yd = 0.0;
                    if (zero_count == 0) {
                        for (size_t j = 0; j < m; ++j) {
                            const double dj =
                                lanes_.dot[g_scratch_bases[j] + l];
                            yd += dj * (prod_nz / g_scratch_values[j]);
                        }
                    } else if (zero_count == 1) {
                        const double dz =
                            lanes_.dot[g_scratch_bases[zero_idx] + l];
                        yd = dz * prod_nz;
                    }
                    // if zero_count >= 2, yd stays 0
                    lanes_.dot[ybase + l] = yd;
                }
            }
            break;
        }

        case Operator::Divide: {
            if (u->inputs.size() != 2)
                break;
            const int ia = u->inputs[0]->id, ib = u->inputs[1]->id;
            const size_t abase = lanes_.base(ia), bbase = lanes_.base(ib);
            const double aval = u->inputs[0]->value, bval = u->inputs[1]->value;
            if (bval == 0.0) {
                std::fill_n(&lanes_.dot[ybase], L, 0.0);
            } else {
                const double invb2 = 1.0 / (bval * bval);
                for (size_t l = 0; l < L; ++l) {
                    const double ad = lanes_.dot[abase + l];
                    const double bd = lanes_.dot[bbase + l];
                    lanes_.dot[ybase + l] = (ad * bval - aval * bd) * invb2;
                }
            }
            break;
        }

        case Operator::Sin: {
            if (u->inputs.size() != 1)
                break;
            const int ix = u->inputs[0]->id;
            const size_t xbase = lanes_.base(ix);
            const double cosx = std::cos(u->inputs[0]->value);
            for (size_t l = 0; l < L; ++l)
                lanes_.dot[ybase + l] = cosx * lanes_.dot[xbase + l];
            break;
        }

        case Operator::Cos: {
            if (u->inputs.size() != 1)
                break;
            const int ix = u->inputs[0]->id;
            const size_t xbase = lanes_.base(ix);
            const double sinx = std::sin(u->inputs[0]->value);
            for (size_t l = 0; l < L; ++l)
                lanes_.dot[ybase + l] = -sinx * lanes_.dot[xbase + l];
            break;
        }

        case Operator::Tan: {
            if (u->inputs.size() != 1)
                break;
            const int ix = u->inputs[0]->id;
            const size_t xbase = lanes_.base(ix);
            const double tx = std::tan(u->inputs[0]->value);
            const double sec2 = 1.0 + tx * tx;
            for (size_t l = 0; l < L; ++l)
                lanes_.dot[ybase + l] = sec2 * lanes_.dot[xbase + l];
            break;
        }

        case Operator::Exp: {
            if (u->inputs.size() != 1)
                break;
            const int ix = u->inputs[0]->id;
            const size_t xbase = lanes_.base(ix);
            const double ex = std::exp(u->inputs[0]->value);
            for (size_t l = 0; l < L; ++l)
                lanes_.dot[ybase + l] = ex * lanes_.dot[xbase + l];
            break;
        }

        case Operator::Log: {
            if (u->inputs.size() != 1)
                break;
            const int ix = u->inputs[0]->id;
            const size_t xbase = lanes_.base(ix);
            const double x = u->inputs[0]->value;
            if (x > 0.0) {
                for (size_t l = 0; l < L; ++l)
                    lanes_.dot[ybase + l] = lanes_.dot[xbase + l] / x;
            } else {
                std::fill_n(&lanes_.dot[ybase], L, 0.0);
            }
            break;
        }

        case Operator::Tanh: {
            if (u->inputs.size() != 1)
                break;
            const int ix = u->inputs[0]->id;
            const size_t xbase = lanes_.base(ix);
            const double th = std::tanh(u->inputs[0]->value);
            const double sech2 = 1.0 - th * th;
            for (size_t l = 0; l < L; ++l)
                lanes_.dot[ybase + l] = sech2 * lanes_.dot[xbase + l];
            break;
        }

        case Operator::Relu: {
            if (u->inputs.size() != 1)
                break;
            const int ix = u->inputs[0]->id;
            const size_t xbase = lanes_.base(ix);
            const double xv = u->inputs[0]->value;
            if (xv > 0.0) {
                std::copy_n(&lanes_.dot[xbase], L, &lanes_.dot[ybase]);
            } else {
                std::fill_n(&lanes_.dot[ybase], L, 0.0);
            }
            break;
        }

        case Operator::Max: {
            if (u->inputs.size() != 2)
                break;
            const int ia = u->inputs[0]->id, ib = u->inputs[1]->id;
            const size_t abase = lanes_.base(ia), bbase = lanes_.base(ib);
            const double a = u->inputs[0]->value, b = u->inputs[1]->value;
            if (a >= b) {
                std::copy_n(&lanes_.dot[abase], L, &lanes_.dot[ybase]);
            } else {
                std::copy_n(&lanes_.dot[bbase], L, &lanes_.dot[ybase]);
            }
            break;
        }

            // ---- New lane kernels ----

        case Operator::Gelu: {
            if (u->inputs.size() != 1)
                break;
            const int ix = u->inputs[0]->id;
            const size_t xbase = lanes_.base(ix);
            constexpr double inv_sqrt2 = 0.70710678118654752440;
            constexpr double inv_sqrt2pi = 0.39894228040143267794;
            const double x = u->inputs[0]->value;
            const double Phi = 0.5 * (1.0 + std::erf(x * inv_sqrt2));
            const double phi = inv_sqrt2pi * std::exp(-0.5 * x * x);
            const double f1 = Phi + x * phi; // f'(x)
            for (size_t l = 0; l < L; ++l)
                lanes_.dot[ybase + l] = f1 * lanes_.dot[xbase + l];
            break;
        }

        case Operator::Silu: {
            if (u->inputs.size() != 1)
                break;
            const int ix = u->inputs[0]->id;
            const size_t xbase = lanes_.base(ix);
            const double x = u->inputs[0]->value;
            const double s = 1.0 / (1.0 + std::exp(-x));
            const double sp = s * (1.0 - s);
            const double f1 = s + x * sp;
            for (size_t l = 0; l < L; ++l)
                lanes_.dot[ybase + l] = f1 * lanes_.dot[xbase + l];
            break;
        }

        case Operator::Softmax: {
            const size_t m = u->inputs.size();
            if (m == 0)
                break;

            ensure_scratch_size(m);
            // stable softmax over AoS primals
            double mmax = -std::numeric_limits<double>::infinity();
            for (size_t j = 0; j < m; ++j)
                mmax = std::max(mmax, u->inputs[j]->value);
            double Z = 0.0;
            for (size_t j = 0; j < m; ++j) {
                g_scratch_values[j] = std::exp(u->inputs[j]->value - mmax);
                Z += g_scratch_values[j];
            }
            for (size_t j = 0; j < m; ++j)
                g_scratch_values[j] /= Z; // s_j

            const size_t i = 0; // component index (first input)
            const double si = g_scratch_values[i];

            for (size_t l = 0; l < L; ++l) {
                double avg = 0.0;
                for (size_t j = 0; j < m; ++j)
                    avg += g_scratch_values[j] *
                           lanes_.dot[lanes_.base(u->inputs[j]->id) + l];
                const size_t ibase = lanes_.base(u->inputs[i]->id);
                const double di = lanes_.dot[ibase + l];
                lanes_.dot[ybase + l] = si * (di - avg);
            }
            break;
        }

        default:
            // no-op or add more kernels here
            break;
        }
    }
}

// Reverse lanes: uses precomputed AoS gradients (one scalar reverse
// pass done).
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
        const double w = u->gradient; // AoS gradient already computed

        switch (u->type) {
        case Operator::Var:
        case Operator::cte:
            break;

        case Operator::Add: {
            for (const auto &inp : u->inputs) {
                const size_t ibase = lanes_.base(inp->id);
                for (size_t l = 0; l < L; ++l)
                    lanes_.gdot[ibase + l] += lanes_.gdot[ybase + l];
            }
            break;
        }

        case Operator::Subtract: {
            if (u->inputs.size() != 2)
                break;
            const size_t abase = lanes_.base(u->inputs[0]->id);
            const size_t bbase = lanes_.base(u->inputs[1]->id);
            for (size_t l = 0; l < L; ++l) {
                const double gu = lanes_.gdot[ybase + l];
                lanes_.gdot[abase + l] += gu;
                lanes_.gdot[bbase + l] -= gu;
            }
            break;
        }
        case Operator::Multiply: {
            const size_t m = u->inputs.size();
            if (m == 0)
                break;

            if (m == 2) [[likely]] {
                const size_t abase = lanes_.base(u->inputs[0]->id);
                const size_t bbase = lanes_.base(u->inputs[1]->id);
                const double aval = u->inputs[0]->value,
                             bval = u->inputs[1]->value;
                for (size_t l = 0; l < L; ++l) {
                    const double gu = lanes_.gdot[ybase + l];
                    const double ad = lanes_.dot[abase + l];
                    const double bd = lanes_.dot[bbase + l];
                    lanes_.gdot[abase + l] += gu * bval + w * bd;
                    lanes_.gdot[bbase + l] += gu * aval + w * ad;
                }
            } else {
                ensure_scratch_size(m);
                ensure_base_size(m);

                size_t zero_count = 0, zero_idx = 0;
                double prod_nz = 1.0;
                for (size_t j = 0; j < m; ++j) {
                    const double vj = u->inputs[j]->value;
                    g_scratch_values[j] = vj;
                    g_scratch_bases[j] = lanes_.base(u->inputs[j]->id);
                    if (vj == 0.0) {
                        if (++zero_count == 1)
                            zero_idx = j;
                    } else {
                        prod_nz *= vj;
                    }
                }

                for (size_t l = 0; l < L; ++l) {
                    const double gu = lanes_.gdot[ybase + l];

                    if (zero_count == 0) {
                        // sum over dj / vj
                        double sum_d_over_v = 0.0;
                        for (size_t j = 0; j < m; ++j)
                            sum_d_over_v += lanes_.dot[g_scratch_bases[j] + l] /
                                            g_scratch_values[j];

                        for (size_t i = 0; i < m; ++i) {
                            const double vi = g_scratch_values[i];
                            const double di =
                                lanes_.dot[g_scratch_bases[i] + l];
                            lanes_.gdot[g_scratch_bases[i] + l] +=
                                gu * (prod_nz / vi) +
                                w * ((prod_nz / vi) * (sum_d_over_v - di / vi));
                        }
                    } else if (zero_count == 1) {
                        const double dz =
                            lanes_.dot[g_scratch_bases[zero_idx] + l];
                        lanes_.gdot[g_scratch_bases[zero_idx] + l] +=
                            gu * prod_nz;

                        for (size_t i = 0; i < m; ++i)
                            if (i != zero_idx) {
                                const double vi = g_scratch_values[i];
                                lanes_.gdot[g_scratch_bases[i] + l] +=
                                    w * (dz * (prod_nz / vi));
                            }
                    }
                    // if >=2 zeros: nothing to do
                }
            }
            break;
        }

        case Operator::Divide: {
            if (u->inputs.size() != 2)
                break;
            const size_t abase = lanes_.base(u->inputs[0]->id);
            const size_t bbase = lanes_.base(u->inputs[1]->id);
            const double aval = u->inputs[0]->value, bval = u->inputs[1]->value;
            if (bval != 0.0) {
                const double invb = 1.0 / bval;
                const double invb2 = invb * invb;
                const double invb3 = invb2 * invb;
                for (size_t l = 0; l < L; ++l) {
                    const double gu = lanes_.gdot[ybase + l];
                    const double ad = lanes_.dot[abase + l];
                    const double bd = lanes_.dot[bbase + l];
                    lanes_.gdot[abase + l] += gu * invb + w * (-bd * invb2);
                    lanes_.gdot[bbase + l] +=
                        gu * (-aval * invb2) +
                        w * ((-ad * bval + 2.0 * aval * bd) * invb3);
                }
            }
            break;
        }

        case Operator::Sin: {
            if (u->inputs.size() != 1)
                break;
            const size_t xbase = lanes_.base(u->inputs[0]->id);
            const double x = u->inputs[0]->value;
            const double c = std::cos(x), s = std::sin(x);
            for (size_t l = 0; l < L; ++l) {
                const double gu = lanes_.gdot[ybase + l];
                const double dx = lanes_.dot[xbase + l];
                lanes_.gdot[xbase + l] += gu * c + w * (-s) * dx;
            }
            break;
        }

        case Operator::Cos: {
            if (u->inputs.size() != 1)
                break;
            const size_t xbase = lanes_.base(u->inputs[0]->id);
            const double x = u->inputs[0]->value;
            const double s = std::sin(x), c = std::cos(x);
            for (size_t l = 0; l < L; ++l) {
                const double gu = lanes_.gdot[ybase + l];
                const double dx = lanes_.dot[xbase + l];
                lanes_.gdot[xbase + l] += gu * (-s) + w * (-c) * dx;
            }
            break;
        }

        case Operator::Tan: {
            if (u->inputs.size() != 1)
                break;
            const size_t xbase = lanes_.base(u->inputs[0]->id);
            const double x = u->inputs[0]->value;
            const double t = std::tan(x);
            const double sec2 = 1.0 + t * t;
            for (size_t l = 0; l < L; ++l) {
                const double gu = lanes_.gdot[ybase + l];
                const double dx = lanes_.dot[xbase + l];
                lanes_.gdot[xbase + l] += gu * sec2 + w * (2.0 * t * sec2) * dx;
            }
            break;
        }

        case Operator::Exp: {
            if (u->inputs.size() != 1)
                break;
            const size_t xbase = lanes_.base(u->inputs[0]->id);
            const double ex = std::exp(u->inputs[0]->value);
            for (size_t l = 0; l < L; ++l) {
                const double gu = lanes_.gdot[ybase + l];
                const double dx = lanes_.dot[xbase + l];
                lanes_.gdot[xbase + l] += gu * ex + w * ex * dx;
            }
            break;
        }

        case Operator::Log: {
            if (u->inputs.size() != 1)
                break;
            const size_t xbase = lanes_.base(u->inputs[0]->id);
            const double x = u->inputs[0]->value;
            if (x > 0.0) {
                const double invx = 1.0 / x;
                const double invx2 = invx * invx;
                for (size_t l = 0; l < L; ++l) {
                    const double gu = lanes_.gdot[ybase + l];
                    const double dx = lanes_.dot[xbase + l];
                    lanes_.gdot[xbase + l] += gu * invx + w * (-invx2) * dx;
                }
            }
            break;
        }

        case Operator::Tanh: {
            if (u->inputs.size() != 1)
                break;
            const size_t xbase = lanes_.base(u->inputs[0]->id);
            const double x = u->inputs[0]->value;
            const double t = std::tanh(x);
            const double sech2 = 1.0 - t * t;
            const double fpp = -2.0 * t * sech2;
            for (size_t l = 0; l < L; ++l) {
                const double gu = lanes_.gdot[ybase + l];
                const double dx = lanes_.dot[xbase + l];
                lanes_.gdot[xbase + l] += gu * sech2 + w * fpp * dx;
            }
            break;
        }

        case Operator::Relu: {
            if (u->inputs.size() != 1)
                break;
            const size_t xbase = lanes_.base(u->inputs[0]->id);
            const double x = u->inputs[0]->value;
            if (x > 0.0) {
                for (size_t l = 0; l < L; ++l)
                    lanes_.gdot[xbase + l] += lanes_.gdot[ybase + l];
            }
            break;
        }

        case Operator::Max: {
            if (u->inputs.size() != 2)
                break;
            const size_t abase = lanes_.base(u->inputs[0]->id);
            const size_t bbase = lanes_.base(u->inputs[1]->id);
            const double a = u->inputs[0]->value, b = u->inputs[1]->value;
            if (a >= b) {
                for (size_t l = 0; l < L; ++l)
                    lanes_.gdot[abase + l] += lanes_.gdot[ybase + l];
            } else {
                for (size_t l = 0; l < L; ++l)
                    lanes_.gdot[bbase + l] += lanes_.gdot[ybase + l];
            }
            break;
        }

            // ---- New lane kernels ----

        case Operator::Gelu: {
            if (u->inputs.size() != 1)
                break;
            const size_t xbase = lanes_.base(u->inputs[0]->id);
            constexpr double inv_sqrt2 = 0.70710678118654752440;
            constexpr double inv_sqrt2pi = 0.39894228040143267794;
            const double x = u->inputs[0]->value;
            const double Phi = 0.5 * (1.0 + std::erf(x * inv_sqrt2));
            const double phi = inv_sqrt2pi * std::exp(-0.5 * x * x);
            const double f1 = Phi + x * phi;
            const double f2 = phi * (2.0 - x * x); // f''(x)
            for (size_t l = 0; l < L; ++l) {
                const double gu = lanes_.gdot[ybase + l];
                const double dx = lanes_.dot[xbase + l];
                lanes_.gdot[xbase + l] += gu * f1 + w * f2 * dx;
            }
            break;
        }

        case Operator::Silu: {
            if (u->inputs.size() != 1)
                break;
            const size_t xbase = lanes_.base(u->inputs[0]->id);
            const double x = u->inputs[0]->value;
            const double s = 1.0 / (1.0 + std::exp(-x));
            const double sp = s * (1.0 - s);
            const double f1 = s + x * sp;
            const double f2 = 2.0 * sp + x * sp * (1.0 - 2.0 * s);
            for (size_t l = 0; l < L; ++l) {
                const double gu = lanes_.gdot[ybase + l];
                const double dx = lanes_.dot[xbase + l];
                lanes_.gdot[xbase + l] += gu * f1 + w * f2 * dx;
            }
            break;
        }

        case Operator::Softmax: {
            const size_t m = u->inputs.size();
            if (m == 0)
                break;

            ensure_scratch_size(m);
            // recompute softmax s_j from AoS primals
            double mmax = -std::numeric_limits<double>::infinity();
            for (size_t j = 0; j < m; ++j)
                mmax = std::max(mmax, u->inputs[j]->value);
            double Z = 0.0;
            for (size_t j = 0; j < m; ++j) {
                g_scratch_values[j] = std::exp(u->inputs[j]->value - mmax);
                Z += g_scratch_values[j];
            }
            for (size_t j = 0; j < m; ++j)
                g_scratch_values[j] /= Z;
            const size_t i = 0;
            const double si = g_scratch_values[i];

            for (size_t l = 0; l < L; ++l) {
                const double gu = lanes_.gdot[ybase + l];

                double avg = 0.0;
                for (size_t j = 0; j < m; ++j)
                    avg += g_scratch_values[j] *
                           lanes_.dot[lanes_.base(u->inputs[j]->id) + l];

                const size_t ibase = lanes_.base(u->inputs[i]->id);
                const double di = lanes_.dot[ibase + l];
                const double sdot_i = si * (di - avg);

                for (size_t j = 0; j < m; ++j) {
                    const size_t jbase = lanes_.base(u->inputs[j]->id);
                    const double sj = g_scratch_values[j];
                    const double dj = lanes_.dot[jbase + l];
                    const double sdot_j = sj * (dj - avg);

                    // gu * J + w * J'[d]
                    double add = gu * si * ((j == i) ? (1.0 - sj) : -sj);
                    if (j == i)
                        add += w * sdot_i * (1.0 - 2.0 * si);
                    else
                        add += w * (-sdot_i * sj - si * sdot_j);

                    lanes_.gdot[jbase + l] += add;
                }
            }
            break;
        }

        default:
            // no-op or add more kernels here
            break;
        }
    }
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
void ADGraph::hessianMultiVectorProduct(const ADNodePtr &outputNode,
                                        const double *V_ptr, size_t ldV,
                                        double *Y_ptr, size_t ldY, size_t k) {
    if (!outputNode)
        return;
    if (cache_.dirty)
        rebuildCache_();

    // Prepare variable orders
    initializeNodeVariables();
    const size_t nvars = nodeVariables.size();
    if (nvars == 0 || k == 0)
        return;

    // Configure lanes and buffers
    set_num_lanes(k);
    ensureLaneBuffers_();
    const size_t L = lanes();

    // 1) Reset lanes (no epoch writes)
    std::fill(lanes_.dot.begin(), lanes_.dot.end(), 0.0);
    std::fill(lanes_.gdot.begin(), lanes_.gdot.end(), 0.0);

    // 2) Load input multi-vector V into variable nodes' lane dots.
    //    Layout: element (varIndex i, lane l) at V_ptr[i*ldV + l]
    for (const auto &kv : nodeVariables) {
        const ADNodePtr &varNode = kv.second;
        if (!varNode)
            continue;
        const int ord = varNode->order;
        if (ord < 0)
            continue;

        const size_t vbase = lanes_.base(varNode->id);
        // copy k lanes into the row slice
        for (size_t l = 0; l < k; ++l) {
            lanes_.dot[vbase + l] = V_ptr[size_t(ord) * ldV + l];
        }
        // if L>k, leave the tail at 0.0 (std::fill already did)
    }

    // 3) Single scalar forward pass to compute primals (AoS)
    resetForwardPass();
    computeForwardPass(); // uses AoS values

    // 4) Lane forward pass (propagate dot across graph)
    computeForwardPassWithDotLanes();

    // 5) Scalar reverse pass to compute first-order gradients w.r.t vars
    resetGradients();
    set_epoch_value(outputNode->gradient, outputNode->grad_epoch,
                    cur_grad_epoch_, 1.0);
    initiateBackwardPassFused();

    // 6) Lane reverse pass to propagate grad-dot using (w = gradient) & dots
    // lanes_.gdot is already zeroed above; no epoch seeding necessary.
    resetGradDotAll(); // keep if it zeroes AoS grad_dot fields only
    initiateBackwardPassFusedLanes();

    // 7) Extract outputs for variable nodes into Y
    //    Layout: (i,l) -> Y_ptr[i*ldY + l]
    for (const auto &kv : nodeVariables) {
        const ADNodePtr &varNode = kv.second;
        if (!varNode)
            continue;
        const int ord = varNode->order;
        if (ord < 0)
            continue;

        const size_t gbase = lanes_.base(varNode->id);
        for (size_t l = 0; l < k; ++l) {
            Y_ptr[size_t(ord) * ldY + l] = lanes_.gdot[gbase + l];
        }
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
    markDirty_();
}

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