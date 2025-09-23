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
        // drop-in for n-ary Multiply forward lanes
        case Operator::Multiply: {
            const size_t m = u->inputs.size();
            if (m == 0)
                break;
            if (m == 2) { /* keep your fast path */
            } else {
                ensure_scratch_size(m);
                ensure_base_size(m);

                // Preload scalars and bases (lane-independent)
                size_t zc = 0, zidx = 0;
                for (size_t j = 0; j < m; ++j) {
                    const double vj = u->inputs[j]->value;
                    g_scratch_values[j] = vj;
                    g_scratch_bases[j] = lanes_.base(u->inputs[j]->id);
                    if (vj == 0.0 && (++zc == 1))
                        zidx = j;
                }

                if (zc >= 2) {
                    // All zero
                    std::fill_n(&lanes_.dot[ybase], L, 0.0);
                    break;
                }

                if (zc == 1) {
                    // Only the zero term contributes
                    double prod_nz = 1.0;
                    for (size_t j = 0; j < m; ++j)
                        if (j != zidx)
                            prod_nz *= g_scratch_values[j];
                    const size_t zb = g_scratch_bases[zidx];
                    for (size_t l = 0; l < L; ++l)
                        lanes_.dot[ybase + l] = lanes_.dot[zb + l] * prod_nz;
                    break;
                }

                // zc == 0: use prefix/suffix products (no division in the inner
                // loop) prefix[i] = v0*...*v_{i-1}, suffix[i] =
                // v_{i+1}*...*v_{m-1}
                ensure_scratch_size(3 * m);
                double *prefix = g_scratch_values.data(); // reuse block 0..m-1
                double *suffix =
                    g_scratch_values.data() + m; // reuse block m..2m-1
                double *contrib = g_scratch_values.data() +
                                  2 * m; // temp per i (lane-indep coeff)

                prefix[0] = 1.0;
                for (size_t i = 1; i < m; ++i)
                    prefix[i] = prefix[i - 1] * u->inputs[i - 1]->value;
                suffix[m - 1] = 1.0;
                for (size_t i = m - 1; i-- > 0;)
                    suffix[i] = suffix[i + 1] * u->inputs[i + 1]->value;

                for (size_t i = 0; i < m; ++i)
                    contrib[i] = prefix[i] * suffix[i];

                // lanes
                for (size_t l = 0; l < L; ++l) {
                    double yd = 0.0;
                    for (size_t i = 0; i < m; ++i)
                        yd += contrib[i] * lanes_.dot[g_scratch_bases[i] + l];
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
void ADGraph::hessianMultiVectorProduct(const ADNodePtr &y, const double *V,
                                        size_t ldV, double *Y, size_t ldY,
                                        size_t k) {
    if (!y || k == 0)
        return;
    if (cache_.dirty)
        rebuildCache_();
    initializeNodeVariables();
    const size_t n = nodeVariables.size();
    if (n == 0)
        return;

    set_num_lanes(k);
    ensureLaneBuffers_();

    // 1) zero lanes and load V into variable rows
    std::fill(lanes_.dot.begin(), lanes_.dot.end(), 0.0);
    std::fill(lanes_.gdot.begin(), lanes_.gdot.end(), 0.0);
    for (const auto &kv : nodeVariables) {
        const auto &var = kv.second;
        if (!var)
            continue;
        const int ord = var->order;
        if (ord < 0)
            continue;
        const size_t vbase = lanes_.base(var->id);
        for (size_t l = 0; l < k; ++l)
            lanes_.dot[vbase + l] = V[size_t(ord) * ldV + l];
    }

    // 2) fused scalar forward + lane forward
    resetForwardPass();
    computeForwardPassAndDotLanesTogether();

    // 3) scalar reverse to get w = ∂y/∂x
    resetGradients();
    set_epoch_value(y->gradient, y->grad_epoch, cur_grad_epoch_, 1.0);
    initiateBackwardPassFused();

    // 4) lane reverse using w and per-lane dots
    resetGradDotAll();
    initiateBackwardPassFusedLanes();

    // 5) gather Y
    for (const auto &kv : nodeVariables) {
        const auto &var = kv.second;
        if (!var)
            continue;
        const int ord = var->order;
        if (ord < 0)
            continue;
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
}

// In ADGraph.cpp

ADNodePtr ADGraph::adoptSubgraphAndReturnRoot(const ADNodePtr& root_src) {
    if (!root_src) return nullptr;

    // 1) Collect reachable nodes from root_src in topological order (post-order)
    struct PtrHash { size_t operator()(const ADNode* p) const noexcept {
        return std::hash<const void*>{}(p);
    }};
    struct PtrEq { bool operator()(const ADNode* a, const ADNode* b) const noexcept {
        return a == b;
    }};

    std::unordered_set<const ADNode*, PtrHash, PtrEq> visited;
    visited.reserve(512);
    std::vector<const ADNode*> order;
    order.reserve(512);

    std::function<void(const ADNode*)> dfs = [&](const ADNode* u) {
        if (!u || !visited.insert(u).second) return;
        for (const auto& in : u->inputs) dfs(in.get());
        order.push_back(u); // children first
    };
    dfs(root_src.get());

    // 2) Clone nodes without wiring inputs yet
    std::unordered_map<const ADNode*, ADNodePtr, PtrHash, PtrEq> clone;
    clone.reserve(order.size() * 2);

    for (const ADNode* u : order) {
        // Deep-copy the node metadata but NOT the inputs (we wire after)
        auto v = std::make_shared<ADNode>(*u);
        v->inputs.clear();

        // If you maintain per-graph IDs, assign a new one here (via addNode):
        addNode(v); // must set v->id and register in this graph

        // Optional: clean per-run caches/epochs if your ADNode has them
        // v->val_epoch = v->dot_epoch = v->grad_epoch = 0; etc.

        if (v->type == Operator::Var && !v->name.empty())
            nodeVariables[v->name] = v;

        clone.emplace(u, v);
    }

    // 3) Wire inputs to the cloned counterparts
    for (const ADNode* u : order) {
        ADNodePtr& v = clone[u];
        v->inputs.reserve(u->inputs.size());
        for (const auto& in : u->inputs) {
            auto it = clone.find(in.get());
            // in must be reachable (we built from DFS), assert for safety:
            if (it != clone.end()) v->inputs.push_back(it->second);
        }
        // If you keep any nodeIndex_ mapping by pointer, update it here
        // nodeIndex_[v.get()] = v;  // adjust to your layout
        markDirty_(v.get());
    }

    markDirty_();          // graph-level dirty
    rebuildCache_();       // cheap cache rebuild if you have one
    return clone[root_src.get()];
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

// Fused forward pass: compute primals AND propagate dot lanes in single
// traversal
void ADGraph::computeForwardPassAndDotLanesTogether() {
    if (cache_.dirty)
        rebuildCache_();

    const size_t L = lanes(); // lane count

    for (ADNode *u : cache_.topo) {
        if (!u)
            continue;

        const int uid = u->id;
        const size_t ybase = lanes_.base(uid);

        switch (u->type) {
        case Operator::Var:
        case Operator::cte:
            // Variables: value already set, dots already loaded
            // Constants: value set, dots remain 0
            break;

        case Operator::Add: {
            const size_t m = u->inputs.size();
            if (m == 0)
                break;

            // FUSED: Compute primal AND dot lanes together
            double sum_val = 0.0;
            std::fill_n(&lanes_.dot[ybase], L, 0.0);

            for (const auto &inp : u->inputs) {
                sum_val += inp->value; // Primal computation

                // Dot lane computation
                const size_t ibase = lanes_.base(inp->id);
                for (size_t l = 0; l < L; ++l)
                    lanes_.dot[ybase + l] += lanes_.dot[ibase + l];
            }

            u->value = sum_val; // Store primal result
            break;
        }

        case Operator::Subtract: {
            if (u->inputs.size() != 2)
                break;

            const ADNodePtr &a = u->inputs[0];
            const ADNodePtr &b = u->inputs[1];
            const int ia = a->id, ib = b->id;
            const size_t abase = lanes_.base(ia), bbase = lanes_.base(ib);

            // FUSED: Primal and dot computation
            u->value = a->value - b->value;

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
                const ADNodePtr &a = u->inputs[0];
                const ADNodePtr &b = u->inputs[1];
                const int ia = a->id, ib = b->id;
                const size_t abase = lanes_.base(ia), bbase = lanes_.base(ib);

                // FUSED: Primal and dot computation
                const double aval = a->value;
                const double bval = b->value;
                u->value = aval * bval;

                for (size_t l = 0; l < L; ++l) {
                    const double ad = lanes_.dot[abase + l];
                    const double bd = lanes_.dot[bbase + l];
                    lanes_.dot[ybase + l] = ad * bval + aval * bd;
                }
            } else {
                // N-ary multiply with fused computation
                ensure_scratch_size(m);
                ensure_base_size(m);

                size_t zero_count = 0, zero_idx = 0;
                double prod_val = 1.0;
                double prod_nz = 1.0;

                // Single pass: collect values, bases, and compute primal
                for (size_t j = 0; j < m; ++j) {
                    const double vj = u->inputs[j]->value;
                    g_scratch_values[j] = vj;
                    g_scratch_bases[j] = lanes_.base(u->inputs[j]->id);

                    prod_val *= vj; // Primal product

                    if (vj == 0.0) {
                        if (++zero_count == 1)
                            zero_idx = j;
                    } else {
                        prod_nz *= vj;
                    }
                }

                u->value = prod_val; // Store primal result

                // Dot computation using precomputed values
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
                    lanes_.dot[ybase + l] = yd;
                }
            }
            break;
        }

        case Operator::Divide: {
            if (u->inputs.size() != 2)
                break;

            const ADNodePtr &a = u->inputs[0];
            const ADNodePtr &b = u->inputs[1];
            const int ia = a->id, ib = b->id;
            const size_t abase = lanes_.base(ia), bbase = lanes_.base(ib);

            const double aval = a->value;
            const double bval = b->value;

            // FUSED: Primal and dot computation
            if (bval == 0.0) {
                u->value = std::numeric_limits<double>::infinity();
                std::fill_n(&lanes_.dot[ybase], L, 0.0);
            } else {
                u->value = aval / bval;
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

            const ADNodePtr &x = u->inputs[0];
            const int ix = x->id;
            const size_t xbase = lanes_.base(ix);

            const double xval = x->value;

            // FUSED: Compute sin and cos together (often optimized by compiler)
            const double sinx = std::sin(xval);
            const double cosx = std::cos(xval);

            u->value = sinx;

            for (size_t l = 0; l < L; ++l)
                lanes_.dot[ybase + l] = cosx * lanes_.dot[xbase + l];
            break;
        }

        case Operator::Cos: {
            if (u->inputs.size() != 1)
                break;

            const ADNodePtr &x = u->inputs[0];
            const int ix = x->id;
            const size_t xbase = lanes_.base(ix);

            const double xval = x->value;

            // FUSED: Compute cos and sin together
            const double cosx = std::cos(xval);
            const double sinx = std::sin(xval);

            u->value = cosx;

            for (size_t l = 0; l < L; ++l)
                lanes_.dot[ybase + l] = -sinx * lanes_.dot[xbase + l];
            break;
        }

        case Operator::Tan: {
            if (u->inputs.size() != 1)
                break;

            const ADNodePtr &x = u->inputs[0];
            const int ix = x->id;
            const size_t xbase = lanes_.base(ix);

            const double xval = x->value;
            const double tanx = std::tan(xval);

            u->value = tanx;

            const double sec2 = 1.0 + tanx * tanx;
            for (size_t l = 0; l < L; ++l)
                lanes_.dot[ybase + l] = sec2 * lanes_.dot[xbase + l];
            break;
        }

        case Operator::Exp: {
            if (u->inputs.size() != 1)
                break;

            const ADNodePtr &x = u->inputs[0];
            const int ix = x->id;
            const size_t xbase = lanes_.base(ix);

            const double xval = x->value;
            const double ex = std::exp(xval);

            u->value = ex; // Primal

            // Dot: both f(x) and f'(x) are exp(x)
            for (size_t l = 0; l < L; ++l)
                lanes_.dot[ybase + l] = ex * lanes_.dot[xbase + l];
            break;
        }

        case Operator::Log: {
            if (u->inputs.size() != 1)
                break;

            const ADNodePtr &x = u->inputs[0];
            const int ix = x->id;
            const size_t xbase = lanes_.base(ix);

            const double xval = x->value;

            u->value = std::log(xval);

            if (xval > 0.0) {
                const double invx = 1.0 / xval;
                for (size_t l = 0; l < L; ++l)
                    lanes_.dot[ybase + l] = invx * lanes_.dot[xbase + l];
            } else {
                std::fill_n(&lanes_.dot[ybase], L, 0.0);
            }
            break;
        }

        case Operator::Tanh: {
            if (u->inputs.size() != 1)
                break;

            const ADNodePtr &x = u->inputs[0];
            const int ix = x->id;
            const size_t xbase = lanes_.base(ix);

            const double xval = x->value;
            const double th = std::tanh(xval);

            u->value = th; // Primal

            const double sech2 = 1.0 - th * th; // Derivative
            for (size_t l = 0; l < L; ++l)
                lanes_.dot[ybase + l] = sech2 * lanes_.dot[xbase + l];
            break;
        }

        case Operator::Relu: {
            if (u->inputs.size() != 1)
                break;

            const ADNodePtr &x = u->inputs[0];
            const int ix = x->id;
            const size_t xbase = lanes_.base(ix);

            const double xval = x->value;

            // FUSED: Primal and dot computation
            if (xval > 0.0) {
                u->value = xval;
                std::copy_n(&lanes_.dot[xbase], L, &lanes_.dot[ybase]);
            } else {
                u->value = 0.0;
                std::fill_n(&lanes_.dot[ybase], L, 0.0);
            }
            break;
        }

        case Operator::Max: {
            if (u->inputs.size() != 2)
                break;

            const ADNodePtr &a = u->inputs[0];
            const ADNodePtr &b = u->inputs[1];
            const int ia = a->id, ib = b->id;
            const size_t abase = lanes_.base(ia), bbase = lanes_.base(ib);

            const double aval = a->value;
            const double bval = b->value;

            // FUSED: Primal and dot computation
            if (aval >= bval) {
                u->value = aval;
                std::copy_n(&lanes_.dot[abase], L, &lanes_.dot[ybase]);
            } else {
                u->value = bval;
                std::copy_n(&lanes_.dot[bbase], L, &lanes_.dot[ybase]);
            }
            break;
        }

        case Operator::Gelu: {
            if (u->inputs.size() != 1)
                break;

            const ADNodePtr &x = u->inputs[0];
            const int ix = x->id;
            const size_t xbase = lanes_.base(ix);

            constexpr double inv_sqrt2 = 0.70710678118654752440;
            constexpr double inv_sqrt2pi = 0.39894228040143267794;
            const double xval = x->value;

            // FUSED: Compute GELU and its derivative
            const double erf_term = std::erf(xval * inv_sqrt2);
            const double Phi = 0.5 * (1.0 + erf_term);
            const double phi = inv_sqrt2pi * std::exp(-0.5 * xval * xval);

            u->value = xval * Phi; // GELU(x) = x * Φ(x/√2)

            const double f1 = Phi + xval * phi; // f'(x)
            for (size_t l = 0; l < L; ++l)
                lanes_.dot[ybase + l] = f1 * lanes_.dot[xbase + l];
            break;
        }

        case Operator::Silu: {
            if (u->inputs.size() != 1)
                break;

            const ADNodePtr &x = u->inputs[0];
            const int ix = x->id;
            const size_t xbase = lanes_.base(ix);

            const double xval = x->value;

            // FUSED: Compute SiLU and its derivative
            const double sigmoid = 1.0 / (1.0 + std::exp(-xval));

            u->value = xval * sigmoid; // SiLU(x) = x * σ(x)

            const double sigmoid_prime = sigmoid * (1.0 - sigmoid);
            const double f1 = sigmoid + xval * sigmoid_prime; // f'(x)
            for (size_t l = 0; l < L; ++l)
                lanes_.dot[ybase + l] = f1 * lanes_.dot[xbase + l];
            break;
        }

        case Operator::Softmax: {
            const size_t m = u->inputs.size();
            if (m == 0)
                break;

            ensure_scratch_size(m);

            // FUSED: Compute softmax and prepare for dot computation
            // First pass: find max for numerical stability
            double mmax = -std::numeric_limits<double>::infinity();
            for (size_t j = 0; j < m; ++j)
                mmax = std::max(mmax, u->inputs[j]->value);

            // Second pass: compute exp(x_i - max) and sum
            double Z = 0.0;
            for (size_t j = 0; j < m; ++j) {
                g_scratch_values[j] = std::exp(u->inputs[j]->value - mmax);
                Z += g_scratch_values[j];
            }

            // Third pass: normalize to get softmax values
            for (size_t j = 0; j < m; ++j)
                g_scratch_values[j] /= Z; // s_j = softmax values

            // Store primal result (assuming this is component 0)
            const size_t i = 0; // component index (first input)
            u->value = g_scratch_values[i];

            // Dot computation for softmax derivative
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

            // case Operator::Abs: {
            //     if (u->inputs.size() != 1) break;

            //     const ADNodePtr &x = u->inputs[0];
            //     const int ix = x->id;
            //     const size_t xbase = lanes_.base(ix);

            //     const double xval = x->value;

            //     // FUSED: Primal and dot computation
            //     u->value = std::abs(xval);

            //     if (xval > 0.0) {
            //         std::copy_n(&lanes_.dot[xbase], L, &lanes_.dot[ybase]);
            //     } else if (xval < 0.0) {
            //         for (size_t l = 0; l < L; ++l)
            //             lanes_.dot[ybase + l] = -lanes_.dot[xbase + l];
            //     } else {
            //         // At x=0, abs is not differentiable; use 0 derivative
            //         std::fill_n(&lanes_.dot[ybase], L, 0.0);
            //     }
            //     break;
            // }

            // case Operator::Sqrt: {
            //     if (u->inputs.size() != 1) break;

            //     const ADNodePtr &x = u->inputs[0];
            //     const int ix = x->id;
            //     const size_t xbase = lanes_.base(ix);

            //     const double xval = x->value;

            //     if (xval > 0.0) {
            //         const double sqrtx = std::sqrt(xval);
            //         u->value = sqrtx;

            //         const double deriv = 0.5 / sqrtx; // d/dx sqrt(x) =
            //         1/(2*sqrt(x)) for (size_t l = 0; l < L; ++l)
            //             lanes_.dot[ybase + l] = deriv * lanes_.dot[xbase +
            //             l];
            //     } else {
            //         u->value = 0.0;
            //         std::fill_n(&lanes_.dot[ybase], L, 0.0);
            //     }
            //     break;
            // }

            // case Operator::Pow: {
            //     if (u->inputs.size() != 2) break;

            //     const ADNodePtr &base = u->inputs[0];
            //     const ADNodePtr &exponent = u->inputs[1];
            //     const int ibase = base->id, iexp = exponent->id;
            //     const size_t base_base = lanes_.base(ibase), exp_base =
            //     lanes_.base(iexp);

            //     const double b = base->value;
            //     const double e = exponent->value;

            //     if (b > 0.0) {
            //         const double result = std::pow(b, e);
            //         u->value = result;

            //         // For z = b^e, dz/db = e * b^(e-1), dz/de = b^e * ln(b)
            //         const double db_coeff = e * std::pow(b, e - 1.0);
            //         const double de_coeff = result * std::log(b);

            //         for (size_t l = 0; l < L; ++l) {
            //             const double db = lanes_.dot[base_base + l];
            //             const double de = lanes_.dot[exp_base + l];
            //             lanes_.dot[ybase + l] = db_coeff * db + de_coeff *
            //             de;
            //         }
            //     } else {
            //         u->value = 0.0;
            //         std::fill_n(&lanes_.dot[ybase], L, 0.0);
            //     }
            //     break;
            // }

            // case Operator::Sigmoid: {
            //     if (u->inputs.size() != 1) break;

            //     const ADNodePtr &x = u->inputs[0];
            //     const int ix = x->id;
            //     const size_t xbase = lanes_.base(ix);

            //     const double xval = x->value;
            //     const double sigmoid = 1.0 / (1.0 + std::exp(-xval));

            //     u->value = sigmoid;

            //     const double deriv = sigmoid * (1.0 - sigmoid);
            //     for (size_t l = 0; l < L; ++l)
            //         lanes_.dot[ybase + l] = deriv * lanes_.dot[xbase + l];
            //     break;
            // }

            // case Operator::LeakyRelu: {
            //     if (u->inputs.size() != 1) break;

            //     const ADNodePtr &x = u->inputs[0];
            //     const int ix = x->id;
            //     const size_t xbase = lanes_.base(ix);

            //     const double xval = x->value;
            //     constexpr double alpha = 0.01; // typical leaky ReLU slope

            //     // FUSED: Primal and dot computation
            //     if (xval > 0.0) {
            //         u->value = xval;
            //         std::copy_n(&lanes_.dot[xbase], L, &lanes_.dot[ybase]);
            //     } else {
            //         u->value = alpha * xval;
            //         for (size_t l = 0; l < L; ++l)
            //             lanes_.dot[ybase + l] = alpha * lanes_.dot[xbase +
            //             l];
            //     }
            //     break;
            // }

        default:
            // Fall back to separate computation for unimplemented ops
            _dispatch_op(*u, ForwardFunctor{*this, *u});
            // Note: This will NOT compute dot lanes for unknown ops
            // You may want to add a warning or handle this case differently
            std::fill_n(&lanes_.dot[ybase], L, 0.0);
            break;
        }
    }

    if (cache_.topo.size() != nodes.size()) [[unlikely]]
        std::cerr << "Warning: cycle or dangling inputs in AD graph.\n";
}
// Call this after adoptSubgraph(...) or right before heavy evaluation.
// Centralized symbolic simplification for ADGraph class
void ADGraph::simplifyExpression(std::vector<ADNodePtr>& outputs) {
    auto is_commutative = [](Operator op) -> bool {
        return op == Operator::Add || op == Operator::Multiply || op == Operator::Max;
    };

    auto make_const = [&](double v) -> ADNodePtr {
        auto c = std::make_shared<ADNode>();
        c->type = Operator::cte;
        c->value = v;
        addNode(c);
        return c;
    };

    auto rebuild_ptr_map = [&]() -> std::unordered_map<ADNode*, ADNodePtr> {
        std::unordered_map<ADNode*, ADNodePtr> m;
        m.reserve(nodes.size() * 2);
        for (auto &sp : nodes) if (sp) m.emplace(sp.get(), sp);
        return m;
    };

    auto make_id_maps = [&]() {
        std::unordered_map<ADNode*, int> idx;
        idx.reserve(nodes.size() * 2);
        for (int i = 0; i < (int)nodes.size(); ++i)
            if (nodes[i]) idx[nodes[i].get()] = i;
        return idx;
    };

    auto get_stable_id = [&](ADNode *u,
                             const std::unordered_map<ADNode*, int> &ptr2idx) -> int {
        if (u && u->id >= 0) return u->id; // prefer stable id if you have one
        auto it = ptr2idx.find(u);
        return (it == ptr2idx.end()) ? -1 : it->second;
    };

    auto build_signature = [&](ADNode *u,
                               const std::unordered_map<ADNode*, int> &ptr2idx) -> std::string {
        std::string s;
        s.reserve(64);
        s.append(std::to_string((int)u->type)).push_back('|');

        if (u->type == Operator::cte) {
            s.append("c|").append(std::to_string(u->value));
            return s;
        }
        if (u->type == Operator::Var) {
            s.append("v|").append(std::to_string(get_stable_id(u, ptr2idx)));
            return s;
        }

        s.append(std::to_string((int)u->inputs.size())).push_back('|');
        std::vector<int> child_ids;
        child_ids.reserve(u->inputs.size());
        for (auto &inp : u->inputs) child_ids.push_back(get_stable_id(inp.get(), ptr2idx));
        if (is_commutative(u->type)) std::sort(child_ids.begin(), child_ids.end());
        for (int id : child_ids) { s.append(std::to_string(id)); s.push_back(','); }
        return s;
    };

    auto replace_uses = [&](ADNode *from,
                            ADNode *to,
                            const std::unordered_map<ADNode*, ADNodePtr> &ptr2sp) {
        auto it_sp = ptr2sp.find(to);
        if (it_sp == ptr2sp.end()) return;
        const ADNodePtr &to_sp = it_sp->second;

        // Update all node inputs
        for (auto &n : nodes) {
            if (!n) continue;
            bool touched = false;
            for (auto &inp : n->inputs) {
                if (inp && inp.get() == from) { inp = to_sp; touched = true; }
            }
            if (touched) markDirty_(n.get());
        }
    };

    // Also patch the outputs if they point to `from`
    auto replace_in_outputs = [&](ADNode *from,
                                  ADNode *to,
                                  const std::unordered_map<ADNode*, ADNodePtr> &ptr2sp) {
        auto it_sp = ptr2sp.find(to);
        if (it_sp == ptr2sp.end()) return;
        const ADNodePtr &to_sp = it_sp->second;

        for (auto &r : outputs) {
            if (r && r.get() == from) r = to_sp;
        }
    };

    bool changed = true;
    int iterations = 0;
    const int maxIterations = 10;

    while (changed && iterations < maxIterations) {
        changed = false;
        iterations++;

        if (cache_.dirty) rebuildCache_();

        auto ptr2sp  = rebuild_ptr_map();
        auto ptr2idx = make_id_maps();
        std::unordered_map<std::string, ADNode*> cse_map; // fresh per pass

        std::vector<ADNode*> topo = cache_.topo;

        for (ADNode *u : topo) {
            if (!u) continue;
            if (ptr2sp.find(u) == ptr2sp.end()) continue; // node was removed

            // ---------- Constant folding ----------
            auto all_const = [&]() {
                if (u->inputs.empty()) return false;
                for (auto &inp : u->inputs) if (!inp || inp->type != Operator::cte) return false;
                return true;
            }();

            if (all_const) {
                double result = 0.0;
                bool canFold  = true;
                auto arity_is = [&](size_t k){ return u->inputs.size() == k; };
                auto in = [&](int i){ return u->inputs[i]->value; };

                switch (u->type) {
                    case Operator::Add: { double s=0.0; for (auto &i:u->inputs) s+=i->value; result=s; break; }
                    case Operator::Multiply: { double p=1.0; for (auto &i:u->inputs) p*=i->value; result=p; break; }
                    case Operator::Subtract: if (arity_is(2)) result = in(0)-in(1); else canFold=false; break;
                    case Operator::Divide:   if (arity_is(2) && in(1)!=0.0) result = in(0)/in(1); else canFold=false; break;
                    case Operator::Max:      if (arity_is(2)) result = std::max(in(0),in(1)); else canFold=false; break;
                    case Operator::Sin:      if (arity_is(1)) result = std::sin(in(0)); else canFold=false; break;
                    case Operator::Cos:      if (arity_is(1)) result = std::cos(in(0)); else canFold=false; break;
                    case Operator::Tan:      if (arity_is(1)) result = std::tan(in(0)); else canFold=false; break;
                    case Operator::Exp:      if (arity_is(1)) result = std::exp(in(0)); else canFold=false; break;
                    case Operator::Log:      if (arity_is(1) && in(0)>0.0) result = std::log(in(0)); else canFold=false; break;
                    case Operator::Tanh:     if (arity_is(1)) result = std::tanh(in(0)); else canFold=false; break;
                    case Operator::Relu:     if (arity_is(1)) result = std::max(0.0,in(0)); else canFold=false; break;
                    case Operator::Gelu:     if (arity_is(1)) { double x=in(0); result = x*0.5*(1.0+std::erf(x*0.7071067811865475)); } else canFold=false; break;
                    case Operator::Silu:     if (arity_is(1)) { double x=in(0); result = x/(1.0+std::exp(-x)); } else canFold=false; break;
                    default: canFold=false;
                }

                if (canFold) {
                    u->type = Operator::cte;
                    u->value = result;
                    u->inputs.clear();
                    markDirty_(u);
                    changed = true;
                    continue;
                }
            }

            // ---------- Algebraic simplifications ----------
            auto replace_with = [&](const ADNodePtr &src) {
                u->type = src->type;
                u->value = src->value;
                u->inputs = src->inputs;
                markDirty_(u);
                changed = true;
            };

            switch (u->type) {
                case Operator::Add: {
                    if (u->inputs.empty()) {
                        u->type = Operator::cte; u->value = 0.0; u->inputs.clear();
                        markDirty_(u); changed = true; break;
                    }
                    std::vector<ADNodePtr> newInputs;
                    newInputs.reserve(u->inputs.size());
                    double constSum = 0.0; bool hasConst = false;

                    for (auto &inp : u->inputs) {
                        if (!inp) continue;
                        if (inp->type == Operator::cte) {
                            constSum += inp->value; hasConst = true;
                        } else if (inp->type == Operator::Add) {
                            for (auto &nested : inp->inputs) if (nested) newInputs.push_back(nested);
                            changed = true;
                        } else {
                            newInputs.push_back(inp);
                        }
                    }
                    if (hasConst && constSum != 0.0) {
                        newInputs.push_back(make_const(constSum));
                        changed = true;
                    }
                    std::unordered_map<ADNode*, int> counts;
                    counts.reserve(newInputs.size()*2);
                    for (auto &inp : newInputs) counts[inp.get()]++;
                    std::vector<ADNodePtr> merged; merged.reserve(newInputs.size());
                    for (auto &kv : counts) {
                        ADNode *term = kv.first; int c = kv.second;
                        if (c == 1) {
                            merged.push_back(ptr2sp[term]);
                        } else {
                            auto coeff = make_const((double)c);
                            auto mult  = std::make_shared<ADNode>();
                            mult->type = Operator::Multiply;
                            mult->inputs = {coeff, ptr2sp[term]};
                            addNode(mult);
                            merged.push_back(mult);
                            changed = true;
                        }
                    }
                    if (merged.empty()) {
                        u->type = Operator::cte; u->value = 0.0; u->inputs.clear();
                        markDirty_(u); changed = true;
                    } else if (merged.size() == 1) {
                        replace_with(merged[0]);
                    } else if (merged != u->inputs) {
                        u->inputs = std::move(merged);
                        markDirty_(u); changed = true;
                    }
                    break;
                }

                case Operator::Multiply: {
                    if (u->inputs.empty()) {
                        u->type = Operator::cte; u->value = 1.0; u->inputs.clear();
                        markDirty_(u); changed = true; break;
                    }
                    std::vector<ADNodePtr> newInputs;
                    newInputs.reserve(u->inputs.size());
                    double constProd = 1.0; bool hasConst=false, hasZero=false;

                    for (auto &inp : u->inputs) {
                        if (!inp) continue;
                        if (inp->type == Operator::cte) {
                            if (inp->value == 0.0) { hasZero = true; break; }
                            if (inp->value != 1.0) { constProd *= inp->value; hasConst = true; }
                            if (inp->value != 1.0) changed = true;
                        } else if (inp->type == Operator::Multiply) {
                            for (auto &nested : inp->inputs) if (nested) newInputs.push_back(nested);
                            changed = true;
                        } else {
                            newInputs.push_back(inp);
                        }
                    }
                    if (hasZero) {
                        u->type = Operator::cte; u->value = 0.0; u->inputs.clear();
                        markDirty_(u); changed = true; break;
                    }
                    if (hasConst && constProd != 1.0) {
                        newInputs.insert(newInputs.begin(), make_const(constProd));
                        changed = true;
                    }
                    if (newInputs.empty()) {
                        u->type = Operator::cte; u->value = 1.0; u->inputs.clear();
                        markDirty_(u); changed = true;
                    } else if (newInputs.size() == 1) {
                        replace_with(newInputs[0]);
                    } else if (newInputs != u->inputs) {
                        u->inputs = std::move(newInputs);
                        markDirty_(u); changed = true;
                    }
                    break;
                }

                case Operator::Subtract: {
                    if (u->inputs.size() == 2) {
                        auto &a = u->inputs[0]; auto &b = u->inputs[1];
                        if (b && b->type == Operator::cte && b->value == 0.0) {
                            replace_with(a);
                        } else if (a && b && a.get() == b.get()) {
                            u->type = Operator::cte; u->value = 0.0; u->inputs.clear();
                            markDirty_(u); changed = true;
                        }
                    }
                    break;
                }

                case Operator::Divide: {
                    if (u->inputs.size() == 2) {
                        auto &a = u->inputs[0]; auto &b = u->inputs[1];
                        if (b && b->type == Operator::cte && b->value == 1.0) {
                            replace_with(a);
                        } else if (a && a->type == Operator::cte && a->value == 0.0) {
                            u->type = Operator::cte; u->value = 0.0; u->inputs.clear();
                            markDirty_(u); changed = true;
                        } else if (a && b && a.get() == b.get()) {
                            u->type = Operator::cte; u->value = 1.0; u->inputs.clear();
                            markDirty_(u); changed = true;
                        }
                    }
                    break;
                }

                case Operator::Sin:
                    if (u->inputs.size() == 1) {
                        auto &x = u->inputs[0];
                        if (x && x->type == Operator::cte && x->value == 0.0) {
                            u->type = Operator::cte; u->value = 0.0; u->inputs.clear();
                            markDirty_(u); changed = true;
                        }
                    }
                    break;

                case Operator::Cos:
                    if (u->inputs.size() == 1) {
                        auto &x = u->inputs[0];
                        if (x && x->type == Operator::cte && x->value == 0.0) {
                            u->type = Operator::cte; u->value = 1.0; u->inputs.clear();
                            markDirty_(u); changed = true;
                        }
                    }
                    break;

                case Operator::Exp:
                    if (u->inputs.size() == 1) {
                        auto &x = u->inputs[0];
                        if (x && x->type == Operator::cte && x->value == 0.0) {
                            u->type = Operator::cte; u->value = 1.0; u->inputs.clear();
                            markDirty_(u); changed = true;
                        } else if (x && x->type == Operator::Log && x->inputs.size() == 1) {
                            replace_with(x->inputs[0]); // exp(log(y)) = y
                        }
                    }
                    break;

                case Operator::Log:
                    if (u->inputs.size() == 1) {
                        auto &x = u->inputs[0];
                        if (x && x->type == Operator::cte && x->value == 1.0) {
                            u->type = Operator::cte; u->value = 0.0; u->inputs.clear();
                            markDirty_(u); changed = true;
                        } else if (x && x->type == Operator::Exp && x->inputs.size() == 1) {
                            replace_with(x->inputs[0]); // log(exp(y)) = y
                        }
                    }
                    break;

                default: break;
            }

            // ---------- CSE (and root remap) ----------
            if (u->type != Operator::Var && u->type != Operator::cte) {
                std::string sig = build_signature(u, ptr2idx);
                auto it = cse_map.find(sig);
                if (it != cse_map.end() && it->second != u) {
                    ADNode *canonical = it->second;
                    replace_uses(u, canonical, ptr2sp);
                    replace_in_outputs(u, canonical, ptr2sp); // keep outputs pointing to the canonical
                    changed = true;
                } else {
                    cse_map.emplace(std::move(sig), u);
                }
            }
        }

        if (changed) rebuildCacheFull_();
    }

    // ---------- Final cleanup: keep everything reachable from outputs ----------
    std::unordered_set<ADNode*> reachable;
    reachable.reserve(nodes.size() * 2);

    std::function<void(ADNode*)> dfs = [&](ADNode *node) {
        if (!node || reachable.count(node)) return;
        reachable.insert(node);
        for (auto &inp : node->inputs) if (inp) dfs(inp.get());
    };

    // anchor from outputs
    for (auto &r : outputs) if (r) dfs(r.get());
    // also keep your original anchors if you want
    for (const auto &kv : nodeVariables) if (kv.second) dfs(kv.second.get());
    for (const auto &kv : nodeIndex_)    if (kv.second) dfs(kv.second.get());

    nodes.erase(std::remove_if(nodes.begin(), nodes.end(),
                   [&](const ADNodePtr &sp) {
                       return sp && (reachable.find(sp.get()) == reachable.end());
                   }),
                nodes.end());

    rebuildCacheFull_();
}

