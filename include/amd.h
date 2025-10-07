// amd_array_qg_sota.hpp (single translation unit demo, SoTA-leaning tweaks)
// Build: g++ -O3 -march=native -std=c++17 amd_array_qg_sota.hpp -o amd && ./amd
// (Optional) parallel loops: add -fopenmp

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

//------------------------------------------------------------------------------
// Types
//------------------------------------------------------------------------------
using i32 = int32_t;
using u32 = uint32_t;
using i64 = int64_t;

//------------------------------------------------------------------------------
// Utilities
//------------------------------------------------------------------------------
static inline u32 mix32_(u32 x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static inline uint64_t mix64_(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

template <class T> static inline void dedup_sorted_inplace(std::vector<T> &a) {
    if (a.empty())
        return;
    auto it = std::unique(a.begin(), a.end());
    a.erase(it, a.end());
}

//------------------------------------------------------------------------------
// CSR (pattern-only)
//------------------------------------------------------------------------------
struct CSR {
    i32 n{0};
    std::vector<i32> indptr;  // size n+1
    std::vector<i32> indices; // size nnz

    CSR() = default;
    explicit CSR(i32 n_) : n(n_), indptr(n_ + 1, 0) {}
    i32 nnz() const { return (i32)indices.size(); }

    // Fast strictly upper of A ∪ Aᵀ (i<j), allocation-light, optionally
    // parallel
    CSR strict_upper_union_transpose() const {
        const i32 N = n;
        const auto &AI = indptr;
        const auto &AJ = indices;

        std::vector<i32> cnt(N, 0);

        for (i32 i = 0; i < N; ++i) {
            for (i32 p = AI[i]; p < AI[i + 1]; ++p) {
                const i32 j = AJ[p];
                if (i == j)
                    continue;
                const i32 a = (i < j) ? i : j;
                ++cnt[a];
            }
        }

        CSR U(N);
        U.indptr.assign(N + 1, 0);
        for (i32 i = 0; i < N; ++i)
            U.indptr[i + 1] = U.indptr[i] + cnt[i];
        U.indices.resize(U.indptr.back());
        std::vector<i32> wr(N);
        for (i32 i = 0; i < N; ++i)
            wr[i] = U.indptr[i];

        for (i32 i = 0; i < N; ++i) {
            for (i32 p = AI[i]; p < AI[i + 1]; ++p) {
                const i32 j = AJ[p];
                if (i == j)
                    continue;
                const i32 a = (i < j) ? i : j;
                const i32 b = (i < j) ? j : i;
                U.indices[wr[a]++] = b;
            }
            auto beg = U.indices.begin() + U.indptr[i];
            auto end = U.indices.begin() + wr[i];
            std::sort(beg, end);
            end = std::unique(beg, end);
            wr[i] = (i32)std::distance(U.indices.begin(), end);
        }

        // Compact if rows shrank after unique
        if (wr.back() != (i32)U.indices.size()) {
            std::vector<i32> nip(N + 1, 0);
            for (i32 i = 0; i < N; ++i)
                nip[i + 1] = nip[i] + (wr[i] - U.indptr[i]);
            std::vector<i32> nidx(nip.back());
            for (i32 i = 0; i < N; ++i) {
                const i32 rb = U.indptr[i], re = wr[i], outb = nip[i];
                std::copy(U.indices.begin() + rb, U.indices.begin() + re,
                          nidx.begin() + outb);
            }
            U.indptr.swap(nip);
            U.indices.swap(nidx);
        }

        return U;
    }
};

// inverse permutation (p[new]=old) → ip[old]=new
static std::vector<i32> inverse_permutation(const std::vector<i32> &p) {
    std::vector<i32> ip(p.size());
    for (i32 i = 0; i < (i32)p.size(); ++i)
        ip[p[i]] = i;
    return ip;
}

//------------------------------------------------------------------------------
// Stats
//------------------------------------------------------------------------------
struct AMDStats {
    i32 original_nnz{0};
    i32 original_bandwidth{0};
    i32 reordered_bandwidth{0};
    double bandwidth_reduction{0.0};
    i32 matrix_size{0};
    std::vector<i32> inverse_permutation;
    // extras
    i32 absorbed_elements{0};
    i32 coalesced_variables{0};
    i32 iw_capacity_peak{0};
};

//------------------------------------------------------------------------------
// AMD (array-based) with SoTA-leaning micro-architecture
//------------------------------------------------------------------------------
class AMDReorderingArray {
public:
    explicit AMDReorderingArray(bool aggressive_absorption = true,
                                int dense_cutoff = -1)
        : aggressive_absorption_(aggressive_absorption),
          dense_cutoff_(dense_cutoff) {}

    // Main API: permutation p[new] = old
    std::vector<i32> amd_order(const CSR &A, bool symmetrize = true) {
        CSR Aup = symmetrize ? A.strict_upper_union_transpose() : A;
        n_ = Aup.n;
        if (n_ == 0)
            return {};
        initialize_from_upper_(Aup);
        initialize_buckets_();
        eliminate_all_();
        return perm_;
    }

    std::pair<std::vector<i32>, AMDStats>
    compute_fill_reducing_permutation(const CSR &A, bool symmetrize = true) {
        AMDStats st;
        st.original_nnz = A.nnz();
        st.original_bandwidth = bandwidth_(A);
        auto p = amd_order(A, symmetrize);
        CSR Apr = permute_(A, p);
        st.reordered_bandwidth = bandwidth_(Apr);
        st.bandwidth_reduction =
            (st.original_bandwidth == 0)
                ? 0.0
                : double(st.original_bandwidth - st.reordered_bandwidth) /
                      double(st.original_bandwidth);
        st.matrix_size = A.n;
        st.inverse_permutation = inverse_permutation(p);
        st.absorbed_elements = stats_absorbed_;
        st.coalesced_variables = stats_coalesced_;
        st.iw_capacity_peak = stats_iw_peak_;
        return {p, st};
    }

private:
    // Problem size / storage
    i32 n_{0}, nsym_{0}, nzmax_{0};
    std::vector<i32> pe_, len_, elen_, iw_, nv_, degree_, w_, last_;
    std::vector<i32> head_, next_, prev_, where_;
    std::vector<char> var_active_, elem_active_;
    i32 mindeg_{0};
    i32 nelem_top_{0};
    // outputs/work
    std::vector<i32> order_, perm_, dense_queue_;
    std::vector<std::vector<i32>> sv_members_;
    std::vector<char> in_order_;
    i32 tail_used_{0};
    i32 wflg_{1};
    // options
    bool aggressive_absorption_{true};
    int dense_cutoff_{-1};
    // stats
    i32 stats_absorbed_{0};
    i32 stats_coalesced_{0};
    i32 stats_iw_peak_{0};

    // Initialization from upper pattern
    void initialize_from_upper_(const CSR &U) {
        const i32 n = U.n;
        const auto &indptr = U.indptr;
        const auto &indices = U.indices;
        const i32 m = U.nnz();

        // generous first guess to reduce realloc
        const i32 est_undir = 2 * m;
        nzmax_ = std::max({est_undir + n, (i32)(1.5 * est_undir) + 4 * n, 1});
        nsym_ = n + std::max(8, n);

        pe_.assign(nsym_, 0);
        len_.assign(nsym_, 0);
        elen_.assign(nsym_, -2); // -2 dead, -1 variable, >=0 element
        iw_.assign(nzmax_, 0);
        nv_.assign(nsym_, 1);
        degree_.assign(nsym_, 0);
        w_.assign(nsym_, 0);
        last_.assign(nsym_, -1);
        var_active_.assign(nsym_, 0);
        elem_active_.assign(nsym_, 0);

        std::vector<i32> deg(n, 0);
        for (i32 i = 0; i < n; ++i) {
            for (i32 p = indptr[i]; p < indptr[i + 1]; ++p) {
                const i32 j = indices[p]; // j > i
                ++deg[i];
                ++deg[j];
            }
        }

        i32 pos = 0;
        for (i32 i = 0; i < n; ++i) {
            pe_[i] = pos;
            len_[i] = deg[i];
            degree_[i] = deg[i];
            elen_[i] = -1;
            var_active_[i] = 1;
            pos += deg[i];
        }
        if (pos > nzmax_)
            grow_iw_(pos);

        std::vector<i32> wr(pe_.begin(), pe_.begin() + n);
        for (i32 i = 0; i < n; ++i) {
            for (i32 p = indptr[i]; p < indptr[i + 1]; ++p) {
                const i32 j = indices[p]; // i<j
                iw_[wr[i]++] = j;
                iw_[wr[j]++] = i;
            }
        }
        tail_used_ = pos;

        for (i32 e = n; e < nsym_; ++e) {
            pe_[e] = -1;
            len_[e] = 0;
            elen_[e] = -2;
            nv_[e] = 0;
            elem_active_[e] = 0;
            var_active_[e] = 0;
        }
        nelem_top_ = n;
        perm_.assign(n, -1);
        order_.clear();
        dense_queue_.clear();
        wflg_ = 1;

        sv_members_.assign(n, {});
        for (i32 i = 0; i < n; ++i)
            sv_members_[i] = {i};
        in_order_.assign(n, 0);

        stats_absorbed_ = 0;
        stats_coalesced_ = 0;
        stats_iw_peak_ = nzmax_;
    }

    void initialize_buckets_() {
        const i32 n = n_;
        head_.assign(n + 1, -1);
        next_.assign(nsym_, -1);
        prev_.assign(nsym_, -1);
        where_.assign(nsym_, -1);

        mindeg_ = n;
        for (i32 i = 0; i < n; ++i)
            if (var_active_[i]) {
                const i32 d = std::max(0, std::min(n, degree_[i]));
                bucket_insert_front_(i, d);
                if (d < mindeg_)
                    mindeg_ = d;
            }
        apply_dense_postponement_();
    }

    // Dense postponement: refined threshold
    void apply_dense_postponement_() {
        const i32 n = n_;
        if (dense_cutoff_ == 0)
            return;

        i64 sum = 0;
        for (i32 i = 0; i < n; ++i)
            sum += degree_[i];
        const double avg = n ? double(sum) / double(n) : 0.0;

        const i32 dense_cut =
            (dense_cutoff_ > 0)
                ? dense_cutoff_
                : std::max<i32>(
                      16, std::min(n - 1,
                                   (i32)std::floor(
                                       0.35 * avg +
                                       12.0 * std::sqrt(std::max(1.0, avg)))));

        std::vector<i32> dense_nodes;
        dense_nodes.reserve(n / 8 + 1);
        for (i32 i = 0; i < n; ++i)
            if (var_active_[i] && degree_[i] >= dense_cut)
                dense_nodes.push_back(i);

        if (dense_nodes.empty())
            return;
        for (i32 v : dense_nodes) {
            bucket_remove_(v);
            where_[v] = -1;
        }
        dense_queue_.swap(dense_nodes);
    }

    // Buckets
    void bucket_insert_front_(i32 v, i32 d) {
        i32 h = head_[d];
        prev_[v] = -1;
        next_[v] = h;
        if (h != -1)
            prev_[h] = v;
        head_[d] = v;
        where_[v] = d;
    }
    void bucket_remove_(i32 v) {
        const i32 d = where_[v];
        if (d == -1)
            return;
        const i32 pv = prev_[v], nx = next_[v];
        if (pv != -1)
            next_[pv] = nx;
        else
            head_[d] = nx;
        if (nx != -1)
            prev_[nx] = pv;
        prev_[v] = next_[v] = -1;
        where_[v] = -1;
    }
    void bucket_move_(i32 v, i32 newd) {
        if (!var_active_[v])
            return;
        const i32 od = where_[v];
        const i32 clamped = std::max(0, std::min(n_, newd));
        if (od == clamped)
            return; // no useless churn
        bucket_remove_(v);
        bucket_insert_front_(v, clamped);
    }

    // Growth
    void grow_iw_(i32 need) {
        i32 new_cap = std::max(
            need,
            (i32)(std::max(2 * (i64)nzmax_, (i64)nzmax_ + nzmax_ / 2) + 1024));
        iw_.resize(new_cap);
        nzmax_ = new_cap;
        stats_iw_peak_ = std::max(stats_iw_peak_, nzmax_);
    }
    void grow_nodes_() {
        i32 new_nsym = (i32)(nsym_ + std::max(n_, nsym_ / 2) + 32);
        auto grow = [&](std::vector<i32> &a, i32 fill) {
            a.resize(new_nsym, fill);
        };
        auto growc = [&](std::vector<char> &a, char fill) {
            a.resize(new_nsym, fill);
        };
        grow(pe_, -1);
        grow(len_, 0);
        grow(elen_, -2);
        grow(nv_, 0);
        grow(degree_, 0);
        grow(w_, 0);
        grow(last_, -1);
        grow(next_, -1);
        grow(prev_, -1);
        grow(where_, -1);
        growc(var_active_, 0);
        growc(elem_active_, 0);
        nsym_ = new_nsym;
    }

    // Flag bump
    void bump_wflg_() {
        if (++wflg_ == 0x7ffffff0) {
            wflg_ = 1;
            std::fill(w_.begin(), w_.end(), 0);
        }
    }

    // Pivot selection
    i32 select_pivot_() {
        const i32 n = n_;
        for (;;) {
            while (mindeg_ <= n && head_[mindeg_] == -1)
                ++mindeg_;
            if (mindeg_ > n) {
                while (!dense_queue_.empty()) {
                    const i32 v = dense_queue_.back();
                    dense_queue_.pop_back();
                    if (var_active_[v] && nv_[v] > 0)
                        return v;
                }
                return -1;
            }
            for (i32 v = head_[mindeg_]; v != -1; v = next_[v]) {
                if (var_active_[v] && elen_[v] == -1 && nv_[v] > 0) {
                    bucket_remove_(v);
                    return v;
                }
            }
            ++mindeg_;
        }
    }

    // Elimination driver
    void eliminate_all_() {
        order_.clear();
        std::fill(in_order_.begin(), in_order_.end(), 0);

        for (;;) {
            const i32 piv = select_pivot_();
            if (piv == -1)
                break;

            // Emit reps in supervariable group (sorted for determinism)
            auto &grp = sv_members_[piv];
            if (!grp.empty()) {
                std::sort(grp.begin(), grp.end());
                for (i32 g : grp)
                    if (!in_order_[g]) {
                        order_.push_back(g);
                        in_order_[g] = 1;
                    }
            }

            eliminate_pivot_build_element_(piv);
            maybe_compact_iw_();
        }

        if ((i32)order_.size() < n_) {
            for (i32 i = 0; i < n_; ++i)
                if (!in_order_[i])
                    order_.push_back(i);
        }
        perm_.assign(order_.rbegin(), order_.rend()); // reverse elim order
    }

    // Element helpers
    i32 clean_element_vars_inplace_(i32 e, i32 skip_var) {
        i32 pe = pe_[e], le = elen_[e], rd = pe, wr = pe;
        for (i32 p = rd, E = rd + le; p < E; ++p) {
            i32 v = iw_[p];
            if (v == skip_var)
                continue;
            if (v < 0 || v >= nsym_)
                continue;
            if (elen_[v] == -1 && var_active_[v] && nv_[v] > 0) {
                if (wr >= nzmax_)
                    grow_iw_(wr + 1);
                iw_[wr++] = v;
            }
        }
        elen_[e] = wr - pe;
        if (elen_[e] > 1)
            std::sort(iw_.begin() + pe, iw_.begin() + pe + elen_[e]);
        return elen_[e];
    }

    static inline bool is_subset_sorted_(const i32 *a, i32 na, const i32 *b,
                                         i32 nb) {
        if (na > nb)
            return false;
        i32 i = 0, j = 0;
        while (i < na && j < nb) {
            if (a[i] == b[j]) {
                ++i;
                ++j;
            } else if (a[i] > b[j]) {
                ++j;
            } else
                return false;
        }
        return i == na;
    }

    // Hash for a sorted span
    inline uint64_t hash_sorted_ids_(const i32 *data, i32 len) const {
        uint64_t h = 0x9e3779b97f4a7c15ULL ^ (uint64_t)len;
        for (i32 k = 0; k < len; ++k)
            h = mix64_(h ^ (uint64_t)(uint32_t)data[k] * 0x9ddfea08eb382d69ULL);
        return h;
    }
    inline bool equal_sorted_spans_(const i32 *a, i32 na, const i32 *b,
                                    i32 nb) const {
        if (na != nb)
            return false;
        for (i32 i = 0; i < na; ++i)
            if (a[i] != b[i])
                return false;
        return true;
    }

    // Robin-Hood hashed absorption (equality + small-radius subset checks)
    void absorb_elements_hashed_(const std::vector<i32> &elems) {
        if (elems.empty())
            return;

        i32 live = 0;
        for (i32 e : elems)
            if (elem_active_[e] && elen_[e] > 0)
                ++live;
        if (live <= 1)
            return;

        i32 cap = 1;
        while (cap < (live << 1))
            cap <<= 1;

        struct Slot {
            uint64_t h = 0;
            i32 e = -1;
            i32 dist = 0;
        };
        std::vector<Slot> table(cap);

        auto insert_or_absorb = [&](i32 e) {
            const i32 le = elen_[e];
            const i32 pe = pe_[e];
            uint64_t key =
                hash_sorted_ids_(&iw_[pe], le); // was: const uint64_t h
            const i32 mask = cap - 1;
            i32 pos = (i32)(key & (uint64_t)mask), dist = 0;

            for (;;) {
                Slot &s = table[pos];
                if (s.e == -1) {
                    s.h = key;
                    s.e = e;
                    s.dist = dist;
                    return;
                }
                if (s.h == key) {
                    const i32 lj = elen_[s.e], pj = pe_[s.e];
                    if (equal_sorted_spans_(&iw_[pe], le, &iw_[pj], lj)) {
                        elem_active_[e] = 0;
                        elen_[e] = 0;
                        ++stats_absorbed_;
                        return;
                    }
                }
                if (dist > s.dist) {
                    // Robin–Hood swap
                    std::swap(key, s.h);
                    std::swap(e, s.e);
                    std::swap(dist, s.dist);
                }
                pos = (pos + 1) & mask;
                ++dist;
            }
        };

        // ensure sorted once
        for (i32 e : elems)
            if (elem_active_[e] && elen_[e] > 0)
                std::sort(iw_.begin() + pe_[e],
                          iw_.begin() + pe_[e] + elen_[e]);

        for (i32 e : elems)
            if (elem_active_[e] && elen_[e] > 0)
                insert_or_absorb(e);

        // local subset absorption within small neighborhood
        const int RADIUS = 4;
        for (i32 pos = 0; pos < cap; ++pos) {
            const i32 e = table[pos].e;
            if (e == -1 || !elem_active_[e])
                continue;
            const i32 le = elen_[e], pe = pe_[e];
            for (int r = 1; r <= RADIUS; ++r) {
                const i32 q = (pos + r) & (cap - 1);
                const i32 j = table[q].e;
                if (j == -1 || !elem_active_[j])
                    continue;
                const i32 lj = elen_[j], pj = pe_[j];
                if (le < lj) {
                    if (is_subset_sorted_(&iw_[pe], le, &iw_[pj], lj)) {
                        elem_active_[e] = 0;
                        elen_[e] = 0;
                        ++stats_absorbed_;
                        break;
                    }
                } else if (lj < le) {
                    if (is_subset_sorted_(&iw_[pj], lj, &iw_[pe], le)) {
                        elem_active_[j] = 0;
                        elen_[j] = 0;
                        ++stats_absorbed_;
                    }
                }
            }
        }
    }

    // Coalescence using signature over element-neighbor list (sorted small vec)
    void
    coalesce_variables_by_element_signature_(const std::vector<i32> &vlist) {
        struct Sig {
            uint64_t h;
            i32 deg;
        };
        auto make_sig = [&](i32 v) -> Sig {
            // collect element neighbors of v into tmp (sorted)
            std::vector<i32> tmp;
            tmp.reserve(8);
            for (i32 p = pe_[v], E = pe_[v] + len_[v]; p < E; ++p) {
                const i32 a = iw_[p];
                if (a >= 0 && a < nsym_ && elen_[a] >= 0 && elem_active_[a])
                    tmp.push_back(a);
            }
            std::sort(tmp.begin(), tmp.end());
            dedup_sorted_inplace(tmp);
            uint64_t h = hash_sorted_ids_(tmp.data(), (i32)tmp.size());
            // cache set for later exact compares
            elem_sig_cache_[v].swap(tmp);
            return {h, (i32)elem_sig_cache_[v].size()};
        };

        // build signature buckets
        sig_buckets_.clear();
        sig_buckets_.reserve(vlist.size() * 2);
        for (i32 v : vlist) {
            if (!var_active_[v] || nv_[v] == 0)
                continue;
            Sig s = make_sig(v);
            sig_buckets_[s.h].push_back(v);
        }

        for (auto &kv : sig_buckets_) {
            auto &cands = kv.second;
            if (cands.size() < 2)
                continue;

            // pick rep as first live
            i32 rep = -1;
            for (i32 v : cands)
                if (var_active_[v] && nv_[v] > 0) {
                    rep = v;
                    break;
                }
            if (rep < 0)
                continue;

            const auto &rep_set = elem_sig_cache_[rep];
            bool merged_any = false;
            for (i32 u : cands) {
                if (u == rep)
                    continue;
                if (!var_active_[u] || nv_[u] == 0)
                    continue;
                const auto &uset = elem_sig_cache_[u];
                if (uset.size() == rep_set.size() &&
                    std::equal(uset.begin(), uset.end(), rep_set.begin())) {
                    merge_variable_into_rep_(rep, u, rep_set);
                    ++stats_coalesced_;
                    merged_any = true;
                }
            }
            if (merged_any && var_active_[rep] && nv_[rep] > 0) {
                bump_wflg_();
                const i32 d = std::max(
                    0, std::min(n_, approx_external_degree_(rep, wflg_)));
                degree_[rep] = d;
                bucket_move_(rep, d);
                if (d < mindeg_)
                    mindeg_ = d;
            }
        }

        // clear cache memory
        for (i32 v : vlist)
            elem_sig_cache_[v].clear();
    }

    void merge_variable_into_rep_(i32 rep, i32 u,
                                  const std::vector<i32> &rep_elem_list) {
        if (rep == u || !var_active_[u] || nv_[u] == 0)
            return;
        nv_[rep] += nv_[u];
        nv_[u] = 0;
        var_active_[u] = 0;
        bucket_remove_(u);

        // strip u from dense queue if present
        if (!dense_queue_.empty()) {
            std::vector<i32> tmp;
            tmp.reserve(dense_queue_.size());
            for (i32 x : dense_queue_)
                if (x != u)
                    tmp.push_back(x);
            dense_queue_.swap(tmp);
        }

        if (!sv_members_[u].empty()) {
            sv_members_[rep].insert(sv_members_[rep].end(),
                                    sv_members_[u].begin(),
                                    sv_members_[u].end());
            sv_members_[u].clear();
        }

        // replace u by rep in each element (rep_elem_list is sorted unique)
        for (i32 e : rep_elem_list) {
            if (!elem_active_[e] || elen_[e] <= 0)
                continue;
            const i32 s = pe_[e];
            const i32 l = elen_[e];
            i32 wr = s;
            bool seen_rep = false;
            for (i32 p = s; p < s + l; ++p) {
                i32 v = iw_[p];
                if (v == u)
                    v = rep;
                if (v == rep) {
                    if (seen_rep)
                        continue;
                    seen_rep = true;
                }
                iw_[wr++] = v;
            }
            elen_[e] = wr - s;
            if (elen_[e] > 1)
                std::sort(iw_.begin() + s, iw_.begin() + s + elen_[e]);
        }
        len_[u] = 0;
    }

    // Approx external degree
    i32 approx_external_degree_(i32 v, i32 tag) {
        i32 start = pe_[v], L = len_[v];
        i32 total = 0;

        for (i32 p = start, E = start + L; p < E; ++p) {
            const i32 a = iw_[p];
            if (a < 0 || a >= nsym_)
                continue;
            if (elen_[a] == -1) {
                if (var_active_[a] && nv_[a] > 0 && w_[a] != tag) {
                    w_[a] = tag;
                    total += nv_[a];
                }
            }
        }
        for (i32 p = start, E = start + L; p < E; ++p) {
            const i32 a = iw_[p];
            if (a < 0 || a >= nsym_)
                continue;
            if (elen_[a] >= 0 && elem_active_[a]) {
                for (i32 q = pe_[a], Q = pe_[a] + elen_[a]; q < Q; ++q) {
                    const i32 u = iw_[q];
                    if (0 <= u && u < n_ && elen_[u] == -1) {
                        if (var_active_[u] && nv_[u] > 0 && w_[u] != tag) {
                            w_[u] = tag;
                            total += nv_[u];
                        }
                    }
                }
            }
        }
        if (w_[v] == tag)
            total -= std::max(0, nv_[v]);
        return std::max(0, total);
    }

    void maybe_refresh_degree_(i32 v, i32 iter_k) {
        if ((iter_k & 63) != 0)
            return;
        if (degree_[v] < std::min(n_, 8))
            return;
        bump_wflg_();
        const i32 d = approx_external_degree_(v, wflg_);
        if (d < degree_[v]) {
            degree_[v] = d;
            bucket_move_(v, d);
            if (d < mindeg_)
                mindeg_ = d;
        }
    }

    void maybe_compact_iw_() {
        if (((i32)order_.size() & 255) != 0)
            return;
        i32 write = 0;
        for (i32 i = 0; i < nsym_; ++i) {
            if (elen_[i] >= 0 && elem_active_[i]) {
                const i32 s = pe_[i], l = elen_[i];
                if (l > 0) {
                    if (write + l > nzmax_)
                        grow_iw_(write + l);
                    std::copy(iw_.begin() + s, iw_.begin() + s + l,
                              iw_.begin() + write);
                    pe_[i] = write;
                    write += l;
                }
            } else if (elen_[i] == -1 && var_active_[i]) {
                const i32 s = pe_[i], l = len_[i];
                if (l > 0) {
                    if (write + l > nzmax_)
                        grow_iw_(write + l);
                    std::copy(iw_.begin() + s, iw_.begin() + s + l,
                              iw_.begin() + write);
                    pe_[i] = write;
                    write += l;
                }
            }
        }
        tail_used_ = write;
        stats_iw_peak_ = std::max(stats_iw_peak_, tail_used_);
    }

    // Core elimination of one pivot (build new element, frontier updates)
    void eliminate_pivot_build_element_(i32 piv) {
        if (!var_active_[piv])
            return;
        var_active_[piv] = 0;
        nv_[piv] = 0;

        // Snapshot neighbors
        std::vector<i32> neigh;
        neigh.reserve(len_[piv]);
        for (i32 p = pe_[piv], e = pe_[piv] + len_[piv]; p < e; ++p)
            neigh.push_back(iw_[p]);

        std::vector<i32> varN, elemN;
        varN.reserve(neigh.size());
        for (i32 u : neigh) {
            if (u < 0 || u >= nsym_)
                continue;
            if (elen_[u] == -1) {
                if (var_active_[u] && nv_[u] > 0)
                    varN.push_back(u);
            } else if (elen_[u] >= 0) {
                if (elem_active_[u])
                    elemN.push_back(u);
            }
        }

        // Clean elements
        std::vector<i32> cleaned;
        cleaned.reserve(elemN.size());
        for (i32 e : elemN) {
            if (clean_element_vars_inplace_(e, piv) > 0)
                cleaned.push_back(e);
            else {
                elem_active_[e] = 0;
                elen_[e] = 0;
            }
        }

        if (aggressive_absorption_ && !cleaned.empty())
            absorb_elements_hashed_(cleaned);

        // Frontier: mark and collect new vars from varN + cleaned elements
        bump_wflg_();
        const i32 tag = wflg_;
        std::vector<i32> new_vars;
        new_vars.reserve((i32)varN.size() + 8);
        auto try_push = [&](i32 v) {
            if (0 <= v && v < n_ && elen_[v] == -1 && var_active_[v] &&
                nv_[v] > 0 && w_[v] != tag) {
                w_[v] = tag;
                new_vars.push_back(v);
            }
        };
        for (i32 v : varN)
            try_push(v);
        for (i32 e : cleaned)
            if (elem_active_[e]) {
                for (i32 p = pe_[e], E = pe_[e] + elen_[e]; p < E; ++p)
                    try_push(iw_[p]);
            }

        if (!new_vars.empty()) {
            const i32 e_new = alloc_new_element_();
            store_element_varlist_(e_new, new_vars);
            elem_active_[e_new] = 1;

            // Update var lists and degrees only for frontier
            for (i32 v : new_vars)
                rebuild_var_list_after_fill_(v, piv, e_new);

            // Coalesce twins
            coalesce_variables_by_element_signature_(new_vars);

            // Refresh bucket positions
            bump_wflg_();
            const i32 tag2 = wflg_;
            for (i32 v : new_vars)
                if (var_active_[v] && nv_[v] > 0) {
                    const i32 d = std::max(
                        0, std::min(n_, approx_external_degree_(v, tag2)));
                    if (d != degree_[v]) {
                        degree_[v] = d;
                        bucket_move_(v, d);
                        if (d < mindeg_)
                            mindeg_ = d;
                    }
                    maybe_refresh_degree_(v, (i32)order_.size());
                }
        }
        maybe_compact_iw_();
    }

    // storage helpers
    i32 alloc_new_element_() {
        if (nelem_top_ >= nsym_)
            grow_nodes_();
        const i32 e = nelem_top_++;
        elen_[e] = 0;
        nv_[e] = 0;
        elem_active_[e] = 1;
        var_active_[e] = 0;
        pe_[e] = -1;
        return e;
    }
    void store_element_varlist_(i32 e, std::vector<i32> vlist) {
        std::sort(vlist.begin(), vlist.end());
        dedup_sorted_inplace(vlist);
        const i32 need = (i32)vlist.size();
        const i32 pos = reserve_space_(need);
        pe_[e] = pos;
        elen_[e] = need;
        len_[e] = 0;
        for (i32 i = 0; i < need; ++i)
            iw_[pos + i] = vlist[i];
    }
    i32 reserve_space_(i32 need) {
        const i32 start = tail_used_;
        const i32 end = start + need;
        if (end > nzmax_)
            grow_iw_(end);
        tail_used_ = end;
        stats_iw_peak_ = std::max(stats_iw_peak_, tail_used_);
        return start;
    }

    void rebuild_var_list_after_fill_(i32 v, i32 piv, i32 new_elem) {
        i32 start = pe_[v], L = len_[v], wr = start;
        bool seen_elem = false;
        for (i32 p = start, E = start + L; p < E; ++p) {
            const i32 a = iw_[p];
            if (a == piv)
                continue;
            if (a < 0 || a >= nsym_)
                continue;
            if (elen_[a] == -1) {
                if (var_active_[a] && nv_[a] > 0) {
                    if (wr >= nzmax_)
                        grow_iw_(wr + 1);
                    iw_[wr++] = a;
                }
            } else if (elen_[a] >= 0) {
                if (elem_active_[a]) {
                    if (a == new_elem)
                        seen_elem = true;
                    if (wr >= nzmax_)
                        grow_iw_(wr + 1);
                    iw_[wr++] = a;
                }
            }
        }

        if (elem_active_[new_elem] && !seen_elem) {
            if (wr < start + L) {
                if (wr >= nzmax_)
                    grow_iw_(wr + 1);
                iw_[wr++] = new_elem;
            } else {
                std::vector<i32> seg(iw_.begin() + start, iw_.begin() + wr);
                const i32 ns = reserve_space_((i32)seg.size() + 1);
                pe_[v] = ns;
                std::copy(seg.begin(), seg.end(), iw_.begin() + ns);
                iw_[ns + (i32)seg.size()] = new_elem;
                len_[v] = (i32)seg.size() + 1;
                return;
            }
        }
        len_[v] = wr - start;
    }

    // Lightweight cache for coalescence (element neighbor lists)
    std::unordered_map<i32, std::vector<i32>> elem_sig_cache_;
    std::unordered_map<uint64_t, std::vector<i32>> sig_buckets_;

public:
    // A is n×n CSR (pattern-only). Return B = A[p, :][:, p].
    // EXPECTS: p[new] = old
    static CSR permute_(const CSR &A, const std::vector<i32> &p,
                        bool sort_cols = true, bool dedup = false) {
        const i32 n = A.n;
        if (n == 0)
            return CSR(0);

        const auto &AI = A.indptr;
        const auto &AJ = A.indices;
        std::vector<i32> ip = inverse_permutation(p);

        CSR B(n);
        B.indptr.assign(n + 1, 0);

        for (i32 i = 0; i < n; ++i) {
            const i32 oi = p[i];
            B.indptr[i + 1] = B.indptr[i] + (AI[oi + 1] - AI[oi]);
        }
        B.indices.resize(B.indptr.back());

        for (i32 i = 0; i < n; ++i) {
            const i32 oi = p[i];
            const i32 begA = AI[oi], endA = AI[oi + 1];
            i32 out = B.indptr[i];
            for (i32 k = begA; k < endA; ++k) {
                const i32 j_old = AJ[k];
                B.indices[out++] = ip[j_old];
            }
            if (sort_cols) {
                auto beg = B.indices.begin() + B.indptr[i];
                auto end = B.indices.begin() + out;
                std::sort(beg, end);
                if (dedup) {
                    auto new_end = std::unique(beg, end);
                    (void)new_end; // second pass compacts
                }
            }
        }

        if (dedup) {
            std::vector<i32> nip(n + 1, 0);
            for (i32 i = 0; i < n; ++i) {
                const i32 rb = B.indptr[i], re = B.indptr[i + 1];
                if (re <= rb) {
                    nip[i + 1] = nip[i];
                    continue;
                }
                i32 len = 1;
                for (i32 k = rb + 1; k < re; ++k)
                    if (B.indices[k] != B.indices[k - 1])
                        ++len;
                nip[i + 1] = nip[i] + len;
            }
            std::vector<i32> nidx(nip.back());
            for (i32 i = 0; i < n; ++i) {
                const i32 rb = B.indptr[i], re = B.indptr[i + 1];
                i32 out = nip[i];
                if (re > rb) {
                    nidx[out++] = B.indices[rb];
                    for (i32 k = rb + 1; k < re; ++k)
                        if (B.indices[k] != B.indices[k - 1])
                            nidx[out++] = B.indices[k];
                }
            }
            B.indptr.swap(nip);
            B.indices.swap(nidx);
        }

        return B;
    }

    static i32 bandwidth_(const CSR &A) {
        if (A.nnz() == 0)
            return 0;
        i32 bw = 0;
        for (i32 i = 0; i < A.n; ++i) {
            for (i32 p = A.indptr[i]; p < A.indptr[i + 1]; ++p) {
                const i32 j = A.indices[p];
                bw = std::max(bw, std::abs(j - i));
            }
        }
        return bw;
    }
};

//------------------------------------------------------------------------------
// Demo / quick test
//------------------------------------------------------------------------------
static CSR random_erdos_renyi(i32 n, double p, std::mt19937_64 &rng) {
    // build upper, then mirror (keep pattern, include diagonal)
    std::bernoulli_distribution coin(p);
    std::vector<std::vector<i32>> rows(n);
    for (i32 i = 0; i < n; ++i)
        rows[i].push_back(i);
    for (i32 i = 0; i < n; ++i) {
        for (i32 j = i + 1; j < n; ++j) {
            if (coin(rng)) {
                rows[i].push_back(j);
                rows[j].push_back(i);
            }
        }
    }
    CSR A(n);
    A.indptr[0] = 0;
    for (i32 i = 0; i < n; ++i) {
        std::sort(rows[i].begin(), rows[i].end());
        dedup_sorted_inplace(rows[i]);
        A.indptr[i + 1] = A.indptr[i] + (i32)rows[i].size();
    }
    A.indices.resize(A.indptr.back());
    for (i32 i = 0, w = 0; i < n; ++i)
        for (i32 v : rows[i])
            A.indices[w++] = v;
    return A;
}

int main() {
    std::mt19937_64 rng(42);
    const i32 n = 4000;
    const double p = 4.0 / n; // sparse
    CSR A = random_erdos_renyi(n, p, rng);

    AMDReorderingArray amd(/*aggressive_absorption=*/true, /*dense_cutoff=*/-1);
    auto [perm, st] =
        amd.compute_fill_reducing_permutation(A, /*symmetrize=*/true);

    std::cout << "n=" << st.matrix_size << " nnz=" << st.original_nnz
              << " bw0=" << st.original_bandwidth
              << " bw1=" << st.reordered_bandwidth
              << " red=" << (100.0 * st.bandwidth_reduction) << "%\n";
    std::cout << "absorbed=" << st.absorbed_elements
              << " coalesced=" << st.coalesced_variables
              << " iw_peak=" << st.iw_capacity_peak << "\n";

    // sanity: permutation size and is a bijection
    std::vector<char> seen(n, 0);
    for (i32 i = 0; i < (i32)perm.size(); ++i) {
        if (perm[i] < 0 || perm[i] >= n || seen[perm[i]]) {
            std::cerr << "Permutation invalid at " << i << "\n";
            return 1;
        }
        seen[perm[i]] = 1;
    }
    std::cout << "Permutation OK.\n";
    return 0;
}
