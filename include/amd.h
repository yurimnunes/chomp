// amd_array_qg.hpp (single translation unit demo)
// g++ -O3 -std=c++17 amd_array_qg.hpp -o amd && ./amd

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

using i32 = int32_t;
using u32 = uint32_t;
using i64 = int64_t;

// --------------------------- CSR pattern container ---------------------------

struct CSR {
    i32 n{0};
    std::vector<i32> indptr;  // size n+1
    std::vector<i32> indices; // size nnz

    CSR() = default;
    explicit CSR(i32 n_) : n(n_), indptr(n_ + 1, 0) {}

    i32 nnz() const { return static_cast<i32>(indices.size()); }

    // Build strictly upper-triangular pattern of A ∪ A^T (i < j only).
    // Input: this CSR may be non-symmetric; we only use pattern.
    CSR strict_upper_union_transpose() const {
        const i32 N = n;
        std::vector<std::vector<i32>> adj_up(N);
        for (i32 i = 0; i < N; ++i) {
            for (i32 p = indptr[i]; p < indptr[i + 1]; ++p) {
                i32 j = indices[p];
                if (i == j)
                    continue;
                i32 a = std::min(i, j);
                i32 b = std::max(i, j);
                // store only in row a (strict upper: a < b)
                adj_up[a].push_back(b);
            }
        }
        // unique + sort
        CSR U(N);
        U.indptr[0] = 0;
        for (i32 i = 0; i < N; ++i) {
            auto &row = adj_up[i];
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
            U.indptr[i + 1] = U.indptr[i] + static_cast<i32>(row.size());
        }
        U.indices.resize(U.indptr.back());
        for (i32 i = 0, w = 0; i < N; ++i) {
            auto &row = adj_up[i];
            for (i32 v : row)
                U.indices[w++] = v;
        }
        return U;
    }
};

// Helper: inverse permutation
static std::vector<i32> inverse_permutation(const std::vector<i32> &p) {
    std::vector<i32> inv(p.size());
    for (i32 i = 0; i < (i32)p.size(); ++i)
        inv[p[i]] = i;
    return inv;
}

// --------------------------- AMD Reordering (array) --------------------------

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

class AMDReorderingArray {
public:
    explicit AMDReorderingArray(bool aggressive_absorption = true,
                                int dense_cutoff = -1)
        : aggressive_absorption_(aggressive_absorption),
          dense_cutoff_(dense_cutoff) {}

    // Main API: compute AMD permutation. If symmetrize=true, uses A ∪ A^T and
    // drops diagonal.
    std::vector<i32> amd_order(const CSR &A, bool symmetrize = true) {
        CSR Aup = symmetrize ? A.strict_upper_union_transpose() : A;
        n_ = Aup.n;
        if (n_ == 0)
            return {};

        initialize_from_upper_(Aup);
        initialize_buckets_();
        eliminate_all_();
        return perm_; // already reversed(elim order)
    }

    // Convenience wrapper returning stats and inverse permutation
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
        // extras
        st.absorbed_elements = stats_absorbed_;
        st.coalesced_variables = stats_coalesced_;
        st.iw_capacity_peak = stats_iw_peak_;
        return {p, st};
    }

private:
    // ---------------------- problem size / storage ----------------------
    i32 n_{0};
    i32 nsym_{0};
    i32 nzmax_{0};

    // arrays
    std::vector<i32> pe_, len_, elen_, iw_, nv_, degree_, w_, last_;
    std::vector<i32> head_, next_, prev_, where_;
    i32 mindeg_{0};
    std::vector<char> var_active_, elem_active_;

    // elements arena
    i32 nelem_top_{0};

    // outputs / work
    std::vector<i32> order_, perm_, dense_queue_;
    i32 tail_used_{0};
    i32 wflg_{1};
    std::vector<std::vector<i32>> sv_members_;
    std::vector<char> in_order_;

    // options
    bool aggressive_absorption_{true};
    int dense_cutoff_{-1}; // -1 heuristic, 0 off, >0 explicit

    // stats
    i32 stats_absorbed_{0};
    i32 stats_coalesced_{0};
    i32 stats_iw_peak_{0};

    // ---------------------- initialization ----------------------

    void initialize_from_upper_(const CSR &Aup) {
        const i32 n = Aup.n;
        const auto &indptr = Aup.indptr;
        const auto &indices = Aup.indices;
        const i32 m = Aup.nnz();

        const i32 est_undirected_nnz = 2 * m;
        nzmax_ = std::max({est_undirected_nnz + n,
                           (i32)(1.5 * est_undirected_nnz) + 4 * n, 1});
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

        // degrees (undirected)
        std::vector<i32> deg(n, 0);
        for (i32 i = 0; i < n; ++i) {
            for (i32 p = indptr[i]; p < indptr[i + 1]; ++p) {
                i32 j = indices[p]; // j > i
                ++deg[i];
                ++deg[j];
            }
        }

        // prefix + basic init
        i32 pos = 0;
        for (i32 i = 0; i < n; ++i) {
            pe_[i] = pos;
            len_[i] = deg[i];
            degree_[i] = deg[i];
            elen_[i] = -1;
            var_active_[i] = 1;
            elem_active_[i] = 0;
            pos += deg[i];
        }
        if (pos > nzmax_)
            grow_iw_(pos);

        // fill adjacency both directions
        std::vector<i32> write_ptr(pe_.begin(), pe_.begin() + n);
        for (i32 i = 0; i < n; ++i) {
            for (i32 p = indptr[i]; p < indptr[i + 1]; ++p) {
                i32 j = indices[p]; // i < j
                i32 wi = write_ptr[i], wj = write_ptr[j];
                if (wi >= nzmax_ || wj >= nzmax_)
                    grow_iw_(std::max(wi, wj) + 1);
                iw_[wi] = j;
                iw_[wj] = i;
                ++write_ptr[i];
                ++write_ptr[j];
            }
        }

        tail_used_ = pos;

        // dead elements beyond n
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

        // reset stats
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
        for (i32 i = 0; i < n; ++i) {
            if (var_active_[i]) {
                i32 d = std::max(0, std::min(n, degree_[i]));
                bucket_insert_front_(i, d);
                if (d < mindeg_)
                    mindeg_ = d;
            }
        }
        apply_dense_postponement_();
    }

    // ---------------------- dense postponement ----------------------

    void apply_dense_postponement_() {
        const i32 n = n_;
        if (dense_cutoff_ == 0)
            return;

        // Compute avg degree quickly
        i64 degsum = 0;
        for (i32 i = 0; i < n; ++i)
            degsum += degree_[i];
        double avg_deg = n ? double(degsum) / double(n) : 0.0;

        i32 dense_cut =
            (dense_cutoff_ > 0)
                ? dense_cutoff_
                : std::max<i32>(
                      16, std::min(n - 1,
                                   (i32)std::floor(0.5 * avg_deg +
                                                   10.0 * std::sqrt(std::max(
                                                              1.0, avg_deg)))));

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
        dense_queue_ = std::move(dense_nodes);
    }

    // ---------------------- degree buckets ----------------------

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
        i32 d = where_[v];
        if (d == -1)
            return;
        i32 pv = prev_[v], nx = next_[v];
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
        bucket_remove_(v);
        newd = std::max(0, std::min(n_, newd));
        bucket_insert_front_(v, newd);
    }

    // ---------------------- dynamic growth ----------------------

    void grow_iw_(i32 required_capacity) {
        i32 new_cap = std::max(
            required_capacity,
            (i32)(std::max(2 * (i64)nzmax_, (i64)nzmax_ + nzmax_ / 2) + 1024));
        iw_.resize(new_cap);
        nzmax_ = new_cap;
        stats_iw_peak_ = std::max(stats_iw_peak_, nzmax_);
    }

    void grow_nodes_() {
        i32 new_nsym = (i32)(nsym_ + std::max(n_, nsym_ / 2) + 32);
        auto grow = [&](std::vector<i32> &a, i32 fill) {
            i32 old = (i32)a.size();
            a.resize(new_nsym, fill);
        };
        auto growc = [&](std::vector<char> &a, char fill) {
            i32 old = (i32)a.size();
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

    // ---------------------- elimination loop ----------------------
    // inside AMDReorderingArray
    i32 select_pivot_() {
        const i32 n = n_;
        while (true) {
            while (mindeg_ <= n && head_[mindeg_] == -1)
                ++mindeg_;
            if (mindeg_ > n) {
                // fall back to dense queue
                while (!dense_queue_.empty()) {
                    i32 v = dense_queue_.front();
                    dense_queue_.erase(dense_queue_.begin());
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

    void eliminate_all_() {
        order_.clear();
        std::fill(in_order_.begin(), in_order_.end(), 0);

        for (;;) {
            i32 piv = select_pivot_();
            if (piv == -1)
                break;

            // Emit full supervariable group (original ids)
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
        // final permutation = reverse of elimination order
        perm_.assign(order_.rbegin(), order_.rend());
    }

    // ---------------------- pivot elimination ----------------------

    void eliminate_pivot_build_element_(i32 piv) {
        if (!var_active_[piv])
            return;
        var_active_[piv] = 0;
        nv_[piv] = 0;

        // Snapshot neighbors of piv
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

        // Clean elements: drop piv and dead vars
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

        // --- Frontier-based union: mark & collect directly, no global scan ---
        bump_wflg_();
        i32 tag = wflg_;
        std::vector<i32> new_vars;
        new_vars.reserve((i32)varN.size() + 8);

        auto try_push = [&](i32 v) {
            if (0 <= v && v < n_ && elen_[v] == -1 && var_active_[v] &&
                nv_[v] > 0 && w_[v] != tag) {
                w_[v] = tag;
                new_vars.push_back(v);
            }
        };

        // Add direct variable neighbors first
        for (i32 v : varN)
            try_push(v);

        // Then variables from cleaned elements
        for (i32 e : cleaned)
            if (elem_active_[e]) {
                for (i32 p = pe_[e], E = pe_[e] + elen_[e]; p < E; ++p)
                    try_push(iw_[p]);
            }

        if (!new_vars.empty()) {
            i32 e_new = alloc_new_element_();
            store_element_varlist_(e_new, new_vars);
            elem_active_[e_new] = 1;

            // Update var lists and degrees only for the frontier
            for (i32 v : new_vars)
                rebuild_var_list_after_fill_(v, piv, e_new);

            // Coalesce variables that are element-equivalent among just the
            // frontier
            coalesce_variables_by_element_signature_(new_vars);

            // Recompute approx degrees for the touched variables and re-bucket
            bump_wflg_();
            i32 tag2 = wflg_;
            for (i32 v : new_vars)
                if (var_active_[v] && nv_[v] > 0) {
                    i32 d = std::max(
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

        // Periodic compaction (unchanged)
        maybe_compact_iw_();
    }

    // ---------------------- coalescence ----------------------

    // inside AMDReorderingArray
    void
    coalesce_variables_by_element_signature_(const std::vector<i32> &vlist) {
        std::unordered_map<u32, std::vector<i32>> sig_map;
        sig_map.reserve(vlist.size() * 2);

        for (i32 v : vlist) {
            if (!var_active_[v] || nv_[v] == 0)
                continue;
            auto es = collect_element_neighbors_set_(v);
            u32 sig = hash_id_set_(es);
            sig_map[sig].push_back(v);
        }

        for (auto &kv : sig_map) {
            auto &cands = kv.second;
            if (cands.size() < 2)
                continue;

            i32 rep = -1;
            std::unordered_map<i32, std::set<i32>> sets;
            sets.reserve(cands.size());
            for (i32 v : cands) {
                if (var_active_[v] && nv_[v] > 0) {
                    sets[v] = collect_element_neighbors_set_(v);
                    if (rep == -1)
                        rep = v;
                }
            }
            if (rep == -1)
                continue;
            const auto &rep_set = sets[rep];

            bool merged_any = false;
            for (i32 u : cands) {
                if (u == rep)
                    continue;
                if (!var_active_[u] || nv_[u] == 0)
                    continue;
                const auto &uset = sets[u];
                if (uset.size() == rep_set.size() && uset == rep_set) {
                    merge_variable_into_rep_(rep, u, rep_set);
                    ++stats_coalesced_;
                    merged_any = true;
                }
            }

            if (merged_any && var_active_[rep] && nv_[rep] > 0) {
                // Refresh approximate degree and bucket position for the
                // representative
                bump_wflg_();
                i32 tag = wflg_;
                i32 d = std::max(
                    0, std::min(n_, approx_external_degree_(rep, tag)));
                degree_[rep] = d;
                bucket_move_(rep, d);
                if (d < mindeg_)
                    mindeg_ = d;
            }
        }
    }

    std::set<i32> collect_element_neighbors_set_(i32 v) const {
        std::set<i32> s;
        for (i32 p = pe_[v], E = pe_[v] + len_[v]; p < E; ++p) {
            i32 a = iw_[p];
            if (a >= 0 && a < nsym_ && elen_[a] >= 0 && elem_active_[a])
                s.insert(a);
        }
        return s;
    }

    void merge_variable_into_rep_(i32 rep, i32 u,
                                  const std::set<i32> &rep_elem_set) {
        if (rep == u || !var_active_[u] || nv_[u] == 0)
            return;
        nv_[rep] += nv_[u];
        nv_[u] = 0;
        var_active_[u] = 0;
        bucket_remove_(u);
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
        // replace u by rep in each element
        for (i32 e : rep_elem_set) {
            if (!elem_active_[e] || elen_[e] <= 0)
                continue;
            i32 wr = pe_[e];
            bool seen_rep = false;
            for (i32 p = pe_[e], E = pe_[e] + elen_[e]; p < E; ++p) {
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
            elen_[e] = wr - pe_[e];
            if (elen_[e] > 1) {
                std::sort(iw_.begin() + pe_[e],
                          iw_.begin() + pe_[e] + elen_[e]);
            }
        }
        len_[u] = 0;
    }

    // ---------------------- approx degree ----------------------

    i32 approx_external_degree_(i32 v, i32 tag) {
        i32 start = pe_[v], L = len_[v];
        i32 total = 0;

        // direct variable neighbors
        for (i32 p = start, E = start + L; p < E; ++p) {
            i32 a = iw_[p];
            if (a < 0 || a >= nsym_)
                continue;
            if (elen_[a] == -1) {
                if (var_active_[a] && nv_[a] > 0 && w_[a] != tag) {
                    w_[a] = tag;
                    total += nv_[a];
                }
            }
        }
        // variables in elements
        for (i32 p = start, E = start + L; p < E; ++p) {
            i32 a = iw_[p];
            if (a < 0 || a >= nsym_)
                continue;
            if (elen_[a] >= 0 && elem_active_[a]) {
                for (i32 q = pe_[a], Q = pe_[a] + elen_[a]; q < Q; ++q) {
                    i32 u = iw_[q];
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
        i32 tag = wflg_;
        i32 d = approx_external_degree_(v, tag);
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
                i32 s = pe_[i], l = elen_[i];
                if (l > 0) {
                    if (write + l > nzmax_)
                        grow_iw_(write + l);
                    std::copy(iw_.begin() + s, iw_.begin() + s + l,
                              iw_.begin() + write);
                    pe_[i] = write;
                    write += l;
                }
            } else if (elen_[i] == -1 && var_active_[i]) {
                i32 s = pe_[i], l = len_[i];
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

    void bump_wflg_() {
        if (++wflg_ == 0x7ffffff0) {
            wflg_ = 1;
            std::fill(w_.begin(), w_.end(), 0);
        }
    }

    // ---------------------- element helpers ----------------------

    // inside AMDReorderingArray
    i32 clean_element_vars_inplace_(i32 e, i32 skip_var) {
        i32 pe = pe_[e], le = elen_[e], rd = pe, wr = pe;
        for (i32 p = rd, E = rd + le; p < E; ++p) {
            i32 v = iw_[p];
            if (v == skip_var)
                continue;
            if (v < 0 || v >= nsym_)
                continue;
            // keep only live variables (supervariables)
            if (elen_[v] == -1 && var_active_[v] && nv_[v] > 0) {
                if (wr >= nzmax_)
                    grow_iw_(wr + 1);
                iw_[wr++] = v;
            }
        }
        elen_[e] = wr - pe;
        if (elen_[e] > 1) {
            std::sort(iw_.begin() + pe, iw_.begin() + pe + elen_[e]);
        }
        return elen_[e];
    }

    static inline u32 mix32_(u32 x) {
        // Simple 32-bit mix (xorshift + golden ratio)
        x ^= x >> 16;
        x *= 0x7feb352dU;
        x ^= x >> 15;
        x *= 0x846ca68bU;
        x ^= x >> 16;
        return x;
    }

    static u32 hash_id_set_(const std::set<i32> &ids) {
        u32 xx = 0, ssum = 0;
        u32 len = (u32)ids.size();
        for (i32 v : ids) {
            u32 u = (u32)v;
            xx ^= (u * 0x9e3779b1U);
            ssum += u;
            xx &= 0xFFFFFFFFU;
            ssum &= 0xFFFFFFFFU;
        }
        u32 key =
            (ssum + 1315423911U + ((len & 0xFFFFU) << 16) + xx) & 0x7FFFFFFFU;
        return key;
    }

    void absorb_elements_hashed_(const std::vector<i32> &elems) {
        // table size power-of-two >= 2*len
        i32 m = 1;
        i32 need = std::max<i32>(4, 2 * (i32)elems.size());
        while (m < need)
            m <<= 1;

        std::vector<i32> hhead(m, -1);
        std::vector<i32> hnext(nsym_, -1);

        auto e_hash = [&](i32 eid) -> i32 {
            i32 pe = pe_[eid], le = elen_[eid];
            u32 ssum = 0, xx = 0;
            for (i32 p = pe, E = pe + le; p < E; ++p) {
                u32 v = (u32)iw_[p];
                xx ^= (v * 0x9e3779b1U);
                xx &= 0xFFFFFFFFU;
                ssum += v;
                ssum &= 0xFFFFFFFFU;
            }
            u32 key = (ssum + 1315423911U + ((u32)(le & 0xFFFF) << 16) + xx) &
                      0x7FFFFFFFU;
            return (i32)(key & (u32)(m - 1));
        };

        for (i32 e : elems) {
            if (!elem_active_[e] || elen_[e] <= 0)
                continue;
            i32 b = e_hash(e);
            i32 head = hhead[b];

            // mark vars of e
            bump_wflg_();
            i32 tag_e = wflg_;
            for (i32 p = pe_[e], E = pe_[e] + elen_[e]; p < E; ++p)
                w_[iw_[p]] = tag_e;

            bool absorbed = false;
            for (i32 j = head; j != -1; j = hnext[j]) {
                if (!elem_active_[j] || elen_[j] <= 0)
                    continue;
                i32 len_e = elen_[e], len_j = elen_[j];

                if (len_e == len_j) {
                    bool eq = true;
                    i32 cnt = 0;
                    for (i32 p = pe_[j], E = pe_[j] + len_j; p < E; ++p) {
                        if (w_[iw_[p]] != tag_e) {
                            eq = false;
                            break;
                        }
                        ++cnt;
                    }
                    if (eq && cnt == len_e) {
                        elem_active_[e] = 0;
                        elen_[e] = 0;
                        absorbed = true;
                        ++stats_absorbed_;
                        break;
                    }
                }
                if (len_e < len_j) {
                    bump_wflg_();
                    i32 tag_j = wflg_;
                    for (i32 p = pe_[j], E = pe_[j] + len_j; p < E; ++p)
                        w_[iw_[p]] = tag_j;
                    bool subset = true;
                    for (i32 p = pe_[e], E = pe_[e] + len_e; p < E; ++p) {
                        if (w_[iw_[p]] != tag_j) {
                            subset = false;
                            break;
                        }
                    }
                    if (subset) {
                        elem_active_[e] = 0;
                        elen_[e] = 0;
                        absorbed = true;
                        ++stats_absorbed_;
                        break;
                    }
                } else if (len_j < len_e) {
                    bool subset = true;
                    for (i32 p = pe_[j], E = pe_[j] + len_j; p < E; ++p) {
                        if (w_[iw_[p]] != tag_e) {
                            subset = false;
                            break;
                        }
                    }
                    if (subset) {
                        elem_active_[j] = 0;
                        elen_[j] = 0;
                        ++stats_absorbed_;
                    }
                }
            }
            if (!absorbed) {
                hnext[e] = head;
                hhead[b] = e;
            }
        }
    }

    // storage helpers
    i32 alloc_new_element_() {
        if (nelem_top_ >= nsym_)
            grow_nodes_();
        i32 e = nelem_top_++;
        elen_[e] = 0;
        nv_[e] = 0;
        elem_active_[e] = 1;
        var_active_[e] = 0;
        pe_[e] = -1;
        return e;
    }

    void store_element_varlist_(i32 e, std::vector<i32> vlist) {
        std::sort(vlist.begin(), vlist.end());
        const i32 need = (i32)vlist.size();
        i32 pos = reserve_space_(need);
        pe_[e] = pos;
        elen_[e] = need;
        len_[e] = 0;
        for (i32 i = 0; i < need; ++i)
            iw_[pos + i] = vlist[i];
    }

    i32 reserve_space_(i32 need) {
        i32 start = tail_used_;
        i32 end = start + need;
        if (end > nzmax_)
            grow_iw_(end);
        tail_used_ = end;
        stats_iw_peak_ = std::max(stats_iw_peak_, tail_used_);
        return start;
    }

    void rebuild_var_list_after_fill_(i32 v, i32 piv, i32 new_elem) {
        i32 start = pe_[v], L = len_[v], rd = start, wr = start;
        bool seen_elem = false;
        for (i32 p = rd, E = rd + L; p < E; ++p) {
            i32 a = iw_[p];
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
                // need to relocate window
                std::vector<i32> seg(iw_.begin() + start, iw_.begin() + wr);
                i32 new_start = reserve_space_((i32)seg.size() + 1);
                pe_[v] = new_start;
                std::copy(seg.begin(), seg.end(), iw_.begin() + new_start);
                iw_[new_start + (i32)seg.size()] = new_elem;
                len_[v] = (i32)seg.size() + 1;
                return;
            }
        }
        len_[v] = wr - start;
    }

public:
    // inside AMDReorderingArray
    // A is n×n CSR (pattern-only). Return B = A[p, :][:, p].
    // If you don't need canonicalization, set sort_cols=false and dedup=false
    // for O(nnz).
    static CSR permute_(const CSR &A, const std::vector<i32> &p,
                        bool sort_cols = true, bool dedup = false) {
        const i32 n = A.n;
        if (n == 0)
            return CSR(0);
        const auto &AI = A.indptr;
        const auto &AJ = A.indices;

        // ip: new row -> old row
        std::vector<i32> ip = inverse_permutation(p);

        CSR B(n);
        B.indptr.assign(n + 1, 0);

        // 1) Row lengths (before optional dedup)
        for (i32 i = 0; i < n; ++i) {
            const i32 oi = ip[i];
            B.indptr[i + 1] = B.indptr[i] + (AI[oi + 1] - AI[oi]);
        }
        B.indices.resize(B.indptr.back());

// 2) Column remap fill
#pragma omp parallel for if ((i64)B.indices.size() > (1 << 15))
        for (i32 i = 0; i < n; ++i) {
            const i32 oi = ip[i];
            const i32 begA = AI[oi];
            const i32 endA = AI[oi + 1];
            i32 out = B.indptr[i];
            for (i32 k = begA; k < endA; ++k) {
                const i32 j_old = AJ[k];
                B.indices[out++] = p[j_old];
            }
            if (sort_cols) {
                auto beg = B.indices.begin() + B.indptr[i];
                auto end = B.indices.begin() + out;
                std::sort(beg, end);
                if (dedup) {
                    // compact this *row only* using a temp buffer to avoid UB
                    // (Alternatively: unique in-place then rewrite row.)
                    auto new_end = std::unique(beg, end);
                    const i32 new_len = (i32)std::distance(beg, new_end);
                    const i32 old_len = out - B.indptr[i];
                    if (new_len < old_len) {
                        // rewrite row compactly
                        std::copy(beg, new_end, beg);
                        // we can't shrink the whole indices array per-row
                        // without shifting tails, so we leave extra slots
                        // "garbage" for now and fix in a second pass. Mark the
                        // effective length via a side array.
                    }
                }
            }
        }

        if (dedup) {
            // Second pass: rebuild a compact indices array using the
            // sorted+uniqued rows.
            std::vector<i32> new_indptr(n + 1, 0);
            // compute exact lengths by re-uniqueing cheaply (rows are sorted
            // already)
            for (i32 i = 0; i < n; ++i) {
                const i32 rb = B.indptr[i];
                const i32 re = B.indptr[i + 1];
                if (re <= rb) {
                    new_indptr[i + 1] = new_indptr[i];
                    continue;
                }
                i32 len = 1;
                for (i32 k = rb + 1; k < re; ++k) {
                    if (B.indices[k] != B.indices[k - 1])
                        ++len;
                }
                new_indptr[i + 1] = new_indptr[i] + len;
            }
            std::vector<i32> new_indices(new_indptr.back());
            // fill compacted rows
            for (i32 i = 0; i < n; ++i) {
                const i32 rb = B.indptr[i];
                const i32 re = B.indptr[i + 1];
                i32 out = new_indptr[i];
                if (re > rb) {
                    new_indices[out++] = B.indices[rb];
                    for (i32 k = rb + 1; k < re; ++k) {
                        if (B.indices[k] != B.indices[k - 1])
                            new_indices[out++] = B.indices[k];
                    }
                }
            }
            B.indptr.swap(new_indptr);
            B.indices.swap(new_indices);
        }

        return B;
    }

    static i32 bandwidth_(const CSR &A) {
        if (A.nnz() == 0)
            return 0;
        i32 bw = 0;
        for (i32 i = 0; i < A.n; ++i) {
            for (i32 p = A.indptr[i]; p < A.indptr[i + 1]; ++p) {
                i32 j = A.indices[p];
                bw = std::max(bw, std::abs(j - i));
            }
        }
        return bw;
    }
};

// ====================== Supernode identification (structural)
// ====================== Works on symmetric pattern (A ∪ A^T), lower triangle,
// after applying a permutation p. Based on Liu's etree and Gilbert-Ng-Peyton
// supernode criteria with relaxed amalgamation.

struct SupernodeInfo {
    // Supernode k covers columns [ranges[k].first, ranges[k].second] in
    // permuted space
    std::vector<std::pair<i32, i32>> ranges;
    std::vector<i32> col2sn; // size n, maps column -> supernode id
    std::vector<i32> etree;  // elimination tree parent (size n), -1 is root
    std::vector<i32> post;   // postorder permutation of etree (size n)
};

// Build strictly lower-triangular symmetric pattern of A[p,p] ∪ A[p,p]^T
static CSR make_lower_sym(const CSR &A, const std::vector<i32> &p) {
    const i32 n = A.n;
    CSR P =
        AMDReorderingArray::permute_(A, p, /*sort_cols=*/true, /*dedup=*/true);
    // symmetrize (pattern only) and keep strict lower
    std::vector<std::vector<i32>> rows(n);
    for (i32 i = 0; i < n; ++i) {
        for (i32 k = P.indptr[i]; k < P.indptr[i + 1]; ++k) {
            i32 j = P.indices[k];
            if (j == i)
                continue;
            i32 r = std::max(i, j), c = std::min(i, j);
            rows[r].push_back(c);
        }
    }
    CSR L(n);
    L.indptr[0] = 0;
    for (i32 i = 0; i < n; ++i) {
        auto &r = rows[i];
        std::sort(r.begin(), r.end());
        r.erase(std::unique(r.begin(), r.end()), r.end());
        L.indptr[i + 1] = L.indptr[i] + (i32)r.size();
    }
    L.indices.resize(L.indptr.back());
    for (i32 i = 0, w = 0; i < n; ++i) {
        for (i32 v : rows[i])
            L.indices[w++] = v;
    }
    return L;
}

// Liu’s elimination tree for symmetric (use lower triangle L: row i contains
// cols < i)
static std::vector<i32> etree_from_lower(const CSR &L) {
    const i32 n = L.n;
    std::vector<i32> parent(n, -1), ancestor(n, -1);
    for (i32 j = 0; j < n; ++j) {
        for (i32 p = L.indptr[j]; p < L.indptr[j + 1]; ++p) {
            i32 i = L.indices[p]; // i < j in lower
            // find with path compression
            while (i != -1 && i < j) {
                i32 next = ancestor[i];
                ancestor[i] = j;
                if (next == -1) {
                    parent[i] = j;
                    break;
                }
                i = next;
            }
        }
    }
    return parent;
}

// Postorder of a rooted forest (etree) — iterative DFS
static std::vector<i32> postorder_etree(const std::vector<i32> &parent) {
    const i32 n = (i32)parent.size();
    std::vector<i32> head(n, -1), next(n, -1), root;
    root.reserve(n);
    for (i32 i = 0; i < n; ++i) {
        i32 p = parent[i];
        if (p == -1)
            root.push_back(i);
        else {
            next[i] = head[p];
            head[p] = i;
        }
    }
    std::vector<i32> post;
    post.reserve(n);
    std::vector<std::pair<i32, i32>> st;
    st.reserve(n);
    // state: (node, it = child iterator index via next-list)
    for (i32 r : root) {
        st.emplace_back(r, head[r]);
        while (!st.empty()) {
            auto &[u, it] = st.back();
            if (it == -2) { // done, emit
                post.push_back(u);
                st.pop_back();
                if (!st.empty())
                    st.back().second =
                        next[st.back().second]; // advance parent iterator
                continue;
            }
            if (it == -1) { // no children
                it = -2;
                continue;
            }
            // descend first child in adjacency list
            st.emplace_back(it, head[it]);
        }
    }
    return post;
}

// Compare two sorted index lists A (rows > cutA) and B (rows > cutB) with
// relaxed amalgamation. Returns true if they are equal OR within relaxation
// thresholds:
//   - up to "relax" absolute extra/missing entries, AND
//   - Jaccard >= tau (0..1).
static bool structural_match_relaxed(const i32 *A, i32 lenA, i32 cutA,
                                     const i32 *B, i32 lenB, i32 cutB,
                                     int relax, double tau) {
    // advance to strictly-greater-than-cut
    while (lenA > 0 && *A <= cutA) {
        ++A;
        --lenA;
    }
    while (lenB > 0 && *B <= cutB) {
        ++B;
        --lenB;
    }

    // exact early-out
    if (relax <= 0 && lenA == lenB && std::equal(A, A + lenA, B))
        return true;

    int i = 0, j = 0, inter = 0;
    while (i < lenA && j < lenB) {
        if (A[i] == B[j]) {
            ++inter;
            ++i;
            ++j;
        } else if (A[i] < B[j]) {
            ++i;
        } else {
            ++j;
        }
    }
    int uni = lenA + lenB - inter;
    int diff = uni - inter; // total mismatched count
    if (diff > relax)
        return false;
    double jac = (uni == 0) ? 1.0 : double(inter) / double(uni);
    return jac >= tau;
}

// Identify supernodes on permuted pattern. Returns ranges in PERMUTED indices.
static SupernodeInfo identify_supernodes(
    const CSR &A, const std::vector<i32> &p,
    int relax = 0,    // allow up to this many set diffs when merging
    double tau = 1.0, // Jaccard threshold (1.0 = exact)
    int max_size = std::numeric_limits<int>::max()) {
    const i32 n = A.n;
    SupernodeInfo out;
    out.col2sn.assign(n, -1);

    // 1) Symmetric lower pattern after permutation
    CSR L = make_lower_sym(A, p);

    // 2) Elimination tree and postorder (on permuted space)
    out.etree = etree_from_lower(L);
    out.post = postorder_etree(out.etree);

    // For quick column access
    auto col_ptr = [&](i32 j) { return &L.indices[L.indptr[j]]; };
    auto col_len = [&](i32 j) { return (i32)(L.indptr[j + 1] - L.indptr[j]); };

    // 3) Scan natural column order (0..n-1). Supernode criteria:
    //    parent(k) = k+1 and structure(k)\{<=k} == structure(k+1)\{<=k+1}
    i32 j = 0;
    i32 sn_id = 0;
    while (j < n) {
        i32 t = j;
        while (t + 1 < n) {
            if (out.etree[t] != t + 1)
                break; // must be a chain
            // compare strictly-below patterns with relaxed match
            const i32 *Aj = col_ptr(t);
            const i32 *Ak = col_ptr(t + 1);
            i32 Lj = col_len(t);
            i32 Lk = col_len(t + 1);
            bool ok = structural_match_relaxed(Aj, Lj, t,     // drop ≤ t
                                               Ak, Lk, t + 1, // drop ≤ t+1
                                               relax, tau);
            if (!ok)
                break;
            if ((t + 1 - j + 1) >= max_size)
                break;
            ++t;
        }
        out.ranges.emplace_back(j, t);
        for (i32 c = j; c <= t; ++c)
            out.col2sn[c] = sn_id;
        ++sn_id;
        j = t + 1;
    }
    return out;
}
