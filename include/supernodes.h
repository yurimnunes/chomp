// supernodes_qdldl.hpp
// SoTA-leaning supernode identification using qdldl23's *permuted upper CSC*
// and its symbolic analysis. Works directly on B = P A Pᵀ (upper+diag).
//
// Drop-in usage:
//   #include "qdldl23.hpp"
//   #include "supernodes_qdldl.hpp"
//
//   using namespace qdldl23;
//   SparseD32 B = /* your permuted upper CSC */;
//   Symb32    S = analyze_fast(B);        // etree + column counts
//   auto sn = identify_supernodes_qdldl(B, S,
//                                       /*relax_abs*/ 2,
//                                       /*relax_rel*/ 0.10,
//                                       /*tau*/       0.70,
//                                       /*max_size*/  128);
//
// Returns supernode ranges in *permuted* column space, plus etree and postorder.
//
// © 2025 MIT/Apache-2.0

#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>
#include "qdldl23.hpp"

namespace snode {

// ===== types =====
using i32 = int32_t;

// Supernode metadata (columns are in *permuted* space)
template <typename IntT = i32>
struct SupernodeInfo {
    std::vector<std::pair<IntT, IntT>> ranges; // inclusive [lo, hi] for each supernode
    std::vector<IntT> col2sn;                  // size n; maps col -> supernode id
    std::vector<IntT> etree;                   // elimination tree parent (copy of S.etree)
    std::vector<IntT> post;                    // postorder of etree
};

// ----- robust postorder over an etree forest (iterative DFS) -----
template <typename IntT = i32>
static std::vector<IntT> postorder_etree_safe(const std::vector<IntT>& parent) {
    const IntT n = static_cast<IntT>(parent.size());
    std::vector<IntT> head(n, -1), next(n, -1), roots; roots.reserve(n);
    for (IntT v = 0; v < n; ++v) {
        const IntT p = parent[(size_t)v];
        if (p == -1) roots.push_back(v);
        else { next[(size_t)v] = head[(size_t)p]; head[(size_t)p] = v; }
    }

    std::vector<IntT> post; post.reserve(n);
    std::vector<std::pair<IntT, IntT>> st; st.reserve(n); // (node, child-iter index via head/next)
    for (IntT r : roots) {
        st.emplace_back(r, head[(size_t)r]);
        while (!st.empty()) {
            auto& top = st.back();
            IntT& u  = top.first;
            IntT& it = top.second;
            if (it == -2) { // finished: emit and pop
                post.push_back(u);
                st.pop_back();
                if (!st.empty()) {
                    // advance parent's iterator to next sibling
                    auto& par = st.back();
                    if (par.second != -1) par.second = next[(size_t)par.second];
                }
                continue;
            }
            if (it == -1) { // no children
                it = -2;
                continue;
            }
            // descend first (current) child of u
            st.emplace_back(it, head[(size_t)it]);
        }
    }
    return post;
}

// ----- relaxed structural match on strictly-upper sets (two-pointer) -----
// Compare S(j) = { i | i < j and B(i,j)!=0 } versus S(k) = { i | i < k and B(i,k)!=0 }.
// Cuts drop (<= j) and (<= k), respectively.
// Accept if:
//   - symmetric difference <= relax_abs
//   - and symdiff / |S(j)| <= relax_rel      (if |S(j)| > 0)
//   - and Jaccard >= tau
template <typename IntT = i32>
static inline bool relaxed_match_twoptr(const IntT* Aj, IntT lenj, IntT cutj,
                                        const IntT* Ak, IntT lenk, IntT cutk,
                                        IntT relax_abs, double relax_rel, double tau)
{
    // advance to strictly-above cuts
    IntT ij = 0; while (ij < lenj && Aj[(size_t)ij] <= cutj) ++ij;
    IntT ik = 0; while (ik < lenk && Ak[(size_t)ik] <= cutk) ++ik;
    const IntT nA = lenj - ij;
    const IntT nB = lenk - ik;

    // exact fast-path
    if (relax_abs <= 0 && relax_rel <= 0.0) {
        if (nA != nB) return false;
        for (IntT t = 0; t < nA; ++t)
            if (Aj[(size_t)(ij + t)] != Ak[(size_t)(ik + t)]) return false;
        return true;
    }

    // two-pointer intersection/union
    IntT inter = 0, a = ij, b = ik;
    while (a < lenj && b < lenk) {
        const IntT va = Aj[(size_t)a], vb = Ak[(size_t)b];
        if      (va == vb) { ++inter; ++a; ++b; }
        else if (va <  vb) { ++a; }
        else               { ++b; }
    }
    const IntT uni  = nA + nB - inter;
    const IntT symd = nA + nB - 2 * inter;

    if (symd > relax_abs) return false;
    if (nA > 0) {
        const double rel = double(symd) / double(nA);
        if (rel > relax_rel) return false;
    }
    const double jac = (uni == 0) ? 1.0 : double(inter) / double(uni);
    return jac >= tau;
}

// ----- main: identify supernodes from qdldl's permuted upper CSC & symbolics -----
// Works for any FloatT/IntT supported by qdldl23.
template <typename FloatT = double, typename IntT = int32_t>
static SupernodeInfo<IntT>
identify_supernodes_qdldl(const qdldl23::SparseUpperCSC<FloatT, IntT>& B,
                          const qdldl23::Symbolic<IntT>&               S,
                          IntT relax_abs = 0,
                          double relax_rel = 0.0,
                          double tau = 1.0,
                          IntT max_size = std::numeric_limits<IntT>::max())
{
    const IntT n = B.n;
    SupernodeInfo<IntT> out;
    out.col2sn.assign((size_t)n, IntT{-1});
    out.etree = S.etree;                 // copy for convenience/visibility
    out.post  = postorder_etree_safe<IntT>(out.etree);

    auto col_ptr = [&](IntT j) -> const IntT* {
        return &B.Ai[(size_t)B.Ap[(size_t)j]];
    };
    auto col_len = [&](IntT j) -> IntT {
        return B.Ap[(size_t)j + 1] - B.Ap[(size_t)j];
    };

    IntT j = 0, sid = 0;
    while (j < n) {
        IntT t = j;
        // grow chain j..t while etree[t] == t+1 and strictly-upper sets match (relaxed)
        while (t + 1 < n) {
            if (out.etree[(size_t)t] != t + 1) break; // must be a chain

            const IntT* Sj = col_ptr(t);
            const IntT  Lj = col_len(t);
            const IntT* Sk = col_ptr(t + 1);
            const IntT  Lk = col_len(t + 1);

            const bool ok = relaxed_match_twoptr<IntT>(
                Sj, Lj, t,         // drop rows <= t
                Sk, Lk, t + 1,     // drop rows <= t+1
                relax_abs, relax_rel, tau
            );
            if (!ok) break;

            const IntT width = (t + 1) - j + 1;
            if (width >= max_size) break;

            ++t;
        }
        out.ranges.emplace_back(j, t);
        for (IntT c = j; c <= t; ++c) out.col2sn[(size_t)c] = sid;
        ++sid;
        j = t + 1;
    }
    return out;
}

// ----- convenience overload for qdldl double/int32 aliases -----
inline SupernodeInfo<i32>
identify_supernodes_qdldl(const qdldl23::SparseD32& B, const qdldl23::Symb32& S,
                          i32 relax_abs = 0, double relax_rel = 0.0,
                          double tau = 1.0, i32 max_size = std::numeric_limits<i32>::max())
{
    return identify_supernodes_qdldl<double, int32_t>(B, S, relax_abs, relax_rel, tau, max_size);
}

} // namespace snode
