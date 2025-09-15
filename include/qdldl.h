#pragma once
// qdldl23.hpp — header-only LDLᵀ (QDLDL-style) for upper CSC
// C++23, no threads, exact-zero pivot, no sign forcing.
// © 2025 — MIT/Apache-2.0 compatible with original QDLDL spirit.

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// ===== qdldl23 execution config (optional stdexec) ==========================
// Opt in with: -DQDLDL23_USE_STDEXEC
// Requires the P2300 reference impl headers:
//   <stdexec/execution.hpp> and <exec/static_thread_pool.hpp>
// This path uses a CPU thread pool; no device memory or CUDA required.
#if defined(QDLDL23_USE_STDEXEC)
  #include <stdexec/execution.hpp>
  #include <exec/static_thread_pool.hpp>
  #include <thread>
  #define QDLDL23_HAS_STDEXEC 1
#else
  #define QDLDL23_HAS_STDEXEC 0
#endif

namespace qdldl23 {

#if QDLDL23_HAS_STDEXEC
namespace detail {
  template <class F>
  inline void qd_par_for_n(std::size_t n, F&& f) {
    using namespace stdexec;
    exec::static_thread_pool pool(std::max(1u, std::thread::hardware_concurrency()));
    auto sched = pool.get_scheduler();
    auto snd = schedule(sched)
             | bulk(n, [fn = std::forward<F>(f)](std::size_t i) { fn(i); });
    (void)sync_wait(snd);
  }
} // namespace detail
#else
namespace detail {
  template <class F>
  inline void qd_par_for_n(std::size_t n, F&& f) {
    for (std::size_t i = 0; i < n; ++i) f(i);
  }
} // namespace detail
#endif

// --------- Errors ----------
struct InvalidMatrixError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct FactorizationError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// --------- SparseUpperCSC (upper incl diag, sorted, duplicate-free) ----------
template <std::floating_point FloatT = double,
          std::signed_integral IntT = int32_t>
class SparseUpperCSC {
public:
    std::vector<IntT> Ap;   // size n+1
    std::vector<IntT> Ai;   // size nnz (logical: Ap[n])
    std::vector<FloatT> Ax; // size nnz
    IntT n{0};

    SparseUpperCSC() = default;
    SparseUpperCSC(IntT n_, std::vector<IntT> Ap_, std::vector<IntT> Ai_,
                   std::vector<FloatT> Ax_)
        : Ap(std::move(Ap_)), Ai(std::move(Ai_)), Ax(std::move(Ax_)), n(n_) {
        if (n < 0)
            throw InvalidMatrixError("n < 0");
        if (Ap.size() != static_cast<size_t>(n + 1))
            throw InvalidMatrixError("Ap size != n+1");
        const size_t nnz = static_cast<size_t>(Ap.back());
        if (Ai.size() != nnz || Ax.size() != nnz)
            throw InvalidMatrixError("Ai/Ax size mismatch");
        finalize_upper_inplace(/*require_diag=*/true);
    }

    [[nodiscard]] size_t nnz() const { return static_cast<size_t>(Ap.back()); }

    // Normalize each column: sort by row, coalesce duplicates,
    // forbid lower entries, forbid empty columns, enforce diag if requested.
    void finalize_upper_inplace(bool require_diag = true) {
        if (n == 0)
            return;
        for (IntT j = 0; j < n; ++j) {
            const IntT p0 = Ap[(size_t)j], p1 = Ap[(size_t)j + 1];
            if (p0 > p1)
                throw InvalidMatrixError("Ap must be nondecreasing");
            if (p0 == p1)
                throw InvalidMatrixError("Empty column");

            // collect (row,val)
            std::vector<std::pair<IntT, FloatT>> col;
            col.reserve(static_cast<size_t>(p1 - p0));
            for (IntT p = p0; p < p1; ++p) {
                const IntT i = Ai[(size_t)p];
                if (i < 0 || i >= n)
                    throw InvalidMatrixError("Row index OOB");
                if (i > j)
                    throw InvalidMatrixError(
                        "Lower-triangular entry (need upper+diag only)");
                col.emplace_back(i, Ax[(size_t)p]);
            }
            // sort by row
            std::sort(col.begin(), col.end(),
                      [](auto &a, auto &b) { return a.first < b.first; });

            // coalesce duplicates into Ai/Ax in place
            IntT w = p0;
            bool has_diag = false;
            for (size_t k = 0; k < col.size();) {
                IntT r = col[k].first;
                FloatT sum = col[k].second;
                size_t k2 = k + 1;
                while (k2 < col.size() && col[k2].first == r) {
                    sum += col[k2].second;
                    ++k2;
                }
                Ai[(size_t)w] = r;
                Ax[(size_t)w] = sum;
                if (r == j)
                    has_diag = true;
                ++w;
                k = k2;
            }

            // compact: update Ap deltas for later columns (logical shrink)
            const IntT new_p1 = w;
            const IntT drop = p1 - new_p1;
            if (drop) {
                for (IntT jj = j + 1; jj <= n; ++jj)
                    Ap[(size_t)jj] -= drop;
            }
            if (require_diag && !has_diag)
                throw InvalidMatrixError(
                    "Missing explicit diagonal at column " + std::to_string(j));
        }
    }
};

// --------- Ordering (perm/iperm) ----------
template <std::signed_integral IntT = int32_t> struct Ordering {
    std::vector<IntT> perm;  // size n, maps old->new
    std::vector<IntT> iperm; // size n, maps new->old
    IntT n{0};

    static Ordering identity(IntT n) {
        Ordering o;
        o.n = n;
        o.perm.resize((size_t)n);
        o.iperm.resize((size_t)n);
        std::iota(o.perm.begin(), o.perm.end(), IntT{0});
        std::iota(o.iperm.begin(), o.iperm.end(), IntT{0});
        return o;
    }
    static Ordering from_perm(std::vector<IntT> p) {
        Ordering o;
        o.n = (IntT)p.size();
        o.perm = std::move(p);
        o.iperm.assign((size_t)o.n, IntT{-1});
        std::vector<char> seen((size_t)o.n, 0);
        for (IntT i = 0; i < o.n; ++i) {
            IntT pi = o.perm[(size_t)i];
            if (pi < 0 || pi >= o.n || seen[(size_t)pi])
                throw InvalidMatrixError("Invalid permutation");
            seen[(size_t)pi] = 1;
            o.iperm[(size_t)pi] = i;
        }
        return o;
    }
};

// Symmetric permutation: B = P A Pᵀ (keep **upper**), per-column std::sort +
// coalesce.
template <std::floating_point FloatT = double,
          std::signed_integral IntT = int32_t>
inline SparseUpperCSC<FloatT, IntT>
permute_symmetric_upper(const SparseUpperCSC<FloatT, IntT> &A,
                        const Ordering<IntT> &ord) {
    if (ord.n != A.n)
        throw InvalidMatrixError("Ordering size mismatch");
    const IntT n = A.n;

    // Count entries per permuted column
    std::vector<IntT> cnt((size_t)n, IntT{0});
    for (IntT j = 0; j < n; ++j) {
        const IntT pj = ord.perm[(size_t)j];
        for (IntT p = A.Ap[(size_t)j]; p < A.Ap[(size_t)j + 1]; ++p) {
            const IntT i = A.Ai[(size_t)p];
            const IntT pi = ord.perm[(size_t)i];
            const IntT col = (pi > pj ? pi : pj);
            ++cnt[(size_t)col];
        }
    }

    // Build pointers
    std::vector<IntT> Bp((size_t)n + 1, IntT{0});
    for (IntT j = 0; j < n; ++j)
        Bp[(size_t)j + 1] = Bp[(size_t)j] + cnt[(size_t)j];
    const size_t nnz = (size_t)Bp[(size_t)n];
    std::vector<IntT> Bi(nnz);
    std::vector<FloatT> Bx(nnz);
    std::vector<IntT> head = Bp;

    // Scatter unsorted by column
    for (IntT j = 0; j < n; ++j) {
        const IntT pj = ord.perm[(size_t)j];
        for (IntT p = A.Ap[(size_t)j]; p < A.Ap[(size_t)j + 1]; ++p) {
            const IntT i = A.Ai[(size_t)p];
            const FloatT v = A.Ax[(size_t)p];
            IntT pi = ord.perm[(size_t)i];
            IntT col = pj, row = pi;
            if (row > col)
                std::swap(row, col); // push to upper
            const IntT dst = head[(size_t)col]++;
            Bi[(size_t)dst] = row;
            Bx[(size_t)dst] = v;
        }
    }

    // Per-column sort & coalesce; ensure diagonal present
    for (IntT j = 0; j < n; ++j) {
        IntT p0 = Bp[(size_t)j], p1 = Bp[(size_t)j + 1];
        const IntT len = p1 - p0;
        auto *Ri = &Bi[(size_t)p0];
        auto *Rx = &Bx[(size_t)p0];

        // Stable sort rows
        std::vector<IntT> order((size_t)len);
        std::iota(order.begin(), order.end(), IntT{0});
        std::sort(order.begin(), order.end(), [&](IntT a, IntT b) {
            return Ri[(size_t)a] < Ri[(size_t)b];
        });

        std::vector<IntT> tmpi((size_t)len);
        std::vector<FloatT> tmpx((size_t)len);
        for (IntT k = 0; k < len; ++k) {
            tmpi[(size_t)k] = Ri[(size_t)order[(size_t)k]];
            tmpx[(size_t)k] = Rx[(size_t)order[(size_t)k]];
        }
        std::copy(tmpi.begin(), tmpi.end(), Ri);
        std::copy(tmpx.begin(), tmpx.end(), Rx);

        // coalesce
        IntT w = 0;
        bool has_diag = false;
        for (IntT k = 0; k < len;) {
            IntT r = Ri[(size_t)k];
            FloatT s = Rx[(size_t)k];
            IntT k2 = k + 1;
            while (k2 < len && Ri[(size_t)k2] == r) {
                s += Rx[(size_t)k2];
                ++k2;
            }
            Ri[(size_t)w] = r;
            Rx[(size_t)w] = s;
            if (r == j)
                has_diag = true;
            ++w;
            k = k2;
        }
        const IntT drop = len - w;
        if (drop) {
            for (IntT jj = j + 1; jj <= n; ++jj)
                Bp[(size_t)jj] -= drop;
        }
        if (!has_diag)
            throw InvalidMatrixError(
                "Missing diagonal after permutation at column " +
                std::to_string(j));
    }

    SparseUpperCSC<FloatT, IntT> B;
    B.n = n;
    B.Ap = std::move(Bp);
    B.Ai = std::move(Bi);
    B.Ax = std::move(Bx);
    // Already sorted, coalesced, and checked.
    return B;
}

// --------- Symbolic structure ----------
template <std::signed_integral IntT = int32_t> struct Symbolic {
    std::vector<IntT> etree; // parent[j] or -1
    std::vector<IntT> Lnz;   // strictly-lower count per column of L
    std::vector<IntT> Lp;    // size n+1, cumulated Lnz
    IntT n{0};
};

// Liu (1986) elimination tree for upper CSC (no threads)
template <std::floating_point FloatT = double,
          std::signed_integral IntT = int32_t>
inline void etree_upper_liu(const SparseUpperCSC<FloatT, IntT> &A,
                            std::vector<IntT> &parent) {
    const IntT n = A.n;
    parent.assign((size_t)n, (IntT)-1);
    std::vector<IntT> ancestor((size_t)n, (IntT)-1);

    auto find_root = [&](IntT j) {
        IntT r = j;
        while (ancestor[(size_t)r] != (IntT)-1)
            r = ancestor[(size_t)r];
        // path compression
        IntT cur = j;
        while (ancestor[(size_t)cur] != (IntT)-1) {
            const IntT nxt = ancestor[(size_t)cur];
            ancestor[(size_t)cur] = r;
            cur = nxt;
        }
        return r;
    };

    for (IntT j = 0; j < n; ++j) {
        // traverse strictly upper rows in col j
        for (IntT p = A.Ap[(size_t)j]; p < A.Ap[(size_t)j + 1]; ++p) {
            IntT i = A.Ai[(size_t)p];
            if (i == j)
                continue; // skip diag
            while (true) {
                IntT r = (ancestor[(size_t)i] == (IntT)-1) ? -1 : find_root(i);
                if (r == -1) {
                    ancestor[(size_t)i] = j;
                    parent[(size_t)i] = j;
                    break;
                }
                if (r == j) {
                    break;
                }
                ancestor[(size_t)r] = j;
                parent[(size_t)r] = j;
                i = r;
            }
        }
    }
}

// Gilbert–Ng–Peyton column counts for upper CSC
template <std::floating_point FloatT = double,
          std::signed_integral IntT = int32_t>
inline void colcounts_upper_gnp(const SparseUpperCSC<FloatT, IntT> &A,
                                const std::vector<IntT> &parent,
                                std::vector<IntT> &Lnz) {
    const IntT n = A.n;
    Lnz.assign((size_t)n, IntT{0});
    std::vector<IntT> prevnz((size_t)n, (IntT)-1);

    for (IntT j = 0; j < n; ++j) {
        for (IntT p = A.Ap[(size_t)j]; p < A.Ap[(size_t)j + 1]; ++p) {
            const IntT i = A.Ai[(size_t)p];
            if (i == j)
                continue; // skip diag
            IntT k = i;
            while (k != -1 && k < j && prevnz[(size_t)k] != j) {
                ++Lnz[(size_t)k];
                prevnz[(size_t)k] = j;
                k = parent[(size_t)k];
            }
        }
    }
}

// one-shot symbolic using Liu+GNP
template <std::floating_point FloatT = double,
          std::signed_integral IntT = int32_t>
inline Symbolic<IntT> analyze_fast(const SparseUpperCSC<FloatT, IntT> &A) {
    Symbolic<IntT> S;
    S.n = A.n;
    etree_upper_liu(A, S.etree);
    colcounts_upper_gnp(A, S.etree, S.Lnz);
    S.Lp.resize((size_t)S.n + 1);
    S.Lp[0] = 0;
    for (IntT j = 0; j < S.n; ++j)
        S.Lp[(size_t)j + 1] = S.Lp[(size_t)j] + S.Lnz[(size_t)j];
    return S;
}

// --------- Numeric factorization (QDLDL-style, no pivoting) ----------
template <std::floating_point FloatT = double,
          std::signed_integral IntT = int32_t>
struct LDLFactors {
    std::vector<IntT> Lp;     // size n+1 (strict lower)
    std::vector<IntT> Li;     // size nnz(L)
    std::vector<FloatT> Lx;   // size nnz(L)
    std::vector<FloatT> D;    // size n
    std::vector<FloatT> Dinv; // size n
    IntT n{0};
    IntT num_pos{0};
};

template <std::floating_point FloatT = double,
          std::signed_integral IntT = int32_t>
inline LDLFactors<FloatT, IntT>
refactorize(const SparseUpperCSC<FloatT, IntT> &A, const Symbolic<IntT> &S) {
    const IntT n = S.n;
    if (A.n != n)
        throw InvalidMatrixError("Symbolic/numeric size mismatch");

    LDLFactors<FloatT, IntT> R;
    R.n = n;
    R.num_pos = 0;
    R.Lp = S.Lp;
    const IntT nnzL = R.Lp[(size_t)n];
    R.Li.assign((size_t)nnzL, IntT{0});
    R.Lx.assign((size_t)nnzL, FloatT{0});
    R.D.assign((size_t)n, FloatT{0});
    R.Dinv.assign((size_t)n, FloatT{0});

    // ---------- timestamped work arrays (no clears) ----------
    std::vector<IntT> Lnext((size_t)n, IntT{0});
    for (IntT i = 0; i < n; ++i) Lnext[(size_t)i] = R.Lp[(size_t)i];

    std::vector<FloatT> y((size_t)n, FloatT{0});
    std::vector<IntT> yIdx((size_t)n, IntT{0});
    std::vector<IntT> Ebuf((size_t)n, IntT{0});

    std::vector<IntT> mark((size_t)n, IntT{0});
    IntT curmark = 1;

    auto is_marked = [&](IntT i){ return mark[(size_t)i]==curmark; };
    auto set_mark  = [&](IntT i){ mark[(size_t)i]=curmark; };

    // k = 0
    {
        // first entry in column 0 must be the diagonal (after finalize)
        if (A.Ai[(size_t)A.Ap[0]] != 0)
            throw FactorizationError("Missing diagonal at column 0");
        R.D[0] = A.Ax[(size_t)A.Ap[0]];
        if (R.D[0] == FloatT{0})
            throw FactorizationError("Zero pivot at column 0");
        if (R.D[0] > FloatT{0})
            ++R.num_pos;
        R.Dinv[0] = FloatT{1} / R.D[0];
    }

    for (IntT k = 1; k < n; ++k) {
        // (bump the timestamp; wrap conservatively if needed)
        if (++curmark == std::numeric_limits<IntT>::max()) {
            std::fill(mark.begin(), mark.end(), IntT{0});
            curmark = 1;
        }

        // scatter A(:,k) into y; record diagonal
        IntT nnzY = 0;
        bool diag_seen = false;
        for (IntT p = A.Ap[(size_t)k]; p < A.Ap[(size_t)k + 1]; ++p) {
            const IntT i = A.Ai[(size_t)p];
            const FloatT v = A.Ax[(size_t)p];
            if (i == k) {
                R.D[(size_t)k] = v;
                diag_seen = true;
                continue;
            }
            y[(size_t)i] = v;
            if (!is_marked(i)) {
                set_mark(i);
                // walk up etree to k (exclusive), push reversed onto yIdx
                IntT next = i, nnzE = 0;
                Ebuf[(size_t)nnzE++] = next;
                next = S.etree[(size_t)next];
                while (next != (IntT)-1 && next < k) {
                    if (is_marked(next)) break;
                    set_mark(next);
                    Ebuf[(size_t)nnzE++] = next;
                    next = S.etree[(size_t)next];
                }
                while (nnzE)
                    yIdx[(size_t)nnzY++] = Ebuf[(size_t)--nnzE];
            }
        }
        if (!diag_seen)
            throw FactorizationError("Missing diagonal at column " +
                                     std::to_string(k));

        // eliminate along the listed columns (reverse order)
        for (IntT t = nnzY - 1; t >= 0; --t) {
            const IntT c = yIdx[(size_t)t];
            const IntT j0 = R.Lp[(size_t)c];
            IntT &j1 = Lnext[(size_t)c];
            const FloatT yc = y[(size_t)c];

            // y -= L(:,c)*yc
            for (IntT j = j0; j < j1; ++j)
                y[(size_t)R.Li[(size_t)j]] -= R.Lx[(size_t)j] * yc;

            R.Li[(size_t)j1] = k;
            const FloatT lk = yc * R.Dinv[(size_t)c];
            R.Lx[(size_t)j1] = lk;
            R.D[(size_t)k] -= yc * lk;
            ++j1;

            y[(size_t)c] = FloatT{0};
            if (t == 0) break;
        }

        if (R.D[(size_t)k] == FloatT{0})
            throw FactorizationError("Zero pivot at column " +
                                     std::to_string(k));
        if (R.D[(size_t)k] > FloatT{0})
            ++R.num_pos;
        R.Dinv[(size_t)k] = FloatT{1} / R.D[(size_t)k];
    }

    return R;
}

template <std::floating_point FloatT = double,
          std::signed_integral IntT = int32_t>
inline LDLFactors<FloatT, IntT>
refactorize_with_ordering(const SparseUpperCSC<FloatT, IntT> &A,
                          const Symbolic<IntT> &Sperm,
                          const Ordering<IntT> &ord) {
    const auto B = permute_symmetric_upper(A, ord);
    return refactorize(B, Sperm);
}

template <std::floating_point FloatT = double,
          std::signed_integral IntT = int32_t>
inline LDLFactors<FloatT, IntT>
factorize(const SparseUpperCSC<FloatT, IntT> &A) {
    auto S = analyze_fast(A);
    return refactorize(A, S);
}

template <std::floating_point FloatT = double,
          std::signed_integral IntT = int32_t>
inline LDLFactors<FloatT, IntT>
factorize_with_ordering(const SparseUpperCSC<FloatT, IntT> &A,
                        const Ordering<IntT> &ord) {
    auto B = permute_symmetric_upper(A, ord);
    auto S = analyze_fast(B);
    return refactorize(B, S);
}

// --------- Solves ----------
template <std::floating_point FloatT = double,
          std::signed_integral IntT = int32_t>
inline void L_solve(const LDLFactors<FloatT, IntT> &F, FloatT *x) {
    const IntT n = F.n;
    for (IntT i = 0; i < n; ++i) {
        const FloatT xi = x[(size_t)i];
        for (IntT p = F.Lp[(size_t)i]; p < F.Lp[(size_t)i + 1]; ++p)
            x[(size_t)F.Li[(size_t)p]] -= F.Lx[(size_t)p] * xi;
    }
}
template <std::floating_point FloatT = double,
          std::signed_integral IntT = int32_t>
inline void Lt_solve(const LDLFactors<FloatT, IntT> &F, FloatT *x) {
    const IntT n = F.n;
    for (IntT i = n - 1; i >= 0; --i) {
        FloatT xi = x[(size_t)i];
        for (IntT p = F.Lp[(size_t)i]; p < F.Lp[(size_t)i + 1]; ++p)
            xi -= F.Lx[(size_t)p] * x[(size_t)F.Li[(size_t)p]];
        x[(size_t)i] = xi;
        if (i == 0)
            break;
    }
}
template <std::floating_point FloatT = double,
          std::signed_integral IntT = int32_t>
inline void solve(const LDLFactors<FloatT, IntT> &F, FloatT *x) {
    L_solve(F, x);
    detail::qd_par_for_n(static_cast<std::size_t>(F.n),
        [&](std::size_t i){ x[i] *= F.Dinv[i]; });
    Lt_solve(F, x);
}
template <std::floating_point FloatT = double,
          std::signed_integral IntT = int32_t>
inline void solve_with_ordering(const LDLFactors<FloatT, IntT> &F,
                                const Ordering<IntT> &ord,
                                FloatT *x /* in: b, out: x */) {
    const IntT n = F.n;
    if (ord.n != n)
        throw InvalidMatrixError("ordering mismatch");
    std::vector<FloatT> xp((size_t)n);

    // xp = P x    (permute in)
    detail::qd_par_for_n(static_cast<std::size_t>(n),
        [&](std::size_t i){ xp[(size_t)ord.perm[i]] = x[i]; });

    L_solve(F, xp.data());

    // scale by Dinv
    detail::qd_par_for_n(static_cast<std::size_t>(n),
        [&](std::size_t i){ xp[i] *= F.Dinv[i]; });

    Lt_solve(F, xp.data());

    // x = P x     (permute out; same mapping kept to match caller expectations)
    detail::qd_par_for_n(static_cast<std::size_t>(n),
        [&](std::size_t i){ x[i] = xp[(size_t)ord.perm[i]]; });
}

// --------- Residual-based refinement (single-thread; no atomics) ----------
template <std::floating_point FloatT = double,
          std::signed_integral IntT = int32_t>
inline void refine(const SparseUpperCSC<FloatT, IntT> &A,
                   const LDLFactors<FloatT, IntT> &F, FloatT *x,
                   const FloatT *b, int iters = 2,
                   const Ordering<IntT> *ord = nullptr) {
    const IntT n = A.n;
    std::vector<FloatT> r((size_t)n), t((size_t)n);
    for (int it = 0; it < iters; ++it) {
        // r = b - A*x  (use upper symmetric structure)
        std::fill(r.begin(), r.end(), FloatT{0});
        for (IntT j = 0; j < n; ++j) {
            const FloatT xj = x[(size_t)j];
            for (IntT p = A.Ap[(size_t)j]; p < A.Ap[(size_t)j + 1]; ++p) {
                const IntT i = A.Ai[(size_t)p];
                const FloatT v = A.Ax[(size_t)p];
                r[(size_t)i] += v * xj;
                if (i != j)
                    r[(size_t)j] += v * x[(size_t)i];
            }
        }
        for (IntT i = 0; i < n; ++i)
            r[(size_t)i] = b[(size_t)i] - r[(size_t)i];

        // t = A^{-1} r
        std::copy(r.begin(), r.end(), t.begin());
        if (ord)
            solve_with_ordering(F, *ord, t.data());
        else
            solve(F, t.data());

        // x += t
        detail::qd_par_for_n(static_cast<std::size_t>(n),
            [&](std::size_t i){ x[i] += t[i]; });
    }
}

// --------- Aliases ----------
using SparseD32 = SparseUpperCSC<double, int32_t>;
using SparseD64 = SparseUpperCSC<double, int64_t>;
using SparseF32 = SparseUpperCSC<float, int32_t>;
using Symb32 = Symbolic<int32_t>;
using Symb64 = Symbolic<int64_t>;
using LDL32 = LDLFactors<double, int32_t>;
using LDL64 = LDLFactors<double, int64_t>;

} // namespace qdldl23
