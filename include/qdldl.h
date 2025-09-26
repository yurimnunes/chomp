#pragma once
// qdldl23.hpp — header-only LDLᵀ (QDLDL-style) for upper CSC
// C++23, optimized for performance, exact-zero pivot, no sign forcing.
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
#if defined(QDLDL23_USE_STDEXEC)
  #include <stdexec/execution.hpp>
  #include <exec/static_thread_pool.hpp>
  #include <thread>
  #define QDLDL23_HAS_STDEXEC 1
#else
  #define QDLDL23_HAS_STDEXEC 0
#endif

// SIMD support detection
#if defined(__AVX2__) && defined(__FMA__)
  #include <immintrin.h>
  #define QDLDL23_HAS_AVX2 1
#elif defined(__SSE2__)
  #include <emmintrin.h>
  #define QDLDL23_HAS_SSE2 1
#else
  #define QDLDL23_HAS_AVX2 0
  #define QDLDL23_HAS_SSE2 0
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
  
  // Safe SIMD operations with bounds checking
  inline void simd_scale_array(double* x, const double* scale, size_t n) {
    #if QDLDL23_HAS_AVX2
    const size_t simd_end = n - (n % 4);
    for (size_t i = 0; i < simd_end; i += 4) {
      __m256d vx = _mm256_loadu_pd(&x[i]);
      __m256d vs = _mm256_loadu_pd(&scale[i]);
      vx = _mm256_mul_pd(vx, vs);
      _mm256_storeu_pd(&x[i], vx);
    }
    for (size_t i = simd_end; i < n; ++i) {
      x[i] *= scale[i];
    }
    #else
    for (size_t i = 0; i < n; ++i) {
      x[i] *= scale[i];
    }
    #endif
  }
  
  inline void simd_vector_sub(double* result, const double* a, const double* b, size_t n) {
    #if QDLDL23_HAS_AVX2
    const size_t simd_end = n - (n % 4);
    for (size_t i = 0; i < simd_end; i += 4) {
      __m256d va = _mm256_loadu_pd(&a[i]);
      __m256d vb = _mm256_loadu_pd(&b[i]);
      __m256d vr = _mm256_sub_pd(va, vb);
      _mm256_storeu_pd(&result[i], vr);
    }
    for (size_t i = simd_end; i < n; ++i) {
      result[i] = a[i] - b[i];
    }
    #else
    for (size_t i = 0; i < n; ++i) {
      result[i] = a[i] - b[i];
    }
    #endif
  }
  
  inline void simd_vector_add(double* result, const double* a, const double* b, size_t n) {
    #if QDLDL23_HAS_AVX2
    const size_t simd_end = n - (n % 4);
    for (size_t i = 0; i < simd_end; i += 4) {
      __m256d va = _mm256_loadu_pd(&a[i]);
      __m256d vb = _mm256_loadu_pd(&b[i]);
      __m256d vr = _mm256_add_pd(va, vb);
      _mm256_storeu_pd(&result[i], vr);
    }
    for (size_t i = simd_end; i < n; ++i) {
      result[i] = a[i] + b[i];
    }
    #else
    for (size_t i = 0; i < n; ++i) {
      result[i] = a[i] + b[i];
    }
    #endif
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

    // Optimized normalize: sort by row, coalesce duplicates
    void finalize_upper_inplace(bool require_diag = true) {
        if (n == 0) return;
        
        std::vector<std::pair<IntT, FloatT>> col_workspace;
        
        for (IntT j = 0; j < n; ++j) {
            const IntT p0 = Ap[(size_t)j], p1 = Ap[(size_t)j + 1];
            if (p0 > p1)
                throw InvalidMatrixError("Ap must be nondecreasing");
            if (p0 == p1)
                throw InvalidMatrixError("Empty column");

            const size_t col_size = static_cast<size_t>(p1 - p0);
            col_workspace.clear();
            col_workspace.reserve(col_size);
            
            for (IntT p = p0; p < p1; ++p) {
                const IntT i = Ai[(size_t)p];
                if (i < 0 || i >= n)
                    throw InvalidMatrixError("Row index OOB");
                if (i > j)
                    throw InvalidMatrixError(
                        "Lower-triangular entry (need upper+diag only)");
                col_workspace.emplace_back(i, Ax[(size_t)p]);
            }
            
            // Sort by row - use stable_sort to maintain order for equal elements
            std::stable_sort(col_workspace.begin(), col_workspace.end(),
                           [](const auto &a, const auto &b) { return a.first < b.first; });

            // Coalesce duplicates
            IntT w = p0;
            bool has_diag = false;
            for (size_t k = 0; k < col_workspace.size();) {
                IntT r = col_workspace[k].first;
                FloatT sum = col_workspace[k].second;
                size_t k2 = k + 1;
                while (k2 < col_workspace.size() && col_workspace[k2].first == r) {
                    sum += col_workspace[k2].second;
                    ++k2;
                }
                Ai[(size_t)w] = r;
                Ax[(size_t)w] = sum;
                if (r == j)
                    has_diag = true;
                ++w;
                k = k2;
            }

            // Compact: update Ap deltas for later columns
            const IntT new_p1 = w;
            const IntT drop = p1 - new_p1;
            if (drop > 0) {
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
        std::vector<bool> seen((size_t)o.n, false);
        for (IntT i = 0; i < o.n; ++i) {
            IntT pi = o.perm[(size_t)i];
            if (pi < 0 || pi >= o.n || seen[(size_t)pi])
                throw InvalidMatrixError("Invalid permutation");
            seen[(size_t)pi] = true;
            o.iperm[(size_t)pi] = i;
        }
        return o;
    }
};

// Symmetric permutation: B = P A Pᵀ (keep **upper**), optimized version
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
            const IntT col = std::max(pi, pj);
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

    // Scatter entries
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

    // Sort and coalesce each column
    for (IntT j = 0; j < n; ++j) {
        const IntT p0 = Bp[(size_t)j];
        const IntT p1 = Bp[(size_t)j + 1];
        const IntT len = p1 - p0;
        
        if (len <= 1) continue;

        // Create index array for sorting
        std::vector<IntT> indices((size_t)len);
        std::iota(indices.begin(), indices.end(), IntT{0});
        
        // Sort indices by corresponding row values
        std::stable_sort(indices.begin(), indices.end(), 
                        [&](IntT a, IntT b) {
                            return Bi[(size_t)(p0 + a)] < Bi[(size_t)(p0 + b)];
                        });

        // Apply permutation
        std::vector<IntT> tmp_i((size_t)len);
        std::vector<FloatT> tmp_x((size_t)len);
        for (IntT k = 0; k < len; ++k) {
            tmp_i[(size_t)k] = Bi[(size_t)(p0 + indices[(size_t)k])];
            tmp_x[(size_t)k] = Bx[(size_t)(p0 + indices[(size_t)k])];
        }

        // Coalesce duplicates
        IntT w = 0;
        bool has_diag = false;
        for (IntT k = 0; k < len;) {
            IntT r = tmp_i[(size_t)k];
            FloatT s = tmp_x[(size_t)k];
            IntT k2 = k + 1;
            while (k2 < len && tmp_i[(size_t)k2] == r) {
                s += tmp_x[(size_t)k2];
                ++k2;
            }
            Bi[(size_t)(p0 + w)] = r;
            Bx[(size_t)(p0 + w)] = s;
            if (r == j)
                has_diag = true;
            ++w;
            k = k2;
        }
        
        // Update pointers for dropped entries
        const IntT drop = len - w;
        if (drop > 0) {
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
    return B;
}

// --------- Symbolic structure ----------
template <std::signed_integral IntT = int32_t> struct Symbolic {
    std::vector<IntT> etree; // parent[j] or -1
    std::vector<IntT> Lnz;   // strictly-lower count per column of L
    std::vector<IntT> Lp;    // size n+1, cumulated Lnz
    IntT n{0};
};

// Liu (1986) elimination tree for upper CSC - safe implementation
template <std::floating_point FloatT = double,
          std::signed_integral IntT = int32_t>
inline void etree_upper_liu(const SparseUpperCSC<FloatT, IntT> &A,
                            std::vector<IntT> &parent) {
    const IntT n = A.n;
    parent.assign((size_t)n, IntT{-1});
    std::vector<IntT> ancestor((size_t)n, IntT{-1});

    auto find_root = [&](IntT j) -> IntT {
        IntT r = j;
        while (ancestor[(size_t)r] != IntT{-1}) {
            r = ancestor[(size_t)r];
        }
        // Path compression
        IntT cur = j;
        while (ancestor[(size_t)cur] != IntT{-1}) {
            const IntT nxt = ancestor[(size_t)cur];
            ancestor[(size_t)cur] = r;
            cur = nxt;
        }
        return r;
    };

    for (IntT j = 0; j < n; ++j) {
        for (IntT p = A.Ap[(size_t)j]; p < A.Ap[(size_t)j + 1]; ++p) {
            IntT i = A.Ai[(size_t)p];
            if (i == j) continue; // skip diag
            
            while (true) {
                IntT r = (ancestor[(size_t)i] == IntT{-1}) ? -1 : find_root(i);
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
    std::vector<IntT> prevnz((size_t)n, IntT{-1});

    for (IntT j = 0; j < n; ++j) {
        for (IntT p = A.Ap[(size_t)j]; p < A.Ap[(size_t)j + 1]; ++p) {
            const IntT i = A.Ai[(size_t)p];
            if (i == j) continue; // skip diag
            
            IntT k = i;
            while (k != -1 && k < j && prevnz[(size_t)k] != j) {
                ++Lnz[(size_t)k];
                prevnz[(size_t)k] = j;
                k = parent[(size_t)k];
            }
        }
    }
}

// Symbolic analysis
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

    // Work arrays
    std::vector<IntT> Lnext((size_t)n);
    for (IntT i = 0; i < n; ++i) Lnext[(size_t)i] = R.Lp[(size_t)i];

    std::vector<FloatT> y((size_t)n, FloatT{0});
    std::vector<IntT> yIdx((size_t)n);
    std::vector<IntT> Ebuf((size_t)n);
    std::vector<IntT> mark((size_t)n, IntT{0});
    
    IntT curmark = 1;

    auto is_marked = [&](IntT i) -> bool { 
        return i >= 0 && i < n && mark[(size_t)i] == curmark; 
    };
    auto set_mark = [&](IntT i) { 
        if (i >= 0 && i < n) mark[(size_t)i] = curmark; 
    };

    // Handle k=0 case
    if (n > 0) {
        if (A.Ap[0] >= A.Ap[1] || A.Ai[(size_t)A.Ap[0]] != 0)
            throw FactorizationError("Missing diagonal at column 0");
        R.D[0] = A.Ax[(size_t)A.Ap[0]];
        if (R.D[0] == FloatT{0})
            throw FactorizationError("Zero pivot at column 0");
        if (R.D[0] > FloatT{0})
            ++R.num_pos;
        R.Dinv[0] = FloatT{1} / R.D[0];
    }

    for (IntT k = 1; k < n; ++k) {
        // Increment timestamp with overflow protection
        if (++curmark == std::numeric_limits<IntT>::max()) {
            std::fill(mark.begin(), mark.end(), IntT{0});
            curmark = 1;
        }

        // Scatter A(:,k) into y
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
            
            if (i < 0 || i >= n) continue; // Safety check
            
            y[(size_t)i] = v;
            if (!is_marked(i)) {
                set_mark(i);
                // Walk up elimination tree
                IntT next = i, nnzE = 0;
                while (next != -1 && next < k && nnzE < n) { // Prevent infinite loops
                    Ebuf[(size_t)nnzE++] = next;
                    next = S.etree[(size_t)next];
                    if (next >= 0 && next < k && is_marked(next)) break;
                    if (next >= 0 && next < k) set_mark(next);
                }
                // Reverse order for elimination
                for (IntT e = nnzE - 1; e >= 0 && nnzY < n; --e) {
                    yIdx[(size_t)nnzY++] = Ebuf[(size_t)e];
                }
            }
        }
        
        if (!diag_seen)
            throw FactorizationError("Missing diagonal at column " + std::to_string(k));

        // Eliminate along listed columns
        for (IntT t = nnzY - 1; t >= 0; --t) {
            const IntT c = yIdx[(size_t)t];
            if (c < 0 || c >= n) continue; // Safety check
            
            const IntT j0 = R.Lp[(size_t)c];
            IntT &j1 = Lnext[(size_t)c];
            const FloatT yc = y[(size_t)c];

            // y -= L(:,c)*yc
            for (IntT j = j0; j < j1; ++j) {
                const IntT idx = R.Li[(size_t)j];
                if (idx >= 0 && idx < n) {
                    y[(size_t)idx] -= R.Lx[(size_t)j] * yc;
                }
            }

            if (j1 < (IntT)R.Li.size()) {
                R.Li[(size_t)j1] = k;
                const FloatT lk = yc * R.Dinv[(size_t)c];
                R.Lx[(size_t)j1] = lk;
                R.D[(size_t)k] -= yc * lk;
                ++j1;
            }

            y[(size_t)c] = FloatT{0};
        }

        if (R.D[(size_t)k] == FloatT{0})
            throw FactorizationError("Zero pivot at column " + std::to_string(k));
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
        for (IntT p = F.Lp[(size_t)i]; p < F.Lp[(size_t)i + 1]; ++p) {
            const IntT idx = F.Li[(size_t)p];
            if (idx >= 0 && idx < n) {
                x[(size_t)idx] -= F.Lx[(size_t)p] * xi;
            }
        }
    }
}

template <std::floating_point FloatT = double,
          std::signed_integral IntT = int32_t>
inline void Lt_solve(const LDLFactors<FloatT, IntT> &F, FloatT *x) {
    const IntT n = F.n;
    for (IntT i = n - 1; i >= 0; --i) {
        FloatT xi = x[(size_t)i];
        for (IntT p = F.Lp[(size_t)i]; p < F.Lp[(size_t)i + 1]; ++p) {
            const IntT idx = F.Li[(size_t)p];
            if (idx >= 0 && idx < n) {
                xi -= F.Lx[(size_t)p] * x[(size_t)idx];
            }
        }
        x[(size_t)i] = xi;
        if (i == 0) break;
    }
}

template <std::floating_point FloatT = double,
          std::signed_integral IntT = int32_t>
inline void solve(const LDLFactors<FloatT, IntT> &F, FloatT *x) {
    L_solve(F, x);
    
    // Diagonal scaling with SIMD optimization
    const IntT n = F.n;
    if constexpr (std::is_same_v<FloatT, double>) {
        detail::simd_scale_array(x, F.Dinv.data(), static_cast<size_t>(n));
    } else {
        detail::qd_par_for_n(static_cast<std::size_t>(n),
            [&](std::size_t i){ x[i] *= F.Dinv[i]; });
    }
    
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

    // Permute input: xp = P x
    detail::qd_par_for_n(static_cast<std::size_t>(n),
        [&](std::size_t i){ 
            const IntT pi = ord.perm[i];
            if (pi >= 0 && pi < n) {
                xp[(size_t)pi] = x[i]; 
            }
        });

    L_solve(F, xp.data());

    // Diagonal scaling
    if constexpr (std::is_same_v<FloatT, double>) {
        detail::simd_scale_array(xp.data(), F.Dinv.data(), static_cast<size_t>(n));
    } else {
        detail::qd_par_for_n(static_cast<std::size_t>(n),
            [&](std::size_t i){ xp[i] *= F.Dinv[i]; });
    }

    Lt_solve(F, xp.data());

    // Permute output: x = P xp
    detail::qd_par_for_n(static_cast<std::size_t>(n),
        [&](std::size_t i){ 
            const IntT pi = ord.perm[i];
            if (pi >= 0 && pi < n) {
                x[i] = xp[(size_t)pi]; 
            }
        });
}

// Optimized symmetric SpMV for upper CSC
template <std::floating_point FloatT, std::signed_integral IntT>
inline void sym_spmv_upper(const qdldl23::SparseUpperCSC<FloatT, IntT>& A,
                           const FloatT* __restrict__ x,
                           FloatT* __restrict__ y) {
    const IntT n = A.n;
    std::fill(y, y + (size_t)n, FloatT{0});
    
    for (IntT j = 0; j < n; ++j) {
        const FloatT xj = x[(size_t)j];
        const IntT p0 = A.Ap[(size_t)j];
        const IntT p1 = A.Ap[(size_t)j + 1];
        
        for (IntT p = p0; p < p1; ++p) {
            const IntT i = A.Ai[(size_t)p];
            if (i >= 0 && i < n) {
                const FloatT v = A.Ax[(size_t)p];
                y[(size_t)i] += v * xj;                // A(i,j)*x(j)
                if (i != j) {
                    y[(size_t)j] += v * x[(size_t)i];  // A(j,i)*x(i)
                }
            }
        }
    }
}

// Safe residual-based refinement
template <std::floating_point FloatT, std::signed_integral IntT>
inline void refine(const SparseUpperCSC<FloatT, IntT>& A,
                   const LDLFactors<FloatT, IntT>& F, FloatT* x,
                   const FloatT* b, int iters = 2,
                   const Ordering<IntT>* ord = nullptr) {
    const IntT n = A.n;
    if (n <= 0) return;
    
    std::vector<FloatT> r((size_t)n);
    std::vector<FloatT> t((size_t)n);
    
    for (int it = 0; it < iters; ++it) {
        // Compute residual: r = b - A*x
        sym_spmv_upper(A, x, r.data());
        
        if constexpr (std::is_same_v<FloatT, double>) {
            detail::simd_vector_sub(r.data(), b, r.data(), static_cast<size_t>(n));
        } else {
            for (IntT i = 0; i < n; ++i) {
                r[i] = b[i] - r[i];
            }
        }
        
        // Solve for correction: F * t = r
        std::copy(r.begin(), r.end(), t.begin());
        if (ord) {
            solve_with_ordering(F, *ord, t.data());
        } else {
            solve(F, t.data());
        }
        
        // Apply correction: x = x + t
        if constexpr (std::is_same_v<FloatT, double>) {
            detail::simd_vector_add(x, x, t.data(), static_cast<size_t>(n));
        } else {
            #if QDLDL23_HAS_STDEXEC
            detail::qd_par_for_n((size_t)n, [&](size_t i){ x[i] += t[i]; });
            #else
            for (IntT i = 0; i < n; ++i) {
                x[i] += t[i];
            }
            #endif
        }
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