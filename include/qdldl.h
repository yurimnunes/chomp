#pragma once

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <cstring> // memcpy
#include <limits>
#include <numeric>
#include <queue>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
namespace qdldl {

// ===================== Branch prediction =====================
#if defined(__clang__) || defined(__GNUC__)
#define QDLL_LIKELY(x) (__builtin_expect(!!(x), 1))
#define QDLL_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
#define QDLL_LIKELY(x) (x)
#define QDLL_UNLIKELY(x) (x)
#endif

// ===================== Concepts =====================
template <typename T>
concept FloatingPoint = std::floating_point<T>;
template <typename T>
concept SignedInteger = std::signed_integral<T>;

// ===================== Exceptions =====================
class FactorizationError : public std::runtime_error {
public:
    explicit FactorizationError(const std::string &msg)
        : std::runtime_error(msg) {}
};
class InvalidMatrixError : public std::runtime_error {
public:
    explicit InvalidMatrixError(const std::string &msg)
        : std::runtime_error(msg) {}
};

// ===================== Constants =====================
template <SignedInteger IntType>
constexpr IntType UNKNOWN = static_cast<IntType>(-1);

// ===================== SparseMatrix (CSC, upper incl diag)
// =====================
template <FloatingPoint FloatType = double,
          SignedInteger IntType = std::int64_t>
class SparseMatrix {
public:
    std::vector<IntType> col_ptr_;  // Ap, size n+1
    std::vector<IntType> row_idx_;  // Ai, size nnz = Ap[n]
    std::vector<FloatType> values_; // Ax, size nnz
    IntType n_;                     // square dimension

    SparseMatrix() : n_(0) {}

    SparseMatrix(IntType n, std::vector<IntType> col_ptr,
                 std::vector<IntType> row_idx, std::vector<FloatType> values)
        : col_ptr_(std::move(col_ptr)), row_idx_(std::move(row_idx)),
          values_(std::move(values)), n_(n) {
        if (n_ < 0)
            throw InvalidMatrixError("n must be nonnegative");
        if (col_ptr_.size() != static_cast<std::size_t>(n_ + 1)) {
            throw InvalidMatrixError("Ap size must be n+1");
        }
        const std::size_t nnz =
            static_cast<std::size_t>(col_ptr_[static_cast<std::size_t>(n_)]);
        if (row_idx_.size() != nnz || values_.size() != nnz) {
            throw InvalidMatrixError("Ai and Ax size must equal nnz = Ap[n]");
        }
        validate_structure();
    }

    [[nodiscard]] IntType size() const noexcept { return n_; }
    [[nodiscard]] std::span<const IntType> col_ptr() const noexcept {
        return col_ptr_;
    }
    [[nodiscard]] std::span<const IntType> row_idx() const noexcept {
        return row_idx_;
    }
    [[nodiscard]] std::span<const FloatType> values() const noexcept {
        return values_;
    }

    // Friends for permutation builder
    template <FloatingPoint FT, SignedInteger IT>
    friend SparseMatrix<FT, IT>
    make_csc(IntType n, std::vector<IntType> &&col_ptr,
             std::vector<IntType> &&row_idx, std::vector<FloatType> &&values);

    void validate_structure() const {
        for (IntType j = 0; j < n_; ++j) {
            if (QDLL_UNLIKELY(col_ptr_[static_cast<std::size_t>(j)] >
                              col_ptr_[static_cast<std::size_t>(j + 1)])) {
                throw InvalidMatrixError("Ap must be nondecreasing");
            }
            if (QDLL_UNLIKELY(col_ptr_[static_cast<std::size_t>(j)] ==
                              col_ptr_[static_cast<std::size_t>(j + 1)])) {
                throw InvalidMatrixError("Empty column detected");
            }
        }
        for (IntType j = 0; j < n_; ++j) {
            const IntType p0 = col_ptr_[static_cast<std::size_t>(j)];
            const IntType p1 = col_ptr_[static_cast<std::size_t>(j + 1)];
            for (IntType p = p0; p < p1; ++p) {
                const std::size_t ps = static_cast<std::size_t>(p);
                const IntType i = row_idx_[ps];
                if (QDLL_UNLIKELY(i < 0 || i >= n_)) {
                    throw InvalidMatrixError("Row index out of bounds");
                }
                if (QDLL_UNLIKELY(i > j)) {
                    throw InvalidMatrixError("Lower-triangular entry found "
                                             "(expect upper+diag only)");
                }
            }
        }
    }
};

template <FloatingPoint FT, SignedInteger IT>
inline SparseMatrix<FT, IT> make_csc(IT n, std::vector<IT> &&col_ptr,
                                     std::vector<IT> &&row_idx,
                                     std::vector<FT> &&values) {
    SparseMatrix<FT, IT> A;
    A.n_ = n;
    A.col_ptr_ = std::move(col_ptr);
    A.row_idx_ = std::move(row_idx);
    A.values_ = std::move(values);
    A.validate_structure();
    return A;
}

// ===================== Factorization result =====================
template <FloatingPoint FloatType = double,
          SignedInteger IntType = std::int64_t>
struct LDLFactorization {
    std::vector<IntType> L_col_ptr;  // size n+1
    std::vector<IntType> L_row_idx;  // size nnz(L)
    std::vector<FloatType> L_values; // size nnz(L)
    std::vector<FloatType> D;        // size n
    std::vector<FloatType> D_inv;    // size n
    IntType positive_eigenvalues;    // count of D > 0

    explicit LDLFactorization(IntType n)
        : L_col_ptr(static_cast<std::size_t>(n + 1)), L_row_idx(), L_values(),
          D(static_cast<std::size_t>(n)), D_inv(static_cast<std::size_t>(n)),
          positive_eigenvalues(0) {}
};

// ===================== Working memory (stamp markers) =====================
template <FloatingPoint FloatType = double,
          SignedInteger IntType = std::int64_t>
class WorkingMemory {
public:
    std::vector<IntType> stamp;    // size n
    std::vector<IntType> int_work; // yIdx (n) | elimBuffer (n) | LNextSpace (n)
    std::vector<FloatType> float_work; // yVals (n)
    IntType current_stamp{1};

    explicit WorkingMemory(IntType n)
        : stamp(static_cast<std::size_t>(n), IntType{0}),
          int_work(static_cast<std::size_t>(3 * n), IntType{0}),
          float_work(static_cast<std::size_t>(n), FloatType{0}) {}

    inline void next_column() noexcept {
        if (++current_stamp == 0) {
            std::fill(stamp.begin(), stamp.end(), IntType{0});
            current_stamp = 1;
        }
    }
    inline bool is_marked(IntType i) const noexcept {
        return stamp[static_cast<std::size_t>(i)] == current_stamp;
    }
    inline void mark(IntType i) noexcept {
        stamp[static_cast<std::size_t>(i)] = current_stamp;
    }
};

// ===================== Symbolic structure (for reuse) =====================
template <FloatingPoint FloatType = double,
          SignedInteger IntType = std::int64_t>
struct Symbolic {
    std::vector<IntType> etree;
    std::vector<IntType> Lnz;
    std::vector<IntType> L_col_ptr;
    IntType n{};
};

// ===================== Ordering (perm hooks) =====================
template <SignedInteger IntType = std::int64_t> struct Ordering {
    std::vector<IntType> perm;  // size n, values 0..n-1
    std::vector<IntType> iperm; // inverse permutation
    IntType n{};
};

template <SignedInteger IntType>
inline Ordering<IntType> make_ordering(std::vector<IntType> perm) {
    Ordering<IntType> o;
    o.n = static_cast<IntType>(perm.size());
    o.perm = std::move(perm);
    o.iperm.resize(static_cast<std::size_t>(o.n), IntType{-1});
    std::vector<char> seen(static_cast<std::size_t>(o.n), 0);
    for (IntType i = 0; i < o.n; ++i) {
        IntType p = o.perm[static_cast<std::size_t>(i)];
        if (p < 0 || p >= o.n || seen[static_cast<std::size_t>(p)])
            throw InvalidMatrixError("Invalid permutation");
        seen[static_cast<std::size_t>(p)] = 1;
        o.iperm[static_cast<std::size_t>(p)] = i;
    }
    return o;
}

// Symmetric permutation of upper-triangular CSC: A_perm = P*A*P^T (upper only)
template <FloatingPoint FT = double, SignedInteger IT = int64_t>
inline SparseMatrix<FT, IT>
permute_symmetric_upper(const SparseMatrix<FT, IT> &A,
                        const Ordering<IT> &ord) {
    const IT n = A.size();
    if (ord.n != n)
        throw InvalidMatrixError("Ordering size mismatch");

    const auto Ap = A.col_ptr();
    const auto Ai = A.row_idx();
    const auto Ax = A.values();
    const IT *__restrict P = ord.perm.data();

    // First pass: count nnz per permuted column (upper part only)
    std::vector<IT> counts(static_cast<std::size_t>(n), IT{0});
    for (IT j = 0; j < n; ++j) {
        const IT pj = P[static_cast<std::size_t>(j)];
        const IT p0 = Ap[static_cast<std::size_t>(j)];
        const IT p1 = Ap[static_cast<std::size_t>(j + 1)];
        for (IT p = p0; p < p1; ++p) {
            const IT i = Ai[static_cast<std::size_t>(p)];
            const IT pi = P[static_cast<std::size_t>(i)];
            // keep upper: (pi <= pj); if not, swap roles to push to upper
            if (pi <= pj) {
                counts[static_cast<std::size_t>(pj)]++;
            } else {
                counts[static_cast<std::size_t>(
                    pi)]++; // move to the other column's upper
            }
        }
    }

    // Build column pointers
    std::vector<IT> Cp(static_cast<std::size_t>(n + 1), IT{0});
    for (IT j = 0; j < n; ++j)
        Cp[static_cast<std::size_t>(j + 1)] =
            Cp[static_cast<std::size_t>(j)] +
            counts[static_cast<std::size_t>(j)];
    const IT nnz = Cp[static_cast<std::size_t>(n)];
    std::vector<IT> Ci(static_cast<std::size_t>(nnz));
    std::vector<FT> Cx(static_cast<std::size_t>(nnz));

    // Temp write heads
    std::vector<IT> whead = Cp;

    // Second pass: fill
    for (IT j = 0; j < n; ++j) {
        const IT pj = P[static_cast<std::size_t>(j)];
        const IT p0 = Ap[static_cast<std::size_t>(j)];
        const IT p1 = Ap[static_cast<std::size_t>(j + 1)];
        for (IT p = p0; p < p1; ++p) {
            const IT i = Ai[static_cast<std::size_t>(p)];
            const FT v = Ax[static_cast<std::size_t>(p)];
            const IT pi = P[static_cast<std::size_t>(i)];

            IT col = pj;
            IT row = pi;
            if (row > col) {
                std::swap(row, col);
            } // push to upper

            IT dst = whead[static_cast<std::size_t>(col)]++;
            Ci[static_cast<std::size_t>(dst)] = row;
            Cx[static_cast<std::size_t>(dst)] = v;
        }
    }

    // Ensure row indices in each column are sorted (optional but nice)
    for (IT j = 0; j < n; ++j) {
        IT p0 = Cp[static_cast<std::size_t>(j)];
        IT p1 = Cp[static_cast<std::size_t>(j + 1)];
        // simple insertion sort (columns are usually short)
        for (IT p = p0 + 1; p < p1; ++p) {
            IT ri = Ci[static_cast<std::size_t>(p)];
            FT rv = Cx[static_cast<std::size_t>(p)];
            IT q = p;
            while (q > p0 && Ci[static_cast<std::size_t>(q - 1)] > ri) {
                Ci[static_cast<std::size_t>(q)] =
                    Ci[static_cast<std::size_t>(q - 1)];
                Cx[static_cast<std::size_t>(q)] =
                    Cx[static_cast<std::size_t>(q - 1)];
                --q;
            }
            Ci[static_cast<std::size_t>(q)] = ri;
            Cx[static_cast<std::size_t>(q)] = rv;
        }
    }

    return make_csc<FT, IT>(n, std::move(Cp), std::move(Ci), std::move(Cx));
}

// ===================== Symmetric AMD ordering (header-only)
// =====================
template <FloatingPoint FT = double, SignedInteger IT = int64_t>
inline Ordering<IT>
simple_minimum_degree_ordering(const SparseMatrix<FT, IT> &A) {
    const IT n = A.size();
    const auto Ap = A.col_ptr();
    const auto Ai = A.row_idx();

    if (n <= 0)
        return make_ordering<IT>({});
    if (n == 1)
        return make_ordering<IT>({IT{0}});

    try {
        // Compute initial degrees (count off-diagonal entries)
        std::vector<IT> degree(static_cast<std::size_t>(n), IT{0});

        for (IT j = 0; j < n; ++j) {
            const IT p0 = Ap[static_cast<std::size_t>(j)];
            const IT p1 = Ap[static_cast<std::size_t>(j + 1)];
            for (IT p = p0; p < p1; ++p) {
                const IT i = Ai[static_cast<std::size_t>(p)];
                if (i != j && i >= 0 && i < n) {
                    degree[static_cast<std::size_t>(j)]++;
                    degree[static_cast<std::size_t>(i)]++; // symmetric
                }
            }
        }

        std::vector<bool> eliminated(static_cast<std::size_t>(n), false);
        std::vector<IT> elimination_order;
        elimination_order.reserve(static_cast<std::size_t>(n));

        // Simple greedy minimum degree
        for (IT step = 0; step < n; ++step) {
            IT min_degree = n + 1;
            IT pivot = -1;

            // Find uneliminated vertex with minimum degree
            for (IT v = 0; v < n; ++v) {
                if (!eliminated[static_cast<std::size_t>(v)] &&
                    degree[static_cast<std::size_t>(v)] < min_degree) {
                    min_degree = degree[static_cast<std::size_t>(v)];
                    pivot = v;
                }
            }

            if (pivot == -1)
                break;

            elimination_order.push_back(pivot);
            eliminated[static_cast<std::size_t>(pivot)] = true;

            // Update degrees of neighbors (simple approximation)
            for (IT j = 0; j < n; ++j) {
                if (eliminated[static_cast<std::size_t>(j)])
                    continue;

                const IT p0 = Ap[static_cast<std::size_t>(j)];
                const IT p1 = Ap[static_cast<std::size_t>(j + 1)];
                for (IT p = p0; p < p1; ++p) {
                    const IT i = Ai[static_cast<std::size_t>(p)];
                    if (i == pivot) {
                        degree[static_cast<std::size_t>(j)]--;
                        break;
                    }
                }
            }
        }

        // Convert to permutation
        std::vector<IT> perm(static_cast<std::size_t>(n));
        for (IT i = 0; i < static_cast<IT>(elimination_order.size()); ++i) {
            if (i < n && elimination_order[static_cast<std::size_t>(i)] < n) {
                perm[static_cast<std::size_t>(
                    elimination_order[static_cast<std::size_t>(i)])] = i;
            }
        }

        return make_ordering<IT>(std::move(perm));

    } catch (...) {
        // Fallback to identity
        std::vector<IT> identity(static_cast<std::size_t>(n));
        std::iota(identity.begin(), identity.end(), IT{0});
        return make_ordering<IT>(std::move(identity));
    }
}

// ===================== Reverse Cuthill-McKee Ordering =====================
template <FloatingPoint FT = double, SignedInteger IT = int64_t>
inline Ordering<IT> rcm_ordering(const SparseMatrix<FT, IT> &A) {
    const IT n = A.size();
    const auto Ap = A.col_ptr();
    const auto Ai = A.row_idx();

    if (n <= 0)
        return make_ordering<IT>({});
    if (n == 1)
        return make_ordering<IT>({IT{0}});

    try {
        // Build adjacency lists
        std::vector<std::vector<IT>> adj(static_cast<std::size_t>(n));
        std::vector<IT> degree(static_cast<std::size_t>(n), IT{0});

        for (IT j = 0; j < n; ++j) {
            const IT p0 = Ap[static_cast<std::size_t>(j)];
            const IT p1 = Ap[static_cast<std::size_t>(j + 1)];
            for (IT p = p0; p < p1; ++p) {
                const IT i = Ai[static_cast<std::size_t>(p)];
                if (i != j && i >= 0 && i < n) {
                    adj[static_cast<std::size_t>(i)].push_back(j);
                    adj[static_cast<std::size_t>(j)].push_back(i);
                }
            }
        }

        // Remove duplicates and compute degrees
        for (IT v = 0; v < n; ++v) {
            auto &neighbors = adj[static_cast<std::size_t>(v)];
            std::sort(neighbors.begin(), neighbors.end());
            neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                            neighbors.end());
            degree[static_cast<std::size_t>(v)] =
                static_cast<IT>(neighbors.size());
        }

        // Find starting vertex (minimum degree)
        IT start = 0;
        IT min_deg = degree[0];
        for (IT v = 1; v < n; ++v) {
            if (degree[static_cast<std::size_t>(v)] < min_deg) {
                min_deg = degree[static_cast<std::size_t>(v)];
                start = v;
            }
        }

        // BFS from starting vertex
        std::vector<IT> order;
        std::vector<bool> visited(static_cast<std::size_t>(n), false);
        std::queue<IT> queue;

        queue.push(start);
        visited[static_cast<std::size_t>(start)] = true;

        while (!queue.empty()) {
            IT v = queue.front();
            queue.pop();
            order.push_back(v);

            // Sort neighbors by degree (Cuthill-McKee heuristic)
            auto &neighbors = adj[static_cast<std::size_t>(v)];
            std::sort(neighbors.begin(), neighbors.end(),
                      [&degree](IT a, IT b) {
                          return degree[static_cast<std::size_t>(a)] <
                                 degree[static_cast<std::size_t>(b)];
                      });

            for (IT u : neighbors) {
                if (!visited[static_cast<std::size_t>(u)]) {
                    visited[static_cast<std::size_t>(u)] = true;
                    queue.push(u);
                }
            }
        }

        // Add any unvisited vertices (disconnected components)
        for (IT v = 0; v < n; ++v) {
            if (!visited[static_cast<std::size_t>(v)]) {
                order.push_back(v);
            }
        }

        // Reverse for RCM
        std::reverse(order.begin(), order.end());

        // Convert to permutation
        std::vector<IT> perm(static_cast<std::size_t>(n));
        for (IT i = 0; i < static_cast<IT>(order.size()); ++i) {
            if (i < n && order[static_cast<std::size_t>(i)] < n) {
                perm[static_cast<std::size_t>(
                    order[static_cast<std::size_t>(i)])] = i;
            }
        }

        return make_ordering<IT>(std::move(perm));

    } catch (...) {
        // Fallback to identity
        std::vector<IT> identity(static_cast<std::size_t>(n));
        std::iota(identity.begin(), identity.end(), IT{0});
        return make_ordering<IT>(std::move(identity));
    }
}

template<SignedInteger IT=int64_t>
inline Ordering<IT> natural_ordering(IT n) {
    std::vector<IT> identity(static_cast<std::size_t>(n));
    std::iota(identity.begin(), identity.end(), IT{0});
    return make_ordering<IT>(std::move(identity));
}

// ===================== Solver =====================
template <FloatingPoint FloatType = double,
          SignedInteger IntType = std::int64_t>
class QDLDLSolver {
public:
    // ---------- Elimination tree + L column counts ----------
    [[nodiscard]] static auto
    compute_etree(const SparseMatrix<FloatType, IntType> &A)
        -> std::pair<std::vector<IntType>, std::vector<IntType>> {
        const IntType n = A.size();
        const auto Ap = A.col_ptr();
        const auto Ai = A.row_idx();

        std::vector<IntType> work(static_cast<std::size_t>(n), IntType{0});
        std::vector<IntType> Lnz(static_cast<std::size_t>(n), IntType{0});
        std::vector<IntType> etree(static_cast<std::size_t>(n),
                                   UNKNOWN<IntType>);

        for (IntType j = 0; j < n; ++j) {
            work[static_cast<std::size_t>(j)] = j;

            const IntType p_end = Ap[static_cast<std::size_t>(j + 1)];
            for (IntType p = Ap[static_cast<std::size_t>(j)]; p < p_end; ++p) {
                IntType i = Ai[static_cast<std::size_t>(p)];
                if (QDLL_UNLIKELY(i > j)) {
                    throw InvalidMatrixError(
                        "Lower-triangular entry encountered in etree()");
                }
                while (work[static_cast<std::size_t>(i)] != j) {
                    if (etree[static_cast<std::size_t>(i)] ==
                        UNKNOWN<IntType>) {
                        etree[static_cast<std::size_t>(i)] = j;
                    }
                    ++Lnz[static_cast<std::size_t>(i)];
                    work[static_cast<std::size_t>(i)] = j;
                    i = etree[static_cast<std::size_t>(i)];
                    if (i == UNKNOWN<IntType>)
                        break;
                }
            }
        }

        IntType sum_L = 0;
        constexpr IntType IMAX = std::numeric_limits<IntType>::max();
        for (IntType i = 0; i < n; ++i) {
            const IntType add = Lnz[static_cast<std::size_t>(i)];
            if (QDLL_UNLIKELY(sum_L > IMAX - add)) {
                throw std::overflow_error("Nonzero count overflow in L");
            }
            sum_L += add;
        }
        return {std::move(etree), std::move(Lnz)};
    }

    [[nodiscard]] static Symbolic<FloatType, IntType>
    analyze(const SparseMatrix<FloatType, IntType> &A) {
        Symbolic<FloatType, IntType> S;
        S.n = A.size();
        auto [et, lnz] = compute_etree(A);
        S.etree = std::move(et);
        S.Lnz = std::move(lnz);
        S.L_col_ptr.resize(static_cast<std::size_t>(S.n + 1));
        S.L_col_ptr[0] = static_cast<IntType>(0);
        for (IntType i = 0; i < S.n; ++i) {
            S.L_col_ptr[static_cast<std::size_t>(i + 1)] =
                S.L_col_ptr[static_cast<std::size_t>(i)] +
                S.Lnz[static_cast<std::size_t>(i)];
        }
        return S;
    }

    [[nodiscard]] static Symbolic<FloatType, IntType>
    analyze_with_ordering(const SparseMatrix<FloatType, IntType> &A,
                          const Ordering<IntType> &ord) {
        const auto Aperm = permute_symmetric_upper<FloatType, IntType>(A, ord);
        return analyze(Aperm);
    }

    // ---------- Numeric factorization using precomputed symbolic ----------
    [[nodiscard]] static LDLFactorization<FloatType, IntType>
    refactorize(const SparseMatrix<FloatType, IntType> &A,
                const Symbolic<FloatType, IntType> &S) {
        const IntType n = S.n;
        const auto Ap = A.col_ptr();
        const auto Ai = A.row_idx();
        const auto Ax = A.values();

        LDLFactorization<FloatType, IntType> R(n);
        WorkingMemory<FloatType, IntType> W(n);

        R.L_col_ptr = S.L_col_ptr;
        const IntType nnzL = R.L_col_ptr[static_cast<std::size_t>(n)];
        R.L_row_idx.resize(static_cast<std::size_t>(nnzL));
        R.L_values.resize(static_cast<std::size_t>(nnzL));

        auto yw = std::span<IntType>(W.int_work);
        IntType *__restrict Yidx = yw.data();
        IntType *__restrict Ebuf = yw.data() + static_cast<std::size_t>(n);
        IntType *__restrict Lnext = yw.data() + static_cast<std::size_t>(2 * n);
        FloatType *__restrict yVals = W.float_work.data();

        IntType *__restrict Lp = R.L_col_ptr.data();
        IntType *__restrict Li = R.L_row_idx.data();
        FloatType *__restrict Lx = R.L_values.data();
        FloatType *__restrict D = R.D.data();
        FloatType *__restrict Dinv = R.D_inv.data();

        for (IntType i = 0; i < n; ++i)
            Lnext[static_cast<std::size_t>(i)] =
                Lp[static_cast<std::size_t>(i)];

        for (IntType k = 0; k < n; ++k) {
            W.next_column();
            IntType nnz_y = 0;
            bool diag_seen = false;

            const IntType pend = Ap[static_cast<std::size_t>(k + 1)];
            for (IntType ip = Ap[static_cast<std::size_t>(k)]; ip < pend;
                 ++ip) {
                const IntType b_idx = Ai[static_cast<std::size_t>(ip)];
                if (b_idx == k) {
                    D[static_cast<std::size_t>(k)] =
                        Ax[static_cast<std::size_t>(ip)];
                    diag_seen = true;
                    continue;
                }
                yVals[static_cast<std::size_t>(b_idx)] =
                    Ax[static_cast<std::size_t>(ip)];

                IntType next = S.etree[static_cast<std::size_t>(b_idx)];
                // Add starting node
                if (!W.is_marked(b_idx)) {
                    W.mark(b_idx);
                    Ebuf[0] = b_idx;
                    IntType nnzE = 1;

                    // climb etree
                    while (next != UNKNOWN<IntType> && next < k) {
                        if (W.is_marked(next))
                            break;
                        W.mark(next);
                        Ebuf[static_cast<std::size_t>(nnzE++)] = next;
                        next = S.etree[static_cast<std::size_t>(next)];
                    }
                    while (nnzE > 0) {
                        Yidx[static_cast<std::size_t>(nnz_y++)] =
                            Ebuf[static_cast<std::size_t>(--nnzE)];
                    }
                }
            }

            if (QDLL_UNLIKELY(!diag_seen)) {
                throw FactorizationError("Missing diagonal at column " +
                                         std::to_string(k));
            }

            while (nnz_y > 0) {
                const IntType c = Yidx[static_cast<std::size_t>(--nnz_y)];
                const IntType j0 = Lp[static_cast<std::size_t>(c)];
                IntType &j1 = Lnext[static_cast<std::size_t>(c)];
                const FloatType yc = yVals[static_cast<std::size_t>(c)];

                for (IntType j = j0; j < j1; ++j) {
                    yVals[static_cast<std::size_t>(
                        Li[static_cast<std::size_t>(j)])] -=
                        Lx[static_cast<std::size_t>(j)] * yc;
                }

                Li[static_cast<std::size_t>(j1)] = k;
                const FloatType lk = yc / D[static_cast<std::size_t>(c)];
                Lx[static_cast<std::size_t>(j1)] = lk;
                D[static_cast<std::size_t>(k)] -= yc * lk;
                ++j1;

                yVals[static_cast<std::size_t>(c)] = FloatType{0};
            }

            if (QDLL_UNLIKELY(D[static_cast<std::size_t>(k)] == FloatType{0})) {
                throw FactorizationError("Zero pivot at column " +
                                         std::to_string(k));
            }
            if (D[static_cast<std::size_t>(k)] > FloatType{0})
                ++R.positive_eigenvalues;
            Dinv[static_cast<std::size_t>(k)] =
                FloatType{1} / D[static_cast<std::size_t>(k)];
        }
        return R;
    }

    // With ordering: permute input first, then reuse symbolic of permuted A
    [[nodiscard]] static LDLFactorization<FloatType, IntType>
    refactorize_with_ordering(const SparseMatrix<FloatType, IntType> &A,
                              const Symbolic<FloatType, IntType> &S_permuted,
                              const Ordering<IntType> &ord) {
        const auto Aperm = permute_symmetric_upper<FloatType, IntType>(A, ord);
        return refactorize(Aperm, S_permuted);
    }

    // ---------- One-shot ----------
    [[nodiscard]] static LDLFactorization<FloatType, IntType>
    factorize(const SparseMatrix<FloatType, IntType> &A) {
        auto S = analyze(A);
        return refactorize(A, S);
    }

    [[nodiscard]] static LDLFactorization<FloatType, IntType>
    factorize_with_ordering(const SparseMatrix<FloatType, IntType> &A,
                            const Ordering<IntType> &ord) {
        const auto Aperm = permute_symmetric_upper<FloatType, IntType>(A, ord);
        auto S = analyze(Aperm);
        return refactorize(Aperm, S);
    }

    // ---------- Solves ----------
    static void solve_L(const LDLFactorization<FloatType, IntType> &L,
                        std::span<FloatType> x) {
        const IntType n = static_cast<IntType>(x.size());
        const IntType *__restrict Lp = L.L_col_ptr.data();
        const IntType *__restrict Li = L.L_row_idx.data();
        const FloatType *__restrict Lx = L.L_values.data();

        for (IntType i = 0; i < n; ++i) {
            const FloatType xi = x[static_cast<std::size_t>(i)];
            const IntType j0 = Lp[static_cast<std::size_t>(i)];
            const IntType j1 = Lp[static_cast<std::size_t>(i + 1)];
            for (IntType j = j0; j < j1; ++j) {
                x[static_cast<std::size_t>(Li[static_cast<std::size_t>(j)])] -=
                    Lx[static_cast<std::size_t>(j)] * xi;
            }
        }
    }

    static void solve_Lt(const LDLFactorization<FloatType, IntType> &L,
                         std::span<FloatType> x) {
        const IntType n = static_cast<IntType>(x.size());
        const IntType *__restrict Lp = L.L_col_ptr.data();
        const IntType *__restrict Li = L.L_row_idx.data();
        const FloatType *__restrict Lx = L.L_values.data();

        for (IntType i = n - 1; i >= 0; --i) {
            FloatType xi = x[static_cast<std::size_t>(i)];
            const IntType j0 = Lp[static_cast<std::size_t>(i)];
            const IntType j1 = Lp[static_cast<std::size_t>(i + 1)];
            for (IntType j = j0; j < j1; ++j) {
                xi -= Lx[static_cast<std::size_t>(j)] *
                      x[static_cast<std::size_t>(
                          Li[static_cast<std::size_t>(j)])];
            }
            x[static_cast<std::size_t>(i)] = xi;
            if (i == 0)
                break;
        }
    }

    static void solve(const LDLFactorization<FloatType, IntType> &L,
                      std::span<FloatType> x) {
        solve_L(L, x);
        for (std::size_t i = 0; i < L.D_inv.size(); ++i)
            x[i] *= L.D_inv[i];
        solve_Lt(L, x);
    }

    static void
    solve_with_ordering(const LDLFactorization<FloatType, IntType> &L,
                        const Ordering<IntType> &ord, std::span<FloatType> x) {
        const IntType n = static_cast<IntType>(x.size());
        if (ord.n != n)
            throw std::invalid_argument("Ordering size mismatch");

        // Step 1: Apply permutation to RHS: x_perm = P * x
        std::vector<FloatType> x_perm(static_cast<std::size_t>(n));
        for (IntType i = 0; i < n; ++i) {
            x_perm[static_cast<std::size_t>(
                ord.perm[static_cast<std::size_t>(i)])] =
                x[static_cast<std::size_t>(i)];
        }

        // Step 2: Solve the permuted system: (P*A*P^T) * y = x_perm
        solve_L(L, std::span<FloatType>(x_perm));
        for (std::size_t i = 0; i < L.D_inv.size(); ++i) {
            x_perm[i] *= L.D_inv[i];
        }
        solve_Lt(L, std::span<FloatType>(x_perm));

        // Step 3: Apply inverse permutation: x = P^T * y
        for (IntType i = 0; i < n; ++i) {
            x[static_cast<std::size_t>(i)] = x_perm[static_cast<std::size_t>(
                ord.perm[static_cast<std::size_t>(i)])];
        }
    }

    // NEW: Solve refinement with permutation
    static void
    solve_refine_with_ordering(const LDLFactorization<FloatType, IntType> &L,
                               const Ordering<IntType> &ord,
                               const SparseMatrix<FloatType, IntType> &A_orig,
                               std::span<const FloatType> b,
                               std::span<FloatType> x, int iters = 2) {
        const std::size_t n = static_cast<std::size_t>(x.size());
        if (b.size() != n)
            throw std::invalid_argument("b/x size mismatch");

        std::vector<FloatType> r(n), t(n), Ax(n);

        for (int it = 0; it < iters; ++it) {
            // Compute residual: r = b - A*x
            std::copy(b.begin(), b.end(), r.begin());

            // Compute A*x (using original unpermuted matrix)
            std::fill(Ax.begin(), Ax.end(), FloatType{0});
            const auto Ap = A_orig.col_ptr();
            const auto Ai = A_orig.row_idx();
            const auto Av = A_orig.values();

            // A*x for upper triangular part
            for (IntType j = 0; j < static_cast<IntType>(n); ++j) {
                const IntType p0 = Ap[static_cast<std::size_t>(j)];
                const IntType p1 = Ap[static_cast<std::size_t>(j + 1)];
                for (IntType p = p0; p < p1; ++p) {
                    const IntType i = Ai[static_cast<std::size_t>(p)];
                    const FloatType v = Av[static_cast<std::size_t>(p)];
                    Ax[static_cast<std::size_t>(i)] +=
                        v * x[static_cast<std::size_t>(j)];
                    if (i != j) { // symmetric part
                        Ax[static_cast<std::size_t>(j)] +=
                            v * x[static_cast<std::size_t>(i)];
                    }
                }
            }

            // r = b - A*x
            for (std::size_t i = 0; i < n; ++i) {
                r[i] -= Ax[i];
            }

            // Solve: A * t = r (with permutation)
            std::copy(r.begin(), r.end(), t.begin());
            solve_with_ordering(L, ord, std::span<FloatType>(t));

            // Update: x += t
            for (std::size_t i = 0; i < n; ++i) {
                x[i] += t[i];
            }
        }
    }

    // Convenience method to determine which solve to use
    static void solve_auto(const LDLFactorization<FloatType, IntType> &L,
                           std::span<FloatType> x,
                           const Ordering<IntType> *ord = nullptr) {
        if (ord == nullptr) {
            solve(L, x);
        } else {
            solve_with_ordering(L, *ord, x);
        }
    }

    // ---------- Iterative refinement ----------
    static void solve_refine(const LDLFactorization<FloatType, IntType> &L,
                             std::span<const FloatType> b,
                             std::span<FloatType> x, int iters = 2) {
        const std::size_t n = static_cast<std::size_t>(x.size());
        if (b.size() != n)
            throw std::invalid_argument("b/x size mismatch");

        std::vector<FloatType> r(n), t(n), y(n);

        for (int it = 0; it < iters; ++it) {
            std::copy(b.begin(), b.end(), r.begin());

            // y = (L+I) x
            std::copy(x.begin(), x.end(), y.begin());
            solve_L(L, std::span<FloatType>(y));

            // y = D y
            for (std::size_t i = 0; i < n; ++i)
                y[i] *= L.D[i];

            // y = (L+I)^T y
            solve_Lt(L, std::span<FloatType>(y));

            // r -= y
            for (std::size_t i = 0; i < n; ++i)
                r[i] -= y[i];

            // t = A^{-1} r
            std::copy(r.begin(), r.end(), t.begin());
            solve(L, std::span<FloatType>(t));

            // x += t
            for (std::size_t i = 0; i < n; ++i)
                x[i] += t[i];
        }
    }
};

// ===================== Aliases =====================
using QDLDLSolverD = QDLDLSolver<double, std::int64_t>;
using QDLDLSolverF = QDLDLSolver<float, std::int32_t>;
using SparseMatrixD = SparseMatrix<double, std::int64_t>;
using SparseMatrixF = SparseMatrix<float, std::int32_t>;

} // namespace qdldl
