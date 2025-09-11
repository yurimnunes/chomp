#pragma once

#include <vector>
#include <memory>
#include <chrono>
#include <iostream>
#include <cassert>
#include <cmath>
#include <limits>
#include <algorithm>
#include <unordered_set>
#include <numeric>

namespace simplex {

// Forward declarations
class SparseMatrix;
class LUFactorization;

// Type definitions for clarity and performance
using Real = double;
using Index = int32_t;
using LargeIndex = int64_t;

// Constants for constraint and variable types
enum class BoundType : uint8_t {
    LowerBound = 1,
    UpperBound = 2, 
    Range = 3,
    Fixed = 4,
    Free = 5
};

enum class VariableState : uint8_t {
    Basic = 1,
    AtLower = 2,
    AtUpper = 3,
    AtFixed = 4
};

enum class SolverStatus : uint8_t {
    Uninitialized = 1,
    Initialized = 2,
    PrimalFeasible = 3,
    DualFeasible = 4,
    Optimal = 5,
    Unbounded = 6,
    Infeasible = 7
};

enum class Direction : uint8_t {
    Below = 1,
    Above = 2
};

// Performance timing structure
struct Timings {
    Real matvec = 0.0;
    Real ratiotest = 0.0;
    Real scan = 0.0;
    Real ftran = 0.0;
    Real btran = 0.0;
    Real ftran2 = 0.0;
    Real factor = 0.0;
    Real updatefactor = 0.0;
    Real updateiters = 0.0;
    Real extra = 0.0;

    void print() const {
        std::cout << "Performance Timings:\n"
                  << "  Matrix-vector: " << matvec << "s\n"
                  << "  Ratio test: " << ratiotest << "s\n"
                  << "  Scan: " << scan << "s\n"
                  << "  FTRAN: " << ftran << "s\n"
                  << "  BTRAN: " << btran << "s\n"
                  << "  FTRAN2: " << ftran2 << "s\n"
                  << "  Factorization: " << factor << "s\n"
                  << "  Update factor: " << updatefactor << "s\n"
                  << "  Update iterates: " << updateiters << "s\n"
                  << "  Extra: " << extra << "s\n";
    }
};

// High-performance sparse matrix class
class SparseMatrix {
private:
    Index m_rows, m_cols;
    std::vector<LargeIndex> m_colptr;
    std::vector<Index> m_rowval;
    std::vector<Real> m_nzval;

public:
    SparseMatrix(Index rows, Index cols) 
        : m_rows(rows), m_cols(cols), m_colptr(cols + 1, 0) {}
    
    SparseMatrix(Index rows, Index cols, 
                 std::vector<LargeIndex> colptr,
                 std::vector<Index> rowval,
                 std::vector<Real> nzval)
        : m_rows(rows), m_cols(cols), 
          m_colptr(std::move(colptr)), 
          m_rowval(std::move(rowval)), 
          m_nzval(std::move(nzval)) {}

    // Accessors
    Index rows() const { return m_rows; }
    Index cols() const { return m_cols; }
    LargeIndex nnz() const { return m_nzval.size(); }
    
    const std::vector<LargeIndex>& colptr() const { return m_colptr; }
    const std::vector<Index>& rowval() const { return m_rowval; }
    const std::vector<Real>& nzval() const { return m_nzval; }

    // Matrix-vector multiplication: y = A * x
    void multiply(const std::vector<Real>& x, std::vector<Real>& y) const {
        assert((Index)x.size() == m_cols && (Index)y.size() == m_rows);
        std::fill(y.begin(), y.end(), 0.0);
        for (Index j = 0; j < m_cols; ++j) {
            Real xj = x[j];
            if (std::abs(xj) < 1e-15) continue;
            for (LargeIndex k = m_colptr[j]; k < m_colptr[j + 1]; ++k) {
                y[m_rowval[k]] += m_nzval[k] * xj;
            }
        }
    }

    // Transpose matrix-vector: y = A^T * x
    void multiplyTranspose(const std::vector<Real>& x, std::vector<Real>& y) const {
        assert((Index)x.size() == m_rows && (Index)y.size() == m_cols);
        std::fill(y.begin(), y.end(), 0.0);
        for (Index j = 0; j < m_cols; ++j) {
            Real sum = 0.0;
            for (LargeIndex k = m_colptr[j]; k < m_colptr[j + 1]; ++k) {
                sum += m_nzval[k] * x[m_rowval[k]];
            }
            y[j] = sum;
        }
    }

    // Get column as dense vector
    void getColumn(Index col, std::vector<Real>& result) const {
        assert(col < m_cols && (Index)result.size() == m_rows);
        std::fill(result.begin(), result.end(), 0.0);
        for (LargeIndex k = m_colptr[col]; k < m_colptr[col + 1]; ++k) {
            result[m_rowval[k]] = m_nzval[k];
        }
    }
};

// Linear programming problem data
struct LPData {
    std::vector<Real> c;           // Objective coefficients
    std::vector<Real> lower;       // Lower bounds
    std::vector<Real> upper;       // Upper bounds
    std::vector<BoundType> boundClass;  // Bound types
    std::unique_ptr<SparseMatrix> A;    // Constraint matrix (structural)

    LPData(Index nVars, Index nCons) 
        : c(nVars + nCons), lower(nVars + nCons), upper(nVars + nCons),
          boundClass(nVars + nCons) {}

    Index numVars() const { return A ? A->cols() : 0; }
    Index numCons() const { return A ? A->rows() : 0; }
    Index totalVars() const { return (Index)c.size(); }
};

// Modern LU factorization interface
class LUFactorization {
public:
    virtual ~LUFactorization() = default;
    virtual void factorize(const SparseMatrix& matrix) = 0;
    virtual void solve(std::vector<Real>& rhs) = 0;            // B x = rhs
    virtual void solveTranspose(std::vector<Real>& rhs) = 0;   // B^T x = rhs
    virtual void replaceColumn(const std::vector<Real>& column, Index pos) = 0;
};

// Simplified LU implementation (toy; store L multipliers)
class SimpleLUFactorization : public LUFactorization {
private:
    std::vector<std::vector<Real>> U_; // upper triangle + diag
    std::vector<Index> P_;             // pivot order (row permutation)
    Index n_ = 0;

public:
    void factorize(const SparseMatrix& matrix) override {
        n_ = matrix.rows();
        U_.assign(n_, std::vector<Real>(n_, 0.0));
        P_.resize(n_);
        std::iota(P_.begin(), P_.end(), 0);

        // dense copy
        std::vector<Real> col(n_);
        for (Index j = 0; j < n_; ++j) {
            matrix.getColumn(j, col);
            for (Index i = 0; i < n_; ++i) U_[i][j] = col[i];
        }

        // Gaussian elimination with partial pivoting; store L multipliers in lower triangle
        for (Index k = 0; k < n_ - 1; ++k) {
            Index piv = k;
            Real best = std::abs(U_[P_[k]][k]);
            for (Index i = k + 1; i < n_; ++i) {
                Real v = std::abs(U_[P_[i]][k]);
                if (v > best) { best = v; piv = i; }
            }
            std::swap(P_[k], P_[piv]);

            Real diag = U_[P_[k]][k];
            if (std::abs(diag) < 1e-18) continue; // singular-ish, keep going in toy code

            for (Index i = k + 1; i < n_; ++i) {
                Real m = U_[P_[i]][k] / diag;            // multiplier
                U_[P_[i]][k] = m;                         // store L multiplier
                for (Index j = k + 1; j < n_; ++j) {
                    U_[P_[i]][j] -= m * U_[P_[k]][j];     // Schur update (U part)
                }
            }
        }
    }

    // Solve B x = rhs
    void solve(std::vector<Real>& rhs) override {
        // Apply permutation: y = P * rhs (in-place via temp)
        std::vector<Real> y(n_);
        for (Index i = 0; i < n_; ++i) y[i] = rhs[P_[i]];

        // Forward solve L z = y  (L has 1s on diag, multipliers in U_[row][col] for col<row)
        for (Index i = 0; i < n_; ++i) {
            Real sum = y[i];
            for (Index j = 0; j < i; ++j) sum -= U_[P_[i]][j] * y[j];
            y[i] = sum; // z stored back in y
        }

        // Backward solve U x = z
        for (Index i = n_ - 1; i >= 0; --i) {
            Real sum = y[i];
            for (Index j = i + 1; j < n_; ++j) sum -= U_[P_[i]][j] * y[j];
            Real diag = U_[P_[i]][i];
            y[i] = (std::abs(diag) > 1e-18) ? (sum / diag) : 0.0;
            if (i == 0) break; // prevent unsigned wrap
        }

        // Write back to rhs in natural order
        rhs.assign(n_, 0.0);
        for (Index i = 0; i < n_; ++i) rhs[i] = y[i];
    }

    // Solve B^T x = rhs
    void solveTranspose(std::vector<Real>& rhs) override {
        // We need to solve U^T * L^T * x = P * rhs
        std::vector<Real> y(n_);
        for (Index i = 0; i < n_; ++i) y[i] = rhs[P_[i]]; // apply permutation to RHS

        // First: solve U^T w = y
        // U is upper-triangular (on permuted rows). U^T is lower-triangular in natural index order of "i".
        for (Index i = 0; i < n_; ++i) {
            Real sum = y[i];
            for (Index j = 0; j < i; ++j) sum -= U_[P_[j]][i] * y[j]; // note U^T(i,j) = U(j,i)
            Real diag = U_[P_[i]][i];
            y[i] = (std::abs(diag) > 1e-18) ? (sum / diag) : 0.0;
        }

        // Then: solve L^T x = w
        // L has ones on diag, multipliers stored in U_[row][col] for col<row
        for (Index i = n_ - 1; i >= 0; --i) {
            Real sum = y[i];
            for (Index j = i + 1; j < n_; ++j) sum -= U_[P_[j]][i] * y[j]; // L^T(i,j) = L(j,i) = multiplier at (row=j, col=i)
            y[i] = sum;
            if (i == 0) break;
        }

        rhs.assign(n_, 0.0);
        for (Index i = 0; i < n_; ++i) rhs[i] = y[i];
    }

    // Replace column k of the (current) basis by "column" (toy implementation: full refactor in real codes)
    void replaceColumn(const std::vector<Real>& /*column*/, Index /*pos*/) override {
        // NOTE: In production, use Forrest-Tomlin updates.
        // Here we just signal that a refactor is needed; the solver periodically refactors anyway.
        // No-op in toy.
    }
};

// Main dual simplex solver class
class DualSimplexSolver {
private:
    std::unique_ptr<LPData> m_data;
    std::vector<Real> m_c;              // Possibly perturbed objective
    
    Index m_nIter;                      // Iteration counter
    std::vector<Index> m_basicIdx;      // Basic variable indices (row-activity by default)
    std::vector<VariableState> m_variableState;  // Variable states
    SolverStatus m_status;
    
    std::vector<Real> m_x;              // Primal solution (x then row-activity r)
    std::vector<Real> m_d;              // Reduced costs
    std::vector<Real> m_dse;            // Dual steepest-edge weights
    
    Real m_objval;                      // Objective value
    bool m_phase1;                      // Phase I flag
    bool m_didperturb;                  // Perturbation flag
    
    // Tolerances
    Real m_dualTol;
    Real m_primalTol;
    Real m_zeroTol;
    
    std::unique_ptr<LUFactorization> m_factor;
    Timings m_timings;

public:
    DualSimplexSolver(std::unique_ptr<LPData> data) 
        : m_data(std::move(data)), m_nIter(0), m_objval(0.0),
          m_phase1(false), m_didperturb(false),
          m_dualTol(1e-6), m_primalTol(1e-6), m_zeroTol(1e-12),
          m_factor(std::make_unique<SimpleLUFactorization>()) {
        
        Index totalVars = m_data->totalVars();
        Index nCons = m_data->numCons();
        
        m_c = m_data->c;
        m_basicIdx.resize(nCons);
        m_variableState.assign(totalVars, VariableState::AtLower);
        m_x.resize(totalVars, 0.0);
        m_d.resize(totalVars, 0.0);
        m_dse.assign(totalVars, 1.0);
        m_status = SolverStatus::Uninitialized;
        
        // Initialize row-activity variables r_i as basic (indices: nVars ... nVars+nCons-1)
        for (Index i = 0; i < nCons; ++i) {
            m_variableState[m_data->numVars() + i] = VariableState::Basic;
        }
        
        // Set non-basic structural variables at appropriate bounds
        for (Index i = 0; i < m_data->numVars(); ++i) {
            if (m_data->boundClass[i] == BoundType::UpperBound) {
                m_variableState[i] = VariableState::AtUpper;
            }
        }
    }

    void solve() {
        if (m_status == SolverStatus::Uninitialized) {
            initialize(true);
        }
        
        if (m_status == SolverStatus::Initialized) {
            makeFeasible(); // attempt quick dual feasibility
        }

        // If still not dual-feasible, apply a one-shot perturbation to get going (teaching toy)
        if (m_status != SolverStatus::DualFeasible && m_status != SolverStatus::Optimal) {
            perturbForFeasibility();
            calculateDualSolution();
            checkFeasibility();
        }
        
        // Main iteration loop (only iterate when dual-feasible)
        for (Index iter = 0; iter < 200000; ++iter) {
            if (m_status == SolverStatus::Optimal) break;
            if (m_status != SolverStatus::DualFeasible) break; // don’t assert; exit gracefully

            iterate();
            
            // Periodic refactorization hook (our replaceColumn is a no-op)
            if (m_nIter > 0 && (m_nIter % 50 == 0)) {
                initialize(true);
                calculateDualSolution();
                checkFeasibility();
            }
            
            ++m_nIter;
        }
        
        // Clean up perturbations
        m_c = m_data->c;
        initialize(true);
        calculateDualSolution();
        checkFeasibility();
        
        if (m_status != SolverStatus::Optimal) {
            std::cout << "Note: Solver stopped without certified optimality (toy implementation)\n";
        }
        
        if (!m_phase1) {
            m_timings.print();
        }
    }

    // Getters for solution
    Real getObjectiveValue() const { return m_objval; }
    const std::vector<Real>& getSolution() const { return m_x; }
    SolverStatus getStatus() const { return m_status; }
    const Timings& getTimings() const { return m_timings; }

private:
    void initialize(bool reinvert) {
        auto start = std::chrono::high_resolution_clock::now();
        
        Index nCons = m_data->numCons();
        Index nVars = m_data->numVars();
        
        if (reinvert) {
            // Find basic variables
            m_basicIdx.clear();
            for (Index i = 0; i < m_data->totalVars(); ++i) {
                if (m_variableState[i] == VariableState::Basic) {
                    m_basicIdx.push_back(i);
                }
            }
            if ((Index)m_basicIdx.size() != nCons) {
                // Fallback: set the last nCons as basic row-activity vars
                m_basicIdx.resize(nCons);
                for (Index i = 0; i < nCons; ++i) m_basicIdx[i] = nVars + i;
            }
            
            // Basis matrix: Identity (+1.0) for row-activity variables
            std::vector<LargeIndex> colptr(nCons + 1);
            std::vector<Index> rowval; rowval.reserve(nCons);
            std::vector<Real> nzval;  nzval.reserve(nCons);
            for (Index j = 0; j < nCons; ++j) {
                colptr[j] = j;
                rowval.push_back(j);
                nzval.push_back(+1.0); // IMPORTANT: +1 (r − A x = 0)
            }
            colptr[nCons] = nCons;
            
            auto basisMatrix = std::make_unique<SparseMatrix>(
                nCons, nCons, std::move(colptr), std::move(rowval), std::move(nzval));
            
            m_factor->factorize(*basisMatrix);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        m_timings.factor += std::chrono::duration<Real>(end - start).count();
        
        // Initialize primal variables at bounds
        for (Index i = 0; i < m_data->totalVars(); ++i) {
            if (m_variableState[i] == VariableState::Basic || 
                m_data->boundClass[i] == BoundType::Free) {
                m_x[i] = 0.0; // basic r will be set by calculateBasicSolution()
            } else if (m_variableState[i] == VariableState::AtLower) {
                m_x[i] = m_data->lower[i];
            } else {
                m_x[i] = m_data->upper[i];
            }
        }
        
        // Calculate basic solution r = A * x_N  (since constraints are r − A x = 0)
        calculateBasicSolution();

        // Duals and reduced costs
        calculateDualSolution();
        m_objval = std::inner_product(m_x.begin(), m_x.end(), m_c.begin(), 0.0);
        checkFeasibility();
    }

    void calculateBasicSolution() {
        Index nCons = m_data->numCons();
        Index nVars = m_data->numVars();
        
        // rhs = A * x_N (only structural, non-basic contributions)
        std::vector<Real> xN(nVars, 0.0);
        for (Index j = 0; j < nVars; ++j) {
            if (m_variableState[j] != VariableState::Basic) xN[j] = m_x[j];
        }
        std::vector<Real> rhs(nCons, 0.0);
        m_data->A->multiply(xN, rhs); // rhs = A x_N

        // Solve B * x_B = rhs, but B is identity, still call to keep interface
        m_factor->solve(rhs);

        // Set basic (row-activity) variable values: r = rhs
        for (Index i = 0; i < nCons; ++i) {
            m_x[m_basicIdx[i]] = rhs[i];
        }
    }

    void calculateDualSolution() {
        Index nCons = m_data->numCons();
        Index nVars = m_data->numVars();
        
        // y = (B^T)^{-1} * c_B
        std::vector<Real> y(nCons);
        for (Index i = 0; i < nCons; ++i) y[i] = m_c[m_basicIdx[i]];
        m_factor->solveTranspose(y);
        
        // Reduced costs for structural variables j: d_j = c_j − a_j^T y
        for (Index j = 0; j < nVars; ++j) {
            if (m_variableState[j] == VariableState::Basic) { m_d[j] = 0.0; continue; }
            Real val = 0.0;
            for (LargeIndex k = m_data->A->colptr()[j]; k < m_data->A->colptr()[j + 1]; ++k) {
                val += y[m_data->A->rowval()[k]] * m_data->A->nzval()[k];
            }
            m_d[j] = m_c[j] - val;
        }
        
        // Reduced costs for row-activity variables (columns of +I): d_idx = c_idx − e_i^T y = c_idx − y_i
        for (Index i = 0; i < nCons; ++i) {
            Index idx = i + nVars;
            if (m_variableState[idx] == VariableState::Basic) m_d[idx] = 0.0;
            else m_d[idx] = m_c[idx] - y[i];
        }
    }

    void checkFeasibility() {
        Real dualInfeas = 0.0, primalInfeas = 0.0;
        Index nDualInfeas = 0, nPrimalInfeas = 0;
        
        // Check dual feasibility on non-basic variables
        for (Index i = 0; i < m_data->totalVars(); ++i) {
            if (m_variableState[i] == VariableState::Basic) continue;
            bool infeas = false;
            if (m_data->boundClass[i] == BoundType::Free && std::abs(m_d[i]) > m_dualTol) {
                infeas = true;
            } else if (m_variableState[i] == VariableState::AtLower && 
                       m_d[i] < -m_dualTol &&
                       m_data->boundClass[i] != BoundType::Fixed) {
                infeas = true;
            } else if (m_variableState[i] == VariableState::AtUpper && 
                       m_d[i] > m_dualTol &&
                       m_data->boundClass[i] != BoundType::Fixed) {
                infeas = true;
            }
            if (infeas) { dualInfeas += std::abs(m_d[i]); ++nDualInfeas; }
        }
        
        // Check primal feasibility of basics against their bounds
        for (Index k = 0; k < m_data->numCons(); ++k) {
            Index bidx = m_basicIdx[k];
            if (m_x[bidx] < m_data->lower[bidx] - m_primalTol) {
                primalInfeas += (m_data->lower[bidx] - m_x[bidx]);
                ++nPrimalInfeas;
            } else if (m_x[bidx] > m_data->upper[bidx] + m_primalTol) {
                primalInfeas += (m_x[bidx] - m_data->upper[bidx]);
                ++nPrimalInfeas;
            }
        }
        
        if (dualInfeas > 0) {
            m_status = (primalInfeas > 0) ? SolverStatus::Initialized : SolverStatus::PrimalFeasible;
        } else {
            m_status = (primalInfeas > 0) ? SolverStatus::DualFeasible : SolverStatus::Optimal;
        }

        m_objval = std::inner_product(m_x.begin(), m_x.end(), m_c.begin(), 0.0);

        std::cout << "Iteration " << m_nIter << " Obj: " << m_objval;
        if (primalInfeas > 0) std::cout << " Primal inf " << primalInfeas << " (" << nPrimalInfeas << ")";
        if (dualInfeas > 0)   std::cout << " Dual inf "   << dualInfeas   << " (" << nDualInfeas   << ")";
        if (m_phase1) std::cout << " (Phase I)";
        std::cout << "\n";
    }

    Index selectLeavingVariable() {
        auto start = std::chrono::high_resolution_clock::now();
        Real maxRatio = 0.0; Index maxIdx = -1;
        for (Index k = 0; k < m_data->numCons(); ++k) {
            Index i = m_basicIdx[k];
            Real r = m_data->lower[i] - m_x[i];
            if (r > m_primalTol) {
                Real ratio = r * r / std::max(m_dse[i], 1e-8);
                if (ratio > maxRatio) { maxRatio = ratio; maxIdx = k; }
            }
            r = m_x[i] - m_data->upper[i];
            if (r > m_primalTol) {
                Real ratio = r * r / std::max(m_dse[i], 1e-8);
                if (ratio > maxRatio) { maxRatio = ratio; maxIdx = k; }
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        m_timings.scan += std::chrono::duration<Real>(end - start).count();
        return maxIdx;
    }

    Index selectEnteringVariable(const std::vector<Real>& alpha) {
        auto start = std::chrono::high_resolution_clock::now();
        const Real pivotTol = 1e-9;
        Real thetaMax = 1e25;
        Index enter = -1;
        Real maxAlpha = 0.0;
        std::vector<Index> candidates;

        // Pass 1: compute minimal ratio
        for (Index i = 0; i < m_data->totalVars(); ++i) {
            if (m_variableState[i] == VariableState::Basic || m_data->boundClass[i] == BoundType::Fixed) continue;
            bool ok = false; Real ratio = 0.0;
            if (m_variableState[i] == VariableState::AtLower && alpha[i] > pivotTol) {
                ok = true; ratio = (m_d[i] + m_dualTol) / alpha[i];
            } else if (m_variableState[i] == VariableState::AtUpper && alpha[i] < -pivotTol) {
                ok = true; ratio = (m_d[i] - m_dualTol) / alpha[i];
            } else if (m_data->boundClass[i] == BoundType::Free && std::abs(alpha[i]) > pivotTol) {
                ok = true; ratio = (alpha[i] > 0) ? (m_d[i] + m_dualTol)/alpha[i] : (m_d[i]-m_dualTol)/alpha[i];
            }
            if (ok) { candidates.push_back(i); thetaMax = std::min(thetaMax, ratio); }
        }

        // Pass 2: select largest |alpha| among eligible
        for (Index i : candidates) {
            Real ratio = m_d[i] / alpha[i];
            if (ratio <= thetaMax && std::abs(alpha[i]) > maxAlpha) {
                maxAlpha = std::abs(alpha[i]); enter = i;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        m_timings.ratiotest += std::chrono::duration<Real>(end - start).count();
        return enter;
    }

    void iterate() {
        if (m_status != SolverStatus::DualFeasible) return;

        Index leave = selectLeavingVariable();
        if (leave == -1) { m_status = SolverStatus::Optimal; return; }
        
        Index leaveIdx = m_basicIdx[leave];
        Direction leaveType;
        Real delta;
        if (m_x[leaveIdx] > m_data->upper[leaveIdx]) {
            leaveType = Direction::Above;
            delta = m_x[leaveIdx] - m_data->upper[leaveIdx];
        } else {
            leaveType = Direction::Below;
            delta = m_x[leaveIdx] - m_data->lower[leaveIdx];
        }
        
        // Compute rho = B^{-T} e_leave
        std::vector<Real> rho(m_data->numCons(), 0.0);
        rho[leave] = 1.0;
        auto start = std::chrono::high_resolution_clock::now();
        m_factor->solveTranspose(rho);
        auto end = std::chrono::high_resolution_clock::now();
        m_timings.btran += std::chrono::duration<Real>(end - start).count();
        
        // Compute tableau row α = [ −A^T  |  +I ]^T * rho  BUT our reduced-cost formula uses columns:
        // For structural j: α_j = (A column j)^T rho; for row-activity i: α_{nVars+i} = +rho[i].
        std::vector<Real> alpha(m_data->totalVars(), 0.0);
        start = std::chrono::high_resolution_clock::now();
        calculatePivotRow(rho, alpha);
        end = std::chrono::high_resolution_clock::now();
        m_timings.matvec += std::chrono::duration<Real>(end - start).count();

        if (leaveType == Direction::Below) for (Real& a : alpha) a = -a;

        Index enterIdx = selectEnteringVariable(alpha);
        if (enterIdx == -1) { m_status = SolverStatus::Unbounded; return; }
        
        performPivot(leave, enterIdx, leaveType, delta, rho, alpha);
    }

    void calculatePivotRow(const std::vector<Real>& rho, std::vector<Real>& alpha) {
        Index nVars = m_data->numVars();
        Index nCons = m_data->numCons();
        
        // Price structural variables
        for (Index j = 0; j < nVars; ++j) {
            if (m_variableState[j] == VariableState::Basic) continue;
            Real val = 0.0;
            for (LargeIndex k = m_data->A->colptr()[j]; k < m_data->A->colptr()[j + 1]; ++k) {
                val += rho[m_data->A->rowval()[k]] * m_data->A->nzval()[k];
            }
            alpha[j] = val;
        }
        // Price row-activity (identity) variables: +rho[i]
        for (Index i = 0; i < nCons; ++i) {
            Index idx = i + nVars;
            if (m_variableState[idx] == VariableState::Basic) continue;
            alpha[idx] = +rho[i]; // IMPORTANT: + sign
        }
    }

    void performPivot(Index leave, Index enterIdx, Direction leaveType, Real delta,
                      const std::vector<Real>& rho, std::vector<Real>& alpha) {

        Index leaveIdx = m_basicIdx[leave];

        // Dual step length
        Real thetad;
        if (m_d[enterIdx] / alpha[enterIdx] < 0.0) {
            thetad = (leaveType == Direction::Below ? 1.0 : -1.0) * 1e-12;
            Real diff = thetad * alpha[enterIdx] - m_d[enterIdx];
            m_d[enterIdx] = thetad * alpha[enterIdx];
            m_c[enterIdx] += diff;
        } else {
            thetad = (leaveType == Direction::Below ? 1.0 : -1.0) * m_d[enterIdx] / alpha[enterIdx];
        }

        if (leaveType == Direction::Below) for (Real& a : alpha) a = -a;

        // Update reduced costs by the tableau row
        auto start = std::chrono::high_resolution_clock::now();
        updateDualVariables(alpha, leaveIdx, enterIdx, thetad);
        auto end = std::chrono::high_resolution_clock::now();
        m_timings.updateiters += std::chrono::duration<Real>(end - start).count();

        // Entering column (structural or row-activity)
        std::vector<Real> enteringColumn(m_data->numCons(), 0.0);
        start = std::chrono::high_resolution_clock::now();
        if (enterIdx < m_data->numVars()) {
            m_data->A->getColumn(enterIdx, enteringColumn);
            // Column in constraints is −A (since r − A x = 0) but B^{-1} multiplies the actual column in B,
            // and we pivot in the tableau sense; sign is handled consistently by alpha/thetad
        } else {
            std::fill(enteringColumn.begin(), enteringColumn.end(), 0.0);
            enteringColumn[enterIdx - m_data->numVars()] = +1.0; // identity column
        }
        end = std::chrono::high_resolution_clock::now();
        m_timings.extra += std::chrono::duration<Real>(end - start).count();

        // Solve for the pivot column: p = B^{-1} * col(enter)
        start = std::chrono::high_resolution_clock::now();
        m_factor->solve(enteringColumn);
        end = std::chrono::high_resolution_clock::now();
        m_timings.ftran += std::chrono::duration<Real>(end - start).count();

        Real thetap = delta / enteringColumn[leave];

        // Update primal (basic) variables and the entering variable
        start = std::chrono::high_resolution_clock::now();
        updatePrimalVariables(enteringColumn, enterIdx, leave, thetap);
        end = std::chrono::high_resolution_clock::now();
        m_timings.updateiters += std::chrono::duration<Real>(end - start).count();

        // DSE update (toy)
        updateDSE(rho, enteringColumn, enterIdx, enteringColumn[leave]);

        // NOTE: Our replaceColumn is a no-op; we rely on periodic refactorization.
        // m_factor->replaceColumn(enteringColumn, leave);

        // Update basis and states
        m_basicIdx[leave] = enterIdx;
        m_variableState[enterIdx] = VariableState::Basic;

        if (leaveType == Direction::Below) {
            m_x[leaveIdx] = m_data->lower[leaveIdx];
            m_variableState[leaveIdx] = VariableState::AtLower;
        } else {
            m_x[leaveIdx] = m_data->upper[leaveIdx];
            m_variableState[leaveIdx] = VariableState::AtUpper;
        }

        // Recompute reduced costs/objective and status snapshot
        calculateDualSolution();
        m_objval = std::inner_product(m_x.begin(), m_x.end(), m_c.begin(), 0.0);
        checkFeasibility();
    }

    void updateDualVariables(const std::vector<Real>& tableauRow, Index leaveIdx, 
                             Index enterIdx, Real thetad) {
        m_d[leaveIdx] = -thetad;
        m_d[enterIdx] = 0.0;
        for (Index i = 0; i < m_data->totalVars(); ++i) {
            if (m_variableState[i] == VariableState::Basic || i == enterIdx) continue;
            Real dnew = m_d[i] - thetad * tableauRow[i];

            if (m_data->boundClass[i] == BoundType::Fixed) { m_d[i] = dnew; continue; }

            // Cost shifting to maintain dual feasibility (toy but robust)
            if (m_variableState[i] == VariableState::AtLower || m_data->boundClass[i] == BoundType::Free) {
                if (dnew >= -m_dualTol) m_d[i] = dnew;
                else { Real delta = -dnew - m_dualTol; m_c[i] += delta; m_d[i] = -m_dualTol; }
            }
            if (m_variableState[i] == VariableState::AtUpper || m_data->boundClass[i] == BoundType::Free) {
                if (dnew <= m_dualTol) m_d[i] = dnew;
                else { Real delta = -dnew + m_dualTol; m_c[i] += delta; m_d[i] =  m_dualTol; }
            }
        }
    }

    void updatePrimalVariables(const std::vector<Real>& tableauColumn, Index enterIdx,
                               Index leave, Real thetap) {
        for (Index i = 0; i < m_data->numCons(); ++i) {
            Index idx = m_basicIdx[i];
            m_x[idx] -= thetap * tableauColumn[i];
        }
        m_x[enterIdx] += thetap;
    }

    void updateDSE(const std::vector<Real>& rho, const std::vector<Real>& tableauColumn,
                   Index enterIdx, Real pivot) {
        auto start = std::chrono::high_resolution_clock::now();
        Real dseEnter = std::inner_product(rho.begin(), rho.end(), rho.begin(), 0.0) / (pivot * pivot);

        std::vector<Real> tau = rho; // tau = B^{-1} rho
        m_factor->solve(tau);

        auto end = std::chrono::high_resolution_clock::now();
        m_timings.ftran2 += std::chrono::duration<Real>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        Real kappa = -2.0 / pivot;
        for (Index i = 0; i < m_data->numCons(); ++i) {
            Index idx = m_basicIdx[i];
            Real col = tableauColumn[i];
            if (std::abs(col) < m_zeroTol) continue;
            m_dse[idx] += col * (col * dseEnter + kappa * tau[i]);
            m_dse[idx] = std::max(m_dse[idx], 1e-4);
        }
        m_dse[enterIdx] = dseEnter;
        end = std::chrono::high_resolution_clock::now();
        m_timings.updateiters += std::chrono::duration<Real>(end - start).count();
    }

    void perturbForFeasibility() {
        std::cout << "Applying small cost perturbations to reach dual feasibility...\n";
        m_didperturb = true;
        for (Index i = 0; i < m_data->totalVars(); ++i) {
            if (m_variableState[i] == VariableState::Basic) continue;
            if ((m_variableState[i] == VariableState::AtLower || m_data->boundClass[i] == BoundType::Free) && m_d[i] < -m_dualTol) {
                Real delta = -m_d[i] - m_dualTol + 1e-8; m_c[i] += delta; m_d[i] = -m_dualTol + 1e-8;
            }
            if ((m_variableState[i] == VariableState::AtUpper || m_data->boundClass[i] == BoundType::Free) && m_d[i] >  m_dualTol) {
                Real delta = -m_d[i] +  m_dualTol - 1e-8; m_c[i] += delta; m_d[i] =  m_dualTol - 1e-8;
            }
        }
        m_status = SolverStatus::DualFeasible; // allow iterations in toy solver
    }

    void makeFeasible() {
        if (m_status == SolverStatus::DualFeasible) return;
        initialize(false);
        flipBounds(); // quick attempt
        if (m_status == SolverStatus::DualFeasible) return;
        std::cout << "Warning: Phase I not implemented; will use perturbation trick.\n";
    }

    void flipBounds() {
        bool didFlip = false;
        for (Index i = 0; i < m_data->totalVars(); ++i) {
            if (m_variableState[i] == VariableState::Basic) continue;
            bool infeas = false;
            if (m_data->boundClass[i] == BoundType::Free && std::abs(m_d[i]) > m_dualTol) infeas = true;
            else if (m_variableState[i] == VariableState::AtLower && m_d[i] < -m_dualTol && m_data->boundClass[i] != BoundType::Fixed) infeas = true;
            else if (m_variableState[i] == VariableState::AtUpper && m_d[i] >  m_dualTol && m_data->boundClass[i] != BoundType::Fixed) infeas = true;
            if (infeas && (m_data->boundClass[i] == BoundType::Range || m_data->boundClass[i] == BoundType::Fixed)) {
                didFlip = true;
                if (m_variableState[i] == VariableState::AtLower) m_variableState[i] = VariableState::AtUpper;
                else if (m_variableState[i] == VariableState::AtUpper) m_variableState[i] = VariableState::AtLower;
            }
        }
        if (didFlip) { initialize(false); calculateDualSolution(); checkFeasibility(); }
    }
};

// Factory function to create LP data from arrays
std::unique_ptr<LPData> createLPData(
    const std::vector<Real>& c,
    const std::vector<Real>& xlb, const std::vector<Real>& xub,
    const std::vector<Real>& l, const std::vector<Real>& u,
    std::unique_ptr<SparseMatrix> A) {
    
    Index nVars = (Index)c.size();
    Index nCons = A->rows();
    
    auto data = std::make_unique<LPData>(nVars, nCons);
    
    auto checkBoundType = [](Real lower, Real upper) -> BoundType {
        const Real inf = std::numeric_limits<Real>::infinity();
        if (lower > -inf) {
            if (upper < inf) return (std::abs(lower - upper) < 1e-7) ? BoundType::Fixed : BoundType::Range;
            else return BoundType::LowerBound;
        } else {
            if (upper < inf) return BoundType::UpperBound;
            else return BoundType::Free;
        }
    };
    
    // Structural variables
    for (Index i = 0; i < nVars; ++i) {
        data->c[i] = c[i];
        data->lower[i] = xlb[i];
        data->upper[i] = xub[i];
        data->boundClass[i] = checkBoundType(xlb[i], xub[i]);
    }
    // Row-activity variables r with bounds l ≤ r ≤ u and zero cost
    for (Index i = 0; i < nCons; ++i) {
        data->c[i + nVars] = 0.0;
        data->lower[i + nVars] = l[i];
        data->upper[i + nVars] = u[i];
        data->boundClass[i + nVars] = checkBoundType(l[i], u[i]);
    }
    data->A = std::move(A);
    return data;
}

// Example usage and test function
void demonstrateSolver() {
    // Example: minimize c^T x subject to x1 + x2 ≤ 2, x ≥ 0
    // Model as ranged row via row-activity r: r − [1 1] x = 0,  −∞ ≤ r ≤ 2
    std::vector<Real> c   = { -1.0, -1.0 };
    std::vector<Real> xlb = {  0.0,  0.0 };
    std::vector<Real> xub = {  std::numeric_limits<Real>::infinity(),
                               std::numeric_limits<Real>::infinity() };
    std::vector<Real> l   = { -std::numeric_limits<Real>::infinity() }; // lower on r
    std::vector<Real> u   = {  2.0 };                                   // upper on r

    // A = [1 1]
    std::vector<LargeIndex> colptr = {0, 1, 2};
    std::vector<Index>      rowval = {0, 0};
    std::vector<Real>       nzval  = {1.0, 1.0};
    auto A = std::make_unique<SparseMatrix>(1, 2, std::move(colptr), std::move(rowval), std::move(nzval));

    auto lpData = createLPData(c, xlb, xub, l, u, std::move(A));
    DualSimplexSolver solver(std::move(lpData));
    solver.solve();

    std::cout << "\nSolution:\n";
    std::cout << "Status: " << static_cast<int>(solver.getStatus()) << "\n";
    std::cout << "Objective: " << solver.getObjectiveValue() << "\n";
    const auto& solution = solver.getSolution();
    for (Index i = 0; i < 2; ++i) {
        std::cout << "x[" << i << "] = " << solution[i] << "\n";
    }
    // Row-activity variable (r) is at index 2 if you want to inspect it.
}

} // namespace simplex
