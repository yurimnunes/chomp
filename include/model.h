// model.h — Precompiled-from-Python AD model (hot path is pure C++)
#pragma once
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/SparseCore>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "ad/ADBindings.h" // GradFn, LagHessFn, compile_*
#include "definitions.h"

// ============================================================================
// Native AD functors (compiled once via Python `ad` module)
struct ADCompiled {
    std::function<double(const dvec &)> val;
    std::function<dvec(const dvec &)> grad;
    std::function<dmat(const dvec &)> hess;
};


// ============================================================================
// Simple LRU cache for evaluation results
// ============================================================================
// Cached results for a given x (keep original structure)
struct EvalEntry {
    dvec x;
    std::size_t hash{0};

    std::optional<double> f;
    std::optional<dvec> g, cI, cE;
    std::optional<dmat> JI, JE, H;
    
    // Add access tracking for LRU
    mutable int access_order{0};
    
    // Move operations for efficiency
    EvalEntry() = default;
    EvalEntry(EvalEntry&& other) noexcept = default;
    EvalEntry& operator=(EvalEntry&& other) noexcept = default;
};

// ============================================================================
// Improved LRU cache with better performance
class EvalCache {
public:
    explicit EvalCache(std::size_t capacity = 16) 
        : capacity_(capacity), current_access_(0) {
        entries_.reserve(capacity);
    }

    // Find existing entry or nullptr
    EvalEntry* find(const Eigen::Ref<const dvec>& x) {
        std::size_t h = hash_vec(x);
        
        // Linear search is faster for small caches due to cache locality
        for (auto& entry : entries_) {
            if (entry.hash == h && entry.x.size() == x.size() && 
                entry.x.isApprox(x, 1e-14)) {
                entry.access_order = ++current_access_;
                return &entry;
            }
        }
        return nullptr;
    }

    // Insert new entry
    EvalEntry& insert(const Eigen::Ref<const dvec>& x) {
        std::size_t h = hash_vec(x);
        
        if (entries_.size() < capacity_) {
            entries_.emplace_back();
            EvalEntry& entry = entries_.back();
            entry.x = x;
            entry.hash = h;
            entry.access_order = ++current_access_;
            return entry;
        }
        
        // Find LRU entry to replace
        auto lru_it = std::min_element(entries_.begin(), entries_.end(),
            [](const EvalEntry& a, const EvalEntry& b) { 
                return a.access_order < b.access_order; 
            });
            
        // Reset the entry
        lru_it->x = x;
        lru_it->hash = h;
        lru_it->access_order = ++current_access_;
        
        // Clear cached values
        lru_it->f.reset();
        lru_it->g.reset();
        lru_it->cI.reset();
        lru_it->cE.reset();
        lru_it->JI.reset();
        lru_it->JE.reset();
        lru_it->H.reset();
        
        return *lru_it;
    }

    // Improved hash function with better distribution
    static std::size_t hash_vec(const Eigen::Ref<const dvec>& v) {
        constexpr std::size_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
        constexpr std::size_t FNV_PRIME = 1099511628211ULL;
        
        std::size_t hash = FNV_OFFSET_BASIS;
        const double* data = v.data();
        const std::size_t size = v.size() * sizeof(double);
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(data);
        
        for (std::size_t i = 0; i < size; ++i) {
            hash ^= bytes[i];
            hash *= FNV_PRIME;
        }
        return hash;
    }

    // Make capacity accessible for adaptive sizing
    std::size_t capacity() const { return capacity_; }
    void set_capacity(std::size_t new_capacity) {
        capacity_ = new_capacity;
        if (entries_.size() > capacity_) {
            // Remove oldest entries
            std::sort(entries_.begin(), entries_.end(),
                [](const EvalEntry& a, const EvalEntry& b) {
                    return a.access_order > b.access_order;
                });
            entries_.resize(capacity_);
        }
        entries_.reserve(capacity_);
    }

private:
    std::size_t capacity_;
    std::vector<EvalEntry> entries_;
    int current_access_;
};


// ============================================================================
// ModelC with LRU multi-point cache
// ============================================================================
class ModelC {
public:
    using Ret = std::unordered_map<std::string,
                                   std::variant<double, dvec, dmat, spmat>>;

    EvalEntry current_vals = EvalEntry();

    std::optional<double> get_f() const { return current_vals.f; }
    std::optional<dvec> get_g() const { return current_vals.g; }
    std::optional<dvec> get_cI() const { return current_vals.cI; }
    std::optional<dvec> get_cE() const { return current_vals.cE; }
    std::optional<dmat> get_JI() const { return current_vals.JI; }
    std::optional<dmat> get_JE() const { return current_vals.JE; }
    std::optional<dvec> get_lb() const { return lb_; }
    std::optional<dvec> get_ub() const { return ub_; }

    std::shared_ptr<GradFn> f_grad_;
    std::shared_ptr<LagHessFn> f_hess_;
    std::vector<std::shared_ptr<GradFn>> cI_compiled_, cE_compiled_;

    int get_mI() const { return mI_; }
    int get_mE() const { return mE_; }
    int get_n() const { return n_; }

    explicit ModelC(nb::object f, std::optional<nb::object> c_ineq,
                    std::optional<nb::object> c_eq, int n,
                    std::optional<nb::object> lb, std::optional<nb::object> ub,
                    bool use_sparse = false,
                    std::optional<nb::object> ad_module = std::nullopt)
        : n_(n), use_sparse_(use_sparse), cache_(16) // keep last 16 points
    {
        f_grad_ = compile_objective_(f);
        f_hess_ = compile_lag_hess_(f, c_ineq, c_eq);

        if (c_ineq) {
            auto ineq_list = nb::cast<std::vector<nb::object>>(*c_ineq);
            for (auto &obj : ineq_list)
                cI_compiled_.push_back(compile_constraint_(obj));
        }
        if (c_eq) {
            auto eq_list = nb::cast<std::vector<nb::object>>(*c_eq);
            for (auto &obj : eq_list)
                cE_compiled_.push_back(compile_constraint_(obj));
        }

        if (lb && !lb->is_none()) {
            lb_ = nb::cast<dvec>(*lb);
            if (lb_.size() != n_)
                throw std::runtime_error("ModelC: lb size mismatch");
        } else {
            lb_ = dvec::Constant(n_, -std::numeric_limits<double>::infinity());
        }
        if (ub && !ub->is_none()) {
            ub_ = nb::cast<dvec>(*ub);
            if (ub_.size() != n_)
                throw std::runtime_error("ModelC: ub size mismatch");
        } else {
            ub_ = dvec::Constant(n_, std::numeric_limits<double>::infinity());
        }

        mI_ = (int)cI_compiled_.size();
        mE_ = (int)cE_compiled_.size();
    }

    // -------------------------------------------------------------------------
    dmat hess(const Eigen::Ref<const dvec> &x,
              const Eigen::Ref<const dvec> &lam,
              const Eigen::Ref<const dvec> &nu) const {
        EvalEntry *e = cache_.find(x);
        if (!e)
            e = &cache_.insert(x);
        if (!e->H)
            e->H = f_hess_->hess_eigen(x, lam, nu);
        return *e->H;
    }

    // -------------------------------------------------------------------------
    void eval_all(
        const Eigen::Ref<const dvec> &x,
        std::optional<std::vector<std::string>> components = std::nullopt) {
        EvalEntry *e = cache_.find(x);
        if (!e)
            e = &cache_.insert(x);

        const std::vector<std::string> want =
            (components && !components->empty())
                ? *components
                : std::vector<std::string>{"f", "g", "cI", "JI", "cE", "JE"};

        // print components
        auto wants = [&](const char *k) {
            return std::find(want.begin(), want.end(), k) != want.end();
        };

        // f and g
        if ((wants("f") || wants("g")) && (!e->f || !e->g)) {
            auto [fv, fg] = f_grad_->value_grad_eigen(x);
            e->f = fv;
            e->g = fg;
            current_vals.f = e->f;
            current_vals.g = e->g;
        }

        // inequalities
        if ((wants("cI") || wants("JI")) && (!e->cI || !e->JI) && mI_ > 0) {
            auto [vals, J] =
                batch_value_grad_from_gradfns_eigen(cI_compiled_, x);
            e->cI = vals;
            e->JI = J;
            current_vals.cI = e->cI;
            current_vals.JI = e->JI;
        }

        // equalities
        if ((wants("cE") || wants("JE")) && (!e->cE || !e->JE) && mE_ > 0) {
            auto [vals, J] =
                batch_value_grad_from_gradfns_eigen(cE_compiled_, x);
            e->cE = vals;
            e->JE = J;
            current_vals.cE = e->cE;
            current_vals.JE = e->JE;
        }
    }

    // -------------------------------------------------------------------------
    double constraint_violation(const Eigen::Ref<const dvec> &x) {
        eval_all(x, std::vector<std::string>{"cI", "cE"});
        dvec cI_v, cE_v;
        if (mI_)
            cI_v = current_vals.cI.value();
        if (mE_)
            cE_v = current_vals.cE.value();

        const double scale =
            std::max<double>(1.0, std::max<double>(n_, mI_ + mE_));
        double theta = 0.0;
        if (mI_)
            theta += (cI_v.array().max(0.0)).sum() / scale;
        if (mE_)
            theta += cE_v.array().abs().sum() / scale;

        return std::isfinite(theta) ? theta
                                    : std::numeric_limits<double>::infinity();
    }

private:
    int n_{0}, mI_{0}, mE_{0};
    bool use_sparse_{false};
    dvec lb_, ub_;

    mutable EvalCache cache_; // <== global per-model multipoint cache

    // Compilation helpers
    std::shared_ptr<LagHessFn>
    compile_lag_hess_(const nb::object &f,
                      const std::optional<nb::object> &c_ineq = nb::none(),
                      const std::optional<nb::object> &c_eq = nb::none()) {
        auto ineq_list = c_ineq ? nb::cast<std::vector<nb::object>>(*c_ineq)
                                : std::vector<nb::object>{};
        auto eq_list = c_eq ? nb::cast<std::vector<nb::object>>(*c_eq)
                            : std::vector<nb::object>{};
        return std::make_shared<LagHessFn>(f, ineq_list, eq_list, (size_t)n_,
                                           true);
    }

    std::shared_ptr<GradFn> compile_objective_(const nb::object f) {
        return std::make_shared<GradFn>(f, (size_t)n_, true);
    }

    std::shared_ptr<GradFn> compile_constraint_(const nb::object &c,
                                                bool vector_input = true) {
        return std::make_shared<GradFn>(c, (size_t)n_, vector_input);
    }
};
