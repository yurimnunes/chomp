#pragma once

#include "Definitions.h"
#include <atomic>
#include <limits>
#include <memory>
#include <string>
#include <vector>

class Expression;
class Variable;

using VariablePtr   = std::shared_ptr<Variable>;
using ExpressionPtr = std::shared_ptr<Expression>;

class Variable : public std::enable_shared_from_this<Variable> {
public:
    explicit Variable(const std::string& name,
                      double value = 0.0,
                      double lb    = -std::numeric_limits<double>::infinity(),
                      double ub    =  std::numeric_limits<double>::infinity())
        : std::enable_shared_from_this<Variable>()              // (ok)
        , name_(name), value_(value), lb_(lb), ub_(ub) {
        if (lb_ > ub_) std::swap(lb_, ub_);
        if (value_ < lb_) value_ = lb_;
        if (value_ > ub_) value_ = ub_;
        order_ = nextOrder_.fetch_add(1, std::memory_order_relaxed);
    }

    // Copy keeps the same metadata; does NOT advance global order
    Variable(const Variable& other)
        : std::enable_shared_from_this<Variable>(other)
        , name_(other.name_)
        , value_(other.value_)
        , lb_(other.lb_)
        , ub_(other.ub_)
        , gradient_(other.gradient_)  // legacy field; not authoritative
        , parents_()                  // do not copy parent links
        , order_(other.order_) {}

    Variable& operator=(const Variable& other) {
        if (this == &other) return *this;
        // NOTE: base part (enable_shared_from_this) cannot be reassigned; this is fine.
        name_     = other.name_;
        value_    = other.value_;
        lb_       = other.lb_;
        ub_       = other.ub_;
        gradient_ = other.gradient_; // legacy
        order_    = other.order_;
        parents_.clear(); // do not copy parent links
        return *this;
    }

    // ---- Identity / metadata ----
    const std::string& getName() const { return name_; }
    void setName(const std::string& newName) { name_ = newName; }

    int  getOrder() const { return order_; }
    static int  getCurrentOrder() { return nextOrder_.load(std::memory_order_relaxed); }
    static void resetOrder(int start = 0) { nextOrder_.store(start, std::memory_order_relaxed); }

    // ---- Value & bounds ----
    double getValue() const { return value_; }
    void   setValue(double newValue) {
        value_ = newValue;
        if (value_ < lb_) value_ = lb_;
        if (value_ > ub_) value_ = ub_;
    }

    double getLowerBound() const { return lb_; }
    void   setLowerBound(double newLB) {
        lb_ = newLB;
        if (lb_ > ub_) std::swap(lb_, ub_);
        if (value_ < lb_) value_ = lb_;
    }

    double getUpperBound() const { return ub_; }
    void   setUpperBound(double newUB) {
        ub_ = newUB;
        if (lb_ > ub_) std::swap(lb_, ub_);
        if (value_ > ub_) value_ = ub_;
    }

    // ---- Gradient (legacy) ----
    double getGradient() const { return gradient_; }
    void   setGradient(double g) { gradient_ = g; }

    // ---- Expression interop ----
    operator ExpressionPtr() const;
    ExpressionPtr operator*(const Variable& other) const;
    ExpressionPtr operator*(double lhs);

    // Parent tracking (weak to avoid cycles)
    void addParent(const ExpressionPtr& parent);
    std::vector<ExpressionPtr> getParents() const;

private:
    std::string name_;
    double      value_;
    double lb_ = -std::numeric_limits<double>::infinity();
    double ub_ =  std::numeric_limits<double>::infinity();

    double gradient_ = 0.0;
    std::vector<std::weak_ptr<Expression>> parents_;

    int  order_ = -1;
    static std::atomic<int> nextOrder_;
};
