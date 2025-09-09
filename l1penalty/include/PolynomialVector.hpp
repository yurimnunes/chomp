#pragma once

#include "Definitions.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>
// include iota library
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <numeric>

class PolynomialVector {
private:
    std::vector<PolynomialPtr> data;

public:
    // Ensure there's always a dummy entry at index 0
    PolynomialVector();
    PolynomialVector(size_t initialSize);
    PolynomialVector(std::vector<std::shared_ptr<Polynomial>>::iterator first,
                     std::vector<std::shared_ptr<Polynomial>>::iterator last);
    void push_back(PolynomialPtr poly);

    PolynomialVector(int initialSize, PolynomialPtr initialPoly);

    // Clear the vector but keep the dummy element
    void clear();
    void swapElements(int idx1, int idx2);

    // Add a new polynomial
    void add(PolynomialPtr poly);

    // Access polynomial (with boundary check) using the get method
    PolynomialPtr get(size_t index);

    // Access polynomial using subscript operator
    PolynomialPtr &operator[](size_t index);

    PolynomialPtr &operator[](int index);

    // Access polynomial using const subscript operator
    const PolynomialPtr &operator[](int index) const;

    // Get size (excluding the dummy entry)
    size_t size() const;

    std::vector<PolynomialPtr>::iterator begin();

    std::vector<PolynomialPtr>::iterator end();

    PolynomialVector subVector(int start, int end);
    
    void reserve(size_t capacity);

};