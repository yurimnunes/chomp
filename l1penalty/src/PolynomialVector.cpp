#include "../include/PolynomialVector.hpp"
#include "../include/Definitions.hpp"
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

PolynomialVector::PolynomialVector() {
    // data.push_back(nullptr); // Insert a default/dummy PolynomialPtr
}
PolynomialVector::PolynomialVector(size_t initialSize) {
    // data.push_back(nullptr); // Insert a default/dummy PolynomialPtr
    for (size_t i = 0; i < initialSize; ++i) {
        data.push_back(nullptr); // Fill the vector with nullptrs
    }
}
void PolynomialVector::push_back(PolynomialPtr poly) { data.push_back(poly); }
void PolynomialVector::clear() {
    data.clear();
    // data.push_back(nullptr); // Reinsert the dummy element
}
void PolynomialVector::add(PolynomialPtr poly) { data.push_back(poly); }

PolynomialPtr PolynomialVector::get(size_t index) {
    if (index < 0 || index >= data.size()) { throw std::out_of_range("Index out of valid range"); }
    return data[index];
}
PolynomialPtr &PolynomialVector::operator[](size_t index) {
    if (index < 0 || index >= data.size()) { throw std::out_of_range("Index out of valid range"); }
    return data[index];
}
PolynomialPtr &PolynomialVector::operator[](int index) {
    if (index < 0 || index >= static_cast<int>(data.size())) {
        static PolynomialPtr nullPtr = nullptr; // Return a null pointer reference for out-of-bounds access
        return nullPtr;
    }
    return data[index];
}
size_t PolynomialVector::size() const {
    // return number of elements in the vector
    return data.size();
}

const PolynomialPtr &PolynomialVector::operator[](int index) const {
    if (index < 0 || index >= static_cast<int>(data.size())) {
        static const PolynomialPtr nullPtr = nullptr; // Return a const null pointer for out-of-bounds access
        return nullPtr;
    }
    return data[index];
}

PolynomialVector::PolynomialVector(int initialSize, PolynomialPtr initialPoly) {
    // data.push_back(nullptr); // Insert a default/dummy PolynomialPtr
    for (int i = 0; i < initialSize; ++i) {
        data.push_back(initialPoly); // Fill with initialPoly
    }
}
void PolynomialVector::swapElements(int idx1, int idx2) {
    if (idx1 < 0 || idx1 >= static_cast<int>(data.size()) || idx2 < 0 || idx2 >= static_cast<int>(data.size())) {
        throw std::out_of_range("Index out of valid range");
    }
    std::swap(data[idx1], data[idx2]);
}
std::vector<PolynomialPtr>::iterator PolynomialVector::begin() { return data.begin(); }
std::vector<PolynomialPtr>::iterator PolynomialVector::end() { return data.end(); }

// define method subvector that return subvector from a to b from the vector
PolynomialVector PolynomialVector::subVector(int start, int end) {
    return PolynomialVector(data.begin() + start, data.begin() + end);
}

PolynomialVector::PolynomialVector(std::vector<std::shared_ptr<Polynomial>>::iterator first,
                                   std::vector<std::shared_ptr<Polynomial>>::iterator last)
    : data(first, last) {}

void PolynomialVector::reserve(size_t capacity) {
    // Reserve memory in the underlying std::vector
    data.reserve(capacity);
}