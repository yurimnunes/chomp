#pragma once
#include "PolynomialVector.hpp"

#include "Definitions.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "Polynomials.hpp"
#include "TRModel.hpp"

class PolynomialVector;

void swapElements(Eigen::MatrixXd &matrix, int col1, int col2);

template <typename T>
void swapVectorElements(std::vector<std::shared_ptr<T>> &vec, int idx1, int idx2) {
    // Check if the indices are within the bounds of the vector
    if (idx1 < 0 || idx1 >= vec.size() || idx2 < 0 || idx2 >= vec.size()) {
        throw std::out_of_range("Index out of range");
    }

    // Swapping elements using std::swap to handle shared pointers correctly
    std::swap(vec[idx1], vec[idx2]);
}

template <typename T>
std::vector<T> getSubvector(const std::vector<T> &vec, size_t start, size_t end) {
    if (start > end || end > vec.size()) { throw std::out_of_range("Invalid start or end indices"); }

    return std::vector<T>(vec.begin() + start, vec.begin() + end);
}
Eigen::VectorXd shiftPoint(Eigen::VectorXd &x, Eigen::VectorXd &shift_center);

Eigen::VectorXd unshiftPoint(Eigen::VectorXd &x, Eigen::VectorXd &shift_center, Eigen::VectorXd &bl,
                             Eigen::VectorXd &bu);

std::pair<Eigen::VectorXd, bool> evaluateNewFValues(const Funcao &funcs, Eigen::VectorXd &point);

Eigen::VectorXd projectToBounds(const Eigen::VectorXd &x, const Eigen::VectorXd &bl, const Eigen::VectorXd &bu);

std::vector<int> argsort(const Eigen::VectorXd &v);

PolynomialVector nfpBasis(int dimension, double radius);

Eigen::VectorXd active2eigen(const std::vector<bool> &is_eactive);