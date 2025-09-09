#pragma once

#ifdef __cplusplus

#include "Definitions.hpp"
#include "PolynomialVector.hpp"
#include "Polynomials.hpp"
#include "TRModel.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <fmt/core.h>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <vector>

std::tuple<Eigen::VectorXd, double> l1_penalty_solve(Funcao &func, Eigen::MatrixXd &initial_points, double mu, double epsilon, double delta,
                        double lambda, Eigen::VectorXd &bl, Eigen::VectorXd &bu, Eigen::VectorXd &con_bl,
                        Eigen::VectorXd &con_bu, Options &options);

#endif
