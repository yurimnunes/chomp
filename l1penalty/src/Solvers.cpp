#include "../include/Solvers.hpp"
#include "gurobi_c++.h"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <limits>
#include <tuple>
#include <vector>

// Definition
GRBEnv GurobiSolver::env;

void GurobiSolver::initializeEnvironment() {
    static bool initialized = false;
    if (!initialized) {
        try {
            GRBEnv env = GRBEnv();
            env.start();
            initialized = true;
        } catch (GRBException &e) {
            std::cerr << "Error code = " << e.getErrorCode() << std::endl;
            std::cerr << e.getMessage() << std::endl;
            throw;
        }
    }
}