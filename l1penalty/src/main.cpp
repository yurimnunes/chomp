#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <vector>
#include "../include/Definitions.hpp"
#include "../include/PolynomialVector.hpp"
#include "../include/l1Solver.hpp"

// Branin function - a classic test function for global optimization
// Global minima at: (-π, 12.275), (π, 2.275), (9.42478, 2.475) with f* = 0.397887
double branin(const Eigen::VectorXd &x) {
    if (x.size() != 2) {
        throw std::invalid_argument("Branin function requires exactly 2 variables");
    }
    
    double x1 = x(0);
    double x2 = x(1);
    
    // Branin function parameters
    double a = 1.0;
    double b = 5.1 / (4.0 * M_PI * M_PI);
    double c = 5.0 / M_PI;
    double r = 6.0;
    double s = 10.0;
    double t = 1.0 / (8.0 * M_PI);
    
    double term1 = a * std::pow(x2 - b * x1 * x1 + c * x1 - r, 2);
    double term2 = s * (1.0 - t) * std::cos(x1);
    double term3 = s;
    
    return term1 + term2 + term3;
}

// Simple bound constraint to keep variables reasonable
double bound_constraint(const Eigen::VectorXd &x) {
    // Keep the search within a reasonable ellipse
    // This helps the local optimizer not get stuck in bad regions
    double x1_scaled = x(0) / 10.0;  // Scale x1 from [-5,10] to roughly [-0.5, 1]
    double x2_scaled = x(1) / 15.0;  // Scale x2 from [0,15] to roughly [0, 1]
    return x1_scaled * x1_scaled + x2_scaled * x2_scaled;
}

int main() {
    std::cout << "Optimizing the Branin function with L1 penalty method" << std::endl;
    std::cout << "======================================================" << std::endl;
    std::cout << "Known global minima:" << std::endl;
    std::cout << "  (-π, 12.275) ≈ (-3.14159, 12.275) with f* = 0.397887" << std::endl;
    std::cout << "  (π, 2.275)   ≈ (3.14159, 2.275)   with f* = 0.397887" << std::endl;
    std::cout << "  (9.42478, 2.475)                   with f* = 0.397887" << std::endl;
    std::cout << std::endl;

    // Test function evaluation at known minima
    std::cout << "Testing function evaluation at known minima:" << std::endl;
    Eigen::VectorXd test_point(2);
    
    test_point << -M_PI, 12.275;
    std::cout << "f(-π, 12.275) = " << branin(test_point) << " (should be ≈ 0.397887)" << std::endl;
    
    test_point << M_PI, 2.275;
    std::cout << "f(π, 2.275) = " << branin(test_point) << " (should be ≈ 0.397887)" << std::endl;
    
    test_point << 9.42478, 2.475;
    std::cout << "f(9.42478, 2.475) = " << branin(test_point) << " (should be ≈ 0.397887)" << std::endl;
    std::cout << std::endl;

    // Set up the function
    Funcao func;
    func.addObjective(branin);
    func.addConstraint(bound_constraint);

    // Variable bounds - Branin function standard domain
    Eigen::VectorXd var_lb(2);
    var_lb << -5.0, 0.0;
    
    Eigen::VectorXd var_ub(2); 
    var_ub << 10.0, 15.0;

    // Constraint bounds
    Eigen::VectorXd con_lb(1);
    con_lb << -100.0; // No effective lower bound
    
    Eigen::VectorXd con_ub(1);
    con_ub << 2.0; // Keep within reasonable ellipse

    // Try different starting points to find different minima
    std::vector<Eigen::Vector2d> starting_points = {
        {-3.0, 12.0},   // Near first minimum (-π, 12.275)
        {3.0, 2.5},     // Near second minimum (π, 2.275)
        {9.0, 2.5},     // Near third minimum (9.42478, 2.475)
        {0.0, 7.5},     // Center of domain
        {-1.0, 5.0}     // Another test point
    };
    
    for (size_t i = 0; i < starting_points.size(); ++i) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "ATTEMPT " << (i+1) << "/5 - Starting from different point" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        Eigen::MatrixXd x0(2, 1);
        x0.col(0) = starting_points[i];
        
        std::cout << "Starting point: (" << x0(0,0) << ", " << x0(1,0) << ")" << std::endl;
        std::cout << "Initial function value: " << branin(x0.col(0)) << std::endl;
        std::cout << "Initial constraint value: " << bound_constraint(x0.col(0)) << std::endl;
        std::cout << std::endl;

        // Parameters that worked for the simple case
        double mu = 1.0;        
        double epsilon = 0.5;   
        double delta = 1e-4;    
        double lambda = 0.01;   

        // Options that worked
        Options options;
        options.tol_radius = 1e-6;      
        options.tol_f = 1e-6;           
        options.tol_measure = 1e-4;     
        options.tol_con = 1e-4;         
        options.pivot_threshold = 1.0 / 8;  
        options.initial_radius = 1.0;   
        options.radius_max = 5.0;       
        options.max_it = 100;           
        options.verbose = true;         

        std::cout << "Starting optimization..." << std::endl;

        try {
            // Execute the optimization
            auto result = l1_penalty_solve(func, x0, mu, epsilon, delta, lambda, 
                                           var_lb, var_ub, con_lb, con_ub, options);
            
            std::cout << std::endl;
            std::cout << "Optimization completed!" << std::endl;
            
            // Print distance to known minima
            std::cout << "\nDistance to known global minima:" << std::endl;
            Eigen::VectorXd final_point = std::get<0>(result);   
            //Eigen::Vector2d final_point(x0(0,0), x0(1,0)); // Assuming x0 is modified in place
            
            Eigen::Vector2d min1(-M_PI, 12.275);
            Eigen::Vector2d min2(M_PI, 2.275);
            Eigen::Vector2d min3(9.42478, 2.475);
            
            std::cout << "  Distance to (-π, 12.275): " << (final_point - min1).norm() << std::endl;
            std::cout << "  Distance to (π, 2.275): " << (final_point - min2).norm() << std::endl;
            std::cout << "  Distance to (9.42478, 2.475): " << (final_point - min3).norm() << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "Error during optimization: " << e.what() << std::endl;
        }
        
        // Break after first success for now
        if (i == 0) break;
    }

    return 0;
}