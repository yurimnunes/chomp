#pragma once

#ifdef __cplusplus

#include "Eigen/Core"
#include <Eigen/Dense>
#include <cmath>
#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/ranges.h>

#include <functional>
#include <map>
#include <memory>
#include <vector>

enum ExitFlag {
    EMPTY                    = 0,
    STATUS_POINT_ADDED       = 1,
    STATUS_POINT_REPLACED    = 2,
    STATUS_OLD_MODEL_REBUILT = 3,
    STATUS_MODEL_REBUILT     = 4,
    STATUS_POINT_EXCHANGED   = 5
};

template <typename T>
T eps(T x) {
    return std::nextafter(std::fabs(x), std::numeric_limits<T>::infinity()) - std::fabs(x);
}

inline void displaySolverInfo() {
    // Solver details
    std::string solverName = "l1penalty solver";
    std::string author     = "Laio Oriel Seman";
    std::string email      = "laio@ieee.org";
    std::string version    = "0.2";
    // algorithm based on A derivative-free exact penalty algorithm: basic ideas, convergence theory and computational
    // studies by Giuliani, C.M., Camponogara, E. & Conn, A.R.
    std::string reference = "based on 'A derivative-free exact penalty algorithm:\n    basic ideas, convergence theory "
                            "and computational studies'\n        by Giuliani, C.M., Camponogara, E. & Conn, A.R.";
    std::string url       = "https://doi.org/10.1007/s40314-021-01748-4";
    // std::string cpuInstructions = getCPUInstructionSet();
    fmt::print("{:-^65}\n", "");
    fmt::print(fmt::fg(fmt::color::yellow) | fmt::emphasis::bold, "Solver Information\n");
    fmt::print("{:-^65}\n", "");
    fmt::print("Name: {}\nAuthor: {}\nEmail: {}\nVersion: {}\n", solverName, author, email, version);
    fmt::print("{}\n {}\n", reference, url);
    fmt::print("{:-^65}\n", "");
}

inline void printHeader() {
    fmt::print("\n");
    fmt::print("|{:^12}|{:^12}|{:^12}|{:^12}|{:^12}|\n", "Iter", "Node", "W.Bound", "Bound", "W.Gap");
    fmt::print("|{:-^12}|{:-^12}|{:-^12}|{:-^12}|{:-^12}|\n", "", "", "", "", "", "");
}

// Print line with variable values
inline void printLine(int depth, int iter, std::string node, double bound, double objective, double gap) {
    fmt::print("|{:^12}|{:^12}|{:^12}|{:^12.2f}|{:^12.2f}|{:^12.2f}|\n", depth, iter, node, bound, objective, gap);
}

// Print line with tree structure and variable values, adjusting for tree level visualization
inline void printLineTab(int job, int depth, int iter, std::string node, double bound, double objective, double gap) {
    // Adjusted for consistent alignment with header and line, considering the "├── " prefix
    fmt::print("├──{:>4} @ {:<3}|{:^12}|{:^12}|{:^12.2f}|{:^12.2f}|{:^12.2f}|\n", job, depth, iter, node, bound,
               objective, gap);
}

inline void pyellow(const std::string &message) {
    // Print "DEBUG:" in yellow
    fmt::print(fmt::fg(fmt::color::yellow), "DEBUG: ");

    // Print the message in default color
    fmt::print("{}\n", message);
}

inline void pyblue(const std::string &message) {
    // Print "DEBUG:" in yellow
    fmt::print(fmt::fg(fmt::color::green), "FUNCTION: ");

    // Print the message in default color
    fmt::print("{}\n", message);
}

inline void pEV(const Eigen::VectorXd &v) {
    // print in matri format
    auto vetor = v;
    fmt::print("{}\n", vetor.transpose());
}

inline void pEM(const Eigen::MatrixXd &m) {
    for (int i = 0; i < m.rows(); ++i) {
        for (int j = 0; j < m.cols(); ++j) { fmt::print("{} ", m(i, j)); }
        fmt::print("\n");
    }
}
// define shared pointer to TRModel
class TRModel;
typedef std::shared_ptr<TRModel> TRModelPtr;
class Polynomial;
typedef std::shared_ptr<Polynomial> PolynomialPtr;

struct Options {

    double      tol_radius              = 1e-6;
    double      tol_f                   = 1e-6;
    double      tol_measure             = 1e-5;
    double      tol_con                 = 1e-5;
    double      eps_c                   = 1e-4;
    double      eta_1                   = 0;
    double      eta_2                   = 0.1;
    double      pivot_threshold         = 1.0 / 16;
    double      add_threshold           = 100;
    double      exchange_threshold      = 1000;
    double      initial_radius          = 1;
    double      radius_max              = 1e3;
    double      radius_factor           = 6;
    double      radius_factor_extra_tol = 2;
    double      gamma_inc               = 2;
    double      gamma_dec               = 0.5;
    double      criticality_mu          = 50;
    double      criticality_beta        = 10;
    double      criticality_omega       = 0.5;
    double      max_iter                = 1e6;
    double      divergence_threshold    = 1e10;
    std::string basis                   = "unused option";
    bool        debug                   = false;
    int         inspect_iteration       = 10;
    bool        verbose                 = false;
    bool        first_feasible_only     = false;
    int         max_it                  = 10000;
};

struct Funcao {
    std::function<double(const Eigen::VectorXd &)>              obj;
    std::vector<std::function<double(const Eigen::VectorXd &)>> con;

    // Default constructor
    Funcao() = default;

    // Constructor that sets up a specific objective function based on the provided vector
    Funcao(Eigen::VectorXd x) {
        // Define an example objective function using 'x' here if needed
        obj = [x](const Eigen::VectorXd &y) {
            return (x - y).squaredNorm(); // Example function measuring squared distance
        };
    }

    double evaluateObjective(const Eigen::VectorXd &x) const {
        return obj(x); // Execute the objective function
    }

    double evaluateConstraint(const Eigen::VectorXd &x, int index) const {
        return con[index](x); // Execute the specified constraint function
    }

    Eigen::VectorXd calcAll(const Eigen::VectorXd &x) const {
        Eigen::VectorXd result(1 + con.size());
        result(0) = obj(x); // Calculate the objective function
        // fmt::print("Objective value: {}\n", result(0));
        for (int i = 0; i < con.size(); ++i) {
            result(i + 1) = con[i](x); // Calculate each constraint function
            // fmt::print("Constraint value: {}\n", result(i + 1));
        }
        return result;
    }

    void addObjective(const std::function<double(const Eigen::VectorXd &)> &obj) {
        this->obj = obj; // Set the objective function
    }

    // Method to add a constraint
    void addConstraint(const std::function<double(const Eigen::VectorXd &)> &constraint) {
        con.push_back(constraint); // Add a constraint function
    }

    // define size method
    int size() const {
        return con.size() + 1; // Return the number of constraints
    }
};

struct Constraint {
    double          c;
    Eigen::VectorXd g;
    Eigen::MatrixXd H;

    Constraint(double c, Eigen::VectorXd g, Eigen::MatrixXd H) : c(c), g(g), H(H) {};

    Constraint shift(const Eigen::VectorXd &s) const {
        // Create a new Constraint initially copying the current state
        Constraint shifted = *this;

        // Calculate new c, g based on the shift vector s
        shifted.c += g.dot(s) + 0.5 * s.transpose() * H * s;
        shifted.g += H * s;

        // Return the new, shifted Constraint
        return shifted;
    }

    void shiftAtplace(const Eigen::VectorXd &s) {
        // Calculate new c, g based on the shift vector s
        c += g.dot(s) + 0.5 * s.transpose() * H * s;
        g += H * s;
    }
};

typedef std::vector<Funcao> Funcoes;

class CModelClass : public std::vector<Constraint> {
public:
    using std::vector<Constraint>::vector;
    using std::vector<Constraint>::operator[];

    void print() const {
        fmt::print("Model constraints:\n");
        for (const auto &constraint : *this) {
            fmt::print("c: {}\n", constraint.c);
            fmt::print("g: ");
            pEV(constraint.g);
            fmt::print("H: ");
            pEM(constraint.H);
        }
    }

    // create operator to access constraint by index
    void setC(int index, const Constraint &constraint) { this->at(index) = constraint; }

    // Method to get an Eigen::VectorXd of all 'c' values from the Constraints
    Eigen::VectorXd c() const {
        Eigen::VectorXd c_values(this->size());
        int             index = 0;
        for (const auto &constraint : *this) { c_values(index++) = constraint.c; }
        return c_values;
    }

    void addConstraint(const Constraint &constraint) { this->push_back(constraint); }

    std::vector<Constraint> filter(std::vector<bool> active) {
        std::vector<Constraint> filtered_cts;
        for (int i = 0; i < active.size(); ++i) {
            if (active[i]) { filtered_cts.push_back(this->at(i)); }
        }
        return filtered_cts;
    }

    struct FilterProxy {
        const CModelClass       &model;
        const std::vector<bool> &filter;

        FilterProxy(const CModelClass &model, const std::vector<bool> &filter) : model(model), filter(filter) {}

        // Convert FilterProxy to CModelClass by applying the filter
        operator CModelClass() const {
            CModelClass filteredModel;
            filteredModel.reserve(filter.size());
            for (size_t i = 0; i < filter.size(); ++i) {
                if (i < model.size() && filter[i]) { filteredModel.push_back(model[i]); }
            }
            return filteredModel;
        }

        // method to get c at position k

        // method to get 'c' values directly from FilterProxy
        Eigen::VectorXd c() const {
            Eigen::VectorXd c_values;
            c_values.resize(filter.size());
            for (size_t i = 0; i < filter.size(); ++i) {
                if (i < model.size() && filter[i]) {
                    c_values.conservativeResize(c_values.size() + 1);
                    c_values(c_values.size() - 1) = model[i].c;
                }
            }
            return c_values;
        }

        // method to get g values directly from FilterProxy
        Eigen::MatrixXd g() const {
            Eigen::MatrixXd g_values;
            g_values.resize(filter.size(), model[0].g.size());
            for (size_t i = 0; i < filter.size(); ++i) {
                if (i < model.size() && filter[i]) { g_values.row(i) = model[i].g; }
            }
            return g_values;
        }
    };

    // Overload [] to return a FilterProxy
    FilterProxy operator[](const std::vector<bool> &filter) const { return FilterProxy(*this, filter); }
};

typedef std::shared_ptr<CModelClass> CModel;

/*
template <typename T>
T shiftListOfModels(T &modelPtr, const Eigen::VectorXd &s) {

    auto model = std::make_shared<TRModel>(*modelPtr);
    //T model = std::make_shared<TRModel>(*modelPtr);
    //T cmodel = std::make_shared<CModelClass>(*modelPtr);
    //fmt::print("cmodel size: {}\n", cmodel->size());
    for (int k = 0; k < modelPtr->size(); ++k) {
        model->at(k).shiftAtplace(s);
    }

    return model;
}
*/

template <typename T>
T shiftListOfModels(T &modelPtr, const Eigen::VectorXd &s) {

    auto model = *(modelPtr);
    // reverse iterate over the model
    for (int k = model.size() - 1; k >= 0; --k) { model.at(k).shiftAtplace(s); }

    return std::make_shared<CModelClass>(model);
}

// define shift_model as template both for TRModel and CModel
template <typename T>
T shiftModel(T &modelOriginal, Eigen::VectorXd &s) {
    // Ensure T is indeed a std::shared_ptr<TRModel>
    static_assert(std::is_same<T, std::shared_ptr<TRModel>>::value, "T must be a std::shared_ptr<TRModel>");

    // Create a new model by copying the original model
    T    model  = std::make_shared<TRModel>(*modelOriginal);
    auto c_comp = model->c() + model->g().transpose() * s + 0.5 * (s.transpose() * model->H() * s).value();
    // Update the new model's parameters
    model->set_c(c_comp);
    model->set_g(model->g() + model->H() * s);

    return model;
}

inline std::vector<double> solveQuadratic(double a, double b, double c) {

    // check if a is zero
    if (a == 0) {
        if (b == 0) { throw std::runtime_error("Both a and b are zero"); }
        return {-c / b};
    }

    std::vector<double> roots;
    double              discriminant = b * b - 4 * a * c;
    if (discriminant >= 0) { // Only interested in real roots
        double sqrtD = std::sqrt(discriminant);

        // Stable quadratic roots according to BKP Horn.
        // http://people.csail.mit.edu/bkph/articles/Quadratics.pdf
        double root1, root2;
        if (b >= 0) {
            root1 = (-b - sqrtD) / (2.0 * a);
            root2 = (2.0 * c) / (-b - sqrtD);
        } else {
            root1 = (2.0 * c) / (-b + sqrtD);
            root2 = (-b + sqrtD) / (2.0 * a);
        }
        // double root1 = (-b + sqrtD) / (2 * a);
        // double root2 = (-b - sqrtD) / (2 * a);
        roots.push_back(root1);
        roots.push_back(root2);
    }
    else {
        // use the real part of the complex roots
        double realPart = -b / (2 * a);
        //double imagPart = std::sqrt(-discriminant) / (2 * a);
        roots.push_back(realPart);
    }
    return roots;
}

struct FModel {
    double          c;
    Eigen::VectorXd g;
    Eigen::MatrixXd H;

    FModel(double c, Eigen::MatrixXd g, Eigen::MatrixXd H) : c(c), g(g), H(H) {}
    FModel() = default;

    void shift(const Eigen::VectorXd &s) {
        c += g.dot(s) + 0.5 * s.transpose() * H * s;
        g += H * s;
    }
};

#endif