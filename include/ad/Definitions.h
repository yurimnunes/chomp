#pragma once

#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iostream>


#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class Variable;
class Expression;
class MyConstraint;
class ADNode;
class ADGraph;

using VariablePtr = std::shared_ptr<Variable>;
using ExpressionPtr = std::shared_ptr<Expression>;
using ConstraintPtr = std::shared_ptr<MyConstraint>;
using ADNodePtr = std::shared_ptr<ADNode>;
using ADGraphPtr = std::shared_ptr<ADGraph>;

enum class Operator {
    NA = 0,
    cte,
    Var,
    Add,
    Subtract,
    Multiply,
    Divide,
    Sin,     // NEW
    Cos,     // NEW
    Tan,
    Exp,
    Log,
    Max,
    Tanh,    // NEW
    Silu,    // NEW
    Gelu,    // NEW
    Relu,    // NEW
    Softmax,
    Abs,     // NEW
    Sqrt,    // NEW
    Pow      // NEW
};
enum class NLP { NA, Bilinear };


#endif // DEFINITIONS_H
