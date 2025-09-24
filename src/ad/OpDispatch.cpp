// #include "../../include/ad/OpDispatch.h"
// #include <memory>

// // Helper functions (these were missing from your implementation)
// inline void set_epoch_value(double& value, int& val_epoch, int cur_val_epoch, double new_value) {
//     if (val_epoch != cur_val_epoch) {
//         val_epoch = cur_val_epoch;
//         value = new_value;
//     } else {
//         value = new_value;  // Still update even if epochs match
//     }
// }

// inline double& ensure_epoch_zero(double& gradient, int& grad_epoch, int cur_grad_epoch) {
//     if (grad_epoch != cur_grad_epoch) {
//         grad_epoch = cur_grad_epoch;
//         gradient = 0.0;
//     }
//     return gradient;
// }

// inline void accumulate_gradient(ADNode& node, ADGraph& graph, double grad_value) {
//     ensure_epoch_zero(node.gradient, node.grad_epoch, graph.cur_grad_epoch_) += grad_value;
// }

// // Initialize jump tables
// const std::array<FastDispatch::ForwardFunc, 17> FastDispatch::forward_table = {{
//     &FastDispatch::forward_var,      // 0: Operator::Var
//     &FastDispatch::forward_cte,      // 1: Operator::cte
//     &FastDispatch::forward_add,      // 2: Operator::Add
//     &FastDispatch::forward_subtract, // 3: Operator::Subtract
//     &FastDispatch::forward_multiply, // 4: Operator::Multiply
//     &FastDispatch::forward_divide,   // 5: Operator::Divide
//     &FastDispatch::forward_sin,      // 6: Operator::Sin
//     &FastDispatch::forward_cos,      // 7: Operator::Cos
//     &FastDispatch::forward_tan,      // 8: Operator::Tan
//     &FastDispatch::forward_exp,      // 9: Operator::Exp
//     &FastDispatch::forward_log,      // 10: Operator::Log
//     &FastDispatch::forward_tanh,     // 11: Operator::Tanh
//     &FastDispatch::forward_relu,     // 12: Operator::Relu
//     &FastDispatch::forward_max,      // 13: Operator::Max
//     &FastDispatch::forward_gelu,     // 14: Operator::Gelu
//     &FastDispatch::forward_silu,     // 15: Operator::Silu
//     &FastDispatch::forward_softmax   // 16: Operator::Softmax
// }};

// const std::array<FastDispatch::BackwardFunc, 17> FastDispatch::backward_table = {{
//     &FastDispatch::backward_var,
//     &FastDispatch::backward_cte,
//     &FastDispatch::backward_add,
//     &FastDispatch::backward_subtract,
//     &FastDispatch::backward_multiply,
//     &FastDispatch::backward_divide,
//     &FastDispatch::backward_sin,
//     &FastDispatch::backward_cos,
//     &FastDispatch::backward_tan,
//     &FastDispatch::backward_exp,
//     &FastDispatch::backward_log,
//     &FastDispatch::backward_tanh,
//     &FastDispatch::backward_relu,
//     &FastDispatch::backward_max,
//     &FastDispatch::backward_gelu,
//     &FastDispatch::backward_silu,
//     &FastDispatch::backward_softmax
// }};

// // Forward implementations
// void FastDispatch::forward_var(ADNode& node, ADGraph& graph) {
//     // Variables already have their values set
//     set_epoch_value(node.value, node.val_epoch, graph.cur_val_epoch_, node.value);
// }

// void FastDispatch::forward_cte(ADNode& node, ADGraph& graph) {
//     // Constants keep their fixed values
//     set_epoch_value(node.value, node.val_epoch, graph.cur_val_epoch_, node.value);
// }

// void FastDispatch::forward_add(ADNode& node, ADGraph& graph) {
//     const size_t m = node.inputs.size();
//     if (m == 0) return;

//     double sum = 0.0;
//     for (const auto& input : node.inputs) {
//         sum += input->value;
//     }
//     set_epoch_value(node.value, node.val_epoch, graph.cur_val_epoch_, sum);
// }

// void FastDispatch::forward_subtract(ADNode& node, ADGraph& graph) {
//     if (node.inputs.size() != 2) return;
    
//     const double a = node.inputs[0]->value;
//     const double b = node.inputs[1]->value;
//     set_epoch_value(node.value, node.val_epoch, graph.cur_val_epoch_, a - b);
// }

// void FastDispatch::forward_multiply(ADNode& node, ADGraph& graph) {
//     const size_t m = node.inputs.size();
//     if (m == 0) return;

//     // Fast implementation with zero detection (matching OpTraits logic)
//     size_t zero_count = 0;
//     double prod_nonzero = 1.0;
    
//     for (const auto& input : node.inputs) {
//         const double val = input->value;
//         if (val == 0.0) {
//             zero_count++;
//         } else {
//             prod_nonzero *= val;
//         }
//     }
    
//     const double result = (zero_count == 0) ? prod_nonzero : 0.0;
//     set_epoch_value(node.value, node.val_epoch, graph.cur_val_epoch_, result);
// }

// void FastDispatch::forward_divide(ADNode& node, ADGraph& graph) {
//     if (node.inputs.size() != 2) return;
    
//     const double a = node.inputs[0]->value;
//     const double b = node.inputs[1]->value;
//     const double result = (b != 0.0) ? a / b : 0.0;  // Using safe division like OpTraits
//     set_epoch_value(node.value, node.val_epoch, graph.cur_val_epoch_, result);
// }

// void FastDispatch::forward_sin(ADNode& node, ADGraph& graph) {
//     if (node.inputs.size() != 1) return;
    
//     const double x = node.inputs[0]->value;
//     set_epoch_value(node.value, node.val_epoch, graph.cur_val_epoch_, std::sin(x));
// }

// void FastDispatch::forward_cos(ADNode& node, ADGraph& graph) {
//     if (node.inputs.size() != 1) return;
    
//     const double x = node.inputs[0]->value;
//     set_epoch_value(node.value, node.val_epoch, graph.cur_val_epoch_, std::cos(x));
// }

// void FastDispatch::forward_tan(ADNode& node, ADGraph& graph) {
//     if (node.inputs.size() != 1) return;
    
//     const double x = node.inputs[0]->value;
//     set_epoch_value(node.value, node.val_epoch, graph.cur_val_epoch_, std::tan(x));
// }

// void FastDispatch::forward_exp(ADNode& node, ADGraph& graph) {
//     if (node.inputs.size() != 1) return;
    
//     const double x = node.inputs[0]->value;
//     set_epoch_value(node.value, node.val_epoch, graph.cur_val_epoch_, std::exp(x));
// }

// void FastDispatch::forward_log(ADNode& node, ADGraph& graph) {
//     if (node.inputs.size() != 1) return;
    
//     const double x = node.inputs[0]->value;
//     // Match OpTraits: safe log with minimum value instead of infinity
//     const double result = (x > 0.0) ? std::log(x) : std::log(1e-16);
//     set_epoch_value(node.value, node.val_epoch, graph.cur_val_epoch_, result);
// }

// void FastDispatch::forward_tanh(ADNode& node, ADGraph& graph) {
//     if (node.inputs.size() != 1) return;
    
//     const double x = node.inputs[0]->value;
//     set_epoch_value(node.value, node.val_epoch, graph.cur_val_epoch_, std::tanh(x));
// }

// void FastDispatch::forward_relu(ADNode& node, ADGraph& graph) {
//     if (node.inputs.size() != 1) return;
    
//     const double x = node.inputs[0]->value;
//     set_epoch_value(node.value, node.val_epoch, graph.cur_val_epoch_, (x > 0.0) ? x : 0.0);
// }

// void FastDispatch::forward_max(ADNode& node, ADGraph& graph) {
//     if (node.inputs.size() != 2) return;
    
//     const double a = node.inputs[0]->value;
//     const double b = node.inputs[1]->value;
//     set_epoch_value(node.value, node.val_epoch, graph.cur_val_epoch_, (a >= b) ? a : b);
// }

// // Helper function for stable sigmoid (matching OpTraits)
// inline double stable_sigmoid(double x) {
//     if (x >= 0.0) {
//         const double z = std::exp(-x);
//         return 1.0 / (1.0 + z);
//     } else {
//         const double z = std::exp(x);
//         return z / (1.0 + z);
//     }
// }

// void FastDispatch::forward_gelu(ADNode& node, ADGraph& graph) {
//     if (node.inputs.size() != 1) return;
    
//     const double x = node.inputs[0]->value;
//     const double z = x * M_SQRT1_2;  // x / sqrt(2)
//     const double result = 0.5 * x * (1.0 + std::erf(z));
//     set_epoch_value(node.value, node.val_epoch, graph.cur_val_epoch_, result);
// }

// void FastDispatch::forward_silu(ADNode& node, ADGraph& graph) {
//     if (node.inputs.size() != 1) return;
    
//     const double x = node.inputs[0]->value;
//     const double sigmoid = stable_sigmoid(x);
//     set_epoch_value(node.value, node.val_epoch, graph.cur_val_epoch_, x * sigmoid);
// }

// void FastDispatch::forward_softmax(ADNode& node, ADGraph& graph) {
//     const size_t m = node.inputs.size();
//     if (m == 0) return;

//     // Collect input values
//     std::vector<double> x(m);
//     for (size_t i = 0; i < m; ++i) {
//         x[i] = node.inputs[i]->value;
//     }

//     // Stable softmax
//     double xmax = -std::numeric_limits<double>::infinity();
//     for (double xi : x) {
//         if (xi > xmax) xmax = xi;
//     }

//     double Z = 0.0;
//     for (double xi : x) {
//         Z += std::exp(xi - xmax);
//     }
//     if (Z <= 0.0) Z = 1.0;

//     // Return first component (matching OpTraits)
//     const double result = std::exp(x[0] - xmax) / Z;
//     set_epoch_value(node.value, node.val_epoch, graph.cur_val_epoch_, result);
// }

// // Backward implementations
// void FastDispatch::backward_var(ADNode& node, ADGraph& graph) {
//     // Variables accumulate gradients - no propagation needed
// }

// void FastDispatch::backward_cte(ADNode& node, ADGraph& graph) {
//     // Constants don't propagate gradients
// }

// void FastDispatch::backward_add(ADNode& node, ADGraph& graph) {
//     const double grad = node.gradient;
//     for (const auto& input : node.inputs) {
//         accumulate_gradient(*input, graph, grad);
//     }
// }

// void FastDispatch::backward_subtract(ADNode& node, ADGraph& graph) {
//     if (node.inputs.size() != 2) return;
    
//     const double grad = node.gradient;
//     accumulate_gradient(*node.inputs[0], graph, grad);
//     accumulate_gradient(*node.inputs[1], graph, -grad);
// }

// void FastDispatch::backward_multiply(ADNode& node, ADGraph& graph) {
//     const size_t m = node.inputs.size();
//     if (m == 0) return;

//     const double grad = node.gradient;
    
//     // Match OpTraits zero-handling logic
//     size_t zero_count = 0;
//     size_t zero_index = (size_t)-1;
//     double prod_nonzero = 1.0;

//     for (size_t i = 0; i < m; ++i) {
//         const double v = node.inputs[i]->value;
//         if (v == 0.0) {
//             if (++zero_count == 1) zero_index = i;
//         } else {
//             prod_nonzero *= v;
//         }
//     }

//     if (zero_count >= 2) return;  // Multiple zeros -> all gradients are zero

//     if (zero_count == 1) {
//         // Only one zero -> gradient flows only to that input
//         accumulate_gradient(*node.inputs[zero_index], graph, grad * prod_nonzero);
//         return;
//     }

//     // No zeros -> standard product rule
//     for (size_t i = 0; i < m; ++i) {
//         const double xi = node.inputs[i]->value;
//         accumulate_gradient(*node.inputs[i], graph, grad * (prod_nonzero / xi));
//     }
// }

// void FastDispatch::backward_divide(ADNode& node, ADGraph& graph) {
//     if (node.inputs.size() != 2) return;
    
//     const double grad = node.gradient;
//     const double a = node.inputs[0]->value;
//     const double b = node.inputs[1]->value;
    
//     if (b != 0.0) {
//         const double inv_b = 1.0 / b;
//         accumulate_gradient(*node.inputs[0], graph, grad * inv_b);
//         accumulate_gradient(*node.inputs[1], graph, grad * (-a * inv_b * inv_b));
//     }
// }

// void FastDispatch::backward_sin(ADNode& node, ADGraph& graph) {
//     if (node.inputs.size() != 1) return;
    
//     const double grad = node.gradient;
//     const double x = node.inputs[0]->value;
//     accumulate_gradient(*node.inputs[0], graph, grad * std::cos(x));
// }

// void FastDispatch::backward_cos(ADNode& node, ADGraph& graph) {
//     if (node.inputs.size() != 1) return;
    
//     const double grad = node.gradient;
//     const double x = node.inputs[0]->value;
//     accumulate_gradient(*node.inputs[0], graph, grad * (-std::sin(x)));
// }

// void FastDispatch::backward_tan(ADNode& node, ADGraph& graph) {
//     if (node.inputs.size() != 1) return;
    
//     const double grad = node.gradient;
//     const double x = node.inputs[0]->value;
//     const double cos_x = std::cos(x);
//     // Match OpTraits numerical stability check
//     const double sec_sq = (std::abs(cos_x) > 1e-12) ? (1.0 / (cos_x * cos_x)) : 0.0;
//     accumulate_gradient(*node.inputs[0], graph, grad * sec_sq);
// }

// void FastDispatch::backward_exp(ADNode& node, ADGraph& graph) {
//     if (node.inputs.size() != 1) return;
    
//     const double grad = node.gradient;
//     const double x = node.inputs[0]->value;
//     accumulate_gradient(*node.inputs[0], graph, grad * std::exp(x));
// }

// void FastDispatch::backward_log(ADNode& node, ADGraph& graph) {
//     if (node.inputs.size() != 1) return;
    
//     const double grad = node.gradient;
//     const double x = node.inputs[0]->value;
//     // Match OpTraits: only propagate if x > 0
//     if (x > 0.0) {
//         accumulate_gradient(*node.inputs[0], graph, grad / x);
//     }
// }

// void FastDispatch::backward_tanh(ADNode& node, ADGraph& graph) {
//     if (node.inputs.size() != 1) return;
    
//     const double grad = node.gradient;
//     const double x = node.inputs[0]->value;
//     const double tanh_x = std::tanh(x);
//     const double sech_sq = 1.0 - tanh_x * tanh_x;
//     accumulate_gradient(*node.inputs[0], graph, grad * sech_sq);
// }

// void FastDispatch::backward_relu(ADNode& node, ADGraph& graph) {
//     if (node.inputs.size() != 1) return;
    
//     const double grad = node.gradient;
//     const double x = node.inputs[0]->value;
//     if (x > 0.0) {
//         accumulate_gradient(*node.inputs[0], graph, grad);
//     }
// }

// void FastDispatch::backward_max(ADNode& node, ADGraph& graph) {
//     if (node.inputs.size() != 2) return;
    
//     const double grad = node.gradient;
//     const double a = node.inputs[0]->value;
//     const double b = node.inputs[1]->value;
    
//     if (a >= b) {
//         accumulate_gradient(*node.inputs[0], graph, grad);
//     } else {
//         accumulate_gradient(*node.inputs[1], graph, grad);
//     }
// }

// void FastDispatch::backward_gelu(ADNode& node, ADGraph& graph) {
//     if (node.inputs.size() != 1) return;
    
//     const double grad = node.gradient;
//     const double x = node.inputs[0]->value;
    
//     // Match OpTraits GELU derivative calculation
//     const double z = x * M_SQRT1_2;
//     const double A = std::sqrt(2.0 / M_PI) * std::exp(-0.5 * x * x);
//     const double derivative = 0.5 * (1.0 + std::erf(z)) + 0.5 * x * A;
    
//     accumulate_gradient(*node.inputs[0], graph, grad * derivative);
// }

// void FastDispatch::backward_silu(ADNode& node, ADGraph& graph) {
//     if (node.inputs.size() != 1) return;
    
//     const double grad = node.gradient;
//     const double x = node.inputs[0]->value;
//     const double sigmoid = stable_sigmoid(x);
//     // Match OpTraits SiLU derivative: σ(x) * (1 + x * (1 - σ(x)))
//     const double derivative = sigmoid * (1.0 + x * (1.0 - sigmoid));
    
//     accumulate_gradient(*node.inputs[0], graph, grad * derivative);
// }

// void FastDispatch::backward_softmax(ADNode& node, ADGraph& graph) {
//     const size_t m = node.inputs.size();
//     if (m == 0) return;

//     const double grad = node.gradient;
    
//     // Collect input values
//     std::vector<double> x(m);
//     for (size_t i = 0; i < m; ++i) {
//         x[i] = node.inputs[i]->value;
//     }

//     // Recompute softmax values for gradient calculation
//     double xmax = -std::numeric_limits<double>::infinity();
//     for (double xi : x) {
//         if (xi > xmax) xmax = xi;
//     }

//     std::vector<double> y(m);
//     double Z = 0.0;
//     for (size_t i = 0; i < m; ++i) {
//         y[i] = std::exp(x[i] - xmax);
//         Z += y[i];
//     }
//     if (Z <= 0.0) Z = 1.0;
    
//     for (size_t i = 0; i < m; ++i) {
//         y[i] /= Z;
//     }

//     // Gradient: ∂y₀/∂xₖ = y₀ * (δ_{k0} - yₖ)
//     const double y0 = y[0];
    
//     for (size_t k = 0; k < m; ++k) {
//         const double delta_k0 = (k == 0) ? 1.0 : 0.0;
//         const double grad_k = y0 * (delta_k0 - y[k]);
//         accumulate_gradient(*node.inputs[k], graph, grad * grad_k);
//     }
// }