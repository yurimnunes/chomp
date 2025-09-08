# NLP Solver

A modular **Nonlinear Programming (NLP) solver** supporting multiple strategies:

* **SQP (Sequential Quadratic Programming)** — Newton-type steps with line search, trust regions, and filter/funnel acceptance.
* **IP (Interior Point)** — log-barrier method with slack variables, inspired by IPOPT, for handling inequality constraints smoothly.

This solver integrates **automatic differentiation**, **sparse linear algebra**, and **regularization** strategies to robustly tackle constrained nonlinear optimization problems.

---

## ✨ Features

* **Flexible modes**

  * `sqp`: Sequential Quadratic Programming with advanced globalization (trust region, line search, filter, funnel).
  * `ip`: Interior Point method with slack variables and log-barrier.
  * `auto`: Automatic mode selection depending on the problem structure.
* **Constraint handling**

  * Equalities and inequalities.
  * Slack variable transformation for IP mode.
* **Linear algebra backends**

  * Sparse factorizations via **QDLDL** (LDLᵀ).
  * Dense/sparse fallback with **SciPy**.
* **Regularization**

  * Adaptive handling of indefinite or rank-deficient Hessians.
  * Preserves sparsity when possible.
* **Extensible**

  * Modular design for models, regularizers, step acceptance, and KKT solvers.

---

Dependencies:

* Python ≥ 3.9
* NumPy, SciPy

---

## 🚀 Usage

### Define a model

```python
import numpy as np
from nlp.nlp import NLPSolver, SQPConfig

# Rosenbrock function
def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

# Circle inequality: x1² + x2² ≤ 25
c_ineq = [lambda x: x[0]**2 + x[1]**2 - 25]

x0 = np.array([3.0, 2.0])

solver = NLPSolver(rosenbrock, c_ineq=c_ineq, x0=x0, config=SQPConfig())
```

### Run SQP

```python
x_opt, lam, info = solver.solve(mode="sqp")
print("Optimal solution (SQP):", x_opt)
```

### Run Interior Point

```python
x_opt, lam, info = solver.solve(mode="ip")
print("Optimal solution (IP):", x_opt)
```

### Automatic Mode Selection

```python
x_opt, lam, info = solver.solve(mode="auto")
```

---

## 🔬 Examples

* **Unconstrained**: Rosenbrock minimization.
* **Inequality-constrained**: Branin function with circular constraint.
* **Equality-constrained**: Quadratic programming with linear equalities.

See [`examples/`](./examples) for more.

---

## 📖 References

* Nocedal & Wright (2006), *Numerical Optimization*
* Wächter & Biegler (2006), *On the Implementation of a Primal-Dual Interior Point Filter Line Search Algorithm for Large-Scale Nonlinear Programming* (IPOPT)
* Gill, Murray & Wright (1981), *Practical Optimization* (SQP foundations)

---

## 🛠️ Development

* **Bindings**: The solver uses **pybind11** to expose C++ LDLᵀ factorizations (`qdldl_cpp`) to Python.
* **Modules**:

  * `Model`: automatic differentiation wrapper for objectives/constraints.
  * `Regularizer`: Hessian conditioning and inertia fixes.
  * `SQPStepper`: line search, trust region, and filter/funnel methods.
  * `InteriorPointStepper`: slack/barrier based primal-dual IPM.
  * `KKT`: sparse KKT matrix assembly and factorization.