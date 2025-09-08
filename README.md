<img src="logo.png" alt="logo" width="400"/>

# CHOMP — *C*HOMP *H*andles *O*ptimization of *M*any *P*roblems (recurrently)

**CHOMP** is a modular **Nonlinear Programming (NLP) solver** that can *chomp through tough constrained problems* using multiple state-of-the-art strategies:

* **SQP (Sequential Quadratic Programming)** — Newton-type steps with globalization (line search, trust region, filter/funnel acceptance).
* **IP (Interior Point)** — log-barrier primal-dual method with slack variables, inspired by **IPOPT**, for smooth handling of inequalities.
* **AUTO** — automatic mode selection depending on problem structure.

CHOMP combines **automatic differentiation**, **sparse linear algebra**, and **advanced regularization** to solve nonlinear constrained optimization problems reliably and efficiently.

---

## ✨ Features

* **Multiple solving strategies**

  * `sqp`: Sequential Quadratic Programming with advanced globalization.
  * `ip`: Interior Point method with slack + log-barrier.
  * `auto`: Automatic mode selection.

* **Constraint handling**

  * Equalities and inequalities.
  * Automatic slack transformation for inequalities in IP mode.

* **Linear algebra backends**

  * Sparse LDLᵀ factorizations via **QDLDL**.
  * Dense/sparse fallbacks using **SciPy**.

* **Robust regularization**

  * Adaptive handling of indefinite or rank-deficient Hessians.
  * Sparsity-preserving regularization strategies.

* **Extensible design**

  * Modular components: `Model`, `Regularizer`, `Stepper`, `KKT`.
  * Easy to plug in new globalization, regularization, or solver backends.

---

## 📦 Installation

```bash
git clone https://github.com/your-org/chomp.git
cd chomp
pip install -e .
```

**Dependencies:**

* Python ≥ 3.9
* NumPy, SciPy
* [qdldl-cpp](https://github.com/oxfordcontrol/qdldl) (exposed via pybind11)

---

## 🚀 Usage

### Define a model (classic API)

```python
import numpy as np
from chomp import NLPSolver, SQPConfig

# Rosenbrock objective
def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

# Circle inequality: x1² + x2² ≤ 25
c_ineq = [lambda x: x[0]**2 + x[1]**2 - 25]

x0 = np.array([3.0, 2.0])

solver = NLPSolver(rosenbrock, c_ineq=c_ineq, x0=x0, config=SQPConfig())
x_opt, lam, info = solver.solve(mode="sqp")
```

---

## 🧩 Wrapper API

CHOMP also provides a **wrapper-based modeling interface** (`nlp/wrapper.py`) that lets you write constraints and objectives using a symbolic API with operator overloading.
This is especially useful for building models programmatically while still supporting autodiff.

### Example: Branin with circular constraint

```python
import math
import numpy as np
from nlp.wrapper import Model, cos  # AD-aware functions
from nlp.nlp import NLPSolver, SQPConfig

def build_branin_model(use_ineq=True, R=60.0, mode="auto"):
    m = Model("branin")

    # Decision variables
    x = m.add_var("x", shape=2)
    x1, x2 = x[0], x[1]

    # Branin constants
    a, b, c, r, s, t = 1.0, 5.1 / (4*math.pi**2), 5.0 / math.pi, 6.0, 10.0, 1/(8*math.pi)

    # Objective
    f = a*(x2 - b*(x1**2) + c*x1 - r)**2 + s*(1 - t)*cos(x1) + s
    m.minimize(f)

    # Optional inequality: circle constraint
    if use_ineq:
        m.add_constr(x1**2 + x2**2 - R**2 <= 0.0)

    # Build solver
    f_fun, c_ineq, c_eq, x0 = m.build()
    solver = NLPSolver(f=f_fun, c_ineq=c_ineq, c_eq=c_eq, x0=np.array([-3.0, 12.0]), config=SQPConfig())
    return solver

solver = build_branin_model()
x_star, hist = solver.solve(max_iter=150, tol=1e-8, verbose=True)
print("x* =", x_star)
```

### Wrapper Features

* **Symbolic expressions** (`Expr`) with operator overloading (`+`, `-`, `*`, `/`, `**`, `@`).
* **Constraint API** with natural syntax:

  ```python
  m.add_constr(x1**2 + x2**2 <= 25)
  ```
* **AD-aware functions**: `sin`, `cos`, `exp`, `log`, `sqrt`, `tanh`, etc.
* **Convenience reductions**: `norm1`, `norm2`, `norm_inf`, `dot`, `esum`.
* **Automatic initial guesses**: built from bounds or user-provided values.

This makes CHOMP both a **solver backend** and a **modeling tool**, all in one lightweight package.

---

### Wrapper Commands

| Command                                                | Description                                                                               | Example                               |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------------- | ------------------------------------- |
| `m.add_var(name, shape=(), lb=None, ub=None, x0=None)` | Add a decision variable (scalar, vector, or matrix). Supports bounds and initial guesses. | `x = m.add_var("x", shape=2)`         |
| `m.set_initial(name, x0)`                              | Set or override the initial guess for a variable.                                         | `m.set_initial("x", [1.0, 2.0])`      |
| `m.minimize(expr)`                                     | Set objective (minimization).                                                             | `m.minimize(x1**2 + x2**2)`           |
| `m.maximize(expr)`                                     | Set objective (maximization).                                                             | `m.maximize(x1 + x2)`                 |
| `m.add_constr(lhs <= rhs)`                             | Add inequality constraint.                                                                | `m.add_constr(x1**2 + x2**2 <= 25)`   |
| `m.add_constr(lhs == rhs)`                             | Add equality constraint.                                                                  | `m.add_constr(x1 + x2 == 1)`          |
| `m.build()`                                            | Compile model into callables: `(f, c_ineq, c_eq, x0)` for use with `NLPSolver`.           | `f_fun, c_ineq, c_eq, x0 = m.build()` |

---

## 🔬 Examples

* **Unconstrained**: Rosenbrock minimization
* **Inequality-constrained**: Branin function + circular constraint
* **Equality-constrained**: Quadratic programming with linear equalities

See [`examples/`](./examples) for runnable demos.

---

## 📖 References

* Nocedal & Wright (2006), *Numerical Optimization*
* Wächter & Biegler (2006), *Primal-Dual Interior Point Filter Line Search Algorithm* (IPOPT)
* Gill, Murray & Wright (1981), *Practical Optimization*

---

## 🛠️ Development Notes

* **Bindings**: `pybind11` exposes **C++ LDLᵀ factorizations** (`qdldl_cpp`) to Python.
* **Modules**:

  * `Model`: automatic differentiation wrapper for objectives/constraints.
  * `Regularizer`: Hessian conditioning and inertia correction.
  * `SQPStepper`: globalization via trust region, line search, filter, funnel.
  * `InteriorPointStepper`: slack/barrier-based primal-dual IPM.
  * `KKT`: sparse KKT system assembly and factorization.
* **Wrapper**: lightweight symbolic layer for building problems declaratively.