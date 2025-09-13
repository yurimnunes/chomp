<img src="docs/logo.png" alt="logo" width="400"/>

# CHOMP — *C*HOMP *H*andles *O*ptimization of *M*any *P*roblems (recurrently)

**CHOMP** is a modular **Nonlinear Programming (NLP) solver** that can *chomp through tough constrained problems* using multiple state-of-the-art strategies:

* **SQP (Sequential Quadratic Programming)** — Newton-type steps with globalization (line search, trust region, filter/funnel acceptance).
* **IP (Interior Point)** — log-barrier primal-dual method with slack variables, inspired by **IPOPT**, for smooth handling of inequalities.
* **DFO-L1 (Derivative-Free L1 Exact Penalty)** — trust-region method for problems where gradients are unavailable, using exact penalty formulation with specialized handling of non-smooth constraints.
* **AUTO** — automatic mode selection depending on problem structure.

CHOMP combines **automatic differentiation**, **sparse linear algebra**, and **advanced regularization** to solve nonlinear constrained optimization problems reliably and efficiently.

---

## ✨ Features

- **Multiple solving strategies**
  - `sqp`: Sequential Quadratic Programming with advanced globalization.
  - `ip`: Interior Point method with slack + log-barrier.
  - `dfo`: Derivative-Free Optimization using L1 exact penalty trust-region method.
  - `auto`: Automatic mode selection.

- **Constraint handling**
  - Equalities and inequalities.
  - Automatic slack transformation (IP mode).
  - Black-box constraints via exact penalty (DFO).
  - Infeasible iterate acceptance.

- **Linear algebra backends**
  - Sparse LDLᵀ via **QDLDL** (pybind11 module `qdldl_cpp`).
  - Dense/sparse fallbacks using **SciPy**.

- **Derivative-free capabilities**
  - Model-based optimization with polynomial interpolation.
  - Adaptive geometry management for trust regions.
  - Black-box objective/constraint support.

- **Robust regularization**
  - Indefinite / rank-deficient Hessian handling.
  - Sparsity-preserving strategies.

- **Extensible design**
  - Modular: `Model`, `Regularizer`, `Stepper`, `KKT`.
  - Easy to extend with new globalization or linear algebra backends.

---

## 📦 Installation

### Quick install (Python)

```bash
git clone https://github.com/your-org/chomp.git
cd chomp
pip install -e .
````

**Runtime Dependencies**

* Python ≥ 3.9
* NumPy, SciPy

> If you build from source (below), the C++/pybind modules (`ad`, `l1core`, `qdldl_cpp`, etc.) are compiled and installed automatically.

---

## 🔨 Build from source (CMake + pybind11)

CHOMP uses **CMake** to build its C++ solver kernels and expose them via **pybind11** to Python.

```bash
git clone https://github.com/your-org/chomp.git
cd chomp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

This builds all enabled Python modules (e.g., `ad`, `l1core`, `simplex_core`, `qdldl_cpp`, `osqp_cpp`, `piqp_cpp`, `amdqg`) and places the generated `.so`/`.pyd` files where Python can import them (editable install recommended).

### Build Options

Toggle with `-D<OPTION>=ON/OFF` when calling `cmake ..`:

| Option             | Default | Description                                       |
| ------------------ | ------- | ------------------------------------------------- |
| `AD_ENABLE_OPENMP` | OFF     | Enable OpenMP parallelization in C++ kernels.     |
| `AD_ENABLE_PIQP`   | ON      | Build PIQP (primal-dual QP) Python bindings.      |
| `AD_ENABLE_OSQP`   | ON      | Build OSQP Python bindings.                       |
| `AD_ENABLE_QDLDL`  | ON      | Build QDLDL sparse LDLᵀ bindings.                 |
| `AD_ENABLE_AMDQG`  | ON      | Build AMDQG ordering library bindings.            |
| `AD_ENABLE_L1CORE` | ON      | Build L1 penalty trust-region core (DFO backend). |

Example:

```bash
cmake .. -DAD_ENABLE_OPENMP=ON -DAD_ENABLE_OSQP=OFF
cmake --build . -j
```

**Dependencies (CMake side)**

* **pybind11 ≥ 2.10**
* **Eigen ≥ 3.3**
* **fmt ≥ 9**
* **Python ≥ 3.9 (Interpreter + Development headers)**
* (Optional) OpenMP

If not found on your system, these are fetched automatically via **CPM.cmake** during configure.

**Platform notes**

* **Linux/macOS**: GCC ≥ 11 / Clang ≥ 14 recommended. Apple Clang is fine.
* **Windows**: MSVC 2022 recommended; consider `-DCMAKE_BUILD_TYPE=Release`.

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

### Derivative-free optimization (black-box functions)

```python
import numpy as np
from chomp import NLPSolver, DFOConfig

def expensive_simulation(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2 + 0.1 * np.sin(10 * x[0])

def constraint_simulation(x):
    return x[0]**2 + x[1]**2 - 4  # x1² + x2² ≤ 4

x0 = np.array([0.5, 0.5])
solver = NLPSolver(expensive_simulation, c_ineq=[constraint_simulation],
                   x0=x0, config=DFOConfig())
x_opt, lam, info = solver.solve(mode="dfo")
```

---

## 🧩 Wrapper API

Symbolic modeling via `nlp/wrapper.py` (operator-overloaded expressions with AD).

**Example: Branin with circular constraint**

```python
import math
import numpy as np
from nlp.wrapper import Model, cos
from nlp.nlp import NLPSolver, SQPConfig

def build_branin_model(use_ineq=True, R=60.0):
    m = Model("branin")

    x = m.add_var("x", shape=2)
    x1, x2 = x[0], x[1]

    a, b, c, r, s, t = 1.0, 5.1/(4*math.pi**2), 5.0/math.pi, 6.0, 10.0, 1/(8*math.pi)
    f = a*(x2 - b*(x1**2) + c*x1 - r)**2 + s*(1 - t)*cos(x1) + s
    m.minimize(f)

    if use_ineq:
        m.add_constr(x1**2 + x2**2 - R**2 <= 0.0)

    f_fun, c_ineq, c_eq, x0 = m.build()
    solver = NLPSolver(f=f_fun, c_ineq=c_ineq, c_eq=c_eq, x0=np.array([-3.0, 12.0]),
                       config=SQPConfig())
    return solver

solver = build_branin_model()
x_star, hist = solver.solve(max_iter=150, tol=1e-8, verbose=True)
print("x* =", x_star)
```

See **Wrapper Commands** and function list in the original README for full syntax.

---

## 🔬 Examples

* **Unconstrained**: Rosenbrock minimization
* **Inequality-constrained**: Branin + circular constraint
* **Equality-constrained**: Quadratic programming with linear equalities
* **Derivative-free**: Black-box simulation optimization with exact penalty
* **Mixed problems**: Combining analytical and simulation-based constraints

Run the notebooks and scripts in [`examples/`](./examples).

---

## 📖 References

* Nocedal & Wright (2006), *Numerical Optimization*
* Wächter & Biegler (2006), *Primal-Dual Interior Point Filter Line Search Algorithm* (IPOPT)
* Gill, Murray & Wright (1981), *Practical Optimization*
* Giuliani, Camponogara & Conn (2022), *A derivative-free exact penalty algorithm: basic ideas, convergence theory and computational studies*, Computational and Applied Mathematics

---

## 🛠️ Development

* C++ solver kernels exposed via **pybind11**.
* Modules:

  * `Model`: AD wrapper for objectives/constraints
  * `Regularizer`: Hessian conditioning and inertia correction
  * `SQPStepper`: globalization via trust region, line search, filter/funnel
  * `InteriorPointStepper`: slack/barrier-based primal-dual IPM
  * `DFOStepper`: derivative-free trust-region with L1 exact penalty
  * `KKT`: sparse KKT system assembly and factorization
  * Optional bindings: `qdldl_cpp`, `osqp_cpp`, `piqp_cpp`, `amdqg`, `simplex_core`, `l1core`