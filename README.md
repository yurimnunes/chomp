# capyAD 🦫

**capyAD** is a lightweight, C++/Python automatic differentiation library built around **expression graphs** with full support for gradients, Hessians, and efficient Hessian–vector products (HVP).
It is designed for **optimization, scientific computing, and machine learning research** — offering clean abstractions, high performance, and extensibility.

---

## ✨ Features

* **Expression Graph Engine**

  * Nodes for variables, constants, unary/binary/n-ary operations
  * Cycle detection, deduplication, and canonical variable mapping
* **Automatic Differentiation**

  * Reverse-mode (backpropagation) for efficient gradients
  * Forward-over-reverse for Hessian–vector products
  * Full dense Hessian reconstruction
* **Epoch-based Caching**

  * Safe reuse of node values/gradients across calls
  * Avoids recomputation with robust invalidation
* **Introspection Tools**

  * Expression pretty-printing
  * Graph visualization utilities (`printTree`, SVG diagrams)
* **Python Bindings**

  * Seamless integration into scientific workflows
  * `gradient`, `hessian`, and `hvp` APIs from Python

---

## 📦 Installation

```bash
git clone https://github.com/your-org/capyAD.git
cd capyAD
pip install .
```

Requirements:

* Python ≥ 3.9
* C++17 compiler
* `pybind11`, `numpy`

---

## 🧩 Expression Graph

Expressions are represented as **graphs**, not trees. Shared subexpressions are automatically unified.

Example:

```python
import capyAD as ad

x = ad.Var("x")
y = ad.Var("y")

expr = ad.sin(x) + x * y
print(expr)  # (sin(x) + (x * y))
```

**Graph structure:**

![Expression Graph](./docs/expression_graph.svg)

---

## 🔢 Differentiation

### Gradient

```python
f = ad.sin(x) + x * y
grad = ad.gradient(f, {"x": 1.0, "y": 2.0})
print(grad)  # {'x': cos(x) + y, 'y': x}
```

### Hessian–Vector Product (HVP)

Efficiently compute $H v$ without building the full Hessian:

```python
Hv = ad.hvp(f, {"x": 1.0, "y": 2.0}, v=[1.0, 0.5])
print(Hv)  # vector of size = number of variables
```

Dataflow:

![HVP Flow](./docs/hvp_flow.svg)

### Dense Hessian

```python
H = ad.hessian(f, {"x": 1.0, "y": 2.0})
print(H)  # full 2x2 matrix
```

---

## ⚙️ Internals

* **Forward Pass**

  * Computes node values (`value`) and tangents (`dot`)
* **Reverse Pass**

  * Propagates gradients (`gradient`) and second-order terms (`grad_dot`)
* **Epoch Counters**

  * Ensure fresh state without zeroing arrays
  * `cur_val_epoch`, `cur_grad_epoch`, `cur_dot_epoch`, `cur_gdot_epoch`

---

## 🚀 Roadmap

* [ ] CUDA acceleration
* [ ] Better visualization (interactive graph viewers)
* [ ] JAX/PyTorch interop
* [ ] Sparse Hessian support

---

## 📖 Citation

If you use **capyAD** in research, please cite:

```bibtex
@misc{capyad2025,
  title  = {capyAD: A Lightweight Expression Graph AD Library},
  author = {Laio O. Seman},
  year   = {2025},
  url    = {https://github.com/lseman/capyAD}
}
```

---

## 🦫 Why "capyAD"?

Because capybaras are **friendly, efficient, and love groups** — just like our expression graphs.
