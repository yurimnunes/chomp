# nlp/wrapper.py
from __future__ import annotations

import builtins as _bi
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import ad as AD  # optional, your autodiff lib
except Exception:
    AD = None

Scalar = Union[float, int]
ArrayLike = Union[Scalar, np.ndarray]

# ---------- tiny helpers that preserve AD types ----------


def _as_obj_array(x) -> np.ndarray:
    """
    Return a 1-D numpy array with dtype=object, preserving AD Expression elements.
    Works for python lists, tuples, numpy arrays (numeric or object).
    """
    if isinstance(x, np.ndarray):
        if x.dtype == object:
            return x.ravel()
        return np.array(x, dtype=object).ravel()
    return np.array(x, dtype=object).ravel()


def _reshape_like(vec: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    return vec.reshape(shape)


def _is_scalar_like(v: Any) -> bool:
    shp = getattr(v, "shape", None)
    return (shp is None) or (shp == ())


def _sum_ad_aware(x):
    if AD is not None and hasattr(AD, "sum"):
        try:
            return AD.sum(x)
        except Exception:
            pass
    total = 0
    if hasattr(x, "flat"):
        for v in x.flat:
            total = total + v
        return total
    for v in x:
        total = total + v
    return total


def _sqrt_ad_aware(x):
    if AD is not None and hasattr(AD, "sqrt"):
        try:
            return AD.sqrt(x)
        except Exception:
            pass
    return np.sqrt(x)


def _abs_ad_aware(x):
    if AD is not None and hasattr(AD, "abs"):
        try:
            return AD.abs(x)
        except Exception:
            pass
    return np.abs(x)


def _dot_ad_aware(a, b):
    if getattr(a, "ndim", None) == 1 and getattr(b, "ndim", None) == 1:
        return _sum_ad_aware(a * b)
    return a @ b


# ---------- finite differences (standalone and Model wrappers) ----------


def grad_fd(
    f: Callable[[np.ndarray], float],
    x: ArrayLike,
    eps: float = 1e-6,
    scheme: str = "central",
) -> np.ndarray:
    """
    Finite-difference gradient.
    scheme in {"forward", "backward", "central"} (central recommended).
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.size
    g = np.zeros(n, dtype=float)

    if scheme not in {"forward", "backward", "central"}:
        raise ValueError("scheme must be forward|backward|central")

    fx = None if scheme == "central" else float(f(x))

    for i in range(n):
        e = np.zeros(n, dtype=float)
        # Scale step by magnitude of x_i to improve stability
        hi = eps * max(1.0, abs(x[i]))
        if scheme == "forward":
            e[i] = hi
            g[i] = (float(f(x + e)) - fx) / hi
        elif scheme == "backward":
            e[i] = hi
            g[i] = (fx - float(f(x - e))) / hi
        else:  # central
            e[i] = hi
            fph = float(f(x + e))
            fmh = float(f(x - e))
            g[i] = (fph - fmh) / (2.0 * hi)
    return g


def hess_fd(
    f: Callable[[np.ndarray], float],
    x: ArrayLike,
    eps: float = 1e-4,
) -> np.ndarray:
    """
    Symmetric finite-difference Hessian using central differences.
    O(n^2) evaluations; suitable for moderate n.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.size
    H = np.zeros((n, n), dtype=float)

    # Diagonal terms via second derivative
    for i in range(n):
        ei = np.zeros(n, dtype=float)
        hi = eps * max(1.0, abs(x[i]))
        ei[i] = hi
        fph = float(f(x + ei))
        fmh = float(f(x - ei))
        f0 = float(f(x))
        H[i, i] = (fph - 2.0 * f0 + fmh) / (hi * hi)

    # Off-diagonals via mixed partials
    for i in range(n):
        hi = eps * max(1.0, abs(x[i]))
        for j in range(i + 1, n):
            hj = eps * max(1.0, abs(x[j]))
            ei = np.zeros(n, dtype=float)
            ej = np.zeros(n, dtype=float)
            ei[i] = hi
            ej[j] = hj
            fpp = float(f(x + ei + ej))
            fpm = float(f(x + ei - ej))
            fmp = float(f(x - ei + ej))
            fmm = float(f(x - ei - ej))
            val = (fpp - fpm - fmp + fmm) / (4.0 * hi * hj)
            H[i, j] = H[j, i] = val
    return H


# ---------- Expr class with equation-style printing ----------


class Expr:
    __array_priority__ = 1000

    def __init__(
        self,
        fn: Callable[[Dict[str, Any]], Any],
        shape: Tuple[int, ...],
        repr_str: Optional[str] = None,
    ):
        self._fn = fn
        self.shape = shape
        self._repr = repr_str or f"Expr{shape}"

    # --- evaluation ---
    def eval(self, env: Dict[str, Any]) -> Any:
        return self._fn(env)

    # --- printing ---
    def __str__(self) -> str:
        return self._repr

    def __repr__(self) -> str:
        return f"Expr({self._repr}, shape={self.shape})"

    # --- constructors ---
    @staticmethod
    def constant(v: ArrayLike) -> "Expr":
        arr = np.array(v, dtype=object if AD else float)
        shape = () if arr.ndim == 0 else tuple(arr.shape)
        val = arr.item() if shape == () else arr
        # Short repr: numbers print as value; arrays show shape
        if shape == ():
            r = str(val)
        else:
            r = f"const{shape}"
        return Expr(lambda _env: val, shape, repr_str=r)

    @staticmethod
    def from_var(name: str, sl: slice, shape: Tuple[int, ...]) -> "Expr":
        """
        NOTE: `sl` is ignored. `env[name]` already holds the variable block with `shape`.
        Kept only for backward compatibility; will be deprecated.
        """
        def _fn(env):
            val = env[name]
            if shape == ():
                return val if _is_scalar_like(val) else np.array(val, dtype=object if AD else float).reshape(())
            arr = np.array(val, dtype=object if AD else float)
            if arr.shape != shape:
                arr = arr.reshape(shape)
            return arr
        return Expr(_fn, shape, repr_str=name)

    # ---- ops ----
    def _binop(self, other: Any, op: Callable[[Any, Any], Any], sym: str, bcast=True) -> "Expr":
        b = other if isinstance(other, Expr) else Expr.constant(other)

        def _fn(env):
            return op(self.eval(env), b.eval(env))

        out_shape = (
            self.shape
            if not bcast
            else np.broadcast_shapes(self.shape or (), b.shape or ())
        )
        return Expr(_fn, out_shape, repr_str=f"({self} {sym} {b})")

    def _unop(self, op: Callable[[Any], Any], sym_fmt: str) -> "Expr":
        return Expr(lambda env: op(self.eval(env)), self.shape, repr_str=sym_fmt.format(self=self))

    def __add__(self, other):
        return self._binop(other, lambda a, b: a + b, "+")

    def __radd__(self, other):
        return Expr.constant(other)._binop(self, lambda a, b: a + b, "+")

    def __sub__(self, other):
        return self._binop(other, lambda a, b: a - b, "-")

    def __rsub__(self, other):
        return Expr.constant(other)._binop(self, lambda a, b: a - b, "-")

    def __mul__(self, other):
        return self._binop(other, lambda a, b: a * b, "·")  # dot for scalar/broadcast mult

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._binop(other, lambda a, b: a / b, "/")

    def __rtruediv__(self, other):
        return Expr.constant(other)._binop(self, lambda a, b: a / b, "/")

    def __pow__(self, other):
        b = other if isinstance(other, Expr) else Expr.constant(other)
        return Expr(lambda env: self.eval(env) ** b.eval(env),
                    np.broadcast_shapes(self.shape or (), b.shape or ()),
                    repr_str=f"({self})**({b})")

    def __neg__(self):
        return self._unop(lambda a: -a, sym_fmt="-( {self} )")

    def __matmul__(self, other):
        b = other if isinstance(other, Expr) else Expr.constant(other)

        def _fn(env):
            return self.eval(env) @ b.eval(env)

        a_shape, b_shape = self.shape, b.shape
        if len(a_shape) == 2 and len(b_shape) == 2:
            out_shape = (a_shape[0], b_shape[1])
        elif len(a_shape) == 2 and b_shape == ():
            out_shape = a_shape
        elif a_shape == () and len(b_shape) == 2:
            out_shape = b_shape
        elif len(a_shape) == 1 and len(b_shape) == 1:
            out_shape = ()
        elif len(a_shape) == 2 and len(b_shape) == 1:
            out_shape = (a_shape[0],)
        elif len(a_shape) == 1 and len(b_shape) == 2:
            out_shape = (b_shape[1],)
        else:
            dummy_a = np.zeros(a_shape or (), dtype=float)
            dummy_b = np.zeros(b_shape or (), dtype=float)
            out_shape = (dummy_a @ dummy_b).shape
        return Expr(_fn, out_shape, repr_str=f"({self} @ {b})")

    # reductions
    def sum(self) -> "Expr":
        return Expr(lambda env: _sum_ad_aware(self.eval(env)), (),
                    repr_str=f"sum({self})")

    def dot(self, other: "Expr") -> "Expr":
        b = other if isinstance(other, Expr) else Expr.constant(other)
        return Expr(lambda env: _dot_ad_aware(self.eval(env), b.eval(env)), (),
                    repr_str=f"dot({self}, {b})")

    def T(self) -> "Expr":
        out_shape = (self.shape[1], self.shape[0]) if len(self.shape) == 2 else self.shape
        return Expr(lambda env: self.eval(env).T, out_shape, repr_str=f"({self})ᵀ")

    def norm2(self) -> "Expr":
        def _fn(env):
            a = self.eval(env)
            return _sqrt_ad_aware(_sum_ad_aware(a * a))
        return Expr(_fn, (), repr_str=f"||{self}||₂")

    def norm1(self) -> "Expr":
        return Expr(lambda env: _sum_ad_aware(_abs_ad_aware(self.eval(env))), (),
                    repr_str=f"||{self}||₁")

    def norm_inf(self) -> "Expr":
        def _fn(env):
            a = self.eval(env)
            if _is_scalar_like(a):
                return _abs_ad_aware(a)
            flat = a if not hasattr(a, "flat") else a.flat
            m = None
            for v in flat:
                av = _abs_ad_aware(v)
                m = av if (m is None) else (av if av > m else m)
            return 0.0 if m is None else m
        return Expr(_fn, (), repr_str=f"||{self}||_∞")

    def trace(self) -> "Expr":
        return Expr(lambda env: _sum_ad_aware(np.diag(self.eval(env))), (),
                    repr_str=f"tr({self})")

    # comparisons → constraints
    def __le__(self, rhs) -> "Constr":
        r = rhs if isinstance(rhs, Expr) else Expr.constant(rhs)
        return Constr(self, "<=", r)

    def __ge__(self, rhs) -> "Constr":
        r = rhs if isinstance(rhs, Expr) else Expr.constant(rhs)
        return Constr(self, ">=", r)

    def __eq__(self, rhs) -> "Constr":  # type: ignore[override]
        r = rhs if isinstance(rhs, Expr) else Expr.constant(rhs)
        return Constr(self, "==", r)

    # indexing
    def __getitem__(self, idx) -> "Expr":
        def _fn(env):
            return self.eval(env)[idx]
        out_shape = np.empty(self.shape or (), dtype=float)[idx].shape
        idx_repr = _index_repr(idx)
        return Expr(_fn, out_shape if out_shape else (), repr_str=f"{self}{idx_repr}")


# ufunc wrappers that return Expr (AD-op if available)
def _ufunc1(name: str, np_op: Callable, ad_op: Optional[Callable], sym_fmt: str):
    def wrapper(z: Union[Expr, ArrayLike]) -> Expr:
        op = ad_op or np_op
        if isinstance(z, Expr):
            return Expr(lambda env: op(z.eval(env)), z.shape, repr_str=sym_fmt.format(z=z))
        return Expr.constant(z)._unop(op, sym_fmt=sym_fmt)
    wrapper.__name__ = name
    return wrapper


sin  = _ufunc1("sin",  np.sin,  getattr(AD, "sin",  None) if AD else None, sym_fmt="sin({z})")
cos  = _ufunc1("cos",  np.cos,  getattr(AD, "cos",  None) if AD else None, sym_fmt="cos({z})")
exp  = _ufunc1("exp",  np.exp,  getattr(AD, "exp",  None) if AD else None, sym_fmt="exp({z})")
log  = _ufunc1("log",  np.log,  getattr(AD, "log",  None) if AD else None, sym_fmt="log({z})")
sqrt = _ufunc1("sqrt", np.sqrt, getattr(AD, "sqrt", None) if AD else None, sym_fmt="sqrt({z})")
tanh = _ufunc1("tanh", np.tanh, getattr(AD, "tanh", None) if AD else None, sym_fmt="tanh({z})")


# ---------- variables & constraints ----------

class Param(Expr):
    """
    A named parameter placeholder that is *not* part of the decision vector.
    Its value lives in Model._param_values[name] and can be updated at runtime.
    """

    def __init__(self, name: str, shape: Tuple[int, ...]):
        self.name = name
        self.shape = shape

        def _fn(env):
            val = env[name]
            if shape == ():
                return val if _is_scalar_like(val) else np.array(val, dtype=object if AD else float).reshape(())
            arr = np.array(val, dtype=object if AD else float)
            if arr.shape != shape:
                arr = arr.reshape(shape)
            return arr

        super().__init__(_fn, shape, repr_str=name)


class Var(Expr):
    def __init__(
        self,
        name: str,
        offset: int,
        shape: Tuple[int, ...],
        lb: Optional[ArrayLike],
        ub: Optional[ArrayLike],
    ):
        self.name = name
        self.offset = offset
        self.size = int(np.prod(shape)) if shape else 1
        self.lb = None if lb is None else np.array(lb, dtype=object if AD else float).reshape(shape)
        self.ub = None if ub is None else np.array(ub, dtype=object if AD else float).reshape(shape)
        sl = slice(offset, offset + self.size)
        super().__init__(Expr.from_var(name, sl, shape)._fn, shape, repr_str=name)


class Constr:
    def __init__(self, lhs: Expr, sense: str, rhs: Expr):
        if sense not in ("<=", ">=", "=="):
            raise ValueError("sense must be one of <=, >=, ==")
        self.lhs = lhs
        self.rhs = rhs
        self.sense = sense
        self.shape = np.broadcast_shapes(lhs.shape or (), rhs.shape or ())

    def __str__(self) -> str:
        # Equation-style constraint printing
        if self.shape == ():
            return f"{self.lhs} {self.sense} {self.rhs}"
        return f"{self.lhs} {self.sense} {self.rhs}  # shape={self.shape}"

    def __repr__(self) -> str:
        return f"Constr({self.lhs} {self.sense} {self.rhs}, shape={self.shape})"


# ---------- model ----------

class Model:
    def __init__(self, name: str = "nlp"):
        self.name = name
        self._vars: Dict[str, Var] = {}
        self._order: List[str] = []
        self._offset = 0
        self._constrs: List[Constr] = []
        self._objective: Optional[Expr] = None
        self._sense: str = "min"
        self._x0_parts: Dict[str, np.ndarray] = {}
        self._params: Dict[str, Param] = {}
        self._param_values: Dict[str, Any] = {}

        # for convenience after build()
        self._built: Dict[str, Any] = {}
        self._objective_repr: Optional[str] = None
        self._constraints_repr: List[str] = []

    # ----- Parameters API -----
    def add_param(
        self,
        name: str,
        shape: Union[int, Tuple[int, ...]] = (),
        value: Optional[ArrayLike] = None,
    ) -> Param:
        if name in self._vars or name in self._params:
            raise ValueError(f"name '{name}' already used as var/param")
        shape = _normalize_shape(shape)
        p = Param(name, shape)
        self._params[name] = p
        if value is not None:
            self.set_param(name, value)
        return p

    def set_param(self, name: str, value: ArrayLike):
        if name not in self._params:
            raise ValueError(f"unknown param '{name}'")
        shp = self._params[name].shape
        arr = np.array(value, dtype=object if AD else float)
        if shp != () and tuple(arr.shape) != shp:
            raise ValueError(f"param '{name}' expects shape {shp}, got {arr.shape}")
        self._param_values[name] = arr.item() if shp == () else arr.reshape(shp)

    def set_params(self, mapping: Optional[Dict[str, ArrayLike]] = None, /, **kwargs):
        if mapping:
            for k, v in mapping.items():
                self.set_param(k, v)
        for k, v in kwargs.items():
            self.set_param(k, v)

    def get_param(self, name: str) -> Any:
        if name not in self._params:
            raise ValueError(f"unknown param '{name}'")
        return self._param_values.get(name, None)

    # ----- Variables / objective / constraints -----
    def add_var(
        self,
        name: str,
        shape: Union[int, Tuple[int, ...]] = (),
        lb: Optional[ArrayLike] = None,
        ub: Optional[ArrayLike] = None,
        x0: Optional[ArrayLike] = None,
    ) -> Var:
        if name in self._vars:
            raise ValueError(f"variable '{name}' already exists")
        shape = _normalize_shape(shape)
        v = Var(name, self._offset, shape, lb, ub)
        self._vars[name] = v
        self._order.append(name)
        self._offset += v.size
        if x0 is not None:
            self._x0_parts[name] = np.array(x0, dtype=float).reshape(shape)
        return v

    def set_initial(self, name: str, x0: ArrayLike):
        if name not in self._vars:
            raise ValueError(f"unknown var '{name}'")
        v = self._vars[name]
        self._x0_parts[name] = np.array(x0, dtype=float).reshape(v.shape)

    def minimize(self, expr: Expr):
        self._objective = expr
        self._sense = "min"
        self._objective_repr = str(expr)

    def maximize(self, expr: Expr):
        self._objective = -expr
        self._sense = "max"
        self._objective_repr = f"-({expr})"

    def add_constr(self, c: Constr):
        self._constrs.append(c)
        self._constraints_repr.append(str(c))

    # ----- Build problem functions -----
    def build(self):
        if self._objective is None:
            raise ValueError("no objective set")
        n = _bi.sum(v.size for v in self._vars.values())

        def pack_env(x_in) -> Dict[str, Any]:
            vec = (np.asarray(x_in, dtype=float).ravel() if AD is None else _as_obj_array(x_in))
            env: Dict[str, Any] = dict(self._param_values)
            for name in self._order:
                v = self._vars[name]
                sl = slice(v.offset, v.offset + v.size)
                if v.shape == ():
                    env[name] = vec[sl][0]
                else:
                    env[name] = _reshape_like(vec[sl], v.shape)
            return env

        def f(x_in):
            env = pack_env(x_in)
            val = self._objective.eval(env)
            return val if _is_scalar_like(val) else _sum_ad_aware(val)

        # bounds as constraints
        all_cs = self._constrs

        c_ineq: List[Callable[[np.ndarray], Any]] = []
        c_eq: List[Callable[[np.ndarray], Any]] = []
        # --- inside Model.build(), replace make_scalar_funcs with this version ---
        def make_scalar_funcs(c: Constr):
            """
            Emit scalar callables with the convention:
                - Inequality:  ci(x) <= 0
                - Equality:    he(x)  = 0
            """
            if c.sense == "<=":
                # lhs <= rhs  →  (lhs - rhs) <= 0
                def h(env): return c.lhs.eval(env) - c.rhs.eval(env)
                target = c_ineq
            elif c.sense == ">=":
                # lhs >= rhs  →  (rhs - lhs) <= 0
                def h(env): return c.rhs.eval(env) - c.lhs.eval(env)
                target = c_ineq
            else:  # "=="
                # lhs == rhs  →  (lhs - rhs) == 0
                def h(env): return c.lhs.eval(env) - c.rhs.eval(env)
                target = c_eq

            # Probe to determine scalar vs array and vectorize cleanly
            probe_env = pack_env(np.zeros(n, dtype=float))
            sample = h(probe_env)

            if _is_scalar_like(sample):
                def g_scalar(x_in):
                    env = pack_env(x_in)
                    val = h(env)
                    return val if _is_scalar_like(val) else _sum_ad_aware(val)
                target.append(g_scalar)
            else:
                shp = getattr(sample, "shape", None) or np.array(sample).shape
                for idx in np.ndindex(shp):
                    def g_idx(x_in, idx=idx):
                        env = pack_env(x_in)
                        val = h(env)
                        arr = (val if hasattr(val, "__getitem__")
                            else np.array(val, dtype=object if AD else float))
                        return arr[idx]
                    target.append(g_idx)

        for c in all_cs:
            make_scalar_funcs(c)

        # -------- stack global lb/ub aligned with var offsets --------
        lb_vec = np.full(n, -np.inf, dtype=float)
        ub_vec = np.full(n,  +np.inf, dtype=float)

        for name in self._order:
            v = self._vars[name]
            off = v.offset
            if v.lb is not None:
                lb_vec[off : off + v.size] = np.asarray(v.lb, dtype=float).reshape(-1)
            if v.ub is not None:
                ub_vec[off : off + v.size] = np.asarray(v.ub, dtype=float).reshape(-1)

        # -------- Build x0 numeric vector (no shifts) --------
        x0 = np.zeros(n, dtype=float)
        for name in self._order:
            v = self._vars[name]
            if name in self._x0_parts:
                part = self._x0_parts[name].reshape(-1)
            else:
                if v.lb is not None and v.ub is not None:
                    part = (np.array(v.lb, dtype=float) + np.array(v.ub, dtype=float)).reshape(-1) * 0.5
                elif v.lb is not None:
                    part = np.array(v.lb, dtype=float).reshape(-1) + 0.1
                elif v.ub is not None:
                    part = np.array(v.ub, dtype=float).reshape(-1) - 0.1
                else:
                    part = np.zeros(v.size, dtype=float)
            x0[v.offset:v.offset+v.size] = part

        # store for convenience
        self._built = {
            "f": f,
            "c_ineq": c_ineq,
            "c_eq": c_eq,
            "x0": x0,
            "lb": lb_vec,
            "ub": ub_vec,
        }
        return f, c_ineq, c_eq, x0, lb_vec, ub_vec

    # ----- Convenience after build() -----
    def attach_built(self, f=None, c_ineq=None, c_eq=None, x0=None, lb=None, ub=None):
        """
        If you build externally, you can stash results back on the model.
        """
        if f is not None:
            self._built["f"] = f
        if c_ineq is not None:
            self._built["c_ineq"] = c_ineq
        if c_eq is not None:
            self._built["c_eq"] = c_eq
        if x0 is not None:
            self._built["x0"] = np.asarray(x0, dtype=float)
        if lb is not None:
            self._built["lb"] = np.asarray(lb, dtype=float)
        if ub is not None:
            self._built["ub"] = np.asarray(ub, dtype=float)

    def value(self, x: ArrayLike) -> float:
        f = self._built.get("f", None)
        if f is None:
            raise RuntimeError("call build() first")
        return float(f(np.asarray(x, dtype=float)))

    def grad_fd(self, x: ArrayLike, eps: float = 1e-6, scheme: str = "central") -> np.ndarray:
        f = self._built.get("f", None)
        if f is None:
            raise RuntimeError("call build() first")
        return grad_fd(f, x, eps=eps, scheme=scheme)

    def hess_fd(self, x: ArrayLike, eps: float = 1e-4) -> np.ndarray:
        f = self._built.get("f", None)
        if f is None:
            raise RuntimeError("call build() first")
        return hess_fd(f, x, eps=eps)

    # ----- Save / Load (JSON) -----
    def to_dict(self) -> Dict[str, Any]:
        def arr_or_none(a):
            if a is None:
                return None
            aa = np.asarray(a)
            return {"shape": list(aa.shape), "data": aa.tolist()}

        vars_pack = []
        for name in self._order:
            v = self._vars[name]
            x0p = self._x0_parts.get(name, None)
            vars_pack.append(
                {
                    "name": name,
                    "shape": list(v.shape) if v.shape else [],
                    "lb": None if v.lb is None else np.asarray(v.lb, dtype=float).tolist(),
                    "ub": None if v.ub is None else np.asarray(v.ub, dtype=float).tolist(),
                    "x0": None if x0p is None else np.asarray(x0p, dtype=float).tolist(),
                }
            )
        params_pack = []
        for pname, p in self._params.items():
            val = self._param_values.get(pname, None)
            params_pack.append(
                {
                    "name": pname,
                    "shape": list(p.shape) if p.shape else [],
                    "value": None if val is None else np.asarray(val, dtype=float).tolist(),
                }
            )
        d = {
            "name": self.name,
            "sense": self._sense,
            "vars": vars_pack,
            "params": params_pack,
            "objective_repr": self._objective_repr,
            "constraints_repr": self._constraints_repr,
        }
        # also stash built vectors if available
        if self._built:
            d["built"] = {
                k: (np.asarray(v, dtype=float).tolist() if isinstance(v, np.ndarray) else None)
                for k, v in self._built.items()
                if k in {"x0", "lb", "ub"}
            }
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Model":
        m = Model(d.get("name", "nlp"))
        # vars
        for vinfo in d.get("vars", []):
            name = vinfo["name"]
            shape = tuple(vinfo.get("shape", []))
            lb = vinfo.get("lb", None)
            ub = vinfo.get("ub", None)
            x0 = vinfo.get("x0", None)
            m.add_var(name, shape=shape, lb=lb, ub=ub, x0=x0)
        # params
        for pinfo in d.get("params", []):
            pname = pinfo["name"]
            shape = tuple(pinfo.get("shape", []))
            val = pinfo.get("value", None)
            p = m.add_param(pname, shape=shape)
            if val is not None:
                m.set_param(pname, np.asarray(val))
        # sense and reprs (note: we do not reconstruct executable expressions)
        m._sense = d.get("sense", "min")
        m._objective_repr = d.get("objective_repr", None)
        m._constraints_repr = list(d.get("constraints_repr", []))
        # built vectors if present
        if "built" in d:
            b = d["built"]
            if b.get("x0") is not None:
                m._built["x0"] = np.asarray(b["x0"], dtype=float)
            if b.get("lb") is not None:
                m._built["lb"] = np.asarray(b["lb"], dtype=float)
            if b.get("ub") is not None:
                m._built["ub"] = np.asarray(b["ub"], dtype=float)
        return m

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(path: str) -> "Model":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return Model.from_dict(d)


# ---------- convenience API ----------

def esum(expr: Expr) -> Expr:
    return expr.sum()

def dot(a: Expr, b: Expr) -> Expr:
    return a.dot(b)

def norm2(a: Expr) -> Expr:
    return a.norm2()

def norm1(a: Expr) -> Expr:
    return a.norm1()

def norm_inf(a: Expr) -> Expr:
    return a.norm_inf()

def _normalize_shape(shape: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
    if shape == () or shape == 0:
        return ()
    if isinstance(shape, int):
        return (shape,)
    return tuple(int(s) for s in shape)


# ---------- small utils for printing ----------

def _index_repr(idx) -> str:
    # Build a Python-like index string for symbolic printing
    if isinstance(idx, tuple):
        parts = []
        for it in idx:
            parts.append(_single_index_repr(it))
        return "[" + ", ".join(parts) + "]"
    return "[" + _single_index_repr(idx) + "]"

def _single_index_repr(it) -> str:
    if isinstance(it, slice):
        a = "" if it.start is None else str(it.start)
        b = "" if it.stop  is None else str(it.stop)
        c = "" if it.step  is None else str(it.step)
        if c:
            return f"{a}:{b}:{c}"
        return f"{a}:{b}"
    return str(it)
