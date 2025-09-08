# nlp/wrapper.py
from __future__ import annotations

import builtins as _bi
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
        # keep as-is if already object array
        if x.dtype == object:
            return x.ravel()
        return np.array(x, dtype=object).ravel()
    return np.array(x, dtype=object).ravel()

def _reshape_like(vec_obj: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Reshape a 1-D object array to given shape (also object array)."""
    return vec_obj.reshape(shape)

def _is_scalar_like(v: Any) -> bool:
    """True if v behaves like a scalar for our purposes (no 'shape' or empty shape)."""
    shp = getattr(v, "shape", None)
    return (shp is None) or (shp == ())

def _sum_ad_aware(x):
    if AD is not None and hasattr(AD, "sum"):
        try:
            return AD.sum(x)
        except Exception:
            pass
    # Python fallback: fold with '+', keeps AD nodes if elements are AD Expressions
    total = 0
    # If x is numpy array (maybe dtype=object), iterate flat
    if hasattr(x, "flat"):
        for v in x.flat:
            total = total + v
        return total
    # Generic iterable
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
    # Handle 1-D dot explicitly
    if getattr(a, "ndim", None) == 1 and getattr(b, "ndim", None) == 1:
        return _sum_ad_aware(a * b)
    # Fall back to @ which works with numpy object arrays of AD nodes
    return a @ b

# ---------- Expr class ----------

class Expr:
    __array_priority__ = 1000

    def __init__(self, fn: Callable[[Dict[str, Any]], Any], shape: Tuple[int, ...]):
        self._fn   = fn
        self.shape = shape

    def eval(self, env: Dict[str, Any]) -> Any:
        return self._fn(env)

    @staticmethod
    def constant(v: ArrayLike) -> "Expr":
        # store as numpy array to provide shape info; scalars -> shape ()
        vv = np.array(v, dtype=object) if np.ndim(v) else v
        shape = () if np.ndim(v) == 0 else np.array(v).shape
        return Expr(lambda _env: vv, shape)

    @staticmethod
    def from_var(name: str, sl: slice, shape: Tuple[int, ...]) -> "Expr":
        def _fn(env):
            flat = _as_obj_array(env[name])
            return _reshape_like(flat[sl], shape) if shape != () else flat[sl].item()
        return Expr(_fn, shape)

    # ---- ops ----
    def _binop(self, other: Any, op: Callable[[Any, Any], Any], bcast=True) -> "Expr":
        b = other if isinstance(other, Expr) else Expr.constant(other)
        def _fn(env):
            return op(self.eval(env), b.eval(env))
        # broadcast shape via numpy for convenience (shape only)
        out_shape = self.shape if not bcast else np.broadcast_shapes(
            self.shape if self.shape else (),
            b.shape if b.shape else ()
        )
        return Expr(_fn, out_shape)

    def _unop(self, op: Callable[[Any], Any]) -> "Expr":
        return Expr(lambda env: op(self.eval(env)), self.shape)

    def __add__(self, other):  return self._binop(other, lambda a,b: a + b)
    def __radd__(self, other): return self.__add__(other)
    def __sub__(self, other):  return self._binop(other, lambda a,b: a - b)
    def __rsub__(self, other): return Expr.constant(other)._binop(self, lambda a,b: a - b)
    def __mul__(self, other):  return self._binop(other, lambda a,b: a * b)
    def __rmul__(self, other): return self.__mul__(other)
    def __truediv__(self, other):  return self._binop(other, lambda a,b: a / b)
    def __rtruediv__(self, other): return Expr.constant(other)._binop(self, lambda a,b: a / b)
    def __pow__(self, other):  return self._binop(other, lambda a,b: a ** b)
    def __neg__(self):         return self._unop(lambda a: -a)
    def __matmul__(self, other):
        b = other if isinstance(other, Expr) else Expr.constant(other)
        def _fn(env): return self.eval(env) @ b.eval(env)
        a_shape = self.shape; b_shape = b.shape
        out_shape = ()
        if len(a_shape)==2 and len(b_shape)==2:
            out_shape = (a_shape[0], b_shape[1])
        return Expr(_fn, out_shape)

    # reductions
    def sum(self) -> "Expr":
        return Expr(lambda env: _sum_ad_aware(self.eval(env)), ())

    def dot(self, other: "Expr") -> "Expr":
        b = other if isinstance(other, Expr) else Expr.constant(other)
        return Expr(lambda env: _dot_ad_aware(self.eval(env), b.eval(env)), ())

    def T(self) -> "Expr":
        out_shape = (self.shape[1], self.shape[0]) if len(self.shape)==2 else self.shape
        return Expr(lambda env: self.eval(env).T, out_shape)

    def norm2(self) -> "Expr":
        return Expr(lambda env: _sqrt_ad_aware(_sum_ad_aware(self.eval(env)*self.eval(env))), ())

    def norm1(self) -> "Expr":
        return Expr(lambda env: _sum_ad_aware(_abs_ad_aware(self.eval(env))), ())

    def norm_inf(self) -> "Expr":
        # Simple fallback: max over flattened array
        def _fn(env):
            a = self.eval(env)
            flat = a if not hasattr(a, "flat") else list(a.flat)
            return _bi.max(_abs_ad_aware(v) for v in flat)
        return Expr(_fn, ())

    def trace(self) -> "Expr":
        return Expr(lambda env: _sum_ad_aware(np.diag(self.eval(env))), ())

    # comparisons
    def __le__(self, rhs) -> "Constr": return Constr(self, "<=", rhs if isinstance(rhs, Expr) else Expr.constant(rhs))
    def __ge__(self, rhs) -> "Constr": return Constr(self, ">=", rhs if isinstance(rhs, Expr) else Expr.constant(rhs))
    def __eq__(self, rhs) -> "Constr": return Constr(self, "==", rhs if isinstance(rhs, Expr) else Expr.constant(rhs))  # type: ignore[override]

    # indexing
    def __getitem__(self, idx) -> "Expr":
        def _fn(env): 
            return self.eval(env)[idx]
        # use a numeric dummy so indexing returns a numpy array/scalar, never None
        if self.shape:
            out_shape = np.zeros(self.shape, dtype=float)[idx].shape
        else:
            out_shape = ()
        return Expr(_fn, out_shape)
# ufunc wrappers that return Expr (AD-op if available)
def _ufunc1(name: str, np_op: Callable, ad_op: Optional[Callable]):
    def wrapper(z: Union[Expr, ArrayLike]) -> Expr:
        op = ad_op or np_op
        if isinstance(z, Expr):
            return z._unop(op)
        return Expr.constant(z)._unop(op)
    wrapper.__name__ = name
    return wrapper

sin  = _ufunc1("sin",  np.sin,  getattr(AD, "sin",  None) if AD else None)
cos  = _ufunc1("cos",  np.cos,  getattr(AD, "cos",  None) if AD else None)
exp  = _ufunc1("exp",  np.exp,  getattr(AD, "exp",  None) if AD else None)
log  = _ufunc1("log",  np.log,  getattr(AD, "log",  None) if AD else None)
sqrt = _ufunc1("sqrt", np.sqrt, getattr(AD, "sqrt", None) if AD else None)
tanh = _ufunc1("tanh", np.tanh, getattr(AD, "tanh", None) if AD else None)

# ---------- variables & constraints ----------

class Var(Expr):
    def __init__(self, name: str, offset: int, shape: Tuple[int, ...],
                 lb: Optional[ArrayLike], ub: Optional[ArrayLike]):
        self.name   = name
        self.offset = offset
        self.size   = int(np.prod(shape)) if shape else 1
        self.lb     = None if lb is None else np.array(lb, dtype=object).reshape(shape)
        self.ub     = None if ub is None else np.array(ub, dtype=object).reshape(shape)
        sl = slice(offset, offset + self.size)
        super().__init__(Expr.from_var(name, sl, shape)._fn, shape)

class Constr:
    def __init__(self, lhs: Expr, sense: str, rhs: Expr):
        if sense not in ("<=", ">=", "=="):
            raise ValueError("sense must be one of <=, >=, ==")
        self.lhs   = lhs
        self.rhs   = rhs
        self.sense = sense
        self.shape = np.broadcast_shapes(lhs.shape if lhs.shape else (), rhs.shape if rhs.shape else ())

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

    def add_var(self, name: str, shape: Union[int, Tuple[int, ...]] = (),
                lb: Optional[ArrayLike] = None, ub: Optional[ArrayLike] = None,
                x0: Optional[ArrayLike] = None) -> Var:
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

    def minimize(self, expr: Expr): self._objective = expr; self._sense = "min"
    def maximize(self, expr: Expr): self._objective = -expr; self._sense = "max"
    def add_constr(self, c: Constr): self._constrs.append(c)

    def build(self):
        if self._objective is None:
            raise ValueError("no objective set")
        n = _bi.sum(v.size for v in self._vars.values())

        def pack_env(x_in) -> Dict[str, Any]:
            vec = _as_obj_array(x_in)  # preserves AD nodes, handles lists
            env: Dict[str, Any] = {}
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
            # objective must be scalar; if array-like, reduce with AD-aware sum
            return val if _is_scalar_like(val) else _sum_ad_aware(val)

        # bounds as constraints
        bound_cs: List[Constr] = []
        for v in self._vars.values():
            if v.lb is not None: bound_cs.append(Constr(v, ">=", Expr.constant(v.lb)))
            if v.ub is not None: bound_cs.append(Constr(v, "<=", Expr.constant(v.ub)))
        all_cs = self._constrs + bound_cs

        c_ineq: List[Callable[[np.ndarray], Any]] = []
        c_eq:   List[Callable[[np.ndarray], Any]] = []

        def make_scalar_funcs(c: Constr):
            if c.sense == "<=":
                h = lambda env: c.lhs.eval(env) - c.rhs.eval(env); target = c_ineq
            elif c.sense == ">=":
                h = lambda env: c.rhs.eval(env) - c.lhs.eval(env); target = c_ineq
            else:
                h = lambda env: c.lhs.eval(env) - c.rhs.eval(env); target = c_eq

            # Probe on zeros (numeric) only to figure out shape (safe; not used by AD)
            probe_env: Dict[str, Any] = {}
            for name in self._order:
                v = self._vars[name]
                probe_env[name] = 0.0 if v.shape == () else np.zeros(v.shape, dtype=float)
            sample = h(probe_env)

            if _is_scalar_like(sample):
                def g_scalar(x_in):
                    env = pack_env(x_in)
                    val = h(env)
                    return val if _is_scalar_like(val) else _sum_ad_aware(val)
                target.append(g_scalar)
            else:
                # explode into elementwise scalar constraints
                shp = getattr(sample, "shape", None) or np.array(sample).shape
                for idx in np.ndindex(shp):
                    def g_idx(x_in, idx=idx):
                        env = pack_env(x_in)
                        val = h(env)
                        # index without forcing to numpy (supports object arrays / lists)
                        return (val[idx] if hasattr(val, "__getitem__")
                                else np.array(val, dtype=object)[idx])
                    target.append(g_idx)

        for c in all_cs:
            make_scalar_funcs(c)

        # Build x0 numeric vector
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

        return f, c_ineq, c_eq, x0

# ---------- convenience API ----------

def esum(expr: Expr) -> Expr:       return expr.sum()
def dot(a: Expr, b: Expr) -> Expr:  return a.dot(b)
def norm2(a: Expr) -> Expr:         return a.norm2()
def norm1(a: Expr) -> Expr:         return a.norm1()
def norm_inf(a: Expr) -> Expr:      return a.norm_inf()

def _normalize_shape(shape: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
    if shape == () or shape == 0 or shape == 1: return ()
    if isinstance(shape, int): return (shape,)
    return tuple(int(s) for s in shape)
