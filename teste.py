# test_with_wrapper.py
import math
import numpy as np
import os, sys
# get the parent directory of the current working dir
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from nlp.wrapper import Model, norm2, esum, cos  # and AD-aware functions if you prefer
from nlp.nlp import NLPSolver, SQPConfig  # your solver

# Define the Branin objective using wrapper expressions
def build_branin_model(use_ineq: bool = True, R: float = 60.0, mode: str = "auto"):
    m = Model("branin")

    # Two decision variables x1, x2 as a vector (shape=2)
    x = m.add_var("x", shape=2)
    x1 = x[0]
    x2 = x[1]

    # Branin constants
    a = 1.0
    b = 5.1 / (4.0 * math.pi ** 2)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)

    # Objective (scalar Expr). Use cos from wrapper (AD-aware).
    f = a * (x2 - b * (x1 ** 2) + c * x1 - r) ** 2 + s * (1 - t) * cos(x1) + s
    m.minimize(f)

    # Optional inequality: x1^2 + x2^2 - R^2 <= 0
    if use_ineq:
        m.add_constr(x1**2 + x2**2 - (R**2) <= 0.0)

    # Build into callables for NLPSolver
    f_fun, c_ineq, c_eq, x0 = m.build()
    print(f_fun, c_ineq, c_eq, x0)
    x0 = np.array([ -3.0, 12.0 ])  # A common starting point
    
    # Configure solver
    cfg = SQPConfig()
    cfg.mode = mode
    cfg.use_trust_region = True
    cfg.tr_delta0 = 1.0
    cfg.ip_mu_init = 1e-1
    cfg.ip_tau = 0.995
    cfg.ip_sigma_power = 3
    cfg.ip_switch_theta = 1e-5
    cfg.ip_switch_mu = 1e-8
    cfg.ip_stall_iters = 5

    solver = NLPSolver(f=f_fun, c_ineq=c_ineq, c_eq=c_eq, x0=x0, config=cfg)
    return solver

if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    solver = build_branin_model(use_ineq=True, R=60.0, mode="ip")
    x_star, hist = solver.solve(max_iter=150, tol=1e-8, verbose=True)
    print("x* =", x_star, "f* =", float(solver.model.f(x_star)))
