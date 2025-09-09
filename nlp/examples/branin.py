# test_nlp_hybrid.py
# Minimal tests of the hybrid NLPSolver on Rosenbrock and Branin functions.
# Assumes nlp_hybrid.py (from previous message) and sqp_aux.py are importable.

import math

# add parent directory to path
import os
import sys

import numpy as np

# get the parent directory of the current working dir
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import ad as AD

from nlp.nlp import NLPSolver, SQPConfig

# ---------------------------
# Test problems
# ---------------------------

def rosenbrock(x: np.ndarray, a: float = 1.0, b: float = 100.0) -> float:
    """
    Rosenbrock function in 2D:
        f(x, y) = (a - x)^2 + b (y - x^2)^2
    Global min at (x, y) = (a, a^2), f = 0
    """
    x1, x2 = x
    return (a - x1) ** 2 + b * (x2 - x1 ** 2) ** 2


def branin(x: np.ndarray) -> float:
    """
    Branin (2D) on domain x1 in [-5, 10], x2 in [0, 15].
    Standard form (global minima ~ 0.397887 at three points).
    """
    x1, x2 = x
    a = 1.0
    b = 5.1 / (4.0 * math.pi ** 2)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)
    return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * AD.cos(x1) + s


# Optional: an inequality to exercise the interior-point slack path.
# For example, keep inside a big circle:  (x1)^2 + (x2)^2 - R^2 <= 0
def circle_ineq(R: float):
    def g(x: np.ndarray) -> float:
        return x[0] ** 2 + x[1] ** 2 - R ** 2
    return g


# ---------------------------
# Utility to run a single solve
# ---------------------------

def run_solve(name: str,
              f,
              x0: np.ndarray,
              mode: str = "auto",
              use_ineq=False,
              R: float = 50.0,
              max_iter: int = 150):
    """
    name: label for the test
    f: objective function f(x)
    x0: starting point
    mode: "auto"|"ip"|"sqp"
    use_ineq: if True, add circle inequality g(x) <= 0 to use slacks/barrier
    """
    print("=" * 80)
    print(f"{name}: mode={mode} x0={x0}")
    cfg = SQPConfig()
    cfg.mode = mode

    # Gentle defaults for testing
    cfg.tol_stat = 1e-5
    cfg.tol_feas = 1e-5
    cfg.tol_comp = 1e-5

    # Trust-region settings (kept conservative)
    cfg.use_trust_region = True
    cfg.tr_norm_type = "2"
    cfg.tr_delta0 = 1.0

    # Hybrid IP settings (fraction-to-boundary, etc.)
    cfg.ip_mu_init = 1e-1
    cfg.ip_tau = 0.995
    cfg.ip_sigma_power = 3
    cfg.ip_switch_theta = 1e-5
    cfg.ip_switch_mu = 1e-8
    cfg.ip_stall_iters = 5

    # Quiet PIQP by default
    cfg.piqp_verbose = False

    # Build constraint lists
    c_ineq = [circle_ineq(R)] if use_ineq else []
    c_eq = []

    solver = NLPSolver(f=f, c_ineq=c_ineq, c_eq=c_eq, x0=np.array(x0, dtype=float), config=cfg)
    x_star, hist = solver.solve(max_iter=max_iter, tol=1e-8, verbose=True)

    f_star = f(x_star)
    print(f"-> {name} DONE. x* = {x_star}, f* = {f_star:.9f}")
    print("-" * 80)
    return x_star, f_star, hist


# ---------------------------
# Main: run a few scenarios
# ---------------------------

if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    # 1) Rosenbrock (unconstrained)
    # run_solve("Rosenbrock (auto)", rosenbrock, x0=np.array([-1.2, 1.0]), mode="dfo_al")
    # run_solve("Rosenbrock (sqp)",  rosenbrock, x0=np.array([ 2.0,  2.0]), mode="sqp")
    # run_solve("Rosenbrock (ip)",   rosenbrock, x0=np.array([-1.5, 1.5]), mode="sqp")

    # 2) Branin (unconstrained), multiple starts
    starts = [
        np.array([-3.0, 12.0]),
        np.array([ 3.0,  2.0]),
        np.array([ 9.0,  3.0]),
    ]
    # for i, x0 in enumerate(starts, 1):
    #     run_solve(f"Branin #{i} (auto)", branin, x0=x0, mode="sqp")

    # # 3) (Optional) Constrained variants to exercise IP slack path
    # #    Add a large circle inequality so the feasible region is big but nontrivial.
    # run_solve("Rosenbrock + circle g<=0 (auto)", rosenbrock,
    #           x0=np.array([10.0, 10.0]), mode="ip", use_ineq=True, R=50.0)

    run_solve("Branin + circle g<=0 (auto)", branin,
              x0=np.array([3.0, 0.0]), mode="sqp", use_ineq=True, R=60.0)
