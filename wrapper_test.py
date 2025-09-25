import math
import numpy as np
from nlp.wrapper import Model, cos   # AD-aware functions
from chomp import NLPSolver
from nlp.blocks.aux import SQPConfig

# Branin with circular inequality
m = Model("branin")
x = m.add_var("x", shape=2)          # decision vars
x1, x2 = x[0], x[1]

a, b, c, r, s, t = 1.0, 5.1/(4*math.pi**2), 5.0/math.pi, 6.0, 10.0, 1/(8*math.pi)
f = a*(x2 - b*(x1**2) + c*x1 - r)**2 + s*(1 - t)*cos(x1) + s
m.minimize(f)
m.add_constr(x1**2 + x2**2 <= 60.0)

# Compile to callables and solve with SQP
f_fun, cI, cE, x0, lb, ub = m.build()
solver = NLPSolver(f_fun, c_ineq=cI, c_eq=cE, x0=np.array([-3.0, 12.0]),
                   config=SQPConfig())
x_star, info = solver.solve(max_iter=150, tol=1e-8, verbose=True)
#print("x* =", x_star, info.status)
print("x* =", x_star)