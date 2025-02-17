# Task 3

# %%
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp  # type:ignore

# Kickstart: analyze the normalized leaky integrate-and-fire neuron
# =================================================================

# First, we need some symbols to define the equations we want to solve
# it is important to specify what properties these symbols should have,
# for example, if it is a function such as v(t) or a simple variable t
# which is real-valued (i.e., can not be complex...)
#
# ...maybe you need some further symbols?

t = sp.symbols("t")
v = sp.Function("v")(t)
i = sp.symbols("i")

# Before we can solve a (normal or differential) equation, we have
# to define it properly, i.e.: d/dt v(t) = -v(t)+i

diff_eq = sp.Eq(sp.Derivative(v, t), -v + i)

diffsol_2 = sp.dsolve(diff_eq, v, ics={v.subs(t, 0): 0})
print(diffsol_2)

fire = sp.Eq((1), diffsol_2.rhs)
t_fire = sp.solve(fire, t)
print(t_fire)

g = [1 / t_fire[0]]
print("g:", g)

g_fun = sp.lambdify(t, g)
print(g_fun)

# %%

# Define n_bins and compute input current array
n_bins = 100
input_current_numerical = 4 * np.arange(n_bins + 1) / n_bins

# Plot the gain function over time for each input current value


# %%

# We're done! Now let's try it again on the full leaky IAF neuron
# which has a few more parameters: