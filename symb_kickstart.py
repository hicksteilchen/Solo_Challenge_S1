# %%
import numpy as np
import matplotlib.pyplot as plt
import sympy

# Kickstart: analyze the normalized leaky integrate-and-fire neuron
# =================================================================

# First, we need some symbols to define the equations we want to solve
# it is important to specify what properties these symbols should have,
# for example, if it is a function such as v(t) or a simple variable t
# which is real-valued (i.e., can not be complex...)
#
# ...maybe you need some further symbols?
v = sympy.symbols("v", cls=sympy.Function)
t = sympy.symbols("t", real=True)
...

# Before we can solve a (normal or differential) equation, we have
# to define it properly, i.e.: d/dt v(t) = -v(t)+i
...

# ...then we can try to solve! Remember that differential equations
# need a different sympy command than normal equations. Their solution
# is only unique if you provide an initial condition:
...
# ...as an outcome, we here expect something like v(t) = i*(1-exp(-t))

# Now it should be easy for you to take part of the solution equation,
# and express the condition that you are looking for a time t when the
# neuron will fire as a normal equation, and then solve it!




# Typically, there are multple solutions such that you have to select
# which one to process further, and convert the corresponding expression
# into a standard numpy function (let's name it 'gain')...
...

# Okay, let's plot that function:
...

n_bins = 100
input_current_numerical = 4 * np.arange(n_bins + 1) / n_bins
plt.plot(input_current_numerical, gain(input_current_numerical))
plt.show()
# We're done! Now let's try it again on the full leaky IAF neuron
# which has a few more parameters:
...
...
...
...
...
...
