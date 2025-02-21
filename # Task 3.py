# Task 3
# 3 Numerical & Symbolic Mathematics://Analytical solutions
# Python package sp allows you to perform sophisticated analytical computations. We will train to use
# the package on two examples, namely, the integrate-and-fire neuron (where you know already the results
# because you computed them ’by hand’ !) and synaptic transmission of a spike (which is something new!).
# 3.1 Integrate-and-fire neuron
# The leaky IAF neuron in normalized form is given by the differential equation (DEQ)
# dv
# dt = −v(t) + i(t)
# with membrane potential v(t) and input i(t).
# a) Solve the DEQ analytically for an initial condition v(0) = 0.
# b) Compute an expression for the time t_fire it takes the neuron to fire.
# c) Compute the gain function from the previous result.
# d) Convert the analytical gain function into a Numpy function and plot it for i ∈ [0, 4]

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

g_fun = sp.lambdify(i, g[0], 'numpy')
print(g_fun)

# %%


# Plot the gain function over time for each input current value
# Define n_bins and compute input current array
# Compute the input current and gain function values over time
#current_values = input_current(time)
#gain_values = gain_function(time, current_values)

n_bins = 100
input_current_numerical = 4 * np.arange(n_bins + 1) / n_bins
gain_values = g_fun(input_current_numerical)

# Plot the gain function
plt.figure(figsize=(10, 6))
plt.plot(input_current_numerical, gain_values, label='Gain Function')
plt.xlabel('Input Current')
plt.ylabel('Gain')
plt.title('Gain Function for Each Input Current Value')
plt.legend()
plt.grid(True)



# %%
##3.2 

# a) Before firing up sp, let us think about what happens at t = 0 if a spike described by the deltadistribution
#I(t) = δ(t) arrives. Which value will x assume shortly after arrival? 1/tau_x

# b) Use this value as initial condition to solve the first DEQ for t ∈ [0,∞]. Plug in the solution into the
#second DEQ and solve it for an initial condition y(0) = 0 for the same time interval.

t, tau_x, tau_y, a = sp.symbols('t tau_x tau_y a', real=True, positive=True)
x = sp.Function('x')(t)

# differential equation
eq_x = sp.Eq(tau_x * x.diff(t), -x)


sol_x = sp.dsolve(eq_x, x, ics={x.subs(t, 0): 1/tau_x})


x_t = sol_x.rhs

x_t

y = sp.Function('y')(t)


eq_y = sp.Eq(tau_y * y.diff(t), -y + a * x_t)

# Solve with initial condition y(0) = 0
sol_y = sp.dsolve(eq_y, y, ics={y.subs(t, 0): 0})


y_t = sol_y.rhs
y_t


# %% 

#Simplify your solution by assuming that the two time constants are identical, τx = τy. Do you encounter
#a problem? If yes, try to solve again by assuming identical time constants right from the beginning!

#This would lead to division by zero
#I will solve again using tau instead of tau_x and tau_y

x2 = sp.Function('x2')(t)
tau = sp.symbols('tau')

eq_x2 = sp.Eq(tau * x2.diff(t), -x2)
sol_x2 = sp.dsolve(eq_x2, x2, ics={x2.subs(t, 0): 1/tau})
x_t2 = sol_x2.rhs
x_t2

y2 = sp.Function('y')(t)
eq_y2 = sp.Eq(tau * y2.diff(t), -y2 + a * x_t2)
sol_y2 = sp.dsolve(eq_y2, y2, ics={y2.subs(t, 0): 0})


y_t2 = sol_y2.rhs
y_t2

# Define and solve again
eq_x = sp.Eq(tau * x.diff(t), -x)
sol_x = sp.dsolve(eq_x, x, ics={x.subs(t, 0): 1})
x_t = sol_x.rhs

# Define and solve for y again
eq_y = sp.Eq(tau * y.diff(t), -y + a * x_t)
sol_y = sp.dsolve(eq_y, y, ics={y.subs(t, 0): 0})
y_t = sol_y.rhs

print(x_t,y_t)

# %%
#Create a Numpy function out of your normalized solution y(t) and plot it for different choices of the
#parameters. Confirm that the integral over the solution is indeed normalized to 1.
# Convert solution to a numerical function
y_t_lambda = sp.lambdify((t, tau, a), y_t, 'numpy')

# convert to numerical function and plot y(t)
def plot_normalized_solution(tau_val, a_val):
    t_vals = np.linspace(0, 5 * tau_val, 1000)
    y_vals = y_t_lambda(t_vals, tau_val, a_val)
    
    # normalised
    integral = np.trapezoid(y_vals, t_vals)
    y_vals /= integral
    
    plt.plot(t_vals, y_vals, label=f"tau={tau_val}, a={a_val}")
    plt.xlabel("t")
    plt.ylabel("Normalized y(t)")
    plt.legend()
    
    print(f"Integral over y(t) for tau={tau_val}, a={a_val}: {np.trapezoid(y_vals, t_vals)}")

# Plot 
plt.figure(figsize=(8, 5))
plot_normalized_solution(2, 2)
plot_normalized_solution(5, 1)
plot_normalized_solution(1, 2)
plt.show()

# Display results
x_t, y_t


# %%
