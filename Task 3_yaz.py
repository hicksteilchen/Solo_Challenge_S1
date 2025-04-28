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

I = sympy.symbols("I")

# Before we can solve a (normal or differential) equation, we have
# to define it properly, i.e.: d/dt v(t) = -v(t)+i
IAF = sympy.Eq(v(t).diff(t), -v(t)+I)
# ...then we can try to solve! Remember that differential equations
# need a different sympy command than normal equations. Their solution
# is only unique if you provide an initial condition:
solution = sympy.dsolve(IAF, v(t))
print(solution)
# ...as an outcome, we here expect something like v(t) = i*(1-exp(-t))
# Now it should be easy for you to take part of the solution equation,
# and express the condition that you are looking for a time t when the
# neuron will fire as a normal equation, and then solve it!
solution2 = sympy.dsolve(IAF, v(t), ics={v(0): 0})

print(solution2)

# Typically, there are multple solutions such that you have to select
# which one to process further, and convert the corresponding expression
# into a standard numpy function (let's name it 'gain')...
sol_rhs = solution2.rhs

#print(gain(2))

t_fire = sympy.solve(sympy.Eq(sol_rhs, 1), t)[0]
gain = sympy.lambdify((I), 1 / t_fire)


# Okay, let's plot that function:

n_bins = 100
input_current_numerical = 4 * np.arange(n_bins + 1) / n_bins
plt.plot(input_current_numerical, gain(input_current_numerical))
plt.show()  

#%%
#3.2 Synaptic transmission
#a) Before firing up Sympy, let us think about what happens at t = 0 if a spike described by the deltadistribution
#I(t) = δ(t) arrives. Which value will x assume shortly after arrival? 1/tau_x

#b) Use this value as initial condition to solve the first DEQ for t ∈ [0,∞]. Plug in the solution into the
#second DEQ and solve it for an initial condition y(0) = 0 for the same time interval.

t, tau_x, tau_y, a = sympy.symbols('t tau_x tau_y a', real=True, positive=True)
x = sympy.Function('x')(t)

# differential equation
eq_x = sympy.Eq(tau_x * x.diff(t), -x)

# Solve for function x with initial condition mentioned above
sol_x = sympy.dsolve(eq_x, x, ics={x.subs(t, 0): 1/tau_x})


x_t = sol_x.rhs

x_t



# %%
#Now we plug x in y to solve for it as well

y = sympy.Function('y')(t)


eq_y = sympy.Eq(tau_y * y.diff(t), -y + a * x_t)

# Solve with initial condition y(0) = 0
sol_y = sympy.dsolve(eq_y, y, ics={y.subs(t, 0): 0})


y_t = sol_y.rhs
y_t

# %%
#Simplify your solution by assuming that the two time constants are identical, τx = τy. Do you encounter
#a problem? If yes, try to solve again by assuming identical time constants right from the beginning!

#This would lead to division by zero
#I will solve again using tau instead of tau_x and tau_y

x2 = sympy.Function('x2')(t)
tau = sympy.symbols('tau')

eq_x2 = sympy.Eq(tau * x2.diff(t), -x2)
sol_x2 = sympy.dsolve(eq_x2, x2, ics={x2.subs(t, 0): 1/tau})
x_t2 = sol_x2.rhs
x_t2
#%%Now I solve for y again
y2 = sympy.Function('y')(t)
eq_y2 = sympy.Eq(tau * y2.diff(t), -y2 + a * x_t2)
sol_y2 = sympy.dsolve(eq_y2, y2, ics={y2.subs(t, 0): 0})


y_t2 = sol_y2.rhs
y_t2

#And .... it doesn't work
# %%
#Here i will set x(0)=1, as the professor suggested.



# Define and solve again
eq_x = sympy.Eq(tau * x.diff(t), -x)
sol_x = sympy.dsolve(eq_x, x, ics={x.subs(t, 0): 1})
x_t = sol_x.rhs

# Define and solve for y again
eq_y = sympy.Eq(tau * y.diff(t), -y + a * x_t)
sol_y = sympy.dsolve(eq_y, y, ics={y.subs(t, 0): 0})
y_t = sol_y.rhs

print(x_t,y_t)


# %%
#Create a Numpy function out of your normalized solution y(t) and plot it for different choices of the
#parameters. Confirm that the integral over the solution is indeed normalized to 1.
# Convert solution to a numerical function
y_t_lambda = sympy.lambdify((t, tau, a), y_t, 'numpy')

# Define a function to compute and plot y(t)
def plot_normalized_solution(tau_val, a_val):
    t_vals = np.linspace(0, 5 * tau_val, 1000)
    y_vals = y_t_lambda(t_vals, tau_val, a_val)
    
    # Normalize solution
    integral = np.trapz(y_vals, t_vals)
    y_vals /= integral
    
    plt.plot(t_vals, y_vals, label=f"tau={tau_val}, a={a_val}")
    plt.xlabel("t")
    plt.ylabel("Normalized y(t)")
    plt.legend()
    
    print(f"Integral over y(t) for tau={tau_val}, a={a_val}: {np.trapz(y_vals, t_vals)}")

# Plot for different parameter choices
plt.figure(figsize=(8, 5))
plot_normalized_solution(2, 2)
plot_normalized_solution(5, 1)
plot_normalized_solution(1, 2)
plt.show()

# Display results
x_t, y_t
# %%
