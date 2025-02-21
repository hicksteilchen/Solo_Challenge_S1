# Task 2

# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy


def sim_liaf(tau: float, r: float, vrest: float, vthr: float, inp: float, tmax: float, dt:float) -> tuple[np.ndarray, np.ndarray, float]:

    # timesteps required? init time axis and allocate space for v
    nt = int(np.ceil(tmax / dt))
    t_e = dt * np.arange(nt + 1)
    v_e = np.zeros((nt + 1,))

    # initial condition
    v_e[0] = vrest

    # Euler integration
    n = 0  # for counting spikes
    for i in range(nt):

        v_e[i + 1] = v_e[i] + dt / tau * (-(v_e[i] - vrest) + r * inp)
        if v_e[i + 1] > vthr:
            v_e[i + 1] = vrest
            n += 1

    return t_e, v_e, n


# default parameters
tau = 0.010  # [s]
r = 50e6  # [ohm]
vrest = -0.065  # [V]
vthr = -0.050  # [V]
v_0 = -0.065
dt = 1e-3  # [s]

# sim with and without threshold
t_e, v_e, n = sim_liaf(tau=tau, r=r, vrest=vrest, vthr=np.inf, inp=0.35e-9, tmax=0.1, dt=dt)
t_fire, v_fire, n_fire = sim_liaf(tau=tau, r=r, vrest=vrest, vthr=vthr, inp=0.35e-9, tmax=0.1, dt=1e-4)
v_ana = vrest + (r * 0.35e-9) * (1 - np.exp(-t_e / tau))
plt.plot(t_e, v_e)
plt.plot(t_fire, v_fire)
plt.plot(t_e, v_ana, 'k--')
plt.legend(('Euler, no thresh', 'Euler, thresh', 'Analytical, no thresh'))
plt.show()

# gain function
n_sims = 101
tmax = 10
inps = 0.2e-9 + 0.3e-9 * np.arange(n_sims) / (n_sims - 1)
ns = np.zeros((n_sims,))
for i, inp in enumerate(inps):
    t, v, ns[i] = sim_liaf(tau=tau, r=r, vrest=vrest, vthr=vthr, inp=inp, tmax=tmax, dt=dt)
plt.plot(inps, ns / tmax)
plt.show()





# Define LIF model
def lif(t, y, tau_m, v_rest, v_thresh, v_reset, i_e):
    dv_dt = (-(y - v_rest) + i_e) / tau_m  # membrane potential differential equation
    return [dv_dt]

def spike_cond(t, v):
    return v[0] - v_thresh

spike_cond.terminal = True
spike_cond.direction = 1

# Define parameters
tau_m = 10e-3  # membrane time constant (ms)
v_rest = -65e-3  # resting membrane potential (mV)
v_thresh = -55e-3  # spike threshold (mV)
v_reset = -65e-3  # reset potential (mV)
i_e = 3e-9  # input current (nA)
t_span = 1  # simulation time span (ms)
y0 = [v_rest]  # initial membrane potential (mV)


# Simulate LIF model
sol = scipy.integrate.solve_ivp(fun = lambda t, y: lif(t, y, tau_m, v_rest, v_thresh, v_reset, i_e),
                                t_span = t_span, y0 = y0, events=spike_cond, max_step=0.1e-3)

# Extract results
t = sol.t
v = sol.y[0]
#eventcount = sol.events
    

 # spike event
def spike(t: float, v: np.ndarray) -> float:
    return v[0] - v_thres 

spike.terminal = True  # stop if spike
spike.direction = 1 

# to store the results
t_total = []
v_total = []
t_start = 0
V_current = v_0
spike_times = []

# Continue integration after it stopped and reset 
while t_start < t_span:
    sol_int = sol(sim_liaf, [t_start, t_span], [V_current], events=spike)
    
    t_total.extend(sol.t)
    v_total.extend(sol.y[0])
    
    if sol.status == 1:
        spike_times.append(sol.t_events[0][0])
        V_current = v_rest
        t_start = sol.t_events[0][0] + 1e-4
    else:
        break  

# plot
plt.plot(t_total, v_total, label='Membrane Potential')
plt.axhline(y=v_thres, color='r', linestyle='--', label='Threshold')
plt.xlabel('Time (s)')
plt.ylabel('Membrane Potential (V)')
plt.title('LIAF with spikes')

for spike_time in spike_times:
    plt.axvline(x=spike_time, color='g', linestyle='--', label='Spike' if spike_time == spike_times[0] else "")

plt.legend()
plt.show()


# %% 
# Hudgkin-Huxley 


# 4 nonlinear odes for membrane potential V(t)
# 3 gatin variables (n(t), m(t), h(t))

# opening rate
# closing rate


# solve numerically, use euler to update V & gating variables in parallel

# %%

# task2 modify
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#Here are the general parameters
v_thres: float = -0.05
v_rest:float = -0.065
duration:float = 0.1
V_init:float = -0.065
tau: float = 0.010

#analytical solution
def analytical_solution(t:float = 0.1, v_rest:float = 0.065, v_thresh:float = 0.05, tau:float = 0.010) -> np.ndarray:
    return v_rest + (v_thresh - v_rest) * np.exp(-t / tau)


# LIAF equation function
def leaky(t: float, v: np.ndarray, tau: float = 0.010, r: float = 50e6, vrest: float= -0.065, inp: float = 0.35e-9) -> np.ndarray:
    return np.array([-(v[0] - vrest) / tau + r * inp / tau])

#euler integration 
def euler_integration(v_init: float, duration: float, dt: float, tau: float, r: float, v_rest: float, inp: float) -> np.ndarray:
    num_steps = int(duration / dt)
    t_vals = np.linspace(0, duration, num_steps)
    v_vals = np.zeros(num_steps)
    v_vals[0] = v_init
    
    for i in range(1, num_steps):
        v_vals[i] = v_vals[i-1] + dt * leaky(t_vals[i-1], np.array([v_vals[i-1]]), tau, r, v_rest, inp)[0]
    
    return t_vals, v_vals


# Compare numerical and analytical solutions
time_steps = [10e-2, 10e-3, 10e-4]

plt.figure()

for dt in time_steps:
    t_euler, v_euler = euler_integration(V_init, duration, dt, tau, 50e6, v_rest, 0.35e-9)
    v_analytic = analytical_solution(t_euler, v_rest, v_thres, tau)
    
    plt.plot(t_euler, v_euler, label=f'Euler Δt = {dt}')
    plt.plot(t_euler, v_analytic, '--', label=f'Analytical Δt = {dt}')
    
plt.xlabel('Time (s)')
plt.ylabel('Membrane Potential (V)')
plt.title('Impact of Different Time Step Sizes on Numerical Solution')
plt.legend()
plt.show()

# %%
#Now using the solver
plt.figure()

# Euler
t_euler, v_euler = euler_integration(V_init, duration, dt, tau, 50e6, v_rest, 0.35e-9)

# Solver
sol = solve_ivp(leaky, [0, duration], [V_init])

# Graficar las soluciones
plt.plot(t_euler, v_euler, label='Euler Method Δt = 10e-4')
plt.plot(sol.t, sol.y[0], label='RK45 Method')
plt.xlabel('Time (s)')
plt.ylabel('Membrane Potential (V)')
plt.title('Comparison of Euler and RK45 Methods')
plt.legend()
plt.show()

# %%

# spike event
def spike(t: float, v: np.ndarray) -> float:
    return v[0] - v_thres 

spike.terminal = True  # stop if spike
spike.direction = 1 

# to store the results
t_total = []
v_total = []
t_start = 0
V_current = V_init
spike_times = []

# Continue integration after it stopped and reset 
while t_start < duration:
    sol = solve_ivp(leaky, [t_start, duration], [V_current], events=spike)
    
    t_total.extend(sol.t)
    v_total.extend(sol.y[0])
    
    if sol.status == 1:
        spike_times.append(sol.t_events[0][0])
        V_current = v_rest
        t_start = sol.t_events[0][0] + 1e-4
    else:
        break  

# plot
plt.plot(t_total, v_total, label='Membrane Potential')
plt.axhline(y=v_thres, color='r', linestyle='--', label='Threshold')
plt.xlabel('Time (s)')
plt.ylabel('Membrane Potential (V)')
plt.title('LIAF with spikes')

for spike_time in spike_times:
    plt.axvline(x=spike_time, color='g', linestyle='--', label='Spike' if spike_time == spike_times[0] else "")

plt.legend()
plt.show()

# %%

# Here I include the max step so that the plot doesn't look edgy

# to store the results
t_total = []
v_total = []
t_start = 0
V_current = V_init
spike_times = []

# Continue integration after it stopped and reset 
while t_start < duration:
    sol = solve_ivp(leaky, [t_start, duration], [V_current], max_step = 1e-4, events=spike)
    
    t_total.extend(sol.t)
    v_total.extend(sol.y[0])
    
    if sol.status == 1:
        spike_times.append(sol.t_events[0][0])
        V_current = v_rest
        t_start = sol.t_events[0][0] + 1e-4
    else:
        break  

# plot
plt.plot(t_total, v_total, label='Membrane Potential')
plt.axhline(y=v_thres, color='r', linestyle='--', label='Threshold')
plt.xlabel('Time (s)')
plt.ylabel('Membrane Potential (V)')
plt.title('LIAF with spikes')

for spike_time in spike_times:
    plt.axvline(x=spike_time, color='g', linestyle='--', label='Spike' if spike_time == spike_times[0] else "")

plt.legend()
plt.show()

# %% 
# Hodgkin Huxley

# Coverted to use the formulas in the appendix
C = 1  # µF
g_Na = 40  # mS
g_K = 35  # mS
g_L = 0.3  # mS
V_Na = 55  # mV
V_K = -77  # mV
V_L = -65  # mV

# def exp_limits(V):
#     exp_limit_n  = np.clip(-(V - 25) / 9, -100, 100)  # restrict exp range to prevent overflow
#     exp_limit_m = np.clip(-(V + 35) / 9, -100, 100)
#     exp_limit_h_1 = np.clip(-(V + 90) / 12, -100, 100)
#     exp_limit_h_2 = np.clip((V + 62) / 6 - (V + 90) / 12, -100, 100)
#     return exp_limit_n, exp_limit_m, exp_limit_h_1, exp_limit_h_2

#these are the functions on the appendix
def alpha_n(V):
    exp_limit_n  = np.clip(-(V - 25) / 9, -100, 100)
    return 0.02 * (V - 25) / (1 - np.exp(exp_limit_n))

def beta_n(V):
    exp_limit_n  = np.clip(-(V - 25) / 9, -100, 100)
    return -0.002 * (V - 25) / (1 - np.exp((exp_limit_n)))

def alpha_m(V):
    exp_limit_m = np.clip(-(V + 35) / 9, -100, 100)
    return 0.182 * (V + 35) / (1 - np.exp(exp_limit_m))

def beta_m(V):
    exp_limit_m = np.clip(-(V + 35) / 9, -100, 100)
    return -0.124 * (V + 35) / (1 - np.exp(exp_limit_m))

def alpha_h(V):
    exp_limit_h_1 = np.clip(-(V + 90) / 12, -100, 100)
    return 0.25 * np.exp(exp_limit_h_1)

def beta_h(V):
    exp_limit_h_2 = np.clip((V + 62) / 6 - (V + 90) / 12, -100, 100)
    return 0.25 * np.exp(exp_limit_h_2)

#define function to integrate equations
def HH_equations(t, y, I_ext):
    V, n, m, h = y
    exp_limit_n  = np.clip(-(V - 25) / 9, -100, 100)
    exp_limit_m = np.clip(-(V + 35) / 9, -100, 100)
    exp_limit_h_1 = np.clip(-(V + 90) / 12, -100, 100)
    exp_limit_h_2 = np.clip((V + 62) / 6 - (V + 90) / 12, -100, 100)
    
    dVdt = (-g_K * n**4 * (V - V_K) - g_Na * m**3 * h * (V - V_Na) - g_L * (V - V_L) + I_ext) / C
    dndt = alpha_n(V) * (1 - n) - beta_n(V) * n
    dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h
    
    return [dVdt, dndt, dmdt, dhdt]

def euler(dt, T, I_ext):
    time = np.arange(0, T, dt)
    V = np.zeros(len(time))
    n = np.zeros(len(time))
    m = np.zeros(len(time))
    h = np.zeros(len(time))
    
    V[0] = -65  # Initialize in -65mV 
    n[0] = alpha_n(V[0]) / (alpha_n(V[0]) + beta_n(V[0]))
    m[0] = alpha_m(V[0]) / (alpha_m(V[0]) + beta_m(V[0]))
    h[0] = alpha_h(V[0]) / (alpha_h(V[0]) + beta_h(V[0]))
    
    for i in range(1, len(time)):
        dVdt, dndt, dmdt, dhdt = HH_equations(time[i-1], [V[i-1], n[i-1], m[i-1], h[i-1]], I_ext)
        V[i] = V[i-1] + dt * dVdt
        n[i] = n[i-1] + dt * dndt
        m[i] = m[i-1] + dt * dmdt
        h[i] = h[i-1] + dt * dhdt
    
    return time, V, n, m, h

# Parameters
T = 50  # total time in miliseconds
dt = 10e-5  # Step size in miliseconds
I_ext = 2 # External input current, which I varied until observed spikes

time, V, n, m, h = euler(dt, T, I_ext)

# Plot
plt.figure(figsize=(10, 5))

plt.subplot(2, 1, 1)
plt.plot(time, V, label='Membrane Voltage (mV)')
plt.ylabel('Voltage (mV)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time, n, label='n-gate')
plt.plot(time, m, label='m-gate')
plt.plot(time, h, label='h-gate')
plt.ylabel('Gating Variables')
plt.xlabel('Time (ms)')
plt.legend()
plt.show()

# %%
# solve using solve_ivp
def HH_equations_ivp(t, y):
    return HH_equations(t, y, I_ext) #I use the equations I already had before

sol_ivp = solve_ivp(HH_equations_ivp, [0, T], [-65, n[0], m[0], h[0]], method='RK45', t_eval=time)

# plot comparison
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time, V, label='Euler Method', alpha=0.6)
plt.plot(sol_ivp.t, sol_ivp.y[0], label='solve_ivp', alpha=0.6)
plt.ylabel('Voltage (mV)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time, n, label='n-gate (Euler)', alpha=0.6)
plt.plot(time, m, label='m-gate (Euler)', alpha=0.6)
plt.plot(time, h, label='h-gate (Euler)', alpha=0.6)
plt.plot(sol_ivp.t, sol_ivp.y[1], label='n-gate (solve_ivp)', linestyle='dashed', alpha=0.6)
plt.plot(sol_ivp.t, sol_ivp.y[2], label='m-gate (solve_ivp)', linestyle='dashed', alpha=0.6)
plt.plot(sol_ivp.t, sol_ivp.y[3], label='h-gate (solve_ivp)', linestyle='dashed', alpha=0.6)
plt.ylabel('Gating Variables')
plt.xlabel('Time (ms)')
plt.legend()
plt.show()