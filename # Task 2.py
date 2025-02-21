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
