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
t_span = (0, .1)  # simulation time span (ms)
y0 = [v_rest]  # initial membrane potential (mV)


# Simulate LIF model
sol = scipy.integrate.solve_ivp(fun = lambda t, y: lif(t, y, tau_m, v_rest, v_thresh, v_reset, i_e),
                                t_span = t_span, y0 = y0, events=spike_cond, max_step=0.1e-3)

# Extract results
t = sol.t
v = sol.y[0]
#eventcount = sol.events
    

plt.clf()
plt.figure(figsize=(6, 4))
plt.plot(t * 1e3, v * 1e3, label="Membrane Potential")
plt.axhline(v_thresh * 1e3, linestyle="--", color="r", label="Threshold")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.title("LIF Neuron Simulation with Spike Event")
plt.legend()
plt.show()


# %% 
# Hudgkin-Huxley 