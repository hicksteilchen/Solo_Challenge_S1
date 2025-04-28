# %%
import numpy as np
import matplotlib.pyplot as plt
import math 


# pure
total_simtime = 0.2

#-3
dt = 0.1e-3
tau = 10e-3
V_tresh = -50e-3
V_rest = -65e-3
V_reset = -65e-3
R = 50e6
ext_input = np.array([0.3e-9, 0.32e-9, 0.35e-9])
time = np.arange(0, total_simtime, dt)
total_spikes = []
total_potential = []
gleak = 0.0001
Input = 3.5e-9
alpha = 0.001
Vc = -55e-3
delta_T = 3e-3
tresh_e = -55e-3

V = np.full(len(time), V_rest)
spike_train = np.zeros(len(time))

for t in range(1, len(time)):
    dV = (-gleak) * ((-(V[t - 1] - V_rest) + R * Input)* dt / tau)
    V[t] = V[t-1] + dV

    if V[t] >= V_tresh:
        V[t] = V_reset
        spike_train[t] = 1

    total_potential.append(V)
    total_spikes.append(spike_train)

# %%

class IAF:
    V_rest: float
    V_tresh: float
    V: list
    Input: float
    simulation_time: float
    tau: float
    R = float
    dt = float
    time = list
    t_sim = float
    dv = list
    spike_train = list
    # all_potentials = list
    # all_spikes = list

    def __init__(self):
        self.V_rest = -65e-3
        self.Input = 2.5e-9
        self.simulation_time = 100e-3
        self.dt = 0.1e-3
        self.tau = 10e-3
        self.R = 50e6
        self.V_tresh = -50e-3
        self.time = np.arange(0, self.simulation_time, self.dt)
        self.V = np.full(len(self.time), self.V_rest)  # Initialize V as an array
        self.spiketrain = np.zeros(len(self.time))
        self.dv = np.zeros(len(self.time))

    def check_spike(self):
        for t in range(1, len(self.time)):
            if self.V[t] >= self.V_tresh:
                self.V[t] = self.V_rest
                self.spiketrain[t] = 1


# %%

plt.clf()
plt.figure(figsize=(12, 8))

# membrane potential

for i, inp in enumerate(Input):
    plt.subplot(len(Input), 2 , 2 * i +1)
plt.plot(time, total_potential[i])

for i, inp in enumerate(Input):
    plt.subplot(len(Input), 2, 2 *i + 2)
   plt.plot(time, total_spikes[i])

plt.xlabel("time")
plt.tight_layout()
plt.show()
# %%


class QIAF(IAF):
    def __init__(self):
        super().__init__()
        self.alpha = 0.001
        self.Vc = -55e-3
        
    def eulers_input(self, t):
        return (
        (
        (-(self.V[t - 1] - self.V_rest) * self.alpha * (self.Vc - self.V[t]) 
         + self.R * self.Input) * self.dt / self.tau)
        )
# %%
class EIAF(IAF):
    def __init__(self):
        super().__init__()
        self.deT = 3e-3
        self.tresh = -55e-3
        
    def eulers_input(self, t):
        return (((self.V[t-1] -self.V_rest) + self.deT * math.exp((self.V[t-1] - self.tresh) / self.deT) + self.R * self.Input * self.dt / self.tau)
        )

#%%
plt.clf
plt.figure(figsize=(12,8))

for i, inp in enumerate(Input):
    plt.subplot(len(Input), 2, 2 * i + 1)
    plt.plot(time, total_potential[i], label=f"inp = {inp :.1f} nA")
    plt.axhline(V_tresh, color="r", linestyle="--", label="Threshold")
    plt.ylabel("V (mV)")
    plt.title(f"Membrane Potential (inp = {inp:.1f} nA)")
    plt.legend()
    plt.grid()

# Plot spike trains
for i, inp in enumerate(Input):
    plt.subplot(len(Input), 2, 2 * i + 2)
    plt.plot(time,total_spikes[i], drawstyle="steps-post", label=f"inp = {inp:.1f} nA")
    plt.title(f"Spike Train (inp = {inp:.1f} nA)")
    plt.ylabel("Spikes")
    plt.grid()

plt.xlabel("Time (ms)")
plt.tight_layout()
plt.show()
# %%
