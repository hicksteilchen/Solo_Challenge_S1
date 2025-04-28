# %%
import numpy as np
import matplotlib.pyplot as plt

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



class LIAF(IAF):

    def __init__(self):
        super().__init__()

    def update_potential(self):
        for t in range(1, len(self.time)):
            self.dv[t] = (-(self.V[t-1] - self.V_rest) + self.R * self.Input) * self.dt / self.tau
            self.V[t] += self.dv[t]


instance_LIAF = LIAF()

all_spikes_lif = instance_LIAF.check_spike()
all_potentials_lif = instance_LIAF.update_potential()


# Plot results
plt.clf()
plt.figure(figsize=(12, 8))

# Plot membrane potentials
plt.subplot(2, 1, 1)
plt.plot(instance_LIAF.time, instance_LIAF.V[t], label="Membrane Potential")
plt.axhline(instance_LIAF.V_tresh, color='r', linestyle='--', label='Threshold')
plt.ylabel('V (mV)')
plt.legend()
plt.grid()

# Plot spike trains
plt.subplot(2, 1, 2)
plt.plot(instance_LIAF.time, instance_LIAF.spiketrain, label="Spikes")
plt.ylabel('Spikes')
plt.xlabel('Time (s)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# %%
instance_LIAF = LIAF()
print(instance_LIAF.Input)
# %%
