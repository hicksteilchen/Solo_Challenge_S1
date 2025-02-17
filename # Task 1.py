# Task 1

# %% 
# 1.1 a) Make a plan
# How would you organize your hierarchy?
# - fr
# What attributes should classes have, what methods would you provide?
# methods: update membrane potential, check for spikes, general LIF
# How are you going to initialize your class when instantiated?
# Which methods could you re-use multiple times by inheritance?
# 

# %%
# 1.1 b) Implement yout code

# %%
# 1.1 c) Test your code


# %%

# 1.2. Model eqations and parameters
# implement leaky, quadratic and exponential integrate and fire neurons
#  and plot them



import numpy as np
import matplotlib.pyplot as plt


class IAF:
    def __init__(self):
        self.V_rest = -65e-3
        self.V_thresh = -45e-3
        self.Input = .5e-9
        self.simulation_time = 100e-3
        self.dt = 0.1e-3
        self.tau = 10e-3
        self.R = 50e6
        self.gleak = 0.0001

        self.time = np.arange(0, self.simulation_time, self.dt)
        self.V = np.full(len(self.time), self.V_rest)  # Initialize membrane potential
        self.spiketrain = np.zeros(len(self.time))
        self.dv = np.zeros(len(self.time))

    def eulers_input(self, t):
        """Placeholder function to be overridden in subclasses"""
        raise NotImplementedError("Subclasses must implement this method.")

    def analyse_iaf(self):
        for t in range(1, len(self.time)):
            self.dv[t] = self.eulers_input(t)
            self.V[t] = self.V[t - 1] + self.dv[t]

            if self.V[t] >= self.V_thresh:
                self.V[t] = self.V_rest
                self.spiketrain[t] = 1


class LIAF(IAF):
    def __init__(self):
        super().__init__()

    def eulers_input(self, t):
        """Compute Euler update for LIAF"""
        return (-self.gleak * (self.V[t - 1] - self.V_rest) + (self.R * self.Input)) * (
            self.dt / self.tau
        )


class QIAF(IAF):
    def __init__(self):
        super().__init__()
        self.alpha = 100e-3
        self.Vc = -45e-3
        self.V_thresh = -55e-3

    def eulers_input(self, t):
        """Compute Euler update for QIAF"""
        return ((-self.gleak * (self.V[t - 1] - self.V_rest) * self.alpha * (self.Vc - self.V[t - 1])
                      + self.R * self.Input)* (self.dt / self.tau))



class EIAF(IAF):
    def __init__(self):
        super().__init__()
        self.deT = 3e-3
        self.V_tresh = -55e-3

    def eulers_input(self, t):
        """Compute Euler update for EIAF"""
        return ((-self.gleak * 
                (self.V[t - 1] - self.V_rest)
                + self.deT
                * np.exp((self.V[t - 1] - self.V_thresh) / self.deT)
                + self.R * self.Input )* (self.dt / self.tau))


# Run and plot LIAF
instance_liaf = LIAF()
instance_liaf.analyse_iaf()

plt.clf()
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(instance_liaf.time, instance_liaf.V, label="Membrane Potential")
plt.axhline(instance_liaf.V_thresh, color="r", linestyle="--", label="Threshold")
plt.ylabel("V (V)")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(
    instance_liaf.time, instance_liaf.spiketrain, label="Spikes", drawstyle="steps-post"
)
plt.ylabel("Spikes")
plt.xlabel("Time (s)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Run and plot QIAF
instance_qiaf = QIAF()
instance_qiaf.analyse_iaf()

plt.clf()
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(instance_qiaf.time, instance_qiaf.V, label="Membrane Potential")
plt.axhline(instance_qiaf.V_thresh, color="r", linestyle="--", label="Threshold")
plt.ylabel("V (V)")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(
    instance_qiaf.time, instance_qiaf.spiketrain, label="Spikes", drawstyle="steps-post"
)
plt.ylabel("Spikes")
plt.xlabel("Time (s)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


# Run and plot EIAF
instance_eiaf = EIAF()
instance_eiaf.analyse_iaf()

plt.clf()
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(instance_eiaf.time, instance_eiaf.V, label="Membrane Potential")
plt.axhline(instance_eiaf.V_thresh, color="r", linestyle="--", label="Threshold")
plt.ylabel("V (V)")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(
    instance_eiaf.time, instance_eiaf.spiketrain, label="Spikes", drawstyle="steps-post"
)
plt.ylabel("Spikes")
plt.xlabel("Time (s)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
