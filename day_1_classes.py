import numpy as np
import matplotlib.pyplot as plt


class IAF:
    def __init__(self):
        self.V_rest = -65e-3
        self.V_thresh = -45e-3
        self.Input = 2.5e-9
        self.simulation_time = 100e-3
        self.dt = 0.1e-3
        self.tau = 10e-3
        self.R = 50e6
        self.gleak = 0.001

        self.time = np.arange(0, self.simulation_time, self.dt)
        self.V = np.full(len(self.time), self.V_rest)  # Initialize V as an array
        self.spiketrain = np.zeros(len(self.time))
        self.dv = np.zeros(len(self.time))

    def check_spike(self, t):
        """Checks for spikes at time step t and resets potential if needed."""
        if self.V[t] >= self.V_thresh:
            self.V[t] = self.V_rest  # Reset potential after a spike
            self.spiketrain[t] = 1  # Register spike


class LIAF(IAF):
    def __init__(self):
        super().__init__()

    def update_potential(self):
        """Updates the membrane potential over time using Euler's method."""
        for t in range(1, len(self.time)):
            self.dv[t] = self.gleak * (
                (-(self.V[t - 1] - self.V_rest) + self.R * self.Input)
                * self.dt
                / self.tau
            )
            self.V[t] = self.V[t - 1] + self.dv[t]
            self.check_spike(t)  # Check for spike after updating potential


# Create instance and simulate
instance_LIAF = LIAF()
instance_LIAF.update_potential()  # Update the full time series

# Store results
allpot = instance_LIAF.V
allspikes = instance_LIAF.spiketrain

# Plot results
plt.figure(figsize=(12, 8))

# Plot membrane potential
plt.subplot(2, 1, 1)
plt.plot(instance_LIAF.time, allpot, label="Membrane Potential")
plt.axhline(instance_LIAF.V_thresh, color="r", linestyle="--", label="Threshold")
plt.ylabel("V (V)")
plt.legend()
plt.grid()

# Plot spike train
plt.subplot(2, 1, 2)
plt.plot(instance_LIAF.time, allspikes, label="Spikes", drawstyle="steps-post")
plt.ylabel("Spikes")
plt.xlabel("Time (s)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()



#class QLIF(IAF):
#    def __init__(self):
#        super().__init__()
#        
#        self.alpha = 0.001
#        self.Vc = -55e-3
