# %% [markdown]
# ### Fourier transform and power spectra
#
# First we create a time-varying signal with well-defined frequency content:
#

# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.integrate
import scipy.signal

# properties of time series: length, sampling rate...
t_max = 3
dt_bin = 0.0025
n_bins = np.ceil(t_max / dt_bin)
t = dt_bin * np.arange(n_bins)

# harmonic signal: frequency, offset, amplitude...
f_sin = 42
a_sin = 3.2
a_ofs = 2.1
signal = a_ofs + a_sin * np.sin(2 * np.pi * f_sin * t) + 20 * np.random.random(int(n_bins))

# ...just show the signal
plt.figure(1)
plt.plot(t, signal)
plt.xlabel('time t')
plt.ylabel('signal s(t)')
plt.show()

# %% [markdown]
#
# ### Tasks
# 1. Compute the power spectrum of the signal, and confirm that you have a peak at the frequency corresponding to the sine in the signal.


def power(
        signal: np.ndarray, # signal
        t: float,           # time - from signal? 
        dt: float# time steps/sampling interval          
):
    N = signal.shape
    T = t[-1]              # number of data points * sampling intervals

    df = 1 / T.max()                # frequency resolution
    nyq = 1 / dt / 2                # nyquist frequency
    faxis = np.arange(0, nyq, df)   # frequency axis

    f_signal = np.fft.fft(signal - signal.mean())       # fourier transformation of signal (absolute value?)
    S_spectrum = 2 * dt **2 / T * (f_signal * np.conj(f_signal)) # compute spectrum (whole signal **2)
    S_spectrum = S_spectrum[:int(len(signal) / 2)]          # ignore negative frequencies
    

    
    return f_signal, S_spectrum, faxis

f_signal, pwsig, pwaxis = power(signal, t, dt_bin)


plt.figure(1)
plt.plot(pwaxis, pwsig)
plt.xlabel('frequency')
plt.ylabel('signal s(t)')
plt.show()

#

# 2. Check if Parseval's theorem is satisfied.

pw_var = np.var(pwsig)
print (pw_var)


wel = scipy.signal.welch(pwsig)
meanwel = np.mean(wel)
print(meanwel)


#sumsig = np.sum(pwsig)
#print (sumsig)
#vasi = np.var(pwsig)
#print(vasi)


# 3. Implement a function which computes the power spectrum by averaging over the power spectra obtained from chunks of the data. Plot it in comparison to the non-averaged spectrum.
## - average over power spectra over part of a time series to
# reduce noise in the power estimate


# 4. Try to add some noise to the signal and watch what happens with the power spectra...
#

# %%
# YOUR CODE...
