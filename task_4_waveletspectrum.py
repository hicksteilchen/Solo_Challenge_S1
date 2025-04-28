# %% [markdown]
# ### Time-varying signals and wavelet transform
#
# For exploring the wavelet transform, we create a signal with a sine wave whose frequency changes over time. 
#

# %%
import numpy as np
import matplotlib.pyplot as plt
import pywt #type:ignore

# Calculate the wavelet scales we requested
def wavelet_transform_morlet(
        data: np.ndarray,
        n_freqs: int,
        freq_min: float,
        freq_max: float,
        dt: float,
        bandwidth: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # wavelet scales derived from parameters
    s_spacing: np.ndarray = (1.0 / (n_freqs - 1)) * np.log2(freq_max / freq_min)
    scale: np.ndarray = np.power(2, np.arange(0, n_freqs) * s_spacing)
    freq_axis: np.ndarray = freq_min * scale
    wave_scales: np.ndarray = 1.0 / (freq_axis * dt)

    # the wavelet we want to use
    mother = pywt.ContinuousWavelet(f"cmor{bandwidth}-1.0")

    # one or multiple time series? --> expand
    data_2d = data
    if data.ndim == 1:
        data_2d = data_2d[np.newaxis, :]
    n_trials = data_2d.shape[0]

    complex_spectrum = np.zeros([n_trials, n_freqs, data_2d.shape[1]], dtype=np.complex128)
    for i_trial in range(n_trials):
        complex_spectrum[i_trial, :, :], freq_axis_wavelets = pywt.cwt(
            data=data_2d[i_trial, :], scales=wave_scales, wavelet=mother, sampling_period=dt
        )

    # one or multiple time series? <-- flatten
    if data.ndim == 1:
        complex_spectrum = complex_spectrum[0, :, :]

    # generate time axis and cone-of-influence
    t_axis = dt * np.arange(data_2d.shape[1])
    t_coi = (bandwidth * 3) / 2 / np.pi * np.sqrt(2) / freq_axis_wavelets

    return complex_spectrum, t_axis, freq_axis_wavelets, t_coi


def wavelet_dsignal_show(
        wavelet_dsignal: np.ndarray,
        t_axis: np.ndarray,
        f_axis: np.ndarray,
        t_coi: None | np.ndarray = None
):

    # average over first dimension, if signal_wavelet has three dims
    to_show = wavelet_dsignal  # np.abs(wavelet_dsignal) ** 2
    if to_show.ndim == 3:
        to_show = to_show.mean(axis=0)

    # compute and plot power, but show just a few frequencies from all in vector
    f_pick = np.arange(0, f_axis.size, max(1, int(f_axis.size / 15)))
    plt.pcolor(t_axis, np.arange(f_axis.size), to_show)
    ax = plt.gca()
    ax.set_yticks(0.5 + f_pick)
    ax.set_yticklabels([str(int(f * 100) / 100) for f in f_axis[f_pick]])

    # cone-of-influence
    if t_coi is not None:
        # t_coi = (show_coi_bandwidth * 4) / 2 / np.pi * np.sqrt(2) / f_axis
        plt.plot(t_axis[0] + t_coi, np.arange(f_axis.size), 'w-')
        plt.plot(t_axis[-1] - t_coi, np.arange(f_axis.size), 'w-')

    # labeling
    plt.xlabel('time t')
    plt.ylabel('frequency f')
    plt.colorbar()

    return



# properties of time series: length, sampling rate...
t_max = 3
dt_bin = 0.0025
n_bins = np.ceil(t_max / dt_bin)
t = dt_bin * np.arange(n_bins)
n_freqs = 200

# a chirp with decreasing frequency!
f_max = 42
f_min = 4.2
a_sin = 3.2
a_ofs = 2.1
f = f_min + (f_max - f_min) * (1 - t / t_max)
dphi = f * 2 * np.pi * dt_bin
signal = a_ofs + a_sin * np.sin(np.cumsum(dphi))  # + 14 * np.random.random(int(n_bins))

# ...just show the signal
plt.figure(1)
plt.plot(t, signal)
plt.xlabel('time t')
plt.ylabel('signal s(t)')
plt.show()


# %% [markdown]
# ### Tasks
#
# 1. Compute and display the power spectrum of the signal, using the Fourier transform

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
# 2. Compute the wavelet transform over a convenient frequency range, and display the power or amplitude spectrum


#Wavelet Transform in Python

complex_spectrum, t_axis, freq_axis_wavelets, t_coi = wavelet_transform_morlet(signal, n_freqs, freq_min=2, freq_max= 100, dt=dt_bin, bandwidth = 1.5)



# Invoking the complex morlet wavelet object
#mother = "cmor1.5-1.0"
#mother = pywt.ContinuousWavelet('cmor1.5-1.0')

#widths = mother.upper_bound - mother.lower_bound

#fs = 1 / dt_bin

#frequencies = np.array([100, 50, 33.33333333, 25]) / fs # normalize
#scale = pywt.frequency2scale('cmor1.5-1.0', frequencies)
#scale

# ...applied with the parameters we want:
#cwtmatr, freqs = pywt.cwt(
#    pwsig, mscales, mother, dt_bin
#)


#cwtmatr = np.abs(cwtmatr[:-1, :-1])


# **Revised Plotting Using imshow**
fig, ax = plt.subplots(figsize=(10, 6))

# Use imshow with proper aspect ratio and interpolation
extent = [t[0], t[-1], f[-1], f[0]]  # Ensures correct orientation
ax.imshow(complex_spectrum, aspect='auto', extent=extent, cmap='jet', origin='lower')

ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")
ax.set_title("Continuous Wavelet Transform (Scaleogram)")
ax.set_yscale("log")  # Log scale for better frequency visualization
plt.colorbar(ax.imshow(complex_spectrum, aspect='auto', extent=extent, cmap='jet', origin='lower'), ax=ax)

plt.show()

# %% 


#wavelet_dsignal_show(complex_spectrum, t_axis, freq_axis_wavelets, t_coi)




# 3. Compare the two spectra: what does the wavelet spectrum tell you what the 'normal' power spectrum can not do?
#
# 4. Add the cone-of-influence to the wavelet spectrum, and arrange the frequency axis logarithmically. Are the 'edges' of the signal with the typical artefacts from the Fourier/wavelet transforms 'wrapping around' the signal borders indeed masked by the cone? 
#
# 5. Try also to add some noise to the signal and observe the changes in the spectra.
#

# %%
# YOUR CODE...
