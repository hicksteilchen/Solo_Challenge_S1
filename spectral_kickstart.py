'''
spectral.py

Provides tools for spectral analysis

Version 1.5 - adapted for ANDA 2024 / adapted for S1 course 2025
Date 07.02.2025
'''

# %%
import numpy as np
import pywt  #type: ignore
import matplotlib.pyplot as plt
import scipy  #type:ignore


def hilbert_bandpassed(
        signal: np.ndarray,
        f_mid: float,
        f_width: float,
        dt: float
) -> tuple[np.ndarray, np.ndarray]:

    # we design a 3-pole lowpass filter at 0.1x Nyquist frequency
    nyqf = 0.5 / dt
    b, a = scipy.signal.butter(3, [(f_mid - f_width) / nyqf, (f_mid + f_width) / nyqf], 'band')
    filtered = scipy.signal.filtfilt(b, a, signal, method='gust', axis=-1)
    analytic = scipy.signal.hilbert(filtered, axis=-1)

    return analytic, filtered


def phase_locking_value(signal_a: np.ndarray, signal_b: np.ndarray) -> np.ndarray:

    phase_a = signal_a / np.abs(signal_a)
    mphase_b = np.conj(signal_b) / np.abs(signal_b)
    plv = np.mean(phase_a * mphase_b)

    return plv


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


def coherence(
        a1: np.ndarray,
        a2: np.ndarray,
        tau_max: float,
        ntau: float,
        dt: float,
        ts: float | None = None,
        te: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """a1 and a2 are (trials, freqs, time) arrays, ts and te time start/end indices
       tau_max is the maximal time delay (dimension time)
       returns C of shape (freqs, taus) where taus is of
       lenght 2*ntau+1 linearly from -tau_max to +tau_max
        """

    ntrials, nfreqs, ntime = a1.shape
    assert a2.shape == a1.shape

    if ts is None:
        ts = 0
    if te is None:
        te = ntime

    taus = np.linspace(-tau_max, tau_max, 2 * ntau + 1)
    c = np.zeros((nfreqs, len(taus)))

    a2_conj = np.conj(a2)
    a1_abs2 = np.abs(a1) ** 2
    a2_abs2 = np.abs(a2) ** 2

    # zero time delay at index ntau
    c[:, ntau] = np.abs(np.sum(a1[:, :, ts:te] * a2_conj[:, :, ts:te], axis=(0, 2))) ** 2 / \
        np.sum(a1_abs2[:, :, ts:te], axis=(0, 2)) / np.sum(a2_abs2[:, :, ts:te], axis=(0, 2))

    for itau in range(1, ntau + 1):
        tau = taus[ntau + itau]  # absolute tau value
        taui = int(tau / dt)  # index shift by tau

        # shift by +tau
        c[:, ntau + itau] = np.abs(np.sum(a1[:, :, (ts + taui):te] * a2_conj[:, :, ts:(te - taui)], axis=(0, 2))) ** 2 / \
            np.sum(a1_abs2[:, :, (ts + taui):te], axis=(0, 2)) / np.sum(a2_abs2[:, :, ts:(te - taui)], axis=(0, 2))

        # shift by -tau -> switch t boundaries
        c[:, ntau - itau] = np.abs(np.sum(a1[:, :, ts:(te - taui)] * a2_conj[:, :, (ts + taui):te], axis=(0, 2))) ** 2 / \
            np.sum(a1_abs2[:, :, ts:(te - taui)], axis=(0, 2)) / np.sum(a2_abs2[:, :, (ts + taui):te], axis=(0, 2))            

    return c, taus


def power(signal: np.ndarray, dt_bin: float) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the power spectrum of a signal

    Parameters:
    signal: one-dimensional vector with signal samples
    dt_bin: sampling interval in seconds

    Returns:
    signal_power: one-dimensional vector with signal power at different frequencies
    f_axis: 1D-vector, corresponding frequencies for each entry in signal_power
    df: frequency resolution (scalar)
    """

    # YOUR CODE...

    return signal_power, f_axis, df


def power_average(signal: np.ndarray, dt_bin: float, n_average: float) -> tuple[np.ndarray, np.ndarray, float]:

    n_bins_total = signal.shape[-1]
    n_bins = n_bins_total // n_average
    assert n_bins > 0, "Signal has too few bins for averaging!"

    shape_chunks = signal.shape[:-1] + (n_average, n_bins)
    signal_chunks = np.reshape(signal[..., :n_bins * n_average], shape_chunks)

    signal_power_chunks, f_axis, df = power(signal_chunks, dt_bin)
    signal_power = signal_power_chunks.mean(axis=-2)

    return signal_power, f_axis, df


if __name__ == "__main__":

    print("Computing an example!")

    t_max = 3
    f_sin = 42
    dt_bin = 0.0025
    n_bins = np.ceil(t_max / dt_bin)
    a_sin = 3.2
    a_ofs = 2.1

    t = dt_bin * np.arange(n_bins)
    signal = a_ofs + a_sin * np.sin(2 * np.pi * f_sin * t)

    signal_power, f_axis, df = power(signal, dt_bin)

    plt.plot(f_axis, signal_power)
    plt.xlabel("frequency f [Hz]")
    plt.ylabel("spectral power [1/Hz]")
    plt.show()

    print("Checking the Tafelrunde:")
    print(f"var={np.var(signal):.3f}, int={np.sum(signal_power[1:])*df:.3f}")
    print(f"mean={np.mean(signal):.3f}, zeropow={np.sqrt(signal_power[0]*df):.3f}")

    complex_spectrum, freq_axis_wavelets, time_axis_wavelet, coi = wavelet_transform_morlet(
        signal, n_freqs=100, freq_min=f_sin / 2, freq_max=f_sin * 2, dt=dt_bin,
        bandwidth=1.5)

    wavelet_dsignal_show(np.abs(complex_spectrum), freq_axis_wavelets, time_axis_wavelet, coi)
    plt.show()

    # plt.imshow(abs(complex_spectrum) ** 2, cmap="hot", aspect="auto", interpolation="None")
    # plt.colorbar()

# %%
