import numpy as np
import matplotlib.pyplot as plt
from interface import config

from lib.tests import pad_delay_wrapper, mimo_pad_wrapper, mimo_convolve_wrapper, mimo_lerp_wrapper, mimo_hybrid_convolve_wrapper
from lib.directions import calculate_coefficients, calculate_delays

def generate_sig(frequency):
    start_time = 0
    end_time = 1
    sample_rate = config.fs
    time = np.arange(start_time, end_time, 1/sample_rate)
    theta = 0
    amplitude = 1
    sinewave = amplitude * np.sin(2 * np.pi * frequency * time + theta)[:config.N_SAMPLES]
    return sinewave


def generate_signals_all(signal: np.ndarray):
    signals = np.repeat(signal, config.N_MICROPHONES, axis=0).reshape((config.N_SAMPLES, config.N_MICROPHONES)).T
    return np.float32(signals)

signal = generate_sig(8000)

signals = generate_signals_all(signal)

a = mimo_pad_wrapper(signals)
plt.imshow(a.T)
plt.show()


c = mimo_lerp_wrapper(signals)
plt.imshow(c.T)
plt.show()

d = mimo_hybrid_convolve_wrapper(signals)
plt.imshow(d.T)

plt.show()