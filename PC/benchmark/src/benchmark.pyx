# cython: language_level=3
# distutils: language=c

import numpy as np
cimport numpy as np

# It's necessary to call "import_array" if you use any part of the numpy PyArray_* API.
np.import_array()

import sys
sys.path.insert(0, "") # Access local modules located in . Enables 'from . import MODULE'

# Create specific data-types "ctypedef" assigns a corresponding compile-time type to DTYPE_t.
ctypedef np.float32_t DTYPE_t

# Constants
DTYPE_arr = np.float32

from config cimport *

try:
    from lib.directions import calculate_coefficients, active_microphones, compute_convolve_h
except:
    print("You must build the directions library")
    # exit(1)

# Exposing all beamforming algorithms in C
cdef extern from "algorithms/pad_and_sum.c":
    void load_coefficients_pad(int *whole_samples, int n)
    void unload_coefficients_pad()
    void pad_delay(float *signal, float *out, int pos_pad)
    void miso_pad(float *signals, float *out, int *adaptive_array, int n, int offset)
    void mimo_pad(float *signals, float *image, int *adaptive_array, int n)

cdef extern from "algorithms/convolve_and_sum.c":
    void convolve_delay_naive_add(float *signal, float *h, float *out)
    void convolve_delay_vectorized(float *signal, float *h, float *out)
    void convolve_delay_vectorized_add(float *signal, float *h, float *out)
    void convolve_delay_naive(float *signal, float *out, float *h)
    void convolve_naive(float *signals, float *out, int *adaptive_array, int n, int offset)
    void mimo_convolve_naive(float *signals, float *image, int *adaptive_array, int n)
    void miso_convolve_vectorized(float *signals, float *out, int *adaptive_array, int n, int offset)
    void mimo_convolve_vectorized(float *signals, float *image, int *adaptive_array, int n)
    void load_coefficients_convolve(float *h, int n)
    void unload_coefficients_convolve()
    

# Exporting functions

cdef void _load_coefficients_pad(np.ndarray whole_samples):
    active_mics, n_active_mics = active_microphones()

    cdef np.ndarray[int, ndim=1, mode="c"] active_micro = np.ascontiguousarray(active_mics.astype(np.int32))

    cdef np.ndarray[int, ndim=3, mode="c"] i32_whole_samples

    i32_whole_samples = np.ascontiguousarray(whole_samples.astype(np.int32))

    # Pass int pointer to C function
    load_coefficients_pad(&i32_whole_samples[0, 0, 0], whole_samples.size)

def pad_coefficients_load(whole_samples, n):
    _load_coefficients_pad(whole_samples)

cdef np.ndarray _pad_delay(signal, out, pos_pad):
    cdef np.ndarray[np.float32_t, ndim=1, mode = 'c'] _signal = np.ascontiguousarray(signal)
    cdef np.ndarray[np.float32_t, ndim=1, mode = 'c'] _out = np.ascontiguousarray(out)
    pad_delay(&_signal[0], &_out[0], pos_pad)

    return _out

def pad_delay_wrapper(signal, out, pos_pad) -> np.ndarray:
    return _pad_delay(signal, out, pos_pad)

cdef np.ndarray _mimo_pad(signals):
    cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] _signals = np.ascontiguousarray(signals)
    cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] mimo_arr
    
    image = np.zeros((MAX_RES_X, MAX_RES_Y), dtype=DTYPE_arr)
    mimo_arr = np.ascontiguousarray(image)

    active_mics, n_active_mics = active_microphones()

    # cdef np.ndarray[int, ndim=1, mode="c"] active_micro = np.ascontiguousarray(active_mics.astype(np.int32))

    whole_samples, fractional_samples = calculate_coefficients()
    active_mics, n_active_mics = active_microphones()

    cdef np.ndarray[int, ndim=1, mode="c"] active_micro = np.ascontiguousarray(active_mics.astype(np.int32))

    cdef np.ndarray[int, ndim=3, mode="c"] i32_whole_samples

    i32_whole_samples = np.ascontiguousarray(whole_samples.astype(np.int32))

    # Pass int pointer to C function
    load_coefficients_pad(&i32_whole_samples[0, 0, 0], whole_samples.size)

    mimo_pad(&_signals[0, 0], &mimo_arr[0, 0], &active_micro[0], int(n_active_mics))

    return mimo_arr

def mimo_pad_wrapper(signals):
    return _mimo_pad(signals)


cdef _convolve_coefficients_load(h):
    cdef np.ndarray[float, ndim=4, mode="c"] f32_h = np.ascontiguousarray(h)
    load_coefficients_convolve(&f32_h[0, 0, 0, 0], int(h.size))

def convolve_coefficients_load(h):
    _convolve_coefficients_load(h)

cdef np.ndarray _mimo_convolve(signals):
    cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] _signals = np.ascontiguousarray(signals)
    cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] mimo_arr
    
    image = np.zeros((MAX_RES_X, MAX_RES_Y), dtype=DTYPE_arr)
    mimo_arr = np.ascontiguousarray(image)

    h = compute_convolve_h();
    active_mics, n_active_mics = active_microphones()

    cdef np.ndarray[int, ndim=1, mode="c"] active_micro = np.ascontiguousarray(active_mics.astype(np.int32))

    _convolve_coefficients_load(h)

    mimo_convolve_vectorized(&_signals[0, 0], &mimo_arr[0, 0], &active_micro[0], int(n_active_mics))

    return mimo_arr

def mimo_convolve_wrapper(signals):
    return _mimo_convolve(signals)



