# cython: language_level=3
# distutils: language=c

import numpy as np
cimport numpy as np

# It's necessary to call "import_array" if you use any part of the numpy PyArray_* API.
np.import_array()

import sys
sys.path.insert(0, "") # Access local modules located in . Enables 'from . import MODULE'


cdef extern from "play.h":
    int start_mic()
    int start_mic2(int *adaptive_array, int n)


cdef void start2():
    try:
        from lib.directions import calculate_coefficients, active_microphones, compute_convolve_h
    except:
        print("You must build the directions library")
        exit(1)

    active_mics, n_active_mics = active_microphones()

    cdef np.ndarray[int, ndim=1, mode="c"] active_micro = np.ascontiguousarray(active_mics.astype(np.int32))
    start_mic2(&active_micro[0], int(n_active_mics))


def start():
    # start_mic()
    start2()