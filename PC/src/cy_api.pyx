# cython: language_level=3
# distutils: language=c

# File name: cy_api.pyx
# Description: Cython Module With Added NumPy Support For beamformer
# Author: Irreq
# Date: 2022-12-29

# Main API for C functions

from .utils import *

import math
#import numpy as np

import numpy as np
cimport numpy as np

# It's necessary to call "import_array" if you use any part of the numpy PyArray_* API.
np.import_array()
import config

def calc_r_prime(d):
    half = d/2
    r_prime = np.zeros((2, config.rows*config.columns))
    element_index = 0
    for array in range(config.ACTIVE_ARRAYS):
        for row in range(config.rows):
            for col in range(config.columns):
                r_prime[0,element_index] = col * d + half + array*config.columns*d + array*config.ARRAY_SEPARATION - config.columns* config.ACTIVE_ARRAYS * half
                r_prime[1, element_index] = row * d - config.rows * half + half
                element_index += 1
    r_prime[0,:] -= config.ACTIVE_ARRAYS*config.ARRAY_SEPARATION/2
    active_mics, n_active_mics = active_microphones()

    r_prime = r_prime[:,active_mics]
    return r_prime

def active_microphones():
    mode = config.SKIP_N_MICS
    rows = np.arange(0, config.rows, mode)
    columns = np.arange(0, config.columns*config.ACTIVE_ARRAYS, mode)

    mics = np.linspace(0, config.rows*config.columns-1, config.rows*config.columns)   # mics in one array
    arr_elem = config.rows*config.columns                       # elements in one array
    microphones = np.linspace(0, config.rows*config.columns-1,config.rows*config.columns).reshape((config.rows, config.columns))

    for a in range(config.ACTIVE_ARRAYS-1):
        a += 1
        array = mics[0+a*arr_elem : arr_elem+a*arr_elem].reshape((config.rows, config.columns))
        microphones = np.hstack((microphones, array))

    active_mics = []
    for r in rows:
        for c in columns:
            mic = microphones[r,c]
            active_mics.append(int(mic))
    return np.sort(active_mics), len(active_mics)

def calculate_delays():
    c = config.PROPAGATION_SPEED             # from config
    fs = config.SAMPLE_RATE          # from config
    N_SAMPLES = config.N_SAMPLES    # from config
    d = config.ELEMENT_DISTANCE            # distance between elements, from config

    alpha = 180  # total scanning angle (bildvinkel) in theta-direction [degrees], from config
    z_scan = 10  # distance to scanning window, from config

    x_res = 31  # resolution in x, from config
    y_res = 11  # resolution in y, from config
    AS = 16/9   # aspect ratio, from config

    # Calculations for time delay starts below
    r_prime = calc_r_prime(d)  # matrix holding the xy positions of each microphone
    x_i = r_prime[0,:]                      # x-coord for microphone i
    y_i = r_prime[1,:]                      # y-coord for microphone i

    # outer limits of scanning window in xy-plane
    x_scan_max = z_scan*np.tan((alpha/2)*math.pi/180)
    x_scan_min = - x_scan_max
    y_scan_max = x_scan_max/AS
    y_scan_min = -y_scan_max

    # scanning window
    x_scan = np.linspace(x_scan_min,x_scan_max, x_res).reshape(x_res,1,1)
    y_scan = np.linspace(y_scan_min,y_scan_max,y_res).reshape(1, y_res, 1)
    r_scan = np.sqrt(x_scan**2 + y_scan**2 + z_scan**2) # distance between middle of array to the xy-scanning coordinate

    # calculate time delay (in number of samples)
    samp_delay = (fs/c) * (x_scan*x_i + y_scan*y_i) / r_scan            # with shape: (x_res, y_res, n_active_mics)
    # adjust such that the microphone furthest away from the beam direction have 0 delay
    samp_delay -= np.amin(samp_delay, axis=2).reshape(x_res, y_res, 1)

    return samp_delay
    # active_mics: holds index of active microphones for a specific mode
    # n_active_mics: number of active microphones
    active_mics, n_active_mics = active_microphones()


def calculate_coefficients():

    samp_delay = calculate_delays()

    whole_sample_delay = samp_delay.astype(np.int32)
    fractional_sample_delay = samp_delay - whole_sample_delay

    return whole_sample_delay, fractional_sample_delay


cdef public main():
    cdef int n = n_microphones

    print(calculate_coefficients())

    #for i in range(n):
    #    print(i)

def entrypoint():
    main()
    #print("Working")
    #print(n_microphones)
    #test()