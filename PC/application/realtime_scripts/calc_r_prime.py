import numpy as np
import matplotlib.pyplot as plt
import config
import realtime_scripts.active_microphones as am

from matplotlib import rc # only for plot
camera_offset = 0.11    # [m]

def calc_r_prime(d):
    half = d/2
    r_prime = np.zeros((2, config.N_MICROPHONES))
    element_index = 0
    for array in range(config.ACTIVE_ARRAYS):
        array *= -1
        for row in range(config.rows):
            for col in range(config.columns):
                r_prime[0,element_index] = -col * d - half + array*config.columns*d + array*config.ARRAY_SEPARATION + config.columns* config.ACTIVE_ARRAYS * half
                r_prime[1, element_index] = row * d - config.rows * half + half - camera_offset
                element_index += 1
    r_prime[0,:] += (config.ACTIVE_ARRAYS-1)*config.ARRAY_SEPARATION/2
    active_mics = am.active_microphones()
    
    r_prime_all = r_prime
    r_prime = r_prime[:,active_mics]


    
    return r_prime_all, r_prime

