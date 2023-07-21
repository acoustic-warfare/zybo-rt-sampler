# cython: language_level=3
# distutils: language=c

import numpy as np
cimport numpy as np

# It's necessary to call "import_array" if you use any part of the numpy PyArray_* API.
np.import_array()

import sys
sys.path.insert(0, "") # Access local modules located in . Enables 'from . import MODULE'

from config cimport *

def calc_r_prime(d):
    half = d/2
    r_prime = np.zeros((2, COLUMNS * ROWS))
    element_index = 0
    for array in range(ACTIVE_ARRAYS):
        for row in range(ROWS):
            for col in range(COLUMNS):
                r_prime[0,element_index] = col * d + half + array*COLUMNS*d + array*ARRAY_SEPARATION - COLUMNS * ACTIVE_ARRAYS * half
                r_prime[1, element_index] = row * d - ROWS * half + half
                element_index += 1
    r_prime[0,:] -= ACTIVE_ARRAYS*ARRAY_SEPARATION/2
    active_mics, n_active_mics = active_microphones()

    r_prime = r_prime[:,active_mics]
    return r_prime

def active_microphones():
    mode = SKIP_N_MICS
    rows = np.arange(0, ROWS, mode)
    columns = np.arange(0, COLUMNS*ACTIVE_ARRAYS, mode)

    arr_elem = ROWS*COLUMNS                       # elements in one array
    mics = np.linspace(0, arr_elem-1, arr_elem)   # mics in one array
    
    microphones = np.linspace(0, arr_elem-1,arr_elem).reshape((ROWS, COLUMNS))

    for a in range(ACTIVE_ARRAYS-1):
        a += 1
        array = mics[0+a*arr_elem : arr_elem+a*arr_elem].reshape((ROWS, COLUMNS))
        microphones = np.hstack((microphones, array))

    active_mics = []
    for r in rows:
        for c in columns:
            mic = microphones[r,c]
            active_mics.append(int(mic))
    return np.sort(active_mics), len(active_mics)

def calculate_delays():
    c = PROPAGATION_SPEED             # from config
    fs = SAMPLE_RATE          # from config
    #N_SAMPLES = N_SAMPLES    # from config
    d = ELEMENT_DISTANCE            # distance between elements, from config

    alpha = VIEW_ANGLE  # total scanning angle (bildvinkel) in theta-direction [degrees], from config
    z_scan = 10  # distance to scanning window, from config

    x_res = MAX_RES_X  # resolution in x, from config
    y_res = MAX_RES_Y  # resolution in y, from config
    AS = 16/9   # aspect ratio, from config

    # Calculations for time delay starts below
    r_prime = calc_r_prime(d)  # matrix holding the xy positions of each microphone
    x_i = r_prime[0,:]                      # x-coord for microphone i
    y_i = r_prime[1,:]                      # y-coord for microphone i

    # outer limits of scanning window in xy-plane
    x_scan_max = z_scan*np.tan((alpha/2)*np.pi/180)
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
    print(samp_delay.shape)
    return samp_delay
    # active_mics: holds index of active microphones for a specific mode
    # n_active_mics: number of active microphones
    active_mics, n_active_mics = active_microphones()

def calculate_delays_():

    distance = 0.02

    samp_delay = np.zeros((MAX_RES_X, MAX_RES_Y, COLUMNS*ROWS*ACTIVE_ARRAYS), dtype=np.float32)

    for xi, x in enumerate(np.linspace(-MAX_ANGLE, MAX_ANGLE, MAX_RES_X)):
        azimuth = x * -np.pi / 180.0
        x_factor = np.sin(azimuth)
        for yi , y in enumerate(np.linspace(-MAX_ANGLE, MAX_ANGLE, MAX_RES_Y)):
            elevation = y * -np.pi / 180.0
            y_factor = np.sin(elevation)

            smallest = 0

            for row in range(ROWS):
                for col in range(COLUMNS):
                    half = distance / 2.0
                    tmp_col = col * distance - COLUMNS * half + half
                    tmp_row = row * distance - ROWS * half + half

                    tmp_delay = tmp_col * x_factor + tmp_row * y_factor
                    if (tmp_delay < smallest):
                        smallest = tmp_delay

                    samp_delay[xi, yi, row * COLUMNS + col] = tmp_delay

            samp_delay[xi, yi, :] -= smallest

    samp_delay *= SAMPLE_RATE / PROPAGATION_SPEED

    return samp_delay

def get_h(delay, N=8):
    tau = - delay  # Fractional delay [samples].
    epsilon = 1e-9
    n = np.arange(N)

    sinc = n - (8 - 1) / 2 - (0.5 + tau) + epsilon

    h = np.sin(sinc*np.pi)/(sinc*np.pi)

    blackman = 0.42 - 0.5 * np.cos(2*np.pi * n / 8) + 0.08 * np.cos(4 * np.pi * n / 8)

    h *= blackman
    
    # Normalize to get unity gain.
    h /= np.sum(h)

    return h

def get_h2(delay, N=64):
    
    epsilon = 1e-9

    tau = 0.5 - delay + epsilon # Fractional delay [samples].
    h = np.zeros(N, dtype=np.float32)

    sum_ = 0
    for i in range(N):
        hi = i - (N - 1) / 2 - tau
        hi = np.sin(hi*np.pi) / (hi * np.pi)
        n = i * 2 - N + 1

        black = 0.42 + 0.5 * np.cos(np.pi * n / (N - 1 + epsilon)) + 0.08 * np.cos(2 * np.pi * n / (N - 1 + epsilon))
        hi *= black
        sum_ += hi
        h[i] = hi

    h /= sum_
    return h


def compute_convolve_h():
    # delays = calculate_delays_()

    # h = np.zeros((MAX_RES_X, MAX_RES_Y, ROWS * COLUMNS, N_TAPS), dtype=np.float32)

    samp_delay = calculate_delays()

    print(samp_delay.shape)

    h = np.zeros((*samp_delay.shape, N_TAPS), dtype=np.float32)

    #h = np.zeros((MAX_RES_X, MAX_RES_Y, COLUMNS*ROWS*ACTIVE_ARRAYS, 8), dtype=np.float32)

    
    for y in range(MAX_RES_Y):
        for x in range(MAX_RES_X):
            for i in range(samp_delay.shape[2]):
                h[x, y, i] = get_h2(samp_delay[x, y, i], N=N_TAPS)
    return np.float32(h)
    # for y in range(MAX_RES_Y):
    #     for x in range(MAX_RES_X):
    #         for i in range(64):
    #             # print
    #             h[x, y, i] = get_h(delays[x, y, i], N=N_TAPS)

    # return np.float32(h)





def calculate_coefficients():

    samp_delay = calculate_delays()

    #whole_sample_delay = samp_delay.astype(np.int32)
    whole_sample_delay = samp_delay.astype(int)
    fractional_sample_delay = samp_delay - whole_sample_delay

    h = np.zeros((*samp_delay.shape, 8), dtype=np.float32)

    #h = np.zeros((MAX_RES_X, MAX_RES_Y, COLUMNS*ROWS*ACTIVE_ARRAYS, 8), dtype=np.float32)

    for x in range(MAX_RES_X):
        for y in range(MAX_RES_Y):
            for i in range(samp_delay.shape[2]):
                h[x, y, i] = get_h(fractional_sample_delay[x, y, i])
    
    return whole_sample_delay, h
