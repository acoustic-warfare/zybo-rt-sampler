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

try:
    from lib.directions import calculate_coefficients, active_microphones, compute_convolve_h
except:
    print("You must build the directions library")
    exit(1)

steer_offset = 0


from config cimport *

cdef extern from "api.h":
    int load(bint)
    void get_data(float *signals)
    void stop_receiving()
    void pad_mimo(float *image, int *adaptive_array, int n)
    void convolve_mimo_vectorized(float *image, int *adaptive_array, int n)
    void convolve_mimo_naive(float *image, int *adaptive_array, int n)

    void load_coefficients2(int *whole_sample_delay, int n)
    void mimo_truncated(float *image, int *adaptive_array, int n)

    void miso_steer_listen(float *out, int *adaptive_array, int n, int steer_offset)
    # void miso_steer_listen2(int *adaptive_array, int n, int steer_offset)

def connect(replay_mode: bool = False, verbose=True) -> None:
    """
    Connect to a Zybo data-stream

    [NOTICE]

    You must remember to disconnect after you are done, to let the internal c child process terminate
    safely.

    Args:
        replay_mode     bool    True for using replay mode everything else or nothing
                                will result in using real data

    Kwargs:
        verbose         bool    If you want to display terminal output or not

    """
    assert isinstance(replay_mode, bool), "Replay mode must be either True or False"

    if load(replay_mode * 1) == -1:
        print("Wrong FPGA protocol data format received, disconnecting")
        disconnect()

    if verbose:
        print("Receiver process is forked.\nContinue your program!\n")


def disconnect():
    """
    Disconnect from a stream

    This is done by killing the child receiving process
    remember to call this function before calling 'exit()'
    
    """
    stop_receiving()


def receive(signals: np.ndarray[N_MICROPHONES, N_SAMPLES]) -> None:
    """
    Receive the N_SAMPLES latest samples from the Zybo.

    [NOTICE]

    It is important to have the correct datatype and shape as defined in src/config.json

    Usage:

        >>>data = np.empty((N_MICROPHONES, N_SAMPLES), dtype=np.float32)
        >>>receive(data)

    Args:
        signals     np.ndarray The array to be filled with the latest microphone data
    
    """
    assert signals.shape == (N_MICROPHONES, N_SAMPLES), "Arrays do not match shape"
    assert signals.dtype == np.float32, "Arrays dtype do not match"

    cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] sig = np.ascontiguousarray(signals)
    
    get_data(&sig[0, 0])

# Exposing all beamforming algorithms in C
cdef extern from "algorithms/pad_and_sum.h":
    void load_coefficients_pad(int *whole_samples, int n)
    void unload_coefficients_pad()
    void pad_delay(float *signal, float *out, int pos_pad)
    void miso_pad(float *signals, float *out, int *adaptive_array, int n, int offset)
    void mimo_pad(float *signals, float *image, int *adaptive_array, int n)


cdef extern from "algorithms/convolve_and_sum.h":
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



cdef _convolve_coefficients_load(h):
    cdef np.ndarray[float, ndim=4, mode="c"] f32_h = np.ascontiguousarray(h)
    load_coefficients_convolve(&f32_h[0, 0, 0, 0], int(h.size))


from multiprocessing import JoinableQueue, Process, Value

cdef void api(q: JoinableQueue, running: Value):
    whole_samples, fractional_samples = calculate_coefficients()
    active_mics, n_active_mics = active_microphones()

    cdef np.ndarray[int, ndim=1, mode="c"] active_micro = np.ascontiguousarray(active_mics.astype(np.int32))

    cdef np.ndarray[int, ndim=3, mode="c"] i32_whole_samples

    i32_whole_samples = np.ascontiguousarray(whole_samples.astype(np.int32))

    # Pass int pointer to C function
    load_coefficients_pad(&i32_whole_samples[0, 0, 0], whole_samples.size)

    x = np.zeros((MAX_RES_X, MAX_RES_Y), dtype=DTYPE_arr)

    cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] mimo_arr
    mimo_arr = np.ascontiguousarray(x)

    while running.value:
        pad_mimo(&mimo_arr[0, 0], &active_micro[0], int(n_active_mics))
        q.put(mimo_arr)

    # q.join()


    unload_coefficients_pad()

cdef void api_convolve(q: JoinableQueue, running: Value):

    cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] mimo_arr
    
    image = np.zeros((MAX_RES_X, MAX_RES_Y), dtype=DTYPE_arr)
    mimo_arr = np.ascontiguousarray(image)

    h = compute_convolve_h()
    active_mics, n_active_mics = active_microphones()

    cdef np.ndarray[int, ndim=1, mode="c"] active_micro = np.ascontiguousarray(active_mics.astype(np.int32))

    _convolve_coefficients_load(h)

    while running.value:
        convolve_mimo_vectorized(&mimo_arr[0, 0], &active_micro[0], int(n_active_mics))
        q.put(mimo_arr)

    unload_coefficients_convolve()


cdef void api_old(q: JoinableQueue, running: Value):
    whole_samples, fractional_samples = calculate_coefficients()
    active_mics, n_active_mics = active_microphones()

    cdef np.ndarray[int, ndim=1, mode="c"] active_micro = np.ascontiguousarray(active_mics.astype(np.int32))

    cdef np.ndarray[int, ndim=3, mode="c"] i32_whole_samples

    print(len(whole_samples), whole_samples.shape, whole_samples.size)

    i32_whole_samples = np.ascontiguousarray(whole_samples.astype(np.int32))

    # Pass int pointer to C function
    load_coefficients2(&i32_whole_samples[0, 0, 0], whole_samples.size)

    x = np.zeros((MAX_RES_X, MAX_RES_Y), dtype=DTYPE_arr)

    cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] mimo_arr
    mimo_arr = np.ascontiguousarray(x)

    #bf.connect()

    while running.value:
        mimo_truncated(&mimo_arr[0, 0], &active_micro[0], int(n_active_mics))
        q.put(mimo_arr)

def steer(azimuth: float, elevation: float):
    """Steer a MISO into a specific direction"""
    assert -90<=azimuth<=90, "Invalid range"
    assert -90<=elevation<=90, "Invalid range"

    azimuth += 90
    azimuth /= 180
    azimuth = int(azimuth * MAX_RES_X)
    elevation += 90
    elevation /= 180
    elevation = int(elevation * MAX_RES_Y)

    _, n_active_mics = active_microphones()

    global steer_offset
    steer_offset = elevation * MAX_RES_X * n_active_mics + azimuth * n_active_mics

cdef void api_miso(q: JoinableQueue, running: Value):
    cdef np.ndarray[np.float32_t, ndim=1, mode = 'c'] out = np.ascontiguousarray(np.zeros(N_SAMPLES, dtype=DTYPE_arr))
    
    whole_samples, fractional_samples = calculate_coefficients()
    active_mics, n_active_mics = active_microphones()

    cdef np.ndarray[int, ndim=1, mode="c"] active_micro = np.ascontiguousarray(active_mics.astype(np.int32))

    cdef np.ndarray[int, ndim=3, mode="c"] i32_whole_samples

    i32_whole_samples = np.ascontiguousarray(whole_samples.astype(np.int32))

    # Pass int pointer to C function
    load_coefficients_pad(&i32_whole_samples[0, 0, 0], whole_samples.size)

    while running.value:
        global steer_offset
        miso_steer_listen(&out[0], &active_micro[0], int(n_active_mics), steer_offset)
        q.put(out)

# cdef void api_miso2(running: Value):
#     cdef np.ndarray[np.float32_t, ndim=1, mode = 'c'] out = np.ascontiguousarray(np.zeros(N_SAMPLES, dtype=DTYPE_arr))
    
#     whole_samples, fractional_samples = calculate_coefficients()
#     active_mics, n_active_mics = active_microphones()

#     cdef np.ndarray[int, ndim=1, mode="c"] active_micro = np.ascontiguousarray(active_mics.astype(np.int32))

#     cdef np.ndarray[int, ndim=3, mode="c"] i32_whole_samples

#     i32_whole_samples = np.ascontiguousarray(whole_samples.astype(np.int32))

#     # Pass int pointer to C function
#     load_coefficients_pad(&i32_whole_samples[0, 0, 0], whole_samples.size)

#     steer(0, 0)

#     while running.value:
#         global steer_offset
#         miso_steer_listen2(&active_micro[0], int(n_active_mics), steer_offset)



# Web interface
def uti_api(q: JoinableQueue, running: Value):
    api(q, running)

def conv_api(q: JoinableQueue, running: Value):
    api_convolve(q, running)

def miso_api(q: JoinableQueue, running: Value):
    api_miso(q, running)

# def miso_api2(running: Value):
#     api_miso2(running)

# Testing

def main():
    jobs = 1
    q = JoinableQueue(maxsize=2)

    v = Value('i', 1)

    from lib.visual import Viewer

    consumer = Viewer().loop

    connect()


    try:

        producers = [
            Process(target=api, args=(q, v))
            for _ in range(jobs)
        ]

        # daemon=True is important here
        consumers = [
            Process(target=consumer, args=(q, v), daemon=True)
            for _ in range(jobs * 1)
        ]

        # + order here doesn't matter
        for p in consumers + producers:
            p.start()

        for p in producers:
            p.join()



    finally:

        v.value = 0

        disconnect()


