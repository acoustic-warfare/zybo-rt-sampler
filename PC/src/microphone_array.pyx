# cython: language_level=3
# distutils: language=c

#TODO Implement adaptive beamforming properly

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

from lib.directions import calculate_coefficients, active_microphones

from lib.VideoPlayer import Viewer

from config cimport *

WINDOW_DIMENSIONS = (APPLICATION_WINDOW_WIDTH, APPLICATION_WINDOW_HEIGHT)

# C defined functions
cdef extern from "beamformer.h":
    void load_coefficients(int *whole_sample_delay)
    void work_test(float *image)
    int load(bint)
    void myread(float *signal)
    void signal_handler()
    void kill_child()
    void mimo(float *image)
    void mimo_truncated(float *image, int *adaptive_array, int n)


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

    if replay_mode: # True
        if load(1) == -1:
            print("Wrong FPGA protocol data format received, disconnecting")
            disconnect()
    else: # Default for real data
        if load(0) == -1:
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
    kill_child()

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
    myread(&sig[0, 0])


# Truncate and sum beamformer for MIMO application
cdef void trunc_mimo(src, replay=False):

    whole_samples, fractional_samples = calculate_coefficients()
    active_mics, n_active_mics = active_microphones()

    cdef np.ndarray[int, ndim=1, mode="c"] active_micro = np.ascontiguousarray(active_mics.astype(np.int32))

    cdef np.ndarray[int, ndim=3, mode="c"] i32_whole_samples

    i32_whole_samples = np.ascontiguousarray(whole_samples.astype(np.int32))

    # Pass int pointer to C function
    load_coefficients(&i32_whole_samples[0, 0, 0])

    x = np.zeros((MAX_RES_X, MAX_RES_Y), dtype=DTYPE_arr)

    cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] mimo_arr
    mimo_arr = np.ascontiguousarray(x)

    connect(replay)
    v = Viewer(src, False, replay)

    while True:
        mimo_truncated(&mimo_arr[0, 0], &active_micro[0], int(n_active_mics))

        v.show(mimo_arr)
    
    disconnect() # TODO will never disconnect except: Ctrl-C
        

# TODO axis are inverted, x must be -x and so on
cdef void convolve_mimo(src, replay=False):

    v = Viewer(src, True, replay)
    connect(replay)

    x = np.zeros((MAX_RES, MAX_RES), dtype=DTYPE_arr)

    cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] mimo_arr
    mimo_arr = np.ascontiguousarray(x)

    while True:
        mimo(&mimo_arr[0, 0])
        v.show(mimo_arr.T)

    disconnect() # TODO will never disconnect except: Ctrl-C


def convolve_backend(src, replayMode):
    convolve_mimo(src, replayMode)

def trunc_backend(src, replayMode):
    trunc_mimo(src, replayMode)

