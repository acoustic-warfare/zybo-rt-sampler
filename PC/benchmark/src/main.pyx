# cython: language_level=3
# distutils: language=c

__doc__ = """
This is the Python -> Cython -> C interface used to communicate with the microphone_array,
perform beamforming, start receivers and playbacks.

The reason for all functionality to be located in this file is because the functions inside
needs to communicate with each other in the C scope. Therefore, they won't share variables
if they are located in different runtimes. As the user, you may simply inside your python
interpreter or python program use:

>>> from beamformer import WHAT_YOU_NEED_TO_IMPORT

Most of the functionality can be further explained in src/api.c which this file "sits" on top
of.
"""

import numpy as np
cimport numpy as np

# It's necessary to call "import_array" if you use any part of the numpy PyArray_* API.
np.import_array()

import sys
# Access local modules located in . Enables 'from . import MODULE'
sys.path.insert(0, "") 

# Create specific data-types "ctypedef" assigns a corresponding compile-time type to DTYPE_t.
ctypedef np.float32_t DTYPE_t

# Constants
DTYPE_arr = np.float32

try:
    from lib.directions import calculate_coefficients, active_microphones, compute_convolve_h, calculate_delay_miso, calculate_delays
except:
    print("You must build the directions library")
    sys.exit(1)

# Import configuration variables from config.pxd <- config.h
from config cimport *

# API must contain all C functions that needs IPC
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
    int load_miso()
    void load_pa(int *adaptive_array, int n)
    void stop_miso()
    void steer(int offset)

# Exposing all pad and sum beamforming algorithms in C
cdef extern from "algorithms/pad_and_sum.h":
    void load_coefficients_pad(int *whole_samples, int n)
    void load_coefficients_pad2(int *whole_miso, int n)
    void unload_coefficients_pad()
    void unload_coefficients_pad2()
    void pad_delay(float *signal, float *out, int pos_pad)
    void miso_pad(float *signals, float *out, int *adaptive_array, int n, int offset)
    void mimo_pad(float *signals, float *image, int *adaptive_array, int n)

# Exposing all convolve and sum beamforming algorithms in C
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

# Exposing all convolve and sum beamforming algorithms in C
cdef extern from "algorithms/lerp_and_sum.h":
    void lerp_delay(float *signal, float *out, float h, int pad)
    void miso_lerp(float *signals, float *out, int *adaptive_array, int n, int offset)
    void load_coefficients_lerp(float *delays, int n)
    void unload_coefficients_lerp()


# ---- BEGIN LIBRARY FUNCTIONS ----

def connect(replay_mode: bool = False, verbose=True) -> None:
    """
    Connect to a Zybo data-stream

    [NOTICE]

    You must remember to disconnect after you are done, to let the internal C
    child process terminate safely.

    Args:
        replay_mode     bool    True for using replay mode everything 
                                else or nothing will result in using real data

    Kwargs:
        verbose         bool    If you want to display terminal output or not

    """
    assert isinstance(replay_mode, bool), "Replay mode must be either True or False"

    if load(replay_mode * 1) == -1:
        print("Wrong FPGA protocol data format received, disconnecting")
        disconnect()

    if verbose:
        print("Receiver process is forked.\nContinue your program!\n")


def disconnect() -> None:
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
    This function is "slow" in the regard that is checks if the `signals` is
    of correct data-type and shape, but fine if you only need the latest sample.
 
    It is important to have the correct datatype and shape as defined 
    in src/config.json

    Usage:

        >>>data = np.empty((N_MICROPHONES, N_SAMPLES), dtype=np.float32)
        >>>receive(data)

    Args:
        signals     np.ndarray The array to be filled with the 
                    latest microphone data
    
    """
    assert signals.shape == (N_MICROPHONES, N_SAMPLES), "Arrays do not match shape"
    assert signals.dtype == np.float32, "Arrays dtype do not match"

    cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] sig = np.ascontiguousarray(signals)
    
    get_data(&sig[0, 0])


# ---- BEGIN BEAMFORMING FUNCTIONS ----

from multiprocessing import JoinableQueue, Process, Value


cdef _convolve_coefficients_load(h):
    cdef np.ndarray[float, ndim=4, mode="c"] f32_h = np.ascontiguousarray(h)
    load_coefficients_convolve(&f32_h[0, 0, 0, 0], int(h.size))


cdef void _loop_mimo_pad(q: JoinableQueue, running: Value):
    """Producer loop for MIMO using pad-delay algorithm"""
    
    # Calculating time delay for each microphone and each direction
    cdef np.ndarray[int, ndim=3, mode="c"] i32_whole_samples
    whole_samples, fractional_samples = calculate_coefficients()
    i32_whole_samples = np.ascontiguousarray(whole_samples.astype(np.int32))

    # Pass int pointer to C function
    load_coefficients_pad(&i32_whole_samples[0, 0, 0], whole_samples.size)

    # Finding which microphones to use
    cdef np.ndarray[int, ndim=1, mode="c"] active_micro
    active_mics, n_active_mics = active_microphones()
    active_micro = np.ascontiguousarray(active_mics.astype(np.int32))

    # Setting up output buffer
    cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] power_map
    _power_map = np.zeros((MAX_RES_X, MAX_RES_Y), dtype=DTYPE_arr)
    power_map = np.ascontiguousarray(_power_map)

    while running.value:
        pad_mimo(&power_map[0, 0], &active_micro[0], int(n_active_mics))
        q.put(power_map)
    
    # Unload when done
    unload_coefficients_pad()

cdef void _loop_miso_pad(q: JoinableQueue, running: Value):
    """Consumer loop for MISO using pad-delay algorithm"""

    # Calculating time delay for each microphone and each direction
    cdef np.ndarray[int, ndim=3, mode="c"] i32_whole_samples
    whole_samples, fractional_samples = calculate_coefficients()
    i32_whole_samples = np.ascontiguousarray(whole_samples.astype(np.int32))

    # Pass int pointer to C function
    load_coefficients_pad(&i32_whole_samples[0, 0, 0], whole_samples.size)

    # Finding which microphones to use
    cdef np.ndarray[int, ndim=1, mode="c"] active_micro
    active_mics, n_active_mics = active_microphones()
    active_micro = np.ascontiguousarray(active_mics.astype(np.int32))

    print("Cython: Starting miso")
    # Setup audio playback (Order is important)
    load_miso()
    n_active_mics = 64
    print("Cython: enabling microphones")
    load_pa(&active_micro[0], int(n_active_mics))

    print("Cython: Steering beam")
    steer_cartesian_degree(0, 0) # Listen at zero bearing

    while running.value:
        try:
            (x, y) = q.get()
            q.task_done()
            steer4(x, y)
        except Exception as e:
            print(e)

    print("Cython: Stopping audio playback")
    stop_miso()
    unload_coefficients_pad()


cdef void _loop_miso_lerp(q: JoinableQueue, running: Value):
    # Calculating time delay for each microphone and each direction
    cdef np.ndarray[float, ndim=3, mode="c"] f32_fractional_samples
    fractional_samples = calculate_delays()

    f32_fractional_samples = np.ascontiguousarray(fractional_samples.astype(np.float32))

    load_coefficients_lerp(&f32_fractional_samples[0, 0, 0], fractional_samples.size)

    # Finding which microphones to use
    cdef np.ndarray[int, ndim=1, mode="c"] active_micro
    active_mics, n_active_mics = active_microphones()
    active_micro = np.ascontiguousarray(active_mics.astype(np.int32))

    print("Cython: Starting miso")
    # Setup audio playback (Order is important)
    load_miso()
    n_active_mics = 64
    print("Cython: enabling microphones")
    load_pa(&active_micro[0], int(n_active_mics))

    print("Cython: Steering beam")
    steer_cartesian_degree(0, 0) # Listen at zero bearing

    while running.value:
        try:
            (x, y) = q.get()
            q.task_done()
            steer4(x, y)
        except Exception as e:
            print(e)

    print("Cython: Stopping audio playback")
    stop_miso()
    unload_coefficients_lerp()

cdef void _loop_mimo_and_miso_pad(q_steer: JoinableQueue, q_out: JoinableQueue, running: Value):
    # Calculating time delay for each microphone and each direction
    cdef np.ndarray[int, ndim=3, mode="c"] i32_whole_samples
    whole_samples, fractional_samples = calculate_coefficients()
    i32_whole_samples = np.ascontiguousarray(whole_samples.astype(np.int32))

    # Pass int pointer to C function
    load_coefficients_pad(&i32_whole_samples[0, 0, 0], whole_samples.size)

    # Finding which microphones to use
    cdef np.ndarray[int, ndim=1, mode="c"] active_micro
    active_mics, n_active_mics = active_microphones()
    active_micro = np.ascontiguousarray(active_mics.astype(np.int32))

    # Setting up output buffer
    cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] power_map
    _power_map = np.zeros((MAX_RES_X, MAX_RES_Y), dtype=DTYPE_arr)
    power_map = np.ascontiguousarray(_power_map)

    print("Cython: Starting miso")
    # Setup audio playback (Order is important)
    load_miso()
    # n_active_mics = 64
    print("Cython: enabling microphones")
    load_pa(&active_micro[0], int(n_active_mics))

    print("Cython: Steering beam")
    steer_cartesian_degree(0, 0) # Listen at zero bearing


    import queue
    while running.value:
        pad_mimo(&power_map[0, 0], &active_micro[0], int(n_active_mics))
        q_out.put(power_map)

        try:
            (x, y) = q_steer.get(block=False)
            q_steer.task_done()
            steer4(x, y)
        except queue.Empty:
            pass
        except Exception as e:
            print(e)
    
    # Unload when done
    stop_miso()
    unload_coefficients_pad()


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
    
    unload_coefficients_pad()

"""
The following functions are producers of data since the only create their coefficients
during call. They are also meant to be run in a separate Process and to be stopped
by the Variable `running`.

All these functions use a queue to put their 


"""

cdef void api_with_miso(q: JoinableQueue, running: Value):
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
    import time
    load_miso()
    # time.sleep(1)
    load_pa(&active_micro[0], int(n_active_mics))
    steer(0)

    steer_cartesian_degree(0, 0)

    while running.value:
        pad_mimo(&mimo_arr[0, 0], &active_micro[0], int(n_active_mics))
        q.put(mimo_arr)

    stop_miso()
    unload_coefficients_pad()

cdef void just_miso(q: JoinableQueue, running: Value):
    whole_samples, fractional_samples = calculate_coefficients()
    active_mics, n_active_mics = active_microphones()

    cdef np.ndarray[int, ndim=1, mode="c"] active_micro = np.ascontiguousarray(active_mics.astype(np.int32))

    cdef np.ndarray[int, ndim=3, mode="c"] i32_whole_samples

    i32_whole_samples = np.ascontiguousarray(whole_samples.astype(np.int32))

    # Pass int pointer to C function
    load_coefficients_pad(&i32_whole_samples[0, 0, 0], whole_samples.size)

    
    load_miso()
    load_pa(&active_micro[0], int(n_active_mics))
    # steer(0) # This will set the offset to zero, which is quite bad

    steer_cartesian_degree(0, 0)

    import time
    while running.value:
        time.sleep(0.1) # Do nothing during loop

    # When not running anymore, stop the audio playback and free the coefficients
    stop_miso()
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


def steer_cartesian_degree(azimuth: float, elevation: float):
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

    steer_offset = int(elevation * MAX_RES_X * n_active_mics + azimuth * n_active_mics)
    
    steer(steer_offset)

def steer4(azimuth: float, elevation: float):
    """Steer a MISO into a specific direction"""
    # print("Lol got angles from python")
    
    azimuth = int(azimuth * MAX_RES_X)
    elevation = int(elevation * MAX_RES_Y)

    _, n_active_mics = active_microphones()

    steer_offset = int(elevation * MAX_RES_X * n_active_mics + azimuth * n_active_mics)
    
    print(steer_offset)
    steer(steer_offset)


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



# Web interface
def uti_api(q: JoinableQueue, running: Value):
    api(q, running)

def uti_api_with_miso(q: JoinableQueue, running: Value):
    api_with_miso(q, running)

def conv_api(q: JoinableQueue, running: Value):
    api_convolve(q, running)

def miso_api(q: JoinableQueue, running: Value):
    api_miso(q, running)

def just_miso_api(q: JoinableQueue, running: Value):
    just_miso(q, running)

def b(q: JoinableQueue, running: Value):
    _loop_mimo_pad(q, running)

def just_miso_loop(q: JoinableQueue, running: Value):
    """Dummy loop for testing miso"""
    import time
    while running.value:
        try:
            time.sleep(0.1)
        except KeyboardInterrupt:
            running.value = 0


# Testing
def _main(consumer, producer):
    jobs = 1
    q = JoinableQueue(maxsize=2)

    v = Value('i', 1)

    connect()

    try:

        producers = [
            Process(target=producer, args=(q, v))
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

    finally: # Stop the program
        v.value = 0
        disconnect()

def mimo():
    from lib.visual import Viewer
    consumer = Viewer().loop
    # producer = uti_api
    producer = b
    _main(consumer, producer)

# def __miso():
#     producer = uti_api_with_miso #just_miso_api
#     from lib.visual import Viewer
#     consumer = Viewer(cb=steer2).loop
#     # consumer = just_miso_loop
#     _main(consumer, producer)


def lop(q: JoinableQueue, running: Value):
    _loop_miso_pad(q, running)

def lop2(q: JoinableQueue, running: Value):
    _loop_miso_lerp(q, running)

def multi(q_steer: JoinableQueue, q_out: JoinableQueue, running: Value):
    _loop_mimo_and_miso_pad(q_steer, q_out, running)

def _miso():
    producer = lop2
    from lib.visual import Front
    

    q_rec = JoinableQueue(maxsize=2)
    q_out = JoinableQueue(maxsize=2)

    v = Value('i', 1)
    f = Front(q_rec, q_out, v)
    consumer = f.loop

    print("Cython: Connecting to FPGA")
    connect()

    try:

        producers = [
            Process(target=producer, args=(q_out, v))
        ]

        # daemon=True is important here
        consumers = [
            Process(target=consumer, daemon=True)
        ]

        # + order here doesn't matter
        for p in consumers + producers:
            p.start()

        for p in producers:
            p.join()


    finally:

        # Stop the program
        v.value = 0
        disconnect()



def miso():
    producer = multi
    from lib.visual import Front
    

    q_rec = JoinableQueue(maxsize=2)
    q_out = JoinableQueue(maxsize=2)

    v = Value('i', 1)
    f = Front(q_rec, q_out, v)
    consumer = f.multi_loop

    print("Cython: Connecting to FPGA")
    connect()

    try:

        producers = [
            Process(target=producer, args=(q_out, q_rec, v))
        ]

        # daemon=True is important here
        consumers = [
            Process(target=consumer, daemon=True)
        ]

        # + order here doesn't matter
        for p in consumers + producers:
            p.start()

        for p in producers:
            p.join()


    finally:

        # Stop the program
        v.value = 0
        disconnect()