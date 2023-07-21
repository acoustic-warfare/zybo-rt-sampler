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


from config cimport *

cdef extern from "api.h":
    int load(bint)
    void get_data(float *signals)
    void stop_receiving()
    void pad_mimo(float *image, int *adaptive_array, int n)
    void convolve_mimo_vectorized(float *image, int *adaptive_array, int n)
    void convolve_mimo_naive(float *image, int *adaptive_array, int n)
    # void load_coefficients_pad(int *whole_samples, int n)
    # void unload_coefficients_pad()

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


available_backends = {

}

cdef int pad_tests():
    whole_samples, fractional_samples = calculate_coefficients()
    active_mics, n_active_mics = active_microphones()

    cdef np.ndarray[int, ndim=1, mode="c"] active_micro = np.ascontiguousarray(active_mics.astype(np.int32))

    cdef np.ndarray[int, ndim=3, mode="c"] i32_whole_samples

    assert whole_samples.shape == (MAX_RES_X, MAX_RES_Y, n_active_mics), f"whole samples do not match: {whole_samples.shape} != {(MAX_RES_X, MAX_RES_Y, n_active_mics)}"

    i32_whole_samples = np.ascontiguousarray(whole_samples.astype(np.int32))

    # Pass int pointer to C function
    load_coefficients_pad(&i32_whole_samples[0, 0, 0], whole_samples.size)

    x = np.zeros((MAX_RES_X, MAX_RES_Y), dtype=DTYPE_arr)

    cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] mimo_arr
    mimo_arr = np.ascontiguousarray(x)

    cdef np.ndarray[np.float32_t, ndim=1, mode = 'c'] sig_arr
    signal = np.ones(N_SAMPLES, dtype=DTYPE_arr)
    sig_arr = np.ascontiguousarray(signal)

    cdef np.ndarray[np.float32_t, ndim=1, mode = 'c'] out_arr
    out = np.zeros(N_SAMPLES, dtype=DTYPE_arr)
    out_arr = np.ascontiguousarray(out)

    pad_delay(&sig_arr[0], &out_arr[0], 2)

    print(sig_arr, out_arr)

    unload_coefficients_pad()


cdef void pad_mimo_api(bf: object):
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

    while bf.running:
        pad_mimo(&mimo_arr[0, 0], &active_micro[0], int(n_active_mics))
        bf.show(mimo_arr)

    unload_coefficients_pad()


cdef _convolve_coefficients_load(h):
    cdef np.ndarray[float, ndim=4, mode="c"] f32_h = np.ascontiguousarray(h)
    load_coefficients_convolve(&f32_h[0, 0, 0, 0], int(h.size))
    
cdef void convolve_mimo_api(bf: object):
    cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] mimo_arr
    
    image = np.zeros((MAX_RES_X, MAX_RES_Y), dtype=DTYPE_arr)
    mimo_arr = np.ascontiguousarray(image)

    h = compute_convolve_h()
    active_mics, n_active_mics = active_microphones()

    cdef np.ndarray[int, ndim=1, mode="c"] active_micro = np.ascontiguousarray(active_mics.astype(np.int32))

    _convolve_coefficients_load(h)

    while bf.running:
        convolve_mimo_vectorized(&mimo_arr[0, 0], &active_micro[0], int(n_active_mics))
        bf.show(mimo_arr)

    unload_coefficients_convolve()


WINDOW_DIMENSIONS = (APPLICATION_WINDOW_WIDTH, APPLICATION_WINDOW_HEIGHT)
APPLICATION_NAME = "Demo App"

import cv2
import matplotlib.pyplot as plt

def generate_color_map(name="jet"):
    
    cmap = plt.cm.get_cmap(name)

    cdef np.ndarray[np.uint8_t, ndim=2, mode="c"] colors 

    # Generate color lookup table
    colors = np.empty((256, 3), dtype=np.uint8)

    for i in range(256):
        colors[i] = (np.array(cmap(255 - i)[:3]) * 255).astype(np.uint8)

    return colors


colors = generate_color_map()

def calculate_heatmap(image):
    """"""
    lmax = np.max(image)

    image /= lmax

    small_heatmap = np.zeros((MAX_RES_Y, MAX_RES_X, 3), dtype=np.uint8)

    if lmax>1e-8:
        for x in range(MAX_RES_X):
            for y in range(MAX_RES_Y):
                d = image[x, y]

                if d > 0.9:
                    val = int(255 * d ** MISO_POWER)

                    small_heatmap[MAX_RES_Y - 1 - y, x] = colors[val]


    heatmap = cv2.resize(small_heatmap, WINDOW_DIMENSIONS, interpolation=cv2.INTER_LINEAR)

    return heatmap


import cv2
# import multiprocessing

# from multiprocessing import shared_memory
cdef class Beamformer:
    cdef public bint connected, replay_mode, verbose, running, can_read
    cdef public object output, capture
    def __init__(self, replay_mode = False, verbose = True):
        self.connected = False
        self.replay_mode = replay_mode
        self.verbose = verbose
        self.running = True
        self.can_read = False
        self.output = np.zeros((MAX_RES_X, MAX_RES_Y), dtype=np.float32)

        self.capture = cv2.VideoCapture("/dev/video0")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, APPLICATION_WINDOW_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, APPLICATION_WINDOW_HEIGHT)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        thread = threading.Thread(target=self.viewer)
        thread.start()

    def connect(self):
        if not self.connected:
            connect(self.replay_mode)
            self.running = True
    
    def disconnect(self):
        if self.connected:
            disconnect()
            self.running = False

    def loop(self):
        while self.running:
            if self.can_read:
                self.can_read = False
                yield self.output

    def show(self, arr):
        self.output = arr.copy()
        self.can_read = True

    def viewer(self):

        while self.running:

            status, frame = self.capture.read()
            frame = cv2.flip(frame, 1) # Nobody likes looking out of the array :(
            try:
                frame = cv2.resize(frame, WINDOW_DIMENSIONS)
            except cv2.error as e:
                print("An error ocurred with image processing! Check if camera and antenna connected properly")
                self.running = False
                break

            res = calculate_heatmap(self.output)

            image = cv2.addWeighted(frame, 0.6, res, 0.8, 0)

            cv2.imshow(APPLICATION_NAME, image)
            # cv2.setMouseCallback(APPLICATION_NAME, self.mouse_click_handler)
            cv2.waitKey(1)


import threading

def test(bf: Beamformer):
    connect()
    try:
        pad_mimo_api(bf)
    finally:
        disconnect()


cdef void speed(bf: object):
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

    capture = cv2.VideoCapture(2)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, APPLICATION_WINDOW_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, APPLICATION_WINDOW_HEIGHT)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    while bf.running:
        pad_mimo(&mimo_arr[0, 0], &active_micro[0], int(n_active_mics))
        # print(mimo_arr)
        # bf.show(mimo_arr)

        status, frame = capture.read()
        frame = cv2.flip(frame, 1) # Nobody likes looking out of the array :(
        try:
            frame = cv2.resize(frame, WINDOW_DIMENSIONS)
        except cv2.error as e:
            print("An error ocurred with image processing! Check if camera and antenna connected properly")
            bf.running = False
            break

        res = calculate_heatmap(mimo_arr)

        image = cv2.addWeighted(frame, 0.6, res, 0.8, 0)

        cv2.imshow(APPLICATION_NAME, image)
        # cv2.setMouseCallback(APPLICATION_NAME, self.mouse_click_handler)
        cv2.waitKey(1)

    unload_coefficients_pad()

def gen(bf: Beamformer):
    while bf.running:
        if bf.can_read:
            bf.can_read = False
            yield bf.output


def runner(bf):
    capture = cv2.VideoCapture("/dev/video2")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, APPLICATION_WINDOW_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, APPLICATION_WINDOW_HEIGHT)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    try:
        for output in gen(bf):
            status, frame = capture.read()
            frame = cv2.flip(frame, 1) # Nobody likes looking out of the array :(
            try:
                frame = cv2.resize(frame, WINDOW_DIMENSIONS)
            except cv2.error as e:
                print("An error ocurred with image processing! Check if camera and antenna connected properly")
                bf.running = False
                break

            res = calculate_heatmap(output)

            image = cv2.addWeighted(frame, 0.6, res, 0.8, 0)

            cv2.imshow(APPLICATION_NAME, image)
            # cv2.setMouseCallback(APPLICATION_NAME, self.mouse_click_handler)
            cv2.waitKey(1)
    finally:
        bf.running = False



import queue
import random
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
    # whole_samples, fractional_samples = calculate_coefficients()
    # active_mics, n_active_mics = active_microphones()

    # cdef np.ndarray[int, ndim=1, mode="c"] active_micro = np.ascontiguousarray(active_mics.astype(np.int32))

    # cdef np.ndarray[int, ndim=3, mode="c"] i32_whole_samples

    # i32_whole_samples = np.ascontiguousarray(whole_samples.astype(np.int32))

    # # Pass int pointer to C function
    # load_coefficients_pad(&i32_whole_samples[0, 0, 0], whole_samples.size)

    # x = np.zeros((MAX_RES_X, MAX_RES_Y), dtype=DTYPE_arr)

    # cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] mimo_arr
    # mimo_arr = np.ascontiguousarray(x)

    # while running.value:
    #     pad_mimo(&mimo_arr[0, 0], &active_micro[0], int(n_active_mics))
    #     q.put(mimo_arr)

    # # q.join()

    # unload_coefficients_pad()

    cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] mimo_arr
    
    image = np.zeros((MAX_RES_X, MAX_RES_Y), dtype=DTYPE_arr)
    mimo_arr = np.ascontiguousarray(image)

    h = compute_convolve_h()
    active_mics, n_active_mics = active_microphones()

    cdef np.ndarray[int, ndim=1, mode="c"] active_micro = np.ascontiguousarray(active_mics.astype(np.int32))

    _convolve_coefficients_load(h)

    while running.value:
        convolve_mimo_naive(&mimo_arr[0, 0], &active_micro[0], int(n_active_mics))
        q.put(mimo_arr)
        # print(mimo_arr)

    unload_coefficients_convolve()

def uti_api(q: JoinableQueue, running: Value):
    api(q, running)

def consumer(q: JoinableQueue, v: Value):
    capture = cv2.VideoCapture("/dev/video0")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, APPLICATION_WINDOW_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, APPLICATION_WINDOW_HEIGHT)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    while True:
        try:
            output = q.get(block=False)
            q.task_done()
            status, frame = capture.read()
            frame = cv2.flip(frame, 1) # Nobody likes looking out of the array :(
            try:
                frame = cv2.resize(frame, WINDOW_DIMENSIONS)
            except cv2.error as e:
                print("An error ocurred with image processing! Check if camera and antenna connected properly")
                v.value = 0
                break

            res = calculate_heatmap(output)

            image = cv2.addWeighted(frame, 0.6, res, 0.8, 0)

            cv2.imshow(APPLICATION_NAME, image)
            cv2.waitKey(1)
        except queue.Empty:
            pass
        except KeyboardInterrupt:
            v.value = 0
            break


def main():
    jobs = 1
    q = JoinableQueue(maxsize=2)

    v = Value('i', 1)

    connect()

    try:

        producers = [
            Process(target=api_convolve, args=(q, v))
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


