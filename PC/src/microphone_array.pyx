# cython: language_level=3
# distutils: language=c


"""
Wrapper for interfacing with the C-backend

This module contains 3 important functions:

- connect: Used for establishing a connection to a microphone data-stream

- receive: Used for retrieving data from the stream

- disconnect: Used for closing the connection and destroy processes

"""

#TODO Implement adaptive beamforming properly

import cv2
import matplotlib.pyplot as plt

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

from lib.directions import calculate_coefficients

WINDOW_DIMENSIONS = (APPLICATION_WINDOW_WIDTH, APPLICATION_WINDOW_HEIGHT)

# C defined functions exposed to python
cdef extern from "beamformer.h":
    void load_coefficients(int *whole_sample_delay)
    void work_test(float *image)
    int load(bint)
    void myread(float *signal)
    void signal_handler()
    void kill_child()
    void miso(float *signal)
    void steer(float theta, float phi)


def generate_color_map(name="jet"):
    """
    Faster lookup for converting a value to a color
    """
    cmap = plt.cm.get_cmap(name)

    cdef np.ndarray[np.uint8_t, ndim=2, mode="c"] colors 

    # Generate color lookup table
    colors = np.empty((256, 3), dtype=np.uint8)

    for i in range(256):
        colors[i] = (np.array(cmap(255 - i)[:3]) * 255).astype(np.uint8)

    return colors


colors = generate_color_map()

def calculate_heatmap(image):
    """
    TODO

    Calculate a heatmap based on the image received from the algorithm
    """
    lmax = np.max(image)

    image /= lmax

    small_heatmap = np.zeros((MAX_RES_Y, MAX_RES_X, 3), dtype=np.uint8)

    if lmax>1e-9:
        for x in range(MAX_RES_X):
            for y in range(MAX_RES_Y):
                d = image[x, y]

                if d > 0.9:
                    val = int(255 * d ** MISO_POWER)

                    small_heatmap[MAX_RES_Y - 1 - y, x] = colors[val]

    heatmap = cv2.resize(small_heatmap, (WINDOW_DIMENSIONS), interpolation=cv2.INTER_LINEAR)

    return heatmap

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

def disconnect() -> None:
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


cdef void loop():

    whole_samples, fractional_samples = calculate_coefficients()

    cdef np.ndarray[int, ndim=3, mode="c"] samples

    samples = np.ascontiguousarray(whole_samples.astype(np.int32))

    load_coefficients(&samples[0, 0, 0])

    connect()

    x = np.zeros((MAX_RES_X, MAX_RES_Y), dtype=DTYPE_arr)

    cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] arr2
    arr2 = np.ascontiguousarray(x)
    
    capture = cv2.VideoCapture(CAMERA_SOURCE)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, APPLICATION_WINDOW_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, APPLICATION_WINDOW_HEIGHT)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    while True:
        work_test(&arr2[0, 0])

        heatmap = calculate_heatmap(arr2)
        #heatmap = cv2.flip(heatmap, 1)

        status, frame = capture.read()
        frame = cv2.flip(frame, 1) # Nobody likes looking out of the array :(
        try:
            frame = cv2.resize(frame, WINDOW_DIMENSIONS)
        except cv2.error as e:
            print("An error ocurred with image processing! Check if camera and antenna connected properly")
            #os.system("killall python3")
            break

        print(frame.shape, heatmap.shape)

        image = cv2.addWeighted(frame, 0.6, heatmap, 0.8, 0)
        cv2.imshow("Demo", image)
        cv2.waitKey(1)

def main():
    loop()

