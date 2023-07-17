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

from config cimport *

# C defined functions
cdef extern from "beamformer.h":
    void load_coefficients(int *whole_sample_delay)
    # void work_test(float *image)
    # int load(bint)
    # void myread(float *signal)
    # void signal_handler()
    # void kill_child()
    void mimo_truncated(float *image, int *adaptive_array, int n)


# from lib.Beamformer import Beamformer
# from lib.VideoPlayer import VideoPlayer


cdef tuple setup():
    whole_samples, fractional_samples = calculate_coefficients()
    active_mics, n_active_mics = active_microphones()

    #cdef np.ndarray[int, ndim=1, mode="c"] active_micro 
    active_micro = np.ascontiguousarray(active_mics.astype(np.int32))

    cdef np.ndarray[int, ndim=3, mode="c"] i32_whole_samples 
    i32_whole_samples = np.ascontiguousarray(whole_samples.astype(np.int32))

    # Pass int pointer to C function
    load_coefficients(&i32_whole_samples[0, 0, 0])

    mimo_output = np.zeros((MAX_RES_X, MAX_RES_Y), dtype=DTYPE_arr)

    cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] mimo_arr
    mimo_arr = np.ascontiguousarray(mimo_output)

    return mimo_arr, active_micro, n_active_mics

# cdef class TruncAndSum(Beamformer):
#     cdef public object image, active_micro
#     cdef public int n_active_mics
#     def __init__(self):
#         super().__init__()

#     def update_setup(self):
#         cdef int i = 0
#         self.image, self.active_micro, self.n_active_mics = setup()

#     def calculate_heatmap(self) -> np.ndarray:

#         mimo_truncated(&self.image[0, 0], &self.active_micro[0], int(self.n_active_mics))
        
#         return super().calculate_heatmap()

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
WINDOW_DIMENSIONS = (APPLICATION_WINDOW_WIDTH, APPLICATION_WINDOW_HEIGHT)
APPLICATION_NAME = "Demo App"
cdef class VideoPlayer:
    cdef public object capture, previous, frame, status
    cdef public int X, Y

    def __init__(self, src: int=CAMERA_SOURCE, replay_mode=False, steer_callback=None):
        if replay_mode:
            self.capture = cv2.VideoCapture(src)
        else:
            self.capture = cv2.VideoCapture(src, 200)

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, APPLICATION_WINDOW_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, APPLICATION_WINDOW_HEIGHT)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        #cv2.setMouseCallback(APPLICATION_NAME, self.mouse_click_handler)

    def display(self, image):
        self.status, self.frame = self.capture.read()
        try:
            self.frame = cv2.resize(self.frame, WINDOW_DIMENSIONS)
        except cv2.error as e:
            print("An error ocurred with image processing! Check if camera and antenna connected properly")
            # os.system("killall python3")
        dst = cv2.addWeighted(self.frame, 0.6, image, 0.8, 0)
        if False:
            dst = cv2.flip(dst, 1)

        cv2.imshow(APPLICATION_NAME, dst)
        cv2.waitKey(1)

    def mouse_click_handler(self, event, x, y, flags, params):
        """Steers the antenna to listen in a specific direction"""
        if event == cv2.EVENT_LBUTTONDOWN:
            horizontal = (x / self.X) * MAX_ANGLE * 2 - MAX_ANGLE
            vertical = (y / self.Y) * MAX_ANGLE * 2 - MAX_ANGLE
            self.steer(-horizontal, vertical)
            print(f"{horizontal}, {vertical}")

    def calculate_heatmap(image):
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

from lib.microphone_array import connect, disconnect


# cdef class test:
#     cdef object mimo_output
#     cdef public VideoPlayer vp
#     def __init__(self):
#         self.vp = VideoPlayer()

#     def do(self):
#         whole_samples, fractional_samples = calculate_coefficients()
#         active_mics, n_active_mics = active_microphones()

#         cdef np.ndarray[int, ndim=1, mode="c"] active_micro 
#         active_micro = np.ascontiguousarray(active_mics.astype(np.int32))

#         cdef np.ndarray[int, ndim=3, mode="c"] i32_whole_samples 
#         i32_whole_samples = np.ascontiguousarray(whole_samples.astype(np.int32))

#         # Pass int pointer to C function
#         load_coefficients(&i32_whole_samples[0, 0, 0])

#         connect()

#         self.mimo_output = np.zeros((MAX_RES_X, MAX_RES_Y), dtype=DTYPE_arr)

#         cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] mimo_arr
#         mimo_arr = np.ascontiguousarray(self.mimo_output)
        
#         while True:
#             mimo_truncated(&mimo_arr[0, 0], &active_micro[0], int(n_active_mics))
#             image = self.vp.calculate_heatmap(mimo_arr)
#             self.vp.display(image)

cdef void bro():
    whole_samples, fractional_samples = calculate_coefficients()
    active_mics, n_active_mics = active_microphones()

    cdef np.ndarray[int, ndim=1, mode="c"] active_micro 
    active_micro = np.ascontiguousarray(active_mics.astype(np.int32))

    cdef np.ndarray[int, ndim=3, mode="c"] i32_whole_samples 
    i32_whole_samples = np.ascontiguousarray(whole_samples.astype(np.int32))

    # Pass int pointer to C function
    load_coefficients(&i32_whole_samples[0, 0, 0])

    connect()

    mimo_output = np.zeros((MAX_RES_X, MAX_RES_Y), dtype=DTYPE_arr)

    cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] mimo_arr
    mimo_arr = np.ascontiguousarray(mimo_output)

    vp = VideoPlayer()
    
    while True:
        mimo_truncated(&mimo_arr[0, 0], &active_micro[0], int(n_active_mics))
        image = vp.calculate_heatmap(mimo_arr)
        vp.display(image)
    


def run():
    bro()
    #from lib.microphone_array import connect, disconnect
    #connect()
    # tsbf = TruncAndSum()

    # v = VideoPlayer(tsbf)

    # v.display()

    #disconnect()
    exit()


