import cv2
import matplotlib.pyplot as plt

import time

import sys
sys.path.insert(0, "") # Access local modules located in . Enables 'from . import MODULE'


from config cimport *

import numpy as np
cimport numpy as np

# It's necessary to call "import_array" if you use any part of the numpy PyArray_* API.
np.import_array()

import os
import signal

#TODO: Signal handler

WINDOW_DIMENSIONS = (APPLICATION_WINDOW_WIDTH, APPLICATION_WINDOW_HEIGHT)
APPLICATION_NAME = "Demo App"

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

cdef class VideoPlayer(object):
    cdef public object capture, small_heatmap, previous, steer, frame, status
    cdef public float FPS
    cdef public tuple shape
    cdef public int FPS_MS, X, Y
    cdef public object beamformer
    def __init__(self, beamformer, src=CAMERA_SOURCE, replay_mode=False):
        self.X, self.Y = WINDOW_DIMENSIONS

        if replay_mode:
            self.capture = cv2.VideoCapture(src)
        else:
            self.capture = cv2.VideoCapture(src, 200)

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, APPLICATION_WINDOW_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, APPLICATION_WINDOW_HEIGHT)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.shape = (MAX_RES_Y, MAX_RES_X, 3)
        self.small_heatmap = np.zeros(self.shape, dtype=np.uint8)
        self.previous = np.zeros((self.Y, self.X, 3), dtype=np.uint8)

        self.FPS = 1/256
        self.FPS_MS = int(self.FPS * 1000)
        self.beamformer = beamformer
        self.steer = beamformer.get_antenna_data()[1]

    def display(self):
        while True:
            self.status, self.frame = self.capture.read()
            try:
                self.frame = cv2.resize(self.frame, WINDOW_DIMENSIONS)
            except cv2.error as e:
                print("An error ocurred with image processing! Check if camera and antenna connected properly")
                os.system("killall python3")
            dst = self.add_heatmap_to_frame()
            if True:
                dst = cv2.flip(dst, 1)

            cv2.imshow(APPLICATION_NAME, dst)
            cv2.setMouseCallback(APPLICATION_NAME, self.mouse_click_handler)
            cv2.waitKey(1)
    
    def add_heatmap_to_frame(self):
        return cv2.addWeighted(self.frame, 0.6, self.beamformer.calculate_heatmap(), 0.8, 0)

    
    def mouse_click_handler(self, event, x, y, flags, params):
        """Steers the antenna to listen in a specific direction"""
        if event == cv2.EVENT_LBUTTONDOWN:
            horizontal = (x / self.X) * MAX_ANGLE * 2 - MAX_ANGLE
            vertical = (y / self.Y) * MAX_ANGLE * 2 - MAX_ANGLE
            self.steer(-horizontal, vertical)
            print(f"{horizontal}, {vertical}")

class Viewer:
    def __init__(self, src, convolveBackend=False, replayMode=False):
        if replayMode:
            if not convolveBackend:
                self.wait = 70
            else:
                self.wait = 1
        else:
            self.wait = 1

        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, APPLICATION_WINDOW_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, APPLICATION_WINDOW_HEIGHT)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    def show(self, small_heatmap):
        status, frame = self.capture.read()
        frame = cv2.flip(frame, 1) # Nobody likes looking out of the array :(
        try:
            frame = cv2.resize(frame, WINDOW_DIMENSIONS)
        except cv2.error as e:
            print("An error ocurred with image processing! Check if camera and antenna connected properly")
            exit()

        image = cv2.addWeighted(frame, 0.6, calculate_heatmap(small_heatmap), 0.8, 0)
        cv2.imshow("Demo", image)
        cv2.waitKey(self.wait)
