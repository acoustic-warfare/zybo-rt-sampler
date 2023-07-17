import cv2, time

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

cdef class VideoPlayer(object):
    cdef public object capture, small_heatmap, previous, steer, frame, status
    cdef public float FPS
    cdef public tuple shape
    cdef public int FPS_MS, X, Y
    cdef public object beamformer
    def __init__(self, beamformer, src=2, replay_mode=False):
        self.X, self.Y = WINDOW_DIMENSIONS

        if replay_mode:
            self.capture = cv2.VideoCapture(src)
        else:
            self.capture = cv2.VideoCapture(src, 200)

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.shape = (MAX_RES, MAX_RES, 3)
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
            #dst = cv2.resize(dst, (1920, 1080))
            cv2.imshow(APPLICATION_NAME, dst)
            cv2.setMouseCallback(APPLICATION_NAME, self.mouse_click_handler)
            #cv2.waitKey(self.FPS_MS)
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
