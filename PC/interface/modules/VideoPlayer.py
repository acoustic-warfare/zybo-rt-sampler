import cv2, time
import config
import numpy as np
import os
import signal

#TODO: Signal handler

class VideoPlayer(object):
    def __init__(self, beamformer, src=2, replay_mode=False):
        self.X, self.Y = config.WINDOW_SIZE

        if replay_mode:
            self.capture = cv2.VideoCapture(src)
        else:
            self.capture = cv2.VideoCapture(src, 200)

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.shape = (config.MAX_RES, config.MAX_RES, 3)
        self.small_heatmap = np.zeros(self.shape, dtype=np.uint8)
        self.previous = np.zeros((self.Y, self.X, 3), dtype=np.uint8)

        self.FPS = 1/256
        self.FPS_MS = int(self.FPS * 1000)
        self.beamformer = beamformer
        self.steer = beamformer.get_antenna_data()[1]
       # if(not replay_mode):
            #print("HEHHEHHE")
            ## Start frame retrieval thread
            #self.thread = Thread(target=self.update, args=())
            #self.thread.daemon = True
            #self.thread.start()

        ## Start playing sound on another thread
        #self.thread2 = Thread(target=self.play_sound, args=())
        #self.thread2.daemon = True
        #self.thread2.start()

    def display(self):
        while True:
            self.status, self.frame = self.capture.read()
            try:
                self.frame = cv2.resize(self.frame, config.WINDOW_SIZE)
            except cv2.error as e:
                print("An error ocurred with image processing! Check if camera and antenna connected properly")
                os.system("killall python3")
            dst = self.add_heatmap_to_frame()
            if config.FLIP_IMAGE:
                dst = cv2.flip(dst, 1)
            #dst = cv2.resize(dst, (1920, 1080))
            cv2.imshow(config.APPLICATION_NAME, dst)
            cv2.setMouseCallback(config.APPLICATION_NAME, self.mouse_click_handler)
            cv2.waitKey(self.FPS_MS)
    
    def add_heatmap_to_frame(self):
        return cv2.addWeighted(self.frame, 0.6, self.beamformer.calculate_heatmap(), 0.8, 0)

    
    def mouse_click_handler(self, event, x, y, flags, params):
        """Steers the antenna to listen in a specific direction"""
        if event == cv2.EVENT_LBUTTONDOWN:
            horizontal = (x / self.X) * config.MAX_ANGLE * 2 - config.MAX_ANGLE
            vertical = (y / self.Y) * config.MAX_ANGLE * 2 - config.MAX_ANGLE
            self.steer(-horizontal, vertical)
            print(f"{horizontal}, {vertical}")
