import cv2

from multiprocessing import JoinableQueue, Process, Value

from lib.beamformer import *
from lib.visual import calculate_heatmap

import queue
import config
import numpy as np
#from realtime_scripts.phase_shift_algorithm_peak_detection import main
WINDOW_DIMENSIONS = (1920, 1080)# (720, 480)
APPLICATION_WINDOW_WIDTH, APPLICATION_WINDOW_HEIGHT = WINDOW_DIMENSIONS

te = 0
class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        #self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        #self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        #self.video.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
        self.v = Value('i', 1)

        self.setup()
    
    def __del__(self):
        self.video.release()

    def setup(self):
        jobs = 1
        self.q = JoinableQueue(maxsize=2)

        

        connect()
        self.processStarted = False
        self.p = Process(target=uti_api, args=(self.q, self.v))
        #self.p.start()

    def startBeamforming(self, backend = 0):
        if self.processStarted:
                self.v.value = 0
                self.p.join()
                self.v.value = 1
        if backend == 0:
            self.p = Process(target=uti_api, args=(self.q, self.v))
            self.processStarted = True
        elif backend == 1:
            self.p = Process(target=conv_api, args=(self.q, self.v))
            self.processStarted = True
        elif backend == 2:
            #TODO: Change conv_api to new backend
            self.p = Process(target=self.fftbackend, args=(self.q, self.v))
            self.processStarted = True
        self.p.start()

    def fftbackend(self, q: JoinableQueue, v: Value):
        data = np.empty((config.N_MICROPHONES, config.N_SAMPLES), dtype=np.float32)
        while self.v.value:
            receive(data)
            #heatmap_data = main(data.T)
            #self.q.put(heatmap_data)
        

    def handle_image(self, image):
        image = cv2.flip(image, 1) # Nobody likes looking out of the array :(
        image = cv2.resize(image, WINDOW_DIMENSIONS)
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        try:
            output = self.q.get(block=False)
            self.q.task_done()
            res = calculate_heatmap(output)
            res = cv2.resize(res, WINDOW_DIMENSIONS)
            image = cv2.addWeighted(image, 0.6, res, 0.8, 0)

        except queue.Empty:
            pass

        return image
    
    def get_frame(self):
        success, image = self.video.read()
        image = self.handle_image(image)

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
    
    def disableBeamforming(self):
        if self.processStarted:
            self.v.value = 0
            self.p.join()
            self.v.value = 1
    
    def quit(self):
        self.v.value = 0
        self.p.join()
        disconnect()
        self.video.release()

def gen(camera):
    while camera.v.value:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
