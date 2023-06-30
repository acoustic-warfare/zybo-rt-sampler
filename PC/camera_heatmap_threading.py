from threading import Thread
import cv2, time
import numpy as np

class ThreadedCamera(object):
    def __init__(self, src=0):
        self.light_blue = [27,  170, 222]
        self.blue       = [66,  106, 253]
        self.dark_blue  = [60,   40, 170]
        self.yellow     = [250, 250,  20]
        self.orange     = [244, 185,  60]
        self.green      = [100, 200, 100]
        self.colors = [self.light_blue, self.blue, self.dark_blue, self.yellow, self.orange, self.green]

        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.shape = (30, 32, 3)
        self.small_heatmap = np.zeros(self.shape, dtype=np.uint8)
        # FPS = 1/self.eeeeeellllll......ffffffffffff......X
        # X = desired FPS
        self.FPS = 1/30
        self.FPS_MS = int(self.FPS * 1000)
        
        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        
    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(self.FPS)
            
    def show_frame(self):
        dst = cv2.addWeighted(self.frame, 0.6, self.simulate_heatmap(), 0.4, 0)
        cv2.imshow('frame', dst)
        cv2.waitKey(self.FPS_MS)

    # Very unoptimized. Use numpy functionalities
    def simulate_heatmap(self):
        random_heatmap = np.random.randint(0, 6, (30, 32), dtype=np.uint8)
        for i in range(30):
            for j in range(32):
                self.small_heatmap[i][j] = self.colors[random_heatmap[i][j]]
        
        heatmap = cv2.resize(self.small_heatmap, (1280, 720), interpolation=cv2.INTER_LINEAR)
        return heatmap


if __name__ == '__main__':
    threaded_camera = ThreadedCamera()
    while True:
        try:
            threaded_camera.show_frame()
        except AttributeError:
            pass
