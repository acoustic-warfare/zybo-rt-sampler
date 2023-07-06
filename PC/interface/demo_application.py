from threading import Thread
import cv2, time
import numpy as np

import ctypes

import pyaudio
import sys

# Local
import config
import os
rogue_child_stopper = False

# Load shared library, make sure to compile using make command
def get_antenna_data(replay_mode=False):
    
    lib = ctypes.cdll.LoadLibrary("../lib/beamformer.so")

    init = lib.load
    init.restype = int
    init.argtypes = [
        ctypes.c_bool
    ]
    # MISO
    get_data = lib.miso
    get_data.restype = None
    get_data.argtypes = [
        ctypes.POINTER(ctypes.c_float)
    ]

    # MIMO
    get_image = lib.mimo
    get_image.restype = None
    get_image.argtypes = [
        np.ctypeslib.ndpointer(dtype=config.NP_DTYPE, # Coefficients
        ndim=2,
        flags='C_CONTIGUOUS'
    )
    ]

    steer = lib.steer
    steer.restype = None
    steer.argtypes = [
        ctypes.c_float,
        ctypes.c_float
    ]
    # Initiate antenna from the C side
    
    init(replay_mode)

    return get_data, steer, get_image



class RealtimeSoundplayer(object):
    def __init__(self, mic_index = 4):
        self.f = get_data
        self.gtf = get_image
        self.out = np.empty(config.N_SAMPLES, dtype=config.NP_DTYPE)
        self.out_pointer = self.out.ctypes.data_as(
            ctypes.POINTER(ctypes.c_float))
        self.mic_index = mic_index
    
    def this_callback(self, in_data, frame_count, time_info, status):
        """This is a pyaudio callback when an output is finished and new data should be gathered"""
        self.f(self.out_pointer)

        return self.out, pyaudio.paContinue

    def play_sound(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=config.fs,
                        input=False,
                        output=True,
                        stream_callback=self.this_callback,
                        frames_per_buffer=config.N_SAMPLES)

        stream.start_stream()

        while stream.is_active():
            # Do nothing for some time
            time.sleep(0.1)

        stream.stop_stream()
        stream.close()

        p.terminate()


class ThreadedCamera(object):
    def __init__(self, src=0, replay_mode=False, sound_command=""):
        self.X, self.Y = config.WINDOW_SIZE

        self.light_blue = [27,  170, 222]
        self.blue       = [66,  106, 253]
        self.dark_blue  = [60,   40, 170]
        self.yellow     = [250, 250,  20]
        self.orange     = [244, 185,  60]
        self.green      = [100, 200, 100]
        self.colors = [self.light_blue, self.blue, self.dark_blue, self.yellow, self.orange, self.green]
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
        # FPS = 1/self.eeeeeellllll......ffffffffffff......X
        # X = desired FPS
        self.FPS = 1/256
        self.FPS_MS = int(self.FPS * 1000)
        self.sound_command = sound_command

       # if(not replay_mode):
            #print("HEHHEHHE")
            ## Start frame retrieval thread
            #self.thread = Thread(target=self.update, args=())
            #self.thread.daemon = True
            #self.thread.start()

        # Start playing sound on another thread
        self.thread2 = Thread(target=self.play_sound, args=())
        self.thread2.daemon = True
        self.thread2.start()

        

    def replay_sound(self):
        time.sleep(6)
        os.system(sound_command)
        
    def update(self):
        """Retrieve camera image"""
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            
    def show_frame(self):
        frame = cv2.resize(self.frame, config.WINDOW_SIZE)
        dst = cv2.addWeighted(frame, 0.6, self.calculate_heatmap(), 0.8, 0)

        cv2.imshow(config.APPLICATION_NAME, dst)
        cv2.setMouseCallback(config.APPLICATION_NAME, self.mouse_click_handler)
        cv2.waitKey(1)

    def replay(self):
        while True:
            self.status, self.frame = self.capture.read()
            frame = cv2.resize(self.frame, config.WINDOW_SIZE)
            dst = cv2.addWeighted(frame, 0.6, self.calculate_heatmap(), 0.8, 0)

            cv2.imshow(config.APPLICATION_NAME, dst)
            cv2.setMouseCallback(config.APPLICATION_NAME, self.mouse_click_handler)
            if cv2.waitKey(self.FPS_MS) & 0xFF == ord('q'):
                break

        # After the loop release the cap object
        self.capture.release()
        # Destroy all the windows
        cv2.destroyAllWindows()

    # Very unoptimized. Use numpy functionalities
    def calculate_heatmap(self):
        """"""
        get_image(image)
        lmax = np.max(image)
        if lmax != 0.0:
            for i in range(config.MAX_RES):
                for j in range(config.MAX_RES):
                    val = int(255 * image[i][j]/lmax)

                    if (lmax < 0.0000001):
                        val = 0

                    else:
                        if val < 220:
                            val = 0

                    self.small_heatmap[i][j] = [val, 0, 0]

        heatmap = cv2.resize(self.small_heatmap, config.WINDOW_SIZE, interpolation=cv2.INTER_LINEAR)

        # Apply the old heatmap to make it smooth
        heatmap = cv2.addWeighted(self.previous, 0.5, heatmap, 0.5, 0)
        self.previous = heatmap

        return heatmap
    
    def mouse_click_handler(self, event, x, y, flags, params):
        """Steers the antenna to listen in a specific direction"""
        if event == cv2.EVENT_LBUTTONDOWN:
            horizontal = (x / self.X) * config.MAX_ANGLE * 2 - config.MAX_ANGLE
            vertical = (y / self.Y) * config.MAX_ANGLE * 2 - config.MAX_ANGLE
            steer(-horizontal, vertical)
            print(f"{horizontal}, {vertical}")

    def play_sound(self):
        sound_player = RealtimeSoundplayer()
        sound_player.play_sound()

def test_camera(src, replay_mode, sound_command = ""):
    threaded_camera = ThreadedCamera(src, replay_mode, sound_command)
    
    if replay_mode:
        thread3 = Thread(target=threaded_camera.replay_sound, args=())
        thread3.daemon = True
        thread3.start()

    threaded_camera.replay()
        

def test_sound():
    """Play sound in the current direction"""
    out = np.empty(config.N_SAMPLES, dtype=config.NP_DTYPE)
    while True:
        get_data(out.ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)))
        
        get_image(image)

if __name__ == '__main__':
    image = np.empty((config.MAX_RES, config.MAX_RES), dtype=config.NP_DTYPE)
    src = 0
    replay_mode = False
    sound_command = ""
    if len(sys.argv) > 1 and sys.argv[1] == "replay":
        replay_mode = True
        if len(sys.argv) == 3:
            src = "../replays/" + str(sys.argv[2]) + "/replay.avi"
            sound_command = "udpreplay -i lo ../replays/"+ str(sys.argv[2]) + "/replay.pcap"

    get_data, steer, get_image = get_antenna_data(replay_mode)
    test_camera(src, replay_mode, sound_command)
    
    
