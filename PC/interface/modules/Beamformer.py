import ctypes
import config
import numpy as np
import cv2
import matplotlib.pyplot as plt

cmap = plt.cm.get_cmap("jet")

#print(cmap.N, cmap(100))
#exit()

class Beamformer(object):
    def __init__(self, replay_mode=False): 
        self.lib = ctypes.cdll.LoadLibrary("../lib/beamformer.so")
        
        init = self.lib.load
        init.restype = int
        init.argtypes = [
            ctypes.c_bool
        ]
        # MISO
        self.get_data = self.lib.miso
        self.get_data.restype = None
        self.get_data.argtypes = [
            ctypes.POINTER(ctypes.c_float)
        ]

        # MIMO
        self.get_image = self.lib.mimo
        self.get_image.restype = None
        self.get_image.argtypes = [
            np.ctypeslib.ndpointer(dtype=config.NP_DTYPE, # Coefficients
            ndim=2,
            flags='C_CONTIGUOUS'
        )
        ]

        self.steer = self.lib.steer
        self.steer.restype = None
        self.steer.argtypes = [
            ctypes.c_float,
            ctypes.c_float
        ]

        self.sound_data = self.lib.myread
        self.sound_data.restype = None
        self.sound_data.argtypes = [
            ctypes.POINTER(ctypes.c_float)
        ]

        self.X, self.Y = config.WINDOW_SIZE
        self.shape = (config.MAX_RES, config.MAX_RES, 3)
        self.image = np.empty((config.MAX_RES, config.MAX_RES), dtype=config.NP_DTYPE)
        self.small_heatmap = np.zeros(self.shape, dtype=np.uint8)
        self.previous = np.zeros((self.Y, self.X, 3), dtype=np.uint8)
        
        # Initiate antenna from the C side
        init(replay_mode)

    def get_antenna_data(self):
        return self.get_data, self.steer, self.get_image 
    
    def get_sound_data(self):
        return self.sound_data
    
    def calculate_heatmap(self):
        """"""
        self.get_image(self.image)
        lmax = np.max(self.image)

        #print(lmax)
        #lmax = 1e-6

        #if 
        #lmax = 1.0
        if 1>lmax>1e-7:
        #if True:
            for i in range(config.MAX_RES):
                for j in range(config.MAX_RES):
                    val = min(int(255 * (self.image[i][j] / lmax) ** 25), 255)


                    if val < 15:
                        color = np.zeros(3)

                    else:

                        color = np.array(cmap(255 - val)[:3]) * 255

                    self.small_heatmap[i][j] = color.astype(np.uint8)

                    #self.small_heatmap[i][j] = [0, val, val]

        else:
            self.small_heatmap = np.zeros(self.shape, dtype=np.uint8)

        heatmap = cv2.resize(self.small_heatmap, config.WINDOW_SIZE, interpolation=cv2.INTER_LINEAR)

        # Apply the old heatmap to make it smooth
        heatmap = cv2.addWeighted(self.previous, 0.5, heatmap, 0.5, 0)
        self.previous = heatmap

        return heatmap