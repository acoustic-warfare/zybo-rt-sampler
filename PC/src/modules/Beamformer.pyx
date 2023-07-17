# cython: language_level=3
# distutils: language=c

#TODO Implement adaptive beamforming properly

import cv2

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

WINDOW_DIMENSIONS = (APPLICATION_WINDOW_WIDTH, APPLICATION_WINDOW_HEIGHT)



# For nice colors
import matplotlib.pyplot as plt

def generate_color_map(name="jet") -> np.ndarray:
    
    cmap = plt.cm.get_cmap(name)

    cdef np.ndarray[np.uint8_t, ndim=2, mode="c"] colors 

    # Generate color lookup table
    colors = np.empty((256, 3), dtype=np.uint8)

    for i in range(256):
        colors[i] = (np.array(cmap(255 - i)[:3]) * 255).astype(np.uint8)

    return colors

cdef class Beamformer:

    cdef public str cmap_name
    cdef public object colors, image
    cdef public bint replay

    def __init__(self, replay: bool=False, cmap_name="jet"):
        self.cmap_name = cmap_name
        self.replay = replay
    
    def calculate_cmap(self):
        self.colors = generate_color_map(name=self.cmap_name)


    def calculate_heatmap(self) -> np.ndarray:
        lmax = np.max(self.image)

        self.image /= lmax

        small_heatmap = np.zeros((MAX_RES_Y, MAX_RES_X, 3), dtype=np.uint8)

        if lmax>1e-8:
            for x in range(MAX_RES_X):
                for y in range(MAX_RES_Y):
                    d = self.image[x, y]

                    if d > 0.9:
                        val = int(255 * d ** MISO_POWER)

                        small_heatmap[MAX_RES_Y - 1 - y, x] = self.colors[val]


        heatmap = cv2.resize(small_heatmap, WINDOW_DIMENSIONS, interpolation=cv2.INTER_LINEAR)

        return heatmap

cdef class ConvolveAndSum(Beamformer):
    def __init__(self):
        super().__init__()



