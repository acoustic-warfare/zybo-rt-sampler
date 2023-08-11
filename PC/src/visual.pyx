# cython: language_level=3
# distutils: language=c

import numpy as np
cimport numpy as np

# It's necessary to call "import_array" if you use any part of the numpy PyArray_* API.
np.import_array()

import sys
sys.path.insert(0, "") # Access local modules located in . Enables 'from . import MODULE'


from config cimport *

import cv2

WINDOW_DIMENSIONS = (APPLICATION_WINDOW_WIDTH, APPLICATION_WINDOW_HEIGHT)
APPLICATION_NAME = "Demo App"

import matplotlib.pyplot as plt

def generate_color_map(name="jet"):
    
    cmap = plt.cm.get_cmap(name)

    # cdef np.ndarray[np.uint8_t, ndim=2, mode="c"] colors 

    # Generate color lookup table
    colors = np.empty((256, 3), dtype=np.uint8)

    for i in range(256):
        colors[i] = (np.array(cmap(255 - i)[:3]) * 255).astype(np.uint8)

    return colors


colors = generate_color_map()


def calculate_heatmap_old(image):
    """"""
    lmax = np.max(image)

    image /= lmax

    # image = image.T

    small_heatmap = np.zeros((MAX_RES_Y, MAX_RES_X, 3), dtype=np.uint8)
    # small_heatmap = np.zeros((MAX_RES_X, MAX_RES_Y, 3), dtype=np.uint8)

    if lmax>1e-8:
        for x in range(MAX_RES_X):
            for y in range(MAX_RES_Y):
                d = image[x, y]

                if d > 0.9:
                    val = int(255 * d ** MISO_POWER)

                    small_heatmap[MAX_RES_Y - 1 - y, x] = colors[val]
                    # small_heatmap[x, y] = colors[val]

    # cv2.imshow()

    # small_heatmap = np.reshape(small_heatmap, (MAX_RES_Y, MAX_RES_X, 3))


    heatmap = cv2.resize(small_heatmap, WINDOW_DIMENSIONS, interpolation=cv2.INTER_LINEAR)
    # heatmap = cv2.resize(small_heatmap, (1000, 1000), interpolation=cv2.INTER_NEAREST)
    return heatmap

def calculate_heatmap(image):
    """"""
    lmax = np.max(image)

    image /= lmax

    # image = image.T

    small_heatmap = np.zeros((MAX_RES_Y, MAX_RES_X, 3), dtype=np.uint8)
    # small_heatmap = np.zeros((MAX_RES_X, MAX_RES_Y, 3), dtype=np.uint8)

    if lmax>1e-8:
        for x in range(MAX_RES_X):
            for y in range(MAX_RES_Y):
                d = image[x, y]

                if d > 0.9:
                    val = int(255 * d ** MISO_POWER)

                    # small_heatmap[y, MAX_RES_X - 1 - x] = colors[val]
                    small_heatmap[MAX_RES_Y - 1 - y, x] = colors[val]
                    # small_heatmap[x, y] = colors[val]

    # cv2.imshow()

    # small_heatmap = np.reshape(small_heatmap, (MAX_RES_Y, MAX_RES_X, 3))


    heatmap = cv2.resize(small_heatmap, WINDOW_DIMENSIONS, interpolation=cv2.INTER_LINEAR)
    # heatmap = cv2.resize(small_heatmap, (1000, 1000), interpolation=cv2.INTER_NEAREST)
    return heatmap

def calculate_heatmap2_(img):
    """"""
    lmax = np.max(img)

    # image /= lmax

    small_heatmap = np.zeros((MAX_RES_Y, MAX_RES_X, 3), dtype=np.uint8)

    kernel = np.ones((6,6))

    img2 = np.ones_like(img)
    loc_max = cv2.dilate(img, kernel) == img
    res = np.int8(img2 * loc_max)

    if lmax>1e-8:
        for x in range(MAX_RES_X):
            for y in range(MAX_RES_Y):
                d = res[x, y]

                if d > 0.9:
                    val = int(255 * d ** MISO_POWER)

                    small_heatmap[MAX_RES_Y - 1 - y, x] = colors[val]


    heatmap = cv2.resize(small_heatmap, WINDOW_DIMENSIONS, interpolation=cv2.INTER_LINEAR)

    

    return heatmap


def local_max(img: np.ndarray, threshold: float) -> np.ndarray:
    padded_img = np.pad(img, ((1,1),(1,1)), constant_values=-np.inf)

    # Determines if each location is bigger than adjacent neighbors
    adjacentmax =(
    (padded_img[1:-1,1:-1] > threshold) &
    (padded_img[0:-2,1:-1] <= padded_img[1:-1,1:-1]) &
    (padded_img[2:,  1:-1] <= padded_img[1:-1,1:-1]) &
    (padded_img[1:-1,0:-2] <= padded_img[1:-1,1:-1]) &
    (padded_img[1:-1,2:  ] <= padded_img[1:-1,1:-1])
    )

    return adjacentmax


from scipy.ndimage import gaussian_filter
def calculate_heatmap2(img):

    img = gaussian_filter(img, sigma=8)
    peaks = local_max(img, threshold=-np.inf)

    small_heatmap = np.zeros((MAX_RES_Y, MAX_RES_X, 3), dtype=np.uint8)

    dd = np.copy(img)

    old_max = dd.max()

    dd /= old_max

    dd **=10

    r = 4

    rang = (np.log10(old_max) + 10) / r

    dd *= rang

    for x in range(MAX_RES_X):
        for y in range(MAX_RES_Y):
            d = dd[x, y]

            if d > 0.4:
                val = max(min(int(255 * d), 255), 0)
                small_heatmap[MAX_RES_Y - 1 - y, x] = colors[val]

    heatmap = cv2.resize(small_heatmap, WINDOW_DIMENSIONS, interpolation=cv2.INTER_LINEAR)

    return heatmap

import queue
from multiprocessing import JoinableQueue, Value


class Viewer:
    def __init__(self, src="/dev/video2"):
        self.src = src
        self.capture = cv2.VideoCapture(self.src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, APPLICATION_WINDOW_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, APPLICATION_WINDOW_HEIGHT)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    def loop(self, q: JoinableQueue, v: Value):
        while v.value == 1:
            try:
                output = q.get(block=False)
                q.task_done()
                status, frame = self.capture.read()
                frame = cv2.flip(frame, 1) # Nobody likes looking out of the array :(
                try:
                    frame = cv2.resize(frame, WINDOW_DIMENSIONS)
                except cv2.error as e:
                    print("An error ocurred with image processing! Check if camera and antenna connected properly")
                    v.value = 0
                    break

                res = calculate_heatmap2(output)

                image = cv2.addWeighted(frame, 0.6, res, 0.8, 0)

                cv2.imshow(APPLICATION_NAME, image)
                cv2.waitKey(1)
            except queue.Empty:
                pass
            except KeyboardInterrupt:
                v.value = 0
                break