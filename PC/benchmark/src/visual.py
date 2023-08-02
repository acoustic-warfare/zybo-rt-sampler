import cv2

import numpy as np
import matplotlib.pyplot as plt

from interface.config import *

WINDOW_DIMENSIONS = (APPLICATION_WINDOW_WIDTH, APPLICATION_WINDOW_HEIGHT)
WINDOW_DIMENSIONS = (1920,1080)
APPLICATION_NAME = "Demo App"

POWER = 5

try:
    from lib.kf import *
    kf = CyKF()
except ModuleNotFoundError:
    print("Unable to find kalman filter, has it been compiled?")

def generate_color_map(name="jet") -> np.ndarray:
    """Create a color lookup table for values between 0 - 255

    Args:
        name (str, optional): Matplotlib CMap. Defaults to "jet".

    Returns:
        np.ndarray: the lookup-table
    """
    
    cmap = plt.cm.get_cmap(name)

    # cdef np.ndarray[np.uint8_t, ndim=2, mode="c"] colors 

    # Generate color lookup table
    colors = np.empty((256, 3), dtype=np.uint8)

    for i in range(256):
        colors[i] = (np.array(cmap(255 - i)[:3]) * 255).astype(np.uint8)

    return colors


colors = generate_color_map()


def calculate_heatmap(image, threshold=5e-8, amount = 0.5, exponent = POWER):
    """Create a heatmap over the perceived powerlevel

    Args:
        image (np.ndarray[MAX_RES_X, MAX_RES_Y]): The calculated powerlevels for each anlge
        threshold (float, optional): minimum max value to print out. Defaults to 5e-8.

    Returns:
        (heatmap, bool): the calculated heatmap and if it should be output or not
    """
    # placeholder
    should_overlay = False
    small_heatmap = np.zeros((MAX_RES_Y, MAX_RES_X, 3), dtype=np.uint8)
    
    max_power_level = np.max(image)

    # Normalize the image
    image /= max_power_level

    # Only calculate heatmap if the maximum powerlevel is above a certain threshold 
    if max_power_level > threshold:

        should_overlay = True
        # Convert image value in range between [0, 1] to a RGB color value
        for x in range(MAX_RES_X):
            for y in range(MAX_RES_Y):
                power_level = image[x, y]

                # Only paint levels above a certain amount, i.e 50%
                if power_level >= amount:
                    power_level -= amount
                    power_level /= amount

                    # Some heatmaps are very flat, so the power of the power
                    # May give more sharper results
                    color_val = int(255 * power_level ** exponent)

                    # This indexing is a bit strange, but CV2 orders it like this (Same as flip operation)
                    small_heatmap[MAX_RES_Y - 1 - y, MAX_RES_X - 1 - x] = colors[color_val]


    # Must resize to fit camera dimensions
    heatmap = cv2.resize(small_heatmap, WINDOW_DIMENSIONS, interpolation=cv2.INTER_LINEAR)
    
    return heatmap, should_overlay

def calculate_heatmap_fft(image, threshold=5e-8):
    """"""
    lmax = np.max(image)

    # print(lmax)

    # image[image < threshold] = 0.0

    image /= lmax
    should_overlay = False

    # image = image.T

    small_heatmap = np.zeros((11, 11, 3), dtype=np.uint8)
    # small_heatmap = np.zeros((MAX_RES_X, MAX_RES_Y, 3), dtype=np.uint8)

    if lmax>threshold:
        for x in range(11):
            for y in range(11):
                d = image[x, y]

                if d >= 0.5:
                    d -= 0.5
                    d*= 2
                    val = int(255 * d ** POWER)

                    small_heatmap[11 - 1 - y, 11 - 1 - x] = colors[val]
                    should_overlay = True

    heatmap = cv2.resize(small_heatmap, WINDOW_DIMENSIONS, interpolation=cv2.INTER_LINEAR)
    # heatmap = cv2.resize(small_heatmap, (1000, 1000), interpolation=cv2.INTER_NEAREST)
    return heatmap, should_overlay

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
                    val = int(255 * d ** POWER)

                    small_heatmap[MAX_RES_Y - 1 - y, x] = colors[val]
                    # small_heatmap[MAX_RES_Y - 1 - y, MAX_RES_X - 1 - x] = colors[val]


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

    r = 6

    rang = (np.log10(old_max) + 10) / r

    dd *= rang

    

    for x in range(MAX_RES_X):
        for y in range(MAX_RES_Y):
            d = dd[x, y]

            if d > 0.4:
                val = max(min(int(255 * d), 255), 0)
                # small_heatmap[MAX_RES_Y - 1 - y, x] = colors[val]
                small_heatmap[MAX_RES_Y - 1 - y, MAX_RES_X - 1 - x] = colors[val]
    
    x, y = np.unravel_index(dd.argmax(), dd.shape)
    d = dd[x, y]
    # print(d)

    if d > 0.4:
        # small_heatmap[MAX_RES_Y - 1 - y, x] = colors[255]
        small_heatmap[MAX_RES_Y - 1 - y, MAX_RES_X - 1 - x] = colors[255]
        kf.update([x, y, 0])

        x, y, _ = kf.get_state()
        # print(x, y)

        if x < 0:
            x = 0
        elif x >= MAX_RES_X:
            x = MAX_RES_X - 1
        else:
            x = int(x)

        if y < 0:
            y = 0
        elif y >= MAX_RES_Y:
            y = MAX_RES_Y - 1
        else:
            y = int(y)
        
        # small_heatmap[MAX_RES_Y - 1 - y, x] = colors[200]
        small_heatmap[MAX_RES_Y - 1 - y, MAX_RES_X - 1 - x] = colors[200]

        x, y, _ = kf.predict(1)
        # print(x, y)

        if x < 0:
            x = 0
        elif x >= MAX_RES_X:
            x = MAX_RES_X - 1
        else:
            x = int(x)

        if y < 0:
            y = 0
        elif y >= MAX_RES_Y:
            y = MAX_RES_Y - 1
        else:
            y = int(y)
        
        # small_heatmap[MAX_RES_Y - 1 - y, x] = colors[180]
        small_heatmap[MAX_RES_Y - 1 - y, MAX_RES_X - 1 - x] = colors[180]


    heatmap = cv2.resize(small_heatmap, WINDOW_DIMENSIONS, interpolation=cv2.INTER_LINEAR)

    return heatmap, True

import queue
from multiprocessing import JoinableQueue, Value


class Viewer:
    """Test viewer used for outputting calculated heatmaps onto a screen
    """
    def __init__(self, src="/dev/video2"):
        """constructor with the camera source to use

        Args:
            src (str, optional): which camera index to use. Defaults to "/dev/video2".
        """
        self.src = src
        self.capture = cv2.VideoCapture(self.src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, APPLICATION_WINDOW_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, APPLICATION_WINDOW_HEIGHT)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    def loop(self, q: JoinableQueue, v: Value):
        """Threaded or Multiprocessing loop that should not be called by the user

        Args:
            q (JoinableQueue): FIFO containing the latest powermaps from the algorithm
            v (Value): a value that will stop this thread or process when other than 1
        """
        prev = np.zeros((1080, 1920, 3), dtype=np.uint8)
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

                res1, should_overlay = calculate_heatmap(output)

                res = cv2.addWeighted(prev, 0.5, res1, 0.5, 0)
                prev = res

                if should_overlay:
                    image = cv2.addWeighted(frame, 0.9, res, 0.9, 0)
                else:
                    image = frame

                cv2.imshow(APPLICATION_NAME, image)
                cv2.waitKey(1)
            except queue.Empty:
                pass
            except KeyboardInterrupt:
                v.value = 0
                break