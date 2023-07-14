# cython: language_level=3
# distutils: language=c

#TODO Implement adaptive beamforming properly

import cv2
import matplotlib.pyplot as plt

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

# External Configs initially defined in src/config.json
cdef extern from "config.h":
    int N_MICROPHONES
    int N_SAMPLES
    int COLUMNS
    int ROWS
    int MAX_RES_X
    int MAX_RES_Y
    float MAX_ANGLE
    float SAMPLE_RATE
    float ELEMENT_DISTANCE
    float ARRAY_SEPARATION
    int ACTIVE_ARRAYS
    int SKIP_N_MICS
    float PROPAGATION_SPEED
    int MISO_POWER
    float VIEW_ANGLE
    int APPLICATION_WINDOW_WIDTH
    int APPLICATION_WINDOW_HEIGHT
    int CAMERA_SOURCE


WINDOW_DIMENSIONS = (APPLICATION_WINDOW_WIDTH, APPLICATION_WINDOW_HEIGHT)

# C defined functions
cdef extern from "beamformer.h":
    void load_coefficients(int *whole_sample_delay)
    void work_test(float *image)
    int load(bint)
    void myread(float *signal)
    void signal_handler()
    void kill_child()


def calc_r_prime(d):
    half = d/2
    r_prime = np.zeros((2, COLUMNS * ROWS))
    element_index = 0
    for array in range(ACTIVE_ARRAYS):
        for row in range(ROWS):
            for col in range(COLUMNS):
                r_prime[0,element_index] = col * d + half + array*COLUMNS*d + array*ARRAY_SEPARATION - COLUMNS * ACTIVE_ARRAYS * half
                r_prime[1, element_index] = row * d - ROWS * half + half
                element_index += 1
    r_prime[0,:] -= ACTIVE_ARRAYS*ARRAY_SEPARATION/2
    active_mics, n_active_mics = active_microphones()

    r_prime = r_prime[:,active_mics]
    return r_prime

def active_microphones():
    mode = SKIP_N_MICS
    rows = np.arange(0, ROWS, mode)
    columns = np.arange(0, COLUMNS*ACTIVE_ARRAYS, mode)

    arr_elem = ROWS*COLUMNS                       # elements in one array
    mics = np.linspace(0, arr_elem-1, arr_elem)   # mics in one array
    
    microphones = np.linspace(0, arr_elem-1,arr_elem).reshape((ROWS, COLUMNS))

    for a in range(ACTIVE_ARRAYS-1):
        a += 1
        array = mics[0+a*arr_elem : arr_elem+a*arr_elem].reshape((ROWS, COLUMNS))
        microphones = np.hstack((microphones, array))

    active_mics = []
    for r in rows:
        for c in columns:
            mic = microphones[r,c]
            active_mics.append(int(mic))
    return np.sort(active_mics), len(active_mics)

def calculate_delays():
    c = PROPAGATION_SPEED             # from config
    fs = SAMPLE_RATE          # from config
    #N_SAMPLES = N_SAMPLES    # from config
    d = ELEMENT_DISTANCE            # distance between elements, from config

    alpha = VIEW_ANGLE  # total scanning angle (bildvinkel) in theta-direction [degrees], from config
    z_scan = 10  # distance to scanning window, from config

    x_res = MAX_RES_X  # resolution in x, from config
    y_res = MAX_RES_Y  # resolution in y, from config
    AS = 16/9   # aspect ratio, from config

    # Calculations for time delay starts below
    r_prime = calc_r_prime(d)  # matrix holding the xy positions of each microphone
    x_i = r_prime[0,:]                      # x-coord for microphone i
    y_i = r_prime[1,:]                      # y-coord for microphone i

    # outer limits of scanning window in xy-plane
    x_scan_max = z_scan*np.tan((alpha/2)*np.pi/180)
    x_scan_min = - x_scan_max
    y_scan_max = x_scan_max/AS
    y_scan_min = -y_scan_max

    # scanning window
    x_scan = np.linspace(x_scan_min,x_scan_max, x_res).reshape(x_res,1,1)
    y_scan = np.linspace(y_scan_min,y_scan_max,y_res).reshape(1, y_res, 1)
    r_scan = np.sqrt(x_scan**2 + y_scan**2 + z_scan**2) # distance between middle of array to the xy-scanning coordinate

    # calculate time delay (in number of samples)
    samp_delay = (fs/c) * (x_scan*x_i + y_scan*y_i) / r_scan            # with shape: (x_res, y_res, n_active_mics)
    # adjust such that the microphone furthest away from the beam direction have 0 delay
    samp_delay -= np.amin(samp_delay, axis=2).reshape(x_res, y_res, 1)

    return samp_delay
    # active_mics: holds index of active microphones for a specific mode
    # n_active_mics: number of active microphones
    active_mics, n_active_mics = active_microphones()

def calculate_delays_():

    distance = 0.02

    samp_delay = np.zeros((MAX_RES_X, MAX_RES_Y, COLUMNS*ROWS*ACTIVE_ARRAYS), dtype=np.float32)

    for xi, x in enumerate(np.linspace(-MAX_ANGLE, MAX_ANGLE, MAX_RES_X)):
        azimuth = x * -np.pi / 180.0
        x_factor = np.sin(azimuth)
        for yi , y in enumerate(np.linspace(-MAX_ANGLE, MAX_ANGLE, MAX_RES_Y)):
            elevation = y * -np.pi / 180.0
            y_factor = np.sin(elevation)

            smallest = 0

            for row in range(ROWS):
                for col in range(COLUMNS):
                    half = distance / 2.0
                    tmp_col = col * distance - COLUMNS * half + half
                    tmp_row = row * distance - ROWS * half + half

                    tmp_delay = tmp_col * x_factor + tmp_row * y_factor
                    if (tmp_delay < smallest):
                        smallest = tmp_delay

                    samp_delay[xi, yi, row * COLUMNS + col] = tmp_delay

            samp_delay[xi, yi, :] -= smallest

    samp_delay *= SAMPLE_RATE / PROPAGATION_SPEED

    return samp_delay

def get_h(delay, N=8):
    tau = - delay  # Fractional delay [samples].
    epsilon = 1e-9
    n = np.arange(N)

    sinc = n - (8 - 1) / 2 - (0.5 + tau) + epsilon

    h = np.sin(sinc*np.pi)/(sinc*np.pi)

    blackman = 0.42 - 0.5 * np.cos(2*np.pi * n / 8) + 0.08 * np.cos(4 * np.pi * n / 8)

    h *= blackman
    
    # Normalize to get unity gain.
    h /= np.sum(h)

    return h



def calculate_coefficients():

    samp_delay = calculate_delays()

    #whole_sample_delay = samp_delay.astype(np.int32)
    whole_sample_delay = samp_delay.astype(int)
    fractional_sample_delay = samp_delay - whole_sample_delay

    h = np.zeros((MAX_RES_X, MAX_RES_Y, COLUMNS*ROWS*ACTIVE_ARRAYS, 8), dtype=np.float32)

    for x in range(MAX_RES_X):
        for y in range(MAX_RES_Y):
            for i in range(COLUMNS*ROWS*ACTIVE_ARRAYS):
                h[x, y, i] = get_h(fractional_sample_delay[x, y, i])
    
    return whole_sample_delay, h


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

    if lmax>1e-7:
        for x in range(MAX_RES_X):
            for y in range(MAX_RES_Y):
                d = image[x, y]

                if d > 0.9:
                    val = int(255 * d ** MISO_POWER)

                    small_heatmap[MAX_RES_Y - 1 - y, x] = colors[val]


    heatmap = cv2.resize(small_heatmap, WINDOW_DIMENSIONS, interpolation=cv2.INTER_LINEAR)

    return heatmap

def connect(replay_mode: bool = False, verbose=True) -> None:
    assert isinstance(replay_mode, bool), "Replay mode must be either True or False"

    if replay_mode: # True
        if load(1) == -1:
            print("Wrong FPGA protocol data format received, disconnecting")
            disconnect()
    else: # Default for real data
        if load(0) == -1:
            print("Wrong FPGA protocol data format received, disconnecting")
            disconnect()

    if verbose:
        print("Receiver process is forked.\nContinue your program!\n")

def disconnect():
    kill_child()

def receive(signals: np.ndarray[N_MICROPHONES, N_SAMPLES]) -> None:
    assert signals.shape == (N_MICROPHONES, N_SAMPLES), "Arrays do not match shape"
    assert signals.dtype == np.float32, "Arrays dtype do not match"

    cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] sig = np.ascontiguousarray(signals)
    myread(&sig[0, 0])


cdef void loop():

    whole_samples, fractional_samples = calculate_coefficients()

    cdef np.ndarray[int, ndim=3, mode="c"] samples

    samples = np.ascontiguousarray(whole_samples.astype(np.int32))

    load_coefficients(&samples[0, 0, 0])

    connect()

    x = np.zeros((MAX_RES_X, MAX_RES_Y), dtype=DTYPE_arr)

    cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] arr2
    arr2 = np.ascontiguousarray(x)
    
    capture = cv2.VideoCapture(CAMERA_SOURCE)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, APPLICATION_WINDOW_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, APPLICATION_WINDOW_HEIGHT)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    while True:
        work_test(&arr2[0, 0])

        heatmap = calculate_heatmap(arr2)

        status, frame = capture.read()
        frame = cv2.flip(frame, 1) # Nobody likes looking out of the array :(
        try:
            frame = cv2.resize(frame, WINDOW_DIMENSIONS)
        except cv2.error as e:
            print("An error ocurred with image processing! Check if camera and antenna connected properly")
            #os.system("killall python3")
            break

        image = cv2.addWeighted(frame, 0.6, heatmap, 0.8, 0)
        cv2.imshow("Demo", image)
        cv2.waitKey(1)






def main():
    loop()

