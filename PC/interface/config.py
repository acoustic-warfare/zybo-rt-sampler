import numpy
import ctypes
#Do not edit this config file! Add constants and expressions in config.json and build with make. 

#General constants for both c and python.
N_MICROPHONES = 192
N_SAMPLES = 1024
N_TAPS = 64
COLUMNS = 8
ROWS = 8
MAX_RES = 20
MAX_RES_X = 20
MAX_RES_Y = 20
Z = 10.0
MAX_ANGLE = 45.0
VIEW_ANGLE = 68.0
UDP_PORT = 21844
SAMPLE_RATE = 48828.0
ELEMENT_DISTANCE = 0.02
ARRAY_SEPARATION = 0.0
ACTIVE_ARRAYS = 1
SKIP_N_MICS = 2
PROPAGATION_SPEED = 340.0
APPLICATION_WINDOW_WIDTH = 720
APPLICATION_WINDOW_HEIGHT = 480
CAMERA_SOURCE = 0
FLIP_IMAGE = 1
APPLICATION_NAME = "BEEEEEAAAAAAM FOOOOOOORMING"
UDP_IP = "10.0.0.1"
UDP_REPLAY_IP = "127.0.0.1"
FPGA_PROTOCOL_VERSION = 2
BUFFER_LENGTH = N_SAMPLES * N_MICROPHONES

#Python specific constants
azimuth = 0.0
elevation = 0.0
columns = 8
rows = 8
distance = 0.02
fs = 48828
propagation_speed = 340.0
TIMEOUT = 30
FLIP_IMAGE = True
DTYPE = ctypes.c_int32
NP_DTYPE = numpy.float32
WINDOW_SIZE = (720, 480)
