N_MICROPHONES = 64
N_SAMPLES = 128
UDP_PORT = 21844
UDP_IP = "10.0.0.1"
BUFFER_LENGTH = N_SAMPLES * N_MICROPHONES
import ctypes
DTYPE = ctypes.c_int32
azimuth = 0.0
elevation = 0.0
columns = 8
rows = 8
distance = 0.02
fs = 44100.0
propagation_speed = 340.0
TIMEOUT = 30
