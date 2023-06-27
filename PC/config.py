# Antenna related
N_MICROPHONES = 64
N_SAMPLES = 128 
BUFFER_LENGTH = N_SAMPLES * N_MICROPHONES

azimuth = 0.0
elevation = 0.0
columns = 8
rows = 8
distance = 0.02
fs = 44100.0
propagation_speed = 340.0

# FPGA related
import ctypes
D_TYPE = ctypes.c_int32
UDP_IP = "0.0.0.0"
UDP_PORT = 21844
TIMEOUT = 30