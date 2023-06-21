import numpy as np
import ctypes
import sounddevice as sd

def get_antenna_data():

    lib = ctypes.cdll.LoadLibrary("./lib/libsampler.so")


    init = lib.load
    init.restype = int


    get_data = lib.myread
    get_data.restype = None
    get_data.argtypes = [
        ctypes.POINTER(ctypes.c_float)
    ]

    init()

    # N_SAMPLES = 256 * 64
    # out = np.empty(N_SAMPLES, dtype=np.float32)
    # out_pointer = out.ctypes.data_as(
    #     ctypes.POINTER(ctypes.c_float))
    
    # f = lambda arr: get_data(arr.ctypes.data_as(
    #     ctypes.POINTER(ctypes.c_float)))
    
    return get_data

f = get_antenna_data()

signals = np.zeros(256 * 64, dtype=np.float32)

N_SAMPLES = 256 * 64
out = np.empty(N_SAMPLES, dtype=np.float32)
out_pointer = out.ctypes.data_as(
    ctypes.POINTER(ctypes.c_float))


while True:
    f(out_pointer)
    b = out.reshape((256, 64))

    # len(b[b != -1])

    sound = b[:,1]

    sd.play(sound, 48828)

    # print(sound)