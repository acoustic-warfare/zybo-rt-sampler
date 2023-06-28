import numpy as np
import ctypes
import config
import matplotlib.pyplot as plt

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
    
    return get_data


def get_samples():
    f = get_antenna_data()
    
    out = np.empty(config.BUFFER_LENGTH, dtype=np.float32)
    out_pointer = out.ctypes.data_as(
        ctypes.POINTER(ctypes.c_float))
    times = np.linspace(0, config.N_SAMPLES/config.fs, num=config.N_SAMPLES)
    while(True):
        f(out_pointer)
        b = out.reshape((config.N_SAMPLES, config.N_MICROPHONES))
        print(b[0])
        #plt.figure(figsize=(15, 5))
        #plt.plot(times, b[:,4])
        #plt.title('Left Channel')
        #plt.ylabel('Signal Value')
        #plt.xlabel('Time (s)')
        #plt.xlim(0, config.N_SAMPLES/config.fs)
        #plt.show()    
        #break;  

get_samples()
exit()