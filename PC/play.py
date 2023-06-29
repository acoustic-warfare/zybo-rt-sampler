import numpy as np
import ctypes
#import matplotlib.pyplot as plt

#import sounddevice as sd
import pyaudio
import time
import config

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

f = get_antenna_data()


out = np.empty(config.BUFFER_LENGTH, dtype=np.float32)
out_pointer = out.ctypes.data_as(
    ctypes.POINTER(ctypes.c_float))


#choice = int(input("Choose mic"))
def this_callback(in_data, frame_count, time_info, status):
    f(out_pointer)
    b = out.reshape((config.N_SAMPLES, config.N_MICROPHONES))
    sound = b[:,4] / 1.0# / 32.0 #/ 2**15

    return sound, pyaudio.paContinue

def play_sound():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=config.fs,
                    input=False,
                    output=True,
                    stream_callback=this_callback,
                    frames_per_buffer=config.N_SAMPLES)

    stream.start_stream()

    while stream.is_active():  # <--------------------------------------------
        #print(rms)    # may be losing some values if sleeping too long, didn't check
        time.sleep(0.1)

    stream.stop_stream()
    stream.close()

    p.terminate()

    exit()

#play_sound()