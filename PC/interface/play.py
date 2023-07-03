import time
import ctypes

import pyaudio

import numpy as np

# Local
import config

MICROPHONE_INDEX = 4 # 0 - 63

# Load shared library, make sure to compile using make command
def get_antenna_data():
    lib = ctypes.cdll.LoadLibrary("../lib/libsampler.so")

    init = lib.load
    init.restype = int

    get_data = lib.myread
    get_data.restype = None
    get_data.argtypes = [
        ctypes.POINTER(ctypes.c_float)
    ]

    # Initiate antenna from the C side
    init()
    
    return get_data


class RealtimeSoundplayer(object):
    def __init__(self, mic_index = 4):
        self.f = get_antenna_data()
        self.out = np.empty(config.BUFFER_LENGTH, dtype=config.NP_DTYPE)
        self.out_pointer = self.out.ctypes.data_as(
            ctypes.POINTER(ctypes.c_float))
        self.mic_index = mic_index
    
    def this_callback(self, in_data, frame_count, time_info, status):
        """This is a pyaudio callback when an output is finished and new data should be gathered"""
        self.f(self.out_pointer)

        antenna_array = self.out.reshape((config.N_SAMPLES, config.N_MICROPHONES))
        sound = self.simple_beamforming(antenna_array) 
        #sound = np.ascontiguousarray(antenna_array[:, self.mic_index])
        #level = np.sum(np.abs(sound2**2))/sound2.shape[0]

        #first = min(int(level*50*30), 50)

        #print("="*first+" "*(50-first), end="\r")

        #print(level, end="\r")
        #print(sound)
        return sound, pyaudio.paContinue

    def play_sound(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=config.fs,
                        input=False,
                        output=True,
                        stream_callback=self.this_callback,
                        frames_per_buffer=config.N_SAMPLES)

        stream.start_stream()

        while stream.is_active():
            # Do nothing for some time
            time.sleep(0.1)

        stream.stop_stream()
        stream.close()

        p.terminate()

        exit()

    def simple_beamforming(self, antenna_array):
        sum_array = antenna_array.sum(axis=1) / 54
        sound = np.ascontiguousarray(sum_array)
        return sound


if __name__ == "__main__":
    mic = 4 #int(input("Mic index"))
    rtsp = RealtimeSoundplayer(mic)
    rtsp.play_sound()