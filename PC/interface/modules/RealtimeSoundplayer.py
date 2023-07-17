import time
import ctypes
import pyaudio
import numpy as np
import os, signal, sys
# Local
import config
MICROPHONE_INDEX = 4 # 0 - 63

class RealtimeSoundplayer(object):
    def __init__(self, beamformer, mic_index = 4):
        self.f = beamformer.get_antenna_data()[0]
        self.out = np.empty(config.BUFFER_LENGTH, dtype=config.NP_DTYPE)
        self.out_pointer = self.out.ctypes.data_as(
            ctypes.POINTER(ctypes.c_float))
        self.mic_index = mic_index
    
    def this_callback(self, in_data, frame_count, time_info, status):
        """This is a pyaudio callback when an output is finished and new data should be gathered"""
        self.f(self.out_pointer)
        
        antenna_array = self.out.reshape((config.N_MICROPHONES, config.N_SAMPLES))
        sound = antenna_array[0]*45.0#np.ascontiguousarray(antenna_array[self.mic_index])
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

    def get_samples(self):
        print("HEJ")
        self.f(self.out_pointer)
        
        out = np.empty(config.BUFFER_LENGTH, dtype=np.float32)
        out_pointer = out.ctypes.data_as(
            ctypes.POINTER(ctypes.c_float))
        while(True):
            self.f(out_pointer)
            b = out.reshape((config.N_SAMPLES, config.N_MICROPHONES))
            print(b[0])

def signal_handler(signum, frame):
    sys.exit()


if __name__ == "__main__":
    rtsp = RealtimeSoundplayer(MICROPHONE_INDEX)
    rtsp.play_sound()