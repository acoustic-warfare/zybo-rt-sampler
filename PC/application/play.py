import pyaudio

from lib.beamformer import connect, receive, disconnect

import config
import numpy as np
import time

class RealtimeSoundplayer(object):
    def __init__(self, mic_index = 4):
        self.data = np.zeros((config.N_MICROPHONES, config.N_SAMPLES), dtype=np.float32)

        self.mic_index = mic_index
        #connect()
    def this_callback(self, in_data, frame_count, time_info, status):
        """This is a pyaudio callback when an output is finished and new data should be gathered"""
        receive(self.data)
        # print(self.data.T[0])


        #antenna_array = self.data.reshape((config.N_MICROPHONES, config.N_SAMPLES))
        #sound = antenna_array[4]
        #print(sound)
        #sound = self.data[4]
        #sound = np.ascontiguousarray(self.data[2] * 2.0)
        # sound = in_data
        return self.data[30]/1.0, pyaudio.paContinue

    def play_sound(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=int(config.fs),
                        input=False,
                        output=True,
                        stream_callback=self.this_callback,
                        frames_per_buffer=config.N_SAMPLES)

        stream.start_stream()

        while stream.is_active():
            print("Running")
            #time.sleep(0.1)
            
        stream.stop_stream()
        stream.close()

        p.terminate()

        # exit()

if __name__ == "__main__":
    connect()
    try:
        r = RealtimeSoundplayer()
        r.play_sound()
    finally:
        disconnect()
