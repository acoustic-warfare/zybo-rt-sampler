from threading import Thread
from modules.RealtimeSoundplayerNew import RealtimeSoundplayer
from modules.Beamformer import Beamformer
from modules.VideoPlayer import VideoPlayer
from modules.Replay import Replay
from modules.ArgumentParser import ArgParser
from lib.microphone_array import convolve_backend, trunc_backend, receive
import numpy as np
import config
import time

def display_video_sound_heatmap(src, replayMode, replayNumber, convolveBackend):
    #videoPlayer = VideoPlayer(beamformer, src, replayMode)
    soundPlayer = RealtimeSoundplayer(receive=receive)

    if replayMode:
        replay = Replay(replayNumber)

    if replayMode:
        thread3 = Thread(target=replay.beginReplayTransmission, args=())
        thread3.daemon = True
        thread3.start()
    time.sleep(0.1)

    if convolveBackend:
        thread1 = Thread(target=convolve_backend, args=(src, replayMode))
    else:
        thread1 = Thread(target=trunc_backend, args=(src, replayMode))

    thread1.daemon = True
    thread1.start()

    data = np.zeros((config.N_MICROPHONES, config.N_SAMPLES), dtype=np.float32)
    time.sleep(2)
    soundPlayer.play_sound()
    #thread1 = Thread(target=main, args=(replayMode, src))
    #thread1.daemon = True
    #thread1.start()
    time.sleep(1000)
    #soundPlayer.get_samples()

if __name__ == '__main__':
    argumentParser = ArgParser()
    
    display_video_sound_heatmap(argumentParser.getSrc(), argumentParser.getReplayMode(), argumentParser.getReplayNumber(), argumentParser.getBeamformingAlgorithm())
    