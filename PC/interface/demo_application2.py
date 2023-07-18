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
    #Create a SoundPlayer object
    soundPlayer = RealtimeSoundplayer(receive=receive)

    #Prepare a replay transmission
    if replayMode:
        replay = Replay(replayNumber)

    #Start a replay transmission
    if replayMode:
        thread1 = Thread(target=replay.beginReplayTransmission, args=())
        thread1.daemon = True
        thread1.start()
    time.sleep(0.1)

    #Prepare selected backed
    if convolveBackend:
        thread2 = Thread(target=convolve_backend, args=(src, replayMode))
    else:
        thread2 = Thread(target=trunc_backend, args=(src, replayMode))
    #Start backend
    thread2.daemon = True
    thread2.start()

    time.sleep(1)
    #Play sound
    soundPlayer.play_sound()

if __name__ == '__main__':
    #Parse input arguments
    argumentParser = ArgParser()
    
    display_video_sound_heatmap(argumentParser.getSrc(), argumentParser.getReplayMode(), argumentParser.getReplayNumber(), argumentParser.getBeamformingAlgorithm())
    