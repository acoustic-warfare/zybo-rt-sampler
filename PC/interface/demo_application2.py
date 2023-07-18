from threading import Thread
from modules.RealtimeSoundplayerNew import RealtimeSoundplayer
from modules.Beamformer import Beamformer
from modules.VideoPlayer import VideoPlayer
from modules.Replay import Replay
from modules.ArgumentParser import ArgParser
from lib.microphone_array import convolve_backend, trunc_backend, receive
import time

def display_video_sound_heatmap(src, replayMode, replayNumber, convolveBackend):
    #videoPlayer = VideoPlayer(beamformer, src, replayMode)
    #soundPlayer = RealtimeSoundplayer(receive=receive)

    if replayMode:
        replay = Replay(replayNumber)

    if replayMode:
        thread3 = Thread(target=replay.beginReplayTransmission, args=())
        thread3.daemon = True
        thread3.start()
    time.sleep(0.1)

    if convolveBackend:
        convolve_backend(src, replayMode)
    else:
        trunc_backend(src, replayMode) 
   # thread1 = Thread(target=main, args=(replayMode, src))
    #thread1.daemon = True
    #thread1.start()
    #time.sleep(1000)
    #soundPlayer.get_samples()

if __name__ == '__main__':
    argumentParser = ArgParser()
    
    display_video_sound_heatmap(argumentParser.getSrc(), argumentParser.getReplayMode(), argumentParser.getReplayNumber(), argumentParser.getBeamformingAlgorithm())
    