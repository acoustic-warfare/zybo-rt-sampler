from threading import Thread
import cv2, time
import numpy as np
from modules.RealtimeSoundplayer import RealtimeSoundplayer
from modules.Beamformer import Beamformer
from modules.VideoPlayer import VideoPlayer
from modules.Replay import Replay
from modules.ArgumentParser import ArgumentParser, ArgParser

import ctypes

import pyaudio
import sys

# Local
import config
import os

def display_video_sound_heatmap(src, beamformer, replayMode, replayNumber):
    videoPlayer = VideoPlayer(beamformer, src, replayMode)
    soundPlayer = RealtimeSoundplayer(beamformer)

    if replayMode:
        replay = Replay(replayNumber, 6)

    thread1 = Thread(target=soundPlayer.play_sound, args=())
    thread1.daemon = True
    thread1.start()

    if replayMode:
        thread3 = Thread(target=replay.beginReplayTransmission, args=())
        thread3.daemon = True
        thread3.start()

    videoPlayer.display()


if __name__ == '__main__':
    argumentParser = ArgParser()
    
    beamformer = Beamformer(argumentParser.getReplayMode())
    display_video_sound_heatmap(argumentParser.getSrc(), beamformer, argumentParser.getReplayMode(), argumentParser.getReplayNumber())
    
    
