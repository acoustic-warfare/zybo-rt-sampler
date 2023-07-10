from threading import Thread
import cv2, time
import numpy as np
from modules.RealtimeSoundplayer import RealtimeSoundplayer
from modules.Beamformer import Beamformer
from modules.VideoPlayer import VideoPlayer
import ctypes

import pyaudio
import sys

# Local
import config
import os
rogue_child_stopper = False

def play_sound():
    sound_player = RealtimeSoundplayer()
    sound_player.play_sound()

def display_video_sound_heatmap(src, beamformer, replay_mode, sound_command = ""):
    videoPlayer = VideoPlayer(beamformer, src, replay_mode)
    soundPlayer = RealtimeSoundplayer(beamformer, sound_command=sound_command)
    
    if replay_mode:
        thread3 = Thread(target=soundPlayer.replay_sound, args=())
        thread3.daemon = True
        thread3.start()

    videoPlayer.display()

        

#def test_sound():
    #"""Play sound in the current direction"""
    #out = np.empty(config.N_SAMPLES, dtype=config.NP_DTYPE)
    #while True:
        #get_data(out.ctypes.data_as(
            #ctypes.POINTER(ctypes.c_float)))
        
        #get_image(image)

if __name__ == '__main__':
    print(config.FLIP_IMAGE)
    src = 2
    replay_mode = False
    sound_command = ""
    if len(sys.argv) > 1 and sys.argv[1] == "replay":
        replay_mode = True
        if len(sys.argv) == 3:
            src = "../replays/" + str(sys.argv[2]) + "/replay.avi"
            sound_command = "udpreplay -i lo ../replays/"+ str(sys.argv[2]) + "/replay.pcap"
    
    beamformer = Beamformer(replay_mode)
    display_video_sound_heatmap(src, beamformer, replay_mode, sound_command)
    
    
