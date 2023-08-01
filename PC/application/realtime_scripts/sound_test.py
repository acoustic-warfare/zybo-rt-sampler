from threading import Thread
from modules.RealtimeSoundplayerNew import RealtimeSoundplayer
from modules.Beamformer import Beamformer
from modules.VideoPlayer import VideoPlayer
from modules.Replay import Replay
from modules.ArgumentParser import ArgParser
import cv2
import os
from lib.microphone_array import connect, disconnect, receive, convolve_backend_2, trunc_backend_2
import numpy as np
import config
import time
import phase_shift_algorithm_peak_detection

def display_video_sound_heatmap(src, replayMode, replayNumber, convolveBackend):
    #Create a SoundPlayer object
    soundPlayer = RealtimeSoundplayer(receive=receive)
    connect(True)
    #convolve_backend_2(src, replayMode)
    capture = cv2.VideoCapture(src)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    
    FPS = 1/256
    FPS_MS = int(FPS * 1000)
    #FPS_MS = 1

    #Prepare a replay transmission
    if replayMode:
        replay = Replay(replayNumber)

    #Start a replay transmission
    if replayMode:
        thread1 = Thread(target=replay.beginReplayTransmission, args=())
        thread1.daemon = True
        thread1.start()
    #time.sleep(1)
    #Play sound
    thread2 = Thread(target=soundPlayer.play_sound, args=())
    thread2.daemon = True
    thread2.start()

    #time.sleep(2)
    
    times = np.array([])
    i = 0
    while True:
            start = time.time()
            status, frame = capture.read()
            try:
                frame = cv2.resize(frame, config.WINDOW_SIZE)
            except cv2.error as e:
                print("An error ocurred with image processing! Check if camera and antenna connected properly")
                os.system("killall python3")

            data = np.empty((config.N_MICROPHONES, config.N_SAMPLES), dtype=np.float32)
            receive(data)

            #print(np.shape(frame))
            
            heatmap_data = phase_shift_algorithm_peak_detection.main(data.T)
            
            heatmap_data *= 255
            heatmap = heatmap_data.astype('uint8')
            heatmap = cv2.resize(heatmap, (720, 480))
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            #print(np.shape(heatmap_data))

            dst = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
            if config.FLIP_IMAGE:
                dst = cv2.flip(dst, 1)
            #dst = cv2.applyColorMap(dst, cv2.COLORMAP_JET)
            dst = cv2.resize(dst, (1920, 1080))
            cv2.imshow(config.APPLICATION_NAME, dst)
            cv2.waitKey(FPS_MS)
            end = time.time()

            if i > 6:
                sim_time = round((end - start), 4)  # single loop simulation time
                #print('Individual loop time:', sim_time, 's')
                times = np.append(times, sim_time)
                avg_time = round(np.sum(times)/(i-3), 4)
                print('Avg simulation time:', avg_time, 's')
                print('FPS =', 1/avg_time)
            i += 1
if __name__ == '__main__':
    #Parse input arguments
    argumentParser = ArgParser()
    
    display_video_sound_heatmap(argumentParser.getSrc(), argumentParser.getReplayMode(), argumentParser.getReplayNumber(), argumentParser.getBeamformingAlgorithm())
