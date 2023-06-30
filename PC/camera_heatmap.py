# importing the libraries
import cv2
import numpy as np
import random
from play import play_sound
from multiprocessing import Process

light_blue = [27,  170, 222]
blue       = [66,  106, 253]
dark_blue  = [60,   40, 170]
yellow     = [250, 250,  20]
orange     = [244, 185,  60]
green      = [100, 200, 100]
colors = [light_blue, blue, dark_blue, yellow, orange, green]

shape = (30, 32, 3)
small_heatmap = np.zeros(shape, dtype=np.uint8)

# Very unoptimized. Use numpy functionalities
def simulate_heatmap():
    random_heatmap = np.random.randint(0, 6, (30, 32), dtype=np.uint8)
    for i in range(30):
        for j in range(32):
            small_heatmap[i][j] = colors[random_heatmap[i][j]]
    
    heatmap = cv2.resize(small_heatmap, (1280, 720), interpolation=cv2.INTER_LINEAR)
    return heatmap

def display_camera():
    # Setup camera
    cap = cv2.VideoCapture(0)
#    print("IMPORTANT: YOU NEED PHOTO PERMISSION IF YOU ARE AT A SECURE/VITAL INSTALLATION!!!")
 #   if input("Do you have a valid photo permission? (Y/n)") == "Y":
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Generate a random heatmap
        heatmap = simulate_heatmap()
        # Add heatmap to current frame
        dst = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
        cv2.imshow('WebCam', dst)
        if cv2.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

# This function works but generates errors if antenna is offline
def display_camera_and_sound():
    sound = Process(target=play_sound)
    display = Process(target=display_camera)
    display.start()
    sound.start()
    display.join()
    sound.join()       
    

display_camera_and_sound()