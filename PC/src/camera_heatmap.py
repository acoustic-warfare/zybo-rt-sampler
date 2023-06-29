# importing the libraries
import cv2
import numpy as np
import random

light_blue = [27, 170, 222]
blue = [66, 106, 253]
dark_blue = [60, 40, 170]
yellow = [250, 250, 20]
orange = [244, 185, 60]
green = [100, 200, 100]
colors = [light_blue, blue, dark_blue, yellow, orange, green]

shape = (30, 32, 3)
tmp_heatmap = np.zeros(shape, dtype=np.uint8)

def simulate_heatmap():
    # Read heatmap and resize
    for i in range(30):
        for j in range(32):
            #Ta ett randomvärde på color
            index = random.randint(0, 5)
            tmp_heatmap[i][j] = colors[index]
    
    tmp_heatmap_2 = cv2.resize(tmp_heatmap, (1280, 720))
    return tmp_heatmap_2

def display_camera():
    # Setup camera
    cap = cv2.VideoCapture(0)

    # Read heatmap and resize
    
    print("IMPORTANT: YOU NEED PHOTO PERMISSION IF YOU ARE AT A SECURE/VITAL INSTALLATION!!!")
    if input("Do you have a valid photo permission? (Y/n)") == "Y":
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            # Add heatmap to current frame
            heatmap = simulate_heatmap()
            dst = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
            cv2.imshow('WebCam', dst)
            if cv2.waitKey(1) == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

display_camera()