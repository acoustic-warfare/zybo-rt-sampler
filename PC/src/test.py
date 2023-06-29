# importing the libraries
import cv2
import numpy as np

# Setup camera
cap = cv2.VideoCapture(0)

# Read heatmap and resize
heatmap = cv2.imread('/home/mario/Pictures/heatmap.png')
heatmap = cv2.resize(heatmap, (1280, 720))

print("IMPORTANT: YOU NEED PHOTO PERMISSION IF YOU ARE AT A SECURE/VITAL INSTALLATION!!!")
if input("Do you have a valid photo permission? (Y/n)") == "Y":
    count = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Add heatmap to current frame
        dst = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
        cv2.imshow('WebCam', dst)
        if cv2.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
