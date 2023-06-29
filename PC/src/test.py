# importing the libraries
import cv2
import numpy as np

# Setup camera
cap = cv2.VideoCapture(0)

# Read logo and resize
logo = cv2.imread('/home/mario/Pictures/heatmap.png')
#size = 600
logo = cv2.resize(logo, (1280, 720))

# Create a mask of logo
#mg2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
#ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)

print("IMPORTANT: YOU NEED PHOTO PERMISSION IF YOU ARE AT A SECURE/VITAL INSTALLATION!!!")
if input("Do you have a valid photo permission? (Y/n)") == "Y":
    count = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Region of Image (ROI), where we want to insert logo
        #roi = frame[-size-10:-10, -size-10:-10]
        # Set an index of where the mask is
        #roi[np.where(mask)] = 0
        #roi += logo
        #print(frame.shape)
        #print(logo.shape)
        dst = cv2.addWeighted(frame, 0.7, logo, 0.3, 0)
        cv2.imshow('WebCam', dst)
        if cv2.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
