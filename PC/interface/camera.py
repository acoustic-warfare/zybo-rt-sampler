# import the opencv library
import cv2
import numpy as np

# define a video capture object
vid = cv2.VideoCapture(2, 200)

#vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
#vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 896)
print(cv2.videoio_registry.getCameraBackends())
while(True):

    # Capture the video frame

    # by frame
    ret, frame = vid.read()
    # print(frame.shape)
    #frame = cv2.resize(frame, (1920, 1080))
    # Display the resulting frame
   # frame = np.random.randint(0,255, (1000, 1000), dtype='uint8')
    #print(frame)
    #frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
