from cv2 import cv2
import object_detection_module
import numpy as np


cap = cv2.VideoCapture('coral.mp4')


while True:
    # Capture webcame frame
    frame = cap.read()[1]
    images = object_detection_module.extract(frame)
    cv2.imshow("extraction result", np.hstack([images['original'], images['purple'], images['white']]))
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break


    # Release camera and close any open windows
    cap.release()
    cv2.destroyAllWindows()