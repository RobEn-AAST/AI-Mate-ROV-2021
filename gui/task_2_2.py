""" Driver code for task 2.2"""
import cv2
from object_detection_module import COLORS, AREA, MIN_MATCH_COUNT, extract
from object_detection_module import detect, print_contours, addassistant
import numpy as np
import keyboard
# Define Script constants
SAMPLING_RATE = 250
DEBUG = False  # Select debug mode
MANUAL = False  # Turn manual mode off and on
# Define Script variables
FRAME_COUNT = 0  # Frames tracking variable
SAMPLE_COUNTER = 0  # Samples trackingcounter
SIZE_TOLERENCE = 0.3  # tolerence of size

contours = []
assistant = cv2.imread("assistant.jpeg")
oldimage = cv2.imread("old.jpeg")



# Read old image and resize it to the size of the camera dimensions

def colonyhealthfunction():
    global assistant
    global oldimage
    global old_image
    cap = cv2.VideoCapture(f'udpsrc port=5{inputCam}00 ! application/x-rtp, encoding-name=JPEG,payload=26 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink', cv.CAP_GSTREAMER)

# Get default camera window size
    frame = cap.read()[1]
    Height, Width = frame.shape[:2]
    assistant = cv2.resize(assistant, (Width, Height))
    oldimage = cv2.resize(oldimage, (Width, Height))
    global FRAME_COUNT
    global SAMPLE_COUNTER
    while True:
        # Capture webcame frame
        frame = cv2.flip(cap.read()[1], 1)
        tmp = extract(frame)
        if keyboard.is_pressed('b'):
            # Increment the sample counter by 1
            SAMPLE_COUNTER += 1
            contours = detect(frame, oldimage, debug=DEBUG)  # Detect changes
            SAMPLE_COUNTER = 0  # Reset the sampling counter

            frame = print_contours(frame, contours, COLORS["GREEN"], "growth", area=AREA)

        else:
            SAMPLE_COUNTER = 0  # Reset sample counter
            contours = []
            frame = addassistant(frame,assistant)
        FRAME_COUNT += 1
        cv2.imshow("footage", frame)

        # close the stream when enter key is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break


    # Release camera and close any open windows
    cap.release()
    cv2.destroyAllWindows()

colonyhealthfunction()
