""" Driver code for task 2.2"""
import cv2
from object_detection_module import COLORS, AREA, MIN_MATCH_COUNT, extract, UPPER_PURPLE, LOWER_PURPLE
from object_detection_module import check_for_matches, detect, print_contours, addassistant
import numpy as np

# Define Script constants
SAMPLING_RATE = 250
DEBUG = False  # Select debug mode
MANUAL = False  # Turn manual mode off and on
# Define Script variables
FRAME_COUNT = 0  # Frames tracking variable
SAMPLE_COUNTER = 0  # Samples trackingcounter
SIZE_TOLERENCE = 0.3  # tolerence of size
contours = {'growth_cnts': [], 'death_cnts': [], 'blotching_cnts': [], 'recovery_cnts': []}
old_image = cv2.imread("old.png")


# trackbars 
def nothing(x):
    pass

# def trackbars():
#     cv2.namedWindow('tracks')
#     cv2.createTrackbar('upper1', 'tracks', 0, 255, nochange)
#     cv2.createTrackbar('upper 2', 'tracks', 0, 255, nochange)
#     cv2.createTrackbar('upper 3', 'tracks', 0, 255, nochange)

#     cv2.createTrackbar('lower 1', 'tracks', 0, 255, nochange)
#     cv2.createTrackbar('lower 2', 'tracks', 0, 255, nochange)
#     cv2.createTrackbar('lower 3', 'tracks', 0, 255, nochange)




# Read old image and resize it to the size of the camera dimensions

def colonyhealthfunction():
    global old_image
    cap = cv2.VideoCapture('coral.mp4')

# Get default camera window size
    frame = cap.read()[1]
    Height, Width = frame.shape[:2]
    old_image = cv2.resize(old_image, (Width, Height))
    global FRAME_COUNT
    global SAMPLE_COUNTER
    while True:
        # Capture webcame frame
        frame = cv2.flip(cap.read()[1], 1)
        images = extract(frame)
        is_matching = check_for_matches(frame, old_image, debug=DEBUG)
        if is_matching:
            # Increment the sample counter by 1
            frame = cv2.resize(frame, (old_image.shape[1], old_image.shape[0]))
            SAMPLE_COUNTER += 1
            contours = detect(frame, extract(old_image), debug=DEBUG)  # Detect changes
            SAMPLE_COUNTER = 0  # Reset the sampling counter

            frame = print_contours(frame, contours["growth_cnts"], COLORS["GREEN"], "growth", area=AREA)
            frame = print_contours(frame, contours["death_cnts"], COLORS["YELLOW"], "death", area=AREA)
            frame = print_contours(frame, contours["blotching_cnts"], COLORS["RED"], "blotching", area=AREA)
            frame = print_contours(frame, contours["recovery_cnts"], COLORS["BLUE"], "recovery", area=AREA)

        else:
            SAMPLE_COUNTER = 0  # Reset sample counter
            contours = {'growth_cnts': [], 'death_cnts': [], 'blotching_cnts': [], 'recovery_cnts': []}
            frame = addassistant(frame,old_image)
        FRAME_COUNT += 1
        cv2.imshow("extraction result", np.hstack([images['original'], images['purple'], images['white']]))

        # close the stream when enter key is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break


    # Release camera and close any open windows
    cap.release()
    cv2.destroyAllWindows()

colonyhealthfunction()

# trackbars()
# cv2.namedWindow('tracks')
# cv2.createTrackbar('upper 1', 'tracks', 255, 255, nothing)
# cv2.createTrackbar('upper 2', 'tracks', 98, 255, nothing)
# cv2.createTrackbar('upper 3', 'tracks', 255, 255, nothing)

# cv2.createTrackbar('lower 1', 'tracks', 0, 255, nothing)
# cv2.createTrackbar('lower 2', 'tracks', 50, 255, nothing)
# cv2.createTrackbar('lower 3', 'tracks', 90, 255, nothing)

# cap = cv2.VideoCapture('coral.mp4')
# et, frame = cap.read()
# while(True):
#     trackbar1u = cv2.getTrackbarPos('upper 1', 'tracks')
#     trackbar2u = cv2.getTrackbarPos('upper 2', 'tracks')
#     trackbar3u = cv2.getTrackbarPos('upper 3', 'tracks')
#     trackbar1l = cv2.getTrackbarPos('lower 1', 'tracks')
#     trackbar2l = cv2.getTrackbarPos('lower 2', 'tracks')
#     trackbar3l = cv2.getTrackbarPos('lower 3', 'tracks')
#     print(UPPER_PURPLE)
#     UPPER_PURPLE = np.array([trackbar1u, trackbar2u, trackbar3u])
#     LOWER_PURPLE = np.array([trackbar1l, trackbar2l, trackbar3l])
   
   
#     frame = cv2.resize(frame, (400,400))
#     images = extract(frame)
#     cv2.imshow("extraction result", np.hstack([images['original'], images['purple'], images['white']]))
#     k = cv2.waitKey(100000) & 0xFF
#     if k == 27:
#         break
#     print(UPPER_PURPLE)

