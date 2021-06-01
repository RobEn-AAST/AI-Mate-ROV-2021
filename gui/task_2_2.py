""" Driver code for task 2.2"""
import cv2 as cv
from object_detection_module import COLORS, AREA, MIN_MATCH_COUNT
from object_detection_module import check_for_matches, detect, print_contours, addassistant
import object_detection_module

old_image = cv.imread("old.png")
# Initalize camera

# old_image_boxed = detect_image(old_image)
# Define Script constants
SAMPLING_RATE = 250
DEBUG = False  # Select debug mode
MANUAL = False  # Turn manual mode off and on
# Define Script variables
FRAME_COUNT = 0  # Frames tracking variable
SAMPLE_COUNTER = 0  # Samples trackingcounter
SIZE_TOLERENCE = 0.3  # tolerence of size
contours = {'growth_cnts': [], 'death_cnts': [], 'blotching_cnts': [], 'recovery_cnts': []}


# Read old image and resize it to the size of the camera dimensions



def colonyhealthfunction(inputCam):
    global old_image
    cap = cv.VideoCapture(f'udpsrc port=5{inputCam}00 ! application/x-rtp, encoding-name=JPEG,payload=26 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink', cv2.CAP_GSTREAMER)

# Get default camera window size
    frame = cap.read()[1]
    Height, Width = frame.shape[:2]
    old_image = cv.resize(old_image, (Width, Height))
    global FRAME_COUNT
    global SAMPLE_COUNTER
    while True:
        # Capture webcame frame
        frame = cv.flip(cap.read()[1], 1)
        is_matching, number_of_matches = check_for_matches(frame, old_image, debug=DEBUG)
        if is_matching:
            is_matching = detect_image(frame)
            # Increment the sample counter by 1
            (x1,y1),(x2,y2) = is_matching # x1, y1, x2, y2
            frame = frame[y1:y2, x1:x2]
            frame = cv.resize(frame, (old_image_size[1], old_image_size[0]))
            cv.imshow("deb", frame)
            SAMPLE_COUNTER += 1
            # ratio = adjust_distance(old_image=old_image, frame=frame)
            cv.waitKey(0)
            contours = detect(frame, object_detection_module.extract(old_image), debug=DEBUG)  # Detect changes
            SAMPLE_COUNTER = 0  # Reset the sampling counter

            frame = print_contours(frame, contours["growth_cnts"], COLORS["GREEN"], "growth", area=AREA)
            frame = print_contours(frame, contours["death_cnts"], COLORS["YELLOW"], "death", area=AREA)
            frame = print_contours(frame, contours["blotching_cnts"], COLORS["RED"], "blotching", area=AREA)
            frame = print_contours(frame, contours["recovery_cnts"], COLORS["BLUE"], "recovery", area=AREA)

        else:
            SAMPLE_COUNTER = 0  # Reset sample counter
            contours = {'growth_cnts': [], 'death_cnts': [], 'blotching_cnts': [], 'recovery_cnts': []}
        FRAME_COUNT += 1
        frame = addassistant(frame,old_image)
        cv.imshow("Object Tracker", frame)
        # close the stream when enter key is pressed
        if cv.waitKey(30) & 0xFF == ord('q'):
            break


    # Release camera and close any open windows
    cap.release()
    cv.destroyAllWindows()

#colonyhealthfunction()