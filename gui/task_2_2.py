""" Driver code for task 2.2"""
import cv2 as cv
from object_detection_module import COLORS, AREA, simplest_cb, adjust_angle, extract
from object_detection_module import check_for_matches, detect, print_contours, adjust_distance
from Detector.darknet import LoadWeights, detect_image

#Load Weights
LoadWeights()

# Initalize camera
cap = cv.VideoCapture(0)

# Get default camera window size
frame = cap.read()[1]
Height, Width = frame.shape[:2]

# Read old image and resize it to the size of the camera dimensions
old_image = cv.imread("old.png")
old_image = simplest_cb(old_image, 1)
(x1,y1),(x2,y2) = detect_image(old_image)
old_image = old_image[y1:y2, x1:x2]
old_image = cv.resize(old_image, (Width, Height))
old_image_ext = extract(old_image, True)
old_image_size = old_image.shape[0:2]
# old_image_boxed = detect_image(old_image)
# Define Script constants
SAMPLING_RATE = 250
DEBUG = True  # Select debug mode
MANUAL = False  # Turn manual mode off and on
# Define Script variables
FRAME_COUNT = 0  # Frames tracking variable
SAMPLE_COUNTER = 0  # Samples trackingcounter
SIZE_TOLERENCE = 0.3  # tolerence of size
contours = {'growth_cnts': [], 'death_cnts': [], 'blotching_cnts': [], 'recovery_cnts': []}
SURENESS = 0
SURENESS_rate = 25
while True:
    # Capture webcame frame
    frame = cv.flip(cap.read()[1], 1)
    frame = simplest_cb(frame, 1)

    # Display the result
    cv.imshow("Object Tracker", frame)
    if cv.waitKey(1) == 13:  # 13 is the Enter Key
        break

    # is_matching, number_of_matches = check_for_matches(frame, old_image, debug=DEBUG)
    is_matching = detect_image(frame)
    # If the current frame matches the old image

    if is_matching is not None:
        print(SURENESS)
        if SURENESS != SURENESS_rate:
            SURENESS += 1
            continue
        else:
            SURENESS = 0
        is_matching = detect_image(frame)
        # Increment the sample counter by 1
        (x1,y1),(x2,y2) = is_matching # x1, y1, x2, y2
        frame = frame[y1:y2, x1:x2]
        frame = cv.resize(frame, (old_image_size[1], old_image_size[0]))
        frame = adjust_angle(frame)
        cv.imshow("deb", frame)
        SAMPLE_COUNTER += 1
        # ratio = adjust_distance(old_image=old_image, frame=frame)
        cv.waitKey(0)
        contours = detect(frame, old_image_ext, debug=DEBUG)  # Detect changes
        SAMPLE_COUNTER = 0  # Reset the sampling counter

        frame = print_contours(frame, contours["growth_cnts"], COLORS["GREEN"], "growth", area=AREA)
        frame = print_contours(frame, contours["death_cnts"], COLORS["YELLOW"], "death", area=AREA)
        frame = print_contours(frame, contours["blotching_cnts"], COLORS["RED"], "blotching", area=AREA)
        frame = print_contours(frame, contours["recovery_cnts"], COLORS["BLUE"], "recovery", area=AREA)

    else:
        SAMPLE_COUNTER = 0  # Reset sample counter
        SURENESS = 0 
        contours = {'growth_cnts': [], 'death_cnts': [], 'blotching_cnts': [], 'recovery_cnts': []}

    FRAME_COUNT += 1
    # close the stream when enter key is pressed
    if cv.waitKey(1) == 13:  # 13 is the Enter Key
        break


# Release camera and close any open windows
cap.release()
cv.destroyAllWindows()
