""" Driver code for task 2.2"""
import cv2
from object_detection_module import COLORS, AREA, MIN_MATCH_COUNT, extract
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


def simplest_cb(img, percent=1):
    out_channels = []
    cumstops = (
        img.shape[0] * img.shape[1] * percent / 200.0,
        img.shape[0] * img.shape[1] * (1 - percent / 200.0)
    )
    for channel in cv2.split(img):
        cumhist = np.cumsum(cv2.calcHist([channel], [0], None, [256], (0,256)))
        low_cut, high_cut = np.searchsorted(cumhist, cumstops)
        lut = np.concatenate((
            np.zeros(low_cut),
            np.around(np.linspace(0, 255, high_cut - low_cut + 1)),
            255 * np.ones(255 - high_cut)
        ))
        out_channels.append(cv2.LUT(channel, lut.astype('uint8')))
    return cv2.merge(out_channels)


# Read old image and resize it to the size of the camera dimensions

def colonyhealthfunction():
    global old_image
    cap = cv2.VideoCapture('coral.mp4')

# Get default camera window size
    frame = cap.read()[1]
    frame = simplest_cb(frame)
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
        cv2.imshow("Object Tracker", frame)
        cv2.imshow("extraction result", np.hstack([images['original'], images['purple'], images['white']]))

        # close the stream when enter key is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break


    # Release camera and close any open windows
    cap.release()
    cv2.destroyAllWindows()

# colonyhealthfunction()

# trackbars()
cv2.namedWindow('tracks purple')
cv2.namedWindow('tracks white')
cv2.createTrackbar('purple upper 1', 'tracks purple', 215, 255, nothing)
cv2.createTrackbar('purple upper 2', 'tracks purple', 224, 255, nothing)
cv2.createTrackbar('purple upper 3', 'tracks purple', 165, 255, nothing)

cv2.createTrackbar('purple lower 1', 'tracks purple', 109, 255, nothing)
cv2.createTrackbar('purple lower 2', 'tracks purple', 0, 255, nothing)
cv2.createTrackbar('purple lower 3', 'tracks purple', 55, 255, nothing)

cv2.createTrackbar('white upper 1', 'tracks white', 255, 255, nothing)
cv2.createTrackbar('white upper 2', 'tracks white', 255, 255, nothing)
cv2.createTrackbar('white upper 3', 'tracks white', 255, 255, nothing)

cv2.createTrackbar('white lower 1', 'tracks white', 95, 255, nothing)
cv2.createTrackbar('white lower 2', 'tracks white', 99, 255, nothing)
cv2.createTrackbar('white lower 3', 'tracks white', 109, 255, nothing)


# frame = cv2.imread('test.png')
cap = cv2.VideoCapture('coral.mp4')
# frame = simplest_cb(frame)

while(True):
    try:
        frame = cap.read()[1]
        frame = cv2.resize(frame, (400,400))

        trackbar1up = cv2.getTrackbarPos('purple upper 1', 'tracks purple')
        trackbar2up = cv2.getTrackbarPos('purple upper 2', 'tracks purple')
        trackbar3up = cv2.getTrackbarPos('purple upper 3', 'tracks purple')
        trackbar1lp = cv2.getTrackbarPos('purple lower 1', 'tracks purple')
        trackbar2lp = cv2.getTrackbarPos('purple lower 2', 'tracks purple')
        trackbar3lp = cv2.getTrackbarPos('purple lower 3', 'tracks purple')

        trackbar1uw = cv2.getTrackbarPos('white upper 1', 'tracks white')
        trackbar2uw = cv2.getTrackbarPos('white upper 2', 'tracks white')
        trackbar3uw = cv2.getTrackbarPos('white upper 3', 'tracks white')
        trackbar1lw = cv2.getTrackbarPos('white lower 1', 'tracks white')
        trackbar2lw = cv2.getTrackbarPos('white lower 2', 'tracks white')
        trackbar3lw = cv2.getTrackbarPos('white lower 3', 'tracks white')
        # print(UPPER_PURPLE)
        UPPER_PURPLE = np.array([trackbar1up, trackbar2up, trackbar3up])
        LOWER_PURPLE = np.array([trackbar1lp, trackbar2lp, trackbar3lp])
        orig = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        radius = 9
        gray = cv2.GaussianBlur(gray, (radius, radius), 0)
        _, White_val, _, White_loc = cv2.minMaxLoc(gray)
        # White_val = cv2.cvtColor(White_val, cv2.COLOR_BGR2HSV)
        White_loc = White_loc[1], White_loc[0]
        (a,b,c) = frame[White_loc][0], frame[White_loc][1], frame[White_loc][2]
        hsv_up_white = cv2.cvtColor((a,b,c), cv2.COLOR_BGR2HSV)
        cv2.setTrackbarPos('white upper 1','tracks white',hsv_up_white[0])
        cv2.setTrackbarPos('white upper 2','tracks white',hsv_up_white[1])
        cv2.setTrackbarPos('white upper 3','tracks white',hsv_up_white[2])
 
        print(White_val)
        LOWER_WHITE = np.array([trackbar1lw, trackbar2lw, trackbar3lw])
        UPPER_WHITE = np.array([trackbar1uw, trackbar2uw, trackbar3uw])
        images = extract(frame, UPPER_PURPLE, LOWER_PURPLE, UPPER_WHITE, LOWER_WHITE)
        
        # nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(images['purple'], connectivity=8)
        # sizes = stats[1:, -1]; nb_components = nb_components - 1
        # min_size = cv2.getTrackbarPos('tracks','AREA')
        # img2 = np.zeros((output.shape))
        # for i in range(0, nb_components):
        #     if sizes[i] >= min_size:
        #         img2[output == i + 1] = 255

        
        # cv2.imshow("extraction result", np.hstack([images['original'], images['purple'], images['white']]))
        cv2.imshow("extraction result", np.hstack([images['original'], images['purple'], images['white']]))
        cv2.waitKey(80)
        # print(UPPER_PURPLE)
    except Exception:
        cap = cv2.VideoCapture('coral.mp4')

