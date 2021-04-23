""" Driver code for task 2.2"""
import cv2 as cv
from object_detection_module import COLORS,AREA
from object_detection_module import check_for_matches,detect,print_contours,adjust_distance
# Initalize camera
cap = cv.VideoCapture(0)

# Get default camera window size
frame = cap.read()[1]
Height, Width = frame.shape[:2]

# Read old image and resize it to the size of the camera dimensions
old_image = cv.imread("old.png")
old_image = cv.resize(old_image,(Width,Height))

# Define Script constants
SAMPLING_RATE = 250
DEBUG = False # Select debug mode
MANUAL = False # Turn manual mode off and on
#Define Script variables
FRAME_COUNT = 0 # Frames tracking variable
SAMPLE_COUNTER = 0 # Samples trackingcounter
SIZE_TOLERENCE = 0.3 # tolerence of size
contours = { 'growth_cnts' : [],'death_cnts' : [],'blotching_cnts' : [] ,'recovery_cnts' : [] }
while True:
    # Capture webcame frame
    frame = cv.flip(cap.read()[1],1)
    is_matching,number_of_matches = check_for_matches(frame,old_image,debug = DEBUG)
    # If the current frame matches the old image
    if is_matching:
        #Increment the sample counter by 1
        SAMPLE_COUNTER += 1
        ratio =  adjust_distance(old_image=old_image,frame=frame)
        if ratio <  1 + SIZE_TOLERENCE and ratio > 1 - SIZE_TOLERENCE:
            if SAMPLE_COUNTER > SAMPLING_RATE:
                contours = detect(frame,old_image,debug = DEBUG) # Detect changes
                SAMPLE_COUNTER = 0 # Reset the sampling counter
                #filter contours to get only large ones which eliminates contours caused by noise
                frame = print_contours(frame,contours["growth_cnts"],COLORS["GREEN"],"growth",area = AREA)
                frame = print_contours(frame,contours["death_cnts"],COLORS["YELLOW"],"death",area = AREA)
                frame = print_contours(frame,contours["blotching_cnts"],COLORS["RED"],"blotching",area=AREA)
                frame = print_contours(frame,contours["recovery_cnts"],COLORS["BLUE"],"recovery",area=AREA)
        elif ratio > 1:
            print("move forward !!!")
        elif ratio < 1:
            print("move backward !!!")
    else:
        SAMPLE_COUNTER = 0 # Reset sample counter
        contours = {'growth_cnts':[],'death_cnts':[],'blotching_cnts' :[],'recovery_cnts':[]}
    # Display the result
    cv.imshow("Object Tracker", frame)
    FRAME_COUNT += 1
    #close the stream when enter key is pressed
    if cv.waitKey(1) == 13: #13 is the Enter Key
        break

# Release camera and close any open windows
cap.release()
cv.destroyAllWindows()
