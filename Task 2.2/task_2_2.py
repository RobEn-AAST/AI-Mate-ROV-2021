""" Driver code for task 2.2"""
import cv2 as cv
import object_detection_module as od

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

#Define Script variables
FRAME_COUNT = 0 # Frames tracking variable
SAMPLE_COUNTER = 0 # Samples trackingcounter
contours = { 'growth_cnts' : [],'death_cnts' : [],'blotching_cnts' : [] ,'recovery_cnts' : [] }
while True:
    # Capture webcame frame
    frame = cv.flip(cap.read()[1],1)
    is_matching,number_of_matches = od.check_for_matches(frame,old_image,debug = DEBUG)
    # If the current frame matches the old image
    if is_matching:
        #Increment the sample counter by 1
        SAMPLE_COUNTER += 1
        if SAMPLE_COUNTER > SAMPLING_RATE:
            contours = od.detect(frame,old_image,debug = DEBUG) # Detect changes
            SAMPLE_COUNTER = 0 # Reset the sampling counter
            #filter contours to get only large ones which eliminates contours caused by noise
            od.filter_contours(frame,contours.get('growth_cnts'),od.GREEN,"growth",area = od.AREA)
            od.filter_contours(frame,contours.get('death_cnts'),od.YELLOW,"death",area = od.AREA)
            od.filter_contours(frame,contours.get('blotching_cnts'),od.RED,"blotching",area=od.AREA)
            od.filter_contours(frame,contours.get('recovery_cnts'),od.BLUE,"recovery",area=od.AREA)
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
