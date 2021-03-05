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
SAMPLING_RATE = 250 # Using a sampling rate constant to analyze only a sample from the stream to reduce latency
DEBUG = False # Select debug mode

#Define Script variables
frame_count = 0 # Frames tracking variable
sample_counter = 0 # Samples trackingcounter
contours = { 'growth_cnts' : [],'death_cnts' : [],'blotching_cnts' : [] ,'recovery_cnts' : [] } # Dictionary to store the contours

while True:
    
    # Capture webcame frame
    frame = cv.flip(cap.read()[1],1)
    
    is_matching,number_of_matches = od.check_for_matches(frame,old_image,debug = DEBUG)

    # If the current frame matches the old image 
    if is_matching:
    
        #Increment the sample counter by 1
        sample_counter += 1

        if sample_counter > SAMPLING_RATE:
            contours = od.detect(frame,old_image,debug = DEBUG) # Detect changes
            sample_counter = 0 # Reset the sampling counter 
    
        #filter contours to get only large ones which eliminates contours caused by noise
        od.filter_contours(frame,contours.get('growth_cnts'),od.GREEN,"growth",area = od.AREA) # Draw growth contours
        od.filter_contours(frame,contours.get('death_cnts'),od.YELLOW,"death",area = od.AREA) # Draw death contours
        od.filter_contours(frame,contours.get('blotching_cnts'),od.RED,"blotching",area = od.AREA) # Draw bloaching contours
        od.filter_contours(frame,contours.get('recovery_cnts'),od.BLUE,"recovery",area = od.AREA) # Draw recovery contours
    
    # If they dont match
    else:
        sample_counter = 0 # Reset sample counter
        contours = { 'growth_cnts' : [],'death_cnts' : [],'blotching_cnts' : [] ,'recovery_cnts' : [] } # Clear the contours
    
    # Display the result
    cv.imshow("Object Tracker", frame)
    
    # Print debug info if debug mode is on
    if DEBUG :
        print("Frame " + str(frame_count) + " has been processed" + (" but no matches found." if not is_matching else ("and found" + str(number_of_matches) +" matches")))
        frame_count += 1
    
    #close the stream when enter key is pressed
    if cv.waitKey(1) == 13: #13 is the Enter Key
        break

# Release camera and close any open windows
cap.release()
cv.destroyAllWindows()