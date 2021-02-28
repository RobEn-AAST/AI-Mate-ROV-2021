#Object Tracking
import cv2
import object_detection_module
# Initalize camera
cap = cv2.VideoCapture(0)
# Get default camera window size
frame = cap.read()[1]
Height, Width = frame.shape[:2]
old_image = cv2.imread("old.png")
old_image = cv2.resize(old_image,(Width,Height))
area = 250

green = (0, 255, 0)
yellow = (0, 255, 255)
red = (0, 0, 255)
blue = (255, 0, 0)
#create back ground subtraction model using MOG2 algorithm
backSub = cv2.createBackgroundSubtractorMOG2() 
## To change to KNN Algorithm replace the previous line with >> backSub = cv2.createBackgroundSubtractorKNN()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
while True:
    # Capture webcame frame
    frame = cv2.flip(cap.read()[1],1)
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #update the background model
    fgMask = backSub.apply(frame,15000)
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv_img, object_detection_module.lower_purple, object_detection_module.upper_purple)
    contours= cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    if contours.count == 0:
        continue
    max_area = max(cv2.contourArea(c) for c in contours)    
    # Check if contours captured a purple object with an bigger object than minimum area
    if len(contours) > 0 and max_area <= area:
        #detect changes
        contours = object_detection_module.detect(frame,old_image,fgMask,debug = False)
        #filter contours to get only large ones which eliminates contours caused by noise
        object_detection_module.filter_contours(frame,contours.get('growth_cnts'),green,"growth",area = area)
        object_detection_module.filter_contours(frame,contours.get('death_cnts'),yellow,"death",area = area)
        object_detection_module.filter_contours(frame,contours.get('blotching_cnts'),red,"blotching",area = area)
        object_detection_module.filter_contours(frame,contours.get('recovery_cnts'),blue,"recovery",area = area)
    cv2.imshow("Object Tracker", frame)
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
# Release camera and close any open windows
cap.release()
cv2.destroyAllWindows()