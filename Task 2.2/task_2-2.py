#Object Tracking
import cv2
import object_detection_module as od
# Initalize camera
cap = cv2.VideoCapture(0)
# Get default camera window size
frame = cap.read()[1]
Height, Width = frame.shape[:2]
old_image = cv2.imread("old.png")
old_image = cv2.resize(old_image,(Width,Height))
while True:
    # Capture webcame frame
    frame = cv2.flip(cap.read()[1],1)
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #update the background model
    fgMask = od.update_background_model(frame)
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv_img, od.LOWER_PURPLE, od.UPPER_PURPLE)
    contours= cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    # Check if contours captured a purple object with an bigger object than minimum area
    if len(contours) > 0 and max(cv2.contourArea(c) for c in contours) <= od.AREA and od.check_for_matches(frame,old_image,debug = False)[0]:
        #detect changes
        contours = od.detect(od.align(old_image,frame,debug = False),old_image,fgMask,debug = False)
        #filter contours to get only large ones which eliminates contours caused by noise
        od.filter_contours(frame,contours.get('growth_cnts'),od.GREEN,"growth",area = od.AREA)
        od.filter_contours(frame,contours.get('death_cnts'),od.YELLOW,"death",area = od.AREA)
        od.filter_contours(frame,contours.get('blotching_cnts'),od.RED,"blotching",area = od.AREA)
        od.filter_contours(frame,contours.get('recovery_cnts'),od.BLUE,"recovery",area = od.AREA)
    cv2.imshow("Object Tracker", frame)
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
# Release camera and close any open windows
cap.release()
cv2.destroyAllWindows()