#Object Tracking
import cv2
import object_detection_module
# Initalize camera
cap = cv2.VideoCapture(0)
flag = False

# Create empty points array
points = []

# Get default camera window size
frame = cap.read()[1]
Height, Width = frame.shape[:2]
old_image = cv2.imread("old.png")
old_image = cv2.resize(old_image,(Width,Height))
frame_count = 0

#create back ground subtraction model using MOG2 algorithm
backSub = cv2.createBackgroundSubtractorMOG2() 
## To change to KNN Algorithm replace the previous line with >> backSub = cv2.createBackgroundSubtractorKNN()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


while True:

    # Capture webcame frame
    frame = cap.read()[1]
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #update the background model
    fgMask = backSub.apply(frame,15000)
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv_img, object_detection_module.lower_purple, object_detection_module.upper_purple)
    
    _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
    # Create empty centre array to store centroid center of mass
    center =   int(Height/2), int(Width/2)
    if len(contours) > 0:
        
        # Get the largest contour and its center 
        c = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        try:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        except:
            center =   int(Height/2), int(Width/2)
        # Allow only countors that have a larger than 15 pixel radius
        if radius > 25:
            # Draw cirlce and leave the last center creating a trail
            #use our function here
            contours = object_detection_module.detect(cv2.flip(frame, 1),old_image,fgMask,debug = False)
            # Log center points 
            points.append(center)
            frame_count = 0
            flag = True
        else:
            # Count frames 
            frame_count += 1
            flag = False
        
        # If we count 10 frames without object lets delete our trail
        if frame_count == 10:
            points = []
            flag = False
            # when frame_count reaches 20 let's clear our trail 
            frame_count = 0
            
    # Display our object tracker
    frame = cv2.flip(frame, 1)
    area = 150
    if flag:
        for c in contours.get('growth_cnts'):
            if cv2.contourArea(c) > area:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                x, y, w, h = cv2.boundingRect(approx)
                # green
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "growth " , (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7,(0, 255, 0), 2)
        for c in contours.get('death_cnts'):
            if cv2.contourArea(c) > area:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                x, y, w, h = cv2.boundingRect(approx)
                # cyan
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, "death ", (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7,(0, 255, 255), 2)
        for c in contours.get('blotching_cnts'):
            if cv2.contourArea(c) > area:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                x, y, w, h = cv2.boundingRect(approx)
                # blue
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "blotching ", (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7,(0, 0, 255), 2)
        for c in contours.get('recovery_cnts'):
            if cv2.contourArea(c) > area:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # red
                cv2.putText(frame, "recovery " , (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7,(255,0 , 0), 2)
    cv2.imshow("Object Tracker", frame)
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

# Release camera and close any open windows
cap.release()
cv2.destroyAllWindows()