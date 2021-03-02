import cv2
import object_detection_module as od
# Initalize camera
cap = cv2.VideoCapture(0)
# Get default camera window size
frame = cap.read()[1]
Height, Width = frame.shape[:2]
old_image = cv2.imread("old.png")
old_image = cv2.resize(old_image,(Width,Height))
SAMPLING_RATE = 250
UPDATE_RATE = 5
DEBUG = True

frame_count = 0
sample_counter = 0
contours = { 'growth_cnts' : [],'death_cnts' : [],'blotching_cnts' : [] ,'recovery_cnts' : [] }

while True:
    # Capture webcame frame
    frame = cv2.flip(cap.read()[1],1)
    is_matching,number_of_matches = od.check_for_matches(frame,old_image,debug = DEBUG)
    if is_matching:
        sample_counter += 1
        #detect changes
        if sample_counter > SAMPLING_RATE:
            contours = od.detect(frame,old_image,debug = DEBUG)
            sample_counter = 0
        #filter contours to get only large ones which eliminates contours caused by noise
        od.filter_contours(frame,contours.get('growth_cnts'),od.GREEN,"growth",area = od.AREA)
        od.filter_contours(frame,contours.get('death_cnts'),od.YELLOW,"death",area = od.AREA)
        od.filter_contours(frame,contours.get('blotching_cnts'),od.RED,"blotching",area = od.AREA)
        od.filter_contours(frame,contours.get('recovery_cnts'),od.BLUE,"recovery",area = od.AREA)
    else:
        sample_counter = 0
        contours = { 'growth_cnts' : [],'death_cnts' : [],'blotching_cnts' : [] ,'recovery_cnts' : [] }
    cv2.imshow("Object Tracker", frame)
    if DEBUG :
        print("Frame " + str(frame_count) + " has been processed" + (" but no matches found." if not is_matching else ("and found" + str(number_of_matches) +" matches")))
        frame_count += 1
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
# Release camera and close any open windows
cap.release()
cv2.destroyAllWindows()