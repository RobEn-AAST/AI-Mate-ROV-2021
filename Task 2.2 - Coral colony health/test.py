from object_detection_module import adjust_angle
import cv2




im1 = cv2.imread("reference5.png")
im2  = cv2.imread("reference6.png")



print(adjust_angle(im1,im2))