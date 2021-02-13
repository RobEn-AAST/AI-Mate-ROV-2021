import cv2
import numpy as np
 
lower_purple = np.array([130,50,90])
upper_purple = np.array([170,255,255])
lower_white = np.array([0,3,220])
upper_white = np.array([255,255,255])
kernel = np.ones((2,2),np.uint8)

old_image = cv2.cvtColor(cv2.imread("E:\\2.jpeg"), cv2.COLOR_BGR2HSV)
new_image = cv2.cvtColor(cv2.imread("E:\\2.jpeg"), cv2.COLOR_BGR2HSV)

mask1 = cv2.inRange(old_image, lower_purple, upper_purple)
mask2 = cv2.inRange(new_image, lower_purple, upper_purple)
mask3 = cv2.inRange(old_image, lower_white, upper_white)
mask4 = cv2.inRange(new_image, lower_white, upper_white)
old_image_purple = cv2.bitwise_and(old_image,old_image,mask=mask1)
old_image_white = cv2.bitwise_and(old_image,old_image,mask=mask3)
new_image_purple = cv2.bitwise_and(new_image,new_image,mask=mask2)
new_image_white = cv2.bitwise_and(new_image,new_image,mask=mask4)
old_image_purple[mask1>0] = (255,255,255)
new_image_purple[mask2>0] = (255,255,255)
old_image_white[mask3>0] = (255,255,255)
new_image_white[mask4>0] = (255,255,255)

new_image = cv2.bitwise_xor(new_image_white,new_image_purple)
old_image = cv2.bitwise_xor(old_image_purple,old_image_white)
orb = cv2.ORB_create(nfeatures=100000000)
 
kp1, des1 = orb.detectAndCompute(old_image,None)
kp2, des2 = orb.detectAndCompute(new_image,None)
 
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
 
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
print(len(good))