# USAGE
# python align_document.py --template form_w4.png --image scans/scan_01.jpg
# note : not disabling image shows unless we finish project
# import the necessary packages
import numpy as np
import imutils
import cv2

lower_purple = np.array([130, 50, 90])
upper_purple = np.array([170, 255, 255])
lower_white = np.array([0, 3, 220])
upper_white = np.array([255, 255, 255])
kernel = np.ones((5, 5), np.uint8)
kernel2 = np.ones((15, 15), np.uint8)

def detect(new_image, old_image,fgmask):
    # create masks
    
    mask1 = cv2.inRange(old_image, lower_purple,upper_purple)
    mask2 = cv2.inRange(new_image, lower_purple,upper_purple)
    mask3 = cv2.inRange(old_image, lower_white, upper_white)
    mask4 = cv2.inRange(new_image, lower_white, upper_white)
    # extract purple parts and white parts from 2 images
    old_image_purple = cv2.bitwise_and(old_image, old_image, mask=mask1)
    old_image_white = cv2.bitwise_and(old_image, old_image, mask=mask3)
    new_image_purple = cv2.bitwise_and(new_image, new_image, mask=mask2)
    new_image_white = cv2.bitwise_and(new_image, new_image, mask=mask4)
    old_image_purple[mask1 > 0] = (255, 255, 255)
    new_image_purple[mask2 > 0] = (255, 255, 255)
    old_image_white[mask3 > 0] = (255, 255, 255)
    new_image_white[mask4 > 0] = (255, 255, 255)
    old_image_purple_g = cv2.cvtColor(old_image_purple, cv2.COLOR_BGR2GRAY)
    new_image_purple_g = cv2.cvtColor(new_image_purple, cv2.COLOR_BGR2GRAY)
    old_image_white_g = cv2.cvtColor(old_image_white, cv2.COLOR_BGR2GRAY)
    new_image_white_g = cv2.cvtColor(new_image_white, cv2.COLOR_BGR2GRAY)
    (thresh, old_image_purple_g) = cv2.threshold(old_image_purple_g, 127, 255, cv2.THRESH_BINARY)
    (thresh, new_image_purple_g) = cv2.threshold(new_image_purple_g, 127, 255, cv2.THRESH_BINARY)
    (thresh, old_image_white_g) = cv2.threshold(old_image_white_g, 127, 255, cv2.THRESH_BINARY)
    (thresh, new_image_white_g) = cv2.threshold(new_image_white_g, 127, 255, cv2.THRESH_BINARY)
    # apply opening morphology on results  to reduce noise
    old_image_white = cv2.morphologyEx(old_image_white, cv2.MORPH_OPEN,kernel)
    new_image_white = cv2.morphologyEx(new_image_white, cv2.MORPH_OPEN,kernel)
    # find the common purple part between the 2 images using bitwise xor
    common_purple = cv2.bitwise_xor(fgmask, new_image_purple_g)
    cv2.imshow("common purple",common_purple)
    # growth = new image && common part
    growth = cv2.bitwise_and(new_image_purple_g, common_purple)
    # death  = old image && common
    death = cv2.bitwise_and(old_image_purple_g, common_purple)
    # bloching = white new image && purple old image
    blotching = cv2.bitwise_and(new_image_white, old_image_purple)
    # recovery = purple new image && white old image
    recovery = cv2.bitwise_and(new_image_purple, old_image_white)
    # apply opening morphology on results  to reduce noise
    growth = cv2.morphologyEx(growth, cv2.MORPH_OPEN, kernel)
    death = cv2.morphologyEx(death, cv2.MORPH_OPEN, kernel2)
    blotching = cv2.morphologyEx(blotching, cv2.MORPH_OPEN, kernel)
    recovery = cv2.morphologyEx(recovery, cv2.MORPH_OPEN, kernel)
    # transform the growth and death images to the grey scale domain to get contours
    blotching_grey_scale = cv2.cvtColor(blotching, cv2.COLOR_BGR2GRAY)
    recovery_grey_scale = cv2.cvtColor(recovery, cv2.COLOR_BGR2GRAY)
    growth_thresh = cv2.threshold(growth, 127, 255, 0)[1]
    death_thresh = cv2.threshold(death, 127, 255, 0)[1]
    blotching_thresh = cv2.threshold(blotching_grey_scale, 127, 255, 0)[1]
    recovery_thresh = cv2.threshold(recovery_grey_scale, 127, 255, 0)[1]
    # find the contours
    growth_cnts = cv2.findContours(growth_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
    death_cnts = cv2.findContours(death_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
    blotching_cnts = cv2.findContours(blotching_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
    recovery_cnts = cv2.findContours(recovery_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
    # draw the contours on the preview image
    return { 'growth_cnts' : growth_cnts,'death_cnts' : death_cnts,'blotching_cnts' : blotching_cnts,'recovery_cnts' : recovery_cnts }