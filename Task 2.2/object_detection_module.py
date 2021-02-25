import numpy as np
import cv2

lower_purple = np.array([130, 50, 90])
upper_purple = np.array([170, 255, 255])
lower_white = np.array([0, 3, 220])
upper_white = np.array([255, 255, 255])
kernel = np.ones((5, 5), np.uint8)

def extract(image,debug = False):
    purple_mask = cv2.inRange(image, lower_purple,upper_purple)
    white_mask = cv2.inRange(image, lower_white, upper_white)
    white  = cv2.bitwise_and(image, image, mask=white_mask)
    purple = cv2.bitwise_and(image, image, mask=purple_mask)
    white[white_mask > 0] = (255, 255, 255)
    purple[purple_mask > 0] = (255, 255, 255)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN,kernel)
    if debug:
        cv2.imshow("extraction result",np.hstack([image,purple,white]))
    return { 'original' : image,'white' : white,'purple' : purple}

def togrey(image_dict,debug = False):
    image = cv2.cvtColor(image_dict.get('original'), cv2.COLOR_BGR2GRAY)
    purple = cv2.cvtColor(image_dict.get('white'), cv2.COLOR_BGR2GRAY)
    white = cv2.cvtColor(image_dict.get('purple'), cv2.COLOR_BGR2GRAY)
    purple = cv2.threshold(purple, 127, 255, cv2.THRESH_BINARY)[1]
    white = cv2.threshold(white, 127, 255, cv2.THRESH_BINARY)[1]
    if debug:
        cv2.imshow("extraction result",np.hstack([image,purple,white]))
    return { 'original' : image,'white' : white,'purple' : purple}

def detect(new_image, old_image,fgmask,debug = False):
    oldimage = extract(old_image)
    newimage = extract(new_image)
    oldimage_g = togrey(oldimage)
    newimage_g = togrey(newimage)
    # find the common purple part between the 2 images using bitwise xor
    common_purple = cv2.bitwise_xor(fgmask, newimage_g.get('purple'))
    common_purple = cv2.subtract(common_purple,oldimage_g.get('white'))
    common_purple = cv2.subtract(common_purple,newimage_g.get('white'))
    # growth = new image && common part
    growth = cv2.bitwise_and(newimage_g.get('purple'), common_purple)
    # death  = old image && common
    death = cv2.bitwise_and(oldimage_g.get('purple'), common_purple)
    # bloching = white new image && purple old image
    blotching = cv2.bitwise_and(oldimage.get('white'), oldimage.get('purple'))
    # recovery = purple new image && white old image
    recovery = cv2.bitwise_and(newimage.get('purple'), oldimage.get('white'))
    # apply opening morphology on results  to reduce noise
    growth = cv2.morphologyEx(growth, cv2.MORPH_OPEN, kernel)
    death = cv2.morphologyEx(death, cv2.MORPH_OPEN, kernel)
    blotching = cv2.morphologyEx(blotching, cv2.MORPH_OPEN, kernel)
    recovery = cv2.morphologyEx(recovery, cv2.MORPH_OPEN, kernel)
    # transform the growth and death images to the grey scale domain to get contours
    blotching = cv2.cvtColor(blotching, cv2.COLOR_BGR2GRAY)
    recovery = cv2.cvtColor(recovery, cv2.COLOR_BGR2GRAY)
    blotching = cv2.threshold(blotching, 127, 255, 0)[1]
    recovery = cv2.threshold(recovery, 127, 255, 0)[1]
    # find the contours
    growth_cnts = cv2.findContours(growth, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
    death_cnts = cv2.findContours(death, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
    blotching_cnts = cv2.findContours(blotching, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
    recovery_cnts = cv2.findContours(recovery, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
    # draw the contours on the preview image
    if debug:
        cv2.imshow("common purple",common_purple)
        cv2.imshow("growth",growth)
        cv2.imshow("death",death)
        cv2.imshow("blotching",blotching)
        cv2.imshow("recovery",recovery)
    return { 'growth_cnts' : growth_cnts,'death_cnts' : death_cnts,'blotching_cnts' : blotching_cnts,'recovery_cnts' : recovery_cnts }