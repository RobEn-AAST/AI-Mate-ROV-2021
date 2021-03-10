""" Module that contains functions , objects and constants used in task 2.2"""
from math import pi
from math import atan2
from cv2 import cv2
import numpy as np


# Defining module Constants
LOWER_PURPLE = np.array([130, 50, 90]) # Purple HSV lower boundry
UPPER_PURPLE = np.array([170, 255, 255]) # Purple HSV upper boundry
LOWER_WHITE = np.array([0, 6, 180]) # White HSV lower boundry
UPPER_WHITE = np.array([255, 255, 255]) # White HSV upper boundry
KERNEL_ALLIGN = np.ones((5, 5), np.uint8) # Kernel used for opening effect
MAX_FEATURES = 100 # Maximum number of features to be detected
GOOD_MATCH_PERCENT = 0.15 # Matching tolerence
AREA = 250 # Minimum contour area
GREEN = (0, 255, 0) # Green BGR code
YELLOW = (0, 255, 255) # Yellow BGR code
RED = (0, 0, 255) # Red BGR code
BLUE = (255, 0, 0) # Blue BGR code
AREA_TOLERANCE = 1000 # Difference in area tolerence

# Objects used for feature matching
bf = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
orb = cv2.ORB_create(MAX_FEATURES)

def check_for_matches(old_image,new_image,debug = False):
    """ checks if 2 images match and returns if they match and the number of good matches """
    #detect matches and sort them
    matches = bf.match(
                        cv2.cvtColor(old_image,cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(new_image,cv2.COLOR_BGR2GRAY)
                        )
    matches.sort(key=lambda x: x.distance, reverse=False)
  # Remove not so good matches
    numgoodmatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numgoodmatches]
    num_of_matches = len(matches)
   # Print debug info if debug mode is on
    if debug:
        print("matching percentage = " + str( (numgoodmatches/MAX_FEATURES) * 100))
    return (num_of_matches >= MAX_FEATURES,numgoodmatches)


def extract(image,debug = False):
    """ Python function that extracts purple and white parts out of an image """
    purple_mask = cv2.inRange(image, LOWER_PURPLE,UPPER_PURPLE)
    white_mask = cv2.inRange(image, LOWER_WHITE, UPPER_WHITE)
    white  = cv2.bitwise_and(image, image, mask=white_mask)
    purple = cv2.bitwise_and(image, image, mask=purple_mask)
    white[white_mask > 0] = (255, 255, 255)
    purple[purple_mask > 0] = (255, 255, 255)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN,KERNEL_ALLIGN)
   # Print debug info if debug mode is on
    if debug:
        cv2.imshow("extraction result",np.hstack([image,purple,white]))
    return { 'original' : image,'white' : white,'purple' : purple}

def to_black_and_white(image_dict,debug = False):
    """ Python function that changes image to black and white """
    image = cv2.cvtColor(image_dict.get('original'), cv2.COLOR_BGR2GRAY)
    purple = cv2.cvtColor(image_dict.get('white'), cv2.COLOR_BGR2GRAY)
    white = cv2.cvtColor(image_dict.get('purple'), cv2.COLOR_BGR2GRAY)
    purple = cv2.threshold(purple, 127, 255, cv2.THRESH_BINARY)[1]
    white = cv2.threshold(white, 127, 255, cv2.THRESH_BINARY)[1]
   # Print debug info if debug mode is on
    if debug:
        cv2.imshow("extraction result",np.hstack([image,purple,white]))
    return { 'original' : image,'white' : white,'purple' : purple}

def detect(new_image, old_image,debug = False):
    """ detects the changes in the new image returning
         contours at their places """
    oldimage = extract(old_image)
    newimage = extract(new_image)
    # find the common purple part between the 2 images using bitwise xor
    common_purple = cv2.bitwise_xor(old_image.get('purple'), newimage.get('purple'))
    common_purple = cv2.subtract(common_purple,old_image.get('white'))
    common_purple = cv2.subtract(common_purple,new_image.get('white'))
    # growth = new image && common part
    growth = cv2.bitwise_and(newimage.get('purple'), common_purple)
    # death  = old image && common
    death = cv2.bitwise_and(oldimage.get('purple'), common_purple)
    # bloching = white new image && purple old image
    blotching = cv2.bitwise_and(oldimage.get('white'), oldimage.get('purple'))
    # recovery = purple new image && white old image
    recovery = cv2.bitwise_and(newimage.get('purple'), oldimage.get('white'))
    # apply opening morphology on results  to reduce noise
    growth = cv2.morphologyEx(to_black_and_white(growth), cv2.MORPH_OPEN, KERNEL_ALLIGN)
    death = cv2.morphologyEx(to_black_and_white(death), cv2.MORPH_OPEN, KERNEL_ALLIGN)
    blotching = cv2.morphologyEx(to_black_and_white(blotching), cv2.MORPH_OPEN, KERNEL_ALLIGN)
    recovery = cv2.morphologyEx(to_black_and_white(recovery), cv2.MORPH_OPEN, KERNEL_ALLIGN)
    # find the contours
    growth_cnts = cv2.findContours(growth, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
    death_cnts = cv2.findContours(death, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
    blotching_cnts = cv2.findContours(blotching, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
    recovery_cnts = cv2.findContours(recovery, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
    # draw the contours on the preview image if debug mode is on
    if debug:
        cv2.imshow("common purple",common_purple)
        cv2.imshow("growth",growth)
        cv2.imshow("death",death)
        cv2.imshow("blotching",blotching)
        cv2.imshow("recovery",recovery)
    return { 'growth_cnts' : growth_cnts,
    'death_cnts' : death_cnts,'blotching_cnts' : blotching_cnts,'recovery_cnts' : recovery_cnts }

def filter_contours(image,contours,color,text,area = 150):
    """ Python function that filters the contours according to a tolerence area"""
    for _c in contours:
        if cv2.contourArea(_c) > area:
            peri = cv2.arcLength(_c, True)
            approx = cv2.approxPolyDP(_c, 0.02 * peri, True)
            _x, _y, _w, _h = cv2.boundingRect(approx)
            cv2.rectangle(image, (_x, _y), (_x + _w, _y + _h), color, 2)
            cv2.putText(image, text , (_x + _w + 20, _y + 20), cv2.FONT_HERSHEY_COMPLEX,0.7,color,2)

def remove_back_ground(image):
    """ utilizes the color extraction function extract() to remove background from image """
    parts = extract(image,debug=False)
    return cv2.bitwise_xor(parts.get('purple'),parts.get('white'))

def get_colony_area(image):
    """ draws rectangle around the white chunk of color and calculate its area """
    image  = cv2.cvtColor(remove_back_ground(image),cv2.COLOR_BGR2GRAY)
    cv2.imwrite("result1.png",image)
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    _c = max(contours, key = cv2.contourArea)
    return cv2.contourArea(_c)

def adjust_distance(oldimage,frame):
    """ compares 2 objects in 2 images and commands the ROV to move in order to adjust them """
    frame_d = get_colony_area(frame)
    oldimage_d = get_colony_area(oldimage)
    if oldimage_d - AREA_TOLERANCE < frame_d and oldimage_d + AREA_TOLERANCE > frame_d :
        return "Distance is perfect !!!"
    elif oldimage_d > frame_d :
        return "move forward !!!"
    else :
        return "move backward !!!"

def find_angle(old_image,new_image):
    """using homography to find rotation angle """
    min_match_count = 10
    sift = cv2.SIFT_create()
    old_image = cv2.cvtColor(old_image, cv2.COLOR_BGR2GRAY)
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(old_image,None)
    kp2, des2 = sift.detectAndCompute(new_image,None)
    flann_index_kdtree = 1
    index_params = dict(algorithm = flann_index_kdtree, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
# store all the good matches as per Lowe's ratio test.
    good = []
    for _m,_n in matches:
        if _m.distance < 0.7* _n.distance:
            good.append(_m)
    _n = len(good)
    if _n > min_match_count:
        src_pts = np.float32([ kp1[_m.queryIdx].pt for _m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[_m.trainIdx].pt for _m in good ]).reshape(-1,1,2)
    else:
        return 0
    _m = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)[0]
    if np.shape(_m) == ():
        return 0
    ## derive rotation angle from homography
    return - atan2(_m[0,1], _m[0,0]) * 180 / pi

# End of module