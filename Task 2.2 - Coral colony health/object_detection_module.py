""" 
Module that contains utilities needed task 2.2

task description : Using image recognition to determine the health of a coral colony by comparing its 
current condition picture to an image taken in the past to compare between them and find changes.

Prequisites :-

1 - open cv
2 - numpy
3 - math -> if not already included while installing python

"""
from math import pi,atan2
from cv2 import cv2
from numpy import array,ones,uint8,hstack,float32,shape
from numpy.core.numeric import False_


# Defining module Constants
LOWER_PURPLE = array([130, 50, 90]) # Purple HSV lower boundry
UPPER_PURPLE = array([170, 255, 255]) # Purple HSV upper boundry
LOWER_WHITE = array([0, 6, 180]) # White HSV lower boundry
UPPER_WHITE = array([255, 255, 255]) # White HSV upper boundry
KERNEL_ALLIGN = ones((5, 5), uint8) # Kernel used for opening effect
MAX_FEATURES = 100 # Maximum number of features to be detected
GOOD_MATCH_PERCENT = 0.15 # Matching tolerence
AREA = 250 # Minimum contour area
COLORS = {
    "GREEN" : (0, 255, 0),
    "YELLOW" : (0, 255, 255),
    "RED" : (0, 0, 255),
    "BLUE" : (255, 0, 0)
    }
AREA_TOLERANCE = 1000 # Difference in area tolerence
MIN_MATCH_COUNT = 10 # Matching Tolerence


# Objects used for feature matching
flann = cv2.FlannBasedMatcher(dict(algorithm = 1, trees = 5), dict(checks = 50))
bf = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
sift = cv2.SIFT_create()

def check_for_matches(old_image,new_image,debug = False):
    """ 
checks if 2 images match and returns if they match and the number of good matches
    
arguments :-
------------

1 - old_image -> an numpy array representing the old image ( can be read using open cv ).
2 - new_image -> an numpy array representing the new image ( can be read using open cv ).
3 - debug -> boolean that controls if debug mode is on or off.

outputs :-
----------

1 - whether there are matches or not
2 - the number of good matches

    """
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
    """ 
Python function that extracts purple and white parts out of an image 
    
arguments :-
------------

1 - image -> an numpy array representing the image to be extracted from ( can be read using open cv ).
2 - debug -> boolean that controls if debug mode is on or off.

outputs :-
----------

- A dictionary that contains:
    - original image.
    - white parts of image.
    - purple parts of image.
    
    
    """
    purple_mask = cv2.inRange(image, LOWER_PURPLE,UPPER_PURPLE)
    white_mask = cv2.inRange(image, LOWER_WHITE, UPPER_WHITE)
    white  = cv2.bitwise_and(image, image, mask=white_mask)
    purple = cv2.bitwise_and(image, image, mask=purple_mask)
    white[white_mask > 0] = (255, 255, 255)
    purple[purple_mask > 0] = (255, 255, 255)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN,KERNEL_ALLIGN)
   # Print debug info if debug mode is on
    if debug:
        cv2.imshow("extraction result",hstack([image,purple,white]))
    return { 'original' : image,'white' : white,'purple' : purple}

def to_black_and_white(image_dict,debug = False):
    """ 
Python function that changes image to black and white 
    
arguments :-
------------

1 - image-dict -> a dictionary of numpy array representing the image to be extracted from ( can be made using the extract() function ).
2 - debug -> boolean that controls if debug mode is on or off.

    outputs :-
    ----------

- A dictionary that contains:
    - original image in black and white.
    - white parts of image in black and white.
    - purple parts of image in black and white.
    
    """
    image = cv2.cvtColor(image_dict['original'], cv2.COLOR_BGR2GRAY)
    purple = cv2.cvtColor(image_dict['white'], cv2.COLOR_BGR2GRAY)
    white = cv2.cvtColor(image_dict['purple'], cv2.COLOR_BGR2GRAY)
    purple = cv2.threshold(purple, 127, 255, cv2.THRESH_BINARY)[1]
    white = cv2.threshold(white, 127, 255, cv2.THRESH_BINARY)[1]
   # Print debug info if debug mode is on
    if debug:
        cv2.imshow("extraction result",hstack([image,purple,white]))
    return { 'original' : image,'white' : white,'purple' : purple}

def detect(new_image, old_image,debug = False):
    """ 
detects the changes in the new image returning contours at their places
    
arguments :-
------------

1 - old_image -> an numpy array representing the old image ( can be read using open cv ).
2 - new_image -> an numpy array representing the new image ( can be read using open cv ).
3 - debug -> boolean that controls if debug mode is on or off.

outputs :-
----------

- A dictionary that contains:
    - growth contours.
    - death contours.
    - blotching contours.
    - recovery contours.
    
    """
    oldimage = extract(old_image)
    newimage = extract(new_image)
    # find the diff purple part between the 2 images using bitwise xor
    diff_purple = cv2.bitwise_xor(old_image["purple"], newimage["purple"])
    diff_purple = cv2.subtract(diff_purple,old_image["white"])
    diff_purple = cv2.subtract(diff_purple,new_image["white"])
    # growth = new image && common part
    growth = cv2.bitwise_and(newimage["purple"], diff_purple)
    # death = old image && common
    death = cv2.bitwise_and(oldimage["purple"], diff_purple)
    # bloching = white new image && purple old image
    blotching = cv2.bitwise_and(new_image["white"], oldimage["purple"])
    # recovery = purple new image && white old image
    recovery = cv2.bitwise_and(newimage["purple"], oldimage["white"])
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
        cv2.imshow("common purple",diff_purple)
        cv2.imshow("growth",growth)
        cv2.imshow("death",death)
        cv2.imshow("blotching",blotching)
        cv2.imshow("recovery",recovery)
    return { 
        'growth_cnts' : growth_cnts,
        'death_cnts' : death_cnts,
        'blotching_cnts' : blotching_cnts,
        'recovery_cnts' : recovery_cnts
        }

def print_contours(image,contours,color,text,area = 150):
    out  = image.copy()
    """ 
Python function that draw the contours according to a tolerence area
    
arguments :-
------------

1 - image -> an numpy array representing the the image to draw the contours on ( can be read using open cv ).
2 - contours -> the contours to be drawn.
3 - color -> the color of the contours and the text.
4 - area ( optional ) -> tolerence of contours area ( has a default value of 150 )

outputs :-
----------

- The image after printing the contours.
    
    """
    for _c in contours:
        if cv2.contourArea(_c) > area:
            peri = cv2.arcLength(_c, True)
            approx = cv2.approxPolyDP(_c, 0.02 * peri, True)
            _x, _y, _w, _h = cv2.boundingRect(approx)
            cv2.rectangle(out, (_x, _y), (_x + _w, _y + _h), color, 2)
            cv2.putText(out, text , (_x + _w + 20, _y + 20), cv2.FONT_HERSHEY_COMPLEX,0.7,color,2)
    return out

def remove_back_ground(image):
    """ 
utilizes the color extraction function extract() to remove background from image
    
arguments :-
------------

- image -> an numpy array representing the image to remove the area from ( can be read using open cv ).

outputs :-
----------

- The image after removing the background.
    
"""
    parts = extract(image,debug=False)
    return cv2.bitwise_xor(parts["purple"],parts["white"])

def get_colony_area(image):
    """ 
gets a contour rectangle around the white chunk of color and calculate its area 
    
arguments :-
------------

- image -> an numpy array representing the image to be procesed ( can be read using open cv ).

outputs :-
----------

- A number representing the colony area.
    
    """
    image  = cv2.cvtColor(remove_back_ground(image),cv2.COLOR_BGR2GRAY)
    cv2.imwrite("result1.png",image)
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    _c = max(contours, key = cv2.contourArea)
    return float(cv2.contourArea(_c))

def adjust_distance(old_image,frame):
    """ 
compares 2 objects in 2 images and returns the difference in size between the 2 objects in them
    
arguments :-
------------

1 - old_image -> an numpy array representing the old image ( can be read using open cv ).
2 - frame -> an numpy array representing the current frame in the camera stream ( can be read using open cv ).

outputs :-
----------

- Ratio between new object size and old object size.
    
    """
    frame_d = get_colony_area(frame)
    old_image_d = get_colony_area(old_image)
    return frame_d/old_image_d

def find_angle(old_image,new_image):
    """
using homography to find rotation angle 

arguments :-
------------

1 - old_image -> an numpy array representing the old image ( can be read using open cv ).
2 - frame -> an numpy array representing the current frame in the camera stream ( can be read using open cv ).

    
outputs :-
----------

- Rotation angle between old image and current frame.
    
    """
    old_image = cv2.cvtColor(old_image, cv2.COLOR_BGR2GRAY)
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(old_image,None)
    kp2, des2 = sift.detectAndCompute(new_image,None)
    matches = flann.knnMatch(des1,des2,k=2)
# store all the good matches as per Lowe's ratio test.
    good = []
    for _m,_n in matches:
        if _m.distance < 0.7* _n.distance:
            good.append(_m)
    _n = len(good)
    if _n > MIN_MATCH_COUNT:
        src_pts = float32([ kp1[_m.queryIdx].pt for _m in good ]).reshape(-1,1,2)
        dst_pts = float32([ kp2[_m.trainIdx].pt for _m in good ]).reshape(-1,1,2)
    else:
        return 0
    _m = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)[0]
    if shape(_m) == ():
        return -1
    ## derive rotation angle from homography
    return - atan2(_m[0,1], _m[0,0]) * 180 / pi

def adjust_angle(old_image,frame):
    #put your code here ya samir
    pass
    


# End of module
