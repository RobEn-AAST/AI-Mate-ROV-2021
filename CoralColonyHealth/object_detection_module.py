""" 
Module that contains utilities needed task 2.2

task description : Using image recognition to determine the health of a coral colony by comparing its 
current condition picture to an image taken in the past to compare between them and find changes.

Prequisites :-

1 - open cv
2 - numpy
3 - math -> if not already included while installing python

"""
from cv2 import cv2
import numpy as np


oldimage = cv2.imread("old.png")

# Defining module Constants
LOWER_PURPLE = np.array([100, 50, 140])  # Purple HSV lower boundry
UPPER_PURPLE = np.array([170, 255, 255])  # Purple HSV upper boundry
LOWER_WHITE = np.array([150, 150, 160])  # White HSV lower boundry
UPPER_WHITE = np.array([255, 255, 255])  # White HSV upper boundry
KERNEL_ALLIGN = np.ones((5, 5), np.uint8)  # Kernel used for opening effect
KERNEL_ERODE = np.ones((3,3),np.uint8)
MAX_FEATURES = 100  # Maximum number of features to be detected
GOOD_MATCH_PERCENT = 0.15  # Matching tolerence
AREA = 250  # Minimum contour area
COLORS = {
    "GREEN": (0, 255, 0),
    "YELLOW": (0, 255, 255),
    "RED": (0, 0, 255),
    "BLUE": (255, 0, 0)
}
AREA_TOLERANCE = 1000  # Difference in area tolerence
MIN_MATCH_COUNT = 10  # Matching Tolerence

def extract(image, debug=False):
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
    purple_mask = cv2.inRange(image, LOWER_PURPLE, UPPER_PURPLE)
    white_mask = cv2.inRange(image, LOWER_WHITE, UPPER_WHITE)
    white = cv2.bitwise_and(image, image, mask=white_mask)
    purple = cv2.bitwise_and(image, image, mask=purple_mask)
    white[white_mask > 0] = (255, 255, 255)
    purple[purple_mask > 0] = (255, 255, 255)
    purple = cv2.erode(purple,KERNEL_ERODE,iterations = 1)
    white = cv2.erode(white,KERNEL_ERODE,iterations = 1)
    # Print debug info if debug mode is on
    if debug:
        cv2.imshow("extraction result", np.hstack([image, purple, white]))
        cv2.waitKey(0)
    return {'original': image, 'white': white, 'purple': purple}

old_image_ext = extract(oldimage)


def to_black_and_white(image, debug=False):
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
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Print debug info if debug mode is on
    if debug:
        cv2.imshow("extraction result", image)
    return image


def detect(new_image, debug=False):
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

    newimage = extract(new_image, debug= debug)
    # find the diff purple part between the 2 images using bitwise xor
    diff_purple = cv2.bitwise_xor(old_image_ext["purple"], newimage["purple"])
    diff_purple = cv2.subtract(diff_purple, old_image_ext["white"])
    diff_purple = cv2.subtract(diff_purple, newimage["white"])
    # growth = new image && common part
    growth = cv2.bitwise_and(newimage["purple"], diff_purple)
    # death = old image && common
    death = cv2.bitwise_and(old_image_ext["purple"], diff_purple)
    # bloching = white new image && purple old image
    blotching = cv2.bitwise_and(newimage["white"], old_image_ext["purple"])
    # recovery = purple new image && white old image
    recovery = cv2.bitwise_and(newimage["purple"], old_image_ext["white"])
    # apply opening morphology on results  to reduce noise
    growth = cv2.morphologyEx(to_black_and_white(growth), cv2.MORPH_OPEN, KERNEL_ALLIGN)
    death = cv2.morphologyEx(to_black_and_white(death), cv2.MORPH_OPEN, KERNEL_ALLIGN)
    blotching = cv2.morphologyEx(to_black_and_white(blotching), cv2.MORPH_OPEN, KERNEL_ALLIGN)
    recovery = cv2.morphologyEx(to_black_and_white(recovery), cv2.MORPH_OPEN, KERNEL_ALLIGN)
    # find the contours
    growth_cnts = cv2.findContours(growth, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    death_cnts = cv2.findContours(death, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    blotching_cnts = cv2.findContours(blotching, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    recovery_cnts = cv2.findContours(recovery, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    # draw the contours on the preview image if debug mode is on
    if debug:
        cv2.imshow("common purple", diff_purple)
        cv2.imshow("growth", growth)
        cv2.imshow("death", death)
        cv2.imshow("blotching", blotching)
        cv2.imshow("recovery", recovery)
        cv2.waitKey(0)
    return {
        'growth_cnts': growth_cnts,
        'death_cnts': death_cnts,
        'blotching_cnts': blotching_cnts,
        'recovery_cnts': recovery_cnts
    }


def print_contours(image, contours, color, text, area=150):
    if contours == None:
        return image.copy()
    out = image.copy()
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
            cv2.putText(out, text, (_x + _w + 20, _y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
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
    parts = extract(image, debug=False)
    return cv2.bitwise_xor(parts["purple"], parts["white"])


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
    image = cv2.cvtColor(remove_back_ground(image), cv2.COLOR_BGR2GRAY)
    cv2.imwrite("result1.png", image)
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    _c = max(contours, key=cv2.contourArea)
    return float(cv2.contourArea(_c))


def adjust_distance(frame):
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
    old_image_d = get_colony_area(oldimage)
    return frame_d / old_image_d
# End of module
