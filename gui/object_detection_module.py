""" 
Module that contains utilities needed task 2.2

task description : Using image recognition to determine the health of a coral colony by comparing its 
current condition picture to an image taken in the past to compare between them and find changes.

Prequisites :-

1 - open cv
2 - numpy
3 - math -> if not already included while installing python

"""
import cv2
import numpy as np

# Defining module Constants
UPPER_PURPLE = np.array([215, 224, 165])
LOWER_PURPLE = np.array([109, 0, 55])
LOWER_WHITE = np.array([150, 150, 160])  # White HSV lower boundry
UPPER_WHITE = np.array([255, 255, 255])  # White HSV upper boundry
KERNEL_ALLIGN = np.ones((5, 5), np.uint8)  # Kernel used for opening effect
KERNEL_ERODE = np.ones((3,3),np.uint8)
MAX_FEATURES = 100  # Maximum number of features to be detected
GOOD_MATCH_PERCENT = 0.15  # Matching tolerence
AREA = 450  # Minimum contour area
COLORS = {
    "GREEN": (0, 255, 0),
    "YELLOW": (0, 255, 255),
    "RED": (0, 0, 255),
    "BLUE": (255, 0, 0)
}
AREA_TOLERANCE = 1000  # Difference in area tolerence
MIN_MATCH_COUNT = 10  # Matching Tolerence




def extract(image,debug=False):
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
    image = cv2.cvtColor(image ,cv2.COLOR_BGR2HSV)
    purple_mask = cv2.inRange(image, LOWER_PURPLE, UPPER_PURPLE)
    purple = cv2.bitwise_and(image, image, mask=purple_mask)
    purple[purple_mask > 0] = (255, 255, 255)
    purple = cv2.erode(purple,KERNEL_ERODE,iterations = 1)
    # Print debug info if debug mode is on
    if debug:
        cv2.imshow("extraction result", np.hstack([image, purple]))
    return purple


oldimage = cv2.imread("old.jpeg")
width,heigth = oldimage.shape[1],oldimage.shape[0]
oldimage = extract(oldimage)
bf = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)





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


def detect(new_image, oldimage,debug=False):
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
    diff_purple = cv2.bitwise_xor(oldimage, newimage)
    # growth = new image && common part
    growth = cv2.bitwise_and(newimage, diff_purple)
    # death = old image && common
    # bloching = white new image && purple old image
    # apply opening morphology on results  to reduce noise
    growth = cv2.morphologyEx(to_black_and_white(growth), cv2.MORPH_OPEN, KERNEL_ALLIGN)
    # find the contours
    growth_cnts = cv2.findContours(growth, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    # draw the contours on the preview image if debug mode is on
    if debug:
        cv2.imshow("common purple", diff_purple)
        cv2.imshow("growth", growth)
        cv2.waitKey(0)
    return growth_cnts


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
    image = cv2.cvtColor(extract(image), cv2.COLOR_BGR2GRAY)
    cv2.imwrite("result1.png", image)
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    _c = max(contours, key=cv2.contourArea)
    return float(cv2.contourArea(_c))

def Pixelated_Clean_Image_beta(image, kernel=np.ones((5,5),np.uint8), noiseKernel=np.ones((2,2),np.uint8)):
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, noiseKernel)
    dilated = cv2.dilate(opened,kernel,iterations = 1)
    return dilated

def addassistant(background,overlay):
    overlay_tmp = overlay.copy()
    background_tmp = background.copy()
    mask = cv2.inRange(overlay_tmp, np.array([255,255,255]), np.array([255,255,255]))
    background_tmp[mask > 0] = (255, 255, 255)
    return background_tmp
# End of module
