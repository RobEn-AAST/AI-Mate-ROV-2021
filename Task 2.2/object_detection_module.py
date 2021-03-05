import numpy as np
import cv2

# Defining module Constants
LOWER_PURPLE = np.array([130, 50, 90]) # Purple HSV lower boundry
UPPER_PURPLE = np.array([170, 255, 255]) # Purple HSV upper boundry
LOWER_WHITE = np.array([0, 3, 220]) # White HSV lower boundry
UPPER_WHITE = np.array([255, 255, 255]) # White HSV upper boundry
KERNEL_ALLIGN = np.ones((2, 2), np.uint8) # Kernel used for opening effect
MAX_FEATURES = 100 # Maximum number of features to be detected
GOOD_MATCH_PERCENT = 0.15 # Matching tolerence
AREA = 250 # Minimum contour area
GREEN = (0, 255, 0) # Green BGR code
YELLOW = (0, 255, 255) # Yellow BGR code
RED = (0, 0, 255) # Red BGR code
BLUE = (255, 0, 0) # Blue BGR code

# Objects used for feature matching
bf = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
orb = cv2.ORB_create(MAX_FEATURES)

# Python function that checks if 2 images match and returns if they match and the number of good matches
def check_for_matches(old,new,debug = False):
    matches = bf.match(cv2.cvtColor(old, cv2.COLOR_BGR2GRAY),cv2.cvtColor(new, cv2.COLOR_BGR2GRAY))
    matches.sort(key=lambda x: x.distance, reverse=False)
  # Remove not so good matches
    matches.sort(key=lambda x: x.distance, reverse=False)
    numgoodmatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numgoodmatches]
    num_of_matches = len(matches)
   # Print debug info if debug mode is on
    if debug:
        print("matching percentage = " + str( (numgoodmatches/MAX_FEATURES) * 100))
    return (num_of_matches >= MAX_FEATURES,numgoodmatches)

#python function that uses feature matching + homography to allign the new image
def align(old,new,debug = False):
    keypoints1, descriptors1 = orb.detectAndCompute(cv2.cvtColor(old, cv2.COLOR_BGR2GRAY), None)
    keypoints2, descriptors2 = orb.detectAndCompute(cv2.cvtColor(new, cv2.COLOR_BGR2GRAY), None)
    matches = bf.match(descriptors1, descriptors2, None)
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    h = cv2.findHomography(points1, points2, cv2.RANSAC)[0]
    height, width = (old.shape[0],old.shape[1])
    output = cv2.warpPerspective(new, h, (width, height))
   # Print debug info if debug mode is on    
    if debug:
        cv2.imshow("alligned image",output)
    return output

# Python function that extracts purple and white parts out of an image
def extract(image,debug = False):
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

# Python function that changes image to black and white
def to_black_and_white(image_dict,debug = False):
    image = cv2.cvtColor(image_dict.get('original'), cv2.COLOR_BGR2GRAY)
    purple = cv2.cvtColor(image_dict.get('white'), cv2.COLOR_BGR2GRAY)
    white = cv2.cvtColor(image_dict.get('purple'), cv2.COLOR_BGR2GRAY)
    purple = cv2.threshold(purple, 127, 255, cv2.THRESH_BINARY)[1]
    white = cv2.threshold(white, 127, 255, cv2.THRESH_BINARY)[1]
   # Print debug info if debug mode is on    
    if debug:
        cv2.imshow("extraction result",np.hstack([image,purple,white]))
    return { 'original' : image,'white' : white,'purple' : purple}

# Python function that detects the changes in the new image returning contours at their places
def detect(new_image, old_image,debug = False):
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

# Python function that filters the contours according to a tolerence area
def filter_contours(image,contours,color,text,area = 150):
    for c in contours:
        if cv2.contourArea(c) > area:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, text , (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7,color, 2)