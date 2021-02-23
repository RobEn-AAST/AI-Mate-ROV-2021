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
old_image = cv2.imread("old.png")

def detect(new_image, old_image = old_image):
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
    # apply opening morphology on results  to reduce noise
    old_image_white = cv2.morphologyEx(old_image_white, cv2.MORPH_OPEN,kernel)
    new_image_white = cv2.morphologyEx(new_image_white, cv2.MORPH_OPEN,kernel)
    # find the common purple part between the 2 images using bitwise xor
    common_purple = cv2.bitwise_xor(old_image_purple, new_image_purple)
    common_purple = cv2.bitwise_xor(common_purple,new_image_white)
    # growth = new image && common part
    growth = cv2.bitwise_and(new_image_purple, common_purple)
    # death  = old image && common
    death = cv2.bitwise_and(old_image_purple, common_purple)
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
    growth_grey_scale = cv2.cvtColor(growth, cv2.COLOR_BGR2GRAY)
    death_grey_scale = cv2.cvtColor(death, cv2.COLOR_BGR2GRAY)
    blotching_grey_scale = cv2.cvtColor(blotching, cv2.COLOR_BGR2GRAY)
    recovery_grey_scale = cv2.cvtColor(recovery, cv2.COLOR_BGR2GRAY)
    growth_thresh = cv2.threshold(growth_grey_scale, 127, 255, 0)[1]
    death_thresh = cv2.threshold(death_grey_scale, 127, 255, 0)[1]
    blotching_thresh = cv2.threshold(blotching_grey_scale, 127, 255, 0)[1]
    recovery_thresh = cv2.threshold(recovery_grey_scale, 127, 255, 0)[1]
    # find the contours
    growth_cnts = cv2.findContours(growth_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
    death_cnts = cv2.findContours(death_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
    blotching_cnts = cv2.findContours(blotching_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
    recovery_cnts = cv2.findContours(recovery_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
    # draw the contours on the preview image
    return { 'growth_cnts' : growth_cnts,'death_cnts' : death_cnts,'blotching_cnts' : blotching_cnts,'recovery_cnts' : recovery_cnts }

def align_and_detect(image1, template = cv2.imread('old.png'), maxfeatures=500, keepercent=0.2,
                 debug=True):
    hsv = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    lower_val = np.array([130, 50, 90])
    upper_val = np.array([170, 255, 255])
    mask = cv2.inRange(hsv, lower_val, upper_val)
    if np.sum(mask) <= 1000:
        return { 'growth_cnts' : [],'death_cnts' : [],'blotching_cnts' : [],'recovery_cnts' : [] }
    # convert both the input image and template to grayscale
    imagegray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    templategray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # use ORB to detect keypoints and extract (binary) local
    # invariant features
    orb = cv2.ORB_create(maxfeatures)
    (kpsa, descsa) = orb.detectAndCompute(imagegray, None)
    (kpsb, descsb) = orb.detectAndCompute(templategray, None)

    # match the features
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsa, descsb, None)

    # sort the matches by their distance (the smaller the distance,
    # the "more similar" the features are)
    matches = sorted(matches, key=lambda x: x.distance)

    # keep only the top matches
    keep = int(len(matches) * keepercent)
    matches = matches[:keep]
    if len(matches) < 50:
        return { 'growth_cnts' : [],'death_cnts' : [],'blotching_cnts' : [],'recovery_cnts' : [] }
    # check to see if we should visualize the matched keypoints
    if debug:
        matchedvis = cv2.drawMatches(image1, kpsa, template, kpsb,
                                     matches, None)
        matchedvis = imutils.resize(matchedvis, width=1000)

    # allocate memory for the keypoints (x,y-coordinates) from the
    # top matches -- we'll use these coordinates to compute our
    # homography matrix
    ptsa = np.zeros((len(matches), 2), dtype="float")
    ptsb = np.zeros((len(matches), 2), dtype="float")

    # loop over the top matches
    for (i, m) in enumerate(matches):
        # indicate that the two keypoints in the respective images
        # map to each other
        ptsa[i] = kpsa[m.queryIdx].pt
        ptsb[i] = kpsb[m.trainIdx].pt

    # compute the homography matrix between the two sets of matched
    # points
    if len(ptsa) == 0 or len(ptsb) == 0:
        return { 'growth_cnts' : [],'death_cnts' : [],'blotching_cnts' : [],'recovery_cnts' : [] }
    H = cv2.findHomography(ptsa, ptsb, method=cv2.RANSAC)[0]
    # use the homography matrix to align the images
    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(image1, H, (w, h))
    old = template
    # return the aligned image
    return detect(old,aligned)