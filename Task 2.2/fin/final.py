# USAGE
# python align_document.py --template form_w4.png --image scans/scan_01.jpg
# note : not disabling image shows unless we finish project
# import the necessary packages
import numpy as np
import imutils
import cv2


# define images class
class image:
    def __init__(self, img):
        self.original = cv2.resize(img, (500, 500))
        self.hsv = cv2.cvtColor(self.original, cv2.COLOR_BGR2HSV)


class changes_detector:
    def __init__(self):
        self.lower_purple = np.array([130, 50, 90])
        self.upper_purple = np.array([170, 255, 255])
        self.lower_white = np.array([0, 3, 220])
        self.upper_white = np.array([255, 255, 255])
        self.kernel = np.ones((5, 5), np.uint8)
        self.kernel2 = np.ones((15, 15), np.uint8)
    def detect(self, old_image, new_image):
        preview = new_image.original.copy()
        # create masks
        mask1 = cv2.inRange(old_image.hsv, self.lower_purple, self.upper_purple)
        mask2 = cv2.inRange(new_image.hsv, self.lower_purple, self.upper_purple)
        mask3 = cv2.inRange(old_image.hsv, self.lower_white, self.upper_white)
        mask4 = cv2.inRange(new_image.hsv, self.lower_white, self.upper_white)
        # extract purple parts and white parts from 2 images
        old_image_purple = cv2.bitwise_and(old_image.hsv, old_image.hsv, mask=mask1)
        old_image_white = cv2.bitwise_and(old_image.hsv, old_image.hsv, mask=mask3)
        new_image_purple = cv2.bitwise_and(new_image.hsv, new_image.hsv, mask=mask2)
        new_image_white = cv2.bitwise_and(new_image.hsv, new_image.hsv, mask=mask4)
        old_image_purple[mask1 > 0] = (255, 255, 255)
        new_image_purple[mask2 > 0] = (255, 255, 255)
        old_image_white[mask3 > 0] = (255, 255, 255)
        new_image_white[mask4 > 0] = (255, 255, 255)
        # apply opening morphology on results  to reduce noise
        old_image_white = cv2.morphologyEx(old_image_white, cv2.MORPH_OPEN, self.kernel)
        new_image_white = cv2.morphologyEx(new_image_white, cv2.MORPH_OPEN, self.kernel)
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
        growth = cv2.morphologyEx(growth, cv2.MORPH_OPEN, self.kernel)
        death = cv2.morphologyEx(death, cv2.MORPH_OPEN, self.kernel2)
        blotching = cv2.morphologyEx(blotching, cv2.MORPH_OPEN, self.kernel)
        recovery = cv2.morphologyEx(recovery, cv2.MORPH_OPEN, self.kernel)
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

def align_images(image1, template, maxFeatures=500, keepPercent=0.2,
                 debug=False,detector = changes_detector):
    hsv = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    lower_val = np.array([130, 50, 90])
    upper_val = np.array([170, 255, 255])
    mask = cv2.inRange(hsv, lower_val, upper_val)
    if np.sum(mask) <= 1000:
        return { 'growth_cnts' : [],'death_cnts' : [],'blotching_cnts' : [],'recovery_cnts' : [] }
    # convert both the input image and template to grayscale
    imageGray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # use ORB to detect keypoints and extract (binary) local
    # invariant features
    orb = cv2.ORB_create(maxFeatures)
    (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
    (kpsB, descsB) = orb.detectAndCompute(templateGray, None)

    # match the features
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)

    # sort the matches by their distance (the smaller the distance,
    # the "more similar" the features are)
    matches = sorted(matches, key=lambda x: x.distance)

    # keep only the top matches
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]
    if len(matches) < 50:
        return { 'growth_cnts' : [],'death_cnts' : [],'blotching_cnts' : [],'recovery_cnts' : [] }
    # check to see if we should visualize the matched keypoints
    if debug:
        matchedVis = cv2.drawMatches(image1, kpsA, template, kpsB,
                                     matches, None)
        matchedVis = imutils.resize(matchedVis, width=1000)

    # allocate memory for the keypoints (x,y-coordinates) from the
    # top matches -- we'll use these coordinates to compute our
    # homography matrix
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")

    # loop over the top matches
    for (i, m) in enumerate(matches):
        # indicate that the two keypoints in the respective images
        # map to each other
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    # compute the homography matrix between the two sets of matched
    # points
    if len(ptsA) == 0 or len(ptsB) == 0:
        return { 'growth_cnts' : [],'death_cnts' : [],'blotching_cnts' : [],'recovery_cnts' : [] }
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    # use the homography matrix to align the images
    (h, w) = template.shape[:2]
    aligned = image(cv2.warpPerspective(image1, H, (w, h)))
    old = image(template)
    # return the aligned image
    return detector.detect(old,aligned)

# load the input image and template from disk
template = cv2.imread('old.png')

# align the images

# resize both the aligned and template images so we can easily
# visualize them on our screen

# load the image
old_image = image(template)

# create images detector
imagedetector = changes_detector()

cap = cv2.VideoCapture(0)
while not cap.isOpened():
    cap = cv2.VideoCapture(0)
    cv2.waitKey(1000)
    print ("Wait for the header")

while True:
    flag, frame = cap.read() # get the frame
    if flag:

        contours = align_images(frame.copy(), template, debug=True,detector= imagedetector)
        area = 120
        for c in contours.get('growth_cnts'):
            if cv2.contourArea(c) > area:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                x, y, w, h = cv2.boundingRect(approx)
                # green
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "growth " , (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                            (0, 255, 0), 2)
        for c in contours.get('death_cnts'):
            if cv2.contourArea(c) > area:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                x, y, w, h = cv2.boundingRect(approx)
                # cyan
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, "death ", (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                            (0, 255, 255), 2)
        for c in contours.get('blotching_cnts'):
            if cv2.contourArea(c) > area:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                x, y, w, h = cv2.boundingRect(approx)
                # blue
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "blotching ", (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                            (0, 0, 255), 2)
        for c in contours.get('recovery_cnts'):
            if cv2.contourArea(c) > area:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # red
                cv2.putText(frame, "recovery " , (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                            (255,0 , 0), 2)
    cv2.imshow("result",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()