# USAGE
# python align_document.py --template form_w4.png --image scans/scan_01.jpg
# note : not disabling image shows unless we finish project
# import the necessary packages
import numpy as np
import imutils
import cv2


def align_images(image, template, maxFeatures=500, keepPercent=0.2,
                 debug=False):
    # convert both the input image and template to grayscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

    # check to see if we should visualize the matched keypoints
    if debug:
        matchedVis = cv2.drawMatches(image, kpsA, template, kpsB,
                                     matches, None)
        matchedVis = imutils.resize(matchedVis, width=1000)
        cv2.imshow("Matched Keypoints", matchedVis)
        cv2.waitKey(0)

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
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)

    # use the homography matrix to align the images
    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))

    # return the aligned image
    return aligned


# construct the argument parser and parse the arguments

# load the input image and template from disk
print("[INFO] loading images...")
image = cv2.imread('new.png')
template = cv2.imread('temp.png')

# align the images
print("[INFO] aligning images...")
aligned = align_images(image, template, debug=True)

# resize both the aligned and template images so we can easily
# visualize them on our screen
aligned = imutils.resize(aligned, width=700)
template = imutils.resize(template, width=700)

# our first output visualization of the image alignment will be a
# side-by-side comparison of the output aligned image and the
# template
stacked = np.hstack([aligned, template])

# our second image alignment visualization will be *overlaying* the
# aligned image on the template, that way we can obtain an idea of
# how good our image alignment is
overlay = template.copy()
output = aligned.copy()
cv2.imwrite('over.png', overlay)
cv2.imwrite('out.png', output)
cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
# show the two output image alignment visualizations
cv2.imshow("Image Alignment Stacked", stacked)
cv2.imshow("Image Alignment Overlay", output)
cv2.waitKey(0)


# define images class
class image:
    def __init__(self, img):
        self.original = cv2.resize(img, (500, 500))
        # self.original = cv2.resize(img, img.shape[:2])
        self.hsv = cv2.cvtColor(self.original, cv2.COLOR_BGR2HSV)


class changes_detector:
    def __init__(self):
        self.lower_purple = np.array([130, 50, 90])
        self.upper_purple = np.array([170, 255, 255])
        self.lower_white = np.array([0, 3, 220])
        self.upper_white = np.array([255, 255, 255])
        self.kernel = np.ones((2, 2), np.uint8)

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
        death = cv2.morphologyEx(death, cv2.MORPH_OPEN, self.kernel)
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
        area = 120
        for c in growth_cnts:
            if cv2.contourArea(c) > area:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                x, y, w, h = cv2.boundingRect(approx)
                # green
                cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(preview, "growth " , (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                            (0, 255, 0), 2)
                # cv2.drawContours(preview, c, -1, (0, 255, 0), 1)
        for c in death_cnts:
            if cv2.contourArea(c) > area:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                x, y, w, h = cv2.boundingRect(approx)
                # cyan
                cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(preview, "death ", (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                            (0, 255, 255), 2)
                # cv2.drawContours(preview, c, -1, (0, 255, 255), 1)
        for c in blotching_cnts:
            if cv2.contourArea(c) > area:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                x, y, w, h = cv2.boundingRect(approx)
                # blue
                cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(preview, "blotching ", (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                            (0, 0, 255), 2)
                # cv2.drawContours(preview, c, -1, (0, 0, 255), 1)
        for c in recovery_cnts:
            if cv2.contourArea(c) > area:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(preview, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # red
                cv2.putText(preview, "recovery " , (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                            (255,0 , 0), 2)
                # cv2.drawContours(preview, c, -1, (255, 0, 0), 1)
        # cv2.drawContours(preview, growth_cnts, -1, (0, 255, 0), 1)
        # cv2.drawContours(preview, death_cnts, -1, (0, 255, 255), 1)
        # cv2.drawContours(preview, blotching_cnts, -1, (0, 0, 255), 1)
        # cv2.drawContours(preview, recovery_cnts, -1, (255, 0, 0), 1)

        return np.hstack([old_image.original, new_image.original, preview])


# load the image
old_image = image(template)
cv2.imshow('temp', template)
cv2.imshow('aligned', aligned)
cv2.waitKey(0)
new_image = image(aligned)
# create images detector
imagedetector = changes_detector()
# draw output
cv2.imshow("input_images", imagedetector.detect(old_image, new_image))
cv2.waitKey(0)
