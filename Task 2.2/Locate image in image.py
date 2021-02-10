import cv2 as cv
import numpy as np
import cv2
import imutils
from skimage.metrics import structural_similarity

SmallImageDir = "G:/tmp/project ai/2.2/cred.png"
BigImageDir = "G:/tmp/project ai/2.2/3.png"

SmallImageColored = cv.imread(SmallImageDir)
BigImageColored = cv.imread(BigImageDir)

SmallImage = cv.imread(SmallImageDir, cv.IMREAD_GRAYSCALE)
BigImage = cv.imread(BigImageDir, cv.IMREAD_GRAYSCALE)
if SmallImage is None or BigImage is None:
    print('Could not open or find the images!')
    exit(0)
# -- Step 1: Detect the keypoints using SURF Detector, compute the descriptors

detector = cv.xfeatures2d.SIFT_create()
keypoints1, descriptors1 = detector.detectAndCompute(SmallImage, None)
keypoints2, descriptors2 = detector.detectAndCompute(BigImage, None)
# -- Step 2: Matching descriptor vectors with a FLANN based matcher
# Since SURF is a floating-point descriptor NORM_L2 is used
matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE)
knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)
# -- Filter matches using the Lowe's ratio test
ratio_thresh = 0.8
good_matches = []
for m, n in knn_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)
# -- Draw matches
img_matches = np.empty((max(SmallImage.shape[0], BigImage.shape[0]), SmallImage.shape[1] + BigImage.shape[1], 3),
                       dtype=np.uint8)
cv.drawMatches(SmallImage, keypoints1, BigImage, keypoints2, good_matches, img_matches,
               flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# -- Show detected matches
cv.imshow('Matches', img_matches)
# cv.waitKey()

list_kp1 = [keypoints1[mat.queryIdx].pt for mat in good_matches]
list_kp2 = [keypoints2[mat.trainIdx].pt for mat in good_matches]
maxX = maxY = 0
(minX, minY) = list_kp2[0]
for (i, j) in list_kp2:
    if (i > maxX):
        maxX = i
    if (i < minX):
        minX = i
    if (j > maxY):
        maxY = j
    if (j < minY):
        minY = j

minX = int(minX)
minY = int(minY)
maxX = int(maxX)
maxY = int(maxY)
print(minX, minY, maxX, maxY)
factorx = BigImage.shape[1] - maxX
factory = BigImage.shape[0] - maxY
if minY - factory < 0:
    factory = 0
if minX - factorx < 0:
    factorx = 0


maxX1 = maxY1 = 0
(minX1, minY1) = list_kp2[0]
for (i, j) in list_kp1:
    if (i > maxX1):
        maxX1 = i
    if (i < minX1):
        minX1 = i
    if (j > maxY1):
        maxY1 = j
    if (j < minY1):
        minY1 = j

minX1 = int(minX1)
minY1 = int(minY1)
maxX1 = int(maxX1)
maxY1 = int(maxY1)
print(minX1, minY1, maxX1, maxY1)
factorx = SmallImage.shape[1] - maxX1
factory = SmallImage.shape[0] - maxY1
if minY1 - factory < 0:
    factory = 0
if minX1 - factorx < 0:
    factorx = 0

crop_img2 = BigImageColored[minY - factory:maxY + factory, minX - factorx:maxX + factorx]
cv.imshow("cropped from the Bigger Image", crop_img2)
crop_img = SmallImageColored[minY1 - factory:maxY1 + factory, minX1 - factorx:maxX1 + factorx]
crop_img = cv.resize(crop_img, (crop_img2.shape[1], crop_img2.shape[0]))
cv.imshow("cropped from the Smaller Image", crop_img)
cv.imshow("Original Bigger Image", BigImageColored)
cv.imshow("Original Smaller Image", SmallImageColored)
cv.waitKey(0)
cv.imwrite("G:/tmp/project ai/2.2/cc1.png", crop_img)
cv.imwrite("G:/tmp/project ai/2.2/cc2.png", crop_img2)
