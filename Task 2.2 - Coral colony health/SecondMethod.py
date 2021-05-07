import cv2
import numpy as np
from math import atan2, degrees
import imutils
import matplotlib.pyplot as plt


LOWER_PURPLE = np.array([130, 50, 90])
UPPER_PURPLE = np.array([170, 255, 255])
LOWER_WHITE = np.array([0, 0, 212])
UPPER_WHITE = np.array([131, 255, 255])
KERNEL = kernal = np.ones((5, 5), "uint8")




def simplest_cb(img, percent=1):
    out_channels = []
    cumstops = (
        img.shape[0] * img.shape[1] * percent / 200.0,
        img.shape[0] * img.shape[1] * (1 - percent / 200.0)
    )
    for channel in cv2.split(img):
        cumhist = np.cumsum(cv2.calcHist([channel], [0], None, [256], (0,256)))
        low_cut, high_cut = np.searchsorted(cumhist, cumstops)
        lut = np.concatenate((
            np.zeros(low_cut),
            np.around(np.linspace(0, 255, high_cut - low_cut + 1)),
            255 * np.ones(255 - high_cut)
        ))
        out_channels.append(cv2.LUT(channel, lut.astype('uint8')))
    return cv2.merge(out_channels)


#detects growth, death ,recovery and blotching by mouse movement
'''def ColorPicker(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = img1[y, x]
        blue1, green1, red1 = img1[y, x]
        blue2, green2, red2 = img2[y, x]
        if img1[y, x].all() == 0 and img2[y, x].all() == 0:
            print("black")
        elif abs(int(blue2)- int(blue1)) <= 30 and abs(int(green2)- int(green1)) <= 30 and abs(int(red2)- int(red1)) <= 30:
            print("no change")
        elif blue1 > 155 and green1 > 155 and red1 > 155 and blue2 < 150 and green2 < 120 and red2 > 105:
            print("recovery")
        elif blue2 > 155 and green2 > 155 and red2 > 155 and blue1 < 150 and green1 < 120 and red1 > 105:
            print("blotching")
        elif img1[y, x].all() == 0 and blue2 > 0 and green2 > 0 and red2 > 0:
            print("growth")
        elif img2[y, x].all() == 0 and blue1 > 0 and green1 > 0 and red1 > 0:
            print("death")

cv2.namedWindow("img2")
cv2.setMouseCallback("img2", ColorPicker)
while (1):
    cv2.imshow('img1', img1)
    cv2.imshow("img2", img2)
    if cv2.waitKey(10) & 0xFF == 27:
        break
'''


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def GetAngleOfLineBetweenTwoPoints(p1, p2):
    xDiff = p2[0] - p1[0]
    yDiff = p2[1] - p1[1]
    return degrees(atan2(yDiff, xDiff))


def adjust_angle(cv_image_camera, radius=9, iter=5, debug=False):
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", help="path to the image file", default="reference2.png")
    # ap.add_argument("-r", "--radius", type=int, default=9,
    #                 help="radius of Gaussian blur; must be odd")
    # args = vars(ap.parse_args())

    image = None
    arr = []


    for i in range(iter):
        if image is None:
            image = cv_image_camera.copy()


        orig = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (radius, radius), 0)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
        image = orig.copy()

        cv2.circle(image, minLoc, radius, (255, 0, 0), 2)
        cv2.circle(image, minLoc, radius, (255, 255, 255), cv2.FILLED)
        arr.append(minLoc)
        if debug:
            print(minLoc)
    if debug:
        cv2.imshow("cen", image)
        cv2.waitKey(0)

    avgarr = []
    min_point = arr[0]

    for i in arr:
        for j in arr:
            x1 = i[0]
            x2 = j[0]
            y1 = i[1]
            y2 = j[1]
            if i == j or x1 > x2 or y1 < y2:
                continue

            ang = -GetAngleOfLineBetweenTwoPoints(i, j)
            if debug:
                print("points", x1, x2, y1, y2)
                print("check", x2 - x1, y1 - y2)
                print(ang)
            if min_point[0] < x1:
                min_point = i

            avgarr.append(ang)

    avgang = 0

    for i in avgarr:
        if i > 45:
            continue
        avgang += i

    avgang /= avgarr.__len__()
    if debug:
        print("avgang =", avgang)

    final = rotate(cv_image_camera, -avgang, min_point)

    if debug:
        cv2.imshow("final", final)
        cv2.waitKey(0)

    return final


def adjust_image(img, show_final=False, debug=False):
    x = simplest_cb(img, 1)
    final = adjust_angle(x, debug=debug)
    if show_final:
        cv2.imshow("final", final)
        cv2.waitKey(0)
    return final


def ColorPicker(img1, img2, NoOfPixels):
    img1 = simplest_cb(img1, percent=1)
    img2 = simplest_cb(img2, percent=1)
    height, width, channels = img1.shape

    hsvimg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsvimg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)


    white_mask1 = cv2.inRange(hsvimg1, LOWER_WHITE, UPPER_WHITE)
    purple_mask1 = cv2.inRange(hsvimg1, LOWER_PURPLE, UPPER_PURPLE)
    white_mask2 = cv2.inRange(hsvimg2, LOWER_WHITE, UPPER_WHITE)
    purple_mask2 = cv2.inRange(hsvimg2, LOWER_PURPLE, UPPER_PURPLE)
    white_mask1 = cv2.dilate(white_mask1, KERNEL)
    white_mask2 = cv2.dilate(white_mask2, KERNEL)


    white1 = cv2.bitwise_and(img1, img1, mask=white_mask1)
    purple1 = cv2.bitwise_and(img1, img1, mask=purple_mask1)
    white2 = cv2.bitwise_and(img2, img2, mask=white_mask2)
    purple2 = cv2.bitwise_and(img2, img2, mask=purple_mask2)

    img1 = cv2.add(white1, purple1)
    img2 = cv2.add(white2, purple2)

    recovery = []
    blotching = []
    growth = []
    death = []

    for y in range(0, height+1, NoOfPixels):
        for x in range(0, width+1, NoOfPixels):
            blue1, green1, red1 = img1[y, x]
            blue2, green2, red2 = img2[y, x]
            if img1[y, x].all() == 0 and img2[y, x].all() == 0:
                continue
            elif abs(int(blue2)- int(blue1)) <= 30 and abs(int(green2)- int(green1)) <= 30 and abs(int(red2)- int(red1)) <= 30:
                continue
            elif blue1 > 155 and green1 > 155 and red1 > 155 and blue2 < 150 and green2 < 120 and red2 > 105:
                recovery.append(f"[{y},{x}]")
            elif blue2 > 155 and green2 > 155 and red2 > 155 and blue1 < 150 and green1 < 120 and red1 > 105:
                blotching.append(f"[{y},{x}]")
            elif img1[y, x].all() == 0 and blue2 > 0 and green2 > 0 and red2 > 0:
                growth.append(f"[{y},{x}]")
            elif img2[y, x].all() == 0 and blue1 > 0 and green1 > 0 and red1 > 0:
                death.append(f"[{y},{x}]")

    return(f"death: {death}\ngrowth: {growth}\nblotching: {blotching}\nrecovery: {recovery}")

if __name__ == '__main__':
    import os
    for i in os.listdir('.'):
        if i.__contains__('.png'):
            img = adjust_image(cv2.imread(i), True, debug=False)
            cv2.imwrite(i, img)
            # cv2.imshow(img)
            # cv2.waitKey(0)

