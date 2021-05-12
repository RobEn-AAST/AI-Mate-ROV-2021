from cv2 import cv2
from math import atan2, degrees
import numpy as np


def simplest_cb(img, percent=1):
    out_channels = []
    cumstops = (
        img.shape[0] * img.shape[1] * percent / 200.0,
        img.shape[0] * img.shape[1] * (1 - percent / 200.0)
    )
    for channel in cv2.split(img):
        cumhist = np.cumsum(cv2.calcHist([channel], [0], None, [256], (0, 256)))
        low_cut, high_cut = np.searchsorted(cumhist, cumstops)
        lut = np.concatenate((
            np.zeros(low_cut),
            np.around(np.linspace(0, 255, high_cut - low_cut + 1)),
            255 * np.ones(255 - high_cut)
        ))
        out_channels.append(cv2.LUT(channel, lut.astype('uint8')))
    return cv2.merge(out_channels)

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


def adjust_angle(frame, radius=9, iter=5, safety_angle=45, debug=False, show_images=False, minVal_max_limit=30,
                 Sure_Image=False):
    arr = []
    image = frame.copy()
    minVal_fail_counter = 0
    for i in range(iter):  # iterations is important -- don't touch
        playground_image = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (radius, radius), 0)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
        # print("minVal =", minVal)
        if minVal > minVal_max_limit:
            minVal_fail_counter += 1
            continue
        image = playground_image.copy()
        cv2.circle(image, minLoc, radius, (255, 0, 0), 2)
        cv2.circle(image, minLoc, radius, (255, 255, 255), cv2.FILLED)
        arr.append(minLoc)
        if debug:
            print(minLoc)
    if show_images:
        cv2.imshow("cen", image)
        cv2.waitKey(0)

    # limit using minval
    if minVal_fail_counter >= iter - 1:
        if Sure_Image:
            print(" ** NOTE ** minVal failure, trying to solve issue")
            return adjust_angle(frame, radius, iter, debug=True, safety_angle=safety_angle, Sure_Image=Sure_Image,
                                minVal_max_limit=minVal_max_limit + 7)
        return frame

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

    average_angle = 0
    average_angle_counter = 0

    if avgarr.__len__() == 0:
        if Sure_Image:  # limit when no spots
            print(" ** NOTE ** no spots failure, trying to solve issue")
            return adjust_angle(frame, radius, iter + 1, debug=True, safety_angle=safety_angle,
                                Sure_Image=Sure_Image,
                                minVal_max_limit=minVal_max_limit)
        return frame

    for i in avgarr:
        if i > safety_angle:
            continue
        average_angle += i
        average_angle_counter += 1

    if average_angle_counter < 2:
        if Sure_Image:  # limit using safety_angle
            print(" ** NOTE ** safety angle exceeded, trying to solve issue")
            return adjust_angle(frame, radius, iter, debug=True, safety_angle=safety_angle + 10,
                                Sure_Image=Sure_Image,
                                minVal_max_limit=minVal_max_limit)
        return frame

    average_angle /= average_angle_counter
    if debug:
        print("average_angle =", average_angle)

    final = rotate(frame, -average_angle, min_point)

    if show_images:
        cv2.imshow("final", final)
        cv2.waitKey(0)

    return final


def adjust_image(img, debug=False, safety_angle=40, show_images=False, show_final=False, Sure_Image=False,
                 minVal_max_limit=30):
    final = adjust_angle(simplest_cb(img, 1), debug=debug, safety_angle=safety_angle, show_images=show_images,
                         Sure_Image=Sure_Image, minVal_max_limit=minVal_max_limit)
    if show_final:
        cv2.imshow("final", final)
        cv2.waitKey(0)
    return final