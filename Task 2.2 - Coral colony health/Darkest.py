import cv2
import numpy as np
from math import atan2, degrees

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

if __name__ == '__main__':
    adjust_image(cv2.imread("reference2.png"), True, False)