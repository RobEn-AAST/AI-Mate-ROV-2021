import cv2
import numpy as np

# type values which you get it suitable from trackbar
a = 0
b = 15
c = 45
d = 0
e = 255
f = 255

cap = cv2.VideoCapture(0)


def empty():
    pass


cv2.namedWindow('image')

cv2.createTrackbar("LH", "image", a, 255, empty)
cv2.createTrackbar("LS", "image", b, 255, empty)
cv2.createTrackbar("LV", "image", c, 255, empty)
cv2.createTrackbar("HH", "image", d, 255, empty)
cv2.createTrackbar("HS", "image", e, 255, empty)
cv2.createTrackbar("HV", "image", f, 255, empty)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    a = cv2.getTrackbarPos('LH', 'image')
    d = cv2.getTrackbarPos('HH', 'image')
    b = cv2.getTrackbarPos('LS', 'image')
    e = cv2.getTrackbarPos('HS', 'image')
    c = cv2.getTrackbarPos('LV', 'image')
    f = cv2.getTrackbarPos('HV', 'image')

    lred = np.array([a, b, c])
    ured = np.array([d, e, f])

    mask = cv2.inRange(hsv, lred, ured)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
