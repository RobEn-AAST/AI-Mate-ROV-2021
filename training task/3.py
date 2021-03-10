import cv2
import numpy as np

b = 60  # initial value of threshold1
c = 30  # initial value of threshold2

cap = cv2.VideoCapture(0)


def empty():
    pass


cv2.namedWindow("Parameters")
cv2.createTrackbar("Threshold1", "Parameters", b, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", c, 255, empty)

while True:
    success, img = cap.read()
    imgContour = img.copy()
    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

    cv2.imshow("Result", imgCanny)

    cv2.imshow("IMG", img)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
