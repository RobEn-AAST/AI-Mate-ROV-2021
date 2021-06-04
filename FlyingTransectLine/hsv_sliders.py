
import cv2 as cv
import numpy as np

def set_slider(vals):
    cv.setTrackbarPos("Hue min", "TrackBars", vals[0][0])
    cv.setTrackbarPos("Hue max", "TrackBars", vals[1][0])
    cv.setTrackbarPos("Sat min", "TrackBars", vals[0][1])
    cv.setTrackbarPos("Sat max", "TrackBars", vals[1][1])
    cv.setTrackbarPos("Val min", "TrackBars", vals[0][2])
    cv.setTrackbarPos("Val max", "TrackBars", vals[1][2])

def track_bars():
    cv.namedWindow("TrackBars")
    cv.resizeWindow("TrackBars", 640, 240)
    """cv.createTrackbar("Hue min",
                          "TrackBars", 0, 179,
                          color_detect(img_hsv,
                                       cv.getTrackbarPos("Hue min", "TrackBars"),
                                       cv.getTrackbarPos("Hue max", "TrackBars")))"""
    cv.createTrackbar("Hue min", "TrackBars", 0, 179, lambda a: a+0)
    cv.createTrackbar("Hue max", "TrackBars", 0, 179, lambda a: a+0)
    cv.createTrackbar("Sat min", "TrackBars", 0, 255, lambda a: a+0)
    cv.createTrackbar("Sat max", "TrackBars", 0, 255, lambda a: a+0)
    cv.createTrackbar("Val min", "TrackBars", 0, 255, lambda a: a+0)
    cv.createTrackbar("Val max", "TrackBars", 0, 255, lambda a: a+0)


def color_detect_trackbars(img_orig, vals):
    track_bars()
    img_orig = cv.resize(img_orig, (500, 390))
    img_hsv = cv.cvtColor(img_orig, cv.COLOR_BGR2HSV)
    flag_called_once = True
    while True:
        cv.imshow("original hsv image", img_hsv)
        h_min = cv.getTrackbarPos("Hue min", "TrackBars")
        h_max = cv.getTrackbarPos("Hue max", "TrackBars")
        s_min = cv.getTrackbarPos("Sat min", "TrackBars")
        s_max = cv.getTrackbarPos("Sat max", "TrackBars")
        v_min = cv.getTrackbarPos("Val min", "TrackBars")
        v_max = cv.getTrackbarPos("Val max", "TrackBars")
        lower = np.array([h_min, s_min, v_min])
        higher = np.array([h_max, s_max, v_max])
        mask = cv.inRange(img_hsv, lower, higher)
        
        if flag_called_once:
            set_slider(vals)
            flag_called_once = False
        
        cv.imshow("masked hsv image", mask)
        img_res = cv.bitwise_and(img_orig, img_orig, mask=mask)
        cv.imshow("masked orig image", img_res)
        if cv.waitKey(1) & 0xFF == ord('x'):
            cv.destroyAllWindows()
            return lower, higher

