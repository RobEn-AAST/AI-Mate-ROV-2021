import numpy as np
import cv2 as cv


class PipeRange:
    def __init__(self, min_x, max_x, y_co=0.75):
        self.MinX = min_x
        self.MaxX = max_x
        self.Yco = y_co


L_RANGE = PipeRange(0.08, 0.3)  # permitted range for the left blue pipe within X-axis
R_RANGE = PipeRange(0.7, 0.92)  # permitted range for the right blue pipe within X-axis
BLUE_PIPE_COLOR_RANGE = [[99, 173, 80], [112, 255, 174]]  # the HSV color range for the blue pipes
PIPES_DISTANCE = [0.78, 0.62]  # permitted distance between both blue pipes [0]:max-distance, [1]:min-distance


def color_detection(img_orig, values_min, values_max, show_detection=True):
    """
    this function detects the pipes by given colors values.

    :return: a mask of the pipes.
    """
    img_hsv = cv.cvtColor(img_orig, cv.COLOR_BGR2HSV)
    mask = cv.inRange(img_hsv, np.array(values_min), np.array(values_max))
    if show_detection:
        img_res = cv.bitwise_and(img_orig, img_orig, mask=mask)
        cv.imshow("Masked hsv image", mask)
        cv.imshow("Masked orig image", img_res)
    return mask


def get_contours(masked_img, frame, draw_contours=False):
    """
    this function find the contours for both blue pipes and determine
    the coordinates for both.

    :return: the coordinates of left and right blue pipes in two 2D arrays.
    """
    contours, _ = cv.findContours(masked_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    mask_shape = masked_img.shape
    left_coordinate, right_coordinate = \
        np.array([[0, mask_shape[0]+10], [0, 0]]), np.array([[0,  mask_shape[0]+10], [0, 0]])

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 100:
            peri = cv.arcLength(cnt, False)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, False)
            x, y, w, h = cv.boundingRect(approx)
            if y < left_coordinate[0][1] or y < right_coordinate[0][1]:
                if draw_contours:
                    cv.drawContours(frame, cnt, -1, (255, 0, 255), 3)
                if x < mask_shape[1]//2:
                    left_coordinate[0] = x, y
                else:
                    right_coordinate[0] = x, y
            if y+h > left_coordinate[1][1] or y+h > right_coordinate[1][1]:
                if draw_contours:
                    cv.drawContours(frame, cnt, -1, (255, 0, 255), 3)
                if x < mask_shape[1] // 2:
                    left_coordinate[1] = x + w//2, y+h
                else:
                    right_coordinate[1] = x + w//2, y+h
    return left_coordinate, right_coordinate


def send_commands(frame, midpoint_right, midpoint_left):
    """
    this function prints/sends commands through serial port.
    """
    # distance between the blue pipes
    dist = midpoint_right[0] - midpoint_left[0]

    # sending data according to the position of the blue pipes
    if midpoint_left[0] <= int(frame.shape[1] * L_RANGE.MinX):
        print('go left')
    elif midpoint_left[0] >= int(frame.shape[1] * L_RANGE.MaxX):
        print('go right')
    elif midpoint_right[0] <= int(frame.shape[1] * R_RANGE.MinX):
        print('go left')
    elif midpoint_right[0] >= int(frame.shape[1] * R_RANGE.MaxX):
        print('go right')

    # sending data according to the distance between the pipes
    if dist >= int(frame.shape[1]*PIPES_DISTANCE[0]):
        print('go up')
    elif dist <= int(frame.shape[1]*PIPES_DISTANCE[1]):
        print('go down')


def highlight_pipes(frame, left_co, right_co):
    """
    this function is responsible for drawing on the screen each frame for visualization purposes
    and it calculates the mid-point of the two blue pipes.

    :return: the two mid-point of both blue pipes.
    """
    left_pipe_up, left_pipe_low = left_co
    right_pipe_up, right_pipe_low = right_co

    # drawing two horizontal lines detecting visualizing the permitted x-range for the lines
    cv.line(frame, (int(frame.shape[1] * L_RANGE.MinX), int(frame.shape[0] * L_RANGE.Yco)),
            (int(frame.shape[1] * L_RANGE.MaxX), int(frame.shape[0] * L_RANGE.Yco)), (0, 204, 0), 5)
    cv.line(frame, (int(frame.shape[1] * R_RANGE.MinX), int(frame.shape[0] * R_RANGE.Yco)),
            (int(frame.shape[1] * R_RANGE.MaxX), int(frame.shape[0] * R_RANGE.Yco)), (0, 204, 0), 5)

    # drawing bluish vertical lines over the blue pipes
    cv.line(frame, (left_co[0][0], left_co[0][1]), (left_co[1][0], left_co[1][1]), (153, 153, 0), 3)
    cv.line(frame, (right_co[0][0], right_co[0][1]), (right_co[1][0], right_co[1][1]), (153, 153, 0), 3)

    # calculating the two center points of the pipes and drawing two violet circles at them
    midpoint_right = [(right_pipe_up[0] + right_pipe_low[0]) // 2, (right_pipe_up[1] + right_pipe_low[1]) // 2]
    midpoint_left = [(left_pipe_up[0] + left_pipe_low[0]) // 2, (left_pipe_up[1] + left_pipe_low[1]) // 2]
    cv.circle(frame, (midpoint_right[0], midpoint_right[1]), 10, (204, 0, 204), -1)
    cv.circle(frame, (midpoint_left[0], midpoint_left[1]), 10, (204, 0, 204), -1)

    # return the two mid-points of the two blue pipes
    return midpoint_right, midpoint_left


def detect_pipes(frame, mask_blue):
    """
    this function calls get_contours() function, receives both blue pipes coordinates from it,
    calls highlight_pipes() function to draw the guides on each frame and receives the mid-points
    from it.

    :return: mid-points coordinates of both blue pipes from the return value of highlight_pipes() function.
    """
    # get_contours() function returns two 2d arrays carrying the coordinates of both blue pipes
    left_co, right_co = get_contours(mask_blue, frame, draw_contours=False)

    # returns mid-points coordinates from the return value
    return highlight_pipes(frame, left_co, right_co)


def read_video():
    """
    this function is responsible for reading the video/camera frame by frame
    and call multiple functions. May consider it as the main function
    """
    vid = cv.VideoCapture("vid.mp4")
    while True:
        readable, frame = vid.read()
        if not readable:
            break
        cv.imshow("Camera Frames", frame)

        # this function detects the pipes by there color and creates and mask for the detection
        mask_blue = \
            color_detection(frame, BLUE_PIPE_COLOR_RANGE[0], BLUE_PIPE_COLOR_RANGE[1], show_detection=False)

        # this function detects the blue pipes and draw guides for visualization
        midpoint_right, midpoint_left = detect_pipes(frame, mask_blue)

        # this function is fully responsible for printing/sending commands through serial port
        send_commands(frame, midpoint_right, midpoint_left)

        cv.imshow("Modified frames", frame)

        if cv.waitKey(30) & 0xFF == ord('q'):
            break
    vid.release()


read_video()  # this function is responsible for reading video/camera frames and call every other function
cv.destroyAllWindows()
