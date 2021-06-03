import cv2
import numpy as np
import imutils


def re_find_width(image):
    Wmin, Wmax = 0,0
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if (image[y,x]!=0):
                Wmin = x
                break

    for y in range(image.shape[0], 0):
        for x in range(image.shape[1], 0):
            if (image[y,x]!=0):
                Wmax = x
                break

    return Wmin, Wmax

def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated



def find_width(Image, radius=21, iter=20, drift=20, debug=False):
    image = None
    arr = []
    avg_val = None
    

    for i in range(iter):
        if image is None:
            image = Image.copy()


        orig = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (radius, radius), 0)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
        image = orig.copy()

        cv2.circle(image, minLoc, radius, (255, 0, 0), 2)
        cv2.circle(image, minLoc, radius, (255, 255, 255), cv2.FILLED)

        if avg_val == None:
            avg_val = minVal
        elif minVal - avg_val > drift:
            break
        
        arr.append(minLoc)


        if debug:
            print(minLoc)

    if debug:
        cv2.imshow("buttom black bar spots", image)
        cv2.waitKey(0)

    xmin,ymin = arr[0]
    xmax,ymax = arr[0]

    # ymin, xmin = arr[0]
    # ymax, xmax = arr[0]


    for x,y in arr:
        if x < xmin:
            xmin = x
            ymin = y
        elif x > xmax:
            xmax = x
            ymax = y
        
    #width = xmax - xmin
    #height = ymax - ymin
    return xmin, ymin, xmax, ymax

image = cv2.imread('gui/A.S./t.jpeg')
xmin, ymin, xmax, ymax = find_width(image, debug=True)
print(xmin, xmax, ymin, ymax)
print(image.shape)
coral_image = image[:, xmin:xmax]
print(coral_image.shape)
cv2.imshow('coral crop', coral_image)
cv2.waitKey(0)

img=image
img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lower mask (0-10)
lower_red = np.array([0,50,50])
upper_red = np.array([10,255,255])
mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

# upper mask (170-180)
lower_red = np.array([170,50,50])
upper_red = np.array([180,255,255])
mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

# join my masks
mask = mask0+mask1

# set my output img to zero everywhere except my mask
output_img = img.copy()
output_img[np.where(mask==0)] = 0

# or your HSV image, which I *believe* is what you want
output_hsv = img_hsv.copy()
output_hsv[np.where(mask==0)] = 0

cv2.imshow('x', output_img)
cv2.imshow('xa', output_hsv)
cv2.imshow('xa', mask)

cv2.waitKey(0)
# mask_masked = 
print(re_find_width(mask))

