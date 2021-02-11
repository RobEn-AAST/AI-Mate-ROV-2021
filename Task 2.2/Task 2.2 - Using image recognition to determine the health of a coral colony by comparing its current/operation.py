# import the necessary packages
import numpy as np
import argparse
import cv2

#define images class
class image:
    def __init__(self,directory):
        self.original = cv2.resize(cv2.imread(directory),(250,250))
        self.hsv = cv2.cvtColor(self.original, cv2.COLOR_BGR2HSV)

class changes_detector:
    def __init__(self):
        self.lower_purple = np.array([130,50,90])
        self.upper_purple = np.array([170,255,255])
        self.lower_white = np.array([0,3,220])
        self.upper_white = np.array([255,255,255])
        self.kernel = np.ones((2,2),np.uint8)

    def detect(self,old_image,new_image):
        preview = new_image.original.copy()
        #create masks
        mask1 = cv2.inRange(old_image.hsv, self.lower_purple, self.upper_purple)
        mask2 = cv2.inRange(new_image.hsv, self.lower_purple, self.upper_purple)
        mask3 = cv2.inRange(old_image.hsv, self.lower_white, self.upper_white)
        mask4 = cv2.inRange(new_image.hsv, self.lower_white, self.upper_white)
        #extract purple parts and white parts from 2 images
        old_image_purple = cv2.bitwise_and(old_image.hsv,old_image.hsv,mask=mask1)
        old_image_white = cv2.bitwise_and(old_image.hsv,old_image.hsv,mask=mask3)
        new_image_purple = cv2.bitwise_and(new_image.hsv,new_image.hsv,mask=mask2)
        new_image_white = cv2.bitwise_and(new_image.hsv,new_image.hsv,mask=mask4)
        old_image_purple[mask1>0] = (255,255,255)
        new_image_purple[mask2>0] = (255,255,255)
        old_image_white[mask3>0] = (255,255,255)
        new_image_white[mask4>0] = (255,255,255)
        #apply opening morphology on results  to reduce noise
        old_image_white = cv2.morphologyEx(old_image_white, cv2.MORPH_OPEN, self.kernel)
        new_image_white = cv2.morphologyEx(new_image_white, cv2.MORPH_OPEN, self.kernel)
        #find the common purple part between the 2 images using bitwise xor
        common_purple = cv2.bitwise_xor(old_image_purple,new_image_purple)
        # growth = new image && common part
        growth = cv2.bitwise_and(new_image_purple,common_purple)
        #death  = old image && common 
        death = cv2.bitwise_and(old_image_purple,common_purple)
        #bloching = white new image && purple old image
        blotching = cv2.bitwise_and(new_image_white,old_image_purple)
        #recovery = purple new image && white old image
        recovery = cv2.bitwise_and(new_image_purple,old_image_white)
        #apply opening morphology on results  to reduce noise
        growth = cv2.morphologyEx(growth, cv2.MORPH_OPEN, self.kernel)
        death = cv2.morphologyEx(death, cv2.MORPH_OPEN, self.kernel)
        blotching = cv2.morphologyEx(blotching, cv2.MORPH_OPEN, self.kernel)
        recovery = cv2.morphologyEx(recovery, cv2.MORPH_OPEN, self.kernel)
        #transform the growth and death images to the grey scale domain to get contours
        growth_grey_scale = cv2.cvtColor(growth, cv2.COLOR_BGR2GRAY)
        death_grey_scale = cv2.cvtColor(death, cv2.COLOR_BGR2GRAY)
        blotching_grey_scale = cv2.cvtColor(blotching, cv2.COLOR_BGR2GRAY)
        recovery_grey_scale = cv2.cvtColor(recovery, cv2.COLOR_BGR2GRAY)
        growth_thresh = cv2.threshold(growth_grey_scale, 127, 255, 0)[1]
        death_thresh = cv2.threshold(death_grey_scale, 127, 255, 0)[1]
        blotching_thresh = cv2.threshold(blotching_grey_scale, 127, 255, 0)[1]
        recovery_thresh = cv2.threshold(recovery_grey_scale, 127, 255, 0)[1]
        #find the contours
        growth_cnts = cv2.findContours(growth_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        death_cnts = cv2.findContours(death_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        blotching_cnts = cv2.findContours(blotching_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        recovery_cnts = cv2.findContours(recovery_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        #draw the contours on the preview image
        cv2.drawContours(preview, growth_cnts, -1, (0,255,0), 1)
        cv2.drawContours(preview, death_cnts, -1, (0,255,255), 1)
        cv2.drawContours(preview, blotching_cnts, -1, (0,0,255), 1)
        cv2.drawContours(preview, recovery_cnts, -1, (255,0,0), 1)
        return np.hstack([old_image.original,new_image.original,preview])

# load the image
ap = argparse.ArgumentParser(description="Task 2.2 - Using image recognition to determine the health of a coral colony by comparing its current")
ap.add_argument("-o","--oldimage",type=str,metavar='', help = "path to the old image",required=True)
ap.add_argument("-n","--newimage",type=str,metavar='', help = "path to the new image",required=True)
args = ap.parse_args()
#create 2 images
old_image = image(args.oldimage)
new_image = image(args.newimage)
#create images detector
imagedetector = changes_detector()
#draw output
cv2.imshow("input_images", imagedetector.detect(old_image,new_image))
cv2.waitKey(0)