import numpy as np
from cv2 import cv2
from sklearn.cluster import KMeans
from object_detection_module import remove_back_ground

def pixelate_internal(img, w, h):
    w = int(w)
    h = int(h)
    height, width = img.shape[:2]
    temp = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)


def colorClustering(idx, img, k):
    clusterValues = []
    for _ in range(0, k):
        clusterValues.append([])
    
    for r in range(0, idx.shape[0]):
        for c in range(0, idx.shape[1]):
            clusterValues[idx[r][c]].append(img[r][c])

    imgC = np.copy(img)

    clusterAverages = []
    for i in range(0, k):
        clusterAverages.append(np.average(clusterValues[i], axis=0))
    
    for r in range(0, idx.shape[0]):
        for c in range(0, idx.shape[1]):
            imgC[r][c] = clusterAverages[idx[r][c]]
            
    return imgC

def segmentImgClrRGB(img, k):
    imgC = np.copy(img)
    h = img.shape[0]
    w = img.shape[1]
    imgC.shape = (img.shape[0] * img.shape[1], 3)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(imgC).labels_
    kmeans.shape = (h, w)
    return kmeans


def kMeansImage(image, k):
    idx = segmentImgClrRGB(image, k)
    return colorClustering(idx, image, k)


def PixelArt(image, radius=90):
    img32 = pixelate_internal(image, radius, radius)
    return img32


def PixelArt_with_Kmeans(image, radius=90, K_factor=3):
    img32 = pixelate_internal(image, radius, radius)
    KMI = kMeansImage(img32, K_factor)
    return KMI


def Pixelated_Clean_Image_beta(image, kernel=np.ones((5,5),np.uint8), noiseKernel=np.ones((2,2),np.uint8)):
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, noiseKernel)
    dilated = cv2.dilate(opened,kernel,iterations = 1)
    return dilated


def addimages(background,overlay):
    overlay_tmp = overlay.copy()
    background_tmp = background.copy()
    mask = cv2.inRange(overlay_tmp, np.array([255,255,255]), np.array([255,255,255]))
    background_tmp[mask > 0] = (255, 255, 255)
    return background_tmp
img = cv2.resize(cv2.imread("old.png"),(500,500))



cap = cv2.VideoCapture(0)

while True:
    # Capture webcame frame
    frame = cv2.resize(cv2.flip(cap.read()[1], 1),(500,500))
    frame = addimages(frame,remove_back_ground(Pixelated_Clean_Image_beta(img)))
    cv2.imshow("result",frame)
    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()