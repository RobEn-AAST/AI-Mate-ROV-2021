import cv2

a = 2000  # to control how many points of matching

img1 = cv2.imread('gray.png', 0)
img2 = cv2.imread('imgg2.png', 0)

orb = cv2.ORB_create(nfeatures=a)

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])
        print(len(good))
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

cv2. imwrite('C:/Users/moham/PycharmProjects/trial 2/matching_pic.png', img3)
cv2.waitKey(0)
