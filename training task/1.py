import cv2

image = cv2.imread('img1.PNG')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('C:/Users/moham/PycharmProjects/trial 2/gray.png', gray)
redsq_gray = cv2.imread('gray.png')

height, width, channel = redsq_gray.shape
upper_left = (width // 4, height // 4)
bottom_right = (width * 3 // 4, height * 3 // 4)
cv2.rectangle(redsq_gray, upper_left, bottom_right, (0, 0, 255), -1)

cv2.imwrite('C:/Users/moham/PycharmProjects/trial 2/final.png', redsq_gray)
cv2.imshow('Original image', image)
cv2.imshow('Gray image', redsq_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
