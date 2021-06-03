# import cv2
# import numpy as np

# def stitchingFunction():

#     color = [255, 255, 255]  # background of stitching
#     b = 159            # initial value of threshold1
#     c = 59             # initial value of threshold2
#     v = 5000           # put suitable area to control shape detection
#     dim = (360, 144)   # size of forward & top & back sides
#     dim2 = (144, 144)  # size of right & left sides
#     p = 15             # thickness of borders which used to stitch
#     position = (4, 20)

#     frameWidth = 640
#     frameHeight = 480
#     cap = cv2.VideoCapture(0)
#     cap.set(3, frameWidth)
#     cap.set(4, frameHeight)
#     o = 0


#     def empty():
#         pass


#     cv2.namedWindow("Parameters")
#     cv2.resizeWindow("Parameters", 640, 240)
#     cv2.createTrackbar("Threshold1", "Parameters", b, 255, empty)
#     cv2.createTrackbar("Threshold2", "Parameters", c, 255, empty)
#     cv2.createTrackbar("Area", "Parameters", v, 30000, empty)


#     def stack_images(scale, img_array):
#         rows = len(img_array)
#         cols = len(img_array[0])
#         rows_available = isinstance(img_array[0], list)
#         width = img_array[0][0].shape[1]
#         height = img_array[0][0].shape[0]
#         if rows_available:
#             for x in range(0, rows):
#                 for y in range(0, cols):
#                     if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
#                         img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
#                     else:
#                         img_array[x][y] = cv2.resize(img_array[x][y],
#                                                     (img_array[0][0].shape[1], img_array[0][0].shape[0]),
#                                                     None, scale, scale)
#                     if len(img_array[x][y].shape) == 2:
#                         img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
#             image_blank = np.zeros((height, width, 3), np.uint8)
#             hor = [image_blank] * rows
#             for x in range(0, rows):
#                 hor[x] = np.hstack(img_array[x])
#             ver = np.vstack(hor)
#         else:
#             for x in range(0, rows):
#                 if img_array[x].shape[:2] == img_array[0].shape[:2]:
#                     img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
#                 else:
#                     img_array[x] = cv2.resize(img_array[x], (img_array[0].shape[1],
#                                                             img_array[0].shape[0]), None, scale, scale)
#                 if len(img_array[x].shape) == 2:
#                     img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
#             hor = np.hstack(img_array)
#             ver = hor
#         return ver


#     def get_contours(image, img_contour, i):
#         global u
#         contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#         for cnt in contours:
#             u = 0
#             area = cv2.contourArea(cnt)
#             area_min = cv2.getTrackbarPos("Area", "Parameters")
#             if area > area_min:
#                 # cv2.drawContours(img_contour, cnt, -1, (255, 0, 255), 7) # to Contour the Edges of Box.
#                 peri = cv2.arcLength(cnt, True)
#                 approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
#                 print(len(approx))
#                 x, y, w, h = cv2.boundingRect(approx)
#                 cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)
#                 k = cv2.waitKey(1)

#                 if k % 256 == 27:
#                     # ESC pressed
#                     print("Escape hit, closing...")
#                     break

#                 elif k % 256 == 32:
#                     if len(approx) == 4:
#                         cropped_contour = img_contour[y+5:y+h-5, x+5:x+w-5]
#                         image_name = "img_{}.png".format(i)
#                         cv2.imwrite(image_name, cropped_contour)
#                         print("{} written!".format(image_name))
#                         u = 1

#         return u


#     while True:
#         if o < 5:
#             success, img = cap.read()
#             imgContour = img.copy()
#             imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
#             imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
#             threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
#             threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
#             imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
#             kernel = np.ones((5, 5))
#             imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

#             if get_contours(imgDil, imgContour, o) == 1:
#                 o += 1

#             img_stack = stack_images(0.8, ([img, imgGray, imgCanny], [imgDil, imgContour, imgContour]))
#             cv2.imshow("Result", img_stack)
#         if o >= 5:
#             # resizing images
#             top = cv2.resize(cv2.imread("img_0.png"), dim, interpolation=cv2.INTER_AREA)
#             forward = cv2.resize(cv2.imread("img_1.png"), dim, interpolation=cv2.INTER_AREA)
#             back = cv2.resize(cv2.imread("img_3.png"), dim, interpolation=cv2.INTER_AREA)
#             right = cv2.resize(cv2.imread("img_2.png"), dim2, interpolation=cv2.INTER_AREA)
#             left = cv2.resize(cv2.imread("img_4.png"), dim2, interpolation=cv2.INTER_AREA)

#             # stitching
#             vis1 = np.concatenate((top, forward), axis=0)
#             left = cv2.copyMakeBorder(left, 144, 0, 0, 0, cv2.BORDER_CONSTANT, value=color)
#             right = cv2.copyMakeBorder(right, 144, 0, 0, 0, cv2.BORDER_CONSTANT, value=color)
#             back = cv2.copyMakeBorder(back, 144, 0, 0, 0, cv2.BORDER_CONSTANT, value=color)
#             vis = np.concatenate((left, vis1, right, back), axis=1)
#             cv2.imwrite('out.png', vis)
#             cv2.imshow('out.png', vis)

#         if cv2.waitKey(1) & 0xFF == ord('q') & 0xFF == ord('27'):
#             break

import cv2
import numpy as np
def mainStitching(inputCam):
	frameWidth = 640
	frameHeight = 480
	rect_width = 400
	rect_height = 200
	square_size = rect_height


	x_s_square = frameWidth//2-square_size//2
	y_s_square = frameHeight//2-square_size//2

	x_e_square = frameWidth//2+square_size//2
	y_e_square = frameHeight//2+square_size//2

	s_square_coordinatees = (x_s_square,y_s_square)
	e_square_coordinatees = (x_e_square, y_e_square)


	x_s_rect = frameWidth//2-rect_width//2
	y_s_rect = frameHeight//2-rect_height//2

	x_e_rect = frameWidth//2+rect_width//2
	y_e_rect = frameHeight//2+rect_height//2

	s_rect_coordinatees = (x_s_rect, y_s_rect)
	e_rect_coordinatees = (x_e_rect,y_e_rect)

	color = [255, 255, 255]  # background of stitching
	b = 159            # initial value of threshold1
	c = 59             # initial value of threshold2
	v = 5000           # put suitable area to control shape detection
	dim = (360, 144)   # size of forward & top & back sides
	dim2 = (144, 144)  # size of right & left sides
	p = 15             # thickness of borders which used to stitch
	position = (4, 20)

	frameWidth = 640
	frameHeight = 480
	cap = cv2.VideoCapture(f'udpsrc port=5{inputCam}00 ! application/x-rtp, encoding-name=JPEG,payload=26 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
	cap.set(3, frameWidth)
	cap.set(4, frameHeight)
	o = 0
	count = 0


	counter = 1

	imgs = []

	while True:
		success, img = cap.read()
		# print(success)
		height, width, channels = img.shape
		# print(height, width)
		# x1 = width//2 - 300
		# y1 = height//2 - 300
		# x2 = width // 2 + (width * 0.5)
		# y2 = height // 2 + (height * 0.5)
		# print(int(x1),int(y1),"   ", int(x2),int(y2))



		# cv2.rectangle(img, x_square_coordinatees, y_square_coordinatees, (0,0,0), 5)
		# img0 = img[100:100+300, 100:100+300]


		# cv2.rectangle(img, x_rect_coordinatees, y_rect_coordinatees,(0,255,0),3)
		# img1 = img[100:100+150, 100:100+350]
		img_cpy = img.copy()
		# if counter <=2:
		# 	cv2.rectangle(img_cpy, s_square_coordinatees, e_square_coordinatees, (0,0,0), 3)		#img0 = img[100:100+300, 100:100+300]
		# elif counter <=5:
		# 	cv2.rectangle(img_cpy, s_rect_coordinatees, e_rect_coordinatees,(0,255,0),3)
		if counter == 5:
			cv2.rectangle(img_cpy, s_rect_coordinatees, e_rect_coordinatees,(0,255,0),3)
		elif counter > 5:
			break
		elif counter % 2 != 0:
			cv2.rectangle(img_cpy, s_square_coordinatees, e_square_coordinatees, (255,0,0), 3)		#img0 = img[100:100+300, 100:100+300]
		elif counter % 2 == 0:		
			cv2.rectangle(img_cpy, s_rect_coordinatees, e_rect_coordinatees,(0,0,255),3)



		# cv2.imshow("img0", img0)
		# cv2.imshow("img1", img1)

		if cv2.waitKey(1) & 0xFF == ord('s'):
			#cv2.rectangle(img, x_square_coordinatees, y_square_coordinatees, (0,0,0), 5)
			if counter<=2:
				img0 = img[y_s_square:y_e_square, x_s_square:x_e_square]
				imgs.append(img0)
				cv2.imshow("img0", img0)
			else:
				img1 = img[y_s_rect:y_e_rect, x_s_rect:x_e_rect]
				imgs.append(img1)
				cv2.imshow("img1", img1)
			counter+=1
			#break
		
		#if counter == 0:
			#key= cv2.waitKey(30)
			#if key  == cv2.waitKey(1) & 0xFF == ord('S') :
			#	cv2.rectangle(img, x_square_coordinatees, y_square_coordinatees, (0,0,0), 5)
			#	img0 = img[100:100+300, 100:100+300]
			#	cv2.imshow("img0", img0)
			#	counter+=1
			#	print(counter)

		# print(imgs)
		cv2.imshow("Result", img_cpy)

	for i, img in enumerate(imgs):
		print(f"{i}.jpg")
		cv2.imwrite(f"{i}.jpg",img)

	top = cv2.resize(imgs[4], dim, interpolation=cv2.INTER_AREA)
	forward = cv2.resize(imgs[1], dim, interpolation=cv2.INTER_AREA)
	back = cv2.resize(imgs[3], dim, interpolation=cv2.INTER_AREA)
	right = cv2.resize(imgs[2], dim2, interpolation=cv2.INTER_AREA)
	left = cv2.resize(imgs[0], dim2, interpolation=cv2.INTER_AREA)
	cv2.imwrite(f"top.jpg",top)
	cv2.imwrite(f"forward.jpg",forward)
	cv2.imwrite(f"back.jpg",back)
	cv2.imwrite(f"left.jpg",left)


	# for i, img in enumerate(imgs):
	# 	print(f"{i}.jpg")
	# 	cv2.imwrite(f"{i}.jpg",img)


	vis1 = np.concatenate((top, forward), axis=0)
	left = cv2.copyMakeBorder(left, 144, 0, 0, 0, cv2.BORDER_CONSTANT, value=color)
	right = cv2.copyMakeBorder(right, 144, 0, 0, 0, cv2.BORDER_CONSTANT, value=color)
	back = cv2.copyMakeBorder(back, 144, 0, 0, 0, cv2.BORDER_CONSTANT, value=color)
	vis = np.concatenate((left, vis1, right, back), axis=1)
	cv2.imshow('out.jpg', vis)
	cv2.waitKey(0)
	cv2.imwrite('out.jpg', vis)

		# if cv2.waitKey(1) & 0xFF == ord('q') & 0xFF == ord('27'):
		#     break


