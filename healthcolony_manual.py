import cv2
import numpy as np

drawing = False
s_point = (0,0)
e_point = (0,0)

num_rect= 0

GREEN = (0,255,0)
RED = (0,0,255)
BLUE = (255,0,0)
YELLOW = (45,255,255)
BLACK = (0,0,0)

read=True
s_list = []
e_list = []
color_list = []
def mouse_drawing(event, x, y, flags, params):
    global s_point, e_point, drawing, num_rect, s_list, e_list, read
    if event == cv2.EVENT_LBUTTONDOWN:
      read = False
      s_point = (x, y)

    elif event == cv2.EVENT_RBUTTONDOWN:
      drawing = True
      num_rect = num_rect + 1
      e_point= (x, y)
      s_list.append(s_point)
      e_list.append(e_point)
      color_list.append(BLACK)
        
cap = cv2.VideoCapture(0)



cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_drawing)
_, frame = cap.read()

while True:
   if read:
    _, frame = cap.read()
   if drawing :
    # cv2.rectangle(frame,s_point,e_point,(0,0,255),0)
    for i in range(num_rect):
        cv2.rectangle(frame,s_list[i],e_list[i],color_list[i],3)
    
    if cv2.waitKey(1) & 0xFF == ord('g'):
      color_list[num_rect-1] = GREEN
    elif cv2.waitKey(1) & 0xFF == ord('b'):
      color_list[num_rect-1] = RED
    elif cv2.waitKey(1) & 0xFF == ord('d'):
      color_list[num_rect-1] = YELLOW
    elif cv2.waitKey(1) & 0xFF == ord('r'):
      color_list[num_rect-1] = BLUE


   cv2.imshow("Frame", frame)
   key = cv2.waitKey(25)
#    print(key)
   if key== 13:    
     print('done')
     read = True
   elif key == 27:
     break

cap.release()
cv2.destroyAllWindows()
