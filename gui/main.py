import tkinter as tk
from tkinter import *
from stitching import stitchingFunction
from flying_the_transect_line import read_video
from task_2_2 import colonyhealthfunction


cam = "0" # camera 0

def changeString(camera):
   global cam = camera
   print(cam)


window = tk.Tk()
window.geometry('500x500')
window.title("run rov")

   # the button for transect line mission 

button1 = tk.Button(window, text = "flying transect line" , fg = "red", width = 13, height = 5, command = lambda : read_video(cam))
button1.place(x=185, y= 100)

   # the button for color detection mission

button2 = tk.Button(window, text = "coral color" , fg = "green", width = 13, height = 5, command = lambda : colonyhealthfunction(cam))
button2.place(x=185, y= 200)


   # the button for stitching mission


button3 = tk.Button(window, text = "stitching" , fg = "blue", width = 13, height = 5, command = stitchingFunction)
button3.place(x=185, y= 300)

   # button for camera 0

button4 = tk.Button(window, text = "camera 0" , fg = "black", width = 10, height = 2, command = lambda : changeString("0"))
button4.place(x=100, y= 400)

   # button for camera 1

button5 = tk.Button(window, text = "camera 1" , fg = "black", width = 10, height = 2, command = lambda : changeString("1"))
button5.place(x=297, y= 400)



window.mainloop()