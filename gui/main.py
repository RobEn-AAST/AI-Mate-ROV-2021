import tkinter as tk
from tkinter import *
from stitching import stitchingFunction
from flying_the_transect_line import read_video
from task_2_2 import colonyhealthfunction
from musselsTask import calcLitres
from saveVideo import recordVideo

cam = "0" # camera 0

def changeString(cameraNum):
   global cam
   cam = cameraNum


window = tk.Tk()
window.geometry('700x700')
window.title("run rov")

   # the button for transect line mission 

button1 = tk.Button(window, text = "flying transect line" , fg = "red", width = 13, height = 5, command = lambda : read_video(cam))
button1.place(x=100, y= 150)

   # the button for color detection mission

button2 = tk.Button(window, text = "coral color" , fg = "green", width = 13, height = 5, command = lambda : colonyhealthfunction(cam))
button2.place(x=100, y= 250)


   # the button for stitching mission


button3 = tk.Button(window, text = "stitching" , fg = "blue", width = 13, height = 5, command = stitchingFunction)
button3.place(x=100, y= 350)

   # button for camera 0

button4 = tk.Button(window, text = "camera 0" , fg = "black", width = 10, height = 2, command = lambda : changeString("0"))
button4.place(x=250, y= 50)

   # button for camera 1

button5 = tk.Button(window, text = "camera 1" , fg = "black", width = 10, height = 2, command = lambda : changeString("1"))
button5.place(x=400, y= 50)

   # button for mussel calculating task

label = tk.Label(window, text = "Mussel Calculating" , fg = "green")
label.place(x= 350, y= 150)
entry = tk.Entry(window)
entry.place(x= 350, y = 170)

   # button for calculate
button6 = tk.Button(window, text = "calculate" , fg = "black", width = 4, height = 1, command = lambda : calcLitres(int(entry.get())))
button6.place(x=500, y= 170)

   # show output
res = tk.Label(window, text = f"Result : {calcLitres(7)}", fg = "red")
res.place(x= 350, y= 195)

   # button for saving video

button7 = tk.Button(window, text = "record" , fg = "red", width = 5, height = 2, command = lambda : recordVideo(cam))
button7.place(x=350, y= 95)







window.mainloop()