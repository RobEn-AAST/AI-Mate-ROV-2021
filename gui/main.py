import tkinter as tk
from tkinter import *
from stitching import mainStitching
# from flying_the_transect_line import read_video
from task_2_2 import colonyhealthfunction
from musselsTask import calcLitres
from saveVideo import recordVideo
from grid import start_drawing 

cam = "0" # camera 0

def changeString(cameraNum):
   global cam
   cam = cameraNum

def showlitres():
   amount = calcLitres(int(entry.get()))
   
   res.config(text = amount)
   print(amount)
   return amount


window = tk.Tk()
window.geometry('1500x700')
window.title("run rov")

   # the button for transect line mission 

# button1 = tk.Button(window, text = "FLYING  \n TRANSECT \nLINE" , fg = "blue", width = 14, height = 5, command = lambda : read_video(cam))
# button1.place(x=100, y= 150)

   # the button for color detection mission

button2 = tk.Button(window, text = "CORAL COLOR" , fg = "blue", width = 14, height = 5, command = lambda : colonyhealthfunction(cam))
button2.place(x=100, y= 250)


   # the button for stitching mission


button3 = tk.Button(window, text = "STITCHING" , fg = "blue", width = 14, height = 5, command = lambda : mainStitching(cam))
button3.place(x=100, y= 350)

   # button for camera 0

button4 = tk.Button(window, text = "camera 0" , fg = "black", width = 10, height = 2, command = lambda : changeString("0"))
button4.place(x=250, y= 50)

   # button for camera 1

button5 = tk.Button(window, text = "camera 1" , fg = "black", width = 10, height = 2, command = lambda : changeString("1"))
button5.place(x=400, y= 50)

   # button for mussel calculating task

label = tk.Label(window, text = "Mussel Calculating" , fg = "green")
label.place(x= 350, y= 350)
entry = tk.Entry(window)
entry.place(x= 350, y = 370)


   # button for calculate
button6 = tk.Button(window, text = "calculate" , fg = "black", width = 4, height = 1, command = showlitres)
button6.place(x=500, y= 370)

   # show output
strLabel = tk.Label(window, text = "Total Amount : " , fg = "red")
strLabel.place(x= 350, y= 398)
res = tk.Label(window, text = 0, fg = "red")
res.place(x= 470, y= 398)

   # button for saving video

button7 = tk.Button(window, text = "record" , fg = "red", width = 5, height = 2, command = lambda : recordVideo(cam))
button7.place(x=100, y= 50)

   # button for grid mission

button8 = tk.Button(window, text = "GRID \n TASK" , fg = "blue", width = 14, height = 5, command = lambda: start_drawing())
button8.place(x=100, y= 450)



window.mainloop()