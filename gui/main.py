import cv2
import tkinter as tk
from tkinter import *
window = tk.Tk()
window.geometry('500x500')
window.title("run rov")
button1 = tk.Button(window, text = "flying transect line" , fg = "red", width = 10, height = 5)
button1.place(x=185, y= 100)
button2 = tk.Button(window, text = "coral color" , fg = "green", width = 10, height = 5)
button2.place(x=185, y= 200)
button3 = tk.Button(window, text = "stitching" , fg = "blue", width = 10, height = 5)
button3.place(x=185, y= 300)

window.mainloop()