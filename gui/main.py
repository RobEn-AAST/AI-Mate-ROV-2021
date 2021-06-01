import tkinter as tk
from tkinter import *
from stitching import stitchingFunction

window = tk.Tk()
window.geometry('500x500')
window.title("run rov")

   # the button for transect line mission 


button1 = tk.Button(window, text = "flying transect line" , fg = "red", width = 10, height = 5)
button1.place(x=185, y= 100)

   # the button for color detection mission

button2 = tk.Button(window, text = "coral color" , fg = "green", width = 10, height = 5)
button2.place(x=185, y= 200)


   # the button for stitching mission


button3 = tk.Button(window, text = "stitching" , fg = "blue", width = 10, height = 5, command = stitchingFunction)
button3.place(x=185, y= 300)



window.mainloop()