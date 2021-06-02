import tkinter as tk
from tkinter import *
from tkinter import Label
import numpy as np
from tkinter import font as tkFont 


window = tk.Tk()
window.geometry('1500x600')
window.title("grid mission")


helv36 = tkFont.Font(family='Helvetica', size=36, weight=tkFont.BOLD)
def changeColor(reef):
    global color
    if reef == "yellow":
        color = 1 
    if reef == "blue":
        color = 2
        
    if reef == "green":
        color = 3
        
    if reef == "red":
        color = 4
    return color


def showChar(buttons,row, column):  
    global color
    if color == 1:
        txtColor = "yellow"
    elif color == 2 :
        txtColor = "blue"
    elif color == 3 :
        txtColor = "green"
    elif color == 4 :
        txtColor = "red"
    else :
        txtColor = "black"
    if buttons[column].cget('text') != "":
        buttons[column].configure(text = "")
        return
    
    
    if row == 0:
        buttons[column].configure(text = "O", fg = txtColor, font = helv36) 

    
    if row == 1:
        buttons[column].configure(text = "O", fg = txtColor, font = helv36)
    
    if row == 2:
        buttons[column].configure(text = "O", fg = txtColor, font = helv36)


def give_fun(btns,i):
    btns[i].configure(command = lambda: showChar(btns,0,i))
# buttons for first row

def start_drawing():

    buttons1 = []
    for i in range(9):
        buttons1.append(Button(master = window, text = ""))
        buttons1[i].grid(row = 0, column = i, ipadx = 40, ipady = 60)
        give_fun(buttons1,i)

        # buttons for second row

    buttons2 = []
    for i in range(9):
        buttons2.append(Button(master = window, text = ""))
        buttons2[i].grid(row = 1, column = i, ipadx = 40, ipady = 60)
        give_fun(buttons2,i)

        # buttons for third row

    buttons3 = []
    for i in range(9):
        buttons3.append(Button(master = window, text = ""))
        buttons3[i].grid(row = 2, column = i, ipadx = 40, ipady = 60)
        give_fun(buttons3,i)
        
        # button for sea star

    seaStar = Button(window , text = "Sea Star", fg = 'black', command = lambda: changeColor("yellow"))
    seaStar.grid(row = 5, column = 2, ipadx = 20, ipady = 10)
        
        # button for fragments
    fragments = Button(window , text = "Fragments", fg = 'black', command = lambda: changeColor("green"))
    fragments.grid(row = 5, column = 3, ipadx = 20, ipady = 10)

        # button for colony
    colony = Button(window , text = "Colony", fg = 'black', command = lambda: changeColor("blue"))
    colony.grid(row = 5, column = 4, ipadx = 20, ipady = 10)

        # button for sponge
    sponge = Button(window , text = "Sponge", fg = 'black', command = lambda: changeColor("red"))
    sponge.grid(row = 5, column = 5, ipadx = 20, ipady = 10)

    
    window.mainloop()


start_drawing()
