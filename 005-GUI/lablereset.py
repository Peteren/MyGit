from tkinter import Tk, Checkbutton, Label
from tkinter import StringVar, IntVar
from tkinter import *

root = Tk()

#设置窗口大小
form_width = 1024
form_height = 768
#获取屏幕尺寸以计算布局参数，使窗口居屏幕中央
screenwidth = root.winfo_screenwidth()  
screenheight = root.winfo_screenheight() 
alignstr = '%dx%d+%d+%d' % (form_width, form_height, (screenwidth-form_width)/2, (screenheight-form_height)/2)   
root.geometry(alignstr)
#设置窗口是否可变长、宽，True：可变，False：不可变
root.resizable(width=False, height=False)

text = StringVar()
text.set('old')
mytest = 1

def printcoords():
    global mytest
    if mytest == 1:   # if clicked
        text.set('0')
        mytest = 0 
    else:
        text.set('1')
        mytest = 1 
		
lb = Label(root, textvariable=text)
lb.pack()

Button(root,text='Choose an image',command=printcoords).pack()

root.mainloop()