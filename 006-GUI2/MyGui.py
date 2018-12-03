import io  
import os
import subprocess

from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

#========================================
import numpy as np
import cv2
from keras.models import model_from_json
model = model_from_json(open('ani.json').read())
model.load_weights('ani.h5')

#Predicting on single images
def convert_to_array(img):
    im = cv2.imread(img)
    img = Image.fromarray(im, 'RGB')
    image = img.resize((100, 100))
    return np.array(image)
def get_animal_name(label):
    if label==0:
        return "cat"
    if label==1:
        return "dog"
    if label==2:
        return "bird"
    if label==3:
        return "horse"
def predict_animal(file):
    ar=convert_to_array(file)
    ar=ar/255
    label=1
    a=[]
    a.append(ar)
    a=np.array(a)
    score=model.predict(a,verbose=1)
    label_index=np.argmax(score)
    acc=np.max(score)
    animal=get_animal_name(label_index)
    return "The predicted Animal is a "+animal+" with accuracy ="+str(acc)
#========================================

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

#设置窗口标题
root.title('Animal Classifier')

#设置窗口图标
#root.iconbitmap('*****.xbm')
#root.iconbitmap('@/home/nika90/Documents/GUI2/MyGui.xbm')

#setting up a tkinter canvas without scrollbars
frame = Frame(root, bd=0, relief=SUNKEN)
frame.grid_rowconfigure(0, weight=1)
frame.grid_columnconfigure(0, weight=1)
canvas = Canvas(frame, bd=0)
canvas.grid(row=0, column=0, sticky=N+S+E+W)
frame.pack(fill=BOTH,expand=1)

text = StringVar()
text.set('')

#对一个pil_image对象进行缩放，让它在一个矩形框内，还能保持比例 
def resize(w, h, w_box, h_box, pil_image):  
    f1 = 1.0*w_box/w # 1.0 forces float division in Python2  
    f2 = 1.0*h_box/h  
    factor = min([f1, f2])  
    width = int(w*factor)  
    height = int(h*factor)  
    return pil_image.resize((width, height), Image.ANTIALIAS)  

#function to be called when mouse is clicked
def printcoords():
    File = filedialog.askopenfilename(parent=root, initialdir="C:/",title='Choose an image.')

	#=================================================================
    w_box = 1024  
    h_box = 700  
    #以一个PIL图像对象打开  
    pil_image = Image.open(File)  
    #获取图像的原始大小  
    w, h = pil_image.size
    pil_image_resized = resize(w, h, w_box, h_box, pil_image)  
    #=================================================================
	
    filename = ImageTk.PhotoImage(pil_image_resized)
    canvas.image = filename  # <--- keep reference of your image
    canvas.create_image(0,0,anchor='nw',image=filename)
    text.set(predict_animal(File))
	
lb = Label(root, textvariable=text)
lb.pack()
Button(root,text='Choose an image',command=printcoords).pack()

root.mainloop()