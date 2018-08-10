from Tkinter import *
from PIL import Image, ImageTk
import random
import os, sys, time
import re


tkpi = None 
afname = ""
imagenum = 0
maximgs = 20

def changeImage():
    global afname
    if started and imagenum[0] <= maximgs:
        #gets list of file names in certain directory. In this case, the directory it is in
        adirlist = os.listdir('./imagesPC')
        adirsize = len(adirlist)
        arandInt = random.randint(0, adirsize-1)

        afname = adirlist[arandInt]
        aimage = Image.open('./imagesPC/' + afname)
        aimage = aimage.resize((256, 256),Image.ANTIALIAS)
    
        #find out which image will be displayed in which label...   
        tkpi = ImageTk.PhotoImage(aimage)     #Creates a Tkinter compatible photo image
        #randInt = random.randint(0, 1)
 
        alabel_image.config(image=tkpi,width="256",height="256")
        imagenum[0] += 1
        val = imagenum[0]
        if val <= maximgs :       
            taskstring.set(str(val)+"/"+str(maximgs))
            root.update() 

        time.sleep(1.0)
#         if val < maximgs/2 :  
#             time.sleep(1.5) #note: after is ms and sleep is in secs
#         else :
#             time.sleep(0.1)
        displayGray()
        root.after(100,changeImage)


def displayGray() :
    global tkpi
    
    tkpi = ImageTk.PhotoImage(file="gray.png")
    alabel_image.config(image=tkpi,width="256",height="256")

def startcallback():
    global started
    started = True   
    changeImage()
    startButton.place_forget()


###############
root = Tk()
root.title("UB - View Real Image")
root.configure(background='gray')
root.minsize(width=800, height=600)
root.maxsize(width=800, height=600)
root.geometry("800x600+30+30") 
try:
    user = sys.argv[1]
except IndexError:
    user = ""
started = False

#Create the labels to hold the images and other widgets
imgxpad = 120
imgypad = 20
alabel_image = Label(root)
alabel_image.place(x=400-256/2,y=imgypad)

#start button and task label
startButton = Button(root, text="Start", command=startcallback)
startButton.place(x=10, y=600-40, width=120, height=20)
taskstring = StringVar()
tasklabel = Label(root,textvariable=taskstring)
tasklabel.place(x=365, y=600-40, width=120, height=20)


imagenum = [0]
displayGray()
root.mainloop()

