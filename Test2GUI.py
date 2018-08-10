from Tkinter import *
from PIL import Image, ImageTk
import random
import os, sys, time
import re


tkpi1 = None #create this global variable so that the image is not derefrenced
tkpi2 = None
afname = ""
bfname = ""
imagenum = 0
maximgs = 20
sleepVal = 0

def changeImage():
  global afname
  global bfname
  global sleepVal
  global l_val
  global r_val #AR=1 PC = 2

  if started and imagenum[0] <= maximgs:
    #get random images from directory
    l_val = 0.0
    r_val = 0.0
    bothSame = 0
				
    randInt_ar = random.randint(0, 1)
    randInt_pc = random.randint(0, 1)

    if randInt_ar == 1:
      #randomly select a file from AR list
      adirlist = os.listdir('./imagesAR')
      adirsize = len(adirlist)
      arandInt = random.randint(0, adirsize-1)
      afname = adirlist[arandInt]
      aimage = Image.open('./imagesAR/' + afname)
      aimage = aimage.resize((256, 256),Image.ANTIALIAS)
      l_val =1.0
    else:
      #randomly select a file from PC list
      adirlist = os.listdir('./imagesPC')
      adirsize = len(adirlist)
      arandInt = random.randint(0, adirsize-1)
      afname = adirlist[arandInt]
      aimage = Image.open('./imagesPC/' + afname)
      aimage = aimage.resize((256, 256),Image.ANTIALIAS)
      l_val = 2.0
    f.write(user + "\t" + afname+ "\t")

    if randInt_pc == 1:
      bdirlist = os.listdir('./imagesPC')
      bdirsize = len(bdirlist)
      brandInt = random.randint(0, bdirsize-1)
      bfname = bdirlist[brandInt]
      bimage = Image.open('./imagesPC/' + bfname)
      bimage = bimage.resize((256, 256),Image.ANTIALIAS)
      r_val = 2.0
    else:
      bdirlist = os.listdir('./imagesAR')
      bdirsize = len(bdirlist)      
      brandInt = random.randint(0, adirsize-1)
      bfname = bdirlist[brandInt]
      bimage = Image.open('./imagesAR/' + bfname)
      bimage = bimage.resize((256, 256),Image.ANTIALIAS)
      r_val = 1.0
    f.write(user + "\t" + bfname + "\n")

     #find out which image will be displayed in which label...   
    tkpi1 = ImageTk.PhotoImage(aimage)	   #Creates a Tkinter compatible photo image
    tkpi2 = ImageTk.PhotoImage(bimage)	   
    randInt = random.randint(0, 1)
    if randInt > 0 :
      tmpimg = tkpi1
      tkpi1 = tkpi2
      tkpi2 = tmpimg
      tmpname = afname
      afname = bfname
      bfname = tmpname
    #print "left filenm = " + afname + "; right filenm = " + bfname 
    #		  else:	   
    #			  tkpi1 = ImageTk.PhotoImage(file="gray.png")
    #			  tkpi2 = ImageTk.PhotoImage(file="gray.png")		 

    alabel_image.config(image=tkpi1,width="256",height="256")
    blabel_image.config(image=tkpi2, width="256", height="256")
    imagenum[0] += 1
    val = imagenum[0]
    if val <= maximgs :		  
      taskstring.set(str(val)+"/"+str(maximgs))
      root.update() 

    #user has to hit the start/pick button which calls changeImg
    sleeparray = [450, 500, 550, 600, 650, 700]
    randIdx = random.randint(0, 5)
    sleepVal = sleeparray[randIdx]

    sleepVal = 1000
    time.sleep(sleepVal/1000) #note: after is ms and sleep is in secs
    displayGray()


def displayGray() :
	global tkpi1 #need globals so that the images do not get derefrenced out of function
	global tkpi2

	tkpi1 = ImageTk.PhotoImage(file="gray.png")
	tkpi2 = ImageTk.PhotoImage(file="gray.png") 
	alabel_image.config(image=tkpi1,width="256",height="256")
	blabel_image.config(image=tkpi2, width="256", height="256")

def pickAcallback():
  if started and imagenum[0] <= maximgs:
    tstr = "\t"
    if (l_val == r_val and l_val != 0.0):
      f.write(user + tstr + str(sleepVal) + tstr + "S picked Same\n")
    else:
      f.write(user + tstr + str(sleepVal) + tstr + "F picked Same\n")
    changeImage()

def pickBcallback():
  if started and imagenum[0] <= maximgs:
    tstr = "\t"
    if (l_val != r_val):
      f.write(user + tstr + str(sleepVal) + tstr + "S picked Diff\n")
    else:
      f.write(user + tstr + str(sleepVal) + tstr + "F picked Diff\n")
    changeImage()
    

def startcallback():
  global started
  started = True	 
  changeImage()
  startButton.place_forget()

def on_closing():
  f.close()
  root.destroy()


###############
root = Tk()
root.title("RIT - Are these the same person")
root.configure(background='gray')
root.minsize(width=800, height=600)
root.maxsize(width=800, height=600)
root.geometry("800x600+30+30") 
try:
  user = sys.argv[1]
except IndexError:
  user = ""
started = False

#bdirlist = os.listdir('./imagesPC')
#bdirsize = len(bdirlist)


#Create the labels to hold the images and other widgets
imgxpad = 120
imgypad = 20
alabel_image = Label(root)
alabel_image.place(x=imgxpad,y=imgypad)
blabel_image = Label(root)
blabel_image.place(x=800-256-imgxpad,y=20)	
#image pick buttons
aButton = Button(root, text="Yes-Same", command=pickAcallback)
aButton.place(x=imgxpad+256/4 +85, y=imgypad*2+256,width=120, height=20)
bButton = Button(root, text="No-Diff", command=pickBcallback)
bButton.place(x=800-256/4*3-imgxpad-85, y=imgypad*2+256, width=120, height=20)
#start button and task label
startButton = Button(root, text="Start", command=startcallback)
startButton.place(x=10, y=600-40, width=120, height=20)
taskstring = StringVar()
tasklabel = Label(root,textvariable=taskstring)
tasklabel.place(x=365, y=600-40, width=120, height=20)

#I/O output log file
timestr = time.strftime("%Y%m%d%H%M%S")
filenm = "outfile_"+timestr+".txt"
f = open(filenm,'w')


imagenum = [0]
displayGray()
root.protocol("WM_DELETE_WINDOW", on_closing)  #close the file handle on closing window
root.mainloop()

