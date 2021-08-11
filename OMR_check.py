import cv2
import numpy as np
import utlis
############################################
path = "1.jpg"
widthImg = 700
heightImg = 700
###########################################
img = cv2.imread(path)

#big so rescaling it

img = cv2.resize(img,(widthImg,heightImg))

#grayscale preprocessing
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#add blur to gray image
imgBlur = cv2.GaussianBlur(imgGray,(5,5),1) #kernel,sigma

#detect edges
imgCanny = cv2.Canny(imgBlur,10,50) #sample threshold

#stacking all images and display
imageArray = ([img,imgGray,imgBlur,imgCanny])

imagStacked = utlis.stackImages(imageArray,0.5)#scale given, labels not given

cv2.imshow("Stacked Images",imagStacked)


#cv2.imshow("Answer Script",img)

cv2.waitKey(0)
