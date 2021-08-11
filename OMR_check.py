import cv2
import numpy as np
import utlis
############################################
path = "1.jpg"
widthImg = 700
heightImg = 700
###########################################
img = cv2.imread(path)
#imgContours = img.copy()  don't make a copy before resizing
#big so rescaling it

#Preprocessing

img = cv2.resize(img,(widthImg,heightImg))
imgContours = img.copy()
#grayscale preprocessing
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#add blur to gray image
imgBlur = cv2.GaussianBlur(imgGray,(5,5),1) #kernel,sigma

#detect edges
imgCanny = cv2.Canny(imgBlur,10,50) #sample threshold

#contours
contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #outeredges,no approximation

cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) # DRAW ALL DETECTED CONTOURS, copied image,all of them,green selected,thickness
       
#stacking all images and display
imgBlank = np.zeros_like(img)
imageArray = ([img,imgGray,imgBlur,imgCanny],[imgContours,imgBlank,imgBlank,imgBlank])

imagStacked = utlis.stackImages(imageArray,0.5)#scale given, labels not given

cv2.imshow("Stacked Images",imagStacked)


#cv2.imshow("Answer Script",img)

cv2.waitKey(0)
