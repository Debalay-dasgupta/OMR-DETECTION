import cv2
import numpy as np
from numpy.lib import utils
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
imgBiggestContours = img.copy()
imgGradePoints = img.copy()
#grayscale preprocessing
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#add blur to gray image
imgBlur = cv2.GaussianBlur(imgGray,(5,5),1) #kernel,sigma

#detect edges
imgCanny = cv2.Canny(imgBlur,10,50) #sample threshold

#contours
contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #outeredges,no approximation

cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) # DRAW ALL DETECTED CONTOURS, copied image,all of them,green selected,thickness



#finding rectangles
#reCton is in decreasing order

rectCon = utlis.rectContour(contours)
biggestContour = rectCon[0]
#print(len(biggestContour)) #but we need only the corner points


#corner points 
#where mcq present
biggestContour = utlis.getCornerPoints(rectCon[0])
print(biggestContour)

#where grade box present probably second largest
gradePoints = utlis.getCornerPoints(rectCon[1])
print(gradePoints)

#to check whether desired rectangle is derived

if biggestContour.size !=0 and gradePoints.size != 0:
    cv2.drawContours(imgBiggestContours,biggestContour,-1,(0,255,0),20)
    cv2.drawContours(imgGradePoints,gradePoints,-1,(255,0,0),20) #more thick
    



#stacking all images and display
imgBlank = np.zeros_like(img)

imageArray = ([img,imgGray,imgBlur,imgCanny],[imgContours,imgBiggestContours,imgGradePoints,imgBlank])

imagStacked = utlis.stackImages(imageArray,0.5)#scale given, labels not given

cv2.imshow("Stacked Images",imagStacked)


#cv2.imshow("Answer Script",img)

cv2.waitKey(0)
