import cv2
import numpy as np
############################################
path = "1.jpg"
widthImg = 700
heightImg = 700

img = cv2.imread(path)

#big so rescaling it

img = cv2.resize(img,(widthImg,heightImg))


cv2.imshow("Answer Script",img)

cv2.waitKey(0)

###check changes
