import cv2
import numpy as np

img=cv2.imread("resources/face1.JPG")
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img=cv2.GaussianBlur(img,(3,3),5)


#find corners of rectangle face
contours,hierarchy=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

#for contour in contours:
#    cv2.drawContours(img,contour,-1,(0,0,0),3)


largest_contour=max(contours, key = cv2.contourArea)
rect=cv2.minAreaRect(largest_contour)
box=cv2.boxPoints(rect)



#crop and transform
width,height=400,200

pts1=np.float32(box)
#pts2=np.float32([[0,0],[width,0],[0,height],[width,height]])
pts2=np.float32([[width,height],[0,height],[0,0],[width,0]])

matrix=cv2.getPerspectiveTransform(pts1,pts2)
imgOutput=cv2.warpPerspective(img,matrix,(width,height))

#cv2.drawContours(img,box,-1,(0,0,0),3)
cv2.imshow('boxed',img)

cv2.imshow("output",imgOutput)

cv2.waitKey(0)
