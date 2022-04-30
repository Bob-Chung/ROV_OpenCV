import numpy as np
import cv2 as cv
import os 

frame=cv.imread('underwater photos/image58.jpg')

def pos1(event,x,y,flags,param):
    global mouseX,mouseY,tmp,counter,x1,y1,x2,y2,x3,y3,x4,y4,cropped,img1,img2,img3,img4,img5,mosaic,imgCount
    
    if event == cv.EVENT_LBUTTONDOWN:
        tmp = frame.copy()
        #cv.circle(tmp,(x,y),3, (255,0,0), -1)
        cv.imshow('img', tmp)
        counter=0
        #mouseX,mouseY = x,y

def pos2(event,x,y,flags,param):
    global mouseX,mouseY,tmp,counter,x1,y1,x2,y2,x3,y3,x4,y4,cropped,img1,img2,img3,img4,img5,mosaic,imgCount
    if event == cv.EVENT_LBUTTONDOWN:
        #tmp = frame.copy()
        #cv.circle(tmp,(x,y),3, (255,0,0), -1)
        cv.imshow('img', tmp)
        counter=counter+1
        if counter==1:
            (x1,y1)=(x,y)
        elif counter==2:
            (x2,y2)=(x,y)
        elif counter==3:
            (x3,y3)=(x,y)
        elif counter==4:
            (x4,y4)=(x,y)
            counter=0
            cropped=cropImg(tmp)
            cv.imshow('cropped', cropped)
        mouseX,mouseY = x,y

def pos3(event,x,y,flags,param):
    global mouseX,mouseY,tmp,counter,x1,y1,x2,y2,x3,y3,x4,y4,cropped,img1,img2,img3,img4,img5,mosaic,imgCount
    
    if event == cv.EVENT_LBUTTONDOWN:
        cv.imwrite("cropped_photoset/image5.jpg",cropped)


            

        

def cropImg(temp):
    scale=500
    pts1 = np.float32([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])

    widthDist = np.linalg.norm(pts1[0] - pts1[1])
    heightDist = np.linalg.norm(pts1[0] - pts1[2])

    if (widthDist/heightDist)>1.5:
        width,height=2*scale,scale
    elif (widthDist/heightDist)>0.67:
        width,height=scale,scale
    else: width,height=scale,2*scale



    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv.getPerspectiveTransform(pts1,pts2)
    temp1 = cv.warpPerspective(temp, matrix, (width,height))
    return temp1

cv.namedWindow('img')
cv.setMouseCallback('img',pos2)
cv.namedWindow('cropped')
cv.setMouseCallback('cropped',pos3)
while True:
    
    

    cv.imshow('Video', frame)
    cv.setMouseCallback('Video',pos1)
    #print(imgCount)
    key = cv.waitKey(100) & 0xFF

capture.release()
cv.destroyAllWindows()
