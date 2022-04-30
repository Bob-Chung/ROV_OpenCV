import cv2
import numpy as np
global mouseX,mouseY,tmp,counter,x1,y1,x2,y2,x3,y3,x4,y4,cropped,img1,img2,img3,img4,img5,mosaic,imgCount
counter=0
imgCount=0

path = cv2.VideoCapture("resources/video3.mp4") #set to 0 for webcam stream
capture = path


def pos2(event,x,y,flags,param):
    global tmp,counter,x1,y1,x2,y2,x3,y3,x4,y4,cropped,img1,img2,img3,img4,img5,mosaic,imgCount
    if event == cv2.EVENT_LBUTTONDOWN:
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
            stitch()

def stitch():
    global tmp,counter,x1,y1,x2,y2,x3,y3,x4,y4,cropped,img1,img2,img3,img4,img5,mosaic,imgCount
    
    print(imgCount)
    imgCount=imgCount+1
    if imgCount==1:
        img1=cropped
        mosaic=img1
    elif imgCount==2:
        img2=cropped
        mosaic=cv2.hconcat([mosaic, img2])
    elif imgCount==3:
        img3=cropped
        mosaic=cv2.hconcat([mosaic, img3])
    elif imgCount==4:
        img4=cropped
        mosaic=cv2.hconcat([mosaic, img4])
    elif imgCount==5:
        img5=cropped
        ph1 = np.zeros(img1.shape, dtype=np.uint8)
        ph3_width=img2.shape[1]+img3.shape[1]+img4.shape[1]-img5.shape[1]
        ph3 = np.zeros((150,ph3_width,3), dtype=np.uint8)
        imgUpper=cv2.hconcat([ph1,img5,ph3])
        imgLower=mosaic
        mosaic = cv2.vconcat([imgUpper, imgLower])
        imgCount=0
    
    cv2.imshow('photomosaic', mosaic)



def cropImg(temp):
    scale=150
    pts1 = np.float32([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])

    widthDist = np.linalg.norm(pts1[0] - pts1[1])
    heightDist = np.linalg.norm(pts1[0] - pts1[2])

    if (widthDist/heightDist)>1.5:
        width,height=2*scale,scale
    elif (widthDist/heightDist)>0.67:
        width,height=scale,scale
    else: width,height=scale,2*scale



    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    temp1 = cv2.warpPerspective(temp, matrix, (width,height))
    return temp1

while True:
    isTrue, frame = capture.read()
    tmp=frame.copy()
    if isTrue:
        if counter==1:
            cv2.circle(frame,(x1,y1),3, (255,0,0), -1)
        elif counter==2:
            cv2.circle(frame,(x1,y1),3, (255,0,0), -1)
            cv2.circle(frame,(x2,y2),3, (255,0,0), -1)
        elif counter==3:
            cv2.circle(frame,(x1,y1),3, (255,0,0), -1)
            cv2.circle(frame,(x2,y2),3, (255,0,0), -1)
            cv2.circle(frame,(x3,y3),3, (255,0,0), -1)
        cv2.imshow('Video', frame)
        cv2.setMouseCallback('Video',pos2)
        key = cv2.waitKey(100) & 0xFF
    else: capture = path

capture.release()
cv2.destroyAllWindows()
