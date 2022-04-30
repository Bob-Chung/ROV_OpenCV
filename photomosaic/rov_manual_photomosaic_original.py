import cv2
import numpy as np

path = cv2.VideoCapture("resources/video3.mp4") #set to 0 for webcam stream
capture = path
def pos1(event,x,y,flags,param):
    global mouseX,mouseY,tmp,counter,x1,y1,x2,y2,x3,y3,x4,y4,cropped,img1,img2,img3,img4,img5,mosaic,imgCount
    
    if event == cv2.EVENT_LBUTTONDOWN:
        tmp = frame.copy()
        #cv.circle(tmp,(x,y),3, (255,0,0), -1)
        cv2.imshow('img', tmp)
        counter=0
        #mouseX,mouseY = x,y

def pos2(event,x,y,flags,param):
    global mouseX,mouseY,tmp,counter,x1,y1,x2,y2,x3,y3,x4,y4,cropped,img1,img2,img3,img4,img5,mosaic,imgCount
    if event == cv2.EVENT_LBUTTONDOWN:
        #tmp = frame.copy()
        cv2.circle(tmp,(x,y),3, (255,0,0), -1)
        cv2.imshow('img', tmp)
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
            cv2.imshow('cropped', cropped)
        mouseX,mouseY = x,y

def pos3(event,x,y,flags,param):
    global mouseX,mouseY,tmp,counter,x1,y1,x2,y2,x3,y3,x4,y4,cropped,img1,img2,img3,img4,img5,mosaic,imgCount
    
    if event == cv2.EVENT_LBUTTONDOWN:
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
            print(img2.shape,img3.shape,img2.shape+img3.shape)
            ph4 = np.zeros(img2.shape+img3.shape+img4.shape-img5.shape, dtype=np.uint8)
            imgUpper=cv2.hconcat([ph1,img5,ph3,ph4])
            imgLower=mosaic
            mosaic = cv2.vconcat([imgUpper, imgLower])


            

        cv2.imshow('photomosaic', mosaic)


def pos4(event,x,y,flags,param):
    global mouseX,mouseY,tmp,counter,x1,y1,x2,y2,x3,y3,x4,y4,cropped,img1,img2,img3,img4,img5,mosaic,imgCount
    
    if event == cv2.EVENT_LBUTTONDBLCLK:
        imgCount=0

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

cv2.namedWindow('img')
cv2.setMouseCallback('img',pos2)
cv2.namedWindow('cropped')
cv2.setMouseCallback('cropped',pos3)
cv2.namedWindow('photomosaic')
cv2.setMouseCallback('photomosaic',pos4)

while True:
    isTrue, frame = capture.read()
    
    if isTrue:
        cv2.imshow('Video', frame)
        cv2.setMouseCallback('Video',pos1)
        #print(imgCount)
        key = cv2.waitKey(100) & 0xFF
    else: capture = path

capture.release()
cv2.destroyAllWindows()
