import cv2
import numpy as np

from recthelper import RectHelper

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver




path=cv2.VideoCapture("resources/video3.mp4") #set to 0 for cam streaming or file path for video file (i.e. "resources/video2")
cap=path
i=0

while True:
    processor = RectHelper()
    success, img=cap.read()
    #img = cv2.imread('resources/face1.jpg')
    if success:
        img_copy=img.copy()
        img=processor.find_rectangle(img)
        
        grayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        processedImage=processor.edgeDetect(img)
        imgContour=img_copy.copy()
        contours, heirarchy=processor.contour(imgContour)
        cv2.drawContours(imgContour, contours, -1, (0, 0, 0), 2)

        outputImage=stackImages(0.8,([img,grayImage],[imgContour,processedImage]))
        croppedImage=processor.save_rectangle(img)
        recImage=processor.find_all_rectangle(croppedImage)
        cv2.imshow("output",outputImage)
        cv2.imshow("cropped",recImage)
    else:
        cap=path #reconnect video if it is disconnected

    key = cv2.waitKey(1)
    if key == 27: #press escape to exit
        break
    elif key == 113: #press q to save image into folder named "output"
        i=i+1
        file_name='output/image'+str(i)+'.jpg'
        print(file_name)
        cv2.imwrite(file_name, img_copy)



cv2.destroyAllWindows()
