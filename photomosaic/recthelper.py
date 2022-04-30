import cv2
import sys
import numpy as np

# PIPELINE OF DETECTING RECTANGLE
# - Find contour
#   - Greyscale
#   - Gaussian blur a bit
#   - Set threshold for the gradient range to be considered as edge
#     (1st number - higher: soft threshold;
#      2nd number - lower: hard threshold;
#      Between two numbers: considered edge of neighbouring is edge)
#   - Canny edge detection (resulting image: black backbround with countours as white lines)
#   - Set consideration size for dilation and erosion
#   - Dilation + Erosion (for thicker countour lines)
#   - Return contour
# - Find biggest rectangle
#   - Set area threshold (to filter away small regions)
#   - Return largest 4-sided shape
# - Reorder points for warp
#       [0]        [1]
#
#       [2]        [3]
# - Get transformation matrix
# - Transform and return rectangular image

class RectHelper:
    def stackImages(self,scale,imgArray):
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
    
    def save_rectangle(self, img):
        widthImg = 600
        heightImg = 300
        imgBigContour = img.copy()
        contours, hierarchy = self.contour(imgBigContour)
        #cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
        
        
        biggest, maxArea = self.biggestContour(contours)
        #print(biggest)
        if biggest.size != 0:
            biggest=self.reorder(biggest)
            
            #"""
            #FOR DISPLAYING THE BIGGEST CONTOUR
            #imgBigContour = img.copy()
            cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 10) # DRAW THE BIGGEST CONTOUR
            imgBigContour = self.drawRectangle(imgBigContour,biggest,2)
            #img = imgBigContour
            #"""
            
            scale=300

            width = np.linalg.norm(biggest[0] - biggest[1])
            height = np.linalg.norm(biggest[0] - biggest[2])

            if width<2 or height<2:
                heightImg,widthImg=scale,2*scale
            elif(width/height)>=0.8 and (width/height)<=1.2:
                heightImg,widthImg=scale,scale
            elif(height/width)>=1.6 and (height/width)<=2.4:
                heightImg,widthImg=2*scale,scale
            else: heightImg,widthImg=scale,2*scale

            #heightImg, widthImg, c = img.shape
            pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
            pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
            return imgWarpColored
        return img

    def find_rectangle(self, img): #outputs image with largest rectangle found drawn on it
        #widthImg = 600
        #heightImg = 300
        imgBigContour = img.copy()
        contours, hierarchy = self.contour(imgBigContour)
        #cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
        
        
        biggest, maxArea = self.biggestContour(contours)
        #print(biggest)
        if biggest.size != 0:
            biggest=self.reorder(biggest)
            
            #"""
            #FOR DISPLAYING THE BIGGEST CONTOUR
            #imgBigContour = img.copy()
            cv2.drawContours(img, biggest, -1, (0, 255, 0), 10) # DRAW THE BIGGEST CONTOUR
            imgBigContour = self.drawRectangle(img,biggest,2)
            #img = imgBigContour
            #"""
            
            #heightImg, widthImg, c = img.shape
            #pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
            #pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
            #matrix = cv2.getPerspectiveTransform(pts1, pts2)
            #imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
            #return imgWarpColored
        return img

    def find_all_rectangle(self, img): #outputs image with all rectangles found drawn on it
        imgContour = img.copy()
        contours, hierarchy = self.contour(imgContour)
        #cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
        
        
        
        for i in contours:
            area = cv2.contourArea(i)
            if area > 50:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02 * peri, True)
                if len(approx) == 4:
                    biggest = approx
                    biggest=self.reorder(biggest)
                #cv2.drawContours(img, biggest, -1, (0, 255, 0), 10) # DRAW THE BIGGEST CONTOUR
                    imgContour = self.drawRectangle(img,biggest,2)
            
        return img

    def edgeDetect(self, img):
        imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1) # ADD GAUSSIAN BLUR
        thres = (60, 20)

        imgCanny = cv2.Canny(imgBlur,thres[0],thres[1]) # APPLY CANNY EDGE DETECTION
        kernel = np.ones((3, 3))
        imgDilate = cv2.dilate(imgCanny, kernel, iterations=2) # APPLY DILATION
        imgErode = cv2.erode(imgDilate, kernel, iterations=1) # APPLY EROSION

        imgCanny2 = cv2.Canny(imgErode,thres[0],thres[1]) # APPLY CANNY EDGE DETECTION

        return imgCanny2

    def contour(self, img):
        #self.initializeTrackbars()
        #thres=self.valTrackbars() # GET TRACK BAR VALUES FOR THRESHOLDS
        #print(thres)
        
        
        imgErode = self.edgeDetect(img)
        #thresh, img_bw = cv2.threshold(imgCanny, 110, 255, 0)
        contours, hierarchy = cv2.findContours(imgErode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
        #cv2.imshow("Middle", imgErode)

        return contours, hierarchy
        #return img
    
    def reorder(self, myPoints):
        myPoints = myPoints.reshape((4, 2))
        myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
        add = myPoints.sum(1)
        myPointsNew[0] = myPoints[np.argmin(add)]
        myPointsNew[3] =myPoints[np.argmax(add)]
        diff = np.diff(myPoints, axis=1)
        myPointsNew[1] =myPoints[np.argmin(diff)]
        myPointsNew[2] = myPoints[np.argmax(diff)]
        return myPointsNew

    def biggestContour(self, contours):
        biggest = np.array([])
        max_area = 0
        for i in contours:
            area = cv2.contourArea(i)
            if area > 50:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02 * peri, True)
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area
        return biggest,max_area

    def drawRectangle(self, img, biggest, thickness):
        cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
        cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
        cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
        cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
        return img

    def nothing(x, y):
        pass

    def initializeTrackbars(self, intialTracbarVals=0):
        cv2.namedWindow("Trackbars")
        cv2.resizeWindow("Trackbars", 360, 240)
        cv2.createTrackbar("Threshold1", "Trackbars", 60, 255, self.nothing)
        cv2.createTrackbar("Threshold2", "Trackbars", 30, 255, self.nothing)

    def valTrackbars(self):
        Threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
        Threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
        src = Threshold1,Threshold2
        return src
