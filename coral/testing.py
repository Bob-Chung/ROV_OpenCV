import cv2, matplotlib.pyplot as plt
import numpy as np, math
plt.rcParams['figure.figsize'] = [15, 8]

LOWER_WHITE = np.array([80, 0, 180]) 
UPPER_WHITE = np.array([125, 100, 255])
LOWER_PINK = np.array([140, 60, 125])
UPPER_PINK = np.array([170, 255, 255])
# Functions

# return the parts of image that is within the given color range
def filter(srcimg, lower_color, upper_color):
    mask = cv2.inRange(srcimg, lower_color, upper_color)
    return cv2.bitwise_and(srcimg, srcimg, mask = mask)

def mask2rgb(img,rgb):
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    img[np.where((img!=[0,0,0]).all(axis=2))] = rgb
    return img

def filter2color(img, lower_color, upper_color, color):
    img = cv2.GaussianBlur(img, (15, 15),0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img, lower_color, upper_color)
    img = cv2.bitwise_and(img, img, mask = mask)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    img[np.where((img!=[0,0,0]).all(axis=2))] = color
    return img

def findM(img1,img2):
    MIN_MATCH_COUNT = 10
    white=(255,255,255)
    img1 = filter2color(img1, LOWER_WHITE, UPPER_WHITE, white) + filter2color(img1, LOWER_PINK, UPPER_PINK, white)         # queryImage
    img2 = filter2color(img2, LOWER_WHITE, UPPER_WHITE, white) + filter2color(img2, LOWER_PINK, UPPER_PINK, white) 
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w,d = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    return M

def allRGB(img):
    all_rgb_codes = img.reshape(-1, img.shape[-1])
    unique_rgbs = np.unique(all_rgb_codes, axis=0)
    return unique_rgbs

def Erode(img):
    kernel = np.ones((13, 13), np.uint8)
    img=cv2.erode(img,kernel)
    img=cv2.dilate(img,kernel)
    img = cv2.Canny(img, 50, 150)
    return img

def drawRec(img,frame,color):
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    try: hierarchy = hierarchy[0]
    except: hierarchy = []

    # computes the bounding box for the contour, and draws it on the frame,
    for contour, hier in zip(contours, hierarchy):
        (x,y,w,h) = cv2.boundingRect(contour)
        print("test")
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)

    return frame

def findDif(old,new):
    white = (255,255,255)
    red = (255,0,0)
    img1=old
    img2=new

    LOWER_WHITE = np.array([80, 0, 180]) 
    UPPER_WHITE = np.array([125, 100, 255]) 
    img1w = filter2color(img1, LOWER_WHITE, UPPER_WHITE, white)

    img2w = filter2color(img2, LOWER_WHITE, UPPER_WHITE, white)

    LOWER_PINK = np.array([140, 60, 125])
    UPPER_PINK = np.array([170, 255, 255])
    img1p = filter2color(img1, LOWER_PINK, UPPER_PINK, red)

    img2p = filter2color(img2, LOWER_PINK, UPPER_PINK, red)

    img1wp = img1w + img1p
    img2wp = img2w + img2p

    M=findM(img1,img2)
    transformed=cv2.warpPerspective(img1, M, (img2.shape[1],img2.shape[0]))
    transformed=filter2color(transformed, LOWER_WHITE, UPPER_WHITE, white) + filter2color(transformed, LOWER_PINK, UPPER_PINK, red)
    img1wp=transformed
    dif=np.zeros(img2.shape,dtype=np.uint8)
    dif=img1wp-img2wp
    green = (0,130,0)
    yellow = (0,255,255)
    cyan = (255,0,255)
    blue = (255,0,0)
    red = (0,0,255)
    black = (0,0,0)
    white = (255,255,255)

    Gdif=np.zeros(img2.shape,dtype=np.uint8)
    Gdif[np.where((dif==[1,0,0]).all(axis=2))] = green

    Ydif=np.zeros(img2.shape,dtype=np.uint8)
    Ydif[np.where((dif==[255,0,0]).all(axis=2))] = yellow
    Ydif[np.where((dif==[255,255,255]).all(axis=2))] = yellow

    Rdif=np.zeros(img2.shape,dtype=np.uint8)
    Rdif[np.where((dif==[0,1,1]).all(axis=2))] = red

    Udif=np.zeros(img2.shape,dtype=np.uint8)
    Udif[np.where((dif==[0,255,255]).all(axis=2))] = blue

    Gdif=Erode(Gdif)
    img2=drawRec(Gdif,img2,green)
    Ydif=Erode(Ydif)
    img2=drawRec(Ydif,img2,yellow)
    Rdif=Erode(Rdif)
    img2=drawRec(Rdif,img2,red)
    Udif=Erode(Udif)
    img2=drawRec(Udif,img2,blue)

    dif=Gdif+Ydif+Rdif+Udif

    concat=cv2.hconcat([Gdif,Ydif,Rdif,Udif])

    return img2

def Resize(img, scale):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def empty(a):
    pass


cv2.namedWindow("mask hsv")
cv2.resizeWindow("mask hsv",640,600)
cv2.createTrackbar("White Hmax","mask hsv",UPPER_WHITE[0],179,empty)
cv2.createTrackbar("White Hmin","mask hsv",LOWER_WHITE[0],179,empty)
cv2.createTrackbar("White Smax","mask hsv",UPPER_WHITE[1],255,empty)
cv2.createTrackbar("White Smin","mask hsv",LOWER_WHITE[1],255,empty)
cv2.createTrackbar("White Vmax","mask hsv",UPPER_WHITE[2],255,empty)
cv2.createTrackbar("White Vmin","mask hsv",LOWER_WHITE[2],255,empty)

cv2.createTrackbar("Pink Hmax","mask hsv",UPPER_PINK[0],179,empty)
cv2.createTrackbar("Pink Hmin","mask hsv",LOWER_PINK[0],179,empty)
cv2.createTrackbar("Pink Smax","mask hsv",UPPER_PINK[1],255,empty)
cv2.createTrackbar("Pink Smin","mask hsv",LOWER_PINK[1],255,empty)
cv2.createTrackbar("Pink Vmax","mask hsv",UPPER_PINK[2],255,empty)
cv2.createTrackbar("Pink Vmin","mask hsv",LOWER_PINK[2],255,empty)

img1 = cv2.imread("resources/coralcap.jpg") #past photo of coral
#img2 = cv2.imread("resources/Now_Coral_Pic.png")
path = cv2.VideoCapture("resources/coral1.mp4") #stream from cam
capture = path

while True:

    #isTrue, img2 = capture.read()

    #if isTrue:
        
    #img2=Resize(img2,0.5)
        
    #tuning mask part
    whmax=cv2.getTrackbarPos("White Hmax","mask hsv")
    whmin=cv2.getTrackbarPos("White Hmin","mask hsv")
    wsmax=cv2.getTrackbarPos("White Smax","mask hsv")
    wsmin=cv2.getTrackbarPos("White Smin","mask hsv")
    wvmax=cv2.getTrackbarPos("White Vmax","mask hsv")
    wvmin=cv2.getTrackbarPos("White Vmin","mask hsv")
    phmax=cv2.getTrackbarPos("Pink Hmax","mask hsv")
    phmin=cv2.getTrackbarPos("Pink Hmin","mask hsv")
    psmax=cv2.getTrackbarPos("Pink Smax","mask hsv")
    psmin=cv2.getTrackbarPos("Pink Smin","mask hsv")
    pvmax=cv2.getTrackbarPos("Pink Vmax","mask hsv")
    pvmin=cv2.getTrackbarPos("Pink Vmin","mask hsv")
    LOWER_WHITE = np.array([whmin, wsmin, wvmin]) 
    UPPER_WHITE = np.array([whmax, wsmax, wvmax])
    LOWER_PINK = np.array([phmin, psmin, pvmin])
    UPPER_PINK = np.array([phmax, psmax, pvmax])
    img1w = filter2color(img1,LOWER_WHITE,UPPER_WHITE,(255,255,255))
    img1p = filter2color(img1,LOWER_PINK,UPPER_PINK,(0,0,255))
    #img2w = filter2color(img2,LOWER_WHITE,UPPER_WHITE,(255,255,255))
    #img2p = filter2color(img2,LOWER_PINK,UPPER_PINK,(0,0,255))
    img1wp=img1w+img1p
    #img2wp=img2w+img2p
    cv2.imshow('img1',img1wp)
    #cv2.imshow('img2',img2wp)

    #stream
    """
    try:
        output1=findDif(img1,img2)
    except:
        cv2.imshow("output",img2)
    else:
        cv2.imshow("output",output1)
    """
    cv2.waitKey(1)
