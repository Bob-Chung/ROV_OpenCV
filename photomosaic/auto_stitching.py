import cv2
import numpy as np
#from recthelper import RectHelper
#processor = RectHelper()
rCounter=0
sCounter=0
scale=500

"""
def pos1(event,x,y,flags,param):
    global mouseX,mouseY,tmp,counter,x1,y1,x2,y2,x3,y3,x4,y4,cropped,img1,img2,img3,img4,img5,mosaic,imgCount
    
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)
"""

def rgb_distance(p1,p2): #return rgb difference
    dist=np.sqrt(np.sum((p1-p2)**2, axis=0))
    return dist

def contrast(img): #return rgb difference of bottom rectangle and centre of image, used to find rectangle with four connected sides
    p1,p2=(0,0,0),(0,0,0)
    counter=0
    for i in range(333,667):
        for j in range(420,440):
            p1=p1+img[j,i]
            counter=counter+1
    p1=p1/counter
    
    counter=0
    for i in range(333,667):
        for j in range(240,260):
            p2=p2+img[j,i]
            counter=counter+1
    p2=p2/counter

    dist=rgb_distance(p1,p2)
    return dist

def recColor(img): #return color in order (top,bottom,left,right)
    counter=0
    p1=(0,0,0)
    for i in range(333,667):
        for j in range(60,80):
            p1=p1+img[j,i]
            counter=counter+1
    p1=p1/counter

    counter=0
    p2=(0,0,0)
    for i in range(333,667):
        for j in range(420,440):
            p2=p2+img[j,i]
            counter=counter+1
    p2=p2/counter

    counter=0
    p3=(0,0,0)
    for i in range(60,80):
        for j in range(167,333):
            p3=p3+img[j,i]
            counter=counter+1
    p3=p3/counter

    counter=0
    p4=(0,0,0)
    for i in range(920,940):
        for j in range(167,333):
            p4=p4+img[j,i]
            counter=counter+1
    p4=p4/counter

    return (p1,p2,p3,p4)

def sqrColor(img): #return color in order (top,bottom,left,right)
    counter=0
    p1=(0,0,0)
    for i in range(167,333):
        for j in range(60,80):
            p1=p1+img[j,i]
            counter=counter+1
    p1=p1/counter

    counter=0
    p2=(0,0,0)
    for i in range(167,333):
        for j in range(420,440):
            p2=p2+img[j,i]
            counter=counter+1
    p2=p2/counter

    counter=0
    p3=(0,0,0)
    for i in range(60,80):
        for j in range(167,333):
            p3=p3+img[j,i]
            counter=counter+1
    p3=p3/counter

    counter=0
    p4=(0,0,0)
    for i in range(420,440):
        for j in range(167,333):
            p4=p4+img[j,i]
            counter=counter+1
    p4=p4/counter

    return (p1,p2,p3,p4)

def inacc(mid,side1,side2,side3,side4):
    total=rgb_distance(mid[0],side4[0])+rgb_distance(mid[1],side2[0])+rgb_distance(mid[2],side1[0])+rgb_distance(mid[3],side3[0])+rgb_distance(side1[3],side2[2])+rgb_distance(side2[3],side3[2])+rgb_distance(side3[3],side4[2])+rgb_distance(side4[3],side1[2])
    return total

def stitch(img5,img1,img2,img3,img4):
    ph1 = np.zeros(img1.shape, dtype=np.uint8)
    ph3 = np.zeros(img3.shape, dtype=np.uint8)
    ph4 = np.zeros(img4.shape, dtype=np.uint8)
    imgUpper=cv2.hconcat([ph1,img5,ph3,ph4])
    imgLower=cv2.hconcat([img1,img2,img3,img4])
    mosaic = cv2.vconcat([imgUpper, imgLower])
    return mosaic

def recElim(list):
    l=len(list)
    while(l>3):
        min_dif=10000
        for i in range(l):
            for j in range(i+1, l):
                color1=recColor(list[i])
                color2=recColor(list[j])
                dif=0
                for k in range(4):
                    dif=dif+rgb_distance(color1[k],color2[k])
                print(i,j,dif)
                if dif<min_dif:
                    target=i
                    min_dif=dif
                dif=0
                for k in range(4):
                    dif=dif+rgb_distance(color1[k],color2[k+(-1)**(k)])
                print(i,j,dif)
                if dif<min_dif:
                    target=i
                    min_dif=dif
        list.pop(target)
        l=l-1
    return list

def sqrElim(list):
    l=len(list)
    while(l>2):
        min_dif=10000
        for i in range(l):
            for j in range(i+1, l):
                color1=sqrColor(list[i])
                color2=sqrColor(list[j])
                dif=0
                for k in range(4):
                    dif=dif+rgb_distance(color1[k],color2[k])
                print(i,j,dif)
                if dif<min_dif:
                    target=i
                    min_dif=dif
        list.pop(target)
        l=l-1
    return list

#sqr=[np.zeros((1,1,1), np.uint8),np.zeros((1,1,1), np.uint8)]
#rec=[np.zeros((1,1,1), np.uint8),np.zeros((1,1,1), np.uint8),np.zeros((1,1,1), np.uint8)]
sqr=[]
rec=[]
i=0
while True:
    i=i+1
    input_name='cropped_photoset/image'+str(i)+'.jpg'
    img=cv2.imread(input_name)
    if img is None:
        break
    height, width, channels=img.shape
    if width/height == 1:
        img=cv2.resize(img,(500,500))
        sqr.append(img)
        #sCounter=sCounter+1
    elif width/height == 2:
        img=cv2.resize(img,(1000,500))
        rec.append(img)
        #rCounter=rCounter+1

if(len(sqr)<2 or len(rec)<3):
    print("not enough images")
    exit()

#eliminate extra images
sqrElim(sqr)
recElim(rec)

#reorder items in rec array so that rec[0] is the centre rectangle
if contrast(rec[1])>contrast(rec[0]):
    temp=rec[0]
    rec[0]=rec[1]
    rec[1]=temp
if contrast(rec[2])>contrast(rec[0]):
    temp=rec[0]
    rec[0]=rec[2]
    rec[2]=temp
#cv2.imshow("middle",rec[0])

midColor=recColor(rec[0])
rec1Color=recColor(rec[1])
rec2Color=recColor(rec[2])
sqr0Color=sqrColor(sqr[0])
sqr1Color=sqrColor(sqr[1])

#case1:    (r0)        /case2:    (r0)        /case3:    (r0)        /case4:    (r0)        /
#      (s0)(r1)(s1)(r2)/      (s1)(r1)(s0)(r2)/      (s0)(r2)(s1)(r1)/      (s1)(r2)(s0)(r1)/

#val1,val2,va3,val4 measures the inaccuracies of case 1, case 2, case 3, case 4
#the case with lowest val will be chosen

val1=inacc(midColor,sqr0Color,rec1Color,sqr1Color,rec2Color)
val2=inacc(midColor,sqr1Color,rec1Color,sqr0Color,rec2Color)
val3=inacc(midColor,sqr0Color,rec2Color,sqr1Color,rec1Color)
val4=inacc(midColor,sqr1Color,rec2Color,sqr0Color,rec1Color)

print(val1,val2,val3,val4)

mosaic=stitch(rec[0],sqr[0],rec[1],sqr[1],rec[2])
min=val1
if(val2<min):
    mosaic=stitch(rec[0],sqr[1],rec[1],sqr[0],rec[2])
    min=val2
if(val3<min):
    mosaic=stitch(rec[0],sqr[0],rec[2],sqr[1],rec[1])
    min=val3
if(val4<min):
    mosaic=stitch(rec[0],sqr[1],rec[2],sqr[0],rec[1])
    min=val4


#rescale image
scale_percent = 30
width = int(mosaic.shape[1] * scale_percent / 100)
height = int(mosaic.shape[0] * scale_percent / 100)
dsize = (width, height)
mosaic=cv2.resize(mosaic,dsize)

cv2.imshow("output",mosaic)
#cv2.setMouseCallback('middle',pos1)
cv2.waitKey(0)
