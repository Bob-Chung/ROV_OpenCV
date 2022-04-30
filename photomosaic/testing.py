import cv2
import numpy as np

from recthelper import RectHelper
color=(0,0,0)
thick=2
def rec(img):
    img=cv2.resize(img,(1000,500))
    img=cv2.rectangle(img,(333,60),(667,80),color,thick)
    img=cv2.rectangle(img,(333,420),(667,440),color,thick)
    img=cv2.rectangle(img,(60,167),(80,333),color,thick)
    img=cv2.rectangle(img,(920,167),(940,333),color,thick)
    return img
def sqr(img):
    img=cv2.resize(img,(500,500))
    img=cv2.rectangle(img,(167,60),(333,80),color,thick)
    img=cv2.rectangle(img,(167,420),(333,440),color,thick)
    img=cv2.rectangle(img,(60,167),(80,333),color,thick)
    img=cv2.rectangle(img,(420,167),(440,333),color,thick)
    return img


i=0
while True:
    i=i+1
    input_name='cropped_photoset/image'+str(i)+'.jpg'
    output_name='doc_img/image'+str(i)+'.jpg'
    img=cv2.imread(input_name)
    if img is None:
        break
    height, width, channels=img.shape
    if width/height == 1:
        img=sqr(img)
    elif width/height == 2:
        img=rec(img)
    cv2.imwrite(output_name,img)


cv2.waitKey(0)
