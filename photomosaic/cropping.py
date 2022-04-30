#this program reads all images from the file "output" and return the cropped version into the file "cropped_output"
#file names are expected to be "image[number].jpg" and are sequential
import cv2
import numpy as np
from recthelper import RectHelper
processor=RectHelper()
i=0
while True:
    i=i+1
    input_name='output/image'+str(i)+'.jpg'
    output_name='cropped_output/image'+str(i)+'.jpg'
    img=cv2.imread(input_name)
    if img is None:
        break
    print(input_name + " read")
    cropped_image=processor.save_rectangle(img)
    cv2.imwrite(output_name,cropped_image)