import cv2
import sys
import numpy as np

from recthelper import RectHelper

class PhotoMosaic:
    def createMosaic(self, img):
        processor = RectHelper()
        return processor.save_rectangle(img)


