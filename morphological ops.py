# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("/Users/yanagaur/Downloads/BSE_Google_noisy.jpg",0)
ret , th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU )
median = cv2.medianBlur(img, 3)
plt.imshow(median,cmap="grey")
ret , th = cv2.threshold(median,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU )
kernel = np.ones((3,3),np.uint8)
print(kernel)

erosion = cv2.erode(th, kernel , iterations=1)
dial = cv2.dilate(erosion, kernel, iterations= 1)

opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel) # 1 erosion + 1 dialation

plt.imshow(opening,cmap="grey")
plt.imshow(dial,cmap="grey")
plt.imshow(erosion,cmap="gray")





#plt.imshow(img)
plt.imshow(th,cmap="gray")
plt.hist(img.flat,bins=100,range=(0,255))

