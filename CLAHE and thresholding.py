# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("/Users/yanagaur/Downloads/Alloy.jpg",0)
eq_img = cv2.equalizeHist(img) # equalized i.e increased contrast but added noise becuase of stretching
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl_img= clahe.apply(img)
#plt.hist(img.flat, bins=100 , range=(0,255))
#plt.hist(eq_img.flat, bins=100 , range=(0,255))
plt.hist(cl_img.flat, bins=100 , range=(100,255))

ret, thresh1 = cv2.threshold(cl_img, 190, 150, cv2.THRESH_BINARY)
ret2, thresh2 = cv2.threshold(cl_img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
plt.imshow(img,cmap="gray")
plt.imshow(thresh1,cmap="gray")
plt.imshow(thresh2,cmap="gray")



plt.imshow(eq_img,cmap="gray")
plt.imshow(cl_img,cmap="gray")


