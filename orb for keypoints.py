# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("/Users/yanagaur/Downloads/monkey.jpg")
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create(50)
kp,des = orb.detectAndCompute(img1,None)
img2 = cv2.drawKeypoints(img1, kp, None, flags=None)
plt.imshow(img2)
