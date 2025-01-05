# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("/Users/yanagaur/Downloads/grains.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# harris corner detector
harris = cv2.cornerHarris(gray, 2, 3, 0.04)
img[harris>0.01*harris.max()] = [255,0,0]
plt.imshow(img)

#################################################################################
#shi tomashi corner detector
import cv2
import numpy as np

img = cv2.imread("/Users/yanagaur/Downloads/grains.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray, 50, 0.01, 10)
corners = np.int32(corners)

for i in corners:
    x,y = i.ravel()
    print(x,y)
    cv2.circle(img,(x,y),3,255,-1)
    

plt.imshow(img)


#################################################################################
# fast corner detector

detector = cv2.FastFeatureDetector_create(50)
kp = detector.detect(img , None)
img2 = cv2.drawKeypoints(img , kp , None , flags=0)
plt.imshow(img2)

#################################################################################
#ORB

import cv2
import numpy as np

orb = cv2.ORB_create(50)
kp , des = orb.detectAndCompute(img, None)
img3 = cv2.drawKeypoints(img , kp , None , flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
plt.imshow(img3)

