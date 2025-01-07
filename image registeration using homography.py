# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
im2 = cv2.imread("/Users/yanagaur/Downloads/monkey.jpg") 
im1 = cv2.imread("/Users/yanagaur/Downloads/monkey_distorted.jpg")
img1 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(50)

kp1 , des1 = orb.detectAndCompute(img1,None)
kp2 , des2 = orb.detectAndCompute(img2,None)

matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
matches = matcher.match(des1,des2,None)

matches = sorted(matches, key = lambda x:x.distance)

points1 = np.zeros((len(matches),2), dtype = np.float32)
points2 = np.zeros((len(matches),2), dtype = np.float32)

for i, match in enumerate(matches):
    points1[i, :]= kp1[match.queryIdx].pt
    points2[i, :]= kp2[match.trainIdx].pt

h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
height, width , channels =im2.shape
im1Reg = cv2.warpPerspective(im1, h, (width,height))
#plt.imshow(im1Reg)


img3 = cv2.drawMatches(im1, kp1, im2, kp2, matches[:10],None)


#img3 = cv2.drawKeypoints(img1, kp1, None, flags=None)
#img4 = cv2.drawKeypoints(img2, kp2, None, flags=None)

plt.imshow(img3)
#plt.imshow(img4)



