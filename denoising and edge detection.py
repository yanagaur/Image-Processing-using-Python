# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("/Users/yanagaur/Downloads/BSE_Google_noisy.jpg")
kernel = np.ones((5,5), np.float32)/25
filt_2D = cv2.filter2D(img,-1,kernel)
blur = cv2.blur(img,(5,5))
gaussian_blur= cv2.GaussianBlur(img, (5,5),0)
median_blur= cv2.medianBlur(img, 5)
bilateral_blur= cv2.bilateralFilter(img,9,75,75)


cv2.imshow("Image Window", img)
cv2.imshow("2D custom filter", filt_2D)
cv2.imshow("blurred", blur)
cv2.imshow("gaussian blur",gaussian_blur)
cv2.imshow("median blur", median_blur)
cv2.imshow("bilateral blur", bilateral_blur)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to close
        break

cv2.destroyAllWindows()

########################################################################################################################################################\


img = cv2.imread("/Users/yanagaur/Downloads/Neuron.jpg",1)
edges= cv2.Canny(img,100,200)
cv2.imshow("original", img)
cv2.imshow("edges", edges)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to close
        break

cv2.destroyAllWindows()
