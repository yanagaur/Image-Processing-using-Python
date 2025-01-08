# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2 
import numpy as np
import matplotlib.pyplot as plt 
from scipy import ndimage
from skimage import io, color, measure

#step 1 
img = cv2.imread("/Users/yanagaur/Downloads/grains2.jpg",0)
pixels_to_um= 0.5 # 1 pixel = 0.5 um or 500 nm

#step 2
#plt.hist(img.flat, bins = 100 , range = (0,255))
ret , thresh = cv2.threshold(img, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#step3
kernel = np.ones((3,3),np.uint8)
eroded = cv2.erode(thresh, kernel,iterations=1)
dilated = cv2.dilate(eroded,kernel , iterations= 1)


mask = dilated == 255

#step 4
s= [[1,1,1],[1,1,1],[1,1,1]]
label_mask, num_label = ndimage.label(mask, structure=s)
img2 = color.label2rgb(label_mask, bg_label=0)
plt.imshow(img2)

#step 5
clusters = measure.regionprops(label_mask, img)
propList = ['Area',
            'equivalent_diameter', #Added... verify if it works
            'orientation', #Added, verify if it works. Angle btwn x-axis and major axis.
            'MajorAxisLength',
            'MinorAxisLength',
            'Perimeter',
            'MinIntensity',
            'MeanIntensity',
            'MaxIntensity']    

output_file = open('image_measurements.csv', 'w')
output_file.write(',' + ",".join(propList) + '\n')

for cluster_props in clusters:
    #output cluster properties to the excel file
    output_file.write(str(cluster_props['Label']))
    for i,prop in enumerate(propList):
        if(prop == 'Area'): 
            to_print = cluster_props[prop]*pixels_to_um**2   #Convert pixel square to um square
        elif(prop == 'orientation'): 
            to_print = cluster_props[prop]*57.2958  #Convert to degrees from radians
        elif(prop.find('Intensity') < 0):          # Any prop without Intensity in its name
            to_print = cluster_props[prop]*pixels_to_um
        else: 
            to_print = cluster_props[prop]     #Reamining props, basically the ones with Intensity in its name
        output_file.write(',' + str(to_print))
    output_file.write('\n')
output_file.close()

#plt.imshow(dilated,cmap="gray")




