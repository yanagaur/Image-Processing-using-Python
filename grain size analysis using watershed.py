# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, color, io


img1 = cv2.imread("/Users/yanagaur/Downloads/grains2.jpg")
img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)


pixels_to_um = 0.5 # 1 pixel = 500 nm (got this from the metadata of original image)

#Threshold image to binary using OTSU. ALl thresholded pixels will be set to 255
ret1, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)

plt.imshow(opening, cmap= "gray")

from skimage.segmentation import clear_border
opening = clear_border(opening) 
sure_bg = cv2.dilate(opening,kernel,iterations=2)


sure_bg = cv2.dilate(opening,kernel,iterations=2)


dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)


ret2, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0)


sure_fg = np.uint8(sure_fg)

unknown = cv2.subtract(sure_bg,sure_fg)

ret3, markers = cv2.connectedComponents(sure_fg)
plt.imshow(sure_fg, cmap= "gray")


markers = markers+10


markers[unknown==255] = 0

markers = cv2.watershed(img1,markers)

img1[markers == -1] = [0,255,255]  

img2 = color.label2rgb(markers, bg_label=0)

plt.imshow(img1)
plt.imshow(img2)

regions = measure.regionprops(markers, intensity_image=img)

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
output_file.write('Grain #' + "," + "," + ",".join(propList) + '\n') #join strings in array by commas, 
#First cell to print grain number
#Second cell blank as we will not print Label column

grain_number = 1
for region_props in regions:
    output_file.write(str(grain_number) + ',')
    #output cluster properties to the excel file
#    output_file.write(str(region_props['Label']))
    for i,prop in enumerate(propList):
        if(prop == 'Area'): 
            to_print = region_props[prop]*pixels_to_um**2   #Convert pixel square to um square
        elif(prop == 'orientation'): 
            to_print = region_props[prop]*57.2958  #Convert to degrees from radians
        elif(prop.find('Intensity') < 0):          # Any prop without Intensity in its name
            to_print = region_props[prop]*pixels_to_um
        else: 
            to_print = region_props[prop]     #Reamining props, basically the ones with Intensity in its name
        output_file.write(',' + str(to_print))
    output_file.write('\n')
    grain_number += 1
    
output_file.close()   #Closes the file, otherwise it would be read only. 


