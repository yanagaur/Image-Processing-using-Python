# -*- coding: utf-8 -*-
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu
import glob


time = 0
time_list = []
area_list = []
path = "/Users/yanagaur/Downloads/scratch_assay/*.*"

for file in glob.glob(path):
    img = io.imread(file)
    entropy_img = entropy(img, disk(10))
    thresh = threshold_otsu(entropy_img)
    binary = entropy_img <= thresh
    #print(thresh)
    #plt.imshow(entropy_img)
    #plt.imshow(binary)
    scratch_area = np.sum(binary == 1)
    #print(time , scratch_area)
    time_list.append(time)
    area_list.append(scratch_area)
    time += 1

#print(time_list,area_list)
plt.plot(time_list , area_list, 'bo')
    
from scipy.stats import linregress
print(linregress(time_list, area_list))