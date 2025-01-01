# -*- coding: utf-8 -*-
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float

my_image = io.imread("/Users/yanagaur/Downloads/Image_1.png")
#print(my_image)
'''print(my_image.min(),my_image.max())
my_float_image = img_as_float(my_image)
print(my_float_image.min(),my_float_image.max())
#random_image = np.random.random([500, 500])
#plt.imshow(random_image)
#print(random_image)
#print(random_image.min(),random_image.max())
dark_image = my_image*0.5
print(dark_image.max())'''

my_image[10:200, 10:200, :]= [255,0,0]
my_image[300:600, 300:600, :]= [255,255,0]
plt.imshow(my_image)



