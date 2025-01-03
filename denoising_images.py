# -*- coding: utf-8 -*-

import numpy 
from matplotlib import pyplot as plt

def gaussian_kernel(size, size_y=None):
    size = int(size)
    if not size_y:
        size_y = size
    
    else:
        size_y = int(size_y)
    x,y = numpy.mgrid[-size:size+1, -size_y:size_y+1]
    g = numpy.exp(-(x**2/float(size)*y**2/float(size_y)))
    return g/ g.sum()


gaussian_kernel_aaray = gaussian_kernel(3)
print(gaussian_kernel_aaray)
plt.imshow(gaussian_kernel_aaray, cmap= plt.get_cmap('jet'), interpolation = 'nearest')
plt.colorbar()
plt.show()

from skimage import io , img_as_float , img_as_ubyte
from scipy import ndimage as nd
from matplotlib import pyplot as plt
from skimage.restoration import denoise_nl_means, estimate_sigma
import numpy as np

img = img_as_float(io.imread("/Users/yanagaur/Downloads/noisy_img.jpg"))
gaussian_image = nd.gaussian_filter(img , sigma= 3) # cleans noise but there is a lot of information loss # avoid using gaussian 
io.imshow(gaussian_image)

median_img = nd.median_filter(img, size= 5) #doesn't remove all noise , but very less information loss and preserves edges

io.imshow(median_img)

sigma_est = np.mean(estimate_sigma(img, channel_axis = -1))

patch_kw = dict(patch_size=5,      
                patch_distance=3,  
                multichannel=True)

denoise_img = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=False,
                               patch_size=5, patch_distance=3, channel_axis = -1)
"""
denoise_img = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=False,
                           **patch_kw)
"""
denoise_img_as_8byte = img_as_ubyte(denoise_img)

plt.imshow(denoise_img)
#plt.imshow(denoise_img_as_8byte, cmap=plt.cm.gray, interpolation='nearest')