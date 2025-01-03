# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import io , img_as_float , img_as_ubyte
import matplotlib.pyplot as plt

img = img_as_float(io.imread("/Users/yanagaur/Downloads/BSE_100sigma_noisy.jpg"))
sigma_est = np.mean(estimate_sigma(img, channel_axis = None))

denoise = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode = False , patch_size=5 , patch_distance = 3 , channel_axis = None)
denoise_ubyte = img_as_ubyte(denoise)
#plt.imshow(denoise, cmap='gray')

#plt.hist(denoise_ubyte.flat, bins=100, range = (0,255))
# Define refined thresholds
segm1 = (img <= 90)  # Low intensity
segm2 = (img > 90) & (img <= 160)  # Medium intensity
segm3 = (img > 160)  # High intensity

# Create an empty RGB image for segmentation visualization
all_segments = np.zeros((img.shape[0], img.shape[1], 3), dtype=float)

# Assign colors to each segment
all_segments[segm1] = (1, 0, 0)  # Red for Segment 1 (low intensity)
all_segments[segm2] = (0, 1, 0)  # Green for Segment 2 (medium intensity)
all_segments[segm3] = (0, 0, 1)  # Blue for Segment 3 (high intensity)

# Display the segmented image
plt.figure()
plt.imshow(all_segments)
plt.title("Refined Segmented Image")
plt.axis('off')
plt.show()
