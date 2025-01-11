# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import pydicom
from PIL import Image

dataset = pydicom.dcmread("/Users/yanagaur/Downloads/CT_small.dcm")
img=dataset.pixel_array
plt.imshow(img, cmap=plt.cm.bone)
tiff_image = Image.fromarray(img)
tiff_image.save("/Users/yanagaur/Downloads/dcm_to_tiff_converted.tif", format="TIFF")

##########################################################################
#Denoising filters
#####################################################################

#Gaussian

from skimage import img_as_float
from skimage.metrics import peak_signal_noise_ratio
from matplotlib import pyplot as plt
from skimage import io
from scipy import ndimage as nd

noisy_img = img_as_float(io.imread("/Users/yanagaur/Downloads/MRI_noisy.tif"))
#Need to convert to float as we will be doing math on the array
#Also, most skimage functions need float numbers
ref_img = img_as_float(io.imread("/Users/yanagaur/Downloads/MRI_clean.tif"))

gaussian_img = nd.gaussian_filter(noisy_img, sigma=5)
plt.imshow(noisy_img, cmap='gray')
plt.imshow(gaussian_img, cmap='gray')
plt.imsave("images/MRI_images/Gaussian_smoothed.tif", gaussian_img, cmap='gray')

noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
gaussian_cleaned_psnr = peak_signal_noise_ratio(ref_img, gaussian_img)
print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", gaussian_cleaned_psnr)


#######################################################################
#Bilateral, TV and Wavelet

from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage import img_as_float

noisy_img = img_as_float(io.imread("/Users/yanagaur/Downloads/MRI_noisy.tif"))
sigma_est = estimate_sigma(noisy_img, channel_axis = None, average_sigmas=True)

denoise_bilateral = denoise_bilateral(noisy_img, sigma_spatial=15,
                channel_axis = None)

noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
bilateral_cleaned_psnr = peak_signal_noise_ratio(ref_img, denoise_bilateral)
print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", bilateral_cleaned_psnr)

plt.imshow(denoise_bilateral, cmap='gray')

###### TV ###############
denoise_TV = denoise_tv_chambolle(noisy_img, weight=0.3, channel_axis = None)
noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
TV_cleaned_psnr = peak_signal_noise_ratio(ref_img, denoise_TV)
print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", TV_cleaned_psnr)
plt.imshow(denoise_TV, cmap='gray')


####Wavelet #################
wavelet_smoothed = denoise_wavelet(noisy_img, channel_axis = None,
                           method='BayesShrink', mode='soft',
                           rescale_sigma=True)
noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
Wavelet_cleaned_psnr = peak_signal_noise_ratio(ref_img, wavelet_smoothed)
print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", Wavelet_cleaned_psnr)

plt.imshow(wavelet_smoothed, cmap='gray')

#####################
#Shift invariant wavelet denoising
#https://scikit-image.org/docs/dev/auto_examples/filters/plot_cycle_spinning.html
#Not sure if this is doing anything, check

import matplotlib.pyplot as plt

from skimage.restoration import denoise_wavelet, cycle_spin
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio
from skimage import io


noisy_img = img_as_float(io.imread("/Users/yanagaur/Downloads/MRI_noisy.tif"))
#Need to convert to float as we will be doing math on the array
#Also, most skimage functions need float numbers
ref_img = img_as_float(io.imread("/Users/yanagaur/Downloads/MRI_clean.tif"))


denoise_kwargs = dict(channel_axis = None, wavelet='db1', method='BayesShrink',
                      rescale_sigma=True)

all_psnr = []
max_shifts = 3     #0, 1, 3, 5

Shft_inv_wavelet = cycle_spin(noisy_img, func=denoise_wavelet, max_shifts = max_shifts,
                            func_kw=denoise_kwargs, channel_axis = None)

noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
shft_cleaned_psnr = peak_signal_noise_ratio(ref_img, Shft_inv_wavelet)
print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", shft_cleaned_psnr)

plt.imshow(Shft_inv_wavelet, cmap='gray')


##########################################################################
#Anisotropic Diffusion

import matplotlib.pyplot as plt
import cv2
from skimage import io
from medpy.filter.smoothing import anisotropic_diffusion
from skimage import img_as_float
from skimage.metrics import peak_signal_noise_ratio


#img = io.imread("MRI_images/MRI_noisy.tif", as_gray=True)
noisy_img = img_as_float(io.imread("/Users/yanagaur/Downloads/MRI_noisy.tif"))
#Need to convert to float as we will be doing math on the array
#Also, most skimage functions need float numbers
ref_img = img_as_float(io.imread("/Users/yanagaur/Downloads/MRI_clean.tif"))

# niter= number of iterations
#kappa = Conduction coefficient (20 to 100)
#gamma = speed of diffusion (<=0.25)
#Option: Perona Malik equation 1 or 2. A value of 3 is for Turkey's biweight function 
img_aniso_filtered = anisotropic_diffusion(noisy_img, niter=50, kappa=50, gamma=0.2, option=2) 

noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
anisotropic_cleaned_psnr = peak_signal_noise_ratio(ref_img, img_aniso_filtered)
print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", anisotropic_cleaned_psnr)


plt.imshow(img_aniso_filtered, cmap='gray')


##########################################################################
#NLM from SKIMAGE

from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_ubyte, img_as_float
from matplotlib import pyplot as plt
from skimage import io
import numpy as np
from skimage.metrics import peak_signal_noise_ratio


noisy_img = img_as_float(io.imread("/Users/yanagaur/Downloads/MRI_noisy.tif"))
#Need to convert to float as we will be doing math on the array
#Also, most skimage functions need float numbers
ref_img = img_as_float(io.imread("/Users/yanagaur/Downloads/MRI_clean.tif"))

sigma_est = np.mean(estimate_sigma(noisy_img, channel_axis = None))


NLM_skimg_denoise_img = denoise_nl_means(noisy_img, h=1.15 * sigma_est, fast_mode=True,
                               patch_size=9, patch_distance=5, channel_axis = None)

noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
NLM_skimg_cleaned_psnr = peak_signal_noise_ratio(ref_img, NLM_skimg_denoise_img)
print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", NLM_skimg_cleaned_psnr)


denoise_img_as_8byte = img_as_ubyte(NLM_skimg_denoise_img)

plt.imshow(NLM_skimg_denoise_img)
plt.imshow(denoise_img_as_8byte, cmap=plt.cm.gray, interpolation='nearest')
#plt.imsave("images/MRI_images/NLM_skimage_denoised.tif", denoise_img_as_8byte, cmap='gray')
###########################################################################

#NLM opencv
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_photo/py_non_local_means/py_non_local_means.html
# cv2.fastNlMeansDenoising() - works with a single grayscale images
# cv2.fastNlMeansDenoisingColored() - works with a color image.

import numpy as np
from matplotlib import pyplot as plt
from skimage import img_as_ubyte, img_as_float
from matplotlib import pyplot as plt
from skimage import io
import numpy as np
from skimage.metrics import peak_signal_noise_ratio

noisy_img = img_as_float(io.imread("/Users/yanagaur/Downloads/MRI_noisy.tif"))
#Need to convert to float as we will be doing math on the array
#Also, most skimage functions need float numbers
ref_img = img_as_float(io.imread("/Users/yanagaur/Downloads/MRI_clean.tif"))

# fastNlMeansDenoising(InputArray src, OutputArray dst, float h=3, int templateWindowSize=7, int searchWindowSize=21 )

NLM_CV2_denoise_img = cv2.fastNlMeansDenoising(noisy_img, None, 3, 7, 21)



plt.imshow(NLM_CV2_denoise_img, cmap='gray')


###########################################################################

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import img_as_ubyte, img_as_float, io
from skimage.metrics import peak_signal_noise_ratio

# Load noisy and reference images as float
noisy_img = img_as_float(io.imread("/Users/yanagaur/Downloads/MRI_noisy.tif"))
ref_img = img_as_float(io.imread("/Users/yanagaur/Downloads/MRI_clean.tif"))

# Convert noisy image to uint8 for OpenCV
noisy_img_uint8 = img_as_ubyte(noisy_img)

# Apply Non-Local Means Denoising
NLM_CV2_denoise_img = cv2.fastNlMeansDenoising(noisy_img_uint8, None, 3, 7, 21)

# Convert back to float for comparison/visualization if needed
NLM_CV2_denoise_img_float = img_as_float(NLM_CV2_denoise_img)

# Display the denoised image
plt.imshow(NLM_CV2_denoise_img_float, cmap='gray')
plt.title("Denoised Image (Non-Local Means)")
plt.axis('off')
plt.show()

# Calculate and display PSNR
noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
denoise_psnr = peak_signal_noise_ratio(ref_img, NLM_CV2_denoise_img_float)
print(f"PSNR of input noisy image = {noise_psnr:.2f}")
print(f"PSNR of cleaned image = {denoise_psnr:.2f}")

###########################################################################
###########################################################################

#BM3D Block-matching and 3D filtering
#pip install bm3d

import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.metrics import peak_signal_noise_ratio
import bm3d
import numpy as np

noisy_img = img_as_float(io.imread("/Users/yanagaur/Downloads/MRI_noisy.tif"))
ref_img = img_as_float(io.imread("/Users/yanagaur/Downloads/MRI_clean.tif"))


BM3D_denoised_image = bm3d.bm3d(noisy_img, sigma_psd=0.2, stage_arg=bm3d.BM3DStages.ALL_STAGES)
#BM3D_denoised_image = bm3d.bm3d(noisy_img, sigma_psd=0.2, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)

 #Also try stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING                     


noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
BM3D_cleaned_psnr = peak_signal_noise_ratio(ref_img, BM3D_denoised_image)
print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", BM3D_cleaned_psnr)


plt.imshow(BM3D_denoised_image, cmap='gray')
plt.imsave("images/MRI_images/BM3D_denoised.tif", BM3D_denoised_image, cmap='gray')

