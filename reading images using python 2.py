#PIL
#Numpy

from PIL import Image
import numpy as np

img = Image.open("/Users/yanagaur/Downloads/Image_2.png")
print(type(img)) #not a numpy array
img.show() # show in a diff window
print(img.format)
img1 = np.asarray(img)#convert to numpy array
print(type(img1))

########################################################################

#matplotlib
#pyplot

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

img = mpimg.imread("/Users/yanagaur/Downloads/Image_2.png")
print(type(img))
print(img.shape)
plt.imshow(img) # show output in console
plt.colorbar()

########################################################################

#scikit image

from skimage import io, img_as_float, img_as_ubyte

image =io.imread("/Users/yanagaur/Downloads/Image_2.png")
print(type(image)) # numpy array
plt.imshow(image)
image_float = img_as_float(image)
print(type(image))
image1 =io.imread("/Users/yanagaur/Downloads/Image_2.png").astype(np.float)
print(image1)
image_byte= img_as_ubyte(image)

########################################################################

# import opencv

import cv2
import matplotlib.pyplot as plt
img = cv2.imread("/Users/yanagaur/Downloads/Image_2.png",0) #1 for color #0 for b/w
cv2.imshow("grey image",img)
cv2.waitKey(1)
cv2.destroyAllWindows()


########################################################################

#glob

import cv2
import glob

path = "/Users/yanagaur/Downloads/DetectIQ Letterhead Final 2.pdf/*"

for file in glob.glob(path):
    print(file)
    a= cv2.imread(file)
    print(a)
    cv2.imshow('image',a)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    


