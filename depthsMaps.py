import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


#left_image = cv.imread('tsukuba_l.png', cv.IMREAD_GRAYSCALE)
#right_image = cv.imread('tsukuba_r.png', cv.IMREAD_GRAYSCALE)

left_image = cv.imread('/Users/mostafazaky/Desktop/depth maps/IMG_2074.png', cv.IMREAD_GRAYSCALE)
right_image = cv.imread('/Users/mostafazaky/Desktop/depth maps/IMG_2075.png', cv.IMREAD_GRAYSCALE)

#left_image = cv.imread('items2_l.png', cv.IMREAD_GRAYSCALE)
#right_image = cv.imread('items2_r.png', cv.IMREAD_GRAYSCALE)         



stereo = cv.StereoBM_create(numDisparities=0, blockSize=15)
# For each pixel algorithm will find the best disparity from 0
# Larger block size implies smoother, though less accurate disparity map
depth = stereo.compute(left_image, right_image)

print(depth)

cv.imshow("Left", left_image)
cv.imshow("right", right_image)

plt.imshow(depth, cmap="plasma")
plt.axis('off')
plt.show()
