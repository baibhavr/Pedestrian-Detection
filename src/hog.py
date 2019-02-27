"""
Created on Tue Apr 09 11:58:40 2013

@author: baibhav
"""

import numpy as np
from scipy import sqrt, pi, arctan2, cos, sin
from scipy.ndimage import uniform_filter

def hog(image, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), visualise=False, normalise=True):
    
    image = np.atleast_2d(image)
  
    if image.ndim > 2:
        raise ValueError("Only supports greyscale images")
  
    if normalise:
        image = sqrt(image)
  
    if image.dtype.kind == 'u':
        image = image.astype('float')
  
    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    gx[:, :-1] = np.diff(image, n=1, axis=1)
    gy[:-1, :] = np.diff(image, n=1, axis=0)
  
    magnitude = sqrt(gx**2 + gy**2)
    orientation = arctan2(gy, gx) * (180 / pi) % 180
  
    sy, sx = image.shape
    cx, cy = pixels_per_cell
    bx, by = cells_per_block
  
    n_cellsx = int(np.floor(sx // cx))  # number of cells in x
    n_cellsy = int(np.floor(sy // cy))  # number of cells in y
  
    orientation_histogram = np.zeros((n_cellsy, n_cellsx, orientations))
    subsample = np.index_exp[cy / 2:cy * n_cellsy:cy, cx / 2:cx * n_cellsx:cx]
    for i in range(orientations):
        temp_ori = np.where(orientation < 180 / orientations * (i + 1),
                            orientation, -1)
        temp_ori = np.where(orientation >= 180 / orientations * i,
                            temp_ori, -1)
        cond2 = temp_ori > -1
        temp_mag = np.where(cond2, magnitude, 0)
  
        temp_filt = uniform_filter(temp_mag, size=(cy, cx))
        orientation_histogram[:, :, i] = temp_filt[subsample]
  
    # now for each cell, compute the histogram
    hog_image = None
  
    if visualise:
        from skimage import draw
  
        radius = min(cx, cy) // 2 - 1
        hog_image = np.zeros((sy, sx), dtype=float)
        for x in range(n_cellsx):
            for y in range(n_cellsy):
                for o in range(orientations):
                    centre = tuple([y * cy + cy // 2, x * cx + cx // 2])
                    dx = radius * cos(float(o) / orientations * np.pi)
                    dy = radius * sin(float(o) / orientations * np.pi)
                    rr, cc = draw.bresenham(centre[0] - dy, centre[1] - dx,
                                            centre[0] + dy, centre[1] + dx)
                    hog_image[rr, cc] += orientation_histogram[y, x, o]
  
    n_blocksx = (n_cellsx - bx) + 1
    n_blocksy = (n_cellsy - by) + 1
    normalised_blocks = np.zeros((n_blocksy, n_blocksx,
                                  by, bx, orientations))
  
    for x in range(n_blocksx):
        for y in range(n_blocksy):
            block = orientation_histogram[y:y + by, x:x + bx, :]
            eps = 1e-5
            normalised_blocks[y, x, :] = block / sqrt(block.sum()**2 + eps)
#     print normalised_blocks.shape
   
    if visualise:
        return normalised_blocks.ravel(), hog_image
    else:
        return normalised_blocks.ravel()
    
'''
##############
    Usage
##############    
'''

'''
from skimage import io, color
import matplotlib.pyplot as plt

im = io.imread("/home/baibhav/Desktop/21141_10152775396190072_1336237297_n_sm.jpg")
im = color.rgb2gray(im)
hog_v,hog_image = hog(im, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(3, 3), visualise=True, normalise=True)
print hog_v.shape
plt.imshow(hog_image);plt.set_cmap ('gray');plt.show() # Display HOG Image

io.imsave("/home/baibhav/Desktop/myhog.jpg", hog_image)
'''
