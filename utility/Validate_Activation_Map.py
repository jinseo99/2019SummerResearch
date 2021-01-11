"""
By: Jinseo Lee

Alter existing images for validating CAM, Grad-CAM, Grad-CAM++
"""

from matplotlib import pyplot as plt
import numpy as np

def overlayShape(img):
    altered = np.copy(img[100])
    altered[0:150,0:150] = 0
    return altered


# plt.imshow(altered[...,0], cmap = 'gray')
# pred = model.predict(np.expand_dims(altered, axis =0))